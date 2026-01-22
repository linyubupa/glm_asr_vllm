# SPDX-License-Identifier: Apache-2.0
# Copyright contributors to the GLM-ASR vLLM integration

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import (Any, Iterable, List, Mapping, Optional, Sequence,
                    Tuple, Union)

from functools import cached_property

from transformers import AutoTokenizer, WhisperFeatureExtractor, BatchFeature, AutoConfig

from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.inputs.data import PromptType
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models import SupportsPP
from vllm.model_executor.models.interfaces import (
    SupportsLoRA,
    SupportsMultiModal,
    SupportsTranscription,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
from vllm.model_executor.models.utils import init_vllm_registered_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalDataDict, MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems, MultiModalDataParser
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    MultiModalFieldConfig,
    PromptReplacement,
    PromptUpdate,
)
from vllm.multimodal.profiling import BaseDummyInputsBuilder, ProcessorInputs
from vllm.sequence import IntermediateTensors

# 假设 glmasr_audio.py 包含 WhisperSpecialEncoder
from .glmasr_audio import WhisperSpecialEncoder

logger = init_logger(__name__)

ISO639_1_SUPPORTED_LANGS = {
    "en": "English",
    "zh": "Chinese",
}

# ----------------------------
# 1. Audio MLP Adapter
# ----------------------------
class AudioMLPAdapter(nn.Module):
    def __init__(self, cfg: Any, lm_hidden_size: int):
        super().__init__()
        self.merge_factor = getattr(cfg, "merge_factor", 4)
        whisper_cfg = cfg.whisper_config
        self.whisper = WhisperSpecialEncoder(
            whisper_cfg,
            use_rope=getattr(cfg, "use_rope", True),
        )
        self.layer_norm = nn.LayerNorm(whisper_cfg.d_model)
        
        self.lm_hidden = lm_hidden_size
        mlp_intermediate = 4096
        
        # [DEBUG LOG] 检查 Adapter 初始化维度
        # logger.info(f"DEBUG: Initializing AudioMLPAdapter with lm_hidden_size={self.lm_hidden}")
        
        self.adapting = nn.Sequential(
            nn.Linear(whisper_cfg.d_model * self.merge_factor, mlp_intermediate, bias=True),
            nn.GELU(),
            nn.Linear(mlp_intermediate, self.lm_hidden, bias=True),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio is None or audio.numel() == 0:
            # logger.info("DEBUG: Adapter forward received EMPTY audio tensor")
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype
            return torch.zeros((1, 1, self.lm_hidden), device=device, dtype=dtype)

        # --- ROBUST DIMENSION CHECK ---
        # 目标维度: (Batch, Mel_Bins, Time) -> 3D
        # 当前收到: [32, 1, 128, 500] -> 4D
        if audio.ndim == 4:
            if audio.shape[1] == 1:
                # 情况 A: [B, 1, D, T] -> 压缩掉多余的 1
                audio = audio.squeeze(1)
            elif audio.shape[-1] == 1:
                # 情况 B: [B, D, T, 1]
                audio = audio.squeeze(-1)
            else:
                # 情况 C: 如果是 [Batch, Num_Frames, D_Model] 这种已经被预处理过的
                # 则需要根据 whisper 编码器的预期来调整
                pass
        encoded = self.whisper(audio)[0] 
        encoded = self.layer_norm(encoded)

        b, t, d = encoded.shape
        t_merged = max(t // self.merge_factor, 1)
        t_trunc = t_merged * self.merge_factor

        if t < t_trunc:
            padding = torch.zeros((b, t_trunc - t, d), device=encoded.device, dtype=encoded.dtype)
            encoded = torch.cat([encoded, padding], dim=1)
        else:
            encoded = encoded[:, :t_trunc, :]

        encoded = encoded.reshape(b, t_merged, d * self.merge_factor)
        out = self.adapting(encoded)
        
        # [DEBUG LOG] 检查输出维度
        if out.shape[-1] == 0:
            logger.error(f"DEBUG: Adapter OUTPUT DIMENSION IS ZERO! Shape: {out.shape}")
        
        return out

# ----------------------------
# 2. 递归特征提取
# ----------------------------
def _extract_audio_features_list_from_out_mm_kwargs(out_mm_kwargs: MultiModalKwargsItems) -> List[torch.Tensor]:
    item = out_mm_kwargs.get("audio")
    def _recursive(x):
        if isinstance(x, torch.Tensor):
            return [x[i] for i in range(x.shape[0])] if x.ndim >= 3 else [x]
        if hasattr(x, "data"): return _recursive(getattr(x, "data"))
        if isinstance(x, (list, tuple)):
            res = []
            for e in x: res.extend(_recursive(e))
            return res
        if isinstance(x, Mapping):
            for k in ("audio", "input_features", "data"):
                if k in x: return _recursive(x[k])
        return []
    feats = _recursive(item)
    if not feats:
        logger.error(f"DEBUG: Failed to extract audio from item type {type(item)}")
        raise TypeError(f"Extract failed for {type(item)}")
    return [t.squeeze(0) if t.ndim == 3 and t.shape[0] == 1 else t for t in feats]



@dataclass(unsafe_hash=True) 
class GlmasrProcessorAdapter:
    model_path: str
    merge_factor: int
    audio_token: str = "<|audio|>"
    bos_audio_token: str = "<|begin_of_audio|>"
    eos_audio_token: str = "<|end_of_audio|>"

    def __post_init__(self):
        # 这里的 feature_extractor 通常返回的是 mel 频谱 (Whisper 风格)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

    def calculate_audio_token_count(self, mel_frames: int) -> int:
        # GLM-ASR 的下采样逻辑
        downsampled = (mel_frames + 1) // 2
        return max(downsampled // self.merge_factor, 1)

    def __call__(
        self,
        text: Union[str, List[str], None] = None,
        audios: Union[np.ndarray, List[np.ndarray], torch.Tensor, None] = None,
        return_tensors: Optional[str] = "pt",
        **kwargs
    ) -> BatchFeature:
        # 1. 规范化 text 输入
        if text is None:
            text = []
        if isinstance(text, str):
            text = [text]
            
        # 2. 规范化 audios 输入
        if audios is None:
            audios = []
        elif not isinstance(audios, (list, tuple)):
            # 如果是单张 Tensor 或 ndarray，包装成 list
            audios = [audios]

        # 3. 处理文本 (Tokenize)
        if text:
            # GLM-ASR 在 vLLM 中通常由外部注入 prompt 模板，这里只需简单分词
            # 注意：padding=True 确保多 batch 时对齐
            encodings = self.tokenizer(text, return_tensors="pt", add_special_tokens=False, padding=True)
            input_ids = encodings["input_ids"]
        else:
            # 如果完全没有文本（极少见），返回空 tensor
            input_ids = torch.tensor([[]], dtype=torch.long)

        # 4. 如果没有音频数据，直接返回文本部分 (修复 ValueError 的关键)
        if len(audios) == 0:
            return BatchFeature({"input_ids": input_ids})

        # 5. 处理音频特征提取 (Whisper 特征)
        # WhisperFeatureExtractor 接受 list of numpy/tensor
        audio_features = self.feature_extractor(
            audios, 
            sampling_rate=16000, 
            return_tensors="pt",
            padding="longest"
        )

        # 6. 返回 BatchFeature
        # 注意：这里的 Key "audio" 必须与你模型代码中的 embed_multimodal 保持一致
        return BatchFeature({
            "input_ids": input_ids,
            "audio": audio_features["input_features"]
        })

class GlmasrProcessingInfo(BaseProcessingInfo):
    # def get_tokenizer(self) -> AutoTokenizer:
    #     return AutoTokenizer.from_pretrained(self.ctx.model_config.model, use_fast=True)
    # def get_hf_processor(self, **kwargs) -> GlmasrProcessorAdapter:
    #     return GlmasrProcessorAdapter(self.ctx.model_config.model, getattr(self.ctx.model_config.hf_config, "merge_factor", 4))
    
    @cached_property
    def _tokenizer(self) -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(self.ctx.model_config.model, use_fast=True)

    @cached_property
    def _hf_processor(self) -> GlmasrProcessorAdapter:
        # 这里内部会初始化 feature_extractor + tokenizer（只做一次）
        return GlmasrProcessorAdapter(
            self.ctx.model_config.model,
            getattr(self.ctx.model_config.hf_config, "merge_factor", 4),
        )

    def get_tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def get_hf_processor(self, **kwargs) -> GlmasrProcessorAdapter:
        return self._hf_processor
    
    def get_supported_mm_limits(self) -> Mapping[str, int]:
        return {"audio": 1}
    def get_mm_max_tokens_per_item(self, seq_len, mm_counts) -> Mapping[str, int]:
        return {"audio": 512}

class GlmasrDummyInputsBuilder(BaseDummyInputsBuilder[GlmasrProcessingInfo]):
    def get_dummy_text(self, mm_counts) -> str:
        return "<|user|>\n<|begin_of_audio|><|audio|><|end_of_audio|>\n<|assistant|>\n"
    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None) -> MultiModalDataDict:
        num = mm_counts.get("audio", 0)
        return {"audio": self._get_dummy_audios(length=16000 * 5, num_audios=num)} if num > 0 else {}
    def get_dummy_processor_inputs(self, seq_len, mm_counts, mm_options=None) -> ProcessorInputs:
        dummy_text = self.get_dummy_text(mm_counts)
        dummy_mm_data = self.get_dummy_mm_data(seq_len, mm_counts, mm_options)
        proc = self.info.get_hf_processor()
        wavs = [a[0] if isinstance(a, tuple) else a for a in dummy_mm_data["audio"]]
        processed = proc(text=dummy_text, audios=wavs)
        return ProcessorInputs(prompt=processed["input_ids"][0].tolist(), mm_data=dummy_mm_data)

class GlmasrMultiModalProcessor(BaseMultiModalProcessor[GlmasrProcessingInfo]):
    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs) -> Mapping[str, MultiModalFieldConfig]:
        return {"audio": MultiModalFieldConfig.batched("audio")}
    
    # 在 glmasr.py 的 GlmasrMultiModalProcessor 类中修改
    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[PromptUpdate]:
        # --- 新增防御性代码 ---
        # 如果 mm_items 为空，说明本次请求的所有音频数据都已命中缓存，不需要计算 Prompt 更新
        if not mm_items:
            return []
        # ---------------------

        proc = self.info.get_hf_processor()
        audio_token_id = proc.tokenizer.convert_tokens_to_ids(proc.audio_token)
        
        # 因为上面做了判断，这里调用 extract 就不再会因为缓存命中而拿到 None
        audio_feats_list = _extract_audio_features_list_from_out_mm_kwargs(out_mm_kwargs)
    
        def get_replacement(item_idx):
            feat_shape = audio_feats_list[item_idx].shape
            n = proc.calculate_audio_token_count(int(feat_shape[-1]))
            return [int(audio_token_id)] * n
        
        return [PromptReplacement(modality="audio", target=[int(audio_token_id)], replacement=get_replacement)] 
    
    def _get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser(target_sr=16000)

# ----------------------------
# 3. 模型定义
# ----------------------------
@MULTIMODAL_REGISTRY.register_processor(GlmasrMultiModalProcessor, info=GlmasrProcessingInfo, dummy_inputs=GlmasrDummyInputsBuilder)
class GlmasrForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP, SupportsLoRA, SupportsTranscription):
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"]}
    supported_languages = ISO639_1_SUPPORTED_LANGS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        
        # [DEBUG LOG] 检查原始配置
        # logger.info(f"DEBUG: hf_config keys: {list(self.config.__dict__.keys()) if hasattr(self.config, '__dict__') else 'N/A'}")
        
        lm_cfg = self.config.lm_config
        if isinstance(lm_cfg, dict):
            # logger.info(f"DEBUG: lm_config is a dict, converting to AutoConfig. Keys: {list(lm_cfg.keys())}")
            if "model_type" not in lm_cfg: lm_cfg["model_type"] = "llama"
            lm_cfg = AutoConfig.for_model(**lm_cfg)

        self.language_model = init_vllm_registered_model(vllm_config=vllm_config, hf_config=lm_cfg, prefix=f"{prefix}language_model")
        
        # [DEBUG LOG] 检查语言模型初始化后的维度
        real_hidden = getattr(self.language_model.config, "hidden_size", -1)
        # logger.info(f"DEBUG: Initialized language_model. hidden_size={real_hidden}")
        
        if real_hidden <= 0:
            logger.warning("DEBUG: language_model.config.hidden_size is <= 0! Hardcoding to 2048.")
            real_hidden = 2048
            
        self.audio_encoder = AudioMLPAdapter(self.config, real_hidden)

    
    def get_language_model(self) -> nn.Module:
        """
        必须实现此方法，以便 vLLM V1 引擎能够访问语言模型的 Embedding 层。
        """
        return self.language_model
    def get_mm_mapping(self) -> MultiModelKeys:
        return MultiModelKeys.from_string_field(language_model="language_model", connector="audio_encoder.adapting", tower_model=["audio_encoder.whisper"])

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = default_weight_loader
        audio_params = dict(self.audio_encoder.named_parameters())
        def llm_weights_iterator():
            for name, loaded_weight in weights:
                if name.startswith("audio_encoder."):
                    inner = name[len("audio_encoder."):]
                    if inner in audio_params: loader(audio_params[inner], loaded_weight)
                elif name.startswith("model.") or name.startswith("lm_head."): yield name, loaded_weight
                elif name.startswith("language_model."): yield name[len("language_model."):], loaded_weight
        self.language_model.load_weights(llm_weights_iterator())

    def embed_multimodal(self, audio: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        # [DEBUG LOG] 检查多模态输入
        # logger.info(f"DEBUG: embed_multimodal received audio shape: {audio.shape}")
        
        param = next(self.audio_encoder.parameters())
        if isinstance(audio, list):
            # 1. 检查是否为空
            if not audio:
                # 极其罕见的情况，返回空列表或根据逻辑处理
                return []
            
            # 2. 检查维度一致性并进行 Padding
            # 假设 audio[i] 的形状通常是 [Channels, Mel, Time] 或 [Mel, Time]
            # 我们关注最后一个维度 (Time)
            max_len = 0
            shapes = []
            for a in audio:
                shapes.append(a.shape)
                if a.shape[-1] > max_len:
                    max_len = a.shape[-1]
            
            # 如果存在长度不一致的情况，或者需要强制对齐
            if any(s[-1] != max_len for s in shapes):
                padded_audio = []
                for a in audio:
                    diff = max_len - a.shape[-1]
                    if diff > 0:
                        # F.pad 的参数格式是 (pad_left, pad_right, pad_top, pad_bottom, ...)
                        # 我们只需要在最后一个维度的右侧补零
                        padded = torch.nn.functional.pad(a, (0, diff), value=0.0)
                        padded_audio.append(padded)
                    else:
                        padded_audio.append(a)
                audio = torch.stack(padded_audio)
            else:
                # 长度完全一致，直接 stack
                audio = torch.stack(audio)
        out = self.audio_encoder(audio.to(device=param.device, dtype=param.dtype))
        
        # [DEBUG LOG] 检查映射后的维度
        # logger.info(f"DEBUG: embed_multimodal output shape: {out.shape}")
        return list(out.unbind(0))

    def forward(
        self, 
        input_ids: torch.Tensor, 
        positions: torch.Tensor, 
        intermediate_tensors: Optional[IntermediateTensors] = None, 
        inputs_embeds: Optional[torch.Tensor] = None, 
        **kwargs: Any
    ) -> Union[torch.Tensor, IntermediateTensors]:
        
        # [DEBUG LOG] 检查 Forward 输入维度
        if inputs_embeds is not None:
            # logger.info(f"DEBUG: Forward inputs_embeds shape: {inputs_embeds.shape}")
            if inputs_embeds.shape[-1] == 0:
                logger.error("DEBUG: inputs_embeds HAS ZERO WIDTH!")
            input_ids = None # V1 要求互斥
        else:
            if input_ids is not None:
                logger.info(f"DEBUG: Forward input_ids shape: {input_ids.shape}")
            else:
                # Profile 极少数兜底
                logger.info("DEBUG: Both input_ids and inputs_embeds are NONE. Creating dummy.")
                input_ids = torch.zeros((1, 1), dtype=torch.long, device=positions.device)

        return self.language_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Optional[Any] = None):
        return self.language_model.compute_logits(hidden_states)

    # def compute_logprobs(self, logits: torch.Tensor, sampling_metadata: Any):
    #     # 同样建议给这里的 metadata 增加兼容性处理（如果 V1 报错的话）
    #     return self.language_model.compute_logprobs(logits, sampling_metadata)

    @classmethod
    def get_speech_to_text_config(cls, model_config: ModelConfig,task_type: str = "") -> SpeechToTextConfig:
        return SpeechToTextConfig(sample_rate=16000, max_audio_clip_s=30.0)

    @classmethod
    def get_generation_prompt(cls, audio, model_config, stt_config, **kwargs) -> PromptType:
        return {"prompt": "<|user|>\n<|begin_of_audio|><|audio|><|end_of_audio|>\n<|assistant|>\n", "multi_modal_data": {"audio": (audio, 16000)}}