# coding=utf-8
from __future__ import annotations

import base64
import io
import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import librosa

from transformers import AutoTokenizer, WhisperFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin
from transformers.utils import logging

logger = logging.get_logger(__name__)

_AUDIO_MARKER = "<<AUDIO_TOKENS>>"


def _normalize_dtype_name(name: str) -> str:
    name = name.strip().lower()
    alias = {
        "fp16": "float16",
        "float16": "float16",
        "half": "float16",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "float32": "float32",
        "float": "float32",
    }
    return alias.get(name, name)


def _resolve_torch_dtype(x: Any, default: str = "float32") -> torch.dtype:
    if isinstance(x, torch.dtype):
        return x
    if x is None:
        x = default
    if isinstance(x, str):
        name = _normalize_dtype_name(x)
        if not hasattr(torch, name):
            raise ValueError(f"Unknown torch dtype string: {x} (normalized: {name})")
        return getattr(torch, name)
    raise TypeError(f"audio_dtype/audio_torch_dtype must be str or torch.dtype or None, got {type(x)}")


class GlmasrProcessor(ProcessorMixin):
    """
    ✅ 特性：
    - 支持 audio part: path / array / base64
    - batch：先把原始音频列表送进 feature_extractor，并在 feature_extractor 内 padding 统一长度
    - 用 pad 后的 mel T 计算 num_tokens -> prompt 中 audio_token 数 batch 内一致
    - tokenize=False：返回 prompt（单条 str / batch List[str]）
    - tokenize=True：返回 BatchFeature，字段仅包含模型能吃的 input_ids/attention_mask/audios
    """

    attributes = ["feature_extractor", "tokenizer"]
    valid_kwargs = ["merge_factor", "audio_token", "audio_dtype"]

    feature_extractor_class = ("WhisperFeatureExtractor", "SequenceFeatureExtractor")
    tokenizer_class = ("PreTrainedTokenizerFast", "PreTrainedTokenizer", "LlamaTokenizerFast")

    def __init__(
        self,
        feature_extractor,
        tokenizer,
        merge_factor: int = 4,
        audio_token: str = "<|audio|>",
        audio_dtype: str = "float32",
        **kwargs,
    ):
        super().__init__(feature_extractor, tokenizer)
        self.merge_factor = int(merge_factor)
        self.audio_token = str(audio_token)
        self.audio_dtype = str(audio_dtype)

        self.bos_audio_token = "<|begin_of_audio|>"
        self.eos_audio_token = "<|end_of_audio|>"
        self.user_token = "<|user|>"
        self.assistant_token = "<|assistant|>"

    # =========================
    # from_pretrained
    # =========================
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "GlmasrProcessor":
        trust_remote_code = bool(kwargs.pop("trust_remote_code", False))

        passthrough_keys = {
            "cache_dir",
            "force_download",
            "local_files_only",
            "token",
            "revision",
            "subfolder",
        }
        shared_kwargs = {k: kwargs[k] for k in list(kwargs.keys()) if k in passthrough_keys}

        merge_factor = 4
        audio_token = "<|audio|>"
        audio_dtype = "float32"
        tokenizer_cfg: Dict[str, Any] = {}
        feat_cfg: Dict[str, Any] = {}

        proc_cfg_path = os.path.join(pretrained_model_name_or_path, "processor_config.json")
        if os.path.isfile(proc_cfg_path):
            with open(proc_cfg_path, "r", encoding="utf-8") as f:
                proc_cfg = json.load(f)
            merge_factor = int(proc_cfg.get("merge_factor", merge_factor))
            audio_token = str(proc_cfg.get("audio_token", audio_token))
            audio_dtype = str(proc_cfg.get("audio_dtype", audio_dtype))
            tokenizer_cfg = proc_cfg.get("tokenizer_config", {}) or {}
            feat_cfg = proc_cfg.get("feature_extractor_config", {}) or {}

        feature_extractor = WhisperFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
            **shared_kwargs,
        )
        for k, v in feat_cfg.items():
            if hasattr(feature_extractor, k):
                try:
                    setattr(feature_extractor, k, v)
                except Exception:
                    pass

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            **shared_kwargs,
        )
        for k, v in tokenizer_cfg.items():
            if hasattr(tokenizer, k):
                try:
                    setattr(tokenizer, k, v)
                except Exception:
                    pass

        return cls(
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            merge_factor=merge_factor,
            audio_token=audio_token,
            audio_dtype=audio_dtype,
        )

    # =========================
    # audio helpers
    # =========================
    def _load_audio_file(self, path: str, sampling_rate: int = 16000) -> np.ndarray:
        audio_array, _ = librosa.load(path, sr=int(sampling_rate), mono=True)
        return np.asarray(audio_array, dtype=np.float32)

    def _strip_data_url_prefix(self, b64: str) -> str:
        # 支持 "data:audio/wav;base64,AAAA..."
        if "," in b64 and b64[:30].lower().startswith("data:"):
            return b64.split(",", 1)[1]
        return b64

    def _load_audio_base64(self, b64: str, sampling_rate: int = 16000) -> np.ndarray:
        """
        base64 -> bytes -> waveform float32 @ sampling_rate
        兼容 wav/flac/ogg 等；若 soundfile 不支持的格式，会 fallback 用 librosa/audioread（依赖系统解码器）。
        """
        b64 = self._strip_data_url_prefix(b64)
        raw = base64.b64decode(b64)

        # 1) 优先用 soundfile（更快更稳，支持 wav/flac/ogg）
        try:
            import soundfile as sf  # librosa 通常会带这个依赖

            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            wav = data.mean(axis=1)  # (T,)
            if int(sr) != int(sampling_rate):
                wav = librosa.resample(wav, orig_sr=int(sr), target_sr=int(sampling_rate))
            return np.asarray(wav, dtype=np.float32)
        except Exception:
            pass

        # 2) fallback：写到内存临时文件交给 librosa（可能需要 ffmpeg/gstreamer 等解码器）
        try:
            # librosa.load 主要吃路径；这里用 BytesIO 尝试（有些后端可行）
            bio = io.BytesIO(raw)
            wav, _sr = librosa.load(bio, sr=int(sampling_rate), mono=True)
            return np.asarray(wav, dtype=np.float32)
        except Exception as e:
            raise ValueError(
                "Failed to decode base64 audio. Ensure the base64 is an encoded audio file "
                "(e.g., WAV/FLAC/OGG)."
            ) from e

    def calculate_audio_token_count(self, mel_frames: int) -> int:
        # 保持你原逻辑：Whisper 下采样 + merge_factor
        downsampled = (int(mel_frames) + 1) // 2
        merged = downsampled // max(self.merge_factor, 1)
        return max(int(merged), 1)

    def _build_templates_and_audios(
        self,
        conversations: List[List[dict]],
        sampling_rate: int,
        add_generation_prompt: bool,
    ) -> tuple[List[str], List[np.ndarray]]:
        prompts_template: List[str] = []
        audios_raw: List[np.ndarray] = []

        for conv in conversations:
            conv_str = ""
            audio_found = False
            audio_raw_this: Optional[np.ndarray] = None

            for msg in conv:
                role = msg["role"]
                content = msg["content"]

                if role == "user":
                    conv_str += f"{self.user_token}\n"
                elif role == "assistant":
                    conv_str += f"{self.assistant_token}\n"
                else:
                    conv_str += f"<|{role}|>\n"

                if isinstance(content, str):
                    conv_str += f"\n{content}"
                elif isinstance(content, list):
                    for part in content:
                        ptype = part.get("type")
                        if ptype == "audio":
                            if audio_found:
                                raise ValueError("Only ONE audio per sample is supported.")
                            audio_found = True

                            if "array" in part:
                                arr = part["array"]
                                if isinstance(arr, torch.Tensor):
                                    arr = arr.detach().cpu().numpy()
                                audio_raw_this = np.asarray(arr, dtype=np.float32).reshape(-1)

                            elif "path" in part:
                                audio_raw_this = self._load_audio_file(
                                    part["path"], sampling_rate=sampling_rate
                                )

                            elif "base64" in part:
                                audio_raw_this = self._load_audio_base64(
                                    part["base64"], sampling_rate=sampling_rate
                                )

                            else:
                                raise ValueError("Audio part must contain 'path' or 'array' or 'base64'.")

                            # 先放 marker，等 feature_extractor pad 后再替换成 <|audio|>*N
                            conv_str += f"{self.bos_audio_token}{_AUDIO_MARKER}{self.eos_audio_token}"

                        elif ptype == "text":
                            conv_str += f"\n{part.get('text', '')}"
                        else:
                            raise ValueError(f"Unknown content part type: {ptype}")
                else:
                    raise ValueError(f"Unsupported message content type: {type(content)}")

            if add_generation_prompt:
                conv_str += f"{self.assistant_token}\n"

            if not audio_found or audio_raw_this is None:
                raise ValueError("This processor expects audio in each sample, but got none.")

            prompts_template.append(conv_str)
            audios_raw.append(audio_raw_this)

        return prompts_template, audios_raw

    # =========================
    # apply_chat_template
    # =========================
    def apply_chat_template(
        self,
        conversation: Union[List[dict], List[List[dict]]],
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> Union[BatchFeature, str, List[str]]:
        if chat_template is not None:
            logger.warning("chat_template argument is ignored (custom template is built in apply_chat_template).")

        tokenize = kwargs.pop("tokenize", True)
        return_tensors = kwargs.pop("return_tensors", "pt")
        kwargs.pop("return_dict", None)

        # dtype override for returned "audios"
        audio_torch_dtype = kwargs.pop("audio_torch_dtype", None)
        audio_dtype_override = kwargs.pop("audio_dtype", None)
        dtype_source = audio_torch_dtype if audio_torch_dtype is not None else audio_dtype_override
        target_dtype = _resolve_torch_dtype(dtype_source, default=getattr(self, "audio_dtype", "float32"))

        # tokenizer args
        text_kwargs = dict(kwargs.pop("text_kwargs", {}) or {})
        for k in ("padding", "truncation", "max_length", "add_special_tokens"):
            if k in kwargs and k not in text_kwargs:
                text_kwargs[k] = kwargs.pop(k)

        # audio args: feature_extractor 内 padding（你要求的）
        sampling_rate = int(kwargs.pop("sampling_rate", 16000))
        audio_padding = kwargs.pop("audio_padding", "longest")  # "longest" / "max_length" / False
        audio_max_length = kwargs.pop("audio_max_length", None)  # samples, only when padding="max_length"
        audio_pad_to_multiple_of = kwargs.pop("audio_pad_to_multiple_of", None)

        if kwargs:
            logger.warning(f"Ignored unused kwargs in apply_chat_template: {list(kwargs.keys())}")

        # normalize batch
        if isinstance(conversation, list) and conversation and isinstance(conversation[0], dict):
            conversations = [conversation]
            is_single = True
        else:
            conversations = conversation  # type: ignore
            is_single = False

        # 1) build prompt templates + raw audios
        prompt_templates, audios_raw = self._build_templates_and_audios(
            conversations=conversations,
            sampling_rate=sampling_rate,
            add_generation_prompt=add_generation_prompt,
        )

        # 2) feature_extractor batch pad -> (B, F, T)
        feat = self.feature_extractor(
            audios_raw,
            sampling_rate=sampling_rate,
            return_tensors="np",
            return_attention_mask=False,
            padding=audio_padding,
            max_length=audio_max_length,
            pad_to_multiple_of=audio_pad_to_multiple_of,
        )
        input_features = feat["input_features"]  # (B, F, T)
        if not isinstance(input_features, np.ndarray):
            input_features = np.asarray(input_features)

        _, _, T = input_features.shape
        num_tokens = self.calculate_audio_token_count(int(T))

        # 3) replace marker with fixed-length audio tokens (batch-consistent)
        audio_tokens_str = "".join([self.audio_token] * num_tokens)
        prompts = [p.replace(_AUDIO_MARKER, audio_tokens_str) for p in prompt_templates]

        # 4) tokenize=False -> prompts only
        if not tokenize:
            return prompts[0] if is_single else prompts

        # 5) tokenize text + attach audios
        text_kwargs.setdefault("padding", "longest")
        text_kwargs.setdefault("add_special_tokens", False)
        text_kwargs["return_tensors"] = return_tensors

        enc = self.tokenizer(prompts, **text_kwargs)
        data: Dict[str, Any] = dict(enc)

        data["audios"] = torch.tensor(input_features, dtype=target_dtype)
        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def __call__(
        self,
        text: Union[str, List[str]],
        audios: Union[np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]],
        sampling_rate: int = 16000,
        return_tensors: str = "pt",
        **tokenizer_kwargs,
    ) -> BatchFeature:
        """
        vLLM-friendly:
        - 不做 prompt update（不扩展 <|audio|>）
        - 仅：text -> tokenizer；audios(raw waveform) -> feature_extractor -> input_features
        - 输出字段固定：input_ids / attention_mask / audios
        """
        # normalize audios -> List[np.ndarray]
        if isinstance(audios, (np.ndarray, torch.Tensor)):
            audios_list = [audios]
        else:
            audios_list = list(audios)

        audios_np: List[np.ndarray] = []
        for a in audios_list:
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
            a = np.asarray(a, dtype=np.float32).reshape(-1)
            audios_np.append(a)

        # audio -> mel (B, F, T)
        feat = self.feature_extractor(
            audios_np,
            sampling_rate=int(sampling_rate),
            return_tensors="np",
            return_attention_mask=False,
            padding="longest",
        )
        input_features = feat["input_features"]  # (B, F, T)
        if not isinstance(input_features, np.ndarray):
            input_features = np.asarray(input_features)

        # text -> ids
        tokenizer_kwargs = dict(tokenizer_kwargs or {})
        tokenizer_kwargs.setdefault("padding", "longest")
        tokenizer_kwargs.setdefault("add_special_tokens", False)
        tokenizer_kwargs["return_tensors"] = return_tensors

        enc = self.tokenizer(text, **tokenizer_kwargs)
        data: Dict[str, Any] = dict(enc)

        # attach audios
        data["audios"] = torch.tensor(
            input_features,
            dtype=_resolve_torch_dtype(getattr(self, "audio_dtype", "float32")),
        )
        return BatchFeature(data=data, tensor_type=return_tensors)
    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "audios"]


__all__ = ["GlmasrProcessor"]