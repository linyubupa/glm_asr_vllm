from typing import Optional, List, Tuple, Union
import torch
from torch import Tensor, nn
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from .configuration_glmasr import GlmasrConfig
from .modeling_audio import WhisperSpecialEncoder

class AudioMLPAdapter(nn.Module):
    def __init__(self, config: GlmasrConfig):
        super().__init__()
        whisper_config = config.whisper_config
        self.merge_factor = config.merge_factor
        
        # 音频编码器
        self.whisper = WhisperSpecialEncoder(
            whisper_config,
            use_rope=config.use_rope,
        )
        # 禁用 Whisper 自带的 LayerNorm，使用我们自己的
        self.whisper.layer_norm = nn.Identity()
        self.layer_norm = nn.LayerNorm(whisper_config.hidden_size)
        
        # 激活函数选择
        act_fn_map = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "selu": nn.SELU(),
        }
        act = act_fn_map.get(config.mlp_adapter_act, nn.GELU())
        
        # 投影层：将 Whisper 维度映射到 LLM 维度
        # 输入维度 = Whisper Hidden * Merge Factor
        input_dim = whisper_config.hidden_size * self.merge_factor
        output_dim = config.lm_config.hidden_size
        
        self.adapting = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            act,
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(self, audios: Tensor) -> Tensor:
        """
        Args:
            audios: (Batch, raw_audio_len) 或者 Mel features
        Returns:
            adapted_features: (Batch, Seq_Len_Merged, LLM_Hidden_Dim)
        """
        bsz = audios.size(0)
        
        # 1. Whisper 编码
        encoded = self.whisper(audios)[0] # (B, T, D)
        encoded = self.layer_norm(encoded)
        
        # 2. 时序压缩 (Merge Factor)
        # 截断多余的帧以确保能被 factor 整除
        seq_len = encoded.size(1)
        if seq_len % self.merge_factor != 0:
            target_len = (seq_len // self.merge_factor) * self.merge_factor
            encoded = encoded[:, :target_len, :]
            
        # Reshape: (B, T, D) -> (B, T/k, D*k)
        encoded = encoded.reshape(bsz, -1, encoded.size(-1) * self.merge_factor)
        
        # 3. MLP 投影
        adapted = self.adapting(encoded)
        
        return adapted


class GlmasrForConditionalGeneration(LlamaForCausalLM):
    config_class = GlmasrConfig
    _no_split_modules = ["WhisperSpecialEncoder", "LlamaDecoderLayer"]

    def __init__(self, config: GlmasrConfig):
        # 初始化 Llama 父类
        super().__init__(config.lm_config)
        self.audio_encoder = AudioMLPAdapter(config)
        
        # 验证并保存 Audio Token ID
        self.audio_token_id = getattr(config.lm_config, "audio_token_id", None)
        if self.audio_token_id is None:
            raise ValueError("`audio_token_id` must be defined in lm_config.")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        audios: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # 1. 获取基础文本 Embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # 2. 音频特征注入逻辑
        # 仅当提供了音频数据且 input_ids 中包含音频占位符时执行
        if audios is not None and input_ids is not None:
            num_audio_tokens = (input_ids == self.audio_token_id).sum().item()
            
            if num_audio_tokens > 0:
                # 编码音频
                audio_features = self.audio_encoder(audios) # (B, Seq_Audio, Dim)
                
                # 展平音频特征: (Total_Audio_Tokens, Dim)
                audio_features_flat = audio_features.reshape(-1, audio_features.shape[-1])
                
                # 验证数量匹配 (可选，但在调试时很有用)
                if audio_features_flat.shape[0] != num_audio_tokens:
                    # 如果不匹配，通常是因为 merge_factor 或输入的音频长度与预留的 token 数量不一致
                    # 这里可以选择抛出错误，或者在推理时为了容错进行截断/填充，目前选择严格模式
                    pass 

                # 创建 Mask 并进行 Scatter 替换
                audio_mask = (input_ids == self.audio_token_id) # (B, Seq_Text)
                
                # 扩展 Mask 维度以匹配 Embeddings: (B, Seq_Text, 1)
                inputs_embeds = inputs_embeds.masked_scatter(
                    audio_mask.unsqueeze(-1),
                    audio_features_flat.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
                )

        # 3. 调用 Llama 模型核心
        outputs = self.model(
            input_ids=None, # 已经有了 inputs_embeds
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        if not return_dict:
            return (logits,) + outputs[1:]

        return CausalLMOutputWithPast(
            loss=None, # 如果需要计算 Loss，可以在此处添加逻辑
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
    
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        # 处理缓存：如果使用了 KV Cache，只输入最后一个 Token
        # if past_key_values:
        #     input_ids = input_ids[:, -1:]
        def _cache_seq_len(past_key_values):
            if past_key_values is None:
                return 0
            # 新式 Cache（DynamicCache/StaticCache 等）
            if hasattr(past_key_values, "get_seq_length"):
                return int(past_key_values.get_seq_length())
            # 旧式 tuple(list of tuples)
            # 形如 past_key_values[layer][0] = key: [B, heads, kv_len, head_dim]
            try:
                return int(past_key_values[0][0].shape[-2])
            except Exception:
                return 0
        past_len = _cache_seq_len(past_key_values)
        if past_len > 0:
            input_ids = input_ids[:, -1:]
        # 构造模型输入字典
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "audios": kwargs.get("audios", None), # 确保 audios 透传给 forward
        }
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs["inputs_embeds"] = inputs_embeds
            del model_inputs["input_ids"]

        return model_inputs

__all__ = ["GlmasrModel"]