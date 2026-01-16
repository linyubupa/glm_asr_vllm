from typing import Any, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention
from transformers import WhisperConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.whisper.modeling_whisper import WhisperEncoder, WhisperEncoderLayer
from transformers.utils import logging

logger = logging.get_logger(__name__)

# ==========================================
# 1. Rotary Embedding 核心组件
# ==========================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1):
        super().__init__()
        self.dim = dim
        self.rope_ratio = rope_ratio

    @torch.no_grad()
    def get_emb(self, seq_len: int, dtype: torch.dtype, device: torch.device, base: int = 10000):
        """生成 RoPE 缓存"""
        base = base * self.rope_ratio
        # 计算频率 theta
        inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim))
        
        # 生成位置索引
        t = torch.arange(seq_len, device=device, dtype=torch.float)
        freqs = torch.outer(t, inv_freq) # [seq_len, dim/2]
        
        # 构造 cos 和 sin 缓存
        # 形状: [seq_len, dim/2, 2]
        emb = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        
        if dtype in (torch.float16, torch.bfloat16):
            emb = emb.to(dtype)
        return emb

def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    x: [batch, num_heads, seq_len, head_dim]
    rope_cache: [1, seq_len, dim/2, 2]
    """
    b, nh, sq, hd = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    
    # 将 x 分为旋转部分和不旋转部分
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    
    # 调整 x_rot 形状以匹配 rope_cache: [b, nh, sq, rot_dim/2, 2]
    x_shaped = x_rot.reshape(b, nh, sq, rot_dim // 2, 2)
    
    # 计算旋转: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    cos = rope_cache[..., 0] # [1, sq, rot_dim/2]
    sin = rope_cache[..., 1] # [1, sq, rot_dim/2]
    
    # 增加 head 维度
    cos = cos.unsqueeze(1) # [1, 1, sq, rot_dim/2]
    sin = sin.unsqueeze(1) # [1, 1, sq, rot_dim/2]
    
    x_out = torch.stack([
        x_shaped[..., 0] * cos - x_shaped[..., 1] * sin,
        x_shaped[..., 1] * cos + x_shaped[..., 0] * sin
    ], dim=-1)
    
    x_out = x_out.flatten(3) # 合并最后两维到 rot_dim
    return torch.cat([x_out, x_pass], dim=-1)

# ==========================================
# 2. 基于 SDPA 的 RoPE Attention
# ==========================================

class WhisperRoPESdpaAttention(nn.Module):
    """
    使用 PyTorch 原生 scaled_dot_product_attention 替代 WhisperFlashAttention2。
    """
    def __init__(self, config: WhisperConfig, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        
        # Whisper 标准投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rotary_pos_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        
        bsz, q_len, _ = hidden_states.size()

        # 1. 投影映射
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 2. 变形为 [batch, heads, seq, dim] 并确保内存连续
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        key_states = key_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        value_states = value_states.view(bsz, -1, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # 3. 应用 RoPE
        if rotary_pos_emb is not None:
            query_states = apply_rotary_pos_emb(query_states, rotary_pos_emb)
            key_states = apply_rotary_pos_emb(key_states, rotary_pos_emb)

        # 4. 数据类型对齐 (处理 fp32 LayerNorm 带来的类型不匹配)
        target_dtype = self.q_proj.weight.dtype
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

        # 5. SDPA 计算 (关键：不要手动乘以 scaling, SDPA 内部会自动处理)
        # 注意: 如果传入了 4D attention_mask，SDPA 会正确应用它
        attn_output = scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.is_causal,
        )

        # 6. 恢复形状并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output, None, None

# ==========================================
# 3. 封装好的 Encoder 层和 Encoder
# ==========================================

class WhisperSpecialEncoderLayer(WhisperEncoderLayer):
    def __init__(self, config: WhisperConfig):
        super().__init__(config)
        # 替换 Self-Attention 为我们的 RoPE SDPA 版本
        self.self_attn = WhisperRoPESdpaAttention(
            config=config,
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Any]:
        
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            rotary_pos_emb=rotary_pos_emb,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return (hidden_states, None) # 保持与 Whisper 接口一致的 tuple 长度

class WhisperSpecialEncoder(WhisperEncoder):
    def __init__(self, config: WhisperConfig, use_rope=True, rope_ratio=1):
        super().__init__(config)
        self.use_rope = use_rope
        # 覆盖父类的层列表
        self.layers = nn.ModuleList(
            [WhisperSpecialEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        
        if use_rope:
            # 计算 RoPE 维度: 通常是 head_dim 的一部分
            head_dim = config.d_model // config.encoder_attention_heads
            self.rotary_embedding = RotaryEmbedding(head_dim // 2, rope_ratio)

    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        position_ids=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Whisper 卷积特征提取
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1) # [B, T_down, D]

        if self.use_rope:
            # 生成旋转编码缓存
            rotary_embs = self.rotary_embedding.get_emb(
                seq_len=inputs_embeds.shape[1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device
            )
            # 形状调整为 [1, seq_len, dim/2, 2] 以便广播
            rotary_embs = rotary_embs.unsqueeze(0)
            hidden_states = inputs_embeds 
        else:
            rotary_embs = None
            # 回退到绝对位置编码
            embed_pos = self.embed_positions.weight[:inputs_embeds.shape[1]]
            hidden_states = inputs_embeds + embed_pos

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    None, # attention_mask
                    (head_mask[idx] if head_mask is not None else None),
                    output_attentions,
                    rotary_embs,
                    position_ids,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask=None,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    output_attentions=output_attentions,
                    rotary_pos_emb=rotary_embs,
                    position_ids=position_ids,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)

        hidden_states = self.layer_norm(hidden_states)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )