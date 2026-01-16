import os
from typing import Any, Dict, Optional, Union

from transformers import LlamaConfig, PretrainedConfig, WhisperConfig

class GlmasrConfig(PretrainedConfig):
    model_type = "glmasr"
    is_composition = True

    def __init__(
        self,
        lm_config: Optional[Union[Dict[str, Any], LlamaConfig]] = None,
        whisper_config: Optional[Union[Dict[str, Any], WhisperConfig]] = None,
        adapter_type: str = "mlp",
        merge_factor: int = 4,
        spec_aug: bool = False,
        use_rope: bool = True,
        max_whisper_length: int = 1500,
        mlp_adapter_act: str = "gelu",
        **kwargs,
    ):
        # 1. 处理 lm_config
        if isinstance(lm_config, dict):
            lm_config["model_type"] = lm_config.get("model_type", "llama")
            self.lm_config = LlamaConfig(**lm_config)
        elif isinstance(lm_config, LlamaConfig):
            self.lm_config = lm_config
        else:
            self.lm_config = LlamaConfig()

        # 2. 处理 whisper_config
        if isinstance(whisper_config, dict):
            whisper_config["model_type"] = whisper_config.get("model_type", "whisper")
            self.whisper_config = WhisperConfig(**whisper_config)
        elif isinstance(whisper_config, WhisperConfig):
            self.whisper_config = whisper_config
        else:
            self.whisper_config = WhisperConfig()

        # 3. 设置自定义参数
        self.adapter_type = adapter_type
        self.merge_factor = merge_factor
        self.spec_aug = spec_aug
        self.use_rope = use_rope
        self.max_whisper_length = max_whisper_length
        self.mlp_adapter_act = mlp_adapter_act

        # 4. 调用父类初始化
        # 将 lm_config 中的关键参数透传给父类，以便 AutoModel 识别 (如 hidden_size)
        super().__init__(**kwargs)

    def to_dict(self):
        """
        重写 to_dict 以确保序列化时包含子配置的完整字典
        """
        output = super().to_dict()
        output["lm_config"] = self.lm_config.to_dict()
        output["whisper_config"] = self.whisper_config.to_dict()
        return output

__all__ = ["GlmasrConfig"]