# GLM-ASR with vLLM (中文文档)

一个集成 GLM-ASR 模型与 vLLM 的高性能语音识别（ASR）项目。本项目提供本地推理功能以及基于 Docker 的可扩展 API 服务器。

## 特性

- **音频转写**：使用 GLM-ASR 模型对音频文件进行语音识别
- **音频描述**：生成音频内容的文本描述
- **OpenAI 兼容 API**：vLLM 服务器提供 OpenAI 兼容的 API 接口
- **Docker 支持**：支持使用 Docker 和 Docker Compose 轻松部署
- **高性能**：利用 vLLM 实现高效的 GPU 加速推理
- **灵活的音频输入**：支持多种音频格式和输入方式

## 项目结构

```
glm_asr_vllm/
├── model/                  # 模型配置和实现
│   ├── configuration_glmasr.py    # GLM-ASR 配置文件
│   ├── modeling_glmasr.py         # GLM-ASR 模型实现
│   ├── modeling_audio.py          # 音频编码/解码
│   └── processing_glmasr.py       # 音频处理工具
├── server/                # vLLM 集成文件
│   ├── glmasr_audio.py     # vLLM 的音频处理
│   ├── glm_asr.py          # GLM-ASR vLLM 模型封装
│   ├── registry.py         # 模型注册表（vLLM）
│   └── server_ws.py        # WebSocket 服务器
├── wavs/                  # 示例音频文件
├── docker-compose.yaml     # Docker Compose 配置
├── dockerfile             # Docker 镜像构建配置
├── hf_demo.py             # HuggingFace Transformers 演示
└── test_vllm_api.py       # OpenAI API 客户端测试脚本
```

## 环境要求

- Python 3.12+
- 支持 CUDA 的 GPU（推荐）
- Docker（用于容器化部署）
- Docker Compose（可选）

## 安装

### 方式一：本地安装

1. 克隆仓库：
```bash
git clone <repository-url>
cd glm_asr_vllm
```

2. 安装依赖：
```bash
pip install torch transformers soundfile librosa openai
```

3. 从 HuggingFace 下载模型并放置到 `./model/` 目录：

```bash
# 使用 huggingface-cli 下载（推荐）
huggingface-cli download bupalinyu/glm-asr-eligant --local-dir ./model

# 或者使用 git lfs
git lfs install
git clone https://huggingface.co/bupalinyu/glm-asr-eligant ./model
```

**注意**：HuggingFace 上的模型 ID 为 `bupalinyu/glm-asr-eligant`。下载后，请确保所有模型文件都在 `./model/` 目录中。

### 方式二：Docker 部署

1. 构建 Docker 镜像：
```bash
docker build -t vllm-glmasr:latest .
```

2. 使用 Docker Compose 部署：
```bash
docker-compose up -d
```

## 使用方法

### HuggingFace Transformers 演示

运行本地演示脚本对音频文件进行转写：

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "./model/",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to("cuda")

processor = AutoProcessor.from_pretrained("./model/", trust_remote_code=True)

# 定义对话
conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": "./wavs/dufu.wav"},
                {"type": "text", "text": "Please transcribe this audio."},
            ],
        }
    ],
]

# 处理并生成
inputs = processor.apply_chat_template(
    conversations,
    return_tensors="pt",
    sampling_rate=16000,
    audio_padding="longest",
).to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

print(processor.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
```

运行演示：
```bash
python hf_demo.py
```

### vLLM API 服务器

#### 启动服务器

使用 Docker Compose：
```bash
docker-compose up -d
```

或手动使用 Docker：
```bash
docker run -d \
  --name vllm-glmasr \
  --gpus all \
  --ipc host \
  --shm-size 8gb \
  -p 8300:8300 \
  -e CUDA_VISIBLE_DEVICES=2 \
  vllm-glmasr:latest
```

服务器将在 `http://localhost:8300` 上可用

#### API 客户端示例

使用 OpenAI 兼容的 API 进行音频转写：

```python
import base64
import io
import soundfile as sf
import librosa
import numpy as np
from openai import OpenAI

# 配置客户端
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8300/v1"
)

# 加载并准备音频
def load_wav_16k(path: str):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    return audio, sr

# 转换为 base64
def wav_to_base64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# 转写音频
pcm, sr = load_wav_16k("path/to/audio.wav")
audio_b64 = wav_to_base64(pcm, sr)

resp = client.chat.completions.create(
    model="glm-asr-eligant",
    max_completion_tokens=256,
    temperature=0.0,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please transcribe this audio.<|audio|>"},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    },
                },
            ],
        }
    ],
)

print(resp.choices[0].message.content)
```

运行测试脚本：
```bash
python test_vllm_api.py
```

## 配置说明

### Docker Compose 配置

修改 [docker-compose.yaml](docker-compose.yaml) 来调整：
- **GPU 选择**：`CUDA_VISIBLE_DEVICES` 环境变量
- **端口映射**：`ports` 映射（默认：`8300:8300`）
- **GPU 内存**：`gpu-memory-utilization` 参数（默认：`0.1`）
- **模型长度**：`max-model-len` 参数（默认：`4096`）

### vLLM 服务器参数

在 [docker-compose.yaml](docker-compose.yaml:28-38) 中配置的关键参数：
- `--host`：服务器主机地址（默认：`0.0.0.0`）
- `--port`：服务器端口（默认：`8300`）
- `--served-model-name`：API 调用的模型名称（默认：`glm-asr-eligant`）
- `--dtype`：数据类型（默认：`auto`）
- `--tensor-parallel-size`：张量并行大小（默认：`1`）
- `--max-model-len`：最大模型序列长度（默认：`4096`）
- `--trust-remote-code`：允许执行远程代码
- `--gpu-memory-utilization`：GPU 内存利用率 0-1（默认：`0.1`）
- `--api-key`：API 认证密钥（默认：`EMPTY`）

## 模型架构

GLM-ASR 结合了：
- **Whisper 编码器**：音频特征提取
- **LLM 骨干**：文本生成（基于 GLM 架构）
- **多模态适配器**：桥接音频和文本表示

来自 [configuration_glmasr.py](model/configuration_glmasr.py:6-21) 的关键配置：
- **适配器类型**：MLP（默认），合并因子为 4
- **RoPE**：启用旋转位置编码
- **Spec Aug**：频谱增强（默认禁用）
- **最大 Whisper 长度**：1500 tokens
- **MLP 激活**：GELU

## 音频输入要求

- **采样率**：16 kHz（如有需要会自动重采样）
- **声道**：单声道（立体声会被下混为单声道）
- **格式**：WAV、FLAC、OGG（通过 base64 编码）
- **时长**：受 `max_model_len` 参数限制

处理器 ([processing_glmasr.py](model/processing_glmasr.py:52)) 支持：
- 音频文件路径
- NumPy 数组
- Base64 编码的音频
- 支持填充的批量处理

## API 参考

### 聊天补全接口

```
POST /v1/chat/completions
```

请求体：
```json
{
  "model": "glm-asr-eligant",
  "max_completion_tokens": 256,
  "temperature": 0.0,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Please transcribe this audio.<|audio|>"
        },
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64_encoded_audio>",
            "format": "wav"
          }
        }
      ]
    }
  ]
}
```

## vLLM 集成

本项目通过以下方式将 GLM-ASR 与 vLLM 集成：
- [server/registry.py](server/registry.py:415)：在 vLLM 模型注册表中注册 `GlmasrForConditionalGeneration`
- [server/glmasr_audio.py](server/glmasr_audio.py)：vLLM 的音频处理工具
- [server/glm_asr.py](server/glm_asr.py)：用于 vLLM 推理的 GLM-ASR 模型封装
- [dockerfile](dockerfile:12-14)：将自定义 vLLM 模型文件复制到容器中

## 故障排查

### GPU 内存问题
在 [docker-compose.yaml](docker-compose.yaml:37) 中降低 `gpu-memory-utilization` 或减小 `max-model-len`

### 推理速度慢
- 使用 `--tensor-parallel-size` 启用张量并行
- 通过 `CUDA_VISIBLE_DEVICES` 确保 GPU 选择正确
- 使用 `nvidia-smi` 检查 GPU 利用率

### 连接被拒绝
- 验证 Docker 容器是否运行：`docker ps`
- 检查端口映射是否正确
- 确保防火墙允许 8300 端口的流量

### 模型加载问题
- 验证模型权重文件在正确的目录（`./model/`）
- 检查 `trust_remote_code` 是否已启用
- 确保有足够的磁盘空间存放模型文件

## 许可证

本项目使用 Apache 2.0 许可证（参见 [server/registry.py](server/registry.py:1)）。

## 致谢

- **GLM-ASR 模型**：原始模型作者
- **vLLM**：高性能 LLM 推理引擎
- **Transformers**：HuggingFace 模型工具
- **Whisper**：OpenAI 音频编码器
