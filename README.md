# GLM-ASR with vLLM

An audio speech recognition (ASR) project that integrates the GLM-ASR model with vLLM for high-performance inference. This project provides both local inference capabilities and a scalable API server using Docker.

This project is an extension/supplement to the original [GLM-ASR project](https://github.com/zai-org/GLM-ASR), adding vLLM integration for production-ready deployment and OpenAI-compatible API support.

## Features

- **Audio Transcription**: Transcribe audio files using GLM-ASR model
- **Audio Description**: Generate textual descriptions of audio content
- **OpenAI-Compatible API**: vLLM server provides OpenAI-compatible API endpoints
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **High Performance**: Leverages vLLM for efficient GPU-accelerated inference
- **Flexible Audio Input**: Supports various audio formats and input methods

## Project Structure

```
glm_asr_vllm/
├── model/                  # Model configuration and implementation
│   ├── configuration_glmasr.py    # GLM-ASR configuration
│   ├── modeling_glmasr.py         # GLM-ASR model implementation
│   ├── modeling_audio.py          # Audio encoding/decoding
│   └── processing_glmasr.py       # Audio processing utilities
├── server/                # vLLM integration files
│   ├── glmasr_audio.py     # Audio processing for vLLM
│   ├── glm_asr.py          # GLM-ASR vLLM model wrapper
│   ├── registry.py         # Model registry (vLLM)
│   └── server_ws.py        # WebSocket server
├── wavs/                  # Sample audio files
├── docker-compose.yaml     # Docker Compose configuration
├── dockerfile             # Docker image build configuration
├── hf_demo.py             # HuggingFace Transformers demo
└── test_vllm_api.py       # OpenAI API client test script
```

## Prerequisites

- Python 3.12+
- CUDA-capable GPU (recommended)
- Docker (for containerized deployment)
- Docker Compose (optional)

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd glm_asr_vllm
```

2. Install dependencies:
```bash
pip install torch transformers soundfile librosa openai
```

3. Download the model from HuggingFace and place it in the `./model/` directory:

```bash
# Download using huggingface-cli (recommended)
huggingface-cli download bupalinyu/glm-asr-eligant --local-dir ./model

# Or use git lfs
git lfs install
git clone https://huggingface.co/bupalinyu/glm-asr-eligant ./model
```

**Note**: The model ID on HuggingFace is `bupalinyu/glm-asr-eligant`. After downloading, ensure all model files are in the `./model/` directory.

### Option 2: Docker Deployment

1. Build the Docker image:
```bash
docker build -t vllm-glmasr:latest .
```

2. Deploy using Docker Compose:
```bash
docker-compose up -d
```

## Usage

### HuggingFace Transformers Demo

Run the local demo script to transcribe audio files:

```python
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "./model/",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
).to("cuda")

processor = AutoProcessor.from_pretrained("./model/", trust_remote_code=True)

# Define conversations
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

# Process and generate
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

Run the demo:
```bash
python hf_demo.py
```

### vLLM API Server

#### Start the Server

Using Docker Compose:
```bash
docker-compose up -d
```

Or manually with Docker:
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

Server will be available at `http://localhost:8300`

#### API Client Example

Use the OpenAI-compatible API to transcribe audio:

```python
import base64
import io
import soundfile as sf
import librosa
import numpy as np
from openai import OpenAI

# Configure client
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8300/v1"
)

# Load and prepare audio
def load_wav_16k(path: str):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    return audio, sr

# Convert to base64
def wav_to_base64(wav: np.ndarray, sr: int) -> str:
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# Transcribe
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

Run the test script:
```bash
python test_vllm_api.py
```

## Configuration

### Docker Compose Settings

Modify [docker-compose.yaml](docker-compose.yaml) to adjust:
- **GPU Selection**: `CUDA_VISIBLE_DEVICES` environment variable
- **Port**: `ports` mapping (default: `8300:8300`)
- **GPU Memory**: `gpu-memory-utilization` parameter (default: `0.1`)
- **Model Length**: `max-model-len` parameter (default: `4096`)

### vLLM Server Parameters

Key parameters configured in [docker-compose.yaml](docker-compose.yaml:28-38):
- `--host`: Server host address (default: `0.0.0.0`)
- `--port`: Server port (default: `8300`)
- `--served-model-name`: Model name for API calls (default: `glm-asr-eligant`)
- `--dtype`: Data type (default: `auto`)
- `--tensor-parallel-size`: Tensor parallelism size (default: `1`)
- `--max-model-len`: Maximum model sequence length (default: `4096`)
- `--trust-remote-code`: Allow remote code execution
- `--gpu-memory-utilization`: GPU memory utilization 0-1 (default: `0.1`)
- `--api-key`: API key for authentication (default: `EMPTY`)

## Model Architecture

GLM-ASR combines:
- **Whisper Encoder**: Audio feature extraction
- **LLM Backbone**: Text generation (based on GLM architecture)
- **Multimodal Adapter**: Bridges audio and text representations

Key configurations from [configuration_glmasr.py](model/configuration_glmasr.py:6-21):
- **Adapter Type**: MLP (default) with merge factor of 4
- **RoPE**: Rotary Position Embeddings enabled
- **Spec Aug**: Spectral augmentation (disabled by default)
- **Max Whisper Length**: 1500 tokens
- **MLP Activation**: GELU

## Audio Input Requirements

- **Sampling Rate**: 16 kHz (audio will be resampled if needed)
- **Channels**: Mono (stereo will be downmixed to mono)
- **Formats**: WAV, FLAC, OGG (via base64 encoding)
- **Duration**: Limited by `max_model_len` parameter

The processor ([processing_glmasr.py](model/processing_glmasr.py:52)) supports:
- Audio file paths
- NumPy arrays
- Base64 encoded audio
- Batch processing with padding

## API Reference

### Chat Completions Endpoint

```
POST /v1/chat/completions
```

Request body:
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

## vLLM Integration

The project integrates GLM-ASR with vLLM through:
- [server/registry.py](server/registry.py:415): Registers `GlmasrForConditionalGeneration` in vLLM's model registry
- [server/glmasr_audio.py](server/glmasr_audio.py): Audio processing utilities for vLLM
- [server/glm_asr.py](server/glm_asr.py): GLM-ASR model wrapper for vLLM inference
- [dockerfile](dockerfile:12-14): Copies custom vLLM model files into the container

## Troubleshooting

### GPU Memory Issues
Reduce `gpu-memory-utilization` or decrease `max-model-len` in [docker-compose.yaml](docker-compose.yaml:37)

### Slow Inference
- Enable tensor parallelism with `--tensor-parallel-size`
- Ensure proper GPU selection via `CUDA_VISIBLE_DEVICES`
- Check GPU utilization with `nvidia-smi`

### Connection Refused
- Verify the Docker container is running: `docker ps`
- Check port mapping is correct
- Ensure firewall allows traffic on port 8300

### Model Loading Issues
- Verify model weights are in the correct directory (`./model/`)
- Check `trust_remote_code` is enabled
- Ensure sufficient disk space for model files

## License

This project uses the Apache 2.0 license (see [server/registry.py](server/registry.py:1)).

## Acknowledgments

- **GLM-ASR Model**: Original model authors
- **vLLM**: High-performance LLM inference engine
- **Transformers**: HuggingFace model utilities
- **Whisper**: OpenAI audio encoder
