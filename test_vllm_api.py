import io
import base64
import requests
import numpy as np
import soundfile as sf
import librosa
from openai import OpenAI

# ====== 配置 ======
OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://0.0.0.0:8300/v1"   # 你的 vLLM server
MODEL = "glm-asr-eligant"  # None=自动从 /v1/models 拿第一个；或手动写 "glm-asr-eligant"
AUDIO_WAV = "/data/yumu/data/audio_data/WenetSpeech4TTS/raw_audio/WenetSpeech4TTS_Rest_8/wavs/Y0000021148_sgfFHEnLHNs_S00595.wav"


def load_wav_16k(path: str):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
        sr = 16000
    return audio, sr

def pcm_to_wav_bytes(pcm: np.ndarray, sr: int) -> bytes:
    """
    把 numpy PCM 编码成 WAV 文件字节（不落盘）。
    推荐 PCM_16，兼容性最好。
    """
    buf = io.BytesIO()
    sf.write(buf, pcm, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()

def wav_bytes_to_b64(wav_bytes: bytes) -> str:
    return base64.b64encode(wav_bytes).decode("utf-8")

# ====== 主逻辑 ======
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

model_id = MODEL

pcm, sr = load_wav_16k(AUDIO_WAV)
wav_bytes = pcm_to_wav_bytes(pcm, sr)
audio_b64 = wav_bytes_to_b64(wav_bytes)

resp = client.chat.completions.create(
    model=model_id,
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
