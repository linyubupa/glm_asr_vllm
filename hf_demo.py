import torch
from transformers import AutoModelForCausalLM, AutoProcessor
model_path = "./model/" # 或你的模型路径
audio_path = "./wavs/wuyuejianpai.wav"
audio_path2 = "./wavs/dufu.wav"

device = "cuda"

# 1. 加载
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).to(device)

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


# 2. 构造对话

conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": "./wavs/tanjianci.wav"},
                {"type": "text", "text": "Please describe this audio."},
            ],
        }
    ],
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

# 3. Apply Template
inputs = processor.apply_chat_template(
    conversations,
    return_tensors="pt",
    sampling_rate=16000,
    audio_padding="longest",
).to(device)


# 4. Generate
# 因为 processor 默认返回了 return_dict=True，这里 **inputs 可以正常工作
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)

# 5. Decode
print(processor.decode(outputs[0][len(inputs["input_ids"][0]):], skip_special_tokens=True))
