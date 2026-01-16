# Dockerfile
FROM vllm/vllm-openai:v0.12.0

# 目标目录（你指定的 vLLM models 路径）
ARG VLLM_MODELS_DIR=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models

# 确保目录存在（通常已经存在，但做一下更稳）
RUN mkdir -p ${VLLM_MODELS_DIR}

# 1) 覆盖/新增你的模型文件
# registry.py 会直接替换原有文件（同名覆盖）
COPY ./server/glmasr_audio.py ${VLLM_MODELS_DIR}/glmasr_audio.py
COPY ./server/glmasr.py       ${VLLM_MODELS_DIR}/glmasr.py
COPY ./server/registry.py     ${VLLM_MODELS_DIR}/registry.py

# 2) 拷贝模型目录到镜像里（你可以改成 /data/model 或 /models）
COPY ./model /model

RUN pip install --no-cache-dir \
      websockets torchaudio soundfile fastapi uvicorn loguru librosa \
      -i https://pypi.tuna.tsinghua.edu.cn/simple
# （可选）做个简单校验：确保文件已就位
RUN python3 -c "import os; p='${VLLM_MODELS_DIR}'; \
print('models_dir=', p); \
print('has_glmasr.py=', os.path.exists(os.path.join(p,'glmasr.py'))); \
print('has_glmasr_audio.py=', os.path.exists(os.path.join(p,'glmasr_audio.py'))); \
print('has_registry.py=', os.path.exists(os.path.join(p,'registry.py'))); \
print('model_dir_exists=', os.path.exists('/model'))"
