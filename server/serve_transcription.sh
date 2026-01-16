#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/yumu/glmasrinfer/model}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8300}"
TP="${TP:-1}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVED_NAME="${SERVED_NAME:-glm-asr-eligant}"

vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name "$SERVED_NAME" \
  --dtype "$DTYPE" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  --api-key EMPTY \
  --gpu-memory-utilization 0.2 \
  --disable-log-stats

