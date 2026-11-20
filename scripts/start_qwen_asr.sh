#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${QWEN_ASR_RUNTIME_DIR:-$root/models/qwen-asr/service}"
model_dir="${QWEN_ASR_MODEL_DIR:-$root/models/qwen-asr/checkpoint}"
python_bin="${QWEN_ASR_PYTHON:-python}"
host="${QWEN_ASR_HOST:-127.0.0.1}"
port="${QWEN_ASR_PORT:-8003}"
gpu_memory_utilization="${QWEN_ASR_GPU_MEMORY_UTILIZATION:-0.35}"
max_new_tokens="${QWEN_ASR_MAX_NEW_TOKENS:-48}"
chunk_size_sec="${QWEN_ASR_CHUNK_SIZE_SEC:-1.0}"
unfixed_chunk_num="${QWEN_ASR_UNFIXED_CHUNK_NUM:-2}"
unfixed_token_num="${QWEN_ASR_UNFIXED_TOKEN_NUM:-5}"
export PYTHONPYCACHEPREFIX="${QWEN_ASR_PYTHONPYCACHEPREFIX:-/tmp/xlive-qwen-asr-pycache}"

for path in "$model_dir/config.json" "$model_dir/model.safetensors.index.json"; do
  [[ -s "$path" ]] || { echo "missing Qwen3-ASR artifact: $path" >&2; exit 1; }
done

shopt -s nullglob
shards=("$model_dir"/model-*-of-*.safetensors)
(( ${#shards[@]} >= 1 )) || { echo "missing Qwen3-ASR weight shards in $model_dir" >&2; exit 1; }

exec "$python_bin" "$runtime_dir/qwen_streaming_server.py" \
  --host "$host" \
  --port "$port" \
  --model "$model_dir" \
  --gpu-memory-utilization "$gpu_memory_utilization" \
  --max-new-tokens "$max_new_tokens" \
  --chunk-size-sec "$chunk_size_sec" \
  --unfixed-chunk-num "$unfixed_chunk_num" \
  --unfixed-token-num "$unfixed_token_num"
