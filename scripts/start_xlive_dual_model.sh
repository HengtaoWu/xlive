#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${XLIVE_VLLM_PYTHON:-$root/.venvs/xlive-inference/bin/python}"
base="${XLIVE_MODEL_DIR:-$root/models/x-live-dual}"
checkpoint="${XLIVE_DUAL_CHECKPOINT:-$root/models/x-live-dual}"
event_path="${XLIVE_DUAL_EVENT_PATH:-$root/logs/xlive-dual-tool-events.jsonl}"
host="${XLIVE_VLLM_HOST:-127.0.0.1}"
port="${XLIVE_VLLM_PORT:-8002}"
model_name="${XLIVE_SERVED_MODEL:-x-live-dual}"
tp_size="${XLIVE_TP_SIZE:-4}"
max_model_len="${XLIVE_MAX_MODEL_LEN:-16384}"
gpu_mem_util="${XLIVE_GPU_MEM_UTIL:-0.82}"
max_num_seqs="${XLIVE_MAX_NUM_SEQS:-8}"
tool_max_tokens="${XLIVE_DUAL_TOOL_MAX_TOKENS:-512}"

[[ "$base" == /* ]] || base="$root/$base"
[[ "$checkpoint" == /* ]] || checkpoint="$root/$checkpoint"
[[ "$event_path" == /* ]] || event_path="$root/$event_path"

[[ -x "$python_bin" ]] || {
  echo "vLLM Python not found: $python_bin" >&2
  exit 1
}
for path in \
  "$base/config.json" \
  "$checkpoint/tokenizer.json" \
  "$checkpoint/tokenizer_config.json" \
  "$checkpoint/dialogue/adapter_model.safetensors" \
  "$checkpoint/tool/adapter_model.safetensors" \
  "$checkpoint/action_embedding.safetensors" \
  "$checkpoint/action_gate/model.safetensors"; do
  [[ -s "$path" ]] || { echo "missing X-Live Dual artifact: $path" >&2; exit 1; }
done

mkdir -p "$(dirname "$event_path")"
: > "$event_path"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_PLUGINS=xlive_action_gated
export XLIVE_ACTION_CHECKPOINT="$checkpoint"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

additional_config="{\"xlive\":{\"checkpoint\":\"$checkpoint\",\"tool_event_path\":\"$event_path\",\"tool_max_tokens\":$tool_max_tokens}}"
hf_overrides='{"architectures":["XLiveActionGatedQwen3MoeForCausalLM"]}'

echo "X-Live Dual base: $base"
echo "Dual checkpoint: $checkpoint"
echo "served model: $model_name"
echo "endpoint: http://$host:$port/v1"
echo "tensor parallel: $tp_size"
echo "tool events: $event_path"

exec "$python_bin" -m vllm.entrypoints.openai.api_server \
  --model "$base" \
  --tokenizer "$checkpoint" \
  --served-model-name "$model_name" \
  --host "$host" \
  --port "$port" \
  --dtype bfloat16 \
  --tensor-parallel-size "$tp_size" \
  --gpu-memory-utilization "$gpu_mem_util" \
  --max-model-len "$max_model_len" \
  --max-num-seqs "$max_num_seqs" \
  --enable-lora \
  --max-loras 2 \
  --max-lora-rank 32 \
  --lora-target-modules qkv_proj q_proj k_proj v_proj o_proj \
  --enforce-eager \
  --no-async-scheduling \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --scheduler-cls xlive_vllm.scheduler.XLiveScheduler \
  --worker-cls xlive_vllm.worker.XLiveWorker \
  --hf-overrides "$hf_overrides" \
  --additional-config "$additional_config"
