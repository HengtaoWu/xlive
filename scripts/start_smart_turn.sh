#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${SMART_TURN_PYTHON:-python}"
host="${SMART_TURN_HOST:-127.0.0.1}"
port="${SMART_TURN_PORT:-8005}"
model="${SMART_TURN_MODEL_PATH:-$root/models/smart-turn/checkpoint/smart-turn-v3.2-gpu.onnx}"
threshold="${SMART_TURN_THRESHOLD:-0.5}"
export PYTHONPYCACHEPREFIX="${SMART_TURN_PYTHONPYCACHEPREFIX:-/tmp/xlive-smart-turn-pycache}"

[[ -s "$model" ]] || { echo "Smart Turn model not found: $model" >&2; exit 2; }

exec "$python_bin" "$root/models/smart-turn/service/smart_turn_server.py" \
  --host "$host" --port "$port" --model "$model" --threshold "$threshold"
