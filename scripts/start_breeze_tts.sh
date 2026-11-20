#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${BREEZE_RUNTIME_DIR:-$root/models/breeze-tts}"
model_dir="${BREEZE_MODEL_DIR:-$root/models/breeze-tts/checkpoint}"
python_bin="${BREEZE_PYTHON:-python}"
host="${BREEZE_TTS_HOST:-127.0.0.1}"
port="${BREEZE_TTS_PORT:-8004}"
reference_audio="${BREEZE_REFERENCE_AUDIO:-$root/models/breeze-tts/reference-zh.wav}"
reference_text="${BREEZE_REFERENCE_TEXT:-这是中文参考音频的准确文字稿}"
export PYTHONPYCACHEPREFIX="${BREEZE_PYTHONPYCACHEPREFIX:-/tmp/xlive-breeze-pycache}"

for path in \
  "$runtime_dir/service/api.py" \
  "$reference_audio" \
  "$model_dir/config.json" \
  "$model_dir/model-00001-of-00002.safetensors" \
  "$model_dir/model-00002-of-00002.safetensors" \
  "$model_dir/audio_tokenizer/model.safetensors"; do
  if [[ ! -s "$path" ]]; then
    echo "missing Breeze TTS artifact: $path" >&2
    exit 1
  fi
done

if [[ "${BREEZE_ALLOW_NON_CUDA:-0}" != "1" ]]; then
  "$python_bin" -c \
    'import torch; raise SystemExit(0 if torch.cuda.is_available() else "Breeze TTS requires a CUDA-visible worker (set BREEZE_ALLOW_NON_CUDA=1 only for deliberate debugging)")'
fi

args=(
  -m service.api "$model_dir"
  --host "$host"
  --port "$port"
  --default-reference-audio "$reference_audio"
  --default-reference-text "$reference_text"
)
if [[ "${BREEZE_FAST_ALL:-1}" == "1" ]]; then
  args+=(--fast-all)
fi

cd "$runtime_dir"
exec "$python_bin" "${args[@]}"
