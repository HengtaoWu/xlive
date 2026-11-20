#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Download Models From Huggingface or ModelScope.
XLIVE_HF_REPO="${XLIVE_HF_REPO:-HengtaoWu/X-Live-Dual}"
XLIVE_MODELSCOPE_REPO="${XLIVE_MODELSCOPE_REPO:-HengtaoWu/X-Live-Dual}"
QWEN_ASR_HF_REPO="${QWEN_ASR_HF_REPO:-Qwen/Qwen3-ASR-1.7B}"
QWEN_ASR_MODELSCOPE_REPO="${QWEN_ASR_MODELSCOPE_REPO:-Qwen/Qwen3-ASR-1.7B}"
BREEZE_HF_REPO="${BREEZE_HF_REPO:-BreezeBlue/Breeze-TTS-2}"
SMART_TURN_HF_REPO="${SMART_TURN_HF_REPO:-pipecat-ai/smart-turn-v3}"
MODELSCOPE_CN_ENDPOINT="${MODELSCOPE_CN_ENDPOINT:-https://modelscope.cn}"
MODELSCOPE_AI_ENDPOINT="${MODELSCOPE_AI_ENDPOINT:-https://www.modelscope.ai}"

XLIVE_DIR="$ROOT/models/x-live-dual"
QWEN_ASR_DIR="$ROOT/models/qwen-asr/checkpoint"
BREEZE_DIR="$ROOT/models/breeze-tts/checkpoint"
SMART_TURN_DIR="$ROOT/models/smart-turn/checkpoint"

# Only the files the runtime actually loads.
BREEZE_FILES=(
  LICENSE
  config.json
  generation_config.json
  model.safetensors.index.json
  model-00001-of-00002.safetensors
  model-00002-of-00002.safetensors
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  audio_tokenizer/config.json
  audio_tokenizer/configuration.json
  audio_tokenizer/model.safetensors
  audio_tokenizer/preprocessor_config.json
)

SMART_TURN_FILES=(smart-turn-v3.2-gpu.onnx smart-turn-v3.2-cpu.onnx)

SOURCE="huggingface"
CHECK_ONLY=0
FORCE=0

log_download() {
  printf '\033[34m[download]\033[0m %-16s %s\n' "$1" "$2"
}

log_skip() {
  printf '\033[33m[skip]\033[0m     %-16s %s\n' "$1" "already complete"
}

log_ok() {
  printf '\033[32m[ok]\033[0m       %-16s %s\n' "$1" "$2"
}

log_failed() {
  printf '\033[31m[failed]\033[0m   %-16s %s\n' "$1" "$2" >&2
}

log_status() {
  local color="$1"
  local symbol="$2"
  local label="$3"
  local destination="$4"
  printf '\033[%sm%s\033[0m  \033[1m%-16s\033[0m \033[2m%s\033[0m\n' \
    "$color" "$symbol" "$label" "$destination"
}

print_welcome() {
  printf '\n\033[1mX-Live Model Setup\033[0m\n'
  printf 'Source: %s\n\n' "$SOURCE"
}

print_finished() {
  printf '\n\033[1;32mModel download complete.\033[0m\n'
}

usage() {
  cat <<'EOF'
Download the models required by X-Live into their expected runtime paths.

Usage:
  bash scripts/download_models.sh [--force]
  bash scripts/download_models.sh modelscope [--force]
  bash scripts/download_models.sh --check

Hugging Face is used by default for every model.
ModelScope mode covers X-Live and Qwen3-ASR. Smart Turn and Breeze TTS 2 are
downloaded from Hugging Face because their required artifacts are hosted there.
Complete components are skipped unless --force is specified.

Destinations:
  models/x-live-dual/
  models/qwen-asr/checkpoint/
  models/breeze-tts/checkpoint/
  models/smart-turn/checkpoint/
EOF
}

for arg in "$@"; do
  case "$arg" in
    huggingface|modelscope) SOURCE="$arg" ;;
    --check) CHECK_ONLY=1 ;;
    --force) FORCE=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

have_file() {
  [[ -s "$1" ]]
}

validate_file_list() {
  local root="$1"
  shift
  local name
  for name in "$@"; do
    have_file "$root/$name" || return 1
  done
}

validate_xlive() {
  local path
  for path in \
    "$XLIVE_DIR/config.json" \
    "$XLIVE_DIR/model.safetensors.index.json" \
    "$XLIVE_DIR/chat_template.jinja" \
    "$XLIVE_DIR/tokenizer.json" \
    "$XLIVE_DIR/tokenizer_config.json" \
    "$XLIVE_DIR/tools.json" \
    "$XLIVE_DIR/system_prompt.txt" \
    "$XLIVE_DIR/xlive_protocol.json"; do
    have_file "$path" || return 1
  done
  for path in \
    "$XLIVE_DIR/action_gate/config.json" \
    "$XLIVE_DIR/action_gate/model.safetensors" \
    "$XLIVE_DIR/action_embedding.safetensors" \
    "$XLIVE_DIR/dialogue/adapter_config.json" \
    "$XLIVE_DIR/dialogue/adapter_model.safetensors" \
    "$XLIVE_DIR/tool/adapter_config.json" \
    "$XLIVE_DIR/tool/adapter_model.safetensors"; do
    have_file "$path" || return 1
  done
  [[ "$(find "$XLIVE_DIR" -maxdepth 1 -type f -name 'model-*-of-*.safetensors' | wc -l)" -eq 13 ]]
}

validate_qwen_asr() {
  have_file "$QWEN_ASR_DIR/config.json" \
    && have_file "$QWEN_ASR_DIR/model.safetensors.index.json" \
    && [[ "$(find "$QWEN_ASR_DIR" -maxdepth 1 -type f -name 'model-*-of-*.safetensors' | wc -l)" -ge 1 ]]
}

validate_breeze() {
  validate_file_list "$BREEZE_DIR" "${BREEZE_FILES[@]}"
}

validate_smart_turn() {
  validate_file_list "$SMART_TURN_DIR" "${SMART_TURN_FILES[@]}"
}

report_status() {
  local failed=0
  local label validator destination
  while IFS='|' read -r label validator destination; do
    if "$validator"; then
      log_status "1;32" "✔" "$label" "$destination"
    else
      log_status "1;31" "✘" "$label" "$destination"
      failed=1
    fi
  done <<EOF
X-Live Dual|validate_xlive|$XLIVE_DIR
Qwen3-ASR|validate_qwen_asr|$QWEN_ASR_DIR
Breeze TTS 2|validate_breeze|$BREEZE_DIR
Smart Turn v3.2|validate_smart_turn|$SMART_TURN_DIR
EOF
  return "$failed"
}

if [[ "$CHECK_ONLY" == "1" ]]; then
  report_status
  exit $?
fi

HF_BIN=""
if [[ -x "$ROOT/.venvs/runtime/bin/hf" ]]; then
  HF_BIN="$ROOT/.venvs/runtime/bin/hf"
elif command -v hf >/dev/null 2>&1; then
  HF_BIN="hf"
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN="huggingface-cli"
else
  log_failed "Hugging Face" "CLI missing; install it with: python -m pip install -U huggingface_hub"
  exit 1
fi

hf_download() {
  local repo="$1"
  local destination="$2"
  shift 2
  local args=(download "$repo")
  if [[ "$#" -gt 0 ]]; then
    args+=("$@")
  fi
  args+=(--local-dir "$destination")
  if [[ "$FORCE" == "1" ]]; then
    args+=(--force-download)
  fi
  "$HF_BIN" "${args[@]}"
}

require_modelscope() {
  if [[ -x "$ROOT/.venvs/runtime/bin/modelscope" ]]; then
    MODELSCOPE_BIN="$ROOT/.venvs/runtime/bin/modelscope"
  elif command -v modelscope >/dev/null 2>&1; then
    MODELSCOPE_BIN="$(command -v modelscope)"
  else
    log_failed "ModelScope" "CLI missing; install it with: python -m pip install -U modelscope"
    exit 1
  fi
}

modelscope_download() {
  local endpoint="$1"
  local repo="$2"
  local destination="$3"
  shift 3
  require_modelscope
  MODELSCOPE_ENDPOINT="$endpoint" "$MODELSCOPE_BIN" download \
    --model "$repo" \
    "$@" \
    --local_dir "$destination"
}

download_hf_component() {
  local label="$1"
  local validator="$2"
  local repo="$3"
  local destination="$4"
  shift 4

  if [[ "$FORCE" != "1" ]] && "$validator"; then
    log_skip "$label"
    return
  fi

  log_download "$label" "https://huggingface.co/$repo"

  mkdir -p "$destination"
  hf_download "$repo" "$destination" "$@"

  if "$validator"; then
    log_ok "$label" "$destination"
  else
    log_failed "$label" "required files are missing in $destination"
    exit 1
  fi
}

download_modelscope_component() {
  local label="$1"
  local validator="$2"
  local endpoint="$3"
  local repo="$4"
  local destination="$5"
  shift 5

  if [[ "$FORCE" != "1" ]] && "$validator"; then
    log_skip "$label"
    return
  fi

  log_download "$label" "$endpoint/models/$repo"

  mkdir -p "$destination"
  modelscope_download "$endpoint" "$repo" "$destination" "$@"

  if "$validator"; then
    log_ok "$label" "$destination"
  else
    log_failed "$label" "required files are missing in $destination"
    exit 1
  fi
}

download_xlive() {
  if [[ "$FORCE" != "1" ]] && validate_xlive; then
    log_skip "X-Live Dual"
    return
  fi

  mkdir -p "$XLIVE_DIR"

  case "$SOURCE" in
    huggingface)
      log_download "X-Live Dual" "https://huggingface.co/$XLIVE_HF_REPO"
      hf_download "$XLIVE_HF_REPO" "$XLIVE_DIR"
      ;;

    modelscope)
      log_download "X-Live Dual" "$MODELSCOPE_CN_ENDPOINT/models/$XLIVE_MODELSCOPE_REPO"
      modelscope_download \
        "$MODELSCOPE_CN_ENDPOINT" "$XLIVE_MODELSCOPE_REPO" "$XLIVE_DIR"
      ;;
  esac

  if validate_xlive; then
    log_ok "X-Live Dual" "$XLIVE_DIR"
  else
    log_failed "X-Live Dual" "unified backbone or Dual artifacts are incomplete in $XLIVE_DIR"
    exit 1
  fi
}

print_welcome
download_xlive
case "$SOURCE" in
  huggingface)
    download_hf_component \
      "Qwen3-ASR 1.7B" validate_qwen_asr "$QWEN_ASR_HF_REPO" "$QWEN_ASR_DIR"
    ;;
  modelscope)
    download_modelscope_component \
      "Qwen3-ASR 1.7B" validate_qwen_asr "$MODELSCOPE_CN_ENDPOINT" \
      "$QWEN_ASR_MODELSCOPE_REPO" "$QWEN_ASR_DIR"
    ;;
esac
download_hf_component \
  "Breeze TTS 2" validate_breeze "$BREEZE_HF_REPO" "$BREEZE_DIR" \
  "${BREEZE_FILES[@]}"
download_hf_component \
  "Smart Turn v3.2" validate_smart_turn "$SMART_TURN_HF_REPO" "$SMART_TURN_DIR" \
  "${SMART_TURN_FILES[@]}"

printf '\n\033[1mModel status\033[0m\n'
report_status
print_finished
