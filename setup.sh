#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_root="$root/.venvs"

# Ignore cluster-wide pip constraints when creating isolated environments.
unset PIP_CONSTRAINT PIP_BUILD_CONSTRAINT

log_step() {
  printf '\033[34m[setup]\033[0m    %s\n' "$1"
}

log_ok() {
  printf '\033[32m[ready]\033[0m    %s\n' "$1"
}

fail() {
  printf '\033[31m[failed]\033[0m   %s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "$1 is required"
}

sync_environment() {
  local name="$1"
  local prefix="$env_root/$name"
  local definition="$root/models/environments/$name.yml"
  if [[ ! -x "$prefix/bin/python" ]]; then
    log_step "Creating $name"
    conda env create --yes --quiet --prefix "$prefix" --file "$definition"
  else
    log_step "Updating $name"
    conda env update --quiet --prune \
      --prefix "$prefix" --file "$definition"
  fi
  log_ok "$name is synchronized"
}

install_qwen_asr() {
  local python_bin="$env_root/qwen-asr/bin/python"
  log_step "Installing Qwen3-ASR streaming dependencies from official PyPI"
  "$python_bin" -m pip install --upgrade --no-cache-dir \
    --index-url https://pypi.org/simple \
    'qwen-asr[vllm]==0.0.6' 'websockets==16.0'
  log_ok "qwen-asr dependencies are synchronized"
}

install_smart_turn() {
  local python_bin="$env_root/smart-turn/bin/python"
  log_step "Installing Smart Turn dependencies from official PyPI"
  "$python_bin" -m pip install --upgrade --no-cache-dir \
    --index-url https://pypi.org/simple \
    'onnxruntime-gpu==1.23.2' 'transformers==4.57.6' 'websockets==16.0' 'numpy==1.26.4'
  log_ok "smart-turn dependencies are synchronized"
}

[[ "$(uname -s)" == "Linux" ]] || fail "X-Live currently supports Linux deployment"
command -v conda >/dev/null 2>&1 || fail "Please install Conda and make conda available on PATH."
require_command npm
require_command nvidia-smi

log_step "Checking the Codex backend"
codex_bin="${XLIVE_CODEX_BIN:-codex}"
command -v "$codex_bin" >/dev/null 2>&1 \
  || fail "Codex CLI not found; install it and put 'codex' on PATH, or set XLIVE_CODEX_BIN"

mkdir -p "$env_root"

printf '\n\033[1mX-Live Environment Setup\033[0m\n'

sync_environment runtime
sync_environment qwen-asr
install_qwen_asr
sync_environment xlive-inference
log_step "Installing the X-Live Dual vLLM plugin"
"$env_root/xlive-inference/bin/python" -m pip install \
  --no-deps --no-build-isolation -e "$root"
log_ok "X-Live Dual vLLM plugin is installed"
sync_environment breeze-tts
sync_environment smart-turn
install_smart_turn

log_step "Setting up the browser application"
npm --prefix "$root/src/web" ci

log_step "Validating the installation"
"$env_root/runtime/bin/python" -c 'import importlib.util as i; assert all(i.find_spec(x) for x in ("fastapi", "numpy", "xlive"))'
"$env_root/qwen-asr/bin/python" -c 'import importlib.util as i; assert all(i.find_spec(x) for x in ("qwen_asr", "vllm", "websockets"))'
"$env_root/xlive-inference/bin/python" -c 'import importlib.util as i; assert all(i.find_spec(x) for x in ("vllm", "xlive_vllm"))'
"$env_root/xlive-inference/bin/python" -c 'from importlib.metadata import entry_points; assert any(e.name == "xlive_action_gated" for e in entry_points(group="vllm.general_plugins"))'
"$env_root/breeze-tts/bin/python" -c 'import importlib.util as i; assert all(i.find_spec(x) for x in ("qwen_tts", "torch", "transformers"))'
"$env_root/smart-turn/bin/python" -c 'import importlib.util as i; assert all(i.find_spec(x) for x in ("numpy", "onnxruntime", "transformers", "websockets"))'
"$codex_bin" --version
"$codex_bin" exec --help \
  | grep -q -- '--approve-for-me' \
  || fail "installed Codex CLI does not support --approve-for-me"
PYTHONPATH="$root/src" \
XLIVE_COMPONENTS_FACTORY="xlive.deployment:create_codex_components" \
XLIVE_CODEX_BIN="$codex_bin" \
XLIVE_CODEX_WORKSPACE="$root" \
  "$env_root/runtime/bin/python" -m xlive.cli check

printf '\n\033[1;32mX-Live environments are ready.\033[0m\n'
printf 'Authenticate Codex:         \033[1m%s login\033[0m\n' "$codex_bin"
printf 'Download models with: \033[1mbash scripts/download_models.sh\033[0m\n'
printf 'Then start with:      \033[1mbash start.sh\033[0m\n'
