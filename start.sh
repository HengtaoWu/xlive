#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root"

env_root="$root/.venvs"
xlive_python="$env_root/runtime/bin/python"
export QWEN_ASR_PYTHON="$env_root/qwen-asr/bin/python"
export BREEZE_PYTHON="$env_root/breeze-tts/bin/python"
export SMART_TURN_PYTHON="$env_root/smart-turn/bin/python"
export XLIVE_VLLM_BIN="$env_root/xlive-inference/bin/vllm"
export XLIVE_VLLM_PYTHON="$env_root/xlive-inference/bin/python"

for executable in \
  "$xlive_python" "$QWEN_ASR_PYTHON" "$BREEZE_PYTHON" \
  "$SMART_TURN_PYTHON" "$XLIVE_VLLM_BIN" "$XLIVE_VLLM_PYTHON"; do
  if [[ ! -x "$executable" ]]; then
    printf '\033[31m[failed]\033[0m   environment is not ready; run: bash setup.sh\n' >&2
    exit 2
  fi
done

startup_timeout="${XLIVE_STARTUP_TIMEOUT_SECONDS:-900}"

# Fixed local topology. The published ports are not user configuration.
web_host="0.0.0.0"
web_port="8000"
qwen_asr_host="127.0.0.1"
qwen_asr_port="8003"
breeze_host="127.0.0.1"
breeze_port="8004"
smart_turn_host="127.0.0.1"
smart_turn_port="8005"
inference_host="127.0.0.1"
inference_port="8002"
runtime_host="127.0.0.1"
runtime_port="8001"

log_start() {
  printf '\033[34m[start]\033[0m    %-16s %s\n' "$1" "$2"
}

log_ready() {
  printf '\033[32m[ready]\033[0m    %-16s %s\n' "$1" "$2"
}

log_info() {
  printf '\033[36m[check]\033[0m    %s\n' "$1"
}

log_failed() {
  printf '\033[31m[failed]\033[0m   %-16s %s\n' "$1" "$2" >&2
}

probe_host() {
  case "$1" in
    0.0.0.0) printf '127.0.0.1' ;;
    ::|\[::\]) printf '[::1]' ;;
    *) printf '%s' "$1" ;;
  esac
}

# Reference deployment: four model GPUs, one ASR/turn GPU, and one TTS GPU.
inference_gpus="${XLIVE_MODEL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
qwen_asr_gpus="${QWEN_ASR_CUDA_VISIBLE_DEVICES:-4}"
smart_turn_gpus="${SMART_TURN_CUDA_VISIBLE_DEVICES:-4}"
breeze_gpus="${BREEZE_CUDA_VISIBLE_DEVICES:-5}"
XLIVE_TP_SIZE="${XLIVE_TP_SIZE:-4}"
XLIVE_GPU_MEM_UTIL="${XLIVE_GPU_MEM_UTIL:-0.82}"

export QWEN_ASR_HOST="$qwen_asr_host" QWEN_ASR_PORT="$qwen_asr_port"
export BREEZE_TTS_HOST="$breeze_host" BREEZE_TTS_PORT="$breeze_port"
export SMART_TURN_HOST="$smart_turn_host" SMART_TURN_PORT="$smart_turn_port"
export XLIVE_VLLM_HOST="$inference_host" XLIVE_VLLM_PORT="$inference_port"
export XLIVE_HOST="$runtime_host" XLIVE_PORT="$runtime_port"
export XLIVE_TP_SIZE XLIVE_GPU_MEM_UTIL
export QWEN_ASR_URL="ws://$qwen_asr_host:$qwen_asr_port"
export BREEZE_TTS_URL="http://$breeze_host:$breeze_port"
export SMART_TURN_URL="ws://$smart_turn_host:$smart_turn_port/turn"
export XLIVE_MODEL_API_BASE="http://$inference_host:$inference_port/v1"
export XLIVE_MODEL_BACKEND=dual
export XLIVE_DUAL_EVENT_PATH="${XLIVE_DUAL_EVENT_PATH:-$root/logs/xlive-dual-tool-events.jsonl}"
export PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}"

export XLIVE_COMPONENTS_FACTORY="${XLIVE_COMPONENTS_FACTORY:-xlive.deployment:create_codex_components}"
export XLIVE_CODEX_BIN="${XLIVE_CODEX_BIN:-codex}"
export XLIVE_CODEX_APPROVE_FOR_ME="${XLIVE_CODEX_APPROVE_FOR_ME:-true}"

for command in awk curl npm setsid "$XLIVE_CODEX_BIN"; do
  if [[ "$command" == */* ]]; then
    [[ -x "$command" ]] || { echo "executable not found: $command" >&2; exit 2; }
  elif ! command -v "$command" >/dev/null 2>&1; then
    echo "executable not found on PATH: $command" >&2
    exit 2
  fi
done

if ! "$XLIVE_CODEX_BIN" login status >/dev/null 2>&1; then
  log_failed "codex" "not authenticated; run: $XLIVE_CODEX_BIN login"
  exit 2
fi

if [[ ! -d "$root/src/web/node_modules" ]]; then
  echo "web dependencies are missing; run: (cd '$root/src/web' && npm install)" >&2
  exit 2
fi

printf '\n\033[1mX-Live Service Setup\033[0m\n'
printf '\033[2mCreated by Hengtao Wu.\033[0m\n'

log_info "Validating model artifacts"
if ! bash "$root/scripts/download_models.sh" --check; then
  log_failed "models" "required artifacts are missing; run: bash scripts/download_models.sh"
  exit 2
fi

log_info "Validating runtime configuration"
"$xlive_python" -m xlive.cli check

printf '\n\033[1mGPU allocation\033[0m\n'
printf '  %-16s %s\n' "X-Live inference" "$inference_gpus"
printf '  %-16s %s\n' "Breeze TTS 2" "$breeze_gpus"
printf '  %-16s %s\n' "Smart Turn" "$smart_turn_gpus"
printf '  %-16s %s\n\n' "Qwen3-ASR" "$qwen_asr_gpus"

run_id="$(date -u +%Y%m%dT%H%M%SZ)"
log_dir="${XLIVE_LOG_DIR:-$root/logs/runtime/$run_id}"
mkdir -p "$log_dir"
ln -sfn "$log_dir" "$root/logs/runtime/latest"

declare -a service_names=()
declare -a service_pids=()
shutting_down=0

cleanup() {
  if (( shutting_down )); then
    return
  fi
  shutting_down=1
  trap - INT TERM HUP
  if (( ${#service_pids[@]} > 0 )); then
    echo
    printf '\033[33m[stop]\033[0m     X-Live services\n'
    for ((i=${#service_pids[@]}-1; i>=0; i--)); do
      if kill -0 "${service_pids[$i]}" 2>/dev/null; then
        kill -- "-${service_pids[$i]}" 2>/dev/null || true
      fi
    done
    for pid in "${service_pids[@]}"; do
      wait "$pid" 2>/dev/null || true
    done
  fi
}

trap cleanup EXIT
trap 'exit 130' INT TERM HUP

port_is_open() {
  local port="$1"
  local port_hex
  port_hex="$(printf '%04X' "$port")"
  awk -v suffix=":$port_hex" '
    $4 == "0A" && substr($2, length($2) - length(suffix) + 1) == suffix {
      found = 1
    }
    END { exit(found ? 0 : 1) }
  ' /proc/net/tcp /proc/net/tcp6
}

assert_port_free() {
  local name="$1"
  local port="$2"
  if port_is_open "$port"; then
    log_failed "$name" "port $port is already in use"
    exit 2
  fi
}

start_process() {
  local name="$1"
  shift
  local log_file="$log_dir/$name.log"
  log_start "$name" "$log_file"
  setsid "$@" >"$log_file" 2>&1 &
  service_names+=("$name")
  service_pids+=("$!")
}

latest_pid() {
  local index=$((${#service_pids[@]} - 1))
  printf '%s' "${service_pids[$index]}"
}

wait_for_tcp() {
  local name="$1"
  local host="$2"
  local port="$3"
  local pid="$4"
  local deadline=$((SECONDS + startup_timeout))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log_failed "$name" "exited before becoming ready; see $log_dir/$name.log"
      return 1
    fi
    if port_is_open "$port"; then
      log_ready "$name" "$host:$port"
      return 0
    fi
    sleep 1
  done
  log_failed "$name" "not ready after ${startup_timeout}s; see $log_dir/$name.log"
  return 1
}

wait_for_http() {
  local name="$1"
  local url="$2"
  local pid="$3"
  local deadline=$((SECONDS + startup_timeout))
  while (( SECONDS < deadline )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      log_failed "$name" "exited before becoming ready; see $log_dir/$name.log"
      return 1
    fi
    if curl --fail --silent --show-error --max-time 2 "$url" >/dev/null 2>&1; then
      log_ready "$name" "$url"
      return 0
    fi
    sleep 1
  done
  log_failed "$name" "not ready after ${startup_timeout}s; see $log_dir/$name.log"
  return 1
}

assert_port_free "Qwen3-ASR" "$qwen_asr_port"
start_process qwen-asr env CUDA_VISIBLE_DEVICES="$qwen_asr_gpus" "$root/scripts/start_qwen_asr.sh"
wait_for_tcp qwen-asr "$qwen_asr_host" "$qwen_asr_port" "$(latest_pid)"

assert_port_free "Breeze TTS" "$breeze_port"
start_process breeze-tts env CUDA_VISIBLE_DEVICES="$breeze_gpus" "$root/scripts/start_breeze_tts.sh"
wait_for_http breeze-tts "http://$breeze_host:$breeze_port/health" "$(latest_pid)"

assert_port_free "Smart Turn" "$smart_turn_port"
start_process smart-turn env CUDA_VISIBLE_DEVICES="$smart_turn_gpus" "$root/scripts/start_smart_turn.sh"
wait_for_tcp smart-turn "$smart_turn_host" "$smart_turn_port" "$(latest_pid)"

assert_port_free "X-Live inference" "$inference_port"
start_process xlive-inference env CUDA_VISIBLE_DEVICES="$inference_gpus" "$root/scripts/start_xlive_dual_model.sh"
wait_for_http xlive-inference "http://$inference_host:$inference_port/v1/models" "$(latest_pid)"

assert_port_free "X-Live runtime" "$runtime_port"
start_process xlive-runtime "$xlive_python" -m xlive.cli serve
wait_for_http xlive-runtime "http://$runtime_host:$runtime_port/healthz" "$(latest_pid)"

assert_port_free "X-Live web" "$web_port"
start_process xlive-web npm --prefix "$root/src/web" run dev -- --host "$web_host" --port "$web_port"
web_access_host="$(probe_host "$web_host")"
wait_for_http xlive-web "http://$web_access_host:$web_port/" "$(latest_pid)"

printf '\n\033[1;32mX-Live is ready.\033[0m\n'
cat <<EOF
  Web:       http://$web_access_host:$web_port/
  Runtime:   http://$runtime_host:$runtime_port/
  Inference: http://$inference_host:$inference_port/v1
  Logs:      $log_dir

Press Ctrl-C to stop the complete stack.
EOF

set +e
wait -n "${service_pids[@]}"
status=$?
set -e

for ((i=0; i<${#service_pids[@]}; i++)); do
  if ! kill -0 "${service_pids[$i]}" 2>/dev/null; then
    log_failed "${service_names[$i]}" "exited unexpectedly (status $status); see $log_dir/${service_names[$i]}.log"
    break
  fi
done
exit "$status"
