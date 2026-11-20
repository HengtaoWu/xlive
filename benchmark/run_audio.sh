#!/usr/bin/env bash
set -Eeuo pipefail
bench_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$bench_root"
export PYTHONPATH="$bench_root${PYTHONPATH:+:$PYTHONPATH}"
bench_python="${ASYNCBENCH_PYTHON:-python}"
exec "$bench_python" -u -m async_voice_bench.run_audio "$@"
