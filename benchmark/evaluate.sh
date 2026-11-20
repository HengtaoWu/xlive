#!/usr/bin/env bash
set -Eeuo pipefail
bench_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$bench_root"
bench_python="${ASYNCBENCH_PYTHON:-python}"
exec "$bench_python" -u -m async_voice_bench.cli evaluate --judge-model gpt-5.2-2025-12-11 "$@"
