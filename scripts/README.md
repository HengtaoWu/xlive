# Deployment settings

The default deployment uses six GPUs: **0–3 for X-Live Dual, 4 for ASR + Smart Turn, 5 for TTS**. The values below are a reference; edit the code locations listed afterward.

```ini
# Runtime
XLIVE_HOST=127.0.0.1
XLIVE_PORT=8001
XLIVE_LOG_LEVEL=info
XLIVE_ENABLE_PERSISTENCE=true
XLIVE_DATA_DIR=data
XLIVE_LESSON_PREFETCH_DELAY_SECONDS=5

# Backend agent
XLIVE_COMPONENTS_FACTORY=xlive.deployment:create_codex_components
XLIVE_CODEX_WORKSPACE=.
XLIVE_CODEX_BIN=codex
XLIVE_CODEX_SANDBOX=workspace-write
XLIVE_CODEX_APPROVE_FOR_ME=true
XLIVE_CODEX_REASONING_EFFORT=medium
XLIVE_CODEX_MAX_CONCURRENCY=1
XLIVE_CODEX_TASK_TIMEOUT_SECONDS=1800

# X-Live model
XLIVE_MODEL_DIR=models/x-live-dual
XLIVE_MODEL_ARTIFACTS=models/x-live-dual
XLIVE_MODEL_API_BASE=http://127.0.0.1:8002/v1
XLIVE_TP_SIZE=4
XLIVE_MODEL_TEMPERATURE=0.3
XLIVE_MODEL_MAX_TOKENS=512
XLIVE_MODEL_TIMEOUT_SECONDS=120
XLIVE_MODEL_MAX_TOOL_ROUNDS=4
XLIVE_MODEL_BACKEND=dual
XLIVE_DUAL_CHECKPOINT=models/x-live-dual
XLIVE_DUAL_EVENT_PATH=logs/xlive-dual-tool-events.jsonl

# Speech services
QWEN_ASR_URL=ws://127.0.0.1:8003
QWEN_ASR_TIMEOUT_SECONDS=30
BREEZE_TTS_URL=http://127.0.0.1:8004
BREEZE_TTS_TIMEOUT_SECONDS=120
SMART_TURN_URL=ws://127.0.0.1:8005/turn
SMART_TURN_TIMEOUT_SECONDS=5

# GPUs and startup
XLIVE_MODEL_CUDA_VISIBLE_DEVICES=0,1,2,3
XLIVE_GPU_MEM_UTIL=0.82
QWEN_ASR_CUDA_VISIBLE_DEVICES=4
SMART_TURN_CUDA_VISIBLE_DEVICES=4
BREEZE_CUDA_VISIBLE_DEVICES=5
XLIVE_STARTUP_TIMEOUT_SECONDS=900
```

| Change | Edit here |
| --- | --- |
| GPUs, ports, startup timeout, backend factory and approval | [start.sh](../start.sh) |
| Model/checkpoint paths | [start_xlive_dual_model.sh](start_xlive_dual_model.sh) |
| Model artifacts, temperature, token limits and timeout | `from_env()` / `_DEFAULT_ARTIFACTS_DIR` in [frontend config](../src/xlive/frontend_model/config.py) |
| Runtime logging, persistence and data directory | `from_env()` in [application.py](../src/xlive/application.py) |
| Codex workspace, sandbox, reasoning and task limits | `from_env()` in [Codex config](../src/xlive/agent/codex/config.py) |
| Speech request timeouts | `from_env()` in [ASR](../src/xlive/speech/asr/qwen_asr.py), [TTS](../src/xlive/speech/tts/breeze.py), [Smart Turn](../src/xlive/speech/turn_detector/smart_turn.py) |

Change `XLIVE_TP_SIZE` with the model GPU count. Keep inference/runtime model paths consistent; download/check paths are in [download_models.sh](download_models.sh). When changing the runtime port, also update the proxy `target` in [webpack.app.config.mjs](../src/web/webpack.app.config.mjs) and check [browser connection settings](../src/web/app/src/app-config.ts).

`XLIVE_CODEX_WORKSPACE=.` means the repository root under `start.sh`. The launcher enables `XLIVE_CODEX_APPROVE_FOR_ME`; direct Python startup defaults to false.

# 部署配置

默认部署使用 6 张 GPU。X-Live Dual 使用 0–3 号卡，Qwen3-ASR 和 Smart Turn 共用 4 号卡，Breeze TTS 使用 5 号卡。

| 配置项 | 修改位置 |
| --- | --- |
| GPU 分配、服务端口、启动超时、Agent 后端和审批设置 | [start.sh](../start.sh) |
| 推理模型与权重路径 | [start_xlive_dual_model.sh](start_xlive_dual_model.sh) |
| 运行时加载的模型文件、采样温度、输出 token 上限和请求超时 | [前端模型配置](../src/xlive/frontend_model/config.py) 中的 `from_env()` 和 `_DEFAULT_ARTIFACTS_DIR` |
| 运行时日志、持久化和数据目录 | [application.py](../src/xlive/application.py) 中的 `from_env()` |
| Codex 工作目录、沙箱、推理强度、并发数和任务超时 | [Codex 配置](../src/xlive/agent/codex/config.py) 中的 `from_env()` |
| 语音服务请求超时 | 各适配器的 `from_env()` 方法，见 [ASR](../src/xlive/speech/asr/qwen_asr.py)、[TTS](../src/xlive/speech/tts/breeze.py) 和 [Smart Turn](../src/xlive/speech/turn_detector/smart_turn.py) |

调整 X-Live Dual 使用的 GPU 数量时，将 `XLIVE_TP_SIZE` 设为相同数量。迁移模型时，确保推理服务和运行时指向同一套模型文件，并核对[下载脚本](download_models.sh)中的下载和校验路径。

修改运行时端口后，同步更新网页代理的 `target`，并核对浏览器的连接设置。[代理配置](../src/web/webpack.app.config.mjs) · [浏览器连接配置](../src/web/app/src/app-config.ts)

通过 `start.sh` 启动时，`XLIVE_CODEX_WORKSPACE=.` 指向仓库根目录，启动脚本会开启 `XLIVE_CODEX_APPROVE_FOR_ME`。直接用 Python 启动时，该选项默认关闭。
