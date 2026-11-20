# Runtime Environments

X-Live uses separate Conda environments because the realtime runtime, Qwen3-ASR,
vLLM, Breeze TTS 2, and Smart Turn require different Python and CUDA dependencies.

The repository-level `setup.sh` manages these environments on Linux. It
requires `conda` and `npm` on PATH and an NVIDIA driver visible through `nvidia-smi`.

It creates or updates:

| Environment | Python | Purpose |
| --- | --- | --- |
| `.venvs/runtime` | 3.10 | X-Live runtime |
| `.venvs/qwen-asr` | 3.10 | Qwen3-ASR 1.7B streaming service |
| `.venvs/xlive-inference` | 3.12 | vLLM model serving |
| `.venvs/breeze-tts` | 3.11 | Breeze TTS 2 serving |
| `.venvs/smart-turn` | 3.10 | Smart Turn v3.2 semantic endpointing |

Web dependencies are installed under `src/web/node_modules/`. The Codex CLI is
expected on `PATH`; set `XLIVE_CODEX_BIN` to use a specific binary.

### Manual creation

If you need to manage the environments yourself:

```bash
conda env create -p .venvs/runtime -f models/environments/runtime.yml
conda env create -p .venvs/qwen-asr -f models/environments/qwen-asr.yml
conda env create -p .venvs/xlive-inference -f models/environments/xlive-inference.yml
conda env create -p .venvs/breeze-tts -f models/environments/breeze-tts.yml
conda env create -p .venvs/smart-turn -f models/environments/smart-turn.yml
```

Each YAML file pins its Python version and base dependencies. The launch scripts
use these fixed paths automatically.

`qwen-asr` is obtained from official PyPI because some cluster mirrors do not
carry it. `setup.sh` does this automatically; after a manual environment
creation, run:

```bash
env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT \
  .venvs/qwen-asr/bin/python -m pip install --index-url https://pypi.org/simple \
  'qwen-asr[vllm]==0.0.6' 'websockets==16.0'
```

`smart-turn` is installed from official PyPI as well:

```bash
env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT \
  .venvs/smart-turn/bin/python -m pip install --index-url https://pypi.org/simple \
  'onnxruntime-gpu==1.23.2' 'transformers==4.57.6' 'websockets==16.0' 'numpy==1.26.4'
```

# 环境配置

由于实时运行时、Qwen3-ASR、vLLM、Breeze TTS 2 与 Smart Turn 对 Python 和 CUDA 的依赖
不同，X-Live 为它们使用相互独立的 Conda 环境。

仓库根目录的 `setup.sh` 用于在 Linux 上管理这些环境。请先将 `conda` 和 `npm` 加入 PATH，并确保 `nvidia-smi` 能检测到 NVIDIA 驱动。

脚本会创建或更新以下环境：

| 环境 | Python | 用途 |
| --- | --- | --- |
| `.venvs/runtime` | 3.10 | X-Live 运行时 |
| `.venvs/qwen-asr` | 3.10 | Qwen3-ASR 1.7B 流式服务 |
| `.venvs/xlive-inference` | 3.12 | vLLM 模型服务 |
| `.venvs/breeze-tts` | 3.11 | Breeze TTS 2 服务 |
| `.venvs/smart-turn` | 3.10 | Smart Turn v3.2 语义端点检测 |

前端依赖安装在 `src/web/node_modules/` 下；Codex CLI 需位于 `PATH` 中，也可以
通过 `XLIVE_CODEX_BIN` 指定具体路径。

### 手动创建

如需自行管理环境，可运行：

```bash
conda env create -p .venvs/runtime -f models/environments/runtime.yml
conda env create -p .venvs/qwen-asr -f models/environments/qwen-asr.yml
conda env create -p .venvs/xlive-inference -f models/environments/xlive-inference.yml
conda env create -p .venvs/breeze-tts -f models/environments/breeze-tts.yml
conda env create -p .venvs/smart-turn -f models/environments/smart-turn.yml
```

每个 YAML 文件都固定了 Python 版本与基础依赖，启动脚本会自动使用这些约定路径。

部分集群镜像没有收录 `qwen-asr`，因此它从官方 PyPI 获取。`setup.sh` 会自动完成；
若手动创建环境，请执行：

```bash
env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT \
  .venvs/qwen-asr/bin/python -m pip install --index-url https://pypi.org/simple \
  'qwen-asr[vllm]==0.0.6' 'websockets==16.0'
```

`smart-turn` 同样从官方 PyPI 安装：

```bash
env -u PIP_CONSTRAINT -u PIP_BUILD_CONSTRAINT \
  .venvs/smart-turn/bin/python -m pip install --index-url https://pypi.org/simple \
  'onnxruntime-gpu==1.23.2' 'transformers==4.57.6' 'websockets==16.0' 'numpy==1.26.4'
```
