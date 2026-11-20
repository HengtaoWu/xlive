# X-Live Model Assets

## English

X-Live keeps model checkpoints and serving artifacts under `models/`. Download
the checkpoints after cloning the repository.

### Download

Hugging Face is the default source:

```bash
bash scripts/download_models.sh
```

ModelScope mirrors are available for X-Live and Qwen3-ASR. Breeze TTS 2 and
Smart Turn are downloaded from Hugging Face:

```bash
bash scripts/download_models.sh modelscope
```

Downloads resume when possible and complete components are skipped. Use
`--force` to refresh existing files, or check the local layout without network
access:

```bash
bash scripts/download_models.sh --check
```

### Directory layout

| Path under `models/` | Contents |
| --- | --- |
| `x-live-dual/` | Backbone and Dual runtime artifacts |
| `qwen-asr/checkpoint/`, `qwen-asr/service/` | Qwen3-ASR weights and streaming bridge |
| `breeze-tts/checkpoint/`, `breeze-tts/service/` | Breeze TTS 2 weights and server |
| `breeze-tts/reference-zh.wav` | Default reference voice |
| `smart-turn/checkpoint/`, `smart-turn/service/` | Smart Turn ONNX weights and server |
| `environments/` | Conda definitions |

Qwen3-ASR is the authoritative transcript provider. It receives 16 kHz mono
PCM16, produces streaming partial and final transcripts, and runs on a GPU
through its own isolated environment. Smart Turn processes only waveform audio
at browser-VAD pauses and decides whether the current user turn is complete;
it has no transcript channel.

Environment definitions are documented in
[environments/README.md](environments/README.md). Model weights and copied
serving sources retain their upstream license terms.

## 中文

X-Live 将模型权重与服务资源统一放在 `models/` 目录下。请在克隆仓库后下载模型权重。

### 下载

默认从 Hugging Face 下载：

```bash
bash scripts/download_models.sh
```

X-Live 与 Qwen3-ASR 提供 ModelScope 镜像；Breeze TTS 2 与 Smart Turn 从
Hugging Face 下载：

```bash
bash scripts/download_models.sh modelscope
```

下载会尽可能断点续传，并自动跳过已经完整的组件。使用 `--force` 可重新下载已有文件，
也可以在不访问网络的情况下检查本地目录：

```bash
bash scripts/download_models.sh --check
```

### 目录结构

| `models/` 下的路径 | 内容 |
| --- | --- |
| `x-live-dual/` | 骨干模型与 Dual 组件 |
| `qwen-asr/checkpoint/`、`qwen-asr/service/` | ASR 权重与流式服务 |
| `breeze-tts/checkpoint/`、`breeze-tts/service/` | TTS 权重与流式服务 |
| `breeze-tts/reference-zh.wav` | 默认参考音频 |
| `smart-turn/checkpoint/`、`smart-turn/service/` | Smart Turn ONNX 权重与服务 |
| `environments/` | Conda 环境定义 |

Qwen3-ASR 是唯一的转写权威：它接收 16 kHz 单声道 PCM16，输出流式增量与最终
转写，并在独立 GPU 环境中运行。Smart Turn 在浏览器 VAD 检测到停顿后仅处理原始波形，
判断当前用户轮次是否完成；它不接收或产出转写文本。

环境定义详见 [environments/README.md](environments/README.md)。模型权重与引入的
服务源码继续遵循各自的上游许可条款。
