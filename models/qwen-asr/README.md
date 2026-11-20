# Qwen3-ASR

Qwen3-ASR provides streaming speech recognition for X-Live. It runs as a
separate GPU service; the runtime consumes partial and final transcripts through
the adapter in `src/xlive/speech/asr/qwen_asr.py`.

The service accepts 16 kHz mono PCM16 over a WebSocket (`start` → audio chunks →
`end`) and returns incremental `partial` results plus one `final` transcript.
It is the only source of transcript text in X-Live; turn completion is decided
separately by Smart Turn.

The serving source and model checkpoint have separate upstream
license terms.

# Qwen3-ASR

Qwen3-ASR 为 X-Live 提供流式语音识别。它作为独立的 GPU 服务运行，X-Live 运行时通过
`src/xlive/speech/asr/qwen_asr.py` 中的适配器接收增量与最终转写。

服务通过 WebSocket 接收 16 kHz 单声道 PCM16（`start` → 音频分片 → `end`），返回
`partial` 增量结果和一份 `final` 转写。它是 X-Live 中唯一的转写来源；话轮是否结束由
Smart Turn 单独判断。

服务源码与模型权重分别遵循各自的
上游许可条款。
