# Smart Turn v3.2

Smart Turn provides semantic endpointing for X-Live. After the browser VAD
detects a pause, the runtime sends the recent waveform to this service to decide
whether the user's turn is complete. It does not receive or produce transcripts.

Inputs retain the last eight seconds of the current turn. Shorter inputs are
zero-padded at the beginning, so the real speech endpoint remains at the end of
the model window, following the [upstream input specification](https://github.com/pipecat-ai/smart-turn#notes-on-input-format).
The runtime retains a candidate endpoint for 500 ms before committing ASR
(`XLiveSpeechManagerConfig.semantic_endpoint_grace_ms`). Resumed speech cancels
that endpoint and continues the same ASR turn. This adds up to 500 ms to endpoint
confirmation.

The service exposes a small WebSocket protocol (`classify` with 16 kHz PCM16;
the response is `speak` or `nonidle` with a confidence score) and listens on
port 8005 by default. X-Live uses the model through
`src/xlive/speech/turn_detector/smart_turn.py`.

The serving source and model checkpoint have separate upstream
license terms.

# Smart Turn v3.2

Smart Turn 为 X-Live 提供语义端点检测。浏览器 VAD 检测到停顿时，运行时会把最近的
音频波形发送给该服务，由它判断当前用户轮次是否已经结束。它不接收也不产出转写文本。

输入保留当前轮次的最后八秒；不足八秒时在开头补零，保持真实话尾位于窗口末端。
运行时对候选端点保留 500ms 确认时间，用户继续说话时取消提交并延续同一个 ASR
轮次。该配置为 `XLiveSpeechManagerConfig.semantic_endpoint_grace_ms`，会增加
最多 500ms 的端点确认等待。

服务提供一个小型 WebSocket 协议（`classify` 携带 16 kHz PCM16，响应为 `speak` 或
`nonidle` 并附带置信度），默认监听 8005 端口。X-Live 通过
`src/xlive/speech/turn_detector/smart_turn.py` 中的适配器调用该服务。

服务源码与模型权重分别遵循各自的
上游许可条款。
