# Breeze TTS 2

Breeze TTS 2 provides streaming speech synthesis for X-Live. It runs as a
separate CUDA service; the X-Live runtime consumes its PCM stream through the
adapter in `src/xlive/speech/tts/breeze.py`.

X-Live ships two default reference voice for reproducible demos while allowing it
to be replaced at runtime. The serving source and model checkpoint have
separate upstream license terms.

# Breeze TTS 2

Breeze TTS 2 为 X-Live 提供流式语音合成。它作为独立的 CUDA 服务运行，X-Live
运行时通过 `src/xlive/speech/tts/breeze.py` 中的适配器接收其 PCM 音频流。

X-Live 提供两份默认参考音频，便于复现实验，同时也支持在运行时替换。服务源码与模型权重分别遵循各自的上游许可条款。
