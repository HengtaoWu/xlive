from __future__ import annotations

import base64
import json
from ...log_utils import logger
from typing import Any, TYPE_CHECKING

from starlette.websockets import WebSocketState

if TYPE_CHECKING:
    from fastapi import WebSocket

from ...runtime.event_bus import EventBus
from ...runtime.manager import EventListenerMixin
from ..tts_wire import encode_tts_chunk_frame
from ...runtime.protocol_events import (
    ASRResultPartial,
    ASRResultFinal,
    VADSpeechStart,
    VADSpeechEnd,
    TTSStarted,
    TTSStopped,
    TTSPaused,
    TTSResumed,
    TTSFinished,
    ResponseUpdate,
    ResponseFinish,
    TaskStatusUpdated,
    WhiteboardUpdated,
    ErrorOccurred,
    TTSChunkReady,
    TTSVoiceChange,
    LatencyMetricsUpdated,
    ToolCallOccurred,
    FullAudioFrameReady,
)


class OutputGateway(EventListenerMixin):
    """Forward backend events to the frontend WebSocket."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        websocket: WebSocket,
        config: dict[str, Any] | None = None,
    ):
        self.event_bus = event_bus
        self.session_id = session_id
        self.websocket = websocket
        self.config: dict[str, Any] = config or {}
        self._full_audio_chunk_bytes = 128 * 1024

    def _is_connected(self) -> bool:
        return self.websocket.client_state == WebSocketState.CONNECTED

    def _ws_state_name(self) -> str:
        return self.websocket.client_state.name

    async def send_signal(self, message: dict) -> None:
        """Send a JSON payload to the frontend.

        Parameters
        ----------
        message : dict
            JSON-serializable payload to send over the WebSocket.
        """
        if not self._is_connected():
            logger.warning(
                "WebSocket not connected, skip send - session: %s, state: %s",
                self.session_id,
                self._ws_state_name(),
            )
            return
        try:
            await self.websocket.send_text(json.dumps(message))
        except Exception as e:
            error_msg = str(e).lower()
            if "disconnect" in error_msg or "closed" in error_msg:
                logger.warning(
                    "WebSocket disconnected, cannot send message - session: %s",
                    self.session_id,
                )
            else:
                logger.error(
                    "Failed to send WebSocket message - session: %s, error: %s",
                    self.session_id,
                    e,
                )

    async def _send_binary(self, data: bytes) -> None:
        """Send raw binary data (e.g., audio chunks)."""
        if not self._is_connected():
            logger.warning(
                "WebSocket not connected, skip audio send - session: %s, state: %s",
                self.session_id,
                self._ws_state_name(),
            )
            return
        try:
            await self.websocket.send_bytes(data)
        except Exception as e:
            error_msg = str(e).lower()
            if "disconnect" in error_msg or "closed" in error_msg:
                logger.warning(
                    "WebSocket disconnected, cannot send audio - session: %s",
                    self.session_id,
                )
            else:
                logger.error(
                    "Failed to send binary data - session: %s, error: %s",
                    self.session_id,
                    e,
                )

    async def _forward(self, action: str, data: Any) -> None:
        await self.send_signal({"action": action, "data": data})

    async def _forward_asr(self, action: str, event: ASRResultPartial | ASRResultFinal) -> None:
        """Forward an ASR result (partial or final) to the frontend."""
        display_text = event.display_text or event.text
        await self._forward(action, {"text": display_text})

    async def send_session_attached(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Send the session id and authoritative persisted visible history."""
        payload: dict[str, Any] = {"session_id": self.session_id}
        if messages is not None:
            payload["messages"] = messages
        if tasks is not None:
            payload["tasks"] = tasks
        await self._forward("session_attached", payload)

    @EventListenerMixin.event_handler(ASRResultPartial, priority=5)
    async def _send_update_asr_signal(self, event: ASRResultPartial) -> None:
        await self._forward_asr("update_asr", event)

    @EventListenerMixin.event_handler(ASRResultFinal, priority=5)
    async def _send_finish_asr_signal(self, event: ASRResultFinal) -> None:
        await self._forward_asr("finish_asr", event)

    @EventListenerMixin.event_handler(TTSStarted, priority=5)
    async def _on_start_tts(self, event: TTSStarted) -> None:
        await self._forward(
            "start_tts",
            {
                "generation_id": event.generation_id,
                "utterance_id": event.utterance_id,
            },
        )

    @EventListenerMixin.event_handler(TTSStopped, priority=5)
    async def _send_stop_tts_signal(self, event: TTSStopped) -> None:
        await self._forward(
            "stop_tts",
            {
                "generation_id": event.generation_id,
                "utterance_id": event.utterance_id,
            },
        )

    @EventListenerMixin.event_handler(ErrorOccurred, priority=5)
    async def _send_error_signal(self, event: ErrorOccurred) -> None:
        await self.send_signal({
            "action": "error", "data": event.error_message,
            "error_type": event.error_type,
            "recoverable": event.error_type == "xlive_task_operation_rejected",
        })

    @EventListenerMixin.event_handler(TTSPaused, priority=5)
    async def _send_pause_tts_signal(self, event: TTSPaused) -> None:
        await self._forward("pause_tts", "")

    @EventListenerMixin.event_handler(TTSResumed, priority=5)
    async def _send_resume_tts_signal(self, event: TTSResumed) -> None:
        await self._forward("resume_tts", "")

    @EventListenerMixin.event_handler(ResponseUpdate, priority=5)
    async def _send_update_resp_signal(self, event: ResponseUpdate) -> None:
        await self._forward(
            "update_resp",
            {
                "text": event.text,
                "generation_id": event.generation_id,
                "utterance_id": event.utterance_id,
            },
        )

    @EventListenerMixin.event_handler(ResponseFinish, priority=5)
    async def _send_finish_resp_signal(self, event: ResponseFinish) -> None:
        await self._forward(
            "finish_resp",
            {
                "text": event.text,
                "generation_id": event.generation_id,
                "utterance_id": event.utterance_id,
            },
        )

    @EventListenerMixin.event_handler(TaskStatusUpdated, priority=5)
    async def _send_task_updated_signal(self, event: TaskStatusUpdated) -> None:
        await self._forward(
            "task_updated",
            {
                "task_id": event.task_id,
                "title": event.title,
                "status": event.status,
                "detail": event.detail,
                "backend": event.backend,
                "updates": list(event.updates),
                "updated_at": event.timestamp,
            },
        )

    @EventListenerMixin.event_handler(WhiteboardUpdated, priority=5)
    async def _send_whiteboard_updated_signal(self, event: WhiteboardUpdated) -> None:
        await self._forward(
            "whiteboard_updated",
            {
                "task_id": event.task_id,
                "title": event.title,
                "version": event.version,
                "state": event.state,
                "operations": list(event.operations),
            },
        )

    @EventListenerMixin.event_handler(TTSFinished, priority=5)
    async def _send_tts_finished_signal(self, event: TTSFinished) -> None:
        await self._forward(
            "tts_finished",
            {
                "generation_id": event.generation_id,
                "utterance_id": event.utterance_id,
            },
        )

    @EventListenerMixin.event_handler(TTSChunkReady, priority=5)
    async def _send_tts_chunk_signal(self, event: TTSChunkReady) -> None:
        if not event.audio_chunk:
            return

        identity = (event.generation_id, event.utterance_id, event.chunk_id)
        if any(identity) and not all(identity):
            logger.error(
                "TTS chunk has incomplete output identity - session: %s",
                self.session_id,
            )
            return

        # Chunks without an identity predate the framed wire format; send raw PCM.
        if not all(identity):
            await self._send_binary(event.audio_chunk)
            return

        try:
            payload = encode_tts_chunk_frame(
                event.audio_chunk,
                generation_id=event.generation_id,
                utterance_id=event.utterance_id,
                chunk_id=event.chunk_id,
                sample_rate=event.sample_rate,
            )
        except ValueError as e:
            logger.error(
                "Failed to encode TTS chunk - session: %s, error: %s",
                self.session_id,
                e,
            )
            return
        await self._send_binary(payload)

    @EventListenerMixin.event_handler(FullAudioFrameReady, priority=5)
    async def _send_full_audio_frame_signal(self, event: FullAudioFrameReady) -> None:
        if not event.audio_chunk:
            return
        total_bytes = len(event.audio_chunk)
        for start in range(0, total_bytes, self._full_audio_chunk_bytes):
            chunk = event.audio_chunk[start : start + self._full_audio_chunk_bytes]
            await self._forward(
                "full_audio_frame",
                {
                    "audio_base64": base64.b64encode(chunk).decode("ascii"),
                    "sample_rate": event.sample_rate,
                    "channels": event.channels,
                    "format": event.format,
                },
            )

    @EventListenerMixin.event_handler(TTSVoiceChange, priority=5)
    async def _send_voice_changed_signal(self, event: TTSVoiceChange) -> None:
        await self._forward("voice_changed", {"name": event.voice_name})

    @EventListenerMixin.event_handler(VADSpeechStart, priority=5)
    async def _send_vad_start_signal(self, event: VADSpeechStart) -> None:
        if event.origin == "client":
            return
        await self._forward("vad_speech_start", {"origin": event.origin})

    @EventListenerMixin.event_handler(VADSpeechEnd, priority=5)
    async def _send_vad_end_signal(self, event: VADSpeechEnd) -> None:
        if event.origin == "client":
            return
        await self._forward("vad_speech_end", {"origin": event.origin})

    @EventListenerMixin.event_handler(LatencyMetricsUpdated, priority=5)
    async def _send_latency_metrics_signal(self, event: LatencyMetricsUpdated) -> None:
        await self._forward(
            "latency_metrics",
            {
                "network_latency_ms": event.network_latency_ms,
                "asr_latency_ms": event.asr_latency_ms,
                "llm_first_token_ms": event.llm_first_token_ms,
                "llm_sentence_ms": event.llm_sentence_ms,
                "tts_first_chunk_ms": event.tts_first_chunk_ms,
            },
        )

    @EventListenerMixin.event_handler(ToolCallOccurred, priority=5)
    async def _send_tool_called_signal(self, event: ToolCallOccurred) -> None:
        if event.name in {"tool_call_result", "direct_audio"}:
            return
        await self._forward(
            "tool_called",
            {
                "name": event.name,
                "args": event.args,
                "called_at": event.timestamp,
            },
        )
