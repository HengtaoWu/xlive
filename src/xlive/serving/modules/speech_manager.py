"""X-Live-owned speech ingress on top of X-Talk's WebSocket audio events."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping, cast
from uuid import uuid4

from ...runtime.event_bus import EventBus
from ...runtime.protocol_events import (
    ASRResultFinal,
    ASRResultPartial,
    AudioFrameReceived,
    TTSPlaybackFinished,
    TTSPlaybackStarted,
    TTSStopped,
    VADSpeechEnd,
    VADSpeechStart,
)
from ...runtime.manager import Manager

from ...speech import (
    ASRUpdate,
    StreamingASRAdapter,
    TurnStateAdapter,
    TurnStateUpdate,
    XLiveTurnStateObserved,
    XLiveUserCommitted,
    XLiveUserSnapshotObserved,
    XLiveUserVocalizationDiscarded,
    XLiveSpeechManagerConfig,
)
from ...runtime.output import UtteranceState
from ..composition import get_xlive_core
from ..events import XLiveInterruptionRequested
from ...speech.config import (
    ASR_ADAPTER_CONFIG_KEY,
    SPEECH_MANAGER_CONFIG_KEY,
    TURN_ADAPTER_CONFIG_KEY,
)


class XLiveSpeechManager(Manager):
    """Coordinate acoustic VAD, Qwen3-ASR, and one semantic turn detector.

    Qwen3-ASR owns every transcript snapshot and final commit. The selected
    semantic detector contributes turn state only. Smart Turn evaluates each
    VAD-delimited candidate endpoint, while other turn detectors may publish states during
    the active audio segment.
    """

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.config = config or {}
        self.core = get_xlive_core(self.config)
        self.asr = cast(
            StreamingASRAdapter,
            _require_adapter(self.config, ASR_ADAPTER_CONFIG_KEY),
        )
        self.turn_detector = cast(
            TurnStateAdapter,
            _require_adapter(self.config, TURN_ADAPTER_CONFIG_KEY),
        )
        manager_config = self.config.get(SPEECH_MANAGER_CONFIG_KEY)
        if manager_config is None:
            manager_config = XLiveSpeechManagerConfig()
        if not isinstance(manager_config, XLiveSpeechManagerConfig):
            raise TypeError(
                f"{SPEECH_MANAGER_CONFIG_KEY} must be XLiveSpeechManagerConfig"
            )
        self.speech_config = manager_config

        bytes_per_ms = self.speech_config.sample_rate * 2 // 1_000
        self._asr_chunk_bytes = bytes_per_ms * self.speech_config.asr_chunk_ms
        self._prebuffer_limit = bytes_per_ms * self.speech_config.prebuffer_ms

        self._lock = asyncio.Lock()
        self._prebuffer = bytearray()
        self._asr_buffer = bytearray()
        self._turn_audio = bytearray()
        self._asr_active = False
        self._turn_detector_active = False
        self._frontend_speech_active = False
        self._barge_in_detected = False
        self._bot_speaking_at_turn_start = False
        self._pending_endpoint_task: asyncio.Task[None] | None = None
        self._bot_speaking = False
        self._playing_generation_id: str | None = None
        self._turn_id: str | None = None
        self._revision = 0
        self._recognized_text = ""
        self._audio_end_ms = 0.0

    @Manager.event_handler(AudioFrameReceived, priority=80)
    async def _on_audio_frame(self, event: AudioFrameReceived) -> None:
        if event.sample_rate != self.speech_config.sample_rate:
            raise ValueError(
                f"speech input must be {self.speech_config.sample_rate} Hz, "
                f"got {event.sample_rate}"
            )
        if len(event.audio_data) % 2:
            raise ValueError("speech input must contain complete PCM16 samples")
        if not event.audio_data:
            return
        async with self._lock:
            await self._consume_audio(event.audio_data)

    @Manager.event_handler(VADSpeechStart, priority=80)
    async def _on_vad_speech_start(self, event: VADSpeechStart) -> None:
        if event.origin == "turn_detector":
            return
        async with self._lock:
            if not self._frontend_speech_active:
                self._cancel_pending_endpoint()
            if not self._asr_active:
                self._barge_in_detected = False
                self._bot_speaking_at_turn_start = self._bot_speaking
            self._frontend_speech_active = True
            await self._ensure_asr_started()
            await self._start_turn_detector()
            await self._drain_asr_chunks()

    @Manager.event_handler(VADSpeechEnd, priority=80)
    async def _on_vad_speech_end(self, event: VADSpeechEnd) -> None:
        if event.origin == "turn_detector":
            return
        async with self._lock:
            if not self._frontend_speech_active:
                return
            self._frontend_speech_active = False
            if not self._asr_active:
                return
            if not self._turn_audio:
                await self._discard_turn()
                return
            if not self._turn_detector_active:
                raise RuntimeError("semantic turn detector is not active")
            state = await self.turn_detector.finish()
            self._turn_detector_active = False
            await self._accept_turn_state(state, detector_finished=True)

    @Manager.event_handler(TTSPlaybackStarted, priority=80)
    async def _on_playback_started(self, event: TTSPlaybackStarted) -> None:
        utterance = self.core.output.utterance(event.utterance_id)
        if (
            self.core.output.is_generation_live(event.generation_id)
            and utterance is not None
            and utterance.generation_id == event.generation_id
            and utterance.state is UtteranceState.PLAYING
            and event.chunk_id in utterance.chunk_ids
        ):
            self._bot_speaking = True
            self._playing_generation_id = event.generation_id

    @Manager.event_handler(TTSPlaybackFinished, priority=80)
    async def _on_playback_finished(self, event: TTSPlaybackFinished) -> None:
        utterance = self.core.output.utterance(event.utterance_id)
        if (
            event.generation_id == self._playing_generation_id
            and utterance is not None
            and utterance.generation_id == event.generation_id
            and utterance.state is UtteranceState.PLAYED
        ):
            self._bot_speaking = False

    @Manager.event_handler(TTSStopped, priority=80)
    async def _on_tts_stopped(self, event: TTSStopped) -> None:
        if event.generation_id == self._playing_generation_id:
            self._bot_speaking = False

    async def _consume_audio(self, audio: bytes) -> None:
        self._audio_end_ms += (
            len(audio) / (self.speech_config.sample_rate * 2) * 1_000
        )
        if self._asr_active:
            self._asr_buffer.extend(audio)
        else:
            self._append_prebuffer(audio)

        if self._asr_active:
            self._turn_audio.extend(audio)

        if self._asr_active:
            await self._drain_asr_chunks()
        if self._turn_detector_active:
            state = await self.turn_detector.push_audio(audio)
            if state is not None:
                await self._accept_turn_state(state, detector_finished=False)

    async def _accept_turn_state(
        self,
        state: TurnStateUpdate,
        *,
        detector_finished: bool,
    ) -> None:
        payload: dict[str, Any] = {
            "source": self.turn_detector.provider,
            "controller_state": state.controller_state,
            "audio_end_ms": round(self._audio_end_ms, 3),
            "bot_speaking": self._bot_speaking,
        }
        optional_values = {
            "turn_class": state.turn_class,
            "event": state.event,
            "reason": state.reason,
            "confidence": state.confidence,
            "source_timestamp": state.source_timestamp,
        }
        payload.update(
            {key: value for key, value in optional_values.items() if value is not None}
        )
        envelope = self.core.record_input(
            "turn_state",
            payload,
            turn_id=self._turn_id,
            revision=self._revision if self._turn_id is not None else None,
        )
        await self.event_bus.publish(
            XLiveTurnStateObserved(
                session_id=self.session_id,
                envelope=envelope,
            ),
            wait_for_completion=True,
        )

        if state.event == "backchannel":
            if detector_finished:
                if self._bot_speaking_at_turn_start:
                    await self._discard_turn()
                else:
                    await self._finish_turn(
                        endpoint_source=self.turn_detector.provider
                    )
            return

        if (
            self._bot_speaking_at_turn_start
            and state.event == "accept"
            and not self._barge_in_detected
        ):
            self._barge_in_detected = True
            await self.event_bus.publish(
                XLiveInterruptionRequested(
                    session_id=self.session_id,
                    reason=f"{self.turn_detector.provider}_barge_in",
                    generation_id=self.core.output.current_generation_id or "",
                ),
                wait_for_completion=False,
            )

        if state.controller_state == "speak" and state.event == "accept":
            self._cancel_pending_endpoint()
            if (
                detector_finished
                and self.turn_detector.provider == "smart_turn"
                and self.speech_config.semantic_endpoint_grace_ms > 0
            ):
                # Keep the ASR turn open if speech resumes after a candidate endpoint.
                self._schedule_endpoint(
                    self.speech_config.semantic_endpoint_grace_ms,
                    self.turn_detector.provider,
                )
                return
            if not detector_finished:
                await self.turn_detector.finish()
                self._turn_detector_active = False
            await self._finish_turn(endpoint_source=self.turn_detector.provider)
        elif detector_finished and state.event == "continue":
            self._schedule_endpoint_fallback()

    async def _ensure_asr_started(self) -> None:
        if self._asr_active:
            return
        self._turn_id = str(uuid4())
        self._revision = 0
        self._recognized_text = ""
        await self.asr.start(sample_rate=self.speech_config.sample_rate)
        self._asr_active = True
        if self._prebuffer:
            self._asr_buffer.extend(self._prebuffer)
            self._turn_audio.extend(self._prebuffer)

    async def _start_turn_detector(self) -> None:
        if self._turn_detector_active:
            return
        await self.turn_detector.start(
            bot_speaking=self._bot_speaking_at_turn_start
        )
        self._turn_detector_active = True
        if self._turn_audio:
            state = await self.turn_detector.push_audio(bytes(self._turn_audio))
            if state is not None:
                await self._accept_turn_state(state, detector_finished=False)
        self._prebuffer.clear()

    async def _drain_asr_chunks(self, *, include_tail: bool = False) -> None:
        while len(self._asr_buffer) >= self._asr_chunk_bytes:
            chunk = bytes(self._asr_buffer[: self._asr_chunk_bytes])
            del self._asr_buffer[: self._asr_chunk_bytes]
            update = await self.asr.push_audio(chunk)
            if update is not None:
                await self._accept_asr_update(update)
        if include_tail and self._asr_buffer:
            tail = bytes(self._asr_buffer)
            self._asr_buffer.clear()
            update = await self.asr.push_audio(tail)
            if update is not None:
                await self._accept_asr_update(update)

    async def _accept_asr_update(self, update: ASRUpdate) -> None:
        if self._turn_id is None:
            raise RuntimeError("received Qwen3-ASR output without an active turn")
        text = update.text.strip()
        changed = text != self._recognized_text
        if changed:
            self._revision += 1
            self._recognized_text = text
        if not text:
            return
        if not changed and not update.is_final:
            return

        payload: dict[str, Any] = {
            "source": "qwen_asr",
            "text": text,
            "is_final": update.is_final,
            "audio_end_ms": round(self._audio_end_ms, 3),
        }
        if update.first_partial_latency is not None:
            payload["first_partial_latency"] = update.first_partial_latency
        envelope = self.core.record_input(
            "user_snapshot",
            payload,
            turn_id=self._turn_id,
            revision=self._revision,
        )
        await self.event_bus.publish(
            XLiveUserSnapshotObserved(
                session_id=self.session_id,
                envelope=envelope,
            ),
            wait_for_completion=True,
        )
        if update.is_final:
            await self.event_bus.publish(
                ASRResultFinal(
                    session_id=self.session_id,
                    text=text,
                    display_text=text,
                )
            )
        else:
            await self.event_bus.publish(
                ASRResultPartial(
                    session_id=self.session_id,
                    text=text,
                    display_text=text,
                    speech_pause=False,
                )
            )

    async def _finish_turn(self, *, endpoint_source: str) -> None:
        if not self._asr_active or self._turn_id is None:
            self._reset_turn_state()
            return
        await self._drain_asr_chunks(include_tail=True)
        final = await self.asr.finish()
        await self._accept_asr_update(final)
        if self._recognized_text:
            envelope = self.core.record_input(
                "user_commit",
                {
                    "source": "qwen_asr",
                    "text": self._recognized_text,
                    "audio_end_ms": round(self._audio_end_ms, 3),
                    "endpoint_source": endpoint_source,
                },
                turn_id=self._turn_id,
                revision=self._revision,
            )
            await self.event_bus.publish(
                XLiveUserCommitted(
                    session_id=self.session_id,
                    envelope=envelope,
                ),
                wait_for_completion=True,
            )
        else:
            await self.event_bus.publish(
                XLiveUserVocalizationDiscarded(session_id=self.session_id),
                wait_for_completion=True,
            )
        self._reset_turn_state()

    async def _discard_turn(self) -> None:
        """Close an unaccepted ASR stream without creating visible text."""

        try:
            if self._asr_active:
                await self.asr.finish()
            if self._turn_detector_active:
                await self.turn_detector.aclose()
        finally:
            self._reset_turn_state()
        # Remove any provisional Qwen3-ASR caption and release the interaction
        # manager's user-speech delivery barrier without invoking X-Live.
        await self.event_bus.publish(
            ASRResultPartial(
                session_id=self.session_id,
                text="",
                display_text="",
                speech_pause=False,
            )
        )
        await self.event_bus.publish(
            XLiveUserVocalizationDiscarded(session_id=self.session_id),
            wait_for_completion=True,
        )

    def _append_prebuffer(self, audio: bytes) -> None:
        if self._prebuffer_limit == 0:
            return
        self._prebuffer.extend(audio)
        if len(self._prebuffer) > self._prebuffer_limit:
            del self._prebuffer[: len(self._prebuffer) - self._prebuffer_limit]

    def _reset_turn_state(self) -> None:
        self._cancel_pending_endpoint()
        self._asr_active = False
        self._turn_detector_active = False
        self._barge_in_detected = False
        self._bot_speaking_at_turn_start = False
        self._asr_buffer.clear()
        self._prebuffer.clear()
        self._turn_audio.clear()
        self._turn_id = None
        self._revision = 0
        self._recognized_text = ""

    async def shutdown(self) -> None:
        async with self._lock:
            await asyncio.gather(
                self.asr.aclose(),
                self.turn_detector.aclose(),
                return_exceptions=True,
            )
            self._reset_turn_state()
            self._frontend_speech_active = False

    def _cancel_pending_endpoint(self) -> None:
        task = self._pending_endpoint_task
        self._pending_endpoint_task = None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    def _schedule_endpoint_fallback(self) -> None:
        self._schedule_endpoint(
            self.speech_config.semantic_max_pause_ms,
            f"{self.turn_detector.provider}_max_pause",
        )

    def _schedule_endpoint(self, delay_ms: int, endpoint_source: str) -> None:
        self._cancel_pending_endpoint()
        turn_id = self._turn_id
        if turn_id is None:
            return
        self._pending_endpoint_task = asyncio.create_task(
            self._finish_after_semantic_pause(turn_id, delay_ms, endpoint_source),
            name=f"semantic-endpoint:{self.session_id}",
        )

    async def _finish_after_semantic_pause(
        self, turn_id: str, delay_ms: int, endpoint_source: str
    ) -> None:
        try:
            await asyncio.sleep(delay_ms / 1_000)
            async with self._lock:
                if (
                    self._turn_id == turn_id
                    and self._asr_active
                    and not self._frontend_speech_active
                ):
                    await self._finish_turn(endpoint_source=endpoint_source)
        except asyncio.CancelledError:
            return


def _require_adapter(
    config: Mapping[str, Any],
    key: str,
) -> Any:
    adapter = config.get(key)
    if adapter is None:
        raise RuntimeError(f"X-Talk service is missing {key}")
    return adapter
