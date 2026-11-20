"""X-Live-owned bridge from accepted model text to X-Talk TTS events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ...runtime.event_bus import EventBus
from ...runtime.protocol_events import (
    BaseEvent,
    ResponseFinish,
    ResponseUpdate,
    TTSChunkReady,
    TTSChunkPlayed,
    TTSPlaybackFinished,
    TTSPlaybackStarted,
    TurnTTSFlushRequested,
    TurnTTSStartRequested,
    TurnTTSStopRequested,
    TurnTTSTextAppendRequested,
)
from ...runtime.manager import Manager

from ..composition import get_xlive_core
from ...runtime.output import GenerationState, UtteranceState
from ...runtime.session import RealtimeSessionCore


@dataclass
class XLiveTextStartRequested(BaseEvent):
    """Expose one text delta, then start its spoken utterance."""

    TYPE: ClassVar[str] = "xlive.output.text_start_requested"
    generation_id: str = ""
    text: str = ""
    utterance_id: str = ""
    include_in_history: bool = True


@dataclass
class XLiveTextAppendRequested(BaseEvent):
    """Append an accepted model text delta to an existing utterance."""

    TYPE: ClassVar[str] = "xlive.output.text_append_requested"
    generation_id: str = ""
    utterance_id: str = ""
    text: str = ""


@dataclass
class XLiveTextFlushRequested(BaseEvent):
    """Finish sending text for one utterance to TTS."""

    TYPE: ClassVar[str] = "xlive.output.text_flush_requested"
    generation_id: str = ""
    utterance_id: str = ""


@dataclass
class XLiveGenerationCancelRequested(BaseEvent):
    """Cancel model output and TTS through the single X-Live authority."""

    TYPE: ClassVar[str] = "xlive.output.generation_cancel_requested"
    generation_id: str = ""
    utterance_id: str = ""
    reason: str = ""


@dataclass
class XLiveGenerationSealRequested(BaseEvent):
    """Declare that a generation will not produce another utterance."""

    TYPE: ClassVar[str] = "xlive.output.generation_seal_requested"
    generation_id: str = ""


class XLiveOutputBridge(Manager):
    """Translate X-Live output lifecycle requests into tagged X-Talk TTS events."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.core: RealtimeSessionCore = get_xlive_core(config or {})
        self._responses: dict[str, str] = {}
        self._finished_responses: set[str] = set()
        self._audible_generations: set[str] = set()
        self._stopped_generation_id: str | None = None
        self._recorded_playback_starts: set[tuple[str, str, str]] = set()
        self._recorded_ready_chunks: set[tuple[str, str, str]] = set()
        self._recorded_played_chunks: set[tuple[str, str, str]] = set()
        self._recorded_playback_finishes: set[tuple[str, str]] = set()

    @Manager.event_handler(XLiveTextStartRequested, priority=110)
    async def _handle_text_start(self, event: XLiveTextStartRequested) -> None:
        text = event.text
        if not text:
            raise ValueError("X-Live text start requires a non-empty first delta")
        utterance = self.core.begin_utterance(
            event.generation_id,
            text,
            utterance_id=event.utterance_id or None,
            include_in_history=event.include_in_history,
        )
        await self._publish_generated_response(event.generation_id)
        identity = {
            "generation_id": event.generation_id,
            "utterance_id": utterance.utterance_id,
        }
        await self.event_bus.publish(
            TurnTTSStartRequested(session_id=self.session_id, **identity),
            wait_for_completion=True,
        )
        await self.event_bus.publish(
            TurnTTSTextAppendRequested(
                session_id=self.session_id,
                text=text,
                **identity,
            ),
            wait_for_completion=True,
        )

    @Manager.event_handler(XLiveTextAppendRequested, priority=110)
    async def _handle_text_append(self, event: XLiveTextAppendRequested) -> None:
        if not self.core.append_utterance_text(
            event.generation_id, event.utterance_id, event.text
        ):
            return
        await self._publish_generated_response(event.generation_id)
        await self.event_bus.publish(
            TurnTTSTextAppendRequested(
                session_id=self.session_id,
                generation_id=event.generation_id,
                utterance_id=event.utterance_id,
                text=event.text,
            ),
            wait_for_completion=True,
        )

    @Manager.event_handler(XLiveTextFlushRequested, priority=110)
    async def _handle_text_flush(self, event: XLiveTextFlushRequested) -> None:
        utterance = self.core.output.utterance(event.utterance_id)
        if (
            utterance is None
            or utterance.generation_id != event.generation_id
            or not self.core.output.is_generation_live(event.generation_id)
        ):
            return
        await self.event_bus.publish(
            TurnTTSFlushRequested(
                session_id=self.session_id,
                generation_id=event.generation_id,
                utterance_id=event.utterance_id,
            ),
            wait_for_completion=True,
        )

    @Manager.event_handler(XLiveGenerationCancelRequested, priority=110)
    async def _handle_generation_cancel(
        self, event: XLiveGenerationCancelRequested
    ) -> None:
        generation = self.core.output.generation(event.generation_id)
        if generation is None:
            return
        cancelled = self.core.cancel_generation(
            event.generation_id,
            reason=event.reason or "cancelled",
        )
        generation = self.core.output.generation(event.generation_id)
        if (
            not cancelled
            and generation is not None
            and generation.state is not GenerationState.CANCELLED
        ):
            return
        if self._stopped_generation_id == event.generation_id:
            return
        self._stopped_generation_id = event.generation_id
        await self.event_bus.publish(
            TurnTTSStopRequested(
                session_id=self.session_id,
                generation_id=event.generation_id,
                utterance_id=event.utterance_id,
                reason=event.reason,
            ),
            wait_for_completion=True,
        )
        self._audible_generations.discard(event.generation_id)

    @Manager.event_handler(XLiveGenerationSealRequested, priority=110)
    async def _handle_generation_seal(
        self, event: XLiveGenerationSealRequested
    ) -> None:
        generation = self.core.output.generation(event.generation_id)
        if generation is None:
            return
        if generation.state is GenerationState.ACTIVE:
            self.core.seal_generation(event.generation_id)
        elif generation.state not in (
            GenerationState.SEALED,
            GenerationState.COMPLETED,
        ):
            return
        await self._publish_generated_response(event.generation_id, final=True)

    @Manager.event_handler(TTSPlaybackFinished, priority=10)
    async def _handle_playback_finished(self, event: TTSPlaybackFinished) -> None:
        utterance = self.core.output.utterance(event.utterance_id)
        identity = (event.generation_id, event.utterance_id)
        if (
            utterance is not None
            and utterance.generation_id == event.generation_id
            and utterance.state is UtteranceState.PLAYED
            and identity not in self._recorded_playback_finishes
        ):
            self._recorded_playback_finishes.add(identity)
            self.core.events.append(
                "utterance_playback_finished",
                {},
                generation_id=event.generation_id,
                utterance_id=event.utterance_id,
                created_at=event.timestamp,
            )

    @Manager.event_handler(TTSPlaybackStarted, priority=10)
    async def _handle_playback_started(self, event: TTSPlaybackStarted) -> None:
        utterance = self.core.output.utterance(event.utterance_id)
        identity = (event.generation_id, event.utterance_id, event.chunk_id)
        if (
            utterance is None
            or utterance.generation_id != event.generation_id
            or utterance.state is not UtteranceState.PLAYING
            or event.chunk_id not in utterance.chunk_ids
            or identity in self._recorded_playback_starts
        ):
            return
        self._recorded_playback_starts.add(identity)
        self.core.events.append(
            "utterance_playback_started",
            {"chunk_id": event.chunk_id},
            generation_id=event.generation_id,
            utterance_id=event.utterance_id,
            created_at=event.timestamp,
        )

    @Manager.event_handler(TTSChunkReady, priority=10)
    async def _handle_chunk_ready(self, event: TTSChunkReady) -> None:
        utterance = self.core.output.utterance(event.utterance_id)
        identity = (event.generation_id, event.utterance_id, event.chunk_id)
        if (
            utterance is None
            or utterance.generation_id != event.generation_id
            or event.chunk_id not in utterance.chunk_ids
            or identity in self._recorded_ready_chunks
        ):
            return
        self._recorded_ready_chunks.add(identity)
        if event.generation_id not in self._audible_generations:
            self._audible_generations.add(event.generation_id)
            generation = self.core.output.generation(event.generation_id)
            await self._publish_generated_response(
                event.generation_id,
                final=(
                    generation is not None
                    and generation.state
                    in (GenerationState.SEALED, GenerationState.COMPLETED)
                ),
            )
        self.core.events.append(
            "utterance_audio_chunk_ready",
            {
                "chunk_id": event.chunk_id,
                "sample_rate": event.sample_rate,
            },
            generation_id=event.generation_id,
            utterance_id=event.utterance_id,
            created_at=event.timestamp,
        )

    @Manager.event_handler(TTSChunkPlayed, priority=10)
    async def _handle_chunk_played(self, event: TTSChunkPlayed) -> None:
        utterance = self.core.output.utterance(event.utterance_id)
        identity = (event.generation_id, event.utterance_id, event.chunk_id)
        if (
            utterance is None
            or utterance.generation_id != event.generation_id
            or event.chunk_id not in utterance.played_chunk_ids
            or identity in self._recorded_played_chunks
        ):
            return
        self._recorded_played_chunks.add(identity)
        self.core.events.append(
            "utterance_chunk_played",
            {"chunk_id": event.chunk_id},
            generation_id=event.generation_id,
            utterance_id=event.utterance_id,
            created_at=event.timestamp,
        )

    async def _publish_generated_response(
        self, generation_id: str, *, final: bool = False
    ) -> None:
        """Project accepted model text after its first audio is ready."""

        generation = self.core.output.generation(generation_id)
        if (
            generation is None
            or generation_id not in self._audible_generations
            or generation.state is GenerationState.CANCELLED
        ):
            return

        generated_parts: list[str] = []
        for utterance_id in generation.utterance_ids:
            utterance = self.core.output.utterance(utterance_id)
            if utterance is not None and utterance.include_in_history:
                generated_parts.append(utterance.text)
        generated_text = "".join(generated_parts)
        utterance_id = generation.utterance_ids[-1] if generation.utterance_ids else ""

        if generated_text and generated_text != self._responses.get(generation_id):
            self._responses[generation_id] = generated_text
            await self.event_bus.publish(
                ResponseUpdate(
                    session_id=self.session_id,
                    text=generated_text,
                    generation_id=generation_id,
                    utterance_id=utterance_id,
                ),
                wait_for_completion=True,
            )

        if (
            final
            and generation_id not in self._finished_responses
            and generated_text
        ):
            self._finished_responses.add(generation_id)
            await self.event_bus.publish(
                ResponseFinish(
                    session_id=self.session_id,
                    text=generated_text,
                    generation_id=generation_id,
                    utterance_id=utterance_id,
                ),
                wait_for_completion=True,
            )

    async def shutdown(self) -> None:
        current_id = self.core.output.current_generation_id
        if current_id is not None:
            self.core.cancel_generation(current_id, reason="session_stopped")
