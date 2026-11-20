"""Deterministic scheduling between accepted inputs and the frontend model."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import replace
from typing import Any, cast
from uuid import uuid4

from ...runtime.event_bus import EventBus
from ...runtime.protocol_events import (
    ErrorOccurred,
    SessionReady,
    TaskStatusUpdated,
    TaskCancelRequested,
    WhiteboardUpdated,
    TTSChunkReady,
    TTSPlaybackFinished,
    TTSPlaybackStarted,
    VADSpeechStart,
)
from ...runtime.manager import Manager

from ...agent import InvalidTaskOperation, SessionTaskHub, TaskBackendEvent
from ...frontend_model import FrontendModelAdapter
from ...runtime.events import EventEnvelope
from ...runtime.output import GenerationState
from ...runtime.session import RealtimeSessionCore
from ...speech.events import (
    XLiveUserCommitted,
    XLiveUserVocalizationDiscarded,
)
from ..composition import (
    LESSON_PREFETCH_DELAY_CONFIG_KEY,
    TASK_HUB_CONFIG_KEY,
    XLIVE_MODEL_ADAPTER_CONFIG_KEY,
    get_xlive_core,
)
from ..events import XLiveInterruptionRequested, XLiveTaskBackendEventReceived
from .model_runner import XLiveModelRunner
from .output_bridge import XLiveGenerationCancelRequested


logger = logging.getLogger(__name__)


class XLiveInteractionManager(Manager):
    """Keep exactly one replaceable frontend-model run per live session.

    Each final user commit and every accepted Agent backend event starts one
    replaceable X-Live frontend model run. Transcript snapshots and raw turn-detector states
    remain perception data and never invoke the frontend model.
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
        self.core: RealtimeSessionCore = get_xlive_core(self.config)
        model = self.config.get(XLIVE_MODEL_ADAPTER_CONFIG_KEY)
        if model is None:
            raise RuntimeError(
                f"X-Talk service is missing {XLIVE_MODEL_ADAPTER_CONFIG_KEY}"
            )
        self.model = cast(FrontendModelAdapter, model)
        configured_hub = self.config.get(TASK_HUB_CONFIG_KEY)
        self._task_hub = (
            configured_hub if isinstance(configured_hub, SessionTaskHub) else None
        )
        if self._task_hub is not None:
            persistence = self.config.get("persistence_store")
            user_id = self.config.get("user_id")
            if persistence is not None and isinstance(user_id, str) and user_id:
                self._task_hub.attach_persistence(persistence, user_id)
        self.runner = XLiveModelRunner(
            self.event_bus,
            self.session_id,
            self.core,
            self.model,
            task_hub=self._task_hub,
        )
        self._replace_lock = asyncio.Lock()
        self._active_task: asyncio.Task[None] | None = None
        self._active_generation_id: str | None = None
        self._pending_backend_runs: deque[EventEnvelope] = deque()
        self._backend_deliveries: dict[str, EventEnvelope] = {}
        self._pending_backend_events: deque[TaskBackendEvent] = deque()
        self._user_speaking = False
        self._user_generation_id: str | None = None
        self._active_lesson_task_id: str | None = None
        self._active_lesson_version: int | None = None
        self._pending_lesson_task_id: str | None = None
        self._pending_lesson_version: int | None = None
        self._interrupted_lesson_task_id: str | None = None
        self._lesson_resume_versions: dict[str, int] = {}
        self._lesson_resume_generations: set[str] = set()
        self._lesson_playback_finished: set[str] = set()
        self._lesson_playback_started: set[str] = set()
        self._lesson_prefetch_delay_seconds = float(
            self.config.get(LESSON_PREFETCH_DELAY_CONFIG_KEY, 5.0)
        )
        self._lesson_prefetch_timers: dict[str, asyncio.Task[None]] = {}
        self._lesson_prefetch_sources: dict[str, int] = {}
        self._lesson_prefetch_events: dict[tuple[str, int], deque[TaskBackendEvent]] = (
            {}
        )
        self._lesson_board_ready: dict[tuple[str, int], bool] = {}
        self._pending_lesson_boards: dict[tuple[str, int], WhiteboardUpdated] = {}
        self._lesson_speech_finished: set[tuple[str, int]] = set()
        self._ui_cancelled_task_ids: set[str] = set()
        self._model_whiteboard_versions: dict[str, int] = {}
        self._greeting_started = False
        self._closed = False
        self._backend_events: asyncio.Queue[TaskBackendEvent] = asyncio.Queue(
            maxsize=self.core.agent_tools.max_events_per_task
        )
        self._backend_event_pump: asyncio.Task[None] | None = None
        event_sink_setter = getattr(
            self.core.agent_tools.backend,
            "set_event_sink",
            None,
        )
        self._event_sink_setter = (
            event_sink_setter if callable(event_sink_setter) else None
        )
        self._hub_subscription: str | None = None
        if self._task_hub is not None:
            self._hub_subscription = self._task_hub.subscribe(self.submit_task_event)
        elif self._event_sink_setter is not None:
            self._event_sink_setter(self.submit_task_event)

    @Manager.event_handler(SessionReady, priority=60)
    async def _on_session_ready(self, event: SessionReady) -> None:
        """Let the model greet once after a blank browser session is ready."""

        if not event.request_greeting:
            return
        async with self._replace_lock:
            if self._closed or self._greeting_started or self._user_speaking:
                return
            self._greeting_started = True
            envelope = self.core.record_input("session_start", {"language": event.language})
            await self._cancel_active_locked("session_greeting")
            generation_id = str(uuid4())
            self._active_generation_id = generation_id
            self._active_task = asyncio.create_task(
                self._run(envelope, generation_id),
                name=f"xlive-model-{self.session_id}-{generation_id}",
            )

    @Manager.event_handler(VADSpeechStart, priority=130)
    async def _on_user_speech_start(self, event: VADSpeechStart) -> None:
        """Freeze backend delivery as soon as the browser detects speech."""

        if event.origin == "turn_detector":
            return
        async with self._replace_lock:
            if not self._closed:
                self._user_speaking = True

    @Manager.event_handler(XLiveUserVocalizationDiscarded, priority=130)
    async def _on_user_vocalization_discarded(
        self, _event: XLiveUserVocalizationDiscarded
    ) -> None:
        prefetched_events: list[TaskBackendEvent] = []
        async with self._replace_lock:
            if self._closed:
                return
            self._user_speaking = False
            pending_backend_events = self._take_pending_backend_events_locked()
            lesson_task_id = self._pending_lesson_task_id
            lesson_version = self._pending_lesson_version
            self._pending_lesson_task_id = None
            self._pending_lesson_version = None
            if lesson_task_id:
                lesson_task_id, prefetched_events = self._finish_lesson_step_locked(
                    lesson_task_id, lesson_version
                )
            self._start_next_backend_run_locked()
        for backend_event in pending_backend_events:
            await self._deliver_backend_event_to_model(backend_event)
        for backend_event in prefetched_events:
            await self._handle_accepted_backend_event(backend_event)
        if lesson_task_id:
            await self._advance_lesson(lesson_task_id)

    @Manager.event_handler(XLiveUserCommitted, priority=60)
    async def _on_user_commit(self, event: XLiveUserCommitted) -> None:
        await self._capture_interrupted_lesson()
        await self._replace_run(event.envelope, user_commit=True)

    @Manager.event_handler(XLiveTaskBackendEventReceived, priority=60)
    async def _on_task_backend_event(
        self, event: XLiveTaskBackendEventReceived
    ) -> None:
        backend_event = event.backend_event
        if backend_event is None:
            raise ValueError("task backend event is required")
        async with self._replace_lock:
            if self._closed:
                return

        # A shared Task Hub has already accepted, persisted, and applied this
        # event before notifying us. The direct-backend path applies it here.
        if self._task_hub is None:
            accepted = await self.core.agent_tools.apply_backend_event(backend_event)
            if not accepted:
                return

        if await self._stage_prefetched_backend_event(backend_event):
            return
        await self._handle_accepted_backend_event(backend_event)

    async def _stage_prefetched_backend_event(
        self, backend_event: TaskBackendEvent
    ) -> bool:
        """Hold one prepared lesson step behind the current playback."""

        async with self._replace_lock:
            source_version = self._lesson_prefetch_sources.get(backend_event.task_id)
            if source_version is None or backend_event.version != source_version + 1:
                return False
            key = (backend_event.task_id, backend_event.version)
            self._lesson_prefetch_events.setdefault(key, deque()).append(backend_event)
            return True

    async def _handle_accepted_backend_event(
        self, backend_event: TaskBackendEvent
    ) -> None:
        if backend_event.kind.value == "activity":
            return
        # Wait for the step whiteboard; preparation events only update task status.
        if backend_event.narrate and backend_event.whiteboard is None:
            await self._publish_task_status(backend_event)
            return
        if backend_event.whiteboard is not None:
            backend_event = replace(backend_event, narrate=True)
        if backend_event.kind.value in {"cancelled", "failed"}:
            self._pending_lesson_boards = {
                key: board for key, board in self._pending_lesson_boards.items()
                if key[0] != backend_event.task_id
            }
        ui_updates = [asyncio.create_task(self._publish_task_status(backend_event))]

        lesson_to_advance: str | None = None
        if backend_event.whiteboard is not None:
            whiteboard = backend_event.whiteboard
            self._pending_lesson_boards[(backend_event.task_id, backend_event.version)] = WhiteboardUpdated(
                    session_id=self.session_id,
                    task_id=backend_event.task_id,
                    title=whiteboard.title or backend_event.title,
                    version=whiteboard.revision,
                    state=whiteboard.state,
                    operations=list(whiteboard.operations),
            )
            self._model_whiteboard_versions[backend_event.task_id] = max(
                backend_event.version,
                self._model_whiteboard_versions.get(backend_event.task_id, 0),
            )
            if backend_event.kind.value == "progress":
                async with self._replace_lock:
                    key = (backend_event.task_id, backend_event.version)
                    self._lesson_board_ready[key] = whiteboard.auto_continue
                    if (
                        whiteboard.auto_continue
                        and self._active_generation_id in self._lesson_playback_started
                        and self._active_lesson_task_id == backend_event.task_id
                        and self._active_lesson_version == backend_event.version
                    ):
                        self._schedule_lesson_prefetch_locked(
                            backend_event.task_id,
                            backend_event.version,
                            self._active_generation_id,
                        )
                    if key in self._lesson_speech_finished:
                        self._lesson_speech_finished.discard(key)
                        should_advance = self._lesson_board_ready.pop(key)
                        if should_advance:
                            lesson_to_advance = backend_event.task_id
        if (
            backend_event.task_id in self._ui_cancelled_task_ids
            and backend_event.kind.value == "cancelled"
        ):
            self._ui_cancelled_task_ids.discard(backend_event.task_id)
            await asyncio.gather(*ui_updates)
            return
        # Narrate each step once; progress and completion events only update status.
        status_only = (
            backend_event.kind.value == "progress"
            and backend_event.whiteboard is None
            and not backend_event.narrate
        )
        whiteboard_already_narrated = (
            backend_event.kind.value == "completed"
            and self._model_whiteboard_versions.get(backend_event.task_id, 0)
            >= backend_event.version
        )
        if status_only or whiteboard_already_narrated:
            await asyncio.gather(*ui_updates)
            if lesson_to_advance:
                await self._advance_lesson(lesson_to_advance)
            return
        closed = False
        frozen = False
        async with self._replace_lock:
            if self._closed:
                closed = True
            elif self._backend_delivery_frozen_locked():
                self._pending_backend_events.append(backend_event)
                frozen = True
        if closed:
            await asyncio.gather(*ui_updates, return_exceptions=True)
            return
        if frozen:
            await asyncio.gather(*ui_updates)
            return
        await asyncio.gather(
            self._deliver_backend_event_to_model(backend_event),
            *ui_updates,
        )

    async def _publish_task_status(self, backend_event: TaskBackendEvent) -> None:
        """Expose backend truth to the task UI without invoking the LLM."""

        runtime = (
            self._task_hub.runtime
            if self._task_hub is not None
            else self.core.agent_tools
        )
        record = await runtime.task(backend_event.task_id)
        await self.event_bus.publish(
            TaskStatusUpdated(
                session_id=self.session_id,
                task_id=backend_event.task_id,
                title=(record.title if record is not None else backend_event.title),
                status=(
                    "running"
                    if backend_event.kind.value == "progress"
                    else backend_event.kind.value
                ),
                detail=backend_event.detail,
                backend="Codex",
                updates=[] if record is None else list(record.updates),
            )
        )

    async def _deliver_backend_event_to_model(
        self, backend_event: TaskBackendEvent
    ) -> None:
        """Create the model-visible event only after the user-turn barrier."""

        envelope = self.core.record_applied_task_event(backend_event)
        await self._queue_backend_run(envelope)

    @Manager.event_handler(TaskCancelRequested, priority=100)
    async def _on_task_cancel_requested(self, event: TaskCancelRequested) -> None:
        """Honor a direct whiteboard close without asking the LLM to infer intent."""

        runtime = (
            self._task_hub.runtime
            if self._task_hub is not None
            else self.core.agent_tools
        )
        record = await runtime.task(event.task_id)
        if record is None or record.status.value != "running":
            return
        async with self._replace_lock:
            self._ui_cancelled_task_ids.add(event.task_id)
            self._pending_backend_runs = deque(
                item
                for item in self._pending_backend_runs
                if item.task_id != event.task_id
            )
            self._pending_backend_events = deque(
                item
                for item in self._pending_backend_events
                if item.task_id != event.task_id
            )
            self._lesson_resume_versions.pop(event.task_id, None)
            if self._interrupted_lesson_task_id == event.task_id:
                self._interrupted_lesson_task_id = None
            if self._pending_lesson_task_id == event.task_id:
                self._pending_lesson_task_id = None
                self._pending_lesson_version = None
            self._clear_lesson_barriers_locked(event.task_id)
            if self._active_lesson_task_id == event.task_id:
                await self._cancel_active_locked("whiteboard_closed")
        await runtime.cancel(event.task_id)
        if self._task_hub is not None:
            await self._task_hub.persist_task(event.task_id)

    @Manager.event_handler(TTSPlaybackStarted, priority=0)
    async def _on_playback_started(self, event: TTSPlaybackStarted) -> None:
        """Begin preparing one following lesson step behind live playback."""

        async with self._replace_lock:
            if event.generation_id != self._active_generation_id:
                return
            task_id = self._active_lesson_task_id
            version = self._active_lesson_version
            if task_id is None or version is None:
                return
            self._lesson_playback_started.add(event.generation_id)
            if self._lesson_board_ready.get((task_id, version)) is True:
                self._schedule_lesson_prefetch_locked(
                    task_id, version, event.generation_id
                )

    @Manager.event_handler(TTSChunkReady, priority=100)
    async def _show_board_with_audio(self, event: TTSChunkReady) -> None:
        # Send the step whiteboard before its first audio chunk.
        async with self._replace_lock:
            if event.generation_id != self._active_generation_id:
                return
            generation = self.core.output.generation(event.generation_id)
            if generation is None or generation.state == GenerationState.CANCELLED:
                return
            key = (self._active_lesson_task_id, self._active_lesson_version)
            board = self._pending_lesson_boards.pop(key, None)
        if board is not None:
            await self.event_bus.publish(board, wait_for_completion=True)

    def _schedule_lesson_prefetch_locked(
        self, task_id: str, version: int, generation_id: str | None
    ) -> None:
        if (
            generation_id is None
            or task_id in self._lesson_prefetch_sources
            or task_id in self._lesson_prefetch_timers
        ):
            return
        timer = asyncio.create_task(
            self._prefetch_lesson_after_delay(task_id, version, generation_id),
            name=f"xlive-lesson-prefetch-{task_id}-{version}",
        )
        self._lesson_prefetch_timers[task_id] = timer
        timer.add_done_callback(
            lambda done, current_task_id=task_id: self._forget_prefetch_timer(
                current_task_id, done
            )
        )

    def _forget_prefetch_timer(self, task_id: str, timer: asyncio.Task[None]) -> None:
        if self._lesson_prefetch_timers.get(task_id) is timer:
            self._lesson_prefetch_timers.pop(task_id, None)

    async def _prefetch_lesson_after_delay(
        self, task_id: str, version: int, generation_id: str
    ) -> None:
        try:
            await asyncio.sleep(self._lesson_prefetch_delay_seconds)
            async with self._replace_lock:
                if (
                    self._closed
                    or self._user_speaking
                    or self._active_generation_id != generation_id
                    or self._active_lesson_task_id != task_id
                    or self._active_lesson_version != version
                    or self._lesson_board_ready.get((task_id, version)) is not True
                ):
                    return
                self._lesson_prefetch_sources[task_id] = version
            runtime = (
                self._task_hub.runtime
                if self._task_hub is not None
                else self.core.agent_tools
            )
            record = await runtime.task(task_id)
            if (
                record is None
                or record.status.value != "running"
                or record.version != version
            ):
                async with self._replace_lock:
                    if self._lesson_prefetch_sources.get(task_id) == version:
                        self._lesson_prefetch_sources.pop(task_id, None)
                return
            await runtime.update(
                task_id,
                self._next_lesson_step_instruction(prefetched=True),
                record_update=False,
            )
            if self._task_hub is not None:
                await self._task_hub.persist_task(task_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._replace_lock:
                if self._lesson_prefetch_sources.get(task_id) == version:
                    self._discard_lesson_prefetch_locked(task_id)
            logger.exception("Could not prefetch whiteboard lesson %s", task_id)

    @staticmethod
    def _next_lesson_step_instruction(*, prefetched: bool) -> str:
        timing = (
            "The current whiteboard step is already playing; prepare the following "
            "step now, but X-Live will withhold it until playback finishes. "
            if prefetched
            else "The previous whiteboard step has finished playing. "
        )
        return (
            timing + "Write exactly one next planned lesson step with concise factual "
            "teaching context in narration. Advance to the next section; do not "
            "elaborate the current section unless a user update explicitly requested "
            "it. Keep spoken guidance out of board operations. If the lesson is "
            "complete, finish it instead of writing another step."
        )

    def _release_lesson_prefetch_locked(
        self, task_id: str, source_version: int
    ) -> list[TaskBackendEvent] | None:
        if self._lesson_prefetch_sources.get(task_id) != source_version:
            return None
        self._lesson_prefetch_sources.pop(task_id, None)
        self._lesson_prefetch_timers.pop(task_id, None)
        key = (task_id, source_version + 1)
        return list(self._lesson_prefetch_events.pop(key, ()))

    def _discard_lesson_prefetch_locked(self, task_id: str) -> None:
        source_version = self._lesson_prefetch_sources.pop(task_id, None)
        timer = self._lesson_prefetch_timers.pop(task_id, None)
        if source_version is None and timer is not None and not timer.done():
            timer.cancel()
        if source_version is not None:
            self._lesson_prefetch_events.pop((task_id, source_version + 1), None)

    @Manager.event_handler(TTSPlaybackFinished, priority=0)
    async def _on_playback_finished(self, event: TTSPlaybackFinished) -> None:
        """Release a prepared lesson step only after current speech is heard."""

        lesson_task_id: str | None = None
        prefetched_events: list[TaskBackendEvent] = []
        async with self._replace_lock:
            generation = self.core.output.generation(event.generation_id)
            if generation is not None and generation.state is GenerationState.COMPLETED:
                self._backend_deliveries.pop(event.generation_id, None)
                await self.model.record_playback_outcome(event.generation_id, completed=True, started=True)
            if event.generation_id != self._active_generation_id:
                self._start_next_backend_run_locked()
                return
            generation = self.core.output.generation(event.generation_id)
            if generation is None:
                return
            if generation.state is not GenerationState.COMPLETED:
                # Playback can finish just before the model stream is sealed.
                self._lesson_playback_finished.add(event.generation_id)
                return
            lesson_task_id = self._active_lesson_task_id
            lesson_version = self._active_lesson_version
            resume_generation = event.generation_id in self._lesson_resume_generations
            self._lesson_resume_generations.discard(event.generation_id)
            self._active_lesson_task_id = None
            self._active_lesson_version = None
            self._active_generation_id = None
            self._lesson_playback_started.discard(event.generation_id)
            if lesson_task_id and self._user_speaking:
                self._pending_lesson_task_id = lesson_task_id
                self._pending_lesson_version = lesson_version
                lesson_task_id = None
            elif lesson_task_id and resume_generation:
                lesson_task_id = self._mark_lesson_speech_finished_locked(
                    lesson_task_id, lesson_version
                )
            elif lesson_task_id:
                lesson_task_id, prefetched_events = self._finish_lesson_step_locked(
                    lesson_task_id, lesson_version
                )
            self._start_next_backend_run_locked()
        for backend_event in prefetched_events:
            await self._handle_accepted_backend_event(backend_event)
        if lesson_task_id:
            await self._advance_lesson(lesson_task_id)

    def _mark_lesson_speech_finished_locked(
        self, task_id: str, version: int | None
    ) -> str | None:
        """Release a lesson only after both speech and its board commit finish."""

        if version is None:
            return None
        key = (task_id, version)
        if key not in self._lesson_board_ready:
            self._lesson_speech_finished.add(key)
            return None
        should_advance = self._lesson_board_ready.pop(key)
        self._lesson_speech_finished.discard(key)
        return task_id if should_advance else None

    def _finish_lesson_step_locked(
        self, task_id: str, version: int | None
    ) -> tuple[str | None, list[TaskBackendEvent]]:
        lesson_task_id = self._mark_lesson_speech_finished_locked(task_id, version)
        if lesson_task_id is None or version is None:
            return lesson_task_id, []
        prefetched_events = self._release_lesson_prefetch_locked(task_id, version)
        if prefetched_events is not None:
            return None, prefetched_events
        timer = self._lesson_prefetch_timers.pop(task_id, None)
        if timer is not None and not timer.done():
            timer.cancel()
        return lesson_task_id, []

    def _clear_lesson_barriers_locked(self, task_id: str) -> None:
        self._pending_lesson_boards = {
            key: board for key, board in self._pending_lesson_boards.items()
            if key[0] != task_id
        }
        self._lesson_board_ready = {
            key: value
            for key, value in self._lesson_board_ready.items()
            if key[0] != task_id
        }
        self._lesson_speech_finished = {
            key for key in self._lesson_speech_finished if key[0] != task_id
        }
        self._discard_lesson_prefetch_locked(task_id)

    async def _advance_lesson(self, task_id: str) -> None:
        """Resume a lesson or release its already prepared next step."""

        async with self._replace_lock:
            if self._closed or self._user_speaking or self._user_generation_id:
                if not self._closed:
                    self._pending_lesson_task_id = task_id
                    self._pending_lesson_version = self._lesson_resume_versions.get(
                        task_id
                    )
                return
            expected_version = self._lesson_resume_versions.pop(task_id, None)
        runtime = self._task_hub.runtime if self._task_hub else self.core.agent_tools
        record = await runtime.task(task_id)
        if record is None:
            async with self._replace_lock:
                self._discard_lesson_prefetch_locked(task_id)
            return

        prefetched_events: list[TaskBackendEvent] | None = None
        async with self._replace_lock:
            source_version = self._lesson_prefetch_sources.get(task_id)
            if (
                source_version is not None
                and expected_version == source_version
                and record.version <= source_version + 1
                and record.status.value in {"running", "completed"}
            ):
                prefetched_events = self._release_lesson_prefetch_locked(
                    task_id, source_version
                )
            elif source_version is not None and (
                record.version > source_version + 1
                or record.status.value not in {"running", "completed"}
            ):
                self._discard_lesson_prefetch_locked(task_id)
        if prefetched_events is not None:
            for backend_event in prefetched_events:
                await self._handle_accepted_backend_event(backend_event)
            return
        if record.status.value != "running":
            return
        if expected_version is not None and record.version != expected_version:
            # The frontend model updated this lesson during the interruption.
            return
        try:
            await runtime.update(
                task_id,
                self._next_lesson_step_instruction(prefetched=False),
                record_update=False,
            )
            if self._task_hub is not None:
                await self._task_hub.persist_task(task_id)
        except Exception:
            logger.exception("Could not advance whiteboard lesson %s", task_id)

    async def _capture_interrupted_lesson(self) -> None:
        """Remember a paused lesson so the accepted user turn can resume it."""

        async with self._replace_lock:
            task_id = self._active_lesson_task_id or self._pending_lesson_task_id
            lesson_version = (
                self._active_lesson_version
                if self._active_lesson_task_id
                else self._pending_lesson_version
            )
            prefetched_source = (
                self._lesson_prefetch_sources.get(task_id) if task_id else None
            )
        if not task_id:
            return
        runtime = self._task_hub.runtime if self._task_hub else self.core.agent_tools
        record = await runtime.task(task_id)
        if record is None or record.status.value != "running":
            return
        async with self._replace_lock:
            self._interrupted_lesson_task_id = task_id
            self._lesson_resume_versions[task_id] = (
                prefetched_source
                if prefetched_source == lesson_version
                else record.version
            )

    @Manager.event_handler(XLiveInterruptionRequested, priority=120)
    async def _on_interruption(self, event: XLiveInterruptionRequested) -> None:
        # A non-blocking barge-in may arrive after a newer user commit has
        # already started another generation. Never let stale cleanup cancel it.
        if (
            event.generation_id is not None
            and event.generation_id != self._active_generation_id
        ):
            return
        await self._capture_interrupted_lesson()
        await self._cancel_active(event.reason or "user_barge_in")

    async def submit_task_event(self, event: TaskBackendEvent) -> None:
        """Enqueue a backend event without re-entering the active model call."""

        if self._closed:
            return
        if self._backend_event_pump is None or self._backend_event_pump.done():
            self._backend_event_pump = asyncio.create_task(
                self._pump_backend_events(),
                name=f"xlive-agent-events-{self.session_id}",
            )
        await self._backend_events.put(event)

    async def _pump_backend_events(self) -> None:
        while True:
            event = await self._backend_events.get()
            try:
                await self.event_bus.publish(
                    XLiveTaskBackendEventReceived(
                        session_id=self.session_id,
                        backend_event=event,
                    ),
                    wait_for_completion=True,
                )
            finally:
                self._backend_events.task_done()

    async def _replace_run(
        self,
        envelope: EventEnvelope,
        *,
        user_commit: bool = False,
    ) -> None:
        """Cancel the prior run/output, then launch the latest accepted input."""

        async with self._replace_lock:
            if self._closed:
                return
            interrupted_lesson_task_id = self._interrupted_lesson_task_id
            await self._cancel_active_locked("input_superseded")
            generation_id = str(uuid4())
            self._active_generation_id = generation_id
            if user_commit:
                self._user_speaking = False
                self._user_generation_id = generation_id
                self._pending_lesson_task_id = None
                self._pending_lesson_version = None
                self._active_lesson_task_id = interrupted_lesson_task_id
                self._active_lesson_version = (
                    self._lesson_resume_versions.get(interrupted_lesson_task_id)
                    if interrupted_lesson_task_id
                    else None
                )
                self._interrupted_lesson_task_id = None
                if interrupted_lesson_task_id:
                    self._lesson_resume_generations.add(generation_id)
            self._active_task = asyncio.create_task(
                self._run(envelope, generation_id),
                name=f"xlive-model-{self.session_id}-{generation_id}",
            )

    async def _queue_backend_run(self, envelope: EventEnvelope) -> None:
        """Queue backend updates without interrupting user-facing speech."""

        async with self._replace_lock:
            if self._closed:
                return
            self._pending_backend_runs.append(envelope)
            self._retire_completed_generation_locked()
            self._start_next_backend_run_locked()

    def _retire_completed_generation_locked(self) -> None:
        generation_id = self._active_generation_id
        if generation_id is None:
            return
        generation = self.core.output.generation(generation_id)
        if generation is None or generation.state in (
            GenerationState.COMPLETED,
            GenerationState.CANCELLED,
        ):
            self._active_generation_id = None

    def _start_next_backend_run_locked(self) -> None:
        if (
            self._closed
            or self._backend_delivery_frozen_locked()
            or self._active_lesson_task_id is not None
            or len(self.core.output.live_generation_ids()) >= 2
        ):
            return
        if self._active_task is not None and not self._active_task.done():
            return
        if not self._pending_backend_runs:
            return
        envelope = self._pending_backend_runs.popleft()
        generation_id = str(uuid4())
        self._active_generation_id = generation_id
        whiteboard = envelope.payload.get("whiteboard")
        if not whiteboard and not envelope.payload.get("narrate"):
            self._backend_deliveries[generation_id] = envelope
        is_lesson_narration = envelope.payload.get("narrate") is True
        is_legacy_auto_step = (
            isinstance(whiteboard, dict) and whiteboard.get("auto_continue") is True
        )
        self._active_lesson_task_id = (
            envelope.task_id if is_lesson_narration or is_legacy_auto_step else None
        )
        self._active_lesson_version = (
            int(envelope.payload.get("version", 0))
            if self._active_lesson_task_id
            else None
        )
        self._active_task = asyncio.create_task(
            self._run(envelope, generation_id),
            name=f"xlive-model-{self.session_id}-{generation_id}",
        )

    async def _run(
        self,
        envelope: EventEnvelope,
        generation_id: str,
    ) -> None:
        try:
            if generation_id in self._backend_deliveries:
                runtime = self._task_hub.runtime if self._task_hub else self.core.agent_tools
                record = await runtime.task(envelope.task_id)
                if record is None or record.version != envelope.payload.get("version"):
                    self._backend_deliveries.pop(generation_id, None)
                    return
            await self.runner.run(
                envelope,
                generation_id=generation_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Frontend model request failed: session=%s generation=%s",
                self.session_id, generation_id,
            )
            self.core.events.append(
                "model_run_failed",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                turn_id=envelope.turn_id,
                revision=envelope.revision,
                task_id=envelope.task_id,
                generation_id=generation_id,
            )
            await self.event_bus.publish(
                ErrorOccurred(
                    session_id=self.session_id,
                    error_type=("xlive_task_operation_rejected" if isinstance(exc, InvalidTaskOperation)
                                else "xlive_model_run_failed"),
                    error_message=str(exc),
                )
            )
        finally:
            task = asyncio.current_task()
            if task is not None:
                asyncio.create_task(
                    self._finish_run(task, generation_id),
                    name=f"xlive-finish-{self.session_id}-{generation_id}",
                )

    async def _finish_run(
        self,
        task: asyncio.Task[None],
        generation_id: str,
    ) -> None:
        """Retire a model task without deadlocking cancellation under the lock."""

        pending_backend_events: list[TaskBackendEvent] = []
        prefetched_events: list[TaskBackendEvent] = []
        lesson_task_id: str | None = None
        async with self._replace_lock:
            if self._active_task is task:
                self._active_task = None
            if self._user_generation_id == generation_id:
                self._user_generation_id = None
                if not self._user_speaking:
                    pending_backend_events = self._take_pending_backend_events_locked()
            if self._active_generation_id == generation_id:
                generation = self.core.output.generation(generation_id)
                if generation is not None and generation.state is GenerationState.COMPLETED:
                    self._backend_deliveries.pop(generation_id, None)
                    await self.model.record_playback_outcome(generation_id, completed=True, started=True)
                if (
                    generation_id in self._lesson_playback_finished
                    and generation is not None
                    and generation.state is GenerationState.COMPLETED
                ):
                    self._lesson_playback_finished.discard(generation_id)
                    lesson_task_id = self._active_lesson_task_id
                    lesson_version = self._active_lesson_version
                    resume_generation = generation_id in self._lesson_resume_generations
                    self._lesson_resume_generations.discard(generation_id)
                    self._active_lesson_task_id = None
                    self._active_lesson_version = None
                    self._active_generation_id = None
                    self._lesson_playback_started.discard(generation_id)
                    if lesson_task_id and self._user_speaking:
                        self._pending_lesson_task_id = lesson_task_id
                        self._pending_lesson_version = lesson_version
                        lesson_task_id = None
                    elif lesson_task_id and resume_generation:
                        lesson_task_id = self._mark_lesson_speech_finished_locked(
                            lesson_task_id, lesson_version
                        )
                    elif lesson_task_id:
                        lesson_task_id, prefetched_events = (
                            self._finish_lesson_step_locked(
                                lesson_task_id, lesson_version
                            )
                        )
                else:
                    self._retire_completed_generation_locked()
            self._start_next_backend_run_locked()
        for backend_event in pending_backend_events:
            await self._deliver_backend_event_to_model(backend_event)
        for backend_event in prefetched_events:
            await self._handle_accepted_backend_event(backend_event)
        if lesson_task_id:
            await self._advance_lesson(lesson_task_id)

    def _backend_delivery_frozen_locked(self) -> bool:
        return self._user_speaking or self._user_generation_id is not None

    def _take_pending_backend_events_locked(self) -> list[TaskBackendEvent]:
        pending = list(self._pending_backend_events)
        self._pending_backend_events.clear()
        return pending

    async def _cancel_active(self, reason: str) -> None:
        async with self._replace_lock:
            await self._cancel_active_locked(reason)

    async def _cancel_active_locked(self, reason: str) -> None:
        task = self._active_task
        generation_id = self._active_generation_id
        live_generations = [
            (self.core.output.generation(gid), any(
                self.core.output.utterance(uid).played_chunk_ids
                or self.core.output.utterance(uid).state.value == "playing"
                for uid in self.core.output.generation(gid).utterance_ids
            )) for gid in self.core.output.live_generation_ids()
        ]
        self._active_task = None
        self._active_generation_id = None
        self._active_lesson_task_id = None
        self._active_lesson_version = None
        if generation_id is not None:
            self._lesson_playback_finished.discard(generation_id)
            self._lesson_playback_started.discard(generation_id)
            self._lesson_resume_generations.discard(generation_id)

        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        for generation, started in reversed(live_generations):
            live_id = generation.generation_id
            utterance_id = generation.utterance_ids[-1] if generation.utterance_ids else ""
            await self.model.record_playback_outcome(live_id, completed=False, started=started)
            pending = self._backend_deliveries.pop(live_id, None)
            if pending is not None and reason in {"input_superseded", "smart_turn_barge_in", "x2_turn_barge_in", "user_barge_in"}:
                runtime = self._task_hub.runtime if self._task_hub else self.core.agent_tools
                record = await runtime.task(pending.task_id)
                if record is not None and record.version == pending.payload.get("version"):
                    self._pending_backend_runs.appendleft(pending)
            await self.event_bus.publish(
                XLiveGenerationCancelRequested(
                    session_id=self.session_id,
                    generation_id=live_id,
                    utterance_id=utterance_id,
                    reason=reason,
                ),
                wait_for_completion=True,
            )

    async def shutdown(self) -> None:
        async with self._replace_lock:
            self._closed = True
            self._pending_backend_runs.clear()
            self._backend_deliveries.clear()
            self._pending_backend_events.clear()
            self._pending_lesson_task_id = None
            self._pending_lesson_version = None
            self._active_lesson_task_id = None
            self._active_lesson_version = None
            self._interrupted_lesson_task_id = None
            self._lesson_resume_versions.clear()
            self._lesson_resume_generations.clear()
            self._lesson_playback_finished.clear()
            self._lesson_playback_started.clear()
            for task_id in tuple(self._lesson_prefetch_timers):
                self._discard_lesson_prefetch_locked(task_id)
            self._lesson_prefetch_events.clear()
            self._lesson_board_ready.clear()
            self._pending_lesson_boards.clear()
            self._lesson_speech_finished.clear()
            self._ui_cancelled_task_ids.clear()
            self._model_whiteboard_versions.clear()
            if self._task_hub is not None and self._hub_subscription is not None:
                self._task_hub.unsubscribe(self._hub_subscription)
                self._hub_subscription = None
            elif self._event_sink_setter is not None:
                self._event_sink_setter(None)
            if self._backend_event_pump is not None:
                self._backend_event_pump.cancel()
                await asyncio.gather(
                    self._backend_event_pump,
                    return_exceptions=True,
                )
                self._backend_event_pump = None
            await self._cancel_active_locked("session_stopped")
        close_operations = [self.model.aclose()]
        if self._task_hub is None:
            close_operations.append(self.core.agent_tools.aclose())
        await asyncio.gather(*close_operations, return_exceptions=True)
