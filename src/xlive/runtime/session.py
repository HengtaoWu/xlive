"""Provider-neutral orchestration for one X-Live realtime session."""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections import deque
from dataclasses import asdict, dataclass, replace
from typing import Any

from .events import EventEnvelope, SessionEventLog
from ..frontend_model.contract import (
    FrontendModelAdapter,
    ModelOutput,
    ModelOutputKind,
    NativeToolName,
    validate_model_output,
)
from .output import GenerationState, OutputController, UtteranceRecord
from ..agent.runtime import AgentToolRuntime, DelegationUserContext, TaskBackendEvent


class StaleGenerationError(RuntimeError):
    """Raised before an obsolete model output can cause a visible side effect."""


@dataclass(frozen=True, slots=True)
class GenerationContext:
    generation_id: str
    turn_id: str
    revision: int
    user_context: DelegationUserContext | None = None


@dataclass(frozen=True, slots=True)
class AcceptedModelOutput:
    generation_id: str
    output: ModelOutput
    event: EventEnvelope
    tool_result: dict[str, Any] | None = None


class RealtimeSessionCore:
    """Track accepted events, model output, tool calls, and playback state."""

    def __init__(
        self,
        session_id: str,
        agent_tools: AgentToolRuntime,
        *,
        event_capacity: int = 1_000,
    ) -> None:
        self.session_id = session_id
        self.agent_tools = agent_tools
        self.events = SessionEventLog(session_id, capacity=event_capacity)
        self.output = OutputController()
        self._turn_revisions: dict[str, int] = {}
        self._generations: dict[str, GenerationContext] = {}
        self._accepted_tool_call_ids: set[str] = set()
        self._task_contexts: dict[str, tuple[str, int]] = {}
        self._user_contexts: deque[tuple[int, DelegationUserContext]] = deque(
            maxlen=event_capacity
        )

    def record_input(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        turn_id: str | None = None,
        revision: int | None = None,
        task_id: str | None = None,
        event_id: str | None = None,
    ) -> EventEnvelope:
        if (turn_id is None) != (revision is None):
            raise ValueError("turn_id and revision must be provided together")
        revision_advanced = False
        if revision is not None:
            if revision < 0:
                raise ValueError("revision must be non-negative")
            previous_revision = self._turn_revisions.get(turn_id)
            revision_advanced = (
                previous_revision is not None and revision > previous_revision
            )
            self._turn_revisions[turn_id] = (
                revision
                if previous_revision is None
                else max(revision, previous_revision)
            )
        event = self.events.append(
            kind,
            payload,
            event_id=event_id,
            turn_id=turn_id,
            revision=revision,
            task_id=task_id,
        )
        if event.kind == "user_commit":
            user_context = self._user_context_from_event(event)
            if user_context is not None:
                self._user_contexts.append((event.sequence, user_context))
        if revision_advanced:
            current_id = self.output.current_generation_id
            context = None if current_id is None else self._generations.get(current_id)
            if (
                current_id is not None
                and context is not None
                and context.turn_id == turn_id
                and context.revision < revision
            ):
                self.cancel_generation(current_id, reason="revision_superseded")
        return event

    @staticmethod
    def _user_context_from_event(event: EventEnvelope) -> DelegationUserContext | None:
        text = event.payload.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        source = event.payload.get("source", "unspecified")
        return DelegationUserContext(
            text=text,
            event_id=event.event_id,
            turn_id=event.turn_id,
            revision=event.revision,
            input_source=source if isinstance(source, str) else "unspecified",
        )

    def delegation_context_for(self, event: EventEnvelope) -> DelegationUserContext | None:
        if event.session_id != self.session_id:
            raise ValueError("cannot use another session's user context")
        if event.kind == "user_commit":
            return self._user_context_from_event(event)
        # Include only user input available when the backend event occurred.
        for sequence, context in reversed(self._user_contexts):
            if sequence <= event.sequence:
                return replace(context, trigger_kind=event.kind)
        return None

    def begin_generation(
        self,
        turn_id: str,
        revision: int,
        *,
        generation_id: str | None = None,
        source_event_id: str | None = None,
        user_context: DelegationUserContext | None = None,
        preserve_playback: bool = False,
    ) -> GenerationContext:
        latest = self._turn_revisions.get(turn_id)
        if latest is not None and revision != latest:
            raise StaleGenerationError(
                f"turn {turn_id} revision {revision} is stale; latest is {latest}"
            )
        previous_id = self.output.current_generation_id
        record = self.output.begin_generation(
            turn_id, revision, generation_id=generation_id,
            preserve_playback=preserve_playback,
        )
        if previous_id is not None and previous_id != record.generation_id:
            previous = self.output.generation(previous_id)
            if previous is not None and previous.state is GenerationState.CANCELLED:
                self.events.append(
                    "generation_cancelled",
                    {"reason": "superseded"},
                    turn_id=previous.turn_id,
                    revision=previous.revision,
                    generation_id=previous.generation_id,
                )
        context = GenerationContext(
            generation_id=record.generation_id,
            turn_id=turn_id,
            revision=revision,
            user_context=user_context,
        )
        self._generations[record.generation_id] = context
        generation_payload: dict[str, Any] = {}
        if source_event_id is not None:
            generation_payload["source_event_id"] = source_event_id
        self.events.append(
            "generation_started",
            generation_payload,
            turn_id=turn_id,
            revision=revision,
            generation_id=record.generation_id,
        )
        return context

    async def accept_model_output(
        self, generation_id: str, output: ModelOutput
    ) -> AcceptedModelOutput:
        context = self._require_fresh_generation(generation_id)
        validate_model_output(output)

        tool_result: dict[str, Any] | None = None
        task_id: str | None = None
        if output.kind is ModelOutputKind.TOOL_CALL:
            assert output.tool_call is not None
            if output.tool_call.call_id in self._accepted_tool_call_ids:
                raise ValueError(
                    f"duplicate native tool call: {output.tool_call.call_id}"
                )
            # Reserve the call id before awaiting the backend so a retry cannot
            # start the same external task twice.
            self._accepted_tool_call_ids.add(output.tool_call.call_id)
            tool_result = await self.agent_tools.execute(
                output.tool_call, user_context=context.user_context
            )
            task_id = tool_result.get("task_id")
            if isinstance(task_id, str):
                self._task_contexts.setdefault(
                    task_id, (context.turn_id, context.revision)
                )

        payload: dict[str, Any] = {
            "kind": output.kind.value,
            "text": output.text,
        }
        if output.tool_call is not None:
            payload["tool_call"] = {
                "call_id": output.tool_call.call_id,
                "name": output.tool_call.name.value,
                "arguments": dict(output.tool_call.arguments),
            }
            if (
                output.tool_call.name in {NativeToolName.DELEGATE, NativeToolName.UPDATE}
                and context.user_context is not None
            ):
                payload["delegation_user_context"] = asdict(context.user_context)
        event = self.events.append(
            "model_output",
            payload,
            turn_id=context.turn_id,
            revision=context.revision,
            task_id=task_id,
            generation_id=generation_id,
        )
        if tool_result is not None:
            self.events.append(
                "tool_result",
                dict(tool_result),
                turn_id=context.turn_id,
                revision=context.revision,
                task_id=task_id,
                generation_id=generation_id,
            )
        return AcceptedModelOutput(generation_id, output, event, tool_result)

    async def run_model(
        self,
        model: FrontendModelAdapter,
        event: EventEnvelope,
        *,
        generation_id: str | None = None,
    ) -> AsyncIterator[AcceptedModelOutput]:
        """Run one replaceable model adapter with deterministic output checks."""
        if event.kind == "task_event":
            turn_id = f"task:{event.task_id or event.event_id}"
            revision = event.sequence
        elif event.turn_id is None or event.revision is None:
            turn_id = f"event:{event.event_id}"
            revision = event.sequence
        else:
            turn_id = event.turn_id
            revision = event.revision
        context = self.begin_generation(
            turn_id,
            revision,
            generation_id=generation_id,
            source_event_id=event.event_id,
            user_context=self.delegation_context_for(event),
            preserve_playback=event.kind == "task_event",
        )
        try:
            async for output in model.stream(
                event, generation_id=context.generation_id
            ):
                accepted = await self.accept_model_output(
                    context.generation_id, output
                )
                if accepted.tool_result is not None:
                    assert output.tool_call is not None
                    observer = getattr(model, "on_tool_result", None)
                    if callable(observer):
                        await observer(output.tool_call, accepted.tool_result)
                yield accepted
        except BaseException:
            self.output.cancel_generation(context.generation_id)
            await model.cancel(context.generation_id)
            raise
        else:
            self.seal_generation(context.generation_id)

    def record_applied_task_event(self, event: TaskBackendEvent) -> EventEnvelope:
        """Record a task event already accepted by a shared task runtime."""

        turn_context = self._task_contexts.get(event.task_id)
        payload = {
            "kind": event.kind.value,
            "version": event.version,
            "title": event.title,
            "detail": event.detail,
            "narrate": event.narrate,
            "backend_event_id": event.event_id,
        }
        if event.whiteboard is not None:
            payload["whiteboard"] = {
                "title": event.whiteboard.title,
                "state": event.whiteboard.state,
                "version": event.whiteboard.revision,
                "operations": list(event.whiteboard.operations),
                "auto_continue": event.whiteboard.auto_continue,
            }
        return self.events.append(
            "task_event",
            payload,
            turn_id=None if turn_context is None else turn_context[0],
            revision=None if turn_context is None else turn_context[1],
            task_id=event.task_id,
        )

    def begin_utterance(
        self,
        generation_id: str,
        text: str,
        *,
        utterance_id: str | None = None,
        include_in_history: bool = True,
    ) -> UtteranceRecord:
        self._require_fresh_generation(generation_id)
        record = self.output.begin_utterance(
            generation_id,
            text,
            utterance_id=utterance_id,
            include_in_history=include_in_history,
        )
        self.events.append(
            "utterance_started",
            {"text": text, "include_in_history": include_in_history},
            generation_id=generation_id,
            utterance_id=record.utterance_id,
        )
        return record

    def finish_synthesis(self, generation_id: str, utterance_id: str) -> bool:
        accepted = self.output.finish_synthesis(generation_id, utterance_id)
        if accepted:
            self.events.append(
                "utterance_synthesized",
                {},
                generation_id=generation_id,
                utterance_id=utterance_id,
            )
        return accepted

    def seal_generation(self, generation_id: str) -> bool:
        """Declare that no more utterances will be added to a generation."""

        accepted = self.output.seal_generation(generation_id)
        if accepted:
            context = self._generations.get(generation_id)
            self.events.append(
                "generation_sealed",
                {},
                turn_id=None if context is None else context.turn_id,
                revision=None if context is None else context.revision,
                generation_id=generation_id,
            )
        return accepted

    def append_utterance_text(
        self, generation_id: str, utterance_id: str, text: str
    ) -> bool:
        self._require_fresh_generation(generation_id)
        accepted = self.output.append_utterance_text(
            generation_id, utterance_id, text
        )
        if accepted:
            self.events.append(
                "utterance_text_appended",
                {"text": text},
                generation_id=generation_id,
                utterance_id=utterance_id,
            )
        return accepted

    def cancel_generation(self, generation_id: str, *, reason: str) -> bool:
        context = self._generations.get(generation_id)
        cancelled = self.output.cancel_generation(generation_id)
        if cancelled:
            self.events.append(
                "generation_cancelled",
                {"reason": reason},
                turn_id=None if context is None else context.turn_id,
                revision=None if context is None else context.revision,
                generation_id=generation_id,
            )
        return cancelled

    def _require_fresh_generation(self, generation_id: str) -> GenerationContext:
        context = self._generations.get(generation_id)
        record = self.output.generation(generation_id)
        if context is None or record is None:
            raise StaleGenerationError(f"unknown generation: {generation_id}")
        latest = self._turn_revisions.get(context.turn_id, context.revision)
        if (
            generation_id != self.output.current_generation_id
            or record.state is not GenerationState.ACTIVE
            or context.revision != latest
        ):
            self.output.cancel_generation(generation_id)
            raise StaleGenerationError(f"generation is stale: {generation_id}")
        return context
