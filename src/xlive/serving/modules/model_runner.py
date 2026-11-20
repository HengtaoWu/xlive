"""Drive one explicit frontend-model run into the X-Live output lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from uuid import uuid4

from ...runtime.event_bus import EventBus
from ...runtime.protocol_events import (
    LLMFirstChunk,
    TaskStatusUpdated,
    ToolCallOccurred,
)

from ...agent.task_hub import SessionTaskHub
from ...frontend_model.contract import (
    FrontendModelAdapter,
    ModelOutputKind,
)
from ...runtime.events import EventEnvelope
from ...runtime.session import AcceptedModelOutput, RealtimeSessionCore
from .output_bridge import (
    XLiveGenerationCancelRequested,
    XLiveGenerationSealRequested,
    XLiveTextAppendRequested,
    XLiveTextFlushRequested,
    XLiveTextStartRequested,
)


class XLiveModelRunner:
    """Convert a model stream for an accepted input into serving events."""

    def __init__(
        self,
        event_bus: EventBus,
        session_id: str,
        core: RealtimeSessionCore,
        model: FrontendModelAdapter,
        task_hub: SessionTaskHub | None = None,
    ) -> None:
        self.event_bus = event_bus
        self.session_id = session_id
        self.core = core
        self.model = model
        self.task_hub = task_hub

    async def stream(
        self,
        event: EventEnvelope,
        *,
        generation_id: str | None = None,
    ) -> AsyncIterator[AcceptedModelOutput]:
        """Run the model and expose accepted outputs as they are validated."""

        resolved_generation_id = generation_id or str(uuid4())
        utterance_id: str | None = None
        first_output_seen = False
        model_stream = self.core.run_model(
            self.model,
            event,
            generation_id=resolved_generation_id,
        )
        try:
            async for accepted in model_stream:
                if not first_output_seen:
                    first_output_seen = True
                    await self.event_bus.publish(
                        LLMFirstChunk(session_id=self.session_id)
                    )
                output = accepted.output
                if output.kind is ModelOutputKind.TOOL_CALL:
                    assert output.tool_call is not None
                    await self.event_bus.publish(
                        ToolCallOccurred(
                            session_id=self.session_id,
                            name=output.tool_call.name.value,
                            args=dict(output.tool_call.arguments),
                        )
                    )
                if accepted.tool_result is not None:
                    result = accepted.tool_result
                    task_id = result.get("task_id")
                    if isinstance(task_id, str) and task_id:
                        if self.task_hub is not None:
                            await self.task_hub.persist_task(task_id)
                            record = await self.task_hub.runtime.task(task_id)
                            if record is not None:
                                result = {
                                    "task_id": record.task_id,
                                    "status": record.status.value,
                                    "title": record.title,
                                    "detail": record.detail,
                                    "updates": list(record.updates),
                                }
                        status = result["status"]
                        if status in {"starting", "cancelling"}:
                            status = "running"
                        await self.event_bus.publish(
                            TaskStatusUpdated(
                                session_id=self.session_id,
                                task_id=task_id,
                                title=result["title"],
                                status=status,
                                detail=result["detail"],
                                backend="Codex",
                                updates=[
                                    item for item in result["updates"] if item.strip()
                                ],
                            )
                        )
                if output.kind is ModelOutputKind.TEXT_DELTA:
                    if utterance_id is None:
                        utterance_id = str(uuid4())
                        await self.event_bus.publish(
                            XLiveTextStartRequested(
                                session_id=self.session_id,
                                generation_id=resolved_generation_id,
                                utterance_id=utterance_id,
                                text=output.text,
                                include_in_history=True,
                            ),
                            wait_for_completion=True,
                        )
                    else:
                        await self.event_bus.publish(
                            XLiveTextAppendRequested(
                                session_id=self.session_id,
                                generation_id=resolved_generation_id,
                                utterance_id=utterance_id,
                                text=output.text,
                            ),
                            wait_for_completion=True,
                        )
                yield accepted
        except BaseException as exc:
            await model_stream.aclose()
            if utterance_id is not None:
                if isinstance(exc, GeneratorExit):
                    reason = "model_stream_closed"
                elif isinstance(exc, asyncio.CancelledError):
                    reason = "model_stream_cancelled"
                else:
                    reason = "model_stream_failed"
                await self.event_bus.publish(
                    XLiveGenerationCancelRequested(
                        session_id=self.session_id,
                        generation_id=resolved_generation_id,
                        utterance_id=utterance_id,
                        reason=reason,
                    ),
                    wait_for_completion=True,
                )
            raise
        else:
            if utterance_id is not None:
                await self.event_bus.publish(
                    XLiveTextFlushRequested(
                        session_id=self.session_id,
                        generation_id=resolved_generation_id,
                        utterance_id=utterance_id,
                    ),
                    wait_for_completion=True,
                )
            await self.event_bus.publish(
                XLiveGenerationSealRequested(
                    session_id=self.session_id,
                    generation_id=resolved_generation_id,
                ),
                wait_for_completion=True,
            )

    async def run(
        self,
        event: EventEnvelope,
        *,
        generation_id: str | None = None,
    ) -> str:
        """Consume a complete model stream and return its generation id."""

        resolved_generation_id = generation_id or str(uuid4())
        async for _accepted in self.stream(
            event,
            generation_id=resolved_generation_id,
        ):
            pass
        return resolved_generation_id
