"""Provider-neutral contract for the trained X-Live frontend model."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from ..runtime.events import EventEnvelope


class ModelOutputKind(str, Enum):
    TEXT_DELTA = "text_delta"
    TOOL_CALL = "tool_call"


class NativeToolName(str, Enum):
    DELEGATE = "delegate"
    UPDATE = "update"
    CANCEL = "cancel"
    INSPECT = "inspect"


@dataclass(frozen=True, slots=True)
class NativeToolCall:
    call_id: str
    name: NativeToolName
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelOutput:
    kind: ModelOutputKind
    text: str = ""
    tool_call: NativeToolCall | None = None


class InvalidModelOutput(ValueError):
    """Raised when X-Live frontend output violates the serving contract."""


def validate_model_output(output: ModelOutput) -> None:
    """Accept only formal text deltas or one parsed native tool call."""
    if output.kind is ModelOutputKind.TEXT_DELTA:
        if not output.text or output.tool_call is not None:
            raise InvalidModelOutput("TEXT_DELTA requires text only")
        return

    if output.kind is ModelOutputKind.TOOL_CALL:
        if output.tool_call is None or output.text:
            raise InvalidModelOutput("TOOL_CALL requires a native call and no text")
        return

    raise InvalidModelOutput(f"unsupported output kind: {output.kind}")


class FrontendModelAdapter(Protocol):
    """One session-scoped X-Live frontend model serving adapter."""

    async def stream(
        self,
        event: EventEnvelope,
        *,
        generation_id: str,
    ) -> AsyncIterator[ModelOutput]: ...

    async def cancel(self, generation_id: str) -> None: ...

    async def record_playback_outcome(self, generation_id: str, *, completed: bool, started: bool) -> None: ...

    async def ingest_context_event(self, event: EventEnvelope) -> None: ...

    async def aclose(self) -> None: ...
