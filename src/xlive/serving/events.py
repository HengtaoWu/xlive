"""X-Live-owned serving events that cross adapter boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..runtime.protocol_events import BaseEvent

from ..agent.runtime import TaskBackendEvent


@dataclass
class XLiveInterruptionRequested(BaseEvent):
    """Request cancellation of the current model/output generation."""

    TYPE: ClassVar[str] = "xlive.interaction.interruption_requested"
    reason: str = "user_barge_in"
    generation_id: str | None = None


@dataclass
class XLiveTaskBackendEventReceived(BaseEvent):
    """Carry one asynchronous Agent backend event into the session runtime."""

    TYPE: ClassVar[str] = "xlive.agent.backend_event_received"
    backend_event: TaskBackendEvent | None = None
