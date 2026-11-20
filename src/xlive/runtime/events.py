"""Runtime event envelopes and a bounded in-memory replay log."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field, replace
import time
from typing import Any
from uuid import uuid4


class DuplicateEventError(ValueError):
    """Raised when an event id is appended more than once."""


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event_id: str
    session_id: str
    sequence: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    revision: int | None = None
    task_id: str | None = None
    generation_id: str | None = None
    utterance_id: str | None = None
    created_at: float = field(default_factory=time.time)


class SessionEventLog:
    """Assign monotonic sequence numbers and retain a bounded replay window."""

    def __init__(self, session_id: str, *, capacity: int = 1_000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.session_id = session_id
        self.capacity = capacity
        self._next_sequence = 1
        self._events: deque[EventEnvelope] = deque(maxlen=capacity)
        self._event_ids: set[str] = set()

    def append(
        self,
        kind: str,
        payload: dict[str, Any] | None = None,
        *,
        event_id: str | None = None,
        turn_id: str | None = None,
        revision: int | None = None,
        task_id: str | None = None,
        generation_id: str | None = None,
        utterance_id: str | None = None,
        created_at: float | None = None,
    ) -> EventEnvelope:
        if not kind:
            raise ValueError("kind is required")
        resolved_event_id = event_id or str(uuid4())
        if resolved_event_id in self._event_ids:
            raise DuplicateEventError(resolved_event_id)
        if len(self._events) == self.capacity:
            evicted = self._events[0]
            self._event_ids.remove(evicted.event_id)
        event = EventEnvelope(
            event_id=resolved_event_id,
            session_id=self.session_id,
            sequence=self._next_sequence,
            kind=kind,
            payload=deepcopy(payload or {}),
            turn_id=turn_id,
            revision=revision,
            task_id=task_id,
            generation_id=generation_id,
            utterance_id=utterance_id,
            created_at=time.time() if created_at is None else created_at,
        )
        self._next_sequence += 1
        self._events.append(event)
        self._event_ids.add(event.event_id)
        return event

    def replay(self, *, after_sequence: int = 0) -> list[EventEnvelope]:
        return [
            replace(event, payload=deepcopy(event.payload))
            for event in self._events
            if event.sequence > after_sequence
        ]
