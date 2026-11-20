from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from typing import Any
from uuid import uuid4

from .runtime import AgentBackend, AgentEventSink, AgentToolRuntime, TaskBackendEvent, TaskEventKind


logger = logging.getLogger(__name__)


class SessionTaskHub:
    """Keep one task runtime alive across transient WebSocket services."""

    def __init__(self, session_id: str, backend: AgentBackend) -> None:
        self.session_id = session_id
        self.runtime = AgentToolRuntime(backend)
        self._backend = backend
        self._subscribers: dict[str, AgentEventSink] = {}
        self._lock = asyncio.Lock()
        self._persistence: Any | None = None
        self._user_id: str | None = None
        self._closed = False
        setter = getattr(backend, "set_event_sink", None)
        if callable(setter):
            setter(self._receive_backend_event)

    def attach_persistence(self, persistence: Any, user_id: str) -> None:
        """Attach the authoritative session store once a user connects."""

        self._persistence = persistence
        self._user_id = user_id

    def subscribe(self, sink: AgentEventSink) -> str:
        token = str(uuid4())
        self._subscribers[token] = sink
        return token

    def unsubscribe(self, token: str) -> None:
        self._subscribers.pop(token, None)

    async def persist_task(self, task_id: str, *, backend: str = "Codex") -> None:
        record = await self.runtime.task(task_id)
        if record is None or self._persistence is None or self._user_id is None:
            return
        self._persistence.upsert_task(
            user_id=self._user_id,
            session_id=self.session_id,
            task_id=record.task_id,
            instruction=record.instruction,
            title=record.title,
            backend=backend,
            status=record.status.value,
            version=record.version,
            detail=record.detail,
            updates=list(record.updates),
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    async def _receive_backend_event(self, event: TaskBackendEvent) -> None:
        if self._closed:
            return
        accepted = await self.runtime.apply_backend_event(event)
        if not accepted:
            return
        if self._persistence is not None and self._user_id is not None:
            await self.persist_task(event.task_id)
            self._persistence.append_task_event(
                user_id=self._user_id,
                session_id=self.session_id,
                event=event,
            )
        if event.kind is TaskEventKind.ACTIVITY:
            return
        async with self._lock:
            subscribers = tuple(self._subscribers.values())
        if subscribers:
            results = await asyncio.gather(
                *(sink(event) for sink in subscribers),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, BaseException):
                    logger.error(
                        "task event subscriber failed - session: %s, task: %s, error: %s",
                        self.session_id,
                        event.task_id,
                        result,
                    )

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            self._subscribers.clear()
        setter = getattr(self._backend, "set_event_sink", None)
        if callable(setter):
            setter(None)
        await self.runtime.aclose()


class TaskHubRegistry:
    """Process-level registry keyed by persistent X-Live session id."""

    def __init__(self, backend_factory: Callable[[str], AgentBackend]) -> None:
        self._backend_factory = backend_factory
        self._hubs: dict[str, SessionTaskHub] = {}
        self._closed = False

    def get_or_create(self, session_id: str) -> SessionTaskHub:
        if self._closed:
            raise RuntimeError("task hub registry is closed")
        hub = self._hubs.get(session_id)
        if hub is None:
            hub = SessionTaskHub(session_id, self._backend_factory(session_id))
            self._hubs[session_id] = hub
        return hub

    async def remove(self, session_id: str) -> bool:
        """Close and forget one persistent session backend."""

        hub = self._hubs.pop(session_id, None)
        if hub is None:
            return False
        await hub.aclose()
        return True

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        hubs = tuple(self._hubs.values())
        self._hubs.clear()
        if hubs:
            await asyncio.gather(*(hub.aclose() for hub in hubs), return_exceptions=True)
