import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Callable, Any, Set, Type, Union
import time

from ..log_utils import logger

from .protocol_events import BaseEvent, ErrorOccurred


@dataclass
class EventHandler:
    handler: Callable
    priority: int = 0


class EventBus:
    """Publish and subscribe session events with async dispatch support."""

    MAX_ERROR_EVENT_DEPTH = 3
    ERROR_EVENT_COOLDOWN = 1.0
    ERROR_EVENT_RATE_LIMIT = 10

    def __init__(self):
        self._handlers: Dict[str, List[EventHandler]] = defaultdict(list)
        self._active_tasks: Set[asyncio.Task] = set()
        self._error_event_depth = 0
        self._error_event_lock = asyncio.Lock()
        self._error_event_times: List[float] = []
        self._last_error_event_time = 0.0

    def _get_event_key(self, event_identifier: Union[Type[BaseEvent], str]) -> str:
        """Normalize event identifier (class or string) into type string."""
        if isinstance(event_identifier, str):
            return event_identifier

        event_type = getattr(event_identifier, "TYPE", None)
        if isinstance(event_type, str) and event_type:
            return event_type
        return event_identifier.__name__

    def subscribe(
        self,
        event_class: Union[Type[BaseEvent], str],
        handler: Callable[[BaseEvent], Any],
        priority: int = 0,
    ) -> None:
        """Subscribe a handler to an event type. Higher priority runs first."""
        event_handler = EventHandler(handler=handler, priority=priority)

        key = self._get_event_key(event_class)
        self._handlers[key].append(event_handler)
        self._handlers[key].sort(key=lambda h: h.priority, reverse=True)

    def unsubscribe(
        self, event_class: Union[Type[BaseEvent], str], handler: Callable
    ) -> bool:
        key = self._get_event_key(event_class)
        handlers = self._handlers.get(key, [])
        for i, event_handler in enumerate(handlers):
            if event_handler.handler == handler:
                del handlers[i]
                return True
        return False

    async def publish(
        self, event: BaseEvent, wait_for_completion: bool = False
    ) -> bool:
        """Publish an event to all matching handlers."""
        try:
            event_type = event.event_type
            handlers = self._handlers.get(event_type, [])
            if not handlers:
                return True

            if wait_for_completion:
                # Some event chains need deterministic completion before returning.
                for handler in handlers:
                    await self._handle_event_safe(handler, event)
                return True

            for handler in handlers:
                task = asyncio.create_task(self._handle_event_safe(handler, event))
                self._active_tasks.add(task)
                task.add_done_callback(self._active_tasks.discard)

            return True

        except Exception as e:
            logger.error("Failed to publish event: %s", e)

            if event.event_type != "error.occurred":
                await self._publish_error_event_safe(
                    session_id=event.session_id,
                    error_type="event_bus_publish_error",
                    error_message=str(e),
                )

            return False

    async def _handle_event_safe(self, handler: EventHandler, event: BaseEvent) -> None:
        try:
            if asyncio.iscoroutinefunction(handler.handler):
                await handler.handler(event)
            else:
                handler.handler(event)

        except Exception as e:
            logger.error(
                "Event handler raised: %s, event_type: %s", e, event.event_type
            )

            if event.event_type != "error.occurred":
                await self._publish_error_event_safe(
                    session_id=event.session_id,
                    error_type="event_handler_error",
                    error_message=str(e),
                )

    async def _publish_error_event_safe(
        self,
        session_id: str,
        error_type: str,
        error_message: str,
    ) -> None:
        """Publish an error event guarded by depth, cooldown, and rate limits."""
        if self._error_event_depth >= self.MAX_ERROR_EVENT_DEPTH:
            logger.error(
                "Max error event depth (%d) reached, dropping error event: %s",
                self.MAX_ERROR_EVENT_DEPTH,
                error_type,
            )
            return

        current_time = time.time()
        if current_time - self._last_error_event_time < self.ERROR_EVENT_COOLDOWN:
            logger.debug(
                "Error event cooldown active, dropping error event: %s",
                error_type,
            )
            return

        self._error_event_times = [
            t for t in self._error_event_times if current_time - t < 1.0
        ]
        if len(self._error_event_times) >= self.ERROR_EVENT_RATE_LIMIT:
            logger.warning(
                "Error event rate limit (%d/s) exceeded, dropping error event: %s",
                self.ERROR_EVENT_RATE_LIMIT,
                error_type,
            )
            return

        async with self._error_event_lock:
            self._error_event_depth += 1
            try:
                error_event = ErrorOccurred(
                    session_id=session_id,
                    error_type=error_type,
                    error_message=error_message,
                )

                await self.publish(error_event, wait_for_completion=False)
                self._last_error_event_time = current_time
                self._error_event_times.append(current_time)
            finally:
                self._error_event_depth -= 1

    def reset_error_tracking(self) -> None:
        self._error_event_depth = 0
        self._error_event_times.clear()
        self._last_error_event_time = 0.0

    async def _wait_for_all_handlers(self, timeout: float = 3.0):
        if not self._active_tasks:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._active_tasks, return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for handlers; cancelling %d pending tasks",
                len(self._active_tasks),
            )
            for task in self._active_tasks:
                if not task.done():
                    task.cancel()

    async def shutdown(self) -> None:
        await self._wait_for_all_handlers(timeout=3.0)
        self._handlers.clear()
        self._active_tasks.clear()
        self.reset_error_tracking()
