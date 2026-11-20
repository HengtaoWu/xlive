import asyncio
from collections import deque
from dataclasses import dataclass, field
from fastapi import WebSocket

from ..log_utils import logger


@dataclass
class QueueWaiter:
    websocket: WebSocket
    granted: asyncio.Event = field(default_factory=asyncio.Event)
    cancelled: bool = False
    holds_slot: bool = False


class SessionLimiter:
    """Manage concurrent WebSocket sessions with a FIFO wait queue."""

    def __init__(self, max_sessions: int = 0):
        self.max_sessions = max_sessions
        self.active_sessions = 0
        self.wait_queue: deque[QueueWaiter] = deque()
        self._lock = asyncio.Lock()

    @property
    def is_limited(self) -> bool:
        return self.max_sessions > 0

    async def acquire(self, websocket: WebSocket) -> QueueWaiter | None:
        """Acquire a slot, or queue and wait for one. Returns None if the client left."""
        waiter = QueueWaiter(websocket=websocket)

        if not self.is_limited:
            waiter.granted.set()
            return waiter

        async with self._lock:
            if self.active_sessions < self.max_sessions:
                self.active_sessions += 1
                waiter.granted.set()
                waiter.holds_slot = True
                return waiter
            self.wait_queue.append(waiter)
            position = len(self.wait_queue)

        try:
            await websocket.send_json(
                {
                    "action": "queue_status",
                    "data": {
                        "position": position,
                        "limit": self.max_sessions,
                        "active": self.active_sessions,
                    },
                }
            )
        except Exception as e:
            logger.warning(f"Failed to send queue_status: {e}")
            await self._remove_waiter(waiter)
            return None

        update_task = asyncio.create_task(
            self._periodic_queue_update(waiter), name="queue_update"
        )

        try:
            await waiter.granted.wait()
        except asyncio.CancelledError:
            waiter.cancelled = True
            await self._remove_waiter(waiter)
            update_task.cancel()
            return None
        finally:
            update_task.cancel()

        if waiter.cancelled:
            return None

        try:
            await websocket.send_json({"action": "queue_granted"})
        except Exception as e:
            logger.warning(f"Failed to send queue_granted: {e}")
            await self.release(waiter)
            return None

        return waiter

    async def release(self, waiter: QueueWaiter | None) -> None:
        """Release a slot and wake the next queued waiter."""
        if waiter is None:
            return

        if not self.is_limited:
            return

        async with self._lock:
            if waiter in self.wait_queue:
                self.wait_queue.remove(waiter)
                return

            if not waiter.holds_slot:
                return

            waiter.holds_slot = False

            if self.wait_queue:
                next_waiter = self.wait_queue.popleft()
                next_waiter.holds_slot = True
                next_waiter.granted.set()
            else:
                self.active_sessions -= 1

    async def _remove_waiter(self, waiter: QueueWaiter) -> None:
        async with self._lock:
            if waiter in self.wait_queue:
                self.wait_queue.remove(waiter)

    async def _periodic_queue_update(self, waiter: QueueWaiter) -> None:
        while not waiter.granted.is_set() and not waiter.cancelled:
            await asyncio.sleep(5)
            if waiter.granted.is_set() or waiter.cancelled:
                break

            async with self._lock:
                if waiter not in self.wait_queue:
                    break
                position = list(self.wait_queue).index(waiter) + 1

            try:
                await waiter.websocket.send_json(
                    {
                        "action": "queue_status",
                        "position": position,
                        "limit": self.max_sessions,
                        "active": self.active_sessions,
                    }
                )
            except Exception:
                # Connection lost; cancel and wake acquire() to avoid leaks
                waiter.cancelled = True
                waiter.granted.set()
                async with self._lock:
                    if waiter in self.wait_queue:
                        self.wait_queue.remove(waiter)
                break
