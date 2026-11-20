"""Client for the self-hosted Smart Turn semantic endpointing service."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import json
import os
from typing import Any, Protocol, cast

import websockets

from ...config_values import parse_positive_float
from ..contracts import TurnStateUpdate


class SmartTurnProtocolError(RuntimeError):
    """Raised when the Smart Turn bridge returns an invalid response."""


class _WebSocket(Protocol):
    async def send(self, message: str | bytes) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


Connector = Callable[[str], Awaitable[_WebSocket]]


@dataclass(frozen=True, slots=True)
class SmartTurnConfig:
    """Configuration for one locally-hosted Smart Turn service."""

    server_url: str = "ws://127.0.0.1:8005/turn"
    sample_rate: int = 16_000
    timeout: float = 5.0

    @classmethod
    def from_env(cls) -> "SmartTurnConfig":
        return cls(
            server_url=os.getenv("SMART_TURN_URL", "ws://127.0.0.1:8005/turn"),
            timeout=parse_positive_float(
                os.getenv("SMART_TURN_TIMEOUT_SECONDS", "5.0"),
                "SMART_TURN_TIMEOUT_SECONDS",
            ),
        )

    def __post_init__(self) -> None:
        if not self.server_url.startswith(("ws://", "wss://")):
            raise ValueError("Smart Turn server_url must use ws:// or wss://")
        if self.sample_rate != 16_000:
            raise ValueError("Smart Turn requires 16 kHz PCM input")
        if self.timeout <= 0:
            raise ValueError("Smart Turn timeout must be positive")


async def _default_connect(url: str) -> _WebSocket:
    return await websockets.connect(url, max_size=2**22)


class SmartTurnWebSocketAdapter:
    """Classify VAD-delimited audio segments without using transcripts."""

    def __init__(
        self,
        session_id: str,
        config: SmartTurnConfig | None = None,
        *,
        connector: Connector = _default_connect,
    ) -> None:
        if not session_id:
            raise ValueError("session_id is required")
        self.session_id = session_id
        self.config = config if config is not None else SmartTurnConfig()
        self._connector = connector
        self._websocket: _WebSocket | None = None
        self._lock = asyncio.Lock()
        self._audio = bytearray()
        self._active = False

    @property
    def provider(self) -> str:
        return "smart_turn"

    async def start(self, *, bot_speaking: bool) -> None:
        if self._active:
            raise RuntimeError("Smart Turn segment is already active")
        self._audio.clear()
        self._active = True

    async def push_audio(self, pcm16: bytes) -> TurnStateUpdate | None:
        if not isinstance(pcm16, bytes):
            raise TypeError("Smart Turn audio must be bytes")
        if not pcm16 or len(pcm16) % 2:
            raise ValueError("Smart Turn requires non-empty whole PCM16 samples")
        if not self._active:
            raise RuntimeError("Smart Turn segment has not started")
        self._audio.extend(pcm16)
        return None

    async def finish(self) -> TurnStateUpdate:
        if not self._active:
            raise RuntimeError("Smart Turn segment has not started")
        if not self._audio:
            raise ValueError("Smart Turn requires non-empty whole PCM16 samples")
        payload = {
            "type": "classify",
            "session_id": self.session_id,
            "sample_rate": self.config.sample_rate,
            "audio": base64.b64encode(self._audio).decode("ascii"),
        }
        async with self._lock:
            try:
                if self._websocket is None:
                    self._websocket = await self._connector(self.config.server_url)
                await self._websocket.send(json.dumps(payload))
                raw = await asyncio.wait_for(
                    self._websocket.recv(), timeout=self.config.timeout
                )
                state = _parse_turn_state(raw, self.session_id)
                self._audio.clear()
                self._active = False
                return state
            except BaseException:
                self._audio.clear()
                self._active = False
                websocket = self._websocket
                self._websocket = None
                if websocket is not None:
                    await websocket.close()
                raise

    async def aclose(self) -> None:
        async with self._lock:
            self._audio.clear()
            self._active = False
            websocket, self._websocket = self._websocket, None
            if websocket is not None:
                await websocket.close()


def _parse_turn_state(raw: str | bytes, session_id: str) -> TurnStateUpdate:
    if not isinstance(raw, str):
        raise SmartTurnProtocolError("Smart Turn response must be JSON text")
    try:
        message = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SmartTurnProtocolError("Smart Turn returned invalid JSON") from exc
    if not isinstance(message, dict):
        raise SmartTurnProtocolError("Smart Turn response must be a JSON object")
    if message.get("type") == "error":
        raise SmartTurnProtocolError(str(message.get("error", "unknown error")))
    if message.get("type") != "turn_state":
        raise SmartTurnProtocolError(
            f"expected turn_state response, got {message.get('type')!r}"
        )
    if message.get("session_id") != session_id:
        raise SmartTurnProtocolError("Smart Turn response session_id mismatch")
    state = message.get("state")
    if not isinstance(state, dict):
        raise SmartTurnProtocolError("Smart Turn state must be an object")
    controller_state = state.get("state")
    if controller_state not in {"nonidle", "speak"}:
        raise SmartTurnProtocolError(
            f"unknown Smart Turn state: {controller_state!r}"
        )
    confidence = state.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        raise SmartTurnProtocolError("Smart Turn confidence must be numeric")
    timestamp = message.get("ts")
    if timestamp is not None and not isinstance(timestamp, (int, float)):
        raise SmartTurnProtocolError("Smart Turn ts must be numeric")
    complete = controller_state == "speak"
    return TurnStateUpdate(
        controller_state=cast(Any, controller_state),
        turn_class="turn_end" if complete else "speaking",
        event="accept" if complete else "continue",
        reason="smart_turn_complete" if complete else "smart_turn_continue",
        confidence=None if confidence is None else float(confidence),
        source_timestamp=None if timestamp is None else float(timestamp),
    )
