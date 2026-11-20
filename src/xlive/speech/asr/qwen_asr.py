"""Client for X-Live's Qwen3-ASR streaming WebSocket protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import asyncio
import json
import os
from typing import Any, Protocol

import websockets

from ...config_values import parse_positive_float
from ..contracts import ASRUpdate


class QwenASRProtocolError(RuntimeError):
    """Raised when the Qwen3-ASR bridge violates the streaming protocol."""


class _WebSocket(Protocol):
    async def send(self, message: str | bytes) -> None: ...
    async def recv(self) -> str | bytes: ...
    async def close(self) -> None: ...


Connector = Callable[[str], Awaitable[_WebSocket]]


@dataclass(frozen=True, slots=True)
class QwenASRConfig:
    server_url: str = "ws://127.0.0.1:8003"
    sample_rate: int = 16_000
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "QwenASRConfig":
        return cls(
            server_url=os.getenv("QWEN_ASR_URL", "ws://127.0.0.1:8003"),
            timeout=parse_positive_float(
                os.getenv("QWEN_ASR_TIMEOUT_SECONDS", "30.0"),
                "QWEN_ASR_TIMEOUT_SECONDS",
            ),
        )

    def __post_init__(self) -> None:
        if not self.server_url.startswith(("ws://", "wss://")):
            raise ValueError("Qwen3-ASR server_url must use ws:// or wss://")
        if self.sample_rate != 16_000:
            raise ValueError("Qwen3-ASR requires 16 kHz input")
        if self.timeout <= 0:
            raise ValueError("Qwen3-ASR timeout must be positive")


async def _default_connect(url: str) -> _WebSocket:
    return await websockets.connect(url)


class QwenASRWebSocketAdapter:
    """Session-scoped adapter for the local Qwen3-ASR streaming bridge."""

    def __init__(
        self,
        config: QwenASRConfig | None = None,
        *,
        connector: Connector = _default_connect,
    ) -> None:
        self.config = config if config is not None else QwenASRConfig()
        self._connector = connector
        self._websocket: _WebSocket | None = None
        self._started = False
        self._lock = asyncio.Lock()

    async def start(self, *, sample_rate: int = 16_000) -> None:
        if sample_rate != self.config.sample_rate:
            raise ValueError(f"Qwen3-ASR expected {self.config.sample_rate} Hz, got {sample_rate}")
        async with self._lock:
            await self._start_locked()

    async def push_audio(self, pcm16: bytes) -> ASRUpdate | None:
        _validate_pcm16(pcm16)
        if not pcm16:
            return None
        async with self._lock:
            await self._start_locked()
            assert self._websocket is not None
            try:
                await self._websocket.send(pcm16)
                message = await self._receive_json_locked()
                _raise_if_error(message)
                if message.get("type") != "partial":
                    raise QwenASRProtocolError(f"expected Qwen3-ASR partial response, got {message.get('type')!r}")
                return ASRUpdate(text=_require_text(message), is_final=False)
            except BaseException:
                await self._close_locked()
                raise

    async def finish(self) -> ASRUpdate:
        async with self._lock:
            if not self._started:
                return ASRUpdate(text="", is_final=True)
            assert self._websocket is not None
            try:
                await self._websocket.send(json.dumps({"type": "end"}))
                message = await self._receive_json_locked()
                _raise_if_error(message)
                if message.get("type") != "final":
                    raise QwenASRProtocolError(f"expected Qwen3-ASR final response, got {message.get('type')!r}")
                latency = message.get("first_partial_latency")
                if latency is not None and not isinstance(latency, (int, float)):
                    raise QwenASRProtocolError("first_partial_latency must be numeric")
                update = ASRUpdate(
                    text=_require_text(message),
                    is_final=True,
                    first_partial_latency=None if latency is None else float(latency),
                )
                await self._close_locked()
                return update
            except BaseException:
                await self._close_locked()
                raise

    async def aclose(self) -> None:
        async with self._lock:
            await self._close_locked()

    async def _start_locked(self) -> None:
        if self._started:
            return
        self._websocket = await self._connector(self.config.server_url)
        try:
            await self._websocket.send(json.dumps({"type": "start", "sample_rate": self.config.sample_rate}))
            message = await self._receive_json_locked()
            _raise_if_error(message)
            if message.get("type") != "started":
                raise QwenASRProtocolError(f"expected Qwen3-ASR started response, got {message.get('type')!r}")
            if message.get("sample_rate") != self.config.sample_rate:
                raise QwenASRProtocolError("Qwen3-ASR acknowledged an unexpected sample rate")
            self._started = True
        except BaseException:
            await self._close_locked()
            raise

    async def _receive_json_locked(self) -> dict[str, Any]:
        assert self._websocket is not None
        raw = await asyncio.wait_for(self._websocket.recv(), timeout=self.config.timeout)
        if not isinstance(raw, str):
            raise QwenASRProtocolError("Qwen3-ASR control response must be JSON text")
        try:
            message = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise QwenASRProtocolError("Qwen3-ASR returned invalid JSON") from exc
        if not isinstance(message, dict):
            raise QwenASRProtocolError("Qwen3-ASR response must be a JSON object")
        return message

    async def _close_locked(self) -> None:
        websocket, self._websocket = self._websocket, None
        self._started = False
        if websocket is not None:
            await websocket.close()


def _validate_pcm16(pcm16: bytes) -> None:
    if not isinstance(pcm16, bytes):
        raise TypeError("Qwen3-ASR audio must be bytes")
    if len(pcm16) % 2:
        raise ValueError("Qwen3-ASR PCM16 audio must contain complete samples")


def _require_text(message: dict[str, Any]) -> str:
    text = message.get("text")
    if not isinstance(text, str):
        raise QwenASRProtocolError("Qwen3-ASR response text must be a string")
    return text


def _raise_if_error(message: dict[str, Any]) -> None:
    if message.get("type") == "error":
        raise QwenASRProtocolError(str(message.get("error", message.get("text", "unknown error"))))
