"""Provider-neutral streaming speech adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class ASRUpdate:
    """One authoritative transcript snapshot from the configured ASR."""

    text: str
    is_final: bool = False
    first_partial_latency: float | None = None


@dataclass(frozen=True, slots=True)
class TurnStateUpdate:
    """One semantic endpointing result; transcripts are never included."""

    controller_state: Literal["idle", "nonidle", "speak"]
    turn_class: Literal[
        "idle",
        "noidle",
        "speaking",
        "turn_end",
        "backchannel",
        "uncertain",
    ] | None = None
    event: str | None = None
    reason: str | None = None
    confidence: float | None = None
    source_timestamp: float | None = None


class StreamingASRAdapter(Protocol):
    """Async, session-scoped ASR stream."""

    async def start(self, *, sample_rate: int = 16_000) -> None: ...

    async def push_audio(self, pcm16: bytes) -> ASRUpdate | None: ...

    async def finish(self) -> ASRUpdate: ...

    async def aclose(self) -> None: ...


class TurnStateAdapter(Protocol):
    """Async, session-scoped turn-state stream."""

    @property
    def provider(self) -> str: ...

    async def start(self, *, bot_speaking: bool) -> None: ...

    async def push_audio(
        self,
        pcm16: bytes,
    ) -> TurnStateUpdate | None: ...

    async def finish(self) -> TurnStateUpdate: ...

    async def aclose(self) -> None: ...
