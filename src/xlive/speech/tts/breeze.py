"""Streaming adapter for the official Breeze TTS 2 HTTP service."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any

import numpy as np

from ...config_values import (
    parse_nonnegative_float,
    parse_nonnegative_int,
    parse_positive_float,
)


@dataclass(frozen=True, slots=True)
class BreezeTTSConfig:
    """Connection settings for the official Breeze TTS 2 streaming API."""

    base_url: str = "http://127.0.0.1:8004"
    sample_rate: int = 24_000
    channels: int = 1
    frame_duration_ms: int = 20
    request_timeout_seconds: float = 120.0
    busy_retry_attempts: int = 100
    busy_retry_delay_seconds: float = 0.1
    trim_edge_silence: bool = False

    @property
    def speech_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/audio/speech"

    @property
    def frame_bytes(self) -> int:
        return self.sample_rate * self.channels * 2 * self.frame_duration_ms // 1_000

    @classmethod
    def from_env(cls) -> "BreezeTTSConfig":
        return cls(
            trim_edge_silence=os.getenv("BREEZE_TRIM_EDGE_SILENCE", "0") == "1",
            base_url=os.getenv("BREEZE_TTS_URL", "http://127.0.0.1:8004"),
            request_timeout_seconds=parse_positive_float(
                os.getenv("BREEZE_TTS_TIMEOUT_SECONDS", "120"),
                "BREEZE_TTS_TIMEOUT_SECONDS",
            ),
            busy_retry_attempts=parse_nonnegative_int(
                os.getenv("BREEZE_TTS_BUSY_RETRY_ATTEMPTS", "100"),
                "BREEZE_TTS_BUSY_RETRY_ATTEMPTS",
            ),
            busy_retry_delay_seconds=parse_nonnegative_float(
                os.getenv("BREEZE_TTS_BUSY_RETRY_DELAY_SECONDS", "0.1"),
                "BREEZE_TTS_BUSY_RETRY_DELAY_SECONDS",
            ),
        )


class BreezeTTSError(RuntimeError):
    """Raised when the Breeze service rejects or corrupts a request."""


class BreezeTTSBusyError(BreezeTTSError):
    """Raised when the upstream single-concurrency service returns HTTP 409."""


@dataclass(frozen=True, slots=True)
class BreezeVoiceProfile:
    """Default voice inputs sent with every synthesis request."""

    instruction: str | None = None
    cfg_scale: float = 1.0
    reference_audio: bytes | None = field(default=None, repr=False)
    reference_text: str = ""
    reference_filename: str = "reference.wav"
    seed: int = 42

    def validate(self) -> None:
        if self.cfg_scale <= 0:
            raise ValueError("cfg_scale must be greater than zero")
        if (self.reference_audio is None) != (not self.reference_text.strip()):
            raise ValueError(
                "reference_audio and reference_text must be provided together"
            )


class BreezeTTS:
    """X-Talk-compatible client for a separately hosted Breeze TTS 2 server.

    Model loading remains outside the client process. Clones share a semaphore
    because the official server currently accepts one inference request at a
    time, while each clone owns its own HTTP connection pool.
    """

    def __init__(
        self,
        config: BreezeTTSConfig | None = None,
        profile: BreezeVoiceProfile | None = None,
        *,
        session: Any | None = None,
        request_gate: asyncio.Semaphore | None = None,
    ) -> None:
        self.config = config if config is not None else BreezeTTSConfig()
        self.profile = profile if profile is not None else BreezeVoiceProfile()
        self.profile.validate()
        if self.config.sample_rate != 24_000 or self.config.channels != 1:
            raise ValueError("Breeze TTS 2 output is fixed at 24 kHz mono PCM16")
        if self.config.frame_bytes < 2 or self.config.frame_bytes % 2:
            raise ValueError("frame size must contain aligned PCM16 samples")
        if self.config.busy_retry_attempts < 0:
            raise ValueError("busy retry attempts cannot be negative")
        if self.config.busy_retry_delay_seconds < 0:
            raise ValueError("busy retry delay cannot be negative")
        self._session = session
        self._owns_session = session is None
        self._request_gate = (
            request_gate if request_gate is not None else asyncio.Semaphore(1)
        )
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._active_responses: set[Any] = set()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def clone(self) -> "BreezeTTS":
        return BreezeTTS(
            self.config,
            self.profile,
            request_gate=self._request_gate,
        )

    def set_voice(self, *, voice_names: list[str]) -> None:
        voices = {
            "zh": ("reference-zh.wav", "这是中文参考音频的准确文字稿"),
            "en": ("reference-en.wav", "This is accurate transcript for the English audio."),
        }
        if len(voice_names) != 1 or voice_names[0] not in voices:
            raise ValueError("voice must be zh or en")
        filename, transcript = voices[voice_names[0]]
        path = Path(__file__).resolve().parents[4] / "models" / "breeze-tts" / filename
        profile = replace(
            self.profile,
            reference_audio=path.read_bytes(),
            reference_text=transcript,
            reference_filename=filename,
        )
        profile.validate()
        self.profile = profile

    def synthesize(self, text: str) -> bytes:
        raise RuntimeError("BreezeTTS is streaming and async-only")

    async def async_synthesize(self, text: str, **kwargs: Any) -> bytes:
        chunks = [chunk async for chunk in self.async_synthesize_stream(text, **kwargs)]
        return b"".join(chunks)

    async def async_synthesize_stream(self, text: str, **kwargs: Any):
        if not self.config.trim_edge_silence:
            async for chunk in self._raw_synthesize_stream(text, **kwargs):
                yield chunk
            return
        # Buffer silence to preserve internal pauses; retain 80 ms at the end.
        pending = bytearray()
        started = False
        frame_bytes = self.config.frame_bytes
        async for chunk in self._raw_synthesize_stream(text, **kwargs):
            for offset in range(0, len(chunk), frame_bytes):
                frame = chunk[offset:offset + frame_bytes]
                values = np.frombuffer(frame, dtype='<i2').astype(np.float32)
                quiet = float(np.sqrt(np.mean(values * values))) < 184.0
                if quiet:
                    pending.extend(frame)
                    if not started:
                        pending = pending[-frame_bytes:]
                    continue
                if pending:
                    yield bytes(pending)
                    pending.clear()
                yield frame
                started = True
        if started and pending:
            tail_bytes = self.config.sample_rate * 2 * 80 // 1000
            yield bytes(pending[:tail_bytes])
        if not started:
            raise BreezeTTSError('Breeze synthesis contained no audible speech')

    async def _raw_synthesize_stream(self, text: str, **kwargs: Any):
        text = text.strip()
        if not text:
            raise ValueError("text must be non-empty")
        profile = self._resolve_profile(kwargs)
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks.add(task)
        try:
            async with self._request_gate:
                session = await self._get_session()
                for attempt in range(self.config.busy_retry_attempts + 1):
                    form = self._build_form(text, profile)
                    async with session.post(
                        self.config.speech_url, data=form
                    ) as response:
                        if response.status == 409:
                            detail = (await response.text())[:500]
                            if attempt < self.config.busy_retry_attempts:
                                await asyncio.sleep(
                                    self.config.busy_retry_delay_seconds
                                )
                                continue
                            raise BreezeTTSBusyError(
                                f"Breeze HTTP 409: {detail}"
                            )
                        if response.status != 200:
                            detail = (await response.text())[:500]
                            raise BreezeTTSError(
                                f"Breeze HTTP {response.status}: {detail}"
                            )
                        self._validate_response(response)
                        self._active_responses.add(response)
                        try:
                            async for chunk in self._aligned_chunks(response):
                                yield chunk
                        finally:
                            self._active_responses.discard(response)
                        return
        finally:
            if task is not None:
                self._active_tasks.discard(task)

    async def cancel(self) -> None:
        """Cancel this session's local streams and close their HTTP responses."""
        current = asyncio.current_task()
        tasks = [task for task in self._active_tasks if task is not current]
        for response in tuple(self._active_responses):
            response.close()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def aclose(self) -> None:
        await self.cancel()
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> Any:
        if self._session is None:
            import aiohttp

            timeout = aiohttp.ClientTimeout(
                total=self.config.request_timeout_seconds
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    def _resolve_profile(self, overrides: dict[str, Any]) -> BreezeVoiceProfile:
        allowed = {
            "instruction",
            "cfg_scale",
            "reference_audio",
            "reference_text",
            "reference_filename",
            "seed",
        }
        unknown = set(overrides) - allowed
        if unknown:
            raise TypeError(f"unsupported Breeze options: {sorted(unknown)}")
        values = {
            "instruction": self.profile.instruction,
            "cfg_scale": self.profile.cfg_scale,
            "reference_audio": self.profile.reference_audio,
            "reference_text": self.profile.reference_text,
            "reference_filename": self.profile.reference_filename,
            "seed": self.profile.seed,
        }
        values.update(overrides)
        profile = BreezeVoiceProfile(**values)
        profile.validate()
        return profile

    @staticmethod
    def _build_form(text: str, profile: BreezeVoiceProfile) -> Any:
        fields: dict[str, Any] = {
            "text": text,
            "cfg_scale": str(profile.cfg_scale),
            "seed": str(profile.seed),
        }
        if profile.instruction:
            fields["instruction"] = profile.instruction.strip()
        if profile.reference_audio is None:
            return fields

        import aiohttp

        form = aiohttp.FormData()
        for name, value in fields.items():
            form.add_field(name, value)
        form.add_field("ref_text", profile.reference_text.strip())
        form.add_field(
            "ref_audio",
            profile.reference_audio,
            filename=Path(profile.reference_filename).name,
            content_type="application/octet-stream",
        )
        return form

    def _validate_response(self, response: Any) -> None:
        content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
        if content_type != "audio/pcm":
            raise BreezeTTSError(f"unexpected Breeze content type: {content_type}")
        raw_sample_rate = response.headers.get("X-Sample-Rate")
        if raw_sample_rate is not None:
            try:
                sample_rate = int(raw_sample_rate)
            except ValueError as exc:
                raise BreezeTTSError("invalid Breeze sample-rate header") from exc
            if sample_rate != self.sample_rate:
                raise BreezeTTSError(
                    f"Breeze returned {sample_rate} Hz; expected {self.sample_rate} Hz"
                )
        if response.headers.get("X-Sample-Format", "s16le") != "s16le":
            raise BreezeTTSError("Breeze response is not PCM16 little-endian")

    async def _aligned_chunks(self, response: Any):
        pending = bytearray()
        frame_bytes = self.config.frame_bytes
        async for chunk in response.content.iter_chunked(frame_bytes):
            if not chunk:
                continue
            pending.extend(chunk)
            while len(pending) >= frame_bytes:
                yield bytes(pending[:frame_bytes])
                del pending[:frame_bytes]
        if len(pending) % 2:
            raise BreezeTTSError("Breeze response ended with an incomplete PCM16 sample")
        if pending:
            yield bytes(pending)
