"""Session configuration shared by X-Live speech adapters and serving."""

from __future__ import annotations

from dataclasses import dataclass


ASR_ADAPTER_CONFIG_KEY = "xlive_asr_adapter"
TURN_ADAPTER_CONFIG_KEY = "xlive_turn_adapter"
SPEECH_MANAGER_CONFIG_KEY = "xlive_speech_manager_config"


@dataclass(frozen=True, slots=True)
class XLiveSpeechManagerConfig:
    """Deterministic buffering parameters for ASR and semantic endpointing."""

    sample_rate: int = 16_000
    asr_chunk_ms: int = 500
    prebuffer_ms: int = 320
    semantic_max_pause_ms: int = 2_500
    semantic_endpoint_grace_ms: int = 500

    def __post_init__(self) -> None:
        if self.sample_rate != 16_000:
            raise ValueError(
                "Qwen3-ASR and semantic VAD integration requires 16 kHz"
            )
        if self.asr_chunk_ms <= 0:
            raise ValueError("asr_chunk_ms must be positive")
        if self.prebuffer_ms < 0:
            raise ValueError("prebuffer_ms must be non-negative")
        if self.semantic_max_pause_ms <= 0:
            raise ValueError("semantic_max_pause_ms must be positive")
        if self.semantic_endpoint_grace_ms < 0:
            raise ValueError("semantic_endpoint_grace_ms must be non-negative")
