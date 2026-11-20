"""Configuration values for the X-Live frontend model serving boundary."""

from dataclasses import dataclass
import os
from pathlib import Path

from ..config_values import (
    ConfigurationValueError,
    parse_nonnegative_float,
    parse_positive_float,
    parse_positive_int,
)


_DEFAULT_ARTIFACTS_DIR = (
    Path(__file__).resolve().parents[3] / "models" / "x-live-dual"
)


@dataclass(frozen=True, slots=True)
class FrontendModelConfig:
    """Frontend endpoint and generation settings.

    base_model identifies the training source; served_model_name identifies the endpoint model."""

    base_model: str = "HengtaoWu/X-Live-Base"
    served_model_name: str = "x-live-dual"
    api_base: str = "http://127.0.0.1:8002/v1"
    api_key: str | None = None
    artifacts_dir: Path = _DEFAULT_ARTIFACTS_DIR
    temperature: float = 0.3
    max_tokens: int = 512
    timeout_seconds: float = 120.0
    max_tool_rounds: int = 4
    backend: str = "dual"
    dual_event_path: Path = Path("logs/xlive-dual-tool-events.jsonl")
    dual_event_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.backend != "dual":
            raise ConfigurationValueError(
                "XLIVE_MODEL_BACKEND must be 'dual', "
                f"got {self.backend!r}"
            )

    @classmethod
    def from_env(cls) -> "FrontendModelConfig":
        backend = os.getenv("XLIVE_MODEL_BACKEND", "dual").strip().lower()
        artifacts = os.getenv("XLIVE_MODEL_ARTIFACTS")
        return cls(
            base_model=os.getenv("XLIVE_BASE_MODEL", "HengtaoWu/X-Live-Base"),
            served_model_name=os.getenv("XLIVE_SERVED_MODEL", "x-live-dual"),
            api_base=os.getenv("XLIVE_MODEL_API_BASE", "http://127.0.0.1:8002/v1"),
            api_key=os.getenv("XLIVE_MODEL_API_KEY") or None,
            artifacts_dir=(
                _DEFAULT_ARTIFACTS_DIR
                if not artifacts
                else Path(artifacts).expanduser()
            ),
            temperature=parse_nonnegative_float(
                os.getenv("XLIVE_MODEL_TEMPERATURE", "0.3"),
                "XLIVE_MODEL_TEMPERATURE",
            ),
            max_tokens=parse_positive_int(
                os.getenv("XLIVE_MODEL_MAX_TOKENS", "512"),
                "XLIVE_MODEL_MAX_TOKENS",
            ),
            timeout_seconds=parse_positive_float(
                os.getenv("XLIVE_MODEL_TIMEOUT_SECONDS", "120"),
                "XLIVE_MODEL_TIMEOUT_SECONDS",
            ),
            max_tool_rounds=parse_positive_int(
                os.getenv("XLIVE_MODEL_MAX_TOOL_ROUNDS", "4"),
                "XLIVE_MODEL_MAX_TOOL_ROUNDS",
            ),
            backend=backend,
            dual_event_path=Path(
                os.getenv(
                    "XLIVE_DUAL_EVENT_PATH",
                    "logs/xlive-dual-tool-events.jsonl",
                )
            ).expanduser(),
            dual_event_timeout_seconds=parse_positive_float(
                os.getenv("XLIVE_DUAL_EVENT_TIMEOUT_SECONDS", "180"),
                "XLIVE_DUAL_EVENT_TIMEOUT_SECONDS",
            ),
        )
