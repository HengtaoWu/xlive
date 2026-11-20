"""Production application assembly for the X-Live WebSocket runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import importlib
import os
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import XLiveAPI
from .config_values import (
    ConfigurationValueError,
    parse_boolean,
    parse_nonnegative_float,
    parse_nonnegative_int,
    parse_optional_positive_int,
    parse_port,
)
from .speech.tts import (
    BreezeTTS,
    BreezeTTSConfig,
    BreezeVoiceProfile,
)

from .agent import AgentBackend, TaskHubRegistry
from .frontend_model import (
    FrontendModelAdapter,
    FrontendModelConfig,
    create_frontend_model_adapter,
)
from .serving import create_xlive_service_prototype
from .speech import (
    XLiveSpeechManagerConfig,
    SmartTurnConfig,
    SmartTurnWebSocketAdapter,
    QwenASRConfig,
    QwenASRWebSocketAdapter,
)


class XLiveConfigurationError(ConfigurationValueError):
    """Raised when the process cannot assemble a real X-Live runtime."""


@dataclass(frozen=True, slots=True)
class XLiveApplicationComponents:
    """Session-scoped dependencies whose concrete protocols are still external.

    X-Live owns transport, speech, and the default frontend-model adapter. The
    routed Agent backend remains deployment-owned. A custom frontend factory is
    accepted only as an explicit override for tests or alternate deployments.
    """

    agent_backend_factory: Callable[[str], AgentBackend]
    frontend_model_factory: Callable[[str], FrontendModelAdapter] | None = None
    breeze_voice_profile: BreezeVoiceProfile | None = None

    def validate(self) -> None:
        if self.breeze_voice_profile is not None:
            try:
                self.breeze_voice_profile.validate()
            except ValueError as exc:
                raise XLiveConfigurationError(str(exc)) from exc


@dataclass(frozen=True, slots=True)
class XLiveServerConfig:
    """Process and X-Talk service settings loaded from environment variables."""

    host: str = "127.0.0.1"
    port: int = 8001
    log_level: str = "info"
    max_sessions: int | None = None
    enable_persistence: bool = True
    data_dir: str = "data"
    auth_secret: str | None = None
    components_factory: str | None = None
    cors_origins: tuple[str, ...] = ()
    lesson_prefetch_delay_seconds: float = 5.0
    semantic_endpoint_grace_ms: int = 500

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "XLiveServerConfig":
        values = os.environ if environ is None else environ
        try:
            config = cls(
                host=values.get("XLIVE_HOST", "127.0.0.1").strip(),
                port=parse_port(values.get("XLIVE_PORT", "8001"), "XLIVE_PORT"),
                log_level=values.get("XLIVE_LOG_LEVEL", "info").strip().lower(),
                max_sessions=parse_optional_positive_int(
                    values.get("XLIVE_MAX_SESSIONS"), "XLIVE_MAX_SESSIONS"
                ),
                enable_persistence=parse_boolean(
                    values.get("XLIVE_ENABLE_PERSISTENCE", "true"),
                    "XLIVE_ENABLE_PERSISTENCE",
                ),
                data_dir=values.get("XLIVE_DATA_DIR", "data").strip(),
                auth_secret=values.get("XTALK_AUTH_SECRET") or None,
                components_factory=values.get("XLIVE_COMPONENTS_FACTORY") or None,
                cors_origins=tuple(
                    origin.strip()
                    for origin in values.get("XLIVE_CORS_ORIGINS", "").split(",")
                    if origin.strip()
                ),
                lesson_prefetch_delay_seconds=parse_nonnegative_float(
                    values.get("XLIVE_LESSON_PREFETCH_DELAY_SECONDS", "5.0"),
                    "XLIVE_LESSON_PREFETCH_DELAY_SECONDS",
                ),
                semantic_endpoint_grace_ms=parse_nonnegative_int(
                    values.get("XLIVE_SEMANTIC_ENDPOINT_GRACE_MS", "500"),
                    "XLIVE_SEMANTIC_ENDPOINT_GRACE_MS",
                ),
            )
        except ConfigurationValueError as exc:
            raise XLiveConfigurationError(str(exc)) from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not self.host:
            raise XLiveConfigurationError("XLIVE_HOST cannot be empty")
        if not 1 <= self.port <= 65_535:
            raise XLiveConfigurationError("XLIVE_PORT must be between 1 and 65535")
        if self.log_level not in {"critical", "error", "warning", "info", "debug"}:
            raise XLiveConfigurationError(
                "XLIVE_LOG_LEVEL must be critical, error, warning, info, or debug"
            )
        if self.max_sessions is not None and self.max_sessions < 1:
            raise XLiveConfigurationError("XLIVE_MAX_SESSIONS must be positive")
        if not self.data_dir:
            raise XLiveConfigurationError("XLIVE_DATA_DIR cannot be empty")

    def service_config(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "enable_persistence": self.enable_persistence,
            "data_dir": self.data_dir,
        }
        if self.auth_secret is not None:
            result["auth_secret"] = self.auth_secret
        return result


class XLivePipeline:
    """Small X-Talk pipeline containing only X-Live's selected TTS engine."""

    def __init__(self, tts: Any) -> None:
        self._tts = tts

    def clone(self) -> "XLivePipeline":
        return type(self)(self._tts.clone())

    def get_tts_model(self) -> Any:
        return self._tts

    def get_speed_controller(self) -> None:
        return None


def load_application_components(spec: str) -> XLiveApplicationComponents:
    """Load ``module:callable`` and materialize the external components factory."""

    module_name, separator, attribute_name = spec.strip().partition(":")
    if not separator or not module_name or not attribute_name:
        raise XLiveConfigurationError(
            "XLIVE_COMPONENTS_FACTORY must use module:callable syntax"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise XLiveConfigurationError(
            f"cannot import component module {module_name!r}: {exc}"
        ) from exc
    factory = getattr(module, attribute_name, None)
    if factory is None:
        raise XLiveConfigurationError(f"component factory {spec!r} does not exist")
    try:
        components = factory()
    except ValueError as exc:
        raise XLiveConfigurationError(str(exc)) from exc
    if not isinstance(components, XLiveApplicationComponents):
        raise XLiveConfigurationError(
            f"component factory {spec!r} must return XLiveApplicationComponents"
        )
    components.validate()
    return components


def create_application(
    components: XLiveApplicationComponents,
    *,
    server_config: XLiveServerConfig | None = None,
    breeze_config: BreezeTTSConfig | None = None,
    qwen_asr_config: QwenASRConfig | None = None,
    smart_turn_config: SmartTurnConfig | None = None,
) -> Any:
    """Build the real FastAPI/X-Talk application without opening upstream I/O.

    External model, ASR, turn and TTS connections remain lazy and are created
    only for live sessions. Importing this module and running ``xlive check``
    therefore never contacts a model service.
    """

    components.validate()
    config = XLiveServerConfig.from_env() if server_config is None else server_config
    config.validate()

    tts_config = BreezeTTSConfig.from_env() if breeze_config is None else breeze_config
    asr_config = QwenASRConfig.from_env() if qwen_asr_config is None else qwen_asr_config
    turn_config = (
        SmartTurnConfig.from_env() if smart_turn_config is None else smart_turn_config
    )
    turn_adapter_factory = lambda session_id: SmartTurnWebSocketAdapter(
        session_id, turn_config
    )
    pipeline = XLivePipeline(
        BreezeTTS(tts_config, profile=components.breeze_voice_profile)
    )
    frontend_config = FrontendModelConfig.from_env()
    frontend_model_factory = components.frontend_model_factory or (
        lambda _session_id: create_frontend_model_adapter(frontend_config)
    )
    task_hubs = TaskHubRegistry(components.agent_backend_factory)
    service = create_xlive_service_prototype(
        pipeline=pipeline,
        agent_backend_factory=components.agent_backend_factory,
        frontend_model_factory=frontend_model_factory,
        asr_adapter_factory=lambda _session_id: QwenASRWebSocketAdapter(asr_config),
        turn_adapter_factory=turn_adapter_factory,
        service_config=config.service_config(),
        task_hub_registry=task_hubs,
        lesson_prefetch_delay_seconds=config.lesson_prefetch_delay_seconds,
        speech_manager_config=XLiveSpeechManagerConfig(
            semantic_endpoint_grace_ms=config.semantic_endpoint_grace_ms,
        ),
    )
    runtime = XLiveAPI(
        service_prototype=service,
        max_sessions=config.max_sessions,
        session_delete_callback=task_hubs.remove,
    )
    app = FastAPI(title="X-Live", version="0.1.0")
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    @app.get("/healthz")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    runtime.mount_routes(app)
    app.state.xlive = runtime
    app.state.xlive_task_hubs = task_hubs
    app.router.add_event_handler("shutdown", task_hubs.aclose)
    return app


def create_application_from_env(
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Create an application from environment configuration."""

    config = XLiveServerConfig.from_env(environ)
    if not config.components_factory:
        raise XLiveConfigurationError("XLIVE_COMPONENTS_FACTORY is required")
    components = load_application_components(config.components_factory)
    return create_application(components, server_config=config)
