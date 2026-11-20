"""Per-session X-Live composition helpers for the X-Talk serving stack."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..frontend_model import FrontendModelAdapter
from ..runtime.session import RealtimeSessionCore
from ..agent.runtime import AgentBackend, AgentToolRuntime
from ..agent.task_hub import TaskHubRegistry
from ..speech import (
    StreamingASRAdapter,
    TurnStateAdapter,
    XLiveSpeechManagerConfig,
)
from ..speech.config import (
    ASR_ADAPTER_CONFIG_KEY,
    SPEECH_MANAGER_CONFIG_KEY,
    TURN_ADAPTER_CONFIG_KEY,
)
from ..speech.text_normalizer import normalize_text_for_speech


XLIVE_CORE_CONFIG_KEY = "xlive_core"
OUTPUT_CONTROLLER_CONFIG_KEY = "output_controller"
XLIVE_MODEL_ADAPTER_CONFIG_KEY = "xlive_model_adapter"
CONVERSATION_HISTORY_ADAPTER_CONFIG_KEY = "conversation_history_adapter"
TASK_HUB_CONFIG_KEY = "xlive_task_hub"
LESSON_PREFETCH_DELAY_CONFIG_KEY = "xlive_lesson_prefetch_delay_seconds"
TTS_TEXT_TRANSFORM_CONFIG_KEY = "tts_text_transform"


class XLiveSessionConfigFactory:
    """Create the X-Live-owned runtime objects for one live X-Talk session.

    The factory is synchronous because X-Talk constructs managers synchronously.
    Backends may open connections lazily in their asynchronous methods.
    """

    def __init__(
        self,
        agent_backend_factory: Callable[[str], AgentBackend],
        *,
        event_capacity: int = 1_000,
        frontend_model_factory: Callable[[str], FrontendModelAdapter] | None = None,
        asr_adapter_factory: Callable[[str], StreamingASRAdapter] | None = None,
        turn_adapter_factory: Callable[[str], TurnStateAdapter] | None = None,
        speech_manager_config: XLiveSpeechManagerConfig | None = None,
        task_hub_registry: TaskHubRegistry | None = None,
        lesson_prefetch_delay_seconds: float = 5.0,
    ) -> None:
        if (asr_adapter_factory is None) != (turn_adapter_factory is None):
            raise ValueError(
                "asr_adapter_factory and turn_adapter_factory must be configured together"
            )
        self._agent_backend_factory = agent_backend_factory
        self._event_capacity = event_capacity
        self._frontend_model_factory = frontend_model_factory
        self._asr_adapter_factory = asr_adapter_factory
        self._turn_adapter_factory = turn_adapter_factory
        self._speech_manager_config = speech_manager_config
        self._task_hub_registry = task_hub_registry
        if lesson_prefetch_delay_seconds < 0:
            raise ValueError("lesson_prefetch_delay_seconds cannot be negative")
        self._lesson_prefetch_delay_seconds = lesson_prefetch_delay_seconds

    def __call__(self, session_id: str) -> dict[str, Any]:
        hub = (
            self._task_hub_registry.get_or_create(session_id)
            if self._task_hub_registry is not None
            else None
        )
        backend = self._agent_backend_factory(session_id) if hub is None else None
        agent_tools = AgentToolRuntime(backend) if hub is None else hub.runtime
        core = RealtimeSessionCore(
            session_id,
            agent_tools,
            event_capacity=self._event_capacity,
        )
        config = {
            XLIVE_CORE_CONFIG_KEY: core,
            OUTPUT_CONTROLLER_CONFIG_KEY: core.output,
            LESSON_PREFETCH_DELAY_CONFIG_KEY: self._lesson_prefetch_delay_seconds,
            TTS_TEXT_TRANSFORM_CONFIG_KEY: normalize_text_for_speech,
        }
        if hub is not None:
            config[TASK_HUB_CONFIG_KEY] = hub
        if self._frontend_model_factory is not None:
            frontend_model = self._frontend_model_factory(session_id)
            config[XLIVE_MODEL_ADAPTER_CONFIG_KEY] = frontend_model
            config[CONVERSATION_HISTORY_ADAPTER_CONFIG_KEY] = frontend_model
        if self._asr_adapter_factory is not None:
            assert self._turn_adapter_factory is not None
            config.update(
                {
                    ASR_ADAPTER_CONFIG_KEY: self._asr_adapter_factory(session_id),
                    TURN_ADAPTER_CONFIG_KEY: self._turn_adapter_factory(session_id),
                    SPEECH_MANAGER_CONFIG_KEY: (
                        self._speech_manager_config or XLiveSpeechManagerConfig()
                    ),
                }
            )
        return config


def get_xlive_core(config: Mapping[str, Any]) -> RealtimeSessionCore:
    """Return the session core from a live X-Talk service configuration."""

    core = config.get(XLIVE_CORE_CONFIG_KEY)
    if not isinstance(core, RealtimeSessionCore):
        raise RuntimeError("X-Talk service is missing its X-Live session core")
    return core
