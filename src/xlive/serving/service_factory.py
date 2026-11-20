"""Minimal X-Talk service composition owned by X-Live."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..runtime.manager import Manager
from .modules.latency_manager import LatencyManager
from .modules.queued_tts_manager import QueuedTTSManager
from .service import Service

from .composition import XLiveSessionConfigFactory
from .modules.interaction_manager import XLiveInteractionManager
from .modules.output_bridge import XLiveOutputBridge
from ..agent.runtime import AgentBackend
from ..agent.task_hub import TaskHubRegistry
from ..frontend_model import FrontendModelAdapter
from ..speech import (
    StreamingASRAdapter,
    TurnStateAdapter,
    XLiveSpeechManagerConfig,
)
from .modules.speech_manager import XLiveSpeechManager


def create_xlive_service_prototype(
    *,
    pipeline: Any,
    agent_backend_factory: Callable[[str], AgentBackend],
    service_config: dict[str, Any] | None = None,
    extra_manager_classes: Sequence[type[Manager]] = (),
    event_capacity: int = 1_000,
    frontend_model_factory: Callable[[str], FrontendModelAdapter] | None = None,
    asr_adapter_factory: Callable[[str], StreamingASRAdapter] | None = None,
    turn_adapter_factory: Callable[[str], TurnStateAdapter] | None = None,
    speech_manager_config: XLiveSpeechManagerConfig | None = None,
    task_hub_registry: TaskHubRegistry | None = None,
    lesson_prefetch_delay_seconds: float = 5.0,
) -> Service:
    """Build the service prototype; clones create per-session runtime objects.

    ASR and turn detection factories must be supplied together."""

    manager_classes: list[type[Manager]] = [XLiveOutputBridge, QueuedTTSManager]
    if frontend_model_factory is not None:
        manager_classes.insert(0, XLiveInteractionManager)
    if asr_adapter_factory is not None:
        manager_classes.append(XLiveSpeechManager)
    manager_classes.append(LatencyManager)
    for manager_class in extra_manager_classes:
        if manager_class not in manager_classes:
            manager_classes.append(manager_class)

    return Service(
        pipeline=pipeline,
        service_config=service_config,
        manager_classes=manager_classes,
        session_config_factory=XLiveSessionConfigFactory(
            agent_backend_factory,
            event_capacity=event_capacity,
            frontend_model_factory=frontend_model_factory,
            asr_adapter_factory=asr_adapter_factory,
            turn_adapter_factory=turn_adapter_factory,
            speech_manager_config=speech_manager_config,
            task_hub_registry=task_hub_registry,
            lesson_prefetch_delay_seconds=lesson_prefetch_delay_seconds,
        ),
    )
