"""X-Live-owned integration with the X-Talk serving framework."""

from .composition import XLiveSessionConfigFactory, get_xlive_core
from .modules.model_runner import XLiveModelRunner
from .events import XLiveInterruptionRequested, XLiveTaskBackendEventReceived
from .modules.interaction_manager import XLiveInteractionManager
from .modules.output_bridge import (
    XLiveGenerationCancelRequested,
    XLiveGenerationSealRequested,
    XLiveOutputBridge,
    XLiveTextAppendRequested,
    XLiveTextFlushRequested,
    XLiveTextStartRequested,
)
from .service_factory import create_xlive_service_prototype
from ..speech import XLiveSpeechManagerConfig
from .modules.speech_manager import XLiveSpeechManager

__all__ = [
    "XLiveSessionConfigFactory",
    "get_xlive_core",
    "XLiveModelRunner",
    "XLiveInteractionManager",
    "XLiveInterruptionRequested",
    "XLiveTaskBackendEventReceived",
    "XLiveGenerationCancelRequested",
    "XLiveGenerationSealRequested",
    "XLiveOutputBridge",
    "XLiveTextAppendRequested",
    "XLiveTextFlushRequested",
    "XLiveTextStartRequested",
    "create_xlive_service_prototype",
    "XLiveSpeechManager",
    "XLiveSpeechManagerConfig",
]
