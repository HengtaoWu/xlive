"""Public engineering boundary for the trained X-Live frontend model."""

from .config import FrontendModelConfig
from .adapter import XLiveFrontendModelAdapter, FrontendModelProtocolError
from .dual_adapter import (
    XLiveDualFrontendModelAdapter,
    create_frontend_model_adapter,
)
from .contract import (
    FrontendModelAdapter,
    InvalidModelOutput,
    ModelOutput,
    ModelOutputKind,
    NativeToolCall,
    NativeToolName,
    validate_model_output,
)

__all__ = [
    "FrontendModelConfig",
    "FrontendModelAdapter",
    "XLiveFrontendModelAdapter",
    "XLiveDualFrontendModelAdapter",
    "create_frontend_model_adapter",
    "FrontendModelProtocolError",
    "InvalidModelOutput",
    "ModelOutput",
    "ModelOutputKind",
    "NativeToolCall",
    "NativeToolName",
    "validate_model_output",
]
