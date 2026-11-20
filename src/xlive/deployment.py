"""Built-in deployment factories for the local X-Live."""

from __future__ import annotations

import asyncio

from .agent import CodexAgentBackend, CodexBackendConfig
from .application import XLiveApplicationComponents


def create_codex_components() -> XLiveApplicationComponents:
    """Create the single Codex backend used by every delegated task."""

    config = CodexBackendConfig.from_env()
    concurrency_gate = asyncio.Semaphore(config.max_concurrency)
    return XLiveApplicationComponents(
        agent_backend_factory=lambda _session_id: CodexAgentBackend(
            config,
            concurrency_gate=concurrency_gate,
        )
    )
