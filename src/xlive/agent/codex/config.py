from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from ...config_values import (
    parse_boolean,
    parse_positive_float,
    parse_positive_int,
)


@dataclass(frozen=True, slots=True)
class CodexBackendConfig:
    """Deployment settings for one Codex-backed X-Live task worker."""

    workspace: str
    codex_bin: str = "codex"
    model: str | None = None
    reasoning_effort: str = "medium"
    sandbox: str = "workspace-write"
    web_search_enabled: bool = True
    approve_for_me: bool = False
    max_concurrency: int = 1
    task_timeout_seconds: float = 1800.0
    result_max_chars: int = 12000
    codex_home: str | None = None

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None
    ) -> "CodexBackendConfig":
        values = os.environ if environ is None else environ
        config = cls(
            workspace=values.get("XLIVE_CODEX_WORKSPACE", os.getcwd()),
            codex_bin=values.get("XLIVE_CODEX_BIN", "codex"),
            model=values.get("XLIVE_CODEX_MODEL") or None,
            reasoning_effort=values.get(
                "XLIVE_CODEX_REASONING_EFFORT", "medium"
            ),
            sandbox=values.get("XLIVE_CODEX_SANDBOX", "workspace-write"),
            web_search_enabled=parse_boolean(
                values.get("XLIVE_CODEX_WEB_SEARCH_ENABLED", "true"),
                "XLIVE_CODEX_WEB_SEARCH_ENABLED",
            ),
            approve_for_me=parse_boolean(
                values.get("XLIVE_CODEX_APPROVE_FOR_ME", "false"),
                "XLIVE_CODEX_APPROVE_FOR_ME",
            ),
            max_concurrency=parse_positive_int(
                values.get("XLIVE_CODEX_MAX_CONCURRENCY", "1"),
                "XLIVE_CODEX_MAX_CONCURRENCY",
            ),
            task_timeout_seconds=parse_positive_float(
                values.get("XLIVE_CODEX_TASK_TIMEOUT_SECONDS", "1800"),
                "XLIVE_CODEX_TASK_TIMEOUT_SECONDS",
            ),
            result_max_chars=parse_positive_int(
                values.get("XLIVE_CODEX_RESULT_MAX_CHARS", "12000"),
                "XLIVE_CODEX_RESULT_MAX_CHARS",
            ),
            codex_home=values.get("XLIVE_CODEX_HOME") or None,
        )
        config.validate()
        return config

    def validate(self) -> None:
        workspace = Path(self.workspace).expanduser()
        if not workspace.is_dir():
            raise ValueError(f"Codex workspace does not exist: {workspace}")
        if not self.codex_bin:
            raise ValueError("XLIVE_CODEX_BIN cannot be empty")
        if "/" in self.codex_bin:
            if not os.access(Path(self.codex_bin).expanduser(), os.X_OK):
                raise ValueError(f"Codex executable is not usable: {self.codex_bin}")
        elif shutil.which(self.codex_bin) is None:
            raise ValueError(f"Codex executable was not found: {self.codex_bin}")
        if self.sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError(
                "XLIVE_CODEX_SANDBOX must be read-only, workspace-write or danger-full-access"
            )
        if self.sandbox == "danger-full-access" and self.approve_for_me:
            raise ValueError("Full access cannot be combined with approve-for-me")
        if not self.reasoning_effort:
            raise ValueError("XLIVE_CODEX_REASONING_EFFORT cannot be empty")
        if self.max_concurrency < 1:
            raise ValueError("XLIVE_CODEX_MAX_CONCURRENCY must be positive")
        if self.task_timeout_seconds <= 0:
            raise ValueError("XLIVE_CODEX_TASK_TIMEOUT_SECONDS must be positive")
        if self.result_max_chars < 1:
            raise ValueError("XLIVE_CODEX_RESULT_MAX_CHARS must be positive")
