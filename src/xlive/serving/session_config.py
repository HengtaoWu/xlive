"""Dependency-light construction of per-session service configuration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any


SessionConfigFactory = Callable[[str], dict[str, Any]]


def build_session_config(
    base: Mapping[str, Any] | None,
    session_id: str,
    *,
    factory: SessionConfigFactory | None = None,
    materialize: bool = False,
) -> dict[str, Any]:
    config = dict(base or {})
    if not config.get("data_dir"):
        config["data_dir"] = "data"
    if not materialize or factory is None:
        return config

    generated = factory(session_id)
    if not isinstance(generated, dict):
        raise TypeError("session_config_factory must return a dict")
    collisions = config.keys() & generated.keys()
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(
            f"session_config_factory cannot replace existing keys: {names}"
        )
    config.update(generated)
    return config
