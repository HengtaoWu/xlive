from __future__ import annotations


class ConfigurationValueError(ValueError):
    """A configuration string does not match its expected form."""


def parse_port(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationValueError(f"{name} must be an integer") from exc
    if not 1 <= value <= 65_535:
        raise ConfigurationValueError(f"{name} must be between 1 and 65535")
    return value


def parse_optional_positive_int(raw: str | None, name: str) -> int | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ConfigurationValueError(f"{name} must be positive")
    return value


def parse_positive_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationValueError(f"{name} must be an integer") from exc
    if value < 1:
        raise ConfigurationValueError(f"{name} must be positive")
    return value


def parse_nonnegative_int(raw: str, name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigurationValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ConfigurationValueError(f"{name} cannot be negative")
    return value


def parse_boolean(raw: str, name: str) -> bool:
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationValueError(f"{name} must be a boolean")


def parse_nonnegative_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationValueError(f"{name} must be a number") from exc
    if value < 0:
        raise ConfigurationValueError(f"{name} cannot be negative")
    return value


def parse_positive_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigurationValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ConfigurationValueError(f"{name} must be positive")
    return value
