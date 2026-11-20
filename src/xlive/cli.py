"""Command-line entrypoint for validating and serving X-Live."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

import uvicorn

from .application import (
    XLiveConfigurationError,
    XLiveServerConfig,
    create_application_from_env,
    load_application_components,
)
from .config_values import ConfigurationValueError
from .frontend_model import FrontendModelConfig
from .speech import QwenASRConfig, SmartTurnConfig
from .speech.tts import BreezeTTSConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="xlive")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "check", help="validate local configuration without contacting services"
    )
    subcommands.add_parser("serve", help="start the X-Live HTTP/WebSocket server")
    args = parser.parse_args(argv)

    try:
        config = XLiveServerConfig.from_env()
        if args.command == "check":
            return _check(config)
        return _serve(config)
    except ConfigurationValueError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2


def _check(config: XLiveServerConfig) -> int:
    if not config.components_factory:
        raise XLiveConfigurationError("XLIVE_COMPONENTS_FACTORY is required")
    load_application_components(config.components_factory)
    frontend = FrontendModelConfig.from_env()
    breeze = BreezeTTSConfig.from_env()
    asr = QwenASRConfig.from_env()
    turn_url = SmartTurnConfig.from_env().server_url

    print("X-Live client configuration is valid.")
    print(f"  listen: {config.host}:{config.port}")
    print(f"  frontend model: {frontend.served_model_name}")
    print(f"  frontend backend: {frontend.backend}")
    print(f"  frontend API: {frontend.api_base}")
    print(f"  Breeze TTS: {breeze.base_url}")
    print(f"  Qwen3-ASR: {asr.server_url}")
    print(f"  semantic VAD: smart_turn ({turn_url})")
    print(f"  persistence: {'enabled' if config.enable_persistence else 'disabled'}")
    print("  latency metrics: enabled")
    if config.auth_secret is None:
        print(
            "  warning: XTALK_AUTH_SECRET is unset; tokens will not survive restart",
            file=sys.stderr,
        )
    return 0


def _serve(config: XLiveServerConfig) -> int:
    if not config.components_factory:
        raise XLiveConfigurationError("XLIVE_COMPONENTS_FACTORY is required")
    app = create_application_from_env()
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
