import importlib
from typing import AsyncIterator, Protocol


class ModelFailure(RuntimeError):
    pass


class DuplexAgent(Protocol):
    async def start(self, context: dict) -> None: ...
    async def send_audio(self, pcm: bytes) -> None: ...
    def events(self) -> AsyncIterator[dict]: ...
    async def tool_result(self, call_id: str, result: dict) -> None: ...
    async def backend_event(self, event: dict) -> None: ...
    async def playback(self, event: dict) -> None: ...
    async def aclose(self) -> None: ...


def load_callable(spec):
    module, separator, name = spec.partition(':')
    if not separator or not module or not name:
        raise ValueError('Expected module:callable')
    value = getattr(importlib.import_module(module), name)
    if not callable(value):
        raise TypeError(f'Not callable: {spec}')
    return value
