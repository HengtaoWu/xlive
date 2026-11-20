"""OpenAI-compatible adapter for the X-Live frontend model."""

from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
import copy
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    import aiohttp

from ..runtime.events import EventEnvelope
from .config import FrontendModelConfig
from .contract import (
    ModelOutput,
    ModelOutputKind,
    NativeToolCall,
    NativeToolName,
)

_EXPECTED_ARGUMENTS = {
    "delegate": {"task"},
    "update": {"task_id", "instruction"},
    "cancel": {"task_id"},
    "inspect": {"task_id"},
}

class FrontendModelProtocolError(RuntimeError):
    """The model endpoint violated the X-Live frontend model wire contract."""


class XLiveFrontendModelAdapter:
    """Session-scoped vLLM adapter with canonical OpenAI chat history.

    X-Live frontend model streams natural text and the four trained native task tools.
    """

    def __init__(
        self,
        config: FrontendModelConfig,
        *,
        http_session: Any | None = None,
    ) -> None:
        artifacts = Path(config.artifacts_dir).expanduser().resolve()
        prompt_path = artifacts / "system_prompt.txt"
        tools_path = artifacts / "tools.json"
        if not prompt_path.is_file() or not tools_path.is_file():
            raise FileNotFoundError(
                "X-Live frontend model artifacts must contain system_prompt.txt and "
                f"tools.json: {artifacts}"
            )
        self.config = config
        self.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        self.tools = json.loads(tools_path.read_text(encoding="utf-8"))
        self._validate_tools()
        self._messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self._seen_input_events: set[str] = set()
        self._http = http_session
        self._owns_http = http_session is None
        self._responses: dict[str, Any] = {}
        self._history_lock = asyncio.Lock()
        self._generation_records: dict[str, list[dict[str, Any]]] = {}

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return a detached history snapshot for diagnostics and tests."""
        return copy.deepcopy(self._messages)

    def restore_history(self, messages: list[dict[str, Any]]) -> None:
        """Replace canonical chat history with persisted visible turns.

        Persistence intentionally stores only final user and assistant text.
        Task state is restored separately from the live task ledger.
        """
        restored: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"user", "assistant"}:
                continue
            if not isinstance(content, str) or not content.strip():
                continue
            restored.append({"role": role, "content": content.strip()})
        self._messages = restored
        self._seen_input_events.clear()

    def restore_tasks(self, tasks: list[dict[str, Any]]) -> None:
        """Seed durable task truth without adding user-visible chat messages."""

        for task in tasks:
            task_id = task.get("task_id")
            status = task.get("status")
            if not isinstance(task_id, str) or not task_id:
                continue
            if not isinstance(status, str) or not status:
                continue
            payload: dict[str, Any] = {
                "task_id": task_id,
                "status": status,
                "result": str(task.get("detail") or ""),
            }
            title = task.get("title")
            if isinstance(title, str) and title:
                payload["title"] = title
            instruction = task.get("instruction")
            if isinstance(instruction, str) and instruction:
                payload["instruction"] = instruction
            self._messages.append(
                {
                    "role": "tool",
                    "name": "task_event",
                    "content": json.dumps(
                        payload, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )

    async def stream(
        self,
        event: EventEnvelope,
        *,
        generation_id: str,
    ) -> AsyncIterator[ModelOutput]:
        ephemeral_prompt = await self._record_input(event)
        # Backend events are read-only model inputs.
        allow_tools = event.kind == "user_commit"
        tool_rounds = 0
        while True:
            content_parts: list[str] = []
            calls: dict[int, dict[str, str]] = {}
            async for delta in self._completion(
                generation_id,
                ephemeral_user_message=ephemeral_prompt,
                allow_tools=allow_tools,
            ):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    content_parts.append(content)
                    yield ModelOutput(ModelOutputKind.TEXT_DELTA, text=content)
                for item in delta.get("tool_calls") or []:
                    index = int(item.get("index", 0) or 0)
                    call = calls.setdefault(
                        index, {"id": "", "name": "", "arguments": ""}
                    )
                    if isinstance(item.get("id"), str):
                        call["id"] = item["id"] or call["id"]
                    function = item.get("function") or {}
                    if isinstance(function.get("name"), str):
                        call["name"] += function["name"]
                    if isinstance(function.get("arguments"), str):
                        call["arguments"] += function["arguments"]

            content = "".join(content_parts)
            if calls and not allow_tools:
                raise FrontendModelProtocolError(
                    f"tool calls are not allowed for {event.kind}"
                )
            if len(calls) > 1:
                raise FrontendModelProtocolError(
                    "parallel tool calls are not supported"
                )
            if not calls:
                if content:
                    async with self._history_lock:
                        self._append_generation_record(generation_id, {"role": "assistant", "content": content})
                return

            native_call, wire_call = self._parse_tool_call(calls[min(calls)])
            async with self._history_lock:
                self._append_generation_record(generation_id,
                    {
                        "role": "assistant",
                        "content": content or None,
                        "tool_calls": [wire_call],
                    }
                )
            yield ModelOutput(ModelOutputKind.TOOL_CALL, tool_call=native_call)
            tool_rounds += 1
            if tool_rounds >= self.config.max_tool_rounds:
                return
            # The core records the tool receipt before this generator resumes,
            # so the next request continues from a valid assistant/tool pair.

    async def on_tool_result(
        self, call: NativeToolCall, result: dict[str, Any]
    ) -> None:
        receipt: dict[str, Any] = {"task_id": result.get("task_id")}
        if call.name is NativeToolName.UPDATE:
            receipt["status"] = "updated"
        elif call.name is NativeToolName.CANCEL:
            receipt["status"] = "cancelled"
        else:
            receipt["status"] = result.get("status")
        if call.name is NativeToolName.INSPECT and result.get("detail"):
            receipt["result"] = result["detail"]
        async with self._history_lock:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.call_id,
                    "content": json.dumps(
                        receipt, ensure_ascii=False, separators=(",", ":")
                    ),
                }
            )

    async def cancel(self, generation_id: str) -> None:
        response = self._responses.pop(generation_id, None)
        if response is not None:
            response.close()

    def _append_generation_record(self, generation_id: str, record: dict[str, Any]) -> None:
        self._messages.append(record)
        self._generation_records.setdefault(generation_id, []).append(record)

    async def record_playback_outcome(self, generation_id: str, *, completed: bool, started: bool) -> None:
        async with self._history_lock:
            records = self._generation_records.pop(generation_id, [])
            if completed:
                return
            for record in records:
                if not started:
                    if record.get("tool_calls"):
                        record["content"] = None
                    else:
                        self._messages = [item for item in self._messages if item is not record]
            if started:
                self._messages.append({
                    "role": "tool", "name": "playback_status",
                    "content": "The previous spoken response was interrupted. Its generated text was only partly played; do not assume the user heard the full response. Completed background tasks remain available.",
                })

    async def ingest_context_event(self, event: EventEnvelope) -> None:
        """Add a task event to model history without starting a generation."""

        if event.kind != "task_event":
            raise FrontendModelProtocolError(
                f"only task_event can be ingested without generation: {event.kind}"
            )
        await self._record_input(event)

    async def aclose(self) -> None:
        for response in tuple(self._responses.values()):
            response.close()
        self._responses.clear()
        if self._owns_http and self._http is not None:
            await self._http.close()
        self._http = None

    async def _record_input(self, event: EventEnvelope) -> str | None:
        ephemeral_prompt: str | None = None
        async with self._history_lock:
            if event.event_id in self._seen_input_events:
                if event.kind == "task_event":
                    return (
                        "The spoken delivery of the following background result was interrupted or cancelled before completion. "
                        "The user's latest question has now been handled. Briefly deliver the still-relevant result, "
                        "without repeating unrelated dialogue: " + str(event.payload.get("detail", ""))
                    )
                return None
            if event.kind == "user_commit":
                text = event.payload.get("text")
                if not isinstance(text, str) or not text.strip():
                    raise FrontendModelProtocolError(
                        "user_commit requires non-empty text"
                    )
                self._messages.append({"role": "user", "content": text.strip()})
            elif event.kind == "task_event":
                detail = event.payload.get("detail", "")
                payload = {
                    "task_id": event.task_id,
                    "status": event.payload.get("kind", "progress"),
                    "result": str(detail),
                }
                title = event.payload.get("title")
                if isinstance(title, str) and title:
                    payload["title"] = title
                if event.payload.get("narrate") is True:
                    payload["narrate"] = True
                whiteboard = event.payload.get("whiteboard")
                if isinstance(whiteboard, dict):
                    payload["whiteboard"] = whiteboard
                self._messages.append(
                    {
                        "role": "tool",
                        "name": "task_event",
                        "content": json.dumps(
                            payload, ensure_ascii=False, separators=(",", ":")
                        ),
                    }
                )
            elif event.kind == "session_start":
                if event.payload.get("language") == "en":
                    ephemeral_prompt = (
                        "This is a new, empty conversation. The user has not asked a question or assigned a task. "
                        "Give one brief, natural greeting in English and invite the user to speak."
                    )
                else:
                    ephemeral_prompt = (
                        "这是一个全新的空白会话，用户尚未提出任何问题、主题或任务。"
                        "请主动说一句简短、自然的中文问候并邀请用户开口，例如“你好，今天想聊点什么？”。"
                    )
            else:
                raise FrontendModelProtocolError(
                    f"unsupported X-Live frontend model input event: {event.kind}"
                )
            self._seen_input_events.add(event.event_id)
        return ephemeral_prompt

    async def _completion(
        self,
        generation_id: str,
        *,
        ephemeral_user_message: str | None = None,
        allow_tools: bool = True,
    ) -> AsyncIterator[dict[str, Any]]:
        if self._http is None:
            import aiohttp

            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._http = aiohttp.ClientSession(timeout=timeout)
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        async with self._history_lock:
            messages = copy.deepcopy(self._messages)
        if ephemeral_user_message:
            messages.append({"role": "user", "content": ephemeral_user_message})
        body = {
            "model": self.config.served_model_name,
            "messages": messages,
            "stream": True,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if allow_tools:
            body.update(
                {
                    "tools": self.tools,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                }
            )
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        response = await self._http.post(url, headers=headers, json=body)
        self._responses[generation_id] = response
        try:
            if response.status >= 400:
                detail = await response.text()
                raise FrontendModelProtocolError(
                    f"frontend model HTTP {response.status}: {detail[:1000]}"
                )
            async for raw_line in response.content:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    packet = json.loads(data)
                    choices = packet.get("choices") or []
                    if choices:
                        yield choices[0].get("delta") or {}
                except (json.JSONDecodeError, AttributeError, TypeError) as exc:
                    raise FrontendModelProtocolError(
                        f"invalid model stream packet: {data[:300]}"
                    ) from exc
        finally:
            self._responses.pop(generation_id, None)
            response.close()

    def _parse_tool_call(
        self, call: dict[str, str]
    ) -> tuple[NativeToolCall, dict[str, Any]]:
        name = call["name"]
        if name not in _EXPECTED_ARGUMENTS:
            raise FrontendModelProtocolError(
                f"unknown X-Live frontend model tool: {name!r}"
            )
        try:
            arguments = json.loads(call["arguments"] or "{}")
        except json.JSONDecodeError as exc:
            raise FrontendModelProtocolError(
                "tool arguments are not valid JSON"
            ) from exc
        if (
            not isinstance(arguments, dict)
            or set(arguments) != _EXPECTED_ARGUMENTS[name]
        ):
            raise FrontendModelProtocolError(f"invalid {name} arguments: {arguments!r}")
        for key, value in arguments.items():
            if not isinstance(value, str) or not value.strip():
                raise FrontendModelProtocolError(f"{name}.{key} must be non-empty text")
            arguments[key] = value.strip()
        call_id = call["id"] or f"call_{uuid4().hex}"
        native = NativeToolCall(call_id, NativeToolName(name), arguments)
        wire = {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(
                    arguments, ensure_ascii=False, separators=(",", ":")
                ),
            },
        }
        return native, wire

    def _validate_tools(self) -> None:
        try:
            names = [item["function"]["name"] for item in self.tools]
        except (KeyError, TypeError) as exc:
            raise FrontendModelProtocolError(
                "tools.json is not an OpenAI tool list"
            ) from exc
        if names != list(_EXPECTED_ARGUMENTS):
            raise FrontendModelProtocolError(
                f"unexpected X-Live frontend model tools: {names}"
            )
