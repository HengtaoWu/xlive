"""Frontend adapter for the action-gated X-Live Dual vLLM server."""

from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
import copy
from contextlib import suppress
import json
import logging
from typing import Any
from uuid import uuid4

from .adapter import (
    FrontendModelProtocolError,
    XLiveFrontendModelAdapter,
)
from .config import FrontendModelConfig
from .contract import ModelOutput, ModelOutputKind, NativeToolCall
from ..runtime.events import EventEnvelope


_TERMINAL_MARKERS = ("<|im_end|>", "<|endoftext|>")
_LOGGER = logging.getLogger(__name__)


class XLiveDualFrontendModelAdapter(XLiveFrontendModelAdapter):
    """Join the public Dialogue stream with the hidden Tool side channel.

    The custom vLLM runner keeps natural-language tokens on the OpenAI stream
    and writes gate/tool events to an append-only JSONL file. Each event names
    its parent completion, so concurrent sessions can safely share one file.
    """

    def __init__(
        self,
        config: FrontendModelConfig,
        *,
        http_session: Any | None = None,
    ) -> None:
        super().__init__(config, http_session=http_session)
        self._event_path = config.dual_event_path.expanduser().resolve()
        self._event_path.parent.mkdir(parents=True, exist_ok=True)
        self._event_path.touch(exist_ok=True)
        if config.dual_event_timeout_seconds <= 0:
            raise ValueError("dual event timeout must be positive")

    async def stream(
        self,
        event: EventEnvelope,
        *,
        generation_id: str,
    ) -> AsyncIterator[ModelOutput]:
        """Stream dialogue and tool calls independently, preserving assistant/tool history order."""

        ephemeral_prompt = await self._record_input(event)
        allow_tools = event.kind == "user_commit"
        content_parts: list[str] = []
        assistant_record: dict[str, Any] | None = None
        native_call: NativeToolCall | None = None

        async for delta in self._completion(
            generation_id,
            ephemeral_user_message=ephemeral_prompt,
            allow_tools=allow_tools,
        ):
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
                if assistant_record is not None:
                    # The record precedes the tool receipt in history, while
                    # its content is completed by the still-running branch.
                    async with self._history_lock:
                        assistant_record["content"] = "".join(content_parts)
                yield ModelOutput(ModelOutputKind.TEXT_DELTA, text=content)

            calls = delta.get("tool_calls") or []
            if not calls:
                continue
            if not allow_tools:
                raise FrontendModelProtocolError(
                    f"tool calls are not allowed for {event.kind}"
                )
            if native_call is not None or len(calls) != 1:
                raise FrontendModelProtocolError(
                    "Dual emitted more than one tool call"
                )
            item = calls[0]
            function = item.get("function") or {}
            native_call, wire_call = self._parse_tool_call(
                {
                    "id": str(item.get("id") or ""),
                    "name": str(function.get("name") or ""),
                    "arguments": str(function.get("arguments") or ""),
                }
            )
            assistant_record = {
                "role": "assistant",
                "content": "".join(content_parts) or None,
                "tool_calls": [wire_call],
            }
            async with self._history_lock:
                self._append_generation_record(generation_id, assistant_record)
            # RealtimeSessionCore executes and records the task before asking
            # this generator for the next Dialogue delta.
            yield ModelOutput(ModelOutputKind.TOOL_CALL, tool_call=native_call)

        content = "".join(content_parts)
        if native_call is None:
            if content:
                async with self._history_lock:
                    self._append_generation_record(generation_id, {"role": "assistant", "content": content})
            return

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
            "tools": self.tools,
            # Tool generation is internal to the split runner. Stock vLLM
            # must never attempt a second, public tool-choice pass.
            "tool_choice": "none",
            "parallel_tool_calls": False,
            "stream": True,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        offset = self._event_path.stat().st_size
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        response = await self._http.post(url, headers=headers, json=body)
        self._responses[generation_id] = response
        completion_id = ""
        saw_choice = False
        action_task: asyncio.Task[tuple[dict[str, Any] | None, str]] | None = None
        next_line: asyncio.Task[bytes] | None = None
        stream_done = False
        action_done = False
        try:
            if response.status >= 400:
                detail = await response.text()
                raise FrontendModelProtocolError(
                    f"frontend model HTTP {response.status}: {detail[:1000]}"
                )
            iterator = response.content.__aiter__()
            next_line = asyncio.create_task(anext(iterator))
            while not stream_done:
                waiters: set[asyncio.Task[Any]] = {next_line}
                if action_task is not None:
                    waiters.add(action_task)
                completed, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )

                # Dispatch ready Tool events before waiting for more Dialogue tokens.
                if action_task is not None and action_task in completed:
                    try:
                        result = action_task.result()
                        tool_delta = self._action_delta(
                            result, allow_tools=allow_tools
                        )
                    except FrontendModelProtocolError as exc:
                        # Keep Dialogue and TTS running if the Tool branch is malformed.
                        _LOGGER.warning(
                            "discarding invalid Dual Tool side channel: %s", exc
                        )
                        tool_delta = None
                    action_task = None
                    action_done = True
                    if tool_delta is not None:
                        yield tool_delta

                if next_line not in completed:
                    continue
                try:
                    raw_line = next_line.result()
                except StopAsyncIteration:
                    stream_done = True
                    break
                line = raw_line.decode("utf-8").strip()
                next_line = asyncio.create_task(anext(iterator))
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    stream_done = True
                    break
                try:
                    packet = json.loads(data)
                    packet_id = packet.get("id")
                    if isinstance(packet_id, str) and packet_id:
                        completion_id = completion_id or packet_id
                        if action_task is None and not action_done:
                            action_task = asyncio.create_task(
                                self._wait_for_action(completion_id, offset)
                            )
                    choices = packet.get("choices") or []
                    if choices:
                        saw_choice = True
                        delta = choices[0].get("delta") or {}
                        content = delta.get("content")
                        if isinstance(content, str) and content:
                            yield {"content": content}
                except (json.JSONDecodeError, AttributeError, TypeError) as exc:
                    raise FrontendModelProtocolError(
                        f"invalid model stream packet: {data[:300]}"
                    ) from exc

            if not saw_choice or not completion_id:
                raise FrontendModelProtocolError(
                    "Dual vLLM returned no completion choices or request id"
                )
            if action_task is not None:
                try:
                    result = await action_task
                    tool_delta = self._action_delta(
                        result, allow_tools=allow_tools
                    )
                except FrontendModelProtocolError as exc:
                    _LOGGER.warning(
                        "discarding invalid Dual Tool side channel: %s", exc
                    )
                    tool_delta = None
                action_task = None
                action_done = True
                if tool_delta is not None:
                    yield tool_delta
            if not action_done:
                raise FrontendModelProtocolError(
                    f"Dual side channel was not started: {completion_id}"
                )
        finally:
            for task in (next_line, action_task):
                if task is not None and not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
            self._responses.pop(generation_id, None)
            response.close()

    def _action_delta(
        self,
        result: tuple[dict[str, Any] | None, str],
        *,
        allow_tools: bool,
    ) -> dict[str, Any] | None:
        event, tool_text = result
        if event is None:
            return None
        if not allow_tools:
            raise FrontendModelProtocolError(
                "Dual gate selected a tool for an event that forbids tool calls"
            )
        parsed = self._parse_tool_branch(tool_text)
        self._validate_gated_tool(event, parsed)
        return {
            "tool_calls": [
                {
                    "index": 0,
                    "id": f"call_{uuid4().hex}",
                    "function": {
                        "name": parsed["name"],
                        "arguments": json.dumps(
                            parsed["arguments"],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                }
            ]
        }

    async def _wait_for_action(
        self, completion_id: str, offset: int
    ) -> tuple[dict[str, Any] | None, str]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.config.dual_event_timeout_seconds
        pending = ""
        gate: dict[str, Any] | None = None
        tool: dict[str, Any] | None = None
        while loop.time() < deadline:
            with self._event_path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()
            if chunk:
                pending += chunk
                lines = pending.splitlines(keepends=True)
                pending = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    pending = lines.pop()
                for line in lines:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    parent = str(event.get("parent_request_id", ""))
                    if parent != completion_id and not parent.startswith(
                        completion_id + "-"
                    ):
                        continue
                    if event.get("kind") == "gate":
                        gate = event
                    elif "token_ids" in event:
                        tool = event
            if gate is not None:
                if int(gate.get("action_id", 0)) == 0:
                    return None, ""
                if tool is not None:
                    return tool, self._decode_tool_tokens(tool)
            await asyncio.sleep(0.02)
        raise FrontendModelProtocolError(
            f"timed out waiting for Dual side channel: {completion_id}"
        )

    def _decode_tool_tokens(self, event: dict[str, Any]) -> str:
        # The runtime adapter has no tokenizer dependency, so the server
        # publishes decoded_text alongside the raw token ids.
        text = event.get("decoded_text")
        if not isinstance(text, str):
            raise FrontendModelProtocolError(
                "Dual tool event is missing decoded_text"
            )
        return text

    @staticmethod
    def _parse_tool_branch(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        for marker in _TERMINAL_MARKERS:
            while cleaned.endswith(marker):
                cleaned = cleaned[: -len(marker)].rstrip()

        # Extract JSON from decoder wrappers, then apply schema and tool validation.
        object_start = cleaned.find("{")
        if object_start < 0:
            raise FrontendModelProtocolError(
                f"Dual Tool branch emitted impure protocol text: {text!r}"
            )
        try:
            parsed, consumed = json.JSONDecoder().raw_decode(cleaned[object_start:])
        except json.JSONDecodeError as exc:
            raise FrontendModelProtocolError(
                "Dual Tool branch emitted invalid JSON"
            ) from exc
        trailing = cleaned[object_start + consumed :]
        for marker in (*_TERMINAL_MARKERS, "<tool_call>", "</tool_call>"):
            trailing = trailing.replace(marker, "")
        if trailing.strip():
            raise FrontendModelProtocolError(
                f"Dual Tool branch emitted trailing protocol text: {text!r}"
            )
        if not isinstance(parsed, dict) or set(parsed) != {"name", "arguments"}:
            raise FrontendModelProtocolError("invalid Dual Tool call envelope")
        if not isinstance(parsed["name"], str) or not isinstance(
            parsed["arguments"], dict
        ):
            raise FrontendModelProtocolError("invalid Dual Tool call fields")
        return parsed

    @staticmethod
    def _validate_gated_tool(
        event: dict[str, Any], parsed: dict[str, Any]
    ) -> None:
        """Check metadata fields. Gate/decoder disagreements do not override the decoded tool name."""

        expected = event.get("expected_tool_name")
        actual = parsed.get("name")
        if not isinstance(expected, str) or not expected:
            raise FrontendModelProtocolError(
                "Dual tool event is missing expected_tool_name"
            )
        if not isinstance(actual, str) or not actual:
            raise FrontendModelProtocolError(
                "Dual Tool branch is missing the decoded tool name"
            )


def create_frontend_model_adapter(
    config: FrontendModelConfig,
) -> XLiveFrontendModelAdapter:
    """Create the action-gated Dual frontend adapter."""

    if config.backend == "dual":
        return XLiveDualFrontendModelAdapter(config)
    raise ValueError(
        f"XLIVE_MODEL_BACKEND must be 'dual', got {config.backend!r}"
    )
