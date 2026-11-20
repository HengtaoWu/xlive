from __future__ import annotations

import json
import re


_PROGRESS_RE = re.compile(
    r"<xlive_progress>\s*(\{.*?\})\s*</xlive_progress>", re.DOTALL
)
_RESULT_RE = re.compile(
    r"<xlive_result>\s*(\{.*?\})\s*</xlive_result>", re.DOTALL
)
_WHITEBOARD_RE = re.compile(
    r"<xlive_whiteboard>\s*(.*?)\s*</xlive_whiteboard>", re.DOTALL
)
_ALLOWED_WHITEBOARD_KINDS = {"heading", "text", "equation"}
_ALLOWED_WHITEBOARD_OPERATIONS = {"clear", "append", "replace", "highlight", "remove"}


def _json_marker(message: str, pattern: re.Pattern[str]) -> dict | None:
    match = pattern.search(message)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _json_after_open_tag(message: str, tag: str) -> dict | None:
    """Decode one JSON object even if a transport rewrites the closing tag."""

    marker = f"<{tag}>"
    start = message.find(marker)
    if start < 0:
        return None
    candidate = message[start + len(marker):].lstrip()
    try:
        payload, _end = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def last_agent_message(output: str) -> str:
    """Return the last completed Codex agent-message item from JSONL output."""

    result = ""
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "item.completed":
            continue
        item = event.get("item") or {}
        if item.get("type") == "agent_message" and item.get("text"):
            result = str(item["text"]).strip()
    return result


def parse_progress(message: str) -> tuple[str, str, bool] | None:
    """Parse one private progress marker into title, detail, and narration intent."""

    payload = _json_marker(message, _PROGRESS_RE)
    if payload is None:
        return None
    title = str(payload.get("title") or "").strip()[:80]
    detail = str(payload.get("message") or "").strip()
    if not detail:
        return None
    return title, detail, payload.get("narrate") is True


def parse_whiteboard(message: str) -> dict | None:
    """Parse and validate one private whiteboard update marker."""

    payload = _json_marker(message, _WHITEBOARD_RE)
    if payload is None:
        # Some Codex transports append a DSML tool suffix instead of preserving
        # the closing tag. The JSON object itself is still authoritative.
        payload = _json_after_open_tag(message, "xlive_whiteboard")
    if payload is None:
        return None

    title = str(payload.get("title") or "").strip()[:80]
    narration = str(payload.get("narration") or "").strip()
    state = str(payload.get("state") or "open").strip()
    if not title or not narration or state not in {"open", "closed"}:
        return None

    operations = []
    raw_operations = payload.get("operations")
    if isinstance(raw_operations, list):
        for raw in raw_operations[:50]:
            if not isinstance(raw, dict):
                continue
            operation_type = str(raw.get("type") or "")
            if operation_type not in _ALLOWED_WHITEBOARD_OPERATIONS:
                continue
            if operation_type == "clear":
                operations.append({"type": "clear"})
                continue
            block_id = str(raw.get("id") or "").strip()[:80]
            if not block_id:
                continue
            operation = {"type": operation_type, "id": block_id}
            if operation_type == "append":
                kind = str(raw.get("kind") or "")
                content = str(raw.get("content") or "").strip()
                if kind not in _ALLOWED_WHITEBOARD_KINDS or not content:
                    continue
                operation.update({"kind": kind, "content": content[:4000]})
            elif operation_type == "replace":
                content = str(raw.get("content") or "").strip()
                if not content:
                    continue
                operation["content"] = content[:4000]
            elif operation_type == "highlight" and isinstance(raw.get("highlighted"), bool):
                operation["highlighted"] = raw["highlighted"]
            operations.append(operation)

    return {
        "title": title,
        "narration": narration,
        "state": state,
        "awaiting_user": payload.get("awaiting_user") is True,
        "auto_continue": payload.get("auto_continue") is True,
        "operations": operations,
    }


def clean_agent_result(message: str) -> str:
    """Turn the private result marker into model-visible natural language."""

    payload = _json_marker(message, _RESULT_RE)
    if payload is None:
        return message.strip()
    summary = str(payload.get("summary") or "").strip()
    details = str(payload.get("details") or "").strip()
    if summary and details:
        return f"{summary}\n{details}"
    return summary or details or message.strip()


def decode(value: bytes | str) -> str:
    """Decode subprocess output without allowing malformed bytes to abort a task."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
