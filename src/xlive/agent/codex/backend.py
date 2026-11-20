from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import signal
from typing import Any
from uuid import uuid4

from .config import CodexBackendConfig
from .prompts import INSPECT_PROMPT, task_prompt, update_prompt
from .protocol import (
    clean_agent_result,
    decode,
    last_agent_message,
    parse_progress,
    parse_whiteboard,
)
from ..runtime import (
    AgentEventSink,
    TaskBackendEvent,
    TaskEventKind,
    WhiteboardUpdate,
)


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _CodexTask:
    instruction: str
    version: int = 1
    updates: list[str] = field(default_factory=list)
    state: str = "queued"
    detail: str = "Codex task is queued."
    title: str = ""
    thread_id: str | None = None
    serial: int = 1
    process: Any | None = None
    runner: asyncio.Task[None] | None = None
    whiteboard_revision: int = 0
    whiteboard_open: bool = False
    awaiting_user: bool = False
    awaiting_playback: bool = False


ProcessFactory = Callable[..., Awaitable[Any]]
ProcessTerminator = Callable[[Any], Awaitable[None]]


class CodexAgentBackend:
    """Execute X-Live tasks through non-interactive Codex CLI subprocesses."""

    def __init__(
        self,
        config: CodexBackendConfig,
        *,
        concurrency_gate: asyncio.Semaphore | None = None,
        process_factory: ProcessFactory | None = None,
        process_terminator: ProcessTerminator | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self._gate = (
            concurrency_gate
            if concurrency_gate is not None
            else asyncio.Semaphore(config.max_concurrency)
        )
        # Inspection is an independent, read-only fork. It must not wait behind
        # the long-running task it is supposed to observe.
        self._inspect_gate = asyncio.Semaphore(2)
        self._process_factory = process_factory or asyncio.create_subprocess_exec
        self._process_terminator = process_terminator or _terminate_process
        self._tasks: dict[str, _CodexTask] = {}
        self._lock = asyncio.Lock()
        self._event_sink: AgentEventSink | None = None
        self._closed = False

    def set_event_sink(self, sink: AgentEventSink | None) -> None:
        self._event_sink = sink

    async def delegate(self, task_id: str, instruction: str) -> None:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Codex delegate instruction cannot be empty")
        async with self._lock:
            if self._closed:
                raise RuntimeError("Codex backend is closed")
            if task_id in self._tasks:
                raise ValueError(f"Codex task already exists: {task_id}")
            task = _CodexTask(instruction=instruction)
            self._tasks[task_id] = task
            self._launch(task_id, task)

    async def update(self, task_id: str, instruction: str, version: int) -> None:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("Codex update instruction cannot be empty")
        async with self._lock:
            task = self._require(task_id)
            if task.state in {"failed", "cancelled"}:
                raise RuntimeError(f"Codex task is already {task.state}: {task_id}")
            task.serial += 1
            task.version = version
            task.updates.append(instruction)
            task.awaiting_user = False
            task.awaiting_playback = False
            task.state = "restarting"
            task.detail = "Codex is restarting with the updated requirement."
            runner, process = self._detach_run(task)
        await self._stop_run(runner, process)
        async with self._lock:
            task = self._require_active(task_id)
            task.state = "queued"
            task.detail = "Updated Codex task is queued."
            self._launch(task_id, task)

    async def cancel(self, task_id: str) -> None:
        async with self._lock:
            task = self._require_active(task_id)
            task.serial += 1
            task.state = "cancelled"
            task.detail = "Codex task was cancelled."
            runner, process = self._detach_run(task)
            version = task.version
            whiteboard = None
            if task.whiteboard_open:
                task.whiteboard_revision += 1
                task.whiteboard_open = False
                whiteboard = WhiteboardUpdate(
                    revision=task.whiteboard_revision,
                    title=task.title or "Whiteboard",
                    state="closed",
                )
        await self._stop_run(runner, process)
        await self._emit(
            TaskBackendEvent(
                event_id=str(uuid4()),
                task_id=task_id,
                version=version,
                kind=TaskEventKind.CANCELLED,
                detail="Codex task was cancelled.",
                title=task.title,
                whiteboard=whiteboard,
            )
        )

    async def inspect(self, task_id: str) -> TaskBackendEvent:
        async with self._lock:
            task = self._require(task_id)
            kind = {
                "completed": TaskEventKind.COMPLETED,
                "failed": TaskEventKind.FAILED,
                "cancelled": TaskEventKind.CANCELLED,
            }.get(task.state)
            if kind is not None or task.thread_id is None:
                return TaskBackendEvent(
                    event_id=str(uuid4()),
                    task_id=task_id,
                    version=task.version,
                    kind=kind or TaskEventKind.PROGRESS,
                    detail=task.detail,
                    title=task.title,
                )
            thread_id = task.thread_id
            version = task.version
            title = task.title

        async with self._inspect_gate:
            process = await self._process_factory(
                *self._inspect_command(thread_id),
                cwd=str(Path(self.config.workspace).expanduser().resolve()),
                env=self._subprocess_env(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=min(self.config.task_timeout_seconds, 120.0),
                )
            except TimeoutError:
                await self._process_terminator(process)
                detail = "检查超时；原任务仍在继续。"
            else:
                if process.returncode:
                    error = decode(stderr).strip()[-1000:]
                    detail = "暂时无法读取任务进度。"
                    if error:
                        logger.warning("Codex inspect failed: %s", error)
                else:
                    detail = clean_agent_result(last_agent_message(decode(stdout)))
                    if not detail:
                        detail = "Codex 尚未给出新的可确认进度。"
            return TaskBackendEvent(
                event_id=str(uuid4()),
                task_id=task_id,
                version=version,
                kind=TaskEventKind.PROGRESS,
                detail=detail,
                title=title,
            )

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            runs = []
            for task in self._tasks.values():
                task.serial += 1
                runner, process = self._detach_run(task)
                runs.append((runner, process))
            self._event_sink = None
        await asyncio.gather(
            *(self._stop_run(runner, process) for runner, process in runs),
            return_exceptions=True,
        )

    def _launch(self, task_id: str, task: _CodexTask) -> None:
        serial = task.serial
        version = task.version
        task.runner = asyncio.create_task(
            self._run(task_id, serial, version),
            name=f"xlive-codex-{task_id}-v{version}",
        )

    async def _run(self, task_id: str, serial: int, version: int) -> None:
        process = None
        try:
            async with self._gate:
                async with self._lock:
                    task = self._tasks.get(task_id)
                    if task is None or task.serial != serial or self._closed:
                        return
                    task.state = "running"
                    task.detail = "Codex is working on the task."
                    command = self._command(task)
                process = await self._process_factory(
                    *command,
                    cwd=str(Path(self.config.workspace).expanduser().resolve()),
                    env=self._subprocess_env(),
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
                async with self._lock:
                    task = self._tasks.get(task_id)
                    if task is None or task.serial != serial or self._closed:
                        stale = True
                    else:
                        stale = False
                        task.process = process
                if stale:
                    await self._process_terminator(process)
                    return
                try:
                    stdout, stderr = await asyncio.wait_for(
                        self._collect_output(process, task_id, serial, version),
                        timeout=self.config.task_timeout_seconds,
                    )
                except TimeoutError:
                    await self._process_terminator(process)
                    await self._finish(
                        task_id,
                        serial,
                        version,
                        TaskEventKind.FAILED,
                        "Codex task timed out.",
                    )
                    return
                return_code = process.returncode
                if return_code:
                    detail = decode(stderr).strip()[-2000:]
                    message = f"Codex exited with status {return_code}."
                    if detail:
                        message += f"\n{detail}"
                    await self._finish(
                        task_id, serial, version, TaskEventKind.FAILED, message
                    )
                    return
                result = clean_agent_result(last_agent_message(decode(stdout)))
                if not result:
                    await self._finish(
                        task_id,
                        serial,
                        version,
                        TaskEventKind.FAILED,
                        "Codex completed without a final agent message.",
                    )
                    return
                async with self._lock:
                    current = self._tasks.get(task_id)
                    awaiting_user = (
                        current is not None
                        and current.serial == serial
                        and (current.awaiting_user or current.awaiting_playback)
                    )
                    if awaiting_user:
                        current.state = "running"
                        current.detail = result[-self.config.result_max_chars :]
                if awaiting_user:
                    return
                await self._finish(
                    task_id,
                    serial,
                    version,
                    TaskEventKind.COMPLETED,
                    result[-self.config.result_max_chars :],
                )
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                await self._process_terminator(process)
            raise
        except Exception as exc:
            await self._finish(
                task_id,
                serial,
                version,
                TaskEventKind.FAILED,
                f"Codex backend error: {type(exc).__name__}: {exc}",
            )
        finally:
            async with self._lock:
                task = self._tasks.get(task_id)
                current = asyncio.current_task()
                if task is not None and task.serial == serial:
                    if task.process is process:
                        task.process = None
                    if task.runner is current:
                        task.runner = None

    async def _finish(
        self,
        task_id: str,
        serial: int,
        version: int,
        kind: TaskEventKind,
        detail: str,
    ) -> None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.serial != serial or self._closed:
                return
            task.state = kind.value
            task.detail = detail
            title = task.title
        await self._emit(
            TaskBackendEvent(
                event_id=str(uuid4()),
                task_id=task_id,
                version=version,
                kind=kind,
                detail=detail,
                title=title,
            )
        )

    async def _collect_output(
        self, process: Any, task_id: str, serial: int, version: int
    ) -> tuple[bytes, bytes]:
        stdout_stream = getattr(process, "stdout", None)
        stderr_stream = getattr(process, "stderr", None)
        if stdout_stream is None or not hasattr(stdout_stream, "readline"):
            stdout, stderr = await process.communicate()
            for line in decode(stdout).splitlines():
                await self._handle_json_event(task_id, serial, version, line)
            return stdout, stderr

        stderr_task = asyncio.create_task(stderr_stream.read())
        chunks: list[bytes] = []
        try:
            while True:
                line = await stdout_stream.readline()
                if not line:
                    break
                chunks.append(line)
                await self._handle_json_event(
                    task_id, serial, version, decode(line).strip()
                )
            await process.wait()
            return b"".join(chunks), await stderr_task
        finally:
            if not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)

    async def _handle_json_event(
        self, task_id: str, serial: int, version: int, line: str
    ) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return
        if event.get("type") == "thread.started" and event.get("thread_id"):
            async with self._lock:
                task = self._tasks.get(task_id)
                if task is not None and task.serial == serial:
                    task.thread_id = str(event["thread_id"])
            return
        if event.get("type") not in {"item.started", "item.completed"}:
            return
        item = event.get("item") or {}
        activity = self._public_activity(item, event["type"])
        if activity is not None:
            async with self._lock:
                task = self._tasks.get(task_id)
                if task is None or task.serial != serial or self._closed:
                    return
            await self._emit(TaskBackendEvent(
                event_id=str(uuid4()), task_id=task_id, version=version,
                kind=TaskEventKind.ACTIVITY,
                detail=json.dumps(activity, ensure_ascii=False),
            ))
        if event["type"] != "item.completed" or item.get("type") != "agent_message":
            return
        message = str(item.get("text") or "")
        progress = parse_progress(message)
        if progress is not None:
            title, detail, narrate = progress
            async with self._lock:
                task = self._tasks.get(task_id)
                if task is None or task.serial != serial or self._closed:
                    return
                # The task name is established once. Updates must not rename it.
                if title and not task.title:
                    task.title = title
                task.detail = detail
                current_title = task.title
            await self._emit(
                TaskBackendEvent(
                    event_id=str(uuid4()),
                    task_id=task_id,
                    version=version,
                    kind=TaskEventKind.PROGRESS,
                    detail=detail,
                    title=current_title,
                    narrate=narrate,
                )
            )

        board = parse_whiteboard(message)
        if board is None:
            return
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.serial != serial or self._closed:
                return
            # Only one teachable step is accepted from each Codex invocation.
            if task.awaiting_user or task.awaiting_playback:
                return
            if board["title"] and task.whiteboard_revision == 0:
                # The first board title becomes the stable lesson/task title.
                # Later teaching updates add requirements without renaming it.
                task.title = board["title"]
            task.whiteboard_revision += 1
            task.whiteboard_open = board["state"] == "open"
            task.awaiting_user = board["awaiting_user"]
            task.awaiting_playback = board["auto_continue"]
            task.detail = board["narration"]
            current_title = task.title or board["title"]
            revision = task.whiteboard_revision
        await self._emit(
            TaskBackendEvent(
                event_id=str(uuid4()),
                task_id=task_id,
                version=version,
                kind=TaskEventKind.PROGRESS,
                detail=board["narration"],
                title=current_title,
                whiteboard=WhiteboardUpdate(
                    revision=revision,
                    title=board["title"],
                    state=board["state"],
                    operations=tuple(board["operations"]),
                    auto_continue=board["auto_continue"],
                ),
            )
        )

    @staticmethod
    def _public_activity(item: dict, event_type: str) -> dict[str, str] | None:
        kind = item.get("type")
        if kind == "agent_message" and event_type == "item.completed":
            message = str(item.get("text") or "").strip()
            progress = parse_progress(message)
            board = parse_whiteboard(message)
            if progress is not None:
                text = progress[1]
            elif board is not None:
                text = board["narration"]
            elif "<xlive_" in message and "<xlive_result>" not in message:
                return None
            else:
                text = clean_agent_result(message)
            return {"role": "assistant", "content": text} if text else None
        tools = {
            "command_execution": "Command",
            "file_change": "File changes",
            "mcp_tool_call": "Tool call",
            "web_search": "Web search",
        }
        if kind not in tools:
            return None
        status = "Started" if event_type == "item.started" else "Completed"
        if item.get("status") == "failed":
            status = "Failed"
        elif kind == "command_execution" and item.get("exit_code") not in (None, 0):
            status = "Failed"
        return {"role": "tool", "content": f"{tools[kind]} · {status}"}

    async def _emit(self, event: TaskBackendEvent) -> None:
        sink = self._event_sink
        if sink is None:
            return
        try:
            await sink(event)
        except Exception:
            logger.exception("Failed to deliver Codex event for task %s", event.task_id)

    def _command(self, task: _CodexTask) -> list[str]:
        if task.thread_id:
            return self._resume_command(
                task.thread_id,
                task.updates[-1],
                original_instruction=task.instruction,
                updates=task.updates,
            )
        command = [
            self.config.codex_bin,
            "exec",
            *self._approval_args(),
            *self._web_search_args(),
            "--json",
            "--color",
            "never",
        ]
        if not self.config.approve_for_me and self.config.sandbox != "danger-full-access":
            command.extend(["--sandbox", self.config.sandbox])
        command.extend(
            [
                "--skip-git-repo-check",
                "-C",
                str(Path(self.config.workspace).expanduser().resolve()),
                "-c",
                f'model_reasoning_effort="{self.config.reasoning_effort}"',
            ]
        )
        if self.config.model:
            command.extend(["--model", self.config.model])
        command.append(task_prompt(task.instruction, task.updates))
        return command

    def _resume_command(
        self,
        thread_id: str,
        instruction: str,
        *,
        original_instruction: str | None = None,
        updates: tuple[str, ...] | list[str] = (),
    ) -> list[str]:
        command = [
            self.config.codex_bin,
            "exec",
            *self._approval_args(),
            *self._web_search_args(),
            "resume",
            "--json",
            "--skip-git-repo-check",
            "-c",
            f'model_reasoning_effort="{self.config.reasoning_effort}"',
        ]
        if self.config.model:
            command.extend(["--model", self.config.model])
        command.extend(
            [
                thread_id,
                update_prompt(
                    instruction,
                    original_instruction=original_instruction,
                    updates=updates,
                ),
            ]
        )
        return command

    def _inspect_command(self, thread_id: str) -> list[str]:
        command = [
            self.config.codex_bin,
            "exec",
            *self._approval_args(),
            *self._web_search_args(),
            "fork",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
        ]
        if not self.config.approve_for_me and self.config.sandbox != "danger-full-access":
            command.extend(["-c", 'sandbox_mode="read-only"'])
        command.extend(
            [
                "-c",
                f'model_reasoning_effort="{self.config.reasoning_effort}"',
            ]
        )
        if self.config.model:
            command.extend(["--model", self.config.model])
        command.extend(
            [
                thread_id,
                INSPECT_PROMPT.strip(),
            ]
        )
        return command

    def _approval_args(self) -> list[str]:
        if self.config.sandbox == "danger-full-access":
            return ["--dangerously-bypass-approvals-and-sandbox"]
        return ["--approve-for-me"] if self.config.approve_for_me else []

    def _web_search_args(self) -> list[str]:
        return (
            []
            if self.config.web_search_enabled
            else ["-c", 'web_search="disabled"']
        )

    def _subprocess_env(self) -> dict[str, str]:
        values = dict(os.environ)
        if self.config.codex_home:
            values["CODEX_HOME"] = str(
                Path(self.config.codex_home).expanduser().resolve()
            )
        return values

    def _require(self, task_id: str) -> _CodexTask:
        return self._tasks[task_id]

    def _require_active(self, task_id: str) -> _CodexTask:
        task = self._require(task_id)
        if task.state in {"completed", "failed", "cancelled"}:
            raise RuntimeError(f"Codex task is already {task.state}: {task_id}")
        return task

    @staticmethod
    def _detach_run(
        task: _CodexTask,
    ) -> tuple[asyncio.Task[None] | None, Any | None]:
        runner, process = task.runner, task.process
        task.runner = None
        task.process = None
        return runner, process

    async def _stop_run(
        self,
        runner: asyncio.Task[None] | None,
        process: Any | None,
    ) -> None:
        if process is not None and process.returncode is None:
            await self._process_terminator(process)
        if runner is not None and not runner.done():
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)


async def _terminate_process(process: Any) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
        return
    except TimeoutError:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    await process.wait()
