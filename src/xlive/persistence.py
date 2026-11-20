from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .agent.runtime import TaskBackendEvent


def _build_title(text: str, *, max_chars: int = 32) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 1].rstrip()}…"


def _decode_updates(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(decoded, list):
        return []
    return [str(item) for item in decoded if str(item).strip()]


class PersistenceStore:
    """SQLite-backed storage for users, sessions, and final text messages."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _initialize(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_tasks (
                    task_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    instruction TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    backend TEXT NOT NULL DEFAULT 'Codex',
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    detail TEXT NOT NULL DEFAULT '',
                    updates_json TEXT NOT NULL DEFAULT '[]',
                    whiteboard_json TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    completed_at REAL,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS chat_task_events (
                    event_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES chat_tasks(task_id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chat_sessions_user
                    ON chat_sessions(user_id);

                CREATE INDEX IF NOT EXISTS idx_chat_messages_session
                    ON chat_messages(session_id, id ASC);

                CREATE INDEX IF NOT EXISTS idx_chat_tasks_session
                    ON chat_tasks(session_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_chat_task_events_task
                    ON chat_task_events(task_id, created_at ASC);
                """
            )
            columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(chat_sessions)").fetchall()
            }
            if "created_at" not in columns:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN created_at TEXT")
            if "updated_at" not in columns:
                conn.execute("ALTER TABLE chat_sessions ADD COLUMN updated_at TEXT")
            task_columns = {
                str(row["name"])
                for row in conn.execute("PRAGMA table_info(chat_tasks)").fetchall()
            }
            if "updates_json" not in task_columns:
                conn.execute(
                    "ALTER TABLE chat_tasks ADD COLUMN updates_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "whiteboard_json" not in task_columns:
                conn.execute(
                    "ALTER TABLE chat_tasks ADD COLUMN whiteboard_json TEXT NOT NULL DEFAULT ''"
                )
            conn.execute(
                """
                UPDATE chat_sessions
                SET created_at = COALESCE(created_at, CURRENT_TIMESTAMP),
                    updated_at = COALESCE(updated_at, created_at, CURRENT_TIMESTAMP)
                """
            )

    def ensure_user(self, user_id: str) -> dict[str, Any]:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (id) VALUES (?)",
                (user_id,),
            )
        return {"id": user_id}

    def create_session(
        self, user_id: str, *, session_id: str | None = None
    ) -> dict[str, Any]:
        self.ensure_user(user_id)
        resolved_session_id = session_id or str(uuid.uuid4())
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, user_id, title)
                VALUES (?, ?, NULL)
                """,
                (resolved_session_id, user_id),
            )
        return {
            "session_id": resolved_session_id,
            "title": None,
        }

    def get_session(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM chat_sessions
                WHERE id = ? AND user_id = ?
                """,
                (session_id, user_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "session_id": str(row["id"]),
            "user_id": str(row["user_id"]),
            "title": row["title"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def user_owns_session(self, user_id: str, session_id: str) -> bool:
        return self.get_session(user_id, session_id) is not None

    def get_or_create_session(
        self, user_id: str, session_id: str | None = None
    ) -> dict[str, Any]:
        if session_id is None:
            return self.create_session(user_id)
        session = self.get_session(user_id, session_id)
        if session is None:
            raise KeyError(session_id)
        return session

    def list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.title, s.updated_at, COALESCE(MAX(m.id), 0) AS activity_rank
                FROM chat_sessions AS s
                LEFT JOIN chat_messages AS m ON m.session_id = s.id
                WHERE s.user_id = ?
                GROUP BY s.id, s.title, s.updated_at, s.rowid
                ORDER BY s.updated_at DESC, activity_rank DESC, s.rowid DESC
                """,
                (user_id,),
            ).fetchall()
        return [
            {
                "session_id": str(row["id"]),
                "title": row["title"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def list_messages(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "role": str(row["role"]),
                "content": str(row["content"]),
            }
            for row in rows
        ]

    def get_session_detail(self, user_id: str, session_id: str) -> dict[str, Any]:
        session = self.get_session(user_id, session_id)
        if session is None:
            raise KeyError(session_id)
        return {
            "session_id": session["session_id"],
            "title": session["title"],
            "updated_at": session["updated_at"],
            "messages": self.list_messages(user_id, session_id),
            "tasks": self.list_tasks(user_id, session_id),
        }

    def upsert_task(
        self,
        *,
        user_id: str,
        session_id: str,
        task_id: str,
        status: str,
        instruction: str = "",
        title: str = "",
        backend: str = "Codex",
        version: int = 1,
        detail: str = "",
        updates: list[str] | None = None,
        created_at: float | None = None,
        updated_at: float | None = None,
    ) -> None:
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        now = time.time()
        created = now if created_at is None else float(created_at)
        updated = now if updated_at is None else float(updated_at)
        terminal = status in {"completed", "failed", "cancelled"}
        updates_json = (
            ""
            if updates is None
            else json.dumps(
                [str(item) for item in updates if str(item).strip()],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_tasks (
                    task_id, session_id, instruction, title, backend, status,
                    version, detail, updates_json, created_at, updated_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    instruction = CASE WHEN excluded.instruction != ''
                        THEN excluded.instruction ELSE chat_tasks.instruction END,
                    title = CASE WHEN chat_tasks.title = '' AND excluded.title != ''
                        THEN excluded.title ELSE chat_tasks.title END,
                    backend = CASE WHEN excluded.backend != ''
                        THEN excluded.backend ELSE chat_tasks.backend END,
                    status = excluded.status,
                    version = MAX(chat_tasks.version, excluded.version),
                    detail = CASE WHEN excluded.detail != ''
                        THEN excluded.detail ELSE chat_tasks.detail END,
                    updates_json = CASE WHEN excluded.updates_json != ''
                        THEN excluded.updates_json ELSE chat_tasks.updates_json END,
                    updated_at = MAX(chat_tasks.updated_at, excluded.updated_at),
                    completed_at = excluded.completed_at
                """,
                (
                    task_id,
                    session_id,
                    instruction,
                    title,
                    backend,
                    status,
                    max(1, int(version)),
                    detail,
                    updates_json,
                    created,
                    updated,
                    updated if terminal else None,
                ),
            )

    def append_task_event(
        self,
        *,
        user_id: str,
        session_id: str,
        event: TaskBackendEvent,
    ) -> None:
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        kind = event.kind.value
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chat_task_events (
                    event_id, task_id, session_id, version, kind,
                    detail, title, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.task_id,
                    session_id,
                    int(event.version),
                    str(kind),
                    event.detail,
                    event.title,
                    time.time(),
                ),
            )
            whiteboard = event.whiteboard
            if whiteboard is not None:
                row = conn.execute(
                    "SELECT whiteboard_json FROM chat_tasks WHERE task_id = ?",
                    (event.task_id,),
                ).fetchone()
                try:
                    snapshot = json.loads(str(row["whiteboard_json"] or "{}")) if row else {}
                except (TypeError, ValueError):
                    snapshot = {}
                blocks = list(snapshot.get("blocks") or [])
                for operation in whiteboard.operations:
                    op = dict(operation)
                    kind = op.get("type")
                    if kind == "clear":
                        blocks = []
                    elif kind == "append":
                        block = {
                            "id": op.get("id"), "kind": op.get("kind"),
                            "content": op.get("content"), "highlighted": False,
                        }
                        blocks = [item for item in blocks if item.get("id") != block["id"]]
                        blocks.append(block)
                    elif kind == "remove":
                        blocks = [item for item in blocks if item.get("id") != op.get("id")]
                    elif kind == "replace":
                        blocks = [
                            {**item, "content": op.get("content")}
                            if item.get("id") == op.get("id") else item
                            for item in blocks
                        ]
                    elif kind == "highlight":
                        blocks = [
                            {**item, "highlighted": op.get("highlighted", True)}
                            if item.get("id") == op.get("id") else item
                            for item in blocks
                        ]
                snapshot = {
                    "open": whiteboard.state == "open",
                    "taskId": event.task_id,
                    "title": whiteboard.title or event.title or "Whiteboard",
                    "version": int(whiteboard.revision),
                    "blocks": blocks,
                }
                conn.execute(
                    "UPDATE chat_tasks SET whiteboard_json = ? WHERE task_id = ?",
                    (json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), event.task_id),
                )

    def list_task_messages(self, user_id: str, session_id: str, task_id: str) -> list[dict[str, Any]]:
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        with self._lock, self._connect() as conn:
            task = conn.execute(
                "SELECT instruction, created_at FROM chat_tasks WHERE session_id = ? AND task_id = ?",
                (session_id, task_id),
            ).fetchone()
            if task is None:
                raise KeyError(task_id)
            rows = conn.execute(
                "SELECT event_id, kind, detail, created_at FROM chat_task_events "
                "WHERE session_id = ? AND task_id = ? ORDER BY rowid ASC",
                (session_id, task_id),
            ).fetchall()
        messages = []
        for row in rows:
            if row["kind"] != "activity":
                continue
            try:
                message = json.loads(row["detail"])
            except (TypeError, ValueError):
                continue
            if (not isinstance(message, dict) or message.get("role") not in ("assistant", "tool")
                    or not isinstance(message.get("content"), str) or not message["content"].strip()):
                continue
            messages.append({
                "id": row["event_id"], "role": message["role"],
                "content": message["content"], "created_at": float(row["created_at"]),
            })
        if not messages:
            for row in rows:
                if row["kind"] in ("progress", "completed") and row["detail"].strip():
                    messages.append({"id": row["event_id"], "role": "assistant",
                                     "content": row["detail"], "created_at": float(row["created_at"])})
        messages.insert(0, {
            "id": f"delegate:{task_id}", "role": "user", "content": str(task["instruction"]),
            "created_at": float(task["created_at"]),
        })
        return messages

    def list_tasks(self, user_id: str, session_id: str) -> list[dict[str, Any]]:
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, instruction, title, backend, status, version,
                       detail, updates_json, whiteboard_json, created_at, updated_at, completed_at
                FROM chat_tasks
                WHERE session_id = ?
                ORDER BY created_at ASC, task_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "task_id": str(row["task_id"]),
                "instruction": str(row["instruction"]),
                "title": str(row["title"]),
                "backend": str(row["backend"]),
                "status": (
                    "running"
                    if str(row["status"]) in {"starting", "cancelling"}
                    else str(row["status"])
                ),
                "version": int(row["version"]),
                "detail": str(row["detail"]),
                "updates": _decode_updates(row["updates_json"]),
                "whiteboard": (
                    json.loads(str(row["whiteboard_json"]))
                    if str(row["whiteboard_json"] or "").strip() else None
                ),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
                "completed_at": (
                    None
                    if row["completed_at"] is None
                    else float(row["completed_at"])
                ),
            }
            for row in rows
        ]

    def rename_session(
        self, user_id: str, session_id: str, title: str
    ) -> dict[str, Any]:
        normalized = " ".join(title.split())
        if not normalized:
            raise ValueError("Session title cannot be empty")
        if len(normalized) > 80:
            raise ValueError("Session title must be at most 80 characters")
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE chat_sessions
                SET title = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND user_id = ?
                """,
                (normalized, session_id, user_id),
            )
        session = self.get_session(user_id, session_id)
        if session is None:
            raise KeyError(session_id)
        return {
            "session_id": session["session_id"],
            "title": session["title"],
            "updated_at": session["updated_at"],
        }

    def delete_session(self, user_id: str, session_id: str) -> None:
        """Delete one owned session and its cascaded messages and tasks."""

        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM chat_sessions WHERE id = ? AND user_id = ?",
                (session_id, user_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(session_id)

    def append_message(
        self,
        *,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
    ) -> None:
        normalized = content.strip()
        if not normalized:
            return
        if role not in {"user", "assistant"}:
            raise ValueError(f"Unsupported role: {role}")
        if not self.user_owns_session(user_id, session_id):
            raise KeyError(session_id)

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, normalized),
            )
            conn.execute(
                "UPDATE chat_sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )

            if role == "user":
                row = conn.execute(
                    "SELECT title FROM chat_sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                title = row["title"] if row is not None else None
                if not title:
                    conn.execute(
                        "UPDATE chat_sessions SET title = ? WHERE id = ?",
                        (_build_title(normalized), session_id),
                    )
