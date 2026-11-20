from __future__ import annotations

from inspect import isawaitable
import uuid
from pathlib import Path
from typing import Callable, Any
from fastapi import (
    HTTPException,
    Request,
    WebSocket,
    status,
)

from .auth import JWTAuth, extract_bearer_token, resolve_auth_config
from .persistence import PersistenceStore
from .serving.service_manager import ServiceManager
from .serving.session_limiter import SessionLimiter
from .serving.service import Service


class XLiveAPI:
    """Mount X-Live authentication, session, and WebSocket entrypoints."""

    def __init__(
        self,
        *,
        service_prototype: Service,
        max_sessions: int | None = None,
        session_delete_callback: Callable[[str], Any] | None = None,
    ):
        """Wrap a service prototype with auth, persistence, and session routing."""
        service_config = dict(service_prototype.service_config)
        self._persistence_enabled = self._is_persistence_enabled(service_config)
        data_dir = service_config.get("data_dir") or "data"
        auth_secret, auth_ttl_seconds = resolve_auth_config(service_config)

        self._persistence = (
            PersistenceStore(Path(data_dir).expanduser().resolve() / "chat_history.sqlite3")
            if self._persistence_enabled
            else None
        )
        self._auth = JWTAuth(secret=auth_secret, ttl_seconds=auth_ttl_seconds)
        self._service_manager = ServiceManager(
            service_prototype=service_prototype,
            persistence_store=self._persistence,
        )
        self._pipeline = service_prototype.pipeline
        self._session_delete_callback = session_delete_callback
        self._session_limiter = (
            SessionLimiter(max_sessions) if max_sessions is not None else None
        )

    def set_session_limit(self, limit: int):
        """Set or replace the concurrent session limit."""
        self._session_limiter = SessionLimiter(limit)

    def _login(self) -> dict[str, Any]:
        user_id = str(uuid.uuid4())
        user = (
            self._persistence.ensure_user(user_id)
            if self._persistence is not None
            else {"id": user_id}
        )
        return {
            "access_token": self._auth.issue_token(sub=user_id),
            "user": user,
        }

    def _verify_access_token(self, token: str) -> str:
        user_id = self._auth.verify_token(token)
        if self._persistence is not None:
            self._persistence.ensure_user(user_id)
        return user_id

    def _list_sessions(self, user_id: str) -> list[dict[str, Any]]:
        if self._persistence is None:
            return []
        self._persistence.ensure_user(user_id)
        return self._persistence.list_sessions(user_id)

    def _get_session_detail(self, user_id: str, session_id: str) -> dict[str, Any]:
        if self._persistence is None:
            raise KeyError(session_id)
        self._persistence.ensure_user(user_id)
        return self._persistence.get_session_detail(user_id, session_id)

    def _rename_session(
        self, user_id: str, session_id: str, title: str
    ) -> dict[str, Any]:
        if self._persistence is None:
            raise KeyError(session_id)
        self._persistence.ensure_user(user_id)
        return self._persistence.rename_session(user_id, session_id, title)

    async def _delete_session(self, user_id: str, session_id: str) -> None:
        if self._persistence is None:
            raise KeyError(session_id)
        self._persistence.ensure_user(user_id)
        if not self._persistence.user_owns_session(user_id, session_id):
            raise KeyError(session_id)
        if self._session_delete_callback is not None:
            result = self._session_delete_callback(session_id)
            if isawaitable(result):
                await result
        self._persistence.delete_session(user_id, session_id)

    def mount_routes(
        self,
        app: Any,
        *,
        login_path: str = "/api/auth/login",
        sessions_path: str = "/api/sessions",
        session_detail_path: str = "/api/sessions/{session_id}",
        ws_path: str = "/ws",
    ) -> None:
        """Mount the authentication, session, and WebSocket routes."""

        def _require_http_user(request: Request) -> str:
            token = extract_bearer_token(request.headers.get("authorization"))
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Missing bearer token",
                )
            try:
                return self._verify_access_token(token)
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=str(exc),
                ) from exc

        def _require_ws_user(websocket: WebSocket) -> str:
            token = websocket.query_params.get("access_token")
            if token is None:
                token = extract_bearer_token(websocket.headers.get("authorization"))
            if not token:
                raise ValueError("Missing access token")
            return self._verify_access_token(token)

        @app.post(login_path)
        async def _login_route() -> dict[str, Any]:
            return self._login()

        @app.get(sessions_path)
        async def _list_sessions_route(request: Request) -> dict[str, Any]:
            user_id = _require_http_user(request)
            return {"sessions": self._list_sessions(user_id)}

        @app.get(session_detail_path)
        async def _session_detail_route(
            request: Request, session_id: str
        ) -> dict[str, Any]:
            user_id = _require_http_user(request)
            try:
                return self._get_session_detail(user_id, session_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                ) from exc

        @app.get(session_detail_path + "/tasks/{task_id}/messages")
        async def _task_messages_route(request: Request, session_id: str, task_id: str) -> dict[str, Any]:
            user_id = _require_http_user(request)
            try:
                if self._persistence is None:
                    raise KeyError(session_id)
                messages = self._persistence.list_task_messages(user_id, session_id, task_id)
            except KeyError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found") from exc
            return {"messages": messages}

        @app.patch(session_detail_path)
        async def _rename_session_route(
            request: Request, session_id: str
        ) -> dict[str, Any]:
            user_id = _require_http_user(request)
            payload = await request.json()
            title = payload.get("title") if isinstance(payload, dict) else None
            if not isinstance(title, str):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="title must be a string",
                )
            try:
                return self._rename_session(user_id, session_id, title)
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                ) from exc
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(exc),
                ) from exc

        @app.delete(session_detail_path)
        async def _delete_session_route(
            request: Request, session_id: str
        ) -> dict[str, str]:
            user_id = _require_http_user(request)
            try:
                await self._delete_session(user_id, session_id)
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
                ) from exc
            return {"session_id": session_id}

        @app.websocket(ws_path)
        async def _websocket_route(websocket: WebSocket) -> None:
            try:
                user_id = _require_ws_user(websocket)
            except ValueError:
                await websocket.accept()
                await websocket.close(code=1008, reason="Unauthorized")
                return
            await self.connect(websocket, user_id=user_id)

    async def connect(self, websocket: WebSocket, user_id: str | None = None):
        """Accept a WebSocket session and hand it to the service manager.

        With a session limit configured, the socket is admitted through the
        limiter queue first.
        """
        if self._session_limiter:
            await websocket.accept()
            waiter = await self._session_limiter.acquire(websocket)
            if waiter is None:
                try:
                    await websocket.close(code=1000)
                except Exception:
                    pass
                return
            await self._service_manager.connect(
                websocket=websocket, already_accepted=True, user_id=user_id
            )
            await self._session_limiter.release(waiter)
            return

        await self._service_manager.connect(websocket=websocket, user_id=user_id)

    @staticmethod
    def _is_persistence_enabled(service_config: dict[str, Any]) -> bool:
        """Return whether session persistence is enabled for the service."""
        value = service_config.get("enable_persistence")
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"0", "false", "no", "off"}:
                return False
            if normalized in {"1", "true", "yes", "on"}:
                return True
        return bool(value)
