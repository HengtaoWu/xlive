from __future__ import annotations

import uuid
from inspect import signature
from typing import Any, Callable, Coroutine, TYPE_CHECKING, Type

if TYPE_CHECKING:
    from fastapi import WebSocket
    from ..application import XLivePipeline as Pipeline

from ..log_utils import logger

from ..runtime.event_bus import EventBus
from ..runtime.manager import Manager
from .session_config import SessionConfigFactory, build_session_config
from .modules.output_gateway import OutputGateway
from .modules.input_gateway import InputGateway
from ..runtime.protocol_events import BaseEvent, LLMAgentLoop
from ..runtime.manager import EventListenerMixin, EventOverrides


class Service:
    """Orchestrate a session-scoped pipeline and manager stack.

    ``session_config_factory`` is only invoked for live sessions, so generated
    state is never shared by cloned services. With ``_websocket=None`` the
    instance is a prototype and skips all runtime setup.
    """

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        service_config: dict[str, Any] | None = None,
        manager_classes: list[Type[Manager]] | None = None,
        session_config_factory: SessionConfigFactory | None = None,
        _websocket: WebSocket | None = None,
        _session_id: str | None = None,
        _event_overrides: dict[Type[EventListenerMixin], EventOverrides] | None = None,
    ):
        self.session_id = str(uuid.uuid4()) if _session_id is None else _session_id
        self._session_config_factory = session_config_factory
        pipeline = pipeline.clone()
        self.pipeline = pipeline  # Keep pipeline reference for later model switches
        # Per-session config (shared with managers/gateways)
        self.service_config = build_session_config(
            service_config,
            self.session_id,
            factory=session_config_factory,
            materialize=_websocket is not None,
        )

        self.event_bus = EventBus()

        self._manager_classes: list[Type[Manager]] = list(manager_classes or [])
        if self._should_enable_persistence_manager():
            from .modules.persistence_manager import PersistenceManager

            if PersistenceManager not in self._manager_classes:
                self._manager_classes.append(PersistenceManager)
        self._managers: list[Manager] = []
        self._event_overrides: dict[Type[EventListenerMixin], EventOverrides] = (
            self._clone_event_overrides(_event_overrides)
        )

        # When _websocket is None this is a prototype instance with no runtime logic
        self.input_gateway: InputGateway | None = None
        self.output_gateway: OutputGateway | None = None
        if _websocket is None:
            return
        for manager_cls in self._manager_classes:
            manager = self._instantiate_manager(manager_cls)
            self._managers.append(manager)
        self.input_gateway = InputGateway(
            self.event_bus,
            self.session_id,
            _websocket,
            config=self.service_config,
        )
        self.output_gateway = OutputGateway(
            self.event_bus,
            self.session_id,
            _websocket,
            config=self.service_config,
            _event_overrides=self._event_overrides.get(OutputGateway),
        )

    def _get_or_create_overrides(
        self, event_listener_cls: Type[Manager]
    ) -> EventOverrides:
        overrides = self._event_overrides.get(event_listener_cls)
        if overrides is None:
            overrides = EventOverrides()
            self._event_overrides[event_listener_cls] = overrides
        return overrides

    def unsubscribe_event(
        self,
        *,
        event_listener_cls: Type[EventListenerMixin],
        event_type: Type[BaseEvent],
        method_name: str | None = None,
    ) -> None:
        """Disable an automatic event subscription for a listener class."""
        overrides = self._get_or_create_overrides(event_listener_cls)

        if method_name is None:
            overrides.disable.append(("*", event_type))
        else:
            overrides.disable.append((method_name, event_type))

    def subscribe_event(
        self,
        *,
        event_listener_cls: Type[EventListenerMixin],
        event_type: Type[BaseEvent],
        method_or_handler: (
            str
            | Callable[[EventListenerMixin, BaseEvent], Any]
            | Callable[[BaseEvent], Any]
            | Callable[[EventListenerMixin, BaseEvent], Coroutine[Any, Any, Any]]
            | Callable[[BaseEvent], Coroutine[Any, Any, Any]]
        ),
        priority: int = 0,
        enabled_if: Callable[[EventListenerMixin], bool] | None = None,
    ) -> None:
        """Register an additional event subscription for a listener class."""
        overrides = self._get_or_create_overrides(event_listener_cls)

        if isinstance(method_or_handler, str):
            overrides.extra.append(
                {
                    "method_name": method_or_handler,
                    "event_type": event_type,
                    "priority": priority,
                    "enabled_if": enabled_if,
                }
            )
        elif callable(method_or_handler):
            overrides.extra.append(
                {
                    "handler": method_or_handler,
                    "event_type": event_type,
                    "priority": priority,
                    "enabled_if": enabled_if,
                }
            )
        else:
            raise TypeError(
                "method_or_handler must be a method name (str) or a callable"
            )

    def _instantiate_manager(self, manager_cls: Type[Manager]) -> Manager:
        # Locate overrides for this manager class
        overrides = self._event_overrides.get(manager_cls)

        # Determine whether pipeline is needed
        if "pipeline" in signature(manager_cls.__init__).parameters:
            kwargs: dict[str, Any] = dict(
                event_bus=self.event_bus,
                session_id=self.session_id,
                pipeline=self.pipeline,
                config=self.service_config,
            )
        else:
            kwargs = dict(
                event_bus=self.event_bus,
                session_id=self.session_id,
                config=self.service_config,
            )

        # Pass overrides via hidden parameter to EventListenerMeta.__call__
        if overrides is not None:
            kwargs["_event_overrides"] = overrides

        return manager_cls(**kwargs)

    def register_manager(self, manager_cls: Type[Manager]):
        """Register a manager class on the service."""
        self._manager_classes.append(manager_cls)
        # Instantiate immediately when running in a live session
        if self.input_gateway is not None:
            self._managers.append(self._instantiate_manager(manager_cls))

    def unregister_manager(self, manager_cls: Type[Manager]):
        """Remove a manager class from the service."""
        self._manager_classes.remove(manager_cls)
        # Remove concrete manager if already active and unsubscribe its handlers
        if self.input_gateway is not None:
            for m in self._managers:
                if isinstance(m, manager_cls):
                    self._unsubscribe_manager_handlers(m)
            self._managers = [
                m for m in self._managers if not isinstance(m, manager_cls)
            ]

    def _unsubscribe_manager_handlers(self, manager: Manager) -> None:
        """Unsubscribe all event handlers registered by a manager instance."""
        for base in reversed(manager.__class__.mro()):
            handlers_meta = getattr(base, "__event_handlers_meta__", [])
            for method_name, meta_list in handlers_meta:
                method = getattr(manager, method_name, None)
                if method is None:
                    continue
                for meta in meta_list:
                    ev_type = meta["event_type"]
                    self.event_bus.unsubscribe(ev_type, method)

    async def handle_message_loop(self, already_accepted: bool = False) -> None:
        """Run the WebSocket message loop for a live session."""
        if self.input_gateway is None or self.output_gateway is None:
            raise RuntimeError(
                "This Service instance is a prototype and cannot handle messages."
            )
        await self.input_gateway.handle_connection(already_accepted=already_accepted)
        await self.event_bus.publish(
            LLMAgentLoop(session_id=self.session_id),
            wait_for_completion=True,
        )
        await self.input_gateway.handle_message_loop()

    def restore_conversation(
        self,
        *,
        messages: list[dict[str, Any]],
        tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Restore persisted visible turns into every configured chat owner."""
        restored_ids: set[int] = set()
        history_adapter = self.service_config.get("conversation_history_adapter")
        restorer = getattr(history_adapter, "restore_history", None)
        if callable(restorer):
            restorer(messages)
            restored_ids.add(id(history_adapter))
        task_restorer = getattr(history_adapter, "restore_tasks", None)
        if callable(task_restorer):
            task_restorer(tasks or [])

        get_agent = getattr(self.pipeline, "get_agent", None)
        agent = get_agent() if callable(get_agent) else None
        agent_restorer = getattr(agent, "restore_history", None)
        if callable(agent_restorer) and id(agent) not in restored_ids:
            agent_restorer(messages)

    async def send_session_attached(
        self,
        *,
        messages: list[dict[str, Any]] | None = None,
        tasks: list[dict[str, Any]] | None = None,
    ) -> None:
        """Notify the client and optionally include authoritative history."""
        if self.output_gateway is None:
            raise RuntimeError(
                "This Service instance is a prototype and cannot send session events."
            )
        await self.output_gateway.send_session_attached(messages=messages, tasks=tasks)

    async def stop(self) -> None:
        """Stop the service and shut down all managers."""
        for manager in self._managers:
            try:
                await manager.shutdown()
            except Exception as e:
                logger.error(
                    "Error while stopping manager %s - session: %s, error: %s",
                    type(manager).__name__,
                    self.session_id,
                    e,
                )
        try:
            await self.event_bus.shutdown()
        except Exception as e:
            logger.error(
                "Error while stopping event bus - session: %s, error: %s",
                self.session_id,
                e,
            )

    def clone(
        self,
        new_websocket: WebSocket,
        *,
        session_id: str | None = None,
        service_config_overrides: dict[str, Any] | None = None,
    ) -> "Service":
        """Clone the service prototype for a new WebSocket session."""
        cloned_service_config = dict(self.service_config)
        if service_config_overrides:
            cloned_service_config.update(service_config_overrides)
        new_service = type(self)(
            pipeline=self.pipeline,
            service_config=cloned_service_config,
            manager_classes=self._manager_classes,
            session_config_factory=self._session_config_factory,
            _websocket=new_websocket,
            _session_id=session_id,
            _event_overrides=self._event_overrides,
        )
        return new_service

    def _should_enable_persistence_manager(self) -> bool:
        persistence_store = self.service_config.get("persistence_store")
        user_id = self.service_config.get("user_id")
        return (
            persistence_store is not None and isinstance(user_id, str) and bool(user_id)
        )

    @staticmethod
    def _clone_event_overrides(
        source: dict[Type[EventListenerMixin], EventOverrides] | None,
    ) -> dict[Type[EventListenerMixin], EventOverrides]:
        if source is None:
            return {}
        cloned: dict[Type[EventListenerMixin], EventOverrides] = {}
        for listener_cls, overrides in source.items():
            cloned[listener_cls] = EventOverrides(
                disable=[
                    (method_name, event_type)
                    for method_name, event_type in overrides.disable
                ],
                extra=[dict(item) for item in overrides.extra],
            )
        return cloned
