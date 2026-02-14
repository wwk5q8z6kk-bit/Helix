"""Lightweight in-memory event bus satisfying the IEventBus protocol.

Used for intra-process pub/sub messaging between orchestrator, learning,
and swarm subsystems.  No external dependencies (Kafka/RabbitMQ removed).
"""

import inspect
import logging
import uuid
from typing import Any, Callable, Awaitable, Dict, Optional

from core.interfaces.event_bus import EventType

logger = logging.getLogger(__name__)


class InMemoryEventBus:
    """Simple async event bus for single-process use.

    Satisfies ``core.interfaces.IEventBus`` via structural subtyping.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[EventType, Dict[str, Callable]] = {}

    async def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None,
    ) -> None:
        if source:
            data = {**data, "_source": source}
        handlers = self._subscribers.get(event_type, {})
        for handler in handlers.values():
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception:
                logger.exception("Event handler failed for %s", event_type.value)

    async def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Dict[str, Any]], Awaitable[None]],
        *,
        replay_missed: bool = False,
    ) -> str:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = {}
        sub_id = uuid.uuid4().hex[:12]
        self._subscribers[event_type][sub_id] = handler
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> None:
        for handlers in self._subscribers.values():
            handlers.pop(subscription_id, None)
