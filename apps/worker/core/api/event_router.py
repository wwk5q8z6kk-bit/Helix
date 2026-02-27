"""EventRouter — classifies and routes external events to internal handlers."""

import inspect
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class EventRouter:
    """Routes external events to registered handlers and the event bus."""

    def __init__(self, event_bus=None):
        self._event_bus = event_bus
        self._handlers: Dict[str, List[Callable]] = {}

    def register_handler(self, event_type: str, handler: Callable) -> None:
        """Register a handler for an event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)

    async def route_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Route an event to appropriate handlers."""
        event_id = uuid.uuid4().hex[:12]
        logger.info("Routing event %s type=%s source=%s", event_id, event_type, source)

        handled_by = []

        # Call registered handlers
        for handler in self._handlers.get(event_type, []):
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
                handled_by.append(handler.__name__ if hasattr(handler, "__name__") else str(handler))
            except Exception:
                logger.exception("Event handler failed for %s", event_type)

        # Publish to internal event bus
        if self._event_bus is not None:
            try:
                from core.interfaces import EventType

                try:
                    bus_event_type = EventType(event_type)
                except ValueError:
                    bus_event_type = EventType.SYSTEM_EVENT
                await self._event_bus.publish(
                    bus_event_type,
                    {"event_id": event_id, "source": source, **data},
                    source=source,
                )
            except Exception:
                logger.exception("Failed to publish event to bus")

        return {
            "status": "routed",
            "event_id": event_id,
            "event_type": event_type,
            "handled_by": handled_by,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
