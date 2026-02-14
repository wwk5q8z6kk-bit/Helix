"""Interface for event bus and pub/sub messaging.

Decouples components by allowing communication through published events.
"""

from typing import Protocol, Callable, Dict, Any, Awaitable, Optional
from enum import Enum


class EventType(Enum):
    """Standard event types in the system."""
    # Task lifecycle
    TASK_RECEIVED = "task_received"
    TASK_ROUTED = "task_routed"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    # Agent lifecycle
    AGENT_REGISTERED = "agent_registered"
    AGENT_UNREGISTERED = "agent_unregistered"
    # LLM / generation
    LLM_CALL_STARTED = "llm_call_started"
    LLM_CALL_COMPLETED = "llm_call_completed"
    GENERATION_STARTED = "generation_started"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_FAILED = "generation_failed"
    # Feedback / learning
    FEEDBACK_RECEIVED = "feedback_received"
    PREFERENCE_LOGGED = "preference_logged"
    RM_SCORE_COMPUTED = "rm_score_computed"
    QUALITY_ASSESSED = "quality_assessed"
    # Workflow / monitoring
    WORKFLOW_PROGRESS = "workflow_progress"
    STRATEGIC_EVENT_DETECTED = "strategic_event_detected"
    MODEL_RETRAINED = "model_retrained"
    TOOL_PERFORMANCE_UPDATED = "tool_performance_updated"
    # System
    SYSTEM_EVENT = "system_event"
    # Errors
    ERROR_OCCURRED = "error_occurred"


class IEventBus(Protocol):
    """Interface for publish-subscribe event messaging.

    Decouples components by allowing them to communicate
    through published events rather than direct imports.
    """

    async def publish(
        self,
        event_type: EventType,
        data: Dict[str, Any],
        source: Optional[str] = None
    ) -> None:
        """Publish an event to the bus.

        Args:
            event_type: Type of event
            data: Event data/payload
            source: Optional source identifier
        """
        ...

    async def subscribe(
        self,
        event_type: EventType,
        handler: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> str:
        """Subscribe to events of a type.

        Args:
            event_type: Type of events to listen for
            handler: Async function to handle events

        Returns:
            Subscription ID for later unsubscribe
        """
        ...

    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events.

        Args:
            subscription_id: ID from subscribe()
        """
        ...
