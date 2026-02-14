"""Feedback handler -- routes quality feedback to the learning pipeline."""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class FeedbackHandler:
    """Routes feedback to LearningOptimizer and publishes events."""

    def __init__(self, event_bus=None, learning_optimizer=None):
        self._event_bus = event_bus
        self._learning_optimizer = learning_optimizer

    async def handle_feedback(
        self,
        task_id: str,
        score: float,
        feedback: str = "",
        agent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process feedback for a completed task."""
        logger.info("Feedback received for task %s: score=%.2f", task_id, score)

        # Route to learning optimizer if available
        if self._learning_optimizer is not None:
            try:
                self._learning_optimizer.complete_session(
                    session_id=task_id,
                    success=score >= 0.5,
                    quality=score * 100,
                )
            except Exception:
                logger.exception("Failed to record feedback in learning optimizer")

        # Publish event
        if self._event_bus is not None:
            try:
                from core.interfaces import EventType

                await self._event_bus.publish(
                    EventType.FEEDBACK_RECEIVED,
                    {"task_id": task_id, "score": score, "feedback": feedback, "agent_id": agent_id},
                    source="feedback_handler",
                )
            except Exception:
                logger.exception("Failed to publish feedback event")

        return {
            "status": "recorded",
            "task_id": task_id,
            "score": score,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
