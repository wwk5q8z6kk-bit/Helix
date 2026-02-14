"""Helix learning subsystem — feedback collection and RL-based optimization."""

from core.learning.rl_integration import RLIntegration
from core.learning.learning_optimizer import LearningOptimizer

import logging

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Lightweight feedback collector that logs execution results for future RL use."""

    async def collect_execution_feedback(
        self,
        task_id: str,
        prompt: str,
        output: str,
        swarm_used: str,
        execution_result: dict,
        quality_score: float,
        execution_time: float,
    ) -> None:
        logger.debug(
            "feedback task=%s swarm=%s quality=%.2f time=%.1fs",
            task_id, swarm_used, quality_score, execution_time,
        )


_feedback_collector: FeedbackCollector | None = None


def get_feedback_collector() -> FeedbackCollector:
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector
