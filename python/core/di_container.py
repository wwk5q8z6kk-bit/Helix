"""Dependency injection container for Helix.

Lightweight wiring of core services at application startup.
Uses lazy initialization — services are created on first access.
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class HelixContainer:
    """Central service container for the Helix Python AI layer."""

    def __init__(self) -> None:
        self._bridge = None
        self._llm_router = None
        self._event_bus = None
        self._orchestrator = None
        self._learning_optimizer = None

    @property
    def bridge(self):
        if self._bridge is None:
            from core.rust_bridge import RustCoreBridge
            self._bridge = RustCoreBridge()
        return self._bridge

    @property
    def llm_router(self):
        if self._llm_router is None:
            from core.llm.intelligent_llm_router import IntelligentLLMRouter
            self._llm_router = IntelligentLLMRouter()
        return self._llm_router

    @property
    def event_bus(self):
        if self._event_bus is None:
            from core.event_bus import InMemoryEventBus
            self._event_bus = InMemoryEventBus()
        return self._event_bus

    @property
    def orchestrator(self):
        if self._orchestrator is None:
            try:
                from core.orchestration.unified_orchestrator import create_orchestrator
                self._orchestrator = create_orchestrator(
                    event_bus=self.event_bus,
                    memory_store=self.bridge,
                    llm_router=self.llm_router,
                )
                logger.info("Orchestrator initialized")
            except Exception:
                logger.exception("Failed to initialize orchestrator")
        return self._orchestrator

    @property
    def learning_optimizer(self):
        if self._learning_optimizer is None:
            from core.learning.learning_optimizer import LearningOptimizer

            self._learning_optimizer = LearningOptimizer()
            self._subscribe_learning_optimizer()
        return self._learning_optimizer

    def _subscribe_learning_optimizer(self) -> None:
        """Subscribe the learning optimizer to TASK_COMPLETED events."""
        import asyncio
        from core.interfaces.event_bus import EventType

        optimizer = self._learning_optimizer

        async def on_task_completed(data):
            agent_id = data.get("agent_id", "unknown")
            quality = data.get("quality_score", 0.0)
            duration_ms = data.get("execution_time", 0.0) * 1000

            # Ensure agent is registered
            if agent_id not in optimizer.agent_learning:
                optimizer.register_agent(agent_id, agent_id)

            learning = optimizer.agent_learning[agent_id]
            learning.performance.update_metrics(duration_ms, quality * 100)

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.event_bus.subscribe(EventType.TASK_COMPLETED, on_task_completed))
            else:
                loop.run_until_complete(self.event_bus.subscribe(EventType.TASK_COMPLETED, on_task_completed))
        except RuntimeError:
            logger.debug("No event loop available for learning optimizer subscription")

    def status(self) -> Dict[str, Any]:
        """Report which services are initialized."""
        return {
            "bridge": self._bridge is not None,
            "llm_router": self._llm_router is not None,
            "event_bus": self._event_bus is not None,
            "orchestrator": self._orchestrator is not None,
            "learning_optimizer": self._learning_optimizer is not None,
        }


# Global container
_container: Optional[HelixContainer] = None


def get_container() -> HelixContainer:
    global _container
    if _container is None:
        _container = HelixContainer()
    return _container


def init_container() -> HelixContainer:
    return get_container()


def shutdown_container() -> None:
    global _container
    _container = None
