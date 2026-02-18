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
        self._swarm_orchestrator = None
        self._reasoner = None
        self._feedback_handler = None
        self._budget_tracker = None
        self._composio_bridge = None
        self._analytics_engine = None
        self._report_generator = None
        self._notification_service = None
        self._scheduled_actions = None
        self._resource_manager = None
        self._source_registry = None

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
                    rust_bridge=self.bridge,
                    notification_service=self.notification_service,
                )
                # Register a general-purpose agent so execute_task() can find a handler
                from core.orchestration.unified_orchestrator import AgentCapabilities, AgentRole, TaskType
                general = AgentCapabilities(
                    agent_id="general",
                    role=AgentRole.DEVELOPMENT,
                    skills={"general", "reasoning", "coding", "review", "testing"},
                    task_types=set(TaskType),
                    max_concurrent_tasks=5,
                    performance_score=0.8,
                )
                self._orchestrator.register_agent("general", general)
                logger.info("Orchestrator initialized with default agent")
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

    @property
    def swarm_orchestrator(self):
        if self._swarm_orchestrator is None:
            try:
                from core.swarms.swarm_orchestrator import SwarmOrchestrator
                self._swarm_orchestrator = SwarmOrchestrator()
                logger.info("SwarmOrchestrator initialized")
            except Exception:
                logger.exception("Failed to initialize SwarmOrchestrator")
        return self._swarm_orchestrator

    @property
    def reasoner(self):
        if self._reasoner is None:
            from core.reasoning.agentic_reasoner import AgenticReasoner
            try:
                rb = self._bridge  # use already-created bridge; avoid recursive init
                cb = self._composio_bridge
            except Exception:
                rb, cb = None, None
            self._reasoner = AgenticReasoner(rust_bridge=rb, composio_bridge=cb)
            logger.info("AgenticReasoner initialized")
        return self._reasoner

    @property
    def budget_tracker(self):
        if self._budget_tracker is None:
            from core.middleware.budget_tracker import BudgetTracker
            self._budget_tracker = BudgetTracker()
            logger.info("BudgetTracker initialized")
        return self._budget_tracker

    @property
    def resource_manager(self):
        if self._resource_manager is None:
            from core.scheduling.resource_manager import ResourceManager
            orch = self.orchestrator
            if orch is not None:
                self._resource_manager = ResourceManager(
                    scheduler=orch._scheduler,
                    budget_tracker=self.budget_tracker,
                )
                # Inject back into orchestrator (breaks circular init)
                orch._resource_manager = self._resource_manager
                # Share breaker instances with LLM router (single source of truth)
                self.llm_router.set_circuit_breakers(self._resource_manager._breakers)
                logger.info("ResourceManager initialized and wired to orchestrator + router")
        return self._resource_manager

    @property
    def source_registry(self):
        if self._source_registry is None:
            from core.sources import SourceRegistry
            self._source_registry = SourceRegistry()
            logger.info("SourceRegistry initialized")
        return self._source_registry

    @property
    def composio_bridge(self):
        if self._composio_bridge is None:
            from core.composio.composio_bridge import ComposioBridge
            self._composio_bridge = ComposioBridge()
            logger.info("ComposioBridge initialized (mode=%s)", self._composio_bridge.tool_mode.value)
        return self._composio_bridge

    @property
    def analytics_engine(self):
        if self._analytics_engine is None:
            from core.analytics.analytics_engine import AnalyticsEngine
            self._analytics_engine = AnalyticsEngine()
            # Inject into LLM router so usage is recorded automatically
            self.llm_router.set_analytics(self._analytics_engine)
            logger.info("AnalyticsEngine initialized and wired to LLM router")
        return self._analytics_engine

    @property
    def report_generator(self):
        if self._report_generator is None:
            from core.analytics.report_generator import ReportGenerator
            self._report_generator = ReportGenerator(
                analytics_engine=self.analytics_engine,
                llm_router=self.llm_router,
            )
            logger.info("ReportGenerator initialized")
        return self._report_generator

    @property
    def notification_service(self):
        if self._notification_service is None:
            from core.notifications.notification_service import NotificationService
            self._notification_service = NotificationService(bridge=self.bridge)
            logger.info("NotificationService initialized with Rust bridge")
        return self._notification_service

    @property
    def scheduled_actions(self):
        if self._scheduled_actions is None:
            from core.analytics.scheduled_actions import ScheduledActions
            self._scheduled_actions = ScheduledActions(
                bridge=self.bridge,
                analytics_engine=self.analytics_engine,
            )
            logger.info("ScheduledActions initialized")
        return self._scheduled_actions

    @property
    def feedback_handler(self):
        if self._feedback_handler is None:
            from core.feedback.feedback_handler import FeedbackHandler
            self._feedback_handler = FeedbackHandler(
                event_bus=self.event_bus,
                learning_optimizer=self.learning_optimizer,
            )
            logger.info("FeedbackHandler initialized")
        return self._feedback_handler

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
            "swarm_orchestrator": self._swarm_orchestrator is not None,
            "reasoner": self._reasoner is not None,
            "feedback_handler": self._feedback_handler is not None,
            "budget_tracker": self._budget_tracker is not None,
            "composio_bridge": self._composio_bridge is not None,
            "analytics_engine": self._analytics_engine is not None,
            "report_generator": self._report_generator is not None,
            "notification_service": self._notification_service is not None,
            "scheduled_actions": self._scheduled_actions is not None,
            "resource_manager": self._resource_manager is not None,
            "source_registry": self._source_registry is not None,
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
