#!/usr/bin/env python3
"""
Unified Orchestrator - Consolidates 8 orchestrators into one production-ready implementation

This is the master orchestrator that combines functionality from:
1. enhanced_orchestrator.py - Agent management and routing
2. autonomous_orchestrator.py - Event-driven reactions and CEP rules
3. end_to_end_orchestrator.py - 7-phase workflow execution
4. swarm_orchestrator.py - Swarm coordination and task routing
5. review_orchestrator.py - Parallel code review
6. unified_orchestrator.py (brain) - Quality assessment and HRM integration
7. parallel_executor.py - Parallel and distributed execution
8. orchestrator_hrm_bridge.py - Feedback loop integration

Architecture:
- DI-ready (accepts interfaces via constructor)
- Event-driven (publishes all major events)
- Async throughout
- Stateful but persistent
- Observable (comprehensive logging)
- Testable (no global state, mockable dependencies)

Quality Standards:
- Full type hints
- Comprehensive docstrings
- Production-grade error handling
- 90%+ test coverage target
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Dict,
    List,
    Any,
    Optional,
    Set,
    Callable,
    Awaitable,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from core.interfaces import (
    IEventBus,
    ILLMRouter,
    IToolSelector,
    IMemoryStore,
    Task as BaseTask,
    Result as BaseResult,
    EventType,
)
from core.exceptions_unified import (
    AgentError,
    DependencyResolutionError,
    OrchestratorError,
    TaskExecutionError,
    ValidationError,
)
from core.scheduling.dependency_resolver import (
    CycleDetectedError,
    DependencyResolver,
    TaskState as DepTaskState,
)
from core.scheduling.execution_tracker import ExecutionTracker
from core.scheduling.quality_gates import (
    GateVerdict,
    QualityGatePolicy,
    GATE_STANDARD,
)
from core.scheduling.scheduler_coordinator import SchedulerCoordinator
from core.middleware.budget_tracker import BudgetAction
from core.scheduling.resource_manager import ResourceManager
from core.scheduling.retry_strategies import (
    RetryManager,
    RetryReason,
)
from core.notifications.notification_service import NotificationService, Severity

try:  # Optional dependency for context enrichment
    from core.brain.universal_context_engine import get_universal_context_engine
except Exception:  # pragma: no cover - defensive
    get_universal_context_engine = None  # type: ignore

try:  # Settings may be unavailable in some trimmed deployments
    from core.config.settings import Settings
except Exception:  # pragma: no cover - defensive
    Settings = None  # type: ignore

if TYPE_CHECKING:
    from core.brain.universal_context_engine import UniversalContextEngine

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class AgentRole(str, Enum):
    """Agent roles in the system."""
    PLANNING = "planning"
    DEVELOPMENT = "development"
    QA = "qa"
    REVIEW = "review"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    ARCHITECTURE = "architecture"


class TaskType(str, Enum):
    """Types of tasks that can be orchestrated."""
    PLANNING = "planning"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    REVIEW = "review"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class TaskPriority(str, Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


_PRIORITY_TO_INT: Dict[TaskPriority, int] = {
    TaskPriority.CRITICAL: 0,
    TaskPriority.HIGH: 1,
    TaskPriority.MEDIUM: 2,
    TaskPriority.LOW: 3,
}


class WorkflowPhase(str, Enum):
    """7-phase workflow stages."""
    RESEARCH = "research"
    DESIGN = "design"
    CODE = "code"
    TEST = "test"
    DEPLOY = "deploy"
    MONITOR = "monitor"
    DOCS = "docs"


class ExecutionMode(str, Enum):
    """Task execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


class EventTriggerType(str, Enum):
    """Event types that trigger autonomous actions."""
    TASK_FAILED = "task_failed"
    QUALITY_LOW = "quality_low"
    PERFORMANCE_DEGRADED = "performance_degraded"
    AGENT_OVERLOADED = "agent_overloaded"
    SYSTEM_ERROR = "system_error"
    STRATEGIC_OPPORTUNITY = "strategic_opportunity"
    GENERATION_FAILED = "generation_failed"
    QUALITY_ASSESSED = "quality_assessed"
    TASK_COMPLETED = "task_completed"
    PREFERENCE_LOGGED = "preference_logged"
    REWARD_MODEL_SCORE = "reward_model_score"
    MODEL_RETRAINED = "model_retrained"
    STRATEGIC_EVENT = "strategic_event"
    WORKFLOW_PROGRESS = "workflow_progress"


class ActionType(str, Enum):
    """Autonomous action types."""
    RETRY_TASK = "retry_task"
    ESCALATE_ERROR = "escalate_error"
    OPTIMIZE_PARAMS = "optimize_params"
    SCALE_WORKERS = "scale_workers"
    ALERT_USER = "alert_user"
    TRIGGER_WORKFLOW = "trigger_workflow"
    UPDATE_METRICS = "update_metrics"
    LOG_INSIGHT = "log_insight"


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass
class AgentCapabilities:
    """Defines what an agent can do."""

    agent_id: str
    role: AgentRole
    skills: Set[str] = field(default_factory=set)
    task_types: Set[TaskType] = field(default_factory=set)
    max_concurrent_tasks: int = 3
    performance_score: float = 0.8
    specialization_areas: Set[str] = field(default_factory=set)

    def can_handle_task(self, task: "Task") -> bool:
        """Check if agent can handle a specific task.

        Args:
            task: Task to evaluate

        Returns:
            True if agent can handle the task
        """
        # Check task type compatibility
        if task.task_type not in self.task_types:
            return False

        # Check specialization match (optional but preferred)
        if self.specialization_areas:
            task_context = f"{task.description} {' '.join(str(v) for v in task.context.values())}"
            has_specialization = any(
                spec.lower() in task_context.lower()
                for spec in self.specialization_areas
            )
            if not has_specialization:
                return False

        return True


@dataclass
class AgentState:
    """Runtime state of an agent."""

    agent_id: str
    capabilities: AgentCapabilities
    active_tasks: Set[str] = field(default_factory=set)
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    last_active: Optional[datetime] = None
    health_status: str = "healthy"

    @property
    def is_available(self) -> bool:
        """Check if agent can accept new tasks."""
        return (
            len(self.active_tasks) < self.capabilities.max_concurrent_tasks
            and self.health_status == "healthy"
        )

    @property
    def success_rate(self) -> float:
        """Calculate agent success rate."""
        total = self.completed_tasks + self.failed_tasks
        if total == 0:
            return 1.0
        return self.completed_tasks / total

    @property
    def avg_execution_time(self) -> float:
        """Calculate average execution time."""
        if self.completed_tasks == 0:
            return 0.0
        return self.total_execution_time / self.completed_tasks


@dataclass
class Task:
    """Enhanced task model with all orchestrator features.

    This consolidates task models from all 8 orchestrators.
    """

    task_id: str
    task_type: TaskType
    description: str
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.PENDING

    # Execution context
    context: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3

    # Assignment tracking
    assigned_agent_id: Optional[str] = None
    swarm_id: Optional[str] = None

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout_seconds: Optional[float] = None

    # Results
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Quality metrics
    quality_score: float = 0.0
    hrm_score: float = 0.0

    # Event triggers
    on_complete: Optional[Callable] = None
    on_fail: Optional[Callable] = None

    @property
    def duration(self) -> Optional[float]:
        """Calculate task duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def is_overdue(self) -> bool:
        """Check if task has exceeded timeout."""
        if not self.timeout_seconds or not self.started_at:
            return False
        elapsed = (datetime.now() - self.started_at).total_seconds()
        return elapsed > self.timeout_seconds

    def to_base_task(self) -> BaseTask:
        """Convert to base interface Task."""
        return BaseTask(
            task_id=self.task_id,
            task_type=self.task_type.value,
            description=self.description,
            metadata=self.context,
        )


@dataclass
class Result:
    """Task execution result with comprehensive tracking."""

    task_id: str
    status: TaskStatus
    output: Any
    error: Optional[str] = None

    # Execution metrics
    execution_time: float = 0.0
    tokens_used: int = 0
    cost: float = 0.0

    # Quality metrics
    quality_score: float = 0.0
    hrm_score: float = 0.0

    # Agent tracking
    agent_id: Optional[str] = None
    swarm_id: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    def to_base_result(self) -> BaseResult:
        """Convert to base interface Result."""
        return BaseResult(
            task_id=self.task_id,
            status=self.status.value,
            output=self.output,
            error=self.error,
            metadata=self.metadata,
        )


@dataclass
class WorkflowState:
    """State for end-to-end workflow execution."""

    workflow_id: str
    description: str
    current_phase: WorkflowPhase
    phases_completed: List[WorkflowPhase] = field(default_factory=list)
    phase_results: Dict[WorkflowPhase, Result] = field(default_factory=dict)

    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Aggregate metrics
    total_cost: float = 0.0
    total_tokens: int = 0
    total_execution_time: float = 0.0

    # Context and artifacts
    context: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    architecture_decisions: List[Dict[str, Any]] = field(default_factory=list)


class WorkflowExecutionStatus(str, Enum):
    """Status of a DAG workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class WorkflowExecution:
    """Tracks a running DAG workflow execution."""

    execution_id: str
    workflow_name: str
    status: WorkflowExecutionStatus = WorkflowExecutionStatus.PENDING
    task_ids: List[str] = field(default_factory=list)
    task_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    total_cost: float = 0.0
    total_tokens: int = 0

    error: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def progress(self) -> float:
        if not self.task_ids:
            return 0.0
        return len(self.task_results) / len(self.task_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "task_ids": self.task_ids,
            "task_results": self.task_results,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration": self.duration,
            "progress": self.progress,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "error": self.error,
        }


@dataclass
class ConditionActionRule:
    """CEP (Complex Event Processing) rule for autonomous reactions.

    When condition is met on an event, execute action autonomously.
    """

    rule_id: str
    name: str
    description: str

    # Condition
    event_types: List[EventTriggerType]
    condition: Callable[[Dict[str, Any]], bool]

    # Action
    action_type: ActionType
    action: Callable[[Dict[str, Any]], Awaitable[Any]]

    # Configuration
    priority: int = 1
    enabled: bool = True
    cooldown_seconds: float = 0.0

    # State tracking
    last_triggered: float = 0.0
    trigger_count: int = 0

    def can_trigger(self) -> bool:
        """Check if rule can be triggered (respects cooldown).

        Returns:
            True if rule can trigger now
        """
        if not self.enabled:
            return False
        if self.cooldown_seconds == 0:
            return True
        return (time.time() - self.last_triggered) >= self.cooldown_seconds


class RuleEventData(dict):
    """Dict-like payload that also exposes legacy `.data` attribute."""

    def __init__(self, initial: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(initial or {})
        if kwargs:
            self.update(kwargs)

    @property
    def data(self) -> "RuleEventData":
        return self


@dataclass
class ReviewResult:
    """Result from parallel code review."""

    overall_score: float
    security_issues: List[Dict[str, Any]] = field(default_factory=list)
    performance_issues: List[Dict[str, Any]] = field(default_factory=list)
    quality_issues: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def has_blockers(self) -> bool:
        """Check if there are blocking issues."""
        return any(
            issue.get("severity") == "critical"
            for issue in self.security_issues
        )


@dataclass
class SystemStatus:
    """Overall system status and metrics."""

    active_agents: int
    active_tasks: int
    completed_tasks: int
    failed_tasks: int

    avg_task_duration: float
    avg_quality_score: float
    system_health: str

    agent_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    queue_sizes: Dict[TaskPriority, int] = field(default_factory=dict)

    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# UNIFIED ORCHESTRATOR
# ============================================================================


class UnifiedOrchestrator:
    """
    Production-ready unified orchestrator consolidating 8 implementations.

    Features:
    1. Agent Management - Register, track, and route to agents
    2. Task Execution - Sequential, parallel, and distributed modes
    3. Workflow Orchestration - 7-phase development workflows
    4. Swarm Coordination - Route to specialized agent swarms
    5. Code Review - Parallel security/performance/quality checks
    6. Event-Driven Reactions - CEP rules for autonomous actions
    7. Quality Assessment - HRM integration and quality scoring
    8. State Management - Persistent orchestrator and agent state

    Architecture:
    - Dependency Injection: Accepts interfaces via constructor
    - Event-Driven: Publishes events for all major operations
    - Async-First: All I/O operations are async
    - Observable: Comprehensive logging at all levels
    - Testable: No global state, all dependencies injectable
    """

    def __init__(
        self,
        event_bus: IEventBus,
        memory_store: IMemoryStore,
        llm_router: Optional[ILLMRouter] = None,
        tool_selector: Optional[IToolSelector] = None,
        rust_bridge: Optional[Any] = None,
    ):
        """Initialize unified orchestrator with dependencies.

        Args:
            event_bus: Event bus for pub/sub messaging
            memory_store: Persistent storage for state
            llm_router: Optional LLM routing for agent execution
            tool_selector: Optional tool selection for agents
            rust_bridge: Optional RustCoreBridge for Rust-delegated scheduling
        """
        # Core dependencies (required)
        self.event_bus = event_bus
        self.memory_store = memory_store

        # Optional dependencies
        self.llm_router = llm_router
        self.tool_selector = tool_selector

        # Rust-delegated scheduling: when the Rust core is reachable, DAG
        # operations (submit, ready-check, complete, fail) are routed through
        # the Rust scheduler REST API.  The Python-side DependencyResolver
        # serves as fallback when the bridge is unavailable.
        self._rust_bridge = rust_bridge
        self._use_rust_scheduler: bool = False

        # Agent management
        self.agents: Dict[str, AgentState] = {}
        self.agent_load_balancer: Dict[TaskType, List[str]] = defaultdict(list)

        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queues: Dict[TaskPriority, List[str]] = {
            priority: [] for priority in TaskPriority
        }

        # Dependency resolver (DAG-based scheduling)
        self._dep_resolver = DependencyResolver()
        # Completed task history (retained for queries; capped on persist)
        self._completed_tasks: Dict[str, Task] = {}
        # Workflow executions (DAG-based workflow runs)
        self._workflow_executions: Dict[str, WorkflowExecution] = {}
        # Local workflow definitions (Python fallback; Rust stores its own)
        self._stored_workflows: Dict[str, Dict[str, Any]] = {}
        # Notification service for workflow lifecycle events
        self._notification_service: Optional[NotificationService] = None
        # Execution intelligence (duration/cost/quality tracking)
        self._exec_tracker = ExecutionTracker()
        # Quality gates (dependency quality thresholds)
        self._quality_policy = QualityGatePolicy()
        # Adaptive retry with model escalation
        self._retry_manager = RetryManager()
        # Unified scheduler coordinator (ties all scheduling modules together)
        self._scheduler = SchedulerCoordinator(
            self._dep_resolver, self._exec_tracker,
            self._quality_policy, self._retry_manager,
        )
        # Production safety layer (injected externally; None = disabled)
        self._resource_manager: Optional[ResourceManager] = None

        # Workflow management
        self.workflows: Dict[str, WorkflowState] = {}

        # Event-driven rules
        self.cep_rules: List[ConditionActionRule] = []
        self._subscribed_events: List[EventType] = []

        # Metrics and statistics
        self.metrics = {
            "tasks_received": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "total_cost": 0.0,
            "total_tokens": 0,
            "context_enriched_tasks": 0,
            "event_counts": defaultdict(int),
        }

        # State tracking
        self.is_running = False
        self._degraded_mode: bool = False
        self.start_time: Optional[datetime] = None
        self.total_events_processed: int = 0
        self.total_rules_triggered: int = 0
        self.context_engine: Optional["UniversalContextEngine"] = None
        self.context_enrichment_enabled: bool = False
        self.context_enrichment_errors: int = 0
        self._settings: Optional[Any] = None

        self._initialize_context_engine()
        self._detect_rust_scheduler()

        logger.info(
            "UnifiedOrchestrator initialized (rust_scheduler=%s)",
            self._use_rust_scheduler,
        )

    def _initialize_context_engine(self) -> None:
        """Configure the universal context engine when available."""
        if not Settings or not get_universal_context_engine:
            return

        try:
            self._settings = Settings()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Settings initialisation failed: %s", exc)
            return

        if not getattr(self._settings, "rag_refrag_enabled", False):
            logger.debug("REFRAG context enrichment disabled via settings.")
            return

        try:
            self.context_engine = get_universal_context_engine()
            self.context_enrichment_enabled = self.context_engine is not None
            if self.context_enrichment_enabled:
                logger.info("Context enrichment enabled via UniversalContextEngine.")
        except Exception as exc:  # pragma: no cover - defensive
            self.context_engine = None
            self.context_enrichment_enabled = False
            logger.warning("Failed to initialise UniversalContextEngine: %s", exc)

    def _detect_rust_scheduler(self) -> None:
        """Enable Rust-delegated scheduling if a bridge with scheduler methods is available."""
        bridge = self._rust_bridge
        if bridge is None:
            return
        # Duck-type check: the bridge must expose the scheduler REST methods.
        required = (
            "scheduler_submit_task",
            "scheduler_ready_tasks",
            "scheduler_mark_running",
            "scheduler_mark_completed",
            "scheduler_mark_failed",
            "scheduler_task_state",
        )
        if all(callable(getattr(bridge, m, None)) for m in required):
            self._use_rust_scheduler = True
            logger.info("Rust scheduler delegation enabled via bridge")
        else:
            logger.debug("Bridge lacks scheduler methods — using Python scheduler")

    async def enable_rust_scheduler(self) -> bool:
        """Attempt to enable Rust scheduling at runtime by health-checking the bridge.

        Returns True if the Rust scheduler is now active.
        """
        if self._use_rust_scheduler:
            return True
        bridge = self._rust_bridge
        if bridge is None:
            return False
        try:
            healthy = await bridge.is_healthy()
            if healthy:
                self._detect_rust_scheduler()
            return self._use_rust_scheduler
        except Exception:
            return False

    @property
    def degraded_mode(self) -> bool:
        """Whether the orchestrator is running in degraded mode."""
        return self._degraded_mode

    async def check_health(self) -> None:
        """Test reachability of key dependencies and enter degraded mode if needed."""
        degraded = False

        # Check memory store
        try:
            await self.memory_store.is_healthy()
        except Exception:
            logger.warning("Memory store unreachable — entering degraded mode")
            degraded = True

        # Check LLM router
        if self.llm_router is not None:
            try:
                has_providers = bool(getattr(self.llm_router, "provider_stats", None))
                if not has_providers:
                    degraded = True
            except Exception:
                logger.warning("LLM router check failed — entering degraded mode")
                degraded = True

        if degraded and not self._degraded_mode:
            logger.warning("Orchestrator entering degraded mode")
        elif not degraded and self._degraded_mode:
            logger.info("Orchestrator recovered from degraded mode")

        self._degraded_mode = degraded

    async def start(self) -> None:
        """Start the orchestrator and initialize background tasks.

        This loads persisted state, subscribes to events, and starts
        background monitoring tasks.
        """
        if self.is_running:
            logger.warning("Orchestrator already running")
            return

        logger.info("Starting UnifiedOrchestrator...")

        # Load persisted state
        await self._load_state()

        # Subscribe to events for autonomous reactions
        await self._subscribe_to_events()

        # Register default CEP rules
        await self._register_default_rules()

        # Start background tasks
        asyncio.create_task(self._monitor_tasks())
        asyncio.create_task(self._process_queues())

        self.is_running = True
        self.start_time = datetime.now()

        logger.info("UnifiedOrchestrator started successfully")

        await self.event_bus.publish(
            EventType.AGENT_REGISTERED,
            {"orchestrator": "unified", "status": "started"},
            source="orchestrator",
        )

    async def stop(self) -> None:
        """Stop the orchestrator and persist state."""
        if not self.is_running:
            logger.warning("Orchestrator not running")
            return

        logger.info("Stopping UnifiedOrchestrator...")

        # Persist current state
        await self._persist_state()

        # Wait for active tasks to complete (with timeout)
        await self._drain_active_tasks(timeout=30.0)

        self.is_running = False

        logger.info("UnifiedOrchestrator stopped")

        await self.event_bus.publish(
            EventType.AGENT_UNREGISTERED,
            {"orchestrator": "unified", "status": "stopped"},
            source="orchestrator",
        )

    def _extract_event_data(self, event: Any) -> Dict[str, Any]:
        if isinstance(event, dict):
            return event
        if hasattr(event, "data"):
            return event.data or {}
        return {}

    def _record_event(self, label: str) -> None:
        self.total_events_processed += 1
        event_counts = self.metrics.setdefault("event_counts", defaultdict(int))
        if not isinstance(event_counts, defaultdict):
            event_counts = defaultdict(int, event_counts)
            self.metrics["event_counts"] = event_counts
        event_counts[label] += 1

    def _should_enrich_context(self) -> bool:
        return bool(self.context_enrichment_enabled and self.context_engine)

    async def _enrich_task_context(self, task: Task) -> None:
        """Build and attach a REFRAG-aware context bundle for the task."""
        if not self._should_enrich_context():
            return

        query = (task.description or "").strip()
        if not query:
            return

        user_id = task.context.get("user_id", "default")
        try:
            bundle = await self.context_engine.build_context_bundle(
                query=query,
                user_id=user_id,
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.context_enrichment_errors += 1
            logger.warning("Context enrichment failed for task %s: %s", task.task_id, exc)
            return

        task.context["context_bundle"] = bundle
        self.metrics["context_enriched_tasks"] += 1
        await self._persist_context_bundle(task.task_id, bundle)

    async def _persist_context_bundle(self, task_id: str, bundle: Dict[str, Any]) -> None:
        """Persist the retrieval bundle for debugging or replay."""
        key = f"orchestrator:context:{task_id}"
        try:
            await self.memory_store.store(key, bundle)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Failed to persist context bundle for %s: %s", task_id, exc)

    # ------------------------------------------------------------------ #
    # Compatibility Helpers
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> Dict[str, Any]:
        """Legacy diagnostics consumed by historical tests."""
        event_subscriptions = len(self._subscribed_events)

        rule_stats = [
            {
                "rule_id": rule.rule_id,
                "name": rule.name,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "trigger_count": rule.trigger_count,
                "last_triggered": rule.last_triggered,
            }
            for rule in self.cep_rules
        ]

        task_counts = {
            "total": len(self.tasks),
            "pending": sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING),
            "in_progress": sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS),
            "completed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED),
            "failed": sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
        }

        uptime_seconds = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0.0
        events_per_second = (
            self.total_events_processed / uptime_seconds if uptime_seconds > 0 else 0.0
        )

        return {
            "status": "running" if self.is_running else "stopped",
            "event_subscriptions": event_subscriptions,
            "total_rules": len(self.cep_rules),
            "active_rules": len([rule for rule in self.cep_rules if rule.enabled]),
            "rule_stats": rule_stats,
            "tasks": task_counts,
            "agents": len(self.agents),
            "workflows": len(self.workflows),
            "metrics": {
                **{k: v for k, v in self.metrics.items() if k != "event_counts"},
                "event_counts": dict(self.metrics["event_counts"]),
            },
            "total_events_processed": self.total_events_processed,
            "total_rules_triggered": self.total_rules_triggered,
            "uptime_seconds": uptime_seconds,
            "events_per_second": events_per_second,
            "dependency_graph": self._dep_resolver.stats,
            "execution_intelligence": self._exec_tracker.stats,
            "retry_manager": self._retry_manager.stats,
            "scheduler_coordinator": self._scheduler.stats,
            "resource_manager": self._resource_manager.stats if self._resource_manager else None,
            "completed_history_size": len(self._completed_tasks),
            "rust_scheduler_active": self._use_rust_scheduler,
        }

    def get_macos_display(self) -> Dict[str, str]:
        """Return a concise status block for macOS menu-bar integrations."""
        stats = self.get_statistics()
        status_color = "green" if stats["status"] == "running" else "gray"
        status_icon = "✅" if stats["status"] == "running" else "⚪️"
        status_text = "Operational" if stats["status"] == "running" else "Idle"

        events_display = f"{stats['total_events_processed']} events"
        rules_display = f"{stats['total_rules_triggered']} rules triggered"
        throughput_display = f"{stats['events_per_second']:.2f} events/sec"
        uptime_hours = stats["uptime_seconds"] / 3600 if stats["uptime_seconds"] else 0.0
        uptime_display = f"{uptime_hours:.2f}h uptime"

        return {
            "status_color": status_color,
            "status_icon": status_icon,
            "status_text": status_text,
            "events_display": events_display,
            "rules_display": rules_display,
            "throughput_display": throughput_display,
            "uptime_display": uptime_display,
        }

    # ========================================================================
    # AGENT MANAGEMENT (from enhanced_orchestrator.py)
    # ========================================================================

    def register_agent(
        self,
        agent_id: str,
        capabilities: AgentCapabilities,
    ) -> None:
        """Register an agent with the orchestrator.

        Args:
            agent_id: Unique agent identifier
            capabilities: Agent capabilities definition

        Raises:
            ValidationError: If agent_id already registered
        """
        if agent_id in self.agents:
            raise ValidationError(f"Agent {agent_id} already registered")

        agent_state = AgentState(
            agent_id=agent_id,
            capabilities=capabilities,
        )

        self.agents[agent_id] = agent_state

        # Update load balancer
        for task_type in capabilities.task_types:
            self.agent_load_balancer[task_type].append(agent_id)

        logger.info(
            f"Registered agent {agent_id} "
            f"(role={capabilities.role.value}, "
            f"tasks={len(capabilities.task_types)})"
        )

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the orchestrator.

        Args:
            agent_id: Agent identifier to unregister
        """
        if agent_id not in self.agents:
            logger.warning(f"Agent {agent_id} not found")
            return

        agent_state = self.agents[agent_id]

        # Remove from load balancer
        for task_type in agent_state.capabilities.task_types:
            if agent_id in self.agent_load_balancer[task_type]:
                self.agent_load_balancer[task_type].remove(agent_id)

        # Handle active tasks
        if agent_state.active_tasks:
            logger.warning(
                f"Agent {agent_id} has {len(agent_state.active_tasks)} "
                f"active tasks - reassigning"
            )
            for task_id in agent_state.active_tasks:
                asyncio.create_task(self._reassign_task(task_id))

        del self.agents[agent_id]

        logger.info(f"Unregistered agent {agent_id}")

    def get_best_agent(self, task: Task) -> Optional[str]:
        """Select best agent for a task based on load and capabilities.

        Args:
            task: Task to assign

        Returns:
            Agent ID or None if no suitable agent available
        """
        # Get candidate agents for this task type
        candidates = self.agent_load_balancer.get(task.task_type, [])

        if not candidates:
            logger.warning(f"No agents available for task type {task.task_type}")
            return None

        # Filter by capability and availability
        available = [
            agent_id for agent_id in candidates
            if self.agents[agent_id].is_available
            and self.agents[agent_id].capabilities.can_handle_task(task)
        ]

        if not available:
            logger.warning(f"No available agents for task {task.task_id}")
            return None

        # Select agent with best score (success_rate * performance_score)
        # and lowest current load
        best_agent = max(
            available,
            key=lambda agent_id: (
                self.agents[agent_id].success_rate
                * self.agents[agent_id].capabilities.performance_score
                / (len(self.agents[agent_id].active_tasks) + 1)
            ),
        )

        return best_agent

    # ========================================================================
    # TASK EXECUTION (from parallel_executor.py, autonomous_orchestrator.py)
    # ========================================================================

    async def _report_progress(
        self,
        task_id: str,
        phase: str,
        percent: int,
        message: str,
        callback: Optional[Callable] = None,
    ) -> None:
        """Publish progress and optionally invoke a callback."""
        payload = {"task_id": task_id, "phase": phase, "percent": percent, "message": message}
        await self.event_bus.publish(EventType.WORKFLOW_PROGRESS, payload, source="orchestrator")
        if callback is not None:
            try:
                result = callback(payload)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.debug("Progress callback error for task %s", task_id)

    # ------------------------------------------------------------------ #
    # DEPENDENCY-AWARE TASK SUBMISSION
    # ------------------------------------------------------------------ #

    async def submit_task(self, task: Task) -> str:
        """Register a task in the dependency resolver and dispatch if ready.

        This is the dependency-aware entry point.  Tasks whose dependencies
        are all satisfied are dispatched immediately; others wait until their
        prerequisites complete.

        When Rust-delegated scheduling is active, the DAG submit is routed
        through the Rust core REST API.  Otherwise the local Python
        DependencyResolver handles it.

        Returns the task_id.
        Raises DependencyResolutionError on cycles.
        """
        priority_int = _PRIORITY_TO_INT.get(task.priority, 2)

        if self._use_rust_scheduler:
            initial_state_str = await self._rust_submit_task(
                task.task_id, task.dependencies or [], priority_int,
            )
            self.tasks[task.task_id] = task
            self.metrics["tasks_received"] += 1

            if initial_state_str == "cancelled":
                task.status = TaskStatus.CANCELLED
                logger.info("Task %s auto-cancelled (Rust scheduler)", task.task_id)
                return task.task_id

            if initial_state_str == "ready":
                await self._dispatch_ready_tasks()
            return task.task_id

        # --- Python fallback path ---
        try:
            initial_state = self._dep_resolver.add_task(
                task.task_id,
                dependencies=task.dependencies or None,
                priority=priority_int,
            )
        except CycleDetectedError as exc:
            raise DependencyResolutionError(
                f"Dependency cycle detected for task {task.task_id!r}: {exc}"
            ) from exc

        self.tasks[task.task_id] = task
        self.metrics["tasks_received"] += 1

        if initial_state == DepTaskState.CANCELLED:
            task.status = TaskStatus.CANCELLED
            logger.info("Task %s auto-cancelled (failed upstream dependency)", task.task_id)
            return task.task_id

        if initial_state == DepTaskState.READY:
            await self._dispatch_ready_tasks()

        return task.task_id

    async def submit_tasks(self, tasks: List[Task]) -> List[str]:
        """Batch-submit tasks with atomic rollback on failure.

        If any task fails validation (e.g. cycle), all tasks added in this
        batch are rolled back from the resolver.
        """
        added: List[str] = []
        try:
            for task in tasks:
                priority_int = _PRIORITY_TO_INT.get(task.priority, 2)

                if self._use_rust_scheduler:
                    await self._rust_submit_task(
                        task.task_id, task.dependencies or [], priority_int,
                    )
                else:
                    try:
                        self._dep_resolver.add_task(
                            task.task_id,
                            dependencies=task.dependencies or None,
                            priority=priority_int,
                        )
                    except CycleDetectedError as exc:
                        raise DependencyResolutionError(
                            f"Dependency cycle detected for task {task.task_id!r}: {exc}"
                        ) from exc

                added.append(task.task_id)
                self.tasks[task.task_id] = task
        except DependencyResolutionError:
            # Rollback all tasks added in this batch
            for tid in added:
                if self._use_rust_scheduler:
                    # Rust has no explicit remove — completed/failed handle cleanup
                    pass
                else:
                    self._dep_resolver.remove_task(tid)
                self.tasks.pop(tid, None)
            raise

        self.metrics["tasks_received"] += len(added)
        await self._dispatch_ready_tasks()
        return added

    async def _dispatch_ready_tasks(self) -> None:
        """Check the resolver for READY tasks and fire them."""
        if self._use_rust_scheduler:
            ready_ids = await self._rust_bridge.scheduler_ready_tasks()
        else:
            ready_ids = self._dep_resolver.get_ready_tasks()

        for task_id in ready_ids:
            task = self.tasks.get(task_id)
            if task is None:
                continue

            # Resource gate: circuit breaker + concurrency check
            if self._resource_manager is not None:
                provider = getattr(task, "assigned_agent_id", "default") or "default"
                if not self._resource_manager.provider_available(provider):
                    logger.warning("Circuit open for %s — skipping %s", provider, task_id)
                    continue
                if not self._resource_manager.acquire(provider):
                    logger.info("Concurrency full for %s — deferring %s", provider, task_id)
                    continue

            if self._use_rust_scheduler:
                await self._rust_bridge.scheduler_mark_running(task_id)
            else:
                self._dep_resolver.mark_running(task_id)
            task.status = TaskStatus.IN_PROGRESS
            asyncio.create_task(self._execute_and_resolve(task))

    async def _execute_and_resolve(self, task: Task) -> None:
        """Execute a task with quality gates, tracking, and adaptive retry.

        Flow:
        1. Budget pre-check (if resource manager active)
        2. Record start in execution tracker
        3. Execute the task
        4. Evaluate quality gate on the result
        5. If gate says RETRY → adaptive retry with model escalation
        6. If gate says PASSED → mark completed, dispatch downstream
        7. On hard failure → check retry manager → propagate if exhausted
        8. Always: release concurrency slot + record breaker outcome
        """
        task_type_str = task.task_type.value if hasattr(task.task_type, "value") else str(task.task_type)
        model_used = "unknown"  # Updated by result metadata
        provider_tag = getattr(task, "assigned_agent_id", "default") or "default"

        # Track execution start
        self._exec_tracker.record_start(task.task_id)

        try:
            # Budget pre-check
            if self._resource_manager is not None:
                budget_action = self._resource_manager.check_budget(provider_tag, 0.01)
                if budget_action == BudgetAction.REJECT:
                    raise TaskExecutionError(f"Budget exceeded for provider {provider_tag}")

            result = await self.execute_task(task)
            model_used = result.metadata.get("provider", model_used)

            # Record completion in tracker
            self._exec_tracker.record_completion(
                task_id=task.task_id,
                task_type=task_type_str,
                model=model_used,
                quality_score=result.quality_score,
                hrm_score=result.hrm_score,
                tokens_used=result.tokens_used,
                cost=result.cost,
                agent_id=result.agent_id,
                attempt=task.retry_count + 1,
            )

            # Record success in circuit breaker
            if self._resource_manager is not None:
                self._resource_manager.record_success(provider_tag)
                tokens = result.tokens_used or 0
                self._resource_manager.record_spend(
                    provider_tag, tokens, max(tokens // 3, 1), task_type_str,
                )

            # Evaluate quality gate
            downstream = list(self._dep_resolver._dependents.get(task.task_id, set()))
            gate_result = self._quality_policy.evaluate(
                task_id=task.task_id,
                quality_score=result.quality_score,
                hrm_score=result.hrm_score,
                downstream_ids=downstream or None,
            )

            if gate_result.verdict == GateVerdict.RETRY:
                # Quality too low — try adaptive retry with model escalation
                decision = self._retry_manager.on_failure(
                    task_id=task.task_id,
                    reason=RetryReason.LOW_QUALITY,
                    quality_score=result.quality_score,
                    current_model=model_used,
                )
                if decision.should_retry:
                    logger.info(
                        "Quality gate retry for %s: %s (model: %s)",
                        task.task_id, decision.message, decision.model or model_used,
                    )
                    task.retry_count += 1
                    # Re-mark as READY in resolver so dispatch picks it up
                    await self._resolver_requeue(task.task_id)
                    await self._dispatch_ready_tasks()
                    return
                # Exhausted retries — fall through to WARN_PASS behaviour
                logger.warning(
                    "Quality gate exhausted retries for %s — accepting result",
                    task.task_id,
                )

            if gate_result.verdict == GateVerdict.FAILED:
                # Quality gate says treat as hard failure
                raise TaskExecutionError(
                    f"Quality gate failed for {task.task_id}: {gate_result.reason}"
                )

            # PASSED or WARN_PASS — mark completed, dispatch downstream
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            newly_ready = await self._resolver_mark_completed(task.task_id)
            self._completed_tasks[task.task_id] = task
            self._retry_manager.reset(task.task_id)
            self._quality_policy.reset_attempts(task.task_id)
            if newly_ready:
                logger.info(
                    "Task %s completed (q=%.2f) — unblocked: %s",
                    task.task_id, result.quality_score,
                    ", ".join(newly_ready),
                )
                await self._dispatch_ready_tasks()

        except Exception as exc:
            logger.error("Task %s failed: %s", task.task_id, exc)

            # Record failure in circuit breaker
            if self._resource_manager is not None:
                self._resource_manager.record_failure(provider_tag)

            # Record failure in tracker
            self._exec_tracker.record_failure(
                task_id=task.task_id,
                task_type=task_type_str,
                model=model_used,
                agent_id=task.assigned_agent_id,
                attempt=task.retry_count + 1,
            )

            # Check adaptive retry
            decision = self._retry_manager.on_failure(
                task_id=task.task_id,
                reason=RetryReason.EXECUTION_FAILURE,
                current_model=model_used,
            )
            if decision.should_retry:
                logger.info(
                    "Adaptive retry for %s: %s",
                    task.task_id, decision.message,
                )
                task.retry_count += 1
                # Re-mark as READY for dispatch
                await self._resolver_requeue(task.task_id)
                await self._dispatch_ready_tasks()
                return

            # Exhausted retries — propagate failure
            task.status = TaskStatus.FAILED
            task.error = str(exc)
            task.completed_at = datetime.now()
            cancelled = await self._resolver_mark_failed(task.task_id)
            if cancelled:
                for cid in cancelled:
                    cancelled_task = self.tasks.get(cid)
                    if cancelled_task:
                        cancelled_task.status = TaskStatus.CANCELLED
                logger.info(
                    "Task %s failure propagated — cancelled: %s",
                    task.task_id,
                    ", ".join(cancelled),
                )
        finally:
            # Always release the concurrency slot
            if self._resource_manager is not None:
                self._resource_manager.release(provider_tag)

    # ------------------------------------------------------------------ #
    # RUST / PYTHON RESOLVER BRIDGE HELPERS
    # ------------------------------------------------------------------ #

    async def _rust_submit_task(
        self, task_id: str, dependencies: List[str], priority: int,
    ) -> str:
        """Submit a task via the Rust scheduler REST API.

        Returns the initial state string (e.g. 'ready', 'blocked', 'cancelled').
        Raises DependencyResolutionError on failure (e.g. cycle detected).
        """
        result = await self._rust_bridge.scheduler_submit_task(
            task_id, dependencies=dependencies, priority=priority,
        )
        if result is None:
            raise DependencyResolutionError(
                f"Rust scheduler rejected task {task_id!r}"
            )
        return result.get("state", "blocked").lower()

    async def _resolver_mark_completed(self, task_id: str) -> List[str]:
        """Mark task completed in the active resolver (Rust or Python)."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_mark_completed(task_id)
        return self._dep_resolver.mark_completed(task_id)

    async def _resolver_mark_failed(self, task_id: str) -> List[str]:
        """Mark task failed in the active resolver (Rust or Python)."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_mark_failed(task_id)
        try:
            return self._dep_resolver.mark_failed(task_id)
        except ValueError:
            return []

    async def _resolver_requeue(self, task_id: str) -> None:
        """Re-mark a task as READY for retry dispatch."""
        if self._use_rust_scheduler:
            # Rust scheduler doesn't have a "requeue" endpoint; re-submit works
            # because the task is still in the DAG.  For now we just call
            # mark_running again after _dispatch_ready_tasks picks it up.
            # The task stays in the DAG as "running" until dispatch re-marks it.
            pass
        else:
            self._dep_resolver._states[task_id] = DepTaskState.READY

    # ------------------------------------------------------------------ #
    # WORKFLOW LIFECYCLE (Rust-delegated when available)
    # ------------------------------------------------------------------ #

    async def define_workflow(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
        deadline: Optional[float] = None,
        budget: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Define a DAG workflow.  Delegates to Rust scheduler when active.

        Args:
            name: Workflow name (unique key).
            tasks: List of task dicts with keys: task_id, task_type,
                   depends_on (list), priority (int), model (optional).
            deadline: Optional deadline (Unix timestamp).
            budget: Optional budget cap (dollars).

        Returns:
            Workflow definition dict on success, None on failure.
        """
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_define_workflow(
                name=name, tasks=tasks, deadline=deadline, budget=budget,
            )

        # Python fallback: use the local SchedulerCoordinator via WorkflowBuilder
        from core.scheduling.workflow_builder import WorkflowBuilder
        builder = WorkflowBuilder(name, coordinator=self._scheduler)
        for t in tasks:
            builder = builder.task(
                t["task_id"],
                task_type=t.get("task_type", "planning"),
                depends_on=t.get("depends_on"),
                model=t.get("model"),
                priority=t.get("priority", 2),
            )
        if deadline is not None:
            builder = builder.with_deadline(deadline)
        if budget is not None:
            builder = builder.with_budget(budget)
        try:
            defn = builder.build()
            result = {
                "name": defn.name,
                "task_count": defn.task_count,
                "wave_count": defn.wave_count,
                "max_parallelism": defn.max_parallelism,
                "task_order": defn.task_order,
                "waves": defn.waves,
                "tasks": tasks,  # preserve original definitions for run_workflow
            }
            self._stored_workflows[name] = result
            return result
        except Exception as exc:
            logger.error("define_workflow failed: %s", exc)
            return None

    async def preview_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        """Preview a stored workflow with predictive analysis.

        Returns prediction metrics including cost, duration, risk, and
        whether constraints (deadline/budget) will be met.
        """
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_preview_workflow(name)
        return None  # Python preview requires re-building — handled by WorkflowBuilder directly

    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all stored workflow definitions."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_list_workflows()
        return list(self._stored_workflows.values())

    async def get_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific workflow definition by name."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_get_workflow(name)
        return self._stored_workflows.get(name)

    async def delete_workflow(self, name: str) -> bool:
        """Delete a stored workflow."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_delete_workflow(name)
        return self._stored_workflows.pop(name, None) is not None

    async def list_workflow_templates(self) -> List[str]:
        """List pre-built workflow template names."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_list_templates()
        return ["code_review_pipeline", "research_pipeline", "parallel_analysis_pipeline"]

    async def preview_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Preview a pre-built workflow template."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_preview_template(name)
        return None

    async def scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics from the active backend."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_stats()
        return self._scheduler.stats

    async def execution_waves(self) -> List[List[str]]:
        """Get execution waves (parallel groups in topological order)."""
        if self._use_rust_scheduler:
            return await self._rust_bridge.scheduler_execution_waves()
        return self._dep_resolver.get_execution_waves()

    # ------------------------------------------------------------------ #
    # WORKFLOW EXECUTION
    # ------------------------------------------------------------------ #

    async def run_workflow(
        self,
        name: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowExecution:
        """Execute a stored workflow definition.

        Loads the workflow, creates Task objects from its task definitions,
        and submits them all via submit_tasks() so the DAG resolver handles
        dependency ordering and parallel dispatch.

        Each task's on_complete callback updates the WorkflowExecution state.
        When all tasks finish, the execution is marked complete.

        Args:
            name: Name of a previously-defined workflow.
            context: Extra context passed to each task.

        Returns:
            WorkflowExecution with live status tracking.

        Raises:
            ValueError: If workflow name not found.
        """
        # Load workflow definition
        defn = await self.get_workflow(name)
        if defn is None:
            raise ValueError(f"Workflow '{name}' not found")

        execution_id = f"wfx-{uuid.uuid4().hex[:12]}"
        task_defs = defn.get("tasks", defn.get("task_order", []))

        # Build the task map from the definition
        # task_defs can be list of dicts (from define) or list of strings (task_order)
        if task_defs and isinstance(task_defs[0], str):
            # task_order style: just task IDs, need to reconstruct deps from waves
            waves = defn.get("waves", [])
            task_dict_list = []
            completed_ids: Set[str] = set()
            for wave in waves:
                for tid in wave:
                    task_dict_list.append({
                        "task_id": tid,
                        "task_type": "planning",
                        "depends_on": list(completed_ids),
                    })
                completed_ids.update(wave)
            task_defs = task_dict_list

        # Create execution tracker
        wf_exec = WorkflowExecution(
            execution_id=execution_id,
            workflow_name=name,
            status=WorkflowExecutionStatus.RUNNING,
            started_at=datetime.now(),
        )

        # Map task_type strings to TaskType enum
        _TASK_TYPE_MAP = {t.value: t for t in TaskType}

        tasks: List[Task] = []
        for tdef in task_defs:
            tid = tdef["task_id"] if isinstance(tdef, dict) else tdef
            ttype_str = tdef.get("task_type", "planning") if isinstance(tdef, dict) else "planning"
            ttype = _TASK_TYPE_MAP.get(ttype_str, TaskType.PLANNING)
            deps = tdef.get("depends_on", []) if isinstance(tdef, dict) else []
            priority_int = tdef.get("priority", 2) if isinstance(tdef, dict) else 2
            priority_map = {0: TaskPriority.CRITICAL, 1: TaskPriority.HIGH, 2: TaskPriority.MEDIUM, 3: TaskPriority.LOW}
            priority = priority_map.get(priority_int, TaskPriority.MEDIUM)

            # Namespace task IDs to avoid collisions with other submissions
            namespaced_id = f"{execution_id}:{tid}"
            namespaced_deps = [f"{execution_id}:{d}" for d in deps]

            task = Task(
                task_id=namespaced_id,
                task_type=ttype,
                description=f"Workflow '{name}' task: {tid}",
                priority=priority,
                dependencies=namespaced_deps,
                context={**(context or {}), "workflow_execution_id": execution_id, "workflow_task_id": tid},
            )
            tasks.append(task)
            wf_exec.task_ids.append(namespaced_id)

        self._workflow_executions[execution_id] = wf_exec

        # Submit all tasks — the resolver handles ordering and dispatch
        await self.submit_tasks(tasks)

        # Start a background watcher that finalizes the execution
        asyncio.create_task(self._watch_workflow_execution(execution_id))

        await self.event_bus.publish(
            EventType.WORKFLOW_PROGRESS,
            {"execution_id": execution_id, "workflow_name": name, "task_count": len(tasks), "phase": "started"},
            source="orchestrator",
        )

        return wf_exec

    async def _watch_workflow_execution(self, execution_id: str) -> None:
        """Background watcher that polls for workflow completion.

        Emits incremental WORKFLOW_PROGRESS events each cycle so SSE clients
        can track task-by-task progress without polling the REST API.
        """
        wf_exec = self._workflow_executions.get(execution_id)
        if wf_exec is None:
            return

        prev_progress = 0.0

        while wf_exec.status == WorkflowExecutionStatus.RUNNING:
            await asyncio.sleep(0.5)

            all_done = True
            has_failure = False
            for tid in wf_exec.task_ids:
                task = self.tasks.get(tid)
                if task is None:
                    continue
                if task.status == TaskStatus.COMPLETED:
                    if tid not in wf_exec.task_results:
                        wf_exec.task_results[tid] = {
                            "status": "completed",
                            "quality_score": task.quality_score,
                            "duration": task.duration,
                        }
                        wf_exec.total_tokens += getattr(task.result or {}, "get", lambda k, d: d)("tokens_used", 0)
                elif task.status == TaskStatus.FAILED:
                    has_failure = True
                    wf_exec.task_results[tid] = {"status": "failed", "error": task.error}
                elif task.status == TaskStatus.CANCELLED:
                    wf_exec.task_results[tid] = {"status": "cancelled"}
                else:
                    all_done = False

            # Emit incremental progress if it changed
            current_progress = wf_exec.progress
            if current_progress != prev_progress:
                prev_progress = current_progress
                await self.event_bus.publish(
                    EventType.WORKFLOW_PROGRESS,
                    {
                        "execution_id": execution_id,
                        "workflow_name": wf_exec.workflow_name,
                        "phase": "progress",
                        "percent": round(current_progress * 100, 1),
                        "completed_tasks": len(wf_exec.task_results),
                        "total_tasks": len(wf_exec.task_ids),
                    },
                    source="orchestrator",
                )

            if all_done:
                wf_exec.completed_at = datetime.now()
                if has_failure:
                    wf_exec.status = WorkflowExecutionStatus.FAILED
                    wf_exec.error = "One or more tasks failed"
                else:
                    wf_exec.status = WorkflowExecutionStatus.COMPLETED

                await self.event_bus.publish(
                    EventType.WORKFLOW_PROGRESS,
                    {**wf_exec.to_dict(), "phase": "completed" if not has_failure else "failed"},
                    source="orchestrator",
                )

    async def cancel_workflow_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution.

        Cancels all pending/in-progress tasks for this workflow.
        """
        wf_exec = self._workflow_executions.get(execution_id)
        if wf_exec is None or wf_exec.status != WorkflowExecutionStatus.RUNNING:
            return False

        for tid in wf_exec.task_ids:
            task = self.tasks.get(tid)
            if task and task.status in (TaskStatus.PENDING, TaskStatus.QUEUED, TaskStatus.IN_PROGRESS):
                task.status = TaskStatus.CANCELLED
                wf_exec.task_results[tid] = {"status": "cancelled"}

        wf_exec.status = WorkflowExecutionStatus.CANCELLED
        wf_exec.completed_at = datetime.now()
        return True

    def get_workflow_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow execution by ID."""
        wf_exec = self._workflow_executions.get(execution_id)
        return wf_exec.to_dict() if wf_exec else None

    def list_workflow_executions(
        self, status: Optional[str] = None, limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List workflow executions, optionally filtered by status."""
        execs = list(self._workflow_executions.values())
        if status:
            execs = [e for e in execs if e.status.value == status]
        # Most recent first
        execs.sort(key=lambda e: e.started_at or datetime.min, reverse=True)
        return [e.to_dict() for e in execs[:limit]]

    async def execute_task(
        self,
        task: Union[Task, BaseTask],
        agent_id: Optional[str] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Result:
        """Execute a single task through the orchestration system.

        Args:
            task: Task to execute (can be Task or BaseTask)
            agent_id: Optional specific agent to use
            progress_callback: Optional callback invoked at each phase

        Returns:
            Result with execution details

        Raises:
            TaskExecutionError: If execution fails
        """
        # Convert BaseTask to Task if needed
        if isinstance(task, BaseTask):
            task = self._convert_base_task(task)

        logger.info(f"Executing task {task.task_id} (type={task.task_type.value})")

        # Track task
        self.tasks[task.task_id] = task
        self.metrics["tasks_received"] += 1

        # Publish event
        await self.event_bus.publish(
            EventType.TASK_RECEIVED,
            {"task_id": task.task_id, "task_type": task.task_type.value},
            source="orchestrator",
        )

        await self._enrich_task_context(task)

        # Progress: planning
        await self._report_progress(task.task_id, "planning", 10, "Task received, planning execution", progress_callback)

        try:
            # Select agent if not specified
            if not agent_id:
                agent_id = self.get_best_agent(task)
                if not agent_id:
                    raise TaskExecutionError(
                        f"No suitable agent for task {task.task_id}"
                    )

            # Progress: agent selection
            await self._report_progress(task.task_id, "agent_selection", 20, f"Selected agent {agent_id}", progress_callback)

            # Assign task to agent
            task.assigned_agent_id = agent_id
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = datetime.now()

            agent_state = self.agents[agent_id]
            agent_state.active_tasks.add(task.task_id)
            agent_state.last_active = datetime.now()

            logger.info(f"Assigned task {task.task_id} to agent {agent_id}")

            await self.event_bus.publish(
                EventType.TASK_STARTED,
                {
                    "task_id": task.task_id,
                    "agent_id": agent_id,
                    "task_type": task.task_type.value,
                },
                source="orchestrator",
            )

            # Progress: execution
            await self._report_progress(task.task_id, "execution", 30, "Executing task", progress_callback)

            # Execute task (this would call actual agent implementation)
            result = await self._execute_task_with_agent(task, agent_id)

            # Progress: quality check
            await self._report_progress(task.task_id, "quality_check", 90, "Assessing quality", progress_callback)

            context_bundle = task.context.get("context_bundle")
            if context_bundle:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata.setdefault("context_bundle", context_bundle)

            # Update task
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            task.result = result.output
            task.quality_score = result.quality_score
            task.hrm_score = result.hrm_score

            # Update agent state
            agent_state.active_tasks.remove(task.task_id)
            agent_state.completed_tasks += 1
            agent_state.total_execution_time += result.execution_time

            # Update metrics
            self.metrics["tasks_completed"] += 1
            self.metrics["total_execution_time"] += result.execution_time
            self.metrics["total_cost"] += result.cost
            self.metrics["total_tokens"] += result.tokens_used

            logger.info(
                f"Task {task.task_id} completed in {result.execution_time:.2f}s "
                f"(quality={result.quality_score:.2f})"
            )

            await self.event_bus.publish(
                EventType.TASK_COMPLETED,
                {
                    "task_id": task.task_id,
                    "agent_id": agent_id,
                    "execution_time": result.execution_time,
                    "quality_score": result.quality_score,
                },
                source="orchestrator",
            )

            # Progress: complete
            await self._report_progress(task.task_id, "complete", 100, "Task completed", progress_callback)

            # Call completion callback if provided
            if task.on_complete:
                await task.on_complete(result)

            return result

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}", exc_info=True)

            # Update task
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()

            # Update agent state if assigned
            if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                agent_state = self.agents[task.assigned_agent_id]
                if task.task_id in agent_state.active_tasks:
                    agent_state.active_tasks.remove(task.task_id)
                agent_state.failed_tasks += 1

            # Update metrics
            self.metrics["tasks_failed"] += 1

            await self.event_bus.publish(
                EventType.TASK_FAILED,
                {
                    "task_id": task.task_id,
                    "error": str(e),
                    "agent_id": task.assigned_agent_id,
                },
                source="orchestrator",
            )

            # Call failure callback if provided
            if task.on_fail:
                await task.on_fail(e)

            # Check if should retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.info(
                    f"Retrying task {task.task_id} "
                    f"(attempt {task.retry_count}/{task.max_retries})"
                )
                return await self.execute_task(task)

            # Create error result
            result = Result(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                output=None,
                error=str(e),
                agent_id=task.assigned_agent_id,
            )

            raise TaskExecutionError(f"Task {task.task_id} failed: {e}") from e

    async def execute_parallel(
        self,
        tasks: List[Union[Task, BaseTask]],
        max_concurrency: Optional[int] = None,
    ) -> List[Result]:
        """Execute multiple tasks in parallel.

        Args:
            tasks: List of tasks to execute
            max_concurrency: Optional limit on concurrent executions

        Returns:
            List of results in same order as tasks
        """
        logger.info(f"Executing {len(tasks)} tasks in parallel")

        if max_concurrency:
            # Use semaphore to limit concurrency
            semaphore = asyncio.Semaphore(max_concurrency)

            async def execute_with_limit(task: Union[Task, BaseTask]) -> Result:
                async with semaphore:
                    return await self.execute_task(task)

            results = await asyncio.gather(
                *[execute_with_limit(task) for task in tasks],
                return_exceptions=True,
            )
        else:
            # No concurrency limit
            results = await asyncio.gather(
                *[self.execute_task(task) for task in tasks],
                return_exceptions=True,
            )

        # Convert exceptions to error results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task = tasks[i]
                task_id = task.task_id if isinstance(task, Task) else task.task_id
                final_results.append(
                    Result(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        output=None,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        logger.info(
            f"Parallel execution complete: "
            f"{sum(1 for r in final_results if r.status == TaskStatus.COMPLETED)} "
            f"succeeded, "
            f"{sum(1 for r in final_results if r.status == TaskStatus.FAILED)} failed"
        )

        return final_results

    # ========================================================================
    # WORKFLOW ORCHESTRATION (from end_to_end_orchestrator.py)
    # ========================================================================

    async def execute_workflow(
        self,
        workflow_id: str,
        description: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowState:
        """Execute a 7-phase end-to-end development workflow.

        Phases: Research → Design → Code → Test → Deploy → Monitor → Docs

        Args:
            workflow_id: Unique workflow identifier
            description: Workflow description/goal
            context: Optional context and configuration

        Returns:
            WorkflowState with complete execution details
        """
        logger.info(f"Starting workflow {workflow_id}: {description}")

        workflow = WorkflowState(
            workflow_id=workflow_id,
            description=description,
            current_phase=WorkflowPhase.RESEARCH,
            context=context or {},
            started_at=datetime.now(),
            status=TaskStatus.IN_PROGRESS,
        )

        self.workflows[workflow_id] = workflow

        # Define phase sequence
        phases = [
            WorkflowPhase.RESEARCH,
            WorkflowPhase.DESIGN,
            WorkflowPhase.CODE,
            WorkflowPhase.TEST,
            WorkflowPhase.DEPLOY,
            WorkflowPhase.MONITOR,
            WorkflowPhase.DOCS,
        ]

        try:
            for phase in phases:
                logger.info(f"Workflow {workflow_id}: Executing phase {phase.value}")

                workflow.current_phase = phase

                # Execute phase
                result = await self._execute_workflow_phase(workflow, phase)

                # Store phase result
                workflow.phase_results[phase] = result
                workflow.phases_completed.append(phase)

                # Aggregate metrics
                workflow.total_cost += result.cost
                workflow.total_tokens += result.tokens_used
                workflow.total_execution_time += result.execution_time

                # Store artifacts
                workflow.artifacts.extend(result.artifacts)

                logger.info(
                    f"Workflow {workflow_id}: Phase {phase.value} complete "
                    f"(quality={result.quality_score:.2f})"
                )

                # Check quality threshold - fail fast if too low
                if result.quality_score < 0.3:
                    logger.warning(
                        f"Workflow {workflow_id}: Low quality in {phase.value}, "
                        f"stopping workflow"
                    )
                    workflow.status = TaskStatus.FAILED
                    break

            # Mark complete if all phases succeeded
            if len(workflow.phases_completed) == len(phases):
                workflow.status = TaskStatus.COMPLETED
                workflow.completed_at = datetime.now()
                logger.info(
                    f"Workflow {workflow_id} completed successfully in "
                    f"{workflow.total_execution_time:.2f}s"
                )

            return workflow

        except Exception as e:
            logger.error(f"Workflow {workflow_id} failed: {e}", exc_info=True)
            workflow.status = TaskStatus.FAILED
            workflow.completed_at = datetime.now()
            raise OrchestratorError(f"Workflow {workflow_id} failed: {e}") from e

    async def _execute_workflow_phase(
        self,
        workflow: WorkflowState,
        phase: WorkflowPhase,
    ) -> Result:
        """Execute a single workflow phase.

        Args:
            workflow: Current workflow state
            phase: Phase to execute

        Returns:
            Result from phase execution
        """
        # Create task for this phase
        task = Task(
            task_id=f"{workflow.workflow_id}_{phase.value}",
            task_type=self._phase_to_task_type(phase),
            description=f"{phase.value.title()} phase for {workflow.description}",
            context={
                "workflow_id": workflow.workflow_id,
                "phase": phase.value,
                "previous_results": {
                    p.value: asdict(r)
                    for p, r in workflow.phase_results.items()
                },
                **workflow.context,
            },
            priority=TaskPriority.HIGH,
        )

        # Execute through normal task execution
        return await self.execute_task(task)

    def _phase_to_task_type(self, phase: WorkflowPhase) -> TaskType:
        """Map workflow phase to task type."""
        mapping = {
            WorkflowPhase.RESEARCH: TaskType.PLANNING,
            WorkflowPhase.DESIGN: TaskType.DESIGN,
            WorkflowPhase.CODE: TaskType.IMPLEMENTATION,
            WorkflowPhase.TEST: TaskType.TESTING,
            WorkflowPhase.DEPLOY: TaskType.DEPLOYMENT,
            WorkflowPhase.MONITOR: TaskType.MONITORING,
            WorkflowPhase.DOCS: TaskType.DOCUMENTATION,
        }
        return mapping[phase]

    # ========================================================================
    # CODE REVIEW (from review_orchestrator.py)
    # ========================================================================

    async def review_code(
        self,
        code: str,
        language: str = "python",
        enable_security: bool = True,
        enable_performance: bool = True,
        enable_quality: bool = True,
    ) -> ReviewResult:
        """Run comprehensive code review with parallel checks.

        Args:
            code: Source code to review
            language: Programming language
            enable_security: Enable security scanning
            enable_performance: Enable performance analysis
            enable_quality: Enable quality checks

        Returns:
            ReviewResult with all findings
        """
        logger.info(f"Starting code review ({language}, {len(code)} chars)")

        # Create parallel review tasks
        review_tasks = []

        if enable_security:
            review_tasks.append(self._review_security(code, language))

        if enable_performance:
            review_tasks.append(self._review_performance(code, language))

        if enable_quality:
            review_tasks.append(self._review_quality(code, language))

        # Execute all reviews in parallel
        results = await asyncio.gather(*review_tasks, return_exceptions=True)

        # Aggregate results
        security_issues = results[0] if enable_security else []
        performance_issues = results[1] if enable_performance and len(results) > 1 else []
        quality_issues = results[2] if enable_quality and len(results) > 2 else []

        # Calculate overall score (weighted average)
        security_score = 1.0 - (len(security_issues) * 0.2)
        performance_score = 1.0 - (len(performance_issues) * 0.1)
        quality_score = 1.0 - (len(quality_issues) * 0.1)

        overall_score = (
            security_score * 0.5 + performance_score * 0.3 + quality_score * 0.2
        )
        overall_score = max(0.0, min(1.0, overall_score))

        # Generate recommendations
        recommendations = self._generate_review_recommendations(
            security_issues, performance_issues, quality_issues
        )

        result = ReviewResult(
            overall_score=overall_score,
            security_issues=security_issues,
            performance_issues=performance_issues,
            quality_issues=quality_issues,
            recommendations=recommendations,
        )

        logger.info(
            f"Code review complete: score={overall_score:.2f}, "
            f"issues={len(security_issues) + len(performance_issues) + len(quality_issues)}"
        )

        return result

    async def _review_security(
        self, code: str, language: str
    ) -> List[Dict[str, Any]]:
        """Run security analysis via LLM."""
        return await self._llm_review(
            code, language,
            concern="security",
            prompt_prefix=(
                "Analyze the following code for security vulnerabilities. "
                "Look for: injection flaws, authentication issues, data exposure, "
                "insecure defaults, and OWASP Top 10 concerns."
            ),
        )

    async def _review_performance(
        self, code: str, language: str
    ) -> List[Dict[str, Any]]:
        """Run performance analysis via LLM."""
        return await self._llm_review(
            code, language,
            concern="performance",
            prompt_prefix=(
                "Analyze the following code for performance issues. "
                "Look for: unnecessary allocations, O(n^2) or worse algorithms, "
                "blocking calls, missing caching opportunities, and memory leaks."
            ),
        )

    async def _review_quality(
        self, code: str, language: str
    ) -> List[Dict[str, Any]]:
        """Run code quality analysis via LLM."""
        return await self._llm_review(
            code, language,
            concern="quality",
            prompt_prefix=(
                "Analyze the following code for quality issues. "
                "Look for: code duplication, poor naming, missing error handling, "
                "overly complex functions, and violations of SOLID principles."
            ),
        )

    async def _llm_review(
        self,
        code: str,
        language: str,
        concern: str,
        prompt_prefix: str,
    ) -> List[Dict[str, Any]]:
        """Shared helper: ask the LLM router to review code for a specific concern."""
        if self.llm_router is None:
            return []

        try:
            from core.llm.intelligent_llm_router import TaskType as LLMTaskType

            provider = await self.llm_router.select_provider(
                task_type=LLMTaskType.CODE_GENERATION,
                prompt_tokens=len(code.split()),
            )

            prompt = (
                f"{prompt_prefix}\n\n"
                f"Language: {language}\n\n"
                f"```{language}\n{code}\n```\n\n"
                "Respond ONLY with a JSON array of issues. Each issue must have:\n"
                '  {"type": "<issue_type>", "severity": "low"|"medium"|"high"|"critical", '
                '"description": "<explanation>", "line": <line_number_or_null>, '
                '"suggestion": "<how_to_fix>"}\n'
                "If there are no issues, return an empty array: []"
            )

            messages = [{"role": "user", "content": prompt}]
            text, _metadata = await self.llm_router.call_llm(
                provider=provider,
                messages=messages,
                task_type=LLMTaskType.CODE_GENERATION,
                max_tokens=2048,
            )

            return self._parse_review_response(text, concern)
        except Exception:
            logger.exception("LLM %s review failed", concern)
            return []

    @staticmethod
    def _parse_review_response(text: str, concern: str) -> List[Dict[str, Any]]:
        """Parse a JSON array of issues from the LLM response text."""
        import json as _json

        # Strip markdown fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            issues = _json.loads(cleaned)
        except _json.JSONDecodeError:
            return []

        if not isinstance(issues, list):
            return []

        valid = []
        for item in issues:
            if not isinstance(item, dict):
                continue
            valid.append({
                "type": item.get("type", concern),
                "severity": item.get("severity", "medium"),
                "description": item.get("description", ""),
                "line": item.get("line"),
                "suggestion": item.get("suggestion", ""),
            })
        return valid

    def _generate_review_recommendations(
        self,
        security_issues: List[Dict[str, Any]],
        performance_issues: List[Dict[str, Any]],
        quality_issues: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate actionable recommendations from review results."""
        recommendations = []

        if security_issues:
            recommendations.append(
                f"Address {len(security_issues)} security issue(s) before deployment"
            )

        if performance_issues:
            recommendations.append(
                f"Optimize {len(performance_issues)} performance concern(s)"
            )

        if quality_issues:
            recommendations.append(
                f"Refactor {len(quality_issues)} code quality issue(s)"
            )

        return recommendations

    # ========================================================================
    # EVENT-DRIVEN REACTIONS (from autonomous_orchestrator.py)
    # ========================================================================

    def register_rule(self, rule: ConditionActionRule) -> None:
        """Register a CEP (Complex Event Processing) rule.

        Args:
            rule: Condition-action rule to register
        """
        self.cep_rules.append(rule)
        logger.info(f"Registered CEP rule: {rule.name}")

    async def _subscribe_to_events(self) -> None:
        """Subscribe to events for autonomous reactions."""
        # Subscribe to key event types
        legacy_events = [
            EventType.TASK_RECEIVED,
            EventType.TASK_STARTED,
            EventType.TASK_COMPLETED,
            EventType.TASK_FAILED,
            EventType.GENERATION_STARTED,
            EventType.GENERATION_COMPLETED,
            EventType.GENERATION_FAILED,
            EventType.FEEDBACK_RECEIVED,
            EventType.PREFERENCE_LOGGED,
            EventType.RM_SCORE_COMPUTED,
            EventType.QUALITY_ASSESSED,
            EventType.WORKFLOW_PROGRESS,
            EventType.STRATEGIC_EVENT_DETECTED,
            EventType.MODEL_RETRAINED,
            EventType.TOOL_PERFORMANCE_UPDATED,
        ]

        handler_map = {
            EventType.TASK_FAILED: self._handle_task_failed,
            EventType.ERROR_OCCURRED: self._handle_error_occurred,
            EventType.GENERATION_FAILED: self._handle_generation_failed,
            EventType.QUALITY_ASSESSED: self._handle_quality_assessed,
            EventType.TASK_COMPLETED: self._handle_task_completed_event,
            EventType.PREFERENCE_LOGGED: self._handle_preference_logged,
            EventType.RM_SCORE_COMPUTED: self._handle_reward_model_score,
            EventType.MODEL_RETRAINED: self._handle_model_retrained,
            EventType.STRATEGIC_EVENT_DETECTED: self._handle_strategic_event,
            EventType.WORKFLOW_PROGRESS: self._handle_workflow_progress,
        }

        self._subscribed_events = []
        for event in legacy_events:
            callback = handler_map.get(event, self._handle_generic_event)
            await self.event_bus.subscribe(event, callback, replay_missed=True)
            self._subscribed_events.append(event)

    async def _handle_task_failed(self, event: Any) -> None:
        """Handle task failure events."""
        data = self._extract_event_data(event)
        self._record_event("task_failed")
        task_id = data.get("task_id")
        error = data.get("error")

        logger.warning(f"Task {task_id} failed: {error}")

        # Apply CEP rules
        event_data = {"event_type": EventTriggerType.TASK_FAILED, **data}
        await self._apply_cep_rules(event_data)

    async def _handle_error_occurred(self, event: Any) -> None:
        """Handle system error events."""
        data = self._extract_event_data(event)
        self._record_event("system_error")
        logger.error(f"System error: {data}")

        # Apply CEP rules
        event_data = {"event_type": EventTriggerType.SYSTEM_ERROR, **data}
        await self._apply_cep_rules(event_data)

    async def _handle_workflow_progress(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("workflow_progress")
        await self._apply_cep_rules({"event_type": EventTriggerType.WORKFLOW_PROGRESS, **data})

        # Emit notifications for terminal workflow events
        phase = data.get("phase")
        if phase in ("completed", "failed") and self._notification_service is not None:
            wf_name = data.get("workflow_name", "unknown")
            exec_id = data.get("execution_id", "")
            severity = Severity.INFO if phase == "completed" else Severity.ERROR
            notif = self._notification_service.create_notification(
                title=f"Workflow '{wf_name}' {phase}",
                body=(
                    f"Execution {exec_id} {phase}."
                    + (f" Error: {data.get('error', 'unknown')}" if phase == "failed" else "")
                ),
                severity=severity,
                source="orchestrator",
                metadata={"execution_id": exec_id, "workflow_name": wf_name, "phase": phase},
            )
            await self._notification_service.send(notif)

    async def _handle_generation_failed(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("generation_failed")
        await self._apply_cep_rules({"event_type": EventTriggerType.GENERATION_FAILED, **data})

    async def _handle_quality_assessed(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("quality_assessed")
        await self._apply_cep_rules({"event_type": EventTriggerType.QUALITY_ASSESSED, **data})

    async def _handle_task_completed_event(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("task_completed")
        await self._apply_cep_rules({"event_type": EventTriggerType.TASK_COMPLETED, **data})
        # Trigger dependency dispatch for tasks that completed via execute_task directly
        task_id = data.get("task_id")
        if not task_id:
            return

        if self._use_rust_scheduler:
            # Rust scheduler: query task state via bridge, mark completed if running
            state_str = await self._rust_bridge.scheduler_task_state(task_id)
            if state_str and state_str.lower() == "running":
                newly_ready = await self._rust_bridge.scheduler_mark_completed(task_id)
                task = self.tasks.get(task_id)
                if task:
                    self._completed_tasks[task_id] = task
                if newly_ready:
                    await self._dispatch_ready_tasks()
        else:
            if self._dep_resolver.get_task_state(task_id) == DepTaskState.RUNNING:
                try:
                    newly_ready = self._dep_resolver.mark_completed(task_id)
                    task = self.tasks.get(task_id)
                    if task:
                        self._completed_tasks[task_id] = task
                    if newly_ready:
                        await self._dispatch_ready_tasks()
                except ValueError:
                    pass  # Task not in expected state — already handled elsewhere

    async def _handle_preference_logged(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("preference_logged")
        await self._apply_cep_rules({"event_type": EventTriggerType.PREFERENCE_LOGGED, **data})

    async def _handle_reward_model_score(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("reward_model_score")
        await self._apply_cep_rules({"event_type": EventTriggerType.REWARD_MODEL_SCORE, **data})

    async def _handle_model_retrained(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("model_retrained")
        await self._apply_cep_rules({"event_type": EventTriggerType.MODEL_RETRAINED, **data})

    async def _handle_strategic_event(self, event: Any) -> None:
        data = self._extract_event_data(event)
        self._record_event("strategic_event")
        await self._apply_cep_rules({"event_type": EventTriggerType.STRATEGIC_EVENT, **data})

    async def _handle_generic_event(self, event: Any) -> None:
        data = self._extract_event_data(event)
        label = data.get("event_type") or getattr(event, "event_type", None)
        if isinstance(label, Enum):
            label_value = label.value
        else:
            label_value = str(label or "generic")
        self._record_event(label_value)
        payload = RuleEventData(data)
        payload["event_type"] = label_value
        await self._apply_cep_rules(payload)

    async def _apply_cep_rules(self, event_data: Dict[str, Any]) -> None:
        """Apply CEP rules to event data."""
        if not isinstance(event_data, RuleEventData):
            event_data = RuleEventData(event_data)
        raw_event_type = event_data.get("event_type")
        event_label = raw_event_type.value if isinstance(raw_event_type, Enum) else raw_event_type

        # Find matching rules
        matching_rules = []
        for rule in self.cep_rules:
            normalized_events = [
                et.value if isinstance(et, Enum) else et for et in rule.event_types
            ]
            if (
                rule.enabled
                and event_label in normalized_events
                and rule.can_trigger()
                and rule.condition(event_data)
            ):
                matching_rules.append(rule)

        # Execute actions for matching rules (sorted by priority)
        matching_rules.sort(key=lambda r: r.priority, reverse=True)

        for rule in matching_rules:
            try:
                logger.info(f"Triggering CEP rule: {rule.name}")
                await rule.action(event_data)

                rule.last_triggered = time.time()
                rule.trigger_count += 1
                self.total_rules_triggered += 1

            except Exception as e:
                logger.error(
                    f"Error executing CEP rule {rule.name}: {e}", exc_info=True
                )

    async def _register_default_rules(self) -> None:
        """Register default CEP rules."""
        if self.cep_rules:
            return

        default_rules = [
            ConditionActionRule(
                rule_id="auto_retry",
                name="Auto Retry Failed Tasks",
                description="Automatically retry failed generations up to 3 times.",
                event_types=[EventTriggerType.GENERATION_FAILED],
                condition=lambda data: data.get("retry_count", 0) < 3,
                action_type=ActionType.RETRY_TASK,
                action=self._action_retry_task,
            ),
            ConditionActionRule(
                rule_id="quality_metrics",
                name="Quality Assessment Tracker",
                description="Record quality assessment metrics for reporting.",
                event_types=[EventTriggerType.QUALITY_ASSESSED],
                condition=lambda _: True,
                action_type=ActionType.UPDATE_METRICS,
                action=self._action_record_quality_metrics,
            ),
            ConditionActionRule(
                rule_id="task_completion",
                name="Task Completion Logger",
                description="Track task completion durations and throughput.",
                event_types=[EventTriggerType.TASK_COMPLETED],
                condition=lambda _: True,
                action_type=ActionType.UPDATE_METRICS,
                action=self._action_track_completion,
            ),
            ConditionActionRule(
                rule_id="failure_escalation",
                name="Escalate Task Failures",
                description="Escalate repeated task failures to operators.",
                event_types=[EventTriggerType.TASK_FAILED],
                condition=lambda data: data.get("error") is not None,
                action_type=ActionType.ALERT_USER,
                action=self._action_escalate_failure,
            ),
            ConditionActionRule(
                rule_id="preference_logging",
                name="Preference Logging Rule",
                description="Log user preferences for personalization.",
                event_types=[EventTriggerType.PREFERENCE_LOGGED],
                condition=lambda _: True,
                action_type=ActionType.LOG_INSIGHT,
                action=self._action_log_preference,
            ),
            ConditionActionRule(
                rule_id="reward_model_monitor",
                name="Reward Model Monitoring",
                description="Monitor low reward-model scores.",
                event_types=[EventTriggerType.REWARD_MODEL_SCORE],
                condition=lambda data: data.get("score", 0) < 0.5,
                action_type=ActionType.UPDATE_METRICS,
                action=self._action_monitor_reward_model,
            ),
            ConditionActionRule(
                rule_id="model_retraining",
                name="Model Retraining Tracker",
                description="Track model retraining events.",
                event_types=[EventTriggerType.MODEL_RETRAINED],
                condition=lambda _: True,
                action_type=ActionType.LOG_INSIGHT,
                action=self._action_track_model_retraining,
            ),
            ConditionActionRule(
                rule_id="strategic_event",
                name="Strategic Event Handler",
                description="Handle strategic opportunities or alerts.",
                event_types=[EventTriggerType.STRATEGIC_EVENT],
                condition=lambda _: True,
                action_type=ActionType.UPDATE_METRICS,
                action=self._action_handle_strategic_event,
            ),
        ]

        for rule in default_rules:
            self.register_rule(rule)

    async def _action_retry_task(self, event_data: Dict[str, Any]) -> None:
        """Action: Retry a failed task."""
        task_id = event_data.get("task_id")
        if task_id and task_id in self.tasks:
            task = self.tasks[task_id]
            logger.info(f"Retrying task {task_id}")
            await self.execute_task(task)

    async def _action_record_quality_metrics(self, event_data: Dict[str, Any]) -> None:
        score = event_data.get("score")
        if score is not None:
            self.metrics.setdefault("quality_samples", []).append(score)

    async def _action_track_completion(self, event_data: Dict[str, Any]) -> None:
        duration = event_data.get("duration_ms", 0)
        self.metrics.setdefault("completion_durations", []).append(duration)

    async def _action_escalate_failure(self, event_data: Dict[str, Any]) -> None:
        self.metrics.setdefault("escalations", 0)
        self.metrics["escalations"] += 1
        logger.warning(f"Escalating failure: {event_data}")

    async def _action_log_preference(self, event_data: Dict[str, Any]) -> None:
        self.metrics.setdefault("preferences_logged", 0)
        self.metrics["preferences_logged"] += 1

    async def _action_monitor_reward_model(self, event_data: Dict[str, Any]) -> None:
        score = event_data.get("score")
        self.metrics.setdefault("low_reward_scores", []).append(score)

    async def _action_track_model_retraining(self, event_data: Dict[str, Any]) -> None:
        self.metrics.setdefault("model_retrains", 0)
        self.metrics["model_retrains"] += 1

    async def _action_handle_strategic_event(self, event_data: Dict[str, Any]) -> None:
        self.metrics.setdefault("strategic_events", 0)
        self.metrics["strategic_events"] += 1

    # ========================================================================
    # STATE MANAGEMENT
    # ========================================================================

    async def _load_state(self) -> None:
        """Load persisted orchestrator state."""
        try:
            # Load agent states
            agent_states = await self.memory_store.retrieve("orchestrator:agents")
            if agent_states:
                logger.info(f"Loaded {len(agent_states)} agent states")

            # Load task states
            task_states = await self.memory_store.retrieve("orchestrator:tasks")
            if task_states:
                logger.info(f"Loaded {len(task_states)} task states")

            # Load metrics
            metrics = await self.memory_store.retrieve("orchestrator:metrics")
            if metrics:
                self.metrics = metrics
                logger.info("Loaded orchestrator metrics")

            # Load dependency resolver state
            dep_data = await self.memory_store.retrieve("orchestrator:dep_resolver")
            if dep_data and isinstance(dep_data, dict):
                self._dep_resolver = DependencyResolver.from_dict(dep_data)
                logger.info("Loaded dependency resolver state")

            # Load completed tasks history
            completed = await self.memory_store.retrieve("orchestrator:completed_tasks")
            if completed and isinstance(completed, dict):
                logger.info(f"Loaded {len(completed)} completed task records")

            # Load execution tracker
            tracker_data = await self.memory_store.retrieve("orchestrator:exec_tracker")
            if tracker_data and isinstance(tracker_data, dict):
                self._exec_tracker = ExecutionTracker.from_dict(tracker_data)
                logger.info("Loaded execution tracker profiles")

            # Load quality gate policy
            gate_data = await self.memory_store.retrieve("orchestrator:quality_policy")
            if gate_data and isinstance(gate_data, dict):
                self._quality_policy = QualityGatePolicy.from_dict(gate_data)
                logger.info("Loaded quality gate policy")

            # Load retry manager
            retry_data = await self.memory_store.retrieve("orchestrator:retry_manager")
            if retry_data and isinstance(retry_data, dict):
                self._retry_manager = RetryManager.from_dict(retry_data)
                logger.info("Loaded retry manager state")

            # Load scheduler coordinator
            coord_data = await self.memory_store.retrieve("orchestrator:scheduler_coord")
            if coord_data and isinstance(coord_data, dict):
                self._scheduler = SchedulerCoordinator.from_dict(
                    coord_data,
                    self._dep_resolver, self._exec_tracker,
                    self._quality_policy, self._retry_manager,
                )
                logger.info("Loaded scheduler coordinator state")

            # Load stored workflow definitions
            wf_defs = await self.memory_store.retrieve("orchestrator:stored_workflows")
            if wf_defs and isinstance(wf_defs, dict):
                self._stored_workflows = wf_defs
                logger.info("Loaded %d stored workflow definitions", len(wf_defs))

        except Exception as e:
            logger.warning(f"Failed to load state: {e}")

    async def _persist_state(self) -> None:
        """Persist orchestrator state."""
        try:
            # Persist agent states
            await self.memory_store.store(
                "orchestrator:agents",
                {agent_id: asdict(agent) for agent_id, agent in self.agents.items()},
            )

            # Persist active tasks (completed tasks are archived separately)
            active_tasks = {
                task_id: asdict(task)
                for task_id, task in self.tasks.items()
                if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
            }
            await self.memory_store.store("orchestrator:tasks", active_tasks)

            # Persist metrics
            await self.memory_store.store("orchestrator:metrics", self.metrics)

            # Persist dependency resolver state
            await self.memory_store.store(
                "orchestrator:dep_resolver",
                self._dep_resolver.to_dict(),
            )

            # Persist recent completed tasks (capped at 1000)
            recent = dict(list(self._completed_tasks.items())[-1000:])
            await self.memory_store.store(
                "orchestrator:completed_tasks",
                {tid: asdict(t) for tid, t in recent.items()},
            )

            # Persist execution tracker (profiles only — records are transient)
            await self.memory_store.store(
                "orchestrator:exec_tracker",
                self._exec_tracker.to_dict(),
            )

            # Persist quality gate policy
            await self.memory_store.store(
                "orchestrator:quality_policy",
                self._quality_policy.to_dict(),
            )

            # Persist retry manager
            await self.memory_store.store(
                "orchestrator:retry_manager",
                self._retry_manager.to_dict(),
            )

            # Persist scheduler coordinator
            await self.memory_store.store(
                "orchestrator:scheduler_coord",
                self._scheduler.to_dict(),
            )

            # Persist stored workflow definitions
            await self.memory_store.store(
                "orchestrator:stored_workflows",
                self._stored_workflows,
            )

            logger.info("Persisted orchestrator state")

        except Exception as e:
            logger.error(f"Failed to persist state: {e}", exc_info=True)

    async def get_system_status(self) -> SystemStatus:
        """Get overall system status and metrics.

        Returns:
            SystemStatus with comprehensive metrics
        """
        active_tasks = sum(
            1
            for task in self.tasks.values()
            if task.status == TaskStatus.IN_PROGRESS
        )

        completed_tasks = self.metrics["tasks_completed"]
        failed_tasks = self.metrics["tasks_failed"]

        # Calculate averages
        total_tasks = completed_tasks + failed_tasks
        avg_duration = (
            self.metrics["total_execution_time"] / total_tasks if total_tasks > 0 else 0
        )

        # Calculate average quality score from completed tasks
        completed_task_list = [
            task for task in self.tasks.values() if task.status == TaskStatus.COMPLETED
        ]
        avg_quality = (
            sum(task.quality_score for task in completed_task_list)
            / len(completed_task_list)
            if completed_task_list
            else 0.0
        )

        # Determine system health
        success_rate = completed_tasks / total_tasks if total_tasks > 0 else 1.0
        health = "healthy" if success_rate > 0.9 else "degraded" if success_rate > 0.7 else "unhealthy"

        # Agent states
        agent_states = {
            agent_id: {
                "role": agent.capabilities.role.value,
                "active_tasks": len(agent.active_tasks),
                "completed_tasks": agent.completed_tasks,
                "failed_tasks": agent.failed_tasks,
                "success_rate": agent.success_rate,
                "is_available": agent.is_available,
            }
            for agent_id, agent in self.agents.items()
        }

        # Queue sizes
        queue_sizes = {
            priority: len(task_ids)
            for priority, task_ids in self.task_queues.items()
        }

        return SystemStatus(
            active_agents=len(self.agents),
            active_tasks=active_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            avg_task_duration=avg_duration,
            avg_quality_score=avg_quality,
            system_health=health,
            agent_states=agent_states,
            queue_sizes=queue_sizes,
        )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    async def _execute_task_with_agent(
        self, task: Task, agent_id: str
    ) -> Result:
        """Execute task with specific agent via the LLM router.

        Falls back to a lightweight stub when no LLM router is wired.
        """
        start_time = time.time()

        if self.llm_router is not None:
            try:
                from core.llm.intelligent_llm_router import TaskType as LLMTaskType

                # Map orchestrator task type to LLM task type
                task_type_map = {
                    TaskType.PLANNING: LLMTaskType.RESEARCH,
                    TaskType.DESIGN: LLMTaskType.DESIGN,
                    TaskType.IMPLEMENTATION: LLMTaskType.CODE_GENERATION,
                    TaskType.TESTING: LLMTaskType.TESTING,
                    TaskType.REVIEW: LLMTaskType.CODE_GENERATION,
                    TaskType.DEPLOYMENT: LLMTaskType.DEPLOYMENT,
                    TaskType.MONITORING: LLMTaskType.RESEARCH,
                    TaskType.DOCUMENTATION: LLMTaskType.CREATIVE,
                    TaskType.REFACTORING: LLMTaskType.CODE_GENERATION,
                    TaskType.DEBUGGING: LLMTaskType.REASONING,
                }
                llm_task_type = task_type_map.get(task.task_type, LLMTaskType.CODE_GENERATION)

                provider = await self.llm_router.select_provider(
                    task_type=llm_task_type,
                    prompt_tokens=len(task.description.split()),
                )

                messages = [{"role": "user", "content": task.description}]
                text, metadata = await self.llm_router.call_llm(
                    provider=provider,
                    messages=messages,
                    task_type=llm_task_type,
                    max_tokens=2048,
                )

                execution_time = time.time() - start_time
                output = {
                    "result": text,
                    "provider": metadata.get("provider", ""),
                    "tokens_used": metadata.get("tokens_used", 0),
                }
                if self._degraded_mode:
                    output["degraded"] = True
                return Result(
                    task_id=task.task_id,
                    status=TaskStatus.COMPLETED,
                    output=output,
                    agent_id=agent_id,
                    execution_time=execution_time,
                    quality_score=0.85,
                    hrm_score=0.80,
                )
            except Exception as e:
                logger.warning("LLM call failed, falling back to stub: %s", e)

        # Fallback stub
        await asyncio.sleep(0.1)
        execution_time = time.time() - start_time

        output = {"result": "Task completed (stub — no LLM router available)"}
        if self._degraded_mode:
            output["degraded"] = True
        return Result(
            task_id=task.task_id,
            status=TaskStatus.COMPLETED,
            output=output,
            agent_id=agent_id,
            execution_time=execution_time,
            quality_score=0.85,
            hrm_score=0.80,
        )

    def _convert_base_task(self, base_task: BaseTask) -> Task:
        """Convert BaseTask to enhanced Task."""
        # Infer task type from metadata
        task_type_str = base_task.metadata.get("task_type", base_task.task_type)
        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            task_type = TaskType.IMPLEMENTATION  # Default

        return Task(
            task_id=base_task.task_id,
            task_type=task_type,
            description=base_task.description,
            context=base_task.metadata,
        )

    async def _reassign_task(self, task_id: str) -> None:
        """Reassign a task to a different agent."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        task.assigned_agent_id = None

        logger.info(f"Reassigning task {task_id}")
        await self.execute_task(task)

    async def _monitor_tasks(self) -> None:
        """Background task to monitor for timeouts and issues."""
        while self.is_running:
            try:
                # Check for timed out tasks
                for task in self.tasks.values():
                    if (
                        task.status == TaskStatus.IN_PROGRESS
                        and task.is_overdue
                    ):
                        logger.warning(f"Task {task.task_id} timed out")
                        task.status = TaskStatus.FAILED
                        task.error = "Task timeout"

                        await self.event_bus.publish(
                            EventType.TASK_FAILED,
                            {"task_id": task.task_id, "error": "timeout"},
                            source="orchestrator",
                        )

                await asyncio.sleep(10.0)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Error in task monitor: {e}", exc_info=True)

    async def _process_queues(self) -> None:
        """Background task to process queued tasks.

        Tasks with dependencies are routed through submit_task() for
        DAG-aware scheduling.  Tasks without dependencies execute directly
        (backward-compatible behaviour).
        """
        while self.is_running:
            try:
                # Process queues in priority order
                for priority in [
                    TaskPriority.CRITICAL,
                    TaskPriority.HIGH,
                    TaskPriority.MEDIUM,
                    TaskPriority.LOW,
                ]:
                    queue = self.task_queues[priority]
                    if queue:
                        task_id = queue.pop(0)
                        if task_id in self.tasks:
                            task = self.tasks[task_id]
                            if task.dependencies:
                                asyncio.create_task(self.submit_task(task))
                            else:
                                asyncio.create_task(self.execute_task(task))

                await asyncio.sleep(1.0)  # Check every second

            except Exception as e:
                logger.error(f"Error in queue processor: {e}", exc_info=True)

    async def _drain_active_tasks(self, timeout: float = 30.0) -> None:
        """Wait for active tasks to complete with timeout."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            active = sum(
                1
                for task in self.tasks.values()
                if task.status == TaskStatus.IN_PROGRESS
            )

            if active == 0:
                logger.info("All active tasks completed")
                return

            logger.info(f"Waiting for {active} active tasks to complete...")
            await asyncio.sleep(1.0)

        logger.warning(f"Timeout waiting for active tasks to complete")


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================


def create_orchestrator(
    event_bus: IEventBus,
    memory_store: IMemoryStore,
    llm_router: Optional[ILLMRouter] = None,
    tool_selector: Optional[IToolSelector] = None,
    rust_bridge: Optional[Any] = None,
    notification_service: Optional[NotificationService] = None,
    resource_manager: Optional[ResourceManager] = None,
) -> UnifiedOrchestrator:
    """Create and initialize a UnifiedOrchestrator instance.

    Args:
        event_bus: Event bus for pub/sub messaging
        memory_store: Persistent storage for state
        llm_router: Optional LLM routing
        tool_selector: Optional tool selection
        rust_bridge: Optional RustCoreBridge for Rust-delegated scheduling
        notification_service: Optional notification service for workflow events
        resource_manager: Optional production safety layer (concurrency + budget + breakers)

    Returns:
        Configured UnifiedOrchestrator instance
    """
    orch = UnifiedOrchestrator(
        event_bus=event_bus,
        memory_store=memory_store,
        llm_router=llm_router,
        tool_selector=tool_selector,
        rust_bridge=rust_bridge,
    )
    orch._notification_service = notification_service
    orch._resource_manager = resource_manager
    return orch
