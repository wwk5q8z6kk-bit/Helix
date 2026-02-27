"""Declarative workflow builder with predictive analysis for Helix.

Provides a fluent API for defining task DAGs with integrated quality gates,
retry strategies, model preferences, deadlines, and budget constraints.

The killer feature: **predictive preview** — before executing a workflow, the
builder uses execution history to predict whether it will meet its deadline,
stay within budget, and achieve quality targets.  This "what-if" analysis is
unique to Helix and has no equivalent in Airflow, Prefect, Dagster, or Temporal.

Usage::

    wf = (WorkflowBuilder("code_review", coordinator)
          .task("plan",   task_type="planning", model="flash")
          .task("code",   task_type="coding",   depends_on=["plan"])
          .task("test",   task_type="testing",  depends_on=["code"])
          .task("review", task_type="review",   depends_on=["test"])
          .with_deadline(time.time() + 3600)
          .with_budget(5.0)
          .gate("review", GATE_STRICT))

    preview = wf.preview()        # Predictive analysis
    definition = wf.build()       # Immutable workflow definition
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.scheduling.dependency_resolver import (
    CycleDetectedError,
    DependencyResolver,
)
from core.scheduling.quality_gates import (
    GateAction,
    QualityGate,
    QualityGatePolicy,
    GATE_LENIENT,
    GATE_STANDARD,
    GATE_STRICT,
)
from core.scheduling.retry_strategies import (
    RetryStrategy,
    strategy_budget_conscious,
    strategy_fast_to_premium,
    strategy_quality_first,
    strategy_no_retry,
)
from core.scheduling.scheduler_coordinator import SchedulerCoordinator

logger = logging.getLogger(__name__)


# ── Value objects ────────────────────────────────────────────────────


@dataclass
class WorkflowTask:
    """A task definition within a workflow."""

    task_id: str
    task_type: str
    depends_on: List[str] = field(default_factory=list)
    model: Optional[str] = None  # None = auto-assign via Pareto
    priority: int = 2  # 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW
    gate: Optional[QualityGate] = None
    retry_strategy: Optional[RetryStrategy] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowStatus(str, Enum):
    """Lifecycle status of a workflow definition."""

    DRAFT = "draft"
    VALIDATED = "validated"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowDefinition:
    """Immutable, validated workflow definition ready for execution."""

    name: str
    tasks: Dict[str, WorkflowTask]
    task_order: List[str]  # topological order
    waves: List[List[str]]  # parallel execution waves
    deadline: Optional[float] = None
    budget: Optional[float] = None
    default_gate: Optional[QualityGate] = None
    default_strategy: Optional[RetryStrategy] = None
    model_assignments: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_count(self) -> int:
        return len(self.tasks)

    @property
    def wave_count(self) -> int:
        return len(self.waves)

    @property
    def max_parallelism(self) -> int:
        return max((len(w) for w in self.waves), default=0)


@dataclass
class TaskPrediction:
    """Predicted execution metrics for a single task."""

    task_id: str
    model: str
    predicted_duration: float
    predicted_cost: float
    predicted_quality: float
    predicted_success_rate: float
    meets_quality_gate: bool
    quality_margin: float  # How far above/below the gate threshold


@dataclass
class WorkflowPreview:
    """Predictive analysis of a workflow before execution.

    Answers the questions:
    - Will it finish on time?
    - Will it stay within budget?
    - Will quality targets be met?
    - Where are the risks?
    """

    name: str
    task_predictions: List[TaskPrediction]
    waves: List[List[str]]
    wave_durations: List[float]
    critical_path: List[str]

    # Aggregate predictions
    total_predicted_duration: float
    total_predicted_cost: float
    avg_predicted_quality: float

    # Constraint analysis
    deadline: Optional[float]
    budget: Optional[float]
    meets_deadline: Optional[bool]  # None if no deadline set
    meets_budget: Optional[bool]  # None if no budget set
    deadline_margin: Optional[float]  # Seconds of slack (negative = late)
    budget_margin: Optional[float]  # Dollars of slack (negative = over)

    # Risk analysis
    at_risk_tasks: List[str]  # Tasks predicted to fail quality gates
    bottleneck_tasks: List[str]  # Critical path chokepoints
    low_confidence_tasks: List[str]  # Tasks with <70% predicted success

    @property
    def feasible(self) -> bool:
        """Whether the workflow can execute within time and budget constraints.

        Quality gate risk is reported separately via ``at_risk_tasks`` —
        a workflow is still *feasible* even if some tasks may not hit their
        quality targets (they can be retried or escalated at runtime).
        """
        if self.meets_deadline is False:
            return False
        if self.meets_budget is False:
            return False
        return True

    @property
    def risk_level(self) -> str:
        """Overall risk: low, medium, high."""
        risks = 0
        if self.meets_deadline is False:
            risks += 2
        if self.meets_budget is False:
            risks += 1
        risks += len(self.at_risk_tasks)
        risks += len(self.low_confidence_tasks)
        if risks == 0:
            return "low"
        elif risks <= 2:
            return "medium"
        return "high"


# ── Workflow Builder ─────────────────────────────────────────────────


class WorkflowBuilder:
    """Fluent API for building task DAGs with scheduling intelligence.

    Chain calls to define tasks, dependencies, quality gates, and
    constraints, then call ``.preview()`` for predictive analysis or
    ``.build()`` for an immutable workflow definition.
    """

    def __init__(
        self,
        name: str,
        coordinator: Optional[SchedulerCoordinator] = None,
    ) -> None:
        self._name = name
        self._coordinator = coordinator
        self._tasks: Dict[str, WorkflowTask] = {}
        self._insertion_order: List[str] = []
        self._deadline: Optional[float] = None
        self._budget: Optional[float] = None
        self._default_gate: Optional[QualityGate] = None
        self._default_strategy: Optional[RetryStrategy] = None
        self._metadata: Dict[str, Any] = {}
        # Edge-level gates: (upstream, downstream) → gate
        self._edge_gates: Dict[Tuple[str, str], QualityGate] = {}

    # ── Fluent task definition ────────────────────────────────────────

    def task(
        self,
        task_id: str,
        task_type: str = "general",
        depends_on: Optional[List[str]] = None,
        model: Optional[str] = None,
        priority: int = 2,
        gate: Optional[QualityGate] = None,
        retry_strategy: Optional[RetryStrategy] = None,
        **metadata: Any,
    ) -> "WorkflowBuilder":
        """Add a task to the workflow.

        Args:
            task_id: Unique identifier for this task.
            task_type: Task category (coding, testing, review, etc.).
            depends_on: List of task_ids that must complete first.
            model: Specific model to use (None = auto-assign).
            priority: 0=CRITICAL through 3=LOW.
            gate: Quality gate for this task's output.
            retry_strategy: Retry strategy override.
            **metadata: Additional key-value metadata.
        """
        if task_id in self._tasks:
            raise ValueError(f"Duplicate task_id: {task_id}")

        self._tasks[task_id] = WorkflowTask(
            task_id=task_id,
            task_type=task_type,
            depends_on=list(depends_on or []),
            model=model,
            priority=priority,
            gate=gate,
            retry_strategy=retry_strategy,
            metadata=dict(metadata),
        )
        self._insertion_order.append(task_id)
        return self

    def edge_gate(
        self,
        upstream: str,
        downstream: str,
        gate: QualityGate,
    ) -> "WorkflowBuilder":
        """Set a quality gate on a specific dependency edge."""
        self._edge_gates[(upstream, downstream)] = gate
        return self

    # ── Fluent constraint definition ──────────────────────────────────

    def with_deadline(self, deadline: float) -> "WorkflowBuilder":
        """Set an absolute deadline (unix timestamp)."""
        self._deadline = deadline
        return self

    def with_deadline_in(self, seconds: float) -> "WorkflowBuilder":
        """Set a deadline relative to now."""
        self._deadline = time.time() + seconds
        return self

    def with_budget(self, budget: float) -> "WorkflowBuilder":
        """Set a total budget cap (dollars)."""
        self._budget = budget
        return self

    def with_default_gate(self, gate: QualityGate) -> "WorkflowBuilder":
        """Set the default quality gate for all tasks."""
        self._default_gate = gate
        return self

    def with_default_strategy(self, strategy: RetryStrategy) -> "WorkflowBuilder":
        """Set the default retry strategy for all tasks."""
        self._default_strategy = strategy
        return self

    def with_metadata(self, **kwargs: Any) -> "WorkflowBuilder":
        """Attach metadata to the workflow."""
        self._metadata.update(kwargs)
        return self

    # ── Internal DAG analysis ──────────────────────────────────────────

    def _compute_topo_order(self) -> Tuple[List[str], List[List[str]]]:
        """Kahn's algorithm over the builder's task definitions.

        Returns (topological_order, waves) where each wave is a list of
        tasks that can execute in parallel.  Unlike DependencyResolver
        (designed for incremental runtime scheduling), this operates on
        the full graph at once so insertion order doesn't matter.
        """
        in_degree: Dict[str, int] = {tid: 0 for tid in self._tasks}
        dependents: Dict[str, List[str]] = {tid: [] for tid in self._tasks}

        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep in self._tasks:
                    in_degree[task.task_id] += 1
                    dependents[dep].append(task.task_id)

        waves: List[List[str]] = []
        order: List[str] = []
        current = sorted(
            [tid for tid, deg in in_degree.items() if deg == 0],
            key=lambda t: (self._tasks[t].priority, self._insertion_order.index(t)),
        )

        while current:
            waves.append(list(current))
            order.extend(current)
            next_wave: List[str] = []
            for tid in current:
                for dep_tid in dependents[tid]:
                    in_degree[dep_tid] -= 1
                    if in_degree[dep_tid] == 0:
                        next_wave.append(dep_tid)
            next_wave.sort(
                key=lambda t: (self._tasks[t].priority, self._insertion_order.index(t)),
            )
            current = next_wave

        return order, waves

    # ── Validation ────────────────────────────────────────────────────

    def validate(self) -> List[str]:
        """Validate the workflow, returning a list of errors (empty = valid).

        Checks:
        1. At least one task defined.
        2. All dependency references exist.
        3. No cycles in the dependency graph.
        4. Budget is positive (if set).
        5. Deadline is in the future (if set).
        """
        errors: List[str] = []

        if not self._tasks:
            errors.append("Workflow has no tasks")
            return errors

        # Check dependency references
        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep not in self._tasks:
                    errors.append(
                        f"Task '{task.task_id}' depends on unknown task '{dep}'"
                    )

        # Check for cycles using a temporary resolver
        if not errors:
            try:
                resolver = DependencyResolver()
                for tid in self._insertion_order:
                    task = self._tasks[tid]
                    resolver.add_task(tid, dependencies=task.depends_on, priority=task.priority)
            except CycleDetectedError as e:
                errors.append(f"Cycle detected: {' → '.join(e.cycle)}")

        if self._budget is not None and self._budget <= 0:
            errors.append(f"Budget must be positive, got {self._budget}")

        if self._deadline is not None and self._deadline <= time.time():
            errors.append("Deadline is in the past")

        return errors

    # ── Predictive Preview ────────────────────────────────────────────

    def preview(self) -> WorkflowPreview:
        """Generate predictive analysis without executing.

        Uses the SchedulerCoordinator's execution history to predict
        duration, cost, quality, and constraint feasibility.
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Cannot preview invalid workflow: {'; '.join(errors)}")

        # Compute waves via standalone Kahn's (not DependencyResolver)
        topo_order, waves = self._compute_topo_order()
        task_types = {tid: t.task_type for tid, t in self._tasks.items()}

        # Determine model assignments
        model_assignments: Dict[str, str] = {}
        for tid, task in self._tasks.items():
            if task.model:
                model_assignments[tid] = task.model
            elif self._coordinator:
                gate = task.gate or self._default_gate or GATE_STANDARD
                rec = self._coordinator.recommend_model(
                    task.task_type,
                    quality_requirement=gate.min_quality,
                    budget_remaining=self._budget,
                )
                model_assignments[tid] = rec.model if rec else "default"
            else:
                model_assignments[tid] = "default"

        # Generate per-task predictions
        predictions: List[TaskPrediction] = []
        at_risk: List[str] = []
        low_confidence: List[str] = []

        tracker = self._coordinator._tracker if self._coordinator else None

        for tid, task in self._tasks.items():
            model = model_assignments[tid]
            gate = task.gate or self._default_gate or GATE_STANDARD

            if tracker:
                duration = tracker.predict_duration(task.task_type, model)
                cost = tracker.predict_cost(task.task_type, model)
                quality = tracker.predict_quality(task.task_type, model)
                success = tracker.predict_success_rate(task.task_type, model)
            else:
                duration, cost, quality, success = 10.0, 0.01, 0.5, 0.9

            meets_gate = quality >= gate.min_quality
            margin = quality - gate.min_quality

            predictions.append(TaskPrediction(
                task_id=tid,
                model=model,
                predicted_duration=duration,
                predicted_cost=cost,
                predicted_quality=quality,
                predicted_success_rate=success,
                meets_quality_gate=meets_gate,
                quality_margin=margin,
            ))

            if not meets_gate:
                at_risk.append(tid)
            if success < 0.7:
                low_confidence.append(tid)

        # Wave durations
        task_durations = {p.task_id: p.predicted_duration for p in predictions}
        wave_durations = [
            max((task_durations.get(tid, 10.0) for tid in wave), default=0.0)
            for wave in waves
        ]
        total_duration = sum(wave_durations)
        total_cost = sum(p.predicted_cost for p in predictions)
        avg_quality = (
            sum(p.predicted_quality for p in predictions) / len(predictions)
            if predictions else 0.0
        )

        # Critical path (use coordinator if available)
        critical_path: List[str] = []
        bottlenecks: List[str] = []
        if self._coordinator:
            # Build a temporary resolver for critical-path analysis
            cp_resolver = DependencyResolver()
            for tid in topo_order:
                task = self._tasks[tid]
                cp_resolver.add_task(tid, dependencies=task.depends_on, priority=task.priority)

            cp = self._coordinator._tracker.compute_critical_path(
                cp_resolver, task_types, model_assignments,
            )
            critical_path = cp.path

            # Bottlenecks: single-task waves on critical path
            cp_set = set(critical_path)
            for wave in waves:
                if len(wave) == 1 and wave[0] in cp_set:
                    bottlenecks.append(wave[0])
        else:
            # Approximate: flatten topological order
            for wave in waves:
                critical_path.extend(wave)

        # Constraint analysis
        meets_deadline = None
        deadline_margin = None
        if self._deadline is not None:
            remaining = self._deadline - time.time()
            deadline_margin = remaining - total_duration
            meets_deadline = deadline_margin >= 0

        meets_budget = None
        budget_margin = None
        if self._budget is not None:
            budget_margin = self._budget - total_cost
            meets_budget = budget_margin >= 0

        return WorkflowPreview(
            name=self._name,
            task_predictions=predictions,
            waves=waves,
            wave_durations=wave_durations,
            critical_path=critical_path,
            total_predicted_duration=total_duration,
            total_predicted_cost=total_cost,
            avg_predicted_quality=avg_quality,
            deadline=self._deadline,
            budget=self._budget,
            meets_deadline=meets_deadline,
            meets_budget=meets_budget,
            deadline_margin=deadline_margin,
            budget_margin=budget_margin,
            at_risk_tasks=at_risk,
            bottleneck_tasks=bottlenecks,
            low_confidence_tasks=low_confidence,
        )

    # ── Build ─────────────────────────────────────────────────────────

    def build(self) -> WorkflowDefinition:
        """Validate and produce an immutable workflow definition.

        Raises ValueError if validation fails.
        """
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid workflow: {'; '.join(errors)}")

        # Compute topological order and waves (standalone Kahn's)
        topo_order, waves = self._compute_topo_order()

        # Model assignments
        model_assignments: Dict[str, str] = {}
        for tid, task in self._tasks.items():
            if task.model:
                model_assignments[tid] = task.model
            elif self._coordinator:
                gate = task.gate or self._default_gate or GATE_STANDARD
                rec = self._coordinator.recommend_model(
                    task.task_type, quality_requirement=gate.min_quality,
                    budget_remaining=self._budget,
                )
                model_assignments[tid] = rec.model if rec else "default"
            else:
                model_assignments[tid] = "default"

        return WorkflowDefinition(
            name=self._name,
            tasks=copy.deepcopy(self._tasks),
            task_order=topo_order,
            waves=waves,
            deadline=self._deadline,
            budget=self._budget,
            default_gate=self._default_gate,
            default_strategy=self._default_strategy,
            model_assignments=model_assignments,
            metadata=dict(self._metadata),
        )


# ── Pre-built workflow templates ─────────────────────────────────────


def code_review_pipeline(
    coordinator: Optional[SchedulerCoordinator] = None,
) -> WorkflowBuilder:
    """Standard code review workflow: plan → implement → test → review.

    - Planning uses a fast model (low cost).
    - Implementation uses standard quality with strict gates.
    - Testing gets lenient gates (tests either pass or fail).
    - Review uses strict gates with premium model preference.
    """
    return (
        WorkflowBuilder("code_review", coordinator)
        .task("plan", task_type="planning", priority=1)
        .task("implement", task_type="coding", depends_on=["plan"], priority=0,
              gate=GATE_STRICT, retry_strategy=strategy_fast_to_premium())
        .task("test", task_type="testing", depends_on=["implement"],
              gate=GATE_LENIENT, retry_strategy=strategy_no_retry())
        .task("review", task_type="review", depends_on=["test"],
              gate=GATE_STRICT, retry_strategy=strategy_quality_first())
        .with_default_gate(GATE_STANDARD)
    )


def research_pipeline(
    coordinator: Optional[SchedulerCoordinator] = None,
) -> WorkflowBuilder:
    """Research workflow: gather → analyze → synthesize → report.

    All tasks are sequential (each depends on the previous).
    Uses budget-conscious strategy since research tasks can be long.
    """
    return (
        WorkflowBuilder("research", coordinator)
        .task("gather", task_type="research", priority=1)
        .task("analyze", task_type="reasoning", depends_on=["gather"], priority=1)
        .task("synthesize", task_type="creative", depends_on=["analyze"])
        .task("report", task_type="writing", depends_on=["synthesize"])
        .with_default_gate(GATE_STANDARD)
        .with_default_strategy(strategy_budget_conscious())
    )


def parallel_analysis_pipeline(
    coordinator: Optional[SchedulerCoordinator] = None,
) -> WorkflowBuilder:
    """Fan-out/fan-in pattern: one input → N parallel analyses → merge.

    Demonstrates diamond-shaped DAG for maximum parallelism.
    """
    return (
        WorkflowBuilder("parallel_analysis", coordinator)
        .task("prepare", task_type="planning", priority=0)
        .task("analyze_security", task_type="review", depends_on=["prepare"])
        .task("analyze_performance", task_type="review", depends_on=["prepare"])
        .task("analyze_quality", task_type="review", depends_on=["prepare"])
        .task("merge_results", task_type="reasoning",
              depends_on=["analyze_security", "analyze_performance", "analyze_quality"],
              gate=GATE_STRICT)
        .with_default_gate(GATE_STANDARD)
    )
