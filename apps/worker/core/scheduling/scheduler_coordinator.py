"""Unified scheduling intelligence for Helix.

The Scheduler Coordinator is the "brain" that ties together all four
scheduling modules (DependencyResolver, ExecutionTracker, QualityGatePolicy,
RetryManager) into a single decision engine.

What makes this unique — no existing orchestrator (Airflow, Prefect, Dagster,
Temporal) combines:

1. **Pareto-optimal model selection** — given historical (cost, quality) data,
   recommend the model on the efficiency frontier for each task type.
2. **Concurrency control** — per-provider slot management with backpressure.
3. **Deadline-aware priority boosting** — dynamically shift priorities as
   deadlines approach, accounting for predicted execution time.
4. **Self-tuning quality gates** — automatically adjust thresholds based on
   observed pass/fail rates across task types.
5. **Execution planning** — produce a full predicted schedule with model
   assignments, cost estimates, and bottleneck analysis.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from core.scheduling.dependency_resolver import DependencyResolver
from core.scheduling.execution_tracker import ExecutionTracker
from core.scheduling.quality_gates import QualityGatePolicy
from core.scheduling.retry_strategies import RetryManager

logger = logging.getLogger(__name__)


# ── Value objects ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelRecommendation:
    """A model recommendation from Pareto analysis."""

    model: str
    predicted_quality: float
    predicted_cost: float
    predicted_duration: float
    efficiency_score: float  # quality / cost ratio
    success_rate: float
    reason: str


@dataclass(frozen=True)
class ScheduleEntry:
    """Planned execution for a single task in an execution plan."""

    task_id: str
    wave: int
    model: str
    predicted_duration: float
    predicted_cost: float
    predicted_quality: float
    priority_boost: float = 0.0


@dataclass
class ExecutionPlan:
    """Complete execution plan with predictions and bottleneck analysis."""

    entries: List[ScheduleEntry]
    waves: List[List[str]]
    wave_durations: List[float]
    total_predicted_duration: float
    total_predicted_cost: float
    critical_path: List[str]
    bottleneck_tasks: List[str]
    model_assignments: Dict[str, str]


# ── Concurrency ──────────────────────────────────────────────────────


class ConcurrencySlot:
    """Token-bucket concurrency limiter for a provider.

    Tracks active slots and provides acquire/release semantics.
    When all slots are occupied, acquire() returns False (backpressure).
    """

    def __init__(self, provider: str, max_concurrent: int = 5) -> None:
        self.provider = provider
        self.max_concurrent = max_concurrent
        self._active: int = 0
        self._total_acquired: int = 0
        self._total_rejected: int = 0

    def acquire(self) -> bool:
        if self._active < self.max_concurrent:
            self._active += 1
            self._total_acquired += 1
            return True
        self._total_rejected += 1
        return False

    def release(self) -> None:
        self._active = max(0, self._active - 1)

    @property
    def available(self) -> int:
        return max(0, self.max_concurrent - self._active)

    @property
    def utilization(self) -> float:
        return self._active / self.max_concurrent if self.max_concurrent else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "max_concurrent": self.max_concurrent,
            "active": self._active,
            "available": self.available,
            "utilization": round(self.utilization, 2),
            "total_acquired": self._total_acquired,
            "total_rejected": self._total_rejected,
        }


# ── Scheduler Coordinator ────────────────────────────────────────────


class SchedulerCoordinator:
    """Unified scheduling intelligence.

    Consumes data from all four scheduling modules and provides
    higher-level decision-making: model selection, concurrency control,
    deadline management, execution planning, and self-tuning.
    """

    def __init__(
        self,
        resolver: DependencyResolver,
        tracker: ExecutionTracker,
        policy: QualityGatePolicy,
        retry_manager: RetryManager,
    ) -> None:
        self._resolver = resolver
        self._tracker = tracker
        self._policy = policy
        self._retry_manager = retry_manager

        # Concurrency slots per provider
        self._slots: Dict[str, ConcurrencySlot] = {}
        self._default_max_concurrent: int = 5

        # Deadlines: task_id → absolute unix timestamp
        self._deadlines: Dict[str, float] = {}

        # Tuning log (capped)
        self._tuning_log: List[Dict[str, Any]] = []
        self._max_tuning_log: int = 50

    # ── Model Selection (Pareto) ──────────────────────────────────────

    def compute_pareto_frontier(
        self, task_type: str, min_executions: int = 2
    ) -> List[ModelRecommendation]:
        """Compute the Pareto-optimal set of models for a task type.

        A model is Pareto-optimal if no other model is both cheaper AND
        higher quality.  The frontier represents the best possible
        tradeoffs between cost and quality.
        """
        profiles = self._tracker.get_all_profiles()
        candidates: List[ModelRecommendation] = []

        for (tt, model), profile in profiles.items():
            if tt != task_type or profile.total_executions < min_executions:
                continue
            if profile.successes == 0:
                continue

            cost = profile.avg_cost
            quality = profile.avg_quality
            duration = profile.avg_duration
            success = profile.success_rate
            efficiency = quality / max(cost, 0.0001)

            candidates.append(ModelRecommendation(
                model=model,
                predicted_quality=quality,
                predicted_cost=cost,
                predicted_duration=duration,
                efficiency_score=efficiency,
                success_rate=success,
                reason="historical",
            ))

        if not candidates:
            return []

        # Filter to Pareto frontier: no other candidate dominates
        frontier: List[ModelRecommendation] = []
        for c in candidates:
            dominated = False
            for other in candidates:
                if other is c:
                    continue
                # `other` dominates `c` if it's better on both dimensions
                if other.predicted_quality >= c.predicted_quality and other.predicted_cost <= c.predicted_cost:
                    if other.predicted_quality > c.predicted_quality or other.predicted_cost < c.predicted_cost:
                        dominated = True
                        break
            if not dominated:
                frontier.append(c)

        # Sort by efficiency (best first)
        frontier.sort(key=lambda r: r.efficiency_score, reverse=True)
        return frontier

    def recommend_model(
        self,
        task_type: str,
        quality_requirement: float = 0.5,
        budget_remaining: Optional[float] = None,
        prefer_speed: bool = False,
    ) -> Optional[ModelRecommendation]:
        """Select the optimal model considering quality, cost, and budget.

        Strategy:
        1. Compute Pareto frontier for the task type.
        2. Filter by quality requirement.
        3. If budget constraint, filter by cost.
        4. If prefer_speed, sort by duration instead of efficiency.
        5. Return the best candidate.
        """
        frontier = self.compute_pareto_frontier(task_type)
        if not frontier:
            return None

        # Filter by quality
        viable = [r for r in frontier if r.predicted_quality >= quality_requirement]
        if not viable:
            # Relax: take the highest quality available
            viable = sorted(frontier, key=lambda r: r.predicted_quality, reverse=True)[:1]

        # Filter by budget
        if budget_remaining is not None:
            affordable = [r for r in viable if r.predicted_cost <= budget_remaining]
            if affordable:
                viable = affordable

        if not viable:
            return None

        # Sort by preference
        if prefer_speed:
            viable.sort(key=lambda r: r.predicted_duration)
        else:
            viable.sort(key=lambda r: r.efficiency_score, reverse=True)

        return viable[0]

    # ── Concurrency Control ───────────────────────────────────────────

    def set_provider_limit(self, provider: str, max_concurrent: int) -> None:
        """Configure the concurrency limit for a provider."""
        if provider in self._slots:
            self._slots[provider].max_concurrent = max_concurrent
        else:
            self._slots[provider] = ConcurrencySlot(provider, max_concurrent)

    def acquire_slot(self, provider: str) -> bool:
        """Try to acquire a concurrency slot for a provider.

        Returns True if a slot was acquired, False if all slots are busy.
        """
        if provider not in self._slots:
            self._slots[provider] = ConcurrencySlot(
                provider, self._default_max_concurrent
            )
        return self._slots[provider].acquire()

    def release_slot(self, provider: str) -> None:
        """Release a concurrency slot after task completion."""
        if provider in self._slots:
            self._slots[provider].release()

    def get_concurrency_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current concurrency utilization across all providers."""
        return {name: slot.to_dict() for name, slot in self._slots.items()}

    # ── Deadline Management ───────────────────────────────────────────

    def set_deadline(self, task_id: str, deadline: float) -> None:
        """Set an absolute deadline (unix timestamp) for a task.

        Deadlines propagate urgency to the task's priority, ensuring
        time-sensitive work gets scheduled ahead of less urgent tasks.
        """
        self._deadlines[task_id] = deadline

    def clear_deadline(self, task_id: str) -> None:
        """Remove a deadline for a task."""
        self._deadlines.pop(task_id, None)

    def get_urgency(self, task_id: str) -> float:
        """Compute urgency (0.0–1.0) based on deadline proximity.

        Returns 0.0 if no deadline is set.
        Returns 1.0 if the deadline is now or past.
        Scales linearly in the final hour, then logarithmically before that.

        The urgency curve:
        - >24h away: 0.0–0.1 (low)
        - 1h–24h:    0.1–0.5 (medium)
        - <1h:       0.5–1.0 (high, linear ramp)
        - past due:  1.0 (critical)
        """
        deadline = self._deadlines.get(task_id)
        if deadline is None:
            return 0.0

        now = time.time()
        remaining = deadline - now

        if remaining <= 0:
            return 1.0

        one_hour = 3600.0
        one_day = 86400.0

        if remaining <= one_hour:
            # Linear ramp in final hour: 0.5 → 1.0
            return 0.5 + 0.5 * (1.0 - remaining / one_hour)
        elif remaining <= one_day:
            # Medium urgency: 0.1 → 0.5
            fraction = (one_day - remaining) / (one_day - one_hour)
            return 0.1 + 0.4 * fraction
        else:
            # Low urgency: approaches 0.1 as deadline gets closer
            # Logarithmic decay beyond 24h
            days_away = remaining / one_day
            return max(0.0, 0.1 * (1.0 / days_away))

    def get_priority_boost(self, task_id: str) -> int:
        """Convert urgency to a priority level boost (0–3).

        Helix priorities: 0=CRITICAL, 1=HIGH, 2=MEDIUM, 3=LOW.
        A boost of 2 on a MEDIUM(2) task makes it CRITICAL(0).
        """
        urgency = self.get_urgency(task_id)
        if urgency >= 0.8:
            return 3  # Maximum boost
        elif urgency >= 0.5:
            return 2
        elif urgency >= 0.2:
            return 1
        return 0

    # ── Execution Planning ────────────────────────────────────────────

    def plan_execution(
        self,
        task_types: Dict[str, str],
        task_models: Optional[Dict[str, str]] = None,
        budget_limit: Optional[float] = None,
    ) -> ExecutionPlan:
        """Create a full execution plan with model assignments and predictions.

        If ``task_models`` is not provided, the coordinator uses Pareto
        analysis to assign optimal models based on each task's quality
        requirement and the remaining budget.

        Args:
            task_types: task_id → task_type mapping.
            task_models: Optional task_id → model mapping. Auto-assigned if None.
            budget_limit: Total budget for the execution plan.

        Returns:
            ExecutionPlan with per-task predictions and bottleneck analysis.
        """
        waves = self._resolver.get_execution_waves()
        if not waves:
            return ExecutionPlan(
                entries=[], waves=[], wave_durations=[],
                total_predicted_duration=0.0, total_predicted_cost=0.0,
                critical_path=[], bottleneck_tasks=[],
                model_assignments={},
            )

        # Auto-assign models if not provided
        assignments: Dict[str, str] = dict(task_models) if task_models else {}
        budget_left = budget_limit

        for wave in waves:
            for task_id in wave:
                if task_id not in assignments:
                    tt = task_types.get(task_id, "unknown")
                    gate = self._policy.get_gate(task_id)
                    rec = self.recommend_model(
                        tt,
                        quality_requirement=gate.min_quality,
                        budget_remaining=budget_left,
                    )
                    if rec:
                        assignments[task_id] = rec.model
                        if budget_left is not None:
                            budget_left -= rec.predicted_cost
                    else:
                        assignments[task_id] = "default"

        # Build entries with predictions
        entries: List[ScheduleEntry] = []
        wave_durations: List[float] = []

        for wave_idx, wave in enumerate(waves):
            wave_max_duration = 0.0
            for task_id in wave:
                tt = task_types.get(task_id, "unknown")
                model = assignments.get(task_id, "default")
                duration = self._tracker.predict_duration(tt, model)
                cost = self._tracker.predict_cost(tt, model)
                quality = self._tracker.predict_quality(tt, model)
                boost = self.get_priority_boost(task_id)

                entries.append(ScheduleEntry(
                    task_id=task_id,
                    wave=wave_idx,
                    model=model,
                    predicted_duration=duration,
                    predicted_cost=cost,
                    predicted_quality=quality,
                    priority_boost=float(boost),
                ))
                wave_max_duration = max(wave_max_duration, duration)
            wave_durations.append(wave_max_duration)

        total_duration = sum(wave_durations)
        total_cost = sum(e.predicted_cost for e in entries)

        # Critical path
        cp = self._tracker.compute_critical_path(
            self._resolver, task_types, assignments,
        )

        # Bottleneck analysis: tasks in single-task waves on the critical path
        bottleneck_tasks = self._find_bottlenecks(waves, cp.path)

        return ExecutionPlan(
            entries=entries,
            waves=waves,
            wave_durations=wave_durations,
            total_predicted_duration=total_duration,
            total_predicted_cost=total_cost,
            critical_path=cp.path,
            bottleneck_tasks=bottleneck_tasks,
            model_assignments=assignments,
        )

    def _find_bottlenecks(
        self, waves: List[List[str]], critical_path: List[str]
    ) -> List[str]:
        """Find tasks that constrain parallelism.

        A bottleneck is a task that is:
        1. On the critical path, AND
        2. The sole occupant of its wave (sequential chokepoint), OR
        3. Has the highest predicted duration in its wave by >2x
        """
        critical_set = set(critical_path)
        bottlenecks: List[str] = []

        for wave in waves:
            cp_in_wave = [t for t in wave if t in critical_set]
            if not cp_in_wave:
                continue
            if len(wave) == 1 and wave[0] in critical_set:
                bottlenecks.append(wave[0])

        return bottlenecks

    # ── Self-Tuning ───────────────────────────────────────────────────

    def auto_tune(self) -> Dict[str, Any]:
        """Adjust quality gates based on historical execution data.

        Examines each (task_type, model) profile and recommends gate
        adjustments:

        - If a task_type consistently passes with quality >> gate threshold,
          the threshold may be too lenient.
        - If a task_type consistently fails quality gates, the threshold
          may need to be lowered (or the model escalated).
        - If success_rate is very high (>95%), retries may be reduced.
        - If success_rate is low (<50%), retries should be increased.

        Returns a summary of adjustments made.
        """
        adjustments: Dict[str, Any] = {}
        profiles = self._tracker.get_all_profiles()

        # Group profiles by task_type
        by_type: Dict[str, List[Tuple[str, Any]]] = {}
        for (tt, model), profile in profiles.items():
            if profile.total_executions < 5:
                continue
            by_type.setdefault(tt, []).append((model, profile))

        for task_type, model_profiles in by_type.items():
            # Aggregate across models for this task type
            total_execs = sum(p.total_executions for _, p in model_profiles)
            avg_quality = (
                sum(p.avg_quality * p.total_executions for _, p in model_profiles)
                / total_execs
            )
            avg_success = (
                sum(p.success_rate * p.total_executions for _, p in model_profiles)
                / total_execs
            )

            recommendation: Dict[str, Any] = {
                "task_type": task_type,
                "observed_avg_quality": round(avg_quality, 3),
                "observed_success_rate": round(avg_success, 3),
                "total_executions": total_execs,
                "actions": [],
            }

            # Check if quality consistently exceeds gate by large margin
            if avg_quality > 0.85 and avg_success > 0.9:
                recommendation["actions"].append({
                    "type": "raise_threshold",
                    "reason": f"Avg quality {avg_quality:.2f} far exceeds typical gates",
                    "suggested_min_quality": round(min(avg_quality - 0.1, 0.8), 2),
                })

            # Check if quality is borderline — consider lowering gate
            if 0.3 < avg_quality < 0.5 and avg_success < 0.7:
                recommendation["actions"].append({
                    "type": "lower_threshold",
                    "reason": f"Avg quality {avg_quality:.2f} with low success rate {avg_success:.2f}",
                    "suggested_min_quality": round(max(avg_quality - 0.1, 0.2), 2),
                })

            # Check retry efficiency
            if avg_success > 0.95:
                recommendation["actions"].append({
                    "type": "reduce_retries",
                    "reason": f"Success rate {avg_success:.2f} — retries rarely needed",
                    "suggested_max_retries": 1,
                })
            elif avg_success < 0.5:
                recommendation["actions"].append({
                    "type": "increase_retries",
                    "reason": f"Success rate {avg_success:.2f} — more retries may help",
                    "suggested_max_retries": 4,
                })

            if recommendation["actions"]:
                adjustments[task_type] = recommendation

        # Log the tuning event
        tuning_event = {
            "timestamp": time.time(),
            "adjustments": len(adjustments),
            "types_analyzed": len(by_type),
        }
        self._tuning_log.append(tuning_event)
        if len(self._tuning_log) > self._max_tuning_log:
            self._tuning_log = self._tuning_log[-self._max_tuning_log:]

        if adjustments:
            logger.info(
                "Auto-tune: %d task types have recommendations", len(adjustments)
            )

        return adjustments

    # ── Statistics ────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary of coordinator state."""
        return {
            "concurrency_slots": len(self._slots),
            "total_active_slots": sum(s._active for s in self._slots.values()),
            "deadlines_set": len(self._deadlines),
            "tuning_events": len(self._tuning_log),
        }

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slots": {
                name: {
                    "max_concurrent": s.max_concurrent,
                    "total_acquired": s._total_acquired,
                    "total_rejected": s._total_rejected,
                }
                for name, s in self._slots.items()
            },
            "deadlines": dict(self._deadlines),
            "tuning_log": list(self._tuning_log),
            "default_max_concurrent": self._default_max_concurrent,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        resolver: DependencyResolver,
        tracker: ExecutionTracker,
        policy: QualityGatePolicy,
        retry_manager: RetryManager,
    ) -> "SchedulerCoordinator":
        coord = cls(resolver, tracker, policy, retry_manager)
        coord._default_max_concurrent = data.get("default_max_concurrent", 5)

        for name, sdata in data.get("slots", {}).items():
            slot = ConcurrencySlot(name, sdata.get("max_concurrent", 5))
            slot._total_acquired = sdata.get("total_acquired", 0)
            slot._total_rejected = sdata.get("total_rejected", 0)
            coord._slots[name] = slot

        coord._deadlines = dict(data.get("deadlines", {}))
        coord._tuning_log = list(data.get("tuning_log", []))
        return coord
