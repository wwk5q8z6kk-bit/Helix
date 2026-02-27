"""Execution intelligence layer for Helix task scheduling.

Tracks per-task execution metrics, builds historical profiles keyed by
(task_type, model), and computes critical-path analysis over the DAG.

This is the *data* layer that enables quality gates, adaptive retry, and
cost-optimised scheduling decisions.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from core.scheduling.dependency_resolver import DependencyResolver, TaskState

logger = logging.getLogger(__name__)


# ── Value objects ────────────────────────────────────────────────────


@dataclass
class ExecutionRecord:
    """Immutable record of a single task execution."""

    task_id: str
    task_type: str
    model: str
    agent_id: Optional[str] = None

    # Timing
    start_time: float = 0.0
    end_time: float = 0.0

    # Outcome
    success: bool = False
    quality_score: float = 0.0
    hrm_score: float = 0.0

    # Cost
    tokens_used: int = 0
    cost: float = 0.0

    # Retry context
    attempt: int = 1

    @property
    def duration(self) -> float:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0


@dataclass
class ExecutionProfile:
    """Aggregated statistics for a (task_type, model) pair.

    Updated incrementally — no need to recompute from scratch.
    """

    task_type: str
    model: str
    total_executions: int = 0
    successes: int = 0
    failures: int = 0

    # Running averages (incrementally updated)
    avg_duration: float = 0.0
    avg_quality: float = 0.0
    avg_hrm: float = 0.0
    avg_cost: float = 0.0
    avg_tokens: float = 0.0

    # Extremes
    min_duration: float = float("inf")
    max_duration: float = 0.0
    best_quality: float = 0.0
    worst_quality: float = 1.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total_executions if self.total_executions else 0.0

    def update(self, record: ExecutionRecord) -> None:
        """Incrementally update the profile with a new execution record."""
        self.total_executions += 1
        if record.success:
            self.successes += 1
        else:
            self.failures += 1

        # Incremental average: new_avg = old_avg + (new_val - old_avg) / n
        n = self.total_executions
        self.avg_duration += (record.duration - self.avg_duration) / n
        self.avg_quality += (record.quality_score - self.avg_quality) / n
        self.avg_hrm += (record.hrm_score - self.avg_hrm) / n
        self.avg_cost += (record.cost - self.avg_cost) / n
        self.avg_tokens += (record.tokens_used - self.avg_tokens) / n

        self.min_duration = min(self.min_duration, record.duration)
        self.max_duration = max(self.max_duration, record.duration)
        self.best_quality = max(self.best_quality, record.quality_score)
        self.worst_quality = min(self.worst_quality, record.quality_score)


@dataclass
class CriticalPathResult:
    """Result of critical-path analysis over a DAG."""

    path: List[str]  # task_ids on the critical path
    total_duration: float  # predicted total duration
    total_cost: float  # predicted total cost
    waves: List[List[str]]  # parallel execution waves
    wave_durations: List[float]  # predicted duration per wave


# ── Tracker ──────────────────────────────────────────────────────────


class ExecutionTracker:
    """Tracks execution metrics and builds predictive profiles.

    Designed to be attached to the orchestrator and fed ExecutionRecords
    after each task completes.
    """

    def __init__(self) -> None:
        # All execution records, keyed by task_id (most recent attempt)
        self._records: Dict[str, ExecutionRecord] = {}
        # Historical profiles keyed by (task_type, model)
        self._profiles: Dict[Tuple[str, str], ExecutionProfile] = {}
        # In-flight tasks (task_id → start_time)
        self._in_flight: Dict[str, float] = {}
        # Default duration estimate when no history exists
        self._default_duration: float = 10.0
        # Default cost estimate when no history exists
        self._default_cost: float = 0.01

    # ── Recording ────────────────────────────────────────────────────

    def record_start(self, task_id: str) -> None:
        """Mark a task as started."""
        self._in_flight[task_id] = time.time()

    def record_completion(
        self,
        task_id: str,
        task_type: str,
        model: str,
        quality_score: float = 0.0,
        hrm_score: float = 0.0,
        tokens_used: int = 0,
        cost: float = 0.0,
        agent_id: Optional[str] = None,
        attempt: int = 1,
    ) -> ExecutionRecord:
        """Record a successful task completion."""
        start = self._in_flight.pop(task_id, time.time())
        record = ExecutionRecord(
            task_id=task_id,
            task_type=task_type,
            model=model,
            agent_id=agent_id,
            start_time=start,
            end_time=time.time(),
            success=True,
            quality_score=quality_score,
            hrm_score=hrm_score,
            tokens_used=tokens_used,
            cost=cost,
            attempt=attempt,
        )
        self._records[task_id] = record
        self._update_profile(record)
        return record

    def record_failure(
        self,
        task_id: str,
        task_type: str,
        model: str,
        agent_id: Optional[str] = None,
        attempt: int = 1,
    ) -> ExecutionRecord:
        """Record a task failure."""
        start = self._in_flight.pop(task_id, time.time())
        record = ExecutionRecord(
            task_id=task_id,
            task_type=task_type,
            model=model,
            agent_id=agent_id,
            start_time=start,
            end_time=time.time(),
            success=False,
            attempt=attempt,
        )
        self._records[task_id] = record
        self._update_profile(record)
        return record

    # ── Prediction ───────────────────────────────────────────────────

    def predict_duration(self, task_type: str, model: str) -> float:
        """Predict how long a task will take based on historical data."""
        profile = self._profiles.get((task_type, model))
        if profile and profile.total_executions >= 3:
            return profile.avg_duration
        # Fall back to task_type average across all models
        type_profiles = [p for (t, _), p in self._profiles.items() if t == task_type]
        if type_profiles:
            total = sum(p.avg_duration * p.total_executions for p in type_profiles)
            count = sum(p.total_executions for p in type_profiles)
            return total / count if count else self._default_duration
        return self._default_duration

    def predict_quality(self, task_type: str, model: str) -> float:
        """Predict expected quality score based on historical data."""
        profile = self._profiles.get((task_type, model))
        if profile and profile.successes >= 3:
            return profile.avg_quality
        return 0.5  # neutral default

    def predict_cost(self, task_type: str, model: str) -> float:
        """Predict expected cost based on historical data."""
        profile = self._profiles.get((task_type, model))
        if profile and profile.total_executions >= 3:
            return profile.avg_cost
        return self._default_cost

    def predict_success_rate(self, task_type: str, model: str) -> float:
        """Predict success probability based on historical data."""
        profile = self._profiles.get((task_type, model))
        if profile and profile.total_executions >= 5:
            return profile.success_rate
        return 0.9  # optimistic default

    # ── Critical path analysis ───────────────────────────────────────

    def compute_critical_path(
        self,
        resolver: DependencyResolver,
        task_types: Dict[str, str],
        task_models: Dict[str, str],
    ) -> CriticalPathResult:
        """Compute the critical path through the DAG.

        The critical path is the longest path by predicted duration —
        the theoretical minimum time to complete all tasks.

        Args:
            resolver: The dependency resolver with the current DAG.
            task_types: Mapping of task_id → task_type string.
            task_models: Mapping of task_id → model string.

        Returns:
            CriticalPathResult with path, durations, and cost estimates.
        """
        waves = resolver.get_execution_waves()
        if not waves:
            return CriticalPathResult([], 0.0, 0.0, [], [])

        # Predict duration for each task
        durations: Dict[str, float] = {}
        costs: Dict[str, float] = {}
        for wave in waves:
            for task_id in wave:
                tt = task_types.get(task_id, "unknown")
                model = task_models.get(task_id, "unknown")
                durations[task_id] = self.predict_duration(tt, model)
                costs[task_id] = self.predict_cost(tt, model)

        # Wave durations = max duration within each wave (parallel execution)
        wave_durations = [
            max(durations.get(tid, self._default_duration) for tid in wave)
            for wave in waves
        ]

        # Critical path = longest path through the DAG
        # Use dynamic programming: earliest_finish[task] = max(earliest_finish[dep]) + duration[task]
        earliest_finish: Dict[str, float] = {}
        predecessors: Dict[str, Optional[str]] = {}

        for wave in waves:
            for task_id in wave:
                deps = resolver._dependencies.get(task_id, set())
                if not deps:
                    earliest_finish[task_id] = durations.get(task_id, 0.0)
                    predecessors[task_id] = None
                else:
                    best_pred = None
                    best_ef = 0.0
                    for dep in deps:
                        ef = earliest_finish.get(dep, 0.0)
                        if ef > best_ef:
                            best_ef = ef
                            best_pred = dep
                    earliest_finish[task_id] = best_ef + durations.get(task_id, 0.0)
                    predecessors[task_id] = best_pred

        # Trace back from the task with the largest earliest_finish
        if not earliest_finish:
            return CriticalPathResult([], 0.0, 0.0, waves, wave_durations)

        end_task = max(earliest_finish, key=earliest_finish.get)  # type: ignore[arg-type]
        path: List[str] = []
        current: Optional[str] = end_task
        while current is not None:
            path.append(current)
            current = predecessors.get(current)
        path.reverse()

        total_duration = sum(wave_durations)
        total_cost = sum(costs.values())

        return CriticalPathResult(
            path=path,
            total_duration=total_duration,
            total_cost=total_cost,
            waves=waves,
            wave_durations=wave_durations,
        )

    # ── Queries ──────────────────────────────────────────────────────

    def get_profile(self, task_type: str, model: str) -> Optional[ExecutionProfile]:
        """Get the execution profile for a (task_type, model) pair."""
        return self._profiles.get((task_type, model))

    def get_record(self, task_id: str) -> Optional[ExecutionRecord]:
        """Get the most recent execution record for a task."""
        return self._records.get(task_id)

    def get_all_profiles(self) -> Dict[Tuple[str, str], ExecutionProfile]:
        """Get all execution profiles."""
        return dict(self._profiles)

    def get_best_model_for(self, task_type: str, min_quality: float = 0.5) -> Optional[str]:
        """Find the model with the best quality/cost ratio for a task type."""
        candidates: List[Tuple[str, float]] = []
        for (tt, model), profile in self._profiles.items():
            if tt != task_type or profile.successes < 2:
                continue
            if profile.avg_quality >= min_quality:
                # Quality/cost efficiency — higher is better
                efficiency = profile.avg_quality / max(profile.avg_cost, 0.001)
                candidates.append((model, efficiency))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        return {
            "total_records": len(self._records),
            "in_flight": len(self._in_flight),
            "profiles": len(self._profiles),
            "total_cost": sum(r.cost for r in self._records.values()),
            "total_tokens": sum(r.tokens_used for r in self._records.values()),
            "avg_quality": (
                sum(r.quality_score for r in self._records.values() if r.success)
                / max(1, sum(1 for r in self._records.values() if r.success))
            ),
        }

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serialise for persistence."""
        return {
            "profiles": {
                f"{tt}:{model}": {
                    "task_type": p.task_type,
                    "model": p.model,
                    "total_executions": p.total_executions,
                    "successes": p.successes,
                    "failures": p.failures,
                    "avg_duration": p.avg_duration,
                    "avg_quality": p.avg_quality,
                    "avg_hrm": p.avg_hrm,
                    "avg_cost": p.avg_cost,
                    "avg_tokens": p.avg_tokens,
                    "min_duration": p.min_duration if p.min_duration != float("inf") else None,
                    "max_duration": p.max_duration,
                    "best_quality": p.best_quality,
                    "worst_quality": p.worst_quality,
                }
                for (tt, model), p in self._profiles.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionTracker":
        """Restore from persisted state."""
        tracker = cls()
        for _key, pdata in data.get("profiles", {}).items():
            tt = pdata["task_type"]
            model = pdata["model"]
            profile = ExecutionProfile(task_type=tt, model=model)
            profile.total_executions = pdata.get("total_executions", 0)
            profile.successes = pdata.get("successes", 0)
            profile.failures = pdata.get("failures", 0)
            profile.avg_duration = pdata.get("avg_duration", 0.0)
            profile.avg_quality = pdata.get("avg_quality", 0.0)
            profile.avg_hrm = pdata.get("avg_hrm", 0.0)
            profile.avg_cost = pdata.get("avg_cost", 0.0)
            profile.avg_tokens = pdata.get("avg_tokens", 0.0)
            min_d = pdata.get("min_duration")
            profile.min_duration = min_d if min_d is not None else float("inf")
            profile.max_duration = pdata.get("max_duration", 0.0)
            profile.best_quality = pdata.get("best_quality", 0.0)
            profile.worst_quality = pdata.get("worst_quality", 1.0)
            tracker._profiles[(tt, model)] = profile
        return tracker

    # ── Internal ─────────────────────────────────────────────────────

    def _update_profile(self, record: ExecutionRecord) -> None:
        """Update the (task_type, model) profile with a new record."""
        key = (record.task_type, record.model)
        if key not in self._profiles:
            self._profiles[key] = ExecutionProfile(
                task_type=record.task_type,
                model=record.model,
            )
        self._profiles[key].update(record)
