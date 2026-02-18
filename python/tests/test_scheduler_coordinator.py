"""Tests for the Scheduler Coordinator — unified scheduling intelligence."""

import time
import pytest
from unittest.mock import AsyncMock

from core.scheduling.dependency_resolver import DependencyResolver
from core.scheduling.execution_tracker import ExecutionTracker
from core.scheduling.quality_gates import (
    QualityGatePolicy,
    QualityGate,
    GateAction,
    GATE_STANDARD,
    GATE_STRICT,
)
from core.scheduling.retry_strategies import RetryManager
from core.scheduling.scheduler_coordinator import (
    ConcurrencySlot,
    ExecutionPlan,
    ModelRecommendation,
    ScheduleEntry,
    SchedulerCoordinator,
)


@pytest.fixture(autouse=True)
def _reset_container():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


def _build_coordinator(
    resolver=None, tracker=None, policy=None, retry_mgr=None,
) -> SchedulerCoordinator:
    """Helper to build a coordinator with optional overrides."""
    return SchedulerCoordinator(
        resolver=resolver or DependencyResolver(),
        tracker=tracker or ExecutionTracker(),
        policy=policy or QualityGatePolicy(),
        retry_manager=retry_mgr or RetryManager(),
    )


def _seed_tracker(tracker: ExecutionTracker, task_type: str, model: str,
                   n: int = 5, quality: float = 0.8, cost: float = 0.05) -> None:
    """Seed a tracker with N execution records."""
    for i in range(n):
        tracker.record_start(f"{task_type}_{model}_{i}")
        tracker.record_completion(
            f"{task_type}_{model}_{i}",
            task_type=task_type, model=model,
            quality_score=quality, cost=cost,
        )


# ========================================================================
# PARETO-OPTIMAL MODEL SELECTION
# ========================================================================


class TestParetoFrontier:
    def test_single_model_is_frontier(self):
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "opus", quality=0.9, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        frontier = coord.compute_pareto_frontier("coding")
        assert len(frontier) == 1
        assert frontier[0].model == "opus"

    def test_dominated_model_excluded(self):
        """If model A is strictly better and cheaper than B, B is excluded."""
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "opus", quality=0.95, cost=0.05)
        _seed_tracker(tracker, "coding", "flash", quality=0.7, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        frontier = coord.compute_pareto_frontier("coding")
        # Opus dominates flash (higher quality AND lower cost)
        assert len(frontier) == 1
        assert frontier[0].model == "opus"

    def test_pareto_frontier_two_non_dominated(self):
        """Two models on the frontier: one cheap/low-quality, one expensive/high."""
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "flash", quality=0.6, cost=0.001)
        _seed_tracker(tracker, "coding", "opus", quality=0.95, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        frontier = coord.compute_pareto_frontier("coding")
        assert len(frontier) == 2
        models = {r.model for r in frontier}
        assert models == {"flash", "opus"}

    def test_pareto_sorted_by_efficiency(self):
        """Frontier is sorted by quality/cost ratio."""
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "flash", quality=0.6, cost=0.001)
        _seed_tracker(tracker, "coding", "opus", quality=0.95, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        frontier = coord.compute_pareto_frontier("coding")
        # Flash has much higher efficiency (600 vs 9.5)
        assert frontier[0].model == "flash"

    def test_empty_frontier_no_history(self):
        coord = _build_coordinator()
        frontier = coord.compute_pareto_frontier("coding")
        assert frontier == []

    def test_min_executions_filter(self):
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "opus", n=1, quality=0.9, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        # Default min_executions=2, but only 1 record
        frontier = coord.compute_pareto_frontier("coding")
        assert frontier == []

        # Lower threshold
        frontier = coord.compute_pareto_frontier("coding", min_executions=1)
        assert len(frontier) == 1


class TestModelRecommendation:
    def test_recommend_with_quality_filter(self):
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "flash", quality=0.6, cost=0.001)
        _seed_tracker(tracker, "coding", "opus", quality=0.95, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        # Require quality >= 0.9 — only opus qualifies
        rec = coord.recommend_model("coding", quality_requirement=0.9)
        assert rec is not None
        assert rec.model == "opus"

    def test_recommend_with_budget_filter(self):
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "flash", quality=0.6, cost=0.001)
        _seed_tracker(tracker, "coding", "opus", quality=0.95, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        # Budget only allows $0.01 — flash is the only affordable option
        rec = coord.recommend_model("coding", budget_remaining=0.01)
        assert rec is not None
        assert rec.model == "flash"

    def test_recommend_prefer_speed(self):
        tracker = ExecutionTracker()
        # Flash is fast, opus is slow
        for i in range(5):
            tracker.record_start(f"flash_{i}")
            time.sleep(0.001)
            tracker.record_completion(
                f"flash_{i}", "coding", "flash", quality_score=0.7, cost=0.001,
            )
        for i in range(5):
            tracker.record_start(f"opus_{i}")
            time.sleep(0.005)  # 5x slower
            tracker.record_completion(
                f"opus_{i}", "coding", "opus", quality_score=0.9, cost=0.10,
            )
        coord = _build_coordinator(tracker=tracker)

        rec = coord.recommend_model("coding", prefer_speed=True)
        assert rec is not None
        assert rec.model == "flash"

    def test_recommend_returns_none_no_history(self):
        coord = _build_coordinator()
        assert coord.recommend_model("coding") is None

    def test_recommend_relaxes_quality_if_nothing_meets(self):
        """If no model meets quality_requirement, return the highest available."""
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "flash", quality=0.5, cost=0.001)
        coord = _build_coordinator(tracker=tracker)

        rec = coord.recommend_model("coding", quality_requirement=0.99)
        assert rec is not None
        assert rec.model == "flash"


# ========================================================================
# CONCURRENCY CONTROL
# ========================================================================


class TestConcurrency:
    def test_acquire_and_release(self):
        coord = _build_coordinator()
        coord.set_provider_limit("anthropic", 2)

        assert coord.acquire_slot("anthropic") is True
        assert coord.acquire_slot("anthropic") is True
        assert coord.acquire_slot("anthropic") is False  # Full

        coord.release_slot("anthropic")
        assert coord.acquire_slot("anthropic") is True  # Freed up

    def test_default_limit(self):
        coord = _build_coordinator()
        # No explicit limit — should use default (5)
        for _ in range(5):
            assert coord.acquire_slot("openai") is True
        assert coord.acquire_slot("openai") is False

    def test_release_without_acquire_is_safe(self):
        coord = _build_coordinator()
        coord.release_slot("nonexistent")  # Should not raise

    def test_concurrency_status(self):
        coord = _build_coordinator()
        coord.set_provider_limit("anthropic", 3)
        coord.acquire_slot("anthropic")
        coord.acquire_slot("anthropic")

        status = coord.get_concurrency_status()
        assert "anthropic" in status
        assert status["anthropic"]["active"] == 2
        assert status["anthropic"]["available"] == 1
        assert status["anthropic"]["utilization"] == pytest.approx(0.67, abs=0.01)

    def test_slot_statistics(self):
        slot = ConcurrencySlot("test", max_concurrent=2)
        slot.acquire()
        slot.acquire()
        slot.acquire()  # rejected

        assert slot._total_acquired == 2
        assert slot._total_rejected == 1


# ========================================================================
# DEADLINE MANAGEMENT
# ========================================================================


class TestDeadlines:
    def test_no_deadline_zero_urgency(self):
        coord = _build_coordinator()
        assert coord.get_urgency("t1") == 0.0

    def test_past_deadline_max_urgency(self):
        coord = _build_coordinator()
        coord.set_deadline("t1", time.time() - 100)
        assert coord.get_urgency("t1") == 1.0

    def test_imminent_deadline_high_urgency(self):
        coord = _build_coordinator()
        coord.set_deadline("t1", time.time() + 60)  # 1 minute away
        urgency = coord.get_urgency("t1")
        assert urgency > 0.9  # Very urgent

    def test_far_deadline_low_urgency(self):
        coord = _build_coordinator()
        coord.set_deadline("t1", time.time() + 86400 * 7)  # 1 week away
        urgency = coord.get_urgency("t1")
        assert urgency < 0.1

    def test_priority_boost_from_urgency(self):
        coord = _build_coordinator()
        # No deadline → no boost
        assert coord.get_priority_boost("t1") == 0

        # Imminent deadline → max boost
        coord.set_deadline("t2", time.time() + 30)
        assert coord.get_priority_boost("t2") == 3

    def test_clear_deadline(self):
        coord = _build_coordinator()
        coord.set_deadline("t1", time.time() + 60)
        assert coord.get_urgency("t1") > 0
        coord.clear_deadline("t1")
        assert coord.get_urgency("t1") == 0.0

    def test_urgency_curve_medium_range(self):
        """6 hours out should be medium urgency (0.1–0.5)."""
        coord = _build_coordinator()
        coord.set_deadline("t1", time.time() + 6 * 3600)
        urgency = coord.get_urgency("t1")
        assert 0.1 < urgency < 0.5


# ========================================================================
# EXECUTION PLANNING
# ========================================================================


class TestExecutionPlanning:
    def test_plan_empty_dag(self):
        coord = _build_coordinator()
        plan = coord.plan_execution({})
        assert plan.entries == []
        assert plan.total_predicted_duration == 0.0
        assert plan.total_predicted_cost == 0.0

    def test_plan_linear_chain(self):
        resolver = DependencyResolver()
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["B"])

        coord = _build_coordinator(resolver=resolver)
        plan = coord.plan_execution(
            task_types={"A": "plan", "B": "code", "C": "test"},
            task_models={"A": "flash", "B": "codex", "C": "flash"},
        )

        assert len(plan.entries) == 3
        assert len(plan.waves) == 3
        assert plan.total_predicted_duration > 0
        assert plan.model_assignments == {"A": "flash", "B": "codex", "C": "flash"}

    def test_plan_auto_assigns_models(self):
        """When no task_models provided, coordinator picks models from history."""
        resolver = DependencyResolver()
        resolver.add_task("A")

        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "opus", quality=0.9, cost=0.10)
        _seed_tracker(tracker, "coding", "flash", quality=0.6, cost=0.001)

        coord = _build_coordinator(resolver=resolver, tracker=tracker)
        plan = coord.plan_execution(task_types={"A": "coding"})

        # Should auto-assign based on Pareto analysis
        assert plan.model_assignments["A"] in ("opus", "flash")
        assert len(plan.entries) == 1

    def test_plan_with_budget_limit(self):
        resolver = DependencyResolver()
        for i in range(5):
            resolver.add_task(f"t{i}")

        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "opus", quality=0.95, cost=0.50)
        _seed_tracker(tracker, "coding", "flash", quality=0.6, cost=0.001)

        coord = _build_coordinator(resolver=resolver, tracker=tracker)
        plan = coord.plan_execution(
            task_types={f"t{i}": "coding" for i in range(5)},
            budget_limit=1.0,  # Can afford ~2 opus or ~1000 flash
        )

        # Should assign models within budget
        assert len(plan.entries) == 5
        assert plan.total_predicted_cost > 0

    def test_plan_bottleneck_detection(self):
        """Single-task waves on critical path are bottlenecks."""
        resolver = DependencyResolver()
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])  # Sequential bottleneck
        resolver.add_task("C", dependencies=["B"])

        coord = _build_coordinator(resolver=resolver)
        plan = coord.plan_execution(
            task_types={"A": "code", "B": "code", "C": "test"},
            task_models={"A": "x", "B": "x", "C": "x"},
        )

        # All three tasks are in single-task waves (linear chain)
        # and on the critical path → all are bottlenecks
        assert len(plan.bottleneck_tasks) >= 1

    def test_plan_entries_have_wave_numbers(self):
        resolver = DependencyResolver()
        resolver.add_task("A")
        resolver.add_task("B")
        resolver.add_task("C", dependencies=["A", "B"])

        coord = _build_coordinator(resolver=resolver)
        plan = coord.plan_execution(
            task_types={"A": "x", "B": "x", "C": "x"},
            task_models={"A": "m", "B": "m", "C": "m"},
        )

        waves = {e.task_id: e.wave for e in plan.entries}
        assert waves["A"] == 0
        assert waves["B"] == 0
        assert waves["C"] == 1


# ========================================================================
# SELF-TUNING
# ========================================================================


class TestAutoTune:
    def test_tune_no_data(self):
        coord = _build_coordinator()
        adjustments = coord.auto_tune()
        assert adjustments == {}

    def test_tune_high_quality_suggests_raise(self):
        tracker = ExecutionTracker()
        _seed_tracker(tracker, "coding", "opus", n=10, quality=0.95, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        adjustments = coord.auto_tune()
        assert "coding" in adjustments
        actions = adjustments["coding"]["actions"]
        action_types = {a["type"] for a in actions}
        assert "raise_threshold" in action_types

    def test_tune_low_success_suggests_increase_retries(self):
        tracker = ExecutionTracker()
        # Seed with many failures
        for i in range(10):
            tracker.record_start(f"t{i}")
            if i < 4:  # 40% success rate
                tracker.record_completion(f"t{i}", "coding", "flash", quality_score=0.3, cost=0.001)
            else:
                tracker.record_failure(f"t{i}", "coding", "flash")

        coord = _build_coordinator(tracker=tracker)
        adjustments = coord.auto_tune()

        if "coding" in adjustments:
            action_types = {a["type"] for a in adjustments["coding"]["actions"]}
            assert "increase_retries" in action_types

    def test_tune_high_success_suggests_reduce_retries(self):
        tracker = ExecutionTracker()
        # 100% success rate
        _seed_tracker(tracker, "review", "opus", n=10, quality=0.92, cost=0.10)
        coord = _build_coordinator(tracker=tracker)

        adjustments = coord.auto_tune()
        if "review" in adjustments:
            action_types = {a["type"] for a in adjustments["review"]["actions"]}
            assert "reduce_retries" in action_types

    def test_tuning_log_capped(self):
        coord = _build_coordinator()
        for _ in range(100):
            coord.auto_tune()
        assert len(coord._tuning_log) <= coord._max_tuning_log


# ========================================================================
# SERIALIZATION
# ========================================================================


class TestSerialization:
    def test_roundtrip(self):
        resolver = DependencyResolver()
        tracker = ExecutionTracker()
        policy = QualityGatePolicy()
        retry_mgr = RetryManager()

        coord = SchedulerCoordinator(resolver, tracker, policy, retry_mgr)
        coord.set_provider_limit("anthropic", 3)
        coord.set_deadline("t1", time.time() + 3600)
        coord.acquire_slot("anthropic")

        data = coord.to_dict()
        restored = SchedulerCoordinator.from_dict(
            data, resolver, tracker, policy, retry_mgr,
        )

        assert "anthropic" in restored._slots
        assert restored._slots["anthropic"].max_concurrent == 3
        assert "t1" in restored._deadlines
        assert restored._default_max_concurrent == 5

    def test_stats(self):
        coord = _build_coordinator()
        coord.set_provider_limit("anthropic", 3)
        coord.acquire_slot("anthropic")
        coord.set_deadline("t1", time.time() + 3600)
        coord.auto_tune()

        stats = coord.stats
        assert stats["concurrency_slots"] == 1
        assert stats["total_active_slots"] == 1
        assert stats["deadlines_set"] == 1
        assert stats["tuning_events"] == 1


# ========================================================================
# ORCHESTRATOR INTEGRATION
# ========================================================================


class TestOrchestratorIntegration:
    @pytest.fixture
    def orchestrator(self):
        from core.orchestration.unified_orchestrator import UnifiedOrchestrator
        bus = AsyncMock()
        bus.publish = AsyncMock()
        bus.subscribe = AsyncMock()
        mem = AsyncMock()
        mem.retrieve = AsyncMock(return_value=None)
        mem.store = AsyncMock()
        return UnifiedOrchestrator(event_bus=bus, memory_store=mem)

    @pytest.mark.asyncio
    async def test_orchestrator_has_scheduler(self, orchestrator):
        assert hasattr(orchestrator, '_scheduler')
        assert isinstance(orchestrator._scheduler, SchedulerCoordinator)

    @pytest.mark.asyncio
    async def test_statistics_include_scheduler(self, orchestrator):
        stats = orchestrator.get_statistics()
        assert "scheduler_coordinator" in stats

    @pytest.mark.asyncio
    async def test_persist_includes_scheduler(self, orchestrator):
        await orchestrator._persist_state()
        calls = orchestrator.memory_store.store.call_args_list
        keys = [c[0][0] for c in calls]
        assert "orchestrator:scheduler_coord" in keys

    @pytest.mark.asyncio
    async def test_scheduler_references_same_modules(self, orchestrator):
        """Coordinator holds references to the same module instances."""
        assert orchestrator._scheduler._resolver is orchestrator._dep_resolver
        assert orchestrator._scheduler._tracker is orchestrator._exec_tracker
        assert orchestrator._scheduler._policy is orchestrator._quality_policy
        assert orchestrator._scheduler._retry_manager is orchestrator._retry_manager
