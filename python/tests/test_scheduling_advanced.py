"""Tests for execution tracking, quality gates, and adaptive retry strategies."""

import time
import pytest
from unittest.mock import AsyncMock

from core.scheduling.execution_tracker import (
    ExecutionTracker,
    ExecutionRecord,
    ExecutionProfile,
)
from core.scheduling.quality_gates import (
    GateAction,
    GateVerdict,
    QualityGate,
    QualityGatePolicy,
    GATE_STRICT,
    GATE_STANDARD,
    GATE_LENIENT,
    GATE_DISABLED,
)
from core.scheduling.retry_strategies import (
    EscalationLevel,
    EscalationStep,
    RetryDecision,
    RetryManager,
    RetryReason,
    RetryStrategy,
    strategy_fast_to_premium,
    strategy_quality_first,
    strategy_budget_conscious,
    strategy_no_retry,
)
from core.scheduling.dependency_resolver import DependencyResolver


@pytest.fixture(autouse=True)
def _reset_container():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


# ========================================================================
# EXECUTION TRACKER TESTS
# ========================================================================


class TestExecutionTracker:
    def test_record_completion(self):
        tracker = ExecutionTracker()
        tracker.record_start("t1")
        record = tracker.record_completion(
            "t1", task_type="coding", model="gpt-5.3-codex",
            quality_score=0.85, tokens_used=1000, cost=0.03,
        )
        assert record.success is True
        assert record.quality_score == 0.85
        assert record.duration > 0

    def test_record_failure(self):
        tracker = ExecutionTracker()
        tracker.record_start("t1")
        record = tracker.record_failure("t1", task_type="coding", model="gpt-5.3-codex")
        assert record.success is False

    def test_profile_incremental_update(self):
        tracker = ExecutionTracker()
        for i in range(5):
            tracker.record_start(f"t{i}")
            tracker.record_completion(
                f"t{i}", task_type="coding", model="gpt-5.3-codex",
                quality_score=0.8 + i * 0.02, cost=0.03,
            )
        profile = tracker.get_profile("coding", "gpt-5.3-codex")
        assert profile is not None
        assert profile.total_executions == 5
        assert profile.successes == 5
        assert 0.8 <= profile.avg_quality <= 0.9

    def test_predict_duration_with_history(self):
        tracker = ExecutionTracker()
        # Build history
        for i in range(5):
            tracker.record_start(f"t{i}")
            time.sleep(0.001)  # tiny delay for nonzero duration
            tracker.record_completion(f"t{i}", task_type="coding", model="opus")
        duration = tracker.predict_duration("coding", "opus")
        assert duration > 0

    def test_predict_duration_without_history(self):
        tracker = ExecutionTracker()
        # Should return default
        duration = tracker.predict_duration("unknown", "unknown")
        assert duration == 10.0  # default

    def test_predict_quality_with_history(self):
        tracker = ExecutionTracker()
        for i in range(5):
            tracker.record_start(f"t{i}")
            tracker.record_completion(
                f"t{i}", task_type="review", model="opus",
                quality_score=0.9,
            )
        q = tracker.predict_quality("review", "opus")
        assert abs(q - 0.9) < 0.01

    def test_predict_quality_without_history(self):
        tracker = ExecutionTracker()
        assert tracker.predict_quality("x", "y") == 0.5

    def test_get_best_model_for(self):
        tracker = ExecutionTracker()
        # Model A: high quality, high cost
        for i in range(3):
            tracker.record_start(f"a{i}")
            tracker.record_completion(
                f"a{i}", task_type="coding", model="opus",
                quality_score=0.95, cost=0.10,
            )
        # Model B: decent quality, low cost
        for i in range(3):
            tracker.record_start(f"b{i}")
            tracker.record_completion(
                f"b{i}", task_type="coding", model="flash",
                quality_score=0.75, cost=0.001,
            )
        best = tracker.get_best_model_for("coding", min_quality=0.7)
        # Flash should win on quality/cost efficiency
        assert best == "flash"

    def test_critical_path_linear_chain(self):
        tracker = ExecutionTracker()
        resolver = DependencyResolver()
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["B"])

        result = tracker.compute_critical_path(
            resolver,
            task_types={"A": "plan", "B": "code", "C": "test"},
            task_models={"A": "flash", "B": "codex", "C": "flash"},
        )
        assert result.path == ["A", "B", "C"]
        assert len(result.waves) == 3
        assert result.total_duration > 0

    def test_critical_path_diamond(self):
        tracker = ExecutionTracker()
        resolver = DependencyResolver()
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["A"])
        resolver.add_task("D", dependencies=["B", "C"])

        result = tracker.compute_critical_path(
            resolver,
            task_types={"A": "plan", "B": "code", "C": "code", "D": "test"},
            task_models={"A": "x", "B": "x", "C": "x", "D": "x"},
        )
        assert "A" in result.path
        assert "D" in result.path
        assert len(result.waves) == 3

    def test_serialization_roundtrip(self):
        tracker = ExecutionTracker()
        tracker.record_start("t1")
        tracker.record_completion("t1", "coding", "opus", quality_score=0.9, cost=0.05)

        data = tracker.to_dict()
        restored = ExecutionTracker.from_dict(data)

        profile = restored.get_profile("coding", "opus")
        assert profile is not None
        assert profile.total_executions == 1
        assert abs(profile.avg_quality - 0.9) < 0.01

    def test_stats(self):
        tracker = ExecutionTracker()
        tracker.record_start("t1")
        tracker.record_completion("t1", "coding", "opus", quality_score=0.8, cost=0.02)

        s = tracker.stats
        assert s["total_records"] == 1
        assert s["profiles"] == 1
        assert abs(s["total_cost"] - 0.02) < 0.001


# ========================================================================
# QUALITY GATES TESTS
# ========================================================================


class TestQualityGates:
    def test_gate_passed(self):
        policy = QualityGatePolicy()
        result = policy.evaluate("t1", quality_score=0.8, hrm_score=0.7)
        assert result.verdict == GateVerdict.PASSED

    def test_gate_retry_on_low_quality(self):
        policy = QualityGatePolicy(default_gate=GATE_STANDARD)
        result = policy.evaluate("t1", quality_score=0.3)
        assert result.verdict == GateVerdict.RETRY

    def test_gate_retry_exhausted_becomes_fail(self):
        gate = QualityGate(min_quality=0.8, max_retries=2, gate_action=GateAction.FAIL)
        policy = QualityGatePolicy(default_gate=gate)
        # 3 attempts, all below threshold
        for _ in range(2):
            result = policy.evaluate("t1", quality_score=0.3)
            assert result.verdict == GateVerdict.RETRY
        result = policy.evaluate("t1", quality_score=0.3)
        assert result.verdict == GateVerdict.FAILED

    def test_gate_warn_and_pass(self):
        policy = QualityGatePolicy(default_gate=GATE_LENIENT)
        # First attempt retries, second passes with warning
        result = policy.evaluate("t1", quality_score=0.1)
        assert result.verdict == GateVerdict.RETRY
        result = policy.evaluate("t1", quality_score=0.1)
        assert result.verdict == GateVerdict.WARN_PASS

    def test_gate_block_action(self):
        gate = QualityGate(min_quality=0.9, max_retries=0, gate_action=GateAction.BLOCK)
        policy = QualityGatePolicy(default_gate=gate)
        result = policy.evaluate("t1", quality_score=0.5)
        assert result.verdict == GateVerdict.BLOCKED

    def test_gate_disabled_always_passes(self):
        policy = QualityGatePolicy(default_gate=GATE_DISABLED)
        result = policy.evaluate("t1", quality_score=0.0)
        assert result.verdict == GateVerdict.PASSED

    def test_task_level_override(self):
        policy = QualityGatePolicy(default_gate=GATE_LENIENT)
        policy.set_task_gate("critical_task", GATE_STRICT)

        # Regular task — lenient
        r1 = policy.evaluate("regular", quality_score=0.4)
        assert r1.verdict == GateVerdict.PASSED  # 0.4 > 0.3 (lenient)

        # Critical task — strict
        r2 = policy.evaluate("critical_task", quality_score=0.4)
        assert r2.verdict == GateVerdict.RETRY  # 0.4 < 0.7 (strict)

    def test_edge_level_override(self):
        policy = QualityGatePolicy(default_gate=GATE_LENIENT)
        strict_edge = QualityGate(min_quality=0.9, max_retries=1, gate_action=GateAction.FAIL)
        policy.set_edge_gate("upstream", "downstream", strict_edge)

        result = policy.evaluate(
            "upstream", quality_score=0.5,
            downstream_ids=["downstream"],
        )
        assert result.verdict == GateVerdict.RETRY

    def test_hrm_gate(self):
        gate = QualityGate(min_quality=0.5, min_hrm=0.7, max_retries=1, gate_action=GateAction.FAIL)
        policy = QualityGatePolicy(default_gate=gate)
        # Quality OK but HRM too low
        result = policy.evaluate("t1", quality_score=0.8, hrm_score=0.3)
        assert result.verdict == GateVerdict.RETRY
        assert "hrm" in result.reason

    def test_escalation_model_suggestion(self):
        gate = QualityGate(
            min_quality=0.8, max_retries=3,
            escalation_models=("gpt-5.3-codex", "claude-opus-4-6", "claude-opus-4-6-thinking"),
        )
        policy = QualityGatePolicy(default_gate=gate)
        r1 = policy.evaluate("t1", quality_score=0.3)
        assert r1.suggested_model == "gpt-5.3-codex"
        r2 = policy.evaluate("t1", quality_score=0.3)
        assert r2.suggested_model == "claude-opus-4-6"

    def test_reset_attempts(self):
        policy = QualityGatePolicy(default_gate=GATE_STANDARD)
        policy.evaluate("t1", quality_score=0.1)  # attempt 1
        policy.evaluate("t1", quality_score=0.1)  # attempt 2
        policy.reset_attempts("t1")
        # After reset, should be attempt 1 again
        result = policy.evaluate("t1", quality_score=0.1)
        assert result.attempt == 1

    def test_serialization_roundtrip(self):
        policy = QualityGatePolicy(default_gate=GATE_STRICT)
        policy.set_task_gate("important", GATE_STANDARD)
        policy.evaluate("important", quality_score=0.1)  # build some state

        data = policy.to_dict()
        restored = QualityGatePolicy.from_dict(data)

        assert restored._default_gate.min_quality == GATE_STRICT.min_quality
        assert "important" in restored._task_gates


# ========================================================================
# RETRY STRATEGIES TESTS
# ========================================================================


class TestRetryStrategies:
    def test_simple_retry(self):
        strategy = RetryStrategy(name="simple")  # No ladder, defaults to 3 attempts
        d = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is True
        d = strategy.decide(2, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is True
        d = strategy.decide(3, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is False

    def test_escalation_ladder(self):
        strategy = strategy_fast_to_premium()
        # Attempt 1: flash (1 attempt)
        d1 = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d1.should_retry is True
        assert d1.model == "gpt-5.3-codex"  # Escalated past flash

        # Attempt 2: codex (first of 2)
        d2 = strategy.decide(2, RetryReason.EXECUTION_FAILURE)
        assert d2.should_retry is True
        assert d2.model == "gpt-5.3-codex"

        # Attempt 3: codex → opus
        d3 = strategy.decide(3, RetryReason.EXECUTION_FAILURE)
        assert d3.should_retry is True
        assert d3.model == "claude-opus-4-6"

        # Attempt 4: exhausted
        d4 = strategy.decide(4, RetryReason.EXECUTION_FAILURE)
        assert d4.should_retry is False

    def test_quality_first_strategy(self):
        strategy = strategy_quality_first()
        d = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is True
        assert d.model == "claude-opus-4-6"

    def test_budget_conscious_strategy(self):
        strategy = strategy_budget_conscious()
        d = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is True
        assert "flash" in d.model or "mini" in d.model

    def test_no_retry_strategy(self):
        strategy = strategy_no_retry()
        d = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is False

    def test_quality_retry_disabled(self):
        strategy = RetryStrategy(name="no_quality_retry", retry_on_low_quality=False)
        d = strategy.decide(1, RetryReason.LOW_QUALITY, quality_score=0.1)
        assert d.should_retry is False

    def test_quality_meets_threshold_no_retry(self):
        strategy = RetryStrategy(
            name="test", retry_on_low_quality=True, min_acceptable_quality=0.5,
        )
        d = strategy.decide(1, RetryReason.LOW_QUALITY, quality_score=0.6)
        assert d.should_retry is False

    def test_budget_checker_blocks_escalation(self):
        strategy = RetryStrategy(
            name="budget_blocked",
            ladder=[
                EscalationStep(EscalationLevel.PREMIUM, "claude-opus-4-6", max_attempts=2),
            ],
            budget_checker=lambda model, cost: False,  # Always reject
        )
        d = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is False
        assert d.budget_check_passed is False

    def test_budget_checker_allows_escalation(self):
        strategy = RetryStrategy(
            name="budget_ok",
            ladder=[
                EscalationStep(EscalationLevel.FAST, "flash", max_attempts=2),
            ],
            budget_checker=lambda model, cost: True,  # Always allow
        )
        d = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d.should_retry is True

    def test_timeout_multiplier(self):
        strategy = RetryStrategy(
            name="slow",
            ladder=[
                EscalationStep(EscalationLevel.FAST, "flash", max_attempts=1),
                EscalationStep(EscalationLevel.PREMIUM, "opus", max_attempts=1, timeout_multiplier=3.0),
            ],
        )
        d = strategy.decide(1, RetryReason.EXECUTION_FAILURE)
        assert d.model == "opus"
        assert d.timeout_multiplier == 3.0


class TestRetryManager:
    def test_on_failure_returns_decision(self):
        manager = RetryManager()
        d = manager.on_failure("t1", RetryReason.EXECUTION_FAILURE)
        assert isinstance(d, RetryDecision)

    def test_retry_attempts_increment(self):
        manager = RetryManager(default_strategy=strategy_fast_to_premium())
        d1 = manager.on_failure("t1", RetryReason.EXECUTION_FAILURE)
        assert d1.attempt == 1
        d2 = manager.on_failure("t1", RetryReason.EXECUTION_FAILURE)
        assert d2.attempt == 2

    def test_reset_clears_state(self):
        manager = RetryManager()
        manager.on_failure("t1")
        manager.on_failure("t1")
        manager.reset("t1")
        d = manager.on_failure("t1")
        assert d.attempt == 1

    def test_per_task_strategy(self):
        manager = RetryManager(default_strategy=strategy_fast_to_premium())
        manager.set_strategy("critical", strategy_quality_first())

        d_default = manager.on_failure("regular")
        d_critical = manager.on_failure("critical")

        # Critical should use quality_first (opus), regular should use fast_to_premium
        assert d_critical.model == "claude-opus-4-6"

    def test_history_tracking(self):
        manager = RetryManager()
        manager.on_failure("t1")
        manager.on_failure("t1")
        history = manager.get_history("t1")
        assert len(history) == 2

    def test_stats(self):
        manager = RetryManager()
        manager.on_failure("t1")
        manager.on_failure("t2")
        s = manager.stats
        assert s["tasks_with_retries"] == 2
        assert s["total_retry_decisions"] == 2

    def test_serialization(self):
        manager = RetryManager()
        manager.on_failure("t1")
        data = manager.to_dict()
        restored = RetryManager.from_dict(data)
        assert restored._task_attempts.get("t1") == 1


# ========================================================================
# INTEGRATION: ALL THREE SYSTEMS TOGETHER
# ========================================================================


class TestIntegratedScheduling:
    """Test that quality gates, execution tracking, and retry strategies
    work together through the orchestrator."""

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
    async def test_orchestrator_has_all_systems(self, orchestrator):
        assert hasattr(orchestrator, '_exec_tracker')
        assert hasattr(orchestrator, '_quality_policy')
        assert hasattr(orchestrator, '_retry_manager')

    @pytest.mark.asyncio
    async def test_statistics_include_all_systems(self, orchestrator):
        stats = orchestrator.get_statistics()
        assert "dependency_graph" in stats
        assert "execution_intelligence" in stats
        assert "retry_manager" in stats
        assert "completed_history_size" in stats

    @pytest.mark.asyncio
    async def test_persist_includes_all_systems(self, orchestrator):
        await orchestrator._persist_state()
        calls = orchestrator.memory_store.store.call_args_list
        keys = [c[0][0] for c in calls]
        assert "orchestrator:exec_tracker" in keys
        assert "orchestrator:quality_policy" in keys
        assert "orchestrator:retry_manager" in keys
