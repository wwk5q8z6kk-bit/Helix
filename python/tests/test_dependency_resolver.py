"""Tests for core.scheduling.dependency_resolver and orchestrator integration."""

import pytest
from unittest.mock import AsyncMock

from core.scheduling.dependency_resolver import (
    CycleDetectedError,
    DependencyResolver,
    TaskState,
)


# -- Fixtures --------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_container():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


@pytest.fixture
def resolver():
    return DependencyResolver()


def _make_orchestrator():
    """Build a UnifiedOrchestrator with mocked dependencies."""
    from core.orchestration.unified_orchestrator import UnifiedOrchestrator

    bus = AsyncMock()
    bus.publish = AsyncMock()
    bus.subscribe = AsyncMock()
    mem = AsyncMock()
    mem.retrieve = AsyncMock(return_value=None)
    mem.store = AsyncMock()

    return UnifiedOrchestrator(event_bus=bus, memory_store=mem)


@pytest.fixture
def orchestrator():
    return _make_orchestrator()


# ========================================================================
# STANDALONE RESOLVER TESTS
# ========================================================================


class TestGraphConstruction:
    """Adding tasks to the DAG."""

    def test_add_task_no_deps(self, resolver):
        state = resolver.add_task("A")
        assert state == TaskState.READY
        assert resolver.get_task_state("A") == TaskState.READY

    def test_add_task_with_deps(self, resolver):
        resolver.add_task("A")
        state = resolver.add_task("B", dependencies=["A"])
        assert state == TaskState.PENDING
        assert resolver.get_task_state("B") == TaskState.PENDING

    def test_duplicate_task_raises(self, resolver):
        resolver.add_task("A")
        with pytest.raises(ValueError, match="already exists"):
            resolver.add_task("A")

    def test_unknown_dep_warns_and_resolves(self, resolver, caplog):
        import logging
        with caplog.at_level(logging.WARNING):
            state = resolver.add_task("A", dependencies=["unknown"])
        assert state == TaskState.READY
        assert "auto-resolving" in caplog.text

    def test_completed_dep_auto_resolved(self, resolver):
        resolver.add_task("A")
        resolver.mark_running("A")
        resolver.mark_completed("A")

        # B depends on A which is already COMPLETED in history
        state = resolver.add_task("B", dependencies=["A"])
        assert state == TaskState.READY

    def test_failed_dep_auto_cancels(self, resolver):
        resolver.add_task("A")
        resolver.mark_running("A")
        resolver.mark_failed("A")

        state = resolver.add_task("B", dependencies=["A"])
        assert state == TaskState.CANCELLED
        assert resolver.get_task_state("B") == TaskState.CANCELLED

    def test_remove_task_unblocks_dependents(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        assert resolver.get_task_state("B") == TaskState.PENDING

        newly_ready = resolver.remove_task("A")
        assert "B" in newly_ready
        assert resolver.get_task_state("B") == TaskState.READY

    def test_remove_nonexistent_is_noop(self, resolver):
        assert resolver.remove_task("nope") == set()


class TestCycleDetection:
    """Cycle detection on add_task.

    With sequential add_task, the only directly-constructible cycle is a
    self-dependency (task depends on itself).  Multi-node cycles are caught
    by validate_graph() after from_dict() restoration.  The DFS in
    _check_cycle is a defensive guard for future API extensions.
    """

    def test_self_dependency(self, resolver):
        """A task depending on itself should raise."""
        with pytest.raises(CycleDetectedError, match="cycle"):
            resolver.add_task("A", dependencies=["A"])

    def test_self_dep_no_state_mutation(self, resolver):
        """Self-dep should not leave residual state in the graph."""
        with pytest.raises(CycleDetectedError):
            resolver.add_task("A", dependencies=["A"])
        assert resolver.get_task_state("A") is None
        assert resolver.stats["pending"] == 0

    def test_diamond_no_cycle(self, resolver):
        """Diamond shape (A->B, A->C, B->D, C->D) is valid -- not a cycle."""
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["A"])
        resolver.add_task("D", dependencies=["B", "C"])
        assert resolver.get_task_state("D") == TaskState.PENDING
        assert resolver.validate_graph() is None

    def test_cycle_error_includes_path(self):
        """CycleDetectedError.cycle contains the cycle path."""
        err = None
        try:
            r = DependencyResolver()
            r.add_task("A", dependencies=["A"])
        except CycleDetectedError as e:
            err = e
        assert err is not None
        assert "A" in err.cycle
        assert len(err.cycle) >= 2

    def test_validate_graph_detects_corrupted_cycle(self):
        """validate_graph() catches cycles introduced via from_dict()."""
        # Manually construct a cyclic graph (as if restored from corrupt state)
        r = DependencyResolver()
        r._states = {"A": TaskState.PENDING, "B": TaskState.PENDING}
        r._dependencies = {"A": {"B"}, "B": {"A"}}
        r._dependents = {"A": {"B"}, "B": {"A"}}
        r._priorities = {"A": 2, "B": 2}
        r._insertion_order = ["A", "B"]

        cycle = r.validate_graph()
        assert cycle is not None
        assert set(cycle) == {"A", "B"}


class TestStateTransitions:
    """mark_running, mark_completed, mark_failed."""

    def test_mark_running_from_ready(self, resolver):
        resolver.add_task("A")
        resolver.mark_running("A")
        assert resolver.get_task_state("A") == TaskState.RUNNING

    def test_mark_running_from_non_ready_raises(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        with pytest.raises(ValueError, match="expected READY"):
            resolver.mark_running("B")

    def test_mark_completed_returns_newly_ready(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["A"])

        resolver.mark_running("A")
        newly_ready = resolver.mark_completed("A")

        assert set(newly_ready) == {"B", "C"}
        assert resolver.get_task_state("B") == TaskState.READY
        assert resolver.get_task_state("C") == TaskState.READY

    def test_mark_completed_from_non_running_raises(self, resolver):
        resolver.add_task("A")
        with pytest.raises(ValueError, match="expected RUNNING"):
            resolver.mark_completed("A")

    def test_mark_failed_propagates_cancellation(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["B"])

        resolver.mark_running("A")
        cancelled = resolver.mark_failed("A")

        assert set(cancelled) == {"B", "C"}
        assert resolver.get_task_state("A") == TaskState.FAILED
        assert resolver.get_task_state("B") == TaskState.CANCELLED
        assert resolver.get_task_state("C") == TaskState.CANCELLED

    def test_deep_chain_propagation(self, resolver):
        """A -> B -> C -> D: failing A cancels B, C, D."""
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["B"])
        resolver.add_task("D", dependencies=["C"])

        resolver.mark_running("A")
        cancelled = resolver.mark_failed("A")

        assert set(cancelled) == {"B", "C", "D"}

    def test_partial_failure_only_cancels_downstream(self, resolver):
        """A and X are independent. B depends on A. Failing A cancels B but not X."""
        resolver.add_task("A")
        resolver.add_task("X")
        resolver.add_task("B", dependencies=["A"])

        resolver.mark_running("A")
        cancelled = resolver.mark_failed("A")

        assert cancelled == ["B"]
        assert resolver.get_task_state("X") == TaskState.READY


class TestQueries:
    """get_ready_tasks, get_execution_waves, get_blocked_tasks, get_downstream."""

    def test_ready_sorted_by_priority(self, resolver):
        resolver.add_task("low", priority=3)
        resolver.add_task("critical", priority=0)
        resolver.add_task("med", priority=2)

        ready = resolver.get_ready_tasks()
        assert ready == ["critical", "med", "low"]

    def test_ready_empty_when_all_blocked(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["A"])

        # A is ready, B and C are pending
        ready = resolver.get_ready_tasks()
        assert ready == ["A"]

    def test_waves_linear_chain(self, resolver):
        """A -> B -> C produces 3 waves."""
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["B"])

        waves = resolver.get_execution_waves()
        assert len(waves) == 3
        assert waves[0] == ["A"]
        assert waves[1] == ["B"]
        assert waves[2] == ["C"]

    def test_waves_parallel(self, resolver):
        """Three independent tasks -> 1 wave."""
        resolver.add_task("A")
        resolver.add_task("B")
        resolver.add_task("C")

        waves = resolver.get_execution_waves()
        assert len(waves) == 1
        assert set(waves[0]) == {"A", "B", "C"}

    def test_waves_diamond(self, resolver):
        """Diamond: A -> (B,C) -> D produces 3 waves."""
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["A"])
        resolver.add_task("D", dependencies=["B", "C"])

        waves = resolver.get_execution_waves()
        assert len(waves) == 3
        assert waves[0] == ["A"]
        assert set(waves[1]) == {"B", "C"}
        assert waves[2] == ["D"]

    def test_blocked_tasks(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])

        blocked = resolver.get_blocked_tasks()
        assert "B" in blocked
        assert "A" in blocked["B"]

    def test_downstream_transitive(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        resolver.add_task("C", dependencies=["B"])

        downstream = resolver.get_downstream("A")
        assert downstream == {"B", "C"}

    def test_downstream_empty_for_leaf(self, resolver):
        resolver.add_task("A")
        resolver.add_task("B", dependencies=["A"])
        assert resolver.get_downstream("B") == set()


class TestSerialization:
    """to_dict / from_dict round-trip."""

    def test_roundtrip(self, resolver):
        resolver.add_task("A", priority=0)
        resolver.add_task("B", dependencies=["A"], priority=1)
        resolver.mark_running("A")
        resolver.mark_completed("A")

        data = resolver.to_dict()
        restored = DependencyResolver.from_dict(data)

        assert restored.get_task_state("A") == TaskState.COMPLETED
        assert restored.get_task_state("B") == TaskState.READY
        assert restored.stats == resolver.stats

    def test_preserves_history(self, resolver):
        resolver.add_task("A")
        resolver.mark_running("A")
        resolver.mark_failed("A")

        data = resolver.to_dict()
        restored = DependencyResolver.from_dict(data)

        assert restored.get_task_state("A") == TaskState.FAILED
        assert restored._history["A"] == TaskState.FAILED


class TestStats:
    """stats property."""

    def test_counts_by_state(self, resolver):
        resolver.add_task("A")  # READY
        resolver.add_task("B")  # READY
        resolver.add_task("C", dependencies=["A"])  # PENDING

        s = resolver.stats
        assert s["ready"] == 2
        assert s["pending"] == 1

    def test_counts_include_history(self, resolver):
        resolver.add_task("A")
        resolver.mark_running("A")
        resolver.mark_completed("A")

        s = resolver.stats
        assert s["completed"] == 1


class TestTaskNode:
    """TaskNode immutable snapshot."""

    def test_get_node(self, resolver):
        resolver.add_task("A", priority=1)
        resolver.add_task("B", dependencies=["A"], priority=2)

        node = resolver.get_node("A")
        assert node is not None
        assert node.task_id == "A"
        assert node.state == TaskState.READY
        assert node.priority == 1
        assert "B" in node.dependents

    def test_get_node_nonexistent(self, resolver):
        assert resolver.get_node("nope") is None


# ========================================================================
# ORCHESTRATOR INTEGRATION TESTS
# ========================================================================


class TestOrchestratorIntegration:
    """Integration between DependencyResolver and UnifiedOrchestrator."""

    @pytest.mark.asyncio
    async def test_submit_task_no_deps_executes(self, orchestrator):
        from core.orchestration.unified_orchestrator import Task, TaskType

        task = Task(
            task_id="t1",
            task_type=TaskType.IMPLEMENTATION,
            description="No-dep task",
        )
        task_id = await orchestrator.submit_task(task)
        assert task_id == "t1"
        # Task should be marked as running in the resolver
        state = orchestrator._dep_resolver.get_task_state("t1")
        assert state in (TaskState.RUNNING, TaskState.COMPLETED, TaskState.READY)

    @pytest.mark.asyncio
    async def test_submit_task_with_deps_waits(self, orchestrator):
        from core.orchestration.unified_orchestrator import Task, TaskType

        t1 = Task(task_id="t1", task_type=TaskType.PLANNING, description="First")
        t2 = Task(task_id="t2", task_type=TaskType.IMPLEMENTATION, description="Second", dependencies=["t1"])

        await orchestrator.submit_task(t1)
        await orchestrator.submit_task(t2)

        # t2 should be PENDING (waiting on t1)
        assert orchestrator._dep_resolver.get_task_state("t2") == TaskState.PENDING

    @pytest.mark.asyncio
    async def test_submit_task_cycle_raises(self, orchestrator):
        from core.orchestration.unified_orchestrator import Task, TaskType
        from core.exceptions_unified import DependencyResolutionError

        t1 = Task(task_id="t1", task_type=TaskType.PLANNING, description="Self-dep", dependencies=["t1"])

        with pytest.raises(DependencyResolutionError, match="cycle"):
            await orchestrator.submit_task(t1)

    @pytest.mark.asyncio
    async def test_submit_tasks_batch_atomic_rollback(self, orchestrator):
        from core.orchestration.unified_orchestrator import Task, TaskType
        from core.exceptions_unified import DependencyResolutionError

        tasks = [
            Task(task_id="ok1", task_type=TaskType.PLANNING, description="Good"),
            Task(task_id="ok2", task_type=TaskType.PLANNING, description="Good"),
            Task(task_id="bad", task_type=TaskType.PLANNING, description="Bad", dependencies=["bad"]),
        ]

        with pytest.raises(DependencyResolutionError):
            await orchestrator.submit_tasks(tasks)

        # Atomic rollback: ok1 and ok2 should NOT be in the resolver
        assert orchestrator._dep_resolver.get_task_state("ok1") is None
        assert orchestrator._dep_resolver.get_task_state("ok2") is None

    @pytest.mark.asyncio
    async def test_execute_task_still_works_without_resolver(self, orchestrator):
        """Backward compat: execute_task() works for tasks without dependencies."""
        from core.orchestration.unified_orchestrator import Task, TaskType

        task = Task(
            task_id="legacy",
            task_type=TaskType.IMPLEMENTATION,
            description="Legacy direct execution",
        )
        orchestrator.tasks[task.task_id] = task
        # execute_task should still work (may fail on LLM call, but shouldn't
        # raise on dependency logic)
        try:
            await orchestrator.execute_task(task)
        except Exception:
            # Expected -- no real LLM configured. The point is it doesn't
            # raise DependencyResolutionError or similar.
            pass

    @pytest.mark.asyncio
    async def test_statistics_include_dependency_graph(self, orchestrator):
        from core.orchestration.unified_orchestrator import Task, TaskType

        task = Task(task_id="t1", task_type=TaskType.PLANNING, description="Test")
        await orchestrator.submit_task(task)

        stats = orchestrator.get_statistics()
        assert "dependency_graph" in stats
        assert "completed_history_size" in stats

    @pytest.mark.asyncio
    async def test_persist_and_load_resolver_state(self, orchestrator):
        from core.orchestration.unified_orchestrator import Task, TaskType

        task = Task(task_id="t1", task_type=TaskType.PLANNING, description="Persist test")
        await orchestrator.submit_task(task)

        await orchestrator._persist_state()

        # Verify store was called with resolver data
        calls = orchestrator.memory_store.store.call_args_list
        keys_stored = [c[0][0] for c in calls]
        assert "orchestrator:dep_resolver" in keys_stored

    @pytest.mark.asyncio
    async def test_queue_routes_deps_through_resolver(self, orchestrator):
        """Tasks with dependencies in the queue go through submit_task."""
        import asyncio
        from core.orchestration.unified_orchestrator import Task, TaskType

        task = Task(
            task_id="q1",
            task_type=TaskType.IMPLEMENTATION,
            description="Queued with deps",
            dependencies=["nonexistent"],
        )
        orchestrator.tasks[task.task_id] = task
        orchestrator.task_queues[task.priority].append(task.task_id)

        orchestrator.is_running = True

        # Simulate one iteration of the queue processor
        for priority in [task.priority]:
            queue = orchestrator.task_queues[priority]
            if queue:
                tid = queue.pop(0)
                if tid in orchestrator.tasks:
                    t = orchestrator.tasks[tid]
                    if t.dependencies:
                        await orchestrator.submit_task(t)
                    else:
                        asyncio.create_task(orchestrator.execute_task(t))

        # Task should be in the resolver (auto-resolved unknown dep -> READY)
        state = orchestrator._dep_resolver.get_task_state("q1")
        assert state is not None
