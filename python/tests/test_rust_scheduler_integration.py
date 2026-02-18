"""Tests for Rust-delegated scheduling in the UnifiedOrchestrator.

Verifies that when a RustCoreBridge is provided with scheduler methods,
the orchestrator routes DAG operations through the Rust REST API instead
of the local Python DependencyResolver.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from core.orchestration.unified_orchestrator import (
    UnifiedOrchestrator,
    Task,
    TaskType,
    TaskPriority,
    TaskStatus,
    create_orchestrator,
)
from core.scheduling.dependency_resolver import TaskState as DepTaskState
import core.di_container as di_mod


@pytest.fixture(autouse=True)
def _reset_container():
    di_mod._container = None
    yield
    di_mod._container = None


def _make_mock_bridge():
    """Create a mock RustCoreBridge with all scheduler methods."""
    bridge = AsyncMock()
    bridge.is_healthy = AsyncMock(return_value=True)
    bridge.store = AsyncMock()
    bridge.retrieve = AsyncMock(return_value=None)
    bridge.delete = AsyncMock()
    bridge.list_keys = AsyncMock(return_value=[])

    # Scheduler methods
    bridge.scheduler_submit_task = AsyncMock(return_value={"task_id": "t1", "state": "ready"})
    bridge.scheduler_ready_tasks = AsyncMock(return_value=[])
    bridge.scheduler_mark_running = AsyncMock(return_value=True)
    bridge.scheduler_mark_completed = AsyncMock(return_value=[])
    bridge.scheduler_mark_failed = AsyncMock(return_value=[])
    bridge.scheduler_task_state = AsyncMock(return_value="running")
    bridge.scheduler_execution_waves = AsyncMock(return_value=[])
    bridge.scheduler_stats = AsyncMock(return_value={})
    return bridge


def _make_task(task_id="t1", deps=None, priority=TaskPriority.MEDIUM):
    return Task(
        task_id=task_id,
        task_type=TaskType.IMPLEMENTATION,
        priority=priority,
        description=f"Test task {task_id}",
        dependencies=deps or [],
    )


@pytest.fixture
def mock_bridge():
    return _make_mock_bridge()


@pytest.fixture
def event_bus():
    eb = AsyncMock()
    eb.publish = AsyncMock()
    eb.subscribe = AsyncMock()
    return eb


@pytest.fixture
def orchestrator(event_bus, mock_bridge):
    """Orchestrator with Rust scheduler enabled."""
    return UnifiedOrchestrator(
        event_bus=event_bus,
        memory_store=mock_bridge,
        rust_bridge=mock_bridge,
    )


@pytest.fixture
def orchestrator_no_rust(event_bus, mock_bridge):
    """Orchestrator without Rust bridge (Python fallback)."""
    return UnifiedOrchestrator(
        event_bus=event_bus,
        memory_store=mock_bridge,
    )


# --- Detection ---


def test_rust_scheduler_detected(orchestrator):
    assert orchestrator._use_rust_scheduler is True


def test_rust_scheduler_not_detected_without_bridge(orchestrator_no_rust):
    assert orchestrator_no_rust._use_rust_scheduler is False


def test_rust_scheduler_not_detected_with_plain_object():
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()
    event_bus.subscribe = AsyncMock()
    memory = AsyncMock()
    memory.retrieve = AsyncMock(return_value=None)
    # Pass a plain object without scheduler methods
    orch = UnifiedOrchestrator(
        event_bus=event_bus,
        memory_store=memory,
        rust_bridge=object(),
    )
    assert orch._use_rust_scheduler is False


# --- submit_task ---


async def test_submit_task_routes_to_rust(orchestrator, mock_bridge):
    task = _make_task("t1")
    result = await orchestrator.submit_task(task)
    assert result == "t1"
    mock_bridge.scheduler_submit_task.assert_awaited_once_with(
        "t1", dependencies=[], priority=2,
    )


async def test_submit_task_with_deps_routes_to_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_submit_task.return_value = {"task_id": "t2", "state": "blocked"}
    task = _make_task("t2", deps=["t1"])
    result = await orchestrator.submit_task(task)
    assert result == "t2"
    mock_bridge.scheduler_submit_task.assert_awaited_once_with(
        "t2", dependencies=["t1"], priority=2,
    )


async def test_submit_task_cancelled_by_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_submit_task.return_value = {"task_id": "t1", "state": "cancelled"}
    task = _make_task("t1")
    await orchestrator.submit_task(task)
    assert task.status == TaskStatus.CANCELLED


async def test_submit_task_ready_dispatches(orchestrator, mock_bridge):
    mock_bridge.scheduler_submit_task.return_value = {"task_id": "t1", "state": "ready"}
    mock_bridge.scheduler_ready_tasks.return_value = ["t1"]
    task = _make_task("t1")
    await orchestrator.submit_task(task)
    # Should call ready_tasks and mark_running
    mock_bridge.scheduler_ready_tasks.assert_awaited()
    mock_bridge.scheduler_mark_running.assert_awaited_with("t1")


async def test_submit_task_failure_raises(orchestrator, mock_bridge):
    mock_bridge.scheduler_submit_task.return_value = None
    task = _make_task("t1")
    with pytest.raises(Exception, match="Rust scheduler rejected"):
        await orchestrator.submit_task(task)


# --- submit_tasks batch ---


async def test_submit_tasks_batch_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_submit_task.side_effect = [
        {"task_id": "a", "state": "ready"},
        {"task_id": "b", "state": "blocked"},
    ]
    mock_bridge.scheduler_ready_tasks.return_value = ["a"]

    tasks = [_make_task("a"), _make_task("b", deps=["a"])]
    result = await orchestrator.submit_tasks(tasks)
    assert result == ["a", "b"]
    assert mock_bridge.scheduler_submit_task.await_count == 2


# --- _dispatch_ready_tasks ---


async def test_dispatch_uses_rust_ready_tasks(orchestrator, mock_bridge):
    mock_bridge.scheduler_ready_tasks.return_value = ["t1"]
    task = _make_task("t1")
    orchestrator.tasks["t1"] = task

    # Patch execute_task to avoid real execution
    with patch.object(orchestrator, "execute_task", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = MagicMock(
            quality_score=0.9, hrm_score=0.8, tokens_used=100,
            cost=0.01, metadata={"provider": "test"}, agent_id="a1",
        )
        await orchestrator._dispatch_ready_tasks()

    mock_bridge.scheduler_ready_tasks.assert_awaited_once()
    mock_bridge.scheduler_mark_running.assert_awaited_once_with("t1")
    assert task.status == TaskStatus.IN_PROGRESS


# --- _resolver_mark_completed ---


async def test_resolver_mark_completed_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_mark_completed.return_value = ["t2"]
    result = await orchestrator._resolver_mark_completed("t1")
    assert result == ["t2"]
    mock_bridge.scheduler_mark_completed.assert_awaited_once_with("t1")


async def test_resolver_mark_completed_python(orchestrator_no_rust):
    """Falls back to Python resolver."""
    orch = orchestrator_no_rust
    orch._dep_resolver.add_task("t1", priority=0)
    orch._dep_resolver.add_task("t2", dependencies=["t1"], priority=0)
    orch._dep_resolver.mark_running("t1")
    result = await orch._resolver_mark_completed("t1")
    assert "t2" in result


# --- _resolver_mark_failed ---


async def test_resolver_mark_failed_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_mark_failed.return_value = ["t2", "t3"]
    result = await orchestrator._resolver_mark_failed("t1")
    assert result == ["t2", "t3"]
    mock_bridge.scheduler_mark_failed.assert_awaited_once_with("t1")


async def test_resolver_mark_failed_python(orchestrator_no_rust):
    orch = orchestrator_no_rust
    orch._dep_resolver.add_task("t1", priority=0)
    orch._dep_resolver.add_task("t2", dependencies=["t1"], priority=0)
    orch._dep_resolver.mark_running("t1")
    result = await orch._resolver_mark_failed("t1")
    assert "t2" in result


# --- enable_rust_scheduler at runtime ---


async def test_enable_rust_scheduler_runtime(orchestrator_no_rust, mock_bridge):
    orch = orchestrator_no_rust
    assert orch._use_rust_scheduler is False
    orch._rust_bridge = mock_bridge
    result = await orch.enable_rust_scheduler()
    assert result is True
    assert orch._use_rust_scheduler is True


async def test_enable_rust_scheduler_fails_unhealthy(orchestrator_no_rust, mock_bridge):
    orch = orchestrator_no_rust
    mock_bridge.is_healthy.return_value = False
    orch._rust_bridge = mock_bridge
    result = await orch.enable_rust_scheduler()
    assert result is False


# --- get_statistics ---


def test_statistics_includes_rust_flag(orchestrator):
    stats = orchestrator.get_statistics()
    assert "rust_scheduler_active" in stats
    assert stats["rust_scheduler_active"] is True


def test_statistics_rust_flag_false(orchestrator_no_rust):
    stats = orchestrator_no_rust.get_statistics()
    assert stats["rust_scheduler_active"] is False


# --- create_orchestrator factory ---


def test_create_orchestrator_with_bridge():
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()
    event_bus.subscribe = AsyncMock()
    bridge = _make_mock_bridge()
    orch = create_orchestrator(
        event_bus=event_bus,
        memory_store=bridge,
        rust_bridge=bridge,
    )
    assert orch._use_rust_scheduler is True


def test_create_orchestrator_without_bridge():
    event_bus = AsyncMock()
    event_bus.publish = AsyncMock()
    event_bus.subscribe = AsyncMock()
    memory = AsyncMock()
    memory.retrieve = AsyncMock(return_value=None)
    orch = create_orchestrator(
        event_bus=event_bus,
        memory_store=memory,
    )
    assert orch._use_rust_scheduler is False


# --- Workflow lifecycle (Rust-delegated) ---


async def test_define_workflow_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_define_workflow = AsyncMock(return_value={
        "name": "wf1", "task_count": 2, "wave_count": 2,
    })
    result = await orchestrator.define_workflow(
        name="wf1",
        tasks=[
            {"task_id": "a", "task_type": "code_generation"},
            {"task_id": "b", "task_type": "testing", "depends_on": ["a"]},
        ],
    )
    assert result["name"] == "wf1"
    mock_bridge.scheduler_define_workflow.assert_awaited_once()


async def test_define_workflow_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.define_workflow(
        name="wf1",
        tasks=[
            {"task_id": "a", "task_type": "code_generation"},
            {"task_id": "b", "task_type": "testing", "depends_on": ["a"]},
        ],
    )
    assert result is not None
    assert result["name"] == "wf1"
    assert result["task_count"] == 2
    assert result["wave_count"] == 2
    assert result["task_order"] == ["a", "b"]


async def test_define_workflow_python_with_deadline_and_budget(orchestrator_no_rust):
    import time
    result = await orchestrator_no_rust.define_workflow(
        name="wf2",
        tasks=[{"task_id": "x", "task_type": "planning"}],
        deadline=time.time() + 3600,
        budget=50.0,
    )
    assert result is not None
    assert result["name"] == "wf2"


async def test_list_workflows_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_list_workflows = AsyncMock(return_value=[{"name": "a"}, {"name": "b"}])
    result = await orchestrator.list_workflows()
    assert len(result) == 2


async def test_list_workflows_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.list_workflows()
    assert result == []


async def test_get_workflow_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_get_workflow = AsyncMock(return_value={"name": "wf1"})
    result = await orchestrator.get_workflow("wf1")
    assert result["name"] == "wf1"


async def test_get_workflow_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.get_workflow("nope")
    assert result is None


async def test_delete_workflow_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_delete_workflow = AsyncMock(return_value=True)
    result = await orchestrator.delete_workflow("wf1")
    assert result is True


async def test_delete_workflow_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.delete_workflow("nope")
    assert result is False


async def test_preview_workflow_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_preview_workflow = AsyncMock(return_value={"feasible": True})
    result = await orchestrator.preview_workflow("wf1")
    assert result["feasible"] is True


async def test_preview_workflow_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.preview_workflow("any")
    assert result is None


async def test_list_templates_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_list_templates = AsyncMock(return_value=["t1", "t2"])
    result = await orchestrator.list_workflow_templates()
    assert result == ["t1", "t2"]


async def test_list_templates_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.list_workflow_templates()
    assert "code_review_pipeline" in result
    assert "research_pipeline" in result


async def test_preview_template_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_preview_template = AsyncMock(return_value={"name": "t1"})
    result = await orchestrator.preview_template("t1")
    assert result["name"] == "t1"


async def test_preview_template_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.preview_template("any")
    assert result is None


async def test_scheduler_stats_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_stats = AsyncMock(return_value={"total": 10, "ready": 3})
    result = await orchestrator.scheduler_stats()
    assert result["total"] == 10


async def test_scheduler_stats_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.scheduler_stats()
    assert isinstance(result, dict)


async def test_execution_waves_rust(orchestrator, mock_bridge):
    mock_bridge.scheduler_execution_waves = AsyncMock(return_value=[["a"], ["b", "c"]])
    result = await orchestrator.execution_waves()
    assert len(result) == 2
    assert result[0] == ["a"]


async def test_execution_waves_python_fallback(orchestrator_no_rust):
    result = await orchestrator_no_rust.execution_waves()
    assert isinstance(result, list)


# --- Workflow Execution ---


async def test_run_workflow_not_found(orchestrator_no_rust):
    with pytest.raises(ValueError, match="not found"):
        await orchestrator_no_rust.run_workflow("nonexistent")


async def test_run_workflow_python(orchestrator_no_rust):
    """Define a workflow then run it — Python fallback path."""
    orch = orchestrator_no_rust

    # Define a 2-task workflow
    result = await orch.define_workflow(
        name="test-wf",
        tasks=[
            {"task_id": "a", "task_type": "planning"},
            {"task_id": "b", "task_type": "testing", "depends_on": ["a"]},
        ],
    )
    assert result is not None

    # Patch execute_task to return immediately
    with patch.object(orch, "execute_task", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = MagicMock(
            quality_score=0.9, hrm_score=0.8, tokens_used=100,
            cost=0.01, metadata={"provider": "test"}, agent_id="a1",
            status="completed",
        )
        wf_exec = await orch.run_workflow("test-wf")

    assert wf_exec.execution_id.startswith("wfx-")
    assert wf_exec.workflow_name == "test-wf"
    assert wf_exec.status.value == "running"
    assert len(wf_exec.task_ids) == 2
    # Task IDs are namespaced
    assert all(wf_exec.execution_id in tid for tid in wf_exec.task_ids)


async def test_run_workflow_rust(orchestrator, mock_bridge):
    """Run workflow with Rust scheduler — verifies define + get path."""
    mock_bridge.scheduler_define_workflow = AsyncMock(return_value={"name": "wf1"})
    mock_bridge.scheduler_get_workflow = AsyncMock(return_value={
        "name": "wf1",
        "tasks": [
            {"task_id": "a", "task_type": "planning"},
            {"task_id": "b", "task_type": "testing", "depends_on": ["a"]},
        ],
    })
    mock_bridge.scheduler_submit_task.return_value = {"task_id": "t", "state": "ready"}
    mock_bridge.scheduler_ready_tasks.return_value = []

    with patch.object(orchestrator, "execute_task", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = MagicMock(
            quality_score=0.9, hrm_score=0.8, tokens_used=100,
            cost=0.01, metadata={"provider": "test"}, agent_id="a1",
        )
        wf_exec = await orchestrator.run_workflow("wf1")

    assert wf_exec.workflow_name == "wf1"
    assert len(wf_exec.task_ids) == 2


async def test_cancel_workflow_execution(orchestrator_no_rust):
    orch = orchestrator_no_rust

    await orch.define_workflow(
        name="cancel-me",
        tasks=[{"task_id": "a", "task_type": "planning"}],
    )

    with patch.object(orch, "execute_task", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = MagicMock(
            quality_score=0.9, hrm_score=0.8, tokens_used=100,
            cost=0.01, metadata={}, agent_id="a1",
        )
        wf_exec = await orch.run_workflow("cancel-me")

    cancelled = await orch.cancel_workflow_execution(wf_exec.execution_id)
    assert cancelled is True
    assert wf_exec.status.value == "cancelled"


async def test_cancel_nonexistent_returns_false(orchestrator_no_rust):
    result = await orchestrator_no_rust.cancel_workflow_execution("bogus-id")
    assert result is False


async def test_get_workflow_execution(orchestrator_no_rust):
    orch = orchestrator_no_rust

    await orch.define_workflow(
        name="get-me",
        tasks=[{"task_id": "x", "task_type": "planning"}],
    )

    with patch.object(orch, "execute_task", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = MagicMock(
            quality_score=0.9, hrm_score=0.8, tokens_used=100,
            cost=0.01, metadata={}, agent_id="a1",
        )
        wf_exec = await orch.run_workflow("get-me")

    result = orch.get_workflow_execution(wf_exec.execution_id)
    assert result is not None
    assert result["workflow_name"] == "get-me"
    assert result["execution_id"] == wf_exec.execution_id


async def test_get_workflow_execution_not_found(orchestrator_no_rust):
    result = orchestrator_no_rust.get_workflow_execution("bogus")
    assert result is None


async def test_list_workflow_executions(orchestrator_no_rust):
    orch = orchestrator_no_rust

    await orch.define_workflow(
        name="list-me",
        tasks=[{"task_id": "y", "task_type": "planning"}],
    )

    with patch.object(orch, "execute_task", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = MagicMock(
            quality_score=0.9, hrm_score=0.8, tokens_used=100,
            cost=0.01, metadata={}, agent_id="a1",
        )
        await orch.run_workflow("list-me")

    execs = orch.list_workflow_executions()
    assert len(execs) >= 1
    assert execs[0]["workflow_name"] == "list-me"


async def test_list_workflow_executions_filter_by_status(orchestrator_no_rust):
    orch = orchestrator_no_rust

    await orch.define_workflow(
        name="filter-wf",
        tasks=[{"task_id": "z", "task_type": "planning"}],
    )

    with patch.object(orch, "execute_task", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = MagicMock(
            quality_score=0.9, hrm_score=0.8, tokens_used=100,
            cost=0.01, metadata={}, agent_id="a1",
        )
        await orch.run_workflow("filter-wf")

    running = orch.list_workflow_executions(status="running")
    assert len(running) >= 1
    completed = orch.list_workflow_executions(status="completed")
    # Nothing completed yet
    assert len(completed) == 0


async def test_workflow_execution_to_dict():
    from core.orchestration.unified_orchestrator import WorkflowExecution, WorkflowExecutionStatus
    wf = WorkflowExecution(
        execution_id="wfx-test123",
        workflow_name="my-wf",
        status=WorkflowExecutionStatus.RUNNING,
        task_ids=["a", "b"],
    )
    d = wf.to_dict()
    assert d["execution_id"] == "wfx-test123"
    assert d["workflow_name"] == "my-wf"
    assert d["status"] == "running"
    assert d["progress"] == 0.0


async def test_workflow_execution_progress():
    from core.orchestration.unified_orchestrator import WorkflowExecution, WorkflowExecutionStatus
    wf = WorkflowExecution(
        execution_id="wfx-prog",
        workflow_name="wf",
        task_ids=["a", "b", "c", "d"],
        task_results={"a": {"status": "completed"}, "b": {"status": "completed"}},
    )
    assert wf.progress == 0.5


# --- Workflow Notification Integration ---


async def test_workflow_completion_sends_notification(orchestrator):
    """Completed workflow should emit an INFO notification."""
    from core.notifications.notification_service import NotificationService

    notif_service = NotificationService()
    orchestrator._notification_service = notif_service

    # Simulate a WORKFLOW_PROGRESS event with phase=completed
    await orchestrator._handle_workflow_progress({
        "execution_id": "wfx-abc",
        "workflow_name": "my-pipeline",
        "phase": "completed",
        "status": "completed",
    })

    notifications = await notif_service.list()
    assert len(notifications) == 1
    n = notifications[0]
    assert "my-pipeline" in n.title
    assert "completed" in n.title
    assert n.severity.value == "info"
    assert n.metadata["execution_id"] == "wfx-abc"


async def test_workflow_failure_sends_error_notification(orchestrator):
    """Failed workflow should emit an ERROR notification with error detail."""
    from core.notifications.notification_service import NotificationService

    notif_service = NotificationService()
    orchestrator._notification_service = notif_service

    await orchestrator._handle_workflow_progress({
        "execution_id": "wfx-fail",
        "workflow_name": "bad-wf",
        "phase": "failed",
        "error": "task T3 timed out",
    })

    notifications = await notif_service.list()
    assert len(notifications) == 1
    n = notifications[0]
    assert "bad-wf" in n.title
    assert "failed" in n.title
    assert n.severity.value == "error"
    assert "task T3 timed out" in n.body


async def test_workflow_progress_no_notification(orchestrator):
    """Incremental progress events should NOT trigger notifications."""
    from core.notifications.notification_service import NotificationService

    notif_service = NotificationService()
    orchestrator._notification_service = notif_service

    await orchestrator._handle_workflow_progress({
        "execution_id": "wfx-mid",
        "workflow_name": "wf",
        "phase": "progress",
        "percent": 42.0,
    })

    notifications = await notif_service.list()
    assert len(notifications) == 0


async def test_workflow_notification_skipped_when_no_service(orchestrator):
    """When notification_service is None, no error should occur."""
    orchestrator._notification_service = None

    # Should not raise
    await orchestrator._handle_workflow_progress({
        "execution_id": "wfx-x",
        "workflow_name": "wf",
        "phase": "completed",
    })
