"""Tests for the MCP server tool dispatch (core/mcp/server.py).

Verifies that call_tool() routes all 8 tools correctly and returns
proper error for unknown tools.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def _reset_container():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


# --- Tool listing ---


async def test_list_tools_returns_nine():
    """list_tools() should expose all 9 tool definitions."""
    from core.mcp.server import list_tools

    tools = await list_tools()
    assert len(tools) == 9
    names = {t.name for t in tools}
    assert names == {
        "helix_generate", "helix_search", "helix_orchestrate",
        "helix_swarm", "helix_review", "helix_reason", "helix_learn",
        "helix_schedule", "helix_notify",
    }


# --- Unknown tool ---


async def test_unknown_tool_returns_error():
    """Calling an undefined tool should return an error dict."""
    from core.mcp.server import call_tool

    result = await call_tool("helix_nonexistent", {})
    assert len(result) == 1
    data = json.loads(result[0].text)
    assert "error" in data
    assert "Unknown tool" in data["error"]


# --- helix_generate ---


async def test_generate_dispatches_to_router():
    """helix_generate should call the LLM router."""
    from core.mcp.server import call_tool

    mock_router = MagicMock()
    mock_router.select_provider = AsyncMock(return_value=MagicMock(value="claude"))
    mock_router.call_llm = AsyncMock(return_value=("Hello", {"provider": "claude", "tokens_used": 10, "cost": 0.0}))

    with patch("core.mcp.server.get_llm_router", new_callable=AsyncMock, return_value=mock_router):
        result = await call_tool("helix_generate", {"prompt": "Say hi"})
        data = json.loads(result[0].text)
        assert data["content"] == "Hello"
        mock_router.call_llm.assert_awaited_once()


# --- helix_search ---


async def test_search_dispatches_to_bridge():
    """helix_search should call bridge.semantic_search."""
    from core.mcp.server import call_tool, bridge

    bridge.semantic_search = AsyncMock(return_value=[{"id": "1", "score": 0.9}])
    result = await call_tool("helix_search", {"query": "test"})
    data = json.loads(result[0].text)
    assert data["count"] == 1
    bridge.semantic_search.assert_awaited_once()


# --- helix_orchestrate ---


async def test_orchestrate_dispatches_correctly():
    """helix_orchestrate should route via LLM router."""
    from core.mcp.server import call_tool

    mock_router = MagicMock()
    mock_provider = MagicMock(value="gpt")
    mock_router.select_provider = AsyncMock(return_value=mock_provider)
    mock_router.call_llm = AsyncMock(return_value=("Result", {"provider": "gpt"}))

    with patch("core.mcp.server.get_llm_router", new_callable=AsyncMock, return_value=mock_router):
        result = await call_tool("helix_orchestrate", {"task": "build an API"})
        data = json.loads(result[0].text)
        assert "result" in data


# --- helix_review ---


async def test_review_dispatches_correctly():
    """helix_review should call the LLM router with review prompt."""
    from core.mcp.server import call_tool

    mock_router = MagicMock()
    mock_router.select_provider = AsyncMock(return_value=MagicMock(value="claude"))
    mock_router.call_llm = AsyncMock(return_value=("No issues", {"provider": "claude"}))

    with patch("core.mcp.server.get_llm_router", new_callable=AsyncMock, return_value=mock_router):
        result = await call_tool("helix_review", {"code": "print('hi')", "language": "python"})
        data = json.loads(result[0].text)
        assert data["review"] == "No issues"


# --- helix_swarm ---


async def test_swarm_dispatches_to_orchestrator():
    """helix_swarm should call SwarmOrchestrator.route_and_execute."""
    from core.mcp.server import call_tool

    mock_result = MagicMock()
    mock_result.task_id = "t1"
    mock_result.swarm_used = "implementation"
    mock_result.task_category.value = "implementation"
    mock_result.quality_score = 0.9
    mock_result.routing_confidence = 0.85
    mock_result.swarm_result.success = True
    mock_result.swarm_result.output = "Done"

    mock_orch = MagicMock()
    mock_orch.route_and_execute = AsyncMock(return_value=mock_result)

    with patch("core.swarms.swarm_orchestrator.get_swarm_orchestrator", return_value=mock_orch):
        result = await call_tool("helix_swarm", {"task": "build API", "swarm_type": "implementation"})
        data = json.loads(result[0].text)
        assert data["status"] == "completed"
        assert data["swarm_used"] == "implementation"


# --- helix_reason ---


async def test_reason_dispatches_to_agentic_reasoner():
    """helix_reason should call AgenticReasoner.solve."""
    from core.mcp.server import call_tool

    mock_traj = MagicMock()
    mock_traj.steps = [
        MagicMock(content="Step 1", step_type=MagicMock(value="think"), confidence=0.9),
    ]
    mock_traj.final_answer = "The answer is 42"
    mock_traj.success = True

    mock_reward = MagicMock()
    mock_reward.final_score = 0.88

    mock_reasoner = MagicMock()
    mock_reasoner.solve = AsyncMock(return_value=(mock_traj, mock_reward))

    with patch("core.reasoning.agentic_reasoner.get_agentic_reasoner", return_value=mock_reasoner):
        result = await call_tool("helix_reason", {"goal": "What is 6*7?"})
        data = json.loads(result[0].text)
        assert data["conclusion"] == "The answer is 42"
        assert data["confidence"] == 0.88
        assert data["success"] is True
        assert len(data["steps"]) == 1


# --- helix_learn ---


async def test_learn_returns_statistics():
    """helix_learn should return learning statistics."""
    from core.mcp.server import call_tool

    with patch("core.learning.learning_optimizer.LearningOptimizer") as MockOpt:
        mock_opt = MockOpt.return_value
        mock_opt.get_learning_statistics.return_value = {
            "total_agents": 3,
            "learning_iterations": 10,
            "success_rate": 85.0,
            "average_quality": 78.5,
            "agent_statistics": [
                {"agent_id": "a1", "agent_name": "coder"},
                {"agent_id": "a2", "agent_name": "tester"},
            ],
        }

        result = await call_tool("helix_learn", {"query": "coder", "limit": 5})
        data = json.loads(result[0].text)
        assert data["total_agents"] == 3
        # "coder" matches by name filter
        assert len(data["patterns"]) >= 1


# --- helix_schedule ---


async def test_schedule_define():
    """helix_schedule define should delegate to bridge."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_define_workflow = AsyncMock(return_value={"name": "wf1", "tasks": 2})
    result = await call_tool("helix_schedule", {
        "action": "define", "name": "wf1",
        "tasks": [
            {"task_id": "a", "task_type": "code_generation"},
            {"task_id": "b", "task_type": "testing", "depends_on": ["a"]},
        ],
    })
    data = json.loads(result[0].text)
    assert data["name"] == "wf1"
    bridge.scheduler_define_workflow.assert_awaited_once()


async def test_schedule_define_missing_name():
    """define without name should return error."""
    from core.mcp.server import call_tool

    result = await call_tool("helix_schedule", {"action": "define"})
    data = json.loads(result[0].text)
    assert "error" in data


async def test_schedule_list():
    """helix_schedule list should delegate to bridge."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_list_workflows = AsyncMock(return_value=[{"name": "wf1"}, {"name": "wf2"}])
    result = await call_tool("helix_schedule", {"action": "list"})
    data = json.loads(result[0].text)
    assert len(data["workflows"]) == 2


async def test_schedule_get():
    """helix_schedule get should return workflow."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_get_workflow = AsyncMock(return_value={"name": "wf1", "tasks": []})
    result = await call_tool("helix_schedule", {"action": "get", "name": "wf1"})
    data = json.loads(result[0].text)
    assert data["name"] == "wf1"


async def test_schedule_get_not_found():
    """helix_schedule get missing workflow returns error."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_get_workflow = AsyncMock(return_value=None)
    result = await call_tool("helix_schedule", {"action": "get", "name": "nope"})
    data = json.loads(result[0].text)
    assert "error" in data


async def test_schedule_preview():
    """helix_schedule preview should delegate to bridge."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_preview_workflow = AsyncMock(return_value={"waves": 3, "feasible": True})
    result = await call_tool("helix_schedule", {"action": "preview", "name": "wf1"})
    data = json.loads(result[0].text)
    assert data["feasible"] is True


async def test_schedule_delete():
    """helix_schedule delete should delegate to bridge."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_delete_workflow = AsyncMock(return_value=True)
    result = await call_tool("helix_schedule", {"action": "delete", "name": "wf1"})
    data = json.loads(result[0].text)
    assert data["deleted"] is True


async def test_schedule_templates():
    """helix_schedule templates should list available templates."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_list_templates = AsyncMock(return_value=["code_review", "research"])
    result = await call_tool("helix_schedule", {"action": "templates"})
    data = json.loads(result[0].text)
    assert "code_review" in data["templates"]


async def test_schedule_stats():
    """helix_schedule stats should return scheduler stats."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_stats = AsyncMock(return_value={"total": 5, "ready": 2})
    result = await call_tool("helix_schedule", {"action": "stats"})
    data = json.loads(result[0].text)
    assert data["total"] == 5


async def test_schedule_waves():
    """helix_schedule waves should return execution waves."""
    from core.mcp.server import call_tool, bridge

    bridge.scheduler_execution_waves = AsyncMock(return_value=[["a"], ["b", "c"]])
    result = await call_tool("helix_schedule", {"action": "waves"})
    data = json.loads(result[0].text)
    assert len(data["waves"]) == 2


async def test_schedule_unknown_action():
    """helix_schedule with unknown action returns error."""
    from core.mcp.server import call_tool

    result = await call_tool("helix_schedule", {"action": "bogus"})
    data = json.loads(result[0].text)
    assert "error" in data
    assert "bogus" in data["error"]


async def test_schedule_run():
    """helix_schedule run should launch a workflow execution."""
    from core.mcp.server import call_tool

    mock_exec = MagicMock()
    mock_exec.to_dict.return_value = {
        "execution_id": "wfx-abc",
        "workflow_name": "wf1",
        "status": "running",
    }

    mock_orch = MagicMock()
    mock_orch.run_workflow = AsyncMock(return_value=mock_exec)

    with patch("core.di_container.get_container") as mock_gc:
        mock_gc.return_value.orchestrator = mock_orch
        result = await call_tool("helix_schedule", {"action": "run", "name": "wf1"})
        data = json.loads(result[0].text)
        assert data["execution_id"] == "wfx-abc"
        assert data["status"] == "running"


async def test_schedule_run_not_found():
    """helix_schedule run with missing workflow returns error."""
    from core.mcp.server import call_tool

    mock_orch = MagicMock()
    mock_orch.run_workflow = AsyncMock(side_effect=ValueError("Workflow 'nope' not found"))

    with patch("core.di_container.get_container") as mock_gc:
        mock_gc.return_value.orchestrator = mock_orch
        result = await call_tool("helix_schedule", {"action": "run", "name": "nope"})
        data = json.loads(result[0].text)
        assert "error" in data


async def test_schedule_executions():
    """helix_schedule executions should list runs."""
    from core.mcp.server import call_tool

    mock_orch = MagicMock()
    mock_orch.list_workflow_executions.return_value = [{"execution_id": "wfx-1"}]

    with patch("core.di_container.get_container") as mock_gc:
        mock_gc.return_value.orchestrator = mock_orch
        result = await call_tool("helix_schedule", {"action": "executions"})
        data = json.loads(result[0].text)
        assert len(data["executions"]) == 1


async def test_schedule_execution_status():
    """helix_schedule execution_status returns execution details."""
    from core.mcp.server import call_tool

    mock_orch = MagicMock()
    mock_orch.get_workflow_execution.return_value = {
        "execution_id": "wfx-abc", "status": "completed",
    }

    with patch("core.di_container.get_container") as mock_gc:
        mock_gc.return_value.orchestrator = mock_orch
        result = await call_tool("helix_schedule", {
            "action": "execution_status", "execution_id": "wfx-abc",
        })
        data = json.loads(result[0].text)
        assert data["status"] == "completed"


async def test_schedule_cancel_execution():
    """helix_schedule cancel_execution cancels a running workflow."""
    from core.mcp.server import call_tool

    mock_orch = MagicMock()
    mock_orch.cancel_workflow_execution = AsyncMock(return_value=True)

    with patch("core.di_container.get_container") as mock_gc:
        mock_gc.return_value.orchestrator = mock_orch
        result = await call_tool("helix_schedule", {
            "action": "cancel_execution", "execution_id": "wfx-abc",
        })
        data = json.loads(result[0].text)
        assert data["cancelled"] is True


# --- Error handling ---


async def test_tool_exception_returns_error():
    """If a handler throws, call_tool catches and returns error JSON."""
    from core.mcp.server import call_tool

    with patch("core.mcp.server.get_llm_router", new_callable=AsyncMock, side_effect=Exception("boom")):
        result = await call_tool("helix_generate", {"prompt": "hi"})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "boom" in data["error"]


# --- helix_notify ---


async def test_notify_list_empty():
    """helix_notify list on fresh service returns empty list."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService
        mock_gc.return_value.notification_service = NotificationService()
        result = await call_tool("helix_notify", {"action": "list"})
        data = json.loads(result[0].text)
        assert data["total"] == 0
        assert data["notifications"] == []


async def test_notify_list_with_notifications():
    """helix_notify list returns stored notifications."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService, Severity
        svc = NotificationService()
        n = svc.create_notification("Test", "body", Severity.INFO)
        await svc.send(n)
        mock_gc.return_value.notification_service = svc
        result = await call_tool("helix_notify", {"action": "list"})
        data = json.loads(result[0].text)
        assert data["total"] == 1
        assert data["notifications"][0]["title"] == "Test"


async def test_notify_list_filter_severity():
    """helix_notify list with severity filter."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService, Severity
        svc = NotificationService()
        await svc.send(svc.create_notification("Info", "i", Severity.INFO))
        await svc.send(svc.create_notification("Error", "e", Severity.ERROR))
        mock_gc.return_value.notification_service = svc
        result = await call_tool("helix_notify", {"action": "list", "severity": "error"})
        data = json.loads(result[0].text)
        assert len(data["notifications"]) == 1
        assert data["notifications"][0]["severity"] == "error"


async def test_notify_get():
    """helix_notify get returns a specific notification."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService, Severity
        svc = NotificationService()
        n = svc.create_notification("Find me", "body", Severity.WARNING)
        await svc.send(n)
        mock_gc.return_value.notification_service = svc
        result = await call_tool("helix_notify", {"action": "get", "notification_id": n.id})
        data = json.loads(result[0].text)
        assert data["title"] == "Find me"


async def test_notify_get_not_found():
    """helix_notify get with missing ID returns error."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService
        mock_gc.return_value.notification_service = NotificationService()
        result = await call_tool("helix_notify", {"action": "get", "notification_id": "bogus"})
        data = json.loads(result[0].text)
        assert "error" in data


async def test_notify_mark_read():
    """helix_notify mark_read marks a notification as read."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService, Severity
        svc = NotificationService()
        n = svc.create_notification("Read me", "body", Severity.INFO)
        await svc.send(n)
        mock_gc.return_value.notification_service = svc
        result = await call_tool("helix_notify", {"action": "mark_read", "notification_id": n.id})
        data = json.loads(result[0].text)
        assert data["marked_read"] is True


async def test_notify_unread_count():
    """helix_notify unread_count returns counts."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService, Severity
        svc = NotificationService()
        await svc.send(svc.create_notification("A", "a", Severity.INFO))
        await svc.send(svc.create_notification("B", "b", Severity.ERROR))
        await svc.mark_read((await svc.list())[0].id)
        mock_gc.return_value.notification_service = svc
        result = await call_tool("helix_notify", {"action": "unread_count"})
        data = json.loads(result[0].text)
        assert data["total"] == 2
        assert data["unread"] == 1


async def test_notify_unknown_action():
    """helix_notify with unknown action returns error."""
    from core.mcp.server import call_tool

    with patch("core.di_container.get_container") as mock_gc:
        from core.notifications.notification_service import NotificationService
        mock_gc.return_value.notification_service = NotificationService()
        result = await call_tool("helix_notify", {"action": "bogus"})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "bogus" in data["error"]
