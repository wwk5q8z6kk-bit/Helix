"""Tests for the MCP server tool dispatch (core/mcp/server.py).

Verifies that call_tool() routes all 7 tools correctly and returns
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


async def test_list_tools_returns_seven():
    """list_tools() should expose all 7 tool definitions."""
    from core.mcp.server import list_tools

    tools = await list_tools()
    assert len(tools) == 7
    names = {t.name for t in tools}
    assert names == {
        "helix_generate", "helix_search", "helix_orchestrate",
        "helix_swarm", "helix_review", "helix_reason", "helix_learn",
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


# --- Error handling ---


async def test_tool_exception_returns_error():
    """If a handler throws, call_tool catches and returns error JSON."""
    from core.mcp.server import call_tool

    with patch("core.mcp.server.get_llm_router", new_callable=AsyncMock, side_effect=Exception("boom")):
        result = await call_tool("helix_generate", {"prompt": "hi"})
        data = json.loads(result[0].text)
        assert "error" in data
        assert "boom" in data["error"]
