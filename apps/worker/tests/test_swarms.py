"""Tests for the SwarmOrchestrator task routing and execution."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.exceptions_unified import AgentError
from core.swarms.swarm_orchestrator import (
    SwarmOrchestrator,
    TaskCategory,
    RoutingDecision,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset global singleton between tests."""
    import core.swarms.swarm_orchestrator as mod
    mod._orchestrator = None
    yield
    mod._orchestrator = None


@pytest.fixture
def orchestrator():
    """Create a SwarmOrchestrator with mocked brain components."""
    with (
        patch("core.swarms.swarm_orchestrator.get_trajectory_tracker"),
        patch("core.swarms.swarm_orchestrator.get_process_reward_model"),
        patch("core.swarms.swarm_orchestrator.get_agentic_reasoner"),
        patch("core.swarms.swarm_orchestrator.get_shared_state_manager"),
        patch("core.swarms.swarm_orchestrator.get_a2a_protocol"),
        patch("core.swarms.swarm_orchestrator.get_knowledge_pool"),
        patch("core.swarms.swarm_orchestrator.get_strategy_recommender") as mock_rec,
        patch("core.swarms.swarm_orchestrator.get_feedback_collector"),
    ):
        mock_rec.return_value.recommend_strategy.side_effect = AgentError("no history")
        orch = SwarmOrchestrator()
        return orch


# --- Task Classification ---


async def test_categorize_debugging_task(orchestrator):
    """Debugging keywords should route to debugging swarm."""
    decision = await orchestrator._classify_and_route("debug the login crash", {})
    assert decision.task_category == TaskCategory.DEBUGGING


async def test_categorize_testing_task(orchestrator):
    """Testing keywords route to testing swarm."""
    decision = await orchestrator._classify_and_route("write unit tests for auth module", {})
    assert decision.task_category == TaskCategory.TESTING


async def test_categorize_architecture_task(orchestrator):
    """Architecture keywords route correctly."""
    decision = await orchestrator._classify_and_route("design system architecture for microservices", {})
    assert decision.task_category == TaskCategory.ARCHITECTURE


async def test_categorize_requirements_task(orchestrator):
    """Requirements phrases route to requirements swarm."""
    decision = await orchestrator._classify_and_route("gather requirements and user stories from stakeholders", {})
    assert decision.task_category == TaskCategory.REQUIREMENTS


async def test_categorize_code_review_task(orchestrator):
    """Code review phrases route correctly."""
    decision = await orchestrator._classify_and_route("code review the authentication module", {})
    assert decision.task_category == TaskCategory.CODE_REVIEW


async def test_categorize_deployment_task(orchestrator):
    """Deployment keywords route correctly."""
    decision = await orchestrator._classify_and_route("deploy the application to production with CI/CD", {})
    assert decision.task_category == TaskCategory.DEPLOYMENT


async def test_categorize_unknown_defaults_to_implementation(orchestrator):
    """Ambiguous tasks default to implementation."""
    decision = await orchestrator._classify_and_route("do something with the thing", {})
    assert decision.task_category == TaskCategory.IMPLEMENTATION


async def test_swarm_name_mapping(orchestrator):
    """Each category maps to the correct swarm name."""
    assert orchestrator._get_swarm_name_for_category(TaskCategory.DEBUGGING) == "debugging"
    assert orchestrator._get_swarm_name_for_category(TaskCategory.TESTING) == "testing"
    assert orchestrator._get_swarm_name_for_category(TaskCategory.ARCHITECTURE) == "architecture"
    assert orchestrator._get_swarm_name_for_category(TaskCategory.REQUIREMENTS) == "requirements"
    assert orchestrator._get_swarm_name_for_category(TaskCategory.UNKNOWN) == "implementation"


async def test_routing_confidence_above_threshold(orchestrator):
    """Strong keyword matches produce confidence >= 0.4."""
    decision = await orchestrator._classify_and_route("debug and troubleshoot the error", {})
    assert decision.confidence >= 0.4


def test_get_available_swarms(orchestrator):
    """Available swarms list should contain all 7 swarm types."""
    swarms = orchestrator.get_available_swarms()
    assert len(swarms) == 7
    assert "implementation" in swarms
    assert "debugging" in swarms


def test_get_routing_stats(orchestrator):
    """Routing stats returns valid structure."""
    stats = orchestrator.get_routing_stats()
    assert "learned_patterns" in stats
    assert "available_swarms" in stats
    assert stats["available_swarms"] == 7
