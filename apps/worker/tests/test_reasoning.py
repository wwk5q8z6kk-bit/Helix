"""Tests for the Agentic Reasoning Engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.reasoning.agentic_reasoner import (
    AgenticReasoner,
    AgentAction,
    PlanningStrategy,
    Goal,
    Plan,
    ToolDefinition,
    ReflectionResult,
)
from core.reasoning.trajectory_tracker import StepType


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset global singleton between tests."""
    import core.reasoning.agentic_reasoner as mod
    mod._agentic_reasoner_instance = None
    yield
    mod._agentic_reasoner_instance = None


@pytest.fixture
def mock_tracker():
    tracker = MagicMock()
    traj = MagicMock()
    traj.trajectory_id = "traj-001"
    traj.steps = []
    traj.final_answer = None
    traj.success = False

    def add_step(content, step_type, confidence, metadata=None):
        step = MagicMock()
        step.step_id = f"step-{len(traj.steps)}"
        step.content = content
        step.step_type = step_type
        step.confidence = confidence
        step.tool_result = None
        step.verification_result = None
        traj.steps.append(step)
        return step

    traj.add_step = add_step
    traj.complete = MagicMock()
    tracker.create_trajectory.return_value = traj
    tracker.save_step = MagicMock()
    tracker.update_trajectory = MagicMock()
    return tracker


@pytest.fixture
def mock_prm():
    prm = MagicMock()
    step_reward = MagicMock()
    step_reward.overall_score = 0.85
    prm.score_step = AsyncMock(return_value=step_reward)

    traj_reward = MagicMock()
    traj_reward.final_score = 0.82
    traj_reward.temporal_coherence = 0.80
    traj_reward.efficiency_bonus = 0.05
    prm.score_trajectory = AsyncMock(return_value=traj_reward)
    return prm


@pytest.fixture
def reasoner(mock_tracker, mock_prm):
    return AgenticReasoner(
        tracker=mock_tracker,
        resilient_reasoner=MagicMock(),
        prm=mock_prm,
        max_steps=10,
        reflection_frequency=50,  # High to avoid reflection in tests
    )


# --- Goal Decomposition ---


async def test_decompose_simple_goal(reasoner, mock_tracker):
    """Short goals should not be decomposed into subgoals."""
    traj = mock_tracker.create_trajectory.return_value
    goal = await reasoner._decompose_goal("Fix the bug", traj, {})
    assert isinstance(goal, Goal)
    assert goal.description == "Fix the bug"
    assert len(goal.subgoals) == 0  # Short problem, no decomposition


async def test_decompose_complex_goal(reasoner, mock_tracker):
    """Long goals should be decomposed into subgoals."""
    traj = mock_tracker.create_trajectory.return_value
    long_problem = "Design and implement a distributed caching layer with TTL expiration and LRU eviction strategy"
    goal = await reasoner._decompose_goal(long_problem, traj, {})
    assert len(goal.subgoals) > 0


# --- Planning ---


async def test_forward_planning(reasoner, mock_tracker):
    """Forward planning produces sequential steps."""
    traj = mock_tracker.create_trajectory.return_value
    goal = Goal(goal_id="g1", description="Test goal")
    plan = await reasoner._create_plan(goal, PlanningStrategy.FORWARD, traj, {})
    assert isinstance(plan, Plan)
    assert plan.strategy == PlanningStrategy.FORWARD
    assert len(plan.steps) > 0


async def test_backward_planning(reasoner, mock_tracker):
    """Backward planning creates steps in correct execution order."""
    traj = mock_tracker.create_trajectory.return_value
    goal = Goal(goal_id="g1", description="Test goal")
    plan = await reasoner._create_plan(goal, PlanningStrategy.BACKWARD, traj, {})
    assert plan.strategy == PlanningStrategy.BACKWARD
    assert len(plan.steps) > 0


async def test_hierarchical_planning_with_subgoals(reasoner, mock_tracker):
    """Hierarchical planning includes decompose + per-subgoal steps."""
    traj = mock_tracker.create_trajectory.return_value
    sub = Goal(goal_id="sg1", description="Sub-task")
    reasoner.goals["sg1"] = sub
    goal = Goal(goal_id="g1", description="Main", subgoals=["sg1"])
    reasoner.goals["g1"] = goal

    plan = await reasoner._create_plan(goal, PlanningStrategy.HIERARCHICAL, traj, {})
    assert plan.strategy == PlanningStrategy.HIERARCHICAL
    # Should have decompose + (plan+execute+verify per subgoal) + synthesize
    assert len(plan.steps) >= 4


# --- Execution ---


async def test_solve_returns_trajectory_and_reward(reasoner):
    """solve() should return a (trajectory, reward) tuple."""
    trajectory, reward = await reasoner.solve("Simple problem")
    assert trajectory is not None
    assert reward is not None
    assert reward.final_score == 0.82


async def test_solve_records_steps(reasoner, mock_tracker):
    """solve() should record steps in the trajectory."""
    traj = mock_tracker.create_trajectory.return_value
    await reasoner.solve("Build a REST API")
    # Should have at least decompose + plan + some execution steps
    assert len(traj.steps) >= 2


async def test_solve_completes_trajectory(reasoner, mock_tracker):
    """solve() should call trajectory.complete()."""
    traj = mock_tracker.create_trajectory.return_value
    await reasoner.solve("Fix the issue")
    traj.complete.assert_called_once()


# --- Tool Registration ---


def test_register_tool(reasoner):
    """Tools can be registered for agent use."""
    tool = ToolDefinition(
        name="search",
        description="Search the web",
        parameters={"query": "string"},
        execute=AsyncMock(),
    )
    reasoner.register_tool(tool)
    assert "search" in reasoner.tools


# --- Reflection ---


async def test_reflection_should_continue(reasoner, mock_tracker):
    """Reflection with good results should continue."""
    traj = mock_tracker.create_trajectory.return_value
    recent_results = [
        {"action": "think", "reward": 0.8},
        {"action": "tool", "reward": 0.9},
    ]
    reflection = await reasoner._reflect(traj, recent_results, {})
    assert isinstance(reflection, ReflectionResult)
    assert reflection.should_continue is True


async def test_reflection_stops_on_low_quality(reasoner, mock_tracker):
    """Reflection with poor results should stop."""
    traj = mock_tracker.create_trajectory.return_value
    # Simulate lots of existing steps so should_continue check fails
    traj.steps = [MagicMock() for _ in range(55)]
    recent_results = [
        {"action": "think", "reward": 0.1},
        {"action": "tool", "reward": 0.2},
    ]
    reflection = await reasoner._reflect(traj, recent_results, {})
    assert reflection.should_continue is False
