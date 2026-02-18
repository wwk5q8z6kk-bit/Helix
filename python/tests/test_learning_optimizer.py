"""Tests for the learning optimizer (core/learning/learning_optimizer.py)."""

import json
import pytest

from core.learning.learning_optimizer import (
    AgentPerformance,
    LearningOptimizer,
    LearningSession,
    RewardType,
)


@pytest.fixture
def optimizer():
    return LearningOptimizer()


# --- Agent registration ---


def test_register_agent(optimizer):
    learning = optimizer.register_agent("a1", "Agent-1")
    assert "a1" in optimizer.agent_learning
    assert learning.agent_name == "Agent-1"
    assert learning.performance.tasks_completed == 0


# --- Sessions ---


def test_start_session(optimizer):
    session = optimizer.start_session("s1", "do stuff", ["a1"])
    assert "s1" in optimizer.sessions
    assert session.task_description == "do stuff"
    assert session.agents_involved == ["a1"]


def test_complete_session_success(optimizer):
    optimizer.register_agent("a1", "Agent-1")
    optimizer.start_session("s1", "task", ["a1"])
    result = optimizer.complete_session("s1", success=True, quality=90)

    assert result is not None
    assert result.success is True
    assert result.output_quality == 90
    # Should have at least SUCCESS + QUALITY rewards
    reward_types = [r.reward_type for r in result.reward_signals]
    assert RewardType.SUCCESS in reward_types
    assert RewardType.QUALITY in reward_types
    assert "s1" not in optimizer.sessions  # moved to history
    assert len(optimizer.session_history) == 1


def test_complete_session_failure(optimizer):
    optimizer.register_agent("a1", "Agent-1")
    optimizer.start_session("s1", "task", ["a1"])
    result = optimizer.complete_session("s1", success=False, quality=40)

    reward_types = [r.reward_type for r in result.reward_signals]
    assert RewardType.SUCCESS not in reward_types
    assert RewardType.QUALITY in reward_types


def test_complete_nonexistent_session(optimizer):
    assert optimizer.complete_session("nope", True, 100) is None


# --- AgentPerformance ---


def test_update_metrics():
    perf = AgentPerformance(agent_id="a1", agent_name="Agent-1")
    perf.update_metrics(duration_ms=200.0, quality=80.0)

    assert perf.tasks_completed == 1
    assert perf.total_duration_ms == 200.0
    assert perf.avg_duration_ms == 200.0
    # EMA: 0.1 * 80 + 0.9 * 0 = 8.0
    assert perf.quality_score == pytest.approx(8.0)
    assert perf.last_updated is not None

    perf.update_metrics(duration_ms=100.0, quality=90.0)
    assert perf.tasks_completed == 2
    assert perf.avg_duration_ms == 150.0


# --- Collaboration ---


def test_collaboration_recording(optimizer):
    optimizer.register_agent("a1", "Agent-1")
    optimizer.register_agent("a2", "Agent-2")
    optimizer.start_session("s1", "collab task", ["a1", "a2"])
    optimizer.complete_session(
        "s1", success=True, quality=80,
        collaboration_metrics={"effectiveness": 0.9},
    )

    a1 = optimizer.agent_learning["a1"]
    a2 = optimizer.agent_learning["a2"]
    assert "a2" in a1.collaboration_history
    assert "a1" in a2.collaboration_history


# --- Improvement potential ---


def test_improvement_potential(optimizer):
    learning = optimizer.register_agent("a1", "Agent-1")
    learning.performance.update_metrics(200, 80)
    potential = learning.calculate_improvement_potential()
    # success_rate = 100 (1/1), quality ~8.0 (EMA), avg = 54
    assert potential > 0


# --- Top agents ---


def test_top_agents(optimizer):
    l1 = optimizer.register_agent("a1", "Agent-1")
    l2 = optimizer.register_agent("a2", "Agent-2")
    l1.performance.update_metrics(100, 90)
    l2.performance.update_metrics(100, 50)

    top = optimizer.get_top_agents(limit=2)
    assert len(top) == 2
    # Agent-1 should rank higher (better quality)
    assert top[0].agent_id == "a1"


# --- Collaboration graph ---


def test_collaboration_graph(optimizer):
    optimizer.register_agent("a1", "Agent-1")
    optimizer.register_agent("a2", "Agent-2")
    optimizer.start_session("s1", "t", ["a1", "a2"])
    optimizer.complete_session("s1", True, 80, {"effectiveness": 0.8})

    graph = optimizer.get_collaboration_graph()
    assert "a1" in graph
    assert "a2" in graph["a1"]


# --- Statistics ---


def test_learning_statistics(optimizer):
    optimizer.register_agent("a1", "Agent-1")
    optimizer.start_session("s1", "t1", ["a1"])
    optimizer.complete_session("s1", True, 85)
    optimizer.start_session("s2", "t2", ["a1"])
    optimizer.complete_session("s2", False, 40)

    stats = optimizer.get_learning_statistics()
    assert stats["total_agents"] == 1
    assert stats["total_sessions"] == 2
    assert stats["successful_sessions"] == 1
    assert stats["success_rate"] == 50.0
    assert stats["learning_iterations"] == 2


# --- Export / Import ---


def test_export_import_state(optimizer):
    optimizer.register_agent("a1", "Agent-1")
    optimizer.start_session("s1", "task", ["a1"])
    optimizer.complete_session("s1", True, 90)

    exported = optimizer.export_learning_state()
    data = json.loads(exported)
    assert "agents" in data
    assert "a1" in data["agents"]

    assert optimizer.import_learning_state(exported) is True
    assert optimizer.import_learning_state("{bad json") is False


# --- Reset ---


def test_reset(optimizer):
    optimizer.register_agent("a1", "Agent-1")
    optimizer.start_session("s1", "task", ["a1"])
    optimizer.complete_session("s1", True, 90)

    optimizer.reset()
    assert len(optimizer.agent_learning) == 0
    assert len(optimizer.sessions) == 0
    assert len(optimizer.session_history) == 0
    assert optimizer.learning_iterations == 0
    assert len(optimizer.global_rewards) == 0
