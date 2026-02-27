"""Multi-agent learning optimizer with reinforcement learning from collaboration."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class RewardType(str, Enum):
    """Types of rewards for agent learning."""
    SUCCESS = "success"  # Task completed successfully
    EFFICIENCY = "efficiency"  # Task completed with good metrics
    COLLABORATION = "collaboration"  # Good teamwork
    INNOVATION = "innovation"  # Novel approach
    QUALITY = "quality"  # High quality output


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    agent_id: str
    agent_name: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    quality_score: float = 0.0  # 0-100
    collaboration_score: float = 0.0  # 0-100
    innovation_score: float = 0.0  # 0-100
    total_reward: float = 0.0
    learning_iterations: int = 0
    last_updated: Optional[datetime] = None

    def update_metrics(self, duration_ms: float, quality: float) -> None:
        """Update performance metrics."""
        self.tasks_completed += 1
        self.total_duration_ms += duration_ms

        if self.tasks_completed > 0:
            self.avg_duration_ms = self.total_duration_ms / self.tasks_completed

        # Exponential moving average for quality
        alpha = 0.1
        self.quality_score = (
            alpha * quality + (1 - alpha) * self.quality_score
        )

        self.last_updated = datetime.now(tz=timezone.utc)

    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return (self.tasks_completed / total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "avg_duration_ms": self.avg_duration_ms,
            "quality_score": self.quality_score,
            "collaboration_score": self.collaboration_score,
            "innovation_score": self.innovation_score,
            "total_reward": self.total_reward,
            "success_rate": self.success_rate(),
        }


@dataclass
class RewardSignal:
    """Reward signal for agent learning."""
    agent_id: str
    reward_type: RewardType
    value: float  # Reward magnitude
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "reward_type": self.reward_type.value,
            "value": self.value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class LearningSession:
    """A learning session for agents to improve from collaboration."""
    session_id: str
    task_description: str
    agents_involved: List[str]
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    success: bool = False
    output_quality: float = 0.0  # 0-100
    collaboration_metrics: Dict[str, float] = field(default_factory=dict)
    reward_signals: List[RewardSignal] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)

    def complete(
        self,
        success: bool,
        quality: float,
        collaboration_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Mark session as complete."""
        self.end_time = datetime.now(tz=timezone.utc)
        self.success = success
        self.output_quality = quality
        self.collaboration_metrics = collaboration_metrics or {}

        if self.start_time and self.end_time:
            self.duration_ms = (
                (self.end_time - self.start_time).total_seconds() * 1000
            )

    def add_reward(
        self,
        agent_id: str,
        reward_type: RewardType,
        value: float,
        reason: str,
    ) -> None:
        """Add reward signal for an agent."""
        signal = RewardSignal(
            agent_id=agent_id,
            reward_type=reward_type,
            value=value,
            reason=reason,
        )
        self.reward_signals.append(signal)

    def add_insight(self, insight: str) -> None:
        """Add learning insight."""
        self.insights.append(insight)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "task_description": self.task_description,
            "agents_involved": self.agents_involved,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "output_quality": self.output_quality,
            "collaboration_metrics": self.collaboration_metrics,
            "reward_count": len(self.reward_signals),
            "insights_count": len(self.insights),
        }


class AgentLearning:
    """Learning state for an individual agent."""

    def __init__(self, agent_id: str, agent_name: str):
        """Initialize agent learning."""
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.performance = AgentPerformance(agent_id, agent_name)
        self.reward_history: List[RewardSignal] = []
        self.learned_patterns: Dict[str, float] = defaultdict(float)
        self.collaboration_history: Dict[str, int] = defaultdict(int)
        self.last_improvement_session: Optional[datetime] = None

    def add_reward(self, signal: RewardSignal) -> None:
        """Add reward signal."""
        self.reward_history.append(signal)
        self.performance.total_reward += signal.value

        # Update learned patterns
        pattern_key = f"{signal.reward_type.value}_pattern"
        self.learned_patterns[pattern_key] += signal.value

    def record_collaboration(self, partner_id: str) -> None:
        """Record collaboration with another agent."""
        self.collaboration_history[partner_id] += 1

    def calculate_improvement_potential(self) -> float:
        """Calculate improvement potential (0-100)."""
        success_rate = self.performance.success_rate()
        quality = self.performance.quality_score

        # Consider both success and quality
        potential = (success_rate + quality) / 2

        # Adjust based on collaboration effectiveness
        if self.collaboration_history:
            col_score = (
                sum(self.collaboration_history.values())
                / len(self.collaboration_history)
            )
            potential = (potential + col_score) / 2

        return potential

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "performance": self.performance.to_dict(),
            "reward_count": len(self.reward_history),
            "improvement_potential": self.calculate_improvement_potential(),
            "collaboration_partners": len(self.collaboration_history),
            "learned_patterns": dict(self.learned_patterns),
        }


class LearningOptimizer:
    """Orchestrator for multi-agent learning and reinforcement."""

    def __init__(self):
        """Initialize learning optimizer."""
        self.agent_learning: Dict[str, AgentLearning] = {}
        self.sessions: Dict[str, LearningSession] = {}
        self.session_history: List[LearningSession] = []
        self.learning_iterations: int = 0
        self.global_rewards: List[RewardSignal] = []

        logger.info("Initialized LearningOptimizer")

    def register_agent(self, agent_id: str, agent_name: str) -> AgentLearning:
        """Register an agent for learning."""
        learning = AgentLearning(agent_id, agent_name)
        self.agent_learning[agent_id] = learning
        logger.info(f"Registered agent {agent_name} ({agent_id}) for learning")
        return learning

    def start_session(
        self,
        session_id: str,
        task_description: str,
        agents: List[str],
    ) -> LearningSession:
        """Start a new learning session."""
        session = LearningSession(
            session_id=session_id,
            task_description=task_description,
            agents_involved=agents,
            start_time=datetime.now(tz=timezone.utc),
        )
        self.sessions[session_id] = session
        logger.info(
            f"Started learning session {session_id} "
            f"with {len(agents)} agents"
        )
        return session

    def complete_session(
        self,
        session_id: str,
        success: bool,
        quality: float,
        collaboration_metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[LearningSession]:
        """Complete a learning session and generate rewards."""
        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return None

        session.complete(success, quality, collaboration_metrics)
        self._generate_rewards(session)
        self._extract_insights(session)

        self.session_history.append(session)
        del self.sessions[session_id]

        self.learning_iterations += 1
        logger.info(
            f"Completed learning session {session_id} "
            f"(success: {success}, quality: {quality:.1f})"
        )

        return session

    def _generate_rewards(self, session: LearningSession) -> None:
        """Generate reward signals based on session outcome."""
        # Success reward
        if session.success:
            for agent_id in session.agents_involved:
                reward_value = 10.0
                session.add_reward(
                    agent_id,
                    RewardType.SUCCESS,
                    reward_value,
                    f"Task completed successfully",
                )

        # Quality reward
        quality_multiplier = session.output_quality / 100.0
        for agent_id in session.agents_involved:
            reward_value = 5.0 * quality_multiplier
            session.add_reward(
                agent_id,
                RewardType.QUALITY,
                reward_value,
                f"High quality output (score: {session.output_quality:.1f})",
            )

        # Collaboration reward
        if len(session.agents_involved) > 1:
            col_score = session.collaboration_metrics.get(
                "effectiveness", 0.5
            )
            for agent_id in session.agents_involved:
                reward_value = 5.0 * col_score
                session.add_reward(
                    agent_id,
                    RewardType.COLLABORATION,
                    reward_value,
                    "Effective team collaboration",
                )

                # Record collaborations
                for partner_id in session.agents_involved:
                    if partner_id != agent_id:
                        learning = self.agent_learning.get(agent_id)
                        if learning:
                            learning.record_collaboration(partner_id)

        # Efficiency reward
        if session.duration_ms < 5000:  # Fast execution
            for agent_id in session.agents_involved:
                reward_value = 3.0
                session.add_reward(
                    agent_id,
                    RewardType.EFFICIENCY,
                    reward_value,
                    f"Efficient execution ({session.duration_ms:.0f}ms)",
                )

        # Apply rewards to agents
        for signal in session.reward_signals:
            learning = self.agent_learning.get(signal.agent_id)
            if learning:
                learning.add_reward(signal)
                self.global_rewards.append(signal)

    def _extract_insights(self, session: LearningSession) -> None:
        """Extract learning insights from session."""
        insights = []

        # Quality insight
        if session.output_quality > 80:
            insights.append(
                f"High quality output achieved ({session.output_quality:.1f}%)"
            )

        # Efficiency insight
        if session.duration_ms < 3000:
            insights.append(
                f"Task completed efficiently in {session.duration_ms:.0f}ms"
            )

        # Collaboration insight
        if len(session.agents_involved) > 1:
            col_score = session.collaboration_metrics.get(
                "effectiveness", 0.0
            )
            if col_score > 0.7:
                insights.append(
                    f"Strong team collaboration (effectiveness: {col_score:.1f})"
                )

        for insight in insights:
            session.add_insight(insight)
            logger.info(f"Insight: {insight}")

    def get_agent_learning(self, agent_id: str) -> Optional[AgentLearning]:
        """Get learning state for an agent."""
        return self.agent_learning.get(agent_id)

    def get_top_agents(self, limit: int = 5) -> List[AgentLearning]:
        """Get top performing agents by improvement potential."""
        agents = list(self.agent_learning.values())
        agents.sort(
            key=lambda a: a.calculate_improvement_potential(),
            reverse=True,
        )
        return agents[:limit]

    def get_collaboration_graph(
        self,
    ) -> Dict[str, Dict[str, int]]:
        """Get collaboration graph between agents."""
        graph: Dict[str, Dict[str, int]] = {}

        for agent_id, learning in self.agent_learning.items():
            graph[agent_id] = dict(learning.collaboration_history)

        return graph

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get overall learning statistics."""
        total_agents = len(self.agent_learning)
        total_sessions = len(self.session_history)
        successful_sessions = sum(
            1 for s in self.session_history if s.success
        )
        total_rewards_distributed = sum(
            signal.value for signal in self.global_rewards
        )
        avg_quality = (
            sum(s.output_quality for s in self.session_history)
            / total_sessions
            if total_sessions > 0
            else 0.0
        )

        agent_stats = [
            learning.to_dict() for learning in self.agent_learning.values()
        ]
        agent_stats.sort(
            key=lambda a: a["improvement_potential"], reverse=True
        )

        return {
            "total_agents": total_agents,
            "learning_iterations": self.learning_iterations,
            "total_sessions": total_sessions,
            "successful_sessions": successful_sessions,
            "success_rate": (
                (successful_sessions / total_sessions * 100)
                if total_sessions > 0
                else 0.0
            ),
            "average_quality": avg_quality,
            "total_rewards_distributed": total_rewards_distributed,
            "avg_reward_per_agent": (
                total_rewards_distributed / total_agents
                if total_agents > 0
                else 0.0
            ),
            "agent_statistics": agent_stats,
        }

    def export_learning_state(self) -> str:
        """Export learning state as JSON."""
        state = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "statistics": self.get_learning_statistics(),
            "agents": {
                agent_id: learning.to_dict()
                for agent_id, learning in self.agent_learning.items()
            },
            "collaboration_graph": self.get_collaboration_graph(),
            "session_history": [s.to_dict() for s in self.session_history],
        }
        return json.dumps(state, indent=2)

    def import_learning_state(self, json_str: str) -> bool:
        """Import learning state from JSON."""
        try:
            state = json.loads(json_str)
            logger.info(
                f"Imported learning state for "
                f"{len(state.get('agents', {}))} agents"
            )
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Failed to import learning state: {e}")
            return False

    def reset(self) -> None:
        """Reset learning optimizer."""
        self.agent_learning.clear()
        self.sessions.clear()
        self.session_history.clear()
        self.learning_iterations = 0
        self.global_rewards.clear()
        logger.info("Reset LearningOptimizer")
