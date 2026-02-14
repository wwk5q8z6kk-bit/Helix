"""Reinforcement learning integration for multi-agent collaboration."""

import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
import random

logger = logging.getLogger(__name__)


class PolicyType(str, Enum):
    """Types of RL policies."""
    EPSILON_GREEDY = "epsilon_greedy"
    SOFTMAX = "softmax"
    UCB = "ucb"  # Upper Confidence Bound
    THOMPSON_SAMPLING = "thompson_sampling"


@dataclass
class QValue:
    """Q-value for state-action pair in reinforcement learning."""
    state: str
    action: str
    value: float = 0.0
    visit_count: int = 0
    last_updated: Optional[datetime] = None

    def update(self, new_value: float) -> None:
        """Update Q-value with new experience."""
        self.visit_count += 1
        # Incremental Q-value update
        alpha = 1.0 / self.visit_count  # Learning rate
        self.value = self.value + alpha * (new_value - self.value)
        self.last_updated = datetime.now(tz=timezone.utc)


@dataclass
class Experience:
    """Single experience for learning: (state, action, reward, next_state)."""
    state: str
    action: str
    reward: float
    next_state: str
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "state": self.state,
            "action": self.action,
            "reward": self.reward,
            "next_state": self.next_state,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PolicyDecision:
    """Decision made by a policy."""
    agent_id: str
    state: str
    action: str
    confidence: float  # 0-1, how confident in this action
    policy_type: PolicyType
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_id": self.agent_id,
            "state": self.state,
            "action": self.action,
            "confidence": self.confidence,
            "policy_type": self.policy_type.value,
            "num_alternatives": len(self.alternatives),
        }


class ReinforcementPolicy:
    """Base class for RL policies."""

    def __init__(self, policy_type: PolicyType, learning_rate: float = 0.1):
        """Initialize policy."""
        self.policy_type = policy_type
        self.learning_rate = learning_rate
        self.q_values: Dict[Tuple[str, str], QValue] = {}
        self.state_visits: Dict[str, int] = defaultdict(int)
        self.action_space: List[str] = []

    def set_action_space(self, actions: List[str]) -> None:
        """Set possible actions."""
        self.action_space = actions

    def select_action(
        self,
        state: str,
        agent_id: str,
        epsilon: float = 0.1,
    ) -> PolicyDecision:
        """Select action based on policy."""
        if self.policy_type == PolicyType.EPSILON_GREEDY:
            return self._epsilon_greedy(state, agent_id, epsilon)
        elif self.policy_type == PolicyType.SOFTMAX:
            return self._softmax(state, agent_id)
        elif self.policy_type == PolicyType.UCB:
            return self._ucb(state, agent_id)
        else:  # THOMPSON_SAMPLING
            return self._thompson_sampling(state, agent_id)

    def _epsilon_greedy(
        self,
        state: str,
        agent_id: str,
        epsilon: float,
    ) -> PolicyDecision:
        """Epsilon-greedy policy: explore with prob epsilon, exploit otherwise."""
        self.state_visits[state] += 1

        # Get Q-values for all actions in this state
        q_values_for_state = [
            self.q_values.get((state, action), QValue(state, action)).value
            for action in self.action_space
        ]

        if random.random() < epsilon:
            # Explore: random action
            action = random.choice(self.action_space)
            alternatives = list(zip(self.action_space, q_values_for_state))
            confidence = 1.0 / len(self.action_space)
        else:
            # Exploit: best action
            max_q = max(q_values_for_state) if q_values_for_state else 0.0
            best_actions = [
                action
                for action, q in zip(self.action_space, q_values_for_state)
                if q == max_q
            ]
            action = random.choice(best_actions)
            alternatives = list(zip(self.action_space, q_values_for_state))
            confidence = 0.9  # High confidence when exploiting

        return PolicyDecision(
            agent_id=agent_id,
            state=state,
            action=action,
            confidence=confidence,
            policy_type=self.policy_type,
            alternatives=alternatives,
        )

    def _softmax(
        self,
        state: str,
        agent_id: str,
        temperature: float = 1.0,
    ) -> PolicyDecision:
        """Softmax policy: probabilistic action selection."""
        self.state_visits[state] += 1

        # Get Q-values
        q_values_for_state = [
            self.q_values.get((state, action), QValue(state, action)).value
            for action in self.action_space
        ]

        # Softmax probabilities
        exp_values = [
            math.exp(q / temperature) for q in q_values_for_state
        ]
        total = sum(exp_values)
        probabilities = [ev / total for ev in exp_values]

        # Select action by probability
        action = random.choices(
            self.action_space, weights=probabilities, k=1
        )[0]
        max_prob = max(probabilities) if probabilities else 0.0

        alternatives = list(zip(self.action_space, probabilities))

        return PolicyDecision(
            agent_id=agent_id,
            state=state,
            action=action,
            confidence=max_prob,
            policy_type=self.policy_type,
            alternatives=alternatives,
        )

    def _ucb(self, state: str, agent_id: str) -> PolicyDecision:
        """Upper Confidence Bound policy."""
        self.state_visits[state] += 1

        # UCB1: Q(a) + C * sqrt(ln(N) / n(a))
        ucb_values = []
        for action in self.action_space:
            q_val = self.q_values.get((state, action), QValue(state, action))
            n_total = self.state_visits[state]
            n_action = q_val.visit_count + 1

            c = 1.41  # Exploration constant
            ucb = q_val.value + c * ((n_total / n_action) ** 0.5)
            ucb_values.append(ucb)

        # Select action with highest UCB
        action = self.action_space[ucb_values.index(max(ucb_values))]
        confidence = max(ucb_values) / (max(ucb_values) + 1)

        alternatives = list(zip(self.action_space, ucb_values))

        return PolicyDecision(
            agent_id=agent_id,
            state=state,
            action=action,
            confidence=confidence,
            policy_type=self.policy_type,
            alternatives=alternatives,
        )

    def _thompson_sampling(
        self,
        state: str,
        agent_id: str,
    ) -> PolicyDecision:
        """Thompson sampling policy (simplified)."""
        self.state_visits[state] += 1

        # Simplified Thompson sampling with beta-like distribution
        sampled_values = []
        for action in self.action_space:
            q_val = self.q_values.get((state, action), QValue(state, action))
            # Sample from posterior (simplified)
            variance = 1.0 / (q_val.visit_count + 1)
            sampled = q_val.value + random.gauss(0, variance ** 0.5)
            sampled_values.append(sampled)

        # Select action with highest sampled value
        action = self.action_space[sampled_values.index(max(sampled_values))]
        confidence = 0.7  # Moderate confidence

        alternatives = list(zip(self.action_space, sampled_values))

        return PolicyDecision(
            agent_id=agent_id,
            state=state,
            action=action,
            confidence=confidence,
            policy_type=self.policy_type,
            alternatives=alternatives,
        )

    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        next_action: Optional[str] = None,
    ) -> None:
        """Update Q-value using Q-learning or SARSA."""
        key = (state, action)

        if key not in self.q_values:
            self.q_values[key] = QValue(state, action)

        # Get next Q-value
        if next_action:
            # SARSA: use specific next action
            next_key = (next_state, next_action)
            next_q = self.q_values.get(next_key, QValue(next_state, next_action)).value
        else:
            # Q-learning: use max Q for next state
            next_q_values = [
                self.q_values.get((next_state, a), QValue(next_state, a)).value
                for a in self.action_space
            ]
            next_q = max(next_q_values) if next_q_values else 0.0

        # Bellman equation
        gamma = 0.99  # Discount factor
        target = reward + gamma * next_q
        self.q_values[key].update(target)


class RLIntegration:
    """Reinforcement learning integration for agent collaboration."""

    def __init__(self, policy_type: PolicyType = PolicyType.EPSILON_GREEDY):
        """Initialize RL integration."""
        self.policy = ReinforcementPolicy(policy_type)
        self.experiences: List[Experience] = []
        self.agent_policies: Dict[str, ReinforcementPolicy] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.99
        self.epsilon = 0.1  # For epsilon-greedy

        logger.info(f"Initialized RLIntegration with {policy_type.value} policy")

    def register_agent_policy(
        self,
        agent_id: str,
        policy_type: PolicyType,
    ) -> ReinforcementPolicy:
        """Register an RL policy for an agent."""
        policy = ReinforcementPolicy(policy_type, self.learning_rate)
        self.agent_policies[agent_id] = policy
        return policy

    def record_experience(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        agent_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Experience:
        """Record an experience for learning."""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        self.experiences.append(experience)

        # Update agent policy
        if agent_id in self.agent_policies:
            policy = self.agent_policies[agent_id]
            policy.update_q_value(state, action, reward, next_state)

        return experience

    def get_agent_action(
        self,
        agent_id: str,
        state: str,
        action_space: List[str],
    ) -> PolicyDecision:
        """Get recommended action for an agent."""
        if agent_id not in self.agent_policies:
            self.register_agent_policy(agent_id, PolicyType.EPSILON_GREEDY)

        policy = self.agent_policies[agent_id]
        policy.set_action_space(action_space)

        return policy.select_action(state, agent_id, self.epsilon)

    def batch_update(self, batch_size: int = 32) -> Dict[str, Any]:
        """Perform batch learning update."""
        if len(self.experiences) < batch_size:
            return {"updated": False, "experience_count": len(self.experiences)}

        # Sample batch
        batch = random.sample(self.experiences, batch_size)

        updates = 0
        total_reward = 0.0

        for exp in batch:
            agent_id = exp.agent_id
            if agent_id in self.agent_policies:
                policy = self.agent_policies[agent_id]
                policy.update_q_value(
                    exp.state,
                    exp.action,
                    exp.reward,
                    exp.next_state,
                )
                updates += 1
                total_reward += exp.reward

        return {
            "updated": True,
            "batch_size": batch_size,
            "updates": updates,
            "avg_reward": total_reward / updates if updates > 0 else 0.0,
        }

    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get RL statistics for an agent."""
        if agent_id not in self.agent_policies:
            return {"error": "Agent not found"}

        policy = self.agent_policies[agent_id]
        agent_experiences = [
            e for e in self.experiences if e.agent_id == agent_id
        ]

        return {
            "agent_id": agent_id,
            "experiences": len(agent_experiences),
            "q_values_learned": len(policy.q_values),
            "states_visited": len(policy.state_visits),
            "avg_reward": (
                sum(e.reward for e in agent_experiences) / len(agent_experiences)
                if agent_experiences
                else 0.0
            ),
            "policy_type": policy.policy_type.value,
        }

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global RL statistics."""
        total_reward = sum(e.reward for e in self.experiences)
        avg_reward = (
            total_reward / len(self.experiences)
            if self.experiences
            else 0.0
        )

        agent_stats = [
            self.get_agent_statistics(agent_id)
            for agent_id in self.agent_policies.keys()
        ]

        return {
            "total_experiences": len(self.experiences),
            "total_agents": len(self.agent_policies),
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "agent_statistics": agent_stats,
        }

    def export_policies(self) -> Dict[str, Any]:
        """Export all learned policies."""
        policies = {}
        for agent_id, policy in self.agent_policies.items():
            q_dict = {
                f"{s}_{a}": q.value
                for (s, a), q in policy.q_values.items()
            }
            policies[agent_id] = {
                "policy_type": policy.policy_type.value,
                "learning_rate": policy.learning_rate,
                "q_values": q_dict,
                "states_visited": dict(policy.state_visits),
            }
        return policies


# Import math for softmax
import math
