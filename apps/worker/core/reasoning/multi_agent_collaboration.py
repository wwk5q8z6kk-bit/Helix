"""
Helix Multi-Agent Collaboration System
Enhancement 11: Enable multiple autonomous agents to collaborate on complex problems

Based on Microsoft AutoGen, Multi-Agent RL (MARL), and Swarm Intelligence research
Builds on Enhancement 10 (Agentic Reasoning)
"""

import asyncio
import json
import uuid
from datetime import datetime
from core.exceptions_unified import AgentError
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

from core.reasoning.agentic_reasoner import (
    AgenticReasoner,
    PlanningStrategy,
    AgentAction,
    Goal
)
from core.reasoning.process_reward_model import (
    ProcessRewardModel,
    TrajectoryReward
)
from core.reasoning.trajectory_tracker import (
    ReasoningTrajectory,
    TrajectoryTracker
)

logger = logging.getLogger(__name__)


# ============================================================================
# MESSAGE PROTOCOL
# ============================================================================

class MessageType(Enum):
    """Types of messages agents can exchange"""
    REQUEST = "request"                  # Request help/information
    RESPONSE = "response"                # Respond to request
    BROADCAST = "broadcast"              # Announce to all agents
    NOTIFY = "notify"                    # Notify specific agent
    PROPOSE = "propose"                  # Propose solution
    VOTE = "vote"                        # Vote on proposal
    CONSENSUS = "consensus"              # Consensus reached
    SHARE_KNOWLEDGE = "share_knowledge"  # Share findings/knowledge
    TASK_CLAIM = "task_claim"           # Claim a task
    TASK_COMPLETE = "task_complete"     # Task completion notification


@dataclass
class AgentMessage:
    """Message exchanged between agents"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_ids: List[str]  # Empty list means broadcast to all
    content: Dict[str, Any]
    timestamp: datetime
    requires_response: bool = False
    parent_message_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_ids": self.recipient_ids,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "requires_response": self.requires_response,
            "parent_message_id": self.parent_message_id,
            "metadata": self.metadata
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'AgentMessage':
        """Deserialize from dictionary"""
        return AgentMessage(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_ids=data["recipient_ids"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            requires_response=data.get("requires_response", False),
            parent_message_id=data.get("parent_message_id"),
            metadata=data.get("metadata", {})
        )


# ============================================================================
# SHARED CONTEXT
# ============================================================================

@dataclass
class KnowledgeEntry:
    """Entry in shared knowledge base"""
    key: str
    value: Any
    contributor_id: str
    confidence: float
    timestamp: datetime
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class SharedContext:
    """Thread-safe shared context for multi-agent collaboration"""

    def __init__(self):
        self._lock = asyncio.Lock()
        self._global_context: Dict[str, Any] = {}
        self._knowledge_base: Dict[str, KnowledgeEntry] = {}
        self._agent_contributions: Dict[str, List[KnowledgeEntry]] = defaultdict(list)
        self._version_history: Dict[str, List[KnowledgeEntry]] = defaultdict(list)

    async def add_knowledge(
        self,
        agent_id: str,
        key: str,
        value: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add or update knowledge in shared context"""
        async with self._lock:
            # Check if key exists
            if key in self._knowledge_base:
                # Update existing entry
                old_entry = self._knowledge_base[key]
                new_version = old_entry.version + 1

                # Archive old version
                self._version_history[key].append(old_entry)

                # Create updated entry
                entry = KnowledgeEntry(
                    key=key,
                    value=value,
                    contributor_id=agent_id,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    version=new_version,
                    metadata=metadata or {}
                )
            else:
                # Create new entry
                entry = KnowledgeEntry(
                    key=key,
                    value=value,
                    contributor_id=agent_id,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    metadata=metadata or {}
                )

            # Update knowledge base
            self._knowledge_base[key] = entry
            self._agent_contributions[agent_id].append(entry)

            logger.debug(f"Agent {agent_id} added knowledge: {key} (confidence: {confidence:.2f})")

    async def get_knowledge(
        self,
        key: str,
        include_history: bool = False
    ) -> Optional[Any]:
        """Retrieve knowledge from shared context"""
        async with self._lock:
            if key not in self._knowledge_base:
                return None

            entry = self._knowledge_base[key]

            if include_history:
                return {
                    "current": entry,
                    "history": self._version_history.get(key, [])
                }

            return entry.value

    async def get_knowledge_entry(self, key: str) -> Optional[KnowledgeEntry]:
        """Get complete knowledge entry with metadata"""
        async with self._lock:
            return self._knowledge_base.get(key)

    async def merge_contexts(
        self,
        contexts: List[Dict[str, Any]],
        strategy: str = "highest_confidence"
    ) -> Dict[str, Any]:
        """
        Merge multiple agent contexts using specified strategy

        Strategies:
        - highest_confidence: Use value with highest confidence
        - average: Average numerical values, concatenate others
        - newest: Use most recent value
        - consensus: Use value that appears most frequently
        """
        async with self._lock:
            merged = {}

            # Group values by key
            key_values: Dict[str, List[Tuple[Any, float, datetime]]] = defaultdict(list)

            for context in contexts:
                for key, value in context.items():
                    # Default confidence and timestamp if not specified
                    conf = context.get(f"{key}_confidence", 1.0)
                    ts = context.get(f"{key}_timestamp", datetime.now())
                    key_values[key].append((value, conf, ts))

            # Apply merge strategy
            for key, values in key_values.items():
                if strategy == "highest_confidence":
                    # Use value with highest confidence
                    merged[key] = max(values, key=lambda x: x[1])[0]

                elif strategy == "newest":
                    # Use most recent value
                    merged[key] = max(values, key=lambda x: x[2])[0]

                elif strategy == "consensus":
                    # Use most frequent value
                    value_counts = {}
                    for val, _, _ in values:
                        val_key = str(val)
                        value_counts[val_key] = value_counts.get(val_key, 0) + 1
                    most_common = max(value_counts.items(), key=lambda x: x[1])[0]
                    # Find original value
                    for val, _, _ in values:
                        if str(val) == most_common:
                            merged[key] = val
                            break

                elif strategy == "average":
                    # Average if all numerical, otherwise use highest confidence
                    if all(isinstance(v[0], (int, float)) for v in values):
                        merged[key] = sum(v[0] for v in values) / len(values)
                    else:
                        merged[key] = max(values, key=lambda x: x[1])[0]

            return merged

    async def get_agent_contributions(self, agent_id: str) -> List[KnowledgeEntry]:
        """Get all contributions from a specific agent"""
        async with self._lock:
            return self._agent_contributions.get(agent_id, []).copy()

    async def get_all_knowledge(self) -> Dict[str, KnowledgeEntry]:
        """Get entire knowledge base"""
        async with self._lock:
            return self._knowledge_base.copy()

    async def clear(self) -> None:
        """Clear all shared context (for testing)"""
        async with self._lock:
            self._global_context.clear()
            self._knowledge_base.clear()
            self._agent_contributions.clear()
            self._version_history.clear()


# ============================================================================
# COMMUNICATION PROTOCOL
# ============================================================================

class CommunicationProtocol:
    """Handles message passing between agents"""

    def __init__(self):
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._agent_inboxes: Dict[str, asyncio.Queue] = {}
        self._message_history: List[AgentMessage] = []
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._running = False
        self._router_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the communication protocol"""
        if not self._running:
            self._running = True
            self._router_task = asyncio.create_task(self._message_router())
            logger.info("Communication protocol started")

    async def stop(self) -> None:
        """Stop the communication protocol"""
        if self._running:
            self._running = False
            if self._router_task:
                self._router_task.cancel()
                try:
                    await self._router_task
                except asyncio.CancelledError:
                    pass
            logger.info("Communication protocol stopped")

    def register_agent(self, agent_id: str) -> None:
        """Register an agent for communication"""
        if agent_id not in self._agent_inboxes:
            self._agent_inboxes[agent_id] = asyncio.Queue()
            logger.debug(f"Registered agent: {agent_id}")

    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self._agent_inboxes:
            del self._agent_inboxes[agent_id]
            logger.debug(f"Unregistered agent: {agent_id}")

    async def send_message(self, message: AgentMessage) -> None:
        """Send a message"""
        await self._message_queue.put(message)
        self._message_history.append(message)
        logger.debug(f"Message sent: {message.message_type.value} from {message.sender_id}")

    async def send_request(
        self,
        sender_id: str,
        recipient_id: str,
        content: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """
        Send request and wait for response

        Returns response content or None if timeout
        """
        message_id = str(uuid.uuid4())
        message = AgentMessage(
            message_id=message_id,
            message_type=MessageType.REQUEST,
            sender_id=sender_id,
            recipient_ids=[recipient_id],
            content=content,
            timestamp=datetime.now(),
            requires_response=True
        )

        # Create future for response
        future = asyncio.Future()
        self._response_futures[message_id] = future

        # Send message
        await self.send_message(message)

        # Wait for response with timeout
        try:
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Request from {sender_id} to {recipient_id} timed out")
            return None
        finally:
            # Clean up future
            if message_id in self._response_futures:
                del self._response_futures[message_id]

    async def send_response(
        self,
        sender_id: str,
        recipient_id: str,
        parent_message_id: str,
        content: Dict[str, Any]
    ) -> None:
        """Send response to a request"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            sender_id=sender_id,
            recipient_ids=[recipient_id],
            content=content,
            timestamp=datetime.now(),
            parent_message_id=parent_message_id
        )

        await self.send_message(message)

        # Resolve future if exists
        if parent_message_id in self._response_futures:
            self._response_futures[parent_message_id].set_result(content)

    async def broadcast(
        self,
        sender_id: str,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Broadcast message to all agents"""
        message = AgentMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.BROADCAST,
            sender_id=sender_id,
            recipient_ids=[],  # Empty means all
            content=content,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        await self.send_message(message)

    async def receive_message(
        self,
        agent_id: str,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """Receive next message for agent"""
        if agent_id not in self._agent_inboxes:
            return None

        inbox = self._agent_inboxes[agent_id]

        try:
            if timeout:
                message = await asyncio.wait_for(inbox.get(), timeout=timeout)
            else:
                message = await inbox.get()
            return message
        except asyncio.TimeoutError:
            return None

    async def _message_router(self) -> None:
        """Route messages to appropriate agent inboxes"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self._message_queue.get(),
                    timeout=0.1
                )

                # Broadcast to all
                if not message.recipient_ids:
                    for agent_id, inbox in self._agent_inboxes.items():
                        if agent_id != message.sender_id:  # Don't send to self
                            await inbox.put(message)
                else:
                    # Send to specific recipients
                    for recipient_id in message.recipient_ids:
                        if recipient_id in self._agent_inboxes:
                            await self._agent_inboxes[recipient_id].put(message)

            except asyncio.TimeoutError:
                # No messages, continue
                continue
            except AgentError as e:
                logger.error(f"Error in message router: {e}")

    async def get_message_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100
    ) -> List[AgentMessage]:
        """Get message history with optional filters"""
        messages = self._message_history.copy()

        # Filter by agent
        if agent_id:
            messages = [m for m in messages if m.sender_id == agent_id or agent_id in m.recipient_ids]

        # Filter by type
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]

        # Limit
        return messages[-limit:]


# ============================================================================
# COLLABORATION STRATEGIES
# ============================================================================

class CollaborationStrategy(Enum):
    """Strategies for multi-agent collaboration"""
    DIVIDE_AND_CONQUER = "divide_and_conquer"      # Split problem into subgoals
    PARALLEL_EXPLORATION = "parallel_exploration"   # Multiple approaches simultaneously
    SEQUENTIAL_REFINEMENT = "sequential_refinement" # Pass solution between agents
    VOTING = "voting"                              # All solve independently, vote on best
    LEADER_FOLLOWER = "leader_follower"           # One leads, others assist


# ============================================================================
# CONSENSUS PROTOCOL
# ============================================================================

@dataclass
class ConsensusResult:
    """Result of consensus mechanism"""
    selected_solution: Any
    votes: Dict[str, int]  # solution_id -> vote_count
    quality_scores: Dict[str, float]  # solution_id -> quality
    consensus_type: str
    unanimous: bool
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConsensusProtocol:
    """Consensus building for multi-agent decisions"""

    def __init__(self, prm: Optional[ProcessRewardModel] = None):
        self.prm = prm or ProcessRewardModel()

    async def majority_vote(
        self,
        proposals: List[Any],
        threshold: float = 0.5
    ) -> Optional[Any]:
        """
        Simple majority voting

        Returns winning proposal or None if no majority
        """
        if not proposals:
            return None

        # Count votes
        vote_counts: Dict[str, int] = {}
        for proposal in proposals:
            proposal_key = str(proposal)
            vote_counts[proposal_key] = vote_counts.get(proposal_key, 0) + 1

        # Find majority
        total_votes = len(proposals)
        for proposal_key, count in vote_counts.items():
            if count / total_votes >= threshold:
                # Return original proposal
                for proposal in proposals:
                    if str(proposal) == proposal_key:
                        return proposal

        return None

    async def quality_weighted_consensus(
        self,
        agent_results: List[Tuple[str, ReasoningTrajectory]],
        context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        Weight consensus by PRM quality scores

        Returns solution with highest weighted quality
        """
        if not agent_results:
            raise ValueError("No agent results provided")

        # Score each trajectory
        scored_results = []
        for agent_id, trajectory in agent_results:
            reward = await self.prm.score_trajectory(trajectory, context)
            scored_results.append((agent_id, trajectory, reward.final_score))

        # Find best solution
        best_agent, best_trajectory, best_score = max(
            scored_results,
            key=lambda x: x[2]
        )

        # Create consensus result
        votes = {agent_id: 1 for agent_id, _, _ in scored_results}
        quality_scores = {agent_id: score for agent_id, _, score in scored_results}

        return ConsensusResult(
            selected_solution=best_trajectory.final_answer,
            votes=votes,
            quality_scores=quality_scores,
            consensus_type="quality_weighted",
            unanimous=len(set(quality_scores.values())) == 1,
            confidence=best_score,
            metadata={"best_agent": best_agent, "score_spread": max(quality_scores.values()) - min(quality_scores.values())}
        )

    async def rank_based_consensus(
        self,
        agent_results: List[Tuple[str, ReasoningTrajectory]],
        context: Optional[Dict[str, Any]] = None
    ) -> ConsensusResult:
        """
        Rank solutions and select top-ranked

        Each agent ranks all solutions, aggregate rankings
        """
        if not agent_results:
            raise ValueError("No agent results provided")

        # Score all solutions
        scores = []
        for agent_id, trajectory in agent_results:
            reward = await self.prm.score_trajectory(trajectory, context)
            scores.append((agent_id, trajectory, reward.final_score))

        # Sort by score
        scores.sort(key=lambda x: x[2], reverse=True)

        # Top-ranked solution
        best_agent, best_trajectory, best_score = scores[0]

        quality_scores = {agent_id: score for agent_id, _, score in scores}

        return ConsensusResult(
            selected_solution=best_trajectory.final_answer,
            votes={agent_id: len(scores) - i for i, (agent_id, _, _) in enumerate(scores)},
            quality_scores=quality_scores,
            consensus_type="rank_based",
            unanimous=False,
            confidence=best_score,
            metadata={"rankings": [(agent_id, score) for agent_id, _, score in scores]}
        )

    async def unanimous_consensus(
        self,
        proposals: List[Any]
    ) -> Optional[Any]:
        """
        Require unanimous agreement

        Returns solution only if all agree
        """
        if not proposals:
            return None

        # Check if all identical
        first = str(proposals[0])
        if all(str(p) == first for p in proposals):
            return proposals[0]

        return None


# ============================================================================
# COLLABORATIVE RESULT
# ============================================================================

@dataclass
class CollaborativeResult:
    """Result from collaborative multi-agent problem solving"""
    final_solution: Any
    consensus_result: ConsensusResult
    agent_contributions: Dict[str, Dict[str, Any]]  # agent_id -> {trajectory, reward, etc}
    collaboration_strategy: CollaborationStrategy
    total_time: float
    num_agents: int
    trajectories: List[ReasoningTrajectory]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# COLLABORATIVE AGENTIC REASONER
# ============================================================================

class CollaborativeAgenticReasoner:
    """Multi-agent collaborative reasoning system"""

    def __init__(
        self,
        num_agents: int = 3,
        communication_protocol: Optional[CommunicationProtocol] = None,
        shared_context: Optional[SharedContext] = None,
        consensus_protocol: Optional[ConsensusProtocol] = None,
        max_steps_per_agent: int = 50,
        reflection_frequency: int = 5
    ):
        """
        Initialize collaborative reasoning system

        Args:
            num_agents: Number of collaborative agents
            communication_protocol: Protocol for agent communication
            shared_context: Shared knowledge/context system
            consensus_protocol: Protocol for reaching consensus
            max_steps_per_agent: Max reasoning steps per agent
            reflection_frequency: How often agents reflect
        """
        self.num_agents = num_agents
        self.communication = communication_protocol or get_communication_protocol()
        self.shared_context = shared_context or get_shared_context()
        self.consensus = consensus_protocol or get_consensus_protocol()

        # Create individual agents
        self.agents: List[AgenticReasoner] = []
        for i in range(num_agents):
            agent = AgenticReasoner(
                max_steps=max_steps_per_agent,
                reflection_frequency=reflection_frequency
            )
            agent.agent_id = f"agent_{i}"
            self.agents.append(agent)

        # Register agents with communication protocol
        for agent in self.agents:
            self.communication.register_agent(agent.agent_id)

        logger.info(f"CollaborativeAgenticReasoner initialized with {num_agents} agents")

    async def collaborative_solve(
        self,
        problem: str,
        collaboration_strategy: CollaborationStrategy = CollaborationStrategy.DIVIDE_AND_CONQUER,
        initial_context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CollaborativeResult:
        """
        Solve problem collaboratively using specified strategy

        Args:
            problem: Problem to solve
            collaboration_strategy: Strategy for collaboration
            initial_context: Initial context for all agents
            metadata: Metadata for tracking

        Returns:
            CollaborativeResult with final solution and details
        """
        start_time = datetime.now()

        # Start communication protocol
        await self.communication.start()

        try:
            # Add initial context to shared memory
            if initial_context:
                for key, value in initial_context.items():
                    await self.shared_context.add_knowledge(
                        agent_id="system",
                        key=key,
                        value=value,
                        confidence=1.0
                    )

            # Execute based on strategy
            if collaboration_strategy == CollaborationStrategy.DIVIDE_AND_CONQUER:
                result = await self._divide_and_conquer(problem, initial_context, metadata)
            elif collaboration_strategy == CollaborationStrategy.PARALLEL_EXPLORATION:
                result = await self._parallel_exploration(problem, initial_context, metadata)
            elif collaboration_strategy == CollaborationStrategy.VOTING:
                result = await self._voting(problem, initial_context, metadata)
            elif collaboration_strategy == CollaborationStrategy.SEQUENTIAL_REFINEMENT:
                result = await self._sequential_refinement(problem, initial_context, metadata)
            elif collaboration_strategy == CollaborationStrategy.LEADER_FOLLOWER:
                result = await self._leader_follower(problem, initial_context, metadata)
            else:
                raise ValueError(f"Unknown collaboration strategy: {collaboration_strategy}")

            # Calculate total time
            total_time = (datetime.now() - start_time).total_seconds()
            result.total_time = total_time
            result.metadata["collaboration_strategy"] = collaboration_strategy.value

            return result

        finally:
            # Stop communication protocol
            await self.communication.stop()

    async def _divide_and_conquer(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> CollaborativeResult:
        """
        Divide problem into subproblems, assign to agents, synthesize results
        """
        logger.info("Starting divide-and-conquer collaboration")

        # Use first agent to decompose problem
        decomposer = self.agents[0]
        decompose_trajectory, _ = await decomposer.solve(
            problem=f"Decompose this problem into {self.num_agents} subproblems: {problem}",
            planning_strategy=PlanningStrategy.HIERARCHICAL,
            initial_context=initial_context,
            metadata={"role": "decomposer"}
        )

        # Extract subproblems from goals
        subproblems = []
        if decomposer.goals:
            root_goal = list(decomposer.goals.values())[0]
            for subgoal_id in root_goal.subgoals[:self.num_agents]:
                if subgoal_id in decomposer.goals:
                    subproblems.append(decomposer.goals[subgoal_id].description)

        # If not enough subgoals, split problem text
        if len(subproblems) < self.num_agents:
            subproblems = [f"Aspect {i+1} of: {problem}" for i in range(self.num_agents)]

        # Assign subproblems to agents in parallel
        tasks = []
        for i, (agent, subproblem) in enumerate(zip(self.agents, subproblems)):
            task = agent.solve(
                problem=subproblem,
                planning_strategy=PlanningStrategy.FORWARD,
                initial_context=initial_context,
                metadata={"subproblem_id": i, **(metadata or {})}
            )
            tasks.append((agent.agent_id, task))

        # Execute in parallel
        results = []
        for agent_id, task in tasks:
            trajectory, reward = await task
            results.append((agent_id, trajectory))

            # Share solution with other agents
            await self.shared_context.add_knowledge(
                agent_id=agent_id,
                key=f"solution_{agent_id}",
                value=trajectory.final_answer,
                confidence=reward.final_score
            )

        # Build consensus on final solution
        consensus_result = await self.consensus.quality_weighted_consensus(
            agent_results=results,
            context=initial_context
        )

        # Build agent contributions
        agent_contributions = {}
        for agent_id, trajectory in results:
            agent_contributions[agent_id] = {
                "trajectory": trajectory,
                "solution": trajectory.final_answer,
                "steps": len(trajectory.steps)
            }

        return CollaborativeResult(
            final_solution=consensus_result.selected_solution,
            consensus_result=consensus_result,
            agent_contributions=agent_contributions,
            collaboration_strategy=CollaborationStrategy.DIVIDE_AND_CONQUER,
            total_time=0.0,  # Will be set by caller
            num_agents=self.num_agents,
            trajectories=[t for _, t in results],
            metadata={"subproblems": subproblems}
        )

    async def _parallel_exploration(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> CollaborativeResult:
        """
        All agents explore problem independently with different strategies
        """
        logger.info("Starting parallel exploration")

        # Assign different planning strategies to agents
        strategies = [
            PlanningStrategy.FORWARD,
            PlanningStrategy.BACKWARD,
            PlanningStrategy.HIERARCHICAL
        ]

        # Execute in parallel with different strategies
        tasks = []
        for i, agent in enumerate(self.agents):
            strategy = strategies[i % len(strategies)]
            task = agent.solve(
                problem=problem,
                planning_strategy=strategy,
                initial_context=initial_context,
                metadata={"strategy": strategy.value, **(metadata or {})}
            )
            tasks.append((agent.agent_id, task, strategy))

        # Collect results
        results = []
        for agent_id, task, strategy in tasks:
            trajectory, reward = await task
            results.append((agent_id, trajectory))

            # Share findings
            await self.communication.broadcast(
                sender_id=agent_id,
                content={
                    "type": "exploration_result",
                    "strategy": strategy.value,
                    "quality": reward.final_score,
                    "key_findings": trajectory.final_answer
                }
            )

        # Consensus on best approach
        consensus_result = await self.consensus.quality_weighted_consensus(
            agent_results=results,
            context=initial_context
        )

        # Build contributions
        agent_contributions = {}
        for agent_id, trajectory in results:
            agent_contributions[agent_id] = {
                "trajectory": trajectory,
                "solution": trajectory.final_answer,
                "strategy": tasks[[t[0] for t in tasks].index(agent_id)][2].value
            }

        return CollaborativeResult(
            final_solution=consensus_result.selected_solution,
            consensus_result=consensus_result,
            agent_contributions=agent_contributions,
            collaboration_strategy=CollaborationStrategy.PARALLEL_EXPLORATION,
            total_time=0.0,
            num_agents=self.num_agents,
            trajectories=[t for _, t in results]
        )

    async def _voting(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> CollaborativeResult:
        """
        All agents solve independently, then vote on best solution
        """
        logger.info("Starting voting-based collaboration")

        # All agents solve the same problem
        tasks = []
        for agent in self.agents:
            task = agent.solve(
                problem=problem,
                planning_strategy=PlanningStrategy.FORWARD,
                initial_context=initial_context,
                metadata=metadata
            )
            tasks.append((agent.agent_id, task))

        # Collect all solutions
        results = []
        for agent_id, task in tasks:
            trajectory, reward = await task
            results.append((agent_id, trajectory))

        # Rank-based consensus (all agents vote)
        consensus_result = await self.consensus.rank_based_consensus(
            agent_results=results,
            context=initial_context
        )

        # Build contributions
        agent_contributions = {}
        for agent_id, trajectory in results:
            agent_contributions[agent_id] = {
                "trajectory": trajectory,
                "solution": trajectory.final_answer,
                "votes": consensus_result.votes.get(agent_id, 0)
            }

        return CollaborativeResult(
            final_solution=consensus_result.selected_solution,
            consensus_result=consensus_result,
            agent_contributions=agent_contributions,
            collaboration_strategy=CollaborationStrategy.VOTING,
            total_time=0.0,
            num_agents=self.num_agents,
            trajectories=[t for _, t in results]
        )

    async def _sequential_refinement(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> CollaborativeResult:
        """
        Agents work sequentially, each refining previous agent's solution
        """
        logger.info("Starting sequential refinement")

        current_solution = None
        all_trajectories = []
        agent_contributions = {}

        for i, agent in enumerate(self.agents):
            if i == 0:
                # First agent solves original problem
                task_problem = problem
            else:
                # Subsequent agents refine previous solution
                task_problem = f"Improve and refine this solution: {current_solution}\n\nOriginal problem: {problem}"

            # Agent solves
            trajectory, reward = await agent.solve(
                problem=task_problem,
                planning_strategy=PlanningStrategy.FORWARD,
                initial_context=initial_context,
                metadata={"iteration": i, **(metadata or {})}
            )

            # Update current solution
            current_solution = trajectory.final_answer
            all_trajectories.append((agent.agent_id, trajectory))

            # Record contribution
            agent_contributions[agent.agent_id] = {
                "trajectory": trajectory,
                "solution": current_solution,
                "iteration": i,
                "quality": reward.final_score
            }

            # Share with next agent
            if i < len(self.agents) - 1:
                await self.shared_context.add_knowledge(
                    agent_id=agent.agent_id,
                    key=f"refinement_{i}",
                    value=current_solution,
                    confidence=reward.final_score
                )

        # Final solution is last agent's output
        final_trajectory = all_trajectories[-1][1]

        # Create consensus result (sequential, so last one wins)
        consensus_result = ConsensusResult(
            selected_solution=current_solution,
            votes={agent_id: i+1 for i, (agent_id, _) in enumerate(all_trajectories)},
            quality_scores={agent_id: agent_contributions[agent_id]["quality"]
                          for agent_id in agent_contributions},
            consensus_type="sequential",
            unanimous=False,
            confidence=agent_contributions[all_trajectories[-1][0]]["quality"]
        )

        return CollaborativeResult(
            final_solution=current_solution,
            consensus_result=consensus_result,
            agent_contributions=agent_contributions,
            collaboration_strategy=CollaborationStrategy.SEQUENTIAL_REFINEMENT,
            total_time=0.0,
            num_agents=self.num_agents,
            trajectories=[t for _, t in all_trajectories]
        )

    async def _leader_follower(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]]
    ) -> CollaborativeResult:
        """
        One agent leads (hierarchical planning), others assist with subgoals
        """
        logger.info("Starting leader-follower collaboration")

        # First agent is leader
        leader = self.agents[0]
        followers = self.agents[1:]

        # Leader decomposes problem
        leader_trajectory, leader_reward = await leader.solve(
            problem=problem,
            planning_strategy=PlanningStrategy.HIERARCHICAL,
            initial_context=initial_context,
            metadata={"role": "leader", **(metadata or {})}
        )

        # Extract subgoals
        subgoals = []
        if leader.goals:
            root_goal = list(leader.goals.values())[0]
            for subgoal_id in root_goal.subgoals:
                if subgoal_id in leader.goals:
                    subgoals.append(leader.goals[subgoal_id].description)

        # Assign subgoals to followers
        follower_results = []
        for i, (follower, subgoal) in enumerate(zip(followers, subgoals)):
            trajectory, reward = await follower.solve(
                problem=subgoal,
                planning_strategy=PlanningStrategy.FORWARD,
                initial_context=initial_context,
                metadata={"role": "follower", "subgoal_id": i, **(metadata or {})}
            )
            follower_results.append((follower.agent_id, trajectory))

        # Combine leader and follower results
        all_results = [(leader.agent_id, leader_trajectory)] + follower_results

        # Leader's solution takes precedence
        agent_contributions = {}
        agent_contributions[leader.agent_id] = {
            "trajectory": leader_trajectory,
            "solution": leader_trajectory.final_answer,
            "role": "leader"
        }

        for agent_id, trajectory in follower_results:
            agent_contributions[agent_id] = {
                "trajectory": trajectory,
                "solution": trajectory.final_answer,
                "role": "follower"
            }

        # Consensus favors leader
        consensus_result = ConsensusResult(
            selected_solution=leader_trajectory.final_answer,
            votes={leader.agent_id: len(self.agents)},  # Leader gets all votes
            quality_scores={leader.agent_id: leader_reward.final_score},
            consensus_type="leader_based",
            unanimous=False,
            confidence=leader_reward.final_score
        )

        return CollaborativeResult(
            final_solution=leader_trajectory.final_answer,
            consensus_result=consensus_result,
            agent_contributions=agent_contributions,
            collaboration_strategy=CollaborationStrategy.LEADER_FOLLOWER,
            total_time=0.0,
            num_agents=self.num_agents,
            trajectories=[t for _, t in all_results],
            metadata={"subgoals": subgoals}
        )


# ============================================================================
# SINGLETON ACCESS
# ============================================================================

_shared_context_instance: Optional[SharedContext] = None
_communication_protocol_instance: Optional[CommunicationProtocol] = None
_consensus_protocol_instance: Optional[ConsensusProtocol] = None
_collaborative_reasoner_instance: Optional[CollaborativeAgenticReasoner] = None


def get_shared_context() -> SharedContext:
    """Get or create singleton SharedContext"""
    global _shared_context_instance
    if _shared_context_instance is None:
        _shared_context_instance = SharedContext()
    return _shared_context_instance


def get_communication_protocol() -> CommunicationProtocol:
    """Get or create singleton CommunicationProtocol"""
    global _communication_protocol_instance
    if _communication_protocol_instance is None:
        _communication_protocol_instance = CommunicationProtocol()
    return _communication_protocol_instance


def get_consensus_protocol(prm: Optional[ProcessRewardModel] = None) -> ConsensusProtocol:
    """Get or create singleton ConsensusProtocol"""
    global _consensus_protocol_instance
    if _consensus_protocol_instance is None:
        _consensus_protocol_instance = ConsensusProtocol(prm)
    return _consensus_protocol_instance


def get_collaborative_reasoner(
    num_agents: int = 3,
    max_steps_per_agent: int = 50,
    reflection_frequency: int = 5
) -> CollaborativeAgenticReasoner:
    """Get or create singleton CollaborativeAgenticReasoner"""
    global _collaborative_reasoner_instance
    if _collaborative_reasoner_instance is None:
        _collaborative_reasoner_instance = CollaborativeAgenticReasoner(
            num_agents=num_agents,
            max_steps_per_agent=max_steps_per_agent,
            reflection_frequency=reflection_frequency
        )
    return _collaborative_reasoner_instance
