"""
BaseSwarm - Foundation class for all specialized agent swarms
Wired to Helix brain components (Enhancements 7-12)
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

from core.reasoning import (
    AgenticReasoner,
    CollaborativeAgenticReasoner,
    TrajectoryTracker,
    ProcessRewardModel,
    TrajectoryPattern,
    PatternLibrary,
    PlanningStrategy,
    CollaborationStrategy,
    ReasoningTrajectory,
    TrajectoryReward,
    get_agentic_reasoner,
    get_trajectory_tracker,
    get_process_reward_model,
    get_pattern_library,
    recommend_strategy_for_problem
)

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Task to be executed by swarm"""
    task_id: str
    description: str
    task_type: str  # e.g., "testing", "review", "implementation"
    context: Dict[str, Any] = field(default_factory=dict)
    requirements: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class SwarmResult:
    """Result from swarm execution"""
    task_id: str
    success: bool
    output: Any
    trajectory: ReasoningTrajectory
    quality: TrajectoryReward
    agents_involved: int
    execution_time: float
    patterns_used: List[TrajectoryPattern] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseSwarm:
    """
    Base class for all specialized swarms

    Features:
    - Wired to Helix brain (Enhancements 7-12)
    - Multi-agent collaboration (Enhancement 11)
    - Historical learning (Enhancement 12)
    - Quality assessment (Enhancement 9)
    - Trajectory tracking (Enhancement 8)
    - Resilient reasoning (Enhancement 7 via Enhancement 10)

    Each swarm specializes in a domain (testing, review, implementation, etc.)
    and learns from execution to improve over time.
    """

    def __init__(
        self,
        domain: str,
        num_agents: int = 3,
        reasoner: Optional[AgenticReasoner] = None,
        tracker: Optional[TrajectoryTracker] = None,
        prm: Optional[ProcessRewardModel] = None,
        pattern_library: Optional[PatternLibrary] = None
    ):
        """
        Initialize swarm with brain component wiring

        Args:
            domain: Swarm domain (e.g., "testing", "review")
            num_agents: Number of agents in swarm
            reasoner: AgenticReasoner (Enhancement 10) - defaults to singleton
            tracker: TrajectoryTracker (Enhancement 8) - defaults to singleton
            prm: ProcessRewardModel (Enhancement 9) - defaults to singleton
            pattern_library: PatternLibrary (Enhancement 12) - defaults to domain-specific
        """
        self.domain = domain
        self.num_agents = num_agents

        # Wire to existing Helix brain (singletons)
        self.reasoner = reasoner or get_agentic_reasoner()
        self.tracker = tracker or get_trajectory_tracker()
        self.prm = prm or get_process_reward_model()

        # Domain-specific pattern library (Enhancement 12)
        if pattern_library:
            self.pattern_library = pattern_library
        else:
            # Create domain-specific library
            import os
            from pathlib import Path
            _home = Path(os.environ.get("HELIX_HOME", Path.home() / ".helix"))
            self.pattern_library = PatternLibrary(
                storage_path=str(_home / "data" / "swarms" / f"{domain}_patterns.json")
            )

        # Create multi-agent collaborative reasoner (Enhancement 11)
        # Note: CollaborativeAgenticReasoner creates its own agents internally
        # Those agents will inherit from the shared singletons
        self.collaborative_reasoner = CollaborativeAgenticReasoner(
            num_agents=num_agents,
            max_steps_per_agent=30
        )

        logger.info(
            f"Initialized {domain} swarm with {num_agents} agents "
            f"(wired to Enhancements 7-12)"
        )

    async def execute(
        self,
        task: Task,
        collaboration_strategy: Optional[CollaborationStrategy] = None
    ) -> SwarmResult:
        """
        Execute task using multi-agent collaboration

        Workflow:
        1. Get historical patterns (Enhancement 12)
        2. Recommend strategy (Enhancement 12)
        3. Execute with collaborative reasoning (Enhancement 11)
        4. Quality assessment (Enhancement 9)
        5. Learn from execution (Enhancement 12)

        Args:
            task: Task to execute
            collaboration_strategy: How agents should collaborate (optional)

        Returns:
            SwarmResult with trajectory, quality, and output
        """
        start_time = datetime.now()

        logger.info(f"[{self.domain}] Starting task: {task.task_id}")

        # Quality Momentum Boosting: Aim to improve on previous stage
        previous_quality = task.context.get('previous_quality', 0.5)
        quality_target = min(0.95, previous_quality + 0.05)  # Aim 5% higher than previous stage

        if previous_quality > 0:
            logger.info(
                f"[{self.domain}] Quality momentum: previous={previous_quality:.3f}, "
                f"target={quality_target:.3f} (+5%)"
            )

        # 1. Get historical patterns for this task type (Enhancement 12)
        patterns = self.pattern_library.get_patterns_by_type(task.task_type)
        logger.info(f"[{self.domain}] Found {len(patterns)} historical patterns")

        # 2. Recommend best strategy based on history (Enhancement 12)
        recommended_strategy, confidence, metadata = recommend_strategy_for_problem(
            task.description
        )
        logger.info(
            f"[{self.domain}] Recommended strategy: {recommended_strategy.value} "
            f"(confidence: {confidence:.2%})"
        )

        # 3. Map planning strategy to collaboration strategy if not provided
        if collaboration_strategy is None:
            collaboration_strategy = self._map_to_collaboration_strategy(
                recommended_strategy
            )

        # 4. Execute with multi-agent collaboration (Enhancement 11)
        # This uses Enhancement 10 (AgenticReasoner) for each agent
        # Which uses Enhancement 7 (ResilientReasoner) for self-correction
        # And Enhancement 8 (TrajectoryTracker) for persistence
        logger.info(
            f"[{self.domain}] Executing with {self.num_agents} agents "
            f"using {collaboration_strategy.value}"
        )

        result = await self.collaborative_reasoner.collaborative_solve(
            problem=task.description,
            collaboration_strategy=collaboration_strategy,
            initial_context={
                'patterns': [p.to_dict() for p in patterns],
                'recommendation_metadata': metadata,
                'task_type': task.task_type,
                'requirements': task.requirements,
                'quality_target': quality_target,  # Quality momentum boosting
                'previous_quality': previous_quality,  # Context for improvement
                **task.context
            }
        )

        # 5. Quality assessment (Enhancement 9)
        # Evaluate the overall quality of the collaborative trajectories
        # Use the first/primary trajectory for quality assessment
        primary_trajectory = result.trajectories[0] if result.trajectories else None

        if primary_trajectory:
            quality = await self.prm.score_trajectory(primary_trajectory)
            # Safely access quality attributes (TrajectoryReward may have different structure)
            correctness = getattr(quality, 'correctness', getattr(quality, 'final_score', 0.5))
            efficiency = getattr(quality, 'efficiency', getattr(quality, 'final_score', 0.5))
            logger.info(
                f"[{self.domain}] Quality: {getattr(quality, 'final_score', 0.5):.3f} "
                f"(correctness: {correctness:.3f}, "
                f"efficiency: {efficiency:.3f})"
            )
        else:
            # Fallback if no trajectories available
            from core.reasoning import TrajectoryReward, RewardDimension
            quality = TrajectoryReward(
                correctness=0.5,
                efficiency=0.5,
                creativity=0.5,
                coherence=0.5,
                completeness=0.5,
                final_score=0.5,
                step_rewards=[],
                aggregation_method='average',
                metadata={}
            )
            logger.warning(f"[{self.domain}] No trajectories available, using default quality")

        # 6. Learn from execution (Enhancement 12)
        # If quality is high, save pattern for future use
        await self._learn_from_execution(task, result, quality, recommended_strategy)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        # Create swarm result
        swarm_result = SwarmResult(
            task_id=task.task_id,
            success=quality.final_score > 0.5,  # Success if quality > 50%
            output=result.final_solution,
            trajectory=primary_trajectory if primary_trajectory else result.trajectories[0] if result.trajectories else None,
            quality=quality,
            agents_involved=len(result.agent_contributions),
            execution_time=execution_time,
            patterns_used=patterns,
            metadata={
                'strategy': collaboration_strategy.value,
                'confidence': confidence,
                'recommendation_metadata': metadata,
                'num_steps': len(primary_trajectory.steps) if primary_trajectory else 0,
                'consensus_method': result.consensus_result.consensus_method if hasattr(result.consensus_result, 'consensus_method') else 'unknown'
            }
        )

        logger.info(
            f"[{self.domain}] Completed task {task.task_id} "
            f"(success: {swarm_result.success}, "
            f"quality: {quality.final_score:.3f}, "
            f"time: {execution_time:.2f}s)"
        )

        return swarm_result

    async def _learn_from_execution(
        self,
        task: Task,
        result: Any,
        quality: TrajectoryReward,
        strategy: PlanningStrategy
    ):
        """
        Learn from execution by storing successful patterns

        Enhancement 12 integration: Save high-quality patterns to domain library

        Args:
            task: Executed task
            result: Collaborative result
            quality: Quality assessment
            strategy: Strategy used
        """
        # Only save high-quality executions (>70% quality)
        if quality.final_score > 0.7:
            logger.info(
                f"[{self.domain}] High quality execution ({quality.final_score:.3f}) "
                f"- extracting pattern"
            )

            # Extract pattern from this execution
            pattern = self._extract_pattern(task, result, quality, strategy)

            # Add to domain-specific pattern library
            self.pattern_library.add_pattern(pattern)

            logger.info(
                f"[{self.domain}] Saved pattern {pattern.pattern_id} "
                f"to library"
            )
        else:
            logger.debug(
                f"[{self.domain}] Quality {quality.final_score:.3f} below threshold "
                f"- not saving pattern"
            )

    def _extract_pattern(
        self,
        task: Task,
        result: Any,
        quality: TrajectoryReward,
        strategy: PlanningStrategy
    ) -> TrajectoryPattern:
        """
        Extract a reusable pattern from successful execution

        Args:
            task: Executed task
            result: Collaborative result
            quality: Quality assessment
            strategy: Strategy used

        Returns:
            TrajectoryPattern for future use
        """
        # Get common step types from trajectory
        common_steps = []
        if hasattr(result, 'trajectory') and hasattr(result.trajectory, 'steps'):
            step_types = [step.step_type.value for step in result.trajectory.steps]
            # Get unique step types in order
            seen = set()
            for st in step_types:
                if st not in seen:
                    common_steps.append(st)
                    seen.add(st)

        # Create pattern
        pattern_id = f"{self.domain}_{task.task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        pattern = TrajectoryPattern(
            pattern_id=pattern_id,
            problem_type=task.task_type,
            successful_strategy=strategy,
            avg_quality=quality.final_score,
            avg_steps=len(result.trajectory.steps) if hasattr(result, 'trajectory') else 0,
            common_steps=common_steps,
            success_rate=1.0,  # First occurrence
            sample_count=1
        )

        return pattern

    def _map_to_collaboration_strategy(
        self,
        planning_strategy: PlanningStrategy
    ) -> CollaborationStrategy:
        """
        Map planning strategy to collaboration strategy

        Strategy mapping based on best practices:
        - HIERARCHICAL → DIVIDE_AND_CONQUER (break into subtasks)
        - FORWARD → SEQUENTIAL_REFINEMENT (build incrementally)
        - BACKWARD → PARALLEL_EXPLORATION (explore from goal)
        - BREADTH_FIRST → PARALLEL_EXPLORATION (explore broadly)

        Args:
            planning_strategy: Recommended planning strategy

        Returns:
            Appropriate collaboration strategy
        """
        mapping = {
            PlanningStrategy.HIERARCHICAL: CollaborationStrategy.DIVIDE_AND_CONQUER,
            PlanningStrategy.FORWARD: CollaborationStrategy.SEQUENTIAL_REFINEMENT,
            PlanningStrategy.BACKWARD: CollaborationStrategy.PARALLEL_EXPLORATION,
            PlanningStrategy.BIDIRECTIONAL: CollaborationStrategy.PARALLEL_EXPLORATION,
        }

        return mapping.get(
            planning_strategy,
            CollaborationStrategy.DIVIDE_AND_CONQUER  # Default
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get swarm statistics

        Returns:
            Dictionary with swarm stats
        """
        patterns = list(self.pattern_library.patterns.values())

        return {
            'domain': self.domain,
            'num_agents': self.num_agents,
            'patterns_learned': len(patterns),
            'avg_quality': sum(p.avg_quality for p in patterns) / len(patterns) if patterns else 0.0,
            'total_samples': sum(p.sample_count for p in patterns),
        }


# Singleton pattern library for global access
_global_pattern_libraries: Dict[str, PatternLibrary] = {}


def get_swarm_pattern_library(domain: str) -> PatternLibrary:
    """
    Get or create domain-specific pattern library

    Args:
        domain: Swarm domain

    Returns:
        PatternLibrary for this domain
    """
    if domain not in _global_pattern_libraries:
        import os
        from pathlib import Path
        _home = Path(os.environ.get("HELIX_HOME", Path.home() / ".helix"))
        _global_pattern_libraries[domain] = PatternLibrary(
            storage_path=str(_home / "data" / "swarms" / f"{domain}_patterns.json")
        )
    return _global_pattern_libraries[domain]
