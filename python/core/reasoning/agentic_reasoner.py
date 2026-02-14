"""
Helix Agentic Reasoning Engine
Autonomous multi-step problem solving with self-correction and quality assessment
Based on Microsoft ARTIST, OpenAI o1/o3, and Agent-R research (2025)
"""

import asyncio
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from core.reasoning.trajectory_tracker import (
    ReasoningStep,
    ReasoningTrajectory,
    StepType,
    TrajectoryTracker,
    get_trajectory_tracker
)

from core.reasoning.resilient_reasoner import (
    ResilientReasoner,
    get_resilient_reasoner
)

from core.reasoning.process_reward_model import (
    ProcessRewardModel,
    StepReward,
    TrajectoryReward,
    get_process_reward_model
)


class AgentAction(str, Enum):
    """Types of actions an agent can take"""
    THINK = "think"                    # Internal reasoning
    PLAN = "plan"                      # Create/update plan
    TOOL = "tool"                      # Execute tool
    RETRIEVE = "retrieve"              # Retrieve information
    VERIFY = "verify"                  # Verify step/solution
    REFLECT = "reflect"                # Reflect on progress
    DECOMPOSE = "decompose"            # Break down problem
    SYNTHESIZE = "synthesize"          # Combine information


class PlanningStrategy(str, Enum):
    """Strategies for planning"""
    FORWARD = "forward"                # Plan from start to goal
    BACKWARD = "backward"              # Work backwards from goal
    BIDIRECTIONAL = "bidirectional"    # Plan from both ends
    HIERARCHICAL = "hierarchical"      # Multi-level planning


@dataclass
class Goal:
    """A goal to achieve"""
    goal_id: str
    description: str
    parent_goal: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """A plan to achieve a goal"""
    plan_id: str
    goal_id: str
    steps: List[Dict[str, Any]]        # Planned steps
    strategy: PlanningStrategy
    confidence: float                  # 0.0 to 1.0
    estimated_difficulty: float        # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """Definition of a tool the agent can use"""
    name: str
    description: str
    parameters: Dict[str, Any]
    execute: Callable                  # Async function to execute tool
    cost: float = 1.0                  # Relative cost (time/money)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectionResult:
    """Result of self-reflection"""
    what_worked: List[str]
    what_failed: List[str]
    lessons_learned: List[str]
    adjustments_needed: List[str]
    confidence_change: float           # -1.0 to 1.0
    should_continue: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgenticReasoner:
    """
    Autonomous Agentic Reasoning Engine

    Features:
    - Goal decomposition and hierarchical planning
    - Multi-step autonomous execution
    - Tool selection and execution
    - Continuous self-reflection and improvement
    - Integration with PRM for quality assessment
    - Resilient self-correction on errors
    """

    def __init__(
        self,
        tracker: Optional[TrajectoryTracker] = None,
        resilient_reasoner: Optional[ResilientReasoner] = None,
        prm: Optional[ProcessRewardModel] = None,
        max_steps: int = 50,
        reflection_frequency: int = 5
    ):
        """
        Initialize Agentic Reasoner

        Args:
            tracker: Trajectory tracker for recording reasoning
            resilient_reasoner: Self-correction engine
            prm: Process reward model for quality assessment
            max_steps: Maximum steps before stopping
            reflection_frequency: Reflect every N steps
        """
        self.tracker = tracker or get_trajectory_tracker()
        self.resilient_reasoner = resilient_reasoner or get_resilient_reasoner(self.tracker)
        self.prm = prm or get_process_reward_model()

        self.max_steps = max_steps
        self.reflection_frequency = reflection_frequency

        # Tool registry
        self.tools: Dict[str, ToolDefinition] = {}

        # Goal hierarchy
        self.goals: Dict[str, Goal] = {}

        # Active plans
        self.plans: Dict[str, Plan] = {}

    def register_tool(self, tool: ToolDefinition):
        """Register a tool that the agent can use"""
        self.tools[tool.name] = tool

    async def solve(
        self,
        problem: str,
        initial_context: Optional[Dict[str, Any]] = None,
        planning_strategy: PlanningStrategy = PlanningStrategy.FORWARD,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[ReasoningTrajectory, TrajectoryReward]:
        """
        Autonomously solve a problem with agentic reasoning

        Args:
            problem: Problem description
            initial_context: Optional context/information
            planning_strategy: How to plan the solution
            metadata: Additional metadata

        Returns:
            Tuple of (trajectory, reward_assessment)
        """

        # Create trajectory
        trajectory = self.tracker.create_trajectory(
            problem=problem,
            metadata={
                **(metadata or {}),
                'planning_strategy': planning_strategy.value,
                'agentic': True
            }
        )

        context = initial_context or {}
        context['problem'] = problem
        context['planning_strategy'] = planning_strategy

        # Phase 1: Goal Decomposition
        root_goal = await self._decompose_goal(problem, trajectory, context)

        # Phase 2: Planning
        plan = await self._create_plan(root_goal, planning_strategy, trajectory, context)

        # Phase 3: Execution with Reflection
        execution_result = await self._execute_with_reflection(
            plan,
            trajectory,
            context
        )

        # Phase 4: Final Verification
        final_verification = await self._verify_solution(
            trajectory,
            root_goal,
            context
        )

        # Complete trajectory
        success = final_verification.get('success', False)
        result = execution_result.get('result', 'No result')
        # Convert dict results to string for storage
        if isinstance(result, dict):
            result = json.dumps(result)
        trajectory.complete(
            final_answer=result,
            success=success
        )

        # Save to database
        self.tracker.update_trajectory(trajectory)

        # Assess quality with PRM
        reward = await self.prm.score_trajectory(trajectory)

        return trajectory, reward

    async def _decompose_goal(
        self,
        problem: str,
        trajectory: ReasoningTrajectory,
        context: Dict[str, Any]
    ) -> Goal:
        """Decompose problem into hierarchical goals"""

        # Add decomposition step
        decompose_step = trajectory.add_step(
            content=f"Decompose problem: {problem}",
            step_type=StepType.THINK,
            confidence=0.9,
            metadata={'action': str(AgentAction.DECOMPOSE.value)}
        )
        self.tracker.save_step(trajectory.trajectory_id, decompose_step)

        # Create root goal
        root_goal = Goal(
            goal_id=str(uuid.uuid4())[:8],
            description=problem,
            success_criteria={'problem_solved': True}
        )

        self.goals[root_goal.goal_id] = root_goal

        # Decompose into subgoals (simplified heuristic)
        if len(problem.split()) > 10:  # Complex problem
            subgoals = self._heuristic_decompose(problem)
            for sg_desc in subgoals:
                subgoal = Goal(
                    goal_id=str(uuid.uuid4())[:8],
                    description=sg_desc,
                    parent_goal=root_goal.goal_id
                )
                self.goals[subgoal.goal_id] = subgoal
                root_goal.subgoals.append(subgoal.goal_id)

        return root_goal

    def _heuristic_decompose(self, problem: str) -> List[str]:
        """Heuristic problem decomposition"""
        # Simple heuristic: break into understand, plan, implement, verify
        return [
            f"Understand: {problem[:50]}...",
            f"Plan solution approach",
            f"Implement solution",
            f"Verify and test solution"
        ]

    async def _create_plan(
        self,
        goal: Goal,
        strategy: PlanningStrategy,
        trajectory: ReasoningTrajectory,
        context: Dict[str, Any]
    ) -> Plan:
        """Create a plan to achieve the goal"""

        # Add planning step
        plan_step = trajectory.add_step(
            content=f"Create {strategy.value} plan for: {goal.description}",
            step_type=StepType.THINK,
            confidence=0.85,
            metadata={'action': str(AgentAction.PLAN.value), 'strategy': str(strategy.value)}
        )
        self.tracker.save_step(trajectory.trajectory_id, plan_step)

        # Create plan steps based on strategy
        if strategy == PlanningStrategy.FORWARD:
            planned_steps = self._plan_forward(goal, context)
        elif strategy == PlanningStrategy.BACKWARD:
            planned_steps = self._plan_backward(goal, context)
        elif strategy == PlanningStrategy.HIERARCHICAL:
            planned_steps = self._plan_hierarchical(goal, context)
        else:
            planned_steps = self._plan_forward(goal, context)

        plan = Plan(
            plan_id=str(uuid.uuid4())[:8],
            goal_id=goal.goal_id,
            steps=planned_steps,
            strategy=strategy,
            confidence=0.8,
            estimated_difficulty=min(len(planned_steps) / 10.0, 1.0)
        )

        self.plans[plan.plan_id] = plan
        return plan

    def _plan_forward(self, goal: Goal, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Forward planning: start → goal"""
        steps = []

        # If goal has subgoals, plan for each
        if goal.subgoals:
            for subgoal_id in goal.subgoals:
                subgoal = self.goals[subgoal_id]
                steps.append({
                    'action': AgentAction.THINK.value,
                    'description': f"Work on: {subgoal.description}",
                    'goal_id': subgoal.goal_id
                })
        else:
            # Simple sequential plan
            steps = [
                {'action': AgentAction.THINK.value, 'description': 'Analyze problem'},
                {'action': AgentAction.RETRIEVE.value, 'description': 'Gather information'},
                {'action': AgentAction.TOOL.value, 'description': 'Execute solution'},
                {'action': AgentAction.VERIFY.value, 'description': 'Verify result'}
            ]

        return steps

    def _plan_backward(self, goal: Goal, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Backward planning: goal → start"""
        # Work backwards from goal
        steps = [
            {'action': AgentAction.VERIFY.value, 'description': 'Verify goal achieved'},
            {'action': AgentAction.TOOL.value, 'description': 'Execute final step'},
            {'action': AgentAction.THINK.value, 'description': 'Determine prerequisites'},
            {'action': AgentAction.RETRIEVE.value, 'description': 'Gather needed info'}
        ]
        return list(reversed(steps))  # Reverse for execution

    def _plan_hierarchical(self, goal: Goal, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Hierarchical planning: multi-level"""
        steps = []

        # Top-level plan
        steps.append({'action': AgentAction.DECOMPOSE.value, 'description': 'Break into subproblems'})

        # For each subgoal
        if goal.subgoals:
            for subgoal_id in goal.subgoals:
                subgoal = self.goals[subgoal_id]
                steps.extend([
                    {'action': AgentAction.PLAN.value, 'description': f'Plan: {subgoal.description}'},
                    {'action': AgentAction.TOOL.value, 'description': f'Execute: {subgoal.description}'},
                    {'action': AgentAction.VERIFY.value, 'description': f'Verify: {subgoal.description}'}
                ])

        # Synthesis
        steps.append({'action': AgentAction.SYNTHESIZE.value, 'description': 'Combine results'})

        return steps

    async def _execute_with_reflection(
        self,
        plan: Plan,
        trajectory: ReasoningTrajectory,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute plan with periodic reflection"""

        results = []
        step_count = 0

        for i, planned_step in enumerate(plan.steps):
            if step_count >= self.max_steps:
                break

            # Execute step
            result = await self._execute_planned_step(
                planned_step,
                trajectory,
                context
            )
            results.append(result)
            step_count += 1

            # Periodic reflection
            if (i + 1) % self.reflection_frequency == 0:
                reflection = await self._reflect(
                    trajectory,
                    results[-self.reflection_frequency:],
                    context
                )

                # Adjust if needed
                if not reflection.should_continue:
                    break

                if reflection.adjustments_needed:
                    # Could dynamically adjust plan here
                    context['adjustments'] = reflection.adjustments_needed

        return {
            'result': results[-1] if results else None,
            'steps_executed': step_count,
            'results': results
        }

    async def _execute_planned_step(
        self,
        planned_step: Dict[str, Any],
        trajectory: ReasoningTrajectory,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single planned step"""

        action = AgentAction(planned_step['action'])
        description = planned_step['description']

        # Map action to step type
        step_type_map = {
            AgentAction.THINK: StepType.THINK,
            AgentAction.PLAN: StepType.THINK,
            AgentAction.TOOL: StepType.TOOL,
            AgentAction.RETRIEVE: StepType.RETRIEVE,
            AgentAction.VERIFY: StepType.VERIFY,
            AgentAction.REFLECT: StepType.REFLECT,
            AgentAction.DECOMPOSE: StepType.THINK,
            AgentAction.SYNTHESIZE: StepType.THINK
        }

        step_type = step_type_map.get(action, StepType.THINK)

        # Create step
        step = trajectory.add_step(
            content=description,
            step_type=step_type,
            confidence=0.8,
            metadata={'planned_action': str(action.value)}
        )

        # Execute based on action type
        if action == AgentAction.TOOL:
            result = await self._execute_tool_step(step, context)
            step.tool_result = result
        elif action == AgentAction.RETRIEVE:
            result = await self._execute_retrieval_step(step, context)
            step.tool_result = result
        elif action == AgentAction.VERIFY:
            result = await self._execute_verification_step(step, trajectory, context)
            step.verification_result = result
        else:
            result = {'executed': True, 'action': action.value}

        # Save step
        self.tracker.save_step(trajectory.trajectory_id, step)

        # Score step with PRM
        step_reward = await self.prm.score_step(
            step,
            context={'previous_steps': trajectory.steps[:-1]}
        )

        return {
            'step_id': step.step_id,
            'action': action.value,
            'result': result,
            'reward': step_reward.overall_score
        }

    async def _execute_tool_step(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> Any:
        """Execute a tool"""

        # Simulated tool execution
        # In production, would select and execute actual tools
        return {
            'status': 'success',
            'output': 'Tool executed successfully',
            'simulated': True
        }

    async def _execute_retrieval_step(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> Any:
        """Execute information retrieval"""

        # Simulated retrieval
        return {
            'status': 'success',
            'documents': ['doc1', 'doc2'],
            'simulated': True
        }

    async def _execute_verification_step(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory,
        context: Dict[str, Any]
    ) -> bool:
        """Execute verification"""

        # Use PRM to verify quality
        recent_steps = trajectory.steps[-5:]  # Last 5 steps

        total_quality = 0.0
        for s in recent_steps:
            if s.step_id != step.step_id:
                reward = await self.prm.score_step(s, context)
                total_quality += reward.overall_score

        avg_quality = total_quality / max(len(recent_steps) - 1, 1)

        # Pass if average quality > 0.7
        return avg_quality > 0.7

    async def _reflect(
        self,
        trajectory: ReasoningTrajectory,
        recent_results: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> ReflectionResult:
        """Reflect on recent progress"""

        # Add reflection step
        reflection_step = trajectory.add_step(
            content=f"Reflect on last {len(recent_results)} steps",
            step_type=StepType.REFLECT,
            confidence=0.85,
            metadata={'action': str(AgentAction.REFLECT.value)}
        )
        self.tracker.save_step(trajectory.trajectory_id, reflection_step)

        # Analyze recent results
        successes = [r for r in recent_results if r.get('reward', 0) > 0.7]
        failures = [r for r in recent_results if r.get('reward', 0) < 0.5]

        avg_reward = sum(r.get('reward', 0) for r in recent_results) / len(recent_results)

        # Generate reflection
        what_worked = [
            f"High-quality step: {r['action']}"
            for r in successes[:3]
        ]

        what_failed = [
            f"Low-quality step: {r['action']}"
            for r in failures[:3]
        ]

        lessons = []
        if avg_reward < 0.6:
            lessons.append("Need to improve step quality")
        if len(failures) > len(successes):
            lessons.append("Too many low-quality steps - adjust approach")

        adjustments = []
        if avg_reward < 0.5:
            adjustments.append("Consider alternative strategy")

        return ReflectionResult(
            what_worked=what_worked,
            what_failed=what_failed,
            lessons_learned=lessons,
            adjustments_needed=adjustments,
            confidence_change=avg_reward - 0.7,  # Relative to threshold
            should_continue=len(trajectory.steps) < self.max_steps and avg_reward > 0.3,
            metadata={'avg_reward': avg_reward}
        )

    async def _verify_solution(
        self,
        trajectory: ReasoningTrajectory,
        goal: Goal,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Final verification of solution"""

        # Add final verification step
        verify_step = trajectory.add_step(
            content=f"Final verification: {goal.description}",
            step_type=StepType.VERIFY,
            confidence=0.9,
            metadata={'action': str(AgentAction.VERIFY.value), 'final': True}
        )

        # Score entire trajectory
        traj_reward = await self.prm.score_trajectory(trajectory)

        # Success if final score > 0.7
        success = traj_reward.final_score > 0.7

        verify_step.verification_result = success
        self.tracker.save_step(trajectory.trajectory_id, verify_step)

        return {
            'success': success,
            'final_score': traj_reward.final_score,
            'temporal_coherence': traj_reward.temporal_coherence,
            'efficiency': traj_reward.efficiency_bonus
        }


# Singleton instance
_agentic_reasoner_instance = None


def get_agentic_reasoner(
    tracker: Optional[TrajectoryTracker] = None,
    resilient_reasoner: Optional[ResilientReasoner] = None,
    prm: Optional[ProcessRewardModel] = None,
    max_steps: int = 50,
    reflection_frequency: int = 5
) -> AgenticReasoner:
    """Get singleton agentic reasoner instance"""
    global _agentic_reasoner_instance
    if _agentic_reasoner_instance is None:
        _agentic_reasoner_instance = AgenticReasoner(
            tracker, resilient_reasoner, prm, max_steps, reflection_frequency
        )
    return _agentic_reasoner_instance
