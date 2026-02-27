"""
Helix Process Reward Models (PRM)
Step-level reward calculation for reasoning trajectories
Based on OpenAI o1/o3, ThinkPRM, and Multi-Layer GRPO research (2025)
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from core.reasoning.trajectory_tracker import (
    ReasoningStep,
    ReasoningTrajectory,
    StepType,
    TrajectoryTracker
)


class RewardDimension(str, Enum):
    """Dimensions for step-level reward calculation"""
    LOGICAL_CONSISTENCY = "logical_consistency"
    FACTUAL_ACCURACY = "factual_accuracy"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"


@dataclass
class StepReward:
    """Reward for a single reasoning step"""
    step_id: str
    overall_score: float                           # 0.0 to 1.0
    component_scores: Dict[str, float]             # Score per dimension
    verification_cot: str                          # Chain-of-thought explanation
    confidence: float                              # Confidence in scoring
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryReward:
    """Aggregated reward for entire trajectory"""
    trajectory_id: str
    overall_score: float                           # 0.0 to 1.0
    step_scores: List[StepReward]                  # Individual step rewards
    temporal_coherence: float                      # Smoothness of reasoning
    efficiency_bonus: float                        # Bonus for efficient reasoning
    penalty: float                                 # Penalty for errors/backtracks
    final_score: float                             # Overall + coherence + efficiency - penalty
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessRewardModel:
    """
    Process Reward Model (PRM) for step-level scoring

    Features:
    - Step-level reward calculation across 5 dimensions
    - Verification chain-of-thought generation
    - Trajectory-level aggregation with temporal coherence
    - Research-backed scoring methodology
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize Process Reward Model

        Args:
            weights: Dimension weights (default: equal weighting)
            confidence_threshold: Minimum confidence for valid steps
        """
        self.weights = weights or {
            RewardDimension.LOGICAL_CONSISTENCY: 0.25,
            RewardDimension.FACTUAL_ACCURACY: 0.20,
            RewardDimension.RELEVANCE: 0.20,
            RewardDimension.COMPLETENESS: 0.20,
            RewardDimension.CLARITY: 0.15
        }
        self.confidence_threshold = confidence_threshold

    async def score_step(
        self,
        step: ReasoningStep,
        context: Optional[Dict[str, Any]] = None
    ) -> StepReward:
        """
        Score individual reasoning step across all dimensions

        Args:
            step: Reasoning step to score
            context: Optional context (previous steps, problem, etc.)

        Returns:
            StepReward with component and overall scores
        """
        context = context or {}

        # Calculate component scores
        scores = {
            'logical_consistency': await self._check_logical_consistency(step, context),
            'factual_accuracy': await self._check_factual_accuracy(step, context),
            'relevance': await self._check_relevance(step, context),
            'completeness': await self._check_completeness(step, context),
            'clarity': await self._check_clarity(step, context)
        }

        # Calculate weighted overall score
        overall_score = sum(
            scores[dim.value] * self.weights[dim]
            for dim in RewardDimension
        )

        # Generate verification chain-of-thought
        verification_cot = self._generate_verification_cot(step, scores, context)

        # Confidence based on step confidence and score variance
        score_variance = self._calculate_variance(list(scores.values()))
        confidence = min(step.confidence, 1.0 - score_variance)

        return StepReward(
            step_id=step.step_id,
            overall_score=overall_score,
            component_scores=scores,
            verification_cot=verification_cot,
            confidence=confidence,
            metadata={
                'step_type': step.step_type.value,
                'original_confidence': step.confidence
            }
        )

    async def score_trajectory(
        self,
        trajectory: ReasoningTrajectory,
        context: Optional[Dict[str, Any]] = None
    ) -> TrajectoryReward:
        """
        Score entire reasoning trajectory with aggregation

        Args:
            trajectory: Complete reasoning trajectory
            context: Optional additional context

        Returns:
            TrajectoryReward with step-level and trajectory-level scores
        """
        context = context or {}
        context['problem'] = trajectory.problem

        # Score each step
        step_rewards = []
        for i, step in enumerate(trajectory.steps):
            step_context = {
                **context,
                'previous_steps': trajectory.steps[:i],
                'step_number': i
            }
            reward = await self.score_step(step, step_context)
            step_rewards.append(reward)

        # Calculate overall score (average of step scores)
        if step_rewards:
            overall_score = sum(r.overall_score for r in step_rewards) / len(step_rewards)
        else:
            overall_score = 0.0

        # Calculate temporal coherence
        temporal_coherence = self._calculate_temporal_coherence(step_rewards)

        # Calculate efficiency bonus
        efficiency_bonus = self._calculate_efficiency_bonus(trajectory)

        # Calculate penalty
        penalty = self._calculate_penalty(trajectory)

        # Final score with coherence and efficiency
        # Special case: empty trajectory
        if not step_rewards:
            final_score = 0.0
        else:
            final_score = (
                overall_score * 0.6 +              # Base score
                temporal_coherence * 0.2 +          # Smooth reasoning
                efficiency_bonus * 0.1 -            # Efficient path
                penalty * 0.1                       # Errors/backtracks
            )
            final_score = max(0.0, min(1.0, final_score))  # Clamp to [0, 1]

        return TrajectoryReward(
            trajectory_id=trajectory.trajectory_id,
            overall_score=overall_score,
            step_scores=step_rewards,
            temporal_coherence=temporal_coherence,
            efficiency_bonus=efficiency_bonus,
            penalty=penalty,
            final_score=final_score,
            metadata={
                'num_steps': len(trajectory.steps),
                'success': trajectory.success,
                'total_backtracks': trajectory.total_backtracks
            }
        )

    # ========== Dimension Scoring Methods ==========

    async def _check_logical_consistency(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> float:
        """
        Check logical consistency of reasoning step

        Criteria:
        - No contradiction with previous steps
        - Follows from previous conclusions
        - Internal consistency
        """
        score = 1.0

        # Check for contradictions with previous steps
        previous_steps = context.get('previous_steps', [])
        if previous_steps:
            # Simple check: verify not identical to recent steps
            recent = previous_steps[-3:]
            if any(step.content == prev.content for prev in recent):
                score -= 0.3  # Repetition penalty

            # Check for contradictory conclusions
            if step.step_type == StepType.VERIFY:
                # If previous verification passed but this one failed (or vice versa)
                prev_verifications = [s for s in recent if s.step_type == StepType.VERIFY]
                if prev_verifications:
                    last_verification = prev_verifications[-1]
                    if (step.verification_result != last_verification.verification_result
                        and step.parent_step == last_verification.parent_step):
                        score -= 0.4  # Contradiction

        # Confidence penalty (stronger)
        if step.confidence < self.confidence_threshold:
            score -= 0.4  # Increased from 0.2

        return max(0.0, min(1.0, score))

    async def _check_factual_accuracy(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> float:
        """
        Check factual accuracy of step

        Criteria:
        - Tool results are valid
        - Retrieved information is used correctly
        - No hallucinated facts
        """
        score = 1.0

        # For tool steps, check if result is valid
        if step.step_type == StepType.TOOL:
            if step.tool_result is None:
                score = 0.0  # Tool failed
            elif isinstance(step.tool_result, dict):
                # Valid structured result
                score = 0.95
            elif step.tool_result:
                # Some result returned
                score = 0.85
            else:
                score = 0.5  # Empty result

        # For retrieval steps, check if data was retrieved
        elif step.step_type == StepType.RETRIEVE:
            if step.tool_result:
                score = 0.9
            else:
                score = 0.5  # No data retrieved

        # For verification steps, trust the verification result
        elif step.step_type == StepType.VERIFY:
            if step.verification_result is True:
                score = 0.95
            elif step.verification_result is False:
                score = 0.3  # Verification failed
            else:
                score = 0.6  # Unknown

        return max(0.0, min(1.0, score))

    async def _check_relevance(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> float:
        """
        Check relevance to problem/goal

        Criteria:
        - Related to original problem
        - Contributes to solution
        - Not tangential
        """
        score = 0.7  # Base relevance score (reduced from 0.8)

        problem = context.get('problem', '')

        # Check if step is on a backtracked path (less relevant)
        if step.backtrack_from:
            score -= 0.35  # Increased penalty from 0.2

        # Check step type relevance
        if step.step_type == StepType.THINK:
            score += 0.1  # Thinking is always somewhat relevant
        elif step.step_type == StepType.TOOL:
            score += 0.15  # Tool usage is action-oriented
        elif step.step_type == StepType.VERIFY:
            score += 0.1  # Verification ensures correctness
        elif step.step_type == StepType.RETRIEVE:
            score += 0.05  # Retrieval gathers info

        # Confidence boost
        if step.confidence > 0.8:
            score += 0.1

        return max(0.0, min(1.0, score))

    async def _check_completeness(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> float:
        """
        Check completeness of reasoning step

        Criteria:
        - Sufficient detail
        - Addresses necessary aspects
        - Not overly vague
        """
        score = 0.7  # Base completeness

        # Content length as proxy for detail
        content_length = len(step.content)
        if content_length > 100:
            score += 0.2
        elif content_length > 50:
            score += 0.1
        elif content_length < 20:
            score -= 0.2  # Too brief

        # Tool usage indicates action was taken
        if step.tool_used:
            score += 0.1

        # Verification step completeness
        if step.step_type == StepType.VERIFY:
            if step.verification_result is not None:
                score += 0.15

        # Alternative consideration shows thoroughness
        if step.alternatives_considered:
            score += 0.05 * len(step.alternatives_considered)

        return max(0.0, min(1.0, score))

    async def _check_clarity(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> float:
        """
        Check clarity of reasoning step

        Criteria:
        - Clear expression
        - Unambiguous intent
        - Well-structured
        """
        score = 0.75  # Base clarity

        # High confidence usually means clear understanding
        if step.confidence > 0.85:
            score += 0.2
        elif step.confidence < 0.5:
            score -= 0.2

        # Steps with metadata show structured thinking
        if step.metadata:
            score += 0.05

        # Tool steps with results are clear
        if step.step_type == StepType.TOOL and step.tool_result:
            score += 0.1

        # Verification steps with explicit results are clear
        if step.step_type == StepType.VERIFY and step.verification_result is not None:
            score += 0.1

        return max(0.0, min(1.0, score))

    # ========== Aggregation Methods ==========

    def _calculate_temporal_coherence(self, step_rewards: List[StepReward]) -> float:
        """
        Calculate temporal coherence (smoothness of reasoning)

        Measures:
        - Smoothness: Small score changes between consecutive steps
        - Monotonicity: Preference for improving scores
        - Consistency: Low variance in scores
        """
        if len(step_rewards) < 2:
            return 1.0

        scores = [r.overall_score for r in step_rewards]

        # Smoothness: average absolute difference between consecutive steps
        differences = [abs(scores[i+1] - scores[i]) for i in range(len(scores) - 1)]
        avg_diff = sum(differences) / len(differences)
        smoothness = 1.0 - min(avg_diff, 1.0)

        # Monotonicity: count score improvements vs decreases
        improvements = sum(1 for d in [scores[i+1] - scores[i] for i in range(len(scores) - 1)] if d > 0)
        monotonicity = improvements / len(differences) if differences else 0.5

        # Consistency: inverse of variance
        variance = self._calculate_variance(scores)
        consistency = 1.0 - min(variance, 1.0)

        # Weighted combination
        coherence = (
            smoothness * 0.4 +
            monotonicity * 0.3 +
            consistency * 0.3
        )

        return coherence

    def _calculate_efficiency_bonus(self, trajectory: ReasoningTrajectory) -> float:
        """
        Calculate efficiency bonus

        Rewards:
        - Fewer steps to solution
        - Few backtracks
        - High success rate
        """
        if not trajectory.steps:
            return 0.0

        # Step efficiency (fewer is better, up to a point) - MORE STRICT
        step_count = len(trajectory.steps)
        if step_count <= 5:
            step_efficiency = 1.0
        elif step_count <= 10:
            step_efficiency = 0.7  # Reduced from 0.8
        elif step_count <= 15:
            step_efficiency = 0.5  # Reduced from 0.6
        elif step_count <= 20:
            step_efficiency = 0.35  # New tier
        else:
            step_efficiency = 0.2  # Reduced from 0.4

        # Backtrack penalty (increased)
        backtrack_penalty = min(trajectory.total_backtracks * 0.25, 0.9)  # Increased from 0.2
        backtrack_efficiency = 1.0 - backtrack_penalty

        # Success bonus
        success_bonus = 1.0 if trajectory.success else 0.0

        # Combined efficiency
        efficiency = (
            step_efficiency * 0.4 +
            backtrack_efficiency * 0.3 +
            success_bonus * 0.3
        )

        return efficiency

    def _calculate_penalty(self, trajectory: ReasoningTrajectory) -> float:
        """
        Calculate penalty for errors and inefficiencies

        Penalties:
        - Tool errors
        - Verification failures
        - Excessive backtracks
        - Low confidence steps
        """
        penalty = 0.0

        for step in trajectory.steps:
            # Tool error penalty
            if step.step_type == StepType.TOOL and step.tool_result is None:
                penalty += 0.1

            # Verification failure penalty
            if step.step_type == StepType.VERIFY and step.verification_result is False:
                penalty += 0.15

            # Low confidence penalty
            if step.confidence < self.confidence_threshold:
                penalty += 0.05

            # Backtrack penalty (already counted in total_backtracks)
            if step.backtrack_from:
                penalty += 0.08

        # Overall failure penalty
        if not trajectory.success:
            penalty += 0.3

        return min(penalty, 1.0)  # Cap at 1.0

    # ========== Helper Methods ==========

    def _generate_verification_cot(
        self,
        step: ReasoningStep,
        scores: Dict[str, float],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate verification chain-of-thought explanation

        Explains why the step received its scores
        """
        cot_parts = [f"Verification for step {step.step_id}:"]

        # Overall assessment
        avg_score = sum(scores.values()) / len(scores)
        if avg_score > 0.8:
            cot_parts.append("✓ Strong reasoning step")
        elif avg_score > 0.6:
            cot_parts.append("~ Adequate reasoning step")
        else:
            cot_parts.append("✗ Weak reasoning step")

        # Component breakdowns
        cot_parts.append("\nComponent Scores:")
        for dim, score in scores.items():
            status = "✓" if score > 0.7 else ("~" if score > 0.5 else "✗")
            cot_parts.append(f"  {status} {dim}: {score:.2f}")

        # Key observations
        observations = []

        if step.confidence < self.confidence_threshold:
            observations.append(f"Low confidence ({step.confidence:.2f})")

        if step.step_type == StepType.TOOL:
            if step.tool_result:
                observations.append(f"Tool '{step.tool_used}' executed successfully")
            else:
                observations.append(f"Tool '{step.tool_used}' failed")

        if step.backtrack_from:
            observations.append("Backtracked from previous attempt")

        if step.alternatives_considered:
            observations.append(f"Considered {len(step.alternatives_considered)} alternatives")

        if observations:
            cot_parts.append("\nObservations:")
            for obs in observations:
                cot_parts.append(f"  • {obs}")

        return "\n".join(cot_parts)

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        squared_diffs = [(v - mean) ** 2 for v in values]
        variance = sum(squared_diffs) / len(values)

        return math.sqrt(variance)  # Return standard deviation


# Singleton instance
_process_reward_model_instance = None


def get_process_reward_model(
    weights: Optional[Dict[str, float]] = None,
    confidence_threshold: float = 0.5
) -> ProcessRewardModel:
    """Get singleton process reward model instance"""
    global _process_reward_model_instance
    if _process_reward_model_instance is None:
        _process_reward_model_instance = ProcessRewardModel(weights, confidence_threshold)
    return _process_reward_model_instance
