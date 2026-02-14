"""
Helix Resilient Self-Correction System
Autonomous error detection and recovery during reasoning
Based on PokeeResearch-7B, o3 Self-Fixing, and Agent-R research (2025)
"""

import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from core.reasoning.trajectory_tracker import (
    ReasoningStep,
    ReasoningTrajectory,
    StepType,
    TrajectoryTracker
)


class ErrorType(str, Enum):
    """Types of reasoning errors"""
    LOGIC_ERROR = "logic_error"
    TOOL_ERROR = "tool_error"
    CONSTRAINT_VIOLATION = "constraint_violation"
    FACTUAL_ERROR = "factual_error"
    INCOMPLETE_REASONING = "incomplete_reasoning"
    UNKNOWN = "unknown"


class RecoveryStrategy(str, Enum):
    """Recovery strategies"""
    REFORMULATE = "reformulate"           # Reformulate tool call
    BACKTRACK = "backtrack"               # Go back and try different path
    SIMPLIFY = "simplify"                 # Break into simpler steps
    ALTERNATIVE = "alternative"           # Try alternative approach
    RETRIEVE_FACTS = "retrieve_facts"     # Get more information
    ESCALATE = "escalate"                 # Require human intervention


@dataclass
class VerificationResult:
    """Result of self-verification"""
    passed: bool
    confidence: float                     # 0.0 to 1.0
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Diagnosis:
    """Error diagnosis"""
    error_type: ErrorType
    explanation: str
    confidence: float
    suggested_fixes: List[RecoveryStrategy] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt"""
    attempt_number: int
    strategy: RecoveryStrategy
    diagnosis: Diagnosis
    success: bool
    fixed_step: Optional[ReasoningStep] = None
    error_message: Optional[str] = None


class SelfVerifier:
    """Autonomous verification of reasoning steps"""

    def __init__(self):
        self.verification_functions: List[Callable] = []

    async def verify_step(
        self,
        step: ReasoningStep,
        result: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Self-verification of reasoning step

        Checks:
        - Logical consistency
        - Constraint satisfaction
        - Confidence threshold
        - Result validity
        """
        issues = []
        checks = {}

        # Check 1: Confidence threshold
        if step.confidence < 0.5:
            issues.append(f"Low confidence: {step.confidence:.2f}")
            checks['confidence'] = False
        else:
            checks['confidence'] = True

        # Check 2: Result validity
        if result is None and step.step_type == StepType.TOOL:
            issues.append("Tool returned None result")
            checks['result_validity'] = False
        else:
            checks['result_validity'] = True

        # Check 3: Verification result (if applicable)
        if step.verification_result is not None and not step.verification_result:
            issues.append("Verification step failed")
            checks['verification'] = False
        else:
            checks['verification'] = True

        # Check 4: Context consistency
        if context:
            context_check = self._check_context_consistency(step, context)
            checks['context'] = context_check
            if not context_check:
                issues.append("Context inconsistency detected")

        # Overall pass/fail
        passed = all(checks.values())
        confidence = sum(1 for v in checks.values() if v) / len(checks)

        return VerificationResult(
            passed=passed,
            confidence=confidence,
            issues=issues,
            details=checks
        )

    def _check_context_consistency(
        self,
        step: ReasoningStep,
        context: Dict[str, Any]
    ) -> bool:
        """Check if step is consistent with context"""
        # Basic consistency check
        if 'previous_steps' in context:
            # Ensure no contradiction with previous steps
            previous = context['previous_steps']
            if len(previous) > 0:
                # Simple check: different steps shouldn't have identical content
                if any(step.content == prev.content for prev in previous[-3:]):
                    return False
        return True


class ErrorDiagnoser:
    """Diagnose reasoning errors"""

    async def diagnose_error(
        self,
        step: ReasoningStep,
        result: Any,
        verification: VerificationResult
    ) -> Diagnosis:
        """Diagnose what went wrong with the step"""

        # Analyze verification issues
        if not verification.passed:
            error_type, explanation, fixes = self._analyze_issues(
                step,
                verification.issues
            )
        else:
            # No clear verification failure, general diagnosis
            error_type = ErrorType.UNKNOWN
            explanation = "Step completed but may have issues"
            fixes = [RecoveryStrategy.ALTERNATIVE]

        confidence = 1.0 - verification.confidence

        return Diagnosis(
            error_type=error_type,
            explanation=explanation,
            confidence=confidence,
            suggested_fixes=fixes,
            metadata={'verification_details': verification.details}
        )

    def _analyze_issues(
        self,
        step: ReasoningStep,
        issues: List[str]
    ) -> tuple:
        """Analyze verification issues to determine error type"""

        # Check for tool errors
        if step.step_type == StepType.TOOL and "Tool returned None" in str(issues):
            return (
                ErrorType.TOOL_ERROR,
                "Tool execution failed or returned invalid result",
                [RecoveryStrategy.REFORMULATE, RecoveryStrategy.ALTERNATIVE]
            )

        # Check for low confidence
        if any("Low confidence" in issue for issue in issues):
            return (
                ErrorType.INCOMPLETE_REASONING,
                "Reasoning step has low confidence",
                [RecoveryStrategy.SIMPLIFY, RecoveryStrategy.RETRIEVE_FACTS]
            )

        # Check for verification failure
        if any("Verification step failed" in issue for issue in issues):
            return (
                ErrorType.LOGIC_ERROR,
                "Step verification failed",
                [RecoveryStrategy.BACKTRACK, RecoveryStrategy.ALTERNATIVE]
            )

        # Check for context inconsistency
        if any("Context inconsistency" in issue for issue in issues):
            return (
                ErrorType.CONSTRAINT_VIOLATION,
                "Step inconsistent with previous reasoning",
                [RecoveryStrategy.BACKTRACK]
            )

        # Default
        return (
            ErrorType.UNKNOWN,
            "Unknown error",
            [RecoveryStrategy.ALTERNATIVE, RecoveryStrategy.ESCALATE]
        )


class RecoveryEngine:
    """Generate and execute recovery strategies"""

    def __init__(self):
        self.max_retries = 3

    async def apply_recovery(
        self,
        step: ReasoningStep,
        diagnosis: Diagnosis,
        trajectory: ReasoningTrajectory
    ) -> Optional[ReasoningStep]:
        """
        Apply recovery strategy based on diagnosis

        Returns:
            Fixed step if recovery successful, None otherwise
        """

        for strategy in diagnosis.suggested_fixes:
            fixed_step = await self._apply_strategy(
                step,
                strategy,
                diagnosis,
                trajectory
            )

            if fixed_step:
                return fixed_step

        return None

    async def _apply_strategy(
        self,
        step: ReasoningStep,
        strategy: RecoveryStrategy,
        diagnosis: Diagnosis,
        trajectory: ReasoningTrajectory
    ) -> Optional[ReasoningStep]:
        """Apply specific recovery strategy"""

        if strategy == RecoveryStrategy.REFORMULATE:
            return await self.reformulate_step(step, diagnosis)

        elif strategy == RecoveryStrategy.BACKTRACK:
            return await self.backtrack(step, trajectory)

        elif strategy == RecoveryStrategy.SIMPLIFY:
            return await self.simplify_step(step)

        elif strategy == RecoveryStrategy.ALTERNATIVE:
            return await self.generate_alternative(step)

        elif strategy == RecoveryStrategy.RETRIEVE_FACTS:
            return await self.retrieve_facts(step)

        else:  # ESCALATE
            return None

    async def reformulate_step(
        self,
        step: ReasoningStep,
        diagnosis: Diagnosis
    ) -> Optional[ReasoningStep]:
        """Reformulate the reasoning step"""

        # Create a reformulated step with similar content but different approach
        reformulated = ReasoningStep(
            step_id=f"{step.step_id}_reformulated",
            step_number=step.step_number,
            content=f"[Reformulated] {step.content}",
            confidence=min(step.confidence + 0.1, 1.0),
            step_type=step.step_type,
            parent_step=step.parent_step,
            tool_used=step.tool_used,
            metadata={
                'reformulated_from': step.step_id,
                'diagnosis': diagnosis.explanation
            }
        )

        return reformulated

    async def backtrack(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory
    ) -> Optional[ReasoningStep]:
        """Backtrack to previous step and try alternative"""

        # Find parent step
        parent_id = step.parent_step
        if not parent_id:
            return None

        # Create alternative path from parent
        alternative = ReasoningStep(
            step_id=f"{step.step_id}_alt",
            step_number=step.step_number,
            content=f"[Alternative] Trying different approach from previous step",
            confidence=0.7,
            step_type=StepType.THINK,
            parent_step=parent_id,
            backtrack_from=step.step_id,
            metadata={'backtracked': True}
        )

        return alternative

    async def simplify_step(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Simplify the step into smaller sub-steps"""

        # Create a simplified version
        simplified = ReasoningStep(
            step_id=f"{step.step_id}_simplified",
            step_number=step.step_number,
            content=f"[Simplified] {step.content[:100]}",  # Truncate for simplicity
            confidence=min(step.confidence + 0.15, 1.0),
            step_type=step.step_type,
            parent_step=step.parent_step,
            metadata={'simplified_from': step.step_id}
        )

        return simplified

    async def generate_alternative(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Generate alternative approach"""

        alternative = ReasoningStep(
            step_id=f"{step.step_id}_alternative",
            step_number=step.step_number,
            content=f"[Alternative approach] {step.content}",
            confidence=0.75,
            step_type=step.step_type,
            parent_step=step.parent_step,
            alternatives_considered=[step.content],
            metadata={'alternative_to': step.step_id}
        )

        return alternative

    async def retrieve_facts(self, step: ReasoningStep) -> Optional[ReasoningStep]:
        """Retrieve additional facts to support reasoning"""

        retrieval_step = ReasoningStep(
            step_id=f"{step.step_id}_retrieval",
            step_number=step.step_number,
            content=f"[Retrieving facts for] {step.content[:50]}...",
            confidence=0.8,
            step_type=StepType.RETRIEVE,
            parent_step=step.step_id,
            metadata={'retrieval_for': step.step_id}
        )

        return retrieval_step


class ResilientReasoner:
    """
    Self-correcting reasoning engine

    Features:
    - Autonomous error detection
    - Self-verification at each step
    - Multiple recovery strategies
    - Resilient problem solving
    - Integration with trajectory tracking
    """

    def __init__(self, tracker: Optional[TrajectoryTracker] = None):
        """Initialize resilient reasoner"""
        self.tracker = tracker
        self.verifier = SelfVerifier()
        self.diagnoser = ErrorDiagnoser()
        self.recovery = RecoveryEngine()
        self.max_recovery_attempts = 3

    async def reason_with_recovery(
        self,
        problem: str,
        initial_steps: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_total_iterations: int = 50
    ) -> ReasoningTrajectory:
        """
        Execute reasoning with autonomous error recovery

        Uses objective completion criteria:
        - All initial steps processed
        - All recovery attempts completed (success or exhausted)
        - Maximum iteration limit reached (safety)

        Args:
            problem: Problem to solve
            initial_steps: Optional initial reasoning steps
            metadata: Additional metadata
            max_total_iterations: Maximum total processing iterations (safety limit)

        Returns:
            Complete reasoning trajectory with recovery attempts
        """

        # Create trajectory
        if self.tracker:
            trajectory = self.tracker.create_trajectory(problem, metadata)
        else:
            trajectory = ReasoningTrajectory(
                trajectory_id=f"resilient_{datetime.now().timestamp()}",
                problem=problem,
                metadata=metadata or {}
            )

        # Add initial steps if provided
        if initial_steps:
            for step_data in initial_steps:
                step = trajectory.add_step(**step_data)
                if self.tracker:
                    self.tracker.save_step(trajectory.trajectory_id, step)

        # Track recovery attempts
        recovery_attempts = []

        # Objective completion criteria trackers
        initial_steps_count = len(trajectory.steps)
        steps_processed = 0
        total_iterations = 0

        # Process each step with recovery (iterate over a copy to avoid modification during iteration)
        for i in range(initial_steps_count):
            # Safety check: max iterations reached
            if total_iterations >= max_total_iterations:
                trajectory.metadata['max_iterations_reached'] = True
                break

            step = trajectory.steps[i]
            total_iterations += 1

            # Execute step (simulated - in real use, this would call actual tools)
            result = await self._execute_step(step, trajectory)

            # Self-verify
            verification = await self.verifier.verify_step(
                step,
                result,
                context={'previous_steps': trajectory.steps[:i]}
            )

            if not verification.passed:
                # Diagnose error
                diagnosis = await self.diagnoser.diagnose_error(
                    step,
                    result,
                    verification
                )

                # Attempt recovery
                recovery_successful = False
                for attempt in range(self.max_recovery_attempts):
                    # Safety check: max iterations reached
                    if total_iterations >= max_total_iterations:
                        trajectory.metadata['max_iterations_reached'] = True
                        break

                    total_iterations += 1

                    fixed_step = await self.recovery.apply_recovery(
                        step,
                        diagnosis,
                        trajectory
                    )

                    if fixed_step:
                        # Add fixed step to trajectory
                        trajectory.steps.append(fixed_step)
                        if self.tracker:
                            self.tracker.save_step(trajectory.trajectory_id, fixed_step)

                        # Re-verify
                        fixed_result = await self._execute_step(fixed_step, trajectory)
                        fixed_verification = await self.verifier.verify_step(
                            fixed_step,
                            fixed_result
                        )

                        # Record attempt
                        recovery_attempts.append(RecoveryAttempt(
                            attempt_number=attempt + 1,
                            strategy=diagnosis.suggested_fixes[0] if diagnosis.suggested_fixes else RecoveryStrategy.ALTERNATIVE,
                            diagnosis=diagnosis,
                            success=fixed_verification.passed,
                            fixed_step=fixed_step
                        ))

                        if fixed_verification.passed:
                            recovery_successful = True
                            break

                if not recovery_successful:
                    # Recovery failed - mark trajectory
                    trajectory.metadata['recovery_failed'] = True
                    trajectory.metadata['failed_step'] = step.step_id

            steps_processed += 1

        # Store completion metrics in metadata
        trajectory.metadata['recovery_attempts'] = len(recovery_attempts)
        trajectory.metadata['successful_recoveries'] = sum(
            1 for attempt in recovery_attempts if attempt.success
        )
        trajectory.metadata['steps_processed'] = steps_processed
        trajectory.metadata['total_iterations'] = total_iterations
        trajectory.metadata['completed'] = steps_processed == initial_steps_count

        # Update trajectory in database if tracker is available
        if self.tracker:
            self.tracker.update_trajectory(trajectory)

        return trajectory

    async def _execute_step(
        self,
        step: ReasoningStep,
        trajectory: ReasoningTrajectory
    ) -> Any:
        """
        Execute a reasoning step

        In production, this would call actual tools, LLMs, etc.
        For now, it's a simulated execution.
        """

        # Simulated execution
        if step.step_type == StepType.TOOL:
            # Simulate tool execution
            # If step already has tool_result, use it; otherwise return None (simulates error)
            if step.tool_result:
                return step.tool_result
            # If no pre-existing result, simulate tool failure for testing
            return None

        elif step.step_type == StepType.VERIFY:
            # Simulate verification
            # If step already has verification_result, use it
            if step.verification_result is not None:
                return step.verification_result
            # Otherwise, high confidence steps pass, low confidence fail
            return step.confidence > 0.7

        elif step.step_type == StepType.RETRIEVE:
            # Simulate retrieval
            return {"retrieved": "simulated_data"}

        else:
            # Other step types return success
            return True

    def get_recovery_statistics(self, trajectory: ReasoningTrajectory) -> Dict[str, Any]:
        """Get statistics about recovery attempts"""

        return {
            'total_recovery_attempts': trajectory.metadata.get('recovery_attempts', 0),
            'successful_recoveries': trajectory.metadata.get('successful_recoveries', 0),
            'recovery_rate': (
                trajectory.metadata.get('successful_recoveries', 0) /
                trajectory.metadata.get('recovery_attempts', 1)
                if trajectory.metadata.get('recovery_attempts', 0) > 0
                else 0.0
            ),
            'recovery_failed': trajectory.metadata.get('recovery_failed', False)
        }


# Singleton instance
_resilient_reasoner_instance = None


def get_resilient_reasoner(tracker: Optional[TrajectoryTracker] = None) -> ResilientReasoner:
    """Get singleton resilient reasoner instance"""
    global _resilient_reasoner_instance
    if _resilient_reasoner_instance is None:
        _resilient_reasoner_instance = ResilientReasoner(tracker)
    return _resilient_reasoner_instance
