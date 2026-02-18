"""Adaptive retry strategies with model escalation for Helix.

Goes beyond "retry 3 times with the same model."  Provides configurable
retry strategies that can:

- Escalate to a more capable model on each attempt
- Check budget before escalating to expensive models
- Use different agents for retries
- Respect quality gates (retry on low quality, not just failures)

The retry_manager sits between the orchestrator and the scheduler,
intercepting task failures and low-quality completions to decide the
optimal recovery path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Strategy types ───────────────────────────────────────────────────


class RetryReason(str, Enum):
    """Why a retry was triggered."""

    EXECUTION_FAILURE = "execution_failure"  # Task raised an exception
    LOW_QUALITY = "low_quality"  # Quality gate failed
    TIMEOUT = "timeout"  # Task exceeded timeout
    BUDGET_EXCEEDED = "budget_exceeded"  # Provider over budget mid-execution


class EscalationLevel(str, Enum):
    """Model capability tiers for escalation."""

    FAST = "fast"  # Cheapest/fastest (e.g. gemini-flash, gpt-4.1-mini)
    STANDARD = "standard"  # Default tier (e.g. gpt-5.3-codex, claude-sonnet)
    PREMIUM = "premium"  # High quality (e.g. claude-opus, gpt-5.3)
    REASONING = "reasoning"  # Deep reasoning (e.g. claude-opus-thinking, o3)


@dataclass(frozen=True)
class EscalationStep:
    """A single step in the model escalation ladder."""

    level: EscalationLevel
    model: str
    max_attempts: int = 1  # How many times to try at this level
    timeout_multiplier: float = 1.0  # Scale timeout for slower models


@dataclass(frozen=True)
class RetryDecision:
    """The retry manager's decision after a task failure or low quality."""

    should_retry: bool
    reason: RetryReason
    attempt: int  # Current attempt number (1-indexed)
    model: Optional[str] = None  # Model to use for retry (None = same)
    agent_id: Optional[str] = None  # Agent to use for retry (None = same)
    timeout_multiplier: float = 1.0
    budget_check_passed: bool = True
    message: str = ""


# ── Strategy definitions ─────────────────────────────────────────────


@dataclass
class RetryStrategy:
    """Configurable retry strategy with model escalation.

    The escalation ladder defines the progression of models to try.
    When a task fails or quality is too low, the strategy walks up the
    ladder, trying each level before giving up.

    Example ladder:
        1. gemini-3-flash (2 attempts) — fast and cheap
        2. gpt-5.3-codex (2 attempts) — standard quality
        3. claude-opus-4-6 (1 attempt) — premium, last resort
    """

    name: str = "default"

    # Escalation ladder — ordered from cheapest to most capable
    ladder: List[EscalationStep] = field(default_factory=list)

    # Whether to retry on low quality (not just failures)
    retry_on_low_quality: bool = True

    # Minimum quality to accept without retry
    min_acceptable_quality: float = 0.5

    # Budget check callback — if provided, called before each escalation
    # Signature: (model: str, estimated_cost: float) -> bool
    budget_checker: Optional[Callable[[str, float], bool]] = field(
        default=None, repr=False
    )

    # Max retries when no ladder is configured (0 = no retries at all)
    max_retries: int = 3

    # Estimated cost per attempt (for budget checking)
    estimated_cost_per_attempt: float = 0.01

    @property
    def total_max_attempts(self) -> int:
        """Total attempts across all escalation levels."""
        return sum(step.max_attempts for step in self.ladder) if self.ladder else self.max_retries

    def decide(
        self,
        attempt: int,
        reason: RetryReason,
        quality_score: float = 0.0,
        current_model: Optional[str] = None,
    ) -> RetryDecision:
        """Decide whether and how to retry.

        Args:
            attempt: The attempt number that just completed (1-indexed).
            reason: Why retry is being considered.
            quality_score: The quality score from the failed/low-quality attempt.
            current_model: The model used in the attempt that just completed.

        Returns:
            RetryDecision with retry parameters.
        """
        # Quality-based retry: if quality is above threshold, don't retry
        if reason == RetryReason.LOW_QUALITY:
            if not self.retry_on_low_quality:
                return RetryDecision(
                    should_retry=False,
                    reason=reason,
                    attempt=attempt,
                    message="Quality retry disabled in strategy",
                )
            if quality_score >= self.min_acceptable_quality:
                return RetryDecision(
                    should_retry=False,
                    reason=reason,
                    attempt=attempt,
                    message=f"Quality {quality_score:.2f} meets minimum {self.min_acceptable_quality:.2f}",
                )

        # No ladder — use simple retry count
        if not self.ladder:
            if attempt < self.max_retries:
                return RetryDecision(
                    should_retry=True,
                    reason=reason,
                    attempt=attempt,
                    message=f"Simple retry (attempt {attempt}/{self.max_retries})",
                )
            return RetryDecision(
                should_retry=False,
                reason=reason,
                attempt=attempt,
                message="Max retries exhausted (no ladder)",
            )

        # Walk the escalation ladder
        cumulative = 0
        for step in self.ladder:
            cumulative += step.max_attempts
            if attempt < cumulative:
                # Budget check
                if self.budget_checker is not None:
                    cost = self.estimated_cost_per_attempt * step.timeout_multiplier
                    if not self.budget_checker(step.model, cost):
                        logger.warning(
                            "Budget check failed for model %s — skipping escalation",
                            step.model,
                        )
                        return RetryDecision(
                            should_retry=False,
                            reason=reason,
                            attempt=attempt,
                            budget_check_passed=False,
                            message=f"Budget exceeded for {step.model}",
                        )

                logger.info(
                    "Escalating to %s (%s, attempt %d)",
                    step.model, step.level.value, attempt,
                )
                return RetryDecision(
                    should_retry=True,
                    reason=reason,
                    attempt=attempt,
                    model=step.model,
                    timeout_multiplier=step.timeout_multiplier,
                    message=f"Escalating to {step.level.value}: {step.model}",
                )

        # Exhausted all levels
        return RetryDecision(
            should_retry=False,
            reason=reason,
            attempt=attempt,
            message=f"Exhausted all {len(self.ladder)} escalation levels",
        )


# ── Pre-built strategies ─────────────────────────────────────────────


def strategy_fast_to_premium() -> RetryStrategy:
    """Flash → Codex → Opus escalation. Good for coding tasks."""
    return RetryStrategy(
        name="fast_to_premium",
        ladder=[
            EscalationStep(EscalationLevel.FAST, "gemini-3-flash", max_attempts=1),
            EscalationStep(EscalationLevel.STANDARD, "gpt-5.3-codex", max_attempts=2),
            EscalationStep(EscalationLevel.PREMIUM, "claude-opus-4-6", max_attempts=1, timeout_multiplier=2.0),
        ],
        retry_on_low_quality=True,
        min_acceptable_quality=0.5,
    )


def strategy_quality_first() -> RetryStrategy:
    """Start with premium, fall back to reasoning. For critical tasks."""
    return RetryStrategy(
        name="quality_first",
        ladder=[
            EscalationStep(EscalationLevel.PREMIUM, "claude-opus-4-6", max_attempts=2, timeout_multiplier=1.5),
            EscalationStep(EscalationLevel.REASONING, "claude-opus-4-6-thinking", max_attempts=1, timeout_multiplier=3.0),
        ],
        retry_on_low_quality=True,
        min_acceptable_quality=0.7,
    )


def strategy_budget_conscious() -> RetryStrategy:
    """Only cheap models, more attempts. For non-critical tasks."""
    return RetryStrategy(
        name="budget_conscious",
        ladder=[
            EscalationStep(EscalationLevel.FAST, "gemini-3-flash", max_attempts=3),
            EscalationStep(EscalationLevel.FAST, "gpt-4.1-mini", max_attempts=2),
        ],
        retry_on_low_quality=True,
        min_acceptable_quality=0.3,
    )


def strategy_no_retry() -> RetryStrategy:
    """No retries at all. Task either succeeds or fails."""
    return RetryStrategy(
        name="no_retry",
        ladder=[],
        retry_on_low_quality=False,
        max_retries=0,
    )


# ── Retry Manager ────────────────────────────────────────────────────


class RetryManager:
    """Manages retry strategies per task and tracks retry state.

    The manager is the single point of contact for the orchestrator
    when deciding what to do after a task failure or low-quality result.
    """

    def __init__(self, default_strategy: Optional[RetryStrategy] = None) -> None:
        self._default_strategy = default_strategy or strategy_fast_to_premium()
        self._task_strategies: Dict[str, RetryStrategy] = {}
        self._task_attempts: Dict[str, int] = {}
        self._retry_history: Dict[str, List[RetryDecision]] = {}

    def set_strategy(self, task_id: str, strategy: RetryStrategy) -> None:
        """Assign a specific retry strategy to a task."""
        self._task_strategies[task_id] = strategy

    def get_strategy(self, task_id: str) -> RetryStrategy:
        """Get the applicable strategy for a task."""
        return self._task_strategies.get(task_id, self._default_strategy)

    def on_failure(
        self,
        task_id: str,
        reason: RetryReason = RetryReason.EXECUTION_FAILURE,
        quality_score: float = 0.0,
        current_model: Optional[str] = None,
    ) -> RetryDecision:
        """Called when a task fails or produces low-quality output.

        Returns a RetryDecision indicating whether to retry and with what model.
        """
        attempt = self._task_attempts.get(task_id, 0) + 1
        self._task_attempts[task_id] = attempt

        strategy = self.get_strategy(task_id)
        decision = strategy.decide(
            attempt=attempt,
            reason=reason,
            quality_score=quality_score,
            current_model=current_model,
        )

        # Record in history
        self._retry_history.setdefault(task_id, []).append(decision)

        return decision

    def reset(self, task_id: str) -> None:
        """Reset retry state for a task (e.g. on successful completion)."""
        self._task_attempts.pop(task_id, None)
        self._task_strategies.pop(task_id, None)

    def get_history(self, task_id: str) -> List[RetryDecision]:
        """Get the retry decision history for a task."""
        return self._retry_history.get(task_id, [])

    @property
    def stats(self) -> Dict[str, Any]:
        """Summary statistics."""
        total_retries = sum(len(h) for h in self._retry_history.values())
        successful_retries = sum(
            1 for h in self._retry_history.values()
            for d in h if d.should_retry
        )
        return {
            "tasks_with_retries": len(self._retry_history),
            "total_retry_decisions": total_retries,
            "retries_approved": successful_retries,
            "retries_denied": total_retries - successful_retries,
            "active_tasks": len(self._task_attempts),
        }

    # ── Serialisation ────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_attempts": dict(self._task_attempts),
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetryManager":
        manager = cls()
        manager._task_attempts = dict(data.get("task_attempts", {}))
        return manager
