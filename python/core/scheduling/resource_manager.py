"""Production safety facade: concurrency + budget + circuit breakers.

Composes existing building blocks — no new algorithms:
- SchedulerCoordinator._slots (ConcurrencySlot per-provider)
- BudgetTracker (daily caps per-provider + global)
- CircuitBreaker (per-provider CLOSED→OPEN→HALF_OPEN state machine)
"""

import logging
from typing import Any, Dict

from core.exceptions_unified import (
    CircuitBreaker,
    CircuitBreakerConfig,
)
from core.middleware.budget_tracker import BudgetAction, BudgetTracker
from core.scheduling.scheduler_coordinator import SchedulerCoordinator

logger = logging.getLogger(__name__)


class ResourceManager:
    """Thin facade that gates task dispatch through three safety checks.

    Usage in the hot path:
        1. provider_available(p) — circuit breaker allows requests?
        2. acquire(p)           — global + per-provider concurrency slot
        3. check_budget(p, est) — daily spend cap
        4. ... execute ...
        5. record_success/failure(p) — update breaker state
        6. record_spend(p, ...)      — update budget
        7. release(p)                — free concurrency slot
    """

    def __init__(
        self,
        scheduler: SchedulerCoordinator,
        budget_tracker: BudgetTracker,
        max_global_concurrent: int = 10,
    ) -> None:
        self._scheduler = scheduler
        self._budget = budget_tracker
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._global_active: int = 0
        self._max_global: int = max_global_concurrent

    # ── Concurrency ──────────────────────────────────────────────────

    def can_dispatch(self, provider: str = "default") -> bool:
        """Non-mutating check: could we dispatch to this provider right now?"""
        if self._global_active >= self._max_global:
            return False
        slot = self._scheduler._slots.get(provider)
        if slot is None:
            return True  # no slot yet → will be created on acquire
        return slot._active < slot.max_concurrent

    def acquire(self, provider: str = "default") -> bool:
        """Acquire global + per-provider concurrency slot. Returns False if full."""
        if self._global_active >= self._max_global:
            return False
        if not self._scheduler.acquire_slot(provider):
            return False
        self._global_active += 1
        return True

    def release(self, provider: str = "default") -> None:
        """Release global + per-provider concurrency slot."""
        self._scheduler.release_slot(provider)
        self._global_active = max(0, self._global_active - 1)

    # ── Budget ───────────────────────────────────────────────────────

    def check_budget(self, provider: str, estimated_cost: float) -> BudgetAction:
        """Delegate to existing BudgetTracker."""
        return self._budget.check_budget(provider, estimated_cost)

    def record_spend(
        self,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str = "general",
    ) -> None:
        """Record actual spend after execution."""
        self._budget.record_usage(provider, input_tokens, output_tokens, task_type)

    # ── Circuit Breaker ──────────────────────────────────────────────

    def get_breaker(self, provider: str) -> CircuitBreaker:
        """Get or create a per-provider circuit breaker."""
        if provider not in self._breakers:
            self._breakers[provider] = CircuitBreaker(
                name=f"provider:{provider}",
                config=CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout_sec=60,
                    success_threshold=2,
                ),
            )
        return self._breakers[provider]

    def provider_available(self, provider: str) -> bool:
        """Check if the provider's circuit breaker allows requests."""
        return self.get_breaker(provider).can_execute()

    def record_success(self, provider: str) -> None:
        self.get_breaker(provider).record_success()

    def record_failure(self, provider: str) -> None:
        self.get_breaker(provider).record_failure()

    # ── Dashboard ────────────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "global_active": self._global_active,
            "max_global_concurrent": self._max_global,
            "concurrency": self._scheduler.get_concurrency_status(),
            "budget": self._budget.get_dashboard(),
            "circuit_breakers": {
                name: cb.get_status() for name, cb in self._breakers.items()
            },
        }
