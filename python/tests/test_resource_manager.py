"""Tests for ResourceManager production safety facade.

Covers three subsystems:
- Concurrency: global + per-provider slot management
- Budget: delegation to BudgetTracker
- Circuit breaker: per-provider CLOSED→OPEN→HALF_OPEN→CLOSED
Plus integration scenarios combining all three.
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock

from core.exceptions_unified import CircuitBreakerState
from core.middleware.budget_tracker import BudgetAction, BudgetTracker
from core.scheduling.scheduler_coordinator import ConcurrencySlot, SchedulerCoordinator
from core.scheduling.resource_manager import ResourceManager


# ── Helpers ──────────────────────────────────────────────────────────


def _make_scheduler() -> SchedulerCoordinator:
    """Create a SchedulerCoordinator with mocked sub-components."""
    resolver = MagicMock()
    tracker = MagicMock()
    policy = MagicMock()
    retry_mgr = MagicMock()
    return SchedulerCoordinator(resolver, tracker, policy, retry_mgr)


def _make_rm(max_global: int = 10, daily_budget: float = 100.0) -> ResourceManager:
    scheduler = _make_scheduler()
    budget = BudgetTracker(daily_budget=daily_budget)
    return ResourceManager(scheduler, budget, max_global_concurrent=max_global)


# =========================================================================
# Concurrency tests
# =========================================================================


class TestConcurrency:
    def test_acquire_within_global_limit(self):
        rm = _make_rm(max_global=3)
        assert rm.acquire("provA") is True
        assert rm.acquire("provA") is True
        assert rm._global_active == 2

    def test_acquire_exceeds_global_limit(self):
        rm = _make_rm(max_global=2)
        assert rm.acquire("a") is True
        assert rm.acquire("b") is True
        assert rm.acquire("c") is False  # global full
        assert rm._global_active == 2

    def test_release_frees_global_slot(self):
        rm = _make_rm(max_global=1)
        assert rm.acquire("x") is True
        assert rm.acquire("x") is False
        rm.release("x")
        assert rm.acquire("x") is True

    def test_per_provider_limit_respected(self):
        rm = _make_rm(max_global=20)
        # Configure provider "slow" with max 2 concurrent
        rm._scheduler._slots["slow"] = ConcurrencySlot("slow", max_concurrent=2)
        assert rm.acquire("slow") is True
        assert rm.acquire("slow") is True
        assert rm.acquire("slow") is False  # provider full
        # Different provider still works
        assert rm.acquire("fast") is True

    def test_global_and_provider_both_checked(self):
        rm = _make_rm(max_global=1)
        assert rm.acquire("a") is True
        # Provider "b" has capacity but global is full
        assert rm.acquire("b") is False

    def test_can_dispatch_mirrors_acquire(self):
        rm = _make_rm(max_global=1)
        assert rm.can_dispatch("a") is True
        rm.acquire("a")
        assert rm.can_dispatch("a") is False  # global full

    def test_release_below_zero_safe(self):
        rm = _make_rm(max_global=5)
        rm.release("phantom")
        rm.release("phantom")
        assert rm._global_active == 0  # never goes negative

    def test_multiple_providers_independent(self):
        rm = _make_rm(max_global=10)
        rm._scheduler._slots["alpha"] = ConcurrencySlot("alpha", max_concurrent=1)
        rm._scheduler._slots["beta"] = ConcurrencySlot("beta", max_concurrent=1)
        assert rm.acquire("alpha") is True
        assert rm.acquire("alpha") is False  # alpha full
        assert rm.acquire("beta") is True   # beta independent


# =========================================================================
# Budget tests
# =========================================================================


class TestBudget:
    def test_check_budget_delegates(self):
        rm = _make_rm(daily_budget=100.0)
        action = rm.check_budget("gpt-5.3-codex", 0.01)
        assert action == BudgetAction.ALLOW

    def test_check_budget_reject(self):
        rm = _make_rm(daily_budget=0.001)
        rm._budget.default_action = BudgetAction.REJECT
        # Spend almost everything
        rm._budget.record_usage("gpt-5.3-codex", 100_000, 100_000)
        action = rm.check_budget("gpt-5.3-codex", 10.0)
        assert action == BudgetAction.REJECT

    def test_check_budget_downgrade(self):
        rm = _make_rm(daily_budget=999.0)
        # Exhaust provider-specific budget (gpt-5.3-codex default = $40)
        rm._budget._daily_spend["gpt-5.3-codex"] = 39.99
        action = rm.check_budget("gpt-5.3-codex", 1.0)
        assert action == BudgetAction.DOWNGRADE

    def test_record_spend_delegates(self):
        rm = _make_rm()
        rm.record_spend("gemini-3-flash", input_tokens=1000, output_tokens=500, task_type="code")
        dashboard = rm._budget.get_dashboard()
        assert dashboard["total_spent_today"] > 0
        assert "gemini-3-flash" in dashboard["provider_spend"]

    def test_check_budget_allow(self):
        rm = _make_rm(daily_budget=100.0)
        action = rm.check_budget("ollama", 0.0)
        assert action == BudgetAction.ALLOW

    def test_budget_integration_with_concurrency(self):
        rm = _make_rm(max_global=5, daily_budget=100.0)
        # Both checks pass
        assert rm.acquire("gpt-5.3-codex") is True
        assert rm.check_budget("gpt-5.3-codex", 0.01) == BudgetAction.ALLOW


# =========================================================================
# Circuit breaker tests
# =========================================================================


class TestCircuitBreaker:
    def test_provider_available_initially_true(self):
        rm = _make_rm()
        assert rm.provider_available("new-provider") is True

    def test_failures_open_circuit(self):
        rm = _make_rm()
        for _ in range(5):
            rm.record_failure("flaky")
        assert rm.provider_available("flaky") is False

    def test_success_resets_in_half_open(self):
        rm = _make_rm()
        breaker = rm.get_breaker("recover")
        # Force open
        for _ in range(5):
            breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        # Manually transition to half-open for testing
        breaker._half_open()
        assert breaker.state == CircuitBreakerState.HALF_OPEN
        # Two successes (threshold=2) should close it
        breaker.record_success()
        breaker.record_success()
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_half_open_after_timeout(self):
        rm = _make_rm()
        breaker = rm.get_breaker("timeout-test")
        # Open the breaker
        for _ in range(5):
            breaker.record_failure()
        assert breaker.state == CircuitBreakerState.OPEN
        # Pretend recovery_timeout has passed
        breaker.last_state_change = datetime.now(tz=timezone.utc) - timedelta(seconds=120)
        # can_execute should transition to HALF_OPEN
        assert breaker.can_execute() is True
        assert breaker.state == CircuitBreakerState.HALF_OPEN

    def test_per_provider_isolation(self):
        rm = _make_rm()
        # Open breaker for provider A
        for _ in range(5):
            rm.record_failure("a")
        assert rm.provider_available("a") is False
        # Provider B is unaffected
        assert rm.provider_available("b") is True

    def test_record_success_in_half_open_closes(self):
        rm = _make_rm()
        breaker = rm.get_breaker("closing")
        breaker._open()
        breaker._half_open()
        breaker.record_success()
        breaker.record_success()
        assert breaker.state == CircuitBreakerState.CLOSED

    def test_get_breaker_lazy_creation(self):
        rm = _make_rm()
        assert "lazy" not in rm._breakers
        breaker = rm.get_breaker("lazy")
        assert "lazy" in rm._breakers
        assert breaker.name == "provider:lazy"

    def test_breaker_status_in_stats(self):
        rm = _make_rm()
        rm.get_breaker("stats-test")  # force creation
        stats = rm.stats
        assert "stats-test" in stats["circuit_breakers"]
        assert stats["circuit_breakers"]["stats-test"]["state"] == "closed"


# =========================================================================
# Integration tests
# =========================================================================


class TestIntegration:
    def test_full_lifecycle(self):
        rm = _make_rm(max_global=5)
        provider = "gpt-5.3-codex"

        # 1. Check availability
        assert rm.provider_available(provider) is True
        # 2. Acquire slot
        assert rm.acquire(provider) is True
        # 3. Check budget
        assert rm.check_budget(provider, 0.01) == BudgetAction.ALLOW
        # 4. (simulate execution)
        # 5. Record success
        rm.record_success(provider)
        # 6. Record spend
        rm.record_spend(provider, input_tokens=500, output_tokens=200)
        # 7. Release slot
        rm.release(provider)
        assert rm._global_active == 0

    def test_circuit_open_blocks_dispatch(self):
        rm = _make_rm(max_global=10)
        provider = "broken"
        for _ in range(5):
            rm.record_failure(provider)
        # Circuit is open — provider_available returns False
        assert rm.provider_available(provider) is False
        # Other providers still work
        assert rm.provider_available("healthy") is True

    def test_budget_reject_before_execution(self):
        rm = _make_rm(daily_budget=0.001)
        rm._budget.default_action = BudgetAction.REJECT
        rm._budget.record_usage("gpt-5.3-codex", 1_000_000, 1_000_000)
        action = rm.check_budget("gpt-5.3-codex", 1.0)
        assert action == BudgetAction.REJECT

    def test_stats_dashboard_complete(self):
        rm = _make_rm()
        rm.acquire("prov1")
        rm.get_breaker("prov1")
        stats = rm.stats
        assert "global_active" in stats
        assert "max_global_concurrent" in stats
        assert "concurrency" in stats
        assert "budget" in stats
        assert "circuit_breakers" in stats
        assert stats["global_active"] == 1

    def test_concurrent_providers(self):
        rm = _make_rm(max_global=10)
        providers = ["alpha", "beta", "gamma"]
        for p in providers:
            assert rm.acquire(p) is True
        assert rm._global_active == 3
        # Release in different order
        rm.release("beta")
        assert rm._global_active == 2
        rm.release("alpha")
        rm.release("gamma")
        assert rm._global_active == 0

    def test_idempotent_release(self):
        rm = _make_rm(max_global=5)
        rm.acquire("x")
        rm.release("x")
        rm.release("x")  # double release
        rm.release("x")  # triple release
        assert rm._global_active == 0
