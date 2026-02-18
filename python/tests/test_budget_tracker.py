"""Tests for the budget tracker (core/middleware/budget_tracker.py)."""

import pytest
from datetime import date
from unittest.mock import patch

from core.middleware.budget_tracker import (
    BudgetTracker,
    BudgetAction,
    ProviderCostConfig,
    UsageRecord,
)


@pytest.fixture
def tracker():
    return BudgetTracker(daily_budget=10.0, default_action="reject")


# --- Initialization ---

def test_default_costs_initialized(tracker):
    """Should initialize cost configs for all known providers."""
    assert len(tracker._provider_configs) == 20
    assert "claude-opus-4-6" in tracker._provider_configs
    assert "ollama" in tracker._provider_configs


def test_default_budget_value():
    t = BudgetTracker()
    assert t.daily_budget == 100.0
    assert t.default_action == BudgetAction.DOWNGRADE


# --- Cost estimation ---

def test_estimate_cost_known_provider(tracker):
    cost = tracker.estimate_cost("claude-opus-4-6", input_tokens=1000, output_tokens=500)
    # 1 * 0.015 + 0.5 * 0.075 = 0.015 + 0.0375 = 0.0525
    assert cost == pytest.approx(0.0525, rel=1e-4)


def test_estimate_cost_unknown_provider(tracker):
    cost = tracker.estimate_cost("nonexistent-provider", input_tokens=1000, output_tokens=500)
    assert cost == 0.0


def test_ollama_free(tracker):
    cost = tracker.estimate_cost("ollama", input_tokens=10000, output_tokens=5000)
    assert cost == 0.0


# --- Budget checking ---

def test_check_budget_allows_under_limit(tracker):
    action = tracker.check_budget("claude-haiku-4-5", estimated_cost=0.01)
    assert action == BudgetAction.ALLOW


def test_check_budget_rejects_over_global_limit(tracker):
    """When total spend would exceed daily_budget, default action is returned."""
    # tracker daily_budget = 10.0, default_action = reject
    action = tracker.check_budget("claude-opus-4-6", estimated_cost=11.0)
    assert action == BudgetAction.REJECT


def test_check_budget_downgrades_over_provider_limit(tracker):
    """When provider spend exceeds its per-provider daily budget, should downgrade."""
    # claude-opus-4-6 has daily_budget=30.0, but our tracker daily_budget=10.0
    # Use a provider with low budget, and record enough to exceed it
    tracker._provider_configs["test-provider"] = ProviderCostConfig(
        name="test-provider",
        cost_per_1k_input_tokens=1.0,
        cost_per_1k_output_tokens=1.0,
        daily_budget=0.5,
    )
    tracker._daily_spend["test-provider"] = 0.4
    action = tracker.check_budget("test-provider", estimated_cost=0.2)
    assert action == BudgetAction.DOWNGRADE


# --- Usage recording ---

def test_record_usage(tracker):
    record = tracker.record_usage("claude-sonnet-4-5", input_tokens=500, output_tokens=200, task_type="code_gen")
    assert isinstance(record, UsageRecord)
    assert record.provider == "claude-sonnet-4-5"
    assert record.cost > 0
    assert tracker._daily_spend["claude-sonnet-4-5"] == record.cost


# --- Date reset ---

def test_date_reset_clears_spend(tracker):
    tracker.record_usage("gemini-3-flash", input_tokens=100, output_tokens=50)
    assert tracker._daily_spend.get("gemini-3-flash", 0) > 0

    # Simulate date change by directly setting _today to yesterday
    from datetime import timedelta
    tracker._today = tracker._today - timedelta(days=1)
    tracker._check_date_reset()

    assert len(tracker._daily_spend) == 0


# --- Dashboard ---

def test_dashboard_structure(tracker):
    tracker.record_usage("grok-4", input_tokens=200, output_tokens=100)
    dashboard = tracker.get_dashboard()
    assert "daily_budget" in dashboard
    assert "total_spent_today" in dashboard
    assert "budget_remaining" in dashboard
    assert "provider_spend" in dashboard
    assert "total_requests_today" in dashboard
    assert "provider_configs" in dashboard
    assert dashboard["total_requests_today"] == 1


# --- Suggest cheaper ---

def test_suggest_cheaper_provider(tracker):
    suggestion = tracker.suggest_cheaper_provider(required_quality=7.0)
    assert suggestion is not None
    # Ollama is free but quality 7.5, gemini-flash is 0.0003 with quality 8.2
    # With cost 0.0, ollama should be cheapest
    assert suggestion == "ollama"
