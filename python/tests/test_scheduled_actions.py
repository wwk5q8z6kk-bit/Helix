"""Tests for scheduled actions (core/analytics/scheduled_actions.py)."""

import pytest
from unittest.mock import AsyncMock

from core.analytics.analytics_engine import AnalyticsEngine
from core.analytics.scheduled_actions import ActionResult, ScheduledActions


@pytest.fixture
def analytics(tmp_path):
    return AnalyticsEngine(data_dir=str(tmp_path))


@pytest.fixture
def actions(analytics):
    return ScheduledActions(analytics_engine=analytics)


@pytest.fixture
def actions_with_bridge(analytics):
    bridge = AsyncMock()
    bridge.request = AsyncMock(return_value={"cleaned": 5})
    return ScheduledActions(bridge=bridge, analytics_engine=analytics)


# --- Stale Cleanup ---


@pytest.mark.asyncio
async def test_stale_cleanup_no_bridge(actions):
    result = await actions.run_stale_cleanup(days=30)
    assert isinstance(result, ActionResult)
    assert result.action == "stale_cleanup"
    assert result.success is True
    assert "0 stale entries" in result.message


@pytest.mark.asyncio
async def test_stale_cleanup_with_bridge(actions_with_bridge):
    result = await actions_with_bridge.run_stale_cleanup(days=60)
    assert result.success is True
    assert "5 stale entries" in result.message
    assert result.metadata["cleaned"] == 5


# --- Health Check ---


@pytest.mark.asyncio
async def test_health_check_no_bridge(actions):
    result = await actions.run_health_check()
    assert isinstance(result, ActionResult)
    assert result.action == "health_check"
    # No bridge = rust_core unhealthy, but analytics should be healthy
    assert "rust_core" in result.metadata["checks"]


@pytest.mark.asyncio
async def test_health_check_with_healthy_bridge(analytics):
    bridge = AsyncMock()
    bridge.request = AsyncMock(return_value={"status": "ok"})
    sa = ScheduledActions(bridge=bridge, analytics_engine=analytics)

    # Pre-populate analytics so dashboard works
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    result = await sa.run_health_check()
    assert result.success is True
    assert result.metadata["checks"]["rust_core"] is True
    assert result.metadata["checks"]["analytics"] is True


# --- Cache Warming ---


@pytest.mark.asyncio
async def test_cache_warming_no_bridge(actions):
    result = await actions.run_cache_warming()
    assert isinstance(result, ActionResult)
    assert result.action == "cache_warming"
    assert result.success is True


# --- History ---


@pytest.mark.asyncio
async def test_history(actions):
    await actions.run_stale_cleanup()
    await actions.run_health_check()
    await actions.run_cache_warming()

    history = await actions.get_history()
    assert len(history) == 3
    assert history[0].action == "stale_cleanup"
    assert history[2].action == "cache_warming"


@pytest.mark.asyncio
async def test_history_limit(actions):
    for _ in range(10):
        await actions.run_cache_warming()

    history = await actions.get_history(limit=3)
    assert len(history) == 3


# --- Analytics Save ---


@pytest.mark.asyncio
async def test_analytics_save(actions, analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    result = await actions.run_analytics_save()
    assert result.action == "analytics_save"
    assert result.success is True
    assert "persisted" in result.message


@pytest.mark.asyncio
async def test_analytics_save_no_engine():
    sa = ScheduledActions(bridge=None, analytics_engine=None)
    result = await sa.run_analytics_save()
    assert result.action == "analytics_save"
    assert result.success is False
    assert "No analytics engine" in result.message
