"""Tests for the analytics engine (core/analytics/analytics_engine.py)."""

import json
import os
import pytest
from datetime import datetime, timedelta, timezone

from core.analytics.analytics_engine import (
    AnalyticsEngine,
    DashboardData,
    ProviderPerformance,
    TokenUsageEntry,
    TrendData,
)


@pytest.fixture
def engine(tmp_path):
    return AnalyticsEngine(data_dir=str(tmp_path))


# --- Recording ---


def test_record_usage(engine):
    engine.record_usage("claude", "claude-opus-4-6", tokens_in=100, tokens_out=50, cost=0.01)
    assert engine._request_count == 1
    assert len(engine._usage_log) == 1
    assert engine._usage_log[0].provider == "claude"


def test_record_usage_with_latency(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01, latency_ms=150.0)
    assert engine._latencies["claude"] == [150.0]


def test_record_error(engine):
    engine.record_error("openai", "rate limit exceeded")
    assert engine._error_count == 1
    assert engine._errors["openai"] == 1


def test_record_multiple_errors(engine):
    engine.record_error("openai", "error 1")
    engine.record_error("openai", "error 2")
    engine.record_error("claude", "error 3")
    assert engine._error_count == 3
    assert engine._errors["openai"] == 2
    assert engine._errors["claude"] == 1


# --- Dashboard ---


@pytest.mark.asyncio
async def test_empty_dashboard(engine):
    dashboard = await engine.get_dashboard()
    assert isinstance(dashboard, DashboardData)
    assert dashboard.total_requests == 0
    assert dashboard.total_tokens == 0
    assert dashboard.total_cost == 0.0
    assert dashboard.active_providers == 0
    assert dashboard.requests_today == 0
    assert len(dashboard.top_providers) == 0
    assert len(dashboard.token_trend) == 7


@pytest.mark.asyncio
async def test_dashboard_aggregates_correctly(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    engine.record_usage("openai", "gpt-4", tokens_in=200, tokens_out=100, cost=0.02)
    engine.record_usage("claude", "sonnet", tokens_in=50, tokens_out=25, cost=0.005)

    dashboard = await engine.get_dashboard()
    assert dashboard.total_requests == 3
    assert dashboard.total_tokens == 525  # 100+50+200+100+50+25
    assert dashboard.total_cost == pytest.approx(0.035)
    assert dashboard.active_providers == 2
    assert dashboard.requests_today == 3


@pytest.mark.asyncio
async def test_dashboard_today_filter(engine):
    # Record usage and manually backdate one entry
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    engine.record_usage("openai", "gpt-4", tokens_in=200, tokens_out=100, cost=0.02)

    # Backdate the first entry to yesterday
    engine._usage_log[0].timestamp = datetime.now(tz=timezone.utc) - timedelta(days=1)

    dashboard = await engine.get_dashboard()
    assert dashboard.requests_today == 1


# --- Trends ---


@pytest.mark.asyncio
async def test_trends_daily(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    trends = await engine.get_trends(period="daily", days=7)
    assert isinstance(trends, TrendData)
    assert trends.period == "daily"
    assert len(trends.data_points) == 7
    assert "tokens" in trends.data_points[-1]
    assert trends.summary.startswith("Daily")


@pytest.mark.asyncio
async def test_trends_weekly(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    trends = await engine.get_trends(period="weekly", days=14)
    assert isinstance(trends, TrendData)
    assert trends.period == "weekly"
    assert len(trends.data_points) == 2


@pytest.mark.asyncio
async def test_trends_monthly(engine):
    trends = await engine.get_trends(period="monthly", days=60)
    assert isinstance(trends, TrendData)
    assert trends.period == "monthly"
    assert len(trends.data_points) == 2


# --- Provider Performance ---


@pytest.mark.asyncio
async def test_provider_performance(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01, latency_ms=100.0)
    engine.record_usage("claude", "sonnet", tokens_in=200, tokens_out=100, cost=0.02, latency_ms=200.0)
    engine.record_usage("openai", "gpt-4", tokens_in=300, tokens_out=150, cost=0.03, latency_ms=50.0)
    engine.record_error("openai", "rate limit")

    providers = await engine.get_provider_performance()
    assert len(providers) == 2

    claude = next(p for p in providers if p.provider == "claude")
    assert claude.total_requests == 2
    assert claude.avg_latency_ms == 150.0
    assert claude.total_tokens == 450  # 100+50+200+100
    assert claude.total_cost == pytest.approx(0.03)

    openai = next(p for p in providers if p.provider == "openai")
    assert openai.total_requests == 1
    assert openai.error_rate == pytest.approx(0.5)  # 1 error / (1 request + 1 error)


@pytest.mark.asyncio
async def test_provider_performance_empty(engine):
    providers = await engine.get_provider_performance()
    assert providers == []


# --- Token Usage ---


@pytest.mark.asyncio
async def test_token_usage_unfiltered(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    engine.record_usage("openai", "gpt-4", tokens_in=200, tokens_out=100, cost=0.02)

    entries = await engine.get_token_usage()
    assert len(entries) == 2


@pytest.mark.asyncio
async def test_token_usage_filter_by_provider(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    engine.record_usage("openai", "gpt-4", tokens_in=200, tokens_out=100, cost=0.02)
    engine.record_usage("claude", "sonnet", tokens_in=50, tokens_out=25, cost=0.005)

    entries = await engine.get_token_usage(provider="claude")
    assert len(entries) == 2
    assert all(e.provider == "claude" for e in entries)


@pytest.mark.asyncio
async def test_token_usage_filter_by_date(engine):
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    engine.record_usage("openai", "gpt-4", tokens_in=200, tokens_out=100, cost=0.02)

    # Backdate first entry
    engine._usage_log[0].timestamp = datetime.now(tz=timezone.utc) - timedelta(hours=2)

    since = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    entries = await engine.get_token_usage(since=since)
    assert len(entries) == 1
    assert entries[0].provider == "openai"


@pytest.mark.asyncio
async def test_multiple_providers(engine):
    providers = ["claude", "openai", "gemini", "grok", "ollama"]
    for p in providers:
        engine.record_usage(p, f"{p}-model", tokens_in=100, tokens_out=50, cost=0.01)

    dashboard = await engine.get_dashboard()
    assert dashboard.active_providers == 5
    assert dashboard.total_requests == 5


# --- Persistence ---


def test_save_creates_file(tmp_path):
    engine = AnalyticsEngine(data_dir=str(tmp_path))
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    engine.save()

    filepath = tmp_path / "analytics.json"
    assert filepath.exists()

    data = json.loads(filepath.read_text())
    assert data["request_count"] == 1
    assert len(data["usage_log"]) == 1
    assert data["usage_log"][0]["provider"] == "claude"


def test_save_and_load_roundtrip(tmp_path):
    engine1 = AnalyticsEngine(data_dir=str(tmp_path))
    engine1.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01, latency_ms=120.0)
    engine1.record_usage("openai", "gpt-4", tokens_in=200, tokens_out=100, cost=0.02)
    engine1.record_error("openai", "rate limit")
    engine1.save()

    engine2 = AnalyticsEngine(data_dir=str(tmp_path))
    assert engine2._request_count == 2
    assert engine2._error_count == 1
    assert len(engine2._usage_log) == 2
    assert engine2._usage_log[0].provider == "claude"
    assert engine2._usage_log[1].provider == "openai"
    assert engine2._latencies["claude"] == [120.0]
    assert engine2._errors["openai"] == 1


def test_load_missing_file(tmp_path):
    # Should not crash when no file exists
    engine = AnalyticsEngine(data_dir=str(tmp_path))
    assert engine._request_count == 0
    assert len(engine._usage_log) == 0


def test_load_corrupted_file(tmp_path):
    filepath = tmp_path / "analytics.json"
    filepath.write_text("not valid json{{{")

    # Should not crash on invalid JSON
    engine = AnalyticsEngine(data_dir=str(tmp_path))
    assert engine._request_count == 0
    assert len(engine._usage_log) == 0


def test_save_creates_directory(tmp_path):
    nested = str(tmp_path / "nested" / "dir")
    engine = AnalyticsEngine(data_dir=nested)
    engine.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    engine.save()

    assert os.path.exists(os.path.join(nested, "analytics.json"))
