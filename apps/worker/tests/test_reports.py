"""Tests for the report generator (core/analytics/report_generator.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

from core.analytics.analytics_engine import AnalyticsEngine, DashboardData, ProviderPerformance
from core.analytics.report_generator import Report, ReportGenerator


@pytest.fixture
def analytics():
    return AnalyticsEngine()


@pytest.fixture
def generator(analytics):
    return ReportGenerator(analytics_engine=analytics)


@pytest.fixture
def generator_no_analytics():
    return ReportGenerator()


# --- Report Generation ---


@pytest.mark.asyncio
async def test_generate_summary(generator, analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    report = await generator.generate("summary")
    assert isinstance(report, Report)
    assert report.report_type == "summary"
    assert "System Summary" in report.content
    assert "Total requests: 1" in report.content
    assert report.created_at is not None
    assert report.id is not None


@pytest.mark.asyncio
async def test_generate_summary_no_analytics(generator_no_analytics):
    report = await generator_no_analytics.generate("summary")
    assert "No analytics data available" in report.content


@pytest.mark.asyncio
async def test_generate_trend(generator, analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    report = await generator.generate("trend")
    assert report.report_type == "trend"
    assert "Trend Analysis" in report.content
    assert "Date" in report.content  # table header


@pytest.mark.asyncio
async def test_generate_activity(generator, analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01, latency_ms=100.0)

    report = await generator.generate("activity")
    assert report.report_type == "activity"
    assert "Activity Digest" in report.content
    assert "claude" in report.content


@pytest.mark.asyncio
async def test_generate_health(generator, analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    report = await generator.generate("health")
    assert report.report_type == "health"
    assert "Health Report" in report.content


@pytest.mark.asyncio
async def test_generate_unknown_type(generator):
    report = await generator.generate("foobar")
    assert "Unknown report type: foobar" in report.content


@pytest.mark.asyncio
async def test_get_report(generator, analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    report = await generator.generate("summary")
    fetched = await generator.get(report.id)
    assert fetched is not None
    assert fetched.id == report.id


@pytest.mark.asyncio
async def test_get_missing_report(generator):
    result = await generator.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_list_reports(generator, analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)
    await generator.generate("summary")
    await generator.generate("trend")
    await generator.generate("health")

    all_reports = await generator.list_reports()
    assert len(all_reports) == 3

    summary_only = await generator.list_reports(report_type="summary")
    assert len(summary_only) == 1
    assert summary_only[0].report_type == "summary"


@pytest.mark.asyncio
async def test_health_report_flags_unhealthy_providers(generator, analytics):
    analytics.record_usage("badprovider", "model", tokens_in=100, tokens_out=50, cost=0.01)
    # Record many errors to create high error rate
    for _ in range(5):
        analytics.record_error("badprovider", "timeout")

    report = await generator.generate("health")
    assert "Warnings" in report.content
    assert "badprovider" in report.content


@pytest.mark.asyncio
async def test_summary_with_llm(analytics):
    analytics.record_usage("claude", "opus", tokens_in=100, tokens_out=50, cost=0.01)

    mock_llm = AsyncMock()
    mock_llm.generate.return_value = {"text": "AI-generated analysis here."}

    gen = ReportGenerator(analytics_engine=analytics, llm_router=mock_llm)
    report = await gen.generate("summary")
    assert "AI Analysis" in report.content
    assert "AI-generated analysis here." in report.content
