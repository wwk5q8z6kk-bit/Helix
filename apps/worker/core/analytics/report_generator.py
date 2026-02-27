"""LLM-powered report generation for Helix."""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Report:
    id: str
    title: str
    content: str  # markdown
    report_type: str  # "summary", "trend", "activity", "health"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


class ReportGenerator:
    """Generates analytical reports using LLM and analytics data."""

    def __init__(self, analytics_engine=None, llm_router=None):
        self._analytics = analytics_engine
        self._llm = llm_router
        self._reports: Dict[str, Report] = {}

    async def generate(self, report_type: str = "summary", context: Optional[Dict[str, Any]] = None) -> Report:
        """Generate a report of the specified type."""
        report_id = str(uuid.uuid4())

        if report_type == "summary":
            content = await self._generate_summary(context)
        elif report_type == "trend":
            content = await self._generate_trend_analysis(context)
        elif report_type == "activity":
            content = await self._generate_activity_digest(context)
        elif report_type == "health":
            content = await self._generate_health_report(context)
        else:
            content = f"Unknown report type: {report_type}"

        report = Report(
            id=report_id,
            title=f"Helix {report_type.title()} Report",
            content=content,
            report_type=report_type,
            created_at=datetime.now(tz=timezone.utc),
        )
        self._reports[report_id] = report
        return report

    async def get(self, report_id: str) -> Optional[Report]:
        return self._reports.get(report_id)

    async def list_reports(self, report_type: Optional[str] = None, limit: int = 50) -> List[Report]:
        reports = list(self._reports.values())
        if report_type:
            reports = [r for r in reports if r.report_type == report_type]
        reports.sort(key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return reports[:limit]

    async def _generate_summary(self, context: Optional[Dict[str, Any]]) -> str:
        """Generate system summary using analytics + optional LLM."""
        if self._analytics:
            dashboard = await self._analytics.get_dashboard()
            summary = "# System Summary\n\n"
            summary += f"- Total requests: {dashboard.total_requests}\n"
            summary += f"- Total tokens: {dashboard.total_tokens}\n"
            summary += f"- Total cost: ${dashboard.total_cost:.2f}\n"
            summary += f"- Active providers: {dashboard.active_providers}\n"

            if self._llm and dashboard.total_requests > 0:
                try:
                    prompt = f"Summarize these system metrics in 2-3 sentences: {summary}"
                    response = await self._llm.generate(prompt=prompt, max_tokens=200)
                    summary += f"\n## AI Analysis\n{response.get('text', '')}\n"
                except Exception as e:
                    logger.warning("LLM summary failed: %s", e)

            return summary
        return "# System Summary\n\nNo analytics data available."

    async def _generate_trend_analysis(self, context: Optional[Dict[str, Any]]) -> str:
        """Generate trend analysis report."""
        if self._analytics:
            trends = await self._analytics.get_trends(period="daily", days=7)
            report = "# Trend Analysis\n\n"
            report += f"Period: {trends.period}\n\n"
            report += f"{trends.summary}\n\n"
            report += "| Date | Requests | Tokens | Cost |\n"
            report += "|------|----------|--------|------|\n"
            for dp in trends.data_points:
                date_str = dp.get("date", dp.get("period_start", ""))
                report += f"| {date_str} | {dp.get('requests', 0)} | {dp.get('tokens', 0)} | ${dp.get('cost', 0):.2f} |\n"
            return report
        return "# Trend Analysis\n\nNo analytics data available."

    async def _generate_activity_digest(self, context: Optional[Dict[str, Any]]) -> str:
        """Generate activity digest."""
        if self._analytics:
            providers = await self._analytics.get_provider_performance()
            report = "# Activity Digest\n\n"
            report += f"Active providers: {len(providers)}\n\n"
            for p in providers:
                report += f"## {p.provider}\n"
                report += f"- Requests: {p.total_requests}\n"
                report += f"- Avg latency: {p.avg_latency_ms:.1f}ms\n"
                report += f"- Error rate: {p.error_rate:.1%}\n"
                report += f"- Total cost: ${p.total_cost:.2f}\n\n"
            return report
        return "# Activity Digest\n\nNo analytics data available."

    async def _generate_health_report(self, context: Optional[Dict[str, Any]]) -> str:
        """Generate system health report."""
        if self._analytics:
            dashboard = await self._analytics.get_dashboard()
            providers = await self._analytics.get_provider_performance()

            report = "# System Health Report\n\n"
            report += f"## Overview\n"
            report += f"- Total requests processed: {dashboard.total_requests}\n"
            report += f"- Requests today: {dashboard.requests_today}\n"
            report += f"- Active providers: {dashboard.active_providers}\n\n"

            # Flag providers with high error rates
            unhealthy = [p for p in providers if p.error_rate > 0.1]
            if unhealthy:
                report += "## Warnings\n\n"
                for p in unhealthy:
                    report += f"- **{p.provider}**: error rate {p.error_rate:.1%}\n"
            else:
                report += "## Status: All providers healthy\n"

            return report
        return "# System Health Report\n\nNo analytics data available."
