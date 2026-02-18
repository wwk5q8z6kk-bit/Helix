"""Analytics engine for Helix platform metrics and insights."""

import json
import logging
import os
import tempfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TokenUsageEntry:
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    cost: float
    timestamp: datetime


@dataclass
class ProviderPerformance:
    provider: str
    total_requests: int
    avg_latency_ms: float
    error_rate: float
    total_tokens: int
    total_cost: float


@dataclass
class DashboardData:
    total_requests: int
    total_tokens: int
    total_cost: float
    active_providers: int
    requests_today: int
    top_providers: List[ProviderPerformance]
    token_trend: List[Dict[str, Any]]  # [{date, tokens, cost}]


@dataclass
class TrendData:
    period: str  # "daily", "weekly", "monthly"
    data_points: List[Dict[str, Any]]
    summary: str


class AnalyticsEngine:
    """Collects and analyzes platform usage metrics."""

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = data_dir or os.environ.get("HELIX_HOME", os.path.expanduser("~/.helix"))
        self._usage_log: List[TokenUsageEntry] = []
        self._request_count: int = 0
        self._error_count: int = 0
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._errors: Dict[str, int] = defaultdict(int)
        self.load()

    def record_usage(
        self,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        cost: float,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a single LLM usage event."""
        entry = TokenUsageEntry(
            provider=provider,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            timestamp=datetime.now(tz=timezone.utc),
        )
        self._usage_log.append(entry)
        self._request_count += 1
        if latency_ms > 0:
            self._latencies[provider].append(latency_ms)
        logger.debug("Recorded usage: provider=%s model=%s tokens=%d cost=%.4f", provider, model, tokens_in + tokens_out, cost)

    def record_error(self, provider: str, error: str) -> None:
        """Record a provider error."""
        self._error_count += 1
        self._errors[provider] += 1
        logger.warning("Provider error: provider=%s error=%s", provider, error)

    async def get_dashboard(self) -> DashboardData:
        """Aggregate dashboard view of system metrics."""
        now = datetime.now(tz=timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        total_tokens = sum(e.tokens_in + e.tokens_out for e in self._usage_log)
        total_cost = sum(e.cost for e in self._usage_log)

        today_entries = [e for e in self._usage_log if e.timestamp >= today_start]
        requests_today = len(today_entries)

        providers = set(e.provider for e in self._usage_log)
        top_providers = await self.get_provider_performance()

        # Build 7-day token trend
        token_trend: List[Dict[str, Any]] = []
        for i in range(6, -1, -1):
            day = (now - timedelta(days=i)).date()
            day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
            day_end = day_start + timedelta(days=1)
            day_entries = [e for e in self._usage_log if day_start <= e.timestamp < day_end]
            token_trend.append({
                "date": day.isoformat(),
                "tokens": sum(e.tokens_in + e.tokens_out for e in day_entries),
                "cost": sum(e.cost for e in day_entries),
            })

        return DashboardData(
            total_requests=self._request_count,
            total_tokens=total_tokens,
            total_cost=total_cost,
            active_providers=len(providers),
            requests_today=requests_today,
            top_providers=top_providers,
            token_trend=token_trend,
        )

    async def get_trends(self, period: str = "daily", days: int = 30) -> TrendData:
        """Token usage and cost trends over time."""
        now = datetime.now(tz=timezone.utc)
        data_points: List[Dict[str, Any]] = []

        if period == "weekly":
            num_periods = days // 7
            for i in range(num_periods - 1, -1, -1):
                period_end = now - timedelta(weeks=i)
                period_start = period_end - timedelta(weeks=1)
                entries = [e for e in self._usage_log if period_start <= e.timestamp < period_end]
                data_points.append({
                    "period_start": period_start.date().isoformat(),
                    "period_end": period_end.date().isoformat(),
                    "requests": len(entries),
                    "tokens": sum(e.tokens_in + e.tokens_out for e in entries),
                    "cost": sum(e.cost for e in entries),
                })
        elif period == "monthly":
            num_periods = days // 30
            for i in range(num_periods - 1, -1, -1):
                period_end = now - timedelta(days=30 * i)
                period_start = period_end - timedelta(days=30)
                entries = [e for e in self._usage_log if period_start <= e.timestamp < period_end]
                data_points.append({
                    "period_start": period_start.date().isoformat(),
                    "period_end": period_end.date().isoformat(),
                    "requests": len(entries),
                    "tokens": sum(e.tokens_in + e.tokens_out for e in entries),
                    "cost": sum(e.cost for e in entries),
                })
        else:  # daily
            for i in range(days - 1, -1, -1):
                day = (now - timedelta(days=i)).date()
                day_start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
                day_end = day_start + timedelta(days=1)
                entries = [e for e in self._usage_log if day_start <= e.timestamp < day_end]
                data_points.append({
                    "date": day.isoformat(),
                    "requests": len(entries),
                    "tokens": sum(e.tokens_in + e.tokens_out for e in entries),
                    "cost": sum(e.cost for e in entries),
                })

        total_tokens = sum(dp.get("tokens", 0) for dp in data_points)
        total_cost = sum(dp.get("cost", 0.0) for dp in data_points)
        summary = f"{period.title()} trend over {days} days: {total_tokens} tokens, ${total_cost:.2f} cost"

        return TrendData(period=period, data_points=data_points, summary=summary)

    async def get_provider_performance(self) -> List[ProviderPerformance]:
        """Per-provider performance breakdown."""
        provider_entries: Dict[str, List[TokenUsageEntry]] = defaultdict(list)
        for entry in self._usage_log:
            provider_entries[entry.provider].append(entry)

        result: List[ProviderPerformance] = []
        for provider, entries in provider_entries.items():
            total_requests = len(entries)
            latencies = self._latencies.get(provider, [])
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            total_errors = self._errors.get(provider, 0)
            error_rate = total_errors / (total_requests + total_errors) if (total_requests + total_errors) > 0 else 0.0
            total_tokens = sum(e.tokens_in + e.tokens_out for e in entries)
            total_cost = sum(e.cost for e in entries)

            result.append(ProviderPerformance(
                provider=provider,
                total_requests=total_requests,
                avg_latency_ms=avg_latency,
                error_rate=error_rate,
                total_tokens=total_tokens,
                total_cost=total_cost,
            ))

        result.sort(key=lambda p: p.total_requests, reverse=True)
        return result

    async def get_token_usage(
        self, provider: Optional[str] = None, since: Optional[datetime] = None
    ) -> List[TokenUsageEntry]:
        """Detailed token usage log."""
        entries = self._usage_log
        if provider:
            entries = [e for e in entries if e.provider == provider]
        if since:
            entries = [e for e in entries if e.timestamp >= since]
        return entries

    def save(self) -> None:
        """Persist analytics data to disk as JSON."""
        os.makedirs(self._data_dir, exist_ok=True)
        filepath = os.path.join(self._data_dir, "analytics.json")

        usage_dicts = []
        for entry in self._usage_log:
            d = asdict(entry)
            d["timestamp"] = entry.timestamp.isoformat()
            usage_dicts.append(d)

        data = {
            "usage_log": usage_dicts,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "latencies": dict(self._latencies),
            "errors": dict(self._errors),
        }

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(dir=self._data_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp_path, filepath)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.info("Analytics data saved to %s (%d entries)", filepath, len(self._usage_log))

    def load(self) -> None:
        """Load analytics data from disk if available."""
        filepath = os.path.join(self._data_dir, "analytics.json")
        if not os.path.exists(filepath):
            return

        try:
            with open(filepath) as f:
                data = json.load(f)

            self._usage_log = [
                TokenUsageEntry(
                    provider=d["provider"],
                    model=d["model"],
                    tokens_in=d["tokens_in"],
                    tokens_out=d["tokens_out"],
                    cost=d["cost"],
                    timestamp=datetime.fromisoformat(d["timestamp"]),
                )
                for d in data.get("usage_log", [])
            ]
            self._request_count = data.get("request_count", 0)
            self._error_count = data.get("error_count", 0)

            latencies = data.get("latencies", {})
            self._latencies = defaultdict(list, {k: list(v) for k, v in latencies.items()})

            errors = data.get("errors", {})
            self._errors = defaultdict(int, {k: int(v) for k, v in errors.items()})

            logger.info("Analytics data loaded from %s (%d entries)", filepath, len(self._usage_log))
        except Exception as e:
            logger.warning("Failed to load analytics data from %s: %s", filepath, e)
