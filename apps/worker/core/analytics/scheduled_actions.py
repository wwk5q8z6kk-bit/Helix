"""Scheduled maintenance actions for Helix."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    action: str
    success: bool
    message: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScheduledActions:
    """Periodic maintenance and health actions."""

    def __init__(self, bridge=None, analytics_engine=None):
        self._bridge = bridge
        self._analytics = analytics_engine
        self._history: List[ActionResult] = []

    async def run_stale_cleanup(self, days: int = 90) -> ActionResult:
        """Clean up stale/orphaned data."""
        now = datetime.now(tz=timezone.utc)
        try:
            cleaned = 0
            if self._bridge:
                try:
                    response = await self._bridge.request("POST", "/api/v1/maintenance/cleanup", json={"days": days})
                    cleaned = response.get("cleaned", 0)
                except Exception as e:
                    logger.warning("Rust bridge cleanup failed: %s", e)

            result = ActionResult(
                action="stale_cleanup",
                success=True,
                message=f"Cleaned {cleaned} stale entries older than {days} days",
                timestamp=now,
                metadata={"days": days, "cleaned": cleaned},
            )
        except Exception as e:
            result = ActionResult(
                action="stale_cleanup",
                success=False,
                message=f"Cleanup failed: {e}",
                timestamp=now,
                metadata={"days": days, "error": str(e)},
            )
        self._history.append(result)
        return result

    async def run_health_check(self) -> ActionResult:
        """Comprehensive system health check."""
        now = datetime.now(tz=timezone.utc)
        checks: Dict[str, bool] = {}

        # Check Rust core connectivity
        if self._bridge:
            try:
                await self._bridge.request("GET", "/api/v1/health")
                checks["rust_core"] = True
            except Exception:
                checks["rust_core"] = False
        else:
            checks["rust_core"] = False

        # Check analytics subsystem
        if self._analytics:
            try:
                await self._analytics.get_dashboard()
                checks["analytics"] = True
            except Exception:
                checks["analytics"] = False
        else:
            checks["analytics"] = False

        all_ok = all(checks.values())
        failed = [k for k, v in checks.items() if not v]
        message = "All systems healthy" if all_ok else f"Unhealthy: {', '.join(failed)}"

        result = ActionResult(
            action="health_check",
            success=all_ok,
            message=message,
            timestamp=now,
            metadata={"checks": checks},
        )
        self._history.append(result)
        return result

    async def run_cache_warming(self) -> ActionResult:
        """Pre-warm frequently accessed caches."""
        now = datetime.now(tz=timezone.utc)
        warmed = 0

        if self._bridge:
            try:
                response = await self._bridge.request("POST", "/api/v1/maintenance/warm-cache")
                warmed = response.get("warmed", 0)
            except Exception as e:
                logger.warning("Cache warming failed: %s", e)

        result = ActionResult(
            action="cache_warming",
            success=True,
            message=f"Warmed {warmed} cache entries",
            timestamp=now,
            metadata={"warmed": warmed},
        )
        self._history.append(result)
        return result

    async def run_analytics_save(self) -> ActionResult:
        """Persist analytics data to disk."""
        now = datetime.now(tz=timezone.utc)
        try:
            if self._analytics:
                self._analytics.save()
                result = ActionResult(
                    action="analytics_save",
                    success=True,
                    message="Analytics data persisted to disk",
                    timestamp=now,
                )
            else:
                result = ActionResult(
                    action="analytics_save",
                    success=False,
                    message="No analytics engine available",
                    timestamp=now,
                )
        except Exception as e:
            result = ActionResult(
                action="analytics_save",
                success=False,
                message=f"Analytics save failed: {e}",
                timestamp=now,
                metadata={"error": str(e)},
            )
        self._history.append(result)
        return result

    async def get_history(self, limit: int = 50) -> List[ActionResult]:
        return self._history[-limit:]
