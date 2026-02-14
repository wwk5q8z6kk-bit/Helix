"""Tower structured logging — records system events to JSONL with ring buffer."""

import json
import logging
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_TOWER_PATH = os.path.join(
    os.environ.get("HELIX_HOME", os.path.expanduser("~/.helix")),
    "tower_events.jsonl",
)


class TowerLogHandler:
    """Structured event logger with JSONL persistence and in-memory ring buffer."""

    def __init__(self, path: Optional[str] = None, buffer_size: int = 1000):
        self._path = path or _DEFAULT_TOWER_PATH
        self._buffer: deque = deque(maxlen=buffer_size)
        # Ensure directory exists
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        event_type: str,
        source: str,
        duration_ms: Optional[float] = None,
        **metadata: Any,
    ) -> Dict[str, Any]:
        """Log a structured tower event."""
        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "event_type": event_type,
            "source": source,
        }
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        if metadata:
            entry["metadata"] = metadata

        # Add to ring buffer
        self._buffer.append(entry)

        # Append to JSONL file
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.exception("Failed to write tower event to %s", self._path)

        return entry

    def get_recent(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent events from the ring buffer."""
        limit = min(limit, len(self._buffer))
        return list(self._buffer)[-limit:]

    def get_summary(self, hours: float = 1.0) -> Dict[str, int]:
        """Count events by type within a time window."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        counts: Dict[str, int] = {}
        for entry in self._buffer:
            try:
                ts = datetime.fromisoformat(entry["timestamp"].rstrip("Z"))
                if ts >= cutoff:
                    et = entry.get("event_type", "unknown")
                    counts[et] = counts.get(et, 0) + 1
            except Exception:
                pass
        return counts
