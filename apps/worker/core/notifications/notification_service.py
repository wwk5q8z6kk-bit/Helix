from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from enum import Enum
import uuid


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Notification:
    id: str
    title: str
    body: str
    severity: Severity = Severity.INFO
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    read: bool = False

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)


class NotificationService:
    """Local notification service for the Python AI layer.

    Stores notifications in memory and provides filtering/retrieval.
    In production this can be extended to forward to the Rust core
    notification router via the bridge.
    """

    def __init__(self, max_stored: int = 1000, bridge=None):
        self._notifications: List[Notification] = []
        self._max_stored = max_stored
        self._bridge = bridge

    async def send(self, notification: Notification) -> None:
        """Store a notification."""
        self._notifications.append(notification)
        # Evict oldest when over capacity
        while len(self._notifications) > self._max_stored:
            self._notifications.pop(0)

    async def list(
        self,
        severity: Optional[Severity] = None,
        read: Optional[bool] = None,
        limit: int = 50,
    ) -> List[Notification]:
        """List notifications with optional filters, newest first.

        When a Rust bridge is configured, attempts to fetch from Rust core first
        and falls back to the local in-memory store on failure.
        """
        if self._bridge is not None:
            try:
                raw = await self._bridge.list_notifications(
                    severity=severity.value if severity else None,
                    unread_only=(read is False),
                )
                # Convert dicts to Notification objects
                return [
                    Notification(
                        id=n.get("id", ""),
                        title=n.get("title", ""),
                        body=n.get("body", ""),
                        severity=Severity(n.get("severity", "info")),
                        source=n.get("source", ""),
                        metadata=n.get("metadata", {}),
                        read=n.get("read", False),
                    )
                    for n in raw[:limit]
                ]
            except Exception:
                pass  # Fall through to local store
        results = reversed(self._notifications)
        if severity is not None:
            results = [n for n in results if n.severity == severity]
        else:
            results = list(results)
        if read is not None:
            results = [n for n in results if n.read == read]
        return results[:limit]

    async def get(self, notification_id: str) -> Optional[Notification]:
        """Get a notification by ID."""
        if self._bridge is not None:
            try:
                n = await self._bridge.get_notification(notification_id)
                if n is not None:
                    return Notification(
                        id=n.get("id", ""),
                        title=n.get("title", ""),
                        body=n.get("body", ""),
                        severity=Severity(n.get("severity", "info")),
                        source=n.get("source", ""),
                        metadata=n.get("metadata", {}),
                        read=n.get("read", False),
                    )
            except Exception:
                pass
        for n in self._notifications:
            if n.id == notification_id:
                return n
        return None

    async def mark_read(self, notification_id: str) -> bool:
        """Mark a notification as read. Returns True if found."""
        if self._bridge is not None:
            try:
                await self._bridge.mark_notification_read(notification_id)
                return True
            except Exception:
                pass
        for n in self._notifications:
            if n.id == notification_id:
                n.read = True
                return True
        return False

    async def count(self) -> int:
        """Total notification count."""
        return len(self._notifications)

    async def count_unread(self) -> int:
        """Unread notification count."""
        return sum(1 for n in self._notifications if not n.read)

    def create_notification(
        self,
        title: str,
        body: str,
        severity: Severity = Severity.INFO,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Notification:
        """Create a Notification with a generated ID."""
        return Notification(
            id=str(uuid.uuid4()),
            title=title,
            body=body,
            severity=severity,
            source=source,
            metadata=metadata or {},
        )
