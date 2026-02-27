"""Tests for the notification service (core/notifications/notification_service.py)."""

import pytest

from core.notifications.notification_service import (
    Notification,
    NotificationService,
    Severity,
)


@pytest.fixture
def svc():
    return NotificationService(max_stored=100)


def _make(title="t", severity=Severity.INFO) -> Notification:
    import uuid
    return Notification(id=str(uuid.uuid4()), title=title, body="body", severity=severity)


# --- Send & count ---


@pytest.mark.asyncio
async def test_send_and_count(svc):
    await svc.send(_make("first"))
    assert await svc.count() == 1


# --- List newest first ---


@pytest.mark.asyncio
async def test_list_newest_first(svc):
    for i in range(3):
        await svc.send(_make(f"n{i}"))
    items = await svc.list()
    assert items[0].title == "n2"
    assert items[2].title == "n0"


# --- Filter by severity ---


@pytest.mark.asyncio
async def test_list_filter_by_severity(svc):
    await svc.send(_make("info", Severity.INFO))
    await svc.send(_make("err", Severity.ERROR))
    await svc.send(_make("info2", Severity.INFO))

    errors = await svc.list(severity=Severity.ERROR)
    assert len(errors) == 1
    assert errors[0].title == "err"


# --- Filter by read status ---


@pytest.mark.asyncio
async def test_list_filter_by_read(svc):
    n1 = _make("a")
    n2 = _make("b")
    await svc.send(n1)
    await svc.send(n2)
    await svc.mark_read(n1.id)

    unread = await svc.list(read=False)
    assert len(unread) == 1
    assert unread[0].title == "b"


# --- Get by ID ---


@pytest.mark.asyncio
async def test_get_by_id(svc):
    n = _make("hello")
    await svc.send(n)
    found = await svc.get(n.id)
    assert found is not None
    assert found.title == "hello"


@pytest.mark.asyncio
async def test_get_nonexistent(svc):
    assert await svc.get("no-such-id") is None


# --- Mark read ---


@pytest.mark.asyncio
async def test_mark_read(svc):
    n = _make("r")
    await svc.send(n)
    assert await svc.mark_read(n.id) is True
    found = await svc.get(n.id)
    assert found.read is True


@pytest.mark.asyncio
async def test_mark_read_nonexistent(svc):
    assert await svc.mark_read("no-such-id") is False


# --- Count unread ---


@pytest.mark.asyncio
async def test_count_unread(svc):
    for _ in range(3):
        await svc.send(_make())
    await svc.mark_read(svc._notifications[0].id)
    assert await svc.count_unread() == 2


# --- Eviction ---


@pytest.mark.asyncio
async def test_max_stored_eviction():
    svc = NotificationService(max_stored=3)
    for i in range(5):
        await svc.send(_make(f"n{i}"))
    assert await svc.count() == 3
    items = await svc.list()
    # Oldest (n0, n1) should have been evicted
    titles = {n.title for n in items}
    assert "n0" not in titles
    assert "n1" not in titles


# --- create_notification helper ---


def test_create_notification_helper(svc):
    n = svc.create_notification(
        title="Alert", body="Something happened", severity=Severity.CRITICAL
    )
    assert n.id  # non-empty UUID
    assert n.title == "Alert"
    assert n.severity == Severity.CRITICAL
    assert n.read is False
