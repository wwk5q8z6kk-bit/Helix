"""Tests for EventRouter (core/api/event_router.py)."""

import pytest
from unittest.mock import AsyncMock


async def test_route_event_returns_status():
    from core.api.event_router import EventRouter

    router = EventRouter()
    result = await router.route_event("task_completed", "test", {"key": "val"})

    assert result["status"] == "routed"
    assert result["event_type"] == "task_completed"
    assert "event_id" in result
    assert "timestamp" in result


async def test_route_event_calls_sync_handler():
    from core.api.event_router import EventRouter

    received = []

    def handler(data):
        received.append(data)

    router = EventRouter()
    router.register_handler("ping", handler)
    result = await router.route_event("ping", "test", {"x": 1})

    assert len(received) == 1
    assert received[0]["x"] == 1
    assert "handler" in result["handled_by"][0]


async def test_route_event_calls_async_handler():
    from core.api.event_router import EventRouter

    received = []

    async def handler(data):
        received.append(data)

    router = EventRouter()
    router.register_handler("ping", handler)
    await router.route_event("ping", "test", {"y": 2})

    assert len(received) == 1
    assert received[0]["y"] == 2


async def test_route_event_multiple_handlers():
    from core.api.event_router import EventRouter

    results_a = []
    results_b = []

    async def handler_a(data):
        results_a.append(data)

    async def handler_b(data):
        results_b.append(data)

    router = EventRouter()
    router.register_handler("multi", handler_a)
    router.register_handler("multi", handler_b)
    result = await router.route_event("multi", "test", {})

    assert len(results_a) == 1
    assert len(results_b) == 1
    assert len(result["handled_by"]) == 2


async def test_route_event_handler_error_doesnt_crash():
    from core.api.event_router import EventRouter

    async def bad_handler(data):
        raise ValueError("boom")

    good_results = []

    async def good_handler(data):
        good_results.append(data)

    router = EventRouter()
    router.register_handler("err", bad_handler)
    router.register_handler("err", good_handler)

    # Should not raise
    result = await router.route_event("err", "test", {"ok": True})
    assert len(good_results) == 1


async def test_route_event_publishes_to_event_bus():
    from core.api.event_router import EventRouter

    bus = AsyncMock()
    bus.publish = AsyncMock()

    router = EventRouter(event_bus=bus)
    await router.route_event("task_completed", "api", {"task_id": "t1"})

    bus.publish.assert_awaited_once()
    call_args = bus.publish.call_args
    # First positional arg is the EventType
    from core.interfaces.event_bus import EventType
    assert call_args.args[0] == EventType.TASK_COMPLETED


async def test_route_event_uses_system_event_for_unknown_type():
    from core.api.event_router import EventRouter

    bus = AsyncMock()
    bus.publish = AsyncMock()

    router = EventRouter(event_bus=bus)
    await router.route_event("custom_unknown_type", "ext", {})

    bus.publish.assert_awaited_once()
    from core.interfaces.event_bus import EventType
    assert bus.publish.call_args.args[0] == EventType.SYSTEM_EVENT


async def test_route_event_no_bus_ok():
    from core.api.event_router import EventRouter

    router = EventRouter(event_bus=None)
    result = await router.route_event("test", "src", {})
    assert result["status"] == "routed"


async def test_route_event_no_handlers_returns_empty_list():
    from core.api.event_router import EventRouter

    router = EventRouter()
    result = await router.route_event("unregistered", "test", {})
    assert result["handled_by"] == []
