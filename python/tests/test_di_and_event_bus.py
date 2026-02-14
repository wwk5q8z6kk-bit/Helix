"""Tests for DI container (core/di_container.py) and event bus (core/event_bus.py)."""

import pytest
from unittest.mock import patch, MagicMock


# --- DI Container ---

@pytest.fixture(autouse=True)
def _reset():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


def test_container_singleton():
    from core.di_container import get_container
    c1 = get_container()
    c2 = get_container()
    assert c1 is c2


def test_container_status_all_uninitialized():
    from core.di_container import get_container
    container = get_container()
    s = container.status()
    assert s == {"bridge": False, "llm_router": False, "event_bus": False, "orchestrator": False, "learning_optimizer": False}


def test_container_bridge_lazy_init():
    from core.di_container import get_container
    container = get_container()
    bridge = container.bridge
    assert bridge is not None
    assert container.status()["bridge"] is True


def test_container_llm_router_lazy_init():
    from core.di_container import get_container
    container = get_container()
    router = container.llm_router
    assert router is not None
    from core.llm.intelligent_llm_router import IntelligentLLMRouter
    assert isinstance(router, IntelligentLLMRouter)


def test_container_event_bus_lazy_init():
    from core.di_container import get_container
    container = get_container()
    bus = container.event_bus
    assert bus is not None
    from core.event_bus import InMemoryEventBus
    assert isinstance(bus, InMemoryEventBus)


def test_shutdown_clears_container():
    from core.di_container import get_container, shutdown_container
    c1 = get_container()
    shutdown_container()
    from core import di_container
    assert di_container._container is None


# --- InMemoryEventBus ---


async def test_event_bus_publish_subscribe():
    from core.event_bus import InMemoryEventBus
    from core.interfaces.event_bus import EventType

    bus = InMemoryEventBus()
    received = []

    async def handler(data):
        received.append(data)

    await bus.subscribe(EventType.TASK_STARTED, handler)
    await bus.publish(EventType.TASK_STARTED, {"task_id": "t1"})

    assert len(received) == 1
    assert received[0]["task_id"] == "t1"


async def test_event_bus_multiple_subscribers():
    from core.event_bus import InMemoryEventBus
    from core.interfaces.event_bus import EventType

    bus = InMemoryEventBus()
    results_a = []
    results_b = []

    async def handler_a(data):
        results_a.append(data)

    async def handler_b(data):
        results_b.append(data)

    await bus.subscribe(EventType.TASK_COMPLETED, handler_a)
    await bus.subscribe(EventType.TASK_COMPLETED, handler_b)
    await bus.publish(EventType.TASK_COMPLETED, {"ok": True})

    assert len(results_a) == 1
    assert len(results_b) == 1


async def test_event_bus_unsubscribe():
    from core.event_bus import InMemoryEventBus
    from core.interfaces.event_bus import EventType

    bus = InMemoryEventBus()
    received = []

    async def handler(data):
        received.append(data)

    sub_id = await bus.subscribe(EventType.TASK_STARTED, handler)
    await bus.unsubscribe(sub_id)
    await bus.publish(EventType.TASK_STARTED, {"task_id": "t2"})

    assert len(received) == 0


async def test_event_bus_source_tag():
    from core.event_bus import InMemoryEventBus
    from core.interfaces.event_bus import EventType

    bus = InMemoryEventBus()
    received = []

    async def handler(data):
        received.append(data)

    await bus.subscribe(EventType.TASK_STARTED, handler)
    await bus.publish(EventType.TASK_STARTED, {"x": 1}, source="test-agent")

    assert received[0]["_source"] == "test-agent"


async def test_event_bus_handler_error_doesnt_crash():
    from core.event_bus import InMemoryEventBus
    from core.interfaces.event_bus import EventType

    bus = InMemoryEventBus()

    async def bad_handler(data):
        raise ValueError("boom")

    results = []

    async def good_handler(data):
        results.append(data)

    await bus.subscribe(EventType.TASK_STARTED, bad_handler)
    await bus.subscribe(EventType.TASK_STARTED, good_handler)
    # Should not raise — bad handler's error is logged and swallowed
    await bus.publish(EventType.TASK_STARTED, {"ok": True})
    assert len(results) == 1


async def test_event_bus_sync_handler():
    from core.event_bus import InMemoryEventBus
    from core.interfaces.event_bus import EventType

    bus = InMemoryEventBus()
    received = []

    def sync_handler(data):
        received.append(data)

    await bus.subscribe(EventType.TASK_COMPLETED, sync_handler)
    await bus.publish(EventType.TASK_COMPLETED, {"val": 42})

    assert len(received) == 1
    assert received[0]["val"] == 42
