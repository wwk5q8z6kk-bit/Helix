"""Tests for feedback handler and /api/feedback endpoint."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from core.feedback.feedback_handler import FeedbackHandler


# --- FeedbackHandler unit tests ---


async def test_handle_feedback_returns_recorded():
    handler = FeedbackHandler()
    result = await handler.handle_feedback(task_id="t1", score=0.9, feedback="great")
    assert result["status"] == "recorded"
    assert result["task_id"] == "t1"
    assert result["score"] == 0.9
    assert "timestamp" in result


async def test_handle_feedback_publishes_event():
    from core.event_bus import InMemoryEventBus
    from core.interfaces import EventType

    bus = InMemoryEventBus()
    received = []

    async def on_feedback(data):
        received.append(data)

    await bus.subscribe(EventType.FEEDBACK_RECEIVED, on_feedback)

    handler = FeedbackHandler(event_bus=bus)
    await handler.handle_feedback(task_id="t2", score=0.7, feedback="ok", agent_id="agent-1")

    assert len(received) == 1
    assert received[0]["task_id"] == "t2"
    assert received[0]["score"] == 0.7
    assert received[0]["agent_id"] == "agent-1"


async def test_handle_feedback_tolerates_missing_bus():
    handler = FeedbackHandler(event_bus=None)
    result = await handler.handle_feedback(task_id="t3", score=0.5)
    assert result["status"] == "recorded"


async def test_handle_feedback_tolerates_bus_error():
    bad_bus = MagicMock()
    bad_bus.publish = AsyncMock(side_effect=RuntimeError("bus down"))

    handler = FeedbackHandler(event_bus=bad_bus)
    result = await handler.handle_feedback(task_id="t4", score=0.3)
    assert result["status"] == "recorded"


# --- API endpoint tests ---


@pytest.fixture(autouse=True)
def _reset_container():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


def test_feedback_endpoint_success():
    from httpx import ASGITransport, AsyncClient
    from core.api.helix_api import app
    import asyncio

    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/feedback", json={"task_id": "t5", "score": 0.8, "feedback": "nice"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "recorded"
            assert data["task_id"] == "t5"

    asyncio.get_event_loop().run_until_complete(_run())


def test_feedback_endpoint_validates_score():
    from httpx import ASGITransport, AsyncClient
    from core.api.helix_api import app
    import asyncio

    async def _run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/feedback", json={"task_id": "t6", "score": 1.5})
            assert resp.status_code == 422  # validation error

    asyncio.get_event_loop().run_until_complete(_run())
