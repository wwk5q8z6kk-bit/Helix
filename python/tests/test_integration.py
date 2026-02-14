"""Integration tests for Python→Rust bridge and DI container wiring.

Tests that exercise real component wiring (no mocks where possible).
Tests against the live Rust server are skipped if not running.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# --- RustCoreBridge graceful failure ---


async def test_bridge_is_healthy_returns_false_when_no_server():
    """Bridge.is_healthy() should return False when Rust core is unreachable."""
    from core.rust_bridge import RustCoreBridge

    bridge = RustCoreBridge(base_url="http://127.0.0.1:19999")  # unused port
    result = await bridge.is_healthy()
    assert result is False


async def test_bridge_semantic_search_fails_gracefully():
    """semantic_search() should raise when server is not running."""
    from core.rust_bridge import RustCoreBridge

    bridge = RustCoreBridge(base_url="http://127.0.0.1:19999")
    with pytest.raises(Exception):
        await bridge.semantic_search("test query", limit=5, namespace="default")


# --- DI Container wiring ---


@pytest.fixture(autouse=True)
def _reset_container():
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


def test_container_creates_all_services():
    """Accessing all properties should not raise."""
    from core.di_container import get_container

    container = get_container()
    assert container.bridge is not None
    assert container.llm_router is not None
    assert container.event_bus is not None


def test_container_status_reports_initialized():
    """Status should reflect which services have been accessed."""
    from core.di_container import get_container

    container = get_container()
    status = container.status()
    # Nothing accessed yet
    assert status["bridge"] is False
    assert status["swarm_orchestrator"] is False
    assert status["reasoner"] is False
    assert status["feedback_handler"] is False

    # Access bridge
    _ = container.bridge
    status = container.status()
    assert status["bridge"] is True


def test_container_feedback_handler_wired_to_event_bus():
    """FeedbackHandler should receive the container's event bus."""
    from core.di_container import get_container

    container = get_container()
    handler = container.feedback_handler
    assert handler._event_bus is container.event_bus


def test_container_feedback_handler_wired_to_learning_optimizer():
    """FeedbackHandler should receive the container's learning optimizer."""
    from core.di_container import get_container

    container = get_container()
    handler = container.feedback_handler
    assert handler._learning_optimizer is container.learning_optimizer


def test_container_singleton():
    """get_container() should return the same instance."""
    from core.di_container import get_container

    c1 = get_container()
    c2 = get_container()
    assert c1 is c2


def test_container_shutdown():
    """shutdown_container() should reset the singleton."""
    from core.di_container import get_container, shutdown_container

    c1 = get_container()
    shutdown_container()
    c2 = get_container()
    assert c1 is not c2


# --- Health endpoint integration ---


async def test_health_degraded_when_rust_down():
    """Health endpoint returns 503 + degraded when Rust is unreachable."""
    from httpx import AsyncClient, ASGITransport
    from core.api.helix_api import app
    from core.di_container import get_container

    container = get_container()
    mock_bridge = AsyncMock()
    mock_bridge.is_healthy = AsyncMock(return_value=False)
    container._bridge = mock_bridge

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["rust_core_connected"] is False


async def test_health_ok_when_rust_up():
    """Health endpoint returns 200 + healthy when Rust is running."""
    from httpx import AsyncClient, ASGITransport
    from core.api.helix_api import app
    from core.di_container import get_container

    container = get_container()
    mock_bridge = AsyncMock()
    mock_bridge.is_healthy = AsyncMock(return_value=True)
    container._bridge = mock_bridge

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"


# --- New REST endpoints ---


async def test_reason_endpoint_exists():
    """POST /api/reason should be routable."""
    from httpx import AsyncClient, ASGITransport
    from core.api.helix_api import app
    from core.di_container import get_container

    container = get_container()
    mock_bridge = AsyncMock()
    mock_bridge.is_healthy = AsyncMock(return_value=True)
    container._bridge = mock_bridge

    # Mock the reasoner
    mock_traj = MagicMock()
    mock_traj.steps = []
    mock_traj.final_answer = "result"
    mock_traj.success = True
    mock_reward = MagicMock()
    mock_reward.final_score = 0.9

    mock_reasoner = MagicMock()
    mock_reasoner.solve = AsyncMock(return_value=(mock_traj, mock_reward))
    container._reasoner = mock_reasoner

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/reason", json={"goal": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["confidence"] == 0.9


async def test_swarm_endpoint_exists():
    """POST /api/swarm/execute should be routable."""
    from httpx import AsyncClient, ASGITransport
    from core.api.helix_api import app
    from core.di_container import get_container

    container = get_container()
    mock_bridge = AsyncMock()
    mock_bridge.is_healthy = AsyncMock(return_value=True)
    container._bridge = mock_bridge

    # Mock the swarm orchestrator
    mock_result = MagicMock()
    mock_result.task_id = "t1"
    mock_result.swarm_used = "implementation"
    mock_result.task_category.value = "implementation"
    mock_result.quality_score = 0.85
    mock_result.routing_confidence = 0.8
    mock_result.swarm_result.success = True
    mock_result.swarm_result.output = "Done"

    mock_orch = MagicMock()
    mock_orch.route_and_execute = AsyncMock(return_value=mock_result)
    container._swarm_orchestrator = mock_orch

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/swarm/execute", json={"task": "build API"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
