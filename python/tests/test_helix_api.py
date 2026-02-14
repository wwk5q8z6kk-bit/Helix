"""Tests for the Helix FastAPI application (core/api/helix_api.py).

Uses httpx AsyncClient via FastAPI's TestClient pattern.
All external services (Rust bridge, LLM providers) are mocked.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport


@pytest.fixture(autouse=True)
def _reset_container():
    """Reset DI container between tests."""
    from core import di_container
    di_container._container = None
    yield
    di_container._container = None


@pytest.fixture
def mock_bridge():
    bridge = AsyncMock()
    bridge.is_healthy = AsyncMock(return_value=True)
    bridge.semantic_search = AsyncMock(return_value=[{"id": "n1", "content": "hello", "score": 0.9}])
    return bridge


@pytest.fixture
def mock_router():
    from core.llm.intelligent_llm_router import LLMProvider, ProviderStats
    router = MagicMock()
    router.provider_stats = {
        LLMProvider.CLAUDE_OPUS: ProviderStats(
            provider=LLMProvider.CLAUDE_OPUS,
            success_rate=0.98, avg_latency_ms=1500,
            cost_per_1k_tokens=0.015, quality_score=9.9, availability=0.95,
        ),
    }
    router.select_provider = AsyncMock(return_value=LLMProvider.CLAUDE_OPUS)
    router.call_llm = AsyncMock(return_value=("Hello world", {"provider": "claude-opus-4-6", "tokens_used": 42, "cost": 0.001}))
    router.get_router_stats = AsyncMock(return_value={"total_requests": 0})
    return router


@pytest.fixture
def app(mock_bridge, mock_router):
    """Create app with mocked services."""
    from core.api.helix_api import app as real_app
    from core.di_container import get_container
    container = get_container()
    container._bridge = mock_bridge
    container._llm_router = mock_router
    return real_app


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# --- Health ---

async def test_health_returns_ok(client, mock_bridge):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["python_ai"] is True
    assert data["rust_core_connected"] is True
    mock_bridge.is_healthy.assert_awaited_once()


async def test_health_rust_down(client, mock_bridge):
    mock_bridge.is_healthy.return_value = False
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["rust_core_connected"] is False


# --- Models ---

async def test_list_models(client):
    resp = await client.get("/api/models")
    assert resp.status_code == 200
    providers = resp.json()["providers"]
    assert len(providers) == 1
    assert providers[0]["name"] == "claude-opus-4-6"
    assert providers[0]["quality_score"] == 9.9


# --- Generate ---

async def test_generate_success(client, mock_router):
    resp = await client.post("/api/generate", json={"prompt": "Say hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "Hello world"
    assert data["provider"] == "claude-opus-4-6"
    assert data["tokens_used"] == 42
    mock_router.select_provider.assert_awaited_once()
    mock_router.call_llm.assert_awaited_once()


async def test_generate_with_task_type(client, mock_router):
    resp = await client.post("/api/generate", json={"prompt": "test", "task_type": "research"})
    assert resp.status_code == 200
    # Verify the task type was parsed
    call_args = mock_router.select_provider.call_args
    from core.llm.intelligent_llm_router import TaskType
    assert call_args.kwargs["task_type"] == TaskType.RESEARCH


async def test_generate_invalid_task_type_falls_back(client, mock_router):
    resp = await client.post("/api/generate", json={"prompt": "test", "task_type": "nonexistent"})
    assert resp.status_code == 200  # Should not error — falls back to CODE_GENERATION


async def test_generate_llm_failure_returns_502(client, mock_router):
    mock_router.call_llm.side_effect = Exception("API down")
    resp = await client.post("/api/generate", json={"prompt": "test"})
    assert resp.status_code == 502


# --- Search ---

async def test_search_success(client, mock_bridge):
    resp = await client.post("/api/search", json={"query": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "hello"


async def test_search_with_namespace(client, mock_bridge):
    resp = await client.post("/api/search", json={"query": "q", "namespace": "custom"})
    assert resp.status_code == 200
    mock_bridge.semantic_search.assert_awaited_with(query="q", limit=10, namespace="custom")


async def test_search_failure_returns_502(client, mock_bridge):
    mock_bridge.semantic_search.side_effect = Exception("connection refused")
    resp = await client.post("/api/search", json={"query": "test"})
    assert resp.status_code == 502


# --- Stats ---

async def test_stats(client):
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    assert resp.json()["total_requests"] == 0


# --- Validation ---

async def test_generate_rejects_missing_prompt(client):
    resp = await client.post("/api/generate", json={})
    assert resp.status_code == 422


async def test_generate_rejects_bad_temperature(client):
    resp = await client.post("/api/generate", json={"prompt": "hi", "temperature": 5.0})
    assert resp.status_code == 422
