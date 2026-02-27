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
    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["rust_core_connected"] is False


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


# --- Scheduling endpoints ---

@pytest.fixture
def sched_app(mock_bridge, mock_router):
    """Create app with mocked orchestrator that has scheduler methods."""
    from core.api.helix_api import app as real_app
    from core.di_container import get_container

    container = get_container()
    container._bridge = mock_bridge
    container._llm_router = mock_router

    # Mock the orchestrator's scheduler methods
    orch = MagicMock()
    orch.define_workflow = AsyncMock(return_value={
        "name": "wf1", "task_count": 2, "wave_count": 2,
        "max_parallelism": 1, "task_order": ["a", "b"], "waves": [["a"], ["b"]],
    })
    orch.list_workflows = AsyncMock(return_value=[{"name": "wf1"}, {"name": "wf2"}])
    orch.get_workflow = AsyncMock(return_value={"name": "wf1", "tasks": []})
    orch.delete_workflow = AsyncMock(return_value=True)
    orch.preview_workflow = AsyncMock(return_value={"feasible": True, "waves": 2})
    orch.list_workflow_templates = AsyncMock(return_value=["code_review_pipeline", "research_pipeline"])
    orch.preview_template = AsyncMock(return_value={"name": "code_review_pipeline", "waves": 3})
    orch.scheduler_stats = AsyncMock(return_value={"total": 5, "ready": 2})
    orch.execution_waves = AsyncMock(return_value=[["a"], ["b", "c"]])
    container._orchestrator = orch
    return real_app


@pytest.fixture
async def sched_client(sched_app):
    transport = ASGITransport(app=sched_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_define_workflow(sched_client):
    resp = await sched_client.post("/api/scheduler/workflows", json={
        "name": "wf1",
        "tasks": [{"task_id": "a", "task_type": "code_generation"}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "wf1"
    assert data["task_count"] == 2


async def test_define_workflow_failure(sched_client):
    from core.di_container import get_container
    get_container()._orchestrator.define_workflow.return_value = None
    resp = await sched_client.post("/api/scheduler/workflows", json={
        "name": "bad", "tasks": [],
    })
    assert resp.status_code == 400


async def test_list_workflows(sched_client):
    resp = await sched_client.get("/api/scheduler/workflows")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2


async def test_get_workflow(sched_client):
    resp = await sched_client.get("/api/scheduler/workflows/wf1")
    assert resp.status_code == 200
    assert resp.json()["name"] == "wf1"


async def test_get_workflow_not_found(sched_client):
    from core.di_container import get_container
    get_container()._orchestrator.get_workflow.return_value = None
    resp = await sched_client.get("/api/scheduler/workflows/nope")
    assert resp.status_code == 404


async def test_delete_workflow(sched_client):
    resp = await sched_client.delete("/api/scheduler/workflows/wf1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "deleted"


async def test_delete_workflow_not_found(sched_client):
    from core.di_container import get_container
    get_container()._orchestrator.delete_workflow.return_value = False
    resp = await sched_client.delete("/api/scheduler/workflows/nope")
    assert resp.status_code == 404


async def test_preview_workflow(sched_client):
    resp = await sched_client.post("/api/scheduler/workflows/wf1/preview")
    assert resp.status_code == 200
    assert resp.json()["feasible"] is True


async def test_preview_workflow_not_found(sched_client):
    from core.di_container import get_container
    get_container()._orchestrator.preview_workflow.return_value = None
    resp = await sched_client.post("/api/scheduler/workflows/nope/preview")
    assert resp.status_code == 404


async def test_list_templates(sched_client):
    resp = await sched_client.get("/api/scheduler/templates")
    assert resp.status_code == 200
    templates = resp.json()["templates"]
    assert "code_review_pipeline" in templates


async def test_preview_template(sched_client):
    resp = await sched_client.post("/api/scheduler/templates/code_review_pipeline/preview")
    assert resp.status_code == 200
    assert resp.json()["waves"] == 3


async def test_preview_template_not_found(sched_client):
    from core.di_container import get_container
    get_container()._orchestrator.preview_template.return_value = None
    resp = await sched_client.post("/api/scheduler/templates/bogus/preview")
    assert resp.status_code == 404


async def test_scheduler_stats(sched_client):
    resp = await sched_client.get("/api/scheduler/stats")
    assert resp.status_code == 200
    assert resp.json()["total"] == 5


async def test_scheduler_waves(sched_client):
    resp = await sched_client.get("/api/scheduler/waves")
    assert resp.status_code == 200
    waves = resp.json()["waves"]
    assert len(waves) == 2
    assert waves[0] == ["a"]


# --- Workflow Execution endpoints ---


@pytest.fixture
def exec_app(mock_bridge, mock_router):
    """Create app with mocked orchestrator that has execution methods."""
    from core.api.helix_api import app as real_app
    from core.di_container import get_container

    container = get_container()
    container._bridge = mock_bridge
    container._llm_router = mock_router

    mock_exec_dict = {
        "execution_id": "wfx-test123",
        "workflow_name": "my-wf",
        "status": "running",
        "task_ids": ["a", "b"],
        "progress": 0.0,
    }

    mock_wf_exec = MagicMock()
    mock_wf_exec.to_dict.return_value = mock_exec_dict

    orch = MagicMock()
    orch.run_workflow = AsyncMock(return_value=mock_wf_exec)
    orch.list_workflow_executions = MagicMock(return_value=[mock_exec_dict])
    orch.get_workflow_execution = MagicMock(return_value=mock_exec_dict)
    orch.cancel_workflow_execution = AsyncMock(return_value=True)
    container._orchestrator = orch
    return real_app


@pytest.fixture
async def exec_client(exec_app):
    transport = ASGITransport(app=exec_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_run_workflow_endpoint(exec_client):
    resp = await exec_client.post("/api/scheduler/workflows/my-wf/run", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution_id"] == "wfx-test123"
    assert data["status"] == "running"


async def test_run_workflow_not_found(exec_client):
    from core.di_container import get_container
    get_container()._orchestrator.run_workflow.side_effect = ValueError("not found")
    resp = await exec_client.post("/api/scheduler/workflows/bogus/run", json={})
    assert resp.status_code == 404


async def test_list_executions(exec_client):
    resp = await exec_client.get("/api/scheduler/executions")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 1


async def test_get_execution(exec_client):
    resp = await exec_client.get("/api/scheduler/executions/wfx-test123")
    assert resp.status_code == 200
    assert resp.json()["execution_id"] == "wfx-test123"


async def test_get_execution_not_found(exec_client):
    from core.di_container import get_container
    get_container()._orchestrator.get_workflow_execution.return_value = None
    resp = await exec_client.get("/api/scheduler/executions/bogus")
    assert resp.status_code == 404


async def test_cancel_execution(exec_client):
    resp = await exec_client.post("/api/scheduler/executions/wfx-test123/cancel")
    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelled"


async def test_cancel_execution_not_running(exec_client):
    from core.di_container import get_container
    get_container()._orchestrator.cancel_workflow_execution.return_value = False
    resp = await exec_client.post("/api/scheduler/executions/bogus/cancel")
    assert resp.status_code == 404


# --- Workflow Execution SSE Progress ---


async def test_execution_progress_not_found(exec_client):
    from core.di_container import get_container
    get_container()._orchestrator.get_workflow_execution.return_value = None
    resp = await exec_client.get("/api/scheduler/executions/bogus/progress")
    assert resp.status_code == 404


async def test_execution_progress_already_completed(exec_client):
    from core.di_container import get_container
    get_container()._orchestrator.get_workflow_execution.return_value = {
        "execution_id": "wfx-done", "status": "completed", "progress": 1.0,
    }
    resp = await exec_client.get("/api/scheduler/executions/wfx-done/progress")
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    # Body should contain the terminal event
    body = resp.text
    assert "event: completed" in body
    assert "wfx-done" in body


async def test_execution_progress_already_failed(exec_client):
    from core.di_container import get_container
    get_container()._orchestrator.get_workflow_execution.return_value = {
        "execution_id": "wfx-bad", "status": "failed", "error": "boom",
    }
    resp = await exec_client.get("/api/scheduler/executions/wfx-bad/progress")
    assert resp.status_code == 200
    body = resp.text
    assert "event: failed" in body


# --- Notification endpoints ---


@pytest.fixture
def notif_app(mock_bridge, mock_router):
    """Create app with a pre-populated notification service."""
    from core.api.helix_api import app as real_app
    from core.di_container import get_container
    from core.notifications.notification_service import NotificationService, Severity

    container = get_container()
    container._bridge = mock_bridge
    container._llm_router = mock_router

    # Create notification service without bridge so tests use local in-memory store
    svc = NotificationService()
    container._notification_service = svc
    # Pre-populate some notifications
    n1 = svc.create_notification("Workflow A completed", "All tasks done.", Severity.INFO, "orchestrator")
    n2 = svc.create_notification("Workflow B failed", "Task T3 timed out.", Severity.ERROR, "orchestrator")
    import asyncio
    asyncio.get_event_loop().run_until_complete(svc.send(n1))
    asyncio.get_event_loop().run_until_complete(svc.send(n2))
    # Store IDs for lookup tests
    real_app.state.test_notif_ids = [n1.id, n2.id]
    return real_app


@pytest.fixture
async def notif_client(notif_app):
    transport = ASGITransport(app=notif_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_list_notifications(notif_client):
    resp = await notif_client.get("/api/notifications")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 2
    assert len(data["notifications"]) == 2
    # Newest first
    assert "failed" in data["notifications"][0]["title"].lower()


async def test_list_notifications_filter_severity(notif_client):
    resp = await notif_client.get("/api/notifications?severity=error")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["notifications"]) == 1
    assert data["notifications"][0]["severity"] == "error"


async def test_list_notifications_invalid_severity(notif_client):
    resp = await notif_client.get("/api/notifications?severity=bogus")
    assert resp.status_code == 400


async def test_get_notification(notif_client, notif_app):
    nid = notif_app.state.test_notif_ids[0]
    resp = await notif_client.get(f"/api/notifications/{nid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == nid
    assert "Workflow A" in data["title"]


async def test_get_notification_not_found(notif_client):
    resp = await notif_client.get("/api/notifications/nonexistent")
    assert resp.status_code == 404


async def test_mark_notification_read(notif_client, notif_app):
    nid = notif_app.state.test_notif_ids[1]
    resp = await notif_client.post(f"/api/notifications/{nid}/read")
    assert resp.status_code == 200
    assert resp.json()["status"] == "read"

    # Verify it's marked as read
    resp2 = await notif_client.get(f"/api/notifications/{nid}")
    assert resp2.json()["read"] is True


async def test_mark_notification_read_not_found(notif_client):
    resp = await notif_client.post("/api/notifications/bogus/read")
    assert resp.status_code == 404


async def test_list_notifications_unread_only(notif_client, notif_app):
    # Mark one as read first
    nid = notif_app.state.test_notif_ids[0]
    await notif_client.post(f"/api/notifications/{nid}/read")

    resp = await notif_client.get("/api/notifications?unread_only=true")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["notifications"]) == 1
    assert data["unread"] == 1


# --- Rate limiting ---


async def test_rate_limit_header_present(client):
    """Rate limit middleware should attach X-RateLimit-Remaining header."""
    resp = await client.get("/api/stats")
    assert resp.status_code == 200
    assert "X-RateLimit-Remaining" in resp.headers


async def test_rate_limit_excluded_for_health(client):
    """Health endpoint should be excluded from rate limiting (no rate limit headers)."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    # /health is excluded from rate limiting, so no rate limit header
    assert "X-RateLimit-Remaining" not in resp.headers


# --- Source connector endpoints ---


async def test_list_sources(client, mock_bridge):
    mock_bridge.list_sources = AsyncMock(return_value=[{"id": "s1", "type": "directory"}])
    resp = await client.get("/api/sources")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["sources"]) == 1
    assert data["sources"][0]["id"] == "s1"


async def test_register_source(client, mock_bridge):
    mock_bridge.register_source = AsyncMock(return_value={"id": "s2", "type": "rss", "name": "blog"})
    resp = await client.post("/api/sources", json={"type": "rss", "name": "blog"})
    assert resp.status_code == 200
    assert resp.json()["type"] == "rss"


async def test_poll_source(client, mock_bridge):
    mock_bridge.poll_source = AsyncMock(return_value=[{"title": "doc1"}])
    resp = await client.post("/api/sources/s1/poll")
    assert resp.status_code == 200
    data = resp.json()
    assert data["source_id"] == "s1"
    assert len(data["documents"]) == 1
