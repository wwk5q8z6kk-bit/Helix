"""Tests for the Rust core bridge (core/rust_bridge.py).

Uses aiohttp test utilities to mock HTTP responses from the Rust core.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from aiohttp import ClientSession
from core.rust_bridge import RustCoreBridge


@pytest.fixture
def bridge():
    return RustCoreBridge(base_url="http://127.0.0.1:9470")


# --- Health ---

async def test_is_healthy_true(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"status": "ok"})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    assert await bridge.is_healthy() is True


async def test_is_healthy_false_on_error(bridge):
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(side_effect=Exception("conn refused"))
    mock_session.closed = False
    bridge._session = mock_session

    assert await bridge.is_healthy() is False


async def test_is_healthy_false_on_bad_status(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"status": "degraded"})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    assert await bridge.is_healthy() is False


# --- Store ---

async def test_store_sends_post(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 201
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    await bridge.store("test-key", "test-value")
    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args
    assert call_args[0][0] == "/api/v1/nodes"
    assert call_args[1]["json"]["title"] == "test-key"
    assert call_args[1]["json"]["content"] == "test-value"


# --- Retrieve ---

async def test_retrieve_returns_content(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"nodes": [{"content": "found-it", "title": "key1"}]})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    result = await bridge.retrieve("key1")
    assert result == "found-it"


async def test_retrieve_returns_none_when_empty(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"nodes": []})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    result = await bridge.retrieve("missing-key")
    assert result is None


# --- Semantic search ---

async def test_semantic_search(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"results": [{"id": "1", "score": 0.95}]})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    results = await bridge.semantic_search("test query", limit=5, namespace="ns1")
    assert len(results) == 1
    assert results[0]["score"] == 0.95


# --- Graph ---

async def test_graph_query(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 200
    mock_resp.json = AsyncMock(return_value={"nodes": [{"id": "n1"}], "edges": [{"from": "n1", "to": "n2"}]})
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.get = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    result = await bridge.graph_query("n1", depth=3)
    assert len(result["nodes"]) == 1
    assert len(result["edges"]) == 1


async def test_graph_link_success(bridge):
    mock_resp = AsyncMock()
    mock_resp.status = 201
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    mock_session.post = MagicMock(return_value=mock_resp)
    mock_session.closed = False
    bridge._session = mock_session

    result = await bridge.graph_link("n1", "n2", "related_to")
    assert result is True


# --- Auth token ---

def test_bridge_accepts_auth_token():
    bridge = RustCoreBridge(auth_token="secret123")
    assert bridge.auth_token == "secret123"


# --- Close ---

async def test_close_session(bridge):
    mock_session = AsyncMock(spec=ClientSession)
    mock_session.closed = False
    bridge._session = mock_session

    await bridge.close()
    mock_session.close.assert_awaited_once()


# --- Scheduler ---

def _make_mock_session(method, status, json_data=None):
    """Helper: build a mock aiohttp session with one mocked HTTP method."""
    mock_resp = AsyncMock()
    mock_resp.status = status
    if json_data is not None:
        mock_resp.json = AsyncMock(return_value=json_data)
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock(spec=ClientSession)
    setattr(mock_session, method, MagicMock(return_value=mock_resp))
    mock_session.closed = False
    return mock_session


async def test_scheduler_submit_task(bridge):
    bridge._session = _make_mock_session("post", 201, {"task_id": "t1", "state": "ready"})
    result = await bridge.scheduler_submit_task("t1", dependencies=["t0"], priority=1)
    assert result is not None
    assert result["state"] == "ready"


async def test_scheduler_submit_task_failure(bridge):
    bridge._session = _make_mock_session("post", 400)
    result = await bridge.scheduler_submit_task("bad")
    assert result is None


async def test_scheduler_list_workflows(bridge):
    bridge._session = _make_mock_session("get", 200, [{"name": "wf1"}, {"name": "wf2"}])
    result = await bridge.scheduler_list_workflows()
    assert len(result) == 2


async def test_scheduler_list_workflows_error(bridge):
    bridge._session = _make_mock_session("get", 500)
    result = await bridge.scheduler_list_workflows()
    assert result == []


async def test_scheduler_get_workflow(bridge):
    bridge._session = _make_mock_session("get", 200, {"name": "wf1", "task_count": 3})
    result = await bridge.scheduler_get_workflow("wf1")
    assert result is not None
    assert result["task_count"] == 3


async def test_scheduler_get_workflow_not_found(bridge):
    bridge._session = _make_mock_session("get", 404)
    result = await bridge.scheduler_get_workflow("missing")
    assert result is None


async def test_scheduler_delete_workflow(bridge):
    bridge._session = _make_mock_session("delete", 200)
    assert await bridge.scheduler_delete_workflow("wf1") is True


async def test_scheduler_delete_workflow_not_found(bridge):
    bridge._session = _make_mock_session("delete", 404)
    assert await bridge.scheduler_delete_workflow("missing") is False


async def test_scheduler_ready_tasks(bridge):
    bridge._session = _make_mock_session("get", 200, ["t1", "t2"])
    result = await bridge.scheduler_ready_tasks()
    assert result == ["t1", "t2"]


async def test_scheduler_ready_tasks_empty(bridge):
    bridge._session = _make_mock_session("get", 200, [])
    result = await bridge.scheduler_ready_tasks()
    assert result == []


async def test_scheduler_mark_running(bridge):
    bridge._session = _make_mock_session("post", 200)
    assert await bridge.scheduler_mark_running("t1") is True


async def test_scheduler_mark_completed(bridge):
    bridge._session = _make_mock_session("post", 200, {"newly_ready": ["t2", "t3"]})
    result = await bridge.scheduler_mark_completed("t1")
    assert result == ["t2", "t3"]


async def test_scheduler_mark_failed(bridge):
    bridge._session = _make_mock_session("post", 200, {"cancelled_downstream": ["t2"]})
    result = await bridge.scheduler_mark_failed("t1")
    assert result == ["t2"]


async def test_scheduler_task_state(bridge):
    bridge._session = _make_mock_session("get", 200, {"task_id": "t1", "state": "running"})
    result = await bridge.scheduler_task_state("t1")
    assert result == "running"


async def test_scheduler_task_state_not_found(bridge):
    bridge._session = _make_mock_session("get", 404)
    result = await bridge.scheduler_task_state("missing")
    assert result is None


async def test_scheduler_execution_waves(bridge):
    bridge._session = _make_mock_session("get", 200, [["a"], ["b", "c"]])
    result = await bridge.scheduler_execution_waves()
    assert len(result) == 2
    assert result[0] == ["a"]


async def test_scheduler_stats(bridge):
    bridge._session = _make_mock_session("get", 200, {"active_tasks": 5})
    result = await bridge.scheduler_stats()
    assert result["active_tasks"] == 5


async def test_scheduler_define_workflow(bridge):
    bridge._session = _make_mock_session("post", 201, {"name": "wf1", "task_count": 2})
    result = await bridge.scheduler_define_workflow(
        name="wf1",
        tasks=[{"task_id": "a", "task_type": "planning", "priority": 1}],
    )
    assert result is not None
    assert result["name"] == "wf1"


async def test_scheduler_list_templates(bridge):
    bridge._session = _make_mock_session("get", 200, {"templates": ["code_review", "research"]})
    result = await bridge.scheduler_list_templates()
    assert "code_review" in result


# --- Sources ---

async def test_list_sources(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "s1", "name": "dir-watch"}])
    result = await bridge.list_sources()
    assert len(result) == 1
    assert result[0]["id"] == "s1"


async def test_list_sources_error(bridge):
    bridge._session = _make_mock_session("get", 500)
    result = await bridge.list_sources()
    assert result == []


async def test_register_source(bridge):
    bridge._session = _make_mock_session("post", 201, {"id": "s1", "source_type": "directory"})
    result = await bridge.register_source("directory", "my-dir", {"path": "/tmp"})
    assert result is not None
    assert result["source_type"] == "directory"


async def test_register_source_failure(bridge):
    bridge._session = _make_mock_session("post", 400)
    result = await bridge.register_source("bad", "bad")
    assert result is None


async def test_get_source_status(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "s1", "status": "healthy"})
    result = await bridge.get_source_status("s1")
    assert result["status"] == "healthy"


async def test_remove_source(bridge):
    bridge._session = _make_mock_session("delete", 204)
    assert await bridge.remove_source("s1") is True


async def test_poll_source(bridge):
    bridge._session = _make_mock_session("post", 200, [{"title": "doc1"}])
    result = await bridge.poll_source("s1")
    assert len(result) == 1


# --- Jobs ---

async def test_list_jobs(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "j1", "status": "running"}])
    result = await bridge.list_jobs()
    assert len(result) == 1
    assert result[0]["id"] == "j1"


async def test_list_jobs_error(bridge):
    bridge._session = _make_mock_session("get", 500)
    result = await bridge.list_jobs()
    assert result == []


async def test_job_stats(bridge):
    bridge._session = _make_mock_session("get", 200, {"pending": 3, "running": 1, "completed": 10})
    result = await bridge.job_stats()
    assert result["pending"] == 3


async def test_get_job(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "j1", "payload": "data"})
    result = await bridge.get_job("j1")
    assert result is not None
    assert result["id"] == "j1"


async def test_retry_job(bridge):
    bridge._session = _make_mock_session("post", 200)
    assert await bridge.retry_job("j1") is True


async def test_retry_job_not_found(bridge):
    bridge._session = _make_mock_session("post", 404)
    assert await bridge.retry_job("missing") is False


async def test_cancel_job(bridge):
    bridge._session = _make_mock_session("post", 204)
    assert await bridge.cancel_job("j1") is True


async def test_cancel_job_not_found(bridge):
    bridge._session = _make_mock_session("post", 404)
    assert await bridge.cancel_job("missing") is False


async def test_dead_letter_queue(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "j5", "error": "timeout"}])
    result = await bridge.dead_letter_queue()
    assert len(result) == 1
    assert result[0]["error"] == "timeout"


async def test_purge_jobs(bridge):
    bridge._session = _make_mock_session("post", 200)
    assert await bridge.purge_jobs() is True


# --- Workflows ---

async def test_list_workflows(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "w1", "name": "etl"}])
    result = await bridge.list_workflows()
    assert len(result) == 1
    assert result[0]["name"] == "etl"


async def test_get_workflow(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "w1", "steps": 3})
    result = await bridge.get_workflow("w1")
    assert result is not None
    assert result["steps"] == 3


async def test_execute_workflow(bridge):
    bridge._session = _make_mock_session("post", 201, {"execution_id": "e1", "status": "started"})
    result = await bridge.execute_workflow("w1", variables={"input": "data"})
    assert result is not None
    assert result["execution_id"] == "e1"


async def test_execute_workflow_failure(bridge):
    bridge._session = _make_mock_session("post", 400)
    result = await bridge.execute_workflow("bad")
    assert result is None


async def test_list_executions(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "e1"}, {"id": "e2"}])
    result = await bridge.list_executions()
    assert len(result) == 2


async def test_get_execution(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "e1", "status": "completed"})
    result = await bridge.get_execution("e1")
    assert result is not None
    assert result["status"] == "completed"


async def test_cancel_execution(bridge):
    bridge._session = _make_mock_session("post", 200)
    assert await bridge.cancel_execution("e1") is True


# --- Notifications ---

async def test_list_notifications(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "n1", "message": "alert"}])
    result = await bridge.list_notifications()
    assert len(result) == 1


async def test_list_notifications_filtered(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "n2", "severity": "warning"}])
    result = await bridge.list_notifications(severity="warning", unread_only=True)
    assert len(result) == 1
    assert result[0]["severity"] == "warning"


async def test_get_notification(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "n1", "message": "disk full"})
    result = await bridge.get_notification("n1")
    assert result is not None
    assert result["message"] == "disk full"


async def test_mark_notification_read(bridge):
    bridge._session = _make_mock_session("post", 200)
    assert await bridge.mark_notification_read("n1") is True


async def test_list_alert_rules(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "r1", "name": "cpu-high"}])
    result = await bridge.list_alert_rules()
    assert len(result) == 1
    assert result[0]["name"] == "cpu-high"


async def test_create_alert_rule(bridge):
    bridge._session = _make_mock_session("post", 201, {"id": "r1", "name": "mem-alert"})
    result = await bridge.create_alert_rule("mem-alert", {"metric": "memory", "threshold": 90}, ["email"], severity="critical")
    assert result is not None
    assert result["name"] == "mem-alert"


async def test_create_alert_rule_failure(bridge):
    bridge._session = _make_mock_session("post", 400)
    result = await bridge.create_alert_rule("bad", {}, [])
    assert result is None


async def test_update_alert_rule(bridge):
    bridge._session = _make_mock_session("put", 200)
    assert await bridge.update_alert_rule("r1", {"severity": "warning"}) is True


async def test_delete_alert_rule(bridge):
    bridge._session = _make_mock_session("delete", 204)
    assert await bridge.delete_alert_rule("r1") is True


# --- Adapters ---

async def test_list_adapters(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "a1", "type": "slack"}])
    result = await bridge.list_adapters()
    assert len(result) == 1
    assert result[0]["type"] == "slack"


async def test_register_adapter(bridge):
    bridge._session = _make_mock_session("post", 201, {"id": "a1", "adapter_type": "discord"})
    result = await bridge.register_adapter("discord", "my-discord", {"token": "abc"})
    assert result is not None
    assert result["adapter_type"] == "discord"


async def test_register_adapter_failure(bridge):
    bridge._session = _make_mock_session("post", 400)
    result = await bridge.register_adapter("bad", "bad")
    assert result is None


async def test_get_adapter_status(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "a1", "status": "connected"})
    result = await bridge.get_adapter_status("a1")
    assert result["status"] == "connected"


async def test_remove_adapter(bridge):
    bridge._session = _make_mock_session("delete", 204)
    assert await bridge.remove_adapter("a1") is True


async def test_send_message(bridge):
    bridge._session = _make_mock_session("post", 200)
    assert await bridge.send_message("a1", "hello world", channel="#general") is True


async def test_send_message_failure(bridge):
    bridge._session = _make_mock_session("post", 500)
    assert await bridge.send_message("a1", "hello") is False


async def test_adapter_health(bridge):
    bridge._session = _make_mock_session("post", 200, {"healthy": True, "latency_ms": 42})
    result = await bridge.adapter_health("a1")
    assert result["healthy"] is True


async def test_adapter_health_unhealthy(bridge):
    bridge._session = _make_mock_session("post", 503)
    result = await bridge.adapter_health("a1")
    assert result == {"healthy": False}


# --- Tunnels ---

async def test_list_tunnels(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "t1", "type": "cloudflare"}])
    result = await bridge.list_tunnels()
    assert len(result) == 1
    assert result[0]["type"] == "cloudflare"


async def test_register_tunnel(bridge):
    bridge._session = _make_mock_session("post", 201, {"id": "t1", "tunnel_type": "ngrok", "url": "https://abc.ngrok.io"})
    result = await bridge.register_tunnel("ngrok", {"port": 8080})
    assert result is not None
    assert result["tunnel_type"] == "ngrok"


async def test_register_tunnel_failure(bridge):
    bridge._session = _make_mock_session("post", 400)
    result = await bridge.register_tunnel("bad", {})
    assert result is None


async def test_get_tunnel_status(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "t1", "status": "active"})
    result = await bridge.get_tunnel_status("t1")
    assert result["status"] == "active"


async def test_remove_tunnel(bridge):
    bridge._session = _make_mock_session("delete", 204)
    assert await bridge.remove_tunnel("t1") is True


async def test_tunnel_health(bridge):
    bridge._session = _make_mock_session("post", 200, {"healthy": True})
    result = await bridge.tunnel_health("t1")
    assert result["healthy"] is True


async def test_tunnel_health_unhealthy(bridge):
    bridge._session = _make_mock_session("post", 503)
    result = await bridge.tunnel_health("t1")
    assert result == {"healthy": False}


# --- Rate Limits ---

async def test_list_rate_limits(bridge):
    bridge._session = _make_mock_session("get", 200, [{"adapter_id": "a1", "rpm": 60}])
    result = await bridge.list_rate_limits()
    assert len(result) == 1
    assert result[0]["rpm"] == 60


async def test_update_rate_limit(bridge):
    bridge._session = _make_mock_session("put", 200)
    assert await bridge.update_rate_limit("a1", 100, 1000, burst_size=20) is True


# --- API Keys ---

async def test_list_api_keys(bridge):
    bridge._session = _make_mock_session("get", 200, [{"id": "k1", "name": "prod-key"}])
    result = await bridge.list_api_keys()
    assert len(result) == 1
    assert result[0]["name"] == "prod-key"


async def test_create_api_key(bridge):
    bridge._session = _make_mock_session("post", 201, {"id": "k1", "key": "hx_abc123", "name": "test-key"})
    result = await bridge.create_api_key("test-key", scopes=["read", "write"])
    assert result is not None
    assert result["name"] == "test-key"


async def test_create_api_key_failure(bridge):
    bridge._session = _make_mock_session("post", 400)
    result = await bridge.create_api_key("bad")
    assert result is None


async def test_get_api_key(bridge):
    bridge._session = _make_mock_session("get", 200, {"id": "k1", "name": "my-key"})
    result = await bridge.get_api_key("k1")
    assert result is not None
    assert result["name"] == "my-key"


async def test_revoke_api_key(bridge):
    bridge._session = _make_mock_session("delete", 204)
    assert await bridge.revoke_api_key("k1") is True


# --- Audit ---

async def test_query_audit(bridge):
    bridge._session = _make_mock_session("get", 200, [{"action": "login", "user": "admin"}])
    result = await bridge.query_audit(filters={"action": "login"})
    assert len(result) == 1
    assert result[0]["action"] == "login"


async def test_query_audit_no_filters(bridge):
    bridge._session = _make_mock_session("get", 200, [{"action": "store"}, {"action": "delete"}])
    result = await bridge.query_audit()
    assert len(result) == 2


async def test_query_audit_error(bridge):
    bridge._session = _make_mock_session("get", 500)
    result = await bridge.query_audit()
    assert result == []
