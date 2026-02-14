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
