"""Tests for Composio integration (core/composio/composio_bridge.py)."""

import pytest
from unittest.mock import patch, MagicMock

from core.composio.composio_bridge import ComposioBridge, ToolMode


@pytest.fixture
def bridge():
    return ComposioBridge()


# --- Initialization ---

def test_default_tool_mode_sovereign(bridge):
    """Default tool mode should be sovereign when HELIX_TOOL_MODE is not set."""
    assert bridge.tool_mode == ToolMode.SOVEREIGN


def test_tool_mode_from_env():
    with patch.dict("os.environ", {"HELIX_TOOL_MODE": "hybrid"}):
        b = ComposioBridge()
        assert b.tool_mode == ToolMode.HYBRID


def test_client_not_initialized(bridge):
    assert bridge._client is None


# --- ensure_client ---

def test_ensure_client_raises_without_composio(bridge):
    """Should raise RuntimeError when composio-core is not installed."""
    with pytest.raises(RuntimeError, match="composio-core is not installed"):
        bridge._ensure_client()


# --- discover_tools ---

async def test_discover_tools_with_mock(bridge):
    mock_tool = MagicMock()
    mock_tool.name = "GITHUB_CREATE_ISSUE"
    mock_tool.description = "Create a GitHub issue"
    mock_tool.category = "github"

    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool]
    bridge._client = mock_client

    tools = await bridge.discover_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "GITHUB_CREATE_ISSUE"


async def test_discover_tools_filter_category(bridge):
    mock_tool_gh = MagicMock()
    mock_tool_gh.name = "GITHUB_CREATE_ISSUE"
    mock_tool_gh.description = "Create a GitHub issue"
    mock_tool_gh.category = "github"

    mock_tool_slack = MagicMock()
    mock_tool_slack.name = "SLACK_SEND_MESSAGE"
    mock_tool_slack.description = "Send a Slack message"
    mock_tool_slack.category = "slack"

    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool_gh, mock_tool_slack]
    bridge._client = mock_client

    tools = await bridge.discover_tools(category="slack")
    assert len(tools) == 1
    assert tools[0]["name"] == "SLACK_SEND_MESSAGE"


# --- execute_tool ---

async def test_execute_tool_success(bridge):
    mock_client = MagicMock()
    mock_client.execute_action.return_value = {"data": "created"}
    bridge._client = mock_client

    result = await bridge.execute_tool("GITHUB_CREATE_ISSUE", {"title": "Bug"})
    assert result["success"] is True
    assert result["result"]["data"] == "created"


async def test_execute_tool_failure(bridge):
    mock_client = MagicMock()
    mock_client.execute_action.side_effect = Exception("auth failed")
    bridge._client = mock_client

    result = await bridge.execute_tool("GITHUB_CREATE_ISSUE", {"title": "Bug"})
    assert result["success"] is False
    assert "auth failed" in result["error"]


# --- get_tool_definitions ---

def test_get_tool_definitions_sovereign_returns_empty(bridge):
    """In sovereign mode, no Composio tool definitions should be returned."""
    assert bridge.tool_mode == ToolMode.SOVEREIGN
    definitions = bridge.get_tool_definitions()
    assert definitions == []


def test_get_tool_definitions_hybrid_returns_tools():
    with patch.dict("os.environ", {"HELIX_TOOL_MODE": "hybrid"}):
        b = ComposioBridge()

    mock_tool = MagicMock()
    mock_tool.name = "SLACK_SEND"
    mock_tool.description = "Send Slack message"
    mock_tool.parameters = {"channel": {"type": "string"}}

    mock_client = MagicMock()
    mock_client.get_tools.return_value = [mock_tool]
    b._client = mock_client

    defs = b.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "composio_SLACK_SEND"
    assert "channel" in defs[0]["inputSchema"]["properties"]
