"""Composio integration bridge for external tool access."""

import logging
import os
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ToolMode(Enum):
    SOVEREIGN = "sovereign"
    COMPOSIO = "composio"
    HYBRID = "hybrid"


class ComposioBridge:
    """Bridge to Composio for external tool access (OAuth apps, SaaS integrations)."""

    def __init__(self):
        self._client = None
        self.tool_mode = ToolMode(os.getenv("HELIX_TOOL_MODE", "sovereign"))

    def _ensure_client(self):
        """Lazy-import and initialize composio-core client."""
        if self._client is not None:
            return
        try:
            from composio import ComposioToolSet
            api_key = os.getenv("COMPOSIO_API_KEY")
            self._client = ComposioToolSet(api_key=api_key) if api_key else ComposioToolSet()
            logger.info("Composio client initialized (tool_mode=%s)", self.tool_mode.value)
        except ImportError:
            logger.warning("composio-core not installed — Composio features unavailable")
            raise RuntimeError("composio-core is not installed. Install with: pip install composio-core")

    async def discover_tools(self, category: Optional[str] = None) -> List[Dict]:
        """Discover available Composio tools, optionally filtered by category."""
        self._ensure_client()
        try:
            tools = self._client.get_tools()
            results = []
            for tool in tools:
                entry = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "category": getattr(tool, "category", "general"),
                }
                if category and entry["category"] != category:
                    continue
                results.append(entry)
            return results
        except Exception as e:
            logger.error("Failed to discover Composio tools: %s", e)
            return []

    async def execute_tool(self, tool_name: str, params: Dict, entity_id: str = "default") -> Dict:
        """Execute a Composio tool by name."""
        self._ensure_client()
        try:
            result = self._client.execute_action(
                action=tool_name,
                params=params,
                entity_id=entity_id,
            )
            return {"success": True, "result": result}
        except Exception as e:
            logger.error("Composio tool execution failed: %s", e)
            return {"success": False, "error": str(e)}

    async def initiate_oauth(self, app_name: str, redirect_url: str = "") -> Dict:
        """Initiate OAuth flow for a Composio-connected app."""
        self._ensure_client()
        try:
            entity = self._client.get_entity(id="default")
            connection = entity.initiate_connection(app_name=app_name, redirect_url=redirect_url or None)
            return {
                "success": True,
                "redirect_url": getattr(connection, "redirectUrl", ""),
                "connection_status": getattr(connection, "connectionStatus", "initiated"),
            }
        except Exception as e:
            logger.error("OAuth initiation failed for %s: %s", app_name, e)
            return {"success": False, "error": str(e)}

    async def check_connection(self, app_name: str, entity_id: str = "default") -> bool:
        """Check if a Composio app connection is active."""
        self._ensure_client()
        try:
            entity = self._client.get_entity(id=entity_id)
            connection = entity.get_connection(app=app_name)
            return connection is not None
        except Exception:
            return False

    def get_tool_definitions(self) -> List[Dict]:
        """Return MCP-compatible tool definitions for Composio tools."""
        if self.tool_mode == ToolMode.SOVEREIGN:
            return []
        try:
            self._ensure_client()
            tools = self._client.get_tools()
            definitions = []
            for tool in tools:
                definitions.append({
                    "name": f"composio_{getattr(tool, 'name', str(tool))}",
                    "description": getattr(tool, "description", "Composio tool"),
                    "inputSchema": {
                        "type": "object",
                        "properties": getattr(tool, "parameters", {}),
                    },
                })
            return definitions
        except Exception as e:
            logger.warning("Could not load Composio tool definitions: %s", e)
            return []
