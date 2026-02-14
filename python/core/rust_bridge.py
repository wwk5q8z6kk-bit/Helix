"""
Helix Rust Bridge — connects Python AI layer to Rust core via REST/gRPC.

The Rust core (hx-server) runs on :9470 and provides:
- Memory storage and retrieval
- Semantic search (vector + full-text)
- Knowledge graph operations
- Credential store (API keys via OS keychain / encrypted file)

This bridge implements IMemoryStore so the orchestrator can use Rust-backed
storage transparently through dependency injection.
"""

import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)

RUST_CORE_BASE = "http://127.0.0.1:9470"


class RustCoreBridge:
    """Async HTTP client for the Helix Rust core REST API."""

    def __init__(self, base_url: str = RUST_CORE_BASE, auth_token: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            self._session = aiohttp.ClientSession(
                base_url=self.base_url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # --- Health ---

    async def health_check(self) -> Dict[str, Any]:
        session = await self._get_session()
        async with session.get("/health") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"status": "unhealthy", "code": resp.status}

    async def is_healthy(self) -> bool:
        try:
            result = await self.health_check()
            return result.get("status") == "ok"
        except Exception:
            return False

    # --- Memory (IMemoryStore implementation) ---

    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a knowledge node via the Rust core."""
        session = await self._get_session()
        payload = {"content": str(value), "kind": "fact", "title": key, "namespace": "helix"}
        if ttl:
            payload["metadata"] = {"ttl": ttl}
        async with session.post("/api/v1/nodes", json=payload) as resp:
            if resp.status not in (200, 201):
                logger.error(f"Rust store failed: {resp.status}")

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve by searching for the key as title."""
        session = await self._get_session()
        async with session.get("/api/v1/search", params={"q": key, "limit": "1"}) as resp:
            if resp.status == 200:
                data = await resp.json()
                nodes = data.get("nodes", data.get("results", []))
                if nodes:
                    return nodes[0].get("content")
            return None

    async def delete(self, key: str) -> None:
        """Delete by key (title search then delete)."""
        # First find the node
        session = await self._get_session()
        async with session.get("/api/v1/search", params={"q": key, "limit": "1"}) as resp:
            if resp.status == 200:
                data = await resp.json()
                nodes = data.get("nodes", data.get("results", []))
                if nodes:
                    node_id = nodes[0].get("id")
                    async with session.delete(f"/api/v1/nodes/{node_id}") as del_resp:
                        if del_resp.status not in (200, 204):
                            logger.error(f"Rust delete failed: {del_resp.status}")

    async def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List node titles matching pattern."""
        session = await self._get_session()
        params = {"limit": "100"}
        if pattern:
            params["q"] = pattern
        async with session.get("/api/v1/search", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                nodes = data.get("nodes", data.get("results", []))
                return [n.get("title", n.get("id", "")) for n in nodes]
            return []

    # --- Semantic search ---

    async def semantic_search(self, query: str, limit: int = 10, namespace: str = "default") -> List[Dict]:
        """Semantic vector search via Rust core."""
        session = await self._get_session()
        params = {"q": query, "limit": str(limit), "ns": namespace}
        async with session.get("/api/v1/search", params=params) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("nodes", data.get("results", []))
            return []

    # --- Knowledge graph ---

    async def graph_query(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Query knowledge graph around a node."""
        session = await self._get_session()
        async with session.get(f"/api/v1/graph/{node_id}", params={"depth": str(depth)}) as resp:
            if resp.status == 200:
                return await resp.json()
            return {"nodes": [], "edges": []}

    async def graph_link(self, source_id: str, target_id: str, relation: str) -> bool:
        """Create a graph edge."""
        session = await self._get_session()
        payload = {"source_id": source_id, "target_id": target_id, "relation": relation}
        async with session.post("/api/v1/graph/edges", json=payload) as resp:
            return resp.status in (200, 201)

    # --- Credentials ---

    async def get_secret(self, key: str) -> Optional[str]:
        """Get a secret from the Rust credential store."""
        session = await self._get_session()
        async with session.get(f"/api/v1/secrets/{key}") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("value")
            return None

    async def secrets_status(self) -> Dict[str, Any]:
        """Get credential store status."""
        session = await self._get_session()
        async with session.get("/api/v1/secrets/status") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"available": False}
