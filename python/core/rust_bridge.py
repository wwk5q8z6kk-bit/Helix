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

    # --- DAG Scheduler ---

    async def scheduler_define_workflow(
        self,
        name: str,
        tasks: List[Dict[str, Any]],
        deadline: Optional[float] = None,
        budget: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Define a workflow in the Rust DAG scheduler."""
        session = await self._get_session()
        payload: Dict[str, Any] = {"name": name, "tasks": tasks}
        if deadline is not None:
            payload["deadline"] = deadline
        if budget is not None:
            payload["budget"] = budget
        async with session.post("/api/v1/scheduler/workflows", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            logger.error("scheduler_define_workflow failed: %s", resp.status)
            return None

    async def scheduler_list_workflows(self) -> List[Dict[str, Any]]:
        """List all stored workflow definitions."""
        session = await self._get_session()
        async with session.get("/api/v1/scheduler/workflows") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def scheduler_get_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a workflow definition by name."""
        session = await self._get_session()
        async with session.get(f"/api/v1/scheduler/workflows/{name}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def scheduler_delete_workflow(self, name: str) -> bool:
        """Delete a stored workflow."""
        session = await self._get_session()
        async with session.delete(f"/api/v1/scheduler/workflows/{name}") as resp:
            return resp.status in (200, 204)

    async def scheduler_preview_workflow(self, name: str) -> Optional[Dict[str, Any]]:
        """Get predictive preview of a stored workflow."""
        session = await self._get_session()
        async with session.post(f"/api/v1/scheduler/workflows/{name}/preview") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def scheduler_list_templates(self) -> List[str]:
        """List pre-built workflow template names."""
        session = await self._get_session()
        async with session.get("/api/v1/scheduler/templates") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("templates", [])
            return []

    async def scheduler_preview_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Preview a pre-built template."""
        session = await self._get_session()
        async with session.post(f"/api/v1/scheduler/templates/{name}/preview") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def scheduler_submit_task(
        self, task_id: str, dependencies: Optional[List[str]] = None, priority: int = 2
    ) -> Optional[Dict[str, Any]]:
        """Submit a task to the live DAG."""
        session = await self._get_session()
        payload = {
            "task_id": task_id,
            "dependencies": dependencies or [],
            "priority": priority,
        }
        async with session.post("/api/v1/scheduler/tasks", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            logger.error("scheduler_submit_task failed: %s", resp.status)
            return None

    async def scheduler_task_state(self, task_id: str) -> Optional[str]:
        """Get a task's current state."""
        session = await self._get_session()
        async with session.get(f"/api/v1/scheduler/tasks/{task_id}") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("state")
            return None

    async def scheduler_ready_tasks(self) -> List[str]:
        """Get all tasks currently ready for execution."""
        session = await self._get_session()
        async with session.get("/api/v1/scheduler/tasks/ready") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def scheduler_mark_running(self, task_id: str) -> bool:
        """Mark a task as running."""
        session = await self._get_session()
        async with session.post(f"/api/v1/scheduler/tasks/{task_id}/run") as resp:
            return resp.status == 200

    async def scheduler_mark_completed(self, task_id: str) -> List[str]:
        """Mark a task as completed. Returns newly-ready downstream task IDs."""
        session = await self._get_session()
        async with session.post(f"/api/v1/scheduler/tasks/{task_id}/complete") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("newly_ready", [])
            return []

    async def scheduler_mark_failed(self, task_id: str) -> List[str]:
        """Mark a task as failed. Returns cancelled downstream task IDs."""
        session = await self._get_session()
        async with session.post(f"/api/v1/scheduler/tasks/{task_id}/fail") as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("cancelled_downstream", [])
            return []

    async def scheduler_execution_waves(self) -> List[List[str]]:
        """Get execution waves (parallel groups in topological order)."""
        session = await self._get_session()
        async with session.get("/api/v1/scheduler/tasks/waves") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def scheduler_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        session = await self._get_session()
        async with session.get("/api/v1/scheduler/stats") as resp:
            if resp.status == 200:
                return await resp.json()
            return {}

    # --- Sources ---

    async def list_sources(self) -> List[Dict[str, Any]]:
        """List all registered source connectors."""
        session = await self._get_session()
        async with session.get("/api/v1/sources") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def register_source(self, source_type: str, name: str, settings: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """Register a new source connector."""
        session = await self._get_session()
        payload = {"source_type": source_type, "name": name, "settings": settings or {}}
        async with session.post("/api/v1/sources", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return None

    async def get_source_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a source connector."""
        session = await self._get_session()
        async with session.get(f"/api/v1/sources/{source_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def remove_source(self, source_id: str) -> bool:
        """Remove a source connector."""
        session = await self._get_session()
        async with session.delete(f"/api/v1/sources/{source_id}") as resp:
            return resp.status in (200, 204)

    async def poll_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Trigger a poll on a source connector."""
        session = await self._get_session()
        async with session.post(f"/api/v1/sources/{source_id}/poll") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    # --- Jobs ---

    async def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs in the queue."""
        session = await self._get_session()
        async with session.get("/api/v1/jobs") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def job_stats(self) -> Dict[str, Any]:
        """Get job queue statistics."""
        session = await self._get_session()
        async with session.get("/api/v1/jobs/stats") as resp:
            if resp.status == 200:
                return await resp.json()
            return {}

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific job."""
        session = await self._get_session()
        async with session.get(f"/api/v1/jobs/{job_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def retry_job(self, job_id: str) -> bool:
        """Retry a failed job."""
        session = await self._get_session()
        async with session.post(f"/api/v1/jobs/{job_id}/retry") as resp:
            return resp.status in (200, 201)

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        session = await self._get_session()
        async with session.post(f"/api/v1/jobs/{job_id}/cancel") as resp:
            return resp.status in (200, 204)

    async def dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Get jobs in the dead-letter queue."""
        session = await self._get_session()
        async with session.get("/api/v1/jobs/dead-letter") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def purge_jobs(self) -> bool:
        """Purge completed jobs from the queue."""
        session = await self._get_session()
        async with session.post("/api/v1/jobs/purge") as resp:
            return resp.status in (200, 204)

    # --- Workflows ---

    async def list_workflows(self) -> List[Dict[str, Any]]:
        """List all workflow definitions."""
        session = await self._get_session()
        async with session.get("/api/v1/workflows") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow definition by ID."""
        session = await self._get_session()
        async with session.get(f"/api/v1/workflows/{workflow_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def execute_workflow(self, workflow_id: str, variables: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Execute a workflow."""
        session = await self._get_session()
        payload = {"variables": variables or {}}
        async with session.post(f"/api/v1/workflows/{workflow_id}/execute", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return None

    async def list_executions(self) -> List[Dict[str, Any]]:
        """List all workflow executions."""
        session = await self._get_session()
        async with session.get("/api/v1/workflows/executions") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a workflow execution."""
        session = await self._get_session()
        async with session.get(f"/api/v1/workflows/executions/{execution_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running workflow execution."""
        session = await self._get_session()
        async with session.post(f"/api/v1/workflows/executions/{execution_id}/cancel") as resp:
            return resp.status in (200, 204)

    # --- Notifications ---

    async def list_notifications(self, severity: Optional[str] = None, unread_only: bool = False) -> List[Dict[str, Any]]:
        """List notifications with optional filtering."""
        session = await self._get_session()
        params: Dict[str, str] = {}
        if severity:
            params["severity"] = severity
        if unread_only:
            params["unread_only"] = "true"
        async with session.get("/api/v1/notifications", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def get_notification(self, notification_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific notification."""
        session = await self._get_session()
        async with session.get(f"/api/v1/notifications/{notification_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def mark_notification_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        session = await self._get_session()
        async with session.post(f"/api/v1/notifications/{notification_id}/read") as resp:
            return resp.status in (200, 204)

    async def list_alert_rules(self) -> List[Dict[str, Any]]:
        """List all alert rules."""
        session = await self._get_session()
        async with session.get("/api/v1/notifications/alerts") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def create_alert_rule(self, name: str, condition: Dict[str, Any], channels: List[str], severity: str = "info") -> Optional[Dict[str, Any]]:
        """Create a new alert rule."""
        session = await self._get_session()
        payload = {"name": name, "condition": condition, "channels": channels, "severity": severity}
        async with session.post("/api/v1/notifications/alerts", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return None

    async def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing alert rule."""
        session = await self._get_session()
        async with session.put(f"/api/v1/notifications/alerts/{rule_id}", json=updates) as resp:
            return resp.status in (200, 204)

    async def delete_alert_rule(self, rule_id: str) -> bool:
        """Delete an alert rule."""
        session = await self._get_session()
        async with session.delete(f"/api/v1/notifications/alerts/{rule_id}") as resp:
            return resp.status in (200, 204)

    # --- Adapters ---

    async def list_adapters(self) -> List[Dict[str, Any]]:
        """List all registered adapters."""
        session = await self._get_session()
        async with session.get("/api/v1/adapters") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def register_adapter(self, adapter_type: str, name: str, config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Register a new adapter."""
        session = await self._get_session()
        payload = {"adapter_type": adapter_type, "name": name, "config": config or {}}
        async with session.post("/api/v1/adapters", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return None

    async def get_adapter_status(self, adapter_id: str) -> Optional[Dict[str, Any]]:
        """Get adapter status."""
        session = await self._get_session()
        async with session.get(f"/api/v1/adapters/{adapter_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def remove_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter."""
        session = await self._get_session()
        async with session.delete(f"/api/v1/adapters/{adapter_id}") as resp:
            return resp.status in (200, 204)

    async def send_message(self, adapter_id: str, content: str, channel: Optional[str] = None) -> bool:
        """Send a message through an adapter."""
        session = await self._get_session()
        payload: Dict[str, Any] = {"content": content}
        if channel:
            payload["channel"] = channel
        async with session.post(f"/api/v1/adapters/{adapter_id}/send", json=payload) as resp:
            return resp.status in (200, 201)

    async def adapter_health(self, adapter_id: str) -> Dict[str, Any]:
        """Check adapter health."""
        session = await self._get_session()
        async with session.post(f"/api/v1/adapters/{adapter_id}/health") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"healthy": False}

    # --- Tunnels ---

    async def list_tunnels(self) -> List[Dict[str, Any]]:
        """List all registered tunnels."""
        session = await self._get_session()
        async with session.get("/api/v1/tunnels") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def register_tunnel(self, tunnel_type: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Register and start a tunnel."""
        session = await self._get_session()
        payload = {"tunnel_type": tunnel_type, **config}
        async with session.post("/api/v1/tunnels", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return None

    async def get_tunnel_status(self, tunnel_id: str) -> Optional[Dict[str, Any]]:
        """Get tunnel status."""
        session = await self._get_session()
        async with session.get(f"/api/v1/tunnels/{tunnel_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def remove_tunnel(self, tunnel_id: str) -> bool:
        """Stop and remove a tunnel."""
        session = await self._get_session()
        async with session.delete(f"/api/v1/tunnels/{tunnel_id}") as resp:
            return resp.status in (200, 204)

    async def tunnel_health(self, tunnel_id: str) -> Dict[str, Any]:
        """Check tunnel health."""
        session = await self._get_session()
        async with session.post(f"/api/v1/tunnels/{tunnel_id}/health") as resp:
            if resp.status == 200:
                return await resp.json()
            return {"healthy": False}

    # --- Rate Limits ---

    async def list_rate_limits(self) -> List[Dict[str, Any]]:
        """List all configured rate limits."""
        session = await self._get_session()
        async with session.get("/api/v1/rate-limits") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def update_rate_limit(self, adapter_id: str, requests_per_minute: int, requests_per_hour: int, burst_size: int = 10) -> bool:
        """Update rate limit for an adapter."""
        session = await self._get_session()
        payload = {
            "adapter_id": adapter_id,
            "requests_per_minute": requests_per_minute,
            "requests_per_hour": requests_per_hour,
            "burst_size": burst_size,
        }
        async with session.put("/api/v1/rate-limits", json=payload) as resp:
            return resp.status in (200, 204)

    # --- API Keys ---

    async def list_api_keys(self) -> List[Dict[str, Any]]:
        """List all API keys."""
        session = await self._get_session()
        async with session.get("/api/v1/keys") as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    async def create_api_key(self, name: str, scopes: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """Create a new API key."""
        session = await self._get_session()
        payload: Dict[str, Any] = {"name": name}
        if scopes:
            payload["scopes"] = scopes
        async with session.post("/api/v1/keys", json=payload) as resp:
            if resp.status in (200, 201):
                return await resp.json()
            return None

    async def get_api_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get API key details."""
        session = await self._get_session()
        async with session.get(f"/api/v1/keys/{key_id}") as resp:
            if resp.status == 200:
                return await resp.json()
            return None

    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        session = await self._get_session()
        async with session.delete(f"/api/v1/keys/{key_id}") as resp:
            return resp.status in (200, 204)

    # --- Audit ---

    async def query_audit(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Query the audit log."""
        session = await self._get_session()
        params = {}
        if filters:
            for k, v in filters.items():
                params[k] = str(v)
        async with session.get("/api/v1/audit/query", params=params) as resp:
            if resp.status == 200:
                return await resp.json()
            return []
