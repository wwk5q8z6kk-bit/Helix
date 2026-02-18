"""
Helix Platform — FastAPI application entry point.

Provides REST API for the Python AI layer:
- /health — service health status
- /api/generate — LLM generation via intelligent router
- /api/search — semantic search via Rust core bridge
- /api/workflow/start — orchestrated task execution
- /api/models — available LLM providers
"""

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from core.di_container import get_container, shutdown_container
from core.interfaces import EventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = Field(default=2048, ge=1, le=128000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    task_type: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str
    provider: str
    model: str
    tokens_used: int = 0
    cost: float = 0.0


class EventRequest(BaseModel):
    type: str
    source: str = "external"
    data: Dict[str, Any] = Field(default_factory=dict)


class WorkflowDefRequest(BaseModel):
    name: str
    tasks: list
    deadline: Optional[float] = None
    budget: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    namespace: Optional[str] = None


class WorkflowRequest(BaseModel):
    task: str
    strategy: str = "auto"
    context: Dict[str, Any] = Field(default_factory=dict)


class SwarmRequest(BaseModel):
    task: str
    swarm_type: str = "implementation"
    context: Dict[str, Any] = Field(default_factory=dict)


class ReasonRequest(BaseModel):
    goal: str
    max_steps: int = Field(default=5, ge=1, le=20)
    context: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Helix Python AI starting up")
    container = get_container()
    logger.info("DI container initialized with %d services", len(container.status()))
    yield
    shutdown_container()
    logger.info("Helix Python AI shutting down")


app = FastAPI(
    title="Helix AI",
    version="0.2.0",
    description="Helix Platform — Python AI Layer",
    lifespan=lifespan,
)

def _cors_origins() -> list[str]:
    env = os.environ.get("HELIX_ENV", "development")
    raw = os.environ.get("HELIX_CORS_ORIGINS", "")
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    if env == "development":
        return ["*"]
    return ["http://localhost:3000", "http://localhost:8200"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware stack
from core.middleware.request_id import RequestIDMiddleware
from core.middleware.security import SecurityHeadersMiddleware
from core.middleware.rate_limiter import RateLimitMiddleware

app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check — reports Python AI and Rust core status."""
    container = get_container()
    rust_ok = await container.bridge.is_healthy()
    status = "healthy" if rust_ok else "degraded"
    body = {
        "status": status,
        "python_ai": True,
        "rust_core_connected": rust_ok,
    }
    if not rust_ok:
        return JSONResponse(content=body, status_code=503)
    return body


@app.get("/api/models")
async def list_models():
    """List available LLM providers and their stats."""
    container = get_container()
    router = container.llm_router
    return {
        "providers": [
            {
                "name": provider.value,
                "quality_score": stats.quality_score,
                "avg_latency_ms": int(stats.avg_latency_ms),
                "cost_per_1k_tokens": stats.cost_per_1k_tokens,
                "success_rate": stats.success_rate,
            }
            for provider, stats in router.provider_stats.items()
        ]
    }


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """Generate text via the intelligent LLM router."""
    from core.llm.intelligent_llm_router import TaskType

    container = get_container()
    router = container.llm_router

    # Map string task_type to enum
    task_str = (req.task_type or "code_generation").lower()
    try:
        task_type = TaskType(task_str)
    except ValueError:
        task_type = TaskType.CODE_GENERATION

    try:
        # Select best provider for the task
        provider = await router.select_provider(
            task_type=task_type,
            prompt_tokens=len(req.prompt.split()),
        )

        # Build messages in chat format
        messages = [{"role": "user", "content": req.prompt}]

        text, metadata = await router.call_llm(
            provider=provider,
            messages=messages,
            task_type=task_type,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        return GenerateResponse(
            text=text,
            provider=metadata.get("provider", provider.value),
            model=provider.value,
            tokens_used=metadata.get("tokens_used", 0),
            cost=metadata.get("cost", 0.0),
        )
    except Exception as e:
        logger.error("LLM generation failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/search")
async def search(req: SearchRequest):
    """Semantic search via Rust core."""
    container = get_container()
    try:
        results = await container.bridge.semantic_search(
            query=req.query,
            limit=req.limit,
            namespace=req.namespace or "default",
        )
        return {"results": results}
    except Exception as e:
        logger.error("Search failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/workflow/start")
async def start_workflow(req: WorkflowRequest):
    """Start an orchestrated workflow via the UnifiedOrchestrator."""
    from core.orchestration.unified_orchestrator import Task, TaskType as OrcTaskType

    container = get_container()
    orchestrator = container.orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not available")

    try:
        task_id = uuid.uuid4().hex[:12]

        # Map strategy to task type
        try:
            task_type = OrcTaskType(req.strategy)
        except ValueError:
            task_type = OrcTaskType.IMPLEMENTATION

        task = Task(
            task_id=task_id,
            task_type=task_type,
            description=req.task,
            context=req.context,
        )
        result = await orchestrator.execute_task(task=task)
        return {
            "status": "completed",
            "task_id": task_id,
            "result": {
                "task_id": result.task_id,
                "status": result.status,
                "output": result.output,
            },
        }
    except Exception as e:
        logger.error("Workflow failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/stats")
async def router_stats():
    """Get LLM router statistics."""
    container = get_container()
    return await container.llm_router.get_router_stats()


@app.get("/api/board")
async def board():
    """System dashboard — aggregates service state, provider stats, and recent events."""
    from core.observability.tower_log import TowerLogHandler
    from datetime import datetime, timezone

    container = get_container()

    result: Dict[str, Any] = {
        "services": container.status(),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }

    # Provider stats (safe access)
    try:
        result["providers"] = [
            {
                "name": p.value,
                "quality_score": s.quality_score,
                "avg_latency_ms": int(s.avg_latency_ms),
                "success_rate": s.success_rate,
            }
            for p, s in container.llm_router.provider_stats.items()
        ]
    except Exception:
        result["providers"] = []

    # Orchestrator status (safe access)
    try:
        if container.orchestrator:
            status = await container.orchestrator.get_system_status()
            result["orchestrator"] = {
                "active_agents": status.active_agents,
                "active_tasks": status.active_tasks,
                "completed_tasks": status.completed_tasks,
                "system_health": status.system_health,
            }
    except Exception:
        result["orchestrator"] = {}

    # Recent tower events
    try:
        tower = TowerLogHandler()
        result["recent_events"] = tower.get_recent(limit=10)
    except Exception:
        result["recent_events"] = []

    return result


@app.post("/api/event")
async def receive_event(req: EventRequest):
    """Receive and route an external event."""
    from core.api.event_router import EventRouter

    container = get_container()
    router = EventRouter(event_bus=container.event_bus)
    return await router.route_event(
        event_type=req.type,
        source=req.source,
        data=req.data,
    )


@app.get("/tower")
async def tower_events(limit: int = 100):
    """Get recent tower events."""
    from core.observability.tower_log import TowerLogHandler

    handler = TowerLogHandler()
    limit = min(max(1, limit), 1000)
    return {"events": handler.get_recent(limit=limit)}


@app.get("/api/progress/{task_id}")
async def stream_progress(task_id: str):
    """Stream task progress as Server-Sent Events."""
    container = get_container()
    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(data):
        if data.get("task_id") == task_id:
            await progress_queue.put(data)

    sub_id = await container.event_bus.subscribe(EventType.WORKFLOW_PROGRESS, on_progress)

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(progress_queue.get(), timeout=30.0)
                    yield {"event": "progress", "data": json.dumps(data)}
                    if data.get("percent", 0) >= 100:
                        break
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            await container.event_bus.unsubscribe(sub_id)

    return EventSourceResponse(event_generator())


@app.post("/api/reason")
async def reason(req: ReasonRequest):
    """Multi-step reasoning via the Agentic Reasoner."""
    container = get_container()
    reasoner = container.reasoner
    if reasoner is None:
        raise HTTPException(status_code=503, detail="Reasoner not available")

    try:
        reasoner.max_steps = req.max_steps
        trajectory, reward = await reasoner.solve(problem=req.goal, initial_context=req.context)
        return {
            "steps": [
                {"content": s.content, "type": s.step_type.value, "confidence": s.confidence}
                for s in trajectory.steps
            ],
            "conclusion": trajectory.final_answer or "",
            "success": trajectory.success,
            "confidence": reward.final_score,
        }
    except Exception as e:
        logger.error("Reasoning failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/swarm/execute")
async def swarm_execute(req: SwarmRequest):
    """Execute a specialized agent swarm for complex tasks."""
    container = get_container()
    orchestrator = container.swarm_orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Swarm orchestrator not available")

    try:
        context = req.context.copy()
        context["swarm_type"] = req.swarm_type
        result = await orchestrator.route_and_execute(req.task, context)
        return {
            "status": "completed" if result.swarm_result.success else "failed",
            "task_id": result.task_id,
            "swarm_used": result.swarm_used,
            "category": result.task_category.value,
            "quality_score": result.quality_score,
            "routing_confidence": result.routing_confidence,
            "output": str(result.swarm_result.output),
        }
    except Exception as e:
        logger.error("Swarm execution failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/budget")
async def budget_dashboard():
    """Get budget and cost tracking dashboard."""
    container = get_container()
    return container.budget_tracker.get_dashboard()


@app.get("/api/resources")
async def get_resource_status():
    """Resource manager dashboard — concurrency, budget, circuit breakers."""
    container = get_container()
    orch = container.orchestrator
    if orch is None or orch._resource_manager is None:
        return {"status": "not_configured"}
    return orch._resource_manager.stats


class FeedbackRequest(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""
    agent_id: Optional[str] = None


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit quality feedback for a completed task."""
    container = get_container()
    handler = container.feedback_handler
    return await handler.handle_feedback(
        task_id=req.task_id,
        score=req.score,
        feedback=req.feedback,
        agent_id=req.agent_id,
    )


# ---------------------------------------------------------------------------
# Composio integration endpoints
# ---------------------------------------------------------------------------


@app.get("/api/composio/tools")
async def composio_tools(category: Optional[str] = None):
    """List available Composio tools."""
    container = get_container()
    bridge = container.composio_bridge
    try:
        tools = await bridge.discover_tools(category=category)
        return {"tools": tools, "tool_mode": bridge.tool_mode.value}
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


class ComposioExecuteRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    entity_id: str = "default"


@app.post("/api/composio/execute")
async def composio_execute(req: ComposioExecuteRequest):
    """Execute a Composio tool."""
    container = get_container()
    bridge = container.composio_bridge
    try:
        result = await bridge.execute_tool(
            tool_name=req.tool_name,
            params=req.params,
            entity_id=req.entity_id,
        )
        if not result.get("success"):
            raise HTTPException(status_code=502, detail=result.get("error", "Tool execution failed"))
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/composio/oauth/{app_name}")
async def composio_oauth(app_name: str, redirect_url: str = ""):
    """Initiate OAuth flow for a Composio app."""
    container = get_container()
    bridge = container.composio_bridge
    try:
        result = await bridge.initiate_oauth(app_name=app_name, redirect_url=redirect_url)
        if not result.get("success"):
            raise HTTPException(status_code=502, detail=result.get("error", "OAuth initiation failed"))
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))


# ---------------------------------------------------------------------------
# Analytics & Reports endpoints (Phase 15)
# ---------------------------------------------------------------------------


class ReportRequest(BaseModel):
    report_type: str = "summary"
    context: Dict[str, Any] = Field(default_factory=dict)


@app.get("/api/analytics/dashboard")
async def analytics_dashboard():
    """Aggregated system analytics dashboard."""
    container = get_container()
    dashboard = await container.analytics_engine.get_dashboard()
    return {
        "total_requests": dashboard.total_requests,
        "total_tokens": dashboard.total_tokens,
        "total_cost": dashboard.total_cost,
        "active_providers": dashboard.active_providers,
        "requests_today": dashboard.requests_today,
        "top_providers": [
            {
                "provider": p.provider,
                "total_requests": p.total_requests,
                "avg_latency_ms": p.avg_latency_ms,
                "error_rate": p.error_rate,
                "total_cost": p.total_cost,
            }
            for p in dashboard.top_providers
        ],
    }


@app.get("/api/analytics/trends")
async def analytics_trends(period: str = "daily", days: int = 30):
    """Token usage and cost trends over time."""
    container = get_container()
    trends = await container.analytics_engine.get_trends(period=period, days=days)
    return {"period": trends.period, "data_points": trends.data_points, "summary": trends.summary}


@app.get("/api/analytics/providers")
async def analytics_providers():
    """Per-provider performance breakdown."""
    container = get_container()
    providers = await container.analytics_engine.get_provider_performance()
    return {
        "providers": [
            {
                "provider": p.provider,
                "total_requests": p.total_requests,
                "avg_latency_ms": p.avg_latency_ms,
                "error_rate": p.error_rate,
                "total_tokens": p.total_tokens,
                "total_cost": p.total_cost,
            }
            for p in providers
        ]
    }


@app.get("/api/analytics/token-usage")
async def analytics_token_usage(provider: Optional[str] = None):
    """Detailed token usage log."""
    container = get_container()
    usage = await container.analytics_engine.get_token_usage(provider=provider)
    return {
        "usage": [
            {
                "provider": u.provider,
                "model": u.model,
                "tokens_in": u.tokens_in,
                "tokens_out": u.tokens_out,
                "cost": u.cost,
                "timestamp": u.timestamp.isoformat(),
            }
            for u in usage[-100:]
        ]
    }


@app.post("/api/reports/generate")
async def generate_report(req: ReportRequest):
    """Generate an analytical report."""
    container = get_container()
    report = await container.report_generator.generate(
        report_type=req.report_type,
        context=req.context,
    )
    return {
        "id": report.id,
        "title": report.title,
        "content": report.content,
        "report_type": report.report_type,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a previously generated report."""
    container = get_container()
    report = await container.report_generator.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return {
        "id": report.id,
        "title": report.title,
        "content": report.content,
        "report_type": report.report_type,
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


# ---------------------------------------------------------------------------
# Scheduling endpoints (Rust-first DAG scheduler)
# ---------------------------------------------------------------------------


@app.post("/api/scheduler/workflows")
async def define_workflow(req: WorkflowDefRequest):
    """Define a DAG workflow in the scheduler."""
    container = get_container()
    result = await container.orchestrator.define_workflow(
        name=req.name, tasks=req.tasks,
        deadline=req.deadline, budget=req.budget,
    )
    if result is None:
        raise HTTPException(status_code=400, detail="Failed to define workflow")
    return result


@app.get("/api/scheduler/workflows")
async def list_workflows():
    """List all stored workflow definitions."""
    container = get_container()
    return await container.orchestrator.list_workflows()


@app.get("/api/scheduler/workflows/{name}")
async def get_workflow(name: str):
    """Get a workflow definition by name."""
    container = get_container()
    result = await container.orchestrator.get_workflow(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    return result


@app.delete("/api/scheduler/workflows/{name}")
async def delete_workflow(name: str):
    """Delete a stored workflow."""
    container = get_container()
    deleted = await container.orchestrator.delete_workflow(name)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    return {"status": "deleted", "name": name}


@app.post("/api/scheduler/workflows/{name}/preview")
async def preview_workflow(name: str):
    """Get predictive preview of a stored workflow."""
    container = get_container()
    result = await container.orchestrator.preview_workflow(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Workflow '{name}' not found")
    return result


@app.get("/api/scheduler/templates")
async def list_templates():
    """List pre-built workflow template names."""
    container = get_container()
    return {"templates": await container.orchestrator.list_workflow_templates()}


@app.post("/api/scheduler/templates/{name}/preview")
async def preview_template(name: str):
    """Preview a pre-built workflow template."""
    container = get_container()
    result = await container.orchestrator.preview_template(name)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Template '{name}' not found")
    return result


@app.get("/api/scheduler/stats")
async def scheduler_stats():
    """Get scheduler statistics."""
    container = get_container()
    return await container.orchestrator.scheduler_stats()


@app.get("/api/scheduler/waves")
async def scheduler_waves():
    """Get execution waves (parallel groups in topological order)."""
    container = get_container()
    return {"waves": await container.orchestrator.execution_waves()}


# ---------------------------------------------------------------------------
# Workflow Execution endpoints
# ---------------------------------------------------------------------------


class WorkflowRunRequest(BaseModel):
    context: Optional[Dict[str, Any]] = None


@app.post("/api/scheduler/workflows/{name}/run")
async def run_workflow(name: str, req: WorkflowRunRequest = WorkflowRunRequest()):
    """Execute a stored workflow."""
    container = get_container()
    try:
        wf_exec = await container.orchestrator.run_workflow(name, context=req.context)
        return wf_exec.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/api/scheduler/executions")
async def list_executions(status: Optional[str] = None, limit: int = 50):
    """List workflow executions."""
    container = get_container()
    return container.orchestrator.list_workflow_executions(status=status, limit=limit)


@app.get("/api/scheduler/executions/{execution_id}")
async def get_execution(execution_id: str):
    """Get a workflow execution by ID."""
    container = get_container()
    result = container.orchestrator.get_workflow_execution(execution_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    return result


@app.post("/api/scheduler/executions/{execution_id}/cancel")
async def cancel_execution(execution_id: str):
    """Cancel a running workflow execution."""
    container = get_container()
    cancelled = await container.orchestrator.cancel_workflow_execution(execution_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not running")
    return {"status": "cancelled", "execution_id": execution_id}


@app.get("/api/scheduler/executions/{execution_id}/progress")
async def stream_execution_progress(execution_id: str):
    """Stream workflow execution progress as Server-Sent Events.

    Emits events:
      - "progress": incremental updates with percent, completed/total tasks
      - "completed"/"failed": terminal event with full execution summary
      - "ping": keepalive every 30s
    """
    container = get_container()

    # Verify execution exists
    wf_exec = container.orchestrator.get_workflow_execution(execution_id)
    if wf_exec is None:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")

    # If already finished, return final state as a single event
    if isinstance(wf_exec, dict) and wf_exec.get("status") in ("completed", "failed", "cancelled"):
        async def done_generator():
            yield {"event": wf_exec["status"], "data": json.dumps(wf_exec)}
        return EventSourceResponse(done_generator())

    progress_queue: asyncio.Queue = asyncio.Queue()

    async def on_progress(data):
        if data.get("execution_id") == execution_id:
            await progress_queue.put(data)

    sub_id = await container.event_bus.subscribe(EventType.WORKFLOW_PROGRESS, on_progress)

    async def event_generator():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(progress_queue.get(), timeout=30.0)
                    phase = data.get("phase", "progress")
                    yield {"event": phase, "data": json.dumps(data)}
                    if phase in ("completed", "failed"):
                        break
                except asyncio.TimeoutError:
                    yield {"event": "ping", "data": ""}
        finally:
            await container.event_bus.unsubscribe(sub_id)

    return EventSourceResponse(event_generator())


# --- Source Connectors (delegated to Rust core) ---


class SourceRegistrationRequest(BaseModel):
    type: str
    name: str
    settings: Dict[str, str] = Field(default_factory=dict)


@app.get("/api/sources")
async def list_sources():
    """List registered source connectors (via Rust core)."""
    container = get_container()
    try:
        return {"sources": await container.bridge.list_sources()}
    except Exception as e:
        logger.error("Failed to list sources: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/sources")
async def register_source(req: SourceRegistrationRequest):
    """Register a new source connector (via Rust core)."""
    container = get_container()
    try:
        result = await container.bridge.register_source(
            source_type=req.type, name=req.name, settings=req.settings,
        )
        if result is None:
            raise HTTPException(status_code=400, detail="Failed to register source")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to register source: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/sources/{source_id}")
async def get_source_status(source_id: str):
    """Get a source connector's status (via Rust core)."""
    container = get_container()
    try:
        result = await container.bridge.get_source_status(source_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Source not found")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get source status: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.delete("/api/sources/{source_id}")
async def remove_source(source_id: str):
    """Remove a source connector (via Rust core)."""
    container = get_container()
    try:
        removed = await container.bridge.remove_source(source_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Source not found")
        return {"status": "removed", "source_id": source_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to remove source: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/api/sources/{source_id}/poll")
async def poll_source(source_id: str):
    """Trigger a poll on a source connector (via Rust core)."""
    container = get_container()
    try:
        documents = await container.bridge.poll_source(source_id)
        return {"source_id": source_id, "documents": documents}
    except Exception as e:
        logger.error("Failed to poll source: %s", e)
        raise HTTPException(status_code=502, detail=str(e))


# --- Notifications ---


@app.get("/api/notifications")
async def list_notifications(
    severity: Optional[str] = None,
    unread_only: bool = False,
    limit: int = 50,
):
    """List AI-layer notifications (workflow events, quality alerts)."""
    container = get_container()
    svc = container.notification_service
    from core.notifications.notification_service import Severity as Sev

    sev = None
    if severity:
        try:
            sev = Sev(severity)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")

    items = await svc.list(severity=sev, read=False if unread_only else None, limit=limit)
    return {
        "notifications": [
            {
                "id": n.id,
                "title": n.title,
                "body": n.body,
                "severity": n.severity.value,
                "source": n.source,
                "metadata": n.metadata,
                "created_at": n.created_at.isoformat() if n.created_at else None,
                "read": n.read,
            }
            for n in items
        ],
        "total": await svc.count(),
        "unread": await svc.count_unread(),
    }


@app.get("/api/notifications/{notification_id}")
async def get_notification(notification_id: str):
    """Get a single notification by ID."""
    container = get_container()
    n = await container.notification_service.get(notification_id)
    if n is None:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {
        "id": n.id,
        "title": n.title,
        "body": n.body,
        "severity": n.severity.value,
        "source": n.source,
        "metadata": n.metadata,
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "read": n.read,
    }


@app.post("/api/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: str):
    """Mark a notification as read."""
    container = get_container()
    found = await container.notification_service.mark_read(notification_id)
    if not found:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "read", "notification_id": notification_id}
