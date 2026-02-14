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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware stack
from core.middleware.request_id import RequestIDMiddleware
from core.middleware.security import SecurityHeadersMiddleware

app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check — reports Python AI and Rust core status."""
    container = get_container()
    rust_ok = await container.bridge.is_healthy()
    return {
        "status": "healthy",
        "python_ai": True,
        "rust_core_connected": rust_ok,
    }


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
    from datetime import datetime

    container = get_container()

    result: Dict[str, Any] = {
        "services": container.status(),
        "timestamp": datetime.utcnow().isoformat(),
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


class FeedbackRequest(BaseModel):
    task_id: str
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""
    agent_id: Optional[str] = None


@app.post("/api/feedback")
async def submit_feedback(req: FeedbackRequest):
    """Submit quality feedback for a completed task."""
    from core.feedback.feedback_handler import FeedbackHandler

    container = get_container()
    handler = FeedbackHandler(event_bus=container.event_bus)
    return await handler.handle_feedback(
        task_id=req.task_id,
        score=req.score,
        feedback=req.feedback,
        agent_id=req.agent_id,
    )
