"""
Helix MCP Server — Model Context Protocol server exposing Helix AI tools.

Run: python -m core.mcp.server
"""

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from core.mcp.tools import get_tool_definitions
from core.llm.intelligent_llm_router import IntelligentLLMRouter, TaskType, LLMProvider, get_llm_router
from core.rust_bridge import RustCoreBridge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("helix")
bridge = RustCoreBridge()


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return available Helix tools, including Composio tools when enabled."""
    tools = [
        Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
        for t in get_tool_definitions()
    ]

    # Include Composio tools in composio/hybrid mode
    try:
        from core.composio.composio_bridge import ComposioBridge, ToolMode
        composio = ComposioBridge()
        if composio.tool_mode in (ToolMode.COMPOSIO, ToolMode.HYBRID):
            for t in composio.get_tool_definitions():
                tools.append(Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]))
    except Exception:
        pass  # Composio unavailable — continue with built-in tools only

    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "helix_generate":
            result = await _handle_generate(arguments)
        elif name == "helix_search":
            result = await _handle_search(arguments)
        elif name == "helix_orchestrate":
            result = await _handle_orchestrate(arguments)
        elif name == "helix_swarm":
            result = await _handle_swarm(arguments)
        elif name == "helix_review":
            result = await _handle_review(arguments)
        elif name == "helix_reason":
            result = await _handle_reason(arguments)
        elif name == "helix_learn":
            result = await _handle_learn(arguments)
        elif name == "helix_schedule":
            result = await _handle_schedule(arguments)
        elif name == "helix_notify":
            result = await _handle_notify(arguments)
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=json.dumps({"error": str(e)}))]


async def _handle_generate(args: dict) -> dict:
    """Handle helix_generate tool."""
    router = await get_llm_router()
    prompt = args["prompt"]
    provider_name = args.get("provider", "auto")
    max_tokens = args.get("max_tokens", 4096)
    temperature = args.get("temperature", 0.7)

    messages = [{"role": "user", "content": prompt}]

    if provider_name == "auto":
        provider = await router.select_provider(
            TaskType.CODE_GENERATION, len(prompt.split()) * 2
        )
    else:
        provider = LLMProvider(provider_name)

    text, metadata = await router.call_llm(
        provider, messages, TaskType.CODE_GENERATION, max_tokens, temperature
    )
    return {"content": text, **metadata}


async def _handle_search(args: dict) -> dict:
    """Handle helix_search tool — delegates to Rust core."""
    query = args["query"]
    limit = args.get("limit", 10)
    namespace = args.get("namespace", "default")

    results = await bridge.semantic_search(query, limit, namespace)
    return {"query": query, "results": results, "count": len(results)}


async def _handle_orchestrate(args: dict) -> dict:
    """Handle helix_orchestrate tool."""
    router = await get_llm_router()
    task = args["task"]
    task_type_str = args.get("task_type", "code_generation")
    quality = args.get("quality", 8.0)
    speed = args.get("speed_critical", False)

    task_type = TaskType(task_type_str)
    provider = await router.select_provider(
        task_type, len(task.split()) * 2, quality, speed
    )

    messages = [{"role": "user", "content": task}]
    text, metadata = await router.call_llm(provider, messages, task_type)
    return {"result": text, "routing": metadata}


async def _handle_swarm(args: dict) -> dict:
    """Handle helix_swarm tool — routes to SwarmOrchestrator."""
    from core.swarms.swarm_orchestrator import get_swarm_orchestrator

    orchestrator = get_swarm_orchestrator()
    task = args["task"]
    context = args.get("context", {})
    context["swarm_type"] = args.get("swarm_type", "implementation")

    result = await orchestrator.route_and_execute(task, context)
    return {
        "status": "completed" if result.swarm_result.success else "failed",
        "task_id": result.task_id,
        "swarm_used": result.swarm_used,
        "category": result.task_category.value,
        "quality_score": result.quality_score,
        "routing_confidence": result.routing_confidence,
        "output": str(result.swarm_result.output),
    }


async def _handle_review(args: dict) -> dict:
    """Handle helix_review tool."""
    router = await get_llm_router()
    code = args["code"]
    language = args.get("language", "unknown")
    focus = args.get("focus", ["bugs", "security", "performance"])

    prompt = f"Review this {language} code for {', '.join(focus)}:\n\n```{language}\n{code}\n```"
    messages = [{"role": "user", "content": prompt}]

    provider = await router.select_provider(TaskType.TESTING, len(prompt.split()) * 2, 9.0)
    text, metadata = await router.call_llm(provider, messages, TaskType.TESTING)
    return {"review": text, "focus": focus, **metadata}


async def _handle_reason(args: dict) -> dict:
    """Handle helix_reason tool — multi-step agentic reasoning."""
    from core.reasoning.agentic_reasoner import get_agentic_reasoner

    reasoner = get_agentic_reasoner()
    goal = args["goal"]
    max_steps = args.get("max_steps", 5)
    context = args.get("context", {})

    # AgenticReasoner uses solve() — limit steps to max_steps
    reasoner.max_steps = max_steps
    trajectory, reward = await reasoner.solve(problem=goal, initial_context=context)

    return {
        "steps": [
            {"content": s.content, "type": s.step_type.value, "confidence": s.confidence}
            for s in trajectory.steps
        ],
        "conclusion": trajectory.final_answer or "",
        "success": trajectory.success,
        "confidence": reward.final_score,
        "steps_executed": len(trajectory.steps),
    }


async def _handle_learn(args: dict) -> dict:
    """Handle helix_learn tool — query learned patterns and agent performance."""
    from core.learning.learning_optimizer import LearningOptimizer

    query = args["query"]
    limit = args.get("limit", 5)

    optimizer = LearningOptimizer()
    stats = optimizer.get_learning_statistics()

    # Filter agent stats by query relevance
    agent_stats = stats.get("agent_statistics", [])
    query_lower = query.lower()
    matched = [
        a for a in agent_stats
        if query_lower in a.get("agent_name", "").lower()
        or query_lower in a.get("agent_id", "").lower()
    ]

    return {
        "query": query,
        "patterns": matched[:limit] if matched else agent_stats[:limit],
        "total_agents": stats.get("total_agents", 0),
        "learning_iterations": stats.get("learning_iterations", 0),
        "success_rate": stats.get("success_rate", 0.0),
        "average_quality": stats.get("average_quality", 0.0),
    }


async def _handle_schedule(args: dict) -> dict:
    """Handle helix_schedule tool — DAG workflow scheduling via Rust core."""
    action = args["action"]
    name = args.get("name")

    if action == "define":
        if not name or "tasks" not in args:
            return {"error": "define requires 'name' and 'tasks'"}
        result = await bridge.scheduler_define_workflow(
            name=name,
            tasks=args["tasks"],
            deadline=args.get("deadline"),
            budget=args.get("budget"),
        )
        return result if result else {"error": "Failed to define workflow"}

    elif action == "preview":
        if not name:
            return {"error": "preview requires 'name'"}
        result = await bridge.scheduler_preview_workflow(name)
        return result if result else {"error": f"Workflow '{name}' not found"}

    elif action == "list":
        return {"workflows": await bridge.scheduler_list_workflows()}

    elif action == "get":
        if not name:
            return {"error": "get requires 'name'"}
        result = await bridge.scheduler_get_workflow(name)
        return result if result else {"error": f"Workflow '{name}' not found"}

    elif action == "delete":
        if not name:
            return {"error": "delete requires 'name'"}
        deleted = await bridge.scheduler_delete_workflow(name)
        return {"deleted": deleted, "name": name}

    elif action == "templates":
        return {"templates": await bridge.scheduler_list_templates()}

    elif action == "preview_template":
        if not name:
            return {"error": "preview_template requires 'name'"}
        result = await bridge.scheduler_preview_template(name)
        return result if result else {"error": f"Template '{name}' not found"}

    elif action == "stats":
        return await bridge.scheduler_stats()

    elif action == "waves":
        return {"waves": await bridge.scheduler_execution_waves()}

    elif action == "run":
        if not name:
            return {"error": "run requires 'name'"}
        from core.di_container import get_container
        container = get_container()
        try:
            wf_exec = await container.orchestrator.run_workflow(
                name, context=args.get("context"),
            )
            return wf_exec.to_dict()
        except ValueError as exc:
            return {"error": str(exc)}

    elif action == "executions":
        from core.di_container import get_container
        container = get_container()
        return {"executions": container.orchestrator.list_workflow_executions()}

    elif action == "execution_status":
        exec_id = args.get("execution_id")
        if not exec_id:
            return {"error": "execution_status requires 'execution_id'"}
        from core.di_container import get_container
        container = get_container()
        result = container.orchestrator.get_workflow_execution(exec_id)
        return result if result else {"error": f"Execution '{exec_id}' not found"}

    elif action == "cancel_execution":
        exec_id = args.get("execution_id")
        if not exec_id:
            return {"error": "cancel_execution requires 'execution_id'"}
        from core.di_container import get_container
        container = get_container()
        cancelled = await container.orchestrator.cancel_workflow_execution(exec_id)
        return {"cancelled": cancelled, "execution_id": exec_id}

    else:
        return {"error": f"Unknown schedule action: {action}"}


async def _handle_notify(args: dict) -> dict:
    """Handle helix_notify tool."""
    from core.di_container import get_container
    from core.notifications.notification_service import Severity

    svc = get_container().notification_service
    action = args.get("action", "list")

    if action == "list":
        sev = None
        sev_str = args.get("severity")
        if sev_str:
            try:
                sev = Severity(sev_str)
            except ValueError:
                return {"error": f"Invalid severity: {sev_str}"}

        limit = args.get("limit", 20)
        items = await svc.list(severity=sev, limit=limit)
        return {
            "notifications": [
                {
                    "id": n.id,
                    "title": n.title,
                    "body": n.body,
                    "severity": n.severity.value,
                    "source": n.source,
                    "created_at": n.created_at.isoformat() if n.created_at else None,
                    "read": n.read,
                }
                for n in items
            ],
            "total": await svc.count(),
            "unread": await svc.count_unread(),
        }

    elif action == "get":
        nid = args.get("notification_id")
        if not nid:
            return {"error": "notification_id required for 'get' action"}
        n = await svc.get(nid)
        if n is None:
            return {"error": f"Notification '{nid}' not found"}
        return {
            "id": n.id, "title": n.title, "body": n.body,
            "severity": n.severity.value, "source": n.source,
            "metadata": n.metadata,
            "created_at": n.created_at.isoformat() if n.created_at else None,
            "read": n.read,
        }

    elif action == "mark_read":
        nid = args.get("notification_id")
        if not nid:
            return {"error": "notification_id required for 'mark_read' action"}
        found = await svc.mark_read(nid)
        if not found:
            return {"error": f"Notification '{nid}' not found"}
        return {"marked_read": True, "notification_id": nid}

    elif action == "unread_count":
        return {"unread": await svc.count_unread(), "total": await svc.count()}

    else:
        return {"error": f"Unknown notify action: {action}"}


async def main():
    """Run the MCP server on stdio."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        logger.info("Helix MCP server starting on stdio")
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
