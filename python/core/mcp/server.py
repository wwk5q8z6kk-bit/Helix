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
    """Return available Helix tools."""
    return [
        Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
        for t in get_tool_definitions()
    ]


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


async def main():
    """Run the MCP server on stdio."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        logger.info("Helix MCP server starting on stdio")
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
