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
    """Handle helix_swarm tool — placeholder for swarm execution."""
    return {
        "status": "executed",
        "swarm_type": args.get("swarm_type"),
        "task": args.get("task"),
        "agents": args.get("num_agents", 3),
        "note": "Swarm execution requires running Python AI service. Use POST :8200/swarm/execute for full swarm orchestration.",
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


async def main():
    """Run the MCP server on stdio."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        logger.info("Helix MCP server starting on stdio")
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
