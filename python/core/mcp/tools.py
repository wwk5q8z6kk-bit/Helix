"""
Helix MCP Tool Definitions

Each tool maps to the orchestrator/router/swarm system:
- helix_orchestrate: Run a task through the unified orchestrator
- helix_generate: Generate content via intelligent LLM routing
- helix_search: Semantic search in the knowledge base (via Rust core)
- helix_swarm: Execute a specialized swarm for complex tasks
- helix_review: Code review using the code review swarm
"""

from typing import Any, Dict

TOOL_DEFINITIONS = [
    {
        "name": "helix_orchestrate",
        "description": "Run a task through the Helix unified orchestrator. Routes to the best agent/model combination based on task type, quality requirements, and budget.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task description"},
                "task_type": {
                    "type": "string",
                    "enum": ["research", "design", "code_generation", "testing", "deployment", "refinement", "reasoning", "creative"],
                    "description": "Type of task for optimal routing",
                },
                "quality": {"type": "number", "description": "Minimum quality score 0-10 (default 8.0)", "default": 8.0},
                "speed_critical": {"type": "boolean", "description": "Prioritize speed over quality", "default": False},
            },
            "required": ["task"],
        },
    },
    {
        "name": "helix_generate",
        "description": "Generate content using the intelligent LLM router. Automatically selects the best provider based on task requirements.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt to send to the LLM"},
                "provider": {
                    "type": "string",
                    "enum": ["auto", "claude-opus-4-6", "claude-sonnet-4-5", "claude-haiku-4-5",
                             "gpt-5.3-codex", "gpt-4.1-mini", "gemini-3-flash", "gemini-3-pro",
                             "grok-4", "deepseek-r1"],
                    "description": "Provider to use (default: auto-select)",
                    "default": "auto",
                },
                "max_tokens": {"type": "integer", "description": "Maximum tokens to generate", "default": 4096},
                "temperature": {"type": "number", "description": "Temperature 0.0-2.0", "default": 0.7},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "helix_search",
        "description": "Semantic search in the Helix knowledge base. Searches stored memories, notes, and knowledge nodes using vector similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Maximum results", "default": 10},
                "namespace": {"type": "string", "description": "Namespace to search in", "default": "default"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "helix_swarm",
        "description": "Execute a specialized agent swarm for complex multi-step tasks. Available swarms: implementation, code_review, testing, debugging, architecture, deployment, requirements.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task for the swarm to execute"},
                "swarm_type": {
                    "type": "string",
                    "enum": ["implementation", "code_review", "testing", "debugging", "architecture", "deployment", "requirements"],
                    "description": "Type of specialized swarm",
                },
                "num_agents": {"type": "integer", "description": "Number of agents in swarm", "default": 3},
                "context": {"type": "object", "description": "Additional context for the swarm"},
            },
            "required": ["task", "swarm_type"],
        },
    },
    {
        "name": "helix_review",
        "description": "Review code using the code review swarm. Multiple agents analyze code in parallel for bugs, style, security, and performance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Code to review"},
                "language": {"type": "string", "description": "Programming language"},
                "focus": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["bugs", "security", "performance", "style", "architecture"]},
                    "description": "Review focus areas",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "helix_reason",
        "description": "Multi-step reasoning using the Agentic Reasoner. Breaks down complex goals into steps and reasons through each one.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "The reasoning goal or question"},
                "max_steps": {"type": "integer", "description": "Maximum reasoning steps", "default": 5},
                "context": {"type": "object", "description": "Additional context for reasoning"},
            },
            "required": ["goal"],
        },
    },
    {
        "name": "helix_learn",
        "description": "Query learned patterns and strategies from the pattern library. Returns matching patterns based on task similarity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Pattern search query"},
                "limit": {"type": "integer", "description": "Maximum patterns to return", "default": 5},
            },
            "required": ["query"],
        },
    },
]


def get_tool_definitions() -> list:
    """Return MCP tool definitions."""
    return TOOL_DEFINITIONS
