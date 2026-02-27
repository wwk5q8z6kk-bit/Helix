# Helix

A dual-layer AI platform combining a Rust systems core with a Python AI orchestration layer.

The Rust layer handles storage, search, knowledge graphs, credentials, tunnels, adapters, a job queue, workflow engine, and notifications. The Python layer provides LLM routing across 20 providers, task orchestration, collaborative agent swarms, multi-step reasoning, budget tracking, and analytics.

## Architecture

```
helix/
├── apps/
│   ├── api/             # axum REST + tonic gRPC + WebSocket server
│   ├── cli/             # CLI binary (setup wizard, chat, keychain, MCP)
│   └── worker/          # Python AI layer (FastAPI :8200)
├── packages/
│   ├── core/            # Domain types, credentials (keyring + encrypted file)
│   ├── engine/          # Orchestration (adapters, tunnels, jobs, workflows, notifications, multimodal)
│   ├── graph/           # petgraph knowledge graph
│   ├── mcp/             # MCP protocol + plugin system
│   ├── memory/          # SQLite + LanceDB vector storage, Tantivy full-text search
│   └── plugins/         # Plugin runtime
├── config/              # Default configuration
├── infra/               # Docker Compose
├── migrations/          # SQLite schema migrations
└── scripts/             # start.sh, stop.sh, health-check.sh
```

## Quick Start

### Prerequisites

- Rust toolchain (stable)
- Python 3.11+
- `protoc` (`brew install protobuf`)

### Rust Core

```bash
cargo build --release
# Binary: target/release/hx
```

### Python AI Layer

```bash
cd apps/worker
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m uvicorn core.api.helix_api:app --port 8200
```

### Docker

```bash
docker compose -f infra/docker-compose.yml up
```

### Ports

| Service | Port |
|---------|------|
| Rust REST API | 9470 |
| Rust gRPC | 50051 |
| Python AI API | 8200 |

## Configuration

Copy `.env.example` to `.env` and fill in at least one LLM provider key. See the file for all available options.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `HELIX_HOME` | `~/.helix` | Data directory |
| `HELIX_ENV` | `development` | Environment (`development` / `production`) |
| `HELIX_DAILY_BUDGET` | `100` | Daily LLM spend cap (USD) |
| `HELIX_TOOL_MODE` | `sovereign` | Tool mode (`sovereign`, `composio`, `hybrid`) |

## CLI

```bash
hx setup [--quick]           # Interactive setup wizard
hx chat [--server URL]       # Chat REPL
hx store <content>           # Store knowledge node
hx recall <query>            # Recall knowledge
hx search <query>            # Full-text search
hx server start [--port N]   # Start server
hx keychain init             # Initialize sealed vault
hx mcp                       # Start MCP server on stdio
```

## Key Components

**Rust Core**
- **Multimodal Pipeline** — 9 processors (PDF, Image, Audio, DOCX, HTML, CSV/Excel, EPUB, Code, JSON/YAML)
- **Workflow Engine** — TOML/JSON definitions with conditions, loops, parallelism
- **Job Queue** — SQLite-backed durable queue with retry, dead-letter, priority
- **Notification Router** — multi-channel dispatch with alert rules, cooldown, quiet hours
- **DAG Scheduler** — dependency-aware task scheduling with quality gates
- **Adapters** — Slack, Discord, Email, Telegram, Matrix, Webhook, CLI Chat
- **Tunnels** — Cloudflare, Ngrok, Tailscale, Bore, SSH, Custom

**Python AI Layer**
- **LLM Router** — routes to 20 providers (Claude, GPT, Gemini, Grok, DeepSeek, OpenRouter, Ollama, Mistral, Together, Fireworks, Perplexity, Cloudflare, Venice, Cohere, Bedrock, custom)
- **Swarm Orchestration** — collaborative agent swarms for complex tasks
- **Reasoning Engine** — multi-step agentic reasoning with process reward models
- **Budget Tracker** — per-provider cost tracking with daily caps and auto-downgrade
- **MCP Server** — stdio MCP with 9 tools + Composio integration
- **Analytics** — token usage trends, provider performance, search quality metrics

## Testing

```bash
# Rust
cargo check --workspace
cargo test --workspace

# Python
cd apps/worker
pytest -v
```

## License

[Apache-2.0](LICENSE)
