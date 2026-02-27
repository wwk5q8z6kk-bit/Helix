# Helix

Dual-layer AI platform: Rust core (storage, search, graph, credentials, tunnels, adapters, job queue, workflow engine, notifications) + Python AI layer (LLM routing, orchestration, swarms, reasoning, budget tracking, analytics, Composio).

## Architecture

```
helix/
├── apps/
│   ├── api/             # axum REST + tonic gRPC + WebSocket server (hx-server)
│   ├── cli/             # CLI binary: setup wizard, chat, keychain, MCP (hx-cli)
│   └── worker/          # Python AI layer (FastAPI :8200)
│       └── core/
│           ├── api/             # helix_api.py, event_router.py
│           ├── llm/             # 20-provider LLM router
│           ├── orchestration/   # Task orchestrator
│           ├── swarms/          # Collaborative agent swarms
│           ├── reasoning/       # Multi-step reasoning + PRM
│           ├── mcp/             # MCP stdio server (9 tools + Composio)
│           ├── middleware/      # Rate limiting, security, budget tracker
│           └── analytics/       # Usage trends, reports
├── packages/
│   ├── core/            # Domain types, credentials (keyring + encrypted file)
│   ├── engine/
│   │   ├── engine/      # Orchestration (adapters, tunnels, jobs, workflows, notifications, multimodal)
│   │   └── scheduler/   # DAG-based task scheduling, quality gates
│   ├── graph/           # petgraph knowledge graph
│   ├── mcp/             # MCP protocol + plugin system
│   ├── memory/
│   │   ├── storage/     # SQLite + LanceDB vector storage
│   │   └── index/       # Tantivy full-text search
│   └── plugins/         # WASM plugin runtime
├── config/              # Default configuration (helix.toml)
├── infra/               # Docker Compose
├── migrations/          # SQLite schema migrations
└── scripts/             # start.sh, stop.sh, health-check.sh
```

## Build & Run

### Rust
```bash
cargo build --release
# Binary: target/release/hx
# Requires: protoc (brew install protobuf)
```

### Python
```bash
cd apps/worker
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                    # All tests
python -m uvicorn core.api.helix_api:app --port 8200
```

### Ports
- Rust REST: 9470, gRPC: 50051
- Python API: 8200

## Key Entry Points
- **Python API**: `apps/worker/core/api/helix_api.py` — FastAPI app
- **LLM Router**: `apps/worker/core/llm/intelligent_llm_router.py` — routes to 20 providers (Claude, GPT, Gemini, Grok, DeepSeek, OpenRouter, Ollama, Mistral, Together, Fireworks, Perplexity, Cloudflare, Venice, Cohere, Bedrock, custom:URL)
- **MCP Server**: `apps/worker/core/mcp/server.py` — stdio MCP with 9 tools (helix_schedule 13 actions, helix_notify 4 actions) + Composio
- **Rust Bridge**: `apps/worker/core/rust_bridge.py` — async HTTP client to Rust core
- **Orchestrator**: `apps/worker/core/orchestration/unified_orchestrator.py` — task decomposition + delegation (wired to LLM)
- **DI Container**: `apps/worker/core/di_container.py` — lazy service wiring
- **Event Router**: `apps/worker/core/api/event_router.py` — external event classification + routing
- **Tower Log**: `apps/worker/core/observability/tower_log.py` — structured JSONL event logging
- **Feedback Handler**: `apps/worker/core/feedback/feedback_handler.py` — quality feedback → learning optimizer
- **Budget Tracker**: `apps/worker/core/middleware/budget_tracker.py` — per-provider cost tracking with daily caps
- **Composio Bridge**: `apps/worker/core/composio/composio_bridge.py` — Composio tool discovery + execution
- **Analytics Engine**: `apps/worker/core/analytics/analytics_engine.py` — token usage trends, provider performance, search quality metrics
- **Report Generator**: `apps/worker/core/analytics/report_generator.py` — LLM-powered namespace summaries and trend reports
- **Multimodal Pipeline**: `packages/engine/engine/src/multimodal/mod.rs` — 9 processors (PDF, Image, Audio, DOCX, HTML, CSV/Excel, EPUB, Code, JSON/YAML)
- **Source Connectors**: `packages/engine/engine/src/sources/mod.rs` — SourceConnector trait + directory, RSS, GitHub, URL scraper
- **Workflow Engine**: `packages/engine/engine/src/workflow/mod.rs` — TOML/JSON workflow definitions with conditions, loops, parallelism
- **Job Queue**: `packages/engine/engine/src/jobs/queue.rs` — SQLite-backed durable queue with retry, dead-letter, priority
- **Notification Router**: `packages/engine/engine/src/notifications/router.rs` — multi-channel dispatch with alert rules, cooldown, quiet hours
- **Rate Limiter**: `packages/engine/engine/src/rate_limit.rs` — per-adapter rate limiting (requests/min, /hour, burst)
- **Scheduling Service**: `packages/engine/engine/src/scheduling.rs` — async facade over hx-scheduler DAG coordinator

## API Endpoints

### Python API (:8200)
- `GET /health` — health check (200 healthy, 503 degraded if Rust core down)
- `POST /api/generate` — LLM generation via router
- `POST /api/search` — semantic search via Rust core
- `POST /api/reason` — multi-step agentic reasoning
- `POST /api/swarm/execute` — specialized swarm execution
- `GET /api/stats` — system statistics
- `POST /api/event` — external event ingestion (→ EventRouter)
- `GET /tower` — recent tower events (ring buffer)
- `GET /api/board` — aggregated system dashboard
- `GET /api/progress/{task_id}` — SSE progress streaming
- `POST /api/feedback` — quality feedback (score 0.0-1.0)
- `GET /api/budget` — cost/budget dashboard
- `GET /api/composio/tools` — list Composio tools
- `POST /api/composio/execute` — execute Composio tool
- `GET /api/composio/oauth/{app_name}` — initiate OAuth flow
- `GET /api/analytics/dashboard` — system dashboard metrics
- `GET /api/analytics/trends` — token usage and activity trends
- `GET /api/analytics/providers` — provider performance stats
- `GET /api/analytics/token-usage` — detailed token usage breakdown
- `POST /api/reports/generate` — generate LLM-powered report
- `GET /api/reports/{report_id}` — retrieve generated report
- **Scheduler (Python→Rust delegation)**: `POST/GET /api/scheduler/workflows`, `GET/DELETE /api/scheduler/workflows/{name}`, `POST /api/scheduler/workflows/{name}/preview`, `POST /api/scheduler/workflows/{name}/run`, `GET /api/scheduler/templates`, `POST /api/scheduler/templates/{name}/preview`, `GET /api/scheduler/stats`, `GET /api/scheduler/waves`, `GET /api/scheduler/executions`, `GET /api/scheduler/executions/{id}`, `POST /api/scheduler/executions/{id}/cancel`, `GET /api/scheduler/executions/{id}/progress` (SSE)
- **Notifications (Python AI layer)**: `GET /api/notifications` (?severity, ?unread_only, ?limit), `GET /api/notifications/{id}`, `POST /api/notifications/{id}/read`

### Rust API (:9470)
- `POST /api/v1/adapters` — register adapter (Slack, Discord, Email, Telegram, Matrix, Webhook, CLI Chat)
- `GET /api/v1/adapters` — list adapters
- `GET /api/v1/adapters/statuses` — all adapter statuses
- `GET /api/v1/adapters/:id` — adapter status
- `DELETE /api/v1/adapters/:id` — remove adapter
- `POST /api/v1/adapters/:id/send` — send message
- `POST /api/v1/adapters/:id/health` — health check
- `POST /api/v1/tunnels` — register and start tunnel (Cloudflare, Ngrok, Tailscale, Bore, SSH, Custom)
- `GET /api/v1/tunnels` — list tunnels
- `GET /api/v1/tunnels/:id` — tunnel status
- `DELETE /api/v1/tunnels/:id` — stop and remove tunnel
- `POST /api/v1/tunnels/:id/health` — tunnel health check
- `POST /api/v1/pair/initiate` — gateway pairing (admin, returns OTP)
- `POST /api/v1/pair/confirm` — confirm pairing (returns bearer token)
- **Sources**: `GET/POST /api/v1/sources`, `GET/DELETE /api/v1/sources/:id`, `POST /api/v1/sources/:id/poll`
- **Jobs**: `GET /api/v1/jobs`, `GET /api/v1/jobs/stats`, `GET /api/v1/jobs/dead-letter`, `POST /api/v1/jobs/purge`, `GET /api/v1/jobs/:id`, `POST /api/v1/jobs/:id/retry`, `POST /api/v1/jobs/:id/cancel`
- **Workflows**: `GET /api/v1/workflows`, `GET /api/v1/workflows/:id`, `POST /api/v1/workflows/:id/execute`, `GET /api/v1/workflows/executions`, `GET /api/v1/workflows/executions/:id`, `POST /api/v1/workflows/executions/:id/cancel`
- **Scheduler**: `POST/GET /api/v1/scheduler/workflows`, `GET/DELETE /api/v1/scheduler/workflows/:name`, `POST /api/v1/scheduler/tasks`, `GET /api/v1/scheduler/tasks/ready`, `GET /api/v1/scheduler/stats`
- **Notifications**: `GET /api/v1/notifications`, `GET /api/v1/notifications/:id`, `POST /api/v1/notifications/:id/read`, `GET/POST /api/v1/notifications/alerts`, `PUT/DELETE /api/v1/notifications/alerts/:id`
- **Audit**: `GET /api/v1/audit/query`
- **API Keys**: `GET/POST /api/v1/keys`, `GET/DELETE /api/v1/keys/:id`
- **Outbound Webhooks**: `GET/POST /api/v1/webhooks/outbound`, `GET/DELETE /api/v1/webhooks/outbound/:id`, `POST /api/v1/webhooks/outbound/:id/test`, `GET /api/v1/webhooks/outbound/:id/deliveries`
- **Plugin Marketplace**: `GET /api/v1/plugins/marketplace`, `GET /api/v1/plugins/marketplace/search`, `GET /api/v1/plugins/marketplace/:id`, `POST /api/v1/plugins/marketplace/:id/install`
- **Rate Limits**: `GET/PUT /api/v1/rate-limits`

## Environment Variables

### LLM Provider API Keys
- `ANTHROPIC_API_KEY` — Claude models
- `OPENAI_API_KEY` — GPT models
- `GOOGLE_API_KEY` — Gemini models
- `XAI_API_KEY` — Grok models
- `DEEPSEEK_API_KEY` — DeepSeek models
- `OPENROUTER_API_KEY` — OpenRouter
- `MISTRAL_API_KEY` — Mistral
- `TOGETHER_API_KEY` — Together AI
- `FIREWORKS_API_KEY` — Fireworks AI
- `PERPLEXITY_API_KEY` — Perplexity
- `CLOUDFLARE_API_KEY` / `CLOUDFLARE_ACCOUNT_ID` — Cloudflare AI
- `VENICE_API_KEY` — Venice AI
- `COHERE_API_KEY` — Cohere
- AWS Bedrock uses `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

### Helix Configuration
- `HELIX_HOME` — data directory (default: `~/.helix`)
- `HELIX_ENV` — environment (`development` allows `*` CORS; production restricts origins)
- `HELIX_CORS_ORIGINS` — comma-separated allowed origins (overrides default)
- `HELIX_RUST_URL` — Rust core URL (default: `http://127.0.0.1:9470`)
- `HELIX_DAILY_BUDGET` — daily LLM spend cap (default: $100)
- `HELIX_BUDGET_ACTION` — action when budget exceeded: `downgrade` or `reject` (default: `downgrade`)
- `COMPOSIO_API_KEY` — Composio integration
- `HELIX_TOOL_MODE` — tool mode: `sovereign`, `composio`, or `hybrid` (default: `sovereign`)

## CLI Commands
```bash
hx setup [--quick]           # Interactive setup wizard (or quick defaults)
hx chat [--server URL]       # Interactive chat REPL
hx store <content>           # Store knowledge node
hx recall <query>            # Recall knowledge
hx search <query>            # Full-text search
hx server start [--port N]   # Start server
hx keychain init             # Initialize sealed vault
hx mcp                       # Start MCP server on stdio
```

## Testing
```bash
cd apps/worker && source .venv/bin/activate
pytest -v                              # All tests
pytest tests/test_helix_api.py         # API endpoint tests
pytest tests/test_llm_router.py        # Router selection tests
pytest tests/test_rust_bridge.py       # Bridge mock tests
pytest tests/test_di_and_event_bus.py  # DI + event bus tests
pytest tests/test_event_router.py      # Event routing tests
pytest tests/test_tower.py             # Tower logging tests
pytest tests/test_feedback.py          # Feedback loop tests
pytest tests/test_swarms.py            # Swarm routing tests
pytest tests/test_reasoning.py         # Reasoning engine tests
pytest tests/test_mcp.py               # MCP tool dispatch tests
pytest tests/test_integration.py       # Integration tests
pytest tests/test_budget_tracker.py    # Budget tracking tests
pytest tests/test_composio.py          # Composio integration tests
pytest tests/test_analytics.py         # Analytics engine tests
pytest tests/test_reports.py           # Report generator tests
pytest tests/test_scheduled_actions.py # Scheduled maintenance tests
pytest tests/test_rust_scheduler_integration.py  # Rust scheduler delegation tests
```

```bash
cargo check                           # Full workspace check
cargo test --lib -p hx-server         # Server tests (apps/api)
cargo test --lib -p hx-engine         # Engine tests (packages/engine/engine)
cargo test --lib -p hx-cli            # CLI tests (apps/cli)
```

## Conventions
- Python: ruff, line-length 120, Python 3.11+
- Rust: standard cargo fmt/clippy
- Magic bytes: `HXB1` (sealed blob prefix)
- Exceptions: `core/exceptions_unified.py` (single hierarchy)
- Interfaces: `core/interfaces/` (Protocol-based, structural subtyping)
- Validation: `apps/api/src/validation.rs` uses typed `ValidationError` enum (thiserror)
- Dependencies: `sse-starlette` for SSE streaming
- Adapters: `ExternalAdapter` trait (send/poll/health_check/status)
- Tunnels: `Tunnel` trait (start/stop/health_check/status) — child processes with `kill_on_drop`
- Channels: `Channel` trait wraps adapters with platform normalization
- Multimodal: `ModalityProcessor` trait (name/handles/process/status) — registered in `MultiModalPipeline`
- Sources: `SourceConnector` trait (poll/health_check/status) — `SourceRegistry` follows `AdapterRegistry` pattern
- Jobs: `JobQueue` backed by SQLite with exponential backoff (`min(5 * 2^retries, 3600)`)
- Workflows: TOML/JSON definitions, variable interpolation `{{var.name}}` / `{{result.step.field}}`
- Notifications: `NotificationChannel` trait, `AlertRule` conditions with cooldown + quiet hours
- Config: `packages/engine/engine/src/config.rs` — all config structs with `#[serde(default)]` for backward compat
- Scheduling: Rust-first DAG scheduling — orchestrator delegates to Rust core via `RustCoreBridge` when available, falls back to Python `DependencyResolver` when offline

## Infrastructure
- `scripts/start.sh` — starts Rust core + Python AI
- `scripts/stop.sh` — stops both services
- `.github/workflows/ci.yml` — Python pytest + Rust check/test
- `infra/docker-compose.yml` — helix-rust (:9470) + helix-python (:8200)
- `apps/worker/Dockerfile` — Python 3.12-slim + uv for fast installs
- `.env.example` — template for all environment variables
