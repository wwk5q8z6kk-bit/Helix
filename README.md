# Helix

A dual-layer AI platform combining a Rust systems core with a Python AI orchestration layer. Local-first by default — all data stays on your machine.

The Rust layer handles storage, search, knowledge graphs, credentials, encryption, tunnels, adapters, a job queue, workflow engine, sync, federation, and notifications. The Python layer provides LLM routing across 20 providers, task orchestration, collaborative agent swarms, multi-step reasoning, budget tracking, and analytics.

Both layers share a dual-layer middleware pipeline — auth, rate limiting, circuit breakers, output sanitization, audit logging, and security headers are enforced consistently across Rust (Tower/Axum) and Python (Starlette/FastAPI), with budget enforcement and safeguards (blocked senders, auto-approve rules with confidence thresholds) layered on top.

## Architecture

```
helix/
├── apps/
│   ├── api/             # axum REST + tonic gRPC + WebSocket server
│   ├── cli/             # CLI binary (setup wizard, chat, keychain, MCP)
│   └── worker/          # Python AI layer (FastAPI :8200)
│       └── core/
│           ├── api/             # helix_api.py, event_router.py
│           ├── llm/             # 20-provider LLM router
│           ├── orchestration/   # Task orchestrator
│           ├── swarms/          # Collaborative agent swarms
│           ├── reasoning/       # Multi-step reasoning + PRM
│           ├── mcp/             # MCP stdio server (9 tools + Composio)
│           ├── middleware/      # Rate limiting, security, budget tracker
│           ├── analytics/       # Usage trends, reports
│           ├── composio/        # Composio tool bridge
│           ├── feedback/        # Quality feedback loop
│           ├── learning/        # Learning optimizer
│           ├── scheduling/      # DAG coordination
│           ├── sources/         # Source connector orchestration
│           └── observability/   # Tower structured logging
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

## Quick Start

### Prerequisites

- Rust toolchain (stable)
- Python 3.11+ with [uv](https://github.com/astral-sh/uv)
- `protoc` (`brew install protobuf`)

### Build & Run

```bash
# Rust core
cargo build --release
# Binary: target/release/hx

# Start server
./target/release/hx server start --foreground

# Python AI layer
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

API docs are served by the running Rust server at `/api/docs` (Swagger UI) and `/api/openapi.json`.

## CLI

```bash
hx setup [--quick]           # Interactive setup wizard
hx chat [--server URL]       # Chat REPL
hx store <content>           # Store knowledge node
hx recall <query>            # Recall knowledge (semantic + full-text)
hx search <query>            # Full-text search
hx server start [--port N]   # Start server
hx keychain init             # Initialize sealed vault
hx secret set/get/list/delete # Manage secrets
hx mcp                       # Start MCP server on stdio
hx config show/set/validate  # Manage configuration
hx backup                    # Backup data
hx export                    # Export nodes
hx import                    # Import data (Obsidian supported)
hx graph                     # Graph traversal
hx stats                     # System statistics
hx db                        # Database maintenance
hx encrypt/decrypt           # Encryption utilities
```

## Configuration

Default config: `config/helix.toml`. CLI uses `~/.helix/config.toml`.

Copy `.env.example` to `.env` and fill in at least one LLM provider key.

### Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HELIX_HOME` | `~/.helix` | Data directory |
| `HELIX_ENV` | `development` | Environment (`development` / `production`) |
| `HELIX_DAILY_BUDGET` | `100` | Daily LLM spend cap (USD) |
| `HELIX_TOOL_MODE` | `sovereign` | Tool mode (`sovereign`, `composio`, `hybrid`) |
| `HELIX_RUST_URL` | `http://127.0.0.1:9470` | Rust core URL for Python layer |

### LLM Provider Keys

At least one is needed for the Python AI layer:

- `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`, `DEEPSEEK_API_KEY`
- `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `PERPLEXITY_API_KEY`
- `CLOUDFLARE_API_KEY` + `CLOUDFLARE_ACCOUNT_ID`, `VENICE_API_KEY`, `COHERE_API_KEY`
- AWS Bedrock: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

### Embedding Providers

Helix supports local and remote embeddings:

1. `local_fastembed` (default) — local ONNX inference, no API key needed. Build with `--features local-embeddings`.
2. `openai` — remote API, requires `OPENAI_API_KEY`.

Recommended local models: `bge-small-en-v1.5`, `all-minilm-l6-v2`.

## Authentication

Helix supports multiple auth modes:

1. **Shared bearer token** — `HELIX_AUTH_TOKEN`
2. **HS256 JWT** — `HELIX_JWT_SECRET`
3. **OAuth2 client-credentials** — local flow backed by the Sovereign Keychain

### Roles and Namespace Scope

Enforced across REST, gRPC, and WebSocket:

- **Roles:** `admin` (all namespaces), `write` (scoped), `read` (scoped read-only)
- **Shared token scope:** `HELIX_AUTH_ROLE`, `HELIX_AUTH_NAMESPACE`
- **JWT claims:** `role`, `namespace`
- **Rate limiting:** `HELIX_RATE_LIMIT_REQUESTS` (default 120/min), `HELIX_RATE_LIMIT_WINDOW_SECS`
- **Namespace quotas:** `HELIX_NAMESPACE_NODE_QUOTA`

### Consumer Profiles & ABAC

- Consumer profiles with attribute-based access control (ABAC)
- Permission templates define tiers (`view`, `edit`, `action`, `admin`) and scopes
- Access keys map to templates and can be created/revoked via REST

### Public Shares

Read-only node sharing with high-entropy tokens:

- `POST /api/v1/shares` — create share
- `GET /public/shares/{token}` — view (HTML or JSON via `Accept` header)
- `DELETE /api/v1/shares/{id}` — revoke

## Middleware Pipeline

Helix runs a dual-layer request processing pipeline across both Rust and Python:

**Rust (Tower/Axum):**
- Auth middleware — JWT, shared token, OAuth2, and consumer token resolution across REST, gRPC, and WebSocket
- Audit middleware — structured audit logging with query API
- Metrics middleware — Prometheus-style counters, latency histograms, `/metrics` endpoint
- Rate limiting — per-identity with configurable window
- Namespace quotas — per-namespace node limits
- Circuit breaker — Closed/Open/HalfOpen state machine for downstream service protection
- Output sanitizer — automatic secret redaction (raw + base64 variants) in command output
- Safeguards — blocked sender lists, auto-approve rules with confidence thresholds
- CORS + request tracing

**Python (Starlette/FastAPI):**
- Security headers — CSP, HSTS, X-Frame-Options, X-Content-Type-Options, Referrer-Policy, Permissions-Policy
- Rate limiting — token bucket algorithm with Redis backend and in-memory fallback
- Request ID — correlation IDs for distributed tracing
- Budget enforcement — per-provider LLM cost tracking with daily caps
- CORS with configurable origins

## Encryption & Keychain

### Encryption at Rest

AES-256-GCM + Argon2id key derivation. Enable with `HELIX_ENCRYPTION_ENABLED=true`.

### Sovereign Keychain

Sealed vault with Shamir secret sharing for credential storage:

- `hx keychain init` — initialize vault
- `hx secret set/get/list/delete` — manage secrets
- REST: `/api/v1/keychain/*`, `/api/v1/secrets/*`
- Supports keyring + AES-256-GCM encrypted file + environment variable backends

## Rust Core Features

### Hybrid Search

Three search backends combined with configurable weights:

- **SQLite FTS5** — full-text search with BM25 ranking
- **LanceDB** — vector similarity search with local or remote embeddings
- **petgraph** — knowledge graph traversal and relationship-aware boosting

### Adapters

External communication adapters with health checks and rate limiting:

- Slack, Discord, Email (IMAP/SMTP), Telegram, Matrix, Webhook, CLI Chat
- `ExternalAdapter` trait: send/poll/health_check/status
- REST: `/api/v1/adapters/*`

### Tunnels

Expose local services via reverse tunnels:

- Cloudflare, Ngrok, Tailscale, Bore, SSH, Custom
- `Tunnel` trait with child process management (`kill_on_drop`)
- REST: `/api/v1/tunnels/*`

### Multimodal Pipeline

9 content processors for document ingestion:

- PDF, Image (OCR), Audio (Whisper transcription), DOCX, HTML, CSV/Excel, EPUB, Code, JSON/YAML
- `ModalityProcessor` trait registered in `MultiModalPipeline`

### Workflow Engine

Declarative workflows defined in TOML/JSON:

- Variable interpolation (`{{var.name}}`, `{{result.step.field}}`)
- Conditions, loops, parallelism
- REST: `/api/v1/workflows/*`

### Job Queue

SQLite-backed durable queue:

- Retry with exponential backoff (`min(5 * 2^retries, 3600)`)
- Dead-letter queue, priority levels
- REST: `/api/v1/jobs/*`

### DAG Scheduler

Dependency-aware task scheduling with quality gates:

- Rust-first DAG coordination
- Python delegation via `RustCoreBridge` with offline fallback
- REST: `/api/v1/scheduler/*`

### Notification Router

Multi-channel dispatch (Slack, Email, Webhook, In-App):

- Alert rules with conditions, cooldown, quiet hours
- Outbound webhooks with delivery tracking
- REST: `/api/v1/notifications/*`, `/api/v1/webhooks/outbound/*`

### Source Connectors

Ingest from external sources:

- Directory watcher, RSS feeds, GitHub repos, URL scraper
- `SourceConnector` trait: poll/health_check/status
- REST: `/api/v1/sources/*`

### Federation

Peer-to-peer knowledge sharing between Helix instances:

- REST: `/api/v1/federation/*`
- Gateway pairing with OTP: `/api/v1/pair/*`

### Sync

Vector-clock based sync with conflict resolution:

- Clock-based versioning
- Snapshot support
- REST: `/api/v1/sync/*`, `/api/v1/conflicts/*`

## Knowledge Management

### AI Auto-Tagging

Hybrid tag enrichment during ingest/update:

1. Lexical keyword extraction from title/content
2. Semantic tag transfer from similar existing nodes

Configure via `[ai]` in `helix.toml` or `HELIX_AI_AUTO_TAGGING_*` env vars.

### Auto Backlinking

Automatic `references` relationships from content references:

- Wiki links: `[[Target Title]]`, `[[Target Title|alias]]`
- Markdown links: `[label](Target Title)`
- Mentions: `@TargetSlug`, `@"Target Title"`
- Source URL references

Configure via `[linking]` in `helix.toml`.

### Daily Notes

Template-driven daily notes with midnight scheduler:

- `POST /api/v1/daily-notes/ensure` — idempotent creation
- Auto-linking for `task` and `event` nodes
- Configurable templates via `[daily_notes]`

### Recurring Tasks

Recurrence templates with automatic instance materialization:

- Frequencies: daily, weekly, monthly with interval/count/until
- Instances linked to templates via `derived_from`
- Auto-linked to daily notes
- REST: `/api/v1/tasks/due`, `/api/v1/tasks/prioritize`

### Focus Planner

Deterministic task prioritization using importance + due date + status + effort heuristics. Optional metadata: `task_priority`, `task_status`, `task_estimate_minutes`.

### Templates

Template packs with version history, instantiation, and variable substitution:

- Built-in template packs: `GET /api/v1/template-packs`
- Template CRUD with versioning and restore
- Instantiation with `{{variable}}` substitution

### Node Versioning

Rolling version history for all nodes with field-level change tracking and restore.

### Calendar

- `GET /api/v1/calendar/items` — day/week/month views
- `GET /api/v1/calendar/ical` — iCal export
- `POST /api/v1/calendar/ical/import` — iCal import
- Google Calendar sync via `[google_calendar]` config

### Attachments

File attachments with extraction and search:

- Upload: `POST /api/v1/files/upload` (max 10 MB)
- PDF extraction (`pdftotext`), image OCR (`tesseract`), audio transcription (Whisper)
- Searchable text chunks per attachment

### Import/Export

- `GET /api/v1/export` — full vault export with relationships
- `POST /api/v1/import` — bulk import with namespace override
- Obsidian vault import via CLI

### Saved Searches & Views

Persistent queries and view configurations:

- REST: `/api/v1/search/saved/*`, `/api/v1/saved_views/*`
- View types: list, kanban, calendar

## AI Features (Rust)

### Watcher Agent

Proactive monitoring agent that scans vault activity, detects intents, and generates insight proposals. Configure via `[watcher]` in `helix.toml`.

### Autonomy Rules

Policy engine controlling when agentic actions auto-apply vs defer:

- Rule types: `global`, `domain`, `contact`, `tag` with cascading priority
- Controls: confidence threshold, quiet hours, max actions/hour, allow/deny intent types
- Feedback loop: `POST /api/v1/agent/feedback`, reflection stats, calibration

### AI Writing Assist

Retrieval-assisted completion and linking:

- `POST /api/v1/assist/completion` — sentence suggestions with grounding
- `POST /api/v1/assist/autocomplete` — inline prefix completions
- `POST /api/v1/assist/links` — semantic wiki-link targets
- `POST /api/v1/assist/transform` — summarize, action items, refine

### LLM Integration

OpenAI-compatible chat completion for assist/briefing features:

```toml
[llm]
enabled = true
base_url = "http://localhost:11434/v1"
model = "llama3.2"
```

Supports Ollama, any OpenAI-compatible endpoint, or remote providers via `HELIX_LLM_API_KEY`.

### MCP Server

Model Context Protocol server over stdio for agent interop:

- `hx mcp --access-key <token>`
- Tools scoped to access key namespace/tags/kinds
- Write operations submit proposals for owner approval

## Python AI Layer

### LLM Router

Intelligent routing across 20 providers:

Claude, GPT, Gemini, Grok, DeepSeek, OpenRouter, Ollama, Mistral, Together, Fireworks, Perplexity, Cloudflare, Venice, Cohere, Bedrock, and custom endpoints.

### Swarm Orchestration

Collaborative agent swarms for complex multi-step tasks with specialized roles.

### Reasoning Engine

Multi-step agentic reasoning with process reward models (PRM) for quality evaluation.

### Budget Tracker

Per-provider cost tracking with daily caps:

- `HELIX_DAILY_BUDGET` — daily spend limit (default $100)
- `HELIX_BUDGET_ACTION` — `downgrade` (auto-switch to cheaper model) or `reject`
- REST: `GET /api/budget`

### Composio Integration

External tool execution via Composio:

- `COMPOSIO_API_KEY` — integration key
- `HELIX_TOOL_MODE` — `sovereign` (native only), `composio` (external only), `hybrid` (both)
- OAuth flows for connected apps

### Analytics

- Token usage trends and activity metrics
- Provider performance stats
- LLM-powered report generation

### Event Router

External event classification and routing into the platform.

### Feedback & Learning

Quality feedback loop (scores 0.0-1.0) feeding into a learning optimizer for continuous improvement.

## Infrastructure

### Relay & Exchange

Contact-based communication relay with identity management:

- Relay contacts with vault addresses
- Exchange inbox for inbound/outbound message management
- Conversations threading

### API Keys & Webhooks

- `GET/POST /api/v1/keys` — API key management
- `GET/POST /api/v1/webhooks/outbound` — outbound webhook registration with delivery tracking

### Plugin System

WASM plugin runtime with marketplace:

- `GET /api/v1/plugins/marketplace` — browse and install plugins

### Audit Logging

Structured audit trail: `GET /api/v1/audit/query`

### Metrics & Diagnostics

- Prometheus-style metrics at `/metrics`
- Embedding diagnostics: `GET /api/v1/diagnostics/embedding`

### Profile

Owner profile for outbound identity (display name, email, timezone, signature):

```toml
[profile]
display_name = "Helix Owner"
primary_email = ""
timezone = "UTC"
```

## Testing

```bash
# Rust
cargo check --workspace
cargo test --workspace

# Python
cd apps/worker
source .venv/bin/activate
pytest -v
```

## License

[Apache-2.0](LICENSE)
