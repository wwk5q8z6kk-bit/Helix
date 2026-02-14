# Helix

Dual-layer AI platform: Rust core (storage, search, graph, credentials) + Python AI layer (LLM routing, orchestration, swarms, reasoning).

## Architecture

```
helix/
├── rust/crates/         # 9 Rust crates (workspace)
│   ├── hx-core          # Domain types, credentials (keyring + encrypted file)
│   ├── hx-storage       # SQLite + LanceDB vector storage
│   ├── hx-index         # Tantivy full-text search
│   ├── hx-graph         # petgraph knowledge graph
│   ├── hx-engine        # Orchestrates core subsystems
│   ├── hx-server        # axum REST + tonic gRPC + UDS
│   ├── hx-cli           # CLI binary
│   ├── hx-adapters      # External integrations
│   └── hx-proto         # Protobuf definitions
└── python/              # Python AI layer
    ├── core/
    │   ├── api/
    │   │   ├── helix_api.py              # FastAPI app (:8200)
    │   │   └── event_router.py           # External event classification + routing
    │   ├── llm/intelligent_llm_router.py  # 9-provider LLM router
    │   ├── rust_bridge.py            # HTTP client to Rust core (:9470)
    │   ├── di_container.py           # Lazy DI wiring
    │   ├── event_bus.py              # In-memory pub/sub
    │   ├── orchestration/            # Task orchestrator (wired to real LLM)
    │   ├── swarms/                   # Collaborative agent swarms
    │   ├── reasoning/                # Multi-step reasoning + PRM
    │   ├── learning/                 # RL feedback + optimizer
    │   ├── feedback/                 # Quality feedback handler
    │   ├── observability/            # Tower structured logging (JSONL)
    │   ├── mcp/                      # MCP stdio server (7 tools)
    │   └── middleware/               # Request ID, rate limiting, security headers
    ├── tests/                        # pytest (74 tests)
    └── scripts/                      # start.sh, stop.sh
```

## Build & Run

### Rust
```bash
cd rust && cargo build --release
# Binary: target/release/hx
# Requires: protoc (brew install protobuf)
```

### Python
```bash
cd python
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest                    # 74 tests
python -m uvicorn core.api.helix_api:app --port 8200
```

### Ports
- Rust REST: 9470, gRPC: 50051
- Python API: 8200

## Key Entry Points
- **Python API**: `python/core/api/helix_api.py` — FastAPI app
- **LLM Router**: `python/core/llm/intelligent_llm_router.py` — routes to Claude/GPT/Gemini/Grok/DeepSeek
- **MCP Server**: `python/core/mcp/server.py` — stdio MCP with 7 tools
- **Rust Bridge**: `python/core/rust_bridge.py` — async HTTP client to Rust core
- **Orchestrator**: `python/core/orchestration/unified_orchestrator.py` — task decomposition + delegation (wired to LLM)
- **DI Container**: `python/core/di_container.py` — lazy service wiring
- **Event Router**: `python/core/api/event_router.py` — external event classification + routing
- **Tower Log**: `python/core/observability/tower_log.py` — structured JSONL event logging
- **Feedback Handler**: `python/core/feedback/feedback_handler.py` — quality feedback → learning optimizer

## API Endpoints
- `GET /health` — health check (Rust core + Python)
- `POST /api/generate` — LLM generation via router
- `POST /api/search` — semantic search via Rust core
- `GET /api/stats` — system statistics
- `POST /api/event` — external event ingestion (→ EventRouter)
- `GET /tower` — recent tower events (ring buffer)
- `GET /api/board` — aggregated system dashboard
- `GET /api/progress/{task_id}` — SSE progress streaming
- `POST /api/feedback` — quality feedback (score 0.0-1.0)

## Environment Variables
- `ANTHROPIC_API_KEY` — Claude models
- `OPENAI_API_KEY` — GPT models
- `GOOGLE_API_KEY` — Gemini models
- `XAI_API_KEY` — Grok models
- `DEEPSEEK_API_KEY` — DeepSeek models
- `HELIX_HOME` — data directory (default: `~/.helix`)

## Testing
```bash
cd python && source .venv/bin/activate
pytest -v              # All 74 tests
pytest tests/test_helix_api.py        # API endpoint tests (14)
pytest tests/test_llm_router.py       # Router selection tests (12)
pytest tests/test_rust_bridge.py      # Bridge mock tests (10)
pytest tests/test_di_and_event_bus.py # DI + event bus tests (12)
pytest tests/test_event_router.py     # Event routing tests (9)
pytest tests/test_tower.py            # Tower logging tests (11)
pytest tests/test_feedback.py         # Feedback loop tests (6)
```

```bash
cd rust
cargo check                           # Full workspace check
cargo test --lib -p hx-server         # Server tests (209 pass, 7 pre-existing vault-sealed)
cargo test --lib -p hx-cli            # CLI tests
```

## Conventions
- Python: ruff, line-length 120, Python 3.11+
- Rust: standard cargo fmt/clippy
- Magic bytes: `HXB1` (sealed blob prefix)
- Exceptions: `core/exceptions_unified.py` (single hierarchy)
- Interfaces: `core/interfaces/` (Protocol-based, structural subtyping)
- Validation: `hx-server/src/validation.rs` uses typed `ValidationError` enum (thiserror)
- Dependencies: `sse-starlette` for SSE streaming

## Infrastructure
- `scripts/start.sh` — starts Rust core + Python API
- `scripts/stop.sh` — stops both services
- `.github/workflows/ci.yml` — Python pytest + Rust check/test
- `docker-compose.yml` — helix-rust (:9470) + helix-python (:8200)
