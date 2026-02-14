#!/usr/bin/env bash
# Helix Platform Launcher — starts Rust core + Python AI service
set -euo pipefail

HELIX_HOME="${HELIX_HOME:-$HOME/.helix}"
HELIX_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUST_PORT=9470
PYTHON_PORT=8200
LOG_DIR="$HELIX_HOME/logs"

mkdir -p "$HELIX_HOME" "$LOG_DIR"

log() { echo "[helix] $(date '+%H:%M:%S') $*"; }

# --- Load secrets from macOS Keychain (if available) ---
load_secrets() {
    for key in ANTHROPIC_API_KEY OPENAI_API_KEY GOOGLE_API_KEY XAI_API_KEY DEEPSEEK_API_KEY; do
        if [ -z "${!key:-}" ]; then
            val=$(security find-generic-password -s "helix-$key" -w 2>/dev/null || true)
            if [ -n "$val" ]; then
                export "$key=$val"
                log "Loaded $key from Keychain"
            fi
        fi
    done
}

# --- Start Rust Core ---
start_rust() {
    log "Starting Rust core on :$RUST_PORT..."
    if command -v hx &>/dev/null; then
        hx server start &>"$LOG_DIR/rust-core.log" &
    elif [ -f "$HELIX_ROOT/rust/target/release/hx" ]; then
        "$HELIX_ROOT/rust/target/release/hx" server start &>"$LOG_DIR/rust-core.log" &
    else
        log "WARNING: hx binary not found. Rust core not started."
        log "Build with: cd $HELIX_ROOT/rust && cargo build --release -p hx-cli"
        return 1
    fi
    RUST_PID=$!
    log "Rust core PID: $RUST_PID"

    # Wait for health
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$RUST_PORT/health" &>/dev/null; then
            log "Rust core healthy on :$RUST_PORT"
            return 0
        fi
        sleep 1
    done
    log "WARNING: Rust core health check timed out"
    return 1
}

# --- Start Python AI ---
start_python() {
    log "Starting Python AI on :$PYTHON_PORT..."
    cd "$HELIX_ROOT/python"

    if [ -d ".venv" ]; then
        source .venv/bin/activate
    fi

    python3 -m uvicorn core.api.helix_api:app \
        --host 127.0.0.1 \
        --port "$PYTHON_PORT" \
        --log-level info \
        &>"$LOG_DIR/python-ai.log" &
    PYTHON_PID=$!
    log "Python AI PID: $PYTHON_PID"

    # Wait for health
    for i in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$PYTHON_PORT/health" &>/dev/null; then
            log "Python AI healthy on :$PYTHON_PORT"
            return 0
        fi
        sleep 1
    done
    log "WARNING: Python AI health check timed out"
    return 1
}

# --- Main ---
main() {
    log "=== Helix Platform Starting ==="
    log "Home: $HELIX_HOME"
    log "Root: $HELIX_ROOT"

    load_secrets

    start_rust || log "Rust core failed to start (continuing with Python AI only)"
    start_python || { log "Python AI failed to start"; exit 1; }

    log "=== Helix Platform Ready ==="
    log "  Rust Core:  http://127.0.0.1:$RUST_PORT"
    log "  Python AI:  http://127.0.0.1:$PYTHON_PORT"
    log "  Logs:       $LOG_DIR/"

    # Wait for children
    wait
}

main "$@"
