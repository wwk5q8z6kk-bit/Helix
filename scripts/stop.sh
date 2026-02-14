#!/usr/bin/env bash
# Helix Platform Shutdown — stops Rust core and Python AI service
set -euo pipefail

log() { echo "[helix] $(date '+%H:%M:%S') $*"; }

log "Stopping Helix services..."

if pkill -f "hx server start" 2>/dev/null; then
    log "Stopped Rust core"
else
    log "Rust core was not running"
fi

if pkill -f "uvicorn core.api.helix_api" 2>/dev/null; then
    log "Stopped Python AI"
else
    log "Python AI was not running"
fi

log "Done"
