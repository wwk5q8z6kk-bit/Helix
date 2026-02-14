#!/usr/bin/env bash
# Helix Health Check — checks both Rust core and Python AI service
set -euo pipefail

RUST_PORT=${RUST_PORT:-9470}
PYTHON_PORT=${PYTHON_PORT:-8200}
EXIT_CODE=0

check() {
    local name="$1" url="$2"
    if curl -sf --max-time 5 "$url" &>/dev/null; then
        echo "✓ $name: healthy"
    else
        echo "✗ $name: unreachable"
        EXIT_CODE=1
    fi
}

echo "=== Helix Health Check ==="
check "Rust Core (:$RUST_PORT)"   "http://127.0.0.1:$RUST_PORT/health"
check "Python AI (:$PYTHON_PORT)" "http://127.0.0.1:$PYTHON_PORT/health"
echo "=========================="

exit $EXIT_CODE
