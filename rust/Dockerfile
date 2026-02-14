# Helix Rust Core — Multi-stage Docker Build
# Builds the hx CLI binary with all 9 workspace crates

# --- Builder stage ---
FROM rust:1.83-slim AS builder

RUN apt-get update && apt-get install -y protobuf-compiler pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ crates/
COPY proto/ proto/
COPY migrations/ migrations/

RUN cargo build --release -p hx-cli

# --- Runtime stage ---
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y libssl3 curl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/hx /usr/local/bin/hx

EXPOSE 9470 50051

CMD ["hx", "server", "start"]
