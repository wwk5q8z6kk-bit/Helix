# Contributing to Helix

Thank you for your interest in contributing to Helix.

## Getting Started

1. Fork the repository and clone your fork.
2. Install prerequisites: Rust (stable), Python 3.12+, `protoc` (`brew install protobuf` or `apt install protobuf-compiler`).
3. Build and test:

```bash
# Rust
cargo build --workspace
cargo test --workspace

# Python
cd apps/worker
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -v
```

## Submitting Changes

1. Create a feature branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure all checks pass:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

cd apps/worker
ruff check .
ruff format --check .
pytest -v
```

4. Open a pull request against `main`.

## Code Style

- **Rust:** Standard `cargo fmt` and `clippy` with `-D warnings`.
- **Python:** `ruff` for linting and formatting, line-length 120.
- Keep changes focused — one feature or fix per PR.

## Reporting Issues

Open a GitHub issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Environment details (OS, Rust version, Python version)

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
