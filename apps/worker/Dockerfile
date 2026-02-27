# Helix Python AI Layer
# FastAPI app with LLM routing, orchestration, swarms, reasoning

FROM python:3.12-slim

RUN pip install uv

WORKDIR /app

# Copy dependency manifest first for caching
COPY pyproject.toml ./
RUN uv pip install -e . --system

# Copy application code
COPY . .

EXPOSE 8200

CMD ["uvicorn", "core.api.helix_api:app", "--host", "0.0.0.0", "--port", "8200"]
