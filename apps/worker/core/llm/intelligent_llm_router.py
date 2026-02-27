"""
Intelligent LLM Router for Multi-Provider Support (Helix 2026)

Selects the best LLM provider for each task based on:
- Cost vs quality tradeoff
- Task complexity
- Speed requirements
- Budget constraints
- Provider availability
"""

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from core.exceptions_unified import LLMProviderError

if TYPE_CHECKING:
    from core.exceptions_unified import CircuitBreaker

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers — 2026 model roster"""
    # Anthropic
    CLAUDE_OPUS = "claude-opus-4-6"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    CLAUDE_HAIKU = "claude-haiku-4-5"
    # OpenAI
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    # Google
    GEMINI_FLASH = "gemini-2.0-flash"
    GEMINI_PRO = "gemini-2.5-pro"
    # xAI
    GROK_3 = "grok-3"
    # DeepSeek
    DEEPSEEK = "deepseek-r1"
    # OpenAI-compatible aggregators/providers
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    MISTRAL = "mistral-large-latest"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    PERPLEXITY = "perplexity"
    # Non-OpenAI-compatible
    COHERE = "cohere"
    BEDROCK = "bedrock"
    # More OpenAI-compatible
    CLOUDFLARE = "cloudflare"
    VENICE = "venice"
    # Arbitrary custom endpoint
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Provider configuration for OpenAI-compatible providers
# ---------------------------------------------------------------------------

PROVIDER_CONFIG: Dict[LLMProvider, Dict[str, Any]] = {
    LLMProvider.OPENROUTER: {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "anthropic/claude-sonnet-4-5",
        "extra_headers": {"HTTP-Referer": "https://helix.ai"},
    },
    LLMProvider.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "env_key": None,
        "default_model": "llama3.1:latest",
    },
    LLMProvider.MISTRAL: {
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "default_model": "mistral-large-latest",
    },
    LLMProvider.TOGETHER: {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    },
    LLMProvider.FIREWORKS: {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "env_key": "FIREWORKS_API_KEY",
        "default_model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
    },
    LLMProvider.PERPLEXITY: {
        "base_url": "https://api.perplexity.ai",
        "env_key": "PERPLEXITY_API_KEY",
        "default_model": "sonar-pro",
    },
    LLMProvider.CLOUDFLARE: {
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "env_key": "CLOUDFLARE_API_KEY",
        "default_model": "@cf/meta/llama-3.1-70b-instruct",
    },
    LLMProvider.VENICE: {
        "base_url": "https://api.venice.ai/api/v1",
        "env_key": "VENICE_API_KEY",
        "default_model": "llama-3.3-70b",
    },
}


class TaskType(Enum):
    """Task types with different optimization criteria"""
    RESEARCH = "research"
    DESIGN = "design"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    REFINEMENT = "refinement"
    REASONING = "reasoning"
    CREATIVE = "creative"


@dataclass
class ProviderStats:
    """Statistics for a provider"""
    provider: LLMProvider
    success_rate: float
    avg_latency_ms: float
    cost_per_1k_tokens: float
    quality_score: float  # 0-10
    availability: float  # 0-1
    last_used: Optional[datetime] = None
    request_count: int = 0
    error_count: int = 0

    def get_efficiency_score(self, task_type: TaskType) -> float:
        """Calculate efficiency score for task type"""
        quality_tasks = {
            TaskType.RESEARCH, TaskType.DESIGN,
            TaskType.CODE_GENERATION, TaskType.TESTING,
            TaskType.REASONING, TaskType.CREATIVE,
        }
        if task_type in quality_tasks:
            return (self.quality_score * 0.7 + (1 - self.avg_latency_ms / 5000) * 0.3) * self.success_rate
        else:
            return ((1 - self.avg_latency_ms / 5000) * 0.7 + self.quality_score / 10 * 0.3) * self.success_rate

    def get_cost_efficiency(self) -> float:
        """Cost per quality point"""
        if self.quality_score == 0:
            return float('inf')
        return self.cost_per_1k_tokens / self.quality_score


class IntelligentLLMRouter:
    """
    Routes requests to optimal LLM provider.

    Selects provider based on:
    - Task type and complexity
    - Current budget
    - Speed requirements
    - Provider health and availability
    """

    # Default provider statistics — class-level so other modules can read
    # provider quality/cost data without instantiating a full router.
    DEFAULT_PROVIDER_STATS: Dict[LLMProvider, "ProviderStats"] = {}

    def __init__(self):
        self.provider_stats: Dict[LLMProvider, ProviderStats] = {}
        self.budget_remaining = 100.0
        self.budget_limit = 100.0
        self.request_history: List[Dict] = []
        self._analytics = None
        self._initialize_providers()

    def _initialize_providers(self):
        """Initialize provider statistics with 2026 model data"""
        self.provider_stats = {
            LLMProvider.CLAUDE_OPUS: ProviderStats(
                provider=LLMProvider.CLAUDE_OPUS,
                success_rate=0.98, avg_latency_ms=1500,
                cost_per_1k_tokens=0.015, quality_score=9.9, availability=0.95,
            ),
            LLMProvider.CLAUDE_SONNET: ProviderStats(
                provider=LLMProvider.CLAUDE_SONNET,
                success_rate=0.97, avg_latency_ms=700,
                cost_per_1k_tokens=0.003, quality_score=9.5, availability=0.98,
            ),
            LLMProvider.CLAUDE_HAIKU: ProviderStats(
                provider=LLMProvider.CLAUDE_HAIKU,
                success_rate=0.96, avg_latency_ms=300,
                cost_per_1k_tokens=0.0008, quality_score=8.5, availability=0.99,
            ),
            LLMProvider.GPT4O: ProviderStats(
                provider=LLMProvider.GPT4O,
                success_rate=0.97, avg_latency_ms=1000,
                cost_per_1k_tokens=0.005, quality_score=9.6, availability=0.94,
            ),
            LLMProvider.GPT4O_MINI: ProviderStats(
                provider=LLMProvider.GPT4O_MINI,
                success_rate=0.96, avg_latency_ms=400,
                cost_per_1k_tokens=0.0003, quality_score=8.8, availability=0.97,
            ),
            LLMProvider.GEMINI_FLASH: ProviderStats(
                provider=LLMProvider.GEMINI_FLASH,
                success_rate=0.95, avg_latency_ms=250,
                cost_per_1k_tokens=0.0003, quality_score=8.2, availability=0.98,
            ),
            LLMProvider.GEMINI_PRO: ProviderStats(
                provider=LLMProvider.GEMINI_PRO,
                success_rate=0.96, avg_latency_ms=900,
                cost_per_1k_tokens=0.007, quality_score=9.4, availability=0.95,
            ),
            LLMProvider.GROK_3: ProviderStats(
                provider=LLMProvider.GROK_3,
                success_rate=0.95, avg_latency_ms=800,
                cost_per_1k_tokens=0.005, quality_score=9.3, availability=0.93,
            ),
            LLMProvider.DEEPSEEK: ProviderStats(
                provider=LLMProvider.DEEPSEEK,
                success_rate=0.94, avg_latency_ms=1200,
                cost_per_1k_tokens=0.0014, quality_score=9.2, availability=0.90,
            ),
            # New providers
            LLMProvider.OPENROUTER: ProviderStats(
                provider=LLMProvider.OPENROUTER,
                success_rate=0.95, avg_latency_ms=800,
                cost_per_1k_tokens=0.004, quality_score=9.0, availability=0.96,
            ),
            LLMProvider.OLLAMA: ProviderStats(
                provider=LLMProvider.OLLAMA,
                success_rate=0.92, avg_latency_ms=500,
                cost_per_1k_tokens=0.0, quality_score=7.5, availability=0.85,
            ),
            LLMProvider.MISTRAL: ProviderStats(
                provider=LLMProvider.MISTRAL,
                success_rate=0.95, avg_latency_ms=600,
                cost_per_1k_tokens=0.004, quality_score=9.1, availability=0.94,
            ),
            LLMProvider.TOGETHER: ProviderStats(
                provider=LLMProvider.TOGETHER,
                success_rate=0.94, avg_latency_ms=450,
                cost_per_1k_tokens=0.0009, quality_score=8.4, availability=0.93,
            ),
            LLMProvider.FIREWORKS: ProviderStats(
                provider=LLMProvider.FIREWORKS,
                success_rate=0.94, avg_latency_ms=350,
                cost_per_1k_tokens=0.0009, quality_score=8.3, availability=0.94,
            ),
            LLMProvider.PERPLEXITY: ProviderStats(
                provider=LLMProvider.PERPLEXITY,
                success_rate=0.93, avg_latency_ms=700,
                cost_per_1k_tokens=0.005, quality_score=8.8, availability=0.92,
            ),
            LLMProvider.COHERE: ProviderStats(
                provider=LLMProvider.COHERE,
                success_rate=0.94, avg_latency_ms=650,
                cost_per_1k_tokens=0.003, quality_score=8.6, availability=0.93,
            ),
            LLMProvider.BEDROCK: ProviderStats(
                provider=LLMProvider.BEDROCK,
                success_rate=0.97, avg_latency_ms=900,
                cost_per_1k_tokens=0.008, quality_score=9.3, availability=0.97,
            ),
            LLMProvider.CLOUDFLARE: ProviderStats(
                provider=LLMProvider.CLOUDFLARE,
                success_rate=0.93, avg_latency_ms=400,
                cost_per_1k_tokens=0.0005, quality_score=7.8, availability=0.95,
            ),
            LLMProvider.VENICE: ProviderStats(
                provider=LLMProvider.VENICE,
                success_rate=0.92, avg_latency_ms=550,
                cost_per_1k_tokens=0.001, quality_score=8.0, availability=0.91,
            ),
            LLMProvider.CUSTOM: ProviderStats(
                provider=LLMProvider.CUSTOM,
                success_rate=0.90, avg_latency_ms=800,
                cost_per_1k_tokens=0.002, quality_score=7.0, availability=0.80,
            ),
        }

        # Populate class-level defaults on first init so other modules
        # (e.g. BudgetTracker.suggest_cheaper_provider) can read provider
        # quality data without instantiating a full router.
        if not IntelligentLLMRouter.DEFAULT_PROVIDER_STATS:
            IntelligentLLMRouter.DEFAULT_PROVIDER_STATS = dict(self.provider_stats)

    async def select_provider(
        self,
        task_type: TaskType,
        prompt_tokens: int,
        required_quality: float = 8.0,
        speed_critical: bool = False,
        cost_critical: bool = False,
    ) -> LLMProvider:
        """Select optimal provider for task."""
        viable = [p for p in self.provider_stats.values() if p.quality_score >= required_quality]
        if not viable:
            viable = list(self.provider_stats.values())

        estimated_cost = (prompt_tokens / 1000) * max(p.cost_per_1k_tokens for p in viable)
        if estimated_cost > self.budget_remaining:
            viable = sorted(viable, key=lambda p: p.cost_per_1k_tokens)

        scored = []
        for provider in viable:
            if speed_critical:
                score = (1 - provider.avg_latency_ms / 5000) * 0.9 + (provider.quality_score / 10) * 0.1
            elif cost_critical:
                score = provider.get_cost_efficiency()
            else:
                score = provider.get_efficiency_score(task_type)
            scored.append((provider.provider, score))

        best = max(scored, key=lambda x: x[1])[0]
        logger.info(f"Selected {best.value} for {task_type.value}")
        return best

    async def call_llm(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        task_type: TaskType,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Tuple[str, Dict[str, Any]]:
        """Call LLM with selected provider."""
        import time
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

        start_time = time.time()

        # Circuit breaker pre-check (injected via set_circuit_breakers)
        breaker = getattr(self, "_circuit_breakers", {}).get(provider.value)
        if breaker and not breaker.can_execute():
            breaker.record_rejection()
            raise LLMProviderError(
                message=f"Circuit breaker open for {provider.value}",
            )

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
               retry=retry_if_exception_type((ConnectionError, TimeoutError)))
        async def _call():
            pv = provider.value

            # Check PROVIDER_CONFIG first (OpenAI-compatible providers)
            if provider in PROVIDER_CONFIG:
                cfg = PROVIDER_CONFIG[provider]
                env_key = cfg.get("env_key")
                api_key = os.getenv(env_key) if env_key else None
                if env_key and not api_key:
                    raise LLMProviderError(message=f"{env_key} not set")
                base_url = cfg["base_url"]
                # Cloudflare needs account_id substitution
                if "{account_id}" in base_url:
                    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
                    if not account_id:
                        raise LLMProviderError(message="CLOUDFLARE_ACCOUNT_ID not set")
                    base_url = base_url.replace("{account_id}", account_id)
                return await self._call_openai_compatible(
                    base_url=base_url,
                    api_key=api_key,
                    model=cfg["default_model"],
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_headers=cfg.get("extra_headers"),
                )

            # Cohere (non-OpenAI-compatible)
            if pv == "cohere":
                return await self._call_cohere(messages, max_tokens, temperature)
            # Bedrock (boto3 sync, run in executor)
            if pv == "bedrock":
                return await self._call_bedrock(messages, max_tokens, temperature)
            # Custom provider (custom:base_url format)
            if pv == "custom":
                custom_url = os.getenv("HELIX_CUSTOM_LLM_URL")
                custom_key = os.getenv("HELIX_CUSTOM_LLM_KEY")
                custom_model = os.getenv("HELIX_CUSTOM_LLM_MODEL", "default")
                if not custom_url:
                    raise LLMProviderError(message="HELIX_CUSTOM_LLM_URL not set")
                return await self._call_openai_compatible(
                    base_url=custom_url,
                    api_key=custom_key,
                    model=custom_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

            # Original providers
            if pv.startswith("claude"):
                return await self._call_anthropic(provider, messages, max_tokens, temperature)
            elif pv.startswith("gpt") or pv.startswith("o1"):
                return await self._call_openai(provider, messages, max_tokens, temperature)
            elif pv.startswith("gemini"):
                return await self._call_gemini(provider, messages, max_tokens, temperature)
            elif pv.startswith("grok"):
                return await self._call_grok(provider, messages, max_tokens, temperature)
            elif pv.startswith("deepseek"):
                return await self._call_deepseek(provider, messages, max_tokens, temperature)
            raise LLMProviderError(message=f"Unsupported provider: {pv}")

        try:
            text, tokens = await _call()
            latency_ms = int((time.time() - start_time) * 1000)
            stats = self.provider_stats[provider]
            cost = (tokens / 1000) * stats.cost_per_1k_tokens
            stats.request_count += 1
            stats.last_used = datetime.now()
            stats.avg_latency_ms = (stats.avg_latency_ms * 0.9) + (latency_ms * 0.1)
            self.budget_remaining -= cost
            if self._analytics is not None:
                self._analytics.record_usage(
                    provider=provider.value,
                    model=provider.value,
                    tokens_in=tokens,
                    tokens_out=0,
                    cost=cost,
                    latency_ms=float(latency_ms),
                )
            metadata = {"provider": provider.value, "task_type": task_type.value,
                        "tokens_used": tokens, "cost": cost, "latency_ms": latency_ms, "success": True}
            self.request_history.append({**metadata, "timestamp": datetime.now()})
            if breaker:
                breaker.record_success()
            return text, metadata
        except LLMProviderError as e:
            self.provider_stats[provider].error_count += 1
            if self._analytics is not None:
                self._analytics.record_error(provider.value, str(e))
            if breaker:
                breaker.record_failure()
            raise

    def set_circuit_breakers(self, breakers: Dict[str, "CircuitBreaker"]) -> None:
        """Inject per-provider circuit breakers (shared with ResourceManager)."""
        self._circuit_breakers = breakers

    def set_analytics(self, analytics_engine) -> None:
        """Inject analytics engine for usage recording."""
        self._analytics = analytics_engine

    async def _call_openai_compatible(
        self,
        base_url: str,
        api_key: Optional[str],
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, int]:
        """Call any OpenAI-compatible API endpoint."""
        import httpx

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            if r.status_code != 200:
                raise LLMProviderError(message=f"OpenAI-compatible API error ({base_url}): {r.status_code}")
            d = r.json()
            return d["choices"][0]["message"]["content"], d["usage"]["total_tokens"]

    async def _call_cohere(self, messages, max_tokens, temperature):
        """Call Cohere v2 chat API."""
        import httpx

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise LLMProviderError(message="COHERE_API_KEY not set")

        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                "https://api.cohere.com/v2/chat",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "command-r-plus",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            if r.status_code != 200:
                raise LLMProviderError(message=f"Cohere API error: {r.status_code}")
            d = r.json()
            text = d["message"]["content"][0]["text"]
            tokens = d.get("usage", {}).get("billed_units", {})
            total = tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0)
            return text, total

    async def _call_bedrock(self, messages, max_tokens, temperature):
        """Call AWS Bedrock via boto3 (sync, run in executor)."""
        import json as _json

        def _invoke():
            import boto3

            client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            response = client.invoke_model(
                modelId=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
                contentType="application/json",
                accept="application/json",
                body=_json.dumps(body),
            )
            result = _json.loads(response["body"].read())
            text = result["content"][0]["text"]
            total = result["usage"]["input_tokens"] + result["usage"]["output_tokens"]
            return text, total

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _invoke)

    async def _call_anthropic(self, provider, messages, max_tokens, temperature):
        import httpx

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMProviderError(message="ANTHROPIC_API_KEY not set")
        model_map = {
            LLMProvider.CLAUDE_OPUS: "claude-opus-4-6-20250501",
            LLMProvider.CLAUDE_SONNET: "claude-sonnet-4-5-20250929",
            LLMProvider.CLAUDE_HAIKU: "claude-haiku-4-5-20251001",
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post("https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": model_map.get(provider, "claude-sonnet-4-5-20250929"),
                      "max_tokens": max_tokens, "temperature": temperature, "messages": messages})
            if r.status_code != 200:
                raise LLMProviderError(message=f"Anthropic API error: {r.status_code}")
            d = r.json()
            return d["content"][0]["text"], d["usage"]["input_tokens"] + d["usage"]["output_tokens"]

    async def _call_openai(self, provider, messages, max_tokens, temperature):
        import httpx

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError(message="OPENAI_API_KEY not set")
        model_map = {LLMProvider.GPT4O: "gpt-4o", LLMProvider.GPT4O_MINI: "gpt-4o-mini"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_map.get(provider, "gpt-4o-mini"),
                      "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
            if r.status_code != 200:
                raise LLMProviderError(message=f"OpenAI API error: {r.status_code}")
            d = r.json()
            return d["choices"][0]["message"]["content"], d["usage"]["total_tokens"]

    async def _call_gemini(self, provider, messages, max_tokens, temperature):
        import httpx

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMProviderError(message="GOOGLE_API_KEY not set")
        model = "gemini-2.0-flash" if provider == LLMProvider.GEMINI_FLASH else "gemini-2.5-pro"
        prompt = "\n".join(m.get("content", "") for m in messages)
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}})
            if r.status_code != 200:
                raise LLMProviderError(message=f"Gemini API error: {r.status_code}")
            d = r.json()
            text = d["candidates"][0]["content"]["parts"][0]["text"]
            tokens = d.get("usageMetadata", {}).get("totalTokenCount", len(prompt.split()) + len(text.split()))
            return text, tokens

    async def _call_grok(self, provider, messages, max_tokens, temperature):
        import httpx

        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise LLMProviderError(message="XAI_API_KEY not set")
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post("https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "grok-3", "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
            if r.status_code != 200:
                raise LLMProviderError(message=f"Grok API error: {r.status_code}")
            d = r.json()
            return d["choices"][0]["message"]["content"], d["usage"]["total_tokens"]

    async def _call_deepseek(self, provider, messages, max_tokens, temperature):
        import httpx

        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMProviderError(message="DEEPSEEK_API_KEY not set")
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post("https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "deepseek-reasoner", "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
            if r.status_code != 200:
                raise LLMProviderError(message=f"DeepSeek API error: {r.status_code}")
            d = r.json()
            return d["choices"][0]["message"]["content"], d["usage"]["total_tokens"]

    async def get_router_stats(self) -> Dict[str, Any]:
        return {
            "budget_remaining": f"${self.budget_remaining:.2f}",
            "budget_limit": f"${self.budget_limit:.2f}",
            "total_requests": sum(p.request_count for p in self.provider_stats.values()),
            "total_errors": sum(p.error_count for p in self.provider_stats.values()),
            "providers": [
                {"name": p.provider.value, "requests": p.request_count,
                 "success_rate": f"{p.success_rate*100:.1f}%", "avg_latency_ms": f"{p.avg_latency_ms:.0f}",
                 "quality_score": f"{p.quality_score:.1f}/10", "cost_per_1k": f"${p.cost_per_1k_tokens:.5f}"}
                for p in self.provider_stats.values()
            ],
        }


_router: Optional[IntelligentLLMRouter] = None


async def get_llm_router() -> IntelligentLLMRouter:
    global _router
    if _router is None:
        _router = IntelligentLLMRouter()
        logger.info("Initialized Helix LLM router with 20 providers (2026 models)")
    return _router
