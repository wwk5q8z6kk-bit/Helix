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
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from core.exceptions_unified import LLMProviderError

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers — 2026 model roster"""
    # Anthropic
    CLAUDE_OPUS = "claude-opus-4-6"
    CLAUDE_SONNET = "claude-sonnet-4-5"
    CLAUDE_HAIKU = "claude-haiku-4-5"
    # OpenAI
    GPT5_CODEX = "gpt-5.3-codex"
    GPT4_MINI = "gpt-4.1-mini"
    # Google
    GEMINI_FLASH = "gemini-3-flash"
    GEMINI_PRO = "gemini-3-pro"
    # xAI
    GROK_4 = "grok-4"
    # DeepSeek
    DEEPSEEK = "deepseek-r1"


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

    def __init__(self):
        self.provider_stats: Dict[LLMProvider, ProviderStats] = {}
        self.budget_remaining = 100.0
        self.budget_limit = 100.0
        self.request_history: List[Dict] = []
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
            LLMProvider.GPT5_CODEX: ProviderStats(
                provider=LLMProvider.GPT5_CODEX,
                success_rate=0.97, avg_latency_ms=1000,
                cost_per_1k_tokens=0.010, quality_score=9.6, availability=0.94,
            ),
            LLMProvider.GPT4_MINI: ProviderStats(
                provider=LLMProvider.GPT4_MINI,
                success_rate=0.96, avg_latency_ms=400,
                cost_per_1k_tokens=0.0004, quality_score=8.8, availability=0.97,
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
            LLMProvider.GROK_4: ProviderStats(
                provider=LLMProvider.GROK_4,
                success_rate=0.95, avg_latency_ms=800,
                cost_per_1k_tokens=0.005, quality_score=9.3, availability=0.93,
            ),
            LLMProvider.DEEPSEEK: ProviderStats(
                provider=LLMProvider.DEEPSEEK,
                success_rate=0.94, avg_latency_ms=1200,
                cost_per_1k_tokens=0.0014, quality_score=9.2, availability=0.90,
            ),
        }

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

        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
               retry=retry_if_exception_type((ConnectionError, TimeoutError)))
        async def _call():
            pv = provider.value
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
            metadata = {"provider": provider.value, "task_type": task_type.value,
                        "tokens_used": tokens, "cost": cost, "latency_ms": latency_ms, "success": True}
            self.request_history.append({**metadata, "timestamp": datetime.now()})
            return text, metadata
        except LLMProviderError:
            self.provider_stats[provider].error_count += 1
            raise

    async def _call_anthropic(self, provider, messages, max_tokens, temperature):
        import os, httpx
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
        import os, httpx
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMProviderError(message="OPENAI_API_KEY not set")
        model_map = {LLMProvider.GPT5_CODEX: "gpt-5.3-codex", LLMProvider.GPT4_MINI: "gpt-4.1-mini"}
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_map.get(provider, "gpt-4.1-mini"),
                      "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
            if r.status_code != 200:
                raise LLMProviderError(message=f"OpenAI API error: {r.status_code}")
            d = r.json()
            return d["choices"][0]["message"]["content"], d["usage"]["total_tokens"]

    async def _call_gemini(self, provider, messages, max_tokens, temperature):
        import os, httpx
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMProviderError(message="GOOGLE_API_KEY not set")
        model = "gemini-3-flash" if provider == LLMProvider.GEMINI_FLASH else "gemini-3-pro"
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
        import os, httpx
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise LLMProviderError(message="XAI_API_KEY not set")
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post("https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "grok-4", "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
            if r.status_code != 200:
                raise LLMProviderError(message=f"Grok API error: {r.status_code}")
            d = r.json()
            return d["choices"][0]["message"]["content"], d["usage"]["total_tokens"]

    async def _call_deepseek(self, provider, messages, max_tokens, temperature):
        import os, httpx
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
        logger.info("Initialized Helix LLM router with 9 providers (2026 models)")
    return _router
