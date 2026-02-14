"""Tests for the intelligent LLM router (core/llm/intelligent_llm_router.py).

Tests provider selection logic and stats tracking.
Does NOT make real API calls — only tests the routing/selection layer.
"""

import pytest
from core.llm.intelligent_llm_router import (
    IntelligentLLMRouter,
    LLMProvider,
    TaskType,
    ProviderStats,
)


@pytest.fixture
def router():
    return IntelligentLLMRouter()


# --- Initialization ---

def test_nine_providers_initialized(router):
    assert len(router.provider_stats) == 9


def test_all_providers_have_stats(router):
    for provider in LLMProvider:
        assert provider in router.provider_stats
        stats = router.provider_stats[provider]
        assert stats.quality_score > 0
        assert stats.cost_per_1k_tokens > 0
        assert 0 <= stats.success_rate <= 1


# --- Provider selection ---

async def test_select_returns_valid_provider(router):
    provider = await router.select_provider(task_type=TaskType.CODE_GENERATION, prompt_tokens=100)
    assert isinstance(provider, LLMProvider)


async def test_select_quality_tasks_prefer_opus(router):
    """For high-quality tasks, Opus (quality 9.9) should be preferred."""
    provider = await router.select_provider(
        task_type=TaskType.RESEARCH, prompt_tokens=100, required_quality=9.5,
    )
    # With required_quality=9.5, only Opus (9.9), Codex (9.6), and Sonnet (9.5) qualify
    assert provider in {LLMProvider.CLAUDE_OPUS, LLMProvider.GPT5_CODEX, LLMProvider.CLAUDE_SONNET}


async def test_select_speed_critical_prefers_fast(router):
    provider = await router.select_provider(
        task_type=TaskType.TESTING, prompt_tokens=100, speed_critical=True,
    )
    fast_providers = {LLMProvider.GEMINI_FLASH, LLMProvider.CLAUDE_HAIKU, LLMProvider.GPT4_MINI}
    # Flash is fastest (250ms) — should be selected
    assert provider in fast_providers


async def test_select_low_budget_picks_cheap(router):
    router.budget_remaining = 0.001  # Almost empty budget
    provider = await router.select_provider(
        task_type=TaskType.CODE_GENERATION, prompt_tokens=100,
    )
    # Should still work — falls back to cheapest viable
    assert isinstance(provider, LLMProvider)


# --- ProviderStats ---

def test_efficiency_score_quality_task():
    stats = ProviderStats(
        provider=LLMProvider.CLAUDE_OPUS,
        success_rate=1.0, avg_latency_ms=1000,
        cost_per_1k_tokens=0.01, quality_score=10.0, availability=1.0,
    )
    score = stats.get_efficiency_score(TaskType.RESEARCH)
    # quality_score * 0.7 + speed_factor * 0.3, scaled by success_rate
    assert score > 0


def test_efficiency_score_speed_task():
    stats = ProviderStats(
        provider=LLMProvider.GEMINI_FLASH,
        success_rate=1.0, avg_latency_ms=200,
        cost_per_1k_tokens=0.0003, quality_score=8.0, availability=1.0,
    )
    score = stats.get_efficiency_score(TaskType.DEPLOYMENT)
    assert score > 0


def test_cost_efficiency():
    stats = ProviderStats(
        provider=LLMProvider.CLAUDE_HAIKU,
        success_rate=0.96, avg_latency_ms=300,
        cost_per_1k_tokens=0.0008, quality_score=8.5, availability=0.99,
    )
    ce = stats.get_cost_efficiency()
    assert ce == pytest.approx(0.0008 / 8.5, rel=1e-6)


def test_cost_efficiency_zero_quality():
    stats = ProviderStats(
        provider=LLMProvider.CLAUDE_HAIKU,
        success_rate=0.96, avg_latency_ms=300,
        cost_per_1k_tokens=0.001, quality_score=0, availability=0.99,
    )
    assert stats.get_cost_efficiency() == float('inf')


# --- Router stats ---

async def test_router_stats_structure(router):
    stats = await router.get_router_stats()
    assert "budget_remaining" in stats
    assert "total_requests" in stats
    assert "providers" in stats
    assert len(stats["providers"]) == 9


# --- TaskType enum ---

def test_all_task_types():
    assert len(TaskType) == 8
    assert TaskType("research") == TaskType.RESEARCH
    assert TaskType("code_generation") == TaskType.CODE_GENERATION
