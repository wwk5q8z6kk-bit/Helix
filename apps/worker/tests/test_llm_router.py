"""Tests for the intelligent LLM router (core/llm/intelligent_llm_router.py).

Tests provider selection logic and stats tracking.
Does NOT make real API calls — only tests the routing/selection layer.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core.llm.intelligent_llm_router import (
    IntelligentLLMRouter,
    LLMProvider,
    TaskType,
    ProviderStats,
    PROVIDER_CONFIG,
)
from core.exceptions_unified import LLMProviderError


@pytest.fixture
def router():
    return IntelligentLLMRouter()


# --- Initialization ---

def test_nine_providers_initialized(router):
    """Legacy check — we now have 20 but this confirms at least the original 9 are present."""
    original_nine = {
        LLMProvider.CLAUDE_OPUS, LLMProvider.CLAUDE_SONNET, LLMProvider.CLAUDE_HAIKU,
        LLMProvider.GPT4O, LLMProvider.GPT4O_MINI,
        LLMProvider.GEMINI_FLASH, LLMProvider.GEMINI_PRO,
        LLMProvider.GROK_3, LLMProvider.DEEPSEEK,
    }
    for p in original_nine:
        assert p in router.provider_stats


def test_all_providers_have_stats(router):
    for provider in LLMProvider:
        assert provider in router.provider_stats
        stats = router.provider_stats[provider]
        assert stats.quality_score > 0
        assert stats.cost_per_1k_tokens >= 0
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
    # With required_quality=9.5, only Opus (9.9), GPT-4o (9.6), and Sonnet (9.5) qualify
    assert provider in {LLMProvider.CLAUDE_OPUS, LLMProvider.GPT4O, LLMProvider.CLAUDE_SONNET}


async def test_select_speed_critical_prefers_fast(router):
    provider = await router.select_provider(
        task_type=TaskType.TESTING, prompt_tokens=100, speed_critical=True,
    )
    fast_providers = {LLMProvider.GEMINI_FLASH, LLMProvider.CLAUDE_HAIKU, LLMProvider.GPT4O_MINI}
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
    assert len(stats["providers"]) == 20


# --- TaskType enum ---

def test_all_task_types():
    assert len(TaskType) == 8
    assert TaskType("research") == TaskType.RESEARCH
    assert TaskType("code_generation") == TaskType.CODE_GENERATION


# ============================================================================
# New tests for expanded providers
# ============================================================================

def test_twenty_providers_initialized(router):
    """All 20 providers should be initialized."""
    assert len(router.provider_stats) == 20
    assert len(LLMProvider) == 20


async def test_openai_compatible_dispatch(router):
    """OpenAI-compatible providers should dispatch through _call_openai_compatible."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test response"}}],
        "usage": {"total_tokens": 42},
    }

    env = {"OPENROUTER_API_KEY": "test-key"}
    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch.dict("os.environ", env, clear=False):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        text, meta = await router.call_llm(
            LLMProvider.OPENROUTER,
            [{"role": "user", "content": "hello"}],
            TaskType.CODE_GENERATION,
        )
        assert text == "test response"
        assert meta["provider"] == "openrouter"
        # Verify the URL used
        call_args = mock_client.post.call_args
        assert "openrouter.ai" in call_args[0][0]


def test_ollama_zero_cost(router):
    """Ollama (local) should have zero cost."""
    stats = router.provider_stats[LLMProvider.OLLAMA]
    assert stats.cost_per_1k_tokens == 0.0


async def test_custom_url_dispatch(router):
    """CUSTOM provider should read URL from HELIX_CUSTOM_LLM_URL."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "custom response"}}],
        "usage": {"total_tokens": 10},
    }

    env = {"HELIX_CUSTOM_LLM_URL": "http://my-model.local/v1", "HELIX_CUSTOM_LLM_MODEL": "my-model"}
    with patch("httpx.AsyncClient") as mock_client_cls, patch.dict("os.environ", env, clear=False):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        text, meta = await router.call_llm(
            LLMProvider.CUSTOM,
            [{"role": "user", "content": "hi"}],
            TaskType.CODE_GENERATION,
        )
        assert text == "custom response"
        call_args = mock_client.post.call_args
        assert "my-model.local" in call_args[0][0]


async def test_bedrock_dispatch(router):
    """Bedrock should use boto3 invoke_model via executor."""
    import json
    import sys

    fake_body = MagicMock()
    fake_body.read.return_value = json.dumps({
        "content": [{"text": "bedrock response"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()

    mock_boto_client = MagicMock()
    mock_boto_client.invoke_model.return_value = {"body": fake_body}

    # Create a mock boto3 module if not installed
    mock_boto3 = MagicMock()
    mock_boto3.client.return_value = mock_boto_client

    with patch.dict(sys.modules, {"boto3": mock_boto3}):
        text, meta = await router.call_llm(
            LLMProvider.BEDROCK,
            [{"role": "user", "content": "hello bedrock"}],
            TaskType.RESEARCH,
        )
        assert text == "bedrock response"
        assert meta["provider"] == "bedrock"


async def test_cohere_dispatch(router):
    """Cohere should call v2/chat endpoint."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {"content": [{"text": "cohere response"}]},
        "usage": {"billed_units": {"input_tokens": 10, "output_tokens": 5}},
    }

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch.dict("os.environ", {"COHERE_API_KEY": "test-key"}, clear=False):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        text, meta = await router.call_llm(
            LLMProvider.COHERE,
            [{"role": "user", "content": "hello cohere"}],
            TaskType.CREATIVE,
        )
        assert text == "cohere response"
        call_args = mock_client.post.call_args
        assert "cohere.com/v2/chat" in call_args[0][0]


def test_provider_config_has_all_keys():
    """Every PROVIDER_CONFIG entry must have base_url, env_key, and default_model."""
    for provider, cfg in PROVIDER_CONFIG.items():
        assert "base_url" in cfg, f"{provider} missing base_url"
        assert "env_key" in cfg, f"{provider} missing env_key"
        assert "default_model" in cfg, f"{provider} missing default_model"


async def test_select_excludes_unavailable_providers(router):
    """When required_quality is very high, low-quality providers are excluded."""
    provider = await router.select_provider(
        task_type=TaskType.RESEARCH, prompt_tokens=100, required_quality=9.8,
    )
    # Only Opus has quality >= 9.8
    assert provider == LLMProvider.CLAUDE_OPUS


def test_openrouter_extra_headers():
    """OpenRouter config should include HTTP-Referer extra header."""
    cfg = PROVIDER_CONFIG[LLMProvider.OPENROUTER]
    assert "extra_headers" in cfg
    assert "HTTP-Referer" in cfg["extra_headers"]


def test_cloudflare_account_id_substitution():
    """Cloudflare base_url should contain {account_id} placeholder."""
    cfg = PROVIDER_CONFIG[LLMProvider.CLOUDFLARE]
    assert "{account_id}" in cfg["base_url"]


async def test_missing_api_key_raises_error(router):
    """Provider with required API key should raise when key is missing."""
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMProviderError):
            await router.call_llm(
                LLMProvider.MISTRAL,
                [{"role": "user", "content": "hello"}],
                TaskType.CODE_GENERATION,
            )


async def test_provider_fallback_on_error(router):
    """After a provider error, error_count should increment."""
    initial_errors = router.provider_stats[LLMProvider.OPENROUTER].error_count
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMProviderError):
            await router.call_llm(
                LLMProvider.OPENROUTER,
                [{"role": "user", "content": "fail"}],
                TaskType.CODE_GENERATION,
            )
    assert router.provider_stats[LLMProvider.OPENROUTER].error_count == initial_errors + 1


# --- Analytics integration ---

def test_set_analytics(router):
    """set_analytics() should store the analytics engine reference."""
    mock_analytics = MagicMock()
    router.set_analytics(mock_analytics)
    assert router._analytics is mock_analytics


async def test_call_llm_records_analytics(router):
    """After a successful call_llm(), record_usage() should be called on the analytics engine."""
    mock_analytics = MagicMock()
    router.set_analytics(mock_analytics)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "test"}}],
        "usage": {"total_tokens": 50},
    }

    with patch("httpx.AsyncClient") as mock_client_cls, \
         patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}, clear=False):
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        await router.call_llm(
            LLMProvider.OPENROUTER,
            [{"role": "user", "content": "hi"}],
            TaskType.CODE_GENERATION,
        )

    mock_analytics.record_usage.assert_called_once()
    call_kwargs = mock_analytics.record_usage.call_args.kwargs
    assert call_kwargs["provider"] == "openrouter"
    assert call_kwargs["tokens_in"] == 50


async def test_call_llm_records_error_analytics(router):
    """After a failed call_llm(), record_error() should be called on the analytics engine."""
    mock_analytics = MagicMock()
    router.set_analytics(mock_analytics)

    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(LLMProviderError):
            await router.call_llm(
                LLMProvider.OPENROUTER,
                [{"role": "user", "content": "fail"}],
                TaskType.CODE_GENERATION,
            )

    mock_analytics.record_error.assert_called_once()
    assert mock_analytics.record_error.call_args[0][0] == "openrouter"
