"""Integration tests for orchestrator review methods wired to LLM router."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def orchestrator():
    """Create an orchestrator with a mocked LLM router."""
    from core.orchestration.unified_orchestrator import UnifiedOrchestrator

    event_bus = MagicMock()
    memory_store = MagicMock()
    llm_router = AsyncMock()
    return UnifiedOrchestrator(
        event_bus=event_bus,
        memory_store=memory_store,
        llm_router=llm_router,
    )


@pytest.fixture
def orchestrator_no_llm():
    """Create an orchestrator without an LLM router."""
    from core.orchestration.unified_orchestrator import UnifiedOrchestrator

    event_bus = MagicMock()
    memory_store = MagicMock()
    return UnifiedOrchestrator(
        event_bus=event_bus,
        memory_store=memory_store,
        llm_router=None,
    )


# Sample code used as review input (deliberately contains a vulnerability for testing)
SAMPLE_CODE = "def foo():\n    return dangerous_function(input())\n"


# --- Test 1: review_code with mock LLM returning a security issue ---

@pytest.mark.asyncio
async def test_review_code_with_mock_llm(orchestrator):
    """Mock llm_router returns a security issue; review_code surfaces it."""
    issues_json = json.dumps([
        {
            "type": "injection",
            "severity": "critical",
            "description": "dangerous_function(input()) allows arbitrary code execution",
            "line": 2,
            "suggestion": "Validate input before passing to dangerous_function",
        }
    ])

    orchestrator.llm_router.select_provider = AsyncMock(return_value="test-provider")
    orchestrator.llm_router.call_llm = AsyncMock(return_value=(issues_json, {"provider": "test"}))

    result = await orchestrator.review_code(SAMPLE_CODE, language="python")
    assert result.security_issues, "Expected at least one security issue"
    assert result.security_issues[0]["severity"] == "critical"
    assert "dangerous_function" in result.security_issues[0]["description"]


# --- Test 2: review_code with no LLM router returns empty issues gracefully ---

@pytest.mark.asyncio
async def test_review_code_no_llm_router(orchestrator_no_llm):
    """With llm_router=None, review methods return empty lists."""
    result = await orchestrator_no_llm.review_code(SAMPLE_CODE, language="python")
    assert result.security_issues == []
    assert result.performance_issues == []
    assert result.quality_issues == []


# --- Test 3: _review_security parses LLM JSON response correctly ---

@pytest.mark.asyncio
async def test_review_security_parses_llm_response(orchestrator):
    """_review_security correctly parses a multi-issue JSON response."""
    issues_json = json.dumps([
        {"type": "xss", "severity": "high", "description": "XSS via innerHTML", "line": 10, "suggestion": "Escape output"},
        {"type": "sqli", "severity": "critical", "description": "SQL injection", "line": None, "suggestion": "Use parameterized queries"},
    ])

    orchestrator.llm_router.select_provider = AsyncMock(return_value="mock")
    orchestrator.llm_router.call_llm = AsyncMock(return_value=(issues_json, {}))

    issues = await orchestrator._review_security(SAMPLE_CODE, "python")
    assert len(issues) == 2
    assert issues[0]["type"] == "xss"
    assert issues[1]["severity"] == "critical"
    assert issues[1]["line"] is None


# --- Test 4: _review_performance parses LLM JSON response ---

@pytest.mark.asyncio
async def test_review_performance_parses_llm_response(orchestrator):
    """_review_performance correctly parses performance issues."""
    issues_json = json.dumps([
        {"type": "n_plus_one", "severity": "medium", "description": "N+1 query pattern", "line": 5, "suggestion": "Batch queries"},
    ])

    orchestrator.llm_router.select_provider = AsyncMock(return_value="mock")
    orchestrator.llm_router.call_llm = AsyncMock(return_value=(issues_json, {}))

    issues = await orchestrator._review_performance(SAMPLE_CODE, "python")
    assert len(issues) == 1
    assert issues[0]["type"] == "n_plus_one"
    assert issues[0]["severity"] == "medium"


# --- Test 5: _review_quality parses LLM JSON response ---

@pytest.mark.asyncio
async def test_review_quality_parses_llm_response(orchestrator):
    """_review_quality correctly parses quality issues."""
    # Wrap in markdown code fence to test fence-stripping logic
    raw = "```json\n" + json.dumps([
        {"type": "naming", "severity": "low", "description": "Poor variable name 'x'", "suggestion": "Use descriptive names"},
    ]) + "\n```"

    orchestrator.llm_router.select_provider = AsyncMock(return_value="mock")
    orchestrator.llm_router.call_llm = AsyncMock(return_value=(raw, {}))

    issues = await orchestrator._review_quality(SAMPLE_CODE, "python")
    assert len(issues) == 1
    assert issues[0]["type"] == "naming"
    assert issues[0]["severity"] == "low"
    # Line defaults to None when not provided
    assert issues[0]["line"] is None


# --- Test 6: _parse_review_response handles invalid JSON gracefully ---

def test_parse_review_response_invalid_json():
    """Invalid JSON returns empty list instead of crashing."""
    from core.orchestration.unified_orchestrator import UnifiedOrchestrator
    result = UnifiedOrchestrator._parse_review_response("not json at all", "security")
    assert result == []


# --- Test 7: _parse_review_response ignores non-dict items ---

def test_parse_review_response_filters_non_dicts():
    """Non-dict entries in the array are silently skipped."""
    from core.orchestration.unified_orchestrator import UnifiedOrchestrator
    raw = json.dumps([
        {"type": "xss", "severity": "high", "description": "XSS"},
        "this is not a dict",
        42,
    ])
    result = UnifiedOrchestrator._parse_review_response(raw, "security")
    assert len(result) == 1
    assert result[0]["type"] == "xss"
