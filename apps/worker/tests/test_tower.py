"""Tests for TowerLogHandler (core/observability/tower_log.py)."""

import json
import os
import tempfile

import pytest


@pytest.fixture
def tower():
    """Create a TowerLogHandler writing to a temp file."""
    from core.observability.tower_log import TowerLogHandler

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tower_events.jsonl")
        yield TowerLogHandler(path=path, buffer_size=100)


def test_log_event_returns_entry(tower):
    entry = tower.log_event("task_started", "test-agent")

    assert entry["event_type"] == "task_started"
    assert entry["source"] == "test-agent"
    assert "timestamp" in entry


def test_log_event_with_duration(tower):
    entry = tower.log_event("llm_call", "router", duration_ms=123.4)

    assert entry["duration_ms"] == 123.4


def test_log_event_with_metadata(tower):
    entry = tower.log_event("task_completed", "orchestrator", task_id="t1", quality=0.95)

    assert entry["metadata"]["task_id"] == "t1"
    assert entry["metadata"]["quality"] == 0.95


def test_log_event_no_metadata_key_when_empty(tower):
    entry = tower.log_event("ping", "test")

    assert "metadata" not in entry


def test_get_recent_returns_logged_events(tower):
    tower.log_event("a", "s1")
    tower.log_event("b", "s2")
    tower.log_event("c", "s3")

    recent = tower.get_recent(limit=2)
    assert len(recent) == 2
    assert recent[0]["event_type"] == "b"
    assert recent[1]["event_type"] == "c"


def test_get_recent_limit_exceeds_buffer(tower):
    tower.log_event("a", "s1")
    recent = tower.get_recent(limit=100)
    assert len(recent) == 1


def test_get_recent_empty_buffer(tower):
    assert tower.get_recent() == []


def test_ring_buffer_eviction():
    from core.observability.tower_log import TowerLogHandler

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tower.jsonl")
        handler = TowerLogHandler(path=path, buffer_size=3)

        handler.log_event("a", "s")
        handler.log_event("b", "s")
        handler.log_event("c", "s")
        handler.log_event("d", "s")  # evicts "a"

        recent = handler.get_recent(limit=10)
        assert len(recent) == 3
        types = [e["event_type"] for e in recent]
        assert types == ["b", "c", "d"]


def test_jsonl_persistence(tower):
    tower.log_event("persist_test", "writer", task_id="p1")

    with open(tower._path) as f:
        lines = f.readlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["event_type"] == "persist_test"


def test_get_summary_counts_by_type(tower):
    tower.log_event("task_started", "s1")
    tower.log_event("task_started", "s1")
    tower.log_event("task_completed", "s1")

    summary = tower.get_summary(hours=1.0)
    assert summary["task_started"] == 2
    assert summary["task_completed"] == 1


def test_get_summary_empty(tower):
    assert tower.get_summary() == {}
