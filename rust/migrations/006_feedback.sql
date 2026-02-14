CREATE TABLE IF NOT EXISTS agent_feedback (
    id TEXT PRIMARY KEY,
    intent_id TEXT,
    intent_type TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence_at_time REAL,
    user_edit_delta REAL,
    response_time_ms INTEGER,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_agent_feedback_type ON agent_feedback(intent_type);
CREATE INDEX IF NOT EXISTS idx_agent_feedback_action ON agent_feedback(action);
CREATE INDEX IF NOT EXISTS idx_agent_feedback_created ON agent_feedback(created_at);

CREATE TABLE IF NOT EXISTS agent_confidence_overrides (
    intent_type TEXT PRIMARY KEY,
    base_adjustment REAL NOT NULL DEFAULT 0.0,
    auto_apply_threshold REAL NOT NULL DEFAULT 0.95,
    suppress_below REAL NOT NULL DEFAULT 0.1,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (6, datetime('now'));
