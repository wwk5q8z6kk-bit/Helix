CREATE TABLE IF NOT EXISTS autonomy_rules (
    id TEXT PRIMARY KEY,
    rule_type TEXT NOT NULL,         -- 'global', 'domain', 'contact', 'tag'
    scope_key TEXT,                  -- NULL for global, domain name, contact id, or tag
    auto_apply_threshold REAL NOT NULL DEFAULT 0.95,
    max_actions_per_hour INTEGER NOT NULL DEFAULT 10,
    allowed_intent_types TEXT,       -- comma-separated or NULL for all
    blocked_intent_types TEXT,       -- comma-separated or NULL for none
    quiet_hours_start TEXT,          -- HH:MM format or NULL
    quiet_hours_end TEXT,
    quiet_hours_timezone TEXT DEFAULT 'UTC',
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_autonomy_rules_type ON autonomy_rules(rule_type);
CREATE INDEX IF NOT EXISTS idx_autonomy_rules_scope ON autonomy_rules(scope_key);

CREATE TABLE IF NOT EXISTS autonomy_action_log (
    id TEXT PRIMARY KEY,
    rule_id TEXT,
    intent_type TEXT NOT NULL,
    decision TEXT NOT NULL,          -- 'auto_apply', 'defer', 'block', 'queue_for_later'
    confidence REAL,
    reason TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
CREATE INDEX IF NOT EXISTS idx_autonomy_action_log_created ON autonomy_action_log(created_at);
CREATE INDEX IF NOT EXISTS idx_autonomy_action_log_decision ON autonomy_action_log(decision);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (7, datetime('now'));
