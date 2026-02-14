-- Access policies: ABAC with default-deny, scopes, TTL, and HITL flags.
CREATE TABLE IF NOT EXISTS access_policies (
    id TEXT PRIMARY KEY,
    secret_key TEXT NOT NULL,
    consumer TEXT NOT NULL,
    allowed INTEGER NOT NULL DEFAULT 0,
    scopes_json TEXT NOT NULL DEFAULT '[]',
    max_ttl_seconds INTEGER,
    expires_at TEXT,
    require_approval INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(secret_key, consumer)
);

CREATE INDEX IF NOT EXISTS idx_access_policies_consumer ON access_policies(consumer);
CREATE INDEX IF NOT EXISTS idx_access_policies_secret_key ON access_policies(secret_key);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (12, datetime('now'));
