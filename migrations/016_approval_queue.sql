-- Phase G.5 QW4: HITL approval queue for proxy requests

CREATE TABLE IF NOT EXISTS proxy_approvals (
    id            TEXT PRIMARY KEY NOT NULL,
    consumer      TEXT NOT NULL,
    secret_key    TEXT NOT NULL,
    intent        TEXT NOT NULL,
    request_summary TEXT NOT NULL DEFAULT '',
    state         TEXT NOT NULL DEFAULT 'pending',  -- pending, approved, denied, expired
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expires_at    TEXT NOT NULL,
    decided_at    TEXT,
    decided_by    TEXT,
    deny_reason   TEXT,
    scopes        TEXT NOT NULL DEFAULT '[]'  -- JSON array of allowed scopes
);

CREATE INDEX IF NOT EXISTS idx_proxy_approvals_state
    ON proxy_approvals(state);
CREATE INDEX IF NOT EXISTS idx_proxy_approvals_consumer
    ON proxy_approvals(consumer);
CREATE INDEX IF NOT EXISTS idx_proxy_approvals_expires
    ON proxy_approvals(expires_at)
    WHERE state = 'pending';

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (16, datetime('now'));
