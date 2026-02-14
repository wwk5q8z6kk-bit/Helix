-- Migration 005: Relay Safeguards
-- Blocked senders, auto-approve rules, and undo snapshots for Exchange Inbox

CREATE TABLE IF NOT EXISTS blocked_senders (
    id TEXT PRIMARY KEY,
    sender_type TEXT NOT NULL,      -- 'agent', 'mcp', 'webhook', 'watcher', 'relay'
    sender_pattern TEXT NOT NULL,   -- exact match or glob pattern
    reason TEXT,
    blocked_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    expires_at TEXT                 -- optional expiry
);
CREATE INDEX IF NOT EXISTS idx_blocked_senders_type ON blocked_senders(sender_type);

CREATE TABLE IF NOT EXISTS auto_approve_rules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sender_pattern TEXT,            -- glob pattern or NULL for any
    action_types TEXT,              -- comma-separated list or NULL for any
    min_confidence REAL NOT NULL DEFAULT 0.8,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS proposal_undo_snapshots (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    snapshot_data TEXT NOT NULL,     -- JSON snapshot of state before approval
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    expires_at TEXT NOT NULL,       -- 30 seconds after created_at
    used INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_proposal_undo_proposal ON proposal_undo_snapshots(proposal_id);
CREATE INDEX IF NOT EXISTS idx_proposal_undo_expires ON proposal_undo_snapshots(expires_at);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (5, datetime('now'));
