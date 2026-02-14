-- Knowledge Conflict Detection
CREATE TABLE IF NOT EXISTS conflicts (
    id TEXT PRIMARY KEY,
    node_a TEXT NOT NULL,
    node_b TEXT NOT NULL,
    conflict_type TEXT NOT NULL DEFAULT 'contradiction',
    score REAL NOT NULL DEFAULT 0.0,
    explanation TEXT NOT NULL DEFAULT '',
    resolved INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_conflicts_resolved ON conflicts(resolved);
CREATE INDEX IF NOT EXISTS idx_conflicts_node_a ON conflicts(node_a);
CREATE INDEX IF NOT EXISTS idx_conflicts_node_b ON conflicts(node_b);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (14, datetime('now'));
