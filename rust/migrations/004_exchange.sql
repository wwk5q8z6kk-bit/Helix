CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    node_id TEXT,
    target_node_id TEXT,
    sender TEXT NOT NULL,
    action TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'pending',
    confidence REAL NOT NULL DEFAULT 0.5,
    diff_preview TEXT,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT,
    resolved_at TEXT,
    FOREIGN KEY (target_node_id) REFERENCES knowledge_nodes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_proposals_state ON proposals(state);
CREATE INDEX IF NOT EXISTS idx_proposals_created_at ON proposals(created_at);
CREATE INDEX IF NOT EXISTS idx_proposals_sender ON proposals(sender);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (4, datetime('now'));
