CREATE TABLE IF NOT EXISTS node_comments (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    author TEXT,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_node_comments_node_id
    ON node_comments(node_id);

CREATE INDEX IF NOT EXISTS idx_node_comments_resolved_at
    ON node_comments(resolved_at);
