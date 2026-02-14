-- Public share links for read-only node access
CREATE TABLE IF NOT EXISTS public_shares (
    id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked_at TEXT,
    FOREIGN KEY (node_id) REFERENCES knowledge_nodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_public_shares_node_id
    ON public_shares (node_id);
CREATE INDEX IF NOT EXISTS idx_public_shares_token_hash
    ON public_shares (token_hash);
