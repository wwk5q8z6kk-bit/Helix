CREATE TABLE IF NOT EXISTS owner_profile (
    id TEXT PRIMARY KEY DEFAULT 'owner',
    display_name TEXT NOT NULL DEFAULT 'MindVault Owner',
    avatar_url TEXT,
    bio TEXT,
    email TEXT,
    preferred_namespace TEXT NOT NULL DEFAULT 'default',
    default_node_kind TEXT NOT NULL DEFAULT 'fact',
    preferred_llm_provider TEXT,
    timezone TEXT NOT NULL DEFAULT 'UTC',
    signature_name TEXT,
    signature_public_key TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
INSERT OR IGNORE INTO owner_profile (id) VALUES ('owner');

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (10, datetime('now'));
