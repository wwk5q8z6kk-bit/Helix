-- MindVault initial schema

CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id TEXT PRIMARY KEY NOT NULL,
    kind TEXT NOT NULL,
    title TEXT,
    content TEXT NOT NULL,
    source TEXT,
    namespace TEXT NOT NULL DEFAULT 'default',
    importance REAL NOT NULL DEFAULT 0.5,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    version INTEGER NOT NULL DEFAULT 1,
    expires_at TEXT,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS node_tags (
    node_id TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (node_id, tag)
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY NOT NULL,
    from_node TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    to_node TEXT NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS changelog (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    operation TEXT NOT NULL,  -- 'create', 'update', 'delete'
    diff_json TEXT,
    timestamp TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS permission_templates (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    tier TEXT NOT NULL,
    scope_namespace TEXT,
    scope_tags_json TEXT,
    allow_kinds_json TEXT,
    allow_actions_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS access_keys (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT,
    template_id TEXT NOT NULL REFERENCES permission_templates(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    expires_at TEXT,
    revoked_at TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_permission_templates_name ON permission_templates(name);
CREATE INDEX IF NOT EXISTS idx_permission_templates_tier ON permission_templates(tier);
CREATE INDEX IF NOT EXISTS idx_access_keys_template_id ON access_keys(template_id);
CREATE INDEX IF NOT EXISTS idx_access_keys_hash ON access_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_access_keys_revoked ON access_keys(revoked_at);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON knowledge_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_namespace ON knowledge_nodes(namespace);
CREATE INDEX IF NOT EXISTS idx_nodes_kind_namespace ON knowledge_nodes(kind, namespace);
CREATE INDEX IF NOT EXISTS idx_nodes_importance ON knowledge_nodes(importance);
CREATE INDEX IF NOT EXISTS idx_nodes_created_at ON knowledge_nodes(created_at);
CREATE INDEX IF NOT EXISTS idx_nodes_updated_at ON knowledge_nodes(updated_at);
CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON knowledge_nodes(last_accessed_at);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON node_tags(tag);
CREATE INDEX IF NOT EXISTS idx_rel_from ON relationships(from_node);
CREATE INDEX IF NOT EXISTS idx_rel_to ON relationships(to_node);
CREATE INDEX IF NOT EXISTS idx_rel_kind ON relationships(kind);
CREATE INDEX IF NOT EXISTS idx_changelog_node ON changelog(node_id);
CREATE INDEX IF NOT EXISTS idx_changelog_ts ON changelog(timestamp);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, datetime('now'));
INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (2, datetime('now'));
