-- Contact Identity & Trust Models
CREATE TABLE IF NOT EXISTS contact_identities (
    id TEXT PRIMARY KEY,
    contact_id TEXT NOT NULL,
    identity_type TEXT NOT NULL,
    identity_value TEXT NOT NULL,
    verified INTEGER NOT NULL DEFAULT 0,
    verified_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

CREATE INDEX IF NOT EXISTS idx_contact_identities_contact ON contact_identities(contact_id);

CREATE TABLE IF NOT EXISTS trust_models (
    contact_id TEXT PRIMARY KEY,
    can_query INTEGER NOT NULL DEFAULT 0,
    can_inject_context INTEGER NOT NULL DEFAULT 0,
    can_auto_reply INTEGER NOT NULL DEFAULT 0,
    allowed_namespaces TEXT NOT NULL DEFAULT '[]',
    max_confidence_override REAL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (15, datetime('now'));
