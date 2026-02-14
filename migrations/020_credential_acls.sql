-- Migration 020: Per-domain access control lists
CREATE TABLE IF NOT EXISTS credential_acls (
    id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domains(id),
    subject TEXT NOT NULL,
    can_read INTEGER NOT NULL DEFAULT 1,
    can_write INTEGER NOT NULL DEFAULT 0,
    can_admin INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    expires_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_credential_acls_domain_subject ON credential_acls(domain_id, subject);
