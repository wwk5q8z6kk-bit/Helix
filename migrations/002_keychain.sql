-- Sovereign Keychain schema (applied to keychain.sqlite)

CREATE TABLE IF NOT EXISTS keychain_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version INTEGER NOT NULL DEFAULT 1,
    master_salt TEXT NOT NULL,
    verification_blob TEXT NOT NULL,
    key_epoch INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    last_rotated_at TEXT,
    macos_keychain_service TEXT
);

CREATE TABLE IF NOT EXISTS key_epochs (
    epoch INTEGER PRIMARY KEY,
    wrapped_key TEXT,
    created_at TEXT NOT NULL,
    grace_expires_at TEXT,
    retired_at TEXT
);

CREATE TABLE IF NOT EXISTS domains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    derivation_info TEXT NOT NULL,
    epoch INTEGER NOT NULL REFERENCES key_epochs(epoch),
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    credential_count INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_domains_name ON domains(name);
CREATE INDEX IF NOT EXISTS idx_domains_epoch ON domains(epoch);

CREATE TABLE IF NOT EXISTS credentials (
    id TEXT PRIMARY KEY,
    domain_id TEXT NOT NULL REFERENCES domains(id),
    name TEXT NOT NULL,
    description TEXT,
    kind TEXT NOT NULL,
    encrypted_value TEXT NOT NULL,
    derivation_info TEXT NOT NULL,
    epoch INTEGER NOT NULL REFERENCES key_epochs(epoch),
    state TEXT NOT NULL DEFAULT 'active',
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed_at TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    expires_at TEXT,
    archived_at TEXT,
    destroyed_at TEXT,
    delegation_id TEXT,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_credentials_domain_id ON credentials(domain_id);
CREATE INDEX IF NOT EXISTS idx_credentials_state ON credentials(state);
CREATE INDEX IF NOT EXISTS idx_credentials_kind ON credentials(kind);
CREATE INDEX IF NOT EXISTS idx_credentials_epoch ON credentials(epoch);
CREATE INDEX IF NOT EXISTS idx_credentials_expires_at ON credentials(expires_at);

CREATE TABLE IF NOT EXISTS credential_tags (
    credential_id TEXT NOT NULL REFERENCES credentials(id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    PRIMARY KEY (credential_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_credential_tags_tag ON credential_tags(tag);

CREATE TABLE IF NOT EXISTS delegations (
    id TEXT PRIMARY KEY,
    credential_id TEXT NOT NULL REFERENCES credentials(id),
    delegatee TEXT NOT NULL,
    parent_id TEXT,
    can_read INTEGER NOT NULL DEFAULT 1,
    can_use INTEGER NOT NULL DEFAULT 0,
    can_delegate INTEGER NOT NULL DEFAULT 0,
    chain_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT,
    revoked_at TEXT,
    max_depth INTEGER NOT NULL DEFAULT 1,
    depth INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_delegations_credential_id ON delegations(credential_id);
CREATE INDEX IF NOT EXISTS idx_delegations_delegatee ON delegations(delegatee);
CREATE INDEX IF NOT EXISTS idx_delegations_parent_id ON delegations(parent_id);

CREATE TABLE IF NOT EXISTS keychain_audit (
    id TEXT NOT NULL,
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    action TEXT NOT NULL,
    subject TEXT NOT NULL,
    resource_id TEXT,
    details_json TEXT,
    entry_hash TEXT NOT NULL,
    previous_hash TEXT,
    timestamp TEXT NOT NULL,
    source_ip TEXT
);

CREATE INDEX IF NOT EXISTS idx_keychain_audit_action ON keychain_audit(action);
CREATE INDEX IF NOT EXISTS idx_keychain_audit_timestamp ON keychain_audit(timestamp);
CREATE INDEX IF NOT EXISTS idx_keychain_audit_subject ON keychain_audit(subject);

CREATE TABLE IF NOT EXISTS access_patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    credential_id TEXT NOT NULL,
    accessor TEXT NOT NULL,
    source_ip TEXT,
    timestamp TEXT NOT NULL,
    hour_of_day INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_access_patterns_credential_id ON access_patterns(credential_id);
CREATE INDEX IF NOT EXISTS idx_access_patterns_timestamp ON access_patterns(timestamp);

CREATE TABLE IF NOT EXISTS breach_alerts (
    id TEXT PRIMARY KEY,
    credential_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    description TEXT NOT NULL,
    details_json TEXT,
    timestamp TEXT NOT NULL,
    acknowledged_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_breach_alerts_credential_id ON breach_alerts(credential_id);
CREATE INDEX IF NOT EXISTS idx_breach_alerts_severity ON breach_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_breach_alerts_timestamp ON breach_alerts(timestamp);
