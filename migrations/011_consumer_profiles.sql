-- Consumer profiles: named identities for AI consumers with bearer token auth.
CREATE TABLE IF NOT EXISTS consumer_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    token_hash TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_consumer_profiles_name ON consumer_profiles(name);
CREATE INDEX IF NOT EXISTS idx_consumer_profiles_token_hash ON consumer_profiles(token_hash);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (11, datetime('now'));
