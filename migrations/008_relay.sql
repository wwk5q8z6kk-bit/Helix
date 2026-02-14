-- Phase 3.2: Communication Relay
CREATE TABLE IF NOT EXISTS relay_contacts (
    id TEXT PRIMARY KEY NOT NULL,
    display_name TEXT NOT NULL,
    public_key TEXT NOT NULL,
    vault_address TEXT,
    trust_level TEXT NOT NULL DEFAULT 'relay_only',
    autonomy_rule_id TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_relay_contacts_name ON relay_contacts(display_name);

CREATE TABLE IF NOT EXISTS relay_channels (
    id TEXT PRIMARY KEY NOT NULL,
    name TEXT,
    channel_type TEXT NOT NULL DEFAULT 'direct',
    member_contact_ids TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_relay_channels_type ON relay_channels(channel_type);

CREATE TABLE IF NOT EXISTS relay_messages (
    id TEXT PRIMARY KEY NOT NULL,
    channel_id TEXT NOT NULL REFERENCES relay_channels(id),
    thread_id TEXT,
    sender_contact_id TEXT,
    recipient_contact_id TEXT,
    direction TEXT NOT NULL DEFAULT 'outbound',
    content TEXT NOT NULL,
    content_type TEXT NOT NULL DEFAULT 'text',
    status TEXT NOT NULL DEFAULT 'pending',
    vault_node_id TEXT,
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_relay_messages_channel ON relay_messages(channel_id);
CREATE INDEX IF NOT EXISTS idx_relay_messages_thread ON relay_messages(thread_id);
CREATE INDEX IF NOT EXISTS idx_relay_messages_status ON relay_messages(status);
CREATE INDEX IF NOT EXISTS idx_relay_messages_created ON relay_messages(created_at);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (8, datetime('now'));
