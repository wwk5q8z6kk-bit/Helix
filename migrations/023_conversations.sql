-- Phase 3: Conversation store for multi-turn dialogues.
-- Supports conversation-aware assist and context window management.

CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    expired_at TEXT
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,           -- 'user', 'assistant', 'system'
    content TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_conversation_turns_conv_id
    ON conversation_turns(conversation_id, created_at);

CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (23, datetime('now'));
