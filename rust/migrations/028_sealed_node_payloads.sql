ALTER TABLE knowledge_nodes ADD COLUMN payload_ciphertext TEXT;
ALTER TABLE knowledge_nodes ADD COLUMN payload_wrapped_dek TEXT;

CREATE INDEX IF NOT EXISTS idx_nodes_payload_ciphertext
    ON knowledge_nodes(payload_ciphertext);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (28, datetime('now'));
