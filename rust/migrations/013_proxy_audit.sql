-- Proxy audit log: tracks all proxy operations for forensics and compliance.
CREATE TABLE IF NOT EXISTS proxy_audit_log (
    id TEXT PRIMARY KEY,
    consumer TEXT NOT NULL,
    secret_ref TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    intent TEXT NOT NULL DEFAULT '',
    timestamp TEXT NOT NULL,
    success INTEGER,
    sanitized INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    request_summary TEXT NOT NULL DEFAULT '',
    response_status INTEGER
);

CREATE INDEX IF NOT EXISTS idx_proxy_audit_consumer ON proxy_audit_log(consumer);
CREATE INDEX IF NOT EXISTS idx_proxy_audit_timestamp ON proxy_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_proxy_audit_secret_ref ON proxy_audit_log(secret_ref);

INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (13, datetime('now'));
