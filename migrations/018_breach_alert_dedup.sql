-- Breach alert deduplication index (Enhancement 5)
CREATE INDEX IF NOT EXISTS idx_breach_alerts_dedup
    ON breach_alerts(credential_id, alert_type, timestamp);
