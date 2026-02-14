-- Keychain security enhancements: brute-force lockout + audit HMAC signatures.

ALTER TABLE keychain_meta ADD COLUMN failed_unseal_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE keychain_meta ADD COLUMN locked_until TEXT;
ALTER TABLE keychain_audit ADD COLUMN signature TEXT;
