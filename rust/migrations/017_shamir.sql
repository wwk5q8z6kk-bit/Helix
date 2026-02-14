-- Shamir VEK splitting: store threshold and total share count in vault metadata.

ALTER TABLE keychain_meta ADD COLUMN shamir_threshold INTEGER;
ALTER TABLE keychain_meta ADD COLUMN shamir_total INTEGER;
