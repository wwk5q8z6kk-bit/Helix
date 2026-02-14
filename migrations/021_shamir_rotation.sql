-- Migration 021: Shamir share rotation tracking
ALTER TABLE keychain_meta ADD COLUMN shamir_last_rotated_at TEXT;
