-- Migration 019: Metadata encryption flag for credentials
ALTER TABLE credentials ADD COLUMN metadata_encrypted INTEGER NOT NULL DEFAULT 0;
