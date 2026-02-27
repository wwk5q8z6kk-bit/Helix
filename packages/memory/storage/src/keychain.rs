//! SQLite-backed implementation of the KeychainStore trait.

use std::sync::Mutex;

use async_trait::async_trait;
use base64::Engine;
use chrono::{DateTime, Utc};
use rand::RngCore;
use rusqlite::{params, Connection, OptionalExtension};
use uuid::Uuid;

use hx_core::error::{HxError, MvResult};
use hx_core::model::keychain::*;
use hx_core::traits::KeychainStore;

/// SQLite-backed keychain store. Uses a separate `keychain.sqlite` database.
pub struct SqliteKeychainStore {
    conn: Mutex<Connection>,
}

impl SqliteKeychainStore {
    pub fn open(path: &std::path::Path) -> MvResult<Self> {
        let conn = Connection::open(path)
            .map_err(|e| HxError::Storage(format!("open keychain db: {e}")))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000; PRAGMA secure_delete=ON;",
        )
        .map_err(|e| HxError::Storage(format!("keychain pragma: {e}")))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.run_migrations()?;
        Ok(store)
    }

    pub fn open_in_memory() -> MvResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| HxError::Storage(format!("open keychain in-memory: {e}")))?;
        conn.execute_batch("PRAGMA foreign_keys=ON;")
            .map_err(|e| HxError::Storage(format!("keychain pragma: {e}")))?;
        let store = Self {
            conn: Mutex::new(conn),
        };
        store.run_migrations()?;
        Ok(store)
    }

    fn run_migrations(&self) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let migration_sql = include_str!("../../../migrations/002_keychain.sql");
        conn.execute_batch(migration_sql)
            .map_err(|e| HxError::Migration(format!("keychain migration failed: {e}")))?;

        // Security enhancements migration (idempotent — ignore "duplicate column" errors)
        let security_sql = include_str!("../../../migrations/009_keychain_security.sql");
        if let Err(e) = conn.execute_batch(security_sql) {
            let msg = e.to_string();
            if !msg.contains("duplicate column") {
                return Err(HxError::Migration(format!(
                    "keychain security migration failed: {e}"
                )));
            }
        }

        // Shamir VEK splitting migration (idempotent)
        let shamir_sql = include_str!("../../../migrations/017_shamir.sql");
        if let Err(e) = conn.execute_batch(shamir_sql) {
            let msg = e.to_string();
            if !msg.contains("duplicate column") {
                return Err(HxError::Migration(format!(
                    "shamir migration failed: {e}"
                )));
            }
        }

        // Breach alert deduplication index (idempotent — CREATE INDEX IF NOT EXISTS)
        let dedup_sql = include_str!("../../../migrations/018_breach_alert_dedup.sql");
        conn.execute_batch(dedup_sql)
            .map_err(|e| HxError::Migration(format!("breach alert dedup migration failed: {e}")))?;

        // Metadata encryption column (idempotent — column may already exist)
        let meta_enc_sql = include_str!("../../../migrations/019_metadata_encryption.sql");
        if let Err(e) = conn.execute_batch(meta_enc_sql) {
            let msg = e.to_string();
            if !msg.contains("duplicate column") {
                return Err(HxError::Migration(format!("metadata encryption migration failed: {e}")));
            }
        }

        // Domain ACLs (idempotent — CREATE TABLE/INDEX IF NOT EXISTS)
        let acl_sql = include_str!("../../../migrations/020_credential_acls.sql");
        conn.execute_batch(acl_sql)
            .map_err(|e| HxError::Migration(format!("credential ACLs migration failed: {e}")))?;

        // Shamir rotation tracking (idempotent — column may already exist)
        let shamir_rot_sql = include_str!("../../../migrations/021_shamir_rotation.sql");
        if let Err(e) = conn.execute_batch(shamir_rot_sql) {
            let msg = e.to_string();
            if !msg.contains("duplicate column") {
                return Err(HxError::Migration(format!("shamir rotation migration failed: {e}")));
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn row_to_domain(row: &rusqlite::Row<'_>) -> rusqlite::Result<DomainKey> {
    Ok(DomainKey {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        name: row.get(1)?,
        description: row.get(2)?,
        derivation_info: row.get(3)?,
        epoch: row.get::<_, i64>(4)? as u64,
        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(5)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_default(),
        revoked_at: row
            .get::<_, Option<String>>(6)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        credential_count: row.get::<_, i64>(7)? as u64,
    })
}

fn row_to_credential(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredCredential> {
    Ok(StoredCredential {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        domain_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        name: row.get(2)?,
        description: row.get(3)?,
        kind: row.get(4)?,
        encrypted_value: row.get(5)?,
        derivation_info: row.get(6)?,
        epoch: row.get::<_, i64>(7)? as u64,
        state: row
            .get::<_, String>(8)?
            .parse()
            .unwrap_or(CredentialState::Active),
        metadata: row
            .get::<_, Option<String>>(9)?
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default(),
        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(10)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_default(),
        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(11)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_default(),
        last_accessed_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(12)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_default(),
        access_count: row.get::<_, i64>(13)? as u64,
        expires_at: row
            .get::<_, Option<String>>(14)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        archived_at: row
            .get::<_, Option<String>>(15)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        destroyed_at: row
            .get::<_, Option<String>>(16)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        delegation_id: row
            .get::<_, Option<String>>(17)?
            .and_then(|s| Uuid::parse_str(&s).ok()),
        version: row.get::<_, i32>(18)? as u32,
        metadata_encrypted: row.get::<_, bool>(19).unwrap_or(false),
        tags: Vec::new(), // loaded separately
    })
}

fn row_to_delegation(row: &rusqlite::Row<'_>) -> rusqlite::Result<Delegation> {
    Ok(Delegation {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        credential_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        delegatee: row.get(2)?,
        parent_id: row
            .get::<_, Option<String>>(3)?
            .and_then(|s| Uuid::parse_str(&s).ok()),
        permissions: DelegationPermissions {
            can_read: row.get::<_, bool>(4)?,
            can_use: row.get::<_, bool>(5)?,
            can_delegate: row.get::<_, bool>(6)?,
        },
        chain_hash: row.get(7)?,
        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_default(),
        expires_at: row
            .get::<_, Option<String>>(9)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        revoked_at: row
            .get::<_, Option<String>>(10)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
        max_depth: row.get::<_, i32>(11)? as u32,
        depth: row.get::<_, i32>(12)? as u32,
    })
}

fn row_to_acl(row: &rusqlite::Row<'_>) -> rusqlite::Result<DomainAcl> {
    Ok(DomainAcl {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        domain_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
        subject: row.get(2)?,
        can_read: row.get(3)?,
        can_write: row.get(4)?,
        can_admin: row.get(5)?,
        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_default(),
        expires_at: row
            .get::<_, Option<String>>(7)?
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
            .map(|dt| dt.with_timezone(&Utc)),
    })
}

fn row_to_audit_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<KeychainAuditEntry> {
    Ok(KeychainAuditEntry {
        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
        sequence: row.get(1)?,
        action: row
            .get::<_, String>(2)?
            .parse()
            .unwrap_or(KeychainAuditAction::VaultInitialized),
        subject: row.get(3)?,
        resource_id: row.get(4)?,
        details: row
            .get::<_, Option<String>>(5)?
            .and_then(|s| serde_json::from_str(&s).ok()),
        entry_hash: row.get(6)?,
        previous_hash: row.get(7)?,
        timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
            .map(|dt| dt.with_timezone(&Utc))
            .unwrap_or_default(),
        source_ip: row.get(9)?,
        signature: row.get(10)?,
    })
}

fn load_credential_tags(conn: &Connection, credential_id: &str) -> Vec<String> {
    let mut stmt = match conn
        .prepare("SELECT tag FROM credential_tags WHERE credential_id = ?1 ORDER BY tag")
    {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    stmt.query_map(params![credential_id], |row| row.get(0))
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
}

fn save_credential_tags_inner(
    conn: &Connection,
    credential_id: &str,
    tags: &[String],
) -> rusqlite::Result<()> {
    conn.execute(
        "DELETE FROM credential_tags WHERE credential_id = ?1",
        params![credential_id],
    )?;
    let mut stmt =
        conn.prepare("INSERT INTO credential_tags (credential_id, tag) VALUES (?1, ?2)")?;
    for tag in tags {
        stmt.execute(params![credential_id, tag])?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// KeychainStore implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl KeychainStore for SqliteKeychainStore {
    // -- Vault lifecycle --

    async fn get_vault_meta(&self) -> MvResult<Option<VaultMeta>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT schema_version, master_salt, verification_blob, key_epoch, \
                 created_at, last_rotated_at, macos_keychain_service, \
                 shamir_threshold, shamir_total, shamir_last_rotated_at \
                 FROM keychain_meta WHERE id = 1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let meta = stmt
            .query_row([], |row| {
                Ok(VaultMeta {
                    schema_version: row.get::<_, i32>(0)? as u32,
                    master_salt: row.get(1)?,
                    verification_blob: row.get(2)?,
                    key_epoch: row.get::<_, i64>(3)? as u64,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(4)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_default(),
                    last_rotated_at: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                    macos_keychain_service: row.get(6)?,
                    shamir_threshold: row.get::<_, Option<i32>>(7)?.map(|v| v as u8),
                    shamir_total: row.get::<_, Option<i32>>(8)?.map(|v| v as u8),
                    shamir_last_rotated_at: row
                        .get::<_, Option<String>>(9)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                })
            })
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(meta)
    }

    async fn save_vault_meta(&self, meta: &VaultMeta) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO keychain_meta (id, schema_version, master_salt, verification_blob, \
             key_epoch, created_at, last_rotated_at, macos_keychain_service, \
             shamir_threshold, shamir_total, shamir_last_rotated_at) \
             VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10) \
             ON CONFLICT(id) DO UPDATE SET \
             schema_version=?1, master_salt=?2, verification_blob=?3, \
             key_epoch=?4, last_rotated_at=?6, macos_keychain_service=?7, \
             shamir_threshold=?8, shamir_total=?9, shamir_last_rotated_at=?10",
            params![
                meta.schema_version as i32,
                meta.master_salt,
                meta.verification_blob,
                meta.key_epoch as i64,
                meta.created_at.to_rfc3339(),
                meta.last_rotated_at.map(|dt| dt.to_rfc3339()),
                meta.macos_keychain_service,
                meta.shamir_threshold.map(|v| v as i32),
                meta.shamir_total.map(|v| v as i32),
                meta.shamir_last_rotated_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    // -- Key epochs --

    async fn insert_key_epoch(&self, epoch: &KeyEpoch) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO key_epochs (epoch, wrapped_key, created_at, grace_expires_at, retired_at) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                epoch.epoch as i64,
                epoch.wrapped_key,
                epoch.created_at.to_rfc3339(),
                epoch.grace_expires_at.map(|dt| dt.to_rfc3339()),
                epoch.retired_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get_key_epoch(&self, epoch: u64) -> MvResult<Option<KeyEpoch>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT epoch, wrapped_key, created_at, grace_expires_at, retired_at \
                 FROM key_epochs WHERE epoch = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        stmt.query_row(params![epoch as i64], |row| {
            Ok(KeyEpoch {
                epoch: row.get::<_, i64>(0)? as u64,
                wrapped_key: row.get(1)?,
                created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_default(),
                grace_expires_at: row
                    .get::<_, Option<String>>(3)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
                retired_at: row
                    .get::<_, Option<String>>(4)?
                    .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                    .map(|dt| dt.with_timezone(&Utc)),
            })
        })
        .optional()
        .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn list_key_epochs(&self) -> MvResult<Vec<KeyEpoch>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT epoch, wrapped_key, created_at, grace_expires_at, retired_at \
                 FROM key_epochs ORDER BY epoch",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| {
                Ok(KeyEpoch {
                    epoch: row.get::<_, i64>(0)? as u64,
                    wrapped_key: row.get(1)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(2)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_default(),
                    grace_expires_at: row
                        .get::<_, Option<String>>(3)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                    retired_at: row
                        .get::<_, Option<String>>(4)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                })
            })
            .map_err(|e| HxError::Storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn retire_key_epoch(&self, epoch: u64) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE key_epochs SET retired_at = ?1 WHERE epoch = ?2",
            params![Utc::now().to_rfc3339(), epoch as i64],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    // -- Domains --

    async fn insert_domain(&self, domain: &DomainKey) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO domains (id, name, description, derivation_info, epoch, created_at, revoked_at, credential_count) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                domain.id.to_string(),
                domain.name,
                domain.description,
                domain.derivation_info,
                domain.epoch as i64,
                domain.created_at.to_rfc3339(),
                domain.revoked_at.map(|dt| dt.to_rfc3339()),
                domain.credential_count as i64,
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get_domain(&self, id: Uuid) -> MvResult<Option<DomainKey>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, derivation_info, epoch, created_at, revoked_at, credential_count \
                 FROM domains WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        stmt.query_row(params![id.to_string()], row_to_domain)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn get_domain_by_name(&self, name: &str) -> MvResult<Option<DomainKey>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, derivation_info, epoch, created_at, revoked_at, credential_count \
                 FROM domains WHERE name = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        stmt.query_row(params![name], row_to_domain)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn list_domains(&self) -> MvResult<Vec<DomainKey>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, derivation_info, epoch, created_at, revoked_at, credential_count \
                 FROM domains ORDER BY created_at",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], row_to_domain)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn revoke_domain(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE domains SET revoked_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    // -- Credentials --

    async fn insert_credential(&self, cred: &StoredCredential) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let metadata_json = serde_json::to_string(&cred.metadata).ok();
        conn.execute(
            "INSERT INTO credentials (id, domain_id, name, description, kind, encrypted_value, \
             derivation_info, epoch, state, metadata_json, created_at, updated_at, \
             last_accessed_at, access_count, expires_at, archived_at, destroyed_at, \
             delegation_id, version, metadata_encrypted) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20)",
            params![
                cred.id.to_string(),
                cred.domain_id.to_string(),
                cred.name,
                cred.description,
                cred.kind,
                cred.encrypted_value,
                cred.derivation_info,
                cred.epoch as i64,
                cred.state.as_str(),
                metadata_json,
                cred.created_at.to_rfc3339(),
                cred.updated_at.to_rfc3339(),
                cred.last_accessed_at.to_rfc3339(),
                cred.access_count as i64,
                cred.expires_at.map(|dt| dt.to_rfc3339()),
                cred.archived_at.map(|dt| dt.to_rfc3339()),
                cred.destroyed_at.map(|dt| dt.to_rfc3339()),
                cred.delegation_id.map(|id| id.to_string()),
                cred.version as i32,
                cred.metadata_encrypted,
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;

        save_credential_tags_inner(&conn, &cred.id.to_string(), &cred.tags)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        // Increment domain credential count
        conn.execute(
            "UPDATE domains SET credential_count = credential_count + 1 WHERE id = ?1",
            params![cred.domain_id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn get_credential(&self, id: Uuid) -> MvResult<Option<StoredCredential>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, domain_id, name, description, kind, encrypted_value, \
                 derivation_info, epoch, state, metadata_json, created_at, updated_at, \
                 last_accessed_at, access_count, expires_at, archived_at, destroyed_at, \
                 delegation_id, version, metadata_encrypted FROM credentials WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut cred = match stmt
            .query_row(params![id.to_string()], row_to_credential)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?
        {
            Some(c) => c,
            None => return Ok(None),
        };

        cred.tags = load_credential_tags(&conn, &id.to_string());
        Ok(Some(cred))
    }

    async fn update_credential(&self, cred: &StoredCredential) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let metadata_json = serde_json::to_string(&cred.metadata).ok();
        conn.execute(
            "UPDATE credentials SET domain_id=?2, name=?3, description=?4, kind=?5, \
             encrypted_value=?6, derivation_info=?7, epoch=?8, state=?9, metadata_json=?10, \
             updated_at=?11, last_accessed_at=?12, access_count=?13, expires_at=?14, \
             archived_at=?15, destroyed_at=?16, delegation_id=?17, version=?18, \
             metadata_encrypted=?19 WHERE id = ?1",
            params![
                cred.id.to_string(),
                cred.domain_id.to_string(),
                cred.name,
                cred.description,
                cred.kind,
                cred.encrypted_value,
                cred.derivation_info,
                cred.epoch as i64,
                cred.state.as_str(),
                metadata_json,
                cred.updated_at.to_rfc3339(),
                cred.last_accessed_at.to_rfc3339(),
                cred.access_count as i64,
                cred.expires_at.map(|dt| dt.to_rfc3339()),
                cred.archived_at.map(|dt| dt.to_rfc3339()),
                cred.destroyed_at.map(|dt| dt.to_rfc3339()),
                cred.delegation_id.map(|id| id.to_string()),
                cred.version as i32,
                cred.metadata_encrypted,
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;

        save_credential_tags_inner(&conn, &cred.id.to_string(), &cred.tags)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(())
    }

    async fn list_credentials(
        &self,
        domain_id: Option<Uuid>,
        state: Option<CredentialState>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<StoredCredential>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut sql = String::from(
            "SELECT id, domain_id, name, description, kind, encrypted_value, \
             derivation_info, epoch, state, metadata_json, created_at, updated_at, \
             last_accessed_at, access_count, expires_at, archived_at, destroyed_at, \
             delegation_id, version, metadata_encrypted FROM credentials WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(did) = domain_id {
            sql.push_str(" AND domain_id = ?");
            param_values.push(Box::new(did.to_string()));
        }
        if let Some(st) = state {
            sql.push_str(" AND state = ?");
            param_values.push(Box::new(st.as_str().to_string()));
        }
        sql.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));

        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params_ref.as_slice(), row_to_credential)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut creds: Vec<StoredCredential> = rows
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        // Load tags for each credential
        for cred in &mut creds {
            cred.tags = load_credential_tags(&conn, &cred.id.to_string());
        }

        Ok(creds)
    }

    async fn count_credentials(&self, domain_id: Option<Uuid>) -> MvResult<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let (sql, param): (String, Option<String>) = match domain_id {
            Some(did) => (
                "SELECT COUNT(*) FROM credentials WHERE domain_id = ?1".into(),
                Some(did.to_string()),
            ),
            None => ("SELECT COUNT(*) FROM credentials".into(), None),
        };

        let count: i64 = if let Some(p) = &param {
            conn.query_row(&sql, params![p], |row| row.get(0))
        } else {
            conn.query_row(&sql, [], |row| row.get(0))
        }
        .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(count as usize)
    }

    async fn shred_credential(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let id_str = id.to_string();

        // 3-pass random overwrite of all sensitive fields
        for _pass in 0..3 {
            let mut random_value = [0u8; 64];
            rand::rngs::OsRng.fill_bytes(&mut random_value);
            let overwrite = base64::engine::general_purpose::STANDARD.encode(random_value);
            let mut random_name = [0u8; 16];
            rand::rngs::OsRng.fill_bytes(&mut random_name);
            let name_overwrite = base64::engine::general_purpose::STANDARD.encode(random_name);

            conn.execute(
                "UPDATE credentials SET encrypted_value = ?1, name = ?2, description = ?2, \
                 derivation_info = ?2, metadata_json = ?2, state = 'destroyed', destroyed_at = ?3 \
                 WHERE id = ?4",
                params![overwrite, name_overwrite, Utc::now().to_rfc3339(), id_str],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        }

        // Delete tags
        conn.execute(
            "DELETE FROM credential_tags WHERE credential_id = ?1",
            params![id_str],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;

        // Delete the row
        conn.execute("DELETE FROM credentials WHERE id = ?1", params![id_str])
            .map_err(|e| HxError::Storage(e.to_string()))?;

        // Checkpoint WAL and vacuum to ensure overwritten data is flushed
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")
            .map_err(|e| HxError::Storage(e.to_string()))?;

        // VACUUM cannot run inside a transaction; best-effort
        let _ = conn.execute_batch("VACUUM;");

        Ok(())
    }

    async fn touch_credential(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE credentials SET last_accessed_at = ?1, access_count = access_count + 1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    // -- Delegations --

    async fn insert_delegation(&self, delegation: &Delegation) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO delegations (id, credential_id, delegatee, parent_id, can_read, \
             can_use, can_delegate, chain_hash, created_at, expires_at, revoked_at, max_depth, depth) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
            params![
                delegation.id.to_string(),
                delegation.credential_id.to_string(),
                delegation.delegatee,
                delegation.parent_id.map(|id| id.to_string()),
                delegation.permissions.can_read,
                delegation.permissions.can_use,
                delegation.permissions.can_delegate,
                delegation.chain_hash,
                delegation.created_at.to_rfc3339(),
                delegation.expires_at.map(|dt| dt.to_rfc3339()),
                delegation.revoked_at.map(|dt| dt.to_rfc3339()),
                delegation.max_depth as i32,
                delegation.depth as i32,
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get_delegation(&self, id: Uuid) -> MvResult<Option<Delegation>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, credential_id, delegatee, parent_id, can_read, can_use, can_delegate, \
                 chain_hash, created_at, expires_at, revoked_at, max_depth, depth \
                 FROM delegations WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        stmt.query_row(params![id.to_string()], row_to_delegation)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn list_delegations(&self, credential_id: Uuid) -> MvResult<Vec<Delegation>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, credential_id, delegatee, parent_id, can_read, can_use, can_delegate, \
                 chain_hash, created_at, expires_at, revoked_at, max_depth, depth \
                 FROM delegations WHERE credential_id = ?1 ORDER BY created_at",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![credential_id.to_string()], row_to_delegation)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn revoke_delegation(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE delegations SET revoked_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn revoke_delegations_for_credential(&self, credential_id: Uuid) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE delegations SET revoked_at = ?1 WHERE credential_id = ?2 AND revoked_at IS NULL",
            params![Utc::now().to_rfc3339(), credential_id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    // -- Audit --

    async fn append_audit_entry(&self, entry: &KeychainAuditEntry) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let details_json = entry.details.as_ref().map(|d| d.to_string());
        conn.execute(
            "INSERT INTO keychain_audit (id, action, subject, resource_id, details_json, \
             entry_hash, previous_hash, timestamp, source_ip, signature) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                entry.id.to_string(),
                entry.action.as_str(),
                entry.subject,
                entry.resource_id,
                details_json,
                entry.entry_hash,
                entry.previous_hash,
                entry.timestamp.to_rfc3339(),
                entry.source_ip,
                entry.signature.as_deref(),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn list_audit_entries(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<KeychainAuditEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, sequence, action, subject, resource_id, details_json, \
                 entry_hash, previous_hash, timestamp, source_ip, signature \
                 FROM keychain_audit ORDER BY sequence DESC LIMIT ?1 OFFSET ?2",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit as i64, offset as i64], row_to_audit_entry)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn get_latest_audit_entry(&self) -> MvResult<Option<KeychainAuditEntry>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, sequence, action, subject, resource_id, details_json, \
                 entry_hash, previous_hash, timestamp, source_ip, signature \
                 FROM keychain_audit ORDER BY sequence DESC LIMIT 1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        stmt.query_row([], row_to_audit_entry)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn verify_audit_chain(&self) -> MvResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, sequence, action, subject, resource_id, details_json, \
                 entry_hash, previous_hash, timestamp, source_ip, signature \
                 FROM keychain_audit ORDER BY sequence ASC",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let entries: Vec<KeychainAuditEntry> = stmt
            .query_map([], row_to_audit_entry)
            .map_err(|e| HxError::Storage(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut prev_hash: Option<String> = None;
        for entry in &entries {
            let expected = crate::vault_crypto::VaultCrypto::compute_audit_hash(
                prev_hash.as_deref(),
                entry.sequence,
                entry.action.as_str(),
                &entry.subject,
                entry.resource_id.as_deref(),
                &entry.timestamp.to_rfc3339(),
            );
            if entry.entry_hash != expected {
                return Ok(false);
            }
            if entry.previous_hash.as_deref() != prev_hash.as_deref() {
                return Ok(false);
            }
            prev_hash = Some(entry.entry_hash.clone());
        }

        Ok(true)
    }

    // -- Breach detection --

    async fn record_access_pattern(&self, pattern: &AccessPattern) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO access_patterns (credential_id, accessor, source_ip, timestamp, hour_of_day, day_of_week) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                pattern.credential_id.to_string(),
                pattern.accessor,
                pattern.source_ip,
                pattern.timestamp.to_rfc3339(),
                pattern.hour_of_day as i32,
                pattern.day_of_week as i32,
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get_access_patterns(
        &self,
        credential_id: Uuid,
        limit: usize,
    ) -> MvResult<Vec<AccessPattern>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT credential_id, accessor, source_ip, timestamp, hour_of_day, day_of_week \
                 FROM access_patterns WHERE credential_id = ?1 ORDER BY timestamp DESC LIMIT ?2",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![credential_id.to_string(), limit as i64], |row| {
                Ok(AccessPattern {
                    credential_id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    accessor: row.get(1)?,
                    source_ip: row.get(2)?,
                    timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(3)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_default(),
                    hour_of_day: row.get::<_, i32>(4)? as u8,
                    day_of_week: row.get::<_, i32>(5)? as u8,
                })
            })
            .map_err(|e| HxError::Storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn insert_breach_alert(&self, alert: &BreachAlert) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let details_json = alert.details.as_ref().map(|d| d.to_string());
        conn.execute(
            "INSERT INTO breach_alerts (id, credential_id, alert_type, severity, description, \
             details_json, timestamp, acknowledged_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                alert.id.to_string(),
                alert.credential_id.to_string(),
                alert.alert_type.as_str(),
                alert.severity.as_str(),
                alert.description,
                details_json,
                alert.timestamp.to_rfc3339(),
                alert.acknowledged_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn list_breach_alerts(&self, limit: usize, offset: usize) -> MvResult<Vec<BreachAlert>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, credential_id, alert_type, severity, description, details_json, \
                 timestamp, acknowledged_at FROM breach_alerts ORDER BY timestamp DESC LIMIT ?1 OFFSET ?2",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                Ok(BreachAlert {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap_or_default(),
                    credential_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap_or_default(),
                    alert_type: row
                        .get::<_, String>(2)?
                        .parse()
                        .unwrap_or(BreachAlertType::UnusualFrequency),
                    severity: row
                        .get::<_, String>(3)?
                        .parse()
                        .unwrap_or(BreachSeverity::Low),
                    description: row.get(4)?,
                    details: row
                        .get::<_, Option<String>>(5)?
                        .and_then(|s| serde_json::from_str(&s).ok()),
                    timestamp: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
                        .map(|dt| dt.with_timezone(&Utc))
                        .unwrap_or_default(),
                    acknowledged_at: row
                        .get::<_, Option<String>>(7)?
                        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
                        .map(|dt| dt.with_timezone(&Utc)),
                })
            })
            .map_err(|e| HxError::Storage(e.to_string()))?;

        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn acknowledge_breach_alert(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE breach_alerts SET acknowledged_at = ?1 WHERE id = ?2",
            params![Utc::now().to_rfc3339(), id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn has_recent_breach_alert(
        &self,
        credential_id: Uuid,
        alert_type: &str,
        within_secs: u64,
    ) -> MvResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let cutoff = (Utc::now() - chrono::Duration::seconds(within_secs as i64)).to_rfc3339();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM breach_alerts \
                 WHERE credential_id = ?1 AND alert_type = ?2 AND timestamp > ?3",
                params![credential_id.to_string(), alert_type, cutoff],
                |row| row.get(0),
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(count > 0)
    }

    // -- Tags --

    async fn get_credential_tags(&self, credential_id: Uuid) -> MvResult<Vec<String>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(load_credential_tags(&conn, &credential_id.to_string()))
    }

    async fn save_credential_tags(&self, credential_id: Uuid, tags: &[String]) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        save_credential_tags_inner(&conn, &credential_id.to_string(), tags)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn set_lockout_state(&self, attempts: u32, locked_until: Option<String>) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE keychain_meta SET failed_unseal_attempts = ?1, locked_until = ?2 WHERE id = 1",
            params![attempts as i64, locked_until],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get_lockout_state(&self) -> MvResult<(u32, Option<String>)> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = conn
            .query_row(
                "SELECT failed_unseal_attempts, locked_until FROM keychain_meta WHERE id = 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)? as u32,
                        row.get::<_, Option<String>>(1)?,
                    ))
                },
            )
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result.unwrap_or((0, None)))
    }

    // --- Domain ACLs ---

    async fn insert_acl(&self, acl: &DomainAcl) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO credential_acls \
             (id, domain_id, subject, can_read, can_write, can_admin, created_at, expires_at) \
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                acl.id.to_string(),
                acl.domain_id.to_string(),
                acl.subject,
                acl.can_read,
                acl.can_write,
                acl.can_admin,
                acl.created_at.to_rfc3339(),
                acl.expires_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn get_acls_for_domain(&self, domain_id: Uuid) -> MvResult<Vec<DomainAcl>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, domain_id, subject, can_read, can_write, can_admin, \
                 created_at, expires_at FROM credential_acls WHERE domain_id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![domain_id.to_string()], row_to_acl)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        rows.collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn get_acl_for_subject(
        &self,
        domain_id: Uuid,
        subject: &str,
    ) -> MvResult<Option<DomainAcl>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, domain_id, subject, can_read, can_write, can_admin, \
                 created_at, expires_at FROM credential_acls \
                 WHERE domain_id = ?1 AND subject = ?2",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        stmt.query_row(params![domain_id.to_string(), subject], row_to_acl)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))
    }

    async fn delete_acl(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "DELETE FROM credential_acls WHERE id = ?1",
            params![id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> SqliteKeychainStore {
        SqliteKeychainStore::open_in_memory().unwrap()
    }

    #[tokio::test]
    async fn vault_meta_crud() {
        let s = store();
        assert!(s.get_vault_meta().await.unwrap().is_none());

        let meta = VaultMeta {
            schema_version: 1,
            master_salt: "salt123".into(),
            verification_blob: "blob456".into(),
            key_epoch: 0,
            created_at: Utc::now(),
            last_rotated_at: None,
            macos_keychain_service: None,
            shamir_threshold: None,
            shamir_total: None,
            shamir_last_rotated_at: None,
        };
        s.save_vault_meta(&meta).await.unwrap();

        let loaded = s.get_vault_meta().await.unwrap().unwrap();
        assert_eq!(loaded.master_salt, "salt123");
    }

    #[tokio::test]
    async fn key_epoch_lifecycle() {
        let s = store();
        let epoch = KeyEpoch {
            epoch: 0,
            wrapped_key: None,
            created_at: Utc::now(),
            grace_expires_at: None,
            retired_at: None,
        };
        s.insert_key_epoch(&epoch).await.unwrap();
        assert_eq!(s.list_key_epochs().await.unwrap().len(), 1);
        s.retire_key_epoch(0).await.unwrap();
        let loaded = s.get_key_epoch(0).await.unwrap().unwrap();
        assert!(loaded.retired_at.is_some());
    }

    #[tokio::test]
    async fn domain_crud() {
        let s = store();
        // Need epoch 0 first
        s.insert_key_epoch(&KeyEpoch {
            epoch: 0,
            wrapped_key: None,
            created_at: Utc::now(),
            grace_expires_at: None,
            retired_at: None,
        })
        .await
        .unwrap();

        let domain = DomainKey::new("api-keys", "domain:api-keys").with_epoch(0);
        s.insert_domain(&domain).await.unwrap();
        let loaded = s.get_domain(domain.id).await.unwrap().unwrap();
        assert_eq!(loaded.name, "api-keys");

        let by_name = s.get_domain_by_name("api-keys").await.unwrap().unwrap();
        assert_eq!(by_name.id, domain.id);

        let all = s.list_domains().await.unwrap();
        assert_eq!(all.len(), 1);

        s.revoke_domain(domain.id).await.unwrap();
        let revoked = s.get_domain(domain.id).await.unwrap().unwrap();
        assert!(revoked.revoked_at.is_some());
    }

    #[tokio::test]
    async fn credential_crud_and_shred() {
        let s = store();
        s.insert_key_epoch(&KeyEpoch {
            epoch: 0,
            wrapped_key: None,
            created_at: Utc::now(),
            grace_expires_at: None,
            retired_at: None,
        })
        .await
        .unwrap();
        let domain = DomainKey::new("test", "domain:test").with_epoch(0);
        s.insert_domain(&domain).await.unwrap();

        let cred = StoredCredential::new(
            domain.id,
            "my-key",
            "api_key",
            "encrypted-data".into(),
            "cred:my-key",
        )
        .with_tags(vec!["prod".into(), "api".into()]);

        s.insert_credential(&cred).await.unwrap();
        let loaded = s.get_credential(cred.id).await.unwrap().unwrap();
        assert_eq!(loaded.name, "my-key");
        assert_eq!(loaded.tags, vec!["api", "prod"]);

        assert_eq!(s.count_credentials(None).await.unwrap(), 1);

        s.touch_credential(cred.id).await.unwrap();
        let touched = s.get_credential(cred.id).await.unwrap().unwrap();
        assert_eq!(touched.access_count, 1);

        s.shred_credential(cred.id).await.unwrap();
        assert!(s.get_credential(cred.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delegation_crud() {
        let s = store();
        s.insert_key_epoch(&KeyEpoch {
            epoch: 0,
            wrapped_key: None,
            created_at: Utc::now(),
            grace_expires_at: None,
            retired_at: None,
        })
        .await
        .unwrap();
        let domain = DomainKey::new("test", "domain:test").with_epoch(0);
        s.insert_domain(&domain).await.unwrap();
        let cred = StoredCredential::new(domain.id, "key1", "api_key", "enc".into(), "cred:key1");
        s.insert_credential(&cred).await.unwrap();

        let delegation = Delegation {
            id: Uuid::now_v7(),
            credential_id: cred.id,
            delegatee: "alice".into(),
            parent_id: None,
            permissions: DelegationPermissions::default(),
            chain_hash: "hash123".into(),
            created_at: Utc::now(),
            expires_at: None,
            revoked_at: None,
            max_depth: 2,
            depth: 0,
        };
        s.insert_delegation(&delegation).await.unwrap();
        let loaded = s.get_delegation(delegation.id).await.unwrap().unwrap();
        assert_eq!(loaded.delegatee, "alice");

        let list = s.list_delegations(cred.id).await.unwrap();
        assert_eq!(list.len(), 1);

        s.revoke_delegation(delegation.id).await.unwrap();
        let revoked = s.get_delegation(delegation.id).await.unwrap().unwrap();
        assert!(revoked.revoked_at.is_some());
    }
}
