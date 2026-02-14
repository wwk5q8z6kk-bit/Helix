use std::path::Path;
use std::sync::Mutex;

use async_trait::async_trait;
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use chrono::Utc;
use rusqlite::types::Type;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::sealed_runtime::{runtime_root_key, sealed_mode_enabled};
use crate::vault_crypto::VaultCrypto;
use hx_core::*;

/// Default number of connections in the pool.
/// SQLite WAL mode supports 1 writer + N readers, so even a small pool
/// eliminates head-of-line blocking for concurrent read queries.
const DEFAULT_POOL_SIZE: usize = 4;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SealedNodePayload {
    title: Option<String>,
    content: String,
    source: Option<String>,
    metadata: std::collections::HashMap<String, serde_json::Value>,
}

pub struct SqliteNodeStore {
    /// Connection pool — round-robin across `DEFAULT_POOL_SIZE` connections.
    /// Each connection is independently protected by a Mutex so callers can
    /// run synchronous rusqlite operations without holding an async lock.
    pool: Vec<Mutex<Connection>>,
    /// Atomic counter for round-robin slot selection.
    next_slot: std::sync::atomic::AtomicUsize,
}

impl SqliteNodeStore {
    /// Execute a synchronous closure with a pooled database connection.
    ///
    /// Picks the next connection via round-robin, locks it, runs the
    /// closure, then releases. Because the closure is `FnOnce` (not async),
    /// the `MutexGuard` is guaranteed to drop before any `.await` — making
    /// the enclosing future `Send`.
    fn with_conn<F, T>(&self, f: F) -> MvResult<T>
    where
        F: FnOnce(&Connection) -> MvResult<T>,
    {
        let idx = self
            .next_slot
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.pool.len();
        let conn = self.pool[idx]
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        f(&conn)
    }

    /// Access a pooled connection via round-robin, matching `with_conn`
    /// distribution. Callers must `.lock()` the returned mutex.
    #[inline]
    fn conn(&self) -> &Mutex<Connection> {
        let idx = self
            .next_slot
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            % self.pool.len();
        &self.pool[idx]
    }
}

impl SqliteNodeStore {
    fn open_connection(path: &Path) -> MvResult<Connection> {
        let conn = Connection::open(path)
            .map_err(|e| HxError::Storage(format!("failed to open sqlite: {e}")))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
        )
        .map_err(|e| HxError::Storage(format!("pragma error: {e}")))?;

        Ok(conn)
    }

    pub fn open(path: &Path) -> MvResult<Self> {
        let mut pool = Vec::with_capacity(DEFAULT_POOL_SIZE);
        for _ in 0..DEFAULT_POOL_SIZE {
            pool.push(Mutex::new(Self::open_connection(path)?));
        }

        let store = Self {
            pool,
            next_slot: std::sync::atomic::AtomicUsize::new(0),
        };
        store.run_migrations()?;
        Ok(store)
    }

    pub fn open_in_memory() -> MvResult<Self> {
        // In-memory DBs: use a shared cache URI so all pool connections see
        // the same data. Without this, each Connection::open_in_memory()
        // gets its own isolated database.
        //
        // SQLITE_OPEN_URI is required for rusqlite to parse the URI; the
        // default OpenFlags do NOT include it.
        let uri = format!(
            "file:memdb{}?mode=memory&cache=shared",
            uuid::Uuid::new_v4()
        );
        let flags = rusqlite::OpenFlags::SQLITE_OPEN_READ_WRITE
            | rusqlite::OpenFlags::SQLITE_OPEN_CREATE
            | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX
            | rusqlite::OpenFlags::SQLITE_OPEN_URI;
        let mut pool = Vec::with_capacity(DEFAULT_POOL_SIZE);
        for _ in 0..DEFAULT_POOL_SIZE {
            let conn = Connection::open_with_flags(&uri, flags)
                .map_err(|e| HxError::Storage(format!("failed to open in-memory sqlite: {e}")))?;
            conn.execute_batch("PRAGMA foreign_keys=ON;")
                .map_err(|e| HxError::Storage(format!("pragma error: {e}")))?;
            pool.push(Mutex::new(conn));
        }

        let store = Self {
            pool,
            next_slot: std::sync::atomic::AtomicUsize::new(0),
        };
        store.run_migrations()?;
        Ok(store)
    }

    pub fn open_read_only(path: &Path) -> MvResult<Self> {
        let conn = Connection::open_with_flags(path, rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY)
            .map_err(|e| HxError::Storage(format!("failed to open sqlite (read-only): {e}")))?;

        conn.execute_batch("PRAGMA foreign_keys=ON; PRAGMA query_only=ON;")
            .map_err(|e| HxError::Storage(format!("pragma error: {e}")))?;

        // Read-only mode: single connection is fine (no write contention)
        Ok(Self {
            pool: vec![Mutex::new(conn)],
            next_slot: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn run_migrations(&self) -> MvResult<()> {
        // Migrations run on slot 0 only — they need exclusive access.
        let conn = self.pool[0]
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        // Table-driven migration registry.
        // Versions 002, 009, 017-021 are keychain-only and applied in
        // SqliteKeychainStore — they are intentionally excluded here.
        const MIGRATIONS: &[(i64, &str)] = &[
            (1, include_str!("../../../migrations/001_initial.sql")),
            (3, include_str!("../../../migrations/003_agentic.sql")),
            (4, include_str!("../../../migrations/004_exchange.sql")),
            (
                5,
                include_str!("../../../migrations/005_relay_safeguards.sql"),
            ),
            (6, include_str!("../../../migrations/006_feedback.sql")),
            (7, include_str!("../../../migrations/007_autonomy.sql")),
            (8, include_str!("../../../migrations/008_relay.sql")),
            (10, include_str!("../../../migrations/010_profile.sql")),
            (
                11,
                include_str!("../../../migrations/011_consumer_profiles.sql"),
            ),
            (
                12,
                include_str!("../../../migrations/012_access_policies.sql"),
            ),
            (13, include_str!("../../../migrations/013_proxy_audit.sql")),
            (14, include_str!("../../../migrations/014_conflicts.sql")),
            (
                15,
                include_str!("../../../migrations/015_contact_identity.sql"),
            ),
            (
                16,
                include_str!("../../../migrations/016_approval_queue.sql"),
            ),
            (
                22,
                include_str!("../../../migrations/022_adapter_poll_state.sql"),
            ),
            (
                23,
                include_str!("../../../migrations/023_conversations.sql"),
            ),
            (24, include_str!("../../../migrations/024_plans.sql")),
            (
                25,
                include_str!("../../../migrations/025_public_shares.sql"),
            ),
            (
                26,
                include_str!("../../../migrations/026_node_comments.sql"),
            ),
            (
                27,
                include_str!("../../../migrations/027_mcp_connectors.sql"),
            ),
            (
                28,
                include_str!("../../../migrations/028_sealed_node_payloads.sql"),
            ),
        ];

        // Migration 001 must always run first to create schema_version table.
        // After that, check which versions are already applied.
        conn.execute_batch(MIGRATIONS[0].1)
            .map_err(|e| HxError::Migration(format!("migration 001 failed: {e}")))?;

        let max_version: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        for &(version, sql) in &MIGRATIONS[1..] {
            if version <= max_version {
                continue;
            }
            conn.execute_batch(sql)
                .map_err(|e| HxError::Migration(format!("migration {version:03} failed: {e}")))?;
        }

        tracing::debug!(
            applied_up_to = MIGRATIONS.last().map(|(v, _)| *v).unwrap_or(0),
            "Migrations complete"
        );

        Ok(())
    }

    fn as_sql_conversion_error(column: usize, message: impl Into<String>) -> rusqlite::Error {
        rusqlite::Error::FromSqlConversionFailure(
            column,
            Type::Text,
            Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                message.into(),
            )),
        )
    }

    fn derive_namespace_kek(namespace: &str) -> MvResult<[u8; 32]> {
        let root = runtime_root_key().ok_or(HxError::VaultSealed)?;
        let mut crypto = VaultCrypto::new();
        crypto.set_master_key(Zeroizing::new(root));
        let key = crypto
            .derive_namespace_kek(namespace)
            .map_err(|err| HxError::Storage(format!("derive namespace key failed: {err}")))?;
        Ok(*key)
    }

    fn encrypt_node_payload(node: &KnowledgeNode) -> MvResult<(String, String)> {
        let kek = Self::derive_namespace_kek(&node.namespace)?;
        let dek = VaultCrypto::generate_node_dek();
        let payload = SealedNodePayload {
            title: node.title.clone(),
            content: node.content.clone(),
            source: node.source.clone(),
            metadata: node.metadata.clone(),
        };
        let plaintext = serde_json::to_vec(&payload)
            .map_err(|err| HxError::Storage(format!("serialize sealed payload: {err}")))?;
        let ciphertext = VaultCrypto::aes_gcm_encrypt_pub(&dek, &plaintext)
            .map_err(|err| HxError::Storage(format!("encrypt sealed payload: {err}")))?;
        let wrapped_dek = VaultCrypto::wrap_node_dek(&kek, &dek)
            .map_err(|err| HxError::Storage(format!("wrap node dek failed: {err}")))?;
        Ok((wrapped_dek, BASE64.encode(ciphertext)))
    }

    fn decrypt_node_payload(
        namespace: &str,
        wrapped_dek: &str,
        payload_ciphertext: &str,
    ) -> MvResult<SealedNodePayload> {
        let kek = Self::derive_namespace_kek(namespace)?;
        let dek = VaultCrypto::unwrap_node_dek(&kek, wrapped_dek)
            .map_err(|err| HxError::Storage(format!("unwrap node dek failed: {err}")))?;
        let ciphertext = BASE64
            .decode(payload_ciphertext)
            .map_err(|err| HxError::Storage(format!("decode sealed payload failed: {err}")))?;
        let plaintext = VaultCrypto::aes_gcm_decrypt_pub(&dek, &ciphertext)
            .map_err(|err| HxError::Storage(format!("decrypt sealed payload failed: {err}")))?;
        let payload: SealedNodePayload = serde_json::from_slice(&plaintext)
            .map_err(|err| HxError::Storage(format!("parse sealed payload failed: {err}")))?;
        Ok(payload)
    }

    fn project_node_for_storage(
        node: &KnowledgeNode,
    ) -> MvResult<(
        Option<String>,
        String,
        Option<String>,
        Option<String>,
        Option<String>,
        Option<String>,
    )> {
        if sealed_mode_enabled() {
            let (wrapped_dek, ciphertext) = Self::encrypt_node_payload(node)?;
            Ok((
                None,
                String::new(),
                None,
                None,
                Some(ciphertext),
                Some(wrapped_dek),
            ))
        } else {
            Ok((
                node.title.clone(),
                node.content.clone(),
                node.source.clone(),
                Some(serde_json::to_string(&node.metadata)?),
                None,
                None,
            ))
        }
    }

    fn row_to_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<KnowledgeNode> {
        let id_str: String = row.get(0)?;
        let kind_str: String = row.get(1)?;
        let mut title: Option<String> = row.get(2)?;
        let content: String = row.get(3)?;
        let mut source: Option<String> = row.get(4)?;
        let namespace: String = row.get(5)?;
        let importance: f64 = row.get(6)?;
        let created_at: String = row.get(7)?;
        let updated_at: String = row.get(8)?;
        let last_accessed_at: String = row.get(9)?;
        let access_count: u64 = row.get(10)?;
        let version: u32 = row.get(11)?;
        let expires_at: Option<String> = row.get(12)?;
        let mut metadata_json: Option<String> = row.get(13)?;
        let payload_ciphertext: Option<String> = row.get(14).ok();
        let wrapped_dek: Option<String> = row.get(15).ok();

        if let (Some(payload_ciphertext), Some(wrapped_dek)) = (payload_ciphertext, wrapped_dek) {
            let payload = Self::decrypt_node_payload(&namespace, &wrapped_dek, &payload_ciphertext)
                .map_err(|err| {
                    Self::as_sql_conversion_error(
                        14,
                        format!("failed to decrypt node payload: {err}"),
                    )
                })?;
            title = payload.title;
            source = payload.source;
            metadata_json = Some(serde_json::to_string(&payload.metadata).map_err(|err| {
                Self::as_sql_conversion_error(
                    13,
                    format!("failed to reserialize node metadata: {err}"),
                )
            })?);

            return Ok(KnowledgeNode {
                id: parse_uuid_str(0, &id_str)?,
                kind: kind_str
                    .parse()
                    .map_err(|err: String| Self::as_sql_conversion_error(1, err))?,
                title,
                content: payload.content,
                source,
                namespace,
                tags: Vec::new(),
                importance,
                temporal: TemporalMeta {
                    created_at: parse_dt_strict(7, &created_at)?,
                    updated_at: parse_dt_strict(8, &updated_at)?,
                    last_accessed_at: parse_dt_strict(9, &last_accessed_at)?,
                    access_count,
                    version,
                    expires_at: parse_optional_dt_strict(12, expires_at)?,
                },
                metadata: parse_metadata_json(metadata_json)?,
            });
        }

        let id = parse_uuid_str(0, &id_str)?;
        let kind: NodeKind = kind_str.parse().map_err(|err: String| {
            rusqlite::Error::FromSqlConversionFailure(
                1,
                Type::Text,
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, err)),
            )
        })?;
        let metadata = parse_metadata_json(metadata_json)?;

        Ok(KnowledgeNode {
            id,
            kind,
            title,
            content,
            source,
            namespace,
            tags: Vec::new(), // loaded separately
            importance,
            temporal: TemporalMeta {
                created_at: parse_dt_strict(7, &created_at)?,
                updated_at: parse_dt_strict(8, &updated_at)?,
                last_accessed_at: parse_dt_strict(9, &last_accessed_at)?,
                access_count,
                version,
                expires_at: parse_optional_dt_strict(12, expires_at)?,
            },
            metadata,
        })
    }

    fn load_tags(conn: &Connection, node_id: Uuid) -> MvResult<Vec<String>> {
        let mut stmt = conn
            .prepare("SELECT tag FROM node_tags WHERE node_id = ?1 ORDER BY tag")
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut tags = Vec::new();
        let rows = stmt
            .query_map(params![node_id.to_string()], |row| row.get(0))
            .map_err(|e| HxError::Storage(e.to_string()))?;

        for row in rows {
            tags.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }

        Ok(tags)
    }

    fn save_tags(conn: &Connection, node_id: Uuid, tags: &[String]) -> MvResult<()> {
        conn.execute(
            "DELETE FROM node_tags WHERE node_id = ?1",
            params![node_id.to_string()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare("INSERT INTO node_tags (node_id, tag) VALUES (?1, ?2)")
            .map_err(|e| HxError::Storage(e.to_string()))?;
        for tag in tags {
            stmt.execute(params![node_id.to_string(), tag])
                .map_err(|e| HxError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    fn log_change(
        conn: &Connection,
        node_id: Uuid,
        op: ChangeOp,
        diff: Option<&serde_json::Value>,
    ) -> MvResult<()> {
        let diff_str = diff.map(|d| serde_json::to_string(d).unwrap_or_default());
        conn.execute(
            "INSERT INTO changelog (node_id, operation, diff_json, timestamp) VALUES (?1, ?2, ?3, ?4)",
            params![
                node_id.to_string(),
                op.as_str(),
                diff_str,
                Utc::now().to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }
}

fn parse_uuid_str(column: usize, s: &str) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(s)
        .map_err(|err| rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(err)))
}

fn parse_dt_strict(column: usize, s: &str) -> rusqlite::Result<chrono::DateTime<Utc>> {
    chrono::DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(err)))
}

fn parse_optional_dt_strict(
    column: usize,
    s: Option<String>,
) -> rusqlite::Result<Option<chrono::DateTime<Utc>>> {
    match s {
        Some(value) => parse_dt_strict(column, &value).map(Some),
        None => Ok(None),
    }
}

fn parse_metadata_json(
    metadata_json: Option<String>,
) -> rusqlite::Result<std::collections::HashMap<String, serde_json::Value>> {
    match metadata_json {
        Some(raw) => serde_json::from_str(&raw).map_err(|err| {
            rusqlite::Error::FromSqlConversionFailure(13, Type::Text, Box::new(err))
        }),
        None => Ok(Default::default()),
    }
}

fn row_to_public_share(row: &rusqlite::Row<'_>) -> rusqlite::Result<PublicShare> {
    let id_str: String = row.get(0)?;
    let node_id_str: String = row.get(1)?;
    let token_hash: String = row.get(2)?;
    let created_at_str: String = row.get(3)?;
    let expires_at_str: Option<String> = row.get(4)?;
    let revoked_at_str: Option<String> = row.get(5)?;

    Ok(PublicShare {
        id: parse_uuid_str(0, &id_str)?,
        node_id: parse_uuid_str(1, &node_id_str)?,
        token_hash,
        created_at: parse_dt_strict(3, &created_at_str)?,
        expires_at: parse_optional_dt_strict(4, expires_at_str)?,
        revoked_at: parse_optional_dt_strict(5, revoked_at_str)?,
    })
}

fn row_to_node_comment(row: &rusqlite::Row<'_>) -> rusqlite::Result<NodeComment> {
    let id_str: String = row.get(0)?;
    let node_id_str: String = row.get(1)?;
    let author: Option<String> = row.get(2)?;
    let body: String = row.get(3)?;
    let created_at_str: String = row.get(4)?;
    let updated_at_str: String = row.get(5)?;
    let resolved_at_str: Option<String> = row.get(6)?;

    Ok(NodeComment {
        id: parse_uuid_str(0, &id_str)?,
        node_id: parse_uuid_str(1, &node_id_str)?,
        author,
        body,
        created_at: parse_dt_strict(4, &created_at_str)?,
        updated_at: parse_dt_strict(5, &updated_at_str)?,
        resolved_at: parse_optional_dt_strict(6, resolved_at_str)?,
    })
}

fn row_to_mcp_connector(row: &rusqlite::Row<'_>) -> rusqlite::Result<McpConnector> {
    let id_str: String = row.get(0)?;
    let name: String = row.get(1)?;
    let description: Option<String> = row.get(2)?;
    let publisher: Option<String> = row.get(3)?;
    let version: String = row.get(4)?;
    let homepage_url: Option<String> = row.get(5)?;
    let repository_url: Option<String> = row.get(6)?;
    let config_schema_json: String = row.get(7)?;
    let capabilities_json: String = row.get(8)?;
    let verified: i32 = row.get(9)?;
    let created_at_str: String = row.get(10)?;
    let updated_at_str: String = row.get(11)?;

    let config_schema =
        serde_json::from_str(&config_schema_json).unwrap_or_else(|_| serde_json::json!({}));
    let capabilities: Vec<String> = serde_json::from_str(&capabilities_json).unwrap_or_default();

    Ok(McpConnector {
        id: parse_uuid_str(0, &id_str)?,
        name,
        description,
        publisher,
        version,
        homepage_url,
        repository_url,
        config_schema,
        capabilities,
        verified: verified != 0,
        created_at: parse_dt_strict(10, &created_at_str)?,
        updated_at: parse_dt_strict(11, &updated_at_str)?,
    })
}

#[async_trait]
impl NodeStore for SqliteNodeStore {
    async fn insert(&self, node: &KnowledgeNode) -> MvResult<()> {
        self.with_conn(|conn| {
            let (title, content, source, metadata_json, payload_ciphertext, payload_wrapped_dek) =
                Self::project_node_for_storage(node)?;

            conn.execute(
                "INSERT INTO knowledge_nodes (id, kind, title, content, source, namespace, importance,
                 created_at, updated_at, last_accessed_at, access_count, version, expires_at, metadata_json, payload_ciphertext, payload_wrapped_dek)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                params![
                    node.id.to_string(),
                    node.kind.as_str(),
                    title,
                    content,
                    source,
                    node.namespace,
                    node.importance,
                    node.temporal.created_at.to_rfc3339(),
                    node.temporal.updated_at.to_rfc3339(),
                    node.temporal.last_accessed_at.to_rfc3339(),
                    node.temporal.access_count,
                    node.temporal.version,
                    node.temporal.expires_at.map(|dt| dt.to_rfc3339()),
                    metadata_json,
                    payload_ciphertext,
                    payload_wrapped_dek,
                ],
            )
            .map_err(|e| HxError::Storage(format!("insert failed: {e}")))?;

            Self::save_tags(conn, node.id, &node.tags)?;
            Self::log_change(conn, node.id, ChangeOp::Create, None)?;
            Ok(())
        })
    }

    async fn get(&self, id: Uuid) -> MvResult<Option<KnowledgeNode>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, kind, title, content, source, namespace, importance,
                     created_at, updated_at, last_accessed_at, access_count, version,
                     expires_at, metadata_json, payload_ciphertext, payload_wrapped_dek FROM knowledge_nodes WHERE id = ?1",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;

            let node = stmt
                .query_row(params![id.to_string()], Self::row_to_node)
                .optional()
                .map_err(|e| HxError::Storage(e.to_string()))?;

            if let Some(mut node) = node {
                node.tags = Self::load_tags(conn, node.id)?;
                Ok(Some(node))
            } else {
                Ok(None)
            }
        })
    }

    async fn update(&self, node: &KnowledgeNode) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let (title, content, source, metadata_json, payload_ciphertext, payload_wrapped_dek) =
            Self::project_node_for_storage(node)?;

        let rows = conn
            .execute(
                "UPDATE knowledge_nodes SET kind = ?2, title = ?3, content = ?4, source = ?5, payload_ciphertext = ?14, payload_wrapped_dek = ?15,
                 namespace = ?6, importance = ?7, updated_at = ?8, last_accessed_at = ?9,
                 access_count = ?10, version = ?11, expires_at = ?12, metadata_json = ?13
                 WHERE id = ?1",
                params![
                    node.id.to_string(),
                    node.kind.as_str(),
                    title,
                    content,
                    source,
                    node.namespace,
                    node.importance,
                    node.temporal.updated_at.to_rfc3339(),
                    node.temporal.last_accessed_at.to_rfc3339(),
                    node.temporal.access_count,
                    node.temporal.version,
                    node.temporal.expires_at.map(|dt| dt.to_rfc3339()),
                    metadata_json,
                    payload_ciphertext,
                    payload_wrapped_dek,
                ],
            )
            .map_err(|e| HxError::Storage(format!("update failed: {e}")))?;

        if rows == 0 {
            return Err(HxError::NodeNotFound(node.id));
        }

        Self::save_tags(&conn, node.id, &node.tags)?;
        Self::log_change(&conn, node.id, ChangeOp::Update, None)?;
        Ok(())
    }

    async fn delete(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Self::log_change(&conn, id, ChangeOp::Delete, None)?;
        let rows = conn
            .execute(
                "DELETE FROM knowledge_nodes WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(format!("delete failed: {e}")))?;
        Ok(rows > 0)
    }

    async fn list(
        &self,
        filters: &QueryFilters,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<KnowledgeNode>> {
        self.with_conn(|conn| {
        let mut sql = String::from(
            "SELECT id, kind, title, content, source, namespace, importance,
             created_at, updated_at, last_accessed_at, access_count, version,
             expires_at, metadata_json, payload_ciphertext, payload_wrapped_dek FROM knowledge_nodes WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(ref ns) = filters.namespace {
            sql.push_str(&format!(" AND namespace = ?{param_idx}"));
            param_values.push(Box::new(ns.clone()));
            param_idx += 1;
        }

        if let Some(ref kinds) = filters.kinds {
            if !kinds.is_empty() {
                let placeholders: Vec<String> = kinds
                    .iter()
                    .map(|_| {
                        let p = format!("?{param_idx}");
                        param_idx += 1;
                        p
                    })
                    .collect();
                sql.push_str(&format!(" AND kind IN ({})", placeholders.join(",")));
                for k in kinds {
                    param_values.push(Box::new(k.as_str().to_string()));
                }
            }
        }

        if let Some(min_imp) = filters.min_importance {
            sql.push_str(&format!(" AND importance >= ?{param_idx}"));
            param_values.push(Box::new(min_imp));
            param_idx += 1;
        }

        if let Some(ref after) = filters.created_after {
            sql.push_str(&format!(" AND created_at >= ?{param_idx}"));
            param_values.push(Box::new(after.to_rfc3339()));
            param_idx += 1;
        }

        if let Some(ref before) = filters.created_before {
            sql.push_str(&format!(" AND created_at <= ?{param_idx}"));
            param_values.push(Box::new(before.to_rfc3339()));
            param_idx += 1;
        }

        if let Some(ref filter_tags) = filters.tags {
            if !filter_tags.is_empty() {
                let placeholders: Vec<String> = filter_tags
                    .iter()
                    .map(|_| {
                        let p = format!("?{param_idx}");
                        param_idx += 1;
                        p
                    })
                    .collect();

                sql.push_str(&format!(
                    " AND EXISTS (SELECT 1 FROM node_tags nt WHERE nt.node_id = knowledge_nodes.id AND nt.tag IN ({}))",
                    placeholders.join(",")
                ));

                for tag in filter_tags {
                    param_values.push(Box::new(tag.clone()));
                }
            }
        }

        sql.push_str(&format!(
            " ORDER BY updated_at DESC LIMIT ?{param_idx} OFFSET ?{}",
            param_idx + 1
        ));
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params_refs.as_slice(), Self::row_to_node)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut nodes = Vec::new();
        for row in rows {
            let mut node = row.map_err(|e| HxError::Storage(e.to_string()))?;
            node.tags = Self::load_tags(conn, node.id)?;
            nodes.push(node);
        }

        Ok(nodes)
        })
    }

    async fn touch(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let rows = conn
            .execute(
                "UPDATE knowledge_nodes SET last_accessed_at = ?2, access_count = access_count + 1 WHERE id = ?1",
                params![id.to_string(), now],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        if rows == 0 {
            return Err(HxError::NodeNotFound(id));
        }
        Ok(())
    }

    async fn count(&self, filters: &QueryFilters) -> MvResult<usize> {
        self.with_conn(|conn| {
        let mut sql = String::from("SELECT COUNT(*) FROM knowledge_nodes WHERE 1=1");
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(ref ns) = filters.namespace {
            sql.push_str(&format!(" AND namespace = ?{param_idx}"));
            param_values.push(Box::new(ns.clone()));
            param_idx += 1;
        }

        if let Some(ref kinds) = filters.kinds {
            if !kinds.is_empty() {
                let placeholders: Vec<String> = kinds
                    .iter()
                    .map(|_| {
                        let p = format!("?{param_idx}");
                        param_idx += 1;
                        p
                    })
                    .collect();
                sql.push_str(&format!(" AND kind IN ({})", placeholders.join(",")));
                for k in kinds {
                    param_values.push(Box::new(k.as_str().to_string()));
                }
            }
        }
        if let Some(min_imp) = filters.min_importance {
            sql.push_str(&format!(" AND importance >= ?{param_idx}"));
            param_values.push(Box::new(min_imp));
            param_idx += 1;
        }

        if let Some(ref after) = filters.created_after {
            sql.push_str(&format!(" AND created_at >= ?{param_idx}"));
            param_values.push(Box::new(after.to_rfc3339()));
            param_idx += 1;
        }

        if let Some(ref before) = filters.created_before {
            sql.push_str(&format!(" AND created_at <= ?{param_idx}"));
            param_values.push(Box::new(before.to_rfc3339()));
            param_idx += 1;
        }

        if let Some(ref filter_tags) = filters.tags {
            if !filter_tags.is_empty() {
                let placeholders: Vec<String> = filter_tags
                    .iter()
                    .map(|_| {
                        let p = format!("?{param_idx}");
                        param_idx += 1;
                        p
                    })
                    .collect();

                sql.push_str(&format!(
                    " AND EXISTS (SELECT 1 FROM node_tags nt WHERE nt.node_id = knowledge_nodes.id AND nt.tag IN ({}))",
                    placeholders.join(",")
                ));

                for tag in filter_tags {
                    param_values.push(Box::new(tag.clone()));
                }
            }
        }

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let count: usize = conn
            .query_row(&sql, params_refs.as_slice(), |row| row.get(0))
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(count)
        })
    }
}

impl SqliteNodeStore {
    /// Find a node by its `source` field (exact match).
    /// Used for dedup during imports (e.g. Obsidian vault import).
    pub async fn find_by_source(&self, source: &str) -> MvResult<Option<KnowledgeNode>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, kind, title, content, source, namespace, importance,
                 created_at, updated_at, last_accessed_at, access_count, version,
                 expires_at, metadata_json, payload_ciphertext, payload_wrapped_dek FROM knowledge_nodes WHERE source = ?1 LIMIT 1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let row = stmt
            .query_row(params![source], Self::row_to_node)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        match row {
            Some(mut node) => {
                node.tags = Self::load_tags(&conn, node.id)?;
                Ok(Some(node))
            }
            None => {
                if !sealed_mode_enabled() {
                    return Ok(None);
                }
                let mut stmt = conn
                    .prepare(
                        "SELECT id, kind, title, content, source, namespace, importance,
                         created_at, updated_at, last_accessed_at, access_count, version,
                         expires_at, metadata_json, payload_ciphertext, payload_wrapped_dek FROM knowledge_nodes WHERE payload_ciphertext IS NOT NULL",
                    )
                    .map_err(|e| HxError::Storage(e.to_string()))?;
                let rows = stmt
                    .query_map([], Self::row_to_node)
                    .map_err(|e| HxError::Storage(e.to_string()))?;
                for row in rows {
                    let mut node = row.map_err(|e| HxError::Storage(e.to_string()))?;
                    if node.source.as_deref() == Some(source) {
                        node.tags = Self::load_tags(&conn, node.id)?;
                        return Ok(Some(node));
                    }
                }
                Ok(None)
            }
        }
    }

    pub async fn insert_permission_template(&self, template: &PermissionTemplate) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let scope_tags_json = serde_json::to_string(&template.scope_tags)?;
        let allow_kinds_json = serde_json::to_string(
            &template
                .allow_kinds
                .iter()
                .map(|kind| kind.as_str())
                .collect::<Vec<_>>(),
        )?;
        let allow_actions_json = serde_json::to_string(&template.allow_actions)?;

        conn.execute(
            "INSERT INTO permission_templates (id, name, description, tier, scope_namespace, scope_tags_json, allow_kinds_json, allow_actions_json, created_at, updated_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                template.id.to_string(),
                template.name,
                template.description,
                template.tier.as_str(),
                template.scope_namespace,
                scope_tags_json,
                allow_kinds_json,
                allow_actions_json,
                template.created_at.to_rfc3339(),
                template.updated_at.to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    pub async fn update_permission_template(&self, template: &PermissionTemplate) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let scope_tags_json = serde_json::to_string(&template.scope_tags)?;
        let allow_kinds_json = serde_json::to_string(
            &template
                .allow_kinds
                .iter()
                .map(|kind| kind.as_str())
                .collect::<Vec<_>>(),
        )?;
        let allow_actions_json = serde_json::to_string(&template.allow_actions)?;

        conn.execute(
            "UPDATE permission_templates SET name = ?2, description = ?3, tier = ?4, scope_namespace = ?5, scope_tags_json = ?6, allow_kinds_json = ?7, allow_actions_json = ?8, updated_at = ?9 WHERE id = ?1",
            params![
                template.id.to_string(),
                template.name,
                template.description,
                template.tier.as_str(),
                template.scope_namespace,
                scope_tags_json,
                allow_kinds_json,
                allow_actions_json,
                template.updated_at.to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    pub async fn get_permission_template(&self, id: Uuid) -> MvResult<Option<PermissionTemplate>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, tier, scope_namespace, scope_tags_json, allow_kinds_json, allow_actions_json, created_at, updated_at FROM permission_templates WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let row = stmt
            .query_row(params![id.to_string()], |row| {
                row_to_permission_template(row)
            })
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(row)
    }

    pub async fn get_permission_template_by_name(
        &self,
        name: &str,
    ) -> MvResult<Option<PermissionTemplate>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, tier, scope_namespace, scope_tags_json, allow_kinds_json, allow_actions_json, created_at, updated_at FROM permission_templates WHERE lower(name) = lower(?1)",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let row = stmt
            .query_row(params![name], |row| row_to_permission_template(row))
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(row)
    }

    pub async fn list_permission_templates(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<PermissionTemplate>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, tier, scope_namespace, scope_tags_json, allow_kinds_json, allow_actions_json, created_at, updated_at FROM permission_templates ORDER BY created_at DESC LIMIT ?1 OFFSET ?2",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit as i64, offset as i64], |row| {
                row_to_permission_template(row)
            })
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut templates = Vec::new();
        for row in rows {
            templates.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }

        Ok(templates)
    }

    pub async fn delete_permission_template(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let affected = conn
            .execute(
                "DELETE FROM permission_templates WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(affected > 0)
    }

    pub async fn insert_access_key(&self, key: &AccessKey) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        conn.execute(
            "INSERT INTO access_keys (id, name, template_id, key_hash, created_at, last_used_at, expires_at, revoked_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                key.id.to_string(),
                key.name,
                key.template_id.to_string(),
                key.key_hash,
                key.created_at.to_rfc3339(),
                key.last_used_at.map(|dt| dt.to_rfc3339()),
                key.expires_at.map(|dt| dt.to_rfc3339()),
                key.revoked_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    pub async fn list_access_keys(&self) -> MvResult<Vec<AccessKey>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, name, template_id, key_hash, created_at, last_used_at, expires_at, revoked_at FROM access_keys ORDER BY created_at DESC",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], |row| row_to_access_key(row))
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut keys = Vec::new();
        for row in rows {
            keys.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }

        Ok(keys)
    }

    pub async fn get_access_key(&self, id: Uuid) -> MvResult<Option<AccessKey>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, name, template_id, key_hash, created_at, last_used_at, expires_at, revoked_at FROM access_keys WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let row = stmt
            .query_row(params![id.to_string()], |row| row_to_access_key(row))
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(row)
    }

    pub async fn get_access_key_by_hash(&self, key_hash: &str) -> MvResult<Option<AccessKey>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut stmt = conn
            .prepare(
                "SELECT id, name, template_id, key_hash, created_at, last_used_at, expires_at, revoked_at FROM access_keys WHERE key_hash = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let row = stmt
            .query_row(params![key_hash], |row| row_to_access_key(row))
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(row)
    }

    pub async fn update_access_key_last_used(
        &self,
        id: Uuid,
        when: chrono::DateTime<chrono::Utc>,
    ) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        conn.execute(
            "UPDATE access_keys SET last_used_at = ?2 WHERE id = ?1",
            params![id.to_string(), when.to_rfc3339()],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    pub async fn revoke_access_key(
        &self,
        id: Uuid,
        when: chrono::DateTime<chrono::Utc>,
    ) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let affected = conn
            .execute(
                "UPDATE access_keys SET revoked_at = ?2 WHERE id = ?1",
                params![id.to_string(), when.to_rfc3339()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok(affected > 0)
    }
}

fn row_to_permission_template(row: &rusqlite::Row<'_>) -> rusqlite::Result<PermissionTemplate> {
    let id_str: String = row.get(0)?;
    let name: String = row.get(1)?;
    let description: Option<String> = row.get(2)?;
    let tier_str: String = row.get(3)?;
    let scope_namespace: Option<String> = row.get(4)?;
    let scope_tags_json: Option<String> = row.get(5)?;
    let allow_kinds_json: Option<String> = row.get(6)?;
    let allow_actions_json: Option<String> = row.get(7)?;
    let created_at: String = row.get(8)?;
    let updated_at: String = row.get(9)?;

    let id = parse_uuid_str(0, &id_str)?;
    let tier: PermissionTier = tier_str.parse().map_err(|err: String| {
        rusqlite::Error::FromSqlConversionFailure(
            3,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, err)),
        )
    })?;

    let scope_tags = parse_json_vec::<String>(5, scope_tags_json)?;
    let allow_kinds_raw = parse_json_vec::<String>(6, allow_kinds_json)?;
    let mut allow_kinds = Vec::new();
    for raw in allow_kinds_raw {
        let parsed: NodeKind = raw.parse().map_err(|err: String| {
            rusqlite::Error::FromSqlConversionFailure(
                6,
                Type::Text,
                Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, err)),
            )
        })?;
        allow_kinds.push(parsed);
    }
    let allow_actions = parse_json_vec::<String>(7, allow_actions_json)?;

    Ok(PermissionTemplate {
        id,
        name,
        description,
        tier,
        scope_namespace,
        scope_tags,
        allow_kinds,
        allow_actions,
        created_at: parse_dt_strict(8, &created_at)?,
        updated_at: parse_dt_strict(9, &updated_at)?,
    })
}

fn row_to_access_key(row: &rusqlite::Row<'_>) -> rusqlite::Result<AccessKey> {
    let id_str: String = row.get(0)?;
    let name: Option<String> = row.get(1)?;
    let template_id_str: String = row.get(2)?;
    let key_hash: String = row.get(3)?;
    let created_at: String = row.get(4)?;
    let last_used_at: Option<String> = row.get(5)?;
    let expires_at: Option<String> = row.get(6)?;
    let revoked_at: Option<String> = row.get(7)?;

    Ok(AccessKey {
        id: parse_uuid_str(0, &id_str)?,
        name,
        template_id: parse_uuid_str(2, &template_id_str)?,
        key_hash,
        created_at: parse_dt_strict(4, &created_at)?,
        last_used_at: parse_optional_dt_strict(5, last_used_at)?,
        expires_at: parse_optional_dt_strict(6, expires_at)?,
        revoked_at: parse_optional_dt_strict(7, revoked_at)?,
    })
}

fn parse_json_vec<T: serde::de::DeserializeOwned>(
    column: usize,
    raw: Option<String>,
) -> rusqlite::Result<Vec<T>> {
    match raw {
        Some(value) => serde_json::from_str(&value).map_err(|err| {
            rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(err))
        }),
        None => Ok(Vec::new()),
    }
}

// ---------------------------------------------------------------------------
// AgenticStore Implementation
// ---------------------------------------------------------------------------

use hx_core::{
    AgenticStore, CapturedIntent, ChronicleEntry, InsightType, IntentStatus, IntentType,
    ProactiveInsight,
};

#[async_trait]
impl AgenticStore for SqliteNodeStore {
    async fn log_intent(&self, intent: &CapturedIntent) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let params_json = serde_json::to_string(&intent.parameters)?;

        conn.execute(
            "INSERT INTO captured_intents (id, node_id, intent_type, confidence, parameters, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                intent.id.to_string(),
                intent.node_id.to_string(),
                intent.intent_type.as_str(),
                intent.confidence as f64,
                params_json,
                intent.status.as_str(),
                intent.created_at.to_rfc3339(),
                intent.updated_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert intent failed: {e}")))?;
        Ok(())
    }

    async fn get_intent(&self, id: Uuid) -> MvResult<Option<CapturedIntent>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, node_id, intent_type, confidence, parameters, status, created_at, updated_at
                 FROM captured_intents WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], row_to_captured_intent)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_intents(
        &self,
        node_id: Option<Uuid>,
        status: Option<IntentStatus>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<CapturedIntent>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut sql = String::from(
            "SELECT id, node_id, intent_type, confidence, parameters, status, created_at, updated_at
             FROM captured_intents WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(nid) = node_id {
            sql.push_str(&format!(" AND node_id = ?{param_idx}"));
            param_values.push(Box::new(nid.to_string()));
            param_idx += 1;
        }

        if let Some(st) = status {
            sql.push_str(&format!(" AND status = ?{param_idx}"));
            param_values.push(Box::new(st.as_str().to_string()));
            param_idx += 1;
        }

        sql.push_str(&format!(
            " ORDER BY created_at DESC LIMIT ?{param_idx} OFFSET ?{}",
            param_idx + 1
        ));
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_captured_intent)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut intents = Vec::new();
        for row in rows {
            intents.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(intents)
    }

    async fn update_intent_status(&self, id: Uuid, status: IntentStatus) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
            .execute(
                "UPDATE captured_intents SET status = ?2, updated_at = ?3 WHERE id = ?1",
                params![id.to_string(), status.as_str(), now],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn log_insight(&self, insight: &ProactiveInsight) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let related_ids_json = serde_json::to_string(&insight.related_node_ids)?;

        conn.execute(
            "INSERT INTO proactive_insights (id, title, content, insight_type, related_node_ids, importance, created_at, dismissed_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                insight.id.to_string(),
                insight.title,
                insight.content,
                insight.insight_type.as_str(),
                related_ids_json,
                insight.importance as f64,
                insight.created_at.to_rfc3339(),
                insight.dismissed_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert insight failed: {e}")))?;
        Ok(())
    }

    async fn list_insights(&self, limit: usize, offset: usize) -> MvResult<Vec<ProactiveInsight>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, title, content, insight_type, related_node_ids, importance, created_at, dismissed_at
                 FROM proactive_insights
                 WHERE dismissed_at IS NULL
                 ORDER BY created_at DESC
                 LIMIT ?1 OFFSET ?2",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(
                params![limit as i64, offset as i64],
                row_to_proactive_insight,
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut insights = Vec::new();
        for row in rows {
            insights.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(insights)
    }

    async fn delete_insight(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
            .execute(
                "UPDATE proactive_insights SET dismissed_at = ?2 WHERE id = ?1 AND dismissed_at IS NULL",
                params![id.to_string(), now],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn log_chronicle(&self, entry: &ChronicleEntry) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        conn.execute(
            "INSERT INTO chronicle_entries (id, node_id, step_name, logic, input_snapshot, output_snapshot, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                entry.id.to_string(),
                entry.node_id.map(|id| id.to_string()),
                entry.step_name,
                entry.logic,
                entry.input_snapshot,
                entry.output_snapshot,
                entry.timestamp.to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert chronicle failed: {e}")))?;
        Ok(())
    }

    async fn list_chronicles(
        &self,
        node_id: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ChronicleEntry>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let (sql, params_box): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(nid) =
            node_id
        {
            (
                "SELECT id, node_id, step_name, logic, input_snapshot, output_snapshot, timestamp
                 FROM chronicle_entries WHERE node_id = ?1 ORDER BY timestamp DESC LIMIT ?2 OFFSET ?3".to_string(),
                vec![
                    Box::new(nid.to_string()),
                    Box::new(limit as i64),
                    Box::new(offset as i64),
                ],
            )
        } else {
            (
                "SELECT id, node_id, step_name, logic, input_snapshot, output_snapshot, timestamp
                 FROM chronicle_entries ORDER BY timestamp DESC LIMIT ?1 OFFSET ?2"
                    .to_string(),
                vec![Box::new(limit as i64), Box::new(offset as i64)],
            )
        };

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();
        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_chronicle_entry)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(entries)
    }
}

fn row_to_captured_intent(row: &rusqlite::Row<'_>) -> rusqlite::Result<CapturedIntent> {
    let id_str: String = row.get(0)?;
    let node_id_str: String = row.get(1)?;
    let intent_type_str: String = row.get(2)?;
    let confidence: f64 = row.get(3)?;
    let params_json: Option<String> = row.get(4)?;
    let status_str: String = row.get(5)?;
    let created_at: String = row.get(6)?;
    let updated_at: Option<String> = row.get(7)?;

    let id = parse_uuid_str(0, &id_str)?;
    let node_id = parse_uuid_str(1, &node_id_str)?;
    let intent_type: IntentType = intent_type_str
        .parse()
        .unwrap_or(IntentType::Custom(intent_type_str));
    let status: IntentStatus = status_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            5,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;
    let parameters: serde_json::Value = params_json
        .map(|s| serde_json::from_str(&s).unwrap_or(serde_json::Value::Null))
        .unwrap_or(serde_json::Value::Null);

    Ok(CapturedIntent {
        id,
        node_id,
        intent_type,
        confidence: confidence as f32,
        parameters,
        status,
        created_at: parse_dt_strict(6, &created_at)?,
        updated_at: parse_optional_dt_strict(7, updated_at)?,
    })
}

fn row_to_proactive_insight(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProactiveInsight> {
    let id_str: String = row.get(0)?;
    let title: String = row.get(1)?;
    let content: String = row.get(2)?;
    let insight_type_str: String = row.get(3)?;
    let related_ids_json: Option<String> = row.get(4)?;
    let importance: f64 = row.get(5)?;
    let created_at: String = row.get(6)?;
    let dismissed_at: Option<String> = row.get(7)?;

    let id = parse_uuid_str(0, &id_str)?;
    let insight_type: InsightType = insight_type_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            3,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;

    let related_node_ids: Vec<Uuid> = related_ids_json
        .map(|s| {
            let strs: Vec<String> = serde_json::from_str(&s).unwrap_or_default();
            strs.iter()
                .filter_map(|id_str| Uuid::parse_str(id_str).ok())
                .collect()
        })
        .unwrap_or_default();

    Ok(ProactiveInsight {
        id,
        title,
        content,
        insight_type,
        related_node_ids,
        importance: importance as f32,
        created_at: parse_dt_strict(6, &created_at)?,
        dismissed_at: parse_optional_dt_strict(7, dismissed_at)?,
    })
}

fn row_to_chronicle_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChronicleEntry> {
    let id_str: String = row.get(0)?;
    let node_id_str: Option<String> = row.get(1)?;
    let step_name: String = row.get(2)?;
    let logic: String = row.get(3)?;
    let input_snapshot: Option<String> = row.get(4)?;
    let output_snapshot: Option<String> = row.get(5)?;
    let timestamp: String = row.get(6)?;

    let id = parse_uuid_str(0, &id_str)?;
    let node_id = node_id_str.map(|s| Uuid::parse_str(&s).ok()).flatten();

    Ok(ChronicleEntry {
        id,
        node_id,
        step_name,
        logic,
        input_snapshot,
        output_snapshot,
        timestamp: parse_dt_strict(6, &timestamp)?,
    })
}

// ---------------------------------------------------------------------------
// ExchangeStore Implementation
// ---------------------------------------------------------------------------

use chrono::DateTime;
use hx_core::{ExchangeStore, Proposal, ProposalAction, ProposalSender, ProposalState};

fn row_to_proposal(row: &rusqlite::Row<'_>) -> rusqlite::Result<Proposal> {
    let id_str: String = row.get(0)?;
    let node_id_str: Option<String> = row.get(1)?;
    let target_node_id_str: Option<String> = row.get(2)?;
    let sender_str: String = row.get(3)?;
    let action_str: String = row.get(4)?;
    let state_str: String = row.get(5)?;
    let confidence: f64 = row.get(6)?;
    let diff_preview: Option<String> = row.get(7)?;
    let payload_json: Option<String> = row.get(8)?;
    let created_at: String = row.get(9)?;
    let updated_at: Option<String> = row.get(10)?;
    let resolved_at: Option<String> = row.get(11)?;

    let id = parse_uuid_str(0, &id_str)?;
    let node_id = node_id_str.map(|s| Uuid::parse_str(&s).ok()).flatten();
    let target_node_id = target_node_id_str
        .map(|s| Uuid::parse_str(&s).ok())
        .flatten();

    let sender: ProposalSender = sender_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            3,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;
    let action: ProposalAction = action_str
        .parse()
        .unwrap_or(ProposalAction::Custom(action_str));
    let state: ProposalState = state_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            5,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;

    let payload: std::collections::HashMap<String, serde_json::Value> = payload_json
        .map(|s| serde_json::from_str(&s).unwrap_or_default())
        .unwrap_or_default();

    Ok(Proposal {
        id,
        node_id,
        target_node_id,
        sender,
        action,
        state,
        confidence: confidence as f32,
        diff_preview,
        payload,
        created_at: parse_dt_strict(9, &created_at)?,
        updated_at: parse_optional_dt_strict(10, updated_at)?,
        resolved_at: parse_optional_dt_strict(11, resolved_at)?,
    })
}

#[async_trait]
impl ExchangeStore for SqliteNodeStore {
    async fn submit_proposal(&self, proposal: &Proposal) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let payload_json = serde_json::to_string(&proposal.payload)?;

        conn.execute(
            "INSERT INTO proposals (id, node_id, target_node_id, sender, action, state, confidence, diff_preview, payload, created_at, updated_at, resolved_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                proposal.id.to_string(),
                proposal.node_id.map(|id| id.to_string()),
                proposal.target_node_id.map(|id| id.to_string()),
                proposal.sender.as_str(),
                proposal.action.as_str(),
                proposal.state.as_str(),
                proposal.confidence as f64,
                proposal.diff_preview,
                payload_json,
                proposal.created_at.to_rfc3339(),
                proposal.updated_at.map(|dt| dt.to_rfc3339()),
                proposal.resolved_at.map(|dt| dt.to_rfc3339()),
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert proposal failed: {e}")))?;
        Ok(())
    }

    async fn get_proposal(&self, id: Uuid) -> MvResult<Option<Proposal>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, node_id, target_node_id, sender, action, state, confidence, diff_preview, payload, created_at, updated_at, resolved_at
                 FROM proposals WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], row_to_proposal)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_proposals(
        &self,
        state: Option<ProposalState>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<Proposal>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut sql = String::from(
            "SELECT id, node_id, target_node_id, sender, action, state, confidence, diff_preview, payload, created_at, updated_at, resolved_at
             FROM proposals WHERE 1=1",
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(st) = state {
            sql.push_str(&format!(" AND state = ?{param_idx}"));
            param_values.push(Box::new(st.as_str().to_string()));
            param_idx += 1;
        }

        sql.push_str(&format!(
            " ORDER BY created_at DESC LIMIT ?{param_idx} OFFSET ?{}",
            param_idx + 1
        ));
        param_values.push(Box::new(limit as i64));
        param_values.push(Box::new(offset as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_proposal)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut proposals = Vec::new();
        for row in rows {
            proposals.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(proposals)
    }

    async fn resolve_proposal(&self, id: Uuid, state: ProposalState) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
            .execute(
                "UPDATE proposals SET state = ?2, updated_at = ?3, resolved_at = ?3 WHERE id = ?1",
                params![id.to_string(), state.as_str(), now],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn count_proposals(&self, state: Option<ProposalState>) -> MvResult<usize> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let (sql, params_box): (String, Vec<Box<dyn rusqlite::types::ToSql>>) =
            if let Some(st) = state {
                (
                    "SELECT COUNT(*) FROM proposals WHERE state = ?1".to_string(),
                    vec![Box::new(st.as_str().to_string())],
                )
            } else {
                ("SELECT COUNT(*) FROM proposals".to_string(), vec![])
            };

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();

        let count: usize = conn
            .query_row(&sql, params_refs.as_slice(), |row| row.get(0))
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(count)
    }

    async fn expire_proposals(&self, before: DateTime<Utc>) -> MvResult<usize> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
            .execute(
                "UPDATE proposals SET state = 'expired', updated_at = ?1, resolved_at = ?1 WHERE state = 'pending' AND created_at < ?2",
                params![now, before.to_rfc3339()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected)
    }
}

// ---------------------------------------------------------------------------
// SafeguardStore Implementation
// ---------------------------------------------------------------------------

use hx_core::{AutoApproveRule, BlockedSender, SafeguardStore, UndoSnapshot};

fn row_to_blocked_sender(row: &rusqlite::Row<'_>) -> rusqlite::Result<BlockedSender> {
    let id_str: String = row.get(0)?;
    let sender_type: String = row.get(1)?;
    let sender_pattern: String = row.get(2)?;
    let reason: Option<String> = row.get(3)?;
    let blocked_at: String = row.get(4)?;
    let expires_at: Option<String> = row.get(5)?;

    let id = parse_uuid_str(0, &id_str)?;

    Ok(BlockedSender {
        id,
        sender_type,
        sender_pattern,
        reason,
        blocked_at: parse_dt_strict(4, &blocked_at)?,
        expires_at: parse_optional_dt_strict(5, expires_at)?,
    })
}

fn row_to_auto_approve_rule(row: &rusqlite::Row<'_>) -> rusqlite::Result<AutoApproveRule> {
    let id_str: String = row.get(0)?;
    let name: String = row.get(1)?;
    let sender_pattern: Option<String> = row.get(2)?;
    let action_types_csv: Option<String> = row.get(3)?;
    let min_confidence: f64 = row.get(4)?;
    let enabled: bool = row.get(5)?;
    let created_at: String = row.get(6)?;
    let updated_at: Option<String> = row.get(7)?;

    let id = parse_uuid_str(0, &id_str)?;
    let action_types: Vec<String> = action_types_csv
        .map(|csv| {
            csv.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    Ok(AutoApproveRule {
        id,
        name,
        sender_pattern,
        action_types,
        min_confidence: min_confidence as f32,
        enabled,
        created_at: parse_dt_strict(6, &created_at)?,
        updated_at: parse_optional_dt_strict(7, updated_at)?,
    })
}

fn row_to_undo_snapshot(row: &rusqlite::Row<'_>) -> rusqlite::Result<UndoSnapshot> {
    let id_str: String = row.get(0)?;
    let proposal_id_str: String = row.get(1)?;
    let snapshot_data_str: String = row.get(2)?;
    let created_at: String = row.get(3)?;
    let expires_at: String = row.get(4)?;
    let used: bool = row.get(5)?;

    let id = parse_uuid_str(0, &id_str)?;
    let proposal_id = parse_uuid_str(1, &proposal_id_str)?;
    let snapshot_data: serde_json::Value =
        serde_json::from_str(&snapshot_data_str).map_err(|e| {
            rusqlite::Error::FromSqlConversionFailure(
                2,
                Type::Text,
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    e.to_string(),
                )),
            )
        })?;

    Ok(UndoSnapshot {
        id,
        proposal_id,
        snapshot_data,
        created_at: parse_dt_strict(3, &created_at)?,
        expires_at: parse_dt_strict(4, &expires_at)?,
        used,
    })
}

/// Simple glob matching: supports `*` (match all), `prefix*`, and exact match.
fn safeguard_glob_match(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if pattern.ends_with('*') {
        value.starts_with(&pattern[..pattern.len() - 1])
    } else {
        pattern == value
    }
}

#[async_trait]
impl SafeguardStore for SqliteNodeStore {
    async fn add_blocked_sender(&self, sender: &BlockedSender) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
			"INSERT INTO blocked_senders (id, sender_type, sender_pattern, reason, blocked_at, expires_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
			params![
				sender.id.to_string(),
				sender.sender_type,
				sender.sender_pattern,
				sender.reason,
				sender.blocked_at.to_rfc3339(),
				sender.expires_at.map(|dt| dt.to_rfc3339()),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert blocked_sender failed: {e}")))?;
        Ok(())
    }

    async fn remove_blocked_sender(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let affected = conn
            .execute(
                "DELETE FROM blocked_senders WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn list_blocked_senders(&self) -> MvResult<Vec<BlockedSender>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, sender_type, sender_pattern, reason, blocked_at, expires_at
				 FROM blocked_senders ORDER BY blocked_at DESC",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], row_to_blocked_sender)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn is_sender_blocked(&self, sender_type: &str, sender_name: &str) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();

        // Fetch all active rules for this sender_type (not expired)
        let mut stmt = conn
            .prepare(
                "SELECT id, sender_type, sender_pattern, reason, blocked_at, expires_at
				 FROM blocked_senders
				 WHERE sender_type = ?1 AND (expires_at IS NULL OR expires_at > ?2)",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![sender_type, now], row_to_blocked_sender)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        for row in rows {
            let blocked = row.map_err(|e| HxError::Storage(e.to_string()))?;
            if safeguard_glob_match(&blocked.sender_pattern, sender_name) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn add_auto_approve_rule(&self, rule: &AutoApproveRule) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let action_types_csv = if rule.action_types.is_empty() {
            None
        } else {
            Some(rule.action_types.join(","))
        };

        conn.execute(
			"INSERT INTO auto_approve_rules (id, name, sender_pattern, action_types, min_confidence, enabled, created_at, updated_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
			params![
				rule.id.to_string(),
				rule.name,
				rule.sender_pattern,
				action_types_csv,
				rule.min_confidence as f64,
				rule.enabled,
				rule.created_at.to_rfc3339(),
				rule.updated_at.map(|dt| dt.to_rfc3339()),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert auto_approve_rule failed: {e}")))?;
        Ok(())
    }

    async fn remove_auto_approve_rule(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let affected = conn
            .execute(
                "DELETE FROM auto_approve_rules WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn list_auto_approve_rules(&self) -> MvResult<Vec<AutoApproveRule>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, name, sender_pattern, action_types, min_confidence, enabled, created_at, updated_at
				 FROM auto_approve_rules ORDER BY created_at DESC",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], row_to_auto_approve_rule)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn update_auto_approve_rule(&self, rule: &AutoApproveRule) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let action_types_csv = if rule.action_types.is_empty() {
            None
        } else {
            Some(rule.action_types.join(","))
        };
        let now = Utc::now().to_rfc3339();

        let affected = conn
			.execute(
				"UPDATE auto_approve_rules SET name = ?2, sender_pattern = ?3, action_types = ?4, min_confidence = ?5, enabled = ?6, updated_at = ?7 WHERE id = ?1",
				params![
					rule.id.to_string(),
					rule.name,
					rule.sender_pattern,
					action_types_csv,
					rule.min_confidence as f64,
					rule.enabled,
					now,
				],
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn save_undo_snapshot(&self, snapshot: &UndoSnapshot) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let snapshot_json = serde_json::to_string(&snapshot.snapshot_data)?;

        conn.execute(
			"INSERT INTO proposal_undo_snapshots (id, proposal_id, snapshot_data, created_at, expires_at, used)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
			params![
				snapshot.id.to_string(),
				snapshot.proposal_id.to_string(),
				snapshot_json,
				snapshot.created_at.to_rfc3339(),
				snapshot.expires_at.to_rfc3339(),
				snapshot.used,
			],
		)
		.map_err(|e| HxError::Storage(format!("insert undo_snapshot failed: {e}")))?;
        Ok(())
    }

    async fn get_undo_snapshot(&self, proposal_id: Uuid) -> MvResult<Option<UndoSnapshot>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, proposal_id, snapshot_data, created_at, expires_at, used
				 FROM proposal_undo_snapshots WHERE proposal_id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![proposal_id.to_string()], row_to_undo_snapshot)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn mark_undo_used(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let affected = conn
            .execute(
                "UPDATE proposal_undo_snapshots SET used = 1 WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn cleanup_expired_snapshots(&self) -> MvResult<usize> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
            .execute(
                "DELETE FROM proposal_undo_snapshots WHERE expires_at < ?1",
                params![now],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected)
    }
}

// ---------------------------------------------------------------------------
// FeedbackStore – Agent Feedback / Reflection
// ---------------------------------------------------------------------------

fn row_to_agent_feedback(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentFeedback> {
    let id_str: String = row.get(0)?;
    let intent_id_str: Option<String> = row.get(1)?;
    let intent_type: String = row.get(2)?;
    let action: String = row.get(3)?;
    let confidence_at_time: Option<f64> = row.get(4)?;
    let user_edit_delta: Option<f64> = row.get(5)?;
    let response_time_ms: Option<i64> = row.get(6)?;
    let created_at: String = row.get(7)?;

    let id = parse_uuid_str(0, &id_str)?;
    let intent_id = intent_id_str.and_then(|s| Uuid::parse_str(&s).ok());
    let created = chrono::DateTime::parse_from_rfc3339(&created_at)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    Ok(AgentFeedback {
        id,
        intent_id,
        intent_type,
        action,
        confidence_at_time: confidence_at_time.map(|v| v as f32),
        user_edit_delta: user_edit_delta.map(|v| v as f32),
        response_time_ms: response_time_ms.map(|v| v as u64),
        created_at: created,
    })
}

fn row_to_confidence_override(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConfidenceOverride> {
    let intent_type: String = row.get(0)?;
    let base_adjustment: f64 = row.get(1)?;
    let auto_apply_threshold: f64 = row.get(2)?;
    let suppress_below: f64 = row.get(3)?;
    let updated_at: String = row.get(4)?;

    let updated = chrono::DateTime::parse_from_rfc3339(&updated_at)
        .map(|dt| dt.with_timezone(&Utc))
        .unwrap_or_else(|_| Utc::now());

    Ok(ConfidenceOverride {
        intent_type,
        base_adjustment: base_adjustment as f32,
        auto_apply_threshold: auto_apply_threshold as f32,
        suppress_below: suppress_below as f32,
        updated_at: updated,
    })
}

#[async_trait]
impl FeedbackStore for SqliteNodeStore {
    async fn record_feedback(&self, fb: &AgentFeedback) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
			"INSERT INTO agent_feedback (id, intent_id, intent_type, action, confidence_at_time, user_edit_delta, response_time_ms, created_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
			params![
				fb.id.to_string(),
				fb.intent_id.map(|id| id.to_string()),
				fb.intent_type,
				fb.action,
				fb.confidence_at_time.map(|v| v as f64),
				fb.user_edit_delta.map(|v| v as f64),
				fb.response_time_ms.map(|v| v as i64),
				fb.created_at.to_rfc3339(),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert agent_feedback failed: {e}")))?;
        Ok(())
    }

    async fn list_feedback(
        &self,
        intent_type: Option<&str>,
        limit: usize,
    ) -> MvResult<Vec<AgentFeedback>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut sql = String::from(
			"SELECT id, intent_id, intent_type, action, confidence_at_time, user_edit_delta, response_time_ms, created_at
			 FROM agent_feedback WHERE 1=1",
		);
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        let mut param_idx = 1;

        if let Some(it) = intent_type {
            sql.push_str(&format!(" AND intent_type = ?{param_idx}"));
            param_values.push(Box::new(it.to_string()));
            param_idx += 1;
        }

        sql.push_str(&format!(" ORDER BY created_at DESC LIMIT ?{param_idx}"));
        param_values.push(Box::new(limit as i64));

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_agent_feedback)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(results)
    }

    async fn get_acceptance_rate(&self, intent_type: &str) -> MvResult<(usize, usize)> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let total: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM agent_feedback WHERE intent_type = ?1",
                params![intent_type],
                |row| row.get(0),
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let applied: usize = conn
            .query_row(
                "SELECT COUNT(*) FROM agent_feedback WHERE intent_type = ?1 AND action = 'applied'",
                params![intent_type],
                |row| row.get(0),
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        Ok((applied, total))
    }

    async fn set_confidence_override(&self, override_: &ConfidenceOverride) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
			"INSERT OR REPLACE INTO agent_confidence_overrides (intent_type, base_adjustment, auto_apply_threshold, suppress_below, updated_at)
			 VALUES (?1, ?2, ?3, ?4, ?5)",
			params![
				override_.intent_type,
				override_.base_adjustment as f64,
				override_.auto_apply_threshold as f64,
				override_.suppress_below as f64,
				override_.updated_at.to_rfc3339(),
			],
		)
		.map_err(|e| HxError::Storage(format!("upsert confidence_override failed: {e}")))?;
        Ok(())
    }

    async fn get_confidence_override(
        &self,
        intent_type: &str,
    ) -> MvResult<Option<ConfidenceOverride>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT intent_type, base_adjustment, auto_apply_threshold, suppress_below, updated_at
				 FROM agent_confidence_overrides WHERE intent_type = ?1",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![intent_type], row_to_confidence_override)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_confidence_overrides(&self) -> MvResult<Vec<ConfidenceOverride>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT intent_type, base_adjustment, auto_apply_threshold, suppress_below, updated_at
				 FROM agent_confidence_overrides ORDER BY intent_type",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], row_to_confidence_override)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// ProfileStore – Owner Profile
// ---------------------------------------------------------------------------

use hx_core::{OwnerProfile, ProfileStore, UpdateProfileRequest};

#[async_trait]
impl ProfileStore for SqliteNodeStore {
    async fn get_profile(&self) -> MvResult<OwnerProfile> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT display_name, avatar_url, bio, email, preferred_namespace,
                            default_node_kind, preferred_llm_provider, timezone,
                            signature_name, signature_public_key, metadata,
                            created_at, updated_at
                     FROM owner_profile WHERE id = 'owner'",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;

            let profile = stmt
                .query_row([], |row| {
                    let metadata_str: String = row.get(10)?;
                    let metadata: std::collections::HashMap<String, serde_json::Value> =
                        serde_json::from_str(&metadata_str).unwrap_or_default();
                    let created_str: String = row.get(11)?;
                    let updated_str: String = row.get(12)?;
                    Ok(OwnerProfile {
                        display_name: row.get(0)?,
                        avatar_url: row.get(1)?,
                        bio: row.get(2)?,
                        email: row.get(3)?,
                        preferred_namespace: row.get(4)?,
                        default_node_kind: row.get(5)?,
                        preferred_llm_provider: row.get(6)?,
                        timezone: row.get(7)?,
                        signature_name: row.get(8)?,
                        signature_public_key: row.get(9)?,
                        metadata,
                        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                            .map(|d| d.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                        updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                            .map(|d| d.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                    })
                })
                .map_err(|e| HxError::Storage(e.to_string()))?;

            Ok(profile)
        })
    }

    async fn update_profile(&self, req: &UpdateProfileRequest) -> MvResult<OwnerProfile> {
        self.with_conn(|conn| {
            let mut sets: Vec<&str> = Vec::new();
            let mut values: Vec<String> = Vec::new();

            if let Some(ref v) = req.display_name {
                sets.push("display_name = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.avatar_url {
                sets.push("avatar_url = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.bio {
                sets.push("bio = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.email {
                sets.push("email = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.preferred_namespace {
                sets.push("preferred_namespace = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.default_node_kind {
                sets.push("default_node_kind = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.preferred_llm_provider {
                sets.push("preferred_llm_provider = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.timezone {
                sets.push("timezone = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.signature_name {
                sets.push("signature_name = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.signature_public_key {
                sets.push("signature_public_key = ?");
                values.push(v.clone());
            }
            if let Some(ref v) = req.metadata {
                sets.push("metadata = ?");
                values.push(serde_json::to_string(v).unwrap_or_default());
            }

            if !sets.is_empty() {
                sets.push("updated_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')");
                let sql = format!(
                    "UPDATE owner_profile SET {} WHERE id = 'owner'",
                    sets.join(", ")
                );
                let params: Vec<&dyn rusqlite::types::ToSql> = values
                    .iter()
                    .map(|v| v as &dyn rusqlite::types::ToSql)
                    .collect();
                conn.execute(&sql, params.as_slice())
                    .map_err(|e| HxError::Storage(e.to_string()))?;
            }
            Ok(())
        })?;

        self.get_profile().await
    }
}

// ---------------------------------------------------------------------------
// AutonomyStore – Autonomy & Precision Controls (Phase 3.1)
// ---------------------------------------------------------------------------

use hx_core::{AutonomyActionLog, AutonomyDecision, AutonomyRule, AutonomyStore};

fn row_to_autonomy_rule(row: &rusqlite::Row<'_>) -> rusqlite::Result<AutonomyRule> {
    let id_str: String = row.get(0)?;
    let rule_type: String = row.get(1)?;
    let scope_key: Option<String> = row.get(2)?;
    let auto_apply_threshold: f64 = row.get(3)?;
    let max_actions_per_hour: i64 = row.get(4)?;
    let allowed_csv: Option<String> = row.get(5)?;
    let blocked_csv: Option<String> = row.get(6)?;
    let quiet_hours_start: Option<String> = row.get(7)?;
    let quiet_hours_end: Option<String> = row.get(8)?;
    let quiet_hours_timezone: String = row.get(9)?;
    let enabled: bool = row.get(10)?;
    let created_at: String = row.get(11)?;
    let updated_at: Option<String> = row.get(12)?;

    let id = parse_uuid_str(0, &id_str)?;

    let allowed_intent_types: Vec<String> = allowed_csv
        .map(|csv| {
            csv.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let blocked_intent_types: Vec<String> = blocked_csv
        .map(|csv| {
            csv.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();

    Ok(AutonomyRule {
        id,
        rule_type,
        scope_key,
        auto_apply_threshold: auto_apply_threshold as f32,
        max_actions_per_hour: max_actions_per_hour as u32,
        allowed_intent_types,
        blocked_intent_types,
        quiet_hours_start,
        quiet_hours_end,
        quiet_hours_timezone,
        enabled,
        created_at: parse_dt_strict(11, &created_at)?,
        updated_at: parse_optional_dt_strict(12, updated_at)?,
    })
}

fn row_to_autonomy_action_log(row: &rusqlite::Row<'_>) -> rusqlite::Result<AutonomyActionLog> {
    let id_str: String = row.get(0)?;
    let rule_id_str: Option<String> = row.get(1)?;
    let intent_type: String = row.get(2)?;
    let decision_str: String = row.get(3)?;
    let confidence: Option<f64> = row.get(4)?;
    let reason: Option<String> = row.get(5)?;
    let created_at: String = row.get(6)?;

    let id = parse_uuid_str(0, &id_str)?;
    let rule_id = rule_id_str.and_then(|s| Uuid::parse_str(&s).ok());
    let decision: AutonomyDecision = decision_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            3,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;

    Ok(AutonomyActionLog {
        id,
        rule_id,
        intent_type,
        decision,
        confidence: confidence.map(|v| v as f32),
        reason,
        created_at: parse_dt_strict(6, &created_at)?,
    })
}

#[async_trait]
impl AutonomyStore for SqliteNodeStore {
    async fn add_autonomy_rule(&self, rule: &AutonomyRule) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let allowed_csv = if rule.allowed_intent_types.is_empty() {
            None
        } else {
            Some(rule.allowed_intent_types.join(","))
        };
        let blocked_csv = if rule.blocked_intent_types.is_empty() {
            None
        } else {
            Some(rule.blocked_intent_types.join(","))
        };

        conn.execute(
			"INSERT INTO autonomy_rules (id, rule_type, scope_key, auto_apply_threshold, max_actions_per_hour, allowed_intent_types, blocked_intent_types, quiet_hours_start, quiet_hours_end, quiet_hours_timezone, enabled, created_at, updated_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
			params![
				rule.id.to_string(),
				rule.rule_type,
				rule.scope_key,
				rule.auto_apply_threshold as f64,
				rule.max_actions_per_hour as i64,
				allowed_csv,
				blocked_csv,
				rule.quiet_hours_start,
				rule.quiet_hours_end,
				rule.quiet_hours_timezone,
				rule.enabled,
				rule.created_at.to_rfc3339(),
				rule.updated_at.map(|dt| dt.to_rfc3339()),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert autonomy_rule failed: {e}")))?;
        Ok(())
    }

    async fn get_autonomy_rule(&self, id: Uuid) -> MvResult<Option<AutonomyRule>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, rule_type, scope_key, auto_apply_threshold, max_actions_per_hour, allowed_intent_types, blocked_intent_types, quiet_hours_start, quiet_hours_end, quiet_hours_timezone, enabled, created_at, updated_at
				 FROM autonomy_rules WHERE id = ?1",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], row_to_autonomy_rule)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_autonomy_rules(&self) -> MvResult<Vec<AutonomyRule>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, rule_type, scope_key, auto_apply_threshold, max_actions_per_hour, allowed_intent_types, blocked_intent_types, quiet_hours_start, quiet_hours_end, quiet_hours_timezone, enabled, created_at, updated_at
				 FROM autonomy_rules ORDER BY created_at DESC",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], row_to_autonomy_rule)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn update_autonomy_rule(&self, rule: &AutonomyRule) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let allowed_csv = if rule.allowed_intent_types.is_empty() {
            None
        } else {
            Some(rule.allowed_intent_types.join(","))
        };
        let blocked_csv = if rule.blocked_intent_types.is_empty() {
            None
        } else {
            Some(rule.blocked_intent_types.join(","))
        };
        let now = Utc::now().to_rfc3339();

        let affected = conn
			.execute(
				"UPDATE autonomy_rules SET rule_type = ?2, scope_key = ?3, auto_apply_threshold = ?4, max_actions_per_hour = ?5, allowed_intent_types = ?6, blocked_intent_types = ?7, quiet_hours_start = ?8, quiet_hours_end = ?9, quiet_hours_timezone = ?10, enabled = ?11, updated_at = ?12 WHERE id = ?1",
				params![
					rule.id.to_string(),
					rule.rule_type,
					rule.scope_key,
					rule.auto_apply_threshold as f64,
					rule.max_actions_per_hour as i64,
					allowed_csv,
					blocked_csv,
					rule.quiet_hours_start,
					rule.quiet_hours_end,
					rule.quiet_hours_timezone,
					rule.enabled,
					now,
				],
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn delete_autonomy_rule(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let affected = conn
            .execute(
                "DELETE FROM autonomy_rules WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn log_autonomy_action(&self, log: &AutonomyActionLog) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
			"INSERT INTO autonomy_action_log (id, rule_id, intent_type, decision, confidence, reason, created_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
			params![
				log.id.to_string(),
				log.rule_id.map(|id| id.to_string()),
				log.intent_type,
				log.decision.to_string(),
				log.confidence.map(|v| v as f64),
				log.reason,
				log.created_at.to_rfc3339(),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert autonomy_action_log failed: {e}")))?;
        Ok(())
    }

    async fn count_recent_actions(
        &self,
        rule_id: Option<Uuid>,
        since: DateTime<Utc>,
    ) -> MvResult<usize> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let (sql, params_box): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(rid) =
            rule_id
        {
            (
				"SELECT COUNT(*) FROM autonomy_action_log WHERE rule_id = ?1 AND decision = 'auto_apply' AND created_at >= ?2".to_string(),
				vec![Box::new(rid.to_string()), Box::new(since.to_rfc3339())],
			)
        } else {
            (
				"SELECT COUNT(*) FROM autonomy_action_log WHERE decision = 'auto_apply' AND created_at >= ?1".to_string(),
				vec![Box::new(since.to_rfc3339())],
			)
        };

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();

        let count: usize = conn
            .query_row(&sql, params_refs.as_slice(), |row| row.get(0))
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(count)
    }

    async fn list_autonomy_action_log(&self, limit: usize) -> MvResult<Vec<AutonomyActionLog>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, rule_id, intent_type, decision, confidence, reason, created_at
				 FROM autonomy_action_log ORDER BY created_at DESC LIMIT ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(params![limit as i64], row_to_autonomy_action_log)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// RelayStore – Communication Relay Network (Phase 3.2)
// ---------------------------------------------------------------------------

use hx_core::{
    ChannelType, ContentType, MessageDirection, MessageStatus, RelayChannel, RelayContact,
    RelayMessage, RelayStore, TrustLevel,
};

fn row_to_relay_contact(row: &rusqlite::Row<'_>) -> rusqlite::Result<RelayContact> {
    let id_str: String = row.get(0)?;
    let display_name: String = row.get(1)?;
    let public_key: String = row.get(2)?;
    let vault_address: Option<String> = row.get(3)?;
    let trust_level_str: String = row.get(4)?;
    let autonomy_rule_id_str: Option<String> = row.get(5)?;
    let notes: Option<String> = row.get(6)?;
    let created_at_str: String = row.get(7)?;
    let updated_at_str: Option<String> = row.get(8)?;

    let id = parse_uuid_str(0, &id_str)?;
    let trust_level: TrustLevel = trust_level_str.parse().unwrap_or(TrustLevel::RelayOnly);
    let autonomy_rule_id = autonomy_rule_id_str.and_then(|s| Uuid::parse_str(&s).ok());
    let created_at = parse_dt_strict(7, &created_at_str)?;
    let updated_at = parse_optional_dt_strict(8, updated_at_str)?;

    Ok(RelayContact {
        id,
        display_name,
        public_key,
        vault_address,
        trust_level,
        autonomy_rule_id,
        notes,
        created_at,
        updated_at,
    })
}

fn row_to_relay_channel(row: &rusqlite::Row<'_>) -> rusqlite::Result<RelayChannel> {
    let id_str: String = row.get(0)?;
    let name: Option<String> = row.get(1)?;
    let channel_type_str: String = row.get(2)?;
    let member_ids_json: String = row.get(3)?;
    let created_at_str: String = row.get(4)?;
    let updated_at_str: Option<String> = row.get(5)?;

    let id = parse_uuid_str(0, &id_str)?;
    let channel_type: ChannelType = channel_type_str.parse().unwrap_or(ChannelType::Direct);
    let member_contact_ids: Vec<Uuid> = serde_json::from_str(&member_ids_json).unwrap_or_default();
    let created_at = parse_dt_strict(4, &created_at_str)?;
    let updated_at = parse_optional_dt_strict(5, updated_at_str)?;

    Ok(RelayChannel {
        id,
        name,
        channel_type,
        member_contact_ids,
        created_at,
        updated_at,
    })
}

fn row_to_relay_message(row: &rusqlite::Row<'_>) -> rusqlite::Result<RelayMessage> {
    let id_str: String = row.get(0)?;
    let channel_id_str: String = row.get(1)?;
    let thread_id_str: Option<String> = row.get(2)?;
    let sender_contact_id_str: Option<String> = row.get(3)?;
    let recipient_contact_id_str: Option<String> = row.get(4)?;
    let direction_str: String = row.get(5)?;
    let content: String = row.get(6)?;
    let content_type_str: String = row.get(7)?;
    let status_str: String = row.get(8)?;
    let vault_node_id_str: Option<String> = row.get(9)?;
    let metadata_json: String = row.get(10)?;
    let created_at_str: String = row.get(11)?;
    let updated_at_str: Option<String> = row.get(12)?;

    let id = parse_uuid_str(0, &id_str)?;
    let channel_id = parse_uuid_str(1, &channel_id_str)?;
    let thread_id = thread_id_str.and_then(|s| Uuid::parse_str(&s).ok());
    let sender_contact_id = sender_contact_id_str.and_then(|s| Uuid::parse_str(&s).ok());
    let recipient_contact_id = recipient_contact_id_str.and_then(|s| Uuid::parse_str(&s).ok());
    let direction: MessageDirection = direction_str.parse().unwrap_or(MessageDirection::Outbound);
    let content_type: ContentType = content_type_str.parse().unwrap_or(ContentType::Text);
    let status: MessageStatus = status_str.parse().unwrap_or(MessageStatus::Pending);
    let vault_node_id = vault_node_id_str.and_then(|s| Uuid::parse_str(&s).ok());
    let metadata: std::collections::HashMap<String, serde_json::Value> =
        serde_json::from_str(&metadata_json).unwrap_or_default();
    let created_at = parse_dt_strict(11, &created_at_str)?;
    let updated_at = parse_optional_dt_strict(12, updated_at_str)?;

    Ok(RelayMessage {
        id,
        channel_id,
        thread_id,
        sender_contact_id,
        recipient_contact_id,
        direction,
        content,
        content_type,
        status,
        vault_node_id,
        metadata,
        created_at,
        updated_at,
    })
}

#[async_trait]
impl RelayStore for SqliteNodeStore {
    // ── Contacts ──────────────────────────────────────────────────────────

    async fn add_relay_contact(&self, contact: &RelayContact) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
			"INSERT INTO relay_contacts (id, display_name, public_key, vault_address, trust_level, autonomy_rule_id, notes, created_at, updated_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
			params![
				contact.id.to_string(),
				contact.display_name,
				contact.public_key,
				contact.vault_address,
				contact.trust_level.to_string(),
				contact.autonomy_rule_id.map(|id| id.to_string()),
				contact.notes,
				contact.created_at.to_rfc3339(),
				contact.updated_at.map(|dt| dt.to_rfc3339()),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert relay_contact failed: {e}")))?;
        Ok(())
    }

    async fn get_relay_contact(&self, id: Uuid) -> MvResult<Option<RelayContact>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, display_name, public_key, vault_address, trust_level, autonomy_rule_id, notes, created_at, updated_at
				 FROM relay_contacts WHERE id = ?1",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], row_to_relay_contact)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_relay_contacts(&self) -> MvResult<Vec<RelayContact>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, display_name, public_key, vault_address, trust_level, autonomy_rule_id, notes, created_at, updated_at
				 FROM relay_contacts ORDER BY created_at DESC",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], row_to_relay_contact)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn update_relay_contact(&self, contact: &RelayContact) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
			.execute(
				"UPDATE relay_contacts SET display_name = ?2, public_key = ?3, vault_address = ?4, trust_level = ?5, autonomy_rule_id = ?6, notes = ?7, updated_at = ?8 WHERE id = ?1",
				params![
					contact.id.to_string(),
					contact.display_name,
					contact.public_key,
					contact.vault_address,
					contact.trust_level.to_string(),
					contact.autonomy_rule_id.map(|id| id.to_string()),
					contact.notes,
					now,
				],
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn delete_relay_contact(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let affected = conn
            .execute(
                "DELETE FROM relay_contacts WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    // ── Channels ──────────────────────────────────────────────────────────

    async fn add_relay_channel(&self, channel: &RelayChannel) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let member_ids_json = serde_json::to_string(&channel.member_contact_ids)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
			"INSERT INTO relay_channels (id, name, channel_type, member_contact_ids, created_at, updated_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
			params![
				channel.id.to_string(),
				channel.name,
				channel.channel_type.to_string(),
				member_ids_json,
				channel.created_at.to_rfc3339(),
				channel.updated_at.map(|dt| dt.to_rfc3339()),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert relay_channel failed: {e}")))?;
        Ok(())
    }

    async fn get_relay_channel(&self, id: Uuid) -> MvResult<Option<RelayChannel>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, channel_type, member_contact_ids, created_at, updated_at
				 FROM relay_channels WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], row_to_relay_channel)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_relay_channels(&self) -> MvResult<Vec<RelayChannel>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, channel_type, member_contact_ids, created_at, updated_at
				 FROM relay_channels ORDER BY created_at DESC",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map([], row_to_relay_channel)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn delete_relay_channel(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let affected = conn
            .execute(
                "DELETE FROM relay_channels WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    // ── Messages ──────────────────────────────────────────────────────────

    async fn add_relay_message(&self, message: &RelayMessage) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let metadata_json = serde_json::to_string(&message.metadata)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
			"INSERT INTO relay_messages (id, channel_id, thread_id, sender_contact_id, recipient_contact_id, direction, content, content_type, status, vault_node_id, metadata, created_at, updated_at)
			 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
			params![
				message.id.to_string(),
				message.channel_id.to_string(),
				message.thread_id.map(|id| id.to_string()),
				message.sender_contact_id.map(|id| id.to_string()),
				message.recipient_contact_id.map(|id| id.to_string()),
				message.direction.to_string(),
				message.content,
				message.content_type.to_string(),
				message.status.to_string(),
				message.vault_node_id.map(|id| id.to_string()),
				metadata_json,
				message.created_at.to_rfc3339(),
				message.updated_at.map(|dt| dt.to_rfc3339()),
			],
		)
		.map_err(|e| HxError::Storage(format!("insert relay_message failed: {e}")))?;
        Ok(())
    }

    async fn get_relay_message(&self, id: Uuid) -> MvResult<Option<RelayMessage>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, channel_id, thread_id, sender_contact_id, recipient_contact_id, direction, content, content_type, status, vault_node_id, metadata, created_at, updated_at
				 FROM relay_messages WHERE id = ?1",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let result = stmt
            .query_row(params![id.to_string()], row_to_relay_message)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_relay_messages(
        &self,
        channel_id: Uuid,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<RelayMessage>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, channel_id, thread_id, sender_contact_id, recipient_contact_id, direction, content, content_type, status, vault_node_id, metadata, created_at, updated_at
				 FROM relay_messages WHERE channel_id = ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(
                params![channel_id.to_string(), limit as i64, offset as i64],
                row_to_relay_message,
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn update_message_status(&self, id: Uuid, status: MessageStatus) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
            .execute(
                "UPDATE relay_messages SET status = ?2, updated_at = ?3 WHERE id = ?1",
                params![id.to_string(), status.to_string(), now],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn list_thread_messages(
        &self,
        thread_id: Uuid,
        limit: usize,
    ) -> MvResult<Vec<RelayMessage>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
			.prepare(
				"SELECT id, channel_id, thread_id, sender_contact_id, recipient_contact_id, direction, content, content_type, status, vault_node_id, metadata, created_at, updated_at
				 FROM relay_messages WHERE thread_id = ?1 ORDER BY created_at ASC LIMIT ?2",
			)
			.map_err(|e| HxError::Storage(e.to_string()))?;

        let rows = stmt
            .query_map(
                params![thread_id.to_string(), limit as i64],
                row_to_relay_message,
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn count_unread_messages(&self, channel_id: Option<Uuid>) -> MvResult<usize> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let (sql, params_box): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(cid) =
            channel_id
        {
            (
				"SELECT COUNT(*) FROM relay_messages WHERE status NOT IN ('read') AND direction = 'inbound' AND channel_id = ?1".to_string(),
				vec![Box::new(cid.to_string())],
			)
        } else {
            (
				"SELECT COUNT(*) FROM relay_messages WHERE status NOT IN ('read') AND direction = 'inbound'".to_string(),
				vec![],
			)
        };

        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();

        let count: usize = conn
            .query_row(&sql, params_refs.as_slice(), |row| row.get(0))
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(count)
    }
}

// ---------------------------------------------------------------------------
// ConsumerStore – Consumer Profiles for AI identity
// ---------------------------------------------------------------------------

use hx_core::{ConsumerProfile, ConsumerStore};

#[async_trait]
impl ConsumerStore for SqliteNodeStore {
    async fn create_consumer(&self, profile: &ConsumerProfile) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let metadata_json = serde_json::to_string(&profile.metadata)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO consumer_profiles (id, name, description, token_hash, created_at, last_used_at, revoked_at, metadata_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                profile.id.to_string(),
                profile.name,
                profile.description,
                profile.token_hash,
                profile.created_at.to_rfc3339(),
                profile.last_used_at.map(|dt| dt.to_rfc3339()),
                profile.revoked_at.map(|dt| dt.to_rfc3339()),
                metadata_json,
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert consumer_profile failed: {e}")))?;
        Ok(())
    }

    async fn get_consumer(&self, id: Uuid) -> MvResult<Option<ConsumerProfile>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, token_hash, created_at, last_used_at, revoked_at, metadata_json
                 FROM consumer_profiles WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = stmt
            .query_row(params![id.to_string()], row_to_consumer)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn get_consumer_by_name(&self, name: &str) -> MvResult<Option<ConsumerProfile>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, token_hash, created_at, last_used_at, revoked_at, metadata_json
                 FROM consumer_profiles WHERE name = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = stmt
            .query_row(params![name], row_to_consumer)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn get_consumer_by_token_hash(
        &self,
        token_hash: &str,
    ) -> MvResult<Option<ConsumerProfile>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, token_hash, created_at, last_used_at, revoked_at, metadata_json
                 FROM consumer_profiles WHERE token_hash = ?1 AND revoked_at IS NULL",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = stmt
            .query_row(params![token_hash], row_to_consumer)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_consumers(&self) -> MvResult<Vec<ConsumerProfile>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, name, description, token_hash, created_at, last_used_at, revoked_at, metadata_json
                 FROM consumer_profiles ORDER BY created_at DESC",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([], row_to_consumer)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn revoke_consumer(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        let affected = conn
            .execute(
                "UPDATE consumer_profiles SET revoked_at = ?2 WHERE id = ?1 AND revoked_at IS NULL",
                params![id.to_string(), now],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn touch_consumer(&self, id: Uuid) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = Utc::now().to_rfc3339();
        conn.execute(
            "UPDATE consumer_profiles SET last_used_at = ?2 WHERE id = ?1",
            params![id.to_string(), now],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }
}

fn row_to_consumer(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConsumerProfile> {
    let id_str: String = row.get(0)?;
    let name: String = row.get(1)?;
    let description: Option<String> = row.get(2)?;
    let token_hash: String = row.get(3)?;
    let created_at_str: String = row.get(4)?;
    let last_used_at_str: Option<String> = row.get(5)?;
    let revoked_at_str: Option<String> = row.get(6)?;
    let metadata_json: Option<String> = row.get(7)?;

    let id = parse_uuid_str(0, &id_str)?;
    let metadata = metadata_json
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();

    Ok(ConsumerProfile {
        id,
        name,
        description,
        token_hash,
        created_at: parse_dt_strict(4, &created_at_str)?,
        last_used_at: parse_optional_dt_strict(5, last_used_at_str)?,
        revoked_at: parse_optional_dt_strict(6, revoked_at_str)?,
        metadata,
    })
}

// ---------------------------------------------------------------------------
// PolicyStore – Access Policies (ABAC with default-deny)
// ---------------------------------------------------------------------------

use hx_core::{AccessPolicy, PolicyStore};

#[async_trait]
impl PolicyStore for SqliteNodeStore {
    async fn set_policy(&self, policy: &AccessPolicy) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let scopes_json =
            serde_json::to_string(&policy.scopes).map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT OR REPLACE INTO access_policies (id, secret_key, consumer, allowed, scopes_json, max_ttl_seconds, expires_at, require_approval, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                policy.id.to_string(),
                policy.secret_key,
                policy.consumer,
                policy.allowed as i32,
                scopes_json,
                policy.max_ttl_seconds,
                policy.expires_at.map(|dt| dt.to_rfc3339()),
                policy.require_approval as i32,
                policy.created_at.to_rfc3339(),
                policy.updated_at.to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Storage(format!("upsert access_policy failed: {e}")))?;
        Ok(())
    }

    async fn get_policy(&self, id: Uuid) -> MvResult<Option<AccessPolicy>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, secret_key, consumer, allowed, scopes_json, max_ttl_seconds, expires_at, require_approval, created_at, updated_at
                 FROM access_policies WHERE id = ?1",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = stmt
            .query_row(params![id.to_string()], row_to_policy)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn get_policy_for(
        &self,
        secret_key: &str,
        consumer: &str,
    ) -> MvResult<Option<AccessPolicy>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, secret_key, consumer, allowed, scopes_json, max_ttl_seconds, expires_at, require_approval, created_at, updated_at
                 FROM access_policies WHERE secret_key = ?1 AND consumer = ?2",
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = stmt
            .query_row(params![secret_key, consumer], row_to_policy)
            .optional()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(result)
    }

    async fn list_policies(
        &self,
        secret_key: Option<&str>,
        consumer: Option<&str>,
    ) -> MvResult<Vec<AccessPolicy>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let mut sql = "SELECT id, secret_key, consumer, allowed, scopes_json, max_ttl_seconds, expires_at, require_approval, created_at, updated_at FROM access_policies".to_string();
        let mut conditions: Vec<String> = Vec::new();
        let mut params_box: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(sk) = secret_key {
            conditions.push(format!("secret_key = ?{}", params_box.len() + 1));
            params_box.push(Box::new(sk.to_string()));
        }
        if let Some(c) = consumer {
            conditions.push(format!("consumer = ?{}", params_box.len() + 1));
            params_box.push(Box::new(c.to_string()));
        }

        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        sql.push_str(" ORDER BY created_at DESC");

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_policy)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn delete_policy(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let affected = conn
            .execute(
                "DELETE FROM access_policies WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }
}

#[async_trait]
impl ShareStore for SqliteNodeStore {
    async fn insert_public_share(&self, share: &PublicShare) -> MvResult<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO public_shares (id, node_id, token_hash, created_at, expires_at, revoked_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    share.id.to_string(),
                    share.node_id.to_string(),
                    share.token_hash,
                    share.created_at.to_rfc3339(),
                    share.expires_at.map(|dt| dt.to_rfc3339()),
                    share.revoked_at.map(|dt| dt.to_rfc3339()),
                ],
            )
            .map_err(|e| HxError::Storage(format!("insert public_share failed: {e}")))?;
            Ok(())
        })
    }

    async fn get_public_share(&self, id: Uuid) -> MvResult<Option<PublicShare>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, node_id, token_hash, created_at, expires_at, revoked_at
                     FROM public_shares WHERE id = ?1",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let share = stmt
                .query_row(params![id.to_string()], row_to_public_share)
                .optional()
                .map_err(|e| HxError::Storage(format!("select public_share failed: {e}")))?;
            Ok(share)
        })
    }

    async fn get_public_share_by_hash(&self, token_hash: &str) -> MvResult<Option<PublicShare>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, node_id, token_hash, created_at, expires_at, revoked_at
                     FROM public_shares WHERE token_hash = ?1",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let share = stmt
                .query_row(params![token_hash], row_to_public_share)
                .optional()
                .map_err(|e| HxError::Storage(format!("select public_share failed: {e}")))?;
            Ok(share)
        })
    }

    async fn list_public_shares(
        &self,
        node_id: Option<Uuid>,
        include_revoked: bool,
    ) -> MvResult<Vec<PublicShare>> {
        self.with_conn(|conn| {
            let mut sql =
                "SELECT id, node_id, token_hash, created_at, expires_at, revoked_at FROM public_shares"
                    .to_string();
            let mut params_refs: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
            let mut conditions: Vec<String> = Vec::new();

            if let Some(node_id) = node_id {
                conditions.push(format!("node_id = ?{}", params_refs.len() + 1));
                params_refs.push(Box::new(node_id.to_string()));
            }

            if !include_revoked {
                conditions.push("revoked_at IS NULL".to_string());
            }

            if !conditions.is_empty() {
                sql.push_str(" WHERE ");
                sql.push_str(&conditions.join(" AND "));
            }

            sql.push_str(" ORDER BY created_at DESC");

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| HxError::Storage(e.to_string()))?;

            let param_slice: Vec<&dyn rusqlite::ToSql> =
                params_refs.iter().map(|p| p.as_ref() as &dyn rusqlite::ToSql).collect();
            let shares = stmt
                .query_map(param_slice.as_slice(), row_to_public_share)
                .map_err(|e| HxError::Storage(format!("list public_shares failed: {e}")))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| HxError::Storage(format!("collect public_shares failed: {e}")))?;

            Ok(shares)
        })
    }

    async fn revoke_public_share(&self, id: Uuid, revoked_at: DateTime<Utc>) -> MvResult<bool> {
        self.with_conn(|conn| {
            let updated = conn
                .execute(
                    "UPDATE public_shares SET revoked_at = ?2 WHERE id = ?1",
                    params![id.to_string(), revoked_at.to_rfc3339()],
                )
                .map_err(|e| HxError::Storage(format!("revoke public_share failed: {e}")))?;
            Ok(updated > 0)
        })
    }
}

#[async_trait]
impl CommentStore for SqliteNodeStore {
    async fn insert_comment(&self, comment: &NodeComment) -> MvResult<()> {
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO node_comments (id, node_id, author, body, created_at, updated_at, resolved_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    comment.id.to_string(),
                    comment.node_id.to_string(),
                    comment.author,
                    comment.body,
                    comment.created_at.to_rfc3339(),
                    comment.updated_at.to_rfc3339(),
                    comment.resolved_at.map(|dt| dt.to_rfc3339()),
                ],
            )
            .map_err(|e| HxError::Storage(format!("insert comment failed: {e}")))?;
            Ok(())
        })
    }

    async fn get_comment(&self, id: Uuid) -> MvResult<Option<NodeComment>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, node_id, author, body, created_at, updated_at, resolved_at
                     FROM node_comments WHERE id = ?1",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let comment = stmt
                .query_row(params![id.to_string()], row_to_node_comment)
                .optional()
                .map_err(|e| HxError::Storage(format!("select comment failed: {e}")))?;
            Ok(comment)
        })
    }

    async fn list_comments(
        &self,
        node_id: Uuid,
        include_resolved: bool,
    ) -> MvResult<Vec<NodeComment>> {
        self.with_conn(|conn| {
            let mut sql = String::from(
                "SELECT id, node_id, author, body, created_at, updated_at, resolved_at
                 FROM node_comments WHERE node_id = ?1",
            );
            if !include_resolved {
                sql.push_str(" AND resolved_at IS NULL");
            }
            sql.push_str(" ORDER BY created_at DESC");

            let mut stmt = conn
                .prepare(&sql)
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let comments = stmt
                .query_map(params![node_id.to_string()], row_to_node_comment)
                .map_err(|e| HxError::Storage(format!("list comments failed: {e}")))?
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| HxError::Storage(format!("collect comments failed: {e}")))?;
            Ok(comments)
        })
    }

    async fn resolve_comment(&self, id: Uuid, resolved_at: DateTime<Utc>) -> MvResult<bool> {
        self.with_conn(|conn| {
            let updated = conn
                .execute(
                    "UPDATE node_comments SET resolved_at = ?2, updated_at = ?2 WHERE id = ?1",
                    params![id.to_string(), resolved_at.to_rfc3339()],
                )
                .map_err(|e| HxError::Storage(format!("resolve comment failed: {e}")))?;
            Ok(updated > 0)
        })
    }

    async fn delete_comment(&self, id: Uuid) -> MvResult<bool> {
        self.with_conn(|conn| {
            let affected = conn
                .execute(
                    "DELETE FROM node_comments WHERE id = ?1",
                    params![id.to_string()],
                )
                .map_err(|e| HxError::Storage(format!("delete comment failed: {e}")))?;
            Ok(affected > 0)
        })
    }
}

#[async_trait]
impl McpConnectorStore for SqliteNodeStore {
    async fn insert_mcp_connector(&self, connector: &McpConnector) -> MvResult<()> {
        self.with_conn(|conn| {
            let config_schema_json = serde_json::to_string(&connector.config_schema)?;
            let capabilities_json = serde_json::to_string(&connector.capabilities)?;
            conn.execute(
                "INSERT INTO mcp_connectors (id, name, description, publisher, version, homepage_url, repository_url, config_schema, capabilities_json, verified, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    connector.id.to_string(),
                    connector.name,
                    connector.description,
                    connector.publisher,
                    connector.version,
                    connector.homepage_url,
                    connector.repository_url,
                    config_schema_json,
                    capabilities_json,
                    if connector.verified { 1 } else { 0 },
                    connector.created_at.to_rfc3339(),
                    connector.updated_at.to_rfc3339(),
                ],
            )
            .map_err(|e| HxError::Storage(format!("insert mcp connector failed: {e}")))?;
            Ok(())
        })
    }

    async fn get_mcp_connector(&self, id: Uuid) -> MvResult<Option<McpConnector>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, name, description, publisher, version, homepage_url, repository_url, config_schema, capabilities_json, verified, created_at, updated_at
                     FROM mcp_connectors WHERE id = ?1",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let connector = stmt
                .query_row(params![id.to_string()], row_to_mcp_connector)
                .optional()
                .map_err(|e| HxError::Storage(format!("select mcp connector failed: {e}")))?;
            Ok(connector)
        })
    }

    async fn list_mcp_connectors(
        &self,
        publisher: Option<&str>,
        verified: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<McpConnector>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut sql = String::from(
            "SELECT id, name, description, publisher, version, homepage_url, repository_url, config_schema, capabilities_json, verified, created_at, updated_at
             FROM mcp_connectors WHERE 1=1",
        );
        let mut params_box: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

        if let Some(publisher) = publisher {
            sql.push_str(&format!(" AND publisher = ?{}", params_box.len() + 1));
            params_box.push(Box::new(publisher.to_string()));
        }
        if let Some(verified) = verified {
            sql.push_str(&format!(" AND verified = ?{}", params_box.len() + 1));
            params_box.push(Box::new(if verified { 1 } else { 0 }));
        }

        sql.push_str(&format!(
            " ORDER BY name ASC LIMIT ?{} OFFSET ?{}",
            params_box.len() + 1,
            params_box.len() + 2
        ));
        params_box.push(Box::new(limit as i64));
        params_box.push(Box::new(offset as i64));

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_mcp_connector)
            .map_err(|e| HxError::Storage(format!("list mcp connectors failed: {e}")))?;
        let mut connectors = Vec::new();
        for row in rows {
            connectors.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(connectors)
    }

    async fn update_mcp_connector(&self, connector: &McpConnector) -> MvResult<bool> {
        self.with_conn(|conn| {
            let config_schema_json = serde_json::to_string(&connector.config_schema)?;
            let capabilities_json = serde_json::to_string(&connector.capabilities)?;
            let updated = conn
                .execute(
                    "UPDATE mcp_connectors
                     SET name = ?2, description = ?3, publisher = ?4, version = ?5, homepage_url = ?6,
                         repository_url = ?7, config_schema = ?8, capabilities_json = ?9, verified = ?10, updated_at = ?11
                     WHERE id = ?1",
                    params![
                        connector.id.to_string(),
                        connector.name,
                        connector.description,
                        connector.publisher,
                        connector.version,
                        connector.homepage_url,
                        connector.repository_url,
                        config_schema_json,
                        capabilities_json,
                        if connector.verified { 1 } else { 0 },
                        connector.updated_at.to_rfc3339(),
                    ],
                )
                .map_err(|e| HxError::Storage(format!("update mcp connector failed: {e}")))?;
            Ok(updated > 0)
        })
    }

    async fn delete_mcp_connector(&self, id: Uuid) -> MvResult<bool> {
        self.with_conn(|conn| {
            let affected = conn
                .execute(
                    "DELETE FROM mcp_connectors WHERE id = ?1",
                    params![id.to_string()],
                )
                .map_err(|e| HxError::Storage(format!("delete mcp connector failed: {e}")))?;
            Ok(affected > 0)
        })
    }
}

fn row_to_policy(row: &rusqlite::Row<'_>) -> rusqlite::Result<AccessPolicy> {
    let id_str: String = row.get(0)?;
    let secret_key: String = row.get(1)?;
    let consumer: String = row.get(2)?;
    let allowed: i32 = row.get(3)?;
    let scopes_json: String = row.get(4)?;
    let max_ttl_seconds: Option<i64> = row.get(5)?;
    let expires_at_str: Option<String> = row.get(6)?;
    let require_approval: i32 = row.get(7)?;
    let created_at_str: String = row.get(8)?;
    let updated_at_str: String = row.get(9)?;

    let id = parse_uuid_str(0, &id_str)?;
    let scopes: Vec<String> = serde_json::from_str(&scopes_json).unwrap_or_default();

    Ok(AccessPolicy {
        id,
        secret_key,
        consumer,
        allowed: allowed != 0,
        scopes,
        max_ttl_seconds,
        expires_at: parse_optional_dt_strict(6, expires_at_str)?,
        require_approval: require_approval != 0,
        created_at: parse_dt_strict(8, &created_at_str)?,
        updated_at: parse_dt_strict(9, &updated_at_str)?,
    })
}

// ---------------------------------------------------------------------------
// ProxyAuditStore – Proxy Audit Log
// ---------------------------------------------------------------------------

use hx_core::{ProxyAuditEntry, ProxyAuditStore};

#[async_trait]
impl ProxyAuditStore for SqliteNodeStore {
    async fn log_proxy_audit(&self, entry: &ProxyAuditEntry) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO proxy_audit_log (id, consumer, secret_ref, action, target, intent, timestamp, success, sanitized, error, request_summary, response_status)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                entry.id.to_string(),
                entry.consumer,
                entry.secret_ref,
                entry.action,
                entry.target,
                entry.intent,
                entry.timestamp.to_rfc3339(),
                entry.success.map(|v| v as i32),
                entry.sanitized as i32,
                entry.error,
                entry.request_summary,
                entry.response_status,
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert proxy_audit_log failed: {e}")))?;
        Ok(())
    }

    async fn update_proxy_audit(
        &self,
        id: Uuid,
        success: bool,
        sanitized: bool,
        error: Option<&str>,
        response_status: Option<i32>,
    ) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "UPDATE proxy_audit_log SET success = ?2, sanitized = ?3, error = ?4, response_status = ?5 WHERE id = ?1",
            params![
                id.to_string(),
                success as i32,
                sanitized as i32,
                error,
                response_status,
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(())
    }

    async fn list_proxy_audit(
        &self,
        consumer: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ProxyAuditEntry>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let (sql, params_box): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(c) =
            consumer
        {
            (
                "SELECT id, consumer, secret_ref, action, target, intent, timestamp, success, sanitized, error, request_summary, response_status
                 FROM proxy_audit_log WHERE consumer = ?1 ORDER BY timestamp DESC LIMIT ?2 OFFSET ?3".to_string(),
                vec![Box::new(c.to_string()), Box::new(limit as i64), Box::new(offset as i64)],
            )
        } else {
            (
                "SELECT id, consumer, secret_ref, action, target, intent, timestamp, success, sanitized, error, request_summary, response_status
                 FROM proxy_audit_log ORDER BY timestamp DESC LIMIT ?1 OFFSET ?2".to_string(),
                vec![Box::new(limit as i64), Box::new(offset as i64)],
            )
        };

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_proxy_audit)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }
}

fn row_to_proxy_audit(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProxyAuditEntry> {
    let id_str: String = row.get(0)?;
    let consumer: String = row.get(1)?;
    let secret_ref: String = row.get(2)?;
    let action: String = row.get(3)?;
    let target: String = row.get(4)?;
    let intent: String = row.get(5)?;
    let timestamp_str: String = row.get(6)?;
    let success: Option<i32> = row.get(7)?;
    let sanitized: i32 = row.get(8)?;
    let error: Option<String> = row.get(9)?;
    let request_summary: String = row.get(10)?;
    let response_status: Option<i32> = row.get(11)?;

    let id = parse_uuid_str(0, &id_str)?;

    Ok(ProxyAuditEntry {
        id,
        consumer,
        secret_ref,
        action,
        target,
        intent,
        timestamp: parse_dt_strict(6, &timestamp_str)?,
        success: success.map(|v| v != 0),
        sanitized: sanitized != 0,
        error,
        request_summary,
        response_status,
    })
}

// ---------------------------------------------------------------------------
// ApprovalStore
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl ApprovalStore for SqliteNodeStore {
    async fn create_approval(&self, request: &ApprovalRequest) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let scopes_json = serde_json::to_string(&request.scopes)
            .map_err(|e| HxError::Storage(format!("serialize scopes: {e}")))?;
        conn.execute(
            "INSERT INTO proxy_approvals (id, consumer, secret_key, intent, request_summary, state, created_at, expires_at, scopes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                request.id.to_string(),
                request.consumer,
                request.secret_key,
                request.intent,
                request.request_summary,
                request.state.as_str(),
                request.created_at.to_rfc3339(),
                request.expires_at.to_rfc3339(),
                scopes_json,
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert proxy_approvals failed: {e}")))?;
        Ok(())
    }

    async fn get_approval(&self, id: Uuid) -> MvResult<Option<ApprovalRequest>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn.prepare(
            "SELECT id, consumer, secret_key, intent, request_summary, state, created_at, expires_at, decided_at, decided_by, deny_reason, scopes
             FROM proxy_approvals WHERE id = ?1"
        ).map_err(|e| HxError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![id.to_string()], row_to_approval)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(a)) => Ok(Some(a)),
            Some(Err(e)) => Err(HxError::Storage(e.to_string())),
            None => Ok(None),
        }
    }

    async fn list_pending_approvals(
        &self,
        consumer: Option<&str>,
    ) -> MvResult<Vec<ApprovalRequest>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;

        let (sql, params_box): (String, Vec<Box<dyn rusqlite::types::ToSql>>) = if let Some(c) =
            consumer
        {
            (
                "SELECT id, consumer, secret_key, intent, request_summary, state, created_at, expires_at, decided_at, decided_by, deny_reason, scopes
                 FROM proxy_approvals WHERE state = 'pending' AND consumer = ?1 ORDER BY created_at DESC".to_string(),
                vec![Box::new(c.to_string())],
            )
        } else {
            (
                "SELECT id, consumer, secret_key, intent, request_summary, state, created_at, expires_at, decided_at, decided_by, deny_reason, scopes
                 FROM proxy_approvals WHERE state = 'pending' ORDER BY created_at DESC".to_string(),
                vec![],
            )
        };

        let mut stmt = conn
            .prepare(&sql)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> =
            params_box.iter().map(|p| p.as_ref()).collect();
        let rows = stmt
            .query_map(params_refs.as_slice(), row_to_approval)
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
        }
        Ok(result)
    }

    async fn decide_approval(
        &self,
        id: Uuid,
        approved: bool,
        decided_by: Option<&str>,
        deny_reason: Option<&str>,
    ) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let new_state = if approved { "approved" } else { "denied" };
        let now = chrono::Utc::now().to_rfc3339();
        let affected = conn.execute(
            "UPDATE proxy_approvals SET state = ?2, decided_at = ?3, decided_by = ?4, deny_reason = ?5
             WHERE id = ?1 AND state = 'pending'",
            params![
                id.to_string(),
                new_state,
                now,
                decided_by,
                deny_reason,
            ],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected > 0)
    }

    async fn expire_approvals(&self) -> MvResult<usize> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = chrono::Utc::now().to_rfc3339();
        let affected = conn.execute(
            "UPDATE proxy_approvals SET state = 'expired' WHERE state = 'pending' AND expires_at <= ?1",
            params![now],
        )
        .map_err(|e| HxError::Storage(e.to_string()))?;
        Ok(affected)
    }

    async fn find_active_approval(
        &self,
        consumer: &str,
        secret_key: &str,
    ) -> MvResult<Option<ApprovalRequest>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let now = chrono::Utc::now().to_rfc3339();
        let mut stmt = conn.prepare(
            "SELECT id, consumer, secret_key, intent, request_summary, state, created_at, expires_at, decided_at, decided_by, deny_reason, scopes
             FROM proxy_approvals
             WHERE consumer = ?1 AND secret_key = ?2 AND state = 'approved' AND expires_at > ?3
             ORDER BY decided_at DESC LIMIT 1"
        ).map_err(|e| HxError::Storage(e.to_string()))?;

        let mut rows = stmt
            .query_map(params![consumer, secret_key, now], row_to_approval)
            .map_err(|e| HxError::Storage(e.to_string()))?;

        match rows.next() {
            Some(Ok(a)) => Ok(Some(a)),
            Some(Err(e)) => Err(HxError::Storage(e.to_string())),
            None => Ok(None),
        }
    }
}

fn row_to_approval(row: &rusqlite::Row<'_>) -> rusqlite::Result<ApprovalRequest> {
    let id_str: String = row.get(0)?;
    let consumer: String = row.get(1)?;
    let secret_key: String = row.get(2)?;
    let intent: String = row.get(3)?;
    let request_summary: String = row.get(4)?;
    let state_str: String = row.get(5)?;
    let created_at_str: String = row.get(6)?;
    let expires_at_str: String = row.get(7)?;
    let decided_at_str: Option<String> = row.get(8)?;
    let decided_by: Option<String> = row.get(9)?;
    let deny_reason: Option<String> = row.get(10)?;
    let scopes_json: String = row.get(11)?;

    let id = parse_uuid_str(0, &id_str)?;
    let state: ApprovalState = state_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            5,
            rusqlite::types::Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;
    let scopes: Vec<String> = serde_json::from_str(&scopes_json).unwrap_or_default();

    Ok(ApprovalRequest {
        id,
        consumer,
        secret_key,
        intent,
        request_summary,
        state,
        created_at: parse_dt_strict(6, &created_at_str)?,
        expires_at: parse_dt_strict(7, &expires_at_str)?,
        decided_at: decided_at_str.as_deref().and_then(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .ok()
                .map(|d| d.with_timezone(&chrono::Utc))
        }),
        decided_by,
        deny_reason,
        scopes,
    })
}

// ---------------------------------------------------------------------------
// ConflictStore
// ---------------------------------------------------------------------------

#[async_trait]
impl ConflictStore for SqliteNodeStore {
    async fn insert_conflict(&self, alert: &ConflictAlert) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT OR IGNORE INTO conflicts (id, node_a, node_b, conflict_type, score, explanation, resolved, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                alert.id.to_string(),
                alert.node_a.to_string(),
                alert.node_b.to_string(),
                alert.conflict_type.as_str(),
                alert.score,
                alert.explanation,
                alert.resolved as i32,
                alert.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert conflict: {e}")))?;
        Ok(())
    }

    async fn get_conflict(&self, id: Uuid) -> MvResult<Option<ConflictAlert>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = conn
            .query_row(
                "SELECT id, node_a, node_b, conflict_type, score, explanation, resolved, created_at
                 FROM conflicts WHERE id = ?1",
                params![id.to_string()],
                row_to_conflict,
            )
            .optional()
            .map_err(|e| HxError::Storage(format!("get conflict: {e}")))?;
        Ok(result)
    }

    async fn list_conflicts(
        &self,
        resolved: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ConflictAlert>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let limit_i = limit as i64;
        let offset_i = offset as i64;
        let results = match resolved {
            Some(r) => {
                let resolved_i = r as i32;
                let mut stmt = conn
                    .prepare(
                        "SELECT id, node_a, node_b, conflict_type, score, explanation, resolved, created_at
                         FROM conflicts WHERE resolved = ?1 ORDER BY created_at DESC LIMIT ?2 OFFSET ?3",
                    )
                    .map_err(|e| HxError::Storage(format!("list conflicts prepare: {e}")))?;
                let rows = stmt
                    .query_map(params![resolved_i, limit_i, offset_i], row_to_conflict)
                    .map_err(|e| HxError::Storage(format!("list conflicts query: {e}")))?;
                let mut v = Vec::new();
                for row in rows {
                    v.push(row.map_err(|e| HxError::Storage(format!("list conflicts row: {e}")))?);
                }
                v
            }
            None => {
                let mut stmt = conn
                    .prepare(
                        "SELECT id, node_a, node_b, conflict_type, score, explanation, resolved, created_at
                         FROM conflicts ORDER BY created_at DESC LIMIT ?1 OFFSET ?2",
                    )
                    .map_err(|e| HxError::Storage(format!("list conflicts prepare: {e}")))?;
                let rows = stmt
                    .query_map(params![limit_i, offset_i], row_to_conflict)
                    .map_err(|e| HxError::Storage(format!("list conflicts query: {e}")))?;
                let mut v = Vec::new();
                for row in rows {
                    v.push(row.map_err(|e| HxError::Storage(format!("list conflicts row: {e}")))?);
                }
                v
            }
        };
        Ok(results)
    }

    async fn resolve_conflict(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let updated = conn
            .execute(
                "UPDATE conflicts SET resolved = 1 WHERE id = ?1 AND resolved = 0",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(format!("resolve conflict: {e}")))?;
        Ok(updated > 0)
    }
}

fn row_to_conflict(row: &rusqlite::Row<'_>) -> rusqlite::Result<ConflictAlert> {
    let id_str: String = row.get(0)?;
    let node_a_str: String = row.get(1)?;
    let node_b_str: String = row.get(2)?;
    let conflict_type_str: String = row.get(3)?;
    let score: f64 = row.get(4)?;
    let explanation: String = row.get(5)?;
    let resolved_int: i32 = row.get(6)?;
    let created_at_str: String = row.get(7)?;

    let id = parse_uuid_str(0, &id_str)?;
    let node_a = parse_uuid_str(1, &node_a_str)?;
    let node_b = parse_uuid_str(2, &node_b_str)?;
    let conflict_type: ConflictType = conflict_type_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            3,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;

    Ok(ConflictAlert {
        id,
        node_a,
        node_b,
        conflict_type,
        score,
        explanation,
        resolved: resolved_int != 0,
        created_at: parse_dt_strict(7, &created_at_str)?,
    })
}

// ---------------------------------------------------------------------------
// ContactIdentityStore
// ---------------------------------------------------------------------------

#[async_trait]
impl ContactIdentityStore for SqliteNodeStore {
    async fn add_contact_identity(&self, identity: &ContactIdentity) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        conn.execute(
            "INSERT INTO contact_identities (id, contact_id, identity_type, identity_value, verified, verified_at, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                identity.id.to_string(),
                identity.contact_id.to_string(),
                identity.identity_type.as_str(),
                identity.identity_value,
                identity.verified as i32,
                identity.verified_at.map(|dt| dt.to_rfc3339()),
                identity.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Storage(format!("insert contact identity: {e}")))?;
        Ok(())
    }

    async fn list_contact_identities(&self, contact_id: Uuid) -> MvResult<Vec<ContactIdentity>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let mut stmt = conn
            .prepare(
                "SELECT id, contact_id, identity_type, identity_value, verified, verified_at, created_at
                 FROM contact_identities WHERE contact_id = ?1 ORDER BY created_at DESC",
            )
            .map_err(|e| HxError::Storage(format!("list identities prepare: {e}")))?;
        let rows = stmt
            .query_map(params![contact_id.to_string()], row_to_contact_identity)
            .map_err(|e| HxError::Storage(format!("list identities query: {e}")))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row.map_err(|e| HxError::Storage(format!("list identities row: {e}")))?);
        }
        Ok(results)
    }

    async fn delete_contact_identity(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let deleted = conn
            .execute(
                "DELETE FROM contact_identities WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(format!("delete identity: {e}")))?;
        Ok(deleted > 0)
    }

    async fn verify_contact_identity(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let updated = conn
            .execute(
                "UPDATE contact_identities SET verified = 1, verified_at = strftime('%Y-%m-%dT%H:%M:%SZ','now')
                 WHERE id = ?1 AND verified = 0",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Storage(format!("verify identity: {e}")))?;
        Ok(updated > 0)
    }

    async fn get_trust_model(&self, contact_id: Uuid) -> MvResult<Option<TrustModel>> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let result = conn
            .query_row(
                "SELECT contact_id, can_query, can_inject_context, can_auto_reply, allowed_namespaces, max_confidence_override, updated_at
                 FROM trust_models WHERE contact_id = ?1",
                params![contact_id.to_string()],
                row_to_trust_model,
            )
            .optional()
            .map_err(|e| HxError::Storage(format!("get trust model: {e}")))?;
        Ok(result)
    }

    async fn set_trust_model(&self, model: &TrustModel) -> MvResult<()> {
        let conn = self
            .conn()
            .lock()
            .map_err(|e| HxError::Storage(e.to_string()))?;
        let ns_json = serde_json::to_string(&model.allowed_namespaces)
            .map_err(|e| HxError::Storage(format!("serialize namespaces: {e}")))?;
        conn.execute(
            "INSERT INTO trust_models (contact_id, can_query, can_inject_context, can_auto_reply, allowed_namespaces, max_confidence_override, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, strftime('%Y-%m-%dT%H:%M:%SZ','now'))
             ON CONFLICT(contact_id) DO UPDATE SET
               can_query = excluded.can_query,
               can_inject_context = excluded.can_inject_context,
               can_auto_reply = excluded.can_auto_reply,
               allowed_namespaces = excluded.allowed_namespaces,
               max_confidence_override = excluded.max_confidence_override,
               updated_at = excluded.updated_at",
            params![
                model.contact_id.to_string(),
                model.can_query as i32,
                model.can_inject_context as i32,
                model.can_auto_reply as i32,
                ns_json,
                model.max_confidence_override,
            ],
        )
        .map_err(|e| HxError::Storage(format!("set trust model: {e}")))?;
        Ok(())
    }
}

fn row_to_contact_identity(row: &rusqlite::Row<'_>) -> rusqlite::Result<ContactIdentity> {
    let id_str: String = row.get(0)?;
    let contact_id_str: String = row.get(1)?;
    let identity_type_str: String = row.get(2)?;
    let identity_value: String = row.get(3)?;
    let verified_int: i32 = row.get(4)?;
    let verified_at_str: Option<String> = row.get(5)?;
    let created_at_str: String = row.get(6)?;

    let id = parse_uuid_str(0, &id_str)?;
    let contact_id = parse_uuid_str(1, &contact_id_str)?;
    let identity_type: IdentityType = identity_type_str.parse().map_err(|e: String| {
        rusqlite::Error::FromSqlConversionFailure(
            2,
            Type::Text,
            Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
        )
    })?;

    let verified_at = verified_at_str
        .as_deref()
        .map(|s| parse_dt_strict(5, s))
        .transpose()?;

    Ok(ContactIdentity {
        id,
        contact_id,
        identity_type,
        identity_value,
        verified: verified_int != 0,
        verified_at,
        created_at: parse_dt_strict(6, &created_at_str)?,
    })
}

fn row_to_trust_model(row: &rusqlite::Row<'_>) -> rusqlite::Result<TrustModel> {
    let contact_id_str: String = row.get(0)?;
    let can_query: i32 = row.get(1)?;
    let can_inject_context: i32 = row.get(2)?;
    let can_auto_reply: i32 = row.get(3)?;
    let ns_json: String = row.get(4)?;
    let max_confidence_override: Option<f64> = row.get(5)?;
    let updated_at_str: String = row.get(6)?;

    let contact_id = parse_uuid_str(0, &contact_id_str)?;
    let allowed_namespaces: Vec<String> = serde_json::from_str(&ns_json).unwrap_or_default();

    Ok(TrustModel {
        contact_id,
        can_query: can_query != 0,
        can_inject_context: can_inject_context != 0,
        can_auto_reply: can_auto_reply != 0,
        allowed_namespaces,
        max_confidence_override,
        updated_at: parse_dt_strict(6, &updated_at_str)?,
    })
}

// ---------------------------------------------------------------------------
// AdapterPollStore
// ---------------------------------------------------------------------------

#[async_trait]
impl AdapterPollStore for SqliteNodeStore {
    async fn get_poll_state(&self, adapter_name: &str) -> MvResult<Option<AdapterPollState>> {
        let name = adapter_name.to_string();
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare("SELECT adapter_name, cursor, last_poll_at, messages_received FROM adapter_poll_state WHERE adapter_name = ?1")
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let result = stmt
                .query_row(params![name], |row| {
                    Ok(AdapterPollState {
                        adapter_name: row.get(0)?,
                        cursor: row.get(1)?,
                        last_poll_at: row.get(2)?,
                        messages_received: row.get::<_, i64>(3)? as u64,
                    })
                })
                .optional()
                .map_err(|e| HxError::Storage(e.to_string()))?;
            Ok(result)
        })
    }

    async fn upsert_poll_state(
        &self,
        adapter_name: &str,
        cursor: &str,
        messages_received: u64,
    ) -> MvResult<()> {
        let name = adapter_name.to_string();
        let cursor = cursor.to_string();
        let now = Utc::now().to_rfc3339();
        self.with_conn(|conn| {
            conn.execute(
                "INSERT INTO adapter_poll_state (adapter_name, cursor, last_poll_at, messages_received)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(adapter_name) DO UPDATE SET
                     cursor = excluded.cursor,
                     last_poll_at = excluded.last_poll_at,
                     messages_received = adapter_poll_state.messages_received + excluded.messages_received",
                params![name, cursor, now, messages_received as i64],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
            Ok(())
        })
    }

    async fn list_poll_states(&self) -> MvResult<Vec<AdapterPollState>> {
        self.with_conn(|conn| {
            let mut stmt = conn
                .prepare("SELECT adapter_name, cursor, last_poll_at, messages_received FROM adapter_poll_state ORDER BY adapter_name")
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map([], |row| {
                    Ok(AdapterPollState {
                        adapter_name: row.get(0)?,
                        cursor: row.get(1)?,
                        last_poll_at: row.get(2)?,
                        messages_received: row.get::<_, i64>(3)? as u64,
                    })
                })
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let mut result = Vec::new();
            for row in rows {
                result.push(row.map_err(|e| HxError::Storage(e.to_string()))?);
            }
            Ok(result)
        })
    }

    async fn delete_poll_state(&self, adapter_name: &str) -> MvResult<bool> {
        let name = adapter_name.to_string();
        self.with_conn(|conn| {
            let affected = conn
                .execute(
                    "DELETE FROM adapter_poll_state WHERE adapter_name = ?1",
                    params![name],
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            Ok(affected > 0)
        })
    }
}

// ---------------------------------------------------------------------------
// ConversationStore — Phase 3 WI-3a
// ---------------------------------------------------------------------------

#[async_trait]
impl ConversationStore for SqliteNodeStore {
    async fn create_conversation(&self, id: Uuid, title: Option<&str>) -> MvResult<()> {
        let id_s = id.to_string();
        let title_s = title.map(|t| t.to_string());
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO conversations (id, title) VALUES (?1, ?2)",
                params![id_s, title_s],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
            Ok(())
        })
    }

    async fn add_message(
        &self,
        conversation_id: Uuid,
        role: &str,
        content: &str,
    ) -> MvResult<Uuid> {
        let msg_id = Uuid::now_v7();
        let conv_s = conversation_id.to_string();
        let msg_s = msg_id.to_string();
        let role_s = role.to_string();
        let content_s = content.to_string();
        let token_count = (content_s.len() / 4) as i64; // rough estimate
        self.with_conn(move |conn| {
            conn.execute(
                "INSERT INTO conversation_turns (id, conversation_id, role, content, token_count) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                params![msg_s, conv_s, role_s, content_s, token_count],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
            // Update conversation's updated_at
            conn.execute(
                "UPDATE conversations SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') \
                 WHERE id = ?1",
                params![conv_s],
            )
            .map_err(|e| HxError::Storage(e.to_string()))?;
            Ok(msg_id)
        })
    }

    async fn get_messages(
        &self,
        conversation_id: Uuid,
        limit: usize,
    ) -> MvResult<Vec<(Uuid, String, String, chrono::DateTime<chrono::Utc>)>> {
        let conv_s = conversation_id.to_string();
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, role, content, created_at FROM conversation_turns \
                     WHERE conversation_id = ?1 ORDER BY created_at ASC LIMIT ?2",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![conv_s, limit as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                })
                .map_err(|e| HxError::Storage(e.to_string()))?;

            let mut messages = Vec::new();
            for row in rows {
                let (id_str, role, content, ts_str) =
                    row.map_err(|e| HxError::Storage(e.to_string()))?;
                let id = Uuid::parse_str(&id_str)
                    .map_err(|e| HxError::Storage(format!("invalid uuid: {e}")))?;
                let ts = chrono::DateTime::parse_from_rfc3339(&ts_str)
                    .map_err(|e| HxError::Storage(format!("invalid timestamp: {e}")))?
                    .with_timezone(&chrono::Utc);
                messages.push((id, role, content, ts));
            }
            Ok(messages)
        })
    }

    async fn delete_conversation(&self, id: Uuid) -> MvResult<bool> {
        let id_s = id.to_string();
        self.with_conn(move |conn| {
            let affected = conn
                .execute("DELETE FROM conversations WHERE id = ?1", params![id_s])
                .map_err(|e| HxError::Storage(e.to_string()))?;
            Ok(affected > 0)
        })
    }

    async fn list_conversations(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<(Uuid, Option<String>, chrono::DateTime<chrono::Utc>)>> {
        self.with_conn(move |conn| {
            let mut stmt = conn
                .prepare(
                    "SELECT id, title, updated_at FROM conversations \
                     ORDER BY updated_at DESC LIMIT ?1 OFFSET ?2",
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            let rows = stmt
                .query_map(params![limit as i64, offset as i64], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(|e| HxError::Storage(e.to_string()))?;

            let mut convs = Vec::new();
            for row in rows {
                let (id_str, title, ts_str) = row.map_err(|e| HxError::Storage(e.to_string()))?;
                let id = Uuid::parse_str(&id_str)
                    .map_err(|e| HxError::Storage(format!("invalid uuid: {e}")))?;
                let ts = chrono::DateTime::parse_from_rfc3339(&ts_str)
                    .map_err(|e| HxError::Storage(format!("invalid timestamp: {e}")))?
                    .with_timezone(&chrono::Utc);
                convs.push((id, title, ts));
            }
            Ok(convs)
        })
    }

    async fn expire_conversations(&self, max_age_secs: u64) -> MvResult<usize> {
        self.with_conn(move |conn| {
            let affected = conn
                .execute(
                    "DELETE FROM conversations WHERE updated_at < \
                     strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?1)",
                    params![format!("-{max_age_secs} seconds")],
                )
                .map_err(|e| HxError::Storage(e.to_string()))?;
            Ok(affected)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed_runtime::{
        clear_runtime_root_key, set_runtime_root_key, set_sealed_mode_enabled,
    };
    use tempfile::tempdir;
    use uuid::Uuid;

    struct SealedRuntimeReset;

    impl Drop for SealedRuntimeReset {
        fn drop(&mut self) {
            clear_runtime_root_key();
            set_sealed_mode_enabled(false);
        }
    }

    fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() || haystack.len() < needle.len() {
            return false;
        }
        haystack.windows(needle.len()).any(|window| window == needle)
    }

    #[tokio::test]
    async fn test_insert_and_get() {
        let store = SqliteNodeStore::open_in_memory().unwrap();
        let node = KnowledgeNode::new(NodeKind::Fact, "Rust is fast")
            .with_title("Rust Speed")
            .with_tags(vec!["rust".into(), "performance".into()]);

        let id = node.id;
        store.insert(&node).await.unwrap();

        let retrieved = store.get(id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Rust is fast");
        assert_eq!(retrieved.tags, vec!["performance", "rust"]); // sorted
        assert_eq!(retrieved.kind, NodeKind::Fact);
    }

    #[tokio::test]
    async fn test_sealed_node_payload_persists_encrypted_columns() {
        let _reset = SealedRuntimeReset;
        set_sealed_mode_enabled(true);
        set_runtime_root_key([7u8; 32], false);

        let store = SqliteNodeStore::open_in_memory().unwrap();
        let mut node = KnowledgeNode::new(NodeKind::Fact, "sealed-content")
            .with_title("sealed-title")
            .with_namespace("default");
        node.source = Some("sealed-source".to_string());
        node.metadata.insert("k".into(), serde_json::json!("v"));
        let id = node.id;

        store.insert(&node).await.unwrap();

        let raw = store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT title, content, metadata_json, payload_ciphertext, payload_wrapped_dek FROM knowledge_nodes WHERE id = ?1",
                    params![id.to_string()],
                    |row| {
                        Ok((
                            row.get::<_, Option<String>>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<String>>(4)?,
                        ))
                    },
                )
                .map_err(|e| HxError::Storage(e.to_string()))
            })
            .unwrap();

        assert_eq!(raw.0, None);
        assert_eq!(raw.1, "");
        assert_eq!(raw.2, None);
        assert!(raw.3.is_some());
        assert!(raw.4.is_some());

        let roundtrip = store.get(id).await.unwrap().unwrap();
        assert_eq!(roundtrip.title.as_deref(), Some("sealed-title"));
        assert_eq!(roundtrip.content, "sealed-content");
        assert_eq!(roundtrip.source.as_deref(), Some("sealed-source"));
        assert_eq!(
            roundtrip
                .metadata
                .get("k")
                .and_then(serde_json::Value::as_str),
            Some("v")
        );
    }

    #[tokio::test]
    async fn test_sealed_node_update_refreshes_encrypted_payload() {
        let _reset = SealedRuntimeReset;
        set_sealed_mode_enabled(true);
        set_runtime_root_key([9u8; 32], false);

        let store = SqliteNodeStore::open_in_memory().unwrap();
        let mut node = KnowledgeNode::new(NodeKind::Fact, "v1-content")
            .with_title("v1-title")
            .with_namespace("default");
        node.source = Some("v1-source".to_string());
        node.metadata.insert("ver".into(), serde_json::json!("v1"));
        let id = node.id;
        store.insert(&node).await.unwrap();

        node.content = "v2-content".to_string();
        node.title = Some("v2-title".to_string());
        node.source = Some("v2-source".to_string());
        node.temporal.version += 1;
        node.temporal.updated_at = Utc::now();
        node.metadata.insert("ver".into(), serde_json::json!("v2"));
        store.update(&node).await.unwrap();

        let raw = store
            .with_conn(|conn| {
                conn.query_row(
                    "SELECT title, content, metadata_json, payload_ciphertext, payload_wrapped_dek FROM knowledge_nodes WHERE id = ?1",
                    params![id.to_string()],
                    |row| {
                        Ok((
                            row.get::<_, Option<String>>(0)?,
                            row.get::<_, String>(1)?,
                            row.get::<_, Option<String>>(2)?,
                            row.get::<_, Option<String>>(3)?,
                            row.get::<_, Option<String>>(4)?,
                        ))
                    },
                )
                .map_err(|e| HxError::Storage(e.to_string()))
            })
            .unwrap();
        assert_eq!(raw.0, None);
        assert_eq!(raw.1, "");
        assert_eq!(raw.2, None);
        assert!(raw.3.is_some());
        assert!(raw.4.is_some());

        let roundtrip = store.get(id).await.unwrap().unwrap();
        assert_eq!(roundtrip.title.as_deref(), Some("v2-title"));
        assert_eq!(roundtrip.content, "v2-content");
        assert_eq!(roundtrip.source.as_deref(), Some("v2-source"));
        assert_eq!(
            roundtrip
                .metadata
                .get("ver")
                .and_then(serde_json::Value::as_str),
            Some("v2")
        );
    }

    #[tokio::test]
    async fn test_sealed_sqlite_file_does_not_contain_plaintext_marker() {
        let _reset = SealedRuntimeReset;
        set_sealed_mode_enabled(true);
        set_runtime_root_key([11u8; 32], false);

        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("helix.sqlite");
        let store = SqliteNodeStore::open(&db_path).expect("open sqlite store");
        let marker = format!("sealed-sqlite-marker-{}", Uuid::now_v7());

        let mut node = KnowledgeNode::new(NodeKind::Fact, marker.clone())
            .with_title(marker.clone())
            .with_namespace("default");
        node.source = Some(marker.clone());
        node.metadata
            .insert("marker".into(), serde_json::Value::String(marker.clone()));

        store.insert(&node).await.expect("insert sealed node");

        let bytes = std::fs::read(&db_path).expect("read sqlite file");
        assert!(
            !bytes_contains(&bytes, marker.as_bytes()),
            "sqlite file must not contain plaintext marker"
        );
    }

    #[tokio::test]
    async fn test_update() {
        let store = SqliteNodeStore::open_in_memory().unwrap();
        let mut node = KnowledgeNode::new(NodeKind::Fact, "original");
        let id = node.id;
        store.insert(&node).await.unwrap();

        node.content = "updated".into();
        node.temporal.version = 2;
        node.temporal.updated_at = Utc::now();
        store.update(&node).await.unwrap();

        let retrieved = store.get(id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "updated");
        assert_eq!(retrieved.temporal.version, 2);
    }

    #[tokio::test]
    async fn test_delete() {
        let store = SqliteNodeStore::open_in_memory().unwrap();
        let node = KnowledgeNode::new(NodeKind::Fact, "to delete");
        let id = node.id;
        store.insert(&node).await.unwrap();

        assert!(store.delete(id).await.unwrap());
        assert!(store.get(id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_list_with_filters() {
        let store = SqliteNodeStore::open_in_memory().unwrap();

        let n1 = KnowledgeNode::new(NodeKind::Fact, "fact one")
            .with_namespace("dev")
            .with_tags(vec!["rust".into()]);
        let n2 = KnowledgeNode::new(NodeKind::Decision, "decision one")
            .with_namespace("dev")
            .with_tags(vec!["planning".into()]);
        let n3 = KnowledgeNode::new(NodeKind::Fact, "fact two")
            .with_namespace("personal")
            .with_tags(vec!["rust".into()]);

        store.insert(&n1).await.unwrap();
        store.insert(&n2).await.unwrap();
        store.insert(&n3).await.unwrap();

        let filters = QueryFilters {
            namespace: Some("dev".into()),
            ..Default::default()
        };
        let results = store.list(&filters, 10, 0).await.unwrap();
        assert_eq!(results.len(), 2);

        let filters = QueryFilters {
            kinds: Some(vec![NodeKind::Fact]),
            ..Default::default()
        };
        let results = store.list(&filters, 10, 0).await.unwrap();
        assert_eq!(results.len(), 2);

        let filters = QueryFilters {
            tags: Some(vec!["planning".into()]),
            ..Default::default()
        };
        let results = store.list(&filters, 10, 0).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_touch() {
        let store = SqliteNodeStore::open_in_memory().unwrap();
        let node = KnowledgeNode::new(NodeKind::Fact, "touchable");
        let id = node.id;
        store.insert(&node).await.unwrap();

        store.touch(id).await.unwrap();
        let retrieved = store.get(id).await.unwrap().unwrap();
        assert_eq!(retrieved.temporal.access_count, 1);
    }

    #[tokio::test]
    async fn test_count() {
        let store = SqliteNodeStore::open_in_memory().unwrap();
        store
            .insert(&KnowledgeNode::new(NodeKind::Fact, "a"))
            .await
            .unwrap();
        store
            .insert(&KnowledgeNode::new(NodeKind::Fact, "b"))
            .await
            .unwrap();
        store
            .insert(&KnowledgeNode::new(NodeKind::Decision, "c"))
            .await
            .unwrap();

        let all = store.count(&QueryFilters::default()).await.unwrap();
        assert_eq!(all, 3);

        let facts = store
            .count(&QueryFilters {
                kinds: Some(vec![NodeKind::Fact]),
                ..Default::default()
            })
            .await
            .unwrap();
        assert_eq!(facts, 2);
    }

    #[tokio::test]
    async fn test_public_share_lifecycle() {
        let store = SqliteNodeStore::open_in_memory().unwrap();
        let node = KnowledgeNode::new(NodeKind::Fact, "Shared note")
            .with_title("Shared")
            .with_namespace("default");
        let node_id = node.id;
        store.insert(&node).await.unwrap();

        let share = PublicShare {
            id: Uuid::now_v7(),
            node_id,
            token_hash: "hash-abc".to_string(),
            created_at: Utc::now(),
            expires_at: None,
            revoked_at: None,
        };

        store.insert_public_share(&share).await.unwrap();

        let fetched = store.get_public_share(share.id).await.unwrap().unwrap();
        assert_eq!(fetched.node_id, node_id);
        assert_eq!(fetched.token_hash, "hash-abc");

        let by_hash = store
            .get_public_share_by_hash("hash-abc")
            .await
            .unwrap()
            .unwrap();
        assert_eq!(by_hash.id, share.id);

        let shares = store
            .list_public_shares(Some(node_id), false)
            .await
            .unwrap();
        assert_eq!(shares.len(), 1);

        let revoked = store
            .revoke_public_share(share.id, Utc::now())
            .await
            .unwrap();
        assert!(revoked);

        let active = store
            .list_public_shares(Some(node_id), false)
            .await
            .unwrap();
        assert!(active.is_empty());

        let all = store.list_public_shares(Some(node_id), true).await.unwrap();
        assert_eq!(all.len(), 1);
        assert!(all[0].revoked_at.is_some());
    }

    // -----------------------------------------------------------------------
    // Integration tests for Phase 1–3 storage features
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_adapter_poll_state_round_trip() {
        let store = SqliteNodeStore::open_in_memory().unwrap();

        // Initially empty
        let state = store.get_poll_state("discord").await.unwrap();
        assert!(state.is_none());

        // Upsert
        store
            .upsert_poll_state("discord", "cursor-abc-123", 5)
            .await
            .unwrap();

        let state = store.get_poll_state("discord").await.unwrap().unwrap();
        assert_eq!(state.adapter_name, "discord");
        assert_eq!(state.cursor, "cursor-abc-123");
        assert_eq!(state.messages_received, 5);

        // Upsert again — cursor updates, messages_received accumulates
        store
            .upsert_poll_state("discord", "cursor-def-456", 3)
            .await
            .unwrap();

        let state = store.get_poll_state("discord").await.unwrap().unwrap();
        assert_eq!(state.cursor, "cursor-def-456");
        assert_eq!(state.messages_received, 8); // 5 + 3

        // List
        store
            .upsert_poll_state("slack", "slack-cursor", 1)
            .await
            .unwrap();
        let all = store.list_poll_states().await.unwrap();
        assert_eq!(all.len(), 2);

        // Delete
        let deleted = store.delete_poll_state("discord").await.unwrap();
        assert!(deleted);
        assert!(store.get_poll_state("discord").await.unwrap().is_none());

        let not_found = store.delete_poll_state("nonexistent").await.unwrap();
        assert!(!not_found);
    }

    #[tokio::test]
    async fn test_contact_identity_storage() {
        let store = SqliteNodeStore::open_in_memory().unwrap();

        let contact_id = Uuid::now_v7();
        let identity = ContactIdentity {
            id: Uuid::now_v7(),
            contact_id,
            identity_type: IdentityType::Email,
            identity_value: "bob@example.com".into(),
            verified: false,
            verified_at: None,
            created_at: Utc::now(),
        };

        // Add
        store.add_contact_identity(&identity).await.unwrap();

        // List
        let list = store.list_contact_identities(contact_id).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].identity_value, "bob@example.com");
        assert!(!list[0].verified);

        // Verify
        let verified = store.verify_contact_identity(identity.id).await.unwrap();
        assert!(verified);

        let list = store.list_contact_identities(contact_id).await.unwrap();
        assert!(list[0].verified);
        assert!(list[0].verified_at.is_some());

        // Verify again (already verified) — should return false
        let re_verified = store.verify_contact_identity(identity.id).await.unwrap();
        assert!(!re_verified);

        // Delete
        let deleted = store.delete_contact_identity(identity.id).await.unwrap();
        assert!(deleted);
        assert!(store
            .list_contact_identities(contact_id)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn test_trust_model_storage() {
        let store = SqliteNodeStore::open_in_memory().unwrap();

        let contact_id = Uuid::now_v7();

        // Not present initially
        assert!(store.get_trust_model(contact_id).await.unwrap().is_none());

        // Set
        let model = TrustModel {
            contact_id,
            can_query: true,
            can_inject_context: false,
            can_auto_reply: false,
            allowed_namespaces: vec!["research".into(), "notes".into()],
            max_confidence_override: Some(0.9),
            updated_at: Utc::now(),
        };
        store.set_trust_model(&model).await.unwrap();

        let stored = store.get_trust_model(contact_id).await.unwrap().unwrap();
        assert!(stored.can_query);
        assert!(!stored.can_inject_context);
        assert_eq!(stored.allowed_namespaces, vec!["research", "notes"]);
        assert!((stored.max_confidence_override.unwrap() - 0.9).abs() < f64::EPSILON);

        // Update (upsert)
        let updated_model = TrustModel {
            contact_id,
            can_query: true,
            can_inject_context: true,
            can_auto_reply: true,
            allowed_namespaces: vec!["all".into()],
            max_confidence_override: None,
            updated_at: Utc::now(),
        };
        store.set_trust_model(&updated_model).await.unwrap();

        let stored = store.get_trust_model(contact_id).await.unwrap().unwrap();
        assert!(stored.can_inject_context);
        assert!(stored.can_auto_reply);
        assert_eq!(stored.allowed_namespaces, vec!["all"]);
        assert!(stored.max_confidence_override.is_none());
    }
}
