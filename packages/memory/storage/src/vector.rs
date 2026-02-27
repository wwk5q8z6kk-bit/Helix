use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[cfg(feature = "local-embeddings")]
use std::sync::Mutex;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use async_trait::async_trait;
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::sync::OnceCell;
use tokio::sync::RwLock;
use tracing::warn;
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::sealed_runtime::{runtime_root_key, sealed_mode_enabled};
use crate::vault_crypto::VaultCrypto;
use hx_core::*;

#[cfg(feature = "local-embeddings")]
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};

const LANCEDB_SNAPSHOT_CONTEXT: &str = "sealed:lancedb:snapshot";
const LANCEDB_SNAPSHOT_MAGIC: &[u8] = b"MVLDB1";
const LANCEDB_SNAPSHOT_FILENAME: &str = "vectors.snapshot";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SealedVectorSnapshot {
    version: u8,
    rows: Vec<SealedVectorRow>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SealedVectorRow {
    id: String,
    content: String,
    vector: Vec<f32>,
    namespace: Option<String>,
}

#[derive(Debug, Clone)]
struct SealedLanceSnapshotStore {
    snapshot_path: PathBuf,
}

impl SealedLanceSnapshotStore {
    fn open(root: &Path) -> MvResult<Self> {
        std::fs::create_dir_all(root)
            .map_err(|err| HxError::Storage(format!("create sealed lancedb dir failed: {err}")))?;

        Ok(Self {
            snapshot_path: root.join(LANCEDB_SNAPSHOT_FILENAME),
        })
    }

    fn load_snapshot(&self) -> MvResult<Option<SealedVectorSnapshot>> {
        let bytes = match std::fs::read(&self.snapshot_path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(err) => {
                return Err(HxError::Storage(format!(
                    "read sealed vector snapshot failed: {err}"
                )));
            }
        };

        let kek = derive_lancedb_kek()?;
        let snapshot = open_snapshot_envelope(&kek, &bytes)?;
        Ok(Some(snapshot))
    }

    fn save_snapshot(&self, snapshot: &SealedVectorSnapshot) -> MvResult<()> {
        let kek = derive_lancedb_kek()?;
        let encoded = seal_snapshot(&kek, snapshot)?;
        let tmp_path = self.snapshot_path.with_extension("tmp");
        std::fs::write(&tmp_path, encoded).map_err(|err| {
            HxError::Storage(format!("write sealed vector snapshot failed: {err}"))
        })?;
        std::fs::rename(tmp_path, &self.snapshot_path).map_err(|err| {
            HxError::Storage(format!("commit sealed vector snapshot failed: {err}"))
        })?;
        Ok(())
    }
}

fn derive_lancedb_kek() -> MvResult<[u8; 32]> {
    let root = runtime_root_key().ok_or(HxError::VaultSealed)?;
    let mut crypto = VaultCrypto::new();
    crypto.set_master_key(Zeroizing::new(root));
    let key = crypto
        .derive_namespace_kek(LANCEDB_SNAPSHOT_CONTEXT)
        .map_err(|err| HxError::Storage(format!("derive lancedb key failed: {err}")))?;
    Ok(*key)
}

fn seal_snapshot(kek: &[u8; 32], snapshot: &SealedVectorSnapshot) -> MvResult<Vec<u8>> {
    let payload = serde_json::to_vec(snapshot)
        .map_err(|err| HxError::Storage(format!("serialize vector snapshot failed: {err}")))?;
    let dek = VaultCrypto::generate_node_dek();
    let wrapped_dek = VaultCrypto::wrap_node_dek(kek, &dek)
        .map_err(|err| HxError::Storage(format!("wrap vector snapshot DEK failed: {err}")))?;
    let ciphertext = VaultCrypto::aes_gcm_encrypt_pub(&dek, &payload)
        .map_err(|err| HxError::Storage(format!("encrypt vector snapshot failed: {err}")))?;

    let wrapped = wrapped_dek.as_bytes();
    if wrapped.len() > u16::MAX as usize {
        return Err(HxError::Storage("wrapped DEK too large".into()));
    }

    let mut out =
        Vec::with_capacity(LANCEDB_SNAPSHOT_MAGIC.len() + 2 + wrapped.len() + ciphertext.len());
    out.extend_from_slice(LANCEDB_SNAPSHOT_MAGIC);
    out.extend_from_slice(&(wrapped.len() as u16).to_le_bytes());
    out.extend_from_slice(wrapped);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

fn open_snapshot_envelope(kek: &[u8; 32], payload: &[u8]) -> MvResult<SealedVectorSnapshot> {
    let min_header = LANCEDB_SNAPSHOT_MAGIC.len() + 2;
    if payload.len() < min_header || !payload.starts_with(LANCEDB_SNAPSHOT_MAGIC) {
        return Err(HxError::Storage(
            "invalid sealed vector snapshot envelope".into(),
        ));
    }

    let mut wrapped_len_bytes = [0u8; 2];
    wrapped_len_bytes.copy_from_slice(&payload[LANCEDB_SNAPSHOT_MAGIC.len()..min_header]);
    let wrapped_len = u16::from_le_bytes(wrapped_len_bytes) as usize;
    let wrapped_start = min_header;
    let wrapped_end = wrapped_start + wrapped_len;
    if payload.len() < wrapped_end {
        return Err(HxError::Storage(
            "sealed vector snapshot missing wrapped key".into(),
        ));
    }

    let wrapped = std::str::from_utf8(&payload[wrapped_start..wrapped_end])
        .map_err(|err| HxError::Storage(format!("wrapped vector DEK utf8 decode failed: {err}")))?;
    let ciphertext = &payload[wrapped_end..];

    let dek = VaultCrypto::unwrap_node_dek(kek, wrapped)
        .map_err(|err| HxError::Storage(format!("unwrap vector snapshot DEK failed: {err}")))?;
    let plaintext = VaultCrypto::aes_gcm_decrypt_pub(&dek, ciphertext)
        .map_err(|err| HxError::Storage(format!("decrypt vector snapshot failed: {err}")))?;
    let snapshot = serde_json::from_slice::<SealedVectorSnapshot>(&plaintext)
        .map_err(|err| HxError::Storage(format!("parse vector snapshot failed: {err}")))?;
    Ok(snapshot)
}

pub struct LanceVectorStore {
    db: lancedb::Connection,
    table_name: String,
    dimensions: usize,
    table: OnceCell<lancedb::Table>,
    namespace_supported: AtomicBool,
    sealed_snapshot: Option<SealedLanceSnapshotStore>,
}

pub struct InMemoryVectorStore {
    dimensions: usize,
    entries: RwLock<HashMap<Uuid, (Vec<f32>, Option<String>)>>,
}

impl InMemoryVectorStore {
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            entries: RwLock::new(HashMap::new()),
        }
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut left_norm = 0.0f64;
        let mut right_norm = 0.0f64;

        for idx in 0..left.len() {
            let l = left[idx] as f64;
            let r = right[idx] as f64;
            dot += l * r;
            left_norm += l * l;
            right_norm += r * r;
        }

        if left_norm == 0.0 || right_norm == 0.0 {
            0.0
        } else {
            dot / (left_norm.sqrt() * right_norm.sqrt())
        }
    }
}

impl LanceVectorStore {
    pub async fn open(path: &Path, dimensions: usize) -> MvResult<Self> {
        let (db, sealed_snapshot) = if sealed_mode_enabled() {
            let sealed_snapshot = SealedLanceSnapshotStore::open(path)?;
            let memory_uri = format!("memory://helix-{}", Uuid::now_v7());
            let db = lancedb::connect(&memory_uri)
                .execute()
                .await
                .map_err(|e| HxError::Storage(format!("lancedb in-memory connect failed: {e}")))?;
            (db, Some(sealed_snapshot))
        } else {
            let path_str = path
                .to_str()
                .ok_or_else(|| HxError::Storage("invalid lancedb path encoding".into()))?;
            let db = lancedb::connect(path_str)
                .execute()
                .await
                .map_err(|e| HxError::Storage(format!("lancedb connect failed: {e}")))?;
            (db, None)
        };

        let store = Self {
            db,
            table_name: "embeddings".into(),
            dimensions,
            table: OnceCell::new(),
            namespace_supported: AtomicBool::new(true),
            sealed_snapshot,
        };

        store.ensure_table().await?;
        if let Err(err) = store.restore_sealed_snapshot().await {
            if matches!(err, HxError::VaultSealed) {
                tracing::debug!("skipping sealed vector snapshot restore: vault not unsealed yet");
            } else {
                return Err(err);
            }
        }
        Ok(store)
    }

    fn schema_with_namespace(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimensions as i32,
                ),
                false,
            ),
            Field::new("namespace", DataType::Utf8, true),
        ]))
    }

    fn schema_without_namespace(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dimensions as i32,
                ),
                false,
            ),
        ]))
    }

    async fn ensure_table(&self) -> MvResult<()> {
        let tables = self
            .db
            .table_names()
            .execute()
            .await
            .map_err(|e| HxError::Storage(format!("lancedb list tables: {e}")))?;

        if !tables.contains(&self.table_name) {
            let schema = self.schema_with_namespace();
            let batch = RecordBatch::new_empty(schema.clone());
            let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
            let table = self
                .db
                .create_table(&self.table_name, Box::new(batches))
                .execute()
                .await
                .map_err(|e| HxError::Storage(format!("lancedb create table: {e}")))?;
            let _ = self.table.set(table);
        }
        Ok(())
    }

    async fn get_table(&self) -> MvResult<&lancedb::Table> {
        self.table
            .get_or_try_init(|| async {
                self.db
                    .open_table(&self.table_name)
                    .execute()
                    .await
                    .map_err(|e| HxError::Storage(format!("failed to open table: {e}")))
            })
            .await
    }

    async fn restore_sealed_snapshot(&self) -> MvResult<()> {
        let Some(snapshot_store) = &self.sealed_snapshot else {
            return Ok(());
        };
        let Some(snapshot) = snapshot_store.load_snapshot()? else {
            return Ok(());
        };
        if snapshot.rows.is_empty() {
            return Ok(());
        }

        let mut items = Vec::with_capacity(snapshot.rows.len());
        for row in snapshot.rows {
            if row.vector.len() != self.dimensions {
                warn!(
                    id = %row.id,
                    expected_dimensions = self.dimensions,
                    actual_dimensions = row.vector.len(),
                    "skipping sealed vector row due to dimension mismatch"
                );
                continue;
            }
            let id = match Uuid::parse_str(&row.id) {
                Ok(id) => id,
                Err(err) => {
                    warn!(id = %row.id, error = %err, "skipping sealed vector row with invalid UUID");
                    continue;
                }
            };
            items.push((id, row.vector, row.content, row.namespace));
        }

        self.upsert_batch_internal(&items).await?;
        Ok(())
    }

    async fn persist_sealed_snapshot_if_needed(&self) -> MvResult<()> {
        let Some(snapshot_store) = &self.sealed_snapshot else {
            return Ok(());
        };
        let snapshot = self.export_snapshot().await?;
        snapshot_store.save_snapshot(&snapshot)?;
        Ok(())
    }

    async fn export_snapshot(&self) -> MvResult<SealedVectorSnapshot> {
        let table = self.get_table().await?;
        let stream = table
            .query()
            .execute()
            .await
            .map_err(|err| HxError::Storage(format!("lancedb snapshot query failed: {err}")))?;
        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|err| HxError::Storage(format!("lancedb snapshot collect failed: {err}")))?;

        let mut rows = Vec::new();
        for batch in &batches {
            let id_col = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| HxError::Storage("vector snapshot missing id column".into()))?;
            let content_col = batch
                .column_by_name("content")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
                .ok_or_else(|| HxError::Storage("vector snapshot missing content column".into()))?;
            let vector_col = batch
                .column_by_name("vector")
                .and_then(|c| c.as_any().downcast_ref::<FixedSizeListArray>())
                .ok_or_else(|| HxError::Storage("vector snapshot missing vector column".into()))?;
            let namespace_col = batch
                .column_by_name("namespace")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());

            for row_idx in 0..batch.num_rows() {
                let id = id_col.value(row_idx).to_string();
                let content = content_col.value(row_idx).to_string();

                let vector_values = vector_col.value(row_idx);
                let vector_values = vector_values
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        HxError::Storage("vector snapshot list contains non-float data".into())
                    })?;
                let mut vector = Vec::with_capacity(vector_values.len());
                for dim_idx in 0..vector_values.len() {
                    vector.push(vector_values.value(dim_idx));
                }

                let namespace = namespace_col.and_then(|col| {
                    if col.is_null(row_idx) {
                        None
                    } else {
                        Some(col.value(row_idx).to_string())
                    }
                });

                rows.push(SealedVectorRow {
                    id,
                    content,
                    vector,
                    namespace,
                });
            }
        }

        Ok(SealedVectorSnapshot { version: 1, rows })
    }

    fn escape_filter_value(value: &str) -> String {
        value.replace('\'', "''")
    }

    fn is_namespace_schema_error(message: &str) -> bool {
        let lower = message.to_lowercase();
        lower.contains("namespace")
            && (lower.contains("schema") || lower.contains("column") || lower.contains("field"))
    }

    fn should_use_namespace(&self) -> bool {
        self.namespace_supported.load(Ordering::Relaxed)
    }

    fn disable_namespace(&self) {
        if self.namespace_supported.swap(false, Ordering::Relaxed) {
            warn!(
                "LanceDB table does not support namespace column. Falling back to legacy schema. Rebuild the vector table to enable namespace filtering."
            );
        }
    }

    fn build_batch(
        &self,
        ids: Vec<String>,
        contents: Vec<String>,
        all_floats: Vec<f32>,
        include_namespace: bool,
        namespaces: Option<Vec<Option<String>>>,
    ) -> MvResult<(Arc<Schema>, RecordBatch)> {
        let schema = if include_namespace {
            self.schema_with_namespace()
        } else {
            self.schema_without_namespace()
        };

        let id_array = StringArray::from(ids);
        let content_array = StringArray::from(contents);
        let values = Float32Array::from(all_floats);
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = arrow_array::FixedSizeListArray::new(
            field,
            self.dimensions as i32,
            Arc::new(values) as Arc<dyn Array>,
            None,
        );

        let columns: Vec<Arc<dyn Array>> = if include_namespace {
            let namespaces = namespaces.unwrap_or_else(|| vec![None; id_array.len()]);
            if namespaces.len() != id_array.len() {
                return Err(HxError::InvalidInput(format!(
                    "namespace length mismatch: expected {}, got {}",
                    id_array.len(),
                    namespaces.len()
                )));
            }
            let ns_array = StringArray::from(namespaces);
            vec![
                Arc::new(id_array),
                Arc::new(content_array),
                Arc::new(vector_array),
                Arc::new(ns_array),
            ]
        } else {
            vec![
                Arc::new(id_array),
                Arc::new(content_array),
                Arc::new(vector_array),
            ]
        };

        let batch = RecordBatch::try_new(schema.clone(), columns)
            .map_err(|e| HxError::Storage(format!("record batch error: {e}")))?;

        Ok((schema, batch))
    }

    /// Create a vector index if the table has enough rows.
    ///
    /// IVF-PQ indexing requires at least 256 rows to be effective.
    /// This is idempotent — calling it multiple times replaces the existing index.
    pub async fn ensure_index(&self) -> MvResult<()> {
        let table = self.get_table().await?;
        let count = table
            .count_rows(None)
            .await
            .map_err(|e| HxError::Storage(e.to_string()))?;
        if count >= 256 {
            table
                .create_index(&["vector"], lancedb::index::Index::Auto)
                .execute()
                .await
                .map_err(|e| HxError::Storage(format!("index creation failed: {e}")))?;
        }
        Ok(())
    }

    /// Bulk upsert embeddings using merge-insert (much faster than individual upserts).
    pub async fn upsert_batch(
        &self,
        items: &[(Uuid, Vec<f32>, String, Option<String>)],
    ) -> MvResult<()> {
        self.upsert_batch_internal(items).await?;
        self.persist_sealed_snapshot_if_needed().await?;
        Ok(())
    }

    async fn upsert_batch_internal(
        &self,
        items: &[(Uuid, Vec<f32>, String, Option<String>)],
    ) -> MvResult<()> {
        if items.is_empty() {
            return Ok(());
        }
        for (_, emb, _, _) in items {
            if emb.len() != self.dimensions {
                return Err(HxError::InvalidInput(format!(
                    "embedding dimension mismatch: expected {}, got {}",
                    self.dimensions,
                    emb.len()
                )));
            }
        }

        let table = self.get_table().await?;

        let ids: Vec<String> = items.iter().map(|(id, _, _, _)| id.to_string()).collect();
        let contents: Vec<String> = items.iter().map(|(_, _, c, _)| c.clone()).collect();
        let all_floats: Vec<f32> = items
            .iter()
            .flat_map(|(_, emb, _, _)| emb.iter().copied())
            .collect();

        let mut include_namespace = self.should_use_namespace();
        let namespaces = if include_namespace {
            Some(
                items
                    .iter()
                    .map(|(_, _, _, ns)| ns.clone())
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
        let (schema, batch) = self.build_batch(
            ids.clone(),
            contents.clone(),
            all_floats.clone(),
            include_namespace,
            namespaces.clone(),
        )?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let mut merge = table.merge_insert(&["id"]);
        merge
            .when_matched_update_all(None)
            .when_not_matched_insert_all();
        let result = merge.execute(Box::new(batches)).await;

        if let Err(err) = result {
            let message = err.to_string();
            if include_namespace && Self::is_namespace_schema_error(&message) {
                self.disable_namespace();
                include_namespace = false;
                let (schema, batch) =
                    self.build_batch(ids, contents, all_floats, include_namespace, None)?;
                let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
                let mut merge = table.merge_insert(&["id"]);
                merge
                    .when_matched_update_all(None)
                    .when_not_matched_insert_all();
                merge
                    .execute(Box::new(batches))
                    .await
                    .map_err(|e| HxError::Storage(format!("lancedb batch upsert: {e}")))?;
            } else {
                return Err(HxError::Storage(format!("lancedb batch upsert: {err}")));
            }
        }

        Ok(())
    }

    /// Vector search with optional namespace pre-filtering.
    pub async fn search_with_namespace(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        min_score: f64,
        namespace: Option<&str>,
    ) -> MvResult<Vec<(Uuid, f64)>> {
        if embedding.len() != self.dimensions {
            return Err(HxError::InvalidInput(format!(
                "query embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        let table = self.get_table().await?;

        let embedding_fallback = embedding.clone();
        let mut query = table
            .vector_search(embedding)
            .map_err(|e| HxError::Storage(format!("lancedb query build: {e}")))?
            .limit(limit);

        let mut applied_namespace = false;
        if let Some(ns) = namespace {
            if self.should_use_namespace() {
                let escaped = Self::escape_filter_value(ns);
                query = query.only_if(format!("namespace = '{escaped}'"));
                applied_namespace = true;
            }
        }

        let stream = query.execute().await;

        let stream = match stream {
            Ok(stream) => stream,
            Err(err) => {
                let message = err.to_string();
                if applied_namespace && Self::is_namespace_schema_error(&message) {
                    self.disable_namespace();
                    let query = table
                        .vector_search(embedding_fallback)
                        .map_err(|e| HxError::Storage(format!("lancedb query build: {e}")))?
                        .limit(limit);
                    query
                        .execute()
                        .await
                        .map_err(|e| HxError::Storage(format!("lancedb search: {e}")))?
                } else {
                    return Err(HxError::Storage(format!("lancedb search: {err}")));
                }
            }
        };

        let batches: Vec<RecordBatch> = stream
            .try_collect()
            .await
            .map_err(|e| HxError::Storage(format!("lancedb collect: {e}")))?;

        let mut scored = Vec::new();
        for batch in &batches {
            let id_col: Option<&StringArray> = batch
                .column_by_name("id")
                .and_then(|c: &Arc<dyn Array>| c.as_any().downcast_ref::<StringArray>());
            let dist_col: Option<&Float32Array> = batch
                .column_by_name("_distance")
                .and_then(|c: &Arc<dyn Array>| c.as_any().downcast_ref::<Float32Array>());

            if let (Some(ids), Some(distances)) = (id_col, dist_col) {
                for i in 0..ids.len() {
                    if let Ok(uuid) = Uuid::parse_str(ids.value(i)) {
                        let distance = distances.value(i) as f64;
                        let score = 1.0 / (1.0 + distance);
                        scored.push((uuid, score));
                    }
                }
            }
        }

        scored.retain(|(_, score)| *score >= min_score);
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored)
    }
}

#[async_trait]
impl VectorStore for LanceVectorStore {
    async fn upsert(
        &self,
        id: Uuid,
        embedding: Vec<f32>,
        content: &str,
        namespace: Option<&str>,
    ) -> MvResult<()> {
        if embedding.len() != self.dimensions {
            return Err(HxError::InvalidInput(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        // Delete existing if present, then insert
        if let Ok(table) = self.get_table().await {
            let _ = table.delete(&format!("id = '{}'", id)).await;
        }

        let table = self.get_table().await?;

        let ids = vec![id.to_string()];
        let contents = vec![content.to_string()];
        let all_floats = embedding;
        let mut include_namespace = self.should_use_namespace();
        let namespaces = namespace.map(|value| vec![Some(value.to_string())]);

        let (schema, batch) = self.build_batch(
            ids.clone(),
            contents.clone(),
            all_floats.clone(),
            include_namespace,
            namespaces.clone(),
        )?;
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        let result = table.add(Box::new(batches)).execute().await;
        if let Err(err) = result {
            let message = err.to_string();
            if include_namespace && Self::is_namespace_schema_error(&message) {
                self.disable_namespace();
                include_namespace = false;
                let (schema, batch) =
                    self.build_batch(ids, contents, all_floats, include_namespace, None)?;
                let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
                table
                    .add(Box::new(batches))
                    .execute()
                    .await
                    .map_err(|e| HxError::Storage(format!("lancedb upsert: {e}")))?;
            } else {
                return Err(HxError::Storage(format!("lancedb upsert: {err}")));
            }
        }

        self.persist_sealed_snapshot_if_needed().await?;
        Ok(())
    }

    async fn search(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        min_score: f64,
        namespace: Option<&str>,
    ) -> MvResult<Vec<(Uuid, f64)>> {
        self.search_with_namespace(embedding, limit, min_score, namespace)
            .await
    }

    async fn delete(&self, id: Uuid) -> MvResult<()> {
        let table = self.get_table().await?;
        table
            .delete(&format!("id = '{}'", id))
            .await
            .map_err(|e| HxError::Storage(format!("lancedb delete: {e}")))?;
        self.persist_sealed_snapshot_if_needed().await?;
        Ok(())
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn upsert(
        &self,
        id: Uuid,
        embedding: Vec<f32>,
        _content: &str,
        namespace: Option<&str>,
    ) -> MvResult<()> {
        if embedding.len() != self.dimensions {
            return Err(HxError::InvalidInput(format!(
                "embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }
        let mut guard = self.entries.write().await;
        guard.insert(id, (embedding, namespace.map(str::to_string)));
        Ok(())
    }

    async fn search(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        min_score: f64,
        namespace: Option<&str>,
    ) -> MvResult<Vec<(Uuid, f64)>> {
        if embedding.len() != self.dimensions {
            return Err(HxError::InvalidInput(format!(
                "query embedding dimension mismatch: expected {}, got {}",
                self.dimensions,
                embedding.len()
            )));
        }

        let guard = self.entries.read().await;
        let mut results = Vec::new();

        for (id, (candidate, candidate_namespace)) in guard.iter() {
            if let Some(ns) = namespace {
                if candidate_namespace.as_deref() != Some(ns) {
                    continue;
                }
            }
            let score = Self::cosine_similarity(&embedding, candidate);
            if score >= min_score {
                results.push((*id, score));
            }
        }

        results.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if results.len() > limit {
            results.truncate(limit);
        }
        Ok(results)
    }

    async fn delete(&self, id: Uuid) -> MvResult<()> {
        let mut guard = self.entries.write().await;
        guard.remove(&id);
        Ok(())
    }
}

/// OpenAI-compatible embedding client.
pub struct OpenAiEmbedder {
    client: reqwest::Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    dimensions: usize,
}

impl OpenAiEmbedder {
    pub fn new(
        base_url: String,
        api_key: Option<String>,
        model: String,
        dimensions: usize,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key,
            model,
            dimensions,
        }
    }

    /// Create for Ollama (no API key, default base URL).
    pub fn for_ollama(base_url: Option<String>, model: String, dimensions: usize) -> Self {
        Self::new(
            base_url.unwrap_or_else(|| "http://localhost:11434/v1".into()),
            None,
            model,
            dimensions,
        )
    }

    /// Create for any OpenAI-compatible API.
    pub fn for_compatible(
        base_url: String,
        api_key: Option<String>,
        model: String,
        dimensions: usize,
    ) -> Self {
        Self::new(base_url, api_key, model, dimensions)
    }
}

/// Local embedding provider using fastembed (ONNX).
pub struct KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder {
    #[cfg(feature = "local-embeddings")]
    model: Arc<Mutex<TextEmbedding>>,
    model_name: String,
    dimensions: usize,
}

impl KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder {
    pub fn try_new(model_name: &str) -> MvResult<Self> {
        #[cfg(not(feature = "local-embeddings"))]
        {
            Err(HxError::Config(format!(
                "local_fastembed provider requires hx-storage feature 'local-embeddings' (requested model '{model_name}')"
            )))
        }

        #[cfg(feature = "local-embeddings")]
        {
            let resolved_model = fastembed_embedding_model_from_name(model_name).ok_or_else(|| {
                HxError::Config(format!(
                    "unsupported fastembed model '{model_name}'. Supported: bge-small-en-v1.5, all-minilm-l6-v2"
                ))
            })?;

            let options = TextInitOptions::new(resolved_model).with_show_download_progress(false);
            let mut embedding_model = TextEmbedding::try_new(options)
                .map_err(|err| HxError::Embedding(format!("fastembed init failed: {err}")))?;

            // Probe output dimensions once during init so vector store sizing can be aligned.
            let probe = embedding_model
                .embed(vec!["dimension probe"], Some(1))
                .map_err(|err| HxError::Embedding(format!("fastembed probe failed: {err}")))?;
            let dimensions = probe.first().map(Vec::len).ok_or_else(|| {
                HxError::Embedding("fastembed probe returned empty output".into())
            })?;

            Ok(Self {
                model: Arc::new(Mutex::new(embedding_model)),
                model_name: normalized_fastembed_model_name(model_name).to_string(),
                dimensions,
            })
        }
    }

    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[async_trait]
impl Embedder for OpenAiEmbedder {
    async fn embed(&self, text: &str) -> MvResult<Vec<f32>> {
        let results = self.embed_batch(&[text.to_string()]).await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| HxError::Embedding("empty response".into()))
    }

    async fn embed_batch(&self, texts: &[String]) -> MvResult<Vec<Vec<f32>>> {
        #[derive(serde::Serialize)]
        struct EmbedRequest<'a> {
            model: &'a str,
            input: &'a [String],
        }

        #[derive(serde::Deserialize)]
        struct EmbedResponse {
            data: Vec<EmbedData>,
        }

        #[derive(serde::Deserialize)]
        struct EmbedData {
            embedding: Vec<f32>,
        }

        let url = format!("{}/embeddings", self.base_url);
        let mut req_builder = self.client.post(&url).json(&EmbedRequest {
            model: &self.model,
            input: texts,
        });
        if let Some(ref key) = self.api_key {
            req_builder = req_builder.bearer_auth(key);
        }
        let resp = req_builder
            .send()
            .await
            .map_err(|e| HxError::Embedding(format!("request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(HxError::Embedding(format!("API error {status}: {body}")));
        }

        let data: EmbedResponse = resp
            .json()
            .await
            .map_err(|e| HxError::Embedding(format!("parse error: {e}")))?;

        Ok(data.data.into_iter().map(|d| d.embedding).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(feature = "local-embeddings")]
#[async_trait]
impl Embedder for KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder {
    async fn embed(&self, text: &str) -> MvResult<Vec<f32>> {
        let mut values = self.embed_batch(&[text.to_string()]).await?;
        if values.is_empty() {
            return Err(HxError::Embedding("fastembed returned empty batch".into()));
        }
        Ok(values.remove(0))
    }

    async fn embed_batch(&self, texts: &[String]) -> MvResult<Vec<Vec<f32>>> {
        let model = Arc::clone(&self.model);
        let inputs = texts.to_vec();

        tokio::task::spawn_blocking(move || {
            let mut locked = model
                .lock()
                .map_err(|err| HxError::Embedding(format!("fastembed lock poisoned: {err}")))?;
            let embeddings = locked
                .embed(inputs, Some(16))
                .map_err(|err| HxError::Embedding(format!("fastembed inference failed: {err}")))?;
            Ok(embeddings)
        })
        .await
        .map_err(|err| HxError::Embedding(format!("fastembed task join error: {err}")))?
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

#[cfg(not(feature = "local-embeddings"))]
#[async_trait]
impl Embedder for KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder {
    async fn embed(&self, _text: &str) -> MvResult<Vec<f32>> {
        Err(HxError::Config(
            "local_fastembed provider is disabled at compile time; rebuild with feature 'local-embeddings'"
                .into(),
        ))
    }

    async fn embed_batch(&self, _texts: &[String]) -> MvResult<Vec<Vec<f32>>> {
        Err(HxError::Config(
            "local_fastembed provider is disabled at compile time; rebuild with feature 'local-embeddings'"
                .into(),
        ))
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// No-op embedder for testing (returns zero vectors).
pub struct NoopEmbedder {
    dims: usize,
}

impl NoopEmbedder {
    pub fn new(dims: usize) -> Self {
        Self { dims }
    }
}

#[async_trait]
impl Embedder for NoopEmbedder {
    async fn embed(&self, _text: &str) -> MvResult<Vec<f32>> {
        Ok(vec![0.0; self.dims])
    }

    async fn embed_batch(&self, texts: &[String]) -> MvResult<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; self.dims]).collect())
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

#[cfg(any(test, feature = "local-embeddings"))]
fn normalized_fastembed_model_name(model_name: &str) -> &str {
    match model_name.trim().to_ascii_lowercase().as_str() {
        "baai/bge-small-en-v1.5" | "bge-small-en-v1.5" => "bge-small-en-v1.5",
        "sentence-transformers/all-minilm-l6-v2" | "all-minilm-l6-v2" => "all-minilm-l6-v2",
        _ => model_name,
    }
}

#[cfg(any(test, feature = "local-embeddings"))]
fn fastembed_model_key_from_name(model_name: &str) -> Option<&'static str> {
    match normalized_fastembed_model_name(model_name) {
        "bge-small-en-v1.5" => Some("bge-small-en-v1.5"),
        "all-minilm-l6-v2" => Some("all-minilm-l6-v2"),
        _ => None,
    }
}

#[cfg(feature = "local-embeddings")]
fn fastembed_embedding_model_from_name(model_name: &str) -> Option<EmbeddingModel> {
    match fastembed_model_key_from_name(model_name) {
        Some("bge-small-en-v1.5") => Some(EmbeddingModel::BGESmallENV15),
        Some("all-minilm-l6-v2") => Some(EmbeddingModel::AllMiniLML6V2),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sealed_runtime::{
        clear_runtime_root_key, set_runtime_root_key, set_sealed_mode_enabled,
    };
    use tempfile::tempdir;

    fn bytes_contains(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() || haystack.len() < needle.len() {
            return false;
        }
        haystack.windows(needle.len()).any(|window| window == needle)
    }

    struct SealedRuntimeReset;

    impl Drop for SealedRuntimeReset {
        fn drop(&mut self) {
            clear_runtime_root_key();
            set_sealed_mode_enabled(false);
        }
    }

    #[test]
    fn model_name_normalization_works() {
        assert_eq!(
            normalized_fastembed_model_name("BAAI/bge-small-en-v1.5"),
            "bge-small-en-v1.5"
        );
        assert_eq!(
            normalized_fastembed_model_name("sentence-transformers/all-minilm-l6-v2"),
            "all-minilm-l6-v2"
        );
    }

    #[test]
    fn supported_models_map() {
        assert_eq!(
            fastembed_model_key_from_name("bge-small-en-v1.5"),
            Some("bge-small-en-v1.5")
        );
        assert_eq!(
            fastembed_model_key_from_name("all-minilm-l6-v2"),
            Some("all-minilm-l6-v2")
        );
        assert_eq!(fastembed_model_key_from_name("unsupported-model"), None);
    }

    #[tokio::test]
    async fn sealed_lancedb_open_without_runtime_key_succeeds() {
        let _reset = SealedRuntimeReset;
        set_sealed_mode_enabled(true);
        clear_runtime_root_key();

        let dir = tempdir().expect("tempdir");
        let opened = LanceVectorStore::open(dir.path(), 8).await;
        let err = opened.err();
        assert!(
            err.is_none(),
            "sealed LanceDB open should defer key usage: {}",
            err.map(|e| e.to_string()).unwrap_or_default()
        );
    }

    #[tokio::test]
    async fn sealed_lancedb_snapshot_roundtrip() {
        let _reset = SealedRuntimeReset;
        set_sealed_mode_enabled(true);
        set_runtime_root_key([13u8; 32], false);

        let dir = tempdir().expect("tempdir");
        let id = Uuid::now_v7();
        let marker = format!("sealed-vector-content-{}", Uuid::now_v7());

        let store = LanceVectorStore::open(dir.path(), 3).await.unwrap();
        store
            .upsert(
                id,
                vec![0.9, 0.1, 0.0],
                &marker,
                Some("default"),
            )
            .await
            .unwrap();

        let snapshot_path = dir.path().join(LANCEDB_SNAPSHOT_FILENAME);
        let snapshot_bytes = std::fs::read(&snapshot_path).expect("snapshot should exist");
        assert!(
            snapshot_bytes.starts_with(LANCEDB_SNAPSHOT_MAGIC),
            "snapshot must use sealed envelope magic"
        );
        assert!(
            !bytes_contains(&snapshot_bytes, marker.as_bytes()),
            "snapshot must not persist plaintext content"
        );
        drop(store);

        let reopened = LanceVectorStore::open(dir.path(), 3).await.unwrap();
        let hits = reopened
            .search(vec![0.9, 0.1, 0.0], 10, 0.0, Some("default"))
            .await
            .unwrap();
        assert!(hits.iter().any(|(hit_id, _)| *hit_id == id));
    }

    #[cfg(not(feature = "local-embeddings"))]
    #[test]
    fn local_embedder_requires_feature_flag() {
        let err = match KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder::try_new(
            "bge-small-en-v1.5",
        ) {
            Ok(_) => panic!("local embedder should require feature flag in default build"),
            Err(err) => err,
        };
        match err {
            HxError::Config(message) => {
                assert!(
                    message.contains("local-embeddings"),
                    "unexpected config error: {message}"
                );
            }
            other => panic!("expected config error, got {other:?}"),
        }
    }

    #[cfg(feature = "local-embeddings")]
    #[test]
    fn local_embedder_rejects_unsupported_model_name() {
        let err =
            match KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder::try_new("unsupported") {
                Ok(_) => panic!("unsupported model must be rejected"),
                Err(err) => err,
            };
        match err {
            HxError::Config(message) => {
                assert!(
                    message.contains("unsupported fastembed model"),
                    "unexpected config error: {message}"
                );
            }
            other => panic!("expected config error, got {other:?}"),
        }
    }
}
