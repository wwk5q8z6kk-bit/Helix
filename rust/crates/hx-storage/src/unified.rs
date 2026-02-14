use std::path::Path;
use std::sync::Arc;

use hx_core::*;

use crate::sealed_runtime::sealed_mode_enabled;
use crate::sqlite::SqliteNodeStore;
use crate::vector::{LanceVectorStore, NoopEmbedder};

/// Unified storage that wraps SQLite node store + LanceDB vector store + embedder.
pub struct UnifiedStore {
    pub nodes: Arc<SqliteNodeStore>,
    pub vectors: Option<Arc<dyn VectorStore>>,
    pub embedder: Arc<dyn Embedder>,
}

impl UnifiedStore {
    pub async fn open(data_dir: &Path, dimensions: usize) -> MvResult<Self> {
        std::fs::create_dir_all(data_dir)
            .map_err(|e| HxError::Storage(format!("create data dir: {e}")))?;

        let sqlite_path = data_dir.join("helix.sqlite");
        let lancedb_path = if sealed_mode_enabled() {
            data_dir.join("lancedb.sealed")
        } else {
            data_dir.join("lancedb")
        };

        let nodes = Arc::new(SqliteNodeStore::open(&sqlite_path)?);
        let vectors: Arc<dyn VectorStore> =
            Arc::new(LanceVectorStore::open(&lancedb_path, dimensions).await?);
        let embedder: Arc<dyn Embedder> = Arc::new(NoopEmbedder::new(dimensions));

        Ok(Self {
            nodes,
            vectors: Some(vectors),
            embedder,
        })
    }

    pub fn with_embedder(mut self, embedder: Arc<dyn Embedder>) -> Self {
        self.embedder = embedder;
        self
    }

    /// In-memory store for testing (no vector store available).
    pub fn in_memory(dimensions: usize) -> MvResult<Self> {
        let nodes = Arc::new(SqliteNodeStore::open_in_memory()?);
        let embedder: Arc<dyn Embedder> = Arc::new(NoopEmbedder::new(dimensions));

        Ok(Self {
            nodes,
            vectors: None,
            embedder,
        })
    }
}
