use std::sync::Arc;

use hx_core::*;
use hx_graph::store::SqliteGraphStore;
use hx_index::tantivy_index::TantivyFullTextIndex;
use hx_storage::unified::UnifiedStore;

use crate::ai_autotag::KnowledgeVaultIndexNoteEmbeddingAutoTagger;
use crate::config::EngineConfig;
use crate::conflict::ConflictDetector;

/// Ingest pipeline: validates, stores, indexes, and embeds a knowledge node.
pub struct IngestPipeline {
    store: Arc<UnifiedStore>,
    fts: Arc<TantivyFullTextIndex>,
    graph: Arc<SqliteGraphStore>,
    config: EngineConfig,
    knowledge_vault_index_note_embedding_auto_tagger: KnowledgeVaultIndexNoteEmbeddingAutoTagger,
}

impl IngestPipeline {
    pub fn new(
        store: Arc<UnifiedStore>,
        fts: Arc<TantivyFullTextIndex>,
        graph: Arc<SqliteGraphStore>,
        config: EngineConfig,
    ) -> Self {
        Self {
            store,
            fts,
            graph,
            config,
            knowledge_vault_index_note_embedding_auto_tagger:
                KnowledgeVaultIndexNoteEmbeddingAutoTagger::new(),
        }
    }

    /// Ingest a node: store in SQLite, index in Tantivy, embed in LanceDB.
    pub async fn ingest(&self, mut node: KnowledgeNode) -> MvResult<KnowledgeNode> {
        self.apply_ai_auto_tagging_if_enabled(&mut node).await;

        // 1. Store in SQLite
        self.store.nodes.insert(&node).await?;

        // 2. Index in Tantivy
        self.fts.index_node(&node)?;
        self.fts.commit()?;

        // 3. Generate embedding and store in LanceDB (if vector store available)
        if let Some(ref vectors) = self.store.vectors {
            match self.store.embedder.embed(&node.content).await {
                Ok(embedding) => {
                    vectors
                        .upsert(node.id, embedding, &node.content, Some(&node.namespace))
                        .await?;
                }
                Err(e) => {
                    tracing::warn!("embedding failed for node {}: {e}", node.id);
                    // Continue without embedding — FTS still works
                }
            }
        }

        // 4. Conflict detection (best-effort, non-blocking)
        match ConflictDetector::detect_conflicts(&self.store, &node, 0.5).await {
            Ok(alerts) => {
                for alert in &alerts {
                    if let Err(e) = self.store.nodes.insert_conflict(alert).await {
                        tracing::warn!("failed to store conflict alert: {e}");
                    }
                }
                if !alerts.is_empty() {
                    tracing::info!(
                        count = alerts.len(),
                        node_id = %node.id,
                        "conflict alerts generated"
                    );
                }
            }
            Err(e) => {
                tracing::warn!("conflict detection failed for node {}: {e}", node.id);
            }
        }

        tracing::info!("ingested node {} ({})", node.id, node.kind);
        Ok(node)
    }

    /// Ingest a node with relationships.
    pub async fn ingest_with_relations(
        &self,
        node: KnowledgeNode,
        relations: Vec<Relationship>,
    ) -> MvResult<KnowledgeNode> {
        let node = self.ingest(node).await?;

        for rel in &relations {
            self.graph.add_relationship(rel).await?;
        }

        Ok(node)
    }

    /// Update an existing node, re-index and re-embed.
    pub async fn update(&self, mut node: KnowledgeNode) -> MvResult<KnowledgeNode> {
        self.apply_ai_auto_tagging_if_enabled(&mut node).await;

        self.store.nodes.update(&node).await?;

        self.fts.index_node(&node)?;
        self.fts.commit()?;

        if let Some(ref vectors) = self.store.vectors {
            match self.store.embedder.embed(&node.content).await {
                Ok(embedding) => {
                    vectors
                        .upsert(node.id, embedding, &node.content, Some(&node.namespace))
                        .await?;
                }
                Err(e) => {
                    tracing::warn!("re-embedding failed for node {}: {e}", node.id);
                }
            }
        }

        Ok(node)
    }

    /// Delete a node and its index entries.
    pub async fn delete(&self, id: uuid::Uuid) -> MvResult<bool> {
        self.fts.remove_node(id)?;
        self.fts.commit()?;

        if let Some(ref vectors) = self.store.vectors {
            let _ = vectors.delete(id).await;
        }

        self.graph.remove_node_relationships(id).await?;

        // Clean up attachment blob files
        let blob_dir = std::path::PathBuf::from(&self.config.data_dir)
            .join("blobs")
            .join(id.to_string());
        if blob_dir.is_dir() {
            if let Err(e) = tokio::fs::remove_dir_all(&blob_dir).await {
                tracing::warn!(node_id = %id, error = %e, "failed to remove blob directory");
            }
        }

        self.store.nodes.delete(id).await
    }

    async fn apply_ai_auto_tagging_if_enabled(&self, node: &mut KnowledgeNode) {
        if !self.config.ai.auto_tagging_enabled {
            return;
        }

        self.knowledge_vault_index_note_embedding_auto_tagger
            .enrich_node_tags(node, &self.store, &self.fts, &self.config.ai)
            .await;
    }
}
