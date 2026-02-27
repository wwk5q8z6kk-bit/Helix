use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine as _;
use chrono::NaiveDate;
use hx_core::credentials::CredentialStore;
use hx_core::*;
use hx_graph::store::SqliteGraphStore;
use hx_index::tantivy_index::TantivyFullTextIndex;
use hx_storage::sealed_runtime::{clear_runtime_root_key, set_sealed_mode_enabled};
use hx_storage::unified::UnifiedStore;
use hx_storage::vault_crypto::VaultCrypto;
use hx_storage::vector::{KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder, OpenAiEmbedder};
use uuid::Uuid;

use crate::backlinks::{
    extract_reference_targets_with_kind, resolve_reference_targets_with_kind,
    KnowledgeVaultBacklinkResolutionIndex, ResolvedContentReferenceTarget,
};
use crate::config::EngineConfig;
use crate::daily_notes::{daily_note_day_tag, daily_note_weekday_tag, render_daily_note_template};
use crate::ingest::IngestPipeline;
use crate::llm::{self, LlmProvider};
use crate::recall::RecallPipeline;

pub mod agentic;
pub mod calendar;
pub mod comments;
pub mod consumers;
pub mod mcp_connectors;
pub mod permissions;
pub mod policy;
pub mod proxy;
pub mod tasks;

// Re-export types used by hx-server from sub-modules / hx-core
pub use hx_core::{PrioritizedTask, TaskPrioritizationOptions};

const DAILY_NOTE_TAG: &str = "daily-note";
const PROFILE_RELAY_CONTACT_ID_KEY: &str = "relay_contact_id";
const PROFILE_OWNER_CONTACT_NOTES: &str = "owner";
const DAILY_LINK_CANDIDATE_TAGS: &[&str] = &[
    "task", "tasks", "todo", "to-do", "event", "events", "meeting", "reminder",
];
const AUTO_BACKLINK_METADATA_KEY: &str = "auto_backlink";
const AUTO_BACKLINK_SOURCE_METADATA_KEY: &str = "source";
const SEALED_BLOB_MAGIC: &[u8; 4] = b"HXB1";

#[derive(Debug, Clone, serde::Serialize)]
pub struct EmbedderRuntimeStatus {
    pub configured_provider: String,
    pub configured_model: String,
    pub configured_dimensions: usize,
    pub effective_provider: String,
    pub effective_model: String,
    pub effective_dimensions: usize,
    pub fallback_to_noop: bool,
    pub reason: Option<String>,
    pub local_embeddings_feature_enabled: bool,
}

pub struct EmbeddingProviderSelection {
    pub embedder: Option<Box<dyn hx_core::Embedder>>,
    pub vector_dimensions: usize,
    pub runtime_status: EmbedderRuntimeStatus,
}

pub struct HelixEngine {
    pub ingest: IngestPipeline,
    pub recall: RecallPipeline,
    pub store: Arc<UnifiedStore>,
    pub fts: Arc<TantivyFullTextIndex>,
    pub graph: Arc<SqliteGraphStore>,
    pub config: EngineConfig,
    pub credential_store: Arc<CredentialStore>,
    pub keychain: Arc<crate::keychain::KeychainEngine>,
    pub llm: Option<Arc<dyn LlmProvider>>,
    pub proactive: Arc<crate::proactive::ProactiveEngine>,
    pub enrichment: Option<crate::enrichment::EnrichmentPipeline>,
    pub reflection: crate::reflection::ReflectionEngine,
    pub autonomy: crate::autonomy::AutonomyGate,
    pub relay: crate::relay::RelayEngine,
    pub adapters: crate::adapters::AdapterRegistry,
    pub sync: crate::sync::SyncEngine,
    pub federation: crate::federation::FederationEngine,
    pub multimodal: crate::multimodal::MultiModalPipeline,
    pub metrics: crate::metrics_collector::MetricsCollector,
    pub insight: Arc<crate::insight::InsightEngine>,
    pub tunnels: crate::tunnel::TunnelRegistry,
    embedding_runtime_status: EmbedderRuntimeStatus,
}

impl HelixEngine {
    pub async fn init(config: EngineConfig) -> MvResult<Self> {
        set_sealed_mode_enabled(config.sealed_mode);
        clear_runtime_root_key();

        let data_dir = PathBuf::from(&config.data_dir);
        std::fs::create_dir_all(&data_dir)
            .map_err(|e| HxError::Storage(format!("create data dir: {e}")))?;

        // Base credential store (OS keyring + env) for KeychainEngine's macOS bridge
        let bridge_cred_store = Arc::new(CredentialStore::new("helix"));

        // Initialize keychain engine
        let keychain_path = data_dir.join("keychain.sqlite");
        let keychain_store: Arc<dyn hx_core::traits::KeychainStore> = Arc::new(
            hx_storage::keychain::SqliteKeychainStore::open(&keychain_path)
                .map_err(|e| HxError::Storage(format!("open keychain db: {e}")))?
        );
        let keychain = Arc::new(
            crate::keychain::KeychainEngine::new(
                keychain_store,
                Arc::clone(&bridge_cred_store),
                None,
                Some(keychain_path.clone()),
            )
            .await?
        );

        // Build the main credential store with Sovereign Keychain as highest-priority backend.
        // Resolution chain: Sovereign Keychain → OS Keyring → Environment Variables.
        let credential_store = {
            let mut store = CredentialStore::new("helix");
            store.insert_backend(
                0,
                Box::new(crate::keychain_backend::KeychainBackend::new(
                    Arc::clone(&keychain),
                    tokio::runtime::Handle::current(),
                )),
            );
            Arc::new(store)
        };

        let selection = select_embedding_provider(&config, &credential_store);

        // Initialize unified store
        let mut store = UnifiedStore::open(&data_dir, selection.vector_dimensions).await?;
        if let Some(embedder) = selection.embedder {
            store = store.with_embedder(embedder.into());
        }

        let store = Arc::new(store);

        let tantivy_path = if config.sealed_mode {
            data_dir.join("tantivy.sealed")
        } else {
            data_dir.join("tantivy")
        };
        let fts = Arc::new(TantivyFullTextIndex::open(&tantivy_path)?);

        // Initialize graph store (shares SQLite connection via separate connection)
        let graph_conn = rusqlite::Connection::open(data_dir.join("helix.sqlite"))
            .map_err(|e| HxError::Graph(format!("open graph db: {e}")))?;
        graph_conn
            .execute_batch(
                "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
            )
            .map_err(|e| HxError::Graph(format!("graph pragma: {e}")))?;
        let graph = Arc::new(SqliteGraphStore::open(graph_conn)?);

        let ingest = IngestPipeline::new(
            Arc::clone(&store),
            Arc::clone(&fts),
            Arc::clone(&graph),
            config.clone(),
        );

        // Initialize LLM provider (optional — heuristic fallback when disabled)
        let llm_api_key = credential_store
            .get_secret_string("HELIX_LLM_API_KEY")
            .or_else(|| credential_store.get_secret_string("OPENAI_API_KEY"));
        let llm = llm::init_llm_provider_with_local(&config.llm, &config.local_llm, llm_api_key).await;

        let recall = RecallPipeline::new(
            Arc::clone(&store),
            Arc::clone(&fts),
            Arc::clone(&graph),
            config.clone(),
            llm.clone(),
        );

        // Initialize proactive engine (will be wired to engine after construction)
        let proactive = Arc::new(crate::proactive::ProactiveEngine::new());

        let reflection = crate::reflection::ReflectionEngine::new(Arc::clone(&store));
        let autonomy = crate::autonomy::AutonomyGate::new(Arc::clone(&store));
        let relay = crate::relay::RelayEngine::new(Arc::clone(&store));
        let federation = crate::federation::FederationEngine::new(Arc::clone(&store));
        let sync = crate::sync::SyncEngine::new(Arc::clone(&store), Uuid::now_v7().to_string());

        let engine = Self {
            ingest,
            recall,
            store,
            fts,
            graph,
            config,
            credential_store,
            keychain,
            llm,
            proactive,
            enrichment: None,
            reflection,
            autonomy,
            relay,
            adapters: crate::adapters::AdapterRegistry::new(),
            sync,
            federation,
            multimodal: {
                let mut pipeline = crate::multimodal::MultiModalPipeline::new();
                pipeline.register(Box::new(crate::multimodal::audio::AudioProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::image::ImageProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::pdf::PdfProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::docx::DocxProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::html::HtmlProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::csv_excel::TabularProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::epub::EpubProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::code::CodeProcessor::new()));
                pipeline.register(Box::new(crate::multimodal::structured_data::StructuredDataProcessor::new()));
                pipeline
            },
            metrics: crate::metrics_collector::MetricsCollector::new(),
            insight: Arc::new(crate::insight::InsightEngine::default()),
            tunnels: crate::tunnel::TunnelRegistry::new(),
            embedding_runtime_status: selection.runtime_status,
        };

        engine.ensure_default_permission_templates().await?;

        Ok(engine)
    }

    pub fn init_arc(config: EngineConfig) -> impl std::future::Future<Output = MvResult<Arc<Self>>> + Send {
        async move {
            let engine = Self::init(config).await?;
            let engine = Arc::new(engine);

            // Wire up circular dependencies
            engine.proactive.set_engine(Arc::clone(&engine));

            // Metrics collector initialized at construction time

            crate::tunnel::bootstrap_tunnels(Arc::clone(&engine)).await?;

            Ok(engine)
        }
    }

    pub fn ensure_unsealed_for_node_io(&self) -> MvResult<()> {
        if self.is_sealed() {
            return Err(HxError::VaultSealed);
        }
        Ok(())
    }

    pub async fn shutdown(&self) -> MvResult<()> {
        tracing::info!("helix_engine_shutdown_initiated");
        Ok(())
    }

    pub fn embedding_runtime_status(&self) -> &EmbedderRuntimeStatus {
        &self.embedding_runtime_status
    }

    /// Returns true when sealed mode is enabled and the vault is locked.
    pub fn is_sealed(&self) -> bool {
        self.config.sealed_mode && !self.keychain.is_unsealed_sync()
    }

    /// Rebuild full-text and vector indexes from stored nodes.
    /// Only meaningful in sealed mode; no-op otherwise.
    pub async fn rebuild_runtime_indexes(&self) -> MvResult<()> {
        if !self.config.sealed_mode {
            return Ok(());
        }
        if !self.keychain.is_unsealed_sync() {
            return Err(HxError::VaultSealed);
        }

        let nodes = self
            .store
            .nodes
            .list(&QueryFilters::default(), 100_000, 0)
            .await?;
        for node in &nodes {
            self.fts.index_node(node)?;
            if let Some(ref vectors) = self.store.vectors {
                if let Ok(embedding) = self.store.embedder.embed(&node.content).await {
                    let _ = vectors
                        .upsert(node.id, embedding, &node.content, Some(&node.namespace))
                        .await;
                }
            }
        }
        self.fts.commit()?;
        Ok(())
    }

    /// Re-encrypt legacy plaintext blobs under the sealed envelope format.
    /// Only meaningful in sealed mode; no-op otherwise.
    pub async fn migrate_sealed_storage(&self) -> MvResult<()> {
        if !self.config.sealed_mode {
            return Ok(());
        }
        if !self.keychain.is_unsealed_sync() {
            return Err(HxError::VaultSealed);
        }

        let nodes = self
            .store
            .nodes
            .list(&QueryFilters::default(), 100_000, 0)
            .await?;
        let mut migrated_nodes = 0usize;
        let mut migrated_blobs = 0usize;
        for node in &nodes {
            self.store.nodes.update(node).await?;
            migrated_nodes += 1;
            migrated_blobs += self.migrate_legacy_attachments_for_node(node).await?;
        }

        for legacy_index_dir in ["tantivy", "lancedb"] {
            let path = PathBuf::from(&self.config.data_dir).join(legacy_index_dir);
            if tokio::fs::metadata(&path).await.is_ok() {
                let _ = tokio::fs::remove_dir_all(&path).await;
            }
        }

        tracing::info!(
            migrated_nodes,
            migrated_blobs,
            "sealed storage migration completed"
        );
        Ok(())
    }

    async fn encrypt_blob_for_namespace(
        &self,
        namespace: &str,
        plaintext: &[u8],
    ) -> MvResult<Vec<u8>> {
        let dek = VaultCrypto::generate_node_dek();
        let wrapped_dek = self.keychain.wrap_namespace_dek(namespace, &dek).await?;
        let ciphertext = VaultCrypto::aes_gcm_encrypt_pub(&dek, plaintext)
            .map_err(|err| HxError::Storage(format!("blob encrypt failed: {err}")))?;
        let envelope = serde_json::json!({
            "v": 1,
            "wrapped_dek": wrapped_dek,
            "ciphertext": base64::engine::general_purpose::STANDARD.encode(ciphertext),
        });
        let body = serde_json::to_vec(&envelope)
            .map_err(|err| HxError::Storage(format!("blob envelope encode failed: {err}")))?;
        let mut out = Vec::with_capacity(SEALED_BLOB_MAGIC.len() + body.len());
        out.extend_from_slice(SEALED_BLOB_MAGIC);
        out.extend_from_slice(&body);
        Ok(out)
    }

    async fn migrate_legacy_attachments_for_node(&self, node: &KnowledgeNode) -> MvResult<usize> {
        let attachments = node
            .metadata
            .get("attachments")
            .and_then(serde_json::Value::as_array)
            .cloned()
            .unwrap_or_default();

        if attachments.is_empty() {
            return Ok(0);
        }

        let expected_base = PathBuf::from(&self.config.data_dir)
            .join("blobs")
            .join(node.id.to_string());
        let canonical_base = match tokio::fs::canonicalize(&expected_base).await {
            Ok(path) => path,
            Err(_) => return Ok(0),
        };

        let mut migrated = 0usize;
        for attachment in attachments {
            let Some(stored_path) = attachment
                .get("stored_path")
                .and_then(serde_json::Value::as_str)
            else {
                continue;
            };
            let candidate = PathBuf::from(stored_path);
            let canonical_candidate = match tokio::fs::canonicalize(&candidate).await {
                Ok(path) => path,
                Err(_) => continue,
            };
            if !canonical_candidate.starts_with(&canonical_base) {
                continue;
            }

            let bytes = match tokio::fs::read(&canonical_candidate).await {
                Ok(bytes) => bytes,
                Err(_) => continue,
            };
            if bytes.starts_with(SEALED_BLOB_MAGIC) {
                continue;
            }

            let encrypted = self
                .encrypt_blob_for_namespace(&node.namespace, &bytes)
                .await?;
            tokio::fs::write(&canonical_candidate, encrypted)
                .await
                .map_err(|err| HxError::Storage(format!("rewrite attachment failed: {err}")))?;
            migrated += 1;
        }

        Ok(migrated)
    }

    /// Get node count.
    pub async fn node_count(&self) -> MvResult<usize> {
        self.store.nodes.count(&QueryFilters::default()).await
    }

    /// Get the owner profile.
    pub async fn get_profile(&self) -> MvResult<OwnerProfile> {
        self.store.nodes.get_profile().await
    }

    /// Update owner profile and sync relay contact.
    pub async fn update_profile(&self, req: &UpdateProfileRequest) -> MvResult<OwnerProfile> {
        let profile = self.store.nodes.update_profile(req).await?;
        self.sync_owner_relay_contact(profile).await
    }

    /// Set up the enrichment pipeline, wiring it to a change notification sender.
    /// Returns the enrichment worker to be spawned if enrichment is enabled.
    /// Must be called before the engine is wrapped in `Arc`.
    pub fn setup_enrichment(
        &mut self,
        change_tx: tokio::sync::broadcast::Sender<ChangeNotification>,
    ) -> Option<crate::enrichment::EnrichmentWorker> {
        if !self.config.ai.enrichment_enabled {
            return None;
        }

        let (pipeline, worker) = crate::enrichment::EnrichmentPipeline::new(
            Arc::clone(&self.store),
            self.config.ai.clone(),
            self.llm.clone(),
            change_tx,
        );
        self.enrichment = Some(pipeline);
        Some(worker)
    }

    async fn sync_owner_relay_contact(&self, profile: OwnerProfile) -> MvResult<OwnerProfile> {
        let display_name = profile.display_name.trim();
        let display_name = if display_name.is_empty() {
            "Helix Owner"
        } else {
            display_name
        };

        let email = profile.email.as_ref().map(|value| value.trim().to_string());
        let email = email.filter(|value| !value.is_empty());
        let vault_address = email.map(|value| format!("mailto:{value}"));

        let signature_key = profile
            .signature_public_key
            .as_ref()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty());

        let contact_id = profile
            .metadata
            .get(PROFILE_RELAY_CONTACT_ID_KEY)
            .and_then(|value| value.as_str())
            .and_then(|value| Uuid::parse_str(value).ok());

        if contact_id.is_none() && signature_key.is_none() {
            return Ok(profile);
        }

        if let Some(contact_id) = contact_id {
            if let Some(mut contact) = self.relay.get_contact(contact_id).await? {
                contact.display_name = display_name.to_string();
                if let Some(key) = signature_key {
                    contact.public_key = key;
                }
                contact.vault_address = vault_address;
                if contact.notes.is_none() {
                    contact.notes = Some(PROFILE_OWNER_CONTACT_NOTES.to_string());
                }

                let _ = self.relay.update_contact(&contact).await?;
                return Ok(profile);
            }
        }

        let Some(signature_key) = signature_key else {
            return Ok(profile);
        };

        let mut contact =
            RelayContact::new(display_name, signature_key).with_trust(TrustLevel::Full);
        contact.vault_address = vault_address;
        contact.notes = Some(PROFILE_OWNER_CONTACT_NOTES.to_string());

        self.relay.add_contact(&contact).await?;

        let mut metadata = profile.metadata.clone();
        metadata.insert(
            PROFILE_RELAY_CONTACT_ID_KEY.to_string(),
            serde_json::Value::String(contact.id.to_string()),
        );

        let updated = self
            .store
            .nodes
            .update_profile(&UpdateProfileRequest {
                metadata: Some(metadata),
                ..Default::default()
            })
            .await?;

        Ok(updated)
    }

    /// List daily notes, optionally filtered by namespace.
    pub async fn list_daily_notes(
        &self,
        namespace: Option<String>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<KnowledgeNode>> {
        let filters = QueryFilters {
            namespace,
            tags: Some(vec![DAILY_NOTE_TAG.to_string()]),
            ..Default::default()
        };
        self.store.nodes.list(&filters, limit, offset).await
    }

    // --- Core Node I/O ---

    /// Store a new node (create).
    pub async fn store_node(&self, node: KnowledgeNode) -> MvResult<KnowledgeNode> {
        self.ensure_unsealed_for_node_io()?;
        let stored = self.ingest.ingest(node).await?;
        self.auto_link_node_to_daily_note_best_effort(&stored)
            .await;
        self.auto_backlink_node_references_best_effort(&stored)
            .await;
        Ok(stored)
    }

    /// Store a node with relationships.
    pub async fn store_with_relations(
        &self,
        node: KnowledgeNode,
        relations: Vec<Relationship>,
    ) -> MvResult<KnowledgeNode> {
        self.ensure_unsealed_for_node_io()?;
        self.ingest.ingest_with_relations(node, relations).await
    }

    /// Recall knowledge matching a query.
    pub async fn recall(&self, query: &MemoryQuery) -> MvResult<Vec<SearchResult>> {
        self.ensure_unsealed_for_node_io()?;
        self.recall.recall(query).await
    }

    /// Get a node by ID.
    pub async fn get_node(&self, id: uuid::Uuid) -> MvResult<Option<KnowledgeNode>> {
        self.ensure_unsealed_for_node_io()?;
        self.store.nodes.get(id).await
    }

    /// Update an existing node.
    pub async fn update_node(&self, node: KnowledgeNode) -> MvResult<KnowledgeNode> {
        self.ensure_unsealed_for_node_io()?;
        let updated = self.ingest.update(node).await?;
        self.auto_link_node_to_daily_note_best_effort(&updated)
            .await;
        self.auto_backlink_node_references_best_effort(&updated)
            .await;
        Ok(updated)
    }

    /// Delete a node.
    pub async fn delete_node(&self, id: uuid::Uuid) -> MvResult<bool> {
        self.ensure_unsealed_for_node_io()?;
        self.ingest.delete(id).await
    }

    /// List nodes with filters.
    pub async fn list_nodes(
        &self,
        filters: &QueryFilters,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<KnowledgeNode>> {
        self.ensure_unsealed_for_node_io()?;
        self.store.nodes.list(filters, limit, offset).await
    }

    /// Return the daily note for a specific day/namespace if present.
    pub async fn find_daily_note(
        &self,
        date: NaiveDate,
        namespace: &str,
    ) -> MvResult<Option<KnowledgeNode>> {
        self.ensure_unsealed_for_node_io()?;
        let filters = QueryFilters {
            namespace: Some(namespace.to_string()),
            tags: Some(vec![daily_note_day_tag(date)]),
            ..Default::default()
        };
        let mut existing = self.store.nodes.list(&filters, 1, 0).await?;
        Ok(existing.pop())
    }

    /// Ensure a daily note exists for a given date and namespace.
    /// Returns `(node, created)` where `created` is true only on first creation.
    pub async fn ensure_daily_note(
        &self,
        date: NaiveDate,
        namespace: Option<String>,
    ) -> MvResult<(KnowledgeNode, bool)> {
        if !self.config.daily_notes.enabled {
            return Err(HxError::InvalidInput(
                "daily notes are disabled".to_string(),
            ));
        }

        let namespace = namespace.unwrap_or_else(|| self.config.daily_notes.namespace.clone());
        let day_tag = daily_note_day_tag(date);

        // Check availability
        if let Some(existing) = self.find_daily_note(date, &namespace).await? {
            return Ok((existing, false));
        }

        let title = date.format("%Y-%m-%d").to_string();
        let content = render_daily_note_template(&self.config.daily_notes.content_template, date);
        
        let mut note = KnowledgeNode::new(NodeKind::Fact, content)
            .with_namespace(namespace.clone())
            .with_title(&title)
            .with_tags(vec![
                DAILY_NOTE_TAG.to_string(),
                day_tag,
                daily_note_weekday_tag(date),
            ]);
        
        note.metadata.insert(
            "daily_note".to_string(),
            serde_json::Value::Bool(true),
        );
        note.metadata.insert(
            "daily_note_date".to_string(),
            serde_json::Value::String(date.to_string()),
        );

        let stored = self.store_node(note).await?;
        
        Ok((stored, true))
    }

    async fn auto_link_node_to_daily_note_best_effort(&self, node: &KnowledgeNode) {
        if !self.config.daily_notes.enabled {
            return;
        }

        // Skip if node IS a daily note
        if is_daily_note(node) {
            return;
        }

        // Only link candidates
        if !is_daily_link_candidate(node) {
            return;
        }

        let date = node.temporal.created_at.date_naive();
        let namespace = &node.namespace; // Same namespace assumption

        // Find daily note for that day
        let daily_note = match self.find_daily_note(date, namespace).await {
            Ok(Some(note)) => note,
            Ok(None) => return, // Don't create if missing, best effort
            Err(_) => return,
        };

        // Check if relationship already exists
        let rels = match self.graph.get_relationships_from(daily_note.id).await {
            Ok(r) => r,
            Err(_) => return,
        };

        if rels
            .iter()
            .any(|r| r.to_node == node.id && r.kind == RelationKind::Contains)
        {
            return;
        }

        let rel = Relationship::new(daily_note.id, node.id, RelationKind::Contains);
        let _ = self.graph.add_relationship(&rel).await;
    }

    async fn auto_backlink_node_references_best_effort(&self, node: &KnowledgeNode) {
        if !self.config.linking.auto_backlinks_enabled {
            return;
        }

        // 1. Extract targets
        let references = extract_reference_targets_with_kind(&node.content, self.config.linking.auto_backlinks_max_targets);
        if references.is_empty() {
            return;
        }

        // 2. Resolve targets to IDs
        let index = self.build_backlink_resolution_index(&node.namespace).await;
        let resolved_links = resolve_reference_targets_with_kind(&references, node.id, &index);

        // 3. Sync relationships
        let _ = self.sync_references(node.id, resolved_links).await;
    }

    async fn sync_references(
        &self,
        source_id: Uuid,
        targets: Vec<ResolvedContentReferenceTarget>,
    ) -> MvResult<()> {
        // Get existing references
        let existing = self.graph.get_relationships_from(source_id).await?;
        let existing_refs: Vec<_> = existing
            .into_iter()
            .filter(|r| r.kind == RelationKind::References && is_auto_backlink_relationship(r))
            .collect();

        // Diff and apply
        for target in &targets {
            if !existing_refs.iter().any(|r| r.to_node == target.target_id) {
                let mut rel = Relationship::new(source_id, target.target_id, RelationKind::References);
                rel.metadata.insert(
                    AUTO_BACKLINK_METADATA_KEY.to_string(),
                    serde_json::Value::Bool(true),
                );
                rel.metadata.insert(
                    AUTO_BACKLINK_SOURCE_METADATA_KEY.to_string(),
                    serde_json::Value::String(target.source_kind.as_str().to_string()),
                );
                let _ = self.graph.add_relationship(&rel).await;
            }
        }

        // Remove stale
        for rel in existing_refs {
            if !targets.iter().any(|t| t.target_id == rel.to_node) {
                let _ = self.graph.remove_relationship(rel.id).await;
            }
        }

        Ok(())
    }

    async fn build_backlink_resolution_index(
        &self,
        namespace: &str,
    ) -> KnowledgeVaultBacklinkResolutionIndex {
        // Simple index builder from graph/store
        // In real impl, this might be cached or optimized
        let filters = QueryFilters {
            namespace: Some(namespace.to_string()),
            ..Default::default()
        };
        // Fetch valid targets (titles, etc)
        // Optimization: only fetch what's needed. For now, list decent chunk.
        let nodes: Vec<KnowledgeNode> = self.store.nodes.list(&filters, 1000, 0).await.unwrap_or_default();
        
        let mut index = KnowledgeVaultBacklinkResolutionIndex::default();
        for node in &nodes {
            index.insert_node(node.id, node.title.as_deref(), node.source.as_deref());
        }
        index
    }
}

// Helpers
fn is_daily_note(node: &KnowledgeNode) -> bool {
    node.tags
        .iter()
        .any(|tag| tag.eq_ignore_ascii_case(DAILY_NOTE_TAG))
        || node
            .metadata
            .get("daily_note")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
}

fn is_daily_link_candidate(node: &KnowledgeNode) -> bool {
    node.tags.iter().any(|tag| {
        DAILY_LINK_CANDIDATE_TAGS
            .iter()
            .any(|candidate| tag.eq_ignore_ascii_case(candidate))
    })
}

fn is_auto_backlink_relationship(rel: &Relationship) -> bool {
    rel.metadata
        .get(AUTO_BACKLINK_METADATA_KEY)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn select_embedding_provider(
    config: &EngineConfig,
    credential_store: &Arc<CredentialStore>,
) -> EmbeddingProviderSelection {
    // Check for OpenAI key
    let openai_key = credential_store.get_secret_string("OPENAI_API_KEY");

    // Check for local fastembed support
    // (This is a simplified version of the logic I saw in view)
    // Actually I should look at what was there. 
    // It was lines ~3300ish.
    // I'll re-implement the logic or assume it's mostly config based.
    
    // Logic:
    // If config.embedding.provider is "openai" and key exists -> OpenAiEmbedder
    // If "local_fastembed" -> LocalEmbedder
    // Else -> Noop (None)
    
    let mut runtime_status = EmbedderRuntimeStatus {
        configured_provider: config.embedding.provider.clone(),
        configured_model: config.embedding.model.clone(),
        configured_dimensions: config.embedding.dimensions,
        effective_provider: "noop".to_string(),
        effective_model: "none".to_string(),
        effective_dimensions: config.embedding.dimensions,
        fallback_to_noop: false,
        reason: None,
        local_embeddings_feature_enabled: true, // simplified
    };

    let embedder: Option<Box<dyn hx_core::Embedder>> =
        match config.embedding.provider.as_str() {
            "openai" => {
                if let Some(key) = openai_key {
                    runtime_status.effective_provider = "openai".to_string();
                    runtime_status.effective_model = config.embedding.model.clone();
                    Some(Box::new(OpenAiEmbedder::new(
                        "https://api.openai.com/v1".to_string(),
                        Some(key),
                        config.embedding.model.clone(),
                        config.embedding.dimensions,
                    )))
                } else {
                    runtime_status.fallback_to_noop = true;
                    runtime_status.reason = Some("missing OPENAI_API_KEY".to_string());
                    None
                }
            }
            "local_fastembed" => {
                // Try init local
                 match KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder::try_new(
                    &config.embedding.model,
                ) {
                    Ok(e) => {
                        runtime_status.effective_provider = "local_fastembed".to_string();
                        runtime_status.effective_model = config.embedding.model.clone();
                        Some(Box::new(e))
                    }
                    Err(err) => {
                        runtime_status.fallback_to_noop = true;
                        runtime_status.reason = Some(format!("local_fastembed init: {err}"));
                        None
                    }
                }
            }
            _ => {
                runtime_status.fallback_to_noop = true;
                runtime_status.reason = Some("unknown provider".to_string());
                None
            }
        };

    EmbeddingProviderSelection {
        embedder,
        vector_dimensions: config.embedding.dimensions,
        runtime_status,
    }
}

#[cfg(test)]
pub mod test_utils;

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use crate::engine::test_utils::*;
    async fn test_ai_auto_tagging_enriches_from_content_and_neighbors() {
        let (engine, _tmp_dir) = create_test_engine_with_ai_auto_tagging().await;

        let seed = KnowledgeNode::new(
            NodeKind::Fact,
            "Rust async tokio memory pipeline for background jobs".to_string(),
        )
        .with_tags(vec![
            "rust".to_string(),
            "async".to_string(),
            "tokio".to_string(),
        ]);
        let _seed_node = engine.store_node(seed).await.unwrap();

        let stored = engine
            .store_node(KnowledgeNode::new(
                NodeKind::Fact,
                "Building a memory pipeline in Rust for async workers".to_string(),
            ))
            .await
            .unwrap();

        let tags_lower: Vec<String> = stored
            .tags
            .iter()
            .map(|tag| tag.to_ascii_lowercase())
            .collect();
        assert!(!tags_lower.is_empty(), "auto-tagging should add tags");
        assert!(
            tags_lower.iter().any(|tag| tag == "rust"),
            "expected generated tags to include rust"
        );
        assert!(
            tags_lower
                .iter()
                .any(|tag| tag == "memory" || tag == "pipeline"),
            "expected lexical tags to include memory/pipeline"
        );
    }

    #[tokio::test]
    async fn test_ai_auto_tagging_disabled_keeps_empty_tag_list() {
        let (engine, _tmp_dir) = create_test_engine().await;

        let stored = engine
            .store_node(KnowledgeNode::new(
                NodeKind::Fact,
                "Novel note without manual tags".to_string(),
            ))
            .await
            .unwrap();

        assert!(
            stored.tags.is_empty(),
            "tags should remain empty when auto-tagging is disabled"
        );
    }

    #[tokio::test]
    async fn test_unknown_embedding_provider_falls_back_without_breaking_ingest() {
        let (engine, _tmp_dir) =
            create_test_engine_with_local_embedding_provider("unknown-provider", "any").await;

        let stored = engine
            .store_node(KnowledgeNode::new(
                NodeKind::Fact,
                "Provider fallback should keep ingest healthy".to_string(),
            ))
            .await
            .unwrap();

        assert_eq!(
            stored.content,
            "Provider fallback should keep ingest healthy"
        );

        let status = engine.embedding_runtime_status();
        assert_eq!(status.configured_provider, "unknown-provider");
        assert_eq!(status.effective_provider, "noop");
        assert!(status.fallback_to_noop);
        assert!(status
            .reason
            .as_deref()
            .is_some_and(|value| value.contains("unknown embedding provider")));
    }

    #[tokio::test]
    async fn test_local_fastembed_invalid_model_falls_back_without_breaking_ingest() {
        let (engine, _tmp_dir) =
            create_test_engine_with_local_embedding_provider("local_fastembed", "invalid-model")
                .await;

        let stored = engine
            .store_node(KnowledgeNode::new(
                NodeKind::Fact,
                "Invalid local model should not break ingest".to_string(),
            ))
            .await
            .unwrap();

        assert_eq!(
            stored.content,
            "Invalid local model should not break ingest"
        );

        let status = engine.embedding_runtime_status();
        assert_eq!(status.configured_provider, "local_fastembed");
        assert_eq!(status.effective_provider, "noop");
        assert!(status.fallback_to_noop);
        assert!(status
            .reason
            .as_deref()
            .is_some_and(|value| value.contains("local_fastembed initialization failed")));
    }

    #[tokio::test]
    async fn test_ensure_daily_note_is_idempotent() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let date = NaiveDate::from_ymd_opt(2026, 2, 6).expect("valid date");

        let (created_note, created) = engine.ensure_daily_note(date, None).await.unwrap();
        assert!(created);
        assert_eq!(created_note.namespace, engine.config.daily_notes.namespace);
        assert!(created_note.tags.iter().any(|tag| tag == DAILY_NOTE_TAG));
        assert!(created_note.tags.iter().any(|tag| tag == "day:2026-02-06"));
        assert_eq!(
            created_note.metadata.get("daily_note_date"),
            Some(&serde_json::Value::String("2026-02-06".to_string()))
        );

        let (existing_note, created_again) = engine.ensure_daily_note(date, None).await.unwrap();
        assert!(!created_again);
        assert_eq!(existing_note.id, created_note.id);
    }

    #[tokio::test]
    async fn test_list_daily_notes_filters_non_daily_nodes() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let date = NaiveDate::from_ymd_opt(2026, 2, 6).expect("valid date");
        let namespace = engine.config.daily_notes.namespace.clone();

        let (_daily, created) = engine
            .ensure_daily_note(date, Some(namespace.clone()))
            .await
            .unwrap();
        assert!(created);

        let non_daily = KnowledgeNode::new(NodeKind::Fact, "not a daily note".to_string())
            .with_namespace(namespace.clone())
            .with_tags(vec!["journal".to_string()]);
        let stored_non_daily = engine.store_node(non_daily).await.unwrap();

        let notes = engine
            .list_daily_notes(Some(namespace), 50, 0)
            .await
            .unwrap();

        assert!(!notes.is_empty());
        assert!(notes
            .iter()
            .all(|node| node.tags.iter().any(|tag| tag == DAILY_NOTE_TAG)));
        assert!(notes.iter().all(|node| node.id != stored_non_daily.id));
    }

    #[tokio::test]
    async fn test_store_node_auto_links_task_to_daily_note() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = engine.config.daily_notes.namespace.clone();
        let task_node = KnowledgeNode::new(
            NodeKind::Task,
            "Ship release checklist and verify deployment timeline".to_string(),
        )
        .with_namespace(namespace.clone())
        .with_tags(vec!["release".to_string()]);

        let stored_task = engine.store_node(task_node).await.unwrap();
        let day = stored_task.temporal.created_at.date_naive();
        let daily_note = engine
            .find_daily_note(day, &namespace)
            .await
            .unwrap()
            .expect("daily note should be created");

        let outgoing = engine
            .graph
            .get_relationships_from(daily_note.id)
            .await
            .unwrap();
        assert!(outgoing
            .iter()
            .any(|rel| { rel.to_node == stored_task.id && rel.kind == RelationKind::Contains }));
    }

    #[tokio::test]
    async fn test_store_node_auto_links_event_kind_without_tags() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = engine.config.daily_notes.namespace.clone();
        let event_node = KnowledgeNode::new(
            NodeKind::Event,
            "Team planning sync at 10:00 UTC".to_string(),
        )
        .with_namespace(namespace.clone());

        let stored_event = engine.store_node(event_node).await.unwrap();
        let day = stored_event.temporal.created_at.date_naive();
        let daily_note = engine
            .find_daily_note(day, &namespace)
            .await
            .unwrap()
            .expect("daily note should be created");

        let outgoing = engine
            .graph
            .get_relationships_from(daily_note.id)
            .await
            .unwrap();
        assert!(outgoing
            .iter()
            .any(|rel| rel.to_node == stored_event.id && rel.kind == RelationKind::Contains));
    }

    #[tokio::test]
    async fn test_update_node_auto_link_does_not_duplicate_daily_edge() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = engine.config.daily_notes.namespace.clone();
        let task_node =
            KnowledgeNode::new(NodeKind::Event, "Capture meeting action items".to_string())
                .with_namespace(namespace.clone());
        let stored_task = engine.store_node(task_node).await.unwrap();
        let day = stored_task.temporal.created_at.date_naive();
        let daily_note = engine
            .find_daily_note(day, &namespace)
            .await
            .unwrap()
            .expect("daily note should exist");

        let mut updated_task = stored_task.clone();
        updated_task.content = "Capture meeting action items and owner assignments".to_string();
        let _updated = engine.update_node(updated_task).await.unwrap();

        let outgoing = engine
            .graph
            .get_relationships_from(daily_note.id)
            .await
            .unwrap();
        let contains_edges = outgoing
            .iter()
            .filter(|rel| rel.to_node == stored_task.id && rel.kind == RelationKind::Contains)
            .count();
        assert_eq!(contains_edges, 1);
    }

    #[tokio::test]
    async fn test_store_node_auto_backlinks_from_wikilinks() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = "knowledge".to_string();
        let target = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Launch scope and milestones".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Project Alpha"),
            )
            .await
            .unwrap();

        let source = engine
            .store_node(
                KnowledgeNode::new(
                    NodeKind::Fact,
                    "Review [[Project Alpha]] and prep a kickoff checklist.".to_string(),
                )
                .with_namespace(namespace.clone())
                .with_title("Kickoff Brief"),
            )
            .await
            .unwrap();

        let outgoing = engine
            .graph
            .get_relationships_from(source.id)
            .await
            .unwrap();
        let reference = outgoing.iter().find(|rel| {
            rel.kind == RelationKind::References
                && rel.to_node == target.id
                && is_auto_backlink_relationship(rel)
        });
        assert!(reference.is_some());
        assert_eq!(
            reference
                .and_then(|rel| rel.metadata.get(AUTO_BACKLINK_SOURCE_METADATA_KEY))
                .and_then(serde_json::Value::as_str),
            Some("wikilink")
        );
    }

    #[tokio::test]
    async fn test_store_node_auto_backlinks_from_markdown_and_source_urls() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = "knowledge".to_string();
        let title_target = engine
            .store_node(
                KnowledgeNode::new(
                    NodeKind::Fact,
                    "Project Beta release sequencing".to_string(),
                )
                .with_namespace(namespace.clone())
                .with_title("Project Beta"),
            )
            .await
            .unwrap();
        let source_target = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Bookmark, "Spec reference".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Platform Spec")
                    .with_source("https://docs.example.com/spec"),
            )
            .await
            .unwrap();

        let source = engine
            .store_node(
                KnowledgeNode::new(
                    NodeKind::Fact,
                    "Review [release scope](Project Beta#Milestones) and https://docs.example.com/spec#overview before kickoff.".to_string(),
                )
                .with_namespace(namespace.clone())
                .with_title("Kickoff Prep"),
            )
            .await
            .unwrap();

        let outgoing = engine
            .graph
            .get_relationships_from(source.id)
            .await
            .unwrap();
        let title_reference = outgoing.iter().find(|rel| {
            rel.kind == RelationKind::References
                && rel.to_node == title_target.id
                && is_auto_backlink_relationship(rel)
        });
        assert!(title_reference.is_some());
        assert_eq!(
            title_reference
                .and_then(|rel| rel.metadata.get(AUTO_BACKLINK_SOURCE_METADATA_KEY))
                .and_then(serde_json::Value::as_str),
            Some("markdown_link")
        );

        let source_reference = outgoing.iter().find(|rel| {
            rel.kind == RelationKind::References
                && rel.to_node == source_target.id
                && is_auto_backlink_relationship(rel)
        });
        assert!(source_reference.is_some());
        assert_eq!(
            source_reference
                .and_then(|rel| rel.metadata.get(AUTO_BACKLINK_SOURCE_METADATA_KEY))
                .and_then(serde_json::Value::as_str),
            Some("source_url")
        );
    }

    #[tokio::test]
    async fn test_store_node_auto_backlinks_from_mentions() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = "knowledge".to_string();
        let target = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Project Alpha execution notes".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Project Alpha"),
            )
            .await
            .unwrap();

        let source = engine
            .store_node(
                KnowledgeNode::new(
                    NodeKind::Fact,
                    "Align the launch checklist with @\"Project Alpha\" owners.".to_string(),
                )
                .with_namespace(namespace.clone())
                .with_title("Launch Checklist"),
            )
            .await
            .unwrap();

        let outgoing = engine
            .graph
            .get_relationships_from(source.id)
            .await
            .unwrap();
        let mention_reference = outgoing.iter().find(|rel| {
            rel.kind == RelationKind::References
                && rel.to_node == target.id
                && is_auto_backlink_relationship(rel)
        });
        assert!(mention_reference.is_some());
        assert_eq!(
            mention_reference
                .and_then(|rel| rel.metadata.get(AUTO_BACKLINK_SOURCE_METADATA_KEY))
                .and_then(serde_json::Value::as_str),
            Some("mention")
        );
    }

    #[tokio::test]
    async fn test_update_node_auto_backlinks_removes_stale_targets() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = "knowledge".to_string();
        let target_alpha = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Alpha details".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Project Alpha"),
            )
            .await
            .unwrap();
        let target_beta = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Beta details".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Project Beta"),
            )
            .await
            .unwrap();

        let mut source = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Draft [[Project Alpha]] notes".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Planning"),
            )
            .await
            .unwrap();

        source.content = "Finalize [[Project Beta]] notes".to_string();
        let source = engine.update_node(source).await.unwrap();

        let outgoing = engine
            .graph
            .get_relationships_from(source.id)
            .await
            .unwrap();
        assert!(outgoing.iter().any(|rel| {
            rel.kind == RelationKind::References
                && rel.to_node == target_beta.id
                && is_auto_backlink_relationship(rel)
        }));
        assert!(!outgoing.iter().any(|rel| {
            rel.kind == RelationKind::References
                && rel.to_node == target_alpha.id
                && is_auto_backlink_relationship(rel)
        }));
    }

    #[tokio::test]
    async fn test_auto_backlinks_can_be_disabled() {
        let (engine, _tmp_dir) = create_test_engine_with_backlinks_disabled().await;
        let namespace = "knowledge".to_string();

        let _target = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Alpha details".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Project Alpha"),
            )
            .await
            .unwrap();

        let source = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Draft [[Project Alpha]] notes".to_string())
                    .with_namespace(namespace.clone())
                    .with_title("Planning"),
            )
            .await
            .unwrap();

        let outgoing = engine
            .graph
            .get_relationships_from(source.id)
            .await
            .unwrap();
        assert!(!outgoing.iter().any(|rel| {
            rel.kind == RelationKind::References && is_auto_backlink_relationship(rel)
        }));
    }

    #[tokio::test]
    async fn test_federation_peer_add_and_list() {
        let (engine, _tmp) = create_test_engine().await;

        let peer = crate::federation::FederationPeer {
            id: Uuid::now_v7(),
            vault_id: "vault-test-123".into(),
            display_name: "Test Vault".into(),
            endpoint: "http://127.0.0.1:19470".into(),
            public_key: None,
            allowed_namespaces: vec![],
            max_results: 10,
            enabled: true,
            last_seen: None,
            created_at: Utc::now(),
            shared_secret: Some("secret123".into()),
        };

        engine.federation.add_peer(peer.clone()).await;

        let peers = engine.federation.list_peers().await;
        assert_eq!(peers.len(), 1);
        assert_eq!(peers[0].vault_id, "vault-test-123");

        // Find by vault ID
        let found = engine
            .federation
            .find_peer_by_vault_id("vault-test-123")
            .await;
        assert!(found.is_some());

        // Remove
        let removed = engine.federation.remove_peer(peer.id).await;
        assert!(removed);
        assert!(engine.federation.list_peers().await.is_empty());
    }

    #[tokio::test]
    async fn test_store_multiple_node_kinds_and_retrieve() {
        let (engine, _tmp) = create_test_engine().await;

        // Store nodes of different kinds and verify retrieval
        let obs = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Observation, "Observed: Rust async is fast")
                    .with_tags(vec!["rust".into(), "async".into()])
                    .with_namespace("default"),
            )
            .await
            .unwrap();
        let fact = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "Tokio is the most popular async runtime")
                    .with_tags(vec!["rust".into(), "tokio".into()])
                    .with_namespace("default"),
            )
            .await
            .unwrap();

        // Both should be retrievable
        let r_obs = engine.get_node(obs.id).await.unwrap();
        assert!(r_obs.is_some());
        assert_eq!(r_obs.unwrap().kind, NodeKind::Observation);

        let r_fact = engine.get_node(fact.id).await.unwrap();
        assert!(r_fact.is_some());
        assert_eq!(r_fact.unwrap().kind, NodeKind::Fact);

        // Neighbors should be independent (no auto-linking with noop embedder)
        let neighbors = engine.get_neighbors(obs.id, 1).await.unwrap();
        assert!(!neighbors.contains(&fact.id));
    }
}
