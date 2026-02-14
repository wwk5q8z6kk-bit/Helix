use std::cmp::Ordering;
use std::path::PathBuf;
use std::sync::Arc;

use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, NaiveDate, Utc};
use hx_core::credentials::CredentialStore;
use hx_core::*;
use hx_graph::store::SqliteGraphStore;
use hx_index::tantivy_index::TantivyFullTextIndex;
use hx_storage::sealed_runtime::{clear_runtime_root_key, set_sealed_mode_enabled};
use hx_storage::unified::UnifiedStore;
use hx_storage::vault_crypto::VaultCrypto;
use hx_storage::vector::{KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder, OpenAiEmbedder};
use rand::RngCore;
use reqwest::StatusCode as HttpStatusCode;
use sha2::{Digest, Sha256};
use url::form_urlencoded::byte_serialize;
use uuid::Uuid;

use crate::backlinks::{
    extract_reference_targets_with_kind, resolve_reference_targets_with_kind,
    ContentReferenceSourceKind, KnowledgeVaultBacklinkResolutionIndex,
    ResolvedContentReferenceTarget,
};
use crate::config::EngineConfig;
use crate::daily_notes::{daily_note_day_tag, daily_note_weekday_tag, render_daily_note_template};
use crate::ingest::IngestPipeline;
use crate::llm::{self, LlmProvider};
use crate::recall::RecallPipeline;
use crate::recurrence::{
    collect_due_occurrences, parse_optional_metadata_bool, parse_optional_metadata_datetime,
    parse_optional_metadata_u64, parse_task_recurrence_rule, previous_due_at,
    RECURRING_DUE_AT_METADATA_KEY, RECURRING_INSTANCE_METADATA_KEY,
    RECURRING_PARENT_ID_METADATA_KEY, TASK_COMPLETED_METADATA_KEY, TASK_DUE_AT_METADATA_KEY,
    TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY, TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY,
    TASK_RECURRENCE_METADATA_KEY, TASK_REMINDER_SENT_AT_METADATA_KEY,
    TASK_REMINDER_STATUS_METADATA_KEY,
};

const DAILY_NOTE_TAG: &str = "daily-note";
const PROFILE_RELAY_CONTACT_ID_KEY: &str = "relay_contact_id";
const PROFILE_OWNER_CONTACT_NOTES: &str = "owner";
const DAILY_LINK_CANDIDATE_TAGS: &[&str] = &[
    "task", "tasks", "todo", "to-do", "event", "events", "meeting", "reminder",
];
const AUTO_BACKLINK_METADATA_KEY: &str = "auto_backlink";
const AUTO_BACKLINK_SOURCE_METADATA_KEY: &str = "source";
const TASK_AI_PRIORITY_METADATA_KEY: &str = "ai_priority";
const TASK_PRIORITY_METADATA_KEY: &str = "task_priority";
const TASK_PRIORITY_ALT_METADATA_KEY: &str = "priority";
const TASK_STATUS_METADATA_KEY: &str = "task_status";
const TASK_STATUS_ALT_METADATA_KEY: &str = "status";
const TASK_ESTIMATE_MINUTES_METADATA_KEY: &str = "task_estimate_minutes";
const TASK_ESTIMATE_MINUTES_ALT_METADATA_KEY: &str = "task_estimate_min";
const TASK_ESTIMATE_MIN_METADATA_KEY: &str = "estimate_min";
const SEALED_BLOB_MAGIC: &[u8; 4] = b"HXB1";

#[derive(Debug, Clone, serde::Serialize)]
pub struct KnowledgeVaultIndexNoteEmbeddingProviderRuntimeStatus {
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

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TaskRecurrenceRollforwardStats {
    pub scanned_tasks: usize,
    pub recurring_templates: usize,
    pub generated_instances: usize,
    pub updated_templates: usize,
    pub errors: usize,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TaskReminderDispatchStats {
    pub scanned_tasks: usize,
    pub due_tasks: usize,
    pub reminders_marked_sent: usize,
    pub errors: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GoogleCalendarSyncReport {
    pub calendar_id: String,
    pub fetched: usize,
    pub created: usize,
    pub updated: usize,
    pub deleted: usize,
    pub skipped: usize,
    pub exported_created: usize,
    pub exported_updated: usize,
    pub next_sync_token: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct RelayInboundOutcome {
    pub message: RelayMessage,
    pub auto_reply: Option<RelayMessage>,
    pub proposal_id: Option<Uuid>,
}

#[derive(Debug, Clone)]
struct RelayReplySuggestion {
    content: String,
    confidence: f32,
    context_snippets: Vec<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct PrioritizedTask {
    pub task: KnowledgeNode,
    pub score: f64,
    pub rank: usize,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct TaskPrioritizationOptions {
    pub namespace: Option<String>,
    pub limit: usize,
    pub include_completed: bool,
    pub include_without_due: bool,
    pub persist: bool,
    pub now: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct TaskPriorityCandidate {
    task: KnowledgeNode,
    score: f64,
    reason: String,
    due_at: Option<DateTime<Utc>>,
}

struct EmbeddingProviderSelection {
    embedder: Option<Arc<dyn Embedder>>,
    vector_dimensions: usize,
    runtime_status: KnowledgeVaultIndexNoteEmbeddingProviderRuntimeStatus,
}

/// The Helix engine — orchestrates storage, indexing, and search.
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
    pub multimodal: crate::multimodal::MultiModalPipeline,
    pub sync: crate::sync::SyncEngine,
    pub federation: crate::federation::FederationEngine,
    pub metrics: crate::metrics_collector::MetricsCollector,
    pub insight: Arc<crate::insight::InsightEngine>,
    embedding_runtime_status: KnowledgeVaultIndexNoteEmbeddingProviderRuntimeStatus,
}

impl HelixEngine {
    /// Initialize the engine from configuration.
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
                .map_err(|e| HxError::Storage(format!("open keychain db: {e}")))?,
        );
        let keychain = Arc::new(
            crate::keychain::KeychainEngine::new(
                keychain_store,
                Arc::clone(&bridge_cred_store),
                None,
                Some(keychain_path.clone()),
            )
            .await?,
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
            store = store.with_embedder(embedder);
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
        let llm =
            llm::init_llm_provider_with_local(&config.llm, &config.local_llm, llm_api_key).await;

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
                pipeline
            },
            metrics: crate::metrics_collector::MetricsCollector::new(),
            insight: Arc::new(crate::insight::InsightEngine::default()),
            embedding_runtime_status: selection.runtime_status,
        };

        engine.ensure_default_permission_templates().await?;

        Ok(engine)
    }

    /// Initialize the engine and return it wrapped in Arc, with proactive engine properly wired.
    /// Use this when you need proactive features.
    pub async fn init_arc(config: EngineConfig) -> MvResult<Arc<Self>> {
        let engine = Arc::new(Self::init(config).await?);
        engine.proactive.set_engine(Arc::clone(&engine));
        engine.insight.set_engine(Arc::clone(&engine));
        Ok(engine)
    }

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

    /// Returns true when sealed mode is enabled in runtime config.
    pub fn is_sealed(&self) -> bool {
        self.config.sealed_mode && !self.keychain.is_unsealed_sync()
    }

    fn ensure_unsealed_for_node_io(&self) -> MvResult<()> {
        if self.is_sealed() {
            return Err(HxError::VaultSealed);
        }
        Ok(())
    }

    /// Set up the enrichment pipeline. Returns the worker that should be spawned.
    /// Must be called after `init_arc()` and before using enrichment features.
    pub fn setup_enrichment(
        &mut self,
        change_tx: tokio::sync::broadcast::Sender<hx_core::ChangeNotification>,
    ) -> Option<crate::enrichment::EnrichmentWorker> {
        if !self.config.ai.enrichment_enabled {
            tracing::info!("enrichment pipeline disabled by config");
            return None;
        }

        let (pipeline, worker) = crate::enrichment::EnrichmentPipeline::new(
            Arc::clone(&self.store),
            self.config.ai.clone(),
            self.llm.clone(),
            change_tx,
        );
        self.enrichment = Some(pipeline);
        tracing::info!("enrichment pipeline initialized");
        Some(worker)
    }

    async fn ensure_default_permission_templates(&self) -> MvResult<()> {
        let owner_exists = self
            .store
            .nodes
            .get_permission_template_by_name("Owner")
            .await?
            .is_some();
        let assistant_exists = self
            .store
            .nodes
            .get_permission_template_by_name("Assistant")
            .await?
            .is_some();

        if !owner_exists {
            let now = Utc::now();
            let owner = PermissionTemplate {
                id: Uuid::now_v7(),
                name: "Owner".to_string(),
                description: Some("Full access template".to_string()),
                tier: PermissionTier::Admin,
                scope_namespace: None,
                scope_tags: Vec::new(),
                allow_kinds: Vec::new(),
                allow_actions: Vec::new(),
                created_at: now,
                updated_at: now,
            };
            self.store.nodes.insert_permission_template(&owner).await?;
        }

        if !assistant_exists {
            let now = Utc::now();
            let assistant = PermissionTemplate {
                id: Uuid::now_v7(),
                name: "Assistant".to_string(),
                description: Some("Scoped assistant template".to_string()),
                tier: PermissionTier::Action,
                scope_namespace: Some("assistant".to_string()),
                scope_tags: Vec::new(),
                allow_kinds: Vec::new(),
                allow_actions: Vec::new(),
                created_at: now,
                updated_at: now,
            };
            self.store
                .nodes
                .insert_permission_template(&assistant)
                .await?;
        }

        Ok(())
    }

    pub async fn list_permission_templates(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<PermissionTemplate>> {
        self.store
            .nodes
            .list_permission_templates(limit, offset)
            .await
    }

    pub async fn create_permission_template(
        &self,
        name: String,
        description: Option<String>,
        tier: PermissionTier,
        scope_namespace: Option<String>,
        scope_tags: Vec<String>,
        allow_kinds: Vec<NodeKind>,
        allow_actions: Vec<String>,
    ) -> MvResult<PermissionTemplate> {
        let now = Utc::now();
        let template = PermissionTemplate {
            id: Uuid::now_v7(),
            name,
            description,
            tier,
            scope_namespace,
            scope_tags,
            allow_kinds,
            allow_actions,
            created_at: now,
            updated_at: now,
        };
        self.store
            .nodes
            .insert_permission_template(&template)
            .await?;
        Ok(template)
    }

    pub async fn update_permission_template(
        &self,
        template_id: Uuid,
        name: String,
        description: Option<String>,
        tier: PermissionTier,
        scope_namespace: Option<String>,
        scope_tags: Vec<String>,
        allow_kinds: Vec<NodeKind>,
        allow_actions: Vec<String>,
    ) -> MvResult<Option<PermissionTemplate>> {
        let mut existing = match self
            .store
            .nodes
            .get_permission_template(template_id)
            .await?
        {
            Some(template) => template,
            None => return Ok(None),
        };

        existing.name = name;
        existing.description = description;
        existing.tier = tier;
        existing.scope_namespace = scope_namespace;
        existing.scope_tags = scope_tags;
        existing.allow_kinds = allow_kinds;
        existing.allow_actions = allow_actions;
        existing.updated_at = Utc::now();

        self.store
            .nodes
            .update_permission_template(&existing)
            .await?;
        Ok(Some(existing))
    }

    pub async fn delete_permission_template(&self, template_id: Uuid) -> MvResult<bool> {
        self.store
            .nodes
            .delete_permission_template(template_id)
            .await
    }

    pub async fn create_access_key(
        &self,
        template_id: Uuid,
        name: Option<String>,
        expires_at: Option<DateTime<Utc>>,
    ) -> MvResult<(AccessKey, String)> {
        let template = self
            .store
            .nodes
            .get_permission_template(template_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput("permission template not found".to_string()))?;

        let token = generate_access_token();
        let key_hash = hash_access_token(&token);
        let now = Utc::now();

        let access_key = AccessKey {
            id: Uuid::now_v7(),
            name,
            template_id: template.id,
            key_hash,
            created_at: now,
            last_used_at: None,
            expires_at,
            revoked_at: None,
        };

        self.store.nodes.insert_access_key(&access_key).await?;

        Ok((access_key, token))
    }

    pub async fn create_public_share(
        &self,
        node_id: Uuid,
        expires_at: Option<DateTime<Utc>>,
    ) -> MvResult<(PublicShare, String)> {
        let node = self
            .store
            .nodes
            .get(node_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput("node not found".to_string()))?;

        let _ = node;
        let token = generate_share_token();
        let token_hash = hash_share_token(&token);
        let now = Utc::now();

        let share = PublicShare {
            id: Uuid::now_v7(),
            node_id,
            token_hash,
            created_at: now,
            expires_at,
            revoked_at: None,
        };

        self.store.nodes.insert_public_share(&share).await?;

        Ok((share, token))
    }

    pub async fn list_public_shares(
        &self,
        node_id: Option<Uuid>,
        include_revoked: bool,
    ) -> MvResult<Vec<PublicShare>> {
        self.store
            .nodes
            .list_public_shares(node_id, include_revoked)
            .await
    }

    pub async fn revoke_public_share(&self, share_id: Uuid) -> MvResult<bool> {
        self.store
            .nodes
            .revoke_public_share(share_id, Utc::now())
            .await
    }

    pub async fn resolve_public_share(
        &self,
        token: &str,
    ) -> MvResult<Option<(PublicShare, KnowledgeNode)>> {
        let token_hash = hash_share_token(token);
        let share = match self
            .store
            .nodes
            .get_public_share_by_hash(&token_hash)
            .await?
        {
            Some(share) => share,
            None => return Ok(None),
        };

        if !share.is_active() {
            return Ok(None);
        }

        let node = match self.store.nodes.get(share.node_id).await? {
            Some(node) => node,
            None => return Ok(None),
        };

        Ok(Some((share, node)))
    }

    pub async fn create_node_comment(
        &self,
        node_id: Uuid,
        author: Option<String>,
        body: String,
    ) -> MvResult<NodeComment> {
        let _ = self
            .store
            .nodes
            .get(node_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput("node not found".to_string()))?;

        let now = Utc::now();
        let comment = NodeComment {
            id: Uuid::now_v7(),
            node_id,
            author,
            body,
            created_at: now,
            updated_at: now,
            resolved_at: None,
        };

        self.store.nodes.insert_comment(&comment).await?;
        Ok(comment)
    }

    pub async fn list_node_comments(
        &self,
        node_id: Uuid,
        include_resolved: bool,
    ) -> MvResult<Vec<NodeComment>> {
        self.store
            .nodes
            .list_comments(node_id, include_resolved)
            .await
    }

    pub async fn get_node_comment(&self, comment_id: Uuid) -> MvResult<Option<NodeComment>> {
        self.store.nodes.get_comment(comment_id).await
    }

    pub async fn resolve_node_comment(&self, comment_id: Uuid) -> MvResult<bool> {
        self.store
            .nodes
            .resolve_comment(comment_id, Utc::now())
            .await
    }

    pub async fn delete_node_comment(&self, comment_id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_comment(comment_id).await
    }

    pub async fn create_mcp_connector(
        &self,
        name: String,
        description: Option<String>,
        publisher: Option<String>,
        version: String,
        homepage_url: Option<String>,
        repository_url: Option<String>,
        config_schema: serde_json::Value,
        capabilities: Vec<String>,
        verified: bool,
    ) -> MvResult<McpConnector> {
        let now = Utc::now();
        let connector = McpConnector {
            id: Uuid::now_v7(),
            name,
            description,
            publisher,
            version,
            homepage_url,
            repository_url,
            config_schema,
            capabilities,
            verified,
            created_at: now,
            updated_at: now,
        };

        self.store.nodes.insert_mcp_connector(&connector).await?;
        Ok(connector)
    }

    pub async fn list_mcp_connectors(
        &self,
        publisher: Option<&str>,
        verified: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<McpConnector>> {
        self.store
            .nodes
            .list_mcp_connectors(publisher, verified, limit, offset)
            .await
    }

    pub async fn get_mcp_connector(&self, connector_id: Uuid) -> MvResult<Option<McpConnector>> {
        self.store.nodes.get_mcp_connector(connector_id).await
    }

    pub async fn update_mcp_connector(&self, connector: McpConnector) -> MvResult<bool> {
        self.store.nodes.update_mcp_connector(&connector).await
    }

    pub async fn delete_mcp_connector(&self, connector_id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_mcp_connector(connector_id).await
    }

    pub async fn list_access_keys(&self) -> MvResult<Vec<AccessKey>> {
        self.store.nodes.list_access_keys().await
    }

    pub async fn revoke_access_key(&self, key_id: Uuid) -> MvResult<bool> {
        self.store.nodes.revoke_access_key(key_id, Utc::now()).await
    }

    pub async fn resolve_access_key(
        &self,
        token: &str,
    ) -> MvResult<Option<(AccessKey, PermissionTemplate)>> {
        let key_hash = hash_access_token(token);
        let key = match self.store.nodes.get_access_key_by_hash(&key_hash).await? {
            Some(key) => key,
            None => return Ok(None),
        };

        if key.revoked_at.is_some() {
            return Ok(None);
        }

        if let Some(expires_at) = key.expires_at {
            if expires_at < Utc::now() {
                return Ok(None);
            }
        }

        let template = self
            .store
            .nodes
            .get_permission_template(key.template_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput("permission template missing".to_string()))?;

        let _ = self
            .store
            .nodes
            .update_access_key_last_used(key.id, Utc::now())
            .await;

        Ok(Some((key, template)))
    }

    pub async fn sync_google_calendar(&self) -> MvResult<GoogleCalendarSyncReport> {
        let config = &self.config.google_calendar;
        if !config.enabled {
            return Err(HxError::InvalidInput(
                "google calendar sync is disabled".to_string(),
            ));
        }

        let client_id = config
            .client_id
            .as_ref()
            .ok_or_else(|| HxError::InvalidInput("google calendar client_id missing".into()))?;
        let client_secret = config
            .client_secret
            .as_ref()
            .ok_or_else(|| HxError::InvalidInput("google calendar client_secret missing".into()))?;
        let refresh_token = config
            .refresh_token
            .as_ref()
            .ok_or_else(|| HxError::InvalidInput("google calendar refresh_token missing".into()))?;

        let calendar_id = config.calendar_id.trim();
        if calendar_id.is_empty() {
            return Err(HxError::InvalidInput(
                "google calendar_id must not be empty".into(),
            ));
        }

        let access_token =
            google_refresh_access_token(client_id, client_secret, refresh_token).await?;
        let adapter_name = format!("google-calendar:{calendar_id}");
        let existing_sync = self
            .store
            .nodes
            .get_poll_state(&adapter_name)
            .await?
            .and_then(|state| {
                let cursor = state.cursor.trim().to_string();
                if cursor.is_empty() {
                    None
                } else {
                    Some(cursor)
                }
            });

        let mut report = GoogleCalendarSyncReport {
            calendar_id: calendar_id.to_string(),
            fetched: 0,
            created: 0,
            updated: 0,
            deleted: 0,
            skipped: 0,
            exported_created: 0,
            exported_updated: 0,
            next_sync_token: None,
        };

        let mut sync_token = existing_sync;
        for attempt in 0..2 {
            match google_list_events(&access_token, calendar_id, config, sync_token.as_deref())
                .await
            {
                Ok((events, next_sync_token)) => {
                    report.fetched = events.len();
                    report.next_sync_token = next_sync_token.clone();
                    let mut created = 0usize;
                    let mut updated = 0usize;
                    let mut deleted = 0usize;
                    let mut skipped = 0usize;

                    if config.import_events {
                        for event in events {
                            let Some(event_id) = event.id.as_ref() else {
                                skipped += 1;
                                continue;
                            };

                            if matches!(event.status.as_deref(), Some("cancelled")) {
                                if let Some(existing) = self
                                    .store
                                    .nodes
                                    .find_by_source(&event_source(calendar_id, event_id))
                                    .await?
                                {
                                    let _ = self.delete_node(existing.id).await?;
                                    deleted += 1;
                                }
                                continue;
                            }

                            let (start_at, end_at) = match event_times(&event) {
                                Some(times) => times,
                                None => {
                                    skipped += 1;
                                    continue;
                                }
                            };

                            let source = event_source(calendar_id, event_id);
                            let existing = self.store.nodes.find_by_source(&source).await?;
                            let was_existing = existing.is_some();
                            let mut node = if let Some(existing) = existing {
                                existing
                            } else {
                                KnowledgeNode::new(NodeKind::Event, "")
                                    .with_namespace(config.namespace.clone())
                                    .with_tags(vec!["calendar".into(), "google-calendar".into()])
                                    .with_source(source)
                            };

                            let title = event
                                .summary
                                .clone()
                                .filter(|s| !s.trim().is_empty())
                                .unwrap_or_else(|| "Google Calendar Event".to_string());
                            let content = event
                                .description
                                .clone()
                                .filter(|s| !s.trim().is_empty())
                                .unwrap_or_else(|| title.clone());

                            node.title = Some(title);
                            node.content = content;
                            node.metadata.insert(
                                "event_start_at".to_string(),
                                serde_json::Value::String(start_at.to_rfc3339()),
                            );
                            node.metadata.insert(
                                "event_end_at".to_string(),
                                serde_json::Value::String(end_at.to_rfc3339()),
                            );
                            node.metadata.insert(
                                "google_calendar_event_id".to_string(),
                                serde_json::Value::String(event_id.clone()),
                            );
                            node.metadata.insert(
                                "google_calendar_calendar_id".to_string(),
                                serde_json::Value::String(calendar_id.to_string()),
                            );
                            if let Some(updated_at) = event.updated.clone() {
                                node.metadata.insert(
                                    "google_calendar_updated_at".to_string(),
                                    serde_json::Value::String(updated_at),
                                );
                            }
                            if let Some(html_link) = event.html_link.clone() {
                                node.metadata.insert(
                                    "google_calendar_html_link".to_string(),
                                    serde_json::Value::String(html_link),
                                );
                            }

                            if was_existing {
                                node.temporal.updated_at = Utc::now();
                                let _ = self.update_node(node).await?;
                                updated += 1;
                            } else {
                                let _ = self.store_node(node).await?;
                                created += 1;
                            }
                        }
                    } else {
                        skipped = events.len();
                    }

                    report.created = created;
                    report.updated = updated;
                    report.deleted = deleted;
                    report.skipped = skipped;

                    if config.export_events {
                        let (exported_created, exported_updated) =
                            google_export_events(self, &access_token, calendar_id, config).await?;
                        report.exported_created = exported_created;
                        report.exported_updated = exported_updated;
                    }

                    if let Some(next) = next_sync_token {
                        let _ = self
                            .store
                            .nodes
                            .upsert_poll_state(&adapter_name, &next, report.fetched as u64)
                            .await;
                    }

                    return Ok(report);
                }
                Err(GoogleCalendarFetchError::SyncTokenExpired) => {
                    if attempt == 0 {
                        sync_token = None;
                        continue;
                    }
                    return Err(HxError::InvalidInput(
                        "google calendar sync token expired".into(),
                    ));
                }
                Err(GoogleCalendarFetchError::RequestFailed(err)) => {
                    return Err(HxError::Storage(format!(
                        "google calendar sync failed: {err}"
                    )));
                }
            }
        }

        Err(HxError::Storage(
            "google calendar sync failed unexpectedly".into(),
        ))
    }

    // ── Consumer Profiles ────────────────────────────────────────────

    /// Create a new consumer profile with a random bearer token.
    ///
    /// Returns the profile and the raw token (shown once — only the hash is stored).
    pub async fn create_consumer(
        &self,
        name: &str,
        description: Option<&str>,
    ) -> MvResult<(ConsumerProfile, String)> {
        // Check name uniqueness
        if let Some(_existing) = self.store.nodes.get_consumer_by_name(name).await? {
            return Err(HxError::InvalidInput(format!(
                "consumer with name '{name}' already exists"
            )));
        }

        // Generate 32 random bytes
        let mut raw_bytes = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut raw_bytes);

        // Base64url encode as the raw token
        let raw_token = format!("mvc_{}", URL_SAFE_NO_PAD.encode(raw_bytes));

        // Hash with SHA-256 for storage
        let mut hasher = Sha256::new();
        hasher.update(raw_token.as_bytes());
        let digest = hasher.finalize();
        let token_hash = URL_SAFE_NO_PAD.encode(digest);

        let now = Utc::now();
        let profile = ConsumerProfile {
            id: Uuid::now_v7(),
            name: name.to_string(),
            description: description.map(|d| d.to_string()),
            token_hash,
            created_at: now,
            last_used_at: None,
            revoked_at: None,
            metadata: std::collections::HashMap::new(),
        };

        self.store.nodes.create_consumer(&profile).await?;
        Ok((profile, raw_token))
    }

    /// Resolve a raw consumer token to the corresponding profile.
    ///
    /// Hashes the token, looks up by hash, returns None if not found or revoked.
    /// Touches `last_used_at` on success.
    pub async fn resolve_consumer_token(&self, token: &str) -> MvResult<Option<ConsumerProfile>> {
        let mut hasher = Sha256::new();
        hasher.update(token.as_bytes());
        let digest = hasher.finalize();
        let token_hash = URL_SAFE_NO_PAD.encode(digest);

        let profile = match self
            .store
            .nodes
            .get_consumer_by_token_hash(&token_hash)
            .await?
        {
            Some(p) => p,
            None => return Ok(None),
        };

        if profile.revoked_at.is_some() {
            return Ok(None);
        }

        // Touch last_used_at
        let _ = self.store.nodes.touch_consumer(profile.id).await;

        Ok(Some(profile))
    }

    /// List all consumer profiles.
    pub async fn list_consumers(&self) -> MvResult<Vec<ConsumerProfile>> {
        self.store.nodes.list_consumers().await
    }

    /// Get a consumer profile by ID.
    pub async fn get_consumer(&self, id: Uuid) -> MvResult<Option<ConsumerProfile>> {
        self.store.nodes.get_consumer(id).await
    }

    /// Revoke a consumer profile by ID.
    pub async fn revoke_consumer(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.revoke_consumer(id).await
    }

    // ── Access Policies ─────────────────────────────────────────────

    /// Create or update an access policy.
    pub async fn set_policy(&self, policy: &AccessPolicy) -> MvResult<()> {
        self.store.nodes.set_policy(policy).await
    }

    /// Check whether a consumer is allowed access to a specific secret.
    ///
    /// Returns `PolicyDecision::Allow` with TTL/scopes or `PolicyDecision::Deny` with reason.
    /// Default deny: no matching policy means deny.
    pub async fn check_policy(&self, secret_key: &str, consumer: &str) -> MvResult<PolicyDecision> {
        let policy = self
            .store
            .nodes
            .get_policy_for(secret_key, consumer)
            .await?;

        match policy {
            None => Ok(PolicyDecision::Deny {
                reason: format!(
                    "no policy found for consumer '{consumer}' on secret '{secret_key}'"
                ),
            }),
            Some(p) => {
                if !p.allowed {
                    return Ok(PolicyDecision::Deny {
                        reason: format!("policy explicitly denies access"),
                    });
                }

                if p.is_expired() {
                    return Ok(PolicyDecision::Deny {
                        reason: format!("policy has expired"),
                    });
                }

                if p.require_approval {
                    return Ok(PolicyDecision::RequiresApproval {
                        ttl_seconds: p.max_ttl_seconds.unwrap_or(300),
                        scopes: p.scopes.clone(),
                    });
                }

                Ok(PolicyDecision::Allow {
                    ttl_seconds: p.max_ttl_seconds,
                    scopes: p.scopes.clone(),
                })
            }
        }
    }

    /// List access policies with optional filters.
    pub async fn list_policies(
        &self,
        secret_key: Option<&str>,
        consumer: Option<&str>,
    ) -> MvResult<Vec<AccessPolicy>> {
        self.store.nodes.list_policies(secret_key, consumer).await
    }

    /// Delete an access policy by ID.
    pub async fn delete_policy(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_policy(id).await
    }

    // ── Proxy Audit ─────────────────────────────────────────────────

    /// Log a proxy audit entry.
    pub async fn log_proxy_audit(&self, entry: &ProxyAuditEntry) -> MvResult<()> {
        self.store.nodes.log_proxy_audit(entry).await
    }

    /// Update a proxy audit entry with execution results.
    pub async fn update_proxy_audit(
        &self,
        id: Uuid,
        success: bool,
        sanitized: bool,
        error: Option<&str>,
        response_status: Option<i32>,
    ) -> MvResult<()> {
        self.store
            .nodes
            .update_proxy_audit(id, success, sanitized, error, response_status)
            .await
    }

    /// List proxy audit entries with optional consumer filter.
    pub async fn list_proxy_audit(
        &self,
        consumer: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ProxyAuditEntry>> {
        self.store
            .nodes
            .list_proxy_audit(consumer, limit, offset)
            .await
    }

    // ── Proxy Approvals ─────────────────────────────────────────────

    /// Create a new approval request.
    pub async fn create_approval(&self, request: &ApprovalRequest) -> MvResult<()> {
        self.store.nodes.create_approval(request).await
    }

    /// Get a specific approval by ID.
    pub async fn get_approval(&self, id: Uuid) -> MvResult<Option<ApprovalRequest>> {
        self.store.nodes.get_approval(id).await
    }

    /// List pending approvals, optionally filtered by consumer.
    pub async fn list_pending_approvals(
        &self,
        consumer: Option<&str>,
    ) -> MvResult<Vec<ApprovalRequest>> {
        self.store.nodes.list_pending_approvals(consumer).await
    }

    /// Approve or deny an approval request.
    pub async fn decide_approval(
        &self,
        id: Uuid,
        approved: bool,
        decided_by: Option<&str>,
        deny_reason: Option<&str>,
    ) -> MvResult<bool> {
        self.store
            .nodes
            .decide_approval(id, approved, decided_by, deny_reason)
            .await
    }

    /// Expire all past-due pending approvals.
    pub async fn expire_approvals(&self) -> MvResult<usize> {
        self.store.nodes.expire_approvals().await
    }

    /// Find an active (approved, non-expired) approval for a consumer+secret pair.
    pub async fn find_active_approval(
        &self,
        consumer: &str,
        secret_key: &str,
    ) -> MvResult<Option<ApprovalRequest>> {
        self.store
            .nodes
            .find_active_approval(consumer, secret_key)
            .await
    }

    // ── Owner Profile ────────────────────────────────────────────────

    pub async fn get_profile(&self) -> MvResult<OwnerProfile> {
        self.store.nodes.get_profile().await
    }

    pub async fn update_profile(&self, req: &UpdateProfileRequest) -> MvResult<OwnerProfile> {
        let profile = self.store.nodes.update_profile(req).await?;
        self.sync_owner_relay_contact(profile).await
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

    /// Store a knowledge node.
    pub async fn store_node(&self, node: KnowledgeNode) -> MvResult<KnowledgeNode> {
        self.ensure_unsealed_for_node_io()?;
        let stored = self.ingest.ingest(node).await?;
        self.auto_link_node_to_daily_note_best_effort(&stored).await;
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

    /// Receive a relay message and optionally generate an auto-reply or proposal.
    pub async fn receive_relay_message(
        &self,
        message: RelayMessage,
        namespace: &str,
    ) -> MvResult<RelayInboundOutcome> {
        let mut stored = self.relay.receive_message(message, namespace).await?;

        if stored.status == MessageStatus::Failed {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        let Some(sender_id) = stored.sender_contact_id else {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        };

        let Some(contact) = self.relay.get_contact(sender_id).await? else {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        };

        if contact.trust_level == TrustLevel::RelayOnly {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        if stored.content_type != ContentType::Text || stored.content.trim().is_empty() {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        let thread_id = stored.thread_id.unwrap_or(stored.id);

        let subject = stored
            .metadata
            .get("subject")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                if value.to_ascii_lowercase().starts_with("re:") {
                    value.to_string()
                } else {
                    format!("Re: {value}")
                }
            });

        let query = MemoryQuery::new(&stored.content)
            .with_namespace(namespace.to_string())
            .with_limit(6)
            .with_min_score(0.0);

        let results = match self.recall(&query).await {
            Ok(results) => results,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "relay_reply_context_recall_failed"
                );
                return Ok(RelayInboundOutcome {
                    message: stored,
                    auto_reply: None,
                    proposal_id: None,
                });
            }
        };

        let context_snippets = llm::extract_context_snippets(&results, 4);
        if context_snippets.is_empty() {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        let input = if let Some(ref subject) = subject {
            format!("Subject: {subject}\n\n{}", stored.content)
        } else {
            stored.content.clone()
        };

        let mut used_llm = false;
        let mut suggestion_text = None;
        if let Some(ref llm) = self.llm {
            match llm::llm_completion_suggestions(llm.as_ref(), &input, &context_snippets, 1).await
            {
                Ok(mut suggestions) => {
                    if let Some(first) = suggestions.pop() {
                        suggestion_text = Some(first);
                        used_llm = true;
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        provider = %llm.name(),
                        "relay_reply_llm_suggestion_failed"
                    );
                }
            }
        }

        if suggestion_text.is_none() {
            let preview = context_snippets
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            suggestion_text = Some(format!(
                "I have related notes that might help:\n{preview}\n\nWant me to share details?"
            ));
        }

        let mut confidence: f32 = if used_llm { 0.6 } else { 0.4 };
        if context_snippets.len() >= 3 {
            confidence += 0.1;
        }
        if contact.trust_level == TrustLevel::Full {
            confidence += 0.1;
        }
        if contact.trust_level == TrustLevel::ContextInject {
            confidence -= 0.05;
        }
        confidence = confidence.clamp(0.0, 1.0);

        let suggestion = RelayReplySuggestion {
            content: suggestion_text.unwrap_or_default(),
            confidence,
            context_snippets,
        };

        let contact_scope = sender_id.to_string();
        let scope_hints = [("contact", contact_scope.as_str()), ("domain", "relay")];
        let mut decision = self
            .autonomy
            .evaluate("relay.reply", suggestion.confidence, &scope_hints)
            .await?;

        if contact.trust_level != TrustLevel::Full
            && matches!(decision, AutonomyDecision::AutoApply)
        {
            decision = AutonomyDecision::Defer;
        }

        let mut auto_reply = None;
        let mut proposal_id = None;

        match decision {
            AutonomyDecision::AutoApply => {
                let mut reply =
                    RelayMessage::outbound(stored.channel_id, suggestion.content.clone())
                        .with_thread(thread_id)
                        .with_content_type(ContentType::Text);
                reply.recipient_contact_id = Some(sender_id);
                reply
                    .metadata
                    .insert("auto_reply".to_string(), serde_json::Value::Bool(true));
                reply.metadata.insert(
                    "basis_message_id".to_string(),
                    serde_json::Value::String(stored.id.to_string()),
                );
                if let Some(ref subject) = subject {
                    reply.metadata.insert(
                        "subject".to_string(),
                        serde_json::Value::String(subject.clone()),
                    );
                }

                let stored_reply = self.relay.send_message(reply, namespace).await?;
                auto_reply = Some(stored_reply);

                if let Ok(true) = self
                    .relay
                    .update_status(stored.id, MessageStatus::AutoReplied)
                    .await
                {
                    stored.status = MessageStatus::AutoReplied;
                }
            }
            AutonomyDecision::Defer | AutonomyDecision::QueueForLater => {
                let mut payload = std::collections::HashMap::new();
                payload.insert(
                    "channel_id".to_string(),
                    serde_json::Value::String(stored.channel_id.to_string()),
                );
                payload.insert(
                    "content".to_string(),
                    serde_json::Value::String(suggestion.content.clone()),
                );
                payload.insert(
                    "content_type".to_string(),
                    serde_json::Value::String(ContentType::Text.to_string()),
                );
                payload.insert(
                    "basis_message_id".to_string(),
                    serde_json::Value::String(stored.id.to_string()),
                );
                payload.insert(
                    "context_snippets".to_string(),
                    serde_json::Value::Array(
                        suggestion
                            .context_snippets
                            .iter()
                            .map(|snippet| serde_json::Value::String(snippet.clone()))
                            .collect(),
                    ),
                );
                payload.insert(
                    "thread_id".to_string(),
                    serde_json::Value::String(thread_id.to_string()),
                );
                if let Some(recipient_id) = stored.sender_contact_id {
                    payload.insert(
                        "recipient_contact_id".to_string(),
                        serde_json::Value::String(recipient_id.to_string()),
                    );
                }
                if let Some(ref subject) = subject {
                    payload.insert(
                        "subject".to_string(),
                        serde_json::Value::String(subject.clone()),
                    );
                }

                let proposal = Proposal::new(
                    ProposalSender::Relay,
                    ProposalAction::Custom("relay.reply".to_string()),
                )
                .with_confidence(suggestion.confidence)
                .with_diff(suggestion.content.clone())
                .with_payload(payload);

                self.submit_proposal(&proposal).await?;
                proposal_id = Some(proposal.id);

                if let Ok(true) = self
                    .relay
                    .update_status(stored.id, MessageStatus::Deferred)
                    .await
                {
                    stored.status = MessageStatus::Deferred;
                }
            }
            AutonomyDecision::Block => {}
        }

        Ok(RelayInboundOutcome {
            message: stored,
            auto_reply,
            proposal_id,
        })
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

        let daily_namespace =
            namespace.unwrap_or_else(|| self.config.daily_notes.namespace.clone());
        if let Some(existing) = self.find_daily_note(date, &daily_namespace).await? {
            return Ok((existing, false));
        }

        let mut node = KnowledgeNode::new(
            NodeKind::Fact,
            render_daily_note_template(&self.config.daily_notes.content_template, date),
        )
        .with_namespace(daily_namespace)
        .with_tags(vec![
            DAILY_NOTE_TAG.to_string(),
            daily_note_day_tag(date),
            daily_note_weekday_tag(date),
        ])
        .with_importance(self.config.daily_notes.default_importance);

        let title = render_daily_note_template(&self.config.daily_notes.title_template, date);
        if !title.trim().is_empty() {
            node = node.with_title(title);
        }

        node.metadata
            .insert("daily_note".to_string(), serde_json::Value::Bool(true));
        node.metadata.insert(
            "daily_note_date".to_string(),
            serde_json::Value::String(date.to_string()),
        );

        let stored = self.ingest.ingest(node).await?;
        Ok((stored, true))
    }

    /// List daily notes by namespace (or all namespaces when None).
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

    /// Generate due instances for recurring task templates.
    pub async fn rollforward_recurring_tasks(
        &self,
        now: DateTime<Utc>,
        max_instances_per_template: usize,
    ) -> MvResult<TaskRecurrenceRollforwardStats> {
        if !self.config.recurrence.enabled {
            return Ok(TaskRecurrenceRollforwardStats::default());
        }

        let mut stats = TaskRecurrenceRollforwardStats::default();
        let page_size = 200;
        let mut offset = 0usize;

        loop {
            let filters = QueryFilters {
                kinds: Some(vec![NodeKind::Task]),
                ..Default::default()
            };
            let page = self.store.nodes.list(&filters, page_size, offset).await?;
            if page.is_empty() {
                break;
            }
            let page_len = page.len();

            for mut template in page {
                stats.scanned_tasks += 1;
                if is_recurring_instance(&template) {
                    continue;
                }

                let recurrence_rule = match parse_task_recurrence_rule(&template.metadata) {
                    Ok(Some(rule)) if rule.enabled => rule,
                    Ok(Some(_)) | Ok(None) => continue,
                    Err(err) => {
                        stats.errors += 1;
                        tracing::warn!(
                            node_id = %template.id,
                            namespace = %template.namespace,
                            error = %err,
                            "helix_recurrence_rule_parse_failed"
                        );
                        continue;
                    }
                };
                stats.recurring_templates += 1;

                let explicit_due_at = match parse_optional_metadata_datetime(
                    &template.metadata,
                    TASK_DUE_AT_METADATA_KEY,
                ) {
                    Ok(value) => value,
                    Err(err) => {
                        stats.errors += 1;
                        tracing::warn!(
                            node_id = %template.id,
                            namespace = %template.namespace,
                            error = %err,
                            "helix_recurrence_due_at_parse_failed"
                        );
                        continue;
                    }
                };
                let last_generated = match parse_optional_metadata_datetime(
                    &template.metadata,
                    TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY,
                ) {
                    Ok(Some(value)) => value,
                    Ok(None) => explicit_due_at
                        .map(|due| previous_due_at(due, &recurrence_rule))
                        .unwrap_or(template.temporal.created_at),
                    Err(err) => {
                        stats.errors += 1;
                        tracing::warn!(
                            node_id = %template.id,
                            namespace = %template.namespace,
                            error = %err,
                            "helix_recurrence_last_generated_parse_failed"
                        );
                        continue;
                    }
                };
                let generated_count = parse_optional_metadata_u64(
                    &template.metadata,
                    TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY,
                )
                .unwrap_or(0);
                let due_dates = collect_due_occurrences(
                    &recurrence_rule,
                    last_generated,
                    now,
                    max_instances_per_template,
                    generated_count,
                );
                if due_dates.is_empty() {
                    continue;
                }

                let mut created_for_template = 0usize;
                let mut latest_due = last_generated;
                for due_at in due_dates {
                    let mut instance = KnowledgeNode::new(NodeKind::Task, template.content.clone())
                        .with_namespace(template.namespace.clone())
                        .with_importance(template.importance);
                    if let Some(title) = template.title.as_deref() {
                        instance = instance.with_title(title);
                    }
                    if let Some(source) = template.source.as_deref() {
                        instance = instance.with_source(source);
                    }

                    let mut tags = template.tags.clone();
                    if !tags
                        .iter()
                        .any(|tag| tag.eq_ignore_ascii_case("recurring-instance"))
                    {
                        tags.push("recurring-instance".to_string());
                    }
                    instance = instance.with_tags(tags);

                    for (key, value) in &template.metadata {
                        if matches!(
                            key.as_str(),
                            TASK_RECURRENCE_METADATA_KEY
                                | TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY
                                | TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY
                                | RECURRING_INSTANCE_METADATA_KEY
                                | RECURRING_PARENT_ID_METADATA_KEY
                                | RECURRING_DUE_AT_METADATA_KEY
                        ) {
                            continue;
                        }
                        instance.metadata.insert(key.clone(), value.clone());
                    }
                    instance.metadata.insert(
                        RECURRING_INSTANCE_METADATA_KEY.into(),
                        serde_json::Value::Bool(true),
                    );
                    instance.metadata.insert(
                        RECURRING_PARENT_ID_METADATA_KEY.into(),
                        serde_json::Value::String(template.id.to_string()),
                    );
                    instance.metadata.insert(
                        RECURRING_DUE_AT_METADATA_KEY.into(),
                        serde_json::Value::String(due_at.to_rfc3339()),
                    );
                    instance.metadata.insert(
                        TASK_DUE_AT_METADATA_KEY.into(),
                        serde_json::Value::String(due_at.to_rfc3339()),
                    );

                    match self.store_node(instance).await {
                        Ok(stored_instance) => {
                            created_for_template += 1;
                            latest_due = due_at;
                            stats.generated_instances += 1;

                            let rel = Relationship::new(
                                template.id,
                                stored_instance.id,
                                RelationKind::DerivedFrom,
                            );
                            if let Err(err) = self.graph.add_relationship(&rel).await {
                                stats.errors += 1;
                                tracing::warn!(
                                    template_id = %template.id,
                                    instance_id = %stored_instance.id,
                                    error = %err,
                                    "helix_recurrence_parent_instance_link_failed"
                                );
                            }
                        }
                        Err(err) => {
                            stats.errors += 1;
                            tracing::warn!(
                                node_id = %template.id,
                                namespace = %template.namespace,
                                error = %err,
                                "helix_recurrence_instance_create_failed"
                            );
                        }
                    }
                }

                if created_for_template > 0 {
                    template.metadata.insert(
                        TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY.into(),
                        serde_json::Value::String(latest_due.to_rfc3339()),
                    );
                    let total_generated_count =
                        generated_count.saturating_add(created_for_template as u64);
                    template.metadata.insert(
                        TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY.into(),
                        serde_json::Value::Number(serde_json::Number::from(total_generated_count)),
                    );
                    template.temporal.updated_at = now;
                    template.temporal.version = template.temporal.version.saturating_add(1);
                    if let Err(err) = self.ingest.update(template).await {
                        stats.errors += 1;
                        tracing::warn!(
                            error = %err,
                            "helix_recurrence_template_update_failed"
                        );
                    } else {
                        stats.updated_templates += 1;
                    }
                }
            }

            if page_size > 0 && page_len < page_size {
                break;
            }
            offset = offset.saturating_add(page_size);
        }

        Ok(stats)
    }

    /// List due tasks up to `due_before`.
    pub async fn list_due_tasks(
        &self,
        due_before: DateTime<Utc>,
        namespace: Option<String>,
        limit: usize,
        include_completed: bool,
    ) -> MvResult<Vec<KnowledgeNode>> {
        let capped_limit = limit.clamp(1, 1000);
        let page_size = 250;
        let mut offset = 0usize;
        let mut due = Vec::<(DateTime<Utc>, KnowledgeNode)>::new();

        loop {
            let filters = QueryFilters {
                namespace: namespace.clone(),
                kinds: Some(vec![NodeKind::Task]),
                ..Default::default()
            };
            let page = self.store.nodes.list(&filters, page_size, offset).await?;
            if page.is_empty() {
                break;
            }
            let page_len = page.len();

            for node in page {
                let due_at = match parse_optional_metadata_datetime(
                    &node.metadata,
                    TASK_DUE_AT_METADATA_KEY,
                ) {
                    Ok(Some(value)) => value,
                    Ok(None) => continue,
                    Err(err) => {
                        tracing::warn!(
                            node_id = %node.id,
                            namespace = %node.namespace,
                            error = %err,
                            "helix_due_task_parse_failed"
                        );
                        continue;
                    }
                };

                if due_at > due_before {
                    continue;
                }

                let is_completed =
                    parse_optional_metadata_bool(&node.metadata, TASK_COMPLETED_METADATA_KEY)
                        .unwrap_or(false);
                if !include_completed && is_completed {
                    continue;
                }

                due.push((due_at, node));
            }

            if page_size > 0 && page_len < page_size {
                break;
            }
            offset = offset.saturating_add(page_size);
        }

        due.sort_by(|(left_due, left_node), (right_due, right_node)| {
            left_due
                .cmp(right_due)
                .then_with(|| left_node.id.cmp(&right_node.id))
        });

        Ok(due
            .into_iter()
            .take(capped_limit)
            .map(|(_due_at, node)| node)
            .collect())
    }

    /// Prioritize tasks using deterministic heuristic scoring.
    pub async fn prioritize_tasks(
        &self,
        options: TaskPrioritizationOptions,
    ) -> MvResult<Vec<PrioritizedTask>> {
        let limit = options.limit.clamp(1, 200);
        let page_size = 250;
        let mut offset = 0usize;
        let mut candidates: Vec<TaskPriorityCandidate> = Vec::new();

        loop {
            let filters = QueryFilters {
                namespace: options.namespace.clone(),
                kinds: Some(vec![NodeKind::Task]),
                ..Default::default()
            };
            let page = self.store.nodes.list(&filters, page_size, offset).await?;
            if page.is_empty() {
                break;
            }
            let page_len = page.len();

            for node in page {
                let completed =
                    parse_optional_metadata_bool(&node.metadata, TASK_COMPLETED_METADATA_KEY)
                        .unwrap_or(false);
                if !options.include_completed && completed {
                    continue;
                }

                let due_at = match parse_optional_metadata_datetime(
                    &node.metadata,
                    TASK_DUE_AT_METADATA_KEY,
                ) {
                    Ok(value) => value,
                    Err(err) => {
                        tracing::warn!(
                            node_id = %node.id,
                            namespace = %node.namespace,
                            error = %err,
                            "helix_task_priority_due_at_parse_failed"
                        );
                        continue;
                    }
                };

                if due_at.is_none() && !options.include_without_due {
                    continue;
                }

                let (score, reason) = Self::score_task(&node, due_at, options.now);
                candidates.push(TaskPriorityCandidate {
                    task: node,
                    score,
                    reason,
                    due_at,
                });
            }

            if page_size > 0 && page_len < page_size {
                break;
            }
            offset = offset.saturating_add(page_size);
        }

        candidates.sort_by(|left, right| {
            let score_order = right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal);
            if score_order != Ordering::Equal {
                return score_order;
            }

            let due_order = match (left.due_at, right.due_at) {
                (Some(left_due), Some(right_due)) => left_due.cmp(&right_due),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            };
            if due_order != Ordering::Equal {
                return due_order;
            }

            left.task.id.cmp(&right.task.id)
        });

        let mut prioritized = Vec::new();
        for (idx, candidate) in candidates.into_iter().take(limit).enumerate() {
            let rank = idx + 1;
            let mut task = candidate.task;
            let reason = candidate.reason;
            let score = candidate.score;

            if options.persist {
                task.metadata.insert(
                    TASK_AI_PRIORITY_METADATA_KEY.into(),
                    serde_json::json!({
                        "score": score,
                        "rank": rank,
                        "reason": reason,
                        "generated_at": options.now.to_rfc3339(),
                        "algorithm": "heuristic_v1",
                    }),
                );
                task.temporal.updated_at = options.now;
                task.temporal.version = task.temporal.version.saturating_add(1);
                task = self.ingest.update(task).await?;
            }

            prioritized.push(PrioritizedTask {
                task,
                score,
                rank,
                reason,
            });
        }

        Ok(prioritized)
    }

    fn score_task(
        task: &KnowledgeNode,
        due_at: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> (f64, String) {
        let priority_override = Self::parse_optional_metadata_f64(
            &task.metadata,
            TASK_PRIORITY_METADATA_KEY,
        )
        .or_else(|| {
            Self::parse_optional_metadata_f64(&task.metadata, TASK_PRIORITY_ALT_METADATA_KEY)
        });
        let priority_score = if let Some(priority_raw) = priority_override {
            let priority = priority_raw.round().clamp(1.0, 5.0);
            (6.0 - priority) / 5.0
        } else {
            task.importance.clamp(0.0, 1.0)
        };

        let mut due_score = 0.0;
        if let Some(due_at) = due_at {
            let hours = (due_at - now).num_seconds() as f64 / 3600.0;
            if hours <= 0.0 {
                due_score = 1.0;
            } else {
                let days = hours / 24.0;
                due_score = (1.0 - (days / 7.0).min(1.0)).max(0.0);
            }
        }

        let status = task
            .metadata
            .get(TASK_STATUS_METADATA_KEY)
            .or_else(|| task.metadata.get(TASK_STATUS_ALT_METADATA_KEY))
            .and_then(|value| value.as_str())
            .map(|value| value.to_ascii_lowercase());
        let status_score = match status.as_deref() {
            Some("in_progress") => 0.2,
            Some("planned") => 0.12,
            Some("review") => 0.1,
            Some("inbox") => 0.05,
            Some("waiting") => -0.05,
            Some("blocked") => -0.1,
            _ => 0.0,
        };

        let estimate =
            Self::parse_optional_metadata_f64(&task.metadata, TASK_ESTIMATE_MINUTES_METADATA_KEY)
                .or_else(|| {
                    Self::parse_optional_metadata_f64(
                        &task.metadata,
                        TASK_ESTIMATE_MINUTES_ALT_METADATA_KEY,
                    )
                })
                .or_else(|| {
                    Self::parse_optional_metadata_f64(
                        &task.metadata,
                        TASK_ESTIMATE_MIN_METADATA_KEY,
                    )
                });
        let estimate_score = match estimate {
            Some(minutes) => (1.0 - (minutes / 240.0).min(1.0)).max(0.0),
            None => 0.05,
        };

        let completed = parse_optional_metadata_bool(&task.metadata, TASK_COMPLETED_METADATA_KEY)
            .unwrap_or(false);
        let completion_penalty = if completed { -0.4 } else { 0.0 };

        let score = 0.45 * priority_score
            + 0.35 * due_score
            + 0.1 * status_score
            + 0.1 * estimate_score
            + completion_penalty;

        let mut reasons: Vec<String> = Vec::new();
        if let Some(priority_raw) = priority_override {
            let priority = priority_raw.round().clamp(1.0, 5.0) as i64;
            if priority <= 2 {
                reasons.push(format!("High priority (P{priority})"));
            } else if priority >= 4 {
                reasons.push(format!("Lower priority (P{priority})"));
            }
        } else if priority_score >= 0.8 {
            reasons.push("High importance".to_string());
        } else if priority_score <= 0.3 {
            reasons.push("Lower importance".to_string());
        }

        if let Some(due_at) = due_at {
            let delta = due_at - now;
            if delta.num_seconds() <= 0 {
                reasons.push("Overdue".to_string());
            } else {
                let days = delta.num_seconds() as f64 / 86_400.0;
                if days <= 1.0 {
                    reasons.push("Due within 24h".to_string());
                } else if days <= 3.0 {
                    reasons.push("Due soon".to_string());
                } else if days <= 7.0 {
                    reasons.push("Due this week".to_string());
                }
            }
        }

        match status.as_deref() {
            Some("in_progress") => reasons.push("In progress".to_string()),
            Some("planned") => reasons.push("Planned".to_string()),
            Some("waiting") => reasons.push("Waiting".to_string()),
            Some("review") => reasons.push("In review".to_string()),
            Some("blocked") => reasons.push("Blocked".to_string()),
            _ => {}
        }

        if let Some(minutes) = estimate {
            if minutes <= 30.0 {
                reasons.push("Quick win".to_string());
            }
        }

        if completed {
            reasons.push("Completed".to_string());
        }

        let reason = if reasons.is_empty() {
            "Balanced priority".to_string()
        } else {
            reasons.into_iter().take(3).collect::<Vec<_>>().join(", ")
        };

        (score, reason)
    }

    fn parse_optional_metadata_f64(
        metadata: &std::collections::HashMap<String, serde_json::Value>,
        key: &str,
    ) -> Option<f64> {
        match metadata.get(key) {
            Some(serde_json::Value::Number(value)) => value.as_f64(),
            Some(serde_json::Value::String(value)) => value.parse::<f64>().ok(),
            _ => None,
        }
    }

    /// Mark due task reminders as sent by setting metadata fields on each task.
    pub async fn dispatch_due_task_reminders(
        &self,
        now: DateTime<Utc>,
        limit: usize,
    ) -> MvResult<TaskReminderDispatchStats> {
        let due_tasks = self.list_due_tasks(now, None, limit, false).await?;
        let mut stats = TaskReminderDispatchStats {
            scanned_tasks: due_tasks.len(),
            due_tasks: due_tasks.len(),
            ..Default::default()
        };

        for mut task in due_tasks {
            let already_sent = match parse_optional_metadata_datetime(
                &task.metadata,
                TASK_REMINDER_SENT_AT_METADATA_KEY,
            ) {
                Ok(value) => value.is_some(),
                Err(err) => {
                    stats.errors += 1;
                    tracing::warn!(
                        node_id = %task.id,
                        namespace = %task.namespace,
                        error = %err,
                        "helix_task_reminder_sent_at_parse_failed"
                    );
                    continue;
                }
            };
            if already_sent {
                continue;
            }

            task.metadata.insert(
                TASK_REMINDER_STATUS_METADATA_KEY.into(),
                serde_json::Value::String("sent".to_string()),
            );
            task.metadata.insert(
                TASK_REMINDER_SENT_AT_METADATA_KEY.into(),
                serde_json::Value::String(now.to_rfc3339()),
            );
            task.temporal.updated_at = now;
            task.temporal.version = task.temporal.version.saturating_add(1);

            match self.ingest.update(task).await {
                Ok(_updated) => {
                    stats.reminders_marked_sent += 1;
                }
                Err(err) => {
                    stats.errors += 1;
                    tracing::warn!(
                        error = %err,
                        "helix_task_reminder_mark_sent_failed"
                    );
                }
            }
        }

        Ok(stats)
    }

    /// Add a relationship between nodes.
    pub async fn add_relationship(&self, rel: Relationship) -> MvResult<()> {
        self.graph.add_relationship(&rel).await
    }

    /// Get graph neighbors.
    pub async fn get_neighbors(
        &self,
        node_id: uuid::Uuid,
        depth: usize,
    ) -> MvResult<Vec<uuid::Uuid>> {
        self.graph.get_neighbors(node_id, depth).await
    }

    /// Get node count.
    pub async fn node_count(&self) -> MvResult<usize> {
        self.store.nodes.count(&QueryFilters::default()).await
    }

    /// Return the active embedding provider diagnostics for observability.
    pub fn embedding_runtime_status(
        &self,
    ) -> KnowledgeVaultIndexNoteEmbeddingProviderRuntimeStatus {
        self.embedding_runtime_status.clone()
    }

    async fn auto_link_node_to_daily_note_best_effort(&self, node: &KnowledgeNode) {
        if !self.config.daily_notes.enabled {
            return;
        }
        if is_template_node(node) {
            return;
        }
        if is_daily_note(node) {
            return;
        }
        if !is_daily_link_candidate(node) {
            return;
        }

        let day = node.temporal.created_at.date_naive();
        let daily_note = match self.find_daily_note(day, &node.namespace).await {
            Ok(Some(existing)) => existing,
            Ok(None) => match self
                .ensure_daily_note(day, Some(node.namespace.clone()))
                .await
            {
                Ok((created, _created)) => created,
                Err(err) => {
                    tracing::warn!(
                        node_id = %node.id,
                        namespace = %node.namespace,
                        date = %day,
                        error = %err,
                        "helix_daily_note_auto_link_ensure_failed"
                    );
                    return;
                }
            },
            Err(err) => {
                tracing::warn!(
                    node_id = %node.id,
                    namespace = %node.namespace,
                    date = %day,
                    error = %err,
                    "helix_daily_note_auto_link_lookup_failed"
                );
                return;
            }
        };

        let existing_relationships = match self.graph.get_relationships_from(daily_note.id).await {
            Ok(relationships) => relationships,
            Err(err) => {
                tracing::warn!(
                    node_id = %node.id,
                    daily_note_id = %daily_note.id,
                    error = %err,
                    "helix_daily_note_auto_link_relationship_scan_failed"
                );
                return;
            }
        };

        let already_linked = existing_relationships.iter().any(|rel| {
            rel.to_node == node.id
                && matches!(
                    rel.kind,
                    RelationKind::Contains | RelationKind::References | RelationKind::PartOf
                )
        });
        if already_linked {
            return;
        }

        let relationship = Relationship::new(daily_note.id, node.id, RelationKind::Contains);
        if let Err(err) = self.graph.add_relationship(&relationship).await {
            tracing::warn!(
                node_id = %node.id,
                daily_note_id = %daily_note.id,
                error = %err,
                "helix_daily_note_auto_link_insert_failed"
            );
            return;
        }

        tracing::info!(
            node_id = %node.id,
            daily_note_id = %daily_note.id,
            namespace = %node.namespace,
            date = %day,
            "helix_daily_note_auto_linked"
        );
    }

    async fn auto_backlink_node_references_best_effort(&self, node: &KnowledgeNode) {
        if !self.config.linking.auto_backlinks_enabled {
            return;
        }
        if is_template_node(node) {
            return;
        }

        let link_targets = extract_reference_targets_with_kind(
            &node.content,
            self.config.linking.auto_backlinks_max_targets,
        );

        let resolved_target_ids = if link_targets.is_empty() {
            Vec::new()
        } else {
            let index = match self
                .build_backlink_resolution_index(
                    &node.namespace,
                    self.config.linking.auto_backlinks_scan_limit,
                )
                .await
            {
                Ok(index) => index,
                Err(err) => {
                    tracing::warn!(
                        node_id = %node.id,
                        namespace = %node.namespace,
                        error = %err,
                        "helix_backlink_auto_index_build_failed"
                    );
                    return;
                }
            };
            resolve_reference_targets_with_kind(&link_targets, node.id, &index)
        };

        if let Err(err) = self
            .sync_auto_backlink_references(node.id, &resolved_target_ids)
            .await
        {
            tracing::warn!(
                node_id = %node.id,
                namespace = %node.namespace,
                error = %err,
                "helix_backlink_auto_sync_failed"
            );
            return;
        }

        tracing::debug!(
            node_id = %node.id,
            namespace = %node.namespace,
            resolved_links = resolved_target_ids.len(),
            extracted_targets = link_targets.len(),
            "helix_backlink_auto_sync_applied"
        );
    }

    async fn build_backlink_resolution_index(
        &self,
        namespace: &str,
        scan_limit: usize,
    ) -> MvResult<KnowledgeVaultBacklinkResolutionIndex> {
        if scan_limit == 0 {
            return Ok(KnowledgeVaultBacklinkResolutionIndex::default());
        }

        let mut index = KnowledgeVaultBacklinkResolutionIndex::default();
        let filters = QueryFilters {
            namespace: Some(namespace.to_string()),
            ..Default::default()
        };
        let page_size = scan_limit.min(200);
        let mut offset = 0usize;
        let mut scanned = 0usize;

        while scanned < scan_limit {
            let remaining = scan_limit - scanned;
            let limit = remaining.min(page_size);
            let batch = self.store.nodes.list(&filters, limit, offset).await?;
            if batch.is_empty() {
                break;
            }
            offset += batch.len();
            scanned += batch.len();
            for node in batch {
                index.insert_node(node.id, node.title.as_deref(), node.source.as_deref());
            }
        }

        if scanned == scan_limit {
            tracing::debug!(
                namespace = %namespace,
                scan_limit,
                "helix_backlink_auto_index_scan_capped"
            );
        }

        Ok(index)
    }

    async fn sync_auto_backlink_references(
        &self,
        from_node_id: uuid::Uuid,
        desired_target_ids: &[ResolvedContentReferenceTarget],
    ) -> MvResult<()> {
        let mut desired =
            std::collections::HashMap::<uuid::Uuid, ContentReferenceSourceKind>::new();
        for target in desired_target_ids {
            desired.entry(target.node_id).or_insert(target.source_kind);
        }
        let desired_ids: std::collections::HashSet<uuid::Uuid> = desired.keys().copied().collect();
        let existing = self.graph.get_relationships_from(from_node_id).await?;

        let mut existing_auto_map = std::collections::HashMap::<uuid::Uuid, Relationship>::new();
        for rel in &existing {
            if rel.kind == RelationKind::References && is_auto_backlink_relationship(rel) {
                existing_auto_map.insert(rel.to_node, rel.clone());
            }
        }

        for (target_id, source_kind) in desired {
            let desired_source = source_kind.as_str();
            if let Some(existing_rel) = existing_auto_map.get(&target_id) {
                let existing_source = existing_rel
                    .metadata
                    .get(AUTO_BACKLINK_SOURCE_METADATA_KEY)
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("");
                if existing_source == desired_source {
                    continue;
                }
                self.graph.remove_relationship(existing_rel.id).await?;
            }

            let mut relationship =
                Relationship::new(from_node_id, target_id, RelationKind::References);
            relationship.metadata.insert(
                AUTO_BACKLINK_METADATA_KEY.to_string(),
                serde_json::Value::Bool(true),
            );
            relationship.metadata.insert(
                AUTO_BACKLINK_SOURCE_METADATA_KEY.to_string(),
                serde_json::Value::String(desired_source.to_string()),
            );
            self.graph.add_relationship(&relationship).await?;
        }

        for rel in &existing {
            if rel.kind != RelationKind::References || !is_auto_backlink_relationship(rel) {
                continue;
            }
            if desired_ids.contains(&rel.to_node) {
                continue;
            }
            self.graph.remove_relationship(rel.id).await?;
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // Agentic Intelligence Methods
    // -------------------------------------------------------------------------

    /// List captured intents with optional filters.
    pub async fn list_intents(
        &self,
        node_id: Option<Uuid>,
        status: Option<IntentStatus>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<CapturedIntent>> {
        self.store
            .nodes
            .list_intents(node_id, status, limit, offset)
            .await
    }

    /// Update the status of an intent (apply/dismiss).
    pub async fn update_intent_status(&self, id: Uuid, status: IntentStatus) -> MvResult<bool> {
        self.store.nodes.update_intent_status(id, status).await
    }

    /// Get a single intent by ID.
    pub async fn get_intent(&self, id: Uuid) -> MvResult<Option<CapturedIntent>> {
        self.store.nodes.get_intent(id).await
    }

    /// Apply an intent: execute the action and mark as applied.
    pub async fn apply_intent(
        self: &Arc<Self>,
        id: Uuid,
    ) -> MvResult<crate::intent_executor::ExecutionResult> {
        // Get the intent
        let intent = self
            .store
            .nodes
            .get_intent(id)
            .await?
            .ok_or_else(|| HxError::InvalidInput(format!("Intent {} not found", id)))?;

        // Execute the intent
        let executor = crate::intent_executor::IntentExecutor::new(Arc::clone(self));
        let result = executor.execute(&intent).await?;

        // If execution succeeded, mark as applied
        if result.success {
            self.store
                .nodes
                .update_intent_status(id, IntentStatus::Applied)
                .await?;
        }

        Ok(result)
    }

    /// List proactive insights.
    pub async fn list_insights(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ProactiveInsight>> {
        self.store.nodes.list_insights(limit, offset).await
    }

    /// Delete (dismiss) an insight.
    pub async fn delete_insight(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_insight(id).await
    }

    // --- Conflict Detection ---

    /// List conflict alerts.
    pub async fn list_conflicts(
        &self,
        resolved: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ConflictAlert>> {
        self.store
            .nodes
            .list_conflicts(resolved, limit, offset)
            .await
    }

    /// Get a single conflict alert.
    pub async fn get_conflict(&self, id: Uuid) -> MvResult<Option<ConflictAlert>> {
        self.store.nodes.get_conflict(id).await
    }

    /// Resolve (dismiss) a conflict alert.
    pub async fn resolve_conflict(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.resolve_conflict(id).await
    }

    // --- Contact Identity & Trust ---

    /// Add an identity to a relay contact.
    pub async fn add_contact_identity(&self, identity: &ContactIdentity) -> MvResult<()> {
        self.store.nodes.add_contact_identity(identity).await
    }

    /// List identities for a contact.
    pub async fn list_contact_identities(
        &self,
        contact_id: Uuid,
    ) -> MvResult<Vec<ContactIdentity>> {
        self.store.nodes.list_contact_identities(contact_id).await
    }

    /// Delete a contact identity.
    pub async fn delete_contact_identity(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_contact_identity(id).await
    }

    /// Verify a contact identity.
    pub async fn verify_contact_identity(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.verify_contact_identity(id).await
    }

    /// Get trust model for a contact.
    pub async fn get_trust_model(&self, contact_id: Uuid) -> MvResult<Option<TrustModel>> {
        self.store.nodes.get_trust_model(contact_id).await
    }

    /// Set trust model for a contact.
    pub async fn set_trust_model(&self, model: &TrustModel) -> MvResult<()> {
        self.store.nodes.set_trust_model(model).await
    }

    /// List chronicle entries with optional node filter.
    pub async fn list_chronicles(
        &self,
        node_id: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ChronicleEntry>> {
        self.store
            .nodes
            .list_chronicles(node_id, limit, offset)
            .await
    }

    /// Log a chronicle entry for transparency.
    pub async fn log_chronicle(&self, entry: &ChronicleEntry) -> MvResult<()> {
        self.store.nodes.log_chronicle(entry).await
    }

    // --- Feedback / Learning ---

    /// Record feedback for an intent action (apply/dismiss).
    pub async fn record_feedback(&self, fb: &AgentFeedback) -> MvResult<()> {
        self.store.nodes.record_feedback(fb).await
    }

    /// Get acceptance rate for an intent type. Returns (total, applied).
    pub async fn get_acceptance_rate(&self, intent_type: &str) -> MvResult<(usize, usize)> {
        self.store.nodes.get_acceptance_rate(intent_type).await
    }

    /// Get confidence override for an intent type.
    pub async fn get_confidence_override(
        &self,
        intent_type: &str,
    ) -> MvResult<Option<ConfidenceOverride>> {
        self.store.nodes.get_confidence_override(intent_type).await
    }

    /// Recalculate and store a confidence override based on accumulated feedback.
    pub async fn recalculate_confidence(&self, intent_type: &str) -> MvResult<()> {
        let (total, applied) = self.store.nodes.get_acceptance_rate(intent_type).await?;

        // Need at least 5 data points to start adjusting
        if total < 5 {
            return Ok(());
        }

        let rate = applied as f32 / total as f32;

        // base_adjustment: -0.2 to +0.2 based on acceptance rate
        // 50% → 0.0, 100% → +0.2, 0% → -0.2
        let base_adjustment = (rate - 0.5) * 0.4;

        // auto_apply_threshold: lower if acceptance rate is high
        let auto_apply_threshold = if rate > 0.9 && total >= 20 {
            0.9 // auto-apply above 0.9 confidence
        } else {
            0.95 // default: very high threshold
        };

        // suppress_below: raise if acceptance rate is very low
        let suppress_below = if rate < 0.1 && total >= 10 {
            0.5 // suppress weak suggestions for disliked intent types
        } else if rate < 0.3 {
            0.3
        } else {
            0.1 // default
        };

        let override_ = ConfidenceOverride {
            intent_type: intent_type.to_string(),
            base_adjustment,
            auto_apply_threshold,
            suppress_below,
            updated_at: chrono::Utc::now(),
        };

        self.store.nodes.set_confidence_override(&override_).await
    }

    // --- Exchange Inbox ---

    pub async fn submit_proposal(&self, proposal: &Proposal) -> MvResult<()> {
        self.store.nodes.submit_proposal(proposal).await
    }

    pub async fn get_proposal(&self, id: Uuid) -> MvResult<Option<Proposal>> {
        self.store.nodes.get_proposal(id).await
    }

    pub async fn list_proposals(
        &self,
        state: Option<ProposalState>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<Proposal>> {
        self.store.nodes.list_proposals(state, limit, offset).await
    }

    pub async fn resolve_proposal(&self, id: Uuid, state: ProposalState) -> MvResult<bool> {
        self.store.nodes.resolve_proposal(id, state).await
    }

    pub async fn count_proposals(&self, state: Option<ProposalState>) -> MvResult<usize> {
        self.store.nodes.count_proposals(state).await
    }

    pub async fn expire_proposals(&self, before: DateTime<Utc>) -> MvResult<usize> {
        self.store.nodes.expire_proposals(before).await
    }
}

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

fn is_template_node(node: &KnowledgeNode) -> bool {
    node.kind == NodeKind::Template
        || node
            .metadata
            .get("template")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false)
        || node
            .tags
            .iter()
            .any(|tag| tag.eq_ignore_ascii_case("template"))
}

fn is_recurring_instance(node: &KnowledgeNode) -> bool {
    node.metadata
        .get(RECURRING_INSTANCE_METADATA_KEY)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn is_daily_link_candidate(node: &KnowledgeNode) -> bool {
    if matches!(node.kind, NodeKind::Task | NodeKind::Event) {
        return true;
    }

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
    credentials: &CredentialStore,
) -> EmbeddingProviderSelection {
    let provider = config.embedding.provider.trim().to_ascii_lowercase();
    let configured_model = config.embedding.model.clone();
    let configured_dimensions = config.embedding.dimensions;

    let base_status =
        |effective_provider: &str,
         effective_model: String,
         effective_dimensions: usize,
         fallback_to_noop: bool,
         reason: Option<String>| KnowledgeVaultIndexNoteEmbeddingProviderRuntimeStatus {
            configured_provider: config.embedding.provider.clone(),
            configured_model: configured_model.clone(),
            configured_dimensions,
            effective_provider: effective_provider.to_string(),
            effective_model,
            effective_dimensions,
            fallback_to_noop,
            reason,
            local_embeddings_feature_enabled: cfg!(feature = "local-embeddings"),
        };

    match provider.as_str() {
        "openai" => {
            let base_url = config
                .embedding
                .base_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
            let api_key = credentials.get_secret_string("OPENAI_API_KEY");
            if api_key.is_none() && base_url.contains("api.openai.com") {
                let reason = "OPENAI_API_KEY not found in any credential backend".to_string();
                tracing::warn!("helix_openai_embedder_unavailable_falling_back_to_noop");
                return EmbeddingProviderSelection {
                    embedder: None,
                    vector_dimensions: config.embedding.dimensions,
                    runtime_status: base_status(
                        "noop",
                        "noop".to_string(),
                        config.embedding.dimensions,
                        true,
                        Some(reason),
                    ),
                };
            }
            let embedder = OpenAiEmbedder::for_compatible(
                base_url,
                api_key,
                config.embedding.model.clone(),
                config.embedding.dimensions,
            );
            tracing::info!(
                provider = "openai",
                dimensions = config.embedding.dimensions,
                "helix_embedding_provider_initialized"
            );
            EmbeddingProviderSelection {
                embedder: Some(Arc::new(embedder)),
                vector_dimensions: config.embedding.dimensions,
                runtime_status: base_status(
                    "openai",
                    configured_model.clone(),
                    config.embedding.dimensions,
                    false,
                    None,
                ),
            }
        }
        "openai-compatible" | "openai_compatible" => {
            let base_url = config
                .embedding
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:8080/v1".to_string());
            let api_key = credentials
                .get_secret_string("HELIX_EMBEDDING_API_KEY")
                .or_else(|| credentials.get_secret_string("OPENAI_API_KEY"));
            let embedder = OpenAiEmbedder::for_compatible(
                base_url.clone(),
                api_key,
                config.embedding.model.clone(),
                config.embedding.dimensions,
            );
            tracing::info!(
                provider = "openai-compatible",
                base_url = %base_url,
                model = %config.embedding.model,
                dimensions = config.embedding.dimensions,
                "helix_embedding_provider_initialized"
            );
            EmbeddingProviderSelection {
                embedder: Some(Arc::new(embedder)),
                vector_dimensions: config.embedding.dimensions,
                runtime_status: base_status(
                    "openai-compatible",
                    configured_model.clone(),
                    config.embedding.dimensions,
                    false,
                    None,
                ),
            }
        }
        "ollama" => {
            let base_url = config
                .embedding
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434/v1".to_string());
            let model = if config.embedding.model.starts_with("text-embedding-") {
                "nomic-embed-text".to_string()
            } else {
                config.embedding.model.clone()
            };
            let embedder = OpenAiEmbedder::for_ollama(
                Some(base_url.clone()),
                model.clone(),
                config.embedding.dimensions,
            );
            tracing::info!(
                provider = "ollama",
                base_url = %base_url,
                model = %model,
                dimensions = config.embedding.dimensions,
                "helix_embedding_provider_initialized"
            );
            let reason = if model != configured_model {
                Some(format!(
                    "model '{configured_model}' auto-mapped to '{model}' for ollama"
                ))
            } else {
                None
            };
            EmbeddingProviderSelection {
                embedder: Some(Arc::new(embedder)),
                vector_dimensions: config.embedding.dimensions,
                runtime_status: base_status(
                    "ollama",
                    model,
                    config.embedding.dimensions,
                    false,
                    reason,
                ),
            }
        }
        "noop" | "none" | "disabled" => {
            tracing::info!(provider = "noop", "helix_embedding_provider_noop");
            EmbeddingProviderSelection {
                embedder: None,
                vector_dimensions: config.embedding.dimensions,
                runtime_status: base_status(
                    "noop",
                    "noop".to_string(),
                    config.embedding.dimensions,
                    false,
                    None,
                ),
            }
        }
        "local_fastembed" | "fastembed" | "local" => {
            let local_model = default_local_model_if_needed(&config.embedding.model);
            match KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder::try_new(&local_model) {
                Ok(embedder) => {
                    let dimensions = embedder.dimensions();
                    tracing::info!(
                        provider = "local_fastembed",
                        model = %embedder.model_name(),
                        dimensions,
                        "helix_embedding_provider_initialized"
                    );
                    let reason = if local_model != configured_model {
                        Some(format!(
                            "model '{configured_model}' auto-mapped to '{local_model}' for local_fastembed"
                        ))
                    } else {
                        None
                    };
                    EmbeddingProviderSelection {
                        embedder: Some(Arc::new(embedder)),
                        vector_dimensions: dimensions,
                        runtime_status: base_status(
                            "local_fastembed",
                            local_model,
                            dimensions,
                            false,
                            reason,
                        ),
                    }
                }
                Err(err) => {
                    let reason = format!("local_fastembed initialization failed: {err}");
                    tracing::warn!(
                        provider = "local_fastembed",
                        model = %local_model,
                        error = %err,
                        "helix_local_embedder_unavailable_falling_back_to_noop"
                    );
                    EmbeddingProviderSelection {
                        embedder: None,
                        vector_dimensions: config.embedding.dimensions,
                        runtime_status: base_status(
                            "noop",
                            "noop".to_string(),
                            config.embedding.dimensions,
                            true,
                            Some(reason),
                        ),
                    }
                }
            }
        }
        other => {
            let reason = format!("unknown embedding provider '{other}'");
            tracing::warn!(
                provider = %other,
                "helix_unknown_embedding_provider_falling_back_to_noop"
            );
            EmbeddingProviderSelection {
                embedder: None,
                vector_dimensions: config.embedding.dimensions,
                runtime_status: base_status(
                    "noop",
                    "noop".to_string(),
                    config.embedding.dimensions,
                    true,
                    Some(reason),
                ),
            }
        }
    }
}

fn default_local_model_if_needed(config_model: &str) -> String {
    if config_model.starts_with("text-embedding-") {
        tracing::info!(
            configured_model = %config_model,
            fallback_model = "bge-small-en-v1.5",
            "helix_local_embedder_model_auto_selected"
        );
        "bge-small-en-v1.5".to_string()
    } else {
        config_model.to_string()
    }
}

fn generate_access_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let encoded = URL_SAFE_NO_PAD.encode(bytes);
    format!("mvk_{encoded}")
}

fn generate_share_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let encoded = URL_SAFE_NO_PAD.encode(bytes);
    format!("mvs_{encoded}")
}

fn hash_access_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    URL_SAFE_NO_PAD.encode(digest)
}

fn hash_share_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    URL_SAFE_NO_PAD.encode(digest)
}

#[derive(Debug)]
enum GoogleCalendarFetchError {
    SyncTokenExpired,
    RequestFailed(String),
}

#[derive(Debug, serde::Deserialize)]
struct GoogleTokenResponse {
    access_token: String,
}

#[derive(Debug, serde::Deserialize)]
struct GoogleEventsResponse {
    items: Option<Vec<GoogleEvent>>,
    #[serde(rename = "nextPageToken")]
    next_page_token: Option<String>,
    #[serde(rename = "nextSyncToken")]
    next_sync_token: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct GoogleEvent {
    id: Option<String>,
    summary: Option<String>,
    description: Option<String>,
    status: Option<String>,
    updated: Option<String>,
    #[serde(rename = "htmlLink")]
    html_link: Option<String>,
    start: Option<GoogleEventTime>,
    end: Option<GoogleEventTime>,
}

#[derive(Debug, serde::Deserialize)]
struct GoogleEventTime {
    #[serde(rename = "dateTime")]
    date_time: Option<String>,
    date: Option<String>,
}

async fn google_refresh_access_token(
    client_id: &str,
    client_secret: &str,
    refresh_token: &str,
) -> MvResult<String> {
    let client = reqwest::Client::new();
    let response = client
        .post("https://oauth2.googleapis.com/token")
        .form(&[
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("refresh_token", refresh_token),
            ("grant_type", "refresh_token"),
        ])
        .send()
        .await
        .map_err(|e| HxError::Storage(format!("google token request failed: {e}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(HxError::Storage(format!(
            "google token request failed ({status}): {body}"
        )));
    }

    let token = response
        .json::<GoogleTokenResponse>()
        .await
        .map_err(|e| HxError::Storage(format!("google token response parse failed: {e}")))?;

    Ok(token.access_token)
}

async fn google_list_events(
    access_token: &str,
    calendar_id: &str,
    config: &crate::config::GoogleCalendarConfig,
    sync_token: Option<&str>,
) -> Result<(Vec<GoogleEvent>, Option<String>), GoogleCalendarFetchError> {
    let client = reqwest::Client::new();
    let encoded_calendar = byte_serialize(calendar_id.as_bytes()).collect::<String>();
    let url = format!("https://www.googleapis.com/calendar/v3/calendars/{encoded_calendar}/events");

    let max_results = config.max_results.to_string();
    let mut page_token: Option<String> = None;
    let mut events: Vec<GoogleEvent> = Vec::new();

    let next_sync_token = loop {
        let mut request = client.get(&url).bearer_auth(access_token).query(&[
            ("singleEvents", "true"),
            ("showDeleted", "true"),
            ("maxResults", max_results.as_str()),
        ]);

        if let Some(token) = sync_token {
            request = request.query(&[("syncToken", token)]);
        } else {
            let now = Utc::now();
            let time_min = (now - chrono::Duration::days(config.lookback_days)).to_rfc3339();
            let time_max = (now + chrono::Duration::days(config.lookahead_days)).to_rfc3339();
            request = request.query(&[
                ("timeMin", time_min.as_str()),
                ("timeMax", time_max.as_str()),
                ("orderBy", "startTime"),
            ]);
        }

        if let Some(ref token) = page_token {
            request = request.query(&[("pageToken", token.as_str())]);
        }

        let response = request
            .send()
            .await
            .map_err(|e| GoogleCalendarFetchError::RequestFailed(e.to_string()))?;

        if response.status() == HttpStatusCode::GONE {
            return Err(GoogleCalendarFetchError::SyncTokenExpired);
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(GoogleCalendarFetchError::RequestFailed(format!(
                "google events request failed ({status}): {body}"
            )));
        }

        let payload = response
            .json::<GoogleEventsResponse>()
            .await
            .map_err(|e| GoogleCalendarFetchError::RequestFailed(e.to_string()))?;

        if let Some(mut page_items) = payload.items {
            events.append(&mut page_items);
        }

        if payload.next_page_token.is_none() {
            break payload.next_sync_token;
        }

        page_token = payload.next_page_token;
    };

    Ok((events, next_sync_token))
}

fn event_source(calendar_id: &str, event_id: &str) -> String {
    format!("google-calendar:{calendar_id}:{event_id}")
}

fn parse_google_event_time(time: &GoogleEventTime) -> Option<DateTime<Utc>> {
    if let Some(ref dt) = time.date_time {
        return DateTime::parse_from_rfc3339(dt)
            .ok()
            .map(|dt| dt.with_timezone(&Utc));
    }

    let date_str = time.date.as_ref()?;
    let date = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()?;
    let naive = date.and_hms_opt(0, 0, 0)?;
    Some(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

fn event_times(event: &GoogleEvent) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
    let start = event.start.as_ref().and_then(parse_google_event_time)?;
    let end = event
        .end
        .as_ref()
        .and_then(parse_google_event_time)
        .unwrap_or_else(|| start + chrono::Duration::hours(1));
    Some((start, end))
}

async fn google_export_events(
    engine: &HelixEngine,
    access_token: &str,
    calendar_id: &str,
    config: &crate::config::GoogleCalendarConfig,
) -> MvResult<(usize, usize)> {
    let mut exported_created = 0usize;
    let mut exported_updated = 0usize;

    let filters = QueryFilters {
        namespace: Some(config.namespace.clone()),
        kinds: Some(vec![NodeKind::Event]),
        tags: None,
        min_importance: None,
        created_after: None,
        created_before: None,
    };

    let nodes = engine.store.nodes.list(&filters, 1000, 0).await?;
    let client = reqwest::Client::new();
    let encoded_calendar = byte_serialize(calendar_id.as_bytes()).collect::<String>();
    let url = format!("https://www.googleapis.com/calendar/v3/calendars/{encoded_calendar}/events");

    for mut node in nodes {
        let start_at = node
            .metadata
            .get("event_start_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.to_rfc3339());
        let end_at = node
            .metadata
            .get("event_end_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.to_rfc3339());

        let (Some(start_at), Some(end_at)) = (start_at, end_at) else {
            continue;
        };

        let summary = node
            .title
            .clone()
            .unwrap_or_else(|| "Helix Event".to_string());
        let description = node.content.clone();

        let mut payload = serde_json::json!({
            "summary": summary,
            "description": description,
            "start": { "dateTime": start_at },
            "end": { "dateTime": end_at }
        });

        if let Some(ref source) = node.source {
            payload["source"] = serde_json::json!({ "title": "Helix", "url": source });
        }

        if let Some(event_id) = node
            .metadata
            .get("google_calendar_event_id")
            .and_then(|v| v.as_str())
        {
            let request = client
                .patch(format!("{url}/{event_id}"))
                .bearer_auth(access_token)
                .json(&payload);

            let response = request
                .send()
                .await
                .map_err(|e| HxError::Storage(format!("google event update failed: {e}")))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(HxError::Storage(format!(
                    "google event update failed ({status}): {body}"
                )));
            }

            exported_updated += 1;
            continue;
        }

        let response = client
            .post(&url)
            .bearer_auth(access_token)
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Storage(format!("google event create failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(HxError::Storage(format!(
                "google event create failed ({status}): {body}"
            )));
        }

        let created_event = response
            .json::<GoogleEvent>()
            .await
            .map_err(|e| HxError::Storage(format!("google event response parse failed: {e}")))?;

        if let Some(event_id) = created_event.id {
            node.metadata.insert(
                "google_calendar_event_id".to_string(),
                serde_json::Value::String(event_id.clone()),
            );
            node.metadata.insert(
                "google_calendar_calendar_id".to_string(),
                serde_json::Value::String(calendar_id.to_string()),
            );
            if let Some(html_link) = created_event.html_link {
                node.metadata.insert(
                    "google_calendar_html_link".to_string(),
                    serde_json::Value::String(html_link),
                );
            }
            if node.source.is_none() {
                node.source = Some(event_source(calendar_id, &event_id));
            }
            node.temporal.updated_at = Utc::now();
            let _ = engine.update_node(node).await?;
            exported_created += 1;
        }
    }

    Ok((exported_created, exported_updated))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{NaiveDate, TimeZone, Utc};
    use hx_core::{
        ConflictAlert, ConflictType, ContactIdentity, GraphStore, IdentityType, InsightType,
        KnowledgeNode, MessageStatus, NodeKind, ProactiveInsight, ProposalAction, ProposalState,
        RelationKind, Relationship, RelayChannel, RelayContact, RelayMessage, SearchStrategy,
        TrustLevel, TrustModel,
    };
    use tempfile::TempDir;

    async fn create_test_engine() -> (HelixEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = EngineConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        config.embedding.provider = "noop".into();
        let engine = HelixEngine::init(config).await.unwrap();
        (engine, temp_dir)
    }

    async fn create_test_engine_with_ai_auto_tagging() -> (HelixEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = EngineConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        config.embedding.provider = "noop".into();
        config.ai.auto_tagging_enabled = true;
        config.ai.auto_tagging_similarity_seed_limit = 8;
        config.ai.auto_tagging_max_generated_tags = 6;
        config.ai.auto_tagging_max_total_tags = 12;
        let engine = HelixEngine::init(config).await.unwrap();
        (engine, temp_dir)
    }

    async fn create_test_engine_with_local_embedding_provider(
        provider: &str,
        model: &str,
    ) -> (HelixEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = EngineConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        config.embedding.provider = provider.to_string();
        config.embedding.model = model.to_string();
        let engine = HelixEngine::init(config).await.unwrap();
        (engine, temp_dir)
    }

    async fn create_test_engine_with_backlinks_disabled() -> (HelixEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = EngineConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        config.embedding.provider = "noop".into();
        config.linking.auto_backlinks_enabled = false;
        let engine = HelixEngine::init(config).await.unwrap();
        (engine, temp_dir)
    }

    async fn create_test_sealed_engine(unseal_vault: bool) -> (HelixEngine, TempDir) {
        let temp_dir = TempDir::new().unwrap();
        let mut config = EngineConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        config.embedding.provider = "noop".into();
        config.sealed_mode = true;
        let engine = HelixEngine::init(config).await.unwrap();
        engine
            .keychain
            .initialize_vault("test-password", false, "test-suite")
            .await
            .unwrap();
        if unseal_vault {
            engine
                .keychain
                .unseal("test-password", "test-suite")
                .await
                .unwrap();
        } else {
            engine.keychain.seal("test-suite").await.unwrap();
        }
        (engine, temp_dir)
    }

    #[tokio::test]
    async fn test_store_and_retrieve_node() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let node = KnowledgeNode::new(NodeKind::Fact, "Test content".to_string())
            .with_title("Test Title")
            .with_tags(vec!["test".to_string()]);

        let stored_node = engine.store_node(node.clone()).await.unwrap();
        assert_eq!(stored_node.content, "Test content");
        assert_eq!(stored_node.title, Some("Test Title".to_string()));

        let retrieved = engine.get_node(stored_node.id).await.unwrap().unwrap();
        assert_eq!(retrieved.content, "Test content");
        assert_eq!(retrieved.tags, vec!["test"]);
    }

    #[tokio::test]
    async fn test_update_node() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let node = KnowledgeNode::new(NodeKind::Fact, "Original content".to_string());
        let stored = engine.store_node(node).await.unwrap();

        let mut updated = stored.clone();
        updated.content = "Updated content".to_string();
        updated.title = Some("Updated title".to_string());

        let updated_node = engine.update_node(updated).await.unwrap();
        assert_eq!(updated_node.content, "Updated content");
        assert_eq!(updated_node.title, Some("Updated title".to_string()));
    }

    #[tokio::test]
    async fn test_delete_node() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let node = KnowledgeNode::new(NodeKind::Fact, "To delete".to_string());
        let stored = engine.store_node(node).await.unwrap();

        assert!(engine.delete_node(stored.id).await.unwrap());
        assert!(engine.get_node(stored.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn test_node_count() {
        let (engine, _tmp_dir) = create_test_engine().await;

        let count = engine.node_count().await.unwrap();
        assert_eq!(count, 0);

        engine
            .store_node(KnowledgeNode::new(NodeKind::Fact, "First".to_string()))
            .await
            .unwrap();
        engine
            .store_node(KnowledgeNode::new(NodeKind::Fact, "Second".to_string()))
            .await
            .unwrap();

        let count = engine.node_count().await.unwrap();
        assert_eq!(count, 2);
    }

    #[tokio::test]
    async fn test_sealed_engine_blocks_node_io_while_sealed() {
        let (engine, _tmp_dir) = create_test_sealed_engine(false).await;
        let err = engine
            .store_node(KnowledgeNode::new(NodeKind::Fact, "blocked".to_string()))
            .await
            .expect_err("sealed store_node must fail");
        assert!(matches!(err, HxError::VaultSealed));
    }

    #[tokio::test]
    async fn test_sealed_migrate_and_rebuild_require_unseal_then_succeed() {
        let (engine, _tmp_dir) = create_test_sealed_engine(false).await;

        let rebuild_err = engine
            .rebuild_runtime_indexes()
            .await
            .expect_err("sealed rebuild should fail");
        assert!(matches!(rebuild_err, HxError::VaultSealed));

        let migrate_err = engine
            .migrate_sealed_storage()
            .await
            .expect_err("sealed migrate should fail");
        assert!(matches!(migrate_err, HxError::VaultSealed));

        engine
            .keychain
            .unseal("test-password", "test-suite")
            .await
            .unwrap();

        let node = engine
            .store_node(KnowledgeNode::new(
                NodeKind::Fact,
                "sealed migration validation".to_string(),
            ))
            .await
            .unwrap();

        engine
            .rebuild_runtime_indexes()
            .await
            .expect("rebuild after unseal");
        engine
            .migrate_sealed_storage()
            .await
            .expect("migrate after unseal");

        let loaded = engine
            .get_node(node.id)
            .await
            .expect("load node")
            .expect("node exists");
        assert_eq!(loaded.content, "sealed migration validation");
    }

    #[tokio::test]
    async fn test_sealed_restart_cycle_recovers_data_after_unseal_and_rebuild() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().to_string_lossy().to_string();

        let mut config = EngineConfig {
            data_dir: data_dir.clone(),
            ..Default::default()
        };
        config.embedding.provider = "noop".into();
        config.sealed_mode = true;

        let engine = HelixEngine::init(config.clone()).await.unwrap();
        engine
            .keychain
            .initialize_vault("restart-password", false, "test-suite")
            .await
            .unwrap();

        let stored = engine
            .store_node(KnowledgeNode::new(
                NodeKind::Fact,
                "restart lifecycle keeps encrypted knowledge".to_string(),
            ))
            .await
            .unwrap();

        engine.keychain.seal("test-suite").await.unwrap();
        drop(engine);

        let restarted = HelixEngine::init(config).await.unwrap();
        assert!(
            restarted.is_sealed(),
            "engine should start sealed after restart"
        );

        let sealed_err = restarted
            .get_node(stored.id)
            .await
            .expect_err("sealed restart should block node reads");
        assert!(matches!(sealed_err, HxError::VaultSealed));

        restarted
            .keychain
            .unseal("restart-password", "test-suite")
            .await
            .unwrap();
        restarted
            .migrate_sealed_storage()
            .await
            .expect("migrate after restart");
        restarted
            .rebuild_runtime_indexes()
            .await
            .expect("rebuild after restart");

        let loaded = restarted
            .get_node(stored.id)
            .await
            .expect("load node")
            .expect("node exists");
        assert_eq!(
            loaded.content,
            "restart lifecycle keeps encrypted knowledge"
        );

        let recall_results = restarted
            .recall(
                &MemoryQuery::new("restart lifecycle")
                    .with_strategy(SearchStrategy::FullText)
                    .with_limit(10)
                    .with_min_score(0.0),
            )
            .await
            .expect("recall should work after restart rebuild");
        assert!(
            recall_results
                .iter()
                .any(|result| result.node.id == stored.id),
            "recalled results should include the stored node"
        );
    }

    #[tokio::test]
    async fn test_relationships() {
        let (engine, _tmp_dir) = create_test_engine().await;

        let node1 = engine
            .store_node(KnowledgeNode::new(NodeKind::Fact, "Node 1".to_string()))
            .await
            .unwrap();
        let node2 = engine
            .store_node(KnowledgeNode::new(NodeKind::Fact, "Node 2".to_string()))
            .await
            .unwrap();

        let relation = Relationship {
            id: uuid::Uuid::new_v4(),
            from_node: node1.id,
            to_node: node2.id,
            kind: RelationKind::RelatesTo,
            weight: 1.0,
            metadata: Default::default(),
            created_at: chrono::Utc::now(),
        };

        engine.add_relationship(relation.clone()).await.unwrap();

        let neighbors = engine.get_neighbors(node1.id, 1).await.unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], node2.id);
    }

    #[tokio::test]
    async fn test_update_profile_syncs_owner_relay_contact() {
        let (engine, _tmp_dir) = create_test_engine().await;

        let updated = engine
            .update_profile(&UpdateProfileRequest {
                display_name: Some("Owner".to_string()),
                email: Some("owner@example.com".to_string()),
                signature_public_key: Some("pk-owner".to_string()),
                ..Default::default()
            })
            .await
            .unwrap();

        let contact_id = updated
            .metadata
            .get("relay_contact_id")
            .and_then(|value| value.as_str())
            .and_then(|value| uuid::Uuid::parse_str(value).ok())
            .expect("relay_contact_id set");

        let contact = engine.relay.get_contact(contact_id).await.unwrap().unwrap();
        assert_eq!(contact.display_name, "Owner");
        assert_eq!(contact.public_key, "pk-owner");
        assert_eq!(
            contact.vault_address.as_deref(),
            Some("mailto:owner@example.com")
        );
    }

    #[tokio::test]
    async fn test_relay_inbound_creates_reply_proposal() {
        let (engine, _tmp_dir) = create_test_engine().await;

        let contact = RelayContact::new("Alice", "pk-alice").with_trust(TrustLevel::ContextInject);
        engine.relay.add_contact(&contact).await.unwrap();

        let channel = RelayChannel::direct(contact.id);
        engine.relay.create_channel(&channel).await.unwrap();

        let node = KnowledgeNode::new(
            NodeKind::Fact,
            "Project Atlas roadmap lives in the Q2 plan.".to_string(),
        );
        engine.store_node(node).await.unwrap();

        let message =
            RelayMessage::inbound(channel.id, contact.id, "Can you share the Atlas roadmap?");
        let outcome = engine
            .receive_relay_message(message, "default")
            .await
            .unwrap();

        assert!(outcome.proposal_id.is_some());
        assert!(outcome.auto_reply.is_none());
        assert_eq!(outcome.message.status, MessageStatus::Deferred);

        let proposals = engine
            .list_proposals(Some(ProposalState::Pending), 10, 0)
            .await
            .unwrap();
        assert!(proposals.iter().any(|proposal| {
            proposal.action == ProposalAction::Custom("relay.reply".to_string())
        }));
    }

    #[tokio::test]
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
    async fn test_rollforward_recurring_tasks_generates_due_instance() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = "ops".to_string();
        let mut template =
            KnowledgeNode::new(NodeKind::Task, "Daily standup action checklist".to_string())
                .with_namespace(namespace.clone())
                .with_tags(vec!["ops".to_string()]);

        let last_generated = Utc
            .with_ymd_and_hms(2026, 2, 5, 9, 0, 0)
            .single()
            .expect("valid datetime");
        template.metadata.insert(
            TASK_RECURRENCE_METADATA_KEY.into(),
            serde_json::json!({
                "frequency": "daily",
                "interval": 1,
                "enabled": true
            }),
        );
        template.metadata.insert(
            TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY.into(),
            serde_json::Value::String(last_generated.to_rfc3339()),
        );

        let stored_template = engine.store_node(template).await.unwrap();
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 9, 0, 0)
            .single()
            .expect("valid datetime");
        let stats = engine.rollforward_recurring_tasks(now, 4).await.unwrap();
        assert_eq!(stats.generated_instances, 1);

        let tasks = engine
            .list_nodes(
                &QueryFilters {
                    namespace: Some(namespace.clone()),
                    kinds: Some(vec![NodeKind::Task]),
                    ..Default::default()
                },
                50,
                0,
            )
            .await
            .unwrap();

        let instance = tasks
            .iter()
            .find(|node| {
                node.metadata
                    .get(RECURRING_INSTANCE_METADATA_KEY)
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .expect("recurring instance should exist");
        assert_eq!(
            instance
                .metadata
                .get(RECURRING_PARENT_ID_METADATA_KEY)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            Some(stored_template.id.to_string())
        );

        let rels = engine
            .graph
            .get_relationships_from(stored_template.id)
            .await
            .unwrap();
        assert!(rels
            .iter()
            .any(|rel| { rel.to_node == instance.id && rel.kind == RelationKind::DerivedFrom }));
    }

    #[tokio::test]
    async fn test_rollforward_recurring_tasks_is_idempotent_for_same_instant() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let mut template =
            KnowledgeNode::new(NodeKind::Task, "Weekly planning template".to_string())
                .with_namespace("ops");
        template.metadata.insert(
            TASK_RECURRENCE_METADATA_KEY.into(),
            serde_json::json!({
                "frequency": "weekly",
                "interval": 1,
                "enabled": true
            }),
        );
        template.metadata.insert(
            TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 1, 30, 10, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let _stored = engine.store_node(template).await.unwrap();

        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 10, 0, 0)
            .single()
            .expect("valid datetime");
        let first = engine.rollforward_recurring_tasks(now, 4).await.unwrap();
        let second = engine.rollforward_recurring_tasks(now, 4).await.unwrap();

        assert!(first.generated_instances >= 1);
        assert_eq!(second.generated_instances, 0);
    }

    #[tokio::test]
    async fn test_list_due_tasks_filters_and_sorts() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let base = Utc
            .with_ymd_and_hms(2026, 2, 6, 12, 0, 0)
            .single()
            .expect("valid datetime");

        let mut task_a = KnowledgeNode::new(NodeKind::Task, "A".to_string()).with_namespace("ops");
        task_a.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 10, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let mut task_b = KnowledgeNode::new(NodeKind::Task, "B".to_string()).with_namespace("ops");
        task_b.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 11, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let mut task_c = KnowledgeNode::new(NodeKind::Task, "C".to_string()).with_namespace("ops");
        task_c.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 9, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        task_c.metadata.insert(
            TASK_COMPLETED_METADATA_KEY.into(),
            serde_json::Value::Bool(true),
        );

        let _a = engine.store_node(task_a).await.unwrap();
        let _b = engine.store_node(task_b).await.unwrap();
        let _c = engine.store_node(task_c).await.unwrap();

        let due = engine
            .list_due_tasks(base, Some("ops".to_string()), 10, false)
            .await
            .unwrap();
        assert_eq!(due.len(), 2);
        assert_eq!(due[0].content, "A");
        assert_eq!(due[1].content, "B");
    }

    #[tokio::test]
    async fn test_prioritize_tasks_ranks_by_due_and_importance() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 12, 0, 0)
            .single()
            .expect("valid datetime");

        let mut urgent = KnowledgeNode::new(NodeKind::Task, "Urgent".to_string())
            .with_namespace("ops")
            .with_importance(0.9);
        urgent.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 18, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );

        let important = KnowledgeNode::new(NodeKind::Task, "Important".to_string())
            .with_namespace("ops")
            .with_importance(0.8);

        let mut later = KnowledgeNode::new(NodeKind::Task, "Later".to_string())
            .with_namespace("ops")
            .with_importance(0.2);
        later.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 16, 12, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );

        let _urgent = engine.store_node(urgent).await.unwrap();
        let _important = engine.store_node(important).await.unwrap();
        let _later = engine.store_node(later).await.unwrap();

        let prioritized = engine
            .prioritize_tasks(TaskPrioritizationOptions {
                namespace: Some("ops".to_string()),
                limit: 10,
                include_completed: false,
                include_without_due: true,
                persist: false,
                now,
            })
            .await
            .unwrap();

        assert_eq!(prioritized.len(), 3);
        assert_eq!(prioritized[0].task.content, "Urgent");
        assert_eq!(prioritized[1].task.content, "Important");
        assert_eq!(prioritized[2].task.content, "Later");
        assert_eq!(prioritized[0].rank, 1);
    }

    #[tokio::test]
    async fn test_dispatch_due_task_reminders_marks_once() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 12, 0, 0)
            .single()
            .expect("valid datetime");

        let mut task = KnowledgeNode::new(NodeKind::Task, "Follow up on incident".to_string())
            .with_namespace("ops");
        task.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 8, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let stored = engine.store_node(task).await.unwrap();

        let first = engine.dispatch_due_task_reminders(now, 20).await.unwrap();
        assert_eq!(first.reminders_marked_sent, 1);

        let second = engine.dispatch_due_task_reminders(now, 20).await.unwrap();
        assert_eq!(second.reminders_marked_sent, 0);

        let refreshed = engine
            .get_node(stored.id)
            .await
            .unwrap()
            .expect("task exists");
        assert!(refreshed
            .metadata
            .contains_key(TASK_REMINDER_SENT_AT_METADATA_KEY));
    }

    #[tokio::test]
    async fn test_default_permission_templates_seeded() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let templates = engine.list_permission_templates(10, 0).await.unwrap();
        let names: Vec<String> = templates.into_iter().map(|t| t.name).collect();
        assert!(names.contains(&"Owner".to_string()));
        assert!(names.contains(&"Assistant".to_string()));
    }

    #[tokio::test]
    async fn test_access_key_round_trip() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let template = engine
            .create_permission_template(
                "Scoped".to_string(),
                Some("Test".to_string()),
                PermissionTier::Edit,
                Some("ops".to_string()),
                vec!["shared".to_string()],
                vec![NodeKind::Fact],
                vec!["transform".to_string()],
            )
            .await
            .unwrap();

        let (_key, token) = engine
            .create_access_key(template.id, Some("Key".to_string()), None)
            .await
            .unwrap();

        let resolved = engine.resolve_access_key(&token).await.unwrap();
        assert!(resolved.is_some());
        let (_key, resolved_template) = resolved.unwrap();
        assert_eq!(resolved_template.id, template.id);
    }

    // -----------------------------------------------------------------------
    // Integration tests for Phase 1–3 features
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_insight_generate_and_list() {
        let (engine, _tmp) = create_test_engine().await;

        // Store a test insight directly via the underlying store
        let insight = ProactiveInsight {
            id: Uuid::now_v7(),
            title: "Test Insight".into(),
            content: "Something interesting".into(),
            insight_type: InsightType::General,
            related_node_ids: vec![],
            importance: 0.8,
            created_at: Utc::now(),
            dismissed_at: None,
        };
        engine.store.nodes.log_insight(&insight).await.unwrap();

        // List and verify
        let insights = engine.list_insights(10, 0).await.unwrap();
        assert!(!insights.is_empty(), "should have at least one insight");
        assert_eq!(insights[0].title, "Test Insight");

        // Dismiss (delete)
        let deleted = engine.delete_insight(insight.id).await.unwrap();
        assert!(deleted);

        let after = engine.list_insights(10, 0).await.unwrap();
        assert!(
            after.iter().all(|i| i.id != insight.id),
            "insight should be dismissed"
        );
    }

    #[tokio::test]
    async fn test_conflict_detection_on_contradictory_nodes() {
        let (engine, _tmp) = create_test_engine().await;

        // Store two nodes
        let node_a = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "The sky is blue").with_tags(vec!["sky".into()]),
            )
            .await
            .unwrap();
        let node_b = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "The sky is green")
                    .with_tags(vec!["sky".into()]),
            )
            .await
            .unwrap();

        // Manually insert a conflict alert
        let alert = ConflictAlert {
            id: Uuid::now_v7(),
            node_a: node_a.id,
            node_b: node_b.id,
            conflict_type: ConflictType::Contradiction,
            score: 0.95,
            explanation: "Contradictory claims about sky color".into(),
            resolved: false,
            created_at: Utc::now(),
        };
        engine.store.nodes.insert_conflict(&alert).await.unwrap();

        // List unresolved conflicts
        let conflicts = engine.list_conflicts(Some(false), 10, 0).await.unwrap();
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].node_a, node_a.id);

        // Resolve
        let resolved = engine.resolve_conflict(alert.id).await.unwrap();
        assert!(resolved);

        // Verify the conflict is now marked resolved
        let fetched = engine.get_conflict(alert.id).await.unwrap().unwrap();
        assert!(fetched.resolved, "conflict should be resolved");

        // Resolving again should return false (already resolved)
        let re_resolved = engine.resolve_conflict(alert.id).await.unwrap();
        assert!(!re_resolved);
    }

    #[tokio::test]
    async fn test_contact_identity_crud() {
        let (engine, _tmp) = create_test_engine().await;

        let contact_id = Uuid::now_v7();
        let identity = ContactIdentity {
            id: Uuid::now_v7(),
            contact_id,
            identity_type: IdentityType::Email,
            identity_value: "alice@example.com".into(),
            verified: false,
            verified_at: None,
            created_at: Utc::now(),
        };

        // Add
        engine.add_contact_identity(&identity).await.unwrap();

        // List
        let list = engine.list_contact_identities(contact_id).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].identity_value, "alice@example.com");
        assert!(!list[0].verified);

        // Verify
        let verified = engine.verify_contact_identity(identity.id).await.unwrap();
        assert!(verified);

        let list = engine.list_contact_identities(contact_id).await.unwrap();
        assert!(list[0].verified);

        // Delete
        let deleted = engine.delete_contact_identity(identity.id).await.unwrap();
        assert!(deleted);

        let list = engine.list_contact_identities(contact_id).await.unwrap();
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn test_trust_model_defaults_and_update() {
        let (engine, _tmp) = create_test_engine().await;

        let contact_id = Uuid::now_v7();

        // No trust model yet
        let model = engine.get_trust_model(contact_id).await.unwrap();
        assert!(model.is_none());

        // Set with defaults (all false)
        let mut tm = TrustModel {
            contact_id,
            ..Default::default()
        };
        engine.set_trust_model(&tm).await.unwrap();

        let stored = engine.get_trust_model(contact_id).await.unwrap().unwrap();
        assert!(!stored.can_query);
        assert!(!stored.can_inject_context);
        assert!(!stored.can_auto_reply);
        assert!(stored.allowed_namespaces.is_empty());

        // Update
        tm.can_query = true;
        tm.allowed_namespaces = vec!["research".into()];
        engine.set_trust_model(&tm).await.unwrap();

        let updated = engine.get_trust_model(contact_id).await.unwrap().unwrap();
        assert!(updated.can_query);
        assert_eq!(updated.allowed_namespaces, vec!["research"]);
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
