use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::error::MvResult;
use crate::model::*;

/// Storage backend for knowledge nodes (metadata + content).
#[async_trait]
pub trait NodeStore: Send + Sync {
    async fn insert(&self, node: &KnowledgeNode) -> MvResult<()>;
    async fn get(&self, id: Uuid) -> MvResult<Option<KnowledgeNode>>;
    async fn update(&self, node: &KnowledgeNode) -> MvResult<()>;
    async fn delete(&self, id: Uuid) -> MvResult<bool>;
    async fn list(
        &self,
        filters: &QueryFilters,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<KnowledgeNode>>;
    async fn touch(&self, id: Uuid) -> MvResult<()>;
    async fn count(&self, filters: &QueryFilters) -> MvResult<usize>;
}

/// Vector embedding storage + similarity search.
#[async_trait]
pub trait VectorStore: Send + Sync {
    async fn upsert(
        &self,
        id: Uuid,
        embedding: Vec<f32>,
        content: &str,
        namespace: Option<&str>,
    ) -> MvResult<()>;
    async fn search(
        &self,
        embedding: Vec<f32>,
        limit: usize,
        min_score: f64,
        namespace: Option<&str>,
    ) -> MvResult<Vec<(Uuid, f64)>>;
    async fn delete(&self, id: Uuid) -> MvResult<()>;
}

/// Full-text search index.
pub trait FullTextIndex: Send + Sync {
    fn index_node(&self, node: &KnowledgeNode) -> MvResult<()>;
    fn remove_node(&self, id: Uuid) -> MvResult<()>;
    fn search(&self, query: &str, limit: usize) -> MvResult<Vec<(Uuid, f64)>>;
    fn commit(&self) -> MvResult<()>;
}

/// Knowledge graph storage.
#[async_trait]
pub trait GraphStore: Send + Sync {
    async fn add_relationship(&self, rel: &Relationship) -> MvResult<()>;
    async fn get_relationship(&self, id: Uuid) -> MvResult<Option<Relationship>>;
    async fn remove_relationship(&self, id: Uuid) -> MvResult<bool>;
    async fn get_relationships_from(&self, node_id: Uuid) -> MvResult<Vec<Relationship>>;
    async fn get_relationships_to(&self, node_id: Uuid) -> MvResult<Vec<Relationship>>;
    async fn get_neighbors(&self, node_id: Uuid, depth: usize) -> MvResult<Vec<Uuid>>;
    async fn remove_node_relationships(&self, node_id: Uuid) -> MvResult<usize>;
}

/// Embedding provider.
#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed(&self, text: &str) -> MvResult<Vec<f32>>;
    async fn embed_batch(&self, texts: &[String]) -> MvResult<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

/// Keychain storage backend for vault metadata, credentials, delegations, and audit.
#[async_trait]
pub trait KeychainStore: Send + Sync {
    // --- Vault Meta ---
    async fn get_vault_meta(&self) -> MvResult<Option<VaultMeta>>;
    async fn save_vault_meta(&self, meta: &VaultMeta) -> MvResult<()>;

    // --- Key Epochs ---
    async fn insert_key_epoch(&self, epoch: &KeyEpoch) -> MvResult<()>;
    async fn get_key_epoch(&self, epoch: u64) -> MvResult<Option<KeyEpoch>>;
    async fn list_key_epochs(&self) -> MvResult<Vec<KeyEpoch>>;
    async fn retire_key_epoch(&self, epoch: u64) -> MvResult<()>;

    // --- Domains ---
    async fn insert_domain(&self, domain: &DomainKey) -> MvResult<()>;
    async fn get_domain(&self, id: Uuid) -> MvResult<Option<DomainKey>>;
    async fn get_domain_by_name(&self, name: &str) -> MvResult<Option<DomainKey>>;
    async fn list_domains(&self) -> MvResult<Vec<DomainKey>>;
    async fn revoke_domain(&self, id: Uuid) -> MvResult<()>;

    // --- Credentials ---
    async fn insert_credential(&self, cred: &StoredCredential) -> MvResult<()>;
    async fn get_credential(&self, id: Uuid) -> MvResult<Option<StoredCredential>>;
    async fn update_credential(&self, cred: &StoredCredential) -> MvResult<()>;
    async fn list_credentials(
        &self,
        domain_id: Option<Uuid>,
        state: Option<CredentialState>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<StoredCredential>>;
    async fn count_credentials(&self, domain_id: Option<Uuid>) -> MvResult<usize>;
    async fn shred_credential(&self, id: Uuid) -> MvResult<()>;
    async fn touch_credential(&self, id: Uuid) -> MvResult<()>;

    // --- Delegations ---
    async fn insert_delegation(&self, delegation: &Delegation) -> MvResult<()>;
    async fn get_delegation(&self, id: Uuid) -> MvResult<Option<Delegation>>;
    async fn list_delegations(&self, credential_id: Uuid) -> MvResult<Vec<Delegation>>;
    async fn revoke_delegation(&self, id: Uuid) -> MvResult<()>;
    async fn revoke_delegations_for_credential(&self, credential_id: Uuid) -> MvResult<()>;

    // --- Audit ---
    async fn append_audit_entry(&self, entry: &KeychainAuditEntry) -> MvResult<()>;
    async fn list_audit_entries(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<KeychainAuditEntry>>;
    async fn get_latest_audit_entry(&self) -> MvResult<Option<KeychainAuditEntry>>;
    async fn verify_audit_chain(&self) -> MvResult<bool>;

    // --- Breach Detection ---
    async fn record_access_pattern(&self, pattern: &AccessPattern) -> MvResult<()>;
    async fn get_access_patterns(
        &self,
        credential_id: Uuid,
        limit: usize,
    ) -> MvResult<Vec<AccessPattern>>;
    async fn insert_breach_alert(&self, alert: &BreachAlert) -> MvResult<()>;
    async fn list_breach_alerts(&self, limit: usize, offset: usize) -> MvResult<Vec<BreachAlert>>;
    async fn acknowledge_breach_alert(&self, id: Uuid) -> MvResult<()>;
    /// Check if a breach alert of the same type was already recorded for this
    /// credential within the last `within_secs` seconds (dedup window).
    async fn has_recent_breach_alert(
        &self,
        credential_id: Uuid,
        alert_type: &str,
        within_secs: u64,
    ) -> MvResult<bool>;

    // --- Tags ---
    async fn get_credential_tags(&self, credential_id: Uuid) -> MvResult<Vec<String>>;
    async fn save_credential_tags(&self, credential_id: Uuid, tags: &[String]) -> MvResult<()>;

    // --- Lockout State ---
    async fn set_lockout_state(&self, attempts: u32, locked_until: Option<String>) -> MvResult<()>;
    async fn get_lockout_state(&self) -> MvResult<(u32, Option<String>)>;

    // --- Domain ACLs ---
    async fn insert_acl(&self, acl: &DomainAcl) -> MvResult<()>;
    async fn get_acls_for_domain(&self, domain_id: Uuid) -> MvResult<Vec<DomainAcl>>;
    async fn get_acl_for_subject(&self, domain_id: Uuid, subject: &str) -> MvResult<Option<DomainAcl>>;
    async fn delete_acl(&self, id: Uuid) -> MvResult<()>;
}

fn _assert_keychain_store_object_safe(_: &dyn KeychainStore) {}

/// Storage backend for agentic intents, insights, and chronicle entries.
#[async_trait]
pub trait AgenticStore: Send + Sync {
    // --- Intents ---
    async fn log_intent(&self, intent: &CapturedIntent) -> MvResult<()>;
    async fn get_intent(&self, id: Uuid) -> MvResult<Option<CapturedIntent>>;
    async fn list_intents(
        &self,
        node_id: Option<Uuid>,
        status: Option<IntentStatus>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<CapturedIntent>>;
    async fn update_intent_status(&self, id: Uuid, status: IntentStatus) -> MvResult<bool>;

    // --- Insights ---
    async fn log_insight(&self, insight: &ProactiveInsight) -> MvResult<()>;
    async fn list_insights(&self, limit: usize, offset: usize) -> MvResult<Vec<ProactiveInsight>>;
    async fn delete_insight(&self, id: Uuid) -> MvResult<bool>;

    // --- Chronicle ---
    async fn log_chronicle(&self, entry: &ChronicleEntry) -> MvResult<()>;
    async fn list_chronicles(
        &self,
        node_id: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ChronicleEntry>>;
}

fn _assert_agentic_store_object_safe(_: &dyn AgenticStore) {}

/// Storage for exchange inbox proposals.
#[async_trait]
pub trait ExchangeStore: Send + Sync {
    async fn submit_proposal(&self, proposal: &Proposal) -> MvResult<()>;
    async fn get_proposal(&self, id: Uuid) -> MvResult<Option<Proposal>>;
    async fn list_proposals(
        &self,
        state: Option<ProposalState>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<Proposal>>;
    async fn resolve_proposal(&self, id: Uuid, state: ProposalState) -> MvResult<bool>;
    async fn count_proposals(&self, state: Option<ProposalState>) -> MvResult<usize>;
    async fn expire_proposals(&self, before: DateTime<Utc>) -> MvResult<usize>;
}

fn _assert_exchange_store_object_safe(_: &dyn ExchangeStore) {}

/// Storage for relay safeguards: blocked senders, auto-approve rules, undo snapshots.
#[async_trait]
pub trait SafeguardStore: Send + Sync {
    // Blocked senders
    async fn add_blocked_sender(&self, sender: &BlockedSender) -> MvResult<()>;
    async fn remove_blocked_sender(&self, id: Uuid) -> MvResult<bool>;
    async fn list_blocked_senders(&self) -> MvResult<Vec<BlockedSender>>;
    async fn is_sender_blocked(&self, sender_type: &str, sender_name: &str) -> MvResult<bool>;

    // Auto-approve rules
    async fn add_auto_approve_rule(&self, rule: &AutoApproveRule) -> MvResult<()>;
    async fn remove_auto_approve_rule(&self, id: Uuid) -> MvResult<bool>;
    async fn list_auto_approve_rules(&self) -> MvResult<Vec<AutoApproveRule>>;
    async fn update_auto_approve_rule(&self, rule: &AutoApproveRule) -> MvResult<bool>;

    // Undo snapshots
    async fn save_undo_snapshot(&self, snapshot: &UndoSnapshot) -> MvResult<()>;
    async fn get_undo_snapshot(&self, proposal_id: Uuid) -> MvResult<Option<UndoSnapshot>>;
    async fn mark_undo_used(&self, id: Uuid) -> MvResult<bool>;
    async fn cleanup_expired_snapshots(&self) -> MvResult<usize>;
}

fn _assert_safeguard_store_object_safe(_: &dyn SafeguardStore) {}

/// Storage for agent feedback and confidence overrides (reflection / feedback loop).
#[async_trait]
pub trait FeedbackStore: Send + Sync {
    async fn record_feedback(&self, fb: &AgentFeedback) -> MvResult<()>;
    async fn list_feedback(
        &self,
        intent_type: Option<&str>,
        limit: usize,
    ) -> MvResult<Vec<AgentFeedback>>;
    async fn get_acceptance_rate(&self, intent_type: &str) -> MvResult<(usize, usize)>;
    async fn set_confidence_override(&self, override_: &ConfidenceOverride) -> MvResult<()>;
    async fn get_confidence_override(
        &self,
        intent_type: &str,
    ) -> MvResult<Option<ConfidenceOverride>>;
    async fn list_confidence_overrides(&self) -> MvResult<Vec<ConfidenceOverride>>;
}

fn _assert_feedback_store_object_safe(_: &dyn FeedbackStore) {}

/// Storage for autonomy rules and action logs (Phase 3.1 — Autonomy & Precision Controls).
#[async_trait]
pub trait AutonomyStore: Send + Sync {
    async fn add_autonomy_rule(&self, rule: &AutonomyRule) -> MvResult<()>;
    async fn get_autonomy_rule(&self, id: Uuid) -> MvResult<Option<AutonomyRule>>;
    async fn list_autonomy_rules(&self) -> MvResult<Vec<AutonomyRule>>;
    async fn update_autonomy_rule(&self, rule: &AutonomyRule) -> MvResult<bool>;
    async fn delete_autonomy_rule(&self, id: Uuid) -> MvResult<bool>;
    async fn log_autonomy_action(&self, log: &AutonomyActionLog) -> MvResult<()>;
    async fn count_recent_actions(
        &self,
        rule_id: Option<Uuid>,
        since: DateTime<Utc>,
    ) -> MvResult<usize>;
    async fn list_autonomy_action_log(&self, limit: usize) -> MvResult<Vec<AutonomyActionLog>>;
}

fn _assert_autonomy_store_object_safe(_: &dyn AutonomyStore) {}

/// Storage for communication relay: contacts, channels, and messages.
#[async_trait]
pub trait RelayStore: Send + Sync {
    // Contacts
    async fn add_relay_contact(&self, contact: &RelayContact) -> MvResult<()>;
    async fn get_relay_contact(&self, id: Uuid) -> MvResult<Option<RelayContact>>;
    async fn list_relay_contacts(&self) -> MvResult<Vec<RelayContact>>;
    async fn update_relay_contact(&self, contact: &RelayContact) -> MvResult<bool>;
    async fn delete_relay_contact(&self, id: Uuid) -> MvResult<bool>;

    // Channels
    async fn add_relay_channel(&self, channel: &RelayChannel) -> MvResult<()>;
    async fn get_relay_channel(&self, id: Uuid) -> MvResult<Option<RelayChannel>>;
    async fn list_relay_channels(&self) -> MvResult<Vec<RelayChannel>>;
    async fn delete_relay_channel(&self, id: Uuid) -> MvResult<bool>;

    // Messages
    async fn add_relay_message(&self, message: &RelayMessage) -> MvResult<()>;
    async fn get_relay_message(&self, id: Uuid) -> MvResult<Option<RelayMessage>>;
    async fn list_relay_messages(
        &self,
        channel_id: Uuid,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<RelayMessage>>;
    async fn update_message_status(&self, id: Uuid, status: MessageStatus) -> MvResult<bool>;
    async fn list_thread_messages(
        &self,
        thread_id: Uuid,
        limit: usize,
    ) -> MvResult<Vec<RelayMessage>>;
    async fn count_unread_messages(&self, channel_id: Option<Uuid>) -> MvResult<usize>;
}

fn _assert_relay_store_object_safe(_: &dyn RelayStore) {}

// Legacy aliases for backward compatibility with proactive.rs
pub trait InsightStore: AgenticStore {}
impl<T: AgenticStore> InsightStore for T {}

/// Storage for knowledge conflict alerts.
#[async_trait]
pub trait ConflictStore: Send + Sync {
    async fn insert_conflict(&self, alert: &ConflictAlert) -> MvResult<()>;
    async fn get_conflict(&self, id: Uuid) -> MvResult<Option<ConflictAlert>>;
    async fn list_conflicts(
        &self,
        resolved: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ConflictAlert>>;
    async fn resolve_conflict(&self, id: Uuid) -> MvResult<bool>;
}

fn _assert_conflict_store_object_safe(_: &dyn ConflictStore) {}

/// Storage for contact identities and trust models.
#[async_trait]
pub trait ContactIdentityStore: Send + Sync {
    // Identities
    async fn add_contact_identity(&self, identity: &ContactIdentity) -> MvResult<()>;
    async fn list_contact_identities(&self, contact_id: Uuid) -> MvResult<Vec<ContactIdentity>>;
    async fn delete_contact_identity(&self, id: Uuid) -> MvResult<bool>;
    async fn verify_contact_identity(&self, id: Uuid) -> MvResult<bool>;

    // Trust models
    async fn get_trust_model(&self, contact_id: Uuid) -> MvResult<Option<TrustModel>>;
    async fn set_trust_model(&self, model: &TrustModel) -> MvResult<()>;
}

fn _assert_contact_identity_store_object_safe(_: &dyn ContactIdentityStore) {}

/// Storage for owner profile.
#[async_trait]
pub trait ProfileStore: Send + Sync {
	async fn get_profile(&self) -> MvResult<OwnerProfile>;
	async fn update_profile(&self, req: &UpdateProfileRequest) -> MvResult<OwnerProfile>;
}

fn _assert_profile_store_object_safe(_: &dyn ProfileStore) {}

/// Storage for consumer profiles (AI consumer identities).
#[async_trait]
pub trait ConsumerStore: Send + Sync {
    async fn create_consumer(&self, profile: &ConsumerProfile) -> MvResult<()>;
    async fn get_consumer(&self, id: Uuid) -> MvResult<Option<ConsumerProfile>>;
    async fn get_consumer_by_name(&self, name: &str) -> MvResult<Option<ConsumerProfile>>;
    async fn get_consumer_by_token_hash(&self, token_hash: &str) -> MvResult<Option<ConsumerProfile>>;
    async fn list_consumers(&self) -> MvResult<Vec<ConsumerProfile>>;
    async fn revoke_consumer(&self, id: Uuid) -> MvResult<bool>;
    async fn touch_consumer(&self, id: Uuid) -> MvResult<()>;
}

fn _assert_consumer_store_object_safe(_: &dyn ConsumerStore) {}

/// Storage for access policies (ABAC with default-deny).
#[async_trait]
pub trait PolicyStore: Send + Sync {
    async fn set_policy(&self, policy: &AccessPolicy) -> MvResult<()>;
    async fn get_policy(&self, id: Uuid) -> MvResult<Option<AccessPolicy>>;
    async fn get_policy_for(&self, secret_key: &str, consumer: &str) -> MvResult<Option<AccessPolicy>>;
    async fn list_policies(
        &self,
        secret_key: Option<&str>,
        consumer: Option<&str>,
    ) -> MvResult<Vec<AccessPolicy>>;
    async fn delete_policy(&self, id: Uuid) -> MvResult<bool>;
}

fn _assert_policy_store_object_safe(_: &dyn PolicyStore) {}

/// Storage for public share links.
#[async_trait]
pub trait ShareStore: Send + Sync {
    async fn insert_public_share(&self, share: &PublicShare) -> MvResult<()>;
    async fn get_public_share(&self, id: Uuid) -> MvResult<Option<PublicShare>>;
    async fn get_public_share_by_hash(&self, token_hash: &str) -> MvResult<Option<PublicShare>>;
    async fn list_public_shares(
        &self,
        node_id: Option<Uuid>,
        include_revoked: bool,
    ) -> MvResult<Vec<PublicShare>>;
    async fn revoke_public_share(&self, id: Uuid, revoked_at: DateTime<Utc>) -> MvResult<bool>;
}

fn _assert_share_store_object_safe(_: &dyn ShareStore) {}

/// Storage for node comments and annotations.
#[async_trait]
pub trait CommentStore: Send + Sync {
    async fn insert_comment(&self, comment: &NodeComment) -> MvResult<()>;
    async fn get_comment(&self, id: Uuid) -> MvResult<Option<NodeComment>>;
    async fn list_comments(
        &self,
        node_id: Uuid,
        include_resolved: bool,
    ) -> MvResult<Vec<NodeComment>>;
    async fn resolve_comment(&self, id: Uuid, resolved_at: DateTime<Utc>) -> MvResult<bool>;
    async fn delete_comment(&self, id: Uuid) -> MvResult<bool>;
}

fn _assert_comment_store_object_safe(_: &dyn CommentStore) {}

/// Registry for MCP connectors (marketplace catalog).
#[async_trait]
pub trait McpConnectorStore: Send + Sync {
    async fn insert_mcp_connector(&self, connector: &McpConnector) -> MvResult<()>;
    async fn get_mcp_connector(&self, id: Uuid) -> MvResult<Option<McpConnector>>;
    async fn list_mcp_connectors(
        &self,
        publisher: Option<&str>,
        verified: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<McpConnector>>;
    async fn update_mcp_connector(&self, connector: &McpConnector) -> MvResult<bool>;
    async fn delete_mcp_connector(&self, id: Uuid) -> MvResult<bool>;
}

fn _assert_mcp_connector_store_object_safe(_: &dyn McpConnectorStore) {}

/// Storage for proxy audit log entries.
#[async_trait]
pub trait ProxyAuditStore: Send + Sync {
    async fn log_proxy_audit(&self, entry: &ProxyAuditEntry) -> MvResult<()>;
    async fn update_proxy_audit(
        &self,
        id: Uuid,
        success: bool,
        sanitized: bool,
        error: Option<&str>,
        response_status: Option<i32>,
    ) -> MvResult<()>;
    async fn list_proxy_audit(
        &self,
        consumer: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ProxyAuditEntry>>;
}

fn _assert_proxy_audit_store_object_safe(_: &dyn ProxyAuditStore) {}

/// Storage for HITL approval queue entries.
#[async_trait]
pub trait ApprovalStore: Send + Sync {
    async fn create_approval(&self, request: &ApprovalRequest) -> MvResult<()>;
    async fn get_approval(&self, id: Uuid) -> MvResult<Option<ApprovalRequest>>;
    async fn list_pending_approvals(&self, consumer: Option<&str>) -> MvResult<Vec<ApprovalRequest>>;
    async fn decide_approval(
        &self,
        id: Uuid,
        approved: bool,
        decided_by: Option<&str>,
        deny_reason: Option<&str>,
    ) -> MvResult<bool>;
    async fn expire_approvals(&self) -> MvResult<usize>;
    /// Check if there's an approved (non-expired) approval for a consumer+secret pair.
    async fn find_active_approval(
        &self,
        consumer: &str,
        secret_key: &str,
    ) -> MvResult<Option<ApprovalRequest>>;
}

fn _assert_approval_store_object_safe(_: &dyn ApprovalStore) {}

/// Cross-encoder reranker for improving search result ordering.
///
/// Takes a query and a set of candidate documents, returns relevance scores.
/// Implementations may use ONNX models, LLM-based scoring, or heuristics.
#[async_trait]
pub trait Reranker: Send + Sync {
    /// Score each (query, document) pair. Returns scores in the same order as documents.
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> MvResult<Vec<f64>>;

    /// Name of the reranker for logging/diagnostics.
    fn name(&self) -> &str;

    /// Whether the reranker is ready (model loaded, etc.).
    fn is_ready(&self) -> bool;
}

fn _assert_reranker_object_safe(_: &dyn Reranker) {}

/// Session memory store for conversation context in follow-up queries.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Record a query+result turn in the session.
    async fn add_turn(
        &self,
        session_id: &str,
        query: &str,
        result_summary: &str,
    ) -> MvResult<()>;

    /// Get recent turns for a session.
    async fn get_turns(
        &self,
        session_id: &str,
        limit: usize,
    ) -> MvResult<Vec<(String, String)>>;

    /// Clear a session.
    async fn clear_session(&self, session_id: &str) -> MvResult<()>;

    /// Expire sessions older than the given duration.
    async fn expire_sessions(&self, max_age_secs: u64) -> MvResult<usize>;
}

fn _assert_session_store_object_safe(_: &dyn SessionStore) {}

/// Conversation store for persistent multi-turn dialogues.
#[async_trait]
pub trait ConversationStore: Send + Sync {
    async fn create_conversation(
        &self,
        id: Uuid,
        title: Option<&str>,
    ) -> MvResult<()>;

    async fn add_message(
        &self,
        conversation_id: Uuid,
        role: &str,
        content: &str,
    ) -> MvResult<Uuid>;

    async fn get_messages(
        &self,
        conversation_id: Uuid,
        limit: usize,
    ) -> MvResult<Vec<(Uuid, String, String, DateTime<Utc>)>>;

    async fn delete_conversation(&self, id: Uuid) -> MvResult<bool>;

    async fn list_conversations(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<(Uuid, Option<String>, DateTime<Utc>)>>;

    async fn expire_conversations(&self, max_age_secs: u64) -> MvResult<usize>;
}

fn _assert_conversation_store_object_safe(_: &dyn ConversationStore) {}

/// Adapter poll state for cursor persistence across adapter polling cycles.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AdapterPollState {
    pub adapter_name: String,
    pub cursor: String,
    pub last_poll_at: String,
    pub messages_received: u64,
}

/// Storage for adapter poll state (cursor persistence).
#[async_trait]
pub trait AdapterPollStore: Send + Sync {
    async fn get_poll_state(&self, adapter_name: &str) -> MvResult<Option<AdapterPollState>>;
    async fn upsert_poll_state(
        &self,
        adapter_name: &str,
        cursor: &str,
        messages_received: u64,
    ) -> MvResult<()>;
    async fn list_poll_states(&self) -> MvResult<Vec<AdapterPollState>>;
    async fn delete_poll_state(&self, adapter_name: &str) -> MvResult<bool>;
}

fn _assert_adapter_poll_store_object_safe(_: &dyn AdapterPollStore) {}

#[cfg(test)]
mod tests {
    use super::*;

    // Ensure traits are object-safe
    fn _assert_node_store_object_safe(_: &dyn NodeStore) {}
    fn _assert_vector_store_object_safe(_: &dyn VectorStore) {}
    fn _assert_full_text_index_object_safe(_: &dyn FullTextIndex) {}
    fn _assert_graph_store_object_safe(_: &dyn GraphStore) {}
    fn _assert_embedder_object_safe(_: &dyn Embedder) {}
}
