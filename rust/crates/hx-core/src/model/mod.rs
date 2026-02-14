pub mod consumer;
pub use consumer::*;
pub mod exchange;
pub use exchange::*;
pub mod keychain;
pub use keychain::*;
pub mod policy;
pub use policy::*;
pub mod proxy;
pub use proxy::*;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Knowledge Node
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeNode {
    pub id: Uuid,
    pub kind: NodeKind,
    pub title: Option<String>,
    pub content: String,
    pub source: Option<String>,
    pub namespace: String,
    pub tags: Vec<String>,
    pub importance: f64,
    pub temporal: TemporalMeta,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl KnowledgeNode {
    pub fn new(kind: NodeKind, content: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            kind,
            title: None,
            content: content.into(),
            source: None,
            namespace: "default".into(),
            tags: Vec::new(),
            importance: 0.5,
            temporal: TemporalMeta {
                created_at: now,
                updated_at: now,
                last_accessed_at: now,
                access_count: 0,
                version: 1,
                expires_at: None,
            },
            metadata: HashMap::new(),
        }
    }

    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = ns.into();
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Node Kind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Fact,
    Task,
    Event,
    Decision,
    Preference,
    Entity,
    CodeSnippet,
    Project,
    Conversation,
    Procedure,
    Observation,
    Bookmark,
    Template,
    SavedView,
    Proposal,
}

impl NodeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fact => "fact",
            Self::Task => "task",
            Self::Event => "event",
            Self::Decision => "decision",
            Self::Preference => "preference",
            Self::Entity => "entity",
            Self::CodeSnippet => "code_snippet",
            Self::Project => "project",
            Self::Conversation => "conversation",
            Self::Procedure => "procedure",
            Self::Observation => "observation",
            Self::Bookmark => "bookmark",
            Self::Template => "template",
            Self::SavedView => "saved_view",
            Self::Proposal => "proposal",
        }
    }
}

impl std::str::FromStr for NodeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fact" => Ok(Self::Fact),
            "task" => Ok(Self::Task),
            "event" => Ok(Self::Event),
            "decision" => Ok(Self::Decision),
            "preference" => Ok(Self::Preference),
            "entity" => Ok(Self::Entity),
            "code_snippet" => Ok(Self::CodeSnippet),
            "project" => Ok(Self::Project),
            "conversation" => Ok(Self::Conversation),
            "procedure" => Ok(Self::Procedure),
            "observation" => Ok(Self::Observation),
            "bookmark" => Ok(Self::Bookmark),
            "template" => Ok(Self::Template),
            "saved_view" => Ok(Self::SavedView),
            "proposal" => Ok(Self::Proposal),
            _ => Err(format!("unknown node kind: {s}")),
        }
    }
}

impl std::fmt::Display for NodeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Permission Templates
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionTier {
    View,
    Edit,
    Action,
    Admin,
}

impl PermissionTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::View => "view",
            Self::Edit => "edit",
            Self::Action => "action",
            Self::Admin => "admin",
        }
    }
}

impl std::str::FromStr for PermissionTier {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "view" => Ok(Self::View),
            "edit" => Ok(Self::Edit),
            "action" => Ok(Self::Action),
            "admin" => Ok(Self::Admin),
            _ => Err(format!("unknown permission tier: {value}")),
        }
    }
}

impl std::fmt::Display for PermissionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionTemplate {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub tier: PermissionTier,
    pub scope_namespace: Option<String>,
    pub scope_tags: Vec<String>,
    pub allow_kinds: Vec<NodeKind>,
    pub allow_actions: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessKey {
    pub id: Uuid,
    pub name: Option<String>,
    pub template_id: Uuid,
    pub key_hash: String,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicShare {
    pub id: Uuid,
    pub node_id: Uuid,
    pub token_hash: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

impl PublicShare {
    pub fn is_active(&self) -> bool {
        if self.revoked_at.is_some() {
            return false;
        }
        if let Some(expires_at) = self.expires_at {
            return Utc::now() < expires_at;
        }
        true
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeComment {
    pub id: Uuid,
    pub node_id: Uuid,
    pub author: Option<String>,
    pub body: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

impl NodeComment {
    pub fn is_resolved(&self) -> bool {
        self.resolved_at.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConnector {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub publisher: Option<String>,
    pub version: String,
    pub homepage_url: Option<String>,
    pub repository_url: Option<String>,
    pub config_schema: serde_json::Value,
    pub capabilities: Vec<String>,
    pub verified: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Temporal Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMeta {
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub version: u32,
    pub expires_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Relationships
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub id: Uuid,
    pub from_node: Uuid,
    pub to_node: Uuid,
    pub kind: RelationKind,
    pub weight: f64,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

impl Relationship {
    pub fn new(from_node: Uuid, to_node: Uuid, kind: RelationKind) -> Self {
        Self {
            id: Uuid::now_v7(),
            from_node,
            to_node,
            kind,
            weight: 1.0,
            metadata: HashMap::new(),
            created_at: Utc::now(),
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationKind {
    RelatesTo,
    DependsOn,
    DerivedFrom,
    Supersedes,
    Contradicts,
    PartOf,
    Contains,
    References,
    SimilarTo,
    FollowsFrom,
}

impl RelationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::RelatesTo => "relates_to",
            Self::DependsOn => "depends_on",
            Self::DerivedFrom => "derived_from",
            Self::Supersedes => "supersedes",
            Self::Contradicts => "contradicts",
            Self::PartOf => "part_of",
            Self::Contains => "contains",
            Self::References => "references",
            Self::SimilarTo => "similar_to",
            Self::FollowsFrom => "follows_from",
        }
    }
}

impl std::str::FromStr for RelationKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "relates_to" => Ok(Self::RelatesTo),
            "depends_on" => Ok(Self::DependsOn),
            "derived_from" => Ok(Self::DerivedFrom),
            "supersedes" => Ok(Self::Supersedes),
            "contradicts" => Ok(Self::Contradicts),
            "part_of" => Ok(Self::PartOf),
            "contains" => Ok(Self::Contains),
            "references" => Ok(Self::References),
            "similar_to" => Ok(Self::SimilarTo),
            "follows_from" => Ok(Self::FollowsFrom),
            _ => Err(format!("unknown relation kind: {s}")),
        }
    }
}

impl std::fmt::Display for RelationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Query Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryQuery {
    pub text: String,
    pub strategy: SearchStrategy,
    pub filters: QueryFilters,
    pub limit: usize,
    pub min_score: f64,
    /// Optional query rewrite strategy applied before search.
    #[serde(default)]
    pub rewrite_strategy: Option<RewriteStrategy>,
    /// Optional session ID for conversation-aware retrieval.
    #[serde(default)]
    pub session_id: Option<String>,
}

/// Strategy for rewriting queries before search execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RewriteStrategy {
    /// No rewriting — use query as-is.
    None,
    /// Expand abbreviations and add synonyms.
    Expand,
    /// Decompose compound queries into sub-queries, search each, merge results.
    Decompose,
    /// Hypothetical Document Embedding — generate a hypothetical answer and embed that.
    HyDE,
    /// Apply all available rewriting strategies and merge results.
    Auto,
}

impl std::str::FromStr for RewriteStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "none" => Ok(Self::None),
            "expand" => Ok(Self::Expand),
            "decompose" => Ok(Self::Decompose),
            "hyde" => Ok(Self::HyDE),
            "auto" => Ok(Self::Auto),
            _ => Err(format!("unknown rewrite strategy: {s}")),
        }
    }
}

impl std::fmt::Display for RewriteStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::Expand => f.write_str("expand"),
            Self::Decompose => f.write_str("decompose"),
            Self::HyDE => f.write_str("hyde"),
            Self::Auto => f.write_str("auto"),
        }
    }
}

impl MemoryQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            strategy: SearchStrategy::Hybrid,
            filters: QueryFilters::default(),
            limit: 10,
            min_score: 0.0,
            rewrite_strategy: None,
            session_id: None,
        }
    }

    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    pub fn with_namespace(mut self, ns: impl Into<String>) -> Self {
        self.filters.namespace = Some(ns.into());
        self
    }

    pub fn with_kinds(mut self, kinds: Vec<NodeKind>) -> Self {
        self.filters.kinds = Some(kinds);
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.filters.tags = Some(tags);
        self
    }

    pub fn with_rewrite_strategy(mut self, strategy: RewriteStrategy) -> Self {
        self.rewrite_strategy = Some(strategy);
        self
    }

    pub fn with_session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchStrategy {
    Vector,
    FullText,
    Hybrid,
    Graph,
}

impl std::str::FromStr for SearchStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vector" => Ok(Self::Vector),
            "fulltext" | "full_text" => Ok(Self::FullText),
            "hybrid" => Ok(Self::Hybrid),
            "graph" => Ok(Self::Graph),
            _ => Err(format!("unknown search strategy: {s}")),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryFilters {
    pub namespace: Option<String>,
    pub kinds: Option<Vec<NodeKind>>,
    pub tags: Option<Vec<String>>,
    pub min_importance: Option<f64>,
    pub created_after: Option<DateTime<Utc>>,
    pub created_before: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Search Results
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub node: KnowledgeNode,
    pub score: f64,
    pub match_source: MatchSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MatchSource {
    Vector,
    FullText,
    Hybrid,
    Graph,
}

// ---------------------------------------------------------------------------
// Changelog
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    pub id: i64,
    pub node_id: Uuid,
    pub operation: ChangeOp,
    pub diff: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeOp {
    Create,
    Update,
    Delete,
}

impl ChangeOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Create => "create",
            Self::Update => "update",
            Self::Delete => "delete",
        }
    }
}

// ---------------------------------------------------------------------------
// Captured Intents (Agentic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedIntent {
    pub id: Uuid,
    pub node_id: Uuid,
    pub intent_type: IntentType,
    pub confidence: f32,
    pub parameters: serde_json::Value,
    pub status: IntentStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IntentType {
    ScheduleReminder,
    ExtractTask,
    LinkToProject,
    SuggestTag,
    SuggestLink,
    Custom(String),
}

impl IntentType {
    pub fn as_str(&self) -> &str {
        match self {
            Self::ScheduleReminder => "schedule_reminder",
            Self::ExtractTask => "extract_task",
            Self::LinkToProject => "link_to_project",
            Self::SuggestTag => "suggest_tag",
            Self::SuggestLink => "suggest_link",
            Self::Custom(s) => s,
        }
    }
}

impl std::str::FromStr for IntentType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "schedule_reminder" => Ok(Self::ScheduleReminder),
            "extract_task" => Ok(Self::ExtractTask),
            "link_to_project" => Ok(Self::LinkToProject),
            "suggest_tag" => Ok(Self::SuggestTag),
            "suggest_link" => Ok(Self::SuggestLink),
            other => Ok(Self::Custom(other.to_string())),
        }
    }
}

impl std::fmt::Display for IntentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum IntentStatus {
    Suggested,
    Applied,
    Dismissed,
}

impl IntentStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Suggested => "suggested",
            Self::Applied => "applied",
            Self::Dismissed => "dismissed",
        }
    }
}

impl std::str::FromStr for IntentStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "suggested" => Ok(Self::Suggested),
            "applied" => Ok(Self::Applied),
            "dismissed" => Ok(Self::Dismissed),
            _ => Err(format!("unknown intent status: {s}")),
        }
    }
}

impl Default for IntentStatus {
    fn default() -> Self {
        Self::Suggested
    }
}

impl CapturedIntent {
    pub fn new(node_id: Uuid, intent_type: IntentType) -> Self {
        Self {
            id: Uuid::now_v7(),
            node_id,
            intent_type,
            confidence: 0.0,
            parameters: serde_json::Value::Null,
            status: IntentStatus::Suggested,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_parameters(mut self, params: serde_json::Value) -> Self {
        self.parameters = params;
        self
    }
}

// ---------------------------------------------------------------------------
// Proactive Insights (Agentic)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveInsight {
    pub id: Uuid,
    pub title: String,
    pub content: String,
    pub insight_type: InsightType,
    pub related_node_ids: Vec<Uuid>,
    pub importance: f32,
    pub created_at: DateTime<Utc>,
    pub dismissed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InsightType {
    Connection,
    Trend,
    Gap,
    Stale,
    Reminder,
    Cluster,
    General,
    TemporalPattern,
    KnowledgeGap,
    CrossDomain,
    UnlinkedCluster,
    AmbientLink,
    Conflict,
}

impl InsightType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Connection => "connection",
            Self::Trend => "trend",
            Self::Gap => "gap",
            Self::Stale => "stale",
            Self::Reminder => "reminder",
            Self::Cluster => "cluster",
            Self::General => "general",
            Self::TemporalPattern => "temporal_pattern",
            Self::KnowledgeGap => "knowledge_gap",
            Self::CrossDomain => "cross_domain",
            Self::UnlinkedCluster => "unlinked_cluster",
            Self::AmbientLink => "ambient_link",
            Self::Conflict => "conflict",
        }
    }
}

impl std::str::FromStr for InsightType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "connection" => Ok(Self::Connection),
            "trend" => Ok(Self::Trend),
            "gap" => Ok(Self::Gap),
            "stale" => Ok(Self::Stale),
            "reminder" => Ok(Self::Reminder),
            "cluster" => Ok(Self::Cluster),
            "general" => Ok(Self::General),
            "temporal_pattern" => Ok(Self::TemporalPattern),
            "knowledge_gap" => Ok(Self::KnowledgeGap),
            "cross_domain" => Ok(Self::CrossDomain),
            "unlinked_cluster" => Ok(Self::UnlinkedCluster),
            "ambient_link" => Ok(Self::AmbientLink),
            "conflict" => Ok(Self::Conflict),
            _ => Err(format!("unknown insight type: {s}")),
        }
    }
}

impl std::fmt::Display for InsightType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ProactiveInsight {
    pub fn new(
        title: impl Into<String>,
        content: impl Into<String>,
        insight_type: InsightType,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            title: title.into(),
            content: content.into(),
            insight_type,
            related_node_ids: Vec::new(),
            importance: 0.5,
            created_at: Utc::now(),
            dismissed_at: None,
        }
    }

    pub fn with_related_nodes(mut self, ids: Vec<Uuid>) -> Self {
        self.related_node_ids = ids;
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }
}

// ---------------------------------------------------------------------------
// Chronicle Entries (Transparency Logging)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronicleEntry {
    pub id: Uuid,
    pub node_id: Option<Uuid>,
    pub step_name: String,
    pub logic: String,
    pub input_snapshot: Option<String>,
    pub output_snapshot: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl ChronicleEntry {
    pub fn new(step_name: impl Into<String>, logic: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            node_id: None,
            step_name: step_name.into(),
            logic: logic.into(),
            input_snapshot: None,
            output_snapshot: None,
            timestamp: Utc::now(),
        }
    }

    pub fn with_node(mut self, node_id: Uuid) -> Self {
        self.node_id = Some(node_id);
        self
    }

    pub fn with_snapshots(mut self, input: Option<String>, output: Option<String>) -> Self {
        self.input_snapshot = input;
        self.output_snapshot = output;
        self
    }
}

// ---------------------------------------------------------------------------
// Change Notification (shared between engine and server)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeNotification {
    pub node_id: String,
    pub operation: String,
    pub timestamp: String,
    pub namespace: Option<String>,
}

// ---------------------------------------------------------------------------
// Knowledge Conflict Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictAlert {
    pub id: Uuid,
    pub node_a: Uuid,
    pub node_b: Uuid,
    pub conflict_type: ConflictType,
    pub score: f64,
    pub explanation: String,
    pub resolved: bool,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConflictType {
    Contradiction,
    Supersession,
    Ambiguity,
}

impl ConflictType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Contradiction => "contradiction",
            Self::Supersession => "supersession",
            Self::Ambiguity => "ambiguity",
        }
    }
}

impl std::str::FromStr for ConflictType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "contradiction" => Ok(Self::Contradiction),
            "supersession" => Ok(Self::Supersession),
            "ambiguity" => Ok(Self::Ambiguity),
            _ => Err(format!("unknown conflict type: {s}")),
        }
    }
}

impl std::fmt::Display for ConflictType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl ConflictAlert {
    pub fn new(node_a: Uuid, node_b: Uuid, conflict_type: ConflictType, score: f64, explanation: String) -> Self {
        Self {
            id: Uuid::now_v7(),
            node_a,
            node_b,
            conflict_type,
            score,
            explanation,
            resolved: false,
            created_at: Utc::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Contact Identity & Trust
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactIdentity {
    pub id: Uuid,
    pub contact_id: Uuid,
    pub identity_type: IdentityType,
    pub identity_value: String,
    pub verified: bool,
    pub verified_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdentityType {
    Email,
    PublicKey,
    OAuth,
    Phone,
}

impl IdentityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Email => "email",
            Self::PublicKey => "public_key",
            Self::OAuth => "oauth",
            Self::Phone => "phone",
        }
    }
}

impl std::str::FromStr for IdentityType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "email" => Ok(Self::Email),
            "public_key" => Ok(Self::PublicKey),
            "oauth" => Ok(Self::OAuth),
            "phone" => Ok(Self::Phone),
            _ => Err(format!("unknown identity type: {s}")),
        }
    }
}

impl std::fmt::Display for IdentityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustModel {
    pub contact_id: Uuid,
    pub can_query: bool,
    pub can_inject_context: bool,
    pub can_auto_reply: bool,
    pub allowed_namespaces: Vec<String>,
    pub max_confidence_override: Option<f64>,
    pub updated_at: DateTime<Utc>,
}

impl Default for TrustModel {
    fn default() -> Self {
        Self {
            contact_id: Uuid::nil(),
            can_query: false,
            can_inject_context: false,
            can_auto_reply: false,
            allowed_namespaces: Vec::new(),
            max_confidence_override: None,
            updated_at: Utc::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// Owner Profile
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OwnerProfile {
	pub display_name: String,
	pub avatar_url: Option<String>,
	pub bio: Option<String>,
	pub email: Option<String>,
	pub preferred_namespace: String,
	pub default_node_kind: String,
	pub preferred_llm_provider: Option<String>,
	pub timezone: String,
	pub signature_name: Option<String>,
	pub signature_public_key: Option<String>,
	pub metadata: HashMap<String, serde_json::Value>,
	pub created_at: DateTime<Utc>,
	pub updated_at: DateTime<Utc>,
}

impl Default for OwnerProfile {
	fn default() -> Self {
		let now = Utc::now();
		Self {
			display_name: String::new(),
			avatar_url: None,
			bio: None,
			email: None,
			preferred_namespace: "default".into(),
			default_node_kind: "fact".into(),
			preferred_llm_provider: None,
			timezone: "UTC".into(),
			signature_name: None,
			signature_public_key: None,
			metadata: HashMap::new(),
			created_at: now,
			updated_at: now,
		}
	}
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UpdateProfileRequest {
	pub display_name: Option<String>,
	pub avatar_url: Option<String>,
	pub bio: Option<String>,
	pub email: Option<String>,
	pub preferred_namespace: Option<String>,
	pub default_node_kind: Option<String>,
	pub preferred_llm_provider: Option<String>,
	pub timezone: Option<String>,
	pub signature_name: Option<String>,
	pub signature_public_key: Option<String>,
	pub metadata: Option<HashMap<String, serde_json::Value>>,
}

#[cfg(test)]
mod tests {
    use super::NodeKind;

    #[test]
    fn node_kind_parses_new_task_and_event_variants() {
        assert!(matches!("task".parse::<NodeKind>(), Ok(NodeKind::Task)));
        assert!(matches!("event".parse::<NodeKind>(), Ok(NodeKind::Event)));
        assert_eq!(NodeKind::Task.to_string(), "task");
        assert_eq!(NodeKind::Event.to_string(), "event");
    }
}
