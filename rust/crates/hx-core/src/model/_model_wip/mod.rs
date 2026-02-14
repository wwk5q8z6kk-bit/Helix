pub mod chronicle;
pub mod intent;
pub mod keychain;
pub mod proactive;
pub use chronicle::*;
pub use intent::*;
pub use keychain::*;
pub use proactive::*;

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
    pub fn new(kind: NodeKind, content: String) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            kind,
            title: None,
            content,
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
}

impl MemoryQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            strategy: SearchStrategy::Hybrid,
            filters: QueryFilters::default(),
            limit: 10,
            min_score: 0.0,
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
