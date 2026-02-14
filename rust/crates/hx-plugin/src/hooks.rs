//! Hook system for plugin integration points.

use async_trait::async_trait;
use hx_core::{KnowledgeNode, MvResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::manifest::PluginManifest;

/// Points in the Helix lifecycle where plugins can hook in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookPoint {
    /// Before a node is ingested into the vault
    PreIngest,
    /// After a node is ingested
    PostIngest,
    /// Before a search query is executed
    PreSearch,
    /// After search results are returned
    PostSearch,
    /// When a node is changed (created/updated/deleted)
    OnChange,
    /// On a scheduled interval
    Scheduled,
    /// When an intent is detected by the watcher
    OnIntent,
}

impl std::fmt::Display for HookPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PreIngest => write!(f, "pre_ingest"),
            Self::PostIngest => write!(f, "post_ingest"),
            Self::PreSearch => write!(f, "pre_search"),
            Self::PostSearch => write!(f, "post_search"),
            Self::OnChange => write!(f, "on_change"),
            Self::Scheduled => write!(f, "scheduled"),
            Self::OnIntent => write!(f, "on_intent"),
        }
    }
}

/// Context passed to a plugin hook execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookContext {
    pub hook_point: HookPoint,
    pub node: Option<KnowledgeNode>,
    pub query: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl HookContext {
    pub fn new(hook_point: HookPoint) -> Self {
        Self {
            hook_point,
            node: None,
            query: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_node(mut self, node: KnowledgeNode) -> Self {
        self.node = Some(node);
        self
    }

    pub fn with_query(mut self, query: String) -> Self {
        self.query = Some(query);
        self
    }
}

/// Result from a hook execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookResult {
    pub success: bool,
    pub modified_node: Option<KnowledgeNode>,
    pub modified_query: Option<String>,
    pub output: Option<String>,
    pub error: Option<String>,
}

impl HookResult {
    pub fn ok() -> Self {
        Self {
            success: true,
            modified_node: None,
            modified_query: None,
            output: None,
            error: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            success: false,
            modified_node: None,
            modified_query: None,
            output: None,
            error: Some(msg.into()),
        }
    }

    pub fn with_node(mut self, node: KnowledgeNode) -> Self {
        self.modified_node = Some(node);
        self
    }
}

/// Trait for plugin implementations.
#[async_trait]
pub trait Plugin: Send + Sync {
    fn manifest(&self) -> &PluginManifest;
    async fn execute_hook(&self, context: &HookContext) -> MvResult<HookResult>;
}
