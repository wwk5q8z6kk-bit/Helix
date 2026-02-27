//! Sync snapshot types for import/export.

use super::clock::VectorClock;
use chrono::{DateTime, Utc};
use hx_core::KnowledgeNode;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An exported snapshot of vault data for sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSnapshot {
    pub device_id: String,
    pub exported_at: DateTime<Utc>,
    pub node_count: usize,
    pub nodes: Vec<KnowledgeNode>,
    pub clock: VectorClock,
}

/// How a sync conflict was detected.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConflictReason {
    /// Both devices modified the same node independently (concurrent vector clocks).
    ConcurrentEdit,
    /// Timestamps are identical but content differs.
    TimestampCollision,
}

/// A sync conflict that couldn't be auto-resolved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncConflict {
    pub id: Uuid,
    pub node_id: Uuid,
    pub reason: ConflictReason,
    pub local_updated_at: DateTime<Utc>,
    pub remote_updated_at: DateTime<Utc>,
    pub remote_device_id: String,
    pub local_content_preview: String,
    pub remote_content_preview: String,
    pub proposal_id: Option<Uuid>,
    pub resolved: bool,
    pub detected_at: DateTime<Utc>,
}

impl SyncConflict {
    pub fn new(
        node_id: Uuid,
        reason: ConflictReason,
        local_node: &KnowledgeNode,
        remote_node: &KnowledgeNode,
        remote_device_id: &str,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            node_id,
            reason,
            local_updated_at: local_node.temporal.updated_at,
            remote_updated_at: remote_node.temporal.updated_at,
            remote_device_id: remote_device_id.to_string(),
            local_content_preview: local_node.content.chars().take(200).collect(),
            remote_content_preview: remote_node.content.chars().take(200).collect(),
            proposal_id: None,
            resolved: false,
            detected_at: Utc::now(),
        }
    }

    pub fn with_proposal(mut self, proposal_id: Uuid) -> Self {
        self.proposal_id = Some(proposal_id);
        self
    }
}

/// Statistics from a sync import operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SyncStats {
    pub scanned: usize,
    pub inserted: usize,
    pub updated: usize,
    pub skipped: usize,
    pub conflicts: usize,
    /// Detailed conflict records (included when conflicts > 0).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub conflict_details: Vec<SyncConflict>,
}
