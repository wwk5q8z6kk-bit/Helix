use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// A named identity for an AI consumer (agent, tool, service).
///
/// Each consumer gets a unique bearer token (stored as SHA-256 hash).
/// This enables per-consumer access policies and audit trails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerProfile {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub token_hash: String,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ConsumerProfile {
    pub fn is_active(&self) -> bool {
        self.revoked_at.is_none()
    }
}

/// Public view of a consumer profile (excludes token_hash).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsumerProfileSummary {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_used_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl From<ConsumerProfile> for ConsumerProfileSummary {
    fn from(p: ConsumerProfile) -> Self {
        Self {
            id: p.id,
            name: p.name,
            description: p.description,
            created_at: p.created_at,
            last_used_at: p.last_used_at,
            revoked_at: p.revoked_at,
            metadata: p.metadata,
        }
    }
}
