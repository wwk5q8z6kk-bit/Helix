use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// An access policy controlling a consumer's access to a specific secret.
///
/// Implements ABAC with default-deny semantics: no policy = no access.
/// Supports scopes, TTL (zero-standing privileges), expiration, and HITL flags.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPolicy {
    pub id: Uuid,
    pub secret_key: String,
    pub consumer: String,
    pub allowed: bool,
    pub scopes: Vec<String>,
    pub max_ttl_seconds: Option<i64>,
    pub expires_at: Option<DateTime<Utc>>,
    pub require_approval: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl AccessPolicy {
    pub fn is_expired(&self) -> bool {
        self.expires_at
            .map(|exp| Utc::now() > exp)
            .unwrap_or(false)
    }

    pub fn is_effective(&self) -> bool {
        self.allowed && !self.is_expired()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PolicyDecision {
    Allow {
        ttl_seconds: Option<i64>,
        scopes: Vec<String>,
    },
    Deny {
        reason: String,
    },
    /// Policy grants access but requires human-in-the-loop approval first.
    RequiresApproval {
        ttl_seconds: i64,
        scopes: Vec<String>,
    },
}

impl PolicyDecision {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow { .. })
    }

    pub fn requires_approval(&self) -> bool {
        matches!(self, Self::RequiresApproval { .. })
    }
}
