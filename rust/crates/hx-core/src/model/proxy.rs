use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// How a secret should be injected into an HTTP request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SecretInjection {
    BearerHeader,
    BasicAuth { username: String },
    Header { name: String },
    QueryParam { name: String },
}

/// An HTTP proxy request to execute with credential injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpProxyRequest {
    pub method: String,
    pub url: String,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub body: Option<String>,
    pub secret_ref: String,
    pub inject_as: SecretInjection,
    pub intent: String,
}

/// Result of an HTTP proxy execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpProxyResponse {
    pub status: u16,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    pub body: String,
    pub sanitized: bool,
}

/// A command execution proxy request with credential env injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecProxyRequest {
    pub command: String,
    pub args: Vec<String>,
    #[serde(default)]
    pub env_inject: HashMap<String, String>,
    pub working_dir: Option<String>,
    #[serde(default = "default_exec_timeout")]
    pub timeout_seconds: u64,
    pub intent: String,
}

fn default_exec_timeout() -> u64 {
    30
}

/// Result of a command execution proxy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecProxyResponse {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub sanitized: bool,
}

/// An audit entry for proxy operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyAuditEntry {
    pub id: Uuid,
    pub consumer: String,
    pub secret_ref: String,
    pub action: String,
    pub target: String,
    pub intent: String,
    pub timestamp: DateTime<Utc>,
    pub success: Option<bool>,
    pub sanitized: bool,
    pub error: Option<String>,
    pub request_summary: String,
    pub response_status: Option<i32>,
}

// ---------------------------------------------------------------------------
// HITL Approval Queue
// ---------------------------------------------------------------------------

/// State of a proxy approval request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalState {
    Pending,
    Approved,
    Denied,
    Expired,
}

impl ApprovalState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Denied => "denied",
            Self::Expired => "expired",
        }
    }
}

impl std::str::FromStr for ApprovalState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "approved" => Ok(Self::Approved),
            "denied" => Ok(Self::Denied),
            "expired" => Ok(Self::Expired),
            _ => Err(format!("unknown approval state: {s}")),
        }
    }
}

impl std::fmt::Display for ApprovalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A pending approval request for a proxy operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    pub id: Uuid,
    pub consumer: String,
    pub secret_key: String,
    pub intent: String,
    pub request_summary: String,
    pub state: ApprovalState,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub decided_at: Option<DateTime<Utc>>,
    pub decided_by: Option<String>,
    pub deny_reason: Option<String>,
    pub scopes: Vec<String>,
}

impl ApprovalRequest {
    pub fn new(
        consumer: impl Into<String>,
        secret_key: impl Into<String>,
        intent: impl Into<String>,
        request_summary: impl Into<String>,
        ttl_seconds: i64,
        scopes: Vec<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            consumer: consumer.into(),
            secret_key: secret_key.into(),
            intent: intent.into(),
            request_summary: request_summary.into(),
            state: ApprovalState::Pending,
            created_at: now,
            expires_at: now + chrono::Duration::seconds(ttl_seconds),
            decided_at: None,
            decided_by: None,
            deny_reason: None,
            scopes,
        }
    }

    pub fn is_pending(&self) -> bool {
        self.state == ApprovalState::Pending
    }

    pub fn is_expired(&self) -> bool {
        self.state == ApprovalState::Expired || Utc::now() > self.expires_at
    }
}

/// Decision payload for approving/denying a pending request.
#[derive(Debug, Clone, Deserialize)]
pub struct ApprovalDecision {
    pub approved: bool,
    pub decided_by: Option<String>,
    pub deny_reason: Option<String>,
}

impl ProxyAuditEntry {
    pub fn new(
        consumer: impl Into<String>,
        secret_ref: impl Into<String>,
        action: impl Into<String>,
        target: impl Into<String>,
        intent: impl Into<String>,
        request_summary: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            consumer: consumer.into(),
            secret_ref: secret_ref.into(),
            action: action.into(),
            target: target.into(),
            intent: intent.into(),
            timestamp: Utc::now(),
            success: None,
            sanitized: false,
            error: None,
            request_summary: request_summary.into(),
            response_status: None,
        }
    }
}
