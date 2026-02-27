use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{ApprovalDecision, ApprovalRequest, ExecProxyRequest, HttpProxyRequest, ProxyAuditEntry};
use hx_engine::proxy::{ProxyEngine, ProxyError};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::limits::enforce_proxy_rate_limit;
use crate::state::AppState;
use crate::validation::{validate_exec_proxy_request, validate_http_proxy_request};

// ---------------------------------------------------------------------------
// Response DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct HttpProxyResponseDto {
    pub status: u16,
    pub headers: std::collections::HashMap<String, String>,
    pub body: String,
    pub sanitized: bool,
}

#[derive(Serialize)]
pub struct ExecProxyResponseDto {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
    pub sanitized: bool,
}

#[derive(Serialize)]
pub struct AuditEntryResponse {
    pub id: Uuid,
    pub consumer: String,
    pub secret_ref: String,
    pub action: String,
    pub target: String,
    pub intent: String,
    pub timestamp: String,
    pub success: Option<bool>,
    pub sanitized: bool,
    pub error: Option<String>,
    pub request_summary: String,
    pub response_status: Option<i32>,
}

impl From<ProxyAuditEntry> for AuditEntryResponse {
    fn from(e: ProxyAuditEntry) -> Self {
        Self {
            id: e.id,
            consumer: e.consumer,
            secret_ref: e.secret_ref,
            action: e.action,
            target: e.target,
            intent: e.intent,
            timestamp: e.timestamp.to_rfc3339(),
            success: e.success,
            sanitized: e.sanitized,
            error: e.error,
            request_summary: e.request_summary,
            response_status: e.response_status,
        }
    }
}

#[derive(Deserialize)]
pub struct AuditListQuery {
    pub consumer: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/proxy/http
///
/// Proxy an HTTP request with credential injection. Requires consumer authentication.
pub async fn proxy_http(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<HttpProxyRequest>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    let consumer_name = match &auth.consumer_name {
        Some(name) => name.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: "proxy requires consumer authentication (use a consumer bearer token)"
                        .into(),
                }),
            )
                .into_response();
        }
    };

    // Input validation
    if let Err(e) = validate_http_proxy_request(&req) {
        return (StatusCode::BAD_REQUEST, Json(ErrorBody { error: e.to_string() })).into_response();
    }

    // Per-consumer per-secret rate limiting
    if let Err(exceeded) = enforce_proxy_rate_limit(&consumer_name, &req.secret_ref) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorBody {
                error: format!(
                    "rate limit exceeded: {} requests per {}s (retry after {}s)",
                    exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
                ),
            }),
        )
            .into_response();
    }

    match ProxyEngine::execute_http(&req, &consumer_name, &state.engine).await {
        Ok(response) => Json(HttpProxyResponseDto {
            status: response.status,
            headers: response.headers,
            body: response.body,
            sanitized: response.sanitized,
        })
        .into_response(),
        Err(e) => proxy_error_to_response(e),
    }
}

/// POST /api/v1/proxy/exec
///
/// Run a command with credential env injection. Requires consumer authentication.
/// Uses tokio::process::Command with explicit arg list (no shell interpolation).
pub async fn proxy_exec(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<ExecProxyRequest>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    let consumer_name = match &auth.consumer_name {
        Some(name) => name.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: "proxy requires consumer authentication (use a consumer bearer token)"
                        .into(),
                }),
            )
                .into_response();
        }
    };

    // Input validation
    if let Err(e) = validate_exec_proxy_request(&req) {
        return (StatusCode::BAD_REQUEST, Json(ErrorBody { error: e.to_string() })).into_response();
    }

    // Per-consumer rate limiting (use first secret_ref for key)
    let primary_secret_ref = req.env_inject.values().next().map(|s| s.as_str()).unwrap_or("*");
    if let Err(exceeded) = enforce_proxy_rate_limit(&consumer_name, primary_secret_ref) {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            Json(ErrorBody {
                error: format!(
                    "rate limit exceeded: {} requests per {}s (retry after {}s)",
                    exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
                ),
            }),
        )
            .into_response();
    }

    match ProxyEngine::execute_exec(&req, &consumer_name, &state.engine).await {
        Ok(response) => Json(ExecProxyResponseDto {
            exit_code: response.exit_code,
            stdout: response.stdout,
            stderr: response.stderr,
            sanitized: response.sanitized,
        })
        .into_response(),
        Err(e) => proxy_error_to_response(e),
    }
}

/// GET /api/v1/proxy/audit
///
/// List proxy audit log entries. Admins see all; consumers see their own.
pub async fn list_audit(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(query): Query<AuditListQuery>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    // If authenticated as consumer, scope to their own entries
    let consumer_filter = match &auth.consumer_name {
        Some(name) => Some(name.as_str()),
        None => query.consumer.as_deref(),
    };

    let limit = query.limit.min(500);

    match state
        .engine
        .list_proxy_audit(consumer_filter, limit, query.offset)
        .await
    {
        Ok(entries) => {
            let responses: Vec<AuditEntryResponse> =
                entries.into_iter().map(AuditEntryResponse::from).collect();
            Json(responses).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to list audit entries: {e}"),
            }),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// ProxyError → HTTP Response mapping
// ---------------------------------------------------------------------------

fn proxy_error_to_response(err: ProxyError) -> axum::response::Response {
    match err {
        ProxyError::Denied(reason) => (
            StatusCode::FORBIDDEN,
            Json(ErrorBody { error: reason }),
        )
            .into_response(),
        ProxyError::ApprovalRequired {
            approval_id,
            message,
        } => (
            StatusCode::ACCEPTED,
            Json(ApprovalRequiredResponse {
                approval_id,
                message,
                poll_url: format!("/api/v1/proxy/approvals/{approval_id}"),
            }),
        )
            .into_response(),
        ProxyError::Failed(reason) => (
            StatusCode::BAD_GATEWAY,
            Json(ErrorBody { error: reason }),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// Approval DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ApprovalRequiredResponse {
    pub approval_id: Uuid,
    pub message: String,
    pub poll_url: String,
}

#[derive(Serialize)]
pub struct ApprovalResponse {
    pub id: Uuid,
    pub consumer: String,
    pub secret_key: String,
    pub intent: String,
    pub request_summary: String,
    pub state: String,
    pub created_at: String,
    pub expires_at: String,
    pub decided_at: Option<String>,
    pub decided_by: Option<String>,
    pub deny_reason: Option<String>,
    pub scopes: Vec<String>,
}

impl From<ApprovalRequest> for ApprovalResponse {
    fn from(a: ApprovalRequest) -> Self {
        Self {
            id: a.id,
            consumer: a.consumer,
            secret_key: a.secret_key,
            intent: a.intent,
            request_summary: a.request_summary,
            state: a.state.to_string(),
            created_at: a.created_at.to_rfc3339(),
            expires_at: a.expires_at.to_rfc3339(),
            decided_at: a.decided_at.map(|t| t.to_rfc3339()),
            decided_by: a.decided_by,
            deny_reason: a.deny_reason,
            scopes: a.scopes,
        }
    }
}

#[derive(Deserialize)]
pub struct ApprovalListQuery {
    pub consumer: Option<String>,
}

// ---------------------------------------------------------------------------
// Approval Handlers
// ---------------------------------------------------------------------------

/// GET /api/v1/proxy/approvals
///
/// List pending approval requests. Admins see all; consumers see their own.
pub async fn list_approvals(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(query): Query<ApprovalListQuery>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    let consumer_filter = match &auth.consumer_name {
        Some(name) => Some(name.as_str()),
        None => query.consumer.as_deref(),
    };

    match state.engine.list_pending_approvals(consumer_filter).await {
        Ok(approvals) => {
            let responses: Vec<ApprovalResponse> =
                approvals.into_iter().map(ApprovalResponse::from).collect();
            Json(responses).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to list approvals: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/proxy/approvals/:id
///
/// Get a specific approval request by ID.
pub async fn get_approval(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    match state.engine.get_approval(id).await {
        Ok(Some(approval)) => {
            // Consumers can only see their own approvals
            if let Some(ref consumer_name) = auth.consumer_name {
                if approval.consumer != *consumer_name {
                    return (
                        StatusCode::FORBIDDEN,
                        Json(ErrorBody {
                            error: "access denied: approval belongs to a different consumer".into(),
                        }),
                    )
                        .into_response();
                }
            }
            Json(ApprovalResponse::from(approval)).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: format!("approval {id} not found"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to get approval: {e}"),
            }),
        )
            .into_response(),
    }
}

/// POST /api/v1/proxy/approvals/:id
///
/// Decide on an approval request (approve or deny). Admin-only.
pub async fn decide_approval(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(decision): Json<ApprovalDecision>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    // Only admins (non-consumer auth) can decide approvals
    if auth.consumer_name.is_some() {
        return (
            StatusCode::FORBIDDEN,
            Json(ErrorBody {
                error: "only admins can decide approvals (not consumer tokens)".into(),
            }),
        )
            .into_response();
    }

    match state
        .engine
        .decide_approval(
            id,
            decision.approved,
            decision.decided_by.as_deref(),
            decision.deny_reason.as_deref(),
        )
        .await
    {
        Ok(true) => {
            // Re-fetch to return updated state
            match state.engine.get_approval(id).await {
                Ok(Some(approval)) => Json(ApprovalResponse::from(approval)).into_response(),
                _ => (StatusCode::OK, Json(ErrorBody { error: "decided".into() })).into_response(),
            }
        }
        Ok(false) => (
            StatusCode::CONFLICT,
            Json(ErrorBody {
                error: format!("approval {id} is no longer pending (already decided or expired)"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to decide approval: {e}"),
            }),
        )
            .into_response(),
    }
}
