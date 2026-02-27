//! REST handlers for querying the audit log.

use std::sync::Arc;

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::audit::{list_audit_entries, AuditEntry};
use crate::auth::AuthContext;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct AuditQuery {
    pub since: Option<String>,
    pub until: Option<String>,
    pub subject: Option<String>,
    pub action: Option<String>,
    pub status: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Serialize)]
pub struct AuditResponse {
    pub entries: Vec<AuditEntry>,
    pub total: usize,
}

/// GET /api/v1/audit — query audit log with filters.
pub async fn query_audit(
    Extension(auth): Extension<AuthContext>,
    State(_state): State<Arc<AppState>>,
    Query(params): Query<AuditQuery>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, Json(AuditResponse { entries: Vec::new(), total: 0 })).into_response();
    }

    let limit = params.limit.unwrap_or(100).min(1000);
    let offset = params.offset.unwrap_or(0);

    let since: Option<DateTime<Utc>> = params
        .since
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    let entries = list_audit_entries(
        limit,
        offset,
        params.subject.as_deref(),
        params.action.as_deref(),
        since,
    );

    // Post-filter by status if provided
    let entries = if let Some(ref status_filter) = params.status {
        let want_success = status_filter == "success" || status_filter == "ok";
        entries
            .into_iter()
            .filter(|e| e.success == want_success)
            .collect::<Vec<_>>()
    } else {
        entries
    };

    let total = entries.len();
    Json(AuditResponse { entries, total }).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audit_response_serializes() {
        let resp = AuditResponse {
            entries: Vec::new(),
            total: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"total\":0"));
    }

    #[test]
    fn audit_query_deserializes_defaults() {
        let q: AuditQuery = serde_urlencoded::from_str("").unwrap();
        assert!(q.since.is_none());
        assert!(q.limit.is_none());
        assert!(q.subject.is_none());
    }

    #[test]
    fn audit_query_deserializes_with_params() {
        let q: AuditQuery =
            serde_urlencoded::from_str("limit=50&subject=user1&action=store_node").unwrap();
        assert_eq!(q.limit, Some(50));
        assert_eq!(q.subject.as_deref(), Some("user1"));
        assert_eq!(q.action.as_deref(), Some("store_node"));
    }
}
