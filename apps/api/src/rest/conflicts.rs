use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ConflictResponse {
    pub id: String,
    pub node_a: String,
    pub node_b: String,
    pub conflict_type: String,
    pub score: f64,
    pub explanation: String,
    pub resolved: bool,
    pub created_at: String,
}

#[derive(Deserialize)]
pub struct ListConflictsQuery {
    pub resolved: Option<bool>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /api/v1/conflicts
pub async fn list_conflicts(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(q): Query<ListConflictsQuery>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    let limit = q.limit.unwrap_or(50).min(200);
    let offset = q.offset.unwrap_or(0);

    match state.engine.list_conflicts(q.resolved, limit, offset).await {
        Ok(conflicts) => {
            let items: Vec<ConflictResponse> = conflicts
                .into_iter()
                .map(|c| ConflictResponse {
                    id: c.id.to_string(),
                    node_a: c.node_a.to_string(),
                    node_b: c.node_b.to_string(),
                    conflict_type: c.conflict_type.as_str().to_string(),
                    score: c.score,
                    explanation: c.explanation,
                    resolved: c.resolved,
                    created_at: c.created_at.to_rfc3339(),
                })
                .collect();
            Json(items).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to list conflicts: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/conflicts/:id
pub async fn get_conflict(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    match state.engine.get_conflict(id).await {
        Ok(Some(c)) => Json(ConflictResponse {
            id: c.id.to_string(),
            node_a: c.node_a.to_string(),
            node_b: c.node_b.to_string(),
            conflict_type: c.conflict_type.as_str().to_string(),
            score: c.score,
            explanation: c.explanation,
            resolved: c.resolved,
            created_at: c.created_at.to_rfc3339(),
        })
        .into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: "conflict not found".into(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to get conflict: {e}"),
            }),
        )
            .into_response(),
    }
}

/// POST /api/v1/conflicts/:id/resolve
pub async fn resolve_conflict(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    match state.engine.resolve_conflict(id).await {
        Ok(true) => Json(serde_json::json!({"resolved": true})).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: "conflict not found or already resolved".into(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to resolve conflict: {e}"),
            }),
        )
            .into_response(),
    }
}
