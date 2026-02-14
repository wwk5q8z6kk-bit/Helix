use std::sync::Arc;

use axum::{extract::{Path, State}, http::StatusCode, Extension, Json};
use chrono::{DateTime, Utc};
use serde::Deserialize;

use hx_engine::sync::{SyncSnapshot, SyncStats};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::NodeNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        hx_core::HxError::DuplicateNode(_) => (StatusCode::CONFLICT, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

#[derive(Deserialize)]
pub struct SyncExportRequest {
    pub since: Option<String>,
    pub namespace: Option<String>,
}

/// POST /api/v1/sync/export -- Export snapshot
pub async fn sync_export(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(body): Json<SyncExportRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_read(&auth)?;
    let since = body.since.and_then(|s| {
        DateTime::parse_from_rfc3339(&s)
            .ok()
            .map(|d| d.with_timezone(&Utc))
    });
    let snapshot = state
        .engine
        .sync
        .export_snapshot(since, body.namespace.as_deref())
        .await
        .map_err(map_hx_error)?;
    Ok(Json(serde_json::to_value(snapshot).unwrap_or_default()))
}

/// POST /api/v1/sync/import -- Import snapshot
pub async fn sync_import(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(snapshot): Json<SyncSnapshot>,
) -> Result<Json<SyncStats>, (StatusCode, String)> {
    authorize_write(&auth)?;
    let stats = state
        .engine
        .sync
        .import_snapshot(snapshot)
        .await
        .map_err(map_hx_error)?;
    Ok(Json(stats))
}

/// GET /api/v1/sync/status — Device sync status including clock and conflicts
pub async fn sync_status(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_read(&auth)?;
    let node_count = state.engine.node_count().await.map_err(map_hx_error)?;
    let clock = state.engine.sync.clock().await;
    let conflicts = state.engine.sync.unresolved_conflicts().await;
    let last_export = state.engine.sync.last_export().await;
    let last_import = state.engine.sync.last_import().await;
    Ok(Json(serde_json::json!({
        "device_id": state.engine.sync.device_id(),
        "status": "ready",
        "node_count": node_count,
        "clock": clock,
        "unresolved_conflicts": conflicts.len(),
        "conflicts": conflicts,
        "last_export": last_export.map(|dt| dt.to_rfc3339()),
        "last_import": last_import.map(|dt| dt.to_rfc3339()),
    })))
}

/// POST /api/v1/sync/conflicts/{id}/resolve — Mark a sync conflict as resolved
pub async fn resolve_sync_conflict(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid = uuid::Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid conflict id: {e}")))?;

    let resolved = state.engine.sync.resolve_conflict_by_id(uuid).await;
    if resolved {
        Ok(Json(serde_json::json!({ "status": "resolved" })))
    } else {
        Err((StatusCode::NOT_FOUND, "conflict not found".into()))
    }
}
