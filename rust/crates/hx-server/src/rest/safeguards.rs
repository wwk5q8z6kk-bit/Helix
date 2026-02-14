use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use uuid::Uuid;

use hx_core::{AutoApproveRule, BlockedSender, SafeguardStore};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct AddBlockedSenderRequest {
    pub sender_type: String,
    pub sender_pattern: String,
    pub reason: Option<String>,
    pub expires_at: Option<String>,
}

#[derive(Deserialize)]
pub struct AddAutoApproveRuleRequest {
    pub name: String,
    pub sender_pattern: Option<String>,
    pub action_types: Option<Vec<String>>,
    pub min_confidence: f32,
}

#[derive(Deserialize)]
pub struct UpdateAutoApproveRuleRequest {
    pub name: Option<String>,
    pub sender_pattern: Option<Option<String>>,
    pub action_types: Option<Vec<String>>,
    pub min_confidence: Option<f32>,
    pub enabled: Option<bool>,
}

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::NodeNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

// --- Blocked Senders ---

/// GET /api/v1/exchange/blocked-senders
pub async fn list_blocked_senders(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<BlockedSender>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let senders = state
        .engine
        .store
        .nodes
        .list_blocked_senders()
        .await
        .map_err(map_hx_error)?;

    Ok(Json(senders))
}

/// POST /api/v1/exchange/blocked-senders
pub async fn add_blocked_sender(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddBlockedSenderRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let valid_types = ["agent", "mcp", "webhook", "watcher", "relay"];
    if !valid_types.contains(&req.sender_type.as_str()) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "invalid sender_type: {}. Must be one of: {}",
                req.sender_type,
                valid_types.join(", ")
            ),
        ));
    }

    let mut sender = BlockedSender::new(req.sender_type, req.sender_pattern);

    if let Some(reason) = req.reason {
        sender = sender.with_reason(reason);
    }

    if let Some(expires_str) = req.expires_at {
        let expires: DateTime<Utc> = DateTime::parse_from_rfc3339(&expires_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid expires_at: {e}")))?;
        sender = sender.with_expiry(expires);
    }

    state
        .engine
        .store
        .nodes
        .add_blocked_sender(&sender)
        .await
        .map_err(map_hx_error)?;

    Ok((StatusCode::CREATED, Json(sender)).into_response())
}

/// DELETE /api/v1/exchange/blocked-senders/:id
pub async fn remove_blocked_sender(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let removed = state
        .engine
        .store
        .nodes
        .remove_blocked_sender(uuid)
        .await
        .map_err(map_hx_error)?;

    if !removed {
        return Err((
            StatusCode::NOT_FOUND,
            "blocked sender not found".to_string(),
        ));
    }

    Ok(Json(serde_json::json!({ "id": id, "removed": true })))
}

// --- Auto-Approve Rules ---

/// GET /api/v1/exchange/auto-approve-rules
pub async fn list_auto_approve_rules(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<AutoApproveRule>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let rules = state
        .engine
        .store
        .nodes
        .list_auto_approve_rules()
        .await
        .map_err(map_hx_error)?;

    Ok(Json(rules))
}

/// POST /api/v1/exchange/auto-approve-rules
pub async fn add_auto_approve_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<AddAutoApproveRuleRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    if req.min_confidence < 0.0 || req.min_confidence > 1.0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "min_confidence must be between 0.0 and 1.0".to_string(),
        ));
    }

    let mut rule = AutoApproveRule::new(req.name, req.min_confidence);

    if let Some(pattern) = req.sender_pattern {
        rule = rule.with_sender(pattern);
    }

    if let Some(actions) = req.action_types {
        rule = rule.with_actions(actions);
    }

    state
        .engine
        .store
        .nodes
        .add_auto_approve_rule(&rule)
        .await
        .map_err(map_hx_error)?;

    Ok((StatusCode::CREATED, Json(rule)).into_response())
}

/// PUT /api/v1/exchange/auto-approve-rules/:id
pub async fn update_auto_approve_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateAutoApproveRuleRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    // Fetch existing rule
    let rules = state
        .engine
        .store
        .nodes
        .list_auto_approve_rules()
        .await
        .map_err(map_hx_error)?;

    let mut rule = rules.into_iter().find(|r| r.id == uuid).ok_or((
        StatusCode::NOT_FOUND,
        "auto-approve rule not found".to_string(),
    ))?;

    if let Some(name) = req.name {
        rule.name = name;
    }
    if let Some(sender_pattern) = req.sender_pattern {
        rule.sender_pattern = sender_pattern;
    }
    if let Some(actions) = req.action_types {
        rule.action_types = actions;
    }
    if let Some(confidence) = req.min_confidence {
        if confidence < 0.0 || confidence > 1.0 {
            return Err((
                StatusCode::BAD_REQUEST,
                "min_confidence must be between 0.0 and 1.0".to_string(),
            ));
        }
        rule.min_confidence = confidence;
    }
    if let Some(enabled) = req.enabled {
        rule.enabled = enabled;
    }
    rule.updated_at = Some(Utc::now());

    let updated = state
        .engine
        .store
        .nodes
        .update_auto_approve_rule(&rule)
        .await
        .map_err(map_hx_error)?;

    if !updated {
        return Err((
            StatusCode::NOT_FOUND,
            "auto-approve rule not found".to_string(),
        ));
    }

    Ok(Json(rule))
}

/// DELETE /api/v1/exchange/auto-approve-rules/:id
pub async fn remove_auto_approve_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let removed = state
        .engine
        .store
        .nodes
        .remove_auto_approve_rule(uuid)
        .await
        .map_err(map_hx_error)?;

    if !removed {
        return Err((
            StatusCode::NOT_FOUND,
            "auto-approve rule not found".to_string(),
        ));
    }

    Ok(Json(serde_json::json!({ "id": id, "removed": true })))
}

