use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{AutonomyActionLog, AutonomyDecision, AutonomyRule, AutonomyStore};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct CreateRuleRequest {
    pub rule_type: String,
    pub scope_key: Option<String>,
    pub auto_apply_threshold: Option<f32>,
    pub max_actions_per_hour: Option<u32>,
    pub allowed_intent_types: Option<Vec<String>>,
    pub blocked_intent_types: Option<Vec<String>>,
    pub quiet_hours_start: Option<String>,
    pub quiet_hours_end: Option<String>,
    pub quiet_hours_timezone: Option<String>,
    pub enabled: Option<bool>,
}

#[derive(Deserialize)]
pub struct UpdateRuleRequest {
    pub rule_type: Option<String>,
    pub scope_key: Option<String>,
    pub auto_apply_threshold: Option<f32>,
    pub max_actions_per_hour: Option<u32>,
    pub allowed_intent_types: Option<Vec<String>>,
    pub blocked_intent_types: Option<Vec<String>>,
    pub quiet_hours_start: Option<String>,
    pub quiet_hours_end: Option<String>,
    pub quiet_hours_timezone: Option<String>,
    pub enabled: Option<bool>,
}

#[derive(Deserialize)]
pub struct ActionLogQuery {
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct EvaluateRequest {
    pub intent_type: String,
    pub confidence: f32,
    pub scope_hints: Option<Vec<ScopeHint>>,
}

#[derive(Deserialize)]
pub struct ScopeHint {
    pub rule_type: String,
    pub scope_key: String,
}

#[derive(Serialize)]
pub struct EvaluateResponse {
    pub decision: AutonomyDecision,
}

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::NodeNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

// --- Handlers ---

/// GET /api/v1/autonomy/rules
pub async fn list_rules(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<AutonomyRule>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let rules = state
        .engine
        .store
        .nodes
        .list_autonomy_rules()
        .await
        .map_err(map_hx_error)?;

    Ok(Json(rules))
}

/// POST /api/v1/autonomy/rules
pub async fn create_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateRuleRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let valid_types = ["global", "domain", "contact", "tag"];
    if !valid_types.contains(&req.rule_type.as_str()) {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("rule_type must be one of: {}", valid_types.join(", ")),
        ));
    }

    if req.rule_type != "global" && req.scope_key.is_none() {
        return Err((
            StatusCode::BAD_REQUEST,
            "scope_key is required for non-global rules".to_string(),
        ));
    }

    let mut rule = AutonomyRule::global(req.auto_apply_threshold.unwrap_or(0.95));
    rule.rule_type = req.rule_type;
    rule.scope_key = req.scope_key;

    if let Some(max) = req.max_actions_per_hour {
        rule.max_actions_per_hour = max;
    }
    if let Some(allowed) = req.allowed_intent_types {
        rule.allowed_intent_types = allowed;
    }
    if let Some(blocked) = req.blocked_intent_types {
        rule.blocked_intent_types = blocked;
    }
    if let Some(start) = req.quiet_hours_start {
        rule.quiet_hours_start = Some(start);
    }
    if let Some(end) = req.quiet_hours_end {
        rule.quiet_hours_end = Some(end);
    }
    if let Some(tz) = req.quiet_hours_timezone {
        rule.quiet_hours_timezone = tz;
    }
    if let Some(enabled) = req.enabled {
        rule.enabled = enabled;
    }

    state
        .engine
        .store
        .nodes
        .add_autonomy_rule(&rule)
        .await
        .map_err(map_hx_error)?;

    Ok((StatusCode::CREATED, Json(rule)).into_response())
}

/// GET /api/v1/autonomy/rules/:id
pub async fn get_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<AutonomyRule>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid rule id".to_string()))?;

    let rule = state
        .engine
        .store
        .nodes
        .get_autonomy_rule(uuid)
        .await
        .map_err(map_hx_error)?
        .ok_or((StatusCode::NOT_FOUND, "rule not found".to_string()))?;

    Ok(Json(rule))
}

/// PUT /api/v1/autonomy/rules/:id
pub async fn update_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateRuleRequest>,
) -> Result<Json<AutonomyRule>, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid rule id".to_string()))?;

    let mut rule = state
        .engine
        .store
        .nodes
        .get_autonomy_rule(uuid)
        .await
        .map_err(map_hx_error)?
        .ok_or((StatusCode::NOT_FOUND, "rule not found".to_string()))?;

    if let Some(rt) = req.rule_type {
        rule.rule_type = rt;
    }
    if let Some(sk) = req.scope_key {
        rule.scope_key = Some(sk);
    }
    if let Some(threshold) = req.auto_apply_threshold {
        rule.auto_apply_threshold = threshold;
    }
    if let Some(max) = req.max_actions_per_hour {
        rule.max_actions_per_hour = max;
    }
    if let Some(allowed) = req.allowed_intent_types {
        rule.allowed_intent_types = allowed;
    }
    if let Some(blocked) = req.blocked_intent_types {
        rule.blocked_intent_types = blocked;
    }
    if let Some(start) = req.quiet_hours_start {
        rule.quiet_hours_start = Some(start);
    }
    if let Some(end) = req.quiet_hours_end {
        rule.quiet_hours_end = Some(end);
    }
    if let Some(tz) = req.quiet_hours_timezone {
        rule.quiet_hours_timezone = tz;
    }
    if let Some(enabled) = req.enabled {
        rule.enabled = enabled;
    }

    rule.updated_at = Some(Utc::now());

    state
        .engine
        .store
        .nodes
        .update_autonomy_rule(&rule)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(rule))
}

/// DELETE /api/v1/autonomy/rules/:id
pub async fn delete_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid rule id".to_string()))?;

    let deleted = state
        .engine
        .store
        .nodes
        .delete_autonomy_rule(uuid)
        .await
        .map_err(map_hx_error)?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "rule not found".to_string()))
    }
}

/// GET /api/v1/autonomy/action-log
pub async fn list_action_log(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<ActionLogQuery>,
) -> Result<Json<Vec<AutonomyActionLog>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let limit = params.limit.unwrap_or(50).min(500);

    let logs = state
        .engine
        .store
        .nodes
        .list_autonomy_action_log(limit)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(logs))
}

/// POST /api/v1/autonomy/evaluate
pub async fn evaluate(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<EvaluateRequest>,
) -> Result<Json<EvaluateResponse>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let hints: Vec<(String, String)> = req
        .scope_hints
        .unwrap_or_default()
        .into_iter()
        .map(|h| (h.rule_type, h.scope_key))
        .collect();

    let hint_refs: Vec<(&str, &str)> = hints
        .iter()
        .map(|(rt, sk)| (rt.as_str(), sk.as_str()))
        .collect();

    let decision = state
        .engine
        .autonomy
        .evaluate(&req.intent_type, req.confidence, &hint_refs)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(EvaluateResponse { decision }))
}
