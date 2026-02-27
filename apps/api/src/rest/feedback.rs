use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{AgentFeedback, ConfidenceOverride, ReflectionStats};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct RecordFeedbackRequest {
    pub intent_id: Option<String>,
    pub intent_type: String,
    pub action: String,
    pub confidence_at_time: Option<f32>,
    pub user_edit_delta: Option<f32>,
    pub response_time_ms: Option<u64>,
}

#[derive(Deserialize)]
pub struct ListFeedbackQuery {
    pub intent_type: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct ReflectionStatsQuery {
    pub intent_type: String,
}

#[derive(Deserialize)]
pub struct SetOverrideRequest {
    pub base_adjustment: Option<f32>,
    pub auto_apply_threshold: Option<f32>,
    pub suppress_below: Option<f32>,
}

#[derive(Serialize)]
pub struct CalibrateSuggestion {
    pub intent_type: String,
    pub suggested_adjustment: f32,
}

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::NodeNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

// --- Handlers ---

/// POST /api/v1/agent/feedback
pub async fn record_feedback(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<RecordFeedbackRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let mut fb = AgentFeedback::new(&req.intent_type, &req.action);

    if let Some(ref intent_id_str) = req.intent_id {
        let intent_id = Uuid::parse_str(intent_id_str)
            .map_err(|_| (StatusCode::BAD_REQUEST, "invalid intent_id".to_string()))?;
        fb = fb.with_intent(intent_id);
    }

    if let Some(c) = req.confidence_at_time {
        fb = fb.with_confidence(c);
    }

    if let Some(d) = req.user_edit_delta {
        fb = fb.with_delta(d);
    }

    if let Some(ms) = req.response_time_ms {
        fb = fb.with_response_time(ms);
    }

    state
        .engine
        .reflection
        .record_feedback(&fb)
        .await
        .map_err(map_hx_error)?;

    Ok((StatusCode::CREATED, Json(fb)).into_response())
}

/// GET /api/v1/agent/feedback
pub async fn list_feedback(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListFeedbackQuery>,
) -> Result<Json<Vec<AgentFeedback>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let limit = params.limit.unwrap_or(50).min(200);
    let intent_type_ref = params.intent_type.as_deref();

    let feedback = state
        .engine
        .reflection
        .list_feedback(intent_type_ref, limit)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(feedback))
}

/// GET /api/v1/agent/reflection/stats
pub async fn reflection_stats(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<ReflectionStatsQuery>,
) -> Result<Json<ReflectionStats>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let stats = state
        .engine
        .reflection
        .get_stats(&params.intent_type)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(stats))
}

/// POST /api/v1/agent/reflection/calibrate
pub async fn calibrate(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<CalibrateSuggestion>>, (StatusCode, String)> {
    authorize_write(&auth)?;

    let suggestions = state
        .engine
        .reflection
        .recalibrate()
        .await
        .map_err(map_hx_error)?;

    let response: Vec<CalibrateSuggestion> = suggestions
        .into_iter()
        .map(|(intent_type, suggested_adjustment)| CalibrateSuggestion {
            intent_type,
            suggested_adjustment,
        })
        .collect();

    Ok(Json(response))
}

/// GET /api/v1/agent/confidence-overrides
pub async fn list_confidence_overrides(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ConfidenceOverride>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let overrides = state
        .engine
        .reflection
        .list_confidence_overrides()
        .await
        .map_err(map_hx_error)?;

    Ok(Json(overrides))
}

/// PUT /api/v1/agent/confidence-overrides/:type
pub async fn set_confidence_override(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(intent_type): Path<String>,
    Json(req): Json<SetOverrideRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let mut override_ = ConfidenceOverride::new(&intent_type);

    if let Some(adj) = req.base_adjustment {
        override_.base_adjustment = adj;
    }
    if let Some(thresh) = req.auto_apply_threshold {
        override_.auto_apply_threshold = thresh;
    }
    if let Some(suppress) = req.suppress_below {
        override_.suppress_below = suppress;
    }

    state
        .engine
        .reflection
        .set_confidence_override(&override_)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(override_))
}
