use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;

use hx_engine::planner::Planner;

use crate::state::AppState;

#[derive(Deserialize)]
pub struct CreatePlanRequest {
    pub goal: String,
}

/// POST /api/v1/plans — Create a new plan from a goal
pub async fn create_plan(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreatePlanRequest>,
) -> impl IntoResponse {
    let planner = Planner::new(
        state.engine.llm.clone(),
        state.engine.config.planning.clone(),
    );

    if !planner.is_enabled() {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "planning is disabled in configuration"})),
        )
            .into_response();
    }

    let plan = planner.create_plan(&req.goal).await;
    (StatusCode::CREATED, Json(&plan)).into_response()
}

/// POST /api/v1/plans/:id/approve — Approve a draft plan for execution
pub async fn approve_plan(Path(id): Path<String>) -> impl IntoResponse {
    // In a full implementation, this would look up the plan from the store,
    // verify it's in Draft status, and transition to Approved.
    // For now, return success to show the API shape.
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "id": id,
            "status": "approved",
            "message": "plan approved for execution"
        })),
    )
        .into_response()
}

/// GET /api/v1/plans/:id — Get plan details
pub async fn get_plan(Path(id): Path<String>) -> impl IntoResponse {
    // Placeholder — in full implementation, look up from PlanStore
    (
        StatusCode::NOT_FOUND,
        Json(serde_json::json!({"error": format!("plan {id} not found")})),
    )
        .into_response()
}
