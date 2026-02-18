//! REST API endpoints for the hx-scheduler DAG-based scheduling service.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};

use hx_scheduler::WorkflowDefinition;

use crate::auth::AuthContext;
use crate::state::AppState;

// ── DTOs ────────────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct WorkflowDefResponse {
    pub name: String,
    pub task_count: usize,
    pub wave_count: usize,
    pub max_parallelism: usize,
    pub task_order: Vec<String>,
    pub waves: Vec<Vec<String>>,
    pub deadline: Option<f64>,
    pub budget: Option<f64>,
}

impl From<&WorkflowDefinition> for WorkflowDefResponse {
    fn from(d: &WorkflowDefinition) -> Self {
        Self {
            name: d.name.clone(),
            task_count: d.task_count(),
            wave_count: d.wave_count(),
            max_parallelism: d.max_parallelism(),
            task_order: d.task_order.clone(),
            waves: d.waves.clone(),
            deadline: d.deadline,
            budget: d.budget,
        }
    }
}

#[derive(Deserialize)]
pub struct DefineWorkflowDto {
    pub name: String,
    pub tasks: Vec<TaskDto>,
    pub deadline: Option<f64>,
    pub budget: Option<f64>,
}

#[derive(Deserialize)]
pub struct TaskDto {
    pub task_id: String,
    pub task_type: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default = "default_priority")]
    pub priority: i32,
    pub model: Option<String>,
}

fn default_priority() -> i32 {
    2
}

#[derive(Deserialize)]
pub struct SubmitTaskDto {
    pub task_id: String,
    #[serde(default)]
    pub dependencies: Vec<String>,
    #[serde(default = "default_priority")]
    pub priority: i32,
}

#[derive(Serialize)]
pub struct TaskStateResponse {
    pub task_id: String,
    pub state: String,
}

#[derive(Serialize)]
pub struct PreviewResponse {
    pub name: String,
    pub task_count: usize,
    pub waves: Vec<Vec<String>>,
    pub wave_durations: Vec<f64>,
    pub critical_path: Vec<String>,
    pub total_predicted_duration: f64,
    pub total_predicted_cost: f64,
    pub avg_predicted_quality: f64,
    pub feasible: bool,
    pub risk_level: String,
    pub meets_deadline: Option<bool>,
    pub meets_budget: Option<bool>,
    pub deadline_margin: Option<f64>,
    pub budget_margin: Option<f64>,
    pub at_risk_tasks: Vec<String>,
    pub bottleneck_tasks: Vec<String>,
    pub low_confidence_tasks: Vec<String>,
}

#[derive(Serialize)]
pub struct TemplateListResponse {
    pub templates: Vec<String>,
}

fn authorize_admin(auth: &AuthContext) -> Result<(), (StatusCode, String)> {
    if auth.is_admin() {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, "admin permission required".into()))
    }
}

// ── Workflow CRUD ───────────────────────────────────────────────────

/// POST /api/v1/scheduler/workflows
pub async fn define_workflow(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<DefineWorkflowDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let svc = get_scheduling_service(&state)?;

    // Convert DTOs to builder-compatible tuples.
    let tasks: Vec<hx_scheduler::WorkflowTask> = req
        .tasks
        .iter()
        .map(|t| hx_scheduler::WorkflowTask {
            task_id: t.task_id.clone(),
            task_type: t.task_type.clone(),
            depends_on: t.depends_on.clone(),
            model: t.model.clone(),
            priority: t.priority,
            gate: None,
            retry_strategy: None,
            metadata: Default::default(),
        })
        .collect();

    let def = svc
        .define_workflow_from_tasks(&req.name, tasks, req.deadline, req.budget)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(WorkflowDefResponse::from(&def))).into_response())
}

/// GET /api/v1/scheduler/workflows
pub async fn list_workflows(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<WorkflowDefResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let svc = get_scheduling_service(&state)?;
    let defs = svc.list_workflows().await;
    Ok(Json(defs.iter().map(WorkflowDefResponse::from).collect()))
}

/// GET /api/v1/scheduler/workflows/:name
pub async fn get_workflow(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<WorkflowDefResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let svc = get_scheduling_service(&state)?;
    let def = svc
        .get_workflow(&name)
        .await
        .ok_or((StatusCode::NOT_FOUND, format!("workflow '{name}' not found")))?;
    Ok(Json(WorkflowDefResponse::from(&def)))
}

/// DELETE /api/v1/scheduler/workflows/:name
pub async fn delete_workflow(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let svc = get_scheduling_service(&state)?;
    if svc.remove_workflow(&name).await {
        Ok(Json(serde_json::json!({ "status": "deleted" })).into_response())
    } else {
        Err((StatusCode::NOT_FOUND, format!("workflow '{name}' not found")))
    }
}

// ── Preview ─────────────────────────────────────────────────────────

/// POST /api/v1/scheduler/workflows/:name/preview
pub async fn preview_stored_workflow(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<PreviewResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let svc = get_scheduling_service(&state)?;

    // Re-build the workflow to get a fresh preview with current coordinator state.
    let def = svc
        .get_workflow(&name)
        .await
        .ok_or((StatusCode::NOT_FOUND, format!("workflow '{name}' not found")))?;

    // Build tasks tuple list from the definition.
    let tasks: Vec<(&str, &str, Vec<&str>, i32)> = def
        .tasks
        .values()
        .map(|t| {
            let deps: Vec<&str> = t.depends_on.iter().map(|s| s.as_str()).collect();
            (t.task_id.as_str(), t.task_type.as_str(), deps, t.priority)
        })
        .collect();

    // Build temporary task slice for preview.
    let task_tuples: Vec<(&str, &str, &[&str], i32)> = tasks
        .iter()
        .map(|(id, tt, deps, p)| (*id, *tt, deps.as_slice(), *p))
        .collect();

    let preview = svc
        .preview_workflow(&name, &task_tuples, def.deadline, def.budget)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(preview_to_response(preview)))
}

// ── Templates ───────────────────────────────────────────────────────

/// GET /api/v1/scheduler/templates
pub async fn list_templates(
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<TemplateListResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    Ok(Json(TemplateListResponse {
        templates: hx_engine::scheduling::SchedulingService::template_names()
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
    }))
}

/// POST /api/v1/scheduler/templates/:name/preview
pub async fn preview_template(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<PreviewResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let svc = get_scheduling_service(&state)?;
    let preview = svc
        .preview_template(&name)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok(Json(preview_to_response(preview)))
}

// ── Live DAG ────────────────────────────────────────────────────────

/// POST /api/v1/scheduler/tasks
pub async fn submit_task(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<SubmitTaskDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let svc = get_scheduling_service(&state)?;
    let deps: Vec<&str> = req.dependencies.iter().map(|s| s.as_str()).collect();
    let task_state = svc
        .submit_task(&req.task_id, &deps, req.priority)
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(TaskStateResponse {
            task_id: req.task_id,
            state: task_state.to_string(),
        }),
    )
        .into_response())
}

/// GET /api/v1/scheduler/tasks/ready
pub async fn ready_tasks(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<String>>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let svc = get_scheduling_service(&state)?;
    Ok(Json(svc.ready_tasks().await))
}

/// GET /api/v1/scheduler/tasks/waves
pub async fn execution_waves(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<Vec<String>>>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let svc = get_scheduling_service(&state)?;
    Ok(Json(svc.execution_waves().await))
}

/// GET /api/v1/scheduler/tasks/:id
pub async fn get_task_state(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<TaskStateResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let svc = get_scheduling_service(&state)?;
    let ts = svc
        .task_state(&id)
        .await
        .ok_or((StatusCode::NOT_FOUND, format!("task '{id}' not found")))?;
    Ok(Json(TaskStateResponse {
        task_id: id,
        state: ts.to_string(),
    }))
}

/// POST /api/v1/scheduler/tasks/:id/run
pub async fn mark_task_running(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<TaskStateResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let svc = get_scheduling_service(&state)?;
    svc.mark_running(&id)
        .await
        .map_err(|e| (StatusCode::CONFLICT, e.to_string()))?;
    Ok(Json(TaskStateResponse {
        task_id: id,
        state: "running".into(),
    }))
}

/// POST /api/v1/scheduler/tasks/:id/complete
pub async fn mark_task_completed(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let svc = get_scheduling_service(&state)?;
    let newly_ready = svc
        .mark_completed(&id)
        .await
        .map_err(|e| (StatusCode::CONFLICT, e.to_string()))?;
    Ok(Json(serde_json::json!({
        "task_id": id,
        "state": "completed",
        "newly_ready": newly_ready,
    })))
}

/// POST /api/v1/scheduler/tasks/:id/fail
pub async fn mark_task_failed(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let svc = get_scheduling_service(&state)?;
    let cancelled = svc
        .mark_failed(&id)
        .await
        .map_err(|e| (StatusCode::CONFLICT, e.to_string()))?;
    Ok(Json(serde_json::json!({
        "task_id": id,
        "state": "failed",
        "cancelled_downstream": cancelled,
    })))
}

// ── Stats ───────────────────────────────────────────────────────────

/// GET /api/v1/scheduler/stats
pub async fn scheduler_stats(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let svc = get_scheduling_service(&state)?;
    let stats = svc.stats().await;
    Ok(Json(serde_json::to_value(&stats).unwrap_or_default()))
}

// ── Helpers ─────────────────────────────────────────────────────────

fn get_scheduling_service(
    state: &AppState,
) -> Result<&hx_engine::scheduling::SchedulingService, (StatusCode, String)> {
    state.scheduling.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "scheduling service not initialized".into(),
    ))
}

fn preview_to_response(p: hx_scheduler::WorkflowPreview) -> PreviewResponse {
    // Compute derived fields before moving any data out.
    let feasible = p.feasible();
    let risk_level = p.risk_level().to_string();
    let task_count = p.task_predictions.len();

    PreviewResponse {
        name: p.name,
        task_count,
        waves: p.waves,
        wave_durations: p.wave_durations,
        critical_path: p.critical_path,
        total_predicted_duration: p.total_predicted_duration,
        total_predicted_cost: p.total_predicted_cost,
        avg_predicted_quality: p.avg_predicted_quality,
        feasible,
        risk_level,
        meets_deadline: p.meets_deadline,
        meets_budget: p.meets_budget,
        deadline_margin: p.deadline_margin,
        budget_margin: p.budget_margin,
        at_risk_tasks: p.at_risk_tasks,
        bottleneck_tasks: p.bottleneck_tasks,
        low_confidence_tasks: p.low_confidence_tasks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_response_serializes() {
        let resp = PreviewResponse {
            name: "test".into(),
            task_count: 3,
            waves: vec![vec!["a".into()], vec!["b".into(), "c".into()]],
            wave_durations: vec![10.0, 15.0],
            critical_path: vec!["a".into(), "b".into()],
            total_predicted_duration: 25.0,
            total_predicted_cost: 0.05,
            avg_predicted_quality: 0.75,
            feasible: true,
            risk_level: "low".into(),
            meets_deadline: Some(true),
            meets_budget: Some(true),
            deadline_margin: Some(100.0),
            budget_margin: Some(0.95),
            at_risk_tasks: vec![],
            bottleneck_tasks: vec!["a".into()],
            low_confidence_tasks: vec![],
        };
        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json["name"], "test");
        assert_eq!(json["feasible"], true);
        assert_eq!(json["task_count"], 3);
    }

    #[test]
    fn workflow_def_response_from() {
        let def = WorkflowDefinition {
            name: "test-wf".into(),
            tasks: Default::default(),
            task_order: vec!["a".into(), "b".into()],
            waves: vec![vec!["a".into()], vec!["b".into()]],
            deadline: Some(100.0),
            budget: None,
            default_gate: None,
            default_strategy: None,
            model_assignments: Default::default(),
            metadata: Default::default(),
        };
        let resp = WorkflowDefResponse::from(&def);
        assert_eq!(resp.name, "test-wf");
        assert_eq!(resp.task_count, 0);
        assert_eq!(resp.wave_count, 2);
    }
}
