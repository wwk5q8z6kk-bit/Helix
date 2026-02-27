use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_engine::workflow::{ExecutionStatus, Workflow, WorkflowExecution, WorkflowStep, WorkflowTrigger};
use hx_engine::workflow::executor::WorkflowExecutor;

use crate::auth::AuthContext;
use crate::state::AppState;

// --- DTOs ---

#[derive(Serialize)]
pub struct WorkflowResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub trigger_type: String,
    pub step_count: usize,
    pub timeout_secs: u64,
    pub created_at: String,
}

impl From<&Workflow> for WorkflowResponse {
    fn from(wf: &Workflow) -> Self {
        let trigger_type = match &wf.trigger {
            WorkflowTrigger::Manual => "manual".to_string(),
            WorkflowTrigger::Cron { expression } => format!("cron:{expression}"),
            WorkflowTrigger::OnEvent { event_type } => format!("on_event:{event_type}"),
            WorkflowTrigger::OnIngest { namespace } => format!("on_ingest:{namespace}"),
        };
        Self {
            id: wf.id.to_string(),
            name: wf.name.clone(),
            description: wf.description.clone(),
            trigger_type,
            step_count: wf.steps.len(),
            timeout_secs: wf.timeout_secs,
            created_at: wf.created_at.to_rfc3339(),
        }
    }
}

#[derive(Serialize)]
pub struct ExecutionResponse {
    pub id: String,
    pub workflow_id: String,
    pub workflow_name: String,
    pub status: String,
    pub step_count: usize,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub error: Option<String>,
}

impl From<&WorkflowExecution> for ExecutionResponse {
    fn from(exec: &WorkflowExecution) -> Self {
        Self {
            id: exec.id.to_string(),
            workflow_id: exec.workflow_id.to_string(),
            workflow_name: exec.workflow_name.clone(),
            status: serde_json::to_value(exec.status)
                .ok()
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| format!("{:?}", exec.status)),
            step_count: exec.step_results.len(),
            started_at: exec.started_at.to_rfc3339(),
            completed_at: exec.completed_at.map(|dt| dt.to_rfc3339()),
            error: exec.error.clone(),
        }
    }
}

#[derive(Deserialize)]
pub struct ExecuteWorkflowDto {
    #[serde(default)]
    pub variables: HashMap<String, serde_json::Value>,
}

fn authorize_admin(auth: &AuthContext) -> Result<(), (StatusCode, String)> {
    if auth.is_admin() {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, "admin permission required".into()))
    }
}

// --- Handlers ---

/// GET /api/v1/workflows
pub async fn list_workflows(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<WorkflowResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let executor = get_executor(&state)?;
    let workflows = executor.list_workflows().await;
    let response: Vec<WorkflowResponse> = workflows.iter().map(WorkflowResponse::from).collect();
    Ok(Json(response))
}

/// GET /api/v1/workflows/:id
pub async fn get_workflow(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<WorkflowResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = parse_uuid(&id)?;
    let executor = get_executor(&state)?;
    let wf = executor
        .get_workflow(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "workflow not found".to_string()))?;

    Ok(Json(WorkflowResponse::from(&wf)))
}

/// POST /api/v1/workflows/:id/execute
pub async fn execute_workflow(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<ExecuteWorkflowDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = parse_uuid(&id)?;
    let executor = get_executor(&state)?;

    let execution = executor
        .execute(uuid, req.variables)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(ExecutionResponse::from(&execution))).into_response())
}

/// GET /api/v1/workflows/executions
pub async fn list_executions(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<ExecutionResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let executor = get_executor(&state)?;
    let execs = executor.list_executions().await;
    let response: Vec<ExecutionResponse> = execs.iter().map(ExecutionResponse::from).collect();
    Ok(Json(response))
}

/// GET /api/v1/workflows/executions/:id
pub async fn get_execution(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ExecutionResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = parse_uuid(&id)?;
    let executor = get_executor(&state)?;
    let exec = executor
        .get_execution(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "execution not found".to_string()))?;

    Ok(Json(ExecutionResponse::from(&exec)))
}

/// POST /api/v1/workflows/executions/:id/cancel
pub async fn cancel_execution(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = parse_uuid(&id)?;
    let executor = get_executor(&state)?;

    executor
        .cancel(uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({ "status": "cancelled" })).into_response())
}

// --- Helpers ---

fn parse_uuid(s: &str) -> Result<Uuid, (StatusCode, String)> {
    Uuid::parse_str(s).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))
}

fn get_executor(state: &AppState) -> Result<Arc<WorkflowExecutor>, (StatusCode, String)> {
    Ok(Arc::clone(&state.workflow_executor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_response_from_manual_trigger() {
        let wf = Workflow::new("test", WorkflowTrigger::Manual);
        let resp = WorkflowResponse::from(&wf);
        assert_eq!(resp.name, "test");
        assert_eq!(resp.trigger_type, "manual");
        assert_eq!(resp.step_count, 0);
    }

    #[test]
    fn workflow_response_from_cron_trigger() {
        let wf = Workflow::new(
            "cron-wf",
            WorkflowTrigger::Cron {
                expression: "0 * * * *".into(),
            },
        );
        let resp = WorkflowResponse::from(&wf);
        assert_eq!(resp.trigger_type, "cron:0 * * * *");
    }

    #[test]
    fn execution_response_fields() {
        let wf = Workflow::new("exec-test", WorkflowTrigger::Manual);
        let exec = WorkflowExecution::new(&wf);
        let resp = ExecutionResponse::from(&exec);
        assert_eq!(resp.workflow_name, "exec-test");
        assert_eq!(resp.status, "pending");
        assert!(resp.completed_at.is_none());
    }

    #[test]
    fn parse_uuid_valid() {
        let uuid = Uuid::now_v7();
        assert!(parse_uuid(&uuid.to_string()).is_ok());
    }

    #[test]
    fn parse_uuid_invalid() {
        assert!(parse_uuid("not-a-uuid").is_err());
    }
}
