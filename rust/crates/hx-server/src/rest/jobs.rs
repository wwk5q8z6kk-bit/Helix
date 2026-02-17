use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_engine::jobs::{JobStats, JobStatus};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// --- Query Parameters ---

#[derive(Deserialize)]
pub struct ListJobsQuery {
    pub status: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Deserialize)]
pub struct PurgeQuery {
    pub older_than_days: Option<u32>,
}

// --- Response DTOs ---

#[derive(Serialize)]
pub struct JobResponse {
    pub id: String,
    pub job_type: String,
    pub payload: serde_json::Value,
    pub status: String,
    pub priority: i32,
    pub retries: u32,
    pub max_retries: u32,
    pub error: Option<String>,
    pub created_at: String,
    pub scheduled_at: Option<String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
    pub next_retry_at: Option<String>,
    pub idempotency_key: Option<String>,
}

impl From<hx_engine::jobs::Job> for JobResponse {
    fn from(job: hx_engine::jobs::Job) -> Self {
        Self {
            id: job.id.to_string(),
            job_type: job.job_type,
            payload: job.payload,
            status: job.status.to_string(),
            priority: job.priority,
            retries: job.retries,
            max_retries: job.max_retries,
            error: job.error,
            created_at: job.created_at.to_rfc3339(),
            scheduled_at: job.scheduled_at.map(|dt| dt.to_rfc3339()),
            started_at: job.started_at.map(|dt| dt.to_rfc3339()),
            completed_at: job.completed_at.map(|dt| dt.to_rfc3339()),
            next_retry_at: job.next_retry_at.map(|dt| dt.to_rfc3339()),
            idempotency_key: job.idempotency_key,
        }
    }
}

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

// --- Helpers ---

fn require_job_queue(state: &AppState) -> Result<&Arc<hx_engine::jobs::queue::JobQueue>, (StatusCode, String)> {
    state
        .job_queue
        .as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "job queue not initialized".into()))
}

// --- Handlers ---

/// GET /api/v1/jobs — list jobs, optionally filtered by status.
pub async fn list_jobs(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListJobsQuery>,
) -> Result<Json<Vec<JobResponse>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let queue = require_job_queue(&state)?;
    let status = match query.status.as_deref() {
        Some(s) => Some(
            s.parse::<JobStatus>()
                .map_err(|e| (StatusCode::BAD_REQUEST, e))?,
        ),
        None => None,
    };
    let limit = query.limit.unwrap_or(100);

    let jobs = queue
        .list(status, limit)
        .await
        .map_err(map_hx_error)?;
    Ok(Json(jobs.into_iter().map(JobResponse::from).collect()))
}

/// GET /api/v1/jobs/stats — job statistics.
pub async fn job_stats(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<JobStats>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let queue = require_job_queue(&state)?;
    let stats = queue.stats().await.map_err(map_hx_error)?;
    Ok(Json(stats))
}

/// GET /api/v1/jobs/:id — get job detail.
pub async fn get_job(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<JobResponse>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let queue = require_job_queue(&state)?;
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let job = queue
        .get(uuid)
        .await
        .map_err(map_hx_error)?
        .ok_or((StatusCode::NOT_FOUND, "job not found".to_string()))?;

    Ok(Json(JobResponse::from(job)))
}

/// POST /api/v1/jobs/:id/retry — retry a failed or dead job.
pub async fn retry_job(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_write(&auth)?;

    let queue = require_job_queue(&state)?;
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    queue.retry(uuid).await.map_err(map_hx_error)?;
    Ok(Json(serde_json::json!({ "status": "retried" })))
}

/// POST /api/v1/jobs/:id/cancel — cancel a pending or running job.
pub async fn cancel_job(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_write(&auth)?;

    let queue = require_job_queue(&state)?;
    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    queue.cancel(uuid).await.map_err(map_hx_error)?;
    Ok(Json(serde_json::json!({ "status": "cancelled" })))
}

/// GET /api/v1/jobs/dead-letter — list dead letter queue.
pub async fn dead_letter_queue(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListJobsQuery>,
) -> Result<Json<Vec<JobResponse>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let queue = require_job_queue(&state)?;
    let limit = query.limit.unwrap_or(100);

    let jobs = queue
        .dead_letter_queue(limit)
        .await
        .map_err(map_hx_error)?;
    Ok(Json(jobs.into_iter().map(JobResponse::from).collect()))
}

/// POST /api/v1/jobs/purge — purge completed jobs older than N days.
pub async fn purge_jobs(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(query): Query<PurgeQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_write(&auth)?;

    let queue = require_job_queue(&state)?;
    let older_than_days = query.older_than_days.unwrap_or(30);

    let purged = queue
        .purge_completed(older_than_days)
        .await
        .map_err(map_hx_error)?;
    Ok(Json(serde_json::json!({ "purged": purged })))
}
