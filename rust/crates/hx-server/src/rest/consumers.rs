use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::ConsumerProfileSummary;

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Request / Response DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateConsumerRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Serialize)]
pub struct CreateConsumerResponse {
    pub id: Uuid,
    pub name: String,
    pub token: String,
}

#[derive(Serialize)]
pub struct ConsumerSummaryResponse {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub revoked_at: Option<String>,
}

impl From<ConsumerProfileSummary> for ConsumerSummaryResponse {
    fn from(s: ConsumerProfileSummary) -> Self {
        Self {
            id: s.id,
            name: s.name,
            description: s.description,
            created_at: s.created_at.to_rfc3339(),
            last_used_at: s.last_used_at.map(|t| t.to_rfc3339()),
            revoked_at: s.revoked_at.map(|t| t.to_rfc3339()),
        }
    }
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/consumers
pub async fn create_consumer(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateConsumerRequest>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    let name = req.name.trim();
    if name.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: "name is required".into(),
            }),
        )
            .into_response();
    }

    match state
        .engine
        .create_consumer(name, req.description.as_deref())
        .await
    {
        Ok((profile, raw_token)) => Json(CreateConsumerResponse {
            id: profile.id,
            name: profile.name,
            token: raw_token,
        })
        .into_response(),
        Err(e) => {
            let status = if e.to_string().contains("already exists") {
                StatusCode::CONFLICT
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (
                status,
                Json(ErrorBody {
                    error: format!("failed to create consumer: {e}"),
                }),
            )
                .into_response()
        }
    }
}

/// GET /api/v1/consumers
pub async fn list_consumers(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    match state.engine.list_consumers().await {
        Ok(consumers) => {
            let summaries: Vec<ConsumerSummaryResponse> = consumers
                .into_iter()
                .map(|c| ConsumerSummaryResponse::from(ConsumerProfileSummary::from(c)))
                .collect();
            Json(summaries).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to list consumers: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/consumers/:id
pub async fn get_consumer(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    match state.engine.get_consumer(id).await {
        Ok(Some(consumer)) => {
            let summary = ConsumerSummaryResponse::from(ConsumerProfileSummary::from(consumer));
            Json(summary).into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: format!("consumer {id} not found"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to get consumer: {e}"),
            }),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/consumers/:id
pub async fn revoke_consumer(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    match state.engine.revoke_consumer(id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: format!("consumer {id} not found"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to revoke consumer: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/consumers/whoami
pub async fn whoami(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let consumer_name = match &auth.consumer_name {
        Some(name) => name.clone(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: "not authenticated as a consumer (use a consumer bearer token)".into(),
                }),
            )
                .into_response();
        }
    };

    match state.engine.list_consumers().await {
        Ok(consumers) => {
            match consumers.into_iter().find(|c| c.name == consumer_name) {
                Some(consumer) => {
                    let summary =
                        ConsumerSummaryResponse::from(ConsumerProfileSummary::from(consumer));
                    Json(summary).into_response()
                }
                None => (
                    StatusCode::NOT_FOUND,
                    Json(ErrorBody {
                        error: format!("consumer '{consumer_name}' not found"),
                    }),
                )
                    .into_response(),
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to lookup consumer: {e}"),
            }),
        )
            .into_response(),
    }
}
