use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::AccessPolicy;

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Request / Response DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SetPolicyRequest {
    pub secret_key: String,
    pub consumer: String,
    pub allowed: bool,
    #[serde(default)]
    pub scopes: Vec<String>,
    pub max_ttl_seconds: Option<i64>,
    pub expires_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub require_approval: bool,
}

#[derive(Serialize)]
pub struct PolicyResponse {
    pub id: Uuid,
    pub secret_key: String,
    pub consumer: String,
    pub allowed: bool,
    pub scopes: Vec<String>,
    pub max_ttl_seconds: Option<i64>,
    pub expires_at: Option<String>,
    pub require_approval: bool,
    pub created_at: String,
    pub updated_at: String,
}

impl From<AccessPolicy> for PolicyResponse {
    fn from(p: AccessPolicy) -> Self {
        Self {
            id: p.id,
            secret_key: p.secret_key,
            consumer: p.consumer,
            allowed: p.allowed,
            scopes: p.scopes,
            max_ttl_seconds: p.max_ttl_seconds,
            expires_at: p.expires_at.map(|t| t.to_rfc3339()),
            require_approval: p.require_approval,
            created_at: p.created_at.to_rfc3339(),
            updated_at: p.updated_at.to_rfc3339(),
        }
    }
}

#[derive(Deserialize)]
pub struct PolicyListQuery {
    pub secret_key: Option<String>,
    pub consumer: Option<String>,
}

#[derive(Serialize)]
pub struct PolicyMatrixResponse {
    pub secrets: Vec<String>,
    pub consumers: Vec<String>,
    pub matrix: HashMap<String, HashMap<String, bool>>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/policies
pub async fn set_policy(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetPolicyRequest>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    if req.secret_key.trim().is_empty() || req.consumer.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: "secret_key and consumer are required".into(),
            }),
        )
            .into_response();
    }

    let now = Utc::now();
    let policy = AccessPolicy {
        id: Uuid::now_v7(),
        secret_key: req.secret_key,
        consumer: req.consumer,
        allowed: req.allowed,
        scopes: req.scopes,
        max_ttl_seconds: req.max_ttl_seconds,
        expires_at: req.expires_at,
        require_approval: req.require_approval,
        created_at: now,
        updated_at: now,
    };

    match state.engine.set_policy(&policy).await {
        Ok(()) => Json(PolicyResponse::from(policy)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to set policy: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/policies
pub async fn list_policies(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(query): Query<PolicyListQuery>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (
            err.0,
            Json(ErrorBody { error: err.1 }),
        )
            .into_response();
    }

    match state
        .engine
        .list_policies(
            query.secret_key.as_deref(),
            query.consumer.as_deref(),
        )
        .await
    {
        Ok(policies) => {
            let responses: Vec<PolicyResponse> =
                policies.into_iter().map(PolicyResponse::from).collect();
            Json(responses).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to list policies: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/policies/matrix
pub async fn policy_matrix(
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

    match state.engine.list_policies(None, None).await {
        Ok(policies) => {
            let mut secrets_set = std::collections::BTreeSet::new();
            let mut consumers_set = std::collections::BTreeSet::new();
            let mut matrix: HashMap<String, HashMap<String, bool>> = HashMap::new();

            for policy in &policies {
                secrets_set.insert(policy.secret_key.clone());
                consumers_set.insert(policy.consumer.clone());

                matrix
                    .entry(policy.secret_key.clone())
                    .or_default()
                    .insert(policy.consumer.clone(), policy.is_effective());
            }

            Json(PolicyMatrixResponse {
                secrets: secrets_set.into_iter().collect(),
                consumers: consumers_set.into_iter().collect(),
                matrix,
            })
            .into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to build policy matrix: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/policies/my-access
pub async fn my_access(
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

    match state
        .engine
        .list_policies(None, Some(&consumer_name))
        .await
    {
        Ok(policies) => {
            let accessible: Vec<String> = policies
                .into_iter()
                .filter(|p| p.is_effective())
                .map(|p| p.secret_key)
                .collect();
            Json(accessible).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to list accessible secrets: {e}"),
            }),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/policies/:id
pub async fn delete_policy(
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

    match state.engine.delete_policy(id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: format!("policy {id} not found"),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to delete policy: {e}"),
            }),
        )
            .into_response(),
    }
}
