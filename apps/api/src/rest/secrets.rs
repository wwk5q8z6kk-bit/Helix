use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};

use crate::state::AppState;

#[derive(Serialize)]
pub struct BackendStatusDto {
    pub name: String,
    pub available: bool,
    pub keys: Vec<String>,
}

#[derive(Serialize)]
pub struct SecretStatusResponse {
    pub backends: Vec<BackendStatusDto>,
}

#[derive(Deserialize)]
pub struct SetSecretRequest {
    pub key: String,
    pub value: String,
}

#[derive(Serialize)]
pub struct SetSecretResponse {
    pub key: String,
    pub stored_in: String,
}

#[derive(Serialize)]
pub struct DeleteSecretResponse {
    pub key: String,
    pub deleted_from: Vec<String>,
}

#[derive(Deserialize)]
pub struct UnlockRequest {
    pub password: String,
}

#[derive(Serialize)]
pub struct UnlockResponse {
    pub unlocked: bool,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

fn err_json(msg: impl ToString) -> Json<ErrorBody> {
    Json(ErrorBody {
        error: msg.to_string(),
    })
}

/// GET /api/v1/secrets/status
pub async fn secret_status(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let statuses = state.engine.credential_store.status();
    let backends: Vec<BackendStatusDto> = statuses
        .into_iter()
        .map(|s| BackendStatusDto {
            name: s.name,
            available: s.available,
            keys: s.keys,
        })
        .collect();
    Json(SecretStatusResponse { backends })
}

/// POST /api/v1/secrets
pub async fn set_secret(
    State(state): State<Arc<AppState>>,
    Json(req): Json<SetSecretRequest>,
) -> impl IntoResponse {
    if req.key.is_empty() || req.value.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            err_json("key and value are required"),
        )
            .into_response();
    }

    match state.engine.credential_store.set(&req.key, &req.value) {
        Ok(source) => Json(SetSecretResponse {
            key: req.key,
            stored_in: source.to_string(),
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to store secret: {e}")),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/secrets/:key
pub async fn delete_secret(
    State(state): State<Arc<AppState>>,
    Path(key): Path<String>,
) -> impl IntoResponse {
    match state.engine.credential_store.delete(&key) {
        Ok(sources) => Json(DeleteSecretResponse {
            key,
            deleted_from: sources.iter().map(|s| s.to_string()).collect(),
        })
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to delete secret: {e}")),
        )
            .into_response(),
    }
}

/// POST /api/v1/secrets/unlock
pub async fn unlock_encrypted_file(
    State(state): State<Arc<AppState>>,
    Json(req): Json<UnlockRequest>,
) -> impl IntoResponse {
    if req.password.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            err_json("password is required"),
        )
            .into_response();
    }

    match state.engine.credential_store.unlock_encrypted_file(&req.password) {
        Ok(true) => Json(UnlockResponse { unlocked: true }).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            err_json("no encrypted file backend configured"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::UNAUTHORIZED,
            err_json(format!("unlock failed: {e}")),
        )
            .into_response(),
    }
}
