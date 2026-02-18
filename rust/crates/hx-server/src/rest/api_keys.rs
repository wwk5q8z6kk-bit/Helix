//! REST handlers for API key CRUD operations.
//!
//! Extends the existing access key system with a developer-friendly API.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::auth::AuthContext;
use crate::state::AppState;

// --- DTOs ---

#[derive(Serialize)]
pub struct ApiKeyResponse {
    pub id: String,
    pub name: String,
    pub prefix: String,
    pub permissions: Vec<String>,
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub expires_at: Option<String>,
    pub revoked: bool,
}

#[derive(Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
    #[serde(default)]
    pub permissions: Vec<String>,
    pub expires_at: Option<String>,
}

#[derive(Serialize)]
pub struct CreateApiKeyResponse {
    pub key: ApiKeyResponse,
    pub secret: String,
}

#[derive(Deserialize)]
pub struct UpdateApiKeyRequest {
    pub name: Option<String>,
    pub permissions: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct RotateKeyResponse {
    pub key: ApiKeyResponse,
    pub new_secret: String,
}

// --- In-memory store (placeholder for real persistence) ---

#[derive(Clone)]
struct StoredApiKey {
    id: Uuid,
    name: String,
    secret_hash: String,
    permissions: Vec<String>,
    created_at: DateTime<Utc>,
    last_used_at: Option<DateTime<Utc>>,
    expires_at: Option<DateTime<Utc>>,
    revoked: bool,
}

impl StoredApiKey {
    fn prefix(&self) -> String {
        format!("hx_{}", &self.id.to_string()[..8])
    }
}

fn to_response(key: &StoredApiKey) -> ApiKeyResponse {
    ApiKeyResponse {
        id: key.id.to_string(),
        name: key.name.clone(),
        prefix: key.prefix(),
        permissions: key.permissions.clone(),
        created_at: key.created_at.to_rfc3339(),
        last_used_at: key.last_used_at.map(|dt| dt.to_rfc3339()),
        expires_at: key.expires_at.map(|dt| dt.to_rfc3339()),
        revoked: key.revoked,
    }
}

/// GET /api/v1/keys — list all API keys (secrets masked).
pub async fn list_api_keys(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    // Use the existing access key system from the engine
    match state.engine.list_access_keys().await {
        Ok(keys) => {
            let responses: Vec<ApiKeyResponse> = keys
                .iter()
                .map(|k| ApiKeyResponse {
                    id: k.id.to_string(),
                    name: k.name.clone().unwrap_or_default(),
                    prefix: format!("hx_{}", &k.id.to_string()[..8]),
                    permissions: Vec::new(),
                    created_at: k.created_at.to_rfc3339(),
                    last_used_at: k.last_used_at.map(|dt| dt.to_rfc3339()),
                    expires_at: k.expires_at.map(|dt| dt.to_rfc3339()),
                    revoked: k.revoked_at.is_some(),
                })
                .collect();
            Json(responses).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to list keys: {e}"),
        )
            .into_response(),
    }
}

/// POST /api/v1/keys — create a new API key.
pub async fn create_api_key(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    if req.name.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "name is required".to_string()).into_response();
    }

    // Delegate to existing template-based key creation
    // For the developer API, create with default template if available
    let templates = match state.engine.list_permission_templates(100, 0).await {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to list templates: {e}"),
            )
                .into_response()
        }
    };

    let template = match templates.first() {
        Some(t) => t,
        None => {
            return (
                StatusCode::PRECONDITION_FAILED,
                "no permission template exists; create one first".to_string(),
            )
                .into_response()
        }
    };

    let expires = req
        .expires_at
        .as_deref()
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.with_timezone(&Utc));

    match state
        .engine
        .create_access_key(template.id, Some(req.name.clone()), expires)
        .await
    {
        Ok((key, token)) => {
            let response = CreateApiKeyResponse {
                key: ApiKeyResponse {
                    id: key.id.to_string(),
                    name: key.name.clone().unwrap_or_default(),
                    prefix: format!("hx_{}", &key.id.to_string()[..8]),
                    permissions: req.permissions,
                    created_at: key.created_at.to_rfc3339(),
                    last_used_at: None,
                    expires_at: key.expires_at.map(|dt| dt.to_rfc3339()),
                    revoked: false,
                },
                secret: token,
            };
            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to create key: {e}"),
        )
            .into_response(),
    }
}

/// GET /api/v1/keys/:id — get key details (secret masked).
pub async fn get_api_key(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let key_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid key id".to_string()).into_response(),
    };

    match state.engine.list_access_keys().await {
        Ok(keys) => {
            if let Some(key) = keys.iter().find(|k| k.id == key_id) {
                Json(ApiKeyResponse {
                    id: key.id.to_string(),
                    name: key.name.clone().unwrap_or_default(),
                    prefix: format!("hx_{}", &key.id.to_string()[..8]),
                    permissions: Vec::new(),
                    created_at: key.created_at.to_rfc3339(),
                    last_used_at: key.last_used_at.map(|dt| dt.to_rfc3339()),
                    expires_at: key.expires_at.map(|dt| dt.to_rfc3339()),
                    revoked: key.revoked_at.is_some(),
                })
                .into_response()
            } else {
                (StatusCode::NOT_FOUND, "key not found".to_string()).into_response()
            }
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to fetch key: {e}"),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/keys/:id — revoke an API key.
pub async fn revoke_api_key(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let key_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid key id".to_string()).into_response(),
    };

    match state.engine.revoke_access_key(key_id).await {
        Ok(_) => StatusCode::NO_CONTENT.into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("failed to revoke key: {e}"),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_key_response_serializes() {
        let resp = ApiKeyResponse {
            id: "test-id".into(),
            name: "test".into(),
            prefix: "hx_test1234".into(),
            permissions: vec!["read".into()],
            created_at: "2026-01-01T00:00:00Z".into(),
            last_used_at: None,
            expires_at: None,
            revoked: false,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("hx_test1234"));
        assert!(json.contains("\"revoked\":false"));
    }

    #[test]
    fn create_request_deserializes() {
        let json = r#"{"name":"my-key","permissions":["read","write"]}"#;
        let req: CreateApiKeyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name, "my-key");
        assert_eq!(req.permissions, vec!["read", "write"]);
        assert!(req.expires_at.is_none());
    }

    #[test]
    fn create_request_deserializes_with_expiry() {
        let json = r#"{"name":"temp-key","permissions":[],"expires_at":"2026-12-31T23:59:59Z"}"#;
        let req: CreateApiKeyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.expires_at.as_deref(), Some("2026-12-31T23:59:59Z"));
    }

    #[test]
    fn update_request_deserializes_partial() {
        let json = r#"{"name":"renamed"}"#;
        let req: UpdateApiKeyRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.name.as_deref(), Some("renamed"));
        assert!(req.permissions.is_none());
    }
}
