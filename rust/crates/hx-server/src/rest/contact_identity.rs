use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{ContactIdentity, IdentityType, TrustModel};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Serialize)]
pub struct ContactIdentityResponse {
    pub id: String,
    pub contact_id: String,
    pub identity_type: String,
    pub identity_value: String,
    pub verified: bool,
    pub verified_at: Option<String>,
    pub created_at: String,
}

#[derive(Deserialize)]
pub struct AddIdentityRequest {
    pub identity_type: String,
    pub identity_value: String,
}

#[derive(Serialize)]
pub struct TrustModelResponse {
    pub contact_id: String,
    pub can_query: bool,
    pub can_inject_context: bool,
    pub can_auto_reply: bool,
    pub allowed_namespaces: Vec<String>,
    pub max_confidence_override: Option<f64>,
    pub updated_at: String,
}

#[derive(Deserialize)]
pub struct SetTrustModelRequest {
    pub can_query: Option<bool>,
    pub can_inject_context: Option<bool>,
    pub can_auto_reply: Option<bool>,
    pub allowed_namespaces: Option<Vec<String>>,
    pub max_confidence_override: Option<f64>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// GET /api/v1/relay/contacts/:id/identities
pub async fn list_identities(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(contact_id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    match state.engine.list_contact_identities(contact_id).await {
        Ok(identities) => {
            let items: Vec<ContactIdentityResponse> = identities
                .into_iter()
                .map(identity_to_response)
                .collect();
            Json(items).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to list identities: {e}"),
            }),
        )
            .into_response(),
    }
}

/// POST /api/v1/relay/contacts/:id/identities
pub async fn add_identity(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(contact_id): Path<Uuid>,
    Json(req): Json<AddIdentityRequest>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    let identity_type: IdentityType = match req.identity_type.parse() {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody { error: e }),
            )
                .into_response()
        }
    };

    let identity = ContactIdentity {
        id: uuid::Uuid::now_v7(),
        contact_id,
        identity_type,
        identity_value: req.identity_value,
        verified: false,
        verified_at: None,
        created_at: chrono::Utc::now(),
    };

    match state.engine.add_contact_identity(&identity).await {
        Ok(()) => (StatusCode::CREATED, Json(identity_to_response(identity))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to add identity: {e}"),
            }),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/relay/contacts/identities/:id
pub async fn delete_identity(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    match state.engine.delete_contact_identity(id).await {
        Ok(true) => Json(serde_json::json!({"deleted": true})).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: "identity not found".into(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to delete identity: {e}"),
            }),
        )
            .into_response(),
    }
}

/// POST /api/v1/relay/contacts/identities/:id/verify
pub async fn verify_identity(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    match state.engine.verify_contact_identity(id).await {
        Ok(true) => Json(serde_json::json!({"verified": true})).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(ErrorBody {
                error: "identity not found or already verified".into(),
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to verify identity: {e}"),
            }),
        )
            .into_response(),
    }
}

/// GET /api/v1/relay/contacts/:id/trust
pub async fn get_trust_model(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(contact_id): Path<Uuid>,
) -> impl IntoResponse {
    if let Err(err) = authorize_read(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    match state.engine.get_trust_model(contact_id).await {
        Ok(Some(model)) => Json(trust_to_response(model)).into_response(),
        Ok(None) => {
            // Return defaults
            let model = TrustModel {
                contact_id,
                ..Default::default()
            };
            Json(trust_to_response(model)).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to get trust model: {e}"),
            }),
        )
            .into_response(),
    }
}

/// PUT /api/v1/relay/contacts/:id/trust
pub async fn set_trust_model(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(contact_id): Path<Uuid>,
    Json(req): Json<SetTrustModelRequest>,
) -> impl IntoResponse {
    if let Err(err) = authorize_write(&auth) {
        return (err.0, Json(ErrorBody { error: err.1 })).into_response();
    }

    // Get existing or create default
    let existing = state
        .engine
        .get_trust_model(contact_id)
        .await
        .unwrap_or(None)
        .unwrap_or(TrustModel {
            contact_id,
            ..Default::default()
        });

    let model = TrustModel {
        contact_id,
        can_query: req.can_query.unwrap_or(existing.can_query),
        can_inject_context: req.can_inject_context.unwrap_or(existing.can_inject_context),
        can_auto_reply: req.can_auto_reply.unwrap_or(existing.can_auto_reply),
        allowed_namespaces: req.allowed_namespaces.unwrap_or(existing.allowed_namespaces),
        max_confidence_override: req.max_confidence_override.or(existing.max_confidence_override),
        updated_at: chrono::Utc::now(),
    };

    match state.engine.set_trust_model(&model).await {
        Ok(()) => Json(trust_to_response(model)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorBody {
                error: format!("failed to set trust model: {e}"),
            }),
        )
            .into_response(),
    }
}

fn identity_to_response(i: ContactIdentity) -> ContactIdentityResponse {
    ContactIdentityResponse {
        id: i.id.to_string(),
        contact_id: i.contact_id.to_string(),
        identity_type: i.identity_type.as_str().to_string(),
        identity_value: i.identity_value,
        verified: i.verified,
        verified_at: i.verified_at.map(|dt| dt.to_rfc3339()),
        created_at: i.created_at.to_rfc3339(),
    }
}

fn trust_to_response(m: TrustModel) -> TrustModelResponse {
    TrustModelResponse {
        contact_id: m.contact_id.to_string(),
        can_query: m.can_query,
        can_inject_context: m.can_inject_context,
        can_auto_reply: m.can_auto_reply,
        allowed_namespaces: m.allowed_namespaces,
        max_confidence_override: m.max_confidence_override,
        updated_at: m.updated_at.to_rfc3339(),
    }
}
