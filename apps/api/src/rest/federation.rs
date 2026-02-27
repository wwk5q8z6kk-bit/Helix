use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::UpdateProfileRequest;
use hx_engine::federation::{FederatedResult, FederationPeer};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct AddPeerDto {
    pub vault_id: String,
    pub display_name: String,
    pub endpoint: String,
    pub public_key: Option<String>,
    pub allowed_namespaces: Option<Vec<String>>,
    pub max_results: Option<usize>,
    pub shared_secret: Option<String>,
}

#[derive(Deserialize)]
pub struct HandshakeDto {
    pub endpoint: String,
    pub shared_secret: Option<String>,
}

#[derive(Deserialize)]
pub struct FederatedQueryDto {
    pub query: String,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct FederatedQueryResponse {
    pub results: Vec<FederatedResult>,
    pub peer_count: usize,
}

#[derive(Serialize)]
pub struct PeerHealthResponse {
    pub peer_id: String,
    pub healthy: bool,
}

#[derive(Serialize)]
pub struct FederationIdentityResponse {
    pub vault_id: String,
    pub display_name: String,
    pub public_key: Option<String>,
    pub vault_address: Option<String>,
    pub updated_at: String,
}

#[derive(Serialize)]
pub struct HandshakeResponse {
    pub id: String,
    pub status: String,
    pub vault_id: String,
    pub display_name: String,
    pub public_key: Option<String>,
}

// --- Handlers ---

/// List all federation peers.
pub async fn list_peers(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let peers = state.engine.federation.list_peers().await;
    Ok(Json(serde_json::json!({
        "peers": peers,
        "count": peers.len(),
    })))
}

/// Get local federation identity (owner profile + vault id).
pub async fn federation_identity(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let mut profile = state
        .engine
        .get_profile()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let mut vault_id = profile
        .metadata
        .get("vault_id")
        .and_then(|value| value.as_str())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    if vault_id.is_none() {
        let generated = Uuid::now_v7().to_string();
        let mut metadata = profile.metadata.clone();
        metadata.insert("vault_id".to_string(), serde_json::Value::String(generated.clone()));

        profile = state
            .engine
            .update_profile(&UpdateProfileRequest {
                metadata: Some(metadata),
                ..Default::default()
            })
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        vault_id = Some(generated);
    }

    let display_name = profile.display_name.trim();
    let display_name = if display_name.is_empty() {
        "Helix Owner"
    } else {
        display_name
    };

    let public_key = profile
        .signature_public_key
        .as_ref()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let vault_address = profile
        .email
        .as_ref()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(|value| format!("mailto:{value}"));

    Ok(Json(FederationIdentityResponse {
        vault_id: vault_id.unwrap_or_else(|| "unknown".into()),
        display_name: display_name.to_string(),
        public_key,
        vault_address,
        updated_at: profile.updated_at.to_rfc3339(),
    }))
}

/// Add a new federation peer (manual registration).
pub async fn add_peer(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(body): Json<AddPeerDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let mut peer = FederationPeer::new(body.vault_id, body.display_name, body.endpoint);

    if let Some(pk) = body.public_key {
        peer.public_key = Some(pk);
    }
    if let Some(ns) = body.allowed_namespaces {
        peer.allowed_namespaces = ns;
    }
    if let Some(max) = body.max_results {
        peer.max_results = max;
    }
    peer.shared_secret = body.shared_secret;

    let id = peer.id;
    state.engine.federation.add_peer(peer).await;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({ "id": id.to_string(), "status": "added" })),
    ))
}

/// Perform a federation handshake: auto-discover a peer via their identity endpoint.
///
/// Calls `{endpoint}/api/v1/federation/identity`, populates vault_id and
/// display_name from the response, and registers the peer. If `shared_secret`
/// is provided, future queries to this peer will be HMAC-signed.
pub async fn federation_handshake(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(body): Json<HandshakeDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let peer = state
        .engine
        .federation
        .handshake(&body.endpoint, body.shared_secret)
        .await
        .map_err(|e| {
            let status = if e.to_string().contains("already registered") {
                StatusCode::CONFLICT
            } else {
                StatusCode::BAD_GATEWAY
            };
            (status, e.to_string())
        })?;

    Ok((
        StatusCode::CREATED,
        Json(HandshakeResponse {
            id: peer.id.to_string(),
            status: "handshake_complete".into(),
            vault_id: peer.vault_id,
            display_name: peer.display_name,
            public_key: peer.public_key,
        }),
    ))
}

/// Remove a federation peer by ID.
pub async fn remove_peer(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid peer id: {e}")))?;

    let removed = state.engine.federation.remove_peer(uuid).await;
    if removed {
        Ok(Json(serde_json::json!({ "status": "removed" })))
    } else {
        Err((StatusCode::NOT_FOUND, "peer not found".into()))
    }
}

/// Execute a federated query across all enabled peers.
pub async fn federated_query(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(body): Json<FederatedQueryDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let limit = body.limit.unwrap_or(50);
    let results = state
        .engine
        .federation
        .federated_query(&body.query, limit)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let peer_count = state.engine.federation.list_peers().await.len();

    Ok(Json(FederatedQueryResponse {
        results,
        peer_count,
    }))
}

/// Health check a specific federation peer.
pub async fn peer_health(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid peer id: {e}")))?;

    let healthy = state
        .engine
        .federation
        .health_check(uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(PeerHealthResponse {
        peer_id: id,
        healthy,
    }))
}
