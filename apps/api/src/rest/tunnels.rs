//! REST handlers for tunnel management — register, start, stop, health check.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_engine::tunnel::{TunnelConfig, TunnelStatus, TunnelType};

use crate::auth::AuthContext;
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct RegisterTunnelDto {
    pub tunnel_type: String,
    pub name: String,
    pub local_port: u16,
    #[serde(default)]
    pub settings: std::collections::HashMap<String, String>,
}

#[derive(Serialize)]
pub struct TunnelConfigResponse {
    pub id: String,
    pub tunnel_type: String,
    pub name: String,
    pub local_port: u16,
}

impl From<&TunnelConfig> for TunnelConfigResponse {
    fn from(config: &TunnelConfig) -> Self {
        Self {
            id: config.id.to_string(),
            tunnel_type: config.tunnel_type.to_string(),
            name: config.name.clone(),
            local_port: config.local_port,
        }
    }
}

#[derive(Serialize)]
pub struct TunnelStatusResponse {
    pub tunnel_type: String,
    pub name: String,
    pub running: bool,
    pub public_url: Option<String>,
    pub error: Option<String>,
    pub started_at: Option<String>,
}

impl From<TunnelStatus> for TunnelStatusResponse {
    fn from(status: TunnelStatus) -> Self {
        Self {
            tunnel_type: status.tunnel_type.to_string(),
            name: status.name,
            running: status.running,
            public_url: status.public_url,
            error: status.error,
            started_at: status.started_at.map(|dt| dt.to_rfc3339()),
        }
    }
}

fn authorize_admin(auth: &AuthContext) -> Result<(), (StatusCode, String)> {
    if auth.is_admin() {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, "admin permission required".into()))
    }
}

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

// --- Handlers ---

/// POST /api/v1/tunnels — register and start a tunnel.
pub async fn register_tunnel(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterTunnelDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let tunnel_type: TunnelType = req
        .tunnel_type
        .parse()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, e))?;

    let mut config = TunnelConfig::new(tunnel_type, &req.name, req.local_port);
    for (k, v) in req.settings {
        config = config.with_setting(k, v);
    }

    let response = TunnelConfigResponse::from(&config);

    let tunnel: Arc<dyn hx_engine::tunnel::Tunnel> = match tunnel_type {
        TunnelType::Cloudflare => {
            Arc::new(hx_engine::tunnel::cloudflare::CloudflareTunnel::new(config.clone()))
        }
        TunnelType::Ngrok => {
            Arc::new(hx_engine::tunnel::ngrok::NgrokTunnel::new(config.clone()))
        }
        TunnelType::Tailscale => {
            Arc::new(hx_engine::tunnel::tailscale::TailscaleTunnel::new(config.clone()))
        }
        TunnelType::Bore => {
            Arc::new(hx_engine::tunnel::bore::BoreTunnel::new(config.clone()))
        }
        TunnelType::Ssh => {
            Arc::new(hx_engine::tunnel::ssh::SshTunnel::new(config.clone()).map_err(map_hx_error)?)
        }
        TunnelType::Custom => {
            Arc::new(hx_engine::tunnel::custom::CustomTunnel::new(config.clone()).map_err(map_hx_error)?)
        }
    };

    // Start the tunnel
    let public_url = tunnel.start().await.map_err(map_hx_error)?;

    state.engine.tunnels.register(config, tunnel).await;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": response.id,
            "tunnel_type": response.tunnel_type,
            "name": response.name,
            "local_port": response.local_port,
            "public_url": public_url,
        })),
    )
        .into_response())
}

/// GET /api/v1/tunnels — list all tunnels.
pub async fn list_tunnels(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<TunnelConfigResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let configs = state.engine.tunnels.list_configs().await;
    let response: Vec<TunnelConfigResponse> = configs.iter().map(TunnelConfigResponse::from).collect();
    Ok(Json(response))
}

/// GET /api/v1/tunnels/:id — get tunnel status.
pub async fn get_tunnel_status(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<TunnelStatusResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let tunnel = state
        .engine
        .tunnels
        .get(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "tunnel not found".to_string()))?;

    Ok(Json(TunnelStatusResponse::from(tunnel.status())))
}

/// DELETE /api/v1/tunnels/:id — stop and remove a tunnel.
pub async fn remove_tunnel(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    // Stop the tunnel before removing
    if let Some(tunnel) = state.engine.tunnels.get(uuid).await {
        let _ = tunnel.stop().await;
    }

    let removed = state.engine.tunnels.remove(uuid).await;
    if !removed {
        return Err((StatusCode::NOT_FOUND, "tunnel not found".to_string()));
    }

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// POST /api/v1/tunnels/:id/health — health check a tunnel.
pub async fn tunnel_health_check(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let tunnel = state
        .engine
        .tunnels
        .get(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "tunnel not found".to_string()))?;

    let healthy = tunnel.health_check().await.map_err(map_hx_error)?;

    Ok(Json(serde_json::json!({
        "tunnel_id": uuid.to_string(),
        "healthy": healthy,
    })))
}
