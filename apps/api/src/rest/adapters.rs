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

use hx_engine::adapters::{
    AdapterConfig, AdapterOutboundMessage, AdapterStatus, AdapterType,
};

use crate::auth::{authorize_write, AuthContext};
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct RegisterAdapterDto {
    pub adapter_type: String,
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub settings: HashMap<String, String>,
}

#[derive(Serialize)]
pub struct AdapterConfigResponse {
    pub id: String,
    pub adapter_type: String,
    pub name: String,
    pub enabled: bool,
    pub created_at: String,
    pub updated_at: Option<String>,
}

impl From<&AdapterConfig> for AdapterConfigResponse {
    fn from(config: &AdapterConfig) -> Self {
        Self {
            id: config.id.to_string(),
            adapter_type: config.adapter_type.to_string(),
            name: config.name.clone(),
            enabled: config.enabled,
            created_at: config.created_at.to_rfc3339(),
            updated_at: config.updated_at.map(|dt| dt.to_rfc3339()),
        }
    }
}

#[derive(Serialize)]
pub struct AdapterStatusResponse {
    pub adapter_type: String,
    pub name: String,
    pub connected: bool,
    pub last_send: Option<String>,
    pub last_receive: Option<String>,
    pub error: Option<String>,
}

impl From<AdapterStatus> for AdapterStatusResponse {
    fn from(status: AdapterStatus) -> Self {
        Self {
            adapter_type: status.adapter_type.to_string(),
            name: status.name,
            connected: status.connected,
            last_send: status.last_send.map(|dt| dt.to_rfc3339()),
            last_receive: status.last_receive.map(|dt| dt.to_rfc3339()),
            error: status.error,
        }
    }
}

#[derive(Deserialize)]
pub struct SendAdapterMessageDto {
    pub channel: String,
    pub content: String,
    pub thread_id: Option<String>,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
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

/// GET /api/v1/adapters
pub async fn list_adapters(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<AdapterConfigResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let configs = state.engine.adapters.list_configs().await;
    let response: Vec<AdapterConfigResponse> = configs.iter().map(AdapterConfigResponse::from).collect();
    Ok(Json(response))
}

/// POST /api/v1/adapters
pub async fn register_adapter(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterAdapterDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let adapter_type: AdapterType = req
        .adapter_type
        .parse()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, e))?;

    let mut config = AdapterConfig::new(adapter_type, &req.name);
    if let Some(enabled) = req.enabled {
        config.enabled = enabled;
    }
    config.settings = req.settings;

    let response = AdapterConfigResponse::from(&config);

    // Create the actual adapter based on type and register it
    let adapter: Arc<dyn hx_engine::adapters::ExternalAdapter> = match adapter_type {
        AdapterType::Slack => {
            Arc::new(hx_engine::adapters::slack::SlackAdapter::new(config.clone()).map_err(map_hx_error)?)
        }
        AdapterType::Discord => {
            Arc::new(hx_engine::adapters::discord::DiscordAdapter::new(config.clone()).map_err(map_hx_error)?)
        }
        AdapterType::Email => {
            Arc::new(hx_engine::adapters::email::EmailAdapter::new(config.clone()).map_err(map_hx_error)?)
        }
        AdapterType::Telegram => {
            Arc::new(hx_engine::adapters::telegram::TelegramAdapter::new(config.clone()).map_err(map_hx_error)?)
        }
        AdapterType::Matrix => {
            Arc::new(hx_engine::adapters::matrix::MatrixAdapter::new(config.clone()).map_err(map_hx_error)?)
        }
        AdapterType::Webhook => {
            Arc::new(hx_engine::adapters::webhook::WebhookAdapter::new(config.clone()).map_err(map_hx_error)?)
        }
        AdapterType::CliChat => {
            Arc::new(hx_engine::adapters::cli_chat::CliChatAdapter::new(config.clone()).map_err(map_hx_error)?)
        }
    };

    state.engine.adapters.register(config, adapter).await;

    Ok((StatusCode::CREATED, Json(response)).into_response())
}

/// GET /api/v1/adapters/{id}
pub async fn get_adapter_status(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<AdapterStatusResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let adapter = state
        .engine
        .adapters
        .get(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "adapter not found".to_string()))?;

    Ok(Json(AdapterStatusResponse::from(adapter.status())))
}

/// DELETE /api/v1/adapters/{id}
pub async fn remove_adapter(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let removed = state.engine.adapters.remove(uuid).await;
    if !removed {
        return Err((StatusCode::NOT_FOUND, "adapter not found".to_string()));
    }

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// POST /api/v1/adapters/{id}/send
pub async fn send_message(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<SendAdapterMessageDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let message = AdapterOutboundMessage {
        channel: req.channel,
        content: req.content,
        thread_id: req.thread_id,
        metadata: req.metadata,
    };

    state
        .engine
        .adapters
        .send(uuid, &message)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(serde_json::json!({ "status": "sent" })).into_response())
}

/// POST /api/v1/adapters/{id}/health
pub async fn health_check(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let adapter = state
        .engine
        .adapters
        .get(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "adapter not found".to_string()))?;

    let healthy = adapter.health_check().await.map_err(map_hx_error)?;

    Ok(Json(serde_json::json!({
        "adapter_id": uuid.to_string(),
        "healthy": healthy,
    })))
}

/// GET /api/v1/adapters/statuses
pub async fn list_statuses(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<AdapterStatusResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let statuses = state.engine.adapters.list_statuses().await;
    let response: Vec<AdapterStatusResponse> = statuses.into_iter().map(AdapterStatusResponse::from).collect();
    Ok(Json(response))
}
