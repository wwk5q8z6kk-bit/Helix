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

use hx_engine::sources::{SourceConfig, SourceConnector, SourceStatus, SourceType};

use crate::auth::AuthContext;
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct RegisterSourceRequest {
    pub source_type: String,
    pub name: String,
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub settings: HashMap<String, String>,
    #[serde(default)]
    pub poll_interval_secs: Option<u64>,
}

#[derive(Serialize)]
pub struct SourceResponse {
    pub id: String,
    pub source_type: String,
    pub name: String,
    pub enabled: bool,
    pub poll_interval_secs: u64,
    pub created_at: String,
}

impl From<&SourceConfig> for SourceResponse {
    fn from(config: &SourceConfig) -> Self {
        Self {
            id: config.id.to_string(),
            source_type: config.source_type.to_string(),
            name: config.name.clone(),
            enabled: config.enabled,
            poll_interval_secs: config.poll_interval_secs,
            created_at: config.created_at.to_rfc3339(),
        }
    }
}

#[derive(Serialize)]
pub struct SourceStatusResponse {
    pub connected: bool,
    pub last_poll: Option<String>,
    pub documents_fetched: u64,
    pub errors: u64,
    pub message: Option<String>,
}

impl From<SourceStatus> for SourceStatusResponse {
    fn from(status: SourceStatus) -> Self {
        Self {
            connected: status.connected,
            last_poll: status.last_poll.map(|dt| dt.to_rfc3339()),
            documents_fetched: status.documents_fetched,
            errors: status.errors,
            message: status.message,
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

/// Build a concrete SourceConnector from the source type and config.
fn build_connector(
    source_type: SourceType,
    config: &SourceConfig,
) -> Result<Arc<dyn SourceConnector>, (StatusCode, String)> {
    match source_type {
        SourceType::DirectoryWatch => {
            let c = hx_engine::sources::directory::DirectoryWatcher::new(config.clone())
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("directory watcher: {e}")))?;
            Ok(Arc::new(c))
        }
        SourceType::RssFeed => {
            let c = hx_engine::sources::rss::RssFeedConnector::new(config.clone())
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("rss feed: {e}")))?;
            Ok(Arc::new(c))
        }
        SourceType::GitHubIssues => {
            let c = hx_engine::sources::github::GitHubConnector::new(config.clone())
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("github connector: {e}")))?;
            Ok(Arc::new(c))
        }
        SourceType::UrlScraper => {
            let c = hx_engine::sources::url_scraper::UrlScraperConnector::new(config.clone())
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("url scraper: {e}")))?;
            Ok(Arc::new(c))
        }
        SourceType::Custom => Err((
            StatusCode::BAD_REQUEST,
            "custom source type requires programmatic registration".into(),
        )),
    }
}

// --- Handlers ---

/// GET /api/v1/sources
pub async fn list_sources(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<SourceResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let configs = state.source_registry.list_configs().await;
    let response: Vec<SourceResponse> = configs.iter().map(SourceResponse::from).collect();
    Ok(Json(response))
}

/// POST /api/v1/sources
pub async fn register_source(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterSourceRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let source_type: SourceType = req
        .source_type
        .parse()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, e))?;

    let mut config = SourceConfig::new(source_type, &req.name);
    if let Some(enabled) = req.enabled {
        config.enabled = enabled;
    }
    if let Some(interval) = req.poll_interval_secs {
        config.poll_interval_secs = interval;
    }
    for (k, v) in &req.settings {
        config.settings.insert(k.clone(), v.clone());
    }

    let connector = build_connector(source_type, &config)?;
    let resp = SourceResponse::from(&config);
    state.source_registry.register(config, connector).await;

    Ok((StatusCode::CREATED, Json(resp)).into_response())
}

/// GET /api/v1/sources/:id
pub async fn get_source_status(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<SourceStatusResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let status = state
        .source_registry
        .status(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "source not found".to_string()))?;

    Ok(Json(SourceStatusResponse::from(status)))
}

/// DELETE /api/v1/sources/:id
pub async fn remove_source(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let removed = state.source_registry.remove(uuid).await;
    if !removed {
        return Ok((StatusCode::NOT_FOUND, "source not found".to_string()).into_response());
    }

    Ok(StatusCode::NO_CONTENT.into_response())
}

/// POST /api/v1/sources/:id/poll
pub async fn poll_source(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let uuid = Uuid::parse_str(&id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let docs = state
        .source_registry
        .poll(uuid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({ "documents_fetched": docs.len() })))
}
