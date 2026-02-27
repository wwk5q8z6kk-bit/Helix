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

use hx_core::McpConnector;

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct CreateMcpConnectorRequest {
    pub name: String,
    pub description: Option<String>,
    pub publisher: Option<String>,
    pub version: String,
    pub homepage_url: Option<String>,
    pub repository_url: Option<String>,
    pub config_schema: Option<serde_json::Value>,
    pub capabilities: Option<Vec<String>>,
    pub verified: Option<bool>,
}

#[derive(Deserialize)]
pub struct UpdateMcpConnectorRequest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub publisher: Option<String>,
    pub version: Option<String>,
    pub homepage_url: Option<String>,
    pub repository_url: Option<String>,
    pub config_schema: Option<serde_json::Value>,
    pub capabilities: Option<Vec<String>>,
    pub verified: Option<bool>,
}

#[derive(Deserialize)]
pub struct ListMcpConnectorParams {
    pub publisher: Option<String>,
    pub verified: Option<bool>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Serialize)]
pub struct McpConnectorResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub publisher: Option<String>,
    pub version: String,
    pub homepage_url: Option<String>,
    pub repository_url: Option<String>,
    pub config_schema: serde_json::Value,
    pub capabilities: Vec<String>,
    pub verified: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
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

fn connector_to_response(connector: McpConnector) -> McpConnectorResponse {
    McpConnectorResponse {
        id: connector.id.to_string(),
        name: connector.name,
        description: connector.description,
        publisher: connector.publisher,
        version: connector.version,
        homepage_url: connector.homepage_url,
        repository_url: connector.repository_url,
        config_schema: connector.config_schema,
        capabilities: connector.capabilities,
        verified: connector.verified,
        created_at: connector.created_at,
        updated_at: connector.updated_at,
    }
}

/// GET /api/v1/mcp/connectors
pub async fn list_mcp_connectors(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<ListMcpConnectorParams>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_read(&auth) {
        return (status, err_json(message)).into_response();
    }

    let limit = params.limit.unwrap_or(200).min(500);
    let offset = params.offset.unwrap_or(0);
    let publisher = params.publisher.as_deref();
    let verified = params.verified;

    match state
        .engine
        .list_mcp_connectors(publisher, verified, limit, offset)
        .await
    {
        Ok(connectors) => Json(
            connectors
                .into_iter()
                .map(connector_to_response)
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to list connectors: {err}")),
        )
            .into_response(),
    }
}

/// GET /api/v1/mcp/connectors/:id
pub async fn get_mcp_connector(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_read(&auth) {
        return (status, err_json(message)).into_response();
    }

    let connector_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid connector id"),
            )
                .into_response()
        }
    };

    match state.engine.get_mcp_connector(connector_id).await {
        Ok(Some(connector)) => Json(connector_to_response(connector)).into_response(),
        Ok(None) => (StatusCode::NOT_FOUND, err_json("connector not found")).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to load connector: {err}")),
        )
            .into_response(),
    }
}

/// POST /api/v1/mcp/connectors
pub async fn create_mcp_connector(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(req): Json<CreateMcpConnectorRequest>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, err_json("admin role required")).into_response();
    }

    if req.name.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            err_json("name cannot be empty"),
        )
            .into_response();
    }
    if req.version.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            err_json("version cannot be empty"),
        )
            .into_response();
    }

    let config_schema = req.config_schema.unwrap_or_else(|| serde_json::json!({}));
    let capabilities = req.capabilities.unwrap_or_default();
    let verified = req.verified.unwrap_or(false);

    match state
        .engine
        .create_mcp_connector(
            req.name,
            req.description,
            req.publisher,
            req.version,
            req.homepage_url,
            req.repository_url,
            config_schema,
            capabilities,
            verified,
        )
        .await
    {
        Ok(connector) => Json(connector_to_response(connector)).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to create connector: {err}")),
        )
            .into_response(),
    }
}

/// PUT /api/v1/mcp/connectors/:id
pub async fn update_mcp_connector(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(req): Json<UpdateMcpConnectorRequest>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, err_json("admin role required")).into_response();
    }

    let connector_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid connector id"),
            )
                .into_response()
        }
    };

    let mut connector = match state.engine.get_mcp_connector(connector_id).await {
        Ok(Some(connector)) => connector,
        Ok(None) => return (StatusCode::NOT_FOUND, err_json("connector not found")).into_response(),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load connector: {err}")),
            )
                .into_response()
        }
    };

    if let Some(name) = req.name {
        if name.trim().is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                err_json("name cannot be empty"),
            )
                .into_response();
        }
        connector.name = name;
    }
    if let Some(description) = req.description {
        connector.description = Some(description);
    }
    if let Some(publisher) = req.publisher {
        connector.publisher = Some(publisher);
    }
    if let Some(version) = req.version {
        if version.trim().is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                err_json("version cannot be empty"),
            )
                .into_response();
        }
        connector.version = version;
    }
    if let Some(homepage_url) = req.homepage_url {
        connector.homepage_url = Some(homepage_url);
    }
    if let Some(repository_url) = req.repository_url {
        connector.repository_url = Some(repository_url);
    }
    if let Some(schema) = req.config_schema {
        connector.config_schema = schema;
    }
    if let Some(capabilities) = req.capabilities {
        connector.capabilities = capabilities;
    }
    if let Some(verified) = req.verified {
        connector.verified = verified;
    }

    connector.updated_at = Utc::now();

    match state.engine.update_mcp_connector(connector.clone()).await {
        Ok(true) => Json(connector_to_response(connector)).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, err_json("connector not found")).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to update connector: {err}")),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/mcp/connectors/:id
pub async fn delete_mcp_connector(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, err_json("admin role required")).into_response();
    }

    let connector_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid connector id"),
            )
                .into_response()
        }
    };

    match state.engine.delete_mcp_connector(connector_id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, err_json("connector not found")).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to delete connector: {err}")),
        )
            .into_response(),
    }
}
