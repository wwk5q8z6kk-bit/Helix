use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
    Json,
};
use serde::Serialize;

use crate::state::AppState;

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

fn ai_disabled() -> axum::response::Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorBody {
            error: "AI sidecar is disabled. Set ai_sidecar.enabled=true to enable.".to_string(),
        }),
    )
        .into_response()
}

/// Proxy GET /api/v1/ai/health → sidecar /health
pub async fn ai_health(State(state): State<Arc<AppState>>) -> axum::response::Response {
    let config = &state.engine.config.ai_sidecar;
    if !config.enabled {
        return ai_disabled();
    }
    let url = format!("{}/health", config.base_url.trim_end_matches('/'));
    proxy_get(&state.http_client, &url, config.timeout_secs).await
}

/// Proxy GET /api/v1/ai/models → sidecar /v1/models
pub async fn ai_models(State(state): State<Arc<AppState>>) -> axum::response::Response {
    let config = &state.engine.config.ai_sidecar;
    if !config.enabled {
        return ai_disabled();
    }
    let url = format!("{}/v1/models", config.base_url.trim_end_matches('/'));
    proxy_get(&state.http_client, &url, config.timeout_secs).await
}

/// Proxy POST /api/v1/ai/embeddings → sidecar /v1/embeddings
pub async fn ai_embeddings(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Body,
) -> axum::response::Response {
    let config = &state.engine.config.ai_sidecar;
    if !config.enabled {
        return ai_disabled();
    }
    let url = format!("{}/v1/embeddings", config.base_url.trim_end_matches('/'));
    proxy_post(&state.http_client, &url, config.timeout_secs, headers, body).await
}

/// Proxy POST /api/v1/ai/chat/completions → sidecar /v1/chat/completions
pub async fn ai_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: Body,
) -> axum::response::Response {
    let config = &state.engine.config.ai_sidecar;
    if !config.enabled {
        return ai_disabled();
    }
    let url = format!(
        "{}/v1/chat/completions",
        config.base_url.trim_end_matches('/')
    );
    proxy_post(&state.http_client, &url, config.timeout_secs, headers, body).await
}

async fn proxy_get(
    client: &reqwest::Client,
    url: &str,
    timeout_secs: u64,
) -> axum::response::Response {
    let result = client
        .get(url)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .send()
        .await;

    match result {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body = resp.text().await.unwrap_or_default();
            (
                status,
                [(
                    axum::http::header::CONTENT_TYPE,
                    "application/json".to_string(),
                )],
                body,
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(ErrorBody {
                error: format!(
                    "AI sidecar unreachable at {url}: {e}. \
                     Ensure the Python AI service is running (start with ~/.helix/start-ai.sh)."
                ),
            }),
        )
            .into_response(),
    }
}

async fn proxy_post(
    client: &reqwest::Client,
    url: &str,
    timeout_secs: u64,
    headers: HeaderMap,
    body: Body,
) -> axum::response::Response {
    // Read the request body
    let body_bytes = match axum::body::to_bytes(body, 10 * 1024 * 1024).await {
        Ok(bytes) => bytes,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ErrorBody {
                    error: format!("failed to read request body: {e}"),
                }),
            )
                .into_response()
        }
    };

    let mut req = client
        .post(url)
        .timeout(std::time::Duration::from_secs(timeout_secs))
        .body(body_bytes);

    // Forward content-type header if present
    if let Some(ct) = headers.get(axum::http::header::CONTENT_TYPE) {
        if let Ok(ct_str) = ct.to_str() {
            req = req.header("Content-Type", ct_str);
        }
    }

    match req.send().await {
        Ok(resp) => {
            let status =
                StatusCode::from_u16(resp.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            let body = resp.text().await.unwrap_or_default();
            (
                status,
                [(
                    axum::http::header::CONTENT_TYPE,
                    "application/json".to_string(),
                )],
                body,
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            Json(ErrorBody {
                error: format!(
                    "AI sidecar unreachable at {url}: {e}. \
                     Ensure the Python AI service is running (start with ~/.helix/start-ai.sh)."
                ),
            }),
        )
            .into_response(),
    }
}
