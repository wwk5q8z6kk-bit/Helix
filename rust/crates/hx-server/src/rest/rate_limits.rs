//! REST handlers for rate limit configuration.
//!
//! NOTE: Requires `rate_limiter: Arc<RateLimiter>` field on HelixEngine
//! (to be wired by team lead in task #6).

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_engine::rate_limit::RateLimitConfig;

use crate::auth::AuthContext;
use crate::state::AppState;

// --- DTOs ---

#[derive(Serialize)]
pub struct RateLimitEntry {
    pub adapter_id: String,
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub burst_size: u32,
}

#[derive(Deserialize)]
pub struct UpdateRateLimitRequest {
    pub adapter_id: String,
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub burst_size: u32,
}

/// GET /api/v1/rate-limits — list all configured rate limits.
pub async fn list_rate_limits(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let limits = state.rate_limiter.list_limits().await;
    let entries: Vec<RateLimitEntry> = limits
        .into_iter()
        .map(|(id, config)| RateLimitEntry {
            adapter_id: id.to_string(),
            requests_per_minute: config.requests_per_minute,
            requests_per_hour: config.requests_per_hour,
            burst_size: config.burst_size,
        })
        .collect();
    Json(entries).into_response()
}

/// PUT /api/v1/rate-limits — update rate limit for an adapter.
pub async fn update_rate_limit(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<UpdateRateLimitRequest>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let adapter_id = match Uuid::parse_str(&req.adapter_id) {
        Ok(id) => id,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "invalid adapter_id".to_string()).into_response()
        }
    };

    if req.requests_per_minute == 0 || req.requests_per_hour == 0 {
        return (
            StatusCode::BAD_REQUEST,
            "rates must be greater than zero".to_string(),
        )
            .into_response();
    }

    let config = RateLimitConfig {
        requests_per_minute: req.requests_per_minute,
        requests_per_hour: req.requests_per_hour,
        burst_size: req.burst_size,
    };

    state.rate_limiter.set_limit(adapter_id, config).await;

    StatusCode::NO_CONTENT.into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_limit_entry_serializes() {
        let entry = RateLimitEntry {
            adapter_id: "test-id".into(),
            requests_per_minute: 60,
            requests_per_hour: 1000,
            burst_size: 10,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"requests_per_minute\":60"));
    }

    #[test]
    fn update_request_deserializes() {
        let json = r#"{"adapter_id":"abc-123","requests_per_minute":30,"requests_per_hour":500,"burst_size":5}"#;
        let req: UpdateRateLimitRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.adapter_id, "abc-123");
        assert_eq!(req.requests_per_minute, 30);
        assert_eq!(req.requests_per_hour, 500);
        assert_eq!(req.burst_size, 5);
    }
}
