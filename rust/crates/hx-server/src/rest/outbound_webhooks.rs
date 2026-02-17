//! REST handlers for outbound webhook management with HMAC signing.
//!
//! NOTE: Requires `hmac` and `sha2` workspace deps in hx-server's Cargo.toml
//! (to be wired by team lead in task #6).

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use uuid::Uuid;

use crate::auth::AuthContext;
use crate::state::AppState;

type HmacSha256 = Hmac<Sha256>;

// --- DTOs ---

#[derive(Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub backoff_secs: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            backoff_secs: 10,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct OutboundWebhook {
    pub id: Uuid,
    pub url: String,
    pub events: Vec<String>,
    #[serde(skip_serializing)]
    pub secret: Option<String>,
    pub active: bool,
    pub created_at: DateTime<Utc>,
    pub retry_config: RetryConfig,
}

#[derive(Serialize)]
pub struct OutboundWebhookResponse {
    pub id: String,
    pub url: String,
    pub events: Vec<String>,
    pub has_secret: bool,
    pub active: bool,
    pub created_at: String,
    pub retry_config: RetryConfig,
}

impl From<&OutboundWebhook> for OutboundWebhookResponse {
    fn from(wh: &OutboundWebhook) -> Self {
        Self {
            id: wh.id.to_string(),
            url: wh.url.clone(),
            events: wh.events.clone(),
            has_secret: wh.secret.is_some(),
            active: wh.active,
            created_at: wh.created_at.to_rfc3339(),
            retry_config: wh.retry_config.clone(),
        }
    }
}

#[derive(Deserialize)]
pub struct CreateWebhookRequest {
    pub url: String,
    pub events: Vec<String>,
    pub secret: Option<String>,
    #[serde(default)]
    pub retry_config: Option<RetryConfig>,
}

#[derive(Deserialize)]
pub struct UpdateWebhookRequest {
    pub url: Option<String>,
    pub events: Option<Vec<String>>,
    pub secret: Option<String>,
    pub active: Option<bool>,
    pub retry_config: Option<RetryConfig>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WebhookDelivery {
    pub id: Uuid,
    pub webhook_id: Uuid,
    pub event_type: String,
    pub status: String,
    pub response_code: Option<u16>,
    pub attempted_at: DateTime<Utc>,
    pub error: Option<String>,
}

#[derive(Serialize)]
pub struct TestWebhookResponse {
    pub success: bool,
    pub status_code: Option<u16>,
    pub error: Option<String>,
}

/// Sign a payload with HMAC-SHA256, returning base64-encoded signature.
pub fn sign_payload(secret: &str, payload: &[u8]) -> String {
    let mut mac =
        HmacSha256::new_from_slice(secret.as_bytes()).expect("HMAC can take key of any size");
    mac.update(payload);
    let result = mac.finalize();
    BASE64.encode(result.into_bytes())
}

/// Verify an HMAC-SHA256 signature (base64-encoded).
pub fn verify_signature(secret: &str, payload: &[u8], signature: &str) -> bool {
    let expected = sign_payload(secret, payload);
    subtle::ConstantTimeEq::ct_eq(expected.as_bytes(), signature.as_bytes()).into()
}

use hx_engine::notifications::outbound::StoredWebhook;

// --- Helpers ---

fn response_from_stored(wh: &StoredWebhook) -> OutboundWebhookResponse {
    OutboundWebhookResponse {
        id: wh.id.to_string(),
        url: wh.url.clone(),
        events: wh.events.clone(),
        has_secret: wh.secret.is_some(),
        active: wh.active,
        created_at: wh.created_at.to_rfc3339(),
        retry_config: RetryConfig {
            max_retries: wh.retry_max,
            backoff_secs: wh.retry_backoff_secs,
        },
    }
}

// --- Handlers ---

/// GET /api/v1/webhooks/outbound — list all outbound webhooks.
pub async fn list_outbound_webhooks(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let stored = state.outbound_webhook_store.list().await;
    let webhooks: Vec<OutboundWebhookResponse> = stored.iter().map(response_from_stored).collect();
    Json(webhooks).into_response()
}

/// POST /api/v1/webhooks/outbound — register a new outbound webhook.
pub async fn create_outbound_webhook(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateWebhookRequest>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    if req.url.trim().is_empty() {
        return (StatusCode::BAD_REQUEST, "url is required".to_string()).into_response();
    }

    if req.events.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            "at least one event type is required".to_string(),
        )
            .into_response();
    }

    let retry = req.retry_config.unwrap_or_default();
    let stored = StoredWebhook {
        id: Uuid::now_v7(),
        url: req.url,
        events: req.events,
        secret: req.secret,
        active: true,
        created_at: Utc::now(),
        retry_max: retry.max_retries,
        retry_backoff_secs: retry.backoff_secs,
    };

    let registered = state.outbound_webhook_store.register(stored).await;
    let response = response_from_stored(&registered);
    (StatusCode::CREATED, Json(response)).into_response()
}

/// GET /api/v1/webhooks/outbound/:id — get webhook details.
pub async fn get_outbound_webhook(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let webhook_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "invalid webhook id".to_string()).into_response()
        }
    };

    match state.outbound_webhook_store.get(webhook_id).await {
        Some(wh) => Json(response_from_stored(&wh)).into_response(),
        None => (StatusCode::NOT_FOUND, "webhook not found".to_string()).into_response(),
    }
}

/// DELETE /api/v1/webhooks/outbound/:id — remove a webhook.
pub async fn delete_outbound_webhook(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let webhook_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "invalid webhook id".to_string()).into_response()
        }
    };

    if state.outbound_webhook_store.remove(webhook_id).await {
        StatusCode::NO_CONTENT.into_response()
    } else {
        (StatusCode::NOT_FOUND, "webhook not found".to_string()).into_response()
    }
}

/// POST /api/v1/webhooks/outbound/:id/test — send a test event.
pub async fn test_outbound_webhook(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let webhook_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "invalid webhook id".to_string()).into_response()
        }
    };

    let wh = match state.outbound_webhook_store.get(webhook_id).await {
        Some(wh) => wh,
        None => {
            return (StatusCode::NOT_FOUND, "webhook not found".to_string()).into_response()
        }
    };

    let test_payload = serde_json::json!({
        "event": "test",
        "timestamp": Utc::now().to_rfc3339(),
        "webhook_id": wh.id.to_string(),
    });
    let body = serde_json::to_vec(&test_payload).unwrap_or_default();

    let mut request = state.http_client.post(&wh.url).header("Content-Type", "application/json");
    if let Some(ref secret) = wh.secret {
        let sig = sign_payload(secret, &body);
        request = request.header("X-Helix-Signature", sig);
    }

    let (success, status_code, error) = match request.body(body).send().await {
        Ok(resp) => {
            let code = resp.status().as_u16();
            (code < 400, Some(code), None)
        }
        Err(e) => (false, None, Some(e.to_string())),
    };

    // Record the delivery attempt
    let delivery = hx_engine::notifications::outbound::DeliveryRecord {
        id: Uuid::now_v7(),
        webhook_id,
        event_type: "test".to_string(),
        status: if success {
            "delivered".to_string()
        } else {
            "failed".to_string()
        },
        response_code: status_code,
        error: error.clone(),
        delivered_at: Utc::now(),
    };
    state
        .outbound_webhook_store
        .record_delivery(delivery)
        .await;

    Json(TestWebhookResponse {
        success,
        status_code,
        error,
    })
    .into_response()
}

/// GET /api/v1/webhooks/outbound/:id/deliveries — delivery history.
pub async fn list_webhook_deliveries(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let webhook_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (StatusCode::BAD_REQUEST, "invalid webhook id".to_string()).into_response()
        }
    };

    let records = state
        .outbound_webhook_store
        .list_deliveries(webhook_id)
        .await;
    let deliveries: Vec<WebhookDelivery> = records
        .into_iter()
        .map(|r| WebhookDelivery {
            id: r.id,
            webhook_id: r.webhook_id,
            event_type: r.event_type,
            status: r.status,
            response_code: r.response_code,
            attempted_at: r.delivered_at,
            error: r.error,
        })
        .collect();
    Json(deliveries).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_payload_produces_deterministic_output() {
        let sig1 = sign_payload("secret", b"hello world");
        let sig2 = sign_payload("secret", b"hello world");
        assert_eq!(sig1, sig2);
        assert!(!sig1.is_empty());
    }

    #[test]
    fn sign_payload_different_secrets_differ() {
        let sig1 = sign_payload("secret1", b"payload");
        let sig2 = sign_payload("secret2", b"payload");
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn verify_signature_valid() {
        let sig = sign_payload("my-secret", b"test data");
        assert!(verify_signature("my-secret", b"test data", &sig));
    }

    #[test]
    fn verify_signature_invalid() {
        assert!(!verify_signature("secret", b"data", "invalid-sig"));
    }

    #[test]
    fn webhook_response_hides_secret() {
        let wh = OutboundWebhook {
            id: Uuid::nil(),
            url: "https://example.com/hook".into(),
            events: vec!["node.created".into()],
            secret: Some("super-secret".into()),
            active: true,
            created_at: Utc::now(),
            retry_config: RetryConfig::default(),
        };
        let resp = OutboundWebhookResponse::from(&wh);
        assert!(resp.has_secret);
        let json = serde_json::to_string(&resp).unwrap();
        assert!(!json.contains("super-secret"));
    }

    #[test]
    fn retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.backoff_secs, 10);
    }

    #[test]
    fn create_request_deserializes() {
        let json = r#"{"url":"https://example.com","events":["test"],"secret":"s3cret"}"#;
        let req: CreateWebhookRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.url, "https://example.com");
        assert_eq!(req.secret.as_deref(), Some("s3cret"));
    }

    #[test]
    fn delivery_serializes() {
        let delivery = WebhookDelivery {
            id: Uuid::nil(),
            webhook_id: Uuid::nil(),
            event_type: "node.created".into(),
            status: "delivered".into(),
            response_code: Some(200),
            attempted_at: Utc::now(),
            error: None,
        };
        let json = serde_json::to_string(&delivery).unwrap();
        assert!(json.contains("delivered"));
    }
}
