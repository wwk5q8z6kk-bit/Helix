//! Webhook adapter — generic inbound/outbound HTTP webhooks.
//!
//! Configuration keys:
//! - `secret`: HMAC-SHA256 shared secret for signature verification (optional)
//! - `outbound_url`: URL for outbound messages (optional)

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use sha2::Sha256;

use hx_core::{HxError, MvResult};

use super::{
    AdapterConfig, AdapterInboundMessage, AdapterOutboundMessage, AdapterStatus, AdapterType,
    ExternalAdapter,
};

type HmacSha256 = Hmac<Sha256>;

#[derive(Debug)]
pub struct WebhookAdapter {
    config: AdapterConfig,
    client: reqwest::Client,
    inbound_buffer: Arc<Mutex<Vec<AdapterInboundMessage>>>,
    last_send: Mutex<Option<DateTime<Utc>>>,
    last_receive: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

impl WebhookAdapter {
    pub fn new(config: AdapterConfig) -> MvResult<Self> {
        Ok(Self {
            config,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .map_err(|e| HxError::Internal(e.to_string()))?,
            inbound_buffer: Arc::new(Mutex::new(Vec::new())),
            last_send: Mutex::new(None),
            last_receive: Mutex::new(None),
            last_error: Mutex::new(None),
        })
    }

    /// Push an inbound message into the buffer. Called by REST handlers
    /// when a webhook payload is received.
    pub fn push_inbound(&self, msg: AdapterInboundMessage) {
        self.inbound_buffer.lock().unwrap().push(msg);
        *self.last_receive.lock().unwrap() = Some(Utc::now());
    }

    /// Verify an HMAC-SHA256 signature against the configured secret.
    ///
    /// Returns `Ok(true)` if the signature is valid, `Ok(false)` if invalid,
    /// or `Err` if no secret is configured.
    pub fn verify_signature(&self, body: &[u8], signature: &str) -> MvResult<bool> {
        let secret = self
            .config
            .get_setting("secret")
            .ok_or_else(|| HxError::Config("no secret configured for signature verification".into()))?;

        let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
            .map_err(|e| HxError::Internal(format!("hmac init failed: {e}")))?;
        mac.update(body);

        // Signature is expected as hex-encoded
        let expected = hex_encode(mac.finalize().into_bytes().as_slice());
        Ok(constant_time_eq(signature.as_bytes(), expected.as_bytes()))
    }

    /// Get a reference to the inbound buffer for external integration.
    pub fn inbound_buffer(&self) -> Arc<Mutex<Vec<AdapterInboundMessage>>> {
        Arc::clone(&self.inbound_buffer)
    }
}

/// Hex-encode a byte slice.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Constant-time byte comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y))
        == 0
}

#[async_trait]
impl ExternalAdapter for WebhookAdapter {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::Webhook
    }

    async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
        let outbound_url = match self.config.get_setting("outbound_url") {
            Some(url) => url.to_string(),
            None => {
                return Err(HxError::Config(
                    "webhook adapter has no outbound_url configured".into(),
                ));
            }
        };

        let payload = serde_json::json!({
            "channel": message.channel,
            "content": message.content,
            "thread_id": message.thread_id,
            "metadata": message.metadata,
        });

        let resp = self
            .client
            .post(&outbound_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("webhook send failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!("webhook outbound returned {status}: {body}");
            *self.last_error.lock().unwrap() = Some(err.clone());
            return Err(HxError::Internal(err));
        }

        *self.last_send.lock().unwrap() = Some(Utc::now());
        *self.last_error.lock().unwrap() = None;
        Ok(())
    }

    async fn poll(&self, _cursor: Option<&str>) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
        let messages: Vec<AdapterInboundMessage> =
            self.inbound_buffer.lock().unwrap().drain(..).collect();
        let count = messages.len();
        Ok((messages, count.to_string()))
    }

    async fn health_check(&self) -> MvResult<bool> {
        // Webhook adapter is passive — always healthy
        Ok(true)
    }

    fn status(&self) -> AdapterStatus {
        let error = self.last_error.lock().unwrap().clone();
        let last_send = *self.last_send.lock().unwrap();
        let last_receive = *self.last_receive.lock().unwrap();
        AdapterStatus {
            adapter_type: AdapterType::Webhook,
            name: self.config.name.clone(),
            connected: error.is_none(),
            last_send,
            last_receive,
            error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn webhook_config() -> AdapterConfig {
        AdapterConfig::new(AdapterType::Webhook, "test-webhook")
    }

    fn webhook_config_with_secret() -> AdapterConfig {
        webhook_config().with_setting("secret", "my-webhook-secret")
    }

    #[test]
    fn new_succeeds_with_no_settings() {
        let adapter = WebhookAdapter::new(webhook_config());
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "test-webhook");
    }

    #[tokio::test]
    async fn adapter_type_is_webhook() {
        let adapter = WebhookAdapter::new(webhook_config()).unwrap();
        assert_eq!(adapter.adapter_type(), AdapterType::Webhook);
    }

    #[tokio::test]
    async fn health_check_always_true() {
        let adapter = WebhookAdapter::new(webhook_config()).unwrap();
        let healthy = adapter.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn push_and_poll_drains_buffer() {
        let adapter = WebhookAdapter::new(webhook_config()).unwrap();

        let msg = AdapterInboundMessage {
            external_id: "wh-1".into(),
            channel: "incoming".into(),
            sender: "github".into(),
            content: "push event".into(),
            thread_id: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        adapter.push_inbound(msg);

        let (messages, cursor) = adapter.poll(None).await.unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, "push event");
        assert_eq!(cursor, "1");

        // Second poll should be empty
        let (messages, cursor) = adapter.poll(None).await.unwrap();
        assert!(messages.is_empty());
        assert_eq!(cursor, "0");
    }

    #[test]
    fn verify_signature_valid() {
        let adapter = WebhookAdapter::new(webhook_config_with_secret()).unwrap();
        let body = b"test payload";

        // Compute expected signature
        let mut mac = HmacSha256::new_from_slice(b"my-webhook-secret").unwrap();
        mac.update(body);
        let expected = hex_encode(mac.finalize().into_bytes().as_slice());

        let result = adapter.verify_signature(body, &expected).unwrap();
        assert!(result);
    }

    #[test]
    fn verify_signature_invalid() {
        let adapter = WebhookAdapter::new(webhook_config_with_secret()).unwrap();
        let result = adapter.verify_signature(b"test", "bad-signature").unwrap();
        assert!(!result);
    }

    #[test]
    fn verify_signature_no_secret_errors() {
        let adapter = WebhookAdapter::new(webhook_config()).unwrap();
        let result = adapter.verify_signature(b"test", "sig");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn send_without_outbound_url_fails() {
        let adapter = WebhookAdapter::new(webhook_config()).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "test".into(),
            content: "hello".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        let result = adapter.send(&msg).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outbound_url"));
    }

    #[tokio::test]
    async fn send_success_with_mock() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_body("ok")
            .create_async()
            .await;

        let config = webhook_config().with_setting("outbound_url", &server.url());
        let adapter = WebhookAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "target".into(),
            content: "outbound msg".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        adapter.send(&msg).await.unwrap();
        mock.assert_async().await;
        let status = adapter.status();
        assert!(status.last_send.is_some());
    }

    #[tokio::test]
    async fn initial_status_connected() {
        let adapter = WebhookAdapter::new(webhook_config()).unwrap();
        let status = adapter.status();
        assert!(status.connected);
        assert!(status.error.is_none());
        assert_eq!(status.adapter_type, AdapterType::Webhook);
    }
}
