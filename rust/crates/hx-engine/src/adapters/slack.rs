//! Slack adapter — sends messages via incoming webhook, polls via Web API.
//!
//! Configuration keys:
//! - `webhook_url`: Slack incoming webhook URL for outbound messages
//! - `bot_token`: (optional) Slack Bot Token for polling conversations.history
//! - `channel_id`: (optional) Default channel ID for polling

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use hx_core::{HxError, MvResult};

use super::{
    AdapterConfig, AdapterInboundMessage, AdapterOutboundMessage, AdapterStatus, AdapterType,
    ExternalAdapter,
};

#[derive(Debug)]
pub struct SlackAdapter {
    config: AdapterConfig,
    client: reqwest::Client,
    last_send: Mutex<Option<DateTime<Utc>>>,
    last_receive: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

impl SlackAdapter {
    pub fn new(config: AdapterConfig) -> MvResult<Self> {
        if config.get_setting("webhook_url").is_none() {
            return Err(HxError::Config(
                "Slack adapter requires 'webhook_url' setting".into(),
            ));
        }
        Ok(Self {
            config,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .map_err(|e| HxError::Internal(e.to_string()))?,
            last_send: Mutex::new(None),
            last_receive: Mutex::new(None),
            last_error: Mutex::new(None),
        })
    }
}

#[async_trait]
impl ExternalAdapter for SlackAdapter {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::Slack
    }

    async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
        let webhook_url = self
            .config
            .get_setting("webhook_url")
            .ok_or_else(|| HxError::Config("missing webhook_url".into()))?;

        let mut payload = serde_json::json!({
            "text": message.content,
        });

        // Override channel if specified in the message
        if !message.channel.is_empty() {
            payload["channel"] = serde_json::Value::String(message.channel.clone());
        }

        if let Some(ref thread_ts) = message.thread_id {
            payload["thread_ts"] = serde_json::Value::String(thread_ts.clone());
        }

        let resp = self
            .client
            .post(webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("slack send failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!("slack webhook returned {status}: {body}");
            *self.last_error.lock().unwrap() = Some(err.clone());
            return Err(HxError::Internal(err));
        }

        *self.last_send.lock().unwrap() = Some(Utc::now());
        *self.last_error.lock().unwrap() = None;
        Ok(())
    }

    async fn poll(&self, cursor: Option<&str>) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
        let bot_token = match self.config.get_setting("bot_token") {
            Some(t) => t,
            None => return Ok((vec![], cursor.unwrap_or("0").to_string())),
        };

        let channel_id = match self.config.get_setting("channel_id") {
            Some(c) => c,
            None => return Ok((vec![], cursor.unwrap_or("0").to_string())),
        };

        let mut params = vec![("channel", channel_id.to_string()), ("limit", "20".into())];
        if let Some(oldest) = cursor {
            params.push(("oldest", oldest.to_string()));
        }

        let resp = self
            .client
            .get("https://slack.com/api/conversations.history")
            .bearer_auth(bot_token)
            .query(&params)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("slack poll failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| HxError::Internal(format!("slack poll parse failed: {e}")))?;

        if body.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let err = body
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown");
            return Err(HxError::Internal(format!("slack API error: {err}")));
        }

        let messages = body
            .get("messages")
            .and_then(|m| m.as_array())
            .map(|msgs| {
                msgs.iter()
                    .filter_map(|m| {
                        let ts = m.get("ts")?.as_str()?;
                        let text = m.get("text")?.as_str()?;
                        let user = m
                            .get("user")
                            .and_then(|u| u.as_str())
                            .unwrap_or("unknown");
                        let thread_ts = m.get("thread_ts").and_then(|t| t.as_str()).map(String::from);

                        // Parse Slack timestamp (epoch.seq format)
                        let timestamp = ts
                            .split('.')
                            .next()
                            .and_then(|s| s.parse::<i64>().ok())
                            .and_then(|secs| {
                                DateTime::from_timestamp(secs, 0)
                            })
                            .unwrap_or_else(Utc::now);

                        Some(AdapterInboundMessage {
                            external_id: ts.to_string(),
                            channel: channel_id.to_string(),
                            sender: user.to_string(),
                            content: text.to_string(),
                            thread_id: thread_ts,
                            timestamp,
                            metadata: HashMap::new(),
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        // Use the latest message timestamp as cursor
        let new_cursor = messages
            .first()
            .map(|m| m.external_id.clone())
            .unwrap_or_else(|| cursor.unwrap_or("0").to_string());

        if !messages.is_empty() {
            *self.last_receive.lock().unwrap() = Some(Utc::now());
        }

        Ok((messages, new_cursor))
    }

    async fn health_check(&self) -> MvResult<bool> {
        // If we have a bot token, test the auth.test endpoint
        if let Some(token) = self.config.get_setting("bot_token") {
            let resp = self
                .client
                .post("https://slack.com/api/auth.test")
                .bearer_auth(token)
                .send()
                .await
                .map_err(|e| HxError::Internal(format!("slack health check failed: {e}")))?;

            let body: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| HxError::Internal(e.to_string()))?;

            return Ok(body.get("ok") == Some(&serde_json::Value::Bool(true)));
        }

        // If only webhook, we can't easily test it — assume healthy if configured
        Ok(self.config.get_setting("webhook_url").is_some())
    }

    fn status(&self) -> AdapterStatus {
        let error = self.last_error.lock().unwrap().clone();
        let last_send = *self.last_send.lock().unwrap();
        let last_receive = *self.last_receive.lock().unwrap();
        AdapterStatus {
            adapter_type: AdapterType::Slack,
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

    fn slack_config_with_webhook() -> AdapterConfig {
        AdapterConfig::new(AdapterType::Slack, "test-slack")
            .with_setting("webhook_url", "https://hooks.slack.com/services/T00/B00/xxx")
    }

    #[test]
    fn new_requires_webhook_url() {
        // No reqwest::Client created on the error path, so #[test] is fine
        let config = AdapterConfig::new(AdapterType::Slack, "no-webhook");
        let result = SlackAdapter::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("webhook_url"), "expected webhook_url error, got: {err}");
    }

    #[tokio::test]
    async fn new_succeeds_with_webhook_url() {
        let adapter = SlackAdapter::new(slack_config_with_webhook());
        assert!(adapter.is_ok());
    }

    #[tokio::test]
    async fn name_returns_config_name() {
        let adapter = SlackAdapter::new(slack_config_with_webhook()).unwrap();
        assert_eq!(adapter.name(), "test-slack");
    }

    #[tokio::test]
    async fn adapter_type_is_slack() {
        let adapter = SlackAdapter::new(slack_config_with_webhook()).unwrap();
        assert_eq!(adapter.adapter_type(), AdapterType::Slack);
    }

    #[tokio::test]
    async fn initial_status_is_connected_with_no_error() {
        let adapter = SlackAdapter::new(slack_config_with_webhook()).unwrap();
        let status = adapter.status();
        assert!(status.connected);
        assert!(status.last_send.is_none());
        assert!(status.last_receive.is_none());
        assert!(status.error.is_none());
        assert_eq!(status.adapter_type, AdapterType::Slack);
    }

    #[tokio::test]
    async fn poll_without_bot_token_returns_empty() {
        let adapter = SlackAdapter::new(slack_config_with_webhook()).unwrap();
        let (messages, cursor) = adapter.poll(None).await.unwrap();
        assert!(messages.is_empty());
        assert_eq!(cursor, "0");
    }

    #[tokio::test]
    async fn poll_without_channel_id_returns_empty() {
        let config = slack_config_with_webhook()
            .with_setting("bot_token", "xoxb-test-token");
        let adapter = SlackAdapter::new(config).unwrap();
        let (messages, cursor) = adapter.poll(Some("123")).await.unwrap();
        assert!(messages.is_empty());
        assert_eq!(cursor, "123");
    }

    #[tokio::test]
    async fn health_check_webhook_only() {
        let adapter = SlackAdapter::new(slack_config_with_webhook()).unwrap();
        // Webhook-only health check returns true if configured
        let healthy = adapter.health_check().await.unwrap();
        assert!(healthy);
    }

    // --- mockito-based integration tests ---

    #[tokio::test]
    async fn send_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/")
            .with_status(200)
            .with_body("ok")
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Slack, "test-slack")
            .with_setting("webhook_url", &server.url());
        let adapter = SlackAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "#test".into(),
            content: "hello slack".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        adapter.send(&msg).await.unwrap();
        mock.assert_async().await;
        let status = adapter.status();
        assert!(status.last_send.is_some());
    }

    #[tokio::test]
    async fn send_failure() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(500)
            .with_body("error")
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Slack, "test-slack")
            .with_setting("webhook_url", &server.url());
        let adapter = SlackAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "#test".into(),
            content: "fail".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        let result = adapter.send(&msg).await;
        assert!(result.is_err());
        let status = adapter.status();
        assert!(status.error.is_some());
    }

    #[tokio::test]
    async fn send_with_thread_ts() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(200)
            .with_body("ok")
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Slack, "test-slack")
            .with_setting("webhook_url", &server.url());
        let adapter = SlackAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "#test".into(),
            content: "reply".into(),
            thread_id: Some("1234567890.123456".into()),
            metadata: HashMap::new(),
        };
        adapter.send(&msg).await.unwrap();
    }
}
