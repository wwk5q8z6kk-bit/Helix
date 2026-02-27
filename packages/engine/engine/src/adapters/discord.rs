//! Discord adapter — sends messages via webhook URL.
//!
//! Configuration keys:
//! - `webhook_url`: Discord webhook URL for outbound messages

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
pub struct DiscordAdapter {
    config: AdapterConfig,
    client: reqwest::Client,
    last_send: Mutex<Option<DateTime<Utc>>>,
    last_receive: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

impl DiscordAdapter {
    pub fn new(config: AdapterConfig) -> MvResult<Self> {
        if config.get_setting("webhook_url").is_none() {
            return Err(HxError::Config(
                "Discord adapter requires 'webhook_url' setting".into(),
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
impl ExternalAdapter for DiscordAdapter {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::Discord
    }

    async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
        let webhook_url = self
            .config
            .get_setting("webhook_url")
            .ok_or_else(|| HxError::Config("missing webhook_url".into()))?;

        // Discord webhooks accept JSON with "content" field
        // Max 2000 chars per message — truncate if needed
        let content = if message.content.len() > 2000 {
            format!("{}...", &message.content[..1997])
        } else {
            message.content.clone()
        };

        let payload = serde_json::json!({
            "content": content,
        });

        let resp = self
            .client
            .post(webhook_url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("discord send failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!("discord webhook returned {status}: {body}");
            *self.last_error.lock().unwrap() = Some(err.clone());
            return Err(HxError::Internal(err));
        }

        *self.last_send.lock().unwrap() = Some(Utc::now());
        *self.last_error.lock().unwrap() = None;
        Ok(())
    }

    async fn poll(
        &self,
        cursor: Option<&str>,
    ) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
        // If a bot_token and channel_id are configured, use the Discord REST
        // API to poll for messages. Otherwise, fall back to empty (webhook-only).
        let bot_token = match self.config.get_setting("bot_token") {
            Some(t) => t,
            None => return Ok((vec![], cursor.unwrap_or("0").to_string())),
        };

        let channel_id = match self.config.get_setting("channel_id") {
            Some(c) => c,
            None => return Ok((vec![], cursor.unwrap_or("0").to_string())),
        };

        // Discord GET /channels/{channel_id}/messages?after={snowflake}&limit=50
        let url = format!(
            "https://discord.com/api/v10/channels/{channel_id}/messages"
        );

        let mut params: Vec<(&str, String)> = vec![("limit", "50".into())];
        if let Some(after_id) = cursor {
            if after_id != "0" {
                params.push(("after", after_id.to_string()));
            }
        }

        let resp = self
            .client
            .get(&url)
            .header("Authorization", format!("Bot {bot_token}"))
            .query(&params)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("discord poll failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(HxError::Internal(format!(
                "discord API returned {status}: {body}"
            )));
        }

        let body: Vec<serde_json::Value> = resp
            .json()
            .await
            .map_err(|e| HxError::Internal(format!("discord poll parse failed: {e}")))?;

        let messages: Vec<AdapterInboundMessage> = body
            .iter()
            .filter_map(|m| {
                let id = m.get("id")?.as_str()?;
                let content = m.get("content")?.as_str()?;
                let author = m.get("author")?.get("username")?.as_str()?;
                let timestamp_str = m.get("timestamp")?.as_str()?;
                let timestamp = chrono::DateTime::parse_from_rfc3339(timestamp_str)
                    .map(|dt| dt.with_timezone(&Utc))
                    .unwrap_or_else(|_| Utc::now());

                // Check for thread reference
                let thread_id = m
                    .get("message_reference")
                    .and_then(|r| r.get("message_id"))
                    .and_then(|v| v.as_str())
                    .map(String::from);

                Some(AdapterInboundMessage {
                    external_id: id.to_string(),
                    channel: channel_id.to_string(),
                    sender: author.to_string(),
                    content: content.to_string(),
                    thread_id,
                    timestamp,
                    metadata: HashMap::new(),
                })
            })
            .collect();

        // Discord returns messages newest-first; the highest snowflake ID
        // is the newest message, which becomes our next cursor.
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
        // Test the webhook URL with a GET (Discord returns webhook info)
        let webhook_url = match self.config.get_setting("webhook_url") {
            Some(url) => url,
            None => return Ok(false),
        };

        let resp = self
            .client
            .get(webhook_url)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("discord health check failed: {e}")))?;

        Ok(resp.status().is_success())
    }

    fn status(&self) -> AdapterStatus {
        let error = self.last_error.lock().unwrap().clone();
        let last_send = *self.last_send.lock().unwrap();
        let last_receive = *self.last_receive.lock().unwrap();
        AdapterStatus {
            adapter_type: AdapterType::Discord,
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

    fn discord_config_with_webhook() -> AdapterConfig {
        AdapterConfig::new(AdapterType::Discord, "test-discord")
            .with_setting("webhook_url", "https://discord.com/api/webhooks/1234/abcd")
    }

    #[test]
    fn new_requires_webhook_url() {
        // No reqwest::Client created on the error path, so #[test] is fine
        let config = AdapterConfig::new(AdapterType::Discord, "no-webhook");
        let result = DiscordAdapter::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("webhook_url"), "expected webhook_url error, got: {err}");
    }

    #[tokio::test]
    async fn new_succeeds_with_webhook_url() {
        let adapter = DiscordAdapter::new(discord_config_with_webhook());
        assert!(adapter.is_ok());
    }

    #[tokio::test]
    async fn name_returns_config_name() {
        let adapter = DiscordAdapter::new(discord_config_with_webhook()).unwrap();
        assert_eq!(adapter.name(), "test-discord");
    }

    #[tokio::test]
    async fn adapter_type_is_discord() {
        let adapter = DiscordAdapter::new(discord_config_with_webhook()).unwrap();
        assert_eq!(adapter.adapter_type(), AdapterType::Discord);
    }

    #[tokio::test]
    async fn initial_status_is_connected_no_receive() {
        let adapter = DiscordAdapter::new(discord_config_with_webhook()).unwrap();
        let status = adapter.status();
        assert!(status.connected);
        assert!(status.last_send.is_none());
        assert!(status.last_receive.is_none());
        assert!(status.error.is_none());
        assert_eq!(status.adapter_type, AdapterType::Discord);
    }

    #[tokio::test]
    async fn poll_without_bot_token_returns_empty() {
        let adapter = DiscordAdapter::new(discord_config_with_webhook()).unwrap();
        let (messages, cursor) = adapter.poll(None).await.unwrap();
        assert!(messages.is_empty());
        assert_eq!(cursor, "0");

        let (messages, cursor) = adapter.poll(Some("custom-cursor")).await.unwrap();
        assert!(messages.is_empty());
        assert_eq!(cursor, "custom-cursor");
    }

    #[tokio::test]
    async fn poll_without_channel_id_returns_empty() {
        let config = discord_config_with_webhook()
            .with_setting("bot_token", "test-bot-token");
        let adapter = DiscordAdapter::new(config).unwrap();
        let (messages, cursor) = adapter.poll(Some("123")).await.unwrap();
        assert!(messages.is_empty());
        assert_eq!(cursor, "123");
    }

    #[test]
    fn message_truncation_at_2000_chars() {
        // Verify the truncation logic directly
        let long_content = "x".repeat(3000);
        let truncated = if long_content.len() > 2000 {
            format!("{}...", &long_content[..1997])
        } else {
            long_content.clone()
        };
        assert_eq!(truncated.len(), 2000);
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn short_message_not_truncated() {
        let content = "Hello Discord!".to_string();
        let result = if content.len() > 2000 {
            format!("{}...", &content[..1997])
        } else {
            content.clone()
        };
        assert_eq!(result, "Hello Discord!");
    }

    #[test]
    fn message_exactly_2000_chars_not_truncated() {
        let content = "x".repeat(2000);
        let result = if content.len() > 2000 {
            format!("{}...", &content[..1997])
        } else {
            content.clone()
        };
        assert_eq!(result.len(), 2000);
        assert!(!result.ends_with("..."));
    }

    // --- mockito-based integration tests ---

    #[tokio::test]
    async fn send_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/")
            .with_status(204)
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Discord, "test-discord")
            .with_setting("webhook_url", &server.url());
        let adapter = DiscordAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "#test".into(),
            content: "hello".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        adapter.send(&msg).await.unwrap();
        mock.assert_async().await;
        let status = adapter.status();
        assert!(status.last_send.is_some());
        assert!(status.error.is_none());
    }

    #[tokio::test]
    async fn send_failure_status() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(500)
            .with_body("server error")
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Discord, "test-discord")
            .with_setting("webhook_url", &server.url());
        let adapter = DiscordAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "#test".into(),
            content: "fail".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        let result = adapter.send(&msg).await;
        assert!(result.is_err());
        let status = adapter.status();
        assert!(!status.connected);
        assert!(status.error.is_some());
    }

    #[tokio::test]
    async fn send_truncates_long_message() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("POST", "/")
            .with_status(204)
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Discord, "test-discord")
            .with_setting("webhook_url", &server.url());
        let adapter = DiscordAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "#test".into(),
            content: "x".repeat(3000),
            thread_id: None,
            metadata: HashMap::new(),
        };
        // Should not error — message gets truncated internally
        adapter.send(&msg).await.unwrap();
    }

    #[tokio::test]
    async fn health_check_success() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", "/")
            .with_status(200)
            .with_body("{\"name\": \"test webhook\"}")
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Discord, "test-discord")
            .with_setting("webhook_url", &server.url());
        let adapter = DiscordAdapter::new(config).unwrap();
        let healthy = adapter.health_check().await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn health_check_failure() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", "/")
            .with_status(404)
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Discord, "test-discord")
            .with_setting("webhook_url", &server.url());
        let adapter = DiscordAdapter::new(config).unwrap();
        let healthy = adapter.health_check().await.unwrap();
        assert!(!healthy);
    }
}
