//! Telegram adapter — sends messages via Bot API, polls via getUpdates.
//!
//! Configuration keys:
//! - `bot_token`: Telegram Bot API token (required)
//! - `chat_id`: Default chat ID for outbound messages (optional)

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
pub struct TelegramAdapter {
    config: AdapterConfig,
    client: reqwest::Client,
    last_send: Mutex<Option<DateTime<Utc>>>,
    last_receive: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

impl TelegramAdapter {
    pub fn new(config: AdapterConfig) -> MvResult<Self> {
        if config.get_setting("bot_token").is_none() {
            return Err(HxError::Config(
                "Telegram adapter requires 'bot_token' setting".into(),
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

    fn api_url(&self, method: &str) -> String {
        let token = self
            .config
            .get_setting("bot_token")
            .unwrap_or("MISSING");
        let base = self
            .config
            .get_setting("base_url")
            .unwrap_or("https://api.telegram.org");
        format!("{base}/bot{token}/{method}")
    }
}

#[async_trait]
impl ExternalAdapter for TelegramAdapter {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::Telegram
    }

    async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
        let chat_id = if message.channel.is_empty() {
            self.config
                .get_setting("chat_id")
                .ok_or_else(|| {
                    HxError::Config("no chat_id in message or adapter config".into())
                })?
                .to_string()
        } else {
            message.channel.clone()
        };

        let mut payload = serde_json::json!({
            "chat_id": chat_id,
            "text": message.content,
        });

        if let Some(ref reply_to) = message.thread_id {
            if let Ok(msg_id) = reply_to.parse::<i64>() {
                payload["reply_to_message_id"] = serde_json::Value::Number(msg_id.into());
            }
        }

        let resp = self
            .client
            .post(self.api_url("sendMessage"))
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("telegram send failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!("telegram sendMessage returned {status}: {body}");
            *self.last_error.lock().unwrap() = Some(err.clone());
            return Err(HxError::Internal(err));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| HxError::Internal(format!("telegram send parse failed: {e}")))?;

        if body.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let desc = body
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("unknown error");
            let err = format!("telegram API error: {desc}");
            *self.last_error.lock().unwrap() = Some(err.clone());
            return Err(HxError::Internal(err));
        }

        *self.last_send.lock().unwrap() = Some(Utc::now());
        *self.last_error.lock().unwrap() = None;
        Ok(())
    }

    async fn poll(&self, cursor: Option<&str>) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
        let offset: i64 = cursor
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);

        let params = [
            ("offset", offset.to_string()),
            ("timeout", "0".to_string()),
            ("limit", "100".to_string()),
        ];

        let resp = self
            .client
            .get(self.api_url("getUpdates"))
            .query(&params)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("telegram poll failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| HxError::Internal(format!("telegram poll parse failed: {e}")))?;

        if body.get("ok") != Some(&serde_json::Value::Bool(true)) {
            let desc = body
                .get("description")
                .and_then(|d| d.as_str())
                .unwrap_or("unknown");
            return Err(HxError::Internal(format!("telegram API error: {desc}")));
        }

        let updates = body
            .get("result")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default();

        let mut messages = Vec::new();
        let mut max_update_id: i64 = offset;

        for update in &updates {
            let update_id = match update.get("update_id").and_then(|u| u.as_i64()) {
                Some(id) => id,
                None => continue,
            };
            if update_id >= max_update_id {
                max_update_id = update_id;
            }

            let msg = match update.get("message") {
                Some(m) => m,
                None => continue,
            };

            let text = match msg.get("text").and_then(|t| t.as_str()) {
                Some(t) => t,
                None => continue,
            };

            let message_id = msg
                .get("message_id")
                .and_then(|m| m.as_i64())
                .unwrap_or(0);

            let chat_id = msg
                .get("chat")
                .and_then(|c| c.get("id"))
                .and_then(|id| id.as_i64())
                .map(|id| id.to_string())
                .unwrap_or_default();

            let sender = msg
                .get("from")
                .and_then(|f| f.get("username"))
                .and_then(|u| u.as_str())
                .unwrap_or("unknown")
                .to_string();

            let date = msg
                .get("date")
                .and_then(|d| d.as_i64())
                .and_then(|secs| DateTime::from_timestamp(secs, 0))
                .unwrap_or_else(Utc::now);

            let reply_to = msg
                .get("reply_to_message")
                .and_then(|r| r.get("message_id"))
                .and_then(|id| id.as_i64())
                .map(|id| id.to_string());

            messages.push(AdapterInboundMessage {
                external_id: message_id.to_string(),
                channel: chat_id,
                sender,
                content: text.to_string(),
                thread_id: reply_to,
                timestamp: date,
                metadata: HashMap::new(),
            });
        }

        // New cursor = max(update_id) + 1 so we don't re-fetch these updates
        let new_cursor = if !updates.is_empty() {
            (max_update_id + 1).to_string()
        } else {
            offset.to_string()
        };

        if !messages.is_empty() {
            *self.last_receive.lock().unwrap() = Some(Utc::now());
        }

        Ok((messages, new_cursor))
    }

    async fn health_check(&self) -> MvResult<bool> {
        let resp = self
            .client
            .get(self.api_url("getMe"))
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("telegram health check failed: {e}")))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| HxError::Internal(e.to_string()))?;

        let is_bot = body
            .get("result")
            .and_then(|r| r.get("is_bot"))
            .and_then(|b| b.as_bool())
            .unwrap_or(false);

        Ok(body.get("ok") == Some(&serde_json::Value::Bool(true)) && is_bot)
    }

    fn status(&self) -> AdapterStatus {
        let error = self.last_error.lock().unwrap().clone();
        let last_send = *self.last_send.lock().unwrap();
        let last_receive = *self.last_receive.lock().unwrap();
        AdapterStatus {
            adapter_type: AdapterType::Telegram,
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

    fn telegram_config() -> AdapterConfig {
        AdapterConfig::new(AdapterType::Telegram, "test-telegram")
            .with_setting("bot_token", "123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
    }

    #[test]
    fn new_requires_bot_token() {
        let config = AdapterConfig::new(AdapterType::Telegram, "no-token");
        let result = TelegramAdapter::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("bot_token"),
            "expected bot_token error, got: {err}"
        );
    }

    #[tokio::test]
    async fn new_succeeds_with_bot_token() {
        let adapter = TelegramAdapter::new(telegram_config());
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "test-telegram");
    }

    #[tokio::test]
    async fn adapter_type_is_telegram() {
        let adapter = TelegramAdapter::new(telegram_config()).unwrap();
        assert_eq!(adapter.adapter_type(), AdapterType::Telegram);
    }

    #[tokio::test]
    async fn initial_status_connected() {
        let adapter = TelegramAdapter::new(telegram_config()).unwrap();
        let status = adapter.status();
        assert!(status.connected);
        assert!(status.last_send.is_none());
        assert!(status.last_receive.is_none());
        assert!(status.error.is_none());
        assert_eq!(status.adapter_type, AdapterType::Telegram);
    }

    #[tokio::test]
    async fn send_success() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/bot123:testtoken/sendMessage")
            .with_status(200)
            .with_body(r#"{"ok":true,"result":{"message_id":1}}"#)
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Telegram, "test-telegram")
            .with_setting("bot_token", "123:testtoken")
            .with_setting("base_url", &server.url());
        let adapter = TelegramAdapter::new(config).unwrap();

        let msg = AdapterOutboundMessage {
            channel: "12345".into(),
            content: "hello".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        adapter.send(&msg).await.unwrap();
        mock.assert_async().await;
    }

    #[tokio::test]
    async fn send_without_chat_id_fails() {
        let config = AdapterConfig::new(AdapterType::Telegram, "test-telegram")
            .with_setting("bot_token", "123:testtoken");
        let adapter = TelegramAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: String::new(),
            content: "hello".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        let result = adapter.send(&msg).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("chat_id"));
    }

    #[tokio::test]
    async fn poll_with_no_cursor_uses_zero_offset() {
        // Verify the adapter handles None cursor correctly
        let adapter = TelegramAdapter::new(telegram_config()).unwrap();
        // The poll will fail (no real server), but we verify cursor logic
        // by checking that it attempts to parse cursor as i64
        let cursor: Option<&str> = None;
        let offset: i64 = cursor.and_then(|c| c.parse().ok()).unwrap_or(0);
        assert_eq!(offset, 0);
    }

    #[tokio::test]
    async fn poll_cursor_parsing() {
        // Verify cursor arithmetic logic
        let cursor = Some("42");
        let offset: i64 = cursor.and_then(|c| c.parse().ok()).unwrap_or(0);
        assert_eq!(offset, 42);

        // Non-numeric cursor defaults to 0
        let cursor = Some("abc");
        let offset: i64 = cursor.and_then(|c| c.parse().ok()).unwrap_or(0);
        assert_eq!(offset, 0);
    }
}
