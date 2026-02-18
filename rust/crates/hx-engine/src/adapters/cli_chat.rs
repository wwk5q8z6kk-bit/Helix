//! CLI Chat adapter — wraps stdin/stdout for interactive REPL usage.
//!
//! Configuration keys:
//! - `name`: Display name for the chat user (optional, defaults to "user")

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use hx_core::MvResult;

use super::{
    AdapterConfig, AdapterInboundMessage, AdapterOutboundMessage, AdapterStatus, AdapterType,
    ExternalAdapter,
};

#[derive(Debug)]
pub struct CliChatAdapter {
    config: AdapterConfig,
    inbound_buffer: Mutex<Vec<AdapterInboundMessage>>,
    outbound_buffer: Mutex<Vec<AdapterOutboundMessage>>,
    last_send: Mutex<Option<DateTime<Utc>>>,
    last_receive: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

impl CliChatAdapter {
    pub fn new(config: AdapterConfig) -> MvResult<Self> {
        Ok(Self {
            config,
            inbound_buffer: Mutex::new(Vec::new()),
            outbound_buffer: Mutex::new(Vec::new()),
            last_send: Mutex::new(None),
            last_receive: Mutex::new(None),
            last_error: Mutex::new(None),
        })
    }

    /// Display name for the CLI user.
    pub fn display_name(&self) -> &str {
        self.config.get_setting("name").unwrap_or("user")
    }

    /// Push a user message into the inbound buffer (called by REPL input loop).
    pub fn push_message(&self, content: impl Into<String>) {
        let content = content.into();
        let msg = AdapterInboundMessage {
            external_id: format!("cli-{}", Utc::now().timestamp_nanos_opt().unwrap_or(0)),
            channel: "cli".to_string(),
            sender: self.display_name().to_string(),
            content,
            thread_id: None,
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        };
        self.inbound_buffer.lock().unwrap().push(msg);
        *self.last_receive.lock().unwrap() = Some(Utc::now());
    }

    /// Pop the next response from the outbound buffer (called by REPL output loop).
    pub fn pop_response(&self) -> Option<String> {
        let mut buf = self.outbound_buffer.lock().unwrap();
        if buf.is_empty() {
            None
        } else {
            Some(buf.remove(0).content)
        }
    }

    /// Check if there are pending responses.
    pub fn has_responses(&self) -> bool {
        !self.outbound_buffer.lock().unwrap().is_empty()
    }
}

#[async_trait]
impl ExternalAdapter for CliChatAdapter {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::CliChat
    }

    async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
        self.outbound_buffer
            .lock()
            .unwrap()
            .push(message.clone());
        *self.last_send.lock().unwrap() = Some(Utc::now());
        Ok(())
    }

    async fn poll(&self, _cursor: Option<&str>) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
        let messages: Vec<AdapterInboundMessage> =
            self.inbound_buffer.lock().unwrap().drain(..).collect();
        let count = messages.len();
        Ok((messages, count.to_string()))
    }

    async fn health_check(&self) -> MvResult<bool> {
        Ok(true)
    }

    fn status(&self) -> AdapterStatus {
        let error = self.last_error.lock().unwrap().clone();
        let last_send = *self.last_send.lock().unwrap();
        let last_receive = *self.last_receive.lock().unwrap();
        AdapterStatus {
            adapter_type: AdapterType::CliChat,
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

    fn cli_config() -> AdapterConfig {
        AdapterConfig::new(AdapterType::CliChat, "test-cli")
    }

    fn cli_config_with_name() -> AdapterConfig {
        cli_config().with_setting("name", "alice")
    }

    #[test]
    fn new_succeeds_with_no_settings() {
        let adapter = CliChatAdapter::new(cli_config());
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "test-cli");
    }

    #[test]
    fn display_name_defaults_to_user() {
        let adapter = CliChatAdapter::new(cli_config()).unwrap();
        assert_eq!(adapter.display_name(), "user");
    }

    #[test]
    fn display_name_uses_config() {
        let adapter = CliChatAdapter::new(cli_config_with_name()).unwrap();
        assert_eq!(adapter.display_name(), "alice");
    }

    #[tokio::test]
    async fn adapter_type_is_cli_chat() {
        let adapter = CliChatAdapter::new(cli_config()).unwrap();
        assert_eq!(adapter.adapter_type(), AdapterType::CliChat);
    }

    #[tokio::test]
    async fn push_and_poll() {
        let adapter = CliChatAdapter::new(cli_config()).unwrap();

        adapter.push_message("hello from user");
        adapter.push_message("second message");

        let (messages, cursor) = adapter.poll(None).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].content, "hello from user");
        assert_eq!(messages[0].sender, "user");
        assert_eq!(messages[0].channel, "cli");
        assert_eq!(messages[1].content, "second message");
        assert_eq!(cursor, "2");

        // Buffer drained
        let (messages, _) = adapter.poll(None).await.unwrap();
        assert!(messages.is_empty());
    }

    #[tokio::test]
    async fn send_and_pop_response() {
        let adapter = CliChatAdapter::new(cli_config()).unwrap();

        assert!(!adapter.has_responses());
        assert!(adapter.pop_response().is_none());

        let msg = AdapterOutboundMessage {
            channel: "cli".into(),
            content: "assistant reply".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        adapter.send(&msg).await.unwrap();

        assert!(adapter.has_responses());
        let resp = adapter.pop_response().unwrap();
        assert_eq!(resp, "assistant reply");
        assert!(!adapter.has_responses());
    }

    #[tokio::test]
    async fn health_check_always_true() {
        let adapter = CliChatAdapter::new(cli_config()).unwrap();
        assert!(adapter.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn initial_status_connected() {
        let adapter = CliChatAdapter::new(cli_config()).unwrap();
        let status = adapter.status();
        assert!(status.connected);
        assert!(status.error.is_none());
        assert!(status.last_send.is_none());
        assert!(status.last_receive.is_none());
        assert_eq!(status.adapter_type, AdapterType::CliChat);
    }
}
