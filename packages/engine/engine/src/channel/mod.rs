//! Unified channel abstraction over messaging adapters.
//!
//! Provides a higher-level [`Channel`] trait that normalizes platform-specific
//! inbound messages into a common [`ChannelMessage`] structure and offers
//! a uniform send/reply interface.

pub mod normalize;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use hx_core::MvResult;

use super::adapters::{
    AdapterInboundMessage, AdapterType, ExternalAdapter,
};

// ---------------------------------------------------------------------------
// Channel Message Types
// ---------------------------------------------------------------------------

/// A normalized message from any platform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelMessage {
    pub id: String,
    pub platform: String,
    pub external_id: String,
    pub channel_ref: ChannelRef,
    pub sender: SenderIdentity,
    pub content: MessageContent,
    pub thread_ref: Option<ThreadRef>,
    pub timestamp: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

/// Content of a channel message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageContent {
    pub text: String,
    pub attachments: Vec<Attachment>,
    pub embeds: Vec<Embed>,
}

impl MessageContent {
    pub fn text(s: impl Into<String>) -> Self {
        Self {
            text: s.into(),
            attachments: Vec::new(),
            embeds: Vec::new(),
        }
    }
}

/// Identity of the message sender.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SenderIdentity {
    pub platform_id: String,
    pub display_name: String,
    pub is_bot: bool,
}

/// Reference to a channel/room/conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRef {
    pub platform_id: String,
    pub name: Option<String>,
}

/// Reference to a thread within a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadRef {
    pub thread_id: String,
    pub parent_message_id: Option<String>,
}

/// A file or media attachment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    pub filename: String,
    pub content_type: Option<String>,
    pub url: Option<String>,
    pub size_bytes: Option<u64>,
}

/// A rich embed (e.g., Discord embed, Slack block).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embed {
    pub title: Option<String>,
    pub description: Option<String>,
    pub url: Option<String>,
    pub color: Option<u32>,
}

// ---------------------------------------------------------------------------
// Channel Trait
// ---------------------------------------------------------------------------

/// Higher-level abstraction over an [`ExternalAdapter`], providing
/// normalized message types and platform-aware send/reply methods.
#[async_trait]
pub trait Channel: Send + Sync {
    /// Get the underlying adapter.
    fn adapter(&self) -> &dyn ExternalAdapter;

    /// Normalize a raw inbound message into a [`ChannelMessage`].
    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage;

    /// Send a channel message through the underlying adapter.
    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()>;

    /// Reply to an existing message with plain text content.
    async fn reply(&self, original: &ChannelMessage, content: &str) -> MvResult<()>;

    /// Whether this platform supports threaded conversations.
    fn supports_threads(&self) -> bool {
        true
    }

    /// Whether this platform supports rich embeds.
    fn supports_embeds(&self) -> bool {
        false
    }

    /// Maximum message length (0 = unlimited).
    fn max_message_length(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Wrap an [`ExternalAdapter`] in the appropriate platform-specific [`Channel`]
/// implementation based on its [`AdapterType`].
pub fn wrap_adapter(
    adapter: Arc<dyn ExternalAdapter>,
    adapter_type: AdapterType,
) -> Box<dyn Channel> {
    match adapter_type {
        AdapterType::Slack => Box::new(normalize::SlackChannel::new(adapter)),
        AdapterType::Discord => Box::new(normalize::DiscordChannel::new(adapter)),
        AdapterType::Email => Box::new(normalize::EmailChannel::new(adapter)),
        AdapterType::Telegram => Box::new(normalize::TelegramChannel::new(adapter)),
        AdapterType::Matrix => Box::new(normalize::MatrixChannel::new(adapter)),
        AdapterType::Webhook => Box::new(normalize::WebhookChannel::new(adapter)),
        AdapterType::CliChat => Box::new(normalize::CliChatChannel::new(adapter)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapters::{AdapterConfig, AdapterOutboundMessage, AdapterStatus};
    use std::sync::Mutex;

    /// Minimal mock adapter for channel tests.
    struct MockAdapter {
        name: String,
        adapter_type: AdapterType,
        sent: Arc<Mutex<Vec<AdapterOutboundMessage>>>,
    }

    impl MockAdapter {
        fn new(name: &str, at: AdapterType) -> Self {
            Self {
                name: name.into(),
                adapter_type: at,
                sent: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    #[async_trait]
    impl ExternalAdapter for MockAdapter {
        fn name(&self) -> &str {
            &self.name
        }
        fn adapter_type(&self) -> AdapterType {
            self.adapter_type
        }
        async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
            self.sent.lock().unwrap().push(message.clone());
            Ok(())
        }
        async fn poll(
            &self,
            _cursor: Option<&str>,
        ) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
            Ok((vec![], "0".into()))
        }
        async fn health_check(&self) -> MvResult<bool> {
            Ok(true)
        }
        fn status(&self) -> AdapterStatus {
            AdapterStatus {
                adapter_type: self.adapter_type,
                name: self.name.clone(),
                connected: true,
                last_send: None,
                last_receive: None,
                error: None,
            }
        }
    }

    fn make_raw_message() -> AdapterInboundMessage {
        AdapterInboundMessage {
            external_id: "ext-1".into(),
            channel: "#general".into(),
            sender: "user42".into(),
            content: "hello world".into(),
            thread_id: Some("thread-1".into()),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn wrap_adapter_slack() {
        let adapter = Arc::new(MockAdapter::new("s", AdapterType::Slack));
        let channel = wrap_adapter(adapter, AdapterType::Slack);
        assert!(channel.supports_threads());
        assert_eq!(channel.max_message_length(), 40000);
    }

    #[test]
    fn wrap_adapter_discord() {
        let adapter = Arc::new(MockAdapter::new("d", AdapterType::Discord));
        let channel = wrap_adapter(adapter, AdapterType::Discord);
        assert!(channel.supports_embeds());
        assert_eq!(channel.max_message_length(), 2000);
    }

    #[test]
    fn wrap_adapter_telegram() {
        let adapter = Arc::new(MockAdapter::new("t", AdapterType::Telegram));
        let channel = wrap_adapter(adapter, AdapterType::Telegram);
        assert_eq!(channel.max_message_length(), 4096);
    }

    #[test]
    fn wrap_adapter_matrix() {
        let adapter = Arc::new(MockAdapter::new("m", AdapterType::Matrix));
        let channel = wrap_adapter(adapter, AdapterType::Matrix);
        assert!(channel.supports_threads());
    }

    #[test]
    fn wrap_adapter_webhook() {
        let adapter = Arc::new(MockAdapter::new("w", AdapterType::Webhook));
        let channel = wrap_adapter(adapter, AdapterType::Webhook);
        assert!(!channel.supports_threads());
        assert_eq!(channel.max_message_length(), 0);
    }

    #[test]
    fn wrap_adapter_cli_chat() {
        let adapter = Arc::new(MockAdapter::new("c", AdapterType::CliChat));
        let channel = wrap_adapter(adapter, AdapterType::CliChat);
        assert!(!channel.supports_threads());
    }

    #[test]
    fn normalize_inbound_sets_platform() {
        let adapter = Arc::new(MockAdapter::new("s", AdapterType::Slack));
        let channel = wrap_adapter(adapter, AdapterType::Slack);
        let raw = make_raw_message();
        let msg = channel.normalize_inbound(&raw);
        assert_eq!(msg.platform, "slack");
        assert_eq!(msg.external_id, "ext-1");
        assert_eq!(msg.sender.platform_id, "user42");
        assert_eq!(msg.content.text, "hello world");
        assert!(msg.thread_ref.is_some());
    }
}
