//! Per-platform [`Channel`] implementations that wrap an [`ExternalAdapter`]
//! and provide platform-specific normalization logic.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use hx_core::MvResult;

use crate::adapters::{
    AdapterInboundMessage, AdapterOutboundMessage, ExternalAdapter,
};

use super::{
    Channel, ChannelMessage, ChannelRef, MessageContent, SenderIdentity, ThreadRef,
};

// ---------------------------------------------------------------------------
// Helper: default normalization
// ---------------------------------------------------------------------------

fn default_normalize(
    raw: &AdapterInboundMessage,
    platform: &str,
) -> ChannelMessage {
    let thread_ref = raw.thread_id.as_ref().map(|tid| ThreadRef {
        thread_id: tid.clone(),
        parent_message_id: Some(tid.clone()),
    });

    ChannelMessage {
        id: Uuid::now_v7().to_string(),
        platform: platform.to_string(),
        external_id: raw.external_id.clone(),
        channel_ref: ChannelRef {
            platform_id: raw.channel.clone(),
            name: None,
        },
        sender: SenderIdentity {
            platform_id: raw.sender.clone(),
            display_name: raw.sender.clone(),
            is_bot: false,
        },
        content: MessageContent::text(&raw.content),
        thread_ref,
        timestamp: raw.timestamp,
        metadata: raw.metadata.clone(),
    }
}

fn make_outbound(message: &ChannelMessage, thread_id: Option<String>) -> AdapterOutboundMessage {
    AdapterOutboundMessage {
        channel: message.channel_ref.platform_id.clone(),
        content: message.content.text.clone(),
        thread_id,
        metadata: message.metadata.clone(),
    }
}

// ---------------------------------------------------------------------------
// Slack
// ---------------------------------------------------------------------------

pub struct SlackChannel {
    adapter: Arc<dyn ExternalAdapter>,
}

impl SlackChannel {
    pub fn new(adapter: Arc<dyn ExternalAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl Channel for SlackChannel {
    fn adapter(&self) -> &dyn ExternalAdapter {
        self.adapter.as_ref()
    }

    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage {
        let mut msg = default_normalize(raw, "slack");
        // Slack thread_ts is both the thread ID and parent message reference
        if let Some(ref tid) = raw.thread_id {
            msg.thread_ref = Some(ThreadRef {
                thread_id: tid.clone(),
                parent_message_id: Some(tid.clone()),
            });
        }
        msg
    }

    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()> {
        let thread_id = message.thread_ref.as_ref().map(|t| t.thread_id.clone());
        self.adapter.send(&make_outbound(message, thread_id)).await
    }

    async fn reply(&self, original: &ChannelMessage, content: &str) -> MvResult<()> {
        let thread_id = original
            .thread_ref
            .as_ref()
            .map(|t| t.thread_id.clone())
            .or_else(|| Some(original.external_id.clone()));
        let msg = AdapterOutboundMessage {
            channel: original.channel_ref.platform_id.clone(),
            content: content.to_string(),
            thread_id,
            metadata: HashMap::new(),
        };
        self.adapter.send(&msg).await
    }

    fn supports_threads(&self) -> bool {
        true
    }

    fn supports_embeds(&self) -> bool {
        false
    }

    fn max_message_length(&self) -> usize {
        40000
    }
}

// ---------------------------------------------------------------------------
// Discord
// ---------------------------------------------------------------------------

pub struct DiscordChannel {
    adapter: Arc<dyn ExternalAdapter>,
}

impl DiscordChannel {
    pub fn new(adapter: Arc<dyn ExternalAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl Channel for DiscordChannel {
    fn adapter(&self) -> &dyn ExternalAdapter {
        self.adapter.as_ref()
    }

    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage {
        let mut msg = default_normalize(raw, "discord");
        // Discord uses message_reference for replies
        if let Some(ref tid) = raw.thread_id {
            msg.thread_ref = Some(ThreadRef {
                thread_id: tid.clone(),
                parent_message_id: Some(tid.clone()),
            });
        }
        msg
    }

    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()> {
        let thread_id = message.thread_ref.as_ref().map(|t| t.thread_id.clone());
        self.adapter.send(&make_outbound(message, thread_id)).await
    }

    async fn reply(&self, original: &ChannelMessage, content: &str) -> MvResult<()> {
        let thread_id = Some(original.external_id.clone());
        let msg = AdapterOutboundMessage {
            channel: original.channel_ref.platform_id.clone(),
            content: content.to_string(),
            thread_id,
            metadata: HashMap::new(),
        };
        self.adapter.send(&msg).await
    }

    fn supports_threads(&self) -> bool {
        true
    }

    fn supports_embeds(&self) -> bool {
        true
    }

    fn max_message_length(&self) -> usize {
        2000
    }
}

// ---------------------------------------------------------------------------
// Telegram
// ---------------------------------------------------------------------------

pub struct TelegramChannel {
    adapter: Arc<dyn ExternalAdapter>,
}

impl TelegramChannel {
    pub fn new(adapter: Arc<dyn ExternalAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl Channel for TelegramChannel {
    fn adapter(&self) -> &dyn ExternalAdapter {
        self.adapter.as_ref()
    }

    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage {
        let mut msg = default_normalize(raw, "telegram");
        // Telegram uses reply_to_message for threads
        if let Some(ref tid) = raw.thread_id {
            msg.thread_ref = Some(ThreadRef {
                thread_id: tid.clone(),
                parent_message_id: Some(tid.clone()),
            });
        }
        // Extract username vs display name from metadata if available
        if let Some(display) = raw.metadata.get("display_name") {
            msg.sender.display_name = display.clone();
        }
        msg
    }

    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()> {
        let thread_id = message.thread_ref.as_ref().map(|t| t.thread_id.clone());
        self.adapter.send(&make_outbound(message, thread_id)).await
    }

    async fn reply(&self, original: &ChannelMessage, content: &str) -> MvResult<()> {
        let thread_id = Some(original.external_id.clone());
        let msg = AdapterOutboundMessage {
            channel: original.channel_ref.platform_id.clone(),
            content: content.to_string(),
            thread_id,
            metadata: HashMap::new(),
        };
        self.adapter.send(&msg).await
    }

    fn supports_threads(&self) -> bool {
        true
    }

    fn max_message_length(&self) -> usize {
        4096
    }
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

pub struct MatrixChannel {
    adapter: Arc<dyn ExternalAdapter>,
}

impl MatrixChannel {
    pub fn new(adapter: Arc<dyn ExternalAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl Channel for MatrixChannel {
    fn adapter(&self) -> &dyn ExternalAdapter {
        self.adapter.as_ref()
    }

    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage {
        let mut msg = default_normalize(raw, "matrix");
        // Matrix uses m.relates_to for threads
        if let Some(ref tid) = raw.thread_id {
            msg.thread_ref = Some(ThreadRef {
                thread_id: tid.clone(),
                parent_message_id: Some(tid.clone()),
            });
        }
        // Matrix sender is @user:homeserver format
        msg.sender.display_name = raw
            .sender
            .strip_prefix('@')
            .and_then(|s| s.split(':').next())
            .unwrap_or(&raw.sender)
            .to_string();
        msg
    }

    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()> {
        let thread_id = message.thread_ref.as_ref().map(|t| t.thread_id.clone());
        self.adapter.send(&make_outbound(message, thread_id)).await
    }

    async fn reply(&self, original: &ChannelMessage, content: &str) -> MvResult<()> {
        let thread_id = Some(original.external_id.clone());
        let msg = AdapterOutboundMessage {
            channel: original.channel_ref.platform_id.clone(),
            content: content.to_string(),
            thread_id,
            metadata: HashMap::new(),
        };
        self.adapter.send(&msg).await
    }

    fn supports_threads(&self) -> bool {
        true
    }

    fn max_message_length(&self) -> usize {
        // Matrix spec recommends 65536 for event content
        65536
    }
}

// ---------------------------------------------------------------------------
// Email
// ---------------------------------------------------------------------------

pub struct EmailChannel {
    adapter: Arc<dyn ExternalAdapter>,
}

impl EmailChannel {
    pub fn new(adapter: Arc<dyn ExternalAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl Channel for EmailChannel {
    fn adapter(&self) -> &dyn ExternalAdapter {
        self.adapter.as_ref()
    }

    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage {
        let mut msg = default_normalize(raw, "email");
        // Email "channel" is the recipient address; sender is the from address
        if let Some(subject) = raw.metadata.get("subject") {
            msg.metadata
                .insert("subject".to_string(), subject.clone());
        }
        msg
    }

    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()> {
        self.adapter.send(&make_outbound(message, None)).await
    }

    async fn reply(&self, original: &ChannelMessage, content: &str) -> MvResult<()> {
        let mut metadata = HashMap::new();
        if let Some(subject) = original.metadata.get("subject") {
            metadata.insert("subject".to_string(), format!("Re: {subject}"));
        }
        let msg = AdapterOutboundMessage {
            channel: original.sender.platform_id.clone(),
            content: content.to_string(),
            thread_id: None,
            metadata,
        };
        self.adapter.send(&msg).await
    }

    fn supports_threads(&self) -> bool {
        false
    }

    fn max_message_length(&self) -> usize {
        0 // unlimited for email
    }
}

// ---------------------------------------------------------------------------
// Webhook
// ---------------------------------------------------------------------------

pub struct WebhookChannel {
    adapter: Arc<dyn ExternalAdapter>,
}

impl WebhookChannel {
    pub fn new(adapter: Arc<dyn ExternalAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl Channel for WebhookChannel {
    fn adapter(&self) -> &dyn ExternalAdapter {
        self.adapter.as_ref()
    }

    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage {
        default_normalize(raw, "webhook")
    }

    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()> {
        self.adapter.send(&make_outbound(message, None)).await
    }

    async fn reply(&self, original: &ChannelMessage, content: &str) -> MvResult<()> {
        let msg = AdapterOutboundMessage {
            channel: original.channel_ref.platform_id.clone(),
            content: content.to_string(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        self.adapter.send(&msg).await
    }

    fn supports_threads(&self) -> bool {
        false
    }

    fn max_message_length(&self) -> usize {
        0
    }
}

// ---------------------------------------------------------------------------
// CLI Chat
// ---------------------------------------------------------------------------

pub struct CliChatChannel {
    adapter: Arc<dyn ExternalAdapter>,
}

impl CliChatChannel {
    pub fn new(adapter: Arc<dyn ExternalAdapter>) -> Self {
        Self { adapter }
    }
}

#[async_trait]
impl Channel for CliChatChannel {
    fn adapter(&self) -> &dyn ExternalAdapter {
        self.adapter.as_ref()
    }

    fn normalize_inbound(&self, raw: &AdapterInboundMessage) -> ChannelMessage {
        default_normalize(raw, "cli")
    }

    async fn send_message(&self, message: &ChannelMessage) -> MvResult<()> {
        self.adapter.send(&make_outbound(message, None)).await
    }

    async fn reply(&self, _original: &ChannelMessage, content: &str) -> MvResult<()> {
        let msg = AdapterOutboundMessage {
            channel: "cli".to_string(),
            content: content.to_string(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        self.adapter.send(&msg).await
    }

    fn supports_threads(&self) -> bool {
        false
    }

    fn max_message_length(&self) -> usize {
        0
    }
}
