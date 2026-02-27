use async_trait::async_trait;
use chrono::{DateTime, Utc};
use hx_core::MvResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub mod alerts;
pub mod channels;
pub mod outbound;
pub mod router;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    pub id: Uuid,
    pub title: String,
    pub body: String,
    pub severity: Severity,
    pub source: String,
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    #[serde(default)]
    pub read: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotificationChannelType {
    InApp,
    Email,
    Webhook,
    Slack,
}

#[async_trait]
pub trait NotificationChannel: Send + Sync {
    fn name(&self) -> &str;
    fn channel_type(&self) -> NotificationChannelType;
    async fn send(&self, notification: &Notification) -> MvResult<()>;
}

impl Notification {
    pub fn new(title: impl Into<String>, body: impl Into<String>, severity: Severity) -> Self {
        Self {
            id: Uuid::now_v7(),
            title: title.into(),
            body: body.into(),
            severity,
            source: String::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            read: false,
        }
    }

    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn mark_read(&mut self) {
        self.read = true;
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
            Severity::Critical => write!(f, "critical"),
        }
    }
}

impl std::fmt::Display for NotificationChannelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NotificationChannelType::InApp => write!(f, "in_app"),
            NotificationChannelType::Email => write!(f, "email"),
            NotificationChannelType::Webhook => write!(f, "webhook"),
            NotificationChannelType::Slack => write!(f, "slack"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_builder() {
        let notif = Notification::new("Test Title", "Test body", Severity::Warning)
            .with_source("test-system")
            .with_metadata("key", serde_json::json!("value"));

        assert_eq!(notif.title, "Test Title");
        assert_eq!(notif.body, "Test body");
        assert_eq!(notif.severity, Severity::Warning);
        assert_eq!(notif.source, "test-system");
        assert_eq!(notif.metadata.get("key"), Some(&serde_json::json!("value")));
        assert!(!notif.read);
    }

    #[test]
    fn notification_mark_read() {
        let mut notif = Notification::new("Title", "Body", Severity::Info);
        assert!(!notif.read);
        notif.mark_read();
        assert!(notif.read);
    }

    #[test]
    fn severity_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(Severity::Info).unwrap(),
            serde_json::json!("info")
        );
        assert_eq!(
            serde_json::to_value(Severity::Warning).unwrap(),
            serde_json::json!("warning")
        );
        assert_eq!(
            serde_json::to_value(Severity::Error).unwrap(),
            serde_json::json!("error")
        );
        assert_eq!(
            serde_json::to_value(Severity::Critical).unwrap(),
            serde_json::json!("critical")
        );
    }

    #[test]
    fn severity_roundtrips() {
        for sev in [Severity::Info, Severity::Warning, Severity::Error, Severity::Critical] {
            let json = serde_json::to_string(&sev).unwrap();
            let back: Severity = serde_json::from_str(&json).unwrap();
            assert_eq!(back, sev);
        }
    }

    #[test]
    fn channel_type_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(NotificationChannelType::InApp).unwrap(),
            serde_json::json!("in_app")
        );
        assert_eq!(
            serde_json::to_value(NotificationChannelType::Email).unwrap(),
            serde_json::json!("email")
        );
        assert_eq!(
            serde_json::to_value(NotificationChannelType::Webhook).unwrap(),
            serde_json::json!("webhook")
        );
        assert_eq!(
            serde_json::to_value(NotificationChannelType::Slack).unwrap(),
            serde_json::json!("slack")
        );
    }

    #[test]
    fn channel_type_roundtrips() {
        for ct in [
            NotificationChannelType::InApp,
            NotificationChannelType::Email,
            NotificationChannelType::Webhook,
            NotificationChannelType::Slack,
        ] {
            let json = serde_json::to_string(&ct).unwrap();
            let back: NotificationChannelType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, ct);
        }
    }

    #[test]
    fn notification_json_roundtrip() {
        let notif = Notification::new("Title", "Body", Severity::Error)
            .with_source("engine")
            .with_metadata("task_id", serde_json::json!("abc-123"));

        let json = serde_json::to_string(&notif).unwrap();
        let back: Notification = serde_json::from_str(&json).unwrap();
        assert_eq!(back.title, "Title");
        assert_eq!(back.severity, Severity::Error);
        assert_eq!(back.source, "engine");
    }

    #[test]
    fn severity_display() {
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Critical.to_string(), "critical");
    }

    #[test]
    fn channel_type_display() {
        assert_eq!(NotificationChannelType::InApp.to_string(), "in_app");
        assert_eq!(NotificationChannelType::Slack.to_string(), "slack");
    }
}
