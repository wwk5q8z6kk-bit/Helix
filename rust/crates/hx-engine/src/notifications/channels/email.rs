use async_trait::async_trait;
use hx_core::{HxError, MvResult};
use lettre::message::Mailbox;
use lettre::transport::smtp::authentication::Credentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::notifications::{Notification, NotificationChannel, NotificationChannelType};

/// Email notification channel. When SMTP configuration is provided, sends real
/// emails via the `lettre` crate. Otherwise falls back to logging (useful for
/// development and testing).
pub struct EmailNotificationChannel {
    from_address: String,
    to_addresses: Vec<String>,
    /// Optional SMTP host. When `None`, the channel uses placeholder logging.
    smtp_host: Option<String>,
    /// SMTP port (default: 587).
    smtp_port: u16,
    /// SMTP username for authentication.
    username: Option<String>,
    /// SMTP password for authentication.
    password: Option<String>,
    /// Captured notifications for testing.
    sent: Arc<RwLock<Vec<Notification>>>,
}

impl EmailNotificationChannel {
    /// Create a new email notification channel without SMTP configuration.
    /// Uses placeholder logging for send operations.
    pub fn new(from_address: impl Into<String>, to_addresses: Vec<String>) -> Self {
        Self {
            from_address: from_address.into(),
            to_addresses,
            smtp_host: None,
            smtp_port: 587,
            username: None,
            password: None,
            sent: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a new email notification channel with SMTP configuration for
    /// real email delivery.
    pub fn with_smtp(
        from_address: impl Into<String>,
        to_addresses: Vec<String>,
        smtp_host: impl Into<String>,
        smtp_port: u16,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            from_address: from_address.into(),
            to_addresses,
            smtp_host: Some(smtp_host.into()),
            smtp_port,
            username: Some(username.into()),
            password: Some(password.into()),
            sent: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn from_address(&self) -> &str {
        &self.from_address
    }

    pub fn to_addresses(&self) -> &[String] {
        &self.to_addresses
    }

    /// Returns `true` if SMTP is configured for real email delivery.
    pub fn has_smtp(&self) -> bool {
        self.smtp_host.is_some()
    }

    /// Get sent notifications (for testing).
    pub async fn sent_notifications(&self) -> Vec<Notification> {
        self.sent.read().await.clone()
    }
}

#[async_trait]
impl NotificationChannel for EmailNotificationChannel {
    fn name(&self) -> &str {
        "email"
    }

    fn channel_type(&self) -> NotificationChannelType {
        NotificationChannelType::Email
    }

    async fn send(&self, notification: &Notification) -> MvResult<()> {
        if let Some(ref smtp_host) = self.smtp_host {
            // Real SMTP delivery via lettre
            let from_mailbox: Mailbox = self.from_address.parse().map_err(|e| {
                HxError::Config(format!("invalid from_address '{}': {e}", self.from_address))
            })?;

            let username = self
                .username
                .as_deref()
                .ok_or_else(|| HxError::Config("SMTP username required".into()))?;
            let password = self
                .password
                .as_deref()
                .ok_or_else(|| HxError::Config("SMTP password required".into()))?;

            let creds = Credentials::new(username.to_string(), password.to_string());

            let transport = AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(smtp_host)
                .map_err(|e| HxError::Internal(format!("SMTP transport error: {e}")))?
                .port(self.smtp_port)
                .credentials(creds)
                .build();

            // Send to each recipient
            for to_addr in &self.to_addresses {
                let to_mailbox: Mailbox = to_addr
                    .parse()
                    .map_err(|e| HxError::Config(format!("invalid to address '{to_addr}': {e}")))?;

                let email = Message::builder()
                    .from(from_mailbox.clone())
                    .to(to_mailbox)
                    .subject(&notification.title)
                    .body(notification.body.clone())
                    .map_err(|e| HxError::Internal(format!("failed to build email: {e}")))?;

                transport
                    .send(email)
                    .await
                    .map_err(|e| HxError::Internal(format!("email send failed: {e}")))?;
            }
        } else {
            // Fallback: placeholder logging (development/testing)
            tracing::info!(
                from = %self.from_address,
                to = ?self.to_addresses,
                title = %notification.title,
                severity = %notification.severity,
                "email notification (placeholder): would send email"
            );
        }

        // Always capture for testing, regardless of mode
        self.sent.write().await.push(notification.clone());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notifications::Severity;

    #[test]
    fn email_channel_metadata() {
        let channel = EmailNotificationChannel::new(
            "noreply@helix.local",
            vec!["admin@helix.local".to_string()],
        );
        assert_eq!(channel.name(), "email");
        assert_eq!(channel.channel_type(), NotificationChannelType::Email);
        assert_eq!(channel.from_address(), "noreply@helix.local");
        assert_eq!(channel.to_addresses(), &["admin@helix.local"]);
    }

    #[tokio::test]
    async fn email_send_captures_notification() {
        let channel = EmailNotificationChannel::new(
            "noreply@helix.local",
            vec!["admin@helix.local".to_string()],
        );
        let notif = Notification::new("Email Test", "Body", Severity::Warning);
        channel.send(&notif).await.unwrap();

        let sent = channel.sent_notifications().await;
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].title, "Email Test");
    }

    #[tokio::test]
    async fn email_send_multiple() {
        let channel = EmailNotificationChannel::new(
            "from@test.com",
            vec!["to@test.com".to_string()],
        );
        channel
            .send(&Notification::new("First", "b", Severity::Info))
            .await
            .unwrap();
        channel
            .send(&Notification::new("Second", "b", Severity::Error))
            .await
            .unwrap();

        let sent = channel.sent_notifications().await;
        assert_eq!(sent.len(), 2);
    }

    #[tokio::test]
    async fn email_send_always_succeeds() {
        let channel = EmailNotificationChannel::new("a@b.com", vec![]);
        let notif = Notification::new("Test", "Body", Severity::Critical);
        // Should not error even with empty recipient list (it's a placeholder)
        assert!(channel.send(&notif).await.is_ok());
    }

    #[test]
    fn with_smtp_sets_config_fields() {
        let channel = EmailNotificationChannel::with_smtp(
            "noreply@helix.local",
            vec!["admin@helix.local".to_string()],
            "smtp.helix.local",
            465,
            "user@helix.local",
            "secret",
        );
        assert!(channel.has_smtp());
        assert_eq!(channel.smtp_host.as_deref(), Some("smtp.helix.local"));
        assert_eq!(channel.smtp_port, 465);
        assert_eq!(channel.username.as_deref(), Some("user@helix.local"));
        assert_eq!(channel.password.as_deref(), Some("secret"));
        assert_eq!(channel.from_address(), "noreply@helix.local");
        assert_eq!(channel.to_addresses(), &["admin@helix.local"]);
    }

    #[test]
    fn new_has_no_smtp() {
        let channel = EmailNotificationChannel::new(
            "noreply@helix.local",
            vec!["admin@helix.local".to_string()],
        );
        assert!(!channel.has_smtp());
        assert!(channel.smtp_host.is_none());
        assert_eq!(channel.smtp_port, 587);
        assert!(channel.username.is_none());
        assert!(channel.password.is_none());
    }
}
