use async_trait::async_trait;
use hx_core::{HxError, MvResult};

use crate::notifications::{Notification, NotificationChannel, NotificationChannelType};

/// Notification channel that delivers notifications via HTTP POST to a webhook URL.
pub struct WebhookNotificationChannel {
    webhook_url: String,
    client: reqwest::Client,
}

impl WebhookNotificationChannel {
    pub fn new(webhook_url: impl Into<String>) -> Self {
        Self {
            webhook_url: webhook_url.into(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    pub fn url(&self) -> &str {
        &self.webhook_url
    }
}

#[async_trait]
impl NotificationChannel for WebhookNotificationChannel {
    fn name(&self) -> &str {
        "webhook"
    }

    fn channel_type(&self) -> NotificationChannelType {
        NotificationChannelType::Webhook
    }

    async fn send(&self, notification: &Notification) -> MvResult<()> {
        let resp = self
            .client
            .post(&self.webhook_url)
            .header("Content-Type", "application/json")
            .json(notification)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("webhook request failed: {e}")))?;

        if !resp.status().is_success() {
            return Err(HxError::Internal(format!(
                "webhook returned status {}",
                resp.status()
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notifications::Severity;

    #[test]
    fn webhook_channel_metadata() {
        let channel = WebhookNotificationChannel::new("https://example.com/hook");
        assert_eq!(channel.name(), "webhook");
        assert_eq!(channel.channel_type(), NotificationChannelType::Webhook);
        assert_eq!(channel.url(), "https://example.com/hook");
    }

    #[tokio::test]
    async fn webhook_send_to_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(r#"{"ok":true}"#)
            .create_async()
            .await;

        let channel = WebhookNotificationChannel::new(format!("{}/hook", server.url()));
        let notif = Notification::new("Test", "Body", Severity::Info);
        channel.send(&notif).await.unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn webhook_send_failure_returns_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/hook")
            .with_status(500)
            .with_body("internal error")
            .create_async()
            .await;

        let channel = WebhookNotificationChannel::new(format!("{}/hook", server.url()));
        let notif = Notification::new("Test", "Body", Severity::Error);
        let result = channel.send(&notif).await;
        assert!(result.is_err());

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn webhook_connection_failure_returns_error() {
        // Use a URL that won't connect
        let channel = WebhookNotificationChannel::new("http://127.0.0.1:1/nonexistent");
        let notif = Notification::new("Test", "Body", Severity::Critical);
        let result = channel.send(&notif).await;
        assert!(result.is_err());
    }
}
