use async_trait::async_trait;
use hx_core::{HxError, MvResult};

use crate::notifications::{Notification, NotificationChannel, NotificationChannelType, Severity};

/// Notification channel that posts to a Slack incoming webhook URL.
pub struct SlackNotificationChannel {
    webhook_url: String,
    client: reqwest::Client,
}

impl SlackNotificationChannel {
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

    /// Format a notification into a Slack message payload.
    fn format_slack_payload(notification: &Notification) -> serde_json::Value {
        let emoji = match notification.severity {
            Severity::Info => ":information_source:",
            Severity::Warning => ":warning:",
            Severity::Error => ":x:",
            Severity::Critical => ":rotating_light:",
        };

        serde_json::json!({
            "text": format!("{} *{}*\n{}", emoji, notification.title, notification.body),
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": format!(
                            "{} *{}*\n{}",
                            emoji, notification.title, notification.body
                        )
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": format!(
                                "Severity: {} | Source: {} | ID: {}",
                                notification.severity,
                                notification.source,
                                notification.id
                            )
                        }
                    ]
                }
            ]
        })
    }
}

#[async_trait]
impl NotificationChannel for SlackNotificationChannel {
    fn name(&self) -> &str {
        "slack"
    }

    fn channel_type(&self) -> NotificationChannelType {
        NotificationChannelType::Slack
    }

    async fn send(&self, notification: &Notification) -> MvResult<()> {
        let payload = Self::format_slack_payload(notification);

        let resp = self
            .client
            .post(&self.webhook_url)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("Slack webhook request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(HxError::Internal(format!(
                "Slack webhook returned status {status}: {body}"
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slack_channel_metadata() {
        let channel = SlackNotificationChannel::new("https://hooks.slack.com/services/test");
        assert_eq!(channel.name(), "slack");
        assert_eq!(channel.channel_type(), NotificationChannelType::Slack);
    }

    #[test]
    fn format_slack_payload_info() {
        let notif = Notification::new("Test Alert", "Something happened", Severity::Info)
            .with_source("helix");
        let payload = SlackNotificationChannel::format_slack_payload(&notif);

        let text = payload["text"].as_str().unwrap();
        assert!(text.contains(":information_source:"));
        assert!(text.contains("Test Alert"));
    }

    #[test]
    fn format_slack_payload_critical() {
        let notif = Notification::new("Critical!", "System down", Severity::Critical)
            .with_source("monitor");
        let payload = SlackNotificationChannel::format_slack_payload(&notif);

        let text = payload["text"].as_str().unwrap();
        assert!(text.contains(":rotating_light:"));
        assert!(text.contains("Critical!"));
    }

    #[tokio::test]
    async fn slack_send_to_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/services/hook")
            .with_status(200)
            .with_body("ok")
            .create_async()
            .await;

        let channel =
            SlackNotificationChannel::new(format!("{}/services/hook", server.url()));
        let notif = Notification::new("Test", "Body", Severity::Warning);
        channel.send(&notif).await.unwrap();

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn slack_send_failure_returns_error() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("POST", "/services/hook")
            .with_status(403)
            .with_body("invalid_token")
            .create_async()
            .await;

        let channel =
            SlackNotificationChannel::new(format!("{}/services/hook", server.url()));
        let notif = Notification::new("Test", "Body", Severity::Error);
        let result = channel.send(&notif).await;
        assert!(result.is_err());

        mock.assert_async().await;
    }
}
