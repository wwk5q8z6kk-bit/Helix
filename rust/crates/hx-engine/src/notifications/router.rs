use std::sync::Arc;

use chrono::{DateTime, Utc};
use hx_core::MvResult;

use super::alerts::AlertRuleStore;
use super::{Notification, NotificationChannel, NotificationChannelType, Severity};

/// Routes notifications to appropriate channels based on alert rules.
pub struct NotificationRouter {
    channels: Vec<Arc<dyn NotificationChannel>>,
    alert_store: Arc<AlertRuleStore>,
}

impl NotificationRouter {
    pub fn new(alert_store: Arc<AlertRuleStore>) -> Self {
        Self {
            channels: Vec::new(),
            alert_store,
        }
    }

    pub fn add_channel(&mut self, channel: Arc<dyn NotificationChannel>) {
        self.channels.push(channel);
    }

    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }

    pub fn alert_store(&self) -> &Arc<AlertRuleStore> {
        &self.alert_store
    }

    /// Route a notification to all registered channels that match the
    /// notification's target channel types (determined by alert rules).
    /// Returns the names of channels that received the notification.
    pub async fn route(&self, notification: &Notification) -> MvResult<Vec<String>> {
        let mut sent_to = Vec::new();

        for channel in &self.channels {
            match channel.send(notification).await {
                Ok(()) => sent_to.push(channel.name().to_string()),
                Err(e) => {
                    tracing::warn!(
                        channel = %channel.name(),
                        notification_id = %notification.id,
                        error = %e,
                        "failed to route notification to channel"
                    );
                }
            }
        }

        Ok(sent_to)
    }

    /// Route a notification only to channels matching the specified types.
    pub async fn route_to_channels(
        &self,
        notification: &Notification,
        channel_types: &[NotificationChannelType],
    ) -> MvResult<Vec<String>> {
        let mut sent_to = Vec::new();

        for channel in &self.channels {
            if channel_types.contains(&channel.channel_type()) {
                match channel.send(notification).await {
                    Ok(()) => sent_to.push(channel.name().to_string()),
                    Err(e) => {
                        tracing::warn!(
                            channel = %channel.name(),
                            notification_id = %notification.id,
                            error = %e,
                            "failed to route notification to channel"
                        );
                    }
                }
            }
        }

        Ok(sent_to)
    }

    /// Evaluate all alert rules against an event, create notifications for
    /// matching rules, and route them to the appropriate channels.
    pub async fn evaluate_and_notify(
        &self,
        event_type: &str,
        event_data: &serde_json::Value,
    ) -> MvResult<()> {
        let now = Utc::now();
        let matching_rules = self.alert_store.evaluate(event_type, event_data, now).await;

        for rule in matching_rules {
            let notification = Notification::new(
                &rule.name,
                format!(
                    "Alert triggered: {} (event: {event_type})",
                    rule.name
                ),
                rule.severity,
            )
            .with_source("alert-system");

            // Check quiet hours
            if let Some(ref quiet) = rule.quiet_hours {
                if is_in_quiet_hours(quiet, now) {
                    tracing::debug!(
                        rule = %rule.name,
                        "skipping notification during quiet hours"
                    );
                    continue;
                }
            }

            // Route to the rule's configured channels
            self.route_to_channels(&notification, &rule.channels).await?;

            // Update last triggered time
            self.alert_store.set_last_triggered(rule.id, now).await;
        }

        Ok(())
    }
}

/// Check if the current time falls within quiet hours.
/// Uses a simple hour-based check (does not do timezone conversion — uses UTC).
pub fn is_in_quiet_hours(quiet: &super::alerts::QuietHours, now: DateTime<Utc>) -> bool {
    let hour = now.format("%H").to_string().parse::<u8>().unwrap_or(0);

    if quiet.start_hour <= quiet.end_hour {
        // Simple range, e.g., 22..06 would NOT hit this branch
        // This handles e.g., 09..17
        hour >= quiet.start_hour && hour < quiet.end_hour
    } else {
        // Wraps around midnight, e.g., 22..06
        hour >= quiet.start_hour || hour < quiet.end_hour
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notifications::alerts::{AlertCondition, AlertRule, QuietHours};
    use crate::notifications::channels::in_app::InAppChannel;

    fn make_router_with_in_app() -> (Arc<InAppChannel>, NotificationRouter) {
        let store = Arc::new(AlertRuleStore::new());
        let in_app = Arc::new(InAppChannel::new());
        let mut router = NotificationRouter::new(store);
        router.add_channel(Arc::clone(&in_app) as Arc<dyn NotificationChannel>);
        (in_app, router)
    }

    #[tokio::test]
    async fn route_sends_to_all_channels() {
        let (in_app, router) = make_router_with_in_app();
        let notif = Notification::new("Test", "Body", Severity::Info);

        let sent_to = router.route(&notif).await.unwrap();
        assert_eq!(sent_to, vec!["in-app"]);
        assert_eq!(in_app.count().await, 1);
    }

    #[tokio::test]
    async fn route_to_channels_filters() {
        let (in_app, router) = make_router_with_in_app();
        let notif = Notification::new("Test", "Body", Severity::Info);

        // Route only to Slack (which we don't have) — in_app should not receive
        let sent_to = router
            .route_to_channels(&notif, &[NotificationChannelType::Slack])
            .await
            .unwrap();
        assert!(sent_to.is_empty());
        assert_eq!(in_app.count().await, 0);

        // Route to InApp — should work
        let sent_to = router
            .route_to_channels(&notif, &[NotificationChannelType::InApp])
            .await
            .unwrap();
        assert_eq!(sent_to, vec!["in-app"]);
        assert_eq!(in_app.count().await, 1);
    }

    #[tokio::test]
    async fn evaluate_and_notify_triggers_matching_rules() {
        let store = Arc::new(AlertRuleStore::new());
        let in_app = Arc::new(InAppChannel::new());
        let mut router = NotificationRouter::new(Arc::clone(&store));
        router.add_channel(Arc::clone(&in_app) as Arc<dyn NotificationChannel>);

        let rule = AlertRule::new(
            "job-fail-alert",
            AlertCondition::JobFailed { job_type: None },
            Severity::Error,
        )
        .with_cooldown(0)
        .with_channels(vec![NotificationChannelType::InApp]);
        store.add(rule).await;

        router
            .evaluate_and_notify("job_failed", &serde_json::json!({}))
            .await
            .unwrap();

        assert_eq!(in_app.count().await, 1);
        let notifs = in_app.list(None, None, 10).await;
        assert_eq!(notifs[0].title, "job-fail-alert");
    }

    #[tokio::test]
    async fn evaluate_and_notify_skips_non_matching() {
        let store = Arc::new(AlertRuleStore::new());
        let in_app = Arc::new(InAppChannel::new());
        let mut router = NotificationRouter::new(Arc::clone(&store));
        router.add_channel(Arc::clone(&in_app) as Arc<dyn NotificationChannel>);

        let rule = AlertRule::new(
            "job-fail-alert",
            AlertCondition::JobFailed { job_type: None },
            Severity::Error,
        );
        store.add(rule).await;

        router
            .evaluate_and_notify("some_other_event", &serde_json::json!({}))
            .await
            .unwrap();

        assert_eq!(in_app.count().await, 0);
    }

    #[tokio::test]
    async fn evaluate_and_notify_respects_cooldown() {
        let store = Arc::new(AlertRuleStore::new());
        let in_app = Arc::new(InAppChannel::new());
        let mut router = NotificationRouter::new(Arc::clone(&store));
        router.add_channel(Arc::clone(&in_app) as Arc<dyn NotificationChannel>);

        let rule = AlertRule::new(
            "cooldown-test",
            AlertCondition::JobFailed { job_type: None },
            Severity::Warning,
        )
        .with_cooldown(3600)
        .with_channels(vec![NotificationChannelType::InApp]);
        store.add(rule).await;

        // First trigger
        router
            .evaluate_and_notify("job_failed", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(in_app.count().await, 1);

        // Second trigger — should be suppressed by cooldown
        router
            .evaluate_and_notify("job_failed", &serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(in_app.count().await, 1);
    }

    #[test]
    fn quiet_hours_simple_range() {
        let quiet = QuietHours {
            start_hour: 22,
            end_hour: 6,
            timezone: "UTC".into(),
        };

        // 23:00 UTC — should be in quiet hours
        let late_night = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
            .unwrap()
            .and_hms_opt(23, 0, 0)
            .unwrap();
        let late_night = DateTime::<Utc>::from_naive_utc_and_offset(late_night, Utc);
        assert!(is_in_quiet_hours(&quiet, late_night));

        // 03:00 UTC — should be in quiet hours
        let early_morning = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
            .unwrap()
            .and_hms_opt(3, 0, 0)
            .unwrap();
        let early_morning = DateTime::<Utc>::from_naive_utc_and_offset(early_morning, Utc);
        assert!(is_in_quiet_hours(&quiet, early_morning));

        // 12:00 UTC — should NOT be in quiet hours
        let midday = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap();
        let midday = DateTime::<Utc>::from_naive_utc_and_offset(midday, Utc);
        assert!(!is_in_quiet_hours(&quiet, midday));
    }

    #[test]
    fn quiet_hours_daytime_range() {
        let quiet = QuietHours {
            start_hour: 9,
            end_hour: 17,
            timezone: "UTC".into(),
        };

        let noon = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
            .unwrap()
            .and_hms_opt(12, 0, 0)
            .unwrap();
        let noon = DateTime::<Utc>::from_naive_utc_and_offset(noon, Utc);
        assert!(is_in_quiet_hours(&quiet, noon));

        let evening = chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
            .unwrap()
            .and_hms_opt(20, 0, 0)
            .unwrap();
        let evening = DateTime::<Utc>::from_naive_utc_and_offset(evening, Utc);
        assert!(!is_in_quiet_hours(&quiet, evening));
    }

    #[test]
    fn channel_count() {
        let store = Arc::new(AlertRuleStore::new());
        let mut router = NotificationRouter::new(store);
        assert_eq!(router.channel_count(), 0);
        router.add_channel(Arc::new(InAppChannel::new()));
        assert_eq!(router.channel_count(), 1);
    }

    #[tokio::test]
    async fn evaluate_and_notify_quiet_hours_suppresses() {
        let store = Arc::new(AlertRuleStore::new());
        let in_app = Arc::new(InAppChannel::new());
        let mut router = NotificationRouter::new(Arc::clone(&store));
        router.add_channel(Arc::clone(&in_app) as Arc<dyn NotificationChannel>);

        // Create rule with quiet hours that cover all 24 hours (0..24 = never quiet in simple range)
        // Use 0..23 which covers most hours
        let rule = AlertRule::new(
            "quiet-test",
            AlertCondition::JobFailed { job_type: None },
            Severity::Warning,
        )
        .with_cooldown(0)
        .with_channels(vec![NotificationChannelType::InApp])
        .with_quiet_hours(QuietHours {
            start_hour: 0,
            end_hour: 24,
            timezone: "UTC".into(),
        });
        store.add(rule).await;

        router
            .evaluate_and_notify("job_failed", &serde_json::json!({}))
            .await
            .unwrap();

        // Should be suppressed by quiet hours
        assert_eq!(in_app.count().await, 0);
    }
}
