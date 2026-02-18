use chrono::{DateTime, Utc};
use hx_core::{HxError, MvResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use super::Severity;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: Uuid,
    pub name: String,
    pub enabled: bool,
    pub condition: AlertCondition,
    pub channels: Vec<super::NotificationChannelType>,
    pub severity: Severity,
    pub cooldown_secs: u64,
    pub quiet_hours: Option<QuietHours>,
    pub last_triggered: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum AlertCondition {
    JobFailed {
        job_type: Option<String>,
    },
    BudgetExceeded {
        threshold_pct: f64,
    },
    SourceError {
        source_id: Option<String>,
    },
    CustomPattern {
        field: String,
        pattern: String,
    },
    IngestThreshold {
        min_per_hour: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietHours {
    pub start_hour: u8,
    pub end_hour: u8,
    pub timezone: String,
}

impl AlertRule {
    pub fn new(
        name: impl Into<String>,
        condition: AlertCondition,
        severity: Severity,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.into(),
            enabled: true,
            condition,
            channels: vec![super::NotificationChannelType::InApp],
            severity,
            cooldown_secs: 300,
            quiet_hours: None,
            last_triggered: None,
        }
    }

    pub fn with_channels(mut self, channels: Vec<super::NotificationChannelType>) -> Self {
        self.channels = channels;
        self
    }

    pub fn with_cooldown(mut self, secs: u64) -> Self {
        self.cooldown_secs = secs;
        self
    }

    pub fn with_quiet_hours(mut self, quiet: QuietHours) -> Self {
        self.quiet_hours = Some(quiet);
        self
    }
}

/// In-memory store for alert rules.
pub struct AlertRuleStore {
    rules: Arc<RwLock<HashMap<Uuid, AlertRule>>>,
}

impl AlertRuleStore {
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add(&self, rule: AlertRule) -> Uuid {
        let id = rule.id;
        self.rules.write().await.insert(id, rule);
        id
    }

    pub async fn remove(&self, id: Uuid) -> bool {
        self.rules.write().await.remove(&id).is_some()
    }

    pub async fn get(&self, id: Uuid) -> Option<AlertRule> {
        self.rules.read().await.get(&id).cloned()
    }

    pub async fn list(&self) -> Vec<AlertRule> {
        self.rules.read().await.values().cloned().collect()
    }

    pub async fn update(&self, id: Uuid, rule: AlertRule) -> MvResult<()> {
        let mut rules = self.rules.write().await;
        if !rules.contains_key(&id) {
            return Err(HxError::InvalidInput(format!("alert rule not found: {id}")));
        }
        rules.insert(id, rule);
        Ok(())
    }

    pub async fn set_last_triggered(&self, id: Uuid, when: DateTime<Utc>) {
        if let Some(rule) = self.rules.write().await.get_mut(&id) {
            rule.last_triggered = Some(when);
        }
    }

    /// Evaluate whether an alert rule matches a given event and is eligible to fire
    /// (i.e. not in cooldown).
    pub async fn evaluate(
        &self,
        event_type: &str,
        event_data: &serde_json::Value,
        now: DateTime<Utc>,
    ) -> Vec<AlertRule> {
        let rules = self.rules.read().await;
        rules
            .values()
            .filter(|rule| {
                rule.enabled
                    && !is_cooldown_active(rule, now)
                    && matches_condition(&rule.condition, event_type, event_data)
            })
            .cloned()
            .collect()
    }
}

/// Check whether a rule's cooldown period is still active.
pub fn is_cooldown_active(rule: &AlertRule, now: DateTime<Utc>) -> bool {
    if let Some(last) = rule.last_triggered {
        let elapsed = (now - last).num_seconds();
        elapsed >= 0 && (elapsed as u64) < rule.cooldown_secs
    } else {
        false
    }
}

/// Check whether an alert condition matches a given event type and data.
pub fn matches_condition(
    condition: &AlertCondition,
    event_type: &str,
    event_data: &serde_json::Value,
) -> bool {
    match condition {
        AlertCondition::JobFailed { job_type } => {
            if event_type != "job_failed" {
                return false;
            }
            match job_type {
                Some(jt) => event_data
                    .get("job_type")
                    .and_then(|v| v.as_str())
                    .map_or(false, |v| v == jt),
                None => true,
            }
        }
        AlertCondition::BudgetExceeded { threshold_pct } => {
            if event_type != "budget_update" {
                return false;
            }
            event_data
                .get("usage_pct")
                .and_then(|v| v.as_f64())
                .map_or(false, |pct| pct >= *threshold_pct)
        }
        AlertCondition::SourceError { source_id } => {
            if event_type != "source_error" {
                return false;
            }
            match source_id {
                Some(sid) => event_data
                    .get("source_id")
                    .and_then(|v| v.as_str())
                    .map_or(false, |v| v == sid),
                None => true,
            }
        }
        AlertCondition::CustomPattern { field, pattern } => {
            event_data
                .get(field)
                .and_then(|v| v.as_str())
                .map_or(false, |v| v.contains(pattern))
        }
        AlertCondition::IngestThreshold { min_per_hour } => {
            if event_type != "ingest_stats" {
                return false;
            }
            event_data
                .get("items_per_hour")
                .and_then(|v| v.as_u64())
                .map_or(false, |rate| rate < *min_per_hour)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alert_rule_builder() {
        let rule = AlertRule::new(
            "budget-alert",
            AlertCondition::BudgetExceeded { threshold_pct: 90.0 },
            Severity::Warning,
        )
        .with_cooldown(600)
        .with_channels(vec![
            super::super::NotificationChannelType::InApp,
            super::super::NotificationChannelType::Slack,
        ]);

        assert_eq!(rule.name, "budget-alert");
        assert!(rule.enabled);
        assert_eq!(rule.cooldown_secs, 600);
        assert_eq!(rule.channels.len(), 2);
    }

    #[test]
    fn alert_condition_serializes() {
        let cond = AlertCondition::JobFailed {
            job_type: Some("ingestion".into()),
        };
        let json = serde_json::to_value(&cond).unwrap();
        assert_eq!(json["type"], "job_failed");
        assert_eq!(json["job_type"], "ingestion");
    }

    #[test]
    fn matches_job_failed_any() {
        let cond = AlertCondition::JobFailed { job_type: None };
        assert!(matches_condition(&cond, "job_failed", &serde_json::json!({})));
        assert!(!matches_condition(&cond, "other", &serde_json::json!({})));
    }

    #[test]
    fn matches_job_failed_specific_type() {
        let cond = AlertCondition::JobFailed {
            job_type: Some("ingestion".into()),
        };
        assert!(matches_condition(
            &cond,
            "job_failed",
            &serde_json::json!({"job_type": "ingestion"})
        ));
        assert!(!matches_condition(
            &cond,
            "job_failed",
            &serde_json::json!({"job_type": "other"})
        ));
    }

    #[test]
    fn matches_budget_exceeded() {
        let cond = AlertCondition::BudgetExceeded { threshold_pct: 80.0 };
        assert!(matches_condition(
            &cond,
            "budget_update",
            &serde_json::json!({"usage_pct": 85.0})
        ));
        assert!(!matches_condition(
            &cond,
            "budget_update",
            &serde_json::json!({"usage_pct": 50.0})
        ));
    }

    #[test]
    fn matches_custom_pattern() {
        let cond = AlertCondition::CustomPattern {
            field: "message".into(),
            pattern: "error".into(),
        };
        assert!(matches_condition(
            &cond,
            "any_event",
            &serde_json::json!({"message": "an error occurred"})
        ));
        assert!(!matches_condition(
            &cond,
            "any_event",
            &serde_json::json!({"message": "all good"})
        ));
    }

    #[test]
    fn matches_ingest_threshold() {
        let cond = AlertCondition::IngestThreshold { min_per_hour: 100 };
        assert!(matches_condition(
            &cond,
            "ingest_stats",
            &serde_json::json!({"items_per_hour": 50})
        ));
        assert!(!matches_condition(
            &cond,
            "ingest_stats",
            &serde_json::json!({"items_per_hour": 200})
        ));
    }

    #[test]
    fn cooldown_not_active_when_never_triggered() {
        let rule = AlertRule::new(
            "test",
            AlertCondition::JobFailed { job_type: None },
            Severity::Error,
        );
        assert!(!is_cooldown_active(&rule, Utc::now()));
    }

    #[test]
    fn cooldown_active_within_period() {
        let mut rule = AlertRule::new(
            "test",
            AlertCondition::JobFailed { job_type: None },
            Severity::Error,
        )
        .with_cooldown(300);
        rule.last_triggered = Some(Utc::now());
        assert!(is_cooldown_active(&rule, Utc::now()));
    }

    #[test]
    fn cooldown_inactive_after_period() {
        let mut rule = AlertRule::new(
            "test",
            AlertCondition::JobFailed { job_type: None },
            Severity::Error,
        )
        .with_cooldown(1);

        rule.last_triggered = Some(Utc::now() - chrono::Duration::seconds(10));
        assert!(!is_cooldown_active(&rule, Utc::now()));
    }

    #[tokio::test]
    async fn alert_store_crud() {
        let store = AlertRuleStore::new();
        let rule = AlertRule::new(
            "test-rule",
            AlertCondition::JobFailed { job_type: None },
            Severity::Warning,
        );
        let id = rule.id;

        store.add(rule).await;
        assert_eq!(store.list().await.len(), 1);
        assert!(store.get(id).await.is_some());

        store.remove(id).await;
        assert!(store.get(id).await.is_none());
        assert!(store.list().await.is_empty());
    }

    #[tokio::test]
    async fn alert_store_update() {
        let store = AlertRuleStore::new();
        let mut rule = AlertRule::new(
            "test",
            AlertCondition::JobFailed { job_type: None },
            Severity::Info,
        );
        let id = rule.id;
        store.add(rule.clone()).await;

        rule.enabled = false;
        store.update(id, rule).await.unwrap();

        let updated = store.get(id).await.unwrap();
        assert!(!updated.enabled);
    }

    #[tokio::test]
    async fn alert_store_update_nonexistent_fails() {
        let store = AlertRuleStore::new();
        let rule = AlertRule::new("ghost", AlertCondition::JobFailed { job_type: None }, Severity::Info);
        let result = store.update(Uuid::now_v7(), rule).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn alert_store_evaluate() {
        let store = AlertRuleStore::new();
        let rule = AlertRule::new(
            "job-alert",
            AlertCondition::JobFailed { job_type: None },
            Severity::Error,
        )
        .with_cooldown(0);
        store.add(rule).await;

        let matched = store
            .evaluate("job_failed", &serde_json::json!({}), Utc::now())
            .await;
        assert_eq!(matched.len(), 1);

        let not_matched = store
            .evaluate("other_event", &serde_json::json!({}), Utc::now())
            .await;
        assert!(not_matched.is_empty());
    }
}
