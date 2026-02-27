use std::sync::Arc;

use chrono::{NaiveTime, Utc};
use hx_core::*;
use hx_storage::unified::UnifiedStore;
use uuid::Uuid;

pub struct AutonomyGate {
    store: Arc<UnifiedStore>,
}

impl AutonomyGate {
    pub fn new(store: Arc<UnifiedStore>) -> Self {
        Self { store }
    }

    /// Check whether any enabled global rule has quiet hours active right now.
    pub async fn is_in_quiet_hours(&self) -> MvResult<bool> {
        let rules = self.store.nodes.list_autonomy_rules().await?;
        let now = Utc::now();
        for rule in &rules {
            if rule.enabled && rule.rule_type == "global" && self.in_quiet_hours(rule, now) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Evaluate an action against autonomy rules.
    /// Rules cascade: contact > domain > tag > global.
    pub async fn evaluate(
        &self,
        intent_type: &str,
        confidence: f32,
        scope_hints: &[(&str, &str)], // [(rule_type, scope_key)]
    ) -> MvResult<AutonomyDecision> {
        let rules = self.store.nodes.list_autonomy_rules().await?;
        let now = Utc::now();

        // Priority order: find most specific matching rule
        let priority_order = ["contact", "domain", "tag", "global"];

        for rule_type in priority_order {
            let matching_rules: Vec<&AutonomyRule> = rules
                .iter()
                .filter(|r| r.enabled && r.rule_type == rule_type)
                .filter(|r| {
                    if rule_type == "global" {
                        true
                    } else {
                        scope_hints
                            .iter()
                            .any(|(rt, sk)| *rt == rule_type && r.scope_key.as_deref() == Some(*sk))
                    }
                })
                .collect();

            if let Some(rule) = matching_rules.first() {
                return self.apply_rule(rule, intent_type, confidence, now).await;
            }
        }

        // No rules match -- defer by default
        Ok(AutonomyDecision::Defer)
    }

    async fn apply_rule(
        &self,
        rule: &AutonomyRule,
        intent_type: &str,
        confidence: f32,
        now: chrono::DateTime<Utc>,
    ) -> MvResult<AutonomyDecision> {
        // Check blocked intent types
        if !rule.blocked_intent_types.is_empty()
            && rule.blocked_intent_types.iter().any(|t| t == intent_type)
        {
            self.log_action(
                Some(rule.id),
                intent_type,
                AutonomyDecision::Block,
                confidence,
                "blocked intent type",
            )
            .await?;
            return Ok(AutonomyDecision::Block);
        }

        // Check allowed intent types (if specified, must match)
        if !rule.allowed_intent_types.is_empty()
            && !rule.allowed_intent_types.iter().any(|t| t == intent_type)
        {
            self.log_action(
                Some(rule.id),
                intent_type,
                AutonomyDecision::Defer,
                confidence,
                "intent type not in allowed list",
            )
            .await?;
            return Ok(AutonomyDecision::Defer);
        }

        // Check quiet hours
        if self.in_quiet_hours(rule, now) {
            self.log_action(
                Some(rule.id),
                intent_type,
                AutonomyDecision::QueueForLater,
                confidence,
                "quiet hours",
            )
            .await?;
            return Ok(AutonomyDecision::QueueForLater);
        }

        // Check rate limit
        let one_hour_ago = now - chrono::Duration::hours(1);
        let recent_count = self
            .store
            .nodes
            .count_recent_actions(Some(rule.id), one_hour_ago)
            .await?;
        if recent_count >= rule.max_actions_per_hour as usize {
            self.log_action(
                Some(rule.id),
                intent_type,
                AutonomyDecision::Defer,
                confidence,
                "rate limit exceeded",
            )
            .await?;
            return Ok(AutonomyDecision::Defer);
        }

        // Check confidence threshold
        if confidence >= rule.auto_apply_threshold {
            self.log_action(
                Some(rule.id),
                intent_type,
                AutonomyDecision::AutoApply,
                confidence,
                "above threshold",
            )
            .await?;
            return Ok(AutonomyDecision::AutoApply);
        }

        self.log_action(
            Some(rule.id),
            intent_type,
            AutonomyDecision::Defer,
            confidence,
            "below threshold",
        )
        .await?;
        Ok(AutonomyDecision::Defer)
    }

    fn in_quiet_hours(&self, rule: &AutonomyRule, now: chrono::DateTime<Utc>) -> bool {
        let (start_str, end_str) = match (&rule.quiet_hours_start, &rule.quiet_hours_end) {
            (Some(s), Some(e)) => (s.as_str(), e.as_str()),
            _ => return false,
        };

        let start = match NaiveTime::parse_from_str(start_str, "%H:%M") {
            Ok(t) => t,
            Err(_) => return false,
        };
        let end = match NaiveTime::parse_from_str(end_str, "%H:%M") {
            Ok(t) => t,
            Err(_) => return false,
        };

        // Use UTC time (timezone handling is simplified)
        let current = now.time();

        if start <= end {
            current >= start && current < end
        } else {
            // Wraps midnight
            current >= start || current < end
        }
    }

    async fn log_action(
        &self,
        rule_id: Option<Uuid>,
        intent_type: &str,
        decision: AutonomyDecision,
        confidence: f32,
        reason: &str,
    ) -> MvResult<()> {
        let log = AutonomyActionLog {
            id: Uuid::new_v4(),
            rule_id,
            intent_type: intent_type.to_string(),
            decision,
            confidence: Some(confidence),
            reason: Some(reason.to_string()),
            created_at: Utc::now(),
        };
        self.store.nodes.log_autonomy_action(&log).await
    }
}
