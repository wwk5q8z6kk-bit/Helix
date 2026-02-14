use hx_core::*;
use hx_storage::unified::UnifiedStore;
use std::sync::Arc;

/// Tracks user feedback on agent actions to tune confidence over time.
pub struct ReflectionEngine {
    store: Arc<UnifiedStore>,
}

impl ReflectionEngine {
    pub fn new(store: Arc<UnifiedStore>) -> Self {
        Self { store }
    }

    /// Record feedback when a user applies/dismisses an intent or proposal.
    pub async fn record_feedback(&self, feedback: &AgentFeedback) -> MvResult<()> {
        self.store.nodes.record_feedback(feedback).await
    }

    /// List recent feedback entries, optionally filtered by intent type.
    pub async fn list_feedback(
        &self,
        intent_type: Option<&str>,
        limit: usize,
    ) -> MvResult<Vec<AgentFeedback>> {
        self.store.nodes.list_feedback(intent_type, limit).await
    }

    /// Set or update a confidence override for an intent type.
    pub async fn set_confidence_override(&self, override_: &ConfidenceOverride) -> MvResult<()> {
        self.store.nodes.set_confidence_override(override_).await
    }

    /// List all confidence overrides.
    pub async fn list_confidence_overrides(&self) -> MvResult<Vec<ConfidenceOverride>> {
        self.store.nodes.list_confidence_overrides().await
    }

    /// Calculate adjusted confidence for an intent type.
    /// Uses Bayesian update: raw_confidence * (1.0 + acceptance_rate_delta).
    pub async fn adjusted_confidence(
        &self,
        intent_type: &str,
        raw_confidence: f32,
    ) -> MvResult<f32> {
        // Check for explicit override
        if let Some(override_) = self
            .store
            .nodes
            .get_confidence_override(intent_type)
            .await?
        {
            let adjusted = (raw_confidence + override_.base_adjustment).clamp(0.0, 1.0);
            if adjusted < override_.suppress_below {
                return Ok(0.0);
            }
            return Ok(adjusted);
        }

        // Bayesian adjustment based on acceptance rate
        let (applied, total) = self.store.nodes.get_acceptance_rate(intent_type).await?;
        if total < 5 {
            return Ok(raw_confidence); // Not enough data
        }

        let acceptance_rate = applied as f32 / total as f32;
        let baseline = 0.5;
        let delta = acceptance_rate - baseline;
        let adjusted = (raw_confidence * (1.0 + delta)).clamp(0.0, 1.0);
        Ok(adjusted)
    }

    /// Get reflection statistics for a specific intent type.
    pub async fn get_stats(&self, intent_type: &str) -> MvResult<ReflectionStats> {
        let (applied, total) = self.store.nodes.get_acceptance_rate(intent_type).await?;
        let dismissed = total - applied;
        let acceptance_rate = if total > 0 {
            applied as f32 / total as f32
        } else {
            0.0
        };

        let feedback = self
            .store
            .nodes
            .list_feedback(Some(intent_type), 100)
            .await?;
        let avg_confidence = if feedback.is_empty() {
            0.0
        } else {
            let sum: f32 = feedback.iter().filter_map(|f| f.confidence_at_time).sum();
            let count = feedback
                .iter()
                .filter(|f| f.confidence_at_time.is_some())
                .count();
            if count > 0 {
                sum / count as f32
            } else {
                0.0
            }
        };

        let override_info = self
            .store
            .nodes
            .get_confidence_override(intent_type)
            .await?;

        Ok(ReflectionStats {
            intent_type: intent_type.to_string(),
            total_count: total,
            applied_count: applied,
            dismissed_count: dismissed,
            acceptance_rate,
            avg_confidence,
            override_info,
        })
    }

    /// Suggest overrides based on historical performance.
    pub async fn recalibrate(&self) -> MvResult<Vec<(String, f32)>> {
        let overrides = self.store.nodes.list_confidence_overrides().await?;
        let mut suggestions = Vec::new();

        for intent_type in &[
            "schedule_reminder",
            "extract_task",
            "suggest_link",
            "suggest_tag",
        ] {
            let (applied, total) = self.store.nodes.get_acceptance_rate(intent_type).await?;
            if total >= 10 {
                let rate = applied as f32 / total as f32;
                let current_adj = overrides
                    .iter()
                    .find(|o| o.intent_type == *intent_type)
                    .map(|o| o.base_adjustment)
                    .unwrap_or(0.0);
                let suggested = rate - 0.5;
                if (suggested - current_adj).abs() > 0.05 {
                    suggestions.push((intent_type.to_string(), suggested));
                }
            }
        }
        Ok(suggestions)
    }
}
