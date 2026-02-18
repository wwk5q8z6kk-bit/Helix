use super::HelixEngine;
use chrono::{DateTime, Utc};
use hx_core::{
    AgentFeedback, AgenticStore, CapturedIntent, ConfidenceOverride, ConflictAlert, ConflictStore,
    ExchangeStore, FeedbackStore, HxError, IntentStatus, MvResult,
    ProactiveInsight, Proposal, ProposalState,
};
use std::sync::Arc;
use uuid::Uuid;

impl HelixEngine {
    // -------------------------------------------------------------------------
    // Agentic Intelligence Methods
    // -------------------------------------------------------------------------

    /// List captured intents with optional filters.
    pub async fn list_intents(
        &self,
        node_id: Option<Uuid>,
        status: Option<IntentStatus>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<CapturedIntent>> {
        self.store
            .nodes
            .list_intents(node_id, status, limit, offset)
            .await
    }

    /// Update the status of an intent (apply/dismiss).
    pub async fn update_intent_status(&self, id: Uuid, status: IntentStatus) -> MvResult<bool> {
        self.store.nodes.update_intent_status(id, status).await
    }

    /// Get a single intent by ID.
    pub async fn get_intent(&self, id: Uuid) -> MvResult<Option<CapturedIntent>> {
        self.store.nodes.get_intent(id).await
    }

    /// Apply an intent: execute the action and mark as applied.
    pub async fn apply_intent(
        self: &Arc<Self>,
        id: Uuid,
    ) -> MvResult<crate::intent_executor::ExecutionResult> {
        // Get the intent
        let intent = self
            .store
            .nodes
            .get_intent(id)
            .await?
            .ok_or_else(|| HxError::InvalidInput(format!("Intent {} not found", id)))?;

        // Execute the intent
        let executor = crate::intent_executor::IntentExecutor::new(Arc::clone(self));
        let result = executor.execute(&intent).await?;

        // If execution succeeded, mark as applied
        if result.success {
            self.store
                .nodes
                .update_intent_status(id, IntentStatus::Applied)
                .await?;
        }

        Ok(result)
    }

    /// List proactive insights.
    pub async fn list_insights(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ProactiveInsight>> {
        self.store.nodes.list_insights(limit, offset).await
    }

    /// Delete (dismiss) an insight.
    pub async fn delete_insight(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_insight(id).await
    }

    // --- Conflict Detection ---

    /// List conflict alerts.
    pub async fn list_conflicts(
        &self,
        resolved: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ConflictAlert>> {
        self.store
            .nodes
            .list_conflicts(resolved, limit, offset)
            .await
    }

    /// Get a single conflict alert.
    pub async fn get_conflict(&self, id: Uuid) -> MvResult<Option<ConflictAlert>> {
        self.store.nodes.get_conflict(id).await
    }

    /// Resolve (dismiss) a conflict alert.
    pub async fn resolve_conflict(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.resolve_conflict(id).await
    }

    // --- Feedback / Learning ---

    /// Record feedback for an intent action (apply/dismiss).
    pub async fn record_feedback(&self, fb: &AgentFeedback) -> MvResult<()> {
        self.store.nodes.record_feedback(fb).await
    }

    /// Get acceptance rate for an intent type. Returns (total, applied).
    pub async fn get_acceptance_rate(&self, intent_type: &str) -> MvResult<(usize, usize)> {
        self.store.nodes.get_acceptance_rate(intent_type).await
    }

    /// Get confidence override for an intent type.
    pub async fn get_confidence_override(
        &self,
        intent_type: &str,
    ) -> MvResult<Option<ConfidenceOverride>> {
        self.store.nodes.get_confidence_override(intent_type).await
    }

    /// Recalculate and store a confidence override based on accumulated feedback.
    pub async fn recalculate_confidence(&self, intent_type: &str) -> MvResult<()> {
        let (total, applied) = self.store.nodes.get_acceptance_rate(intent_type).await?;

        // Need at least 5 data points to start adjusting
        if total < 5 {
            return Ok(());
        }

        let rate = applied as f32 / total as f32;

        // base_adjustment: -0.2 to +0.2 based on acceptance rate
        // 50% → 0.0, 100% → +0.2, 0% → -0.2
        let base_adjustment = (rate - 0.5) * 0.4;

        // auto_apply_threshold: lower if acceptance rate is high
        let auto_apply_threshold = if rate > 0.9 && total >= 20 {
            0.9 // auto-apply above 0.9 confidence
        } else {
            0.95 // default: very high threshold
        };

        // suppress_below: raise if acceptance rate is very low
        let suppress_below = if rate < 0.1 && total >= 10 {
            0.5 // suppress weak suggestions for disliked intent types
        } else if rate < 0.3 {
            0.3
        } else {
            0.1 // default
        };

        let override_ = ConfidenceOverride {
            intent_type: intent_type.to_string(),
            base_adjustment,
            auto_apply_threshold,
            suppress_below,
            updated_at: chrono::Utc::now(),
        };

        self.store.nodes.set_confidence_override(&override_).await
    }

    // --- Exchange Inbox ---

    pub async fn submit_proposal(&self, proposal: &Proposal) -> MvResult<()> {
        self.store.nodes.submit_proposal(proposal).await
    }

    pub async fn get_proposal(&self, id: Uuid) -> MvResult<Option<Proposal>> {
        self.store.nodes.get_proposal(id).await
    }

    pub async fn list_proposals(
        &self,
        state: Option<ProposalState>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<Proposal>> {
        self.store.nodes.list_proposals(state, limit, offset).await
    }

    pub async fn resolve_proposal(&self, id: Uuid, state: ProposalState) -> MvResult<bool> {
        self.store.nodes.resolve_proposal(id, state).await
    }

    pub async fn count_proposals(&self, state: Option<ProposalState>) -> MvResult<usize> {
        self.store.nodes.count_proposals(state).await
    }

    pub async fn expire_proposals(&self, before: DateTime<Utc>) -> MvResult<usize> {
        self.store.nodes.expire_proposals(before).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_utils::*;
    use hx_core::{ConflictType, InsightType, KnowledgeNode, NodeKind};

    #[tokio::test]
    async fn test_insight_generate_and_list() {
        let (engine, _tmp) = create_test_engine().await;

        // Store a test insight directly via the underlying store
        let insight = ProactiveInsight {
            id: Uuid::now_v7(),
            title: "Test Insight".into(),
            content: "Something interesting".into(),
            insight_type: InsightType::General,
            related_node_ids: vec![],
            importance: 0.8,
            created_at: Utc::now(),
            dismissed_at: None,
        };
        engine.store.nodes.log_insight(&insight).await.unwrap();

        // List and verify
        let insights = engine.list_insights(10, 0).await.unwrap();
        assert!(!insights.is_empty(), "should have at least one insight");
        assert_eq!(insights[0].title, "Test Insight");

        // Dismiss (delete)
        let deleted = engine.delete_insight(insight.id).await.unwrap();
        assert!(deleted);

        let after = engine.list_insights(10, 0).await.unwrap();
        assert!(
            after.iter().all(|i| i.id != insight.id),
            "insight should be dismissed"
        );
    }

    #[tokio::test]
    async fn test_conflict_detection_on_contradictory_nodes() {
        let (engine, _tmp) = create_test_engine().await;

        // Store two nodes
        let node_a = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "The sky is blue").with_tags(vec!["sky".into()]),
            )
            .await
            .unwrap();
        let node_b = engine
            .store_node(
                KnowledgeNode::new(NodeKind::Fact, "The sky is green")
                    .with_tags(vec!["sky".into()]),
            )
            .await
            .unwrap();

        // Manually insert a conflict alert
        let alert = ConflictAlert {
            id: Uuid::now_v7(),
            node_a: node_a.id,
            node_b: node_b.id,
            conflict_type: ConflictType::Contradiction,
            score: 0.95,
            explanation: "Contradictory claims about sky color".into(),
            resolved: false,
            created_at: Utc::now(),
        };
        engine.store.nodes.insert_conflict(&alert).await.unwrap();

        // List unresolved conflicts
        let conflicts = engine.list_conflicts(Some(false), 10, 0).await.unwrap();
        assert!(!conflicts.is_empty());
        assert_eq!(conflicts[0].node_a, node_a.id);

        // Resolve
        let resolved = engine.resolve_conflict(alert.id).await.unwrap();
        assert!(resolved);

        // Verify the conflict is now marked resolved
        let fetched = engine.get_conflict(alert.id).await.unwrap().unwrap();
        assert!(fetched.resolved, "conflict should be resolved");

        // Resolving again should return false (already resolved)
        let re_resolved = engine.resolve_conflict(alert.id).await.unwrap();
        assert!(!re_resolved);
    }

}
