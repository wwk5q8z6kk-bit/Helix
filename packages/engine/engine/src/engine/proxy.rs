use super::HelixEngine;
use hx_core::{
    AgenticStore, AutonomyDecision, ChronicleEntry, ContactIdentity,
    ContactIdentityStore, ContentType, MemoryQuery,
    MessageStatus, MvResult, Proposal, ProposalAction, ProposalSender,
    RelayMessage, TrustLevel, TrustModel,
};
use uuid::Uuid;

use crate::llm;

#[derive(Debug, Clone)]
pub struct RelayInboundOutcome {
    pub message: RelayMessage,
    pub auto_reply: Option<RelayMessage>,
    pub proposal_id: Option<Uuid>,
}

impl HelixEngine {
    // --- Relay / Proxy ---

    /// Receive a relay message and optionally generate an auto-reply or proposal.
    pub async fn receive_relay_message(
        &self,
        message: RelayMessage,
        namespace: &str,
    ) -> MvResult<RelayInboundOutcome> {
        let mut stored = self.relay.receive_message(message, namespace).await?;

        if stored.status == MessageStatus::Failed {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        let Some(sender_id) = stored.sender_contact_id else {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        };

        let Some(contact) = self.relay.get_contact(sender_id).await? else {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        };

        if contact.trust_level == TrustLevel::RelayOnly {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        if stored.content_type != ContentType::Text || stored.content.trim().is_empty() {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        let thread_id = stored.thread_id.unwrap_or(stored.id);

        let subject = stored
            .metadata
            .get("subject")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(|value| {
                if value.to_ascii_lowercase().starts_with("re:") {
                    value.to_string()
                } else {
                    format!("Re: {value}")
                }
            });

        let query = MemoryQuery::new(&stored.content)
            .with_namespace(namespace.to_string())
            .with_limit(6)
            .with_min_score(0.0);

        let results = match self.recall(&query).await {
            Ok(results) => results,
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "relay_reply_context_recall_failed"
                );
                return Ok(RelayInboundOutcome {
                    message: stored,
                    auto_reply: None,
                    proposal_id: None,
                });
            }
        };

        let context_snippets = llm::extract_context_snippets(&results, 4);
        if context_snippets.is_empty() {
            return Ok(RelayInboundOutcome {
                message: stored,
                auto_reply: None,
                proposal_id: None,
            });
        }

        let input = if let Some(ref subject) = subject {
            format!("Subject: {subject}\n\n{}", stored.content)
        } else {
            stored.content.clone()
        };

        let mut used_llm = false;
        let mut suggestion_text = None;
        if let Some(ref llm) = self.llm {
            match llm::llm_completion_suggestions(llm.as_ref(), &input, &context_snippets, 1).await
            {
                Ok(mut suggestions) => {
                    if let Some(first) = suggestions.pop() {
                        suggestion_text = Some(first);
                        used_llm = true;
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        provider = %llm.name(),
                        "relay_reply_llm_suggestion_failed"
                    );
                }
            }
        }

        if suggestion_text.is_none() {
            let preview = context_snippets
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            suggestion_text = Some(format!(
                "I have related notes that might help:\n{preview}\n\nWant me to share details?"
            ));
        }

        let mut confidence: f32 = if used_llm { 0.6 } else { 0.4 };
        if context_snippets.len() >= 3 {
            confidence += 0.1;
        }
        if contact.trust_level == TrustLevel::Full {
            confidence += 0.1;
        }
        if contact.trust_level == TrustLevel::ContextInject {
            confidence -= 0.05;
        }
        confidence = confidence.clamp(0.0, 1.0);

        let suggestion = RelayReplySuggestion {
            content: suggestion_text.unwrap_or_default(),
            confidence,
            context_snippets,
        };

        let contact_scope = sender_id.to_string();
        let scope_hints = [("contact", contact_scope.as_str()), ("domain", "relay")];
        let mut decision = self
            .autonomy
            .evaluate("relay.reply", suggestion.confidence, &scope_hints)
            .await?;

        if contact.trust_level != TrustLevel::Full
            && matches!(decision, AutonomyDecision::AutoApply)
        {
            decision = AutonomyDecision::Defer;
        }

        let mut auto_reply = None;
        let mut proposal_id = None;

        match decision {
            AutonomyDecision::AutoApply => {
                let mut reply =
                    RelayMessage::outbound(stored.channel_id, suggestion.content.clone())
                        .with_thread(thread_id)
                        .with_content_type(ContentType::Text);
                reply.recipient_contact_id = Some(sender_id);
                reply
                    .metadata
                    .insert("auto_reply".to_string(), serde_json::Value::Bool(true));
                reply.metadata.insert(
                    "basis_message_id".to_string(),
                    serde_json::Value::String(stored.id.to_string()),
                );
                if let Some(ref subject) = subject {
                    reply.metadata.insert(
                        "subject".to_string(),
                        serde_json::Value::String(subject.clone()),
                    );
                }

                let stored_reply = self.relay.send_message(reply, namespace).await?;
                auto_reply = Some(stored_reply);

                if let Ok(true) = self
                    .relay
                    .update_status(stored.id, MessageStatus::AutoReplied)
                    .await
                {
                    stored.status = MessageStatus::AutoReplied;
                }
            }
            AutonomyDecision::Defer | AutonomyDecision::QueueForLater => {
                let mut payload = std::collections::HashMap::new();
                payload.insert(
                    "channel_id".to_string(),
                    serde_json::Value::String(stored.channel_id.to_string()),
                );
                payload.insert(
                    "content".to_string(),
                    serde_json::Value::String(suggestion.content.clone()),
                );
                payload.insert(
                    "content_type".to_string(),
                    serde_json::Value::String(ContentType::Text.to_string()),
                );
                payload.insert(
                    "basis_message_id".to_string(),
                    serde_json::Value::String(stored.id.to_string()),
                );
                payload.insert(
                    "context_snippets".to_string(),
                    serde_json::Value::Array(
                        suggestion
                            .context_snippets
                            .iter()
                            .map(|snippet| serde_json::Value::String(snippet.clone()))
                            .collect(),
                    ),
                );
                payload.insert(
                    "thread_id".to_string(),
                    serde_json::Value::String(thread_id.to_string()),
                );
                if let Some(recipient_id) = stored.sender_contact_id {
                    payload.insert(
                        "recipient_contact_id".to_string(),
                        serde_json::Value::String(recipient_id.to_string()),
                    );
                }
                if let Some(ref subject) = subject {
                    payload.insert(
                        "subject".to_string(),
                        serde_json::Value::String(subject.clone()),
                    );
                }

                let proposal = Proposal::new(
                    ProposalSender::Relay,
                    ProposalAction::Custom("relay.reply".to_string()),
                )
                .with_confidence(suggestion.confidence)
                .with_diff(suggestion.content.clone())
                .with_payload(payload);

                self.submit_proposal(&proposal).await?;
                proposal_id = Some(proposal.id);

                if let Ok(true) = self
                    .relay
                    .update_status(stored.id, MessageStatus::Deferred)
                    .await
                {
                    stored.status = MessageStatus::Deferred;
                }
            }
            AutonomyDecision::Block => {}
        }

        Ok(RelayInboundOutcome {
            message: stored,
            auto_reply,
            proposal_id,
        })
    }

    // --- Contact Identity & Trust ---

    /// Add an identity to a relay contact.
    pub async fn add_contact_identity(&self, identity: &ContactIdentity) -> MvResult<()> {
        self.store.nodes.add_contact_identity(identity).await
    }

    /// List identities for a contact.
    pub async fn list_contact_identities(
        &self,
        contact_id: Uuid,
    ) -> MvResult<Vec<ContactIdentity>> {
        self.store.nodes.list_contact_identities(contact_id).await
    }

    /// Delete a contact identity.
    pub async fn delete_contact_identity(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_contact_identity(id).await
    }

    /// Verify a contact identity.
    pub async fn verify_contact_identity(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.verify_contact_identity(id).await
    }

    /// Get trust model for a contact.
    pub async fn get_trust_model(&self, contact_id: Uuid) -> MvResult<Option<TrustModel>> {
        self.store.nodes.get_trust_model(contact_id).await
    }

    /// Set trust model for a contact.
    pub async fn set_trust_model(&self, model: &TrustModel) -> MvResult<()> {
        self.store.nodes.set_trust_model(model).await
    }

    // --- Chronicles / Audit ---

    /// List chronicle entries with optional node filter.
    pub async fn list_chronicles(
        &self,
        node_id: Option<Uuid>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ChronicleEntry>> {
        self.store
            .nodes
            .list_chronicles(node_id, limit, offset)
            .await
    }

    /// Log a chronicle entry for transparency.
    pub async fn log_chronicle(&self, entry: &ChronicleEntry) -> MvResult<()> {
        self.store.nodes.log_chronicle(entry).await
    }
}

// Helper struct for internal use
struct RelayReplySuggestion {
    content: String,
    confidence: f32,
    context_snippets: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_utils::*;
    use chrono::{TimeZone, Utc};
    use hx_core::IdentityType;

    #[tokio::test]
    async fn test_contact_identity_crud() {
        let (engine, _tmp) = create_test_engine().await;

        let contact_id = Uuid::now_v7();
        let identity = ContactIdentity {
            id: Uuid::now_v7(),
            contact_id,
            identity_type: IdentityType::Email,
            identity_value: "alice@example.com".into(),
            verified: false,
            verified_at: None,
            created_at: Utc::now(),
        };

        // Add
        engine.add_contact_identity(&identity).await.unwrap();

        // List
        let list = engine.list_contact_identities(contact_id).await.unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].identity_value, "alice@example.com");
        assert!(!list[0].verified);

        // Verify
        let verified = engine.verify_contact_identity(identity.id).await.unwrap();
        assert!(verified);

        let list = engine.list_contact_identities(contact_id).await.unwrap();
        assert!(list[0].verified);

        // Delete
        let deleted = engine.delete_contact_identity(identity.id).await.unwrap();
        assert!(deleted);

        let list = engine.list_contact_identities(contact_id).await.unwrap();
        assert!(list.is_empty());
    }

    #[tokio::test]
    async fn test_trust_model_defaults_and_update() {
        let (engine, _tmp) = create_test_engine().await;

        let contact_id = Uuid::now_v7();

        // No trust model yet
        let model = engine.get_trust_model(contact_id).await.unwrap();
        assert!(model.is_none());

        // Set with defaults (all false)
        let mut tm = TrustModel {
            contact_id,
            ..Default::default()
        };
        engine.set_trust_model(&tm).await.unwrap();

        let stored = engine.get_trust_model(contact_id).await.unwrap().unwrap();
        assert!(!stored.can_query);
        assert!(!stored.can_inject_context);
        assert!(!stored.can_auto_reply);
        assert!(stored.allowed_namespaces.is_empty());

        // Update
        tm.can_query = true;
        tm.allowed_namespaces = vec!["research".into()];
        engine.set_trust_model(&tm).await.unwrap();

        let updated = engine.get_trust_model(contact_id).await.unwrap().unwrap();
        assert!(updated.can_query);
        assert_eq!(updated.allowed_namespaces, vec!["research"]);
    }

}
