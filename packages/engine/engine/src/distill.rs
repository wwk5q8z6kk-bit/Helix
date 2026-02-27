//! Vault Summarization & Distillation.
//!
//! Generates summaries and briefings from the knowledge vault:
//! - Namespace summarization: cluster by topic, generate per-cluster summaries
//! - Temporal summarization: "what happened this week" digests
//! - Topic deep-dive: comprehensive briefing using multi-hop retrieval
//!
//! All distillation uses the LLM when available, with heuristic fallbacks.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::llm::{ChatMessage, CompletionParams, LlmProvider};

/// A distillation request.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DistillRequest {
    /// Summarize all knowledge in a namespace.
    Namespace { namespace: String, max_nodes: Option<usize> },
    /// Temporal digest for a time range.
    Temporal {
        #[serde(default = "default_days")]
        days: u32,
    },
    /// Deep-dive briefing on a specific topic.
    TopicDeepDive { topic: String },
}

fn default_days() -> u32 {
    7
}

/// Result of a distillation operation.
#[derive(Debug, Clone, Serialize)]
pub struct DistillResult {
    pub kind: String,
    pub title: String,
    pub summary: String,
    pub source_count: usize,
    pub generated_at: String,
}

/// The distillation engine.
pub struct DistillEngine {
    llm: Option<Arc<dyn LlmProvider>>,
}

impl DistillEngine {
    pub fn new(llm: Option<Arc<dyn LlmProvider>>) -> Self {
        Self { llm }
    }

    /// Generate a distillation from a list of node content snippets.
    pub async fn distill(
        &self,
        request: &DistillRequest,
        content_snippets: &[ContentSnippet],
    ) -> DistillResult {
        let now = Utc::now().to_rfc3339();

        if content_snippets.is_empty() {
            return DistillResult {
                kind: request.kind_str().into(),
                title: request.title(),
                summary: "No content found for this distillation request.".to_string(),
                source_count: 0,
                generated_at: now,
            };
        }

        let summary = if let Some(ref llm) = self.llm {
            match self.llm_summarize(llm, request, content_snippets).await {
                Ok(s) => s,
                Err(e) => {
                    warn!(error = %e, "LLM distillation failed, using heuristic");
                    self.heuristic_summarize(content_snippets)
                }
            }
        } else {
            self.heuristic_summarize(content_snippets)
        };

        DistillResult {
            kind: request.kind_str().into(),
            title: request.title(),
            summary,
            source_count: content_snippets.len(),
            generated_at: now,
        }
    }

    /// LLM-based summarization.
    async fn llm_summarize(
        &self,
        llm: &Arc<dyn LlmProvider>,
        request: &DistillRequest,
        snippets: &[ContentSnippet],
    ) -> Result<String, String> {
        let context = snippets
            .iter()
            .take(20)
            .enumerate()
            .map(|(i, s)| {
                let title = s.title.as_deref().unwrap_or("untitled");
                let content = if s.content.len() > 300 {
                    format!("{}...", &s.content[..300])
                } else {
                    s.content.clone()
                };
                format!("[{i}] {title}: {content}")
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let system_prompt = match request {
            DistillRequest::Namespace { namespace, .. } => format!(
                "Summarize the following knowledge from the '{namespace}' namespace. \
                 Identify key themes, important facts, and connections between items. \
                 Be concise but comprehensive."
            ),
            DistillRequest::Temporal { days } => format!(
                "Create a temporal digest of knowledge activity from the past {days} days. \
                 Organize by theme, highlight new insights, and note any trends or patterns."
            ),
            DistillRequest::TopicDeepDive { topic } => format!(
                "Create a comprehensive briefing on '{topic}' based on the retrieved knowledge. \
                 Cover key facts, relationships, open questions, and actionable insights."
            ),
        };

        let messages = vec![
            ChatMessage::system(&system_prompt),
            ChatMessage::user(context),
        ];

        let params = CompletionParams {
            max_tokens: Some(1024),
            temperature: Some(0.3),
            ..Default::default()
        };

        llm.complete(&messages, &params)
            .await
            .map_err(|e| format!("distillation failed: {e}"))
    }

    /// Heuristic fallback: extract titles and first sentences.
    fn heuristic_summarize(&self, snippets: &[ContentSnippet]) -> String {
        let mut lines = Vec::new();

        for (i, snippet) in snippets.iter().take(10).enumerate() {
            let title = snippet.title.as_deref().unwrap_or("untitled");
            let first_sentence = snippet
                .content
                .split('.')
                .next()
                .unwrap_or(&snippet.content);
            let truncated = if first_sentence.len() > 150 {
                format!("{}...", &first_sentence[..147])
            } else {
                first_sentence.to_string()
            };
            lines.push(format!("{}. **{}**: {}", i + 1, title, truncated));
        }

        if snippets.len() > 10 {
            lines.push(format!("... and {} more items.", snippets.len() - 10));
        }

        lines.join("\n")
    }
}

/// A content snippet for distillation input.
#[derive(Debug, Clone)]
pub struct ContentSnippet {
    pub title: Option<String>,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub namespace: String,
    pub tags: Vec<String>,
}

impl DistillRequest {
    fn kind_str(&self) -> &str {
        match self {
            Self::Namespace { .. } => "namespace",
            Self::Temporal { .. } => "temporal",
            Self::TopicDeepDive { .. } => "topic_deep_dive",
        }
    }

    fn title(&self) -> String {
        match self {
            Self::Namespace { namespace, .. } => {
                format!("Namespace Summary: {namespace}")
            }
            Self::Temporal { days } => {
                format!("Digest: Past {days} Days")
            }
            Self::TopicDeepDive { topic } => {
                format!("Deep Dive: {topic}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snippets() -> Vec<ContentSnippet> {
        vec![
            ContentSnippet {
                title: Some("Rust Ownership".to_string()),
                content: "Rust uses an ownership model for memory safety. \
                         Each value has a single owner at a time."
                    .into(),
                created_at: Utc::now(),
                namespace: "default".to_string(),
                tags: vec!["rust".to_string()],
            },
            ContentSnippet {
                title: Some("Borrow Checker".to_string()),
                content: "The borrow checker enforces rules about references \
                         at compile time, preventing data races."
                    .into(),
                created_at: Utc::now(),
                namespace: "default".to_string(),
                tags: vec!["rust".to_string(), "safety".to_string()],
            },
        ]
    }

    #[tokio::test]
    async fn distill_namespace_without_llm() {
        let engine = DistillEngine::new(None);
        let req = DistillRequest::Namespace {
            namespace: "default".to_string(),
            max_nodes: None,
        };
        let result = engine.distill(&req, &sample_snippets()).await;
        assert_eq!(result.kind, "namespace");
        assert_eq!(result.source_count, 2);
        assert!(result.summary.contains("Rust Ownership"));
        assert!(result.summary.contains("Borrow Checker"));
    }

    #[tokio::test]
    async fn distill_empty_content() {
        let engine = DistillEngine::new(None);
        let req = DistillRequest::Temporal { days: 7 };
        let result = engine.distill(&req, &[]).await;
        assert_eq!(result.source_count, 0);
        assert!(result.summary.contains("No content"));
    }

    #[tokio::test]
    async fn distill_topic_deep_dive() {
        let engine = DistillEngine::new(None);
        let req = DistillRequest::TopicDeepDive {
            topic: "Rust memory safety".to_string(),
        };
        let result = engine.distill(&req, &sample_snippets()).await;
        assert_eq!(result.kind, "topic_deep_dive");
        assert!(result.title.contains("Rust memory safety"));
    }

    #[test]
    fn heuristic_summary_truncates() {
        let engine = DistillEngine::new(None);
        let snippets = vec![ContentSnippet {
            title: Some("Long Content".to_string()),
            content: "a".repeat(500),
            created_at: Utc::now(),
            namespace: "test".to_string(),
            tags: vec![],
        }];
        let summary = engine.heuristic_summarize(&snippets);
        assert!(summary.len() < 500);
    }

    #[test]
    fn request_titles() {
        let ns = DistillRequest::Namespace {
            namespace: "work".to_string(),
            max_nodes: None,
        };
        assert!(ns.title().contains("work"));

        let temporal = DistillRequest::Temporal { days: 3 };
        assert!(temporal.title().contains("3"));

        let topic = DistillRequest::TopicDeepDive {
            topic: "AI".to_string(),
        };
        assert!(topic.title().contains("AI"));
    }
}
