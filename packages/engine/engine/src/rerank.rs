//! Cross-encoder reranking for search result quality improvement.
//!
//! After initial retrieval via FTS + vector + graph fusion, the top-N results
//! are re-scored using a cross-encoder model that jointly processes the
//! query-document pair. This produces more accurate relevance scores than
//! bi-encoder similarity alone.
//!
//! Two implementations:
//! - `OnnxReranker`: loads an ONNX cross-encoder model (feature-gated: `reranking`)
//! - `LlmReranker`: uses the LLM provider to score relevance (always available)

use std::sync::Arc;

use hx_core::error::MvResult;
use hx_core::traits::Reranker;
use tracing::{debug, warn};

use crate::config::RerankConfig;
use crate::llm::{ChatMessage, CompletionParams, LlmProvider};

/// LLM-based reranker: asks the LLM to score relevance of each document.
/// Always available when an LLM provider is configured.
pub struct LlmReranker {
    llm: Arc<dyn LlmProvider>,
}

impl LlmReranker {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self { llm }
    }
}

#[async_trait::async_trait]
impl Reranker for LlmReranker {
    async fn rerank(
        &self,
        query: &str,
        documents: &[String],
    ) -> MvResult<Vec<f64>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Batch all documents into a single prompt for efficiency
        let doc_list: String = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let truncated = if doc.len() > 500 {
                    format!("{}...", &doc[..500])
                } else {
                    doc.clone()
                };
                format!("[{i}] {truncated}")
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let messages = vec![
            ChatMessage::system(
                "You are a relevance scoring assistant. Given a query and numbered documents, \
                 rate the relevance of each document to the query on a scale from 0.0 to 1.0. \
                 Return ONLY a JSON array of numbers, one per document, in the same order. \
                 Example: [0.95, 0.3, 0.7]",
            ),
            ChatMessage::user(format!(
                "Query: {query}\n\nDocuments:\n{doc_list}\n\nReturn relevance scores as JSON array:"
            )),
        ];

        let params = CompletionParams {
            max_tokens: Some(128),
            temperature: Some(0.0),
            ..Default::default()
        };

        match self.llm.complete(&messages, &params).await {
            Ok(response) => {
                let response = response.trim();
                // Try to parse as JSON array of floats
                if let Ok(scores) = serde_json::from_str::<Vec<f64>>(response) {
                    if scores.len() == documents.len() {
                        return Ok(scores);
                    }
                }

                // Try extracting numbers from the response
                let scores: Vec<f64> = response
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split(',')
                    .filter_map(|s| s.trim().parse::<f64>().ok())
                    .collect();

                if scores.len() == documents.len() {
                    Ok(scores)
                } else {
                    warn!(
                        expected = documents.len(),
                        got = scores.len(),
                        "LLM reranker returned wrong number of scores, using uniform scores"
                    );
                    Ok(vec![0.5; documents.len()])
                }
            }
            Err(e) => {
                warn!(error = %e, "LLM reranker failed, using uniform scores");
                Ok(vec![0.5; documents.len()])
            }
        }
    }

    fn name(&self) -> &str {
        "llm-reranker"
    }

    fn is_ready(&self) -> bool {
        true
    }
}

/// No-op reranker that returns uniform scores (passthrough).
pub struct NoOpReranker;

#[async_trait::async_trait]
impl Reranker for NoOpReranker {
    async fn rerank(
        &self,
        _query: &str,
        documents: &[String],
    ) -> MvResult<Vec<f64>> {
        // Return descending scores to preserve original ordering
        Ok((0..documents.len())
            .map(|i| 1.0 - (i as f64 * 0.001))
            .collect())
    }

    fn name(&self) -> &str {
        "noop"
    }

    fn is_ready(&self) -> bool {
        true
    }
}

/// Initialize the best available reranker based on config and available providers.
pub fn init_reranker(
    config: &RerankConfig,
    llm: Option<Arc<dyn LlmProvider>>,
) -> Arc<dyn Reranker> {
    if !config.enabled {
        debug!("reranking disabled, using no-op reranker");
        return Arc::new(NoOpReranker);
    }

    // TODO: When `reranking` feature is enabled, try ONNX cross-encoder first
    // #[cfg(feature = "reranking")]
    // if let Some(reranker) = OnnxReranker::try_load(config) {
    //     return Arc::new(reranker);
    // }

    if let Some(llm) = llm {
        debug!("using LLM-based reranker");
        Arc::new(LlmReranker::new(llm))
    } else {
        debug!("no LLM available, using no-op reranker");
        Arc::new(NoOpReranker)
    }
}

/// Apply reranking to search results in-place.
/// Takes the fused (uuid, score) pairs and re-scores using the reranker.
/// Returns a new vector sorted by reranker scores.
pub async fn apply_reranking(
    reranker: &dyn Reranker,
    query: &str,
    results: &mut Vec<(uuid::Uuid, f64)>,
    documents: &[String],
    config: &RerankConfig,
) {
    if results.is_empty() || !reranker.is_ready() {
        return;
    }

    // Only rerank top-N candidates
    let top_n = config.top_n.min(results.len());
    let candidates: Vec<String> = documents.iter().take(top_n).cloned().collect();

    match reranker.rerank(query, &candidates).await {
        Ok(scores) => {
            // Blend reranker scores with original scores (70% reranker, 30% original)
            for (i, score) in scores.iter().enumerate().take(top_n) {
                if i < results.len() {
                    let original_score = results[i].1;
                    results[i].1 = score * 0.7 + original_score * 0.3;
                }
            }

            // Re-sort by blended score
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Apply minimum score filter
            if config.min_score > 0.0 {
                results.retain(|(_, s)| *s >= config.min_score);
            }

            debug!(
                reranker = reranker.name(),
                candidates = top_n,
                "reranking applied"
            );
        }
        Err(e) => {
            warn!(error = %e, "reranking failed, keeping original scores");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn noop_reranker_preserves_order() {
        let reranker = NoOpReranker;
        let docs = vec!["doc1".into(), "doc2".into(), "doc3".into()];
        let scores = reranker.rerank("test", &docs).await.unwrap();
        assert_eq!(scores.len(), 3);
        assert!(scores[0] > scores[1]);
        assert!(scores[1] > scores[2]);
    }

    #[tokio::test]
    async fn noop_reranker_empty() {
        let reranker = NoOpReranker;
        let scores = reranker.rerank("test", &[]).await.unwrap();
        assert!(scores.is_empty());
    }

    #[tokio::test]
    async fn apply_reranking_noop_preserves() {
        let reranker = NoOpReranker;
        let config = RerankConfig {
            enabled: true,
            top_n: 10,
            min_score: 0.0,
            ..Default::default()
        };

        let mut results = vec![
            (uuid::Uuid::new_v4(), 0.9),
            (uuid::Uuid::new_v4(), 0.7),
            (uuid::Uuid::new_v4(), 0.5),
        ];
        let original_ids: Vec<_> = results.iter().map(|(id, _)| *id).collect();

        let docs = vec!["doc1".into(), "doc2".into(), "doc3".into()];
        apply_reranking(&reranker, "test", &mut results, &docs, &config).await;

        // NoOp should roughly preserve order since it returns descending scores
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].0, original_ids[0]);
    }
}
