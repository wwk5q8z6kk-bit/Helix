//! Multi-hop retrieval: iterative retrieve → extract → follow-up → merge.
//!
//! For complex queries that span multiple topics, a single retrieval pass may
//! miss relevant documents that are indirectly related. Multi-hop retrieval
//! iterates:
//! 1. Initial retrieval with the original query
//! 2. Extract key entities/concepts from top results
//! 3. Formulate follow-up queries from extracted entities
//! 4. Retrieve again with follow-up queries
//! 5. Merge and re-rank all results
//!
//! Bounded by max_hops and a token budget to prevent runaway.

use std::collections::HashSet;
use std::sync::Arc;

use tracing::{debug, warn};
use uuid::Uuid;

use crate::config::MultiHopConfig;
use crate::llm::{ChatMessage, CompletionParams, LlmProvider};

/// A single hop's results.
#[derive(Debug, Clone)]
pub struct HopResult {
    pub hop_number: usize,
    pub query: String,
    pub result_ids: Vec<(Uuid, f64)>,
}

/// Orchestrates multi-hop retrieval.
pub struct MultiHopRetriever {
    llm: Option<Arc<dyn LlmProvider>>,
    config: MultiHopConfig,
}

impl MultiHopRetriever {
    pub fn new(llm: Option<Arc<dyn LlmProvider>>, config: MultiHopConfig) -> Self {
        Self { llm, config }
    }

    /// Given initial results and their content, determine follow-up queries.
    /// Returns empty vec if no follow-up is warranted.
    pub async fn plan_follow_ups(
        &self,
        original_query: &str,
        result_contents: &[String],
        seen_queries: &HashSet<String>,
    ) -> Vec<String> {
        if !self.config.enabled || result_contents.is_empty() {
            return vec![];
        }

        let Some(llm) = &self.llm else {
            return self.heuristic_follow_ups(result_contents, seen_queries);
        };

        let context: String = result_contents
            .iter()
            .take(self.config.results_per_hop)
            .enumerate()
            .map(|(i, c)| {
                let truncated = if c.len() > 300 {
                    format!("{}...", &c[..300])
                } else {
                    c.clone()
                };
                format!("[{}] {}", i + 1, truncated)
            })
            .collect::<Vec<_>>()
            .join("\n");

        let seen_list: String = seen_queries
            .iter()
            .map(|q| format!("- {q}"))
            .collect::<Vec<_>>()
            .join("\n");

        let messages = vec![
            ChatMessage::system(
                "You are a follow-up query generator for a knowledge management system. \
                 Given the original query and retrieved documents, identify entities or \
                 concepts mentioned in the documents that could lead to additional relevant \
                 knowledge. Generate 1-2 concise follow-up search queries. \
                 Do NOT repeat queries already searched. \
                 Return ONLY the queries, one per line, no numbering.",
            ),
            ChatMessage::user(format!(
                "Original query: {original_query}\n\n\
                 Retrieved documents:\n{context}\n\n\
                 Already searched:\n{seen_list}\n\n\
                 Generate follow-up queries:"
            )),
        ];

        let params = CompletionParams {
            max_tokens: Some(128),
            temperature: Some(0.2),
            ..Default::default()
        };

        match llm.complete(&messages, &params).await {
            Ok(result) => {
                let follow_ups: Vec<String> = result
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty() && !seen_queries.contains(l))
                    .take(2)
                    .collect();

                debug!(
                    original = original_query,
                    follow_ups = ?follow_ups,
                    "multi-hop follow-up queries generated"
                );

                follow_ups
            }
            Err(e) => {
                warn!(error = %e, "LLM follow-up generation failed");
                self.heuristic_follow_ups(result_contents, seen_queries)
            }
        }
    }

    /// Heuristic follow-up: extract capitalized phrases and uncommon terms.
    fn heuristic_follow_ups(
        &self,
        result_contents: &[String],
        seen_queries: &HashSet<String>,
    ) -> Vec<String> {
        let mut candidates: Vec<String> = Vec::new();
        let stop_words: HashSet<&str> = [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "must", "need", "this",
            "that", "these", "those", "it", "its", "they", "them", "their",
            "we", "our", "you", "your", "he", "she", "his", "her", "i", "my",
            "me", "for", "of", "in", "on", "at", "to", "from", "with", "by",
            "as", "if", "or", "and", "but", "not", "no", "so", "than", "too",
            "very", "just", "about", "also", "more", "some", "any", "all",
        ]
        .into_iter()
        .collect();

        for content in result_contents.iter().take(3) {
            // Extract capitalized multi-word phrases (likely entity names)
            let words: Vec<&str> = content.split_whitespace().collect();
            let mut i = 0;
            while i < words.len() {
                if words[i].starts_with(|c: char| c.is_uppercase()) && words[i].len() > 2 {
                    let mut phrase = vec![words[i]];
                    let mut j = i + 1;
                    while j < words.len()
                        && words[j].starts_with(|c: char| c.is_uppercase())
                        && words[j].len() > 1
                    {
                        phrase.push(words[j]);
                        j += 1;
                    }
                    if phrase.len() >= 2 {
                        let entity = phrase.join(" ");
                        if !seen_queries.contains(&entity) {
                            candidates.push(entity);
                        }
                    }
                    i = j;
                } else {
                    i += 1;
                }
            }

            // Extract uncommon terms (longer words not in stop list)
            for word in content.split_whitespace() {
                let clean = word
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if clean.len() >= 6
                    && !stop_words.contains(clean.as_str())
                    && !seen_queries.contains(&clean)
                    && !candidates.contains(&clean)
                {
                    candidates.push(clean);
                }
            }
        }

        candidates.truncate(2);
        candidates
    }

    /// Estimate token count (rough: ~4 chars per token).
    pub fn estimate_tokens(text: &str) -> usize {
        text.len() / 4
    }

    /// Check if we're within the token budget.
    pub fn within_budget(&self, accumulated_tokens: usize) -> bool {
        accumulated_tokens < self.config.token_budget
    }

    pub fn max_hops(&self) -> usize {
        self.config.max_hops
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> MultiHopConfig {
        MultiHopConfig {
            enabled: true,
            max_hops: 2,
            token_budget: 2048,
            results_per_hop: 5,
        }
    }

    #[tokio::test]
    async fn disabled_returns_empty() {
        let retriever = MultiHopRetriever::new(
            None,
            MultiHopConfig {
                enabled: false,
                ..Default::default()
            },
        );
        let follow_ups = retriever
            .plan_follow_ups("test", &["some content".into()], &HashSet::new())
            .await;
        assert!(follow_ups.is_empty());
    }

    #[tokio::test]
    async fn heuristic_extracts_entities() {
        let retriever = MultiHopRetriever::new(None, test_config());
        let contents = vec![
            "The Authentication Module handles OAuth tokens. \
             The Performance Monitor tracks response times."
                .into(),
        ];
        let follow_ups = retriever
            .plan_follow_ups("auth perf", &contents, &HashSet::new())
            .await;
        // Should extract capitalized phrases like "Authentication Module", "Performance Monitor"
        assert!(!follow_ups.is_empty());
    }

    #[test]
    fn token_estimation() {
        assert_eq!(MultiHopRetriever::estimate_tokens("hello world"), 2);
        assert_eq!(MultiHopRetriever::estimate_tokens(""), 0);
    }

    #[test]
    fn budget_check() {
        let retriever = MultiHopRetriever::new(None, test_config());
        assert!(retriever.within_budget(100));
        assert!(!retriever.within_budget(3000));
    }
}
