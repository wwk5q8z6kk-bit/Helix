//! Query rewriting and decomposition for the recall pipeline.
//!
//! Rewrites raw user queries before search execution to improve recall quality.
//! Supports three strategies:
//! - **Expand**: synonym/abbreviation expansion via LLM
//! - **Decompose**: break compound queries into sub-queries
//! - **HyDE**: generate a hypothetical document and embed that instead
//!
//! Falls back to passthrough when no LLM is available.

use std::sync::Arc;

use tracing::{debug, warn};

use crate::config::QueryRewriteConfig;
use crate::llm::{ChatMessage, CompletionParams, LlmProvider};
use hx_core::model::RewriteStrategy;

/// Result of query rewriting: one or more rewritten query strings.
#[derive(Debug, Clone)]
pub struct RewriteResult {
    /// The rewritten queries. For decomposition, may contain multiple sub-queries.
    pub queries: Vec<String>,
    /// The strategy that was actually applied.
    pub applied_strategy: RewriteStrategy,
    /// If HyDE was used, the hypothetical document for embedding.
    pub hyde_document: Option<String>,
}

/// Rewrites queries using LLM or heuristic fallback.
pub struct QueryRewriter {
    llm: Option<Arc<dyn LlmProvider>>,
    config: QueryRewriteConfig,
}

impl QueryRewriter {
    pub fn new(llm: Option<Arc<dyn LlmProvider>>, config: QueryRewriteConfig) -> Self {
        Self { llm, config }
    }

    /// Rewrite a query according to the given strategy.
    /// Returns the original query wrapped in a RewriteResult if rewriting is
    /// disabled or no LLM is available.
    pub async fn rewrite(
        &self,
        query: &str,
        strategy: Option<RewriteStrategy>,
    ) -> RewriteResult {
        let strategy = strategy.unwrap_or_else(|| {
            if !self.config.enabled {
                return RewriteStrategy::None;
            }
            self.config
                .default_strategy
                .parse()
                .unwrap_or(RewriteStrategy::None)
        });

        if strategy == RewriteStrategy::None {
            return RewriteResult {
                queries: vec![query.to_string()],
                applied_strategy: RewriteStrategy::None,
                hyde_document: None,
            };
        }

        match strategy {
            RewriteStrategy::None => unreachable!(),
            RewriteStrategy::Expand => self.expand(query).await,
            RewriteStrategy::Decompose => self.decompose(query).await,
            RewriteStrategy::HyDE => self.hyde(query).await,
            RewriteStrategy::Auto => self.auto_rewrite(query).await,
        }
    }

    /// Expand abbreviations and add synonyms.
    async fn expand(&self, query: &str) -> RewriteResult {
        let Some(llm) = &self.llm else {
            return self.heuristic_expand(query);
        };

        let messages = vec![
            ChatMessage::system(
                "You are a query expansion assistant for a knowledge management system. \
                 Given a search query, expand abbreviations, add relevant synonyms, and \
                 rephrase to improve search recall. Return ONLY the expanded query, nothing else. \
                 Keep it concise — one line.",
            ),
            ChatMessage::user(format!("Expand this search query: {query}")),
        ];

        let params = CompletionParams {
            max_tokens: Some(128),
            temperature: Some(0.1),
            ..Default::default()
        };

        match llm.complete(&messages, &params).await {
            Ok(expanded) => {
                let expanded = expanded.trim().to_string();
                debug!(original = query, expanded = %expanded, "query expanded");
                RewriteResult {
                    queries: vec![expanded],
                    applied_strategy: RewriteStrategy::Expand,
                    hyde_document: None,
                }
            }
            Err(e) => {
                warn!(error = %e, "LLM expand failed, using heuristic fallback");
                self.heuristic_expand(query)
            }
        }
    }

    /// Decompose a compound query into sub-queries.
    async fn decompose(&self, query: &str) -> RewriteResult {
        let Some(llm) = &self.llm else {
            return self.passthrough(query, RewriteStrategy::Decompose);
        };

        let max_sub = self.config.max_sub_queries;
        let messages = vec![
            ChatMessage::system(format!(
                "You are a query decomposition assistant for a knowledge management system. \
                 Given a compound search query, break it into {max_sub} or fewer independent \
                 sub-queries that together cover the original intent. \
                 Return ONLY the sub-queries, one per line, no numbering or bullets."
            )),
            ChatMessage::user(format!("Decompose this query: {query}")),
        ];

        let params = CompletionParams {
            max_tokens: Some(256),
            temperature: Some(0.1),
            ..Default::default()
        };

        match llm.complete(&messages, &params).await {
            Ok(result) => {
                let sub_queries: Vec<String> = result
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .take(max_sub)
                    .collect();

                if sub_queries.is_empty() {
                    return self.passthrough(query, RewriteStrategy::Decompose);
                }

                debug!(
                    original = query,
                    sub_queries = ?sub_queries,
                    "query decomposed"
                );

                RewriteResult {
                    queries: sub_queries,
                    applied_strategy: RewriteStrategy::Decompose,
                    hyde_document: None,
                }
            }
            Err(e) => {
                warn!(error = %e, "LLM decompose failed, using passthrough");
                self.passthrough(query, RewriteStrategy::Decompose)
            }
        }
    }

    /// HyDE: generate a hypothetical document that would answer the query,
    /// then use that document's embedding for vector search.
    async fn hyde(&self, query: &str) -> RewriteResult {
        let Some(llm) = &self.llm else {
            return self.passthrough(query, RewriteStrategy::HyDE);
        };

        let messages = vec![
            ChatMessage::system(
                "You are a hypothetical document generator for a knowledge management system. \
                 Given a search query, write a short passage (2-4 sentences) that would be a \
                 perfect answer to the query. Write it as if it's an existing knowledge base entry. \
                 Do NOT mention the query itself — just write the content directly.",
            ),
            ChatMessage::user(format!("Generate a hypothetical answer for: {query}")),
        ];

        let params = CompletionParams {
            max_tokens: Some(self.config.hyde_max_tokens),
            temperature: Some(0.5),
            ..Default::default()
        };

        match llm.complete(&messages, &params).await {
            Ok(hyde_doc) => {
                let hyde_doc = hyde_doc.trim().to_string();
                debug!(original = query, hyde_len = hyde_doc.len(), "HyDE document generated");

                RewriteResult {
                    queries: vec![query.to_string()],
                    applied_strategy: RewriteStrategy::HyDE,
                    hyde_document: Some(hyde_doc),
                }
            }
            Err(e) => {
                warn!(error = %e, "LLM HyDE failed, using passthrough");
                self.passthrough(query, RewriteStrategy::HyDE)
            }
        }
    }

    /// Auto mode: pick the best strategy based on query characteristics.
    async fn auto_rewrite(&self, query: &str) -> RewriteResult {
        let word_count = query.split_whitespace().count();

        // Short queries benefit from expansion
        if word_count <= 3 {
            return self.expand(query).await;
        }

        // Queries with conjunctions/commas likely need decomposition
        let has_conjunction = query.contains(" and ")
            || query.contains(" or ")
            || query.contains(',')
            || query.contains(';');
        if has_conjunction && word_count > 5 {
            return self.decompose(query).await;
        }

        // Questions benefit from HyDE
        if query.ends_with('?')
            || query.to_lowercase().starts_with("what ")
            || query.to_lowercase().starts_with("how ")
            || query.to_lowercase().starts_with("why ")
            || query.to_lowercase().starts_with("when ")
            || query.to_lowercase().starts_with("where ")
        {
            return self.hyde(query).await;
        }

        // Default: expansion for medium-length queries
        self.expand(query).await
    }

    /// Heuristic expansion without LLM: basic synonym injection and cleanup.
    fn heuristic_expand(&self, query: &str) -> RewriteResult {
        let mut expanded = query.to_string();

        // Common abbreviation expansions
        let abbreviations = [
            ("API", "API application programming interface"),
            ("DB", "DB database"),
            ("UI", "UI user interface"),
            ("UX", "UX user experience"),
            ("ML", "ML machine learning"),
            ("AI", "AI artificial intelligence"),
            ("PR", "PR pull request"),
            ("CI", "CI continuous integration"),
            ("CD", "CD continuous deployment"),
            ("OS", "OS operating system"),
            ("ORM", "ORM object relational mapping"),
            ("REST", "REST representational state transfer"),
            ("auth", "auth authentication authorization"),
            ("config", "config configuration"),
            ("env", "env environment"),
            ("deps", "deps dependencies"),
            ("impl", "impl implementation"),
            ("fn", "fn function"),
        ];

        for (abbrev, expansion) in &abbreviations {
            // Only expand if the abbreviation appears as a whole word
            let query_upper = query.to_uppercase();
            let abbrev_upper = abbrev.to_uppercase();
            if query_upper
                .split_whitespace()
                .any(|w| w == abbrev_upper)
            {
                expanded = format!("{expanded} {expansion}");
                break; // Only expand one abbreviation to avoid noise
            }
        }

        RewriteResult {
            queries: vec![expanded],
            applied_strategy: RewriteStrategy::Expand,
            hyde_document: None,
        }
    }

    /// Passthrough: return the original query unchanged.
    fn passthrough(&self, query: &str, strategy: RewriteStrategy) -> RewriteResult {
        RewriteResult {
            queries: vec![query.to_string()],
            applied_strategy: strategy,
            hyde_document: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn passthrough_when_disabled() {
        let config = QueryRewriteConfig {
            enabled: false,
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter.rewrite("test query", None).await;
        assert_eq!(result.queries, vec!["test query"]);
        assert_eq!(result.applied_strategy, RewriteStrategy::None);
        assert!(result.hyde_document.is_none());
    }

    #[tokio::test]
    async fn passthrough_with_explicit_none() {
        let config = QueryRewriteConfig {
            enabled: true,
            default_strategy: "auto".into(),
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter
            .rewrite("test query", Some(RewriteStrategy::None))
            .await;
        assert_eq!(result.queries, vec!["test query"]);
        assert_eq!(result.applied_strategy, RewriteStrategy::None);
    }

    #[tokio::test]
    async fn heuristic_expand_abbreviations() {
        let config = QueryRewriteConfig {
            enabled: true,
            default_strategy: "expand".into(),
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter.rewrite("API endpoint", None).await;
        assert!(result.queries[0].contains("application programming interface"));
        assert_eq!(result.applied_strategy, RewriteStrategy::Expand);
    }

    #[tokio::test]
    async fn decompose_fallback_without_llm() {
        let config = QueryRewriteConfig {
            enabled: true,
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter
            .rewrite("test query", Some(RewriteStrategy::Decompose))
            .await;
        assert_eq!(result.queries, vec!["test query"]);
        assert_eq!(result.applied_strategy, RewriteStrategy::Decompose);
    }

    #[tokio::test]
    async fn hyde_fallback_without_llm() {
        let config = QueryRewriteConfig {
            enabled: true,
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter
            .rewrite("what is Rust?", Some(RewriteStrategy::HyDE))
            .await;
        assert_eq!(result.queries, vec!["what is Rust?"]);
        assert!(result.hyde_document.is_none());
    }

    #[tokio::test]
    async fn auto_selects_expand_for_short() {
        let config = QueryRewriteConfig {
            enabled: true,
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter
            .rewrite("API", Some(RewriteStrategy::Auto))
            .await;
        assert_eq!(result.applied_strategy, RewriteStrategy::Expand);
    }

    #[tokio::test]
    async fn auto_selects_decompose_for_compound() {
        let config = QueryRewriteConfig {
            enabled: true,
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter
            .rewrite(
                "find all tasks related to API design and also database performance optimization",
                Some(RewriteStrategy::Auto),
            )
            .await;
        assert_eq!(result.applied_strategy, RewriteStrategy::Decompose);
    }

    #[tokio::test]
    async fn auto_selects_hyde_for_questions() {
        let config = QueryRewriteConfig {
            enabled: true,
            ..Default::default()
        };
        let rewriter = QueryRewriter::new(None, config);
        let result = rewriter
            .rewrite(
                "what is the architecture of the auth system?",
                Some(RewriteStrategy::Auto),
            )
            .await;
        assert_eq!(result.applied_strategy, RewriteStrategy::HyDE);
    }
}
