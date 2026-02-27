use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use hx_core::{FullTextIndex, KnowledgeNode, NodeStore};
use hx_index::tantivy_index::TantivyFullTextIndex;
use hx_storage::unified::UnifiedStore;

use crate::config::AiConfig;

const AUTO_TAGGING_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "we", "with", "you",
];

/// AI-assisted auto tagger that combines lexical extraction with similar-node tag transfer.
#[derive(Debug)]
pub struct KnowledgeVaultIndexNoteEmbeddingAutoTagger {
    stopwords: HashSet<&'static str>,
}

impl Default for KnowledgeVaultIndexNoteEmbeddingAutoTagger {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeVaultIndexNoteEmbeddingAutoTagger {
    pub fn new() -> Self {
        Self {
            stopwords: AUTO_TAGGING_STOPWORDS.iter().copied().collect(),
        }
    }

    pub async fn enrich_node_tags(
        &self,
        node: &mut KnowledgeNode,
        store: &UnifiedStore,
        fts: &TantivyFullTextIndex,
        ai_config: &AiConfig,
    ) {
        if !ai_config.auto_tagging_enabled {
            return;
        }

        let mut candidate_scores =
            self.collect_lexical_candidate_scores(node, ai_config.auto_tagging_min_token_length);

        if let Err(err) = self
            .collect_similarity_candidate_scores(node, store, fts, ai_config, &mut candidate_scores)
            .await
        {
            tracing::warn!(
                node_id = %node.id,
                namespace = %node.namespace,
                error = %err,
                "knowledge_vault_index_note_embedding_auto_tagger_similarity_lookup_failed"
            );
        }

        let generated_tags = self.rank_generated_tags(
            &node.tags,
            &candidate_scores,
            ai_config.auto_tagging_max_generated_tags,
        );
        let merged_tags = self.merge_tags_with_existing(
            &node.tags,
            generated_tags,
            ai_config.auto_tagging_max_total_tags,
        );

        let generated_count = merged_tags
            .len()
            .saturating_sub(node.tags.len().min(merged_tags.len()));
        if generated_count > 0 {
            tracing::info!(
                node_id = %node.id,
                namespace = %node.namespace,
                generated_count,
                total_tag_count = merged_tags.len(),
                "knowledge_vault_index_note_embedding_auto_tagger_enriched"
            );
        } else {
            tracing::debug!(
                node_id = %node.id,
                namespace = %node.namespace,
                "knowledge_vault_index_note_embedding_auto_tagger_noop"
            );
        }

        node.tags = merged_tags;
    }

    fn collect_lexical_candidate_scores(
        &self,
        node: &KnowledgeNode,
        min_token_length: usize,
    ) -> HashMap<String, f64> {
        let mut token_counts: HashMap<String, usize> = HashMap::new();
        let mut add_tokens_from = |text: &str| {
            for token in tokenize_and_normalize(text) {
                if token.len() < min_token_length || self.stopwords.contains(token.as_str()) {
                    continue;
                }
                *token_counts.entry(token).or_insert(0) += 1;
            }
        };

        if let Some(title) = node.title.as_deref() {
            add_tokens_from(title);
        }
        add_tokens_from(&node.content);

        let mut scores = HashMap::new();
        for (token, count) in token_counts {
            scores.insert(token, count as f64);
        }
        scores
    }

    async fn collect_similarity_candidate_scores(
        &self,
        node: &KnowledgeNode,
        store: &UnifiedStore,
        fts: &TantivyFullTextIndex,
        ai_config: &AiConfig,
        candidate_scores: &mut HashMap<String, f64>,
    ) -> Result<(), String> {
        if ai_config.auto_tagging_similarity_seed_limit == 0 {
            return Ok(());
        }

        let query_text = build_similarity_query(node);
        if query_text.trim().len() < 8 {
            return Ok(());
        }

        let similar_results = fts
            .search(
                &query_text,
                ai_config.auto_tagging_similarity_seed_limit + 1,
            )
            .map_err(|err| err.to_string())?;

        for (similar_id, similarity_score) in similar_results {
            if similar_id == node.id {
                continue;
            }

            let similar_node = store
                .nodes
                .get(similar_id)
                .await
                .map_err(|err| err.to_string())?;
            let Some(similar_node) = similar_node else {
                continue;
            };

            if similar_node.namespace != node.namespace {
                continue;
            }

            let boosted_similarity = similarity_score.max(0.01) * 2.0;
            for tag in similar_node.tags {
                if let Some(normalized_tag) = normalize_tag(&tag) {
                    *candidate_scores.entry(normalized_tag).or_insert(0.0) += boosted_similarity;
                }
            }
        }

        Ok(())
    }

    fn rank_generated_tags(
        &self,
        existing_tags: &[String],
        candidate_scores: &HashMap<String, f64>,
        max_generated_tags: usize,
    ) -> Vec<String> {
        if max_generated_tags == 0 {
            return Vec::new();
        }

        let existing: HashSet<String> = existing_tags
            .iter()
            .filter_map(|tag| normalize_tag(tag))
            .collect();

        let mut ranked: Vec<(String, f64)> = candidate_scores
            .iter()
            .filter(|(tag, _)| !existing.contains(*tag))
            .map(|(tag, score)| (tag.clone(), *score))
            .collect();

        ranked.sort_by(|(left_tag, left_score), (right_tag, right_score)| {
            right_score
                .partial_cmp(left_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left_tag.cmp(right_tag))
        });

        ranked
            .into_iter()
            .take(max_generated_tags)
            .map(|(tag, _)| tag)
            .collect()
    }

    fn merge_tags_with_existing(
        &self,
        existing_tags: &[String],
        generated_tags: Vec<String>,
        max_total_tags: usize,
    ) -> Vec<String> {
        let mut merged = Vec::new();
        let mut seen = HashSet::new();

        for tag in existing_tags {
            if let Some(normalized) = normalize_tag(tag) {
                if seen.insert(normalized) {
                    merged.push(tag.clone());
                }
            }
        }

        for tag in generated_tags {
            if max_total_tags > 0 && merged.len() >= max_total_tags {
                break;
            }
            if seen.insert(tag.clone()) {
                merged.push(tag);
            }
        }

        merged
    }
}

fn build_similarity_query(node: &KnowledgeNode) -> String {
    let mut query = String::new();
    if let Some(title) = node.title.as_deref() {
        query.push_str(title);
        query.push(' ');
    }
    query.push_str(&node.content);
    query.chars().take(300).collect()
}

fn tokenize_and_normalize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            current.push(ch.to_ascii_lowercase());
        } else if !current.is_empty() {
            if let Some(tag) = normalize_tag(&current) {
                tokens.push(tag);
            }
            current.clear();
        }
    }

    if !current.is_empty() {
        if let Some(tag) = normalize_tag(&current) {
            tokens.push(tag);
        }
    }

    tokens
}

fn normalize_tag(raw: &str) -> Option<String> {
    let normalized: String = raw
        .chars()
        .filter_map(|ch| {
            let lowered = ch.to_ascii_lowercase();
            if lowered.is_ascii_alphanumeric() || lowered == '-' || lowered == '_' {
                Some(lowered)
            } else {
                None
            }
        })
        .collect();

    if normalized.len() < 2 || normalized.len() > 64 {
        return None;
    }

    Some(normalized)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::NodeKind;

    #[test]
    fn tokenize_and_normalize_splits_expected_tokens() {
        let tokens = tokenize_and_normalize("Rust async memory-pipeline, tokio!");
        assert!(tokens.contains(&"rust".to_string()));
        assert!(tokens.contains(&"async".to_string()));
        assert!(tokens.contains(&"memory".to_string()));
        assert!(tokens.contains(&"pipeline".to_string()));
        assert!(tokens.contains(&"tokio".to_string()));
    }

    #[test]
    fn merge_tags_with_existing_deduplicates_case_insensitive() {
        let auto_tagger = KnowledgeVaultIndexNoteEmbeddingAutoTagger::new();
        let merged = auto_tagger.merge_tags_with_existing(
            &["Rust".to_string(), "backend".to_string()],
            vec!["rust".to_string(), "tokio".to_string()],
            10,
        );
        assert_eq!(merged, vec!["Rust", "backend", "tokio"]);
    }

    #[test]
    fn lexical_candidate_collection_applies_stopwords() {
        let auto_tagger = KnowledgeVaultIndexNoteEmbeddingAutoTagger::new();
        let node = KnowledgeNode::new(
            NodeKind::Fact,
            "The system stores memory and indexes memory quickly".to_string(),
        )
        .with_title("Memory indexing");
        let scores = auto_tagger.collect_lexical_candidate_scores(&node, 4);
        assert!(scores.contains_key("memory"));
        assert!(scores.contains_key("indexing"));
        assert!(!scores.contains_key("the"));
    }
}
