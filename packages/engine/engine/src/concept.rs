//! Concept extraction and graph enrichment.
//!
//! Extracts key concepts from knowledge node content using:
//! 1. RAKE (Rapid Automatic Keyword Extraction) — pure Rust, always available
//! 2. LLM-based entity/concept extraction — when LLM is available
//!
//! Extracted concepts are stored as entity nodes and linked to source nodes
//! via `mentions` relationships, making the knowledge graph self-organizing.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tracing::{debug, warn};
use uuid::Uuid;

use crate::llm::{ChatMessage, CompletionParams, LlmProvider};

/// A concept extracted from text.
#[derive(Debug, Clone)]
pub struct ExtractedConcept {
    pub text: String,
    pub kind: ConceptKind,
    pub score: f64,
}

/// Kind of extracted concept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptKind {
    Keyword,
    Person,
    Organization,
    Location,
    Date,
    Technology,
    Topic,
}

impl std::fmt::Display for ConceptKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Keyword => write!(f, "keyword"),
            Self::Person => write!(f, "person"),
            Self::Organization => write!(f, "organization"),
            Self::Location => write!(f, "location"),
            Self::Date => write!(f, "date"),
            Self::Technology => write!(f, "technology"),
            Self::Topic => write!(f, "topic"),
        }
    }
}

/// Concept extractor combining RAKE and optional LLM extraction.
pub struct ConceptExtractor {
    llm: Option<Arc<dyn LlmProvider>>,
    stop_words: HashSet<&'static str>,
}

impl ConceptExtractor {
    pub fn new(llm: Option<Arc<dyn LlmProvider>>) -> Self {
        Self {
            llm,
            stop_words: Self::default_stop_words(),
        }
    }

    /// Extract concepts from text using all available methods.
    pub async fn extract(&self, text: &str) -> Vec<ExtractedConcept> {
        let mut concepts = Vec::new();

        // Always run RAKE (pure Rust, zero deps)
        let rake_keywords = self.rake_extract(text);
        concepts.extend(rake_keywords);

        // If LLM is available, also extract named entities
        if let Some(ref llm) = self.llm {
            match self.llm_extract(llm, text).await {
                Ok(entities) => {
                    // Merge LLM results, deduplicating against RAKE results
                    let existing: HashSet<String> = concepts
                        .iter()
                        .map(|c| c.text.to_lowercase())
                        .collect();
                    for entity in entities {
                        if !existing.contains(&entity.text.to_lowercase()) {
                            concepts.push(entity);
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "LLM concept extraction failed");
                }
            }
        }

        // Sort by score descending
        concepts.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to top 20 concepts
        concepts.truncate(20);

        debug!(
            count = concepts.len(),
            text_len = text.len(),
            "concepts extracted"
        );

        concepts
    }

    /// RAKE keyword extraction: split text into candidate phrases using stop words
    /// and punctuation as delimiters, then score phrases by word degree/frequency.
    fn rake_extract(&self, text: &str) -> Vec<ExtractedConcept> {
        // Tokenize: split on non-alphanumeric (keeping hyphens within words)
        let words: Vec<&str> = text.split(|c: char| {
            !c.is_alphanumeric() && c != '-' && c != '\''
        })
        .filter(|w| !w.is_empty())
        .collect();

        // Build candidate phrases: sequences of non-stop-words
        let mut phrases: Vec<Vec<String>> = Vec::new();
        let mut current_phrase: Vec<String> = Vec::new();

        for word in &words {
            let lower = word.to_lowercase();
            if self.stop_words.contains(lower.as_str()) || lower.len() < 2 {
                if !current_phrase.is_empty() {
                    phrases.push(current_phrase.clone());
                    current_phrase.clear();
                }
            } else {
                current_phrase.push(lower);
            }
        }
        if !current_phrase.is_empty() {
            phrases.push(current_phrase);
        }

        // Calculate word frequency and degree
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut word_degree: HashMap<String, usize> = HashMap::new();

        for phrase in &phrases {
            let degree = phrase.len();
            for word in phrase {
                *word_freq.entry(word.clone()).or_default() += 1;
                *word_degree.entry(word.clone()).or_default() += degree;
            }
        }

        // Score each phrase: sum of word_degree / word_frequency for each word
        let mut phrase_scores: Vec<(String, f64)> = Vec::new();
        let mut seen = HashSet::new();

        for phrase in &phrases {
            let phrase_text = phrase.join(" ");
            if seen.contains(&phrase_text) || phrase_text.len() < 3 {
                continue;
            }
            seen.insert(phrase_text.clone());

            let score: f64 = phrase
                .iter()
                .map(|w| {
                    let freq = *word_freq.get(w).unwrap_or(&1) as f64;
                    let degree = *word_degree.get(w).unwrap_or(&1) as f64;
                    degree / freq
                })
                .sum();

            phrase_scores.push((phrase_text, score));
        }

        // Sort by score and take top results
        phrase_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        phrase_scores
            .into_iter()
            .take(10)
            .map(|(text, score)| ExtractedConcept {
                text,
                kind: ConceptKind::Keyword,
                score,
            })
            .collect()
    }

    /// LLM-based concept and entity extraction.
    async fn llm_extract(
        &self,
        llm: &Arc<dyn LlmProvider>,
        text: &str,
    ) -> Result<Vec<ExtractedConcept>, String> {
        let truncated = if text.len() > 1000 {
            format!("{}...", &text[..1000])
        } else {
            text.to_string()
        };

        let messages = vec![
            ChatMessage::system(
                "Extract named entities and key concepts from the text. \
                 Return a JSON array of objects with fields:\n\
                 - \"text\": the entity/concept name\n\
                 - \"kind\": one of person, organization, location, date, technology, topic\n\
                 - \"confidence\": 0.0-1.0\n\
                 \n\
                 Return ONLY the JSON array, no other text. Max 10 items.",
            ),
            ChatMessage::user(truncated),
        ];

        let params = CompletionParams {
            max_tokens: Some(512),
            temperature: Some(0.1),
            ..Default::default()
        };

        let result = llm
            .complete(&messages, &params)
            .await
            .map_err(|e| format!("LLM extraction failed: {e}"))?;

        // Parse JSON array from response
        let json_str = if let Some(start) = result.find('[') {
            if let Some(end) = result.rfind(']') {
                &result[start..=end]
            } else {
                return Ok(vec![]);
            }
        } else {
            return Ok(vec![]);
        };

        let raw: Vec<serde_json::Value> =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse: {e}"))?;

        let mut entities = Vec::new();
        for item in raw.iter().take(10) {
            let text = item["text"].as_str().unwrap_or("").to_string();
            if text.is_empty() {
                continue;
            }
            let kind_str = item["kind"].as_str().unwrap_or("topic");
            let kind = match kind_str {
                "person" => ConceptKind::Person,
                "organization" => ConceptKind::Organization,
                "location" => ConceptKind::Location,
                "date" => ConceptKind::Date,
                "technology" => ConceptKind::Technology,
                _ => ConceptKind::Topic,
            };
            let confidence = item["confidence"].as_f64().unwrap_or(0.5);
            entities.push(ExtractedConcept {
                text,
                kind,
                score: confidence,
            });
        }

        Ok(entities)
    }

    /// Default English stop words for RAKE.
    fn default_stop_words() -> HashSet<&'static str> {
        [
            "a", "about", "above", "after", "again", "against", "all", "am", "an",
            "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
            "before", "being", "below", "between", "both", "but", "by", "can",
            "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
            "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
            "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
            "have", "haven't", "having", "he", "her", "here", "hers", "herself",
            "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't",
            "it", "its", "itself", "just", "let", "like", "may", "me", "might",
            "more", "most", "must", "mustn't", "my", "myself", "no", "nor", "not",
            "of", "off", "on", "once", "only", "or", "other", "our", "ours",
            "ourselves", "out", "over", "own", "same", "she", "should", "shouldn't",
            "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
            "themselves", "then", "there", "these", "they", "this", "those",
            "through", "to", "too", "under", "until", "up", "very", "was",
            "wasn't", "we", "were", "weren't", "what", "when", "where", "which",
            "while", "who", "whom", "why", "will", "with", "won't", "would",
            "wouldn't", "you", "your", "yours", "yourself", "yourselves",
        ]
        .into_iter()
        .collect()
    }
}

/// Graph statistics for knowledge exploration.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub orphan_nodes: usize,
    pub hub_nodes: Vec<(Uuid, String, usize)>,
    pub density: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rake_extracts_keywords() {
        let extractor = ConceptExtractor::new(None);
        let text = "Rust programming language provides memory safety \
                    without garbage collection through its ownership system";
        let concepts = extractor.rake_extract(text);
        assert!(!concepts.is_empty());
        // Should extract multi-word phrases
        let texts: Vec<&str> = concepts.iter().map(|c| c.text.as_str()).collect();
        assert!(
            texts.iter().any(|t| t.contains("rust") || t.contains("programming")),
            "expected 'rust' or 'programming' in {:?}",
            texts
        );
    }

    #[test]
    fn rake_handles_empty_text() {
        let extractor = ConceptExtractor::new(None);
        let concepts = extractor.rake_extract("");
        assert!(concepts.is_empty());
    }

    #[test]
    fn rake_handles_all_stop_words() {
        let extractor = ConceptExtractor::new(None);
        let concepts = extractor.rake_extract("the a an is are was were");
        assert!(concepts.is_empty());
    }

    #[tokio::test]
    async fn extract_without_llm() {
        let extractor = ConceptExtractor::new(None);
        let text = "Machine Learning algorithms process training data \
                    to build predictive models for classification tasks";
        let concepts = extractor.extract(text).await;
        assert!(!concepts.is_empty());
        assert!(concepts.iter().all(|c| c.kind == ConceptKind::Keyword));
    }

    #[test]
    fn concept_kind_display() {
        assert_eq!(ConceptKind::Person.to_string(), "person");
        assert_eq!(ConceptKind::Technology.to_string(), "technology");
        assert_eq!(ConceptKind::Keyword.to_string(), "keyword");
    }

    #[test]
    fn rake_scores_multiword_phrases_higher() {
        let extractor = ConceptExtractor::new(None);
        let text = "knowledge graph database stores entity relationships \
                    and supports graph traversal queries efficiently. \
                    knowledge graph systems enable semantic search.";
        let concepts = extractor.rake_extract(text);
        assert!(!concepts.is_empty());
        // Multi-word phrases should score higher due to RAKE's degree/frequency ratio
        let top = &concepts[0];
        assert!(top.score > 1.0, "top score should be > 1.0, got {}", top.score);
    }
}
