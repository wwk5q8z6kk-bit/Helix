use hx_core::*;
use hx_storage::unified::UnifiedStore;
use std::sync::Arc;


/// Heuristic conflict detection: finds contradictions between knowledge nodes.
pub struct ConflictDetector;

/// Negation/opposition keyword pairs for contradiction detection.
const OPPOSITION_PAIRS: &[(&str, &str)] = &[
    ("increase", "decrease"),
    ("enable", "disable"),
    ("enabled", "disabled"),
    ("allow", "deny"),
    ("allow", "block"),
    ("true", "false"),
    ("yes", "no"),
    ("good", "bad"),
    ("safe", "unsafe"),
    ("secure", "insecure"),
    ("always", "never"),
    ("required", "optional"),
    ("deprecated", "recommended"),
    ("supported", "unsupported"),
    ("possible", "impossible"),
    ("correct", "incorrect"),
    ("valid", "invalid"),
    ("success", "failure"),
    ("accept", "reject"),
    ("open", "closed"),
    ("start", "stop"),
    ("add", "remove"),
    ("include", "exclude"),
    ("up", "down"),
    ("fast", "slow"),
    ("new", "old"),
];

impl ConflictDetector {
    /// Check a newly-ingested node against similar existing nodes for contradictions.
    /// Returns a list of conflict alerts to surface in the inbox.
    pub async fn detect_conflicts(
        store: &Arc<UnifiedStore>,
        new_node: &KnowledgeNode,
        threshold: f64,
    ) -> MvResult<Vec<ConflictAlert>> {
        let vectors = match store.vectors.as_ref() {
            Some(v) => v,
            None => return Ok(Vec::new()),
        };

        // Embed the new node's content
        let embedding = match store.embedder.embed(&new_node.content).await {
            Ok(e) => e,
            Err(_) => return Ok(Vec::new()),
        };

        // Find top-5 semantically similar existing nodes
        let similar = vectors
            .search(embedding, 5, 0.5, Some(&new_node.namespace))
            .await?;

        let mut alerts = Vec::new();
        for (candidate_id, similarity) in similar {
            if candidate_id == new_node.id {
                continue;
            }

            let candidate = match store.nodes.get(candidate_id).await? {
                Some(n) => n,
                None => continue,
            };

            // Compute contradiction score
            let (conflict_type, score, explanation) =
                Self::score_contradiction(new_node, &candidate, similarity);

            if score >= threshold {
                alerts.push(ConflictAlert::new(
                    candidate_id,
                    new_node.id,
                    conflict_type,
                    score,
                    explanation,
                ));
            }
        }

        Ok(alerts)
    }

    /// Score the level of contradiction between two nodes.
    /// Returns (conflict_type, score, explanation).
    fn score_contradiction(
        new_node: &KnowledgeNode,
        existing: &KnowledgeNode,
        similarity: f64,
    ) -> (ConflictType, f64, String) {
        let new_lower = new_node.content.to_lowercase();
        let existing_lower = existing.content.to_lowercase();

        // Check for temporal supersession: same topic, newer date
        if similarity > 0.85 && new_node.temporal.created_at > existing.temporal.created_at {
            let age_days = (new_node.temporal.created_at - existing.temporal.created_at)
                .num_days();
            if age_days > 30 {
                let explanation = format!(
                    "Node '{}' may supersede '{}' (same topic, {} days newer)",
                    new_node.title.as_deref().unwrap_or("untitled"),
                    existing.title.as_deref().unwrap_or("untitled"),
                    age_days,
                );
                return (ConflictType::Supersession, 0.7 + (similarity - 0.85) * 2.0, explanation);
            }
        }

        // Check for negation patterns
        let negation_score = Self::negation_score(&new_lower, &existing_lower);

        // Check for opposition keyword pairs
        let opposition_score = Self::opposition_score(&new_lower, &existing_lower);

        let combined = (negation_score * 0.5 + opposition_score * 0.3 + similarity * 0.2)
            .min(1.0);

        if negation_score > 0.3 || opposition_score > 0.3 {
            let explanation = format!(
                "Potential contradiction between '{}' and '{}' (negation={:.2}, opposition={:.2}, similarity={:.2})",
                new_node.title.as_deref().unwrap_or("untitled"),
                existing.title.as_deref().unwrap_or("untitled"),
                negation_score,
                opposition_score,
                similarity,
            );
            (ConflictType::Contradiction, combined, explanation)
        } else if similarity > 0.9 {
            let explanation = format!(
                "Very similar content in '{}' and '{}' — possible ambiguity (similarity={:.2})",
                new_node.title.as_deref().unwrap_or("untitled"),
                existing.title.as_deref().unwrap_or("untitled"),
                similarity,
            );
            (ConflictType::Ambiguity, similarity * 0.6, explanation)
        } else {
            (ConflictType::Ambiguity, 0.0, String::new())
        }
    }

    /// Detect negation patterns (e.g., "X is good" vs "X is not good").
    fn negation_score(a: &str, b: &str) -> f64 {
        let negation_words = ["not", "no", "never", "don't", "doesn't", "isn't", "aren't",
            "won't", "can't", "shouldn't", "cannot", "neither", "nor"];

        let a_has_negation = negation_words.iter().any(|w| {
            a.split_whitespace().any(|word| word == *w)
        });
        let b_has_negation = negation_words.iter().any(|w| {
            b.split_whitespace().any(|word| word == *w)
        });

        // One has negation, other doesn't → likely contradiction
        if a_has_negation != b_has_negation {
            0.6
        } else {
            0.0
        }
    }

    /// Detect opposition keyword pairs.
    fn opposition_score(a: &str, b: &str) -> f64 {
        let a_words: Vec<&str> = a.split_whitespace().collect();
        let b_words: Vec<&str> = b.split_whitespace().collect();

        let mut matches = 0;
        for (w1, w2) in OPPOSITION_PAIRS {
            let a_has_w1 = a_words.iter().any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *w1);
            let b_has_w2 = b_words.iter().any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *w2);
            let a_has_w2 = a_words.iter().any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *w2);
            let b_has_w1 = b_words.iter().any(|w| w.trim_matches(|c: char| !c.is_alphanumeric()) == *w1);

            if (a_has_w1 && b_has_w2) || (a_has_w2 && b_has_w1) {
                matches += 1;
            }
        }

        if matches == 0 {
            0.0
        } else {
            (matches as f64 * 0.3).min(1.0)
        }
    }
}
