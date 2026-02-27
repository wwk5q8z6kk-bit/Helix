use hx_core::SearchResult;

#[derive(Debug, Clone, PartialEq)]
pub struct LinkSuggestionCandidate {
    pub node_id: uuid::Uuid,
    pub title: String,
    pub heading: Option<String>,
    pub preview: Option<String>,
    pub namespace: String,
    pub score: f64,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompletionSuggestionSource {
    pub node_id: uuid::Uuid,
    pub title: String,
    pub namespace: String,
    pub score: f64,
}

fn tokenize_alpha_numeric(value: &str) -> Vec<String> {
    value
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let normalized = token.trim().to_ascii_lowercase();
            if normalized.len() >= 3 {
                Some(normalized)
            } else {
                None
            }
        })
        .collect()
}

fn split_candidate_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|sentence| sentence.len() >= 24 && sentence.len() <= 220)
        .map(|sentence| sentence.replace('\t', " "))
        .map(|sentence| sentence.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect()
}

fn parse_markdown_heading(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    let prefix_len = trimmed.chars().take_while(|ch| *ch == '#').count();
    if prefix_len == 0 || prefix_len > 6 {
        return None;
    }
    let without_prefix = trimmed.get(prefix_len..)?;
    let cleaned = without_prefix.trim().trim_matches('#').trim();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned.to_string())
    }
}

fn normalize_preview(raw: &str) -> Option<String> {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim();
    if trimmed.len() < 8 {
        return None;
    }

    let max_len = 220usize;
    let mut preview = trimmed.to_string();
    if preview.len() > max_len {
        preview.truncate(max_len);
        preview.push_str("...");
    }
    Some(preview)
}

fn extract_markdown_heading_entries(text: &str) -> Vec<(String, Option<String>)> {
    let lines: Vec<&str> = text.lines().collect();
    let mut headings = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(heading) = parse_markdown_heading(line) else {
            continue;
        };

        let dedupe_key = heading.to_ascii_lowercase();
        if !seen.insert(dedupe_key) {
            continue;
        }

        let mut preview_segments = Vec::new();
        for next_line in lines.iter().skip(index + 1) {
            if parse_markdown_heading(next_line).is_some() {
                break;
            }
            let candidate = next_line
                .trim()
                .trim_start_matches(['-', '*', '>', ' '])
                .trim();
            if candidate.is_empty() {
                continue;
            }
            preview_segments.push(candidate.to_string());
            if preview_segments.len() >= 2 {
                break;
            }
        }

        let preview = normalize_preview(&preview_segments.join(" "));
        headings.push((heading, preview));
        if headings.len() >= 12 {
            break;
        }
    }

    headings
}

fn default_node_preview(content: &str) -> Option<String> {
    split_candidate_sentences(content)
        .first()
        .and_then(|sentence| normalize_preview(sentence))
}

fn fallback_completion_suggestions(limit: usize) -> Vec<String> {
    let templates = vec![
        "Add the concrete next step with an owner and due date.",
        "Capture assumptions, blockers, and dependencies before closing this note.",
        "Summarize the key decision and why it matters for future work.",
        "List follow-up questions that should be answered next.",
    ];

    templates
        .into_iter()
        .take(limit)
        .map(str::to_string)
        .collect()
}

fn source_node_title(result: &SearchResult) -> String {
    let title = result
        .node
        .title
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);
    title.unwrap_or_else(|| format!("Untitled {}", result.node.kind.as_str()))
}

pub fn collect_completion_sources(
    search_results: &[SearchResult],
    limit: usize,
) -> Vec<CompletionSuggestionSource> {
    if limit == 0 {
        return Vec::new();
    }

    let mut ranked_results: Vec<&SearchResult> = search_results.iter().collect();
    ranked_results.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut seen = std::collections::HashSet::new();
    let mut sources = Vec::new();
    for result in ranked_results {
        if !seen.insert(result.node.id) {
            continue;
        }

        sources.push(CompletionSuggestionSource {
            node_id: result.node.id,
            title: source_node_title(result),
            namespace: result.node.namespace.clone(),
            score: result.score.max(0.0),
        });

        if sources.len() == limit {
            break;
        }
    }

    sources
}

fn split_input_sentences(input: &str) -> Vec<String> {
    split_candidate_sentences(input)
        .into_iter()
        .map(|sentence| sentence.trim().to_string())
        .filter(|sentence| !sentence.is_empty())
        .collect()
}

fn is_fallback_template_sentence(sentence: &str) -> bool {
    let normalized = sentence.to_ascii_lowercase();
    normalized.contains("next step")
        || normalized.contains("follow-up questions")
        || normalized.contains("capture assumptions")
        || normalized.contains("key decision")
}

fn collect_result_sentences(search_results: &[SearchResult], limit: usize) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for result in search_results {
        for sentence in split_candidate_sentences(&result.node.content) {
            let normalized = sentence.to_ascii_lowercase();
            if !seen.insert(normalized) {
                continue;
            }
            sentences.push(sentence);
            if sentences.len() == limit {
                return sentences;
            }
        }
    }

    sentences
}

pub fn generate_summary_transform(
    input: &str,
    search_results: &[SearchResult],
    sentence_limit: usize,
) -> String {
    let limit = sentence_limit.clamp(1, 6);
    let mut summary_sentences = generate_completion_suggestions(input, search_results, limit)
        .into_iter()
        .map(|sentence| sentence.trim().to_string())
        .filter(|sentence| !sentence.is_empty())
        .collect::<Vec<_>>();

    if summary_sentences.is_empty()
        || summary_sentences
            .iter()
            .all(|sentence| is_fallback_template_sentence(sentence))
    {
        summary_sentences = collect_result_sentences(search_results, limit);
    }

    if summary_sentences.is_empty() {
        summary_sentences = split_input_sentences(input)
            .into_iter()
            .take(limit)
            .collect();
    }

    if summary_sentences.is_empty() {
        return input.trim().to_string();
    }

    summary_sentences
        .into_iter()
        .take(limit)
        .collect::<Vec<_>>()
        .join(" ")
}

fn sentence_to_action_item(sentence: &str) -> Option<String> {
    let cleaned = sentence
        .trim()
        .trim_end_matches('.')
        .trim_end_matches(';')
        .trim();
    if cleaned.len() < 12 {
        return None;
    }
    let normalized = cleaned
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string();
    if normalized.len() < 12 {
        return None;
    }
    Some(normalized)
}

pub fn generate_action_items_transform(
    input: &str,
    search_results: &[SearchResult],
    item_limit: usize,
) -> Vec<String> {
    let limit = item_limit.clamp(1, 8);
    let mut items = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for sentence in generate_completion_suggestions(input, search_results, limit * 2) {
        let Some(item) = sentence_to_action_item(&sentence) else {
            continue;
        };
        let dedupe = item.to_ascii_lowercase();
        if !seen.insert(dedupe) {
            continue;
        }
        items.push(item);
        if items.len() == limit {
            return items;
        }
    }

    for sentence in split_input_sentences(input) {
        let Some(item) = sentence_to_action_item(&sentence) else {
            continue;
        };
        let dedupe = item.to_ascii_lowercase();
        if !seen.insert(dedupe) {
            continue;
        }
        items.push(item);
        if items.len() == limit {
            break;
        }
    }

    if items.is_empty() {
        vec!["Capture the next concrete step and owner.".to_string()]
    } else {
        items
    }
}

pub fn generate_refine_transform(
    input: &str,
    search_results: &[SearchResult],
    limit: usize,
) -> String {
    let summary = generate_summary_transform(input, search_results, limit.clamp(1, 4));
    let actions = generate_action_items_transform(input, search_results, limit.clamp(2, 6));

    let mut markdown = String::new();
    markdown.push_str("### Refined Summary\n");
    markdown.push_str(summary.trim());
    markdown.push_str("\n\n### Suggested Next Steps\n");
    for action in actions {
        markdown.push_str("- ");
        markdown.push_str(action.trim());
        markdown.push('\n');
    }
    markdown.trim_end().to_string()
}

fn extract_decisions(input: &str, search_results: &[SearchResult], limit: usize) -> Vec<String> {
    let mut decisions = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let keywords = [
        "decided",
        "decision",
        "agreed",
        "approved",
        "resolved",
        "we will",
        "we'll",
        "commit",
    ];

    let mut candidates = split_candidate_sentences(input);
    for result in search_results {
        candidates.extend(split_candidate_sentences(&result.node.content));
    }

    for sentence in candidates {
        let normalized = sentence.to_ascii_lowercase();
        if !keywords.iter().any(|keyword| normalized.contains(keyword)) {
            continue;
        }
        if !seen.insert(normalized.clone()) {
            continue;
        }
        let trimmed = sentence.trim();
        if trimmed.len() < 8 {
            continue;
        }
        decisions.push(trimmed.to_string());
        if decisions.len() >= limit {
            break;
        }
    }

    decisions
}

fn extract_questions(input: &str, search_results: &[SearchResult], limit: usize) -> Vec<String> {
    let mut questions = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut candidates = split_candidate_sentences(input);
    for result in search_results {
        candidates.extend(split_candidate_sentences(&result.node.content));
    }

    for sentence in candidates {
        let trimmed = sentence.trim();
        if trimmed.len() < 6 {
            continue;
        }
        if !trimmed.contains('?') && !trimmed.to_ascii_lowercase().starts_with("q:") {
            continue;
        }
        let normalized = trimmed.to_ascii_lowercase();
        if !seen.insert(normalized) {
            continue;
        }
        questions.push(trimmed.trim_end_matches('?').to_string() + "?");
        if questions.len() >= limit {
            break;
        }
    }

    questions
}

pub fn generate_meeting_notes_transform(
    input: &str,
    search_results: &[SearchResult],
    limit: usize,
) -> String {
    let summary = generate_summary_transform(input, search_results, limit.clamp(1, 4));
    let actions = generate_action_items_transform(input, search_results, limit.clamp(2, 6));
    let decisions = extract_decisions(input, search_results, limit.clamp(1, 6));
    let questions = extract_questions(input, search_results, limit.clamp(1, 6));

    let mut markdown = String::new();
    markdown.push_str("### Meeting Summary\n");
    markdown.push_str(summary.trim());
    markdown.push_str("\n\n### Decisions\n");
    if decisions.is_empty() {
        markdown.push_str("- No explicit decisions captured.\n");
    } else {
        for decision in decisions {
            markdown.push_str("- ");
            markdown.push_str(decision.trim());
            markdown.push('\n');
        }
    }

    markdown.push_str("\n### Action Items\n");
    for action in actions {
        markdown.push_str("- [ ] ");
        markdown.push_str(action.trim());
        markdown.push('\n');
    }

    markdown.push_str("\n### Open Questions\n");
    if questions.is_empty() {
        markdown.push_str("- No open questions captured.\n");
    } else {
        for question in questions {
            markdown.push_str("- ");
            markdown.push_str(question.trim());
            markdown.push('\n');
        }
    }

    markdown.trim_end().to_string()
}

pub fn generate_completion_suggestions(
    input: &str,
    search_results: &[SearchResult],
    limit: usize,
) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }

    let query_terms = tokenize_alpha_numeric(input);
    if query_terms.is_empty() {
        return fallback_completion_suggestions(limit);
    }

    let mut scored_candidates: Vec<(f64, String)> = Vec::new();

    for result in search_results {
        let title_terms = result
            .node
            .title
            .as_deref()
            .map(tokenize_alpha_numeric)
            .unwrap_or_default();

        for sentence in split_candidate_sentences(&result.node.content) {
            let sentence_terms = tokenize_alpha_numeric(&sentence);
            let overlap = sentence_terms
                .iter()
                .filter(|token| query_terms.contains(*token))
                .count();
            if overlap == 0 {
                continue;
            }

            let title_overlap = sentence_terms
                .iter()
                .filter(|token| title_terms.contains(*token))
                .count();

            let score =
                (overlap as f64 * 2.0) + (title_overlap as f64 * 0.75) + (result.score * 1.5);
            scored_candidates.push((score, sentence));
        }
    }

    scored_candidates.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suggestions = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (_score, sentence) in scored_candidates {
        let normalized = sentence.to_ascii_lowercase();
        if !seen.insert(normalized) {
            continue;
        }
        suggestions.push(sentence);
        if suggestions.len() == limit {
            break;
        }
    }

    if suggestions.is_empty() {
        fallback_completion_suggestions(limit)
    } else {
        suggestions
    }
}

pub fn generate_autocomplete_completions(
    input: &str,
    search_results: &[SearchResult],
    limit: usize,
) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }

    let normalized_input = input.trim().to_ascii_lowercase();
    if normalized_input.is_empty() {
        return Vec::new();
    }

    let mut scored_candidates: Vec<(f64, String)> = Vec::new();
    for result in search_results {
        let mut candidates = Vec::new();
        if let Some(title) = result.node.title.as_deref() {
            candidates.push(title.to_string());
        }
        candidates.extend(split_candidate_sentences(&result.node.content));

        for candidate in candidates {
            let normalized_candidate = candidate.to_ascii_lowercase();
            if normalized_candidate == normalized_input {
                continue;
            }

            let starts_with = normalized_candidate.starts_with(&normalized_input);
            let contains = normalized_candidate.contains(&normalized_input);
            if !starts_with && !contains {
                continue;
            }

            let score = (if starts_with { 2.5 } else { 1.0 }) + (result.score * 1.5);
            scored_candidates.push((score, candidate));
        }
    }

    scored_candidates.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut completions = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (_score, candidate) in scored_candidates {
        let normalized = candidate.to_ascii_lowercase();
        if !seen.insert(normalized) {
            continue;
        }
        completions.push(candidate);
        if completions.len() == limit {
            break;
        }
    }

    if completions.is_empty() {
        fallback_completion_suggestions(limit)
    } else {
        completions
    }
}

pub fn generate_link_suggestions(
    input: &str,
    search_results: &[SearchResult],
    limit: usize,
    exclude_node_id: Option<uuid::Uuid>,
) -> Vec<LinkSuggestionCandidate> {
    if limit == 0 {
        return Vec::new();
    }

    let normalized_input_raw = input.trim().to_ascii_lowercase();
    if normalized_input_raw.len() < 2 {
        return Vec::new();
    }

    let has_heading_delimiter = normalized_input_raw.contains('#');
    let (raw_title_query, raw_heading_query) = match normalized_input_raw.split_once('#') {
        Some((title_query, heading_query)) => (title_query, Some(heading_query)),
        None => (normalized_input_raw.as_str(), None),
    };
    let normalized_title_query = if raw_title_query.trim().is_empty() {
        normalized_input_raw.as_str()
    } else {
        raw_title_query.trim()
    };
    let normalized_heading_query = raw_heading_query
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let query_terms: std::collections::HashSet<String> =
        tokenize_alpha_numeric(normalized_title_query)
            .into_iter()
            .collect();
    let heading_query_terms: std::collections::HashSet<String> =
        tokenize_alpha_numeric(normalized_heading_query.unwrap_or(normalized_title_query))
            .into_iter()
            .collect();

    let mut scored_candidates: Vec<(f64, LinkSuggestionCandidate)> = Vec::new();
    for result in search_results {
        let node = &result.node;
        if exclude_node_id.is_some_and(|exclude_id| exclude_id == node.id) {
            continue;
        }

        let Some(raw_title) = node
            .title
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };

        let normalized_title = raw_title.to_ascii_lowercase();
        let starts_with = normalized_title.starts_with(normalized_title_query);
        let contains = normalized_title.contains(normalized_title_query);
        let title_terms = tokenize_alpha_numeric(&normalized_title);
        let overlap_count = if query_terms.is_empty() {
            0
        } else {
            title_terms
                .iter()
                .filter(|token| query_terms.contains(*token))
                .count()
        };
        let overlap_ratio = if query_terms.is_empty() {
            0.0
        } else {
            overlap_count as f64 / query_terms.len() as f64
        };

        let title_is_relevant = starts_with || contains || overlap_count > 0 || result.score >= 0.2;
        if !title_is_relevant {
            continue;
        }

        let semantic_score = result.score.max(0.0);
        let lexical_score = if starts_with {
            2.2
        } else if contains {
            1.1
        } else {
            0.25
        };
        let score = semantic_score
            + lexical_score
            + (overlap_ratio * 1.4)
            + (title_terms.len().min(12) as f64 * 0.01);
        let reason = if starts_with {
            "title_prefix_match"
        } else if contains && overlap_count > 0 {
            "title_keyword_match"
        } else if overlap_count > 0 {
            "semantic_keyword_match"
        } else {
            "semantic_match"
        };

        scored_candidates.push((
            score,
            LinkSuggestionCandidate {
                node_id: node.id,
                title: raw_title.to_string(),
                heading: None,
                preview: default_node_preview(&node.content),
                namespace: node.namespace.clone(),
                score,
                reason: reason.to_string(),
            },
        ));

        for (heading, preview) in extract_markdown_heading_entries(&node.content) {
            let normalized_heading = heading.to_ascii_lowercase();
            let heading_query = normalized_heading_query.unwrap_or(normalized_title_query);
            let heading_starts_with = normalized_heading.starts_with(heading_query);
            let heading_contains = normalized_heading.contains(heading_query);

            let heading_terms = tokenize_alpha_numeric(&normalized_heading);
            let heading_overlap_count = if heading_query_terms.is_empty() {
                0
            } else {
                heading_terms
                    .iter()
                    .filter(|token| heading_query_terms.contains(*token))
                    .count()
            };
            let heading_overlap_ratio = if heading_query_terms.is_empty() {
                0.0
            } else {
                heading_overlap_count as f64 / heading_query_terms.len() as f64
            };

            let heading_is_relevant = if has_heading_delimiter {
                if normalized_heading_query.is_some() {
                    heading_starts_with
                        || heading_contains
                        || heading_overlap_count > 0
                        || semantic_score >= 0.45
                } else {
                    title_is_relevant
                }
            } else {
                heading_starts_with
                    || heading_contains
                    || heading_overlap_count > 0
                    || semantic_score >= 0.55
            };
            if !heading_is_relevant {
                continue;
            }

            let lexical_score = if heading_starts_with {
                2.0
            } else if heading_contains {
                1.05
            } else if has_heading_delimiter {
                0.85
            } else {
                0.2
            };
            let score = semantic_score
                + lexical_score
                + (heading_overlap_ratio * 1.6)
                + if has_heading_delimiter { 0.25 } else { 0.0 };
            let reason = if heading_starts_with {
                "heading_prefix_match"
            } else if heading_contains || heading_overlap_count > 0 {
                "heading_keyword_match"
            } else if has_heading_delimiter {
                "heading_context_match"
            } else {
                "heading_semantic_match"
            };

            scored_candidates.push((
                score,
                LinkSuggestionCandidate {
                    node_id: node.id,
                    title: raw_title.to_string(),
                    heading: Some(heading),
                    preview,
                    namespace: node.namespace.clone(),
                    score,
                    reason: reason.to_string(),
                },
            ));
        }
    }

    scored_candidates.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut suggestions = Vec::new();
    let mut seen_targets = std::collections::HashSet::new();
    for (_score, candidate) in scored_candidates {
        let normalized_target = match candidate.heading.as_deref() {
            Some(heading) => format!(
                "{}#{}",
                candidate.title.to_ascii_lowercase(),
                heading.to_ascii_lowercase()
            ),
            None => candidate.title.to_ascii_lowercase(),
        };
        if !seen_targets.insert(normalized_target) {
            continue;
        }
        suggestions.push(candidate);
        if suggestions.len() == limit {
            break;
        }
    }

    if !suggestions.is_empty() {
        return suggestions;
    }

    let mut fallback = Vec::new();
    for result in search_results {
        let node = &result.node;
        if exclude_node_id.is_some_and(|exclude_id| exclude_id == node.id) {
            continue;
        }
        let Some(raw_title) = node
            .title
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
        else {
            continue;
        };
        let normalized_title = raw_title.to_ascii_lowercase();
        if !seen_targets.insert(normalized_title) {
            continue;
        }
        fallback.push(LinkSuggestionCandidate {
            node_id: node.id,
            title: raw_title.to_string(),
            heading: None,
            preview: default_node_preview(&node.content),
            namespace: node.namespace.clone(),
            score: result.score.max(0.0),
            reason: "semantic_match".to_string(),
        });
        if fallback.len() == limit {
            break;
        }
    }

    fallback
}
