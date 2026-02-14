use std::collections::{HashMap, HashSet};

use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentReferenceSourceKind {
    WikiLink,
    MarkdownLink,
    Mention,
    SourceUrl,
}

impl ContentReferenceSourceKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WikiLink => "wikilink",
            Self::MarkdownLink => "markdown_link",
            Self::Mention => "mention",
            Self::SourceUrl => "source_url",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractedContentReferenceTarget {
    pub target: String,
    pub source_kind: ContentReferenceSourceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedContentReferenceTarget {
    pub node_id: Uuid,
    pub source_kind: ContentReferenceSourceKind,
}

#[derive(Debug, Default)]
pub struct KnowledgeVaultBacklinkResolutionIndex {
    ids_by_title: HashMap<String, Uuid>,
    ids_by_source_url: HashMap<String, Uuid>,
    existing_ids: HashSet<Uuid>,
}

impl KnowledgeVaultBacklinkResolutionIndex {
    pub fn insert_node(&mut self, id: Uuid, title: Option<&str>, source: Option<&str>) {
        self.existing_ids.insert(id);
        if let Some(title) = title {
            let normalized = normalize_wiki_link_target_key(title);
            if !normalized.is_empty() {
                self.ids_by_title.entry(normalized).or_insert(id);
            }
        }
        if let Some(source) = source.and_then(canonicalize_reference_url) {
            self.ids_by_source_url.entry(source).or_insert(id);
        }
    }

    fn resolve_target(&self, target: &str) -> Option<Uuid> {
        if let Some(id) = parse_uuid_target(target) {
            return self.existing_ids.contains(&id).then_some(id);
        }

        if let Some(source_url) = canonicalize_reference_url(target) {
            if let Some(id) = self.ids_by_source_url.get(&source_url).copied() {
                return Some(id);
            }
        }

        let normalized = normalize_wiki_link_target_key(target);
        if normalized.is_empty() {
            return None;
        }
        self.ids_by_title.get(&normalized).copied()
    }
}

fn parse_uuid_target(value: &str) -> Option<Uuid> {
    let trimmed = value.trim();
    if let Ok(id) = Uuid::parse_str(trimmed) {
        return Some(id);
    }

    for token in trimmed.split(|ch: char| !(ch.is_ascii_hexdigit() || ch == '-')) {
        if token.len() < 32 {
            continue;
        }
        if let Ok(id) = Uuid::parse_str(token) {
            return Some(id);
        }
    }
    None
}

fn trim_reference_url_suffix(mut value: String) -> String {
    loop {
        let Some(last) = value.chars().last() else {
            return value;
        };

        if matches!(last, '.' | ',' | ';' | ':' | '!' | '?') {
            value.pop();
            continue;
        }

        if last == ')' {
            let opens = value.chars().filter(|ch| *ch == '(').count();
            let closes = value.chars().filter(|ch| *ch == ')').count();
            if closes > opens {
                value.pop();
                continue;
            }
        }

        if matches!(last, ']' | '>') {
            value.pop();
            continue;
        }

        return value;
    }
}

fn canonicalize_reference_url(value: &str) -> Option<String> {
    let first_token = value.trim().split_whitespace().next()?;
    if first_token.is_empty() {
        return None;
    }
    let trimmed = first_token
        .trim_matches(['<', '>', '"', '\''])
        .trim()
        .to_string();
    if trimmed.is_empty() {
        return None;
    }
    let mut canonical = trim_reference_url_suffix(trimmed);
    if canonical.is_empty() {
        return None;
    }
    if let Some((without_fragment, _)) = canonical.split_once('#') {
        canonical = without_fragment.to_string();
    }
    if canonical.ends_with('/') {
        canonical.pop();
    }
    if !canonical.starts_with("http://") && !canonical.starts_with("https://") {
        return None;
    }
    Some(canonical.to_ascii_lowercase())
}

pub fn extract_wiki_link_targets(content: &str, max_targets: usize) -> Vec<String> {
    if max_targets == 0 || content.is_empty() {
        return Vec::new();
    }

    let mut targets = Vec::new();
    let mut search_start = 0usize;

    while search_start < content.len() && targets.len() < max_targets {
        let Some(open_rel) = content[search_start..].find("[[") else {
            break;
        };
        let link_start = search_start + open_rel + 2;
        let Some(close_rel) = content[link_start..].find("]]") else {
            break;
        };
        let link_end = link_start + close_rel;
        search_start = link_end + 2;

        let raw = &content[link_start..link_end];
        let without_alias = raw.split_once('|').map_or(raw, |(left, _)| left);
        let target = without_alias
            .split_once('#')
            .map_or(without_alias, |(left, _)| left)
            .trim();

        if !target.is_empty() {
            targets.push(target.to_string());
        }
    }

    targets
}

fn extract_markdown_link_targets(content: &str, max_targets: usize) -> Vec<String> {
    if max_targets == 0 || content.is_empty() {
        return Vec::new();
    }

    let mut targets = Vec::new();
    let bytes = content.as_bytes();
    let mut cursor = 0usize;

    while cursor < bytes.len() && targets.len() < max_targets {
        let Some(open_rel) = content[cursor..].find('[') else {
            break;
        };
        let open = cursor + open_rel;
        if open > 0 && bytes[open - 1] == b'!' {
            cursor = open + 1;
            continue;
        }

        let label_end_start = open + 1;
        let Some(close_rel) = content[label_end_start..].find(']') else {
            break;
        };
        let close = label_end_start + close_rel;

        if close + 1 >= bytes.len() || bytes[close + 1] != b'(' {
            cursor = close + 1;
            continue;
        }

        let target_start = close + 2;
        let Some(target_close_rel) = content[target_start..].find(')') else {
            break;
        };
        let target_end = target_start + target_close_rel;
        cursor = target_end + 1;

        let raw_target = content[target_start..target_end]
            .trim()
            .trim_matches(['<', '>'])
            .trim();
        if raw_target.is_empty() || raw_target.starts_with('#') {
            continue;
        }

        let target_token =
            if raw_target.starts_with("http://") || raw_target.starts_with("https://") {
                raw_target
                    .split_whitespace()
                    .next()
                    .unwrap_or(raw_target)
                    .trim()
            } else {
                raw_target
            };

        let without_fragment = target_token
            .split_once('#')
            .map_or(target_token, |(left, _)| left)
            .trim();
        if without_fragment.is_empty() {
            continue;
        }
        targets.push(without_fragment.to_string());
    }

    targets
}

fn is_mention_prefix_boundary(prev: Option<char>) -> bool {
    !prev.is_some_and(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | '/'))
}

fn extract_mention_targets(content: &str, max_targets: usize) -> Vec<String> {
    if max_targets == 0 || content.is_empty() {
        return Vec::new();
    }

    let mut targets = Vec::new();
    let mut cursor = 0usize;
    while cursor < content.len() && targets.len() < max_targets {
        let Some(next_rel) = content[cursor..].find('@') else {
            break;
        };
        let mention_start = cursor + next_rel;
        let prev = content[..mention_start].chars().next_back();
        if !is_mention_prefix_boundary(prev) {
            cursor = mention_start + 1;
            continue;
        }

        let after_at = mention_start + 1;
        let Some(first_char) = content[after_at..].chars().next() else {
            break;
        };

        if first_char == '"' || first_char == '\'' {
            let quoted_start = after_at + first_char.len_utf8();
            if let Some(close_rel) = content[quoted_start..].find(first_char) {
                let quoted_end = quoted_start + close_rel;
                let candidate = content[quoted_start..quoted_end].trim();
                if !candidate.is_empty() {
                    targets.push(candidate.to_string());
                }
                cursor = quoted_end + first_char.len_utf8();
                continue;
            }
            cursor = quoted_start;
            continue;
        }

        let mut mention_end = after_at;
        for (offset, ch) in content[after_at..].char_indices() {
            if ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.') {
                mention_end = after_at + offset + ch.len_utf8();
                continue;
            }
            break;
        }
        if mention_end <= after_at {
            cursor = after_at;
            continue;
        }

        let candidate = content[after_at..mention_end].trim();
        if !candidate.is_empty() {
            targets.push(candidate.to_string());
        }
        cursor = mention_end;
    }

    targets
}

fn find_next_url_start(content: &str, start: usize) -> Option<usize> {
    if start >= content.len() {
        return None;
    }
    let rest = &content[start..];
    let http_rel = rest.find("http://");
    let https_rel = rest.find("https://");
    match (http_rel, https_rel) {
        (Some(http), Some(https)) => Some(start + http.min(https)),
        (Some(http), None) => Some(start + http),
        (None, Some(https)) => Some(start + https),
        (None, None) => None,
    }
}

fn extract_bare_url_targets(content: &str, max_targets: usize) -> Vec<String> {
    if max_targets == 0 || content.is_empty() {
        return Vec::new();
    }

    let mut targets = Vec::new();
    let mut cursor = 0usize;
    while cursor < content.len() && targets.len() < max_targets {
        let Some(url_start) = find_next_url_start(content, cursor) else {
            break;
        };

        let mut end = content.len();
        for (offset, ch) in content[url_start..].char_indices() {
            if ch.is_whitespace() || matches!(ch, '<' | '>') {
                end = url_start + offset;
                break;
            }
        }
        if end <= url_start {
            cursor = url_start.saturating_add(1);
            continue;
        }

        let candidate = content[url_start..end].trim();
        cursor = end;
        if let Some(canonical) = canonicalize_reference_url(candidate) {
            targets.push(canonical);
        }
    }

    targets
}

fn push_reference_target(
    raw_target: &str,
    source_kind: ContentReferenceSourceKind,
    targets: &mut Vec<ExtractedContentReferenceTarget>,
    dedupe: &mut HashSet<String>,
    max_targets: usize,
) {
    if targets.len() >= max_targets {
        return;
    }

    let trimmed = raw_target.trim();
    if trimmed.is_empty() {
        return;
    }

    if let Some(url) = canonicalize_reference_url(trimmed) {
        let key = format!("url:{url}");
        if dedupe.insert(key) {
            targets.push(ExtractedContentReferenceTarget {
                target: url,
                source_kind,
            });
        }
        return;
    }

    let normalized = normalize_wiki_link_target_key(trimmed);
    if normalized.is_empty() {
        return;
    }
    let key = format!("title:{normalized}");
    if dedupe.insert(key) {
        targets.push(ExtractedContentReferenceTarget {
            target: trimmed.to_string(),
            source_kind,
        });
    }
}

pub fn extract_reference_targets_with_kind(
    content: &str,
    max_targets: usize,
) -> Vec<ExtractedContentReferenceTarget> {
    if max_targets == 0 || content.is_empty() {
        return Vec::new();
    }

    let mut targets = Vec::new();
    let mut dedupe = HashSet::new();
    for raw in extract_wiki_link_targets(content, max_targets) {
        push_reference_target(
            &raw,
            ContentReferenceSourceKind::WikiLink,
            &mut targets,
            &mut dedupe,
            max_targets,
        );
        if targets.len() == max_targets {
            return targets;
        }
    }
    for raw in extract_markdown_link_targets(content, max_targets) {
        push_reference_target(
            &raw,
            ContentReferenceSourceKind::MarkdownLink,
            &mut targets,
            &mut dedupe,
            max_targets,
        );
        if targets.len() == max_targets {
            return targets;
        }
    }
    for raw in extract_mention_targets(content, max_targets) {
        push_reference_target(
            &raw,
            ContentReferenceSourceKind::Mention,
            &mut targets,
            &mut dedupe,
            max_targets,
        );
        if targets.len() == max_targets {
            return targets;
        }
    }
    for raw in extract_bare_url_targets(content, max_targets) {
        push_reference_target(
            &raw,
            ContentReferenceSourceKind::SourceUrl,
            &mut targets,
            &mut dedupe,
            max_targets,
        );
        if targets.len() == max_targets {
            return targets;
        }
    }
    targets
}

pub fn extract_reference_targets(content: &str, max_targets: usize) -> Vec<String> {
    extract_reference_targets_with_kind(content, max_targets)
        .into_iter()
        .map(|target| target.target)
        .collect()
}

pub fn normalize_wiki_link_target_key(value: &str) -> String {
    value
        .split_whitespace()
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn resolve_reference_targets_with_kind(
    targets: &[ExtractedContentReferenceTarget],
    current_node_id: Uuid,
    index: &KnowledgeVaultBacklinkResolutionIndex,
) -> Vec<ResolvedContentReferenceTarget> {
    let mut resolved = Vec::new();
    let mut seen = HashSet::new();

    for target in targets {
        let Some(id) = index.resolve_target(&target.target) else {
            continue;
        };
        if id == current_node_id {
            continue;
        }
        if seen.insert(id) {
            resolved.push(ResolvedContentReferenceTarget {
                node_id: id,
                source_kind: target.source_kind,
            });
        }
    }

    resolved
}

pub fn resolve_reference_targets(
    targets: &[String],
    current_node_id: Uuid,
    index: &KnowledgeVaultBacklinkResolutionIndex,
) -> Vec<Uuid> {
    let typed_targets = targets
        .iter()
        .map(|target| ExtractedContentReferenceTarget {
            target: target.clone(),
            source_kind: ContentReferenceSourceKind::WikiLink,
        })
        .collect::<Vec<_>>();
    resolve_reference_targets_with_kind(&typed_targets, current_node_id, index)
        .into_iter()
        .map(|target| target.node_id)
        .collect()
}

pub fn resolve_wiki_link_targets(
    targets: &[String],
    current_node_id: Uuid,
    index: &KnowledgeVaultBacklinkResolutionIndex,
) -> Vec<Uuid> {
    resolve_reference_targets(targets, current_node_id, index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_wiki_link_targets_parses_aliases_and_heading_suffix() {
        let text = "Start [[Project Alpha]] and [[Task Queue|queue board]] with [[Roadmap#Q1]]";
        let targets = extract_wiki_link_targets(text, 10);
        assert_eq!(targets, vec!["Project Alpha", "Task Queue", "Roadmap"]);
    }

    #[test]
    fn extract_reference_targets_includes_markdown_and_urls() {
        let text = "Track [[Project Alpha]], [release board](Project Beta#Q2), [spec](https://docs.example.com/spec#overview), and https://docs.example.com/spec.";
        let targets = extract_reference_targets(text, 10);
        assert_eq!(
            targets,
            vec![
                "Project Alpha",
                "Project Beta",
                "https://docs.example.com/spec",
            ]
        );
    }

    #[test]
    fn extract_reference_targets_with_kind_tracks_origin() {
        let text = "[[Project Alpha]] [runbook](Project Beta#milestones) https://docs.example.com/spec#overview";
        let targets = extract_reference_targets_with_kind(text, 10);
        assert_eq!(
            targets,
            vec![
                ExtractedContentReferenceTarget {
                    target: "Project Alpha".to_string(),
                    source_kind: ContentReferenceSourceKind::WikiLink,
                },
                ExtractedContentReferenceTarget {
                    target: "Project Beta".to_string(),
                    source_kind: ContentReferenceSourceKind::MarkdownLink,
                },
                ExtractedContentReferenceTarget {
                    target: "https://docs.example.com/spec".to_string(),
                    source_kind: ContentReferenceSourceKind::SourceUrl,
                },
            ]
        );
    }

    #[test]
    fn extract_reference_targets_with_kind_includes_mentions_and_skips_emails() {
        let text =
            "Email me@example.com then ping @ProjectAlpha and @\"Project Beta\" and @'Ops Board'.";
        let targets = extract_reference_targets_with_kind(text, 10);
        assert_eq!(
            targets,
            vec![
                ExtractedContentReferenceTarget {
                    target: "ProjectAlpha".to_string(),
                    source_kind: ContentReferenceSourceKind::Mention,
                },
                ExtractedContentReferenceTarget {
                    target: "Project Beta".to_string(),
                    source_kind: ContentReferenceSourceKind::Mention,
                },
                ExtractedContentReferenceTarget {
                    target: "Ops Board".to_string(),
                    source_kind: ContentReferenceSourceKind::Mention,
                },
            ]
        );
    }

    #[test]
    fn resolve_reference_targets_matches_node_source_urls() {
        let current_id = Uuid::now_v7();
        let by_source = Uuid::now_v7();
        let by_title = Uuid::now_v7();

        let mut index = KnowledgeVaultBacklinkResolutionIndex::default();
        index.insert_node(current_id, Some("Current"), None);
        index.insert_node(
            by_source,
            Some("Spec Document"),
            Some("https://docs.example.com/spec"),
        );
        index.insert_node(by_title, Some("Project Alpha"), None);

        let targets = vec![
            "https://docs.example.com/spec#section-1".to_string(),
            "Project Alpha".to_string(),
            "missing".to_string(),
        ];
        let resolved = resolve_reference_targets(&targets, current_id, &index);
        assert_eq!(resolved, vec![by_source, by_title]);
    }

    #[test]
    fn resolve_reference_targets_with_kind_preserves_first_source_kind_per_node() {
        let current_id = Uuid::now_v7();
        let shared = Uuid::now_v7();

        let mut index = KnowledgeVaultBacklinkResolutionIndex::default();
        index.insert_node(current_id, Some("Current"), None);
        index.insert_node(
            shared,
            Some("Project Alpha"),
            Some("https://docs.example.com/alpha"),
        );

        let targets = vec![
            ExtractedContentReferenceTarget {
                target: "Project Alpha".to_string(),
                source_kind: ContentReferenceSourceKind::WikiLink,
            },
            ExtractedContentReferenceTarget {
                target: "https://docs.example.com/alpha".to_string(),
                source_kind: ContentReferenceSourceKind::SourceUrl,
            },
        ];
        let resolved = resolve_reference_targets_with_kind(&targets, current_id, &index);
        assert_eq!(
            resolved,
            vec![ResolvedContentReferenceTarget {
                node_id: shared,
                source_kind: ContentReferenceSourceKind::WikiLink,
            }]
        );
    }

    #[test]
    fn resolve_wiki_link_targets_prefers_uuid_or_title_and_deduplicates() {
        let current_id = Uuid::now_v7();
        let target_id = Uuid::now_v7();
        let other_id = Uuid::now_v7();

        let mut index = KnowledgeVaultBacklinkResolutionIndex::default();
        index.insert_node(current_id, Some("Current"), None);
        index.insert_node(target_id, Some("Project Alpha"), None);
        index.insert_node(other_id, Some("Roadmap"), None);

        let targets = vec![
            "Project Alpha".to_string(),
            "project   alpha".to_string(),
            target_id.to_string(),
            current_id.to_string(),
            "Roadmap".to_string(),
            "Missing".to_string(),
        ];
        let resolved = resolve_wiki_link_targets(&targets, current_id, &index);
        assert_eq!(resolved, vec![target_id, other_id]);
    }
}
