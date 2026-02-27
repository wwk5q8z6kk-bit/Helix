use hx_core::{KnowledgeNode, NodeKind, SearchStrategy};
use serde_json::Value;

pub const SAVED_SEARCH_TAG: &str = "saved-search";
pub const SAVED_SEARCH_SOURCE: &str = "helix:saved-search";
pub const SAVED_SEARCH_MARKER_METADATA_KEY: &str = "saved_search";
pub const SAVED_SEARCH_QUERY_METADATA_KEY: &str = "saved_search_query";
pub const SAVED_SEARCH_STRATEGY_METADATA_KEY: &str = "saved_search_strategy";
pub const SAVED_SEARCH_LIMIT_METADATA_KEY: &str = "saved_search_limit";
pub const SAVED_SEARCH_DESCRIPTION_METADATA_KEY: &str = "saved_search_description";
pub const SAVED_SEARCH_TARGET_NAMESPACE_METADATA_KEY: &str = "saved_search_target_namespace";
pub const SAVED_SEARCH_KINDS_METADATA_KEY: &str = "saved_search_kinds";
pub const SAVED_SEARCH_TAG_FILTERS_METADATA_KEY: &str = "saved_search_tags";
pub const SAVED_SEARCH_MIN_SCORE_METADATA_KEY: &str = "saved_search_min_score";
pub const SAVED_SEARCH_MIN_IMPORTANCE_METADATA_KEY: &str = "saved_search_min_importance";
const DEFAULT_SAVED_SEARCH_LIMIT: usize = 10;
const MAX_SAVED_SEARCH_LIMIT: usize = 200;

#[derive(Debug, Clone)]
pub struct SavedSearchDefinition {
    pub name: String,
    pub description: Option<String>,
    pub query: String,
    pub strategy: SearchStrategy,
    pub limit: usize,
    pub target_namespace: Option<String>,
    pub kinds: Vec<NodeKind>,
    pub tags: Vec<String>,
    pub min_score: Option<f64>,
    pub min_importance: Option<f64>,
}

pub fn is_saved_search_node(node: &KnowledgeNode) -> bool {
    let tagged = node
        .tags
        .iter()
        .any(|tag| tag.eq_ignore_ascii_case(SAVED_SEARCH_TAG));
    let marker = node
        .metadata
        .get(SAVED_SEARCH_MARKER_METADATA_KEY)
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let has_query = !node.content.trim().is_empty()
        || node
            .metadata
            .get(SAVED_SEARCH_QUERY_METADATA_KEY)
            .and_then(Value::as_str)
            .is_some_and(|value| !value.trim().is_empty());

    node.kind == NodeKind::Bookmark && has_query && (tagged || marker)
}

pub fn saved_search_definition_from_node(node: &KnowledgeNode) -> Option<SavedSearchDefinition> {
    if !is_saved_search_node(node) {
        return None;
    }

    let query = node
        .metadata
        .get(SAVED_SEARCH_QUERY_METADATA_KEY)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| node.content.trim())
        .to_string();
    if query.is_empty() {
        return None;
    }

    let name = node
        .title
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Saved Search")
        .to_string();

    let strategy = node
        .metadata
        .get(SAVED_SEARCH_STRATEGY_METADATA_KEY)
        .and_then(Value::as_str)
        .and_then(|value| value.parse::<SearchStrategy>().ok())
        .unwrap_or(SearchStrategy::Hybrid);

    let limit = node
        .metadata
        .get(SAVED_SEARCH_LIMIT_METADATA_KEY)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(DEFAULT_SAVED_SEARCH_LIMIT)
        .clamp(1, MAX_SAVED_SEARCH_LIMIT);

    let description = node
        .metadata
        .get(SAVED_SEARCH_DESCRIPTION_METADATA_KEY)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    let target_namespace = node
        .metadata
        .get(SAVED_SEARCH_TARGET_NAMESPACE_METADATA_KEY)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string);

    let kinds = node
        .metadata
        .get(SAVED_SEARCH_KINDS_METADATA_KEY)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter_map(|value| value.parse::<NodeKind>().ok())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let tags = node
        .metadata
        .get(SAVED_SEARCH_TAG_FILTERS_METADATA_KEY)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let min_score = node
        .metadata
        .get(SAVED_SEARCH_MIN_SCORE_METADATA_KEY)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite());

    let min_importance = node
        .metadata
        .get(SAVED_SEARCH_MIN_IMPORTANCE_METADATA_KEY)
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite());

    Some(SavedSearchDefinition {
        name,
        description,
        query,
        strategy,
        limit,
        target_namespace,
        kinds,
        tags,
        min_score,
        min_importance,
    })
}

pub fn apply_saved_search_definition(node: &mut KnowledgeNode, definition: &SavedSearchDefinition) {
    node.kind = NodeKind::Bookmark;
    node.title = Some(definition.name.clone());
    node.content = definition.query.clone();
    node.source = Some(SAVED_SEARCH_SOURCE.to_string());

    if !node
        .tags
        .iter()
        .any(|tag| tag.eq_ignore_ascii_case(SAVED_SEARCH_TAG))
    {
        node.tags.push(SAVED_SEARCH_TAG.to_string());
    }

    node.metadata.insert(
        SAVED_SEARCH_MARKER_METADATA_KEY.to_string(),
        Value::Bool(true),
    );
    node.metadata.insert(
        SAVED_SEARCH_QUERY_METADATA_KEY.to_string(),
        Value::String(definition.query.clone()),
    );
    node.metadata.insert(
        SAVED_SEARCH_STRATEGY_METADATA_KEY.to_string(),
        Value::String(search_strategy_to_str(definition.strategy).to_string()),
    );
    node.metadata.insert(
        SAVED_SEARCH_LIMIT_METADATA_KEY.to_string(),
        Value::from(definition.limit),
    );

    upsert_optional_string(
        &mut node.metadata,
        SAVED_SEARCH_DESCRIPTION_METADATA_KEY,
        definition.description.as_deref(),
    );
    upsert_optional_string(
        &mut node.metadata,
        SAVED_SEARCH_TARGET_NAMESPACE_METADATA_KEY,
        definition.target_namespace.as_deref(),
    );
    upsert_string_array(
        &mut node.metadata,
        SAVED_SEARCH_KINDS_METADATA_KEY,
        definition.kinds.iter().map(ToString::to_string).collect(),
    );
    upsert_string_array(
        &mut node.metadata,
        SAVED_SEARCH_TAG_FILTERS_METADATA_KEY,
        definition.tags.clone(),
    );
    upsert_optional_f64(
        &mut node.metadata,
        SAVED_SEARCH_MIN_SCORE_METADATA_KEY,
        definition.min_score,
    );
    upsert_optional_f64(
        &mut node.metadata,
        SAVED_SEARCH_MIN_IMPORTANCE_METADATA_KEY,
        definition.min_importance,
    );
}

fn search_strategy_to_str(strategy: SearchStrategy) -> &'static str {
    match strategy {
        SearchStrategy::Vector => "vector",
        SearchStrategy::FullText => "fulltext",
        SearchStrategy::Hybrid => "hybrid",
        SearchStrategy::Graph => "graph",
    }
}

fn upsert_optional_string(
    metadata: &mut std::collections::HashMap<String, Value>,
    key: &str,
    value: Option<&str>,
) {
    if let Some(value) = value.map(str::trim).filter(|value| !value.is_empty()) {
        metadata.insert(key.to_string(), Value::String(value.to_string()));
    } else {
        metadata.remove(key);
    }
}

fn upsert_string_array(
    metadata: &mut std::collections::HashMap<String, Value>,
    key: &str,
    values: Vec<String>,
) {
    if values.is_empty() {
        metadata.remove(key);
        return;
    }

    let normalized: Vec<Value> = values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .map(Value::String)
        .collect();
    if normalized.is_empty() {
        metadata.remove(key);
    } else {
        metadata.insert(key.to_string(), Value::Array(normalized));
    }
}

fn upsert_optional_f64(
    metadata: &mut std::collections::HashMap<String, Value>,
    key: &str,
    value: Option<f64>,
) {
    if let Some(value) = value.filter(|value| value.is_finite()) {
        metadata.insert(key.to_string(), Value::from(value));
    } else {
        metadata.remove(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_and_reads_saved_search_definition_roundtrip() {
        let mut node = KnowledgeNode::new(NodeKind::Fact, "placeholder");
        let definition = SavedSearchDefinition {
            name: "Due Tasks".to_string(),
            description: Some("High priority due tasks".to_string()),
            query: "deadline overdue".to_string(),
            strategy: SearchStrategy::Hybrid,
            limit: 25,
            target_namespace: Some("ops".to_string()),
            kinds: vec![NodeKind::Task],
            tags: vec!["urgent".to_string()],
            min_score: Some(0.2),
            min_importance: Some(0.6),
        };

        apply_saved_search_definition(&mut node, &definition);
        let loaded = saved_search_definition_from_node(&node).expect("definition should parse");

        assert_eq!(loaded.name, definition.name);
        assert_eq!(loaded.query, definition.query);
        assert_eq!(loaded.limit, definition.limit);
        assert_eq!(loaded.strategy, definition.strategy);
        assert_eq!(loaded.kinds, definition.kinds);
        assert_eq!(loaded.tags, definition.tags);
        assert_eq!(loaded.target_namespace, definition.target_namespace);
        assert_eq!(loaded.min_score, definition.min_score);
        assert_eq!(loaded.min_importance, definition.min_importance);
        assert!(is_saved_search_node(&node));
    }

    #[test]
    fn rejects_unmarked_bookmark_without_query() {
        let mut node = KnowledgeNode::new(NodeKind::Bookmark, "".to_string());
        node.tags = vec!["bookmark".to_string()];
        assert!(!is_saved_search_node(&node));
        assert!(saved_search_definition_from_node(&node).is_none());
    }
}
