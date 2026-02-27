use chrono::Utc;
use hx_core::{KnowledgeNode, NodeKind};
use serde::{Deserialize, Serialize};

pub const NODE_VERSIONS_METADATA_KEY: &str = "node_versions";
pub const NODE_VERSION_MAX_ENTRIES: usize = 40;

const TEMPLATE_VERSIONS_METADATA_KEY: &str = "template_versions";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeVersionRecord {
    pub version_id: String,
    pub captured_at: String,
    pub kind: String,
    pub namespace: String,
    pub title: Option<String>,
    pub content: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub importance: f64,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeVersionSummary {
    pub version_id: String,
    pub captured_at: String,
    pub kind: String,
    pub namespace: String,
    pub title: Option<String>,
    pub source: Option<String>,
    pub importance: f64,
    pub tag_count: usize,
    pub content_preview: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeVersionCurrentSnapshot {
    pub kind: String,
    pub namespace: String,
    pub title: Option<String>,
    pub content: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub importance: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeVersionDiffSummary {
    pub version_line_count: usize,
    pub current_line_count: usize,
    pub added_line_count: usize,
    pub removed_line_count: usize,
    pub added_line_samples: Vec<String>,
    pub removed_line_samples: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeVersionFieldChange {
    pub field: String,
    pub changed: bool,
    pub version_value: String,
    pub current_value: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct NodeVersionDetailResponse {
    pub version: NodeVersionRecord,
    pub current: NodeVersionCurrentSnapshot,
    pub diff: NodeVersionDiffSummary,
    pub field_changes: Vec<NodeVersionFieldChange>,
}

pub fn node_metadata_for_version_snapshot(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> std::collections::HashMap<String, serde_json::Value> {
    let mut sanitized = metadata.clone();
    sanitized.remove(NODE_VERSIONS_METADATA_KEY);
    sanitized.remove(TEMPLATE_VERSIONS_METADATA_KEY);
    sanitized
}

pub fn node_version_snapshot_from_node(node: &KnowledgeNode) -> NodeVersionRecord {
    NodeVersionRecord {
        version_id: uuid::Uuid::now_v7().to_string(),
        captured_at: Utc::now().to_rfc3339(),
        kind: node.kind.to_string(),
        namespace: node.namespace.clone(),
        title: node.title.clone(),
        content: node.content.clone(),
        source: node.source.clone(),
        tags: node.tags.clone(),
        importance: node.importance,
        metadata: node_metadata_for_version_snapshot(&node.metadata),
    }
}

pub fn parse_node_versions_from_metadata(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Vec<NodeVersionRecord> {
    let Some(raw_versions) = metadata
        .get(NODE_VERSIONS_METADATA_KEY)
        .and_then(serde_json::Value::as_array)
    else {
        return Vec::new();
    };

    raw_versions
        .iter()
        .filter_map(|item| serde_json::from_value::<NodeVersionRecord>(item.clone()).ok())
        .collect()
}

pub fn set_node_versions_in_metadata(
    metadata: &mut std::collections::HashMap<String, serde_json::Value>,
    versions: &[NodeVersionRecord],
) {
    if versions.is_empty() {
        metadata.remove(NODE_VERSIONS_METADATA_KEY);
        return;
    }

    metadata.insert(
        NODE_VERSIONS_METADATA_KEY.to_string(),
        serde_json::Value::Array(
            versions
                .iter()
                .filter_map(|item| serde_json::to_value(item).ok())
                .collect(),
        ),
    );
}

pub fn push_node_version_snapshot(
    versions: &mut Vec<NodeVersionRecord>,
    snapshot: NodeVersionRecord,
) {
    versions.push(snapshot);
    if versions.len() > NODE_VERSION_MAX_ENTRIES {
        let to_drop = versions.len() - NODE_VERSION_MAX_ENTRIES;
        versions.drain(0..to_drop);
    }
}

pub fn node_authored_fields_changed(existing: &KnowledgeNode, incoming: &KnowledgeNode) -> bool {
    existing.kind != incoming.kind
        || existing.namespace != incoming.namespace
        || existing.title != incoming.title
        || existing.content != incoming.content
        || existing.source != incoming.source
        || existing.tags != incoming.tags
        || (existing.importance - incoming.importance).abs() > f64::EPSILON
        || node_metadata_for_version_snapshot(&existing.metadata)
            != node_metadata_for_version_snapshot(&incoming.metadata)
}

pub fn node_version_summaries(versions: Vec<NodeVersionRecord>) -> Vec<NodeVersionSummary> {
    versions
        .into_iter()
        .rev()
        .map(|item| NodeVersionSummary {
            version_id: item.version_id,
            captured_at: item.captured_at,
            kind: item.kind,
            namespace: item.namespace,
            title: item.title,
            source: item.source,
            importance: item.importance,
            tag_count: item.tags.len(),
            content_preview: trim_content_preview(&item.content, 180),
        })
        .collect()
}

pub fn node_version_diff_summary(
    version_content: &str,
    current_content: &str,
) -> NodeVersionDiffSummary {
    const MAX_DIFF_SAMPLES: usize = 12;
    const MAX_DIFF_SAMPLE_CHARS: usize = 180;

    let version_lines = text_line_items(version_content);
    let current_lines = text_line_items(current_content);
    let version_set = version_lines
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();
    let current_set = current_lines
        .iter()
        .cloned()
        .collect::<std::collections::HashSet<_>>();

    let removed_line_samples = version_lines
        .iter()
        .filter(|line| !current_set.contains(*line))
        .take(MAX_DIFF_SAMPLES)
        .map(|line| trim_preview_line(line, MAX_DIFF_SAMPLE_CHARS))
        .collect::<Vec<_>>();
    let added_line_samples = current_lines
        .iter()
        .filter(|line| !version_set.contains(*line))
        .take(MAX_DIFF_SAMPLES)
        .map(|line| trim_preview_line(line, MAX_DIFF_SAMPLE_CHARS))
        .collect::<Vec<_>>();

    NodeVersionDiffSummary {
        version_line_count: version_lines.len(),
        current_line_count: current_lines.len(),
        added_line_count: current_lines
            .iter()
            .filter(|line| !version_set.contains(*line))
            .count(),
        removed_line_count: version_lines
            .iter()
            .filter(|line| !current_set.contains(*line))
            .count(),
        added_line_samples,
        removed_line_samples,
    }
}

pub fn node_version_field_changes(
    version: &NodeVersionRecord,
    current: &KnowledgeNode,
) -> Vec<NodeVersionFieldChange> {
    let version_title = display_optional_text(version.title.as_deref());
    let current_title = display_optional_text(current.title.as_deref());
    let version_source = display_optional_text(version.source.as_deref());
    let current_source = display_optional_text(current.source.as_deref());
    let version_tags = display_tags(&version.tags);
    let current_tags = display_tags(&current.tags);
    let version_metadata_keys = metadata_keys_for_compare(&version.metadata);
    let current_metadata_keys = metadata_keys_for_compare(&current.metadata);

    vec![
        NodeVersionFieldChange {
            field: "kind".to_string(),
            changed: !version.kind.eq_ignore_ascii_case(&current.kind.to_string()),
            version_value: version.kind.clone(),
            current_value: current.kind.to_string(),
        },
        NodeVersionFieldChange {
            field: "namespace".to_string(),
            changed: version.namespace != current.namespace,
            version_value: version.namespace.clone(),
            current_value: current.namespace.clone(),
        },
        NodeVersionFieldChange {
            field: "title".to_string(),
            changed: version_title != current_title,
            version_value: version_title,
            current_value: current_title,
        },
        NodeVersionFieldChange {
            field: "source".to_string(),
            changed: version_source != current_source,
            version_value: version_source,
            current_value: current_source,
        },
        NodeVersionFieldChange {
            field: "importance".to_string(),
            changed: (version.importance - current.importance).abs() > f64::EPSILON,
            version_value: format!("{:.3}", version.importance),
            current_value: format!("{:.3}", current.importance),
        },
        NodeVersionFieldChange {
            field: "tags".to_string(),
            changed: normalize_string_list_for_compare(&version.tags)
                != normalize_string_list_for_compare(&current.tags),
            version_value: version_tags,
            current_value: current_tags,
        },
        NodeVersionFieldChange {
            field: "metadata_keys".to_string(),
            changed: version_metadata_keys != current_metadata_keys,
            version_value: display_tags(&version_metadata_keys),
            current_value: display_tags(&current_metadata_keys),
        },
    ]
}

pub fn node_version_detail_response(
    version: NodeVersionRecord,
    current: &KnowledgeNode,
) -> NodeVersionDetailResponse {
    let field_changes = node_version_field_changes(&version, current);
    let current_snapshot = NodeVersionCurrentSnapshot {
        kind: current.kind.to_string(),
        namespace: current.namespace.clone(),
        title: current.title.clone(),
        content: current.content.clone(),
        source: current.source.clone(),
        tags: current.tags.clone(),
        importance: current.importance,
    };
    let diff = node_version_diff_summary(&version.content, &current_snapshot.content);

    NodeVersionDetailResponse {
        version,
        current: current_snapshot,
        diff,
        field_changes,
    }
}

pub fn apply_node_version(existing: &KnowledgeNode, version: &NodeVersionRecord) -> KnowledgeNode {
    let mut restored = existing.clone();
    if let Ok(kind) = version.kind.parse::<NodeKind>() {
        restored.kind = kind;
    }
    restored.namespace = version.namespace.clone();
    restored.content = version.content.clone();
    restored.title = version.title.clone();
    restored.source = version.source.clone();
    restored.tags = version.tags.clone();
    restored.importance = version.importance;
    restored.metadata = version.metadata.clone();
    restored
}

fn trim_content_preview(raw: &str, max_chars: usize) -> String {
    let mut preview = String::new();
    for ch in raw.chars() {
        if preview.chars().count() >= max_chars {
            break;
        }
        preview.push(ch);
    }
    preview
}

fn text_line_items(raw: &str) -> Vec<String> {
    raw.lines()
        .map(|line| line.trim_end().to_string())
        .filter(|line| !line.is_empty())
        .collect()
}

fn trim_preview_line(raw: &str, max_chars: usize) -> String {
    let mut output = String::new();
    for ch in raw.chars() {
        if output.chars().count() >= max_chars {
            break;
        }
        output.push(ch);
    }
    output
}

fn display_optional_text(value: Option<&str>) -> String {
    value
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(ToString::to_string)
        .unwrap_or_else(|| "<empty>".to_string())
}

fn display_tags(tags: &[String]) -> String {
    if tags.is_empty() {
        "<none>".to_string()
    } else {
        tags.join(", ")
    }
}

fn normalize_string_list_for_compare(raw: &[String]) -> Vec<String> {
    let mut normalized = raw
        .iter()
        .map(|item| item.trim().to_ascii_lowercase())
        .filter(|item| !item.is_empty())
        .collect::<Vec<_>>();
    normalized.sort();
    normalized.dedup();
    normalized
}

fn metadata_keys_for_compare(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
) -> Vec<String> {
    let mut keys = metadata
        .keys()
        .filter(|key| {
            !key.eq_ignore_ascii_case(NODE_VERSIONS_METADATA_KEY)
                && !key.eq_ignore_ascii_case(TEMPLATE_VERSIONS_METADATA_KEY)
        })
        .map(|key| key.to_string())
        .collect::<Vec<_>>();
    keys.sort();
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_metadata_strips_recursive_version_keys() {
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            NODE_VERSIONS_METADATA_KEY.to_string(),
            serde_json::json!([{"version_id": "a"}]),
        );
        metadata.insert(
            TEMPLATE_VERSIONS_METADATA_KEY.to_string(),
            serde_json::json!([{"version_id": "t"}]),
        );
        metadata.insert("priority".to_string(), serde_json::json!("high"));

        let filtered = node_metadata_for_version_snapshot(&metadata);
        assert!(!filtered.contains_key(NODE_VERSIONS_METADATA_KEY));
        assert!(!filtered.contains_key(TEMPLATE_VERSIONS_METADATA_KEY));
        assert_eq!(
            filtered.get("priority").and_then(serde_json::Value::as_str),
            Some("high")
        );
    }

    #[test]
    fn node_version_push_enforces_cap() {
        let mut versions = Vec::new();
        for _ in 0..(NODE_VERSION_MAX_ENTRIES + 5) {
            versions.push(NodeVersionRecord {
                version_id: uuid::Uuid::now_v7().to_string(),
                captured_at: Utc::now().to_rfc3339(),
                kind: "fact".to_string(),
                namespace: "default".to_string(),
                title: None,
                content: "content".to_string(),
                source: None,
                tags: Vec::new(),
                importance: 0.5,
                metadata: std::collections::HashMap::new(),
            });
        }

        push_node_version_snapshot(
            &mut versions,
            NodeVersionRecord {
                version_id: uuid::Uuid::now_v7().to_string(),
                captured_at: Utc::now().to_rfc3339(),
                kind: "fact".to_string(),
                namespace: "default".to_string(),
                title: None,
                content: "latest".to_string(),
                source: None,
                tags: Vec::new(),
                importance: 0.5,
                metadata: std::collections::HashMap::new(),
            },
        );

        assert_eq!(versions.len(), NODE_VERSION_MAX_ENTRIES);
        assert_eq!(
            versions.last().map(|item| item.content.as_str()),
            Some("latest")
        );
    }
}
