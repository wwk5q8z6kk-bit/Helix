//! Structured data processing: JSON and YAML files.
//!
//! Parses JSON natively with serde_json. YAML files are read as plain text
//! with annotation (serde_yaml not yet in deps). Extracts schema information
//! including top-level keys, nesting depth, and total key count.

use async_trait::async_trait;
use hx_core::{HxError, KnowledgeNode, MvResult};

use super::{check_file_size, ModalityProcessor, ModalityStatus, ProcessingResult};

/// Structured data processor for JSON and YAML files.
pub struct StructuredDataProcessor;

impl StructuredDataProcessor {
    pub fn new() -> Self {
        Self
    }
}

/// Compute the maximum nesting depth of a JSON value.
fn nesting_depth(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(map) => {
            1 + map.values().map(nesting_depth).max().unwrap_or(0)
        }
        serde_json::Value::Array(arr) => {
            1 + arr.iter().map(nesting_depth).max().unwrap_or(0)
        }
        _ => 0,
    }
}

/// Count total keys in a JSON value (recursive).
fn total_keys(value: &serde_json::Value) -> usize {
    match value {
        serde_json::Value::Object(map) => {
            map.len() + map.values().map(total_keys).sum::<usize>()
        }
        serde_json::Value::Array(arr) => arr.iter().map(total_keys).sum(),
        _ => 0,
    }
}

/// Flatten a JSON value to searchable text with key paths.
fn flatten_to_text(value: &serde_json::Value, prefix: &str, lines: &mut Vec<String>) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                flatten_to_text(val, &path, lines);
            }
        }
        serde_json::Value::Array(arr) => {
            for (i, val) in arr.iter().enumerate() {
                let path = format!("{prefix}[{i}]");
                flatten_to_text(val, &path, lines);
            }
        }
        serde_json::Value::String(s) => {
            lines.push(format!("{prefix}: {s}"));
        }
        serde_json::Value::Number(n) => {
            lines.push(format!("{prefix}: {n}"));
        }
        serde_json::Value::Bool(b) => {
            lines.push(format!("{prefix}: {b}"));
        }
        serde_json::Value::Null => {
            lines.push(format!("{prefix}: null"));
        }
    }
}

/// Extract top-level keys from a JSON value.
fn top_level_keys(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::Object(map) => map.keys().cloned().collect(),
        _ => Vec::new(),
    }
}

/// Check if content looks like YAML (simple heuristic).
fn looks_like_yaml(content: &str) -> bool {
    let trimmed = content.trim();
    // YAML typically starts with "---" or "key: value" patterns
    trimmed.starts_with("---")
        || trimmed
            .lines()
            .take(5)
            .any(|line| {
                let l = line.trim();
                !l.is_empty()
                    && !l.starts_with('{')
                    && !l.starts_with('[')
                    && l.contains(": ")
            })
}

#[async_trait]
impl ModalityProcessor for StructuredDataProcessor {
    fn name(&self) -> &'static str {
        "structured_data"
    }

    fn handles(&self) -> &[&str] {
        &["application/json", "text/yaml", "application/x-yaml"]
    }

    fn status(&self) -> ModalityStatus {
        ModalityStatus::new(self.name(), true, self.handles())
            .with_detail("json_support", serde_json::json!(true))
            .with_detail("yaml_support", serde_json::json!("read-only (plain text)"))
            .with_note("Full YAML parsing requires serde_yaml crate (not yet in deps)")
    }

    async fn process(&self, file_path: &str, _node: &KnowledgeNode) -> MvResult<ProcessingResult> {
        tracing::info!(file_path, "Processing structured data file");

        check_file_size(file_path).map_err(HxError::Storage)?;

        let content = std::fs::read_to_string(file_path)
            .map_err(|e| HxError::Storage(format!("failed to read file: {e}")))?;

        if content.trim().is_empty() {
            let mut result = ProcessingResult::new(String::new())
                .with_tag("structured".to_string());

            result
                .metadata
                .insert("format".into(), serde_json::json!("empty"));

            return Ok(result);
        }

        // Try parsing as JSON first
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&content) {
            let keys = top_level_keys(&value);
            let depth = nesting_depth(&value);
            let total = total_keys(&value);

            let mut text_lines = Vec::new();
            flatten_to_text(&value, "", &mut text_lines);
            let text = text_lines.join("\n");

            let summary = format!(
                "JSON document: {} top-level keys, depth {}, {} total keys",
                keys.len(),
                depth,
                total
            );

            let mut result = ProcessingResult::new(text)
                .with_tag("structured".to_string())
                .with_tag("json".to_string())
                .with_summary(summary);

            result
                .metadata
                .insert("format".into(), serde_json::json!("json"));
            result
                .metadata
                .insert("top_level_keys".into(), serde_json::json!(keys));
            result
                .metadata
                .insert("nesting_depth".into(), serde_json::json!(depth));
            result
                .metadata
                .insert("total_keys".into(), serde_json::json!(total));

            return Ok(result);
        }

        // Not valid JSON — treat as YAML (plain text with annotation)
        let is_yaml = looks_like_yaml(&content)
            || file_path.ends_with(".yaml")
            || file_path.ends_with(".yml");

        let format_tag = if is_yaml { "yaml" } else { "text" };

        // Count top-level keys heuristically from YAML (lines starting at col 0 with "key:")
        let yaml_keys: Vec<String> = content
            .lines()
            .filter(|line| {
                !line.starts_with(' ')
                    && !line.starts_with('\t')
                    && !line.starts_with('#')
                    && !line.starts_with("---")
                    && line.contains(':')
            })
            .filter_map(|line| {
                line.split(':')
                    .next()
                    .map(|k| k.trim().to_string())
                    .filter(|k| !k.is_empty())
            })
            .collect();

        let annotated = format!("[Format: YAML]\n\n{content}");

        let summary = format!(
            "YAML document: ~{} top-level keys",
            yaml_keys.len()
        );

        let mut result = ProcessingResult::new(annotated)
            .with_tag("structured".to_string())
            .with_tag(format_tag.to_string())
            .with_summary(summary);

        result
            .metadata
            .insert("format".into(), serde_json::json!("yaml"));
        result
            .metadata
            .insert("top_level_keys".into(), serde_json::json!(yaml_keys));
        result
            .metadata
            .insert("nesting_depth".into(), serde_json::json!("unknown"));
        result
            .metadata
            .insert("total_keys".into(), serde_json::json!("unknown"));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::model::{KnowledgeNode, NodeKind};

    #[test]
    fn name_returns_expected() {
        let processor = StructuredDataProcessor::new();
        assert_eq!(processor.name(), "structured_data");
    }

    #[test]
    fn handles_returns_correct_types() {
        let processor = StructuredDataProcessor::new();
        let types = processor.handles();
        assert!(types.contains(&"application/json"));
        assert!(types.contains(&"text/yaml"));
        assert!(types.contains(&"application/x-yaml"));
        assert_eq!(types.len(), 3);
    }

    #[tokio::test]
    async fn process_extracts_text_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        std::fs::write(
            &path,
            r#"{"name": "Alice", "age": 30, "city": "NYC"}"#,
        )
        .unwrap();

        let processor = StructuredDataProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.contains("name: Alice"));
        assert!(result.text_content.contains("age: 30"));
        assert!(result.text_content.contains("city: NYC"));
    }

    #[tokio::test]
    async fn process_populates_metadata_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.json");
        std::fs::write(
            &path,
            r#"{"a": 1, "b": {"c": 2, "d": {"e": 3}}}"#,
        )
        .unwrap();

        let processor = StructuredDataProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert_eq!(result.metadata["format"], serde_json::json!("json"));
        assert_eq!(
            result.metadata["top_level_keys"],
            serde_json::json!(["a", "b"])
        );
        assert_eq!(result.metadata["nesting_depth"], serde_json::json!(3));
        // a=0 keys, b=2 keys (c, d), d=1 key (e) → total = 2 + 2 + 1 = 5? No:
        // top: a, b → 2 keys + recurse(b) → c, d → 2 keys + recurse(d) → e → 1 key = 2+2+1=5
        assert_eq!(result.metadata["total_keys"], serde_json::json!(5));
    }

    #[tokio::test]
    async fn process_adds_tags_json() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        std::fs::write(&path, r#"{"key": "value"}"#).unwrap();

        let processor = StructuredDataProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.suggested_tags.contains(&"structured".to_string()));
        assert!(result.suggested_tags.contains(&"json".to_string()));
    }

    #[tokio::test]
    async fn process_rejects_missing_file() {
        let processor = StructuredDataProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process("/nonexistent/file.json", &node).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn process_handles_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.json");
        std::fs::write(&path, "").unwrap();

        let processor = StructuredDataProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.is_empty());
        assert_eq!(result.metadata["format"], serde_json::json!("empty"));
    }

    #[tokio::test]
    async fn process_handles_yaml_content() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.yaml");
        std::fs::write(
            &path,
            "name: myapp\nversion: 1.0\ndatabase:\n  host: localhost\n  port: 5432\n",
        )
        .unwrap();

        let processor = StructuredDataProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.contains("[Format: YAML]"));
        assert!(result.text_content.contains("name: myapp"));
        assert_eq!(result.metadata["format"], serde_json::json!("yaml"));
        assert!(result.suggested_tags.contains(&"yaml".to_string()));
    }

    #[tokio::test]
    async fn process_handles_malformed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "{not valid json at all!!!}").unwrap();

        let processor = StructuredDataProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        // Should not panic — falls through to YAML path
        let result = processor.process(path.to_str().unwrap(), &node).await;
        assert!(result.is_ok());
    }

    #[test]
    fn status_reflects_availability() {
        let processor = StructuredDataProcessor::new();
        let status = processor.status();
        assert_eq!(status.name, "structured_data");
        assert!(status.available);
        assert_eq!(status.details["json_support"], serde_json::json!(true));
    }

    #[test]
    fn nesting_depth_computes_correctly() {
        let flat: serde_json::Value = serde_json::json!({"a": 1, "b": 2});
        assert_eq!(nesting_depth(&flat), 1);

        let nested: serde_json::Value = serde_json::json!({"a": {"b": {"c": 1}}});
        assert_eq!(nesting_depth(&nested), 3);

        let scalar: serde_json::Value = serde_json::json!(42);
        assert_eq!(nesting_depth(&scalar), 0);
    }

    #[test]
    fn total_keys_counts_correctly() {
        let value: serde_json::Value = serde_json::json!({"a": 1, "b": {"c": 2}});
        // top-level: a, b → 2 keys + recurse(b) → c → 1 key = 3
        assert_eq!(total_keys(&value), 3);
    }
}
