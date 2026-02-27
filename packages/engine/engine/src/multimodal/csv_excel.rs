//! Tabular data processing: CSV and Excel files.
//!
//! Parses CSV files natively (basic splitting with quote awareness).
//! Excel support is noted as unavailable until `calamine` is added to deps.

use async_trait::async_trait;
use hx_core::{HxError, KnowledgeNode, MvResult};

use super::{check_file_size, ModalityProcessor, ModalityStatus, ProcessingResult};

/// Maximum number of rows to include in the extracted text.
const MAX_PREVIEW_ROWS: usize = 50;

/// Tabular data processor for CSV and Excel files.
pub struct TabularProcessor;

impl TabularProcessor {
    pub fn new() -> Self {
        Self
    }
}

/// Parse a CSV string into rows of fields with basic quote handling.
fn parse_csv(content: &str) -> Vec<Vec<String>> {
    let mut rows = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }
        rows.push(parse_csv_line(line));
    }

    rows
}

/// Parse a single CSV line with basic quoting support.
fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' if !in_quotes && current.is_empty() => {
                in_quotes = true;
            }
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    // Escaped quote
                    current.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(current.trim().to_string());
                current = String::new();
            }
            _ => {
                current.push(ch);
            }
        }
    }
    fields.push(current.trim().to_string());
    fields
}

/// Format parsed CSV rows into human-readable text.
fn format_csv_as_text(rows: &[Vec<String>], headers: &[String]) -> String {
    let mut lines = Vec::new();

    if !headers.is_empty() {
        lines.push(format!("Columns: {}", headers.join(", ")));
        lines.push(String::new());
    }

    let data_rows = if !headers.is_empty() && !rows.is_empty() {
        &rows[1..]
    } else {
        rows
    };

    for (i, row) in data_rows.iter().take(MAX_PREVIEW_ROWS).enumerate() {
        if headers.is_empty() {
            lines.push(format!("Row {}: {}", i + 1, row.join(", ")));
        } else {
            let pairs: Vec<String> = headers
                .iter()
                .zip(row.iter())
                .map(|(h, v)| format!("{h}: {v}"))
                .collect();
            lines.push(format!("Row {}: {}", i + 1, pairs.join(", ")));
        }
    }

    if data_rows.len() > MAX_PREVIEW_ROWS {
        lines.push(format!(
            "... and {} more rows",
            data_rows.len() - MAX_PREVIEW_ROWS
        ));
    }

    lines.join("\n")
}

#[async_trait]
impl ModalityProcessor for TabularProcessor {
    fn name(&self) -> &'static str {
        "tabular"
    }

    fn handles(&self) -> &[&str] {
        &[
            "text/csv",
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]
    }

    fn status(&self) -> ModalityStatus {
        ModalityStatus::new(self.name(), true, self.handles())
            .with_detail("csv_support", serde_json::json!(true))
            .with_detail("excel_support", serde_json::json!(false))
            .with_note("Excel support requires calamine crate (not yet in deps); CSV is fully supported")
    }

    async fn process(&self, file_path: &str, _node: &KnowledgeNode) -> MvResult<ProcessingResult> {
        tracing::info!(file_path, "Processing tabular file");

        check_file_size(file_path).map_err(HxError::Storage)?;

        let is_csv = file_path.ends_with(".csv") || file_path.ends_with(".tsv");
        let is_excel = file_path.ends_with(".xls") || file_path.ends_with(".xlsx");

        if is_excel {
            // Excel files require calamine — return placeholder
            let file_size = std::fs::metadata(file_path)
                .map(|m| m.len())
                .unwrap_or(0);

            let mut result =
                ProcessingResult::new(format!("[Excel file: {file_path} - install calamine crate for extraction]"))
                    .with_tag("tabular".to_string())
                    .with_tag("data".to_string());

            result
                .metadata
                .insert("format".into(), serde_json::json!("excel"));
            result
                .metadata
                .insert("file_size".into(), serde_json::json!(file_size));

            return Ok(result);
        }

        // CSV processing
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| HxError::Storage(format!("failed to read CSV file: {e}")))?;

        if content.trim().is_empty() {
            let mut result = ProcessingResult::new(String::new())
                .with_tag("tabular".to_string())
                .with_tag("data".to_string());

            result
                .metadata
                .insert("format".into(), serde_json::json!("csv"));
            result
                .metadata
                .insert("row_count".into(), serde_json::json!(0));
            result
                .metadata
                .insert("column_count".into(), serde_json::json!(0));
            result.metadata.insert(
                "column_names".into(),
                serde_json::json!(Vec::<String>::new()),
            );

            return Ok(result);
        }

        let rows = parse_csv(&content);
        let headers = rows.first().cloned().unwrap_or_default();
        let row_count = if rows.len() > 1 { rows.len() - 1 } else { rows.len() };
        let column_count = headers.len();

        let text = format_csv_as_text(&rows, &headers);

        let schema_summary = if !headers.is_empty() {
            format!(
                "CSV with {} columns ({}) and {} data rows",
                column_count,
                headers.join(", "),
                row_count
            )
        } else {
            format!("CSV with {} rows", rows.len())
        };

        let mut result = ProcessingResult::new(text)
            .with_tag("tabular".to_string())
            .with_tag("data".to_string())
            .with_summary(schema_summary);

        result
            .metadata
            .insert("format".into(), serde_json::json!(if is_csv { "csv" } else { "unknown" }));
        result
            .metadata
            .insert("row_count".into(), serde_json::json!(row_count));
        result
            .metadata
            .insert("column_count".into(), serde_json::json!(column_count));
        result
            .metadata
            .insert("column_names".into(), serde_json::json!(headers));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::model::{KnowledgeNode, NodeKind};

    #[test]
    fn name_returns_expected() {
        let processor = TabularProcessor::new();
        assert_eq!(processor.name(), "tabular");
    }

    #[test]
    fn handles_returns_correct_types() {
        let processor = TabularProcessor::new();
        let types = processor.handles();
        assert!(types.contains(&"text/csv"));
        assert!(types.contains(&"application/vnd.ms-excel"));
        assert!(types.contains(
            &"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ));
        assert_eq!(types.len(), 3);
    }

    #[tokio::test]
    async fn process_extracts_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");
        std::fs::write(&path, "name,age,city\nAlice,30,NYC\nBob,25,LA\n").unwrap();

        let processor = TabularProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.contains("name"));
        assert!(result.text_content.contains("Alice"));
        assert!(result.text_content.contains("Bob"));
    }

    #[tokio::test]
    async fn process_populates_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.csv");
        std::fs::write(&path, "x,y\n1,2\n3,4\n5,6\n").unwrap();

        let processor = TabularProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert_eq!(result.metadata["format"], serde_json::json!("csv"));
        assert_eq!(result.metadata["row_count"], serde_json::json!(3));
        assert_eq!(result.metadata["column_count"], serde_json::json!(2));
        assert_eq!(
            result.metadata["column_names"],
            serde_json::json!(["x", "y"])
        );
    }

    #[tokio::test]
    async fn process_adds_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");
        std::fs::write(&path, "a,b\n1,2\n").unwrap();

        let processor = TabularProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.suggested_tags.contains(&"tabular".to_string()));
        assert!(result.suggested_tags.contains(&"data".to_string()));
    }

    #[tokio::test]
    async fn process_rejects_missing_file() {
        let processor = TabularProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process("/nonexistent/file.csv", &node).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn process_handles_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.csv");
        std::fs::write(&path, "").unwrap();

        let processor = TabularProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.is_empty());
        assert_eq!(result.metadata["row_count"], serde_json::json!(0));
    }

    #[tokio::test]
    async fn process_handles_malformed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.csv");
        std::fs::write(&path, "a,b,c\n1,2\n3,4,5,6,7\n").unwrap();

        let processor = TabularProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        // Ragged rows should not panic
        let result = processor.process(path.to_str().unwrap(), &node).await;
        assert!(result.is_ok());
    }

    #[test]
    fn status_reflects_availability() {
        let processor = TabularProcessor::new();
        let status = processor.status();
        assert_eq!(status.name, "tabular");
        assert!(status.available);
        assert_eq!(
            status.details["csv_support"],
            serde_json::json!(true)
        );
        assert_eq!(
            status.details["excel_support"],
            serde_json::json!(false)
        );
    }

    #[test]
    fn parse_csv_line_handles_quotes() {
        let line = r#"hello,"world, nice",test"#;
        let fields = parse_csv_line(line);
        assert_eq!(fields, vec!["hello", "world, nice", "test"]);
    }

    #[tokio::test]
    async fn process_excel_returns_placeholder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.xlsx");
        std::fs::write(&path, b"PK fake xlsx").unwrap();

        let processor = TabularProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.contains("calamine"));
        assert_eq!(result.metadata["format"], serde_json::json!("excel"));
    }
}
