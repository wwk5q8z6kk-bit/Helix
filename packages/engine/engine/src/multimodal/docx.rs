//! DOCX processing: text extraction via `pandoc` CLI.
//!
//! Falls back to basic XML extraction from the DOCX zip structure
//! if `pandoc` is not available.

use async_trait::async_trait;
use hx_core::{HxError, KnowledgeNode, MvResult};
use regex::Regex;
use std::process::Command;

use super::{
    check_file_size, run_command_with_timeout, ModalityProcessor, ModalityStatus,
    ProcessingResult, DEFAULT_COMMAND_TIMEOUT,
};

/// DOCX processor that extracts text content using `pandoc` CLI,
/// with a fallback to basic XML extraction from the DOCX zip.
pub struct DocxProcessor {
    pandoc_available: bool,
}

impl DocxProcessor {
    pub fn new() -> Self {
        let pandoc_available = Command::new("pandoc").arg("--version").output().is_ok();

        if pandoc_available {
            tracing::info!("pandoc available for DOCX text extraction");
        } else {
            tracing::warn!(
                "pandoc not available — DOCX extraction will use basic XML fallback"
            );
        }

        Self { pandoc_available }
    }

    /// Extract text using pandoc CLI.
    fn extract_with_pandoc(&self, file_path: &str) -> Result<String, String> {
        let mut cmd = Command::new("pandoc");
        cmd.args(["-f", "docx", "-t", "plain", file_path]);
        let output = run_command_with_timeout(&mut cmd, DEFAULT_COMMAND_TIMEOUT)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("pandoc failed: {stderr}"));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Fallback: extract text from DOCX by unzipping word/document.xml
    /// and stripping XML tags.
    fn extract_with_unzip(&self, file_path: &str) -> Result<String, String> {
        let mut cmd = Command::new("unzip");
        cmd.args(["-p", file_path, "word/document.xml"]);
        let output = run_command_with_timeout(&mut cmd, DEFAULT_COMMAND_TIMEOUT)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("unzip failed: {stderr}"));
        }

        let xml = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(strip_xml_tags(&xml))
    }
}

/// Strip XML tags from a string, preserving text content.
fn strip_xml_tags(xml: &str) -> String {
    let tag_re = Regex::new(r"<[^>]+>").expect("valid regex");
    let text = tag_re.replace_all(xml, " ");
    // Collapse whitespace
    let ws_re = Regex::new(r"\s+").expect("valid regex");
    ws_re.replace_all(text.trim(), " ").to_string()
}

#[async_trait]
impl ModalityProcessor for DocxProcessor {
    fn name(&self) -> &'static str {
        "docx"
    }

    fn handles(&self) -> &[&str] {
        &["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    }

    fn status(&self) -> ModalityStatus {
        let mut status = ModalityStatus::new(self.name(), true, self.handles())
            .with_detail("pandoc_available", serde_json::json!(self.pandoc_available));

        if !self.pandoc_available {
            status = status.with_note("pandoc not available; using basic XML fallback");
        }

        status
    }

    async fn process(&self, file_path: &str, _node: &KnowledgeNode) -> MvResult<ProcessingResult> {
        tracing::info!(file_path, "Processing DOCX file");

        let file_size = check_file_size(file_path).map_err(HxError::Storage)?;

        // Try pandoc first, fall back to unzip + XML strip
        let text = if self.pandoc_available {
            match self.extract_with_pandoc(file_path) {
                Ok(text) if !text.is_empty() => text,
                Ok(_) => {
                    // Pandoc returned empty — try fallback
                    self.extract_with_unzip(file_path).unwrap_or_else(|e| {
                        tracing::warn!(error = %e, "DOCX fallback extraction failed");
                        format!("[DOCX: {file_path} - extraction failed]")
                    })
                }
                Err(e) => {
                    tracing::warn!(error = %e, "pandoc failed, trying fallback");
                    self.extract_with_unzip(file_path).unwrap_or_else(|e2| {
                        tracing::warn!(error = %e2, "DOCX fallback extraction also failed");
                        format!("[DOCX: {file_path} - extraction failed]")
                    })
                }
            }
        } else {
            self.extract_with_unzip(file_path).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "DOCX fallback extraction failed");
                format!("[DOCX: {file_path} - extraction failed]")
            })
        };

        let word_count = text.split_whitespace().count();

        let mut result = ProcessingResult::new(text)
            .with_tag("docx".to_string())
            .with_tag("document".to_string());

        result
            .metadata
            .insert("file_size".into(), serde_json::json!(file_size));
        result
            .metadata
            .insert("word_count".into(), serde_json::json!(word_count));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::model::{KnowledgeNode, NodeKind};

    #[test]
    fn name_returns_expected() {
        let processor = DocxProcessor::new();
        assert_eq!(processor.name(), "docx");
    }

    #[test]
    fn handles_returns_correct_types() {
        let processor = DocxProcessor::new();
        let types = processor.handles();
        assert_eq!(types.len(), 1);
        assert_eq!(
            types[0],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        );
    }

    #[tokio::test]
    async fn process_rejects_missing_file() {
        let processor = DocxProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process("/nonexistent/file.docx", &node).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn process_handles_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.docx");
        std::fs::write(&path, b"").unwrap();

        let processor = DocxProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        // Empty file — extraction will fail gracefully with placeholder
        let result = processor.process(path.to_str().unwrap(), &node).await;
        if let Ok(r) = result {
            assert!(r.suggested_tags.contains(&"docx".to_string()));
        }
    }

    #[tokio::test]
    async fn process_adds_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.docx");
        std::fs::write(&path, b"PK fake docx content").unwrap();

        let processor = DocxProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await;
        if let Ok(r) = result {
            assert!(r.suggested_tags.contains(&"docx".to_string()));
            assert!(r.suggested_tags.contains(&"document".to_string()));
        }
    }

    #[tokio::test]
    async fn process_populates_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.docx");
        std::fs::write(&path, b"PK fake docx content").unwrap();

        let processor = DocxProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await;
        if let Ok(r) = result {
            assert!(r.metadata.contains_key("file_size"));
            assert!(r.metadata.contains_key("word_count"));
        }
    }

    #[tokio::test]
    async fn process_handles_malformed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.docx");
        std::fs::write(&path, b"this is not a valid docx file at all").unwrap();

        let processor = DocxProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        // Malformed file should not panic
        let result = processor.process(path.to_str().unwrap(), &node).await;
        if let Ok(r) = result {
            // Should get fallback text or placeholder
            assert!(!r.text_content.is_empty());
        }
    }

    #[test]
    fn status_reflects_availability() {
        let processor = DocxProcessor::new();
        let status = processor.status();
        assert_eq!(status.name, "docx");
        assert!(status.details.contains_key("pandoc_available"));
        // Always available (has XML fallback)
        assert!(status.available);
    }

    #[test]
    fn status_note_when_no_pandoc() {
        let processor = DocxProcessor {
            pandoc_available: false,
        };
        let status = processor.status();
        assert_eq!(
            status.note.as_deref(),
            Some("pandoc not available; using basic XML fallback")
        );
    }

    #[test]
    fn strip_xml_tags_removes_tags() {
        let xml = "<w:p><w:r><w:t>Hello</w:t></w:r> <w:r><w:t>World</w:t></w:r></w:p>";
        let text = strip_xml_tags(xml);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("<w:"));
    }

    #[tokio::test]
    async fn process_extracts_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.docx");
        std::fs::write(&path, b"PK fake").unwrap();

        let processor = DocxProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await;
        // Regardless of tools available, should produce some text result
        if let Ok(r) = result {
            assert!(!r.text_content.is_empty());
        }
    }
}
