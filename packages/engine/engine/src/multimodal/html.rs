//! HTML processing: tag stripping, title and link extraction.
//!
//! Reads HTML files, strips tags using regex, extracts metadata
//! such as the `<title>` and link count.

use async_trait::async_trait;
use hx_core::{HxError, KnowledgeNode, MvResult};
use regex::Regex;

use super::{check_file_size, ModalityProcessor, ModalityStatus, ProcessingResult};

/// HTML processor that strips tags and extracts metadata.
pub struct HtmlProcessor;

impl HtmlProcessor {
    pub fn new() -> Self {
        Self
    }
}

/// Extract the content of the first `<title>` tag.
fn extract_title(html: &str) -> Option<String> {
    let re = Regex::new(r"(?i)<title[^>]*>([\s\S]*?)</title>").expect("valid regex");
    re.captures(html)
        .and_then(|caps| caps.get(1))
        .map(|m| m.as_str().trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Count the number of `<a` tags in the HTML.
fn count_links(html: &str) -> usize {
    let re = Regex::new(r"(?i)<a[\s>]").expect("valid regex");
    re.find_iter(html).count()
}

/// Strip all HTML tags, returning plain text.
fn strip_html_tags(html: &str) -> String {
    let tag_re = Regex::new(r"<[^>]+>").expect("valid regex");
    let text = tag_re.replace_all(html, " ");
    // Decode common HTML entities
    let text = text
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    // Collapse whitespace
    let ws_re = Regex::new(r"\s+").expect("valid regex");
    ws_re.replace_all(text.trim(), " ").to_string()
}

#[async_trait]
impl ModalityProcessor for HtmlProcessor {
    fn name(&self) -> &'static str {
        "html"
    }

    fn handles(&self) -> &[&str] {
        &["text/html", "application/xhtml+xml"]
    }

    fn status(&self) -> ModalityStatus {
        ModalityStatus::new(self.name(), true, self.handles())
    }

    async fn process(&self, file_path: &str, _node: &KnowledgeNode) -> MvResult<ProcessingResult> {
        tracing::info!(file_path, "Processing HTML file");

        check_file_size(file_path).map_err(HxError::Storage)?;

        let html = std::fs::read_to_string(file_path)
            .map_err(|e| HxError::Storage(format!("failed to read HTML file: {e}")))?;

        let title = extract_title(&html);
        let link_count = count_links(&html);
        let text = strip_html_tags(&html);
        let char_count = text.len();

        let summary = title
            .as_ref()
            .map(|t| format!("HTML document: {t}"))
            .or_else(|| {
                let preview: String = text.chars().take(200).collect();
                if preview.is_empty() {
                    None
                } else {
                    Some(format!("HTML document: {preview}..."))
                }
            });

        let mut result = ProcessingResult::new(text)
            .with_tag("html".to_string())
            .with_tag("web".to_string());

        if let Some(s) = summary {
            result = result.with_summary(s);
        }

        if let Some(ref t) = title {
            result
                .metadata
                .insert("title".into(), serde_json::json!(t));
        }
        result
            .metadata
            .insert("link_count".into(), serde_json::json!(link_count));
        result
            .metadata
            .insert("char_count".into(), serde_json::json!(char_count));

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::model::{KnowledgeNode, NodeKind};

    #[test]
    fn name_returns_expected() {
        let processor = HtmlProcessor::new();
        assert_eq!(processor.name(), "html");
    }

    #[test]
    fn handles_returns_correct_types() {
        let processor = HtmlProcessor::new();
        let types = processor.handles();
        assert!(types.contains(&"text/html"));
        assert!(types.contains(&"application/xhtml+xml"));
        assert_eq!(types.len(), 2);
    }

    #[tokio::test]
    async fn process_extracts_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.html");
        std::fs::write(
            &path,
            "<html><head><title>Test Page</title></head><body><p>Hello World</p></body></html>",
        )
        .unwrap();

        let processor = HtmlProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.contains("Hello World"));
        assert!(!result.text_content.contains("<p>"));
    }

    #[tokio::test]
    async fn process_populates_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.html");
        std::fs::write(
            &path,
            "<html><head><title>My Title</title></head><body><a href=\"#\">Link1</a><a href=\"#\">Link2</a></body></html>",
        )
        .unwrap();

        let processor = HtmlProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert_eq!(result.metadata["title"], serde_json::json!("My Title"));
        assert_eq!(result.metadata["link_count"], serde_json::json!(2));
        assert!(result.metadata.contains_key("char_count"));
    }

    #[tokio::test]
    async fn process_adds_tags() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.html");
        std::fs::write(&path, "<html><body>content</body></html>").unwrap();

        let processor = HtmlProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.suggested_tags.contains(&"html".to_string()));
        assert!(result.suggested_tags.contains(&"web".to_string()));
    }

    #[tokio::test]
    async fn process_rejects_missing_file() {
        let processor = HtmlProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process("/nonexistent/file.html", &node).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn process_handles_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.html");
        std::fs::write(&path, "").unwrap();

        let processor = HtmlProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await.unwrap();

        assert!(result.text_content.is_empty());
        assert!(result.suggested_tags.contains(&"html".to_string()));
    }

    #[tokio::test]
    async fn process_handles_malformed() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.html");
        std::fs::write(&path, "<<<not valid>>> html <broken").unwrap();

        let processor = HtmlProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process(path.to_str().unwrap(), &node).await;
        // Should not panic — graceful handling
        assert!(result.is_ok());
    }

    #[test]
    fn status_reflects_availability() {
        let processor = HtmlProcessor::new();
        let status = processor.status();
        assert_eq!(status.name, "html");
        assert!(status.available);
        assert!(status.supported_types.contains(&"text/html".to_string()));
    }

    #[test]
    fn extract_title_finds_title() {
        assert_eq!(
            extract_title("<title>Hello</title>"),
            Some("Hello".to_string())
        );
        assert_eq!(
            extract_title("<TITLE>Upper</TITLE>"),
            Some("Upper".to_string())
        );
        assert_eq!(extract_title("<body>no title</body>"), None);
        assert_eq!(extract_title("<title></title>"), None);
    }

    #[test]
    fn count_links_counts_correctly() {
        assert_eq!(count_links("<a href=\"#\">one</a> <a>two</a>"), 2);
        assert_eq!(count_links("<p>no links</p>"), 0);
    }

    #[test]
    fn strip_html_tags_removes_all_tags() {
        let html = "<div><p>Hello</p> <span>World</span></div>";
        let text = strip_html_tags(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("<div>"));
        assert!(!text.contains("<p>"));
    }
}
