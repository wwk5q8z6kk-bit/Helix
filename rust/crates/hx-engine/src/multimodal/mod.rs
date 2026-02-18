//! Multi-modal processing pipeline.
//! Dispatches to modality-specific processors based on content type.

use async_trait::async_trait;
use hx_core::{KnowledgeNode, MvResult};
use serde::Serialize;
use std::collections::HashMap;
use std::process::{Command, Output};
use std::time::Duration;

/// Default timeout for external tool invocations (120 seconds).
pub const DEFAULT_COMMAND_TIMEOUT: Duration = Duration::from_secs(120);

/// Maximum file size we'll attempt to process (256 MB).
pub const MAX_FILE_SIZE: u64 = 256 * 1024 * 1024;

/// Run an external command with a timeout.
///
/// Spawns the command as a child process and waits up to `timeout` for it to
/// complete.  If the timeout expires the child is killed and an error returned.
pub fn run_command_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
) -> Result<Output, String> {
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn command: {e}"))?;

    let start = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = child
                    .stdout
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                let stderr = child
                    .stderr
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                return Ok(Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(format!(
                        "command timed out after {}s",
                        timeout.as_secs()
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(format!("error waiting for command: {e}")),
        }
    }
}

/// Check whether a file exceeds the processing size limit.
pub fn check_file_size(file_path: &str) -> Result<u64, String> {
    let meta = std::fs::metadata(file_path)
        .map_err(|e| format!("cannot read file metadata: {e}"))?;
    let size = meta.len();
    if size > MAX_FILE_SIZE {
        return Err(format!(
            "file too large ({:.1} MB, limit {:.0} MB)",
            size as f64 / (1024.0 * 1024.0),
            MAX_FILE_SIZE as f64 / (1024.0 * 1024.0)
        ));
    }
    Ok(size)
}

/// A processor for a specific modality (audio, image, PDF, etc.)
#[async_trait]
pub trait ModalityProcessor: Send + Sync {
    /// Human-readable name for this processor.
    fn name(&self) -> &'static str;

    /// The content type this processor handles (e.g. "audio/wav", "image/png", "application/pdf")
    fn handles(&self) -> &[&str];

    /// Process a file and return extracted text content and metadata
    async fn process(&self, file_path: &str, node: &KnowledgeNode) -> MvResult<ProcessingResult>;

    /// Status/introspection for diagnostics endpoints.
    fn status(&self) -> ModalityStatus {
        ModalityStatus::new(self.name(), true, self.handles())
    }
}

/// Result of processing a multi-modal input
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    /// Extracted text content (transcription, OCR, etc.)
    pub text_content: String,
    /// Additional tags to add to the node
    pub suggested_tags: Vec<String>,
    /// Summary/description of the content
    pub summary: Option<String>,
    /// Processing metadata (e.g. duration, dimensions, page count)
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModalityStatus {
    pub name: String,
    pub available: bool,
    pub supported_types: Vec<String>,
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub details: HashMap<String, serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

impl ModalityStatus {
    pub fn new(name: &str, available: bool, supported: &[&str]) -> Self {
        Self {
            name: name.to_string(),
            available,
            supported_types: supported.iter().map(|s| s.to_string()).collect(),
            details: HashMap::new(),
            note: None,
        }
    }

    pub fn with_detail(mut self, key: &str, value: serde_json::Value) -> Self {
        self.details.insert(key.to_string(), value);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

impl ProcessingResult {
    pub fn new(text_content: String) -> Self {
        Self {
            text_content,
            suggested_tags: Vec::new(),
            summary: None,
            metadata: std::collections::HashMap::new(),
        }
    }

    pub fn with_summary(mut self, summary: String) -> Self {
        self.summary = Some(summary);
        self
    }

    pub fn with_tag(mut self, tag: String) -> Self {
        self.suggested_tags.push(tag);
        self
    }
}

/// Pipeline that dispatches to registered modality processors.
pub struct MultiModalPipeline {
    processors: Vec<Box<dyn ModalityProcessor>>,
}

impl MultiModalPipeline {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn register(&mut self, processor: Box<dyn ModalityProcessor>) {
        self.processors.push(processor);
    }

    /// Find the first processor that handles this content type and process the file.
    pub async fn process(
        &self,
        content_type: &str,
        file_path: &str,
        node: &KnowledgeNode,
    ) -> MvResult<Option<ProcessingResult>> {
        for processor in &self.processors {
            if processor.handles().iter().any(|&ct| ct == content_type) {
                let result = processor.process(file_path, node).await?;
                return Ok(Some(result));
            }
        }
        Ok(None)
    }

    /// Check if any processor can handle this content type.
    pub fn can_process(&self, content_type: &str) -> bool {
        self.processors
            .iter()
            .any(|p| p.handles().iter().any(|&ct| ct == content_type))
    }

    /// Return all content types that are supported by registered processors.
    pub fn supported_types(&self) -> Vec<&str> {
        self.processors
            .iter()
            .flat_map(|p| p.handles().iter().copied())
            .collect()
    }

    /// Return status information for each registered processor.
    pub fn status(&self) -> Vec<ModalityStatus> {
        self.processors.iter().map(|p| p.status()).collect()
    }
}

pub mod audio;
pub mod code;
pub mod csv_excel;
pub mod docx;
pub mod epub;
pub mod html;
pub mod image;
pub mod pdf;
pub mod structured_data;

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::model::{KnowledgeNode, NodeKind};

    struct EchoProcessor;

    #[async_trait]
    impl ModalityProcessor for EchoProcessor {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn handles(&self) -> &[&str] {
            &["text/plain"]
        }
        async fn process(
            &self,
            file_path: &str,
            _node: &KnowledgeNode,
        ) -> MvResult<ProcessingResult> {
            Ok(ProcessingResult::new(format!("echo:{file_path}")))
        }
    }

    #[tokio::test]
    async fn pipeline_dispatches_to_matching_processor() {
        let mut pipeline = MultiModalPipeline::new();
        pipeline.register(Box::new(EchoProcessor));
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = pipeline
            .process("text/plain", "/tmp/test.txt", &node)
            .await
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().text_content, "echo:/tmp/test.txt");
    }

    #[tokio::test]
    async fn pipeline_returns_none_for_unknown_type() {
        let mut pipeline = MultiModalPipeline::new();
        pipeline.register(Box::new(EchoProcessor));
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = pipeline
            .process("application/octet-stream", "/tmp/test.bin", &node)
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn pipeline_can_process_registered_type() {
        let mut pipeline = MultiModalPipeline::new();
        pipeline.register(Box::new(EchoProcessor));
        assert!(pipeline.can_process("text/plain"));
        assert!(!pipeline.can_process("video/mp4"));
    }

    #[test]
    fn pipeline_supported_types_lists_all() {
        let mut pipeline = MultiModalPipeline::new();
        pipeline.register(Box::new(EchoProcessor));
        let types = pipeline.supported_types();
        assert_eq!(types, vec!["text/plain"]);
    }

    #[test]
    fn pipeline_status_includes_all_processors() {
        let mut pipeline = MultiModalPipeline::new();
        pipeline.register(Box::new(EchoProcessor));
        let statuses = pipeline.status();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].name, "echo");
    }

    #[test]
    fn processing_result_builder_works() {
        let result = ProcessingResult::new("hello".into())
            .with_tag("a".into())
            .with_tag("b".into())
            .with_summary("sum".into());
        assert_eq!(result.text_content, "hello");
        assert_eq!(result.suggested_tags, vec!["a", "b"]);
        assert_eq!(result.summary.as_deref(), Some("sum"));
    }

    #[test]
    fn run_command_with_timeout_succeeds() {
        let mut cmd = Command::new("echo");
        cmd.arg("hello");
        let output = run_command_with_timeout(&mut cmd, Duration::from_secs(5)).unwrap();
        assert!(output.status.success());
        assert!(String::from_utf8_lossy(&output.stdout).contains("hello"));
    }

    #[test]
    fn run_command_with_timeout_detects_failure() {
        let mut cmd = Command::new("false");
        let output = run_command_with_timeout(&mut cmd, Duration::from_secs(5)).unwrap();
        assert!(!output.status.success());
    }

    #[test]
    fn run_command_with_timeout_kills_slow_process() {
        let mut cmd = Command::new("sleep");
        cmd.arg("60");
        let result = run_command_with_timeout(&mut cmd, Duration::from_millis(200));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("timed out"));
    }

    #[test]
    fn run_command_with_timeout_reports_spawn_failure() {
        let mut cmd = Command::new("nonexistent-binary-xyz");
        let result = run_command_with_timeout(&mut cmd, Duration::from_secs(1));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("failed to spawn"));
    }

    #[test]
    fn check_file_size_accepts_small_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("small.txt");
        std::fs::write(&path, "hello").unwrap();
        let size = check_file_size(path.to_str().unwrap()).unwrap();
        assert_eq!(size, 5);
    }

    #[test]
    fn check_file_size_rejects_missing_file() {
        let result = check_file_size("/nonexistent/path/to/file.bin");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("cannot read file metadata"));
    }

    #[test]
    fn modality_status_builder_works() {
        let status = ModalityStatus::new("test", true, &["a/b"])
            .with_detail("key", serde_json::json!("val"))
            .with_note("note");
        assert_eq!(status.name, "test");
        assert!(status.available);
        assert_eq!(status.supported_types, vec!["a/b"]);
        assert_eq!(
            status.details.get("key"),
            Some(&serde_json::json!("val"))
        );
        assert_eq!(status.note.as_deref(), Some("note"));
    }
}
