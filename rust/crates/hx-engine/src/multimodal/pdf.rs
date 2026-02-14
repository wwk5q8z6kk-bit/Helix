//! PDF processing: text extraction via `pdftotext` CLI (poppler-utils).
//!
//! Falls back to basic metadata extraction if `pdftotext` is not available.
//! Optionally uses `tesseract` for OCR on scanned PDFs.

use async_trait::async_trait;
use hx_core::{KnowledgeNode, HxError, MvResult};
use std::process::Command;

use super::{
    check_file_size, run_command_with_timeout, ModalityProcessor, ModalityStatus,
    ProcessingResult, DEFAULT_COMMAND_TIMEOUT,
};

/// PDF processor that extracts text content using `pdftotext` CLI.
pub struct PdfProcessor {
    pdftotext_available: bool,
    tesseract_available: bool,
    ghostscript_available: bool,
}

impl PdfProcessor {
    pub fn new() -> Self {
        let pdftotext_available = Command::new("pdftotext")
            .arg("-v")
            .output()
            .is_ok();

        let tesseract_available = Command::new("tesseract")
            .arg("--version")
            .output()
            .is_ok();

        let ghostscript_available = Command::new("gs")
            .arg("--version")
            .output()
            .is_ok();

        if pdftotext_available {
            tracing::info!("pdftotext available for PDF text extraction");
        }
        if tesseract_available {
            tracing::info!("tesseract available for OCR fallback on scanned PDFs");
        }
        if tesseract_available && !ghostscript_available {
            tracing::warn!("tesseract available but ghostscript missing; OCR conversion may fail");
        }
        if !pdftotext_available && !tesseract_available {
            tracing::warn!(
                "No PDF extraction tools available (install poppler-utils for pdftotext)"
            );
        }

        Self {
            pdftotext_available,
            tesseract_available,
            ghostscript_available,
        }
    }

    /// Extract text using pdftotext CLI with timeout protection.
    fn extract_with_pdftotext(&self, file_path: &str) -> Result<String, String> {
        let mut cmd = Command::new("pdftotext");
        cmd.arg("-layout").arg(file_path).arg("-");
        let output = run_command_with_timeout(&mut cmd, DEFAULT_COMMAND_TIMEOUT)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("pdftotext failed: {stderr}"));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Get page count using pdfinfo CLI.
    fn get_page_count(&self, file_path: &str) -> Option<u32> {
        let output = Command::new("pdfinfo")
            .arg(file_path)
            .output()
            .ok()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines() {
            if let Some(count_str) = line.strip_prefix("Pages:") {
                return count_str.trim().parse().ok();
            }
        }
        None
    }

    /// Extract text from a scanned PDF using tesseract OCR.
    /// Converts PDF to images first, then runs OCR.
    fn extract_with_ocr(&self, file_path: &str) -> Result<String, String> {
        let temp_dir = std::env::temp_dir();
        let stem = uuid::Uuid::now_v7();
        let tiff_path = temp_dir.join(format!("hx_ocr_{stem}.tiff"));

        // Convert PDF to TIFF using ghostscript (commonly available)
        let mut gs_cmd = Command::new("gs");
        gs_cmd.args([
            "-dNOPAUSE",
            "-dBATCH",
            "-sDEVICE=tiffg4",
            "-r300",
            &format!("-sOutputFile={}", tiff_path.display()),
            file_path,
        ]);
        match run_command_with_timeout(&mut gs_cmd, DEFAULT_COMMAND_TIMEOUT) {
            Ok(output) if output.status.success() => {}
            _ => return Err("ghostscript not available or failed".to_string()),
        }

        // Run tesseract on the TIFF
        let mut tess_cmd = Command::new("tesseract");
        tess_cmd.arg(&tiff_path).arg("stdout");
        let output = run_command_with_timeout(&mut tess_cmd, DEFAULT_COMMAND_TIMEOUT)
            .map_err(|e| format!("tesseract failed: {e}"))?;

        let _ = std::fs::remove_file(&tiff_path);

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("tesseract OCR failed: {stderr}"));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }
}

#[async_trait]
impl ModalityProcessor for PdfProcessor {
    fn name(&self) -> &'static str {
        "pdf"
    }

    fn handles(&self) -> &[&str] {
        &["application/pdf"]
    }

    fn status(&self) -> ModalityStatus {
        let available = self.pdftotext_available || self.tesseract_available;
        let mut status = ModalityStatus::new(self.name(), available, self.handles())
            .with_detail("pdftotext_available", serde_json::json!(self.pdftotext_available))
            .with_detail("tesseract_available", serde_json::json!(self.tesseract_available))
            .with_detail("ghostscript_available", serde_json::json!(self.ghostscript_available));

        if !available {
            status = status.with_note("No PDF extraction backend available");
        } else if self.tesseract_available && !self.ghostscript_available {
            status = status.with_note("OCR available but ghostscript missing for PDF-to-image");
        }

        status
    }

    async fn process(&self, file_path: &str, _node: &KnowledgeNode) -> MvResult<ProcessingResult> {
        tracing::info!(file_path, "Processing PDF file");

        let file_size = check_file_size(file_path)
            .map_err(|e| HxError::Storage(e))?;

        let page_count = self.get_page_count(file_path);

        // Try pdftotext first
        let text = if self.pdftotext_available {
            match self.extract_with_pdftotext(file_path) {
                Ok(text) if !text.is_empty() => text,
                Ok(_) if self.tesseract_available && self.ghostscript_available => {
                    // Empty text from pdftotext likely means scanned PDF — try OCR
                    tracing::info!("PDF appears to be scanned, attempting OCR");
                    self.extract_with_ocr(file_path).unwrap_or_else(|e| {
                        tracing::warn!(error = %e, "OCR failed");
                        format!("[Scanned PDF: {file_path} - OCR failed]")
                    })
                }
                Ok(_) if self.tesseract_available => {
                    format!("[Scanned PDF: {file_path} - install ghostscript for OCR]")
                }
                Ok(_) => format!("[Scanned PDF: {file_path} - install tesseract for OCR]"),
                Err(e) => {
                    tracing::warn!(error = %e, "pdftotext failed");
                    format!("[PDF: {file_path} - text extraction failed]")
                }
            }
        } else {
            format!("[PDF: {file_path} - install poppler-utils for text extraction]")
        };

        let is_extracted = !text.starts_with("[PDF:") && !text.starts_with("[Scanned PDF:");
        let word_count = text.split_whitespace().count();

        let mut result = ProcessingResult::new(text)
            .with_tag("pdf".to_string())
            .with_tag("document".to_string());

        if is_extracted {
            result = result.with_tag("text-extracted".to_string());
        }

        result
            .metadata
            .insert("file_size".into(), serde_json::json!(file_size));
        if let Some(pages) = page_count {
            result
                .metadata
                .insert("page_count".into(), serde_json::json!(pages));
        }
        if is_extracted {
            result
                .metadata
                .insert("word_count".into(), serde_json::json!(word_count));
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::model::{KnowledgeNode, NodeKind};

    #[test]
    fn status_exposes_backend_details_and_types() {
        let processor = PdfProcessor::new();
        let status = processor.status();
        assert_eq!(status.name, "pdf");
        assert!(status.supported_types.contains(&"application/pdf".to_string()));
        assert!(status.details.contains_key("pdftotext_available"));
        assert!(status.details.contains_key("tesseract_available"));
        assert!(status.details.contains_key("ghostscript_available"));
    }

    #[test]
    fn handles_returns_application_pdf() {
        let processor = PdfProcessor::new();
        assert_eq!(processor.handles(), &["application/pdf"]);
    }

    #[test]
    fn name_returns_pdf() {
        let processor = PdfProcessor::new();
        assert_eq!(processor.name(), "pdf");
    }

    #[test]
    fn status_available_reflects_tool_presence() {
        let processor = PdfProcessor::new();
        let status = processor.status();
        // Available if at least one extraction tool is present
        let expected = processor.pdftotext_available || processor.tesseract_available;
        assert_eq!(status.available, expected);
    }

    #[tokio::test]
    async fn process_rejects_missing_file() {
        let processor = PdfProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor.process("/nonexistent/file.pdf", &node).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn process_handles_empty_pdf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.pdf");
        std::fs::write(&path, b"").unwrap();

        let processor = PdfProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        // Empty file should fail the file size or extraction gracefully
        let result = processor
            .process(path.to_str().unwrap(), &node)
            .await;
        // Either error or placeholder text — shouldn't panic
        if let Ok(r) = result {
            assert!(!r.suggested_tags.is_empty()); // always gets "pdf" tag
            assert!(r.suggested_tags.contains(&"pdf".to_string()));
        }
    }

    #[tokio::test]
    async fn process_populates_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.pdf");
        std::fs::write(&path, b"%PDF-1.4 fake content").unwrap();

        let processor = PdfProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor
            .process(path.to_str().unwrap(), &node)
            .await;
        if let Ok(r) = result {
            assert!(r.metadata.contains_key("file_size"));
            assert_eq!(r.metadata["file_size"], serde_json::json!(21));
            assert!(r.suggested_tags.contains(&"pdf".to_string()));
            assert!(r.suggested_tags.contains(&"document".to_string()));
        }
    }

    #[test]
    fn extract_with_pdftotext_on_nonexistent_fails() {
        let processor = PdfProcessor::new();
        if processor.pdftotext_available {
            let result = processor.extract_with_pdftotext("/nonexistent/file.pdf");
            assert!(result.is_err());
        }
    }

    #[test]
    fn page_count_returns_none_for_nonexistent() {
        let processor = PdfProcessor::new();
        assert!(processor.get_page_count("/nonexistent/file.pdf").is_none());
    }

    #[test]
    fn status_note_when_no_tools() {
        // Simulating — we can only test the constructor logic
        let processor = PdfProcessor {
            pdftotext_available: false,
            tesseract_available: false,
            ghostscript_available: false,
        };
        let status = processor.status();
        assert!(!status.available);
        assert_eq!(
            status.note.as_deref(),
            Some("No PDF extraction backend available")
        );
    }

    #[test]
    fn status_note_when_tesseract_without_ghostscript() {
        let processor = PdfProcessor {
            pdftotext_available: false,
            tesseract_available: true,
            ghostscript_available: false,
        };
        let status = processor.status();
        assert!(status.available);
        assert_eq!(
            status.note.as_deref(),
            Some("OCR available but ghostscript missing for PDF-to-image")
        );
    }
}
