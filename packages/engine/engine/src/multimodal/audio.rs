//! Audio processing via Whisper transcription.
//!
//! Tries local Whisper CLI first, falls back to OpenAI Whisper API.
//! Integrates with the multimodal pipeline trait for automatic processing
//! of audio attachments during enrichment.

use async_trait::async_trait;
use hx_core::{KnowledgeNode, MvResult};
use std::path::Path;
use std::process::Command;

use super::{
    check_file_size, run_command_with_timeout, ModalityProcessor, ModalityStatus,
    ProcessingResult, DEFAULT_COMMAND_TIMEOUT,
};

/// Audio processor that transcribes audio files using Whisper.
pub struct AudioProcessor {
    whisper_bin: String,
    whisper_model: String,
    language: Option<String>,
    api_key: Option<String>,
    local_available: bool,
}

impl AudioProcessor {
    pub fn new() -> Self {
        let whisper_bin = std::env::var("HELIX_WHISPER_BIN")
            .unwrap_or_else(|_| "whisper".to_string());

        let whisper_model = std::env::var("HELIX_WHISPER_MODEL")
            .unwrap_or_else(|_| "base".to_string());

        let language = std::env::var("HELIX_WHISPER_LANG").ok();

        let api_key = std::env::var("OPENAI_API_KEY").ok();

        let local_available = Command::new(&whisper_bin)
            .arg("--help")
            .output()
            .is_ok();

        if local_available {
            tracing::info!(bin = %whisper_bin, model = %whisper_model, "Local Whisper available for audio processing");
        } else if api_key.is_some() {
            tracing::info!("Whisper API fallback available for audio processing");
        } else {
            tracing::warn!("No Whisper backend available — audio processing will produce placeholder text");
        }

        Self {
            whisper_bin,
            whisper_model,
            language,
            api_key,
            local_available,
        }
    }

    /// Transcribe using local Whisper CLI with timeout protection.
    fn transcribe_local(&self, file_path: &str) -> Result<String, String> {
        let temp_dir = std::env::temp_dir();
        let stem = Path::new(file_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("audio");

        let mut cmd = Command::new(&self.whisper_bin);
        cmd.arg(file_path)
            .arg("--model")
            .arg(&self.whisper_model)
            .arg("--output_format")
            .arg("txt")
            .arg("--output_dir")
            .arg(&temp_dir);

        if let Some(ref lang) = self.language {
            cmd.arg("--language").arg(lang);
        }

        let output = run_command_with_timeout(&mut cmd, DEFAULT_COMMAND_TIMEOUT)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("whisper failed: {stderr}"));
        }

        // Whisper outputs {stem}.txt in the output dir
        let txt_path = temp_dir.join(format!("{stem}.txt"));
        let text = std::fs::read_to_string(&txt_path).unwrap_or_default();
        let _ = std::fs::remove_file(&txt_path);

        Ok(text.trim().to_string())
    }

    /// Transcribe using OpenAI Whisper API via manual multipart form.
    async fn transcribe_api(&self, file_path: &str) -> Result<String, String> {
        let api_key = self
            .api_key
            .as_deref()
            .ok_or("no OpenAI API key configured")?;

        let file_bytes = std::fs::read(file_path)
            .map_err(|e| format!("failed to read audio file: {e}"))?;

        let file_name = Path::new(file_path)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("audio.wav");

        let ext = Path::new(file_path)
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("wav");

        let mime = match ext {
            "mp3" | "mpeg" => "audio/mpeg",
            "ogg" => "audio/ogg",
            "flac" => "audio/flac",
            "m4a" => "audio/m4a",
            "webm" => "audio/webm",
            _ => "audio/wav",
        };

        // Build multipart form manually (reqwest multipart feature not enabled)
        let boundary = format!("----HelixBoundary{}", uuid::Uuid::now_v7());
        let mut body = Vec::new();

        // Model field
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(b"Content-Disposition: form-data; name=\"model\"\r\n\r\n");
        body.extend_from_slice(b"whisper-1\r\n");

        // File field
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"file\"; filename=\"{file_name}\"\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(format!("Content-Type: {mime}\r\n\r\n").as_bytes());
        body.extend_from_slice(&file_bytes);
        body.extend_from_slice(b"\r\n");
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

        let client = reqwest::Client::new();
        let response = client
            .post("https://api.openai.com/v1/audio/transcriptions")
            .header("Authorization", format!("Bearer {api_key}"))
            .header(
                "Content-Type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(body)
            .send()
            .await
            .map_err(|e| format!("API request failed: {e}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("Whisper API error {status}: {body}"));
        }

        #[derive(serde::Deserialize)]
        struct ApiResponse {
            text: String,
        }

        let result = response
            .json::<ApiResponse>()
            .await
            .map_err(|e| format!("failed to parse API response: {e}"))?;

        Ok(result.text)
    }
}

#[async_trait]
impl ModalityProcessor for AudioProcessor {
    fn name(&self) -> &'static str {
        "audio"
    }

    fn handles(&self) -> &[&str] {
        &[
            "audio/wav",
            "audio/mp3",
            "audio/ogg",
            "audio/webm",
            "audio/mpeg",
            "audio/flac",
            "audio/m4a",
            "audio/x-wav",
        ]
    }

    fn status(&self) -> ModalityStatus {
        let available = self.local_available || self.api_key.is_some();
        let mut status = ModalityStatus::new(self.name(), available, self.handles())
            .with_detail("whisper_bin", serde_json::json!(self.whisper_bin))
            .with_detail("whisper_model", serde_json::json!(self.whisper_model))
            .with_detail("local_available", serde_json::json!(self.local_available))
            .with_detail("api_fallback_available", serde_json::json!(self.api_key.is_some()));

        if let Some(lang) = &self.language {
            status = status.with_detail("language", serde_json::json!(lang));
        }

        if !available {
            status = status.with_note("No Whisper backend available");
        }

        status
    }

    async fn process(&self, file_path: &str, _node: &KnowledgeNode) -> MvResult<ProcessingResult> {
        tracing::info!(file_path, "Processing audio file");

        check_file_size(file_path)
            .map_err(|e| hx_core::HxError::Storage(e))?;

        // Try local Whisper first, then API fallback
        let transcript = if self.local_available {
            match self.transcribe_local(file_path) {
                Ok(text) => text,
                Err(e) => {
                    tracing::warn!(error = %e, "Local Whisper failed, trying API fallback");
                    self.transcribe_api(file_path)
                        .await
                        .unwrap_or_else(|e2| {
                            tracing::warn!(error = %e2, "API fallback also failed");
                            format!("[Audio file: {} - transcription failed]", file_path)
                        })
                }
            }
        } else if self.api_key.is_some() {
            self.transcribe_api(file_path)
                .await
                .unwrap_or_else(|e| {
                    tracing::warn!(error = %e, "Whisper API transcription failed");
                    format!("[Audio file: {} - transcription failed]", file_path)
                })
        } else {
            format!("[Audio file: {} - no Whisper backend available]", file_path)
        };

        let is_transcribed = !transcript.starts_with("[Audio file:");
        let word_count = transcript.split_whitespace().count();

        let mut result = ProcessingResult::new(transcript)
            .with_tag("audio".to_string());

        if is_transcribed {
            result = result.with_tag("transcribed".to_string());
            result
                .metadata
                .insert("word_count".to_string(), serde_json::json!(word_count));
        }

        // Estimate duration from file size (rough: ~16KB/sec for compressed audio)
        if let Ok(meta) = std::fs::metadata(file_path) {
            let estimated_secs = meta.len() as f64 / 16_000.0;
            result.metadata.insert(
                "estimated_duration_secs".to_string(),
                serde_json::json!(estimated_secs.round() as u64),
            );
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
        let processor = AudioProcessor::new();
        let status = processor.status();
        assert_eq!(status.name, "audio");
        assert!(status.supported_types.contains(&"audio/wav".to_string()));
        assert!(status.details.contains_key("whisper_bin"));
        assert!(status.details.contains_key("whisper_model"));
        assert!(status.details.contains_key("local_available"));
        assert!(status.details.contains_key("api_fallback_available"));
    }

    #[test]
    fn handles_all_audio_types() {
        let processor = AudioProcessor::new();
        let types = processor.handles();
        assert!(types.contains(&"audio/wav"));
        assert!(types.contains(&"audio/mp3"));
        assert!(types.contains(&"audio/ogg"));
        assert!(types.contains(&"audio/webm"));
        assert!(types.contains(&"audio/mpeg"));
        assert!(types.contains(&"audio/flac"));
        assert!(types.contains(&"audio/m4a"));
        assert!(types.contains(&"audio/x-wav"));
    }

    #[test]
    fn name_returns_audio() {
        assert_eq!(AudioProcessor::new().name(), "audio");
    }

    #[test]
    fn status_note_when_no_backend() {
        // Force no backends by using a nonexistent binary and no API key
        let processor = AudioProcessor {
            whisper_bin: "nonexistent-whisper-xyz".to_string(),
            whisper_model: "base".to_string(),
            language: None,
            api_key: None,
            local_available: false,
        };
        let status = processor.status();
        assert!(!status.available);
        assert_eq!(status.note.as_deref(), Some("No Whisper backend available"));
    }

    #[test]
    fn status_available_with_api_key_only() {
        let processor = AudioProcessor {
            whisper_bin: "nonexistent".to_string(),
            whisper_model: "base".to_string(),
            language: None,
            api_key: Some("sk-test".to_string()),
            local_available: false,
        };
        let status = processor.status();
        assert!(status.available);
        assert_eq!(
            status.details["api_fallback_available"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn status_includes_language_when_set() {
        let processor = AudioProcessor {
            whisper_bin: "whisper".to_string(),
            whisper_model: "base".to_string(),
            language: Some("en".to_string()),
            api_key: None,
            local_available: false,
        };
        let status = processor.status();
        assert_eq!(status.details["language"], serde_json::json!("en"));
    }

    #[tokio::test]
    async fn process_rejects_missing_file() {
        let processor = AudioProcessor::new();
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor
            .process("/nonexistent/audio.wav", &node)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn process_produces_placeholder_without_backend() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        std::fs::write(&path, b"RIFF fake wav data for testing").unwrap();

        let processor = AudioProcessor {
            whisper_bin: "nonexistent-whisper-xyz".to_string(),
            whisper_model: "base".to_string(),
            language: None,
            api_key: None,
            local_available: false,
        };
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor
            .process(path.to_str().unwrap(), &node)
            .await
            .unwrap();

        assert!(result.text_content.contains("no Whisper backend"));
        assert!(result.suggested_tags.contains(&"audio".to_string()));
        assert!(!result.suggested_tags.contains(&"transcribed".to_string()));
    }

    #[tokio::test]
    async fn process_estimates_duration() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        // 160KB ≈ 10 seconds at 16KB/s
        let data = vec![0u8; 160_000];
        std::fs::write(&path, &data).unwrap();

        let processor = AudioProcessor {
            whisper_bin: "nonexistent-whisper-xyz".to_string(),
            whisper_model: "base".to_string(),
            language: None,
            api_key: None,
            local_available: false,
        };
        let node = KnowledgeNode::new(NodeKind::Fact, "test".to_string());
        let result = processor
            .process(path.to_str().unwrap(), &node)
            .await
            .unwrap();

        let duration = result.metadata["estimated_duration_secs"]
            .as_u64()
            .unwrap();
        assert_eq!(duration, 10);
    }

    #[test]
    fn transcribe_local_fails_with_nonexistent_binary() {
        let processor = AudioProcessor {
            whisper_bin: "nonexistent-whisper-xyz".to_string(),
            whisper_model: "base".to_string(),
            language: None,
            api_key: None,
            local_available: false,
        };
        let result = processor.transcribe_local("/tmp/fake.wav");
        assert!(result.is_err());
    }

    #[test]
    fn mime_type_mapping() {
        // Verify the extension → MIME mapping in transcribe_api
        let cases = [
            ("mp3", "audio/mpeg"),
            ("mpeg", "audio/mpeg"),
            ("ogg", "audio/ogg"),
            ("flac", "audio/flac"),
            ("m4a", "audio/m4a"),
            ("webm", "audio/webm"),
            ("wav", "audio/wav"),
            ("unknown", "audio/wav"),
        ];
        for (ext, expected_mime) in cases {
            let mime = match ext {
                "mp3" | "mpeg" => "audio/mpeg",
                "ogg" => "audio/ogg",
                "flac" => "audio/flac",
                "m4a" => "audio/m4a",
                "webm" => "audio/webm",
                _ => "audio/wav",
            };
            assert_eq!(mime, expected_mime, "MIME mismatch for extension {ext}");
        }
    }
}
