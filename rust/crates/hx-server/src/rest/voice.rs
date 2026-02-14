//! Voice notes and audio transcription pipeline.
//!
//! Supports audio upload with automatic transcription using Whisper.

use std::process::Command;

use uuid::Uuid;

const ENV_WHISPER_BIN: &str = "HELIX_WHISPER_BIN";
const ENV_WHISPER_MODEL: &str = "HELIX_WHISPER_MODEL";
const ENV_WHISPER_LANGUAGE: &str = "HELIX_WHISPER_LANGUAGE";

/// Supported audio formats.
const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "m4a", "webm", "ogg", "flac", "aac", "wma"];

const AUDIO_CONTENT_TYPES: &[&str] = &[
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/x-m4a",
    "audio/webm",
    "audio/ogg",
    "audio/flac",
    "audio/aac",
    "audio/x-aac",
];

/// Configuration for Whisper transcription.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    pub bin_path: String,
    pub model: String,
    pub language: Option<String>,
    pub enabled: bool,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl WhisperConfig {
    pub fn from_env() -> Self {
        let bin_path = std::env::var(ENV_WHISPER_BIN)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "whisper".to_string());

        let model = std::env::var(ENV_WHISPER_MODEL)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "base".to_string());

        let language = std::env::var(ENV_WHISPER_LANGUAGE)
            .ok()
            .filter(|s| !s.trim().is_empty());

        // Check if whisper is available
        let enabled = Command::new(&bin_path).arg("--help").output().is_ok();

        Self {
            bin_path,
            model,
            language,
            enabled,
        }
    }
}

/// Result of transcribing an audio file.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TranscriptionResult {
    pub text: String,
    pub language: Option<String>,
    pub duration_seconds: Option<f64>,
    pub segments: Vec<TranscriptionSegment>,
}

/// A segment of transcription with timing info.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TranscriptionSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

/// Error during transcription.
#[derive(Debug)]
pub enum TranscriptionError {
    WhisperNotAvailable,
    UnsupportedFormat(String),
    TranscriptionFailed(String),
    IoError(std::io::Error),
}

impl std::fmt::Display for TranscriptionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WhisperNotAvailable => write!(f, "Whisper is not available"),
            Self::UnsupportedFormat(ext) => write!(f, "Unsupported audio format: {ext}"),
            Self::TranscriptionFailed(msg) => write!(f, "Transcription failed: {msg}"),
            Self::IoError(e) => write!(f, "IO error: {e}"),
        }
    }
}

impl std::error::Error for TranscriptionError {}

impl From<std::io::Error> for TranscriptionError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

/// Check if the file appears to be an audio file.
pub fn is_audio_file(file_name: &str, content_type: Option<&str>) -> bool {
    // Check content type
    if let Some(ct) = content_type {
        if AUDIO_CONTENT_TYPES
            .iter()
            .any(|&prefix| ct.starts_with(prefix))
        {
            return true;
        }
    }

    // Check extension
    let ext = file_name
        .rsplit('.')
        .next()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    AUDIO_EXTENSIONS.contains(&ext.as_str())
}

/// Transcribe an audio file using Whisper.
pub fn transcribe_audio(
    file_name: &str,
    audio_bytes: &[u8],
    config: &WhisperConfig,
) -> Result<TranscriptionResult, TranscriptionError> {
    if !config.enabled {
        return Err(TranscriptionError::WhisperNotAvailable);
    }

    // Validate audio format
    let ext = file_name
        .rsplit('.')
        .next()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if !AUDIO_EXTENSIONS.contains(&ext.as_str()) {
        return Err(TranscriptionError::UnsupportedFormat(ext));
    }

    // Write audio to temporary file
    let temp_dir = std::env::temp_dir();
    let temp_id = Uuid::now_v7();
    let temp_audio_path = temp_dir.join(format!("helix_voice_{temp_id}.{ext}"));
    std::fs::write(&temp_audio_path, audio_bytes)?;

    // Build whisper command
    let mut cmd = Command::new(&config.bin_path);
    cmd.arg(&temp_audio_path)
        .arg("--model")
        .arg(&config.model)
        .arg("--output_format")
        .arg("txt")
        .arg("--output_dir")
        .arg(&temp_dir);

    if let Some(ref lang) = config.language {
        cmd.arg("--language").arg(lang);
    }

    // Run whisper
    let output = cmd.output()?;

    // Clean up audio file
    let _ = std::fs::remove_file(&temp_audio_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(TranscriptionError::TranscriptionFailed(stderr.to_string()));
    }

    // Read transcription output
    let txt_path = temp_dir.join(format!("helix_voice_{temp_id}.txt"));
    let text = std::fs::read_to_string(&txt_path).unwrap_or_default();

    // Clean up output file
    let _ = std::fs::remove_file(&txt_path);

    Ok(TranscriptionResult {
        text: text.trim().to_string(),
        language: config.language.clone(),
        duration_seconds: None,
        segments: Vec::new(),
    })
}

/// Transcribe audio using OpenAI Whisper API (fallback for when local whisper isn't available).
pub async fn transcribe_audio_api(
    file_name: &str,
    audio_bytes: &[u8],
    api_key: &str,
) -> Result<TranscriptionResult, TranscriptionError> {
    // Validate audio format
    let ext = file_name
        .rsplit('.')
        .next()
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    if !AUDIO_EXTENSIONS.contains(&ext.as_str()) {
        return Err(TranscriptionError::UnsupportedFormat(ext));
    }

    let client = reqwest::Client::new();

    // Create multipart form
    let part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
        .file_name(file_name.to_string())
        .mime_str(&format!("audio/{ext}"))
        .unwrap_or_else(|_| {
            reqwest::multipart::Part::bytes(audio_bytes.to_vec()).file_name(file_name.to_string())
        });

    let form = reqwest::multipart::Form::new()
        .part("file", part)
        .text("model", "whisper-1");

    let response = client
        .post("https://api.openai.com/v1/audio/transcriptions")
        .header("Authorization", format!("Bearer {api_key}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| TranscriptionError::TranscriptionFailed(e.to_string()))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(TranscriptionError::TranscriptionFailed(format!(
            "API error {status}: {body}"
        )));
    }

    #[derive(serde::Deserialize)]
    struct ApiResponse {
        text: String,
    }

    let api_response: ApiResponse = response
        .json()
        .await
        .map_err(|e| TranscriptionError::TranscriptionFailed(e.to_string()))?;

    Ok(TranscriptionResult {
        text: api_response.text,
        language: None,
        duration_seconds: None,
        segments: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_audio_file() {
        assert!(is_audio_file("recording.mp3", None));
        assert!(is_audio_file("voice.wav", None));
        assert!(is_audio_file("memo.m4a", None));
        assert!(is_audio_file("audio.webm", None));
        assert!(!is_audio_file("document.pdf", None));
        assert!(!is_audio_file("image.png", None));

        assert!(is_audio_file("unknown", Some("audio/mpeg")));
        assert!(is_audio_file("unknown", Some("audio/wav")));
        assert!(!is_audio_file("unknown", Some("text/plain")));
    }
}
