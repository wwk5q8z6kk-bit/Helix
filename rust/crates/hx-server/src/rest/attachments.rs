use std::path::{Path, PathBuf};
use std::process::Command;

use super::voice::{is_audio_file, transcribe_audio, TranscriptionError, WhisperConfig};
use hx_engine::multimodal::image::{clip_tags_from_bytes, ClipTag};

#[derive(Debug, Clone)]
pub struct AttachmentTextExtractionOutcome {
    pub status: String,
    pub extracted_text: Option<String>,
    pub extracted_chars: usize,
}

struct AttachmentExtractionTools {
    pdftotext_bin: String,
    tesseract_bin: String,
    ffmpeg_bin: String,
}

impl AttachmentExtractionTools {
    fn from_env() -> Self {
        Self {
            pdftotext_bin: std::env::var("HELIX_ATTACHMENT_PDFTOTEXT_BIN")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "pdftotext".to_string()),
            tesseract_bin: std::env::var("HELIX_ATTACHMENT_TESSERACT_BIN")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "tesseract".to_string()),
            ffmpeg_bin: std::env::var("HELIX_ATTACHMENT_FFMPEG_BIN")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "ffmpeg".to_string()),
        }
    }
}

enum ExternalExtractionError {
    ToolMissing,
    Failed,
}

const TEXTUAL_CONTENT_TYPE_PREFIXES: &[&str] = &[
    "text/",
    "application/json",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
    "application/javascript",
    "application/x-javascript",
    "image/svg+xml",
];

const TEXTUAL_EXTENSIONS: &[&str] = &[
    "txt", "md", "markdown", "csv", "json", "xml", "yaml", "yml", "toml", "ini", "log", "rst",
    "rs", "py", "js", "ts", "tsx", "jsx", "html", "css", "sql", "svg",
];

const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff", "webp"];
const VIDEO_EXTENSIONS: &[&str] = &["mp4", "mov", "mkv", "webm", "avi", "m4v"];

pub fn extract_attachment_search_text(
    file_name: &str,
    content_type: Option<&str>,
    bytes: &[u8],
    max_chars: usize,
) -> AttachmentTextExtractionOutcome {
    let tools = AttachmentExtractionTools::from_env();
    extract_attachment_search_text_with_tools(file_name, content_type, bytes, max_chars, &tools)
}

fn extract_attachment_search_text_with_tools(
    file_name: &str,
    content_type: Option<&str>,
    bytes: &[u8],
    max_chars: usize,
    tools: &AttachmentExtractionTools,
) -> AttachmentTextExtractionOutcome {
    if is_textual_attachment(file_name, content_type) {
        return finalize_extraction_outcome(
            "indexed_text",
            normalize_attachment_search_blob(&String::from_utf8_lossy(bytes), max_chars),
        );
    }

    if is_pdf_attachment(file_name, content_type) {
        return match extract_pdf_text(bytes, max_chars, tools) {
            Ok(text) => finalize_extraction_outcome("indexed_pdf_text", text),
            Err(ExternalExtractionError::ToolMissing) => AttachmentTextExtractionOutcome {
                status: "tool_missing".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
            Err(ExternalExtractionError::Failed) => AttachmentTextExtractionOutcome {
                status: "extraction_failed".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
        };
    }

    if is_image_attachment(file_name, content_type) {
        let ocr_result = extract_image_ocr_text(file_name, bytes, max_chars, tools);
        let clip_tags = clip_tags_from_bytes(bytes).unwrap_or_default();
        let clip_text = format_clip_tags_text(&clip_tags);

        let mut text_parts = Vec::new();
        let mut status = None;
        let mut ocr_error: Option<ExternalExtractionError> = None;

        match ocr_result {
            Ok(text) => {
                if !text.is_empty() {
                    text_parts.push(text);
                    status = Some("indexed_ocr");
                }
            }
            Err(err) => {
                ocr_error = Some(err);
            }
        }

        if let Some(tag_text) = clip_text {
            text_parts.push(tag_text);
            if status.is_none() {
                status = Some("indexed_image_tags");
            }
        }

        if let Some(status) = status {
            return finalize_extraction_outcome(status, text_parts.join("\n"));
        }

        if let Some(err) = ocr_error {
            return match err {
                ExternalExtractionError::ToolMissing => AttachmentTextExtractionOutcome {
                    status: "tool_missing".to_string(),
                    extracted_text: None,
                    extracted_chars: 0,
                },
                ExternalExtractionError::Failed => AttachmentTextExtractionOutcome {
                    status: "extraction_failed".to_string(),
                    extracted_text: None,
                    extracted_chars: 0,
                },
            };
        }

        return AttachmentTextExtractionOutcome {
            status: "empty".to_string(),
            extracted_text: None,
            extracted_chars: 0,
        };
    }

    if is_video_attachment(file_name, content_type) {
        let audio_bytes = match extract_video_audio_bytes(file_name, bytes, tools) {
            Ok(audio) => audio,
            Err(ExternalExtractionError::ToolMissing) => {
                return AttachmentTextExtractionOutcome {
                    status: "tool_missing".to_string(),
                    extracted_text: None,
                    extracted_chars: 0,
                }
            }
            Err(ExternalExtractionError::Failed) => {
                return AttachmentTextExtractionOutcome {
                    status: "extraction_failed".to_string(),
                    extracted_text: None,
                    extracted_chars: 0,
                }
            }
        };

        let whisper_config = WhisperConfig::from_env();
        if !whisper_config.enabled {
            return AttachmentTextExtractionOutcome {
                status: "tool_missing".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            };
        }

        return match transcribe_audio("video-audio.wav", &audio_bytes, &whisper_config) {
            Ok(result) => finalize_extraction_outcome(
                "transcribed",
                normalize_attachment_search_blob(&result.text, max_chars),
            ),
            Err(TranscriptionError::WhisperNotAvailable) => AttachmentTextExtractionOutcome {
                status: "tool_missing".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
            Err(TranscriptionError::UnsupportedFormat(_)) => AttachmentTextExtractionOutcome {
                status: "unsupported".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
            Err(TranscriptionError::TranscriptionFailed(_))
            | Err(TranscriptionError::IoError(_)) => AttachmentTextExtractionOutcome {
                status: "extraction_failed".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
        };
    }

    if is_audio_file(file_name, content_type) {
        let whisper_config = WhisperConfig::from_env();
        if !whisper_config.enabled {
            return AttachmentTextExtractionOutcome {
                status: "tool_missing".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            };
        }

        return match transcribe_audio(file_name, bytes, &whisper_config) {
            Ok(result) => finalize_extraction_outcome(
                "transcribed",
                normalize_attachment_search_blob(&result.text, max_chars),
            ),
            Err(TranscriptionError::WhisperNotAvailable) => AttachmentTextExtractionOutcome {
                status: "tool_missing".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
            Err(TranscriptionError::UnsupportedFormat(_)) => AttachmentTextExtractionOutcome {
                status: "unsupported".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
            Err(TranscriptionError::TranscriptionFailed(_))
            | Err(TranscriptionError::IoError(_)) => AttachmentTextExtractionOutcome {
                status: "extraction_failed".to_string(),
                extracted_text: None,
                extracted_chars: 0,
            },
        };
    }

    AttachmentTextExtractionOutcome {
        status: "unsupported".to_string(),
        extracted_text: None,
        extracted_chars: 0,
    }
}

fn finalize_extraction_outcome(
    status: &str,
    normalized_text: String,
) -> AttachmentTextExtractionOutcome {
    if normalized_text.is_empty() {
        return AttachmentTextExtractionOutcome {
            status: "empty".to_string(),
            extracted_text: None,
            extracted_chars: 0,
        };
    }

    let extracted_chars = normalized_text.chars().count();
    AttachmentTextExtractionOutcome {
        status: status.to_string(),
        extracted_text: Some(normalized_text),
        extracted_chars,
    }
}

pub fn normalize_attachment_search_blob(raw: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }

    let cleaned = raw
        .chars()
        .map(|ch| {
            if ch == '\u{0}' || (ch.is_control() && !matches!(ch, '\n' | '\r' | '\t')) {
                ' '
            } else {
                ch
            }
        })
        .collect::<String>();
    let collapsed = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    truncate_by_chars(collapsed.trim(), max_chars)
}

pub fn split_attachment_search_chunks(
    raw: &str,
    chunk_char_limit: usize,
    max_chunks: usize,
) -> Vec<String> {
    if chunk_char_limit == 0 || max_chunks == 0 {
        return Vec::new();
    }

    let max_total_chars = chunk_char_limit.saturating_mul(max_chunks);
    if max_total_chars == 0 {
        return Vec::new();
    }

    let normalized = normalize_attachment_search_blob(raw, max_total_chars);
    if normalized.is_empty() {
        return Vec::new();
    }

    let chars: Vec<char> = normalized.chars().collect();
    let mut cursor = 0usize;
    let mut chunks = Vec::new();

    while cursor < chars.len() && chunks.len() < max_chunks {
        let target = (cursor + chunk_char_limit).min(chars.len());
        let end = if target >= chars.len() {
            chars.len()
        } else {
            select_chunk_boundary(&chars, cursor, target)
        };

        if end <= cursor {
            break;
        }

        let chunk = chars[cursor..end]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }

        cursor = end;
        while cursor < chars.len() && chars[cursor].is_whitespace() {
            cursor += 1;
        }
    }

    chunks
}

fn select_chunk_boundary(chars: &[char], start: usize, target: usize) -> usize {
    let hard_limit = target.min(chars.len());
    if hard_limit <= start {
        return start;
    }

    let min_backtrack = start + ((hard_limit - start) / 2);
    for idx in (min_backtrack..hard_limit).rev() {
        let previous = chars[idx - 1];
        if is_chunk_boundary_char(previous) {
            return idx;
        }
    }

    let forward_limit = (hard_limit + 72).min(chars.len());
    for idx in hard_limit..forward_limit {
        let previous = chars[idx - 1];
        if is_chunk_boundary_char(previous) {
            return idx;
        }
    }

    hard_limit
}

fn is_chunk_boundary_char(ch: char) -> bool {
    ch.is_whitespace() || matches!(ch, '.' | '!' | '?' | ';' | ':' | ',' | ')' | ']')
}

fn is_textual_attachment(file_name: &str, content_type: Option<&str>) -> bool {
    if let Some(content_type) = content_type {
        let normalized = content_type
            .split(';')
            .next()
            .unwrap_or(content_type)
            .trim()
            .to_ascii_lowercase();
        if TEXTUAL_CONTENT_TYPE_PREFIXES
            .iter()
            .any(|prefix| normalized.starts_with(prefix))
        {
            return true;
        }
    }

    let extension = file_name
        .rsplit('.')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    TEXTUAL_EXTENSIONS.iter().any(|item| *item == extension)
}

fn is_pdf_attachment(file_name: &str, content_type: Option<&str>) -> bool {
    if let Some(content_type) = content_type {
        let normalized = normalize_content_type(content_type);
        if normalized == "application/pdf" {
            return true;
        }
    }
    normalized_extension(file_name) == "pdf"
}

fn is_image_attachment(file_name: &str, content_type: Option<&str>) -> bool {
    if let Some(content_type) = content_type {
        let normalized = normalize_content_type(content_type);
        if normalized.starts_with("image/") && normalized != "image/svg+xml" {
            return true;
        }
    }

    let extension = normalized_extension(file_name);
    IMAGE_EXTENSIONS.iter().any(|value| *value == extension)
}

fn is_video_attachment(file_name: &str, content_type: Option<&str>) -> bool {
    if let Some(content_type) = content_type {
        let normalized = normalize_content_type(content_type);
        if normalized.starts_with("video/") {
            return true;
        }
    }

    let extension = normalized_extension(file_name);
    VIDEO_EXTENSIONS.iter().any(|value| *value == extension)
}

fn normalize_content_type(value: &str) -> String {
    value
        .split(';')
        .next()
        .unwrap_or(value)
        .trim()
        .to_ascii_lowercase()
}

fn normalized_extension(file_name: &str) -> String {
    file_name
        .rsplit('.')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
}

fn truncate_by_chars(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    value.chars().take(max_chars).collect()
}

fn extract_pdf_text(
    bytes: &[u8],
    max_chars: usize,
    tools: &AttachmentExtractionTools,
) -> Result<String, ExternalExtractionError> {
    let input_path = temp_file_path(".pdf");
    if std::fs::write(&input_path, bytes).is_err() {
        return Err(ExternalExtractionError::Failed);
    }

    let output = Command::new(&tools.pdftotext_bin)
        .arg("-q")
        .arg(&input_path)
        .arg("-")
        .output();
    let _ = std::fs::remove_file(&input_path);

    match output {
        Ok(result) => {
            if !result.status.success() {
                return Err(ExternalExtractionError::Failed);
            }
            Ok(normalize_attachment_search_blob(
                &String::from_utf8_lossy(&result.stdout),
                max_chars,
            ))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Err(ExternalExtractionError::ToolMissing)
        }
        Err(_) => Err(ExternalExtractionError::Failed),
    }
}

fn extract_image_ocr_text(
    file_name: &str,
    bytes: &[u8],
    max_chars: usize,
    tools: &AttachmentExtractionTools,
) -> Result<String, ExternalExtractionError> {
    let extension = normalized_extension(file_name);
    let suffix = if extension.is_empty() {
        ".img".to_string()
    } else {
        format!(".{extension}")
    };

    let input_path = temp_file_path(&suffix);
    if std::fs::write(&input_path, bytes).is_err() {
        return Err(ExternalExtractionError::Failed);
    }

    let output_base = temp_file_path("");
    let output = Command::new(&tools.tesseract_bin)
        .arg(&input_path)
        .arg(&output_base)
        .arg("-l")
        .arg("eng")
        .arg("txt")
        .output();
    let _ = std::fs::remove_file(&input_path);

    let output_txt_path = with_txt_extension(&output_base);
    let extraction_result = match output {
        Ok(result) => {
            if !result.status.success() {
                Err(ExternalExtractionError::Failed)
            } else {
                match std::fs::read_to_string(&output_txt_path) {
                    Ok(text) => Ok(normalize_attachment_search_blob(&text, max_chars)),
                    Err(_) => Err(ExternalExtractionError::Failed),
                }
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Err(ExternalExtractionError::ToolMissing)
        }
        Err(_) => Err(ExternalExtractionError::Failed),
    };
    let _ = std::fs::remove_file(&output_txt_path);

    extraction_result
}

fn extract_video_audio_bytes(
    file_name: &str,
    bytes: &[u8],
    tools: &AttachmentExtractionTools,
) -> Result<Vec<u8>, ExternalExtractionError> {
    let extension = normalized_extension(file_name);
    let input_suffix = if extension.is_empty() {
        ".video".to_string()
    } else {
        format!(".{extension}")
    };

    let input_path = temp_file_path(&input_suffix);
    if std::fs::write(&input_path, bytes).is_err() {
        return Err(ExternalExtractionError::Failed);
    }

    let output_path = temp_file_path(".wav");
    let output = Command::new(&tools.ffmpeg_bin)
        .arg("-y")
        .arg("-loglevel")
        .arg("error")
        .arg("-i")
        .arg(&input_path)
        .arg("-vn")
        .arg("-acodec")
        .arg("pcm_s16le")
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg(&output_path)
        .output();
    let _ = std::fs::remove_file(&input_path);

    let extraction_result = match output {
        Ok(result) => {
            if !result.status.success() {
                Err(ExternalExtractionError::Failed)
            } else {
                std::fs::read(&output_path).map_err(|_| ExternalExtractionError::Failed)
            }
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            Err(ExternalExtractionError::ToolMissing)
        }
        Err(_) => Err(ExternalExtractionError::Failed),
    };
    let _ = std::fs::remove_file(&output_path);

    extraction_result
}

fn format_clip_tags_text(tags: &[ClipTag]) -> Option<String> {
    if tags.is_empty() {
        return None;
    }

    let labels = tags
        .iter()
        .map(|tag| tag.label.as_str())
        .collect::<Vec<_>>();

    Some(format!("Image labels: {}", labels.join(", ")))
}

fn temp_file_path(suffix: &str) -> PathBuf {
    std::env::temp_dir().join(format!("helix-att-{}{}", uuid::Uuid::now_v7(), suffix))
}

fn with_txt_extension(path: &Path) -> PathBuf {
    let mut output = path.to_path_buf();
    output.set_extension("txt");
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_text_for_plain_text_payloads() {
        let outcome = extract_attachment_search_text(
            "notes.txt",
            Some("text/plain"),
            b"alpha \n beta\t gamma",
            128,
        );
        assert_eq!(outcome.status, "indexed_text");
        assert_eq!(outcome.extracted_chars, 16);
        assert_eq!(outcome.extracted_text.as_deref(), Some("alpha beta gamma"));
    }

    #[test]
    fn returns_tool_missing_for_image_ocr_without_tesseract() {
        let tools = AttachmentExtractionTools {
            pdftotext_bin: "missing-pdftotext-bin".to_string(),
            tesseract_bin: "missing-tesseract-bin".to_string(),
            ffmpeg_bin: "missing-ffmpeg-bin".to_string(),
        };
        let outcome = extract_attachment_search_text_with_tools(
            "diagram.png",
            Some("image/png"),
            b"\x89PNG",
            128,
            &tools,
        );
        assert_eq!(outcome.status, "tool_missing");
        assert_eq!(outcome.extracted_chars, 0);
        assert!(outcome.extracted_text.is_none());
    }

    #[test]
    fn returns_tool_missing_for_pdf_without_pdftotext() {
        let tools = AttachmentExtractionTools {
            pdftotext_bin: "missing-pdftotext-bin".to_string(),
            tesseract_bin: "missing-tesseract-bin".to_string(),
            ffmpeg_bin: "missing-ffmpeg-bin".to_string(),
        };
        let outcome = extract_attachment_search_text_with_tools(
            "doc.pdf",
            Some("application/pdf"),
            b"%PDF-1.4",
            128,
            &tools,
        );
        assert_eq!(outcome.status, "tool_missing");
        assert_eq!(outcome.extracted_chars, 0);
        assert!(outcome.extracted_text.is_none());
    }

    #[test]
    fn returns_tool_missing_for_audio_without_whisper() {
        let previous = std::env::var("HELIX_WHISPER_BIN").ok();
        std::env::set_var("HELIX_WHISPER_BIN", "missing-whisper-bin");

        let outcome = extract_attachment_search_text("note.mp3", Some("audio/mpeg"), b"ID3", 128);

        assert_eq!(outcome.status, "tool_missing");
        assert_eq!(outcome.extracted_chars, 0);
        assert!(outcome.extracted_text.is_none());

        match previous {
            Some(value) => std::env::set_var("HELIX_WHISPER_BIN", value),
            None => std::env::remove_var("HELIX_WHISPER_BIN"),
        }
    }

    #[test]
    fn returns_tool_missing_for_video_without_ffmpeg() {
        let tools = AttachmentExtractionTools {
            pdftotext_bin: "missing-pdftotext-bin".to_string(),
            tesseract_bin: "missing-tesseract-bin".to_string(),
            ffmpeg_bin: "missing-ffmpeg-bin".to_string(),
        };

        let outcome = extract_attachment_search_text_with_tools(
            "demo.mp4",
            Some("video/mp4"),
            b"not-a-real-video",
            128,
            &tools,
        );

        assert_eq!(outcome.status, "tool_missing");
        assert_eq!(outcome.extracted_chars, 0);
        assert!(outcome.extracted_text.is_none());
    }

    #[test]
    fn normalizes_and_truncates_control_characters() {
        let normalized = normalize_attachment_search_blob("A\u{0}B\u{7}  C", 3);
        assert_eq!(normalized, "A B");
    }

    #[test]
    fn split_attachment_search_chunks_prefers_sentence_boundaries() {
        let text = "First sentence explains context. Second sentence adds implementation details and rollout notes. Third sentence closes the summary.";
        let chunks = split_attachment_search_chunks(text, 55, 8);
        assert!(chunks.len() >= 2);
        assert_eq!(
            chunks.first().cloned(),
            Some("First sentence explains context. Second sentence adds".to_string())
        );
        assert!(chunks.iter().all(|chunk| chunk.len() <= 90));
    }

    #[test]
    fn split_attachment_search_chunks_respects_limits() {
        let text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau";
        let chunks = split_attachment_search_chunks(text, 18, 3);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 25));
    }

    #[test]
    fn detects_textual_by_extension_case_insensitive() {
        assert!(is_textual_attachment("README.MD", None));
        assert!(is_textual_attachment("notes.TXT", None));
    }

    #[test]
    fn pdf_detection_accepts_content_type_with_parameters() {
        assert!(is_pdf_attachment("file.bin", Some("application/pdf; charset=binary")));
        assert!(is_pdf_attachment("report.PDF", None));
    }

    #[test]
    fn svg_is_textual_not_image() {
        assert!(is_textual_attachment("diagram.svg", Some("image/svg+xml")));
        assert!(!is_image_attachment("diagram.svg", Some("image/svg+xml")));
    }

    #[test]
    fn image_detection_handles_uppercase_extensions() {
        assert!(is_image_attachment("photo.JPEG", None));
        assert!(is_image_attachment("scan.PNG", None));
    }

    #[test]
    fn video_detection_prefers_content_type() {
        assert!(is_video_attachment("clip.unknown", Some("video/mp4")));
        assert!(is_video_attachment("clip.MOV", None));
    }
}
