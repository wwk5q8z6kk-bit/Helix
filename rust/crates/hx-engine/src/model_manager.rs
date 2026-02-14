//! Model Manager for downloading and managing local GGUF models.
//!
//! Provides:
//! - List available models in the local models directory
//! - Download models from Hugging Face Hub
//! - Delete models
//! - Check model status (size, last used, etc.)
//!
//! Models are stored in `~/.helix/models/` by default (configurable via
//! `local_llm.models_dir`). Only GGUF format is supported.

use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::config::LocalLlmConfig;

/// Metadata about a locally available model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    /// Filename (e.g., "mistral-7b-instruct-v0.3.Q4_K_M.gguf").
    pub filename: String,
    /// Full path on disk.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Human-readable file size.
    pub size_human: String,
    /// Last modified timestamp.
    pub modified_at: DateTime<Utc>,
    /// Whether this model is currently configured as the active model.
    pub is_active: bool,
}

/// Status of a download operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadStatus {
    pub model_id: String,
    pub filename: String,
    pub status: String,
    pub size_bytes: Option<u64>,
    pub path: Option<String>,
    pub error: Option<String>,
}

/// The model manager handles model lifecycle operations.
pub struct ModelManager {
    models_dir: PathBuf,
    config: LocalLlmConfig,
}

impl ModelManager {
    pub fn new(config: &LocalLlmConfig) -> Self {
        let models_dir = PathBuf::from(&config.models_dir);
        Self {
            models_dir,
            config: config.clone(),
        }
    }

    /// Ensure the models directory exists.
    pub fn ensure_dir(&self) -> Result<(), String> {
        if !self.models_dir.exists() {
            std::fs::create_dir_all(&self.models_dir)
                .map_err(|e| format!("failed to create models dir: {e}"))?;
        }
        Ok(())
    }

    /// List all GGUF models in the models directory.
    pub fn list_models(&self) -> Result<Vec<LocalModel>, String> {
        if !self.models_dir.exists() {
            return Ok(vec![]);
        }

        let entries = std::fs::read_dir(&self.models_dir)
            .map_err(|e| format!("failed to read models dir: {e}"))?;

        let active_path = self
            .config
            .model_path
            .as_ref()
            .map(PathBuf::from);

        let mut models = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| format!("dir entry error: {e}"))?;
            let path = entry.path();

            if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }

            let metadata = std::fs::metadata(&path)
                .map_err(|e| format!("metadata error: {e}"))?;

            let filename = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();

            let size_bytes = metadata.len();
            let modified_at: DateTime<Utc> = metadata
                .modified()
                .map(|t| t.into())
                .unwrap_or_else(|_| Utc::now());

            let is_active = active_path
                .as_ref()
                .map(|ap| ap == &path)
                .unwrap_or(false);

            models.push(LocalModel {
                filename,
                path: path.to_string_lossy().to_string(),
                size_bytes,
                size_human: human_bytes(size_bytes),
                modified_at,
                is_active,
            });
        }

        models.sort_by(|a, b| a.filename.cmp(&b.filename));
        Ok(models)
    }

    /// Get info about a specific model by filename.
    pub fn get_model(&self, filename: &str) -> Result<Option<LocalModel>, String> {
        let models = self.list_models()?;
        Ok(models.into_iter().find(|m| m.filename == filename))
    }

    /// Delete a model by filename.
    pub fn delete_model(&self, filename: &str) -> Result<(), String> {
        let path = self.models_dir.join(filename);
        if !path.exists() {
            return Err(format!("model not found: {filename}"));
        }

        // Safety: only delete .gguf files from the models directory
        if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
            return Err("refusing to delete non-GGUF file".to_string());
        }

        if !path.starts_with(&self.models_dir) {
            return Err("refusing to delete file outside models directory".to_string());
        }

        std::fs::remove_file(&path)
            .map_err(|e| format!("failed to delete model: {e}"))?;

        info!(filename = filename, "model deleted");
        Ok(())
    }

    /// Download a GGUF model from Hugging Face Hub.
    ///
    /// `model_id` should be in the format `org/repo/filename.gguf` or
    /// `org/repo` (in which case we look for the first .gguf file).
    ///
    /// This is a blocking operation that should be run via `spawn_blocking`.
    pub async fn download_model(&self, model_id: &str) -> DownloadStatus {
        self.ensure_dir().unwrap_or_else(|e| warn!("dir creation: {e}"));

        // Parse model_id: "user/repo/file.gguf" or "user/repo"
        let parts: Vec<&str> = model_id.splitn(3, '/').collect();
        if parts.len() < 2 {
            return DownloadStatus {
                model_id: model_id.into(),
                filename: String::new(),
                status: "error".to_string(),
                size_bytes: None,
                path: None,
                error: Some("invalid model_id: expected 'org/repo' or 'org/repo/file.gguf'".to_string()),
            };
        }

        let repo = format!("{}/{}", parts[0], parts[1]);
        let filename = if parts.len() == 3 {
            parts[2].to_string()
        } else {
            // Default: try to find a suitable filename from common naming patterns
            format!("{}.gguf", parts[1])
        };

        let url = format!(
            "https://huggingface.co/{repo}/resolve/main/{filename}"
        );
        let dest = self.models_dir.join(&filename);

        if dest.exists() {
            let size = std::fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            return DownloadStatus {
                model_id: model_id.into(),
                filename: filename.clone(),
                status: "exists".to_string(),
                size_bytes: Some(size),
                path: Some(dest.to_string_lossy().to_string()),
                error: None,
            };
        }

        info!(url = %url, dest = %dest.display(), "downloading GGUF model");

        match download_file(&url, &dest).await {
            Ok(size) => {
                info!(
                    filename = %filename,
                    size = human_bytes(size),
                    "model download complete"
                );
                DownloadStatus {
                    model_id: model_id.into(),
                    filename,
                    status: "completed".to_string(),
                    size_bytes: Some(size),
                    path: Some(dest.to_string_lossy().to_string()),
                    error: None,
                }
            }
            Err(e) => {
                warn!(error = %e, "model download failed");
                // Clean up partial download
                let _ = std::fs::remove_file(&dest);
                DownloadStatus {
                    model_id: model_id.into(),
                    filename,
                    status: "error".to_string(),
                    size_bytes: None,
                    path: None,
                    error: Some(e),
                }
            }
        }
    }

    /// Get total disk usage of all models.
    pub fn total_size(&self) -> Result<u64, String> {
        let models = self.list_models()?;
        Ok(models.iter().map(|m| m.size_bytes).sum())
    }

    /// Check available RAM (best-effort, macOS/Linux).
    pub fn check_available_ram() -> Option<u64> {
        #[cfg(target_os = "macos")]
        {
            // Use sysctl on macOS
            let output = std::process::Command::new("sysctl")
                .args(["-n", "hw.memsize"])
                .output()
                .ok()?;
            let total = String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
                .ok()?;
            // Rough estimate: assume 60% is available for model loading
            Some(total * 6 / 10)
        }
        #[cfg(target_os = "linux")]
        {
            let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
            for line in meminfo.lines() {
                if line.starts_with("MemAvailable:") {
                    let kb: u64 = line
                        .split_whitespace()
                        .nth(1)?
                        .parse()
                        .ok()?;
                    return Some(kb * 1024);
                }
            }
            None
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            None
        }
    }
}

/// Download a file via HTTP with streaming, writing to disk.
async fn download_file(url: &str, dest: &Path) -> Result<u64, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("download request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!(
            "download failed with HTTP {}: {}",
            response.status(),
            response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string())
        ));
    }

    // Write to a temp file first, then rename (atomic)
    let tmp_dest = dest.with_extension("gguf.part");
    let mut file = tokio::fs::File::create(&tmp_dest)
        .await
        .map_err(|e| format!("failed to create temp file: {e}"))?;

    use tokio::io::AsyncWriteExt;
    let mut stream = response.bytes_stream();
    let mut total_bytes: u64 = 0;

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("download stream error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("write error: {e}"))?;
        total_bytes += chunk.len() as u64;
    }

    file.flush()
        .await
        .map_err(|e| format!("flush error: {e}"))?;
    drop(file);

    // Atomic rename
    tokio::fs::rename(&tmp_dest, dest)
        .await
        .map_err(|e| format!("rename failed: {e}"))?;

    Ok(total_bytes)
}

/// Format bytes into human-readable string.
fn human_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.0} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_bytes_formatting() {
        assert_eq!(human_bytes(0), "0 B");
        assert_eq!(human_bytes(512), "512 B");
        assert_eq!(human_bytes(1024), "1 KB");
        assert_eq!(human_bytes(1_500_000), "1.4 MB");
        assert_eq!(human_bytes(3_000_000_000), "2.8 GB");
    }

    #[test]
    fn list_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let config = LocalLlmConfig {
            models_dir: dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        let mgr = ModelManager::new(&config);
        let models = mgr.list_models().unwrap();
        assert!(models.is_empty());
    }

    #[test]
    fn list_models_finds_gguf() {
        let dir = tempfile::tempdir().unwrap();

        // Create fake .gguf files
        std::fs::write(dir.path().join("model-a.gguf"), b"fake gguf").unwrap();
        std::fs::write(dir.path().join("model-b.gguf"), b"another model").unwrap();
        std::fs::write(dir.path().join("readme.txt"), b"not a model").unwrap();

        let config = LocalLlmConfig {
            models_dir: dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        let mgr = ModelManager::new(&config);
        let models = mgr.list_models().unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].filename, "model-a.gguf");
        assert_eq!(models[1].filename, "model-b.gguf");
    }

    #[test]
    fn delete_model_works() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.gguf"), b"data").unwrap();

        let config = LocalLlmConfig {
            models_dir: dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        let mgr = ModelManager::new(&config);
        assert!(mgr.delete_model("test.gguf").is_ok());
        assert!(!dir.path().join("test.gguf").exists());
    }

    #[test]
    fn delete_non_gguf_rejected() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("secret.txt"), b"data").unwrap();

        let config = LocalLlmConfig {
            models_dir: dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        let mgr = ModelManager::new(&config);
        assert!(mgr.delete_model("secret.txt").is_err());
    }

    #[test]
    fn total_size_sums_correctly() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.gguf"), b"12345").unwrap();
        std::fs::write(dir.path().join("b.gguf"), b"123").unwrap();

        let config = LocalLlmConfig {
            models_dir: dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        let mgr = ModelManager::new(&config);
        assert_eq!(mgr.total_size().unwrap(), 8);
    }

    #[test]
    fn check_ram_returns_some() {
        // This should work on macOS/Linux CI
        let ram = ModelManager::check_available_ram();
        if cfg!(any(target_os = "macos", target_os = "linux")) {
            assert!(ram.is_some());
            assert!(ram.unwrap() > 0);
        }
    }

    #[test]
    fn nonexistent_dir_returns_empty() {
        let config = LocalLlmConfig {
            models_dir: "/tmp/helix-test-nonexistent-dir-12345".to_string(),
            ..Default::default()
        };
        let mgr = ModelManager::new(&config);
        let models = mgr.list_models().unwrap();
        assert!(models.is_empty());
    }
}
