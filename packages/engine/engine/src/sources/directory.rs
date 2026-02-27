use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use walkdir::WalkDir;

use hx_core::MvResult;

use super::{SourceConfig, SourceConnector, SourceDocument, SourceStatus, SourceType};

pub struct DirectoryWatcher {
    config: SourceConfig,
    path: PathBuf,
    recursive: bool,
    patterns: Vec<String>,
    ignore_hidden: bool,
    connected: AtomicBool,
    documents_fetched: AtomicU64,
    errors: AtomicU64,
    last_poll: Mutex<Option<DateTime<Utc>>>,
}

impl DirectoryWatcher {
    pub fn new(config: SourceConfig) -> MvResult<Self> {
        let path = config
            .get_setting("path")
            .ok_or_else(|| hx_core::HxError::InvalidInput("missing 'path' setting".into()))?;
        let path = PathBuf::from(path);

        let recursive = config
            .get_setting("recursive")
            .map(|v| v == "true")
            .unwrap_or(true);

        let patterns: Vec<String> = config
            .get_setting("patterns")
            .map(|p| p.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default();

        let ignore_hidden = config
            .get_setting("ignore_hidden")
            .map(|v| v == "true")
            .unwrap_or(true);

        Ok(Self {
            config,
            path,
            recursive,
            patterns,
            ignore_hidden,
            connected: AtomicBool::new(false),
            documents_fetched: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            last_poll: Mutex::new(None),
        })
    }

    fn matches_pattern(&self, filename: &str) -> bool {
        if self.patterns.is_empty() {
            return true;
        }
        for pattern in &self.patterns {
            if pattern.starts_with("*.") {
                let ext = &pattern[1..]; // e.g. ".txt"
                if filename.ends_with(ext) {
                    return true;
                }
            } else if pattern.contains('*') {
                // Simple wildcard: just check contains for the non-star part
                let parts: Vec<&str> = pattern.split('*').collect();
                let mut matched = true;
                let mut remaining = filename;
                for part in &parts {
                    if part.is_empty() {
                        continue;
                    }
                    if let Some(idx) = remaining.find(part) {
                        remaining = &remaining[idx + part.len()..];
                    } else {
                        matched = false;
                        break;
                    }
                }
                if matched {
                    return true;
                }
            } else if filename == pattern {
                return true;
            }
        }
        false
    }
}

#[async_trait]
impl SourceConnector for DirectoryWatcher {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn source_type(&self) -> SourceType {
        SourceType::DirectoryWatch
    }

    async fn poll(&self) -> MvResult<Vec<SourceDocument>> {
        let max_depth = if self.recursive { usize::MAX } else { 1 };
        let walker = WalkDir::new(&self.path).max_depth(max_depth);

        let mut documents = Vec::new();
        let now = Utc::now();

        for entry in walker.into_iter().filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }

            let file_name = entry.file_name().to_string_lossy();

            if self.ignore_hidden && file_name.starts_with('.') {
                continue;
            }

            if !self.matches_pattern(&file_name) {
                continue;
            }

            match std::fs::read_to_string(entry.path()) {
                Ok(content) => {
                    let mut metadata = HashMap::new();
                    metadata.insert(
                        "file_path".into(),
                        serde_json::Value::String(entry.path().to_string_lossy().into_owned()),
                    );
                    if let Ok(md) = entry.metadata() {
                        if let Ok(modified) = md.modified() {
                            let dt: DateTime<Utc> = modified.into();
                            metadata.insert(
                                "modified_at".into(),
                                serde_json::Value::String(dt.to_rfc3339()),
                            );
                        }
                        metadata.insert(
                            "size_bytes".into(),
                            serde_json::Value::Number(serde_json::Number::from(md.len())),
                        );
                    }

                    documents.push(SourceDocument {
                        title: Some(file_name.into_owned()),
                        content,
                        source_url: Some(format!("file://{}", entry.path().display())),
                        metadata,
                        fetched_at: now,
                    });
                }
                Err(_) => {
                    self.errors.fetch_add(1, Ordering::SeqCst);
                }
            }
        }

        self.documents_fetched
            .fetch_add(documents.len() as u64, Ordering::SeqCst);
        self.connected.store(true, Ordering::SeqCst);
        *self.last_poll.lock().unwrap() = Some(now);

        Ok(documents)
    }

    fn status(&self) -> SourceStatus {
        SourceStatus {
            connected: self.connected.load(Ordering::SeqCst),
            last_poll: *self.last_poll.lock().unwrap(),
            documents_fetched: self.documents_fetched.load(Ordering::SeqCst),
            errors: self.errors.load(Ordering::SeqCst),
            message: None,
        }
    }

    async fn health_check(&self) -> MvResult<bool> {
        Ok(self.path.is_dir())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_config(path: &str) -> SourceConfig {
        SourceConfig::new(SourceType::DirectoryWatch, "test-dir")
            .with_setting("path", path)
    }

    #[test]
    fn new_requires_path_setting() {
        let config = SourceConfig::new(SourceType::DirectoryWatch, "no-path");
        let result = DirectoryWatcher::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn new_with_valid_config() {
        let config = make_config("/tmp");
        let watcher = DirectoryWatcher::new(config).unwrap();
        assert_eq!(watcher.name(), "test-dir");
        assert_eq!(watcher.source_type(), SourceType::DirectoryWatch);
        assert!(watcher.recursive);
        assert!(watcher.ignore_hidden);
    }

    #[test]
    fn pattern_matching_extension() {
        let config = make_config("/tmp").with_setting("patterns", "*.txt,*.md");
        let watcher = DirectoryWatcher::new(config).unwrap();
        assert!(watcher.matches_pattern("readme.txt"));
        assert!(watcher.matches_pattern("doc.md"));
        assert!(!watcher.matches_pattern("image.png"));
    }

    #[test]
    fn pattern_matching_exact() {
        let config = make_config("/tmp").with_setting("patterns", "Makefile");
        let watcher = DirectoryWatcher::new(config).unwrap();
        assert!(watcher.matches_pattern("Makefile"));
        assert!(!watcher.matches_pattern("makefile"));
    }

    #[test]
    fn empty_patterns_match_all() {
        let config = make_config("/tmp");
        let watcher = DirectoryWatcher::new(config).unwrap();
        assert!(watcher.matches_pattern("anything.xyz"));
    }

    #[tokio::test]
    async fn poll_reads_files_from_temp_dir() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "hello world").unwrap();
        std::fs::write(dir.path().join("other.md"), "markdown").unwrap();

        let config = make_config(dir.path().to_str().unwrap())
            .with_setting("patterns", "*.txt");
        let watcher = DirectoryWatcher::new(config).unwrap();

        let docs = watcher.poll().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "hello world");
        assert_eq!(docs[0].title.as_deref(), Some("test.txt"));
    }

    #[tokio::test]
    async fn poll_ignores_hidden_files() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(".hidden"), "secret").unwrap();
        std::fs::write(dir.path().join("visible.txt"), "public").unwrap();

        let config = make_config(dir.path().to_str().unwrap());
        let watcher = DirectoryWatcher::new(config).unwrap();

        let docs = watcher.poll().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title.as_deref(), Some("visible.txt"));
    }

    #[tokio::test]
    async fn poll_non_recursive() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("top.txt"), "top").unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(sub.join("nested.txt"), "nested").unwrap();

        let config = make_config(dir.path().to_str().unwrap())
            .with_setting("recursive", "false");
        let watcher = DirectoryWatcher::new(config).unwrap();

        let docs = watcher.poll().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title.as_deref(), Some("top.txt"));
    }

    #[tokio::test]
    async fn health_check_valid_directory() {
        let dir = TempDir::new().unwrap();
        let config = make_config(dir.path().to_str().unwrap());
        let watcher = DirectoryWatcher::new(config).unwrap();
        assert!(watcher.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn health_check_invalid_directory() {
        let config = make_config("/nonexistent/path/12345");
        let watcher = DirectoryWatcher::new(config).unwrap();
        assert!(!watcher.health_check().await.unwrap());
    }

    #[tokio::test]
    async fn status_tracks_poll_count() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "a").unwrap();
        std::fs::write(dir.path().join("b.txt"), "b").unwrap();

        let config = make_config(dir.path().to_str().unwrap());
        let watcher = DirectoryWatcher::new(config).unwrap();

        assert!(!watcher.status().connected);
        assert_eq!(watcher.status().documents_fetched, 0);

        watcher.poll().await.unwrap();

        assert!(watcher.status().connected);
        assert_eq!(watcher.status().documents_fetched, 2);
        assert!(watcher.status().last_poll.is_some());
    }
}
