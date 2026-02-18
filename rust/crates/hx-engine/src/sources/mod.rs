pub mod directory;
pub mod github;
pub mod rss;
pub mod url_scraper;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

use hx_core::MvResult;

// ---------------------------------------------------------------------------
// Source Type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceType {
    DirectoryWatch,
    RssFeed,
    GitHubIssues,
    UrlScraper,
    Custom,
}

impl SourceType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::DirectoryWatch => "directory_watch",
            Self::RssFeed => "rss_feed",
            Self::GitHubIssues => "github_issues",
            Self::UrlScraper => "url_scraper",
            Self::Custom => "custom",
        }
    }
}

impl std::str::FromStr for SourceType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "directory_watch" => Ok(Self::DirectoryWatch),
            "rss_feed" => Ok(Self::RssFeed),
            "github_issues" => Ok(Self::GitHubIssues),
            "url_scraper" => Ok(Self::UrlScraper),
            "custom" => Ok(Self::Custom),
            _ => Err(format!("unknown source type: {s}")),
        }
    }
}

impl std::fmt::Display for SourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Source Document
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceDocument {
    pub title: Option<String>,
    pub content: String,
    pub source_url: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub fetched_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Source Status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceStatus {
    pub connected: bool,
    pub last_poll: Option<DateTime<Utc>>,
    pub documents_fetched: u64,
    pub errors: u64,
    pub message: Option<String>,
}

// ---------------------------------------------------------------------------
// Source Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    pub id: Uuid,
    pub source_type: SourceType,
    pub name: String,
    pub enabled: bool,
    pub settings: HashMap<String, String>,
    pub poll_interval_secs: u64,
    pub created_at: DateTime<Utc>,
}

impl SourceConfig {
    pub fn new(source_type: SourceType, name: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            source_type,
            name: name.into(),
            enabled: true,
            settings: HashMap::new(),
            poll_interval_secs: 300,
            created_at: Utc::now(),
        }
    }

    pub fn with_setting(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.settings.insert(key.into(), value.into());
        self
    }

    pub fn get_setting(&self, key: &str) -> Option<&str> {
        self.settings.get(key).map(|s| s.as_str())
    }
}

// ---------------------------------------------------------------------------
// Source Connector Trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait SourceConnector: Send + Sync {
    fn name(&self) -> &str;
    fn source_type(&self) -> SourceType;
    async fn poll(&self) -> MvResult<Vec<SourceDocument>>;
    fn status(&self) -> SourceStatus;
    async fn health_check(&self) -> MvResult<bool>;
}

// ---------------------------------------------------------------------------
// Source Registry
// ---------------------------------------------------------------------------

pub struct SourceRegistry {
    sources: RwLock<HashMap<Uuid, (SourceConfig, Arc<dyn SourceConnector>)>>,
}

impl SourceRegistry {
    pub fn new() -> Self {
        Self {
            sources: RwLock::new(HashMap::new()),
        }
    }

    pub async fn register(&self, config: SourceConfig, connector: Arc<dyn SourceConnector>) {
        let id = config.id;
        self.sources.write().await.insert(id, (config, connector));
    }

    pub async fn remove(&self, id: Uuid) -> bool {
        self.sources.write().await.remove(&id).is_some()
    }

    pub async fn get(&self, id: Uuid) -> Option<Arc<dyn SourceConnector>> {
        self.sources.read().await.get(&id).map(|(_, c)| Arc::clone(c))
    }

    pub async fn list(&self) -> Vec<(Uuid, Arc<dyn SourceConnector>)> {
        self.sources
            .read()
            .await
            .iter()
            .map(|(&id, (_, c))| (id, Arc::clone(c)))
            .collect()
    }

    pub async fn list_configs(&self) -> Vec<SourceConfig> {
        self.sources
            .read()
            .await
            .values()
            .map(|(c, _)| c.clone())
            .collect()
    }

    pub async fn poll(&self, id: Uuid) -> MvResult<Vec<SourceDocument>> {
        let sources = self.sources.read().await;
        match sources.get(&id) {
            Some((_, connector)) => connector.poll().await,
            None => Err(hx_core::HxError::InvalidInput(format!(
                "source {id} not found"
            ))),
        }
    }

    pub async fn status(&self, id: Uuid) -> Option<SourceStatus> {
        self.sources
            .read()
            .await
            .get(&id)
            .map(|(_, c)| c.status())
    }

    pub async fn health_check(&self, id: Uuid) -> MvResult<bool> {
        let sources = self.sources.read().await;
        match sources.get(&id) {
            Some((_, connector)) => connector.health_check().await,
            None => Err(hx_core::HxError::InvalidInput(format!(
                "source {id} not found"
            ))),
        }
    }
}

impl Default for SourceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct MockConnector {
        name: String,
        poll_count: AtomicU64,
    }

    impl MockConnector {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                poll_count: AtomicU64::new(0),
            }
        }
    }

    #[async_trait]
    impl SourceConnector for MockConnector {
        fn name(&self) -> &str {
            &self.name
        }

        fn source_type(&self) -> SourceType {
            SourceType::Custom
        }

        async fn poll(&self) -> MvResult<Vec<SourceDocument>> {
            self.poll_count.fetch_add(1, Ordering::SeqCst);
            Ok(vec![SourceDocument {
                title: Some("test".into()),
                content: "test content".into(),
                source_url: None,
                metadata: HashMap::new(),
                fetched_at: Utc::now(),
            }])
        }

        fn status(&self) -> SourceStatus {
            SourceStatus {
                connected: true,
                last_poll: None,
                documents_fetched: self.poll_count.load(Ordering::SeqCst),
                errors: 0,
                message: None,
            }
        }

        async fn health_check(&self) -> MvResult<bool> {
            Ok(true)
        }
    }

    // --- SourceConfig tests ---

    #[test]
    fn source_config_new_sets_defaults() {
        let config = SourceConfig::new(SourceType::RssFeed, "my-feed");
        assert_eq!(config.source_type, SourceType::RssFeed);
        assert_eq!(config.name, "my-feed");
        assert!(config.enabled);
        assert!(config.settings.is_empty());
        assert_eq!(config.poll_interval_secs, 300);
    }

    #[test]
    fn source_config_with_setting_and_get() {
        let config = SourceConfig::new(SourceType::GitHubIssues, "test")
            .with_setting("repo", "owner/name")
            .with_setting("state", "open");

        assert_eq!(config.get_setting("repo"), Some("owner/name"));
        assert_eq!(config.get_setting("state"), Some("open"));
        assert_eq!(config.get_setting("nonexistent"), None);
    }

    #[test]
    fn source_config_serialization_roundtrip() {
        let config = SourceConfig::new(SourceType::DirectoryWatch, "watcher")
            .with_setting("path", "/tmp");

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: SourceConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "watcher");
        assert_eq!(deserialized.source_type, SourceType::DirectoryWatch);
        assert_eq!(deserialized.get_setting("path"), Some("/tmp"));
    }

    // --- SourceType tests ---

    #[test]
    fn source_type_display_and_from_str() {
        assert_eq!(SourceType::DirectoryWatch.to_string(), "directory_watch");
        assert_eq!(SourceType::RssFeed.to_string(), "rss_feed");
        assert_eq!(SourceType::GitHubIssues.to_string(), "github_issues");
        assert_eq!(SourceType::UrlScraper.to_string(), "url_scraper");
        assert_eq!(SourceType::Custom.to_string(), "custom");

        assert_eq!("directory_watch".parse::<SourceType>().unwrap(), SourceType::DirectoryWatch);
        assert_eq!("rss_feed".parse::<SourceType>().unwrap(), SourceType::RssFeed);
        assert_eq!("github_issues".parse::<SourceType>().unwrap(), SourceType::GitHubIssues);
        assert_eq!("url_scraper".parse::<SourceType>().unwrap(), SourceType::UrlScraper);
        assert_eq!("custom".parse::<SourceType>().unwrap(), SourceType::Custom);
    }

    #[test]
    fn source_type_from_str_rejects_unknown() {
        assert!("DirectoryWatch".parse::<SourceType>().is_err());
        assert!("".parse::<SourceType>().is_err());
    }

    #[test]
    fn source_type_as_str_matches_display() {
        for st in [
            SourceType::DirectoryWatch, SourceType::RssFeed, SourceType::GitHubIssues,
            SourceType::UrlScraper, SourceType::Custom,
        ] {
            assert_eq!(st.as_str(), st.to_string());
        }
    }

    // --- SourceDocument tests ---

    #[test]
    fn source_document_serialization_roundtrip() {
        let doc = SourceDocument {
            title: Some("Test Doc".into()),
            content: "some content".into(),
            source_url: Some("https://example.com".into()),
            metadata: HashMap::from([("key".into(), serde_json::json!("value"))]),
            fetched_at: Utc::now(),
        };

        let json = serde_json::to_string(&doc).unwrap();
        let deserialized: SourceDocument = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.title.as_deref(), Some("Test Doc"));
        assert_eq!(deserialized.content, "some content");
        assert_eq!(deserialized.source_url.as_deref(), Some("https://example.com"));
    }

    // --- SourceStatus tests ---

    #[test]
    fn source_status_serialization() {
        let status = SourceStatus {
            connected: true,
            last_poll: Some(Utc::now()),
            documents_fetched: 42,
            errors: 1,
            message: Some("ok".into()),
        };

        let json = serde_json::to_value(&status).unwrap();
        assert_eq!(json["connected"], true);
        assert_eq!(json["documents_fetched"], 42);
        assert_eq!(json["errors"], 1);
        assert_eq!(json["message"], "ok");
    }

    // --- Registry tests ---

    #[tokio::test]
    async fn register_and_poll() {
        let registry = SourceRegistry::new();
        let connector = Arc::new(MockConnector::new("test"));
        let config = SourceConfig::new(SourceType::Custom, "test-source");
        let id = config.id;

        registry.register(config, connector).await;

        let docs = registry.poll(id).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "test content");
    }

    #[tokio::test]
    async fn remove_source() {
        let registry = SourceRegistry::new();
        let connector = Arc::new(MockConnector::new("rm"));
        let config = SourceConfig::new(SourceType::Custom, "rm-source");
        let id = config.id;

        registry.register(config, connector).await;
        assert!(registry.get(id).await.is_some());

        assert!(registry.remove(id).await);
        assert!(registry.get(id).await.is_none());
    }

    #[tokio::test]
    async fn health_check_registered() {
        let registry = SourceRegistry::new();
        let connector = Arc::new(MockConnector::new("hc"));
        let config = SourceConfig::new(SourceType::Custom, "hc-source");
        let id = config.id;

        registry.register(config, connector).await;

        let healthy = registry.health_check(id).await.unwrap();
        assert!(healthy);
    }

    #[tokio::test]
    async fn list_configs_returns_all() {
        let registry = SourceRegistry::new();
        assert!(registry.list_configs().await.is_empty());

        let c1 = Arc::new(MockConnector::new("a"));
        let c2 = Arc::new(MockConnector::new("b"));

        registry
            .register(SourceConfig::new(SourceType::RssFeed, "feed-1"), c1)
            .await;
        registry
            .register(SourceConfig::new(SourceType::GitHubIssues, "gh-1"), c2)
            .await;

        let configs = registry.list_configs().await;
        assert_eq!(configs.len(), 2);
    }

    #[tokio::test]
    async fn poll_nonexistent_returns_error() {
        let registry = SourceRegistry::new();
        let result = registry.poll(Uuid::now_v7()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn health_check_nonexistent_returns_error() {
        let registry = SourceRegistry::new();
        let result = registry.health_check(Uuid::now_v7()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn remove_nonexistent_returns_false() {
        let registry = SourceRegistry::new();
        assert!(!registry.remove(Uuid::now_v7()).await);
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let registry = SourceRegistry::new();
        assert!(registry.get(Uuid::now_v7()).await.is_none());
    }

    #[tokio::test]
    async fn status_returns_connector_status() {
        let registry = SourceRegistry::new();
        let connector = Arc::new(MockConnector::new("stat"));
        let config = SourceConfig::new(SourceType::Custom, "stat-source");
        let id = config.id;

        registry.register(config, connector).await;

        let status = registry.status(id).await.unwrap();
        assert!(status.connected);
        assert_eq!(status.documents_fetched, 0);
    }

    #[tokio::test]
    async fn default_creates_empty_registry() {
        let registry = SourceRegistry::default();
        assert!(registry.list_configs().await.is_empty());
    }
}
