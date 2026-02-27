use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use hx_core::MvResult;

use super::{SourceConfig, SourceConnector, SourceDocument, SourceStatus, SourceType};

pub struct UrlScraperConnector {
    config: SourceConfig,
    urls: Vec<String>,
    selector: Option<String>,
    client: reqwest::Client,
    connected: AtomicBool,
    documents_fetched: AtomicU64,
    errors: AtomicU64,
    last_poll: Mutex<Option<DateTime<Utc>>>,
}

impl UrlScraperConnector {
    pub fn new(config: SourceConfig) -> MvResult<Self> {
        let urls_str = config
            .get_setting("urls")
            .ok_or_else(|| hx_core::HxError::InvalidInput("missing 'urls' setting".into()))?;

        let urls: Vec<String> = urls_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if urls.is_empty() {
            return Err(hx_core::HxError::InvalidInput("'urls' setting is empty".into()));
        }

        let selector = config.get_setting("selector").map(|s| s.to_string());

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| hx_core::HxError::Internal(format!("http client error: {e}")))?;

        Ok(Self {
            config,
            urls,
            selector,
            client,
            connected: AtomicBool::new(false),
            documents_fetched: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            last_poll: Mutex::new(None),
        })
    }

    fn strip_html(html: &str) -> String {
        let mut result = String::with_capacity(html.len());
        let mut in_tag = false;
        let mut in_script = false;
        let mut in_style = false;
        let lower = html.to_lowercase();
        let chars: Vec<char> = html.chars().collect();
        let lower_chars: Vec<char> = lower.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if !in_tag && chars[i] == '<' {
                in_tag = true;
                // Check for script/style tags
                let remaining: String = lower_chars[i..].iter().collect();
                if remaining.starts_with("<script") {
                    in_script = true;
                } else if remaining.starts_with("<style") {
                    in_style = true;
                } else if remaining.starts_with("</script") {
                    in_script = false;
                } else if remaining.starts_with("</style") {
                    in_style = false;
                }
            } else if in_tag && chars[i] == '>' {
                in_tag = false;
            } else if !in_tag && !in_script && !in_style {
                result.push(chars[i]);
            }
            i += 1;
        }

        // Decode common HTML entities
        let result = result
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'")
            .replace("&nbsp;", " ");

        // Collapse whitespace
        let mut collapsed = String::with_capacity(result.len());
        let mut prev_ws = false;
        for ch in result.chars() {
            if ch.is_whitespace() {
                if !prev_ws {
                    collapsed.push(' ');
                }
                prev_ws = true;
            } else {
                collapsed.push(ch);
                prev_ws = false;
            }
        }

        collapsed.trim().to_string()
    }

    fn extract_title(html: &str) -> Option<String> {
        let lower = html.to_lowercase();
        let start = lower.find("<title")?;
        let after = &html[start..];
        let gt = after.find('>')?;
        let content = &after[gt + 1..];
        let end = content.to_lowercase().find("</title>")?;
        let title = content[..end].trim();
        if title.is_empty() {
            None
        } else {
            Some(title.to_string())
        }
    }
}

#[async_trait]
impl SourceConnector for UrlScraperConnector {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn source_type(&self) -> SourceType {
        SourceType::UrlScraper
    }

    async fn poll(&self) -> MvResult<Vec<SourceDocument>> {
        let now = Utc::now();
        let mut documents = Vec::new();

        for url in &self.urls {
            match self.client.get(url).send().await {
                Ok(response) => {
                    if !response.status().is_success() {
                        self.errors.fetch_add(1, Ordering::SeqCst);
                        continue;
                    }

                    match response.text().await {
                        Ok(html) => {
                            let title = Self::extract_title(&html);
                            let content = Self::strip_html(&html);

                            let mut metadata = HashMap::new();
                            metadata.insert(
                                "content_length".into(),
                                serde_json::json!(content.len()),
                            );
                            if let Some(ref sel) = self.selector {
                                metadata.insert(
                                    "selector".into(),
                                    serde_json::Value::String(sel.clone()),
                                );
                            }

                            documents.push(SourceDocument {
                                title,
                                content,
                                source_url: Some(url.clone()),
                                metadata,
                                fetched_at: now,
                            });
                        }
                        Err(_) => {
                            self.errors.fetch_add(1, Ordering::SeqCst);
                        }
                    }
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
        if let Some(first_url) = self.urls.first() {
            let resp = self
                .client
                .head(first_url)
                .send()
                .await
                .map_err(|e| hx_core::HxError::Internal(format!("url scraper health check error: {e}")))?;
            Ok(resp.status().is_success())
        } else {
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(urls: &str) -> SourceConfig {
        SourceConfig::new(SourceType::UrlScraper, "test-scraper")
            .with_setting("urls", urls)
    }

    #[test]
    fn new_requires_urls() {
        let config = SourceConfig::new(SourceType::UrlScraper, "no-urls");
        let result = UrlScraperConnector::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn new_rejects_empty_urls() {
        let config = make_config("");
        let result = UrlScraperConnector::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn new_with_valid_config() {
        let config = make_config("https://example.com,https://example.org");
        let connector = UrlScraperConnector::new(config).unwrap();
        assert_eq!(connector.name(), "test-scraper");
        assert_eq!(connector.source_type(), SourceType::UrlScraper);
        assert_eq!(connector.urls.len(), 2);
    }

    #[test]
    fn new_with_selector() {
        let config = make_config("https://example.com")
            .with_setting("selector", "article.content");
        let connector = UrlScraperConnector::new(config).unwrap();
        assert_eq!(connector.selector.as_deref(), Some("article.content"));
    }

    #[test]
    fn strip_html_basic() {
        assert_eq!(
            UrlScraperConnector::strip_html("<p>Hello <b>world</b></p>"),
            "Hello world"
        );
    }

    #[test]
    fn strip_html_removes_script_and_style() {
        let html = "<html><head><style>body{}</style></head><body><script>alert(1)</script><p>Content</p></body></html>";
        let text = UrlScraperConnector::strip_html(html);
        assert!(text.contains("Content"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("body{}"));
    }

    #[test]
    fn strip_html_entities() {
        assert_eq!(
            UrlScraperConnector::strip_html("A&amp;B &lt;C&gt; &nbsp;D"),
            "A&B <C> D"
        );
    }

    #[test]
    fn extract_title_basic() {
        let html = "<html><head><title>My Page</title></head><body></body></html>";
        assert_eq!(UrlScraperConnector::extract_title(html), Some("My Page".into()));
    }

    #[test]
    fn extract_title_missing() {
        assert_eq!(UrlScraperConnector::extract_title("<html><body></body></html>"), None);
    }

    #[test]
    fn initial_status() {
        let config = make_config("https://example.com");
        let connector = UrlScraperConnector::new(config).unwrap();
        let status = connector.status();
        assert!(!status.connected);
        assert_eq!(status.documents_fetched, 0);
    }

    #[tokio::test]
    async fn poll_with_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let html = "<html><head><title>Test Page</title></head><body><p>Hello World</p></body></html>";

        let mock = server
            .mock("GET", "/page")
            .with_status(200)
            .with_header("content-type", "text/html")
            .with_body(html)
            .create_async()
            .await;

        let url = format!("{}/page", server.url());
        let config = make_config(&url);
        let connector = UrlScraperConnector::new(config).unwrap();

        let docs = connector.poll().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title.as_deref(), Some("Test Page"));
        assert!(docs[0].content.contains("Hello World"));
        assert!(docs[0].source_url.as_deref() == Some(url.as_str()));

        assert!(connector.status().connected);
        assert_eq!(connector.status().documents_fetched, 1);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_check_with_mock() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("HEAD", "/page")
            .with_status(200)
            .create_async()
            .await;

        let url = format!("{}/page", server.url());
        let config = make_config(&url);
        let connector = UrlScraperConnector::new(config).unwrap();

        assert!(connector.health_check().await.unwrap());
        mock.assert_async().await;
    }
}
