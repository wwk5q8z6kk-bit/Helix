use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};

use hx_core::MvResult;

use super::{SourceConfig, SourceConnector, SourceDocument, SourceStatus, SourceType};

pub struct RssFeedConnector {
    config: SourceConfig,
    feed_url: String,
    max_items: usize,
    client: reqwest::Client,
    connected: AtomicBool,
    documents_fetched: AtomicU64,
    errors: AtomicU64,
    last_poll: Mutex<Option<DateTime<Utc>>>,
}

impl RssFeedConnector {
    pub fn new(config: SourceConfig) -> MvResult<Self> {
        let feed_url = config
            .get_setting("feed_url")
            .ok_or_else(|| hx_core::HxError::InvalidInput("missing 'feed_url' setting".into()))?
            .to_string();

        let max_items = config
            .get_setting("max_items")
            .and_then(|v| v.parse().ok())
            .unwrap_or(50);

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .map_err(|e| hx_core::HxError::Internal(format!("http client error: {e}")))?;

        Ok(Self {
            config,
            feed_url,
            max_items,
            client,
            connected: AtomicBool::new(false),
            documents_fetched: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            last_poll: Mutex::new(None),
        })
    }

    fn parse_items(xml: &str, max_items: usize) -> Vec<RssItem> {
        let mut items = Vec::new();

        // Try RSS 2.0 <item> tags first, then Atom <entry> tags
        let tag_pairs = [("item", "item"), ("entry", "entry")];
        for (open_tag, close_tag) in &tag_pairs {
            let open = format!("<{}", open_tag);
            let close = format!("</{}>", close_tag);
            let mut search_from = 0;

            while items.len() < max_items {
                let start = match xml[search_from..].find(&open) {
                    Some(pos) => search_from + pos,
                    None => break,
                };
                let end = match xml[start..].find(&close) {
                    Some(pos) => start + pos + close.len(),
                    None => break,
                };

                let item_xml = &xml[start..end];

                let title = Self::extract_tag(item_xml, "title");
                let link = Self::extract_tag(item_xml, "link")
                    .or_else(|| Self::extract_attr(item_xml, "link", "href"));
                let description = Self::extract_tag(item_xml, "description")
                    .or_else(|| Self::extract_tag(item_xml, "summary"))
                    .or_else(|| Self::extract_tag(item_xml, "content"));
                let pub_date = Self::extract_tag(item_xml, "pubDate")
                    .or_else(|| Self::extract_tag(item_xml, "published"))
                    .or_else(|| Self::extract_tag(item_xml, "updated"));

                items.push(RssItem {
                    title,
                    link,
                    description,
                    pub_date,
                });

                search_from = end;
            }

            if !items.is_empty() {
                break;
            }
        }

        items
    }

    fn extract_tag(xml: &str, tag: &str) -> Option<String> {
        let open = format!("<{}", tag);
        let close = format!("</{}>", tag);

        let start_pos = xml.find(&open)?;
        let after_open = &xml[start_pos + open.len()..];
        // Skip past the > of the opening tag (handles attributes)
        let content_start = after_open.find('>')? + 1;
        let content = &after_open[content_start..];
        let end_pos = content.find(&close)?;

        let text = &content[..end_pos];
        // Strip CDATA
        let text = text
            .trim()
            .strip_prefix("<![CDATA[")
            .and_then(|s| s.strip_suffix("]]>"))
            .unwrap_or(text.trim());

        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    }

    fn extract_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
        let open = format!("<{}", tag);
        let start_pos = xml.find(&open)?;
        let after_open = &xml[start_pos + open.len()..];
        let tag_end = after_open.find('>')?;
        let tag_content = &after_open[..tag_end];

        let attr_prefix = format!("{}=\"", attr);
        let attr_start = tag_content.find(&attr_prefix)?;
        let value_start = attr_start + attr_prefix.len();
        let value_end = tag_content[value_start..].find('"')?;
        Some(tag_content[value_start..value_start + value_end].to_string())
    }

    fn strip_html_tags(html: &str) -> String {
        let mut result = String::with_capacity(html.len());
        let mut in_tag = false;
        for ch in html.chars() {
            match ch {
                '<' => in_tag = true,
                '>' => in_tag = false,
                _ if !in_tag => result.push(ch),
                _ => {}
            }
        }
        // Decode common HTML entities
        result
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
            .replace("&#39;", "'")
            .replace("&apos;", "'")
    }
}

struct RssItem {
    title: Option<String>,
    link: Option<String>,
    description: Option<String>,
    pub_date: Option<String>,
}

#[async_trait]
impl SourceConnector for RssFeedConnector {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn source_type(&self) -> SourceType {
        SourceType::RssFeed
    }

    async fn poll(&self) -> MvResult<Vec<SourceDocument>> {
        let response = self
            .client
            .get(&self.feed_url)
            .send()
            .await
            .map_err(|e| hx_core::HxError::Internal(format!("rss fetch error: {e}")))?;

        if !response.status().is_success() {
            self.errors.fetch_add(1, Ordering::SeqCst);
            return Err(hx_core::HxError::Internal(format!(
                "rss fetch returned status {}",
                response.status()
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| hx_core::HxError::Internal(format!("rss body read error: {e}")))?;

        let items = Self::parse_items(&body, self.max_items);
        let now = Utc::now();

        let documents: Vec<SourceDocument> = items
            .into_iter()
            .map(|item| {
                let content = item
                    .description
                    .as_deref()
                    .map(Self::strip_html_tags)
                    .unwrap_or_default();

                let mut metadata = HashMap::new();
                if let Some(ref date) = item.pub_date {
                    metadata.insert("pub_date".into(), serde_json::Value::String(date.clone()));
                }

                SourceDocument {
                    title: item.title,
                    content,
                    source_url: item.link,
                    metadata,
                    fetched_at: now,
                }
            })
            .collect();

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
        let resp = self
            .client
            .head(&self.feed_url)
            .send()
            .await
            .map_err(|e| hx_core::HxError::Internal(format!("rss health check error: {e}")))?;
        Ok(resp.status().is_success())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(url: &str) -> SourceConfig {
        SourceConfig::new(SourceType::RssFeed, "test-rss")
            .with_setting("feed_url", url)
    }

    #[test]
    fn new_requires_feed_url() {
        let config = SourceConfig::new(SourceType::RssFeed, "no-url");
        let result = RssFeedConnector::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn new_with_valid_config() {
        let config = make_config("https://example.com/feed.xml");
        let connector = RssFeedConnector::new(config).unwrap();
        assert_eq!(connector.name(), "test-rss");
        assert_eq!(connector.source_type(), SourceType::RssFeed);
        assert_eq!(connector.max_items, 50);
    }

    #[test]
    fn new_custom_max_items() {
        let config = make_config("https://example.com/feed.xml")
            .with_setting("max_items", "10");
        let connector = RssFeedConnector::new(config).unwrap();
        assert_eq!(connector.max_items, 10);
    }

    #[test]
    fn parse_rss2_items() {
        let xml = r#"<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>First Post</title>
      <link>https://example.com/1</link>
      <description>First description</description>
      <pubDate>Mon, 01 Jan 2024 00:00:00 GMT</pubDate>
    </item>
    <item>
      <title>Second Post</title>
      <link>https://example.com/2</link>
      <description><![CDATA[Second <b>description</b>]]></description>
    </item>
  </channel>
</rss>"#;

        let items = RssFeedConnector::parse_items(xml, 50);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].title.as_deref(), Some("First Post"));
        assert_eq!(items[0].link.as_deref(), Some("https://example.com/1"));
        assert_eq!(items[0].description.as_deref(), Some("First description"));
        assert!(items[0].pub_date.is_some());
        assert_eq!(items[1].title.as_deref(), Some("Second Post"));
        assert_eq!(items[1].description.as_deref(), Some("Second <b>description</b>"));
    }

    #[test]
    fn parse_atom_entries() {
        let xml = r#"<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Atom Post</title>
    <link href="https://example.com/atom/1"/>
    <summary>Atom summary</summary>
    <updated>2024-01-01T00:00:00Z</updated>
  </entry>
</feed>"#;

        let items = RssFeedConnector::parse_items(xml, 50);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].title.as_deref(), Some("Atom Post"));
        assert_eq!(items[0].link.as_deref(), Some("https://example.com/atom/1"));
        assert_eq!(items[0].description.as_deref(), Some("Atom summary"));
    }

    #[test]
    fn parse_items_respects_max() {
        let xml = r#"<rss><channel>
            <item><title>A</title><description>a</description></item>
            <item><title>B</title><description>b</description></item>
            <item><title>C</title><description>c</description></item>
        </channel></rss>"#;

        let items = RssFeedConnector::parse_items(xml, 2);
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn strip_html_tags_basic() {
        assert_eq!(
            RssFeedConnector::strip_html_tags("<p>Hello <b>world</b></p>"),
            "Hello world"
        );
    }

    #[test]
    fn strip_html_entities() {
        assert_eq!(
            RssFeedConnector::strip_html_tags("A &amp; B &lt; C &gt; D"),
            "A & B < C > D"
        );
    }

    #[test]
    fn initial_status_is_disconnected() {
        let config = make_config("https://example.com/feed.xml");
        let connector = RssFeedConnector::new(config).unwrap();
        let status = connector.status();
        assert!(!status.connected);
        assert_eq!(status.documents_fetched, 0);
        assert!(status.last_poll.is_none());
    }

    #[tokio::test]
    async fn poll_with_mock_server() {
        let mut server = mockito::Server::new_async().await;
        let rss_body = r#"<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item>
    <title>Mock Item</title>
    <link>https://example.com/mock</link>
    <description>Mock description</description>
  </item>
</channel></rss>"#;

        let mock = server
            .mock("GET", "/feed.xml")
            .with_status(200)
            .with_header("content-type", "application/xml")
            .with_body(rss_body)
            .create_async()
            .await;

        let url = format!("{}/feed.xml", server.url());
        let config = make_config(&url);
        let connector = RssFeedConnector::new(config).unwrap();

        let docs = connector.poll().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].title.as_deref(), Some("Mock Item"));
        assert_eq!(docs[0].content, "Mock description");

        assert!(connector.status().connected);
        assert_eq!(connector.status().documents_fetched, 1);

        mock.assert_async().await;
    }

    #[tokio::test]
    async fn health_check_with_mock() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("HEAD", "/feed.xml")
            .with_status(200)
            .create_async()
            .await;

        let url = format!("{}/feed.xml", server.url());
        let config = make_config(&url);
        let connector = RssFeedConnector::new(config).unwrap();

        assert!(connector.health_check().await.unwrap());
        mock.assert_async().await;
    }
}
