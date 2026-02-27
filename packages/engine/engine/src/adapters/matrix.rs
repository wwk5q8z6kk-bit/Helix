//! Matrix adapter — sends room messages, polls via sync endpoint.
//!
//! Configuration keys:
//! - `homeserver_url`: Matrix homeserver URL (required)
//! - `access_token`: Matrix access token (required)
//! - `room_id`: Default room ID (optional)

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use hx_core::{HxError, MvResult};

use super::{
    AdapterConfig, AdapterInboundMessage, AdapterOutboundMessage, AdapterStatus, AdapterType,
    ExternalAdapter,
};

#[derive(Debug)]
pub struct MatrixAdapter {
    config: AdapterConfig,
    client: reqwest::Client,
    last_send: Mutex<Option<DateTime<Utc>>>,
    last_receive: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

impl MatrixAdapter {
    pub fn new(config: AdapterConfig) -> MvResult<Self> {
        if config.get_setting("homeserver_url").is_none() {
            return Err(HxError::Config(
                "Matrix adapter requires 'homeserver_url' setting".into(),
            ));
        }
        if config.get_setting("access_token").is_none() {
            return Err(HxError::Config(
                "Matrix adapter requires 'access_token' setting".into(),
            ));
        }
        Ok(Self {
            config,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .map_err(|e| HxError::Internal(e.to_string()))?,
            last_send: Mutex::new(None),
            last_receive: Mutex::new(None),
            last_error: Mutex::new(None),
        })
    }

    fn homeserver(&self) -> &str {
        self.config
            .get_setting("homeserver_url")
            .unwrap_or("http://localhost:8008")
    }

    fn access_token(&self) -> &str {
        self.config
            .get_setting("access_token")
            .unwrap_or("")
    }
}

#[async_trait]
impl ExternalAdapter for MatrixAdapter {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::Matrix
    }

    async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
        let room_id = if message.channel.is_empty() {
            self.config
                .get_setting("room_id")
                .ok_or_else(|| {
                    HxError::Config("no room_id in message or adapter config".into())
                })?
                .to_string()
        } else {
            message.channel.clone()
        };

        let txn_id = Uuid::now_v7();
        let url = format!(
            "{}/_matrix/client/v3/rooms/{}/send/m.room.message/{}",
            self.homeserver(),
            room_id,
            txn_id,
        );

        let payload = serde_json::json!({
            "msgtype": "m.text",
            "body": message.content,
        });

        let resp = self
            .client
            .put(&url)
            .bearer_auth(self.access_token())
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("matrix send failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            let err = format!("matrix send returned {status}: {body}");
            *self.last_error.lock().unwrap() = Some(err.clone());
            return Err(HxError::Internal(err));
        }

        *self.last_send.lock().unwrap() = Some(Utc::now());
        *self.last_error.lock().unwrap() = None;
        Ok(())
    }

    async fn poll(&self, cursor: Option<&str>) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
        let mut url = format!(
            "{}/_matrix/client/v3/sync?timeout=0",
            self.homeserver(),
        );
        if let Some(since) = cursor {
            url.push_str(&format!("&since={since}"));
        }

        let resp = self
            .client
            .get(&url)
            .bearer_auth(self.access_token())
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("matrix poll failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(HxError::Internal(format!(
                "matrix sync returned {status}: {body}"
            )));
        }

        let body: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| HxError::Internal(format!("matrix poll parse failed: {e}")))?;

        let next_batch = body
            .get("next_batch")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();

        let mut messages = Vec::new();

        // Parse rooms.join.*.timeline.events[]
        if let Some(rooms) = body.get("rooms").and_then(|r| r.get("join")) {
            if let Some(rooms_obj) = rooms.as_object() {
                for (room_id, room_data) in rooms_obj {
                    let events = room_data
                        .get("timeline")
                        .and_then(|t| t.get("events"))
                        .and_then(|e| e.as_array());

                    if let Some(events) = events {
                        for event in events {
                            let event_type = event
                                .get("type")
                                .and_then(|t| t.as_str())
                                .unwrap_or("");

                            if event_type != "m.room.message" {
                                continue;
                            }

                            let event_id = event
                                .get("event_id")
                                .and_then(|e| e.as_str())
                                .unwrap_or("")
                                .to_string();

                            let sender = event
                                .get("sender")
                                .and_then(|s| s.as_str())
                                .unwrap_or("unknown")
                                .to_string();

                            let content = event
                                .get("content")
                                .and_then(|c| c.get("body"))
                                .and_then(|b| b.as_str())
                                .unwrap_or("")
                                .to_string();

                            let origin_ts = event
                                .get("origin_server_ts")
                                .and_then(|t| t.as_i64())
                                .and_then(|ms| DateTime::from_timestamp_millis(ms))
                                .unwrap_or_else(Utc::now);

                            // Matrix threads use m.relates_to with rel_type m.thread
                            let thread_id = event
                                .get("content")
                                .and_then(|c| c.get("m.relates_to"))
                                .and_then(|r| r.get("event_id"))
                                .and_then(|e| e.as_str())
                                .map(String::from);

                            if content.is_empty() {
                                continue;
                            }

                            messages.push(AdapterInboundMessage {
                                external_id: event_id,
                                channel: room_id.clone(),
                                sender,
                                content,
                                thread_id,
                                timestamp: origin_ts,
                                metadata: HashMap::new(),
                            });
                        }
                    }
                }
            }
        }

        if !messages.is_empty() {
            *self.last_receive.lock().unwrap() = Some(Utc::now());
        }

        let new_cursor = if next_batch.is_empty() {
            cursor.unwrap_or("").to_string()
        } else {
            next_batch
        };

        Ok((messages, new_cursor))
    }

    async fn health_check(&self) -> MvResult<bool> {
        let url = format!(
            "{}/_matrix/client/v3/account/whoami",
            self.homeserver(),
        );

        let resp = self
            .client
            .get(&url)
            .bearer_auth(self.access_token())
            .send()
            .await
            .map_err(|e| HxError::Internal(format!("matrix health check failed: {e}")))?;

        Ok(resp.status().is_success())
    }

    fn status(&self) -> AdapterStatus {
        let error = self.last_error.lock().unwrap().clone();
        let last_send = *self.last_send.lock().unwrap();
        let last_receive = *self.last_receive.lock().unwrap();
        AdapterStatus {
            adapter_type: AdapterType::Matrix,
            name: self.config.name.clone(),
            connected: error.is_none(),
            last_send,
            last_receive,
            error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix_config() -> AdapterConfig {
        AdapterConfig::new(AdapterType::Matrix, "test-matrix")
            .with_setting("homeserver_url", "https://matrix.example.org")
            .with_setting("access_token", "syt_test_token_xyz")
    }

    #[test]
    fn new_requires_homeserver_url() {
        let config = AdapterConfig::new(AdapterType::Matrix, "no-hs")
            .with_setting("access_token", "tok");
        let result = MatrixAdapter::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("homeserver_url"),
            "expected homeserver_url error, got: {err}"
        );
    }

    #[test]
    fn new_requires_access_token() {
        let config = AdapterConfig::new(AdapterType::Matrix, "no-token")
            .with_setting("homeserver_url", "https://matrix.example.org");
        let result = MatrixAdapter::new(config);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("access_token"),
            "expected access_token error, got: {err}"
        );
    }

    #[tokio::test]
    async fn new_succeeds_with_required_settings() {
        let adapter = MatrixAdapter::new(matrix_config());
        assert!(adapter.is_ok());
        assert_eq!(adapter.unwrap().name(), "test-matrix");
    }

    #[tokio::test]
    async fn adapter_type_is_matrix() {
        let adapter = MatrixAdapter::new(matrix_config()).unwrap();
        assert_eq!(adapter.adapter_type(), AdapterType::Matrix);
    }

    #[tokio::test]
    async fn initial_status_connected() {
        let adapter = MatrixAdapter::new(matrix_config()).unwrap();
        let status = adapter.status();
        assert!(status.connected);
        assert!(status.last_send.is_none());
        assert!(status.last_receive.is_none());
        assert!(status.error.is_none());
        assert_eq!(status.adapter_type, AdapterType::Matrix);
    }

    #[tokio::test]
    async fn send_without_room_id_fails() {
        let config = AdapterConfig::new(AdapterType::Matrix, "test-matrix")
            .with_setting("homeserver_url", "https://matrix.example.org")
            .with_setting("access_token", "tok");
        let adapter = MatrixAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: String::new(),
            content: "hello".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        let result = adapter.send(&msg).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("room_id"));
    }

    #[tokio::test]
    async fn send_success_with_mock() {
        let mut server = mockito::Server::new_async().await;
        let mock = server
            .mock("PUT", mockito::Matcher::Regex(
                r"/_matrix/client/v3/rooms/!room:example\.org/send/m\.room\.message/.*".into(),
            ))
            .with_status(200)
            .with_body(r#"{"event_id":"$abc123"}"#)
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Matrix, "test-matrix")
            .with_setting("homeserver_url", &server.url())
            .with_setting("access_token", "tok");
        let adapter = MatrixAdapter::new(config).unwrap();
        let msg = AdapterOutboundMessage {
            channel: "!room:example.org".into(),
            content: "hello matrix".into(),
            thread_id: None,
            metadata: HashMap::new(),
        };
        adapter.send(&msg).await.unwrap();
        mock.assert_async().await;
        let status = adapter.status();
        assert!(status.last_send.is_some());
    }

    #[tokio::test]
    async fn health_check_success() {
        let mut server = mockito::Server::new_async().await;
        server
            .mock("GET", "/_matrix/client/v3/account/whoami")
            .with_status(200)
            .with_body(r#"{"user_id":"@bot:example.org"}"#)
            .create_async()
            .await;

        let config = AdapterConfig::new(AdapterType::Matrix, "test-matrix")
            .with_setting("homeserver_url", &server.url())
            .with_setting("access_token", "tok");
        let adapter = MatrixAdapter::new(config).unwrap();
        let healthy = adapter.health_check().await.unwrap();
        assert!(healthy);
    }
}
