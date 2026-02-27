//! Outbound webhook store — in-memory registry of webhooks and delivery records.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Maximum delivery records stored per webhook.
const MAX_DELIVERIES_PER_WEBHOOK: usize = 100;

/// A registered outbound webhook.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StoredWebhook {
    pub id: Uuid,
    pub url: String,
    pub events: Vec<String>,
    pub secret: Option<String>,
    pub active: bool,
    pub created_at: DateTime<Utc>,
    pub retry_max: u32,
    pub retry_backoff_secs: u64,
}

/// Record of a single webhook delivery attempt.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliveryRecord {
    pub id: Uuid,
    pub webhook_id: Uuid,
    pub event_type: String,
    pub status: String,
    pub response_code: Option<u16>,
    pub error: Option<String>,
    pub delivered_at: DateTime<Utc>,
}

/// In-memory store for outbound webhooks and their delivery history.
pub struct OutboundWebhookStore {
    webhooks: RwLock<HashMap<Uuid, StoredWebhook>>,
    deliveries: RwLock<HashMap<Uuid, VecDeque<DeliveryRecord>>>,
}

impl OutboundWebhookStore {
    pub fn new() -> Self {
        Self {
            webhooks: RwLock::new(HashMap::new()),
            deliveries: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new webhook. Returns the stored webhook.
    pub async fn register(&self, webhook: StoredWebhook) -> StoredWebhook {
        let id = webhook.id;
        let cloned = webhook.clone();
        self.webhooks.write().await.insert(id, webhook);
        cloned
    }

    /// Get a webhook by ID.
    pub async fn get(&self, id: Uuid) -> Option<StoredWebhook> {
        self.webhooks.read().await.get(&id).cloned()
    }

    /// List all webhooks.
    pub async fn list(&self) -> Vec<StoredWebhook> {
        self.webhooks.read().await.values().cloned().collect()
    }

    /// Remove a webhook and its delivery records. Returns true if it existed.
    pub async fn remove(&self, id: Uuid) -> bool {
        self.deliveries.write().await.remove(&id);
        self.webhooks.write().await.remove(&id).is_some()
    }

    /// Record a delivery attempt, capping at MAX_DELIVERIES_PER_WEBHOOK.
    pub async fn record_delivery(&self, record: DeliveryRecord) {
        let wh_id = record.webhook_id;
        let mut map = self.deliveries.write().await;
        let queue = map.entry(wh_id).or_insert_with(VecDeque::new);
        if queue.len() >= MAX_DELIVERIES_PER_WEBHOOK {
            queue.pop_front();
        }
        queue.push_back(record);
    }

    /// List delivery records for a webhook (oldest first).
    pub async fn list_deliveries(&self, webhook_id: Uuid) -> Vec<DeliveryRecord> {
        self.deliveries
            .read()
            .await
            .get(&webhook_id)
            .map(|q| q.iter().cloned().collect())
            .unwrap_or_default()
    }
}

impl Default for OutboundWebhookStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_webhook(url: &str) -> StoredWebhook {
        StoredWebhook {
            id: Uuid::now_v7(),
            url: url.to_string(),
            events: vec!["node.created".to_string()],
            secret: Some("test-secret".to_string()),
            active: true,
            created_at: Utc::now(),
            retry_max: 3,
            retry_backoff_secs: 10,
        }
    }

    fn make_delivery(webhook_id: Uuid, status: &str) -> DeliveryRecord {
        DeliveryRecord {
            id: Uuid::now_v7(),
            webhook_id,
            event_type: "node.created".to_string(),
            status: status.to_string(),
            response_code: Some(200),
            error: None,
            delivered_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn register_and_get() {
        let store = OutboundWebhookStore::new();
        let wh = make_webhook("https://example.com/hook");
        let id = wh.id;
        store.register(wh).await;

        let got = store.get(id).await;
        assert!(got.is_some());
        assert_eq!(got.unwrap().url, "https://example.com/hook");
    }

    #[tokio::test]
    async fn get_nonexistent() {
        let store = OutboundWebhookStore::new();
        assert!(store.get(Uuid::nil()).await.is_none());
    }

    #[tokio::test]
    async fn list_webhooks() {
        let store = OutboundWebhookStore::new();
        store.register(make_webhook("https://a.com")).await;
        store.register(make_webhook("https://b.com")).await;
        let all = store.list().await;
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn remove_webhook() {
        let store = OutboundWebhookStore::new();
        let wh = make_webhook("https://example.com/hook");
        let id = wh.id;
        store.register(wh).await;
        store
            .record_delivery(make_delivery(id, "delivered"))
            .await;

        assert!(store.remove(id).await);
        assert!(store.get(id).await.is_none());
        assert!(store.list_deliveries(id).await.is_empty());
        // Remove again returns false
        assert!(!store.remove(id).await);
    }

    #[tokio::test]
    async fn delivery_recording() {
        let store = OutboundWebhookStore::new();
        let wh = make_webhook("https://example.com");
        let id = wh.id;
        store.register(wh).await;

        store
            .record_delivery(make_delivery(id, "delivered"))
            .await;
        store.record_delivery(make_delivery(id, "failed")).await;

        let records = store.list_deliveries(id).await;
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].status, "delivered");
        assert_eq!(records[1].status, "failed");
    }

    #[tokio::test]
    async fn delivery_cap_enforcement() {
        let store = OutboundWebhookStore::new();
        let wh = make_webhook("https://example.com");
        let id = wh.id;
        store.register(wh).await;

        // Insert MAX + 5 records
        for i in 0..(MAX_DELIVERIES_PER_WEBHOOK + 5) {
            store
                .record_delivery(DeliveryRecord {
                    id: Uuid::now_v7(),
                    webhook_id: id,
                    event_type: format!("event_{i}"),
                    status: "delivered".to_string(),
                    response_code: Some(200),
                    error: None,
                    delivered_at: Utc::now(),
                })
                .await;
        }

        let records = store.list_deliveries(id).await;
        assert_eq!(records.len(), MAX_DELIVERIES_PER_WEBHOOK);
        // Oldest should have been evicted — first record should be event_5
        assert_eq!(records[0].event_type, "event_5");
    }

    #[tokio::test]
    async fn list_deliveries_nonexistent() {
        let store = OutboundWebhookStore::new();
        assert!(store.list_deliveries(Uuid::nil()).await.is_empty());
    }

    #[test]
    fn stored_webhook_serializes() {
        let wh = make_webhook("https://example.com");
        let json = serde_json::to_string(&wh).unwrap();
        assert!(json.contains("https://example.com"));
        assert!(json.contains("node.created"));
    }

    #[test]
    fn delivery_record_serializes() {
        let record = make_delivery(Uuid::nil(), "delivered");
        let json = serde_json::to_string(&record).unwrap();
        assert!(json.contains("delivered"));
    }

    #[test]
    fn default_impl() {
        let _store = OutboundWebhookStore::default();
    }
}
