use async_trait::async_trait;
use hx_core::MvResult;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::notifications::{Notification, NotificationChannel, NotificationChannelType, Severity};

/// In-app notification channel that stores notifications in memory.
pub struct InAppChannel {
    notifications: Arc<RwLock<Vec<Notification>>>,
    max_stored: usize,
}

impl InAppChannel {
    pub fn new() -> Self {
        Self {
            notifications: Arc::new(RwLock::new(Vec::new())),
            max_stored: 1000,
        }
    }

    pub fn with_max_stored(mut self, max: usize) -> Self {
        self.max_stored = max;
        self
    }

    pub async fn list(
        &self,
        severity: Option<Severity>,
        read: Option<bool>,
        limit: usize,
    ) -> Vec<Notification> {
        let store = self.notifications.read().await;
        store
            .iter()
            .rev()
            .filter(|n| severity.map_or(true, |s| n.severity == s))
            .filter(|n| read.map_or(true, |r| n.read == r))
            .take(limit)
            .cloned()
            .collect()
    }

    pub async fn get(&self, id: Uuid) -> Option<Notification> {
        let store = self.notifications.read().await;
        store.iter().find(|n| n.id == id).cloned()
    }

    pub async fn mark_read(&self, id: Uuid) -> bool {
        let mut store = self.notifications.write().await;
        if let Some(notif) = store.iter_mut().find(|n| n.id == id) {
            notif.mark_read();
            true
        } else {
            false
        }
    }

    pub async fn count(&self) -> usize {
        self.notifications.read().await.len()
    }

    pub async fn count_unread(&self) -> usize {
        self.notifications.read().await.iter().filter(|n| !n.read).count()
    }
}

#[async_trait]
impl NotificationChannel for InAppChannel {
    fn name(&self) -> &str {
        "in-app"
    }

    fn channel_type(&self) -> NotificationChannelType {
        NotificationChannelType::InApp
    }

    async fn send(&self, notification: &Notification) -> MvResult<()> {
        let mut store = self.notifications.write().await;
        store.push(notification.clone());

        // Evict oldest when over capacity
        while store.len() > self.max_stored {
            store.remove(0);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn send_and_list_notifications() {
        let channel = InAppChannel::new();
        let notif = Notification::new("Test", "Body", Severity::Info);
        channel.send(&notif).await.unwrap();

        let listed = channel.list(None, None, 50).await;
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].title, "Test");
    }

    #[tokio::test]
    async fn list_filters_by_severity() {
        let channel = InAppChannel::new();
        channel.send(&Notification::new("Info", "b", Severity::Info)).await.unwrap();
        channel.send(&Notification::new("Error", "b", Severity::Error)).await.unwrap();

        let infos = channel.list(Some(Severity::Info), None, 50).await;
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].title, "Info");
    }

    #[tokio::test]
    async fn list_filters_by_read_status() {
        let channel = InAppChannel::new();
        let notif = Notification::new("Unread", "b", Severity::Info);
        let id = notif.id;
        channel.send(&notif).await.unwrap();
        channel.send(&Notification::new("Another", "b", Severity::Info)).await.unwrap();

        channel.mark_read(id).await;

        let unread = channel.list(None, Some(false), 50).await;
        assert_eq!(unread.len(), 1);
        assert_eq!(unread[0].title, "Another");
    }

    #[tokio::test]
    async fn get_notification_by_id() {
        let channel = InAppChannel::new();
        let notif = Notification::new("Find Me", "b", Severity::Warning);
        let id = notif.id;
        channel.send(&notif).await.unwrap();

        let found = channel.get(id).await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().title, "Find Me");
    }

    #[tokio::test]
    async fn get_nonexistent_returns_none() {
        let channel = InAppChannel::new();
        assert!(channel.get(Uuid::now_v7()).await.is_none());
    }

    #[tokio::test]
    async fn mark_read_returns_false_for_missing() {
        let channel = InAppChannel::new();
        assert!(!channel.mark_read(Uuid::now_v7()).await);
    }

    #[tokio::test]
    async fn evicts_oldest_when_over_capacity() {
        let channel = InAppChannel::new().with_max_stored(2);
        channel.send(&Notification::new("First", "b", Severity::Info)).await.unwrap();
        channel.send(&Notification::new("Second", "b", Severity::Info)).await.unwrap();
        channel.send(&Notification::new("Third", "b", Severity::Info)).await.unwrap();

        assert_eq!(channel.count().await, 2);
        let listed = channel.list(None, None, 10).await;
        // Most recent first
        assert_eq!(listed[0].title, "Third");
        assert_eq!(listed[1].title, "Second");
    }

    #[tokio::test]
    async fn count_unread() {
        let channel = InAppChannel::new();
        let notif = Notification::new("Read Me", "b", Severity::Info);
        let id = notif.id;
        channel.send(&notif).await.unwrap();
        channel.send(&Notification::new("Unread", "b", Severity::Info)).await.unwrap();

        assert_eq!(channel.count_unread().await, 2);
        channel.mark_read(id).await;
        assert_eq!(channel.count_unread().await, 1);
    }

    #[test]
    fn channel_metadata() {
        let channel = InAppChannel::new();
        assert_eq!(channel.name(), "in-app");
        assert_eq!(channel.channel_type(), NotificationChannelType::InApp);
    }
}
