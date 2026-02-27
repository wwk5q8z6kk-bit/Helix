use std::sync::Arc;

use hx_core::*;
use hx_storage::unified::UnifiedStore;
use uuid::Uuid;

pub struct RelayEngine {
    store: Arc<UnifiedStore>,
}

impl RelayEngine {
    pub fn new(store: Arc<UnifiedStore>) -> Self {
        Self { store }
    }

    // --- Contacts ---

    pub async fn add_contact(&self, contact: &RelayContact) -> MvResult<()> {
        self.store.nodes.add_relay_contact(contact).await
    }

    pub async fn get_contact(&self, id: Uuid) -> MvResult<Option<RelayContact>> {
        self.store.nodes.get_relay_contact(id).await
    }

    pub async fn list_contacts(&self) -> MvResult<Vec<RelayContact>> {
        self.store.nodes.list_relay_contacts().await
    }

    pub async fn update_contact(&self, contact: &RelayContact) -> MvResult<bool> {
        self.store.nodes.update_relay_contact(contact).await
    }

    pub async fn delete_contact(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_relay_contact(id).await
    }

    // --- Channels ---

    pub async fn create_channel(&self, channel: &RelayChannel) -> MvResult<()> {
        self.store.nodes.add_relay_channel(channel).await
    }

    pub async fn get_channel(&self, id: Uuid) -> MvResult<Option<RelayChannel>> {
        self.store.nodes.get_relay_channel(id).await
    }

    pub async fn list_channels(&self) -> MvResult<Vec<RelayChannel>> {
        self.store.nodes.list_relay_channels().await
    }

    pub async fn delete_channel(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_relay_channel(id).await
    }

    // --- Messages ---

    /// Send a message: store it, create a vault node for searchability, return the message.
    pub async fn send_message(
        &self,
        mut message: RelayMessage,
        namespace: &str,
    ) -> MvResult<RelayMessage> {
        // Create a KnowledgeNode for this conversation message
        let node = KnowledgeNode::new(NodeKind::Conversation, message.content.clone())
            .with_namespace(namespace.to_string())
            .with_tags(vec![
                "relay".to_string(),
                format!("channel:{}", message.channel_id),
            ]);

        self.store.nodes.insert(&node).await?;
        message.vault_node_id = Some(node.id);

        self.store.nodes.add_relay_message(&message).await?;
        Ok(message)
    }

    /// Receive an inbound message (from another vault).
    pub async fn receive_message(
        &self,
        mut message: RelayMessage,
        namespace: &str,
    ) -> MvResult<RelayMessage> {
        message.direction = MessageDirection::Inbound;
        message.status = MessageStatus::Delivered;

        let mut blocked = false;
        if let Some(sender_id) = message.sender_contact_id {
            if let Some(contact) = self.store.nodes.get_relay_contact(sender_id).await? {
                let mut candidates = vec![contact.display_name.clone(), contact.public_key.clone()];
                if let Some(addr) = contact.vault_address.clone() {
                    candidates.push(addr);
                }
                for candidate in candidates {
                    if self
                        .store
                        .nodes
                        .is_sender_blocked("relay", &candidate)
                        .await?
                    {
                        blocked = true;
                        break;
                    }
                }
            }
        }

        if blocked {
            message.status = MessageStatus::Failed;
            message
                .metadata
                .insert("blocked".to_string(), serde_json::Value::Bool(true));
        } else {
            let node = KnowledgeNode::new(NodeKind::Conversation, message.content.clone())
                .with_namespace(namespace.to_string())
                .with_tags(vec![
                    "relay".to_string(),
                    "inbound".to_string(),
                    format!("channel:{}", message.channel_id),
                ]);

            self.store.nodes.insert(&node).await?;
            message.vault_node_id = Some(node.id);
        }

        self.store.nodes.add_relay_message(&message).await?;
        Ok(message)
    }

    pub async fn get_message(&self, id: Uuid) -> MvResult<Option<RelayMessage>> {
        self.store.nodes.get_relay_message(id).await
    }

    pub async fn list_messages(
        &self,
        channel_id: Uuid,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<RelayMessage>> {
        self.store
            .nodes
            .list_relay_messages(channel_id, limit, offset)
            .await
    }

    pub async fn list_thread(&self, thread_id: Uuid, limit: usize) -> MvResult<Vec<RelayMessage>> {
        self.store
            .nodes
            .list_thread_messages(thread_id, limit)
            .await
    }

    pub async fn mark_read(&self, id: Uuid) -> MvResult<bool> {
        self.store
            .nodes
            .update_message_status(id, MessageStatus::Read)
            .await
    }

    pub async fn update_status(&self, id: Uuid, status: MessageStatus) -> MvResult<bool> {
        self.store.nodes.update_message_status(id, status).await
    }

    pub async fn unread_count(&self, channel_id: Option<Uuid>) -> MvResult<usize> {
        self.store.nodes.count_unread_messages(channel_id).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inbound_message_creates_vault_node() {
        let store = Arc::new(UnifiedStore::in_memory(384).unwrap());
        let engine = RelayEngine::new(Arc::clone(&store));

        let contact = RelayContact::new("Alice", "pk-alice");
        store.nodes.add_relay_contact(&contact).await.unwrap();

        let channel = RelayChannel::direct(contact.id);
        store.nodes.add_relay_channel(&channel).await.unwrap();

        let message = RelayMessage::inbound(channel.id, contact.id, "Hello");
        let stored = engine.receive_message(message, "default").await.unwrap();

        assert_eq!(stored.status, MessageStatus::Delivered);
        assert!(stored.vault_node_id.is_some());
    }

    #[tokio::test]
    async fn blocked_sender_skips_vault_node() {
        let store = Arc::new(UnifiedStore::in_memory(384).unwrap());
        let engine = RelayEngine::new(Arc::clone(&store));

        let contact = RelayContact::new("Spam Bot", "pk-spam");
        store.nodes.add_relay_contact(&contact).await.unwrap();

        let channel = RelayChannel::direct(contact.id);
        store.nodes.add_relay_channel(&channel).await.unwrap();

        let blocked = BlockedSender::new("relay", "Spam Bot");
        store.nodes.add_blocked_sender(&blocked).await.unwrap();

        let message = RelayMessage::inbound(channel.id, contact.id, "Buy now");
        let stored = engine.receive_message(message, "default").await.unwrap();

        assert_eq!(stored.status, MessageStatus::Failed);
        assert!(stored.vault_node_id.is_none());
        assert_eq!(
            stored.metadata.get("blocked"),
            Some(&serde_json::Value::Bool(true))
        );
    }
}
