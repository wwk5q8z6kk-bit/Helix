use super::HelixEngine;
use chrono::Utc;
use hx_core::{CommentStore, MvResult, NodeComment};
use uuid::Uuid;

impl HelixEngine {
    /// Create a comment on a node.
    pub async fn create_node_comment(
        &self,
        node_id: Uuid,
        author: Option<String>,
        body: String,
    ) -> MvResult<NodeComment> {
        let now = Utc::now();
        let comment = NodeComment {
            id: Uuid::now_v7(),
            node_id,
            author,
            body,
            created_at: now,
            updated_at: now,
            resolved_at: None,
        };
        self.store.nodes.insert_comment(&comment).await?;
        Ok(comment)
    }

    /// List comments for a node, optionally including resolved ones.
    pub async fn list_node_comments(
        &self,
        node_id: Uuid,
        include_resolved: bool,
    ) -> MvResult<Vec<NodeComment>> {
        self.store
            .nodes
            .list_comments(node_id, include_resolved)
            .await
    }

    /// Get a single comment by ID.
    pub async fn get_node_comment(&self, id: Uuid) -> MvResult<Option<NodeComment>> {
        self.store.nodes.get_comment(id).await
    }

    /// Resolve (mark as resolved) a comment.
    pub async fn resolve_node_comment(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.resolve_comment(id, Utc::now()).await
    }

    /// Delete a comment.
    pub async fn delete_node_comment(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_comment(id).await
    }
}
