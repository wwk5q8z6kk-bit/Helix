use super::HelixEngine;
use chrono::Utc;
use hx_core::{McpConnector, McpConnectorStore, MvResult};
use uuid::Uuid;

impl HelixEngine {
    /// List MCP connectors with optional filters.
    pub async fn list_mcp_connectors(
        &self,
        publisher: Option<&str>,
        verified: Option<bool>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<McpConnector>> {
        self.store
            .nodes
            .list_mcp_connectors(publisher, verified, limit, offset)
            .await
    }

    /// Get a single MCP connector by ID.
    pub async fn get_mcp_connector(&self, id: Uuid) -> MvResult<Option<McpConnector>> {
        self.store.nodes.get_mcp_connector(id).await
    }

    /// Create a new MCP connector.
    pub async fn create_mcp_connector(
        &self,
        name: String,
        description: Option<String>,
        publisher: Option<String>,
        version: String,
        homepage_url: Option<String>,
        repository_url: Option<String>,
        config_schema: serde_json::Value,
        capabilities: Vec<String>,
        verified: bool,
    ) -> MvResult<McpConnector> {
        let now = Utc::now();
        let connector = McpConnector {
            id: Uuid::now_v7(),
            name,
            description,
            publisher,
            version,
            homepage_url,
            repository_url,
            config_schema,
            capabilities,
            verified,
            created_at: now,
            updated_at: now,
        };
        self.store.nodes.insert_mcp_connector(&connector).await?;
        Ok(connector)
    }

    /// Update an existing MCP connector. Returns true if the connector was found and updated.
    pub async fn update_mcp_connector(&self, connector: McpConnector) -> MvResult<bool> {
        self.store.nodes.update_mcp_connector(&connector).await
    }

    /// Delete an MCP connector by ID.
    pub async fn delete_mcp_connector(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_mcp_connector(id).await
    }
}
