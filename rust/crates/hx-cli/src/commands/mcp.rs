use std::sync::Arc;

use anyhow::Result;
use hx_mcp::auth::McpContext;
use hx_mcp::server::McpServer;

pub async fn run(config_path: &str, access_key: Option<String>, allow_unscoped: bool) -> Result<()> {
    let engine = super::load_engine(config_path).await?;
    let engine = Arc::new(engine);
    let access_key = access_key.or_else(|| std::env::var("HELIX_MCP_ACCESS_KEY").ok());
    let allow_unscoped = allow_unscoped
        || std::env::var("HELIX_MCP_ALLOW_UNSCOPED")
            .map(|v| matches!(v.as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

    let context = if let Some(token) = access_key {
        McpContext::from_access_key(&engine, &token)
            .await
            .map_err(|e| anyhow::anyhow!("MCP access key error: {e}"))?
    } else if allow_unscoped {
        McpContext::unscoped_read_only()
    } else {
        return Err(anyhow::anyhow!(
            "MCP access key required. Set HELIX_MCP_ACCESS_KEY or pass --access-key (use --allow-unscoped for read-only)."
        ));
    };
    let server = McpServer::new(engine, context);
    server
        .run_stdio()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;
    Ok(())
}
