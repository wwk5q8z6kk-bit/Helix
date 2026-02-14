use std::collections::HashMap;
use std::sync::Arc;

use hx_core::*;
use hx_engine::engine::HelixEngine;
use serde_json::json;

use crate::auth::McpContext;
use crate::protocol::{ResourceContent, ResourceDefinition};

/// Return all resource definitions exposed by this MCP server.
pub fn list_resources(ctx: &McpContext) -> Vec<ResourceDefinition> {
    if ctx.scope().ensure_action("mcp.read").is_err() {
        return vec![];
    }
    vec![
        ResourceDefinition {
            uri: "helix://recent".into(),
            name: "Recent Nodes".into(),
            description: "The most recently created or updated knowledge nodes.".into(),
            mime_type: "application/json".into(),
        },
        ResourceDefinition {
            uri: "helix://tags".into(),
            name: "Tag Cloud".into(),
            description: "All tags with occurrence counts.".into(),
            mime_type: "application/json".into(),
        },
        ResourceDefinition {
            uri: "helix://stats".into(),
            name: "Vault Statistics".into(),
            description: "Overview statistics for the vault (node count, kinds breakdown).".into(),
            mime_type: "application/json".into(),
        },
    ]
}

/// Read a resource by URI.
pub async fn read_resource(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    uri: &str,
) -> Result<ResourceContent, String> {
    ctx.scope()
        .ensure_action("mcp.read")
        .map_err(|e| format!("scope error: {e}"))?;
    match uri {
        "helix://recent" => read_recent(engine, ctx).await,
        "helix://tags" => read_tags(engine, ctx).await,
        "helix://stats" => read_stats(engine, ctx).await,
        _ => Err(format!("unknown resource URI: {uri}")),
    }
}

async fn read_recent(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
) -> Result<ResourceContent, String> {
    let mut filters = QueryFilters::default();
    ctx.scope()
        .apply_filters(&mut filters)
        .map_err(|e| format!("scope error: {e}"))?;
    let nodes = engine
        .list_nodes(&filters, 20, 0)
        .await
        .map_err(|e| format!("list_nodes failed: {e}"))?;

    let items: Vec<serde_json::Value> = nodes
        .iter()
        .map(|n| {
            json!({
                "id": n.id.to_string(),
                "kind": n.kind.as_str(),
                "title": n.title,
                "namespace": n.namespace,
                "tags": n.tags,
                "created_at": n.temporal.created_at.to_rfc3339(),
                "updated_at": n.temporal.updated_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(ResourceContent {
        uri: "helix://recent".into(),
        mime_type: "application/json".into(),
        text: serde_json::to_string_pretty(&json!({ "nodes": items, "count": items.len() }))
            .unwrap_or_default(),
    })
}

async fn read_tags(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
) -> Result<ResourceContent, String> {
    let mut filters = QueryFilters::default();
    ctx.scope()
        .apply_filters(&mut filters)
        .map_err(|e| format!("scope error: {e}"))?;
    let limit = ctx.scope().resource_limit;
    let nodes = engine
        .list_nodes(&filters, limit, 0)
        .await
        .map_err(|e| format!("list_nodes failed: {e}"))?;

    let mut tag_counts: HashMap<String, usize> = HashMap::new();
    for node in &nodes {
        for tag in &node.tags {
            *tag_counts.entry(tag.clone()).or_default() += 1;
        }
    }

    let mut tag_list: Vec<serde_json::Value> = tag_counts
        .iter()
        .map(|(tag, count)| json!({"tag": tag, "count": count}))
        .collect();
    tag_list.sort_by(|a, b| {
        let ca = a["count"].as_u64().unwrap_or(0);
        let cb = b["count"].as_u64().unwrap_or(0);
        cb.cmp(&ca)
    });

    Ok(ResourceContent {
        uri: "helix://tags".into(),
        mime_type: "application/json".into(),
        text: serde_json::to_string_pretty(&json!({
            "tags": tag_list,
            "total_unique": tag_counts.len(),
            "truncated": nodes.len() >= limit,
        }))
        .unwrap_or_default(),
    })
}

async fn read_stats(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
) -> Result<ResourceContent, String> {
    let mut filters = QueryFilters::default();
    ctx.scope()
        .apply_filters(&mut filters)
        .map_err(|e| format!("scope error: {e}"))?;

    let limit = ctx.scope().resource_limit;
    let nodes = engine
        .list_nodes(&filters, limit, 0)
        .await
        .map_err(|e| format!("list_nodes failed: {e}"))?;

    let mut kind_counts: HashMap<&str, usize> = HashMap::new();
    for node in &nodes {
        *kind_counts.entry(node.kind.as_str()).or_default() += 1;
    }

    let total = nodes.len();
    let truncated = nodes.len() >= limit;

    let mut stats = json!({
        "total_nodes": total,
        "kinds": kind_counts,
        "truncated": truncated,
    });

    if ctx.scope().is_admin() {
        let embedding_status = engine.embedding_runtime_status();
        stats["embedding"] = json!({
            "provider": embedding_status.effective_provider,
            "model": embedding_status.effective_model,
            "dimensions": embedding_status.effective_dimensions,
        });
    }

    Ok(ResourceContent {
        uri: "helix://stats".into(),
        mime_type: "application/json".into(),
        text: serde_json::to_string_pretty(&stats).unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::{NodeKind, PermissionTier};
    use hx_engine::config::EngineConfig;
    use tempfile::TempDir;

    use crate::auth::{McpContext, McpScope};

    async fn test_engine() -> (Arc<HelixEngine>, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mut config = EngineConfig {
            data_dir: tmp.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        config.embedding.provider = "noop".into();
        let engine = HelixEngine::init(config).await.unwrap();
        (Arc::new(engine), tmp)
    }

    fn read_only_ctx() -> McpContext {
        McpContext::unscoped_read_only()
    }

    fn admin_ctx() -> McpContext {
        McpContext::with_scope(McpScope {
            namespace: None,
            tags: Vec::new(),
            kinds: Vec::new(),
            allow_write: true,
            allow_actions: Vec::new(),
            resource_limit: 1000,
            tier: PermissionTier::Admin,
        })
    }

    // --- list_resources ---

    #[test]
    fn list_resources_returns_three() {
        let ctx = read_only_ctx();
        let resources = list_resources(&ctx);
        assert_eq!(resources.len(), 3);
        let uris: Vec<&str> = resources.iter().map(|r| r.uri.as_str()).collect();
        assert!(uris.contains(&"helix://recent"));
        assert!(uris.contains(&"helix://tags"));
        assert!(uris.contains(&"helix://stats"));
    }

    #[test]
    fn list_resources_blocked_scope_returns_empty() {
        // A scope with no mcp.read action but with a restrictive allow_actions list
        let ctx = McpContext::with_scope(McpScope {
            namespace: None,
            tags: Vec::new(),
            kinds: Vec::new(),
            allow_write: false,
            allow_actions: vec!["other.action".into()],
            resource_limit: 1000,
            tier: PermissionTier::View,
        });
        let resources = list_resources(&ctx);
        assert!(resources.is_empty());
    }

    // --- read_resource ---

    #[tokio::test]
    async fn read_recent_empty_vault() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let result = read_resource(&engine, &ctx, "helix://recent").await;
        assert!(result.is_ok());
        let content = result.unwrap();
        assert_eq!(content.uri, "helix://recent");
        assert_eq!(content.mime_type, "application/json");
        let parsed: serde_json::Value = serde_json::from_str(&content.text).unwrap();
        assert_eq!(parsed["count"], 0);
    }

    #[tokio::test]
    async fn read_recent_with_nodes() {
        let (engine, _tmp) = test_engine().await;
        let node = KnowledgeNode::new(NodeKind::Fact, "test content")
            .with_tags(vec!["hello".into()]);
        engine.store_node(node).await.unwrap();

        let ctx = read_only_ctx();
        let content = read_resource(&engine, &ctx, "helix://recent")
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content.text).unwrap();
        assert_eq!(parsed["count"], 1);
    }

    #[tokio::test]
    async fn read_tags_empty_vault() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let content = read_resource(&engine, &ctx, "helix://tags")
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content.text).unwrap();
        assert_eq!(parsed["total_unique"], 0);
    }

    #[tokio::test]
    async fn read_tags_aggregates_counts() {
        let (engine, _tmp) = test_engine().await;
        let n1 = KnowledgeNode::new(NodeKind::Fact, "a")
            .with_tags(vec!["rust".into(), "programming".into()]);
        let n2 = KnowledgeNode::new(NodeKind::Fact, "b")
            .with_tags(vec!["rust".into()]);
        engine.store_node(n1).await.unwrap();
        engine.store_node(n2).await.unwrap();

        let ctx = read_only_ctx();
        let content = read_resource(&engine, &ctx, "helix://tags")
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content.text).unwrap();
        assert_eq!(parsed["total_unique"], 2);
        // "rust" should appear with count 2
        let tags = parsed["tags"].as_array().unwrap();
        let rust_tag = tags.iter().find(|t| t["tag"] == "rust").unwrap();
        assert_eq!(rust_tag["count"], 2);
    }

    #[tokio::test]
    async fn read_stats_empty_vault() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let content = read_resource(&engine, &ctx, "helix://stats")
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content.text).unwrap();
        assert_eq!(parsed["total_nodes"], 0);
        // Non-admin should not see embedding info
        assert!(parsed.get("embedding").is_none());
    }

    #[tokio::test]
    async fn read_stats_admin_includes_embedding() {
        let (engine, _tmp) = test_engine().await;
        let ctx = admin_ctx();
        let content = read_resource(&engine, &ctx, "helix://stats")
            .await
            .unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content.text).unwrap();
        assert!(parsed.get("embedding").is_some());
    }

    #[tokio::test]
    async fn read_unknown_uri() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let result = read_resource(&engine, &ctx, "helix://nonexistent").await;
        assert!(result.is_err());
    }
}
