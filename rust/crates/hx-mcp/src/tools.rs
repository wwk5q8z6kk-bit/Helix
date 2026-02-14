use std::collections::HashMap;
use std::sync::Arc;

use hx_core::*;
use hx_engine::engine::HelixEngine;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::auth::McpContext;
use crate::protocol::{ToolDefinition, ToolResult};

/// Return all tool definitions exposed by this MCP server.
pub fn list_tools(ctx: &McpContext) -> Vec<ToolDefinition> {
    let mut tools = Vec::new();
    let allow_read = ctx.scope().ensure_action("mcp.read").is_ok();
    let allow_propose = ctx.scope().ensure_action("mcp.propose").is_ok() && ctx.can_write();

    if allow_read {
        tools.push(ToolDefinition {
            name: "helix_search_vault".into(),
            description:
                "Hybrid search across the vault (vector + full-text + graph). Scoped by access."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return",
                        "default": 10
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Search strategy: hybrid, vector, fulltext, graph",
                        "default": "hybrid"
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Minimum relevance score threshold",
                        "default": 0.0
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Filter by namespace"
                    },
                    "kinds": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by node kinds"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by tags"
                    }
                },
                "required": ["query"]
            }),
        });
        tools.push(ToolDefinition {
            name: "helix_get_node".into(),
            description: "Get a single knowledge node by UUID (scoped).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "UUID of the node"
                    }
                },
                "required": ["id"]
            }),
        });
        tools.push(ToolDefinition {
            name: "helix_list_recent".into(),
            description: "List recent nodes (scoped).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 20
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Filter by namespace"
                    },
                    "kinds": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by node kinds"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Filter by tags"
                    }
                }
            }),
        });
    }

    if allow_propose {
        tools.push(ToolDefinition {
            name: "helix_propose_node".into(),
            description: "Submit a proposal to create a new node (owner must approve).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The main content/body of the knowledge node"
                    },
                    "kind": {
                        "type": "string",
                        "description": "Node kind: fact, task, event, decision, preference, entity, code_snippet, project, conversation, procedure, observation, bookmark, template, saved_view",
                        "default": "fact"
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional title for the node"
                    },
                    "source": {
                        "type": "string",
                        "description": "Optional source URL or reference"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Tags for categorization"
                    },
                    "namespace": {
                        "type": "string",
                        "description": "Namespace (default: 'default')",
                        "default": "default"
                    },
                    "importance": {
                        "type": "number",
                        "description": "Importance score from 0.0 to 1.0",
                        "default": 0.5
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Additional key-value metadata"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score for the proposal (0.0-1.0)"
                    },
                    "diff_preview": {
                        "type": "string",
                        "description": "Optional diff/preview text"
                    }
                },
                "required": ["content"]
            }),
        });
        tools.push(ToolDefinition {
            name: "helix_propose_tag".into(),
            description: "Suggest a tag for an existing node (owner must approve).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target_node_id": {
                        "type": "string",
                        "description": "UUID of the node to tag"
                    },
                    "tag": {
                        "type": "string",
                        "description": "Tag to suggest"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score for the proposal (0.0-1.0)"
                    },
                    "diff_preview": {
                        "type": "string",
                        "description": "Optional diff/preview text"
                    }
                },
                "required": ["target_node_id", "tag"]
            }),
        });
        tools.push(ToolDefinition {
            name: "helix_propose_update".into(),
            description: "Propose updates to an existing node (owner must approve).".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "id": {
                        "type": "string",
                        "description": "UUID of the node to update"
                    },
                    "content": {
                        "type": "string",
                        "description": "New content (replaces existing)"
                    },
                    "title": {
                        "type": "string",
                        "description": "New title"
                    },
                    "tags": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "New tags (replaces existing)"
                    },
                    "importance": {
                        "type": "number",
                        "description": "New importance (0.0 to 1.0)"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata fields to merge"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score for the proposal (0.0-1.0)"
                    },
                    "diff_preview": {
                        "type": "string",
                        "description": "Optional diff/preview text"
                    }
                },
                "required": ["id"]
            }),
        });
    }

    tools
}

/// Dispatch a tool call to the appropriate handler.
pub async fn call_tool(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    name: &str,
    params: Value,
) -> ToolResult {
    match name {
        "helix_search_vault" => tool_search_vault(engine, ctx, params, None).await,
        "helix_recall" => {
            tool_search_vault(engine, ctx, params, Some(SearchStrategy::Hybrid)).await
        }
        "helix_search" => {
            tool_search_vault(engine, ctx, params, Some(SearchStrategy::FullText)).await
        }
        "helix_get_node" => tool_get_node(engine, ctx, params).await,
        "helix_list_recent" | "helix_list_nodes" => {
            tool_list_recent(engine, ctx, params).await
        }
        "helix_propose_node" | "helix_store" => tool_propose_node(engine, ctx, params).await,
        "helix_propose_tag" => tool_propose_tag(engine, ctx, params).await,
        "helix_propose_update" | "helix_update_node" => {
            tool_propose_update(engine, ctx, params).await
        }
        "helix_delete_node" => tool_propose_delete(engine, ctx, params).await,
        _ => ToolResult::error(format!("unknown tool: {name}")),
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

async fn tool_search_vault(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    params: Value,
    strategy_override: Option<SearchStrategy>,
) -> ToolResult {
    if let Err(e) = ctx.scope().ensure_action("mcp.read") {
        return ToolResult::error(e);
    }

    let text = match params.get("query").and_then(|v| v.as_str()) {
        Some(q) => q.to_string(),
        None => return ToolResult::error("missing required parameter: query"),
    };

    let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
    let min_score = params
        .get("min_score")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let strategy = match strategy_override {
        Some(s) => s,
        None => {
            let strategy_str = params
                .get("strategy")
                .and_then(|v| v.as_str())
                .unwrap_or("hybrid");
            match strategy_str.parse::<SearchStrategy>() {
                Ok(s) => s,
                Err(e) => return ToolResult::error(format!("invalid strategy: {e}")),
            }
        }
    };

    let mut query = MemoryQuery::new(text)
        .with_strategy(strategy)
        .with_limit(limit)
        .with_min_score(min_score);

    if let Some(ns) = params.get("namespace").and_then(|v| v.as_str()) {
        query = query.with_namespace(ns);
    }
    if let Some(kinds) = params.get("kinds").and_then(|v| v.as_array()) {
        let parsed: Result<Vec<NodeKind>, _> = kinds
            .iter()
            .filter_map(|k| k.as_str())
            .map(|s| s.parse::<NodeKind>().map_err(|e| e.to_string()))
            .collect();
        match parsed {
            Ok(k) => query = query.with_kinds(k),
            Err(e) => return ToolResult::error(format!("invalid kind filter: {e}")),
        }
    }
    if let Some(tags) = params.get("tags").and_then(|v| v.as_array()) {
        let tag_vec: Vec<String> = tags
            .iter()
            .filter_map(|t| t.as_str().map(String::from))
            .collect();
        query = query.with_tags(tag_vec);
    }

    if let Err(e) = ctx.scope().apply_filters(&mut query.filters) {
        return ToolResult::error(e);
    }

    match engine.recall(&query).await {
        Ok(results) => {
            let items: Vec<Value> = results
                .iter()
                .map(|r| {
                    json!({
                        "id": r.node.id.to_string(),
                        "kind": r.node.kind.as_str(),
                        "title": r.node.title,
                        "content": r.node.content,
                        "namespace": r.node.namespace,
                        "tags": r.node.tags,
                        "score": r.score,
                        "match_source": format!("{:?}", r.match_source).to_lowercase(),
                    })
                })
                .collect();
            ToolResult::text(
                serde_json::to_string_pretty(&json!({ "results": items, "count": items.len() }))
                    .unwrap_or_default(),
            )
        }
        Err(e) => ToolResult::error(format!("search failed: {e}")),
    }
}

async fn tool_get_node(engine: &Arc<HelixEngine>, ctx: &McpContext, params: Value) -> ToolResult {
    if let Err(e) = ctx.scope().ensure_action("mcp.read") {
        return ToolResult::error(e);
    }

    let id = match parse_uuid(&params, "id") {
        Ok(id) => id,
        Err(e) => return e,
    };

    match engine.get_node(id).await {
        Ok(Some(node)) => match ctx.scope().check_node(&node) {
            Ok(()) => ToolResult::text(serde_json::to_string_pretty(&node).unwrap_or_default()),
            Err(e) => ToolResult::error(e),
        },
        Ok(None) => ToolResult::error(format!("node not found: {id}")),
        Err(e) => ToolResult::error(format!("get_node failed: {e}")),
    }
}

async fn tool_list_recent(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    params: Value,
) -> ToolResult {
    if let Err(e) = ctx.scope().ensure_action("mcp.read") {
        return ToolResult::error(e);
    }

    let limit = params.get("limit").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
    let mut filters = QueryFilters::default();

    if let Some(ns) = params.get("namespace").and_then(|v| v.as_str()) {
        filters.namespace = Some(ns.to_string());
    }
    if let Some(kinds) = params.get("kinds").and_then(|v| v.as_array()) {
        let parsed: Result<Vec<NodeKind>, _> = kinds
            .iter()
            .filter_map(|k| k.as_str())
            .map(|s| s.parse::<NodeKind>().map_err(|e| e.to_string()))
            .collect();
        match parsed {
            Ok(k) => filters.kinds = Some(k),
            Err(e) => return ToolResult::error(format!("invalid kind filter: {e}")),
        }
    }
    if let Some(tags) = params.get("tags").and_then(|v| v.as_array()) {
        let tag_vec: Vec<String> = tags
            .iter()
            .filter_map(|t| t.as_str().map(String::from))
            .collect();
        filters.tags = Some(tag_vec);
    }

    if let Err(e) = ctx.scope().apply_filters(&mut filters) {
        return ToolResult::error(e);
    }

    match engine.list_nodes(&filters, limit, 0).await {
        Ok(nodes) => {
            let items: Vec<Value> = nodes
                .iter()
                .map(|n| {
                    json!({
                        "id": n.id.to_string(),
                        "kind": n.kind.as_str(),
                        "title": n.title,
                        "content_preview": truncate(&n.content, 200),
                        "namespace": n.namespace,
                        "tags": n.tags,
                        "importance": n.importance,
                        "updated_at": n.temporal.updated_at.to_rfc3339(),
                    })
                })
                .collect();
            ToolResult::text(
                serde_json::to_string_pretty(&json!({
                    "nodes": items,
                    "count": items.len(),
                }))
                .unwrap_or_default(),
            )
        }
        Err(e) => ToolResult::error(format!("list_recent failed: {e}")),
    }
}

async fn tool_propose_node(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    params: Value,
) -> ToolResult {
    if let Err(e) = ctx.scope().ensure_action("mcp.propose") {
        return ToolResult::error(e);
    }
    if let Err(e) = ctx.scope().ensure_write_allowed() {
        return ToolResult::error(e);
    }

    let content = match params.get("content").and_then(|v| v.as_str()) {
        Some(c) => c.to_string(),
        None => return ToolResult::error("missing required parameter: content"),
    };

    let kind_str = params
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("fact");
    let kind: NodeKind = match kind_str.parse() {
        Ok(k) => k,
        Err(e) => return ToolResult::error(format!("invalid kind: {e}")),
    };

    if let Err(e) = ctx.scope().ensure_kind(kind) {
        return ToolResult::error(e);
    }

    let namespace = match ctx
        .scope()
        .normalize_namespace(params.get("namespace").and_then(|v| v.as_str()))
    {
        Ok(ns) => ns.unwrap_or_else(|| "default".to_string()),
        Err(e) => return ToolResult::error(e),
    };

    let mut tags: Vec<String> = params
        .get("tags")
        .and_then(|v| v.as_array())
        .map(|tags| {
            tags.iter()
                .filter_map(|t| t.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    tags = ctx.scope().ensure_tags_for_proposal(tags);

    let mut payload: HashMap<String, Value> = HashMap::new();
    payload.insert("content".into(), Value::String(content));
    payload.insert("kind".into(), Value::String(kind.as_str().to_string()));
    payload.insert("namespace".into(), Value::String(namespace));

    if let Some(title) = params.get("title").and_then(|v| v.as_str()) {
        payload.insert("title".into(), Value::String(title.to_string()));
    }
    if let Some(source) = params.get("source").and_then(|v| v.as_str()) {
        payload.insert("source".into(), Value::String(source.to_string()));
    }
    if !tags.is_empty() {
        payload.insert(
            "tags".into(),
            Value::Array(tags.into_iter().map(Value::String).collect()),
        );
    }
    if let Some(importance) = params.get("importance").and_then(|v| v.as_f64()) {
        payload.insert(
            "importance".into(),
            Value::Number(serde_json::Number::from_f64(importance).unwrap_or_else(|| serde_json::Number::from(0u64))),
        );
    }
    if let Some(metadata) = params.get("metadata").and_then(|v| v.as_object()) {
        payload.insert("metadata".into(), Value::Object(metadata.clone()));
    }

    let mut proposal = Proposal::new(ProposalSender::Mcp, ProposalAction::CreateNode)
        .with_payload(payload);

    if let Some(confidence) = params.get("confidence").and_then(|v| v.as_f64()) {
        if !(0.0..=1.0).contains(&confidence) {
            return ToolResult::error("confidence must be between 0.0 and 1.0");
        }
        proposal = proposal.with_confidence(confidence as f32);
    }
    if let Some(diff) = params.get("diff_preview").and_then(|v| v.as_str()) {
        proposal = proposal.with_diff(diff);
    }

    match engine.submit_proposal(&proposal).await {
        Ok(()) => ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "proposal_id": proposal.id.to_string(),
                "state": proposal.state.as_str(),
            }))
            .unwrap_or_default(),
        ),
        Err(e) => ToolResult::error(format!("proposal failed: {e}")),
    }
}

async fn tool_propose_tag(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    params: Value,
) -> ToolResult {
    if let Err(e) = ctx.scope().ensure_action("mcp.propose") {
        return ToolResult::error(e);
    }
    if let Err(e) = ctx.scope().ensure_write_allowed() {
        return ToolResult::error(e);
    }

    let target_id = match parse_uuid(&params, "target_node_id") {
        Ok(id) => id,
        Err(e) => return e,
    };

    let tag = match params.get("tag").and_then(|v| v.as_str()) {
        Some(t) if !t.trim().is_empty() => t.trim().to_string(),
        _ => return ToolResult::error("missing required parameter: tag"),
    };

    let node = match engine.get_node(target_id).await {
        Ok(Some(n)) => n,
        Ok(None) => return ToolResult::error(format!("node not found: {target_id}")),
        Err(e) => return ToolResult::error(format!("get_node failed: {e}")),
    };

    if let Err(e) = ctx.scope().check_node(&node) {
        return ToolResult::error(e);
    }

    let mut payload = HashMap::new();
    payload.insert("tag".into(), Value::String(tag));

    let mut proposal =
        Proposal::new(ProposalSender::Mcp, ProposalAction::SuggestTag).with_target(target_id);
    proposal = proposal.with_payload(payload);

    if let Some(confidence) = params.get("confidence").and_then(|v| v.as_f64()) {
        if !(0.0..=1.0).contains(&confidence) {
            return ToolResult::error("confidence must be between 0.0 and 1.0");
        }
        proposal = proposal.with_confidence(confidence as f32);
    }
    if let Some(diff) = params.get("diff_preview").and_then(|v| v.as_str()) {
        proposal = proposal.with_diff(diff);
    }

    match engine.submit_proposal(&proposal).await {
        Ok(()) => ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "proposal_id": proposal.id.to_string(),
                "state": proposal.state.as_str(),
            }))
            .unwrap_or_default(),
        ),
        Err(e) => ToolResult::error(format!("proposal failed: {e}")),
    }
}

async fn tool_propose_update(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    params: Value,
) -> ToolResult {
    if let Err(e) = ctx.scope().ensure_action("mcp.propose") {
        return ToolResult::error(e);
    }
    if let Err(e) = ctx.scope().ensure_write_allowed() {
        return ToolResult::error(e);
    }

    let target_id = match parse_uuid(&params, "id") {
        Ok(id) => id,
        Err(e) => return e,
    };

    let existing = match engine.get_node(target_id).await {
        Ok(Some(node)) => node,
        Ok(None) => return ToolResult::error(format!("node not found: {target_id}")),
        Err(e) => return ToolResult::error(format!("get_node failed: {e}")),
    };

    if let Err(e) = ctx.scope().check_node(&existing) {
        return ToolResult::error(e);
    }

    let mut payload: HashMap<String, Value> = HashMap::new();

    if let Some(content) = params.get("content").and_then(|v| v.as_str()) {
        payload.insert("content".into(), Value::String(content.to_string()));
    }
    if let Some(title) = params.get("title").and_then(|v| v.as_str()) {
        payload.insert("title".into(), Value::String(title.to_string()));
    }
    if let Some(importance) = params.get("importance").and_then(|v| v.as_f64()) {
        payload.insert(
            "importance".into(),
            Value::Number(serde_json::Number::from_f64(importance).unwrap_or_else(|| serde_json::Number::from(0u64))),
        );
    }
    if let Some(metadata) = params.get("metadata").and_then(|v| v.as_object()) {
        payload.insert("metadata".into(), Value::Object(metadata.clone()));
    }
    if let Some(tags) = params.get("tags").and_then(|v| v.as_array()) {
        let tag_vec: Vec<String> = tags
            .iter()
            .filter_map(|t| t.as_str().map(String::from))
            .collect();
        let scoped_tags = ctx.scope().ensure_tags_for_proposal(tag_vec);
        payload.insert(
            "tags".into(),
            Value::Array(scoped_tags.into_iter().map(Value::String).collect()),
        );
    }

    if payload.is_empty() {
        return ToolResult::error("no update fields provided");
    }

    let mut proposal =
        Proposal::new(ProposalSender::Mcp, ProposalAction::UpdateNode).with_target(target_id);
    proposal = proposal.with_payload(payload);

    if let Some(confidence) = params.get("confidence").and_then(|v| v.as_f64()) {
        if !(0.0..=1.0).contains(&confidence) {
            return ToolResult::error("confidence must be between 0.0 and 1.0");
        }
        proposal = proposal.with_confidence(confidence as f32);
    }
    if let Some(diff) = params.get("diff_preview").and_then(|v| v.as_str()) {
        proposal = proposal.with_diff(diff);
    }

    match engine.submit_proposal(&proposal).await {
        Ok(()) => ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "proposal_id": proposal.id.to_string(),
                "state": proposal.state.as_str(),
            }))
            .unwrap_or_default(),
        ),
        Err(e) => ToolResult::error(format!("proposal failed: {e}")),
    }
}

async fn tool_propose_delete(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    params: Value,
) -> ToolResult {
    if let Err(e) = ctx.scope().ensure_action("mcp.propose") {
        return ToolResult::error(e);
    }
    if let Err(e) = ctx.scope().ensure_write_allowed() {
        return ToolResult::error(e);
    }

    let target_id = match parse_uuid(&params, "id") {
        Ok(id) => id,
        Err(e) => return e,
    };

    let existing = match engine.get_node(target_id).await {
        Ok(Some(node)) => node,
        Ok(None) => return ToolResult::error(format!("node not found: {target_id}")),
        Err(e) => return ToolResult::error(format!("get_node failed: {e}")),
    };

    if let Err(e) = ctx.scope().check_node(&existing) {
        return ToolResult::error(e);
    }

    let mut proposal =
        Proposal::new(ProposalSender::Mcp, ProposalAction::DeleteNode).with_target(target_id);

    if let Some(confidence) = params.get("confidence").and_then(|v| v.as_f64()) {
        if !(0.0..=1.0).contains(&confidence) {
            return ToolResult::error("confidence must be between 0.0 and 1.0");
        }
        proposal = proposal.with_confidence(confidence as f32);
    }
    if let Some(diff) = params.get("diff_preview").and_then(|v| v.as_str()) {
        proposal = proposal.with_diff(diff);
    }

    match engine.submit_proposal(&proposal).await {
        Ok(()) => ToolResult::text(
            serde_json::to_string_pretty(&json!({
                "proposal_id": proposal.id.to_string(),
                "state": proposal.state.as_str(),
            }))
            .unwrap_or_default(),
        ),
        Err(e) => ToolResult::error(format!("proposal failed: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn parse_uuid(params: &Value, field: &str) -> Result<Uuid, ToolResult> {
    let s = params
        .get(field)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ToolResult::error(format!("missing required parameter: {field}")))?;
    Uuid::parse_str(s).map_err(|e| ToolResult::error(format!("invalid UUID for {field}: {e}")))
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len).collect();
        format!("{truncated}...")
    }
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

    fn writable_ctx() -> McpContext {
        McpContext::with_scope(McpScope {
            namespace: None,
            tags: Vec::new(),
            kinds: Vec::new(),
            allow_write: true,
            allow_actions: vec!["mcp.read".into(), "mcp.propose".into()],
            resource_limit: 1000,
            tier: PermissionTier::Edit,
        })
    }

    fn read_only_ctx() -> McpContext {
        McpContext::unscoped_read_only()
    }

    fn scoped_ctx(namespace: &str) -> McpContext {
        McpContext::with_scope(McpScope {
            namespace: Some(namespace.to_string()),
            tags: Vec::new(),
            kinds: Vec::new(),
            allow_write: true,
            allow_actions: vec!["mcp.read".into(), "mcp.propose".into()],
            resource_limit: 1000,
            tier: PermissionTier::Edit,
        })
    }

    // --- list_tools ---

    #[test]
    fn list_tools_read_only_shows_read_tools_only() {
        let ctx = read_only_ctx();
        let tools = list_tools(&ctx);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"helix_search_vault"));
        assert!(names.contains(&"helix_get_node"));
        assert!(names.contains(&"helix_list_recent"));
        // propose tools should NOT be listed
        assert!(!names.contains(&"helix_propose_node"));
        assert!(!names.contains(&"helix_propose_tag"));
        assert!(!names.contains(&"helix_propose_update"));
    }

    #[test]
    fn list_tools_writable_shows_all_tools() {
        let ctx = writable_ctx();
        let tools = list_tools(&ctx);
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"helix_search_vault"));
        assert!(names.contains(&"helix_propose_node"));
        assert!(names.contains(&"helix_propose_tag"));
        assert!(names.contains(&"helix_propose_update"));
    }

    // --- search_vault ---

    #[tokio::test]
    async fn search_vault_missing_query() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let result = call_tool(&engine, &ctx, "helix_search_vault", json!({})).await;
        assert!(result.is_error());
    }

    #[tokio::test]
    async fn search_vault_returns_results() {
        let (engine, _tmp) = test_engine().await;
        // Store a node first
        let node = hx_core::KnowledgeNode::new(NodeKind::Fact, "Rust is a systems language")
            .with_title("Rust Facts")
            .with_tags(vec!["rust".into()]);
        engine.store_node(node).await.unwrap();

        let ctx = read_only_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_search_vault",
            json!({"query": "rust systems", "strategy": "fulltext"}),
        )
        .await;
        assert!(!result.is_error());
    }

    // --- get_node ---

    #[tokio::test]
    async fn get_node_not_found() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let random_id = Uuid::now_v7().to_string();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_get_node",
            json!({"id": random_id}),
        )
        .await;
        assert!(result.is_error());
    }

    #[tokio::test]
    async fn get_node_scope_blocks_namespace() {
        let (engine, _tmp) = test_engine().await;
        let mut node = hx_core::KnowledgeNode::new(NodeKind::Fact, "private info");
        node.namespace = "private".to_string();
        let stored = engine.store_node(node).await.unwrap();

        let ctx = scoped_ctx("work");
        let result = call_tool(
            &engine,
            &ctx,
            "helix_get_node",
            json!({"id": stored.id.to_string()}),
        )
        .await;
        assert!(result.is_error());
    }

    #[tokio::test]
    async fn get_node_invalid_uuid() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_get_node",
            json!({"id": "not-a-uuid"}),
        )
        .await;
        assert!(result.is_error());
    }

    // --- propose_node ---

    #[tokio::test]
    async fn propose_node_read_only_blocked() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_propose_node",
            json!({"content": "new fact"}),
        )
        .await;
        assert!(result.is_error());
    }

    #[tokio::test]
    async fn propose_node_creates_pending() {
        let (engine, _tmp) = test_engine().await;
        let ctx = writable_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_propose_node",
            json!({"content": "a new knowledge node", "kind": "fact"}),
        )
        .await;
        assert!(!result.is_error());
        // Check the response contains a proposal_id
        if let Some(crate::protocol::ToolContent::Text { text }) = result.content.first() {
            let parsed: Value = serde_json::from_str(text).unwrap();
            assert!(parsed.get("proposal_id").is_some());
            assert_eq!(parsed["state"], "pending");
        } else {
            panic!("expected text content");
        }
    }

    #[tokio::test]
    async fn propose_node_missing_content() {
        let (engine, _tmp) = test_engine().await;
        let ctx = writable_ctx();
        let result = call_tool(&engine, &ctx, "helix_propose_node", json!({})).await;
        assert!(result.is_error());
    }

    #[tokio::test]
    async fn propose_node_invalid_confidence() {
        let (engine, _tmp) = test_engine().await;
        let ctx = writable_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_propose_node",
            json!({"content": "test", "confidence": 2.0}),
        )
        .await;
        assert!(result.is_error());
    }

    // --- propose_update ---

    #[tokio::test]
    async fn propose_update_no_fields() {
        let (engine, _tmp) = test_engine().await;
        let node = hx_core::KnowledgeNode::new(NodeKind::Fact, "original");
        let stored = engine.store_node(node).await.unwrap();

        let ctx = writable_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_propose_update",
            json!({"id": stored.id.to_string()}),
        )
        .await;
        assert!(result.is_error());
    }

    #[tokio::test]
    async fn propose_update_nonexistent_node() {
        let (engine, _tmp) = test_engine().await;
        let ctx = writable_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_propose_update",
            json!({"id": Uuid::now_v7().to_string(), "content": "new"}),
        )
        .await;
        assert!(result.is_error());
    }

    // --- propose_tag ---

    #[tokio::test]
    async fn propose_tag_nonexistent_node() {
        let (engine, _tmp) = test_engine().await;
        let ctx = writable_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_propose_tag",
            json!({"target_node_id": Uuid::now_v7().to_string(), "tag": "new-tag"}),
        )
        .await;
        assert!(result.is_error());
    }

    #[tokio::test]
    async fn propose_tag_missing_tag() {
        let (engine, _tmp) = test_engine().await;
        let ctx = writable_ctx();
        let result = call_tool(
            &engine,
            &ctx,
            "helix_propose_tag",
            json!({"target_node_id": Uuid::now_v7().to_string()}),
        )
        .await;
        assert!(result.is_error());
    }

    // --- unknown tool ---

    #[tokio::test]
    async fn unknown_tool_returns_error() {
        let (engine, _tmp) = test_engine().await;
        let ctx = read_only_ctx();
        let result = call_tool(&engine, &ctx, "nonexistent_tool", json!({})).await;
        assert!(result.is_error());
    }

    // --- truncate helper ---

    #[test]
    fn truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_long_string_adds_ellipsis() {
        let result = truncate("hello world", 5);
        assert_eq!(result, "hello...");
    }
}
