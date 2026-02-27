use std::sync::Arc;

use hx_core::*;
use hx_engine::engine::HelixEngine;
use serde_json::Value;

use crate::auth::McpContext;
use crate::protocol::{PromptArgument, PromptContent, PromptDefinition, PromptMessage};

/// Return all prompt definitions exposed by this MCP server.
pub fn list_prompts(ctx: &McpContext) -> Vec<PromptDefinition> {
    if ctx.scope().ensure_action("mcp.read").is_err() {
        return vec![];
    }

    let mut prompts = vec![
        PromptDefinition {
            name: "summarize".into(),
            description: "Summarize content using your Helix knowledge as context.".into(),
            arguments: vec![
                PromptArgument {
                    name: "content".into(),
                    description: "The text to summarize.".into(),
                    required: true,
                },
                PromptArgument {
                    name: "namespace".into(),
                    description: "Namespace for context retrieval.".into(),
                    required: false,
                },
            ],
        },
        PromptDefinition {
            name: "extract_tasks".into(),
            description: "Extract actionable tasks from a block of text.".into(),
            arguments: vec![PromptArgument {
                name: "content".into(),
                description: "The text to extract tasks from.".into(),
                required: true,
            }],
        },
    ];

    if ctx.scope().ensure_kind(NodeKind::Task).is_ok() {
        prompts.push(PromptDefinition {
            name: "daily_briefing".into(),
            description:
                "Generate a daily briefing based on recent vault activity, due tasks, and insights."
                    .into(),
            arguments: vec![PromptArgument {
                name: "namespace".into(),
                description: "Namespace to scope the briefing to.".into(),
                required: false,
            }],
        });
    }

    prompts
}

/// Get prompt messages for a named prompt with the given arguments.
pub async fn get_prompt(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    name: &str,
    args: &Value,
) -> Result<Vec<PromptMessage>, String> {
    match name {
        "summarize" => prompt_summarize(engine, ctx, args).await,
        "extract_tasks" => prompt_extract_tasks(engine, ctx, args).await,
        "daily_briefing" => prompt_daily_briefing(engine, ctx, args).await,
        _ => Err(format!("unknown prompt: {name}")),
    }
}

async fn prompt_summarize(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    args: &Value,
) -> Result<Vec<PromptMessage>, String> {
    ctx.scope()
        .ensure_action("mcp.read")
        .map_err(|e| format!("scope error: {e}"))?;
    let content = args
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or("missing required argument: content")?;

    // Retrieve relevant context from the vault
    let mut query = MemoryQuery::new(content)
        .with_strategy(SearchStrategy::Hybrid)
        .with_limit(5);

    if let Some(ns) = args.get("namespace").and_then(|v| v.as_str()) {
        query = query.with_namespace(ns);
    }
    ctx.scope()
        .apply_filters(&mut query.filters)
        .map_err(|e| format!("scope error: {e}"))?;

    let context = match engine.recall(&query).await {
        Ok(results) => {
            let ctx_parts: Vec<String> = results
                .iter()
                .map(|r| {
                    let title = r.node.title.as_deref().unwrap_or("(untitled)");
                    format!(
                        "- [{} | {}] {}: {}",
                        r.node.kind,
                        title,
                        r.node.namespace,
                        truncate(&r.node.content, 300)
                    )
                })
                .collect();
            if ctx_parts.is_empty() {
                "No relevant context found in vault.".to_string()
            } else {
                format!(
                    "Relevant knowledge from your vault:\n{}",
                    ctx_parts.join("\n")
                )
            }
        }
        Err(_) => "Could not retrieve vault context.".to_string(),
    };

    Ok(vec![
		PromptMessage {
			role: "user".into(),
			content: PromptContent::Text {
				text: format!(
					"Please summarize the following content. Use the vault context below to enrich your summary with related knowledge.\n\n\
					 ## Vault Context\n{context}\n\n\
					 ## Content to Summarize\n{content}"
				),
			},
		},
	])
}

async fn prompt_extract_tasks(
    _engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    args: &Value,
) -> Result<Vec<PromptMessage>, String> {
    ctx.scope()
        .ensure_action("mcp.read")
        .map_err(|e| format!("scope error: {e}"))?;

    let content = args
        .get("content")
        .and_then(|v| v.as_str())
        .ok_or("missing required argument: content")?;

    Ok(vec![PromptMessage {
        role: "user".into(),
        content: PromptContent::Text {
            text: format!(
                "Extract all actionable tasks from the following text. For each task, provide:\n\
				 - A clear, concise title\n\
				 - Any due date if mentioned\n\
				 - Priority (high/medium/low) based on urgency cues\n\
				 - Relevant tags\n\n\
				 Format each task as a JSON object with fields: title, due_date, priority, tags.\n\
				 Return a JSON array of tasks.\n\n\
				 ## Text\n{content}"
            ),
        },
    }])
}

async fn prompt_daily_briefing(
    engine: &Arc<HelixEngine>,
    ctx: &McpContext,
    args: &Value,
) -> Result<Vec<PromptMessage>, String> {
    ctx.scope()
        .ensure_action("mcp.read")
        .map_err(|e| format!("scope error: {e}"))?;

    let namespace = args.get("namespace").and_then(|v| v.as_str());

    // Get recent nodes
    let mut filters = QueryFilters::default();
    if let Some(ns) = namespace {
        filters.namespace = Some(ns.to_string());
    }
    ctx.scope()
        .apply_filters(&mut filters)
        .map_err(|e| format!("scope error: {e}"))?;
    let recent_nodes = engine
        .list_nodes(&filters, 15, 0)
        .await
        .map_err(|e| format!("list_nodes failed: {e}"))?;

    let recent_summary: Vec<String> = recent_nodes
        .iter()
        .map(|n| {
            let title = n.title.as_deref().unwrap_or("(untitled)");
            format!("- [{}] {}: {}", n.kind, title, truncate(&n.content, 150))
        })
        .collect();

    // Get due tasks
    let mut task_filters = QueryFilters {
        kinds: Some(vec![NodeKind::Task]),
        ..Default::default()
    };
    if let Some(ns) = namespace {
        task_filters.namespace = Some(ns.to_string());
    }
    ctx.scope()
        .apply_filters(&mut task_filters)
        .map_err(|e| format!("scope error: {e}"))?;
    let tasks = engine
        .list_nodes(&task_filters, 50, 0)
        .await
        .map_err(|e| format!("list_nodes failed: {e}"))?;

    let open_tasks: Vec<String> = tasks
        .iter()
        .filter(|t| {
            t.metadata
                .get("task_status")
                .and_then(|v| v.as_str())
                .unwrap_or("inbox")
                != "done"
        })
        .take(10)
        .map(|t| {
            let title = t.title.as_deref().unwrap_or("(untitled)");
            let status = t
                .metadata
                .get("task_status")
                .and_then(|v| v.as_str())
                .unwrap_or("inbox");
            let due = t
                .metadata
                .get("task_due_at")
                .and_then(|v| v.as_str())
                .unwrap_or("no due date");
            format!("- [{status}] {title} (due: {due})")
        })
        .collect();

    let mut count_filters = QueryFilters::default();
    if let Some(ns) = namespace {
        count_filters.namespace = Some(ns.to_string());
    }
    ctx.scope()
        .apply_filters(&mut count_filters)
        .map_err(|e| format!("scope error: {e}"))?;
    let count_nodes = engine
        .list_nodes(&count_filters, ctx.scope().resource_limit, 0)
        .await
        .map_err(|e| format!("list_nodes failed: {e}"))?;
    let total_nodes = count_nodes.len();
    let truncated = count_nodes.len() >= ctx.scope().resource_limit;
    let namespace_label = namespace
        .map(|ns| ns.to_string())
        .or_else(|| ctx.scope().namespace.clone())
        .unwrap_or_else(|| "default".to_string());

    Ok(vec![PromptMessage {
        role: "user".into(),
        content: PromptContent::Text {
            text: format!(
                 "Generate a daily briefing for my Helix knowledge base.\n\n\
				 ## Vault Stats\n\
				 - Total nodes: {total_nodes}\n\
				 - Namespace: {namespace_label}\n\
                 - Stats truncated: {truncated}\n\n\
				 ## Recent Activity\n{}\n\n\
				 ## Open Tasks\n{}\n\n\
				 Please provide:\n\
				 1. A quick summary of recent activity\n\
				 2. Key tasks to focus on today\n\
				 3. Any patterns or connections you notice\n\
				 4. Suggested priorities for the day",
                if recent_summary.is_empty() {
                    "No recent activity.".to_string()
                } else {
                    recent_summary.join("\n")
                },
                if open_tasks.is_empty() {
                    "No open tasks.".to_string()
                } else {
                    open_tasks.join("\n")
                },
            ),
        },
    }])
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_len).collect();
        format!("{truncated}...")
    }
}
