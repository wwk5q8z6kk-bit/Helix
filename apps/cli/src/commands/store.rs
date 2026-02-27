use anyhow::Result;
use hx_core::*;

#[allow(clippy::too_many_arguments)]
pub async fn run(
    content: String,
    kind: String,
    title: Option<String>,
    source: Option<String>,
    tags: Vec<String>,
    ns: String,
    importance: f64,
    config_path: &str,
) -> Result<()> {
    let engine = super::load_engine(config_path).await?;

    let node_kind: NodeKind = kind.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    let mut node = KnowledgeNode::new(node_kind, content);
    if let Some(t) = title {
        node = node.with_title(t);
    }
    if let Some(s) = source {
        node = node.with_source(s);
    }
    if !tags.is_empty() {
        node = node.with_tags(tags);
    }
    node = node.with_namespace(ns);
    node = node.with_importance(importance);

    let stored = engine.store_node(node).await?;

    println!("stored: {}", stored.id);
    println!("  kind: {}", stored.kind);
    if let Some(ref title) = stored.title {
        println!("  title: {title}");
    }
    println!("  namespace: {}", stored.namespace);
    if !stored.tags.is_empty() {
        println!("  tags: {}", stored.tags.join(", "));
    }
    println!("  importance: {:.2}", stored.importance);

    Ok(())
}
