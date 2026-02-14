use anyhow::Result;
use hx_core::*;

pub async fn show(id: String, depth: usize, config_path: &str) -> Result<()> {
    let engine = super::load_engine(config_path).await?;

    let uuid = uuid::Uuid::parse_str(&id)?;

    // Show the node itself
    if let Some(node) = engine.get_node(uuid).await? {
        println!("node: {} ({})", node.id, node.kind);
        if let Some(ref title) = node.title {
            println!("  title: {title}");
        }
        println!(
            "  content: {}",
            node.content.chars().take(200).collect::<String>()
        );
    } else {
        println!("node {id} not found");
        return Ok(());
    }

    // Show neighbors
    let neighbors = engine.get_neighbors(uuid, depth).await?;
    if neighbors.is_empty() {
        println!("\nno neighbors within depth {depth}");
    } else {
        println!("\nneighbors (depth {depth}):");
        for nid in &neighbors {
            if let Ok(Some(node)) = engine.get_node(*nid).await {
                println!(
                    "  {} ({}) - {}",
                    nid,
                    node.kind,
                    node.title
                        .as_deref()
                        .unwrap_or(&node.content.chars().take(60).collect::<String>())
                );
            } else {
                println!("  {nid} (not found)");
            }
        }
    }

    Ok(())
}

pub async fn link(from: String, to: String, kind: String, config_path: &str) -> Result<()> {
    let engine = super::load_engine(config_path).await?;

    let from_uuid = uuid::Uuid::parse_str(&from)?;
    let to_uuid = uuid::Uuid::parse_str(&to)?;
    let rel_kind: RelationKind = kind.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    let rel = Relationship::new(from_uuid, to_uuid, rel_kind);
    let rel_id = rel.id;
    engine.add_relationship(rel).await?;

    println!("created relationship: {rel_id}");
    println!("  {} --[{}]--> {}", from, kind, to);

    Ok(())
}
