use anyhow::Result;
use hx_core::*;

pub async fn run(
    query: String,
    limit: usize,
    search_type: String,
    config_path: &str,
) -> Result<()> {
    let engine = super::load_engine(config_path).await?;

    let strategy = match search_type.as_str() {
        "fulltext" | "full_text" | "fts" => SearchStrategy::FullText,
        "vector" | "vec" => SearchStrategy::Vector,
        "graph" => SearchStrategy::Graph,
        _ => SearchStrategy::Hybrid,
    };

    let mq = MemoryQuery::new(&query)
        .with_strategy(strategy)
        .with_limit(limit);

    let results = engine.recall(&mq).await?;

    if results.is_empty() {
        println!("no results found");
        return Ok(());
    }

    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. [{:.4}] {} | {} | {}",
            i + 1,
            result.score,
            result.node.id,
            result.node.kind,
            result.node.content.chars().take(100).collect::<String>()
        );
    }

    Ok(())
}
