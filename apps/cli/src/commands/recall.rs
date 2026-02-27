use anyhow::Result;
use hx_core::*;

pub async fn run(
    query: String,
    limit: usize,
    strategy: String,
    min_score: f64,
    ns: Option<String>,
    config_path: &str,
) -> Result<()> {
    let engine = super::load_engine(config_path).await?;

    let strat: SearchStrategy = strategy.parse().map_err(|e: String| anyhow::anyhow!(e))?;

    let mut mq = MemoryQuery::new(&query)
        .with_strategy(strat)
        .with_limit(limit)
        .with_min_score(min_score);

    if let Some(namespace) = ns {
        mq = mq.with_namespace(namespace);
    }

    let results = engine.recall(&mq).await?;

    if results.is_empty() {
        println!("no results found");
        return Ok(());
    }

    println!("found {} result(s):\n", results.len());

    for (i, result) in results.iter().enumerate() {
        println!(
            "{}. [score: {:.4}] [{}] {}",
            i + 1,
            result.score,
            result.node.kind,
            result.node.id
        );
        if let Some(ref title) = result.node.title {
            println!("   title: {title}");
        }
        // Show first 200 chars of content
        let preview: String = result.node.content.chars().take(200).collect();
        println!("   {preview}");
        if !result.node.tags.is_empty() {
            println!("   tags: {}", result.node.tags.join(", "));
        }
        println!();
    }

    Ok(())
}
