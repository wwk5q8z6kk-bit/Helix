use anyhow::Result;
use hx_core::*;

pub async fn run(config_path: &str) -> Result<()> {
    let engine = super::load_engine(config_path).await?;

    let total = engine.node_count().await?;
    println!("Helix Stats");
    println!("===============");
    println!("total nodes: {total}");

    // Count by kind
    let kinds = [
        NodeKind::Fact,
        NodeKind::Decision,
        NodeKind::Preference,
        NodeKind::Entity,
        NodeKind::CodeSnippet,
        NodeKind::Project,
        NodeKind::Conversation,
        NodeKind::Procedure,
        NodeKind::Observation,
        NodeKind::Bookmark,
    ];

    println!("\nby kind:");
    for kind in &kinds {
        let filters = QueryFilters {
            kinds: Some(vec![*kind]),
            ..Default::default()
        };
        let count = engine.store.nodes.count(&filters).await?;
        if count > 0 {
            println!("  {}: {count}", kind);
        }
    }

    // Data directory size
    let config = super::load_config(config_path)?;
    let data_dir = std::path::Path::new(&config.data_dir);
    if data_dir.exists() {
        let size = dir_size(data_dir);
        println!("\ndata size: {}", format_bytes(size));
    }

    Ok(())
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let ft = entry.file_type();
            if let Ok(ft) = ft {
                if ft.is_file() {
                    total += entry.metadata().map(|m| m.len()).unwrap_or(0);
                } else if ft.is_dir() {
                    total += dir_size(&entry.path());
                }
            }
        }
    }
    total
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.2} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.2} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} bytes")
    }
}
