//! Export commands for Helix CLI.

use anyhow::{Context, Result};
use chrono::Utc;
use hx_core::traits::GraphStore;
use hx_core::QueryFilters;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::io::{BufWriter, Write};

use super::{load_engine, shellexpand};

#[derive(Serialize)]
struct ExportedNode {
    id: String,
    kind: String,
    title: Option<String>,
    content: String,
    namespace: String,
    importance: f64,
    tags: Vec<String>,
    source: Option<String>,
    metadata: HashMap<String, serde_json::Value>,
    created_at: String,
    updated_at: String,
}

#[derive(Serialize)]
struct ExportedEdge {
    from: String,
    to: String,
    kind: String,
    weight: f64,
}

#[derive(Serialize)]
struct ExportManifest {
    version: String,
    exported_at: String,
    node_count: usize,
    edge_count: usize,
    namespaces: Vec<String>,
}

#[derive(Serialize)]
struct JsonExport {
    manifest: ExportManifest,
    nodes: Vec<ExportedNode>,
    edges: Vec<ExportedEdge>,
}

/// Export data to JSON format.
pub async fn json(
    output: Option<String>,
    namespace: Option<String>,
    config_path: &str,
) -> Result<()> {
    let engine = load_engine(config_path).await?;

    println!("Exporting to JSON...");

    // Get all nodes
    let filters = QueryFilters {
        namespace: namespace.clone(),
        ..Default::default()
    };
    let nodes = engine
        .list_nodes(&filters, 1_000_000, 0)
        .await
        .context("Failed to list nodes")?;

    println!("Found {} nodes", nodes.len());

    // Collect outgoing edges for each node
    let mut edges: Vec<ExportedEdge> = Vec::new();
    for node in &nodes {
        if let Ok(rels) = engine.graph.get_relationships_from(node.id).await {
            for rel in rels {
                edges.push(ExportedEdge {
                    from: rel.from_node.to_string(),
                    to: rel.to_node.to_string(),
                    kind: rel.kind.to_string(),
                    weight: rel.weight,
                });
            }
        }
    }
    println!("Found {} edges", edges.len());

    // Collect namespaces
    let mut namespaces: Vec<String> = nodes
        .iter()
        .map(|n| n.namespace.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    namespaces.sort();

    // Build export
    let exported_nodes: Vec<ExportedNode> = nodes
        .into_iter()
        .map(|n| ExportedNode {
            id: n.id.to_string(),
            kind: n.kind.to_string(),
            title: n.title,
            content: n.content,
            namespace: n.namespace,
            importance: n.importance,
            tags: n.tags,
            source: n.source,
            metadata: n.metadata,
            created_at: n.temporal.created_at.to_rfc3339(),
            updated_at: n.temporal.updated_at.to_rfc3339(),
        })
        .collect();

    let manifest = ExportManifest {
        version: "1.0".into(),
        exported_at: Utc::now().to_rfc3339(),
        node_count: exported_nodes.len(),
        edge_count: edges.len(),
        namespaces,
    };

    let export = JsonExport {
        manifest,
        nodes: exported_nodes,
        edges,
    };

    // Generate output path
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let output_path = output
        .map(|p| shellexpand(&p))
        .unwrap_or_else(|| format!("helix_export_{timestamp}.json"));

    // Write to file
    let file = fs::File::create(&output_path)
        .with_context(|| format!("Failed to create: {output_path}"))?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &export).context("Failed to write JSON")?;

    let size = fs::metadata(&output_path)?.len();
    let size_str = format_size(size);

    println!();
    println!("Export complete:");
    println!("  Nodes: {}", export.manifest.node_count);
    println!("  Edges: {}", export.manifest.edge_count);
    println!("  Size:  {size_str}");
    println!("  Path:  {output_path}");

    Ok(())
}

/// Export data to Markdown format.
pub async fn markdown(
    output_dir: Option<String>,
    namespace: Option<String>,
    config_path: &str,
) -> Result<()> {
    let engine = load_engine(config_path).await?;

    println!("Exporting to Markdown...");

    // Get all nodes
    let filters = QueryFilters {
        namespace: namespace.clone(),
        ..Default::default()
    };
    let nodes = engine
        .list_nodes(&filters, 1_000_000, 0)
        .await
        .context("Failed to list nodes")?;

    println!("Found {} nodes", nodes.len());

    // Generate output directory
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let output_path = output_dir
        .map(|p| shellexpand(&p))
        .unwrap_or_else(|| format!("helix_export_{timestamp}"));

    // Create output directory
    fs::create_dir_all(&output_path).with_context(|| format!("Failed to create: {output_path}"))?;

    let mut file_count = 0;

    for node in nodes {
        // Create namespace subdirectory
        let ns_dir = format!("{output_path}/{}", node.namespace);
        fs::create_dir_all(&ns_dir)?;

        // Generate filename from title or ID
        let filename = if let Some(ref title) = node.title {
            sanitize_filename(title)
        } else {
            node.id.to_string()
        };
        let filepath = format!("{ns_dir}/{filename}.md");

        // Build markdown content
        let mut content = String::new();

        // YAML frontmatter
        content.push_str("---\n");
        content.push_str(&format!("id: {}\n", node.id));
        content.push_str(&format!("kind: {}\n", node.kind));
        if let Some(ref title) = node.title {
            content.push_str(&format!("title: \"{}\"\n", title.replace('"', "\\\"")));
        }
        content.push_str(&format!("namespace: {}\n", node.namespace));
        content.push_str(&format!("importance: {}\n", node.importance));
        if !node.tags.is_empty() {
            content.push_str(&format!("tags: [{}]\n", node.tags.join(", ")));
        }
        if let Some(ref source) = node.source {
            content.push_str(&format!("source: \"{}\"\n", source.replace('"', "\\\"")));
        }
        content.push_str(&format!(
            "created_at: {}\n",
            node.temporal.created_at.to_rfc3339()
        ));
        content.push_str(&format!(
            "updated_at: {}\n",
            node.temporal.updated_at.to_rfc3339()
        ));
        content.push_str("---\n\n");

        // Title as heading
        if let Some(ref title) = node.title {
            content.push_str(&format!("# {title}\n\n"));
        }

        // Content
        content.push_str(&node.content);
        content.push('\n');

        // Write file
        fs::write(&filepath, content).with_context(|| format!("Failed to write: {filepath}"))?;
        file_count += 1;
    }

    println!();
    println!("Export complete:");
    println!("  Files: {file_count}");
    println!("  Path:  {output_path}/");

    Ok(())
}

/// Export data to CSV format.
pub async fn csv(
    output: Option<String>,
    namespace: Option<String>,
    config_path: &str,
) -> Result<()> {
    let engine = load_engine(config_path).await?;

    println!("Exporting to CSV...");

    // Get all nodes
    let filters = QueryFilters {
        namespace: namespace.clone(),
        ..Default::default()
    };
    let nodes = engine
        .list_nodes(&filters, 1_000_000, 0)
        .await
        .context("Failed to list nodes")?;

    println!("Found {} nodes", nodes.len());

    // Generate output path
    let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
    let output_path = output
        .map(|p| shellexpand(&p))
        .unwrap_or_else(|| format!("helix_export_{timestamp}.csv"));

    // Write CSV
    let file = fs::File::create(&output_path)
        .with_context(|| format!("Failed to create: {output_path}"))?;
    let mut writer = BufWriter::new(file);

    // Header
    writeln!(
        writer,
        "id,kind,title,namespace,importance,tags,source,content,created_at,updated_at"
    )?;

    // Rows
    for node in &nodes {
        let title = node.title.as_deref().unwrap_or("");
        let source = node.source.as_deref().unwrap_or("");
        let tags = node.tags.join(";");

        writeln!(
            writer,
            "{},{},{},{},{},{},{},{},{},{}",
            node.id,
            node.kind,
            csv_escape(title),
            node.namespace,
            node.importance,
            csv_escape(&tags),
            csv_escape(source),
            csv_escape(&node.content),
            node.temporal.created_at.to_rfc3339(),
            node.temporal.updated_at.to_rfc3339(),
        )?;
    }

    writer.flush()?;

    let size = fs::metadata(&output_path)?.len();
    let size_str = format_size(size);

    println!();
    println!("Export complete:");
    println!("  Rows: {}", nodes.len());
    println!("  Size: {size_str}");
    println!("  Path: {output_path}");

    Ok(())
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .take(100) // Limit filename length
        .collect::<String>()
        .trim()
        .to_string()
}

fn format_size(size: u64) -> String {
    if size > 1024 * 1024 {
        format!("{:.1} MB", size as f64 / 1024.0 / 1024.0)
    } else if size > 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else {
        format!("{size} bytes")
    }
}
