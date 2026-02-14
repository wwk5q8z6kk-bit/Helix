use anyhow::{Context, Result};
use hx_core::*;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

pub async fn run(
    from: String,
    path: String,
    namespace: Option<String>,
    dry_run: bool,
    config_path: &str,
) -> Result<()> {
    let path = super::shellexpand(&path);
    let engine = super::load_engine(config_path).await?;
    let namespace = namespace.as_deref();

    match from.as_str() {
        "claude-memory" => import_claude_memory(&engine, &path, namespace, dry_run).await,
        "markdown" | "md" => import_markdown(&engine, &path, namespace, dry_run).await,
        "markdown-dir" | "md-dir" => import_markdown_dir(&engine, &path, namespace, dry_run).await,
        "text" | "txt" => import_text(&engine, &path, namespace, dry_run).await,
        "json" => import_json(&engine, &path, namespace, dry_run).await,
        "csv" => import_csv(&engine, &path, namespace, dry_run).await,
        "obsidian" => {
            let ns = default_namespace(namespace);
            import_obsidian_vault(&engine, &path, ns, dry_run).await
        }
        _ => {
            anyhow::bail!(
                "unknown import format: {from}. Supported: claude-memory, markdown, markdown-dir, text, json, csv, obsidian"
            );
        }
    }
}

async fn import_claude_memory(
    engine: &hx_engine::engine::HelixEngine,
    path: &str,
    namespace: Option<&str>,
    dry_run: bool,
) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    let namespace = default_namespace(namespace);
    let mut count = 0;

    // Parse markdown sections as separate knowledge nodes
    let mut current_section = String::new();
    let mut current_title = String::new();

    for line in content.lines() {
        if line.starts_with("## ") {
            // Save previous section
            if !current_section.trim().is_empty() {
                let node = KnowledgeNode::new(NodeKind::Fact, current_section.trim().to_string())
                    .with_title(&current_title)
                    .with_source(format!("import:claude-memory:{path}"))
                    .with_namespace(namespace)
                    .with_tags(vec!["imported".into(), "claude-memory".into()]);

                if !dry_run {
                    engine.store_node(node).await?;
                }
                count += 1;
            }
            current_title = line.trim_start_matches('#').trim().to_string();
            current_section = String::new();
        } else {
            current_section.push_str(line);
            current_section.push('\n');
        }
    }

    // Save last section
    if !current_section.trim().is_empty() {
        let node = KnowledgeNode::new(NodeKind::Fact, current_section.trim().to_string())
            .with_title(&current_title)
            .with_source(format!("import:claude-memory:{path}"))
            .with_namespace(namespace)
            .with_tags(vec!["imported".into(), "claude-memory".into()]);

        if !dry_run {
            engine.store_node(node).await?;
        }
        count += 1;
    }

    if dry_run {
        println!("DRY RUN: would import {count} nodes from Claude memory ({path})");
    } else {
        println!("imported {count} nodes from Claude memory ({path})");
    }
    Ok(())
}

async fn import_markdown(
    engine: &hx_engine::engine::HelixEngine,
    path: &str,
    namespace: Option<&str>,
    dry_run: bool,
) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown");
    let namespace = default_namespace(namespace);

    let node = KnowledgeNode::new(NodeKind::Fact, content)
        .with_title(filename)
        .with_source(format!("import:markdown:{path}"))
        .with_namespace(namespace)
        .with_tags(vec!["imported".into(), "markdown".into()]);

    if dry_run {
        println!("DRY RUN: would import 1 Markdown node ({filename})");
    } else {
        let stored = engine.store_node(node).await?;
        println!("imported: {} ({})", stored.id, filename);
    }
    Ok(())
}

async fn import_text(
    engine: &hx_engine::engine::HelixEngine,
    path: &str,
    namespace: Option<&str>,
    dry_run: bool,
) -> Result<()> {
    let content = std::fs::read_to_string(path)?;
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("unknown");
    let namespace = default_namespace(namespace);

    let node = KnowledgeNode::new(NodeKind::Observation, content)
        .with_title(filename)
        .with_source(format!("import:text:{path}"))
        .with_namespace(namespace);

    if dry_run {
        println!("DRY RUN: would import 1 text node ({filename})");
    } else {
        let stored = engine.store_node(node).await?;
        println!("imported: {} ({})", stored.id, filename);
    }
    Ok(())
}

/// Import from Helix JSON export format.
async fn import_json(
    engine: &hx_engine::engine::HelixEngine,
    path: &str,
    namespace: Option<&str>,
    dry_run: bool,
) -> Result<()> {
    #[derive(Deserialize)]
    struct JsonExport {
        nodes: Vec<JsonNode>,
    }

    #[derive(Deserialize)]
    struct JsonNode {
        kind: String,
        title: Option<String>,
        content: String,
        namespace: String,
        importance: f64,
        tags: Vec<String>,
        source: Option<String>,
        #[serde(default)]
        metadata: HashMap<String, serde_json::Value>,
    }

    let content = std::fs::read_to_string(path).context("Failed to read JSON file")?;
    let export: JsonExport = serde_json::from_str(&content).context("Failed to parse JSON")?;

    if dry_run {
        println!("DRY RUN: would import {} nodes from JSON", export.nodes.len());
        return Ok(());
    }

    println!("Importing {} nodes from JSON...", export.nodes.len());

    let mut count = 0;
    for json_node in export.nodes {
        let kind = parse_node_kind(&json_node.kind);
        let source = json_node
            .source
            .unwrap_or_else(|| format!("import:json:{path}"));
        let namespace_override = normalize_namespace_override(namespace);
        let namespace = if json_node.namespace.trim().is_empty() {
            namespace_override.unwrap_or("imported")
        } else {
            json_node.namespace.as_str()
        };

        let mut node = KnowledgeNode::new(kind, json_node.content)
            .with_namespace(namespace)
            .with_importance(json_node.importance)
            .with_tags(json_node.tags)
            .with_source(source);

        if let Some(title) = json_node.title {
            node = node.with_title(&title);
        }

        node.metadata = json_node.metadata;

        engine.store_node(node).await?;
        count += 1;
    }

    println!("Imported {count} nodes from JSON");
    Ok(())
}

/// Import from CSV format.
async fn import_csv(
    engine: &hx_engine::engine::HelixEngine,
    path: &str,
    namespace: Option<&str>,
    dry_run: bool,
) -> Result<()> {
    let content = std::fs::read_to_string(path).context("Failed to read CSV file")?;
    let mut lines = content.lines();
    let default_namespace = default_namespace(namespace).to_string();

    // Skip header
    let header = lines.next().context("CSV file is empty")?;
    let columns: Vec<&str> = header.split(',').collect();

    // Find column indices
    let find_col = |name: &str| columns.iter().position(|&c| c.trim() == name);
    let kind_col = find_col("kind").unwrap_or(1);
    let title_col = find_col("title").unwrap_or(2);
    let namespace_col = find_col("namespace").unwrap_or(3);
    let importance_col = find_col("importance").unwrap_or(4);
    let tags_col = find_col("tags").unwrap_or(5);
    let content_col = find_col("content").unwrap_or(7);

    if dry_run {
        println!("DRY RUN: scanning CSV ({path})");
    } else {
        println!("Importing from CSV...");
    }

    let mut count = 0;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }

        let fields = parse_csv_line(line);
        if fields.len() <= content_col {
            continue;
        }

        let kind_str = fields.get(kind_col).map(|s| s.as_str()).unwrap_or("fact");
        let kind = parse_node_kind(kind_str);

        let content = fields.get(content_col).cloned().unwrap_or_default();
        if content.is_empty() {
            continue;
        }

        let title = fields.get(title_col).filter(|s| !s.is_empty()).cloned();
        let namespace = fields
            .get(namespace_col)
            .filter(|s| !s.is_empty())
            .cloned()
            .unwrap_or_else(|| default_namespace.clone());
        let importance: f64 = fields
            .get(importance_col)
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.5);
        let tags: Vec<String> = fields
            .get(tags_col)
            .map(|s| s.split(';').map(|t| t.trim().to_string()).collect())
            .unwrap_or_default();

        if dry_run {
            count += 1;
            continue;
        }

        let mut node = KnowledgeNode::new(kind, content)
            .with_namespace(&namespace)
            .with_importance(importance)
            .with_tags(tags)
            .with_source(format!("import:csv:{path}"));

        if let Some(t) = title {
            node = node.with_title(&t);
        }

        engine.store_node(node).await?;
        count += 1;
    }

    if dry_run {
        println!("DRY RUN: would import {count} nodes from CSV");
    } else {
        println!("Imported {count} nodes from CSV");
    }
    Ok(())
}

/// Import a directory of Markdown files.
async fn import_markdown_dir(
    engine: &hx_engine::engine::HelixEngine,
    path: &str,
    namespace: Option<&str>,
    dry_run: bool,
) -> Result<()> {
    let dir_path = Path::new(path);
    if !dir_path.is_dir() {
        anyhow::bail!("Path is not a directory: {path}");
    }

    if dry_run {
        println!("DRY RUN: scanning Markdown files from: {path}");
    } else {
        println!("Importing Markdown files from: {path}");
    }

    let mut count = 0;
    let mut files_scanned = 0;
    let base_namespace = normalize_namespace_override(namespace);
    for entry in walkdir::WalkDir::new(path)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let file_path = entry.path();
        if !file_path.is_file() {
            continue;
        }

        let ext = file_path.extension().and_then(|e| e.to_str());
        if ext != Some("md") && ext != Some("markdown") {
            continue;
        }

        files_scanned += 1;
        let content = std::fs::read_to_string(file_path)?;
        let filename = file_path
            .file_stem()
            .and_then(|f| f.to_str())
            .unwrap_or("unknown");

        // Extract namespace from subdirectory
        let relative_namespace = file_path
            .parent()
            .and_then(|p| p.strip_prefix(path).ok())
            .and_then(|p| p.to_str())
            .filter(|s| !s.is_empty())
            .map(|s| s.replace('/', "."));
        let namespace = match (base_namespace, relative_namespace) {
            (Some(base), Some(rel)) => format!("{base}.{rel}"),
            (Some(base), None) => base.to_string(),
            (None, Some(rel)) => rel,
            (None, None) => "imported".to_string(),
        };

        // Parse YAML frontmatter if present
        let (metadata, body) = parse_frontmatter(&content);

        let title = metadata
            .get("title")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| filename.to_string());

        let tags: Vec<String> = metadata
            .get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let importance: f64 = metadata
            .get("importance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);

        let node = KnowledgeNode::new(NodeKind::Fact, body)
            .with_title(&title)
            .with_namespace(&namespace)
            .with_importance(importance)
            .with_tags(tags)
            .with_source(format!("import:markdown-dir:{}", file_path.display()));

        if !dry_run {
            engine.store_node(node).await?;
        }
        count += 1;
    }

    if dry_run {
        println!("DRY RUN: scanned {files_scanned} files, would import {count} nodes");
    } else {
        println!("Imported {count} Markdown files");
    }
    Ok(())
}

/// Import from Obsidian vault using the engine-level importer.
async fn import_obsidian_vault(
    engine: &hx_engine::engine::HelixEngine,
    path: &str,
    namespace: &str,
    dry_run: bool,
) -> Result<()> {
    let vault_path = Path::new(path);

    if dry_run {
        println!("DRY RUN: Scanning Obsidian vault at: {path}");
    } else {
        println!("Importing Obsidian vault from: {path}");
    }

    let stats =
        hx_engine::import::obsidian::import_obsidian_vault(vault_path, namespace, engine, dry_run)
            .await?;

    if dry_run {
        println!("\n--- Dry Run Results ---");
        println!("Files scanned:           {}", stats.files_scanned);
        println!("Would create:            {}", stats.nodes_created);
        println!("Would update:            {}", stats.nodes_updated);
        println!("Already up to date:      {}", stats.nodes_skipped);
        println!("Wikilink relationships:  {}", stats.relationships_created);
    } else {
        println!("\n--- Import Results ---");
        println!("Files scanned:           {}", stats.files_scanned);
        println!("Nodes created:           {}", stats.nodes_created);
        println!("Nodes updated:           {}", stats.nodes_updated);
        println!("Nodes skipped (up-to-date): {}", stats.nodes_skipped);
        println!("Relationships created:   {}", stats.relationships_created);
    }

    if !stats.errors.is_empty() {
        println!("\nErrors ({}):", stats.errors.len());
        for err in &stats.errors {
            println!("  - {err}");
        }
    }

    Ok(())
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    let chars: Vec<char> = line.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];

        if in_quotes {
            if c == '"' {
                if i + 1 < chars.len() && chars[i + 1] == '"' {
                    current.push('"');
                    i += 1;
                } else {
                    in_quotes = false;
                }
            } else {
                current.push(c);
            }
        } else if c == '"' {
            in_quotes = true;
        } else if c == ',' {
            fields.push(current.clone());
            current.clear();
        } else {
            current.push(c);
        }

        i += 1;
    }

    fields.push(current);
    fields
}

fn parse_frontmatter(content: &str) -> (HashMap<String, serde_json::Value>, String) {
    let mut metadata = HashMap::new();

    if !content.starts_with("---") {
        return (metadata, content.to_string());
    }

    let parts: Vec<&str> = content.splitn(3, "---").collect();
    if parts.len() < 3 {
        return (metadata, content.to_string());
    }

    let frontmatter = parts[1].trim();
    let body = parts[2].trim();

    // Simple YAML parsing for common fields
    for line in frontmatter.lines() {
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim();
            let value = value.trim().trim_matches('"');

            if key == "tags" {
                // Handle tags: [tag1, tag2] or tags:\n- tag1\n- tag2
                if value.starts_with('[') {
                    let tags: Vec<serde_json::Value> = value
                        .trim_start_matches('[')
                        .trim_end_matches(']')
                        .split(',')
                        .map(|s| serde_json::Value::String(s.trim().to_string()))
                        .collect();
                    metadata.insert(key.to_string(), serde_json::Value::Array(tags));
                }
            } else if let Ok(num) = value.parse::<f64>() {
                metadata.insert(key.to_string(), serde_json::Value::from(num));
            } else {
                metadata.insert(
                    key.to_string(),
                    serde_json::Value::String(value.to_string()),
                );
            }
        }
    }

    (metadata, body.to_string())
}

fn parse_node_kind(kind: &str) -> NodeKind {
    let normalized = kind.trim().to_ascii_lowercase().replace('-', "_");
    match normalized.as_str() {
        // Legacy values from older exports.
        "belief" => NodeKind::Fact,
        "goal" => NodeKind::Task,
        "episode" => NodeKind::Event,
        _ => normalized.parse::<NodeKind>().unwrap_or(NodeKind::Fact),
    }
}

fn normalize_namespace_override(namespace: Option<&str>) -> Option<&str> {
    namespace.filter(|value| !value.trim().is_empty())
}

fn default_namespace(namespace: Option<&str>) -> &str {
    normalize_namespace_override(namespace).unwrap_or("imported")
}
