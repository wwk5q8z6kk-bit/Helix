//! Database maintenance commands for Helix CLI.

use anyhow::{anyhow, Result};
use chrono::Utc;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::env;

use super::load_config;
use hx_core::{Embedder, NodeStore, QueryFilters};
use hx_engine::config::EngineConfig;
use hx_storage::vector::{
    KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder, LanceVectorStore, NoopEmbedder,
    OpenAiEmbedder,
};

const SQLITE_DB_FILE: &str = "helix.sqlite";

/// Run database vacuum to reclaim space.
pub async fn vacuum(config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;
    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");

    if !Path::new(&db_path).exists() {
        println!("Database not found: {db_path}");
        return Ok(());
    }

    let size_before = std::fs::metadata(&db_path)?.len();

    println!("Running VACUUM on database...");
    let conn = rusqlite::Connection::open(&db_path)?;
    conn.execute("VACUUM", [])?;

    let size_after = std::fs::metadata(&db_path)?.len();
    let saved = size_before.saturating_sub(size_after);

    println!();
    println!("Vacuum complete:");
    println!("  Before: {}", format_size(size_before));
    println!("  After:  {}", format_size(size_after));
    if saved > 0 {
        println!("  Saved:  {}", format_size(saved));
    }

    Ok(())
}

/// Check database integrity.
pub async fn check(config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;
    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");

    if !Path::new(&db_path).exists() {
        println!("Database not found: {db_path}");
        return Ok(());
    }

    println!("Checking database integrity...");
    let conn = rusqlite::Connection::open(&db_path)?;

    // Run integrity check
    let result: String = conn.query_row("PRAGMA integrity_check", [], |row| row.get(0))?;

    if result == "ok" {
        println!("Database integrity: OK");
    } else {
        println!("Database integrity issues found:");
        println!("{result}");
    }

    // Check foreign keys
    let fk_result = conn
        .prepare("PRAGMA foreign_key_check")?
        .query_map([], |_row| Ok(()))?
        .count() as i64;

    if fk_result == 0 {
        println!("Foreign key integrity: OK");
    } else {
        println!("Foreign key violations: {fk_result}");
    }

    Ok(())
}

/// Analyze database for query optimization.
pub async fn analyze(config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;
    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");

    if !Path::new(&db_path).exists() {
        println!("Database not found: {db_path}");
        return Ok(());
    }

    println!("Analyzing database for query optimization...");
    let conn = rusqlite::Connection::open(&db_path)?;
    conn.execute("ANALYZE", [])?;

    println!("Analysis complete. Query planner statistics updated.");

    Ok(())
}

/// Show database statistics.
pub async fn info(config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;
    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");

    if !Path::new(&db_path).exists() {
        println!("Database not found: {db_path}");
        return Ok(());
    }

    let conn = rusqlite::Connection::open(&db_path)?;

    println!("Database Information");
    println!("====================");
    println!();
    println!("Path: {db_path}");
    println!("Size: {}", format_size(std::fs::metadata(&db_path)?.len()));
    println!();

    // SQLite version
    let version: String = conn.query_row("SELECT sqlite_version()", [], |row| row.get(0))?;
    println!("SQLite version: {version}");

    // Page size and count
    let page_size: i64 = conn.query_row("PRAGMA page_size", [], |row| row.get(0))?;
    let page_count: i64 = conn.query_row("PRAGMA page_count", [], |row| row.get(0))?;
    let freelist_count: i64 = conn.query_row("PRAGMA freelist_count", [], |row| row.get(0))?;

    println!("Page size: {} bytes", page_size);
    println!("Total pages: {page_count}");
    println!("Free pages: {freelist_count}");

    println!();
    println!("Tables:");

    // List tables with row counts
    let mut stmt = conn.prepare(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )?;

    let tables: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .filter_map(|r| r.ok())
        .collect();

    for table in tables {
        let count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap_or(0);
        println!("  {table}: {count} rows");
    }

    println!();
    println!("Indexes:");

    let mut stmt = conn.prepare(
        "SELECT name, tbl_name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY tbl_name, name"
    )?;

    let indexes: Vec<(String, String)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .filter_map(|r| r.ok())
        .collect();

    for (name, table) in indexes {
        println!("  {name} on {table}");
    }

    // Check for WAL mode
    let journal_mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
    println!();
    println!("Journal mode: {journal_mode}");

    // Check data directory contents
    println!();
    println!("Data Directory Contents:");
    if let Ok(entries) = std::fs::read_dir(data_dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().to_string();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            println!("  {name}: {}", format_size(size));
        }
    }

    Ok(())
}

/// Rebuild indexes.
pub async fn reindex(config_path: &str) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = &config.data_dir;
    let db_path = format!("{data_dir}/{SQLITE_DB_FILE}");

    if !Path::new(&db_path).exists() {
        println!("Database not found: {db_path}");
        return Ok(());
    }

    println!("Rebuilding database indexes...");
    let conn = rusqlite::Connection::open(&db_path)?;
    conn.execute("REINDEX", [])?;

    println!("Index rebuild complete.");

    Ok(())
}

/// Rebuild LanceDB vector index from SQLite nodes.
pub async fn rebuild_vectors(
    config_path: &str,
    batch_size: usize,
    dry_run: bool,
    apply: bool,
    confirm: bool,
) -> Result<()> {
    let config = load_config(config_path)?;
    let data_dir = PathBuf::from(&config.data_dir);
    let sqlite_path = data_dir.join(SQLITE_DB_FILE);
    let lancedb_path = data_dir.join("lancedb");

    if !sqlite_path.exists() {
        println!("Database not found: {}", sqlite_path.display());
        return Ok(());
    }

    let safe_batch = batch_size.max(1);
    let timestamp = Utc::now().format("%Y%m%d%H%M%S").to_string();
    let rebuild_dir = data_dir.join(format!("lancedb-rebuild-{timestamp}"));

    let node_store = hx_storage::sqlite::SqliteNodeStore::open_read_only(&sqlite_path)?;
    let total_nodes = node_store.count(&QueryFilters::default()).await?;

    println!("Vector rebuild plan");
    println!("===================");
    println!("Data dir: {}", data_dir.display());
    println!("SQLite:   {}", sqlite_path.display());
    println!("LanceDB:  {}", lancedb_path.display());
    println!("Nodes:    {}", total_nodes);
    println!("Batch:    {}", safe_batch);
    println!("Output:   {}", rebuild_dir.display());
    println!();
    println!("Note: stop the Helix server before running a rebuild with --apply.");
    println!();

    if dry_run {
        println!("Dry run requested; no changes made.");
        return Ok(());
    }

    if apply && !confirm {
        return Err(anyhow!(
            "--apply requires --confirm to swap rebuilt index into place"
        ));
    }

    if rebuild_dir.exists() {
        return Err(anyhow!(
            "rebuild target already exists: {}",
            rebuild_dir.display()
        ));
    }

    println!("Initializing embedding provider...");
    let runtime = resolve_embedder_for_rebuild(&config);
    let nodes = std::sync::Arc::new(node_store);

    if runtime.fallback_to_noop {
        println!(
            "Warning: embedding provider fell back to noop (reason: {}). Rebuilt vectors will be zeroed.",
            runtime.reason.as_deref().unwrap_or("unknown")
        );
    } else {
        println!(
            "Embedding provider: {} ({}, dims={})",
            runtime.provider, runtime.model, runtime.dimensions
        );
        if let Some(reason) = &runtime.reason {
            println!("Note: {reason}");
        }
    }

    println!("Creating new LanceDB index...");
    let vectors = LanceVectorStore::open(&rebuild_dir, runtime.dimensions).await?;

    let mut offset = 0usize;
    let mut processed = 0usize;
    let filters = QueryFilters::default();

    while processed < total_nodes {
        let batch = nodes.list(&filters, safe_batch, offset).await?;
        if batch.is_empty() {
            break;
        }

        let texts: Vec<String> = batch.iter().map(|node| node.content.clone()).collect();
        let embeddings = runtime.embedder.embed_batch(&texts).await?;

        if embeddings.len() != batch.len() {
            return Err(anyhow!(
                "embedding batch size mismatch: expected {}, got {}",
                batch.len(),
                embeddings.len()
            ));
        }

        let mut items = Vec::with_capacity(batch.len());
        for (node, embedding) in batch.into_iter().zip(embeddings.into_iter()) {
            items.push((
                node.id,
                embedding,
                node.content,
                Some(node.namespace),
            ));
        }

        vectors.upsert_batch(&items).await?;

        processed += items.len();
        offset += items.len();
        println!("Indexed {processed}/{total_nodes} nodes...");
    }

    vectors.ensure_index().await?;
    println!("Vector rebuild complete.");

    if apply {
        let backup_dir = data_dir.join(format!("lancedb-backup-{timestamp}"));
        if lancedb_path.exists() {
            std::fs::rename(&lancedb_path, &backup_dir)?;
            println!("Existing index moved to {}", backup_dir.display());
        }
        std::fs::rename(&rebuild_dir, &lancedb_path)?;
        println!("New index activated at {}", lancedb_path.display());
    } else {
        println!("Rebuilt index left at {}", rebuild_dir.display());
        println!("Run again with --apply --confirm to swap it into place.");
    }

    Ok(())
}

fn format_size(size: u64) -> String {
    if size > 1024 * 1024 * 1024 {
        format!("{:.2} GB", size as f64 / 1024.0 / 1024.0 / 1024.0)
    } else if size > 1024 * 1024 {
        format!("{:.2} MB", size as f64 / 1024.0 / 1024.0)
    } else if size > 1024 {
        format!("{:.2} KB", size as f64 / 1024.0)
    } else {
        format!("{size} bytes")
    }
}

struct EmbedderRuntime {
    embedder: Arc<dyn Embedder>,
    provider: String,
    model: String,
    dimensions: usize,
    fallback_to_noop: bool,
    reason: Option<String>,
}

fn resolve_embedder_for_rebuild(config: &EngineConfig) -> EmbedderRuntime {
    let provider = config.embedding.provider.trim().to_ascii_lowercase();
    let configured_model = config.embedding.model.clone();
    let configured_dimensions = config.embedding.dimensions;

    let noop = |reason: Option<String>| EmbedderRuntime {
        embedder: Arc::new(NoopEmbedder::new(configured_dimensions)),
        provider: "noop".to_string(),
        model: "noop".to_string(),
        dimensions: configured_dimensions,
        fallback_to_noop: true,
        reason,
    };

    match provider.as_str() {
        "openai" => {
            let base_url = config
                .embedding
                .base_url
                .clone()
                .unwrap_or_else(|| "https://api.openai.com/v1".into());
            let api_key = env::var("OPENAI_API_KEY").ok();
            if api_key.is_none() && base_url.contains("api.openai.com") {
                return noop(Some(
                    "OPENAI_API_KEY not found; falling back to noop embeddings".to_string(),
                ));
            }
            let embedder = OpenAiEmbedder::for_compatible(
                base_url,
                api_key,
                config.embedding.model.clone(),
                configured_dimensions,
            );
            EmbedderRuntime {
                embedder: Arc::new(embedder),
                provider: "openai".to_string(),
                model: configured_model,
                dimensions: configured_dimensions,
                fallback_to_noop: false,
                reason: None,
            }
        }
        "openai-compatible" | "openai_compatible" => {
            let base_url = config
                .embedding
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:8080/v1".into());
            let api_key = env::var("HELIX_EMBEDDING_API_KEY")
                .ok()
                .or_else(|| env::var("OPENAI_API_KEY").ok());
            let embedder = OpenAiEmbedder::for_compatible(
                base_url,
                api_key,
                config.embedding.model.clone(),
                configured_dimensions,
            );
            EmbedderRuntime {
                embedder: Arc::new(embedder),
                provider: "openai-compatible".to_string(),
                model: configured_model,
                dimensions: configured_dimensions,
                fallback_to_noop: false,
                reason: None,
            }
        }
        "ollama" => {
            let base_url = config
                .embedding
                .base_url
                .clone()
                .unwrap_or_else(|| "http://localhost:11434/v1".into());
            let model = if config.embedding.model.starts_with("text-embedding-") {
                "nomic-embed-text".to_string()
            } else {
                config.embedding.model.clone()
            };
            let embedder = OpenAiEmbedder::for_ollama(
                Some(base_url),
                model.clone(),
                configured_dimensions,
            );
            let reason = if model != configured_model {
                Some(format!(
                    "model '{configured_model}' auto-mapped to '{model}' for ollama"
                ))
            } else {
                None
            };
            EmbedderRuntime {
                embedder: Arc::new(embedder),
                provider: "ollama".to_string(),
                model,
                dimensions: configured_dimensions,
                fallback_to_noop: false,
                reason,
            }
        }
        "local_fastembed" | "fastembed" | "local" => {
            let local_model = default_local_model_if_needed(&config.embedding.model);
            match KnowledgeVaultIndexNoteEmbeddingFastembedLocalEmbedder::try_new(&local_model) {
                Ok(embedder) => {
                    let dimensions = embedder.dimensions();
                    let reason = if local_model != configured_model {
                        Some(format!(
                            "model '{configured_model}' auto-mapped to '{local_model}' for local_fastembed"
                        ))
                    } else {
                        None
                    };
                    EmbedderRuntime {
                        embedder: Arc::new(embedder),
                        provider: "local_fastembed".to_string(),
                        model: local_model,
                        dimensions,
                        fallback_to_noop: false,
                        reason,
                    }
                }
                Err(err) => noop(Some(format!(
                    "local_fastembed initialization failed: {err}"
                ))),
            }
        }
        other => noop(Some(format!(
            "unknown embedding provider '{other}', falling back to noop"
        ))),
    }
}

fn default_local_model_if_needed(config_model: &str) -> String {
    if config_model.starts_with("text-embedding-") {
        "bge-small-en-v1.5".to_string()
    } else {
        config_model.to_string()
    }
}
