use std::collections::HashMap;
use std::path::{Path, PathBuf};

use hx_core::*;

// ---------------------------------------------------------------------------
// Import stats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct ImportStats {
    pub files_scanned: usize,
    pub nodes_created: usize,
    pub nodes_updated: usize,
    pub nodes_skipped: usize,
    pub relationships_created: usize,
    pub errors: Vec<String>,
}

impl std::fmt::Display for ImportStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "scanned={}, created={}, updated={}, skipped={}, relationships={}, errors={}",
            self.files_scanned,
            self.nodes_created,
            self.nodes_updated,
            self.nodes_skipped,
            self.relationships_created,
            self.errors.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Parsed note
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ImportedNote {
    pub node: KnowledgeNode,
    pub wikilinks: Vec<String>,
    pub relative_path: String,
}

// ---------------------------------------------------------------------------
// Public API: parse a single file
// ---------------------------------------------------------------------------

/// Parse a single Obsidian markdown file into an `ImportedNote`.
pub fn parse_obsidian_note(
    file_path: &Path,
    vault_root: &Path,
    namespace: &str,
) -> MvResult<ImportedNote> {
    let relative = file_path
        .strip_prefix(vault_root)
        .map_err(|e| HxError::InvalidInput(format!("path outside vault root: {e}")))?;
    let relative_str = relative.to_string_lossy().to_string();

    let raw_content = std::fs::read_to_string(file_path)
        .map_err(|e| HxError::Storage(format!("read file {}: {e}", file_path.display())))?;

    let (frontmatter, body) = parse_frontmatter(&raw_content);

    // Determine node kind
    let kind = determine_kind(&frontmatter);

    // Build title from frontmatter or filename
    let title = frontmatter
        .get("title")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| {
            file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("untitled")
                .to_string()
        });

    // Build tags
    let mut tags = extract_frontmatter_tags(&frontmatter);

    // Add folder-based tags
    tags.extend(folder_tags(&relative_str));

    // Add "obsidian" origin tag
    if !tags.contains(&"obsidian".to_string()) {
        tags.push("obsidian".to_string());
    }

    // Deduplicate tags
    tags.sort();
    tags.dedup();

    // Extract wikilinks
    let wikilinks = extract_wikilinks(&body);

    // Build source identifier for dedup
    let source = format!("obsidian://{relative_str}");

    // File mtime for dedup
    let mtime = std::fs::metadata(file_path)
        .and_then(|m| m.modified())
        .ok()
        .map(|t| {
            let dt: chrono::DateTime<chrono::Utc> = t.into();
            dt.to_rfc3339()
        })
        .unwrap_or_default();

    // Build metadata from remaining frontmatter keys
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    metadata.insert("import_mtime".to_string(), serde_json::Value::String(mtime));
    metadata.insert(
        "import_source_path".to_string(),
        serde_json::Value::String(relative_str.clone()),
    );

    if let Some(aliases) = frontmatter.get("aliases") {
        metadata.insert("aliases".to_string(), aliases.clone());
    }

    // Store any extra frontmatter keys into metadata
    let reserved_keys = ["title", "tags", "aliases", "date", "type"];
    for (key, value) in &frontmatter {
        if !reserved_keys.contains(&key.as_str()) {
            metadata.insert(key.clone(), value.clone());
        }
    }

    let mut node = KnowledgeNode::new(kind, body)
        .with_title(&title)
        .with_source(source)
        .with_namespace(namespace)
        .with_tags(tags);

    node.metadata = metadata;

    Ok(ImportedNote {
        node,
        wikilinks,
        relative_path: relative_str,
    })
}

// ---------------------------------------------------------------------------
// Public API: import full vault
// ---------------------------------------------------------------------------

/// Import an entire Obsidian vault directory.
///
/// Steps:
/// 1. Walk all `.md` files (skip hidden dirs like `.obsidian`)
/// 2. Parse each into `ImportedNote`
/// 3. Dedup via source field (skip unchanged, update if mtime differs)
/// 4. Store new/updated nodes via the engine
/// 5. Resolve wikilinks into graph relationships
/// 6. Return stats
pub async fn import_obsidian_vault(
    vault_path: &Path,
    namespace: &str,
    engine: &crate::engine::HelixEngine,
    dry_run: bool,
) -> MvResult<ImportStats> {
    if !vault_path.is_dir() {
        return Err(HxError::InvalidInput(format!(
            "not a directory: {}",
            vault_path.display()
        )));
    }

    let mut stats = ImportStats::default();

    // Phase 1: Parse all notes
    let md_files = collect_md_files(vault_path);
    let mut imported_notes: Vec<ImportedNote> = Vec::new();

    for file_path in &md_files {
        stats.files_scanned += 1;
        match parse_obsidian_note(file_path, vault_path, namespace) {
            Ok(note) => imported_notes.push(note),
            Err(e) => {
                stats.errors.push(format!("{}: {e}", file_path.display()));
            }
        }
    }

    if dry_run {
        // In dry-run mode, report what would happen without storing
        for note in &imported_notes {
            let source = note.node.source.as_deref().unwrap_or("");
            let existing = engine.store.nodes.find_by_source(source).await?;
            match existing {
                Some(existing_node) => {
                    let existing_mtime = existing_node
                        .metadata
                        .get("import_mtime")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let new_mtime = note
                        .node
                        .metadata
                        .get("import_mtime")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if existing_mtime == new_mtime {
                        stats.nodes_skipped += 1;
                    } else {
                        stats.nodes_updated += 1;
                    }
                }
                None => {
                    stats.nodes_created += 1;
                }
            }
        }
        stats.relationships_created = imported_notes.iter().map(|n| n.wikilinks.len()).sum();
        return Ok(stats);
    }

    // Phase 2: Store nodes (with dedup)
    // Build a map of relative_path (stem, lowercased) -> node_id for link resolution
    let mut title_to_id: HashMap<String, uuid::Uuid> = HashMap::new();

    for note in &imported_notes {
        let source = note.node.source.as_deref().unwrap_or("");
        let existing = engine.store.nodes.find_by_source(source).await?;

        match existing {
            Some(existing_node) => {
                // Check mtime for dedup
                let existing_mtime = existing_node
                    .metadata
                    .get("import_mtime")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let new_mtime = note
                    .node
                    .metadata
                    .get("import_mtime")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                if existing_mtime == new_mtime {
                    stats.nodes_skipped += 1;
                    // Still record ID for link resolution
                    let stem = note_stem(&note.relative_path);
                    title_to_id.insert(stem, existing_node.id);
                } else {
                    // Update existing node
                    let mut updated = note.node.clone();
                    updated.id = existing_node.id;
                    updated.temporal = existing_node.temporal.clone();
                    updated.temporal.updated_at = chrono::Utc::now();
                    updated.temporal.version = existing_node.temporal.version + 1;
                    match engine.update_node(updated).await {
                        Ok(stored) => {
                            stats.nodes_updated += 1;
                            let stem = note_stem(&note.relative_path);
                            title_to_id.insert(stem, stored.id);
                        }
                        Err(e) => {
                            stats
                                .errors
                                .push(format!("update {}: {e}", note.relative_path));
                        }
                    }
                }
            }
            None => {
                // New node
                match engine.store_node(note.node.clone()).await {
                    Ok(stored) => {
                        stats.nodes_created += 1;
                        let stem = note_stem(&note.relative_path);
                        title_to_id.insert(stem, stored.id);
                    }
                    Err(e) => {
                        stats
                            .errors
                            .push(format!("create {}: {e}", note.relative_path));
                    }
                }
            }
        }
    }

    // Phase 3: Resolve wikilinks into relationships
    for note in &imported_notes {
        let source = note.node.source.as_deref().unwrap_or("");
        // Look up the node ID for this note
        let from_id = {
            let stem = note_stem(&note.relative_path);
            match title_to_id.get(&stem) {
                Some(id) => *id,
                None => {
                    // Try to find by source if not in the map yet
                    match engine.store.nodes.find_by_source(source).await? {
                        Some(n) => n.id,
                        None => continue,
                    }
                }
            }
        };

        for link_target in &note.wikilinks {
            let target_key = link_target.to_lowercase();
            if let Some(&to_id) = title_to_id.get(&target_key) {
                if from_id == to_id {
                    continue; // Skip self-links
                }
                let rel = Relationship::new(from_id, to_id, RelationKind::References);
                match engine.graph.add_relationship(&rel).await {
                    Ok(()) => {
                        stats.relationships_created += 1;
                    }
                    Err(e) => {
                        stats.errors.push(format!(
                            "link {} -> {}: {e}",
                            note.relative_path, link_target
                        ));
                    }
                }
            }
            // Unresolved links are silently skipped — target note may not exist
        }
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Collect all .md files from a directory, skipping hidden dirs/files.
fn collect_md_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|e: &walkdir::DirEntry| {
            // Skip hidden directories/files (e.g. .obsidian, .git, .trash)
            !e.file_name().to_string_lossy().starts_with('.')
        })
        .filter_map(|e: Result<walkdir::DirEntry, walkdir::Error>| e.ok())
    {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|ext| ext.to_str()) == Some("md") {
            files.push(path.to_path_buf());
        }
    }
    files
}

/// Extract wikilinks `[[target]]` and `[[target|display text]]` from markdown content.
/// Uses a character-based parser (no regex dependency).
fn extract_wikilinks(content: &str) -> Vec<String> {
    let mut links = Vec::new();
    let bytes = content.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i + 1 < len {
        if bytes[i] == b'[' && bytes[i + 1] == b'[' {
            i += 2; // skip opening [[
            let start = i;

            // Find the closing ]]
            while i + 1 < len && !(bytes[i] == b']' && bytes[i + 1] == b']') {
                i += 1;
            }

            if i + 1 < len {
                let inner = &content[start..i];
                // Handle [[target|display text]] — take the target part
                let target = match inner.split_once('|') {
                    Some((target, _display)) => target.trim(),
                    None => inner.trim(),
                };
                if !target.is_empty() {
                    // Normalize: strip heading anchors (e.g. "Note#heading" -> "Note")
                    let target = match target.split_once('#') {
                        Some((base, _anchor)) if !base.is_empty() => base,
                        _ => target,
                    };
                    let normalized = target.to_lowercase();
                    if !links.contains(&normalized) {
                        links.push(normalized);
                    }
                }
                i += 2; // skip closing ]]
            }
        } else {
            i += 1;
        }
    }

    links
}

/// Parse YAML frontmatter from markdown content.
/// Returns (frontmatter_map, content_without_frontmatter).
///
/// Handles simple `key: value` pairs and `key: [item1, item2]` arrays.
/// Does NOT add an external YAML crate — parses manually.
fn parse_frontmatter(content: &str) -> (HashMap<String, serde_json::Value>, String) {
    let mut metadata = HashMap::new();

    if !content.starts_with("---") {
        return (metadata, content.to_string());
    }

    // Split into frontmatter block and body
    let parts: Vec<&str> = content.splitn(3, "---").collect();
    if parts.len() < 3 {
        return (metadata, content.to_string());
    }

    let frontmatter = parts[1].trim();
    let body = parts[2].trim().to_string();

    // Track multiline list state
    let mut current_list_key: Option<String> = None;
    let mut current_list_items: Vec<serde_json::Value> = Vec::new();

    for line in frontmatter.lines() {
        // Check if this is a list continuation item (starts with "- ")
        let trimmed = line.trim();
        if trimmed.starts_with("- ") {
            if let Some(ref _key) = current_list_key {
                let item = trimmed.trim_start_matches("- ").trim().trim_matches('"');
                if !item.is_empty() {
                    current_list_items.push(serde_json::Value::String(item.to_string()));
                }
                continue;
            }
        }

        // If we were accumulating a list, flush it
        if let Some(key) = current_list_key.take() {
            if !current_list_items.is_empty() {
                metadata.insert(key, serde_json::Value::Array(current_list_items.clone()));
                current_list_items.clear();
            }
        }

        // Parse key: value lines
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_string();
            let value = value.trim();

            if value.is_empty() {
                // Could be start of a multiline list
                current_list_key = Some(key);
                current_list_items.clear();
                continue;
            }

            // Handle inline arrays: [item1, item2]
            if value.starts_with('[') && value.ends_with(']') {
                let inner = &value[1..value.len() - 1];
                let items: Vec<serde_json::Value> = inner
                    .split(',')
                    .map(|s| {
                        serde_json::Value::String(
                            s.trim().trim_matches('"').trim_matches('\'').to_string(),
                        )
                    })
                    .filter(|v| !v.as_str().unwrap_or("").is_empty())
                    .collect();
                metadata.insert(key, serde_json::Value::Array(items));
                continue;
            }

            // Handle booleans
            let value_str = value.trim_matches('"').trim_matches('\'');
            match value_str {
                "true" | "True" | "TRUE" => {
                    metadata.insert(key, serde_json::Value::Bool(true));
                }
                "false" | "False" | "FALSE" => {
                    metadata.insert(key, serde_json::Value::Bool(false));
                }
                _ => {
                    // Try number
                    if let Ok(num) = value_str.parse::<f64>() {
                        metadata.insert(key, serde_json::json!(num));
                    } else {
                        metadata.insert(key, serde_json::Value::String(value_str.to_string()));
                    }
                }
            }
        }
    }

    // Flush any remaining list
    if let Some(key) = current_list_key {
        if !current_list_items.is_empty() {
            metadata.insert(key, serde_json::Value::Array(current_list_items));
        }
    }

    (metadata, body)
}

/// Determine node kind from frontmatter.
fn determine_kind(frontmatter: &HashMap<String, serde_json::Value>) -> NodeKind {
    // Check `type` field
    if let Some(type_val) = frontmatter.get("type") {
        if let Some(type_str) = type_val.as_str() {
            let lower = type_str.to_lowercase();
            if lower == "todo" || lower == "task" {
                return NodeKind::Task;
            }
            if lower == "event" {
                return NodeKind::Event;
            }
            if lower == "project" {
                return NodeKind::Project;
            }
            if lower == "decision" {
                return NodeKind::Decision;
            }
            if lower == "procedure" {
                return NodeKind::Procedure;
            }
        }
    }

    // If there's a `date` key, treat as Event (daily note)
    if frontmatter.contains_key("date") {
        return NodeKind::Event;
    }

    NodeKind::Fact
}

/// Extract tags from frontmatter (handles both array and string formats).
fn extract_frontmatter_tags(frontmatter: &HashMap<String, serde_json::Value>) -> Vec<String> {
    let mut tags = Vec::new();
    if let Some(tags_val) = frontmatter.get("tags") {
        match tags_val {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    if let Some(s) = item.as_str() {
                        let trimmed = s.trim().to_string();
                        if !trimmed.is_empty() {
                            tags.push(trimmed);
                        }
                    }
                }
            }
            serde_json::Value::String(s) => {
                // Comma-separated string: "tag1, tag2"
                for part in s.split(',') {
                    let trimmed = part.trim().to_string();
                    if !trimmed.is_empty() {
                        tags.push(trimmed);
                    }
                }
            }
            _ => {}
        }
    }
    tags
}

/// Build folder-based tags from relative path.
/// `projects/foo/note.md` -> `["folder:projects", "folder:projects/foo"]`
fn folder_tags(relative_path: &str) -> Vec<String> {
    let path = Path::new(relative_path);
    let parent = match path.parent() {
        Some(p) if p.as_os_str().is_empty() => return Vec::new(),
        Some(p) => p,
        None => return Vec::new(),
    };

    let mut tags = Vec::new();
    let mut accumulated = String::new();

    for component in parent.components() {
        let segment = component.as_os_str().to_string_lossy();
        if accumulated.is_empty() {
            accumulated = segment.to_string();
        } else {
            accumulated = format!("{accumulated}/{segment}");
        }
        tags.push(format!("folder:{accumulated}"));
    }

    tags
}

/// Extract the file stem (lowercased) for use as a lookup key.
fn note_stem(relative_path: &str) -> String {
    Path::new(relative_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_wikilinks_simple() {
        let content = "Hello [[World]] and [[Another Note]]";
        let links = extract_wikilinks(content);
        assert_eq!(links, vec!["world", "another note"]);
    }

    #[test]
    fn test_extract_wikilinks_with_display_text() {
        let content = "See [[Target Note|display text]] for details";
        let links = extract_wikilinks(content);
        assert_eq!(links, vec!["target note"]);
    }

    #[test]
    fn test_extract_wikilinks_with_heading_anchor() {
        let content = "See [[Note#Section]] for heading link";
        let links = extract_wikilinks(content);
        assert_eq!(links, vec!["note"]);
    }

    #[test]
    fn test_extract_wikilinks_dedup() {
        let content = "[[Note]] and [[note]] again";
        let links = extract_wikilinks(content);
        assert_eq!(links, vec!["note"]);
    }

    #[test]
    fn test_extract_wikilinks_empty_and_nested() {
        let content = "[[]] and [[ ]] and normal text";
        let links = extract_wikilinks(content);
        assert!(links.is_empty());
    }

    #[test]
    fn test_parse_frontmatter_basic() {
        let content =
            "---\ntitle: My Note\ntags: [tag1, tag2]\ndate: 2024-01-15\n---\nBody content here.";
        let (fm, body) = parse_frontmatter(content);
        assert_eq!(fm.get("title").unwrap().as_str().unwrap(), "My Note");
        assert_eq!(fm.get("tags").unwrap().as_array().unwrap().len(), 2);
        assert_eq!(body, "Body content here.");
    }

    #[test]
    fn test_parse_frontmatter_multiline_tags() {
        let content = "---\ntags:\n- alpha\n- beta\n---\nContent";
        let (fm, body) = parse_frontmatter(content);
        let tags = fm.get("tags").unwrap().as_array().unwrap();
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].as_str().unwrap(), "alpha");
        assert_eq!(tags[1].as_str().unwrap(), "beta");
        assert_eq!(body, "Content");
    }

    #[test]
    fn test_parse_frontmatter_no_frontmatter() {
        let content = "Just regular content\nwith no frontmatter";
        let (fm, body) = parse_frontmatter(content);
        assert!(fm.is_empty());
        assert_eq!(body, content);
    }

    #[test]
    fn test_folder_tags() {
        let tags = folder_tags("projects/foo/note.md");
        assert_eq!(tags, vec!["folder:projects", "folder:projects/foo"]);
    }

    #[test]
    fn test_folder_tags_root_file() {
        let tags = folder_tags("note.md");
        assert!(tags.is_empty());
    }

    #[test]
    fn test_folder_tags_deep() {
        let tags = folder_tags("a/b/c/d.md");
        assert_eq!(tags, vec!["folder:a", "folder:a/b", "folder:a/b/c"]);
    }

    #[test]
    fn test_determine_kind_task() {
        let mut fm = HashMap::new();
        fm.insert("type".to_string(), serde_json::Value::String("task".to_string()));
        assert_eq!(determine_kind(&fm), NodeKind::Task);
    }

    #[test]
    fn test_determine_kind_todo() {
        let mut fm = HashMap::new();
        fm.insert("type".to_string(), serde_json::Value::String("todo".to_string()));
        assert_eq!(determine_kind(&fm), NodeKind::Task);
    }

    #[test]
    fn test_determine_kind_event_from_date() {
        let mut fm = HashMap::new();
        fm.insert(
            "date".to_string(),
            serde_json::Value::String("2024-01-15".to_string()),
        );
        assert_eq!(determine_kind(&fm), NodeKind::Event);
    }

    #[test]
    fn test_determine_kind_default_fact() {
        let fm = HashMap::new();
        assert_eq!(determine_kind(&fm), NodeKind::Fact);
    }

    #[test]
    fn test_note_stem() {
        assert_eq!(note_stem("projects/foo/My Note.md"), "my note");
        assert_eq!(note_stem("note.md"), "note");
    }

    #[test]
    fn test_extract_frontmatter_tags_string() {
        let mut fm = HashMap::new();
        fm.insert(
            "tags".to_string(),
            serde_json::Value::String("alpha, beta, gamma".to_string()),
        );
        let tags = extract_frontmatter_tags(&fm);
        assert_eq!(tags, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_parse_frontmatter_boolean_values() {
        let content = "---\ndraft: true\npublished: false\n---\nBody";
        let (fm, _body) = parse_frontmatter(content);
        assert_eq!(fm.get("draft").unwrap(), &serde_json::Value::Bool(true));
        assert_eq!(
            fm.get("published").unwrap(),
            &serde_json::Value::Bool(false)
        );
    }
}
