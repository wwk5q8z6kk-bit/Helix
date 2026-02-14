//! Structured audit logging for Helix.
//!
//! Provides comprehensive audit trails for compliance and debugging.
//! Logs: subject, role, namespace, action, resource, status, latency, request_id.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use axum::{extract::Request, middleware::Next, response::Response};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::auth::AuthContext;

const ENV_AUDIT_ENABLED: &str = "HELIX_AUDIT_ENABLED";
const ENV_AUDIT_FILE: &str = "HELIX_AUDIT_FILE";
const ENV_AUDIT_WEBHOOK: &str = "HELIX_AUDIT_WEBHOOK";
const ENV_AUDIT_CONSOLE: &str = "HELIX_AUDIT_CONSOLE";
const ENV_AUDIT_STORE_MAX_ENTRIES: &str = "HELIX_AUDIT_STORE_MAX_ENTRIES";

/// Audit log entry for a single API request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Unique request identifier for correlation.
    pub request_id: String,
    /// Timestamp of the request.
    pub timestamp: DateTime<Utc>,
    /// Subject (user ID) from auth context.
    pub subject: Option<String>,
    /// Role of the authenticated user.
    pub role: String,
    /// Namespace scope (if any).
    pub namespace: Option<String>,
    /// HTTP method.
    pub method: String,
    /// Request path.
    pub path: String,
    /// Extracted action name (e.g., "store_node", "recall", "delete_node").
    pub action: String,
    /// Resource ID if applicable (e.g., node ID).
    pub resource_id: Option<String>,
    /// HTTP status code of response.
    pub status_code: u16,
    /// Whether the request succeeded.
    pub success: bool,
    /// Request latency in milliseconds.
    pub latency_ms: u64,
    /// Additional context (error message, etc.).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Query parameters (sanitized).
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub query_params: HashMap<String, String>,
}

/// Configuration for audit logging.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub enabled: bool,
    pub file_path: Option<PathBuf>,
    pub webhook_url: Option<String>,
    pub console_output: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

impl AuditConfig {
    pub fn from_env() -> Self {
        let enabled = std::env::var(ENV_AUDIT_ENABLED)
            .map(|v| v.trim().eq_ignore_ascii_case("true") || v.trim() == "1")
            .unwrap_or(false);

        let file_path = std::env::var(ENV_AUDIT_FILE)
            .ok()
            .filter(|s| !s.trim().is_empty())
            .map(|s| shellexpand(&s))
            .map(PathBuf::from);

        let webhook_url = std::env::var(ENV_AUDIT_WEBHOOK)
            .ok()
            .filter(|s| !s.trim().is_empty());

        let console_output = std::env::var(ENV_AUDIT_CONSOLE)
            .map(|v| v.trim().eq_ignore_ascii_case("true") || v.trim() == "1")
            .unwrap_or(false);

        Self {
            enabled,
            file_path,
            webhook_url,
            console_output,
        }
    }
}

fn shellexpand(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    s.to_string()
}

/// Global audit logger instance.
static AUDIT_LOGGER: OnceLock<AuditLogger> = OnceLock::new();
static AUDIT_STORE: OnceLock<AuditStore> = OnceLock::new();

/// Audit logger that writes to configured sinks.
pub struct AuditLogger {
    config: AuditConfig,
    file_writer: Option<Mutex<BufWriter<File>>>,
    http_client: Option<reqwest::Client>,
}

impl AuditLogger {
    pub fn init(config: AuditConfig) -> &'static Self {
        AUDIT_LOGGER.get_or_init(|| {
            let file_writer = config.file_path.as_ref().and_then(|path| {
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .ok()
                    .map(|f| Mutex::new(BufWriter::new(f)))
            });

            let http_client = config.webhook_url.as_ref().map(|_| {
                reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(5))
                    .build()
                    .expect("failed to create HTTP client")
            });

            let max_entries = std::env::var(ENV_AUDIT_STORE_MAX_ENTRIES)
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(5_000);
            let _ = AUDIT_STORE.set(AuditStore::new(max_entries));

            Self {
                config,
                file_writer,
                http_client,
            }
        })
    }

    pub fn global() -> Option<&'static Self> {
        AUDIT_LOGGER.get()
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    pub fn log(&self, entry: &AuditEntry) {
        if !self.config.enabled {
            return;
        }

        let json = match serde_json::to_string(entry) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("failed to serialize audit entry: {e}");
                return;
            }
        };

        // Write to file
        if let Some(ref writer) = self.file_writer {
            if let Ok(mut guard) = writer.lock() {
                let _ = writeln!(guard, "{json}");
                let _ = guard.flush();
            }
        }

        // Log to console
        if self.config.console_output {
            tracing::info!(
                target: "helix::audit",
                request_id = %entry.request_id,
                subject = ?entry.subject,
                role = %entry.role,
                action = %entry.action,
                resource_id = ?entry.resource_id,
                status = entry.status_code,
                latency_ms = entry.latency_ms,
                "audit_log"
            );
        }

        // Send to webhook (async, fire-and-forget)
        if let (Some(ref client), Some(ref url)) = (&self.http_client, &self.config.webhook_url) {
            let client = client.clone();
            let url = url.clone();
            let json = json.clone();
            tokio::spawn(async move {
                let _ = client
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .body(json)
                    .send()
                    .await;
            });
        }

        if let Some(store) = AUDIT_STORE.get() {
            store.push(entry.clone());
        }
    }
}

pub fn list_audit_entries(
    limit: usize,
    offset: usize,
    subject: Option<&str>,
    action: Option<&str>,
    since: Option<DateTime<Utc>>,
) -> Vec<AuditEntry> {
    let Some(store) = AUDIT_STORE.get() else {
        return Vec::new();
    };

    if subject.is_none() && action.is_none() && since.is_none() {
        return store.list(limit, offset);
    }

    let query_limit = limit.saturating_add(offset).max(limit);
    let filtered = store.query(subject, action, since, query_limit);
    filtered.into_iter().skip(offset).take(limit).collect()
}

/// Extract action name from HTTP method and path.
fn extract_action(method: &str, path: &str) -> String {
    // Parse the path to determine the action
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();

    match (method, segments.as_slice()) {
        // Health & diagnostics
        ("GET", ["api", "v1", "health"]) => "health_check".into(),
        ("GET", ["api", "v1", "diagnostics", "embedding"]) => "embedding_diagnostics".into(),

        // Nodes CRUD
        ("POST", ["api", "v1", "nodes"]) => "store_node".into(),
        ("GET", ["api", "v1", "nodes"]) => "list_nodes".into(),
        ("GET", ["api", "v1", "nodes", _id]) => "get_node".into(),
        ("PUT", ["api", "v1", "nodes", _id]) => "update_node".into(),
        ("DELETE", ["api", "v1", "nodes", _id]) => "delete_node".into(),
        ("GET", ["api", "v1", "nodes", _id, "backlinks"]) => "get_node_backlinks".into(),

        // Recall & search
        ("POST", ["api", "v1", "recall"]) => "recall".into(),
        ("GET", ["api", "v1", "search"]) => "search".into(),

        // Graph
        ("POST", ["api", "v1", "graph", "relationships"]) => "add_relationship".into(),
        ("GET", ["api", "v1", "graph", "neighbors", _id]) => "get_neighbors".into(),

        // Daily notes
        ("GET", ["api", "v1", "daily-notes"]) => "list_daily_notes".into(),
        ("POST", ["api", "v1", "daily-notes", "ensure"]) => "ensure_daily_note".into(),

        // Calendar
        ("GET", ["api", "v1", "calendar", "items"]) => "list_calendar_items".into(),
        ("GET", ["api", "v1", "calendar", "ical"]) => "export_ical".into(),
        ("POST", ["api", "v1", "calendar", "ical", "import"]) => "import_ical".into(),

        // Tasks
        ("GET", ["api", "v1", "tasks", "due"]) => "list_due_tasks".into(),
        ("POST", ["api", "v1", "tasks", "prioritize"]) => "prioritize_tasks".into(),
        ("POST", ["api", "v1", "tasks", _id, "complete"]) => "complete_task".into(),
        ("POST", ["api", "v1", "tasks", _id, "reopen"]) => "reopen_task".into(),

        // Templates
        ("GET", ["api", "v1", "templates"]) => "list_templates".into(),
        ("POST", ["api", "v1", "templates"]) => "create_template".into(),
        ("DELETE", ["api", "v1", "templates", _id]) => "delete_template".into(),
        ("POST", ["api", "v1", "templates", _id, "instantiate"]) => "instantiate_template".into(),
        ("POST", ["api", "v1", "templates", _id, "duplicate"]) => "duplicate_template".into(),
        ("GET", ["api", "v1", "templates", _id, "versions"]) => "list_template_versions".into(),
        ("GET", ["api", "v1", "templates", _id, "versions", _vid]) => "get_template_version".into(),
        ("POST", ["api", "v1", "templates", _id, "versions", _vid, "restore"]) => {
            "restore_template_version".into()
        }

        // Template packs
        ("GET", ["api", "v1", "template-packs"]) => "list_template_packs".into(),
        ("POST", ["api", "v1", "template-packs", _pack, "install"]) => {
            "install_template_pack".into()
        }

        // Saved searches
        ("GET", ["api", "v1", "search", "saved"]) => "list_saved_searches".into(),
        ("POST", ["api", "v1", "search", "saved"]) => "create_saved_search".into(),
        ("PUT", ["api", "v1", "search", "saved", _id]) => "update_saved_search".into(),
        ("DELETE", ["api", "v1", "search", "saved", _id]) => "delete_saved_search".into(),
        ("POST", ["api", "v1", "search", "saved", _id, "run"]) => "run_saved_search".into(),

        // Files
        ("POST", ["api", "v1", "files", "upload"]) => "upload_file".into(),
        ("GET", ["api", "v1", "files"]) => "list_attachments_index".into(),
        ("GET", ["api", "v1", "files", _node_id]) => "list_attachments".into(),
        ("GET", ["api", "v1", "files", _node_id, _att_id, "chunks"]) => {
            "get_attachment_chunks".into()
        }
        ("POST", ["api", "v1", "files", _node_id, _att_id, "reindex"]) => {
            "reindex_attachment".into()
        }
        ("GET", ["api", "v1", "files", _node_id, _att_id]) => "download_attachment".into(),
        ("DELETE", ["api", "v1", "files", _node_id, _att_id]) => "delete_attachment".into(),

        // AI Assist
        ("POST", ["api", "v1", "assist", "completion"]) => "assist_completion".into(),
        ("POST", ["api", "v1", "assist", "autocomplete"]) => "assist_autocomplete".into(),
        ("POST", ["api", "v1", "assist", "links"]) => "assist_links".into(),
        ("POST", ["api", "v1", "assist", "transform"]) => "assist_transform".into(),

        // Export/Import
        ("GET", ["api", "v1", "export"]) => "export_bundle".into(),
        ("POST", ["api", "v1", "import"]) => "import_bundle".into(),

        // Audit
        ("GET", ["api", "v1", "audit"]) => "list_audit_logs".into(),

        // Metrics
        ("GET", ["metrics"]) => "metrics".into(),

        // Fallback
        _ => format!("{method}_{}", segments.join("_")),
    }
}

/// Extract resource ID from path if applicable.
fn extract_resource_id(path: &str) -> Option<String> {
    let segments: Vec<&str> = path.trim_matches('/').split('/').collect();

    match segments.as_slice() {
        // Node operations with ID
        ["api", "v1", "nodes", id] => Some(id.to_string()),
        ["api", "v1", "nodes", id, "backlinks"] => Some(id.to_string()),
        ["api", "v1", "tasks", id, ..] => Some(id.to_string()),
        ["api", "v1", "templates", id, ..] => Some(id.to_string()),
        ["api", "v1", "search", "saved", id, ..] => Some(id.to_string()),
        ["api", "v1", "files", node_id, ..] => Some(node_id.to_string()),
        ["api", "v1", "graph", "neighbors", id] => Some(id.to_string()),
        _ => None,
    }
}

/// Extract query parameters (sanitized - no sensitive data).
fn extract_query_params(uri: &axum::http::Uri) -> HashMap<String, String> {
    let mut params = HashMap::new();
    if let Some(query) = uri.query() {
        for pair in query.split('&') {
            if let Some((key, value)) = pair.split_once('=') {
                // Skip sensitive-looking parameters
                let key_lower = key.to_lowercase();
                if key_lower.contains("token")
                    || key_lower.contains("secret")
                    || key_lower.contains("password")
                    || key_lower.contains("key")
                {
                    params.insert(key.to_string(), "[REDACTED]".to_string());
                } else {
                    params.insert(key.to_string(), value.to_string());
                }
            }
        }
    }
    params
}

/// Audit logging middleware.
pub async fn audit_middleware(request: Request, next: Next) -> Response {
    let logger = match AuditLogger::global() {
        Some(l) if l.is_enabled() => l,
        _ => return next.run(request).await,
    };

    let start = Instant::now();
    let request_id = Uuid::now_v7().to_string();

    // Extract request info before passing to next
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    let query_params = extract_query_params(request.uri());

    // Get auth context (may not exist yet if auth middleware runs after)
    let auth_context = request.extensions().get::<AuthContext>().cloned();

    // Run the actual handler
    let response = next.run(request).await;

    // Build audit entry
    let status_code = response.status().as_u16();
    let success = response.status().is_success();
    let latency_ms = start.elapsed().as_millis() as u64;

    let (subject, role, namespace) = match auth_context {
        Some(ctx) => (
            ctx.subject,
            format!("{:?}", ctx.role).to_lowercase(),
            ctx.namespace,
        ),
        None => (None, "unknown".to_string(), None),
    };

    let entry = AuditEntry {
        request_id,
        timestamp: Utc::now(),
        subject,
        role,
        namespace,
        method: method.clone(),
        path: path.clone(),
        action: extract_action(&method, &path),
        resource_id: extract_resource_id(&path),
        status_code,
        success,
        latency_ms,
        error: if !success {
            Some(format!("HTTP {status_code}"))
        } else {
            None
        },
        query_params,
    };

    logger.log(&entry);

    response
}

/// In-memory audit log store for querying recent entries (optional).
pub struct AuditStore {
    entries: Mutex<Vec<AuditEntry>>,
    max_entries: usize,
}

impl AuditStore {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(Vec::with_capacity(max_entries)),
            max_entries,
        }
    }

    pub fn push(&self, entry: AuditEntry) {
        if let Ok(mut guard) = self.entries.lock() {
            if guard.len() >= self.max_entries {
                guard.remove(0);
            }
            guard.push(entry);
        }
    }

    pub fn list(&self, limit: usize, offset: usize) -> Vec<AuditEntry> {
        if let Ok(guard) = self.entries.lock() {
            guard
                .iter()
                .rev()
                .skip(offset)
                .take(limit)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    pub fn query(
        &self,
        subject: Option<&str>,
        action: Option<&str>,
        since: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Vec<AuditEntry> {
        if let Ok(guard) = self.entries.lock() {
            guard
                .iter()
                .rev()
                .filter(|e| {
                    if let Some(s) = subject {
                        if e.subject.as_deref() != Some(s) {
                            return false;
                        }
                    }
                    if let Some(a) = action {
                        if e.action != a {
                            return false;
                        }
                    }
                    if let Some(ts) = since {
                        if e.timestamp < ts {
                            return false;
                        }
                    }
                    true
                })
                .take(limit)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_action() {
        assert_eq!(extract_action("POST", "/api/v1/nodes"), "store_node");
        assert_eq!(extract_action("GET", "/api/v1/nodes/abc-123"), "get_node");
        assert_eq!(extract_action("POST", "/api/v1/recall"), "recall");
        assert_eq!(extract_action("GET", "/api/v1/health"), "health_check");
        assert_eq!(
            extract_action("POST", "/api/v1/tasks/abc/complete"),
            "complete_task"
        );
        assert_eq!(
            extract_action("GET", "/api/v1/files"),
            "list_attachments_index"
        );
        assert_eq!(
            extract_action("GET", "/api/v1/files/node-1/att-1/chunks"),
            "get_attachment_chunks"
        );
        assert_eq!(
            extract_action("POST", "/api/v1/files/node-1/att-1/reindex"),
            "reindex_attachment"
        );
        assert_eq!(
            extract_action("GET", "/api/v1/nodes/node-1/backlinks"),
            "get_node_backlinks"
        );
    }

    #[test]
    fn test_extract_resource_id() {
        assert_eq!(
            extract_resource_id("/api/v1/nodes/abc-123"),
            Some("abc-123".to_string())
        );
        assert_eq!(
            extract_resource_id("/api/v1/tasks/task-1/complete"),
            Some("task-1".to_string())
        );
        assert_eq!(
            extract_resource_id("/api/v1/nodes/abc-123/backlinks"),
            Some("abc-123".to_string())
        );
        assert_eq!(extract_resource_id("/api/v1/nodes"), None);
        assert_eq!(extract_resource_id("/api/v1/health"), None);
    }

    #[test]
    fn test_audit_store_limit() {
        let store = AuditStore::new(3);
        for i in 0..5 {
            store.push(AuditEntry {
                request_id: format!("req-{i}"),
                timestamp: Utc::now(),
                subject: Some("user".into()),
                role: "admin".into(),
                namespace: None,
                method: "GET".into(),
                path: "/test".into(),
                action: "test".into(),
                resource_id: None,
                status_code: 200,
                success: true,
                latency_ms: 10,
                error: None,
                query_params: HashMap::new(),
            });
        }

        let entries = store.list(10, 0);
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].request_id, "req-4");
        assert_eq!(entries[2].request_id, "req-2");
    }
}
