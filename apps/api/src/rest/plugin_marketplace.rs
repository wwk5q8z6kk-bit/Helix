//! REST handlers for plugin marketplace browsing.

use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};

use crate::auth::AuthContext;
use crate::state::AppState;

// --- DTOs ---

#[derive(Clone, Serialize, Deserialize)]
pub struct MarketplacePlugin {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub downloads: u64,
    pub tags: Vec<String>,
    pub installed: bool,
}

#[derive(Deserialize)]
pub struct SearchQuery {
    pub q: Option<String>,
    pub tag: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Serialize)]
pub struct InstallResponse {
    pub plugin_id: String,
    pub installed: bool,
    pub message: String,
}

/// Built-in catalog of available plugins.
fn builtin_catalog() -> Vec<MarketplacePlugin> {
    vec![
        MarketplacePlugin {
            id: "helix-markdown-export".into(),
            name: "Markdown Export".into(),
            description: "Export knowledge nodes as structured Markdown files".into(),
            version: "1.0.0".into(),
            author: "Helix Team".into(),
            downloads: 1250,
            tags: vec!["export".into(), "markdown".into()],
            installed: false,
        },
        MarketplacePlugin {
            id: "helix-github-sync".into(),
            name: "GitHub Sync".into(),
            description: "Two-way sync between Helix nodes and GitHub issues/discussions".into(),
            version: "0.9.0".into(),
            author: "Helix Team".into(),
            downloads: 890,
            tags: vec!["integration".into(), "github".into(), "sync".into()],
            installed: false,
        },
        MarketplacePlugin {
            id: "helix-calendar-view".into(),
            name: "Calendar View".into(),
            description: "Visual calendar view for tasks and daily notes".into(),
            version: "1.2.0".into(),
            author: "Community".into(),
            downloads: 2100,
            tags: vec!["ui".into(), "calendar".into(), "tasks".into()],
            installed: false,
        },
        MarketplacePlugin {
            id: "helix-ai-tagging".into(),
            name: "AI Auto-Tagging".into(),
            description: "Automatically tag nodes using LLM classification".into(),
            version: "0.8.0".into(),
            author: "Helix Team".into(),
            downloads: 670,
            tags: vec!["ai".into(), "tags".into(), "automation".into()],
            installed: false,
        },
        MarketplacePlugin {
            id: "helix-slack-bridge".into(),
            name: "Slack Bridge".into(),
            description: "Forward Slack messages to Helix and reply from knowledge base".into(),
            version: "1.1.0".into(),
            author: "Community".into(),
            downloads: 1800,
            tags: vec!["integration".into(), "slack".into(), "messaging".into()],
            installed: false,
        },
    ]
}

// --- Handlers ---

/// GET /api/v1/plugins/marketplace — list available plugins.
pub async fn list_marketplace_plugins(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    let mut plugins = builtin_catalog();
    let limit = params.limit.unwrap_or(50).min(200);

    // Reflect installed state
    let installed = state.installed_plugins.read().await;
    for plugin in &mut plugins {
        if installed.contains(&plugin.id) {
            plugin.installed = true;
        }
    }
    drop(installed);

    if let Some(ref tag) = params.tag {
        let tag_lower = tag.to_lowercase();
        plugins.retain(|p| p.tags.iter().any(|t| t.to_lowercase() == tag_lower));
    }

    if let Some(ref q) = params.q {
        let q_lower = q.to_lowercase();
        plugins.retain(|p| {
            p.name.to_lowercase().contains(&q_lower)
                || p.description.to_lowercase().contains(&q_lower)
                || p.tags.iter().any(|t| t.to_lowercase().contains(&q_lower))
        });
    }

    plugins.truncate(limit);
    Json(plugins).into_response()
}

/// GET /api/v1/plugins/marketplace/:id — get plugin details.
pub async fn get_marketplace_plugin(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let catalog = builtin_catalog();
    match catalog.into_iter().find(|p| p.id == id) {
        Some(mut plugin) => {
            if state.installed_plugins.read().await.contains(&plugin.id) {
                plugin.installed = true;
            }
            Json(plugin).into_response()
        }
        None => (StatusCode::NOT_FOUND, "plugin not found".to_string()).into_response(),
    }
}

/// POST /api/v1/plugins/marketplace/:id/install — install a plugin.
pub async fn install_marketplace_plugin(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if !auth.is_admin() {
        return (StatusCode::FORBIDDEN, "admin permission required".to_string()).into_response();
    }

    let catalog = builtin_catalog();
    match catalog.iter().find(|p| p.id == id) {
        Some(plugin) => {
            let mut installed = state.installed_plugins.write().await;
            if installed.contains(&plugin.id) {
                return Json(InstallResponse {
                    plugin_id: plugin.id.clone(),
                    installed: true,
                    message: format!("Plugin '{}' is already installed", plugin.name),
                })
                .into_response();
            }
            installed.insert(plugin.id.clone());
            Json(InstallResponse {
                plugin_id: plugin.id.clone(),
                installed: true,
                message: format!(
                    "Plugin '{}' v{} installed successfully",
                    plugin.name, plugin.version
                ),
            })
            .into_response()
        }
        None => (StatusCode::NOT_FOUND, "plugin not found".to_string()).into_response(),
    }
}

/// GET /api/v1/plugins/marketplace/search — search plugins.
pub async fn search_marketplace_plugins(
    State(state): State<Arc<AppState>>,
    Query(params): Query<SearchQuery>,
) -> impl IntoResponse {
    // Delegate to list with search params
    list_marketplace_plugins(State(state), Query(params)).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_is_not_empty() {
        let catalog = builtin_catalog();
        assert!(!catalog.is_empty());
    }

    #[test]
    fn marketplace_plugin_serializes() {
        let plugin = MarketplacePlugin {
            id: "test".into(),
            name: "Test Plugin".into(),
            description: "A test".into(),
            version: "1.0.0".into(),
            author: "test".into(),
            downloads: 100,
            tags: vec!["test".into()],
            installed: false,
        };
        let json = serde_json::to_string(&plugin).unwrap();
        assert!(json.contains("Test Plugin"));
        assert!(json.contains("\"installed\":false"));
    }

    #[test]
    fn search_query_deserializes() {
        let q: SearchQuery = serde_urlencoded::from_str("q=github&tag=integration&limit=10").unwrap();
        assert_eq!(q.q.as_deref(), Some("github"));
        assert_eq!(q.tag.as_deref(), Some("integration"));
        assert_eq!(q.limit, Some(10));
    }

    #[test]
    fn search_query_defaults() {
        let q: SearchQuery = serde_urlencoded::from_str("").unwrap();
        assert!(q.q.is_none());
        assert!(q.tag.is_none());
        assert!(q.limit.is_none());
    }

    #[test]
    fn install_response_serializes() {
        let resp = InstallResponse {
            plugin_id: "test".into(),
            installed: true,
            message: "ok".into(),
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"installed\":true"));
    }
}
