use std::sync::Arc;

use axum::{
    extract::{Multipart, Path, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::Serialize;

use crate::auth::{authorize_read, AuthContext};
use crate::state::AppState;

// --- DTOs ---

#[derive(Serialize)]
pub struct PluginSummary {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub hooks: Vec<String>,
    pub status: String,
}

#[derive(Serialize)]
pub struct HookPointInfo {
    pub name: String,
    pub description: String,
}

const MAX_MANIFEST_BYTES: usize = 256 * 1024;
const MAX_WASM_BYTES: usize = 20 * 1024 * 1024;

fn authorize_admin(auth: &AuthContext) -> Result<(), (StatusCode, String)> {
    if auth.is_admin() {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, "admin permission required".into()))
    }
}

fn validate_plugin_name(name: &str) -> Result<(), (StatusCode, String)> {
    if hx_plugin::PluginManager::is_valid_plugin_name(name) {
        Ok(())
    } else {
        Err((StatusCode::BAD_REQUEST, "invalid plugin name".into()))
    }
}

// --- Handlers ---

/// List registered plugins.
pub async fn list_plugins(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let registry = state.plugin_registry.read().await;
    let mut plugins: std::collections::HashMap<String, PluginSummary> = registry
        .list_plugins()
        .into_iter()
        .map(|m| {
            (
                m.id.clone(),
                PluginSummary {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    version: m.version.clone(),
                    description: m.description.clone(),
                    author: m.author.clone(),
                    hooks: m.hooks.clone(),
                    status: "loaded".to_string(),
                },
            )
        })
        .collect();

    let mgr = state.plugin_manager.read().await.clone();
    match tokio::task::spawn_blocking(move || mgr.discover()).await {
        Ok(Ok(discovered)) => {
            for (name, _path, manifest) in discovered {
                let entry = plugins.entry(manifest.id.clone()).or_insert_with(|| PluginSummary {
                    id: manifest.id.clone(),
                    name: manifest.name.clone(),
                    version: manifest.version.clone(),
                    description: manifest.description.clone(),
                    author: manifest.author.clone(),
                    hooks: manifest.hooks.clone(),
                    status: "installed".to_string(),
                });
                if entry.status != "loaded" {
                    entry.name = manifest.name.clone();
                    entry.version = manifest.version.clone();
                    entry.description = manifest.description.clone();
                    entry.author = manifest.author.clone();
                    entry.hooks = manifest.hooks.clone();
                    entry.status = "installed".to_string();
                }
                let _ = name;
            }
        }
        Ok(Err(err)) => {
            tracing::warn!(error = %err, "plugin discovery failed");
        }
        Err(err) => {
            tracing::warn!(error = %err, "plugin discovery task failed");
        }
    }

    let mut plugins: Vec<PluginSummary> = plugins.into_values().collect();
    plugins.sort_by(|a, b| a.name.cmp(&b.name));
    let count = plugins.len();
    Ok(Json(serde_json::json!({
        "plugins": plugins,
        "count": count,
    })))
}

/// List available hook points that plugins can register for.
pub async fn list_hook_points(
    Extension(auth): Extension<AuthContext>,
    State(_state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let hooks = vec![
        HookPointInfo {
            name: "pre_ingest".into(),
            description: "Before a node is ingested into the vault".into(),
        },
        HookPointInfo {
            name: "post_ingest".into(),
            description: "After a node is ingested".into(),
        },
        HookPointInfo {
            name: "pre_search".into(),
            description: "Before a search query is executed".into(),
        },
        HookPointInfo {
            name: "post_search".into(),
            description: "After search results are returned".into(),
        },
        HookPointInfo {
            name: "on_change".into(),
            description: "When a node is changed (created/updated/deleted)".into(),
        },
        HookPointInfo {
            name: "scheduled".into(),
            description: "On a scheduled interval".into(),
        },
        HookPointInfo {
            name: "on_intent".into(),
            description: "When an intent is detected by the watcher".into(),
        },
    ];

    Ok(Json(serde_json::json!({ "hooks": hooks })))
}

/// Install a plugin via multipart upload (manifest JSON + WASM binary).
pub async fn install_plugin(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let mut manifest_bytes: Option<Vec<u8>> = None;
    let mut wasm_bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("multipart error: {e}")))?
    {
        let name = field.name().unwrap_or("").to_string();
        let data = field
            .bytes()
            .await
            .map_err(|e| (StatusCode::BAD_REQUEST, format!("read field error: {e}")))?;
        match name.as_str() {
            "manifest" => {
                if data.len() > MAX_MANIFEST_BYTES {
                    return Err((StatusCode::PAYLOAD_TOO_LARGE, "manifest too large".into()));
                }
                manifest_bytes = Some(data.to_vec());
            }
            "wasm" => {
                if data.len() > MAX_WASM_BYTES {
                    return Err((StatusCode::PAYLOAD_TOO_LARGE, "wasm too large".into()));
                }
                wasm_bytes = Some(data.to_vec());
            }
            _ => {}
        }
    }

    let manifest_bytes = manifest_bytes
        .ok_or((StatusCode::BAD_REQUEST, "missing 'manifest' field".into()))?;
    let wasm_bytes =
        wasm_bytes.ok_or((StatusCode::BAD_REQUEST, "missing 'wasm' field".into()))?;

    let manifest: hx_plugin::PluginManifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid manifest JSON: {e}")))?;

    let plugin_name = manifest.id.clone();
    validate_plugin_name(&plugin_name)?;

    let mgr = state.plugin_manager.read().await.clone();
    let manifest_clone = manifest.clone();
    let wasm_clone = wasm_bytes.clone();
    let name_clone = plugin_name.clone();
    let id = tokio::task::spawn_blocking(move || {
        mgr.install(&name_clone, &wasm_clone, &manifest_clone)
    })
    .await
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("install task failed: {e}")))?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok((
        StatusCode::CREATED,
        Json(serde_json::json!({
            "id": id.to_string(),
            "name": plugin_name,
            "status": "installed",
        })),
    ))
}

/// Uninstall a plugin by name.
pub async fn uninstall_plugin(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;
    validate_plugin_name(&name)?;

    let mgr = state.plugin_manager.read().await.clone();
    let name_clone = name.clone();
    tokio::task::spawn_blocking(move || mgr.uninstall(&name_clone))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("uninstall task failed: {e}")))?
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Json(serde_json::json!({
        "name": name,
        "status": "uninstalled",
    })))
}

/// Reload / rediscover plugins from the plugins directory.
pub async fn reload_plugins(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let mut runtime = state.plugin_runtime.write().await;
    let loaded = runtime
        .scan_and_load()
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let plugins: Vec<String> = runtime
        .list_plugins()
        .into_iter()
        .map(|p| p.name)
        .collect();

    Ok(Json(serde_json::json!({
        "status": "reloaded",
        "count": loaded,
        "plugins": plugins,
    })))
}

/// GET /api/v1/plugins/runtime — list all loaded plugins from the runtime.
pub async fn runtime_list(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let runtime = state.plugin_runtime.read().await;
    let plugins = runtime.list_plugins();
    let count = plugins.len();

    Ok(Json(serde_json::json!({
        "plugins": plugins,
        "count": count,
    })))
}

/// GET /api/v1/plugins/runtime/:name — get info about a specific loaded plugin.
pub async fn runtime_get_plugin(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;
    validate_plugin_name(&name)?;

    let runtime = state.plugin_runtime.read().await;
    match runtime.get_plugin(&name) {
        Some(info) => Ok(Json(serde_json::json!(info))),
        None => Err((StatusCode::NOT_FOUND, format!("plugin '{name}' not loaded"))),
    }
}

/// POST /api/v1/plugins/runtime/:name/reload — reload a specific plugin.
pub async fn runtime_reload_plugin(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;
    validate_plugin_name(&name)?;

    let mut runtime = state.plugin_runtime.write().await;
    runtime
        .reload_plugin(&name)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let info = runtime.get_plugin(&name);
    Ok(Json(serde_json::json!({
        "status": "reloaded",
        "plugin": info,
    })))
}

/// DELETE /api/v1/plugins/runtime/:name — unload a plugin from the runtime.
pub async fn runtime_unload_plugin(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;
    validate_plugin_name(&name)?;

    let mut runtime = state.plugin_runtime.write().await;
    let unloaded = runtime.unload_plugin(&name);

    if unloaded {
        Ok(Json(serde_json::json!({
            "name": name,
            "status": "unloaded",
        })))
    } else {
        Err((StatusCode::NOT_FOUND, format!("plugin '{name}' not loaded")))
    }
}

/// GET /api/v1/plugins/runtime/:name/hooks — list hooks a plugin subscribes to.
pub async fn runtime_plugin_hooks(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;
    validate_plugin_name(&name)?;

    let runtime = state.plugin_runtime.read().await;
    match runtime.get_plugin_hooks(&name) {
        Some(hooks) => Ok(Json(serde_json::json!({ "plugin": name, "hooks": hooks }))),
        None => Err((StatusCode::NOT_FOUND, format!("plugin '{name}' not loaded"))),
    }
}
