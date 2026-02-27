//! Plugin registry: manages loaded plugins and dispatches hooks.

use std::collections::HashMap;
use std::sync::Arc;

use hx_core::MvResult;

use super::hooks::{HookContext, HookPoint, HookResult, Plugin};

/// Registry of loaded plugins, organized by hook points.
pub struct PluginRegistry {
    plugins: Vec<Arc<dyn Plugin>>,
    hook_index: HashMap<HookPoint, Vec<usize>>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            hook_index: HashMap::new(),
        }
    }

    /// Register a plugin and index its declared hooks.
    pub fn register(&mut self, plugin: Arc<dyn Plugin>) {
        let idx = self.plugins.len();
        let manifest = plugin.manifest();

        for hook_name in &manifest.hooks {
            if let Some(hook_point) = parse_hook_point(hook_name) {
                self.hook_index.entry(hook_point).or_default().push(idx);
            } else {
                tracing::warn!(hook = %hook_name, plugin = %manifest.id, "Unknown hook point");
            }
        }

        tracing::info!(
            plugin_id = %manifest.id,
            plugin_name = %manifest.name,
            hooks = manifest.hooks.len(),
            "Plugin registered"
        );
        self.plugins.push(plugin);
    }

    /// Execute all plugins registered for a given hook point.
    pub async fn dispatch(&self, context: &HookContext) -> MvResult<Vec<HookResult>> {
        let indices = match self.hook_index.get(&context.hook_point) {
            Some(indices) => indices,
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        for &idx in indices {
            let plugin = &self.plugins[idx];
            match plugin.execute_hook(context).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    let manifest = plugin.manifest();
                    tracing::error!(
                        plugin = %manifest.id,
                        error = %e,
                        "Plugin hook execution failed"
                    );
                    results.push(HookResult::error(e.to_string()));
                }
            }
        }

        Ok(results)
    }

    /// Get the number of registered plugins.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
    }

    /// List all registered plugin manifests.
    pub fn list_plugins(&self) -> Vec<&super::manifest::PluginManifest> {
        self.plugins.iter().map(|p| p.manifest()).collect()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_hook_point(s: &str) -> Option<HookPoint> {
    match s {
        "pre_ingest" => Some(HookPoint::PreIngest),
        "post_ingest" => Some(HookPoint::PostIngest),
        "pre_search" => Some(HookPoint::PreSearch),
        "post_search" => Some(HookPoint::PostSearch),
        "on_change" => Some(HookPoint::OnChange),
        "scheduled" => Some(HookPoint::Scheduled),
        "on_intent" => Some(HookPoint::OnIntent),
        _ => None,
    }
}
