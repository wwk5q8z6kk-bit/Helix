//! Plugin manifest definition.

use serde::{Deserialize, Serialize};

/// Plugin manifest (loaded from manifest.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub id: String,
    pub name: String,
    pub version: String,
    pub description: Option<String>,
    pub author: Option<String>,
    pub permissions: Vec<PluginPermission>,
    pub hooks: Vec<String>,
    pub entry_point: Option<String>,

    // ── Community module fields ──────────────────────────────────
    /// Repository URL (e.g. "https://github.com/user/hx-plugin-foo")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repository: Option<String>,
    /// SPDX license identifier (e.g. "MIT", "Apache-2.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    /// Project homepage URL
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub homepage: Option<String>,
    /// SHA-256 hex digest of the WASM binary for integrity verification
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
    /// Minimum Helix version required (semver, e.g. "0.9.0")
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_helix_version: Option<String>,
    /// Discovery keywords for search/filtering
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub keywords: Vec<String>,
}

impl PluginManifest {
    pub fn new(id: impl Into<String>, name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            version: version.into(),
            description: None,
            author: None,
            permissions: Vec::new(),
            hooks: Vec::new(),
            entry_point: None,
            repository: None,
            license: None,
            homepage: None,
            checksum: None,
            min_helix_version: None,
            keywords: Vec::new(),
        }
    }
}

/// Permissions a plugin may request
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PluginPermission {
    ReadNodes,
    WriteNodes,
    Search,
    GraphAccess,
    NetworkAccess,
    FileSystemRead,
    FileSystemWrite,
}
