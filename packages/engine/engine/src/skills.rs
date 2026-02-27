//! Skills registry — loads TOML skill manifests from ~/.helix/skills/.
//!
//! Each skill is a TOML file with metadata about a named capability.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

/// A skill manifest loaded from a TOML file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillManifest {
    /// Unique name of the skill.
    pub name: String,
    /// Semantic version.
    pub version: String,
    /// Human-readable description.
    pub description: String,
    /// The provider or author of the skill.
    pub provider: String,
    /// The entry point (e.g., module path, script, or command).
    pub entry_point: String,
    /// Arbitrary key-value parameters for the skill.
    #[serde(default)]
    pub parameters: HashMap<String, String>,
}

/// Registry of loaded skill manifests, keyed by name.
pub struct SkillsRegistry {
    skills: HashMap<String, SkillManifest>,
}

impl SkillsRegistry {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
        }
    }

    /// Load all `.toml` files from a directory and register them.
    ///
    /// Files that fail to parse are logged and skipped. If two skills share
    /// the same name, the later one wins (dedup by name).
    pub fn load_from_dir(dir: &Path) -> std::io::Result<Self> {
        let mut registry = Self::new();

        if !dir.exists() {
            return Ok(registry);
        }

        let entries = std::fs::read_dir(dir)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("toml") {
                match Self::load_manifest(&path) {
                    Ok(manifest) => {
                        registry.skills.insert(manifest.name.clone(), manifest);
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %path.display(),
                            error = %e,
                            "failed to load skill manifest"
                        );
                    }
                }
            }
        }

        Ok(registry)
    }

    /// Load a single skill manifest from a TOML file.
    fn load_manifest(path: &Path) -> Result<SkillManifest, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;
        toml::from_str(&content).map_err(|e| format!("parse error: {e}"))
    }

    /// Register a skill manifest directly.
    pub fn register(&mut self, manifest: SkillManifest) {
        self.skills.insert(manifest.name.clone(), manifest);
    }

    /// Get a skill manifest by name.
    pub fn get(&self, name: &str) -> Option<&SkillManifest> {
        self.skills.get(name)
    }

    /// List all registered skill manifests.
    pub fn list(&self) -> Vec<&SkillManifest> {
        self.skills.values().collect()
    }

    /// Number of registered skills.
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }
}

impl Default for SkillsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn load_from_toml_string() {
        let toml_str = r#"
name = "summarize"
version = "1.0.0"
description = "Summarize a document"
provider = "helix"
entry_point = "core.skills.summarize"

[parameters]
max_length = "500"
format = "markdown"
"#;
        let manifest: SkillManifest = toml::from_str(toml_str).unwrap();
        assert_eq!(manifest.name, "summarize");
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.provider, "helix");
        assert_eq!(manifest.entry_point, "core.skills.summarize");
        assert_eq!(manifest.parameters.get("max_length").unwrap(), "500");
        assert_eq!(manifest.parameters.get("format").unwrap(), "markdown");
    }

    #[test]
    fn registry_dedup_by_name() {
        let mut registry = SkillsRegistry::new();

        registry.register(SkillManifest {
            name: "search".into(),
            version: "1.0.0".into(),
            description: "old".into(),
            provider: "helix".into(),
            entry_point: "old.search".into(),
            parameters: HashMap::new(),
        });

        registry.register(SkillManifest {
            name: "search".into(),
            version: "2.0.0".into(),
            description: "new".into(),
            provider: "helix".into(),
            entry_point: "new.search".into(),
            parameters: HashMap::new(),
        });

        assert_eq!(registry.len(), 1);
        let skill = registry.get("search").unwrap();
        assert_eq!(skill.version, "2.0.0");
        assert_eq!(skill.description, "new");
    }

    #[test]
    fn load_from_dir_reads_toml_files() {
        let dir = tempfile::tempdir().unwrap();

        let manifest1 = r#"
name = "skill-a"
version = "1.0.0"
description = "Skill A"
provider = "test"
entry_point = "a"
"#;
        let manifest2 = r#"
name = "skill-b"
version = "0.1.0"
description = "Skill B"
provider = "test"
entry_point = "b"
"#;

        fs::write(dir.path().join("skill-a.toml"), manifest1).unwrap();
        fs::write(dir.path().join("skill-b.toml"), manifest2).unwrap();
        // Non-toml file should be ignored
        fs::write(dir.path().join("readme.md"), "# readme").unwrap();

        let registry = SkillsRegistry::load_from_dir(dir.path()).unwrap();
        assert_eq!(registry.len(), 2);
        assert!(registry.get("skill-a").is_some());
        assert!(registry.get("skill-b").is_some());
    }

    #[test]
    fn load_from_nonexistent_dir_returns_empty() {
        let registry =
            SkillsRegistry::load_from_dir(Path::new("/nonexistent/path/skills")).unwrap();
        assert!(registry.is_empty());
    }
}
