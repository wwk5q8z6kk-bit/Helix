//! Plugin lifecycle management: install, uninstall, and discovery.

use crate::manifest::PluginManifest;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use uuid::Uuid;

/// Manages plugin files on disk (install, uninstall, discovery).
#[derive(Clone)]
pub struct PluginManager {
	plugins_dir: PathBuf,
}

impl PluginManager {
	pub fn is_valid_plugin_name(name: &str) -> bool {
		let trimmed = name.trim();
		if trimmed.is_empty() || trimmed.len() > 64 {
			return false;
		}
		if trimmed.contains("..") {
			return false;
		}
		if trimmed.contains('/') || trimmed.contains('\\') {
			return false;
		}
		trimmed
			.chars()
			.all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
	}

	pub fn new(plugins_dir: PathBuf) -> Self {
		Self { plugins_dir }
	}

	pub fn plugins_dir(&self) -> &PathBuf {
		&self.plugins_dir
	}

	/// Install a plugin by writing its WASM bytes and manifest to disk.
	/// If the manifest contains a `checksum` field, the SHA-256 digest of the
	/// WASM binary is verified before installation proceeds.
	/// Returns a freshly generated plugin UUID.
	pub fn install(
		&self,
		name: &str,
		wasm_bytes: &[u8],
		manifest: &PluginManifest,
	) -> Result<Uuid, String> {
		if !Self::is_valid_plugin_name(name) {
			return Err("invalid plugin name".into());
		}

		// Verify WASM checksum if declared in manifest.
		if let Some(expected) = &manifest.checksum {
			let actual = Self::sha256_hex(wasm_bytes);
			if actual != expected.to_lowercase() {
				return Err(format!(
					"checksum mismatch: expected {expected}, got {actual}"
				));
			}
		}

		let id = Uuid::now_v7();
		let plugin_dir = self.plugins_dir.join(name);
		std::fs::create_dir_all(&plugin_dir)
			.map_err(|e| format!("failed to create plugin directory: {e}"))?;

		let wasm_path = plugin_dir.join("plugin.wasm");
		std::fs::write(&wasm_path, wasm_bytes)
			.map_err(|e| format!("failed to write WASM module: {e}"))?;

		let manifest_path = plugin_dir.join("manifest.json");
		let manifest_json =
			serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
		std::fs::write(&manifest_path, manifest_json)
			.map_err(|e| format!("failed to write manifest: {e}"))?;

		tracing::info!(plugin = name, uuid = %id, "Plugin installed");
		Ok(id)
	}

	/// Compute the SHA-256 hex digest of a byte slice.
	pub fn sha256_hex(data: &[u8]) -> String {
		let digest = Sha256::digest(data);
		digest.iter().map(|b| format!("{b:02x}")).collect()
	}

	/// Remove a plugin directory from disk.
	pub fn uninstall(&self, name: &str) -> Result<(), String> {
		if !Self::is_valid_plugin_name(name) {
			return Err("invalid plugin name".into());
		}
		let plugin_dir = self.plugins_dir.join(name);
		if plugin_dir.exists() {
			std::fs::remove_dir_all(&plugin_dir)
				.map_err(|e| format!("failed to remove plugin: {e}"))?;
			tracing::info!(plugin = name, "Plugin uninstalled");
		}
		Ok(())
	}

	/// Scan the plugins directory and return discovered manifest paths.
	pub fn discover(&self) -> Result<Vec<(String, PathBuf, PluginManifest)>, String> {
		let mut found = Vec::new();
		if !self.plugins_dir.exists() {
			return Ok(found);
		}
		for entry in std::fs::read_dir(&self.plugins_dir).map_err(|e| e.to_string())? {
			let entry = entry.map_err(|e| e.to_string())?;
			if entry.path().is_dir() {
				let manifest_path = entry.path().join("manifest.json");
				if manifest_path.exists() {
					let content =
						std::fs::read_to_string(&manifest_path).map_err(|e| e.to_string())?;
					let manifest: PluginManifest =
						serde_json::from_str(&content).map_err(|e| e.to_string())?;
					let name = entry
						.file_name()
						.to_string_lossy()
						.to_string();
					if !Self::is_valid_plugin_name(&name) {
						continue;
					}
					let wasm_path = entry.path().join("plugin.wasm");
					found.push((name, wasm_path, manifest));
				}
			}
		}
		Ok(found)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::manifest::PluginManifest;

	#[test]
	fn install_and_discover() {
		let tmp = std::env::temp_dir().join(format!("hx_plugin_test_{}", Uuid::now_v7()));
		let mgr = PluginManager::new(tmp.clone());

		let manifest = PluginManifest::new("test-plugin", "Test Plugin", "0.1.0");
		let wasm_bytes = b"\0asm fake module";

		let id = mgr.install("test-plugin", wasm_bytes, &manifest).unwrap();
		assert!(!id.is_nil());

		let discovered = mgr.discover().unwrap();
		assert_eq!(discovered.len(), 1);
		assert_eq!(discovered[0].0, "test-plugin");
		assert_eq!(discovered[0].2.name, "Test Plugin");

		mgr.uninstall("test-plugin").unwrap();
		let discovered = mgr.discover().unwrap();
		assert_eq!(discovered.len(), 0);

		// Clean up
		let _ = std::fs::remove_dir_all(&tmp);
	}

	#[test]
	fn uninstall_nonexistent_is_ok() {
		let tmp = std::env::temp_dir().join(format!("hx_plugin_test_ne_{}", Uuid::now_v7()));
		let mgr = PluginManager::new(tmp.clone());
		assert!(mgr.uninstall("no-such-plugin").is_ok());
		let _ = std::fs::remove_dir_all(&tmp);
	}

	#[test]
	fn checksum_verification_passes() {
		let tmp = std::env::temp_dir().join(format!("hx_plugin_test_cs_{}", Uuid::now_v7()));
		let mgr = PluginManager::new(tmp.clone());

		let wasm_bytes = b"\0asm checksum module";
		let checksum = PluginManager::sha256_hex(wasm_bytes);
		let mut manifest = PluginManifest::new("cs-plugin", "Checksum Plugin", "1.0.0");
		manifest.checksum = Some(checksum);

		let id = mgr.install("cs-plugin", wasm_bytes, &manifest).unwrap();
		assert!(!id.is_nil());

		let _ = std::fs::remove_dir_all(&tmp);
	}

	#[test]
	fn checksum_mismatch_rejects() {
		let tmp = std::env::temp_dir().join(format!("hx_plugin_test_csf_{}", Uuid::now_v7()));
		let mgr = PluginManager::new(tmp.clone());

		let wasm_bytes = b"\0asm checksum module";
		let mut manifest = PluginManifest::new("bad-cs", "Bad Checksum", "1.0.0");
		manifest.checksum = Some("0000000000000000000000000000000000000000000000000000000000000000".into());

		let result = mgr.install("bad-cs", wasm_bytes, &manifest);
		assert!(result.is_err());
		assert!(result.unwrap_err().contains("checksum mismatch"));

		let _ = std::fs::remove_dir_all(&tmp);
	}

	#[test]
	fn no_checksum_skips_verification() {
		let tmp = std::env::temp_dir().join(format!("hx_plugin_test_nocs_{}", Uuid::now_v7()));
		let mgr = PluginManager::new(tmp.clone());

		let wasm_bytes = b"\0asm no checksum";
		let manifest = PluginManifest::new("no-cs", "No Checksum", "1.0.0");

		let id = mgr.install("no-cs", wasm_bytes, &manifest).unwrap();
		assert!(!id.is_nil());

		let _ = std::fs::remove_dir_all(&tmp);
	}

	#[test]
	fn manifest_community_fields_serialize() {
		let mut manifest = PluginManifest::new("community-test", "Community Test", "2.0.0");
		manifest.repository = Some("https://github.com/user/hx-plugin".into());
		manifest.license = Some("MIT".into());
		manifest.homepage = Some("https://example.com".into());
		manifest.min_helix_version = Some("0.9.0".into());
		manifest.keywords = vec!["analytics".into(), "dashboard".into()];

		let json = serde_json::to_string(&manifest).unwrap();
		let parsed: PluginManifest = serde_json::from_str(&json).unwrap();

		assert_eq!(parsed.repository.as_deref(), Some("https://github.com/user/hx-plugin"));
		assert_eq!(parsed.license.as_deref(), Some("MIT"));
		assert_eq!(parsed.min_helix_version.as_deref(), Some("0.9.0"));
		assert_eq!(parsed.keywords, vec!["analytics", "dashboard"]);
	}

	#[test]
	fn manifest_without_community_fields_parses() {
		let json = r#"{"id":"old","name":"Old Plugin","version":"0.1.0","permissions":[],"hooks":[]}"#;
		let manifest: PluginManifest = serde_json::from_str(json).unwrap();
		assert!(manifest.repository.is_none());
		assert!(manifest.checksum.is_none());
		assert!(manifest.keywords.is_empty());
	}
}
