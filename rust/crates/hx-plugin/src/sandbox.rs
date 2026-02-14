//! Permission gate for WASM plugin sandboxing.

use crate::manifest::PluginPermission;

/// Gates host-function calls against the plugin's declared permissions.
pub struct PermissionGate {
	permissions: Vec<PluginPermission>,
}

impl PermissionGate {
	pub fn new(permissions: Vec<PluginPermission>) -> Self {
		Self { permissions }
	}

	/// Check whether `action` is allowed by the plugin's permissions.
	pub fn check(&self, action: &str) -> Result<(), String> {
		let required = match action {
			"hx_read_node" => PluginPermission::ReadNodes,
			"hx_search" => PluginPermission::Search,
			"hx_write_node" => PluginPermission::WriteNodes,
			"hx_log" => return Ok(()), // logging is always allowed
			_ => return Err(format!("unknown action: {action}")),
		};
		if self.permissions.contains(&required) {
			Ok(())
		} else {
			Err(format!(
				"permission denied: {required:?} required for {action}"
			))
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn log_always_allowed() {
		let gate = PermissionGate::new(vec![]);
		assert!(gate.check("hx_log").is_ok());
	}

	#[test]
	fn read_requires_permission() {
		let gate = PermissionGate::new(vec![]);
		assert!(gate.check("hx_read_node").is_err());

		let gate = PermissionGate::new(vec![PluginPermission::ReadNodes]);
		assert!(gate.check("hx_read_node").is_ok());
	}

	#[test]
	fn write_requires_permission() {
		let gate = PermissionGate::new(vec![PluginPermission::ReadNodes]);
		assert!(gate.check("hx_write_node").is_err());

		let gate = PermissionGate::new(vec![PluginPermission::WriteNodes]);
		assert!(gate.check("hx_write_node").is_ok());
	}

	#[test]
	fn search_requires_permission() {
		let gate = PermissionGate::new(vec![]);
		assert!(gate.check("hx_search").is_err());

		let gate = PermissionGate::new(vec![PluginPermission::Search]);
		assert!(gate.check("hx_search").is_ok());
	}

	#[test]
	fn unknown_action_rejected() {
		let gate = PermissionGate::new(vec![
			PluginPermission::ReadNodes,
			PluginPermission::WriteNodes,
			PluginPermission::Search,
		]);
		assert!(gate.check("hx_delete_everything").is_err());
	}
}
