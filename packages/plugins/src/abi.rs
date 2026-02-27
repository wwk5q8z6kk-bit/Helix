//! Host functions exposed to WASM plugins via the ABI boundary.

use serde::{Deserialize, Serialize};

/// Request from plugin to host.
#[derive(Debug, Serialize, Deserialize)]
pub struct HostRequest {
	pub method: String,
	pub params: serde_json::Value,
}

/// Response from host to plugin.
#[derive(Debug, Serialize, Deserialize)]
pub struct HostResponse {
	pub success: bool,
	pub data: serde_json::Value,
	pub error: Option<String>,
}

impl HostResponse {
	pub fn ok(data: serde_json::Value) -> Self {
		Self {
			success: true,
			data,
			error: None,
		}
	}

	pub fn err(msg: impl Into<String>) -> Self {
		Self {
			success: false,
			data: serde_json::Value::Null,
			error: Some(msg.into()),
		}
	}
}

/// Available host methods.
pub const HOST_READ_NODE: &str = "hx_read_node";
pub const HOST_WRITE_NODE: &str = "hx_write_node";
pub const HOST_SEARCH: &str = "hx_search";
pub const HOST_LOG: &str = "hx_log";
