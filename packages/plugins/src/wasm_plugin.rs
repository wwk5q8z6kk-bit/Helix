//! WASM plugin loader and executor using wasmtime.

#[cfg(feature = "wasm-runtime")]
mod inner {
	use crate::hooks::{HookContext, HookResult};
	use crate::manifest::PluginManifest;
	use crate::sandbox::PermissionGate;
	use std::path::Path;
	use wasmtime::*;

	/// Default fuel budget per hook execution (~1M instructions).
	const DEFAULT_FUEL: u64 = 1_000_000;

	/// Maximum allowed result size from a plugin (1 MB).
	const MAX_RESULT_SIZE: usize = 1 << 20;

	/// A loaded WASM plugin with its compiled module and permission gate.
	pub struct WasmPlugin {
		manifest: PluginManifest,
		module: Module,
		gate: PermissionGate,
	}

	impl WasmPlugin {
		/// Create a wasmtime Engine configured for plugin execution with fuel metering.
		pub fn create_engine() -> Result<Engine, String> {
			let mut config = Config::new();
			config.consume_fuel(true);
			Engine::new(&config).map_err(|e| format!("failed to create engine: {e}"))
		}

		/// Load a WASM plugin from a file on disk.
		pub fn load(
			engine: &Engine,
			path: &Path,
			manifest: PluginManifest,
		) -> Result<Self, String> {
			let module = Module::from_file(engine, path)
				.map_err(|e| format!("failed to load WASM module: {e}"))?;
			let gate = PermissionGate::new(manifest.permissions.clone());
			Ok(Self {
				manifest,
				module,
				gate,
			})
		}

		/// Load a WASM plugin from in-memory bytes.
		pub fn load_bytes(
			engine: &Engine,
			bytes: &[u8],
			manifest: PluginManifest,
		) -> Result<Self, String> {
			let module = Module::new(engine, bytes)
				.map_err(|e| format!("failed to compile WASM module: {e}"))?;
			let gate = PermissionGate::new(manifest.permissions.clone());
			Ok(Self {
				manifest,
				module,
				gate,
			})
		}

		pub fn manifest(&self) -> &PluginManifest {
			&self.manifest
		}

		pub fn gate(&self) -> &PermissionGate {
			&self.gate
		}

		/// Execute a hook with host function support.
		///
		/// Uses a `Linker` to provide `hx_host_call` to the plugin, enabling
		/// plugins to read/write nodes, search, and log via the host.
		pub fn execute_hook_with_host(
			&self,
			engine: &Engine,
			ctx: &HookContext,
			host_state: crate::host::HostState,
		) -> Result<HookResult, String> {
			let mut linker = Linker::new(engine);
			crate::host::register_host_functions(&mut linker)?;

			let mut store = Store::new(engine, host_state);
			let _ = store.set_fuel(DEFAULT_FUEL);

			let instance = linker
				.instantiate(&mut store, &self.module)
				.map_err(|e| format!("instantiation failed: {e}"))?;

			let memory = instance
				.get_memory(&mut store, "memory")
				.ok_or("plugin does not export 'memory'")?;
			let alloc_fn = instance
				.get_typed_func::<i32, i32>(&mut store, "hx_alloc")
				.map_err(|e| format!("missing hx_alloc export: {e}"))?;
			let hook_fn = instance
				.get_typed_func::<(i32, i32), i64>(&mut store, "hx_plugin_hook")
				.map_err(|e| format!("missing hx_plugin_hook export: {e}"))?;

			let ctx_json = serde_json::to_vec(ctx)
				.map_err(|e| format!("failed to serialize HookContext: {e}"))?;
			let ctx_len = ctx_json.len() as i32;
			let ctx_ptr = alloc_fn
				.call(&mut store, ctx_len)
				.map_err(|e| format!("hx_alloc failed: {e}"))?;
			if ctx_ptr < 0 {
				return Err("hx_alloc returned negative pointer".into());
			}
			let start = ctx_ptr as usize;
			let end = start + ctx_json.len();
			{
				let mem_data = memory.data_mut(&mut store);
				if end > mem_data.len() {
					return Err(format!(
						"guest memory too small: need {} bytes, have {}",
						end,
						mem_data.len()
					));
				}
				mem_data[start..end].copy_from_slice(&ctx_json);
			}

			let packed = hook_fn
				.call(&mut store, (ctx_ptr, ctx_len))
				.map_err(|e| format!("hx_plugin_hook failed: {e}"))?;

			let result_ptr = (packed >> 32) as u32;
			let result_len = (packed & 0xFFFF_FFFF) as u32;
			if result_len == 0 {
				return Ok(HookResult::ok());
			}
			if result_len as usize > MAX_RESULT_SIZE {
				return Err(format!(
					"plugin result too large: {} bytes (max {})",
					result_len, MAX_RESULT_SIZE
				));
			}
			let r_start = result_ptr as usize;
			let r_end = r_start + result_len as usize;
			let mem_data = memory.data(&store);
			if r_end > mem_data.len() {
				return Err(format!(
					"result exceeds guest memory: need {} bytes, have {}",
					r_end,
					mem_data.len()
				));
			}
			serde_json::from_slice(&mem_data[r_start..r_end])
				.map_err(|e| format!("failed to deserialize HookResult: {e}"))
		}

		/// Execute a hook by instantiating the module and calling the
		/// `hx_plugin_hook` export with the JSON-serialized `HookContext`.
		///
		/// # Guest ABI contract
		///
		/// The WASM module must export:
		/// - `memory` — linear memory
		/// - `hx_alloc(size: i32) -> i32` — allocate `size` bytes, return pointer
		/// - `hx_plugin_hook(ptr: i32, len: i32) -> i64` — receive JSON `HookContext`
		///   at `(ptr, len)`, return packed `(result_ptr << 32 | result_len)` for the
		///   JSON-serialized `HookResult`
		pub fn execute_hook(
			&self,
			engine: &Engine,
			ctx: &HookContext,
		) -> Result<HookResult, String> {
			let mut store = Store::new(engine, ());

			// Enable fuel metering if the engine supports it.
			let _ = store.set_fuel(DEFAULT_FUEL);

			let instance = Instance::new(&mut store, &self.module, &[])
				.map_err(|e| format!("instantiation failed: {e}"))?;

			// --- Resolve required guest exports ---

			let memory = instance
				.get_memory(&mut store, "memory")
				.ok_or("plugin does not export 'memory'")?;

			let alloc_fn = instance
				.get_typed_func::<i32, i32>(&mut store, "hx_alloc")
				.map_err(|e| format!("missing hx_alloc export: {e}"))?;

			let hook_fn = instance
				.get_typed_func::<(i32, i32), i64>(&mut store, "hx_plugin_hook")
				.map_err(|e| format!("missing hx_plugin_hook export: {e}"))?;

			// --- Serialize the context and write it into guest memory ---

			let ctx_json = serde_json::to_vec(ctx)
				.map_err(|e| format!("failed to serialize HookContext: {e}"))?;
			let ctx_len = ctx_json.len() as i32;

			let ctx_ptr = alloc_fn
				.call(&mut store, ctx_len)
				.map_err(|e| format!("hx_alloc failed: {e}"))?;

			if ctx_ptr < 0 {
				return Err("hx_alloc returned negative pointer".into());
			}

			let start = ctx_ptr as usize;
			let end = start + ctx_json.len();
			{
				let mem_data = memory.data_mut(&mut store);
				if end > mem_data.len() {
					return Err(format!(
						"guest memory too small: need {} bytes, have {}",
						end,
						mem_data.len()
					));
				}
				mem_data[start..end].copy_from_slice(&ctx_json);
			}

			// --- Call the hook function ---

			let packed = hook_fn
				.call(&mut store, (ctx_ptr, ctx_len))
				.map_err(|e| format!("hx_plugin_hook failed: {e}"))?;

			// Unpack result pointer and length from the i64 return value.
			let result_ptr = (packed >> 32) as u32;
			let result_len = (packed & 0xFFFF_FFFF) as u32;

			// A zero-length result means the plugin has nothing to return.
			if result_len == 0 {
				return Ok(HookResult::ok());
			}

			if result_len as usize > MAX_RESULT_SIZE {
				return Err(format!(
					"plugin result too large: {} bytes (max {})",
					result_len, MAX_RESULT_SIZE
				));
			}

			// --- Read result bytes from guest memory ---

			let r_start = result_ptr as usize;
			let r_end = r_start + result_len as usize;
			let mem_data = memory.data(&store);
			if r_end > mem_data.len() {
				return Err(format!(
					"result exceeds guest memory: need {} bytes, have {}",
					r_end,
					mem_data.len()
				));
			}

			serde_json::from_slice(&mem_data[r_start..r_end])
				.map_err(|e| format!("failed to deserialize HookResult: {e}"))
		}
	}
}

#[cfg(feature = "wasm-runtime")]
pub use inner::WasmPlugin;
