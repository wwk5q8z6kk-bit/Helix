//! Host function implementations for the WASM plugin sandbox.
//!
//! Provides `HostState` that holds a reference to the store and permission gate,
//! and registers host functions with a wasmtime `Linker` so plugins can call
//! `hx_host_call` to read/write nodes, search, and log.

#[cfg(feature = "wasm-runtime")]
mod inner {
    use crate::abi::{HostRequest, HostResponse, HOST_LOG, HOST_READ_NODE, HOST_SEARCH, HOST_WRITE_NODE};
    use crate::sandbox::PermissionGate;
    use std::sync::Arc;
    use wasmtime::*;

    /// State available to host functions during plugin execution.
    pub struct HostState {
        /// Permission gate for the current plugin.
        pub gate: PermissionGate,
        /// Callback that dispatches host requests to the store layer.
        /// This is a synchronous callback because wasmtime host functions run synchronously.
        pub dispatch: Arc<dyn Fn(&HostRequest) -> HostResponse + Send + Sync>,
    }

    /// Register the `hx_host_call` function on a `Linker`.
    ///
    /// The guest ABI:
    /// ```text
    /// hx_host_call(request_ptr: i32, request_len: i32) -> i64
    /// ```
    /// - The guest writes a JSON-serialized `HostRequest` at `(request_ptr, request_len)`
    ///   in its linear memory.
    /// - The host reads the request, checks permissions, dispatches, and writes the
    ///   JSON-serialized `HostResponse` back into guest memory via `hx_alloc`.
    /// - Returns packed `(response_ptr << 32 | response_len)` or `0` on error.
    pub fn register_host_functions(linker: &mut Linker<HostState>) -> Result<(), String> {
        linker
            .func_wrap(
                "env",
                "hx_host_call",
                |mut caller: Caller<'_, HostState>, req_ptr: i32, req_len: i32| -> i64 {
                    // Read request from guest memory
                    let memory = match caller.get_export("memory") {
                        Some(Extern::Memory(m)) => m,
                        _ => return 0,
                    };

                    let start = req_ptr as usize;
                    let end = start + req_len as usize;
                    let data = memory.data(&caller);
                    if end > data.len() {
                        return 0;
                    }

                    let request: HostRequest = match serde_json::from_slice(&data[start..end]) {
                        Ok(r) => r,
                        Err(_) => return 0,
                    };

                    // Check permissions
                    let state = caller.data();
                    if let Err(_) = state.gate.check(&request.method) {
                        let resp = HostResponse::err(format!(
                            "permission denied: {}",
                            request.method
                        ));
                        return write_response_to_guest(&mut caller, &resp);
                    }

                    // Dispatch
                    let dispatch = Arc::clone(&caller.data().dispatch);
                    let response = dispatch(&request);

                    write_response_to_guest(&mut caller, &response)
                },
            )
            .map_err(|e| format!("failed to register hx_host_call: {e}"))?;

        Ok(())
    }

    /// Write a `HostResponse` JSON back into guest memory via `hx_alloc`.
    /// Returns packed `(ptr << 32 | len)` or `0` on failure.
    fn write_response_to_guest(
        caller: &mut Caller<'_, HostState>,
        response: &HostResponse,
    ) -> i64 {
        let json = match serde_json::to_vec(response) {
            Ok(j) => j,
            Err(_) => return 0,
        };
        let len = json.len() as i32;

        // Call guest's hx_alloc to get a pointer
        let alloc_fn = match caller.get_export("hx_alloc") {
            Some(Extern::Func(f)) => f,
            _ => return 0,
        };

        let alloc_typed = match alloc_fn.typed::<i32, i32>(caller.as_context()) {
            Ok(f) => f,
            Err(_) => return 0,
        };

        let ptr = match alloc_typed.call(&mut *caller, len) {
            Ok(p) if p >= 0 => p,
            _ => return 0,
        };

        // Write response into guest memory
        let memory = match caller.get_export("memory") {
            Some(Extern::Memory(m)) => m,
            _ => return 0,
        };

        let start = ptr as usize;
        let end = start + json.len();
        let mem_data = memory.data_mut(&mut *caller);
        if end > mem_data.len() {
            return 0;
        }
        mem_data[start..end].copy_from_slice(&json);

        ((ptr as i64) << 32) | (len as i64)
    }

    /// Create a default dispatch function that handles the standard host methods.
    /// This is a simple implementation that logs calls but delegates actual
    /// store operations to the provided callbacks.
    pub fn create_dispatch(
        read_node: impl Fn(&serde_json::Value) -> HostResponse + Send + Sync + 'static,
        write_node: impl Fn(&serde_json::Value) -> HostResponse + Send + Sync + 'static,
        search: impl Fn(&serde_json::Value) -> HostResponse + Send + Sync + 'static,
    ) -> Arc<dyn Fn(&HostRequest) -> HostResponse + Send + Sync> {
        Arc::new(move |req: &HostRequest| match req.method.as_str() {
            HOST_READ_NODE => read_node(&req.params),
            HOST_WRITE_NODE => write_node(&req.params),
            HOST_SEARCH => search(&req.params),
            HOST_LOG => {
                if let Some(msg) = req.params.as_str() {
                    tracing::info!(plugin_log = msg, "plugin log");
                }
                HostResponse::ok(serde_json::Value::Null)
            }
            other => HostResponse::err(format!("unknown method: {other}")),
        })
    }
}

#[cfg(feature = "wasm-runtime")]
pub use inner::*;
