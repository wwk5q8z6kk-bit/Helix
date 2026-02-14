use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hx_core::ChronicleEntry;
use hx_engine::engine::HelixEngine;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::{debug, error, info};

use crate::auth::McpContext;
use crate::protocol::*;
use crate::{prompts, resources, tools};

const SERVER_NAME: &str = "helix-mcp";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");
const PROTOCOL_VERSION: &str = "2024-11-05";

pub struct RateLimiter {
    calls: Mutex<HashMap<String, Vec<Instant>>>,
    max_per_minute: usize,
}

impl RateLimiter {
    pub fn new(max_per_minute: usize) -> Self {
        Self {
            calls: Mutex::new(HashMap::new()),
            max_per_minute,
        }
    }

    pub fn check(&self, key: &str) -> Result<(), String> {
        let mut map = self.calls.lock().unwrap();
        let now = Instant::now();
        let window = Duration::from_secs(60);
        let entry = map.entry(key.to_string()).or_default();
        entry.retain(|t| now.duration_since(*t) < window);
        if entry.len() >= self.max_per_minute {
            return Err(format!(
                "rate limit exceeded: {} calls/minute",
                self.max_per_minute
            ));
        }
        entry.push(now);
        Ok(())
    }
}

pub struct McpServer {
    engine: Arc<HelixEngine>,
    context: McpContext,
    rate_limiter: RateLimiter,
}

impl McpServer {
    pub fn new(engine: Arc<HelixEngine>, context: McpContext) -> Self {
        Self {
            engine,
            context,
            rate_limiter: RateLimiter::new(60),
        }
    }

    /// Run the MCP server over stdio (line-delimited JSON-RPC).
    pub async fn run_stdio(&self) -> Result<(), Box<dyn std::error::Error>> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let reader = BufReader::new(stdin);
        let mut lines = reader.lines();

        info!(
            "{SERVER_NAME} v{SERVER_VERSION} starting on stdio ({})",
            self.context.summary()
        );

        while let Ok(Some(line)) = lines.next_line().await {
            let line = line.trim().to_string();
            if line.is_empty() {
                continue;
            }

            debug!("recv: {line}");

            let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
                Ok(request) => self.handle_request(request).await,
                Err(e) => JsonRpcResponse::error(Value::Null, -32700, format!("Parse error: {e}")),
            };

            let json = serde_json::to_string(&response)?;
            debug!("send: {json}");
            stdout.write_all(json.as_bytes()).await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }

        info!("{SERVER_NAME} shutting down");
        Ok(())
    }

    async fn handle_request(&self, req: JsonRpcRequest) -> JsonRpcResponse {
        match req.method.as_str() {
            "initialize" => self.handle_initialize(req.id),
            "initialized" => {
                // Client acknowledgment — no response needed, but we return success
                // to avoid breaking line-delimited protocol expectations.
                JsonRpcResponse::success(req.id, json!({}))
            }
            "tools/list" => self.handle_tools_list(req.id),
            "tools/call" => self.handle_tools_call(req.id, req.params).await,
            "resources/list" => self.handle_resources_list(req.id),
            "resources/read" => self.handle_resources_read(req.id, req.params).await,
            "prompts/list" => self.handle_prompts_list(req.id),
            "prompts/get" => self.handle_prompts_get(req.id, req.params).await,
            "ping" => JsonRpcResponse::success(req.id, json!({})),
            method => {
                error!("unknown method: {method}");
                JsonRpcResponse::error(req.id, -32601, format!("Method not found: {method}"))
            }
        }
    }

    // -----------------------------------------------------------------------
    // initialize
    // -----------------------------------------------------------------------

    fn handle_initialize(&self, id: Value) -> JsonRpcResponse {
        let capabilities = ServerCapabilities {
            tools: Some(ToolsCapability {}),
            resources: Some(ResourcesCapability {}),
            prompts: Some(PromptsCapability {}),
        };

        JsonRpcResponse::success(
            id,
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": capabilities,
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                }
            }),
        )
    }

    // -----------------------------------------------------------------------
    // tools
    // -----------------------------------------------------------------------

    fn handle_tools_list(&self, id: Value) -> JsonRpcResponse {
        let tool_defs = tools::list_tools(&self.context);
        JsonRpcResponse::success(id, json!({ "tools": tool_defs }))
    }

    async fn handle_tools_call(&self, id: Value, params: Value) -> JsonRpcResponse {
        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                return JsonRpcResponse::error(id, -32602, "missing 'name' in tools/call params")
            }
        };

        let caller_key = self
            .context
            .key_id()
            .unwrap_or("anonymous")
            .to_string();

        if let Err(e) = self.rate_limiter.check(&caller_key) {
            return JsonRpcResponse::error(id, -32000, e);
        }

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        let result = tools::call_tool(&self.engine, &self.context, &name, arguments).await;

        let is_err = result.is_error();
        let chronicle = ChronicleEntry::new(
            "mcp.tool_call",
            format!(
                "Tool '{}' called by '{}': {}",
                name,
                caller_key,
                if is_err { "error" } else { "success" }
            ),
        );
        let _ = self.engine.log_chronicle(&chronicle).await;

        JsonRpcResponse::success(id, serde_json::to_value(result).unwrap_or(json!({})))
    }

    // -----------------------------------------------------------------------
    // resources
    // -----------------------------------------------------------------------

    fn handle_resources_list(&self, id: Value) -> JsonRpcResponse {
        let resource_defs = resources::list_resources(&self.context);
        JsonRpcResponse::success(id, json!({ "resources": resource_defs }))
    }

    async fn handle_resources_read(&self, id: Value, params: Value) -> JsonRpcResponse {
        let uri = match params.get("uri").and_then(|v| v.as_str()) {
            Some(u) => u.to_string(),
            None => {
                return JsonRpcResponse::error(id, -32602, "missing 'uri' in resources/read params")
            }
        };

        match resources::read_resource(&self.engine, &self.context, &uri).await {
            Ok(content) => JsonRpcResponse::success(id, json!({ "contents": [content] })),
            Err(e) => JsonRpcResponse::error(id, -32602, e),
        }
    }

    // -----------------------------------------------------------------------
    // prompts
    // -----------------------------------------------------------------------

    fn handle_prompts_list(&self, id: Value) -> JsonRpcResponse {
        let prompt_defs = prompts::list_prompts(&self.context);
        JsonRpcResponse::success(id, json!({ "prompts": prompt_defs }))
    }

    async fn handle_prompts_get(&self, id: Value, params: Value) -> JsonRpcResponse {
        let name = match params.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => {
                return JsonRpcResponse::error(id, -32602, "missing 'name' in prompts/get params")
            }
        };

        let arguments = params
            .get("arguments")
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));

        match prompts::get_prompt(&self.engine, &self.context, &name, &arguments).await {
            Ok(messages) => JsonRpcResponse::success(id, json!({ "messages": messages })),
            Err(e) => JsonRpcResponse::error(id, -32602, e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- RateLimiter tests ---

    #[test]
    fn rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(5);
        for _ in 0..5 {
            assert!(limiter.check("test-key").is_ok());
        }
    }

    #[test]
    fn rate_limiter_blocks_over_limit() {
        let limiter = RateLimiter::new(3);
        for _ in 0..3 {
            limiter.check("test-key").unwrap();
        }
        let result = limiter.check("test-key");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("rate limit exceeded"));
    }

    #[test]
    fn rate_limiter_isolates_keys() {
        let limiter = RateLimiter::new(2);
        limiter.check("key-a").unwrap();
        limiter.check("key-a").unwrap();
        assert!(limiter.check("key-a").is_err());
        // key-b should still be allowed
        assert!(limiter.check("key-b").is_ok());
    }

    // --- JsonRpcResponse tests ---

    #[test]
    fn initialize_returns_capabilities() {
        let response = JsonRpcResponse::success(
            json!(1),
            json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": {},
                    "resources": {},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION,
                }
            }),
        );
        let result = response.result.unwrap();
        assert_eq!(result["protocolVersion"], PROTOCOL_VERSION);
        assert!(result["capabilities"]["tools"].is_object());
        assert!(result["capabilities"]["resources"].is_object());
        assert!(result["capabilities"]["prompts"].is_object());
        assert_eq!(result["serverInfo"]["name"], SERVER_NAME);
    }

    #[test]
    fn unknown_method_returns_error() {
        let response = JsonRpcResponse::error(json!(1), -32601, "Method not found: bogus");
        assert!(response.error.is_some());
        let err = response.error.unwrap();
        assert_eq!(err.code, -32601);
        assert!(err.message.contains("bogus"));
    }

    #[test]
    fn success_response_has_no_error() {
        let response = JsonRpcResponse::success(json!(1), json!({"ok": true}));
        assert!(response.result.is_some());
        assert!(response.error.is_none());
        assert_eq!(response.jsonrpc, "2.0");
    }

    #[test]
    fn error_response_has_no_result() {
        let response = JsonRpcResponse::error(json!(1), -32700, "parse error");
        assert!(response.result.is_none());
        assert!(response.error.is_some());
    }
}
