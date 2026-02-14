use std::collections::HashMap;

use hx_core::{HttpProxyRequest, ExecProxyRequest, NodeKind};
use hx_engine::recurrence::validate_recurrence_metadata_for_kind;
use serde_json::Value;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("{field} cannot be empty")]
    EmptyField { field: String },

    #[error("{field} exceeds max length of {max}")]
    TooLong { field: String, max: usize },

    #[error("{field} contains invalid characters")]
    InvalidCharacters { field: String },

    #[error("{field} must be between {min} and {max}")]
    OutOfRange { field: String, min: String, max: String },

    #[error("{field} exceeds maximum count of {max}")]
    TooMany { field: String, max: usize },

    #[error("{0}")]
    Custom(String),
}

impl From<String> for ValidationError {
    fn from(s: String) -> Self {
        ValidationError::Custom(s)
    }
}

impl From<&str> for ValidationError {
    fn from(s: &str) -> Self {
        ValidationError::Custom(s.to_string())
    }
}

impl From<ValidationError> for String {
    fn from(err: ValidationError) -> Self {
        err.to_string()
    }
}

impl ValidationError {
    /// Convert to the `(StatusCode, String)` error tuple pattern used by axum handlers.
    pub fn bad_request(self) -> (axum::http::StatusCode, String) {
        (axum::http::StatusCode::BAD_REQUEST, self.to_string())
    }

    /// Check if the error message contains a substring (convenience for tests).
    pub fn contains(&self, s: &str) -> bool {
        self.to_string().contains(s)
    }
}

impl From<ValidationError> for (axum::http::StatusCode, String) {
    fn from(err: ValidationError) -> Self {
        (axum::http::StatusCode::BAD_REQUEST, err.to_string())
    }
}

const MAX_NODE_CONTENT_LEN: usize = 64 * 1024;
const MAX_NODE_TITLE_LEN: usize = 512;
const MAX_NODE_SOURCE_LEN: usize = 2048;
const MAX_NAMESPACE_LEN: usize = 128;
const MAX_TAGS: usize = 32;
const MAX_TAG_LEN: usize = 64;
const MAX_METADATA_JSON_LEN: usize = 64 * 1024;
const MAX_QUERY_TEXT_LEN: usize = 4096;
const MAX_RECALL_LIMIT: usize = 200;
const MAX_LIST_LIMIT: usize = 500;
const MAX_NEIGHBOR_DEPTH: usize = 8;

// --- Proxy validation constants ---
const MAX_PROXY_URL_LEN: usize = 8192;
const MAX_PROXY_BODY_LEN: usize = 1024 * 1024; // 1 MB
const MAX_PROXY_HEADERS: usize = 32;
const MAX_PROXY_HEADER_NAME_LEN: usize = 256;
const MAX_PROXY_HEADER_VALUE_LEN: usize = 8192;
const MAX_PROXY_ARGS: usize = 64;
const MAX_PROXY_ARG_LEN: usize = 4096;
const MAX_PROXY_TIMEOUT_SECS: u64 = 300;
const MIN_PROXY_INTENT_LEN: usize = 10;
const MAX_PROXY_INTENT_LEN: usize = 500;
const MAX_PROXY_SECRET_REF_LEN: usize = 256;
const MAX_PROXY_COMMAND_LEN: usize = 256;

const ALLOWED_HTTP_METHODS: &[&str] = &[
    "GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS",
];

#[allow(clippy::too_many_arguments)]
pub fn validate_node_payload(
    kind: NodeKind,
    title: Option<&str>,
    content: &str,
    source: Option<&str>,
    namespace: Option<&str>,
    tags: &[String],
    importance: Option<f64>,
    metadata: Option<&HashMap<String, Value>>,
) -> Result<(), ValidationError> {
    validate_required_text("content", content, MAX_NODE_CONTENT_LEN)?;
    validate_optional_text("title", title, MAX_NODE_TITLE_LEN)?;
    validate_optional_text("source", source, MAX_NODE_SOURCE_LEN)?;

    if let Some(namespace) = namespace {
        validate_namespace(namespace)?;
    }

    validate_tags(tags)?;

    if let Some(importance) = importance {
        if !importance.is_finite() || !(0.0..=1.0).contains(&importance) {
            return Err("importance must be a finite value between 0.0 and 1.0".into());
        }
    }

    validate_metadata(metadata)?;
    validate_recurrence_metadata_for_kind(kind, metadata)?;

    Ok(())
}

pub fn validate_query_text(field_name: &str, text: &str) -> Result<(), ValidationError> {
    validate_required_text(field_name, text, MAX_QUERY_TEXT_LEN)
}

pub fn validate_recall_limit(limit: usize) -> Result<(), ValidationError> {
    if limit == 0 {
        return Err("limit must be greater than 0".into());
    }
    if limit > MAX_RECALL_LIMIT {
        return Err(format!("limit must be <= {MAX_RECALL_LIMIT}").into());
    }
    Ok(())
}

pub fn validate_list_limit(limit: usize) -> Result<(), ValidationError> {
    if limit == 0 {
        return Err("limit must be greater than 0".into());
    }
    if limit > MAX_LIST_LIMIT {
        return Err(format!("limit must be <= {MAX_LIST_LIMIT}").into());
    }
    Ok(())
}

pub fn validate_text_input(name: &str, value: &str) -> Result<(), ValidationError> {
    validate_required_text(name, value, MAX_NODE_TITLE_LEN)
}

pub fn validate_namespace_input(namespace: Option<&str>) -> Result<(), ValidationError> {
    if let Some(namespace) = namespace {
        validate_namespace(namespace)?;
    }
    Ok(())
}

pub fn validate_tags_input(tags: &[String]) -> Result<(), ValidationError> {
    validate_tags(tags)
}

pub fn validate_depth(depth: usize) -> Result<(), ValidationError> {
    if depth == 0 {
        return Err("depth must be greater than 0".into());
    }
    if depth > MAX_NEIGHBOR_DEPTH {
        return Err(format!("depth must be <= {MAX_NEIGHBOR_DEPTH}").into());
    }
    Ok(())
}

fn validate_required_text(name: &str, value: &str, max_len: usize) -> Result<(), ValidationError> {
    if value.trim().is_empty() {
        return Err(format!("{name} cannot be empty").into());
    }

    if value.len() > max_len {
        return Err(format!("{name} exceeds max length of {max_len}").into());
    }

    Ok(())
}

fn validate_optional_text(name: &str, value: Option<&str>, max_len: usize) -> Result<(), ValidationError> {
    if let Some(value) = value {
        validate_required_text(name, value, max_len)?;
    }
    Ok(())
}

fn validate_namespace(namespace: &str) -> Result<(), ValidationError> {
    validate_required_text("namespace", namespace, MAX_NAMESPACE_LEN)?;
    if namespace
        .chars()
        .any(|ch| !(ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.' | ':' | '/')))
    {
        return Err("namespace contains invalid characters".into());
    }
    Ok(())
}

fn validate_tags(tags: &[String]) -> Result<(), ValidationError> {
    if tags.len() > MAX_TAGS {
        return Err(format!("tags cannot exceed {MAX_TAGS} items").into());
    }

    for tag in tags {
        validate_required_text("tag", tag, MAX_TAG_LEN)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Proxy validation
// ---------------------------------------------------------------------------

/// Validate an HTTP proxy request before passing to ProxyEngine.
pub fn validate_http_proxy_request(req: &HttpProxyRequest) -> Result<(), ValidationError> {
    // URL
    if req.url.trim().is_empty() {
        return Err("url cannot be empty".into());
    }
    if req.url.len() > MAX_PROXY_URL_LEN {
        return Err(format!("url exceeds max length of {MAX_PROXY_URL_LEN}").into());
    }

    // Method
    let method_upper = req.method.to_ascii_uppercase();
    if !ALLOWED_HTTP_METHODS.contains(&method_upper.as_str()) {
        return Err(format!(
            "method '{}' is not allowed; permitted: {}",
            req.method,
            ALLOWED_HTTP_METHODS.join(", ")
        ).into());
    }

    // Body
    if let Some(ref body) = req.body {
        if body.len() > MAX_PROXY_BODY_LEN {
            return Err(format!(
                "body exceeds max size of {} bytes",
                MAX_PROXY_BODY_LEN
            ).into());
        }
    }

    // Headers
    if req.headers.len() > MAX_PROXY_HEADERS {
        return Err(format!(
            "headers cannot exceed {MAX_PROXY_HEADERS} entries"
        ).into());
    }
    for (name, value) in &req.headers {
        if name.len() > MAX_PROXY_HEADER_NAME_LEN {
            return Err(format!(
                "header name '{}' exceeds max length of {MAX_PROXY_HEADER_NAME_LEN}",
                &name[..64.min(name.len())]
            )
            .into());
        }
        if value.len() > MAX_PROXY_HEADER_VALUE_LEN {
            return Err(format!(
                "header value for '{}' exceeds max length of {MAX_PROXY_HEADER_VALUE_LEN}",
                name
            ).into());
        }
    }

    // Secret ref
    validate_proxy_secret_ref(&req.secret_ref)?;

    // Intent
    validate_proxy_intent(&req.intent)?;

    Ok(())
}

/// Validate an exec proxy request before passing to ProxyEngine.
pub fn validate_exec_proxy_request(req: &ExecProxyRequest) -> Result<(), ValidationError> {
    // Command
    if req.command.trim().is_empty() {
        return Err("command cannot be empty".into());
    }
    if req.command.len() > MAX_PROXY_COMMAND_LEN {
        return Err(format!(
            "command exceeds max length of {MAX_PROXY_COMMAND_LEN}"
        ).into());
    }

    // Args
    if req.args.len() > MAX_PROXY_ARGS {
        return Err(format!("args cannot exceed {MAX_PROXY_ARGS} entries").into());
    }
    for (i, arg) in req.args.iter().enumerate() {
        if arg.len() > MAX_PROXY_ARG_LEN {
            return Err(format!(
                "arg[{i}] exceeds max length of {MAX_PROXY_ARG_LEN}"
            ).into());
        }
    }

    // Env inject: validate secret refs
    for (env_var, secret_key) in &req.env_inject {
        if env_var.trim().is_empty() {
            return Err("env_inject key cannot be empty".into());
        }
        validate_proxy_secret_ref(secret_key)?;
    }

    // Timeout
    if req.timeout_seconds > MAX_PROXY_TIMEOUT_SECS {
        return Err(format!(
            "timeout_seconds cannot exceed {MAX_PROXY_TIMEOUT_SECS}"
        ).into());
    }

    // Intent
    validate_proxy_intent(&req.intent)?;

    Ok(())
}

fn validate_proxy_secret_ref(secret_ref: &str) -> Result<(), ValidationError> {
    if secret_ref.trim().is_empty() {
        return Err("secret_ref cannot be empty".into());
    }
    if secret_ref.len() > MAX_PROXY_SECRET_REF_LEN {
        return Err(format!(
            "secret_ref exceeds max length of {MAX_PROXY_SECRET_REF_LEN}"
        ).into());
    }
    Ok(())
}

fn validate_proxy_intent(intent: &str) -> Result<(), ValidationError> {
    let trimmed = intent.trim();
    if trimmed.len() < MIN_PROXY_INTENT_LEN {
        return Err(format!(
            "intent must be at least {MIN_PROXY_INTENT_LEN} characters"
        ).into());
    }
    if intent.len() > MAX_PROXY_INTENT_LEN {
        return Err(format!(
            "intent exceeds max length of {MAX_PROXY_INTENT_LEN}"
        ).into());
    }
    Ok(())
}

fn validate_metadata(metadata: Option<&HashMap<String, Value>>) -> Result<(), ValidationError> {
    let Some(metadata) = metadata else {
        return Ok(());
    };

    let encoded = serde_json::to_string(metadata)
        .map_err(|err| -> ValidationError { format!("metadata must be valid JSON serializable object: {err}").into() })?;
    if encoded.len() > MAX_METADATA_JSON_LEN {
        return Err(format!(
            "metadata exceeds max serialized size of {MAX_METADATA_JSON_LEN} bytes"
        ).into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_node_payload_rejects_empty_content() {
        let err = validate_node_payload(
            NodeKind::Fact,
            None,
            "   ",
            None,
            Some("default"),
            &[],
            Some(0.5),
            None,
        )
        .expect_err("empty content should fail");
        assert!(err.contains("content cannot be empty"));
    }

    #[test]
    fn validate_node_payload_rejects_bad_namespace() {
        let err = validate_node_payload(
            NodeKind::Fact,
            None,
            "ok",
            None,
            Some("bad namespace"),
            &[],
            Some(0.5),
            None,
        )
        .expect_err("namespace should fail");
        assert!(err.contains("invalid characters"));
    }

    #[test]
    fn validate_node_payload_rejects_invalid_importance() {
        let err = validate_node_payload(
            NodeKind::Fact,
            None,
            "ok",
            None,
            Some("ns"),
            &[],
            Some(1.5),
            None,
        )
        .expect_err("importance should fail");
        assert!(err.contains("importance"));
    }

    #[test]
    fn validate_node_payload_rejects_task_recurrence_for_non_task_kind() {
        let metadata = serde_json::json!({
            "task_recurrence": {
                "frequency": "daily",
                "interval": 1
            }
        });
        let metadata: HashMap<String, Value> =
            serde_json::from_value(metadata).expect("metadata should deserialize");
        let err = validate_node_payload(
            NodeKind::Fact,
            None,
            "ok",
            None,
            Some("ns"),
            &[],
            Some(0.5),
            Some(&metadata),
        )
        .expect_err("task_recurrence on fact should fail");
        assert!(err.contains("task_recurrence"));
    }

    #[test]
    fn validate_limit_checks_boundaries() {
        assert!(validate_recall_limit(1).is_ok());
        assert!(validate_recall_limit(201).is_err());
        assert!(validate_list_limit(500).is_ok());
        assert!(validate_list_limit(501).is_err());
    }

    #[test]
    fn validate_depth_checks_boundaries() {
        assert!(validate_depth(1).is_ok());
        assert!(validate_depth(9).is_err());
    }

    // --- Proxy validation tests ---

    fn make_http_proxy_req() -> HttpProxyRequest {
        HttpProxyRequest {
            method: "GET".into(),
            url: "https://api.example.com/v1/data".into(),
            headers: HashMap::new(),
            body: None,
            secret_ref: "MY_API_KEY".into(),
            inject_as: hx_core::SecretInjection::BearerHeader,
            intent: "fetch user profile data for dashboard display".into(),
        }
    }

    fn make_exec_proxy_req() -> ExecProxyRequest {
        ExecProxyRequest {
            command: "gh".into(),
            args: vec!["pr".into(), "list".into()],
            env_inject: {
                let mut m = HashMap::new();
                m.insert("GH_TOKEN".into(), "GITHUB_TOKEN".into());
                m
            },
            working_dir: None,
            timeout_seconds: 30,
            intent: "list open pull requests for review".into(),
        }
    }

    #[test]
    fn proxy_http_valid_request_passes() {
        assert!(validate_http_proxy_request(&make_http_proxy_req()).is_ok());
    }

    #[test]
    fn proxy_http_rejects_bad_method() {
        let mut req = make_http_proxy_req();
        req.method = "CONNECT".into();
        assert!(validate_http_proxy_request(&req)
            .unwrap_err()
            .contains("not allowed"));
    }

    #[test]
    fn proxy_http_rejects_oversized_body() {
        let mut req = make_http_proxy_req();
        req.body = Some("x".repeat(MAX_PROXY_BODY_LEN + 1));
        assert!(validate_http_proxy_request(&req)
            .unwrap_err()
            .contains("body exceeds"));
    }

    #[test]
    fn proxy_http_rejects_too_many_headers() {
        let mut req = make_http_proxy_req();
        for i in 0..MAX_PROXY_HEADERS + 1 {
            req.headers.insert(format!("X-Header-{i}"), "v".into());
        }
        assert!(validate_http_proxy_request(&req)
            .unwrap_err()
            .contains("headers cannot exceed"));
    }

    #[test]
    fn proxy_http_rejects_short_intent() {
        let mut req = make_http_proxy_req();
        req.intent = "short".into();
        assert!(validate_http_proxy_request(&req)
            .unwrap_err()
            .contains("at least"));
    }

    #[test]
    fn proxy_http_rejects_long_intent() {
        let mut req = make_http_proxy_req();
        req.intent = "x".repeat(MAX_PROXY_INTENT_LEN + 1);
        assert!(validate_http_proxy_request(&req)
            .unwrap_err()
            .contains("exceeds max"));
    }

    #[test]
    fn proxy_exec_valid_request_passes() {
        assert!(validate_exec_proxy_request(&make_exec_proxy_req()).is_ok());
    }

    #[test]
    fn proxy_exec_rejects_too_many_args() {
        let mut req = make_exec_proxy_req();
        req.args = (0..MAX_PROXY_ARGS + 1).map(|i| format!("arg{i}")).collect();
        assert!(validate_exec_proxy_request(&req)
            .unwrap_err()
            .contains("args cannot exceed"));
    }

    #[test]
    fn proxy_exec_rejects_oversized_timeout() {
        let mut req = make_exec_proxy_req();
        req.timeout_seconds = MAX_PROXY_TIMEOUT_SECS + 1;
        assert!(validate_exec_proxy_request(&req)
            .unwrap_err()
            .contains("timeout_seconds"));
    }
}
