use std::collections::HashMap;
use std::net::IpAddr;
use std::time::Duration;

use hx_core::{ApprovalRequest, PolicyDecision, *};
use zeroize::Zeroizing;

use crate::engine::HelixEngine;
use crate::intent_check;
use crate::sanitize::OutputSanitizer;

// ---------------------------------------------------------------------------
// Proxy Error
// ---------------------------------------------------------------------------

/// Typed error for proxy operations, enabling the REST layer to map errors
/// to appropriate HTTP status codes.
#[derive(Debug)]
pub enum ProxyError {
    /// Access denied by policy, scope, constitutional rule, or intent check.
    Denied(String),
    /// Policy exists but requires HITL approval. Contains the approval request ID.
    ApprovalRequired {
        approval_id: uuid::Uuid,
        message: String,
    },
    /// Operational failure (HTTP error, spawn error, timeout, etc).
    Failed(String),
}

impl std::fmt::Display for ProxyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Denied(msg) => write!(f, "denied: {msg}"),
            Self::ApprovalRequired { message, .. } => write!(f, "approval required: {message}"),
            Self::Failed(msg) => write!(f, "failed: {msg}"),
        }
    }
}

impl std::error::Error for ProxyError {}

/// Default commands allowed in exec proxy.
const DEFAULT_EXEC_ALLOWLIST: &[&str] = &[
    "gh", "git", "curl", "aws", "gcloud", "npm", "cargo",
];

/// Constitutional deny patterns checked before any policy evaluation.
/// These patterns are always blocked regardless of policy.
const CONSTITUTIONAL_DENY_PATTERNS: &[&str] = &[
    "rm -rf /",
    "dd if=/dev/zero",
    "mkfs.",
    ":(){:|:&};:",
    "chmod -R 777 /",
];

/// Private IP ranges blocked for SSRF prevention.
fn is_private_ip(addr: IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => {
            v4.is_loopback()           // 127.0.0.0/8
            || v4.is_private()         // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            || v4.is_link_local()      // 169.254.0.0/16
            || v4.is_unspecified()     // 0.0.0.0
        }
        IpAddr::V6(v6) => {
            v6.is_loopback()           // ::1
            || v6.is_unspecified()     // ::
            // IPv6-mapped IPv4 (::ffff:x.x.x.x) — check the inner v4
            || v6.to_ipv4_mapped()
                .map(|v4| is_private_ip(IpAddr::V4(v4)))
                .unwrap_or(false)
            // fd00::/8 (unique local)
            || (v6.segments()[0] & 0xff00) == 0xfd00
            // fe80::/10 (link-local)
            || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Check if a URL targets a private IP (SSRF prevention).
fn check_ssrf(url: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|e| format!("invalid URL: {e}"))?;

    let host = parsed
        .host_str()
        .ok_or_else(|| "URL has no host".to_string())?;

    // Try parsing as IP directly (strip brackets for IPv6 URIs like [::1])
    let host_bare = host.trim_start_matches('[').trim_end_matches(']');
    if let Ok(ip) = host_bare.parse::<IpAddr>() {
        if is_private_ip(ip) {
            return Err(format!("blocked: private IP address {host}"));
        }
    }

    // Check well-known private hostnames
    let host_lower = host.to_ascii_lowercase();
    if host_lower == "localhost"
        || host_lower.ends_with(".local")
        || host_lower.ends_with(".internal")
        || host_lower == "metadata.google.internal"
        || host_lower == "169.254.169.254"
    {
        return Err(format!("blocked: private hostname {host}"));
    }

    // Attempt DNS resolution to catch private IPs behind public hostnames.
    if let Ok(addrs) = std::net::ToSocketAddrs::to_socket_addrs(&(host, 80)) {
        for addr in addrs {
            if is_private_ip(addr.ip()) {
                return Err(format!(
                    "blocked: hostname {host} resolves to private IP {}",
                    addr.ip()
                ));
            }
        }
    }

    Ok(())
}

/// Check constitutional deny patterns against a request description.
fn check_constitutional_rules(description: &str) -> Result<(), String> {
    let lower = description.to_ascii_lowercase();
    for pattern in CONSTITUTIONAL_DENY_PATTERNS {
        if lower.contains(pattern) {
            return Err(format!(
                "blocked by constitutional rule: matches deny pattern '{pattern}'"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scope enforcement
// ---------------------------------------------------------------------------

/// Derive the required scopes for an HTTP request based on its method.
fn derive_http_scopes(method: &str) -> Vec<String> {
    match method.to_ascii_uppercase().as_str() {
        "GET" | "HEAD" | "OPTIONS" => vec!["read".into()],
        _ => vec!["write".into()],
    }
}

/// Derive the required scopes for an exec request based on the command.
fn derive_exec_scopes(command: &str) -> Vec<String> {
    vec![format!("exec:{command}")]
}

/// Check whether the required scopes are satisfied by the allowed scopes.
///
/// - Empty `allowed` = unrestricted (backward compatible with pre-scope policies).
/// - Each required scope must match at least one allowed scope.
/// - Matching is hierarchical prefix: allowed `"read"` matches required `"read"`,
///   allowed `"exec"` matches required `"exec:git"`, allowed `"exec:git"` matches `"exec:git"`.
fn check_scopes(required: &[String], allowed: &[String]) -> Result<(), String> {
    if allowed.is_empty() {
        return Ok(()); // unrestricted
    }

    for req in required {
        if !allowed.iter().any(|a| scope_matches(req, a)) {
            return Err(format!(
                "scope mismatch: operation requires '{req}' but policy allows {:?}",
                allowed
            ));
        }
    }
    Ok(())
}

/// Returns true if `required` scope is covered by `allowed` pattern.
///
/// - Exact match: `"read"` matches `"read"`
/// - Prefix match: `"exec"` matches `"exec:git"` (allowed is a prefix of required)
/// - Wildcard: `"exec:*"` matches `"exec:git"`
fn scope_matches(required: &str, allowed: &str) -> bool {
    if required == allowed {
        return true;
    }
    // allowed "exec" covers required "exec:git"
    if required.starts_with(allowed) && required.as_bytes().get(allowed.len()) == Some(&b':') {
        return true;
    }
    // Wildcard: allowed "exec:*" covers required "exec:git"
    if let Some(prefix) = allowed.strip_suffix(":*") {
        if required == prefix || (required.starts_with(prefix) && required.as_bytes().get(prefix.len()) == Some(&b':')) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Proxy Engine
// ---------------------------------------------------------------------------

/// Proxy engine for HTTP and command operations with credential injection.
pub struct ProxyEngine;

impl ProxyEngine {
    /// Execute an HTTP proxy request with credential injection.
    ///
    /// Flow: constitutional check -> SSRF check -> intent check -> policy check -> scope check -> inject -> execute -> sanitize -> audit
    pub async fn execute_http(
        req: &HttpProxyRequest,
        consumer: &str,
        engine: &HelixEngine,
    ) -> Result<HttpProxyResponse, ProxyError> {
        // 1. Constitutional rules
        let description = format!("{} {} intent={}", req.method, req.url, req.intent);
        check_constitutional_rules(&description).map_err(ProxyError::Denied)?;

        // 2. SSRF prevention
        check_ssrf(&req.url).map_err(ProxyError::Denied)?;

        // 2b. Intent alignment check (rules-based, always runs)
        let intent_result = intent_check::check_intent_rules(&req.intent);
        if intent_result.is_denied() {
            return Err(ProxyError::Denied(format!(
                "intent check failed: {}",
                intent_result.denial_reason().unwrap_or("suspicious intent")
            )));
        }

        // 2c. Optional LLM intent alignment check
        if let Some(ref llm) = engine.llm {
            let llm_result = intent_check::check_intent_llm(
                &req.intent,
                &format!("{} {}", req.method, req.url),
                &req.secret_ref,
                llm.as_ref(),
            )
            .await;

            if llm_result.is_denied() {
                tracing::warn!(
                    consumer = consumer,
                    intent = req.intent,
                    reason = ?llm_result.denial_reason(),
                    "intent flagged by LLM alignment check"
                );
            }
        }

        // 3. Policy check
        let decision = engine
            .check_policy(&req.secret_ref, consumer)
            .await
            .map_err(|e| ProxyError::Failed(format!("policy check failed: {e}")))?;

        // 3a. Handle RequiresApproval
        if let PolicyDecision::RequiresApproval { ttl_seconds, ref scopes } = decision {
            // Check if there's an existing approved approval
            let existing = engine
                .find_active_approval(consumer, &req.secret_ref)
                .await
                .map_err(|e| ProxyError::Failed(format!("approval lookup failed: {e}")))?;

            if existing.is_none() {
                // Create a new approval request
                let approval = ApprovalRequest::new(
                    consumer,
                    &req.secret_ref,
                    &req.intent,
                    format!("{} {}", req.method, req.url),
                    ttl_seconds,
                    scopes.clone(),
                );
                let approval_id = approval.id;
                engine
                    .create_approval(&approval)
                    .await
                    .map_err(|e| ProxyError::Failed(format!("create approval failed: {e}")))?;

                return Err(ProxyError::ApprovalRequired {
                    approval_id,
                    message: format!(
                        "policy requires approval; approval_id={approval_id}, expires in {ttl_seconds}s"
                    ),
                });
            }
            // If approved, fall through to execute
        }

        if !decision.is_allowed() && !decision.requires_approval() {
            return Err(ProxyError::Denied(format!(
                "access denied: consumer '{consumer}' has no policy granting access to '{}'",
                req.secret_ref
            )));
        }

        // 3b. Scope enforcement
        let scopes = match &decision {
            PolicyDecision::Allow { scopes, .. } => scopes,
            PolicyDecision::RequiresApproval { scopes, .. } => scopes,
            _ => unreachable!(),
        };
        let required = derive_http_scopes(&req.method);
        check_scopes(&required, &scopes).map_err(|e| {
            ProxyError::Denied(format!("access denied for '{}': {e}", req.secret_ref))
        })?;

        // 4. Resolve the secret value
        let secret_value = engine
            .credential_store
            .get(&req.secret_ref)
            .map_err(|e| ProxyError::Failed(format!("credential lookup failed: {e}")))?
            .ok_or_else(|| ProxyError::Failed(format!("secret '{}' not found", req.secret_ref)))?;
        let secret_str = Zeroizing::new(secret_value.expose().to_string());

        // 5. Create audit entry
        let audit = ProxyAuditEntry::new(
            consumer,
            &req.secret_ref,
            "http",
            &req.url,
            &req.intent,
            format!("{} {}", req.method, req.url),
        );

        let _ = engine.log_proxy_audit(&audit).await;

        // 6. Build and send HTTP request
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .redirect(reqwest::redirect::Policy::limited(5))
            .build()
            .map_err(|e| ProxyError::Failed(format!("failed to create HTTP client: {e}")))?;

        let method = reqwest::Method::from_bytes(req.method.as_bytes())
            .map_err(|e| ProxyError::Failed(format!("invalid HTTP method: {e}")))?;

        let mut request_builder = client.request(method.clone(), &req.url);

        // Add user-provided headers
        for (key, value) in &req.headers {
            request_builder = request_builder.header(key.as_str(), value.as_str());
        }

        // Inject credential
        request_builder = match &req.inject_as {
            SecretInjection::BearerHeader => {
                request_builder.header("Authorization", format!("Bearer {}", &*secret_str))
            }
            SecretInjection::BasicAuth { username } => {
                use base64::engine::general_purpose::STANDARD;
                use base64::Engine;
                let encoded = STANDARD.encode(format!("{}:{}", username, &*secret_str));
                request_builder.header("Authorization", format!("Basic {encoded}"))
            }
            SecretInjection::Header { name } => {
                request_builder.header(name.as_str(), secret_str.as_str())
            }
            SecretInjection::QueryParam { name } => {
                let mut url =
                    url::Url::parse(&req.url).map_err(|e| ProxyError::Failed(format!("invalid URL: {e}")))?;
                url.query_pairs_mut().append_pair(name, &*secret_str);
                // Rebuild the request with the modified URL
                let mut rebuilt = client.request(method, url.as_str());
                for (key, value) in &req.headers {
                    rebuilt = rebuilt.header(key.as_str(), value.as_str());
                }
                rebuilt
            }
        };

        // Add body if present
        if let Some(body) = &req.body {
            request_builder = request_builder.body(body.clone());
        }

        let result = request_builder.send().await;

        match result {
            Ok(response) => {
                let status = response.status().as_u16();
                let resp_headers: HashMap<String, String> = response
                    .headers()
                    .iter()
                    .map(|(k, v)| {
                        (
                            k.as_str().to_string(),
                            v.to_str().unwrap_or("<binary>").to_string(),
                        )
                    })
                    .collect();

                let body_text = response
                    .text()
                    .await
                    .unwrap_or_else(|e| format!("<error reading body: {e}>"));

                // Sanitize output — clone into sanitizer; secret_str is zeroized on drop
                let secrets_for_sanitize = vec![(req.secret_ref.clone(), (*secret_str).clone())];
                let (sanitized_body, was_sanitized) =
                    OutputSanitizer::sanitize(&body_text, &secrets_for_sanitize);

                // Update audit
                let _ = engine
                    .update_proxy_audit(audit.id, true, was_sanitized, None, Some(status as i32))
                    .await;

                Ok(HttpProxyResponse {
                    status,
                    headers: resp_headers,
                    body: sanitized_body,
                    sanitized: was_sanitized,
                })
            }
            Err(e) => {
                let err_msg = format!("HTTP request failed: {e}");
                let _ = engine
                    .update_proxy_audit(audit.id, false, false, Some(&err_msg), None)
                    .await;
                Err(ProxyError::Failed(err_msg))
            }
        }
    }

    /// Execute a command proxy request with credential env injection.
    ///
    /// Flow: constitutional check -> intent check -> allowlist check -> policy check -> scope check -> inject env -> run -> sanitize -> audit
    pub async fn execute_exec(
        req: &ExecProxyRequest,
        consumer: &str,
        engine: &HelixEngine,
    ) -> Result<ExecProxyResponse, ProxyError> {
        // 1. Constitutional rules
        let description = format!(
            "{} {} intent={}",
            req.command,
            req.args.join(" "),
            req.intent
        );
        check_constitutional_rules(&description).map_err(ProxyError::Denied)?;

        // 2. Intent alignment check (rules-based, always runs)
        let intent_result = intent_check::check_intent_rules(&req.intent);
        if intent_result.is_denied() {
            return Err(ProxyError::Denied(format!(
                "intent check failed: {}",
                intent_result.denial_reason().unwrap_or("suspicious intent")
            )));
        }

        // 2b. Optional LLM intent alignment check
        if let Some(ref llm) = engine.llm {
            let llm_result = intent_check::check_intent_llm(
                &req.intent,
                &format!("{} {}", req.command, req.args.join(" ")),
                &req.env_inject.values().map(|s| s.as_str()).collect::<Vec<_>>().join(","),
                llm.as_ref(),
            )
            .await;

            if llm_result.is_denied() {
                tracing::warn!(
                    consumer = consumer,
                    intent = req.intent,
                    reason = ?llm_result.denial_reason(),
                    "intent flagged by LLM alignment check for command proxy"
                );
            }
        }

        // 3. Command allowlist check
        if !DEFAULT_EXEC_ALLOWLIST.contains(&req.command.as_str()) {
            return Err(ProxyError::Denied(format!(
                "command '{}' is not in the allowed list: {:?}",
                req.command, DEFAULT_EXEC_ALLOWLIST
            )));
        }

        // 4. Policy check for each secret being injected
        let mut resolved_secrets: Vec<(String, Zeroizing<String>)> = Vec::new();
        for (env_var, secret_key) in &req.env_inject {
            let decision = engine
                .check_policy(secret_key, consumer)
                .await
                .map_err(|e| ProxyError::Failed(format!("policy check failed for '{secret_key}': {e}")))?;

            // Handle RequiresApproval
            if let PolicyDecision::RequiresApproval { ttl_seconds, ref scopes } = decision {
                let existing = engine
                    .find_active_approval(consumer, secret_key)
                    .await
                    .map_err(|e| ProxyError::Failed(format!("approval lookup failed: {e}")))?;

                if existing.is_none() {
                    let approval = ApprovalRequest::new(
                        consumer,
                        secret_key,
                        &req.intent,
                        format!("{} {}", req.command, req.args.join(" ")),
                        ttl_seconds,
                        scopes.clone(),
                    );
                    let approval_id = approval.id;
                    engine
                        .create_approval(&approval)
                        .await
                        .map_err(|e| ProxyError::Failed(format!("create approval failed: {e}")))?;

                    return Err(ProxyError::ApprovalRequired {
                        approval_id,
                        message: format!(
                            "policy requires approval for '{secret_key}'; approval_id={approval_id}"
                        ),
                    });
                }
            }

            if !decision.is_allowed() && !decision.requires_approval() {
                return Err(ProxyError::Denied(format!(
                    "access denied: consumer '{consumer}' cannot access secret '{secret_key}'"
                )));
            }

            // Scope enforcement
            let scopes = match &decision {
                PolicyDecision::Allow { scopes, .. } => scopes,
                PolicyDecision::RequiresApproval { scopes, .. } => scopes,
                _ => unreachable!(),
            };
            let required = derive_exec_scopes(&req.command);
            check_scopes(&required, &scopes).map_err(|e| {
                ProxyError::Denied(format!("access denied for '{secret_key}': {e}"))
            })?;

            let secret_value = engine
                .credential_store
                .get(secret_key)
                .map_err(|e| ProxyError::Failed(format!("credential lookup failed for '{secret_key}': {e}")))?
                .ok_or_else(|| ProxyError::Failed(format!("secret '{secret_key}' not found")))?;

            resolved_secrets.push((env_var.clone(), Zeroizing::new(secret_value.expose().to_string())));
        }

        // Build secret ref string for audit
        let secret_refs: Vec<&str> = req.env_inject.values().map(|v| v.as_str()).collect();
        let secret_ref_str = secret_refs.join(",");

        // 4. Create audit entry
        let audit = ProxyAuditEntry::new(
            consumer,
            &secret_ref_str,
            "exec",
            &req.command,
            &req.intent,
            format!("{} {}", req.command, req.args.join(" ")),
        );

        let _ = engine.log_proxy_audit(&audit).await;

        // 5. Build and run process (using tokio::process::Command for safe arg handling)
        let mut cmd = tokio::process::Command::new(&req.command);
        cmd.args(&req.args);

        // Inject secret env vars (deref Zeroizing to get &str)
        for (env_var, value) in &resolved_secrets {
            cmd.env(env_var, &**value);
        }

        // Set working directory if specified
        if let Some(ref wd) = req.working_dir {
            cmd.current_dir(wd);
        }

        // Capture output
        cmd.stdout(std::process::Stdio::piped());
        cmd.stderr(std::process::Stdio::piped());

        let timeout = Duration::from_secs(req.timeout_seconds);

        let child = cmd
            .spawn()
            .map_err(|e| ProxyError::Failed(format!("failed to spawn command '{}': {e}", req.command)))?;

        let output = tokio::time::timeout(timeout, child.wait_with_output())
            .await
            .map_err(|_| {
                ProxyError::Failed(format!(
                    "command '{}' timed out after {}s",
                    req.command, req.timeout_seconds
                ))
            })?
            .map_err(|e| ProxyError::Failed(format!("command '{}' failed: {e}", req.command)))?;

        let stdout_raw = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr_raw = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        // 6. Sanitize output — redact all injected secret values
        // Clone plaintext out of Zeroizing for sanitizer; sanitizer will zeroize its own copies
        let all_secrets: Vec<(String, String)> = req
            .env_inject
            .values()
            .zip(resolved_secrets.iter().map(|(_, v)| v))
            .map(|(key, val)| (key.clone(), val.as_str().to_string()))
            .collect();

        let (sanitized_stdout, stdout_redacted) =
            OutputSanitizer::sanitize(&stdout_raw, &all_secrets);
        let (sanitized_stderr, stderr_redacted) =
            OutputSanitizer::sanitize(&stderr_raw, &all_secrets);
        let was_sanitized = stdout_redacted || stderr_redacted;

        // Update audit
        let success = exit_code == 0;
        let error_msg = if !success {
            Some(format!("exit code {exit_code}"))
        } else {
            None
        };

        let _ = engine
            .update_proxy_audit(
                audit.id,
                success,
                was_sanitized,
                error_msg.as_deref(),
                Some(exit_code),
            )
            .await;

        Ok(ExecProxyResponse {
            exit_code,
            stdout: sanitized_stdout,
            stderr: sanitized_stderr,
            sanitized: was_sanitized,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssrf_blocks_localhost() {
        assert!(check_ssrf("http://localhost:8080/api").is_err());
    }

    #[test]
    fn ssrf_blocks_private_ipv4() {
        assert!(check_ssrf("http://10.0.0.1/secret").is_err());
        assert!(check_ssrf("http://172.16.0.1/secret").is_err());
        assert!(check_ssrf("http://192.168.1.1/secret").is_err());
        assert!(check_ssrf("http://127.0.0.1:9470/api").is_err());
    }

    #[test]
    fn ssrf_blocks_ipv6_loopback() {
        assert!(check_ssrf("http://[::1]:8080/api").is_err());
    }

    #[test]
    fn ssrf_allows_public_ip() {
        assert!(check_ssrf("https://8.8.8.8/dns-query").is_ok());
    }

    #[test]
    fn ssrf_blocks_metadata_endpoints() {
        assert!(check_ssrf("http://169.254.169.254/latest/meta-data/").is_err());
        assert!(check_ssrf("http://metadata.google.internal/computeMetadata/").is_err());
    }

    #[test]
    fn constitutional_blocks_dangerous_commands() {
        assert!(check_constitutional_rules("rm -rf / --no-preserve-root").is_err());
        assert!(check_constitutional_rules("dd if=/dev/zero of=/dev/sda").is_err());
    }

    #[test]
    fn constitutional_allows_safe_commands() {
        assert!(check_constitutional_rules("git status").is_ok());
        assert!(check_constitutional_rules("cargo build").is_ok());
    }

    // --- Scope enforcement tests ---

    #[test]
    fn derive_http_scopes_read_methods() {
        assert_eq!(derive_http_scopes("GET"), vec!["read".to_string()]);
        assert_eq!(derive_http_scopes("HEAD"), vec!["read".to_string()]);
        assert_eq!(derive_http_scopes("OPTIONS"), vec!["read".to_string()]);
    }

    #[test]
    fn derive_http_scopes_write_methods() {
        assert_eq!(derive_http_scopes("POST"), vec!["write".to_string()]);
        assert_eq!(derive_http_scopes("PUT"), vec!["write".to_string()]);
        assert_eq!(derive_http_scopes("PATCH"), vec!["write".to_string()]);
        assert_eq!(derive_http_scopes("DELETE"), vec!["write".to_string()]);
    }

    #[test]
    fn derive_exec_scopes_command() {
        assert_eq!(derive_exec_scopes("gh"), vec!["exec:gh".to_string()]);
        assert_eq!(derive_exec_scopes("git"), vec!["exec:git".to_string()]);
    }

    #[test]
    fn check_scopes_empty_allowed_is_unrestricted() {
        assert!(check_scopes(&["read".into()], &[]).is_ok());
        assert!(check_scopes(&["exec:gh".into()], &[]).is_ok());
    }

    #[test]
    fn check_scopes_exact_match() {
        assert!(check_scopes(&["read".into()], &["read".into()]).is_ok());
        assert!(check_scopes(&["write".into()], &["write".into()]).is_ok());
    }

    #[test]
    fn check_scopes_prefix_match() {
        // Allowed "exec" covers required "exec:git"
        assert!(check_scopes(&["exec:git".into()], &["exec".into()]).is_ok());
    }

    #[test]
    fn check_scopes_wildcard_match() {
        assert!(check_scopes(&["exec:git".into()], &["exec:*".into()]).is_ok());
        assert!(check_scopes(&["exec:gh".into()], &["exec:*".into()]).is_ok());
    }

    #[test]
    fn check_scopes_mismatch() {
        // Policy allows read only, but request requires write
        assert!(check_scopes(&["write".into()], &["read".into()]).is_err());
        // Policy allows exec:gh, but request requires exec:git
        assert!(check_scopes(&["exec:git".into()], &["exec:gh".into()]).is_err());
    }

    #[test]
    fn check_scopes_read_policy_blocks_post() {
        let required = derive_http_scopes("POST");
        assert!(check_scopes(&required, &["read".into()]).is_err());
    }

    #[test]
    fn check_scopes_read_policy_allows_get() {
        let required = derive_http_scopes("GET");
        assert!(check_scopes(&required, &["read".into()]).is_ok());
    }

    #[test]
    fn ssrf_blocks_ipv6_mapped_ipv4() {
        assert!(check_ssrf("http://[::ffff:127.0.0.1]:8080/api").is_err());
        assert!(check_ssrf("http://[::ffff:192.168.1.1]/secret").is_err());
        assert!(check_ssrf("http://[::ffff:10.0.0.1]/secret").is_err());
    }

    #[test]
    fn scope_matches_no_partial_prefix_confusion() {
        // "rea" should NOT match "read"
        assert!(!scope_matches("read", "rea"));
        // "exec:g" should NOT match "exec:git"
        assert!(!scope_matches("exec:git", "exec:g"));
    }
}
