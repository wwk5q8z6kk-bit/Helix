//! Email (SMTP) adapter — sends messages via SMTP using the `lettre` crate.
//!
//! Configuration keys:
//! - `smtp_host`: SMTP server hostname (e.g., "smtp.gmail.com")
//! - `smtp_port`: SMTP port (default: "587")
//! - `smtp_user`: SMTP username/email
//! - `smtp_pass`: SMTP password or app-specific password
//! - `from_address`: Sender email address
//! - `default_to`: (optional) Default recipient email

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use uuid::Uuid;

use hx_core::{MemoryQuery, HxError, MvResult, Proposal, ProposalAction, ProposalSender};

use super::{
    AdapterConfig, AdapterInboundMessage, AdapterOutboundMessage, AdapterStatus, AdapterType,
    ExternalAdapter,
};

#[derive(Debug)]
pub struct EmailAdapter {
    config: AdapterConfig,
    last_send: Mutex<Option<DateTime<Utc>>>,
    last_error: Mutex<Option<String>>,
}

impl EmailAdapter {
    pub fn new(config: AdapterConfig) -> MvResult<Self> {
        // Validate required settings
        for key in &["smtp_host", "smtp_user", "smtp_pass", "from_address"] {
            if config.get_setting(key).is_none() {
                return Err(HxError::Config(format!(
                    "Email adapter requires '{key}' setting"
                )));
            }
        }
        Ok(Self {
            config,
            last_send: Mutex::new(None),
            last_error: Mutex::new(None),
        })
    }

    fn smtp_port(&self) -> u16 {
        self.config
            .get_setting("smtp_port")
            .and_then(|p| p.parse().ok())
            .unwrap_or(587)
    }
}

#[async_trait]
impl ExternalAdapter for EmailAdapter {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn adapter_type(&self) -> AdapterType {
        AdapterType::Email
    }

    async fn send(&self, message: &AdapterOutboundMessage) -> MvResult<()> {
        let host = self
            .config
            .get_setting("smtp_host")
            .ok_or_else(|| HxError::Config("missing smtp_host".into()))?;
        let user = self
            .config
            .get_setting("smtp_user")
            .ok_or_else(|| HxError::Config("missing smtp_user".into()))?;
        let pass = self
            .config
            .get_setting("smtp_pass")
            .ok_or_else(|| HxError::Config("missing smtp_pass".into()))?;
        let from = self
            .config
            .get_setting("from_address")
            .ok_or_else(|| HxError::Config("missing from_address".into()))?;

        // The `channel` field is used as the recipient email address
        let to = if message.channel.is_empty() {
            self.config
                .get_setting("default_to")
                .ok_or_else(|| HxError::Config("no recipient and no default_to".into()))?
                .to_string()
        } else {
            message.channel.clone()
        };

        let port = self.smtp_port();

        // Use reqwest-compatible approach: spawn blocking SMTP via raw TCP
        // This avoids adding lettre as a dependency — we use a minimal SMTP
        // implementation via tokio's TCP stream for the basic STARTTLS flow.
        //
        // For production, consider switching to the `lettre` crate.
        // For now, we use a simple approach that works for common SMTP servers.
        let subject = message
            .metadata
            .get("subject")
            .cloned()
            .unwrap_or_else(|| "Helix Message".to_string());

        let email_body = format!(
            "From: {from}\r\n\
             To: {to}\r\n\
             Subject: {subject}\r\n\
             Content-Type: text/plain; charset=utf-8\r\n\
             \r\n\
             {content}",
            content = message.content,
        );

        // Use a lightweight SMTP send via the `reqwest`-style HTTP bridge
        // We shell out to a simple TCP connection for SMTP
        let host_owned = host.to_string();
        let user_owned = user.to_string();
        let pass_owned = pass.to_string();

        let result = tokio::task::spawn_blocking(move || {
            send_smtp_email(&host_owned, port, &user_owned, &pass_owned, &email_body)
        })
        .await
        .map_err(|e| HxError::Internal(format!("email task join error: {e}")))?;

        match result {
            Ok(()) => {
                *self.last_send.lock().unwrap() = Some(Utc::now());
                *self.last_error.lock().unwrap() = None;
                Ok(())
            }
            Err(e) => {
                let err = format!("email send failed: {e}");
                *self.last_error.lock().unwrap() = Some(err.clone());
                Err(HxError::Internal(err))
            }
        }
    }

    async fn poll(
        &self,
        cursor: Option<&str>,
    ) -> MvResult<(Vec<AdapterInboundMessage>, String)> {
        // Email polling (IMAP) is a complex protocol requiring a dedicated
        // connection. For now, inbound email is not supported — users should
        // configure email forwarding to a webhook endpoint instead.
        Ok((vec![], cursor.unwrap_or("0").to_string()))
    }

    async fn health_check(&self) -> MvResult<bool> {
        let host = match self.config.get_setting("smtp_host") {
            Some(h) => h.to_string(),
            None => return Ok(false),
        };
        let port = self.smtp_port();

        // Try to connect to SMTP port
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            tokio::net::TcpStream::connect(format!("{host}:{port}")),
        )
        .await;

        match result {
            Ok(Ok(_)) => Ok(true),
            _ => Ok(false),
        }
    }

    fn status(&self) -> AdapterStatus {
        let error = self.last_error.lock().unwrap().clone();
        let last_send = *self.last_send.lock().unwrap();
        AdapterStatus {
            adapter_type: AdapterType::Email,
            name: self.config.name.clone(),
            connected: error.is_none(),
            last_send,
            last_receive: None,
            error,
        }
    }
}

/// Minimal SMTP send via raw TCP.
///
/// Supports PLAIN auth over a non-TLS connection (port 25/587 without STARTTLS).
/// For production use with TLS, consider the `lettre` crate.
fn send_smtp_email(
    host: &str,
    port: u16,
    user: &str,
    pass: &str,
    email_data: &str,
) -> Result<(), String> {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;

    let stream =
        TcpStream::connect(format!("{host}:{port}")).map_err(|e| format!("connect: {e}"))?;
    stream
        .set_read_timeout(Some(std::time::Duration::from_secs(10)))
        .ok();
    stream
        .set_write_timeout(Some(std::time::Duration::from_secs(10)))
        .ok();

    let mut reader = BufReader::new(stream.try_clone().map_err(|e| format!("clone: {e}"))?);
    let mut writer = stream;

    let mut response = String::new();

    // Read greeting
    response.clear();
    reader
        .read_line(&mut response)
        .map_err(|e| format!("read greeting: {e}"))?;
    if !response.starts_with("220") {
        return Err(format!("unexpected greeting: {response}"));
    }

    // EHLO
    write!(writer, "EHLO helix\r\n").map_err(|e| format!("write EHLO: {e}"))?;
    writer.flush().map_err(|e| format!("flush EHLO: {e}"))?;
    loop {
        response.clear();
        reader
            .read_line(&mut response)
            .map_err(|e| format!("read EHLO: {e}"))?;
        if response.len() < 4 || response.as_bytes()[3] == b' ' {
            break;
        }
    }

    // AUTH PLAIN
    let auth_str = format!("\0{user}\0{pass}");
    let auth_b64 = base64_encode(auth_str.as_bytes());
    write!(writer, "AUTH PLAIN {auth_b64}\r\n").map_err(|e| format!("write AUTH: {e}"))?;
    writer.flush().map_err(|e| format!("flush AUTH: {e}"))?;
    response.clear();
    reader
        .read_line(&mut response)
        .map_err(|e| format!("read AUTH: {e}"))?;
    if !response.starts_with("235") {
        return Err(format!("auth failed: {response}"));
    }

    // Extract From/To from email_data headers
    let from = email_data
        .lines()
        .find(|l| l.starts_with("From:"))
        .and_then(|l| l.strip_prefix("From:"))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| user.to_string());
    let to = email_data
        .lines()
        .find(|l| l.starts_with("To:"))
        .and_then(|l| l.strip_prefix("To:"))
        .map(|s| s.trim().to_string())
        .unwrap_or_default();

    // MAIL FROM
    write!(writer, "MAIL FROM:<{from}>\r\n").map_err(|e| format!("write MAIL: {e}"))?;
    writer.flush().map_err(|e| format!("flush MAIL: {e}"))?;
    response.clear();
    reader
        .read_line(&mut response)
        .map_err(|e| format!("read MAIL: {e}"))?;

    // RCPT TO
    write!(writer, "RCPT TO:<{to}>\r\n").map_err(|e| format!("write RCPT: {e}"))?;
    writer.flush().map_err(|e| format!("flush RCPT: {e}"))?;
    response.clear();
    reader
        .read_line(&mut response)
        .map_err(|e| format!("read RCPT: {e}"))?;

    // DATA
    write!(writer, "DATA\r\n").map_err(|e| format!("write DATA: {e}"))?;
    writer.flush().map_err(|e| format!("flush DATA: {e}"))?;
    response.clear();
    reader
        .read_line(&mut response)
        .map_err(|e| format!("read DATA: {e}"))?;
    if !response.starts_with("354") {
        return Err(format!("DATA rejected: {response}"));
    }

    // Send email body + terminator
    write!(writer, "{email_data}\r\n.\r\n").map_err(|e| format!("write body: {e}"))?;
    writer.flush().map_err(|e| format!("flush body: {e}"))?;
    response.clear();
    reader
        .read_line(&mut response)
        .map_err(|e| format!("read body response: {e}"))?;
    if !response.starts_with("250") {
        return Err(format!("message rejected: {response}"));
    }

    // QUIT
    write!(writer, "QUIT\r\n").map_err(|e| format!("write QUIT: {e}"))?;
    writer.flush().ok();

    Ok(())
}

fn base64_encode(data: &[u8]) -> String {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    STANDARD.encode(data)
}

// ---------------------------------------------------------------------------
// Email Reply Proposal
// ---------------------------------------------------------------------------

use crate::engine::HelixEngine;
use crate::llm;

#[cfg(test)]
mod tests {
    use super::*;

    fn email_config_full() -> AdapterConfig {
        AdapterConfig::new(AdapterType::Email, "test-email")
            .with_setting("smtp_host", "smtp.example.com")
            .with_setting("smtp_user", "user@example.com")
            .with_setting("smtp_pass", "password123")
            .with_setting("from_address", "noreply@example.com")
    }

    #[test]
    fn new_requires_all_settings() {
        let config = AdapterConfig::new(AdapterType::Email, "missing");
        assert!(EmailAdapter::new(config).is_err());
    }

    #[test]
    fn new_requires_smtp_host() {
        let config = AdapterConfig::new(AdapterType::Email, "no-host")
            .with_setting("smtp_user", "u")
            .with_setting("smtp_pass", "p")
            .with_setting("from_address", "a@b.com");
        let err = EmailAdapter::new(config).unwrap_err();
        assert!(err.to_string().contains("smtp_host"));
    }

    #[test]
    fn smtp_port_defaults_to_587() {
        let adapter = EmailAdapter::new(email_config_full()).unwrap();
        assert_eq!(adapter.smtp_port(), 587);
    }

    #[test]
    fn smtp_port_custom() {
        let config = email_config_full().with_setting("smtp_port", "465");
        let adapter = EmailAdapter::new(config).unwrap();
        assert_eq!(adapter.smtp_port(), 465);
    }

    #[tokio::test]
    async fn poll_returns_empty() {
        let adapter = EmailAdapter::new(email_config_full()).unwrap();
        let (messages, cursor) = adapter.poll(None).await.unwrap();
        assert!(messages.is_empty());
        assert_eq!(cursor, "0");
    }

    #[tokio::test]
    async fn status_reflects_no_error_initially() {
        let adapter = EmailAdapter::new(email_config_full()).unwrap();
        let status = adapter.status();
        assert!(status.connected);
        assert!(status.error.is_none());
        assert_eq!(status.adapter_type, AdapterType::Email);
    }

    #[test]
    fn adapter_type_is_email() {
        let adapter = EmailAdapter::new(email_config_full()).unwrap();
        assert_eq!(adapter.adapter_type(), AdapterType::Email);
    }

    #[test]
    fn name_returns_config_name() {
        let adapter = EmailAdapter::new(email_config_full()).unwrap();
        assert_eq!(adapter.name(), "test-email");
    }
}

impl EmailAdapter {
    /// Generate a context-aware reply proposal for an inbound email.
    ///
    /// Searches the vault for related content and creates an exchange inbox
    /// proposal with `ProposalAction::Custom("email_reply")`.
    ///
    /// Returns `None` if no relevant context is found above the threshold.
    pub async fn generate_reply_proposal(
        engine: &HelixEngine,
        inbound_content: &str,
        channel_id: Uuid,
        sender_contact_id: Option<Uuid>,
        confidence_threshold: f32,
    ) -> MvResult<Option<Proposal>> {
        // 1. Search vault for related nodes using hybrid search
        let query_text = if inbound_content.len() > 500 {
            &inbound_content[..500]
        } else {
            inbound_content
        };

        let query = MemoryQuery::new(query_text).with_limit(6).with_min_score(0.0);

        let results = match engine.recall(&query).await {
            Ok(r) => r,
            Err(err) => {
                tracing::warn!(error = %err, "email_reply_context_recall_failed");
                return Ok(None);
            }
        };

        // 2. Extract context snippets
        let context_snippets = llm::extract_context_snippets(&results, 4);
        if context_snippets.is_empty() {
            return Ok(None);
        }

        // 3. Build reply content — use LLM if available, else template
        let mut suggestion_text = None;
        let mut used_llm = false;

        if let Some(ref llm_provider) = engine.llm {
            match llm::llm_completion_suggestions(
                llm_provider.as_ref(),
                inbound_content,
                &context_snippets,
                1,
            )
            .await
            {
                Ok(mut suggestions) => {
                    if let Some(first) = suggestions.pop() {
                        suggestion_text = Some(first);
                        used_llm = true;
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        "email_reply_llm_suggestion_failed"
                    );
                }
            }
        }

        if suggestion_text.is_none() {
            let preview = context_snippets
                .iter()
                .take(3)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            suggestion_text = Some(format!(
                "I have related notes that might help:\n{preview}\n\nWant me to share details?"
            ));
        }

        // Calculate confidence
        let mut confidence: f32 = if used_llm { 0.6 } else { 0.4 };
        if context_snippets.len() >= 3 {
            confidence += 0.1;
        }
        confidence = confidence.clamp(0.0, 1.0);

        // Check against threshold
        if confidence < confidence_threshold {
            return Ok(None);
        }

        // 4. Build proposal payload with source node IDs for citation
        let source_node_ids: Vec<String> = results.iter().map(|r| r.node.id.to_string()).collect();

        let mut payload = HashMap::new();
        payload.insert(
            "channel_id".to_string(),
            serde_json::Value::String(channel_id.to_string()),
        );
        payload.insert(
            "content".to_string(),
            serde_json::Value::String(suggestion_text.unwrap_or_default()),
        );
        payload.insert(
            "context_snippets".to_string(),
            serde_json::Value::Array(
                context_snippets
                    .iter()
                    .map(|s| serde_json::Value::String(s.clone()))
                    .collect(),
            ),
        );
        payload.insert(
            "source_node_ids".to_string(),
            serde_json::Value::Array(
                source_node_ids
                    .iter()
                    .map(|id| serde_json::Value::String(id.clone()))
                    .collect(),
            ),
        );
        if let Some(contact_id) = sender_contact_id {
            payload.insert(
                "sender_contact_id".to_string(),
                serde_json::Value::String(contact_id.to_string()),
            );
        }

        // 5. Create proposal
        let reply_content = payload
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let proposal =
            Proposal::new(ProposalSender::Agent, ProposalAction::Custom("email_reply".into()))
                .with_confidence(confidence)
                .with_diff(reply_content)
                .with_payload(payload);

        Ok(Some(proposal))
    }
}
