use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing;

use hx_core::MvResult;

use super::worker::JobHandler;
use super::Job;

// ---------------------------------------------------------------------------
// Webhook Delivery Handler
// ---------------------------------------------------------------------------

/// Delivers webhook payloads to external URLs with optional HMAC signing.
pub struct WebhookDeliveryHandler {
    http_client: reqwest::Client,
}

#[derive(Debug, Deserialize, Serialize)]
struct WebhookPayload {
    url: String,
    body: serde_json::Value,
    #[serde(default)]
    headers: std::collections::HashMap<String, String>,
    hmac_secret: Option<String>,
}

impl WebhookDeliveryHandler {
    pub fn new(http_client: reqwest::Client) -> Self {
        Self { http_client }
    }
}

#[async_trait]
impl JobHandler for WebhookDeliveryHandler {
    fn handles(&self) -> &str {
        "webhook_delivery"
    }

    async fn execute(&self, job: &Job) -> MvResult<()> {
        let payload: WebhookPayload = serde_json::from_value(job.payload.clone())
            .map_err(|e| hx_core::HxError::InvalidInput(format!("invalid webhook payload: {e}")))?;

        tracing::info!(
            job_id = %job.id,
            url = %payload.url,
            "delivering webhook"
        );

        let body_str = serde_json::to_string(&payload.body)
            .map_err(|e| hx_core::HxError::Internal(format!("serialize body: {e}")))?;

        let mut request = self
            .http_client
            .post(&payload.url)
            .header("Content-Type", "application/json")
            .body(body_str.clone());

        for (key, value) in &payload.headers {
            request = request.header(key.as_str(), value.as_str());
        }

        if let Some(ref secret) = payload.hmac_secret {
            use hmac::{Hmac, Mac};
            use sha2::Sha256;

            type HmacSha256 = Hmac<Sha256>;
            let mut mac = HmacSha256::new_from_slice(secret.as_bytes())
                .map_err(|e| hx_core::HxError::Internal(format!("hmac init: {e}")))?;
            mac.update(body_str.as_bytes());
            let bytes = mac.finalize().into_bytes();
            let signature: String = bytes.iter().map(|b| format!("{b:02x}")).collect();
            request = request.header("X-Webhook-Signature", format!("sha256={signature}"));
        }

        let response = request
            .send()
            .await
            .map_err(|e| hx_core::HxError::Internal(format!("webhook send: {e}")))?;

        if !response.status().is_success() {
            return Err(hx_core::HxError::Internal(format!(
                "webhook returned status {}",
                response.status()
            )));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Source Poll Handler
// ---------------------------------------------------------------------------

/// Triggers polling for a source connector.
pub struct SourcePollHandler {
    source_registry: std::sync::Arc<crate::sources::SourceRegistry>,
}

impl SourcePollHandler {
    pub fn new(source_registry: std::sync::Arc<crate::sources::SourceRegistry>) -> Self {
        Self { source_registry }
    }
}

#[derive(Debug, Deserialize)]
struct SourcePollPayload {
    source_id: String,
}

#[async_trait]
impl JobHandler for SourcePollHandler {
    fn handles(&self) -> &str {
        "source_poll"
    }

    async fn execute(&self, job: &Job) -> MvResult<()> {
        let payload: SourcePollPayload = serde_json::from_value(job.payload.clone())
            .map_err(|e| hx_core::HxError::InvalidInput(format!("invalid source_poll payload: {e}")))?;

        tracing::info!(
            job_id = %job.id,
            source_id = %payload.source_id,
            "polling source"
        );

        let source_id = uuid::Uuid::parse_str(&payload.source_id)
            .map_err(|e| hx_core::HxError::InvalidInput(format!("invalid source_id uuid: {e}")))?;

        let docs = self.source_registry.poll(source_id).await?;
        tracing::info!(
            job_id = %job.id,
            source_id = %payload.source_id,
            documents = docs.len(),
            "source poll completed"
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Workflow Step Handler
// ---------------------------------------------------------------------------

/// Executes a single step in a workflow.
pub struct WorkflowStepHandler {
    executor: std::sync::Arc<crate::workflow::executor::WorkflowExecutor>,
}

impl WorkflowStepHandler {
    pub fn new(executor: std::sync::Arc<crate::workflow::executor::WorkflowExecutor>) -> Self {
        Self { executor }
    }
}

#[derive(Debug, Deserialize)]
struct WorkflowStepPayload {
    workflow_id: String,
    step_index: usize,
}

#[async_trait]
impl JobHandler for WorkflowStepHandler {
    fn handles(&self) -> &str {
        "workflow_step"
    }

    async fn execute(&self, job: &Job) -> MvResult<()> {
        let payload: WorkflowStepPayload = serde_json::from_value(job.payload.clone())
            .map_err(|e| hx_core::HxError::InvalidInput(format!("invalid workflow_step payload: {e}")))?;

        tracing::info!(
            job_id = %job.id,
            workflow_id = %payload.workflow_id,
            step_index = payload.step_index,
            "executing workflow step"
        );

        let workflow_id = uuid::Uuid::parse_str(&payload.workflow_id)
            .map_err(|e| hx_core::HxError::InvalidInput(format!("invalid workflow_id uuid: {e}")))?;

        let execution = self.executor.execute(workflow_id, std::collections::HashMap::new()).await?;
        tracing::info!(
            job_id = %job.id,
            workflow_id = %payload.workflow_id,
            execution_id = %execution.id,
            "workflow step execution completed"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use uuid::Uuid;

    use crate::jobs::JobStatus;

    fn make_job(job_type: &str, payload: serde_json::Value) -> Job {
        Job {
            id: Uuid::now_v7(),
            job_type: job_type.into(),
            payload,
            status: JobStatus::Running,
            priority: 0,
            retries: 0,
            max_retries: 3,
            error: None,
            created_at: Utc::now(),
            scheduled_at: None,
            started_at: Some(Utc::now()),
            completed_at: None,
            next_retry_at: None,
            idempotency_key: None,
        }
    }

    #[test]
    fn webhook_handler_type() {
        let handler = WebhookDeliveryHandler::new(reqwest::Client::new());
        assert_eq!(handler.handles(), "webhook_delivery");
    }

    #[tokio::test]
    async fn webhook_invalid_payload() {
        let handler = WebhookDeliveryHandler::new(reqwest::Client::new());
        let job = make_job("webhook_delivery", serde_json::json!({"bad": "data"}));
        let result = handler.execute(&job).await;
        assert!(result.is_err());
    }

    #[test]
    fn source_poll_handler_type() {
        let registry = std::sync::Arc::new(crate::sources::SourceRegistry::new());
        let handler = SourcePollHandler::new(registry);
        assert_eq!(handler.handles(), "source_poll");
    }

    #[tokio::test]
    async fn source_poll_valid_payload() {
        let registry = std::sync::Arc::new(crate::sources::SourceRegistry::new());
        let handler = SourcePollHandler::new(registry);
        // source_id must be a valid UUID; the source won't exist, so this should fail
        let job = make_job("source_poll", serde_json::json!({"source_id": "00000000-0000-0000-0000-000000000001"}));
        let result = handler.execute(&job).await;
        // Source not found in registry → error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn source_poll_invalid_payload() {
        let registry = std::sync::Arc::new(crate::sources::SourceRegistry::new());
        let handler = SourcePollHandler::new(registry);
        let job = make_job("source_poll", serde_json::json!({"wrong": "field"}));
        let result = handler.execute(&job).await;
        assert!(result.is_err());
    }

    #[test]
    fn workflow_step_handler_type() {
        let executor = std::sync::Arc::new(crate::workflow::executor::WorkflowExecutor::new());
        let handler = WorkflowStepHandler::new(executor);
        assert_eq!(handler.handles(), "workflow_step");
    }

    #[tokio::test]
    async fn workflow_step_valid_payload() {
        let executor = std::sync::Arc::new(crate::workflow::executor::WorkflowExecutor::new());
        let handler = WorkflowStepHandler::new(executor);
        // workflow_id must be a valid UUID; workflow won't exist, so this should fail
        let job = make_job(
            "workflow_step",
            serde_json::json!({"workflow_id": "00000000-0000-0000-0000-000000000001", "step_index": 2}),
        );
        let result = handler.execute(&job).await;
        // Workflow not found in executor → error
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn workflow_step_invalid_payload() {
        let executor = std::sync::Arc::new(crate::workflow::executor::WorkflowExecutor::new());
        let handler = WorkflowStepHandler::new(executor);
        let job = make_job("workflow_step", serde_json::json!({}));
        let result = handler.execute(&job).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn webhook_payload_deserialization() {
        let payload = serde_json::json!({
            "url": "https://example.com/hook",
            "body": {"event": "test"},
            "headers": {"X-Custom": "value"},
            "hmac_secret": "secret123"
        });
        let parsed: WebhookPayload = serde_json::from_value(payload).unwrap();
        assert_eq!(parsed.url, "https://example.com/hook");
        assert_eq!(parsed.hmac_secret.as_deref(), Some("secret123"));
        assert_eq!(parsed.headers.get("X-Custom").unwrap(), "value");
    }

    #[tokio::test]
    async fn webhook_payload_without_optional_fields() {
        let payload = serde_json::json!({
            "url": "https://example.com/hook",
            "body": {"event": "test"}
        });
        let parsed: WebhookPayload = serde_json::from_value(payload).unwrap();
        assert!(parsed.headers.is_empty());
        assert!(parsed.hmac_secret.is_none());
    }
}
