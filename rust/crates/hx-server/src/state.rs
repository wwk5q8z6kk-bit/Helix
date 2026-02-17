use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use hx_engine::engine::HelixEngine;
use hx_plugin::{PluginManager, PluginRegistry, PluginRuntime};
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, Mutex, RwLock};
use uuid::Uuid;

pub use hx_core::ChangeNotification;
use hx_core::{CapturedIntent, ChronicleEntry, ProactiveInsight};

/// Shared application state.
pub struct AppState {
    pub engine: Arc<HelixEngine>,
    pub change_tx: broadcast::Sender<ChangeNotification>,
    pub reminder_tx: broadcast::Sender<ReminderNotification>,
    pub agent_tx: broadcast::Sender<AgentNotification>,
    pub webhook_config: WebhookConfig,
    pub plugin_registry: Arc<RwLock<PluginRegistry>>,
    pub plugin_manager: Arc<RwLock<PluginManager>>,
    pub plugin_runtime: Arc<RwLock<PluginRuntime>>,
    /// Shared HTTP client for AI sidecar proxy and other outbound requests.
    pub http_client: reqwest::Client,
    /// Counter: requests rejected by sealed-mode middleware.
    pub sealed_blocked_requests: AtomicU64,
    /// In-memory store for gateway pairing sessions.
    pub pairing_store: Arc<Mutex<HashMap<Uuid, hx_core::model::pairing::PairingSession>>>,
    /// Source connector registry (Phase 11).
    pub source_registry: Arc<hx_engine::sources::SourceRegistry>,
    /// Workflow executor (Phase 12).
    pub workflow_executor: Arc<hx_engine::workflow::executor::WorkflowExecutor>,
    /// Job queue (Phase 13).
    pub job_queue: Option<Arc<hx_engine::jobs::queue::JobQueue>>,
    /// In-app notification channel (Phase 14).
    pub in_app_channel: Arc<hx_engine::notifications::channels::in_app::InAppChannel>,
    /// Notification router (Phase 14).
    pub notification_router: Arc<hx_engine::notifications::router::NotificationRouter>,
    /// Rate limiter (Phase 16).
    pub rate_limiter: Arc<hx_engine::rate_limit::RateLimiter>,
    /// DAG-based scheduling service (Phase 17).
    pub scheduling: Option<hx_engine::scheduling::SchedulingService>,
}

/// Notification for task reminders.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReminderNotification {
    pub node_id: String,
    pub title: Option<String>,
    pub content_preview: String,
    pub due_at: Option<DateTime<Utc>>,
    pub namespace: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub notification_type: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentRelatedNode {
    pub id: String,
    pub title: String,
    pub updated_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentNotification {
    Chronicle {
        entry: ChronicleEntry,
        namespace: Option<String>,
    },
    Intent {
        intent: CapturedIntent,
        namespace: Option<String>,
    },
    InsightDiscovered {
        insight: ProactiveInsight,
        namespace: Option<String>,
    },
    RelatedContext {
        nodes: Vec<AgentRelatedNode>,
        namespace: Option<String>,
    },
    NodeEnriched {
        node_id: String,
        namespace: Option<String>,
    },
}

impl AgentNotification {
    pub fn namespace(&self) -> Option<&str> {
        match self {
            Self::Chronicle { namespace, .. }
            | Self::Intent { namespace, .. }
            | Self::InsightDiscovered { namespace, .. }
            | Self::RelatedContext { namespace, .. }
            | Self::NodeEnriched { namespace, .. } => namespace.as_deref(),
        }
    }
}

/// Configuration for webhook notifications.
#[derive(Clone, Debug, Default)]
pub struct WebhookConfig {
    pub reminder_url: Option<String>,
    pub change_url: Option<String>,
    pub keychain_alert_url: Option<String>,
    pub timeout_secs: u64,
}

impl WebhookConfig {
    pub fn from_env() -> Self {
        Self {
            reminder_url: std::env::var("HELIX_WEBHOOK_REMINDER_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            change_url: std::env::var("HELIX_WEBHOOK_CHANGE_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            keychain_alert_url: std::env::var("HELIX_WEBHOOK_KEYCHAIN_ALERT_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            timeout_secs: std::env::var("HELIX_WEBHOOK_TIMEOUT_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
        }
    }
}

impl AppState {
    pub fn new(engine: Arc<HelixEngine>) -> Self {
        let (change_tx, _) = broadcast::channel(256);
        Self::new_with_change_tx(engine, change_tx)
    }

    pub fn new_with_change_tx(
        engine: Arc<HelixEngine>,
        change_tx: broadcast::Sender<ChangeNotification>,
    ) -> Self {
        let (agent_tx, _) = broadcast::channel(256);
        Self::new_with_channels(engine, change_tx, agent_tx)
    }

    pub fn new_with_channels(
        engine: Arc<HelixEngine>,
        change_tx: broadcast::Sender<ChangeNotification>,
        agent_tx: broadcast::Sender<AgentNotification>,
    ) -> Self {
        let (reminder_tx, _) = broadcast::channel(256);
        let plugins_dir = std::env::var("HELIX_PLUGINS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("plugins"));
        let plugin_manager = PluginManager::new(plugins_dir);
        let plugin_runtime = PluginRuntime::new(plugin_manager.clone());
        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("failed to build HTTP client");
        // Notification subsystem: in-app channel + alert store + router
        let in_app_channel = Arc::new(hx_engine::notifications::channels::in_app::InAppChannel::new());
        let alert_store = Arc::new(hx_engine::notifications::alerts::AlertRuleStore::new());
        let mut notification_router = hx_engine::notifications::router::NotificationRouter::new(
            Arc::clone(&alert_store),
        );
        notification_router.add_channel(
            Arc::clone(&in_app_channel) as Arc<dyn hx_engine::notifications::NotificationChannel>,
        );

        // Job queue: attempt SQLite-backed queue, fall back to None on error
        let job_queue = {
            let db_path = format!("{}/jobs.sqlite", engine.config.data_dir);
            match hx_engine::jobs::queue::JobQueue::new(&db_path) {
                Ok(q) => {
                    tracing::info!(path = %db_path, "job queue initialized");
                    Some(Arc::new(q))
                }
                Err(e) => {
                    tracing::warn!(error = %e, "job queue initialization failed; jobs disabled");
                    None
                }
            }
        };

        Self {
            engine,
            change_tx,
            reminder_tx,
            agent_tx,
            webhook_config: WebhookConfig::from_env(),
            plugin_registry: Arc::new(RwLock::new(PluginRegistry::new())),
            plugin_manager: Arc::new(RwLock::new(plugin_manager)),
            plugin_runtime: Arc::new(RwLock::new(plugin_runtime)),
            http_client,
            sealed_blocked_requests: AtomicU64::new(0),
            pairing_store: Arc::new(Mutex::new(HashMap::new())),
            source_registry: Arc::new(hx_engine::sources::SourceRegistry::new()),
            workflow_executor: Arc::new(hx_engine::workflow::executor::WorkflowExecutor::new()),
            job_queue,
            in_app_channel,
            notification_router: Arc::new(notification_router),
            rate_limiter: Arc::new(hx_engine::rate_limit::RateLimiter::new()),
            scheduling: Some(hx_engine::scheduling::SchedulingService::new()),
        }
    }

    pub fn notify_change(&self, node_id: &str, operation: &str, namespace: Option<&str>) {
        let _ = self.change_tx.send(ChangeNotification {
            node_id: node_id.to_string(),
            operation: operation.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            namespace: namespace.map(|ns| ns.to_string()),
        });
    }

    /// Send a reminder notification.
    pub fn notify_reminder(&self, notification: ReminderNotification) {
        let _ = self.reminder_tx.send(notification.clone());

        // Fire-and-forget webhook dispatch
        if let Some(ref url) = self.webhook_config.reminder_url {
            let url = url.clone();
            let timeout = self.webhook_config.timeout_secs;
            tokio::spawn(async move {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(timeout))
                    .build()
                    .ok();
                if let Some(client) = client {
                    let _ = client
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .json(&notification)
                        .send()
                        .await;
                }
            });
        }
    }

    pub fn notify_agent(&self, notification: AgentNotification) {
        let _ = self.agent_tx.send(notification);
    }

    /// Fire-and-forget webhook dispatch for keychain breach alerts.
    pub fn notify_keychain_alert(&self, alert: &hx_core::model::keychain::BreachAlert) {
        if let Some(ref url) = self.webhook_config.keychain_alert_url {
            let url = url.clone();
            let timeout = self.webhook_config.timeout_secs;
            let payload = serde_json::json!(alert);
            tokio::spawn(async move {
                let client = reqwest::Client::builder()
                    .timeout(std::time::Duration::from_secs(timeout))
                    .build()
                    .ok();
                if let Some(client) = client {
                    let _ = client
                        .post(&url)
                        .header("Content-Type", "application/json")
                        .json(&payload)
                        .send()
                        .await;
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::{InsightType, IntentType};

    #[test]
    fn agent_notification_serializes_with_expected_type_tag() {
        let mut intent = CapturedIntent::new(uuid::Uuid::now_v7(), IntentType::ExtractTask);
        intent.confidence = 0.8;
        let notification = AgentNotification::Intent {
            intent,
            namespace: Some("default".to_string()),
        };

        let json = serde_json::to_value(notification).unwrap();
        assert_eq!(json["type"], "intent");
        assert_eq!(json["namespace"], "default");

        let insight = ProactiveInsight::new("t", "c", InsightType::Trend);
        let insight_json = serde_json::to_value(AgentNotification::InsightDiscovered {
            insight,
            namespace: Some("default".to_string()),
        })
        .unwrap();
        assert_eq!(insight_json["type"], "insight_discovered");
    }

    #[test]
    fn agent_notification_namespace_accessor() {
        let entry = ChronicleEntry::new("step", "logic");
        let notification = AgentNotification::Chronicle {
            entry,
            namespace: Some("ops".to_string()),
        };
        assert_eq!(notification.namespace(), Some("ops"));
    }
}
