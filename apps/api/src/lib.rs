pub mod adapter_poll;
pub mod audit;
pub mod auth;
pub mod email;
pub mod grpc;
pub mod limits;
pub mod metrics;
pub mod openapi;
pub mod pairing;
pub mod rest;
pub mod state;
pub mod validation;
pub mod websocket;

use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use hx_core::ChronicleEntry;
use hx_engine::config::EngineConfig;
use hx_engine::engine::HelixEngine;
use hx_engine::intent::IntentEngine;
use hx_engine::watcher::WatcherAgent;
use state::AppState;
use uuid::Uuid;

pub struct ServerConfig {
    pub bind_host: String,
    pub rest_port: u16,
    pub grpc_port: u16,
    pub socket_path: Option<String>,
    pub cors_allowed_origins: Vec<String>,
    pub engine_config: EngineConfig,
}

const SEALED_BLOB_MAGIC: &[u8; 4] = b"HXB1";

fn require_hardware_from_env() -> bool {
    std::env::var("HELIX_REQUIRE_HARDWARE")
        .map(|value| value.eq_ignore_ascii_case("true") || value == "1")
        .unwrap_or(false)
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SealedStorageScanReport {
    pub findings: Vec<String>,
}

impl SealedStorageScanReport {
    pub fn is_clean(&self) -> bool {
        self.findings.is_empty()
    }

    pub fn summary(&self) -> String {
        self.findings.join("; ")
    }
}

pub fn scan_sealed_storage(
    config: &EngineConfig,
) -> Result<SealedStorageScanReport, std::io::Error> {
    let data_dir = PathBuf::from(&config.data_dir);
    if !data_dir.exists() {
        return Ok(SealedStorageScanReport::default());
    }

    let mut report = SealedStorageScanReport::default();
    for legacy_index_dir in ["tantivy", "lancedb"] {
        let path = data_dir.join(legacy_index_dir);
        if path.exists() {
            report.findings.push(format!(
                "legacy index directory present: {}",
                path.display()
            ));
        }
    }

    let blobs_root = data_dir.join("blobs");
    if blobs_root.exists() {
        if let Some(path) = find_first_plaintext_blob(&blobs_root)? {
            report.findings.push(format!(
                "plaintext blob payload detected: {}",
                path.display()
            ));
        }
    }

    Ok(report)
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_host: "127.0.0.1".into(),
            rest_port: 9470,
            grpc_port: 50051,
            socket_path: Some(shellexpand("~/.helix/helix.sock")),
            cors_allowed_origins: Vec::new(),
            engine_config: EngineConfig::default(),
        }
    }
}

/// Start the Helix server with all transports.
pub async fn start_server(
    config: ServerConfig,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,hx_server=debug,hx_engine=debug".parse().unwrap()),
        )
        .init();
    rest::init_observability();

    if let Err(err) = check_bind_safety(&config.bind_host) {
        return Err(err.into());
    }
    startup_sealed_storage_preflight(&config.engine_config)?;

    tracing::info!("initializing Helix engine...");
    let mut engine = HelixEngine::init(config.engine_config).await?;
    if require_hardware_from_env() && !engine.keychain.os_secure_storage_available() {
        return Err(std::io::Error::other(
            "HELIX_REQUIRE_HARDWARE=true but OS secure storage is unavailable",
        )
        .into());
    }
    ensure_startup_unsealed(&engine)?;

    // Create broadcast channel early so enrichment can use it
    let (change_tx, _) = tokio::sync::broadcast::channel::<state::ChangeNotification>(256);

    // Set up enrichment pipeline (before wrapping in Arc)
    let enrichment_worker = engine.setup_enrichment(change_tx.clone());

    let engine = Arc::new(engine);
    engine.proactive.set_engine(Arc::clone(&engine));

    ensure_today_daily_note_on_startup_best_effort(&engine).await;
    spawn_daily_note_scheduler(Arc::clone(&engine));

    // Spawn enrichment worker if enabled
    if let Some(worker) = enrichment_worker {
        tokio::spawn(worker.run());
        tracing::info!("enrichment worker spawned");
    }

    // Shutdown broadcast for background agents
    let (shutdown_tx, _) = tokio::sync::broadcast::channel::<()>(1);

    // Create agent notification channel early so watcher can use it
    let (agent_tx, _) = tokio::sync::broadcast::channel::<state::AgentNotification>(256);

    // Spawn watcher agent with notification forwarding
    spawn_watcher_agent(
        Arc::clone(&engine),
        agent_tx.clone(),
        shutdown_tx.subscribe(),
    );

    let state = Arc::new(AppState::new_with_channels(engine, change_tx, agent_tx));
    spawn_agent_change_processor(Arc::clone(&state), shutdown_tx.subscribe());
    spawn_recurrence_and_reminder_scheduler(Arc::clone(&state));
    spawn_google_calendar_sync(Arc::clone(&state.engine), shutdown_tx.subscribe());
    adapter_poll::spawn_adapter_polling(Arc::clone(&state), shutdown_tx.subscribe());
    email::spawn_email_adapter(Arc::clone(&state), shutdown_tx.subscribe());
    spawn_job_worker(&state, shutdown_tx.subscribe());
    spawn_workflow_scheduler(Arc::clone(&state.workflow_executor), shutdown_tx.subscribe());

    // Background task: expire stale proxy approvals every 60 seconds
    {
        let engine = Arc::clone(&state.engine);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
            interval.tick().await; // first tick is immediate, skip it
            loop {
                interval.tick().await;
                match engine.expire_approvals().await {
                    Ok(0) => {}
                    Ok(n) => tracing::debug!("expired {n} stale proxy approvals"),
                    Err(e) => tracing::warn!("failed to expire proxy approvals: {e}"),
                }
            }
        });
    }

    // Keychain lifecycle scheduler (credential expiry + auto-rotation)
    state
        .engine
        .keychain
        .start_lifecycle_scheduler(&state.engine.config.keychain)
        .await;

    // REST + WebSocket server
    let rest_state = Arc::clone(&state);
    let ws_state = Arc::clone(&state);
    let bind_host = config.bind_host.clone();
    let cors_allowed_origins = config.cors_allowed_origins.clone();
    let rest_port = config.rest_port;
    let rest_handle = tokio::spawn(async move {
        let app = rest::create_router_with_cors(rest_state, &cors_allowed_origins)
            .merge(websocket::ws_router(ws_state));
        tracing::info!("REST API listening on {bind_host}:{rest_port}");
        let listener = tokio::net::TcpListener::bind(format!("{bind_host}:{rest_port}"))
            .await
            .expect("failed to bind REST port");
        axum::serve(listener, app).await.ok();
    });

    // gRPC server
    let grpc_state = Arc::clone(&state);
    let grpc_bind_host = config.bind_host.clone();
    let grpc_port = config.grpc_port;
    let grpc_handle = tokio::spawn(async move {
        tracing::info!("gRPC API listening on {grpc_bind_host}:{grpc_port}");
        let addr = format!("{grpc_bind_host}:{grpc_port}").parse().unwrap();
        let service = grpc::HelixGrpc::new(Arc::clone(&grpc_state));
        let keychain_service = grpc::KeychainGrpc::new(grpc_state);
        tonic::transport::Server::builder()
            .add_service(
                grpc::proto::helix_service_server::HelixServiceServer::new(service),
            )
            .add_service(
                grpc::proto::keychain_service_server::KeychainServiceServer::new(keychain_service),
            )
            .serve(addr)
            .await
            .ok();
    });

    // Unix Domain Socket (REST API over UDS)
    if let Some(ref sock_path) = config.socket_path {
        let uds_state = Arc::clone(&state);
        let sock = sock_path.clone();
        tokio::spawn(async move {
            let _ = std::fs::remove_file(&sock);
            if let Some(parent) = std::path::Path::new(&sock).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            tracing::info!("UDS listening on {sock}");
            let uds_listener = match tokio::net::UnixListener::bind(&sock) {
                Ok(l) => l,
                Err(e) => {
                    tracing::error!("failed to bind UDS at {sock}: {e}");
                    return;
                }
            };

            let app = rest::create_router(uds_state);
            loop {
                match uds_listener.accept().await {
                    Ok((stream, _addr)) => {
                        let app = app.clone();
                        tokio::spawn(async move {
                            let io = hyper_util::rt::TokioIo::new(stream);
                            let service = hyper::service::service_fn(move |req| {
                                let app = app.clone();
                                async move {
                                    let resp = tower::ServiceExt::oneshot(app, req).await;
                                    resp
                                }
                            });
                            if let Err(e) = hyper_util::server::conn::auto::Builder::new(
                                hyper_util::rt::TokioExecutor::new(),
                            )
                            .serve_connection(io, service)
                            .await
                            {
                                tracing::error!("UDS connection error: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        tracing::error!("UDS accept error: {e}");
                    }
                }
            }
        });
    }

    tracing::info!("Helix server started");

    tokio::select! {
        _ = rest_handle => {},
        _ = grpc_handle => {},
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("shutting down...");
            let _ = shutdown_tx.send(());
        }
    }

    Ok(())
}

fn ensure_startup_unsealed(engine: &HelixEngine) -> Result<(), std::io::Error> {
    if engine.config.sealed_mode && !engine.keychain.is_unsealed_sync() {
        return Err(std::io::Error::other("Vault sealed - please unseal"));
    }
    Ok(())
}

fn startup_sealed_storage_preflight(config: &EngineConfig) -> Result<(), std::io::Error> {
    if !config.sealed_mode {
        return Ok(());
    }

    let report = scan_sealed_storage(config)?;
    if report.is_clean() {
        return Ok(());
    }
    let reason = report.summary();
    Err(std::io::Error::other(format!(
        "sealed mode startup preflight failed: {reason}; run `mv server migrate-sealed --passphrase <pw>` to encrypt legacy artifacts"
    )))
}

fn find_first_plaintext_blob(root: &Path) -> Result<Option<PathBuf>, std::io::Error> {
    if !root.exists() {
        return Ok(None);
    }
    if root.is_file() {
        let bytes = std::fs::read(root)?;
        if bytes.starts_with(SEALED_BLOB_MAGIC) {
            return Ok(None);
        }
        return Ok(Some(root.to_path_buf()));
    }

    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&path)?;
            if bytes.starts_with(SEALED_BLOB_MAGIC) {
                continue;
            }
            return Ok(Some(path));
        }
    }

    Ok(None)
}

fn spawn_watcher_agent(
    engine: Arc<HelixEngine>,
    agent_tx: tokio::sync::broadcast::Sender<state::AgentNotification>,
    shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let watcher_config = engine.config.watcher.clone();
    if !watcher_config.enabled {
        tracing::info!("watcher agent disabled by config");
        return;
    }

    let intent_engine = IntentEngine::new(Arc::clone(&engine.store)).with_llm(engine.llm.clone());
    let proactive_engine = Arc::clone(&engine.proactive);

    // Build notifier that forwards discoveries to WebSocket via agent_tx
    let notifier: hx_engine::watcher::WatcherNotifier = Arc::new(move |report| {
        for (intent, namespace) in &report.new_intents {
            let _ = agent_tx.send(state::AgentNotification::Intent {
                intent: intent.clone(),
                namespace: Some(namespace.clone()),
            });
        }
        for (insight, namespace) in &report.new_insights {
            let _ = agent_tx.send(state::AgentNotification::InsightDiscovered {
                insight: insight.clone(),
                namespace: Some(namespace.clone()),
            });
        }
    });

    let agent = Arc::new(
        WatcherAgent::new(
            Arc::clone(&engine),
            intent_engine,
            proactive_engine,
            watcher_config,
        )
        .with_notifier(notifier),
    );

    tokio::spawn(async move {
        agent.run_loop(shutdown_rx).await;
    });

    tracing::info!("watcher agent spawned");
}

fn spawn_agent_change_processor(
    state: Arc<AppState>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    if !state.engine.config.watcher.enabled {
        tracing::info!("agent change processor disabled by config");
        return;
    }

    let intent_engine =
        IntentEngine::new(Arc::clone(&state.engine.store)).with_llm(state.engine.llm.clone());
    let mut change_rx = state.change_tx.subscribe();

    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("agent change processor shutting down");
                    break;
                }
                event = change_rx.recv() => {
                    match event {
                        Ok(notification) => {
                            process_agent_change_notification(&state, &intent_engine, notification).await;
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!("agent change processor lagged by {n} events");
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
            }
        }
    });

    tracing::info!("agent change processor spawned");
}

async fn process_agent_change_notification(
    state: &Arc<AppState>,
    intent_engine: &IntentEngine,
    notification: state::ChangeNotification,
) {
    if state.engine.config.sealed_mode && !state.engine.keychain.is_unsealed_sync() {
        return;
    }

    match notification.operation.as_str() {
        "create" | "update" | "enriched" => {}
        _ => return,
    }

    let node_id = match Uuid::parse_str(&notification.node_id) {
        Ok(id) => id,
        Err(err) => {
            tracing::warn!(
                node_id = notification.node_id,
                error = %err,
                "agent change processor received invalid node id"
            );
            return;
        }
    };

    let node = match state.engine.get_node(node_id).await {
        Ok(Some(node)) => node,
        Ok(None) => return,
        Err(err) => {
            tracing::warn!(node_id = %node_id, error = %err, "agent change processor failed to load node");
            return;
        }
    };

    let namespace = Some(node.namespace.clone());

    if notification.operation == "enriched" {
        state.notify_agent(state::AgentNotification::NodeEnriched {
            node_id: node.id.to_string(),
            namespace,
        });
        return;
    }

    let observed_entry = ChronicleEntry::new(
        "agent_change_observed",
        format!(
            "Observed {} operation on node {}",
            notification.operation, notification.node_id
        ),
    )
    .with_node(node.id);

    if let Err(err) = state.engine.log_chronicle(&observed_entry).await {
        tracing::warn!(node_id = %node.id, error = %err, "failed to log agent change chronicle");
    } else {
        state.notify_agent(state::AgentNotification::Chronicle {
            entry: observed_entry,
            namespace: Some(node.namespace.clone()),
        });
    }

    match intent_engine.extract_intents_and_store(&node).await {
        Ok(intents) => {
            for intent in intents {
                state.notify_agent(state::AgentNotification::Intent {
                    intent: intent.clone(),
                    namespace: Some(node.namespace.clone()),
                });

                let entry = ChronicleEntry::new(
                    "intent_detected",
                    format!(
                        "Detected {} intent with {:.0}% confidence",
                        intent.intent_type,
                        intent.confidence * 100.0
                    ),
                )
                .with_node(node.id);
                if let Err(err) = state.engine.log_chronicle(&entry).await {
                    tracing::warn!(node_id = %node.id, error = %err, "failed to log intent chronicle");
                } else {
                    state.notify_agent(state::AgentNotification::Chronicle {
                        entry,
                        namespace: Some(node.namespace.clone()),
                    });
                }
            }
        }
        Err(err) => {
            tracing::warn!(node_id = %node.id, error = %err, "intent extraction failed");
        }
    }

    match state
        .engine
        .proactive
        .find_related_context(node.id, 5)
        .await
    {
        Ok(related) => {
            let nodes: Vec<state::AgentRelatedNode> = related
                .into_iter()
                .map(|related_node| state::AgentRelatedNode {
                    id: related_node.id.to_string(),
                    title: related_node.title.unwrap_or_else(|| "Untitled".to_string()),
                    updated_at: related_node.temporal.updated_at.to_rfc3339(),
                })
                .collect();

            if !nodes.is_empty() {
                state.notify_agent(state::AgentNotification::RelatedContext {
                    nodes,
                    namespace: Some(node.namespace.clone()),
                });
            }
        }
        Err(err) => {
            tracing::warn!(node_id = %node.id, error = %err, "related context discovery failed");
        }
    }

    match state
        .engine
        .proactive
        .generate_insights(node.namespace.clone())
        .await
    {
        Ok(insights) => {
            for insight in insights {
                state.notify_agent(state::AgentNotification::InsightDiscovered {
                    insight: insight.clone(),
                    namespace: Some(node.namespace.clone()),
                });

                let entry = ChronicleEntry::new(
                    "insight_generated",
                    format!("{}: {}", insight.insight_type, insight.title),
                )
                .with_node(node.id);
                if let Err(err) = state.engine.log_chronicle(&entry).await {
                    tracing::warn!(node_id = %node.id, error = %err, "failed to log insight chronicle");
                } else {
                    state.notify_agent(state::AgentNotification::Chronicle {
                        entry,
                        namespace: Some(node.namespace.clone()),
                    });
                }
            }
        }
        Err(err) => {
            tracing::warn!(node_id = %node.id, error = %err, "proactive insight generation failed");
        }
    }
}

async fn ensure_today_daily_note_on_startup_best_effort(engine: &Arc<HelixEngine>) {
    if !engine.config.daily_notes.enabled {
        return;
    }
    if engine.config.sealed_mode && !engine.keychain.is_unsealed_sync() {
        return;
    }

    let today = Utc::now().date_naive();
    match engine.ensure_daily_note(today, None).await {
        Ok((_node, created)) => {
            tracing::info!(
                date = %today,
                created,
                namespace = %engine.config.daily_notes.namespace,
                "helix_daily_note_startup_ensure_complete"
            );
        }
        Err(err) => {
            tracing::warn!(
                date = %today,
                namespace = %engine.config.daily_notes.namespace,
                error = %err,
                "helix_daily_note_startup_ensure_failed"
            );
        }
    }
}

fn spawn_daily_note_scheduler(engine: Arc<HelixEngine>) {
    if !daily_note_scheduler_enabled(&engine.config) {
        if engine.config.daily_notes.enabled
            && !engine.config.daily_notes.midnight_scheduler_enabled
        {
            tracing::info!(
                namespace = %engine.config.daily_notes.namespace,
                "helix_daily_note_scheduler_disabled_by_config"
            );
        }
        return;
    }

    tokio::spawn(async move {
        loop {
            let sleep_duration = duration_until_next_utc_midnight(Utc::now());
            tracing::info!(
                sleep_seconds = sleep_duration.as_secs(),
                namespace = %engine.config.daily_notes.namespace,
                "helix_daily_note_scheduler_sleep_until_next_utc_midnight"
            );
            tokio::time::sleep(sleep_duration).await;

            if engine.config.sealed_mode && !engine.keychain.is_unsealed_sync() {
                continue;
            }

            let today = Utc::now().date_naive();
            match engine.ensure_daily_note(today, None).await {
                Ok((_node, created)) => {
                    tracing::info!(
                        date = %today,
                        created,
                        namespace = %engine.config.daily_notes.namespace,
                        "helix_daily_note_scheduler_ensure_complete"
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        date = %today,
                        namespace = %engine.config.daily_notes.namespace,
                        error = %err,
                        "helix_daily_note_scheduler_ensure_failed"
                    );
                }
            }
        }
    });
}

fn daily_note_scheduler_enabled(config: &EngineConfig) -> bool {
    config.daily_notes.enabled && config.daily_notes.midnight_scheduler_enabled
}

fn spawn_recurrence_and_reminder_scheduler(state: Arc<AppState>) {
    if !state.engine.config.recurrence.enabled {
        return;
    }

    let interval_secs = state
        .engine
        .config
        .recurrence
        .scheduler_interval_secs
        .max(30);
    let max_instances_per_template = state
        .engine
        .config
        .recurrence
        .max_instances_per_template
        .max(1);
    tokio::spawn(async move {
        loop {
            if state.engine.config.sealed_mode && !state.engine.keychain.is_unsealed_sync() {
                tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
                continue;
            }

            let now = Utc::now();
            match state
                .engine
                .rollforward_recurring_tasks(now, max_instances_per_template)
                .await
            {
                Ok(stats) => {
                    tracing::info!(
                        scanned_tasks = stats.scanned_tasks,
                        recurring_templates = stats.recurring_templates,
                        generated_instances = stats.generated_instances,
                        updated_templates = stats.updated_templates,
                        errors = stats.errors,
                        interval_secs,
                        "helix_recurrence_rollforward_cycle_complete"
                    );
                }
                Err(err) => {
                    tracing::warn!(
                        error = %err,
                        interval_secs,
                        "helix_recurrence_rollforward_cycle_failed"
                    );
                }
            }

            // Dispatch task reminders with WebSocket/webhook notifications
            dispatch_task_reminders_with_notifications(&state, now, interval_secs).await;

            tokio::time::sleep(std::time::Duration::from_secs(interval_secs)).await;
        }
    });
}

fn spawn_google_calendar_sync(
    engine: Arc<HelixEngine>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let config = engine.config.google_calendar.clone();
    if !config.enabled {
        return;
    }

    let interval_secs = config.sync_interval_secs.max(60);
    let calendar_id = config.calendar_id.clone();
    let calendar_id_for_task = calendar_id.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(interval_secs));
        interval.tick().await;
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("google calendar sync shutting down");
                    break;
                }
                _ = interval.tick() => {
                    if engine.config.sealed_mode && !engine.keychain.is_unsealed_sync() {
                        continue;
                    }
                    match engine.sync_google_calendar().await {
                        Ok(report) => {
                            tracing::info!(
                                calendar_id = %report.calendar_id,
                                fetched = report.fetched,
                                created = report.created,
                                updated = report.updated,
                                deleted = report.deleted,
                                skipped = report.skipped,
                                exported_created = report.exported_created,
                                exported_updated = report.exported_updated,
                                "google calendar sync completed"
                            );
                        }
                        Err(err) => {
                            tracing::warn!(
                                calendar_id = %calendar_id_for_task,
                                error = %err,
                                "google calendar sync failed"
                            );
                        }
                    }
                }
            }
        }
    });

    tracing::info!(interval_secs, calendar_id = %calendar_id, "google calendar sync spawned");
}

async fn dispatch_task_reminders_with_notifications(
    state: &Arc<AppState>,
    now: DateTime<Utc>,
    interval_secs: u64,
) {
    use hx_engine::recurrence::{
        parse_optional_metadata_datetime, TASK_DUE_AT_METADATA_KEY,
        TASK_REMINDER_SENT_AT_METADATA_KEY,
    };

    // First, get the list of due tasks before marking them
    let due_tasks = match state.engine.list_due_tasks(now, None, 500, false).await {
        Ok(tasks) => tasks,
        Err(err) => {
            tracing::warn!(
                error = %err,
                interval_secs,
                "helix_task_reminder_list_due_failed"
            );
            return;
        }
    };

    // Filter to those not yet notified
    let mut tasks_to_notify = Vec::new();
    for task in due_tasks {
        let already_sent =
            parse_optional_metadata_datetime(&task.metadata, TASK_REMINDER_SENT_AT_METADATA_KEY)
                .map(|v| v.is_some())
                .unwrap_or(false);

        if !already_sent {
            tasks_to_notify.push(task);
        }
    }

    // Now dispatch reminders (marks them as sent)
    match state.engine.dispatch_due_task_reminders(now, 500).await {
        Ok(stats) => {
            tracing::info!(
                scanned_tasks = stats.scanned_tasks,
                due_tasks = stats.due_tasks,
                reminders_marked_sent = stats.reminders_marked_sent,
                errors = stats.errors,
                interval_secs,
                "helix_task_reminder_dispatch_cycle_complete"
            );

            // Send WebSocket/webhook notifications for tasks we just marked
            for task in tasks_to_notify {
                let due_at =
                    parse_optional_metadata_datetime(&task.metadata, TASK_DUE_AT_METADATA_KEY)
                        .ok()
                        .flatten();

                let content_preview = task.content.chars().take(200).collect::<String>();

                let notification = state::ReminderNotification {
                    node_id: task.id.to_string(),
                    title: task.title.clone(),
                    content_preview,
                    due_at,
                    namespace: Some(task.namespace.clone()),
                    timestamp: now,
                    notification_type: "task_due".to_string(),
                };

                state.notify_reminder(notification);
            }
        }
        Err(err) => {
            tracing::warn!(
                error = %err,
                interval_secs,
                "helix_task_reminder_dispatch_cycle_failed"
            );
        }
    }
}

fn duration_until_next_utc_midnight(now: DateTime<Utc>) -> std::time::Duration {
    let tomorrow = now.date_naive().succ_opt().unwrap_or(now.date_naive());
    let next_midnight_naive = tomorrow
        .and_hms_opt(0, 0, 0)
        .unwrap_or_else(|| now.naive_utc());
    let next_midnight = DateTime::<Utc>::from_naive_utc_and_offset(next_midnight_naive, Utc);
    let seconds = (next_midnight - now).num_seconds().max(1) as u64;
    std::time::Duration::from_secs(seconds)
}

fn auth_enabled_from_env() -> bool {
    auth_enabled_from_env_values(
        std::env::var("HELIX_JWT_SECRET").ok().as_deref(),
        std::env::var("HELIX_AUTH_TOKEN").ok().as_deref(),
    )
}

fn allow_insecure_bind_from_env() -> bool {
    allow_insecure_bind_value(
        std::env::var("HELIX_ALLOW_INSECURE_BIND")
            .ok()
            .as_deref(),
    )
}

pub fn check_bind_safety(bind_host: &str) -> Result<(), String> {
    let auth_enabled = auth_enabled_from_env();
    let allow_insecure_bind = allow_insecure_bind_from_env();
    validate_bind_safety(bind_host, auth_enabled, allow_insecure_bind)
}

fn auth_enabled_from_env_values(jwt_secret: Option<&str>, auth_token: Option<&str>) -> bool {
    jwt_secret
        .map(|value| !value.trim().is_empty())
        .unwrap_or(false)
        || auth_token
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false)
}

fn allow_insecure_bind_value(raw: Option<&str>) -> bool {
    let value = raw.unwrap_or("").trim().to_ascii_lowercase();
    matches!(value.as_str(), "1" | "true" | "yes" | "on")
}

fn is_local_bind_host(bind_host: &str) -> bool {
    let host = bind_host.trim();
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host.parse::<IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => false,
    }
}

fn validate_bind_safety(
    bind_host: &str,
    auth_enabled: bool,
    allow_insecure_bind: bool,
) -> Result<(), String> {
    if is_local_bind_host(bind_host) || auth_enabled || allow_insecure_bind {
        return Ok(());
    }

    Err(format!(
        "refusing to bind to '{bind_host}' without auth; set HELIX_AUTH_TOKEN or HELIX_JWT_SECRET, or override with HELIX_ALLOW_INSECURE_BIND=true (run 'mv server preflight' to verify your config)"
    ))
}

fn spawn_job_worker(
    state: &Arc<state::AppState>,
    shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let Some(ref queue) = state.job_queue else {
        tracing::info!("job queue not initialized; skipping job worker");
        return;
    };

    let worker = hx_engine::jobs::worker::JobWorker::new(Arc::clone(queue))
        .with_handler(Box::new(
            hx_engine::jobs::handlers::WebhookDeliveryHandler::new(reqwest::Client::new()),
        ))
        .with_handler(Box::new(hx_engine::jobs::handlers::SourcePollHandler::new(
            Arc::clone(&state.source_registry),
        )))
        .with_handler(Box::new(
            hx_engine::jobs::handlers::WorkflowStepHandler::new(Arc::clone(
                &state.workflow_executor,
            )),
        ))
        .with_poll_interval(std::time::Duration::from_secs(2))
        .with_concurrency(4);

    tokio::spawn(async move {
        worker.run(shutdown_rx).await;
    });

    tracing::info!("job worker spawned (concurrency=4, poll=2s)");
}

fn spawn_workflow_scheduler(
    executor: Arc<hx_engine::workflow::executor::WorkflowExecutor>,
    shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    let scheduler = hx_engine::workflow::scheduler::WorkflowScheduler::new(executor);
    tokio::spawn(async move {
        scheduler.run(shutdown_rx).await;
    });
    tracing::info!("workflow scheduler spawned");
}

fn shellexpand(s: &str) -> String {
    if let Some(rest) = s.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use hx_engine::engine::HelixEngine;
    use hx_storage::sealed_runtime::{clear_runtime_root_key, set_sealed_mode_enabled};
    use serial_test::serial;
    use tempfile::TempDir;

    struct SealedModeCleanup;

    impl Drop for SealedModeCleanup {
        fn drop(&mut self) {
            set_sealed_mode_enabled(false);
            clear_runtime_root_key();
        }
    }

    #[test]
    fn daily_note_scheduler_respects_enabled_flags() {
        let mut config = EngineConfig::default();
        config.daily_notes.enabled = true;
        config.daily_notes.midnight_scheduler_enabled = true;
        assert!(daily_note_scheduler_enabled(&config));

        config.daily_notes.midnight_scheduler_enabled = false;
        assert!(!daily_note_scheduler_enabled(&config));

        config.daily_notes.enabled = false;
        config.daily_notes.midnight_scheduler_enabled = true;
        assert!(!daily_note_scheduler_enabled(&config));
    }

    #[test]
    fn duration_until_next_midnight_from_midday_is_half_day() {
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 12, 0, 0)
            .single()
            .expect("valid datetime");
        let duration = duration_until_next_utc_midnight(now);
        assert_eq!(duration.as_secs(), 12 * 60 * 60);
    }

    #[test]
    fn duration_until_next_midnight_from_exact_midnight_is_full_day() {
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 0, 0, 0)
            .single()
            .expect("valid datetime");
        let duration = duration_until_next_utc_midnight(now);
        assert_eq!(duration.as_secs(), 24 * 60 * 60);
    }

    #[test]
    fn duration_until_next_midnight_never_zero() {
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 23, 59, 59)
            .single()
            .expect("valid datetime");
        let duration = duration_until_next_utc_midnight(now);
        assert!(duration.as_secs() >= 1);
    }

    #[test]
    fn bind_safety_detects_local_hosts() {
        assert!(is_local_bind_host("127.0.0.1"));
        assert!(is_local_bind_host("127.12.34.56"));
        assert!(is_local_bind_host("::1"));
        assert!(is_local_bind_host("localhost"));
        assert!(!is_local_bind_host("0.0.0.0"));
        assert!(!is_local_bind_host("::"));
        assert!(!is_local_bind_host("192.168.1.10"));
    }

    #[test]
    fn bind_safety_rejects_public_without_auth_or_override() {
        let err = validate_bind_safety("0.0.0.0", false, false).unwrap_err();
        assert!(err.contains("auth"));
    }

    #[test]
    fn bind_safety_error_mentions_preflight_command() {
        let err = validate_bind_safety("0.0.0.0", false, false).unwrap_err();
        assert!(err.contains("mv server preflight"));
    }

    #[test]
    fn bind_safety_allows_public_with_auth() {
        assert!(validate_bind_safety("0.0.0.0", true, false).is_ok());
    }

    #[test]
    fn bind_safety_allows_public_with_override() {
        assert!(validate_bind_safety("0.0.0.0", false, true).is_ok());
    }

    #[test]
    fn auth_enabled_from_env_values_detects_tokens() {
        assert!(!auth_enabled_from_env_values(None, None));
        assert!(auth_enabled_from_env_values(Some("jwt"), None));
        assert!(auth_enabled_from_env_values(None, Some("token")));
        assert!(!auth_enabled_from_env_values(Some(" "), Some("")));
    }

    #[test]
    fn allow_insecure_bind_value_parses_truthy() {
        assert!(allow_insecure_bind_value(Some("true")));
        assert!(allow_insecure_bind_value(Some("1")));
        assert!(allow_insecure_bind_value(Some("yes")));
        assert!(allow_insecure_bind_value(Some("on")));
        assert!(!allow_insecure_bind_value(Some("false")));
        assert!(!allow_insecure_bind_value(Some("0")));
        assert!(!allow_insecure_bind_value(None));
    }

    #[tokio::test]
    #[serial(sealed_mode)]
    async fn startup_rejects_sealed_vault_state() {
        let _cleanup = SealedModeCleanup;
        let temp_dir = TempDir::new().expect("temp dir");
        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.embedding.provider = "noop".into();
        config.sealed_mode = true;

        let engine = HelixEngine::init(config).await.expect("engine init");
        engine
            .keychain
            .initialize_vault("test-password", false, "test")
            .await
            .expect("vault initialized");
        engine.keychain.seal("test").await.expect("vault sealed");

        let err = ensure_startup_unsealed(&engine).expect_err("sealed startup must fail");
        assert!(err.to_string().contains("Vault sealed - please unseal"));
    }

    #[tokio::test]
    #[serial(sealed_mode)]
    async fn startup_allows_unsealed_vault_state() {
        let _cleanup = SealedModeCleanup;
        let temp_dir = TempDir::new().expect("temp dir");
        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.embedding.provider = "noop".into();
        config.sealed_mode = true;

        let engine = HelixEngine::init(config).await.expect("engine init");
        engine
            .keychain
            .initialize_vault("test-password", false, "test")
            .await
            .expect("vault initialized");
        engine
            .keychain
            .unseal("test-password", "test")
            .await
            .expect("vault unsealed");

        ensure_startup_unsealed(&engine).expect("unsealed startup should pass");
    }

    #[test]
    fn startup_preflight_rejects_legacy_index_directories() {
        let temp_dir = TempDir::new().expect("temp dir");
        let data_dir = temp_dir.path();
        std::fs::create_dir_all(data_dir.join("tantivy")).expect("create tantivy");

        let mut config = EngineConfig::default();
        config.data_dir = data_dir.to_string_lossy().to_string();
        config.sealed_mode = true;

        let err =
            startup_sealed_storage_preflight(&config).expect_err("legacy plaintext dirs rejected");
        assert!(err.to_string().contains("legacy index directory present"));
    }

    #[test]
    fn startup_preflight_rejects_plaintext_blob_payloads() {
        let temp_dir = TempDir::new().expect("temp dir");
        let blob_file = temp_dir
            .path()
            .join("blobs")
            .join("default")
            .join("node-a")
            .join("attachment.bin");
        std::fs::create_dir_all(blob_file.parent().expect("blob parent")).expect("create dirs");
        std::fs::write(&blob_file, b"plaintext payload").expect("write plaintext blob");

        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.sealed_mode = true;

        let err =
            startup_sealed_storage_preflight(&config).expect_err("plaintext blob must fail scan");
        assert!(err.to_string().contains("plaintext blob payload detected"));
    }

    #[test]
    fn startup_preflight_accepts_encrypted_blob_payloads() {
        let temp_dir = TempDir::new().expect("temp dir");
        let blob_file = temp_dir
            .path()
            .join("blobs")
            .join("default")
            .join("node-a")
            .join("attachment.bin");
        std::fs::create_dir_all(blob_file.parent().expect("blob parent")).expect("create dirs");
        std::fs::write(
            &blob_file,
            [SEALED_BLOB_MAGIC.as_slice(), b"ciphertext"].concat(),
        )
        .expect("write encrypted blob");

        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.sealed_mode = true;

        startup_sealed_storage_preflight(&config).expect("encrypted blob should pass scan");
    }

    #[test]
    fn sealed_storage_scan_reports_findings_with_sealed_mode_disabled() {
        let temp_dir = TempDir::new().expect("temp dir");
        std::fs::create_dir_all(temp_dir.path().join("tantivy")).expect("create tantivy");

        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.sealed_mode = false;

        let report = scan_sealed_storage(&config).expect("scan");
        assert!(
            report
                .findings
                .iter()
                .any(|finding| finding.contains("legacy index directory present")),
            "expected legacy directory finding, got {:?}",
            report.findings
        );
    }
}
