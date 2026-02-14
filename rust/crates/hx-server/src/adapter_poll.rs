use std::sync::Arc;

use hx_core::{
    AdapterPollStore, ContentType, MvResult, RelayChannel, RelayContact, RelayMessage, TrustLevel,
};
use hx_engine::adapters::{AdapterConfig, AdapterInboundMessage};
use tokio::sync::broadcast;
use uuid::Uuid;

use crate::state::AppState;

#[derive(Debug, Clone)]
pub struct AdapterPollConfig {
    pub enabled: bool,
    pub interval_secs: u64,
}

impl AdapterPollConfig {
    pub fn from_env() -> Self {
        let enabled = std::env::var("HELIX_ADAPTER_POLL_ENABLED")
            .ok()
            .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(true);
        let interval_secs = std::env::var("HELIX_ADAPTER_POLL_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(30);
        Self {
            enabled,
            interval_secs: interval_secs.max(5),
        }
    }
}

pub fn spawn_adapter_polling(state: Arc<AppState>, mut shutdown_rx: broadcast::Receiver<()>) {
    let config = AdapterPollConfig::from_env();
    if !config.enabled {
        tracing::info!("adapter polling disabled");
        return;
    }

    tokio::spawn(async move {
        let mut interval =
            tokio::time::interval(std::time::Duration::from_secs(config.interval_secs));
        interval.tick().await;
        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("adapter poller shutting down");
                    break;
                }
                _ = interval.tick() => {
                    if let Err(err) = poll_once(&state).await {
                        tracing::warn!(error = %err, "adapter poll cycle failed");
                    }
                }
            }
        }
    });

    tracing::info!(
        interval_secs = config.interval_secs,
        "adapter poller spawned"
    );
}

async fn poll_once(state: &Arc<AppState>) -> MvResult<()> {
    if state.engine.config.sealed_mode && !state.engine.keychain.is_unsealed_sync() {
        tracing::debug!("adapter poll cycle skipped: vault is sealed");
        return Ok(());
    }

    let configs = state.engine.adapters.list_configs().await;
    if configs.is_empty() {
        return Ok(());
    }

    for config in configs {
        if !config.enabled {
            continue;
        }

        let adapter = match state.engine.adapters.get(config.id).await {
            Some(adapter) => adapter,
            None => continue,
        };

        let poll_key = adapter_poll_key(&config);
        let cursor = state
            .engine
            .store
            .nodes
            .get_poll_state(&poll_key)
            .await?
            .map(|state| state.cursor);

        let (messages, new_cursor) = match adapter.poll(cursor.as_deref()).await {
            Ok(result) => result,
            Err(err) => {
                tracing::warn!(
                    adapter = %config.name,
                    error = %err,
                    "adapter poll failed"
                );
                continue;
            }
        };

        let message_count = messages.len() as u64;
        if !messages.is_empty() {
            for message in messages {
                if let Err(err) = ingest_adapter_message(state, &config, message).await {
                    tracing::warn!(
                        adapter = %config.name,
                        error = %err,
                        "adapter message ingest failed"
                    );
                }
            }
        }
        if let Err(err) = state
            .engine
            .store
            .nodes
            .upsert_poll_state(&poll_key, &new_cursor, message_count)
            .await
        {
            tracing::warn!(
                adapter = %config.name,
                error = %err,
                "failed to persist adapter poll state"
            );
        }
    }

    Ok(())
}

async fn ingest_adapter_message(
    state: &Arc<AppState>,
    config: &AdapterConfig,
    message: AdapterInboundMessage,
) -> MvResult<()> {
    let adapter_label = config.adapter_type.to_string();
    let namespace = config
        .get_setting("namespace")
        .map(str::to_string)
        .unwrap_or_else(|| "default".to_string());

    let contact = ensure_adapter_contact(state, &adapter_label, &message.sender).await?;
    let channel =
        ensure_adapter_channel(state, &adapter_label, &message.channel, contact.id).await?;

    let mut relay_message = RelayMessage::inbound(channel.id, contact.id, message.content.clone())
        .with_content_type(ContentType::Text);

    if let Some(thread_id) = message.thread_id.as_deref() {
        relay_message = relay_message.with_thread(thread_uuid(&adapter_label, thread_id));
        relay_message.metadata.insert(
            "external_thread_id".to_string(),
            serde_json::Value::String(thread_id.to_string()),
        );
    }

    relay_message.metadata.insert(
        "adapter".to_string(),
        serde_json::Value::String(adapter_label.clone()),
    );
    relay_message.metadata.insert(
        "external_id".to_string(),
        serde_json::Value::String(message.external_id),
    );
    relay_message.metadata.insert(
        "external_channel_id".to_string(),
        serde_json::Value::String(message.channel),
    );
    relay_message.metadata.insert(
        "external_sender".to_string(),
        serde_json::Value::String(message.sender),
    );
    relay_message.metadata.insert(
        "received_at".to_string(),
        serde_json::Value::String(message.timestamp.to_rfc3339()),
    );

    let outcome = state
        .engine
        .receive_relay_message(relay_message, &namespace)
        .await?;

    if let Some(auto_reply) = outcome.auto_reply {
        tracing::info!(
            adapter = %adapter_label,
            message_id = %auto_reply.id,
            "auto-reply generated for adapter message (delivery pending)"
        );
    }

    Ok(())
}

async fn ensure_adapter_contact(
    state: &Arc<AppState>,
    adapter_label: &str,
    sender: &str,
) -> MvResult<RelayContact> {
    let sender = sender.trim();
    let sender = if sender.is_empty() { "unknown" } else { sender };
    let identity = format!("{adapter_label}:{sender}");
    let address = format!("{adapter_label}://{sender}");

    let contacts = state.engine.relay.list_contacts().await?;
    for contact in contacts {
        if contact.public_key == identity || contact.vault_address.as_deref() == Some(&address) {
            return Ok(contact);
        }
    }

    let display_name = format!("{adapter_label}:{sender}");
    let mut contact = RelayContact::new(display_name, identity);
    contact.vault_address = Some(address);
    contact.trust_level = TrustLevel::RelayOnly;
    contact.notes = Some(format!("Auto-created by {adapter_label} adapter"));
    state.engine.relay.add_contact(&contact).await?;
    Ok(contact)
}

async fn ensure_adapter_channel(
    state: &Arc<AppState>,
    adapter_label: &str,
    external_channel_id: &str,
    contact_id: Uuid,
) -> MvResult<RelayChannel> {
    let external_channel_id = external_channel_id.trim();
    let external_channel_id = if external_channel_id.is_empty() {
        "unknown"
    } else {
        external_channel_id
    };
    let name = format!("{adapter_label}:{external_channel_id}");

    let channels = state.engine.relay.list_channels().await?;
    for channel in channels {
        if channel.name.as_deref() == Some(&name) {
            return Ok(channel);
        }
    }

    let channel = RelayChannel::group(name, vec![contact_id]);
    state.engine.relay.create_channel(&channel).await?;
    Ok(channel)
}

fn adapter_poll_key(config: &AdapterConfig) -> String {
    format!("{}:{}", config.adapter_type, config.id)
}

fn thread_uuid(adapter_label: &str, thread_id: &str) -> Uuid {
    let key = format!("relay:{adapter_label}:thread:{thread_id}");
    Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_bytes())
}
