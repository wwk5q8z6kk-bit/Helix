use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use lettre::message::Mailbox;
use lettre::transport::smtp::authentication::Credentials as SmtpCredentials;
use lettre::{AsyncSmtpTransport, AsyncTransport, Message, Tokio1Executor};
use hx_core::{
    ChannelType, ContentType, KnowledgeNode, MessageStatus, HxError, MvResult, RelayChannel,
    RelayContact, RelayMessage, TrustLevel,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::rest::attachments::{
    extract_attachment_search_text, normalize_attachment_search_blob,
    split_attachment_search_chunks,
};
use crate::state::AppState;

const EMAIL_STATE_FILE: &str = "adapters/email_state.json";
const ATTACHMENT_TEXT_INDEX_METADATA_KEY: &str = "attachment_text_index";
const ATTACHMENT_TEXT_CHUNK_INDEX_METADATA_KEY: &str = "attachment_text_chunks";
const ATTACHMENT_SEARCH_BLOB_METADATA_KEY: &str = "attachment_search_text";
const MAX_ATTACHMENT_EXTRACTED_TEXT_CHARS: usize = 12_000;
const MAX_ATTACHMENT_SEARCH_BLOB_CHARS: usize = 64_000;
const MAX_ATTACHMENT_SEARCH_CHUNK_CHARS: usize = 420;
const MAX_ATTACHMENT_SEARCH_CHUNK_COUNT: usize = 32;

#[derive(Debug, Clone)]
struct RuntimeEmailConfig {
    namespace: String,
    max_fetch: usize,
    max_attachment_bytes: usize,
    mark_seen: bool,
    profile_display_name: String,
    profile_primary_email: Option<String>,
    profile_signature: Option<String>,
    imap_host: Option<String>,
    imap_port: u16,
    imap_username: Option<String>,
    imap_folder: String,
    smtp_host: Option<String>,
    smtp_port: u16,
    smtp_username: Option<String>,
    smtp_from: Option<String>,
}

impl RuntimeEmailConfig {
    async fn from_state(state: &AppState) -> Self {
        let cfg = &state.engine.config;
        let profile = state.engine.get_profile().await.ok();
        let profile_display_name = profile
            .as_ref()
            .map(|p| p.display_name.trim().to_string())
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| cfg.profile.display_name.clone());
        let profile_primary_email = profile
            .as_ref()
            .and_then(|p| p.email.as_ref().map(|value| value.trim().to_string()))
            .filter(|value| !value.is_empty())
            .or_else(|| cfg.profile.primary_email.clone());
        let profile_signature = profile
            .as_ref()
            .and_then(|p| {
                p.signature_name
                    .as_ref()
                    .map(|value| value.trim().to_string())
            })
            .filter(|value| !value.is_empty())
            .or_else(|| cfg.profile.signature.clone());
        Self {
            namespace: cfg.email.namespace.clone(),
            max_fetch: cfg.email.max_fetch.max(1),
            max_attachment_bytes: cfg.email.max_attachment_bytes.max(1024),
            mark_seen: cfg.email.mark_seen,
            profile_display_name,
            profile_primary_email,
            profile_signature,
            imap_host: cfg.email.imap_host.clone(),
            imap_port: cfg.email.imap_port,
            imap_username: cfg.email.imap_username.clone(),
            imap_folder: cfg.email.imap_folder.clone(),
            smtp_host: cfg.email.smtp_host.clone(),
            smtp_port: cfg.email.smtp_port,
            smtp_username: cfg.email.smtp_username.clone(),
            smtp_from: cfg.email.smtp_from.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct EmailAdapterState {
    folders: HashMap<String, FolderCursor>,
    thread_by_message_id: HashMap<String, String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct FolderCursor {
    uid_validity: Option<u32>,
    last_uid: u32,
}

#[derive(Debug, Clone)]
struct InboundAttachment {
    file_name: String,
    content_type: Option<String>,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct InboundEmail {
    uid: u32,
    message_id: Option<String>,
    sender_name: Option<String>,
    sender_email: String,
    subject: String,
    body: String,
    sent_at: DateTime<Utc>,
    in_reply_to: Option<String>,
    references: Vec<String>,
    attachments: Vec<InboundAttachment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NodeAttachmentRecord {
    id: String,
    file_name: String,
    content_type: Option<String>,
    size_bytes: usize,
    stored_path: String,
    uploaded_at: Option<String>,
    extraction_status: Option<String>,
    extracted_chars: Option<usize>,
}

fn load_state(path: &Path) -> EmailAdapterState {
    if !path.exists() {
        return EmailAdapterState::default();
    }
    match fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => EmailAdapterState::default(),
    }
}

fn save_state(path: &Path, state: &EmailAdapterState) -> MvResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| {
            HxError::Storage(format!(
                "failed creating email adapter state directory {}: {err}",
                parent.display()
            ))
        })?;
    }
    let payload = serde_json::to_string_pretty(state)
        .map_err(|err| HxError::Storage(format!("serialize email adapter state: {err}")))?;
    fs::write(path, payload).map_err(|err| {
        HxError::Storage(format!(
            "write email adapter state {}: {err}",
            path.display()
        ))
    })
}

fn state_file_path(data_dir: &str) -> PathBuf {
    PathBuf::from(data_dir).join(EMAIL_STATE_FILE)
}

fn normalize_message_id(value: &str) -> String {
    value
        .trim()
        .trim_matches('<')
        .trim_matches('>')
        .to_ascii_lowercase()
}

fn select_smtp_identity(config: &RuntimeEmailConfig) -> MvResult<(String, String)> {
    let username = config
        .smtp_username
        .clone()
        .or_else(|| config.profile_primary_email.clone())
        .ok_or_else(|| {
            HxError::Config("email.smtp_username or profile.primary_email required".into())
        })?;
    let from = config
        .smtp_from
        .clone()
        .or_else(|| config.profile_primary_email.clone())
        .unwrap_or_else(|| username.clone());
    Ok((username, from))
}

fn select_imap_username(config: &RuntimeEmailConfig) -> MvResult<String> {
    config
        .imap_username
        .clone()
        .or_else(|| config.profile_primary_email.clone())
        .ok_or_else(|| {
            HxError::Config("email.imap_username or profile.primary_email required".into())
        })
}

fn resolve_message_subject(message: &RelayMessage) -> String {
    message
        .metadata
        .get("subject")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("Helix Relay Message")
        .to_string()
}

pub async fn send_outbound_relay_if_email_channel(
    state: &AppState,
    message: &RelayMessage,
) -> MvResult<Option<String>> {
    let channel = match state.engine.relay.get_channel(message.channel_id).await? {
        Some(channel) => channel,
        None => return Ok(None),
    };

    let target_email = match resolve_email_channel_target(state, &channel).await? {
        Some(target) => target,
        None => return Ok(None),
    };

    if !state.engine.config.email.enabled {
        return Err(HxError::Config(
            "email adapter is disabled (set email.enabled=true)".into(),
        ));
    }

    let config = RuntimeEmailConfig::from_state(state).await;
    let smtp_host = config
        .smtp_host
        .clone()
        .ok_or_else(|| HxError::Config("email.smtp_host is required for outbound email".into()))?;
    let smtp_password = state
        .engine
        .credential_store
        .get_secret_string("HELIX_EMAIL_SMTP_PASSWORD")
        .ok_or_else(|| {
            HxError::Config(
                "missing secret HELIX_EMAIL_SMTP_PASSWORD (set with `helix secret set`)"
                    .into(),
            )
        })?;
    let (smtp_username, from_address) = select_smtp_identity(&config)?;

    let from_mailbox: Mailbox = if config.profile_display_name.trim().is_empty() {
        from_address
            .parse()
            .map_err(|err| HxError::InvalidInput(format!("invalid smtp_from address: {err}")))?
    } else {
        format!("{} <{}>", config.profile_display_name, from_address)
            .parse()
            .map_err(|err| HxError::InvalidInput(format!("invalid smtp_from address: {err}")))?
    };
    let to_mailbox: Mailbox = target_email
        .parse()
        .map_err(|err| HxError::InvalidInput(format!("invalid email recipient: {err}")))?;

    let subject = resolve_message_subject(message);
    let mut body = message.content.clone();
    if let Some(signature) = config.profile_signature.as_deref().map(str::trim) {
        if !signature.is_empty() {
            body.push_str("\n\n");
            body.push_str(signature);
        }
    }

    let email = Message::builder()
        .from(from_mailbox)
        .to(to_mailbox)
        .subject(subject)
        .body(body)
        .map_err(|err| HxError::Internal(format!("failed building SMTP message: {err}")))?;

    let transport = AsyncSmtpTransport::<Tokio1Executor>::starttls_relay(&smtp_host)
        .map_err(|err| HxError::Config(format!("invalid SMTP relay host {smtp_host}: {err}")))?
        .credentials(SmtpCredentials::new(smtp_username, smtp_password))
        .port(config.smtp_port)
        .build();

    transport
        .send(email)
        .await
        .map_err(|err| HxError::Internal(format!("SMTP send failed: {err}")))?;

    Ok(Some(target_email))
}

async fn resolve_email_channel_target(
    state: &AppState,
    channel: &RelayChannel,
) -> MvResult<Option<String>> {
    if channel.channel_type != ChannelType::Direct || channel.member_contact_ids.len() != 1 {
        return Ok(None);
    }

    let contact_id = channel.member_contact_ids[0];
    let contact = match state.engine.relay.get_contact(contact_id).await? {
        Some(contact) => contact,
        None => return Ok(None),
    };

    Ok(extract_email_from_contact(&contact))
}

fn extract_email_from_contact(contact: &RelayContact) -> Option<String> {
    if let Some(address) = contact.vault_address.as_deref() {
        if let Some(value) = address.strip_prefix("mailto:") {
            let email = value.trim();
            if !email.is_empty() {
                return Some(email.to_ascii_lowercase());
            }
        }
    }

    let value = contact.public_key.trim();
    if value.contains('@') {
        Some(value.to_ascii_lowercase())
    } else {
        None
    }
}

pub fn spawn_email_adapter(
    state: Arc<AppState>,
    mut shutdown_rx: tokio::sync::broadcast::Receiver<()>,
) {
    if !state.engine.config.email.enabled {
        tracing::info!("email adapter disabled by config");
        return;
    }

    let state_file = state_file_path(&state.engine.config.data_dir);
    let mut adapter_state = load_state(&state_file);
    let poll_interval_secs = state.engine.config.email.poll_interval_secs.max(30);

    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(std::time::Duration::from_secs(poll_interval_secs));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("email adapter shutdown received");
                    break;
                }
                _ = ticker.tick() => {
                    if let Err(err) = poll_and_ingest_emails(&state, &mut adapter_state).await {
                        tracing::warn!(error = %err, "email adapter poll cycle failed");
                    }
                    if let Err(err) = save_state(&state_file, &adapter_state) {
                        tracing::warn!(error = %err, "email adapter state save failed");
                    }
                }
            }
        }
    });

    tracing::info!("email adapter spawned");
}

async fn poll_and_ingest_emails(
    state: &Arc<AppState>,
    adapter_state: &mut EmailAdapterState,
) -> MvResult<()> {
    if state.engine.config.sealed_mode && !state.engine.keychain.is_unsealed_sync() {
        tracing::debug!("email adapter poll cycle skipped: vault is sealed");
        return Ok(());
    }

    let config = RuntimeEmailConfig::from_state(state).await;
    let imap_host = config
        .imap_host
        .clone()
        .ok_or_else(|| HxError::Config("email.imap_host is required for inbound polling".into()))?;
    let imap_username = select_imap_username(&config)?;
    let imap_password = state
        .engine
        .credential_store
        .get_secret_string("HELIX_EMAIL_IMAP_PASSWORD")
        .ok_or_else(|| {
            HxError::Config(
                "missing secret HELIX_EMAIL_IMAP_PASSWORD (set with `helix secret set`)"
                    .into(),
            )
        })?;
    let folder_key = format!("{}:{}", imap_host.to_ascii_lowercase(), config.imap_folder);
    let cursor = adapter_state
        .folders
        .get(&folder_key)
        .cloned()
        .unwrap_or_default();

    let request = ImapFetchRequest {
        host: imap_host,
        port: config.imap_port,
        username: imap_username,
        password: imap_password,
        folder: config.imap_folder.clone(),
        mark_seen: config.mark_seen,
        max_fetch: config.max_fetch,
        cursor,
        max_attachment_bytes: config.max_attachment_bytes,
    };

    let outcome = tokio::task::spawn_blocking(move || fetch_inbound_emails(request))
        .await
        .map_err(|err| HxError::Internal(format!("email adapter worker join error: {err}")))?
        .map_err(|err| HxError::Internal(format!("email adapter fetch failed: {err}")))?;

    adapter_state
        .folders
        .insert(folder_key, outcome.cursor.clone());

    let fetched = outcome.emails.len();
    let mut imported = 0usize;

    for email in outcome.emails {
        let imported_one = ingest_inbound_email(state, adapter_state, &config, email).await?;
        if imported_one {
            imported += 1;
        }
    }

    tracing::info!(
        fetched,
        imported,
        folder = %config.imap_folder,
        "email adapter poll cycle complete"
    );

    Ok(())
}

#[derive(Debug, Clone)]
struct ImapFetchRequest {
    host: String,
    port: u16,
    username: String,
    password: String,
    folder: String,
    mark_seen: bool,
    max_fetch: usize,
    cursor: FolderCursor,
    max_attachment_bytes: usize,
}

#[derive(Debug, Clone)]
struct ImapFetchOutcome {
    emails: Vec<InboundEmail>,
    cursor: FolderCursor,
}

async fn ingest_inbound_email(
    state: &Arc<AppState>,
    adapter_state: &mut EmailAdapterState,
    config: &RuntimeEmailConfig,
    email: InboundEmail,
) -> MvResult<bool> {
    let message_id = email
        .message_id
        .clone()
        .map(|value| normalize_message_id(&value));

    if let Some(message_id) = message_id.as_deref() {
        if adapter_state.thread_by_message_id.contains_key(message_id) {
            return Ok(false);
        }
    }

    let contact =
        ensure_email_contact(state, &email.sender_email, email.sender_name.as_deref()).await?;
    let channel = ensure_direct_channel_for_contact(state, contact.id).await?;
    let thread_id = resolve_thread_id(adapter_state, &email);
    let message_content = build_message_content(&email.subject, &email.body);

    let mut metadata = serde_json::Map::new();
    metadata.insert("adapter".into(), serde_json::Value::String("email".into()));
    metadata.insert(
        "email_uid".into(),
        serde_json::Value::Number(email.uid.into()),
    );
    metadata.insert(
        "email_sender".into(),
        serde_json::Value::String(email.sender_email.clone()),
    );
    metadata.insert(
        "email_subject".into(),
        serde_json::Value::String(email.subject.clone()),
    );
    metadata.insert(
        "email_sent_at".into(),
        serde_json::Value::String(email.sent_at.to_rfc3339()),
    );
    if let Some(message_id) = message_id.as_deref() {
        metadata.insert(
            "email_message_id".into(),
            serde_json::Value::String(message_id.to_string()),
        );
    }
    if let Some(in_reply_to) = email.in_reply_to.as_ref() {
        metadata.insert(
            "email_in_reply_to".into(),
            serde_json::Value::String(in_reply_to.clone()),
        );
    }
    if !email.references.is_empty() {
        metadata.insert(
            "email_references".into(),
            serde_json::Value::Array(
                email
                    .references
                    .iter()
                    .map(|value| serde_json::Value::String(value.clone()))
                    .collect(),
            ),
        );
    }
    if !email.attachments.is_empty() {
        metadata.insert(
            "email_attachment_count".into(),
            serde_json::Value::Number((email.attachments.len() as u64).into()),
        );
    }

    let mut relay_message = RelayMessage::inbound(channel.id, contact.id, message_content)
        .with_thread(thread_id)
        .with_content_type(ContentType::Text);
    relay_message.metadata = metadata.into_iter().collect();

    let outcome = state
        .engine
        .receive_relay_message(relay_message, &config.namespace)
        .await?;

    if let Some(mut auto_reply) = outcome.auto_reply {
        match send_outbound_relay_if_email_channel(state, &auto_reply).await {
            Ok(Some(recipient)) => {
                if let Err(err) = state
                    .engine
                    .relay
                    .update_status(auto_reply.id, MessageStatus::Delivered)
                    .await
                {
                    tracing::warn!(error = %err, "relay_auto_reply_status_update_failed");
                } else {
                    auto_reply.status = MessageStatus::Delivered;
                }
                auto_reply.metadata.insert(
                    "email_recipient".to_string(),
                    serde_json::Value::String(recipient),
                );
                auto_reply.metadata.insert(
                    "adapter".to_string(),
                    serde_json::Value::String("email".to_string()),
                );
            }
            Ok(None) => {}
            Err(err) => {
                let _ = state
                    .engine
                    .relay
                    .update_status(auto_reply.id, MessageStatus::Failed)
                    .await;
                tracing::warn!(error = %err, "relay_auto_reply_send_failed");
            }
        }
    }

    let stored = outcome.message;

    if let Some(node_id) = stored.vault_node_id {
        if !email.attachments.is_empty() {
            persist_inbound_attachments(
                state,
                node_id,
                &email.attachments,
                config.max_attachment_bytes,
            )
            .await?;
        }
    }

    if let Some(message_id) = message_id {
        adapter_state
            .thread_by_message_id
            .insert(message_id, thread_id.to_string());
    }

    Ok(true)
}

fn build_message_content(subject: &str, body: &str) -> String {
    let subject = subject.trim();
    let body = body.trim();
    if body.is_empty() {
        subject.to_string()
    } else if subject.is_empty() {
        body.to_string()
    } else {
        format!("{subject}\n\n{body}")
    }
}

fn resolve_thread_id(adapter_state: &EmailAdapterState, email: &InboundEmail) -> Uuid {
    let mut candidates = Vec::new();
    if let Some(message_id) = email.message_id.as_deref() {
        candidates.push(normalize_message_id(message_id));
    }
    if let Some(in_reply_to) = email.in_reply_to.as_deref() {
        candidates.push(normalize_message_id(in_reply_to));
    }
    for value in &email.references {
        candidates.push(normalize_message_id(value));
    }

    for key in candidates {
        if let Some(existing) = adapter_state.thread_by_message_id.get(&key) {
            if let Ok(thread_id) = Uuid::parse_str(existing) {
                return thread_id;
            }
        }
    }

    Uuid::now_v7()
}

async fn ensure_email_contact(
    state: &Arc<AppState>,
    sender_email: &str,
    sender_name: Option<&str>,
) -> MvResult<RelayContact> {
    let normalized_email = sender_email.trim().to_ascii_lowercase();
    let contacts = state.engine.relay.list_contacts().await?;
    for contact in contacts {
        if let Some(email) = extract_email_from_contact(&contact) {
            if email == normalized_email {
                return Ok(contact);
            }
        }
    }

    let display_name = sender_name
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(&normalized_email);
    let mut contact = RelayContact::new(display_name, normalized_email.clone());
    contact = contact.with_address(format!("mailto:{normalized_email}"));
    contact = contact.with_trust(TrustLevel::RelayOnly);
    contact.notes = Some("Auto-created by email adapter".to_string());
    state.engine.relay.add_contact(&contact).await?;
    Ok(contact)
}

async fn ensure_direct_channel_for_contact(
    state: &Arc<AppState>,
    contact_id: Uuid,
) -> MvResult<RelayChannel> {
    let channels = state.engine.relay.list_channels().await?;
    for channel in channels {
        if channel.channel_type == ChannelType::Direct
            && channel.member_contact_ids.len() == 1
            && channel.member_contact_ids[0] == contact_id
        {
            return Ok(channel);
        }
    }

    let channel = RelayChannel::direct(contact_id);
    state.engine.relay.create_channel(&channel).await?;
    Ok(channel)
}

fn fetch_inbound_emails(request: ImapFetchRequest) -> Result<ImapFetchOutcome, String> {
    let tls = native_tls::TlsConnector::builder()
        .build()
        .map_err(|err| format!("build TLS connector: {err}"))?;
    let client = imap::connect(
        (request.host.as_str(), request.port),
        request.host.as_str(),
        &tls,
    )
    .map_err(|err| format!("connect IMAP: {err}"))?;
    let mut session = client
        .login(request.username.clone(), request.password.clone())
        .map_err(|(err, _client)| format!("login IMAP: {err}"))?;

    let mailbox = session
        .select(&request.folder)
        .map_err(|err| format!("select folder {}: {err}", request.folder))?;
    let uid_validity = mailbox.uid_validity;
    let mut last_uid = request.cursor.last_uid;
    if request.cursor.uid_validity != uid_validity {
        last_uid = 0;
    }

    let query = if last_uid == 0 {
        "1:*".to_string()
    } else {
        format!("{}:*", last_uid + 1)
    };

    let mut uids: Vec<u32> = session
        .uid_search(query)
        .map_err(|err| format!("IMAP uid_search failed: {err}"))?
        .into_iter()
        .collect();
    uids.sort_unstable();
    if uids.len() > request.max_fetch {
        let keep = uids.split_off(uids.len() - request.max_fetch);
        uids = keep;
    }

    let mut emails = Vec::new();
    for uid in uids {
        let fetches = session
            .uid_fetch(uid.to_string(), "(UID BODY.PEEK[])")
            .map_err(|err| format!("IMAP uid_fetch for {uid} failed: {err}"))?;

        for fetch in fetches.iter() {
            let Some(raw_body) = fetch.body() else {
                continue;
            };

            if let Ok(email) = parse_inbound_email(uid, raw_body, request.max_attachment_bytes) {
                emails.push(email);
                last_uid = last_uid.max(uid);
                if request.mark_seen {
                    let _ = session.uid_store(uid.to_string(), "+FLAGS (\\Seen)");
                }
            }
        }
    }

    let _ = session.logout();

    Ok(ImapFetchOutcome {
        emails,
        cursor: FolderCursor {
            uid_validity,
            last_uid,
        },
    })
}

fn parse_inbound_email(
    uid: u32,
    raw: &[u8],
    max_attachment_bytes: usize,
) -> Result<InboundEmail, String> {
    use mailparse::MailHeaderMap;

    let parsed = mailparse::parse_mail(raw).map_err(|err| format!("parse email: {err}"))?;
    let from_header = parsed
        .headers
        .get_first_value("From")
        .unwrap_or_else(|| "unknown@local".to_string());
    let (sender_name, sender_email) = parse_from_header(&from_header);
    let subject = parsed
        .headers
        .get_first_value("Subject")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "(no subject)".to_string());
    let message_id = parsed
        .headers
        .get_first_value("Message-ID")
        .map(|value| normalize_message_id(&value))
        .filter(|value| !value.is_empty());
    let in_reply_to = parsed
        .headers
        .get_first_value("In-Reply-To")
        .map(|value| normalize_message_id(&value))
        .filter(|value| !value.is_empty());
    let references = parsed
        .headers
        .get_first_value("References")
        .map(|value| {
            value
                .split_whitespace()
                .map(normalize_message_id)
                .filter(|item| !item.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let sent_at = parsed
        .headers
        .get_first_value("Date")
        .and_then(|value| mailparse::dateparse(&value).ok())
        .and_then(|value| DateTime::<Utc>::from_timestamp(value, 0))
        .unwrap_or_else(Utc::now);

    let mut plain_text_parts = Vec::new();
    let mut html_parts = Vec::new();
    let mut attachments = Vec::new();
    collect_parts(
        &parsed,
        max_attachment_bytes,
        &mut plain_text_parts,
        &mut html_parts,
        &mut attachments,
    );

    let body = if !plain_text_parts.is_empty() {
        plain_text_parts.join("\n\n")
    } else if !html_parts.is_empty() {
        html_parts.join("\n\n")
    } else {
        String::new()
    };

    Ok(InboundEmail {
        uid,
        message_id,
        sender_name,
        sender_email,
        subject,
        body: normalize_attachment_search_blob(&body, MAX_ATTACHMENT_SEARCH_BLOB_CHARS),
        sent_at,
        in_reply_to,
        references,
        attachments,
    })
}

fn collect_parts(
    part: &mailparse::ParsedMail<'_>,
    max_attachment_bytes: usize,
    plain_text_parts: &mut Vec<String>,
    html_parts: &mut Vec<String>,
    attachments: &mut Vec<InboundAttachment>,
) {
    let disposition = part.get_content_disposition();
    let filename = disposition
        .params
        .get("filename")
        .cloned()
        .or_else(|| part.ctype.params.get("name").cloned());
    let is_attachment = matches!(
        disposition.disposition,
        mailparse::DispositionType::Attachment
    ) || filename.is_some();

    if part.subparts.is_empty() {
        let mime = part.ctype.mimetype.to_ascii_lowercase();
        if is_attachment {
            if let Ok(bytes) = part.get_body_raw() {
                if !bytes.is_empty() && bytes.len() <= max_attachment_bytes {
                    let file_name = filename
                        .unwrap_or_else(|| format!("attachment-{}.bin", attachments.len() + 1));
                    attachments.push(InboundAttachment {
                        file_name,
                        content_type: if mime.is_empty() { None } else { Some(mime) },
                        bytes,
                    });
                }
            }
            return;
        }

        if mime == "text/plain" {
            if let Ok(body) = part.get_body() {
                let normalized =
                    normalize_attachment_search_blob(&body, MAX_ATTACHMENT_SEARCH_BLOB_CHARS);
                if !normalized.is_empty() {
                    plain_text_parts.push(normalized);
                }
            }
        } else if mime == "text/html" {
            if let Ok(body) = part.get_body() {
                let text = html_to_text(&body);
                let normalized =
                    normalize_attachment_search_blob(&text, MAX_ATTACHMENT_SEARCH_BLOB_CHARS);
                if !normalized.is_empty() {
                    html_parts.push(normalized);
                }
            }
        }
        return;
    }

    for child in &part.subparts {
        collect_parts(
            child,
            max_attachment_bytes,
            plain_text_parts,
            html_parts,
            attachments,
        );
    }
}

fn parse_from_header(value: &str) -> (Option<String>, String) {
    let trimmed = value.trim();
    if let (Some(start), Some(end)) = (trimmed.rfind('<'), trimmed.rfind('>')) {
        if start < end {
            let email = trimmed[start + 1..end].trim().to_ascii_lowercase();
            if email.contains('@') {
                let name = trimmed[..start].trim().trim_matches('"').trim();
                let name = if name.is_empty() {
                    None
                } else {
                    Some(name.to_string())
                };
                return (name, email);
            }
        }
    }

    let email = trimmed
        .split_whitespace()
        .map(|item| item.trim_matches(|ch: char| ch == ',' || ch == ';' || ch == '"' || ch == '\''))
        .find(|token| token.contains('@'))
        .unwrap_or("unknown@local")
        .to_ascii_lowercase();
    (None, email)
}

fn html_to_text(html: &str) -> String {
    let mut output = String::with_capacity(html.len());
    let mut inside_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => inside_tag = true,
            '>' => inside_tag = false,
            _ if !inside_tag => output.push(ch),
            _ => {}
        }
    }
    output
}

async fn persist_inbound_attachments(
    state: &Arc<AppState>,
    node_id: Uuid,
    attachments: &[InboundAttachment],
    max_attachment_bytes: usize,
) -> MvResult<()> {
    let Some(mut node) = state.engine.get_node(node_id).await? else {
        return Ok(());
    };

    let mut records = node
        .metadata
        .get("attachments")
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<NodeAttachmentRecord>>(value).ok())
        .unwrap_or_default();

    let mut changed = false;
    for attachment in attachments {
        if attachment.bytes.is_empty() || attachment.bytes.len() > max_attachment_bytes {
            continue;
        }

        let attachment_id = Uuid::now_v7().to_string();
        let sanitized_name = sanitize_file_name(&attachment.file_name);
        let relative_path = format!("blobs/{}/{}_{}", node_id, attachment_id, sanitized_name);
        let absolute_path = PathBuf::from(&state.engine.config.data_dir).join(&relative_path);

        if let Some(parent) = absolute_path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
                HxError::Storage(format!(
                    "failed creating attachment dir {}: {err}",
                    parent.display()
                ))
            })?;
        }
        fs::write(&absolute_path, &attachment.bytes).map_err(|err| {
            HxError::Storage(format!(
                "failed writing attachment {}: {err}",
                absolute_path.display()
            ))
        })?;

        let extraction = extract_attachment_search_text(
            &attachment.file_name,
            attachment.content_type.as_deref(),
            &attachment.bytes,
            MAX_ATTACHMENT_EXTRACTED_TEXT_CHARS,
        );

        upsert_attachment_text_index_entry(
            &mut node,
            &attachment_id,
            extraction.extracted_text.as_deref(),
        );
        upsert_attachment_text_chunk_index_entry(
            &mut node,
            &attachment_id,
            extraction.extracted_text.as_deref(),
        );

        records.push(NodeAttachmentRecord {
            id: attachment_id,
            file_name: attachment.file_name.clone(),
            content_type: attachment.content_type.clone(),
            size_bytes: attachment.bytes.len(),
            stored_path: relative_path,
            uploaded_at: Some(Utc::now().to_rfc3339()),
            extraction_status: Some(extraction.status),
            extracted_chars: Some(extraction.extracted_chars),
        });
        changed = true;
    }

    if !changed {
        return Ok(());
    }

    node.metadata.insert(
        "attachments".to_string(),
        serde_json::to_value(&records)
            .map_err(|err| HxError::Internal(format!("serialize attachment records: {err}")))?,
    );
    sync_attachment_search_blob_metadata(&mut node);
    let updated = state.engine.update_node(node).await?;
    state.notify_change(&updated.id.to_string(), "update", Some(&updated.namespace));
    Ok(())
}

fn sanitize_file_name(value: &str) -> String {
    let mut sanitized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();

    if sanitized.is_empty() {
        sanitized = "attachment.bin".to_string();
    }
    if sanitized.len() > 120 {
        sanitized.truncate(120);
    }
    sanitized
}

fn upsert_attachment_text_index_entry(
    node: &mut KnowledgeNode,
    attachment_id: &str,
    extracted_text: Option<&str>,
) {
    let mut index_map = node
        .metadata
        .get(ATTACHMENT_TEXT_INDEX_METADATA_KEY)
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();

    match extracted_text {
        Some(value) if !value.trim().is_empty() => {
            index_map.insert(
                attachment_id.to_string(),
                serde_json::Value::String(value.to_string()),
            );
        }
        _ => {
            index_map.remove(attachment_id);
        }
    }

    if index_map.is_empty() {
        node.metadata.remove(ATTACHMENT_TEXT_INDEX_METADATA_KEY);
    } else {
        node.metadata.insert(
            ATTACHMENT_TEXT_INDEX_METADATA_KEY.to_string(),
            serde_json::Value::Object(index_map),
        );
    }
}

fn upsert_attachment_text_chunk_index_entry(
    node: &mut KnowledgeNode,
    attachment_id: &str,
    extracted_text: Option<&str>,
) {
    let mut chunk_index = node
        .metadata
        .get(ATTACHMENT_TEXT_CHUNK_INDEX_METADATA_KEY)
        .and_then(serde_json::Value::as_object)
        .cloned()
        .unwrap_or_default();

    let chunks = extracted_text
        .map(|value| {
            split_attachment_search_chunks(
                value,
                MAX_ATTACHMENT_SEARCH_CHUNK_CHARS,
                MAX_ATTACHMENT_SEARCH_CHUNK_COUNT,
            )
        })
        .unwrap_or_default();

    if chunks.is_empty() {
        chunk_index.remove(attachment_id);
    } else {
        chunk_index.insert(
            attachment_id.to_string(),
            serde_json::Value::Array(chunks.into_iter().map(serde_json::Value::String).collect()),
        );
    }

    if chunk_index.is_empty() {
        node.metadata
            .remove(ATTACHMENT_TEXT_CHUNK_INDEX_METADATA_KEY);
    } else {
        node.metadata.insert(
            ATTACHMENT_TEXT_CHUNK_INDEX_METADATA_KEY.to_string(),
            serde_json::Value::Object(chunk_index),
        );
    }
}

fn sync_attachment_search_blob_metadata(node: &mut KnowledgeNode) {
    let combined_text = node
        .metadata
        .get(ATTACHMENT_TEXT_INDEX_METADATA_KEY)
        .and_then(serde_json::Value::as_object)
        .map(|values| {
            values
                .values()
                .filter_map(serde_json::Value::as_str)
                .collect::<Vec<_>>()
                .join("\n\n")
        })
        .unwrap_or_default();

    let normalized =
        normalize_attachment_search_blob(&combined_text, MAX_ATTACHMENT_SEARCH_BLOB_CHARS);
    if normalized.is_empty() {
        node.metadata.remove(ATTACHMENT_SEARCH_BLOB_METADATA_KEY);
    } else {
        node.metadata.insert(
            ATTACHMENT_SEARCH_BLOB_METADATA_KEY.to_string(),
            serde_json::Value::String(normalized),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- parse_from_header ---

    #[test]
    fn parse_from_header_with_name() {
        let (name, email) = parse_from_header("John Doe <john@example.com>");
        assert_eq!(name, Some("John Doe".to_string()));
        assert_eq!(email, "john@example.com");
    }

    #[test]
    fn parse_from_header_bare_email() {
        let (name, email) = parse_from_header("user@example.com");
        assert!(name.is_none());
        assert_eq!(email, "user@example.com");
    }

    #[test]
    fn parse_from_header_quoted_name() {
        let (name, email) = parse_from_header("\"Doe, John\" <john@example.com>");
        assert_eq!(name, Some("Doe, John".to_string()));
        assert_eq!(email, "john@example.com");
    }

    #[test]
    fn parse_from_header_no_at_sign() {
        let (name, email) = parse_from_header("invalid");
        assert!(name.is_none());
        assert_eq!(email, "unknown@local");
    }

    // --- normalize_message_id ---

    #[test]
    fn normalize_message_id_strips_angles() {
        assert_eq!(normalize_message_id("<abc@example.com>"), "abc@example.com");
    }

    #[test]
    fn normalize_message_id_trims_whitespace() {
        assert_eq!(normalize_message_id("  <ABC@EX.COM>  "), "abc@ex.com");
    }

    // --- build_message_content ---

    #[test]
    fn build_message_content_subject_and_body() {
        let result = build_message_content("Hello", "World");
        assert_eq!(result, "Hello\n\nWorld");
    }

    #[test]
    fn build_message_content_empty_body() {
        let result = build_message_content("Subject Only", "");
        assert_eq!(result, "Subject Only");
    }

    #[test]
    fn build_message_content_empty_subject() {
        let result = build_message_content("", "Body Only");
        assert_eq!(result, "Body Only");
    }

    // --- sanitize_file_name ---

    #[test]
    fn sanitize_file_name_strips_special_chars() {
        assert_eq!(
            sanitize_file_name("hello world!@#.txt"),
            "hello_world___.txt"
        );
    }

    #[test]
    fn sanitize_file_name_truncates_long() {
        let long_name = "a".repeat(200);
        let result = sanitize_file_name(&long_name);
        assert_eq!(result.len(), 120);
    }

    #[test]
    fn sanitize_file_name_empty_returns_default() {
        assert_eq!(sanitize_file_name(""), "attachment.bin");
    }

    // --- html_to_text ---

    #[test]
    fn html_to_text_strips_tags() {
        assert_eq!(html_to_text("<p>Hello</p>"), "Hello");
    }

    #[test]
    fn html_to_text_preserves_text() {
        assert_eq!(html_to_text("No tags here"), "No tags here");
    }

    #[test]
    fn html_to_text_nested_tags() {
        assert_eq!(
            html_to_text("<div><b>Bold</b> and <i>italic</i></div>"),
            "Bold and italic"
        );
    }

    // --- extract_email_from_contact ---

    #[test]
    fn extract_email_from_contact_mailto() {
        let mut contact = RelayContact::new("Test", "test-key");
        contact.vault_address = Some("mailto:user@example.com".to_string());
        let result = extract_email_from_contact(&contact);
        assert_eq!(result, Some("user@example.com".to_string()));
    }

    #[test]
    fn extract_email_from_contact_public_key_email() {
        let contact = RelayContact::new("Test", "user@example.com");
        let result = extract_email_from_contact(&contact);
        assert_eq!(result, Some("user@example.com".to_string()));
    }

    #[test]
    fn extract_email_from_contact_no_email() {
        let contact = RelayContact::new("Test", "not-an-email");
        let result = extract_email_from_contact(&contact);
        assert!(result.is_none());
    }
}
