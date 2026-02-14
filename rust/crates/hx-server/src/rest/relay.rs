use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{
    ChannelType, ContentType, MessageStatus, RelayChannel, RelayContact, RelayMessage, TrustLevel,
};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct CreateContactDto {
    pub display_name: String,
    pub public_key: String,
    pub vault_address: Option<String>,
    pub trust_level: Option<String>,
    pub notes: Option<String>,
}

#[derive(Deserialize)]
pub struct UpdateContactDto {
    pub display_name: Option<String>,
    pub vault_address: Option<String>,
    pub trust_level: Option<String>,
    pub notes: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateChannelDto {
    pub name: Option<String>,
    pub channel_type: Option<String>,
    pub member_contact_ids: Vec<String>,
}

#[derive(Deserialize)]
pub struct SendMessageDto {
    pub content: String,
    pub content_type: Option<String>,
    pub thread_id: Option<String>,
    pub subject: Option<String>,
}

#[derive(Deserialize)]
pub struct ReceiveMessageDto {
    pub content: String,
    pub content_type: Option<String>,
    pub thread_id: Option<String>,
    pub sender_contact_id: Option<String>,
    pub recipient_contact_id: Option<String>,
}

#[derive(Deserialize)]
pub struct ListMessagesQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Deserialize)]
pub struct UpdateMessageStatusDto {
    pub status: String,
}

#[derive(Deserialize)]
pub struct UnreadQuery {
    pub channel_id: Option<String>,
}

#[derive(Serialize)]
pub struct UnreadCountResponse {
    pub count: usize,
}

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::NodeNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

fn map_email_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::Config(_) | hx_core::HxError::InvalidInput(_) => {
            (StatusCode::BAD_REQUEST, err.to_string())
        }
        _ => (StatusCode::BAD_GATEWAY, err.to_string()),
    }
}

// --- Contact Handlers ---

/// GET /api/v1/relay/contacts
pub async fn list_contacts(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<RelayContact>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let contacts = state
        .engine
        .relay
        .list_contacts()
        .await
        .map_err(map_hx_error)?;

    Ok(Json(contacts))
}

/// POST /api/v1/relay/contacts
pub async fn create_contact(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateContactDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let mut contact = RelayContact::new(req.display_name, req.public_key);

    if let Some(addr) = req.vault_address {
        contact = contact.with_address(addr);
    }

    if let Some(ref trust_str) = req.trust_level {
        let trust: TrustLevel = trust_str
            .parse()
            .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("invalid trust_level: {e}")))?;
        contact = contact.with_trust(trust);
    }

    if let Some(notes) = req.notes {
        contact.notes = Some(notes);
    }

    state
        .engine
        .relay
        .add_contact(&contact)
        .await
        .map_err(map_hx_error)?;

    Ok((StatusCode::CREATED, Json(contact)).into_response())
}

/// GET /api/v1/relay/contacts/:id
pub async fn get_contact(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let contact = state
        .engine
        .relay
        .get_contact(uuid)
        .await
        .map_err(map_hx_error)?;

    match contact {
        Some(c) => Ok(Json(c).into_response()),
        None => Err((StatusCode::NOT_FOUND, "contact not found".to_string())),
    }
}

/// PUT /api/v1/relay/contacts/:id
pub async fn update_contact(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateContactDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let mut contact = state
        .engine
        .relay
        .get_contact(uuid)
        .await
        .map_err(map_hx_error)?
        .ok_or((StatusCode::NOT_FOUND, "contact not found".to_string()))?;

    if let Some(name) = req.display_name {
        contact.display_name = name;
    }
    if let Some(addr) = req.vault_address {
        contact.vault_address = Some(addr);
    }
    if let Some(ref trust_str) = req.trust_level {
        let trust: TrustLevel = trust_str
            .parse()
            .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("invalid trust_level: {e}")))?;
        contact.trust_level = trust;
    }
    if let Some(notes) = req.notes {
        contact.notes = Some(notes);
    }
    contact.updated_at = Some(chrono::Utc::now());

    let updated = state
        .engine
        .relay
        .update_contact(&contact)
        .await
        .map_err(map_hx_error)?;

    if !updated {
        return Err((StatusCode::NOT_FOUND, "contact not found".to_string()));
    }

    Ok(Json(contact).into_response())
}

/// DELETE /api/v1/relay/contacts/:id
pub async fn delete_contact(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let deleted = state
        .engine
        .relay
        .delete_contact(uuid)
        .await
        .map_err(map_hx_error)?;

    if !deleted {
        return Err((StatusCode::NOT_FOUND, "contact not found".to_string()));
    }

    Ok(StatusCode::NO_CONTENT.into_response())
}

// --- Channel Handlers ---

/// GET /api/v1/relay/channels
pub async fn list_channels(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<RelayChannel>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let channels = state
        .engine
        .relay
        .list_channels()
        .await
        .map_err(map_hx_error)?;

    Ok(Json(channels))
}

/// POST /api/v1/relay/channels
pub async fn create_channel(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateChannelDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let member_ids: Vec<Uuid> = req
        .member_contact_ids
        .iter()
        .map(|s| {
            Uuid::parse_str(s).map_err(|_| (StatusCode::BAD_REQUEST, format!("invalid uuid: {s}")))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let channel_type = if let Some(ref ct_str) = req.channel_type {
        ct_str.parse::<ChannelType>().map_err(|e: String| {
            (
                StatusCode::BAD_REQUEST,
                format!("invalid channel_type: {e}"),
            )
        })?
    } else if req.name.is_some() || member_ids.len() > 1 {
        ChannelType::Group
    } else {
        ChannelType::Direct
    };

    let channel = match channel_type {
        ChannelType::Direct => {
            let contact_id = member_ids.first().copied().ok_or((
                StatusCode::BAD_REQUEST,
                "direct channel requires at least one member".to_string(),
            ))?;
            let mut ch = RelayChannel::direct(contact_id);
            ch.name = req.name;
            ch
        }
        ChannelType::Group => {
            let name = req.name.unwrap_or_else(|| "unnamed group".to_string());
            RelayChannel::group(name, member_ids)
        }
    };

    state
        .engine
        .relay
        .create_channel(&channel)
        .await
        .map_err(map_hx_error)?;

    Ok((StatusCode::CREATED, Json(channel)).into_response())
}

/// DELETE /api/v1/relay/channels/:id
pub async fn delete_channel(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let deleted = state
        .engine
        .relay
        .delete_channel(uuid)
        .await
        .map_err(map_hx_error)?;

    if !deleted {
        return Err((StatusCode::NOT_FOUND, "channel not found".to_string()));
    }

    Ok(StatusCode::NO_CONTENT.into_response())
}

// --- Message Handlers ---

/// GET /api/v1/relay/channels/:id/messages
pub async fn list_messages(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<ListMessagesQuery>,
) -> Result<Json<Vec<RelayMessage>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let channel_id =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let limit = params.limit.unwrap_or(50).min(200);
    let offset = params.offset.unwrap_or(0);

    let messages = state
        .engine
        .relay
        .list_messages(channel_id, limit, offset)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(messages))
}

/// POST /api/v1/relay/channels/:id/messages
pub async fn send_message(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<SendMessageDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let channel_id =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let mut message = RelayMessage::outbound(channel_id, req.content);

    if let Some(ref ct_str) = req.content_type {
        let ct: ContentType = ct_str.parse().map_err(|e: String| {
            (
                StatusCode::BAD_REQUEST,
                format!("invalid content_type: {e}"),
            )
        })?;
        message = message.with_content_type(ct);
    }

    if let Some(ref thread_str) = req.thread_id {
        let thread_id = Uuid::parse_str(thread_str)
            .map_err(|_| (StatusCode::BAD_REQUEST, "invalid thread_id".to_string()))?;
        message = message.with_thread(thread_id);
    }

    if let Some(subject) = req.subject.as_ref().map(|value| value.trim()) {
        if !subject.is_empty() {
            message.metadata.insert(
                "subject".to_string(),
                serde_json::Value::String(subject.to_string()),
            );
        }
    }

    let namespace = auth.namespace.as_deref().unwrap_or("default");

    let mut stored = state
        .engine
        .relay
        .send_message(message, namespace)
        .await
        .map_err(map_hx_error)?;

    match crate::email::send_outbound_relay_if_email_channel(&state, &stored).await {
        Ok(Some(recipient)) => {
            state
                .engine
                .relay
                .update_status(stored.id, MessageStatus::Delivered)
                .await
                .map_err(map_hx_error)?;
            stored.status = MessageStatus::Delivered;
            stored.metadata.insert(
                "email_recipient".to_string(),
                serde_json::Value::String(recipient),
            );
            stored.metadata.insert(
                "adapter".to_string(),
                serde_json::Value::String("email".to_string()),
            );
        }
        Ok(None) => {}
        Err(err) => {
            let _ = state
                .engine
                .relay
                .update_status(stored.id, MessageStatus::Failed)
                .await;
            stored.status = MessageStatus::Failed;
            return Err(map_email_error(err));
        }
    }

    Ok((StatusCode::CREATED, Json(stored)).into_response())
}

/// POST /api/v1/relay/channels/:id/inbound
pub async fn receive_message(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<ReceiveMessageDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let channel_id =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let mut message = RelayMessage::outbound(channel_id, req.content);

    if let Some(ref ct_str) = req.content_type {
        let ct: ContentType = ct_str.parse().map_err(|e: String| {
            (
                StatusCode::BAD_REQUEST,
                format!("invalid content_type: {e}"),
            )
        })?;
        message = message.with_content_type(ct);
    }

    if let Some(ref thread_str) = req.thread_id {
        let thread_id = Uuid::parse_str(thread_str)
            .map_err(|_| (StatusCode::BAD_REQUEST, "invalid thread_id".to_string()))?;
        message = message.with_thread(thread_id);
    }

    if let Some(ref sender_str) = req.sender_contact_id {
        let sender_id = Uuid::parse_str(sender_str).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                "invalid sender_contact_id".to_string(),
            )
        })?;
        message.sender_contact_id = Some(sender_id);
    }

    if let Some(ref recipient_str) = req.recipient_contact_id {
        let recipient_id = Uuid::parse_str(recipient_str).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                "invalid recipient_contact_id".to_string(),
            )
        })?;
        message.recipient_contact_id = Some(recipient_id);
    }

    let namespace = auth.namespace.as_deref().unwrap_or("default");

    let outcome = state
        .engine
        .receive_relay_message(message, namespace)
        .await
        .map_err(map_hx_error)?;

    if let Some(mut auto_reply) = outcome.auto_reply {
        match crate::email::send_outbound_relay_if_email_channel(&state, &auto_reply).await {
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

    Ok((StatusCode::CREATED, Json(outcome.message)).into_response())
}

/// POST /api/v1/relay/messages/:id/read
pub async fn mark_read(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let updated = state
        .engine
        .relay
        .mark_read(uuid)
        .await
        .map_err(map_hx_error)?;

    if !updated {
        return Err((StatusCode::NOT_FOUND, "message not found".to_string()));
    }

    Ok(Json(serde_json::json!({ "id": uuid.to_string(), "status": "read" })).into_response())
}

/// POST /api/v1/relay/messages/:id/status
pub async fn update_message_status(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateMessageStatusDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let status: MessageStatus = req
        .status
        .parse()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("invalid status: {e}")))?;

    let updated = state
        .engine
        .relay
        .update_status(uuid, status)
        .await
        .map_err(map_hx_error)?;

    if !updated {
        return Err((StatusCode::NOT_FOUND, "message not found".to_string()));
    }

    Ok(
        Json(serde_json::json!({ "id": uuid.to_string(), "status": status.to_string() }))
            .into_response(),
    )
}

/// GET /api/v1/relay/unread
pub async fn unread_count(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<UnreadQuery>,
) -> Result<Json<UnreadCountResponse>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let channel_id = if let Some(ref cid_str) = params.channel_id {
        Some(
            Uuid::parse_str(cid_str)
                .map_err(|_| (StatusCode::BAD_REQUEST, "invalid channel_id".to_string()))?,
        )
    } else {
        None
    };

    let count = state
        .engine
        .relay
        .unread_count(channel_id)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(UnreadCountResponse { count }))
}
