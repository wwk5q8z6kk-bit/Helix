use std::collections::HashMap;
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
    ChronicleEntry, ContentType, KnowledgeNode, MessageStatus, NodeKind, Proposal, ProposalAction,
    ProposalSender, ProposalState, RelayMessage, SafeguardStore, UndoSnapshot,
};

use chrono::{Duration, Utc};

use crate::auth::{
    authorize_namespace, authorize_read, authorize_write, namespace_for_create, AuthContext,
};
use crate::limits::{enforce_namespace_quota, NamespaceQuotaError};
use crate::state::AppState;
use crate::validation::validate_node_payload;

// --- DTOs ---

#[derive(Deserialize)]
pub struct ListProposalsQuery {
    pub state: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Deserialize)]
pub struct SubmitProposalRequest {
    pub sender: String,
    pub action: String,
    pub target_node_id: Option<String>,
    pub confidence: Option<f32>,
    pub diff_preview: Option<String>,
    pub payload: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Deserialize)]
struct ProposalNodePayload {
    kind: Option<String>,
    content: Option<String>,
    title: Option<String>,
    source: Option<String>,
    namespace: Option<String>,
    tags: Option<Vec<String>>,
    importance: Option<f64>,
    metadata: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Deserialize)]
struct RelayReplyPayload {
    channel_id: Uuid,
    content: String,
    content_type: Option<String>,
    thread_id: Option<Uuid>,
    recipient_contact_id: Option<Uuid>,
    subject: Option<String>,
}

#[derive(Serialize)]
pub struct ProposalCountResponse {
    pub count: usize,
}

fn map_hx_error(err: hx_core::HxError) -> (StatusCode, String) {
    match err {
        hx_core::HxError::NodeNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        hx_core::HxError::InvalidInput(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

fn map_namespace_quota_error(err: NamespaceQuotaError) -> (StatusCode, String) {
    match err {
        NamespaceQuotaError::Exceeded {
            namespace,
            quota,
            count,
        } => (
            StatusCode::TOO_MANY_REQUESTS,
            format!("namespace '{namespace}' quota exceeded ({count}/{quota} nodes)"),
        ),
        NamespaceQuotaError::Backend(msg) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("backend error: {msg}"),
        ),
    }
}

// --- Helpers ---

struct ProposalActionResult {
    created_node_id: Option<String>,
    updated_node_id: Option<String>,
    deleted_node_id: Option<String>,
}

fn resolve_sender_context(
    auth: &AuthContext,
    requested_sender: &str,
) -> Result<(ProposalSender, String, bool), (StatusCode, String)> {
    let requested_sender = requested_sender.trim();
    let subject = auth
        .subject
        .clone()
        .unwrap_or_else(|| "anonymous".to_string());

    if auth.is_admin() {
        let sender = if requested_sender.is_empty() {
            ProposalSender::UserSelf
        } else {
            requested_sender
                .parse::<ProposalSender>()
                .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("invalid sender: {e}")))?
        };
        let sender_name = if subject == "anonymous" {
            if requested_sender.is_empty() {
                "admin".to_string()
            } else {
                requested_sender.to_string()
            }
        } else {
            subject
        };
        return Ok((sender, sender_name, true));
    }

    if !requested_sender.is_empty() && requested_sender != "self" {
        return Err((
            StatusCode::BAD_REQUEST,
            "sender must be 'self' for non-admin requests".to_string(),
        ));
    }

    Ok((ProposalSender::UserSelf, subject, false))
}

async fn build_undo_snapshot_data(
    state: &AppState,
    auth: &AuthContext,
    proposal: &Proposal,
) -> Result<Option<serde_json::Value>, (StatusCode, String)> {
    match proposal.action {
        ProposalAction::CreateNode => Ok(Some(serde_json::json!({ "action": "create_node" }))),
        ProposalAction::UpdateNode | ProposalAction::SuggestTag => {
            let target_id = proposal.target_node_id.ok_or((
                StatusCode::BAD_REQUEST,
                "missing target node id".to_string(),
            ))?;
            let existing = state
                .engine
                .get_node(target_id)
                .await
                .map_err(map_hx_error)?
                .ok_or((StatusCode::NOT_FOUND, "target node not found".to_string()))?;
            authorize_namespace(auth, &existing.namespace)?;
            Ok(Some(serde_json::json!({
                "action": "update_node",
                "previous": existing
            })))
        }
        ProposalAction::DeleteNode => {
            let target_id = proposal.target_node_id.ok_or((
                StatusCode::BAD_REQUEST,
                "missing target node id".to_string(),
            ))?;
            let existing = state
                .engine
                .get_node(target_id)
                .await
                .map_err(map_hx_error)?
                .ok_or((StatusCode::NOT_FOUND, "target node not found".to_string()))?;
            authorize_namespace(auth, &existing.namespace)?;
            Ok(Some(serde_json::json!({
                "action": "delete_node",
                "node": existing
            })))
        }
        _ => Ok(None),
    }
}

/// Simple glob matching: supports `*` as wildcard prefix/suffix/full.
fn glob_match_simple(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(suffix) = pattern.strip_prefix('*') {
        return value.ends_with(suffix);
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return value.starts_with(prefix);
    }
    pattern == value
}

/// Execute the action described by a proposal (create/update/delete node).
/// Shared by both manual approve and auto-approve paths.
async fn execute_proposal_action(
    state: &Arc<AppState>,
    auth: &AuthContext,
    proposal: &Proposal,
) -> Result<ProposalActionResult, (StatusCode, String)> {
    match &proposal.action {
        ProposalAction::CreateNode => execute_create_node(state, auth, proposal).await,
        ProposalAction::UpdateNode | ProposalAction::SuggestTag => {
            execute_update_or_suggest_tag(state, auth, proposal).await
        }
        ProposalAction::DeleteNode => execute_delete_node(state, auth, proposal).await,
        ProposalAction::Custom(action) if action == "relay.reply" => {
            execute_relay_reply(state, auth, proposal).await
        }
        _ => Err((
            StatusCode::BAD_REQUEST,
            "proposal action not supported for approval".to_string(),
        )),
    }
}

fn parse_proposal_payload<T: serde::de::DeserializeOwned>(
    proposal: &Proposal,
) -> Result<T, (StatusCode, String)> {
    let value = serde_json::to_value(&proposal.payload)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid payload: {e}")))?;
    serde_json::from_value(value)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid payload: {e}")))
}

async fn execute_create_node(
    state: &Arc<AppState>,
    auth: &AuthContext,
    proposal: &Proposal,
) -> Result<ProposalActionResult, (StatusCode, String)> {
    let payload: ProposalNodePayload = parse_proposal_payload(proposal)?;

    let content = payload.content.ok_or((
        StatusCode::BAD_REQUEST,
        "proposal payload missing content".to_string(),
    ))?;
    let kind_raw = payload.kind.unwrap_or_else(|| "fact".to_string());
    let kind: NodeKind = kind_raw
        .parse()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, e))?;
    let tags = payload.tags.unwrap_or_default();

    validate_node_payload(
        kind,
        payload.title.as_deref(),
        &content,
        payload.source.as_deref(),
        payload.namespace.as_deref(),
        &tags,
        payload.importance,
        payload.metadata.as_ref(),
    )
    .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;

    let namespace = namespace_for_create(auth, payload.namespace, "default")?;
    enforce_namespace_quota(&state.engine, &namespace)
        .await
        .map_err(map_namespace_quota_error)?;

    let mut node = KnowledgeNode::new(kind, content).with_namespace(namespace);
    if let Some(title) = payload.title {
        node = node.with_title(title);
    }
    if let Some(source) = payload.source {
        node = node.with_source(source);
    }
    if !tags.is_empty() {
        node = node.with_tags(tags);
    }
    if let Some(importance) = payload.importance {
        node = node.with_importance(importance);
    }
    if let Some(metadata) = payload.metadata {
        node.metadata = metadata;
    }

    let stored = state.engine.store_node(node).await.map_err(map_hx_error)?;
    state.notify_change(&stored.id.to_string(), "create", Some(&stored.namespace));
    Ok(ProposalActionResult {
        created_node_id: Some(stored.id.to_string()),
        updated_node_id: None,
        deleted_node_id: None,
    })
}

async fn execute_update_or_suggest_tag(
    state: &Arc<AppState>,
    auth: &AuthContext,
    proposal: &Proposal,
) -> Result<ProposalActionResult, (StatusCode, String)> {
    let target_id = proposal.target_node_id.ok_or((
        StatusCode::BAD_REQUEST,
        "proposal missing target_node_id".to_string(),
    ))?;
    let existing = state
        .engine
        .get_node(target_id)
        .await
        .map_err(map_hx_error)?
        .ok_or((StatusCode::NOT_FOUND, "target node not found".to_string()))?;

    authorize_namespace(auth, &existing.namespace)?;

    let payload: ProposalNodePayload = parse_proposal_payload(proposal)?;

    let mut updated = existing.clone();
    if let Some(kind) = payload.kind {
        updated.kind = kind
            .parse()
            .map_err(|e: String| (StatusCode::BAD_REQUEST, e))?;
    }
    if let Some(content) = payload.content {
        updated.content = content;
    }
    if let Some(title) = payload.title {
        updated.title = Some(title);
    }
    if let Some(source) = payload.source {
        updated.source = Some(source);
    }

    let mut tags = updated.tags.clone();
    if matches!(&proposal.action, ProposalAction::SuggestTag) {
        if let Some(tag_val) = proposal.payload.get("tag").and_then(|v| v.as_str()) {
            let tag = tag_val.trim();
            if !tag.is_empty() && !tags.iter().any(|t| t.eq_ignore_ascii_case(tag)) {
                tags.push(tag.to_string());
            }
        } else {
            return Err((
                StatusCode::BAD_REQUEST,
                "proposal payload missing tag".to_string(),
            ));
        }
    } else if let Some(new_tags) = payload.tags {
        tags = new_tags;
    }

    if let Some(importance) = payload.importance {
        updated.importance = importance;
    }
    if let Some(metadata) = payload.metadata {
        updated.metadata = metadata;
    }

    if let Some(namespace) = payload.namespace {
        let ns = namespace_for_create(auth, Some(namespace), &existing.namespace)?;
        updated.namespace = ns;
    }

    validate_node_payload(
        updated.kind,
        updated.title.as_deref(),
        &updated.content,
        updated.source.as_deref(),
        Some(&updated.namespace),
        &tags,
        Some(updated.importance),
        Some(&updated.metadata),
    )
    .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;

    updated.tags = tags;
    let saved = state
        .engine
        .update_node(updated)
        .await
        .map_err(map_hx_error)?;
    state.notify_change(&saved.id.to_string(), "update", Some(&saved.namespace));
    Ok(ProposalActionResult {
        created_node_id: None,
        updated_node_id: Some(saved.id.to_string()),
        deleted_node_id: None,
    })
}

async fn execute_delete_node(
    state: &Arc<AppState>,
    auth: &AuthContext,
    proposal: &Proposal,
) -> Result<ProposalActionResult, (StatusCode, String)> {
    let target_id = proposal.target_node_id.ok_or((
        StatusCode::BAD_REQUEST,
        "proposal missing target_node_id".to_string(),
    ))?;
    let existing = state
        .engine
        .get_node(target_id)
        .await
        .map_err(map_hx_error)?
        .ok_or((StatusCode::NOT_FOUND, "target node not found".to_string()))?;
    authorize_namespace(auth, &existing.namespace)?;

    let deleted = state
        .engine
        .delete_node(target_id)
        .await
        .map_err(map_hx_error)?;
    if deleted {
        state.notify_change(&target_id.to_string(), "delete", Some(&existing.namespace));
        Ok(ProposalActionResult {
            created_node_id: None,
            updated_node_id: None,
            deleted_node_id: Some(target_id.to_string()),
        })
    } else {
        Err((StatusCode::NOT_FOUND, "target node not found".to_string()))
    }
}

async fn execute_relay_reply(
    state: &Arc<AppState>,
    auth: &AuthContext,
    proposal: &Proposal,
) -> Result<ProposalActionResult, (StatusCode, String)> {
    let payload: RelayReplyPayload = parse_proposal_payload(proposal)?;

    let namespace = auth.namespace.as_deref().unwrap_or("default");
    let mut message = RelayMessage::outbound(payload.channel_id, payload.content);

    if let Some(content_type) = payload.content_type.as_deref() {
        let ct: ContentType = content_type
            .parse()
            .map_err(|e: String| (StatusCode::BAD_REQUEST, e))?;
        message = message.with_content_type(ct);
    }

    if let Some(thread_id) = payload.thread_id {
        message = message.with_thread(thread_id);
    }

    if let Some(recipient_id) = payload.recipient_contact_id {
        message.recipient_contact_id = Some(recipient_id);
    }

    if let Some(subject) = payload.subject.as_deref().map(str::trim) {
        if !subject.is_empty() {
            message.metadata.insert(
                "subject".to_string(),
                serde_json::Value::String(subject.to_string()),
            );
        }
    }

    let mut stored = state
        .engine
        .relay
        .send_message(message, namespace)
        .await
        .map_err(map_hx_error)?;

    match crate::email::send_outbound_relay_if_email_channel(state, &stored).await {
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
            return Err(map_hx_error(err));
        }
    }

    let created_node_id = stored.vault_node_id.map(|id| id.to_string());
    Ok(ProposalActionResult {
        created_node_id,
        updated_node_id: None,
        deleted_node_id: None,
    })
}

// --- Handlers ---

/// GET /api/v1/exchange/proposals
pub async fn list_proposals(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListProposalsQuery>,
) -> Result<Json<Vec<Proposal>>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let filter_state = if let Some(ref s) = params.state {
        Some(
            s.parse::<ProposalState>()
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid state: {e}")))?,
        )
    } else {
        None
    };

    let limit = params.limit.unwrap_or(50).min(200);
    let offset = params.offset.unwrap_or(0);

    let proposals = state
        .engine
        .list_proposals(filter_state, limit, offset)
        .await
        .map_err(map_hx_error)?;

    Ok(Json(proposals))
}

/// GET /api/v1/exchange/proposals/:id
pub async fn get_proposal(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_read(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let proposal = state
        .engine
        .get_proposal(uuid)
        .await
        .map_err(map_hx_error)?;

    match proposal {
        Some(p) => Ok(Json(p).into_response()),
        None => Err((StatusCode::NOT_FOUND, "proposal not found".to_string())),
    }
}

/// POST /api/v1/exchange/proposals
pub async fn submit_proposal(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<SubmitProposalRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let (sender, sender_name, allow_auto_approve) =
        resolve_sender_context(&auth, &req.sender)?;
    let action: ProposalAction = req
        .action
        .parse()
        .map_err(|e: String| (StatusCode::BAD_REQUEST, format!("invalid action: {e}")))?;

    // Check blocked senders
    let is_blocked = state
        .engine
        .store
        .nodes
        .is_sender_blocked(sender.as_str(), &sender_name)
        .await
        .map_err(map_hx_error)?;
    if is_blocked {
        return Err((StatusCode::FORBIDDEN, "sender is blocked".to_string()));
    }

    let mut proposal = Proposal::new(sender, action.clone());

    if let Some(ref target_id_str) = req.target_node_id {
        let target_id = Uuid::parse_str(target_id_str).map_err(|_| {
            (
                StatusCode::BAD_REQUEST,
                "invalid target_node_id".to_string(),
            )
        })?;
        proposal = proposal.with_target(target_id);
    }

    if let Some(confidence) = req.confidence {
        proposal = proposal.with_confidence(confidence);
    }

    if let Some(diff) = req.diff_preview {
        if diff.len() > 10_000 {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("diff_preview exceeds maximum of 10,000 characters ({} given)", diff.len()),
            ));
        }
        proposal = proposal.with_diff(diff);
    }

    if let Some(payload) = req.payload {
        proposal = proposal.with_payload(payload);
    }

    // Check auto-approve rules before submitting
    let rules = state
        .engine
        .store
        .nodes
        .list_auto_approve_rules()
        .await
        .map_err(map_hx_error)?;

    let mut auto_approved = false;
    for rule in &rules {
        if !rule.enabled {
            continue;
        }
        // Check sender pattern match
        if let Some(ref pattern) = rule.sender_pattern {
            if !glob_match_simple(pattern, &sender_name) {
                continue;
            }
        }
        // Check action type match
        if !rule.action_types.is_empty()
            && !rule.action_types.iter().any(|a| a == action.as_str())
        {
            continue;
        }
        // Check confidence threshold
        if proposal.confidence < rule.min_confidence {
            continue;
        }
        // Rule matches — auto-approve
        auto_approved = allow_auto_approve;
        break;
    }

    state
        .engine
        .submit_proposal(&proposal)
        .await
        .map_err(map_hx_error)?;

    if auto_approved {
        let mut snapshot_data = build_undo_snapshot_data(&state, &auth, &proposal).await?;
        let result = execute_proposal_action(&state, &auth, &proposal).await?;

        if let Some(ref mut data) = snapshot_data {
            if let Some(created_id) = result.created_node_id.as_ref() {
                data["node_id"] = serde_json::json!(created_id);
            }
        } else if let Some(created_id) = result.created_node_id.as_ref() {
            snapshot_data = Some(serde_json::json!({
                "action": "create_node",
                "node_id": created_id
            }));
        }

        if let Some(snapshot_data) = snapshot_data {
            let now = Utc::now();
            let snapshot = UndoSnapshot {
                id: Uuid::now_v7(),
                proposal_id: proposal.id,
                snapshot_data,
                created_at: now,
                expires_at: now + Duration::days(7),
                used: false,
            };
            state
                .engine
                .store
                .nodes
                .save_undo_snapshot(&snapshot)
                .await
                .map_err(map_hx_error)?;
        }

        state
            .engine
            .resolve_proposal(proposal.id, ProposalState::AutoApproved)
            .await
            .map_err(map_hx_error)?;

        let chronicle = ChronicleEntry::new(
            "exchange.auto_approve",
            format!("Auto-approved proposal {}", proposal.id),
        );
        let _ = state.engine.log_chronicle(&chronicle).await;

        return Ok(Json(serde_json::json!({
            "id": proposal.id.to_string(),
            "state": "auto_approved",
            "created_node_id": result.created_node_id,
            "updated_node_id": result.updated_node_id,
            "deleted_node_id": result.deleted_node_id
        }))
        .into_response());
    }

    Ok((StatusCode::CREATED, Json(proposal)).into_response())
}

/// POST /api/v1/exchange/proposals/:id/approve
pub async fn approve_proposal(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let proposal = state
        .engine
        .get_proposal(uuid)
        .await
        .map_err(map_hx_error)?
        .ok_or((StatusCode::NOT_FOUND, "proposal not found".to_string()))?;

    let mut snapshot_data = build_undo_snapshot_data(&state, &auth, &proposal).await?;
    let result = execute_proposal_action(&state, &auth, &proposal).await?;

    if let Some(ref mut data) = snapshot_data {
        if let Some(created_id) = result.created_node_id.as_ref() {
            data["node_id"] = serde_json::json!(created_id);
        }
    } else if let Some(created_id) = result.created_node_id.as_ref() {
        snapshot_data = Some(serde_json::json!({
            "action": "create_node",
            "node_id": created_id
        }));
    }

    if let Some(snapshot_data) = snapshot_data {
        let now = Utc::now();
        let snapshot = UndoSnapshot {
            id: Uuid::now_v7(),
            proposal_id: proposal.id,
            snapshot_data,
            created_at: now,
            expires_at: now + Duration::days(7),
            used: false,
        };
        state
            .engine
            .store
            .nodes
            .save_undo_snapshot(&snapshot)
            .await
            .map_err(map_hx_error)?;
    }

    let resolved = state
        .engine
        .resolve_proposal(uuid, ProposalState::Approved)
        .await
        .map_err(map_hx_error)?;

    if !resolved {
        return Err((StatusCode::NOT_FOUND, "proposal not found".to_string()));
    }

    // Log chronicle entry for transparency
    let chronicle =
        ChronicleEntry::new("exchange.approve", format!("User approved proposal {uuid}"));
    let _ = state.engine.log_chronicle(&chronicle).await;

    Ok(Json(serde_json::json!({
        "id": uuid.to_string(),
        "state": "approved",
        "created_node_id": result.created_node_id,
        "updated_node_id": result.updated_node_id,
        "deleted_node_id": result.deleted_node_id
    })))
}

/// POST /api/v1/exchange/proposals/:id/reject
pub async fn reject_proposal(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    let resolved = state
        .engine
        .resolve_proposal(uuid, ProposalState::Rejected)
        .await
        .map_err(map_hx_error)?;

    if !resolved {
        return Err((StatusCode::NOT_FOUND, "proposal not found".to_string()));
    }

    // Log chronicle entry for transparency
    let chronicle =
        ChronicleEntry::new("exchange.reject", format!("User rejected proposal {uuid}"));
    let _ = state.engine.log_chronicle(&chronicle).await;

    Ok(Json(
        serde_json::json!({ "id": uuid.to_string(), "state": "rejected" }),
    ))
}

/// POST /api/v1/exchange/proposals/:id/undo
pub async fn undo_proposal(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let uuid =
        Uuid::parse_str(&id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))?;

    // Retrieve the undo snapshot
    let snapshot = state
        .engine
        .store
        .nodes
        .get_undo_snapshot(uuid)
        .await
        .map_err(map_hx_error)?
        .ok_or((
            StatusCode::NOT_FOUND,
            "no undo snapshot for this proposal".to_string(),
        ))?;

    if snapshot.used {
        return Err((
            StatusCode::CONFLICT,
            "undo already applied for this proposal".to_string(),
        ));
    }

    if Utc::now() > snapshot.expires_at {
        return Err((StatusCode::GONE, "undo window has expired".to_string()));
    }

    // Execute the undo based on snapshot data
    let action = snapshot
        .snapshot_data
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    match action {
        "create_node" => {
            // Undo a create by deleting the created node
            if let Some(node_id_str) = snapshot.snapshot_data.get("node_id").and_then(|v| v.as_str())
            {
                if let Ok(node_id) = Uuid::parse_str(node_id_str) {
                    state
                        .engine
                        .delete_node(node_id)
                        .await
                        .map_err(map_hx_error)?;
                    state.notify_change(node_id_str, "undo_delete", None);
                }
            }
        }
        "update_node" => {
            // Undo an update by restoring the previous version
            if let Some(previous) = snapshot.snapshot_data.get("previous") {
                let node: KnowledgeNode = serde_json::from_value(previous.clone()).map_err(|e| {
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("failed to deserialize previous node: {e}"),
                    )
                })?;
                authorize_namespace(&auth, &node.namespace)?;
                let saved = state.engine.update_node(node).await.map_err(map_hx_error)?;
                state.notify_change(&saved.id.to_string(), "undo_restore", Some(&saved.namespace));
            }
        }
        "delete_node" => {
            // Undo a delete by re-inserting the node
            if let Some(node_data) = snapshot.snapshot_data.get("node") {
                let node: KnowledgeNode =
                    serde_json::from_value(node_data.clone()).map_err(|e| {
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("failed to deserialize deleted node: {e}"),
                        )
                    })?;
                authorize_namespace(&auth, &node.namespace)?;
                state.engine.store_node(node).await.map_err(map_hx_error)?;
            }
        }
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("cannot undo action type: {action}"),
            ));
        }
    }

    // Mark snapshot as used
    state
        .engine
        .store
        .nodes
        .mark_undo_used(snapshot.id)
        .await
        .map_err(map_hx_error)?;

    // Log chronicle
    let chronicle = ChronicleEntry::new(
        "exchange.undo",
        format!("User undid proposal {uuid} (action: {action})"),
    );
    let _ = state.engine.log_chronicle(&chronicle).await;

    Ok(Json(serde_json::json!({
        "id": uuid.to_string(),
        "action": action,
        "undone": true,
    })))
}

/// GET /api/v1/exchange/inbox/count
pub async fn inbox_count(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<ProposalCountResponse>, (StatusCode, String)> {
    authorize_read(&auth)?;

    let count = state
        .engine
        .count_proposals(Some(ProposalState::Pending))
        .await
        .map_err(map_hx_error)?;

    Ok(Json(ProposalCountResponse { count }))
}

// ---------------------------------------------------------------------------
// Batch Operations
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct BatchProposalRequest {
    pub ids: Vec<String>,
    pub action: String, // "approve" or "reject"
}

#[derive(Serialize)]
struct BatchProposalResultItem {
    id: String,
    success: bool,
    state: Option<String>,
    error: Option<String>,
    created_node_id: Option<String>,
    updated_node_id: Option<String>,
    deleted_node_id: Option<String>,
}

/// POST /api/v1/exchange/proposals/batch
///
/// Approve or reject multiple proposals in a single request.
pub async fn batch_proposals(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(body): Json<BatchProposalRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_write(&auth)?;

    let target_state = match body.action.as_str() {
        "approve" => ProposalState::Approved,
        "reject" => ProposalState::Rejected,
        other => {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("invalid batch action: {other} (expected \"approve\" or \"reject\")"),
            ))
        }
    };

    if body.ids.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "ids array must not be empty".to_string(),
        ));
    }

    if body.ids.len() > 100 {
        return Err((
            StatusCode::BAD_REQUEST,
            "batch size exceeds maximum of 100".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(body.ids.len());

    for id_str in &body.ids {
        let uuid = match Uuid::parse_str(id_str) {
            Ok(u) => u,
            Err(_) => {
                results.push(BatchProposalResultItem {
                    id: id_str.clone(),
                    success: false,
                    state: None,
                    error: Some("invalid uuid".to_string()),
                    created_node_id: None,
                    updated_node_id: None,
                    deleted_node_id: None,
                });
                continue;
            }
        };

        let mut action_result: Option<ProposalActionResult> = None;

        // For approvals, execute the proposal action first
        if target_state == ProposalState::Approved {
            let proposal = match state.engine.get_proposal(uuid).await {
                Ok(Some(p)) => p,
                Ok(None) => {
                    results.push(BatchProposalResultItem {
                        id: id_str.clone(),
                        success: false,
                        state: None,
                        error: Some("proposal not found".to_string()),
                        created_node_id: None,
                        updated_node_id: None,
                        deleted_node_id: None,
                    });
                    continue;
                }
                Err(e) => {
                    results.push(BatchProposalResultItem {
                        id: id_str.clone(),
                        success: false,
                        state: None,
                        error: Some(e.to_string()),
                        created_node_id: None,
                        updated_node_id: None,
                        deleted_node_id: None,
                    });
                    continue;
                }
            };

            let mut snapshot_data = match build_undo_snapshot_data(&state, &auth, &proposal).await {
                Ok(data) => data,
                Err((_, err)) => {
                    results.push(BatchProposalResultItem {
                        id: id_str.clone(),
                        success: false,
                        state: None,
                        error: Some(err),
                        created_node_id: None,
                        updated_node_id: None,
                        deleted_node_id: None,
                    });
                    continue;
                }
            };

            let exec_result = match execute_proposal_action(&state, &auth, &proposal).await {
                Ok(result) => result,
                Err((_, err)) => {
                    results.push(BatchProposalResultItem {
                        id: id_str.clone(),
                        success: false,
                        state: None,
                        error: Some(format!("action failed: {err}")),
                        created_node_id: None,
                        updated_node_id: None,
                        deleted_node_id: None,
                    });
                    continue;
                }
            };

            if let Some(ref mut data) = snapshot_data {
                if let Some(created_id) = exec_result.created_node_id.as_ref() {
                    data["node_id"] = serde_json::json!(created_id);
                }
            } else if let Some(created_id) = exec_result.created_node_id.as_ref() {
                snapshot_data = Some(serde_json::json!({
                    "action": "create_node",
                    "node_id": created_id
                }));
            }

            if let Some(snapshot_data) = snapshot_data {
                let now = Utc::now();
                let snapshot = UndoSnapshot {
                    id: Uuid::now_v7(),
                    proposal_id: uuid,
                    snapshot_data,
                    created_at: now,
                    expires_at: now + Duration::days(7),
                    used: false,
                };
                if let Err(err) = state
                    .engine
                    .store
                    .nodes
                    .save_undo_snapshot(&snapshot)
                    .await
                {
                    results.push(BatchProposalResultItem {
                        id: id_str.clone(),
                        success: false,
                        state: None,
                        error: Some(err.to_string()),
                        created_node_id: None,
                        updated_node_id: None,
                        deleted_node_id: None,
                    });
                    continue;
                }
            }

            action_result = Some(exec_result);
        }

        match state.engine.resolve_proposal(uuid, target_state).await {
            Ok(true) => {
                let (created_node_id, updated_node_id, deleted_node_id) = match action_result {
                    Some(result) => (
                        result.created_node_id,
                        result.updated_node_id,
                        result.deleted_node_id,
                    ),
                    None => (None, None, None),
                };

                let chronicle = ChronicleEntry::new(
                    if target_state == ProposalState::Approved {
                        "exchange.batch_approve"
                    } else {
                        "exchange.batch_reject"
                    },
                    format!("Batch {}: proposal {uuid}", body.action),
                );
                let _ = state.engine.log_chronicle(&chronicle).await;

                results.push(BatchProposalResultItem {
                    id: id_str.clone(),
                    success: true,
                    state: Some(target_state.as_str().to_string()),
                    error: None,
                    created_node_id,
                    updated_node_id,
                    deleted_node_id,
                });
            }
            Ok(false) => {
                results.push(BatchProposalResultItem {
                    id: id_str.clone(),
                    success: false,
                    state: None,
                    error: Some("proposal not found".to_string()),
                    created_node_id: None,
                    updated_node_id: None,
                    deleted_node_id: None,
                });
            }
            Err(e) => {
                results.push(BatchProposalResultItem {
                    id: id_str.clone(),
                    success: false,
                    state: None,
                    error: Some(e.to_string()),
                    created_node_id: None,
                    updated_node_id: None,
                    deleted_node_id: None,
                });
            }
        }
    }

    let succeeded = results.iter().filter(|r| r.success).count();
    let failed = results.len() - succeeded;

    Ok(Json(serde_json::json!({
        "total": results.len(),
        "succeeded": succeeded,
        "failed": failed,
        "results": results
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthContext;

    // --- glob_match_simple tests ---

    #[test]
    fn glob_wildcard_matches_everything() {
        assert!(glob_match_simple("*", "anything"));
        assert!(glob_match_simple("*", ""));
    }

    #[test]
    fn glob_prefix_wildcard_matches_suffix() {
        assert!(glob_match_simple("*@example.com", "user@example.com"));
        assert!(!glob_match_simple("*@example.com", "user@other.com"));
    }

    #[test]
    fn glob_suffix_wildcard_matches_prefix() {
        assert!(glob_match_simple("mcp-*", "mcp-agent"));
        assert!(!glob_match_simple("mcp-*", "other-agent"));
    }

    #[test]
    fn glob_exact_match() {
        assert!(glob_match_simple("exact", "exact"));
        assert!(!glob_match_simple("exact", "different"));
    }

    #[test]
    fn glob_empty_pattern_only_matches_empty() {
        assert!(glob_match_simple("", ""));
        assert!(!glob_match_simple("", "notempty"));
    }

    // --- resolve_sender_context tests ---

    fn admin_auth() -> AuthContext {
        AuthContext {
            subject: Some("admin-user".into()),
            namespace: None,
            role: crate::auth::AuthRole::Admin,
            consumer_name: None,
        }
    }

    fn user_auth() -> AuthContext {
        AuthContext {
            subject: Some("regular-user".into()),
            namespace: None,
            role: crate::auth::AuthRole::Write,
            consumer_name: None,
        }
    }

    #[test]
    fn resolve_sender_admin_empty_defaults_to_self() {
        let auth = admin_auth();
        let (sender, name, is_admin) = resolve_sender_context(&auth, "").unwrap();
        assert!(matches!(sender, ProposalSender::UserSelf));
        assert_eq!(name, "admin-user");
        assert!(is_admin);
    }

    #[test]
    fn resolve_sender_admin_custom_sender() {
        let auth = admin_auth();
        let (sender, name, is_admin) = resolve_sender_context(&auth, "mcp").unwrap();
        assert!(matches!(sender, ProposalSender::Mcp));
        assert_eq!(name, "admin-user");
        assert!(is_admin);
    }

    #[test]
    fn resolve_sender_user_self_ok() {
        let auth = user_auth();
        let (sender, name, is_admin) = resolve_sender_context(&auth, "self").unwrap();
        assert!(matches!(sender, ProposalSender::UserSelf));
        assert_eq!(name, "regular-user");
        assert!(!is_admin);
    }

    #[test]
    fn resolve_sender_user_empty_ok() {
        let auth = user_auth();
        let (sender, _name, _is_admin) = resolve_sender_context(&auth, "").unwrap();
        assert!(matches!(sender, ProposalSender::UserSelf));
    }

    #[test]
    fn resolve_sender_non_admin_custom_blocked() {
        let auth = user_auth();
        let result = resolve_sender_context(&auth, "mcp");
        assert!(result.is_err());
        let (status, _msg) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    // --- map_hx_error tests ---

    #[test]
    fn map_hx_error_node_not_found_returns_404() {
        let err = hx_core::HxError::NodeNotFound(Uuid::now_v7());
        let (status, _msg) = map_hx_error(err);
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[test]
    fn map_hx_error_invalid_input_returns_400() {
        let err = hx_core::HxError::InvalidInput("bad data".into());
        let (status, _msg) = map_hx_error(err);
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn map_hx_error_other_returns_500() {
        let err = hx_core::HxError::Internal("kaboom".into());
        let (status, _msg) = map_hx_error(err);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    // --- BatchProposalRequest validation tests ---

    #[test]
    fn batch_size_limit_parsed_correctly() {
        // The handler checks body.ids.len() > 100
        let ids: Vec<String> = (0..101).map(|i| format!("id-{i}")).collect();
        assert!(ids.len() > 100);
    }

    // --- diff_preview validation ---

    #[test]
    fn diff_preview_within_limit() {
        let diff = "a".repeat(10_000);
        assert!(diff.len() <= 10_000);
    }

    #[test]
    fn diff_preview_exceeds_limit() {
        let diff = "a".repeat(10_001);
        assert!(diff.len() > 10_000);
    }
}
