use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::traits::ConversationStore;

use crate::state::AppState;

// ---------------------------------------------------------------------------
// Request / Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct CreateConversationRequest {
    pub title: Option<String>,
}

#[derive(Serialize)]
pub struct ConversationResponse {
    pub id: String,
    pub title: Option<String>,
}

#[derive(Deserialize)]
pub struct SendMessageRequest {
    pub role: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct MessageResponse {
    pub id: String,
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Deserialize)]
pub struct ListParams {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    50
}

#[derive(Serialize)]
pub struct ConversationListItem {
    pub id: String,
    pub title: Option<String>,
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// POST /api/v1/conversations
pub async fn create_conversation(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateConversationRequest>,
) -> impl IntoResponse {
    let id = Uuid::now_v7();
    match state
        .engine
        .store
        .nodes
        .create_conversation(id, req.title.as_deref())
        .await
    {
        Ok(()) => (
            StatusCode::CREATED,
            Json(ConversationResponse {
                id: id.to_string(),
                title: req.title,
            }),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// GET /api/v1/conversations
pub async fn list_conversations(
    State(state): State<Arc<AppState>>,
    Query(params): Query<ListParams>,
) -> impl IntoResponse {
    match state
        .engine
        .store
        .nodes
        .list_conversations(params.limit, params.offset)
        .await
    {
        Ok(convs) => {
            let items: Vec<ConversationListItem> = convs
                .into_iter()
                .map(|(id, title, ts)| ConversationListItem {
                    id: id.to_string(),
                    title,
                    updated_at: ts.to_rfc3339(),
                })
                .collect();
            Json(items).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// POST /api/v1/conversations/:id/message
pub async fn send_message(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<SendMessageRequest>,
) -> impl IntoResponse {
    let conv_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid conversation id"})),
            )
                .into_response()
        }
    };

    match state
        .engine
        .store
        .nodes
        .add_message(conv_id, &req.role, &req.content)
        .await
    {
        Ok(msg_id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "id": msg_id.to_string(),
                "conversation_id": id,
                "role": req.role,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// GET /api/v1/conversations/:id/messages
pub async fn get_messages(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Query(params): Query<ListParams>,
) -> impl IntoResponse {
    let conv_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid conversation id"})),
            )
                .into_response()
        }
    };

    match state
        .engine
        .store
        .nodes
        .get_messages(conv_id, params.limit)
        .await
    {
        Ok(messages) => {
            let items: Vec<MessageResponse> = messages
                .into_iter()
                .map(|(id, role, content, ts)| MessageResponse {
                    id: id.to_string(),
                    role,
                    content,
                    created_at: ts.to_rfc3339(),
                })
                .collect();
            Json(items).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/conversations/:id
pub async fn delete_conversation(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    let conv_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "invalid conversation id"})),
            )
                .into_response()
        }
    };

    match state.engine.store.nodes.delete_conversation(conv_id).await {
        Ok(true) => StatusCode::NO_CONTENT.into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": "conversation not found"})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string()})),
        )
            .into_response(),
    }
}
