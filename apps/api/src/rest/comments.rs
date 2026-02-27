use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{NodeComment, NodeStore};

use crate::auth::{authorize_namespace, authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct CreateCommentRequest {
    pub body: String,
    pub author: Option<String>,
}

#[derive(Deserialize)]
pub struct ListCommentsParams {
    pub include_resolved: Option<bool>,
}

#[derive(Serialize)]
pub struct CommentResponse {
    pub id: String,
    pub node_id: String,
    pub author: Option<String>,
    pub body: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
}

#[derive(Serialize)]
struct ErrorBody {
    error: String,
}

fn err_json(msg: impl ToString) -> Json<ErrorBody> {
    Json(ErrorBody {
        error: msg.to_string(),
    })
}

fn comment_to_response(comment: NodeComment) -> CommentResponse {
    CommentResponse {
        id: comment.id.to_string(),
        node_id: comment.node_id.to_string(),
        author: comment.author,
        body: comment.body,
        created_at: comment.created_at,
        updated_at: comment.updated_at,
        resolved_at: comment.resolved_at,
    }
}

/// POST /api/v1/nodes/:id/comments
pub async fn create_node_comment(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(node_id): Path<String>,
    Json(req): Json<CreateCommentRequest>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }

    let node_id = match Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid node id"),
            )
                .into_response()
        }
    };

    let node = match state.engine.store.nodes.get(node_id).await {
        Ok(Some(node)) => node,
        Ok(None) => return (StatusCode::NOT_FOUND, err_json("node not found")).into_response(),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load node: {err}")),
            )
                .into_response()
        }
    };

    if let Err((status, message)) = authorize_namespace(&auth, &node.namespace) {
        return (status, err_json(message)).into_response();
    }

    if req.body.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            err_json("comment body cannot be empty"),
        )
            .into_response();
    }

    let author = req.author.or_else(|| auth.subject.clone());
    match state
        .engine
        .create_node_comment(node_id, author, req.body)
        .await
    {
        Ok(comment) => Json(comment_to_response(comment)).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to create comment: {err}")),
        )
            .into_response(),
    }
}

/// GET /api/v1/nodes/:id/comments
pub async fn list_node_comments(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(node_id): Path<String>,
    Query(params): Query<ListCommentsParams>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_read(&auth) {
        return (status, err_json(message)).into_response();
    }

    let node_id = match Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid node id"),
            )
                .into_response()
        }
    };

    let node = match state.engine.store.nodes.get(node_id).await {
        Ok(Some(node)) => node,
        Ok(None) => return (StatusCode::NOT_FOUND, err_json("node not found")).into_response(),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load node: {err}")),
            )
                .into_response()
        }
    };

    if let Err((status, message)) = authorize_namespace(&auth, &node.namespace) {
        return (status, err_json(message)).into_response();
    }

    let include_resolved = params.include_resolved.unwrap_or(false);
    match state
        .engine
        .list_node_comments(node_id, include_resolved)
        .await
    {
        Ok(comments) => Json(
            comments
                .into_iter()
                .map(comment_to_response)
                .collect::<Vec<_>>(),
        )
        .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to list comments: {err}")),
        )
            .into_response(),
    }
}

/// PUT /api/v1/nodes/:id/comments/:comment_id/resolve
pub async fn resolve_node_comment(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path((node_id, comment_id)): Path<(String, String)>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }

    let node_id = match Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid node id"),
            )
                .into_response()
        }
    };
    let comment_id = match Uuid::parse_str(&comment_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid comment id"),
            )
                .into_response()
        }
    };

    let node = match state.engine.store.nodes.get(node_id).await {
        Ok(Some(node)) => node,
        Ok(None) => return (StatusCode::NOT_FOUND, err_json("node not found")).into_response(),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load node: {err}")),
            )
                .into_response()
        }
    };

    if let Err((status, message)) = authorize_namespace(&auth, &node.namespace) {
        return (status, err_json(message)).into_response();
    }

    let comment = match state.engine.get_node_comment(comment_id).await {
        Ok(Some(comment)) => comment,
        Ok(None) => return (StatusCode::NOT_FOUND, err_json("comment not found")).into_response(),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load comment: {err}")),
            )
                .into_response()
        }
    };

    if comment.node_id != node_id {
        return (StatusCode::NOT_FOUND, err_json("comment not found")).into_response();
    }

    match state.engine.resolve_node_comment(comment_id).await {
        Ok(true) => match state.engine.get_node_comment(comment_id).await {
            Ok(Some(updated)) => Json(comment_to_response(updated)).into_response(),
            Ok(None) => (StatusCode::NOT_FOUND, err_json("comment not found")).into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load comment: {err}")),
            )
                .into_response(),
        },
        Ok(false) => (StatusCode::NOT_FOUND, err_json("comment not found")).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to resolve comment: {err}")),
        )
            .into_response(),
    }
}

/// DELETE /api/v1/nodes/:id/comments/:comment_id
pub async fn delete_node_comment(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path((node_id, comment_id)): Path<(String, String)>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }

    let node_id = match Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid node id"),
            )
                .into_response()
        }
    };
    let comment_id = match Uuid::parse_str(&comment_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid comment id"),
            )
                .into_response()
        }
    };

    let node = match state.engine.store.nodes.get(node_id).await {
        Ok(Some(node)) => node,
        Ok(None) => return (StatusCode::NOT_FOUND, err_json("node not found")).into_response(),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load node: {err}")),
            )
                .into_response()
        }
    };

    if let Err((status, message)) = authorize_namespace(&auth, &node.namespace) {
        return (status, err_json(message)).into_response();
    }

    let comment = match state.engine.get_node_comment(comment_id).await {
        Ok(Some(comment)) => comment,
        Ok(None) => return (StatusCode::NOT_FOUND, err_json("comment not found")).into_response(),
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load comment: {err}")),
            )
                .into_response()
        }
    };

    if comment.node_id != node_id {
        return (StatusCode::NOT_FOUND, err_json("comment not found")).into_response();
    }

    match state.engine.delete_node_comment(comment_id).await {
        Ok(true) => Json(comment_to_response(comment)).into_response(),
        Ok(false) => (StatusCode::NOT_FOUND, err_json("comment not found")).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to delete comment: {err}")),
        )
            .into_response(),
    }
}
