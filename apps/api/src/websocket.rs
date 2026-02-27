use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    http::HeaderMap,
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use futures::{SinkExt, StreamExt};

use crate::auth::{auth_context_from_headers_with_state, authorize_read};
use crate::limits::enforce_rate_limit;
use crate::state::AppState;

pub fn ws_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/ws/changes", get(ws_handler))
        .route("/ws/reminders", get(ws_reminders_handler))
        .route("/ws/agent", get(ws_agent_handler))
        .with_state(state)
}

async fn ws_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Response {
    let auth = match auth_context_from_headers_with_state(&headers, &state).await {
        Ok(auth) => auth,
        Err(status) => return status.into_response(),
    };

    if let Err((status, _message)) = authorize_read(&auth) {
        return status.into_response();
    }

    if enforce_rate_limit(&auth).is_err() {
        return axum::http::StatusCode::TOO_MANY_REQUESTS.into_response();
    }

    ws.on_upgrade(move |socket| handle_socket(socket, state, auth))
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>, auth: crate::auth::AuthContext) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.change_tx.subscribe();

    // Spawn task to forward change notifications to WebSocket
    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(notification) => {
                    if let Some(ref namespace_scope) = auth.namespace {
                        if notification.namespace.as_deref() != Some(namespace_scope.as_str()) {
                            continue;
                        }
                    }

                    let json = serde_json::json!({
                        "type": "change",
                        "node_id": notification.node_id,
                        "operation": notification.operation,
                        "timestamp": notification.timestamp,
                        "namespace": notification.namespace,
                    });
                    if sender.send(Message::Text(json.to_string())).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("WebSocket client lagged by {n} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Receive pings/close from client
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Close(_) => break,
            Message::Ping(data) => {
                // Pong is handled automatically by axum
                let _ = data;
            }
            _ => {}
        }
    }

    send_task.abort();
}

async fn ws_reminders_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Response {
    let auth = match auth_context_from_headers_with_state(&headers, &state).await {
        Ok(auth) => auth,
        Err(status) => return status.into_response(),
    };

    if let Err((status, _message)) = authorize_read(&auth) {
        return status.into_response();
    }

    if enforce_rate_limit(&auth).is_err() {
        return axum::http::StatusCode::TOO_MANY_REQUESTS.into_response();
    }

    ws.on_upgrade(move |socket| handle_reminders_socket(socket, state, auth))
}

async fn ws_agent_handler(
    ws: WebSocketUpgrade,
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Response {
    let auth = match auth_context_from_headers_with_state(&headers, &state).await {
        Ok(auth) => auth,
        Err(status) => return status.into_response(),
    };

    if let Err((status, _message)) = authorize_read(&auth) {
        return status.into_response();
    }

    if enforce_rate_limit(&auth).is_err() {
        return axum::http::StatusCode::TOO_MANY_REQUESTS.into_response();
    }

    ws.on_upgrade(move |socket| handle_agent_socket(socket, state, auth))
}

async fn handle_agent_socket(
    socket: WebSocket,
    state: Arc<AppState>,
    auth: crate::auth::AuthContext,
) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.agent_tx.subscribe();

    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(notification) => {
                    if let Some(ref namespace_scope) = auth.namespace {
                        if notification.namespace() != Some(namespace_scope.as_str()) {
                            continue;
                        }
                    }

                    let payload = match serde_json::to_string(&notification) {
                        Ok(payload) => payload,
                        Err(err) => {
                            tracing::warn!(error = %err, "failed to serialize agent notification");
                            continue;
                        }
                    };

                    if sender.send(Message::Text(payload)).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("WebSocket agent client lagged by {n} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Close(_) => break,
            Message::Ping(data) => {
                let _ = data;
            }
            _ => {}
        }
    }

    send_task.abort();
}

async fn handle_reminders_socket(
    socket: WebSocket,
    state: Arc<AppState>,
    auth: crate::auth::AuthContext,
) {
    let (mut sender, mut receiver) = socket.split();
    let mut rx = state.reminder_tx.subscribe();

    // Spawn task to forward reminder notifications to WebSocket
    let send_task = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(notification) => {
                    // Filter by namespace if user is scoped
                    if let Some(ref namespace_scope) = auth.namespace {
                        if notification.namespace.as_deref() != Some(namespace_scope.as_str()) {
                            continue;
                        }
                    }

                    let json = serde_json::json!({
                        "type": "reminder",
                        "node_id": notification.node_id,
                        "title": notification.title,
                        "content_preview": notification.content_preview,
                        "due_at": notification.due_at,
                        "namespace": notification.namespace,
                        "timestamp": notification.timestamp.to_rfc3339(),
                    });
                    if sender.send(Message::Text(json.to_string())).await.is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("WebSocket reminders client lagged by {n} events");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            }
        }
    });

    // Receive pings/close from client
    while let Some(Ok(msg)) = receiver.next().await {
        match msg {
            Message::Close(_) => break,
            Message::Ping(data) => {
                let _ = data;
            }
            _ => {}
        }
    }

    send_task.abort();
}
