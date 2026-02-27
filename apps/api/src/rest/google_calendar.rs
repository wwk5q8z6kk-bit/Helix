use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::Serialize;

use hx_core::AdapterPollStore;

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;

#[derive(Serialize)]
pub struct GoogleCalendarStatusResponse {
    pub enabled: bool,
    pub calendar_id: String,
    pub namespace: String,
    pub import_events: bool,
    pub export_events: bool,
    pub last_sync_at: Option<String>,
    pub last_cursor: Option<String>,
    pub messages_received: u64,
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

/// GET /api/v1/calendar/google/status
pub async fn google_calendar_status(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_read(&auth) {
        return (status, err_json(message)).into_response();
    }

    let config = &state.engine.config.google_calendar;
    let adapter_name = format!("google-calendar:{}", config.calendar_id);
    let poll_state = state
        .engine
        .store
        .nodes
        .get_poll_state(&adapter_name)
        .await
        .ok()
        .flatten();

    Json(GoogleCalendarStatusResponse {
        enabled: config.enabled,
        calendar_id: config.calendar_id.clone(),
        namespace: config.namespace.clone(),
        import_events: config.import_events,
        export_events: config.export_events,
        last_sync_at: poll_state.as_ref().map(|s| s.last_poll_at.clone()),
        last_cursor: poll_state.as_ref().map(|s| s.cursor.clone()),
        messages_received: poll_state.map(|s| s.messages_received).unwrap_or(0),
    })
    .into_response()
}

/// POST /api/v1/calendar/google/sync
pub async fn google_calendar_sync(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }

    match state.engine.sync_google_calendar().await {
        Ok(report) => Json(report).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("google calendar sync failed: {err}")),
        )
            .into_response(),
    }
}
