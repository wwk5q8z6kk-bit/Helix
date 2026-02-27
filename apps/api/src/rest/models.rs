use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};

use hx_engine::model_manager::ModelManager;

use crate::state::AppState;

/// GET /api/v1/models — List all local GGUF models.
pub async fn list_models(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mgr = ModelManager::new(&state.engine.config.local_llm);
    match mgr.list_models() {
        Ok(models) => Json(serde_json::json!({
            "models": models,
            "total_size": mgr.total_size().unwrap_or(0),
            "models_dir": state.engine.config.local_llm.models_dir,
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

/// POST /api/v1/models/download — Download a model from Hugging Face Hub.
///
/// Request body: `{"model_id": "org/repo/file.gguf"}`
pub async fn download_model(
    State(state): State<Arc<AppState>>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let model_id = match body.get("model_id").and_then(|v| v.as_str()) {
        Some(id) => id.to_string(),
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": "missing 'model_id' field"})),
            )
                .into_response()
        }
    };

    let mgr = ModelManager::new(&state.engine.config.local_llm);
    let status = mgr.download_model(&model_id).await;

    let code = match status.status.as_str() {
        "completed" | "exists" => StatusCode::OK,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };

    (code, Json(serde_json::json!(status))).into_response()
}

/// DELETE /api/v1/models/{filename} — Delete a local model.
pub async fn delete_model(
    State(state): State<Arc<AppState>>,
    Path(filename): Path<String>,
) -> impl IntoResponse {
    let mgr = ModelManager::new(&state.engine.config.local_llm);
    match mgr.delete_model(&filename) {
        Ok(()) => Json(serde_json::json!({"deleted": filename})).into_response(),
        Err(e) => {
            let code = if e.contains("not found") {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (code, Json(serde_json::json!({"error": e}))).into_response()
        }
    }
}

/// GET /api/v1/models/status — Model subsystem status.
pub async fn model_status(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let config = &state.engine.config.local_llm;
    let mgr = ModelManager::new(config);
    let models = mgr.list_models().unwrap_or_default();
    let available_ram = ModelManager::check_available_ram();

    Json(serde_json::json!({
        "enabled": config.enabled,
        "models_dir": config.models_dir,
        "model_count": models.len(),
        "total_size_bytes": mgr.total_size().unwrap_or(0),
        "configured_model_path": config.model_path,
        "configured_model_id": config.model_id,
        "max_ram_bytes": config.max_ram_bytes,
        "available_ram_bytes": available_ram,
        "gpu_layers": config.gpu_layers,
        "context_size": config.context_size,
        "local_llm_feature_compiled": hx_engine::llm_local::is_feature_enabled(),
    }))
}
