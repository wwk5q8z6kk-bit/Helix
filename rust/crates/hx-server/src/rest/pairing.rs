//! REST handlers for gateway pairing — initiate and confirm flows.

use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::auth::AuthContext;
use crate::pairing::{confirm_pairing, create_pairing_session};
use crate::state::AppState;

// --- DTOs ---

#[derive(Deserialize)]
pub struct InitiatePairingDto {
    pub channel_name: String,
}

#[derive(Debug, Serialize)]
pub struct InitiatePairingResponse {
    pub session_id: String,
    pub otp: String,
}

#[derive(Deserialize)]
pub struct ConfirmPairingDto {
    pub session_id: String,
    pub otp: String,
}

#[derive(Debug, Serialize)]
pub struct ConfirmPairingResponse {
    pub bearer_token: String,
}

// --- Handlers ---

/// POST /api/v1/pair/initiate — requires admin auth.
pub async fn initiate_pairing(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<InitiatePairingDto>,
) -> Result<Json<InitiatePairingResponse>, (StatusCode, String)> {
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin permission required".into()));
    }

    let (session, otp) = create_pairing_session(&req.channel_name);
    let session_id = session.id;
    state.pairing_store.lock().await.insert(session_id, session);

    Ok(Json(InitiatePairingResponse {
        session_id: session_id.to_string(),
        otp,
    }))
}

/// POST /api/v1/pair/confirm — no auth required.
pub async fn confirm_pairing_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ConfirmPairingDto>,
) -> Result<Json<ConfirmPairingResponse>, (StatusCode, String)> {
    let session_id = Uuid::parse_str(&req.session_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid session_id".into()))?;

    let mut sessions = state.pairing_store.lock().await;
    let session = sessions
        .get_mut(&session_id)
        .ok_or((StatusCode::NOT_FOUND, "session not found".into()))?;

    let token = confirm_pairing(session, &req.otp)
        .map_err(|e| (StatusCode::UNAUTHORIZED, e))?;

    Ok(Json(ConfirmPairingResponse {
        bearer_token: token,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthRole;
    use crate::state::AppState;
    use hx_core::model::pairing::PairingState;
    use hx_engine::config::EngineConfig;
    use hx_engine::engine::HelixEngine;
    use tempfile::TempDir;

    fn admin_auth() -> AuthContext {
        AuthContext {
            subject: None,
            role: AuthRole::Admin,
            namespace: None,
            consumer_name: None,
        }
    }

    fn read_auth() -> AuthContext {
        AuthContext {
            subject: None,
            role: AuthRole::Read,
            namespace: None,
            consumer_name: None,
        }
    }

    async fn test_state() -> Arc<AppState> {
        let temp_dir = TempDir::new().expect("temp dir");
        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.embedding.provider = "noop".into();
        let engine = HelixEngine::init(config).await.expect("engine init");
        Arc::new(AppState::new(Arc::new(engine)))
    }

    #[tokio::test]
    async fn initiate_requires_admin() {
        let state = test_state().await;
        let req = InitiatePairingDto {
            channel_name: "test".into(),
        };
        let result = initiate_pairing(
            Extension(read_auth()),
            State(state),
            Json(req),
        )
        .await;
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn initiate_and_confirm_flow() {
        let state = test_state().await;

        // Initiate
        let init_req = InitiatePairingDto {
            channel_name: "slack".into(),
        };
        let init_resp = initiate_pairing(
            Extension(admin_auth()),
            State(state.clone()),
            Json(init_req),
        )
        .await
        .unwrap();

        let otp = init_resp.otp.clone();
        let session_id = init_resp.session_id.clone();

        // Confirm
        let confirm_req = ConfirmPairingDto {
            session_id,
            otp,
        };
        let confirm_resp = confirm_pairing_handler(
            State(state.clone()),
            Json(confirm_req),
        )
        .await
        .unwrap();

        assert!(!confirm_resp.bearer_token.is_empty());

        // Session should be confirmed
        let sessions = state.pairing_store.lock().await;
        let uuid = Uuid::parse_str(&init_resp.session_id).unwrap();
        assert_eq!(sessions[&uuid].state, PairingState::Confirmed);
    }

    #[tokio::test]
    async fn confirm_wrong_otp_fails() {
        let state = test_state().await;

        let init_req = InitiatePairingDto {
            channel_name: "test".into(),
        };
        let init_resp = initiate_pairing(
            Extension(admin_auth()),
            State(state.clone()),
            Json(init_req),
        )
        .await
        .unwrap();

        let confirm_req = ConfirmPairingDto {
            session_id: init_resp.session_id.clone(),
            otp: "000000".into(),
        };
        let result = confirm_pairing_handler(
            State(state),
            Json(confirm_req),
        )
        .await;

        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn confirm_invalid_session_id_fails() {
        let state = test_state().await;

        let confirm_req = ConfirmPairingDto {
            session_id: Uuid::now_v7().to_string(),
            otp: "123456".into(),
        };
        let result = confirm_pairing_handler(
            State(state),
            Json(confirm_req),
        )
        .await;

        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn confirm_duplicate_fails() {
        let state = test_state().await;

        let init_req = InitiatePairingDto {
            channel_name: "dup-test".into(),
        };
        let init_resp = initiate_pairing(
            Extension(admin_auth()),
            State(state.clone()),
            Json(init_req),
        )
        .await
        .unwrap();

        let otp = init_resp.otp.clone();
        let session_id = init_resp.session_id.clone();

        // First confirm succeeds
        let confirm_req = ConfirmPairingDto {
            session_id: session_id.clone(),
            otp: otp.clone(),
        };
        let first = confirm_pairing_handler(
            State(state.clone()),
            Json(confirm_req),
        )
        .await;
        assert!(first.is_ok());

        // Second confirm fails
        let confirm_req2 = ConfirmPairingDto {
            session_id,
            otp,
        };
        let second = confirm_pairing_handler(
            State(state),
            Json(confirm_req2),
        )
        .await;
        assert!(second.is_err());
    }
}
