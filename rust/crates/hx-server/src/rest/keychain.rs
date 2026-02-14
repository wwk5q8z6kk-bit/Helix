//! REST API handlers for the Sovereign Keychain system.
//! All endpoints require admin authorization.

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

use base64::Engine;
use hx_core::{model::keychain::*, ChronicleEntry, HxError};

use crate::auth::{authorize_write, AuthContext};
use crate::limits::{
    enforce_keychain_read_rate_limit, enforce_keychain_unseal_failure_backoff,
    record_keychain_unseal_failure, RateLimitExceeded,
};
use crate::metrics::get_metrics;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_keychain_error(err: HxError) -> (StatusCode, String) {
    match &err {
        HxError::VaultSealed => (StatusCode::LOCKED, err.to_string()),
        HxError::KeychainNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        HxError::KeychainNotInitialized => (StatusCode::PRECONDITION_FAILED, err.to_string()),
        HxError::KeychainInvalidPassword => (StatusCode::UNAUTHORIZED, err.to_string()),
        HxError::KeychainAlreadyInitialized => (StatusCode::CONFLICT, err.to_string()),
        HxError::Keychain(_) => (StatusCode::BAD_REQUEST, err.to_string()),
        _ => (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()),
    }
}

fn require_admin(auth: &AuthContext) -> Result<(), (StatusCode, String)> {
    authorize_write(auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin access required".into()));
    }
    Ok(())
}

async fn log_unseal_attempt(
    state: &Arc<AppState>,
    subject: &str,
    method: &str,
    outcome: &str,
    reason: Option<&str>,
) {
    let logic = match reason {
        Some(reason) => {
            format!("subject={subject} method={method} outcome={outcome} reason={reason}")
        }
        None => format!("subject={subject} method={method} outcome={outcome}"),
    };

    let entry = ChronicleEntry::new("unseal_attempt", logic);
    if let Err(err) = state.engine.log_chronicle(&entry).await {
        tracing::warn!(error = %err, "failed to log unseal_attempt chronicle entry");
    }
}

fn map_unseal_rate_limit_error(exceeded: RateLimitExceeded) -> (StatusCode, String) {
    (
        StatusCode::TOO_MANY_REQUESTS,
        format!(
            "unseal rate limit exceeded: {} failures per {}s (retry after {}s)",
            exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
        ),
    )
}

async fn enforce_unseal_failure_backoff(
    state: &Arc<AppState>,
    subject: &str,
    method: &str,
) -> Result<(), (StatusCode, String)> {
    if let Err(exceeded) = enforce_keychain_unseal_failure_backoff(subject) {
        get_metrics().incr_vault_unseal_rate_limited();
        let reason = format!(
            "rate_limited:max={} window={} retry_after={}",
            exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
        );
        log_unseal_attempt(state, subject, method, "fail", Some(reason.as_str())).await;
        return Err(map_unseal_rate_limit_error(exceeded));
    }
    Ok(())
}

async fn log_and_record_unseal_failure(
    state: &Arc<AppState>,
    subject: &str,
    method: &str,
    reason: &str,
) -> Result<(), (StatusCode, String)> {
    get_metrics().incr_vault_unseal_failure();
    tracing::warn!(subject, method, reason, "vault unseal failed");
    log_unseal_attempt(state, subject, method, "fail", Some(reason)).await;
    if let Err(exceeded) = record_keychain_unseal_failure(subject) {
        get_metrics().incr_vault_unseal_rate_limited();
        let rate_reason = format!(
            "rate_limited:max={} window={} retry_after={}",
            exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
        );
        log_unseal_attempt(state, subject, method, "fail", Some(rate_reason.as_str())).await;
        return Err(map_unseal_rate_limit_error(exceeded));
    }
    Ok(())
}

fn failpoint_matches_data_dir(raw: &str, data_dir: &str) -> bool {
    let value = raw.trim();
    if value.is_empty() {
        return false;
    }
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "0" | "false" | "no" | "off" => false,
        "1" | "true" | "yes" | "on" => true,
        _ => value == data_dir,
    }
}

fn post_unseal_failpoint_enabled(state: &AppState, key: &str) -> bool {
    if !cfg!(debug_assertions) {
        return false;
    }
    let Ok(raw) = std::env::var(key) else {
        return false;
    };
    failpoint_matches_data_dir(&raw, &state.engine.config.data_dir)
}

async fn run_post_unseal_maintenance(
    state: &Arc<AppState>,
    subject: &str,
    method: &str,
) -> Result<(), (StatusCode, String)> {
    if post_unseal_failpoint_enabled(state.as_ref(), "HELIX_TEST_FAIL_POST_UNSEAL_MIGRATE") {
        let err = HxError::Storage("post-unseal migrate failpoint triggered".into());
        get_metrics().incr_vault_migration_failure();
        let reason = format!("post_unseal_migrate_failed:{err}");
        if let Err(rate_limited) =
            log_and_record_unseal_failure(state, subject, method, reason.as_str()).await
        {
            return Err(rate_limited);
        }
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_error(err));
    }

    if let Err(err) = state.engine.migrate_sealed_storage().await {
        get_metrics().incr_vault_migration_failure();
        let reason = format!("post_unseal_migrate_failed:{err}");
        if let Err(rate_limited) =
            log_and_record_unseal_failure(state, subject, method, reason.as_str()).await
        {
            return Err(rate_limited);
        }
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_error(err));
    }
    get_metrics().incr_vault_migration_success();

    if post_unseal_failpoint_enabled(state.as_ref(), "HELIX_TEST_FAIL_POST_UNSEAL_REBUILD") {
        let err = HxError::Storage("post-unseal rebuild failpoint triggered".into());
        get_metrics().incr_vault_rebuild_failure();
        let reason = format!("post_unseal_rebuild_failed:{err}");
        if let Err(rate_limited) =
            log_and_record_unseal_failure(state, subject, method, reason.as_str()).await
        {
            return Err(rate_limited);
        }
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_error(err));
    }

    if let Err(err) = state.engine.rebuild_runtime_indexes().await {
        get_metrics().incr_vault_rebuild_failure();
        let reason = format!("post_unseal_rebuild_failed:{err}");
        if let Err(rate_limited) =
            log_and_record_unseal_failure(state, subject, method, reason.as_str()).await
        {
            return Err(rate_limited);
        }
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_error(err));
    }
    get_metrics().incr_vault_rebuild_success();

    Ok(())
}

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct InitVaultRequest {
    pub password: String,
    #[serde(default)]
    pub macos_bridge: bool,
}

#[derive(Deserialize)]
pub struct UnsealRequest {
    pub password: Option<String>,
    #[serde(default)]
    pub from_macos_keychain: bool,
    #[serde(default)]
    pub from_secure_enclave: bool,
}

#[derive(Serialize)]
pub struct VaultStatusResponse {
    pub state: String,
    pub key_epoch: Option<u64>,
    pub credential_count: Option<usize>,
    pub domain_count: Option<usize>,
    pub created_at: Option<String>,
    pub last_rotated_at: Option<String>,
    pub auto_seal_remaining_secs: Option<u64>,
    pub shamir_enabled: bool,
    pub last_lifecycle_run: Option<String>,
    pub degraded_security: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sealed_blocked_requests: Option<u64>,
}

#[derive(Deserialize)]
pub struct RotateKeyRequest {
    pub new_password: String,
    #[serde(default = "default_grace_hours")]
    pub grace_hours: u32,
}

fn default_grace_hours() -> u32 {
    24
}

#[derive(Deserialize)]
pub struct CreateDomainRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Deserialize)]
pub struct StoreCredentialRequest {
    pub domain_id: String,
    pub name: String,
    pub kind: String,
    pub value: String,
    #[serde(default)]
    pub tags: Vec<String>,
    pub expires_at: Option<String>,
}

#[derive(Deserialize)]
pub struct UpdateCredentialRequest {
    pub value: String,
}

#[derive(Deserialize)]
pub struct CreateDelegationRequest {
    pub credential_id: String,
    pub delegatee: String,
    #[serde(default)]
    pub can_read: bool,
    #[serde(default)]
    pub can_use: bool,
    #[serde(default)]
    pub can_delegate: bool,
    pub expires_at: Option<String>,
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,
}

fn default_max_depth() -> u32 {
    3
}

#[derive(Deserialize)]
pub struct SubDelegateRequest {
    pub delegatee: String,
    #[serde(default)]
    pub can_read: bool,
    #[serde(default)]
    pub can_use: bool,
    #[serde(default)]
    pub can_delegate: bool,
    pub expires_at: Option<String>,
}

#[derive(Deserialize)]
pub struct GenerateProofRequest {
    pub credential_id: String,
    pub challenge_nonce: String,
}

#[derive(Deserialize)]
pub struct VerifyProofRequest {
    pub credential_id: String,
    pub challenge_nonce: String,
    pub proof: String,
    pub generated_at: String,
    pub expires_at: String,
}

#[derive(Deserialize)]
pub struct ListQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    pub domain_id: Option<String>,
    pub state: Option<String>,
}

fn default_limit() -> usize {
    50
}

#[derive(Deserialize)]
pub struct DelegationListQuery {
    pub credential_id: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn init_vault(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<InitVaultRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    state
        .engine
        .keychain
        .initialize_vault(&body.password, body.macos_bridge, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "initialized"})))
}

pub async fn unseal_vault(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<UnsealRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let method = if body.from_secure_enclave {
        "secure_enclave"
    } else if body.from_macos_keychain {
        "macos_keychain"
    } else {
        "preferred"
    };
    enforce_unseal_failure_backoff(&state, subject, method).await?;

    let unseal_result: Result<(), HxError> = if body.from_secure_enclave {
        #[cfg(target_os = "macos")]
        {
            state.engine.keychain.unseal_from_secure_enclave().await
        }
        #[cfg(not(target_os = "macos"))]
        {
            if let Err(rate_limited) = log_and_record_unseal_failure(
                &state,
                subject,
                method,
                "secure_enclave_requires_macos",
            )
            .await
            {
                return Err(rate_limited);
            }
            return Err((
                StatusCode::BAD_REQUEST,
                "Secure Enclave only available on macOS".into(),
            ));
        }
    } else if body.from_macos_keychain {
        state
            .engine
            .keychain
            .unseal_from_macos_keychain(subject)
            .await
    } else {
        state
            .engine
            .keychain
            .unseal_with_preferred_master_key(
                body.password.as_deref().filter(|value| !value.is_empty()),
                subject,
            )
            .await
            .map(|_| ())
    };

    match unseal_result {
        Ok(()) => {
            if state.engine.keychain.degraded_security_mode() {
                tracing::warn!("vault unsealed in degraded security mode (passphrase fallback)");
            }
        }
        Err(err) => {
            let reason = err.to_string();
            if let Err(rate_limited) =
                log_and_record_unseal_failure(&state, subject, method, reason.as_str()).await
            {
                return Err(rate_limited);
            }
            return Err(map_keychain_error(err));
        }
    }

    run_post_unseal_maintenance(&state, subject, method).await?;

    log_unseal_attempt(&state, subject, method, "success", None).await;
    tracing::info!(subject, method, "vault unsealed successfully");

    // Start auto-seal timer after successful unseal
    state.engine.keychain.start_auto_seal().await;

    // Reset blocked request counter now that vault is open.
    state
        .sealed_blocked_requests
        .store(0, std::sync::atomic::Ordering::Relaxed);

    Ok(Json(serde_json::json!({"status": "unsealed"})))
}

pub async fn seal_vault(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    state
        .engine
        .keychain
        .seal(subject)
        .await
        .map_err(map_keychain_error)?;
    tracing::info!(subject, "vault sealed");
    Ok(Json(serde_json::json!({"status": "sealed"})))
}

pub async fn vault_status(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<VaultStatusResponse>, (StatusCode, String)> {
    require_admin(&auth)?;
    let (vault_state, meta) = state
        .engine
        .keychain
        .vault_status()
        .await
        .map_err(map_keychain_error)?;

    let (cred_count, domain_count) = if vault_state != VaultState::Uninitialized {
        let cc = state
            .engine
            .keychain
            .store
            .count_credentials(None)
            .await
            .unwrap_or(0);
        let domains = state
            .engine
            .keychain
            .list_domains()
            .await
            .unwrap_or_default();
        (Some(cc), Some(domains.len()))
    } else {
        (None, None)
    };

    let seal_remaining = state.engine.keychain.auto_seal_remaining().await;
    let shamir_enabled = meta.as_ref().and_then(|m| m.shamir_threshold).is_some();
    let last_lifecycle = state
        .engine
        .keychain
        .last_lifecycle_run()
        .await
        .map(|dt| dt.to_rfc3339());

    let blocked = state
        .sealed_blocked_requests
        .load(std::sync::atomic::Ordering::Relaxed);
    let sealed_blocked_requests = if state.engine.config.sealed_mode {
        Some(blocked)
    } else {
        None
    };

    Ok(Json(VaultStatusResponse {
        state: vault_state.to_string(),
        key_epoch: meta.as_ref().map(|m| m.key_epoch),
        credential_count: cred_count,
        domain_count,
        created_at: meta.as_ref().map(|m| m.created_at.to_rfc3339()),
        last_rotated_at: meta
            .as_ref()
            .and_then(|m| m.last_rotated_at.map(|dt| dt.to_rfc3339())),
        auto_seal_remaining_secs: seal_remaining,
        shamir_enabled,
        last_lifecycle_run: last_lifecycle,
        degraded_security: state.engine.keychain.degraded_security_mode(),
        sealed_blocked_requests,
    }))
}

pub async fn rotate_key(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<RotateKeyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    state
        .engine
        .keychain
        .rotate_master_key(&body.new_password, body.grace_hours, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "rotated"})))
}

pub async fn list_epochs(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<Vec<KeyEpoch>>, (StatusCode, String)> {
    require_admin(&auth)?;
    let epochs = state
        .engine
        .keychain
        .store
        .list_key_epochs()
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(epochs))
}

pub async fn create_domain(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<CreateDomainRequest>,
) -> Result<(StatusCode, Json<DomainKey>), (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let domain = state
        .engine
        .keychain
        .create_domain(&body.name, body.description.as_deref(), subject)
        .await
        .map_err(map_keychain_error)?;
    Ok((StatusCode::CREATED, Json(domain)))
}

pub async fn list_domains(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<Vec<DomainKey>>, (StatusCode, String)> {
    require_admin(&auth)?;
    let domains = state
        .engine
        .keychain
        .list_domains()
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(domains))
}

pub async fn revoke_domain(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    state
        .engine
        .keychain
        .revoke_domain(uuid, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "revoked"})))
}

pub async fn store_credential(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<StoreCredentialRequest>,
) -> Result<(StatusCode, Json<StoredCredential>), (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let domain_id =
        Uuid::parse_str(&body.domain_id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let expires_at = match body.expires_at {
        Some(ref s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid expires_at: {e}")))?,
        ),
        None => None,
    };
    let cred = state
        .engine
        .keychain
        .store_credential(
            domain_id,
            &body.name,
            &body.kind,
            body.value.as_bytes(),
            body.tags,
            expires_at,
            subject,
        )
        .await
        .map_err(map_keychain_error)?;
    Ok((StatusCode::CREATED, Json(cred)))
}

pub async fn list_credentials(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Vec<StoredCredential>>, (StatusCode, String)> {
    require_admin(&auth)?;
    let domain_id = match q.domain_id {
        Some(ref s) => {
            Some(Uuid::parse_str(s).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?)
        }
        None => None,
    };
    let cred_state = match q.state {
        Some(ref s) => Some(
            s.parse::<CredentialState>()
                .map_err(|e| (StatusCode::BAD_REQUEST, e))?,
        ),
        None => None,
    };
    let creds = state
        .engine
        .keychain
        .list_credentials(domain_id, cred_state, q.limit, q.offset)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(creds))
}

pub async fn read_credential(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    if let Err(exceeded) = enforce_keychain_read_rate_limit(subject, &id) {
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            format!(
                "rate limit exceeded: {} requests per {}s (retry after {}s)",
                exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
            ),
        ));
    }
    let (cred, plaintext, alerts) = state
        .engine
        .keychain
        .read_credential(uuid, subject)
        .await
        .map_err(map_keychain_error)?;

    // Dispatch webhook for any new breach alerts
    for alert in &alerts {
        state.notify_keychain_alert(alert);
    }

    let value = String::from_utf8_lossy(&plaintext).to_string();
    Ok(Json(serde_json::json!({
        "credential": cred,
        "value": value,
    })))
}

pub async fn update_credential(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(body): Json<UpdateCredentialRequest>,
) -> Result<Json<StoredCredential>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let cred = state
        .engine
        .keychain
        .update_credential_value(uuid, body.value.as_bytes(), subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(cred))
}

pub async fn archive_credential(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    state
        .engine
        .keychain
        .archive_credential(uuid, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "archived"})))
}

pub async fn destroy_credential(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    state
        .engine
        .keychain
        .destroy_credential(uuid, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "destroyed"})))
}

pub async fn create_delegation(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<CreateDelegationRequest>,
) -> Result<(StatusCode, Json<Delegation>), (StatusCode, String)> {
    require_admin(&auth)?;
    let cred_id = Uuid::parse_str(&body.credential_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let expires_at = match body.expires_at {
        Some(ref s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid expires_at: {e}")))?,
        ),
        None => None,
    };
    let perms = DelegationPermissions {
        can_read: body.can_read,
        can_use: body.can_use,
        can_delegate: body.can_delegate,
    };
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let delegation = state
        .engine
        .keychain
        .create_delegation(
            cred_id,
            &body.delegatee,
            perms,
            expires_at,
            body.max_depth,
            subject,
        )
        .await
        .map_err(map_keychain_error)?;
    Ok((StatusCode::CREATED, Json(delegation)))
}

pub async fn list_delegations(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(q): Query<DelegationListQuery>,
) -> Result<Json<Vec<Delegation>>, (StatusCode, String)> {
    require_admin(&auth)?;
    let cred_id =
        Uuid::parse_str(&q.credential_id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let delegations = state
        .engine
        .keychain
        .list_delegations(cred_id)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(delegations))
}

pub async fn revoke_delegation(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    state
        .engine
        .keychain
        .revoke_delegation(uuid, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "revoked"})))
}

pub async fn sub_delegate(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
    Json(body): Json<SubDelegateRequest>,
) -> Result<(StatusCode, Json<Delegation>), (StatusCode, String)> {
    require_admin(&auth)?;
    let parent_id = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let expires_at = match body.expires_at {
        Some(ref s) => Some(
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid expires_at: {e}")))?,
        ),
        None => None,
    };
    let perms = DelegationPermissions {
        can_read: body.can_read,
        can_use: body.can_use,
        can_delegate: body.can_delegate,
    };
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let delegation = state
        .engine
        .keychain
        .sub_delegate(parent_id, &body.delegatee, perms, expires_at, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok((StatusCode::CREATED, Json(delegation)))
}

pub async fn generate_proof(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<GenerateProofRequest>,
) -> Result<Json<AccessProof>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let cred_id = Uuid::parse_str(&body.credential_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let proof = state
        .engine
        .keychain
        .generate_proof(cred_id, &body.challenge_nonce, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(proof))
}

pub async fn verify_proof(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<VerifyProofRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let cred_id = Uuid::parse_str(&body.credential_id)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let generated_at = chrono::DateTime::parse_from_rfc3339(&body.generated_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                format!("invalid generated_at: {e}"),
            )
        })?;
    let expires_at = chrono::DateTime::parse_from_rfc3339(&body.expires_at)
        .map(|dt| dt.with_timezone(&chrono::Utc))
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid expires_at: {e}")))?;

    let zk_proof = ZkAccessProof {
        credential_id: cred_id,
        challenge_nonce: body.challenge_nonce,
        proof: body.proof,
        generated_at,
        expires_at,
    };
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let valid = state
        .engine
        .keychain
        .verify_proof(&zk_proof, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"valid": valid})))
}

pub async fn list_audit(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Vec<KeychainAuditEntry>>, (StatusCode, String)> {
    require_admin(&auth)?;
    let entries = state
        .engine
        .keychain
        .list_audit_trail(q.limit, q.offset)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(entries))
}

pub async fn verify_audit_integrity(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let result = state
        .engine
        .keychain
        .verify_audit_integrity()
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({
        "result": result.as_str(),
        "valid": result.is_valid(),
        "signatures_checked": result.signatures_checked(),
    })))
}

pub async fn list_alerts(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(q): Query<ListQuery>,
) -> Result<Json<Vec<BreachAlert>>, (StatusCode, String)> {
    require_admin(&auth)?;
    let alerts = state
        .engine
        .keychain
        .list_breach_alerts(q.limit, q.offset)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(alerts))
}

pub async fn acknowledge_alert(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let uuid = Uuid::parse_str(&id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    state
        .engine
        .keychain
        .acknowledge_alert(uuid)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "acknowledged"})))
}

pub async fn run_lifecycle(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let count = state
        .engine
        .keychain
        .run_lifecycle_transitions()
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"transitioned": count})))
}

// ---------------------------------------------------------------------------
// Shamir VEK Splitting
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct EnableShamirRequest {
    pub threshold: u8,
    pub total: u8,
    pub passphrases: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct SubmitShareRequest {
    pub share: String,
    pub passphrase: Option<String>,
}

pub async fn enable_shamir(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<EnableShamirRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let shares = state
        .engine
        .keychain
        .enable_shamir(body.threshold, body.total, subject, body.passphrases)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({
        "status": "enabled",
        "threshold": body.threshold,
        "total": body.total,
        "shares": shares,
    })))
}

pub async fn submit_share(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<SubmitShareRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let status = state
        .engine
        .keychain
        .submit_shamir_share(&body.share, body.passphrase.as_deref())
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::to_value(status).unwrap()))
}

pub async fn shamir_unseal(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let method = "shamir_shares";
    enforce_unseal_failure_backoff(&state, subject, method).await?;
    if let Err(err) = state.engine.keychain.unseal_from_shares(subject).await {
        let reason = err.to_string();
        if let Err(rate_limited) =
            log_and_record_unseal_failure(&state, subject, method, reason.as_str()).await
        {
            return Err(rate_limited);
        }
        return Err(map_keychain_error(err));
    }
    run_post_unseal_maintenance(&state, subject, method).await?;
    log_unseal_attempt(&state, subject, method, "success", None).await;
    // Start auto-seal timer after successful unseal
    state.engine.keychain.start_auto_seal().await;
    state
        .sealed_blocked_requests
        .store(0, std::sync::atomic::Ordering::Relaxed);
    Ok(Json(serde_json::json!({"status": "unsealed"})))
}

#[derive(Deserialize)]
pub struct RotateShamirRequest {
    pub passphrases: Option<Vec<String>>,
}

pub async fn rotate_shamir(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<RotateShamirRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let shares = state
        .engine
        .keychain
        .rotate_shamir_shares(subject, body.passphrases)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({
        "shares": shares,
        "count": shares.len(),
    })))
}

pub async fn shamir_status(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let status = state
        .engine
        .keychain
        .shamir_status()
        .await
        .map_err(map_keychain_error)?;
    match status {
        Some(s) => Ok(Json(serde_json::to_value(s).unwrap())),
        None => Ok(Json(serde_json::json!({"enabled": false}))),
    }
}

// ---------------------------------------------------------------------------
// Backup / Restore
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct BackupRequest {
    pub password: String,
}

#[derive(Deserialize)]
pub struct RestoreRequest {
    pub password: String,
    pub data: String, // base64-encoded backup data
}

// ---------------------------------------------------------------------------
// Domain ACLs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct SetAclRequest {
    pub subject: String,
    #[serde(default = "default_true")]
    pub can_read: bool,
    #[serde(default)]
    pub can_write: bool,
    #[serde(default)]
    pub can_admin: bool,
    pub expires_at: Option<String>,
}

fn default_true() -> bool {
    true
}

pub async fn set_domain_acl(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(domain_id): Path<String>,
    Json(body): Json<SetAclRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let domain_uuid =
        Uuid::parse_str(&domain_id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let expires_at = body
        .expires_at
        .as_deref()
        .map(|s| {
            chrono::DateTime::parse_from_rfc3339(s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid expires_at: {e}")))
        })
        .transpose()?;

    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let acl = state
        .engine
        .keychain
        .set_domain_acl(
            domain_uuid,
            &body.subject,
            body.can_read,
            body.can_write,
            body.can_admin,
            expires_at,
            subject,
        )
        .await
        .map_err(map_keychain_error)?;

    Ok(Json(serde_json::json!(acl)))
}

pub async fn list_domain_acls(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(domain_id): Path<String>,
) -> Result<Json<Vec<serde_json::Value>>, (StatusCode, String)> {
    require_admin(&auth)?;
    let domain_uuid =
        Uuid::parse_str(&domain_id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let acls = state
        .engine
        .keychain
        .list_domain_acls(domain_uuid)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(
        acls.into_iter().map(|a| serde_json::json!(a)).collect(),
    ))
}

pub async fn delete_domain_acl(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(acl_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    require_admin(&auth)?;
    let uuid = Uuid::parse_str(&acl_id).map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    state
        .engine
        .keychain
        .remove_domain_acl(uuid, subject)
        .await
        .map_err(map_keychain_error)?;
    Ok(StatusCode::NO_CONTENT)
}

pub async fn backup_vault(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<BackupRequest>,
) -> Result<(StatusCode, Vec<u8>), (StatusCode, String)> {
    require_admin(&auth)?;
    let data = state
        .engine
        .keychain
        .backup_vault(&body.password)
        .await
        .map_err(map_keychain_error)?;
    Ok((StatusCode::OK, data))
}

pub async fn restore_vault(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(body): Json<RestoreRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    require_admin(&auth)?;
    let data = base64::engine::general_purpose::STANDARD
        .decode(&body.data)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid base64: {e}")))?;
    state
        .engine
        .keychain
        .restore_vault(&data, &body.password)
        .await
        .map_err(map_keychain_error)?;
    Ok(Json(serde_json::json!({"status": "restored"})))
}
