use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    Extension, Json,
};
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use hx_core::{ChronicleEntry, HxError, StoredCredential};
use hx_engine::engine::HelixEngine;

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;
use crate::validation::{validate_list_limit, validate_text_input};

const OAUTH_CLIENT_DOMAIN: &str = "oauth-clients";
const DEFAULT_TOKEN_TTL_SECS: u64 = 3600;
const METADATA_TEMPLATE_ID: &str = "template_id";
const METADATA_DISPLAY_NAME: &str = "display_name";
const METADATA_TOKEN_TTL_SECS: &str = "token_ttl_seconds";
const METADATA_DESCRIPTION: &str = "description";

// ---------------------------------------------------------------------------
// DTOs
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct OAuthClientCreateRequest {
    pub name: String,
    pub template_id: String,
    pub description: Option<String>,
    pub token_ttl_seconds: Option<u64>,
    pub expires_at: Option<String>,
}

#[derive(Serialize)]
pub struct OAuthClientResponse {
    pub client_id: String,
    pub name: String,
    pub template_id: String,
    pub created_at: String,
    pub updated_at: String,
    pub last_used_at: Option<String>,
    pub expires_at: Option<String>,
    pub revoked_at: Option<String>,
    pub token_ttl_seconds: u64,
    pub description: Option<String>,
}

#[derive(Serialize)]
pub struct OAuthClientCreateResponse {
    pub client: OAuthClientResponse,
    pub client_secret: String,
}

#[derive(Deserialize)]
pub struct OAuthClientListQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Deserialize)]
pub struct OAuthTokenRequest {
    pub grant_type: String,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub scope: Option<String>,
}

#[derive(Serialize)]
pub struct OAuthTokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub scope: Option<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_keychain_error(err: HxError) -> (StatusCode, String) {
    match &err {
        HxError::VaultSealed => (StatusCode::LOCKED, err.to_string()),
        HxError::KeychainNotFound(_) => (StatusCode::NOT_FOUND, err.to_string()),
        HxError::KeychainNotInitialized => (StatusCode::PRECONDITION_FAILED, err.to_string()),
        HxError::KeychainAlreadyInitialized => (StatusCode::CONFLICT, err.to_string()),
        HxError::KeychainInvalidPassword => (StatusCode::UNAUTHORIZED, err.to_string()),
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

fn parse_rfc3339(
    field: &str,
    value: Option<String>,
) -> Result<Option<DateTime<Utc>>, (StatusCode, String)> {
    let Some(value) = value else { return Ok(None) };
    DateTime::parse_from_rfc3339(&value)
        .map(|dt| Some(dt.with_timezone(&Utc)))
        .map_err(|_| (StatusCode::BAD_REQUEST, format!("invalid {field} timestamp")))
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.bytes().zip(b.bytes()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn extract_basic_auth(headers: &HeaderMap) -> Option<(String, String)> {
    let auth_header = headers.get(axum::http::header::AUTHORIZATION)?;
    let auth_str = auth_header.to_str().ok()?;
    let prefix = "Basic ";
    if !auth_str.starts_with(prefix) {
        return None;
    }
    let encoded = auth_str.trim_start_matches(prefix);
    let decoded = base64::engine::general_purpose::STANDARD.decode(encoded).ok()?;
    let decoded = String::from_utf8(decoded).ok()?;
    let mut parts = decoded.splitn(2, ':');
    let client_id = parts.next()?.to_string();
    let client_secret = parts.next()?.to_string();
    Some((client_id, client_secret))
}

fn build_client_response(
    client_id: String,
    metadata: &HashMap<String, Value>,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    last_accessed_at: DateTime<Utc>,
    expires_at: Option<DateTime<Utc>>,
    revoked_at: Option<DateTime<Utc>>,
) -> OAuthClientResponse {
    let template_id = metadata
        .get(METADATA_TEMPLATE_ID)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    let display_name = metadata
        .get(METADATA_DISPLAY_NAME)
        .and_then(|v| v.as_str())
        .unwrap_or(&client_id)
        .to_string();
    let token_ttl_seconds = metadata
        .get(METADATA_TOKEN_TTL_SECS)
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_TOKEN_TTL_SECS);
    let description = metadata
        .get(METADATA_DESCRIPTION)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    OAuthClientResponse {
        client_id,
        name: display_name,
        template_id,
        created_at: created_at.to_rfc3339(),
        updated_at: updated_at.to_rfc3339(),
        last_used_at: Some(last_accessed_at.to_rfc3339()),
        expires_at: expires_at.map(|dt| dt.to_rfc3339()),
        revoked_at: revoked_at.map(|dt| dt.to_rfc3339()),
        token_ttl_seconds,
        description,
    }
}

fn credential_invalid_for_token(cred: &StoredCredential, now: DateTime<Utc>) -> bool {
    if let Some(expires_at) = cred.expires_at {
        if expires_at <= now {
            return true;
        }
    }
    if cred.archived_at.is_some() || cred.destroyed_at.is_some() {
        return true;
    }
    false
}

async fn ensure_template_exists(
    engine: &HelixEngine,
    template_id: Uuid,
) -> Result<(), (StatusCode, String)> {
    let template = engine
        .store
        .nodes
        .get_permission_template(template_id)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;
    if template.is_none() {
        return Err((StatusCode::BAD_REQUEST, "permission template not found".into()));
    }
    Ok(())
}

async fn oauth_domain_id(engine: &HelixEngine) -> Result<Uuid, HxError> {
    engine.keychain.find_or_create_domain(OAUTH_CLIENT_DOMAIN, "system").await
}

async fn revoke_oauth_access_keys(
    engine: &HelixEngine,
    client_id: &str,
) -> Result<usize, HxError> {
    let keys = engine.list_access_keys().await?;
    let label = format!("oauth:{client_id}");
    let mut revoked = 0;
    for key in keys {
        if key.name.as_deref() == Some(label.as_str()) {
            if engine.revoke_access_key(key.id).await? {
                revoked += 1;
            }
        }
    }
    Ok(revoked)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

pub async fn create_oauth_client(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(payload): Json<OAuthClientCreateRequest>,
) -> Result<Json<OAuthClientCreateResponse>, (StatusCode, String)> {
    require_admin(&auth)?;

    validate_text_input("name", &payload.name).map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;

    let template_id = Uuid::parse_str(&payload.template_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid template_id".into()))?;
    ensure_template_exists(&state.engine, template_id).await?;

    let expires_at = parse_rfc3339("expires_at", payload.expires_at)?;

    let client_id = Uuid::now_v7().to_string();
    let client_secret = generate_client_secret();

    let domain_id = oauth_domain_id(&state.engine)
        .await
        .map_err(map_keychain_error)?;

    let subject = auth.subject.as_deref().unwrap_or("anonymous");
    let stored = state
        .engine
        .keychain
        .store_credential(
            domain_id,
            &client_id,
            "oauth_client_secret",
            client_secret.as_bytes(),
            vec!["oauth".into(), "client".into()],
            expires_at,
            subject,
        )
        .await
        .map_err(map_keychain_error)?;

    let mut metadata = HashMap::new();
    metadata.insert(
        METADATA_TEMPLATE_ID.to_string(),
        Value::String(template_id.to_string()),
    );
    metadata.insert(
        METADATA_DISPLAY_NAME.to_string(),
        Value::String(payload.name.clone()),
    );
    if let Some(desc) = payload.description.clone() {
        metadata.insert(METADATA_DESCRIPTION.to_string(), Value::String(desc));
    }
    metadata.insert(
        METADATA_TOKEN_TTL_SECS.to_string(),
        Value::Number(
            (payload.token_ttl_seconds.unwrap_or(DEFAULT_TOKEN_TTL_SECS)).into(),
        ),
    );

    let updated = state
        .engine
        .keychain
        .update_credential_metadata(
            stored.id,
            payload.description.clone(),
            Some(stored.tags.clone()),
            Some(metadata),
            stored.expires_at,
            subject,
        )
        .await
        .map_err(map_keychain_error)?;

    let response = build_client_response(
        updated.name.clone(),
        &updated.metadata,
        updated.created_at,
        updated.updated_at,
        updated.last_accessed_at,
        updated.expires_at,
        updated.destroyed_at.or(updated.archived_at),
    );

    let chronicle = ChronicleEntry::new(
        "oauth.client_create",
        format!("Created OAuth client '{}' (id: {})", payload.name, updated.name),
    );
    let _ = state.engine.log_chronicle(&chronicle).await;

    Ok(Json(OAuthClientCreateResponse {
        client: response,
        client_secret,
    }))
}

pub async fn list_oauth_clients(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<OAuthClientListQuery>,
) -> Result<Json<Vec<OAuthClientResponse>>, (StatusCode, String)> {
    authorize_read(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin access required".into()));
    }

    let limit = params.limit.unwrap_or(200).min(500);
    validate_list_limit(limit).map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
    let offset = params.offset.unwrap_or(0);

    let domain_id = oauth_domain_id(&state.engine)
        .await
        .map_err(map_keychain_error)?;

    let creds = state
        .engine
        .keychain
        .list_credentials(Some(domain_id), None, limit, offset)
        .await
        .map_err(map_keychain_error)?;

    let responses = creds
        .into_iter()
        .map(|cred| {
            build_client_response(
                cred.name,
                &cred.metadata,
                cred.created_at,
                cred.updated_at,
                cred.last_accessed_at,
                cred.expires_at,
                cred.destroyed_at.or(cred.archived_at),
            )
        })
        .collect();

    Ok(Json(responses))
}

pub async fn revoke_oauth_client(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(client_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    require_admin(&auth)?;
    let subject = auth.subject.as_deref().unwrap_or("anonymous");

    let domain_id = oauth_domain_id(&state.engine)
        .await
        .map_err(map_keychain_error)?;

    let Some((cred, _)) = state
        .engine
        .keychain
        .read_credential_by_name(domain_id, &client_id)
        .await
        .map_err(map_keychain_error)?
    else {
        return Err((StatusCode::NOT_FOUND, "oauth client not found".into()));
    };

    state
        .engine
        .keychain
        .destroy_credential(cred.id, subject)
        .await
        .map_err(map_keychain_error)?;

    let revoked_keys = match revoke_oauth_access_keys(&state.engine, &client_id).await {
        Ok(count) => Some(count),
        Err(err) => {
            tracing::warn!(
                client_id = %client_id,
                error = %err,
                "failed to revoke access keys for oauth client"
            );
            None
        }
    };

    let chronicle = ChronicleEntry::new(
        "oauth.client_revoke",
        match revoked_keys {
            Some(count) => format!(
                "Revoked OAuth client '{}' and {} access key(s)",
                client_id, count
            ),
            None => format!("Revoked OAuth client '{}' (access key revocation failed)", client_id),
        },
    );
    let _ = state.engine.log_chronicle(&chronicle).await;

    Ok(StatusCode::NO_CONTENT)
}

pub async fn oauth_token(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<Json<OAuthTokenResponse>, (StatusCode, String)> {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");

    let mut params: OAuthTokenRequest = if content_type.starts_with("application/x-www-form-urlencoded")
    {
        serde_urlencoded::from_bytes(&body)
            .map_err(|_| (StatusCode::BAD_REQUEST, "invalid form payload".into()))?
    } else {
        serde_json::from_slice(&body)
            .map_err(|_| (StatusCode::BAD_REQUEST, "invalid json payload".into()))?
    };

    if params.grant_type != "client_credentials" {
        return Err((
            StatusCode::BAD_REQUEST,
            "unsupported grant_type".into(),
        ));
    }

    let (client_id, client_secret) = if let Some((id, secret)) = extract_basic_auth(&headers) {
        (id, secret)
    } else {
        (
            params
                .client_id
                .take()
                .ok_or((StatusCode::BAD_REQUEST, "client_id required".into()))?,
            params
                .client_secret
                .take()
                .ok_or((StatusCode::BAD_REQUEST, "client_secret required".into()))?,
        )
    };

    let domain_id = oauth_domain_id(&state.engine)
        .await
        .map_err(map_keychain_error)?;

    let Some((cred, plaintext)) = state
        .engine
        .keychain
        .read_credential_by_name(domain_id, &client_id)
        .await
        .map_err(map_keychain_error)?
    else {
        let chronicle = ChronicleEntry::new(
            "oauth.token_denied",
            format!("Token request denied for client '{}'", client_id),
        );
        let _ = state.engine.log_chronicle(&chronicle).await;
        return Err((StatusCode::UNAUTHORIZED, "invalid client".into()));
    };

    let now = Utc::now();
    if credential_invalid_for_token(&cred, now) {
        let chronicle = ChronicleEntry::new(
            "oauth.token_denied",
            format!("Token request denied for client '{}' (expired)", client_id),
        );
        let _ = state.engine.log_chronicle(&chronicle).await;
        return Err((StatusCode::UNAUTHORIZED, "invalid client".into()));
    }

    let stored_secret = String::from_utf8(plaintext.to_vec())
        .map_err(|_| (StatusCode::UNAUTHORIZED, "invalid client".into()))?;

    if !constant_time_eq(&stored_secret, &client_secret) {
        let chronicle = ChronicleEntry::new(
            "oauth.token_denied",
            format!("Token request denied for client '{}'", client_id),
        );
        let _ = state.engine.log_chronicle(&chronicle).await;
        return Err((StatusCode::UNAUTHORIZED, "invalid client".into()));
    }

    let template_id = cred
        .metadata
        .get(METADATA_TEMPLATE_ID)
        .and_then(|v| v.as_str())
        .ok_or((StatusCode::BAD_REQUEST, "template_id missing".into()))?;
    let template_id = Uuid::parse_str(template_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid template_id".into()))?;
    ensure_template_exists(&state.engine, template_id).await?;

    let ttl_seconds = cred
        .metadata
        .get(METADATA_TOKEN_TTL_SECS)
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_TOKEN_TTL_SECS);

    let expires_at = Utc::now() + Duration::seconds(ttl_seconds as i64);

    let (access_key, token) = state
        .engine
        .create_access_key(
            template_id,
            Some(format!("oauth:{}", cred.name)),
            Some(expires_at),
        )
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    let _ = access_key;

    let chronicle = ChronicleEntry::new(
        "oauth.token_issued",
        format!("Issued OAuth token for client '{}'", client_id),
    );
    let _ = state.engine.log_chronicle(&chronicle).await;

    Ok(Json(OAuthTokenResponse {
        access_token: token,
        token_type: "Bearer".into(),
        expires_in: ttl_seconds,
        scope: params.scope,
    }))
}

fn generate_client_secret() -> String {
    use rand::RngCore;

    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_engine::config::EngineConfig;
    use hx_engine::engine::HelixEngine;
    use tempfile::TempDir;

    #[test]
    fn constant_time_eq_same_strings() {
        assert!(constant_time_eq("hello", "hello"));
    }

    #[test]
    fn constant_time_eq_different_strings() {
        assert!(!constant_time_eq("hello", "world"));
    }

    #[test]
    fn constant_time_eq_different_lengths() {
        assert!(!constant_time_eq("short", "longer string"));
    }

    #[test]
    fn constant_time_eq_empty_strings() {
        assert!(constant_time_eq("", ""));
    }

    #[test]
    fn extract_basic_auth_valid() {
        let mut headers = HeaderMap::new();
        let encoded = base64::engine::general_purpose::STANDARD.encode("client_id:client_secret");
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Basic {encoded}").parse().unwrap(),
        );
        let result = extract_basic_auth(&headers);
        assert!(result.is_some());
        let (id, secret) = result.unwrap();
        assert_eq!(id, "client_id");
        assert_eq!(secret, "client_secret");
    }

    #[test]
    fn extract_basic_auth_missing_header() {
        let headers = HeaderMap::new();
        assert!(extract_basic_auth(&headers).is_none());
    }

    #[test]
    fn extract_basic_auth_bearer_not_basic() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Bearer some-token".parse().unwrap(),
        );
        assert!(extract_basic_auth(&headers).is_none());
    }

    #[test]
    fn extract_basic_auth_invalid_base64() {
        let mut headers = HeaderMap::new();
        headers.insert(
            axum::http::header::AUTHORIZATION,
            "Basic not-valid-base64!!!".parse().unwrap(),
        );
        assert!(extract_basic_auth(&headers).is_none());
    }

    #[test]
    fn extract_basic_auth_no_colon_separator() {
        let mut headers = HeaderMap::new();
        let encoded = base64::engine::general_purpose::STANDARD.encode("nocolon");
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Basic {encoded}").parse().unwrap(),
        );
        assert!(extract_basic_auth(&headers).is_none());
    }

    #[test]
    fn parse_rfc3339_valid() {
        let result = parse_rfc3339("expires_at", Some("2026-12-31T23:59:59Z".into()));
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn parse_rfc3339_none() {
        let result = parse_rfc3339("expires_at", None);
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn parse_rfc3339_invalid() {
        let result = parse_rfc3339("expires_at", Some("not-a-date".into()));
        assert!(result.is_err());
        let (status, msg) = result.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(msg.contains("expires_at"));
    }

    #[test]
    fn generate_client_secret_is_unique_and_correct_length() {
        let s1 = generate_client_secret();
        let s2 = generate_client_secret();
        assert_ne!(s1, s2);
        // 32 bytes base64url-no-pad = 43 chars
        assert_eq!(s1.len(), 43);
    }

    #[test]
    fn require_admin_rejects_non_admin() {
        use crate::auth::{AuthContext, AuthRole};
        // Write role passes authorize_write but fails is_admin
        let auth = AuthContext {
            subject: Some("user".into()),
            role: AuthRole::Write,
            namespace: None,
            consumer_name: None,
        };
        let result = require_admin(&auth);
        assert!(result.is_err());
        let (status, _) = result.unwrap_err();
        assert_eq!(status, StatusCode::FORBIDDEN);
    }

    #[test]
    fn build_client_response_populates_fields() {
        let mut metadata = HashMap::new();
        metadata.insert(METADATA_TEMPLATE_ID.into(), Value::String("tmpl-1".into()));
        metadata.insert(METADATA_DISPLAY_NAME.into(), Value::String("My Client".into()));
        metadata.insert(METADATA_TOKEN_TTL_SECS.into(), Value::Number(7200.into()));
        metadata.insert(METADATA_DESCRIPTION.into(), Value::String("A test client".into()));

        let now = Utc::now();
        let resp = build_client_response(
            "client-123".into(),
            &metadata,
            now,
            now,
            now,
            None,
            None,
        );

        assert_eq!(resp.client_id, "client-123");
        assert_eq!(resp.name, "My Client");
        assert_eq!(resp.template_id, "tmpl-1");
        assert_eq!(resp.token_ttl_seconds, 7200);
        assert_eq!(resp.description, Some("A test client".into()));
        assert!(resp.expires_at.is_none());
        assert!(resp.revoked_at.is_none());
    }

    #[test]
    fn credential_invalid_for_token_flags_expired() {
        let mut cred = StoredCredential::new(
            Uuid::now_v7(),
            "client",
            "oauth_client_secret",
            "ciphertext".into(),
            "derivation".to_string(),
        );
        let now = Utc::now();
        cred.expires_at = Some(now - Duration::seconds(1));
        assert!(credential_invalid_for_token(&cred, now));
    }

    #[test]
    fn credential_invalid_for_token_allows_valid() {
        let mut cred = StoredCredential::new(
            Uuid::now_v7(),
            "client",
            "oauth_client_secret",
            "ciphertext".into(),
            "derivation".to_string(),
        );
        let now = Utc::now();
        cred.expires_at = Some(now + Duration::seconds(60));
        assert!(!credential_invalid_for_token(&cred, now));
    }

    #[test]
    fn credential_invalid_for_token_flags_archived() {
        let mut cred = StoredCredential::new(
            Uuid::now_v7(),
            "client",
            "oauth_client_secret",
            "ciphertext".into(),
            "derivation".to_string(),
        );
        cred.archived_at = Some(Utc::now());
        assert!(credential_invalid_for_token(&cred, Utc::now()));
    }

    #[tokio::test]
    async fn revoke_oauth_access_keys_revokes_matching() {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        let mut config = EngineConfig {
            data_dir: temp_dir.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        config.embedding.provider = "noop".into();
        let engine = HelixEngine::init(config)
            .await
            .expect("engine should init");

        let templates = engine
            .list_permission_templates(10, 0)
            .await
            .expect("templates should load");
        let template = templates
            .iter()
            .find(|t| t.name == "Assistant")
            .or_else(|| templates.first())
            .expect("template exists");

        let (key_a, _token_a) = engine
            .create_access_key(template.id, Some("oauth:client-123".into()), None)
            .await
            .expect("key should be created");
        let (key_b, _token_b) = engine
            .create_access_key(template.id, Some("other".into()), None)
            .await
            .expect("key should be created");

        let revoked = revoke_oauth_access_keys(&engine, "client-123")
            .await
            .expect("revocation should succeed");
        assert_eq!(revoked, 1);

        let keys = engine.list_access_keys().await.expect("keys should load");
        let key_a = keys.iter().find(|k| k.id == key_a.id).expect("key A present");
        let key_b = keys.iter().find(|k| k.id == key_b.id).expect("key B present");
        assert!(key_a.revoked_at.is_some());
        assert!(key_b.revoked_at.is_none());
    }
}
