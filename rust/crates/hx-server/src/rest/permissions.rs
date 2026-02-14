use std::str::FromStr;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{AccessKey, NodeKind, PermissionTemplate, PermissionTier};

use crate::auth::{authorize_read, authorize_write, AuthContext};
use crate::state::AppState;
use crate::validation::{
    validate_list_limit, validate_namespace_input, validate_tags_input, validate_text_input,
};

#[derive(Deserialize)]
pub struct PermissionTemplateRequest {
    pub name: String,
    pub description: Option<String>,
    pub tier: String,
    pub scope_namespace: Option<String>,
    pub scope_tags: Option<Vec<String>>,
    pub allow_kinds: Option<Vec<String>>,
    pub allow_actions: Option<Vec<String>>,
}

#[derive(Serialize)]
pub struct PermissionTemplateResponse {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub tier: String,
    pub scope_namespace: Option<String>,
    pub scope_tags: Vec<String>,
    pub allow_kinds: Vec<String>,
    pub allow_actions: Vec<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Deserialize)]
pub struct PermissionTemplateListQuery {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
}

#[derive(Deserialize)]
pub struct AccessKeyCreateRequest {
    pub template_id: String,
    pub name: Option<String>,
    pub expires_at: Option<String>,
}

#[derive(Serialize)]
pub struct AccessKeyResponse {
    pub id: String,
    pub name: Option<String>,
    pub template_id: String,
    pub template_name: Option<String>,
    pub created_at: String,
    pub last_used_at: Option<String>,
    pub expires_at: Option<String>,
    pub revoked_at: Option<String>,
}

#[derive(Serialize)]
pub struct AccessKeyCreateResponse {
    pub token: String,
    pub access_key: AccessKeyResponse,
}

fn template_to_response(template: PermissionTemplate) -> PermissionTemplateResponse {
    PermissionTemplateResponse {
        id: template.id.to_string(),
        name: template.name,
        description: template.description,
        tier: template.tier.to_string(),
        scope_namespace: template.scope_namespace,
        scope_tags: template.scope_tags,
        allow_kinds: template
            .allow_kinds
            .into_iter()
            .map(|kind| kind.as_str().to_string())
            .collect(),
        allow_actions: template.allow_actions,
        created_at: template.created_at.to_rfc3339(),
        updated_at: template.updated_at.to_rfc3339(),
    }
}

fn access_key_to_response(key: AccessKey, template_name: Option<String>) -> AccessKeyResponse {
    AccessKeyResponse {
        id: key.id.to_string(),
        name: key.name,
        template_id: key.template_id.to_string(),
        template_name,
        created_at: key.created_at.to_rfc3339(),
        last_used_at: key.last_used_at.map(|dt| dt.to_rfc3339()),
        expires_at: key.expires_at.map(|dt| dt.to_rfc3339()),
        revoked_at: key.revoked_at.map(|dt| dt.to_rfc3339()),
    }
}

fn parse_template_request(
    request: PermissionTemplateRequest,
) -> Result<
    (
        String,
        Option<String>,
        PermissionTier,
        Option<String>,
        Vec<String>,
        Vec<NodeKind>,
        Vec<String>,
    ),
    (StatusCode, String),
> {
    validate_text_input("name", &request.name).map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
    validate_namespace_input(request.scope_namespace.as_deref())
        .map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
    let scope_tags = request.scope_tags.unwrap_or_default();
    validate_tags_input(&scope_tags).map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;

    let tier =
        PermissionTier::from_str(&request.tier).map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;

    let mut allow_kinds = Vec::new();
    if let Some(kinds) = request.allow_kinds {
        for kind in kinds {
            let parsed: NodeKind = kind
                .parse()
                .map_err(|err: String| (StatusCode::BAD_REQUEST, err.to_string()))?;
            allow_kinds.push(parsed);
        }
    }

    let allow_actions = request.allow_actions.unwrap_or_default();

    Ok((
        request.name,
        request.description,
        tier,
        request.scope_namespace,
        scope_tags,
        allow_kinds,
        allow_actions,
    ))
}

pub async fn list_permission_templates(
    Extension(auth): Extension<AuthContext>,
    State(state): State<std::sync::Arc<AppState>>,
    Query(params): Query<PermissionTemplateListQuery>,
) -> Result<Json<Vec<PermissionTemplateResponse>>, (StatusCode, String)> {
    authorize_read(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin role required".into()));
    }

    let limit = params.limit.unwrap_or(200).min(500);
    validate_list_limit(limit).map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
    let offset = params.offset.unwrap_or(0);

    let templates = state
        .engine
        .list_permission_templates(limit, offset)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    Ok(Json(
        templates.into_iter().map(template_to_response).collect(),
    ))
}

pub async fn create_permission_template(
    Extension(auth): Extension<AuthContext>,
    State(state): State<std::sync::Arc<AppState>>,
    Json(payload): Json<PermissionTemplateRequest>,
) -> Result<Json<PermissionTemplateResponse>, (StatusCode, String)> {
    authorize_write(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin role required".into()));
    }

    let (name, description, tier, scope_namespace, scope_tags, allow_kinds, allow_actions) =
        parse_template_request(payload)?;

    let template = state
        .engine
        .create_permission_template(
            name,
            description,
            tier,
            scope_namespace,
            scope_tags,
            allow_kinds,
            allow_actions,
        )
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    Ok(Json(template_to_response(template)))
}

pub async fn update_permission_template(
    Extension(auth): Extension<AuthContext>,
    State(state): State<std::sync::Arc<AppState>>,
    Path(template_id): Path<String>,
    Json(payload): Json<PermissionTemplateRequest>,
) -> Result<Json<PermissionTemplateResponse>, (StatusCode, String)> {
    authorize_write(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin role required".into()));
    }

    let template_id = Uuid::parse_str(&template_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid template id".into()))?;

    let (name, description, tier, scope_namespace, scope_tags, allow_kinds, allow_actions) =
        parse_template_request(payload)?;

    let updated = state
        .engine
        .update_permission_template(
            template_id,
            name,
            description,
            tier,
            scope_namespace,
            scope_tags,
            allow_kinds,
            allow_actions,
        )
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    let Some(updated) = updated else {
        return Err((StatusCode::NOT_FOUND, "template not found".into()));
    };

    Ok(Json(template_to_response(updated)))
}

pub async fn delete_permission_template(
    Extension(auth): Extension<AuthContext>,
    State(state): State<std::sync::Arc<AppState>>,
    Path(template_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    authorize_write(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin role required".into()));
    }

    let template_id = Uuid::parse_str(&template_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid template id".into()))?;

    let deleted = state
        .engine
        .delete_permission_template(template_id)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    if deleted {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "template not found".into()))
    }
}

pub async fn create_access_key(
    Extension(auth): Extension<AuthContext>,
    State(state): State<std::sync::Arc<AppState>>,
    Json(payload): Json<AccessKeyCreateRequest>,
) -> Result<Json<AccessKeyCreateResponse>, (StatusCode, String)> {
    authorize_write(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin role required".into()));
    }

    let template_id = Uuid::parse_str(&payload.template_id)
        .map_err(|_| (StatusCode::BAD_REQUEST, "invalid template id".into()))?;

    let expires_at = match payload.expires_at.as_deref() {
        Some(raw) => Some(
            DateTime::parse_from_rfc3339(raw)
                .map_err(|_| (StatusCode::BAD_REQUEST, "invalid expires_at".into()))?
                .with_timezone(&Utc),
        ),
        None => None,
    };

    let (key, token) = state
        .engine
        .create_access_key(template_id, payload.name, expires_at)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    let template = state
        .engine
        .list_permission_templates(500, 0)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?
        .into_iter()
        .find(|item| item.id == key.template_id);

    let response = AccessKeyCreateResponse {
        token,
        access_key: access_key_to_response(key, template.map(|t| t.name)),
    };

    Ok(Json(response))
}

pub async fn list_access_keys(
    Extension(auth): Extension<AuthContext>,
    State(state): State<std::sync::Arc<AppState>>,
) -> Result<Json<Vec<AccessKeyResponse>>, (StatusCode, String)> {
    authorize_read(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin role required".into()));
    }

    let templates = state
        .engine
        .list_permission_templates(500, 0)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    let template_lookup: std::collections::HashMap<Uuid, String> = templates
        .into_iter()
        .map(|template| (template.id, template.name))
        .collect();

    let keys = state
        .engine
        .list_access_keys()
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    Ok(Json(
        keys.into_iter()
            .map(|key| {
                let template_name = template_lookup.get(&key.template_id).cloned();
                access_key_to_response(key, template_name)
            })
            .collect(),
    ))
}

pub async fn revoke_access_key(
    Extension(auth): Extension<AuthContext>,
    State(state): State<std::sync::Arc<AppState>>,
    Path(key_id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    authorize_write(&auth)?;
    if !auth.is_admin() {
        return Err((StatusCode::FORBIDDEN, "admin role required".into()));
    }

    let key_id =
        Uuid::parse_str(&key_id).map_err(|_| (StatusCode::BAD_REQUEST, "invalid key id".into()))?;

    let revoked = state
        .engine
        .revoke_access_key(key_id)
        .await
        .map_err(|err| (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()))?;

    if revoked {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "access key not found".into()))
    }
}
