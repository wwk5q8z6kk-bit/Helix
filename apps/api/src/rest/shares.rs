use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    response::{Html, IntoResponse, Response},
    Extension, Json,
};
use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_core::{KnowledgeNode, NodeKind, NodeStore, PublicShare, ShareStore};

use crate::auth::{authorize_namespace, authorize_read, authorize_write, scoped_namespace, AuthContext};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct CreatePublicShareRequest {
    pub node_id: String,
    pub expires_at: Option<String>,
}

#[derive(Serialize)]
pub struct CreatePublicShareResponse {
    pub id: String,
    pub node_id: String,
    pub token: String,
    pub url: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Deserialize)]
pub struct ListPublicSharesParams {
    pub node_id: Option<String>,
    pub include_revoked: Option<bool>,
    pub namespace: Option<String>,
}

#[derive(Serialize)]
pub struct PublicShareSummary {
    pub id: String,
    pub node_id: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
}

#[derive(Serialize)]
pub struct PublicShareNode {
    pub id: String,
    pub kind: NodeKind,
    pub title: Option<String>,
    pub content: String,
    pub source: Option<String>,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Serialize)]
pub struct PublicShareContentResponse {
    pub share_id: String,
    pub node: PublicShareNode,
    pub expires_at: Option<DateTime<Utc>>,
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

fn share_summary(share: PublicShare) -> PublicShareSummary {
    PublicShareSummary {
        id: share.id.to_string(),
        node_id: share.node_id.to_string(),
        created_at: share.created_at,
        expires_at: share.expires_at,
        revoked_at: share.revoked_at,
    }
}

fn share_node(node: KnowledgeNode) -> PublicShareNode {
    PublicShareNode {
        id: node.id.to_string(),
        kind: node.kind,
        title: node.title,
        content: node.content,
        source: node.source,
        tags: node.tags,
        created_at: node.temporal.created_at,
        updated_at: node.temporal.updated_at,
    }
}

fn accepts_html(headers: &HeaderMap) -> bool {
    let accept = headers
        .get(header::ACCEPT)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    accept.contains("text/html") || accept.contains("application/xhtml+xml")
}

fn escape_html(input: &str) -> String {
    let mut escaped = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#39;"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn format_ts(ts: DateTime<Utc>) -> String {
    ts.to_rfc3339_opts(SecondsFormat::Secs, true)
}

fn html_response(status: StatusCode, body: String) -> Response {
    let mut response = Html(body).into_response();
    *response.status_mut() = status;
    let headers = response.headers_mut();
    headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    headers.insert(
        header::PRAGMA,
        HeaderValue::from_static("no-cache"),
    );
    headers.insert(
        header::HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );
    headers.insert(
        header::HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("no-referrer"),
    );
    headers.insert(
        header::HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static("default-src 'none'; style-src 'unsafe-inline'; img-src data:"),
    );
    headers.insert(
        header::HeaderName::from_static("x-robots-tag"),
        HeaderValue::from_static("noindex, nofollow, noarchive"),
    );
    response
}

fn render_share_html(share: &PublicShare, node: &KnowledgeNode) -> String {
    let title = escape_html(node.title.as_deref().unwrap_or("Untitled"));
    let content = escape_html(&node.content);
    let kind = node.kind.as_str();
    let created_at = format_ts(node.temporal.created_at);
    let updated_at = format_ts(node.temporal.updated_at);
    let shared_at = format_ts(share.created_at);
    let expires_at = share
        .expires_at
        .map(format_ts)
        .unwrap_or_else(|| "never".to_string());
    let source = node
        .source
        .as_deref()
        .map(escape_html)
        .unwrap_or_else(|| "—".to_string());
    let tags = if node.tags.is_empty() {
        "—".to_string()
    } else {
        node.tags
            .iter()
            .map(|tag| format!("<span class=\"tag\">{}</span>", escape_html(tag)))
            .collect::<Vec<_>>()
            .join(" ")
    };

    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="robots" content="noindex,nofollow" />
  <title>{title} — Helix Share</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1220;
      --card: #111827;
      --border: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --accent: #38bdf8;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #111827 0%, #0b1220 60%);
      color: var(--text);
    }}
    main {{
      max-width: 840px;
      margin: 40px auto;
      padding: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--accent);
      background: rgba(56,189,248,0.12);
      padding: 6px 10px;
      border-radius: 999px;
    }}
    h1 {{
      margin: 16px 0 8px;
      font-size: 28px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 20px;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    .tags {{
      margin: 12px 0 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 11px;
      color: var(--muted);
    }}
    .tag {{
      border: 1px solid var(--border);
      padding: 4px 8px;
      border-radius: 999px;
    }}
    .content {{
      white-space: pre-wrap;
      word-break: break-word;
      background: rgba(15,23,42,0.8);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      line-height: 1.6;
      font-size: 14px;
      color: #f8fafc;
    }}
    footer {{
      margin-top: 16px;
      font-size: 11px;
      color: var(--muted);
    }}
  </style>
</head>
<body>
  <main>
    <div class="card">
      <span class="badge">Public read-only share</span>
      <h1>{title}</h1>
      <div class="meta">
        <span>Kind: {kind}</span>
        <span>Shared: {shared_at}</span>
        <span>Expires: {expires_at}</span>
      </div>
      <div class="meta">
        <span>Created: {created_at}</span>
        <span>Updated: {updated_at}</span>
        <span>Source: {source}</span>
      </div>
      <div class="tags">{tags}</div>
      <div class="content">{content}</div>
      <footer>Generated by Helix · This link is read-only.</footer>
    </div>
  </main>
</body>
</html>"#
    )
}

fn render_error_html(title: &str, message: &str) -> String {
    let title = escape_html(title);
    let message = escape_html(message);
    format!(
        r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="robots" content="noindex,nofollow" />
  <title>{title} — Helix Share</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0b1220;
      --card: #111827;
      --border: #1f2937;
      --text: #e5e7eb;
      --muted: #94a3b8;
    }}
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    main {{
      max-width: 680px;
      margin: 80px auto;
      padding: 24px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 24px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <main>
    <div class="card">
      <h1>{title}</h1>
      <p>{message}</p>
    </div>
  </main>
</body>
</html>"#
    )
}

/// POST /api/v1/shares
pub async fn create_public_share(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Json(req): Json<CreatePublicShareRequest>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }

    let node_id = match Uuid::parse_str(&req.node_id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid node_id"),
            )
                .into_response()
        }
    };

    let node = match state.engine.store.nodes.get(node_id).await {
        Ok(Some(node)) => node,
        Ok(None) => {
            return (StatusCode::NOT_FOUND, err_json("node not found")).into_response()
        }
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

    let expires_at = match req.expires_at {
        Some(raw) => match DateTime::parse_from_rfc3339(&raw) {
            Ok(dt) => Some(dt.with_timezone(&Utc)),
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    err_json("expires_at must be RFC3339 timestamp"),
                )
                    .into_response()
            }
        },
        None => None,
    };

    match state.engine.create_public_share(node_id, expires_at).await {
        Ok((share, token)) => {
            let url = format!("/public/shares/{token}");
            Json(CreatePublicShareResponse {
                id: share.id.to_string(),
                node_id: share.node_id.to_string(),
                token,
                url,
                created_at: share.created_at,
                expires_at: share.expires_at,
            })
            .into_response()
        }
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to create share: {err}")),
        )
            .into_response(),
    }
}

/// GET /api/v1/shares
pub async fn list_public_shares(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Query(params): Query<ListPublicSharesParams>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_read(&auth) {
        return (status, err_json(message)).into_response();
    }

    let include_revoked = params.include_revoked.unwrap_or(false);
    let node_id = match params.node_id {
        Some(id) => match Uuid::parse_str(&id) {
            Ok(id) => Some(id),
            Err(_) => {
                return (
                    StatusCode::BAD_REQUEST,
                    err_json("invalid node_id"),
                )
                    .into_response()
            }
        },
        None => None,
    };

    let shares = match state.engine.list_public_shares(node_id, include_revoked).await {
        Ok(shares) => shares,
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to list shares: {err}")),
            )
                .into_response()
        }
    };

    let requested_namespace = params
        .namespace
        .as_deref()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let namespace_filter = match scoped_namespace(&auth, requested_namespace) {
        Ok(namespace) => namespace,
        Err((status, message)) => return (status, err_json(message)).into_response(),
    };

    let mut filtered = Vec::new();
    for share in shares {
        let node = match state.engine.store.nodes.get(share.node_id).await {
            Ok(Some(node)) => node,
            _ => continue,
        };

        if let Some(ref namespace) = namespace_filter {
            if node.namespace != *namespace {
                continue;
            }
        }

        if authorize_namespace(&auth, &node.namespace).is_err() {
            continue;
        }

        filtered.push(share_summary(share));
    }

    Json(filtered).into_response()
}

/// DELETE /api/v1/shares/:id
pub async fn revoke_public_share(
    State(state): State<Arc<AppState>>,
    Extension(auth): Extension<AuthContext>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    if let Err((status, message)) = authorize_write(&auth) {
        return (status, err_json(message)).into_response();
    }

    let share_id = match Uuid::parse_str(&id) {
        Ok(id) => id,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                err_json("invalid share id"),
            )
                .into_response()
        }
    };

    let share = match state.engine.store.nodes.get_public_share(share_id).await {
        Ok(Some(share)) => share,
        Ok(None) => {
            return (StatusCode::NOT_FOUND, err_json("share not found")).into_response()
        }
        Err(err) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                err_json(format!("failed to load share: {err}")),
            )
                .into_response()
        }
    };

    let node = match state.engine.store.nodes.get(share.node_id).await {
        Ok(Some(node)) => node,
        _ => {
            return (StatusCode::NOT_FOUND, err_json("node not found")).into_response()
        }
    };

    if let Err((status, message)) = authorize_namespace(&auth, &node.namespace) {
        return (status, err_json(message)).into_response();
    }

    match state.engine.revoke_public_share(share_id).await {
        Ok(true) => {
            // Return post-revoke state when possible so clients see revoked_at immediately.
            if let Ok(Some(updated_share)) = state.engine.store.nodes.get_public_share(share_id).await {
                Json(share_summary(updated_share)).into_response()
            } else {
                // Fallback: mark the preloaded share as revoked for response consistency.
                let mut revoked_share = share;
                if revoked_share.revoked_at.is_none() {
                    revoked_share.revoked_at = Some(Utc::now());
                }
                Json(share_summary(revoked_share)).into_response()
            }
        }
        Ok(false) => (StatusCode::NOT_FOUND, err_json("share not found")).into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            err_json(format!("failed to revoke share: {err}")),
        )
            .into_response(),
    }
}

/// GET /public/shares/:token
pub async fn get_public_share(
    State(state): State<Arc<AppState>>,
    Path(token): Path<String>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let wants_html = accepts_html(&headers);
    match state.engine.resolve_public_share(&token).await {
        Ok(Some((share, node))) => {
            if wants_html {
                html_response(StatusCode::OK, render_share_html(&share, &node))
            } else {
                Json(PublicShareContentResponse {
                    share_id: share.id.to_string(),
                    node: share_node(node),
                    expires_at: share.expires_at,
                })
                .into_response()
            }
        }
        Ok(None) => {
            if wants_html {
                html_response(
                    StatusCode::NOT_FOUND,
                    render_error_html(
                        "Share not found",
                        "This link is invalid, expired, or has been revoked.",
                    ),
                )
            } else {
                (StatusCode::NOT_FOUND, err_json("share not found")).into_response()
            }
        }
        Err(err) => {
            if wants_html {
                html_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    render_error_html("Server error", "Failed to load this share."),
                )
            } else {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    err_json(format!("failed to resolve share: {err}")),
                )
                    .into_response()
            }
        }
    }
}
