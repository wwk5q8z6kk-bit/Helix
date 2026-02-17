use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Extension, Json,
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use hx_engine::notifications::{Notification, NotificationChannelType, Severity};
use hx_engine::notifications::alerts::{AlertCondition, AlertRule, QuietHours};

use crate::auth::AuthContext;
use crate::state::AppState;

// --- DTOs ---

#[derive(Serialize)]
pub struct NotificationResponse {
    pub id: String,
    pub title: String,
    pub body: String,
    pub severity: String,
    pub source: String,
    pub created_at: String,
    pub read: bool,
}

impl From<&Notification> for NotificationResponse {
    fn from(n: &Notification) -> Self {
        Self {
            id: n.id.to_string(),
            title: n.title.clone(),
            body: n.body.clone(),
            severity: n.severity.to_string(),
            source: n.source.clone(),
            created_at: n.created_at.to_rfc3339(),
            read: n.read,
        }
    }
}

#[derive(Deserialize)]
pub struct ListNotificationsQuery {
    pub severity: Option<String>,
    pub read: Option<bool>,
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

#[derive(Serialize)]
pub struct AlertRuleResponse {
    pub id: String,
    pub name: String,
    pub enabled: bool,
    pub condition: serde_json::Value,
    pub channels: Vec<String>,
    pub severity: String,
    pub cooldown_secs: u64,
    pub quiet_hours: Option<QuietHoursResponse>,
    pub last_triggered: Option<String>,
}

#[derive(Serialize)]
pub struct QuietHoursResponse {
    pub start_hour: u8,
    pub end_hour: u8,
    pub timezone: String,
}

impl From<&AlertRule> for AlertRuleResponse {
    fn from(r: &AlertRule) -> Self {
        Self {
            id: r.id.to_string(),
            name: r.name.clone(),
            enabled: r.enabled,
            condition: serde_json::to_value(&r.condition).unwrap_or(serde_json::Value::Null),
            channels: r.channels.iter().map(|c| c.to_string()).collect(),
            severity: r.severity.to_string(),
            cooldown_secs: r.cooldown_secs,
            quiet_hours: r.quiet_hours.as_ref().map(|qh| QuietHoursResponse {
                start_hour: qh.start_hour,
                end_hour: qh.end_hour,
                timezone: qh.timezone.clone(),
            }),
            last_triggered: r.last_triggered.map(|dt| dt.to_rfc3339()),
        }
    }
}

#[derive(Deserialize)]
pub struct CreateAlertRuleDto {
    pub name: String,
    pub condition: AlertCondition,
    pub severity: String,
    #[serde(default)]
    pub channels: Vec<String>,
    #[serde(default = "default_cooldown")]
    pub cooldown_secs: u64,
    pub quiet_hours: Option<CreateQuietHoursDto>,
}

fn default_cooldown() -> u64 {
    300
}

#[derive(Deserialize)]
pub struct CreateQuietHoursDto {
    pub start_hour: u8,
    pub end_hour: u8,
    #[serde(default = "default_timezone")]
    pub timezone: String,
}

fn default_timezone() -> String {
    "UTC".to_string()
}

#[derive(Deserialize)]
pub struct UpdateAlertRuleDto {
    pub name: Option<String>,
    pub enabled: Option<bool>,
    pub condition: Option<AlertCondition>,
    pub severity: Option<String>,
    pub channels: Option<Vec<String>>,
    pub cooldown_secs: Option<u64>,
    pub quiet_hours: Option<CreateQuietHoursDto>,
}

fn authorize_admin(auth: &AuthContext) -> Result<(), (StatusCode, String)> {
    if auth.is_admin() {
        Ok(())
    } else {
        Err((StatusCode::FORBIDDEN, "admin permission required".into()))
    }
}

fn parse_uuid(s: &str) -> Result<Uuid, (StatusCode, String)> {
    Uuid::parse_str(s).map_err(|_| (StatusCode::BAD_REQUEST, "invalid uuid".to_string()))
}

fn parse_severity(s: &str) -> Result<Severity, (StatusCode, String)> {
    match s {
        "info" => Ok(Severity::Info),
        "warning" => Ok(Severity::Warning),
        "error" => Ok(Severity::Error),
        "critical" => Ok(Severity::Critical),
        _ => Err((
            StatusCode::BAD_REQUEST,
            format!("invalid severity: {s}; expected info, warning, error, or critical"),
        )),
    }
}

fn parse_channel_types(channels: &[String]) -> Result<Vec<NotificationChannelType>, (StatusCode, String)> {
    channels
        .iter()
        .map(|c| match c.as_str() {
            "in_app" => Ok(NotificationChannelType::InApp),
            "email" => Ok(NotificationChannelType::Email),
            "webhook" => Ok(NotificationChannelType::Webhook),
            "slack" => Ok(NotificationChannelType::Slack),
            _ => Err((
                StatusCode::BAD_REQUEST,
                format!("invalid channel type: {c}"),
            )),
        })
        .collect()
}

// --- Handlers ---

/// GET /api/v1/notifications
pub async fn list_notifications(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Query(query): Query<ListNotificationsQuery>,
) -> Result<Json<Vec<NotificationResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let severity_filter = query
        .severity
        .as_ref()
        .map(|s| parse_severity(s))
        .transpose()?;

    let notifications = state
        .in_app_channel
        .list(severity_filter, query.read, query.limit)
        .await;
    let response: Vec<NotificationResponse> = notifications.iter().map(NotificationResponse::from).collect();
    Ok(Json(response))
}

/// GET /api/v1/notifications/:id
pub async fn get_notification(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<NotificationResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let uuid = parse_uuid(&id)?;

    let notif = state
        .in_app_channel
        .get(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "notification not found".to_string()))?;

    Ok(Json(NotificationResponse::from(&notif)))
}

/// POST /api/v1/notifications/:id/read
pub async fn mark_notification_read(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let uuid = parse_uuid(&id)?;

    let found = state.in_app_channel.mark_read(uuid).await;
    if !found {
        return Err((StatusCode::NOT_FOUND, "notification not found".to_string()));
    }

    Ok(Json(serde_json::json!({ "status": "marked_read" })))
}

/// GET /api/v1/notifications/alerts
pub async fn list_alert_rules(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
) -> Result<Json<Vec<AlertRuleResponse>>, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let rules = state.notification_router.alert_store().list().await;
    let response: Vec<AlertRuleResponse> = rules.iter().map(AlertRuleResponse::from).collect();
    Ok(Json(response))
}

/// POST /api/v1/notifications/alerts
pub async fn create_alert_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateAlertRuleDto>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;

    let severity = parse_severity(&req.severity)?;
    let channels = if req.channels.is_empty() {
        vec![NotificationChannelType::InApp]
    } else {
        parse_channel_types(&req.channels)?
    };

    let mut rule = AlertRule::new(req.name, req.condition, severity)
        .with_cooldown(req.cooldown_secs)
        .with_channels(channels);

    if let Some(qh) = req.quiet_hours {
        rule = rule.with_quiet_hours(QuietHours {
            start_hour: qh.start_hour,
            end_hour: qh.end_hour,
            timezone: qh.timezone,
        });
    }

    let resp = AlertRuleResponse::from(&rule);
    state.notification_router.alert_store().add(rule).await;
    Ok((StatusCode::CREATED, Json(resp)).into_response())
}

/// PUT /api/v1/notifications/alerts/:id
pub async fn update_alert_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
    Json(req): Json<UpdateAlertRuleDto>,
) -> Result<Json<AlertRuleResponse>, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let uuid = parse_uuid(&id)?;

    let store = state.notification_router.alert_store();
    let mut rule = store
        .get(uuid)
        .await
        .ok_or((StatusCode::NOT_FOUND, "alert rule not found".to_string()))?;

    // Apply partial updates
    if let Some(name) = req.name {
        rule.name = name;
    }
    if let Some(enabled) = req.enabled {
        rule.enabled = enabled;
    }
    if let Some(condition) = req.condition {
        rule.condition = condition;
    }
    if let Some(severity_str) = req.severity {
        rule.severity = parse_severity(&severity_str)?;
    }
    if let Some(channels) = req.channels {
        rule.channels = parse_channel_types(&channels)?;
    }
    if let Some(cooldown) = req.cooldown_secs {
        rule.cooldown_secs = cooldown;
    }
    if let Some(qh) = req.quiet_hours {
        rule.quiet_hours = Some(QuietHours {
            start_hour: qh.start_hour,
            end_hour: qh.end_hour,
            timezone: qh.timezone,
        });
    }

    store
        .update(uuid, rule.clone())
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(AlertRuleResponse::from(&rule)))
}

/// DELETE /api/v1/notifications/alerts/:id
pub async fn delete_alert_rule(
    Extension(auth): Extension<AuthContext>,
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    authorize_admin(&auth)?;
    let uuid = parse_uuid(&id)?;

    let removed = state.notification_router.alert_store().remove(uuid).await;
    if !removed {
        return Ok((StatusCode::NOT_FOUND, "alert rule not found".to_string()).into_response());
    }

    Ok(StatusCode::NO_CONTENT.into_response())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_engine::notifications::alerts::AlertCondition;

    #[test]
    fn notification_response_from_notification() {
        let notif = Notification::new("Alert", "Something failed", Severity::Error)
            .with_source("engine");
        let resp = NotificationResponse::from(&notif);
        assert_eq!(resp.title, "Alert");
        assert_eq!(resp.severity, "error");
        assert_eq!(resp.source, "engine");
        assert!(!resp.read);
    }

    #[test]
    fn alert_rule_response_from_rule() {
        let rule = AlertRule::new(
            "test-rule",
            AlertCondition::JobFailed { job_type: None },
            Severity::Warning,
        )
        .with_cooldown(600);

        let resp = AlertRuleResponse::from(&rule);
        assert_eq!(resp.name, "test-rule");
        assert_eq!(resp.severity, "warning");
        assert_eq!(resp.cooldown_secs, 600);
        assert!(resp.enabled);
        assert!(resp.last_triggered.is_none());
    }

    #[test]
    fn parse_severity_valid() {
        assert_eq!(parse_severity("info").unwrap(), Severity::Info);
        assert_eq!(parse_severity("warning").unwrap(), Severity::Warning);
        assert_eq!(parse_severity("error").unwrap(), Severity::Error);
        assert_eq!(parse_severity("critical").unwrap(), Severity::Critical);
    }

    #[test]
    fn parse_severity_invalid() {
        assert!(parse_severity("unknown").is_err());
    }

    #[test]
    fn parse_channel_types_valid() {
        let types = parse_channel_types(&["in_app".to_string(), "slack".to_string()]).unwrap();
        assert_eq!(types.len(), 2);
        assert_eq!(types[0], NotificationChannelType::InApp);
        assert_eq!(types[1], NotificationChannelType::Slack);
    }

    #[test]
    fn parse_channel_types_invalid() {
        assert!(parse_channel_types(&["invalid".to_string()]).is_err());
    }
}
