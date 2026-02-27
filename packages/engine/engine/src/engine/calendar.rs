use super::HelixEngine;
use chrono::{DateTime, Utc};
use hx_core::{AdapterPollStore, KnowledgeNode, MvResult, NodeKind, NodeStore, HxError};
use reqwest::StatusCode as HttpStatusCode;
use url::form_urlencoded::byte_serialize;

#[derive(Debug, Clone, serde::Serialize)]
pub struct GoogleCalendarSyncReport {
    pub calendar_id: String,
    pub fetched: usize,
    pub created: usize,
    pub updated: usize,
    pub deleted: usize,
    pub skipped: usize,
    pub exported_created: usize,
    pub exported_updated: usize,
    pub next_sync_token: Option<String>,
}

#[derive(Debug)]
enum GoogleCalendarFetchError {
    SyncTokenExpired,
    RequestFailed(String),
}

#[derive(Debug, serde::Deserialize)]
struct GoogleTokenResponse {
    access_token: String,
}

#[derive(Debug, serde::Deserialize)]
struct GoogleEventsResponse {
    items: Option<Vec<GoogleEvent>>,
    #[serde(rename = "nextPageToken")]
    next_page_token: Option<String>,
    #[serde(rename = "nextSyncToken")]
    next_sync_token: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct GoogleEvent {
    id: Option<String>,
    summary: Option<String>,
    description: Option<String>,
    status: Option<String>,
    updated: Option<String>,
    #[serde(rename = "htmlLink")]
    html_link: Option<String>,
    start: Option<GoogleEventTime>,
    end: Option<GoogleEventTime>,
}

#[derive(Debug, serde::Deserialize)]
struct GoogleEventTime {
    #[serde(rename = "dateTime")]
    date_time: Option<String>,
    date: Option<String>,
}

impl HelixEngine {
    pub async fn sync_google_calendar(&self) -> MvResult<GoogleCalendarSyncReport> {
        let config = &self.config.google_calendar;
        if !config.enabled {
            return Err(HxError::InvalidInput(
                "google calendar sync is disabled".to_string(),
            ));
        }

        let client_id = config
            .client_id
            .as_ref()
            .ok_or_else(|| HxError::InvalidInput("google calendar client_id missing".into()))?;
        let client_secret = config
            .client_secret
            .as_ref()
            .ok_or_else(|| HxError::InvalidInput("google calendar client_secret missing".into()))?;
        let refresh_token = config
            .refresh_token
            .as_ref()
            .ok_or_else(|| HxError::InvalidInput("google calendar refresh_token missing".into()))?;

        let calendar_id = config.calendar_id.trim();
        if calendar_id.is_empty() {
            return Err(HxError::InvalidInput(
                "google calendar_id must not be empty".into(),
            ));
        }

        let access_token =
            google_refresh_access_token(client_id, client_secret, refresh_token).await?;
        let adapter_name = format!("google-calendar:{calendar_id}");
        let existing_sync = self
            .store
            .nodes
            .get_poll_state(&adapter_name)
            .await?
            .and_then(|state| {
                let cursor = state.cursor.trim().to_string();
                if cursor.is_empty() {
                    None
                } else {
                    Some(cursor)
                }
            });

        let mut report = GoogleCalendarSyncReport {
            calendar_id: calendar_id.to_string(),
            fetched: 0,
            created: 0,
            updated: 0,
            deleted: 0,
            skipped: 0,
            exported_created: 0,
            exported_updated: 0,
            next_sync_token: None,
        };

        let mut sync_token = existing_sync;
        for attempt in 0..2 {
            match google_list_events(&access_token, calendar_id, config, sync_token.as_deref())
                .await
            {
                Ok((events, next_sync_token)) => {
                    report.fetched = events.len();
                    report.next_sync_token = next_sync_token.clone();
                    let mut created = 0usize;
                    let mut updated = 0usize;
                    let mut deleted = 0usize;
                    let mut skipped = 0usize;

                    if config.import_events {
                        for event in events {
                            let Some(event_id) = event.id.as_ref() else {
                                skipped += 1;
                                continue;
                            };

                            if matches!(event.status.as_deref(), Some("cancelled")) {
                                if let Some(existing) = self
                                    .store
                                    .nodes
                                    .find_by_source(&event_source(calendar_id, event_id))
                                    .await?
                                {
                                    let _ = self.delete_node(existing.id).await?;
                                    deleted += 1;
                                }
                                continue;
                            }

                            let (start_at, end_at) = match event_times(&event) {
                                Some(times) => times,
                                None => {
                                    skipped += 1;
                                    continue;
                                }
                            };

                            let source = event_source(calendar_id, event_id);
                            let existing = self.store.nodes.find_by_source(&source).await?;
                            let was_existing = existing.is_some();
                            let mut node = if let Some(existing) = existing {
                                existing
                            } else {
                                KnowledgeNode::new(NodeKind::Event, "")
                                    .with_namespace(config.namespace.clone())
                                    .with_tags(vec!["calendar".into(), "google-calendar".into()])
                                    .with_source(source)
                            };

                            let title = event
                                .summary
                                .clone()
                                .filter(|s| !s.trim().is_empty())
                                .unwrap_or_else(|| "Google Calendar Event".to_string());
                            let content = event
                                .description
                                .clone()
                                .filter(|s| !s.trim().is_empty())
                                .unwrap_or_else(|| title.clone());

                            node.title = Some(title);
                            node.content = content;
                            node.metadata.insert(
                                "event_start_at".to_string(),
                                serde_json::Value::String(start_at.to_rfc3339()),
                            );
                            node.metadata.insert(
                                "event_end_at".to_string(),
                                serde_json::Value::String(end_at.to_rfc3339()),
                            );
                            node.metadata.insert(
                                "google_calendar_event_id".to_string(),
                                serde_json::Value::String(event_id.clone()),
                            );
                            node.metadata.insert(
                                "google_calendar_calendar_id".to_string(),
                                serde_json::Value::String(calendar_id.to_string()),
                            );
                            if let Some(updated_at) = event.updated.clone() {
                                node.metadata.insert(
                                    "google_calendar_updated_at".to_string(),
                                    serde_json::Value::String(updated_at),
                                );
                            }
                            if let Some(html_link) = event.html_link.clone() {
                                node.metadata.insert(
                                    "google_calendar_html_link".to_string(),
                                    serde_json::Value::String(html_link),
                                );
                            }

                            if was_existing {
                                node.temporal.updated_at = Utc::now();
                                let _ = self.update_node(node).await?;
                                updated += 1;
                            } else {
                                let _ = self.store_node(node).await?;
                                created += 1;
                            }
                        }
                    } else {
                        skipped = events.len();
                    }

                    report.created = created;
                    report.updated = updated;
                    report.deleted = deleted;
                    report.skipped = skipped;

                    if config.export_events {
                        let (exported_created, exported_updated) =
                            google_export_events(self, &access_token, calendar_id, config).await?;
                        report.exported_created = exported_created;
                        report.exported_updated = exported_updated;
                    }

                    if let Some(next) = next_sync_token {
                        let _ = self
                            .store
                            .nodes
                            .upsert_poll_state(&adapter_name, &next, report.fetched as u64)
                            .await;
                    }

                    return Ok(report);
                }
                Err(GoogleCalendarFetchError::SyncTokenExpired) => {
                    if attempt == 0 {
                        sync_token = None;
                        continue;
                    }
                    return Err(HxError::InvalidInput(
                        "google calendar sync token expired".into(),
                    ));
                }
                Err(GoogleCalendarFetchError::RequestFailed(err)) => {
                    return Err(HxError::Storage(format!(
                        "google calendar sync failed: {err}"
                    )));
                }
            }
        }

        Err(HxError::Storage(
            "google calendar sync failed unexpectedly".into(),
        ))
    }
}

async fn google_refresh_access_token(
    client_id: &str,
    client_secret: &str,
    refresh_token: &str,
) -> MvResult<String> {
    let client = reqwest::Client::new();
    let response = client
        .post("https://oauth2.googleapis.com/token")
        .form(&[
            ("client_id", client_id),
            ("client_secret", client_secret),
            ("refresh_token", refresh_token),
            ("grant_type", "refresh_token"),
        ])
        .send()
        .await
        .map_err(|e| HxError::Storage(format!("google token request failed: {e}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(HxError::Storage(format!(
            "google token request failed ({status}): {body}"
        )));
    }

    let token = response
        .json::<GoogleTokenResponse>()
        .await
        .map_err(|e| HxError::Storage(format!("google token response parse failed: {e}")))?;

    Ok(token.access_token)
}

async fn google_list_events(
    access_token: &str,
    calendar_id: &str,
    config: &crate::config::GoogleCalendarConfig,
    sync_token: Option<&str>,
) -> Result<(Vec<GoogleEvent>, Option<String>), GoogleCalendarFetchError> {
    let client = reqwest::Client::new();
    let encoded_calendar = byte_serialize(calendar_id.as_bytes()).collect::<String>();
    let url = format!("https://www.googleapis.com/calendar/v3/calendars/{encoded_calendar}/events");

    let max_results = config.max_results.to_string();
    let mut page_token: Option<String> = None;
    let mut events: Vec<GoogleEvent> = Vec::new();

    let next_sync_token = loop {
        let mut request = client.get(&url).bearer_auth(access_token).query(&[
            ("singleEvents", "true"),
            ("showDeleted", "true"),
            ("maxResults", max_results.as_str()),
        ]);

        if let Some(token) = sync_token {
            request = request.query(&[("syncToken", token)]);
        } else {
            let now = Utc::now();
            let time_min = (now - chrono::Duration::days(config.lookback_days)).to_rfc3339();
            let time_max = (now + chrono::Duration::days(config.lookahead_days)).to_rfc3339();
            request = request.query(&[
                ("timeMin", time_min.as_str()),
                ("timeMax", time_max.as_str()),
                ("orderBy", "startTime"),
            ]);
        }

        if let Some(ref token) = page_token {
            request = request.query(&[("pageToken", token.as_str())]);
        }

        let response = request
            .send()
            .await
            .map_err(|e| GoogleCalendarFetchError::RequestFailed(e.to_string()))?;

        if response.status() == HttpStatusCode::GONE {
            return Err(GoogleCalendarFetchError::SyncTokenExpired);
        }

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(GoogleCalendarFetchError::RequestFailed(format!(
                "google events request failed ({status}): {body}"
            )));
        }

        let payload = response
            .json::<GoogleEventsResponse>()
            .await
            .map_err(|e| GoogleCalendarFetchError::RequestFailed(e.to_string()))?;

        if let Some(mut page_items) = payload.items {
            events.append(&mut page_items);
        }

        if payload.next_page_token.is_none() {
            break payload.next_sync_token;
        }

        page_token = payload.next_page_token;
    };

    Ok((events, next_sync_token))
}

fn event_source(calendar_id: &str, event_id: &str) -> String {
    format!("google-calendar:{calendar_id}:{event_id}")
}

fn parse_google_event_time(time: &GoogleEventTime) -> Option<DateTime<Utc>> {
    if let Some(ref dt) = time.date_time {
        return DateTime::parse_from_rfc3339(dt)
            .ok()
            .map(|dt| dt.with_timezone(&Utc));
    }

    let date_str = time.date.as_ref()?;
    let date = chrono::NaiveDate::parse_from_str(date_str, "%Y-%m-%d").ok()?;
    let naive = date.and_hms_opt(0, 0, 0)?;
    Some(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

fn event_times(event: &GoogleEvent) -> Option<(DateTime<Utc>, DateTime<Utc>)> {
    let start = event.start.as_ref().and_then(parse_google_event_time)?;
    let end = event
        .end
        .as_ref()
        .and_then(parse_google_event_time)
        .unwrap_or_else(|| start + chrono::Duration::hours(1));
    Some((start, end))
}

async fn google_export_events(
    engine: &HelixEngine,
    access_token: &str,
    calendar_id: &str,
    config: &crate::config::GoogleCalendarConfig,
) -> MvResult<(usize, usize)> {
    let mut exported_created = 0usize;
    let mut exported_updated = 0usize;

    let filters = hx_core::QueryFilters {
        namespace: Some(config.namespace.clone()),
        kinds: Some(vec![NodeKind::Event]),
        tags: None,
        min_importance: None,
        created_after: None,
        created_before: None,
    };

    let nodes = engine.store.nodes.list(&filters, 1000, 0).await?;
    let client = reqwest::Client::new();
    let encoded_calendar = byte_serialize(calendar_id.as_bytes()).collect::<String>();
    let url = format!("https://www.googleapis.com/calendar/v3/calendars/{encoded_calendar}/events");

    for mut node in nodes {
        let start_at = node
            .metadata
            .get("event_start_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.to_rfc3339());
        let end_at = node
            .metadata
            .get("event_end_at")
            .and_then(|v| v.as_str())
            .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
            .map(|dt| dt.to_rfc3339());

        let (Some(start_at), Some(end_at)) = (start_at, end_at) else {
            continue;
        };

        let summary = node
            .title
            .clone()
            .unwrap_or_else(|| "Helix Event".to_string());
        let description = node.content.clone();

        let mut payload = serde_json::json!({
            "summary": summary,
            "description": description,
            "start": { "dateTime": start_at },
            "end": { "dateTime": end_at }
        });

        if let Some(ref source) = node.source {
            payload["source"] = serde_json::json!({ "title": "Helix", "url": source });
        }

        if let Some(event_id) = node
            .metadata
            .get("google_calendar_event_id")
            .and_then(|v| v.as_str())
        {
            let request = client
                .patch(format!("{url}/{event_id}"))
                .bearer_auth(access_token)
                .json(&payload);

            let response = request
                .send()
                .await
                .map_err(|e| HxError::Storage(format!("google event update failed: {e}")))?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(HxError::Storage(format!(
                    "google event update failed ({status}): {body}"
                )));
            }

            exported_updated += 1;
            continue;
        }

        let response = client
            .post(&url)
            .bearer_auth(access_token)
            .json(&payload)
            .send()
            .await
            .map_err(|e| HxError::Storage(format!("google event create failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(HxError::Storage(format!(
                "google event create failed ({status}): {body}"
            )));
        }

        let created_event = response
            .json::<GoogleEvent>()
            .await
            .map_err(|e| HxError::Storage(format!("google event response parse failed: {e}")))?;

        if let Some(event_id) = created_event.id {
            node.metadata.insert(
                "google_calendar_event_id".to_string(),
                serde_json::Value::String(event_id.clone()),
            );
            node.metadata.insert(
                "google_calendar_calendar_id".to_string(),
                serde_json::Value::String(calendar_id.to_string()),
            );
            if let Some(html_link) = created_event.html_link {
                node.metadata.insert(
                    "google_calendar_html_link".to_string(),
                    serde_json::Value::String(html_link),
                );
            }
            if node.source.is_none() {
                node.source = Some(event_source(calendar_id, &event_id));
            }
            node.temporal.updated_at = Utc::now();
            let _ = engine.update_node(node).await?;
            exported_created += 1;
        }
    }

    Ok((exported_created, exported_updated))
}
