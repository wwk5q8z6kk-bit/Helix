use super::HelixEngine;
use chrono::{DateTime, Utc};
use hx_core::{
    GraphStore, KnowledgeNode, MvResult, NodeKind, NodeStore, PrioritizedTask, QueryFilters,
    RelationKind, Relationship, TaskPrioritizationOptions, TaskPriorityCandidate,
};
use std::cmp::Ordering;

use crate::recurrence::{
    collect_due_occurrences, parse_optional_metadata_bool, parse_optional_metadata_datetime,
    parse_optional_metadata_u64, parse_task_recurrence_rule, previous_due_at,
    RECURRING_DUE_AT_METADATA_KEY, RECURRING_INSTANCE_METADATA_KEY,
    RECURRING_PARENT_ID_METADATA_KEY, TASK_COMPLETED_METADATA_KEY, TASK_DUE_AT_METADATA_KEY,
    TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY, TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY,
    TASK_RECURRENCE_METADATA_KEY, TASK_REMINDER_SENT_AT_METADATA_KEY,
    TASK_REMINDER_STATUS_METADATA_KEY,
};

// Struct definitions moved from engine/mod.rs
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TaskRecurrenceRollforwardStats {
    pub scanned_tasks: usize,
    pub recurring_templates: usize,
    pub generated_instances: usize,
    pub updated_templates: usize,
    pub errors: usize,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TaskReminderDispatchStats {
    pub scanned_tasks: usize,
    pub due_tasks: usize,
    pub reminders_marked_sent: usize,
    pub errors: usize,
}

// Local metadata keys not in recurrence
const TASK_PRIORITY_METADATA_KEY: &str = "priority";
const TASK_PRIORITY_ALT_METADATA_KEY: &str = "importance";
const TASK_STATUS_METADATA_KEY: &str = "status";
const TASK_STATUS_ALT_METADATA_KEY: &str = "state";
const TASK_ESTIMATE_MINUTES_METADATA_KEY: &str = "estimate_minutes";
const TASK_ESTIMATE_MINUTES_ALT_METADATA_KEY: &str = "duration_minutes";
const TASK_ESTIMATE_MIN_METADATA_KEY: &str = "estimate";
const TASK_AI_PRIORITY_METADATA_KEY: &str = "ai_priority";

impl HelixEngine {
    /// Generate due instances for recurring task templates.
    pub async fn rollforward_recurring_tasks(
        &self,
        now: DateTime<Utc>,
        max_instances_per_template: usize,
    ) -> MvResult<TaskRecurrenceRollforwardStats> {
        if !self.config.recurrence.enabled {
            return Ok(TaskRecurrenceRollforwardStats::default());
        }

        let mut stats = TaskRecurrenceRollforwardStats::default();
        let page_size = 200;
        let mut offset = 0usize;

        loop {
            let filters = QueryFilters {
                kinds: Some(vec![NodeKind::Task]),
                ..Default::default()
            };
            let page: Vec<KnowledgeNode> = self.store.nodes.list(&filters, page_size, offset).await?;
            if page.is_empty() {
                break;
            }
            let page_len = page.len();

            for mut template in page {
                stats.scanned_tasks += 1;
                if is_recurring_instance(&template) {
                    continue;
                }

                let recurrence_rule = match parse_task_recurrence_rule(&template.metadata) {
                    Ok(Some(rule)) if rule.enabled => rule,
                    Ok(Some(_)) | Ok(None) => continue,
                    Err(err) => {
                        stats.errors += 1;
                        tracing::warn!(
                            node_id = %template.id,
                            namespace = %template.namespace,
                            error = %err,
                            "helix_recurrence_rule_parse_failed"
                        );
                        continue;
                    }
                };
                stats.recurring_templates += 1;

                let explicit_due_at = match parse_optional_metadata_datetime(
                    &template.metadata,
                    TASK_DUE_AT_METADATA_KEY,
                ) {
                    Ok(value) => value,
                    Err(err) => {
                        stats.errors += 1;
                        tracing::warn!(
                            node_id = %template.id,
                            namespace = %template.namespace,
                            error = %err,
                            "helix_recurrence_due_at_parse_failed"
                        );
                        continue;
                    }
                };
                let last_generated = match parse_optional_metadata_datetime(
                    &template.metadata,
                    TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY,
                ) {
                    Ok(Some(value)) => value,
                    Ok(None) => explicit_due_at
                        .map(|due| previous_due_at(due, &recurrence_rule))
                        .unwrap_or(template.temporal.created_at),
                    Err(err) => {
                        stats.errors += 1;
                        tracing::warn!(
                            node_id = %template.id,
                            namespace = %template.namespace,
                            error = %err,
                            "helix_recurrence_last_generated_parse_failed"
                        );
                        continue;
                    }
                };
                let generated_count = parse_optional_metadata_u64(
                    &template.metadata,
                    TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY,
                )
                .unwrap_or(0);
                let due_dates = collect_due_occurrences(
                    &recurrence_rule,
                    last_generated,
                    now,
                    max_instances_per_template,
                    generated_count,
                );
                if due_dates.is_empty() {
                    continue;
                }

                let mut created_for_template = 0usize;
                let mut latest_due = last_generated;
                for due_at in due_dates {
                    let mut instance = KnowledgeNode::new(NodeKind::Task, template.content.clone())
                        .with_namespace(template.namespace.clone())
                        .with_importance(template.importance);
                    if let Some(title) = template.title.as_deref() {
                        instance = instance.with_title(title);
                    }
                    if let Some(source) = template.source.as_deref() {
                        instance = instance.with_source(source);
                    }

                    let mut tags = template.tags.clone();
                    if !tags
                        .iter()
                        .any(|tag| tag.eq_ignore_ascii_case("recurring-instance"))
                    {
                        tags.push("recurring-instance".to_string());
                    }
                    instance = instance.with_tags(tags);

                    for (key, value) in &template.metadata {
                        if matches!(
                            key.as_str(),
                            TASK_RECURRENCE_METADATA_KEY
                                | TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY
                                | TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY
                                | RECURRING_INSTANCE_METADATA_KEY
                                | RECURRING_PARENT_ID_METADATA_KEY
                                | RECURRING_DUE_AT_METADATA_KEY
                        ) {
                            continue;
                        }
                        instance.metadata.insert(key.clone(), value.clone());
                    }
                    instance.metadata.insert(
                        RECURRING_INSTANCE_METADATA_KEY.into(),
                        serde_json::Value::Bool(true),
                    );
                    instance.metadata.insert(
                        RECURRING_PARENT_ID_METADATA_KEY.into(),
                        serde_json::Value::String(template.id.to_string()),
                    );
                    instance.metadata.insert(
                        RECURRING_DUE_AT_METADATA_KEY.into(),
                        serde_json::Value::String(due_at.to_rfc3339()),
                    );
                    instance.metadata.insert(
                        TASK_DUE_AT_METADATA_KEY.into(),
                        serde_json::Value::String(due_at.to_rfc3339()),
                    );

                    match self.store_node(instance).await {
                        Ok(stored_instance) => {
                            created_for_template += 1;
                            latest_due = due_at;
                            stats.generated_instances += 1;

                            let rel = Relationship::new(
                                template.id,
                                stored_instance.id,
                                RelationKind::DerivedFrom,
                            );
                            if let Err(err) = self.graph.add_relationship(&rel).await {
                                stats.errors += 1;
                                tracing::warn!(
                                    template_id = %template.id,
                                    instance_id = %stored_instance.id,
                                    error = %err,
                                    "helix_recurrence_parent_instance_link_failed"
                                );
                            }
                        }
                        Err(err) => {
                            stats.errors += 1;
                            tracing::warn!(
                                node_id = %template.id,
                                namespace = %template.namespace,
                                error = %err,
                                "helix_recurrence_instance_create_failed"
                            );
                        }
                    }
                }

                if created_for_template > 0 {
                    template.metadata.insert(
                        TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY.into(),
                        serde_json::Value::String(latest_due.to_rfc3339()),
                    );
                    let total_generated_count =
                        generated_count.saturating_add(created_for_template as u64);
                    template.metadata.insert(
                        TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY.into(),
                        serde_json::Value::Number(serde_json::Number::from(total_generated_count)),
                    );
                    template.temporal.updated_at = now;
                    template.temporal.version = template.temporal.version.saturating_add(1);
                    if let Err(err) = self.ingest.update(template).await {
                        stats.errors += 1;
                        tracing::warn!(
                            error = %err,
                            "helix_recurrence_template_update_failed"
                        );
                    } else {
                        stats.updated_templates += 1;
                    }
                }
            }

            if page_size > 0 && page_len < page_size {
                break;
            }
            offset = offset.saturating_add(page_size);
        }

        Ok(stats)
    }

    /// List due tasks up to `due_before`.
    pub async fn list_due_tasks(
        &self,
        due_before: DateTime<Utc>,
        namespace: Option<String>,
        limit: usize,
        include_completed: bool,
    ) -> MvResult<Vec<KnowledgeNode>> {
        let capped_limit = limit.clamp(1, 1000);
        let page_size = 250;
        let mut offset = 0usize;
        let mut due = Vec::<(DateTime<Utc>, KnowledgeNode)>::new();

        loop {
            let filters = QueryFilters {
                namespace: namespace.clone(),
                kinds: Some(vec![NodeKind::Task]),
                ..Default::default()
            };
            let page: Vec<KnowledgeNode> = self.store.nodes.list(&filters, page_size, offset).await?;
            if page.is_empty() {
                break;
            }
            let page_len = page.len();

            for node in page {
                let due_at = match parse_optional_metadata_datetime(
                    &node.metadata,
                    TASK_DUE_AT_METADATA_KEY,
                ) {
                    Ok(Some(value)) => value,
                    Ok(None) => continue,
                    Err(err) => {
                        tracing::warn!(
                            node_id = %node.id,
                            namespace = %node.namespace,
                            error = %err,
                            "helix_due_task_parse_failed"
                        );
                        continue;
                    }
                };

                if due_at > due_before {
                    continue;
                }

                let is_completed =
                    parse_optional_metadata_bool(&node.metadata, TASK_COMPLETED_METADATA_KEY)
                        .unwrap_or(false);
                if !include_completed && is_completed {
                    continue;
                }

                due.push((due_at, node));
            }

            if page_size > 0 && page_len < page_size {
                break;
            }
            offset = offset.saturating_add(page_size);
        }

        due.sort_by(|(left_due, left_node), (right_due, right_node)| {
            left_due
                .cmp(right_due)
                .then_with(|| left_node.id.cmp(&right_node.id))
        });

        Ok(due
            .into_iter()
            .take(capped_limit)
            .map(|(_due_at, node)| node)
            .collect())
    }

    /// Prioritize tasks using deterministic heuristic scoring.
    pub async fn prioritize_tasks(
        &self,
        options: TaskPrioritizationOptions,
    ) -> MvResult<Vec<PrioritizedTask>> {
        let limit = options.limit.clamp(1, 200);
        let page_size = 250;
        let mut offset = 0usize;
        let mut candidates: Vec<TaskPriorityCandidate> = Vec::new();

        loop {
            let filters = QueryFilters {
                namespace: options.namespace.clone(),
                kinds: Some(vec![NodeKind::Task]),
                ..Default::default()
            };
            let page: Vec<KnowledgeNode> = self.store.nodes.list(&filters, page_size, offset).await?;
            if page.is_empty() {
                break;
            }
            let page_len = page.len();

            for node in page {
                let completed =
                    parse_optional_metadata_bool(&node.metadata, TASK_COMPLETED_METADATA_KEY)
                        .unwrap_or(false);
                if !options.include_completed && completed {
                    continue;
                }

                let due_at = match parse_optional_metadata_datetime(
                    &node.metadata,
                    TASK_DUE_AT_METADATA_KEY,
                ) {
                    Ok(value) => value,
                    Err(err) => {
                        tracing::warn!(
                            node_id = %node.id,
                            namespace = %node.namespace,
                            error = %err,
                            "helix_task_priority_due_at_parse_failed"
                        );
                        continue;
                    }
                };

                if due_at.is_none() && !options.include_without_due {
                    continue;
                }

                let (score, reason) = Self::score_task(&node, due_at, options.now);
                candidates.push(TaskPriorityCandidate {
                    task: node,
                    score,
                    reason,
                    due_at,
                });
            }

            if page_size > 0 && page_len < page_size {
                break;
            }
            offset = offset.saturating_add(page_size);
        }

        candidates.sort_by(|left, right| {
            let score_order = right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal);
            if score_order != Ordering::Equal {
                return score_order;
            }

            let due_order = match (left.due_at, right.due_at) {
                (Some(left_due), Some(right_due)) => left_due.cmp(&right_due),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            };
            if due_order != Ordering::Equal {
                return due_order;
            }

            left.task.id.cmp(&right.task.id)
        });

        let mut prioritized = Vec::new();
        for (idx, candidate) in candidates.into_iter().take(limit).enumerate() {
            let rank = idx + 1;
            let mut task = candidate.task;
            let reason = candidate.reason;
            let score = candidate.score;

            if options.persist {
                task.metadata.insert(
                    TASK_AI_PRIORITY_METADATA_KEY.into(),
                    serde_json::json!({
                        "score": score,
                        "rank": rank,
                        "reason": reason,
                        "generated_at": options.now.to_rfc3339(),
                        "algorithm": "heuristic_v1",
                    }),
                );
                task.temporal.updated_at = options.now;
                task.temporal.version = task.temporal.version.saturating_add(1);
                task = self.ingest.update(task).await?;
            }

            prioritized.push(PrioritizedTask {
                task,
                score,
                rank,
                reason,
            });
        }

        Ok(prioritized)
    }

    fn score_task(
        task: &KnowledgeNode,
        due_at: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> (f64, String) {
        let priority_override = parse_optional_metadata_f64(
            &task.metadata,
            TASK_PRIORITY_METADATA_KEY,
        )
        .or_else(|| {
            parse_optional_metadata_f64(&task.metadata, TASK_PRIORITY_ALT_METADATA_KEY)
        });
        let priority_score = if let Some(priority_raw) = priority_override {
            let priority = priority_raw.round().clamp(1.0, 5.0);
            (6.0 - priority) / 5.0
        } else {
            task.importance.clamp(0.0, 1.0)
        };

        let mut due_score = 0.0;
        if let Some(due_at) = due_at {
            let hours = (due_at - now).num_seconds() as f64 / 3600.0;
            if hours <= 0.0 {
                due_score = 1.0;
            } else {
                let days = hours / 24.0;
                due_score = (1.0 - (days / 7.0).min(1.0)).max(0.0);
            }
        }

        let status = task
            .metadata
            .get(TASK_STATUS_METADATA_KEY)
            .or_else(|| task.metadata.get(TASK_STATUS_ALT_METADATA_KEY))
            .and_then(|value| value.as_str())
            .map(|value| value.to_ascii_lowercase());
        let status_score = match status.as_deref() {
            Some("in_progress") => 0.2,
            Some("planned") => 0.12,
            Some("review") => 0.1,
            Some("inbox") => 0.05,
            Some("waiting") => -0.05,
            Some("blocked") => -0.1,
            _ => 0.0,
        };

        let estimate =
            parse_optional_metadata_f64(&task.metadata, TASK_ESTIMATE_MINUTES_METADATA_KEY)
                .or_else(|| {
                    parse_optional_metadata_f64(
                        &task.metadata,
                        TASK_ESTIMATE_MINUTES_ALT_METADATA_KEY,
                    )
                })
                .or_else(|| {
                    parse_optional_metadata_f64(
                        &task.metadata,
                        TASK_ESTIMATE_MIN_METADATA_KEY,
                    )
                });
        let estimate_score = match estimate {
            Some(minutes) => (1.0 - (minutes / 240.0).min(1.0)).max(0.0),
            None => 0.05,
        };

        let completed = parse_optional_metadata_bool(&task.metadata, TASK_COMPLETED_METADATA_KEY)
            .unwrap_or(false);
        let completion_penalty = if completed { -0.4 } else { 0.0 };

        let score = 0.45 * priority_score
            + 0.35 * due_score
            + 0.1 * status_score
            + 0.1 * estimate_score
            + completion_penalty;

        let mut reasons: Vec<String> = Vec::new();
        if let Some(priority_raw) = priority_override {
            let priority = priority_raw.round().clamp(1.0, 5.0) as i64;
            if priority <= 2 {
                reasons.push(format!("High priority (P{priority})"));
            } else if priority >= 4 {
                reasons.push(format!("Lower priority (P{priority})"));
            }
        } else if priority_score >= 0.8 {
            reasons.push("High importance".to_string());
        } else if priority_score <= 0.3 {
            reasons.push("Lower importance".to_string());
        }

        if let Some(due_at) = due_at {
            let delta = due_at - now;
            if delta.num_seconds() <= 0 {
                reasons.push("Overdue".to_string());
            } else {
                let days = delta.num_seconds() as f64 / 86_400.0;
                if days <= 1.0 {
                    reasons.push("Due within 24h".to_string());
                } else if days <= 3.0 {
                    reasons.push("Due soon".to_string());
                } else if days <= 7.0 {
                    reasons.push("Due this week".to_string());
                }
            }
        }

        match status.as_deref() {
            Some("in_progress") => reasons.push("In progress".to_string()),
            Some("planned") => reasons.push("Planned".to_string()),
            Some("waiting") => reasons.push("Waiting".to_string()),
            Some("review") => reasons.push("In review".to_string()),
            Some("blocked") => reasons.push("Blocked".to_string()),
            _ => {}
        }

        if let Some(minutes) = estimate {
            if minutes <= 30.0 {
                reasons.push("Quick win".to_string());
            }
        }

        if completed {
            reasons.push("Completed".to_string());
        }

        let reason = if reasons.is_empty() {
            "Balanced priority".to_string()
        } else {
            reasons.into_iter().take(3).collect::<Vec<_>>().join(", ")
        };

        (score, reason)
    }

    /// Mark due task reminders as sent by setting metadata fields on each task.
    pub async fn dispatch_due_task_reminders(
        &self,
        now: DateTime<Utc>,
        limit: usize,
    ) -> MvResult<TaskReminderDispatchStats> {
        let due_tasks = self.list_due_tasks(now, None, limit, false).await?;
        let mut stats = TaskReminderDispatchStats {
            scanned_tasks: due_tasks.len(),
            due_tasks: due_tasks.len(),
            ..Default::default()
        };

        for mut task in due_tasks {
            let already_sent = match parse_optional_metadata_datetime(
                &task.metadata,
                TASK_REMINDER_SENT_AT_METADATA_KEY,
            ) {
                Ok(value) => value.is_some(),
                Err(err) => {
                    stats.errors += 1;
                    tracing::warn!(
                        node_id = %task.id,
                        namespace = %task.namespace,
                        error = %err,
                        "helix_task_reminder_sent_at_parse_failed"
                    );
                    continue;
                }
            };
            if already_sent {
                continue;
            }

            task.metadata.insert(
                TASK_REMINDER_STATUS_METADATA_KEY.into(),
                serde_json::Value::String("sent".to_string()),
            );
            task.metadata.insert(
                TASK_REMINDER_SENT_AT_METADATA_KEY.into(),
                serde_json::Value::String(now.to_rfc3339()),
            );
            task.temporal.updated_at = now;
            task.temporal.version = task.temporal.version.saturating_add(1);

            match self.ingest.update(task).await {
                Ok(_updated) => {
                    stats.reminders_marked_sent += 1;
                }
                Err(err) => {
                    stats.errors += 1;
                    tracing::warn!(
                        error = %err,
                        "helix_task_reminder_mark_sent_failed"
                    );
                }
            }
        }

        Ok(stats)
    }
}

// Helpers

fn is_recurring_instance(node: &KnowledgeNode) -> bool {
    node.metadata
        .get(RECURRING_INSTANCE_METADATA_KEY)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn parse_optional_metadata_f64(
    metadata: &std::collections::HashMap<String, serde_json::Value>,
    key: &str,
) -> Option<f64> {
    match metadata.get(key) {
        Some(serde_json::Value::Number(value)) => value.as_f64(),
        Some(serde_json::Value::String(value)) => value.parse::<f64>().ok(),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_utils::*;
    use chrono::TimeZone;

    async fn test_rollforward_recurring_tasks_generates_due_instance() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let namespace = "ops".to_string();
        let mut template =
            KnowledgeNode::new(NodeKind::Task, "Daily standup action checklist".to_string())
                .with_namespace(namespace.clone())
                .with_tags(vec!["ops".to_string()]);

        let last_generated = Utc
            .with_ymd_and_hms(2026, 2, 5, 9, 0, 0)
            .single()
            .expect("valid datetime");
        template.metadata.insert(
            TASK_RECURRENCE_METADATA_KEY.into(),
            serde_json::json!({
                "frequency": "daily",
                "interval": 1,
                "enabled": true
            }),
        );
        template.metadata.insert(
            TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY.into(),
            serde_json::Value::String(last_generated.to_rfc3339()),
        );

        let stored_template = engine.store_node(template).await.unwrap();
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 9, 0, 0)
            .single()
            .expect("valid datetime");
        let stats = engine.rollforward_recurring_tasks(now, 4).await.unwrap();
        assert_eq!(stats.generated_instances, 1);

        let tasks = engine
            .list_nodes(
                &QueryFilters {
                    namespace: Some(namespace.clone()),
                    kinds: Some(vec![NodeKind::Task]),
                    ..Default::default()
                },
                50,
                0,
            )
            .await
            .unwrap();

        let instance = tasks
            .iter()
            .find(|node| {
                node.metadata
                    .get(RECURRING_INSTANCE_METADATA_KEY)
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .expect("recurring instance should exist");
        assert_eq!(
            instance
                .metadata
                .get(RECURRING_PARENT_ID_METADATA_KEY)
                .and_then(serde_json::Value::as_str)
                .map(str::to_string),
            Some(stored_template.id.to_string())
        );

        let rels = engine
            .graph
            .get_relationships_from(stored_template.id)
            .await
            .unwrap();
        assert!(rels
            .iter()
            .any(|rel| { rel.to_node == instance.id && rel.kind == RelationKind::DerivedFrom }));
    }

    #[tokio::test]
    async fn test_rollforward_recurring_tasks_is_idempotent_for_same_instant() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let mut template =
            KnowledgeNode::new(NodeKind::Task, "Weekly planning template".to_string())
                .with_namespace("ops");
        template.metadata.insert(
            TASK_RECURRENCE_METADATA_KEY.into(),
            serde_json::json!({
                "frequency": "weekly",
                "interval": 1,
                "enabled": true
            }),
        );
        template.metadata.insert(
            TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 1, 30, 10, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let _stored = engine.store_node(template).await.unwrap();

        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 10, 0, 0)
            .single()
            .expect("valid datetime");
        let first = engine.rollforward_recurring_tasks(now, 4).await.unwrap();
        let second = engine.rollforward_recurring_tasks(now, 4).await.unwrap();

        assert!(first.generated_instances >= 1);
        assert_eq!(second.generated_instances, 0);
    }

    #[tokio::test]
    async fn test_list_due_tasks_filters_and_sorts() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let base = Utc
            .with_ymd_and_hms(2026, 2, 6, 12, 0, 0)
            .single()
            .expect("valid datetime");

        let mut task_a = KnowledgeNode::new(NodeKind::Task, "A".to_string()).with_namespace("ops");
        task_a.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 10, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let mut task_b = KnowledgeNode::new(NodeKind::Task, "B".to_string()).with_namespace("ops");
        task_b.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 11, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let mut task_c = KnowledgeNode::new(NodeKind::Task, "C".to_string()).with_namespace("ops");
        task_c.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 9, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        task_c.metadata.insert(
            TASK_COMPLETED_METADATA_KEY.into(),
            serde_json::Value::Bool(true),
        );

        let _a = engine.store_node(task_a).await.unwrap();
        let _b = engine.store_node(task_b).await.unwrap();
        let _c = engine.store_node(task_c).await.unwrap();

        let due = engine
            .list_due_tasks(base, Some("ops".to_string()), 10, false)
            .await
            .unwrap();
        assert_eq!(due.len(), 2);
        assert_eq!(due[0].content, "A");
        assert_eq!(due[1].content, "B");
    }

    #[tokio::test]
    async fn test_prioritize_tasks_ranks_by_due_and_importance() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 12, 0, 0)
            .single()
            .expect("valid datetime");

        let mut urgent = KnowledgeNode::new(NodeKind::Task, "Urgent".to_string())
            .with_namespace("ops")
            .with_importance(0.9);
        urgent.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 18, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );

        let important = KnowledgeNode::new(NodeKind::Task, "Important".to_string())
            .with_namespace("ops")
            .with_importance(0.8);

        let mut later = KnowledgeNode::new(NodeKind::Task, "Later".to_string())
            .with_namespace("ops")
            .with_importance(0.2);
        later.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 16, 12, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );

        let _urgent = engine.store_node(urgent).await.unwrap();
        let _important = engine.store_node(important).await.unwrap();
        let _later = engine.store_node(later).await.unwrap();

        let prioritized = engine
            .prioritize_tasks(TaskPrioritizationOptions {
                namespace: Some("ops".to_string()),
                limit: 10,
                include_completed: false,
                include_without_due: true,
                persist: false,
                now,
            })
            .await
            .unwrap();

        assert_eq!(prioritized.len(), 3);
        assert_eq!(prioritized[0].task.content, "Urgent");
        assert_eq!(prioritized[1].task.content, "Important");
        assert_eq!(prioritized[2].task.content, "Later");
        assert_eq!(prioritized[0].rank, 1);
    }

    #[tokio::test]
    async fn test_dispatch_due_task_reminders_marks_once() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 12, 0, 0)
            .single()
            .expect("valid datetime");

        let mut task = KnowledgeNode::new(NodeKind::Task, "Follow up on incident".to_string())
            .with_namespace("ops");
        task.metadata.insert(
            TASK_DUE_AT_METADATA_KEY.into(),
            serde_json::Value::String(
                Utc.with_ymd_and_hms(2026, 2, 6, 8, 0, 0)
                    .single()
                    .expect("valid datetime")
                    .to_rfc3339(),
            ),
        );
        let stored = engine.store_node(task).await.unwrap();

        let first = engine.dispatch_due_task_reminders(now, 20).await.unwrap();
        assert_eq!(first.reminders_marked_sent, 1);

        let second = engine.dispatch_due_task_reminders(now, 20).await.unwrap();
        assert_eq!(second.reminders_marked_sent, 0);

        let refreshed = engine
            .get_node(stored.id)
            .await
            .unwrap()
            .expect("task exists");
        assert!(refreshed
            .metadata
            .contains_key(TASK_REMINDER_SENT_AT_METADATA_KEY));
    }

}
