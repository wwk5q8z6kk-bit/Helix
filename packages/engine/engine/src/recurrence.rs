use std::collections::HashMap;

use chrono::{DateTime, Datelike, Duration, NaiveDate, Timelike, Utc};
use hx_core::NodeKind;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const TASK_RECURRENCE_METADATA_KEY: &str = "task_recurrence";
pub const TASK_DUE_AT_METADATA_KEY: &str = "task_due_at";
pub const TASK_COMPLETED_METADATA_KEY: &str = "task_completed";
pub const TASK_COMPLETED_AT_METADATA_KEY: &str = "task_completed_at";
pub const TASK_RECURRENCE_LAST_GENERATED_AT_METADATA_KEY: &str =
    "task_recurrence_last_generated_at";
pub const TASK_RECURRENCE_GENERATED_COUNT_METADATA_KEY: &str = "task_recurrence_generated_count";
pub const TASK_REMINDER_STATUS_METADATA_KEY: &str = "task_reminder_status";
pub const TASK_REMINDER_SENT_AT_METADATA_KEY: &str = "task_reminder_sent_at";
pub const RECURRING_INSTANCE_METADATA_KEY: &str = "recurring_instance";
pub const RECURRING_PARENT_ID_METADATA_KEY: &str = "recurring_parent_id";
pub const RECURRING_DUE_AT_METADATA_KEY: &str = "recurring_due_at";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskRecurrenceFrequency {
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaskRecurrenceRule {
    pub frequency: TaskRecurrenceFrequency,
    pub interval: u32,
    pub count: Option<u32>,
    pub until: Option<DateTime<Utc>>,
    pub enabled: bool,
}

impl Default for TaskRecurrenceRule {
    fn default() -> Self {
        Self {
            frequency: TaskRecurrenceFrequency::Daily,
            interval: 1,
            count: None,
            until: None,
            enabled: true,
        }
    }
}

pub fn parse_task_recurrence_rule(
    metadata: &HashMap<String, Value>,
) -> Result<Option<TaskRecurrenceRule>, String> {
    let Some(raw_rule) = metadata.get(TASK_RECURRENCE_METADATA_KEY) else {
        return Ok(None);
    };
    let Some(obj) = raw_rule.as_object() else {
        return Err(format!(
            "{TASK_RECURRENCE_METADATA_KEY} must be a JSON object"
        ));
    };

    let frequency = match obj.get("frequency").and_then(Value::as_str) {
        Some("daily") | None => TaskRecurrenceFrequency::Daily,
        Some("weekly") => TaskRecurrenceFrequency::Weekly,
        Some("monthly") => TaskRecurrenceFrequency::Monthly,
        Some(other) => {
            return Err(format!(
                "{TASK_RECURRENCE_METADATA_KEY}.frequency must be one of daily|weekly|monthly (got {other})"
            ));
        }
    };

    let interval = obj
        .get("interval")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .clamp(1, 365) as u32;

    let count = obj
        .get("count")
        .and_then(Value::as_u64)
        .map(|value| value.clamp(1, 10_000) as u32);

    let until = match obj.get("until") {
        Some(value) => Some(parse_json_datetime_value(
            value,
            &format!("{TASK_RECURRENCE_METADATA_KEY}.until"),
        )?),
        None => None,
    };

    let enabled = obj.get("enabled").and_then(Value::as_bool).unwrap_or(true);

    Ok(Some(TaskRecurrenceRule {
        frequency,
        interval,
        count,
        until,
        enabled,
    }))
}

pub fn parse_optional_metadata_datetime(
    metadata: &HashMap<String, Value>,
    key: &str,
) -> Result<Option<DateTime<Utc>>, String> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    Ok(Some(parse_json_datetime_value(value, key)?))
}

pub fn parse_optional_metadata_u64(metadata: &HashMap<String, Value>, key: &str) -> Option<u64> {
    metadata.get(key).and_then(Value::as_u64)
}

pub fn parse_optional_metadata_bool(metadata: &HashMap<String, Value>, key: &str) -> Option<bool> {
    metadata.get(key).and_then(Value::as_bool)
}

pub fn validate_recurrence_metadata_for_kind(
    kind: NodeKind,
    metadata: Option<&HashMap<String, Value>>,
) -> Result<(), String> {
    let Some(metadata) = metadata else {
        return Ok(());
    };

    if metadata.contains_key(TASK_RECURRENCE_METADATA_KEY)
        && !matches!(kind, NodeKind::Task | NodeKind::Template)
    {
        return Err("task_recurrence is only supported for kind=task or kind=template".to_string());
    }

    if let Some(_rule) = parse_task_recurrence_rule(metadata)? {
        if !matches!(kind, NodeKind::Task | NodeKind::Template) {
            return Err("task_recurrence is only supported for kind=task or kind=template".to_string());
        }
    }

    if let Some(_due_at) = parse_optional_metadata_datetime(metadata, TASK_DUE_AT_METADATA_KEY)? {
        if !matches!(kind, NodeKind::Task | NodeKind::Event | NodeKind::Template) {
            return Err(
                "task_due_at is only supported for kind=task, kind=event, or kind=template".to_string(),
            );
        }
    }

    if parse_optional_metadata_bool(metadata, TASK_COMPLETED_METADATA_KEY).is_some()
        && !matches!(kind, NodeKind::Task | NodeKind::Template)
    {
        return Err("task_completed is only supported for kind=task or kind=template".to_string());
    }

    if let Some(_completed_at) =
        parse_optional_metadata_datetime(metadata, TASK_COMPLETED_AT_METADATA_KEY)?
    {
        if !matches!(kind, NodeKind::Task | NodeKind::Template) {
            return Err(
                "task_completed_at is only supported for kind=task or kind=template".to_string(),
            );
        }
    }

    Ok(())
}

pub fn next_due_at(last_due_at: DateTime<Utc>, rule: &TaskRecurrenceRule) -> DateTime<Utc> {
    match rule.frequency {
        TaskRecurrenceFrequency::Daily => last_due_at + Duration::days(rule.interval as i64),
        TaskRecurrenceFrequency::Weekly => last_due_at + Duration::weeks(rule.interval as i64),
        TaskRecurrenceFrequency::Monthly => add_months_utc(last_due_at, rule.interval),
    }
}

pub fn previous_due_at(
    reference_due_at: DateTime<Utc>,
    rule: &TaskRecurrenceRule,
) -> DateTime<Utc> {
    match rule.frequency {
        TaskRecurrenceFrequency::Daily => reference_due_at - Duration::days(rule.interval as i64),
        TaskRecurrenceFrequency::Weekly => reference_due_at - Duration::weeks(rule.interval as i64),
        TaskRecurrenceFrequency::Monthly => subtract_months_utc(reference_due_at, rule.interval),
    }
}

pub fn collect_due_occurrences(
    rule: &TaskRecurrenceRule,
    mut last_generated_at: DateTime<Utc>,
    now: DateTime<Utc>,
    max_instances: usize,
    already_generated_count: u64,
) -> Vec<DateTime<Utc>> {
    if max_instances == 0 || !rule.enabled {
        return Vec::new();
    }

    let mut due = Vec::new();
    loop {
        let candidate = next_due_at(last_generated_at, rule);
        if candidate > now {
            break;
        }
        if let Some(until) = rule.until {
            if candidate > until {
                break;
            }
        }
        if let Some(count) = rule.count {
            let projected_count = already_generated_count + due.len() as u64 + 1;
            if projected_count > count as u64 {
                break;
            }
        }

        due.push(candidate);
        if due.len() >= max_instances {
            break;
        }
        last_generated_at = candidate;
    }

    due
}

fn add_months_utc(value: DateTime<Utc>, months: u32) -> DateTime<Utc> {
    let naive = value.naive_utc();
    let mut year = naive.date().year();
    let mut month = naive.date().month() as i32;
    month += months as i32;

    while month > 12 {
        month -= 12;
        year += 1;
    }

    let month_u32 = month as u32;
    let day = naive
        .date()
        .day()
        .min(days_in_month(year, month_u32))
        .max(1);
    let date = NaiveDate::from_ymd_opt(year, month_u32, day).unwrap_or_else(|| naive.date());

    let next_naive = date
        .and_hms_opt(
            naive.time().hour(),
            naive.time().minute(),
            naive.time().second(),
        )
        .unwrap_or(naive);

    DateTime::<Utc>::from_naive_utc_and_offset(next_naive, Utc)
}

fn subtract_months_utc(value: DateTime<Utc>, months: u32) -> DateTime<Utc> {
    let naive = value.naive_utc();
    let mut year = naive.date().year();
    let mut month = naive.date().month() as i32;
    month -= months as i32;

    while month < 1 {
        month += 12;
        year -= 1;
    }

    let month_u32 = month as u32;
    let day = naive
        .date()
        .day()
        .min(days_in_month(year, month_u32))
        .max(1);
    let date = NaiveDate::from_ymd_opt(year, month_u32, day).unwrap_or_else(|| naive.date());

    let next_naive = date
        .and_hms_opt(
            naive.time().hour(),
            naive.time().minute(),
            naive.time().second(),
        )
        .unwrap_or(naive);

    DateTime::<Utc>::from_naive_utc_and_offset(next_naive, Utc)
}

fn days_in_month(year: i32, month: u32) -> u32 {
    for day in (28..=31).rev() {
        if NaiveDate::from_ymd_opt(year, month, day).is_some() {
            return day;
        }
    }
    28
}

fn parse_json_datetime_value(value: &Value, field_name: &str) -> Result<DateTime<Utc>, String> {
    let as_str = value
        .as_str()
        .ok_or_else(|| format!("{field_name} must be an RFC3339 string"))?;
    DateTime::parse_from_rfc3339(as_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|err| format!("{field_name} must be RFC3339: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn parse_recurrence_rule_accepts_valid_shape() {
        let metadata = serde_json::json!({
            "task_recurrence": {
                "frequency": "weekly",
                "interval": 2,
                "count": 4,
                "enabled": true
            }
        });
        let map: HashMap<String, Value> =
            serde_json::from_value(metadata).expect("metadata should deserialize");
        let rule = parse_task_recurrence_rule(&map)
            .expect("valid recurrence should parse")
            .expect("rule should exist");

        assert_eq!(rule.frequency, TaskRecurrenceFrequency::Weekly);
        assert_eq!(rule.interval, 2);
        assert_eq!(rule.count, Some(4));
        assert!(rule.enabled);
    }

    #[test]
    fn collect_due_occurrences_daily_generates_catch_up() {
        let now = Utc
            .with_ymd_and_hms(2026, 2, 6, 10, 0, 0)
            .single()
            .expect("valid datetime");
        let last = Utc
            .with_ymd_and_hms(2026, 2, 3, 10, 0, 0)
            .single()
            .expect("valid datetime");
        let rule = TaskRecurrenceRule {
            frequency: TaskRecurrenceFrequency::Daily,
            interval: 1,
            count: None,
            until: None,
            enabled: true,
        };

        let due = collect_due_occurrences(&rule, last, now, 10, 0);
        assert_eq!(due.len(), 3);
    }

    #[test]
    fn add_months_preserves_time_and_bounds_day() {
        let start = Utc
            .with_ymd_and_hms(2026, 1, 31, 8, 30, 0)
            .single()
            .expect("valid datetime");
        let next = add_months_utc(start, 1);
        assert_eq!(next.month(), 2);
        assert_eq!(next.day(), 28);
        assert_eq!(next.hour(), 8);
        assert_eq!(next.minute(), 30);
    }

    #[test]
    fn previous_due_for_daily_moves_back_one_interval() {
        let reference = Utc
            .with_ymd_and_hms(2026, 2, 6, 9, 0, 0)
            .single()
            .expect("valid datetime");
        let rule = TaskRecurrenceRule::default();
        let prev = previous_due_at(reference, &rule);
        assert_eq!(
            prev,
            Utc.with_ymd_and_hms(2026, 2, 5, 9, 0, 0)
                .single()
                .expect("valid datetime")
        );
    }
}
