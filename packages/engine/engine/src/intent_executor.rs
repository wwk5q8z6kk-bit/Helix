//! Intent Executor - Handles the actual application of detected intents.
//!
//! When a user clicks "Apply" on an intent, this module executes the
//! appropriate action based on the intent type.

use chrono::{Datelike, Duration, Utc};
use hx_core::*;
use std::sync::Arc;
use uuid::Uuid;

use crate::engine::HelixEngine;

/// Result of executing an intent
#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecutionResult {
    pub success: bool,
    pub message: String,
    pub created_node_id: Option<Uuid>,
    pub modified_node_id: Option<Uuid>,
}

impl ExecutionResult {
    pub fn success(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
            created_node_id: None,
            modified_node_id: None,
        }
    }

    pub fn failure(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
            created_node_id: None,
            modified_node_id: None,
        }
    }

    pub fn with_created(mut self, id: Uuid) -> Self {
        self.created_node_id = Some(id);
        self
    }

    pub fn with_modified(mut self, id: Uuid) -> Self {
        self.modified_node_id = Some(id);
        self
    }
}

/// Executes intents by performing the suggested actions.
pub struct IntentExecutor {
    engine: Arc<HelixEngine>,
}

impl IntentExecutor {
    pub fn new(engine: Arc<HelixEngine>) -> Self {
        Self { engine }
    }

    /// Execute an intent and return the result.
    pub async fn execute(&self, intent: &CapturedIntent) -> MvResult<ExecutionResult> {
        match intent.intent_type {
            IntentType::ScheduleReminder => self.execute_schedule_reminder(intent).await,
            IntentType::ExtractTask => self.execute_extract_task(intent).await,
            IntentType::SuggestLink => self.execute_suggest_link(intent).await,
            IntentType::LinkToProject => self.execute_link_to_project(intent).await,
            IntentType::SuggestTag => self.execute_suggest_tag(intent).await,
            IntentType::Custom(_) => Ok(ExecutionResult::failure("Custom intents not yet supported")),
        }
    }

    /// Schedule a reminder by setting metadata on the source node
    async fn execute_schedule_reminder(&self, intent: &CapturedIntent) -> MvResult<ExecutionResult> {
        // Get the source node
        let node = self
            .engine
            .get_node(intent.node_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput(format!("Node {} not found", intent.node_id)))?;

        // Parse the reminder time from parameters
        let reminder_at = self.parse_reminder_time(&intent.parameters)?;

        // Update the node with reminder metadata
        let mut updated = node.clone();
        updated
            .metadata
            .insert("reminder_at".to_string(), serde_json::json!(reminder_at.to_rfc3339()));
        updated
            .metadata
            .insert("reminder_active".to_string(), serde_json::json!(true));

        self.engine.update_node(updated).await?;

        // Log to chronicle
        let entry = ChronicleEntry::new(
            "intent_executed",
            format!(
                "Scheduled reminder for {} at {}",
                intent.node_id,
                reminder_at.format("%Y-%m-%d %H:%M")
            ),
        )
        .with_node(intent.node_id);
        self.engine.log_chronicle(&entry).await?;

        Ok(ExecutionResult::success(format!(
            "Reminder scheduled for {}",
            reminder_at.format("%Y-%m-%d %H:%M")
        ))
        .with_modified(intent.node_id))
    }

    /// Extract a task from the source node
    async fn execute_extract_task(&self, intent: &CapturedIntent) -> MvResult<ExecutionResult> {
        // Get the source node
        let node = self
            .engine
            .get_node(intent.node_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput(format!("Node {} not found", intent.node_id)))?;

        // Extract task content from intent or fallback to node content
        let task_content = self.extract_task_content_from_intent(intent, &node);

        if task_content.is_empty() {
            return Ok(ExecutionResult::failure("No task content could be extracted"));
        }

        // Check parameters for deadline/priority info
        let deadline = self.extract_deadline_from_params(&intent.parameters);
        let priority = self.extract_priority_from_params(&intent.parameters);

        // Create a new task node
        let mut task = KnowledgeNode::new(NodeKind::Task, task_content.clone())
            .with_title(&self.extract_task_title(&task_content))
            .with_namespace(&node.namespace);

        // Copy tags from source
        task.tags = node.tags.clone();
        if !task.tags.contains(&"extracted".to_string()) {
            task.tags.push("extracted".to_string());
        }

        // Set task metadata
        task.metadata
            .insert("task_status".to_string(), serde_json::json!("inbox"));
        task.metadata
            .insert("source_node_id".to_string(), serde_json::json!(intent.node_id.to_string()));

        if let Some(deadline) = deadline {
            task.metadata
                .insert("task_due_at".to_string(), serde_json::json!(deadline.to_rfc3339()));
        }

        if let Some(priority) = priority {
            task.metadata
                .insert("task_priority".to_string(), serde_json::json!(priority));
        }

        let stored = self.engine.store_node(task).await?;

        // Log to chronicle
        let entry = ChronicleEntry::new(
            "intent_executed",
            format!("Extracted task from note {}", intent.node_id),
        )
        .with_node(stored.id);
        self.engine.log_chronicle(&entry).await?;

        Ok(ExecutionResult::success(format!("Task created: {}", self.extract_task_title(&task_content)))
            .with_created(stored.id))
    }

    /// Add a suggested link/relationship
    async fn execute_suggest_link(&self, intent: &CapturedIntent) -> MvResult<ExecutionResult> {
        let target = intent
            .parameters
            .get("target")
            .and_then(|v| v.as_str())
            .ok_or_else(|| HxError::InvalidInput("Missing target parameter".to_string()))?;

        // Find the target node by title
        let target_node = self.find_node_by_title(target).await?;

        if let Some(target_node) = target_node {
            // Create relationship
            let rel = Relationship::new(intent.node_id, target_node.id, RelationKind::References);
            self.engine.graph.add_relationship(&rel).await?;

            // Log to chronicle
            let entry = ChronicleEntry::new(
                "intent_executed",
                format!("Linked {} to {}", intent.node_id, target),
            )
            .with_node(intent.node_id);
            self.engine.log_chronicle(&entry).await?;

            Ok(ExecutionResult::success(format!("Linked to '{}'", target)))
        } else {
            Ok(ExecutionResult::failure(format!("Target node '{}' not found", target)))
        }
    }

    /// Link to a project/person (similar to suggest_link but may create node if not found)
    async fn execute_link_to_project(&self, intent: &CapturedIntent) -> MvResult<ExecutionResult> {
        let target = intent
            .parameters
            .get("target")
            .and_then(|v| v.as_str())
            .ok_or_else(|| HxError::InvalidInput("Missing target parameter".to_string()))?;

        let link_type = intent
            .parameters
            .get("link_type")
            .and_then(|v| v.as_str())
            .unwrap_or("reference");

        // Try to find existing node
        let target_node = self.find_node_by_title(target).await?;

        let target_id = if let Some(node) = target_node {
            node.id
        } else {
            // Create a placeholder node for the project/person
            let kind = if link_type == "mention" {
                NodeKind::Entity
            } else {
                NodeKind::Project
            };

            let source_node = self.engine.get_node(intent.node_id).await?;
            let namespace = source_node
                .as_ref()
                .map(|n| n.namespace.clone())
                .unwrap_or_default();

            let new_node = KnowledgeNode::new(kind, String::new())
                .with_title(target)
                .with_namespace(&namespace)
                .with_tags(vec!["auto-created".to_string()]);

            let stored = self.engine.store_node(new_node).await?;
            stored.id
        };

        // Create relationship
        let rel = Relationship::new(intent.node_id, target_id, RelationKind::References);
        self.engine.graph.add_relationship(&rel).await?;

        // Log to chronicle
        let entry = ChronicleEntry::new(
            "intent_executed",
            format!("Linked to {} ({})", target, link_type),
        )
        .with_node(intent.node_id);
        self.engine.log_chronicle(&entry).await?;

        Ok(ExecutionResult::success(format!("Linked to '{}'", target)))
    }

    /// Add a suggested tag to the source node
    async fn execute_suggest_tag(&self, intent: &CapturedIntent) -> MvResult<ExecutionResult> {
        let tag = intent
            .parameters
            .get("tag")
            .and_then(|v| v.as_str())
            .ok_or_else(|| HxError::InvalidInput("Missing tag parameter".to_string()))?;

        // Get the source node
        let node = self
            .engine
            .get_node(intent.node_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput(format!("Node {} not found", intent.node_id)))?;

        // Check if tag already exists
        if node.tags.iter().any(|t: &String| t.to_lowercase() == tag.to_lowercase()) {
            return Ok(ExecutionResult::success(format!("Tag '{}' already exists", tag)));
        }

        // Add the tag
        let mut updated = node.clone();
        updated.tags.push(tag.to_string());

        self.engine.update_node(updated).await?;

        // Log to chronicle
        let entry = ChronicleEntry::new("intent_executed", format!("Added tag #{} to note", tag))
            .with_node(intent.node_id);
        self.engine.log_chronicle(&entry).await?;

        Ok(ExecutionResult::success(format!("Added tag #{}", tag)).with_modified(intent.node_id))
    }

    // --- Helper methods ---

    fn parse_reminder_time(&self, params: &serde_json::Value) -> MvResult<chrono::DateTime<Utc>> {
        // Check for absolute datetime
        if let Some(at) = params.get("reminder_at").and_then(|v| v.as_str()) {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(at) {
                return Ok(dt.with_timezone(&Utc));
            }
        }

        // Check for relative time
        let relative = params
            .get("relative_time")
            .and_then(|v| v.as_str())
            .unwrap_or("tomorrow");

        let now = Utc::now();
        let reminder_at = match relative {
            "today" => now.date_naive().and_hms_opt(9, 0, 0).unwrap(),
            "tomorrow" => (now + Duration::days(1)).date_naive().and_hms_opt(9, 0, 0).unwrap(),
            "next_week" => (now + Duration::weeks(1)).date_naive().and_hms_opt(9, 0, 0).unwrap(),
            "next_month" => (now + Duration::days(30)).date_naive().and_hms_opt(9, 0, 0).unwrap(),
            "in_2_days" => (now + Duration::days(2)).date_naive().and_hms_opt(9, 0, 0).unwrap(),
            "in_3_days" => (now + Duration::days(3)).date_naive().and_hms_opt(9, 0, 0).unwrap(),
            other => {
                // Try to parse "in_N_days" pattern
                if other.starts_with("in_") && other.ends_with("_days") {
                    if let Ok(days) = other[3..other.len() - 5].parse::<i64>() {
                        (now + Duration::days(days)).date_naive().and_hms_opt(9, 0, 0).unwrap()
                    } else {
                        (now + Duration::days(1)).date_naive().and_hms_opt(9, 0, 0).unwrap()
                    }
                } else {
                    (now + Duration::days(1)).date_naive().and_hms_opt(9, 0, 0).unwrap()
                }
            }
        };

        Ok(chrono::DateTime::<Utc>::from_naive_utc_and_offset(reminder_at, Utc))
    }

    fn extract_task_content_from_intent(
        &self,
        intent: &CapturedIntent,
        node: &KnowledgeNode,
    ) -> String {
        if let Some(task) = intent.parameters.get("task").and_then(|v| v.as_str()) {
            let trimmed = task.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }

        self.extract_task_content(&node.content)
    }

    fn extract_task_content(&self, content: &str) -> String {
        let lines: Vec<&str> = content.lines().collect();

        // Look for checkbox items first
        for line in &lines {
            let trimmed = line.trim();
            if trimmed.starts_with("- [ ]") || trimmed.starts_with("- []") {
                return trimmed
                    .trim_start_matches("- [ ]")
                    .trim_start_matches("- []")
                    .trim()
                    .to_string();
            }
        }

        // Look for TODO: or TASK: prefixes
        for line in &lines {
            let lower = line.to_lowercase();
            if lower.contains("todo:") {
                if let Some(pos) = lower.find("todo:") {
                    return line[pos + 5..].trim().to_string();
                }
            }
            if lower.contains("task:") {
                if let Some(pos) = lower.find("task:") {
                    return line[pos + 5..].trim().to_string();
                }
            }
        }

        // Look for "need to", "must", "have to" patterns
        for line in &lines {
            let lower = line.to_lowercase();
            for pattern in &["need to ", "must ", "have to ", "should "] {
                if let Some(pos) = lower.find(pattern) {
                    return line[pos..].trim().to_string();
                }
            }
        }

        // Fallback: use first non-empty line
        lines
            .iter()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
            .unwrap_or_default()
    }

    fn extract_task_title(&self, content: &str) -> String {
        // Take first 80 chars, truncate at word boundary
        let title = if content.len() <= 80 {
            content.to_string()
        } else {
            let truncated = &content[..80];
            if let Some(last_space) = truncated.rfind(' ') {
                format!("{}...", &truncated[..last_space])
            } else {
                format!("{}...", truncated)
            }
        };

        // Remove markdown formatting
        title
            .replace("**", "")
            .replace("*", "")
            .replace("`", "")
            .trim()
            .to_string()
    }

    fn extract_deadline_from_params(&self, params: &serde_json::Value) -> Option<chrono::DateTime<Utc>> {
        // Check for explicit deadline
        if let Some(deadline_str) = params.get("deadline").and_then(|v| v.as_str()) {
            if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(deadline_str) {
                return Some(dt.with_timezone(&Utc));
            }
        }

        // Check for relative deadline
        if let Some(relative) = params.get("deadline_relative").and_then(|v| v.as_str()) {
            let now = Utc::now();
            let deadline = match relative {
                "today" => now.date_naive().and_hms_opt(23, 59, 59),
                "tomorrow" => (now + Duration::days(1)).date_naive().and_hms_opt(23, 59, 59),
                "this_week" => {
                    let days_to_friday = 5 - now.weekday().num_days_from_monday() as i64;
                    let days = if days_to_friday <= 0 { 7 + days_to_friday } else { days_to_friday };
                    (now + Duration::days(days)).date_naive().and_hms_opt(23, 59, 59)
                }
                "next_week" => (now + Duration::weeks(1)).date_naive().and_hms_opt(23, 59, 59),
                _ => None,
            };

            return deadline.map(|d| chrono::DateTime::<Utc>::from_naive_utc_and_offset(d, Utc));
        }

        None
    }

    fn extract_priority_from_params(&self, params: &serde_json::Value) -> Option<i32> {
        if let Some(priority) = params.get("priority").and_then(|v| v.as_i64()) {
            return Some(priority as i32);
        }

        if let Some(priority_str) = params.get("priority_label").and_then(|v| v.as_str()) {
            return match priority_str.to_lowercase().as_str() {
                "critical" | "p0" => Some(0),
                "high" | "urgent" | "p1" => Some(1),
                "medium" | "normal" | "p2" => Some(2),
                "low" | "p3" => Some(3),
                _ => None,
            };
        }

        None
    }

    async fn find_node_by_title(&self, title: &str) -> MvResult<Option<KnowledgeNode>> {
        // Search for nodes with matching title
        let filters = QueryFilters::default();
        let nodes = self.engine.list_nodes(&filters, 100, 0).await?;

        let lower_title = title.to_lowercase();
        Ok(nodes.into_iter().find(|n| {
            n.title
                .as_ref()
                .map(|t| t.to_lowercase() == lower_title)
                .unwrap_or(false)
        }))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_extract_task_content_checkbox() {
        let content = "Some notes\n- [ ] Buy groceries\n- [x] Done item";
        let task = extract_task_content_helper(content);
        assert_eq!(task, "Buy groceries");
    }

    #[test]
    fn test_extract_task_content_todo() {
        let content = "Meeting notes\nTODO: Follow up with client";
        let task = extract_task_content_helper(content);
        assert_eq!(task, "Follow up with client");
    }

    #[test]
    fn test_extract_task_title_truncates() {
        let long_content =
            "This is a very long task description that should be truncated at a reasonable word boundary";
        let title = extract_task_title_helper(long_content);
        assert!(title.len() <= 85); // 80 + "..."
        assert!(title.ends_with("..."));
    }

    fn extract_task_content_helper(content: &str) -> String {
        let lines: Vec<&str> = content.lines().collect();

        for line in &lines {
            let trimmed = line.trim();
            if trimmed.starts_with("- [ ]") || trimmed.starts_with("- []") {
                return trimmed
                    .trim_start_matches("- [ ]")
                    .trim_start_matches("- []")
                    .trim()
                    .to_string();
            }
        }

        for line in &lines {
            let lower = line.to_lowercase();
            if lower.contains("todo:") {
                if let Some(pos) = lower.find("todo:") {
                    return line[pos + 5..].trim().to_string();
                }
            }
        }

        String::new()
    }

    fn extract_task_title_helper(content: &str) -> String {
        if content.len() <= 80 {
            content.to_string()
        } else {
            let truncated = &content[..80];
            if let Some(last_space) = truncated.rfind(' ') {
                format!("{}...", &truncated[..last_space])
            } else {
                format!("{}...", truncated)
            }
        }
    }
}
