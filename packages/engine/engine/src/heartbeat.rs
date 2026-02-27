//! HEARTBEAT.md parser and scheduler.
//!
//! Reads task definitions from a markdown file and executes them on cron schedules.
//!
//! ## File Format
//!
//! ```markdown
//! ## Task Name
//! - cron: 0 */5 * * * *
//! - action: http_post https://example.com/hook
//! - enabled: true
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};

/// The action a heartbeat task performs when triggered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HeartbeatAction {
    /// Send an HTTP POST to the given URL.
    HttpPost { url: String },
    /// Run a built-in engine action by name.
    BuiltIn(String),
}

/// A single task parsed from HEARTBEAT.md.
#[derive(Debug, Clone)]
pub struct HeartbeatTask {
    /// Task name (from the `## ` heading).
    pub name: String,
    /// Cron expression for scheduling.
    pub cron_expr: String,
    /// The action to execute.
    pub action: HeartbeatAction,
    /// Whether the task is enabled.
    pub enabled: bool,
}

/// Parse HEARTBEAT.md content into task definitions.
///
/// Expected format:
/// ```text
/// ## Task Name
/// - cron: 0 */5 * * * *
/// - action: http_post https://example.com/hook
/// - enabled: true
/// ```
///
/// Tasks without a `cron` line are skipped with a warning. Actions default to
/// `BuiltIn("noop")` if not specified. Enabled defaults to `true`.
pub fn parse_heartbeat_md(content: &str) -> Vec<HeartbeatTask> {
    let mut tasks = Vec::new();
    let mut current_name: Option<String> = None;
    let mut cron_expr: Option<String> = None;
    let mut action: Option<HeartbeatAction> = None;
    let mut enabled = true;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("## ") {
            // Flush previous task
            if let Some(name) = current_name.take() {
                if let Some(cron) = cron_expr.take() {
                    tasks.push(HeartbeatTask {
                        name,
                        cron_expr: cron,
                        action: action.take().unwrap_or(HeartbeatAction::BuiltIn("noop".into())),
                        enabled,
                    });
                }
            }

            // Start new task
            current_name = Some(trimmed[3..].trim().to_string());
            cron_expr = None;
            action = None;
            enabled = true;
            continue;
        }

        // Parse bullet items within a task
        if current_name.is_some() {
            if let Some(rest) = trimmed.strip_prefix("- cron:") {
                cron_expr = Some(rest.trim().to_string());
            } else if let Some(rest) = trimmed.strip_prefix("- action:") {
                let rest = rest.trim();
                action = Some(parse_action(rest));
            } else if let Some(rest) = trimmed.strip_prefix("- enabled:") {
                enabled = rest.trim().eq_ignore_ascii_case("true");
            }
        }
    }

    // Flush final task
    if let Some(name) = current_name {
        if let Some(cron) = cron_expr {
            tasks.push(HeartbeatTask {
                name,
                cron_expr: cron,
                action: action.unwrap_or(HeartbeatAction::BuiltIn("noop".into())),
                enabled,
            });
        }
    }

    tasks
}

/// Parse an action string like "http_post https://..." or "builtin reindex".
fn parse_action(s: &str) -> HeartbeatAction {
    if let Some(rest) = s.strip_prefix("http_post ") {
        HeartbeatAction::HttpPost {
            url: rest.trim().to_string(),
        }
    } else if let Some(rest) = s.strip_prefix("builtin ") {
        HeartbeatAction::BuiltIn(rest.trim().to_string())
    } else {
        HeartbeatAction::BuiltIn(s.to_string())
    }
}

/// Load and parse a HEARTBEAT.md file from disk.
pub fn load_heartbeat_file(path: &Path) -> std::io::Result<Vec<HeartbeatTask>> {
    let content = std::fs::read_to_string(path)?;
    Ok(parse_heartbeat_md(&content))
}

/// Background scheduler that reads HEARTBEAT.md and runs tasks on their cron schedules.
///
/// This struct holds the parsed tasks and provides a `run` method that the engine
/// can spawn as a background Tokio task. The actual cron evaluation depends on the
/// `cron` crate being wired in by the integration layer.
pub struct HeartbeatScheduler {
    tasks: Vec<HeartbeatTask>,
}

impl HeartbeatScheduler {
    pub fn new(tasks: Vec<HeartbeatTask>) -> Self {
        Self { tasks }
    }

    pub fn from_file(path: &Path) -> std::io::Result<Self> {
        let tasks = load_heartbeat_file(path)?;
        Ok(Self { tasks })
    }

    pub fn tasks(&self) -> &[HeartbeatTask] {
        &self.tasks
    }

    pub fn enabled_tasks(&self) -> Vec<&HeartbeatTask> {
        self.tasks.iter().filter(|t| t.enabled).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_extracts_tasks() {
        let md = "\
## Health Ping
- cron: 0 */5 * * * *
- action: http_post https://example.com/health
- enabled: true

## Nightly Reindex
- cron: 0 0 3 * * *
- action: builtin reindex
- enabled: false
";

        let tasks = parse_heartbeat_md(md);
        assert_eq!(tasks.len(), 2);

        assert_eq!(tasks[0].name, "Health Ping");
        assert_eq!(tasks[0].cron_expr, "0 */5 * * * *");
        assert!(tasks[0].enabled);
        match &tasks[0].action {
            HeartbeatAction::HttpPost { url } => {
                assert_eq!(url, "https://example.com/health");
            }
            _ => panic!("expected HttpPost"),
        }

        assert_eq!(tasks[1].name, "Nightly Reindex");
        assert!(!tasks[1].enabled);
        match &tasks[1].action {
            HeartbeatAction::BuiltIn(name) => assert_eq!(name, "reindex"),
            _ => panic!("expected BuiltIn"),
        }
    }

    #[test]
    fn handles_missing_cron() {
        let md = "\
## No Cron Task
- action: builtin something
- enabled: true

## Valid Task
- cron: 0 0 * * * *
- action: builtin test
";
        let tasks = parse_heartbeat_md(md);
        // "No Cron Task" should be skipped because it lacks a cron line
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].name, "Valid Task");
    }

    #[test]
    fn action_types() {
        let md = "\
## Http Task
- cron: * * * * * *
- action: http_post https://hooks.example.com/notify

## Builtin Task
- cron: * * * * * *
- action: builtin vacuum

## Default Action
- cron: * * * * * *
";
        let tasks = parse_heartbeat_md(md);
        assert_eq!(tasks.len(), 3);

        match &tasks[0].action {
            HeartbeatAction::HttpPost { url } => {
                assert_eq!(url, "https://hooks.example.com/notify");
            }
            _ => panic!("expected HttpPost"),
        }
        match &tasks[1].action {
            HeartbeatAction::BuiltIn(name) => assert_eq!(name, "vacuum"),
            _ => panic!("expected BuiltIn"),
        }
        match &tasks[2].action {
            HeartbeatAction::BuiltIn(name) => assert_eq!(name, "noop"),
            _ => panic!("expected BuiltIn default"),
        }
    }

    #[test]
    fn scheduler_enabled_tasks() {
        let tasks = vec![
            HeartbeatTask {
                name: "a".into(),
                cron_expr: "* * * * *".into(),
                action: HeartbeatAction::BuiltIn("a".into()),
                enabled: true,
            },
            HeartbeatTask {
                name: "b".into(),
                cron_expr: "* * * * *".into(),
                action: HeartbeatAction::BuiltIn("b".into()),
                enabled: false,
            },
        ];
        let scheduler = HeartbeatScheduler::new(tasks);
        assert_eq!(scheduler.tasks().len(), 2);
        assert_eq!(scheduler.enabled_tasks().len(), 1);
        assert_eq!(scheduler.enabled_tasks()[0].name, "a");
    }
}
