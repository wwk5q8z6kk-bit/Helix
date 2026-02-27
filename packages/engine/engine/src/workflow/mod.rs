use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

pub mod executor;
pub mod parser;
pub mod scheduler;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub trigger: WorkflowTrigger,
    pub steps: Vec<WorkflowStep>,
    #[serde(default)]
    pub variables: HashMap<String, serde_json::Value>,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default = "chrono::Utc::now")]
    pub created_at: DateTime<Utc>,
}

fn default_timeout() -> u64 {
    300
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum WorkflowTrigger {
    Manual,
    Cron { expression: String },
    OnEvent { event_type: String },
    OnIngest { namespace: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    pub name: String,
    #[serde(flatten)]
    pub step_type: StepType,
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "action")]
pub enum StepType {
    IngestFile {
        path: String,
        namespace: String,
    },
    RunSearch {
        query: String,
        limit: Option<usize>,
    },
    CallLlm {
        prompt: String,
        model: Option<String>,
    },
    SendMessage {
        channel: String,
        content: String,
    },
    HttpRequest {
        method: String,
        url: String,
        body: Option<String>,
        #[serde(default)]
        headers: Option<HashMap<String, String>>,
    },
    SetVariable {
        var_name: String,
        value: serde_json::Value,
    },
    Conditional {
        condition: String,
        then_steps: Vec<WorkflowStep>,
        #[serde(default)]
        else_steps: Vec<WorkflowStep>,
    },
    Parallel {
        branches: Vec<Vec<WorkflowStep>>,
    },
    Loop {
        collection: String,
        variable: String,
        body: Vec<WorkflowStep>,
        max_iterations: Option<usize>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowExecution {
    pub id: Uuid,
    pub workflow_id: Uuid,
    pub workflow_name: String,
    pub status: ExecutionStatus,
    pub step_results: Vec<StepResult>,
    pub variables: HashMap<String, serde_json::Value>,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub step_name: String,
    pub status: ExecutionStatus,
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
    pub duration_ms: u64,
}

impl Workflow {
    pub fn new(name: impl Into<String>, trigger: WorkflowTrigger) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.into(),
            description: None,
            trigger,
            steps: Vec::new(),
            variables: HashMap::new(),
            timeout_secs: default_timeout(),
            created_at: Utc::now(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_step(mut self, step: WorkflowStep) -> Self {
        self.steps.push(step);
        self
    }

    pub fn with_variable(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.variables.insert(name.into(), value);
        self
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

impl WorkflowStep {
    pub fn new(name: impl Into<String>, step_type: StepType) -> Self {
        Self {
            name: name.into(),
            step_type,
            timeout_secs: None,
        }
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }
}

impl WorkflowExecution {
    pub fn new(workflow: &Workflow) -> Self {
        Self {
            id: Uuid::now_v7(),
            workflow_id: workflow.id,
            workflow_name: workflow.name.clone(),
            status: ExecutionStatus::Pending,
            step_results: Vec::new(),
            variables: workflow.variables.clone(),
            started_at: Utc::now(),
            completed_at: None,
            error: None,
        }
    }
}

impl StepResult {
    pub fn success(step_name: impl Into<String>, output: serde_json::Value, duration_ms: u64) -> Self {
        Self {
            step_name: step_name.into(),
            status: ExecutionStatus::Completed,
            output: Some(output),
            error: None,
            duration_ms,
        }
    }

    pub fn failure(step_name: impl Into<String>, error: impl Into<String>, duration_ms: u64) -> Self {
        Self {
            step_name: step_name.into(),
            status: ExecutionStatus::Failed,
            output: None,
            error: Some(error.into()),
            duration_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_builder_creates_valid_workflow() {
        let wf = Workflow::new("test-workflow", WorkflowTrigger::Manual)
            .with_description("A test workflow")
            .with_timeout(600)
            .with_variable("count", serde_json::json!(0));

        assert_eq!(wf.name, "test-workflow");
        assert_eq!(wf.description.as_deref(), Some("A test workflow"));
        assert_eq!(wf.timeout_secs, 600);
        assert_eq!(wf.variables.get("count"), Some(&serde_json::json!(0)));
        assert!(wf.steps.is_empty());
    }

    #[test]
    fn workflow_with_steps() {
        let wf = Workflow::new("multi-step", WorkflowTrigger::Manual)
            .with_step(WorkflowStep::new(
                "set-greeting",
                StepType::SetVariable {
                    var_name: "greeting".into(),
                    value: serde_json::json!("hello"),
                },
            ))
            .with_step(WorkflowStep::new(
                "search",
                StepType::RunSearch {
                    query: "test query".into(),
                    limit: Some(10),
                },
            ));

        assert_eq!(wf.steps.len(), 2);
        assert_eq!(wf.steps[0].name, "set-greeting");
        assert_eq!(wf.steps[1].name, "search");
    }

    #[test]
    fn step_with_timeout() {
        let step = WorkflowStep::new(
            "slow-step",
            StepType::HttpRequest {
                method: "GET".into(),
                url: "https://example.com".into(),
                body: None,
                headers: None,
            },
        )
        .with_timeout(30);

        assert_eq!(step.timeout_secs, Some(30));
    }

    #[test]
    fn trigger_manual_serializes() {
        let trigger = WorkflowTrigger::Manual;
        let json = serde_json::to_value(&trigger).unwrap();
        assert_eq!(json["type"], "manual");
    }

    #[test]
    fn trigger_cron_serializes() {
        let trigger = WorkflowTrigger::Cron {
            expression: "0 * * * *".into(),
        };
        let json = serde_json::to_value(&trigger).unwrap();
        assert_eq!(json["type"], "cron");
        assert_eq!(json["expression"], "0 * * * *");
    }

    #[test]
    fn trigger_on_event_serializes() {
        let trigger = WorkflowTrigger::OnEvent {
            event_type: "node_created".into(),
        };
        let json = serde_json::to_value(&trigger).unwrap();
        assert_eq!(json["type"], "on_event");
    }

    #[test]
    fn trigger_on_ingest_serializes() {
        let trigger = WorkflowTrigger::OnIngest {
            namespace: "docs".into(),
        };
        let json = serde_json::to_value(&trigger).unwrap();
        assert_eq!(json["type"], "on_ingest");
        assert_eq!(json["namespace"], "docs");
    }

    #[test]
    fn execution_status_serializes_snake_case() {
        assert_eq!(
            serde_json::to_value(ExecutionStatus::Pending).unwrap(),
            serde_json::json!("pending")
        );
        assert_eq!(
            serde_json::to_value(ExecutionStatus::Running).unwrap(),
            serde_json::json!("running")
        );
        assert_eq!(
            serde_json::to_value(ExecutionStatus::Completed).unwrap(),
            serde_json::json!("completed")
        );
        assert_eq!(
            serde_json::to_value(ExecutionStatus::Failed).unwrap(),
            serde_json::json!("failed")
        );
        assert_eq!(
            serde_json::to_value(ExecutionStatus::Cancelled).unwrap(),
            serde_json::json!("cancelled")
        );
    }

    #[test]
    fn execution_status_roundtrips() {
        for status in [
            ExecutionStatus::Pending,
            ExecutionStatus::Running,
            ExecutionStatus::Completed,
            ExecutionStatus::Failed,
            ExecutionStatus::Cancelled,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: ExecutionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }

    #[test]
    fn step_result_success_constructor() {
        let result = StepResult::success("step-1", serde_json::json!({"count": 42}), 150);
        assert_eq!(result.step_name, "step-1");
        assert_eq!(result.status, ExecutionStatus::Completed);
        assert!(result.output.is_some());
        assert!(result.error.is_none());
        assert_eq!(result.duration_ms, 150);
    }

    #[test]
    fn step_result_failure_constructor() {
        let result = StepResult::failure("step-2", "connection timeout", 5000);
        assert_eq!(result.step_name, "step-2");
        assert_eq!(result.status, ExecutionStatus::Failed);
        assert!(result.output.is_none());
        assert_eq!(result.error.as_deref(), Some("connection timeout"));
    }

    #[test]
    fn workflow_execution_inherits_workflow_vars() {
        let wf = Workflow::new("exec-test", WorkflowTrigger::Manual)
            .with_variable("api_key", serde_json::json!("sk-123"));

        let exec = WorkflowExecution::new(&wf);
        assert_eq!(exec.workflow_id, wf.id);
        assert_eq!(exec.workflow_name, "exec-test");
        assert_eq!(exec.status, ExecutionStatus::Pending);
        assert_eq!(
            exec.variables.get("api_key"),
            Some(&serde_json::json!("sk-123"))
        );
        assert!(exec.completed_at.is_none());
        assert!(exec.error.is_none());
    }

    #[test]
    fn workflow_json_roundtrip() {
        let wf = Workflow::new("roundtrip", WorkflowTrigger::Manual)
            .with_step(WorkflowStep::new(
                "search",
                StepType::RunSearch {
                    query: "hello".into(),
                    limit: None,
                },
            ));

        let json = serde_json::to_string(&wf).unwrap();
        let back: Workflow = serde_json::from_str(&json).unwrap();
        assert_eq!(back.name, "roundtrip");
        assert_eq!(back.steps.len(), 1);
    }

    #[test]
    fn conditional_step_serializes() {
        let step = WorkflowStep::new(
            "check-flag",
            StepType::Conditional {
                condition: "flag".into(),
                then_steps: vec![WorkflowStep::new(
                    "then-action",
                    StepType::SetVariable {
                        var_name: "result".into(),
                        value: serde_json::json!("yes"),
                    },
                )],
                else_steps: vec![],
            },
        );
        let json = serde_json::to_value(&step).unwrap();
        assert_eq!(json["action"], "conditional");
        assert_eq!(json["condition"], "flag");
    }

    #[test]
    fn parallel_step_serializes() {
        let step = WorkflowStep::new(
            "fan-out",
            StepType::Parallel {
                branches: vec![
                    vec![WorkflowStep::new(
                        "branch-a",
                        StepType::SetVariable {
                            var_name: "a".into(),
                            value: serde_json::json!(1),
                        },
                    )],
                    vec![WorkflowStep::new(
                        "branch-b",
                        StepType::SetVariable {
                            var_name: "b".into(),
                            value: serde_json::json!(2),
                        },
                    )],
                ],
            },
        );
        let json = serde_json::to_value(&step).unwrap();
        assert_eq!(json["action"], "parallel");
        let branches = json["branches"].as_array().unwrap();
        assert_eq!(branches.len(), 2);
    }
}
