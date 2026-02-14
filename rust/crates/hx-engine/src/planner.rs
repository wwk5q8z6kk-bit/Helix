//! Agent Planning Framework: LLM-driven task decomposition and execution.
//!
//! The planner takes a high-level goal, decomposes it into steps using the LLM,
//! presents the plan for user approval (via AutonomyGate), then executes steps
//! sequentially. Each step's output feeds into subsequent step context.
//!
//! Supported step actions:
//! - `recall`: Search the vault for relevant knowledge
//! - `store`: Create a new knowledge node
//! - `link`: Create a relationship between nodes
//! - `tag`: Add tags to a node
//! - `summarize`: Summarize content using the LLM
//! - `ask_user`: Request user input (pauses execution)

use std::sync::Arc;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::config::PlanningConfig;
use crate::llm::{ChatMessage, CompletionParams, LlmProvider};

/// Status of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Draft,
    Approved,
    Running,
    Completed,
    Failed,
    Cancelled,
}

impl std::fmt::Display for PlanStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Draft => write!(f, "draft"),
            Self::Approved => write!(f, "approved"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Status of a plan step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

impl std::fmt::Display for StepStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Running => write!(f, "running"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

/// Supported actions for plan steps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepAction {
    Recall,
    Store,
    Link,
    Tag,
    Summarize,
    AskUser,
}

impl std::fmt::Display for StepAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Recall => write!(f, "recall"),
            Self::Store => write!(f, "store"),
            Self::Link => write!(f, "link"),
            Self::Tag => write!(f, "tag"),
            Self::Summarize => write!(f, "summarize"),
            Self::AskUser => write!(f, "ask_user"),
        }
    }
}

/// A planned step in a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub step_order: usize,
    pub action: StepAction,
    pub description: String,
    pub status: StepStatus,
    pub input: serde_json::Value,
    pub output: Option<serde_json::Value>,
    pub error: Option<String>,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

/// A plan: goal + ordered steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: String,
    pub title: String,
    pub goal: String,
    pub status: PlanStatus,
    pub steps: Vec<PlanStep>,
    pub created_at: String,
    pub updated_at: String,
    pub completed_at: Option<String>,
    pub metadata: serde_json::Value,
}

/// The planner: decomposes goals into plans and executes them.
pub struct Planner {
    llm: Option<Arc<dyn LlmProvider>>,
    config: PlanningConfig,
}

impl Planner {
    pub fn new(llm: Option<Arc<dyn LlmProvider>>, config: PlanningConfig) -> Self {
        Self { llm, config }
    }

    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Decompose a goal into a draft plan using the LLM.
    /// Returns a Plan in Draft status with generated steps.
    pub async fn create_plan(&self, goal: &str) -> Plan {
        let plan_id = Uuid::now_v7().to_string();
        let now = Utc::now().to_rfc3339();

        let steps = if let Some(ref llm) = self.llm {
            match self.llm_decompose(llm, goal).await {
                Ok(steps) => steps,
                Err(e) => {
                    warn!(error = %e, "LLM plan decomposition failed, using heuristic");
                    self.heuristic_decompose(goal)
                }
            }
        } else {
            self.heuristic_decompose(goal)
        };

        let title = if goal.len() > 60 {
            format!("{}...", &goal[..57])
        } else {
            goal.to_string()
        };

        Plan {
            id: plan_id,
            title,
            goal: goal.to_string(),
            status: PlanStatus::Draft,
            steps,
            created_at: now.clone(),
            updated_at: now,
            completed_at: None,
            metadata: serde_json::json!({}),
        }
    }

    /// Use the LLM to decompose a goal into actionable steps.
    async fn llm_decompose(
        &self,
        llm: &Arc<dyn LlmProvider>,
        goal: &str,
    ) -> Result<Vec<PlanStep>, String> {
        let system = format!(
            "You are a planning assistant for a knowledge management system. \
             Given a user's goal, decompose it into a sequence of steps. \
             Each step must use one of these actions: recall, store, link, tag, summarize, ask_user.\n\
             \n\
             Actions:\n\
             - recall: Search the vault for relevant knowledge. Input: {{\"query\": \"...\"}}\n\
             - store: Create a new knowledge node. Input: {{\"content\": \"...\", \"kind\": \"fact|task|...\"}}\n\
             - link: Create a relationship. Input: {{\"from\": \"<ref>\", \"to\": \"<ref>\", \"label\": \"...\"}}\n\
             - tag: Add tags. Input: {{\"target\": \"<ref>\", \"tags\": [\"...\"]}}\n\
             - summarize: Summarize prior results. Input: {{\"context\": \"previous_results\"}}\n\
             - ask_user: Request user input. Input: {{\"question\": \"...\"}}\n\
             \n\
             Respond with a JSON array of objects, each having:\n\
             - \"action\": one of the actions above\n\
             - \"description\": human-readable description\n\
             - \"input\": JSON object with action parameters\n\
             \n\
             Maximum {max_steps} steps. Be concise and practical.",
            max_steps = self.config.max_steps
        );

        let messages = vec![
            ChatMessage::system(&system),
            ChatMessage::user(format!("Goal: {goal}")),
        ];

        let params = CompletionParams {
            max_tokens: Some(1024),
            temperature: Some(0.3),
            ..Default::default()
        };

        let result = llm
            .complete(&messages, &params)
            .await
            .map_err(|e| format!("LLM completion failed: {e}"))?;

        // Try to extract JSON array from the response
        let json_str = extract_json_array(&result).ok_or("no JSON array in LLM response")?;

        let raw_steps: Vec<serde_json::Value> =
            serde_json::from_str(&json_str).map_err(|e| format!("JSON parse error: {e}"))?;

        let mut steps = Vec::new();
        for (i, raw) in raw_steps.iter().enumerate().take(self.config.max_steps) {
            let action_str = raw["action"]
                .as_str()
                .unwrap_or("recall");
            let action = match action_str {
                "recall" => StepAction::Recall,
                "store" => StepAction::Store,
                "link" => StepAction::Link,
                "tag" => StepAction::Tag,
                "summarize" => StepAction::Summarize,
                "ask_user" => StepAction::AskUser,
                _ => StepAction::Recall,
            };

            steps.push(PlanStep {
                id: Uuid::now_v7().to_string(),
                step_order: i + 1,
                action,
                description: raw["description"]
                    .as_str()
                    .unwrap_or("(no description)")
                    .to_string(),
                status: StepStatus::Pending,
                input: raw["input"].clone(),
                output: None,
                error: None,
                started_at: None,
                completed_at: None,
            });
        }

        if steps.is_empty() {
            return Err("LLM produced no valid steps".to_string());
        }

        debug!(steps = steps.len(), "LLM plan decomposition complete");
        Ok(steps)
    }

    /// Heuristic fallback: create a simple plan when no LLM is available.
    fn heuristic_decompose(&self, goal: &str) -> Vec<PlanStep> {
        let mut steps = Vec::new();

        // Step 1: Always start with a recall to gather context
        steps.push(PlanStep {
            id: Uuid::now_v7().to_string(),
            step_order: 1,
            action: StepAction::Recall,
            description: format!("Search vault for knowledge related to: {goal}"),
            status: StepStatus::Pending,
            input: serde_json::json!({"query": goal}),
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        });

        // Step 2: Summarize findings
        steps.push(PlanStep {
            id: Uuid::now_v7().to_string(),
            step_order: 2,
            action: StepAction::Summarize,
            description: "Summarize the retrieved knowledge".to_string(),
            status: StepStatus::Pending,
            input: serde_json::json!({"context": "previous_results"}),
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        });

        // Step 3: Store the synthesis
        steps.push(PlanStep {
            id: Uuid::now_v7().to_string(),
            step_order: 3,
            action: StepAction::Store,
            description: format!("Store synthesis for: {goal}"),
            status: StepStatus::Pending,
            input: serde_json::json!({
                "content": format!("Synthesis for: {goal}"),
                "kind": "fact"
            }),
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        });

        steps
    }

    /// Execute a single step of a plan.
    /// Returns the updated step with output or error populated.
    pub async fn execute_step(
        &self,
        step: &PlanStep,
        prior_context: &[serde_json::Value],
    ) -> PlanStep {
        let mut result = step.clone();
        result.status = StepStatus::Running;
        result.started_at = Some(Utc::now().to_rfc3339());

        info!(
            step_id = %step.id,
            action = %step.action,
            order = step.step_order,
            "executing plan step"
        );

        // Step execution is delegated to the engine (via callbacks).
        // Here we build the execution context.
        let context_summary = if prior_context.is_empty() {
            String::new()
        } else {
            prior_context
                .iter()
                .filter_map(|v| v.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        };

        match step.action {
            StepAction::Recall => {
                let query = step.input["query"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                result.output = Some(serde_json::json!({
                    "status": "requires_engine_execution",
                    "action": "recall",
                    "query": query,
                    "context": context_summary,
                }));
                result.status = StepStatus::Completed;
            }
            StepAction::Store => {
                result.output = Some(serde_json::json!({
                    "status": "requires_engine_execution",
                    "action": "store",
                    "input": step.input,
                    "context": context_summary,
                }));
                result.status = StepStatus::Completed;
            }
            StepAction::Link => {
                result.output = Some(serde_json::json!({
                    "status": "requires_engine_execution",
                    "action": "link",
                    "input": step.input,
                }));
                result.status = StepStatus::Completed;
            }
            StepAction::Tag => {
                result.output = Some(serde_json::json!({
                    "status": "requires_engine_execution",
                    "action": "tag",
                    "input": step.input,
                }));
                result.status = StepStatus::Completed;
            }
            StepAction::Summarize => {
                if let Some(ref llm) = self.llm {
                    match self.summarize_context(llm, &context_summary).await {
                        Ok(summary) => {
                            result.output = Some(serde_json::json!({"summary": summary}));
                            result.status = StepStatus::Completed;
                        }
                        Err(e) => {
                            result.error = Some(e);
                            result.status = StepStatus::Failed;
                        }
                    }
                } else {
                    // Without LLM, pass through the context as-is
                    result.output = Some(serde_json::json!({
                        "summary": context_summary,
                        "note": "no LLM available, raw context returned"
                    }));
                    result.status = StepStatus::Completed;
                }
            }
            StepAction::AskUser => {
                let question = step.input["question"]
                    .as_str()
                    .unwrap_or("(no question specified)");
                result.output = Some(serde_json::json!({
                    "status": "awaiting_user_input",
                    "question": question,
                }));
                // Keep status as Running — caller must set to Completed
                // after user responds
                result.status = StepStatus::Running;
            }
        }

        result.completed_at = if result.status == StepStatus::Completed
            || result.status == StepStatus::Failed
        {
            Some(Utc::now().to_rfc3339())
        } else {
            None
        };

        result
    }

    /// LLM-based summarization of accumulated context.
    async fn summarize_context(
        &self,
        llm: &Arc<dyn LlmProvider>,
        context: &str,
    ) -> Result<String, String> {
        if context.is_empty() {
            return Ok("No context to summarize.".to_string());
        }

        let messages = vec![
            ChatMessage::system(
                "Summarize the following knowledge retrieval results concisely. \
                 Highlight key facts, connections, and actionable insights.",
            ),
            ChatMessage::user(context.to_string()),
        ];

        let params = CompletionParams {
            max_tokens: Some(512),
            temperature: Some(0.3),
            ..Default::default()
        };

        llm.complete(&messages, &params)
            .await
            .map_err(|e| format!("summarization failed: {e}"))
    }
}

/// Extract a JSON array from text that may contain markdown fences or prose.
fn extract_json_array(text: &str) -> Option<String> {
    // Try to find a JSON array directly
    if let Some(start) = text.find('[') {
        if let Some(end) = text.rfind(']') {
            if end > start {
                return Some(text[start..=end].to_string());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> PlanningConfig {
        PlanningConfig {
            enabled: true,
            max_steps: 5,
            require_approval: true,
        }
    }

    #[test]
    fn heuristic_decompose_creates_three_steps() {
        let planner = Planner::new(None, test_config());
        let steps = planner.heuristic_decompose("Learn about Rust ownership");
        assert_eq!(steps.len(), 3);
        assert_eq!(steps[0].action, StepAction::Recall);
        assert_eq!(steps[1].action, StepAction::Summarize);
        assert_eq!(steps[2].action, StepAction::Store);
    }

    #[tokio::test]
    async fn create_plan_without_llm() {
        let planner = Planner::new(None, test_config());
        let plan = planner.create_plan("Research quantum computing").await;
        assert_eq!(plan.status, PlanStatus::Draft);
        assert_eq!(plan.steps.len(), 3);
        assert!(plan.goal.contains("quantum"));
    }

    #[tokio::test]
    async fn execute_recall_step() {
        let planner = Planner::new(None, test_config());
        let step = PlanStep {
            id: "test-step".to_string(),
            step_order: 1,
            action: StepAction::Recall,
            description: "Search for topic".to_string(),
            status: StepStatus::Pending,
            input: serde_json::json!({"query": "test query"}),
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        };

        let result = planner.execute_step(&step, &[]).await;
        assert_eq!(result.status, StepStatus::Completed);
        assert!(result.output.is_some());
        assert!(result.started_at.is_some());
    }

    #[tokio::test]
    async fn execute_ask_user_step_stays_running() {
        let planner = Planner::new(None, test_config());
        let step = PlanStep {
            id: "ask-step".to_string(),
            step_order: 1,
            action: StepAction::AskUser,
            description: "Ask something".to_string(),
            status: StepStatus::Pending,
            input: serde_json::json!({"question": "What should we focus on?"}),
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        };

        let result = planner.execute_step(&step, &[]).await;
        assert_eq!(result.status, StepStatus::Running); // Stays running until user responds
        assert!(result.completed_at.is_none());
    }

    #[test]
    fn extract_json_array_from_markdown() {
        let text = "Here is the plan:\n```json\n[{\"action\":\"recall\"}]\n```\nDone.";
        let result = extract_json_array(text);
        assert!(result.is_some());
        assert!(result.unwrap().starts_with('['));
    }

    #[test]
    fn extract_json_array_direct() {
        let text = "[{\"action\":\"store\"}]";
        let result = extract_json_array(text);
        assert_eq!(result.unwrap(), text);
    }

    #[test]
    fn plan_status_display() {
        assert_eq!(PlanStatus::Draft.to_string(), "draft");
        assert_eq!(PlanStatus::Running.to_string(), "running");
        assert_eq!(PlanStatus::Completed.to_string(), "completed");
    }
}
