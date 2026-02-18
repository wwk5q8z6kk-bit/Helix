use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use hx_core::{HxError, MvResult};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::{
    ExecutionStatus, StepResult, StepType, Workflow, WorkflowExecution, WorkflowStep,
};

pub struct WorkflowExecutor {
    workflows: Arc<RwLock<HashMap<Uuid, Workflow>>>,
    executions: Arc<RwLock<HashMap<Uuid, WorkflowExecution>>>,
    http_client: reqwest::Client,
}

impl WorkflowExecutor {
    pub fn new() -> Self {
        Self {
            workflows: Arc::new(RwLock::new(HashMap::new())),
            executions: Arc::new(RwLock::new(HashMap::new())),
            http_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
        }
    }

    pub async fn load_workflows(&self, workflows: Vec<Workflow>) {
        let mut store = self.workflows.write().await;
        for wf in workflows {
            store.insert(wf.id, wf);
        }
    }

    pub async fn register_workflow(&self, workflow: Workflow) -> Uuid {
        let id = workflow.id;
        self.workflows.write().await.insert(id, workflow);
        id
    }

    pub async fn get_workflow(&self, id: Uuid) -> Option<Workflow> {
        self.workflows.read().await.get(&id).cloned()
    }

    pub async fn list_workflows(&self) -> Vec<Workflow> {
        self.workflows.read().await.values().cloned().collect()
    }

    pub async fn get_execution(&self, execution_id: Uuid) -> Option<WorkflowExecution> {
        self.executions.read().await.get(&execution_id).cloned()
    }

    pub async fn list_executions(&self) -> Vec<WorkflowExecution> {
        self.executions.read().await.values().cloned().collect()
    }

    pub async fn execute(
        &self,
        workflow_id: Uuid,
        initial_vars: HashMap<String, serde_json::Value>,
    ) -> MvResult<WorkflowExecution> {
        let workflow = self
            .workflows
            .read()
            .await
            .get(&workflow_id)
            .cloned()
            .ok_or_else(|| HxError::InvalidInput(format!("workflow not found: {workflow_id}")))?;

        let mut execution = WorkflowExecution::new(&workflow);
        execution.variables.extend(initial_vars);
        execution.status = ExecutionStatus::Running;

        let exec_id = execution.id;
        self.executions.write().await.insert(exec_id, execution.clone());

        let mut variables = execution.variables.clone();
        let mut results: Vec<StepResult> = Vec::new();

        for step in &workflow.steps {
            // Check if execution was cancelled
            if let Some(exec) = self.executions.read().await.get(&exec_id) {
                if exec.status == ExecutionStatus::Cancelled {
                    let mut exec = execution.clone();
                    exec.status = ExecutionStatus::Cancelled;
                    exec.step_results = results;
                    exec.completed_at = Some(Utc::now());
                    self.executions.write().await.insert(exec_id, exec.clone());
                    return Ok(exec);
                }
            }

            if let Err(e) = execute_step(&self.http_client, step, &mut variables, &mut results).await {
                execution.status = ExecutionStatus::Failed;
                execution.error = Some(e.to_string());
                execution.step_results = results;
                execution.variables = variables;
                execution.completed_at = Some(Utc::now());
                self.executions.write().await.insert(exec_id, execution.clone());
                return Ok(execution);
            }
        }

        execution.status = ExecutionStatus::Completed;
        execution.step_results = results;
        execution.variables = variables;
        execution.completed_at = Some(Utc::now());
        self.executions.write().await.insert(exec_id, execution.clone());
        Ok(execution)
    }

    pub async fn cancel(&self, execution_id: Uuid) -> MvResult<()> {
        let mut execs = self.executions.write().await;
        let exec = execs
            .get_mut(&execution_id)
            .ok_or_else(|| HxError::InvalidInput(format!("execution not found: {execution_id}")))?;

        if exec.status == ExecutionStatus::Running || exec.status == ExecutionStatus::Pending {
            exec.status = ExecutionStatus::Cancelled;
            exec.completed_at = Some(Utc::now());
        }
        Ok(())
    }
}

/// Replace `{{var.name}}` and `{{result.step_name.field}}` placeholders in a template string.
fn interpolate(
    template: &str,
    variables: &HashMap<String, serde_json::Value>,
    results: &[StepResult],
) -> String {
    let mut output = template.to_string();

    // Replace {{var.name}} patterns
    for (key, val) in variables {
        let placeholder = format!("{{{{var.{key}}}}}");
        let replacement = match val {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        output = output.replace(&placeholder, &replacement);
    }

    // Replace {{result.step_name.field}} patterns
    for result in results {
        if let Some(ref output_val) = result.output {
            // Direct reference: {{result.step_name}}
            let direct_placeholder = format!("{{{{result.{}}}}}", result.step_name);
            let direct_replacement = match output_val {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            output = output.replace(&direct_placeholder, &direct_replacement);

            // Field reference: {{result.step_name.field}}
            if let serde_json::Value::Object(map) = output_val {
                for (field, field_val) in map {
                    let field_placeholder =
                        format!("{{{{result.{}.{field}}}}}", result.step_name);
                    let field_replacement = match field_val {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    output = output.replace(&field_placeholder, &field_replacement);
                }
            }
        }
    }

    output
}

/// Execute a single workflow step. This is a free function (not a method)
/// to allow `Box::pin` for recursive calls without self-referential issues.
fn execute_step<'a>(
    http_client: &'a reqwest::Client,
    step: &'a WorkflowStep,
    variables: &'a mut HashMap<String, serde_json::Value>,
    results: &'a mut Vec<StepResult>,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = MvResult<()>> + Send + 'a>> {
    Box::pin(async move {
        let start = Instant::now();
        let step_name = step.name.clone();

        match &step.step_type {
            StepType::SetVariable { var_name, value } => {
                let interpolated_str = interpolate(
                    &serde_json::to_string(value).unwrap_or_default(),
                    variables,
                    results,
                );
                let resolved: serde_json::Value =
                    serde_json::from_str(&interpolated_str).unwrap_or(value.clone());
                variables.insert(var_name.clone(), resolved.clone());
                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({ "variable": var_name, "value": resolved }),
                    start.elapsed().as_millis() as u64,
                ));
            }

            StepType::HttpRequest {
                method,
                url,
                body,
                headers,
            } => {
                let resolved_url = interpolate(url, variables, results);
                let resolved_body = body
                    .as_ref()
                    .map(|b| interpolate(b, variables, results));

                let mut req = match method.to_uppercase().as_str() {
                    "GET" => http_client.get(&resolved_url),
                    "POST" => http_client.post(&resolved_url),
                    "PUT" => http_client.put(&resolved_url),
                    "DELETE" => http_client.delete(&resolved_url),
                    "PATCH" => http_client.patch(&resolved_url),
                    other => {
                        let err = format!("unsupported HTTP method: {other}");
                        results.push(StepResult::failure(
                            &step_name,
                            &err,
                            start.elapsed().as_millis() as u64,
                        ));
                        return Err(HxError::InvalidInput(err));
                    }
                };

                if let Some(ref hdrs) = headers {
                    for (k, v) in hdrs {
                        req = req.header(k.as_str(), v.as_str());
                    }
                }
                if let Some(ref b) = resolved_body {
                    req = req.body(b.clone());
                }

                match req.send().await {
                    Ok(resp) => {
                        let status_code = resp.status().as_u16();
                        let resp_body = resp.text().await.unwrap_or_default();
                        let output = serde_json::json!({
                            "status": status_code,
                            "body": resp_body,
                        });
                        results.push(StepResult::success(
                            &step_name,
                            output,
                            start.elapsed().as_millis() as u64,
                        ));
                    }
                    Err(e) => {
                        let err = format!("HTTP request failed: {e}");
                        results.push(StepResult::failure(
                            &step_name,
                            &err,
                            start.elapsed().as_millis() as u64,
                        ));
                        return Err(HxError::Internal(err));
                    }
                }
            }

            StepType::Conditional {
                condition,
                then_steps,
                else_steps,
            } => {
                let resolved_condition = interpolate(condition, variables, results);
                let is_truthy = evaluate_condition(&resolved_condition, variables);

                let branch = if is_truthy { then_steps } else { else_steps };
                for sub_step in branch {
                    execute_step(http_client, sub_step, variables, results).await?;
                }

                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({ "branch": if is_truthy { "then" } else { "else" } }),
                    start.elapsed().as_millis() as u64,
                ));
            }

            StepType::Loop {
                collection,
                variable,
                body,
                max_iterations,
            } => {
                let items = variables
                    .get(collection)
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();

                let max = max_iterations.unwrap_or(1000);
                let mut iteration_count = 0usize;

                for item in items {
                    if iteration_count >= max {
                        break;
                    }
                    variables.insert(variable.clone(), item);
                    for sub_step in body {
                        execute_step(http_client, sub_step, variables, results).await?;
                    }
                    iteration_count += 1;
                }

                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({ "iterations": iteration_count }),
                    start.elapsed().as_millis() as u64,
                ));
            }

            StepType::Parallel { branches } => {
                // Execute branches concurrently using join_all.
                // Each branch gets its own copy of variables and results.
                let mut handles = Vec::new();
                for branch in branches {
                    let branch = branch.clone();
                    let branch_vars = variables.clone();
                    let client = http_client.clone();

                    handles.push(tokio::spawn(async move {
                        let mut vars = branch_vars;
                        let mut branch_results: Vec<StepResult> = Vec::new();
                        for sub_step in &branch {
                            execute_step(&client, sub_step, &mut vars, &mut branch_results).await?;
                        }
                        Ok::<_, HxError>((vars, branch_results))
                    }));
                }

                let mut all_branch_results = Vec::new();
                for handle in handles {
                    match handle.await {
                        Ok(Ok((_branch_vars, branch_results))) => {
                            all_branch_results.extend(branch_results);
                        }
                        Ok(Err(e)) => {
                            results.push(StepResult::failure(
                                &step_name,
                                e.to_string(),
                                start.elapsed().as_millis() as u64,
                            ));
                            return Err(e);
                        }
                        Err(e) => {
                            let err = format!("parallel branch panicked: {e}");
                            results.push(StepResult::failure(
                                &step_name,
                                &err,
                                start.elapsed().as_millis() as u64,
                            ));
                            return Err(HxError::Internal(err));
                        }
                    }
                }

                results.extend(all_branch_results);
                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({ "branches_completed": branches.len() }),
                    start.elapsed().as_millis() as u64,
                ));
            }

            // Steps that interact with the engine would need engine references;
            // for now these produce descriptive outputs for testing/validation.
            StepType::IngestFile { path, namespace } => {
                let resolved_path = interpolate(path, variables, results);
                let resolved_ns = interpolate(namespace, variables, results);
                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({
                        "action": "ingest_file",
                        "path": resolved_path,
                        "namespace": resolved_ns,
                    }),
                    start.elapsed().as_millis() as u64,
                ));
            }

            StepType::RunSearch { query, limit } => {
                let resolved_query = interpolate(query, variables, results);
                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({
                        "action": "run_search",
                        "query": resolved_query,
                        "limit": limit,
                    }),
                    start.elapsed().as_millis() as u64,
                ));
            }

            StepType::CallLlm { prompt, model } => {
                let resolved_prompt = interpolate(prompt, variables, results);
                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({
                        "action": "call_llm",
                        "prompt": resolved_prompt,
                        "model": model,
                    }),
                    start.elapsed().as_millis() as u64,
                ));
            }

            StepType::SendMessage { channel, content } => {
                let resolved_channel = interpolate(channel, variables, results);
                let resolved_content = interpolate(content, variables, results);
                results.push(StepResult::success(
                    &step_name,
                    serde_json::json!({
                        "action": "send_message",
                        "channel": resolved_channel,
                        "content": resolved_content,
                    }),
                    start.elapsed().as_millis() as u64,
                ));
            }
        }

        Ok(())
    })
}

/// Evaluate a condition string for truthiness. If the condition matches a variable
/// name, check if that variable is truthy. Otherwise treat non-empty, non-"false"
/// strings as truthy.
fn evaluate_condition(
    condition: &str,
    variables: &HashMap<String, serde_json::Value>,
) -> bool {
    if let Some(val) = variables.get(condition) {
        return is_truthy(val);
    }
    !condition.is_empty() && condition != "false" && condition != "0"
}

fn is_truthy(val: &serde_json::Value) -> bool {
    match val {
        serde_json::Value::Null => false,
        serde_json::Value::Bool(b) => *b,
        serde_json::Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        serde_json::Value::String(s) => !s.is_empty() && s != "false" && s != "0",
        serde_json::Value::Array(a) => !a.is_empty(),
        serde_json::Value::Object(o) => !o.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_executor() -> WorkflowExecutor {
        WorkflowExecutor::new()
    }

    #[tokio::test]
    async fn execute_empty_workflow() {
        let executor = make_executor();
        let wf = Workflow::new("empty", super::super::WorkflowTrigger::Manual);
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(exec.status, ExecutionStatus::Completed);
        assert!(exec.step_results.is_empty());
        assert!(exec.completed_at.is_some());
    }

    #[tokio::test]
    async fn execute_set_variable_step() {
        let executor = make_executor();
        let wf = Workflow::new("set-var", super::super::WorkflowTrigger::Manual).with_step(
            WorkflowStep::new(
                "set-x",
                StepType::SetVariable {
                    var_name: "x".into(),
                    value: serde_json::json!(42),
                },
            ),
        );
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(exec.status, ExecutionStatus::Completed);
        assert_eq!(exec.variables.get("x"), Some(&serde_json::json!(42)));
        assert_eq!(exec.step_results.len(), 1);
        assert_eq!(exec.step_results[0].status, ExecutionStatus::Completed);
    }

    #[tokio::test]
    async fn execute_conditional_then_branch() {
        let executor = make_executor();
        let wf = Workflow::new("cond", super::super::WorkflowTrigger::Manual)
            .with_variable("flag", serde_json::json!(true))
            .with_step(WorkflowStep::new(
                "check",
                StepType::Conditional {
                    condition: "flag".into(),
                    then_steps: vec![WorkflowStep::new(
                        "then-set",
                        StepType::SetVariable {
                            var_name: "result".into(),
                            value: serde_json::json!("yes"),
                        },
                    )],
                    else_steps: vec![WorkflowStep::new(
                        "else-set",
                        StepType::SetVariable {
                            var_name: "result".into(),
                            value: serde_json::json!("no"),
                        },
                    )],
                },
            ));
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(exec.status, ExecutionStatus::Completed);
        assert_eq!(exec.variables.get("result"), Some(&serde_json::json!("yes")));
    }

    #[tokio::test]
    async fn execute_conditional_else_branch() {
        let executor = make_executor();
        let wf = Workflow::new("cond-else", super::super::WorkflowTrigger::Manual)
            .with_variable("flag", serde_json::json!(false))
            .with_step(WorkflowStep::new(
                "check",
                StepType::Conditional {
                    condition: "flag".into(),
                    then_steps: vec![WorkflowStep::new(
                        "then-set",
                        StepType::SetVariable {
                            var_name: "result".into(),
                            value: serde_json::json!("yes"),
                        },
                    )],
                    else_steps: vec![WorkflowStep::new(
                        "else-set",
                        StepType::SetVariable {
                            var_name: "result".into(),
                            value: serde_json::json!("no"),
                        },
                    )],
                },
            ));
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(exec.variables.get("result"), Some(&serde_json::json!("no")));
    }

    #[tokio::test]
    async fn execute_loop_step() {
        let executor = make_executor();
        let wf = Workflow::new("loop", super::super::WorkflowTrigger::Manual)
            .with_variable("items", serde_json::json!(["a", "b", "c"]))
            .with_step(WorkflowStep::new(
                "iterate",
                StepType::Loop {
                    collection: "items".into(),
                    variable: "item".into(),
                    body: vec![WorkflowStep::new(
                        "log-item",
                        StepType::SetVariable {
                            var_name: "last_item".into(),
                            value: serde_json::json!("{{var.item}}"),
                        },
                    )],
                    max_iterations: None,
                },
            ));
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(exec.status, ExecutionStatus::Completed);
        // Last item should be "c"
        assert_eq!(exec.variables.get("item"), Some(&serde_json::json!("c")));
    }

    #[tokio::test]
    async fn execute_loop_with_max_iterations() {
        let executor = make_executor();
        let wf = Workflow::new("loop-limited", super::super::WorkflowTrigger::Manual)
            .with_variable("items", serde_json::json!([1, 2, 3, 4, 5]))
            .with_step(WorkflowStep::new(
                "iterate",
                StepType::Loop {
                    collection: "items".into(),
                    variable: "n".into(),
                    body: vec![],
                    max_iterations: Some(2),
                },
            ));
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(exec.status, ExecutionStatus::Completed);
        // Only 2 iterations
        let loop_result = exec
            .step_results
            .iter()
            .find(|r| r.step_name == "iterate")
            .unwrap();
        let iterations = loop_result.output.as_ref().unwrap()["iterations"]
            .as_u64()
            .unwrap();
        assert_eq!(iterations, 2);
    }

    #[tokio::test]
    async fn execute_parallel_branches() {
        let executor = make_executor();
        let wf = Workflow::new("parallel", super::super::WorkflowTrigger::Manual).with_step(
            WorkflowStep::new(
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
            ),
        );
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(exec.status, ExecutionStatus::Completed);
        // Should have results from both branches + the parallel step itself
        assert!(exec.step_results.len() >= 3);
    }

    #[tokio::test]
    async fn cancel_execution() {
        let executor = make_executor();
        let wf = Workflow::new("cancel-me", super::super::WorkflowTrigger::Manual);
        let wf_id = executor.register_workflow(wf).await;

        let exec = executor.execute(wf_id, HashMap::new()).await.unwrap();
        // Already completed — cancelling a completed execution should be a no-op
        let cancel_result = executor.cancel(exec.id).await;
        assert!(cancel_result.is_ok());
    }

    #[tokio::test]
    async fn execute_nonexistent_workflow_fails() {
        let executor = make_executor();
        let result = executor.execute(Uuid::now_v7(), HashMap::new()).await;
        assert!(result.is_err());
    }

    #[test]
    fn interpolate_variables() {
        let mut vars = HashMap::new();
        vars.insert("name".to_string(), serde_json::json!("Alice"));
        vars.insert("count".to_string(), serde_json::json!(42));

        let result = interpolate("Hello {{var.name}}, count={{var.count}}", &vars, &[]);
        assert_eq!(result, "Hello Alice, count=42");
    }

    #[test]
    fn interpolate_step_results() {
        let vars = HashMap::new();
        let results = vec![StepResult::success(
            "search",
            serde_json::json!({"total": 5}),
            10,
        )];

        let result = interpolate("Found {{result.search.total}} items", &vars, &results);
        assert_eq!(result, "Found 5 items");
    }

    #[test]
    fn interpolate_missing_variable_left_as_is() {
        let vars = HashMap::new();

        let result = interpolate("Hello {{var.missing}}", &vars, &[]);
        assert_eq!(result, "Hello {{var.missing}}");
    }

    #[test]
    fn evaluate_condition_truthy_values() {
        let mut vars = HashMap::new();
        vars.insert("flag".to_string(), serde_json::json!(true));
        vars.insert("zero".to_string(), serde_json::json!(0));
        vars.insert("empty".to_string(), serde_json::json!(""));
        vars.insert("null_val".to_string(), serde_json::Value::Null);

        assert!(evaluate_condition("flag", &vars));
        assert!(!evaluate_condition("zero", &vars));
        assert!(!evaluate_condition("empty", &vars));
        assert!(!evaluate_condition("null_val", &vars));
        assert!(!evaluate_condition("false", &vars));
        assert!(evaluate_condition("unknown_literal", &vars));
    }

    #[tokio::test]
    async fn list_workflows_and_executions() {
        let executor = make_executor();
        let wf = Workflow::new("listed", super::super::WorkflowTrigger::Manual);
        let wf_id = executor.register_workflow(wf).await;

        assert_eq!(executor.list_workflows().await.len(), 1);

        executor.execute(wf_id, HashMap::new()).await.unwrap();
        assert_eq!(executor.list_executions().await.len(), 1);
    }

    #[tokio::test]
    async fn initial_vars_override_workflow_vars() {
        let executor = make_executor();
        let wf = Workflow::new("override", super::super::WorkflowTrigger::Manual)
            .with_variable("x", serde_json::json!(1));
        let wf_id = executor.register_workflow(wf).await;

        let mut initial = HashMap::new();
        initial.insert("x".to_string(), serde_json::json!(99));

        let exec = executor.execute(wf_id, initial).await.unwrap();
        assert_eq!(exec.variables.get("x"), Some(&serde_json::json!(99)));
    }
}
