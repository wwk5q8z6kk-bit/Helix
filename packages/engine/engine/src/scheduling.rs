//! Async scheduling service wrapping `hx_scheduler::SchedulerCoordinator`.
//!
//! Provides a thread-safe, async-compatible facade over the synchronous
//! scheduler crate. All mutable operations go through a `tokio::sync::Mutex`
//! so they can be called safely from axum handlers.

use std::collections::HashMap;

use tokio::sync::Mutex;

use hx_scheduler::dag::{DependencyResolver, TaskState};
use hx_scheduler::gates::QualityGatePolicy;
use hx_scheduler::retry::RetryManager;
use hx_scheduler::tracker::ExecutionTracker;
use hx_scheduler::{
    SchedulerCoordinator, WorkflowBuilder, WorkflowDefinition, WorkflowPreview, WorkflowTask,
};

/// Async-safe scheduling service.
///
/// Wraps the synchronous `SchedulerCoordinator` behind a tokio Mutex
/// so handler code can call `&mut self` methods without blocking.
pub struct SchedulingService {
    coordinator: Mutex<SchedulerCoordinator>,
    /// Stored workflow definitions keyed by name.
    workflows: Mutex<HashMap<String, WorkflowDefinition>>,
}

impl SchedulingService {
    pub fn new() -> Self {
        Self {
            coordinator: Mutex::new(SchedulerCoordinator::new(
                DependencyResolver::new(),
                ExecutionTracker::new(),
                QualityGatePolicy::new(None),
                RetryManager::new(None),
            )),
            workflows: Mutex::new(HashMap::new()),
        }
    }

    // ── Workflow definition ────────────────────────────────────────────

    /// Define a workflow using the builder API and store it.
    pub async fn define_workflow(
        &self,
        name: &str,
        tasks: &[(&str, &str, &[&str], i32)], // (task_id, task_type, deps, priority)
        deadline: Option<f64>,
        budget: Option<f64>,
    ) -> Result<WorkflowDefinition, hx_scheduler::SchedulerError> {
        let coord = self.coordinator.lock().await;
        let mut builder = WorkflowBuilder::new(name).with_coordinator(&*coord);

        for &(task_id, task_type, deps, priority) in tasks {
            builder = builder.task(task_id, task_type).priority(priority);
            if !deps.is_empty() {
                builder = builder.depends_on(deps);
            }
        }
        if let Some(d) = deadline {
            builder = builder.with_deadline(d);
        }
        if let Some(b) = budget {
            builder = builder.with_budget(b);
        }

        let definition = builder.build()?;
        drop(coord);

        let mut wfs = self.workflows.lock().await;
        wfs.insert(definition.name.clone(), definition.clone());

        tracing::info!(
            name = %definition.name,
            task_count = definition.task_order.len(),
            "workflow defined"
        );

        Ok(definition)
    }

    /// Define a workflow from pre-built WorkflowTask structs.
    pub async fn define_workflow_from_tasks(
        &self,
        name: &str,
        tasks: Vec<WorkflowTask>,
        deadline: Option<f64>,
        budget: Option<f64>,
    ) -> Result<WorkflowDefinition, hx_scheduler::SchedulerError> {
        let coord = self.coordinator.lock().await;
        let mut builder = WorkflowBuilder::new(name).with_coordinator(&*coord);

        for task in &tasks {
            builder = builder
                .task(&task.task_id, &task.task_type)
                .priority(task.priority);

            if !task.depends_on.is_empty() {
                let deps: Vec<&str> = task.depends_on.iter().map(|s| s.as_str()).collect();
                builder = builder.depends_on(&deps);
            }
            if let Some(ref m) = task.model {
                builder = builder.model(m);
            }
            if let Some(ref g) = task.gate {
                builder = builder.gate(g.clone());
            }
            if let Some(ref s) = task.retry_strategy {
                builder = builder.retry_strategy(s.clone());
            }
        }
        if let Some(d) = deadline {
            builder = builder.with_deadline(d);
        }
        if let Some(b) = budget {
            builder = builder.with_budget(b);
        }

        let definition = builder.build()?;
        drop(coord);

        let mut wfs = self.workflows.lock().await;
        wfs.insert(definition.name.clone(), definition.clone());

        tracing::info!(
            name = %definition.name,
            task_count = definition.task_order.len(),
            "workflow defined"
        );

        Ok(definition)
    }

    /// Preview a workflow without storing it.
    pub async fn preview_workflow(
        &self,
        name: &str,
        tasks: &[(&str, &str, &[&str], i32)],
        deadline: Option<f64>,
        budget: Option<f64>,
    ) -> Result<WorkflowPreview, hx_scheduler::SchedulerError> {
        let coord = self.coordinator.lock().await;
        let mut builder = WorkflowBuilder::new(name).with_coordinator(&*coord);

        for &(task_id, task_type, deps, priority) in tasks {
            builder = builder.task(task_id, task_type).priority(priority);
            if !deps.is_empty() {
                builder = builder.depends_on(deps);
            }
        }
        if let Some(d) = deadline {
            builder = builder.with_deadline(d);
        }
        if let Some(b) = budget {
            builder = builder.with_budget(b);
        }

        builder.preview()
    }

    /// Get a stored workflow definition by name.
    pub async fn get_workflow(&self, name: &str) -> Option<WorkflowDefinition> {
        self.workflows.lock().await.get(name).cloned()
    }

    /// List all stored workflow definitions.
    pub async fn list_workflows(&self) -> Vec<WorkflowDefinition> {
        self.workflows.lock().await.values().cloned().collect()
    }

    /// Remove a stored workflow.
    pub async fn remove_workflow(&self, name: &str) -> bool {
        self.workflows.lock().await.remove(name).is_some()
    }

    // ── Live DAG operations ────────────────────────────────────────────

    /// Submit a task to the live dependency graph.
    pub async fn submit_task(
        &self,
        task_id: &str,
        deps: &[&str],
        priority: i32,
    ) -> Result<TaskState, hx_scheduler::SchedulerError> {
        let mut coord = self.coordinator.lock().await;
        let state = coord.resolver.add_task(task_id, deps, priority)?;
        tracing::debug!(task_id, %state, "task submitted to DAG");
        Ok(state)
    }

    /// Mark a task as running.
    pub async fn mark_running(&self, task_id: &str) -> Result<(), hx_scheduler::SchedulerError> {
        let mut coord = self.coordinator.lock().await;
        coord.resolver.mark_running(task_id)
    }

    /// Mark a task as completed. Returns newly-ready downstream tasks.
    pub async fn mark_completed(
        &self,
        task_id: &str,
    ) -> Result<Vec<String>, hx_scheduler::SchedulerError> {
        let mut coord = self.coordinator.lock().await;
        coord.resolver.mark_completed(task_id)
    }

    /// Mark a task as failed. Returns cancelled downstream tasks.
    pub async fn mark_failed(
        &self,
        task_id: &str,
    ) -> Result<Vec<String>, hx_scheduler::SchedulerError> {
        let mut coord = self.coordinator.lock().await;
        coord.resolver.mark_failed(task_id)
    }

    /// Get all tasks currently ready for execution.
    pub async fn ready_tasks(&self) -> Vec<String> {
        self.coordinator.lock().await.resolver.get_ready_tasks()
    }

    /// Get a task's current state (active or history).
    pub async fn task_state(&self, task_id: &str) -> Option<TaskState> {
        self.coordinator
            .lock()
            .await
            .resolver
            .get_task_state(task_id)
    }

    /// Get execution waves (parallel groups in topological order).
    pub async fn execution_waves(&self) -> Vec<Vec<String>> {
        self.coordinator
            .lock()
            .await
            .resolver
            .get_execution_waves()
    }

    /// Get blocked tasks and their unsatisfied dependencies.
    pub async fn blocked_tasks(&self) -> HashMap<String, std::collections::HashSet<String>> {
        self.coordinator
            .lock()
            .await
            .resolver
            .get_blocked_tasks()
    }

    // ── Model recommendation ───────────────────────────────────────────

    /// Recommend a model for a task type.
    pub async fn recommend_model(
        &self,
        task_type: &str,
        quality_requirement: f64,
        budget_remaining: Option<f64>,
        prefer_speed: bool,
    ) -> Option<hx_scheduler::ModelRecommendation> {
        self.coordinator.lock().await.recommend_model(
            task_type,
            quality_requirement,
            budget_remaining,
            prefer_speed,
        )
    }

    // ── Stats ──────────────────────────────────────────────────────────

    /// Get scheduler statistics.
    pub async fn stats(&self) -> HashMap<String, serde_json::Value> {
        let coord = self.coordinator.lock().await;
        let mut s = coord.stats();

        let wf_count = self.workflows.lock().await.len();
        s.insert(
            "stored_workflows".into(),
            serde_json::Value::Number(wf_count.into()),
        );
        s
    }

    // ── Templates ──────────────────────────────────────────────────────

    /// List pre-built template names.
    pub fn template_names() -> Vec<&'static str> {
        vec![
            "code_review_pipeline",
            "research_pipeline",
            "parallel_analysis_pipeline",
        ]
    }

    /// Preview a pre-built template.
    pub async fn preview_template(
        &self,
        template_name: &str,
    ) -> Result<WorkflowPreview, hx_scheduler::SchedulerError> {
        let coord = self.coordinator.lock().await;
        let builder = match template_name {
            "code_review_pipeline" => {
                hx_scheduler::code_review_pipeline().with_coordinator(&*coord)
            }
            "research_pipeline" => {
                hx_scheduler::research_pipeline().with_coordinator(&*coord)
            }
            "parallel_analysis_pipeline" => {
                hx_scheduler::parallel_analysis_pipeline().with_coordinator(&*coord)
            }
            _ => {
                return Err(hx_scheduler::SchedulerError::InvalidInput(format!(
                    "unknown template: {template_name}"
                )));
            }
        };
        builder.preview()
    }
}

impl Default for SchedulingService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn service_define_and_list() {
        let svc = SchedulingService::new();

        let tasks = [
            ("a", "planning", [].as_slice(), 1),
            ("b", "coding", ["a"].as_slice(), 2),
        ];
        let def = svc
            .define_workflow("test-wf", &tasks, None, None)
            .await
            .unwrap();
        assert_eq!(def.task_order, vec!["a", "b"]);

        let all = svc.list_workflows().await;
        assert_eq!(all.len(), 1);
    }

    #[tokio::test]
    async fn service_submit_and_complete() {
        let svc = SchedulingService::new();
        svc.submit_task("x", &[], 0).await.unwrap();
        svc.submit_task("y", &["x"], 1).await.unwrap();

        let ready = svc.ready_tasks().await;
        assert_eq!(ready, vec!["x"]);

        svc.mark_running("x").await.unwrap();
        let newly_ready = svc.mark_completed("x").await.unwrap();
        assert_eq!(newly_ready, vec!["y"]);
    }

    #[tokio::test]
    async fn service_failure_propagation() {
        let svc = SchedulingService::new();
        svc.submit_task("a", &[], 0).await.unwrap();
        svc.submit_task("b", &["a"], 1).await.unwrap();
        svc.submit_task("c", &["b"], 2).await.unwrap();

        svc.mark_running("a").await.unwrap();
        let cancelled = svc.mark_failed("a").await.unwrap();
        assert!(cancelled.contains(&"b".to_string()));
        assert!(cancelled.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn service_template_preview() {
        let svc = SchedulingService::new();
        let preview = svc
            .preview_template("code_review_pipeline")
            .await
            .unwrap();
        assert!(!preview.waves.is_empty());
    }

    #[tokio::test]
    async fn service_template_unknown() {
        let svc = SchedulingService::new();
        let result = svc.preview_template("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn service_remove_workflow() {
        let svc = SchedulingService::new();
        let tasks = [("solo", "planning", [].as_slice(), 0)];
        svc.define_workflow("remove-me", &tasks, None, None)
            .await
            .unwrap();
        assert!(svc.remove_workflow("remove-me").await);
        assert!(svc.list_workflows().await.is_empty());
    }

    #[tokio::test]
    async fn service_stats() {
        let svc = SchedulingService::new();
        svc.submit_task("t1", &[], 0).await.unwrap();
        let stats = svc.stats().await;
        assert!(stats.contains_key("active_tasks"));
        assert!(stats.contains_key("stored_workflows"));
    }
}
