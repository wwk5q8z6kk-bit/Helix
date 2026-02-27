//! Declarative workflow builder with predictive preview.
//!
//! Fluent API for defining task DAGs with quality gates, retry strategies,
//! model preferences, deadlines, and budget constraints.
//!
//! The unique feature: **predictive preview** — before executing, the builder
//! uses execution history to predict duration, cost, quality, and constraint
//! feasibility.
//!
//! Port of `python/core/scheduling/workflow_builder.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::coordinator::SchedulerCoordinator;
use crate::dag::DependencyResolver;
use crate::error::{Result, SchedulerError};
use crate::gates::{QualityGate, GATE_LENIENT, GATE_STANDARD, GATE_STRICT};
use crate::retry::{
    strategy_budget_conscious, strategy_fast_to_premium, strategy_no_retry,
    strategy_quality_first, RetryStrategy,
};

// ── Value objects ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowStatus {
    Draft,
    Validated,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowTask {
    pub task_id: String,
    pub task_type: String,
    pub depends_on: Vec<String>,
    pub model: Option<String>,
    pub priority: i32,
    pub gate: Option<QualityGate>,
    pub retry_strategy: Option<RetryStrategy>,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Immutable, validated workflow definition ready for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowDefinition {
    pub name: String,
    pub tasks: HashMap<String, WorkflowTask>,
    pub task_order: Vec<String>,
    pub waves: Vec<Vec<String>>,
    pub deadline: Option<f64>,
    pub budget: Option<f64>,
    pub default_gate: Option<QualityGate>,
    pub default_strategy: Option<RetryStrategy>,
    pub model_assignments: HashMap<String, String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

impl WorkflowDefinition {
    pub fn task_count(&self) -> usize {
        self.tasks.len()
    }

    pub fn wave_count(&self) -> usize {
        self.waves.len()
    }

    pub fn max_parallelism(&self) -> usize {
        self.waves.iter().map(|w| w.len()).max().unwrap_or(0)
    }
}

/// Predicted execution metrics for a single task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPrediction {
    pub task_id: String,
    pub model: String,
    pub predicted_duration: f64,
    pub predicted_cost: f64,
    pub predicted_quality: f64,
    pub predicted_success_rate: f64,
    pub meets_quality_gate: bool,
    pub quality_margin: f64,
}

/// Predictive analysis of a workflow before execution.
#[derive(Debug, Clone)]
pub struct WorkflowPreview {
    pub name: String,
    pub task_predictions: Vec<TaskPrediction>,
    pub waves: Vec<Vec<String>>,
    pub wave_durations: Vec<f64>,
    pub critical_path: Vec<String>,

    pub total_predicted_duration: f64,
    pub total_predicted_cost: f64,
    pub avg_predicted_quality: f64,

    pub deadline: Option<f64>,
    pub budget: Option<f64>,
    pub meets_deadline: Option<bool>,
    pub meets_budget: Option<bool>,
    pub deadline_margin: Option<f64>,
    pub budget_margin: Option<f64>,

    pub at_risk_tasks: Vec<String>,
    pub bottleneck_tasks: Vec<String>,
    pub low_confidence_tasks: Vec<String>,
}

impl WorkflowPreview {
    /// Whether the workflow can execute within time and budget constraints.
    pub fn feasible(&self) -> bool {
        if self.meets_deadline == Some(false) {
            return false;
        }
        if self.meets_budget == Some(false) {
            return false;
        }
        true
    }

    /// Overall risk: low, medium, high.
    pub fn risk_level(&self) -> &'static str {
        let mut risks = 0usize;
        if self.meets_deadline == Some(false) {
            risks += 2;
        }
        if self.meets_budget == Some(false) {
            risks += 1;
        }
        risks += self.at_risk_tasks.len();
        risks += self.low_confidence_tasks.len();
        if risks == 0 {
            "low"
        } else if risks <= 2 {
            "medium"
        } else {
            "high"
        }
    }
}

// ── Builder ─────────────────────────────────────────────────────────

/// Fluent API for building task DAGs with scheduling intelligence.
pub struct WorkflowBuilder<'a> {
    name: String,
    coordinator: Option<&'a SchedulerCoordinator>,
    tasks: HashMap<String, WorkflowTask>,
    insertion_order: Vec<String>,
    deadline: Option<f64>,
    budget: Option<f64>,
    default_gate: Option<QualityGate>,
    default_strategy: Option<RetryStrategy>,
    metadata: HashMap<String, serde_json::Value>,
    edge_gates: HashMap<(String, String), QualityGate>,
}

impl<'a> WorkflowBuilder<'a> {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            coordinator: None,
            tasks: HashMap::new(),
            insertion_order: Vec::new(),
            deadline: None,
            budget: None,
            default_gate: None,
            default_strategy: None,
            metadata: HashMap::new(),
            edge_gates: HashMap::new(),
        }
    }

    pub fn with_coordinator(mut self, coord: &'a SchedulerCoordinator) -> Self {
        self.coordinator = Some(coord);
        self
    }

    // ── Fluent task definition ──────────────────────────────────────

    pub fn task(mut self, task_id: &str, task_type: &str) -> Self {
        assert!(
            !self.tasks.contains_key(task_id),
            "Duplicate task_id: {task_id}"
        );
        self.tasks.insert(
            task_id.to_string(),
            WorkflowTask {
                task_id: task_id.to_string(),
                task_type: task_type.to_string(),
                depends_on: Vec::new(),
                model: None,
                priority: 2,
                gate: None,
                retry_strategy: None,
                metadata: HashMap::new(),
            },
        );
        self.insertion_order.push(task_id.to_string());
        self
    }

    /// Modify the most recently added task (builder-pattern helper).
    fn last_mut(&mut self) -> &mut WorkflowTask {
        let id = self.insertion_order.last().expect("no tasks added yet");
        self.tasks.get_mut(id).unwrap()
    }

    pub fn depends_on(mut self, deps: &[&str]) -> Self {
        self.last_mut().depends_on = deps.iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn model(mut self, model: &str) -> Self {
        self.last_mut().model = Some(model.to_string());
        self
    }

    pub fn priority(mut self, p: i32) -> Self {
        self.last_mut().priority = p;
        self
    }

    pub fn gate(mut self, gate: QualityGate) -> Self {
        self.last_mut().gate = Some(gate);
        self
    }

    pub fn retry_strategy(mut self, strategy: RetryStrategy) -> Self {
        self.last_mut().retry_strategy = Some(strategy);
        self
    }

    pub fn edge_gate(mut self, upstream: &str, downstream: &str, gate: QualityGate) -> Self {
        self.edge_gates
            .insert((upstream.to_string(), downstream.to_string()), gate);
        self
    }

    // ── Fluent constraint definition ────────────────────────────────

    pub fn with_deadline(mut self, deadline: f64) -> Self {
        self.deadline = Some(deadline);
        self
    }

    pub fn with_deadline_in(mut self, seconds: f64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        self.deadline = Some(now + seconds);
        self
    }

    pub fn with_budget(mut self, budget: f64) -> Self {
        self.budget = Some(budget);
        self
    }

    pub fn with_default_gate(mut self, gate: QualityGate) -> Self {
        self.default_gate = Some(gate);
        self
    }

    pub fn with_default_strategy(mut self, strategy: RetryStrategy) -> Self {
        self.default_strategy = Some(strategy);
        self
    }

    pub fn with_metadata(mut self, key: &str, value: serde_json::Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    // ── Topological sort ────────────────────────────────────────────

    /// Kahn's algorithm over the builder's tasks. Produces topological
    /// order and parallel execution waves.
    fn compute_topo_order(&self) -> (Vec<String>, Vec<Vec<String>>) {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents_map: HashMap<&str, Vec<&str>> = HashMap::new();

        for (tid, task) in &self.tasks {
            in_degree.entry(tid.as_str()).or_insert(0);
            dependents_map.entry(tid.as_str()).or_default();
            for dep in &task.depends_on {
                if self.tasks.contains_key(dep) {
                    *in_degree.entry(tid.as_str()).or_insert(0) += 1;
                    dependents_map
                        .entry(dep.as_str())
                        .or_default()
                        .push(tid.as_str());
                }
            }
        }

        let mut waves: Vec<Vec<String>> = Vec::new();
        let mut order: Vec<String> = Vec::new();

        let mut current: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        self.sort_tasks(&mut current);

        while !current.is_empty() {
            let wave: Vec<String> = current.iter().map(|s| s.to_string()).collect();
            order.extend(wave.iter().cloned());
            waves.push(wave);

            let mut next: Vec<&str> = Vec::new();
            for &tid in &current {
                if let Some(deps) = dependents_map.get(tid) {
                    for &d in deps {
                        if let Some(deg) = in_degree.get_mut(d) {
                            *deg = deg.saturating_sub(1);
                            if *deg == 0 {
                                next.push(d);
                            }
                        }
                    }
                }
            }
            self.sort_tasks(&mut next);
            current = next;
        }

        (order, waves)
    }

    fn sort_tasks(&self, tasks: &mut Vec<&str>) {
        tasks.sort_by(|a, b| {
            let pa = self.tasks.get(*a).map(|t| t.priority).unwrap_or(2);
            let pb = self.tasks.get(*b).map(|t| t.priority).unwrap_or(2);
            pa.cmp(&pb).then_with(|| {
                let ia = self.insertion_order.iter().position(|x| x == *a);
                let ib = self.insertion_order.iter().position(|x| x == *b);
                ia.cmp(&ib)
            })
        });
    }

    // ── Validation ──────────────────────────────────────────────────

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.tasks.is_empty() {
            errors.push("Workflow has no tasks".into());
            return errors;
        }

        // Check dependency references.
        for task in self.tasks.values() {
            for dep in &task.depends_on {
                if !self.tasks.contains_key(dep) {
                    errors.push(format!(
                        "Task '{}' depends on unknown task '{dep}'",
                        task.task_id
                    ));
                }
            }
        }

        // Check for cycles using a temporary resolver.
        if errors.is_empty() {
            let mut resolver = DependencyResolver::new();
            for tid in &self.insertion_order {
                let task = &self.tasks[tid];
                let deps: Vec<&str> = task.depends_on.iter().map(|s| s.as_str()).collect();
                if let Err(SchedulerError::CycleDetected(cycle)) =
                    resolver.add_task(tid, &deps, task.priority)
                {
                    errors.push(format!("Cycle detected: {}", cycle.join(" -> ")));
                }
            }
        }

        if let Some(budget) = self.budget {
            if budget <= 0.0 {
                errors.push(format!("Budget must be positive, got {budget}"));
            }
        }

        if let Some(deadline) = self.deadline {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            if deadline <= now {
                errors.push("Deadline is in the past".into());
            }
        }

        errors
    }

    // ── Preview ─────────────────────────────────────────────────────

    /// Generate predictive analysis without executing.
    pub fn preview(&self) -> Result<WorkflowPreview> {
        let errors = self.validate();
        if !errors.is_empty() {
            return Err(SchedulerError::ValidationFailed(errors));
        }

        let (topo_order, waves) = self.compute_topo_order();

        // Model assignments.
        let mut model_assignments: HashMap<String, String> = HashMap::new();
        for (tid, task) in &self.tasks {
            if let Some(ref m) = task.model {
                model_assignments.insert(tid.clone(), m.clone());
            } else if let Some(coord) = self.coordinator {
                let gate = task
                    .gate
                    .as_ref()
                    .or(self.default_gate.as_ref())
                    .unwrap_or(&GATE_STANDARD);
                let rec = coord.recommend_model(
                    &task.task_type,
                    gate.min_quality,
                    self.budget,
                    false,
                );
                model_assignments.insert(
                    tid.clone(),
                    rec.map(|r| r.model).unwrap_or_else(|| "default".into()),
                );
            } else {
                model_assignments.insert(tid.clone(), "default".into());
            }
        }

        // Per-task predictions.
        let mut predictions = Vec::new();
        let mut at_risk = Vec::new();
        let mut low_confidence = Vec::new();

        for (tid, task) in &self.tasks {
            let model = model_assignments.get(tid).map(|s| s.as_str()).unwrap_or("default");
            let gate = task
                .gate
                .as_ref()
                .or(self.default_gate.as_ref())
                .unwrap_or(&GATE_STANDARD);

            let (dur, cost, quality, success) = if let Some(coord) = self.coordinator {
                (
                    coord.tracker.predict_duration(&task.task_type, model),
                    coord.tracker.predict_cost(&task.task_type, model),
                    coord.tracker.predict_quality(&task.task_type, model),
                    coord.tracker.predict_success_rate(&task.task_type, model),
                )
            } else {
                (10.0, 0.01, 0.5, 0.9)
            };

            let meets_gate = quality >= gate.min_quality;
            let margin = quality - gate.min_quality;

            predictions.push(TaskPrediction {
                task_id: tid.clone(),
                model: model.to_string(),
                predicted_duration: dur,
                predicted_cost: cost,
                predicted_quality: quality,
                predicted_success_rate: success,
                meets_quality_gate: meets_gate,
                quality_margin: margin,
            });

            if !meets_gate {
                at_risk.push(tid.clone());
            }
            if success < 0.7 {
                low_confidence.push(tid.clone());
            }
        }

        // Wave durations.
        let task_durations: HashMap<String, f64> = predictions
            .iter()
            .map(|p| (p.task_id.clone(), p.predicted_duration))
            .collect();

        let wave_durations: Vec<f64> = waves
            .iter()
            .map(|wave| {
                wave.iter()
                    .map(|tid| task_durations.get(tid).copied().unwrap_or(10.0))
                    .fold(0.0f64, f64::max)
            })
            .collect();

        let total_duration: f64 = wave_durations.iter().sum();
        let total_cost: f64 = predictions.iter().map(|p| p.predicted_cost).sum();
        let avg_quality = if predictions.is_empty() {
            0.0
        } else {
            predictions.iter().map(|p| p.predicted_quality).sum::<f64>() / predictions.len() as f64
        };

        // Critical path.
        let mut critical_path = Vec::new();
        let mut bottlenecks = Vec::new();

        if let Some(coord) = self.coordinator {
            let task_types: HashMap<String, String> = self
                .tasks
                .iter()
                .map(|(tid, t)| (tid.clone(), t.task_type.clone()))
                .collect();

            // Build temporary resolver for critical-path analysis.
            let mut cp_resolver = DependencyResolver::new();
            for tid in &topo_order {
                let task = &self.tasks[tid];
                let deps: Vec<&str> = task.depends_on.iter().map(|s| s.as_str()).collect();
                let _ = cp_resolver.add_task(tid, &deps, task.priority);
            }

            let cp = coord
                .tracker
                .compute_critical_path(&cp_resolver, &task_types, &model_assignments);
            critical_path = cp.path;

            let cp_set: std::collections::HashSet<&str> =
                critical_path.iter().map(|s| s.as_str()).collect();
            for wave in &waves {
                if wave.len() == 1 && cp_set.contains(wave[0].as_str()) {
                    bottlenecks.push(wave[0].clone());
                }
            }
        } else {
            for wave in &waves {
                critical_path.extend(wave.iter().cloned());
            }
        }

        // Constraint analysis.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        let (meets_deadline, deadline_margin) = if let Some(dl) = self.deadline {
            let remaining = dl - now;
            let margin = remaining - total_duration;
            (Some(margin >= 0.0), Some(margin))
        } else {
            (None, None)
        };

        let (meets_budget, budget_margin) = if let Some(bgt) = self.budget {
            let margin = bgt - total_cost;
            (Some(margin >= 0.0), Some(margin))
        } else {
            (None, None)
        };

        Ok(WorkflowPreview {
            name: self.name.clone(),
            task_predictions: predictions,
            waves,
            wave_durations,
            critical_path,
            total_predicted_duration: total_duration,
            total_predicted_cost: total_cost,
            avg_predicted_quality: avg_quality,
            deadline: self.deadline,
            budget: self.budget,
            meets_deadline,
            meets_budget,
            deadline_margin,
            budget_margin,
            at_risk_tasks: at_risk,
            bottleneck_tasks: bottlenecks,
            low_confidence_tasks: low_confidence,
        })
    }

    // ── Build ───────────────────────────────────────────────────────

    /// Validate and produce an immutable workflow definition.
    pub fn build(self) -> Result<WorkflowDefinition> {
        let errors = self.validate();
        if !errors.is_empty() {
            return Err(SchedulerError::ValidationFailed(errors));
        }

        let (topo_order, waves) = self.compute_topo_order();

        // Model assignments.
        let mut model_assignments: HashMap<String, String> = HashMap::new();
        for (tid, task) in &self.tasks {
            if let Some(ref m) = task.model {
                model_assignments.insert(tid.clone(), m.clone());
            } else if let Some(coord) = self.coordinator {
                let gate = task
                    .gate
                    .as_ref()
                    .or(self.default_gate.as_ref())
                    .unwrap_or(&GATE_STANDARD);
                let rec = coord.recommend_model(
                    &task.task_type,
                    gate.min_quality,
                    self.budget,
                    false,
                );
                model_assignments.insert(
                    tid.clone(),
                    rec.map(|r| r.model).unwrap_or_else(|| "default".into()),
                );
            } else {
                model_assignments.insert(tid.clone(), "default".into());
            }
        }

        Ok(WorkflowDefinition {
            name: self.name,
            tasks: self.tasks,
            task_order: topo_order,
            waves,
            deadline: self.deadline,
            budget: self.budget,
            default_gate: self.default_gate,
            default_strategy: self.default_strategy,
            model_assignments,
            metadata: self.metadata,
        })
    }
}

// ── Pre-built templates ─────────────────────────────────────────────

pub fn code_review_pipeline<'a>() -> WorkflowBuilder<'a> {
    WorkflowBuilder::new("code_review")
        .task("plan", "planning")
        .priority(1)
        .task("implement", "coding")
        .depends_on(&["plan"])
        .priority(0)
        .gate(GATE_STRICT.clone())
        .retry_strategy(strategy_fast_to_premium())
        .task("test", "testing")
        .depends_on(&["implement"])
        .gate(GATE_LENIENT.clone())
        .retry_strategy(strategy_no_retry())
        .task("review", "review")
        .depends_on(&["test"])
        .gate(GATE_STRICT.clone())
        .retry_strategy(strategy_quality_first())
        .with_default_gate(GATE_STANDARD.clone())
}

pub fn research_pipeline<'a>() -> WorkflowBuilder<'a> {
    WorkflowBuilder::new("research")
        .task("gather", "research")
        .priority(1)
        .task("analyze", "reasoning")
        .depends_on(&["gather"])
        .priority(1)
        .task("synthesize", "creative")
        .depends_on(&["analyze"])
        .task("report", "writing")
        .depends_on(&["synthesize"])
        .with_default_gate(GATE_STANDARD.clone())
        .with_default_strategy(strategy_budget_conscious())
}

pub fn parallel_analysis_pipeline<'a>() -> WorkflowBuilder<'a> {
    WorkflowBuilder::new("parallel_analysis")
        .task("prepare", "planning")
        .priority(0)
        .task("analyze_security", "review")
        .depends_on(&["prepare"])
        .task("analyze_performance", "review")
        .depends_on(&["prepare"])
        .task("analyze_quality", "review")
        .depends_on(&["prepare"])
        .task("merge_results", "reasoning")
        .depends_on(&["analyze_security", "analyze_performance", "analyze_quality"])
        .gate(GATE_STRICT.clone())
        .with_default_gate(GATE_STANDARD.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_single_task() {
        let defn = WorkflowBuilder::new("test")
            .task("t1", "coding")
            .build()
            .unwrap();
        assert_eq!(defn.task_count(), 1);
        assert!(defn.tasks.contains_key("t1"));
    }

    #[test]
    fn builder_fluent_chain() {
        let defn = WorkflowBuilder::new("test")
            .task("a", "planning")
            .task("b", "coding")
            .depends_on(&["a"])
            .task("c", "testing")
            .depends_on(&["b"])
            .build()
            .unwrap();
        assert_eq!(defn.task_count(), 3);
    }

    #[test]
    #[should_panic(expected = "Duplicate")]
    fn duplicate_task_panics() {
        WorkflowBuilder::new("test")
            .task("t1", "x")
            .task("t1", "y");
    }

    #[test]
    fn topological_order() {
        let defn = WorkflowBuilder::new("test")
            .task("c", "x")
            .depends_on(&["b"])
            .task("a", "x")
            .task("b", "x")
            .depends_on(&["a"])
            .build()
            .unwrap();

        let ia = defn.task_order.iter().position(|x| x == "a").unwrap();
        let ib = defn.task_order.iter().position(|x| x == "b").unwrap();
        let ic = defn.task_order.iter().position(|x| x == "c").unwrap();
        assert!(ia < ib);
        assert!(ib < ic);
    }

    #[test]
    fn waves_parallel() {
        let defn = WorkflowBuilder::new("test")
            .task("a", "x")
            .task("b", "x")
            .task("c", "x")
            .build()
            .unwrap();
        assert_eq!(defn.wave_count(), 1);
        assert_eq!(defn.max_parallelism(), 3);
    }

    #[test]
    fn waves_diamond() {
        let defn = WorkflowBuilder::new("test")
            .task("a", "x")
            .task("b", "x")
            .depends_on(&["a"])
            .task("c", "x")
            .depends_on(&["a"])
            .task("d", "x")
            .depends_on(&["b", "c"])
            .build()
            .unwrap();
        assert_eq!(defn.wave_count(), 3);
        assert_eq!(defn.max_parallelism(), 2);
    }

    #[test]
    fn validation_empty() {
        let errors = WorkflowBuilder::new("test").validate();
        assert!(errors.iter().any(|e| e.contains("no tasks")));
    }

    #[test]
    fn validation_missing_dep() {
        let errors = WorkflowBuilder::new("test")
            .task("b", "x")
            .depends_on(&["a"])
            .validate();
        assert!(errors.iter().any(|e| e.contains("unknown task")));
    }

    #[test]
    fn preview_basic() {
        let preview = WorkflowBuilder::new("test")
            .task("a", "x")
            .task("b", "y")
            .depends_on(&["a"])
            .preview()
            .unwrap();
        assert_eq!(preview.task_predictions.len(), 2);
        assert!(preview.total_predicted_duration > 0.0);
        assert!(preview.total_predicted_cost > 0.0);
    }

    #[test]
    fn preview_feasible_no_constraints() {
        let preview = WorkflowBuilder::new("test")
            .task("t1", "x")
            .preview()
            .unwrap();
        assert!(preview.feasible());
        assert_eq!(preview.risk_level(), "low");
    }

    #[test]
    fn preview_budget_exceeded() {
        let preview = WorkflowBuilder::new("test")
            .task("t1", "x")
            .task("t2", "x")
            .task("t3", "x")
            .with_budget(0.001)
            .preview()
            .unwrap();
        assert_eq!(preview.meets_budget, Some(false));
        assert!(!preview.feasible());
    }

    #[test]
    fn code_review_template() {
        let defn = code_review_pipeline().build().unwrap();
        assert_eq!(defn.name, "code_review");
        assert_eq!(defn.task_count(), 4);
        assert_eq!(defn.wave_count(), 4); // all sequential
        assert!(defn.tasks.contains_key("plan"));
        assert!(defn.tasks.contains_key("implement"));
        assert!(defn.tasks.contains_key("test"));
        assert!(defn.tasks.contains_key("review"));
    }

    #[test]
    fn code_review_ordering() {
        let defn = code_review_pipeline().build().unwrap();
        let o = &defn.task_order;
        assert!(o.iter().position(|x| x == "plan").unwrap() < o.iter().position(|x| x == "implement").unwrap());
        assert!(o.iter().position(|x| x == "implement").unwrap() < o.iter().position(|x| x == "test").unwrap());
        assert!(o.iter().position(|x| x == "test").unwrap() < o.iter().position(|x| x == "review").unwrap());
    }

    #[test]
    fn parallel_analysis_template() {
        let defn = parallel_analysis_pipeline().build().unwrap();
        assert_eq!(defn.task_count(), 5);
        assert_eq!(defn.max_parallelism(), 3);
        assert_eq!(defn.wave_count(), 3);
    }

    #[test]
    fn template_preview_feasible() {
        let preview = code_review_pipeline().preview().unwrap();
        assert_eq!(preview.name, "code_review");
        assert_eq!(preview.task_predictions.len(), 4);
        assert!(preview.feasible()); // no deadline/budget constraints
    }

    #[test]
    fn template_customization() {
        let defn = code_review_pipeline()
            .with_budget(5.0)
            .with_deadline_in(7200.0)
            .build()
            .unwrap();
        assert!((defn.budget.unwrap() - 5.0).abs() < 0.01);
        assert!(defn.deadline.is_some());
    }
}
