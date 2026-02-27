//! Unified scheduling intelligence: Pareto model selection, concurrency control,
//! deadline management, execution planning, and self-tuning.
//!
//! Port of `python/core/scheduling/scheduler_coordinator.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dag::DependencyResolver;
use crate::gates::QualityGatePolicy;
use crate::retry::RetryManager;
use crate::tracker::ExecutionTracker;

// ── Value objects ───────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelRecommendation {
    pub model: String,
    pub predicted_quality: f64,
    pub predicted_cost: f64,
    pub predicted_duration: f64,
    pub efficiency_score: f64,
    pub success_rate: f64,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleEntry {
    pub task_id: String,
    pub wave: usize,
    pub model: String,
    pub predicted_duration: f64,
    pub predicted_cost: f64,
    pub predicted_quality: f64,
    pub priority_boost: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    pub entries: Vec<ScheduleEntry>,
    pub waves: Vec<Vec<String>>,
    pub wave_durations: Vec<f64>,
    pub total_predicted_duration: f64,
    pub total_predicted_cost: f64,
    pub critical_path: Vec<String>,
    pub bottleneck_tasks: Vec<String>,
    pub model_assignments: HashMap<String, String>,
}

// ── Concurrency slot ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencySlot {
    pub provider: String,
    pub max_concurrent: u32,
    active: u32,
    total_acquired: u64,
    total_rejected: u64,
}

impl ConcurrencySlot {
    pub fn new(provider: &str, max_concurrent: u32) -> Self {
        Self {
            provider: provider.to_string(),
            max_concurrent,
            active: 0,
            total_acquired: 0,
            total_rejected: 0,
        }
    }

    pub fn acquire(&mut self) -> bool {
        if self.active < self.max_concurrent {
            self.active += 1;
            self.total_acquired += 1;
            true
        } else {
            self.total_rejected += 1;
            false
        }
    }

    pub fn release(&mut self) {
        self.active = self.active.saturating_sub(1);
    }

    pub fn available(&self) -> u32 {
        self.max_concurrent - self.active
    }

    pub fn utilization(&self) -> f64 {
        if self.max_concurrent == 0 {
            0.0
        } else {
            self.active as f64 / self.max_concurrent as f64
        }
    }
}

// ── Coordinator ─────────────────────────────────────────────────────

/// Unified scheduling intelligence integrating all four subsystems.
pub struct SchedulerCoordinator {
    pub resolver: DependencyResolver,
    pub tracker: ExecutionTracker,
    pub policy: QualityGatePolicy,
    pub retry_manager: RetryManager,
    slots: HashMap<String, ConcurrencySlot>,
    default_max_concurrent: u32,
    deadlines: HashMap<String, f64>,
    tuning_log: Vec<HashMap<String, String>>,
    max_tuning_log: usize,
}

impl SchedulerCoordinator {
    pub fn new(
        resolver: DependencyResolver,
        tracker: ExecutionTracker,
        policy: QualityGatePolicy,
        retry_manager: RetryManager,
    ) -> Self {
        Self {
            resolver,
            tracker,
            policy,
            retry_manager,
            slots: HashMap::new(),
            default_max_concurrent: 5,
            deadlines: HashMap::new(),
            tuning_log: Vec::new(),
            max_tuning_log: 50,
        }
    }

    // ── Pareto model selection ──────────────────────────────────────

    /// Compute the Pareto frontier of models for a task type.
    ///
    /// A model is on the frontier if no other model has BOTH better
    /// quality AND lower cost.
    pub fn compute_pareto_frontier(
        &self,
        task_type: &str,
        min_executions: u64,
    ) -> Vec<ModelRecommendation> {
        // Collect candidates from profiles.
        let candidates: Vec<ModelRecommendation> = self
            .tracker
            .get_all_profiles()
            .values()
            .filter(|p| {
                p.task_type == task_type
                    && p.total_executions >= min_executions
                    && p.successes >= 1
            })
            .map(|p| ModelRecommendation {
                model: p.model.clone(),
                predicted_quality: p.avg_quality,
                predicted_cost: p.avg_cost,
                predicted_duration: p.avg_duration,
                efficiency_score: p.avg_quality / p.avg_cost.max(0.001),
                success_rate: p.success_rate(),
                reason: "historical".into(),
            })
            .collect();

        // Pareto filter: remove dominated models.
        let mut frontier = Vec::new();
        for (i, a) in candidates.iter().enumerate() {
            let dominated = candidates.iter().enumerate().any(|(j, b)| {
                i != j
                    && b.predicted_quality >= a.predicted_quality
                    && b.predicted_cost <= a.predicted_cost
                    && (b.predicted_quality > a.predicted_quality
                        || b.predicted_cost < a.predicted_cost)
            });
            if !dominated {
                frontier.push(a.clone());
            }
        }

        frontier.sort_by(|a, b| {
            b.efficiency_score
                .partial_cmp(&a.efficiency_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        frontier
    }

    /// Recommend a model based on Pareto frontier and constraints.
    pub fn recommend_model(
        &self,
        task_type: &str,
        quality_requirement: f64,
        budget_remaining: Option<f64>,
        prefer_speed: bool,
    ) -> Option<ModelRecommendation> {
        let mut frontier = self.compute_pareto_frontier(task_type, 2);

        frontier.retain(|m| m.predicted_quality >= quality_requirement);

        if let Some(budget) = budget_remaining {
            frontier.retain(|m| m.predicted_cost <= budget);
        }

        if prefer_speed {
            frontier.sort_by(|a, b| {
                a.predicted_duration
                    .partial_cmp(&b.predicted_duration)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        frontier.into_iter().next()
    }

    // ── Concurrency ─────────────────────────────────────────────────

    pub fn set_provider_limit(&mut self, provider: &str, max_concurrent: u32) {
        self.slots
            .insert(provider.to_string(), ConcurrencySlot::new(provider, max_concurrent));
    }

    pub fn acquire_slot(&mut self, provider: &str) -> bool {
        self.slots
            .entry(provider.to_string())
            .or_insert_with(|| ConcurrencySlot::new(provider, self.default_max_concurrent))
            .acquire()
    }

    pub fn release_slot(&mut self, provider: &str) {
        if let Some(slot) = self.slots.get_mut(provider) {
            slot.release();
        }
    }

    // ── Deadlines ───────────────────────────────────────────────────

    pub fn set_deadline(&mut self, task_id: &str, deadline: f64) {
        self.deadlines.insert(task_id.to_string(), deadline);
    }

    pub fn clear_deadline(&mut self, task_id: &str) {
        self.deadlines.remove(task_id);
    }

    /// Urgency score 0.0–1.0 based on deadline proximity.
    pub fn get_urgency(&self, task_id: &str) -> f64 {
        let deadline = match self.deadlines.get(task_id) {
            Some(&d) => d,
            None => return 0.0,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let remaining = deadline - now;

        if remaining <= 0.0 {
            1.0
        } else if remaining <= 3600.0 {
            // Linear ramp in final hour: 0.5 → 1.0
            0.5 + 0.5 * (1.0 - remaining / 3600.0)
        } else if remaining <= 86400.0 {
            // 1h–24h: 0.1 → 0.5
            0.1 + 0.4 * (86400.0 - remaining) / (86400.0 - 3600.0)
        } else {
            // >24h: logarithmic decay toward 0.0
            let days = remaining / 86400.0;
            0.1 / days
        }
    }

    /// Convert urgency to integer priority boost (0–3).
    pub fn get_priority_boost(&self, task_id: &str) -> i32 {
        let u = self.get_urgency(task_id);
        if u >= 0.8 {
            3
        } else if u >= 0.5 {
            2
        } else if u >= 0.2 {
            1
        } else {
            0
        }
    }

    // ── Execution planning ──────────────────────────────────────────

    /// Build a complete execution plan with model assignments and predictions.
    pub fn plan_execution(
        &self,
        task_types: &HashMap<String, String>,
        task_models: Option<&HashMap<String, String>>,
        budget_limit: Option<f64>,
    ) -> ExecutionPlan {
        let waves = self.resolver.get_execution_waves();

        // Auto-assign models where not provided.
        let mut model_assignments: HashMap<String, String> = HashMap::new();
        for (tid, tt) in task_types {
            if let Some(models) = task_models {
                if let Some(m) = models.get(tid) {
                    model_assignments.insert(tid.clone(), m.clone());
                    continue;
                }
            }
            let gate = self.policy.get_gate(tid, None);
            let rec = self.recommend_model(
                tt,
                gate.min_quality,
                budget_limit,
                false,
            );
            model_assignments.insert(
                tid.clone(),
                rec.map(|r| r.model).unwrap_or_else(|| "default".into()),
            );
        }

        // Build schedule entries.
        let mut entries = Vec::new();
        let mut wave_durations = Vec::new();

        for (wi, wave) in waves.iter().enumerate() {
            let mut max_dur: f64 = 0.0;
            for tid in wave {
                let tt = task_types.get(tid).map(|s| s.as_str()).unwrap_or("general");
                let model = model_assignments.get(tid).map(|s| s.as_str()).unwrap_or("default");
                let dur = self.tracker.predict_duration(tt, model);
                let cost = self.tracker.predict_cost(tt, model);
                let quality = self.tracker.predict_quality(tt, model);
                let boost = self.get_priority_boost(tid) as f64;

                entries.push(ScheduleEntry {
                    task_id: tid.clone(),
                    wave: wi,
                    model: model.to_string(),
                    predicted_duration: dur,
                    predicted_cost: cost,
                    predicted_quality: quality,
                    priority_boost: boost,
                });

                if dur > max_dur {
                    max_dur = dur;
                }
            }
            wave_durations.push(max_dur);
        }

        let total_duration: f64 = wave_durations.iter().sum();
        let total_cost: f64 = entries.iter().map(|e| e.predicted_cost).sum();

        // Critical path.
        let cp = self
            .tracker
            .compute_critical_path(&self.resolver, task_types, &model_assignments);

        // Bottlenecks: single-task waves on the critical path.
        let cp_set: std::collections::HashSet<&str> =
            cp.path.iter().map(|s| s.as_str()).collect();
        let bottlenecks: Vec<String> = waves
            .iter()
            .filter(|w| w.len() == 1 && cp_set.contains(w[0].as_str()))
            .map(|w| w[0].clone())
            .collect();

        ExecutionPlan {
            entries,
            waves,
            wave_durations,
            total_predicted_duration: total_duration,
            total_predicted_cost: total_cost,
            critical_path: cp.path,
            bottleneck_tasks: bottlenecks,
            model_assignments,
        }
    }

    // ── Self-tuning ─────────────────────────────────────────────────

    /// Analyze profiles and suggest quality gate / retry adjustments.
    pub fn auto_tune(&mut self) -> HashMap<String, Vec<String>> {
        let mut recommendations: HashMap<String, Vec<String>> = HashMap::new();

        // Group profiles by task_type.
        let mut by_type: HashMap<String, (f64, f64, u64)> = HashMap::new();
        for p in self.tracker.get_all_profiles().values() {
            let entry = by_type
                .entry(p.task_type.clone())
                .or_insert((0.0, 0.0, 0));
            entry.0 += p.avg_quality * p.total_executions as f64;
            entry.1 += p.success_rate() * p.total_executions as f64;
            entry.2 += p.total_executions;
        }

        for (tt, (qual_sum, succ_sum, n)) in &by_type {
            if *n < 5 {
                continue;
            }
            let avg_quality = qual_sum / *n as f64;
            let avg_success = succ_sum / *n as f64;
            let recs = recommendations.entry(tt.clone()).or_default();

            if avg_quality > 0.85 && avg_success > 0.9 {
                recs.push("raise_threshold".into());
            }
            if avg_quality > 0.3 && avg_quality < 0.5 && avg_success < 0.7 {
                recs.push("lower_threshold".into());
            }
            if avg_success > 0.95 {
                recs.push("reduce_retries".into());
            }
            if avg_success < 0.5 {
                recs.push("increase_retries".into());
            }
        }

        // Log tuning event.
        if !recommendations.is_empty() {
            let mut entry = HashMap::new();
            entry.insert("event".into(), "auto_tune".into());
            for (tt, recs) in &recommendations {
                entry.insert(tt.clone(), recs.join(", "));
            }
            self.tuning_log.push(entry);
            if self.tuning_log.len() > self.max_tuning_log {
                self.tuning_log.remove(0);
            }
        }

        recommendations
    }

    // ── Stats ───────────────────────────────────────────────────────

    pub fn stats(&self) -> HashMap<String, serde_json::Value> {
        let mut s = HashMap::new();
        s.insert(
            "active_tasks".into(),
            serde_json::Value::Number(self.resolver.active_count().into()),
        );
        s.insert(
            "deadlines_set".into(),
            serde_json::Value::Number(self.deadlines.len().into()),
        );
        s.insert(
            "concurrency_slots".into(),
            serde_json::Value::Number(self.slots.len().into()),
        );
        s.insert(
            "tuning_events".into(),
            serde_json::Value::Number(self.tuning_log.len().into()),
        );
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coordinator() -> SchedulerCoordinator {
        SchedulerCoordinator::new(
            DependencyResolver::new(),
            ExecutionTracker::new(),
            QualityGatePolicy::new(None),
            RetryManager::new(None),
        )
    }

    fn seed_tracker(tracker: &mut ExecutionTracker, task_type: &str, model: &str, n: u32, quality: f64, cost: f64) {
        for i in 0..n {
            let tid = format!("{task_type}_{model}_{i}");
            tracker.record_start(&tid);
            tracker.record_completion(&tid, task_type, model, quality, cost, 0.0, 0, None, 1);
        }
    }

    #[test]
    fn pareto_single_model() {
        let mut coord = make_coordinator();
        seed_tracker(&mut coord.tracker, "coding", "opus", 5, 0.9, 0.10);
        let frontier = coord.compute_pareto_frontier("coding", 2);
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].model, "opus");
    }

    #[test]
    fn pareto_dominance() {
        let mut coord = make_coordinator();
        seed_tracker(&mut coord.tracker, "coding", "opus", 5, 0.9, 0.10);
        seed_tracker(&mut coord.tracker, "coding", "flash", 5, 0.6, 0.001);
        let frontier = coord.compute_pareto_frontier("coding", 2);
        // Both should be on frontier: flash is cheaper, opus is higher quality.
        assert_eq!(frontier.len(), 2);
    }

    #[test]
    fn pareto_dominated_removed() {
        let mut coord = make_coordinator();
        seed_tracker(&mut coord.tracker, "coding", "good", 5, 0.9, 0.05);
        seed_tracker(&mut coord.tracker, "coding", "bad", 5, 0.5, 0.10);
        let frontier = coord.compute_pareto_frontier("coding", 2);
        // "bad" is dominated by "good" (lower quality AND higher cost).
        assert_eq!(frontier.len(), 1);
        assert_eq!(frontier[0].model, "good");
    }

    #[test]
    fn recommend_model_with_quality() {
        let mut coord = make_coordinator();
        seed_tracker(&mut coord.tracker, "coding", "opus", 5, 0.9, 0.10);
        seed_tracker(&mut coord.tracker, "coding", "flash", 5, 0.4, 0.001);
        let rec = coord.recommend_model("coding", 0.7, None, false);
        assert!(rec.is_some());
        assert_eq!(rec.unwrap().model, "opus"); // flash filtered by quality
    }

    #[test]
    fn concurrency_slot() {
        let mut coord = make_coordinator();
        coord.set_provider_limit("anthropic", 2);
        assert!(coord.acquire_slot("anthropic"));
        assert!(coord.acquire_slot("anthropic"));
        assert!(!coord.acquire_slot("anthropic")); // full
        coord.release_slot("anthropic");
        assert!(coord.acquire_slot("anthropic")); // freed
    }

    #[test]
    fn deadline_urgency() {
        let mut coord = make_coordinator();
        let far_future = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs_f64()
            + 86400.0 * 7.0;
        coord.set_deadline("t1", far_future);
        let u = coord.get_urgency("t1");
        assert!(u < 0.1); // very far away

        let no_deadline = coord.get_urgency("t2");
        assert!((no_deadline - 0.0).abs() < 0.001);
    }

    #[test]
    fn plan_execution_basic() {
        let mut coord = make_coordinator();
        coord.resolver.add_task("a", &[], 2).unwrap();
        coord.resolver.add_task("b", &["a"], 2).unwrap();

        let types: HashMap<String, String> = [
            ("a".into(), "planning".into()),
            ("b".into(), "coding".into()),
        ].into();

        let plan = coord.plan_execution(&types, None, None);
        assert_eq!(plan.waves.len(), 2);
        assert_eq!(plan.entries.len(), 2);
        assert!(plan.total_predicted_duration > 0.0);
    }

    #[test]
    fn auto_tune_high_quality() {
        let mut coord = make_coordinator();
        seed_tracker(&mut coord.tracker, "coding", "opus", 10, 0.95, 0.10);
        let recs = coord.auto_tune();
        if let Some(coding_recs) = recs.get("coding") {
            assert!(coding_recs.contains(&"raise_threshold".to_string()));
        }
    }
}
