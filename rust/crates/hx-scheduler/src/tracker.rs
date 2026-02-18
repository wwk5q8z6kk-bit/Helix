//! Execution tracking with incremental statistics and critical-path analysis.
//!
//! Port of `python/core/scheduling/execution_tracker.py`.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::dag::DependencyResolver;

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

// ── Records ─────────────────────────────────────────────────────────

/// Immutable record of a single task execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub task_id: String,
    pub task_type: String,
    pub model: String,
    pub agent_id: Option<String>,
    pub start_time: f64,
    pub end_time: f64,
    pub success: bool,
    pub quality_score: f64,
    pub hrm_score: f64,
    pub tokens_used: u64,
    pub cost: f64,
    pub attempt: u32,
}

impl ExecutionRecord {
    pub fn duration(&self) -> f64 {
        if self.end_time > self.start_time {
            self.end_time - self.start_time
        } else {
            0.0
        }
    }
}

// ── Profiles (incremental statistics) ───────────────────────────────

/// Aggregated statistics for a (task_type, model) pair.
///
/// Updated incrementally using the formula:
///   `new_avg = old_avg + (new_val - old_avg) / n`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionProfile {
    pub task_type: String,
    pub model: String,
    pub total_executions: u64,
    pub successes: u64,
    pub failures: u64,
    pub avg_duration: f64,
    pub avg_quality: f64,
    pub avg_hrm: f64,
    pub avg_cost: f64,
    pub avg_tokens: f64,
    pub min_duration: f64,
    pub max_duration: f64,
    pub best_quality: f64,
    pub worst_quality: f64,
}

impl ExecutionProfile {
    pub fn new(task_type: &str, model: &str) -> Self {
        Self {
            task_type: task_type.to_string(),
            model: model.to_string(),
            total_executions: 0,
            successes: 0,
            failures: 0,
            avg_duration: 0.0,
            avg_quality: 0.0,
            avg_hrm: 0.0,
            avg_cost: 0.0,
            avg_tokens: 0.0,
            min_duration: f64::INFINITY,
            max_duration: 0.0,
            best_quality: 0.0,
            worst_quality: 1.0,
        }
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successes as f64 / self.total_executions as f64
        }
    }

    /// Incrementally update statistics with a new execution record.
    pub fn update(&mut self, record: &ExecutionRecord) {
        self.total_executions += 1;
        if record.success {
            self.successes += 1;
        } else {
            self.failures += 1;
        }

        let n = self.total_executions as f64;
        let dur = record.duration();

        // Incremental mean: avg += (new - avg) / n
        self.avg_duration += (dur - self.avg_duration) / n;
        self.avg_quality += (record.quality_score - self.avg_quality) / n;
        self.avg_hrm += (record.hrm_score - self.avg_hrm) / n;
        self.avg_cost += (record.cost - self.avg_cost) / n;
        self.avg_tokens += (record.tokens_used as f64 - self.avg_tokens) / n;

        if dur < self.min_duration {
            self.min_duration = dur;
        }
        if dur > self.max_duration {
            self.max_duration = dur;
        }
        if record.quality_score > self.best_quality {
            self.best_quality = record.quality_score;
        }
        if record.quality_score < self.worst_quality {
            self.worst_quality = record.quality_score;
        }
    }
}

// ── Critical-path result ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CriticalPathResult {
    pub path: Vec<String>,
    pub total_duration: f64,
    pub total_cost: f64,
    pub waves: Vec<Vec<String>>,
    pub wave_durations: Vec<f64>,
}

// ── Tracker ─────────────────────────────────────────────────────────

/// Tracks execution metrics and builds predictive profiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTracker {
    records: HashMap<String, ExecutionRecord>,
    profiles: HashMap<(String, String), ExecutionProfile>,
    in_flight: HashMap<String, f64>,
    default_duration: f64,
    default_cost: f64,
}

impl Default for ExecutionTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionTracker {
    pub fn new() -> Self {
        Self {
            records: HashMap::new(),
            profiles: HashMap::new(),
            in_flight: HashMap::new(),
            default_duration: 10.0,
            default_cost: 0.01,
        }
    }

    // ── Recording ───────────────────────────────────────────────────

    pub fn record_start(&mut self, task_id: &str) {
        self.in_flight.insert(task_id.to_string(), now());
    }

    pub fn record_completion(
        &mut self,
        task_id: &str,
        task_type: &str,
        model: &str,
        quality_score: f64,
        cost: f64,
        hrm_score: f64,
        tokens_used: u64,
        agent_id: Option<&str>,
        attempt: u32,
    ) -> ExecutionRecord {
        let start_time = self.in_flight.remove(task_id).unwrap_or_else(now);
        let record = ExecutionRecord {
            task_id: task_id.to_string(),
            task_type: task_type.to_string(),
            model: model.to_string(),
            agent_id: agent_id.map(String::from),
            start_time,
            end_time: now(),
            success: true,
            quality_score,
            hrm_score,
            tokens_used,
            cost,
            attempt,
        };
        self.update_profile(&record);
        self.records.insert(task_id.to_string(), record.clone());
        record
    }

    pub fn record_failure(
        &mut self,
        task_id: &str,
        task_type: &str,
        model: &str,
        agent_id: Option<&str>,
        attempt: u32,
    ) -> ExecutionRecord {
        let start_time = self.in_flight.remove(task_id).unwrap_or_else(now);
        let record = ExecutionRecord {
            task_id: task_id.to_string(),
            task_type: task_type.to_string(),
            model: model.to_string(),
            agent_id: agent_id.map(String::from),
            start_time,
            end_time: now(),
            success: false,
            quality_score: 0.0,
            hrm_score: 0.0,
            tokens_used: 0,
            cost: 0.0,
            attempt,
        };
        self.update_profile(&record);
        self.records.insert(task_id.to_string(), record.clone());
        record
    }

    // ── Prediction ──────────────────────────────────────────────────

    pub fn predict_duration(&self, task_type: &str, model: &str) -> f64 {
        if let Some(p) = self.profiles.get(&(task_type.to_string(), model.to_string())) {
            if p.total_executions >= 3 {
                return p.avg_duration;
            }
        }
        // Fallback: average across all models for this task_type.
        let mut total_dur = 0.0;
        let mut total_n = 0u64;
        for ((tt, _), prof) in &self.profiles {
            if tt == task_type && prof.total_executions >= 1 {
                total_dur += prof.avg_duration * prof.total_executions as f64;
                total_n += prof.total_executions;
            }
        }
        if total_n > 0 {
            total_dur / total_n as f64
        } else {
            self.default_duration
        }
    }

    pub fn predict_quality(&self, task_type: &str, model: &str) -> f64 {
        if let Some(p) = self.profiles.get(&(task_type.to_string(), model.to_string())) {
            if p.successes >= 3 {
                return p.avg_quality;
            }
        }
        0.5 // neutral default
    }

    pub fn predict_cost(&self, task_type: &str, model: &str) -> f64 {
        if let Some(p) = self.profiles.get(&(task_type.to_string(), model.to_string())) {
            if p.total_executions >= 3 {
                return p.avg_cost;
            }
        }
        self.default_cost
    }

    pub fn predict_success_rate(&self, task_type: &str, model: &str) -> f64 {
        if let Some(p) = self.profiles.get(&(task_type.to_string(), model.to_string())) {
            if p.total_executions >= 5 {
                return p.success_rate();
            }
        }
        0.9 // optimistic default
    }

    // ── Critical path ───────────────────────────────────────────────

    /// Dynamic programming critical-path analysis over a DAG.
    pub fn compute_critical_path(
        &self,
        resolver: &DependencyResolver,
        task_types: &HashMap<String, String>,
        task_models: &HashMap<String, String>,
    ) -> CriticalPathResult {
        let waves = resolver.get_execution_waves();
        let mut earliest_finish: HashMap<String, f64> = HashMap::new();
        let mut task_durations: HashMap<String, f64> = HashMap::new();
        let mut wave_durations = Vec::new();

        for wave in &waves {
            let mut max_dur: f64 = 0.0;
            for tid in wave {
                let tt = task_types.get(tid).map(|s| s.as_str()).unwrap_or("general");
                let model = task_models.get(tid).map(|s| s.as_str()).unwrap_or("default");
                let dur = self.predict_duration(tt, model);
                task_durations.insert(tid.clone(), dur);

                // Earliest finish = max(earliest_finish[dep]) + duration
                let max_dep_finish = resolver
                    .get_node(tid)
                    .map(|n| {
                        n.dependencies
                            .iter()
                            .filter_map(|d| earliest_finish.get(d))
                            .cloned()
                            .fold(0.0f64, f64::max)
                    })
                    .unwrap_or(0.0);

                earliest_finish.insert(tid.clone(), max_dep_finish + dur);
                if dur > max_dur {
                    max_dur = dur;
                }
            }
            wave_durations.push(max_dur);
        }

        let total_duration: f64 = wave_durations.iter().sum();
        let total_cost: f64 = task_types
            .iter()
            .map(|(tid, tt)| {
                let model = task_models.get(tid).map(|s| s.as_str()).unwrap_or("default");
                self.predict_cost(tt, model)
            })
            .sum();

        // Trace back from the task with the largest earliest_finish.
        let mut path = Vec::new();
        if let Some((start, _)) = earliest_finish
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        {
            let mut current = start.clone();
            path.push(current.clone());

            loop {
                let node = resolver.get_node(&current);
                if let Some(n) = node {
                    if n.dependencies.is_empty() {
                        break;
                    }
                    // Follow the dependency with the largest earliest_finish.
                    if let Some(prev) = n
                        .dependencies
                        .iter()
                        .max_by(|a, b| {
                            let fa = earliest_finish.get(*a).unwrap_or(&0.0);
                            let fb = earliest_finish.get(*b).unwrap_or(&0.0);
                            fa.partial_cmp(fb).unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        path.push(prev.clone());
                        current = prev.clone();
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            path.reverse();
        }

        CriticalPathResult {
            path,
            total_duration,
            total_cost,
            waves: waves.clone(),
            wave_durations,
        }
    }

    // ── Queries ─────────────────────────────────────────────────────

    pub fn get_profile(&self, task_type: &str, model: &str) -> Option<&ExecutionProfile> {
        self.profiles
            .get(&(task_type.to_string(), model.to_string()))
    }

    pub fn get_record(&self, task_id: &str) -> Option<&ExecutionRecord> {
        self.records.get(task_id)
    }

    pub fn get_all_profiles(&self) -> &HashMap<(String, String), ExecutionProfile> {
        &self.profiles
    }

    /// Find the model with the best quality/cost ratio for a task type.
    pub fn get_best_model_for(
        &self,
        task_type: &str,
        min_quality: f64,
    ) -> Option<String> {
        self.profiles
            .values()
            .filter(|p| {
                p.task_type == task_type
                    && p.successes >= 2
                    && p.avg_quality >= min_quality
            })
            .max_by(|a, b| {
                let ea = a.avg_quality / a.avg_cost.max(0.001);
                let eb = b.avg_quality / b.avg_cost.max(0.001);
                ea.partial_cmp(&eb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.model.clone())
    }

    // ── Internal ────────────────────────────────────────────────────

    fn update_profile(&mut self, record: &ExecutionRecord) {
        let key = (record.task_type.clone(), record.model.clone());
        let profile = self
            .profiles
            .entry(key)
            .or_insert_with(|| ExecutionProfile::new(&record.task_type, &record.model));
        profile.update(record);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(tracker: &mut ExecutionTracker, task_type: &str, model: &str, n: u32, quality: f64, cost: f64) {
        for i in 0..n {
            let tid = format!("{task_type}_{model}_{i}");
            tracker.record_start(&tid);
            tracker.record_completion(&tid, task_type, model, quality, cost, 0.0, 0, None, 1);
        }
    }

    #[test]
    fn profile_incremental_update() {
        let mut t = ExecutionTracker::new();
        seed(&mut t, "coding", "opus", 5, 0.8, 0.05);
        let p = t.get_profile("coding", "opus").unwrap();
        assert_eq!(p.total_executions, 5);
        assert_eq!(p.successes, 5);
        assert!((p.avg_quality - 0.8).abs() < 0.01);
        assert!((p.avg_cost - 0.05).abs() < 0.01);
    }

    #[test]
    fn predict_with_history() {
        let mut t = ExecutionTracker::new();
        seed(&mut t, "coding", "opus", 5, 0.9, 0.10);
        assert!(t.predict_quality("coding", "opus") > 0.8);
        assert!(t.predict_cost("coding", "opus") > 0.05);
    }

    #[test]
    fn predict_default_without_history() {
        let t = ExecutionTracker::new();
        assert!((t.predict_duration("coding", "opus") - 10.0).abs() < 0.01);
        assert!((t.predict_cost("coding", "opus") - 0.01).abs() < 0.001);
        assert!((t.predict_quality("coding", "opus") - 0.5).abs() < 0.01);
    }

    #[test]
    fn best_model_for_task() {
        let mut t = ExecutionTracker::new();
        seed(&mut t, "coding", "opus", 5, 0.9, 0.10);
        seed(&mut t, "coding", "flash", 5, 0.6, 0.001);
        // flash has much better efficiency (0.6/0.001 >> 0.9/0.1)
        let best = t.get_best_model_for("coding", 0.5).unwrap();
        assert_eq!(best, "flash");
    }

    #[test]
    fn critical_path_linear() {
        let mut r = DependencyResolver::new();
        r.add_task("a", &[], 2).unwrap();
        r.add_task("b", &["a"], 2).unwrap();
        r.add_task("c", &["b"], 2).unwrap();

        let types: HashMap<String, String> = [
            ("a".into(), "coding".into()),
            ("b".into(), "coding".into()),
            ("c".into(), "coding".into()),
        ].into();
        let models: HashMap<String, String> = [
            ("a".into(), "opus".into()),
            ("b".into(), "opus".into()),
            ("c".into(), "opus".into()),
        ].into();

        let t = ExecutionTracker::new();
        let cp = t.compute_critical_path(&r, &types, &models);
        assert_eq!(cp.path, vec!["a", "b", "c"]);
        assert_eq!(cp.waves.len(), 3);
    }
}
