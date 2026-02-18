//! Quality gates with three-level precedence: default < task < edge.
//!
//! Port of `python/core/scheduling/quality_gates.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Enums ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateAction {
    Retry,
    WarnAndPass,
    Block,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateVerdict {
    Passed,
    Retry,
    WarnPass,
    Blocked,
    Failed,
}

// ── Quality gate ────────────────────────────────────────────────────

/// Quality threshold for a dependency edge or task output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityGate {
    pub min_quality: f64,
    pub min_hrm: f64,
    pub max_retries: u32,
    pub gate_action: GateAction,
    pub escalation_models: Vec<String>,
}

impl QualityGate {
    pub fn new(min_quality: f64) -> Self {
        Self {
            min_quality,
            min_hrm: 0.0,
            max_retries: 2,
            gate_action: GateAction::Retry,
            escalation_models: Vec::new(),
        }
    }
}

impl Default for QualityGate {
    fn default() -> Self {
        GATE_STANDARD.clone()
    }
}

// Pre-built configurations.
pub static GATE_STRICT: once_cell::sync::Lazy<QualityGate> =
    once_cell::sync::Lazy::new(|| QualityGate {
        min_quality: 0.7,
        min_hrm: 0.6,
        max_retries: 3,
        gate_action: GateAction::Retry,
        escalation_models: Vec::new(),
    });

pub static GATE_STANDARD: once_cell::sync::Lazy<QualityGate> =
    once_cell::sync::Lazy::new(|| QualityGate {
        min_quality: 0.5,
        min_hrm: 0.0,
        max_retries: 2,
        gate_action: GateAction::Retry,
        escalation_models: Vec::new(),
    });

pub static GATE_LENIENT: once_cell::sync::Lazy<QualityGate> =
    once_cell::sync::Lazy::new(|| QualityGate {
        min_quality: 0.3,
        min_hrm: 0.0,
        max_retries: 1,
        gate_action: GateAction::WarnAndPass,
        escalation_models: Vec::new(),
    });

pub static GATE_DISABLED: once_cell::sync::Lazy<QualityGate> =
    once_cell::sync::Lazy::new(|| QualityGate {
        min_quality: 0.0,
        min_hrm: 0.0,
        max_retries: 0,
        gate_action: GateAction::WarnAndPass,
        escalation_models: Vec::new(),
    });

// ── Gate result ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GateResult {
    pub task_id: String,
    pub verdict: GateVerdict,
    pub gate: QualityGate,
    pub quality_score: f64,
    pub hrm_score: f64,
    pub attempt: u32,
    pub reason: String,
    pub suggested_model: Option<String>,
}

// ── Policy ──────────────────────────────────────────────────────────

/// Manages quality gates at three levels: default, task, edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityGatePolicy {
    default_gate: QualityGate,
    task_gates: HashMap<String, QualityGate>,
    edge_gates: HashMap<(String, String), QualityGate>,
    gate_attempts: HashMap<String, u32>,
}

impl Default for QualityGatePolicy {
    fn default() -> Self {
        Self::new(None)
    }
}

impl QualityGatePolicy {
    pub fn new(default_gate: Option<QualityGate>) -> Self {
        Self {
            default_gate: default_gate.unwrap_or_else(|| GATE_STANDARD.clone()),
            task_gates: HashMap::new(),
            edge_gates: HashMap::new(),
            gate_attempts: HashMap::new(),
        }
    }

    pub fn set_default_gate(&mut self, gate: QualityGate) {
        self.default_gate = gate;
    }

    pub fn set_task_gate(&mut self, task_id: &str, gate: QualityGate) {
        self.task_gates.insert(task_id.to_string(), gate);
    }

    pub fn set_edge_gate(&mut self, upstream: &str, downstream: &str, gate: QualityGate) {
        self.edge_gates
            .insert((upstream.to_string(), downstream.to_string()), gate);
    }

    /// Get the applicable gate: edge > task > default.
    pub fn get_gate(&self, task_id: &str, downstream_id: Option<&str>) -> &QualityGate {
        if let Some(ds) = downstream_id {
            if let Some(g) = self
                .edge_gates
                .get(&(task_id.to_string(), ds.to_string()))
            {
                return g;
            }
        }
        self.task_gates
            .get(task_id)
            .unwrap_or(&self.default_gate)
    }

    /// Evaluate a task's output against the applicable gate.
    pub fn evaluate(
        &mut self,
        task_id: &str,
        quality_score: f64,
        hrm_score: f64,
        downstream_ids: Option<&[&str]>,
    ) -> GateResult {
        // Find strictest applicable gate.
        let gate = if let Some(ds_ids) = downstream_ids {
            ds_ids
                .iter()
                .map(|ds| self.get_gate(task_id, Some(ds)))
                .max_by(|a, b| {
                    a.min_quality
                        .partial_cmp(&b.min_quality)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(self.get_gate(task_id, None))
                .clone()
        } else {
            self.get_gate(task_id, None).clone()
        };

        // Increment attempt counter.
        let attempt = self
            .gate_attempts
            .entry(task_id.to_string())
            .and_modify(|a| *a += 1)
            .or_insert(1);
        let attempt = *attempt;

        let quality_ok = quality_score >= gate.min_quality;
        let hrm_ok = gate.min_hrm <= 0.0 || hrm_score >= gate.min_hrm;

        if quality_ok && hrm_ok {
            return GateResult {
                task_id: task_id.to_string(),
                verdict: GateVerdict::Passed,
                gate,
                quality_score,
                hrm_score,
                attempt,
                reason: "Quality meets threshold".into(),
                suggested_model: None,
            };
        }

        // Below threshold — retry if attempts remain.
        if attempt <= gate.max_retries {
            let suggested = gate
                .escalation_models
                .get((attempt - 1) as usize)
                .cloned();
            let reason = format!(
                "Quality {quality_score:.2} below threshold {:.2}, retry {attempt}/{}",
                gate.min_quality, gate.max_retries
            );
            return GateResult {
                task_id: task_id.to_string(),
                verdict: GateVerdict::Retry,
                gate,
                quality_score,
                hrm_score,
                attempt,
                reason,
                suggested_model: suggested,
            };
        }

        // Retries exhausted — apply gate action.
        let (verdict, reason) = match gate.gate_action {
            GateAction::WarnAndPass | GateAction::Retry => (
                GateVerdict::WarnPass,
                "Retries exhausted, passing with warning".into(),
            ),
            GateAction::Block => (
                GateVerdict::Blocked,
                "Retries exhausted, blocking downstream".into(),
            ),
            GateAction::Fail => (
                GateVerdict::Failed,
                "Retries exhausted, treating as failure".into(),
            ),
        };

        GateResult {
            task_id: task_id.to_string(),
            verdict,
            gate,
            quality_score,
            hrm_score,
            attempt,
            reason,
            suggested_model: None,
        }
    }

    pub fn reset_attempts(&mut self, task_id: &str) {
        self.gate_attempts.remove(task_id);
    }

    pub fn remove_task(&mut self, task_id: &str) {
        self.task_gates.remove(task_id);
        self.gate_attempts.remove(task_id);
        self.edge_gates
            .retain(|(u, d), _| u != task_id && d != task_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_passes_above_threshold() {
        let mut policy = QualityGatePolicy::new(None);
        let r = policy.evaluate("t1", 0.8, 0.0, None);
        assert_eq!(r.verdict, GateVerdict::Passed);
    }

    #[test]
    fn gate_retries_below_threshold() {
        let mut policy = QualityGatePolicy::new(None);
        let r = policy.evaluate("t1", 0.3, 0.0, None);
        assert_eq!(r.verdict, GateVerdict::Retry);
        assert_eq!(r.attempt, 1);
    }

    #[test]
    fn gate_exhausts_retries() {
        let mut policy = QualityGatePolicy::new(Some(QualityGate {
            min_quality: 0.9,
            min_hrm: 0.0,
            max_retries: 2,
            gate_action: GateAction::Fail,
            escalation_models: Vec::new(),
        }));
        policy.evaluate("t1", 0.3, 0.0, None); // attempt 1: retry
        policy.evaluate("t1", 0.3, 0.0, None); // attempt 2: retry
        let r = policy.evaluate("t1", 0.3, 0.0, None); // attempt 3: fail
        assert_eq!(r.verdict, GateVerdict::Failed);
    }

    #[test]
    fn edge_gate_precedence() {
        let mut policy = QualityGatePolicy::new(None);
        let strict = QualityGate {
            min_quality: 0.9,
            ..GATE_STANDARD.clone()
        };
        policy.set_edge_gate("a", "b", strict.clone());
        let gate = policy.get_gate("a", Some("b"));
        assert!((gate.min_quality - 0.9).abs() < 0.01);

        // Without downstream: falls through to default.
        let gate2 = policy.get_gate("a", None);
        assert!((gate2.min_quality - 0.5).abs() < 0.01);
    }

    #[test]
    fn warn_and_pass_on_exhaustion() {
        let mut policy = QualityGatePolicy::new(Some(QualityGate {
            min_quality: 0.9,
            min_hrm: 0.0,
            max_retries: 1,
            gate_action: GateAction::WarnAndPass,
            escalation_models: Vec::new(),
        }));
        policy.evaluate("t1", 0.3, 0.0, None); // attempt 1: retry
        let r = policy.evaluate("t1", 0.3, 0.0, None); // attempt 2: warn_pass
        assert_eq!(r.verdict, GateVerdict::WarnPass);
    }

    #[test]
    fn hrm_check() {
        let mut policy = QualityGatePolicy::new(Some(GATE_STRICT.clone()));
        // Quality ok (0.8 >= 0.7) but HRM too low (0.1 < 0.6).
        let r = policy.evaluate("t1", 0.8, 0.1, None);
        assert_eq!(r.verdict, GateVerdict::Retry);
    }
}
