//! Adaptive retry with model escalation ladders.
//!
//! Port of `python/core/scheduling/retry_strategies.py`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Enums ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryReason {
    ExecutionFailure,
    LowQuality,
    Timeout,
    BudgetExceeded,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationLevel {
    Fast,
    Standard,
    Premium,
    Reasoning,
}

// ── Escalation step ─────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EscalationStep {
    pub level: EscalationLevel,
    pub model: String,
    pub max_attempts: u32,
    pub timeout_multiplier: f64,
}

impl EscalationStep {
    pub fn new(level: EscalationLevel, model: &str) -> Self {
        Self {
            level,
            model: model.to_string(),
            max_attempts: 1,
            timeout_multiplier: 1.0,
        }
    }

    pub fn with_attempts(mut self, n: u32) -> Self {
        self.max_attempts = n;
        self
    }

    pub fn with_timeout(mut self, mult: f64) -> Self {
        self.timeout_multiplier = mult;
        self
    }
}

// ── Retry decision ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryDecision {
    pub should_retry: bool,
    pub reason: RetryReason,
    pub attempt: u32,
    pub model: Option<String>,
    pub agent_id: Option<String>,
    pub timeout_multiplier: f64,
    pub budget_check_passed: bool,
    pub message: String,
}

// ── Strategy ────────────────────────────────────────────────────────

/// Configurable retry strategy with model escalation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryStrategy {
    pub name: String,
    pub ladder: Vec<EscalationStep>,
    pub retry_on_low_quality: bool,
    pub min_acceptable_quality: f64,
    /// Budget checker: (model, estimated_cost) -> allowed.
    /// Not serialized — set at runtime.
    #[serde(skip)]
    pub budget_checker: Option<fn(&str, f64) -> bool>,
    pub max_retries: u32,
    pub estimated_cost_per_attempt: f64,
}

impl Default for RetryStrategy {
    fn default() -> Self {
        strategy_fast_to_premium()
    }
}

impl RetryStrategy {
    /// Total max attempts across the ladder, or `max_retries` if no ladder.
    pub fn total_max_attempts(&self) -> u32 {
        if self.ladder.is_empty() {
            self.max_retries
        } else {
            self.ladder.iter().map(|s| s.max_attempts).sum()
        }
    }

    /// Decide whether to retry and with which model.
    pub fn decide(
        &self,
        attempt: u32,
        reason: RetryReason,
        quality_score: f64,
        _current_model: Option<&str>,
    ) -> RetryDecision {
        // Low quality with acceptable score or retry disabled: don't retry.
        if reason == RetryReason::LowQuality
            && (!self.retry_on_low_quality || quality_score >= self.min_acceptable_quality)
        {
            return RetryDecision {
                should_retry: false,
                reason,
                attempt,
                model: None,
                agent_id: None,
                timeout_multiplier: 1.0,
                budget_check_passed: true,
                message: "Quality acceptable or retry disabled".into(),
            };
        }

        if self.ladder.is_empty() {
            // Simple retry (no escalation ladder).
            if attempt < self.max_retries {
                return RetryDecision {
                    should_retry: true,
                    reason,
                    attempt,
                    model: None,
                    agent_id: None,
                    timeout_multiplier: 1.0,
                    budget_check_passed: true,
                    message: format!(
                        "Simple retry {}/{}",
                        attempt + 1,
                        self.max_retries
                    ),
                };
            }
            return RetryDecision {
                should_retry: false,
                reason,
                attempt,
                model: None,
                agent_id: None,
                timeout_multiplier: 1.0,
                budget_check_passed: true,
                message: "Max retries exhausted".into(),
            };
        }

        // Walk the escalation ladder.
        let mut cumulative = 0u32;
        for step in &self.ladder {
            cumulative += step.max_attempts;
            if attempt < cumulative {
                // Budget check.
                if let Some(checker) = self.budget_checker {
                    let estimated = self.estimated_cost_per_attempt * step.timeout_multiplier;
                    if !checker(&step.model, estimated) {
                        return RetryDecision {
                            should_retry: false,
                            reason,
                            attempt,
                            model: Some(step.model.clone()),
                            agent_id: None,
                            timeout_multiplier: step.timeout_multiplier,
                            budget_check_passed: false,
                            message: "Budget check failed".into(),
                        };
                    }
                }
                return RetryDecision {
                    should_retry: true,
                    reason,
                    attempt,
                    model: Some(step.model.clone()),
                    agent_id: None,
                    timeout_multiplier: step.timeout_multiplier,
                    budget_check_passed: true,
                    message: format!(
                        "Escalating to {} ({:?})",
                        step.model, step.level
                    ),
                };
            }
        }

        RetryDecision {
            should_retry: false,
            reason,
            attempt,
            model: None,
            agent_id: None,
            timeout_multiplier: 1.0,
            budget_check_passed: true,
            message: "All escalation levels exhausted".into(),
        }
    }
}

// ── Pre-built strategies ────────────────────────────────────────────

pub fn strategy_fast_to_premium() -> RetryStrategy {
    RetryStrategy {
        name: "fast_to_premium".into(),
        ladder: vec![
            EscalationStep::new(EscalationLevel::Fast, "flash").with_attempts(1),
            EscalationStep::new(EscalationLevel::Standard, "codex").with_attempts(2),
            EscalationStep::new(EscalationLevel::Premium, "opus")
                .with_attempts(1)
                .with_timeout(2.0),
        ],
        retry_on_low_quality: true,
        min_acceptable_quality: 0.5,
        budget_checker: None,
        max_retries: 3,
        estimated_cost_per_attempt: 0.01,
    }
}

pub fn strategy_quality_first() -> RetryStrategy {
    RetryStrategy {
        name: "quality_first".into(),
        ladder: vec![
            EscalationStep::new(EscalationLevel::Premium, "opus")
                .with_attempts(2)
                .with_timeout(1.5),
            EscalationStep::new(EscalationLevel::Reasoning, "opus-thinking")
                .with_attempts(1)
                .with_timeout(3.0),
        ],
        retry_on_low_quality: true,
        min_acceptable_quality: 0.7,
        budget_checker: None,
        max_retries: 3,
        estimated_cost_per_attempt: 0.05,
    }
}

pub fn strategy_budget_conscious() -> RetryStrategy {
    RetryStrategy {
        name: "budget_conscious".into(),
        ladder: vec![
            EscalationStep::new(EscalationLevel::Fast, "flash").with_attempts(3),
            EscalationStep::new(EscalationLevel::Fast, "gpt-4.1-mini").with_attempts(2),
        ],
        retry_on_low_quality: true,
        min_acceptable_quality: 0.3,
        budget_checker: None,
        max_retries: 5,
        estimated_cost_per_attempt: 0.001,
    }
}

pub fn strategy_no_retry() -> RetryStrategy {
    RetryStrategy {
        name: "no_retry".into(),
        ladder: Vec::new(),
        retry_on_low_quality: false,
        min_acceptable_quality: 0.0,
        budget_checker: None,
        max_retries: 0,
        estimated_cost_per_attempt: 0.0,
    }
}

// ── Retry manager ───────────────────────────────────────────────────

/// Manages retry strategies per task and tracks retry state.
#[derive(Debug, Clone, Default)]
pub struct RetryManager {
    default_strategy: RetryStrategy,
    task_strategies: HashMap<String, RetryStrategy>,
    task_attempts: HashMap<String, u32>,
    retry_history: HashMap<String, Vec<RetryDecision>>,
}

impl RetryManager {
    pub fn new(default_strategy: Option<RetryStrategy>) -> Self {
        Self {
            default_strategy: default_strategy.unwrap_or_default(),
            task_strategies: HashMap::new(),
            task_attempts: HashMap::new(),
            retry_history: HashMap::new(),
        }
    }

    pub fn set_strategy(&mut self, task_id: &str, strategy: RetryStrategy) {
        self.task_strategies
            .insert(task_id.to_string(), strategy);
    }

    pub fn get_strategy(&self, task_id: &str) -> &RetryStrategy {
        self.task_strategies
            .get(task_id)
            .unwrap_or(&self.default_strategy)
    }

    /// Handle a failure: increment attempt, consult strategy, return decision.
    pub fn on_failure(
        &mut self,
        task_id: &str,
        reason: RetryReason,
        quality_score: f64,
        current_model: Option<&str>,
    ) -> RetryDecision {
        let attempt = self
            .task_attempts
            .entry(task_id.to_string())
            .and_modify(|a| *a += 1)
            .or_insert(1);
        let attempt = *attempt;

        let strategy = self
            .task_strategies
            .get(task_id)
            .unwrap_or(&self.default_strategy);

        let decision = strategy.decide(attempt, reason, quality_score, current_model);

        self.retry_history
            .entry(task_id.to_string())
            .or_default()
            .push(decision.clone());

        decision
    }

    pub fn reset(&mut self, task_id: &str) {
        self.task_attempts.remove(task_id);
        self.task_strategies.remove(task_id);
    }

    pub fn get_history(&self, task_id: &str) -> &[RetryDecision] {
        self.retry_history
            .get(task_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_to_premium_ladder() {
        let s = strategy_fast_to_premium();
        assert_eq!(s.total_max_attempts(), 4); // 1 + 2 + 1

        let d = s.decide(0, RetryReason::ExecutionFailure, 0.0, None);
        assert!(d.should_retry);
        assert_eq!(d.model.as_deref(), Some("flash"));

        let d = s.decide(1, RetryReason::ExecutionFailure, 0.0, None);
        assert!(d.should_retry);
        assert_eq!(d.model.as_deref(), Some("codex"));

        let d = s.decide(3, RetryReason::ExecutionFailure, 0.0, None);
        assert!(d.should_retry);
        assert_eq!(d.model.as_deref(), Some("opus"));
        assert!((d.timeout_multiplier - 2.0).abs() < 0.01);

        let d = s.decide(4, RetryReason::ExecutionFailure, 0.0, None);
        assert!(!d.should_retry);
    }

    #[test]
    fn no_retry_strategy() {
        let s = strategy_no_retry();
        assert_eq!(s.total_max_attempts(), 0);
        let d = s.decide(0, RetryReason::ExecutionFailure, 0.0, None);
        assert!(!d.should_retry);
    }

    #[test]
    fn low_quality_respects_threshold() {
        let s = strategy_fast_to_premium();
        // Quality above min_acceptable (0.5) — don't retry.
        let d = s.decide(0, RetryReason::LowQuality, 0.6, None);
        assert!(!d.should_retry);

        // Quality below min_acceptable — retry.
        let d = s.decide(0, RetryReason::LowQuality, 0.3, None);
        assert!(d.should_retry);
    }

    #[test]
    fn retry_manager_tracks_attempts() {
        let mut mgr = RetryManager::new(None);
        let d1 = mgr.on_failure("t1", RetryReason::ExecutionFailure, 0.0, None);
        assert!(d1.should_retry);
        assert_eq!(d1.attempt, 1);

        let d2 = mgr.on_failure("t1", RetryReason::ExecutionFailure, 0.0, None);
        assert_eq!(d2.attempt, 2);

        let history = mgr.get_history("t1");
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn reset_clears_state() {
        let mut mgr = RetryManager::new(None);
        mgr.on_failure("t1", RetryReason::ExecutionFailure, 0.0, None);
        mgr.reset("t1");
        let d = mgr.on_failure("t1", RetryReason::ExecutionFailure, 0.0, None);
        assert_eq!(d.attempt, 1);
    }
}
