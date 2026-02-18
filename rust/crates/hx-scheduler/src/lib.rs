//! # hx-scheduler
//!
//! DAG-based task scheduler with quality gates, adaptive retry, Pareto-optimal
//! model selection, and predictive workflow preview.
//!
//! ## Modules
//!
//! - [`dag`] — Dependency resolver with cycle detection and failure propagation
//! - [`tracker`] — Execution tracking with incremental statistics
//! - [`gates`] — Quality gates with three-level precedence
//! - [`retry`] — Adaptive retry with model escalation ladders
//! - [`coordinator`] — Unified scheduling intelligence
//! - [`builder`] — Declarative workflow builder with predictive preview
//! - [`error`] — Error types

pub mod builder;
pub mod coordinator;
pub mod dag;
pub mod error;
pub mod gates;
pub mod retry;
pub mod tracker;

// Re-exports for convenience.
pub use builder::{
    WorkflowBuilder, WorkflowDefinition, WorkflowPreview, WorkflowStatus, WorkflowTask,
    code_review_pipeline, parallel_analysis_pipeline, research_pipeline,
};
pub use coordinator::{
    ConcurrencySlot, ExecutionPlan, ModelRecommendation, ScheduleEntry, SchedulerCoordinator,
};
pub use dag::{DependencyResolver, TaskNode, TaskState};
pub use error::{Result, SchedulerError};
pub use gates::{
    GateAction, GateResult, GateVerdict, QualityGate, QualityGatePolicy,
    GATE_DISABLED, GATE_LENIENT, GATE_STANDARD, GATE_STRICT,
};
pub use retry::{
    EscalationLevel, EscalationStep, RetryDecision, RetryManager, RetryReason, RetryStrategy,
};
pub use tracker::{CriticalPathResult, ExecutionProfile, ExecutionRecord, ExecutionTracker};
