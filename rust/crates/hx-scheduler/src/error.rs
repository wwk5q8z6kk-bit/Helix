use thiserror::Error;

#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("Duplicate task: {0}")]
    DuplicateTask(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Invalid state transition for task '{task}': {from} -> {to}")]
    InvalidTransition {
        task: String,
        from: String,
        to: String,
    },

    #[error("Cycle detected: {}", .0.join(" -> "))]
    CycleDetected(Vec<String>),

    #[error("Validation failed: {}", .0.join("; "))]
    ValidationFailed(Vec<String>),

    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

pub type Result<T> = std::result::Result<T, SchedulerError>;
