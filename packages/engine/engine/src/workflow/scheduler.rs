use std::collections::HashMap;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use tokio::sync::RwLock;
use uuid::Uuid;

use super::executor::WorkflowExecutor;
use super::WorkflowTrigger;

/// Tracks the last run time for each cron-triggered workflow.
pub struct WorkflowScheduler {
    executor: Arc<WorkflowExecutor>,
    last_runs: Arc<RwLock<HashMap<Uuid, DateTime<Utc>>>>,
}

impl WorkflowScheduler {
    pub fn new(executor: Arc<WorkflowExecutor>) -> Self {
        Self {
            executor,
            last_runs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Run the scheduler loop, checking cron triggers every 60 seconds.
    /// Stops when a shutdown signal is received.
    pub async fn run(&self, mut shutdown_rx: tokio::sync::broadcast::Receiver<()>) {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        interval.tick().await; // skip immediate first tick

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("workflow scheduler shutting down");
                    break;
                }
                _ = interval.tick() => {
                    self.check_and_execute().await;
                }
            }
        }
    }

    /// Check all workflows for due cron triggers and execute them.
    pub async fn check_and_execute(&self) {
        let now = Utc::now();
        let workflows = self.executor.list_workflows().await;

        for wf in workflows {
            if let WorkflowTrigger::Cron { ref expression } = wf.trigger {
                let last_run = self.last_runs.read().await.get(&wf.id).copied();

                if self.is_due(expression, last_run, now) {
                    tracing::info!(
                        workflow_id = %wf.id,
                        workflow_name = %wf.name,
                        cron = %expression,
                        "executing scheduled workflow"
                    );

                    self.last_runs.write().await.insert(wf.id, now);

                    let executor = Arc::clone(&self.executor);
                    let wf_id = wf.id;
                    tokio::spawn(async move {
                        match executor.execute(wf_id, HashMap::new()).await {
                            Ok(exec) => {
                                tracing::info!(
                                    workflow_id = %wf_id,
                                    execution_id = %exec.id,
                                    status = ?exec.status,
                                    "scheduled workflow execution completed"
                                );
                            }
                            Err(e) => {
                                tracing::error!(
                                    workflow_id = %wf_id,
                                    error = %e,
                                    "scheduled workflow execution failed"
                                );
                            }
                        }
                    });
                }
            }
        }
    }

    /// Check if a cron expression is due to fire, given the last run time and
    /// the current time.
    fn is_due(
        &self,
        cron_expr: &str,
        last_run: Option<DateTime<Utc>>,
        now: DateTime<Utc>,
    ) -> bool {
        use std::str::FromStr;

        let schedule: cron::Schedule = match cron::Schedule::from_str(cron_expr) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(cron = %cron_expr, error = %e, "invalid cron expression");
                return false;
            }
        };

        // Find the next occurrence after the last run (or epoch if never run)
        let after = last_run.unwrap_or_else(|| {
            DateTime::<Utc>::from_timestamp(0, 0).unwrap_or(now)
        });

        match schedule.after(&after).next() {
            Some(next_time) => next_time <= now,
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workflow::{Workflow, WorkflowTrigger};

    fn make_scheduler() -> (Arc<WorkflowExecutor>, WorkflowScheduler) {
        let executor = Arc::new(WorkflowExecutor::new());
        let scheduler = WorkflowScheduler::new(Arc::clone(&executor));
        (executor, scheduler)
    }

    #[test]
    fn is_due_never_run_and_past_due() {
        let (_executor, scheduler) = make_scheduler();
        let now = Utc::now();
        // "every minute" cron — should always be due if never run
        let due = scheduler.is_due("* * * * * *", None, now);
        assert!(due);
    }

    #[test]
    fn is_due_recently_run_not_due() {
        let (_executor, scheduler) = make_scheduler();
        let now = Utc::now();
        // Just ran — "every hour" should not be due again immediately
        let due = scheduler.is_due("0 0 * * * *", Some(now), now);
        assert!(!due);
    }

    #[test]
    fn is_due_invalid_cron_returns_false() {
        let (_executor, scheduler) = make_scheduler();
        let now = Utc::now();
        let due = scheduler.is_due("not a cron expression", None, now);
        assert!(!due);
    }

    #[tokio::test]
    async fn check_and_execute_triggers_due_workflows() {
        let (executor, scheduler) = make_scheduler();
        let wf = Workflow::new(
            "cron-wf",
            WorkflowTrigger::Cron {
                expression: "* * * * * *".into(), // every second
            },
        );
        let wf_id = executor.register_workflow(wf).await;

        scheduler.check_and_execute().await;

        // Give spawned task time to complete
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let execs = executor.list_executions().await;
        assert!(!execs.is_empty(), "should have created an execution");
        assert_eq!(execs[0].workflow_id, wf_id);
    }

    #[tokio::test]
    async fn check_and_execute_skips_manual_triggers() {
        let (executor, scheduler) = make_scheduler();
        let wf = Workflow::new("manual-wf", WorkflowTrigger::Manual);
        executor.register_workflow(wf).await;

        scheduler.check_and_execute().await;
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;

        let execs = executor.list_executions().await;
        assert!(execs.is_empty(), "manual workflow should not auto-execute");
    }
}
