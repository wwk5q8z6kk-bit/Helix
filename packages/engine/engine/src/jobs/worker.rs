use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::broadcast;
use tracing;

use hx_core::MvResult;

use super::queue::JobQueue;
use super::Job;

/// Trait for handling a specific job type.
#[async_trait]
pub trait JobHandler: Send + Sync {
    /// The job_type string this handler processes.
    fn handles(&self) -> &str;

    /// Execute the job. Return Ok(()) on success, Err on failure.
    async fn execute(&self, job: &Job) -> MvResult<()>;
}

/// Worker that polls the job queue and dispatches to registered handlers.
pub struct JobWorker {
    queue: Arc<JobQueue>,
    handlers: Vec<Box<dyn JobHandler>>,
    poll_interval: Duration,
    concurrency: usize,
}

impl JobWorker {
    pub fn new(queue: Arc<JobQueue>) -> Self {
        Self {
            queue,
            handlers: Vec::new(),
            poll_interval: Duration::from_secs(1),
            concurrency: 1,
        }
    }

    pub fn with_handler(mut self, handler: Box<dyn JobHandler>) -> Self {
        self.handlers.push(handler);
        self
    }

    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    pub fn with_concurrency(mut self, n: usize) -> Self {
        self.concurrency = n.max(1);
        self
    }

    /// Run the worker loop until a shutdown signal is received.
    pub async fn run(&self, mut shutdown_rx: broadcast::Receiver<()>) {
        tracing::info!(
            concurrency = self.concurrency,
            poll_ms = self.poll_interval.as_millis() as u64,
            handlers = self.handlers.len(),
            "job worker started"
        );

        loop {
            tokio::select! {
                _ = shutdown_rx.recv() => {
                    tracing::info!("job worker shutting down");
                    break;
                }
                _ = tokio::time::sleep(self.poll_interval) => {
                    // Try to dequeue up to `concurrency` jobs.
                    for _ in 0..self.concurrency {
                        match self.queue.dequeue().await {
                            Ok(Some(job)) => {
                                self.process_job(job).await;
                            }
                            Ok(None) => break, // No more jobs available.
                            Err(e) => {
                                tracing::error!(error = %e, "dequeue error");
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    async fn process_job(&self, job: Job) {
        let handler = self
            .handlers
            .iter()
            .find(|h| h.handles() == job.job_type);

        let Some(handler) = handler else {
            tracing::warn!(
                job_id = %job.id,
                job_type = %job.job_type,
                "no handler registered for job type"
            );
            if let Err(e) = self
                .queue
                .fail(job.id, &format!("no handler for job type: {}", job.job_type))
                .await
            {
                tracing::error!(error = %e, "failed to mark unhandled job as failed");
            }
            return;
        };

        tracing::info!(
            job_id = %job.id,
            job_type = %job.job_type,
            "processing job"
        );

        match handler.execute(&job).await {
            Ok(()) => {
                if let Err(e) = self.queue.complete(job.id).await {
                    tracing::error!(job_id = %job.id, error = %e, "failed to mark job complete");
                }
            }
            Err(e) => {
                tracing::warn!(
                    job_id = %job.id,
                    error = %e,
                    "job execution failed"
                );
                if let Err(fail_err) = self.queue.fail(job.id, &e.to_string()).await {
                    tracing::error!(
                        job_id = %job.id,
                        error = %fail_err,
                        "failed to mark job as failed"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::{EnqueueOptions, JobStatus};
    use std::sync::atomic::{AtomicU32, Ordering};

    struct CountingHandler {
        job_type: String,
        call_count: Arc<AtomicU32>,
        should_fail: bool,
    }

    #[async_trait]
    impl JobHandler for CountingHandler {
        fn handles(&self) -> &str {
            &self.job_type
        }

        async fn execute(&self, _job: &Job) -> MvResult<()> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            if self.should_fail {
                Err(hx_core::HxError::Internal("intentional failure".into()))
            } else {
                Ok(())
            }
        }
    }

    #[tokio::test]
    async fn worker_processes_job() {
        let queue = Arc::new(JobQueue::in_memory().unwrap());
        let count = Arc::new(AtomicU32::new(0));

        queue
            .enqueue("count_job", serde_json::json!({}), EnqueueOptions::default())
            .await
            .unwrap();

        let worker = JobWorker::new(Arc::clone(&queue))
            .with_handler(Box::new(CountingHandler {
                job_type: "count_job".into(),
                call_count: Arc::clone(&count),
                should_fail: false,
            }))
            .with_poll_interval(Duration::from_millis(50));

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let worker_handle = tokio::spawn(async move {
            worker.run(shutdown_rx).await;
        });

        // Wait for processing.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
        worker_handle.await.unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 1);

        let job = queue.list(Some(JobStatus::Completed), 10).await.unwrap();
        assert_eq!(job.len(), 1);
    }

    #[tokio::test]
    async fn worker_handles_failure() {
        let queue = Arc::new(JobQueue::in_memory().unwrap());
        let count = Arc::new(AtomicU32::new(0));

        queue
            .enqueue(
                "fail_job",
                serde_json::json!({}),
                EnqueueOptions {
                    max_retries: 2,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let worker = JobWorker::new(Arc::clone(&queue))
            .with_handler(Box::new(CountingHandler {
                job_type: "fail_job".into(),
                call_count: Arc::clone(&count),
                should_fail: true,
            }))
            .with_poll_interval(Duration::from_millis(50));

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let worker_handle = tokio::spawn(async move {
            worker.run(shutdown_rx).await;
        });

        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
        worker_handle.await.unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 1);

        let failed = queue.list(Some(JobStatus::Failed), 10).await.unwrap();
        assert_eq!(failed.len(), 1);
    }

    #[tokio::test]
    async fn worker_no_handler_fails_job() {
        let queue = Arc::new(JobQueue::in_memory().unwrap());

        queue
            .enqueue(
                "unknown_type",
                serde_json::json!({}),
                EnqueueOptions {
                    max_retries: 1,
                    ..Default::default()
                },
            )
            .await
            .unwrap();

        let worker = JobWorker::new(Arc::clone(&queue))
            .with_poll_interval(Duration::from_millis(50));

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let worker_handle = tokio::spawn(async move {
            worker.run(shutdown_rx).await;
        });

        tokio::time::sleep(Duration::from_millis(200)).await;
        let _ = shutdown_tx.send(());
        worker_handle.await.unwrap();

        // Should be dead (max_retries=1, one failure = dead).
        let dead = queue.dead_letter_queue(10).await.unwrap();
        assert_eq!(dead.len(), 1);
    }

    #[tokio::test]
    async fn worker_shutdown_is_graceful() {
        let queue = Arc::new(JobQueue::in_memory().unwrap());

        let worker = JobWorker::new(Arc::clone(&queue))
            .with_poll_interval(Duration::from_millis(50));

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let worker_handle = tokio::spawn(async move {
            worker.run(shutdown_rx).await;
        });

        // Immediately shut down.
        let _ = shutdown_tx.send(());
        worker_handle.await.unwrap();
    }

    #[tokio::test]
    async fn worker_multiple_handlers() {
        let queue = Arc::new(JobQueue::in_memory().unwrap());
        let count_a = Arc::new(AtomicU32::new(0));
        let count_b = Arc::new(AtomicU32::new(0));

        queue
            .enqueue("type_a", serde_json::json!({}), EnqueueOptions::default())
            .await
            .unwrap();
        queue
            .enqueue("type_b", serde_json::json!({}), EnqueueOptions::default())
            .await
            .unwrap();

        let worker = JobWorker::new(Arc::clone(&queue))
            .with_handler(Box::new(CountingHandler {
                job_type: "type_a".into(),
                call_count: Arc::clone(&count_a),
                should_fail: false,
            }))
            .with_handler(Box::new(CountingHandler {
                job_type: "type_b".into(),
                call_count: Arc::clone(&count_b),
                should_fail: false,
            }))
            .with_poll_interval(Duration::from_millis(50))
            .with_concurrency(2);

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let worker_handle = tokio::spawn(async move {
            worker.run(shutdown_rx).await;
        });

        tokio::time::sleep(Duration::from_millis(300)).await;
        let _ = shutdown_tx.send(());
        worker_handle.await.unwrap();

        assert_eq!(count_a.load(Ordering::SeqCst), 1);
        assert_eq!(count_b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn worker_concurrency_setting() {
        let queue = Arc::new(JobQueue::in_memory().unwrap());
        let count = Arc::new(AtomicU32::new(0));

        for _ in 0..5 {
            queue
                .enqueue("batch", serde_json::json!({}), EnqueueOptions::default())
                .await
                .unwrap();
        }

        let worker = JobWorker::new(Arc::clone(&queue))
            .with_handler(Box::new(CountingHandler {
                job_type: "batch".into(),
                call_count: Arc::clone(&count),
                should_fail: false,
            }))
            .with_poll_interval(Duration::from_millis(50))
            .with_concurrency(5);

        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        let worker_handle = tokio::spawn(async move {
            worker.run(shutdown_rx).await;
        });

        tokio::time::sleep(Duration::from_millis(300)).await;
        let _ = shutdown_tx.send(());
        worker_handle.await.unwrap();

        assert_eq!(count.load(Ordering::SeqCst), 5);
    }
}
