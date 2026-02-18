pub mod handlers;
pub mod queue;
pub mod worker;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Dead,
    Cancelled,
}

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Dead => "dead",
            Self::Cancelled => "cancelled",
        }
    }
}

impl std::str::FromStr for JobStatus {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "running" => Ok(Self::Running),
            "completed" => Ok(Self::Completed),
            "failed" => Ok(Self::Failed),
            "dead" => Ok(Self::Dead),
            "cancelled" => Ok(Self::Cancelled),
            _ => Err(format!("unknown job status: {s}")),
        }
    }
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Job {
    pub id: Uuid,
    pub job_type: String,
    pub payload: serde_json::Value,
    pub status: JobStatus,
    pub priority: i32,
    pub retries: u32,
    pub max_retries: u32,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
    pub scheduled_at: Option<DateTime<Utc>>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub next_retry_at: Option<DateTime<Utc>>,
    pub idempotency_key: Option<String>,
}

#[derive(Debug, Clone)]
pub struct EnqueueOptions {
    pub priority: i32,
    pub max_retries: u32,
    pub delay_secs: u64,
    pub idempotency_key: Option<String>,
}

impl Default for EnqueueOptions {
    fn default() -> Self {
        Self {
            priority: 0,
            max_retries: 3,
            delay_secs: 0,
            idempotency_key: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobStats {
    pub pending: u64,
    pub running: u64,
    pub completed: u64,
    pub failed: u64,
    pub dead: u64,
    pub cancelled: u64,
    pub total: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_status_serialization_roundtrip() {
        let statuses = vec![
            JobStatus::Pending,
            JobStatus::Running,
            JobStatus::Completed,
            JobStatus::Failed,
            JobStatus::Dead,
            JobStatus::Cancelled,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: JobStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn job_status_from_str() {
        assert_eq!("pending".parse::<JobStatus>().unwrap(), JobStatus::Pending);
        assert_eq!("running".parse::<JobStatus>().unwrap(), JobStatus::Running);
        assert_eq!("completed".parse::<JobStatus>().unwrap(), JobStatus::Completed);
        assert_eq!("failed".parse::<JobStatus>().unwrap(), JobStatus::Failed);
        assert_eq!("dead".parse::<JobStatus>().unwrap(), JobStatus::Dead);
        assert_eq!("cancelled".parse::<JobStatus>().unwrap(), JobStatus::Cancelled);
        assert!("invalid".parse::<JobStatus>().is_err());
    }

    #[test]
    fn job_status_display() {
        assert_eq!(JobStatus::Pending.to_string(), "pending");
        assert_eq!(JobStatus::Dead.to_string(), "dead");
    }

    #[test]
    fn enqueue_options_default() {
        let opts = EnqueueOptions::default();
        assert_eq!(opts.priority, 0);
        assert_eq!(opts.max_retries, 3);
        assert_eq!(opts.delay_secs, 0);
        assert!(opts.idempotency_key.is_none());
    }

    #[test]
    fn job_serialization_roundtrip() {
        let job = Job {
            id: Uuid::now_v7(),
            job_type: "test_job".into(),
            payload: serde_json::json!({"key": "value"}),
            status: JobStatus::Pending,
            priority: 5,
            retries: 0,
            max_retries: 3,
            error: None,
            created_at: Utc::now(),
            scheduled_at: None,
            started_at: None,
            completed_at: None,
            next_retry_at: None,
            idempotency_key: Some("idem-123".into()),
        };

        let json = serde_json::to_string(&job).unwrap();
        let deserialized: Job = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, job.id);
        assert_eq!(deserialized.job_type, "test_job");
        assert_eq!(deserialized.status, JobStatus::Pending);
        assert_eq!(deserialized.priority, 5);
        assert_eq!(deserialized.idempotency_key.as_deref(), Some("idem-123"));
    }

    #[test]
    fn job_stats_serialization() {
        let stats = JobStats {
            pending: 10,
            running: 2,
            completed: 50,
            failed: 3,
            dead: 1,
            cancelled: 0,
            total: 66,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: JobStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.pending, 10);
        assert_eq!(deserialized.total, 66);
    }
}
