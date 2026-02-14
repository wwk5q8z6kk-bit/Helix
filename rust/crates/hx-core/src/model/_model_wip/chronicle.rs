use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChronicleEntry {
    pub id: Uuid,
    pub node_id: Option<Uuid>,
    pub step_name: String,
    pub logic: String,
    pub input_snapshot: Option<String>,
    pub output_snapshot: Option<String>,
    pub timestamp: DateTime<Utc>,
}

impl ChronicleEntry {
    pub fn new(step_name: impl Into<String>, logic: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            node_id: None,
            step_name: step_name.into(),
            logic: logic.into(),
            input_snapshot: None,
            output_snapshot: None,
            timestamp: Utc::now(),
        }
    }

    pub fn with_node(mut self, node_id: Uuid) -> Self {
        self.node_id = Some(node_id);
        self
    }

    pub fn with_snapshots(mut self, input: Option<String>, output: Option<String>) -> Self {
        self.input_snapshot = input;
        self.output_snapshot = output;
        self
    }
}
