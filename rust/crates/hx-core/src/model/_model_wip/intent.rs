use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedIntent {
    pub id: Uuid,
    pub node_id: Uuid,
    pub intent_type: IntentType,
    pub confidence: f32,
    pub parameters: serde_json::Value,
    pub status: IntentStatus,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IntentType {
    ScheduleReminder,
    LinkToProject,
    ExtractTask,
    SuggestTag,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum IntentStatus {
    Suggested,
    Applied,
    Dismissed,
}

impl CapturedIntent {
    pub fn new(node_id: Uuid, intent_type: IntentType) -> Self {
        Self {
            id: Uuid::now_v7(),
            node_id,
            intent_type,
            confidence: 0.0,
            parameters: serde_json::Value::Null,
            status: IntentStatus::Suggested,
            created_at: Utc::now(),
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_parameters(mut self, params: serde_json::Value) -> Self {
        self.parameters = params;
        self
    }
}
