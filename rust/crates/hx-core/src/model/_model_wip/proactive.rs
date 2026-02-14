use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProactiveInsight {
    pub id: Uuid,
    pub title: String,
    pub content: String,
    pub insight_type: InsightType,
    pub related_node_ids: Vec<Uuid>,
    pub importance: f32,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum InsightType {
    Connection,
    Trend,
    Gap,
    Reminder,
    General,
}

impl ProactiveInsight {
    pub fn new(title: impl Into<String>, content: impl Into<String>, insight_type: InsightType) -> Self {
        Self {
            id: Uuid::now_v7(),
            title: title.into(),
            content: content.into(),
            insight_type,
            related_node_ids: Vec::new(),
            importance: 0.5,
            created_at: Utc::now(),
        }
    }

    pub fn with_related_nodes(mut self, ids: Vec<Uuid>) -> Self {
        self.related_node_ids = ids;
        self
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance;
        self
    }
}
