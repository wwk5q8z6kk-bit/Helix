use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Who submitted this proposal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalSender {
    Agent,
    Mcp,
    Webhook,
    Watcher,
    Relay,
    UserSelf,
}

impl ProposalSender {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Agent => "agent",
            Self::Mcp => "mcp",
            Self::Webhook => "webhook",
            Self::Watcher => "watcher",
            Self::Relay => "relay",
            Self::UserSelf => "self",
        }
    }
}

impl std::str::FromStr for ProposalSender {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "agent" => Ok(Self::Agent),
            "mcp" => Ok(Self::Mcp),
            "webhook" => Ok(Self::Webhook),
            "watcher" => Ok(Self::Watcher),
            "relay" => Ok(Self::Relay),
            "self" => Ok(Self::UserSelf),
            _ => Err(format!("unknown proposal sender: {s}")),
        }
    }
}

impl std::fmt::Display for ProposalSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What action the proposal wants to take
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalAction {
    CreateNode,
    UpdateNode,
    DeleteNode,
    SuggestTag,
    SuggestLink,
    ScheduleReminder,
    Custom(String),
}

impl ProposalAction {
    pub fn as_str(&self) -> &str {
        match self {
            Self::CreateNode => "create_node",
            Self::UpdateNode => "update_node",
            Self::DeleteNode => "delete_node",
            Self::SuggestTag => "suggest_tag",
            Self::SuggestLink => "suggest_link",
            Self::ScheduleReminder => "schedule_reminder",
            Self::Custom(s) => s,
        }
    }
}

impl std::str::FromStr for ProposalAction {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "create_node" => Ok(Self::CreateNode),
            "update_node" => Ok(Self::UpdateNode),
            "delete_node" => Ok(Self::DeleteNode),
            "suggest_tag" => Ok(Self::SuggestTag),
            "suggest_link" => Ok(Self::SuggestLink),
            "schedule_reminder" => Ok(Self::ScheduleReminder),
            other => Ok(Self::Custom(other.to_string())),
        }
    }
}

impl std::fmt::Display for ProposalAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Current state of the proposal
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalState {
    Pending,
    Approved,
    Rejected,
    Expired,
    AutoApproved,
}

impl ProposalState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Rejected => "rejected",
            Self::Expired => "expired",
            Self::AutoApproved => "auto_approved",
        }
    }
}

impl std::str::FromStr for ProposalState {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "approved" => Ok(Self::Approved),
            "rejected" => Ok(Self::Rejected),
            "expired" => Ok(Self::Expired),
            "auto_approved" => Ok(Self::AutoApproved),
            _ => Err(format!("unknown proposal state: {s}")),
        }
    }
}

impl std::fmt::Display for ProposalState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A proposal in the Exchange Inbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub id: Uuid,
    pub node_id: Option<Uuid>,
    pub target_node_id: Option<Uuid>,
    pub sender: ProposalSender,
    pub action: ProposalAction,
    pub state: ProposalState,
    pub confidence: f32,
    pub diff_preview: Option<String>,
    pub payload: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
    pub resolved_at: Option<DateTime<Utc>>,
}

impl Proposal {
    pub fn new(sender: ProposalSender, action: ProposalAction) -> Self {
        Self {
            id: Uuid::now_v7(),
            node_id: None,
            target_node_id: None,
            sender,
            action,
            state: ProposalState::Pending,
            confidence: 0.5,
            diff_preview: None,
            payload: HashMap::new(),
            created_at: Utc::now(),
            updated_at: None,
            resolved_at: None,
        }
    }

    pub fn with_target(mut self, target_node_id: Uuid) -> Self {
        self.target_node_id = Some(target_node_id);
        self
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    pub fn with_diff(mut self, diff: impl Into<String>) -> Self {
        self.diff_preview = Some(diff.into());
        self
    }

    pub fn with_payload(mut self, payload: HashMap<String, serde_json::Value>) -> Self {
        self.payload = payload;
        self
    }
}

// ---------------------------------------------------------------------------
// Blocked Senders
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockedSender {
    pub id: Uuid,
    pub sender_type: String,
    pub sender_pattern: String,
    pub reason: Option<String>,
    pub blocked_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

impl BlockedSender {
    pub fn new(sender_type: impl Into<String>, sender_pattern: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            sender_type: sender_type.into(),
            sender_pattern: sender_pattern.into(),
            reason: None,
            blocked_at: Utc::now(),
            expires_at: None,
        }
    }
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = Some(reason.into());
        self
    }
    pub fn with_expiry(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
    pub fn is_active(&self) -> bool {
        match self.expires_at {
            Some(exp) => Utc::now() < exp,
            None => true,
        }
    }
}

// ---------------------------------------------------------------------------
// Auto-Approve Rules
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoApproveRule {
    pub id: Uuid,
    pub name: String,
    pub sender_pattern: Option<String>,
    pub action_types: Vec<String>,
    pub min_confidence: f32,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl AutoApproveRule {
    pub fn new(name: impl Into<String>, min_confidence: f32) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.into(),
            sender_pattern: None,
            action_types: Vec::new(),
            min_confidence,
            enabled: true,
            created_at: Utc::now(),
            updated_at: None,
        }
    }
    pub fn with_sender(mut self, pattern: impl Into<String>) -> Self {
        self.sender_pattern = Some(pattern.into());
        self
    }
    pub fn with_actions(mut self, actions: Vec<String>) -> Self {
        self.action_types = actions;
        self
    }
    pub fn matches(&self, sender: &str, action: &str, confidence: f32) -> bool {
        if !self.enabled {
            return false;
        }
        if confidence < self.min_confidence {
            return false;
        }
        if let Some(ref pattern) = self.sender_pattern {
            if !glob_match(pattern, sender) {
                return false;
            }
        }
        if !self.action_types.is_empty() && !self.action_types.iter().any(|a| a == action) {
            return false;
        }
        true
    }
}

fn glob_match(pattern: &str, value: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if pattern.ends_with('*') {
        value.starts_with(&pattern[..pattern.len() - 1])
    } else {
        pattern == value
    }
}

// ---------------------------------------------------------------------------
// Undo Snapshots
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UndoSnapshot {
    pub id: Uuid,
    pub proposal_id: Uuid,
    pub snapshot_data: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub used: bool,
}

// ---------------------------------------------------------------------------
// Agent Feedback (Reflection / Feedback Loop)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentFeedback {
    pub id: Uuid,
    pub intent_id: Option<Uuid>,
    pub intent_type: String,
    pub action: String,
    pub confidence_at_time: Option<f32>,
    pub user_edit_delta: Option<f32>,
    pub response_time_ms: Option<u64>,
    pub created_at: DateTime<Utc>,
}

impl AgentFeedback {
    pub fn new(intent_type: impl Into<String>, action: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            intent_id: None,
            intent_type: intent_type.into(),
            action: action.into(),
            confidence_at_time: None,
            user_edit_delta: None,
            response_time_ms: None,
            created_at: Utc::now(),
        }
    }

    pub fn with_intent(mut self, id: Uuid) -> Self {
        self.intent_id = Some(id);
        self
    }

    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence_at_time = Some(c);
        self
    }

    pub fn with_delta(mut self, d: f32) -> Self {
        self.user_edit_delta = Some(d);
        self
    }

    pub fn with_response_time(mut self, ms: u64) -> Self {
        self.response_time_ms = Some(ms);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceOverride {
    pub intent_type: String,
    pub base_adjustment: f32,
    pub auto_apply_threshold: f32,
    pub suppress_below: f32,
    pub updated_at: DateTime<Utc>,
}

impl ConfidenceOverride {
    pub fn new(intent_type: impl Into<String>) -> Self {
        Self {
            intent_type: intent_type.into(),
            base_adjustment: 0.0,
            auto_apply_threshold: 0.95,
            suppress_below: 0.1,
            updated_at: Utc::now(),
        }
    }
}

/// Summary statistics for a given intent type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionStats {
    pub intent_type: String,
    pub total_count: usize,
    pub applied_count: usize,
    pub dismissed_count: usize,
    pub acceptance_rate: f32,
    pub avg_confidence: f32,
    pub override_info: Option<ConfidenceOverride>,
}

// ---------------------------------------------------------------------------
// Autonomy & Precision Controls (Phase 3.1)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AutonomyDecision {
    AutoApply,
    Defer,
    Block,
    QueueForLater,
}

impl std::fmt::Display for AutonomyDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AutoApply => write!(f, "auto_apply"),
            Self::Defer => write!(f, "defer"),
            Self::Block => write!(f, "block"),
            Self::QueueForLater => write!(f, "queue_for_later"),
        }
    }
}

impl std::str::FromStr for AutonomyDecision {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto_apply" => Ok(Self::AutoApply),
            "defer" => Ok(Self::Defer),
            "block" => Ok(Self::Block),
            "queue_for_later" => Ok(Self::QueueForLater),
            _ => Err(format!("unknown autonomy decision: {s}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyRule {
    pub id: Uuid,
    pub rule_type: String, // "global", "domain", "contact", "tag"
    pub scope_key: Option<String>,
    pub auto_apply_threshold: f32,
    pub max_actions_per_hour: u32,
    pub allowed_intent_types: Vec<String>,
    pub blocked_intent_types: Vec<String>,
    pub quiet_hours_start: Option<String>, // "HH:MM"
    pub quiet_hours_end: Option<String>,
    pub quiet_hours_timezone: String,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl AutonomyRule {
    pub fn global(threshold: f32) -> Self {
        Self {
            id: Uuid::now_v7(),
            rule_type: "global".into(),
            scope_key: None,
            auto_apply_threshold: threshold,
            max_actions_per_hour: 10,
            allowed_intent_types: Vec::new(),
            blocked_intent_types: Vec::new(),
            quiet_hours_start: None,
            quiet_hours_end: None,
            quiet_hours_timezone: "UTC".into(),
            enabled: true,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn with_scope(mut self, rule_type: impl Into<String>, key: impl Into<String>) -> Self {
        self.rule_type = rule_type.into();
        self.scope_key = Some(key.into());
        self
    }

    pub fn with_max_actions(mut self, max: u32) -> Self {
        self.max_actions_per_hour = max;
        self
    }

    pub fn with_quiet_hours(
        mut self,
        start: impl Into<String>,
        end: impl Into<String>,
        tz: impl Into<String>,
    ) -> Self {
        self.quiet_hours_start = Some(start.into());
        self.quiet_hours_end = Some(end.into());
        self.quiet_hours_timezone = tz.into();
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyActionLog {
    pub id: Uuid,
    pub rule_id: Option<Uuid>,
    pub intent_type: String,
    pub decision: AutonomyDecision,
    pub confidence: Option<f32>,
    pub reason: Option<String>,
    pub created_at: DateTime<Utc>,
}

// ---------------------------------------------------------------------------
// Communication Relay (Phase 3.2)
// ---------------------------------------------------------------------------

/// Trust level for a relay contact
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    RelayOnly,
    ContextInject,
    Full,
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RelayOnly => write!(f, "relay_only"),
            Self::ContextInject => write!(f, "context_inject"),
            Self::Full => write!(f, "full"),
        }
    }
}

impl std::str::FromStr for TrustLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "relay_only" => Ok(Self::RelayOnly),
            "context_inject" => Ok(Self::ContextInject),
            "full" => Ok(Self::Full),
            _ => Err(format!("unknown trust level: {s}")),
        }
    }
}

/// Channel type for relay messaging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelType {
    Direct,
    Group,
}

impl std::fmt::Display for ChannelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Direct => write!(f, "direct"),
            Self::Group => write!(f, "group"),
        }
    }
}

impl std::str::FromStr for ChannelType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "direct" => Ok(Self::Direct),
            "group" => Ok(Self::Group),
            _ => Err(format!("unknown channel type: {s}")),
        }
    }
}

/// Message direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageDirection {
    Inbound,
    Outbound,
}

impl std::fmt::Display for MessageDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inbound => write!(f, "inbound"),
            Self::Outbound => write!(f, "outbound"),
        }
    }
}

impl std::str::FromStr for MessageDirection {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "inbound" => Ok(Self::Inbound),
            "outbound" => Ok(Self::Outbound),
            _ => Err(format!("unknown message direction: {s}")),
        }
    }
}

/// Message delivery status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageStatus {
    Pending,
    Delivered,
    Read,
    Deferred,
    AutoReplied,
    Failed,
}

impl std::fmt::Display for MessageStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Delivered => write!(f, "delivered"),
            Self::Read => write!(f, "read"),
            Self::Deferred => write!(f, "deferred"),
            Self::AutoReplied => write!(f, "auto_replied"),
            Self::Failed => write!(f, "failed"),
        }
    }
}

impl std::str::FromStr for MessageStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "pending" => Ok(Self::Pending),
            "delivered" => Ok(Self::Delivered),
            "read" => Ok(Self::Read),
            "deferred" => Ok(Self::Deferred),
            "auto_replied" => Ok(Self::AutoReplied),
            "failed" => Ok(Self::Failed),
            _ => Err(format!("unknown message status: {s}")),
        }
    }
}

/// Content type for messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentType {
    Text,
    Voice,
    Attachment,
}

impl std::fmt::Display for ContentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Voice => write!(f, "voice"),
            Self::Attachment => write!(f, "attachment"),
        }
    }
}

impl std::str::FromStr for ContentType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "text" => Ok(Self::Text),
            "voice" => Ok(Self::Voice),
            "attachment" => Ok(Self::Attachment),
            _ => Err(format!("unknown content type: {s}")),
        }
    }
}

/// A contact in the relay network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayContact {
    pub id: Uuid,
    pub display_name: String,
    pub public_key: String,
    pub vault_address: Option<String>,
    pub trust_level: TrustLevel,
    pub autonomy_rule_id: Option<Uuid>,
    pub notes: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl RelayContact {
    pub fn new(display_name: impl Into<String>, public_key: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            display_name: display_name.into(),
            public_key: public_key.into(),
            vault_address: None,
            trust_level: TrustLevel::RelayOnly,
            autonomy_rule_id: None,
            notes: None,
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn with_address(mut self, addr: impl Into<String>) -> Self {
        self.vault_address = Some(addr.into());
        self
    }

    pub fn with_trust(mut self, level: TrustLevel) -> Self {
        self.trust_level = level;
        self
    }
}

/// A messaging channel (direct or group)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayChannel {
    pub id: Uuid,
    pub name: Option<String>,
    pub channel_type: ChannelType,
    pub member_contact_ids: Vec<Uuid>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl RelayChannel {
    pub fn direct(contact_id: Uuid) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: None,
            channel_type: ChannelType::Direct,
            member_contact_ids: vec![contact_id],
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn group(name: impl Into<String>, members: Vec<Uuid>) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: Some(name.into()),
            channel_type: ChannelType::Group,
            member_contact_ids: members,
            created_at: Utc::now(),
            updated_at: None,
        }
    }
}

/// A message in the relay system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelayMessage {
    pub id: Uuid,
    pub channel_id: Uuid,
    pub thread_id: Option<Uuid>,
    pub sender_contact_id: Option<Uuid>,
    pub recipient_contact_id: Option<Uuid>,
    pub direction: MessageDirection,
    pub content: String,
    pub content_type: ContentType,
    pub status: MessageStatus,
    pub vault_node_id: Option<Uuid>,
    pub metadata: HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: Option<DateTime<Utc>>,
}

impl RelayMessage {
    pub fn outbound(channel_id: Uuid, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            channel_id,
            thread_id: None,
            sender_contact_id: None,
            recipient_contact_id: None,
            direction: MessageDirection::Outbound,
            content: content.into(),
            content_type: ContentType::Text,
            status: MessageStatus::Pending,
            vault_node_id: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn inbound(channel_id: Uuid, sender: Uuid, content: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            channel_id,
            thread_id: None,
            sender_contact_id: Some(sender),
            recipient_contact_id: None,
            direction: MessageDirection::Inbound,
            content: content.into(),
            content_type: ContentType::Text,
            status: MessageStatus::Delivered,
            vault_node_id: None,
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: None,
        }
    }

    pub fn with_thread(mut self, thread_id: Uuid) -> Self {
        self.thread_id = Some(thread_id);
        self
    }

    pub fn with_content_type(mut self, ct: ContentType) -> Self {
        self.content_type = ct;
        self
    }

    pub fn with_vault_node(mut self, node_id: Uuid) -> Self {
        self.vault_node_id = Some(node_id);
        self
    }
}
