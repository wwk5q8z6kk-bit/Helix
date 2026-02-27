//! Gateway pairing types for secure channel authentication.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PairingState {
    Pending,
    Confirmed,
    Expired,
    Revoked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairingSession {
    pub id: Uuid,
    pub channel_name: String,
    pub otp_hash: String,
    pub state: PairingState,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    pub bearer_token_hash: Option<String>,
}

impl PairingSession {
    pub fn is_expired(&self) -> bool {
        Utc::now() > self.expires_at
    }
}
