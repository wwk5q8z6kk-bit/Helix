use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Vault State
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VaultState {
    Uninitialized,
    Sealed,
    Unsealed,
}

impl VaultState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Uninitialized => "uninitialized",
            Self::Sealed => "sealed",
            Self::Unsealed => "unsealed",
        }
    }
}

impl std::str::FromStr for VaultState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "uninitialized" => Ok(Self::Uninitialized),
            "sealed" => Ok(Self::Sealed),
            "unsealed" => Ok(Self::Unsealed),
            _ => Err(format!("unknown vault state: {s}")),
        }
    }
}

impl std::fmt::Display for VaultState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Vault Metadata
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultMeta {
    pub schema_version: u32,
    pub master_salt: String,
    pub verification_blob: String,
    pub key_epoch: u64,
    pub created_at: DateTime<Utc>,
    pub last_rotated_at: Option<DateTime<Utc>>,
    pub macos_keychain_service: Option<String>,
    /// Shamir threshold (M): minimum shares needed to reconstruct the key.
    pub shamir_threshold: Option<u8>,
    /// Shamir total (N): total shares created.
    pub shamir_total: Option<u8>,
    /// When Shamir shares were last rotated (re-split).
    pub shamir_last_rotated_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Key Epochs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyEpoch {
    pub epoch: u64,
    pub wrapped_key: Option<String>,
    pub created_at: DateTime<Utc>,
    pub grace_expires_at: Option<DateTime<Utc>>,
    pub retired_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Domains
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainKey {
    pub id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub derivation_info: String,
    pub epoch: u64,
    pub created_at: DateTime<Utc>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub credential_count: u64,
}

impl DomainKey {
    pub fn new(name: impl Into<String>, derivation_info: impl Into<String>) -> Self {
        Self {
            id: Uuid::now_v7(),
            name: name.into(),
            description: None,
            derivation_info: derivation_info.into(),
            epoch: 0,
            created_at: Utc::now(),
            revoked_at: None,
            credential_count: 0,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }
}

// ---------------------------------------------------------------------------
// Credential State (lifecycle state machine)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialState {
    Active,
    Expiring,
    Expired,
    Archived,
    Destroyed,
}

impl CredentialState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Expiring => "expiring",
            Self::Expired => "expired",
            Self::Archived => "archived",
            Self::Destroyed => "destroyed",
        }
    }
}

impl std::str::FromStr for CredentialState {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "active" => Ok(Self::Active),
            "expiring" => Ok(Self::Expiring),
            "expired" => Ok(Self::Expired),
            "archived" => Ok(Self::Archived),
            "destroyed" => Ok(Self::Destroyed),
            _ => Err(format!("unknown credential state: {s}")),
        }
    }
}

impl std::fmt::Display for CredentialState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Stored Credential
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredCredential {
    pub id: Uuid,
    pub domain_id: Uuid,
    pub name: String,
    pub description: Option<String>,
    pub kind: String,
    pub encrypted_value: String,
    pub derivation_info: String,
    pub epoch: u64,
    pub state: CredentialState,
    pub tags: Vec<String>,
    pub metadata: std::collections::HashMap<String, serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub last_accessed_at: DateTime<Utc>,
    pub access_count: u64,
    pub expires_at: Option<DateTime<Utc>>,
    pub archived_at: Option<DateTime<Utc>>,
    pub destroyed_at: Option<DateTime<Utc>>,
    pub delegation_id: Option<Uuid>,
    pub version: u32,
    /// Whether `name` and `description` are encrypted at rest.
    pub metadata_encrypted: bool,
}

impl StoredCredential {
    pub fn new(
        domain_id: Uuid,
        name: impl Into<String>,
        kind: impl Into<String>,
        encrypted_value: String,
        derivation_info: impl Into<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::now_v7(),
            domain_id,
            name: name.into(),
            description: None,
            kind: kind.into(),
            encrypted_value,
            derivation_info: derivation_info.into(),
            epoch: 0,
            state: CredentialState::Active,
            tags: Vec::new(),
            metadata: std::collections::HashMap::new(),
            created_at: now,
            updated_at: now,
            last_accessed_at: now,
            access_count: 0,
            expires_at: None,
            archived_at: None,
            destroyed_at: None,
            delegation_id: None,
            version: 1,
            metadata_encrypted: false,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_epoch(mut self, epoch: u64) -> Self {
        self.epoch = epoch;
        self
    }

    pub fn with_expires_at(mut self, expires_at: DateTime<Utc>) -> Self {
        self.expires_at = Some(expires_at);
        self
    }
}

// ---------------------------------------------------------------------------
// Domain ACLs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAcl {
    pub id: Uuid,
    pub domain_id: Uuid,
    pub subject: String,
    pub can_read: bool,
    pub can_write: bool,
    pub can_admin: bool,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Delegations
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationPermissions {
    pub can_read: bool,
    pub can_use: bool,
    pub can_delegate: bool,
}

impl Default for DelegationPermissions {
    fn default() -> Self {
        Self {
            can_read: true,
            can_use: false,
            can_delegate: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Delegation {
    pub id: Uuid,
    pub credential_id: Uuid,
    pub delegatee: String,
    pub parent_id: Option<Uuid>,
    pub permissions: DelegationPermissions,
    pub chain_hash: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub revoked_at: Option<DateTime<Utc>>,
    pub max_depth: u32,
    pub depth: u32,
}

// ---------------------------------------------------------------------------
// Audit
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeychainAuditAction {
    VaultInitialized,
    VaultUnlocked,
    VaultLocked,
    VaultUnlockFailed,
    CredentialStored,
    CredentialRead,
    CredentialUpdated,
    CredentialArchived,
    CredentialDestroyed,
    CredentialRotated,
    DomainCreated,
    DomainRevoked,
    DelegationCreated,
    DelegationRevoked,
    KeyRotated,
    ZkProofGenerated,
    ZkProofVerified,
    ProofGenerated,
    ProofVerified,
    BreachDetected,
    LifecycleTransition,
    ShamirEnabled,
    ShamirUnseal,
    ShamirRotated,
}

impl KeychainAuditAction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VaultInitialized => "vault_initialized",
            Self::VaultUnlocked => "vault_unlocked",
            Self::VaultLocked => "vault_locked",
            Self::VaultUnlockFailed => "vault_unlock_failed",
            Self::CredentialStored => "credential_stored",
            Self::CredentialRead => "credential_read",
            Self::CredentialUpdated => "credential_updated",
            Self::CredentialArchived => "credential_archived",
            Self::CredentialDestroyed => "credential_destroyed",
            Self::CredentialRotated => "credential_rotated",
            Self::DomainCreated => "domain_created",
            Self::DomainRevoked => "domain_revoked",
            Self::DelegationCreated => "delegation_created",
            Self::DelegationRevoked => "delegation_revoked",
            Self::KeyRotated => "key_rotated",
            Self::ZkProofGenerated => "zk_proof_generated",
            Self::ZkProofVerified => "zk_proof_verified",
            Self::ProofGenerated => "proof_generated",
            Self::ProofVerified => "proof_verified",
            Self::BreachDetected => "breach_detected",
            Self::LifecycleTransition => "lifecycle_transition",
            Self::ShamirEnabled => "shamir_enabled",
            Self::ShamirUnseal => "shamir_unseal",
            Self::ShamirRotated => "shamir_rotated",
        }
    }
}

impl std::str::FromStr for KeychainAuditAction {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "vault_initialized" => Ok(Self::VaultInitialized),
            "vault_unlocked" => Ok(Self::VaultUnlocked),
            "vault_locked" => Ok(Self::VaultLocked),
            "vault_unlock_failed" => Ok(Self::VaultUnlockFailed),
            "credential_stored" => Ok(Self::CredentialStored),
            "credential_read" => Ok(Self::CredentialRead),
            "credential_updated" => Ok(Self::CredentialUpdated),
            "credential_archived" => Ok(Self::CredentialArchived),
            "credential_destroyed" => Ok(Self::CredentialDestroyed),
            "credential_rotated" => Ok(Self::CredentialRotated),
            "domain_created" => Ok(Self::DomainCreated),
            "domain_revoked" => Ok(Self::DomainRevoked),
            "delegation_created" => Ok(Self::DelegationCreated),
            "delegation_revoked" => Ok(Self::DelegationRevoked),
            "key_rotated" => Ok(Self::KeyRotated),
            "zk_proof_generated" => Ok(Self::ZkProofGenerated),
            "zk_proof_verified" => Ok(Self::ZkProofVerified),
            "proof_generated" => Ok(Self::ProofGenerated),
            "proof_verified" => Ok(Self::ProofVerified),
            "breach_detected" => Ok(Self::BreachDetected),
            "lifecycle_transition" => Ok(Self::LifecycleTransition),
            "shamir_enabled" => Ok(Self::ShamirEnabled),
            "shamir_unseal" => Ok(Self::ShamirUnseal),
            "shamir_rotated" => Ok(Self::ShamirRotated),
            _ => Err(format!("unknown keychain audit action: {s}")),
        }
    }
}

impl std::fmt::Display for KeychainAuditAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeychainAuditEntry {
    pub id: Uuid,
    pub sequence: i64,
    pub action: KeychainAuditAction,
    pub subject: String,
    pub resource_id: Option<String>,
    pub details: Option<serde_json::Value>,
    pub entry_hash: String,
    pub previous_hash: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub source_ip: Option<String>,
    pub signature: Option<String>,
}

// ---------------------------------------------------------------------------
// Audit Verification Result
// ---------------------------------------------------------------------------

/// Outcome of `verify_audit_integrity()` with distinct sealed/unsealed semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuditVerificationResult {
    /// All checks passed: hash chain valid **and** HMAC signatures verified.
    FullyVerified,
    /// Hash chain is valid, but vault was sealed so HMAC signatures could not be checked.
    ChainOnlyValid,
    /// Either the hash chain is broken or at least one HMAC signature is invalid.
    Failed,
}

impl AuditVerificationResult {
    /// Backward-compatible boolean: `true` for `FullyVerified` and `ChainOnlyValid`.
    pub fn is_valid(&self) -> bool {
        matches!(self, Self::FullyVerified | Self::ChainOnlyValid)
    }

    /// Whether HMAC signatures were actually checked.
    pub fn signatures_checked(&self) -> bool {
        matches!(self, Self::FullyVerified)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FullyVerified => "fully_verified",
            Self::ChainOnlyValid => "chain_only_valid",
            Self::Failed => "failed",
        }
    }
}

impl std::fmt::Display for AuditVerificationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Access Patterns & Breach Detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessPattern {
    pub credential_id: Uuid,
    pub accessor: String,
    pub source_ip: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub hour_of_day: u8,
    pub day_of_week: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreachAlertType {
    UnusualFrequency,
    NewAccessor,
    OffHoursAccess,
    RapidSequentialAccess,
    FailedAccessSpike,
}

impl BreachAlertType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::UnusualFrequency => "unusual_frequency",
            Self::NewAccessor => "new_accessor",
            Self::OffHoursAccess => "off_hours_access",
            Self::RapidSequentialAccess => "rapid_sequential_access",
            Self::FailedAccessSpike => "failed_access_spike",
        }
    }
}

impl std::str::FromStr for BreachAlertType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "unusual_frequency" => Ok(Self::UnusualFrequency),
            "new_accessor" => Ok(Self::NewAccessor),
            "off_hours_access" => Ok(Self::OffHoursAccess),
            "rapid_sequential_access" => Ok(Self::RapidSequentialAccess),
            "failed_access_spike" => Ok(Self::FailedAccessSpike),
            _ => Err(format!("unknown breach alert type: {s}")),
        }
    }
}

impl std::fmt::Display for BreachAlertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreachSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl BreachSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Critical => "critical",
        }
    }
}

impl std::str::FromStr for BreachSeverity {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "critical" => Ok(Self::Critical),
            _ => Err(format!("unknown breach severity: {s}")),
        }
    }
}

impl std::fmt::Display for BreachSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreachAlert {
    pub id: Uuid,
    pub credential_id: Uuid,
    pub alert_type: BreachAlertType,
    pub severity: BreachSeverity,
    pub description: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub acknowledged_at: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// Zero-Knowledge Access Proofs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessProof {
    pub credential_id: Uuid,
    pub challenge_nonce: String,
    pub proof: String,
    pub generated_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// Type alias for backward compatibility.
pub type ZkAccessProof = AccessProof;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vault_state_roundtrip() {
        for state in [
            VaultState::Uninitialized,
            VaultState::Sealed,
            VaultState::Unsealed,
        ] {
            let s = state.as_str();
            let parsed: VaultState = s.parse().unwrap();
            assert_eq!(parsed, state);
            assert_eq!(state.to_string(), s);
        }
    }

    #[test]
    fn credential_state_roundtrip() {
        for state in [
            CredentialState::Active,
            CredentialState::Expiring,
            CredentialState::Expired,
            CredentialState::Archived,
            CredentialState::Destroyed,
        ] {
            let s = state.as_str();
            let parsed: CredentialState = s.parse().unwrap();
            assert_eq!(parsed, state);
        }
    }

    #[test]
    fn domain_key_builder() {
        let dk = DomainKey::new("api-keys", "domain:api-keys")
            .with_description("API key storage")
            .with_epoch(1);
        assert_eq!(dk.name, "api-keys");
        assert_eq!(dk.description.as_deref(), Some("API key storage"));
        assert_eq!(dk.epoch, 1);
    }

    #[test]
    fn stored_credential_builder() {
        let cred = StoredCredential::new(
            Uuid::now_v7(),
            "my-api-key",
            "api_key",
            "encrypted-data".into(),
            "cred:my-api-key",
        )
        .with_description("Test credential")
        .with_tags(vec!["production".into()])
        .with_epoch(1);

        assert_eq!(cred.name, "my-api-key");
        assert_eq!(cred.kind, "api_key");
        assert_eq!(cred.epoch, 1);
        assert_eq!(cred.state, CredentialState::Active);
        assert_eq!(cred.version, 1);
    }

    #[test]
    fn audit_action_roundtrip() {
        let action = KeychainAuditAction::CredentialStored;
        let s = action.as_str();
        let parsed: KeychainAuditAction = s.parse().unwrap();
        assert_eq!(parsed, action);
    }

    #[test]
    fn breach_alert_type_roundtrip() {
        let t = BreachAlertType::RapidSequentialAccess;
        let s = t.as_str();
        let parsed: BreachAlertType = s.parse().unwrap();
        assert_eq!(parsed, t);
    }

    #[test]
    fn breach_severity_roundtrip() {
        let sev = BreachSeverity::Critical;
        let s = sev.as_str();
        let parsed: BreachSeverity = s.parse().unwrap();
        assert_eq!(parsed, sev);
    }
}
