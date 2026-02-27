//! Keychain engine — orchestrates vault lifecycle, credential CRUD,
//! delegations, ZK proofs, audit, and breach detection.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use chrono::{DateTime, Datelike, Timelike, Utc};
use rand::RngCore;
use tokio::sync::RwLock;
use tokio::task::JoinHandle;
use uuid::Uuid;
use zeroize::Zeroizing;

use hx_core::credentials::CredentialStore;
use hx_core::error::{HxError, MvResult};
use hx_core::model::keychain::*;
use hx_core::traits::KeychainStore;
use hx_storage::crypto::EncryptionConfig;
use hx_storage::sealed_runtime::{
    clear_runtime_root_key, runtime_is_degraded_security, set_runtime_root_key,
};
use hx_storage::vault_crypto::{
    validate_argon2_params, ShamirShare, VaultCrypto, VaultCryptoError,
};

use crate::config::KeychainConfig;

// ---------------------------------------------------------------------------
// Breach Detector
// ---------------------------------------------------------------------------

pub struct BreachDetectorConfig {
    pub max_accesses_per_minute: u32,
    pub quiet_hours_start: u8,
    pub quiet_hours_end: u8,
    pub new_accessor_lookback_days: u32,
}

impl Default for BreachDetectorConfig {
    fn default() -> Self {
        Self {
            max_accesses_per_minute: 10,
            quiet_hours_start: 22,
            quiet_hours_end: 6,
            new_accessor_lookback_days: 30,
        }
    }
}

pub struct BreachDetector {
    config: BreachDetectorConfig,
}

impl BreachDetector {
    pub fn new(config: BreachDetectorConfig) -> Self {
        Self { config }
    }

    pub fn analyze(
        &self,
        patterns: &[AccessPattern],
        new_access: &AccessPattern,
    ) -> Vec<BreachAlert> {
        let mut alerts = Vec::new();

        // 1. Rapid sequential access — check accesses in the last minute
        let one_minute_ago = new_access.timestamp - chrono::Duration::seconds(60);
        let recent_count = patterns
            .iter()
            .filter(|p| p.timestamp > one_minute_ago)
            .count() as u32;
        if recent_count >= self.config.max_accesses_per_minute {
            alerts.push(BreachAlert {
                id: Uuid::now_v7(),
                credential_id: new_access.credential_id,
                alert_type: BreachAlertType::RapidSequentialAccess,
                severity: BreachSeverity::High,
                description: format!(
                    "{} accesses in the last minute (threshold: {})",
                    recent_count, self.config.max_accesses_per_minute
                ),
                details: None,
                timestamp: Utc::now(),
                acknowledged_at: None,
            });
        }

        // 2. Off-hours access
        let hour = new_access.hour_of_day;
        let in_quiet = if self.config.quiet_hours_start > self.config.quiet_hours_end {
            hour >= self.config.quiet_hours_start || hour < self.config.quiet_hours_end
        } else {
            hour >= self.config.quiet_hours_start && hour < self.config.quiet_hours_end
        };
        if in_quiet {
            alerts.push(BreachAlert {
                id: Uuid::now_v7(),
                credential_id: new_access.credential_id,
                alert_type: BreachAlertType::OffHoursAccess,
                severity: BreachSeverity::Medium,
                description: format!(
                    "Access at hour {} (quiet hours: {}-{})",
                    hour, self.config.quiet_hours_start, self.config.quiet_hours_end
                ),
                details: None,
                timestamp: Utc::now(),
                acknowledged_at: None,
            });
        }

        // 3. New accessor — check if this accessor has been seen before
        let lookback =
            Utc::now() - chrono::Duration::days(self.config.new_accessor_lookback_days as i64);
        let known = patterns
            .iter()
            .any(|p| p.accessor == new_access.accessor && p.timestamp > lookback);
        if !known && !patterns.is_empty() {
            alerts.push(BreachAlert {
                id: Uuid::now_v7(),
                credential_id: new_access.credential_id,
                alert_type: BreachAlertType::NewAccessor,
                severity: BreachSeverity::Medium,
                description: format!(
                    "New accessor '{}' not seen in the last {} days",
                    new_access.accessor, self.config.new_accessor_lookback_days
                ),
                details: None,
                timestamp: Utc::now(),
                acknowledged_at: None,
            });
        }

        alerts
    }
}

impl Default for BreachDetector {
    fn default() -> Self {
        Self::new(BreachDetectorConfig::default())
    }
}

// ---------------------------------------------------------------------------
// KeychainEngine
// ---------------------------------------------------------------------------

/// Status of Shamir share collection for unseal.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ShamirStatus {
    pub shares_collected: u8,
    pub threshold: u8,
    pub total: u8,
    pub ready: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasterKeySource {
    SecureEnclave,
    OsSecureStorage,
    PassphraseArgon2id,
}

impl MasterKeySource {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SecureEnclave => "secure_enclave",
            Self::OsSecureStorage => "os_secure_storage",
            Self::PassphraseArgon2id => "passphrase_argon2id",
        }
    }
}

pub struct KeychainEngine {
    pub store: Arc<dyn KeychainStore>,
    crypto: RwLock<VaultCrypto>,
    cred_store: Arc<CredentialStore>,
    breach_detector: BreachDetector,
    failed_attempts: AtomicU32,
    locked_until: RwLock<Option<Instant>>,
    last_access: RwLock<Instant>,
    auto_seal_timeout: Duration,
    auto_seal_handle: tokio::sync::Mutex<Option<JoinHandle<()>>>,
    keychain_db_path: Option<std::path::PathBuf>,
    pending_shares: RwLock<Vec<ShamirShare>>,
    lifecycle_handle: tokio::sync::Mutex<Option<JoinHandle<()>>>,
    last_lifecycle_run: RwLock<Option<chrono::DateTime<chrono::Utc>>>,
}

fn map_crypto_err(e: VaultCryptoError) -> HxError {
    match e {
        VaultCryptoError::Sealed => HxError::VaultSealed,
        other => HxError::Keychain(other.to_string()),
    }
}

impl KeychainEngine {
    pub async fn new(
        store: Arc<dyn KeychainStore>,
        cred_store: Arc<CredentialStore>,
        auto_seal_timeout: Option<Duration>,
        keychain_db_path: Option<std::path::PathBuf>,
    ) -> MvResult<Self> {
        clear_runtime_root_key();

        // Disable core dumps to prevent leaking key material
        #[cfg(unix)]
        unsafe {
            let rlim = libc::rlimit {
                rlim_cur: 0,
                rlim_max: 0,
            };
            libc::setrlimit(libc::RLIMIT_CORE, &rlim);
        }

        Ok(Self {
            store,
            crypto: RwLock::new(VaultCrypto::new()),
            cred_store,
            breach_detector: BreachDetector::default(),
            failed_attempts: AtomicU32::new(0),
            locked_until: RwLock::new(None),
            last_access: RwLock::new(Instant::now()),
            auto_seal_timeout: auto_seal_timeout.unwrap_or(Duration::from_secs(900)),
            auto_seal_handle: tokio::sync::Mutex::new(None),
            keychain_db_path,
            pending_shares: RwLock::new(Vec::new()),
            lifecycle_handle: tokio::sync::Mutex::new(None),
            last_lifecycle_run: RwLock::new(None),
        })
    }

    fn touch_last_access(&self) {
        if let Ok(mut last) = self.last_access.try_write() {
            *last = Instant::now();
        }
    }

    fn require_hardware_mode() -> bool {
        std::env::var("HELIX_REQUIRE_HARDWARE")
            .map(|value| value.eq_ignore_ascii_case("true") || value == "1")
            .unwrap_or(false)
    }

    pub fn os_secure_storage_available(&self) -> bool {
        self.cred_store.status().iter().any(|backend| {
            backend.source == hx_core::credentials::SecretSource::OsKeyring && backend.available
        })
    }

    pub fn degraded_security_mode(&self) -> bool {
        runtime_is_degraded_security()
    }

    async fn refresh_runtime_storage_key(&self, degraded_security: bool) -> MvResult<()> {
        let root = {
            let crypto = self.crypto.read().await;
            crypto.extract_master_key().map_err(map_crypto_err)?
        };
        set_runtime_root_key(*root, degraded_security);
        Ok(())
    }

    pub async fn derive_namespace_kek(&self, namespace: &str) -> MvResult<[u8; 32]> {
        let crypto = self.crypto.read().await;
        let key = crypto
            .derive_namespace_kek(namespace)
            .map_err(map_crypto_err)?;
        Ok(*key)
    }

    pub async fn wrap_namespace_dek(&self, namespace: &str, dek: &[u8; 32]) -> MvResult<String> {
        let kek = self.derive_namespace_kek(namespace).await?;
        VaultCrypto::wrap_dek(&kek, dek).map_err(map_crypto_err)
    }

    pub async fn unwrap_namespace_dek(
        &self,
        namespace: &str,
        wrapped_dek: &str,
    ) -> MvResult<[u8; 32]> {
        let kek = self.derive_namespace_kek(namespace).await?;
        VaultCrypto::unwrap_dek(&kek, wrapped_dek).map_err(map_crypto_err)
    }

    async fn try_unseal_from_secure_storage(
        &self,
        subject: &str,
    ) -> MvResult<Option<MasterKeySource>> {
        #[cfg(target_os = "macos")]
        {
            if self
                .cred_store
                .get_secret_string("HELIX_SE_WRAPPED_KEY")
                .is_some()
            {
                self.unseal_from_secure_enclave().await?;
                return Ok(Some(MasterKeySource::SecureEnclave));
            }
        }

        if self
            .cred_store
            .get_secret_string("HELIX_VAULT_KEY")
            .is_some()
        {
            self.unseal_from_os_keyring(subject).await?;
            return Ok(Some(MasterKeySource::OsSecureStorage));
        }

        Ok(None)
    }

    pub async fn unseal_with_preferred_master_key(
        &self,
        passphrase: Option<&str>,
        subject: &str,
    ) -> MvResult<MasterKeySource> {
        if Self::require_hardware_mode() && !self.os_secure_storage_available() {
            return Err(HxError::Keychain(
                "HELIX_REQUIRE_HARDWARE=true but OS secure storage is unavailable".to_string(),
            ));
        }

        match self.try_unseal_from_secure_storage(subject).await {
            Ok(Some(source)) => return Ok(source),
            Ok(None) => {}
            Err(err) => {
                if passphrase.is_none() {
                    return Err(err);
                }
                tracing::warn!(error = %err, "secure storage unseal failed; falling back to passphrase");
            }
        }

        if let Some(passphrase) = passphrase {
            // Secure-storage unseal may set a temporary backoff lock when stored
            // credentials are stale. Allow one immediate passphrase attempt while
            // still enforcing permanent lock limits and normal failure accounting.
            self.unseal_internal(passphrase, subject, true).await?;
            return Ok(MasterKeySource::PassphraseArgon2id);
        }

        Err(HxError::Keychain(
            "no usable master key in OS secure storage and no passphrase provided".to_string(),
        ))
    }

    // -----------------------------------------------------------------------
    // Vault lifecycle
    // -----------------------------------------------------------------------

    pub async fn initialize_vault(
        &self,
        password: &str,
        macos_bridge: bool,
        subject: &str,
    ) -> MvResult<()> {
        // Check if already initialized
        if self.store.get_vault_meta().await?.is_some() {
            return Err(HxError::KeychainAlreadyInitialized);
        }

        // Generate salt
        let mut salt_bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut salt_bytes);
        let salt = BASE64.encode(salt_bytes);

        // Derive master key
        let config = EncryptionConfig::default();
        validate_argon2_params(&config).map_err(map_crypto_err)?;
        {
            let mut crypto = self.crypto.write().await;
            crypto
                .unseal(password, salt_bytes.as_ref(), &config)
                .map_err(map_crypto_err)?;
        }

        // Create verification blob
        let verification_blob = {
            let crypto = self.crypto.read().await;
            crypto
                .generate_verification_blob()
                .map_err(map_crypto_err)?
        };

        // Create epoch 0
        let epoch = KeyEpoch {
            epoch: 0,
            wrapped_key: None,
            created_at: Utc::now(),
            grace_expires_at: None,
            retired_at: None,
        };
        self.store.insert_key_epoch(&epoch).await?;

        // Save vault metadata
        let macos_service = if macos_bridge {
            Some("helix-keychain".to_string())
        } else {
            None
        };
        let meta = VaultMeta {
            schema_version: 1,
            master_salt: salt,
            verification_blob,
            key_epoch: 0,
            created_at: Utc::now(),
            last_rotated_at: None,
            macos_keychain_service: macos_service.clone(),
            shamir_threshold: None,
            shamir_total: None,
            shamir_last_rotated_at: None,
        };
        self.store.save_vault_meta(&meta).await?;

        // macOS Keychain bridge
        if macos_bridge {
            self.store_to_macos_keychain(password)?;
        }

        self.audit_log(KeychainAuditAction::VaultInitialized, subject, None, None)
            .await?;
        self.refresh_runtime_storage_key(true).await?;

        Ok(())
    }

    pub async fn unseal(&self, password: &str, subject: &str) -> MvResult<()> {
        self.unseal_internal(password, subject, false).await
    }

    async fn unseal_internal(
        &self,
        password: &str,
        subject: &str,
        ignore_temporary_lock: bool,
    ) -> MvResult<()> {
        if Self::require_hardware_mode() && !self.os_secure_storage_available() {
            return Err(HxError::Keychain(
                "HELIX_REQUIRE_HARDWARE=true but OS secure storage is unavailable".to_string(),
            ));
        }

        // Check lockout
        if !ignore_temporary_lock {
            let locked = self.locked_until.read().await;
            if let Some(until) = *locked {
                if Instant::now() < until {
                    return Err(HxError::Keychain(
                        "vault is temporarily locked due to failed attempts".to_string(),
                    ));
                }
            }
        }

        let attempts = self.failed_attempts.load(Ordering::SeqCst);
        if attempts >= 20 {
            return Err(HxError::Keychain(
                "vault is permanently locked after 20 failed attempts".to_string(),
            ));
        }

        let meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let salt_bytes = BASE64
            .decode(&meta.master_salt)
            .map_err(|e| HxError::Keychain(format!("invalid salt: {e}")))?;

        let config = EncryptionConfig::default();
        validate_argon2_params(&config).map_err(map_crypto_err)?;

        {
            let mut crypto = self.crypto.write().await;
            crypto
                .unseal(password, &salt_bytes, &config)
                .map_err(map_crypto_err)?;
        }

        // Verify password
        let valid = {
            let crypto = self.crypto.read().await;
            crypto
                .verify_password(&meta.verification_blob)
                .map_err(map_crypto_err)?
        };

        if !valid {
            {
                let mut crypto = self.crypto.write().await;
                crypto.seal();
            }
            let new_attempts = self.failed_attempts.fetch_add(1, Ordering::SeqCst) + 1;
            // Exponential backoff: 1s, 2s, 4s, 8s... capped at 60s
            let backoff_secs = std::cmp::min(1u64 << (new_attempts - 1), 60);
            {
                let mut locked = self.locked_until.write().await;
                *locked = Some(Instant::now() + Duration::from_secs(backoff_secs));
            }
            // Persist lockout state
            let _ = self
                .store
                .set_lockout_state(new_attempts, Some(chrono::Utc::now().to_rfc3339()))
                .await;

            self.audit_log(
                KeychainAuditAction::VaultUnlockFailed,
                subject,
                None,
                Some(serde_json::json!({"attempts": new_attempts})),
            )
            .await?;
            return Err(HxError::KeychainInvalidPassword);
        }

        // Success — reset lockout
        self.failed_attempts.store(0, Ordering::SeqCst);
        {
            let mut locked = self.locked_until.write().await;
            *locked = None;
        }
        let _ = self.store.set_lockout_state(0, None).await;

        // Load grace keys from wrapped key epochs
        let epochs = self.store.list_key_epochs().await?;
        let now = chrono::Utc::now();
        for epoch_entry in &epochs {
            if let Some(ref wrapped) = epoch_entry.wrapped_key {
                // Only load if grace period hasn't expired
                let grace_ok = epoch_entry.grace_expires_at.map_or(false, |exp| exp > now);
                if grace_ok {
                    let mut crypto = self.crypto.write().await;
                    if let Err(e) = crypto.unwrap_grace_key(epoch_entry.epoch, wrapped) {
                        tracing::warn!(epoch = epoch_entry.epoch, error = %e, "failed to unwrap grace key");
                    }
                }
            }
        }

        self.audit_log(KeychainAuditAction::VaultUnlocked, subject, None, None)
            .await?;
        self.refresh_runtime_storage_key(true).await?;

        Ok(())
    }

    pub async fn unseal_from_os_keyring(&self, subject: &str) -> MvResult<()> {
        let password = self
            .cred_store
            .get_secret_string("HELIX_VAULT_KEY")
            .ok_or_else(|| {
                HxError::KeychainNotFound("vault key in OS secure storage".into())
            })?;
        self.unseal(&password, subject).await?;
        self.refresh_runtime_storage_key(false).await?;
        Ok(())
    }

    pub async fn unseal_from_macos_keychain(&self, subject: &str) -> MvResult<()> {
        self.unseal_from_os_keyring(subject).await
    }

    pub fn store_to_macos_keychain(&self, password: &str) -> MvResult<()> {
        use hx_core::credentials::SecretSource;
        self.cred_store
            .set_in("HELIX_VAULT_KEY", password, SecretSource::OsKeyring)
            .map_err(|e| HxError::Keychain(format!("macOS Keychain: {e}")))?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Shamir VEK Splitting
    // -----------------------------------------------------------------------

    /// Enable Shamir secret sharing on the vault. Splits the current master key
    /// into `total` shares with recovery threshold `threshold`.
    /// Returns the base64-encoded shares for export/distribution.
    pub async fn enable_shamir(
        &self,
        threshold: u8,
        total: u8,
        subject: &str,
        passphrases: Option<Vec<String>>,
    ) -> MvResult<Vec<String>> {
        self.touch_last_access();

        if let Some(ref pws) = passphrases {
            if pws.len() != total as usize {
                return Err(HxError::Keychain(format!(
                    "expected {} passphrases (one per share), got {}",
                    total,
                    pws.len()
                )));
            }
        }

        let shares = {
            let crypto = self.crypto.read().await;
            crypto
                .split_master_key(threshold, total)
                .map_err(map_crypto_err)?
        };

        // Encode each share as base64, optionally encrypting with per-share passphrase
        let encoded: Vec<String> = if let Some(ref pws) = passphrases {
            shares
                .iter()
                .zip(pws.iter())
                .map(|(s, pw)| {
                    let encrypted =
                        VaultCrypto::encrypt_share(&s.data, pw).map_err(map_crypto_err)?;
                    Ok(BASE64.encode(encrypted))
                })
                .collect::<MvResult<Vec<String>>>()?
        } else {
            shares.iter().map(|s| BASE64.encode(&s.data)).collect()
        };

        // Update vault meta with Shamir params
        let mut meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;
        meta.shamir_threshold = Some(threshold);
        meta.shamir_total = Some(total);
        self.store.save_vault_meta(&meta).await?;

        self.audit_log(
            KeychainAuditAction::ShamirEnabled,
            subject,
            None,
            Some(serde_json::json!({"threshold": threshold, "total": total})),
        )
        .await?;

        Ok(encoded)
    }

    /// Re-split the current master key into new Shamir shares, invalidating the old set.
    /// Threshold and total are preserved from the initial `enable_shamir` call.
    pub async fn rotate_shamir_shares(
        &self,
        subject: &str,
        passphrases: Option<Vec<String>>,
    ) -> MvResult<Vec<String>> {
        self.touch_last_access();

        let mut meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let threshold = meta
            .shamir_threshold
            .ok_or_else(|| HxError::Keychain("Shamir is not enabled".to_string()))?;
        let total = meta
            .shamir_total
            .ok_or_else(|| HxError::Keychain("Shamir is not enabled".to_string()))?;

        if let Some(ref pws) = passphrases {
            if pws.len() != total as usize {
                return Err(HxError::Keychain(format!(
                    "expected {} passphrases, got {}",
                    total,
                    pws.len()
                )));
            }
        }

        // Re-split with a new random polynomial (same master key, new shares)
        let shares = {
            let crypto = self.crypto.read().await;
            crypto
                .split_master_key(threshold, total)
                .map_err(map_crypto_err)?
        };

        let encoded: Vec<String> = if let Some(ref pws) = passphrases {
            shares
                .iter()
                .zip(pws.iter())
                .map(|(s, pw)| {
                    let encrypted =
                        VaultCrypto::encrypt_share(&s.data, pw).map_err(map_crypto_err)?;
                    Ok(BASE64.encode(encrypted))
                })
                .collect::<MvResult<Vec<String>>>()?
        } else {
            shares.iter().map(|s| BASE64.encode(&s.data)).collect()
        };

        meta.shamir_last_rotated_at = Some(Utc::now());
        self.store.save_vault_meta(&meta).await?;

        self.audit_log(
            KeychainAuditAction::ShamirRotated,
            subject,
            None,
            Some(serde_json::json!({"threshold": threshold, "total": total})),
        )
        .await?;

        Ok(encoded)
    }

    /// Submit a single Shamir share for reconstruction.
    /// If the share was encrypted with a passphrase, provide it to decrypt.
    /// Returns the current status (how many collected, whether ready to unseal).
    pub async fn submit_shamir_share(
        &self,
        share_b64: &str,
        passphrase: Option<&str>,
    ) -> MvResult<ShamirStatus> {
        let raw = BASE64
            .decode(share_b64)
            .map_err(|e| HxError::Keychain(format!("invalid base64 share: {e}")))?;
        let data = if let Some(pw) = passphrase {
            VaultCrypto::decrypt_share(&raw, pw).map_err(map_crypto_err)?
        } else {
            raw
        };

        let meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let threshold = meta
            .shamir_threshold
            .ok_or_else(|| HxError::Keychain("shamir not enabled on this vault".to_string()))?;
        let total = meta.shamir_total.unwrap_or(0);

        let index = {
            let shares = self.pending_shares.read().await;
            (shares.len() + 1) as u8
        };

        let share = ShamirShare { index, data };
        {
            let mut shares = self.pending_shares.write().await;
            shares.push(share);
        }

        let collected = {
            let shares = self.pending_shares.read().await;
            shares.len() as u8
        };

        Ok(ShamirStatus {
            shares_collected: collected,
            threshold,
            total,
            ready: collected >= threshold,
        })
    }

    /// Attempt to unseal the vault using the collected Shamir shares.
    pub async fn unseal_from_shares(&self, subject: &str) -> MvResult<()> {
        let meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let threshold = meta
            .shamir_threshold
            .ok_or_else(|| HxError::Keychain("shamir not enabled on this vault".to_string()))?;

        let shares: Vec<ShamirShare> = {
            let pending = self.pending_shares.read().await;
            pending.clone()
        };

        if (shares.len() as u8) < threshold {
            return Err(HxError::Keychain(format!(
                "need at least {} shares, have {}",
                threshold,
                shares.len()
            )));
        }

        let key = VaultCrypto::recover_from_shares(&shares, threshold).map_err(map_crypto_err)?;

        // Inject the recovered key
        {
            let mut crypto = self.crypto.write().await;
            crypto.set_master_key(key);
        }

        // Verify the key is correct
        let valid = {
            let crypto = self.crypto.read().await;
            crypto
                .verify_password(&meta.verification_blob)
                .map_err(map_crypto_err)?
        };

        if !valid {
            // Key is wrong — seal and clear pending shares
            {
                let mut crypto = self.crypto.write().await;
                crypto.seal();
            }
            {
                let mut pending = self.pending_shares.write().await;
                pending.clear();
            }
            return Err(HxError::Keychain(
                "shamir reconstruction produced invalid key".to_string(),
            ));
        }

        // Clear pending shares
        {
            let mut pending = self.pending_shares.write().await;
            pending.clear();
        }

        self.refresh_runtime_storage_key(true).await?;

        self.audit_log(KeychainAuditAction::ShamirUnseal, subject, None, None)
            .await?;

        Ok(())
    }

    /// Get Shamir status: how many shares collected, threshold, etc.
    pub async fn shamir_status(&self) -> MvResult<Option<ShamirStatus>> {
        let meta = self.store.get_vault_meta().await?;
        let meta = match meta {
            Some(m) => m,
            None => return Ok(None),
        };

        let threshold = match meta.shamir_threshold {
            Some(t) => t,
            None => return Ok(None),
        };
        let total = meta.shamir_total.unwrap_or(0);

        let collected = {
            let shares = self.pending_shares.read().await;
            shares.len() as u8
        };

        Ok(Some(ShamirStatus {
            shares_collected: collected,
            threshold,
            total,
            ready: collected >= threshold,
        }))
    }

    pub async fn seal(&self, subject: &str) -> MvResult<()> {
        // Abort auto-seal task if running
        {
            let mut handle = self.auto_seal_handle.lock().await;
            if let Some(h) = handle.take() {
                h.abort();
            }
        }
        {
            let mut crypto = self.crypto.write().await;
            crypto.seal();
        }
        clear_runtime_root_key();
        self.audit_log(KeychainAuditAction::VaultLocked, subject, None, None)
            .await?;
        Ok(())
    }

    pub async fn vault_status(&self) -> MvResult<(VaultState, Option<VaultMeta>)> {
        let meta = self.store.get_vault_meta().await?;
        let crypto = self.crypto.read().await;
        let state = match &meta {
            None => VaultState::Uninitialized,
            Some(_) if crypto.is_unsealed() => VaultState::Unsealed,
            Some(_) => VaultState::Sealed,
        };
        Ok((state, meta))
    }

    /// Start the auto-seal background task. Checks every 30s and seals on idle.
    pub async fn start_auto_seal(self: &Arc<Self>) {
        let mut handle = self.auto_seal_handle.lock().await;
        if let Some(h) = handle.take() {
            h.abort();
        }
        let engine = Arc::clone(self);
        *handle = Some(tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(30)).await;
                let elapsed = {
                    let last = engine.last_access.read().await;
                    last.elapsed()
                };
                if elapsed >= engine.auto_seal_timeout {
                    tracing::info!("auto-sealing vault after idle timeout");
                    let _ = engine.seal("system").await;
                    return;
                }
            }
        }));
    }

    /// Returns seconds until auto-seal, or None if not applicable.
    pub async fn auto_seal_remaining(&self) -> Option<u64> {
        let crypto = self.crypto.read().await;
        if !crypto.is_unsealed() {
            return None;
        }
        let last = self.last_access.read().await;
        let elapsed = last.elapsed();
        if elapsed >= self.auto_seal_timeout {
            Some(0)
        } else {
            Some((self.auto_seal_timeout - elapsed).as_secs())
        }
    }

    // -----------------------------------------------------------------------
    // Key rotation
    // -----------------------------------------------------------------------

    pub async fn rotate_master_key(
        &self,
        new_password: &str,
        grace_period_hours: u32,
        subject: &str,
    ) -> MvResult<()> {
        self.touch_last_access();

        let meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let old_epoch = meta.key_epoch;
        let new_epoch = old_epoch + 1;

        // Extract old master key before re-keying
        let old_master = {
            let crypto = self.crypto.read().await;
            crypto.extract_master_key().map_err(map_crypto_err)?
        };

        // Generate new salt
        let mut new_salt_bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut new_salt_bytes);
        let new_salt = BASE64.encode(new_salt_bytes);

        // Derive new master key
        {
            let mut crypto = self.crypto.write().await;
            let config = EncryptionConfig::default();
            *crypto = VaultCrypto::new();
            crypto
                .unseal(new_password, &new_salt_bytes, &config)
                .map_err(map_crypto_err)?;

            // Add old key as grace key for re-encryption
            crypto.add_grace_key(old_epoch, old_master.clone());
        }

        // Wrap old master key with new master key for storage
        let wrapped_old_key = {
            let crypto = self.crypto.read().await;
            let new_master = crypto.extract_master_key().map_err(map_crypto_err)?;
            let encrypted = VaultCrypto::aes_gcm_encrypt_pub(&*new_master, &*old_master)
                .map_err(map_crypto_err)?;
            BASE64.encode(encrypted)
        };

        // Create new verification blob
        let verification_blob = {
            let crypto = self.crypto.read().await;
            crypto
                .generate_verification_blob()
                .map_err(map_crypto_err)?
        };

        // Insert new epoch with wrapped old key
        let new_key_epoch = KeyEpoch {
            epoch: new_epoch,
            wrapped_key: Some(wrapped_old_key),
            created_at: Utc::now(),
            grace_expires_at: Some(chrono::DateTime::<Utc>::from(
                std::time::SystemTime::now()
                    + std::time::Duration::from_secs(grace_period_hours as u64 * 3600),
            )),
            retired_at: None,
        };
        self.store.insert_key_epoch(&new_key_epoch).await?;

        // Retire old epoch
        self.store.retire_key_epoch(old_epoch).await?;

        // Update vault meta
        let updated_meta = VaultMeta {
            schema_version: meta.schema_version,
            master_salt: new_salt,
            verification_blob,
            key_epoch: new_epoch,
            created_at: meta.created_at,
            last_rotated_at: Some(Utc::now()),
            macos_keychain_service: meta.macos_keychain_service,
            shamir_threshold: meta.shamir_threshold,
            shamir_total: meta.shamir_total,
            shamir_last_rotated_at: meta.shamir_last_rotated_at,
        };
        self.store.save_vault_meta(&updated_meta).await?;

        // Re-encrypt all credentials: decrypt with OLD epoch grace key, encrypt with NEW master
        self.re_encrypt_all_credentials(old_epoch, new_epoch)
            .await?;

        self.audit_log(
            KeychainAuditAction::KeyRotated,
            subject,
            None,
            Some(serde_json::json!({
                "old_epoch": old_epoch,
                "new_epoch": new_epoch,
                "grace_period_hours": grace_period_hours,
            })),
        )
        .await?;

        Ok(())
    }

    async fn re_encrypt_all_credentials(&self, old_epoch: u64, new_epoch: u64) -> MvResult<()> {
        let creds = self
            .store
            .list_credentials(None, Some(CredentialState::Active), 10000, 0)
            .await?;

        let crypto = self.crypto.read().await;
        for mut cred in creds {
            let domain = self.store.get_domain(cred.domain_id).await?;
            if let Some(domain) = domain {
                // Decrypt with old epoch grace key
                let plaintext = crypto
                    .decrypt_credential_with_epoch(
                        &cred.encrypted_value,
                        &domain.derivation_info,
                        &cred.derivation_info,
                        old_epoch,
                    )
                    .map_err(map_crypto_err)?;

                // Re-encrypt with current (new) master key
                let encrypted = crypto
                    .encrypt_credential(&*plaintext, &domain.derivation_info, &cred.derivation_info)
                    .map_err(map_crypto_err)?;

                // Re-encrypt metadata if it was encrypted with the old key
                if cred.metadata_encrypted {
                    let plain_name = crypto
                        .decrypt_metadata_with_epoch(&cred.name, old_epoch)
                        .map_err(map_crypto_err)?;
                    cred.name = crypto
                        .encrypt_metadata(&plain_name)
                        .map_err(map_crypto_err)?;
                    if let Some(ref desc) = cred.description {
                        let plain_desc = crypto
                            .decrypt_metadata_with_epoch(desc, old_epoch)
                            .map_err(map_crypto_err)?;
                        cred.description = Some(
                            crypto
                                .encrypt_metadata(&plain_desc)
                                .map_err(map_crypto_err)?,
                        );
                    }
                }

                cred.encrypted_value = encrypted;
                cred.epoch = new_epoch;
                cred.updated_at = Utc::now();
                self.store.update_credential(&cred).await?;
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Domains
    // -----------------------------------------------------------------------

    pub async fn create_domain(
        &self,
        name: &str,
        description: Option<&str>,
        subject: &str,
    ) -> MvResult<DomainKey> {
        let meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let derivation_info = format!("domain:{name}");
        let mut domain = DomainKey::new(name, &derivation_info).with_epoch(meta.key_epoch);
        if let Some(desc) = description {
            domain = domain.with_description(desc);
        }

        self.store.insert_domain(&domain).await?;
        self.audit_log(
            KeychainAuditAction::DomainCreated,
            subject,
            Some(&domain.id.to_string()),
            Some(serde_json::json!({"name": name})),
        )
        .await?;

        Ok(domain)
    }

    pub async fn list_domains(&self) -> MvResult<Vec<DomainKey>> {
        self.store.list_domains().await
    }

    pub async fn revoke_domain(&self, id: Uuid, subject: &str) -> MvResult<()> {
        self.store.revoke_domain(id).await?;
        self.audit_log(
            KeychainAuditAction::DomainRevoked,
            subject,
            Some(&id.to_string()),
            None,
        )
        .await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Credential CRUD
    // -----------------------------------------------------------------------

    pub async fn store_credential(
        &self,
        domain_id: Uuid,
        name: &str,
        kind: &str,
        value: &[u8],
        tags: Vec<String>,
        expires_at: Option<chrono::DateTime<Utc>>,
        subject: &str,
    ) -> MvResult<StoredCredential> {
        self.touch_last_access();

        let domain = self
            .store
            .get_domain(domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        if domain.revoked_at.is_some() {
            return Err(HxError::Keychain("domain is revoked".to_string()));
        }

        let cred_derivation = format!("cred:{name}:{}", Uuid::now_v7());

        // Encrypt value and metadata
        let (encrypted, encrypted_name) = {
            let crypto = self.crypto.read().await;
            let enc_val = crypto
                .encrypt_credential(value, &domain.derivation_info, &cred_derivation)
                .map_err(map_crypto_err)?;
            let enc_name = crypto.encrypt_metadata(name).map_err(map_crypto_err)?;
            (enc_val, enc_name)
        };

        let meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let mut cred = StoredCredential::new(
            domain_id,
            &encrypted_name,
            kind,
            encrypted,
            &cred_derivation,
        )
        .with_tags(tags)
        .with_epoch(meta.key_epoch);
        cred.metadata_encrypted = true;

        if let Some(exp) = expires_at {
            cred = cred.with_expires_at(exp);
        }

        self.store.insert_credential(&cred).await?;
        self.audit_log(
            KeychainAuditAction::CredentialStored,
            subject,
            Some(&cred.id.to_string()),
            Some(
                serde_json::json!({"name": name, "kind": kind, "domain_id": domain_id.to_string()}),
            ),
        )
        .await?;

        // Return with plaintext name to the caller
        cred.name = name.to_string();
        Ok(cred)
    }

    pub async fn read_credential(
        &self,
        id: Uuid,
        subject: &str,
    ) -> MvResult<(StoredCredential, Zeroizing<Vec<u8>>, Vec<BreachAlert>)> {
        self.touch_last_access();

        let mut cred = self
            .store
            .get_credential(id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;

        if cred.state == CredentialState::Destroyed {
            return Err(HxError::Keychain(
                "credential has been destroyed".to_string(),
            ));
        }

        let domain = self
            .store
            .get_domain(cred.domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        // Decrypt value and metadata
        let plaintext = {
            let crypto = self.crypto.read().await;
            let pt = crypto
                .decrypt_credential(
                    &cred.encrypted_value,
                    &domain.derivation_info,
                    &cred.derivation_info,
                )
                .map_err(map_crypto_err)?;
            if cred.metadata_encrypted {
                cred.name = crypto
                    .decrypt_metadata(&cred.name)
                    .map_err(map_crypto_err)?;
                if let Some(ref desc) = cred.description {
                    cred.description = Some(crypto.decrypt_metadata(desc).map_err(map_crypto_err)?);
                }
            }
            pt
        };

        // Touch credential
        self.store.touch_credential(id).await?;

        // Record access pattern + breach detection
        let now = Utc::now();
        let pattern = AccessPattern {
            credential_id: id,
            accessor: subject.to_string(),
            source_ip: None,
            timestamp: now,
            hour_of_day: now.hour() as u8,
            day_of_week: now.weekday().num_days_from_monday() as u8,
        };
        self.store.record_access_pattern(&pattern).await?;

        let recent_patterns = self.store.get_access_patterns(id, 100).await?;
        let detected_alerts = self.breach_detector.analyze(&recent_patterns, &pattern);
        let mut inserted_alerts = Vec::new();
        for alert in detected_alerts {
            // Deduplicate: skip if the same alert type was recorded in the last 60 seconds.
            let dup = self
                .store
                .has_recent_breach_alert(alert.credential_id, alert.alert_type.as_str(), 60)
                .await?;
            if dup {
                continue;
            }
            self.store.insert_breach_alert(&alert).await?;
            self.audit_log(
                KeychainAuditAction::BreachDetected,
                subject,
                Some(&id.to_string()),
                Some(serde_json::json!({
                    "alert_type": alert.alert_type.as_str(),
                    "severity": alert.severity.as_str(),
                })),
            )
            .await?;
            inserted_alerts.push(alert);
        }

        self.audit_log(
            KeychainAuditAction::CredentialRead,
            subject,
            Some(&id.to_string()),
            None,
        )
        .await?;

        Ok((cred, plaintext, inserted_alerts))
    }

    pub async fn update_credential_value(
        &self,
        id: Uuid,
        new_value: &[u8],
        subject: &str,
    ) -> MvResult<StoredCredential> {
        self.touch_last_access();

        let mut cred = self
            .store
            .get_credential(id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;

        let domain = self
            .store
            .get_domain(cred.domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        let encrypted = {
            let crypto = self.crypto.read().await;
            crypto
                .encrypt_credential(new_value, &domain.derivation_info, &cred.derivation_info)
                .map_err(map_crypto_err)?
        };

        cred.encrypted_value = encrypted;
        cred.version += 1;
        cred.updated_at = Utc::now();
        self.store.update_credential(&cred).await?;

        self.audit_log(
            KeychainAuditAction::CredentialUpdated,
            subject,
            Some(&id.to_string()),
            Some(serde_json::json!({"new_version": cred.version})),
        )
        .await?;

        // Return with decrypted metadata
        if cred.metadata_encrypted {
            let crypto = self.crypto.read().await;
            cred.name = crypto
                .decrypt_metadata(&cred.name)
                .map_err(map_crypto_err)?;
            if let Some(ref desc) = cred.description {
                cred.description = Some(crypto.decrypt_metadata(desc).map_err(map_crypto_err)?);
            }
        }

        Ok(cred)
    }

    pub async fn update_credential_metadata(
        &self,
        id: Uuid,
        description: Option<String>,
        tags: Option<Vec<String>>,
        metadata: Option<std::collections::HashMap<String, serde_json::Value>>,
        expires_at: Option<chrono::DateTime<Utc>>,
        subject: &str,
    ) -> MvResult<StoredCredential> {
        self.touch_last_access();

        let mut cred = self
            .store
            .get_credential(id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;

        if let Some(desc) = description {
            if cred.metadata_encrypted {
                let crypto = self.crypto.read().await;
                cred.description = Some(crypto.encrypt_metadata(&desc).map_err(map_crypto_err)?);
            } else {
                cred.description = Some(desc);
            }
        }
        if let Some(tags) = tags {
            cred.tags = tags;
        }
        if let Some(metadata) = metadata {
            cred.metadata = metadata;
        }
        if let Some(expires_at) = expires_at {
            cred.expires_at = Some(expires_at);
        }

        cred.updated_at = Utc::now();
        cred.version += 1;
        self.store.update_credential(&cred).await?;

        self.audit_log(
            KeychainAuditAction::CredentialUpdated,
            subject,
            Some(&id.to_string()),
            Some(serde_json::json!({"new_version": cred.version})),
        )
        .await?;

        // Return with decrypted metadata
        if cred.metadata_encrypted {
            let crypto = self.crypto.read().await;
            cred.name = crypto
                .decrypt_metadata(&cred.name)
                .map_err(map_crypto_err)?;
            if let Some(ref desc) = cred.description {
                cred.description = Some(crypto.decrypt_metadata(desc).map_err(map_crypto_err)?);
            }
        }

        Ok(cred)
    }

    pub async fn archive_credential(&self, id: Uuid, subject: &str) -> MvResult<()> {
        self.touch_last_access();

        let mut cred = self
            .store
            .get_credential(id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;

        cred.state = CredentialState::Archived;
        cred.archived_at = Some(Utc::now());
        cred.updated_at = Utc::now();
        self.store.update_credential(&cred).await?;

        self.audit_log(
            KeychainAuditAction::CredentialArchived,
            subject,
            Some(&id.to_string()),
            None,
        )
        .await?;

        Ok(())
    }

    pub async fn destroy_credential(&self, id: Uuid, subject: &str) -> MvResult<()> {
        self.touch_last_access();

        // Revoke all delegations first
        self.store.revoke_delegations_for_credential(id).await?;

        // Shred the credential
        self.store.shred_credential(id).await?;

        self.audit_log(
            KeychainAuditAction::CredentialDestroyed,
            subject,
            Some(&id.to_string()),
            None,
        )
        .await?;

        Ok(())
    }

    pub async fn list_credentials(
        &self,
        domain_id: Option<Uuid>,
        state: Option<CredentialState>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<StoredCredential>> {
        let mut creds = self
            .store
            .list_credentials(domain_id, state, limit, offset)
            .await?;

        // Decrypt metadata for encrypted credentials (best-effort if sealed)
        if let Ok(crypto) = self.crypto.try_read() {
            if crypto.is_unsealed() {
                for cred in &mut creds {
                    if cred.metadata_encrypted {
                        if let Ok(name) = crypto.decrypt_metadata(&cred.name) {
                            cred.name = name;
                        }
                        if let Some(ref desc) = cred.description {
                            if let Ok(d) = crypto.decrypt_metadata(desc) {
                                cred.description = Some(d);
                            }
                        }
                    }
                }
            }
        }

        Ok(creds)
    }

    // -----------------------------------------------------------------------
    // Domain ACLs
    // -----------------------------------------------------------------------

    pub async fn set_domain_acl(
        &self,
        domain_id: Uuid,
        subject: &str,
        can_read: bool,
        can_write: bool,
        can_admin: bool,
        expires_at: Option<DateTime<Utc>>,
        caller: &str,
    ) -> MvResult<DomainAcl> {
        // Verify domain exists
        self.store
            .get_domain(domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        let acl = DomainAcl {
            id: Uuid::now_v7(),
            domain_id,
            subject: subject.to_string(),
            can_read,
            can_write,
            can_admin,
            created_at: Utc::now(),
            expires_at,
        };
        self.store.insert_acl(&acl).await?;
        self.audit_log(
            KeychainAuditAction::DomainCreated,
            caller,
            Some(&domain_id.to_string()),
            Some(serde_json::json!({
                "acl_subject": subject,
                "can_read": can_read,
                "can_write": can_write,
                "can_admin": can_admin,
            })),
        )
        .await?;
        Ok(acl)
    }

    pub async fn remove_domain_acl(&self, acl_id: Uuid, caller: &str) -> MvResult<()> {
        self.store.delete_acl(acl_id).await?;
        self.audit_log(
            KeychainAuditAction::DomainRevoked,
            caller,
            Some(&acl_id.to_string()),
            Some(serde_json::json!({"acl_deleted": true})),
        )
        .await?;
        Ok(())
    }

    pub async fn list_domain_acls(&self, domain_id: Uuid) -> MvResult<Vec<DomainAcl>> {
        self.store.get_acls_for_domain(domain_id).await
    }

    /// Check if `subject` has the requested permission on `domain_id`.
    /// Returns `Ok(())` if allowed, `Err` if denied.
    /// If no ACL exists for the subject, returns `Err` (caller must be Admin).
    pub async fn check_domain_access(
        &self,
        domain_id: Uuid,
        subject: &str,
        need_read: bool,
        need_write: bool,
    ) -> MvResult<()> {
        let acl = self.store.get_acl_for_subject(domain_id, subject).await?;
        let acl = acl.ok_or_else(|| {
            HxError::Keychain(format!("no ACL for subject '{subject}' on domain"))
        })?;

        // Check expiry
        if let Some(expires_at) = acl.expires_at {
            if expires_at <= Utc::now() {
                return Err(HxError::Keychain("ACL has expired".to_string()));
            }
        }

        if need_read && !acl.can_read {
            return Err(HxError::Keychain("ACL denies read access".to_string()));
        }
        if need_write && !acl.can_write {
            return Err(HxError::Keychain("ACL denies write access".to_string()));
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Lifecycle engine
    // -----------------------------------------------------------------------

    pub async fn run_lifecycle_transitions(&self) -> MvResult<u32> {
        let now = Utc::now();
        let mut transitioned = 0u32;

        // Check active credentials approaching expiry (within 7 days)
        let actives = self
            .store
            .list_credentials(None, Some(CredentialState::Active), 10000, 0)
            .await?;

        for mut cred in actives {
            if let Some(expires_at) = cred.expires_at {
                if expires_at <= now {
                    cred.state = CredentialState::Expired;
                    cred.updated_at = now;
                    self.store.update_credential(&cred).await?;
                    transitioned += 1;
                } else if expires_at <= now + chrono::Duration::days(7) {
                    cred.state = CredentialState::Expiring;
                    cred.updated_at = now;
                    self.store.update_credential(&cred).await?;
                    transitioned += 1;
                }
            }
        }

        // Check expiring credentials that have now expired
        let expiring = self
            .store
            .list_credentials(None, Some(CredentialState::Expiring), 10000, 0)
            .await?;

        for mut cred in expiring {
            if let Some(expires_at) = cred.expires_at {
                if expires_at <= now {
                    cred.state = CredentialState::Expired;
                    cred.updated_at = now;
                    self.store.update_credential(&cred).await?;
                    transitioned += 1;
                }
            }
        }

        if transitioned > 0 {
            self.audit_log(
                KeychainAuditAction::LifecycleTransition,
                "system",
                None,
                Some(serde_json::json!({"transitioned": transitioned})),
            )
            .await?;
        }

        Ok(transitioned)
    }

    // -----------------------------------------------------------------------
    // Delegations
    // -----------------------------------------------------------------------

    pub async fn create_delegation(
        &self,
        credential_id: Uuid,
        delegatee: &str,
        permissions: DelegationPermissions,
        expires_at: Option<chrono::DateTime<Utc>>,
        max_depth: u32,
        subject: &str,
    ) -> MvResult<Delegation> {
        self.touch_last_access();

        let cred = self
            .store
            .get_credential(credential_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;

        let domain = self
            .store
            .get_domain(cred.domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        let perms_str = format!(
            "r:{},u:{},d:{}",
            permissions.can_read, permissions.can_use, permissions.can_delegate
        );
        let chain_hash = {
            let crypto = self.crypto.read().await;
            crypto
                .compute_chain_hash(
                    &domain.derivation_info,
                    None,
                    &credential_id.to_string(),
                    delegatee,
                    &perms_str,
                    expires_at.as_ref().map(|dt| dt.to_rfc3339()).as_deref(),
                    0,
                    max_depth,
                )
                .map_err(map_crypto_err)?
        };

        let delegation = Delegation {
            id: Uuid::now_v7(),
            credential_id,
            delegatee: delegatee.to_string(),
            parent_id: None,
            permissions,
            chain_hash,
            created_at: Utc::now(),
            expires_at,
            revoked_at: None,
            max_depth,
            depth: 0,
        };

        self.store.insert_delegation(&delegation).await?;
        self.audit_log(
            KeychainAuditAction::DelegationCreated,
            subject,
            Some(&delegation.id.to_string()),
            Some(serde_json::json!({
                "credential_id": credential_id.to_string(),
                "delegatee": delegatee,
            })),
        )
        .await?;

        Ok(delegation)
    }

    pub async fn sub_delegate(
        &self,
        parent_id: Uuid,
        delegatee: &str,
        permissions: DelegationPermissions,
        expires_at: Option<chrono::DateTime<Utc>>,
        subject: &str,
    ) -> MvResult<Delegation> {
        self.touch_last_access();

        let parent = self
            .store
            .get_delegation(parent_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("parent delegation".into()))?;

        if parent.revoked_at.is_some() {
            return Err(HxError::Keychain(
                "parent delegation is revoked".to_string(),
            ));
        }
        if !parent.permissions.can_delegate {
            return Err(HxError::Keychain(
                "parent delegation does not allow sub-delegation".to_string(),
            ));
        }
        if parent.depth + 1 >= parent.max_depth {
            return Err(HxError::Keychain(
                "delegation depth limit reached".to_string(),
            ));
        }

        let cred = self
            .store
            .get_credential(parent.credential_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;
        let domain = self
            .store
            .get_domain(cred.domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        let perms_str = format!(
            "r:{},u:{},d:{}",
            permissions.can_read, permissions.can_use, permissions.can_delegate
        );
        let chain_hash = {
            let crypto = self.crypto.read().await;
            crypto
                .compute_chain_hash(
                    &domain.derivation_info,
                    Some(&parent.chain_hash),
                    &parent.credential_id.to_string(),
                    delegatee,
                    &perms_str,
                    expires_at.as_ref().map(|dt| dt.to_rfc3339()).as_deref(),
                    parent.depth + 1,
                    parent.max_depth,
                )
                .map_err(map_crypto_err)?
        };

        let delegation = Delegation {
            id: Uuid::now_v7(),
            credential_id: parent.credential_id,
            delegatee: delegatee.to_string(),
            parent_id: Some(parent_id),
            permissions,
            chain_hash,
            created_at: Utc::now(),
            expires_at,
            revoked_at: None,
            max_depth: parent.max_depth,
            depth: parent.depth + 1,
        };

        self.store.insert_delegation(&delegation).await?;
        self.audit_log(
            KeychainAuditAction::DelegationCreated,
            subject,
            Some(&delegation.id.to_string()),
            Some(serde_json::json!({
                "parent_id": parent_id.to_string(),
                "delegatee": delegatee,
                "depth": delegation.depth,
            })),
        )
        .await?;

        Ok(delegation)
    }

    pub async fn revoke_delegation(&self, id: Uuid, subject: &str) -> MvResult<()> {
        self.store.revoke_delegation(id).await?;
        self.audit_log(
            KeychainAuditAction::DelegationRevoked,
            subject,
            Some(&id.to_string()),
            None,
        )
        .await?;
        Ok(())
    }

    pub async fn list_delegations(&self, credential_id: Uuid) -> MvResult<Vec<Delegation>> {
        self.store.list_delegations(credential_id).await
    }

    pub async fn read_credential_via_delegation(
        &self,
        delegation_id: Uuid,
        subject: &str,
    ) -> MvResult<(StoredCredential, Zeroizing<Vec<u8>>, Vec<BreachAlert>)> {
        self.touch_last_access();

        let delegation = self
            .store
            .get_delegation(delegation_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("delegation".into()))?;

        if delegation.revoked_at.is_some() {
            return Err(HxError::Keychain("delegation is revoked".to_string()));
        }
        if let Some(expires_at) = delegation.expires_at {
            if expires_at <= Utc::now() {
                return Err(HxError::Keychain("delegation has expired".to_string()));
            }
        }
        if !delegation.permissions.can_read {
            return Err(HxError::Keychain(
                "delegation does not grant read access".to_string(),
            ));
        }

        self.read_credential(delegation.credential_id, subject)
            .await
    }

    // -----------------------------------------------------------------------
    // Zero-knowledge proofs
    // -----------------------------------------------------------------------

    pub async fn generate_proof(
        &self,
        credential_id: Uuid,
        challenge_nonce: &str,
        subject: &str,
    ) -> MvResult<AccessProof> {
        self.touch_last_access();

        let cred = self
            .store
            .get_credential(credential_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;

        let domain = self
            .store
            .get_domain(cred.domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        // Decrypt the credential to get the raw value for proof generation
        let plaintext = {
            let crypto = self.crypto.read().await;
            crypto
                .decrypt_credential(
                    &cred.encrypted_value,
                    &domain.derivation_info,
                    &cred.derivation_info,
                )
                .map_err(map_crypto_err)?
        };

        let proof = {
            let crypto = self.crypto.read().await;
            crypto
                .generate_zk_proof(&*plaintext, challenge_nonce)
                .map_err(map_crypto_err)?
        };

        let now = Utc::now();
        let zk_proof = AccessProof {
            credential_id,
            challenge_nonce: challenge_nonce.to_string(),
            proof,
            generated_at: now,
            expires_at: now + chrono::Duration::minutes(5),
        };

        self.audit_log(
            KeychainAuditAction::ProofGenerated,
            subject,
            Some(&credential_id.to_string()),
            None,
        )
        .await?;

        Ok(zk_proof)
    }

    pub async fn verify_proof(&self, proof: &AccessProof, subject: &str) -> MvResult<bool> {
        self.touch_last_access();

        if proof.expires_at <= Utc::now() {
            return Ok(false);
        }

        let cred = self
            .store
            .get_credential(proof.credential_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("credential".into()))?;

        let domain = self
            .store
            .get_domain(cred.domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        let plaintext = {
            let crypto = self.crypto.read().await;
            crypto
                .decrypt_credential(
                    &cred.encrypted_value,
                    &domain.derivation_info,
                    &cred.derivation_info,
                )
                .map_err(map_crypto_err)?
        };

        let valid = {
            let crypto = self.crypto.read().await;
            crypto
                .verify_zk_proof(&*plaintext, &proof.challenge_nonce, &proof.proof)
                .map_err(map_crypto_err)?
        };

        self.audit_log(
            KeychainAuditAction::ProofVerified,
            subject,
            Some(&proof.credential_id.to_string()),
            Some(serde_json::json!({"valid": valid})),
        )
        .await?;

        Ok(valid)
    }

    // -----------------------------------------------------------------------
    // Audit + Breach
    // -----------------------------------------------------------------------

    pub async fn list_audit_trail(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<KeychainAuditEntry>> {
        self.store.list_audit_entries(limit, offset).await
    }

    pub async fn verify_audit_integrity(&self) -> MvResult<AuditVerificationResult> {
        use hx_core::model::keychain::AuditVerificationResult;

        // First check the hash chain
        let chain_ok = self.store.verify_audit_chain().await?;
        if !chain_ok {
            return Ok(AuditVerificationResult::Failed);
        }

        // Then verify HMAC signatures on entries that have them
        let crypto = self.crypto.read().await;
        if !crypto.is_unsealed() {
            return Ok(AuditVerificationResult::ChainOnlyValid);
        }

        let entries = self.store.list_audit_entries(100000, 0).await?;
        for entry in &entries {
            if let Some(ref sig) = entry.signature {
                let valid = crypto
                    .verify_audit_signature(
                        entry.sequence,
                        entry.action.as_str(),
                        &entry.subject,
                        entry.resource_id.as_deref(),
                        &entry.entry_hash,
                        &entry.timestamp.to_rfc3339(),
                        sig,
                    )
                    .map_err(map_crypto_err)?;
                if !valid {
                    return Ok(AuditVerificationResult::Failed);
                }
            }
        }

        Ok(AuditVerificationResult::FullyVerified)
    }

    pub async fn backup_vault(&self, password: &str) -> MvResult<Vec<u8>> {
        let db_path = self
            .keychain_db_path
            .as_ref()
            .ok_or_else(|| HxError::Keychain("keychain db path not set".to_string()))?;
        crate::backup::export_vault(db_path, password)
            .map_err(|e| HxError::Keychain(format!("backup failed: {e}")))
    }

    pub async fn restore_vault(&self, data: &[u8], password: &str) -> MvResult<()> {
        let db_path = self
            .keychain_db_path
            .as_ref()
            .ok_or_else(|| HxError::Keychain("keychain db path not set".to_string()))?;
        crate::backup::import_vault(data, password, db_path)
            .map_err(|e| HxError::Keychain(format!("restore failed: {e}")))
    }

    #[cfg(target_os = "macos")]
    pub async fn unseal_from_secure_enclave(&self) -> MvResult<()> {
        let wrapped = self
            .cred_store
            .get_secret_string("HELIX_SE_WRAPPED_KEY")
            .ok_or_else(|| HxError::KeychainNotFound("SE wrapped key".into()))?;
        let key_bytes = crate::secure_enclave::unwrap_key_from_se(&wrapped)
            .map_err(|e| HxError::Keychain(format!("Secure Enclave: {e}")))?;
        if key_bytes.len() != 32 {
            return Err(HxError::Keychain(
                "invalid key length from Secure Enclave".to_string(),
            ));
        }
        let mut key = zeroize::Zeroizing::new([0u8; 32]);
        key.copy_from_slice(&key_bytes);
        {
            let mut crypto = self.crypto.write().await;
            crypto.set_master_key(key);
        }
        self.refresh_runtime_storage_key(false).await?;
        self.audit_log(
            KeychainAuditAction::VaultUnlocked,
            "secure_enclave",
            None,
            None,
        )
        .await?;
        Ok(())
    }

    pub async fn list_breach_alerts(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<BreachAlert>> {
        self.store.list_breach_alerts(limit, offset).await
    }

    pub async fn acknowledge_alert(&self, id: Uuid) -> MvResult<()> {
        self.store.acknowledge_breach_alert(id).await
    }

    // -----------------------------------------------------------------------
    // Private: audit logging
    // -----------------------------------------------------------------------

    async fn audit_log(
        &self,
        action: KeychainAuditAction,
        subject: &str,
        resource_id: Option<&str>,
        details: Option<serde_json::Value>,
    ) -> MvResult<()> {
        let previous = self.store.get_latest_audit_entry().await?;
        let prev_hash = previous.as_ref().map(|e| e.entry_hash.as_str());
        let sequence = previous.as_ref().map(|e| e.sequence + 1).unwrap_or(1);
        let timestamp = Utc::now();

        let entry_hash = VaultCrypto::compute_audit_hash(
            prev_hash,
            sequence,
            action.as_str(),
            subject,
            resource_id,
            &timestamp.to_rfc3339(),
        );

        // Sign the entry if the vault is unsealed
        let signature = {
            let crypto = self.crypto.read().await;
            if crypto.is_unsealed() {
                crypto
                    .sign_audit_entry(
                        sequence,
                        action.as_str(),
                        subject,
                        resource_id,
                        &entry_hash,
                        &timestamp.to_rfc3339(),
                    )
                    .ok()
            } else {
                None
            }
        };

        let entry = KeychainAuditEntry {
            id: Uuid::now_v7(),
            sequence,
            action,
            subject: subject.to_string(),
            resource_id: resource_id.map(|s| s.to_string()),
            details,
            entry_hash,
            previous_hash: prev_hash.map(|s| s.to_string()),
            timestamp,
            source_ip: None,
            signature,
        };

        self.store.append_audit_entry(&entry).await?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Lifecycle Automation
    // -----------------------------------------------------------------------

    /// Start the background lifecycle scheduler. Periodically runs credential
    /// lifecycle transitions and (if enabled) auto-rotates the master key.
    pub async fn start_lifecycle_scheduler(self: &Arc<Self>, config: &KeychainConfig) {
        let mut handle = self.lifecycle_handle.lock().await;
        if let Some(h) = handle.take() {
            h.abort();
        }

        let engine = Arc::clone(self);
        let interval_secs = config.lifecycle_check_interval_secs;
        let auto_rotate = config.auto_rotate_enabled;
        let rotate_days = config.auto_rotate_interval_days;
        let grace_hours = config.auto_rotate_grace_hours;

        *handle = Some(tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
            // Skip the initial immediate tick
            interval.tick().await;

            loop {
                interval.tick().await;

                // Only run if vault is unsealed
                {
                    let crypto = engine.crypto.read().await;
                    if !crypto.is_unsealed() {
                        continue;
                    }
                }

                // Run lifecycle transitions
                match engine.run_lifecycle_transitions().await {
                    Ok(count) => {
                        if count > 0 {
                            tracing::info!(
                                transitioned = count,
                                "lifecycle scheduler: transitions applied"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "lifecycle scheduler: transition check failed");
                    }
                }

                // Record last run time
                {
                    let mut last = engine.last_lifecycle_run.write().await;
                    *last = Some(Utc::now());
                }

                // Auto-rotation check
                if auto_rotate {
                    if let Err(e) = engine.maybe_auto_rotate(rotate_days, grace_hours).await {
                        tracing::warn!(error = %e, "lifecycle scheduler: auto-rotation check failed");
                    }
                }
            }
        }));
    }

    /// Check if auto-rotation is due and perform it if so.
    async fn maybe_auto_rotate(&self, rotate_days: u32, grace_hours: u32) -> MvResult<()> {
        let meta = match self.store.get_vault_meta().await? {
            Some(m) => m,
            None => return Ok(()),
        };

        let last_rotated = meta.last_rotated_at.unwrap_or(meta.created_at);
        let rotate_after = last_rotated + chrono::Duration::days(rotate_days as i64);

        if Utc::now() < rotate_after {
            return Ok(()); // Not yet due
        }

        tracing::info!("auto-rotation triggered: last rotated at {last_rotated}");
        self.auto_rotate(grace_hours).await
    }

    /// Perform automatic key rotation without a password.
    /// Generates a random master key and re-encrypts all credentials.
    async fn auto_rotate(&self, grace_hours: u32) -> MvResult<()> {
        let meta = self
            .store
            .get_vault_meta()
            .await?
            .ok_or_else(|| HxError::KeychainNotInitialized)?;

        let old_epoch = meta.key_epoch;
        let new_epoch = old_epoch + 1;

        // Extract old master key
        let old_master = {
            let crypto = self.crypto.read().await;
            crypto.extract_master_key().map_err(map_crypto_err)?
        };

        // Generate random new master key
        let mut new_key = Zeroizing::new([0u8; 32]);
        rand::rngs::OsRng.fill_bytes(new_key.as_mut());

        // Generate new salt (for metadata consistency)
        let mut new_salt_bytes = [0u8; 16];
        rand::rngs::OsRng.fill_bytes(&mut new_salt_bytes);
        let new_salt = BASE64.encode(new_salt_bytes);

        // Set new master key
        {
            let mut crypto = self.crypto.write().await;
            crypto.set_master_key(Zeroizing::new(*new_key));
            crypto.add_grace_key(old_epoch, old_master.clone());
        }

        // Wrap old master key with new master key
        let wrapped_old_key = {
            let encrypted = VaultCrypto::aes_gcm_encrypt_pub(&*new_key, &*old_master)
                .map_err(map_crypto_err)?;
            BASE64.encode(encrypted)
        };

        // Create new verification blob
        let verification_blob = {
            let crypto = self.crypto.read().await;
            crypto
                .generate_verification_blob()
                .map_err(map_crypto_err)?
        };

        // Insert new epoch
        let new_key_epoch = KeyEpoch {
            epoch: new_epoch,
            wrapped_key: Some(wrapped_old_key),
            created_at: Utc::now(),
            grace_expires_at: Some(chrono::DateTime::<Utc>::from(
                std::time::SystemTime::now()
                    + std::time::Duration::from_secs(grace_hours as u64 * 3600),
            )),
            retired_at: None,
        };
        self.store.insert_key_epoch(&new_key_epoch).await?;
        self.store.retire_key_epoch(old_epoch).await?;

        // Update vault meta
        let updated_meta = VaultMeta {
            schema_version: meta.schema_version,
            master_salt: new_salt,
            verification_blob,
            key_epoch: new_epoch,
            created_at: meta.created_at,
            last_rotated_at: Some(Utc::now()),
            macos_keychain_service: meta.macos_keychain_service.clone(),
            shamir_threshold: meta.shamir_threshold,
            shamir_total: meta.shamir_total,
            shamir_last_rotated_at: meta.shamir_last_rotated_at,
        };
        self.store.save_vault_meta(&updated_meta).await?;

        // Re-encrypt all credentials
        self.re_encrypt_all_credentials(old_epoch, new_epoch)
            .await?;

        // If Shamir is enabled, re-split the new key and store share 1 in OS Keychain
        if let (Some(threshold), Some(total)) = (meta.shamir_threshold, meta.shamir_total) {
            let shares = {
                let crypto = self.crypto.read().await;
                crypto
                    .split_master_key(threshold, total)
                    .map_err(map_crypto_err)?
            };
            // Store first share in OS Keychain for convenience
            if let Some(first_share) = shares.first() {
                let encoded = BASE64.encode(&first_share.data);
                let _ = self.cred_store.set_in(
                    "HELIX_SHAMIR_SHARE_1",
                    &encoded,
                    hx_core::credentials::SecretSource::OsKeyring,
                );
            }
        }

        // Store new key in macOS Keychain if bridge is configured
        if meta.macos_keychain_service.is_some() {
            let key_b64 = BASE64.encode(&*new_key);
            let _ = self.cred_store.set_in(
                "HELIX_VAULT_KEY",
                &key_b64,
                hx_core::credentials::SecretSource::OsKeyring,
            );
        }

        self.audit_log(
            KeychainAuditAction::KeyRotated,
            "system",
            None,
            Some(serde_json::json!({
                "old_epoch": old_epoch,
                "new_epoch": new_epoch,
                "auto": true,
                "grace_period_hours": grace_hours,
            })),
        )
        .await?;

        tracing::info!(old_epoch, new_epoch, "auto-rotation completed");
        Ok(())
    }

    /// Returns the timestamp of the last lifecycle run, if any.
    pub async fn last_lifecycle_run(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        let last = self.last_lifecycle_run.read().await;
        *last
    }

    // -----------------------------------------------------------------------
    // Agent bridge helpers (used by KeychainBackend)
    // -----------------------------------------------------------------------

    /// Synchronous check whether the vault is currently unsealed.
    /// Uses `try_read` to avoid blocking — returns `false` if the lock is contended.
    pub fn is_unsealed_sync(&self) -> bool {
        match self.crypto.try_read() {
            Ok(guard) => guard.is_unsealed(),
            Err(_) => false,
        }
    }

    /// Find a credential by name within a specific domain and decrypt it.
    pub async fn read_credential_by_name(
        &self,
        domain_id: Uuid,
        name: &str,
    ) -> MvResult<Option<(StoredCredential, Zeroizing<Vec<u8>>)>> {
        let creds = self
            .store
            .list_credentials(Some(domain_id), Some(CredentialState::Active), 1000, 0)
            .await?;

        // Decrypt metadata to match by plaintext name
        let crypto = self.crypto.read().await;
        let mut found = None;
        for c in creds {
            let plain_name = if c.metadata_encrypted {
                crypto.decrypt_metadata(&c.name).map_err(map_crypto_err)?
            } else {
                c.name.clone()
            };
            if plain_name == name {
                found = Some(c);
                break;
            }
        }

        let mut cred = match found {
            Some(c) => c,
            None => return Ok(None),
        };

        let domain = self
            .store
            .get_domain(cred.domain_id)
            .await?
            .ok_or_else(|| HxError::KeychainNotFound("domain".into()))?;

        let plaintext = crypto
            .decrypt_credential(
                &cred.encrypted_value,
                &domain.derivation_info,
                &cred.derivation_info,
            )
            .map_err(map_crypto_err)?;

        // Return with decrypted metadata
        if cred.metadata_encrypted {
            cred.name = crypto
                .decrypt_metadata(&cred.name)
                .map_err(map_crypto_err)?;
            if let Some(ref desc) = cred.description {
                cred.description = Some(crypto.decrypt_metadata(desc).map_err(map_crypto_err)?);
            }
        }
        drop(crypto);

        self.store.touch_credential(cred.id).await?;
        Ok(Some((cred, plaintext)))
    }

    /// Find a domain by name, or create it if it does not exist.
    pub async fn find_or_create_domain(&self, name: &str, subject: &str) -> MvResult<Uuid> {
        let domains = self.store.list_domains().await?;
        if let Some(d) = domains
            .iter()
            .find(|d| d.name == name && d.revoked_at.is_none())
        {
            return Ok(d.id);
        }
        let domain = self
            .create_domain(name, Some("Auto-created for agent bridge"), subject)
            .await?;
        Ok(domain.id)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::credentials::{CredentialBackend, CredentialError, SecretSource};
    use hx_storage::keychain::SqliteKeychainStore;

    async fn test_engine() -> KeychainEngine {
        let store = Arc::new(SqliteKeychainStore::open_in_memory().unwrap());
        let cred_store = Arc::new(CredentialStore::env_only());
        KeychainEngine::new(store, cred_store, None, None)
            .await
            .unwrap()
    }

    #[derive(Debug)]
    struct StaticOsKeyringBackend {
        key: String,
        value: String,
    }

    impl CredentialBackend for StaticOsKeyringBackend {
        fn name(&self) -> &str {
            "Static OS Keyring"
        }

        fn source(&self) -> SecretSource {
            SecretSource::OsKeyring
        }

        fn is_available(&self) -> bool {
            true
        }

        fn get(&self, key: &str) -> Result<Option<String>, CredentialError> {
            if key == self.key {
                return Ok(Some(self.value.clone()));
            }
            Ok(None)
        }

        fn set(&self, _key: &str, _value: &str) -> Result<(), CredentialError> {
            Err(CredentialError::Other("read-only test backend".to_string()))
        }

        fn delete(&self, _key: &str) -> Result<(), CredentialError> {
            Ok(())
        }

        fn list_keys(&self) -> Result<Vec<String>, CredentialError> {
            Ok(vec![self.key.clone()])
        }

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    async fn test_engine_with_os_keyring_password(password: &str) -> KeychainEngine {
        let store = Arc::new(SqliteKeychainStore::open_in_memory().unwrap());
        let mut cred_store = CredentialStore::env_only();
        cred_store.insert_backend(
            0,
            Box::new(StaticOsKeyringBackend {
                key: "HELIX_VAULT_KEY".to_string(),
                value: password.to_string(),
            }),
        );
        KeychainEngine::new(store, Arc::new(cred_store), None, None)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn vault_lifecycle() {
        let engine = test_engine().await;

        // Status should be uninitialized
        let (state, _) = engine.vault_status().await.unwrap();
        assert_eq!(state, VaultState::Uninitialized);

        // Initialize
        engine
            .initialize_vault("test-password", false, "test")
            .await
            .unwrap();
        let (state, meta) = engine.vault_status().await.unwrap();
        assert_eq!(state, VaultState::Unsealed);
        assert!(meta.is_some());

        // Seal
        engine.seal("test").await.unwrap();
        let (state, _) = engine.vault_status().await.unwrap();
        assert_eq!(state, VaultState::Sealed);

        // Unseal
        engine.unseal("test-password", "test").await.unwrap();
        let (state, _) = engine.vault_status().await.unwrap();
        assert_eq!(state, VaultState::Unsealed);
    }

    #[tokio::test]
    async fn update_credential_metadata_applies_fields() {
        let engine = test_engine().await;
        engine
            .initialize_vault("test-password", false, "test")
            .await
            .unwrap();
        engine.unseal("test-password", "test").await.unwrap();

        let domain_id = engine
            .find_or_create_domain("oauth-clients", "test")
            .await
            .unwrap();
        let stored = engine
            .store_credential(
                domain_id,
                "client-id",
                "oauth_client_secret",
                b"secret",
                vec!["oauth".to_string()],
                None,
                "test",
            )
            .await
            .unwrap();

        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "template_id".to_string(),
            serde_json::Value::String("template-123".to_string()),
        );
        metadata.insert(
            "token_ttl_seconds".to_string(),
            serde_json::Value::Number(serde_json::Number::from(3600u64)),
        );

        let updated = engine
            .update_credential_metadata(
                stored.id,
                Some("AI Manager".to_string()),
                Some(vec!["oauth".to_string(), "client".to_string()]),
                Some(metadata.clone()),
                None,
                "test",
            )
            .await
            .unwrap();

        assert_eq!(updated.description.as_deref(), Some("AI Manager"));
        assert_eq!(updated.tags.len(), 2);
        assert_eq!(
            updated.metadata.get("template_id").and_then(|v| v.as_str()),
            Some("template-123")
        );
    }

    #[tokio::test]
    async fn wrong_password_rejected() {
        let engine = test_engine().await;
        engine
            .initialize_vault("correct", false, "test")
            .await
            .unwrap();
        engine.seal("test").await.unwrap();
        let result = engine.unseal("wrong", "test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn preferred_unseal_falls_back_to_passphrase_when_secure_storage_missing() {
        let engine = test_engine().await;
        engine
            .initialize_vault("fallback-pass", false, "test")
            .await
            .unwrap();
        engine.seal("test").await.unwrap();

        let source = engine
            .unseal_with_preferred_master_key(Some("fallback-pass"), "test")
            .await
            .unwrap();
        assert_eq!(source, MasterKeySource::PassphraseArgon2id);
    }

    #[tokio::test]
    async fn preferred_unseal_without_sources_fails() {
        let engine = test_engine().await;
        engine
            .initialize_vault("fallback-pass", false, "test")
            .await
            .unwrap();
        engine.seal("test").await.unwrap();

        let result = engine.unseal_with_preferred_master_key(None, "test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn preferred_unseal_prefers_os_secure_storage_over_passphrase() {
        let engine = test_engine_with_os_keyring_password("correct-pass").await;
        engine
            .initialize_vault("correct-pass", false, "test")
            .await
            .unwrap();
        engine.seal("test").await.unwrap();

        let source = engine
            .unseal_with_preferred_master_key(Some("wrong-pass"), "test")
            .await
            .unwrap();
        assert_eq!(source, MasterKeySource::OsSecureStorage);
    }

    #[tokio::test]
    async fn preferred_unseal_falls_back_when_secure_storage_unseal_fails() {
        let engine = test_engine_with_os_keyring_password("wrong-pass").await;
        engine
            .initialize_vault("correct-pass", false, "test")
            .await
            .unwrap();
        engine.seal("test").await.unwrap();

        let source = engine
            .unseal_with_preferred_master_key(Some("correct-pass"), "test")
            .await
            .unwrap();
        assert_eq!(source, MasterKeySource::PassphraseArgon2id);
    }

    #[tokio::test]
    async fn namespace_dek_wrap_unwrap_roundtrip() {
        let engine = test_engine().await;
        engine
            .initialize_vault("wrap-pass", false, "test")
            .await
            .unwrap();

        let dek = hx_storage::vault_crypto::VaultCrypto::generate_node_dek();
        let wrapped = engine.wrap_namespace_dek("default", &dek).await.unwrap();
        let unwrapped = engine
            .unwrap_namespace_dek("default", wrapped.as_str())
            .await
            .unwrap();
        assert_eq!(dek, unwrapped);
    }

    #[tokio::test]
    async fn credential_store_and_read() {
        let engine = test_engine().await;
        engine
            .initialize_vault("pass", false, "test")
            .await
            .unwrap();

        let domain = engine
            .create_domain("api-keys", Some("API key storage"), "test")
            .await
            .unwrap();

        let cred = engine
            .store_credential(
                domain.id,
                "openai-key",
                "api_key",
                b"sk-12345",
                vec!["prod".to_string()],
                None,
                "test",
            )
            .await
            .unwrap();

        let (loaded, plaintext, _alerts) = engine.read_credential(cred.id, "admin").await.unwrap();
        assert_eq!(loaded.name, "openai-key");
        assert_eq!(&*plaintext, b"sk-12345");
    }

    #[tokio::test]
    async fn delegation_chain() {
        let engine = test_engine().await;
        engine
            .initialize_vault("pass", false, "test")
            .await
            .unwrap();
        let domain = engine.create_domain("test", None, "test").await.unwrap();
        let cred = engine
            .store_credential(
                domain.id,
                "key1",
                "api_key",
                b"secret",
                vec![],
                None,
                "test",
            )
            .await
            .unwrap();

        let d1 = engine
            .create_delegation(
                cred.id,
                "alice",
                DelegationPermissions {
                    can_read: true,
                    can_use: false,
                    can_delegate: true,
                },
                None,
                3,
                "test",
            )
            .await
            .unwrap();
        assert_eq!(d1.depth, 0);

        let d2 = engine
            .sub_delegate(
                d1.id,
                "bob",
                DelegationPermissions {
                    can_read: true,
                    can_use: false,
                    can_delegate: false,
                },
                None,
                "test",
            )
            .await
            .unwrap();
        assert_eq!(d2.depth, 1);

        // Bob can read via delegation
        let (_, plaintext, _) = engine
            .read_credential_via_delegation(d2.id, "bob")
            .await
            .unwrap();
        assert_eq!(&*plaintext, b"secret");
    }

    #[tokio::test]
    async fn zk_proof_roundtrip() {
        let engine = test_engine().await;
        engine
            .initialize_vault("pass", false, "test")
            .await
            .unwrap();
        let domain = engine.create_domain("test", None, "test").await.unwrap();
        let cred = engine
            .store_credential(
                domain.id,
                "key1",
                "api_key",
                b"secret-value",
                vec![],
                None,
                "test",
            )
            .await
            .unwrap();

        let proof = engine
            .generate_proof(cred.id, "nonce-123", "test")
            .await
            .unwrap();
        assert!(engine.verify_proof(&proof, "test").await.unwrap());
    }

    #[tokio::test]
    async fn audit_chain_integrity() {
        let engine = test_engine().await;
        engine
            .initialize_vault("pass", false, "test")
            .await
            .unwrap();
        engine.create_domain("test", None, "test").await.unwrap();

        assert!(engine.verify_audit_integrity().await.unwrap().is_valid());

        let trail = engine.list_audit_trail(100, 0).await.unwrap();
        assert!(trail.len() >= 2); // vault_initialized + domain_created
    }

    #[tokio::test]
    async fn shamir_enable_seal_submit_unseal() {
        let engine = test_engine().await;
        engine
            .initialize_vault("pass", false, "test")
            .await
            .unwrap();

        // Enable Shamir 2-of-3
        let shares = engine.enable_shamir(2, 3, "test", None).await.unwrap();
        assert_eq!(shares.len(), 3);

        // Verify meta was updated
        let (_, meta) = engine.vault_status().await.unwrap();
        let meta = meta.unwrap();
        assert_eq!(meta.shamir_threshold, Some(2));
        assert_eq!(meta.shamir_total, Some(3));

        // Seal the vault
        engine.seal("test").await.unwrap();
        let (state, _) = engine.vault_status().await.unwrap();
        assert_eq!(state, VaultState::Sealed);

        // Submit 2 of 3 shares (enough for threshold=2)
        let status = engine.submit_shamir_share(&shares[0], None).await.unwrap();
        assert_eq!(status.shares_collected, 1);
        assert!(!status.ready);

        let status = engine.submit_shamir_share(&shares[2], None).await.unwrap();
        assert_eq!(status.shares_collected, 2);
        assert!(status.ready);

        // Unseal from shares
        engine.unseal_from_shares("test").await.unwrap();
        let (state, _) = engine.vault_status().await.unwrap();
        assert_eq!(state, VaultState::Unsealed);

        // Verify we can still read credentials (master key reconstructed correctly)
        let domain = engine
            .create_domain("test-shamir", None, "test")
            .await
            .unwrap();
        let cred = engine
            .store_credential(
                domain.id,
                "key1",
                "api_key",
                b"shamir-secret",
                vec![],
                None,
                "test",
            )
            .await
            .unwrap();
        let (_, plaintext, _) = engine.read_credential(cred.id, "test").await.unwrap();
        assert_eq!(&*plaintext, b"shamir-secret");
    }
}
