//! Vault cryptographic operations: HKDF key hierarchy, AES-256-GCM, HMAC, audit hashing.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::Argon2;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use rand::RngCore;
use sha2::{Digest, Sha256};
use sharks::{Share, Sharks};
use tracing::warn;
use zeroize::Zeroizing;

use crate::crypto::EncryptionConfig;

const KEY_SIZE: usize = 32;
const NONCE_SIZE: usize = 12;
const VERIFICATION_SENTINEL: &[u8] = b"HELIX_VAULT_SENTINEL_V1";

/// Ciphertext version byte. All new ciphertexts are prefixed with this.
/// Legacy ciphertexts (without prefix) are still supported for decryption.
const CIPHERTEXT_VERSION_1: u8 = 0x01;

// Argon2 parameter floor values (below these, unseal is refused)
const ARGON2_MIN_MEMORY_KIB: u32 = 65536; // 64 MiB
const ARGON2_MIN_ITERATIONS: u32 = 4;
const ARGON2_MIN_PARALLELISM: u32 = 4;

// Argon2 recommended values (below these, a warning is logged)
const ARGON2_REC_MEMORY_KIB: u32 = 65536; // 64 MiB
const ARGON2_REC_ITERATIONS: u32 = 4;
const ARGON2_REC_PARALLELISM: u32 = 4;

/// Validate Argon2 parameters. Returns an error if below the absolute minimum floor.
/// Logs a warning if below recommended production values.
/// Called from the engine layer (not from VaultCrypto::unseal) so tests can use weak params.
pub fn validate_argon2_params(config: &EncryptionConfig) -> Result<(), VaultCryptoError> {
    if config.argon2_memory_kib < ARGON2_MIN_MEMORY_KIB {
        return Err(VaultCryptoError::KeyDerivation(format!(
            "argon2 memory_kib {} is below minimum {ARGON2_MIN_MEMORY_KIB}",
            config.argon2_memory_kib
        )));
    }
    if config.argon2_iterations < ARGON2_MIN_ITERATIONS {
        return Err(VaultCryptoError::KeyDerivation(format!(
            "argon2 iterations {} is below minimum {ARGON2_MIN_ITERATIONS}",
            config.argon2_iterations
        )));
    }
    if config.argon2_parallelism < ARGON2_MIN_PARALLELISM {
        return Err(VaultCryptoError::KeyDerivation(format!(
            "argon2 parallelism {} is below minimum {ARGON2_MIN_PARALLELISM}",
            config.argon2_parallelism
        )));
    }

    if config.argon2_memory_kib < ARGON2_REC_MEMORY_KIB {
        warn!(
            memory_kib = config.argon2_memory_kib,
            recommended = ARGON2_REC_MEMORY_KIB,
            "argon2 memory below recommended production value"
        );
    }
    if config.argon2_iterations < ARGON2_REC_ITERATIONS {
        warn!(
            iterations = config.argon2_iterations,
            recommended = ARGON2_REC_ITERATIONS,
            "argon2 iterations below recommended production value"
        );
    }
    if config.argon2_parallelism < ARGON2_REC_PARALLELISM {
        warn!(
            parallelism = config.argon2_parallelism,
            recommended = ARGON2_REC_PARALLELISM,
            "argon2 parallelism below recommended production value"
        );
    }

    Ok(())
}

/// Maximum number of grace-period keys retained (oldest epochs pruned first).
const MAX_GRACE_KEYS: usize = 5;

/// Vault cryptographic engine. Holds the master key and grace-period keys.
pub struct VaultCrypto {
    master_key: Option<Zeroizing<[u8; KEY_SIZE]>>,
    grace_keys: HashMap<u64, (Instant, Zeroizing<[u8; KEY_SIZE]>)>,
}

impl VaultCrypto {
    pub fn new() -> Self {
        Self {
            master_key: None,
            grace_keys: HashMap::new(),
        }
    }

    /// Derive the master key from a password and salt using Argon2id.
    pub fn unseal(
        &mut self,
        password: &str,
        salt: &[u8],
        config: &EncryptionConfig,
    ) -> Result<(), VaultCryptoError> {
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(
                config.argon2_memory_kib,
                config.argon2_iterations,
                config.argon2_parallelism,
                Some(KEY_SIZE),
            )
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?,
        );

        let mut key = Zeroizing::new([0u8; KEY_SIZE]);
        argon2
            .hash_password_into(password.as_bytes(), salt, key.as_mut())
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?;

        // Lock key memory to prevent swapping to disk
        #[cfg(unix)]
        unsafe {
            let ret = libc::mlock(key.as_ptr() as *const libc::c_void, KEY_SIZE);
            if ret != 0 {
                warn!("mlock failed: {}", std::io::Error::last_os_error());
            }
        }

        self.master_key = Some(key);
        Ok(())
    }

    /// Zeroize all keys (seal the vault).
    pub fn seal(&mut self) {
        // Unlock memory before zeroizing
        if let Some(ref key) = self.master_key {
            #[cfg(unix)]
            unsafe {
                let ret = libc::munlock(key.as_ptr() as *const libc::c_void, KEY_SIZE);
                if ret != 0 {
                    warn!("munlock failed: {}", std::io::Error::last_os_error());
                }
            }
        }
        self.master_key = None;
        self.grace_keys.clear();
    }

    pub fn is_unsealed(&self) -> bool {
        self.master_key.is_some()
    }

    /// Extract a clone of the current master key (for wrapping during rotation).
    pub fn extract_master_key(&self) -> Result<Zeroizing<[u8; KEY_SIZE]>, VaultCryptoError> {
        let master = self.master_key()?;
        Ok(Zeroizing::new(*master))
    }

    /// Set the master key directly (for Secure Enclave / wrapped key injection).
    pub fn set_master_key(&mut self, key: Zeroizing<[u8; KEY_SIZE]>) {
        #[cfg(unix)]
        unsafe {
            let ret = libc::mlock(key.as_ptr() as *const libc::c_void, KEY_SIZE);
            if ret != 0 {
                warn!("mlock failed: {}", std::io::Error::last_os_error());
            }
        }
        self.master_key = Some(key);
    }

    /// Number of grace-period keys currently held.
    pub fn grace_key_count(&self) -> usize {
        self.grace_keys.len()
    }

    /// Add a grace-period key for an old epoch (used during key rotation).
    pub fn add_grace_key(&mut self, epoch: u64, key: Zeroizing<[u8; KEY_SIZE]>) {
        self.grace_keys.insert(epoch, (Instant::now(), key));
        self.prune_grace_keys();
    }

    /// Unwrap a grace key that was encrypted with the current master key.
    pub fn unwrap_grace_key(
        &mut self,
        epoch: u64,
        wrapped_b64: &str,
    ) -> Result<(), VaultCryptoError> {
        let master = self.master_key()?;
        let wrapped = BASE64
            .decode(wrapped_b64)
            .map_err(|e| VaultCryptoError::Decryption(format!("base64: {e}")))?;
        let plaintext = aes_gcm_decrypt(master, &wrapped)?;
        if plaintext.len() != KEY_SIZE {
            return Err(VaultCryptoError::Decryption(
                "invalid wrapped key length".into(),
            ));
        }
        let mut key = Zeroizing::new([0u8; KEY_SIZE]);
        key.copy_from_slice(&plaintext);
        self.grace_keys.insert(epoch, (Instant::now(), key));
        self.prune_grace_keys();
        Ok(())
    }

    /// Remove a grace-period key.
    pub fn remove_grace_key(&mut self, epoch: u64) {
        self.grace_keys.remove(&epoch);
    }

    /// Evict oldest-epoch grace keys when count exceeds `MAX_GRACE_KEYS`.
    fn prune_grace_keys(&mut self) {
        while self.grace_keys.len() > MAX_GRACE_KEYS {
            if let Some(&oldest_epoch) = self.grace_keys.keys().min() {
                self.grace_keys.remove(&oldest_epoch);
            } else {
                break;
            }
        }
    }

    /// Remove grace-period keys older than the given TTL.
    pub fn evict_expired_keys(&mut self, ttl: Duration) {
        let now = Instant::now();
        self.grace_keys
            .retain(|_, (inserted_at, _)| now.duration_since(*inserted_at) < ttl);
    }

    fn master_key(&self) -> Result<&[u8; KEY_SIZE], VaultCryptoError> {
        self.master_key
            .as_ref()
            .map(|k| &**k)
            .ok_or(VaultCryptoError::Sealed)
    }

    /// Derive a domain-level key: HKDF-SHA256(master_key, info=derivation_info).
    pub fn derive_domain_key(
        &self,
        derivation_info: &str,
    ) -> Result<Zeroizing<[u8; KEY_SIZE]>, VaultCryptoError> {
        let master = self.master_key()?;
        let hk = Hkdf::<Sha256>::new(None, master);
        let mut okm = Zeroizing::new([0u8; KEY_SIZE]);
        hk.expand(derivation_info.as_bytes(), okm.as_mut())
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?;
        Ok(okm)
    }

    /// Derive a namespace-scoped KEK for sealed-mode data envelope operations.
    pub fn derive_namespace_kek(
        &self,
        namespace: &str,
    ) -> Result<Zeroizing<[u8; KEY_SIZE]>, VaultCryptoError> {
        self.derive_domain_key(&format!("sealed-namespace-kek:{namespace}"))
    }

    /// Generate a random per-node DEK.
    pub fn generate_node_dek() -> [u8; KEY_SIZE] {
        let mut dek = [0u8; KEY_SIZE];
        OsRng.fill_bytes(&mut dek);
        dek
    }

    /// Derive a credential-level key: two-level HKDF derivation.
    pub fn derive_credential_key(
        &self,
        domain_info: &str,
        cred_info: &str,
    ) -> Result<Zeroizing<[u8; KEY_SIZE]>, VaultCryptoError> {
        let domain_key = self.derive_domain_key(domain_info)?;
        let hk = Hkdf::<Sha256>::new(None, domain_key.as_ref());
        let mut okm = Zeroizing::new([0u8; KEY_SIZE]);
        hk.expand(cred_info.as_bytes(), okm.as_mut())
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?;
        Ok(okm)
    }

    /// Derive a credential key using a specific epoch's grace key.
    fn derive_credential_key_with_epoch(
        &self,
        domain_info: &str,
        cred_info: &str,
        epoch: u64,
    ) -> Result<Zeroizing<[u8; KEY_SIZE]>, VaultCryptoError> {
        let (_, base_key) = self.grace_keys.get(&epoch).ok_or_else(|| {
            VaultCryptoError::KeyDerivation(format!("no grace key for epoch {epoch}"))
        })?;

        let hk_domain = Hkdf::<Sha256>::new(None, base_key.as_ref());
        let mut domain_key = Zeroizing::new([0u8; KEY_SIZE]);
        hk_domain
            .expand(domain_info.as_bytes(), domain_key.as_mut())
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?;

        let hk_cred = Hkdf::<Sha256>::new(None, domain_key.as_ref());
        let mut cred_key = Zeroizing::new([0u8; KEY_SIZE]);
        hk_cred
            .expand(cred_info.as_bytes(), cred_key.as_mut())
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?;

        Ok(cred_key)
    }

    /// Encrypt a credential value using the HKDF-derived credential key.
    pub fn encrypt_credential(
        &self,
        plaintext: &[u8],
        domain_info: &str,
        cred_info: &str,
    ) -> Result<String, VaultCryptoError> {
        let key = self.derive_credential_key(domain_info, cred_info)?;
        let encrypted = aes_gcm_encrypt(&*key, plaintext)?;
        Ok(BASE64.encode(encrypted))
    }

    /// Decrypt a credential value. Returns `Zeroizing<Vec<u8>>` to ensure memory is wiped on drop.
    pub fn decrypt_credential(
        &self,
        encoded: &str,
        domain_info: &str,
        cred_info: &str,
    ) -> Result<Zeroizing<Vec<u8>>, VaultCryptoError> {
        let key = self.derive_credential_key(domain_info, cred_info)?;
        let data = BASE64
            .decode(encoded)
            .map_err(|e| VaultCryptoError::Decryption(format!("base64: {e}")))?;
        aes_gcm_decrypt(&*key, &data).map(Zeroizing::new)
    }

    /// Decrypt using a specific epoch's grace key (for rotation grace period).
    pub fn decrypt_credential_with_epoch(
        &self,
        encoded: &str,
        domain_info: &str,
        cred_info: &str,
        epoch: u64,
    ) -> Result<Zeroizing<Vec<u8>>, VaultCryptoError> {
        let key = self.derive_credential_key_with_epoch(domain_info, cred_info, epoch)?;
        let data = BASE64
            .decode(encoded)
            .map_err(|e| VaultCryptoError::Decryption(format!("base64: {e}")))?;
        aes_gcm_decrypt(&*key, &data).map(Zeroizing::new)
    }

    /// Generate a verification blob by encrypting a known sentinel.
    pub fn generate_verification_blob(&self) -> Result<String, VaultCryptoError> {
        let master = self.master_key()?;
        let encrypted = aes_gcm_encrypt(master, VERIFICATION_SENTINEL)?;
        Ok(BASE64.encode(encrypted))
    }

    /// Verify the password by decrypting the verification blob and checking the sentinel.
    pub fn verify_password(&self, blob: &str) -> Result<bool, VaultCryptoError> {
        let master = self.master_key()?;
        let data = BASE64
            .decode(blob)
            .map_err(|e| VaultCryptoError::Decryption(format!("base64: {e}")))?;
        match aes_gcm_decrypt(master, &data) {
            Ok(plaintext) => Ok(plaintext == VERIFICATION_SENTINEL),
            Err(_) => Ok(false),
        }
    }

    /// Compute an HMAC-SHA256 chain hash for delegation verification.
    /// `depth` and `max_depth` are included in the HMAC input for strengthened delegation chains.
    pub fn compute_chain_hash(
        &self,
        domain_info: &str,
        prev_hash: Option<&str>,
        cred_id: &str,
        delegatee: &str,
        perms: &str,
        expires_at: Option<&str>,
        depth: u32,
        max_depth: u32,
    ) -> Result<String, VaultCryptoError> {
        let master = self.master_key()?;
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(master)
            .map_err(|e| VaultCryptoError::Encryption(e.to_string()))?;
        mac.update(domain_info.as_bytes());
        mac.update(prev_hash.unwrap_or("genesis").as_bytes());
        mac.update(cred_id.as_bytes());
        mac.update(delegatee.as_bytes());
        mac.update(perms.as_bytes());
        mac.update(expires_at.unwrap_or("none").as_bytes());
        mac.update(depth.to_le_bytes().as_slice());
        mac.update(max_depth.to_le_bytes().as_slice());
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    // --- Access Proof (replaces ZK naming) ---

    /// Generate an access proof: HMAC-SHA256(credential_value, challenge_nonce).
    pub fn generate_access_proof(
        &self,
        credential_value: &[u8],
        challenge_nonce: &str,
    ) -> Result<String, VaultCryptoError> {
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(credential_value)
            .map_err(|e| VaultCryptoError::Encryption(e.to_string()))?;
        mac.update(challenge_nonce.as_bytes());
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    /// Verify an access proof.
    pub fn verify_access_proof(
        &self,
        credential_value: &[u8],
        challenge_nonce: &str,
        proof: &str,
    ) -> Result<bool, VaultCryptoError> {
        let expected = self.generate_access_proof(credential_value, challenge_nonce)?;
        Ok(constant_time_eq(expected.as_bytes(), proof.as_bytes()))
    }

    /// Generate a ZK access proof (legacy alias for generate_access_proof).
    pub fn generate_zk_proof(
        &self,
        credential_value: &[u8],
        challenge_nonce: &str,
    ) -> Result<String, VaultCryptoError> {
        self.generate_access_proof(credential_value, challenge_nonce)
    }

    /// Verify a ZK access proof (legacy alias for verify_access_proof).
    pub fn verify_zk_proof(
        &self,
        credential_value: &[u8],
        challenge_nonce: &str,
        proof: &str,
    ) -> Result<bool, VaultCryptoError> {
        self.verify_access_proof(credential_value, challenge_nonce, proof)
    }

    // --- Audit ---

    /// Compute a SHA-256 audit chain hash.
    pub fn compute_audit_hash(
        previous_hash: Option<&str>,
        sequence: i64,
        action: &str,
        subject: &str,
        resource_id: Option<&str>,
        timestamp: &str,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(previous_hash.unwrap_or("genesis").as_bytes());
        hasher.update(sequence.to_string().as_bytes());
        hasher.update(action.as_bytes());
        hasher.update(subject.as_bytes());
        hasher.update(resource_id.unwrap_or("").as_bytes());
        hasher.update(timestamp.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Sign an audit entry with HMAC-SHA256 using an HKDF-derived audit key.
    pub fn sign_audit_entry(
        &self,
        sequence: i64,
        action: &str,
        subject: &str,
        resource_id: Option<&str>,
        entry_hash: &str,
        timestamp: &str,
    ) -> Result<String, VaultCryptoError> {
        let audit_key = self.derive_domain_key("audit-signing")?;
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(audit_key.as_ref())
            .map_err(|e| VaultCryptoError::Encryption(e.to_string()))?;
        mac.update(sequence.to_string().as_bytes());
        mac.update(action.as_bytes());
        mac.update(subject.as_bytes());
        mac.update(resource_id.unwrap_or("").as_bytes());
        mac.update(entry_hash.as_bytes());
        mac.update(timestamp.as_bytes());
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    /// Verify an audit entry's HMAC signature.
    pub fn verify_audit_signature(
        &self,
        sequence: i64,
        action: &str,
        subject: &str,
        resource_id: Option<&str>,
        entry_hash: &str,
        timestamp: &str,
        signature: &str,
    ) -> Result<bool, VaultCryptoError> {
        let expected = self.sign_audit_entry(
            sequence,
            action,
            subject,
            resource_id,
            entry_hash,
            timestamp,
        )?;
        Ok(constant_time_eq(expected.as_bytes(), signature.as_bytes()))
    }

    // --- Public encrypt for key wrapping ---

    // --- Shamir Secret Sharing ---

    /// Split the current master key into N shares with threshold M.
    /// Returns the shares as `ShamirShare` structs for export/distribution.
    pub fn split_master_key(
        &self,
        threshold: u8,
        total: u8,
    ) -> Result<Vec<ShamirShare>, VaultCryptoError> {
        if threshold < 2 {
            return Err(VaultCryptoError::Encryption(
                "shamir threshold must be >= 2".into(),
            ));
        }
        if total < threshold {
            return Err(VaultCryptoError::Encryption(
                "shamir total must be >= threshold".into(),
            ));
        }

        let master = self.master_key()?;
        let sharks = Sharks(threshold);
        let dealer = sharks.dealer(master.as_ref());

        let shares: Vec<ShamirShare> = dealer
            .take(total as usize)
            .enumerate()
            .map(|(i, share)| {
                let bytes: Vec<u8> = (&share).into();
                ShamirShare {
                    index: (i + 1) as u8,
                    data: bytes,
                }
            })
            .collect();

        Ok(shares)
    }

    /// Recover the master key from M-of-N Shamir shares.
    /// The threshold must match the original split parameter.
    pub fn recover_from_shares(
        shares: &[ShamirShare],
        threshold: u8,
    ) -> Result<Zeroizing<[u8; KEY_SIZE]>, VaultCryptoError> {
        if (shares.len() as u8) < threshold {
            return Err(VaultCryptoError::KeyDerivation(format!(
                "need at least {} shares, got {}",
                threshold,
                shares.len()
            )));
        }

        let shark_shares: Vec<Share> = shares
            .iter()
            .map(|s| {
                Share::try_from(s.data.as_slice()).map_err(|e| {
                    VaultCryptoError::KeyDerivation(format!("invalid share data: {e}"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let sharks = Sharks(threshold);
        let recovered = sharks
            .recover(&shark_shares)
            .map_err(|e| VaultCryptoError::KeyDerivation(format!("share recovery failed: {e}")))?;

        if recovered.len() != KEY_SIZE {
            return Err(VaultCryptoError::KeyDerivation(format!(
                "recovered key has wrong length: {} (expected {})",
                recovered.len(),
                KEY_SIZE
            )));
        }

        let mut key = Zeroizing::new([0u8; KEY_SIZE]);
        key.copy_from_slice(&recovered);
        Ok(key)
    }

    // --- Metadata encryption ---

    /// Encrypt a metadata string (name, description) using an HKDF-derived metadata key.
    /// Returns a base64-encoded ciphertext.
    pub fn encrypt_metadata(&self, plaintext: &str) -> Result<String, VaultCryptoError> {
        let key = self.derive_domain_key("metadata-encryption")?;
        let encrypted = aes_gcm_encrypt(&*key, plaintext.as_bytes())?;
        Ok(BASE64.encode(encrypted))
    }

    /// Decrypt a metadata string. Returns the plaintext.
    pub fn decrypt_metadata(&self, encoded: &str) -> Result<String, VaultCryptoError> {
        let key = self.derive_domain_key("metadata-encryption")?;
        let data = BASE64
            .decode(encoded)
            .map_err(|e| VaultCryptoError::Decryption(format!("base64: {e}")))?;
        let plaintext = aes_gcm_decrypt(&*key, &data)?;
        String::from_utf8(plaintext)
            .map_err(|e| VaultCryptoError::Decryption(format!("utf-8: {e}")))
    }

    /// Decrypt a metadata string using a specific epoch's grace key.
    pub fn decrypt_metadata_with_epoch(
        &self,
        encoded: &str,
        epoch: u64,
    ) -> Result<String, VaultCryptoError> {
        let (_, base_key) = self.grace_keys.get(&epoch).ok_or_else(|| {
            VaultCryptoError::KeyDerivation(format!("no grace key for epoch {epoch}"))
        })?;
        let hk = Hkdf::<Sha256>::new(None, base_key.as_ref());
        let mut meta_key = Zeroizing::new([0u8; KEY_SIZE]);
        hk.expand(b"metadata-encryption", meta_key.as_mut())
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?;
        let data = BASE64
            .decode(encoded)
            .map_err(|e| VaultCryptoError::Decryption(format!("base64: {e}")))?;
        let plaintext = aes_gcm_decrypt(&*meta_key, &data)?;
        String::from_utf8(plaintext)
            .map_err(|e| VaultCryptoError::Decryption(format!("utf-8: {e}")))
    }

    /// Encrypt arbitrary data with a provided key (used for wrapping old master keys).
    pub fn aes_gcm_encrypt_pub(
        key: &[u8; KEY_SIZE],
        plaintext: &[u8],
    ) -> Result<Vec<u8>, VaultCryptoError> {
        aes_gcm_encrypt(key, plaintext)
    }

    /// Public helper for AES-256-GCM decryption (expects nonce-prefixed ciphertext).
    pub fn aes_gcm_decrypt_pub(
        key: &[u8; KEY_SIZE],
        data: &[u8],
    ) -> Result<Vec<u8>, VaultCryptoError> {
        aes_gcm_decrypt(key, data)
    }

    /// Wrap a raw DEK under a namespace KEK and encode as base64.
    pub fn wrap_dek(
        namespace_kek: &[u8; KEY_SIZE],
        dek: &[u8; KEY_SIZE],
    ) -> Result<String, VaultCryptoError> {
        let wrapped = aes_gcm_encrypt(namespace_kek, dek)?;
        Ok(BASE64.encode(wrapped))
    }

    /// Unwrap a base64-encoded DEK using a namespace KEK.
    pub fn unwrap_dek(
        namespace_kek: &[u8; KEY_SIZE],
        wrapped_b64: &str,
    ) -> Result<[u8; KEY_SIZE], VaultCryptoError> {
        let wrapped = BASE64
            .decode(wrapped_b64)
            .map_err(|e| VaultCryptoError::Decryption(format!("base64: {e}")))?;
        let plaintext = aes_gcm_decrypt(namespace_kek, &wrapped)?;
        if plaintext.len() != KEY_SIZE {
            return Err(VaultCryptoError::Decryption(
                "invalid unwrapped dek length".to_string(),
            ));
        }
        let mut dek = [0u8; KEY_SIZE];
        dek.copy_from_slice(&plaintext);
        Ok(dek)
    }

    /// Backward-compatible alias for `wrap_dek`.
    pub fn wrap_node_dek(
        namespace_kek: &[u8; KEY_SIZE],
        dek: &[u8; KEY_SIZE],
    ) -> Result<String, VaultCryptoError> {
        Self::wrap_dek(namespace_kek, dek)
    }

    /// Backward-compatible alias for `unwrap_dek`.
    pub fn unwrap_node_dek(
        namespace_kek: &[u8; KEY_SIZE],
        wrapped_b64: &str,
    ) -> Result<[u8; KEY_SIZE], VaultCryptoError> {
        Self::unwrap_dek(namespace_kek, wrapped_b64)
    }
}

impl Default for VaultCrypto {
    fn default() -> Self {
        Self::new()
    }
}

/// A single Shamir share for key splitting/recovery.
#[derive(Debug, Clone)]
pub struct ShamirShare {
    /// Human-readable index (1-based, for display/tracking).
    pub index: u8,
    /// Raw share bytes (x-coordinate + y-values, as serialized by `sharks`).
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Share passphrase encryption (Argon2id + AES-256-GCM)
// ---------------------------------------------------------------------------

/// Salt size for share passphrase encryption.
const SHARE_SALT_SIZE: usize = 16;

impl VaultCrypto {
    /// Encrypt a Shamir share's data with a passphrase.
    ///
    /// Output format: `[salt(16)] || [AES-GCM ciphertext]`
    /// where the AES key is derived via Argon2id(passphrase, salt).
    pub fn encrypt_share(share_data: &[u8], passphrase: &str) -> Result<Vec<u8>, VaultCryptoError> {
        let mut salt = [0u8; SHARE_SALT_SIZE];
        OsRng.fill_bytes(&mut salt);

        let key = Self::derive_share_key(passphrase, &salt)?;
        let ciphertext = aes_gcm_encrypt(&key, share_data)?;

        let mut out = Vec::with_capacity(SHARE_SALT_SIZE + ciphertext.len());
        out.extend_from_slice(&salt);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Decrypt a Shamir share's data with a passphrase.
    pub fn decrypt_share(encrypted: &[u8], passphrase: &str) -> Result<Vec<u8>, VaultCryptoError> {
        if encrypted.len() < SHARE_SALT_SIZE + 1 + NONCE_SIZE + 16 {
            return Err(VaultCryptoError::Decryption(
                "encrypted share too short".into(),
            ));
        }
        let salt = &encrypted[..SHARE_SALT_SIZE];
        let ciphertext = &encrypted[SHARE_SALT_SIZE..];

        let key = Self::derive_share_key(passphrase, salt)?;
        aes_gcm_decrypt(&key, ciphertext)
    }

    /// Derive an AES-256 key from a passphrase and salt using lightweight Argon2id
    /// (tuned for interactive share entry, not vault unsealing).
    fn derive_share_key(passphrase: &str, salt: &[u8]) -> Result<[u8; KEY_SIZE], VaultCryptoError> {
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(
                16384, // 16 MiB — lighter than vault unseal (interactive UX)
                2,
                1,
                Some(KEY_SIZE),
            )
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?,
        );

        let mut key = [0u8; KEY_SIZE];
        argon2
            .hash_password_into(passphrase.as_bytes(), salt, &mut key)
            .map_err(|e| VaultCryptoError::KeyDerivation(e.to_string()))?;
        Ok(key)
    }
}

// ---------------------------------------------------------------------------
// AES-256-GCM helpers (with ciphertext versioning)
// ---------------------------------------------------------------------------

fn aes_gcm_encrypt(key: &[u8; KEY_SIZE], plaintext: &[u8]) -> Result<Vec<u8>, VaultCryptoError> {
    let cipher =
        Aes256Gcm::new_from_slice(key).map_err(|e| VaultCryptoError::Encryption(e.to_string()))?;

    let mut nonce_bytes = [0u8; NONCE_SIZE];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| VaultCryptoError::Encryption(e.to_string()))?;

    // Versioned format: [version(1)] || [nonce(12)] || [ciphertext+tag]
    let mut out = Vec::with_capacity(1 + NONCE_SIZE + ciphertext.len());
    out.push(CIPHERTEXT_VERSION_1);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

fn aes_gcm_decrypt(key: &[u8; KEY_SIZE], data: &[u8]) -> Result<Vec<u8>, VaultCryptoError> {
    // Detect versioned vs. legacy ciphertext
    let (nonce_start, ciphertext_start) = if !data.is_empty() && data[0] == CIPHERTEXT_VERSION_1 {
        // Versioned: [0x01] || [nonce(12)] || [ciphertext+tag]
        if data.len() < 1 + NONCE_SIZE + 16 {
            return Err(VaultCryptoError::Decryption(
                "versioned data too short".into(),
            ));
        }
        (1, 1 + NONCE_SIZE)
    } else {
        // Legacy: [nonce(12)] || [ciphertext+tag]
        if data.len() < NONCE_SIZE + 16 {
            return Err(VaultCryptoError::Decryption("data too short".into()));
        }
        (0, NONCE_SIZE)
    };

    let cipher =
        Aes256Gcm::new_from_slice(key).map_err(|e| VaultCryptoError::Decryption(e.to_string()))?;

    let nonce = Nonce::from_slice(&data[nonce_start..nonce_start + NONCE_SIZE]);
    let ciphertext = &data[ciphertext_start..];

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|e| VaultCryptoError::Decryption(e.to_string()))
}

/// Constant-time comparison to prevent timing attacks.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Hex encoding (no extra dependency needed).
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes.as_ref().iter().map(|b| format!("{b:02x}")).collect()
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum VaultCryptoError {
    Sealed,
    KeyDerivation(String),
    Encryption(String),
    Decryption(String),
}

impl std::fmt::Display for VaultCryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sealed => write!(f, "vault is sealed"),
            Self::KeyDerivation(msg) => write!(f, "key derivation failed: {msg}"),
            Self::Encryption(msg) => write!(f, "encryption failed: {msg}"),
            Self::Decryption(msg) => write!(f, "decryption failed: {msg}"),
        }
    }
}

impl std::error::Error for VaultCryptoError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::EncryptionConfig;

    fn test_crypto() -> VaultCrypto {
        let mut vc = VaultCrypto::new();
        let config = EncryptionConfig {
            enabled: true,
            argon2_memory_kib: 1024, // small for tests
            argon2_iterations: 1,
            argon2_parallelism: 1,
        };
        vc.unseal("test-password", b"test-salt-16byt", &config)
            .unwrap();
        vc
    }

    #[test]
    fn unseal_and_seal() {
        let mut vc = test_crypto();
        assert!(vc.is_unsealed());
        vc.seal();
        assert!(!vc.is_unsealed());
    }

    #[test]
    fn derive_domain_key_deterministic() {
        let vc = test_crypto();
        let k1 = vc.derive_domain_key("domain:api-keys").unwrap();
        let k2 = vc.derive_domain_key("domain:api-keys").unwrap();
        assert_eq!(k1.as_ref(), k2.as_ref());
    }

    #[test]
    fn derive_different_domains_different_keys() {
        let vc = test_crypto();
        let k1 = vc.derive_domain_key("domain:api-keys").unwrap();
        let k2 = vc.derive_domain_key("domain:ssh-keys").unwrap();
        assert_ne!(k1.as_ref(), k2.as_ref());
    }

    #[test]
    fn derive_namespace_kek_is_stable_and_scoped() {
        let vc = test_crypto();
        let a1 = vc.derive_namespace_kek("default").unwrap();
        let a2 = vc.derive_namespace_kek("default").unwrap();
        let b = vc.derive_namespace_kek("other").unwrap();
        assert_eq!(a1.as_ref(), a2.as_ref());
        assert_ne!(a1.as_ref(), b.as_ref());
    }

    #[test]
    fn encrypt_decrypt_credential_roundtrip() {
        let vc = test_crypto();
        let plaintext = b"super-secret-api-key-12345";
        let encrypted = vc
            .encrypt_credential(plaintext, "domain:api", "cred:my-key")
            .unwrap();
        let decrypted = vc
            .decrypt_credential(&encrypted, "domain:api", "cred:my-key")
            .unwrap();
        assert_eq!(&*decrypted, plaintext);
    }

    #[test]
    fn wrong_info_cannot_decrypt() {
        let vc = test_crypto();
        let plaintext = b"secret";
        let encrypted = vc
            .encrypt_credential(plaintext, "domain:a", "cred:x")
            .unwrap();
        let result = vc.decrypt_credential(&encrypted, "domain:b", "cred:x");
        assert!(result.is_err());
    }

    #[test]
    fn verification_blob_roundtrip() {
        let vc = test_crypto();
        let blob = vc.generate_verification_blob().unwrap();
        assert!(vc.verify_password(&blob).unwrap());
    }

    #[test]
    fn chain_hash_deterministic() {
        let vc = test_crypto();
        let h1 = vc
            .compute_chain_hash("d:api", None, "cred-1", "alice", "r", None, 0, 3)
            .unwrap();
        let h2 = vc
            .compute_chain_hash("d:api", None, "cred-1", "alice", "r", None, 0, 3)
            .unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn chain_hash_depth_changes_hash() {
        let vc = test_crypto();
        let h1 = vc
            .compute_chain_hash("d:api", None, "cred-1", "alice", "r", None, 0, 3)
            .unwrap();
        let h2 = vc
            .compute_chain_hash("d:api", None, "cred-1", "alice", "r", None, 1, 3)
            .unwrap();
        assert_ne!(h1, h2);
    }

    #[test]
    fn zk_proof_roundtrip() {
        let vc = test_crypto();
        let value = b"my-api-key";
        let nonce = "random-nonce-123";
        let proof = vc.generate_zk_proof(value, nonce).unwrap();
        assert!(vc.verify_zk_proof(value, nonce, &proof).unwrap());
        assert!(!vc.verify_zk_proof(value, "wrong-nonce", &proof).unwrap());
    }

    #[test]
    fn access_proof_aliases_zk_proof() {
        let vc = test_crypto();
        let value = b"my-api-key";
        let nonce = "test-nonce";
        let proof_old = vc.generate_zk_proof(value, nonce).unwrap();
        let proof_new = vc.generate_access_proof(value, nonce).unwrap();
        assert_eq!(proof_old, proof_new);
        assert!(vc.verify_access_proof(value, nonce, &proof_old).unwrap());
    }

    #[test]
    fn audit_hash_chain() {
        let h1 = VaultCrypto::compute_audit_hash(
            None,
            1,
            "vault_initialized",
            "admin",
            None,
            "2025-01-01T00:00:00Z",
        );
        let h2 = VaultCrypto::compute_audit_hash(
            Some(&h1),
            2,
            "credential_stored",
            "admin",
            Some("cred-1"),
            "2025-01-01T00:01:00Z",
        );
        assert_ne!(h1, h2);
        let h2b = VaultCrypto::compute_audit_hash(
            Some(&h1),
            2,
            "credential_stored",
            "admin",
            Some("cred-1"),
            "2025-01-01T00:01:00Z",
        );
        assert_eq!(h2, h2b);
    }

    #[test]
    fn audit_signature_roundtrip() {
        let vc = test_crypto();
        let sig = vc
            .sign_audit_entry(
                1,
                "vault_initialized",
                "admin",
                None,
                "hash123",
                "2025-01-01T00:00:00Z",
            )
            .unwrap();
        assert!(vc
            .verify_audit_signature(
                1,
                "vault_initialized",
                "admin",
                None,
                "hash123",
                "2025-01-01T00:00:00Z",
                &sig
            )
            .unwrap());
        // Tampered action should fail
        assert!(!vc
            .verify_audit_signature(
                1,
                "tampered",
                "admin",
                None,
                "hash123",
                "2025-01-01T00:00:00Z",
                &sig
            )
            .unwrap());
    }

    #[test]
    fn sealed_vault_rejects_operations() {
        let vc = VaultCrypto::new();
        assert!(vc.derive_domain_key("domain:x").is_err());
        assert!(vc.encrypt_credential(b"test", "d", "c").is_err());
    }

    #[test]
    fn ciphertext_versioning_backward_compat() {
        let vc = test_crypto();
        // Encrypt with versioned format
        let encrypted = vc
            .encrypt_credential(b"test-value", "domain:test", "cred:test")
            .unwrap();
        // Verify the base64-decoded data starts with version byte
        let raw = BASE64.decode(&encrypted).unwrap();
        assert_eq!(raw[0], CIPHERTEXT_VERSION_1);
        // Decrypt should work
        let decrypted = vc
            .decrypt_credential(&encrypted, "domain:test", "cred:test")
            .unwrap();
        assert_eq!(&*decrypted, b"test-value");
    }

    #[test]
    fn extract_and_set_master_key() {
        let mut vc = test_crypto();
        let key = vc.extract_master_key().unwrap();
        vc.seal();
        assert!(!vc.is_unsealed());
        vc.set_master_key(key);
        assert!(vc.is_unsealed());
    }

    #[test]
    fn validate_argon2_rejects_below_floor() {
        let config = EncryptionConfig {
            enabled: true,
            argon2_memory_kib: 1024, // below 16384 floor
            argon2_iterations: 4,
            argon2_parallelism: 4,
        };
        assert!(validate_argon2_params(&config).is_err());
    }

    #[test]
    fn validate_argon2_accepts_good_params() {
        let config = EncryptionConfig::default();
        assert!(validate_argon2_params(&config).is_ok());
    }

    #[test]
    fn wrap_unwrap_node_dek_roundtrip() {
        let vc = test_crypto();
        let kek = vc.derive_namespace_kek("default").unwrap();
        let dek = VaultCrypto::generate_node_dek();
        let wrapped = VaultCrypto::wrap_node_dek(&kek, &dek).unwrap();
        let unwrapped = VaultCrypto::unwrap_node_dek(&kek, &wrapped).unwrap();
        assert_eq!(dek, unwrapped);
    }

    // --- Shamir tests ---

    #[test]
    fn shamir_split_2_of_3_recover_with_2() {
        let vc = test_crypto();
        let original_key = vc.extract_master_key().unwrap();
        let shares = vc.split_master_key(2, 3).unwrap();
        assert_eq!(shares.len(), 3);

        // Any 2 shares should recover the key
        let recovered = VaultCrypto::recover_from_shares(&shares[0..2], 2).unwrap();
        assert_eq!(&*recovered, &*original_key);

        let recovered2 = VaultCrypto::recover_from_shares(&shares[1..3], 2).unwrap();
        assert_eq!(&*recovered2, &*original_key);

        let recovered3 =
            VaultCrypto::recover_from_shares(&[shares[0].clone(), shares[2].clone()], 2).unwrap();
        assert_eq!(&*recovered3, &*original_key);
    }

    #[test]
    fn shamir_split_3_of_5_recover_with_3() {
        let vc = test_crypto();
        let original_key = vc.extract_master_key().unwrap();
        let shares = vc.split_master_key(3, 5).unwrap();
        assert_eq!(shares.len(), 5);

        let recovered = VaultCrypto::recover_from_shares(&shares[0..3], 3).unwrap();
        assert_eq!(&*recovered, &*original_key);
    }

    #[test]
    fn shamir_fail_with_insufficient_shares() {
        let vc = test_crypto();
        let shares = vc.split_master_key(2, 3).unwrap();

        // Only 1 share when threshold is 2
        let result = VaultCrypto::recover_from_shares(&shares[0..1], 2);
        assert!(result.is_err());
    }

    #[test]
    fn shamir_invalid_threshold() {
        let vc = test_crypto();
        assert!(vc.split_master_key(1, 3).is_err()); // threshold < 2
        assert!(vc.split_master_key(5, 3).is_err()); // threshold > total
    }

    #[test]
    fn shamir_sealed_vault_rejects_split() {
        let vc = VaultCrypto::new();
        assert!(vc.split_master_key(2, 3).is_err());
    }
}
