//! Encryption at rest for Helix storage.
//!
//! Uses AES-256-GCM for content encryption and Argon2id for key derivation.

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    Aes256Gcm, Nonce,
};
use argon2::Argon2;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use rand::RngCore;
use zeroize::Zeroizing;

const NONCE_SIZE: usize = 12;
const KEY_SIZE: usize = 32;
const SALT_SIZE: usize = 16;

/// Configuration for encryption at rest.
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Whether encryption is enabled.
    pub enabled: bool,
    /// Key derivation parameters.
    pub argon2_memory_kib: u32,
    pub argon2_iterations: u32,
    pub argon2_parallelism: u32,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            argon2_memory_kib: 65536, // 64 MiB
            argon2_iterations: 4,
            argon2_parallelism: 4,
        }
    }
}

impl EncryptionConfig {
    /// Create config from environment variables.
    pub fn from_env() -> Self {
        let enabled = std::env::var("HELIX_ENCRYPTION_ENABLED")
            .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
            .unwrap_or(false);

        let argon2_memory_kib = std::env::var("HELIX_ENCRYPTION_ARGON2_MEMORY_KIB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(65536);

        let argon2_iterations = std::env::var("HELIX_ENCRYPTION_ARGON2_ITERATIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        let argon2_parallelism = std::env::var("HELIX_ENCRYPTION_ARGON2_PARALLELISM")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(4);

        Self {
            enabled,
            argon2_memory_kib,
            argon2_iterations,
            argon2_parallelism,
        }
    }
}

/// Encryption error types.
#[derive(Debug)]
pub enum CryptoError {
    KeyDerivationFailed(String),
    EncryptionFailed(String),
    DecryptionFailed(String),
    InvalidData(String),
    KeyNotSet,
}

impl std::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::KeyDerivationFailed(msg) => write!(f, "key derivation failed: {msg}"),
            Self::EncryptionFailed(msg) => write!(f, "encryption failed: {msg}"),
            Self::DecryptionFailed(msg) => write!(f, "decryption failed: {msg}"),
            Self::InvalidData(msg) => write!(f, "invalid encrypted data: {msg}"),
            Self::KeyNotSet => write!(f, "encryption key not set"),
        }
    }
}

impl std::error::Error for CryptoError {}

/// Encrypted data format: base64(salt || nonce || ciphertext || tag)
pub struct EncryptedData {
    pub salt: [u8; SALT_SIZE],
    pub nonce: [u8; NONCE_SIZE],
    pub ciphertext: Vec<u8>,
}

impl EncryptedData {
    /// Encode to base64 string for storage.
    pub fn to_base64(&self) -> String {
        let mut data = Vec::with_capacity(SALT_SIZE + NONCE_SIZE + self.ciphertext.len());
        data.extend_from_slice(&self.salt);
        data.extend_from_slice(&self.nonce);
        data.extend_from_slice(&self.ciphertext);
        BASE64.encode(&data)
    }

    /// Decode from base64 string.
    pub fn from_base64(encoded: &str) -> Result<Self, CryptoError> {
        let data = BASE64
            .decode(encoded)
            .map_err(|e| CryptoError::InvalidData(format!("base64 decode failed: {e}")))?;

        if data.len() < SALT_SIZE + NONCE_SIZE + 16 {
            return Err(CryptoError::InvalidData("data too short".into()));
        }

        let mut salt = [0u8; SALT_SIZE];
        let mut nonce = [0u8; NONCE_SIZE];
        salt.copy_from_slice(&data[..SALT_SIZE]);
        nonce.copy_from_slice(&data[SALT_SIZE..SALT_SIZE + NONCE_SIZE]);
        let ciphertext = data[SALT_SIZE + NONCE_SIZE..].to_vec();

        Ok(Self {
            salt,
            nonce,
            ciphertext,
        })
    }
}

/// Key manager for encryption operations.
pub struct KeyManager {
    config: EncryptionConfig,
    master_key: Option<Zeroizing<[u8; KEY_SIZE]>>,
}

impl KeyManager {
    /// Create a new key manager.
    pub fn new(config: EncryptionConfig) -> Self {
        Self {
            config,
            master_key: None,
        }
    }

    /// Create from environment, deriving key from HELIX_ENCRYPTION_KEY.
    pub fn from_env() -> Result<Self, CryptoError> {
        let config = EncryptionConfig::from_env();
        let mut manager = Self::new(config);

        if manager.config.enabled {
            let password = std::env::var("HELIX_ENCRYPTION_KEY")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .ok_or(CryptoError::KeyNotSet)?;

            // Use a fixed salt for the master key derivation
            // In production, this should be stored alongside the encrypted data
            let salt = std::env::var("HELIX_ENCRYPTION_SALT")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| "helix-default-salt-v1".to_string());

            manager.derive_master_key(&password, salt.as_bytes())?;
        }

        Ok(manager)
    }

    /// Derive the master key from a password.
    pub fn derive_master_key(&mut self, password: &str, salt: &[u8]) -> Result<(), CryptoError> {
        let argon2 = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(
                self.config.argon2_memory_kib,
                self.config.argon2_iterations,
                self.config.argon2_parallelism,
                Some(KEY_SIZE),
            )
            .map_err(|e| CryptoError::KeyDerivationFailed(e.to_string()))?,
        );

        let mut key = Zeroizing::new([0u8; KEY_SIZE]);
        argon2
            .hash_password_into(password.as_bytes(), salt, key.as_mut())
            .map_err(|e| CryptoError::KeyDerivationFailed(e.to_string()))?;

        self.master_key = Some(key);
        Ok(())
    }

    /// Check if encryption is enabled and key is set.
    pub fn is_ready(&self) -> bool {
        self.config.enabled && self.master_key.is_some()
    }

    /// Encrypt plaintext data.
    pub fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedData, CryptoError> {
        let key = self.master_key.as_ref().ok_or(CryptoError::KeyNotSet)?;

        let cipher = Aes256Gcm::new_from_slice(key.as_ref())
            .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;

        // Generate random salt and nonce
        let mut salt = [0u8; SALT_SIZE];
        let mut nonce_bytes = [0u8; NONCE_SIZE];
        OsRng.fill_bytes(&mut salt);
        OsRng.fill_bytes(&mut nonce_bytes);

        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;

        Ok(EncryptedData {
            salt,
            nonce: nonce_bytes,
            ciphertext,
        })
    }

    /// Decrypt encrypted data.
    pub fn decrypt(&self, encrypted: &EncryptedData) -> Result<Vec<u8>, CryptoError> {
        let key = self.master_key.as_ref().ok_or(CryptoError::KeyNotSet)?;

        let cipher = Aes256Gcm::new_from_slice(key.as_ref())
            .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))?;

        let nonce = Nonce::from_slice(&encrypted.nonce);

        cipher
            .decrypt(nonce, encrypted.ciphertext.as_ref())
            .map_err(|e| CryptoError::DecryptionFailed(e.to_string()))
    }

    /// Encrypt a string and return base64-encoded result.
    pub fn encrypt_string(&self, plaintext: &str) -> Result<String, CryptoError> {
        if !self.config.enabled {
            return Ok(plaintext.to_string());
        }

        let encrypted = self.encrypt(plaintext.as_bytes())?;
        Ok(encrypted.to_base64())
    }

    /// Decrypt a base64-encoded string.
    pub fn decrypt_string(&self, encoded: &str) -> Result<String, CryptoError> {
        if !self.config.enabled {
            return Ok(encoded.to_string());
        }

        let encrypted = EncryptedData::from_base64(encoded)?;
        let plaintext = self.decrypt(&encrypted)?;
        String::from_utf8(plaintext)
            .map_err(|e| CryptoError::DecryptionFailed(format!("invalid UTF-8: {e}")))
    }

    /// Encrypt JSON metadata.
    pub fn encrypt_metadata(
        &self,
        metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> Result<String, CryptoError> {
        if !self.config.enabled {
            return serde_json::to_string(metadata)
                .map_err(|e| CryptoError::EncryptionFailed(e.to_string()));
        }

        let json = serde_json::to_string(metadata)
            .map_err(|e| CryptoError::EncryptionFailed(e.to_string()))?;
        self.encrypt_string(&json)
    }

    /// Decrypt JSON metadata.
    pub fn decrypt_metadata(
        &self,
        encoded: &str,
    ) -> Result<std::collections::HashMap<String, serde_json::Value>, CryptoError> {
        if !self.config.enabled {
            return serde_json::from_str(encoded)
                .map_err(|e| CryptoError::DecryptionFailed(e.to_string()));
        }

        let json = self.decrypt_string(encoded)?;
        serde_json::from_str(&json).map_err(|e| CryptoError::DecryptionFailed(e.to_string()))
    }
}

impl Default for KeyManager {
    fn default() -> Self {
        Self::new(EncryptionConfig::default())
    }
}

/// Marker for encrypted content in storage.
pub const ENCRYPTED_PREFIX: &str = "enc:v1:";

/// Check if a string is encrypted.
pub fn is_encrypted(s: &str) -> bool {
    s.starts_with(ENCRYPTED_PREFIX)
}

/// Wrap encrypted data with prefix.
pub fn wrap_encrypted(encoded: &str) -> String {
    format!("{ENCRYPTED_PREFIX}{encoded}")
}

/// Unwrap encrypted data, removing prefix.
pub fn unwrap_encrypted(s: &str) -> Option<&str> {
    s.strip_prefix(ENCRYPTED_PREFIX)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_key_manager() -> KeyManager {
        let mut manager = KeyManager::new(EncryptionConfig {
            enabled: true,
            ..Default::default()
        });
        manager
            .derive_master_key("test-password", b"test-salt")
            .unwrap();
        manager
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let manager = test_key_manager();
        let plaintext = "Hello, Helix!";

        let encrypted = manager.encrypt(plaintext.as_bytes()).unwrap();
        let decrypted = manager.decrypt(&encrypted).unwrap();

        assert_eq!(decrypted, plaintext.as_bytes());
    }

    #[test]
    fn test_encrypt_decrypt_string_roundtrip() {
        let manager = test_key_manager();
        let plaintext = "This is sensitive content.";

        let encrypted = manager.encrypt_string(plaintext).unwrap();
        assert_ne!(encrypted, plaintext);

        let decrypted = manager.decrypt_string(&encrypted).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_base64_encoding() {
        let manager = test_key_manager();
        let plaintext = "Test data for base64 encoding";

        let encrypted = manager.encrypt(plaintext.as_bytes()).unwrap();
        let encoded = encrypted.to_base64();

        let decoded = EncryptedData::from_base64(&encoded).unwrap();
        let decrypted = manager.decrypt(&decoded).unwrap();

        assert_eq!(decrypted, plaintext.as_bytes());
    }

    #[test]
    fn test_encryption_disabled() {
        let manager = KeyManager::new(EncryptionConfig {
            enabled: false,
            ..Default::default()
        });

        let plaintext = "Not encrypted";
        let result = manager.encrypt_string(plaintext).unwrap();
        assert_eq!(result, plaintext);

        let decrypted = manager.decrypt_string(plaintext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_metadata_encryption() {
        let manager = test_key_manager();
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "secret".to_string(),
            serde_json::Value::String("classified".to_string()),
        );

        let encrypted = manager.encrypt_metadata(&metadata).unwrap();
        let decrypted = manager.decrypt_metadata(&encrypted).unwrap();

        assert_eq!(metadata, decrypted);
    }

    #[test]
    fn test_encrypted_marker() {
        assert!(is_encrypted("enc:v1:base64data"));
        assert!(!is_encrypted("plain text"));

        let wrapped = wrap_encrypted("base64data");
        assert_eq!(wrapped, "enc:v1:base64data");

        let unwrapped = unwrap_encrypted(&wrapped);
        assert_eq!(unwrapped, Some("base64data"));
    }
}
