use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Where a secret was resolved from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretSource {
    OsKeyring,
    EncryptedFile,
    EnvironmentVariable,
}

impl fmt::Display for SecretSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OsKeyring => write!(f, "os_keyring"),
            Self::EncryptedFile => write!(f, "encrypted_file"),
            Self::EnvironmentVariable => write!(f, "env"),
        }
    }
}

/// A resolved secret value. The inner string is zeroized on drop.
pub struct SecretValue {
    inner: Zeroizing<String>,
    source: SecretSource,
}

impl SecretValue {
    pub fn new(value: String, source: SecretSource) -> Self {
        Self {
            inner: Zeroizing::new(value),
            source,
        }
    }

    /// Expose the secret for use (e.g., passing to an HTTP header).
    pub fn expose(&self) -> &str {
        &self.inner
    }

    /// Which backend resolved this secret.
    pub fn source(&self) -> SecretSource {
        self.source
    }
}

impl fmt::Debug for SecretValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecretValue")
            .field("source", &self.source)
            .field("inner", &"[REDACTED]")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum CredentialError {
    #[error("keyring error: {0}")]
    Keyring(String),
    #[error("encrypted file error: {0}")]
    EncryptedFile(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("{0}")]
    Other(String),
}

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// A secret storage backend.
pub trait CredentialBackend: Send + Sync {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Which source type this backend represents.
    fn source(&self) -> SecretSource;

    /// Whether this backend is currently available.
    fn is_available(&self) -> bool;

    /// Retrieve a secret by key. Returns `Ok(None)` if not found.
    fn get(&self, key: &str) -> Result<Option<String>, CredentialError>;

    /// Store a secret.
    fn set(&self, key: &str, value: &str) -> Result<(), CredentialError>;

    /// Delete a secret.
    fn delete(&self, key: &str) -> Result<(), CredentialError>;

    /// List stored key names.
    fn list_keys(&self) -> Result<Vec<String>, CredentialError>;

    /// Downcast support for concrete backend access.
    fn as_any(&self) -> &dyn std::any::Any;
}

// ---------------------------------------------------------------------------
// Keyring backend
// ---------------------------------------------------------------------------

pub struct KeyringBackend {
    service: String,
}

impl KeyringBackend {
    pub fn new(service: &str) -> Self {
        Self {
            service: service.to_string(),
        }
    }

    fn entry(&self, key: &str) -> Result<keyring::Entry, CredentialError> {
        keyring::Entry::new(&self.service, key).map_err(|e| CredentialError::Keyring(e.to_string()))
    }
}

/// Well-known secret keys that Helix uses.
const KNOWN_SECRET_KEYS: &[&str] = &[
    "OPENAI_API_KEY",
    "HELIX_EMBEDDING_API_KEY",
    "HELIX_ENCRYPTION_KEY",
    "HELIX_EMAIL_IMAP_PASSWORD",
    "HELIX_EMAIL_SMTP_PASSWORD",
];

impl CredentialBackend for KeyringBackend {
    fn name(&self) -> &str {
        "OS Keyring"
    }

    fn source(&self) -> SecretSource {
        SecretSource::OsKeyring
    }

    fn is_available(&self) -> bool {
        // Try creating an entry to see if the keyring service is reachable.
        self.entry("__helix_probe").is_ok()
    }

    fn get(&self, key: &str) -> Result<Option<String>, CredentialError> {
        let entry = self.entry(key)?;
        match entry.get_password() {
            Ok(pw) => Ok(Some(pw)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(CredentialError::Keyring(e.to_string())),
        }
    }

    fn set(&self, key: &str, value: &str) -> Result<(), CredentialError> {
        let entry = self.entry(key)?;
        entry
            .set_password(value)
            .map_err(|e| CredentialError::Keyring(e.to_string()))
    }

    fn delete(&self, key: &str) -> Result<(), CredentialError> {
        let entry = self.entry(key)?;
        match entry.delete_credential() {
            Ok(()) => Ok(()),
            Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(CredentialError::Keyring(e.to_string())),
        }
    }

    fn list_keys(&self) -> Result<Vec<String>, CredentialError> {
        // The keyring crate doesn't support enumeration.
        // Probe known keys to see which ones are stored.
        let mut found = Vec::new();
        for &key in KNOWN_SECRET_KEYS {
            if let Ok(Some(_)) = self.get(key) {
                found.push(key.to_string());
            }
        }
        Ok(found)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Environment variable backend
// ---------------------------------------------------------------------------

pub struct EnvBackend;

impl CredentialBackend for EnvBackend {
    fn name(&self) -> &str {
        "Environment"
    }

    fn source(&self) -> SecretSource {
        SecretSource::EnvironmentVariable
    }

    fn is_available(&self) -> bool {
        true
    }

    fn get(&self, key: &str) -> Result<Option<String>, CredentialError> {
        match std::env::var(key) {
            Ok(v) if !v.is_empty() => Ok(Some(v)),
            _ => Ok(None),
        }
    }

    fn set(&self, _key: &str, _value: &str) -> Result<(), CredentialError> {
        Err(CredentialError::Other(
            "cannot persist secrets to environment variables".into(),
        ))
    }

    fn delete(&self, _key: &str) -> Result<(), CredentialError> {
        Err(CredentialError::Other(
            "cannot delete environment variables".into(),
        ))
    }

    fn list_keys(&self) -> Result<Vec<String>, CredentialError> {
        let mut found = Vec::new();
        for &key in KNOWN_SECRET_KEYS {
            if std::env::var(key).map(|v| !v.is_empty()).unwrap_or(false) {
                found.push(key.to_string());
            }
        }
        Ok(found)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Encrypted file backend
// ---------------------------------------------------------------------------

const FILE_VERSION: u8 = 1;
const SALT_SIZE: usize = 16;
const NONCE_SIZE: usize = 12;
const KEY_SIZE: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argon2Params {
    pub memory_kib: u32,
    pub iterations: u32,
    pub parallelism: u32,
}

impl Default for Argon2Params {
    fn default() -> Self {
        Self {
            memory_kib: 65536,
            iterations: 3,
            parallelism: 4,
        }
    }
}

/// On-disk JSON envelope for the encrypted secrets file.
#[derive(Serialize, Deserialize)]
struct EncryptedFileEnvelope {
    version: u8,
    argon2: Argon2Params,
    salt: String,   // base64
    data: String,   // base64 of [0x01 || nonce(12) || AES-256-GCM(json_map)]
}

enum FileBackendState {
    Locked,
    Unlocked {
        derived_key: Zeroizing<[u8; KEY_SIZE]>,
        secrets: HashMap<String, String>,
        salt: [u8; SALT_SIZE],
        argon2_params: Argon2Params,
    },
}

/// Encrypted file credential backend.
///
/// Stores all secrets as a JSON map encrypted with AES-256-GCM. The encryption
/// key is derived from a master password using Argon2id. The backend starts in
/// a `Locked` state and must be unlocked with the master password before
/// secrets can be read or written.
pub struct EncryptedFileBackend {
    path: PathBuf,
    state: Mutex<FileBackendState>,
}

impl EncryptedFileBackend {
    /// Create a backend pointing at the given file. Starts in `Locked` state.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            state: Mutex::new(FileBackendState::Locked),
        }
    }

    /// Create a new encrypted secrets file with an empty map.
    pub fn init(path: &Path, password: &str) -> Result<(), CredentialError> {
        Self::init_with_params(path, password, Argon2Params::default())
    }

    /// Create a new encrypted secrets file with custom Argon2 parameters.
    pub fn init_with_params(
        path: &Path,
        password: &str,
        params: Argon2Params,
    ) -> Result<(), CredentialError> {
        if path.exists() {
            return Err(CredentialError::EncryptedFile(
                "secrets file already exists — delete it first to re-initialize".into(),
            ));
        }

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                CredentialError::EncryptedFile(format!("create directory: {e}"))
            })?;
        }

        let mut salt = [0u8; SALT_SIZE];
        OsRng.fill_bytes(&mut salt);

        let key = derive_key(password, &salt, &params)?;
        let empty_map: HashMap<String, String> = HashMap::new();
        let plaintext = serde_json::to_vec(&empty_map)
            .map_err(|e| CredentialError::EncryptedFile(format!("serialize: {e}")))?;

        let blob = encrypt_blob(&key, &plaintext)?;

        let envelope = EncryptedFileEnvelope {
            version: FILE_VERSION,
            argon2: params,
            salt: BASE64.encode(salt),
            data: BASE64.encode(blob),
        };

        let json = serde_json::to_string_pretty(&envelope)
            .map_err(|e| CredentialError::EncryptedFile(format!("serialize envelope: {e}")))?;

        atomic_write(path, json.as_bytes())?;
        Ok(())
    }

    /// Decrypt the file and populate the in-memory map.
    pub fn unlock(&self, password: &str) -> Result<(), CredentialError> {
        let file_data = std::fs::read(&self.path).map_err(|e| {
            CredentialError::EncryptedFile(format!("read {}: {e}", self.path.display()))
        })?;

        let envelope: EncryptedFileEnvelope = serde_json::from_slice(&file_data).map_err(|e| {
            CredentialError::EncryptedFile(format!("parse envelope: {e}"))
        })?;

        if envelope.version != FILE_VERSION {
            return Err(CredentialError::EncryptedFile(format!(
                "unsupported file version: {}",
                envelope.version
            )));
        }

        let salt_bytes = BASE64.decode(&envelope.salt).map_err(|e| {
            CredentialError::EncryptedFile(format!("decode salt: {e}"))
        })?;
        if salt_bytes.len() != SALT_SIZE {
            return Err(CredentialError::EncryptedFile("invalid salt length".into()));
        }
        let mut salt = [0u8; SALT_SIZE];
        salt.copy_from_slice(&salt_bytes);

        let key = derive_key(password, &salt, &envelope.argon2)?;

        let blob = BASE64.decode(&envelope.data).map_err(|e| {
            CredentialError::EncryptedFile(format!("decode data: {e}"))
        })?;

        let plaintext = decrypt_blob(&key, &blob)?;

        let secrets: HashMap<String, String> = serde_json::from_slice(&plaintext).map_err(|e| {
            CredentialError::EncryptedFile(format!(
                "decrypt succeeded but JSON is invalid (wrong password?): {e}"
            ))
        })?;

        let mut state = self.state.lock().map_err(|e| {
            CredentialError::EncryptedFile(format!("lock poisoned: {e}"))
        })?;
        *state = FileBackendState::Unlocked {
            derived_key: key,
            secrets,
            salt,
            argon2_params: envelope.argon2,
        };

        Ok(())
    }

    /// Zeroize the key and secrets, returning to `Locked` state.
    pub fn lock(&self) -> Result<(), CredentialError> {
        let mut state = self.state.lock().map_err(|e| {
            CredentialError::EncryptedFile(format!("lock poisoned: {e}"))
        })?;
        *state = FileBackendState::Locked;
        Ok(())
    }

    /// Whether the backend is currently unlocked.
    pub fn is_unlocked(&self) -> bool {
        let state = self.state.lock().ok();
        matches!(state.as_deref(), Some(FileBackendState::Unlocked { .. }))
    }

    /// Re-encrypt and write to disk. Must be called while holding the state lock.
    fn flush_inner(
        path: &Path,
        key: &[u8; KEY_SIZE],
        secrets: &HashMap<String, String>,
        salt: &[u8; SALT_SIZE],
        params: &Argon2Params,
    ) -> Result<(), CredentialError> {
        let plaintext = serde_json::to_vec(secrets)
            .map_err(|e| CredentialError::EncryptedFile(format!("serialize: {e}")))?;

        let blob = encrypt_blob(key, &plaintext)?;

        let envelope = EncryptedFileEnvelope {
            version: FILE_VERSION,
            argon2: params.clone(),
            salt: BASE64.encode(salt),
            data: BASE64.encode(blob),
        };

        let json = serde_json::to_string_pretty(&envelope)
            .map_err(|e| CredentialError::EncryptedFile(format!("serialize envelope: {e}")))?;

        atomic_write(path, json.as_bytes())
    }
}

impl CredentialBackend for EncryptedFileBackend {
    fn name(&self) -> &str {
        "Encrypted File"
    }

    fn source(&self) -> SecretSource {
        SecretSource::EncryptedFile
    }

    fn is_available(&self) -> bool {
        self.path.exists()
    }

    fn get(&self, key: &str) -> Result<Option<String>, CredentialError> {
        let state = self.state.lock().map_err(|e| {
            CredentialError::EncryptedFile(format!("lock poisoned: {e}"))
        })?;
        match &*state {
            FileBackendState::Locked => Ok(None), // silently skip
            FileBackendState::Unlocked { secrets, .. } => Ok(secrets.get(key).cloned()),
        }
    }

    fn set(&self, key: &str, value: &str) -> Result<(), CredentialError> {
        let mut state = self.state.lock().map_err(|e| {
            CredentialError::EncryptedFile(format!("lock poisoned: {e}"))
        })?;
        match &mut *state {
            FileBackendState::Locked => Err(CredentialError::EncryptedFile(
                "encrypted file backend is locked — unlock first".into(),
            )),
            FileBackendState::Unlocked {
                derived_key,
                secrets,
                salt,
                argon2_params,
            } => {
                secrets.insert(key.to_string(), value.to_string());
                Self::flush_inner(&self.path, derived_key, secrets, salt, argon2_params)
            }
        }
    }

    fn delete(&self, key: &str) -> Result<(), CredentialError> {
        let mut state = self.state.lock().map_err(|e| {
            CredentialError::EncryptedFile(format!("lock poisoned: {e}"))
        })?;
        match &mut *state {
            FileBackendState::Locked => Err(CredentialError::EncryptedFile(
                "encrypted file backend is locked — unlock first".into(),
            )),
            FileBackendState::Unlocked {
                derived_key,
                secrets,
                salt,
                argon2_params,
            } => {
                secrets.remove(key);
                Self::flush_inner(&self.path, derived_key, secrets, salt, argon2_params)
            }
        }
    }

    fn list_keys(&self) -> Result<Vec<String>, CredentialError> {
        let state = self.state.lock().map_err(|e| {
            CredentialError::EncryptedFile(format!("lock poisoned: {e}"))
        })?;
        match &*state {
            FileBackendState::Locked => Ok(Vec::new()),
            FileBackendState::Unlocked { secrets, .. } => {
                let mut keys: Vec<String> = secrets.keys().cloned().collect();
                keys.sort();
                Ok(keys)
            }
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// ---------------------------------------------------------------------------
// Crypto helpers
// ---------------------------------------------------------------------------

fn derive_key(
    password: &str,
    salt: &[u8],
    params: &Argon2Params,
) -> Result<Zeroizing<[u8; KEY_SIZE]>, CredentialError> {
    let argon2 = argon2::Argon2::new(
        argon2::Algorithm::Argon2id,
        argon2::Version::V0x13,
        argon2::Params::new(params.memory_kib, params.iterations, params.parallelism, Some(KEY_SIZE))
            .map_err(|e| CredentialError::EncryptedFile(format!("argon2 params: {e}")))?,
    );
    let mut key = Zeroizing::new([0u8; KEY_SIZE]);
    argon2
        .hash_password_into(password.as_bytes(), salt, key.as_mut())
        .map_err(|e| CredentialError::EncryptedFile(format!("key derivation: {e}")))?;
    Ok(key)
}

/// Encrypt plaintext: `[0x01 || nonce(12) || AES-256-GCM(plaintext)]`
fn encrypt_blob(key: &[u8; KEY_SIZE], plaintext: &[u8]) -> Result<Vec<u8>, CredentialError> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| CredentialError::EncryptedFile(format!("cipher init: {e}")))?;

    let mut nonce_bytes = [0u8; NONCE_SIZE];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| CredentialError::EncryptedFile(format!("encrypt: {e}")))?;

    let mut blob = Vec::with_capacity(1 + NONCE_SIZE + ciphertext.len());
    blob.push(0x01); // blob version tag
    blob.extend_from_slice(&nonce_bytes);
    blob.extend_from_slice(&ciphertext);
    Ok(blob)
}

/// Decrypt: expects `[0x01 || nonce(12) || ciphertext+tag]`
fn decrypt_blob(key: &[u8; KEY_SIZE], data: &[u8]) -> Result<Vec<u8>, CredentialError> {
    if data.len() < 1 + NONCE_SIZE + 16 {
        return Err(CredentialError::EncryptedFile("encrypted data too short".into()));
    }
    if data[0] != 0x01 {
        return Err(CredentialError::EncryptedFile(format!(
            "unsupported blob version: {}",
            data[0]
        )));
    }

    let nonce_bytes = &data[1..1 + NONCE_SIZE];
    let ciphertext = &data[1 + NONCE_SIZE..];

    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| CredentialError::EncryptedFile(format!("cipher init: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| CredentialError::EncryptedFile("decryption failed — wrong password?".into()))
}

/// Atomic write: write to a temp file, then rename.
fn atomic_write(path: &Path, data: &[u8]) -> Result<(), CredentialError> {
    let tmp_path = path.with_extension("tmp");
    std::fs::write(&tmp_path, data).map_err(|e| {
        CredentialError::EncryptedFile(format!("write temp file: {e}"))
    })?;
    std::fs::rename(&tmp_path, path).map_err(|e| {
        CredentialError::EncryptedFile(format!("rename: {e}"))
    })?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Credential store (resolution chain)
// ---------------------------------------------------------------------------

/// Status of a single backend in the credential store.
#[derive(Debug, Clone)]
pub struct BackendStatus {
    pub name: String,
    pub source: SecretSource,
    pub available: bool,
    pub keys: Vec<String>,
}

/// The credential store resolves secrets through a prioritized backend chain.
pub struct CredentialStore {
    backends: Vec<Box<dyn CredentialBackend>>,
}

impl CredentialStore {
    /// Create a new credential store with the default backend chain:
    /// OS Keyring → Encrypted File → Environment Variables.
    pub fn new(service_name: &str) -> Self {
        let mut backends: Vec<Box<dyn CredentialBackend>> = Vec::new();

        let keyring = KeyringBackend::new(service_name);
        if keyring.is_available() {
            backends.push(Box::new(keyring));
        } else {
            tracing::warn!("OS keyring not available, skipping keyring backend");
        }

        // Insert encrypted file backend (between keyring and env) if file exists.
        let secrets_path = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(".helix")
            .join("secrets.enc");
        if secrets_path.exists() {
            tracing::info!(
                path = %secrets_path.display(),
                "encrypted file backend available (locked)"
            );
            backends.push(Box::new(EncryptedFileBackend::new(secrets_path)));
        }

        backends.push(Box::new(EnvBackend));

        Self { backends }
    }

    /// Create a store with only the environment backend (for testing).
    pub fn env_only() -> Self {
        Self {
            backends: vec![Box::new(EnvBackend)],
        }
    }

    /// Insert a backend at a specific position in the resolution chain.
    /// Index 0 = highest priority (checked first).
    pub fn insert_backend(&mut self, index: usize, backend: Box<dyn CredentialBackend>) {
        let idx = index.min(self.backends.len());
        self.backends.insert(idx, backend);
    }

    /// Resolve a secret by walking the backend chain (highest priority first).
    pub fn get(&self, key: &str) -> Result<Option<SecretValue>, CredentialError> {
        for backend in &self.backends {
            match backend.get(key) {
                Ok(Some(value)) => {
                    tracing::info!(
                        key = key,
                        source = %backend.source(),
                        "helix_credential_resolved"
                    );
                    return Ok(Some(SecretValue::new(value, backend.source())));
                }
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(
                        key = key,
                        backend = backend.name(),
                        error = %e,
                        "helix_credential_backend_error"
                    );
                    continue;
                }
            }
        }
        Ok(None)
    }

    /// Store a secret in the first writable backend.
    pub fn set(&self, key: &str, value: &str) -> Result<SecretSource, CredentialError> {
        for backend in &self.backends {
            match backend.set(key, value) {
                Ok(()) => return Ok(backend.source()),
                Err(_) => continue,
            }
        }
        Err(CredentialError::Other(
            "no writable backend available".into(),
        ))
    }

    /// Store a secret in a specific backend.
    pub fn set_in(
        &self,
        key: &str,
        value: &str,
        target: SecretSource,
    ) -> Result<(), CredentialError> {
        for backend in &self.backends {
            if backend.source() == target {
                return backend.set(key, value);
            }
        }
        Err(CredentialError::Other(format!(
            "backend {target} not available"
        )))
    }

    /// Delete a secret from all backends that have it.
    pub fn delete(&self, key: &str) -> Result<Vec<SecretSource>, CredentialError> {
        let mut deleted_from = Vec::new();
        for backend in &self.backends {
            if let Ok(Some(_)) = backend.get(key) {
                backend.delete(key)?;
                deleted_from.push(backend.source());
            }
        }
        Ok(deleted_from)
    }

    /// Status of all backends.
    pub fn status(&self) -> Vec<BackendStatus> {
        self.backends
            .iter()
            .map(|b| BackendStatus {
                name: b.name().to_string(),
                source: b.source(),
                available: b.is_available(),
                keys: b.list_keys().unwrap_or_default(),
            })
            .collect()
    }

    /// Attempt to unlock the encrypted file backend with the given password.
    /// Returns `Ok(true)` if unlocked, `Ok(false)` if no encrypted file backend exists.
    pub fn unlock_encrypted_file(&self, password: &str) -> Result<bool, CredentialError> {
        for backend in &self.backends {
            if backend.source() == SecretSource::EncryptedFile {
                // Downcast to EncryptedFileBackend. We know the concrete type because
                // we inserted it ourselves in new(). Use Any for safe downcast.
                let any = backend.as_any();
                if let Some(efb) = any.downcast_ref::<EncryptedFileBackend>() {
                    efb.unlock(password)?;
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Convenience: resolve an API key, returning just the string (for engine integration).
    pub fn get_secret_string(&self, key: &str) -> Option<String> {
        self.get(key)
            .ok()
            .flatten()
            .map(|sv| sv.expose().to_string())
    }

    /// Like `get_secret_string`, but wraps the result in `Zeroizing<String>` so the
    /// plaintext is scrubbed from memory on drop. Prefer this for new code paths.
    pub fn get_secret_zeroized(&self, key: &str) -> Option<Zeroizing<String>> {
        self.get(key)
            .ok()
            .flatten()
            .map(|sv| Zeroizing::new(sv.expose().to_string()))
    }
}

impl fmt::Debug for CredentialStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CredentialStore")
            .field("backends", &self.backends.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_backend_reads_set_var() {
        let backend = EnvBackend;
        std::env::set_var("MV_TEST_CRED_1", "test_value_123");
        let val = backend.get("MV_TEST_CRED_1").unwrap();
        assert_eq!(val.as_deref(), Some("test_value_123"));
        std::env::remove_var("MV_TEST_CRED_1");
    }

    #[test]
    fn env_backend_returns_none_for_missing() {
        let backend = EnvBackend;
        std::env::remove_var("MV_TEST_CRED_MISSING");
        let val = backend.get("MV_TEST_CRED_MISSING").unwrap();
        assert!(val.is_none());
    }

    #[test]
    fn env_backend_returns_none_for_empty() {
        let backend = EnvBackend;
        std::env::set_var("MV_TEST_CRED_EMPTY", "");
        let val = backend.get("MV_TEST_CRED_EMPTY").unwrap();
        assert!(val.is_none());
        std::env::remove_var("MV_TEST_CRED_EMPTY");
    }

    #[test]
    fn env_backend_cannot_set() {
        let backend = EnvBackend;
        assert!(backend.set("X", "Y").is_err());
    }

    #[test]
    fn credential_store_env_only_resolves() {
        std::env::set_var("MV_TEST_CRED_STORE", "from_env");
        let store = CredentialStore::env_only();
        let sv = store.get("MV_TEST_CRED_STORE").unwrap().unwrap();
        assert_eq!(sv.expose(), "from_env");
        assert_eq!(sv.source(), SecretSource::EnvironmentVariable);
        std::env::remove_var("MV_TEST_CRED_STORE");
    }

    #[test]
    fn credential_store_returns_none_when_missing() {
        std::env::remove_var("MV_TEST_CRED_ABSENT");
        let store = CredentialStore::env_only();
        assert!(store.get("MV_TEST_CRED_ABSENT").unwrap().is_none());
    }

    #[test]
    fn secret_value_debug_redacts() {
        let sv = SecretValue::new("super_secret".into(), SecretSource::OsKeyring);
        let debug = format!("{sv:?}");
        assert!(!debug.contains("super_secret"));
        assert!(debug.contains("REDACTED"));
    }

    // --- Encrypted file backend tests ---

    use std::sync::atomic::{AtomicU64, Ordering};

    fn temp_secrets_path() -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let mut path = std::env::temp_dir();
        path.push(format!(
            "hx_test_secrets_{}_{}.enc",
            std::process::id(),
            id,
        ));
        // Clean up any leftover from a previous test run
        let _ = std::fs::remove_file(&path);
        path
    }

    /// Low-cost Argon2 params for fast tests.
    fn test_params() -> Argon2Params {
        Argon2Params {
            memory_kib: 256,
            iterations: 1,
            parallelism: 1,
        }
    }

    fn init_test_file(path: &Path, password: &str) {
        EncryptedFileBackend::init_with_params(path, password, test_params()).unwrap();
    }

    #[test]
    fn encrypted_file_init_creates_file() {
        let path = temp_secrets_path();
        init_test_file(&path, "test-password");
        assert!(path.exists());

        // Verify the file is valid JSON with expected fields
        let data = std::fs::read_to_string(&path).unwrap();
        let envelope: serde_json::Value = serde_json::from_str(&data).unwrap();
        assert_eq!(envelope["version"], 1);
        assert!(envelope["argon2"]["memory_kib"].is_number());
        assert!(envelope["salt"].is_string());
        assert!(envelope["data"].is_string());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn encrypted_file_init_rejects_existing() {
        let path = temp_secrets_path();
        init_test_file(&path, "pw");
        let result = EncryptedFileBackend::init(&path, "pw");
        assert!(result.is_err());
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn encrypted_file_unlock_and_read_empty() {
        let path = temp_secrets_path();
        init_test_file(&path, "pw123");

        let backend = EncryptedFileBackend::new(path.clone());
        assert!(backend.is_available());
        assert!(!backend.is_unlocked());

        // While locked, get returns None (not error)
        assert!(backend.get("ANY_KEY").unwrap().is_none());
        assert!(backend.list_keys().unwrap().is_empty());

        // Unlock
        backend.unlock("pw123").unwrap();
        assert!(backend.is_unlocked());

        // Still empty
        assert!(backend.get("ANY_KEY").unwrap().is_none());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn encrypted_file_wrong_password() {
        let path = temp_secrets_path();
        init_test_file(&path, "correct");

        let backend = EncryptedFileBackend::new(path.clone());
        let result = backend.unlock("wrong");
        assert!(result.is_err());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn encrypted_file_set_get_delete() {
        let path = temp_secrets_path();
        init_test_file(&path, "pass");

        let backend = EncryptedFileBackend::new(path.clone());
        backend.unlock("pass").unwrap();

        // Set
        backend.set("API_KEY", "sk-1234").unwrap();
        assert_eq!(backend.get("API_KEY").unwrap().as_deref(), Some("sk-1234"));

        // List
        let keys = backend.list_keys().unwrap();
        assert_eq!(keys, vec!["API_KEY".to_string()]);

        // Persistence: create a new backend pointing to the same file
        let backend2 = EncryptedFileBackend::new(path.clone());
        backend2.unlock("pass").unwrap();
        assert_eq!(backend2.get("API_KEY").unwrap().as_deref(), Some("sk-1234"));

        // Delete
        backend2.delete("API_KEY").unwrap();
        assert!(backend2.get("API_KEY").unwrap().is_none());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn encrypted_file_set_while_locked_fails() {
        let path = temp_secrets_path();
        init_test_file(&path, "pw");

        let backend = EncryptedFileBackend::new(path.clone());
        // Don't unlock — set should fail
        let result = backend.set("KEY", "VAL");
        assert!(result.is_err());

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn encrypted_file_lock_clears_state() {
        let path = temp_secrets_path();
        init_test_file(&path, "pw");

        let backend = EncryptedFileBackend::new(path.clone());
        backend.unlock("pw").unwrap();
        backend.set("KEY", "VAL").unwrap();
        assert!(backend.is_unlocked());

        // Lock it
        backend.lock().unwrap();
        assert!(!backend.is_unlocked());

        // After locking, get returns None (not the value)
        assert!(backend.get("KEY").unwrap().is_none());
        assert!(backend.list_keys().unwrap().is_empty());

        // Re-unlock — value should be persisted
        backend.unlock("pw").unwrap();
        assert_eq!(backend.get("KEY").unwrap().as_deref(), Some("VAL"));

        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn encrypted_file_data_not_plaintext_on_disk() {
        let path = temp_secrets_path();
        init_test_file(&path, "pw");

        let backend = EncryptedFileBackend::new(path.clone());
        backend.unlock("pw").unwrap();
        backend.set("MY_SECRET_KEY", "super_secret_value_12345").unwrap();

        let file_contents = std::fs::read_to_string(&path).unwrap();
        assert!(!file_contents.contains("super_secret_value_12345"));
        assert!(!file_contents.contains("MY_SECRET_KEY"));

        std::fs::remove_file(&path).unwrap();
    }
}
