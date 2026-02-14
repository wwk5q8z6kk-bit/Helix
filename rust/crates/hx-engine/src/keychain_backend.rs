//! `CredentialBackend` implementation backed by the Sovereign Keychain vault.
//!
//! When the vault is unsealed, credentials in the "api-keys" domain are
//! resolved first in the credential chain, giving priority to vault-stored
//! secrets over OS keyring and environment variables.
//!
//! When sealed, all operations return `Ok(None)` / `Ok(())` to let the
//! chain fall through to lower-priority backends.

use std::sync::Arc;

use hx_core::credentials::{CredentialBackend, CredentialError, SecretSource};
use uuid::Uuid;

use crate::keychain::KeychainEngine;

const API_KEYS_DOMAIN: &str = "api-keys";

/// A credential backend that resolves secrets from the Sovereign Keychain vault.
pub struct KeychainBackend {
    keychain: Arc<KeychainEngine>,
    runtime: tokio::runtime::Handle,
    /// Cached domain id for "api-keys". Lazily populated.
    api_keys_domain: std::sync::Mutex<Option<Uuid>>,
}

impl KeychainBackend {
    pub fn new(keychain: Arc<KeychainEngine>, runtime: tokio::runtime::Handle) -> Self {
        Self {
            keychain,
            runtime,
            api_keys_domain: std::sync::Mutex::new(None),
        }
    }

    /// Resolve the "api-keys" domain id, creating it on first use if needed.
    fn resolve_domain(&self) -> Result<Option<Uuid>, CredentialError> {
        // Fast path: check cache
        {
            let cached = self.api_keys_domain.lock().unwrap();
            if let Some(id) = *cached {
                return Ok(Some(id));
            }
        }

        // Vault must be unsealed
        if !self.keychain.is_unsealed_sync() {
            return Ok(None);
        }

        // Slow path: look up or create the domain (async via block_on)
        let kc = Arc::clone(&self.keychain);
        let domain_id = self
            .runtime
            .block_on(async move { kc.find_or_create_domain(API_KEYS_DOMAIN, "system").await })
            .map_err(|e| CredentialError::EncryptedFile(e.to_string()))?;

        // Cache it
        {
            let mut cached = self.api_keys_domain.lock().unwrap();
            *cached = Some(domain_id);
        }

        Ok(Some(domain_id))
    }
}

impl CredentialBackend for KeychainBackend {
    fn name(&self) -> &str {
        "Sovereign Keychain"
    }

    fn source(&self) -> SecretSource {
        SecretSource::EncryptedFile
    }

    fn is_available(&self) -> bool {
        self.keychain.is_unsealed_sync()
    }

    fn get(&self, key: &str) -> Result<Option<String>, CredentialError> {
        if !self.keychain.is_unsealed_sync() {
            return Ok(None);
        }

        let domain_id = match self.resolve_domain()? {
            Some(id) => id,
            None => return Ok(None),
        };

        let kc = Arc::clone(&self.keychain);
        let key_owned = key.to_string();
        let result = self
            .runtime
            .block_on(async move { kc.read_credential_by_name(domain_id, &key_owned).await });

        match result {
            Ok(Some((_cred, plaintext))) => {
                let value = String::from_utf8(plaintext.to_vec())
                    .map_err(|e| CredentialError::EncryptedFile(format!("invalid utf8: {e}")))?;
                Ok(Some(value))
            }
            Ok(None) => Ok(None),
            Err(hx_core::HxError::VaultSealed) => Ok(None),
            Err(e) => Err(CredentialError::EncryptedFile(e.to_string())),
        }
    }

    fn set(&self, key: &str, value: &str) -> Result<(), CredentialError> {
        if !self.keychain.is_unsealed_sync() {
            return Err(CredentialError::EncryptedFile("vault is sealed".to_string()));
        }

        let domain_id = self
            .resolve_domain()?
            .ok_or_else(|| CredentialError::EncryptedFile("vault is sealed".to_string()))?;

        let kc = Arc::clone(&self.keychain);
        let key_owned = key.to_string();
        let value_owned = value.to_string();
        self.runtime
            .block_on(async move {
                kc.store_credential(
                    domain_id,
                    &key_owned,
                    "api_key",
                    value_owned.as_bytes(),
                    vec![],
                    None,
                    "system",
                )
                .await
            })
            .map_err(|e| CredentialError::EncryptedFile(e.to_string()))?;

        Ok(())
    }

    fn delete(&self, key: &str) -> Result<(), CredentialError> {
        if !self.keychain.is_unsealed_sync() {
            return Err(CredentialError::EncryptedFile("vault is sealed".to_string()));
        }

        let domain_id = match self.resolve_domain()? {
            Some(id) => id,
            None => return Ok(()),
        };

        let kc = Arc::clone(&self.keychain);
        let key_owned = key.to_string();
        let result = self
            .runtime
            .block_on(async move { kc.read_credential_by_name(domain_id, &key_owned).await });

        match result {
            Ok(Some((cred, _))) => {
                let kc = Arc::clone(&self.keychain);
                self.runtime
                    .block_on(async move { kc.destroy_credential(cred.id, "system").await })
                    .map_err(|e| CredentialError::EncryptedFile(e.to_string()))?;
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(e) => Err(CredentialError::EncryptedFile(e.to_string())),
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn list_keys(&self) -> Result<Vec<String>, CredentialError> {
        if !self.keychain.is_unsealed_sync() {
            return Ok(vec![]);
        }

        let domain_id = match self.resolve_domain()? {
            Some(id) => id,
            None => return Ok(vec![]),
        };

        let kc = Arc::clone(&self.keychain);
        let creds = self
            .runtime
            .block_on(async move {
                kc.list_credentials(Some(domain_id), Some(hx_core::model::keychain::CredentialState::Active), 1000, 0)
                    .await
            })
            .map_err(|e| CredentialError::EncryptedFile(e.to_string()))?;

        Ok(creds.into_iter().map(|c| c.name).collect())
    }
}
