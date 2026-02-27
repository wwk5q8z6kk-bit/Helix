use super::HelixEngine;
use hx_core::{ConsumerProfile, ConsumerStore, HxError, MvResult};
use sha2::{Digest, Sha256};
use uuid::Uuid;

/// Compute SHA-256 and return lowercase hex string (no `hex` crate dependency).
fn sha256_hex(data: &[u8]) -> String {
    let hash = Sha256::digest(data);
    let mut out = String::with_capacity(hash.len() * 2);
    for byte in hash {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}

impl HelixEngine {
    pub async fn get_consumer_profile(&self, id: Uuid) -> MvResult<Option<ConsumerProfile>> {
        self.store.nodes.get_consumer(id).await
    }

    pub async fn update_consumer_profile(
        &self,
        profile: &ConsumerProfile,
    ) -> MvResult<ConsumerProfile> {
        // Touch to update last_used_at
        self.store.nodes.touch_consumer(profile.id).await?;
        // Return the profile as-is (the store doesn't have a full update method)
        Ok(profile.clone())
    }

    pub async fn get_or_create_consumer_profile(
        &self,
        name: &str,
        _email: Option<String>,
    ) -> MvResult<ConsumerProfile> {
        // Try to find by name first
        if let Some(existing) = self.store.nodes.get_consumer_by_name(name).await? {
            return Ok(existing);
        }

        let profile = ConsumerProfile {
            id: Uuid::now_v7(),
            name: name.to_string(),
            description: None,
            token_hash: String::new(),
            created_at: chrono::Utc::now(),
            last_used_at: None,
            revoked_at: None,
            metadata: std::collections::HashMap::new(),
        };

        self.store.nodes.create_consumer(&profile).await?;
        Ok(profile)
    }

    pub async fn list_consumer_profiles(&self) -> MvResult<Vec<ConsumerProfile>> {
        self.store.nodes.list_consumers().await
    }

    pub async fn delete_consumer_profile(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.revoke_consumer(id).await
    }

    pub async fn lookup_consumer_by_identity(
        &self,
        _identity_provider: &str,
        _identity_id: &str,
    ) -> MvResult<Option<ConsumerProfile>> {
        // Stub: no explicit identity table exists yet
        Ok(None)
    }

    /// Create a new consumer with a generated bearer token.
    /// Returns the consumer profile and the raw (unhashed) token.
    pub async fn create_consumer(
        &self,
        name: &str,
        description: Option<&str>,
    ) -> MvResult<(ConsumerProfile, String)> {
        // Check for duplicate name
        if let Some(_existing) = self.store.nodes.get_consumer_by_name(name).await? {
            return Err(HxError::InvalidInput(format!(
                "consumer '{name}' already exists"
            )));
        }

        // Generate a random bearer token
        let raw_token = format!("hx_{}", Uuid::now_v7().simple());
        let token_hash = sha256_hex(raw_token.as_bytes());

        let profile = ConsumerProfile {
            id: Uuid::now_v7(),
            name: name.to_string(),
            description: description.map(|s| s.to_string()),
            token_hash,
            created_at: chrono::Utc::now(),
            last_used_at: None,
            revoked_at: None,
            metadata: std::collections::HashMap::new(),
        };

        self.store.nodes.create_consumer(&profile).await?;
        Ok((profile, raw_token))
    }

    /// Revoke a consumer by ID.
    pub async fn revoke_consumer(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.revoke_consumer(id).await
    }

    /// Resolve a raw bearer token to its consumer profile.
    pub async fn resolve_consumer_token(
        &self,
        token: &str,
    ) -> MvResult<Option<ConsumerProfile>> {
        let token_hash = sha256_hex(token.as_bytes());
        let profile = self.store.nodes.get_consumer_by_token_hash(&token_hash).await?;
        // Only return active (non-revoked) consumers
        Ok(profile.filter(|p| p.is_active()))
    }

    /// List all consumer profiles.
    pub async fn list_consumers(&self) -> MvResult<Vec<ConsumerProfile>> {
        self.store.nodes.list_consumers().await
    }

    /// Get a consumer profile by ID.
    pub async fn get_consumer(&self, id: Uuid) -> MvResult<Option<ConsumerProfile>> {
        self.store.nodes.get_consumer(id).await
    }
}
