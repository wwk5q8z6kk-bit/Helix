use super::HelixEngine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, Utc};
use hx_core::{
    AccessKey, HxError, KnowledgeNode, MvResult, NodeKind, NodeStore, PermissionTemplate,
    PermissionTier, PublicShare, ShareStore,
};
use rand::RngCore;
use sha2::{Digest, Sha256};
use uuid::Uuid;

impl HelixEngine {
    pub(crate) async fn ensure_default_permission_templates(&self) -> MvResult<()> {
        let owner_exists = self
            .store
            .nodes
            .get_permission_template_by_name("Owner")
            .await?
            .is_some();
        let assistant_exists = self
            .store
            .nodes
            .get_permission_template_by_name("Assistant")
            .await?
            .is_some();

        if !owner_exists {
            let now = Utc::now();
            let owner = PermissionTemplate {
                id: Uuid::now_v7(),
                name: "Owner".to_string(),
                description: Some("Full access template".to_string()),
                tier: PermissionTier::Admin,
                scope_namespace: None,
                scope_tags: Vec::new(),
                allow_kinds: Vec::new(),
                allow_actions: Vec::new(),
                created_at: now,
                updated_at: now,
            };
            self.store.nodes.insert_permission_template(&owner).await?;
        }

        if !assistant_exists {
            let now = Utc::now();
            let assistant = PermissionTemplate {
                id: Uuid::now_v7(),
                name: "Assistant".to_string(),
                description: Some("Scoped assistant template".to_string()),
                tier: PermissionTier::Action,
                scope_namespace: Some("assistant".to_string()),
                scope_tags: Vec::new(),
                allow_kinds: Vec::new(),
                allow_actions: Vec::new(),
                created_at: now,
                updated_at: now,
            };
            self.store
                .nodes
                .insert_permission_template(&assistant)
                .await?;
        }

        Ok(())
    }

    pub async fn list_permission_templates(
        &self,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<PermissionTemplate>> {
        self.store
            .nodes
            .list_permission_templates(limit, offset)
            .await
    }

    pub async fn create_permission_template(
        &self,
        name: String,
        description: Option<String>,
        tier: PermissionTier,
        scope_namespace: Option<String>,
        scope_tags: Vec<String>,
        allow_kinds: Vec<NodeKind>,
        allow_actions: Vec<String>,
    ) -> MvResult<PermissionTemplate> {
        let now = Utc::now();
        let template = PermissionTemplate {
            id: Uuid::now_v7(),
            name,
            description,
            tier,
            scope_namespace,
            scope_tags,
            allow_kinds,
            allow_actions,
            created_at: now,
            updated_at: now,
        };
        self.store
            .nodes
            .insert_permission_template(&template)
            .await?;
        Ok(template)
    }

    pub async fn update_permission_template(
        &self,
        template_id: Uuid,
        name: String,
        description: Option<String>,
        tier: PermissionTier,
        scope_namespace: Option<String>,
        scope_tags: Vec<String>,
        allow_kinds: Vec<NodeKind>,
        allow_actions: Vec<String>,
    ) -> MvResult<Option<PermissionTemplate>> {
        let mut existing = match self
            .store
            .nodes
            .get_permission_template(template_id)
            .await?
        {
            Some(template) => template,
            None => return Ok(None),
        };

        existing.name = name;
        existing.description = description;
        existing.tier = tier;
        existing.scope_namespace = scope_namespace;
        existing.scope_tags = scope_tags;
        existing.allow_kinds = allow_kinds;
        existing.allow_actions = allow_actions;
        existing.updated_at = Utc::now();

        self.store
            .nodes
            .update_permission_template(&existing)
            .await?;
        Ok(Some(existing))
    }

    pub async fn delete_permission_template(&self, template_id: Uuid) -> MvResult<bool> {
        self.store
            .nodes
            .delete_permission_template(template_id)
            .await
    }

    pub async fn create_access_key(
        &self,
        template_id: Uuid,
        name: Option<String>,
        expires_at: Option<DateTime<Utc>>,
    ) -> MvResult<(AccessKey, String)> {
        let template = self
            .store
            .nodes
            .get_permission_template(template_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput("permission template not found".to_string()))?;

        let token = generate_access_token();
        let key_hash = hash_access_token(&token);
        let now = Utc::now();

        let access_key = AccessKey {
            id: Uuid::now_v7(),
            name,
            template_id: template.id,
            key_hash,
            created_at: now,
            last_used_at: None,
            expires_at,
            revoked_at: None,
        };

        self.store.nodes.insert_access_key(&access_key).await?;

        Ok((access_key, token))
    }

    pub async fn create_public_share(
        &self,
        node_id: Uuid,
        expires_at: Option<DateTime<Utc>>,
    ) -> MvResult<(PublicShare, String)> {
        let node = self
            .store
            .nodes
            .get(node_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput("node not found".to_string()))?;

        let _ = node;
        let token = generate_share_token();
        let token_hash = hash_share_token(&token);
        let now = Utc::now();

        let share = PublicShare {
            id: Uuid::now_v7(),
            node_id,
            token_hash,
            created_at: now,
            expires_at,
            revoked_at: None,
        };

        self.store.nodes.insert_public_share(&share).await?;

        Ok((share, token))
    }

    pub async fn list_public_shares(
        &self,
        node_id: Option<Uuid>,
        include_revoked: bool,
    ) -> MvResult<Vec<PublicShare>> {
        self.store
            .nodes
            .list_public_shares(node_id, include_revoked)
            .await
    }

    pub async fn revoke_public_share(&self, share_id: Uuid) -> MvResult<bool> {
        self.store
            .nodes
            .revoke_public_share(share_id, Utc::now())
            .await
    }

    pub async fn resolve_public_share(
        &self,
        token: &str,
    ) -> MvResult<Option<(PublicShare, KnowledgeNode)>> {
        let token_hash = hash_share_token(token);
        let share = match self
            .store
            .nodes
            .get_public_share_by_hash(&token_hash)
            .await?
        {
            Some(share) => share,
            None => return Ok(None),
        };

        if !share.is_active() {
            return Ok(None);
        }

        let node = match self.store.nodes.get(share.node_id).await? {
            Some(node) => node,
            None => return Ok(None),
        };

        Ok(Some((share, node)))
    }

    pub async fn list_access_keys(&self) -> MvResult<Vec<AccessKey>> {
        self.store.nodes.list_access_keys().await
    }

    pub async fn revoke_access_key(&self, key_id: Uuid) -> MvResult<bool> {
        self.store.nodes.revoke_access_key(key_id, Utc::now()).await
    }

    pub async fn resolve_access_key(
        &self,
        token: &str,
    ) -> MvResult<Option<(AccessKey, PermissionTemplate)>> {
        let key_hash = hash_access_token(token);
        let key = match self.store.nodes.get_access_key_by_hash(&key_hash).await? {
            Some(key) => key,
            None => return Ok(None),
        };

        if key.revoked_at.is_some() {
            return Ok(None);
        }

        if let Some(expires_at) = key.expires_at {
            if expires_at < Utc::now() {
                return Ok(None);
            }
        }

        let template = self
            .store
            .nodes
            .get_permission_template(key.template_id)
            .await?
            .ok_or_else(|| HxError::InvalidInput("permission template missing".to_string()))?;

        let _ = self
            .store
            .nodes
            .update_access_key_last_used(key.id, Utc::now())
            .await;

        Ok(Some((key, template)))
    }
}

fn generate_access_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let encoded = URL_SAFE_NO_PAD.encode(bytes);
    format!("mvk_{encoded}")
}

fn generate_share_token() -> String {
    let mut bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut bytes);
    let encoded = URL_SAFE_NO_PAD.encode(bytes);
    format!("mvs_{encoded}")
}

fn hash_access_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    URL_SAFE_NO_PAD.encode(digest)
}

fn hash_share_token(token: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(token.as_bytes());
    let digest = hasher.finalize();
    URL_SAFE_NO_PAD.encode(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_utils::*;

    async fn test_default_permission_templates_seeded() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let templates = engine.list_permission_templates(10, 0).await.unwrap();
        let names: Vec<String> = templates.into_iter().map(|t| t.name).collect();
        assert!(names.contains(&"Owner".to_string()));
        assert!(names.contains(&"Assistant".to_string()));
    }

    #[tokio::test]
    async fn test_access_key_round_trip() {
        let (engine, _tmp_dir) = create_test_engine().await;
        let template = engine
            .create_permission_template(
                "Scoped".to_string(),
                Some("Test".to_string()),
                PermissionTier::Edit,
                Some("ops".to_string()),
                vec!["shared".to_string()],
                vec![NodeKind::Fact],
                vec!["transform".to_string()],
            )
            .await
            .unwrap();

        let (_key, token) = engine
            .create_access_key(template.id, Some("Key".to_string()), None)
            .await
            .unwrap();

        let resolved = engine.resolve_access_key(&token).await.unwrap();
        assert!(resolved.is_some());
        let (_key, resolved_template) = resolved.unwrap();
        assert_eq!(resolved_template.id, template.id);
    }

    // -----------------------------------------------------------------------
    // Integration tests for Phase 1–3 features
    // -----------------------------------------------------------------------

}
