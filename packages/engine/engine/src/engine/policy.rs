use super::HelixEngine;
use hx_core::{
    AccessPolicy, ApprovalRequest, ApprovalStore, GraphStore, MvResult, PolicyDecision,
    PolicyStore, ProxyAuditEntry, ProxyAuditStore,
};
use uuid::Uuid;

impl HelixEngine {
    // -------------------------------------------------------------------------
    // Policy / Approval / Proxy Audit / Graph delegation methods
    // -------------------------------------------------------------------------

    /// Check the access policy for a given secret key and consumer.
    /// Returns `PolicyDecision::Deny` with reason if no policy is found (default-deny).
    pub async fn check_policy(
        &self,
        secret_key: &str,
        consumer: &str,
    ) -> MvResult<PolicyDecision> {
        let policy = self
            .store
            .nodes
            .get_policy_for(secret_key, consumer)
            .await?;

        match policy {
            Some(p) if p.is_expired() => Ok(PolicyDecision::Deny {
                reason: format!("policy for '{secret_key}' has expired"),
            }),
            Some(p) if p.require_approval => Ok(PolicyDecision::RequiresApproval {
                ttl_seconds: p.max_ttl_seconds.unwrap_or(3600),
                scopes: p.scopes.clone(),
            }),
            Some(p) if p.allowed => Ok(PolicyDecision::Allow {
                ttl_seconds: p.max_ttl_seconds,
                scopes: p.scopes.clone(),
            }),
            Some(_) => Ok(PolicyDecision::Deny {
                reason: format!("policy denies consumer '{consumer}' access to '{secret_key}'"),
            }),
            None => Ok(PolicyDecision::Deny {
                reason: format!(
                    "no policy found for consumer '{consumer}' on secret '{secret_key}'"
                ),
            }),
        }
    }

    /// Find an active (approved, non-expired) approval for a consumer+secret pair.
    pub async fn find_active_approval(
        &self,
        consumer: &str,
        secret_key: &str,
    ) -> MvResult<Option<ApprovalRequest>> {
        self.store
            .nodes
            .find_active_approval(consumer, secret_key)
            .await
    }

    /// Create a new approval request.
    pub async fn create_approval(&self, request: &ApprovalRequest) -> MvResult<()> {
        self.store.nodes.create_approval(request).await
    }

    /// Log a proxy audit entry.
    pub async fn log_proxy_audit(&self, entry: &ProxyAuditEntry) -> MvResult<()> {
        self.store.nodes.log_proxy_audit(entry).await
    }

    /// Update a proxy audit entry with outcome details.
    pub async fn update_proxy_audit(
        &self,
        id: Uuid,
        success: bool,
        sanitized: bool,
        error: Option<&str>,
        response_status: Option<i32>,
    ) -> MvResult<()> {
        self.store
            .nodes
            .update_proxy_audit(id, success, sanitized, error, response_status)
            .await
    }

    /// Get neighbor node IDs up to a given depth.
    pub async fn get_neighbors(&self, node_id: Uuid, depth: usize) -> MvResult<Vec<Uuid>> {
        self.graph.get_neighbors(node_id, depth).await
    }

    /// Add a relationship to the graph.
    pub async fn add_relationship(&self, rel: &hx_core::Relationship) -> MvResult<()> {
        self.graph.add_relationship(rel).await
    }

    /// Set (insert or update) an access policy.
    pub async fn set_policy(&self, policy: &AccessPolicy) -> MvResult<()> {
        self.store.nodes.set_policy(policy).await
    }

    /// List access policies with optional filters.
    pub async fn list_policies(
        &self,
        secret_key: Option<&str>,
        consumer: Option<&str>,
    ) -> MvResult<Vec<AccessPolicy>> {
        self.store.nodes.list_policies(secret_key, consumer).await
    }

    /// Delete an access policy by ID.
    pub async fn delete_policy(&self, id: Uuid) -> MvResult<bool> {
        self.store.nodes.delete_policy(id).await
    }

    /// Get a single approval request by ID.
    pub async fn get_approval(&self, id: Uuid) -> MvResult<Option<ApprovalRequest>> {
        self.store.nodes.get_approval(id).await
    }

    /// List pending approval requests, optionally filtered by consumer.
    pub async fn list_pending_approvals(
        &self,
        consumer: Option<&str>,
    ) -> MvResult<Vec<ApprovalRequest>> {
        self.store.nodes.list_pending_approvals(consumer).await
    }

    /// Decide on an approval request (approve or deny).
    pub async fn decide_approval(
        &self,
        id: Uuid,
        approved: bool,
        decided_by: Option<&str>,
        deny_reason: Option<&str>,
    ) -> MvResult<bool> {
        self.store
            .nodes
            .decide_approval(id, approved, decided_by, deny_reason)
            .await
    }

    /// Expire stale approval requests.
    pub async fn expire_approvals(&self) -> MvResult<usize> {
        self.store.nodes.expire_approvals().await
    }

    /// List proxy audit log entries.
    pub async fn list_proxy_audit(
        &self,
        consumer: Option<&str>,
        limit: usize,
        offset: usize,
    ) -> MvResult<Vec<ProxyAuditEntry>> {
        self.store
            .nodes
            .list_proxy_audit(consumer, limit, offset)
            .await
    }
}
