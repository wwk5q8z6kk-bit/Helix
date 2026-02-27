//! Federated query protocol.
//! Enables read-only queries across trusted peer vaults via REST.
//!
//! Each peer is another Helix instance exposing `/api/v1/recall`.
//! Queries run in parallel with per-peer timeouts and error isolation.
//!
//! ## Request authentication
//! Peers may share an HMAC-SHA256 secret established during handshake.
//! When a shared secret exists, outgoing queries carry `x-helix-signature`
//! and `x-helix-timestamp` headers for replay-protected authentication.

use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use hx_core::*;
use hx_storage::unified::UnifiedStore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use uuid::Uuid;

type HmacSha256 = Hmac<Sha256>;

// ---------------------------------------------------------------------------
// Request signing constants
// ---------------------------------------------------------------------------

/// HTTP header carrying the HMAC-SHA256 signature (base64).
pub const SIGNATURE_HEADER: &str = "x-helix-signature";
/// HTTP header carrying the Unix timestamp used in the signature.
pub const TIMESTAMP_HEADER: &str = "x-helix-timestamp";
/// HTTP header carrying the sender's vault ID (for peer lookup).
pub const VAULT_ID_HEADER: &str = "x-helix-vault-id";
/// Maximum acceptable clock skew between peers (5 minutes).
pub const MAX_CLOCK_SKEW_SECS: i64 = 300;

/// A trusted federation peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationPeer {
    pub id: Uuid,
    pub vault_id: String,
    pub display_name: String,
    pub endpoint: String,
    pub public_key: Option<String>,
    pub allowed_namespaces: Vec<String>,
    pub max_results: usize,
    pub enabled: bool,
    pub last_seen: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    /// HMAC-SHA256 shared secret for request signing (never serialized in API responses).
    #[serde(skip)]
    pub shared_secret: Option<String>,
}

impl FederationPeer {
    pub fn new(
        vault_id: impl Into<String>,
        display_name: impl Into<String>,
        endpoint: impl Into<String>,
    ) -> Self {
        Self {
            id: Uuid::now_v7(),
            vault_id: vault_id.into(),
            display_name: display_name.into(),
            endpoint: endpoint.into(),
            public_key: None,
            allowed_namespaces: Vec::new(),
            max_results: 50,
            enabled: true,
            last_seen: None,
            created_at: Utc::now(),
            shared_secret: None,
        }
    }

    /// Base URL for this peer's REST API (strips trailing slash).
    fn api_base(&self) -> &str {
        self.endpoint.trim_end_matches('/')
    }
}

/// Result from a federated query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    pub source_vault: String,
    pub source_peer_name: String,
    pub node: KnowledgeNode,
    pub relevance_score: f64,
}

/// REST recall response shape (mirrors hx-server's JSON format).
#[derive(Deserialize)]
struct PeerRecallResponse {
    results: Vec<PeerSearchResult>,
}

#[derive(Deserialize)]
struct PeerSearchResult {
    node: KnowledgeNode,
    score: f64,
}

/// REST health response shape.
#[derive(Deserialize)]
struct PeerHealthResponse {
    status: String,
}

/// Response shape from a peer's `/api/v1/federation/identity` endpoint.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PeerIdentityResponse {
    pub vault_id: String,
    pub display_name: String,
    pub public_key: Option<String>,
    pub vault_address: Option<String>,
    pub updated_at: String,
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 Request Signing
// ---------------------------------------------------------------------------

/// Errors from federation request authentication.
#[derive(Debug)]
pub enum FederationAuthError {
    /// Timestamp outside the ±5-minute window.
    TimestampExpired,
    /// Signature is malformed or does not match.
    InvalidSignature,
    /// No peer found for the given vault ID.
    UnknownPeer,
    /// Peer has no shared secret configured.
    NoSecret,
}

impl std::fmt::Display for FederationAuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimestampExpired => write!(f, "request timestamp outside acceptable window"),
            Self::InvalidSignature => write!(f, "HMAC signature verification failed"),
            Self::UnknownPeer => write!(f, "unknown vault ID — peer not registered"),
            Self::NoSecret => write!(f, "peer has no shared secret configured"),
        }
    }
}

impl std::error::Error for FederationAuthError {}

/// Sign a federation request body with HMAC-SHA256.
///
/// The signed message is: `"{timestamp}\n{body}"`. Returns the signature
/// as a base64-encoded string.
pub fn sign_federation_request(secret: &[u8], timestamp: i64, body: &[u8]) -> String {
    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(timestamp.to_string().as_bytes());
    mac.update(b"\n");
    mac.update(body);
    base64::engine::general_purpose::STANDARD.encode(mac.finalize().into_bytes())
}

/// Verify a federation request signature (constant-time comparison).
///
/// Returns `Ok(())` if the signature is valid and the timestamp is within
/// the ±5-minute window. Returns a specific error otherwise.
pub fn verify_federation_request(
    secret: &[u8],
    timestamp: i64,
    body: &[u8],
    signature: &str,
) -> Result<(), FederationAuthError> {
    let now = Utc::now().timestamp();
    if (now - timestamp).abs() > MAX_CLOCK_SKEW_SECS {
        return Err(FederationAuthError::TimestampExpired);
    }

    let sig_bytes = base64::engine::general_purpose::STANDARD
        .decode(signature)
        .map_err(|_| FederationAuthError::InvalidSignature)?;

    let mut mac = HmacSha256::new_from_slice(secret).expect("HMAC accepts any key length");
    mac.update(timestamp.to_string().as_bytes());
    mac.update(b"\n");
    mac.update(body);
    mac.verify_slice(&sig_bytes)
        .map_err(|_| FederationAuthError::InvalidSignature)
}

/// Federation engine manages peers and dispatches queries.
pub struct FederationEngine {
    #[allow(dead_code)]
    store: Arc<UnifiedStore>,
    peers: tokio::sync::RwLock<Vec<FederationPeer>>,
    client: reqwest::Client,
}

impl FederationEngine {
    pub fn new(store: Arc<UnifiedStore>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .connect_timeout(Duration::from_secs(5))
            .user_agent("Helix-Federation/0.1")
            .build()
            .unwrap_or_default();

        Self {
            store,
            peers: tokio::sync::RwLock::new(Vec::new()),
            client,
        }
    }

    /// Add a trusted peer.
    pub async fn add_peer(&self, peer: FederationPeer) {
        self.peers.write().await.push(peer);
    }

    /// Remove a peer by ID.
    pub async fn remove_peer(&self, id: Uuid) -> bool {
        let mut peers = self.peers.write().await;
        let len_before = peers.len();
        peers.retain(|p| p.id != id);
        peers.len() < len_before
    }

    /// List all peers.
    pub async fn list_peers(&self) -> Vec<FederationPeer> {
        self.peers.read().await.clone()
    }

    /// Get a peer by ID.
    pub async fn get_peer(&self, id: Uuid) -> Option<FederationPeer> {
        self.peers.read().await.iter().find(|p| p.id == id).cloned()
    }

    /// Find a peer by vault ID (used for incoming request authentication).
    pub async fn find_peer_by_vault_id(&self, vault_id: &str) -> Option<FederationPeer> {
        self.peers
            .read()
            .await
            .iter()
            .find(|p| p.vault_id == vault_id)
            .cloned()
    }

    /// Perform a handshake with a remote peer by calling their identity endpoint.
    ///
    /// This auto-discovers the peer's vault ID, display name, and public key,
    /// then registers them as a trusted peer. If `shared_secret` is provided,
    /// it will be used for HMAC-signed requests to this peer.
    pub async fn handshake(
        &self,
        endpoint: &str,
        shared_secret: Option<String>,
    ) -> MvResult<FederationPeer> {
        let base = endpoint.trim_end_matches('/');
        let url = format!("{base}/api/v1/federation/identity");

        tracing::info!(%url, "Initiating federation handshake");

        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| HxError::Federation(format!("failed to reach peer: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            return Err(HxError::Federation(format!(
                "peer identity endpoint returned {status}"
            )));
        }

        let identity: PeerIdentityResponse = resp
            .json()
            .await
            .map_err(|e| HxError::Federation(format!("invalid identity response: {e}")))?;

        // Reject if the peer's vault_id collides with an existing peer
        if self.find_peer_by_vault_id(&identity.vault_id).await.is_some() {
            return Err(HxError::Federation(format!(
                "peer with vault_id '{}' is already registered",
                identity.vault_id
            )));
        }

        let mut peer = FederationPeer::new(
            identity.vault_id,
            identity.display_name,
            endpoint.to_string(),
        );
        peer.public_key = identity.public_key;
        peer.shared_secret = shared_secret;

        let result = peer.clone();
        self.add_peer(peer).await;

        tracing::info!(
            vault_id = %result.vault_id,
            name = %result.display_name,
            "Federation handshake complete"
        );

        Ok(result)
    }

    /// Health check a peer by calling its `/api/v1/health` endpoint.
    pub async fn health_check(&self, peer_id: Uuid) -> MvResult<bool> {
        let peers = self.peers.read().await;
        let Some(peer) = peers.iter().find(|p| p.id == peer_id) else {
            return Ok(false);
        };

        if !peer.enabled {
            return Ok(false);
        }

        let url = format!("{}/api/v1/health", peer.api_base());
        tracing::debug!(peer = %peer.display_name, %url, "Federation health check");

        match self.client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                if let Ok(body) = resp.json::<PeerHealthResponse>().await {
                    let healthy = body.status == "ok";
                    // Update last_seen on success
                    drop(peers);
                    self.update_last_seen(peer_id).await;
                    Ok(healthy)
                } else {
                    Ok(false)
                }
            }
            Ok(resp) => {
                tracing::warn!(
                    peer = %peer.display_name,
                    status = %resp.status(),
                    "Peer health check returned non-OK status"
                );
                Ok(false)
            }
            Err(e) => {
                tracing::warn!(
                    peer = %peer.display_name,
                    error = %e,
                    "Peer health check failed"
                );
                Ok(false)
            }
        }
    }

    /// Query across all enabled peers via their REST recall endpoints.
    /// Runs queries in parallel; errors from individual peers are logged but don't fail the batch.
    pub async fn federated_query(
        &self,
        query: &str,
        limit: usize,
    ) -> MvResult<Vec<FederatedResult>> {
        let peers = self.peers.read().await;
        let enabled: Vec<_> = peers.iter().filter(|p| p.enabled).cloned().collect();
        drop(peers);

        if enabled.is_empty() {
            return Ok(Vec::new());
        }

        tracing::info!(
            query = query,
            peer_count = enabled.len(),
            "Dispatching federated query to peers"
        );

        // Fire parallel requests to all enabled peers
        let mut handles = Vec::with_capacity(enabled.len());
        for peer in &enabled {
            let client = self.client.clone();
            let peer = peer.clone();
            let query = query.to_string();
            let per_peer_limit = limit.min(peer.max_results);

            handles.push(tokio::spawn(async move {
                query_peer(&client, &peer, &query, per_peer_limit).await
            }));
        }

        // Collect results, logging errors per-peer
        let mut all_results = Vec::new();
        for (i, handle) in handles.into_iter().enumerate() {
            match handle.await {
                Ok(Ok(results)) => {
                    if !results.is_empty() {
                        tracing::debug!(
                            peer = %enabled[i].display_name,
                            count = results.len(),
                            "Peer returned results"
                        );
                    }
                    all_results.extend(results);
                }
                Ok(Err(e)) => {
                    tracing::warn!(
                        peer = %enabled[i].display_name,
                        error = %e,
                        "Peer query failed"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        peer = %enabled[i].display_name,
                        error = %e,
                        "Peer query task panicked"
                    );
                }
            }
        }

        // Update last_seen for peers that responded (best-effort)
        for peer in &enabled {
            self.update_last_seen(peer.id).await;
        }

        // Sort by relevance descending, then truncate to requested limit
        all_results.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results.truncate(limit);

        Ok(all_results)
    }

    /// Update the last_seen timestamp for a peer.
    async fn update_last_seen(&self, peer_id: Uuid) {
        let mut peers = self.peers.write().await;
        if let Some(peer) = peers.iter_mut().find(|p| p.id == peer_id) {
            peer.last_seen = Some(Utc::now());
        }
    }
}

/// Query a single peer's recall endpoint.
///
/// When the peer has a `shared_secret`, the request is signed with
/// HMAC-SHA256 headers for authentication and replay protection.
async fn query_peer(
    client: &reqwest::Client,
    peer: &FederationPeer,
    query: &str,
    limit: usize,
) -> Result<Vec<FederatedResult>, String> {
    let url = format!("{}/api/v1/recall", peer.api_base());

    let mut body = serde_json::json!({
        "text": query,
        "limit": limit,
        "strategy": "hybrid",
    });

    // Scope to allowed namespaces if configured
    if peer.allowed_namespaces.len() == 1 {
        body["namespace"] = serde_json::json!(peer.allowed_namespaces[0]);
    }

    let body_bytes = serde_json::to_vec(&body).map_err(|e| format!("serialize body: {e}"))?;

    let mut request = client
        .post(&url)
        .header("content-type", "application/json")
        .body(body_bytes.clone());

    // Sign the request when a shared secret is available
    if let Some(ref secret) = peer.shared_secret {
        let timestamp = Utc::now().timestamp();
        let signature = sign_federation_request(secret.as_bytes(), timestamp, &body_bytes);
        request = request
            .header(TIMESTAMP_HEADER, timestamp.to_string())
            .header(SIGNATURE_HEADER, signature)
            .header(VAULT_ID_HEADER, &peer.vault_id);
    }

    let response = request
        .send()
        .await
        .map_err(|e| format!("request failed: {e}"))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        return Err(format!("peer returned {status}: {text}"));
    }

    let recall: PeerRecallResponse = response
        .json()
        .await
        .map_err(|e| format!("failed to parse peer response: {e}"))?;

    Ok(recall
        .results
        .into_iter()
        .map(|r| FederatedResult {
            source_vault: peer.vault_id.clone(),
            source_peer_name: peer.display_name.clone(),
            relevance_score: r.score,
            node: r.node,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_SECRET: &[u8] = b"super-secret-shared-key-for-tests";
    const TEST_BODY: &[u8] = b"{\"text\":\"hello\",\"limit\":10}";

    // -- Signing / verification -----------------------------------------------

    #[test]
    fn sign_produces_base64_output() {
        let sig = sign_federation_request(TEST_SECRET, 1700000000, TEST_BODY);
        // Must be valid base64
        let decoded = base64::engine::general_purpose::STANDARD.decode(&sig);
        assert!(decoded.is_ok(), "signature must be valid base64");
        // HMAC-SHA256 produces 32 bytes
        assert_eq!(decoded.unwrap().len(), 32);
    }

    #[test]
    fn sign_is_deterministic() {
        let a = sign_federation_request(TEST_SECRET, 1700000000, TEST_BODY);
        let b = sign_federation_request(TEST_SECRET, 1700000000, TEST_BODY);
        assert_eq!(a, b);
    }

    #[test]
    fn sign_differs_for_different_timestamps() {
        let a = sign_federation_request(TEST_SECRET, 1700000000, TEST_BODY);
        let b = sign_federation_request(TEST_SECRET, 1700000001, TEST_BODY);
        assert_ne!(a, b);
    }

    #[test]
    fn sign_differs_for_different_bodies() {
        let a = sign_federation_request(TEST_SECRET, 1700000000, b"body-a");
        let b = sign_federation_request(TEST_SECRET, 1700000000, b"body-b");
        assert_ne!(a, b);
    }

    #[test]
    fn sign_differs_for_different_secrets() {
        let a = sign_federation_request(b"secret-1", 1700000000, TEST_BODY);
        let b = sign_federation_request(b"secret-2", 1700000000, TEST_BODY);
        assert_ne!(a, b);
    }

    #[test]
    fn verify_accepts_valid_signature() {
        let now = Utc::now().timestamp();
        let sig = sign_federation_request(TEST_SECRET, now, TEST_BODY);
        let result = verify_federation_request(TEST_SECRET, now, TEST_BODY, &sig);
        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_wrong_secret() {
        let now = Utc::now().timestamp();
        let sig = sign_federation_request(TEST_SECRET, now, TEST_BODY);
        let result = verify_federation_request(b"wrong-secret", now, TEST_BODY, &sig);
        assert!(matches!(result, Err(FederationAuthError::InvalidSignature)));
    }

    #[test]
    fn verify_rejects_tampered_body() {
        let now = Utc::now().timestamp();
        let sig = sign_federation_request(TEST_SECRET, now, TEST_BODY);
        let result = verify_federation_request(TEST_SECRET, now, b"tampered", &sig);
        assert!(matches!(result, Err(FederationAuthError::InvalidSignature)));
    }

    #[test]
    fn verify_rejects_expired_timestamp() {
        let old = Utc::now().timestamp() - MAX_CLOCK_SKEW_SECS - 1;
        let sig = sign_federation_request(TEST_SECRET, old, TEST_BODY);
        let result = verify_federation_request(TEST_SECRET, old, TEST_BODY, &sig);
        assert!(matches!(result, Err(FederationAuthError::TimestampExpired)));
    }

    #[test]
    fn verify_rejects_future_timestamp() {
        let future = Utc::now().timestamp() + MAX_CLOCK_SKEW_SECS + 1;
        let sig = sign_federation_request(TEST_SECRET, future, TEST_BODY);
        let result = verify_federation_request(TEST_SECRET, future, TEST_BODY, &sig);
        assert!(matches!(result, Err(FederationAuthError::TimestampExpired)));
    }

    #[test]
    fn verify_accepts_within_skew_window() {
        // 4 minutes ago — within the 5-minute window
        let recent = Utc::now().timestamp() - 240;
        let sig = sign_federation_request(TEST_SECRET, recent, TEST_BODY);
        let result = verify_federation_request(TEST_SECRET, recent, TEST_BODY, &sig);
        assert!(result.is_ok());
    }

    #[test]
    fn verify_rejects_malformed_base64() {
        let now = Utc::now().timestamp();
        let result = verify_federation_request(TEST_SECRET, now, TEST_BODY, "not!valid!base64!!!");
        assert!(matches!(result, Err(FederationAuthError::InvalidSignature)));
    }

    // -- Peer management ------------------------------------------------------

    #[test]
    fn peer_new_sets_defaults() {
        let peer = FederationPeer::new("vault-1", "Alice", "https://alice.local:9470");
        assert_eq!(peer.vault_id, "vault-1");
        assert_eq!(peer.display_name, "Alice");
        assert!(peer.enabled);
        assert_eq!(peer.max_results, 50);
        assert!(peer.shared_secret.is_none());
    }

    #[test]
    fn shared_secret_is_skipped_in_serialization() {
        let mut peer = FederationPeer::new("v", "n", "http://localhost");
        peer.shared_secret = Some("top-secret".into());
        let json = serde_json::to_string(&peer).unwrap();
        assert!(!json.contains("shared_secret"), "secret must not appear in JSON");
        assert!(!json.contains("top-secret"), "secret value must not leak");
    }

    #[test]
    fn api_base_strips_trailing_slash() {
        let peer = FederationPeer::new("v", "n", "https://peer.local:9470/");
        assert_eq!(peer.api_base(), "https://peer.local:9470");
    }

    // -- Handshake (mockito) --------------------------------------------------

    #[tokio::test]
    async fn handshake_auto_discovers_peer() {
        let mut server = mockito::Server::new_async().await;
        let identity = PeerIdentityResponse {
            vault_id: "remote-vault-123".into(),
            display_name: "Remote Bob".into(),
            public_key: Some("pk-bob".into()),
            vault_address: Some("mailto:bob@example.com".into()),
            updated_at: Utc::now().to_rfc3339(),
        };

        let _mock = server
            .mock("GET", "/api/v1/federation/identity")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::to_string(&identity).unwrap())
            .create_async()
            .await;

        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(
            UnifiedStore::open(tmp.path(), 384)
                .await
                .unwrap(),
        );
        let engine = FederationEngine::new(store);

        let peer = engine
            .handshake(&server.url(), Some("shared-key".into()))
            .await
            .unwrap();

        assert_eq!(peer.vault_id, "remote-vault-123");
        assert_eq!(peer.display_name, "Remote Bob");
        assert_eq!(peer.public_key.as_deref(), Some("pk-bob"));
        assert_eq!(peer.shared_secret.as_deref(), Some("shared-key"));

        // Peer should be registered
        let peers = engine.list_peers().await;
        assert_eq!(peers.len(), 1);
    }

    #[tokio::test]
    async fn handshake_rejects_duplicate_vault_id() {
        let mut server = mockito::Server::new_async().await;
        let identity = PeerIdentityResponse {
            vault_id: "same-vault".into(),
            display_name: "Peer".into(),
            public_key: None,
            vault_address: None,
            updated_at: Utc::now().to_rfc3339(),
        };

        let _mock = server
            .mock("GET", "/api/v1/federation/identity")
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(serde_json::to_string(&identity).unwrap())
            .expect(2)
            .create_async()
            .await;

        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(
            UnifiedStore::open(tmp.path(), 384)
                .await
                .unwrap(),
        );
        let engine = FederationEngine::new(store);

        // First handshake succeeds
        engine.handshake(&server.url(), None).await.unwrap();

        // Second handshake with same vault_id fails
        let err = engine.handshake(&server.url(), None).await.unwrap_err();
        assert!(err.to_string().contains("already registered"));
    }

    #[tokio::test]
    async fn handshake_fails_on_unreachable_peer() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(
            UnifiedStore::open(tmp.path(), 384)
                .await
                .unwrap(),
        );
        let engine = FederationEngine::new(store);

        let err = engine
            .handshake("http://127.0.0.1:1", None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("failed to reach peer"));
    }

    #[tokio::test]
    async fn handshake_fails_on_error_status() {
        let mut server = mockito::Server::new_async().await;
        let _mock = server
            .mock("GET", "/api/v1/federation/identity")
            .with_status(500)
            .create_async()
            .await;

        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(
            UnifiedStore::open(tmp.path(), 384)
                .await
                .unwrap(),
        );
        let engine = FederationEngine::new(store);

        let err = engine.handshake(&server.url(), None).await.unwrap_err();
        assert!(err.to_string().contains("returned 500"));
    }

    // -- find_peer_by_vault_id ------------------------------------------------

    #[tokio::test]
    async fn find_peer_by_vault_id_returns_match() {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(
            UnifiedStore::open(tmp.path(), 384)
                .await
                .unwrap(),
        );
        let engine = FederationEngine::new(store);

        let peer = FederationPeer::new("vault-abc", "Alice", "http://alice");
        engine.add_peer(peer).await;

        assert!(engine.find_peer_by_vault_id("vault-abc").await.is_some());
        assert!(engine.find_peer_by_vault_id("vault-xyz").await.is_none());
    }
}
