//! Full-stack REST API integration tests.
//!
//! Each test spins up a real HelixEngine backed by a tempdir, constructs
//! the axum Router, and sends actual HTTP requests via `tower::ServiceExt`.
//! This validates routing, serialisation, handler logic, and storage in one pass.
//!
//! NOTE: Tests must run sequentially (`--test-threads=1`) because the fastembed
//! ort runtime uses a global mutex that poisons if any test panics.

use std::sync::{Arc, Mutex, OnceLock};

use axum::body::Body;
use axum::http::{Method, Request, StatusCode};
use serde_json::{json, Value};
use tempfile::TempDir;
use tower::ServiceExt; // for `.oneshot()`

use hx_engine::config::EngineConfig;
use hx_engine::engine::HelixEngine;
use hx_server::rest::create_router;
use hx_server::state::AppState;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_config(data_dir: &str) -> EngineConfig {
    let mut cfg = EngineConfig {
        data_dir: data_dir.to_string(),
        ..Default::default()
    };
    cfg.embedding.provider = "noop".to_string();
    cfg
}

async fn setup() -> (axum::Router, TempDir) {
    for key in [
        "HELIX_AUTH_TOKEN",
        "HELIX_AUTH_ROLE",
        "HELIX_AUTH_NAMESPACE",
        "HELIX_JWT_SECRET",
        "HELIX_JWT_ISSUER",
        "HELIX_JWT_AUDIENCE",
    ] {
        std::env::remove_var(key);
    }
    let tmp = TempDir::new().expect("tempdir");
    let config = test_config(&tmp.path().to_string_lossy());
    setup_with_config(config, tmp).await
}

async fn setup_with_config(config: EngineConfig, tmp: TempDir) -> (axum::Router, TempDir) {
    let engine = HelixEngine::init(config).await.expect("engine init");
    let state = Arc::new(AppState::new(Arc::new(engine)));
    let router = create_router(state);
    (router, tmp)
}

fn json_request(method: Method, uri: &str, body: Option<Value>) -> Request<Body> {
    let builder = Request::builder()
        .method(method)
        .uri(uri)
        .header("content-type", "application/json");
    match body {
        Some(val) => builder.body(Body::from(val.to_string())).unwrap(),
        None => builder.body(Body::empty()).unwrap(),
    }
}

async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).to_string()))
}

fn test_env_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

struct ScopedEnvVar {
    key: &'static str,
    original: Option<String>,
    _guard: std::sync::MutexGuard<'static, ()>,
}

impl ScopedEnvVar {
    fn set(key: &'static str, value: impl Into<String>) -> Self {
        let guard = test_env_lock().lock().expect("env lock");
        let original = std::env::var(key).ok();
        std::env::set_var(key, value.into());
        Self {
            key,
            original,
            _guard: guard,
        }
    }
}

impl Drop for ScopedEnvVar {
    fn drop(&mut self) {
        if let Some(original) = self.original.as_deref() {
            std::env::set_var(self.key, original);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

#[tokio::test]
async fn health_endpoint_returns_ok() {
    let (router, _tmp) = setup().await;
    let resp = router
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn sealed_mode_blocks_routes_until_unsealed() {
    let tmp = TempDir::new().expect("tempdir");
    let mut config = test_config(&tmp.path().to_string_lossy());
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    // Non-allowlisted routes are blocked while sealed.
    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    let body: Value = body_json(resp).await;
    assert_eq!(body["error"], "Vault sealed - please unseal");

    // Keychain status remains reachable while sealed.
    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/keychain/status", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Shamir endpoints required for share-based unseal remain reachable while sealed.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::GET,
            "/api/v1/keychain/shamir/status",
            None,
        ))
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/shamir/submit",
            Some(json!({
                "share": "test-share",
                "passphrase": null
            })),
        ))
        .await
        .unwrap();
    assert_ne!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Initializing the vault unseals runtime key state.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({
                "password": "integration-test-password",
                "macos_bridge": false
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Route access is restored while unsealed.
    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Seal again and verify gating returns.
    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/keychain/seal", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Explicit unseal should restore access.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/unseal",
            Some(json!({
                "password": "integration-test-password",
                "from_macos_keychain": false,
                "from_secure_enclave": false
            })),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = router
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ---------------------------------------------------------------------------
// Node CRUD
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_and_get_node() {
    let (router, _tmp) = setup().await;

    // Create a node
    let create_body = json!({
        "kind": "fact",
        "content": "Integration test node",
        "title": "Test Fact",
        "tags": ["integration", "test"]
    });
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(create_body),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    let created: Value = body_json(resp).await;
    let id = created["id"].as_str().expect("node should have an id");
    assert_eq!(created["content"], "Integration test node");
    assert_eq!(created["title"], "Test Fact");

    // Get it back
    let get_uri = format!("/api/v1/nodes/{id}");
    let resp = router
        .oneshot(json_request(Method::GET, &get_uri, None))
        .await
        .unwrap();
    let status = resp.status();
    let fetched: Value = body_json(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "GET {get_uri} returned {status}; body: {fetched}"
    );
    assert_eq!(fetched["id"], id);
    assert_eq!(fetched["content"], "Integration test node");
}

#[tokio::test]
async fn update_node() {
    let (router, _tmp) = setup().await;

    let create_body = json!({
        "kind": "fact",
        "content": "Original",
        "tags": []
    });
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(create_body),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let mut created: Value = body_json(resp).await;
    let id = created["id"].as_str().unwrap().to_string();

    // PUT /nodes/:id expects a full KnowledgeNode, so modify the created node
    created["content"] = json!("Updated content");
    created["title"] = json!("New Title");

    let resp = router
        .oneshot(json_request(
            Method::PUT,
            &format!("/api/v1/nodes/{id}"),
            Some(created),
        ))
        .await
        .unwrap();
    let status = resp.status();
    let updated: Value = body_json(resp).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "PUT /nodes/:id returned {status}; body: {updated}"
    );
    assert_eq!(updated["content"], "Updated content");
    assert_eq!(updated["title"], "New Title");
}

#[tokio::test]
async fn delete_node() {
    let (router, _tmp) = setup().await;

    let create_body = json!({
        "kind": "fact",
        "content": "To be deleted",
        "tags": []
    });
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(create_body),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let created: Value = body_json(resp).await;
    let id = created["id"].as_str().unwrap();

    // Delete returns 200 with { "deleted": true }
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::DELETE,
            &format!("/api/v1/nodes/{id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value = body_json(resp).await;
    assert_eq!(body["deleted"], true);

    // Verify it's gone — get_node returns 404 for missing nodes
    let resp = router
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/nodes/{id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------
// Public Shares
// ---------------------------------------------------------------------------

#[tokio::test]
async fn public_share_lifecycle() {
    let (router, _tmp) = setup().await;

    let create_body = json!({
        "kind": "fact",
        "content": "Share me",
        "title": "Shareable"
    });
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(create_body),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let created: Value = body_json(resp).await;
    let node_id = created["id"].as_str().unwrap();

    let share_resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/shares",
            Some(json!({ "node_id": node_id })),
        ))
        .await
        .unwrap();
    assert_eq!(share_resp.status(), StatusCode::OK);
    let share_body: Value = body_json(share_resp).await;
    let share_id = share_body["id"].as_str().unwrap();
    let token = share_body["token"].as_str().unwrap();

    let list_resp = router
        .clone()
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/shares?node_id={node_id}&include_revoked=true"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(list_resp.status(), StatusCode::OK);
    let list_body: Value = body_json(list_resp).await;
    assert_eq!(list_body.as_array().unwrap().len(), 1);

    let public_resp = router
        .clone()
        .oneshot(json_request(
            Method::GET,
            &format!("/public/shares/{token}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(public_resp.status(), StatusCode::OK);
    let public_body: Value = body_json(public_resp).await;
    assert_eq!(public_body["node"]["id"], node_id);
    assert_eq!(public_body["node"]["content"], "Share me");

    let revoke_resp = router
        .clone()
        .oneshot(json_request(
            Method::DELETE,
            &format!("/api/v1/shares/{share_id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(revoke_resp.status(), StatusCode::OK);

    let missing_resp = router
        .oneshot(json_request(
            Method::GET,
            &format!("/public/shares/{token}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(missing_resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn list_nodes_with_kind_filter() {
    let (router, _tmp) = setup().await;

    // Create a fact and a task
    for (kind, content) in [("fact", "A fact"), ("task", "A task")] {
        let body = json!({ "kind": kind, "content": content, "tags": [] });
        let resp = router
            .clone()
            .oneshot(json_request(Method::POST, "/api/v1/nodes", Some(body)))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::CREATED,
            "creating {kind} should succeed"
        );
    }

    // List only facts
    let resp = router
        .oneshot(json_request(Method::GET, "/api/v1/nodes?kind=fact", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let list: Value = body_json(resp).await;
    let items = list.as_array().expect("should be array");
    assert!(items.iter().all(|n| n["kind"] == "fact"));
}

// ---------------------------------------------------------------------------
// Search / Recall
// ---------------------------------------------------------------------------

#[tokio::test]
async fn recall_returns_results() {
    let (router, _tmp) = setup().await;

    // Store a node first
    let body = json!({
        "kind": "fact",
        "content": "Quantum computing uses qubits for parallel computation",
        "title": "Quantum Computing",
        "tags": ["science"]
    });
    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/nodes", Some(body)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    // Recall using "text" field (not "query")
    let recall_body = json!({
        "text": "quantum",
        "limit": 10,
        "strategy": "fulltext"
    });
    let resp = router
        .oneshot(json_request(
            Method::POST,
            "/api/v1/recall",
            Some(recall_body),
        ))
        .await
        .unwrap();
    let recall_status = resp.status();
    let results: Value = body_json(resp).await;
    assert_eq!(
        recall_status,
        StatusCode::OK,
        "recall returned {recall_status}; body: {results}"
    );
    let items = results.as_array().expect("should be array");
    assert!(
        !items.is_empty(),
        "recall should find the quantum computing node"
    );
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

#[tokio::test]
async fn profile_get_and_update() {
    let (router, _tmp) = setup().await;

    // GET default profile
    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/profile", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let profile: Value = body_json(resp).await;
    assert_eq!(profile["timezone"], "UTC");

    // PUT update
    let update = json!({
        "display_name": "Test User",
        "timezone": "America/New_York",
        "bio": "Integration tester"
    });
    let resp = router
        .clone()
        .oneshot(json_request(Method::PUT, "/api/v1/profile", Some(update)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Verify the update persisted
    let resp = router
        .oneshot(json_request(Method::GET, "/api/v1/profile", None))
        .await
        .unwrap();
    let updated_profile: Value = body_json(resp).await;
    assert_eq!(updated_profile["display_name"], "Test User");
    assert_eq!(updated_profile["timezone"], "America/New_York");
    assert_eq!(updated_profile["bio"], "Integration tester");
}

// ---------------------------------------------------------------------------
// Exchange (Proposals)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn proposal_lifecycle_submit_and_list() {
    let (router, _tmp) = setup().await;

    // Submit a proposal using the correct DTO format:
    // sender must be a valid ProposalSender (agent/mcp/webhook/watcher/relay/self)
    // action is a snake_case string matching ProposalAction variants
    let proposal = json!({
        "sender": "agent",
        "action": "create_node",
        "confidence": 0.85,
        "payload": {
            "kind": "fact",
            "content": "Proposed node from agent"
        }
    });
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/exchange/proposals",
            Some(proposal),
        ))
        .await
        .unwrap();
    // Accept 200 or 201
    assert!(
        resp.status() == StatusCode::OK || resp.status() == StatusCode::CREATED,
        "submit proposal should succeed, got {}",
        resp.status()
    );

    // List proposals
    let resp = router
        .oneshot(json_request(
            Method::GET,
            "/api/v1/exchange/proposals",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let proposals: Value = body_json(resp).await;
    let items = proposals.as_array().expect("should be array");
    assert!(
        !items.is_empty(),
        "should have at least one proposal after submitting"
    );
}

// ---------------------------------------------------------------------------
// Node not found
// ---------------------------------------------------------------------------

#[tokio::test]
async fn get_nonexistent_node_returns_404() {
    let (router, _tmp) = setup().await;
    let fake_id = "00000000-0000-0000-0000-000000000000";
    let resp = router
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/nodes/{fake_id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------
// Parameterized route matching
// ---------------------------------------------------------------------------

#[tokio::test]
async fn parameterized_routes_resolve() {
    let (router, _tmp) = setup().await;

    // Create a real node, then hit several parameterized endpoints
    let body = json!({ "kind": "fact", "content": "route test", "tags": [] });
    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/nodes", Some(body)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let created: Value = body_json(resp).await;
    let id = created["id"].as_str().unwrap();

    // GET /api/v1/nodes/:id  — should resolve (not router-level 404)
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/nodes/{id}"),
            None,
        ))
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "GET /nodes/:id should resolve to handler"
    );

    // GET /api/v1/graph/neighbors/:id — should resolve even if node has no edges
    let resp = router
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/graph/neighbors/{id}"),
            None,
        ))
        .await
        .unwrap();
    // A 200 with empty array is expected; the key thing is it's NOT a router-level 404
    assert_ne!(
        resp.status(),
        StatusCode::NOT_FOUND,
        "GET /graph/neighbors/:id should resolve to handler"
    );
}

// ---------------------------------------------------------------------------
// Invalid payload
// ---------------------------------------------------------------------------

#[tokio::test]
async fn create_node_with_empty_content_returns_400() {
    let (router, _tmp) = setup().await;

    let body = json!({
        "kind": "fact",
        "content": "",
        "tags": []
    });
    let resp = router
        .oneshot(json_request(Method::POST, "/api/v1/nodes", Some(body)))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

// ---------------------------------------------------------------------------
// Sealed-mode: comprehensive route gating
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sealed_mode_blocks_data_routes_comprehensively() {
    let tmp = TempDir::new().expect("tempdir");
    let mut config = test_config(&tmp.path().to_string_lossy());
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    // All data-access routes must return 503 while sealed.
    let blocked_routes: Vec<(Method, &str, Option<Value>)> = vec![
        (Method::GET, "/api/v1/health", None),
        (
            Method::POST,
            "/api/v1/nodes",
            Some(json!({"kind":"fact","content":"x","tags":[]})),
        ),
        (Method::GET, "/api/v1/nodes", None),
        (
            Method::POST,
            "/api/v1/recall",
            Some(json!({"text":"q","limit":5,"strategy":"fulltext"})),
        ),
        (Method::GET, "/api/v1/profile", None),
        (Method::GET, "/api/v1/files", None),
        (Method::GET, "/api/v1/exchange/proposals", None),
    ];

    for (method, uri, body) in &blocked_routes {
        let resp = router
            .clone()
            .oneshot(json_request(method.clone(), uri, body.clone()))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "{method} {uri} should be blocked while sealed, got {}",
            resp.status()
        );
    }

    // Allowlisted keychain routes must NOT be blocked.
    let allowed_routes = ["/api/v1/keychain/status", "/api/v1/keychain/shamir/status"];
    for uri in &allowed_routes {
        let resp = router
            .clone()
            .oneshot(json_request(Method::GET, uri, None))
            .await
            .unwrap();
        assert_ne!(
            resp.status(),
            StatusCode::SERVICE_UNAVAILABLE,
            "GET {uri} should not be blocked while sealed"
        );
    }
}

// ---------------------------------------------------------------------------
// Sealed-mode: unseal/seal transitions produce audit chronicle entries
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sealed_unseal_transitions_produce_audit_entries() {
    let tmp = TempDir::new().expect("tempdir");
    let mut config = test_config(&tmp.path().to_string_lossy());
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    // Init vault (implicitly unseals).
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({"password":"audit-test-pw","macos_bridge":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "vault init should succeed");

    // Seal.
    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/keychain/seal", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "seal should succeed");

    // Unseal.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/unseal",
            Some(json!({"password":"audit-test-pw","from_macos_keychain":false,"from_secure_enclave":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "unseal should succeed");

    // Query chronicle for unseal_attempt entries.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::GET,
            "/api/v1/agent/chronicle?limit=50",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let entries: Value = body_json(resp).await;
    let entries_arr = entries.as_array().expect("chronicle should be array");

    let unseal_entries: Vec<&Value> = entries_arr
        .iter()
        .filter(|e| e["step_name"].as_str() == Some("unseal_attempt"))
        .collect();
    assert!(
        !unseal_entries.is_empty(),
        "chronicle should contain unseal_attempt entries; got: {entries:?}"
    );

    // At least one entry should record outcome=success.
    let has_success = unseal_entries.iter().any(|e| {
        e["logic"]
            .as_str()
            .map(|l| l.contains("outcome=success"))
            .unwrap_or(false)
    });
    assert!(
        has_success,
        "chronicle should contain a successful unseal; entries: {unseal_entries:?}"
    );
}

// ---------------------------------------------------------------------------
// Sealed-mode: full CRUD + recall lifecycle after unseal
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sealed_mode_crud_and_recall_after_unseal() {
    let tmp = TempDir::new().expect("tempdir");
    let mut config = test_config(&tmp.path().to_string_lossy());
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    // Everything blocked while sealed.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(json!({"kind":"fact","content":"blocked","tags":[]})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Init vault.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({"password":"lifecycle-pw","macos_bridge":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // CREATE node.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(json!({"kind":"fact","content":"Sealed lifecycle test node","title":"Lifecycle","tags":["sealed"]})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let created: Value = body_json(resp).await;
    let node_id = created["id"].as_str().expect("created node must have id");

    // GET node.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/nodes/{node_id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let fetched: Value = body_json(resp).await;
    assert_eq!(fetched["content"], "Sealed lifecycle test node");

    // UPDATE node.
    let mut updated_node = created.clone();
    updated_node["content"] = json!("Updated sealed content");
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::PUT,
            &format!("/api/v1/nodes/{node_id}"),
            Some(updated_node),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let updated: Value = body_json(resp).await;
    assert_eq!(updated["content"], "Updated sealed content");

    // RECALL (fulltext search).
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/recall",
            Some(json!({"text":"sealed","limit":10,"strategy":"fulltext"})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let results: Value = body_json(resp).await;
    let items = results.as_array().expect("recall results should be array");
    assert!(
        !items.is_empty(),
        "recall should find the sealed lifecycle node"
    );

    // DELETE node.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::DELETE,
            &format!("/api/v1/nodes/{node_id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Confirm deletion.
    let resp = router
        .oneshot(json_request(
            Method::GET,
            &format!("/api/v1/nodes/{node_id}"),
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

// ---------------------------------------------------------------------------
// Sealed-mode: seal → unseal → recall verifies index rebuild
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sealed_mode_recall_survives_seal_unseal_cycle() {
    let tmp = TempDir::new().expect("tempdir");
    let mut config = test_config(&tmp.path().to_string_lossy());
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    // Init + store data.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({"password":"rebuild-pw","macos_bridge":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(json!({"kind":"fact","content":"Photosynthesis converts sunlight to energy","title":"Biology","tags":["science"]})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);

    // Verify recall works before seal.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/recall",
            Some(json!({"text":"photosynthesis","limit":10,"strategy":"fulltext"})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let pre_seal: Value = body_json(resp).await;
    assert!(
        !pre_seal.as_array().unwrap().is_empty(),
        "recall before seal should find the node"
    );

    // Seal.
    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/keychain/seal", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Recall should be blocked while sealed.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/recall",
            Some(json!({"text":"photosynthesis","limit":10,"strategy":"fulltext"})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Unseal — triggers rebuild_runtime_indexes.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/unseal",
            Some(json!({"password":"rebuild-pw","from_macos_keychain":false,"from_secure_enclave":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Recall should work again with rebuilt indexes.
    let resp = router
        .oneshot(json_request(
            Method::POST,
            "/api/v1/recall",
            Some(json!({"text":"photosynthesis","limit":10,"strategy":"fulltext"})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let post_unseal: Value = body_json(resp).await;
    assert!(
        !post_unseal.as_array().unwrap().is_empty(),
        "recall after unseal should find the node (indexes rebuilt)"
    );
}

// ---------------------------------------------------------------------------
// Sealed-mode: wrong password fails unseal and is logged
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sealed_mode_wrong_password_fails_and_logged() {
    let tmp = TempDir::new().expect("tempdir");
    let mut config = test_config(&tmp.path().to_string_lossy());
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    // Init vault.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({"password":"correct-pw","macos_bridge":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Seal.
    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/keychain/seal", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Attempt unseal with wrong password — should fail.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/unseal",
            Some(json!({"password":"wrong-pw","from_macos_keychain":false,"from_secure_enclave":false})),
        ))
        .await
        .unwrap();
    assert_ne!(
        resp.status(),
        StatusCode::OK,
        "wrong password should not unseal"
    );

    // Still sealed — routes blocked.
    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    // Unseal with correct password to read chronicle.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/unseal",
            Some(json!({"password":"correct-pw","from_macos_keychain":false,"from_secure_enclave":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Chronicle should have a failed unseal entry.
    let resp = router
        .oneshot(json_request(
            Method::GET,
            "/api/v1/agent/chronicle?limit=50",
            None,
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let entries: Value = body_json(resp).await;
    let has_failure = entries.as_array().unwrap().iter().any(|e| {
        e["step_name"].as_str() == Some("unseal_attempt")
            && e["logic"]
                .as_str()
                .map(|l| l.contains("outcome=fail"))
                .unwrap_or(false)
    });
    assert!(
        has_failure,
        "chronicle should contain a failed unseal entry"
    );
}

#[tokio::test]
async fn sealed_mode_unseal_failure_injected_at_migrate_reseals_vault() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().to_string_lossy().to_string();
    let mut config = test_config(&data_dir);
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({"password":"failpoint-pw","macos_bridge":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/keychain/seal", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let _failpoint = ScopedEnvVar::set("HELIX_TEST_FAIL_POST_UNSEAL_MIGRATE", &data_dir);

    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/unseal",
            Some(json!({"password":"failpoint-pw","from_macos_keychain":false,"from_secure_enclave":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    let resp = router
        .oneshot(json_request(Method::GET, "/api/v1/keychain/status", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let status: Value = body_json(resp).await;
    assert_eq!(status["state"], "sealed");
}

#[tokio::test]
async fn sealed_mode_unseal_failure_injected_at_rebuild_reseals_vault() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().to_string_lossy().to_string();
    let mut config = test_config(&data_dir);
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({"password":"failpoint-pw","macos_bridge":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let resp = router
        .clone()
        .oneshot(json_request(Method::POST, "/api/v1/keychain/seal", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let _failpoint = ScopedEnvVar::set("HELIX_TEST_FAIL_POST_UNSEAL_REBUILD", &data_dir);

    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/unseal",
            Some(json!({"password":"failpoint-pw","from_macos_keychain":false,"from_secure_enclave":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);

    let resp = router
        .clone()
        .oneshot(json_request(Method::GET, "/api/v1/health", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

    let resp = router
        .oneshot(json_request(Method::GET, "/api/v1/keychain/status", None))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let status: Value = body_json(resp).await;
    assert_eq!(status["state"], "sealed");
}

// ---------------------------------------------------------------------------
// Sealed-mode: uploaded blob is encrypted on disk
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sealed_blob_encrypted_on_disk() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().to_string_lossy().to_string();
    let mut config = test_config(&data_dir);
    config.sealed_mode = true;
    let (router, _tmp) = setup_with_config(config, tmp).await;

    // Init vault.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/keychain/init",
            Some(json!({"password":"blob-test-pw","macos_bridge":false})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Create a node to attach to.
    let resp = router
        .clone()
        .oneshot(json_request(
            Method::POST,
            "/api/v1/nodes",
            Some(json!({"kind":"fact","content":"Blob test node","tags":[]})),
        ))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::CREATED);
    let created: Value = body_json(resp).await;
    let node_id = created["id"].as_str().unwrap();

    // Upload a file via multipart.
    let plaintext_content = b"TOP SECRET PLAINTEXT PAYLOAD FOR AT-REST AUDIT";
    let boundary = "----TestBoundary12345";
    let mut body_bytes = Vec::new();
    // node_id field
    body_bytes.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body_bytes.extend_from_slice(b"Content-Disposition: form-data; name=\"node_id\"\r\n\r\n");
    body_bytes.extend_from_slice(node_id.as_bytes());
    body_bytes.extend_from_slice(b"\r\n");
    // file field
    body_bytes.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
    body_bytes.extend_from_slice(
        b"Content-Disposition: form-data; name=\"file\"; filename=\"secret.txt\"\r\n",
    );
    body_bytes.extend_from_slice(b"Content-Type: text/plain\r\n\r\n");
    body_bytes.extend_from_slice(plaintext_content);
    body_bytes.extend_from_slice(b"\r\n");
    body_bytes.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());

    let req = Request::builder()
        .method(Method::POST)
        .uri("/api/v1/files/upload")
        .header(
            "content-type",
            format!("multipart/form-data; boundary={boundary}"),
        )
        .body(Body::from(body_bytes))
        .unwrap();

    let resp = router.clone().oneshot(req).await.unwrap();
    let upload_status = resp.status();
    let upload_body: Value = body_json(resp).await;
    assert!(
        upload_status == StatusCode::OK || upload_status == StatusCode::CREATED,
        "upload should succeed, got {upload_status}; body: {upload_body}"
    );

    // Scan blobs/ directory on disk — every file must start with HXB1 magic.
    let blobs_dir = std::path::PathBuf::from(&data_dir).join("blobs");
    assert!(
        blobs_dir.exists(),
        "blobs directory should exist after upload"
    );

    let mut found_blob = false;
    let mut stack = vec![blobs_dir];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read blobs dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !path.is_file() {
                continue;
            }
            let bytes = std::fs::read(&path).expect("read blob file");
            assert!(
                bytes.starts_with(b"HXB1"),
                "blob at {} must start with HXB1 magic prefix; first 4 bytes: {:?}",
                path.display(),
                &bytes[..bytes.len().min(4)]
            );
            // Ensure the plaintext is NOT present in the raw encrypted bytes.
            let blob_str = String::from_utf8_lossy(&bytes);
            assert!(
                !blob_str.contains("TOP SECRET PLAINTEXT"),
                "blob at {} must not contain plaintext content",
                path.display()
            );
            found_blob = true;
        }
    }
    assert!(found_blob, "at least one blob file should exist on disk");
}

// ---------------------------------------------------------------------------
// Sealed-mode: lifecycle survives engine restart (simulated)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sealed_mode_lifecycle_survives_restart() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().to_string_lossy().to_string();

    // --- Session 1: init vault, store data, seal ---
    {
        let mut config = test_config(&data_dir);
        config.sealed_mode = true;
        let engine = HelixEngine::init(config).await.expect("engine init");
        let state = Arc::new(AppState::new(Arc::new(engine)));
        let router = create_router(state.clone());

        // Init vault.
        let resp = router
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/api/v1/keychain/init",
                Some(json!({"password":"restart-pw","macos_bridge":false})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Store a node.
        let resp = router
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/api/v1/nodes",
                Some(json!({"kind":"fact","content":"Mitochondria is the powerhouse of the cell","title":"Biology 101","tags":["bio"]})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // Verify recall works.
        let resp = router
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/api/v1/recall",
                Some(json!({"text":"mitochondria","limit":10,"strategy":"fulltext"})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let results: Value = body_json(resp).await;
        assert!(!results.as_array().unwrap().is_empty());

        // Seal.
        let resp = router
            .oneshot(json_request(Method::POST, "/api/v1/keychain/seal", None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Engine dropped here — simulates process shutdown.
    }

    // --- Session 2: new engine on same data dir, unseal, verify data ---
    {
        let mut config = test_config(&data_dir);
        config.sealed_mode = true;
        let engine = HelixEngine::init(config).await.expect("engine init");
        let state = Arc::new(AppState::new(Arc::new(engine)));
        let router = create_router(state.clone());

        // Routes blocked while sealed.
        let resp = router
            .clone()
            .oneshot(json_request(Method::GET, "/api/v1/health", None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);

        // Unseal with same password.
        let resp = router
            .clone()
            .oneshot(json_request(
                Method::POST,
                "/api/v1/keychain/unseal",
                Some(json!({"password":"restart-pw","from_macos_keychain":false,"from_secure_enclave":false})),
            ))
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::OK,
            "unseal after restart should succeed"
        );

        // Health restored.
        let resp = router
            .clone()
            .oneshot(json_request(Method::GET, "/api/v1/health", None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Node data survives restart.
        let resp = router
            .clone()
            .oneshot(json_request(Method::GET, "/api/v1/nodes?kind=fact", None))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let nodes: Value = body_json(resp).await;
        let items = nodes.as_array().expect("should be array");
        assert!(
            items.iter().any(|n| n["content"]
                .as_str()
                .map(|c| c.contains("Mitochondria"))
                .unwrap_or(false)),
            "node content should survive restart"
        );

        // FTS recall works after post-unseal rebuild.
        let resp = router
            .oneshot(json_request(
                Method::POST,
                "/api/v1/recall",
                Some(json!({"text":"mitochondria","limit":10,"strategy":"fulltext"})),
            ))
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let results: Value = body_json(resp).await;
        assert!(
            !results.as_array().unwrap().is_empty(),
            "recall should work after restart + unseal (indexes rebuilt)"
        );
    }
}
