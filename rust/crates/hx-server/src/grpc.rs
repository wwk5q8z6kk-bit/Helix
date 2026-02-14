use std::pin::Pin;
use std::sync::Arc;

use axum::http::{HeaderMap, HeaderValue};
use futures::Stream;
use serde_json::Value;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use hx_core::*;

use crate::auth::AuthContext;
use crate::limits::{
    enforce_keychain_unseal_failure_backoff, enforce_namespace_quota, enforce_rate_limit,
    record_keychain_unseal_failure, NamespaceQuotaError, RateLimitExceeded,
};
use crate::metrics::get_metrics;
use crate::state::AppState;
use crate::validation::{
    validate_depth, validate_list_limit, validate_node_payload, validate_query_text,
    validate_recall_limit,
};

pub mod proto {
    tonic::include_proto!("helix.v1");
}

use proto::helix_service_server::HelixService;
use proto::*;

pub struct HelixGrpc {
    state: Arc<AppState>,
}

impl HelixGrpc {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

fn node_to_proto(node: &KnowledgeNode) -> KnowledgeNodeProto {
    KnowledgeNodeProto {
        id: node.id.to_string(),
        kind: node.kind.as_str().to_string(),
        title: node.title.clone(),
        content: node.content.clone(),
        source: node.source.clone(),
        namespace: node.namespace.clone(),
        tags: node.tags.clone(),
        importance: node.importance,
        created_at: node.temporal.created_at.to_rfc3339(),
        updated_at: node.temporal.updated_at.to_rfc3339(),
        last_accessed_at: node.temporal.last_accessed_at.to_rfc3339(),
        access_count: node.temporal.access_count,
        version: node.temporal.version,
        expires_at: node.temporal.expires_at.map(|dt| dt.to_rfc3339()),
        metadata_json: serde_json::to_string(&node.metadata).unwrap_or_default(),
    }
}

#[allow(clippy::result_large_err)]
fn parse_metadata_json(
    raw: Option<&str>,
) -> Result<std::collections::HashMap<String, Value>, Status> {
    let Some(raw) = raw else {
        return Ok(Default::default());
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Ok(Default::default());
    }

    serde_json::from_str(trimmed).map_err(|err| {
        Status::invalid_argument(format!("metadata_json must be valid JSON object: {err}"))
    })
}

#[allow(clippy::result_large_err)]
fn parse_optional_datetime(
    raw: Option<&str>,
    field_name: &str,
) -> Result<Option<chrono::DateTime<chrono::Utc>>, Status> {
    match raw.map(str::trim).filter(|value| !value.is_empty()) {
        Some(value) => chrono::DateTime::parse_from_rfc3339(value)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .map(Some)
            .map_err(|err| {
                Status::invalid_argument(format!(
                    "{field_name} must be RFC3339 datetime when provided: {err}"
                ))
            }),
        None => Ok(None),
    }
}

fn match_source_to_str(source: MatchSource) -> &'static str {
    match source {
        MatchSource::Vector => "vector",
        MatchSource::FullText => "full_text",
        MatchSource::Hybrid => "hybrid",
        MatchSource::Graph => "graph",
    }
}

fn parse_kind_list(kinds: &[String]) -> Result<Option<Vec<NodeKind>>, String> {
    if kinds.is_empty() {
        return Ok(None);
    }

    let mut parsed = Vec::with_capacity(kinds.len());
    for kind in kinds {
        parsed.push(kind.parse().map_err(|e: String| e)?);
    }
    Ok(Some(parsed))
}

fn map_namespace_quota_status(err: NamespaceQuotaError) -> Status {
    match err {
        NamespaceQuotaError::Exceeded {
            namespace,
            quota,
            count,
        } => Status::resource_exhausted(format!(
            "namespace '{namespace}' quota exceeded ({count}/{quota} nodes)"
        )),
        NamespaceQuotaError::Backend(message) => Status::internal(message),
    }
}

#[allow(clippy::result_large_err)]
async fn auth_context_from_request_with_state<T>(
    state: &AppState,
    request: &Request<T>,
) -> Result<AuthContext, Status> {
    if let Some(auth) = request.extensions().get::<AuthContext>() {
        return Ok(auth.clone());
    }

    let mut headers = HeaderMap::new();
    if let Some(value) = request.metadata().get("authorization") {
        let value_str = value
            .to_str()
            .map_err(|_| Status::unauthenticated("invalid auth token"))?;
        let header_value = HeaderValue::from_str(value_str)
            .map_err(|_| Status::unauthenticated("invalid auth token"))?;
        headers.insert("authorization", header_value);
    }

    let auth = crate::auth::auth_context_from_headers_with_state(&headers, state)
        .await
        .map_err(|_| Status::unauthenticated("invalid auth token"))?;

    if let Err(rate) = enforce_rate_limit(&auth) {
        return Err(Status::resource_exhausted(format!(
            "rate limit exceeded ({}/{}s); retry in {}s",
            rate.max_requests, rate.window_secs, rate.retry_after_secs
        )));
    }

    Ok(auth)
}

#[allow(clippy::result_large_err)]
fn ensure_read(auth: &AuthContext) -> Result<(), Status> {
    if auth.can_read() {
        Ok(())
    } else {
        Err(Status::permission_denied("read permission required"))
    }
}

#[allow(clippy::result_large_err)]
fn ensure_write(auth: &AuthContext) -> Result<(), Status> {
    if auth.can_write() {
        Ok(())
    } else {
        Err(Status::permission_denied("write permission required"))
    }
}

#[allow(clippy::result_large_err)]
fn ensure_namespace(auth: &AuthContext, namespace: &str) -> Result<(), Status> {
    if auth.allows_namespace(namespace) {
        Ok(())
    } else {
        Err(Status::permission_denied(format!(
            "namespace '{namespace}' is not permitted"
        )))
    }
}

#[allow(clippy::result_large_err)]
fn scoped_namespace_grpc(
    auth: &AuthContext,
    requested_namespace: Option<String>,
) -> Result<Option<String>, Status> {
    if auth.is_admin() {
        return Ok(requested_namespace);
    }

    match (&auth.namespace, requested_namespace) {
        (None, requested) => Ok(requested),
        (Some(allowed), Some(requested)) => {
            if requested == *allowed {
                Ok(Some(requested))
            } else {
                Err(Status::permission_denied(format!(
                    "namespace '{requested}' is not permitted"
                )))
            }
        }
        (Some(allowed), None) => Ok(Some(allowed.clone())),
    }
}

#[allow(clippy::result_large_err)]
fn ensure_vault_unsealed(state: &AppState) -> Result<(), Status> {
    if state.engine.config.sealed_mode && !state.engine.keychain.is_unsealed_sync() {
        get_metrics().incr_vault_sealed_grpc_blocked();
        return Err(Status::unavailable("Vault sealed - please unseal"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::AppState;
    use hx_engine::config::EngineConfig;
    use hx_engine::engine::HelixEngine;
    use tempfile::TempDir;

    #[tokio::test]
    async fn auth_context_from_request_with_state_resolves_access_key() {
        let temp_dir = TempDir::new().expect("temp dir");
        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.embedding.provider = "noop".into();
        let engine = HelixEngine::init(config).await.expect("engine init");
        let state = AppState::new(Arc::new(engine));

        let templates = state
            .engine
            .list_permission_templates(10, 0)
            .await
            .expect("templates");
        let template_id = templates.first().expect("template exists").id;
        let (_key, token) = state
            .engine
            .create_access_key(template_id, Some("grpc-test".into()), None)
            .await
            .expect("create access key");

        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", format!("Bearer {token}").parse().unwrap());

        let auth = auth_context_from_request_with_state(&state, &request)
            .await
            .expect("auth ok");

        assert!(auth.subject.unwrap_or_default().starts_with("access-key:"));
    }

    #[tokio::test]
    async fn auth_context_from_request_with_state_resolves_consumer_token() {
        let temp_dir = TempDir::new().expect("temp dir");
        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.embedding.provider = "noop".into();
        let engine = HelixEngine::init(config).await.expect("engine init");
        let state = AppState::new(Arc::new(engine));

        let (_profile, token) = state
            .engine
            .create_consumer("grpc-consumer", None)
            .await
            .expect("create consumer");

        let mut request = Request::new(());
        request
            .metadata_mut()
            .insert("authorization", format!("Bearer {token}").parse().unwrap());

        let auth = auth_context_from_request_with_state(&state, &request)
            .await
            .expect("auth ok");

        assert_eq!(auth.consumer_name.as_deref(), Some("grpc-consumer"));
    }

    #[tokio::test]
    async fn ensure_vault_unsealed_returns_unavailable_when_sealed() {
        let temp_dir = TempDir::new().expect("temp dir");
        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.embedding.provider = "noop".into();
        config.sealed_mode = true;
        let engine = HelixEngine::init(config).await.expect("engine init");
        engine
            .keychain
            .initialize_vault("test-password", false, "grpc-test")
            .await
            .expect("vault initialized");
        engine
            .keychain
            .seal("grpc-test")
            .await
            .expect("vault sealed");
        let state = AppState::new(Arc::new(engine));

        let err = ensure_vault_unsealed(&state).expect_err("sealed vault must fail");
        assert_eq!(err.code(), tonic::Code::Unavailable);
        assert_eq!(err.message(), "Vault sealed - please unseal");
    }

    #[tokio::test]
    async fn ensure_vault_unsealed_allows_unsealed_state() {
        let temp_dir = TempDir::new().expect("temp dir");
        let mut config = EngineConfig::default();
        config.data_dir = temp_dir.path().to_string_lossy().to_string();
        config.embedding.provider = "noop".into();
        config.sealed_mode = true;
        let engine = HelixEngine::init(config).await.expect("engine init");
        engine
            .keychain
            .initialize_vault("test-password", false, "grpc-test")
            .await
            .expect("vault initialized");
        engine
            .keychain
            .unseal("test-password", "grpc-test")
            .await
            .expect("vault unsealed");
        let state = AppState::new(Arc::new(engine));

        ensure_vault_unsealed(&state).expect("unsealed vault should pass");
    }
}

#[tonic::async_trait]
impl HelixService for HelixGrpc {
    async fn store_node(
        &self,
        request: Request<StoreNodeRequest>,
    ) -> Result<Response<StoreNodeResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_write(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();
        let kind: NodeKind = req
            .kind
            .parse()
            .map_err(|e: String| Status::invalid_argument(e))?;
        let metadata = parse_metadata_json(req.metadata_json.as_deref())?;
        let requested_namespace = if req.namespace.is_empty() {
            None
        } else {
            Some(req.namespace.as_str())
        };
        validate_node_payload(
            kind,
            req.title.as_deref(),
            &req.content,
            req.source.as_deref(),
            requested_namespace,
            &req.tags,
            req.importance,
            Some(&metadata),
        )
        .map_err(Status::invalid_argument)?;

        let mut node = KnowledgeNode::new(kind, req.content);
        if let Some(title) = req.title {
            node = node.with_title(title);
        }
        if let Some(source) = req.source {
            node = node.with_source(source);
        }
        let requested_namespace = if req.namespace.is_empty() {
            None
        } else {
            Some(req.namespace)
        };
        let namespace = requested_namespace
            .or_else(|| auth.namespace.clone())
            .unwrap_or_else(|| "default".to_string());
        ensure_namespace(&auth, &namespace)?;
        enforce_namespace_quota(&self.state.engine, &namespace)
            .await
            .map_err(map_namespace_quota_status)?;
        node = node.with_namespace(namespace);
        if !req.tags.is_empty() {
            node = node.with_tags(req.tags);
        }
        if let Some(importance) = req.importance {
            node = node.with_importance(importance);
        }
        if !metadata.is_empty() {
            node.metadata = metadata;
        }

        let stored = self
            .state
            .engine
            .store_node(node)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        self.state
            .notify_change(&stored.id.to_string(), "create", Some(&stored.namespace));

        Ok(Response::new(StoreNodeResponse {
            node: Some(node_to_proto(&stored)),
        }))
    }

    async fn get_node(
        &self,
        request: Request<GetNodeRequest>,
    ) -> Result<Response<GetNodeResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_read(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();
        let uuid = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("invalid UUID: {e}")))?;

        let node = self
            .state
            .engine
            .get_node(uuid)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let node = match node {
            Some(node) => {
                ensure_namespace(&auth, &node.namespace)?;
                Some(node_to_proto(&node))
            }
            None => None,
        };

        Ok(Response::new(GetNodeResponse { node }))
    }

    async fn update_node(
        &self,
        request: Request<UpdateNodeRequest>,
    ) -> Result<Response<UpdateNodeResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_write(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();
        let proto_node = req
            .node
            .ok_or_else(|| Status::invalid_argument("node required"))?;

        let uuid = Uuid::parse_str(&proto_node.id)
            .map_err(|e| Status::invalid_argument(format!("invalid UUID: {e}")))?;

        let existing = self
            .state
            .engine
            .get_node(uuid)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found("node not found"))?;
        let existing_namespace = existing.namespace.clone();
        ensure_namespace(&auth, &existing_namespace)?;

        let mut node = existing;

        if !proto_node.kind.trim().is_empty() {
            node.kind = proto_node
                .kind
                .parse()
                .map_err(|e: String| Status::invalid_argument(e))?;
        }
        node.content = proto_node.content;
        node.title = proto_node.title;
        node.source = proto_node.source;
        node.tags = proto_node.tags;
        node.importance = proto_node.importance.clamp(0.0, 1.0);
        node.temporal.expires_at =
            parse_optional_datetime(proto_node.expires_at.as_deref(), "expires_at")?;
        node.metadata = parse_metadata_json(Some(&proto_node.metadata_json))?;
        if !auth.is_admin() {
            node.namespace = auth
                .namespace
                .clone()
                .unwrap_or_else(|| node.namespace.clone());
        } else if !proto_node.namespace.is_empty() {
            node.namespace = proto_node.namespace;
        }
        validate_node_payload(
            node.kind,
            node.title.as_deref(),
            &node.content,
            node.source.as_deref(),
            Some(&node.namespace),
            &node.tags,
            Some(node.importance),
            Some(&node.metadata),
        )
        .map_err(Status::invalid_argument)?;
        if node.namespace != existing_namespace {
            enforce_namespace_quota(&self.state.engine, &node.namespace)
                .await
                .map_err(map_namespace_quota_status)?;
        }
        ensure_namespace(&auth, &node.namespace)?;
        node.temporal.updated_at = chrono::Utc::now();
        node.temporal.version += 1;

        let updated = self
            .state
            .engine
            .update_node(node)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        self.state
            .notify_change(&updated.id.to_string(), "update", Some(&updated.namespace));

        Ok(Response::new(UpdateNodeResponse {
            node: Some(node_to_proto(&updated)),
        }))
    }

    async fn delete_node(
        &self,
        request: Request<DeleteNodeRequest>,
    ) -> Result<Response<DeleteNodeResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_write(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();
        let uuid = Uuid::parse_str(&req.id)
            .map_err(|e| Status::invalid_argument(format!("invalid UUID: {e}")))?;

        if let Some(node) = self
            .state
            .engine
            .get_node(uuid)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
        {
            ensure_namespace(&auth, &node.namespace)?;
        }

        let deleted = self
            .state
            .engine
            .delete_node(uuid)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        if deleted {
            self.state.notify_change(&req.id, "delete", None);
        }

        Ok(Response::new(DeleteNodeResponse { deleted }))
    }

    async fn recall(
        &self,
        request: Request<RecallRequest>,
    ) -> Result<Response<RecallResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_read(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();
        validate_query_text("text", &req.text).map_err(Status::invalid_argument)?;

        let strategy: SearchStrategy = if req.strategy.is_empty() {
            SearchStrategy::Hybrid
        } else {
            req.strategy
                .parse()
                .map_err(|e: String| Status::invalid_argument(e))?
        };

        let kinds = parse_kind_list(&req.kinds).map_err(Status::invalid_argument)?;

        let requested_namespace = req.namespace.filter(|ns| !ns.is_empty());
        let namespace = scoped_namespace_grpc(&auth, requested_namespace)?;
        let limit = if req.limit > 0 {
            req.limit as usize
        } else {
            10
        };
        validate_recall_limit(limit).map_err(Status::invalid_argument)?;

        let query = MemoryQuery {
            text: req.text,
            strategy,
            limit,
            min_score: req.min_score,
            filters: QueryFilters {
                namespace,
                kinds,
                tags: if req.tags.is_empty() {
                    None
                } else {
                    Some(req.tags)
                },
                ..Default::default()
            },
            rewrite_strategy: None,
            session_id: None,
        };

        let results = self
            .state
            .engine
            .recall(&query)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let protos: Vec<SearchResultProto> = results
            .into_iter()
            .map(|r| SearchResultProto {
                node: Some(node_to_proto(&r.node)),
                score: r.score,
                match_source: match_source_to_str(r.match_source).to_string(),
            })
            .collect();

        Ok(Response::new(RecallResponse { results: protos }))
    }

    async fn list_nodes(
        &self,
        request: Request<ListNodesRequest>,
    ) -> Result<Response<ListNodesResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_read(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();

        let kinds = parse_kind_list(&req.kinds).map_err(Status::invalid_argument)?;

        let requested_namespace = req.namespace.filter(|ns| !ns.is_empty());
        let namespace = scoped_namespace_grpc(&auth, requested_namespace)?;
        let limit = if req.limit > 0 {
            req.limit as usize
        } else {
            50
        };
        validate_list_limit(limit).map_err(Status::invalid_argument)?;

        let filters = QueryFilters {
            namespace,
            kinds,
            ..Default::default()
        };

        let offset = req.offset as usize;

        let nodes = self
            .state
            .engine
            .list_nodes(&filters, limit, offset)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let total = self
            .state
            .engine
            .store
            .nodes
            .count(&filters)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(ListNodesResponse {
            nodes: nodes.iter().map(node_to_proto).collect(),
            total: total as u64,
        }))
    }

    async fn add_relationship(
        &self,
        request: Request<AddRelationshipRequest>,
    ) -> Result<Response<AddRelationshipResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_write(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();

        let from = Uuid::parse_str(&req.from_node)
            .map_err(|e| Status::invalid_argument(format!("invalid from_node: {e}")))?;
        let to = Uuid::parse_str(&req.to_node)
            .map_err(|e| Status::invalid_argument(format!("invalid to_node: {e}")))?;
        let kind: RelationKind = req
            .kind
            .parse()
            .map_err(|e: String| Status::invalid_argument(e))?;

        let rel = Relationship::new(from, to, kind).with_weight(req.weight);
        let rel_id = rel.id;

        let from_node = self
            .state
            .engine
            .get_node(from)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found("from_node not found"))?;
        let to_node = self
            .state
            .engine
            .get_node(to)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found("to_node not found"))?;

        ensure_namespace(&auth, &from_node.namespace)?;
        ensure_namespace(&auth, &to_node.namespace)?;

        self.state
            .engine
            .add_relationship(rel)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(AddRelationshipResponse {
            id: rel_id.to_string(),
        }))
    }

    async fn get_neighbors(
        &self,
        request: Request<GetNeighborsRequest>,
    ) -> Result<Response<GetNeighborsResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_read(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let req = request.into_inner();
        let uuid = Uuid::parse_str(&req.node_id)
            .map_err(|e| Status::invalid_argument(format!("invalid UUID: {e}")))?;
        let depth = if req.depth > 0 { req.depth as usize } else { 2 };
        validate_depth(depth).map_err(Status::invalid_argument)?;

        let source_node = self
            .state
            .engine
            .get_node(uuid)
            .await
            .map_err(|e| Status::internal(e.to_string()))?
            .ok_or_else(|| Status::not_found("node not found"))?;
        ensure_namespace(&auth, &source_node.namespace)?;

        let neighbors = self
            .state
            .engine
            .get_neighbors(uuid, depth)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let mut visible_neighbors = Vec::new();
        for neighbor_id in neighbors {
            if let Some(node) = self
                .state
                .engine
                .get_node(neighbor_id)
                .await
                .map_err(|e| Status::internal(e.to_string()))?
            {
                if auth.allows_namespace(&node.namespace) {
                    visible_neighbors.push(neighbor_id.to_string());
                }
            }
        }

        Ok(Response::new(GetNeighborsResponse {
            neighbor_ids: visible_neighbors,
        }))
    }

    async fn health(
        &self,
        request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_read(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let count = self
            .state
            .engine
            .node_count()
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(HealthResponse {
            status: "ok".into(),
            node_count: count as u64,
            version: env!("CARGO_PKG_VERSION").into(),
        }))
    }

    type WatchChangesStream = Pin<Box<dyn Stream<Item = Result<ChangeEvent, Status>> + Send>>;

    async fn watch_changes(
        &self,
        request: Request<WatchChangesRequest>,
    ) -> Result<Response<Self::WatchChangesStream>, Status> {
        let auth = auth_context_from_request_with_state(&self.state, &request).await?;
        ensure_read(&auth)?;
        ensure_vault_unsealed(&self.state)?;

        let requested_namespace = request.into_inner().namespace;
        let namespace_filter = scoped_namespace_grpc(&auth, requested_namespace)?;
        let mut rx = self.state.change_tx.subscribe();

        let stream = async_stream::try_stream! {
            loop {
                match rx.recv().await {
                    Ok(notification) => {
                        if let Some(ref namespace) = namespace_filter {
                            if notification.namespace.as_deref() != Some(namespace.as_str()) {
                                continue;
                            }
                        }
                        yield ChangeEvent {
                            node_id: notification.node_id,
                            operation: notification.operation,
                            timestamp: notification.timestamp,
                            node: None,
                        };
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        tracing::warn!("change stream lagged by {n} events");
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        break;
                    }
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }
}

// ---------------------------------------------------------------------------
// Keychain gRPC service
// ---------------------------------------------------------------------------

use proto::keychain_service_server::KeychainService;

pub struct KeychainGrpc {
    state: Arc<AppState>,
}

impl KeychainGrpc {
    pub fn new(state: Arc<AppState>) -> Self {
        Self { state }
    }
}

async fn ensure_admin(
    state: &AppState,
    req: &Request<impl std::fmt::Debug>,
) -> Result<AuthContext, Status> {
    let auth = auth_context_from_request_with_state(state, req).await?;
    if !auth.is_admin() {
        return Err(Status::permission_denied("admin only"));
    }
    Ok(auth)
}

fn map_keychain_status(err: hx_core::HxError) -> Status {
    match &err {
        hx_core::HxError::VaultSealed => Status::failed_precondition(err.to_string()),
        hx_core::HxError::KeychainNotFound(_) => Status::not_found(err.to_string()),
        hx_core::HxError::KeychainInvalidPassword => Status::unauthenticated(err.to_string()),
        hx_core::HxError::KeychainNotInitialized => Status::failed_precondition(err.to_string()),
        hx_core::HxError::KeychainAlreadyInitialized => {
            Status::already_exists(err.to_string())
        }
        _ => Status::internal(err.to_string()),
    }
}

async fn log_unseal_attempt(
    state: &AppState,
    subject: &str,
    method: &str,
    outcome: &str,
    reason: Option<&str>,
) {
    let logic = match reason {
        Some(reason) => {
            format!("subject={subject} method={method} outcome={outcome} reason={reason}")
        }
        None => format!("subject={subject} method={method} outcome={outcome}"),
    };
    let entry = ChronicleEntry::new("unseal_attempt", logic);
    if let Err(err) = state.engine.log_chronicle(&entry).await {
        tracing::warn!(error = %err, "failed to log unseal_attempt chronicle entry");
    }
}

fn map_unseal_rate_limit_status(exceeded: RateLimitExceeded) -> Status {
    Status::resource_exhausted(format!(
        "unseal rate limit exceeded: {} failures per {}s (retry in {}s)",
        exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
    ))
}

async fn enforce_unseal_failure_backoff(
    state: &AppState,
    subject: &str,
    method: &str,
) -> Result<(), Status> {
    if let Err(exceeded) = enforce_keychain_unseal_failure_backoff(subject) {
        get_metrics().incr_vault_unseal_rate_limited();
        let reason = format!(
            "rate_limited:max={} window={} retry_after={}",
            exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
        );
        log_unseal_attempt(state, subject, method, "fail", Some(reason.as_str())).await;
        return Err(map_unseal_rate_limit_status(exceeded));
    }
    Ok(())
}

async fn log_and_record_unseal_failure(
    state: &AppState,
    subject: &str,
    method: &str,
    reason: &str,
) -> Result<(), Status> {
    get_metrics().incr_vault_unseal_failure();
    tracing::warn!(subject, method, reason, "vault unseal failed");
    log_unseal_attempt(state, subject, method, "fail", Some(reason)).await;
    if let Err(exceeded) = record_keychain_unseal_failure(subject) {
        get_metrics().incr_vault_unseal_rate_limited();
        let rate_reason = format!(
            "rate_limited:max={} window={} retry_after={}",
            exceeded.max_requests, exceeded.window_secs, exceeded.retry_after_secs
        );
        log_unseal_attempt(state, subject, method, "fail", Some(rate_reason.as_str())).await;
        return Err(map_unseal_rate_limit_status(exceeded));
    }
    Ok(())
}

fn failpoint_matches_data_dir(raw: &str, data_dir: &str) -> bool {
    let value = raw.trim();
    if value.is_empty() {
        return false;
    }
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "0" | "false" | "no" | "off" => false,
        "1" | "true" | "yes" | "on" => true,
        _ => value == data_dir,
    }
}

fn post_unseal_failpoint_enabled(state: &AppState, key: &str) -> bool {
    if !cfg!(debug_assertions) {
        return false;
    }
    let Ok(raw) = std::env::var(key) else {
        return false;
    };
    failpoint_matches_data_dir(&raw, &state.engine.config.data_dir)
}

async fn run_post_unseal_maintenance(
    state: &AppState,
    subject: &str,
    method: &str,
) -> Result<(), Status> {
    if post_unseal_failpoint_enabled(state, "HELIX_TEST_FAIL_POST_UNSEAL_MIGRATE") {
        let err = hx_core::HxError::Storage("post-unseal migrate failpoint triggered".into());
        get_metrics().incr_vault_migration_failure();
        let reason = format!("post_unseal_migrate_failed:{err}");
        log_and_record_unseal_failure(state, subject, method, reason.as_str()).await?;
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_status(err));
    }

    if let Err(err) = state.engine.migrate_sealed_storage().await {
        get_metrics().incr_vault_migration_failure();
        let reason = format!("post_unseal_migrate_failed:{err}");
        log_and_record_unseal_failure(state, subject, method, reason.as_str()).await?;
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_status(err));
    }
    get_metrics().incr_vault_migration_success();

    if post_unseal_failpoint_enabled(state, "HELIX_TEST_FAIL_POST_UNSEAL_REBUILD") {
        let err = hx_core::HxError::Storage("post-unseal rebuild failpoint triggered".into());
        get_metrics().incr_vault_rebuild_failure();
        let reason = format!("post_unseal_rebuild_failed:{err}");
        log_and_record_unseal_failure(state, subject, method, reason.as_str()).await?;
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_status(err));
    }

    if let Err(err) = state.engine.rebuild_runtime_indexes().await {
        get_metrics().incr_vault_rebuild_failure();
        let reason = format!("post_unseal_rebuild_failed:{err}");
        log_and_record_unseal_failure(state, subject, method, reason.as_str()).await?;
        let _ = state.engine.keychain.seal("system").await;
        return Err(map_keychain_status(err));
    }
    get_metrics().incr_vault_rebuild_success();

    Ok(())
}

#[tonic::async_trait]
impl KeychainService for KeychainGrpc {
    async fn init_vault(
        &self,
        request: Request<proto::InitVaultRequest>,
    ) -> Result<Response<proto::InitVaultResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        let req = request.into_inner();
        let engine = &self.state.engine;

        engine
            .keychain
            .initialize_vault(&req.password, req.macos_bridge, "grpc")
            .await
            .map_err(map_keychain_status)?;

        let (_, meta) = engine
            .keychain
            .vault_status()
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::InitVaultResponse {
            status: "initialized".into(),
            key_epoch: meta.map(|m| m.key_epoch).unwrap_or(0),
        }))
    }

    async fn unseal(
        &self,
        request: Request<proto::UnsealRequest>,
    ) -> Result<Response<proto::UnsealResponse>, Status> {
        let auth = ensure_admin(&self.state, &request).await?;
        let req = request.into_inner();
        let engine = &self.state.engine;
        let subject = auth.subject.as_deref().unwrap_or("grpc");
        let method = if req.from_macos_keychain {
            "macos_keychain"
        } else {
            "preferred"
        };
        enforce_unseal_failure_backoff(&self.state, subject, method).await?;

        let unseal_result = if req.from_macos_keychain {
            engine.keychain.unseal_from_macos_keychain(subject).await
        } else {
            engine
                .keychain
                .unseal_with_preferred_master_key(
                    req.password.as_deref().filter(|value| !value.is_empty()),
                    subject,
                )
                .await
                .map(|_| ())
        };

        match unseal_result {
            Ok(()) => {
                if self.state.engine.keychain.degraded_security_mode() {
                    tracing::warn!(
                        "vault unsealed in degraded security mode (passphrase fallback)"
                    );
                }
                run_post_unseal_maintenance(&self.state, subject, method).await?;
                log_unseal_attempt(&self.state, subject, method, "success", None).await;
            }
            Err(err) => {
                let reason = err.to_string();
                log_and_record_unseal_failure(&self.state, subject, method, reason.as_str())
                    .await?;
                return Err(map_keychain_status(err));
            }
        }

        Ok(Response::new(proto::UnsealResponse {
            status: "unsealed".into(),
        }))
    }

    async fn seal(
        &self,
        request: Request<proto::SealRequest>,
    ) -> Result<Response<proto::SealResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;

        self.state
            .engine
            .keychain
            .seal("grpc")
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::SealResponse {
            status: "sealed".into(),
        }))
    }

    async fn get_vault_status(
        &self,
        request: Request<proto::GetVaultStatusRequest>,
    ) -> Result<Response<proto::KeychainStatusResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;

        let (state, meta) = self
            .state
            .engine
            .keychain
            .vault_status()
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::KeychainStatusResponse {
            status: "ok".into(),
            state: state.as_str().to_string(),
            key_epoch: meta.map(|m| m.key_epoch).unwrap_or(0),
        }))
    }

    async fn rotate_key(
        &self,
        request: Request<proto::RotateKeyRequest>,
    ) -> Result<Response<proto::RotateKeyResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;
        let req = request.into_inner();

        self.state
            .engine
            .keychain
            .rotate_master_key(&req.new_password, req.grace_hours, "grpc")
            .await
            .map_err(map_keychain_status)?;

        let (_, meta) = self
            .state
            .engine
            .keychain
            .vault_status()
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::RotateKeyResponse {
            status: "rotated".into(),
            new_epoch: meta.map(|m| m.key_epoch).unwrap_or(0),
        }))
    }

    async fn store_credential(
        &self,
        request: Request<proto::StoreCredentialRequest>,
    ) -> Result<Response<proto::CredentialResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;
        let req = request.into_inner();

        let domain_id: Uuid = req
            .domain_id
            .parse()
            .map_err(|_| Status::invalid_argument("invalid domain_id UUID"))?;

        let expires_at = req
            .expires_at
            .as_deref()
            .map(|s| {
                chrono::DateTime::parse_from_rfc3339(s)
                    .map(|dt| dt.with_timezone(&chrono::Utc))
                    .map_err(|_| Status::invalid_argument("invalid expires_at"))
            })
            .transpose()?;

        let cred = self
            .state
            .engine
            .keychain
            .store_credential(
                domain_id,
                &req.name,
                &req.kind,
                req.value.as_bytes(),
                req.tags,
                expires_at,
                "grpc",
            )
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::CredentialResponse {
            id: cred.id.to_string(),
            name: cred.name,
            kind: cred.kind,
            domain_id: cred.domain_id.to_string(),
            state: cred.state.as_str().to_string(),
            created_at: cred.created_at.to_rfc3339(),
        }))
    }

    async fn read_credential(
        &self,
        request: Request<proto::ReadCredentialRequest>,
    ) -> Result<Response<proto::ReadCredentialResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;
        let req = request.into_inner();

        let id: Uuid = req
            .id
            .parse()
            .map_err(|_| Status::invalid_argument("invalid credential id UUID"))?;

        let (cred, plaintext, _alerts) = self
            .state
            .engine
            .keychain
            .read_credential(id, &req.subject)
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::ReadCredentialResponse {
            id: cred.id.to_string(),
            name: cred.name,
            kind: cred.kind,
            domain_id: cred.domain_id.to_string(),
            value: String::from_utf8_lossy(&plaintext).to_string(),
            state: cred.state.as_str().to_string(),
            created_at: cred.created_at.to_rfc3339(),
            expires_at: cred.expires_at.map(|dt| dt.to_rfc3339()),
        }))
    }

    async fn list_credentials(
        &self,
        request: Request<proto::ListCredentialsRequest>,
    ) -> Result<Response<proto::ListCredentialsResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;
        let req = request.into_inner();

        let domain_id = req
            .domain
            .as_deref()
            .map(|s| s.parse::<Uuid>())
            .transpose()
            .map_err(|_| Status::invalid_argument("invalid domain UUID"))?;

        let state_filter = req
            .state
            .as_deref()
            .map(|s| {
                s.parse::<hx_core::model::keychain::CredentialState>()
                    .map_err(|_| Status::invalid_argument("invalid state"))
            })
            .transpose()?;

        let creds = self
            .state
            .engine
            .keychain
            .list_credentials(
                domain_id,
                state_filter,
                req.limit as usize,
                req.offset as usize,
            )
            .await
            .map_err(map_keychain_status)?;

        let total = creds.len() as u64;
        let credentials = creds
            .into_iter()
            .map(|c| proto::CredentialResponse {
                id: c.id.to_string(),
                name: c.name,
                kind: c.kind,
                domain_id: c.domain_id.to_string(),
                state: c.state.as_str().to_string(),
                created_at: c.created_at.to_rfc3339(),
            })
            .collect();

        Ok(Response::new(proto::ListCredentialsResponse {
            credentials,
            total,
        }))
    }

    async fn destroy_credential(
        &self,
        request: Request<proto::DestroyCredentialRequest>,
    ) -> Result<Response<proto::DestroyCredentialResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;
        let req = request.into_inner();

        let id: Uuid = req
            .id
            .parse()
            .map_err(|_| Status::invalid_argument("invalid credential id UUID"))?;

        self.state
            .engine
            .keychain
            .destroy_credential(id, "grpc")
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::DestroyCredentialResponse {
            status: "destroyed".into(),
        }))
    }

    async fn generate_proof(
        &self,
        request: Request<proto::GenerateProofRequest>,
    ) -> Result<Response<proto::ZkProofResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;
        let req = request.into_inner();

        let credential_id: Uuid = req
            .credential_id
            .parse()
            .map_err(|_| Status::invalid_argument("invalid credential_id UUID"))?;

        let proof = self
            .state
            .engine
            .keychain
            .generate_proof(credential_id, &req.nonce, "grpc")
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::ZkProofResponse {
            proof: proof.proof,
            expires_at: proof.expires_at.to_rfc3339(),
        }))
    }

    async fn verify_proof(
        &self,
        request: Request<proto::VerifyProofRequest>,
    ) -> Result<Response<proto::VerifyProofResponse>, Status> {
        let _auth = ensure_admin(&self.state, &request).await?;
        ensure_vault_unsealed(&self.state)?;
        let req = request.into_inner();

        // Deserialize the proof from JSON
        let proof: hx_core::model::keychain::ZkAccessProof = serde_json::from_str(&req.proof)
            .map_err(|e| Status::invalid_argument(format!("invalid proof JSON: {e}")))?;

        let valid = self
            .state
            .engine
            .keychain
            .verify_proof(&proof, "grpc")
            .await
            .map_err(map_keychain_status)?;

        Ok(Response::new(proto::VerifyProofResponse {
            valid,
            credential_id: if valid {
                Some(proof.credential_id.to_string())
            } else {
                None
            },
        }))
    }
}
