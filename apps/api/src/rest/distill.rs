use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};

use hx_core::model::QueryFilters;
use hx_core::traits::NodeStore;
use hx_engine::distill::{ContentSnippet, DistillEngine, DistillRequest};

use crate::state::AppState;

/// POST /api/v1/distill — Run a distillation (namespace, temporal, or topic deep-dive)
pub async fn distill(
    State(state): State<Arc<AppState>>,
    Json(req): Json<DistillRequest>,
) -> impl IntoResponse {
    let engine = DistillEngine::new(state.engine.llm.clone());

    // Gather content snippets based on the request type
    let snippets: Vec<ContentSnippet> = match &req {
        DistillRequest::Namespace { namespace, max_nodes } => {
            let limit = max_nodes.unwrap_or(50);
            let filters = QueryFilters {
                namespace: Some(namespace.clone()),
                ..Default::default()
            };
            match state.engine.store.nodes.list(&filters, limit, 0).await {
                Ok(nodes) => nodes
                    .into_iter()
                    .map(|n| ContentSnippet {
                        title: n.title,
                        content: n.content,
                        created_at: n.temporal.created_at,
                        namespace: n.namespace,
                        tags: n.tags,
                    })
                    .collect(),
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": e.to_string()})),
                    )
                        .into_response()
                }
            }
        }
        DistillRequest::Temporal { days } => {
            let since = chrono::Utc::now() - chrono::Duration::days(*days as i64);
            let filters = QueryFilters {
                created_after: Some(since),
                ..Default::default()
            };
            match state.engine.store.nodes.list(&filters, 50, 0).await {
                Ok(nodes) => nodes
                    .into_iter()
                    .map(|n| ContentSnippet {
                        title: n.title,
                        content: n.content,
                        created_at: n.temporal.created_at,
                        namespace: n.namespace,
                        tags: n.tags,
                    })
                    .collect(),
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": e.to_string()})),
                    )
                        .into_response()
                }
            }
        }
        DistillRequest::TopicDeepDive { topic } => {
            let query = hx_core::model::MemoryQuery::new(topic.clone())
                .with_limit(20);
            match state.engine.recall.recall(&query).await {
                Ok(results) => results
                    .into_iter()
                    .map(|r| ContentSnippet {
                        title: r.node.title,
                        content: r.node.content,
                        created_at: r.node.temporal.created_at,
                        namespace: r.node.namespace,
                        tags: r.node.tags,
                    })
                    .collect(),
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({"error": e.to_string()})),
                    )
                        .into_response()
                }
            }
        }
    };

    let result = engine.distill(&req, &snippets).await;
    Json(result).into_response()
}
