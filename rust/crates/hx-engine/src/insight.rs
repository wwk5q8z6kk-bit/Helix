//! Semantic Insight Engine — background discovery of cross-domain patterns,
//! embedding-space clusters, and knowledge gaps.
//!
//! This module wraps [`ProactiveEngine`] with a scheduler that runs periodic
//! scans and a DBSCAN-like clustering algorithm operating on embeddings rather
//! than tags alone.

use crate::engine::HelixEngine;
use hx_core::{InsightType, KnowledgeNode, MvResult, ProactiveInsight, QueryFilters};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use uuid::Uuid;

/// Configuration for the Semantic Insight Engine.
#[derive(Debug, Clone)]
pub struct InsightConfig {
    /// Cosine-similarity threshold for cross-domain connections.
    pub cross_domain_threshold: f64,
    /// Minimum similarity for DBSCAN neighbor detection.
    pub cluster_eps: f64,
    /// Minimum cluster size for DBSCAN.
    pub cluster_min_points: usize,
    /// How many recent nodes to scan per batch.
    pub scan_batch_size: usize,
    /// Interval between automatic scans (seconds). 0 = disabled.
    pub scan_interval_secs: u64,
}

impl Default for InsightConfig {
    fn default() -> Self {
        Self {
            cross_domain_threshold: 0.82,
            cluster_eps: 0.75,
            cluster_min_points: 3,
            scan_batch_size: 50,
            scan_interval_secs: 600, // 10 minutes
        }
    }
}

/// Semantic Insight Engine that discovers patterns across the knowledge vault.
///
/// Follows the `Weak<HelixEngine>` pattern to avoid reference cycles.
pub struct InsightEngine {
    engine: tokio::sync::OnceCell<Weak<HelixEngine>>,
    config: InsightConfig,
}

impl InsightEngine {
    pub fn new(config: InsightConfig) -> Self {
        Self {
            engine: tokio::sync::OnceCell::new(),
            config,
        }
    }

    pub fn set_engine(&self, engine: Arc<HelixEngine>) {
        if self.engine.set(Arc::downgrade(&engine)).is_err() {
            tracing::warn!("InsightEngine engine already set");
        }
    }

    fn engine(&self) -> Option<Arc<HelixEngine>> {
        self.engine.get().and_then(Weak::upgrade)
    }

    pub fn config(&self) -> &InsightConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Full scan: combines all insight detectors
    // -----------------------------------------------------------------------

    /// Run a full insight scan combining cross-domain, clustering, gaps,
    /// stale-knowledge, and trend detection. Returns newly-persisted insights.
    pub async fn full_scan(&self, namespace: Option<&str>) -> MvResult<Vec<ProactiveInsight>> {
        let engine = match self.engine() {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let mut all_insights = Vec::new();

        // 1) Standard proactive insights (clusters, stale, trends, connections)
        let ns = namespace.unwrap_or("default").to_string();
        match engine.proactive.generate_insights(ns.clone()).await {
            Ok(mut insights) => all_insights.append(&mut insights),
            Err(e) => tracing::warn!(error = %e, "proactive insight generation failed"),
        }

        // 2) Embedding-space DBSCAN clusters
        match self.detect_embedding_clusters(namespace).await {
            Ok(mut insights) => all_insights.append(&mut insights),
            Err(e) => tracing::warn!(error = %e, "embedding cluster detection failed"),
        }

        // 3) Knowledge gap detection
        match engine.proactive.find_knowledge_gaps(namespace).await {
            Ok(gaps) => {
                // Deduplicate before adding
                for gap in gaps {
                    all_insights.push(gap);
                }
            }
            Err(e) => tracing::warn!(error = %e, "knowledge gap detection failed"),
        }

        // 4) Cross-domain connections (if we can discover multiple namespaces)
        match self.discover_namespaces_and_scan().await {
            Ok(mut insights) => all_insights.append(&mut insights),
            Err(e) => tracing::warn!(error = %e, "cross-domain scan failed"),
        }

        // Deduplicate against existing insights
        let deduped = self.deduplicate_and_persist(all_insights).await?;

        tracing::info!(
            count = deduped.len(),
            "insight full scan complete"
        );

        Ok(deduped)
    }

    // -----------------------------------------------------------------------
    // DBSCAN-like clustering on embedding vectors
    // -----------------------------------------------------------------------

    /// Detect clusters of semantically similar nodes using a simplified DBSCAN
    /// on embedding vectors. Nodes in a cluster that share no tags or links
    /// are surfaced as `UnlinkedCluster` insights.
    pub async fn detect_embedding_clusters(
        &self,
        namespace: Option<&str>,
    ) -> MvResult<Vec<ProactiveInsight>> {
        let engine = match self.engine() {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let filters = QueryFilters {
            namespace: namespace.map(|s| s.to_string()),
            ..Default::default()
        };
        let nodes = engine
            .list_nodes(&filters, self.config.scan_batch_size, 0)
            .await?;

        if nodes.len() < self.config.cluster_min_points {
            return Ok(Vec::new());
        }

        // Embed all nodes
        let texts: Vec<String> = nodes
            .iter()
            .map(|n| {
                format!(
                    "{} {}",
                    n.title.as_deref().unwrap_or(""),
                    n.content.chars().take(300).collect::<String>()
                )
            })
            .collect();

        let embeddings = engine.store.embedder.embed_batch(&texts).await?;

        if embeddings.len() != nodes.len() {
            return Ok(Vec::new());
        }

        // Simplified DBSCAN
        let clusters = dbscan(
            &embeddings,
            self.config.cluster_eps,
            self.config.cluster_min_points,
        );

        let mut insights = Vec::new();
        for cluster_indices in &clusters {
            let cluster_nodes: Vec<&KnowledgeNode> =
                cluster_indices.iter().map(|&i| &nodes[i]).collect();

            // Check if the cluster has shared tags — if not, it's an unlinked cluster
            let all_tags: HashSet<&String> =
                cluster_nodes.iter().flat_map(|n| n.tags.iter()).collect();
            let shared_tags: Vec<&String> = all_tags
                .iter()
                .filter(|tag| {
                    cluster_nodes
                        .iter()
                        .filter(|n| n.tags.contains(**tag))
                        .count()
                        >= 2
                })
                .copied()
                .collect();

            // Check if they're already linked in the graph
            let node_ids: Vec<Uuid> = cluster_nodes.iter().map(|n| n.id).collect();
            let already_linked = self.any_linked(&engine, &node_ids).await?;

            if shared_tags.is_empty() && !already_linked {
                // Suggest a tag based on the most common individual tag
                let mut tag_counts: HashMap<&String, usize> = HashMap::new();
                for n in &cluster_nodes {
                    for tag in &n.tags {
                        *tag_counts.entry(tag).or_insert(0) += 1;
                    }
                }
                let suggested_tag = tag_counts
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(t, _)| t.clone())
                    .unwrap_or_else(|| "untagged-cluster".to_string());

                let titles: Vec<String> = cluster_nodes
                    .iter()
                    .take(3)
                    .filter_map(|n| n.title.clone())
                    .collect();

                let insight = ProactiveInsight::new(
                    format!(
                        "Unlinked cluster of {} nodes",
                        cluster_nodes.len()
                    ),
                    format!(
                        "{} semantically similar nodes have no shared tags or links. \
                         Examples: {}. Suggested tag: '{}'.",
                        cluster_nodes.len(),
                        if titles.is_empty() {
                            "untitled nodes".to_string()
                        } else {
                            titles.join(", ")
                        },
                        suggested_tag,
                    ),
                    InsightType::UnlinkedCluster,
                )
                .with_related_nodes(node_ids)
                .with_importance(0.65);

                insights.push(insight);
            }
        }

        Ok(insights)
    }

    // -----------------------------------------------------------------------
    // Cross-domain auto-discovery
    // -----------------------------------------------------------------------

    /// Discover all namespaces in the vault and run cross-domain connection
    /// detection across them.
    async fn discover_namespaces_and_scan(&self) -> MvResult<Vec<ProactiveInsight>> {
        let engine = match self.engine() {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        // Get a sample of nodes to discover namespaces
        let nodes = engine
            .list_nodes(&QueryFilters::default(), 200, 0)
            .await?;

        let namespaces: Vec<String> = nodes
            .iter()
            .map(|n| n.namespace.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        if namespaces.len() < 2 {
            return Ok(Vec::new());
        }

        engine
            .proactive
            .detect_cross_domain_connections(
                &namespaces,
                self.config.cross_domain_threshold,
                10,
            )
            .await
    }

    // -----------------------------------------------------------------------
    // Background scheduler
    // -----------------------------------------------------------------------

    /// Start a background task that runs `full_scan` periodically.
    /// Returns a `JoinHandle` that can be used to cancel the task.
    pub fn start_background_scanner(
        self: &Arc<Self>,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let interval_secs = self.config.scan_interval_secs;
        if interval_secs == 0 {
            tracing::info!("insight background scanner disabled (interval=0)");
            return None;
        }

        let this = Arc::clone(self);
        let handle = tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(std::time::Duration::from_secs(interval_secs));
            // Skip the first immediate tick
            interval.tick().await;

            loop {
                interval.tick().await;
                tracing::debug!("running scheduled insight scan");
                match this.full_scan(None).await {
                    Ok(insights) => {
                        if !insights.is_empty() {
                            tracing::info!(
                                count = insights.len(),
                                "scheduled insight scan produced new insights"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "scheduled insight scan failed");
                    }
                }
            }
        });

        tracing::info!(
            interval_secs,
            "insight background scanner started"
        );
        Some(handle)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Check if any pair of nodes in the list are linked in the graph.
    async fn any_linked(
        &self,
        engine: &HelixEngine,
        node_ids: &[Uuid],
    ) -> MvResult<bool> {
        for (i, &id) in node_ids.iter().enumerate() {
            let neighbors = engine.get_neighbors(id, 1).await?;
            let neighbor_set: HashSet<Uuid> = neighbors.into_iter().collect();
            for &other_id in &node_ids[i + 1..] {
                if neighbor_set.contains(&other_id) {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    /// Deduplicate insights against existing ones and persist new ones.
    async fn deduplicate_and_persist(
        &self,
        insights: Vec<ProactiveInsight>,
    ) -> MvResult<Vec<ProactiveInsight>> {
        let engine = match self.engine() {
            Some(e) => e,
            None => return Ok(Vec::new()),
        };

        let existing = hx_core::AgenticStore::list_insights(&*engine.store.nodes, 500, 0).await?;
        let mut seen: HashSet<String> = existing
            .iter()
            .filter(|i| i.dismissed_at.is_none())
            .map(insight_signature)
            .collect();

        let mut persisted = Vec::new();
        for insight in insights {
            let sig = insight_signature(&insight);
            if !seen.insert(sig) {
                continue;
            }
            hx_core::AgenticStore::log_insight(&*engine.store.nodes, &insight).await?;
            persisted.push(insight);
        }

        Ok(persisted)
    }
}

impl Default for InsightEngine {
    fn default() -> Self {
        Self::new(InsightConfig::default())
    }
}

// ---------------------------------------------------------------------------
// DBSCAN implementation (simplified, in-memory)
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let (mut dot, mut norm_a, mut norm_b) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Simplified DBSCAN on cosine similarity.
///
/// Returns a list of clusters, each cluster being a vec of indices into `embeddings`.
/// Noise points (not in any cluster) are not returned.
fn dbscan(embeddings: &[Vec<f32>], eps: f64, min_points: usize) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    // -1 = unvisited, 0 = noise, 1..N = cluster id
    let mut labels: Vec<i32> = vec![-1; n];
    let mut cluster_id: i32 = 0;

    for i in 0..n {
        if labels[i] != -1 {
            continue;
        }

        let neighbors = region_query(embeddings, i, eps);
        if neighbors.len() < min_points {
            labels[i] = 0; // noise
            continue;
        }

        cluster_id += 1;
        labels[i] = cluster_id;

        let mut seed_set: Vec<usize> = neighbors;
        let mut j = 0;
        while j < seed_set.len() {
            let q = seed_set[j];
            if labels[q] == 0 {
                labels[q] = cluster_id;
            }
            if labels[q] == -1 {
                labels[q] = cluster_id;
                let q_neighbors = region_query(embeddings, q, eps);
                if q_neighbors.len() >= min_points {
                    for &nb in &q_neighbors {
                        if !seed_set.contains(&nb) {
                            seed_set.push(nb);
                        }
                    }
                }
            }
            j += 1;
        }
    }

    // Group by cluster
    let mut clusters: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        if label > 0 {
            clusters.entry(label).or_default().push(idx);
        }
    }

    clusters.into_values().collect()
}

/// Find all points within `eps` cosine similarity of point `p`.
fn region_query(embeddings: &[Vec<f32>], p: usize, eps: f64) -> Vec<usize> {
    let mut result = Vec::new();
    for (i, emb) in embeddings.iter().enumerate() {
        if cosine_similarity(&embeddings[p], emb) >= eps {
            result.push(i);
        }
    }
    result
}

/// Stable signature for deduplication (matches ProactiveEngine::insight_signature).
fn insight_signature(insight: &ProactiveInsight) -> String {
    let mut ids: Vec<String> = insight
        .related_node_ids
        .iter()
        .map(|id| id.to_string())
        .collect();
    ids.sort_unstable();
    format!(
        "{}|{}|{}|{}",
        insight.insight_type,
        insight.title,
        insight.content,
        ids.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-9);
    }

    #[test]
    fn dbscan_basic_clusters() {
        // Two tight clusters of 3 points each, well separated
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.99, 0.1, 0.0],
            vec![0.98, 0.0, 0.1],
            // Second cluster
            vec![0.0, 1.0, 0.0],
            vec![0.1, 0.99, 0.0],
            vec![0.0, 0.98, 0.1],
        ];
        let clusters = dbscan(&embeddings, 0.95, 2);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn dbscan_no_clusters_when_spread() {
        let embeddings = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let clusters = dbscan(&embeddings, 0.95, 2);
        assert!(clusters.is_empty());
    }

    #[test]
    fn dbscan_single_cluster() {
        let embeddings = vec![
            vec![1.0, 0.1, 0.0],
            vec![1.0, 0.0, 0.1],
            vec![0.9, 0.1, 0.1],
        ];
        let clusters = dbscan(&embeddings, 0.90, 2);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }
}
