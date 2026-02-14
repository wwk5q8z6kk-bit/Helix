use crate::engine::HelixEngine;
use crate::llm::{ChatMessage, CompletionParams};
use chrono::{Datelike, Duration, Utc};
use hx_core::{
    AgenticStore, InsightType, KnowledgeNode, MemoryQuery, MvResult, ProactiveInsight, QueryFilters,
};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};
use uuid::Uuid;

pub struct ProactiveEngine {
    engine: tokio::sync::OnceCell<Weak<HelixEngine>>,
}

impl ProactiveEngine {
    pub fn new() -> Self {
        tracing::info!("Initializing ProactiveEngine");
        Self {
            engine: tokio::sync::OnceCell::new(),
        }
    }

    pub fn set_engine(&self, engine: Arc<HelixEngine>) {
        if self.engine.set(Arc::downgrade(&engine)).is_err() {
            tracing::warn!("ProactiveEngine engine already set");
        }
    }

    fn engine(&self) -> Arc<HelixEngine> {
        self.engine
            .get()
            .and_then(Weak::upgrade)
            .expect("ProactiveEngine::engine called before set_engine")
    }

    /// Identify nodes related to the given basis node using graph and semantic relevance.
    pub async fn find_related_context(
        &self,
        basis_node_id: Uuid,
        limit: usize,
    ) -> MvResult<Vec<KnowledgeNode>> {
        let span = tracing::info_span!("find_related_context", basis_node_id = %basis_node_id);
        let _enter = span.enter();

        let engine = self.engine();
        let basis_node = match engine.get_node(basis_node_id).await? {
            Some(node) => node,
            None => return Ok(Vec::new()),
        };

        // Find neighbors in the graph.
        let neighbors = engine.get_neighbors(basis_node.id, 1).await?;

        // Find semantically similar nodes (RAG).
        let query_text = basis_node.title.as_deref().unwrap_or(&basis_node.content);
        let recall_query = MemoryQuery::new(query_text.to_string())
            .with_namespace(basis_node.namespace.clone())
            .with_limit(limit.saturating_mul(2).max(limit));
        let recall_results = engine.recall(&recall_query).await?;

        let mut related = Vec::new();
        let mut seen_ids = std::collections::HashSet::new();
        seen_ids.insert(basis_node.id);

        // Add neighbors first (strongest graph signal).
        for neighbor_id in neighbors {
            if seen_ids.insert(neighbor_id) {
                if let Some(neighbor_node) = engine.get_node(neighbor_id).await? {
                    related.push(neighbor_node);
                }
            }
        }

        // Add recall results
        for result in recall_results {
            if seen_ids.insert(result.node.id) {
                related.push(result.node);
            }
        }

        let results: Vec<_> = related.into_iter().take(limit).collect();
        tracing::info!(count = results.len(), "Context surfaced");
        Ok(results)
    }

    /// Generate a summary of the current executive context.
    pub async fn get_executive_summary(&self, namespace: String) -> MvResult<String> {
        let engine = self.engine();
        let filters = QueryFilters {
            namespace: Some(namespace),
            ..Default::default()
        };
        let recent = engine.list_nodes(&filters, 5, 0).await?;

        if recent.is_empty() {
            return Ok("The AI assistant is monitoring your active context. Related information will appear here.".to_string());
        }

        let mut summary = String::from("Recent activity focus:\n");
        for node in recent {
            summary.push_str(&format!(
                "- {}\n",
                node.title.as_deref().unwrap_or(&node.content)
            ));
        }

        Ok(summary)
    }

    /// Generate new proactive insights based on vault patterns.
    pub async fn generate_insights(&self, namespace: String) -> MvResult<Vec<ProactiveInsight>> {
        let engine = self.engine();
        let mut detected = Vec::new();

        // Pattern 1: Recent activity clustering
        if let Some(insight) = self.detect_activity_cluster(&engine, &namespace).await? {
            detected.push(insight);
        }

        // Pattern 2: Stale nodes that might need attention
        if let Some(insight) = self.detect_stale_nodes(&engine, &namespace).await? {
            detected.push(insight);
        }

        // Pattern 3: Trending topics (tags/keywords appearing frequently)
        if let Some(insight) = self.detect_trending_topics(&engine, &namespace).await? {
            detected.push(insight);
        }

        // Pattern 4: Connection opportunities (nodes with similar content but no links)
        if let Some(insight) = self
            .detect_connection_opportunities(&engine, &namespace)
            .await?
        {
            detected.push(insight);
        }

        let existing = engine.store.nodes.list_insights(200, 0).await?;
        let mut seen_signatures: HashSet<String> = existing
            .iter()
            .filter(|insight| insight.dismissed_at.is_none())
            .map(Self::insight_signature)
            .collect();

        let mut insights = Vec::new();
        for insight in detected {
            let signature = Self::insight_signature(&insight);
            if !seen_signatures.insert(signature) {
                continue;
            }
            engine.store.nodes.log_insight(&insight).await?;
            insights.push(insight);
        }

        Ok(insights)
    }

    fn insight_signature(insight: &ProactiveInsight) -> String {
        let mut related_node_ids: Vec<String> = insight
            .related_node_ids
            .iter()
            .map(|id| id.to_string())
            .collect();
        related_node_ids.sort_unstable();
        format!(
            "{}|{}|{}|{}",
            insight.insight_type,
            insight.title,
            insight.content,
            related_node_ids.join(",")
        )
    }

    /// Detect clusters of recent activity
    async fn detect_activity_cluster(
        &self,
        engine: &HelixEngine,
        namespace: &str,
    ) -> MvResult<Option<ProactiveInsight>> {
        let filters = QueryFilters {
            namespace: Some(namespace.to_string()),
            ..Default::default()
        };
        let recent = engine.list_nodes(&filters, 10, 0).await?;

        if recent.len() < 3 {
            return Ok(None);
        }

        // Count tag frequencies
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        for node in &recent {
            for tag in &node.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        // Find tags appearing in multiple nodes
        let common_tags: Vec<_> = tag_counts
            .iter()
            .filter(|(_, &count)| count >= 2)
            .map(|(tag, _)| tag.clone())
            .collect();

        if common_tags.is_empty() {
            return Ok(None);
        }

        let insight = ProactiveInsight::new(
            "Emerging Focus Area",
            format!(
                "You've been working on {} related entries. Common themes: {}",
                recent.len(),
                common_tags.join(", ")
            ),
            InsightType::Cluster,
        )
        .with_related_nodes(recent.iter().map(|n| n.id).collect())
        .with_importance(0.6);

        Ok(Some(insight))
    }

    /// Detect nodes that haven't been accessed in a while
    async fn detect_stale_nodes(
        &self,
        engine: &HelixEngine,
        namespace: &str,
    ) -> MvResult<Option<ProactiveInsight>> {
        let filters = QueryFilters {
            namespace: Some(namespace.to_string()),
            ..Default::default()
        };
        let nodes = engine.list_nodes(&filters, 50, 0).await?;

        let stale_threshold = Utc::now() - Duration::days(30);
        let stale_nodes: Vec<_> = nodes
            .into_iter()
            .filter(|n| {
                n.temporal.last_accessed_at < stale_threshold && n.importance >= 0.5
                // Only care about important nodes
            })
            .take(5)
            .collect();

        if stale_nodes.is_empty() {
            return Ok(None);
        }

        let titles: Vec<_> = stale_nodes
            .iter()
            .filter_map(|n| n.title.clone())
            .take(3)
            .collect();

        let insight = ProactiveInsight::new(
            "Notes Need Attention",
            format!(
                "{} important notes haven't been accessed in over a month{}",
                stale_nodes.len(),
                if !titles.is_empty() {
                    format!(", including: {}", titles.join(", "))
                } else {
                    String::new()
                }
            ),
            InsightType::Stale,
        )
        .with_related_nodes(stale_nodes.iter().map(|n| n.id).collect())
        .with_importance(0.5);

        Ok(Some(insight))
    }

    /// Detect trending topics based on recent tag usage
    async fn detect_trending_topics(
        &self,
        engine: &HelixEngine,
        namespace: &str,
    ) -> MvResult<Option<ProactiveInsight>> {
        let week_ago = Utc::now() - Duration::days(7);
        let filters = QueryFilters {
            namespace: Some(namespace.to_string()),
            created_after: Some(week_ago),
            ..Default::default()
        };
        let recent = engine.list_nodes(&filters, 20, 0).await?;

        if recent.len() < 5 {
            return Ok(None);
        }

        // Count tags
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        for node in &recent {
            for tag in &node.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        // Find top trending tags (at least 3 occurrences)
        let mut trending: Vec<_> = tag_counts
            .into_iter()
            .filter(|(_, count)| *count >= 3)
            .collect();
        trending.sort_by(|a, b| b.1.cmp(&a.1));

        if trending.is_empty() {
            return Ok(None);
        }

        let top_tags: Vec<_> = trending.iter().take(3).map(|(t, _)| t.clone()).collect();

        let insight = ProactiveInsight::new(
            "Trending Topics This Week",
            format!(
                "Your most active topics: {}. Consider creating a summary or project note.",
                top_tags.join(", ")
            ),
            InsightType::Trend,
        )
        .with_importance(0.55);

        Ok(Some(insight))
    }

    /// Detect potential connections between unlinked similar nodes
    async fn detect_connection_opportunities(
        &self,
        engine: &HelixEngine,
        namespace: &str,
    ) -> MvResult<Option<ProactiveInsight>> {
        let filters = QueryFilters {
            namespace: Some(namespace.to_string()),
            ..Default::default()
        };
        let recent = engine.list_nodes(&filters, 10, 0).await?;

        if recent.len() < 2 {
            return Ok(None);
        }

        // Pick the most recent node and find similar ones
        let basis = &recent[0];
        let query_text = basis.title.as_deref().unwrap_or(&basis.content);
        let recall_query = MemoryQuery::new(query_text.to_string())
            .with_namespace(namespace.to_string())
            .with_limit(5)
            .with_min_score(0.7);

        let similar = engine.recall(&recall_query).await?;

        // Filter out nodes that are already connected
        let neighbors = engine.get_neighbors(basis.id, 1).await?;
        let neighbor_set: std::collections::HashSet<_> = neighbors.into_iter().collect();

        let unconnected: Vec<_> = similar
            .iter()
            .filter(|r| r.node.id != basis.id && !neighbor_set.contains(&r.node.id))
            .collect();

        if unconnected.is_empty() {
            return Ok(None);
        }

        let titles: Vec<_> = unconnected
            .iter()
            .filter_map(|r| r.node.title.clone())
            .take(2)
            .collect();

        let mut related = vec![basis.id];
        related.extend(unconnected.iter().map(|r| r.node.id));

        let insight = ProactiveInsight::new(
            "Potential Connections Found",
            format!(
                "\"{}\" might be related to {}. Consider linking them.",
                basis.title.as_deref().unwrap_or("Recent note"),
                if !titles.is_empty() {
                    titles.join(" and ")
                } else {
                    "similar notes".to_string()
                }
            ),
            InsightType::Connection,
        )
        .with_related_nodes(related)
        .with_importance(0.65);

        Ok(Some(insight))
    }

    // -----------------------------------------------------------------------
    // Semantic Insight Engine (Phase 3.3)
    // -----------------------------------------------------------------------

    /// Use LLM to generate deeper analysis of a topic area.
    /// Returns a structured insight with LLM-generated content.
    pub async fn analyze_topic_with_llm(
        &self,
        topic: &str,
        namespace: Option<&str>,
    ) -> MvResult<Option<ProactiveInsight>> {
        let engine = self.engine();

        // Search for related nodes
        let mut query = MemoryQuery::new(topic.to_string()).with_limit(20);
        if let Some(ns) = namespace {
            query = query.with_namespace(ns.to_string());
        }
        let results = engine.recall(&query).await?;
        if results.is_empty() {
            return Ok(None);
        }

        // Build context from top results
        let context: String = results
            .iter()
            .take(10)
            .map(|r| {
                format!(
                    "- [{}] {}",
                    r.node.kind,
                    r.node.content.chars().take(200).collect::<String>()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Try LLM analysis if available
        if let Some(ref llm) = engine.llm {
            let messages = vec![
                ChatMessage::system(
                    "You are an analytical assistant for a personal knowledge vault. \
                     Provide structured analysis that is concise and actionable.",
                ),
                ChatMessage::user(format!(
                    "Analyze these knowledge entries about '{topic}' and provide:\n\
                     1. Key themes and patterns\n\
                     2. Potential gaps or missing information\n\
                     3. Suggested connections between ideas\n\
                     Keep your analysis concise (3-5 bullet points per section).\n\n\
                     Entries:\n{context}"
                )),
            ];

            match llm.complete(&messages, &CompletionParams::default()).await {
                Ok(analysis) => {
                    let insight = ProactiveInsight::new(
                        format!("Deep analysis: {topic}"),
                        analysis,
                        InsightType::Cluster,
                    )
                    .with_importance(0.7);
                    return Ok(Some(insight));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "LLM analysis failed, falling back to heuristic");
                }
            }
        }

        // Fallback: heuristic summary
        let result_nodes: Vec<&KnowledgeNode> = results.iter().map(|r| &r.node).collect();
        let summary = format!(
            "Found {} entries related to '{}'. Top tags: {}",
            results.len(),
            topic,
            self.extract_top_tags(&result_nodes, 5).join(", ")
        );
        let insight = ProactiveInsight::new(
            format!("Topic overview: {topic}"),
            summary,
            InsightType::Cluster,
        )
        .with_importance(0.4);
        Ok(Some(insight))
    }

    /// Detect temporal patterns: recurring themes by day of week.
    pub async fn detect_temporal_patterns(
        &self,
        namespace: Option<&str>,
        days_back: u32,
    ) -> MvResult<Vec<ProactiveInsight>> {
        let engine = self.engine();
        let cutoff = Utc::now() - Duration::days(days_back as i64);

        let filters = QueryFilters {
            namespace: namespace.map(|s| s.to_string()),
            ..Default::default()
        };
        let nodes = engine.list_nodes(&filters, 500, 0).await?;

        // Group tags by day of week
        let mut day_tags: HashMap<u32, HashMap<String, usize>> = HashMap::new();
        for node in &nodes {
            if node.temporal.created_at < cutoff {
                continue;
            }
            let weekday = node.temporal.created_at.weekday().num_days_from_monday();
            let tag_counts = day_tags.entry(weekday).or_default();
            for tag in &node.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }

        let day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ];
        let mut insights = Vec::new();

        for (day, tags) in &day_tags {
            let mut sorted: Vec<_> = tags.iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(a.1));
            let top: Vec<_> = sorted
                .iter()
                .take(3)
                .map(|(t, c)| format!("{t} ({c}x)"))
                .collect();

            if !top.is_empty() && sorted[0].1 >= &3 {
                let day_name = day_names.get(*day as usize).unwrap_or(&"Unknown");
                let insight = ProactiveInsight::new(
                    format!("{day_name} pattern detected"),
                    format!(
                        "You tend to work on these topics on {day_name}s: {}",
                        top.join(", ")
                    ),
                    InsightType::TemporalPattern,
                )
                .with_importance(0.5);
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    /// Gap analysis: find topics with questions but no answers.
    pub async fn find_knowledge_gaps(
        &self,
        namespace: Option<&str>,
    ) -> MvResult<Vec<ProactiveInsight>> {
        let engine = self.engine();

        let filters = QueryFilters {
            namespace: namespace.map(|s| s.to_string()),
            ..Default::default()
        };
        let nodes = engine.list_nodes(&filters, 1000, 0).await?;

        // Find nodes containing questions (heuristic: content contains "?")
        let question_nodes: Vec<_> = nodes.iter().filter(|n| n.content.contains('?')).collect();

        let mut insights = Vec::new();
        for q_node in question_nodes.iter().take(10) {
            // Check if any other node seems to answer this question
            let q_text: String = q_node.content.chars().take(100).collect();
            let mut query = MemoryQuery::new(q_text.clone()).with_limit(5);
            if let Some(ns) = namespace {
                query = query.with_namespace(ns.to_string());
            }
            let related = engine.recall(&query).await?;

            // If the only close match is the question itself, it's a gap
            let answers: Vec<_> = related
                .iter()
                .filter(|r| r.node.id != q_node.id && !r.node.content.contains('?'))
                .collect();

            if answers.is_empty() {
                let insight = ProactiveInsight::new(
                    "Unanswered question",
                    format!("No answer found for: {}", q_text),
                    InsightType::KnowledgeGap,
                )
                .with_related_nodes(vec![q_node.id])
                .with_importance(0.6);
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    /// Generate a conceptual map: nodes grouped by topic clusters.
    pub async fn generate_concept_map(
        &self,
        namespace: Option<&str>,
        max_clusters: usize,
    ) -> MvResult<serde_json::Value> {
        let engine = self.engine();

        let filters = QueryFilters {
            namespace: namespace.map(|s| s.to_string()),
            ..Default::default()
        };
        let nodes = engine.list_nodes(&filters, 200, 0).await?;

        // Build tag-based clusters
        let mut tag_clusters: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
        for node in &nodes {
            let primary_tag = node
                .tags
                .first()
                .cloned()
                .unwrap_or_else(|| "untagged".to_string());
            let entry = serde_json::json!({
                "id": node.id.to_string(),
                "kind": node.kind.to_string(),
                "title": node.title,
                "tags": node.tags,
            });
            tag_clusters.entry(primary_tag).or_default().push(entry);
        }

        // Sort clusters by size and take top N
        let mut sorted_clusters: Vec<_> = tag_clusters.into_iter().collect();
        sorted_clusters.sort_by(|a, b| b.1.len().cmp(&a.1.len()));
        sorted_clusters.truncate(max_clusters);

        let clusters: Vec<serde_json::Value> = sorted_clusters
            .into_iter()
            .map(|(tag, nodes)| {
                serde_json::json!({
                    "topic": tag,
                    "count": nodes.len(),
                    "nodes": nodes,
                })
            })
            .collect();

        Ok(serde_json::json!({
            "clusters": clusters,
            "total_nodes": nodes.len(),
            "generated_at": Utc::now().to_rfc3339(),
        }))
    }

    /// Cross-namespace concept mapping: find shared themes across namespaces.
    pub async fn cross_namespace_concepts(
        &self,
        namespaces: &[String],
        min_overlap: usize,
    ) -> MvResult<Vec<ProactiveInsight>> {
        let engine = self.engine();
        let mut ns_tags: HashMap<String, HashMap<String, usize>> = HashMap::new();

        for ns in namespaces {
            let filters = QueryFilters {
                namespace: Some(ns.clone()),
                ..Default::default()
            };
            let nodes = engine.list_nodes(&filters, 500, 0).await?;
            let tag_counts = ns_tags.entry(ns.clone()).or_default();
            for node in &nodes {
                for tag in &node.tags {
                    *tag_counts.entry(tag.clone()).or_insert(0) += 1;
                }
            }
        }

        // Find tags that appear in multiple namespaces
        let mut shared: HashMap<String, Vec<String>> = HashMap::new();
        for (ns, tags) in &ns_tags {
            for tag in tags.keys() {
                shared.entry(tag.clone()).or_default().push(ns.clone());
            }
        }

        let mut insights = Vec::new();
        for (tag, present_in) in shared {
            if present_in.len() >= min_overlap {
                let insight = ProactiveInsight::new(
                    format!("Cross-namespace theme: {tag}"),
                    format!(
                        "The topic '{}' appears across namespaces: {}",
                        tag,
                        present_in.join(", ")
                    ),
                    InsightType::Connection,
                )
                .with_importance(0.6);
                insights.push(insight);
            }
        }

        Ok(insights)
    }

    // -----------------------------------------------------------------------
    // Cross-Domain Connection Discovery (Phase 1A)
    // -----------------------------------------------------------------------

    /// Find nodes in different namespaces that are semantically similar but unlinked.
    pub async fn detect_cross_domain_connections(
        &self,
        namespaces: &[String],
        similarity_threshold: f64,
        limit: usize,
    ) -> MvResult<Vec<ProactiveInsight>> {
        let engine = self.engine();
        let mut insights = Vec::new();

        if namespaces.len() < 2 {
            return Ok(insights);
        }

        // For each namespace, get recent nodes
        let mut ns_nodes: HashMap<String, Vec<KnowledgeNode>> = HashMap::new();
        for ns in namespaces {
            let filters = QueryFilters {
                namespace: Some(ns.clone()),
                ..Default::default()
            };
            let nodes = engine.list_nodes(&filters, 20, 0).await?;
            ns_nodes.insert(ns.clone(), nodes);
        }

        // Compare nodes across namespaces using semantic search
        for (ns_a, nodes_a) in &ns_nodes {
            for node_a in nodes_a.iter().take(5) {
                let query_text = node_a.title.as_deref().unwrap_or(&node_a.content);
                // Search in other namespaces
                for ns_b in namespaces {
                    if ns_b == ns_a {
                        continue;
                    }
                    let query = MemoryQuery::new(query_text.to_string())
                        .with_namespace(ns_b.to_string())
                        .with_limit(3)
                        .with_min_score(similarity_threshold);

                    let results = engine.recall(&query).await?;

                    for result in &results {
                        if result.node.id == node_a.id {
                            continue;
                        }
                        // Check they're not already linked
                        let neighbors = engine.get_neighbors(node_a.id, 1).await?;
                        if neighbors.contains(&result.node.id) {
                            continue;
                        }

                        let insight = ProactiveInsight::new(
                            format!(
                                "Cross-domain connection: {} ↔ {}",
                                ns_a, ns_b,
                            ),
                            format!(
                                "'{}' ({}) is semantically similar to '{}' ({}) with score {:.2}",
                                node_a.title.as_deref().unwrap_or("untitled"),
                                ns_a,
                                result.node.title.as_deref().unwrap_or("untitled"),
                                ns_b,
                                result.score,
                            ),
                            InsightType::CrossDomain,
                        )
                        .with_related_nodes(vec![node_a.id, result.node.id])
                        .with_importance(0.7);

                        insights.push(insight);

                        if insights.len() >= limit {
                            return Ok(insights);
                        }
                    }
                }
            }
        }

        Ok(insights)
    }

    // -----------------------------------------------------------------------
    // Ambient Synthesis (Phase 1C)
    // -----------------------------------------------------------------------

    /// Discover and suggest relationships for recently-updated nodes.
    /// Intended to run periodically as a background task.
    pub async fn ambient_synthesis(
        &self,
        namespace: Option<&str>,
        batch_size: usize,
        similarity_threshold: f64,
    ) -> MvResult<Vec<ProactiveInsight>> {
        let engine = self.engine();
        let filters = QueryFilters {
            namespace: namespace.map(|s| s.to_string()),
            ..Default::default()
        };
        let recent = engine.list_nodes(&filters, batch_size, 0).await?;

        let mut insights = Vec::new();
        for node in &recent {
            let query_text = node.title.as_deref().unwrap_or(&node.content);
            let mut query = MemoryQuery::new(query_text.to_string())
                .with_limit(4)
                .with_min_score(similarity_threshold);
            if let Some(ns) = namespace {
                query = query.with_namespace(ns.to_string());
            }

            let similar = engine.recall(&query).await?;

            // Check for nodes that aren't already linked and share at least one tag
            let neighbors = engine.get_neighbors(node.id, 1).await?;
            let neighbor_set: HashSet<Uuid> = neighbors.into_iter().collect();
            let node_tags: HashSet<&String> = node.tags.iter().collect();

            for result in &similar {
                if result.node.id == node.id || neighbor_set.contains(&result.node.id) {
                    continue;
                }

                // Must share at least one tag
                let shared_tags: Vec<_> = result
                    .node
                    .tags
                    .iter()
                    .filter(|t| node_tags.contains(t))
                    .collect();

                if shared_tags.is_empty() {
                    continue;
                }

                let insight = ProactiveInsight::new(
                    format!(
                        "Link suggestion: {} → {}",
                        node.title.as_deref().unwrap_or("untitled"),
                        result.node.title.as_deref().unwrap_or("untitled"),
                    ),
                    format!(
                        "These nodes share tags ({}) and are semantically similar ({:.2}). Consider linking them.",
                        shared_tags.iter().map(|t| t.as_str()).collect::<Vec<_>>().join(", "),
                        result.score,
                    ),
                    InsightType::Connection,
                )
                .with_related_nodes(vec![node.id, result.node.id])
                .with_importance(0.55);

                insights.push(insight);
            }
        }

        // Deduplicate and persist
        let existing = engine.store.nodes.list_insights(200, 0).await?;
        let mut seen_signatures: HashSet<String> = existing
            .iter()
            .filter(|i| i.dismissed_at.is_none())
            .map(Self::insight_signature)
            .collect();

        let mut persisted = Vec::new();
        for insight in insights {
            let sig = Self::insight_signature(&insight);
            if !seen_signatures.insert(sig) {
                continue;
            }
            engine.store.nodes.log_insight(&insight).await?;
            persisted.push(insight);
        }

        Ok(persisted)
    }

    /// Convenience wrapper: run ambient synthesis for the "default" namespace
    /// with standard defaults (batch_size=10, similarity_threshold=0.78).
    pub async fn ambient_synthesis_batch(&self) -> MvResult<Vec<ProactiveInsight>> {
        self.ambient_synthesis(Some("default"), 10, 0.78).await
    }

    /// Helper: extract top tags from a slice of nodes.
    fn extract_top_tags(&self, nodes: &[&KnowledgeNode], limit: usize) -> Vec<String> {
        let mut tag_counts: HashMap<String, usize> = HashMap::new();
        for node in nodes {
            for tag in &node.tags {
                *tag_counts.entry(tag.clone()).or_insert(0) += 1;
            }
        }
        let mut sorted: Vec<_> = tag_counts.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(limit).map(|(t, _)| t).collect()
    }
}

impl Default for ProactiveEngine {
    fn default() -> Self {
        Self::new()
    }
}
