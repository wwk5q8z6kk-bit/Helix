use std::sync::Arc;

use hx_core::model::RewriteStrategy;
use hx_core::*;
use hx_graph::store::SqliteGraphStore;
use hx_index::hybrid::{apply_graph_boost, reciprocal_rank_fusion};
use hx_index::tantivy_index::TantivyFullTextIndex;
use hx_storage::unified::UnifiedStore;
use tracing::debug;

use crate::config::EngineConfig;
use crate::llm::LlmProvider;
use crate::multihop::MultiHopRetriever;
use crate::query_rewrite::QueryRewriter;
use crate::rerank::apply_reranking;
use crate::session::InMemorySessionStore;

/// Recall pipeline: searches across FTS + vector + graph and fuses results.
/// Phase 3 additions: query rewriting before search, reranking after fusion,
/// session context injection, and multi-hop iterative retrieval.
pub struct RecallPipeline {
    store: Arc<UnifiedStore>,
    fts: Arc<TantivyFullTextIndex>,
    graph: Arc<SqliteGraphStore>,
    config: EngineConfig,
    rewriter: QueryRewriter,
    reranker: Arc<dyn hx_core::traits::Reranker>,
    session_store: Arc<InMemorySessionStore>,
    multihop: MultiHopRetriever,
}

impl RecallPipeline {
    pub fn new(
        store: Arc<UnifiedStore>,
        fts: Arc<TantivyFullTextIndex>,
        graph: Arc<SqliteGraphStore>,
        config: EngineConfig,
        llm: Option<Arc<dyn LlmProvider>>,
    ) -> Self {
        let rewriter = QueryRewriter::new(llm.clone(), config.query_rewrite.clone());
        let reranker = crate::rerank::init_reranker(&config.rerank, llm.clone());
        let session_store = Arc::new(InMemorySessionStore::new(config.session.clone()));
        let multihop = MultiHopRetriever::new(llm, config.multihop.clone());
        Self {
            store,
            fts,
            graph,
            config,
            rewriter,
            reranker,
            session_store,
            multihop,
        }
    }

    /// Get a reference to the session store for external use (e.g., recording turns).
    pub fn session_store(&self) -> &Arc<InMemorySessionStore> {
        &self.session_store
    }

    /// Execute a memory query and return ranked results.
    pub async fn recall(&self, query: &MemoryQuery) -> MvResult<Vec<SearchResult>> {
        let limit = if query.limit > 0 {
            query.limit
        } else {
            self.config.search.default_limit
        };

        // Fetch extra results for post-filtering and reranking
        let fetch_limit = limit * 3;

        // --- Phase 3: Session Context ---
        // If a session_id is provided, build context from prior turns and
        // prepend to the query for better rewriting.
        let effective_query = if let Some(ref sid) = query.session_id {
            if let Some(ctx) = self.session_store.build_context_string(sid, 3).await {
                format!("{ctx}\nCurrent query: {}", query.text)
            } else {
                query.text.clone()
            }
        } else {
            query.text.clone()
        };

        // --- Phase 3: Query Rewriting ---
        let rewrite_result = self
            .rewriter
            .rewrite(&effective_query, query.rewrite_strategy)
            .await;

        if rewrite_result.applied_strategy != RewriteStrategy::None {
            debug!(
                strategy = %rewrite_result.applied_strategy,
                queries = ?rewrite_result.queries,
                has_hyde = rewrite_result.hyde_document.is_some(),
                "query rewritten"
            );
        }

        // For decomposed queries, search each sub-query and merge
        if rewrite_result.queries.len() > 1 {
            return self
                .recall_decomposed(query, &rewrite_result.queries, fetch_limit, limit)
                .await;
        }

        // Use the rewritten query text (or original if no rewrite)
        let search_text = &rewrite_result.queries[0];
        // For HyDE, use the hypothetical document for vector embedding
        let hyde_text = rewrite_result.hyde_document.as_deref();
        let embed_text = hyde_text.unwrap_or(search_text);

        let mut result_lists: Vec<Vec<(uuid::Uuid, f64)>> = Vec::new();

        match query.strategy {
            SearchStrategy::FullText => {
                let fts_results = self.fts.search(search_text, fetch_limit)?;
                result_lists.push(fts_results);
            }
            SearchStrategy::Vector => {
                if let Some(ref vectors) = self.store.vectors {
                    let embedding = self.store.embedder.embed(embed_text).await?;
                    let vec_results = vectors
                        .search(
                            embedding,
                            fetch_limit,
                            query.min_score,
                            query.filters.namespace.as_deref(),
                        )
                        .await?;
                    result_lists.push(vec_results);
                }
            }
            SearchStrategy::Hybrid => {
                // Full-text search uses rewritten query
                let fts_results = self.fts.search(search_text, fetch_limit)?;
                result_lists.push(fts_results);

                // Vector search uses HyDE doc or rewritten query
                if let Some(ref vectors) = self.store.vectors {
                    match self.store.embedder.embed(embed_text).await {
                        Ok(embedding) => {
                            let vec_results = vectors
                                .search(
                                    embedding,
                                    fetch_limit,
                                    query.min_score,
                                    query.filters.namespace.as_deref(),
                                )
                                .await?;
                            result_lists.push(vec_results);
                        }
                        Err(e) => {
                            tracing::warn!("vector search skipped (embedding failed): {e}");
                        }
                    }
                }
            }
            SearchStrategy::Graph => {
                // First do a text search to find seed nodes, then expand via graph
                let fts_results = self.fts.search(search_text, 5)?;
                let mut all_neighbors = Vec::new();

                for (node_id, _score) in &fts_results {
                    let neighbors = self
                        .graph
                        .get_neighbors(*node_id, self.config.graph.default_traversal_depth)
                        .await?;
                    all_neighbors.extend(neighbors);
                }

                // Combine seed results with graph-expanded results
                let mut combined: Vec<(uuid::Uuid, f64)> = fts_results;
                for neighbor_id in all_neighbors {
                    if !combined.iter().any(|(id, _)| *id == neighbor_id) {
                        combined.push((neighbor_id, 0.3)); // base score for graph neighbors
                    }
                }
                result_lists.push(combined);
            }
        }

        // Fuse results using RRF
        let mut fused = if result_lists.len() > 1 {
            reciprocal_rank_fusion(&result_lists, self.config.search.rrf_k, fetch_limit)
        } else {
            result_lists.into_iter().next().unwrap_or_default()
        };

        // Apply graph boost for hybrid/fulltext/vector strategies
        if query.strategy != SearchStrategy::Graph {
            let mut all_neighbors = Vec::new();
            for (node_id, _) in fused.iter().take(5) {
                if let Ok(neighbors) = self
                    .graph
                    .get_neighbors(*node_id, self.config.graph.default_traversal_depth)
                    .await
                {
                    all_neighbors.extend(neighbors);
                }
            }
            if !all_neighbors.is_empty() {
                apply_graph_boost(
                    &mut fused,
                    &all_neighbors,
                    self.config.graph.graph_boost_factor,
                );
            }
        }

        // --- Phase 3: Cross-Encoder Reranking ---
        if self.config.rerank.enabled && !fused.is_empty() {
            fused = self
                .apply_rerank(search_text, &fused, fetch_limit)
                .await;
        }

        // --- Phase 3: Multi-Hop Retrieval ---
        if self.multihop.is_enabled() && !fused.is_empty() {
            fused = self
                .apply_multihop(query, search_text, fused, fetch_limit)
                .await?;
        }

        // Filter by min_score
        fused.retain(|(_, score)| *score >= query.min_score);

        // Truncate to requested limit
        fused.truncate(limit);

        // Hydrate with full node data
        let results = self
            .hydrate_results(&fused, &query.filters, query.strategy)
            .await?;

        // Record this turn in session memory for future context
        if let Some(ref sid) = query.session_id {
            let summary = results
                .iter()
                .take(3)
                .map(|r| r.node.title.as_deref().unwrap_or("untitled"))
                .collect::<Vec<_>>()
                .join(", ");
            self.session_store
                .add_turn(sid, &query.text, &format!("Found: {summary}"))
                .await;
        }

        Ok(results)
    }

    /// Execute recall for decomposed queries: search each sub-query, merge
    /// and re-rank results using RRF.
    async fn recall_decomposed(
        &self,
        original_query: &MemoryQuery,
        sub_queries: &[String],
        fetch_limit: usize,
        limit: usize,
    ) -> MvResult<Vec<SearchResult>> {
        let mut all_result_lists: Vec<Vec<(uuid::Uuid, f64)>> = Vec::new();

        for sub_q in sub_queries {
            let fts_results = self.fts.search(sub_q, fetch_limit)?;
            let mut sub_results = fts_results;

            if let Some(ref vectors) = self.store.vectors {
                if let Ok(embedding) = self.store.embedder.embed(sub_q).await {
                    let vec_results = vectors
                        .search(
                            embedding,
                            fetch_limit,
                            original_query.min_score,
                            original_query.filters.namespace.as_deref(),
                        )
                        .await?;
                    sub_results = reciprocal_rank_fusion(
                        &[sub_results, vec_results],
                        self.config.search.rrf_k,
                        fetch_limit,
                    );
                }
            }

            all_result_lists.push(sub_results);
        }

        let mut fused = if all_result_lists.len() > 1 {
            reciprocal_rank_fusion(&all_result_lists, self.config.search.rrf_k, fetch_limit)
        } else {
            all_result_lists.into_iter().next().unwrap_or_default()
        };

        // Apply graph boost
        let mut all_neighbors = Vec::new();
        for (node_id, _) in fused.iter().take(5) {
            if let Ok(neighbors) = self
                .graph
                .get_neighbors(*node_id, self.config.graph.default_traversal_depth)
                .await
            {
                all_neighbors.extend(neighbors);
            }
        }
        if !all_neighbors.is_empty() {
            apply_graph_boost(
                &mut fused,
                &all_neighbors,
                self.config.graph.graph_boost_factor,
            );
        }

        fused.retain(|(_, score)| *score >= original_query.min_score);
        fused.truncate(limit);

        self.hydrate_results(&fused, &original_query.filters, SearchStrategy::Hybrid)
            .await
    }

    /// Rerank fused results using the cross-encoder reranker.
    async fn apply_rerank(
        &self,
        query_text: &str,
        fused: &[(uuid::Uuid, f64)],
        max_results: usize,
    ) -> Vec<(uuid::Uuid, f64)> {
        // Collect document content for reranking
        let mut docs = Vec::new();
        let mut ids_scores: Vec<(uuid::Uuid, f64)> = Vec::new();
        for &(id, score) in fused.iter().take(max_results) {
            if let Ok(Some(node)) = self.store.nodes.get(id).await {
                let text = node.title.as_deref().unwrap_or("");
                let snippet = if node.content.len() > 500 {
                    &node.content[..500]
                } else {
                    &node.content
                };
                docs.push(format!("{text} {snippet}"));
                ids_scores.push((id, score));
            }
        }

        if docs.is_empty() {
            return fused.to_vec();
        }

        // apply_reranking mutates in-place
        apply_reranking(
            &*self.reranker,
            query_text,
            &mut ids_scores,
            &docs,
            &self.config.rerank,
        )
        .await;

        ids_scores
    }

    /// Apply multi-hop retrieval: extract entities from initial results,
    /// generate follow-up queries, retrieve additional results, merge.
    async fn apply_multihop(
        &self,
        original_query: &MemoryQuery,
        search_text: &str,
        mut fused: Vec<(uuid::Uuid, f64)>,
        fetch_limit: usize,
    ) -> MvResult<Vec<(uuid::Uuid, f64)>> {
        use std::collections::HashSet;

        let mut seen_queries: HashSet<String> = HashSet::new();
        seen_queries.insert(search_text.to_string());
        let mut accumulated_tokens = MultiHopRetriever::estimate_tokens(search_text);

        for hop in 0..self.multihop.max_hops() {
            if !self.multihop.within_budget(accumulated_tokens) {
                debug!(hop, accumulated_tokens, "multi-hop: token budget exhausted");
                break;
            }

            // Gather content from top results for entity extraction
            let mut result_contents = Vec::new();
            for &(id, _) in fused.iter().take(5) {
                if let Ok(Some(node)) = self.store.nodes.get(id).await {
                    let text = if node.content.len() > 400 {
                        format!("{}...", &node.content[..400])
                    } else {
                        node.content.clone()
                    };
                    accumulated_tokens += MultiHopRetriever::estimate_tokens(&text);
                    result_contents.push(text);
                }
            }

            let follow_ups = self
                .multihop
                .plan_follow_ups(search_text, &result_contents, &seen_queries)
                .await;

            if follow_ups.is_empty() {
                debug!(hop, "multi-hop: no follow-up queries, stopping");
                break;
            }

            for fq in &follow_ups {
                seen_queries.insert(fq.clone());
                accumulated_tokens += MultiHopRetriever::estimate_tokens(fq);

                // Search with follow-up query
                let fts_results = self.fts.search(fq, fetch_limit)?;
                let mut hop_results = fts_results;

                if let Some(ref vectors) = self.store.vectors {
                    if let Ok(embedding) = self.store.embedder.embed(fq).await {
                        let vec_results = vectors
                            .search(
                                embedding,
                                fetch_limit,
                                original_query.min_score,
                                original_query.filters.namespace.as_deref(),
                            )
                            .await?;
                        hop_results =
                            hx_index::hybrid::reciprocal_rank_fusion(
                                &[hop_results, vec_results],
                                self.config.search.rrf_k,
                                fetch_limit,
                            );
                    }
                }

                // Merge new results into fused, boosting new finds slightly
                for (id, score) in hop_results {
                    if let Some(existing) = fused.iter_mut().find(|(eid, _)| *eid == id) {
                        // Boost score of results found in multiple hops
                        existing.1 += score * 0.3;
                    } else {
                        // Discount new results slightly (they're indirect)
                        fused.push((id, score * 0.7));
                    }
                }
            }

            // Re-sort after merging hop results
            fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            debug!(
                hop = hop + 1,
                follow_ups = ?follow_ups,
                total_results = fused.len(),
                "multi-hop: completed hop"
            );
        }

        Ok(fused)
    }

    /// Hydrate UUID+score pairs into full SearchResults with filter application.
    async fn hydrate_results(
        &self,
        fused: &[(uuid::Uuid, f64)],
        filters: &QueryFilters,
        strategy: SearchStrategy,
    ) -> MvResult<Vec<SearchResult>> {
        let mut results = Vec::new();
        for &(node_id, score) in fused {
            let _ = self.store.nodes.touch(node_id).await;
            if let Ok(Some(node)) = self.store.nodes.get(node_id).await {
                if !matches_filters(&node, filters) {
                    continue;
                }
                results.push(SearchResult {
                    node,
                    score,
                    match_source: match strategy {
                        SearchStrategy::Vector => MatchSource::Vector,
                        SearchStrategy::FullText => MatchSource::FullText,
                        SearchStrategy::Hybrid => MatchSource::Hybrid,
                        SearchStrategy::Graph => MatchSource::Graph,
                    },
                });
            }
        }
        Ok(results)
    }
}

fn matches_filters(node: &KnowledgeNode, filters: &QueryFilters) -> bool {
    if let Some(ref ns) = filters.namespace {
        if node.namespace != *ns {
            return false;
        }
    }
    if let Some(ref kinds) = filters.kinds {
        if !kinds.contains(&node.kind) {
            return false;
        }
    }
    if let Some(ref tags) = filters.tags {
        if !tags.iter().any(|t| node.tags.contains(t)) {
            return false;
        }
    }
    if let Some(min_imp) = filters.min_importance {
        if node.importance < min_imp {
            return false;
        }
    }
    if let Some(ref after) = filters.created_after {
        if node.temporal.created_at < *after {
            return false;
        }
    }
    if let Some(ref before) = filters.created_before {
        if node.temporal.created_at > *before {
            return false;
        }
    }
    true
}
