use std::collections::HashMap;
use uuid::Uuid;

/// Reciprocal Rank Fusion (RRF) to combine multiple ranked result lists.
///
/// Each result list is a Vec of (Uuid, score). The RRF formula is:
///   score(d) = sum over all lists: 1 / (k + rank(d))
///
/// where k is a smoothing constant (typically 60).
pub fn reciprocal_rank_fusion(
    result_lists: &[Vec<(Uuid, f64)>],
    k: f64,
    limit: usize,
) -> Vec<(Uuid, f64)> {
    let mut fused_scores: HashMap<Uuid, f64> = HashMap::new();

    for results in result_lists {
        for (rank, (id, _score)) in results.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank as f64) + 1.0);
            *fused_scores.entry(*id).or_default() += rrf_score;
        }
    }

    let mut sorted: Vec<(Uuid, f64)> = fused_scores.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(limit);
    sorted
}

/// Boost scores for nodes that appear as graph neighbors.
pub fn apply_graph_boost(results: &mut [(Uuid, f64)], neighbor_ids: &[Uuid], boost_factor: f64) {
    for (id, score) in results.iter_mut() {
        if neighbor_ids.contains(id) {
            *score *= 1.0 + boost_factor;
        }
    }
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rrf() {
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        let id3 = Uuid::now_v7();

        let list1 = vec![(id1, 0.9), (id2, 0.7), (id3, 0.5)];
        let list2 = vec![(id2, 0.95), (id1, 0.6), (id3, 0.3)];

        let fused = reciprocal_rank_fusion(&[list1, list2], 60.0, 10);
        assert_eq!(fused.len(), 3);
        // Both id1 and id2 appear in both lists, should be top results
        // id2 is rank 0 in list2 and rank 1 in list1 => higher combined
        // id1 is rank 0 in list1 and rank 1 in list2 => same combined
        assert!(fused[0].1 > 0.0);
    }

    #[test]
    fn test_graph_boost() {
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();

        let mut results = vec![(id1, 0.5), (id2, 0.8)];
        apply_graph_boost(&mut results, &[id1], 0.5);

        // id1 was 0.5, boosted to 0.75
        // id2 stays 0.8
        // After sort: id2 first, id1 second
        assert_eq!(results[0].0, id2);
        assert!((results[1].1 - 0.75).abs() < f64::EPSILON);
    }
}
