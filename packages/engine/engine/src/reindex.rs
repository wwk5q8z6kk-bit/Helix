//! Safe atomic reindex of search indexes.
//!
//! Iterates all stored nodes, re-inserts them into the Tantivy full-text index,
//! and re-embeds content for vector search where embeddings are missing.

use serde::{Deserialize, Serialize};

use hx_core::MvResult;

/// Report summarizing a reindex operation.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct ReindexReport {
    /// Number of nodes scanned from the database.
    pub nodes_scanned: usize,
    /// Number of nodes indexed into Tantivy.
    pub tantivy_indexed: usize,
    /// Number of nodes for which new vector embeddings were generated.
    pub vectors_embedded: usize,
    /// Number of nodes for which existing embeddings were reused (skipped).
    pub vectors_skipped: usize,
    /// Any non-fatal errors encountered during reindex.
    pub errors: Vec<String>,
}

/// Node data sufficient for reindexing.
///
/// This is a minimal representation — the actual engine will provide full node data.
/// This struct exists so the reindex logic can be tested independently.
#[derive(Debug, Clone)]
pub struct ReindexNode {
    pub id: String,
    pub title: String,
    pub body: String,
    pub has_embedding: bool,
}

/// Perform a reindex pass over the provided nodes.
///
/// For each node:
/// 1. Index into Tantivy (via the `index_fn` callback)
/// 2. If the node lacks an embedding, generate one (via the `embed_fn` callback)
///
/// Returns a report summarizing the operation.
pub async fn safe_reindex<I, F, E>(
    nodes: I,
    mut index_fn: F,
    mut embed_fn: E,
) -> MvResult<ReindexReport>
where
    I: IntoIterator<Item = ReindexNode>,
    F: FnMut(&ReindexNode) -> Result<(), String>,
    E: FnMut(&ReindexNode) -> Result<(), String>,
{
    let mut report = ReindexReport::default();

    for node in nodes {
        report.nodes_scanned += 1;

        // Re-index in Tantivy
        match index_fn(&node) {
            Ok(()) => report.tantivy_indexed += 1,
            Err(e) => report.errors.push(format!("index {}: {e}", node.id)),
        }

        // Re-embed if missing
        if node.has_embedding {
            report.vectors_skipped += 1;
        } else {
            match embed_fn(&node) {
                Ok(()) => report.vectors_embedded += 1,
                Err(e) => report.errors.push(format!("embed {}: {e}", node.id)),
            }
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_nodes(count: usize, has_embedding: bool) -> Vec<ReindexNode> {
        (0..count)
            .map(|i| ReindexNode {
                id: format!("node-{i}"),
                title: format!("Title {i}"),
                body: format!("Body content for node {i}"),
                has_embedding,
            })
            .collect()
    }

    #[tokio::test]
    async fn report_counts_correct() {
        let nodes = vec![
            ReindexNode {
                id: "a".into(),
                title: "A".into(),
                body: "text a".into(),
                has_embedding: true,
            },
            ReindexNode {
                id: "b".into(),
                title: "B".into(),
                body: "text b".into(),
                has_embedding: false,
            },
            ReindexNode {
                id: "c".into(),
                title: "C".into(),
                body: "text c".into(),
                has_embedding: false,
            },
        ];

        let report = safe_reindex(nodes, |_| Ok(()), |_| Ok(())).await.unwrap();
        assert_eq!(report.nodes_scanned, 3);
        assert_eq!(report.tantivy_indexed, 3);
        assert_eq!(report.vectors_embedded, 2);
        assert_eq!(report.vectors_skipped, 1);
        assert!(report.errors.is_empty());
    }

    #[tokio::test]
    async fn error_handling() {
        let nodes = make_nodes(3, false);

        let report = safe_reindex(
            nodes,
            |node| {
                if node.id == "node-1" {
                    Err("index failure".into())
                } else {
                    Ok(())
                }
            },
            |node| {
                if node.id == "node-2" {
                    Err("embed failure".into())
                } else {
                    Ok(())
                }
            },
        )
        .await
        .unwrap();

        assert_eq!(report.nodes_scanned, 3);
        assert_eq!(report.tantivy_indexed, 2); // node-1 failed
        assert_eq!(report.vectors_embedded, 2); // node-2 failed
        assert_eq!(report.errors.len(), 2);
        assert!(report.errors[0].contains("index node-1"));
        assert!(report.errors[1].contains("embed node-2"));
    }

    #[tokio::test]
    async fn empty_input() {
        let nodes: Vec<ReindexNode> = Vec::new();
        let report = safe_reindex(nodes, |_| Ok(()), |_| Ok(())).await.unwrap();
        assert_eq!(report.nodes_scanned, 0);
        assert_eq!(report.tantivy_indexed, 0);
        assert_eq!(report.vectors_embedded, 0);
        assert_eq!(report.vectors_skipped, 0);
        assert!(report.errors.is_empty());
    }

    #[test]
    fn report_serialization() {
        let report = ReindexReport {
            nodes_scanned: 100,
            tantivy_indexed: 98,
            vectors_embedded: 50,
            vectors_skipped: 48,
            errors: vec!["oops".into()],
        };
        let json = serde_json::to_value(&report).unwrap();
        assert_eq!(json["nodes_scanned"], 100);
        assert_eq!(json["errors"].as_array().unwrap().len(), 1);
    }
}
