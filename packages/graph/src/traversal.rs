use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::{DiGraph, NodeIndex};
use uuid::Uuid;

use hx_core::Relationship;

/// In-memory graph for fast traversal operations.
/// Built from a set of relationships, used for complex graph queries.
pub struct MemoryGraph {
    graph: DiGraph<Uuid, f64>,
    node_map: HashMap<Uuid, NodeIndex>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    pub fn from_relationships(relationships: &[Relationship]) -> Self {
        let mut mg = Self::new();
        for rel in relationships {
            mg.add_edge(rel.from_node, rel.to_node, rel.weight);
        }
        mg
    }

    fn ensure_node(&mut self, id: Uuid) -> NodeIndex {
        *self
            .node_map
            .entry(id)
            .or_insert_with(|| self.graph.add_node(id))
    }

    pub fn add_edge(&mut self, from: Uuid, to: Uuid, weight: f64) {
        let from_idx = self.ensure_node(from);
        let to_idx = self.ensure_node(to);
        self.graph.add_edge(from_idx, to_idx, weight);
    }

    /// BFS neighbors up to a given depth.
    pub fn bfs_neighbors(&self, start: Uuid, depth: usize) -> Vec<Uuid> {
        let start_idx = match self.node_map.get(&start) {
            Some(idx) => *idx,
            None => return vec![],
        };

        let mut visited = HashSet::new();
        visited.insert(start_idx);
        let mut queue = VecDeque::new();
        queue.push_back((start_idx, 0usize));
        let mut results = Vec::new();

        while let Some((node, d)) = queue.pop_front() {
            if d >= depth {
                continue;
            }
            // Outgoing neighbors
            for neighbor in self.graph.neighbors(node) {
                if visited.insert(neighbor) {
                    results.push(self.graph[neighbor]);
                    queue.push_back((neighbor, d + 1));
                }
            }
            // Incoming neighbors (undirected traversal)
            for neighbor in self
                .graph
                .neighbors_directed(node, petgraph::Direction::Incoming)
            {
                if visited.insert(neighbor) {
                    results.push(self.graph[neighbor]);
                    queue.push_back((neighbor, d + 1));
                }
            }
        }

        results
    }

    /// Get shortest path length between two nodes (or None if unreachable).
    pub fn shortest_path_length(&self, from: Uuid, to: Uuid) -> Option<usize> {
        let from_idx = self.node_map.get(&from)?;
        let to_idx = self.node_map.get(&to)?;

        let mut visited = HashSet::new();
        visited.insert(*from_idx);
        let mut queue = VecDeque::new();
        queue.push_back((*from_idx, 0usize));

        while let Some((node, dist)) = queue.pop_front() {
            if node == *to_idx {
                return Some(dist);
            }
            for neighbor in self.graph.neighbors(node) {
                if visited.insert(neighbor) {
                    queue.push_back((neighbor, dist + 1));
                }
            }
        }

        None
    }

    /// Export to DOT format for visualization.
    pub fn to_dot(&self) -> String {
        use std::fmt::Write;
        let mut out = String::from("digraph Helix {\n");
        for edge in self.graph.edge_indices() {
            if let Some((from, to)) = self.graph.edge_endpoints(edge) {
                let from_id = &self.graph[from];
                let to_id = &self.graph[to];
                let weight = self.graph[edge];
                writeln!(
                    out,
                    "  \"{}\" -> \"{}\" [label=\"{:.2}\"];",
                    &from_id.to_string()[..8],
                    &to_id.to_string()[..8],
                    weight
                )
                .ok();
            }
        }
        out.push_str("}\n");
        out
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
}

impl Default for MemoryGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hx_core::RelationKind;

    #[test]
    fn test_bfs_neighbors() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        let d = Uuid::now_v7();

        let rels = vec![
            Relationship::new(a, b, RelationKind::RelatesTo),
            Relationship::new(b, c, RelationKind::RelatesTo),
            Relationship::new(c, d, RelationKind::RelatesTo),
        ];

        let graph = MemoryGraph::from_relationships(&rels);

        let n1 = graph.bfs_neighbors(a, 1);
        assert_eq!(n1.len(), 1);
        assert!(n1.contains(&b));

        let n2 = graph.bfs_neighbors(a, 2);
        assert_eq!(n2.len(), 2);
    }

    #[test]
    fn test_shortest_path() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();

        let mut graph = MemoryGraph::new();
        graph.add_edge(a, b, 1.0);
        graph.add_edge(b, c, 1.0);

        assert_eq!(graph.shortest_path_length(a, c), Some(2));
        assert_eq!(graph.shortest_path_length(a, b), Some(1));
        assert_eq!(graph.shortest_path_length(c, a), None); // directed
    }

    #[test]
    fn test_dot_export() {
        let a = Uuid::now_v7();
        let b = Uuid::now_v7();

        let mut graph = MemoryGraph::new();
        graph.add_edge(a, b, 1.0);

        let dot = graph.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("->"));
    }
}
