use std::sync::Mutex;

use async_trait::async_trait;
use rusqlite::types::Type;
use rusqlite::{params, Connection, OptionalExtension};
use uuid::Uuid;

use hx_core::*;

/// SQLite-backed graph store with petgraph in-memory cache.
pub struct SqliteGraphStore {
    conn: Mutex<Connection>,
}

impl SqliteGraphStore {
    pub fn open(conn: Connection) -> MvResult<Self> {
        // The relationships table is created by the main migration.
        // This store shares the same SQLite database.
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Create an in-memory graph store for testing.
    pub fn open_in_memory() -> MvResult<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| HxError::Graph(format!("open in-memory: {e}")))?;

        conn.execute_batch("PRAGMA foreign_keys=OFF;").ok(); // no FK in standalone test

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS relationships (
                id TEXT PRIMARY KEY NOT NULL,
                from_node TEXT NOT NULL,
                to_node TEXT NOT NULL,
                kind TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_rel_from ON relationships(from_node);
            CREATE INDEX IF NOT EXISTS idx_rel_to ON relationships(to_node);",
        )
        .map_err(|e| HxError::Graph(format!("create table: {e}")))?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn row_to_rel(row: &rusqlite::Row<'_>) -> rusqlite::Result<Relationship> {
        let id_str: String = row.get(0)?;
        let from_str: String = row.get(1)?;
        let to_str: String = row.get(2)?;
        let kind_str: String = row.get(3)?;
        let weight: f64 = row.get(4)?;
        let meta_json: Option<String> = row.get(5)?;
        let created_at_str: String = row.get(6)?;

        Ok(Relationship {
            id: parse_uuid_str(0, &id_str)?,
            from_node: parse_uuid_str(1, &from_str)?,
            to_node: parse_uuid_str(2, &to_str)?,
            kind: kind_str.parse().map_err(|err: String| {
                rusqlite::Error::FromSqlConversionFailure(
                    3,
                    Type::Text,
                    Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, err)),
                )
            })?,
            weight,
            metadata: parse_metadata_json(meta_json)?,
            created_at: chrono::DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .map_err(|err| {
                    rusqlite::Error::FromSqlConversionFailure(6, Type::Text, Box::new(err))
                })?,
        })
    }
}

fn parse_uuid_str(column: usize, s: &str) -> rusqlite::Result<Uuid> {
    Uuid::parse_str(s)
        .map_err(|err| rusqlite::Error::FromSqlConversionFailure(column, Type::Text, Box::new(err)))
}

fn parse_metadata_json(
    meta_json: Option<String>,
) -> rusqlite::Result<std::collections::HashMap<String, serde_json::Value>> {
    match meta_json {
        Some(raw) => serde_json::from_str(&raw)
            .map_err(|err| rusqlite::Error::FromSqlConversionFailure(5, Type::Text, Box::new(err))),
        None => Ok(Default::default()),
    }
}

#[async_trait]
impl GraphStore for SqliteGraphStore {
    async fn add_relationship(&self, rel: &Relationship) -> MvResult<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Graph(e.to_string()))?;
        let meta_json = serde_json::to_string(&rel.metadata)?;

        conn.execute(
            "INSERT OR REPLACE INTO relationships (id, from_node, to_node, kind, weight, metadata_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                rel.id.to_string(),
                rel.from_node.to_string(),
                rel.to_node.to_string(),
                rel.kind.as_str(),
                rel.weight,
                meta_json,
                rel.created_at.to_rfc3339(),
            ],
        )
        .map_err(|e| HxError::Graph(format!("add relationship: {e}")))?;

        Ok(())
    }

    async fn get_relationship(&self, id: Uuid) -> MvResult<Option<Relationship>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Graph(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id, from_node, to_node, kind, weight, metadata_json, created_at FROM relationships WHERE id = ?1")
            .map_err(|e| HxError::Graph(e.to_string()))?;

        let rel = stmt
            .query_row(params![id.to_string()], Self::row_to_rel)
            .optional()
            .map_err(|e| HxError::Graph(e.to_string()))?;

        Ok(rel)
    }

    async fn remove_relationship(&self, id: Uuid) -> MvResult<bool> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Graph(e.to_string()))?;
        let rows = conn
            .execute(
                "DELETE FROM relationships WHERE id = ?1",
                params![id.to_string()],
            )
            .map_err(|e| HxError::Graph(e.to_string()))?;
        Ok(rows > 0)
    }

    async fn get_relationships_from(&self, node_id: Uuid) -> MvResult<Vec<Relationship>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Graph(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id, from_node, to_node, kind, weight, metadata_json, created_at FROM relationships WHERE from_node = ?1")
            .map_err(|e| HxError::Graph(e.to_string()))?;

        let rels = stmt
            .query_map(params![node_id.to_string()], Self::row_to_rel)
            .map_err(|e| HxError::Graph(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Graph(e.to_string()))?;

        Ok(rels)
    }

    async fn get_relationships_to(&self, node_id: Uuid) -> MvResult<Vec<Relationship>> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Graph(e.to_string()))?;
        let mut stmt = conn
            .prepare("SELECT id, from_node, to_node, kind, weight, metadata_json, created_at FROM relationships WHERE to_node = ?1")
            .map_err(|e| HxError::Graph(e.to_string()))?;

        let rels = stmt
            .query_map(params![node_id.to_string()], Self::row_to_rel)
            .map_err(|e| HxError::Graph(e.to_string()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| HxError::Graph(e.to_string()))?;

        Ok(rels)
    }

    async fn get_neighbors(&self, node_id: Uuid, depth: usize) -> MvResult<Vec<Uuid>> {
        if depth == 0 {
            return Ok(vec![]);
        }

        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Graph(e.to_string()))?;
        let mut visited = std::collections::HashSet::new();
        let mut frontier = vec![node_id];
        visited.insert(node_id);

        for _ in 0..depth {
            let mut next_frontier = Vec::new();
            for current in &frontier {
                let id_str = current.to_string();

                // Outgoing edges
                let mut stmt = conn
                    .prepare("SELECT to_node FROM relationships WHERE from_node = ?1")
                    .map_err(|e| HxError::Graph(e.to_string()))?;
                let outgoing: Vec<Uuid> = stmt
                    .query_map(params![id_str], |row| {
                        let s: String = row.get(0)?;
                        parse_uuid_str(0, &s)
                    })
                    .map_err(|e| HxError::Graph(e.to_string()))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| HxError::Graph(e.to_string()))?;

                // Incoming edges
                let mut stmt = conn
                    .prepare("SELECT from_node FROM relationships WHERE to_node = ?1")
                    .map_err(|e| HxError::Graph(e.to_string()))?;
                let incoming: Vec<Uuid> = stmt
                    .query_map(params![id_str], |row| {
                        let s: String = row.get(0)?;
                        parse_uuid_str(0, &s)
                    })
                    .map_err(|e| HxError::Graph(e.to_string()))?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| HxError::Graph(e.to_string()))?;

                for neighbor in outgoing.into_iter().chain(incoming) {
                    if visited.insert(neighbor) {
                        next_frontier.push(neighbor);
                    }
                }
            }
            frontier = next_frontier;
        }

        // Remove the starting node from results
        visited.remove(&node_id);
        Ok(visited.into_iter().collect())
    }

    async fn remove_node_relationships(&self, node_id: Uuid) -> MvResult<usize> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| HxError::Graph(e.to_string()))?;
        let id_str = node_id.to_string();
        let rows = conn
            .execute(
                "DELETE FROM relationships WHERE from_node = ?1 OR to_node = ?1",
                params![id_str],
            )
            .map_err(|e| HxError::Graph(e.to_string()))?;
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_get_relationship() {
        let store = SqliteGraphStore::open_in_memory().unwrap();

        let from = Uuid::now_v7();
        let to = Uuid::now_v7();
        let rel = Relationship::new(from, to, RelationKind::RelatesTo);
        let rel_id = rel.id;

        store.add_relationship(&rel).await.unwrap();

        let retrieved = store.get_relationship(rel_id).await.unwrap().unwrap();
        assert_eq!(retrieved.from_node, from);
        assert_eq!(retrieved.to_node, to);
        assert_eq!(retrieved.kind, RelationKind::RelatesTo);
    }

    #[tokio::test]
    async fn test_get_relationships_from_to() {
        let store = SqliteGraphStore::open_in_memory().unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();

        store
            .add_relationship(&Relationship::new(a, b, RelationKind::DependsOn))
            .await
            .unwrap();
        store
            .add_relationship(&Relationship::new(a, c, RelationKind::References))
            .await
            .unwrap();
        store
            .add_relationship(&Relationship::new(b, c, RelationKind::DerivedFrom))
            .await
            .unwrap();

        let from_a = store.get_relationships_from(a).await.unwrap();
        assert_eq!(from_a.len(), 2);

        let to_c = store.get_relationships_to(c).await.unwrap();
        assert_eq!(to_c.len(), 2);
    }

    #[tokio::test]
    async fn test_neighbors() {
        let store = SqliteGraphStore::open_in_memory().unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();
        let d = Uuid::now_v7();

        // a -> b -> c -> d
        store
            .add_relationship(&Relationship::new(a, b, RelationKind::RelatesTo))
            .await
            .unwrap();
        store
            .add_relationship(&Relationship::new(b, c, RelationKind::RelatesTo))
            .await
            .unwrap();
        store
            .add_relationship(&Relationship::new(c, d, RelationKind::RelatesTo))
            .await
            .unwrap();

        let depth1 = store.get_neighbors(a, 1).await.unwrap();
        assert_eq!(depth1.len(), 1);
        assert!(depth1.contains(&b));

        let depth2 = store.get_neighbors(a, 2).await.unwrap();
        assert_eq!(depth2.len(), 2);
        assert!(depth2.contains(&b));
        assert!(depth2.contains(&c));
    }

    #[tokio::test]
    async fn test_remove_node_relationships() {
        let store = SqliteGraphStore::open_in_memory().unwrap();

        let a = Uuid::now_v7();
        let b = Uuid::now_v7();
        let c = Uuid::now_v7();

        store
            .add_relationship(&Relationship::new(a, b, RelationKind::RelatesTo))
            .await
            .unwrap();
        store
            .add_relationship(&Relationship::new(c, a, RelationKind::References))
            .await
            .unwrap();

        let removed = store.remove_node_relationships(a).await.unwrap();
        assert_eq!(removed, 2);

        let from_a = store.get_relationships_from(a).await.unwrap();
        assert!(from_a.is_empty());
    }
}
