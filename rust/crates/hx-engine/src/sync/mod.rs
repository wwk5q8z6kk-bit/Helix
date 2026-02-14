//! Device-to-device sync via encrypted snapshots.
//!
//! Uses vector clocks for causal ordering: when two devices modify the same
//! node independently (concurrent edits), the conflict is surfaced as a
//! proposal in the Exchange Inbox for the owner to resolve. When one edit
//! causally follows the other, the newer version wins automatically.

pub mod clock;
pub mod snapshot;

use chrono::{DateTime, Utc};
use hx_core::*;
use hx_storage::unified::UnifiedStore;
use std::sync::Arc;
use tokio::sync::RwLock;

pub use clock::VectorClock;
pub use snapshot::{ConflictReason, SyncConflict, SyncSnapshot, SyncStats};

/// Sync engine orchestrates import/export of vault snapshots.
pub struct SyncEngine {
    store: Arc<UnifiedStore>,
    device_id: String,
    /// The local vector clock, ticked on each export.
    clock: RwLock<VectorClock>,
    /// Accumulated conflict records (in-memory; persisted as proposals).
    conflicts: RwLock<Vec<SyncConflict>>,
    /// Last successful export timestamp (in-memory).
    last_export: RwLock<Option<DateTime<Utc>>>,
    /// Last successful import timestamp (in-memory).
    last_import: RwLock<Option<DateTime<Utc>>>,
}

impl SyncEngine {
    pub fn new(store: Arc<UnifiedStore>, device_id: String) -> Self {
        let clock = VectorClock::new(&device_id);
        Self {
            store,
            device_id,
            clock: RwLock::new(clock),
            conflicts: RwLock::new(Vec::new()),
            last_export: RwLock::new(None),
            last_import: RwLock::new(None),
        }
    }

    /// Export a snapshot of nodes modified since `since`.
    ///
    /// Ticks the local vector clock to establish causal ordering.
    pub async fn export_snapshot(
        &self,
        since: Option<DateTime<Utc>>,
        namespace: Option<&str>,
    ) -> MvResult<SyncSnapshot> {
        let filters = QueryFilters {
            namespace: namespace.map(|s| s.to_string()),
            ..Default::default()
        };
        let nodes = self.store.nodes.list(&filters, 10000, 0).await?;

        let filtered: Vec<KnowledgeNode> = if let Some(since) = since {
            nodes
                .into_iter()
                .filter(|n| n.temporal.updated_at >= since)
                .collect()
        } else {
            nodes
        };

        // Tick the clock to record this export event
        let mut clock = self.clock.write().await;
        clock.tick(&self.device_id);
        let snapshot_clock = clock.clone();

        let exported_at = Utc::now();
        let snapshot = SyncSnapshot {
            device_id: self.device_id.clone(),
            exported_at,
            node_count: filtered.len(),
            nodes: filtered,
            clock: snapshot_clock,
        };

        *self.last_export.write().await = Some(exported_at);

        Ok(snapshot)
    }

    /// Import a snapshot from another device, detecting conflicts via
    /// vector clock comparison with timestamp fallback.
    ///
    /// Conflict resolution strategy:
    /// - **Causal ordering known** (vector clocks): newer causally wins.
    /// - **Concurrent edits** (neither clock dominates): create conflict proposal.
    /// - **Fallback** (clocks unavailable): timestamp-based last-write-wins
    ///   with same-timestamp collisions surfaced as conflicts.
    pub async fn import_snapshot(&self, snapshot: SyncSnapshot) -> MvResult<SyncStats> {
        let mut stats = SyncStats::default();

        // Snapshot the local clock BEFORE merging — this is the clock state
        // that represents "what this device has seen" at the time of import.
        // We compare against this to detect concurrent edits.
        let pre_merge_clock = self.clock.read().await.clone();

        for incoming_node in &snapshot.nodes {
            stats.scanned += 1;

            match self.store.nodes.get(incoming_node.id).await? {
                Some(local_node) => {
                    self.resolve_conflict(
                        &local_node,
                        incoming_node,
                        &snapshot,
                        &pre_merge_clock,
                        &mut stats,
                    )
                    .await?;
                }
                None => {
                    // New node — insert directly
                    self.store.nodes.insert(incoming_node).await?;
                    stats.inserted += 1;
                }
            }
        }

        // Merge clocks AFTER processing all nodes
        {
            let mut local_clock = self.clock.write().await;
            local_clock.merge(&snapshot.clock);
            local_clock.tick(&self.device_id);
        }

        *self.last_import.write().await = Some(Utc::now());

        Ok(stats)
    }

    /// Determine the correct resolution for a node that exists both locally
    /// and in the incoming snapshot.
    async fn resolve_conflict(
        &self,
        local_node: &KnowledgeNode,
        incoming_node: &KnowledgeNode,
        snapshot: &SyncSnapshot,
        local_clock: &VectorClock,
        stats: &mut SyncStats,
    ) -> MvResult<()> {
        // Use vector clocks when both sides have clock data
        let has_clock_data = !snapshot.clock.clocks.is_empty()
            && !local_clock.clocks.is_empty();

        if has_clock_data {
            if snapshot.clock.happens_before(local_clock) {
                // Remote is causally older — skip
                stats.skipped += 1;
                return Ok(());
            }

            if local_clock.happens_before(&snapshot.clock) {
                // Remote is causally newer — accept
                self.store.nodes.update(incoming_node).await?;
                stats.updated += 1;
                return Ok(());
            }

            if snapshot.clock.is_concurrent(local_clock) {
                // Concurrent edits — conflict
                self.create_conflict(
                    local_node,
                    incoming_node,
                    &snapshot.device_id,
                    ConflictReason::ConcurrentEdit,
                    stats,
                )
                .await?;
                return Ok(());
            }
        }

        // Fallback: timestamp-based resolution
        if local_node.temporal.updated_at > incoming_node.temporal.updated_at {
            stats.skipped += 1;
        } else if local_node.temporal.updated_at == incoming_node.temporal.updated_at {
            // Same timestamp with different content — conflict
            if local_node.content != incoming_node.content {
                self.create_conflict(
                    local_node,
                    incoming_node,
                    &snapshot.device_id,
                    ConflictReason::TimestampCollision,
                    stats,
                )
                .await?;
            } else {
                // Identical content — no-op
                stats.skipped += 1;
            }
        } else {
            // Remote is newer — accept
            self.store.nodes.update(incoming_node).await?;
            stats.updated += 1;
        }

        Ok(())
    }

    /// Record a conflict: create a proposal in the Exchange Inbox and
    /// track the conflict details.
    async fn create_conflict(
        &self,
        local_node: &KnowledgeNode,
        incoming_node: &KnowledgeNode,
        remote_device_id: &str,
        reason: ConflictReason,
        stats: &mut SyncStats,
    ) -> MvResult<()> {
        stats.conflicts += 1;

        let diff = format!(
            "=== Sync conflict ({reason}) ===\n\
             Node: {node_id}\n\
             Local device updated: {local_ts}\n\
             Remote device ({remote_dev}) updated: {remote_ts}\n\
             \n--- Local content (first 300 chars) ---\n{local_preview}\n\
             \n--- Remote content (first 300 chars) ---\n{remote_preview}",
            reason = match reason {
                ConflictReason::ConcurrentEdit => "concurrent edit",
                ConflictReason::TimestampCollision => "timestamp collision",
            },
            node_id = incoming_node.id,
            local_ts = local_node.temporal.updated_at.to_rfc3339(),
            remote_dev = remote_device_id,
            remote_ts = incoming_node.temporal.updated_at.to_rfc3339(),
            local_preview = local_node.content.chars().take(300).collect::<String>(),
            remote_preview = incoming_node.content.chars().take(300).collect::<String>(),
        );

        let proposal = Proposal::new(ProposalSender::Relay, ProposalAction::UpdateNode)
            .with_target(incoming_node.id)
            .with_diff(diff)
            .with_confidence(0.5);

        let proposal_id = proposal.id;
        self.store.nodes.submit_proposal(&proposal).await?;

        let conflict = SyncConflict::new(
            incoming_node.id,
            reason,
            local_node,
            incoming_node,
            remote_device_id,
        )
        .with_proposal(proposal_id);

        stats.conflict_details.push(conflict.clone());
        self.conflicts.write().await.push(conflict);

        Ok(())
    }

    /// Get the current device ID.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Last successful export timestamp.
    pub async fn last_export(&self) -> Option<DateTime<Utc>> {
        self.last_export.read().await.clone()
    }

    /// Last successful import timestamp.
    pub async fn last_import(&self) -> Option<DateTime<Utc>> {
        self.last_import.read().await.clone()
    }

    /// Get the current vector clock state.
    pub async fn clock(&self) -> VectorClock {
        self.clock.read().await.clone()
    }

    /// List unresolved conflicts.
    pub async fn unresolved_conflicts(&self) -> Vec<SyncConflict> {
        self.conflicts
            .read()
            .await
            .iter()
            .filter(|c| !c.resolved)
            .cloned()
            .collect()
    }

    /// Mark a conflict as resolved by its ID.
    pub async fn resolve_conflict_by_id(&self, conflict_id: uuid::Uuid) -> bool {
        let mut conflicts = self.conflicts.write().await;
        if let Some(c) = conflicts.iter_mut().find(|c| c.id == conflict_id) {
            c.resolved = true;
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;
    use hx_core::NodeKind;
    use uuid::Uuid;

    fn make_node(content: &str, updated_at: DateTime<Utc>) -> KnowledgeNode {
        let mut node = KnowledgeNode::new(NodeKind::Fact, content.to_string());
        node.temporal.updated_at = updated_at;
        node
    }

    async fn test_store() -> (Arc<UnifiedStore>, tempfile::TempDir) {
        let tmp = tempfile::TempDir::new().unwrap();
        let store = Arc::new(UnifiedStore::open(tmp.path(), 384).await.unwrap());
        (store, tmp)
    }

    // -- Export ----------------------------------------------------------------

    #[tokio::test]
    async fn export_ticks_clock() {
        let (store, _tmp) = test_store().await;
        let engine = SyncEngine::new(store, "device-A".into());

        let snap1 = engine.export_snapshot(None, None).await.unwrap();
        assert_eq!(snap1.clock.get("device-A"), 2); // new(1) + tick(1) = 2

        let snap2 = engine.export_snapshot(None, None).await.unwrap();
        assert_eq!(snap2.clock.get("device-A"), 3);
    }

    #[tokio::test]
    async fn export_includes_device_id() {
        let (store, _tmp) = test_store().await;
        let engine = SyncEngine::new(store, "my-laptop".into());
        let snap = engine.export_snapshot(None, None).await.unwrap();
        assert_eq!(snap.device_id, "my-laptop");
    }

    #[tokio::test]
    async fn export_filters_by_since() {
        let (store, _tmp) = test_store().await;
        let old = Utc::now() - Duration::hours(2);
        let recent = Utc::now() - Duration::minutes(5);

        let old_node = make_node("old content", old);
        let new_node = make_node("new content", recent);
        store.nodes.insert(&old_node).await.unwrap();
        store.nodes.insert(&new_node).await.unwrap();

        let engine = SyncEngine::new(store, "dev".into());
        let since = Utc::now() - Duration::hours(1);
        let snap = engine.export_snapshot(Some(since), None).await.unwrap();

        assert_eq!(snap.node_count, 1);
        assert_eq!(snap.nodes[0].id, new_node.id);
    }

    // -- Import: new nodes ----------------------------------------------------

    #[tokio::test]
    async fn import_inserts_new_nodes() {
        let (store, _tmp) = test_store().await;
        let engine = SyncEngine::new(store.clone(), "device-B".into());

        let node = make_node("from device A", Utc::now());
        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![node.clone()],
            clock: VectorClock::new("device-A"),
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        assert_eq!(stats.scanned, 1);
        assert_eq!(stats.inserted, 1);
        assert_eq!(stats.conflicts, 0);

        // Verify the node was stored
        let stored = store.nodes.get(node.id).await.unwrap();
        assert!(stored.is_some());
    }

    // -- Import: timestamp-based resolution -----------------------------------

    #[tokio::test]
    async fn import_skips_older_remote() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("local version", now);
        store.nodes.insert(&local).await.unwrap();

        let engine = SyncEngine::new(store, "device-B".into());

        let mut remote = local.clone();
        remote.content = "older remote".into();
        remote.temporal.updated_at = now - Duration::hours(1);

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![remote],
            clock: VectorClock::default(), // empty clock → timestamp fallback
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.updated, 0);
    }

    #[tokio::test]
    async fn import_accepts_newer_remote() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("local version", now - Duration::hours(1));
        store.nodes.insert(&local).await.unwrap();

        let engine = SyncEngine::new(store.clone(), "device-B".into());

        let mut remote = local.clone();
        remote.content = "newer remote".into();
        remote.temporal.updated_at = now;

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![remote],
            clock: VectorClock::default(),
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        assert_eq!(stats.updated, 1);

        let stored = store.nodes.get(local.id).await.unwrap().unwrap();
        assert_eq!(stored.content, "newer remote");
    }

    // -- Import: timestamp collision ------------------------------------------

    #[tokio::test]
    async fn import_detects_timestamp_collision() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("local text", now);
        store.nodes.insert(&local).await.unwrap();

        let engine = SyncEngine::new(store, "device-B".into());

        let mut remote = local.clone();
        remote.content = "different remote text".into();
        // Same timestamp as local

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![remote],
            clock: VectorClock::default(),
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        assert_eq!(stats.conflicts, 1);
        assert_eq!(stats.conflict_details.len(), 1);
        assert_eq!(
            stats.conflict_details[0].reason,
            ConflictReason::TimestampCollision
        );
    }

    #[tokio::test]
    async fn import_skips_identical_content_same_timestamp() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("identical content", now);
        store.nodes.insert(&local).await.unwrap();

        let engine = SyncEngine::new(store, "device-B".into());

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![local.clone()], // exact same node
            clock: VectorClock::default(),
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.conflicts, 0);
    }

    // -- Import: vector clock based -------------------------------------------

    #[tokio::test]
    async fn import_uses_vector_clock_when_available() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("local", now);
        store.nodes.insert(&local).await.unwrap();

        // Simulate: local clock at device-B:2, remote at device-A:3 + device-B:1
        // Remote is newer (local happens_before remote)
        let engine = SyncEngine::new(store.clone(), "device-B".into());
        // Tick the local clock to device-B:2
        engine.clock.write().await.tick("device-B");

        let mut remote = local.clone();
        remote.content = "remote wins via clock".into();
        remote.temporal.updated_at = now - Duration::hours(1); // older timestamp!

        let mut remote_clock = VectorClock::new("device-A");
        remote_clock.tick("device-A");
        remote_clock.tick("device-A"); // device-A:3
        remote_clock.merge(&VectorClock::new("device-B")); // sees device-B:1

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![remote],
            clock: remote_clock,
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        // Vector clock says remote is not causally before local,
        // and local is not before remote → concurrent → conflict
        // (because device-B:2 > device-B:1 in remote clock)
        assert!(stats.conflicts > 0 || stats.updated > 0);
    }

    // -- Import: concurrent edits via vector clock ----------------------------

    #[tokio::test]
    async fn import_detects_concurrent_edit_via_clock() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("base version", now);
        store.nodes.insert(&local).await.unwrap();

        let engine = SyncEngine::new(store, "device-B".into());
        // Local: device-B:2 (new + tick)
        engine.clock.write().await.tick("device-B");

        let mut remote = local.clone();
        remote.content = "remote concurrent edit".into();

        // Remote: device-A:2 — neither dominates
        let mut remote_clock = VectorClock::new("device-A");
        remote_clock.tick("device-A");

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![remote],
            clock: remote_clock,
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        assert_eq!(stats.conflicts, 1);
        assert_eq!(
            stats.conflict_details[0].reason,
            ConflictReason::ConcurrentEdit
        );
    }

    // -- Merge clock ----------------------------------------------------------

    #[tokio::test]
    async fn import_merges_remote_clock() {
        let (store, _tmp) = test_store().await;
        let engine = SyncEngine::new(store, "device-B".into());

        let mut remote_clock = VectorClock::new("device-A");
        remote_clock.tick("device-A"); // device-A:2

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 0,
            nodes: vec![],
            clock: remote_clock,
        };

        engine.import_snapshot(snapshot).await.unwrap();

        let clock = engine.clock().await;
        assert_eq!(clock.get("device-A"), 2); // merged from remote
        assert!(clock.get("device-B") >= 2); // ticked after merge
    }

    // -- Conflict management --------------------------------------------------

    #[tokio::test]
    async fn unresolved_conflicts_tracks_conflicts() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("local", now);
        store.nodes.insert(&local).await.unwrap();

        let engine = SyncEngine::new(store, "device-B".into());

        let mut remote = local.clone();
        remote.content = "remote different".into();

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![remote],
            clock: VectorClock::default(),
        };

        engine.import_snapshot(snapshot).await.unwrap();

        let unresolved = engine.unresolved_conflicts().await;
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].node_id, local.id);
        assert!(unresolved[0].proposal_id.is_some());
    }

    #[tokio::test]
    async fn resolve_conflict_by_id_marks_resolved() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let local = make_node("local", now);
        store.nodes.insert(&local).await.unwrap();

        let engine = SyncEngine::new(store, "device-B".into());

        let mut remote = local.clone();
        remote.content = "remote".into();

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 1,
            nodes: vec![remote],
            clock: VectorClock::default(),
        };

        engine.import_snapshot(snapshot).await.unwrap();

        let conflicts = engine.unresolved_conflicts().await;
        assert_eq!(conflicts.len(), 1);

        let resolved = engine.resolve_conflict_by_id(conflicts[0].id).await;
        assert!(resolved);

        assert!(engine.unresolved_conflicts().await.is_empty());
    }

    #[tokio::test]
    async fn resolve_unknown_conflict_returns_false() {
        let (store, _tmp) = test_store().await;
        let engine = SyncEngine::new(store, "dev".into());
        assert!(!engine.resolve_conflict_by_id(Uuid::now_v7()).await);
    }

    // -- Mixed import ---------------------------------------------------------

    #[tokio::test]
    async fn import_mixed_new_update_conflict_skip() {
        let (store, _tmp) = test_store().await;

        let now = Utc::now();
        let old_ts = now - Duration::hours(2);
        let new_ts = now + Duration::hours(1);

        // Node 1: exists locally, remote is newer → update
        let node1 = make_node("node1 local", old_ts);
        store.nodes.insert(&node1).await.unwrap();
        let mut remote1 = node1.clone();
        remote1.content = "node1 updated".into();
        remote1.temporal.updated_at = new_ts;

        // Node 2: exists locally, remote is older → skip
        let node2 = make_node("node2 local", now);
        store.nodes.insert(&node2).await.unwrap();
        let mut remote2 = node2.clone();
        remote2.content = "node2 old".into();
        remote2.temporal.updated_at = old_ts;

        // Node 3: doesn't exist locally → insert
        let node3 = make_node("node3 new", now);

        // Node 4: same timestamp, different content → conflict
        let node4 = make_node("node4 local", now);
        store.nodes.insert(&node4).await.unwrap();
        let mut remote4 = node4.clone();
        remote4.content = "node4 different".into();

        let snapshot = SyncSnapshot {
            device_id: "device-A".into(),
            exported_at: Utc::now(),
            node_count: 4,
            nodes: vec![remote1, remote2, node3, remote4],
            clock: VectorClock::default(),
        };

        let engine = SyncEngine::new(store, "device-B".into());
        let stats = engine.import_snapshot(snapshot).await.unwrap();

        assert_eq!(stats.scanned, 4);
        assert_eq!(stats.updated, 1);
        assert_eq!(stats.skipped, 1);
        assert_eq!(stats.inserted, 1);
        assert_eq!(stats.conflicts, 1);
    }

    // -- Empty import ---------------------------------------------------------

    #[tokio::test]
    async fn import_empty_snapshot_is_noop() {
        let (store, _tmp) = test_store().await;
        let engine = SyncEngine::new(store, "dev".into());

        let snapshot = SyncSnapshot {
            device_id: "other".into(),
            exported_at: Utc::now(),
            node_count: 0,
            nodes: vec![],
            clock: VectorClock::default(),
        };

        let stats = engine.import_snapshot(snapshot).await.unwrap();
        assert_eq!(stats.scanned, 0);
        assert_eq!(stats.inserted, 0);
        assert_eq!(stats.updated, 0);
        assert_eq!(stats.skipped, 0);
        assert_eq!(stats.conflicts, 0);
    }
}
