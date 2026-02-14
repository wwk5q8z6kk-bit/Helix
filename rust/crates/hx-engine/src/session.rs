//! Session memory: in-memory ring buffer for conversation context.
//!
//! Tracks recent query/result pairs per session_id so follow-up queries
//! can leverage prior context. Used by the query rewriter and recall pipeline
//! to improve retrieval for conversational interactions.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::RwLock;
use tracing::debug;

use crate::config::SessionConfig;

/// A single turn in a session: the query and a summary of results.
#[derive(Debug, Clone)]
pub struct SessionTurn {
    pub query: String,
    pub result_summary: String,
    pub timestamp: Instant,
}

/// Per-session state: a ring buffer of recent turns.
#[derive(Debug)]
struct SessionState {
    turns: VecDeque<SessionTurn>,
    last_access: Instant,
    max_turns: usize,
}

impl SessionState {
    fn new(max_turns: usize) -> Self {
        Self {
            turns: VecDeque::with_capacity(max_turns),
            last_access: Instant::now(),
            max_turns,
        }
    }

    fn add_turn(&mut self, query: String, result_summary: String) {
        if self.turns.len() >= self.max_turns {
            self.turns.pop_front();
        }
        self.turns.push_back(SessionTurn {
            query,
            result_summary,
            timestamp: Instant::now(),
        });
        self.last_access = Instant::now();
    }

    fn get_context(&self, limit: usize) -> Vec<(String, String)> {
        self.turns
            .iter()
            .rev()
            .take(limit)
            .rev()
            .map(|t| (t.query.clone(), t.result_summary.clone()))
            .collect()
    }
}

/// Thread-safe session store backed by an in-memory HashMap.
pub struct InMemorySessionStore {
    sessions: Arc<RwLock<std::collections::HashMap<String, SessionState>>>,
    config: SessionConfig,
}

impl InMemorySessionStore {
    pub fn new(config: SessionConfig) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(std::collections::HashMap::new())),
            config,
        }
    }

    /// Record a query+result turn in the session.
    pub async fn add_turn(
        &self,
        session_id: &str,
        query: &str,
        result_summary: &str,
    ) {
        if !self.config.enabled {
            return;
        }

        let mut sessions = self.sessions.write().await;
        let state = sessions
            .entry(session_id.to_string())
            .or_insert_with(|| SessionState::new(self.config.max_turns));
        state.add_turn(query.to_string(), result_summary.to_string());

        debug!(session_id, turns = state.turns.len(), "session turn added");
    }

    /// Get recent turns for a session (newest last).
    pub async fn get_context(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Vec<(String, String)> {
        if !self.config.enabled {
            return vec![];
        }

        let sessions = self.sessions.read().await;
        sessions
            .get(session_id)
            .map(|s| s.get_context(limit))
            .unwrap_or_default()
    }

    /// Build a context string from recent session turns for injection into prompts.
    pub async fn build_context_string(&self, session_id: &str, max_turns: usize) -> Option<String> {
        let turns = self.get_context(session_id, max_turns).await;
        if turns.is_empty() {
            return None;
        }

        let context = turns
            .iter()
            .enumerate()
            .map(|(i, (q, r))| format!("Turn {}: Q: {} | A: {}", i + 1, q, r))
            .collect::<Vec<_>>()
            .join("\n");

        Some(format!("Previous conversation context:\n{context}"))
    }

    /// Clear a specific session.
    pub async fn clear_session(&self, session_id: &str) {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
    }

    /// Remove sessions that haven't been accessed within the TTL.
    pub async fn expire_sessions(&self) -> usize {
        let ttl = std::time::Duration::from_secs(self.config.ttl_secs);
        let mut sessions = self.sessions.write().await;
        let before = sessions.len();
        sessions.retain(|_, state| state.last_access.elapsed() < ttl);
        let expired = before - sessions.len();
        if expired > 0 {
            debug!(expired, remaining = sessions.len(), "sessions expired");
        }
        expired
    }

    /// Get the number of active sessions.
    pub async fn session_count(&self) -> usize {
        self.sessions.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SessionConfig {
        SessionConfig {
            enabled: true,
            max_turns: 5,
            ttl_secs: 60,
        }
    }

    #[tokio::test]
    async fn add_and_retrieve_turns() {
        let store = InMemorySessionStore::new(test_config());

        store.add_turn("s1", "what is rust?", "Rust is a systems language").await;
        store.add_turn("s1", "how about memory safety?", "Rust uses ownership").await;

        let context = store.get_context("s1", 10).await;
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].0, "what is rust?");
        assert_eq!(context[1].0, "how about memory safety?");
    }

    #[tokio::test]
    async fn ring_buffer_eviction() {
        let store = InMemorySessionStore::new(SessionConfig {
            enabled: true,
            max_turns: 2,
            ttl_secs: 60,
        });

        store.add_turn("s1", "q1", "r1").await;
        store.add_turn("s1", "q2", "r2").await;
        store.add_turn("s1", "q3", "r3").await;

        let context = store.get_context("s1", 10).await;
        assert_eq!(context.len(), 2);
        assert_eq!(context[0].0, "q2");
        assert_eq!(context[1].0, "q3");
    }

    #[tokio::test]
    async fn disabled_store_returns_empty() {
        let store = InMemorySessionStore::new(SessionConfig {
            enabled: false,
            max_turns: 5,
            ttl_secs: 60,
        });

        store.add_turn("s1", "q1", "r1").await;
        let context = store.get_context("s1", 10).await;
        assert!(context.is_empty());
    }

    #[tokio::test]
    async fn clear_session() {
        let store = InMemorySessionStore::new(test_config());

        store.add_turn("s1", "q1", "r1").await;
        assert_eq!(store.session_count().await, 1);

        store.clear_session("s1").await;
        assert_eq!(store.session_count().await, 0);
    }

    #[tokio::test]
    async fn build_context_string() {
        let store = InMemorySessionStore::new(test_config());

        store.add_turn("s1", "what is X?", "X is a thing").await;
        store.add_turn("s1", "tell me more", "More details about X").await;

        let ctx = store.build_context_string("s1", 5).await;
        assert!(ctx.is_some());
        let ctx = ctx.unwrap();
        assert!(ctx.contains("what is X?"));
        assert!(ctx.contains("tell me more"));
    }
}
