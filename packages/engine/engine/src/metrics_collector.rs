//! Metrics collector for agent activity, knowledge growth, and system health.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Collected metric data point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricPoint {
    pub name: String,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
}

/// Metrics bucket for a time period.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsBucket {
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub metrics: Vec<MetricPoint>,
}

/// Summary statistics across time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_nodes: usize,
    pub nodes_by_kind: HashMap<String, usize>,
    pub nodes_created_last_24h: usize,
    pub nodes_created_last_7d: usize,
    pub agent_intents_total: usize,
    pub agent_intents_applied: usize,
    pub agent_intents_dismissed: usize,
    pub search_queries_total: usize,
    pub avg_search_latency_ms: f64,
    pub active_proposals: usize,
    pub uptime_seconds: u64,
}

impl Default for MetricsSummary {
    fn default() -> Self {
        Self {
            total_nodes: 0,
            nodes_by_kind: HashMap::new(),
            nodes_created_last_24h: 0,
            nodes_created_last_7d: 0,
            agent_intents_total: 0,
            agent_intents_applied: 0,
            agent_intents_dismissed: 0,
            search_queries_total: 0,
            avg_search_latency_ms: 0.0,
            active_proposals: 0,
            uptime_seconds: 0,
        }
    }
}

/// Collects and aggregates metrics from the engine.
pub struct MetricsCollector {
    start_time: DateTime<Utc>,
    counters: Arc<RwLock<HashMap<String, u64>>>,
    gauges: Arc<RwLock<HashMap<String, f64>>>,
    histograms: Arc<RwLock<HashMap<String, Vec<f64>>>>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            start_time: Utc::now(),
            counters: Arc::new(RwLock::new(HashMap::new())),
            gauges: Arc::new(RwLock::new(HashMap::new())),
            histograms: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Increment a counter.
    pub async fn increment(&self, name: &str, amount: u64) {
        let mut counters = self.counters.write().await;
        *counters.entry(name.to_string()).or_insert(0) += amount;
    }

    /// Set a gauge value.
    pub async fn set_gauge(&self, name: &str, value: f64) {
        self.gauges.write().await.insert(name.to_string(), value);
    }

    /// Record a histogram value (e.g., latency).
    pub async fn record_histogram(&self, name: &str, value: f64) {
        self.histograms
            .write()
            .await
            .entry(name.to_string())
            .or_default()
            .push(value);
    }

    /// Get current counter values.
    pub async fn get_counters(&self) -> HashMap<String, u64> {
        self.counters.read().await.clone()
    }

    /// Get current gauge values.
    pub async fn get_gauges(&self) -> HashMap<String, f64> {
        self.gauges.read().await.clone()
    }

    /// Get histogram statistics.
    pub async fn get_histogram_stats(&self, name: &str) -> Option<HistogramStats> {
        let histograms = self.histograms.read().await;
        let values = histograms.get(name)?;
        if values.is_empty() {
            return None;
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = sorted.iter().sum();
        let count = sorted.len();
        let mean = sum / count as f64;
        let p50 = sorted[count / 2];
        let p95 = sorted[(count as f64 * 0.95) as usize];
        let p99 = sorted[((count as f64 * 0.99) as usize).min(count - 1)];

        Some(HistogramStats {
            count,
            sum,
            mean,
            min: sorted[0],
            max: sorted[count - 1],
            p50,
            p95,
            p99,
        })
    }

    /// Get uptime in seconds.
    pub fn uptime_seconds(&self) -> u64 {
        (Utc::now() - self.start_time).num_seconds().max(0) as u64
    }

    /// Snapshot all metrics as JSON.
    pub async fn snapshot(&self) -> serde_json::Value {
        let counters = self.get_counters().await;
        let gauges = self.get_gauges().await;

        serde_json::json!({
            "uptime_seconds": self.uptime_seconds(),
            "counters": counters,
            "gauges": gauges,
            "collected_at": Utc::now().to_rfc3339(),
        })
    }
}

/// Statistics from a histogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramStats {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}
