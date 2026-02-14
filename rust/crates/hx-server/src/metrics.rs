//! Lightweight Prometheus-style metrics for Helix server.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use axum::{
    extract::Request,
    http::{header::CONTENT_TYPE, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};

/// Fixed histogram buckets in milliseconds.
const LATENCY_BUCKETS_MS: &[u64] = &[5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000];

#[derive(Debug)]
pub struct MetricsCounters {
    rest_requests_total: AtomicU64,
    rest_errors_total: AtomicU64,
    grpc_requests_total: AtomicU64,
    grpc_errors_total: AtomicU64,
    vault_sealed_http_requests_blocked_total: AtomicU64,
    vault_sealed_grpc_requests_blocked_total: AtomicU64,
    vault_unseal_failures_total: AtomicU64,
    vault_unseal_rate_limited_total: AtomicU64,
    vault_sealed_migration_success_total: AtomicU64,
    vault_sealed_migration_failures_total: AtomicU64,
    vault_runtime_rebuild_success_total: AtomicU64,
    vault_runtime_rebuild_failures_total: AtomicU64,
    /// Histogram bucket counters for REST request latency.
    /// One counter per bucket + one for +Inf.
    rest_latency_buckets: Vec<AtomicU64>,
    rest_latency_sum_us: AtomicU64,
    rest_latency_count: AtomicU64,
}

impl Default for MetricsCounters {
    fn default() -> Self {
        let buckets: Vec<AtomicU64> = (0..LATENCY_BUCKETS_MS.len() + 1)
            .map(|_| AtomicU64::new(0))
            .collect();
        Self {
            rest_requests_total: AtomicU64::new(0),
            rest_errors_total: AtomicU64::new(0),
            grpc_requests_total: AtomicU64::new(0),
            grpc_errors_total: AtomicU64::new(0),
            vault_sealed_http_requests_blocked_total: AtomicU64::new(0),
            vault_sealed_grpc_requests_blocked_total: AtomicU64::new(0),
            vault_unseal_failures_total: AtomicU64::new(0),
            vault_unseal_rate_limited_total: AtomicU64::new(0),
            vault_sealed_migration_success_total: AtomicU64::new(0),
            vault_sealed_migration_failures_total: AtomicU64::new(0),
            vault_runtime_rebuild_success_total: AtomicU64::new(0),
            vault_runtime_rebuild_failures_total: AtomicU64::new(0),
            rest_latency_buckets: buckets,
            rest_latency_sum_us: AtomicU64::new(0),
            rest_latency_count: AtomicU64::new(0),
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct MetricsSnapshot {
    pub rest_requests_total: u64,
    pub rest_errors_total: u64,
    pub grpc_requests_total: u64,
    pub grpc_errors_total: u64,
    pub vault_sealed_http_requests_blocked_total: u64,
    pub vault_sealed_grpc_requests_blocked_total: u64,
    pub vault_unseal_failures_total: u64,
    pub vault_unseal_rate_limited_total: u64,
    pub vault_sealed_migration_success_total: u64,
    pub vault_sealed_migration_failures_total: u64,
    pub vault_runtime_rebuild_success_total: u64,
    pub vault_runtime_rebuild_failures_total: u64,
}

static METRICS: OnceLock<MetricsCounters> = OnceLock::new();

pub fn init_metrics() {
    let _ = METRICS.get_or_init(MetricsCounters::default);
}

pub fn get_metrics() -> &'static MetricsCounters {
    METRICS.get_or_init(MetricsCounters::default)
}

impl MetricsCounters {
    pub fn incr_rest_request(&self) {
        self.rest_requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_rest_error(&self) {
        self.rest_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_grpc_request(&self) {
        self.grpc_requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_grpc_error(&self) {
        self.grpc_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_sealed_http_blocked(&self) {
        self.vault_sealed_http_requests_blocked_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_sealed_grpc_blocked(&self) {
        self.vault_sealed_grpc_requests_blocked_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_unseal_failure(&self) {
        self.vault_unseal_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_unseal_rate_limited(&self) {
        self.vault_unseal_rate_limited_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_migration_success(&self) {
        self.vault_sealed_migration_success_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_migration_failure(&self) {
        self.vault_sealed_migration_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_rebuild_success(&self) {
        self.vault_runtime_rebuild_success_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn incr_vault_rebuild_failure(&self) {
        self.vault_runtime_rebuild_failures_total
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_rest_latency_us(&self, latency_us: u64) {
        let latency_ms = latency_us / 1000;
        // Increment all buckets where the latency fits (cumulative histogram)
        for (i, &bound) in LATENCY_BUCKETS_MS.iter().enumerate() {
            if latency_ms <= bound {
                self.rest_latency_buckets[i].fetch_add(1, Ordering::Relaxed);
            }
        }
        // +Inf bucket always increments
        self.rest_latency_buckets
            .last()
            .unwrap()
            .fetch_add(1, Ordering::Relaxed);
        self.rest_latency_sum_us
            .fetch_add(latency_us, Ordering::Relaxed);
        self.rest_latency_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            rest_requests_total: self.rest_requests_total.load(Ordering::Relaxed),
            rest_errors_total: self.rest_errors_total.load(Ordering::Relaxed),
            grpc_requests_total: self.grpc_requests_total.load(Ordering::Relaxed),
            grpc_errors_total: self.grpc_errors_total.load(Ordering::Relaxed),
            vault_sealed_http_requests_blocked_total: self
                .vault_sealed_http_requests_blocked_total
                .load(Ordering::Relaxed),
            vault_sealed_grpc_requests_blocked_total: self
                .vault_sealed_grpc_requests_blocked_total
                .load(Ordering::Relaxed),
            vault_unseal_failures_total: self.vault_unseal_failures_total.load(Ordering::Relaxed),
            vault_unseal_rate_limited_total: self
                .vault_unseal_rate_limited_total
                .load(Ordering::Relaxed),
            vault_sealed_migration_success_total: self
                .vault_sealed_migration_success_total
                .load(Ordering::Relaxed),
            vault_sealed_migration_failures_total: self
                .vault_sealed_migration_failures_total
                .load(Ordering::Relaxed),
            vault_runtime_rebuild_success_total: self
                .vault_runtime_rebuild_success_total
                .load(Ordering::Relaxed),
            vault_runtime_rebuild_failures_total: self
                .vault_runtime_rebuild_failures_total
                .load(Ordering::Relaxed),
        }
    }
}

pub async fn metrics_middleware(request: Request, next: Next) -> Response {
    let metrics = get_metrics();
    metrics.incr_rest_request();
    let start = Instant::now();
    let response = next.run(request).await;
    let elapsed_us = start.elapsed().as_micros() as u64;
    metrics.observe_rest_latency_us(elapsed_us);
    if !response.status().is_success() {
        metrics.incr_rest_error();
    }
    response
}

pub async fn metrics_handler() -> impl IntoResponse {
    let m = get_metrics();
    let snapshot = m.snapshot();

    let mut body = format!(
        "# HELP helix_rest_requests_total Total REST requests handled\n\
# TYPE helix_rest_requests_total counter\n\
helix_rest_requests_total {}\n\
# HELP helix_rest_errors_total Total REST requests returning non-2xx\n\
# TYPE helix_rest_errors_total counter\n\
helix_rest_errors_total {}\n\
# HELP helix_grpc_requests_total Total gRPC requests handled\n\
# TYPE helix_grpc_requests_total counter\n\
helix_grpc_requests_total {}\n\
# HELP helix_grpc_errors_total Total gRPC requests returning errors\n\
# TYPE helix_grpc_errors_total counter\n\
helix_grpc_errors_total {}\n",
        snapshot.rest_requests_total,
        snapshot.rest_errors_total,
        snapshot.grpc_requests_total,
        snapshot.grpc_errors_total,
    );
    body.push_str(&format!(
        "# HELP helix_vault_sealed_http_requests_blocked_total Total HTTP requests blocked because vault is sealed\n\
# TYPE helix_vault_sealed_http_requests_blocked_total counter\n\
helix_vault_sealed_http_requests_blocked_total {}\n\
# HELP helix_vault_sealed_grpc_requests_blocked_total Total gRPC requests blocked because vault is sealed\n\
# TYPE helix_vault_sealed_grpc_requests_blocked_total counter\n\
helix_vault_sealed_grpc_requests_blocked_total {}\n\
# HELP helix_vault_unseal_failures_total Total failed vault unseal attempts\n\
# TYPE helix_vault_unseal_failures_total counter\n\
helix_vault_unseal_failures_total {}\n\
# HELP helix_vault_unseal_rate_limited_total Total unseal attempts blocked by rate limiting\n\
# TYPE helix_vault_unseal_rate_limited_total counter\n\
helix_vault_unseal_rate_limited_total {}\n\
# HELP helix_vault_sealed_migration_success_total Total successful post-unseal sealed migrations\n\
# TYPE helix_vault_sealed_migration_success_total counter\n\
helix_vault_sealed_migration_success_total {}\n\
# HELP helix_vault_sealed_migration_failures_total Total failed post-unseal sealed migrations\n\
# TYPE helix_vault_sealed_migration_failures_total counter\n\
helix_vault_sealed_migration_failures_total {}\n\
# HELP helix_vault_runtime_rebuild_success_total Total successful post-unseal runtime index rebuilds\n\
# TYPE helix_vault_runtime_rebuild_success_total counter\n\
helix_vault_runtime_rebuild_success_total {}\n\
# HELP helix_vault_runtime_rebuild_failures_total Total failed post-unseal runtime index rebuilds\n\
# TYPE helix_vault_runtime_rebuild_failures_total counter\n\
helix_vault_runtime_rebuild_failures_total {}\n",
        snapshot.vault_sealed_http_requests_blocked_total,
        snapshot.vault_sealed_grpc_requests_blocked_total,
        snapshot.vault_unseal_failures_total,
        snapshot.vault_unseal_rate_limited_total,
        snapshot.vault_sealed_migration_success_total,
        snapshot.vault_sealed_migration_failures_total,
        snapshot.vault_runtime_rebuild_success_total,
        snapshot.vault_runtime_rebuild_failures_total,
    ));

    // Latency histogram
    body.push_str(
        "# HELP helix_rest_request_duration_seconds REST request latency\n\
# TYPE helix_rest_request_duration_seconds histogram\n",
    );
    for (i, &bound_ms) in LATENCY_BUCKETS_MS.iter().enumerate() {
        let count = m.rest_latency_buckets[i].load(Ordering::Relaxed);
        let bound_s = bound_ms as f64 / 1000.0;
        body.push_str(&format!(
            "helix_rest_request_duration_seconds_bucket{{le=\"{bound_s}\"}} {count}\n"
        ));
    }
    let inf_count = m
        .rest_latency_buckets
        .last()
        .unwrap()
        .load(Ordering::Relaxed);
    body.push_str(&format!(
        "helix_rest_request_duration_seconds_bucket{{le=\"+Inf\"}} {inf_count}\n"
    ));
    let sum_s = m.rest_latency_sum_us.load(Ordering::Relaxed) as f64 / 1_000_000.0;
    let count = m.rest_latency_count.load(Ordering::Relaxed);
    body.push_str(&format!(
        "helix_rest_request_duration_seconds_sum {sum_s}\n\
helix_rest_request_duration_seconds_count {count}\n"
    ));

    let mut response = (StatusCode::OK, body).into_response();
    response.headers_mut().insert(
        CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_counters_increment_and_snapshot() {
        let counters = MetricsCounters::default();
        counters.incr_rest_request();
        counters.incr_rest_error();
        counters.incr_grpc_request();
        counters.incr_vault_sealed_http_blocked();
        counters.incr_vault_sealed_grpc_blocked();
        counters.incr_vault_unseal_failure();
        counters.incr_vault_unseal_rate_limited();
        counters.incr_vault_migration_success();
        counters.incr_vault_migration_failure();
        counters.incr_vault_rebuild_success();
        counters.incr_vault_rebuild_failure();

        let snapshot = counters.snapshot();
        assert_eq!(snapshot.rest_requests_total, 1);
        assert_eq!(snapshot.rest_errors_total, 1);
        assert_eq!(snapshot.grpc_requests_total, 1);
        assert_eq!(snapshot.grpc_errors_total, 0);
        assert_eq!(snapshot.vault_sealed_http_requests_blocked_total, 1);
        assert_eq!(snapshot.vault_sealed_grpc_requests_blocked_total, 1);
        assert_eq!(snapshot.vault_unseal_failures_total, 1);
        assert_eq!(snapshot.vault_unseal_rate_limited_total, 1);
        assert_eq!(snapshot.vault_sealed_migration_success_total, 1);
        assert_eq!(snapshot.vault_sealed_migration_failures_total, 1);
        assert_eq!(snapshot.vault_runtime_rebuild_success_total, 1);
        assert_eq!(snapshot.vault_runtime_rebuild_failures_total, 1);
    }

    #[test]
    fn latency_histogram_buckets() {
        let counters = MetricsCounters::default();
        // 50ms = 50_000us — should land in the 50ms, 100ms, 250ms, ... buckets
        counters.observe_rest_latency_us(50_000);

        // Buckets: 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000
        // 50ms should NOT be in 5ms, 10ms, 25ms buckets
        assert_eq!(counters.rest_latency_buckets[0].load(Ordering::Relaxed), 0); // 5ms
        assert_eq!(counters.rest_latency_buckets[1].load(Ordering::Relaxed), 0); // 10ms
        assert_eq!(counters.rest_latency_buckets[2].load(Ordering::Relaxed), 0); // 25ms
                                                                                 // 50ms should be in 50ms bucket and above
        assert_eq!(counters.rest_latency_buckets[3].load(Ordering::Relaxed), 1); // 50ms
        assert_eq!(counters.rest_latency_buckets[4].load(Ordering::Relaxed), 1); // 100ms
                                                                                 // +Inf always gets it
        assert_eq!(
            counters
                .rest_latency_buckets
                .last()
                .unwrap()
                .load(Ordering::Relaxed),
            1
        );
        assert_eq!(counters.rest_latency_count.load(Ordering::Relaxed), 1);
        assert_eq!(counters.rest_latency_sum_us.load(Ordering::Relaxed), 50_000);
    }
}
