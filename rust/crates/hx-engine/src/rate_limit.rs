//! Rate limiting for adapters and API endpoints.
//!
//! Token-bucket style rate limiter with per-minute, per-hour, and burst limits.

use std::collections::HashMap;

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;

/// Configuration for rate limiting an adapter or endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub requests_per_hour: u32,
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            requests_per_hour: 1000,
            burst_size: 10,
        }
    }
}

/// Error returned when a rate limit is exceeded.
#[derive(Debug)]
pub struct RateLimitExceeded {
    pub retry_after_secs: u64,
}

impl std::fmt::Display for RateLimitExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "rate limit exceeded, retry after {} seconds",
            self.retry_after_secs
        )
    }
}

impl std::error::Error for RateLimitExceeded {}

#[derive(Debug)]
struct RateLimitBucket {
    minute_count: u32,
    hour_count: u32,
    minute_reset: DateTime<Utc>,
    hour_reset: DateTime<Utc>,
}

impl RateLimitBucket {
    fn new() -> Self {
        let now = Utc::now();
        Self {
            minute_count: 0,
            hour_count: 0,
            minute_reset: now + Duration::minutes(1),
            hour_reset: now + Duration::hours(1),
        }
    }

    fn maybe_reset(&mut self, now: DateTime<Utc>) {
        if now >= self.minute_reset {
            self.minute_count = 0;
            self.minute_reset = now + Duration::minutes(1);
        }
        if now >= self.hour_reset {
            self.hour_count = 0;
            self.hour_reset = now + Duration::hours(1);
        }
    }
}

/// Per-adapter rate limiter.
pub struct RateLimiter {
    configs: RwLock<HashMap<Uuid, RateLimitConfig>>,
    buckets: RwLock<HashMap<Uuid, RateLimitBucket>>,
}

impl RateLimiter {
    pub fn new() -> Self {
        Self {
            configs: RwLock::new(HashMap::new()),
            buckets: RwLock::new(HashMap::new()),
        }
    }

    /// Set the rate limit configuration for an adapter.
    pub async fn set_limit(&self, adapter_id: Uuid, config: RateLimitConfig) {
        self.configs.write().await.insert(adapter_id, config);
    }

    /// Get the rate limit configuration for an adapter.
    pub async fn get_limit(&self, adapter_id: Uuid) -> Option<RateLimitConfig> {
        self.configs.read().await.get(&adapter_id).cloned()
    }

    /// Remove the rate limit configuration for an adapter.
    pub async fn remove_limit(&self, adapter_id: Uuid) {
        self.configs.write().await.remove(&adapter_id);
        self.buckets.write().await.remove(&adapter_id);
    }

    /// Check if a request is allowed under the rate limit.
    /// Returns `Ok(())` if allowed, `Err(RateLimitExceeded)` if not.
    pub async fn check(&self, adapter_id: Uuid) -> Result<(), RateLimitExceeded> {
        let configs = self.configs.read().await;
        let config = match configs.get(&adapter_id) {
            Some(c) => c.clone(),
            None => return Ok(()), // No limit configured = always allowed
        };
        drop(configs);

        let now = Utc::now();
        let mut buckets = self.buckets.write().await;
        let bucket = buckets.entry(adapter_id).or_insert_with(RateLimitBucket::new);
        bucket.maybe_reset(now);

        if bucket.minute_count >= config.requests_per_minute {
            let retry_after = (bucket.minute_reset - now).num_seconds().max(1) as u64;
            return Err(RateLimitExceeded {
                retry_after_secs: retry_after,
            });
        }

        if bucket.hour_count >= config.requests_per_hour {
            let retry_after = (bucket.hour_reset - now).num_seconds().max(1) as u64;
            return Err(RateLimitExceeded {
                retry_after_secs: retry_after,
            });
        }

        Ok(())
    }

    /// Record a request for the given adapter.
    pub async fn record(&self, adapter_id: Uuid) {
        let now = Utc::now();
        let mut buckets = self.buckets.write().await;
        let bucket = buckets.entry(adapter_id).or_insert_with(RateLimitBucket::new);
        bucket.maybe_reset(now);
        bucket.minute_count += 1;
        bucket.hour_count += 1;
    }

    /// List all configured rate limits.
    pub async fn list_limits(&self) -> Vec<(Uuid, RateLimitConfig)> {
        self.configs
            .read()
            .await
            .iter()
            .map(|(id, config)| (*id, config.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
        let config = RateLimitConfig::default();
        assert_eq!(config.requests_per_minute, 60);
        assert_eq!(config.requests_per_hour, 1000);
        assert_eq!(config.burst_size, 10);
    }

    #[test]
    fn config_serializes() {
        let config = RateLimitConfig {
            requests_per_minute: 30,
            requests_per_hour: 500,
            burst_size: 5,
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("30"));
        assert!(json.contains("500"));
    }

    #[test]
    fn config_deserializes() {
        let json = r#"{"requests_per_minute":10,"requests_per_hour":100,"burst_size":3}"#;
        let config: RateLimitConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.requests_per_minute, 10);
        assert_eq!(config.requests_per_hour, 100);
        assert_eq!(config.burst_size, 3);
    }

    #[test]
    fn rate_limit_exceeded_display() {
        let err = RateLimitExceeded {
            retry_after_secs: 42,
        };
        assert_eq!(
            err.to_string(),
            "rate limit exceeded, retry after 42 seconds"
        );
    }

    #[tokio::test]
    async fn no_config_always_allows() {
        let limiter = RateLimiter::new();
        let id = Uuid::now_v7();
        assert!(limiter.check(id).await.is_ok());
    }

    #[tokio::test]
    async fn set_and_get_limit() {
        let limiter = RateLimiter::new();
        let id = Uuid::now_v7();
        let config = RateLimitConfig {
            requests_per_minute: 5,
            requests_per_hour: 50,
            burst_size: 2,
        };
        limiter.set_limit(id, config.clone()).await;

        let fetched = limiter.get_limit(id).await;
        assert!(fetched.is_some());
        assert_eq!(fetched.unwrap().requests_per_minute, 5);
    }

    #[tokio::test]
    async fn remove_limit() {
        let limiter = RateLimiter::new();
        let id = Uuid::now_v7();
        limiter
            .set_limit(id, RateLimitConfig::default())
            .await;
        limiter.remove_limit(id).await;
        assert!(limiter.get_limit(id).await.is_none());
    }

    #[tokio::test]
    async fn record_increments_counters() {
        let limiter = RateLimiter::new();
        let id = Uuid::now_v7();
        limiter
            .set_limit(
                id,
                RateLimitConfig {
                    requests_per_minute: 3,
                    requests_per_hour: 100,
                    burst_size: 1,
                },
            )
            .await;

        // First three should be ok
        for _ in 0..3 {
            assert!(limiter.check(id).await.is_ok());
            limiter.record(id).await;
        }

        // Fourth should be rejected (minute limit = 3)
        let result = limiter.check(id).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.retry_after_secs > 0);
    }

    #[tokio::test]
    async fn hour_limit_enforcement() {
        let limiter = RateLimiter::new();
        let id = Uuid::now_v7();
        limiter
            .set_limit(
                id,
                RateLimitConfig {
                    requests_per_minute: 1000, // high minute limit
                    requests_per_hour: 2,      // low hour limit
                    burst_size: 1,
                },
            )
            .await;

        limiter.record(id).await;
        limiter.record(id).await;
        assert!(limiter.check(id).await.is_err());
    }

    #[tokio::test]
    async fn list_limits_returns_all() {
        let limiter = RateLimiter::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        limiter
            .set_limit(id1, RateLimitConfig::default())
            .await;
        limiter
            .set_limit(id2, RateLimitConfig::default())
            .await;

        let limits = limiter.list_limits().await;
        assert_eq!(limits.len(), 2);
    }

    #[tokio::test]
    async fn bucket_resets_after_window() {
        let limiter = RateLimiter::new();
        let id = Uuid::now_v7();
        limiter
            .set_limit(
                id,
                RateLimitConfig {
                    requests_per_minute: 1,
                    requests_per_hour: 100,
                    burst_size: 1,
                },
            )
            .await;

        limiter.record(id).await;
        assert!(limiter.check(id).await.is_err());

        // Manually reset the minute window
        {
            let mut buckets = limiter.buckets.write().await;
            if let Some(bucket) = buckets.get_mut(&id) {
                bucket.minute_reset = Utc::now() - Duration::seconds(1);
                bucket.minute_count = 999; // stale count
            }
        }

        // Now check should pass because reset triggers
        assert!(limiter.check(id).await.is_ok());
    }

    #[tokio::test]
    async fn different_adapters_independent() {
        let limiter = RateLimiter::new();
        let id1 = Uuid::now_v7();
        let id2 = Uuid::now_v7();
        limiter
            .set_limit(
                id1,
                RateLimitConfig {
                    requests_per_minute: 1,
                    requests_per_hour: 100,
                    burst_size: 1,
                },
            )
            .await;
        limiter
            .set_limit(
                id2,
                RateLimitConfig {
                    requests_per_minute: 1,
                    requests_per_hour: 100,
                    burst_size: 1,
                },
            )
            .await;

        limiter.record(id1).await;
        assert!(limiter.check(id1).await.is_err());
        // id2 should still be allowed
        assert!(limiter.check(id2).await.is_ok());
    }
}
