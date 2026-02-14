use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Normal operation — requests pass through.
    Closed,
    /// Too many failures — requests are rejected immediately.
    Open,
    /// Trial period — a limited number of requests are allowed through.
    HalfOpen,
}

/// Configuration for a circuit breaker instance.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening the circuit.
    pub failure_threshold: u32,
    /// How long the circuit stays open before transitioning to half-open.
    pub reset_timeout: Duration,
    /// Maximum number of trial requests allowed in half-open state.
    pub half_open_max_attempts: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(60),
            half_open_max_attempts: 1,
        }
    }
}

struct Inner {
    state: State,
    failure_count: u32,
    last_failure_time: Option<Instant>,
    half_open_attempts: u32,
}

/// A thread-safe circuit breaker for protecting downstream service calls.
///
/// Usage:
/// ```ignore
/// let cb = CircuitBreaker::new(CircuitBreakerConfig::default());
/// if let Err(e) = cb.check() {
///     return Err(e); // Circuit is open, fast-fail
/// }
/// match do_request().await {
///     Ok(val) => { cb.record_success(); Ok(val) }
///     Err(e) => { cb.record_failure(); Err(e) }
/// }
/// ```
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    inner: Mutex<Inner>,
}

#[derive(Debug, Clone)]
pub struct CircuitOpen {
    pub retry_after: Duration,
}

impl std::fmt::Display for CircuitOpen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "circuit breaker open, retry after {}s",
            self.retry_after.as_secs()
        )
    }
}

impl std::error::Error for CircuitOpen {}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            inner: Mutex::new(Inner {
                state: State::Closed,
                failure_count: 0,
                last_failure_time: None,
                half_open_attempts: 0,
            }),
        }
    }

    /// Returns the current state of the circuit breaker.
    pub fn state(&self) -> State {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        self.maybe_transition_to_half_open(&mut inner);
        inner.state
    }

    /// Check if a request is allowed through. Returns `Err(CircuitOpen)` if
    /// the circuit is open and requests should be rejected.
    pub fn check(&self) -> Result<(), CircuitOpen> {
        self.check_at(Instant::now())
    }

    fn check_at(&self, now: Instant) -> Result<(), CircuitOpen> {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        self.maybe_transition_to_half_open_at(&mut inner, now);

        match inner.state {
            State::Closed => Ok(()),
            State::Open => {
                let retry_after = inner
                    .last_failure_time
                    .map(|t| self.config.reset_timeout.saturating_sub(now.duration_since(t)))
                    .unwrap_or(self.config.reset_timeout);
                Err(CircuitOpen { retry_after })
            }
            State::HalfOpen => {
                if inner.half_open_attempts < self.config.half_open_max_attempts {
                    inner.half_open_attempts += 1;
                    Ok(())
                } else {
                    let retry_after = inner
                        .last_failure_time
                        .map(|t| {
                            self.config.reset_timeout.saturating_sub(now.duration_since(t))
                        })
                        .unwrap_or(self.config.reset_timeout);
                    Err(CircuitOpen { retry_after })
                }
            }
        }
    }

    /// Record a successful request. Resets the circuit to closed.
    pub fn record_success(&self) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.state = State::Closed;
        inner.failure_count = 0;
        inner.half_open_attempts = 0;
    }

    /// Record a failed request. May transition the circuit to open.
    pub fn record_failure(&self) {
        self.record_failure_at(Instant::now());
    }

    fn record_failure_at(&self, now: Instant) {
        let mut inner = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        inner.failure_count += 1;
        inner.last_failure_time = Some(now);

        if inner.state == State::HalfOpen {
            // Any failure in half-open immediately re-opens
            inner.state = State::Open;
            inner.half_open_attempts = 0;
        } else if inner.failure_count >= self.config.failure_threshold {
            inner.state = State::Open;
        }
    }

    fn maybe_transition_to_half_open(&self, inner: &mut Inner) {
        self.maybe_transition_to_half_open_at(inner, Instant::now());
    }

    fn maybe_transition_to_half_open_at(&self, inner: &mut Inner, now: Instant) {
        if inner.state == State::Open {
            if let Some(last) = inner.last_failure_time {
                if now.duration_since(last) >= self.config.reset_timeout {
                    inner.state = State::HalfOpen;
                    inner.half_open_attempts = 0;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> CircuitBreakerConfig {
        CircuitBreakerConfig {
            failure_threshold: 3,
            reset_timeout: Duration::from_secs(10),
            half_open_max_attempts: 1,
        }
    }

    #[test]
    fn starts_closed() {
        let cb = CircuitBreaker::new(test_config());
        assert_eq!(cb.state(), State::Closed);
        assert!(cb.check().is_ok());
    }

    #[test]
    fn opens_after_failure_threshold() {
        let cb = CircuitBreaker::new(test_config());
        let now = Instant::now();

        cb.record_failure_at(now);
        cb.record_failure_at(now + Duration::from_secs(1));
        assert_eq!(cb.state(), State::Closed); // 2 failures, threshold is 3

        cb.record_failure_at(now + Duration::from_secs(2));
        assert_eq!(cb.state(), State::Open); // 3 failures → open
        assert!(cb.check_at(now + Duration::from_secs(3)).is_err());
    }

    #[test]
    fn transitions_to_half_open_after_timeout() {
        let cb = CircuitBreaker::new(test_config());
        let now = Instant::now();

        // Trip the breaker
        for i in 0..3 {
            cb.record_failure_at(now + Duration::from_secs(i));
        }
        assert_eq!(cb.state(), State::Open);

        // After reset_timeout, should be half-open
        let after_timeout = now + Duration::from_secs(13); // 2 + 10 = 12, so 13 is past it
        assert!(cb.check_at(after_timeout).is_ok()); // First trial allowed
        // State is now half-open
    }

    #[test]
    fn half_open_success_closes_circuit() {
        let cb = CircuitBreaker::new(test_config());
        let now = Instant::now();

        for i in 0..3 {
            cb.record_failure_at(now + Duration::from_secs(i));
        }

        // Wait for timeout
        let after_timeout = now + Duration::from_secs(13);
        assert!(cb.check_at(after_timeout).is_ok());

        // Successful trial → close
        cb.record_success();
        assert_eq!(cb.state(), State::Closed);
        assert!(cb.check().is_ok());
    }

    #[test]
    fn half_open_failure_reopens_circuit() {
        let cb = CircuitBreaker::new(test_config());
        let now = Instant::now();

        for i in 0..3 {
            cb.record_failure_at(now + Duration::from_secs(i));
        }

        // Wait for timeout
        let after_timeout = now + Duration::from_secs(13);
        assert!(cb.check_at(after_timeout).is_ok()); // trial request

        // Trial fails → back to open
        cb.record_failure_at(after_timeout + Duration::from_secs(1));
        assert_eq!(cb.state(), State::Open);
    }

    #[test]
    fn half_open_rejects_extra_attempts() {
        let cb = CircuitBreaker::new(test_config());
        let now = Instant::now();

        for i in 0..3 {
            cb.record_failure_at(now + Duration::from_secs(i));
        }

        let after_timeout = now + Duration::from_secs(13);
        assert!(cb.check_at(after_timeout).is_ok()); // 1 allowed
        assert!(cb.check_at(after_timeout + Duration::from_millis(1)).is_err()); // 2nd rejected
    }

    #[test]
    fn success_resets_failure_count() {
        let cb = CircuitBreaker::new(test_config());
        let now = Instant::now();

        cb.record_failure_at(now);
        cb.record_failure_at(now + Duration::from_secs(1));
        cb.record_success(); // Reset

        // Need 3 more failures to trip again
        cb.record_failure_at(now + Duration::from_secs(2));
        cb.record_failure_at(now + Duration::from_secs(3));
        assert_eq!(cb.state(), State::Closed);

        cb.record_failure_at(now + Duration::from_secs(4));
        assert_eq!(cb.state(), State::Open);
    }

    #[test]
    fn circuit_open_error_has_retry_after() {
        let cb = CircuitBreaker::new(test_config());
        let now = Instant::now();

        for i in 0..3 {
            cb.record_failure_at(now + Duration::from_secs(i));
        }

        let err = cb.check_at(now + Duration::from_secs(5)).unwrap_err();
        // Last failure at t=2, reset_timeout=10, checked at t=5 → retry_after ≈ 7s
        assert!(err.retry_after.as_secs() >= 6 && err.retry_after.as_secs() <= 8);
    }
}
