//! Vector clock for tracking causal ordering across devices.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A vector clock tracks causal ordering of events across multiple devices.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorClock {
    pub clocks: HashMap<String, u64>,
}

impl VectorClock {
    pub fn new(device_id: &str) -> Self {
        let mut clocks = HashMap::new();
        clocks.insert(device_id.to_string(), 1);
        Self { clocks }
    }

    /// Increment this device's counter.
    pub fn tick(&mut self, device_id: &str) {
        let counter = self.clocks.entry(device_id.to_string()).or_insert(0);
        *counter += 1;
    }

    /// Merge with another vector clock (take max of each).
    pub fn merge(&mut self, other: &VectorClock) {
        for (device, &count) in &other.clocks {
            let local = self.clocks.entry(device.clone()).or_insert(0);
            *local = (*local).max(count);
        }
    }

    /// Check if this clock is causally before another.
    pub fn happens_before(&self, other: &VectorClock) -> bool {
        let mut at_least_one_less = false;

        for (device, &count) in &other.clocks {
            let local = self.clocks.get(device).copied().unwrap_or(0);
            if local > count {
                return false;
            }
            if local < count {
                at_least_one_less = true;
            }
        }

        // Check devices in self but not in other
        for (device, &count) in &self.clocks {
            if !other.clocks.contains_key(device) && count > 0 {
                return false;
            }
        }

        at_least_one_less
    }

    /// Check if two clocks are concurrent (neither happens before the other).
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happens_before(other) && !other.happens_before(self) && self.clocks != other.clocks
    }

    /// Get the counter for a specific device.
    pub fn get(&self, device_id: &str) -> u64 {
        self.clocks.get(device_id).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_clock() {
        let clock = VectorClock::new("device_a");
        assert_eq!(clock.get("device_a"), 1);
        assert_eq!(clock.get("device_b"), 0);
    }

    #[test]
    fn test_tick() {
        let mut clock = VectorClock::new("device_a");
        clock.tick("device_a");
        assert_eq!(clock.get("device_a"), 2);
    }

    #[test]
    fn test_merge() {
        let mut a = VectorClock::new("device_a");
        a.tick("device_a"); // a:2

        let mut b = VectorClock::new("device_b");
        b.tick("device_b"); // b:2

        a.merge(&b);
        assert_eq!(a.get("device_a"), 2);
        assert_eq!(a.get("device_b"), 2);
    }

    #[test]
    fn test_happens_before() {
        let a = VectorClock::new("device_a"); // a:1
        let mut b = VectorClock::new("device_a"); // a:1
        b.tick("device_a"); // a:2

        assert!(a.happens_before(&b));
        assert!(!b.happens_before(&a));
    }

    #[test]
    fn test_concurrent() {
        let a = VectorClock::new("device_a"); // a:1
        let b = VectorClock::new("device_b"); // b:1

        assert!(a.is_concurrent(&b));
    }

    #[test]
    fn test_identical_clocks_not_concurrent() {
        let a = VectorClock::new("device_a");
        let b = VectorClock::new("device_a");
        assert!(!a.is_concurrent(&b));
    }

    #[test]
    fn test_default_is_empty() {
        let clock = VectorClock::default();
        assert_eq!(clock.get("any"), 0);
        assert!(clock.clocks.is_empty());
    }
}
