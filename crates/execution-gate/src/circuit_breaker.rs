//! Per-asset consecutive-loss circuit breaker.
//!
//! After `loss_limit` consecutive losses on an asset the breaker trips and
//! blocks new entries for `cooldown_secs`. A win resets the counter. Faithful
//! port of Ruby's `ConsecutiveLossBreaker` (`ruby/src/services/circuit_breaker.py`).
//!
//! The Python version persists state in Redis with a TTL key for the cooldown;
//! this port keeps the state in-memory and takes the current unix time
//! (`now_secs`) as an explicit argument, which makes the cooldown logic fully
//! deterministic for tests. A Redis-backed adapter for cross-restart durability
//! is a follow-up (see the janus TODO).

use std::collections::HashMap;

/// Default number of consecutive losses before the breaker trips.
pub const DEFAULT_LOSS_LIMIT: u32 = 3;
/// Default cooldown after the breaker trips, in seconds (15 minutes).
pub const DEFAULT_COOLDOWN_SECS: u64 = 900;

#[derive(Debug, Clone, Default)]
struct BreakerState {
    consecutive_losses: u32,
    /// Unix second at which the cooldown ends (if currently/previously tripped).
    cooldown_until_secs: Option<u64>,
}

/// In-memory consecutive-loss circuit breaker, keyed by asset.
#[derive(Debug, Clone)]
pub struct ConsecutiveLossBreaker {
    loss_limit: u32,
    cooldown_secs: u64,
    state: HashMap<String, BreakerState>,
}

impl Default for ConsecutiveLossBreaker {
    fn default() -> Self {
        Self::new(DEFAULT_LOSS_LIMIT, DEFAULT_COOLDOWN_SECS)
    }
}

impl ConsecutiveLossBreaker {
    /// Create a breaker with the given loss limit and cooldown.
    pub fn new(loss_limit: u32, cooldown_secs: u64) -> Self {
        Self {
            loss_limit,
            cooldown_secs,
            state: HashMap::new(),
        }
    }

    pub fn loss_limit(&self) -> u32 {
        self.loss_limit
    }

    pub fn cooldown_secs(&self) -> u64 {
        self.cooldown_secs
    }

    /// Record a trade outcome.
    ///
    /// * Win — resets the consecutive-loss counter to zero. Any active cooldown
    ///   is left to expire on its own (mirroring the Python Redis-TTL semantics:
    ///   a win clears the counter but does not delete the cooldown key).
    /// * Loss — increments the counter; once it reaches `loss_limit` the breaker
    ///   trips and the cooldown is (re)started at `now_secs + cooldown_secs`.
    pub fn record_outcome(&mut self, asset: &str, is_win: bool, now_secs: u64) {
        let entry = self.state.entry(asset.to_string()).or_default();
        if is_win {
            entry.consecutive_losses = 0;
        } else {
            entry.consecutive_losses += 1;
            if entry.consecutive_losses >= self.loss_limit {
                entry.cooldown_until_secs = Some(now_secs.saturating_add(self.cooldown_secs));
            }
        }
    }

    /// Whether the breaker is currently open (blocking entries) at `now_secs`.
    pub fn is_open(&self, asset: &str, now_secs: u64) -> bool {
        self.state
            .get(asset)
            .and_then(|s| s.cooldown_until_secs)
            .map(|until| now_secs < until)
            .unwrap_or(false)
    }

    /// Seconds until the breaker auto-resets; `0` if not open.
    pub fn cooldown_remaining(&self, asset: &str, now_secs: u64) -> u64 {
        self.state
            .get(asset)
            .and_then(|s| s.cooldown_until_secs)
            .map(|until| until.saturating_sub(now_secs))
            .unwrap_or(0)
    }

    /// Current consecutive-loss count for an asset.
    pub fn loss_count(&self, asset: &str) -> u32 {
        self.state
            .get(asset)
            .map(|s| s.consecutive_losses)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_000_000;

    #[test]
    fn fresh_breaker_is_closed() {
        let cb = ConsecutiveLossBreaker::default();
        assert!(!cb.is_open("btc", T0));
        assert_eq!(cb.loss_count("btc"), 0);
        assert_eq!(cb.cooldown_remaining("btc", T0), 0);
    }

    #[test]
    fn trips_after_loss_limit() {
        let mut cb = ConsecutiveLossBreaker::new(3, 900);
        cb.record_outcome("btc", false, T0);
        cb.record_outcome("btc", false, T0);
        assert!(
            !cb.is_open("btc", T0),
            "two losses must not trip a limit-3 breaker"
        );
        cb.record_outcome("btc", false, T0);
        assert!(cb.is_open("btc", T0), "third loss must trip");
        assert_eq!(cb.loss_count("btc"), 3);
        assert_eq!(cb.cooldown_remaining("btc", T0), 900);
    }

    #[test]
    fn win_resets_counter() {
        let mut cb = ConsecutiveLossBreaker::new(3, 900);
        cb.record_outcome("btc", false, T0);
        cb.record_outcome("btc", false, T0);
        cb.record_outcome("btc", true, T0);
        assert_eq!(cb.loss_count("btc"), 0);
        cb.record_outcome("btc", false, T0);
        assert!(
            !cb.is_open("btc", T0),
            "counter reset, single loss must not trip"
        );
    }

    #[test]
    fn cooldown_expires_with_time() {
        let mut cb = ConsecutiveLossBreaker::new(2, 900);
        cb.record_outcome("eth", false, T0);
        cb.record_outcome("eth", false, T0);
        assert!(cb.is_open("eth", T0));
        assert!(cb.is_open("eth", T0 + 899));
        assert!(
            !cb.is_open("eth", T0 + 900),
            "breaker closes exactly at cooldown end"
        );
        assert_eq!(cb.cooldown_remaining("eth", T0 + 900), 0);
    }

    #[test]
    fn assets_are_independent() {
        let mut cb = ConsecutiveLossBreaker::new(2, 900);
        cb.record_outcome("btc", false, T0);
        cb.record_outcome("btc", false, T0);
        assert!(cb.is_open("btc", T0));
        assert!(!cb.is_open("eth", T0));
    }
}
