//! Adaptive confidence threshold for CNN signal gating.
//!
//! Instead of a fixed `min_confidence`, the threshold drifts with recent trade
//! outcomes: a win lowers it (trust the model more), a loss raises it (demand
//! more confidence). The value is clamped to
//! `[base*(1-max_adjustment), base*(1+max_adjustment)]`. Faithful port of Ruby's
//! `AdaptiveThreshold` (`ruby/src/services/adaptive_threshold.py`).
//!
//! The Python version persists the per-asset threshold in Redis; this port keeps
//! it in-memory. A Redis-backed adapter for cross-restart durability is a
//! follow-up (see the janus TODO). As in the Python code, `target_win_rate` is
//! not used by the simple ±learning-rate update and is therefore omitted here.

use std::collections::HashMap;

/// Default starting / center threshold.
pub const DEFAULT_BASE: f64 = 0.60;
/// Default per-outcome adjustment step.
pub const DEFAULT_LEARNING_RATE: f64 = 0.02;
/// Default maximum deviation from base, as a fraction.
pub const DEFAULT_MAX_ADJUSTMENT: f64 = 0.25;

/// Per-asset adaptive confidence threshold.
#[derive(Debug, Clone)]
pub struct AdaptiveThreshold {
    base: f64,
    learning_rate: f64,
    min_thresh: f64,
    max_thresh: f64,
    cached: HashMap<String, f64>,
}

impl Default for AdaptiveThreshold {
    fn default() -> Self {
        Self::new(DEFAULT_BASE, DEFAULT_LEARNING_RATE, DEFAULT_MAX_ADJUSTMENT)
    }
}

impl AdaptiveThreshold {
    /// Create a threshold with the given base, learning rate, and max deviation.
    pub fn new(base_threshold: f64, learning_rate: f64, max_adjustment: f64) -> Self {
        Self {
            base: base_threshold,
            learning_rate,
            min_thresh: base_threshold * (1.0 - max_adjustment),
            max_thresh: base_threshold * (1.0 + max_adjustment),
            cached: HashMap::new(),
        }
    }

    pub fn base(&self) -> f64 {
        self.base
    }

    /// `(min, max)` bounds the threshold is clamped to.
    pub fn bounds(&self) -> (f64, f64) {
        (self.min_thresh, self.max_thresh)
    }

    /// Current threshold for an asset (the base value if no outcome recorded yet).
    pub fn get_threshold(&self, asset: &str) -> f64 {
        self.cached.get(asset).copied().unwrap_or(self.base)
    }

    /// Record a trade outcome and adjust the threshold. Returns the new value.
    ///
    /// * Win — decrease by `learning_rate` (be more aggressive).
    /// * Loss — increase by `learning_rate` (be more conservative).
    ///
    /// The result is clamped to [`bounds`](Self::bounds).
    pub fn record_outcome(&mut self, asset: &str, is_win: bool) -> f64 {
        let current = self.get_threshold(asset);
        let delta = if is_win {
            -self.learning_rate
        } else {
            self.learning_rate
        };
        let new_thresh = (current + delta).clamp(self.min_thresh, self.max_thresh);
        self.cached.insert(asset.to_string(), new_thresh);
        new_thresh
    }

    /// Reset an asset's threshold to base (e.g. after retraining).
    pub fn reset(&mut self, asset: &str) {
        self.cached.insert(asset.to_string(), self.base);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_at_base() {
        let at = AdaptiveThreshold::default();
        assert!((at.get_threshold("btc") - 0.60).abs() < 1e-12);
    }

    #[test]
    fn bounds_are_base_scaled() {
        let at = AdaptiveThreshold::new(0.60, 0.02, 0.25);
        let (lo, hi) = at.bounds();
        assert!((lo - 0.45).abs() < 1e-12);
        assert!((hi - 0.75).abs() < 1e-12);
    }

    #[test]
    fn win_lowers_loss_raises() {
        let mut at = AdaptiveThreshold::new(0.60, 0.02, 0.25);
        let after_win = at.record_outcome("btc", true);
        assert!(
            (after_win - 0.58).abs() < 1e-12,
            "0.60 - 0.02 = 0.58, got {after_win}"
        );
        let after_loss = at.record_outcome("btc", false);
        assert!(
            (after_loss - 0.60).abs() < 1e-12,
            "0.58 + 0.02 = 0.60, got {after_loss}"
        );
    }

    #[test]
    fn clamped_to_bounds() {
        let mut at = AdaptiveThreshold::new(0.60, 0.02, 0.25);
        // 100 wins would drive far below the floor without clamping.
        for _ in 0..100 {
            at.record_outcome("btc", true);
        }
        assert!((at.get_threshold("btc") - 0.45).abs() < 1e-12);
        for _ in 0..200 {
            at.record_outcome("btc", false);
        }
        assert!((at.get_threshold("btc") - 0.75).abs() < 1e-12);
    }

    #[test]
    fn reset_restores_base() {
        let mut at = AdaptiveThreshold::default();
        at.record_outcome("btc", false);
        at.reset("btc");
        assert!((at.get_threshold("btc") - 0.60).abs() < 1e-12);
    }
}
