//! # Cross-Asset Correlation Tracker
//!
//! Monitors rolling return correlations between asset pairs to prevent
//! over-concentration in correlated positions. The tracker maintains a
//! sliding window of log-returns for each asset and computes pairwise
//! Pearson correlation coefficients on demand.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_risk::correlation::{CorrelationTracker, CorrelationConfig};
//!
//! let mut tracker = CorrelationTracker::new(CorrelationConfig::default());
//!
//! // Feed price updates
//! tracker.update("BTCUSD", 67_000.0);
//! tracker.update("ETHUSD", 3_800.0);
//! // ... more updates ...
//!
//! // Query correlation
//! if let Some(corr) = tracker.correlation("BTCUSD", "ETHUSD") {
//!     println!("BTC-ETH correlation: {:.2}", corr);
//! }
//!
//! // Check before opening a new position
//! let current_positions = vec!["BTCUSD".to_string(), "ETHUSD".to_string()];
//! if tracker.would_exceed_correlation_limit("SOLUSD", &current_positions) {
//!     println!("Too many correlated positions — blocking trade");
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::Instant;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the correlation tracker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Number of return observations to keep per asset (rolling window).
    /// More observations → smoother estimate but slower to adapt.
    #[serde(default = "default_window")]
    pub window: usize,

    /// Pearson correlation magnitude above which two assets are considered
    /// "highly correlated" (absolute value, so 0.75 catches both +0.75 and −0.75).
    #[serde(default = "default_correlation_threshold")]
    pub correlation_threshold: f64,

    /// Maximum number of open positions whose assets are mutually correlated
    /// above `correlation_threshold`.
    #[serde(default = "default_max_correlated_positions")]
    pub max_correlated_positions: usize,

    /// Minimum number of overlapping return observations required before
    /// a correlation estimate is considered valid.
    #[serde(default = "default_min_observations")]
    pub min_observations: usize,

    /// Optional list of asset pairs to monitor explicitly.
    /// If empty, the tracker discovers pairs automatically from updates.
    #[serde(default)]
    pub monitored_pairs: Vec<(String, String)>,
}

fn default_window() -> usize {
    100
}
fn default_correlation_threshold() -> f64 {
    0.75
}
fn default_max_correlated_positions() -> usize {
    3
}
fn default_min_observations() -> usize {
    20
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            window: default_window(),
            correlation_threshold: default_correlation_threshold(),
            max_correlated_positions: default_max_correlated_positions(),
            min_observations: default_min_observations(),
            monitored_pairs: Vec::new(),
        }
    }
}

// ============================================================================
// Core Tracker
// ============================================================================

/// Tracks rolling return correlations between asset pairs.
#[derive(Debug, Clone)]
pub struct CorrelationTracker {
    /// Per-asset rolling log-return windows.
    returns: HashMap<String, VecDeque<f64>>,

    /// Last observed price per asset (needed to compute returns).
    last_prices: HashMap<String, f64>,

    /// Cached correlation matrix entries. Key is the canonical pair
    /// `(min(a,b), max(a,b))` so each pair is stored once.
    matrix: HashMap<(String, String), f64>,

    /// Rolling window size (number of return observations to keep).
    window: usize,

    /// Tracker configuration.
    config: CorrelationConfig,

    /// Timestamp of last matrix recalculation.
    last_update: Option<Instant>,
}

impl CorrelationTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: CorrelationConfig) -> Self {
        let window = config.window;
        Self {
            returns: HashMap::new(),
            last_prices: HashMap::new(),
            matrix: HashMap::new(),
            window,
            config,
            last_update: None,
        }
    }

    /// Create a tracker with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CorrelationConfig::default())
    }

    /// Get a reference to the current configuration.
    pub fn config(&self) -> &CorrelationConfig {
        &self.config
    }

    // ────────────────────────────────────────────────────────────────────
    // Price Updates
    // ────────────────────────────────────────────────────────────────────

    /// Feed a new price observation for `asset`.
    ///
    /// The first price for an asset establishes a baseline; the second and
    /// subsequent prices produce log-returns that are appended to the
    /// rolling window. After each update the affected correlation pairs
    /// are recalculated.
    pub fn update(&mut self, asset: &str, price: f64) {
        if price <= 0.0 {
            return; // guard against bad data
        }

        if let Some(&prev_price) = self.last_prices.get(asset) {
            // Compute log-return: ln(P_t / P_{t-1})
            let log_return = (price / prev_price).ln();

            let returns = self
                .returns
                .entry(asset.to_string())
                .or_insert_with(|| VecDeque::with_capacity(self.window + 1));

            returns.push_back(log_return);
            if returns.len() > self.window {
                returns.pop_front();
            }
        }

        self.last_prices.insert(asset.to_string(), price);

        // Recalculate correlations for all pairs involving this asset.
        self.recalculate_for_asset(asset);
        self.last_update = Some(Instant::now());
    }

    /// Feed multiple price updates at once (e.g. from a single tick batch).
    pub fn update_batch(&mut self, prices: &[(&str, f64)]) {
        for &(asset, price) in prices {
            // Inline the price → return conversion without recalculating
            // the matrix on every single price.
            if price <= 0.0 {
                continue;
            }
            if let Some(&prev_price) = self.last_prices.get(asset) {
                let log_return = (price / prev_price).ln();
                let returns = self
                    .returns
                    .entry(asset.to_string())
                    .or_insert_with(|| VecDeque::with_capacity(self.window + 1));
                returns.push_back(log_return);
                if returns.len() > self.window {
                    returns.pop_front();
                }
            }
            self.last_prices.insert(asset.to_string(), price);
        }
        // Recalculate the full matrix once.
        self.recalculate_all();
        self.last_update = Some(Instant::now());
    }

    // ────────────────────────────────────────────────────────────────────
    // Queries
    // ────────────────────────────────────────────────────────────────────

    /// Get the Pearson correlation between two assets.
    ///
    /// Returns `None` if there aren't enough overlapping observations yet
    /// (below `config.min_observations`).
    pub fn correlation(&self, asset_a: &str, asset_b: &str) -> Option<f64> {
        if asset_a == asset_b {
            return Some(1.0);
        }
        let key = canonical_pair(asset_a, asset_b);
        self.matrix.get(&key).copied()
    }

    /// Return all pairs whose absolute correlation exceeds
    /// `config.correlation_threshold`.
    pub fn highly_correlated_pairs(&self) -> Vec<(&str, &str, f64)> {
        let threshold = self.config.correlation_threshold;
        self.matrix
            .iter()
            .filter(|(_, corr)| corr.abs() > threshold)
            .map(|((a, b), corr)| (a.as_str(), b.as_str(), *corr))
            .collect()
    }

    /// Check whether opening a position on `new_asset` would violate the
    /// maximum correlated positions limit, given a list of currently held
    /// position symbols.
    ///
    /// Counts how many of `current_positions` are highly correlated with
    /// `new_asset` and returns `true` if adding another would exceed
    /// `config.max_correlated_positions`.
    pub fn would_exceed_correlation_limit(
        &self,
        new_asset: &str,
        current_positions: &[String],
    ) -> bool {
        let threshold = self.config.correlation_threshold;
        let max = self.config.max_correlated_positions;

        let correlated_count = current_positions
            .iter()
            .filter(|pos| {
                if pos.as_str() == new_asset {
                    return false; // same asset isn't "correlated" in this sense
                }
                self.correlation(pos, new_asset)
                    .is_some_and(|c| c.abs() > threshold)
            })
            .count();

        // Adding the new position means we'd have correlated_count + 1
        // correlated positions (including the new one itself).
        correlated_count + 1 > max
    }

    /// Discover pairs with unexpectedly high correlation that are NOT in
    /// the `monitored_pairs` list. Useful for alerting on hidden risks.
    pub fn discover_hidden_correlations(&self) -> Vec<(&str, &str, f64)> {
        let monitored: HashSet<(String, String)> = self
            .config
            .monitored_pairs
            .iter()
            .map(|(a, b)| canonical_pair(a, b))
            .collect();

        let threshold = self.config.correlation_threshold;

        self.matrix
            .iter()
            .filter(|(pair, corr)| corr.abs() > threshold && !monitored.contains(pair))
            .map(|((a, b), corr)| (a.as_str(), b.as_str(), *corr))
            .collect()
    }

    /// Return all known asset symbols that have at least one return observation.
    pub fn tracked_assets(&self) -> Vec<&str> {
        self.returns.keys().map(String::as_str).collect()
    }

    /// Return the full correlation matrix as a list of `(asset_a, asset_b, corr)`.
    pub fn full_matrix(&self) -> Vec<(&str, &str, f64)> {
        self.matrix
            .iter()
            .map(|((a, b), &c)| (a.as_str(), b.as_str(), c))
            .collect()
    }

    /// Number of return observations currently stored for an asset.
    pub fn observation_count(&self, asset: &str) -> usize {
        self.returns.get(asset).map_or(0, VecDeque::len)
    }

    // ────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ────────────────────────────────────────────────────────────────────

    /// Recalculate correlation for all pairs involving `asset`.
    fn recalculate_for_asset(&mut self, asset: &str) {
        let assets: Vec<String> = self.returns.keys().cloned().collect();
        for other in &assets {
            if other == asset {
                continue;
            }
            if let Some(corr) = self.compute_pearson(asset, other) {
                let key = canonical_pair(asset, other);
                self.matrix.insert(key, corr);
            }
        }
    }

    /// Recalculate the entire correlation matrix.
    fn recalculate_all(&mut self) {
        let assets: Vec<String> = self.returns.keys().cloned().collect();
        for i in 0..assets.len() {
            for j in (i + 1)..assets.len() {
                if let Some(corr) = self.compute_pearson(&assets[i], &assets[j]) {
                    let key = canonical_pair(&assets[i], &assets[j]);
                    self.matrix.insert(key, corr);
                }
            }
        }
    }

    /// Compute the Pearson correlation coefficient between two assets'
    /// return series. Returns `None` if there aren't enough overlapping
    /// observations.
    fn compute_pearson(&self, a: &str, b: &str) -> Option<f64> {
        let rc = self.returns.get(a)?;
        let rb = self.returns.get(b)?;

        // Use the most recent overlapping observations.
        let n = rc.len().min(rb.len());
        if n < self.config.min_observations {
            return None;
        }

        // Tail-aligned: take the last `n` entries from each.
        let offset_a = rc.len() - n;
        let offset_b = rb.len() - n;

        let mean_a: f64 = rc.iter().skip(offset_a).sum::<f64>() / n as f64;
        let mean_b: f64 = rb.iter().skip(offset_b).sum::<f64>() / n as f64;

        let mut cov = 0.0_f64;
        let mut var_a = 0.0_f64;
        let mut var_b = 0.0_f64;

        for i in 0..n {
            let da = rc[offset_a + i] - mean_a;
            let db = rb[offset_b + i] - mean_b;
            cov += da * db;
            var_a += da * da;
            var_b += db * db;
        }

        let denom = (var_a * var_b).sqrt();
        if denom < f64::EPSILON {
            return None; // no variance → correlation undefined
        }

        Some((cov / denom).clamp(-1.0, 1.0))
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Canonical pair key: alphabetically sorted so (B, A) == (A, B).
fn canonical_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// ============================================================================
// Error type for risk-manager integration
// ============================================================================

/// Errors that can arise from correlation limit checks.
#[derive(Debug, Clone, thiserror::Error)]
pub enum CorrelationError {
    #[error(
        "too many correlated positions: {new_asset} is correlated with {correlated_count} \
         existing positions (max {max})"
    )]
    TooManyCorrelatedPositions {
        new_asset: String,
        correlated_count: usize,
        max: usize,
    },
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: generate a synthetic price series with a known drift.
    #[allow(dead_code)]
    fn price_series(start: f64, drift: f64, noise: &[f64]) -> Vec<f64> {
        let mut prices = Vec::with_capacity(noise.len() + 1);
        prices.push(start);
        for (i, &n) in noise.iter().enumerate() {
            let prev = prices[i];
            prices.push(prev * (1.0 + drift + n));
        }
        prices
    }

    #[test]
    fn test_new_tracker() {
        let tracker = CorrelationTracker::with_defaults();
        assert_eq!(tracker.window, 100);
        assert!(tracker.returns.is_empty());
        assert!(tracker.matrix.is_empty());
    }

    #[test]
    fn test_single_asset_no_correlation() {
        let mut tracker = CorrelationTracker::with_defaults();
        tracker.update("BTCUSD", 67_000.0);
        tracker.update("BTCUSD", 67_500.0);

        // Self-correlation is always 1
        assert_eq!(tracker.correlation("BTCUSD", "BTCUSD"), Some(1.0));

        // No pair exists yet
        assert_eq!(tracker.correlation("BTCUSD", "ETHUSD"), None);
    }

    #[test]
    fn test_perfectly_correlated_series() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 5,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // Two assets that move in exactly the same direction.
        for i in 0..30 {
            let price = 100.0 + i as f64 * 2.0;
            tracker.update("A", price);
            tracker.update("B", price * 2.0); // perfectly correlated (same returns)
        }

        let corr = tracker
            .correlation("A", "B")
            .expect("should have enough data");
        assert!(
            (corr - 1.0).abs() < 0.01,
            "Perfectly correlated assets should have corr ≈ 1.0, got {corr}"
        );
    }

    #[test]
    fn test_negatively_correlated_series() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 5,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // Asset A goes up, asset B goes down by the same proportion.
        let mut price_a = 100.0;
        let mut price_b = 100.0;
        for i in 0..30 {
            let change = 0.01 * (1.0 + (i as f64 * 0.1).sin());
            price_a *= 1.0 + change;
            price_b *= 1.0 - change;
            tracker.update("A", price_a);
            tracker.update("B", price_b);
        }

        let corr = tracker
            .correlation("A", "B")
            .expect("should have enough data");
        assert!(
            corr < -0.9,
            "Negatively correlated assets should have corr < -0.9, got {corr}"
        );
    }

    #[test]
    fn test_insufficient_observations_returns_none() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 20,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // Only feed 5 prices → 4 returns, but we need 20.
        for i in 0..5 {
            tracker.update("A", 100.0 + i as f64);
            tracker.update("B", 200.0 + i as f64);
        }

        assert_eq!(
            tracker.correlation("A", "B"),
            None,
            "Should return None when below min_observations"
        );
    }

    #[test]
    fn test_highly_correlated_pairs() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 5,
            correlation_threshold: 0.75,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // A and B: highly correlated
        // A and C: uncorrelated (random-ish)
        for i in 0..30 {
            let p = 100.0 + i as f64 * 2.0;
            tracker.update("A", p);
            tracker.update("B", p * 1.5);
            // C oscillates, roughly uncorrelated with a linear trend
            tracker.update("C", 100.0 + 10.0 * (i as f64 * 0.7).sin());
        }

        let pairs = tracker.highly_correlated_pairs();
        // A-B should appear
        assert!(
            pairs
                .iter()
                .any(|(a, b, _)| (*a == "A" && *b == "B") || (*a == "B" && *b == "A")),
            "A-B should be highly correlated"
        );
    }

    #[test]
    fn test_would_exceed_correlation_limit() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 5,
            correlation_threshold: 0.75,
            max_correlated_positions: 2,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // All three move together.
        for i in 0..30 {
            let p = 100.0 + i as f64 * 2.0;
            tracker.update("BTC", p);
            tracker.update("ETH", p * 0.8);
            tracker.update("SOL", p * 0.3);
        }

        // Already holding BTC and ETH (2 correlated).
        let positions = vec!["BTC".to_string(), "ETH".to_string()];

        // Adding SOL (also correlated) should exceed the limit of 2.
        assert!(
            tracker.would_exceed_correlation_limit("SOL", &positions),
            "Adding a 3rd correlated position should exceed limit of 2"
        );

        // Adding an unknown asset should NOT exceed (no correlation data).
        assert!(
            !tracker.would_exceed_correlation_limit("UNKNOWN", &positions),
            "Unknown asset has no correlation data → should be allowed"
        );
    }

    #[test]
    fn test_would_not_exceed_when_uncorrelated() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 5,
            correlation_threshold: 0.75,
            max_correlated_positions: 3,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // BTC and ETH correlated, DOGE uncorrelated.
        for i in 0..30 {
            let p = 100.0 + i as f64 * 2.0;
            tracker.update("BTC", p);
            tracker.update("ETH", p * 0.8);
            tracker.update("DOGE", 100.0 + 5.0 * (i as f64 * 1.3).sin());
        }

        let positions = vec!["BTC".to_string(), "ETH".to_string()];

        // DOGE is uncorrelated → should be allowed.
        assert!(
            !tracker.would_exceed_correlation_limit("DOGE", &positions),
            "Uncorrelated asset should not trigger correlation limit"
        );
    }

    #[test]
    fn test_discover_hidden_correlations() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 5,
            correlation_threshold: 0.75,
            monitored_pairs: vec![("A".to_string(), "B".to_string())],
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // A, B, and C all correlated, but only A-B is monitored.
        for i in 0..30 {
            let p = 100.0 + i as f64 * 2.0;
            tracker.update("A", p);
            tracker.update("B", p * 1.5);
            tracker.update("C", p * 0.7);
        }

        let hidden = tracker.discover_hidden_correlations();
        // A-C and B-C should be flagged as hidden correlations.
        assert!(
            !hidden.is_empty(),
            "Should discover hidden correlations not in monitored_pairs"
        );
        // A-B should NOT appear (it's in monitored_pairs).
        assert!(
            !hidden
                .iter()
                .any(|(a, b, _)| (*a == "A" && *b == "B") || (*a == "B" && *b == "A")),
            "Monitored pair A-B should not be in hidden correlations"
        );
    }

    #[test]
    fn test_canonical_pair_order() {
        assert_eq!(
            canonical_pair("BTC", "ETH"),
            ("BTC".to_string(), "ETH".to_string())
        );
        assert_eq!(
            canonical_pair("ETH", "BTC"),
            ("BTC".to_string(), "ETH".to_string())
        );
    }

    #[test]
    fn test_update_batch() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 3,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        for i in 0..10 {
            let p = 100.0 + i as f64 * 2.0;
            tracker.update_batch(&[("A", p), ("B", p * 1.5)]);
        }

        // Should have correlations computed.
        let corr = tracker.correlation("A", "B");
        assert!(corr.is_some(), "Batch updates should produce correlations");
        assert!(
            corr.unwrap() > 0.9,
            "Perfectly correlated batch should have corr > 0.9"
        );
    }

    #[test]
    fn test_observation_count() {
        let config = CorrelationConfig {
            window: 10,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // First price establishes baseline, no return.
        tracker.update("A", 100.0);
        assert_eq!(tracker.observation_count("A"), 0);

        // Second price produces first return.
        tracker.update("A", 101.0);
        assert_eq!(tracker.observation_count("A"), 1);

        // Fill past the window.
        for i in 2..20 {
            tracker.update("A", 100.0 + i as f64);
        }
        assert_eq!(
            tracker.observation_count("A"),
            10,
            "Should be capped at window size"
        );
    }

    #[test]
    fn test_tracked_assets() {
        let mut tracker = CorrelationTracker::with_defaults();
        tracker.update("BTCUSD", 67_000.0);
        tracker.update("BTCUSD", 67_500.0);
        tracker.update("ETHUSD", 3_800.0);
        tracker.update("ETHUSD", 3_850.0);

        let assets = tracker.tracked_assets();
        assert!(assets.contains(&"BTCUSD"));
        assert!(assets.contains(&"ETHUSD"));
    }

    #[test]
    fn test_full_matrix() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 3,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        for i in 0..10 {
            let p = 100.0 + i as f64;
            tracker.update("A", p);
            tracker.update("B", p * 2.0);
            tracker.update("C", p * 0.5);
        }

        let matrix = tracker.full_matrix();
        // With 3 assets we expect 3 pairs: A-B, A-C, B-C
        assert_eq!(matrix.len(), 3, "3 assets should produce 3 unique pairs");
    }

    #[test]
    fn test_zero_and_negative_prices_ignored() {
        let mut tracker = CorrelationTracker::with_defaults();
        tracker.update("A", 100.0);
        tracker.update("A", 0.0); // should be ignored
        tracker.update("A", -50.0); // should be ignored
        tracker.update("A", 101.0); // valid

        assert_eq!(
            tracker.observation_count("A"),
            1,
            "Only the 100→101 return should be recorded"
        );
    }

    #[test]
    fn test_correlation_config_defaults() {
        let config = CorrelationConfig::default();
        assert_eq!(config.window, 100);
        assert!((config.correlation_threshold - 0.75).abs() < f64::EPSILON);
        assert_eq!(config.max_correlated_positions, 3);
        assert_eq!(config.min_observations, 20);
        assert!(config.monitored_pairs.is_empty());
    }

    #[test]
    fn test_same_asset_correlation_is_one() {
        let tracker = CorrelationTracker::with_defaults();
        assert_eq!(tracker.correlation("ANYTHING", "ANYTHING"), Some(1.0));
    }

    #[test]
    fn test_rolling_window_drops_old_data() {
        let config = CorrelationConfig {
            window: 5,
            min_observations: 3,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        // Feed 10 prices → 9 returns, but window is 5.
        for i in 0..10 {
            tracker.update("A", 100.0 + i as f64);
            tracker.update("B", 200.0 + i as f64 * 2.0);
        }

        assert_eq!(tracker.observation_count("A"), 5);
        assert_eq!(tracker.observation_count("B"), 5);

        // Correlation should still be valid and high.
        let corr = tracker.correlation("A", "B").expect("enough observations");
        assert!(
            corr > 0.99,
            "Linear trends should be highly correlated, got {corr}"
        );
    }

    #[test]
    fn test_would_exceed_same_asset_not_counted() {
        let config = CorrelationConfig {
            window: 50,
            min_observations: 5,
            correlation_threshold: 0.75,
            max_correlated_positions: 2,
            ..Default::default()
        };
        let mut tracker = CorrelationTracker::new(config);

        for i in 0..30 {
            let p = 100.0 + i as f64;
            tracker.update("BTC", p);
        }

        // Holding BTC already, trying to add BTC again — same-asset
        // should not count as a "correlated position".
        let positions = vec!["BTC".to_string()];
        assert!(
            !tracker.would_exceed_correlation_limit("BTC", &positions),
            "Same asset should not count as correlated for limit purposes"
        );
    }
}
