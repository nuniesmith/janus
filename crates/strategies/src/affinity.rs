//! # Strategy Affinity Tracker
//!
//! Tracks per-asset strategy performance and learns which strategies work best
//! on which assets. The affinity system records trade results and computes
//! performance metrics (win rate, P&L, Sharpe estimate) that downstream
//! components use to gate strategy execution.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_strategies::affinity::StrategyAffinityTracker;
//!
//! let mut tracker = StrategyAffinityTracker::new(10); // min 10 trades for confidence
//!
//! // Record trade outcomes
//! tracker.record_trade_result("EMAFlip", "BTCUSD", 150.0, true);
//! tracker.record_trade_result("EMAFlip", "BTCUSD", -50.0, false);
//!
//! // Query performance
//! let best = tracker.best_strategies_for_asset("BTCUSD", 3);
//! let weight = tracker.weight_for("EMAFlip", "BTCUSD");
//! let should_run = tracker.should_enable("EMAFlip", "BTCUSD", 0.3);
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Composite key for strategy+asset performance lookups.
///
/// Serialises to/from the string `"strategy::asset"` so the `HashMap` can be
/// used as a JSON object (JSON only supports string keys).
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct AffinityKey {
    pub strategy: String,
    pub asset: String,
}

impl AffinityKey {
    pub fn new(strategy: impl Into<String>, asset: impl Into<String>) -> Self {
        Self {
            strategy: strategy.into(),
            asset: asset.into(),
        }
    }
}

impl fmt::Display for AffinityKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}::{}", self.strategy, self.asset)
    }
}

impl std::str::FromStr for AffinityKey {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.split_once("::") {
            Some((strategy, asset)) => Ok(Self {
                strategy: strategy.to_string(),
                asset: asset.to_string(),
            }),
            None => Err(format!(
                "invalid AffinityKey: expected 'strategy::asset', got '{s}'"
            )),
        }
    }
}

impl Serialize for AffinityKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for AffinityKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

/// Performance record for a single strategy on a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyPerformance {
    /// Strategy name (e.g. "EMAFlip")
    pub strategy_name: String,

    /// Asset symbol (e.g. "BTCUSD")
    pub asset: String,

    /// Total number of completed trades
    pub total_trades: u32,

    /// Number of winning trades (pnl > 0)
    pub winning_trades: u32,

    /// Cumulative realised P&L (USD or quote currency)
    pub total_pnl: f64,

    /// Running average of realised risk-reward ratio
    pub avg_rr_realized: f64,

    /// Simple Sharpe-like estimate: mean(pnl) / std(pnl).
    /// Only meaningful after `min_trades_for_confidence` trades.
    pub sharpe_estimate: f64,

    /// Maximum drawdown observed from a cumulative P&L peak (always ≤ 0).
    ///
    /// Calculated as the largest drop from any peak in the running P&L curve.
    /// A value of `-500.0` means the strategy once fell $500 from its best
    /// cumulative P&L before recovering (or not).
    pub max_dd: f64,

    /// Timestamp of the last recorded trade
    pub last_updated: DateTime<Utc>,

    // --- internal accumulators (not part of the public API contract) ---
    /// Running list of individual trade P&L values for variance calculation.
    #[serde(default)]
    pnl_history: Vec<f64>,

    /// Peak cumulative P&L seen so far (used for drawdown calculation).
    #[serde(default)]
    peak_pnl: f64,
}

impl StrategyPerformance {
    /// Create a fresh performance record.
    pub fn new(strategy_name: impl Into<String>, asset: impl Into<String>) -> Self {
        Self {
            strategy_name: strategy_name.into(),
            asset: asset.into(),
            total_trades: 0,
            winning_trades: 0,
            total_pnl: 0.0,
            avg_rr_realized: 0.0,
            sharpe_estimate: 0.0,
            max_dd: 0.0,
            last_updated: Utc::now(),
            pnl_history: Vec::new(),
            peak_pnl: 0.0,
        }
    }

    /// Win rate as a fraction in [0.0, 1.0].
    pub fn win_rate(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        f64::from(self.winning_trades) / f64::from(self.total_trades)
    }

    /// Peak cumulative P&L observed (used internally for drawdown tracking).
    pub fn peak_pnl(&self) -> f64 {
        self.peak_pnl
    }

    /// Average P&L per trade.
    pub fn avg_pnl(&self) -> f64 {
        if self.total_trades == 0 {
            return 0.0;
        }
        self.total_pnl / f64::from(self.total_trades)
    }

    /// Record a new trade result and recompute derived metrics.
    fn record(&mut self, pnl: f64, is_winner: bool, rr_ratio: Option<f64>) {
        self.total_trades += 1;
        if is_winner {
            self.winning_trades += 1;
        }
        self.total_pnl += pnl;
        self.pnl_history.push(pnl);
        self.last_updated = Utc::now();

        // Update running average RR if provided
        if let Some(rr) = rr_ratio {
            let n = f64::from(self.total_trades);
            self.avg_rr_realized = self.avg_rr_realized * ((n - 1.0) / n) + rr / n;
        }

        // Update peak P&L and max drawdown
        if self.total_pnl > self.peak_pnl {
            self.peak_pnl = self.total_pnl;
        }
        let current_dd = self.total_pnl - self.peak_pnl; // always ≤ 0
        if current_dd < self.max_dd {
            self.max_dd = current_dd;
        }

        // Recompute Sharpe estimate: mean / stddev
        self.sharpe_estimate = Self::compute_sharpe(&self.pnl_history);
    }

    /// Compute a simple Sharpe-like ratio from a series of P&L values.
    fn compute_sharpe(pnl_values: &[f64]) -> f64 {
        let n = pnl_values.len();
        if n < 2 {
            return 0.0;
        }

        let mean = pnl_values.iter().sum::<f64>() / n as f64;
        let variance =
            pnl_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        let stddev = variance.sqrt();

        if stddev < f64::EPSILON {
            return 0.0;
        }

        mean / stddev
    }
}

/// Tracks performance of every (strategy, asset) pair and provides
/// queries to decide which strategies to enable for a given asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyAffinityTracker {
    /// Performance records keyed by (strategy, asset).
    affinities: HashMap<AffinityKey, StrategyPerformance>,

    /// Minimum number of trades required before we trust the performance
    /// data enough to make enable/disable decisions.
    pub min_trades_for_confidence: usize,
}

impl StrategyAffinityTracker {
    /// Create a new tracker.
    ///
    /// `min_trades` — how many trades per (strategy, asset) pair before the
    /// tracker considers the data statistically meaningful.
    pub fn new(min_trades: usize) -> Self {
        Self {
            affinities: HashMap::new(),
            min_trades_for_confidence: min_trades,
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Recording
    // ────────────────────────────────────────────────────────────────────

    /// Record the outcome of a completed trade.
    ///
    /// * `strategy` — name of the strategy that generated the trade
    /// * `asset`    — symbol the trade was on
    /// * `pnl`      — realised P&L (positive = profit)
    /// * `is_winner` — whether the trade was a winner
    pub fn record_trade_result(&mut self, strategy: &str, asset: &str, pnl: f64, is_winner: bool) {
        self.record_trade_result_with_rr(strategy, asset, pnl, is_winner, None);
    }

    /// Record trade result with an optional risk-reward ratio.
    pub fn record_trade_result_with_rr(
        &mut self,
        strategy: &str,
        asset: &str,
        pnl: f64,
        is_winner: bool,
        rr_ratio: Option<f64>,
    ) {
        let key = AffinityKey::new(strategy, asset);
        let perf = self
            .affinities
            .entry(key)
            .or_insert_with(|| StrategyPerformance::new(strategy, asset));
        perf.record(pnl, is_winner, rr_ratio);
    }

    // ────────────────────────────────────────────────────────────────────
    // Querying
    // ────────────────────────────────────────────────────────────────────

    /// Get the performance record for a specific (strategy, asset) pair.
    pub fn get_performance(&self, strategy: &str, asset: &str) -> Option<&StrategyPerformance> {
        let key = AffinityKey::new(strategy, asset);
        self.affinities.get(&key)
    }

    /// Return the top-N strategies for a given asset, ranked by a composite
    /// score of Sharpe estimate, win rate, and total P&L.
    ///
    /// Only strategies with at least `min_trades_for_confidence` trades are
    /// included.
    pub fn best_strategies_for_asset(
        &self,
        asset: &str,
        top_n: usize,
    ) -> Vec<&StrategyPerformance> {
        let mut candidates: Vec<&StrategyPerformance> = self
            .affinities
            .values()
            .filter(|p| {
                p.asset == asset && p.total_trades as usize >= self.min_trades_for_confidence
            })
            .collect();

        // Sort by composite score descending
        candidates.sort_by(|a, b| {
            let score_a = Self::composite_score(a);
            let score_b = Self::composite_score(b);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates.truncate(top_n);
        candidates
    }

    /// Compute a normalised weight ∈ [0.0, 1.0] for a strategy on an asset.
    ///
    /// Returns `0.5` (neutral) if insufficient data. Higher is better.
    pub fn weight_for(&self, strategy: &str, asset: &str) -> f64 {
        let key = AffinityKey::new(strategy, asset);
        match self.affinities.get(&key) {
            None => 0.5, // no data → neutral
            Some(perf) => {
                if (perf.total_trades as usize) < self.min_trades_for_confidence {
                    0.5 // insufficient data → neutral
                } else {
                    // Map composite score to [0, 1] via sigmoid-like transform
                    let raw = Self::composite_score(perf);
                    sigmoid(raw)
                }
            }
        }
    }

    /// Decide whether a strategy should be enabled for a given asset.
    ///
    /// `min_weight` — threshold below which the strategy is disabled.
    /// Strategies with insufficient data are **enabled** (benefit of the doubt).
    pub fn should_enable(&self, strategy: &str, asset: &str, min_weight: f64) -> bool {
        let weight = self.weight_for(strategy, asset);
        weight >= min_weight
    }

    /// Get all known strategies for an asset, sorted by weight descending.
    pub fn strategies_for_asset(&self, asset: &str) -> Vec<(&str, f64)> {
        let mut result: Vec<(&str, f64)> = self
            .affinities
            .iter()
            .filter(|(k, _)| k.asset == asset)
            .map(|(k, _)| (k.strategy.as_str(), self.weight_for(&k.strategy, &k.asset)))
            .collect();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        result
    }

    /// List all assets that have recorded trades.
    pub fn known_assets(&self) -> Vec<&str> {
        let mut assets: Vec<&str> = self
            .affinities
            .keys()
            .map(|k| k.asset.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        assets.sort();
        assets
    }

    /// List all strategies that have recorded trades.
    pub fn known_strategies(&self) -> Vec<&str> {
        let mut strategies: Vec<&str> = self
            .affinities
            .keys()
            .map(|k| k.strategy.as_str())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        strategies.sort();
        strategies
    }

    // ────────────────────────────────────────────────────────────────────
    // Introspection
    // ────────────────────────────────────────────────────────────────────

    /// Return the number of tracked (strategy, asset) pairs.
    pub fn pair_count(&self) -> usize {
        self.affinities.len()
    }

    // ────────────────────────────────────────────────────────────────────
    // Persistence
    // ────────────────────────────────────────────────────────────────────

    /// Serialise the tracker state to JSON bytes for persistence (Redis / disk).
    pub fn save_state(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec_pretty(self)
    }

    /// Restore the tracker from previously saved JSON bytes.
    pub fn load_state(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }

    // ────────────────────────────────────────────────────────────────────
    // Internal helpers
    // ────────────────────────────────────────────────────────────────────

    /// Composite performance score combining Sharpe, win rate, avg P&L, and
    /// max drawdown.
    ///
    /// Weights chosen heuristically:
    ///   - Sharpe estimate:  35%  (risk-adjusted returns)
    ///   - Win rate:         25%  (consistency)
    ///   - Avg PnL sign:     25%  (profitability direction)
    ///   - Max drawdown:     15%  (downside risk penalty)
    fn composite_score(perf: &StrategyPerformance) -> f64 {
        let sharpe_component = perf.sharpe_estimate * 0.35;
        let wr_component = (perf.win_rate() - 0.5) * 2.0 * 0.25; // centre around 50%
        let pnl_component = perf.avg_pnl().signum() * perf.avg_pnl().abs().ln_1p() * 0.25;
        // max_dd is ≤ 0, so ln_1p(|max_dd|) is a positive penalty scaled negatively
        let dd_component = -(perf.max_dd.abs().ln_1p()) * 0.15;
        sharpe_component + wr_component + pnl_component + dd_component
    }
}

/// Soft sigmoid mapping ℝ → (0, 1). Used to convert a raw composite score
/// into a bounded weight.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker() {
        let tracker = StrategyAffinityTracker::new(10);
        assert_eq!(tracker.min_trades_for_confidence, 10);
        assert!(tracker.affinities.is_empty());
    }

    #[test]
    fn test_record_and_retrieve() {
        let mut tracker = StrategyAffinityTracker::new(2);

        tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        tracker.record_trade_result("EMAFlip", "BTCUSD", -30.0, false);
        tracker.record_trade_result("EMAFlip", "BTCUSD", 50.0, true);

        let perf = tracker.get_performance("EMAFlip", "BTCUSD").unwrap();
        assert_eq!(perf.total_trades, 3);
        assert_eq!(perf.winning_trades, 2);
        assert!((perf.total_pnl - 120.0).abs() < f64::EPSILON);
        assert!((perf.win_rate() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_weight_no_data_is_neutral() {
        let tracker = StrategyAffinityTracker::new(5);
        let weight = tracker.weight_for("Unknown", "BTCUSD");
        assert!((weight - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weight_insufficient_data_is_neutral() {
        let mut tracker = StrategyAffinityTracker::new(10);
        tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        // Only 1 trade, need 10 → neutral
        let weight = tracker.weight_for("EMAFlip", "BTCUSD");
        assert!((weight - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_weight_good_strategy_above_half() {
        let mut tracker = StrategyAffinityTracker::new(3);
        // Record 5 winning trades
        for _ in 0..5 {
            tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        }
        let weight = tracker.weight_for("EMAFlip", "BTCUSD");
        assert!(
            weight > 0.5,
            "Good strategy weight should be > 0.5, got {weight}"
        );
    }

    #[test]
    fn test_weight_bad_strategy_below_half() {
        let mut tracker = StrategyAffinityTracker::new(3);
        // Record 5 losing trades
        for _ in 0..5 {
            tracker.record_trade_result("BadStrat", "ETHUSD", -100.0, false);
        }
        let weight = tracker.weight_for("BadStrat", "ETHUSD");
        assert!(
            weight < 0.5,
            "Bad strategy weight should be < 0.5, got {weight}"
        );
    }

    #[test]
    fn test_should_enable() {
        let mut tracker = StrategyAffinityTracker::new(3);
        for _ in 0..5 {
            tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        }
        // Good strategy should be enabled at default threshold
        assert!(tracker.should_enable("EMAFlip", "BTCUSD", 0.3));

        // Unknown strategy should be enabled (benefit of the doubt)
        assert!(tracker.should_enable("Unknown", "BTCUSD", 0.3));
    }

    #[test]
    fn test_best_strategies_for_asset() {
        let mut tracker = StrategyAffinityTracker::new(3);

        // Strategy A: great performance
        for _ in 0..5 {
            tracker.record_trade_result("StratA", "BTCUSD", 200.0, true);
        }

        // Strategy B: decent performance
        for _ in 0..5 {
            tracker.record_trade_result("StratB", "BTCUSD", 50.0, true);
        }
        tracker.record_trade_result("StratB", "BTCUSD", -20.0, false);

        // Strategy C: bad performance
        for _ in 0..5 {
            tracker.record_trade_result("StratC", "BTCUSD", -100.0, false);
        }

        let best = tracker.best_strategies_for_asset("BTCUSD", 2);
        assert_eq!(best.len(), 2);
        assert_eq!(best[0].strategy_name, "StratA");
        assert_eq!(best[1].strategy_name, "StratB");
    }

    #[test]
    fn test_best_strategies_excludes_insufficient_data() {
        let mut tracker = StrategyAffinityTracker::new(5);

        // Only 2 trades — below threshold of 5
        tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        tracker.record_trade_result("EMAFlip", "BTCUSD", 50.0, true);

        let best = tracker.best_strategies_for_asset("BTCUSD", 5);
        assert!(best.is_empty());
    }

    #[test]
    fn test_strategies_for_asset() {
        let mut tracker = StrategyAffinityTracker::new(1);
        tracker.record_trade_result("A", "BTCUSD", 100.0, true);
        tracker.record_trade_result("B", "BTCUSD", -50.0, false);
        tracker.record_trade_result("C", "ETHUSD", 30.0, true);

        let strats = tracker.strategies_for_asset("BTCUSD");
        assert_eq!(strats.len(), 2);
        // C should not appear (different asset)
        assert!(strats.iter().all(|(name, _)| *name != "C"));
    }

    #[test]
    fn test_known_assets_and_strategies() {
        let mut tracker = StrategyAffinityTracker::new(1);
        tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        tracker.record_trade_result("MeanRev", "ETHUSD", -10.0, false);

        let assets = tracker.known_assets();
        assert!(assets.contains(&"BTCUSD"));
        assert!(assets.contains(&"ETHUSD"));

        let strats = tracker.known_strategies();
        assert!(strats.contains(&"EMAFlip"));
        assert!(strats.contains(&"MeanRev"));
    }

    #[test]
    fn test_save_and_load_state() {
        let mut tracker = StrategyAffinityTracker::new(5);
        tracker.record_trade_result("EMAFlip", "BTCUSD", 100.0, true);
        tracker.record_trade_result("EMAFlip", "BTCUSD", -30.0, false);

        let saved = tracker.save_state().expect("serialize should succeed");
        let restored =
            StrategyAffinityTracker::load_state(&saved).expect("deserialize should succeed");

        assert_eq!(restored.min_trades_for_confidence, 5);
        let perf = restored.get_performance("EMAFlip", "BTCUSD").unwrap();
        assert_eq!(perf.total_trades, 2);
        assert!((perf.total_pnl - 70.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_record_with_rr_ratio() {
        let mut tracker = StrategyAffinityTracker::new(1);
        tracker.record_trade_result_with_rr("EMAFlip", "BTCUSD", 100.0, true, Some(2.5));
        tracker.record_trade_result_with_rr("EMAFlip", "BTCUSD", 50.0, true, Some(1.5));

        let perf = tracker.get_performance("EMAFlip", "BTCUSD").unwrap();
        assert!((perf.avg_rr_realized - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_performance_avg_pnl() {
        let mut perf = StrategyPerformance::new("X", "Y");
        perf.record(100.0, true, None);
        perf.record(-40.0, false, None);
        assert!((perf.avg_pnl() - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_sharpe_estimate_positive() {
        let mut tracker = StrategyAffinityTracker::new(1);
        // Consistent positive returns → positive Sharpe
        for i in 0..10 {
            tracker.record_trade_result("Consistent", "BTCUSD", 100.0 + i as f64, true);
        }
        let perf = tracker.get_performance("Consistent", "BTCUSD").unwrap();
        assert!(
            perf.sharpe_estimate > 0.0,
            "Sharpe should be positive for consistent gains"
        );
    }

    #[test]
    fn test_sharpe_estimate_negative() {
        let mut tracker = StrategyAffinityTracker::new(1);
        // Consistent negative returns → negative Sharpe
        for i in 0..10 {
            tracker.record_trade_result("Loser", "BTCUSD", -100.0 - i as f64, false);
        }
        let perf = tracker.get_performance("Loser", "BTCUSD").unwrap();
        assert!(
            perf.sharpe_estimate < 0.0,
            "Sharpe should be negative for consistent losses"
        );
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < f64::EPSILON);
        assert!(sigmoid(100.0) > 0.99);
        assert!(sigmoid(-100.0) < 0.01);
    }

    #[test]
    fn test_empty_performance() {
        let perf = StrategyPerformance::new("X", "Y");
        assert_eq!(perf.win_rate(), 0.0);
        assert_eq!(perf.avg_pnl(), 0.0);
        assert_eq!(perf.sharpe_estimate, 0.0);
        assert_eq!(perf.max_dd, 0.0);
    }

    #[test]
    fn test_max_drawdown_tracks_peak_to_trough() {
        let mut tracker = StrategyAffinityTracker::new(1);

        // Build up P&L to 300
        tracker.record_trade_result("Strat", "BTCUSD", 100.0, true);
        tracker.record_trade_result("Strat", "BTCUSD", 100.0, true);
        tracker.record_trade_result("Strat", "BTCUSD", 100.0, true);

        // Drop by 200 (peak 300 → 100)
        tracker.record_trade_result("Strat", "BTCUSD", -150.0, false);
        tracker.record_trade_result("Strat", "BTCUSD", -50.0, false);

        let perf = tracker.get_performance("Strat", "BTCUSD").unwrap();
        assert!(
            (perf.max_dd - (-200.0)).abs() < f64::EPSILON,
            "max_dd should be -200.0, got {}",
            perf.max_dd
        );
        assert!(
            (perf.total_pnl - 100.0).abs() < f64::EPSILON,
            "total_pnl should be 100.0, got {}",
            perf.total_pnl
        );
    }

    #[test]
    fn test_max_drawdown_remembers_worst() {
        let mut tracker = StrategyAffinityTracker::new(1);

        // Up 200, down 150 (dd = -150)
        tracker.record_trade_result("Strat", "BTCUSD", 200.0, true);
        tracker.record_trade_result("Strat", "BTCUSD", -150.0, false);

        // Recover to new peak 250, then drop 100 (dd = -100, but worst is still -150)
        tracker.record_trade_result("Strat", "BTCUSD", 200.0, true);
        tracker.record_trade_result("Strat", "BTCUSD", -100.0, false);

        let perf = tracker.get_performance("Strat", "BTCUSD").unwrap();
        assert!(
            (perf.max_dd - (-150.0)).abs() < f64::EPSILON,
            "max_dd should remember the worst drawdown of -150.0, got {}",
            perf.max_dd
        );
    }

    #[test]
    fn test_max_drawdown_pure_losses() {
        let mut tracker = StrategyAffinityTracker::new(1);

        // All losses from the start — peak stays at 0
        tracker.record_trade_result("Strat", "BTCUSD", -50.0, false);
        tracker.record_trade_result("Strat", "BTCUSD", -30.0, false);

        let perf = tracker.get_performance("Strat", "BTCUSD").unwrap();
        assert!(
            (perf.max_dd - (-80.0)).abs() < f64::EPSILON,
            "max_dd should be -80.0 for pure losses, got {}",
            perf.max_dd
        );
    }

    #[test]
    fn test_max_drawdown_pure_wins() {
        let mut tracker = StrategyAffinityTracker::new(1);

        // All wins — no drawdown
        tracker.record_trade_result("Strat", "BTCUSD", 100.0, true);
        tracker.record_trade_result("Strat", "BTCUSD", 50.0, true);

        let perf = tracker.get_performance("Strat", "BTCUSD").unwrap();
        assert!(
            (perf.max_dd - 0.0).abs() < f64::EPSILON,
            "max_dd should be 0.0 for pure wins, got {}",
            perf.max_dd
        );
    }

    #[test]
    fn test_max_drawdown_survives_serialization() {
        let mut tracker = StrategyAffinityTracker::new(1);
        tracker.record_trade_result("Strat", "BTCUSD", 200.0, true);
        tracker.record_trade_result("Strat", "BTCUSD", -120.0, false);

        let saved = tracker.save_state().expect("serialize");
        let restored = StrategyAffinityTracker::load_state(&saved).expect("deserialize");

        let perf = restored.get_performance("Strat", "BTCUSD").unwrap();
        assert!(
            (perf.max_dd - (-120.0)).abs() < f64::EPSILON,
            "max_dd should survive serialization round-trip, got {}",
            perf.max_dd
        );
        assert!(
            (perf.peak_pnl() - 200.0).abs() < f64::EPSILON,
            "peak_pnl should survive serialization round-trip, got {}",
            perf.peak_pnl()
        );
    }
}
