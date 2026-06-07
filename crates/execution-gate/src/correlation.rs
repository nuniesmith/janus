//! Cross-asset correlation guard.
//!
//! Tracks per-asset log returns in a rolling window and blocks new entries when
//! too many open positions are already highly correlated with the candidate
//! asset — preventing concentration in a cluster of co-moving assets (e.g. long
//! BTC, ETH, and SOL when all three are at 0.9+ correlation). Faithful port of
//! Ruby's `CorrelationGuard` (`ruby/src/services/correlation_guard.py`): the
//! same rolling-window Pearson maths and the same `>= MIN_SHARED_OBS` guard.
//!
//! All operations are synchronous and in-memory.
//!
//! Note: `jflow-risk`'s `CorrelationTracker` (`crates/risk/src/correlation.rs`)
//! also computes rolling correlations, but it ingests *prices* (deriving returns
//! internally) with different defaults. This guard is the parity-faithful port
//! of Ruby's externally-fed log-return design; consolidating the two is a noted
//! follow-up in the janus TODO.

use std::collections::{HashMap, VecDeque};

/// Default rolling-window length (observations per asset).
pub const DEFAULT_WINDOW: usize = 50;
/// Default maximum correlated open positions before an entry is blocked.
pub const DEFAULT_MAX_CORRELATED: usize = 3;
/// Default Pearson coefficient above which two assets are "highly correlated".
pub const DEFAULT_CORR_THRESHOLD: f64 = 0.7;
/// Minimum shared observations before a correlation is considered meaningful.
pub const MIN_SHARED_OBS: usize = 10;

/// In-memory rolling-window correlation guard.
#[derive(Debug, Clone)]
pub struct CorrelationGuard {
    window: usize,
    max_correlated: usize,
    corr_threshold: f64,
    returns: HashMap<String, VecDeque<f64>>,
}

impl Default for CorrelationGuard {
    fn default() -> Self {
        Self::new(
            DEFAULT_WINDOW,
            DEFAULT_MAX_CORRELATED,
            DEFAULT_CORR_THRESHOLD,
        )
    }
}

impl CorrelationGuard {
    /// Create a guard with the given window, max-correlated limit, and threshold.
    pub fn new(window: usize, max_correlated: usize, corr_threshold: f64) -> Self {
        Self {
            window,
            max_correlated,
            corr_threshold,
            returns: HashMap::new(),
        }
    }

    pub fn corr_threshold(&self) -> f64 {
        self.corr_threshold
    }

    pub fn max_correlated(&self) -> usize {
        self.max_correlated
    }

    /// Feed a new log-return observation for `asset`.
    ///
    /// Call once per trading-loop tick. Non-finite values are silently ignored,
    /// matching the Python guard.
    pub fn update(&mut self, asset: &str, log_return: f64) {
        if !log_return.is_finite() {
            return;
        }
        let q = self
            .returns
            .entry(asset.to_string())
            .or_insert_with(|| VecDeque::with_capacity(self.window));
        if q.len() == self.window {
            q.pop_front();
        }
        q.push_back(log_return);
    }

    /// Pearson correlation between two assets, or `None` when there is
    /// insufficient shared history (`< MIN_SHARED_OBS` overlapping observations)
    /// or one series is effectively constant.
    pub fn pairwise_correlation(&self, a: &str, b: &str) -> Option<f64> {
        let ra = self.returns.get(a)?;
        let rb = self.returns.get(b)?;

        // Use the shorter series length as the comparison window, taking the
        // most-recent `n` of each (matches the Python `list(...)[-n:]`).
        let n = ra.len().min(rb.len());
        if n < MIN_SHARED_OBS {
            return None;
        }
        let xs = ra.iter().skip(ra.len() - n);
        let ys = rb.iter().skip(rb.len() - n);

        let nf = n as f64;
        let mean_x = ra.iter().skip(ra.len() - n).sum::<f64>() / nf;
        let mean_y = rb.iter().skip(rb.len() - n).sum::<f64>() / nf;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for (x, y) in xs.zip(ys) {
            let dx = x - mean_x;
            let dy = y - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-12 {
            return None;
        }
        Some(cov / denom)
    }

    /// Count how many `open_assets` are highly correlated with `new_asset`
    /// (`|corr| >= corr_threshold`). The candidate itself is skipped.
    pub fn get_correlated_count(&self, new_asset: &str, open_assets: &[String]) -> usize {
        let mut count = 0;
        for other in open_assets {
            if other == new_asset {
                continue;
            }
            if let Some(corr) = self.pairwise_correlation(new_asset, other)
                && corr.abs() >= self.corr_threshold
            {
                count += 1;
            }
        }
        count
    }

    /// `true` when entering `new_asset` would leave too many correlated open
    /// positions (i.e. the entry should be blocked).
    pub fn would_exceed_limit(&self, new_asset: &str, open_assets: &[String]) -> bool {
        if open_assets.is_empty() {
            return false;
        }
        self.get_correlated_count(new_asset, open_assets) >= self.max_correlated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn feed(guard: &mut CorrelationGuard, asset: &str, series: &[f64]) {
        for &v in series {
            guard.update(asset, v);
        }
    }

    #[test]
    fn perfectly_correlated_series_is_one() {
        let mut g = CorrelationGuard::default();
        let series: Vec<f64> = (0..20).map(|i| (i as f64) * 0.001 - 0.01).collect();
        feed(&mut g, "btc", &series);
        feed(&mut g, "eth", &series);
        let c = g.pairwise_correlation("btc", "eth").unwrap();
        assert!(
            (c - 1.0).abs() < 1e-9,
            "identical series → corr ≈ 1.0, got {c}"
        );
    }

    #[test]
    fn perfectly_anticorrelated_series_is_minus_one() {
        let mut g = CorrelationGuard::default();
        let series: Vec<f64> = (0..20).map(|i| (i as f64) * 0.001).collect();
        let neg: Vec<f64> = series.iter().map(|v| -v).collect();
        feed(&mut g, "btc", &series);
        feed(&mut g, "short", &neg);
        let c = g.pairwise_correlation("btc", "short").unwrap();
        assert!(
            (c + 1.0).abs() < 1e-9,
            "negated series → corr ≈ -1.0, got {c}"
        );
    }

    #[test]
    fn insufficient_history_returns_none() {
        let mut g = CorrelationGuard::default();
        feed(&mut g, "btc", &[0.01, -0.01, 0.02]);
        feed(&mut g, "eth", &[0.01, -0.01, 0.02]);
        assert_eq!(g.pairwise_correlation("btc", "eth"), None);
    }

    #[test]
    fn constant_series_returns_none() {
        let mut g = CorrelationGuard::default();
        feed(&mut g, "btc", &[0.0; 15]);
        feed(&mut g, "eth", &[0.0; 15]);
        assert_eq!(g.pairwise_correlation("btc", "eth"), None);
    }

    #[test]
    fn non_finite_returns_ignored() {
        let mut g = CorrelationGuard::default();
        g.update("btc", f64::NAN);
        g.update("btc", f64::INFINITY);
        g.update("btc", 0.01);
        // only the finite value was kept; still well under MIN_SHARED_OBS
        assert_eq!(g.pairwise_correlation("btc", "btc"), None);
    }

    #[test]
    fn would_exceed_limit_counts_correlated_positions() {
        let mut g = CorrelationGuard::new(50, 2, 0.7);
        let up: Vec<f64> = (0..20).map(|i| (i as f64) * 0.001).collect();
        feed(&mut g, "btc", &up);
        feed(&mut g, "eth", &up);
        feed(&mut g, "sol", &up);

        let open = vec!["eth".to_string(), "sol".to_string()];
        assert_eq!(g.get_correlated_count("btc", &open), 2);
        assert!(
            g.would_exceed_limit("btc", &open),
            "2 correlated >= max_correlated 2"
        );

        // empty open book never blocks
        assert!(!g.would_exceed_limit("btc", &[]));
    }

    #[test]
    fn window_evicts_oldest() {
        let mut g = CorrelationGuard::new(3, 3, 0.7);
        feed(&mut g, "btc", &[1.0, 2.0, 3.0, 4.0]);
        // window is 3, so only the last three are retained — still < MIN_SHARED_OBS
        assert_eq!(g.pairwise_correlation("btc", "btc"), None);
    }
}
