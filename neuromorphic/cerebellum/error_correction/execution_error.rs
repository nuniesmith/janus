//! Execution error tracking — deviation from execution plan
//!
//! Part of the Cerebellum region
//! Component: error_correction
//!
//! Tracks the deviation between intended execution parameters (target price,
//! target fill rate, timing) and actual execution outcomes. Computes
//! slippage, timing drift, and fill-rate shortfall as normalized error
//! signals that downstream correction modules can act on.
//!
//! Key features:
//! - Per-order slippage tracking (price deviation from arrival price)
//! - Timing error: how far the actual fill time drifted from the plan
//! - Fill-rate shortfall: partial fills / unfilled quantity
//! - Rolling statistics with configurable window
//! - Composite error score combining all dimensions

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for execution error tracking
#[derive(Debug, Clone)]
pub struct ExecutionErrorConfig {
    /// Maximum number of observations in the sliding window
    pub window_size: usize,
    /// EMA decay for smoothed error signals
    pub ema_decay: f64,
    /// Weight of slippage in composite error score
    pub slippage_weight: f64,
    /// Weight of timing error in composite error score
    pub timing_weight: f64,
    /// Weight of fill-rate shortfall in composite error score
    pub fill_rate_weight: f64,
    /// Threshold (in basis points) below which slippage is ignored
    pub slippage_dead_zone_bps: f64,
    /// Maximum composite error value (safety clamp)
    pub max_composite_error: f64,
}

impl Default for ExecutionErrorConfig {
    fn default() -> Self {
        Self {
            window_size: 200,
            ema_decay: 0.95,
            slippage_weight: 0.50,
            timing_weight: 0.20,
            fill_rate_weight: 0.30,
            slippage_dead_zone_bps: 0.5,
            max_composite_error: 100.0,
        }
    }
}

/// Side of the executed order
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// A single execution observation
#[derive(Debug, Clone)]
pub struct ExecutionObservation {
    /// Arrival / decision price (mid-price when the algo decided to trade)
    pub arrival_price: f64,
    /// Volume-weighted average fill price achieved
    pub fill_price: f64,
    /// Intended quantity
    pub intended_quantity: f64,
    /// Actually filled quantity
    pub filled_quantity: f64,
    /// Planned execution duration in seconds
    pub planned_duration_secs: f64,
    /// Actual execution duration in seconds
    pub actual_duration_secs: f64,
    /// Side of the order
    pub side: OrderSide,
}

/// Computed error signals for a single execution
#[derive(Debug, Clone)]
pub struct ErrorSignals {
    /// Slippage in basis points (positive = adverse)
    pub slippage_bps: f64,
    /// Absolute slippage in price units
    pub slippage_price: f64,
    /// Timing error as fraction of planned duration (0.0 = on time, >0 = late, <0 = early)
    pub timing_error: f64,
    /// Fill-rate shortfall (0.0 = fully filled, 1.0 = nothing filled)
    pub fill_shortfall: f64,
    /// Composite error score (weighted blend, 0.0 = perfect)
    pub composite: f64,
    /// Whether slippage fell within the dead zone
    pub in_dead_zone: bool,
}

/// Running statistics for execution errors
#[derive(Debug, Clone, Default)]
pub struct ExecutionErrorStats {
    /// Total observations processed
    pub total_observations: u64,
    /// Total slippage cost in price units (sum of adverse slippage × quantity)
    pub total_slippage_cost: f64,
    /// Mean slippage in basis points (EMA-smoothed)
    pub mean_slippage_bps: f64,
    /// Mean timing error (EMA-smoothed)
    pub mean_timing_error: f64,
    /// Mean fill shortfall (EMA-smoothed)
    pub mean_fill_shortfall: f64,
    /// Mean composite error (EMA-smoothed)
    pub mean_composite: f64,
    /// Maximum slippage observed (bps)
    pub max_slippage_bps: f64,
    /// Maximum fill shortfall observed
    pub max_fill_shortfall: f64,
    /// Number of observations where slippage exceeded the dead zone
    pub adverse_slippage_count: u64,
    /// Number of observations with partial fills (shortfall > 0)
    pub partial_fill_count: u64,
    /// Sum of squared slippage (for variance calculation)
    pub sum_sq_slippage: f64,
}

impl ExecutionErrorStats {
    /// Slippage variance (from the EMA-smoothed mean)
    pub fn slippage_variance(&self) -> f64 {
        if self.total_observations < 2 {
            return 0.0;
        }
        let mean_sq = self.sum_sq_slippage / self.total_observations as f64;
        (mean_sq - self.mean_slippage_bps * self.mean_slippage_bps).max(0.0)
    }

    /// Slippage standard deviation
    pub fn slippage_std(&self) -> f64 {
        self.slippage_variance().sqrt()
    }

    /// Fraction of executions with adverse slippage
    pub fn adverse_rate(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.adverse_slippage_count as f64 / self.total_observations as f64
    }

    /// Fraction of executions with partial fills
    pub fn partial_fill_rate(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.partial_fill_count as f64 / self.total_observations as f64
    }
}

/// Execution error tracker
///
/// Measures deviation from execution plans across three dimensions:
/// 1. **Slippage**: price deviation from arrival price (in basis points)
/// 2. **Timing**: deviation from planned execution duration
/// 3. **Fill rate**: shortfall between intended and filled quantity
///
/// All signals are tracked as exponential moving averages and as
/// windowed raw observations for downstream analysis.
pub struct ExecutionError {
    config: ExecutionErrorConfig,
    /// EMA-smoothed slippage (bps)
    ema_slippage: f64,
    /// EMA-smoothed timing error
    ema_timing: f64,
    /// EMA-smoothed fill shortfall
    ema_fill_shortfall: f64,
    /// EMA-smoothed composite error
    ema_composite: f64,
    /// Whether EMAs have been seeded
    ema_initialized: bool,
    /// Recent error signals for windowed analysis
    recent: VecDeque<ErrorSignals>,
    /// Running statistics
    stats: ExecutionErrorStats,
}

impl Default for ExecutionError {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionError {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(ExecutionErrorConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: ExecutionErrorConfig) -> Self {
        let capacity = config.window_size;
        Self {
            ema_slippage: 0.0,
            ema_timing: 0.0,
            ema_fill_shortfall: 0.0,
            ema_composite: 0.0,
            ema_initialized: false,
            recent: VecDeque::with_capacity(capacity),
            stats: ExecutionErrorStats::default(),
            config,
        }
    }

    /// Validate internal state
    pub fn process(&self) -> Result<()> {
        let w_sum =
            self.config.slippage_weight + self.config.timing_weight + self.config.fill_rate_weight;
        if (w_sum - 1.0).abs() > 0.01 {
            return Err(Error::Configuration(format!(
                "ExecutionError: composite weights must sum to ~1.0, got {}",
                w_sum
            )));
        }
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::Configuration(
                "ExecutionError: ema_decay must be in (0, 1)".into(),
            ));
        }
        Ok(())
    }

    /// Record an execution observation and compute error signals
    pub fn record(&mut self, obs: &ExecutionObservation) -> Result<ErrorSignals> {
        if obs.arrival_price <= 0.0 {
            return Err(Error::InvalidInput("arrival_price must be positive".into()));
        }
        if obs.fill_price <= 0.0 {
            return Err(Error::InvalidInput("fill_price must be positive".into()));
        }
        if obs.intended_quantity <= 0.0 {
            return Err(Error::InvalidInput(
                "intended_quantity must be positive".into(),
            ));
        }
        if obs.filled_quantity < 0.0 {
            return Err(Error::InvalidInput(
                "filled_quantity must be non-negative".into(),
            ));
        }

        let signals = self.compute_signals(obs);
        self.update_ema(&signals);
        self.update_stats(&signals, obs);

        // Maintain sliding window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(signals.clone());

        Ok(signals)
    }

    /// Compute error signals from a single observation without recording it
    pub fn compute_signals(&self, obs: &ExecutionObservation) -> ErrorSignals {
        // --- Slippage ---
        // Adverse slippage: buying higher or selling lower than arrival
        let signed_slippage = match obs.side {
            OrderSide::Buy => obs.fill_price - obs.arrival_price,
            OrderSide::Sell => obs.arrival_price - obs.fill_price,
        };
        let slippage_bps = (signed_slippage / obs.arrival_price) * 10_000.0;
        let in_dead_zone = slippage_bps.abs() < self.config.slippage_dead_zone_bps;

        // For composite scoring, only count adverse slippage (positive)
        // and zero out anything in the dead zone
        let effective_slippage_bps = if in_dead_zone {
            0.0
        } else {
            slippage_bps.max(0.0) // only adverse direction contributes
        };

        // --- Timing ---
        let timing_error = if obs.planned_duration_secs > 0.0 {
            (obs.actual_duration_secs - obs.planned_duration_secs) / obs.planned_duration_secs
        } else {
            0.0
        };

        // --- Fill rate ---
        let fill_shortfall = if obs.intended_quantity > 0.0 {
            1.0 - (obs.filled_quantity / obs.intended_quantity).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // --- Composite ---
        // Normalize each component to a 0-100 scale, then combine
        let norm_slippage = effective_slippage_bps.min(100.0); // cap at 100 bps for normalization
        let norm_timing = (timing_error.abs() * 100.0).min(100.0); // 100% late = 100
        let norm_fill = fill_shortfall * 100.0; // 0-100

        let composite = (self.config.slippage_weight * norm_slippage
            + self.config.timing_weight * norm_timing
            + self.config.fill_rate_weight * norm_fill)
            .min(self.config.max_composite_error);

        ErrorSignals {
            slippage_bps,
            slippage_price: signed_slippage,
            timing_error,
            fill_shortfall,
            composite,
            in_dead_zone,
        }
    }

    /// Get the latest EMA-smoothed slippage in basis points
    pub fn smoothed_slippage_bps(&self) -> f64 {
        self.ema_slippage
    }

    /// Get the latest EMA-smoothed timing error
    pub fn smoothed_timing_error(&self) -> f64 {
        self.ema_timing
    }

    /// Get the latest EMA-smoothed fill shortfall
    pub fn smoothed_fill_shortfall(&self) -> f64 {
        self.ema_fill_shortfall
    }

    /// Get the latest EMA-smoothed composite error
    pub fn smoothed_composite(&self) -> f64 {
        self.ema_composite
    }

    /// Get the running statistics
    pub fn stats(&self) -> &ExecutionErrorStats {
        &self.stats
    }

    /// Get the most recent N error signals
    pub fn recent_signals(&self, n: usize) -> Vec<&ErrorSignals> {
        self.recent.iter().rev().take(n).collect()
    }

    /// Get the total number of observations in the window
    pub fn window_size(&self) -> usize {
        self.recent.len()
    }

    /// Compute the windowed mean composite error (not EMA, just arithmetic mean)
    pub fn windowed_mean_composite(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.composite).sum();
        sum / self.recent.len() as f64
    }

    /// Compute the windowed mean slippage in bps
    pub fn windowed_mean_slippage_bps(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|s| s.slippage_bps).sum();
        sum / self.recent.len() as f64
    }

    /// Check if execution quality is degrading (composite error trend is rising)
    pub fn is_degrading(&self) -> bool {
        if self.recent.len() < 10 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|s| s.composite)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|s| s.composite)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;

        second_half > first_half * 1.2 // 20% worse
    }

    /// Reset all state while keeping configuration
    pub fn reset(&mut self) {
        self.ema_slippage = 0.0;
        self.ema_timing = 0.0;
        self.ema_fill_shortfall = 0.0;
        self.ema_composite = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = ExecutionErrorStats::default();
    }

    // ── internal ──

    fn update_ema(&mut self, signals: &ErrorSignals) {
        if !self.ema_initialized {
            self.ema_slippage = signals.slippage_bps;
            self.ema_timing = signals.timing_error;
            self.ema_fill_shortfall = signals.fill_shortfall;
            self.ema_composite = signals.composite;
            self.ema_initialized = true;
        } else {
            let alpha = 1.0 - self.config.ema_decay;
            self.ema_slippage =
                self.config.ema_decay * self.ema_slippage + alpha * signals.slippage_bps;
            self.ema_timing =
                self.config.ema_decay * self.ema_timing + alpha * signals.timing_error;
            self.ema_fill_shortfall =
                self.config.ema_decay * self.ema_fill_shortfall + alpha * signals.fill_shortfall;
            self.ema_composite =
                self.config.ema_decay * self.ema_composite + alpha * signals.composite;
        }
    }

    fn update_stats(&mut self, signals: &ErrorSignals, obs: &ExecutionObservation) {
        self.stats.total_observations += 1;
        self.stats.total_slippage_cost += signals.slippage_price.abs() * obs.filled_quantity;
        self.stats.mean_slippage_bps = self.ema_slippage;
        self.stats.mean_timing_error = self.ema_timing;
        self.stats.mean_fill_shortfall = self.ema_fill_shortfall;
        self.stats.mean_composite = self.ema_composite;
        self.stats.sum_sq_slippage += signals.slippage_bps * signals.slippage_bps;

        if signals.slippage_bps.abs() > self.stats.max_slippage_bps {
            self.stats.max_slippage_bps = signals.slippage_bps.abs();
        }
        if signals.fill_shortfall > self.stats.max_fill_shortfall {
            self.stats.max_fill_shortfall = signals.fill_shortfall;
        }
        if !signals.in_dead_zone && signals.slippage_bps > 0.0 {
            self.stats.adverse_slippage_count += 1;
        }
        if signals.fill_shortfall > 0.0 {
            self.stats.partial_fill_count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perfect_fill() -> ExecutionObservation {
        ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.0,
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Buy,
        }
    }

    #[test]
    fn test_basic() {
        let instance = ExecutionError::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_perfect_fill_zero_error() {
        let mut tracker = ExecutionError::new();
        let signals = tracker.record(&perfect_fill()).unwrap();

        assert!((signals.slippage_bps).abs() < 1e-12);
        assert!((signals.timing_error).abs() < 1e-12);
        assert!((signals.fill_shortfall).abs() < 1e-12);
        assert!((signals.composite).abs() < 1e-12);
    }

    #[test]
    fn test_buy_slippage() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.10, // 10 bps adverse slippage
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!((signals.slippage_bps - 10.0).abs() < 0.1);
        assert!(signals.slippage_price > 0.0);
        assert!(!signals.in_dead_zone);
    }

    #[test]
    fn test_sell_slippage() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 99.90, // 10 bps adverse slippage on sell
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Sell,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!((signals.slippage_bps - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_favorable_slippage_buy() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 99.95, // Favorable: bought cheaper
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        // Negative slippage = favorable
        assert!(signals.slippage_bps < 0.0);
        // Composite should not penalize favorable slippage
        assert!(signals.composite < 1.0);
    }

    #[test]
    fn test_dead_zone() {
        let mut tracker = ExecutionError::with_config(ExecutionErrorConfig {
            slippage_dead_zone_bps: 2.0,
            ..Default::default()
        });
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.01, // 1 bps — within dead zone of 2 bps
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!(signals.in_dead_zone);
    }

    #[test]
    fn test_timing_error_late() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.0,
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 90.0, // 50% late
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!((signals.timing_error - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_timing_error_early() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.0,
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 30.0, // 50% early
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!((signals.timing_error - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn test_partial_fill() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.0,
            intended_quantity: 1000.0,
            filled_quantity: 700.0, // 30% shortfall
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!((signals.fill_shortfall - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_zero_fill() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.0,
            intended_quantity: 1000.0,
            filled_quantity: 0.0, // Total shortfall
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!((signals.fill_shortfall - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_composite_weighted_correctly() {
        let config = ExecutionErrorConfig {
            slippage_weight: 1.0,
            timing_weight: 0.0,
            fill_rate_weight: 0.0,
            slippage_dead_zone_bps: 0.0,
            ..Default::default()
        };
        let mut tracker = ExecutionError::with_config(config);
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.05, // 5 bps
            intended_quantity: 1000.0,
            filled_quantity: 500.0, // 50% shortfall — should not affect composite
            planned_duration_secs: 60.0,
            actual_duration_secs: 120.0, // 100% late — should not affect composite
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        // With 100% slippage weight, composite should equal just the slippage component
        assert!((signals.composite - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_ema_smoothing() {
        let mut tracker = ExecutionError::new();

        // Record several perfect fills
        for _ in 0..10 {
            tracker.record(&perfect_fill()).unwrap();
        }
        assert!(tracker.smoothed_composite().abs() < 1e-6);

        // Record a bad fill
        let bad = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 101.0, // 100 bps adverse
            intended_quantity: 1000.0,
            filled_quantity: 500.0,
            planned_duration_secs: 60.0,
            actual_duration_secs: 120.0,
            side: OrderSide::Buy,
        };
        tracker.record(&bad).unwrap();

        // EMA should have moved up but be smoothed
        assert!(tracker.smoothed_composite() > 0.0);
        assert!(tracker.smoothed_slippage_bps() > 0.0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut tracker = ExecutionError::new();

        for _ in 0..5 {
            let obs = ExecutionObservation {
                arrival_price: 100.0,
                fill_price: 100.02, // 2 bps
                intended_quantity: 1000.0,
                filled_quantity: 800.0, // 20% shortfall
                planned_duration_secs: 60.0,
                actual_duration_secs: 60.0,
                side: OrderSide::Buy,
            };
            tracker.record(&obs).unwrap();
        }

        let stats = tracker.stats();
        assert_eq!(stats.total_observations, 5);
        assert!(stats.total_slippage_cost > 0.0);
        assert!(stats.adverse_slippage_count > 0);
        assert_eq!(stats.partial_fill_count, 5);
        assert!(stats.slippage_std() >= 0.0);
    }

    #[test]
    fn test_adverse_rate() {
        let mut tracker = ExecutionError::new();

        // 3 adverse
        for _ in 0..3 {
            tracker
                .record(&ExecutionObservation {
                    arrival_price: 100.0,
                    fill_price: 100.05,
                    intended_quantity: 100.0,
                    filled_quantity: 100.0,
                    planned_duration_secs: 10.0,
                    actual_duration_secs: 10.0,
                    side: OrderSide::Buy,
                })
                .unwrap();
        }
        // 2 favorable (within dead zone or favorable)
        for _ in 0..2 {
            tracker.record(&perfect_fill()).unwrap();
        }

        let rate = tracker.stats().adverse_rate();
        assert!((rate - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_is_degrading() {
        let mut tracker = ExecutionError::new();

        // First half: good fills
        for _ in 0..10 {
            tracker.record(&perfect_fill()).unwrap();
        }

        // Second half: bad fills
        for _ in 0..10 {
            tracker
                .record(&ExecutionObservation {
                    arrival_price: 100.0,
                    fill_price: 100.50, // 50 bps
                    intended_quantity: 1000.0,
                    filled_quantity: 500.0,
                    planned_duration_secs: 60.0,
                    actual_duration_secs: 120.0,
                    side: OrderSide::Buy,
                })
                .unwrap();
        }

        assert!(tracker.is_degrading());
    }

    #[test]
    fn test_not_degrading_when_consistent() {
        let mut tracker = ExecutionError::new();

        for _ in 0..20 {
            tracker.record(&perfect_fill()).unwrap();
        }

        assert!(!tracker.is_degrading());
    }

    #[test]
    fn test_not_degrading_insufficient_data() {
        let mut tracker = ExecutionError::new();
        for _ in 0..5 {
            tracker.record(&perfect_fill()).unwrap();
        }
        assert!(!tracker.is_degrading());
    }

    #[test]
    fn test_windowed_means() {
        let mut tracker = ExecutionError::new();

        for _ in 0..5 {
            tracker
                .record(&ExecutionObservation {
                    arrival_price: 100.0,
                    fill_price: 100.03, // 3 bps
                    intended_quantity: 1000.0,
                    filled_quantity: 1000.0,
                    planned_duration_secs: 60.0,
                    actual_duration_secs: 60.0,
                    side: OrderSide::Buy,
                })
                .unwrap();
        }

        let mean_slip = tracker.windowed_mean_slippage_bps();
        assert!((mean_slip - 3.0).abs() < 0.1);
        assert!(tracker.windowed_mean_composite() > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut tracker = ExecutionError::new();

        for _ in 0..10 {
            tracker
                .record(&ExecutionObservation {
                    arrival_price: 100.0,
                    fill_price: 100.10,
                    intended_quantity: 1000.0,
                    filled_quantity: 500.0,
                    planned_duration_secs: 60.0,
                    actual_duration_secs: 90.0,
                    side: OrderSide::Buy,
                })
                .unwrap();
        }

        assert!(tracker.stats().total_observations > 0);
        tracker.reset();
        assert_eq!(tracker.stats().total_observations, 0);
        assert_eq!(tracker.window_size(), 0);
        assert!(tracker.smoothed_composite().abs() < 1e-12);
    }

    #[test]
    fn test_invalid_inputs() {
        let mut tracker = ExecutionError::new();
        assert!(
            tracker
                .record(&ExecutionObservation {
                    arrival_price: -1.0,
                    fill_price: 100.0,
                    intended_quantity: 100.0,
                    filled_quantity: 100.0,
                    planned_duration_secs: 10.0,
                    actual_duration_secs: 10.0,
                    side: OrderSide::Buy,
                })
                .is_err()
        );

        assert!(
            tracker
                .record(&ExecutionObservation {
                    arrival_price: 100.0,
                    fill_price: 0.0,
                    intended_quantity: 100.0,
                    filled_quantity: 100.0,
                    planned_duration_secs: 10.0,
                    actual_duration_secs: 10.0,
                    side: OrderSide::Buy,
                })
                .is_err()
        );

        assert!(
            tracker
                .record(&ExecutionObservation {
                    arrival_price: 100.0,
                    fill_price: 100.0,
                    intended_quantity: -1.0,
                    filled_quantity: 100.0,
                    planned_duration_secs: 10.0,
                    actual_duration_secs: 10.0,
                    side: OrderSide::Buy,
                })
                .is_err()
        );
    }

    #[test]
    fn test_recent_signals() {
        let mut tracker = ExecutionError::new();

        for _ in 0..5 {
            tracker.record(&perfect_fill()).unwrap();
        }

        let recent = tracker.recent_signals(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_config_validation_bad_weights() {
        let tracker = ExecutionError::with_config(ExecutionErrorConfig {
            slippage_weight: 0.5,
            timing_weight: 0.5,
            fill_rate_weight: 0.5, // sum = 1.5
            ..Default::default()
        });
        assert!(tracker.process().is_err());
    }

    #[test]
    fn test_config_validation_bad_ema() {
        let tracker = ExecutionError::with_config(ExecutionErrorConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(tracker.process().is_err());
    }

    #[test]
    fn test_zero_planned_duration() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.0,
            intended_quantity: 1000.0,
            filled_quantity: 1000.0,
            planned_duration_secs: 0.0,
            actual_duration_secs: 10.0,
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        // Timing error should be 0 when planned is 0
        assert!((signals.timing_error).abs() < 1e-12);
    }

    #[test]
    fn test_overfill_clamped() {
        let mut tracker = ExecutionError::new();
        let obs = ExecutionObservation {
            arrival_price: 100.0,
            fill_price: 100.0,
            intended_quantity: 1000.0,
            filled_quantity: 1200.0, // Overfill — should not produce negative shortfall
            planned_duration_secs: 60.0,
            actual_duration_secs: 60.0,
            side: OrderSide::Buy,
        };
        let signals = tracker.record(&obs).unwrap();
        assert!((signals.fill_shortfall).abs() < 1e-12);
    }
}
