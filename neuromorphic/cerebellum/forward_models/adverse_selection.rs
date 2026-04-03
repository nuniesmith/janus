//! Adverse selection forward model — predict toxic order flow
//!
//! Part of the Cerebellum region
//! Component: forward_models
//!
//! Predicts the probability and magnitude of adverse selection (informed
//! trading against our resting orders) using a VPIN-inspired toxicity
//! scoring framework with adaptive calibration from realized post-fill
//! price movements.
//!
//! Key features:
//! - Volume-synchronised toxicity metric (VPIN-inspired)
//! - Trade imbalance tracking across configurable volume buckets
//! - Post-fill adverse move measurement for calibration
//! - EMA-smoothed toxicity signal with configurable decay
//! - Regime detection (normal vs elevated vs toxic)
//! - Running statistics with accuracy tracking

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the adverse selection model
#[derive(Debug, Clone)]
pub struct AdverseSelectionConfig {
    /// Number of volume buckets for VPIN calculation
    pub num_buckets: usize,
    /// Volume per bucket (in base currency units)
    pub bucket_volume: f64,
    /// EMA decay for smoothed toxicity signal (0 < decay < 1)
    pub ema_decay: f64,
    /// Threshold for "elevated" toxicity regime
    pub elevated_threshold: f64,
    /// Threshold for "toxic" regime (should be > elevated_threshold)
    pub toxic_threshold: f64,
    /// Minimum observations before adaptive calibration activates
    pub min_samples: usize,
    /// Maximum number of post-fill observations in the sliding window
    pub window_size: usize,
    /// Blend weight for adaptive adjustment (0 = static only, 1 = fully adaptive)
    pub adaptation_weight: f64,
    /// Time horizon (in seconds) for measuring post-fill adverse moves
    pub adverse_horizon_secs: f64,
    /// Minimum adverse move (in bps) to count as "adversely selected"
    pub adverse_threshold_bps: f64,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            num_buckets: 50,
            bucket_volume: 1000.0,
            ema_decay: 0.94,
            elevated_threshold: 0.4,
            toxic_threshold: 0.7,
            min_samples: 15,
            window_size: 500,
            adaptation_weight: 0.3,
            adverse_horizon_secs: 5.0,
            adverse_threshold_bps: 2.0,
        }
    }
}

/// Side of a trade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// A trade event used to update the VPIN buckets
#[derive(Debug, Clone)]
pub struct TradeUpdate {
    /// Side of the trade (aggressor)
    pub side: TradeSide,
    /// Volume of the trade (base currency)
    pub volume: f64,
    /// Price at which the trade occurred
    pub price: f64,
}

/// Toxicity regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityRegime {
    /// Normal market conditions — low adverse selection risk
    Normal,
    /// Elevated toxicity — caution warranted
    Elevated,
    /// Toxic flow detected — high probability of informed trading
    Toxic,
}

/// Result of a toxicity assessment
#[derive(Debug, Clone)]
pub struct ToxicityEstimate {
    /// Raw VPIN toxicity score [0, 1]
    pub vpin: f64,
    /// EMA-smoothed toxicity score [0, 1]
    pub smoothed_toxicity: f64,
    /// Current regime classification
    pub regime: ToxicityRegime,
    /// Adaptive adverse selection probability [0, 1]
    pub adverse_probability: f64,
    /// Whether the model is using adaptive calibration
    pub adapted: bool,
    /// Confidence in the estimate
    pub confidence: f64,
    /// Current buy-sell imbalance fraction [-1, 1]
    pub imbalance: f64,
}

/// Observation of a post-fill price move for calibration
#[derive(Debug, Clone)]
pub struct AdverseObservation {
    /// Toxicity score at the time the fill occurred
    pub toxicity_at_fill: f64,
    /// Price at fill
    pub fill_price: f64,
    /// Price after the adverse horizon
    pub price_after: f64,
    /// Side we were filled on (our side, not aggressor)
    pub our_side: TradeSide,
}

impl AdverseObservation {
    /// Adverse move in basis points (positive = adverse, negative = favorable)
    pub fn adverse_move_bps(&self) -> f64 {
        if self.fill_price <= 0.0 {
            return 0.0;
        }
        let raw = (self.price_after - self.fill_price) / self.fill_price * 10_000.0;
        match self.our_side {
            // If we bought and price dropped, that's adverse
            TradeSide::Buy => -raw,
            // If we sold and price rose, that's adverse
            TradeSide::Sell => raw,
        }
    }

    /// Whether this observation counts as adversely selected
    pub fn is_adverse(&self, threshold_bps: f64) -> bool {
        self.adverse_move_bps() >= threshold_bps
    }
}

/// Running statistics for the adverse selection model
#[derive(Debug, Clone, Default)]
pub struct AdverseSelectionStats {
    /// Total trade updates processed
    pub trade_updates: usize,
    /// Total post-fill observations
    pub observations: usize,
    /// Count of observations classified as adversely selected
    pub adverse_count: usize,
    /// Sum of adverse moves (bps) for adverse observations only
    pub sum_adverse_bps: f64,
    /// Sum of all post-fill moves (bps, signed)
    pub sum_all_moves_bps: f64,
    /// Sum of squared post-fill moves
    pub sum_sq_moves_bps: f64,
    /// Maximum adverse move observed (bps)
    pub max_adverse_bps: f64,
    /// Sum of squared prediction errors (toxicity vs binary adverse outcome)
    pub sum_sq_pred_error: f64,
    /// Number of times regime was Toxic
    pub toxic_regime_count: usize,
    /// Number of times regime was Elevated
    pub elevated_regime_count: usize,
    /// Peak VPIN value observed
    pub peak_vpin: f64,
    /// Count of completed volume buckets
    pub completed_buckets: usize,
}

impl AdverseSelectionStats {
    /// Adverse selection rate: fraction of fills that were adversely selected
    pub fn adverse_rate(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.adverse_count as f64 / self.observations as f64
    }

    /// Mean adverse move for adversely-selected fills (bps)
    pub fn mean_adverse_bps(&self) -> f64 {
        if self.adverse_count == 0 {
            return 0.0;
        }
        self.sum_adverse_bps / self.adverse_count as f64
    }

    /// Mean post-fill move across all observations (bps)
    pub fn mean_move_bps(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_all_moves_bps / self.observations as f64
    }

    /// Variance of post-fill moves
    pub fn move_variance(&self) -> f64 {
        if self.observations < 2 {
            return 0.0;
        }
        let mean = self.mean_move_bps();
        self.sum_sq_moves_bps / self.observations as f64 - mean * mean
    }

    /// Standard deviation of post-fill moves
    pub fn move_std(&self) -> f64 {
        self.move_variance().max(0.0).sqrt()
    }

    /// Brier-like score for toxicity predictions (lower = better)
    pub fn prediction_score(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_sq_pred_error / self.observations as f64
    }
}

/// Internal volume bucket for VPIN calculation
#[derive(Debug, Clone)]
struct VolumeBucket {
    buy_volume: f64,
    sell_volume: f64,
}

impl VolumeBucket {
    fn total(&self) -> f64 {
        self.buy_volume + self.sell_volume
    }

    fn imbalance(&self) -> f64 {
        let total = self.total();
        if total <= 0.0 {
            return 0.0;
        }
        (self.buy_volume - self.sell_volume).abs() / total
    }
}

/// Adverse selection forward model
///
/// Uses VPIN-inspired volume-synchronised toxicity scoring combined with
/// adaptive calibration from realized post-fill adverse moves to predict
/// the probability and severity of adverse selection on resting orders.
pub struct AdverseSelection {
    config: AdverseSelectionConfig,
    /// Completed volume buckets (ring buffer, up to num_buckets)
    buckets: VecDeque<VolumeBucket>,
    /// Current in-progress bucket
    current_bucket: VolumeBucket,
    /// EMA-smoothed toxicity signal
    ema_toxicity: f64,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// EMA of calibration offset (predicted toxicity - actual adverse rate)
    ema_calibration_offset: f64,
    /// Whether calibration EMA has been initialized
    calibration_initialized: bool,
    /// Recent post-fill observations
    recent: VecDeque<AdverseObservation>,
    /// Total observation count
    observation_count: usize,
    /// Running statistics
    stats: AdverseSelectionStats,
}

impl Default for AdverseSelection {
    fn default() -> Self {
        Self::new()
    }
}

impl AdverseSelection {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(AdverseSelectionConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: AdverseSelectionConfig) -> Self {
        Self {
            current_bucket: VolumeBucket {
                buy_volume: 0.0,
                sell_volume: 0.0,
            },
            buckets: VecDeque::with_capacity(config.num_buckets),
            ema_toxicity: 0.0,
            ema_initialized: false,
            ema_calibration_offset: 0.0,
            calibration_initialized: false,
            recent: VecDeque::new(),
            observation_count: 0,
            stats: AdverseSelectionStats::default(),
            config,
        }
    }

    /// Main processing function — validates configuration
    pub fn process(&self) -> Result<()> {
        if self.config.num_buckets == 0 {
            return Err(Error::InvalidInput("num_buckets must be > 0".into()));
        }
        if self.config.bucket_volume <= 0.0 {
            return Err(Error::InvalidInput("bucket_volume must be > 0".into()));
        }
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.elevated_threshold < 0.0 || self.config.elevated_threshold > 1.0 {
            return Err(Error::InvalidInput(
                "elevated_threshold must be in [0, 1]".into(),
            ));
        }
        if self.config.toxic_threshold < self.config.elevated_threshold {
            return Err(Error::InvalidInput(
                "toxic_threshold must be >= elevated_threshold".into(),
            ));
        }
        if self.config.adaptation_weight < 0.0 || self.config.adaptation_weight > 1.0 {
            return Err(Error::InvalidInput(
                "adaptation_weight must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Ingest a trade event and update VPIN buckets
    ///
    /// A single trade may span multiple buckets if its volume exceeds
    /// the remaining capacity of the current bucket.
    pub fn update(&mut self, trade: &TradeUpdate) -> Result<()> {
        if trade.volume <= 0.0 {
            return Err(Error::InvalidInput("trade volume must be > 0".into()));
        }
        if trade.price <= 0.0 {
            return Err(Error::InvalidInput("trade price must be > 0".into()));
        }

        self.stats.trade_updates += 1;

        let mut remaining = trade.volume;
        while remaining > 0.0 {
            let bucket_remaining = self.config.bucket_volume - self.current_bucket.total();
            let fill_amount = remaining.min(bucket_remaining);

            match trade.side {
                TradeSide::Buy => self.current_bucket.buy_volume += fill_amount,
                TradeSide::Sell => self.current_bucket.sell_volume += fill_amount,
            }
            remaining -= fill_amount;

            // Check if bucket is complete
            if self.current_bucket.total() >= self.config.bucket_volume - 1e-12 {
                let completed = std::mem::replace(
                    &mut self.current_bucket,
                    VolumeBucket {
                        buy_volume: 0.0,
                        sell_volume: 0.0,
                    },
                );
                self.buckets.push_back(completed);
                self.stats.completed_buckets += 1;

                // Maintain ring buffer size
                while self.buckets.len() > self.config.num_buckets {
                    self.buckets.pop_front();
                }

                // Update EMA with new VPIN reading
                let vpin = self.compute_raw_vpin();
                if vpin > self.stats.peak_vpin {
                    self.stats.peak_vpin = vpin;
                }
                self.update_ema(vpin);
            }
        }

        Ok(())
    }

    /// Get the current toxicity estimate
    pub fn estimate(&self) -> ToxicityEstimate {
        let vpin = self.compute_raw_vpin();
        let smoothed = if self.ema_initialized {
            self.ema_toxicity
        } else {
            vpin
        };

        let regime = self.classify_regime(smoothed);

        // Adaptive adverse probability
        let adapted = self.is_adapted();
        let adverse_probability = if adapted {
            let offset = self.ema_calibration_offset * self.config.adaptation_weight;
            (smoothed - offset).clamp(0.0, 1.0)
        } else {
            smoothed
        };

        let confidence = self.compute_confidence();
        let imbalance = self.compute_imbalance();

        // Update regime counters (we track in estimate since it's read-only
        // and stats are updated elsewhere for observations)
        ToxicityEstimate {
            vpin,
            smoothed_toxicity: smoothed,
            regime,
            adverse_probability,
            adapted,
            confidence,
            imbalance,
        }
    }

    /// Record a post-fill observation for adaptive calibration
    pub fn observe(&mut self, obs: AdverseObservation) {
        let adverse_bps = obs.adverse_move_bps();
        let is_adverse = obs.is_adverse(self.config.adverse_threshold_bps);
        let toxicity_at_fill = obs.toxicity_at_fill.clamp(0.0, 1.0);
        let actual = if is_adverse { 1.0 } else { 0.0 };

        // Update calibration EMA
        let error = toxicity_at_fill - actual;
        if self.calibration_initialized {
            self.ema_calibration_offset = self.config.ema_decay * self.ema_calibration_offset
                + (1.0 - self.config.ema_decay) * error;
        } else {
            self.ema_calibration_offset = error;
            self.calibration_initialized = true;
        }

        // Update prediction error stats
        self.stats.sum_sq_pred_error += error * error;

        // Update adverse stats
        self.stats.observations += 1;
        self.stats.sum_all_moves_bps += adverse_bps;
        self.stats.sum_sq_moves_bps += adverse_bps * adverse_bps;

        if is_adverse {
            self.stats.adverse_count += 1;
            self.stats.sum_adverse_bps += adverse_bps;
            if adverse_bps > self.stats.max_adverse_bps {
                self.stats.max_adverse_bps = adverse_bps;
            }
        }

        // Update regime stats based on current smoothed toxicity
        let regime = self.classify_regime(if self.ema_initialized {
            self.ema_toxicity
        } else {
            0.0
        });
        match regime {
            ToxicityRegime::Toxic => self.stats.toxic_regime_count += 1,
            ToxicityRegime::Elevated => self.stats.elevated_regime_count += 1,
            ToxicityRegime::Normal => {}
        }

        self.observation_count += 1;

        // Maintain sliding window
        self.recent.push_back(obs);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }
    }

    /// Compute raw VPIN from completed buckets
    ///
    /// VPIN = mean(|buy_volume - sell_volume| / total_volume) across buckets
    fn compute_raw_vpin(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        let sum_imbalance: f64 = self.buckets.iter().map(|b| b.imbalance()).sum();
        sum_imbalance / self.buckets.len() as f64
    }

    /// Compute signed buy-sell imbalance across recent buckets
    fn compute_imbalance(&self) -> f64 {
        if self.buckets.is_empty() {
            // Include current partial bucket
            let total = self.current_bucket.total();
            if total <= 0.0 {
                return 0.0;
            }
            return (self.current_bucket.buy_volume - self.current_bucket.sell_volume) / total;
        }
        let total_buy: f64 = self.buckets.iter().map(|b| b.buy_volume).sum();
        let total_sell: f64 = self.buckets.iter().map(|b| b.sell_volume).sum();
        let total = total_buy + total_sell;
        if total <= 0.0 {
            return 0.0;
        }
        (total_buy - total_sell) / total
    }

    /// Classify toxicity regime
    fn classify_regime(&self, toxicity: f64) -> ToxicityRegime {
        if toxicity >= self.config.toxic_threshold {
            ToxicityRegime::Toxic
        } else if toxicity >= self.config.elevated_threshold {
            ToxicityRegime::Elevated
        } else {
            ToxicityRegime::Normal
        }
    }

    /// Update EMA with a new VPIN reading
    fn update_ema(&mut self, vpin: f64) {
        if self.ema_initialized {
            self.ema_toxicity =
                self.config.ema_decay * self.ema_toxicity + (1.0 - self.config.ema_decay) * vpin;
        } else {
            self.ema_toxicity = vpin;
            self.ema_initialized = true;
        }
    }

    /// Whether adaptive calibration is active
    pub fn is_adapted(&self) -> bool {
        self.calibration_initialized && self.observation_count >= self.config.min_samples
    }

    /// Current smoothed toxicity value
    pub fn smoothed_toxicity(&self) -> f64 {
        if self.ema_initialized {
            self.ema_toxicity
        } else {
            0.0
        }
    }

    /// Current calibration offset
    pub fn calibration_offset(&self) -> f64 {
        self.ema_calibration_offset
    }

    /// Total post-fill observations recorded
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Number of completed volume buckets
    pub fn completed_bucket_count(&self) -> usize {
        self.stats.completed_buckets
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &AdverseSelectionStats {
        &self.stats
    }

    /// Recent post-fill observations in the sliding window
    pub fn recent_observations(&self) -> &VecDeque<AdverseObservation> {
        &self.recent
    }

    /// Windowed adverse rate (fraction of recent fills adversely selected)
    pub fn windowed_adverse_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let adverse_count = self
            .recent
            .iter()
            .filter(|o| o.is_adverse(self.config.adverse_threshold_bps))
            .count();
        adverse_count as f64 / self.recent.len() as f64
    }

    /// Windowed mean adverse move (bps)
    pub fn windowed_mean_adverse_bps(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|o| o.adverse_move_bps()).sum();
        sum / self.recent.len() as f64
    }

    /// Check if adverse selection is getting worse over the window
    pub fn is_worsening(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;

        let first_half_rate = self
            .recent
            .iter()
            .take(mid)
            .filter(|o| o.is_adverse(self.config.adverse_threshold_bps))
            .count() as f64
            / mid as f64;

        let second_half_rate = self
            .recent
            .iter()
            .skip(mid)
            .filter(|o| o.is_adverse(self.config.adverse_threshold_bps))
            .count() as f64
            / (n - mid) as f64;

        // Worsening if second half adverse rate is >30% higher
        second_half_rate > first_half_rate * 1.3 && second_half_rate > 0.1
    }

    /// Compute confidence based on data available
    fn compute_confidence(&self) -> f64 {
        if self.buckets.is_empty() && !self.ema_initialized {
            return 0.0;
        }

        // Bucket fill confidence: how many of the desired buckets do we have
        let bucket_confidence =
            (self.buckets.len() as f64 / self.config.num_buckets as f64).min(1.0);

        // Observation confidence: ramps over min_samples * 2
        let obs_confidence = if self.observation_count == 0 {
            0.5 // Can still provide VPIN without calibration
        } else {
            0.5 + 0.5
                * (self.observation_count as f64 / (self.config.min_samples as f64 * 2.0)).min(1.0)
        };

        // Prediction quality confidence
        let quality_confidence = if self.observation_count > 0 {
            let pred_score = self.stats.prediction_score();
            // Score of 0 = perfect, 1 = worst; map to [0, 1] confidence
            (1.0 - pred_score).max(0.0)
        } else {
            0.5
        };

        // Geometric mean
        (bucket_confidence * obs_confidence * quality_confidence).cbrt()
    }

    /// Reset all state and statistics
    pub fn reset(&mut self) {
        self.buckets.clear();
        self.current_bucket = VolumeBucket {
            buy_volume: 0.0,
            sell_volume: 0.0,
        };
        self.ema_toxicity = 0.0;
        self.ema_initialized = false;
        self.ema_calibration_offset = 0.0;
        self.calibration_initialized = false;
        self.recent.clear();
        self.observation_count = 0;
        self.stats = AdverseSelectionStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> AdverseSelectionConfig {
        AdverseSelectionConfig {
            num_buckets: 5,
            bucket_volume: 100.0,
            min_samples: 3,
            ..Default::default()
        }
    }

    fn buy_trade(volume: f64) -> TradeUpdate {
        TradeUpdate {
            side: TradeSide::Buy,
            volume,
            price: 100.0,
        }
    }

    fn sell_trade(volume: f64) -> TradeUpdate {
        TradeUpdate {
            side: TradeSide::Sell,
            volume,
            price: 100.0,
        }
    }

    #[test]
    fn test_basic() {
        let instance = AdverseSelection::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_empty_estimate() {
        let model = AdverseSelection::new();
        let est = model.estimate();
        assert_eq!(est.vpin, 0.0);
        assert_eq!(est.regime, ToxicityRegime::Normal);
        assert!(!est.adapted);
    }

    #[test]
    fn test_balanced_flow_low_toxicity() {
        let mut model = AdverseSelection::with_config(small_config());

        // Perfectly balanced flow: alternate buy and sell
        for _ in 0..10 {
            model.update(&buy_trade(50.0)).unwrap();
            model.update(&sell_trade(50.0)).unwrap();
        }

        let est = model.estimate();
        // Balanced flow should produce low VPIN
        assert!(
            est.vpin < 0.2,
            "balanced flow should have low vpin, got {}",
            est.vpin
        );
        assert_eq!(est.regime, ToxicityRegime::Normal);
    }

    #[test]
    fn test_one_sided_flow_high_toxicity() {
        let mut model = AdverseSelection::with_config(small_config());

        // All buys — maximally imbalanced
        for _ in 0..10 {
            model.update(&buy_trade(100.0)).unwrap();
        }

        let est = model.estimate();
        assert!(
            est.vpin > 0.8,
            "one-sided flow should have high vpin, got {}",
            est.vpin
        );
    }

    #[test]
    fn test_regime_classification() {
        let model = AdverseSelection::with_config(AdverseSelectionConfig {
            elevated_threshold: 0.4,
            toxic_threshold: 0.7,
            ..Default::default()
        });

        assert_eq!(model.classify_regime(0.2), ToxicityRegime::Normal);
        assert_eq!(model.classify_regime(0.5), ToxicityRegime::Elevated);
        assert_eq!(model.classify_regime(0.8), ToxicityRegime::Toxic);
        assert_eq!(model.classify_regime(0.4), ToxicityRegime::Elevated);
        assert_eq!(model.classify_regime(0.7), ToxicityRegime::Toxic);
    }

    #[test]
    fn test_bucket_completion() {
        let mut model = AdverseSelection::with_config(small_config());

        // Should complete 1 bucket with 100 units of volume
        model.update(&buy_trade(100.0)).unwrap();
        assert_eq!(model.completed_bucket_count(), 1);

        // Another 50 should not complete a second
        model.update(&sell_trade(50.0)).unwrap();
        assert_eq!(model.completed_bucket_count(), 1);

        // 50 more should complete second bucket
        model.update(&buy_trade(50.0)).unwrap();
        assert_eq!(model.completed_bucket_count(), 2);
    }

    #[test]
    fn test_large_trade_spans_multiple_buckets() {
        let mut model = AdverseSelection::with_config(small_config());

        // Single trade of 350 should complete 3 buckets (bucket_volume=100)
        model.update(&buy_trade(350.0)).unwrap();
        assert_eq!(model.completed_bucket_count(), 3);
    }

    #[test]
    fn test_bucket_ring_buffer_capped() {
        let mut model = AdverseSelection::with_config(small_config());

        // Complete 10 buckets but ring buffer should only hold 5
        for _ in 0..10 {
            model.update(&buy_trade(100.0)).unwrap();
        }
        assert_eq!(model.buckets.len(), 5);
        assert_eq!(model.completed_bucket_count(), 10);
    }

    #[test]
    fn test_adverse_observation_buy_adverse() {
        let obs = AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 99.0,
            our_side: TradeSide::Buy,
        };
        // Bought at 100, price dropped to 99 → adverse move of 100 bps
        assert!((obs.adverse_move_bps() - 100.0).abs() < 1e-6);
        assert!(obs.is_adverse(2.0));
    }

    #[test]
    fn test_adverse_observation_sell_adverse() {
        let obs = AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 101.0,
            our_side: TradeSide::Sell,
        };
        // Sold at 100, price rose to 101 → adverse move of 100 bps
        assert!((obs.adverse_move_bps() - 100.0).abs() < 1e-6);
        assert!(obs.is_adverse(2.0));
    }

    #[test]
    fn test_adverse_observation_favorable() {
        let obs = AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 101.0,
            our_side: TradeSide::Buy,
        };
        // Bought at 100, price rose to 101 → favorable (negative adverse bps)
        assert!(obs.adverse_move_bps() < 0.0);
        assert!(!obs.is_adverse(2.0));
    }

    #[test]
    fn test_adverse_observation_zero_fill_price() {
        let obs = AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 0.0,
            price_after: 100.0,
            our_side: TradeSide::Buy,
        };
        assert_eq!(obs.adverse_move_bps(), 0.0);
    }

    #[test]
    fn test_observe_updates_stats() {
        let mut model = AdverseSelection::with_config(small_config());

        model.observe(AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 99.0,
            our_side: TradeSide::Buy,
        });

        assert_eq!(model.stats().observations, 1);
        assert_eq!(model.stats().adverse_count, 1);
        assert!(model.stats().sum_adverse_bps > 0.0);
    }

    #[test]
    fn test_not_adapted_below_min_samples() {
        let mut model = AdverseSelection::with_config(small_config());
        for _ in 0..model.config.min_samples - 1 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 99.5,
                our_side: TradeSide::Buy,
            });
        }
        assert!(!model.is_adapted());
    }

    #[test]
    fn test_adapted_at_min_samples() {
        let mut model = AdverseSelection::with_config(small_config());
        for _ in 0..model.config.min_samples {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 99.5,
                our_side: TradeSide::Buy,
            });
        }
        assert!(model.is_adapted());
    }

    #[test]
    fn test_adaptive_calibration() {
        let mut model = AdverseSelection::with_config(AdverseSelectionConfig {
            min_samples: 3,
            adaptation_weight: 0.5,
            num_buckets: 5,
            bucket_volume: 100.0,
            ..Default::default()
        });

        // Fill some buckets to get a non-zero VPIN
        for _ in 0..10 {
            model.update(&buy_trade(100.0)).unwrap();
        }

        let est_before = model.estimate();

        // Observe that toxicity predictions are consistently too high
        // (predict high toxicity but no adverse moves)
        for _ in 0..20 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.8,
                fill_price: 100.0,
                price_after: 100.0, // no adverse move
                our_side: TradeSide::Buy,
            });
        }

        let est_after = model.estimate();
        assert!(est_after.adapted);
        // Calibration should lower the adverse probability
        assert!(
            est_after.adverse_probability <= est_before.adverse_probability + 0.01,
            "calibration should reduce adverse prob: before={}, after={}",
            est_before.adverse_probability,
            est_after.adverse_probability
        );
    }

    #[test]
    fn test_stats_adverse_rate() {
        let mut model = AdverseSelection::with_config(small_config());

        // 2 adverse, 1 favorable
        model.observe(AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 99.0,
            our_side: TradeSide::Buy,
        });
        model.observe(AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 98.0,
            our_side: TradeSide::Buy,
        });
        model.observe(AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 101.0, // favorable
            our_side: TradeSide::Buy,
        });

        assert!((model.stats().adverse_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_max_adverse_tracked() {
        let mut model = AdverseSelection::with_config(small_config());

        model.observe(AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 99.0,
            our_side: TradeSide::Buy,
        });
        model.observe(AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 97.0, // 300 bps adverse
            our_side: TradeSide::Buy,
        });

        assert!((model.stats().max_adverse_bps - 300.0).abs() < 1e-6);
    }

    #[test]
    fn test_windowed_adverse_rate() {
        let mut model = AdverseSelection::with_config(AdverseSelectionConfig {
            window_size: 4,
            ..small_config()
        });

        // 2 adverse, 2 favorable
        for price_after in &[99.0, 101.0, 98.0, 102.0] {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: *price_after,
                our_side: TradeSide::Buy,
            });
        }

        assert!((model.windowed_adverse_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_is_worsening() {
        let mut model = AdverseSelection::with_config(AdverseSelectionConfig {
            window_size: 20,
            adverse_threshold_bps: 2.0,
            ..small_config()
        });

        // First half: all favorable
        for _ in 0..10 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 101.0, // favorable buy
                our_side: TradeSide::Buy,
            });
        }
        // Second half: all adverse
        for _ in 0..10 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 99.0, // adverse buy
                our_side: TradeSide::Buy,
            });
        }

        assert!(model.is_worsening());
    }

    #[test]
    fn test_not_worsening_consistent() {
        let mut model = AdverseSelection::with_config(AdverseSelectionConfig {
            window_size: 20,
            ..small_config()
        });

        for _ in 0..20 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 101.0,
                our_side: TradeSide::Buy,
            });
        }

        assert!(!model.is_worsening());
    }

    #[test]
    fn test_not_worsening_insufficient_data() {
        let mut model = AdverseSelection::with_config(small_config());
        for _ in 0..4 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 99.0,
                our_side: TradeSide::Buy,
            });
        }
        assert!(!model.is_worsening());
    }

    #[test]
    fn test_imbalance_balanced() {
        let mut model = AdverseSelection::with_config(small_config());
        model.update(&buy_trade(100.0)).unwrap();
        model.update(&sell_trade(100.0)).unwrap();

        let est = model.estimate();
        assert!(
            est.imbalance.abs() < 1e-10,
            "balanced flow should have ~0 imbalance, got {}",
            est.imbalance
        );
    }

    #[test]
    fn test_imbalance_buy_dominant() {
        let mut model = AdverseSelection::with_config(small_config());
        // 3 buy buckets, 1 sell bucket
        model.update(&buy_trade(300.0)).unwrap();
        model.update(&sell_trade(100.0)).unwrap();

        let est = model.estimate();
        assert!(
            est.imbalance > 0.0,
            "buy-dominant flow should have positive imbalance"
        );
    }

    #[test]
    fn test_imbalance_sell_dominant() {
        let mut model = AdverseSelection::with_config(small_config());
        model.update(&sell_trade(300.0)).unwrap();
        model.update(&buy_trade(100.0)).unwrap();

        let est = model.estimate();
        assert!(
            est.imbalance < 0.0,
            "sell-dominant flow should have negative imbalance, got {}",
            est.imbalance
        );
    }

    #[test]
    fn test_reset() {
        let mut model = AdverseSelection::with_config(small_config());

        for _ in 0..10 {
            model.update(&buy_trade(100.0)).unwrap();
        }
        model.observe(AdverseObservation {
            toxicity_at_fill: 0.5,
            fill_price: 100.0,
            price_after: 99.0,
            our_side: TradeSide::Buy,
        });

        assert!(model.completed_bucket_count() > 0);
        assert!(model.observation_count() > 0);

        model.reset();

        assert_eq!(model.completed_bucket_count(), 0);
        assert_eq!(model.observation_count(), 0);
        assert_eq!(model.stats().observations, 0);
        assert!(model.recent_observations().is_empty());
        assert!(!model.is_adapted());
        assert_eq!(model.smoothed_toxicity(), 0.0);
    }

    #[test]
    fn test_invalid_zero_volume_trade() {
        let mut model = AdverseSelection::new();
        assert!(
            model
                .update(&TradeUpdate {
                    side: TradeSide::Buy,
                    volume: 0.0,
                    price: 100.0,
                })
                .is_err()
        );
    }

    #[test]
    fn test_invalid_negative_price_trade() {
        let mut model = AdverseSelection::new();
        assert!(
            model
                .update(&TradeUpdate {
                    side: TradeSide::Buy,
                    volume: 10.0,
                    price: -1.0,
                })
                .is_err()
        );
    }

    #[test]
    fn test_invalid_config_zero_buckets() {
        let model = AdverseSelection::with_config(AdverseSelectionConfig {
            num_buckets: 0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_bucket_volume() {
        let model = AdverseSelection::with_config(AdverseSelectionConfig {
            bucket_volume: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let model = AdverseSelection::with_config(AdverseSelectionConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_threshold_order() {
        let model = AdverseSelection::with_config(AdverseSelectionConfig {
            elevated_threshold: 0.7,
            toxic_threshold: 0.4, // less than elevated
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_adaptation_weight() {
        let model = AdverseSelection::with_config(AdverseSelectionConfig {
            adaptation_weight: -0.1,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_stats_defaults() {
        let stats = AdverseSelectionStats::default();
        assert_eq!(stats.adverse_rate(), 0.0);
        assert_eq!(stats.mean_adverse_bps(), 0.0);
        assert_eq!(stats.mean_move_bps(), 0.0);
        assert_eq!(stats.move_variance(), 0.0);
        assert_eq!(stats.move_std(), 0.0);
        assert_eq!(stats.prediction_score(), 0.0);
    }

    #[test]
    fn test_confidence_zero_no_data() {
        let model = AdverseSelection::new();
        let est = model.estimate();
        assert!((est.confidence - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases_with_data() {
        let mut model = AdverseSelection::with_config(small_config());

        // Add some balanced data
        for _ in 0..10 {
            model.update(&buy_trade(50.0)).unwrap();
            model.update(&sell_trade(50.0)).unwrap();
        }

        let est = model.estimate();
        assert!(
            est.confidence > 0.0,
            "confidence should be > 0 with data, got {}",
            est.confidence
        );
    }

    #[test]
    fn test_ema_smoothing_dampens_spikes() {
        let mut model = AdverseSelection::with_config(small_config());

        // Build up balanced history
        for _ in 0..20 {
            model.update(&buy_trade(50.0)).unwrap();
            model.update(&sell_trade(50.0)).unwrap();
        }
        let baseline = model.smoothed_toxicity();

        // Single spike of one-sided flow
        model.update(&buy_trade(100.0)).unwrap();
        let after_spike = model.smoothed_toxicity();

        // EMA should dampen: smoothed should be closer to baseline than raw
        let raw_vpin = model.estimate().vpin;
        let smoothed_change = (after_spike - baseline).abs();
        let raw_change = (raw_vpin - baseline).abs();

        // Smoothed change should be less than or equal to raw change
        assert!(
            smoothed_change <= raw_change + 1e-10,
            "EMA should dampen: smoothed_change={}, raw_change={}",
            smoothed_change,
            raw_change
        );
    }

    #[test]
    fn test_trade_update_count() {
        let mut model = AdverseSelection::with_config(small_config());

        model.update(&buy_trade(50.0)).unwrap();
        model.update(&sell_trade(30.0)).unwrap();
        model.update(&buy_trade(20.0)).unwrap();

        assert_eq!(model.stats().trade_updates, 3);
    }

    #[test]
    fn test_move_variance() {
        let mut model = AdverseSelection::with_config(small_config());

        // All same adverse move → zero variance
        for _ in 0..5 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 99.0,
                our_side: TradeSide::Buy,
            });
        }

        assert!(
            model.stats().move_variance().abs() < 1e-6,
            "constant moves should have ~0 variance, got {}",
            model.stats().move_variance()
        );
    }

    #[test]
    fn test_window_eviction() {
        let mut model = AdverseSelection::with_config(AdverseSelectionConfig {
            window_size: 3,
            ..small_config()
        });

        for _ in 0..10 {
            model.observe(AdverseObservation {
                toxicity_at_fill: 0.5,
                fill_price: 100.0,
                price_after: 99.0,
                our_side: TradeSide::Buy,
            });
        }

        assert_eq!(model.recent_observations().len(), 3);
        assert_eq!(model.observation_count(), 10);
    }
}
