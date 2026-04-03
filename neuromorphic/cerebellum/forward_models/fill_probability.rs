//! Fill probability prediction for limit orders
//!
//! Part of the Cerebellum region
//! Component: forward_models
//!
//! Predicts the probability that a limit order will be filled within a
//! given time horizon based on observable market microstructure features.
//! Uses a logistic-regression-inspired scoring model with adaptive
//! calibration from observed fill outcomes.
//!
//! Key features:
//! - Feature-based fill scoring (queue position, spread, volatility, time)
//! - Logistic transform for bounded [0, 1] probability output
//! - EMA-smoothed calibration from realized fill outcomes
//! - Brier score tracking for prediction quality assessment
//! - Configurable feature weights with sensible defaults

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the fill probability model
#[derive(Debug, Clone)]
pub struct FillProbabilityConfig {
    /// Weight for queue-position feature (higher = queue position matters more)
    pub weight_queue_position: f64,
    /// Weight for spread feature (wider spread → lower fill probability)
    pub weight_spread: f64,
    /// Weight for volatility feature (higher vol → more fills)
    pub weight_volatility: f64,
    /// Weight for time-in-force feature (longer exposure → more fills)
    pub weight_time: f64,
    /// Bias term in the logistic model
    pub bias: f64,
    /// EMA decay factor for adaptive calibration (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum observations before adaptive calibration activates
    pub min_samples: usize,
    /// Maximum number of observations in the sliding window
    pub window_size: usize,
    /// Blend weight for adaptive vs static model (0 = all static, 1 = all adaptive)
    pub adaptation_weight: f64,
}

impl Default for FillProbabilityConfig {
    fn default() -> Self {
        Self {
            weight_queue_position: -1.5,
            weight_spread: -0.8,
            weight_volatility: 1.2,
            weight_time: 0.6,
            bias: 0.5,
            ema_decay: 0.95,
            min_samples: 20,
            window_size: 500,
            adaptation_weight: 0.3,
        }
    }
}

/// Observable features for a pending limit order
#[derive(Debug, Clone)]
pub struct FillFeatures {
    /// Normalized queue position: 0.0 = front of queue, 1.0 = back of queue
    pub queue_position: f64,
    /// Spread as fraction of mid price (e.g. 0.001 = 10 bps)
    pub spread_fraction: f64,
    /// Recent realized volatility (annualized, e.g. 0.20 = 20%)
    pub volatility: f64,
    /// Time remaining in force as fraction of total horizon (1.0 = just submitted, 0.0 = expiring)
    pub time_remaining: f64,
}

/// Result of a fill probability prediction
#[derive(Debug, Clone)]
pub struct FillEstimate {
    /// Predicted fill probability [0, 1]
    pub probability: f64,
    /// Raw logit score before sigmoid
    pub logit: f64,
    /// Whether the model is using adaptive calibration
    pub adapted: bool,
    /// Confidence in the estimate (based on sample count and Brier score)
    pub confidence: f64,
}

/// Record of a realized fill outcome for calibration
#[derive(Debug, Clone)]
pub struct FillOutcome {
    /// The features at the time the order was placed
    pub features: FillFeatures,
    /// The probability that was predicted
    pub predicted_probability: f64,
    /// Whether the order was actually filled (1.0) or not (0.0)
    pub filled: f64,
}

/// Tracking statistics for the fill probability model
#[derive(Debug, Clone, Default)]
pub struct FillProbabilityStats {
    /// Total predictions made
    pub predictions: usize,
    /// Total outcomes observed
    pub observations: usize,
    /// Sum of squared prediction errors (Brier score numerator)
    pub sum_sq_error: f64,
    /// Sum of predicted probabilities (for calibration analysis)
    pub sum_predicted: f64,
    /// Sum of actual outcomes (for base rate)
    pub sum_actual: f64,
    /// Count of predictions above 0.5 that were filled
    pub true_positives: usize,
    /// Count of predictions above 0.5 that were NOT filled
    pub false_positives: usize,
    /// Count of predictions below 0.5 that were NOT filled
    pub true_negatives: usize,
    /// Count of predictions below 0.5 that were filled
    pub false_negatives: usize,
    /// Maximum observed Brier score for a single prediction
    pub max_single_error: f64,
}

impl FillProbabilityStats {
    /// Mean Brier score (lower is better, 0 = perfect, 0.25 = coin flip)
    pub fn brier_score(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_sq_error / self.observations as f64
    }

    /// Mean predicted probability
    pub fn mean_predicted(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_predicted / self.observations as f64
    }

    /// Actual fill rate (base rate)
    pub fn fill_rate(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_actual / self.observations as f64
    }

    /// Calibration error: |mean predicted - actual fill rate|
    pub fn calibration_error(&self) -> f64 {
        (self.mean_predicted() - self.fill_rate()).abs()
    }

    /// Precision: TP / (TP + FP)
    pub fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f64 / denom as f64
    }

    /// Recall: TP / (TP + FN)
    pub fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f64 / denom as f64
    }

    /// Accuracy: (TP + TN) / total
    pub fn accuracy(&self) -> f64 {
        let total =
            self.true_positives + self.true_negatives + self.false_positives + self.false_negatives;
        if total == 0 {
            return 0.0;
        }
        (self.true_positives + self.true_negatives) as f64 / total as f64
    }
}

/// Fill probability predictor for limit orders
///
/// Uses a logistic scoring model with adaptive calibration from observed
/// fill outcomes. The model combines queue position, spread, volatility,
/// and time features into a fill probability estimate.
pub struct FillProbability {
    config: FillProbabilityConfig,
    /// EMA of the calibration offset (predicted - actual)
    ema_offset: f64,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// Recent outcomes for windowed analysis
    recent: VecDeque<FillOutcome>,
    /// Total observation count (including those evicted from window)
    observation_count: usize,
    /// Running statistics
    stats: FillProbabilityStats,
}

impl Default for FillProbability {
    fn default() -> Self {
        Self::new()
    }
}

impl FillProbability {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(FillProbabilityConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: FillProbabilityConfig) -> Self {
        Self {
            config,
            ema_offset: 0.0,
            ema_initialized: false,
            recent: VecDeque::new(),
            observation_count: 0,
            stats: FillProbabilityStats::default(),
        }
    }

    /// Main processing function — validates config and returns Ok
    pub fn process(&self) -> Result<()> {
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.adaptation_weight < 0.0 || self.config.adaptation_weight > 1.0 {
            return Err(Error::InvalidInput(
                "adaptation_weight must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Predict the fill probability for a limit order given its features
    pub fn predict(&self, features: &FillFeatures) -> Result<FillEstimate> {
        // Validate inputs
        if features.queue_position < 0.0 || features.queue_position > 1.0 {
            return Err(Error::InvalidInput(
                "queue_position must be in [0, 1]".into(),
            ));
        }
        if features.spread_fraction < 0.0 {
            return Err(Error::InvalidInput("spread_fraction must be >= 0".into()));
        }
        if features.volatility < 0.0 {
            return Err(Error::InvalidInput("volatility must be >= 0".into()));
        }
        if features.time_remaining < 0.0 || features.time_remaining > 1.0 {
            return Err(Error::InvalidInput(
                "time_remaining must be in [0, 1]".into(),
            ));
        }

        // Compute raw logit score
        let logit = self.config.bias
            + self.config.weight_queue_position * features.queue_position
            + self.config.weight_spread * features.spread_fraction * 100.0 // scale to percentage
            + self.config.weight_volatility * features.volatility
            + self.config.weight_time * features.time_remaining;

        // Apply sigmoid to get base probability
        let base_prob = sigmoid(logit);

        // Apply adaptive calibration offset if available
        let adapted = self.is_adapted();
        let probability = if adapted {
            let offset = self.ema_offset * self.config.adaptation_weight;
            (base_prob - offset).clamp(0.0, 1.0)
        } else {
            base_prob
        };

        let confidence = self.compute_confidence();

        Ok(FillEstimate {
            probability,
            logit,
            adapted,
            confidence,
        })
    }

    /// Record a realized fill outcome for adaptive calibration
    pub fn observe(&mut self, outcome: FillOutcome) {
        let filled = outcome.filled.clamp(0.0, 1.0);
        let predicted = outcome.predicted_probability.clamp(0.0, 1.0);

        // Update EMA of calibration offset (predicted - actual)
        let error = predicted - filled;
        if self.ema_initialized {
            self.ema_offset =
                self.config.ema_decay * self.ema_offset + (1.0 - self.config.ema_decay) * error;
        } else {
            self.ema_offset = error;
            self.ema_initialized = true;
        }

        // Update stats
        let sq_error = (predicted - filled) * (predicted - filled);
        self.stats.sum_sq_error += sq_error;
        self.stats.sum_predicted += predicted;
        self.stats.sum_actual += filled;
        self.stats.observations += 1;
        if sq_error > self.stats.max_single_error {
            self.stats.max_single_error = sq_error;
        }

        // Classification stats
        let predicted_fill = predicted >= 0.5;
        let actual_fill = filled >= 0.5;
        match (predicted_fill, actual_fill) {
            (true, true) => self.stats.true_positives += 1,
            (true, false) => self.stats.false_positives += 1,
            (false, false) => self.stats.true_negatives += 1,
            (false, true) => self.stats.false_negatives += 1,
        }

        self.observation_count += 1;

        // Maintain sliding window
        self.recent.push_back(outcome);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }
    }

    /// Predict and immediately increment the prediction counter
    pub fn predict_tracked(&mut self, features: &FillFeatures) -> Result<FillEstimate> {
        let estimate = self.predict(features)?;
        self.stats.predictions += 1;
        Ok(estimate)
    }

    /// Whether the adaptive calibration has enough data
    pub fn is_adapted(&self) -> bool {
        self.ema_initialized && self.observation_count >= self.config.min_samples
    }

    /// Current EMA offset (predicted - actual bias)
    pub fn ema_offset(&self) -> f64 {
        self.ema_offset
    }

    /// Total observations recorded
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &FillProbabilityStats {
        &self.stats
    }

    /// Recent outcomes in the sliding window
    pub fn recent_outcomes(&self) -> &VecDeque<FillOutcome> {
        &self.recent
    }

    /// Windowed fill rate (actual fills in recent window)
    pub fn windowed_fill_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|o| o.filled).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed Brier score
    pub fn windowed_brier_score(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum_sq: f64 = self
            .recent
            .iter()
            .map(|o| {
                let diff = o.predicted_probability - o.filled;
                diff * diff
            })
            .sum();
        sum_sq / self.recent.len() as f64
    }

    /// Check if prediction quality is degrading (compare first vs second half of window)
    pub fn is_degrading(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;
        let first_half_brier: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|o| {
                let d = o.predicted_probability - o.filled;
                d * d
            })
            .sum::<f64>()
            / mid as f64;

        let second_half_brier: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|o| {
                let d = o.predicted_probability - o.filled;
                d * d
            })
            .sum::<f64>()
            / (n - mid) as f64;

        // Degrading if second half is materially worse (>20% worse)
        second_half_brier > first_half_brier * 1.2
    }

    /// Reset all adaptive state and statistics
    pub fn reset(&mut self) {
        self.ema_offset = 0.0;
        self.ema_initialized = false;
        self.recent.clear();
        self.observation_count = 0;
        self.stats = FillProbabilityStats::default();
    }

    /// Compute confidence based on sample count and Brier score
    fn compute_confidence(&self) -> f64 {
        if self.observation_count == 0 {
            return 0.0;
        }
        // Sample confidence: ramps up to 1.0 over min_samples * 3
        let sample_confidence =
            (self.observation_count as f64 / (self.config.min_samples as f64 * 3.0)).min(1.0);

        // Quality confidence: Brier score of 0.0 → 1.0, Brier of 0.25 → 0.0
        let brier = self.stats.brier_score();
        let quality_confidence = (1.0 - brier / 0.25).max(0.0);

        // Geometric mean of both factors
        (sample_confidence * quality_confidence).sqrt()
    }
}

/// Standard logistic sigmoid function
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let exp_neg = (-x).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = x.exp();
        exp_pos / (1.0 + exp_pos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_features() -> FillFeatures {
        FillFeatures {
            queue_position: 0.2,
            spread_fraction: 0.001,
            volatility: 0.15,
            time_remaining: 0.8,
        }
    }

    #[test]
    fn test_basic() {
        let instance = FillProbability::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_predict_returns_bounded_probability() {
        let model = FillProbability::new();
        let est = model.predict(&default_features()).unwrap();
        assert!(est.probability >= 0.0 && est.probability <= 1.0);
    }

    #[test]
    fn test_front_of_queue_more_likely() {
        let model = FillProbability::new();
        let front = FillFeatures {
            queue_position: 0.0,
            ..default_features()
        };
        let back = FillFeatures {
            queue_position: 1.0,
            ..default_features()
        };
        let est_front = model.predict(&front).unwrap();
        let est_back = model.predict(&back).unwrap();
        assert!(
            est_front.probability > est_back.probability,
            "front={} should > back={}",
            est_front.probability,
            est_back.probability
        );
    }

    #[test]
    fn test_wider_spread_lower_fill() {
        let model = FillProbability::new();
        let narrow = FillFeatures {
            spread_fraction: 0.0001,
            ..default_features()
        };
        let wide = FillFeatures {
            spread_fraction: 0.01,
            ..default_features()
        };
        let est_narrow = model.predict(&narrow).unwrap();
        let est_wide = model.predict(&wide).unwrap();
        assert!(
            est_narrow.probability > est_wide.probability,
            "narrow={} should > wide={}",
            est_narrow.probability,
            est_wide.probability
        );
    }

    #[test]
    fn test_higher_volatility_more_fills() {
        let model = FillProbability::new();
        let calm = FillFeatures {
            volatility: 0.05,
            ..default_features()
        };
        let volatile = FillFeatures {
            volatility: 0.50,
            ..default_features()
        };
        let est_calm = model.predict(&calm).unwrap();
        let est_volatile = model.predict(&volatile).unwrap();
        assert!(
            est_volatile.probability > est_calm.probability,
            "volatile={} should > calm={}",
            est_volatile.probability,
            est_calm.probability
        );
    }

    #[test]
    fn test_more_time_more_fills() {
        let model = FillProbability::new();
        let little_time = FillFeatures {
            time_remaining: 0.1,
            ..default_features()
        };
        let lots_time = FillFeatures {
            time_remaining: 0.9,
            ..default_features()
        };
        let est_little = model.predict(&little_time).unwrap();
        let est_lots = model.predict(&lots_time).unwrap();
        assert!(
            est_lots.probability > est_little.probability,
            "lots={} should > little={}",
            est_lots.probability,
            est_little.probability
        );
    }

    #[test]
    fn test_not_adapted_below_min_samples() {
        let mut model = FillProbability::new();
        for _ in 0..model.config.min_samples - 1 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.7,
                filled: 1.0,
            });
        }
        assert!(!model.is_adapted());
    }

    #[test]
    fn test_adapted_at_min_samples() {
        let mut model = FillProbability::new();
        for _ in 0..model.config.min_samples {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.7,
                filled: 1.0,
            });
        }
        assert!(model.is_adapted());
    }

    #[test]
    fn test_calibration_offset_adapts() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            min_samples: 5,
            ..Default::default()
        });

        // Consistently over-predict: predict 0.9, actual 0.3
        for _ in 0..20 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.9,
                filled: 0.3,
            });
        }

        // Offset should be positive (we over-predicted)
        assert!(
            model.ema_offset() > 0.0,
            "offset {} should be > 0 when over-predicting",
            model.ema_offset()
        );
    }

    #[test]
    fn test_adaptive_calibration_corrects_bias() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            min_samples: 5,
            adaptation_weight: 0.8,
            ..Default::default()
        });

        // Over-predict consistently
        let features = default_features();
        let base_est = model.predict(&features).unwrap();

        for _ in 0..30 {
            model.observe(FillOutcome {
                features: features.clone(),
                predicted_probability: base_est.probability,
                filled: 0.0, // never fills
            });
        }

        let adapted_est = model.predict(&features).unwrap();
        assert!(adapted_est.adapted);
        // Adapted estimate should be lower since we kept over-predicting
        assert!(
            adapted_est.probability < base_est.probability,
            "adapted={} should < base={}",
            adapted_est.probability,
            base_est.probability
        );
    }

    #[test]
    fn test_brier_score_tracking() {
        let mut model = FillProbability::new();

        // Perfect predictions
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 1.0,
            filled: 1.0,
        });
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.0,
            filled: 0.0,
        });

        assert!(
            model.stats().brier_score() < 1e-10,
            "Perfect predictions should yield Brier ≈ 0, got {}",
            model.stats().brier_score()
        );
    }

    #[test]
    fn test_brier_score_worst_case() {
        let mut model = FillProbability::new();

        // Worst possible predictions
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.0,
            filled: 1.0,
        });
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 1.0,
            filled: 0.0,
        });

        let brier = model.stats().brier_score();
        assert!(
            (brier - 1.0).abs() < 1e-10,
            "Worst predictions should yield Brier ≈ 1, got {}",
            brier
        );
    }

    #[test]
    fn test_classification_stats() {
        let mut model = FillProbability::new();

        // TP: predicted 0.7, filled
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.7,
            filled: 1.0,
        });
        // FP: predicted 0.6, not filled
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.6,
            filled: 0.0,
        });
        // TN: predicted 0.3, not filled
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.3,
            filled: 0.0,
        });
        // FN: predicted 0.4, filled
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.4,
            filled: 1.0,
        });

        assert_eq!(model.stats().true_positives, 1);
        assert_eq!(model.stats().false_positives, 1);
        assert_eq!(model.stats().true_negatives, 1);
        assert_eq!(model.stats().false_negatives, 1);
        assert!((model.stats().accuracy() - 0.5).abs() < 1e-10);
        assert!((model.stats().precision() - 0.5).abs() < 1e-10);
        assert!((model.stats().recall() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fill_rate() {
        let mut model = FillProbability::new();
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.5,
            filled: 1.0,
        });
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.5,
            filled: 1.0,
        });
        model.observe(FillOutcome {
            features: default_features(),
            predicted_probability: 0.5,
            filled: 0.0,
        });
        assert!((model.stats().fill_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_fill_rate() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            window_size: 4,
            ..Default::default()
        });

        for _ in 0..4 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.5,
                filled: 1.0,
            });
        }
        assert!((model.windowed_fill_rate() - 1.0).abs() < 1e-10);

        // Push some non-fills to evict fills
        for _ in 0..4 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.5,
                filled: 0.0,
            });
        }
        assert!((model.windowed_fill_rate()).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_brier_score() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            window_size: 10,
            ..Default::default()
        });

        // All perfect predictions
        for _ in 0..5 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 1.0,
                filled: 1.0,
            });
        }
        assert!(model.windowed_brier_score() < 1e-10);
    }

    #[test]
    fn test_is_degrading() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            window_size: 20,
            ..Default::default()
        });

        // First half: accurate predictions
        for _ in 0..10 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 1.0,
                filled: 1.0,
            });
        }
        // Second half: terrible predictions
        for _ in 0..10 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.0,
                filled: 1.0,
            });
        }
        assert!(model.is_degrading());
    }

    #[test]
    fn test_not_degrading_when_consistent() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            window_size: 20,
            ..Default::default()
        });

        for _ in 0..20 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.7,
                filled: 1.0,
            });
        }
        assert!(!model.is_degrading());
    }

    #[test]
    fn test_not_degrading_insufficient_data() {
        let mut model = FillProbability::new();
        for _ in 0..4 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.0,
                filled: 1.0,
            });
        }
        assert!(!model.is_degrading());
    }

    #[test]
    fn test_reset() {
        let mut model = FillProbability::new();
        for _ in 0..30 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.5,
                filled: 1.0,
            });
        }
        assert!(model.observation_count() > 0);

        model.reset();
        assert_eq!(model.observation_count(), 0);
        assert_eq!(model.stats().observations, 0);
        assert!(!model.is_adapted());
        assert!(model.recent_outcomes().is_empty());
    }

    #[test]
    fn test_invalid_queue_position() {
        let model = FillProbability::new();
        let features = FillFeatures {
            queue_position: 1.5,
            ..default_features()
        };
        assert!(model.predict(&features).is_err());
    }

    #[test]
    fn test_invalid_negative_queue_position() {
        let model = FillProbability::new();
        let features = FillFeatures {
            queue_position: -0.1,
            ..default_features()
        };
        assert!(model.predict(&features).is_err());
    }

    #[test]
    fn test_invalid_spread_fraction() {
        let model = FillProbability::new();
        let features = FillFeatures {
            spread_fraction: -0.01,
            ..default_features()
        };
        assert!(model.predict(&features).is_err());
    }

    #[test]
    fn test_invalid_volatility() {
        let model = FillProbability::new();
        let features = FillFeatures {
            volatility: -1.0,
            ..default_features()
        };
        assert!(model.predict(&features).is_err());
    }

    #[test]
    fn test_invalid_time_remaining() {
        let model = FillProbability::new();
        let features = FillFeatures {
            time_remaining: 2.0,
            ..default_features()
        };
        assert!(model.predict(&features).is_err());
    }

    #[test]
    fn test_invalid_ema_decay() {
        let model = FillProbability::with_config(FillProbabilityConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_adaptation_weight() {
        let model = FillProbability::with_config(FillProbabilityConfig {
            adaptation_weight: 1.5,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_sigmoid_symmetry() {
        let s1 = sigmoid(2.0);
        let s2 = sigmoid(-2.0);
        assert!((s1 + s2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
    }

    #[test]
    fn test_confidence_zero_without_observations() {
        let model = FillProbability::new();
        let est = model.predict(&default_features()).unwrap();
        assert!((est.confidence - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases_with_samples() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            min_samples: 5,
            ..Default::default()
        });

        // Add some well-calibrated observations
        let features = default_features();
        let est0 = model.predict(&features).unwrap();
        let conf0 = est0.confidence;

        for _ in 0..30 {
            let est = model.predict(&features).unwrap();
            model.observe(FillOutcome {
                features: features.clone(),
                predicted_probability: est.probability,
                filled: if est.probability > 0.5 { 1.0 } else { 0.0 },
            });
        }

        let est1 = model.predict(&features).unwrap();
        assert!(
            est1.confidence > conf0,
            "confidence should increase: {} vs {}",
            est1.confidence,
            conf0
        );
    }

    #[test]
    fn test_predict_tracked_increments_predictions() {
        let mut model = FillProbability::new();
        assert_eq!(model.stats().predictions, 0);
        let _ = model.predict_tracked(&default_features()).unwrap();
        assert_eq!(model.stats().predictions, 1);
        let _ = model.predict_tracked(&default_features()).unwrap();
        assert_eq!(model.stats().predictions, 2);
    }

    #[test]
    fn test_window_eviction() {
        let mut model = FillProbability::with_config(FillProbabilityConfig {
            window_size: 3,
            ..Default::default()
        });

        for _ in 0..10 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.5,
                filled: 1.0,
            });
        }

        assert_eq!(model.recent_outcomes().len(), 3);
        assert_eq!(model.observation_count(), 10);
    }

    #[test]
    fn test_calibration_error() {
        let mut model = FillProbability::new();

        // Predict 0.9, actual 0.0 — large calibration error
        for _ in 0..10 {
            model.observe(FillOutcome {
                features: default_features(),
                predicted_probability: 0.9,
                filled: 0.0,
            });
        }

        assert!(
            model.stats().calibration_error() > 0.5,
            "calibration error should be large: {}",
            model.stats().calibration_error()
        );
    }

    #[test]
    fn test_empty_stats_defaults() {
        let stats = FillProbabilityStats::default();
        assert_eq!(stats.brier_score(), 0.0);
        assert_eq!(stats.fill_rate(), 0.0);
        assert_eq!(stats.accuracy(), 0.0);
        assert_eq!(stats.precision(), 0.0);
        assert_eq!(stats.recall(), 0.0);
        assert_eq!(stats.calibration_error(), 0.0);
    }
}
