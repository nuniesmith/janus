//! Temporary (transient) market impact model
//!
//! Part of the Cerebellum region
//! Component: impact
//!
//! Models the transient component of market impact — the portion of
//! price displacement caused by a trade that decays back toward the
//! pre-trade equilibrium over time. Uses a power-law decay schedule
//! with configurable exponent and adaptive calibration from observed
//! impact decay curves.
//!
//! Key features:
//! - Power-law decay model: impact(t) = η · σ · (Q/V)^δ · (1 + t/τ)^(-β)
//! - Configurable decay exponent (β) and time-scale (τ)
//! - EMA-smoothed calibration of decay parameters from observed curves
//! - Half-life estimation for the temporary impact
//! - Cumulative temporary cost estimation over execution horizon
//! - Running statistics with prediction accuracy tracking
//! - Safety clamping via max_impact_fraction

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Configuration for the temporary impact model
#[derive(Debug, Clone)]
pub struct TemporaryImpactConfig {
    /// Base temporary impact coefficient (η)
    pub eta: f64,
    /// Power-law decay exponent (β); higher = faster decay
    pub decay_exponent: f64,
    /// Characteristic decay time-scale in seconds (τ)
    pub decay_timescale_secs: f64,
    /// Volume fraction exponent (δ); typically 0.5–1.0
    pub volume_exponent: f64,
    /// EMA decay factor for adaptive calibration (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum observations before adaptive calibration activates
    pub min_samples: usize,
    /// Maximum number of observations in the sliding window
    pub window_size: usize,
    /// Blend weight for adaptive vs static model (0 = all static, 1 = all adaptive)
    pub adaptation_weight: f64,
    /// Maximum allowed temporary impact as fraction of price (safety clamp)
    pub max_impact_fraction: f64,
    /// Minimum trade size for valid estimation (avoid division by tiny volumes)
    pub min_trade_size: f64,
}

impl Default for TemporaryImpactConfig {
    fn default() -> Self {
        Self {
            eta: 0.15,
            decay_exponent: 0.5,
            decay_timescale_secs: 10.0,
            volume_exponent: 0.6,
            ema_decay: 0.94,
            min_samples: 15,
            window_size: 200,
            adaptation_weight: 0.3,
            max_impact_fraction: 0.05,
            min_trade_size: 1e-10,
        }
    }
}

/// Side of the trade
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeSide {
    Buy,
    Sell,
}

/// Parameters describing a trade for impact estimation
#[derive(Debug, Clone)]
pub struct TradeParams {
    /// Size of the trade in base currency units
    pub trade_size: f64,
    /// Average daily (or bucket) market volume
    pub market_volume: f64,
    /// Current realised volatility (annualised fraction, e.g. 0.20)
    pub volatility: f64,
    /// Reference (pre-trade) price
    pub reference_price: f64,
    /// Side of the trade
    pub side: TradeSide,
}

/// Estimated temporary impact at a given point in time
#[derive(Debug, Clone)]
pub struct TemporaryImpactEstimate {
    /// Instantaneous (peak) temporary impact in price units at t = 0
    pub peak_impact: f64,
    /// Temporary impact at the requested elapsed time
    pub impact_at_t: f64,
    /// Impact as a fraction of the reference price
    pub impact_fraction: f64,
    /// Estimated half-life of the temporary impact (seconds)
    pub half_life_secs: f64,
    /// Whether the model is using adaptive calibration
    pub adapted: bool,
    /// Confidence in the estimate
    pub confidence: f64,
    /// Effective eta used (blended static + adaptive)
    pub effective_eta: f64,
}

/// Observation of an actual impact decay for calibration
#[derive(Debug, Clone)]
pub struct DecayObservation {
    /// Trade parameters at the time of execution
    pub trade: TradeParams,
    /// Observed peak impact right after the trade (price units)
    pub observed_peak_impact: f64,
    /// Observed impact at a later time (price units)
    pub observed_impact_at_t: f64,
    /// Elapsed time since the trade (seconds)
    pub elapsed_secs: f64,
}

/// Running statistics for the temporary impact model
#[derive(Debug, Clone, Default)]
pub struct TemporaryImpactStats {
    /// Total predictions made
    pub predictions: usize,
    /// Total decay observations recorded
    pub observations: usize,
    /// Sum of absolute prediction errors (peak impact)
    pub sum_abs_peak_error: f64,
    /// Sum of squared prediction errors (peak impact)
    pub sum_sq_peak_error: f64,
    /// Sum of absolute decay prediction errors
    pub sum_abs_decay_error: f64,
    /// Sum of squared decay prediction errors
    pub sum_sq_decay_error: f64,
    /// Maximum absolute peak prediction error
    pub max_peak_error: f64,
    /// Maximum absolute decay prediction error
    pub max_decay_error: f64,
    /// Sum of observed peak impacts (for mean)
    pub sum_observed_peak: f64,
    /// Sum of squared observed peak impacts (for variance)
    pub sum_sq_observed_peak: f64,
    /// Number of estimates that were clamped by max_impact_fraction
    pub clamped_count: usize,
}

impl TemporaryImpactStats {
    /// Mean absolute error for peak impact predictions
    pub fn peak_mae(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_abs_peak_error / self.observations as f64
    }

    /// Root mean squared error for peak impact predictions
    pub fn peak_rmse(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        (self.sum_sq_peak_error / self.observations as f64).sqrt()
    }

    /// Mean absolute error for decay predictions
    pub fn decay_mae(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_abs_decay_error / self.observations as f64
    }

    /// Root mean squared error for decay predictions
    pub fn decay_rmse(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        (self.sum_sq_decay_error / self.observations as f64).sqrt()
    }

    /// Mean observed peak impact
    pub fn mean_observed_peak(&self) -> f64 {
        if self.observations == 0 {
            return 0.0;
        }
        self.sum_observed_peak / self.observations as f64
    }

    /// Variance of observed peak impacts
    pub fn observed_peak_variance(&self) -> f64 {
        if self.observations < 2 {
            return 0.0;
        }
        let mean = self.mean_observed_peak();
        let var = self.sum_sq_observed_peak / self.observations as f64 - mean * mean;
        var.max(0.0)
    }
}

/// Temporary (transient) market impact model
///
/// Models the short-lived price displacement caused by a trade that
/// decays back toward equilibrium. Uses a power-law decay schedule
/// with adaptive coefficient calibration.
pub struct TemporaryImpact {
    config: TemporaryImpactConfig,
    /// EMA of the implied eta from observed peak impacts
    ema_eta: f64,
    /// EMA of the implied decay exponent from observed decay curves
    ema_decay_exp: f64,
    /// Whether EMA has been initialized
    ema_initialized: bool,
    /// Total observation count (including evicted from window)
    observation_count: usize,
    /// Recent observations for windowed analysis
    recent: VecDeque<DecayObservation>,
    /// Running statistics
    stats: TemporaryImpactStats,
}

impl Default for TemporaryImpact {
    fn default() -> Self {
        Self::new()
    }
}

impl TemporaryImpact {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(TemporaryImpactConfig::default())
    }

    /// Create a new instance with the given configuration
    pub fn with_config(config: TemporaryImpactConfig) -> Self {
        Self {
            ema_eta: config.eta,
            ema_decay_exp: config.decay_exponent,
            ema_initialized: false,
            observation_count: 0,
            recent: VecDeque::new(),
            stats: TemporaryImpactStats::default(),
            config,
        }
    }

    /// Main processing function — validates configuration
    pub fn process(&self) -> Result<()> {
        if self.config.eta < 0.0 {
            return Err(Error::InvalidInput("eta must be >= 0".into()));
        }
        if self.config.decay_exponent <= 0.0 {
            return Err(Error::InvalidInput("decay_exponent must be > 0".into()));
        }
        if self.config.decay_timescale_secs <= 0.0 {
            return Err(Error::InvalidInput(
                "decay_timescale_secs must be > 0".into(),
            ));
        }
        if self.config.volume_exponent <= 0.0 || self.config.volume_exponent > 2.0 {
            return Err(Error::InvalidInput(
                "volume_exponent must be in (0, 2]".into(),
            ));
        }
        if self.config.ema_decay <= 0.0 || self.config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if self.config.adaptation_weight < 0.0 || self.config.adaptation_weight > 1.0 {
            return Err(Error::InvalidInput(
                "adaptation_weight must be in [0, 1]".into(),
            ));
        }
        if self.config.max_impact_fraction <= 0.0 || self.config.max_impact_fraction > 1.0 {
            return Err(Error::InvalidInput(
                "max_impact_fraction must be in (0, 1]".into(),
            ));
        }
        Ok(())
    }

    /// Estimate the temporary impact of a trade at a given elapsed time
    ///
    /// `elapsed_secs` is the time after the trade; 0.0 gives the peak impact.
    pub fn estimate(
        &self,
        trade: &TradeParams,
        elapsed_secs: f64,
    ) -> Result<TemporaryImpactEstimate> {
        // Validate inputs
        if trade.trade_size < 0.0 {
            return Err(Error::InvalidInput("trade_size must be >= 0".into()));
        }
        if trade.market_volume <= 0.0 {
            return Err(Error::InvalidInput("market_volume must be > 0".into()));
        }
        if trade.volatility < 0.0 {
            return Err(Error::InvalidInput("volatility must be >= 0".into()));
        }
        if trade.reference_price <= 0.0 {
            return Err(Error::InvalidInput("reference_price must be > 0".into()));
        }
        if elapsed_secs < 0.0 {
            return Err(Error::InvalidInput("elapsed_secs must be >= 0".into()));
        }

        // Zero trade → zero impact
        if trade.trade_size < self.config.min_trade_size {
            return Ok(TemporaryImpactEstimate {
                peak_impact: 0.0,
                impact_at_t: 0.0,
                impact_fraction: 0.0,
                half_life_secs: self.compute_half_life(),
                adapted: self.is_adapted(),
                confidence: self.compute_confidence(),
                effective_eta: self.effective_eta(),
            });
        }

        let eta = self.effective_eta();
        let volume_fraction = trade.trade_size / trade.market_volume;
        let volume_term = volume_fraction.powf(self.config.volume_exponent);

        // Peak impact: η · σ · (Q/V)^δ · price
        let raw_peak = eta * trade.volatility * volume_term * trade.reference_price;

        // Safety clamp
        let max_impact = self.config.max_impact_fraction * trade.reference_price;
        let peak_impact = raw_peak.min(max_impact);

        // Decay: (1 + t/τ)^(-β)
        let effective_beta = self.effective_decay_exponent();
        let decay_factor = self.decay_factor(elapsed_secs, effective_beta);
        let impact_at_t = peak_impact * decay_factor;
        let impact_fraction = if trade.reference_price > 0.0 {
            impact_at_t / trade.reference_price
        } else {
            0.0
        };

        let half_life = self.compute_half_life();
        let _clamped = (raw_peak - peak_impact).abs() > 1e-12;

        Ok(TemporaryImpactEstimate {
            peak_impact,
            impact_at_t,
            impact_fraction,
            half_life_secs: half_life,
            adapted: self.is_adapted(),
            confidence: self.compute_confidence(),
            effective_eta: eta,
        })
    }

    /// Estimate the cumulative (integrated) temporary cost over an execution horizon
    ///
    /// Numerically integrates the temporary impact from t=0 to t=horizon_secs
    /// using the trapezoidal rule with `num_steps` intervals.
    pub fn cumulative_cost(
        &self,
        trade: &TradeParams,
        horizon_secs: f64,
        num_steps: usize,
    ) -> Result<f64> {
        if horizon_secs <= 0.0 {
            return Err(Error::InvalidInput("horizon_secs must be > 0".into()));
        }
        if num_steps == 0 {
            return Err(Error::InvalidInput("num_steps must be > 0".into()));
        }

        let dt = horizon_secs / num_steps as f64;
        let mut total = 0.0;

        // Trapezoidal integration
        let est_start = self.estimate(trade, 0.0)?;
        let est_end = self.estimate(trade, horizon_secs)?;
        total += (est_start.impact_at_t + est_end.impact_at_t) / 2.0;

        for i in 1..num_steps {
            let t = i as f64 * dt;
            let est = self.estimate(trade, t)?;
            total += est.impact_at_t;
        }

        Ok(total * dt)
    }

    /// Record an observed impact decay for adaptive calibration
    pub fn observe(&mut self, obs: DecayObservation) {
        // Compute implied eta from observed peak
        let trade = &obs.trade;
        if trade.trade_size < self.config.min_trade_size
            || trade.market_volume <= 0.0
            || trade.volatility <= 0.0
            || trade.reference_price <= 0.0
        {
            // Invalid observation — skip silently
            return;
        }

        let volume_fraction = trade.trade_size / trade.market_volume;
        let volume_term = volume_fraction.powf(self.config.volume_exponent);
        let denom = trade.volatility * volume_term * trade.reference_price;

        let implied_eta = if denom > 1e-15 {
            obs.observed_peak_impact.abs() / denom
        } else {
            self.config.eta
        };

        // Compute implied decay exponent from the observed decay
        let implied_beta = if obs.elapsed_secs > 0.0 && obs.observed_peak_impact.abs() > 1e-15 {
            let ratio = obs.observed_impact_at_t.abs() / obs.observed_peak_impact.abs();
            if ratio > 0.0 && ratio < 1.0 {
                let time_ratio = 1.0 + obs.elapsed_secs / self.config.decay_timescale_secs;
                // ratio = time_ratio^(-β) → β = -ln(ratio) / ln(time_ratio)
                if time_ratio > 1.0 {
                    (-ratio.ln() / time_ratio.ln()).clamp(0.01, 5.0)
                } else {
                    self.config.decay_exponent
                }
            } else {
                self.config.decay_exponent
            }
        } else {
            self.config.decay_exponent
        };

        // Update EMA
        if self.ema_initialized {
            self.ema_eta =
                self.config.ema_decay * self.ema_eta + (1.0 - self.config.ema_decay) * implied_eta;
            self.ema_decay_exp = self.config.ema_decay * self.ema_decay_exp
                + (1.0 - self.config.ema_decay) * implied_beta;
        } else {
            self.ema_eta = implied_eta;
            self.ema_decay_exp = implied_beta;
            self.ema_initialized = true;
        }

        // Compute prediction errors for stats
        let predicted_peak = self.config.eta * denom; // static model prediction
        let peak_error = (predicted_peak - obs.observed_peak_impact.abs()).abs();
        self.stats.sum_abs_peak_error += peak_error;
        self.stats.sum_sq_peak_error += peak_error * peak_error;
        if peak_error > self.stats.max_peak_error {
            self.stats.max_peak_error = peak_error;
        }

        // Decay prediction error
        let predicted_at_t =
            predicted_peak * self.decay_factor(obs.elapsed_secs, self.config.decay_exponent);
        let decay_error = (predicted_at_t - obs.observed_impact_at_t.abs()).abs();
        self.stats.sum_abs_decay_error += decay_error;
        self.stats.sum_sq_decay_error += decay_error * decay_error;
        if decay_error > self.stats.max_decay_error {
            self.stats.max_decay_error = decay_error;
        }

        // General stats
        self.stats.observations += 1;
        self.stats.sum_observed_peak += obs.observed_peak_impact.abs();
        self.stats.sum_sq_observed_peak +=
            obs.observed_peak_impact.abs() * obs.observed_peak_impact.abs();

        self.observation_count += 1;

        // Maintain sliding window
        self.recent.push_back(obs);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }
    }

    /// Compute the decay factor at a given elapsed time
    ///
    /// decay(t) = (1 + t/τ)^(-β)
    fn decay_factor(&self, elapsed_secs: f64, beta: f64) -> f64 {
        if elapsed_secs <= 0.0 {
            return 1.0;
        }
        let time_ratio = 1.0 + elapsed_secs / self.config.decay_timescale_secs;
        time_ratio.powf(-beta)
    }

    /// Compute the half-life of the temporary impact (seconds)
    ///
    /// Solves: (1 + t_half/τ)^(-β) = 0.5
    /// → t_half = τ · (2^(1/β) - 1)
    fn compute_half_life(&self) -> f64 {
        let beta = self.effective_decay_exponent();
        if beta <= 0.0 {
            return f64::INFINITY;
        }
        self.config.decay_timescale_secs * (2.0_f64.powf(1.0 / beta) - 1.0)
    }

    /// Effective eta (blended static + adaptive)
    pub fn effective_eta(&self) -> f64 {
        if self.is_adapted() {
            let w = self.config.adaptation_weight;
            (1.0 - w) * self.config.eta + w * self.ema_eta
        } else {
            self.config.eta
        }
    }

    /// Effective decay exponent (blended static + adaptive)
    pub fn effective_decay_exponent(&self) -> f64 {
        if self.is_adapted() {
            let w = self.config.adaptation_weight;
            (1.0 - w) * self.config.decay_exponent + w * self.ema_decay_exp
        } else {
            self.config.decay_exponent
        }
    }

    /// Whether adaptive calibration is active
    pub fn is_adapted(&self) -> bool {
        self.ema_initialized && self.observation_count >= self.config.min_samples
    }

    /// Current EMA of implied eta
    pub fn ema_eta(&self) -> f64 {
        self.ema_eta
    }

    /// Current EMA of implied decay exponent
    pub fn ema_decay_exponent(&self) -> f64 {
        self.ema_decay_exp
    }

    /// Total observations recorded
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Reference to running statistics
    pub fn stats(&self) -> &TemporaryImpactStats {
        &self.stats
    }

    /// Recent observations in the sliding window
    pub fn recent_observations(&self) -> &VecDeque<DecayObservation> {
        &self.recent
    }

    /// Half-life of the current model (seconds)
    pub fn half_life_secs(&self) -> f64 {
        self.compute_half_life()
    }

    /// Windowed mean observed peak impact
    pub fn windowed_mean_peak(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self
            .recent
            .iter()
            .map(|o| o.observed_peak_impact.abs())
            .sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean decay ratio (observed_at_t / observed_peak)
    pub fn windowed_mean_decay_ratio(&self) -> f64 {
        let valid: Vec<f64> = self
            .recent
            .iter()
            .filter(|o| o.observed_peak_impact.abs() > 1e-15)
            .map(|o| o.observed_impact_at_t.abs() / o.observed_peak_impact.abs())
            .collect();
        if valid.is_empty() {
            return 0.0;
        }
        valid.iter().sum::<f64>() / valid.len() as f64
    }

    /// Check if model quality is degrading (second half of window has larger errors)
    pub fn is_degrading(&self) -> bool {
        let n = self.recent.len();
        if n < 10 {
            return false;
        }
        let mid = n / 2;

        let first_half_err: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|o| {
                let trade = &o.trade;
                if trade.market_volume <= 0.0
                    || trade.volatility <= 0.0
                    || trade.reference_price <= 0.0
                {
                    return 0.0;
                }
                let vf = trade.trade_size / trade.market_volume;
                let vt = vf.powf(self.config.volume_exponent);
                let predicted = self.config.eta * trade.volatility * vt * trade.reference_price;
                (predicted - o.observed_peak_impact.abs()).abs()
            })
            .sum::<f64>()
            / mid as f64;

        let second_half_err: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|o| {
                let trade = &o.trade;
                if trade.market_volume <= 0.0
                    || trade.volatility <= 0.0
                    || trade.reference_price <= 0.0
                {
                    return 0.0;
                }
                let vf = trade.trade_size / trade.market_volume;
                let vt = vf.powf(self.config.volume_exponent);
                let predicted = self.config.eta * trade.volatility * vt * trade.reference_price;
                (predicted - o.observed_peak_impact.abs()).abs()
            })
            .sum::<f64>()
            / (n - mid) as f64;

        second_half_err > first_half_err * 1.2
    }

    /// Compute confidence based on sample count and prediction accuracy
    fn compute_confidence(&self) -> f64 {
        if self.observation_count == 0 {
            return 0.0;
        }
        // Sample confidence ramps over min_samples * 3
        let sample_conf =
            (self.observation_count as f64 / (self.config.min_samples as f64 * 3.0)).min(1.0);

        // Quality confidence based on peak RMSE relative to mean peak
        let quality_conf = if self.stats.observations > 0 {
            let mean_peak = self.stats.mean_observed_peak();
            if mean_peak > 1e-10 {
                let norm_rmse = self.stats.peak_rmse() / mean_peak;
                (1.0 - norm_rmse).clamp(0.0, 1.0)
            } else {
                0.5
            }
        } else {
            0.5
        };

        (sample_conf * quality_conf).sqrt()
    }

    /// Reset all adaptive state and statistics
    pub fn reset(&mut self) {
        self.ema_eta = self.config.eta;
        self.ema_decay_exp = self.config.decay_exponent;
        self.ema_initialized = false;
        self.observation_count = 0;
        self.recent.clear();
        self.stats = TemporaryImpactStats::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_trade() -> TradeParams {
        TradeParams {
            trade_size: 100.0,
            market_volume: 10000.0,
            volatility: 0.20,
            reference_price: 100.0,
            side: TradeSide::Buy,
        }
    }

    fn small_config() -> TemporaryImpactConfig {
        TemporaryImpactConfig {
            eta: 0.1,
            decay_exponent: 0.5,
            decay_timescale_secs: 10.0,
            volume_exponent: 0.5,
            ema_decay: 0.5,
            min_samples: 3,
            window_size: 50,
            adaptation_weight: 0.5,
            max_impact_fraction: 0.05,
            min_trade_size: 1e-10,
        }
    }

    #[test]
    fn test_basic() {
        let instance = TemporaryImpact::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_zero_trade_zero_impact() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = TradeParams {
            trade_size: 0.0,
            ..default_trade()
        };
        let est = model.estimate(&trade, 0.0).unwrap();
        assert_eq!(est.peak_impact, 0.0);
        assert_eq!(est.impact_at_t, 0.0);
    }

    #[test]
    fn test_peak_impact_positive() {
        let model = TemporaryImpact::with_config(small_config());
        let est = model.estimate(&default_trade(), 0.0).unwrap();
        assert!(
            est.peak_impact > 0.0,
            "peak impact should be > 0, got {}",
            est.peak_impact
        );
    }

    #[test]
    fn test_impact_decays_over_time() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = default_trade();

        let est_0 = model.estimate(&trade, 0.0).unwrap();
        let est_5 = model.estimate(&trade, 5.0).unwrap();
        let est_30 = model.estimate(&trade, 30.0).unwrap();

        assert!(
            est_5.impact_at_t < est_0.impact_at_t,
            "impact should decay: t=0 {} > t=5 {}",
            est_0.impact_at_t,
            est_5.impact_at_t
        );
        assert!(
            est_30.impact_at_t < est_5.impact_at_t,
            "impact should decay further: t=5 {} > t=30 {}",
            est_5.impact_at_t,
            est_30.impact_at_t
        );
    }

    #[test]
    fn test_larger_trade_more_impact() {
        let model = TemporaryImpact::with_config(small_config());

        let small_trade = TradeParams {
            trade_size: 50.0,
            ..default_trade()
        };
        let large_trade = TradeParams {
            trade_size: 500.0,
            ..default_trade()
        };

        let est_small = model.estimate(&small_trade, 0.0).unwrap();
        let est_large = model.estimate(&large_trade, 0.0).unwrap();

        assert!(
            est_large.peak_impact > est_small.peak_impact,
            "larger trade should have more impact: {} vs {}",
            est_large.peak_impact,
            est_small.peak_impact
        );
    }

    #[test]
    fn test_higher_volatility_more_impact() {
        let model = TemporaryImpact::with_config(small_config());

        let calm = TradeParams {
            volatility: 0.10,
            ..default_trade()
        };
        let volatile = TradeParams {
            volatility: 0.50,
            ..default_trade()
        };

        let est_calm = model.estimate(&calm, 0.0).unwrap();
        let est_volatile = model.estimate(&volatile, 0.0).unwrap();

        assert!(
            est_volatile.peak_impact > est_calm.peak_impact,
            "higher vol should produce more impact: {} vs {}",
            est_volatile.peak_impact,
            est_calm.peak_impact
        );
    }

    #[test]
    fn test_impact_clamped() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            eta: 100.0, // very large coefficient
            max_impact_fraction: 0.01,
            ..small_config()
        });

        let trade = default_trade();
        let est = model.estimate(&trade, 0.0).unwrap();

        let max_allowed = 0.01 * trade.reference_price;
        assert!(
            est.peak_impact <= max_allowed + 1e-10,
            "impact should be clamped: {} > {}",
            est.peak_impact,
            max_allowed
        );
    }

    #[test]
    fn test_half_life_positive() {
        let model = TemporaryImpact::with_config(small_config());
        let hl = model.half_life_secs();
        assert!(hl > 0.0, "half-life should be positive, got {}", hl);
    }

    #[test]
    fn test_half_life_consistency() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = default_trade();

        let est_peak = model.estimate(&trade, 0.0).unwrap();
        let hl = model.half_life_secs();
        let est_at_hl = model.estimate(&trade, hl).unwrap();

        let ratio = est_at_hl.impact_at_t / est_peak.impact_at_t;
        assert!(
            (ratio - 0.5).abs() < 0.01,
            "at half-life, impact should be ~50% of peak: ratio = {}",
            ratio
        );
    }

    #[test]
    fn test_cumulative_cost_positive() {
        let model = TemporaryImpact::with_config(small_config());
        let cost = model.cumulative_cost(&default_trade(), 10.0, 100).unwrap();
        assert!(cost > 0.0, "cumulative cost should be > 0, got {}", cost);
    }

    #[test]
    fn test_cumulative_cost_increases_with_horizon() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = default_trade();

        let cost_short = model.cumulative_cost(&trade, 5.0, 100).unwrap();
        let cost_long = model.cumulative_cost(&trade, 50.0, 100).unwrap();

        assert!(
            cost_long > cost_short,
            "longer horizon should have higher cumulative cost: {} vs {}",
            cost_long,
            cost_short
        );
    }

    #[test]
    fn test_cumulative_cost_invalid_horizon() {
        let model = TemporaryImpact::with_config(small_config());
        assert!(model.cumulative_cost(&default_trade(), 0.0, 100).is_err());
        assert!(model.cumulative_cost(&default_trade(), -1.0, 100).is_err());
    }

    #[test]
    fn test_cumulative_cost_invalid_steps() {
        let model = TemporaryImpact::with_config(small_config());
        assert!(model.cumulative_cost(&default_trade(), 10.0, 0).is_err());
    }

    #[test]
    fn test_observe_updates_count() {
        let mut model = TemporaryImpact::with_config(small_config());

        model.observe(DecayObservation {
            trade: default_trade(),
            observed_peak_impact: 0.5,
            observed_impact_at_t: 0.3,
            elapsed_secs: 5.0,
        });

        assert_eq!(model.observation_count(), 1);
        assert_eq!(model.stats().observations, 1);
    }

    #[test]
    fn test_not_adapted_below_min_samples() {
        let mut model = TemporaryImpact::with_config(small_config());

        for _ in 0..model.config.min_samples - 1 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 0.5,
                observed_impact_at_t: 0.3,
                elapsed_secs: 5.0,
            });
        }

        assert!(!model.is_adapted());
    }

    #[test]
    fn test_adapted_at_min_samples() {
        let mut model = TemporaryImpact::with_config(small_config());

        for _ in 0..model.config.min_samples {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 0.5,
                observed_impact_at_t: 0.3,
                elapsed_secs: 5.0,
            });
        }

        assert!(model.is_adapted());
    }

    #[test]
    fn test_adaptive_eta_converges() {
        let mut model = TemporaryImpact::with_config(TemporaryImpactConfig {
            eta: 0.1,
            min_samples: 3,
            ema_decay: 0.5,
            adaptation_weight: 0.8,
            ..small_config()
        });

        let trade = default_trade();
        // Simulate observations implying eta ≈ 0.5 (higher than config eta=0.1)
        let vf = trade.trade_size / trade.market_volume;
        let vt = vf.powf(model.config.volume_exponent);
        let denom = trade.volatility * vt * trade.reference_price;
        let implied_peak = 0.5 * denom; // implies eta = 0.5

        for _ in 0..30 {
            model.observe(DecayObservation {
                trade: trade.clone(),
                observed_peak_impact: implied_peak,
                observed_impact_at_t: implied_peak * 0.5,
                elapsed_secs: 5.0,
            });
        }

        let eff_eta = model.effective_eta();
        assert!(
            eff_eta > 0.1,
            "effective eta should adapt upward from 0.1, got {}",
            eff_eta
        );
    }

    #[test]
    fn test_decay_exponent_adapts() {
        let mut model = TemporaryImpact::with_config(TemporaryImpactConfig {
            decay_exponent: 0.5,
            min_samples: 3,
            ema_decay: 0.3,
            adaptation_weight: 0.8,
            decay_timescale_secs: 10.0,
            ..small_config()
        });

        let trade = default_trade();
        // Provide observations that imply faster decay (higher beta)
        // At t=10s with τ=10: time_ratio = 2.0
        // If observed ratio = 0.1, β = -ln(0.1)/ln(2) ≈ 3.32
        for _ in 0..30 {
            model.observe(DecayObservation {
                trade: trade.clone(),
                observed_peak_impact: 1.0,
                observed_impact_at_t: 0.1,
                elapsed_secs: 10.0,
            });
        }

        let eff_beta = model.effective_decay_exponent();
        assert!(
            eff_beta > 0.5,
            "decay exponent should adapt upward from 0.5, got {}",
            eff_beta
        );
    }

    #[test]
    fn test_stats_tracking() {
        let mut model = TemporaryImpact::with_config(small_config());

        for _ in 0..5 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 0.5,
                observed_impact_at_t: 0.3,
                elapsed_secs: 5.0,
            });
        }

        assert_eq!(model.stats().observations, 5);
        assert!(model.stats().sum_observed_peak > 0.0);
        assert!(model.stats().mean_observed_peak() > 0.0);
    }

    #[test]
    fn test_stats_peak_mae() {
        let mut model = TemporaryImpact::with_config(small_config());

        model.observe(DecayObservation {
            trade: default_trade(),
            observed_peak_impact: 0.5,
            observed_impact_at_t: 0.3,
            elapsed_secs: 5.0,
        });

        // MAE should be non-negative
        assert!(model.stats().peak_mae() >= 0.0);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = TemporaryImpactStats::default();
        assert_eq!(stats.peak_mae(), 0.0);
        assert_eq!(stats.peak_rmse(), 0.0);
        assert_eq!(stats.decay_mae(), 0.0);
        assert_eq!(stats.decay_rmse(), 0.0);
        assert_eq!(stats.mean_observed_peak(), 0.0);
        assert_eq!(stats.observed_peak_variance(), 0.0);
    }

    #[test]
    fn test_windowed_mean_peak() {
        let mut model = TemporaryImpact::with_config(TemporaryImpactConfig {
            window_size: 5,
            ..small_config()
        });

        for _ in 0..5 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 2.0,
                observed_impact_at_t: 1.0,
                elapsed_secs: 5.0,
            });
        }

        assert!(
            (model.windowed_mean_peak() - 2.0).abs() < 1e-10,
            "windowed mean peak should be 2.0, got {}",
            model.windowed_mean_peak()
        );
    }

    #[test]
    fn test_windowed_mean_decay_ratio() {
        let mut model = TemporaryImpact::with_config(small_config());

        for _ in 0..5 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 2.0,
                observed_impact_at_t: 1.0,
                elapsed_secs: 5.0,
            });
        }

        let ratio = model.windowed_mean_decay_ratio();
        assert!(
            (ratio - 0.5).abs() < 1e-10,
            "decay ratio should be 0.5, got {}",
            ratio
        );
    }

    #[test]
    fn test_is_degrading() {
        let mut model = TemporaryImpact::with_config(TemporaryImpactConfig {
            window_size: 20,
            ..small_config()
        });

        let trade = default_trade();

        // First half: accurate observations (near model prediction)
        let vf = trade.trade_size / trade.market_volume;
        let vt = vf.powf(model.config.volume_exponent);
        let model_peak = model.config.eta * trade.volatility * vt * trade.reference_price;

        for _ in 0..10 {
            model.observe(DecayObservation {
                trade: trade.clone(),
                observed_peak_impact: model_peak,
                observed_impact_at_t: model_peak * 0.5,
                elapsed_secs: 5.0,
            });
        }

        // Second half: observations far from model
        for _ in 0..10 {
            model.observe(DecayObservation {
                trade: trade.clone(),
                observed_peak_impact: model_peak * 10.0,
                observed_impact_at_t: model_peak * 5.0,
                elapsed_secs: 5.0,
            });
        }

        assert!(model.is_degrading());
    }

    #[test]
    fn test_not_degrading_consistent() {
        let mut model = TemporaryImpact::with_config(TemporaryImpactConfig {
            window_size: 20,
            ..small_config()
        });

        for _ in 0..20 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 0.5,
                observed_impact_at_t: 0.3,
                elapsed_secs: 5.0,
            });
        }

        assert!(!model.is_degrading());
    }

    #[test]
    fn test_not_degrading_insufficient_data() {
        let mut model = TemporaryImpact::with_config(small_config());
        for _ in 0..4 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 10.0,
                observed_impact_at_t: 5.0,
                elapsed_secs: 5.0,
            });
        }
        assert!(!model.is_degrading());
    }

    #[test]
    fn test_confidence_zero_without_observations() {
        let model = TemporaryImpact::with_config(small_config());
        let est = model.estimate(&default_trade(), 0.0).unwrap();
        assert!((est.confidence - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_increases_with_data() {
        let mut model = TemporaryImpact::with_config(small_config());

        let conf0 = model.estimate(&default_trade(), 0.0).unwrap().confidence;

        let trade = default_trade();
        let vf = trade.trade_size / trade.market_volume;
        let vt = vf.powf(model.config.volume_exponent);
        let model_peak = model.config.eta * trade.volatility * vt * trade.reference_price;

        for _ in 0..20 {
            model.observe(DecayObservation {
                trade: trade.clone(),
                observed_peak_impact: model_peak,
                observed_impact_at_t: model_peak * 0.5,
                elapsed_secs: 5.0,
            });
        }

        let conf1 = model.estimate(&default_trade(), 0.0).unwrap().confidence;
        assert!(
            conf1 > conf0,
            "confidence should increase: {} vs {}",
            conf1,
            conf0
        );
    }

    #[test]
    fn test_impact_fraction_correct() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = default_trade();
        let est = model.estimate(&trade, 0.0).unwrap();

        let expected_fraction = est.peak_impact / trade.reference_price;
        assert!(
            (est.impact_fraction - expected_fraction).abs() < 1e-10,
            "fraction should be impact/price: {} vs {}",
            est.impact_fraction,
            expected_fraction
        );
    }

    #[test]
    fn test_reset() {
        let mut model = TemporaryImpact::with_config(small_config());

        for _ in 0..20 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 0.5,
                observed_impact_at_t: 0.3,
                elapsed_secs: 5.0,
            });
        }

        assert!(model.observation_count() > 0);
        assert!(model.is_adapted());

        model.reset();

        assert_eq!(model.observation_count(), 0);
        assert!(!model.is_adapted());
        assert_eq!(model.stats().observations, 0);
        assert!(model.recent_observations().is_empty());
    }

    #[test]
    fn test_invalid_negative_trade_size() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = TradeParams {
            trade_size: -1.0,
            ..default_trade()
        };
        assert!(model.estimate(&trade, 0.0).is_err());
    }

    #[test]
    fn test_invalid_zero_market_volume() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = TradeParams {
            market_volume: 0.0,
            ..default_trade()
        };
        assert!(model.estimate(&trade, 0.0).is_err());
    }

    #[test]
    fn test_invalid_negative_volatility() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = TradeParams {
            volatility: -0.1,
            ..default_trade()
        };
        assert!(model.estimate(&trade, 0.0).is_err());
    }

    #[test]
    fn test_invalid_zero_price() {
        let model = TemporaryImpact::with_config(small_config());
        let trade = TradeParams {
            reference_price: 0.0,
            ..default_trade()
        };
        assert!(model.estimate(&trade, 0.0).is_err());
    }

    #[test]
    fn test_invalid_negative_elapsed() {
        let model = TemporaryImpact::with_config(small_config());
        assert!(model.estimate(&default_trade(), -1.0).is_err());
    }

    #[test]
    fn test_invalid_config_negative_eta() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            eta: -1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_decay_exponent() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            decay_exponent: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_zero_timescale() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            decay_timescale_secs: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_volume_exponent() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            volume_exponent: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_adaptation_weight() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            adaptation_weight: 1.5,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_invalid_config_bad_max_impact() {
        let model = TemporaryImpact::with_config(TemporaryImpactConfig {
            max_impact_fraction: 0.0,
            ..Default::default()
        });
        assert!(model.process().is_err());
    }

    #[test]
    fn test_window_eviction() {
        let mut model = TemporaryImpact::with_config(TemporaryImpactConfig {
            window_size: 5,
            ..small_config()
        });

        for _ in 0..20 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 0.5,
                observed_impact_at_t: 0.3,
                elapsed_secs: 5.0,
            });
        }

        assert_eq!(model.recent_observations().len(), 5);
        assert_eq!(model.observation_count(), 20);
    }

    #[test]
    fn test_windowed_mean_peak_empty() {
        let model = TemporaryImpact::with_config(small_config());
        assert_eq!(model.windowed_mean_peak(), 0.0);
    }

    #[test]
    fn test_windowed_mean_decay_ratio_empty() {
        let model = TemporaryImpact::with_config(small_config());
        assert_eq!(model.windowed_mean_decay_ratio(), 0.0);
    }

    #[test]
    fn test_sell_side_same_magnitude() {
        let model = TemporaryImpact::with_config(small_config());
        let buy_trade = default_trade();
        let sell_trade = TradeParams {
            side: TradeSide::Sell,
            ..default_trade()
        };

        let buy_est = model.estimate(&buy_trade, 0.0).unwrap();
        let sell_est = model.estimate(&sell_trade, 0.0).unwrap();

        // Magnitude should be the same (sign handling is left to caller)
        assert!(
            (buy_est.peak_impact - sell_est.peak_impact).abs() < 1e-10,
            "buy and sell should have same magnitude impact"
        );
    }

    #[test]
    fn test_observe_invalid_inputs_skipped() {
        let mut model = TemporaryImpact::with_config(small_config());

        // Invalid observation (zero volume) should be silently skipped
        model.observe(DecayObservation {
            trade: TradeParams {
                market_volume: 0.0,
                ..default_trade()
            },
            observed_peak_impact: 0.5,
            observed_impact_at_t: 0.3,
            elapsed_secs: 5.0,
        });

        assert_eq!(model.observation_count(), 0);
    }

    #[test]
    fn test_decay_factor_at_zero() {
        let model = TemporaryImpact::with_config(small_config());
        let factor = model.decay_factor(0.0, 0.5);
        assert!((factor - 1.0).abs() < 1e-10, "decay at t=0 should be 1.0");
    }

    #[test]
    fn test_decay_factor_decreases() {
        let model = TemporaryImpact::with_config(small_config());
        let f1 = model.decay_factor(1.0, 0.5);
        let f10 = model.decay_factor(10.0, 0.5);
        let f100 = model.decay_factor(100.0, 0.5);

        assert!(f1 < 1.0);
        assert!(f10 < f1);
        assert!(f100 < f10);
    }

    #[test]
    fn test_higher_decay_exponent_faster_decay() {
        let model = TemporaryImpact::with_config(small_config());

        let slow = model.decay_factor(10.0, 0.3);
        let fast = model.decay_factor(10.0, 1.0);

        assert!(
            fast < slow,
            "higher decay exponent should produce faster decay: {} vs {}",
            fast,
            slow
        );
    }

    #[test]
    fn test_observed_peak_variance_constant() {
        let mut model = TemporaryImpact::with_config(small_config());

        for _ in 0..10 {
            model.observe(DecayObservation {
                trade: default_trade(),
                observed_peak_impact: 1.0,
                observed_impact_at_t: 0.5,
                elapsed_secs: 5.0,
            });
        }

        assert!(
            model.stats().observed_peak_variance() < 1e-10,
            "constant peaks should have ~0 variance, got {}",
            model.stats().observed_peak_variance()
        );
    }
}
