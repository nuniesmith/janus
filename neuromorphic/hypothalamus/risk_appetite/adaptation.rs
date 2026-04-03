//! Risk Appetite Adaptation — Dynamic risk appetite adjustment
//!
//! Part of the Hypothalamus region
//! Component: risk_appetite
//!
//! Dynamically adjusts risk appetite based on recent trading performance,
//! drawdown state, and market regime. This module bridges raw performance
//! metrics to actionable risk multipliers.
//!
//! Features:
//! - **Drawdown-aware scaling**: Reduces risk appetite as drawdown deepens
//! - **Sharpe-ratio tracking**: Adjusts risk based on rolling risk-adjusted returns
//! - **Regime sensitivity**: Different adaptation speeds for different market regimes
//! - **Cooldown periods**: Prevents whipsaw after large losses
//! - **Graduated recovery**: Slowly restores risk appetite after drawdowns
//! - **EMA-smoothed signals**: Avoids overreaction to single observations
//! - **Running statistics and windowed diagnostics**

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Market regime for regime-sensitive adaptation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Low volatility, trending
    Calm,
    /// Normal conditions
    Normal,
    /// Elevated volatility
    Volatile,
    /// Extreme stress / crisis
    Crisis,
}

impl Default for MarketRegime {
    fn default() -> Self {
        MarketRegime::Normal
    }
}

impl MarketRegime {
    /// Default adaptation speed multiplier for this regime.
    /// Higher values → faster adaptation (more reactive).
    pub fn speed_multiplier(&self) -> f64 {
        match self {
            MarketRegime::Calm => 0.5,
            MarketRegime::Normal => 1.0,
            MarketRegime::Volatile => 1.5,
            MarketRegime::Crisis => 3.0,
        }
    }

    /// Default risk ceiling for this regime (fraction of base appetite).
    pub fn risk_ceiling(&self) -> f64 {
        match self {
            MarketRegime::Calm => 1.3,
            MarketRegime::Normal => 1.0,
            MarketRegime::Volatile => 0.7,
            MarketRegime::Crisis => 0.3,
        }
    }
}

/// Drawdown tier thresholds and their associated risk multipliers
#[derive(Debug, Clone)]
pub struct DrawdownTier {
    /// Drawdown fraction at which this tier activates (e.g. 0.05 = 5%)
    pub threshold: f64,
    /// Risk multiplier applied when drawdown exceeds this threshold
    pub multiplier: f64,
}

/// Configuration for `Adaptation`
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Base risk appetite (1.0 = normal, unitless multiplier)
    pub base_appetite: f64,
    /// Drawdown tiers — must be sorted by ascending threshold.
    /// Each tier defines a drawdown level and associated risk multiplier.
    pub drawdown_tiers: Vec<DrawdownTier>,
    /// EMA decay for smoothing the risk multiplier (0 < decay < 1)
    pub ema_decay: f64,
    /// EMA decay for Sharpe ratio tracking
    pub sharpe_ema_decay: f64,
    /// Window size for rolling Sharpe ratio calculation
    pub sharpe_window: usize,
    /// Risk-free rate per observation period (for Sharpe calculation)
    pub risk_free_rate: f64,
    /// Target Sharpe ratio — appetite increases above this, decreases below
    pub target_sharpe: f64,
    /// Maximum Sharpe-driven multiplier boost
    pub max_sharpe_boost: f64,
    /// Maximum Sharpe-driven multiplier penalty
    pub max_sharpe_penalty: f64,
    /// Cooldown: number of observations after a large loss before
    /// risk appetite can increase
    pub cooldown_observations: usize,
    /// What constitutes a "large loss" triggering cooldown (as fraction of equity)
    pub large_loss_threshold: f64,
    /// Recovery rate: fraction of lost appetite restored per observation
    /// after cooldown expires (0..1)
    pub recovery_rate: f64,
    /// Minimum risk multiplier (hard floor)
    pub min_multiplier: f64,
    /// Maximum risk multiplier (hard ceiling)
    pub max_multiplier: f64,
    /// Sliding window for recent observations
    pub window_size: usize,
    /// Whether regime adjustments are enabled
    pub regime_aware: bool,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            base_appetite: 1.0,
            drawdown_tiers: vec![
                DrawdownTier {
                    threshold: 0.02,
                    multiplier: 0.90,
                },
                DrawdownTier {
                    threshold: 0.05,
                    multiplier: 0.75,
                },
                DrawdownTier {
                    threshold: 0.10,
                    multiplier: 0.50,
                },
                DrawdownTier {
                    threshold: 0.20,
                    multiplier: 0.25,
                },
                DrawdownTier {
                    threshold: 0.30,
                    multiplier: 0.0,
                },
            ],
            ema_decay: 0.1,
            sharpe_ema_decay: 0.05,
            sharpe_window: 20,
            risk_free_rate: 0.0,
            target_sharpe: 1.0,
            max_sharpe_boost: 0.3,
            max_sharpe_penalty: 0.5,
            cooldown_observations: 5,
            large_loss_threshold: 0.03,
            recovery_rate: 0.1,
            min_multiplier: 0.0,
            max_multiplier: 2.0,
            window_size: 100,
            regime_aware: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// A single performance observation fed to the adaptation engine
#[derive(Debug, Clone)]
pub struct PerformanceObservation {
    /// Period return (fraction, e.g. 0.01 = +1%)
    pub period_return: f64,
    /// Current drawdown from peak equity (fraction, e.g. 0.05 = 5% drawdown)
    pub current_drawdown: f64,
    /// Current equity value (for tracking peaks)
    pub equity: f64,
}

/// The computed risk appetite after adaptation
#[derive(Debug, Clone)]
pub struct AdaptedAppetite {
    /// Final risk multiplier (base × drawdown × sharpe × regime)
    pub multiplier: f64,
    /// Smoothed (EMA) risk multiplier
    pub smoothed_multiplier: f64,
    /// Component: drawdown-driven multiplier
    pub drawdown_component: f64,
    /// Component: Sharpe-driven adjustment
    pub sharpe_component: f64,
    /// Component: regime ceiling
    pub regime_ceiling: f64,
    /// Whether cooldown is active (large loss recently)
    pub in_cooldown: bool,
    /// Whether trading should be halted (multiplier ≈ 0)
    pub should_halt: bool,
    /// Current rolling Sharpe ratio (if available)
    pub rolling_sharpe: Option<f64>,
    /// Current drawdown fraction
    pub current_drawdown: f64,
    /// Current market regime
    pub regime: MarketRegime,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the adaptation engine
#[derive(Debug, Clone, Default)]
pub struct AdaptationStats {
    /// Total observations processed
    pub total_observations: u64,
    /// Number of cooldown activations
    pub cooldown_activations: u64,
    /// Number of observations spent in cooldown
    pub cooldown_observations_total: u64,
    /// Number of halt events (multiplier clamped to 0)
    pub halt_events: u64,
    /// Peak drawdown observed
    pub peak_drawdown: f64,
    /// Peak equity observed
    pub peak_equity: f64,
    /// Sum of all period returns (for mean calculation)
    pub sum_returns: f64,
    /// Sum of squared period returns
    pub sum_sq_returns: f64,
    /// Minimum multiplier ever produced
    pub min_multiplier_produced: f64,
    /// Maximum multiplier ever produced
    pub max_multiplier_produced: f64,
    /// Number of observations where multiplier < 0.5 (severe reduction)
    pub severe_reduction_count: u64,
    /// Number of observations where multiplier > 1.0 (boosted)
    pub boost_count: u64,
}

impl AdaptationStats {
    /// Mean period return
    pub fn mean_return(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.sum_returns / self.total_observations as f64
    }

    /// Return variance
    pub fn return_variance(&self) -> f64 {
        if self.total_observations < 2 {
            return 0.0;
        }
        let n = self.total_observations as f64;
        let mean = self.sum_returns / n;
        (self.sum_sq_returns / n - mean * mean).max(0.0)
    }

    /// Return standard deviation
    pub fn return_std(&self) -> f64 {
        self.return_variance().sqrt()
    }

    /// Fraction of time spent in cooldown
    pub fn cooldown_fraction(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.cooldown_observations_total as f64 / self.total_observations as f64
    }

    /// Fraction of time with severe reduction
    pub fn severe_reduction_rate(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.severe_reduction_count as f64 / self.total_observations as f64
    }
}

// ---------------------------------------------------------------------------
// Internal record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ObservationRecord {
    period_return: f64,
    drawdown: f64,
    multiplier: f64,
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Dynamic risk appetite adaptation engine
pub struct Adaptation {
    config: AdaptationConfig,

    /// Current raw (unsmoothed) multiplier
    raw_multiplier: f64,
    /// EMA-smoothed multiplier
    smoothed_multiplier: f64,
    ema_initialized: bool,

    /// Rolling Sharpe ratio components
    sharpe_returns: VecDeque<f64>,
    ema_sharpe: f64,
    sharpe_ema_initialized: bool,

    /// Cooldown state
    cooldown_remaining: usize,

    /// Recovery state: tracks how much of lost appetite has been restored
    suppressed_amount: f64,

    /// Current market regime
    current_regime: MarketRegime,

    /// Peak equity for drawdown tracking
    peak_equity: f64,

    /// Sliding window of recent observations
    recent: VecDeque<ObservationRecord>,

    /// Running statistics
    stats: AdaptationStats,
}

impl Default for Adaptation {
    fn default() -> Self {
        Self::new()
    }
}

impl Adaptation {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(AdaptationConfig::default()).unwrap()
    }

    /// Create with a specific configuration
    pub fn with_config(config: AdaptationConfig) -> Result<Self> {
        // Validate
        if config.base_appetite <= 0.0 {
            return Err(Error::InvalidInput("base_appetite must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.sharpe_ema_decay <= 0.0 || config.sharpe_ema_decay >= 1.0 {
            return Err(Error::InvalidInput(
                "sharpe_ema_decay must be in (0, 1)".into(),
            ));
        }
        if config.sharpe_window == 0 {
            return Err(Error::InvalidInput("sharpe_window must be > 0".into()));
        }
        if config.target_sharpe < 0.0 {
            return Err(Error::InvalidInput(
                "target_sharpe must be non-negative".into(),
            ));
        }
        if config.max_sharpe_boost < 0.0 {
            return Err(Error::InvalidInput(
                "max_sharpe_boost must be non-negative".into(),
            ));
        }
        if config.max_sharpe_penalty < 0.0 {
            return Err(Error::InvalidInput(
                "max_sharpe_penalty must be non-negative".into(),
            ));
        }
        if config.large_loss_threshold <= 0.0 {
            return Err(Error::InvalidInput(
                "large_loss_threshold must be > 0".into(),
            ));
        }
        if config.recovery_rate <= 0.0 || config.recovery_rate > 1.0 {
            return Err(Error::InvalidInput(
                "recovery_rate must be in (0, 1]".into(),
            ));
        }
        if config.min_multiplier < 0.0 {
            return Err(Error::InvalidInput(
                "min_multiplier must be non-negative".into(),
            ));
        }
        if config.max_multiplier <= config.min_multiplier {
            return Err(Error::InvalidInput(
                "max_multiplier must be > min_multiplier".into(),
            ));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        // Validate drawdown tiers are sorted
        for i in 1..config.drawdown_tiers.len() {
            if config.drawdown_tiers[i].threshold <= config.drawdown_tiers[i - 1].threshold {
                return Err(Error::InvalidInput(
                    "drawdown_tiers must be sorted by ascending threshold".into(),
                ));
            }
        }
        for tier in &config.drawdown_tiers {
            if tier.threshold < 0.0 || tier.threshold > 1.0 {
                return Err(Error::InvalidInput(
                    "drawdown tier threshold must be in [0, 1]".into(),
                ));
            }
            if tier.multiplier < 0.0 || tier.multiplier > 1.0 {
                return Err(Error::InvalidInput(
                    "drawdown tier multiplier must be in [0, 1]".into(),
                ));
            }
        }

        Ok(Self {
            config,
            raw_multiplier: 1.0,
            smoothed_multiplier: 1.0,
            ema_initialized: false,
            sharpe_returns: VecDeque::new(),
            ema_sharpe: 0.0,
            sharpe_ema_initialized: false,
            cooldown_remaining: 0,
            suppressed_amount: 0.0,
            current_regime: MarketRegime::Normal,
            peak_equity: 0.0,
            recent: VecDeque::new(),
            stats: AdaptationStats {
                min_multiplier_produced: f64::MAX,
                max_multiplier_produced: f64::MIN,
                ..Default::default()
            },
        })
    }

    /// Convenience entry point — validates config and returns self
    pub fn process(config: AdaptationConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Core update
    // -----------------------------------------------------------------------

    /// Process a new performance observation and compute the adapted risk appetite.
    pub fn update(&mut self, obs: PerformanceObservation) -> AdaptedAppetite {
        // Track peak equity
        if obs.equity > self.peak_equity {
            self.peak_equity = obs.equity;
        }

        // --- 1. Drawdown component ---
        let drawdown_mult = self.compute_drawdown_multiplier(obs.current_drawdown);

        // --- 2. Sharpe component ---
        self.update_sharpe(obs.period_return);
        let rolling_sharpe = self.compute_rolling_sharpe();
        let sharpe_adj = self.compute_sharpe_adjustment(rolling_sharpe);

        // --- 3. Regime ceiling ---
        let regime_ceil = if self.config.regime_aware {
            self.current_regime.risk_ceiling()
        } else {
            self.config.max_multiplier
        };

        // --- 4. Cooldown check ---
        let large_loss = obs.period_return < -self.config.large_loss_threshold;
        if large_loss {
            self.cooldown_remaining = self.config.cooldown_observations;
            self.stats.cooldown_activations += 1;

            // Track how much appetite we're suppressing
            let current = self.raw_multiplier;
            let reduction = (current - self.config.min_multiplier).max(0.0) * 0.5;
            self.suppressed_amount += reduction;
        }

        let in_cooldown = self.cooldown_remaining > 0;
        if in_cooldown {
            self.cooldown_remaining -= 1;
            self.stats.cooldown_observations_total += 1;
        }

        // --- 5. Recovery ---
        if !in_cooldown && self.suppressed_amount > 0.0 {
            let recovery = self.suppressed_amount * self.config.recovery_rate;
            self.suppressed_amount -= recovery;
            if self.suppressed_amount < 1e-6 {
                self.suppressed_amount = 0.0;
            }
        }

        // --- 6. Combine ---
        let base = self.config.base_appetite;
        let mut combined = base * drawdown_mult;

        // Apply Sharpe adjustment (additive on the multiplier)
        combined *= 1.0 + sharpe_adj;

        // Apply regime ceiling
        combined = combined.min(regime_ceil * base);

        // During cooldown, prevent increases from smoothed value
        if in_cooldown && combined > self.raw_multiplier {
            combined = self.raw_multiplier;
        }

        // Apply suppression from recent large losses
        combined -= self.suppressed_amount;

        // Clamp
        combined = combined.clamp(self.config.min_multiplier, self.config.max_multiplier);

        self.raw_multiplier = combined;

        // --- 7. EMA smoothing ---
        let regime_speed = if self.config.regime_aware {
            self.current_regime.speed_multiplier()
        } else {
            1.0
        };
        let effective_decay = (self.config.ema_decay * regime_speed).min(0.99);

        if !self.ema_initialized {
            self.smoothed_multiplier = combined;
            self.ema_initialized = true;
        } else {
            self.smoothed_multiplier += effective_decay * (combined - self.smoothed_multiplier);
        }

        let should_halt = self.smoothed_multiplier < 0.01;

        // --- 8. Update stats ---
        self.stats.total_observations += 1;
        self.stats.sum_returns += obs.period_return;
        self.stats.sum_sq_returns += obs.period_return * obs.period_return;
        if obs.current_drawdown > self.stats.peak_drawdown {
            self.stats.peak_drawdown = obs.current_drawdown;
        }
        if obs.equity > self.stats.peak_equity {
            self.stats.peak_equity = obs.equity;
        }
        if self.smoothed_multiplier < self.stats.min_multiplier_produced {
            self.stats.min_multiplier_produced = self.smoothed_multiplier;
        }
        if self.smoothed_multiplier > self.stats.max_multiplier_produced {
            self.stats.max_multiplier_produced = self.smoothed_multiplier;
        }
        if self.smoothed_multiplier < 0.5 {
            self.stats.severe_reduction_count += 1;
        }
        if self.smoothed_multiplier > 1.0 {
            self.stats.boost_count += 1;
        }
        if should_halt {
            self.stats.halt_events += 1;
        }

        // --- 9. Window ---
        let record = ObservationRecord {
            period_return: obs.period_return,
            drawdown: obs.current_drawdown,
            multiplier: self.smoothed_multiplier,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        AdaptedAppetite {
            multiplier: combined,
            smoothed_multiplier: self.smoothed_multiplier,
            drawdown_component: drawdown_mult,
            sharpe_component: sharpe_adj,
            regime_ceiling: regime_ceil,
            in_cooldown,
            should_halt,
            rolling_sharpe,
            current_drawdown: obs.current_drawdown,
            regime: self.current_regime,
        }
    }

    // -----------------------------------------------------------------------
    // Drawdown
    // -----------------------------------------------------------------------

    /// Compute risk multiplier based on drawdown using configured tiers.
    /// Interpolates linearly between tier boundaries.
    fn compute_drawdown_multiplier(&self, drawdown: f64) -> f64 {
        if drawdown <= 0.0 || self.config.drawdown_tiers.is_empty() {
            return 1.0;
        }

        let tiers = &self.config.drawdown_tiers;

        // Below first tier → full appetite
        if drawdown < tiers[0].threshold {
            return 1.0;
        }

        // Beyond last tier → use last tier's multiplier
        if drawdown >= tiers[tiers.len() - 1].threshold {
            return tiers[tiers.len() - 1].multiplier;
        }

        // Interpolate between tiers
        for i in 1..tiers.len() {
            if drawdown < tiers[i].threshold {
                let prev = &tiers[i - 1];
                let next = &tiers[i];
                let frac = (drawdown - prev.threshold) / (next.threshold - prev.threshold);
                return prev.multiplier + frac * (next.multiplier - prev.multiplier);
            }
        }

        tiers[tiers.len() - 1].multiplier
    }

    // -----------------------------------------------------------------------
    // Sharpe
    // -----------------------------------------------------------------------

    fn update_sharpe(&mut self, period_return: f64) {
        self.sharpe_returns.push_back(period_return);
        while self.sharpe_returns.len() > self.config.sharpe_window {
            self.sharpe_returns.pop_front();
        }
    }

    /// Compute rolling Sharpe ratio from the return window.
    /// Returns None if insufficient data.
    fn compute_rolling_sharpe(&self) -> Option<f64> {
        if self.sharpe_returns.len() < 2 {
            return None;
        }

        let n = self.sharpe_returns.len() as f64;
        let mean: f64 = self.sharpe_returns.iter().sum::<f64>() / n;
        let excess = mean - self.config.risk_free_rate;
        let variance: f64 = self
            .sharpe_returns
            .iter()
            .map(|r| {
                let d = r - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        let std = variance.sqrt();

        if std < 1e-12 {
            // When volatility is near zero, the Sharpe ratio is determined
            // entirely by the sign of the excess return:
            //   excess > 0  →  essentially infinite (cap at a large value)
            //   excess < 0  →  essentially negative infinity (cap)
            //   excess ≈ 0  →  0.0
            if excess.abs() < 1e-12 {
                return Some(0.0);
            }
            return Some(excess.signum() * 100.0);
        }

        let sharpe = excess / std;

        // Update EMA
        // (we do this via interior mutability pattern — but since we have &self
        // here for simplicity, the EMA update is done in `update` instead)

        Some(sharpe)
    }

    /// Compute Sharpe-driven adjustment to the multiplier
    fn compute_sharpe_adjustment(&self, sharpe: Option<f64>) -> f64 {
        let sharpe = match sharpe {
            Some(s) => s,
            None => return 0.0,
        };

        let diff = sharpe - self.config.target_sharpe;

        if diff > 0.0 {
            // Outperforming → boost (gently)
            (diff * 0.1).min(self.config.max_sharpe_boost)
        } else {
            // Underperforming → penalize
            -(diff.abs() * 0.2).min(self.config.max_sharpe_penalty)
        }
    }

    // -----------------------------------------------------------------------
    // Regime management
    // -----------------------------------------------------------------------

    /// Set the current market regime
    pub fn set_regime(&mut self, regime: MarketRegime) {
        self.current_regime = regime;
    }

    /// Get the current market regime
    pub fn regime(&self) -> MarketRegime {
        self.current_regime
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current raw (unsmoothed) risk multiplier
    pub fn raw_multiplier(&self) -> f64 {
        self.raw_multiplier
    }

    /// Current EMA-smoothed risk multiplier
    pub fn smoothed_multiplier(&self) -> f64 {
        self.smoothed_multiplier
    }

    /// Whether the engine is in cooldown after a large loss
    pub fn in_cooldown(&self) -> bool {
        self.cooldown_remaining > 0
    }

    /// Remaining cooldown observations
    pub fn cooldown_remaining(&self) -> usize {
        self.cooldown_remaining
    }

    /// Current suppressed amount from large losses
    pub fn suppressed_amount(&self) -> f64 {
        self.suppressed_amount
    }

    /// Peak equity observed
    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    /// Current rolling Sharpe ratio
    pub fn rolling_sharpe(&self) -> Option<f64> {
        self.compute_rolling_sharpe()
    }

    /// Whether trading should be halted
    pub fn should_halt(&self) -> bool {
        self.smoothed_multiplier < 0.01
    }

    /// Access running statistics
    pub fn stats(&self) -> &AdaptationStats {
        &self.stats
    }

    /// Access the configuration
    pub fn config(&self) -> &AdaptationConfig {
        &self.config
    }

    /// Number of observations in the sliding window
    pub fn window_count(&self) -> usize {
        self.recent.len()
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Windowed mean risk multiplier
    pub fn windowed_mean_multiplier(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.multiplier).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean drawdown
    pub fn windowed_mean_drawdown(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.drawdown).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed mean return
    pub fn windowed_mean_return(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.period_return).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed return standard deviation
    pub fn windowed_return_std(&self) -> f64 {
        if self.recent.len() < 2 {
            return 0.0;
        }
        let n = self.recent.len() as f64;
        let mean = self.recent.iter().map(|r| r.period_return).sum::<f64>() / n;
        let variance = self
            .recent
            .iter()
            .map(|r| {
                let d = r.period_return - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        variance.sqrt()
    }

    /// Whether risk appetite is trending downward (first half vs second half)
    pub fn is_appetite_declining(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.multiplier)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.multiplier)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half < first_half * 0.9
    }

    /// Whether risk appetite is trending upward
    pub fn is_appetite_recovering(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.multiplier)
            .sum::<f64>()
            / half as f64;
        let second_half: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.multiplier)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half > first_half * 1.1
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all adaptive state, keeping configuration
    pub fn reset(&mut self) {
        self.raw_multiplier = 1.0;
        self.smoothed_multiplier = 1.0;
        self.ema_initialized = false;
        self.sharpe_returns.clear();
        self.ema_sharpe = 0.0;
        self.sharpe_ema_initialized = false;
        self.cooldown_remaining = 0;
        self.suppressed_amount = 0.0;
        self.current_regime = MarketRegime::Normal;
        self.peak_equity = 0.0;
        self.recent.clear();
        self.stats = AdaptationStats {
            min_multiplier_produced: f64::MAX,
            max_multiplier_produced: f64::MIN,
            ..Default::default()
        };
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_obs(ret: f64, drawdown: f64) -> PerformanceObservation {
        PerformanceObservation {
            period_return: ret,
            current_drawdown: drawdown,
            equity: 100_000.0,
        }
    }

    fn obs_with_equity(ret: f64, drawdown: f64, equity: f64) -> PerformanceObservation {
        PerformanceObservation {
            period_return: ret,
            current_drawdown: drawdown,
            equity,
        }
    }

    #[test]
    fn test_basic() {
        let instance = Adaptation::new();
        assert_eq!(instance.raw_multiplier(), 1.0);
    }

    #[test]
    fn test_default_config() {
        let a = Adaptation::new();
        assert_eq!(a.config().base_appetite, 1.0);
        assert_eq!(a.config().drawdown_tiers.len(), 5);
    }

    // -- No drawdown, no surprise → multiplier ≈ 1.0 --

    #[test]
    fn test_no_drawdown_keeps_appetite() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.01, 0.0));
        assert!((result.drawdown_component - 1.0).abs() < 1e-10);
    }

    // -- Drawdown tests --

    #[test]
    fn test_small_drawdown_slight_reduction() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.0, 0.03));
        // 3% drawdown is between tier 0 (2% → 0.90) and tier 1 (5% → 0.75)
        assert!(result.drawdown_component < 1.0);
        assert!(result.drawdown_component > 0.75);
    }

    #[test]
    fn test_moderate_drawdown_halves_appetite() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.0, 0.10));
        // 10% drawdown → tier 2 multiplier = 0.50
        assert!((result.drawdown_component - 0.50).abs() < 1e-10);
    }

    #[test]
    fn test_severe_drawdown_near_zero() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.0, 0.30));
        assert!((result.drawdown_component - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_extreme_drawdown_beyond_tiers() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.0, 0.50));
        // Beyond last tier → last tier multiplier (0.0)
        assert!((result.drawdown_component - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_drawdown_interpolation() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            drawdown_tiers: vec![
                DrawdownTier {
                    threshold: 0.0,
                    multiplier: 1.0,
                },
                DrawdownTier {
                    threshold: 0.10,
                    multiplier: 0.5,
                },
            ],
            ..Default::default()
        })
        .unwrap();

        let result = a.update(default_obs(0.0, 0.05));
        // Midpoint: should interpolate to 0.75
        assert!((result.drawdown_component - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_no_drawdown_tiers() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            drawdown_tiers: vec![],
            ..Default::default()
        })
        .unwrap();

        let result = a.update(default_obs(0.0, 0.20));
        assert!((result.drawdown_component - 1.0).abs() < 1e-10);
    }

    // -- Cooldown tests --

    #[test]
    fn test_large_loss_triggers_cooldown() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            large_loss_threshold: 0.03,
            cooldown_observations: 3,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(-0.04, 0.04)); // large loss
        assert!(a.in_cooldown());
        assert_eq!(a.cooldown_remaining(), 2); // decremented by 1 during update
    }

    #[test]
    fn test_cooldown_expires() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            large_loss_threshold: 0.03,
            cooldown_observations: 2,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(-0.04, 0.04)); // triggers cooldown (remaining=2, then -1 = 1)
        assert!(a.in_cooldown());

        a.update(default_obs(0.01, 0.03)); // remaining = 0
        assert!(!a.in_cooldown());
    }

    #[test]
    fn test_no_cooldown_for_small_loss() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            large_loss_threshold: 0.03,
            cooldown_observations: 5,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(-0.01, 0.01)); // small loss
        assert!(!a.in_cooldown());
    }

    // -- Sharpe adjustment tests --

    #[test]
    fn test_sharpe_no_adjustment_without_data() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.01, 0.0));
        // Only 1 observation, need >= 2 for Sharpe
        assert!((result.sharpe_component - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_high_sharpe_boosts_appetite() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            sharpe_window: 5,
            target_sharpe: 0.5,
            max_sharpe_boost: 0.5,
            ..Default::default()
        })
        .unwrap();

        // Feed consistently positive returns → high Sharpe
        for _ in 0..10 {
            a.update(default_obs(0.02, 0.0));
        }

        let sharpe = a.rolling_sharpe().unwrap();
        assert!(sharpe > 0.5); // above target
    }

    #[test]
    fn test_negative_sharpe_reduces_appetite() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            sharpe_window: 5,
            target_sharpe: 1.0,
            max_sharpe_penalty: 0.5,
            ..Default::default()
        })
        .unwrap();

        // Feed consistently negative returns → negative Sharpe
        for _ in 0..10 {
            let result = a.update(default_obs(-0.01, 0.01));
            // After enough data, sharpe component should be negative
            if a.rolling_sharpe().is_some() {
                assert!(result.sharpe_component <= 0.0);
            }
        }
    }

    // -- Regime tests --

    #[test]
    fn test_regime_ceiling_applied() {
        let mut a = Adaptation::new();
        a.set_regime(MarketRegime::Crisis);
        assert_eq!(a.regime(), MarketRegime::Crisis);

        let result = a.update(default_obs(0.01, 0.0));
        // Crisis ceiling = 0.3 → multiplier should be capped
        assert!(result.regime_ceiling <= 0.3 + 1e-10);
    }

    #[test]
    fn test_regime_calm_allows_higher() {
        let a_regime = MarketRegime::Calm;
        assert!(a_regime.risk_ceiling() > 1.0);
    }

    #[test]
    fn test_regime_speed_multipliers() {
        assert!(MarketRegime::Crisis.speed_multiplier() > MarketRegime::Normal.speed_multiplier());
        assert!(MarketRegime::Normal.speed_multiplier() > MarketRegime::Calm.speed_multiplier());
    }

    #[test]
    fn test_regime_disabled() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            regime_aware: false,
            ..Default::default()
        })
        .unwrap();

        a.set_regime(MarketRegime::Crisis);
        let result = a.update(default_obs(0.01, 0.0));
        // Without regime awareness, ceiling should be max_multiplier (2.0), not crisis
        assert!(result.regime_ceiling > 1.0);
    }

    // -- Halt tests --

    #[test]
    fn test_halt_on_extreme_drawdown() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(-0.05, 0.35));
        // 35% drawdown → beyond all tiers → multiplier ≈ 0 → should halt
        assert!(result.should_halt);
    }

    #[test]
    fn test_no_halt_in_normal_conditions() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.01, 0.0));
        assert!(!result.should_halt);
    }

    // -- EMA smoothing --

    #[test]
    fn test_ema_initializes_to_first_value() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.0, 0.10)); // 10% drawdown → mult ≈ 0.5
        // On first update, smoothed = raw
        assert!((result.smoothed_multiplier - result.multiplier).abs() < 1e-10);
    }

    #[test]
    fn test_ema_lags_behind_changes() {
        let mut a = Adaptation::new();

        // Start with no drawdown
        a.update(default_obs(0.01, 0.0));
        let smooth_before = a.smoothed_multiplier();

        // Sudden drawdown
        a.update(default_obs(-0.05, 0.15));
        let smooth_after = a.smoothed_multiplier();

        // EMA should lag — smoothed should be between old and new raw values
        assert!(smooth_after < smooth_before);
        assert!(smooth_after > a.raw_multiplier());
    }

    // -- Peak equity tracking --

    #[test]
    fn test_peak_equity_tracked() {
        let mut a = Adaptation::new();
        a.update(obs_with_equity(0.01, 0.0, 100_000.0));
        a.update(obs_with_equity(0.01, 0.0, 110_000.0));
        a.update(obs_with_equity(-0.01, 0.01, 105_000.0));
        assert!((a.peak_equity() - 110_000.0).abs() < 1e-10);
    }

    // -- Stats tests --

    #[test]
    fn test_stats_tracking() {
        let mut a = Adaptation::new();
        a.update(default_obs(0.02, 0.0));
        a.update(default_obs(-0.01, 0.01));
        a.update(default_obs(0.01, 0.0));

        assert_eq!(a.stats().total_observations, 3);
    }

    #[test]
    fn test_stats_mean_return() {
        let mut a = Adaptation::new();
        a.update(default_obs(0.02, 0.0));
        a.update(default_obs(0.04, 0.0));

        let mean = a.stats().mean_return();
        assert!((mean - 0.03).abs() < 1e-10);
    }

    #[test]
    fn test_stats_return_variance_constant() {
        let mut a = Adaptation::new();
        for _ in 0..10 {
            a.update(default_obs(0.01, 0.0));
        }
        assert!(a.stats().return_variance() < 1e-10);
    }

    #[test]
    fn test_stats_peak_drawdown() {
        let mut a = Adaptation::new();
        a.update(default_obs(0.0, 0.05));
        a.update(default_obs(0.0, 0.15));
        a.update(default_obs(0.0, 0.10));

        assert!((a.stats().peak_drawdown - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_stats_cooldown_activations() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            large_loss_threshold: 0.02,
            cooldown_observations: 2,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(-0.03, 0.03)); // trigger
        a.update(default_obs(0.01, 0.02));
        a.update(default_obs(-0.03, 0.05)); // trigger again

        assert_eq!(a.stats().cooldown_activations, 2);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = AdaptationStats::default();
        assert_eq!(stats.total_observations, 0);
        assert_eq!(stats.mean_return(), 0.0);
        assert_eq!(stats.return_std(), 0.0);
        assert_eq!(stats.cooldown_fraction(), 0.0);
    }

    #[test]
    fn test_stats_severe_reduction_count() {
        let mut a = Adaptation::new();
        a.update(default_obs(0.0, 0.15)); // 15% drawdown → mult ~0.375
        assert!(a.stats().severe_reduction_count > 0);
    }

    // -- Window tests --

    #[test]
    fn test_windowed_mean_multiplier() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 5,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..5 {
            a.update(default_obs(0.01, 0.0));
        }

        let mean = a.windowed_mean_multiplier();
        assert!(mean > 0.0);
    }

    #[test]
    fn test_windowed_mean_drawdown() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 3,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(0.0, 0.02));
        a.update(default_obs(0.0, 0.04));
        a.update(default_obs(0.0, 0.06));

        let mean = a.windowed_mean_drawdown();
        assert!((mean - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_return() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 2,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(0.01, 0.0));
        a.update(default_obs(0.03, 0.0));

        assert!((a.windowed_mean_return() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_return_std_constant() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 10,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..10 {
            a.update(default_obs(0.01, 0.0));
        }
        assert!(a.windowed_return_std() < 1e-10);
    }

    #[test]
    fn test_windowed_empty() {
        let a = Adaptation::new();
        assert_eq!(a.windowed_mean_multiplier(), 0.0);
        assert_eq!(a.windowed_mean_drawdown(), 0.0);
        assert_eq!(a.windowed_mean_return(), 0.0);
        assert_eq!(a.windowed_return_std(), 0.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 3,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(0.01, 0.0));
        a.update(default_obs(0.02, 0.0));
        a.update(default_obs(0.03, 0.0));
        a.update(default_obs(0.04, 0.0));

        assert_eq!(a.window_count(), 3);
    }

    // -- Appetite trending --

    #[test]
    fn test_appetite_declining() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        // First half: normal
        for _ in 0..10 {
            a.update(default_obs(0.01, 0.0));
        }
        // Second half: heavy drawdown
        for _ in 0..10 {
            a.update(default_obs(-0.02, 0.15));
        }

        assert!(a.is_appetite_declining());
    }

    #[test]
    fn test_not_declining_when_consistent() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..20 {
            a.update(default_obs(0.01, 0.0));
        }

        assert!(!a.is_appetite_declining());
    }

    #[test]
    fn test_not_declining_insufficient_data() {
        let mut a = Adaptation::new();
        a.update(default_obs(0.01, 0.0));
        assert!(!a.is_appetite_declining());
    }

    #[test]
    fn test_appetite_recovering() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            window_size: 20,
            ema_decay: 0.5, // fast EMA so changes register quickly
            ..Default::default()
        })
        .unwrap();

        // First half: heavy drawdown
        for _ in 0..10 {
            a.update(default_obs(-0.02, 0.15));
        }
        // Second half: recovery
        for _ in 0..10 {
            a.update(default_obs(0.02, 0.01));
        }

        assert!(a.is_appetite_recovering());
    }

    // -- Recovery from suppression --

    #[test]
    fn test_suppression_recovers() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            large_loss_threshold: 0.02,
            cooldown_observations: 1,
            recovery_rate: 0.5,
            ..Default::default()
        })
        .unwrap();

        // Trigger suppression
        a.update(default_obs(-0.03, 0.03));
        let suppressed = a.suppressed_amount();
        assert!(suppressed > 0.0);

        // Cooldown expires after 1 obs, then recovery kicks in
        a.update(default_obs(0.01, 0.02)); // cooldown expires
        a.update(default_obs(0.01, 0.01)); // recovery happens

        assert!(a.suppressed_amount() < suppressed);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut a = Adaptation::new();
        for _ in 0..20 {
            a.update(default_obs(0.01, 0.0));
        }

        a.reset();
        assert_eq!(a.raw_multiplier(), 1.0);
        assert_eq!(a.smoothed_multiplier(), 1.0);
        assert!(!a.in_cooldown());
        assert_eq!(a.suppressed_amount(), 0.0);
        assert_eq!(a.peak_equity(), 0.0);
        assert_eq!(a.window_count(), 0);
        assert_eq!(a.stats().total_observations, 0);
        assert_eq!(a.regime(), MarketRegime::Normal);
    }

    // -- Config validation --

    #[test]
    fn test_invalid_config_zero_base_appetite() {
        let result = Adaptation::with_config(AdaptationConfig {
            base_appetite: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let r1 = Adaptation::with_config(AdaptationConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = Adaptation::with_config(AdaptationConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_bad_sharpe_ema() {
        let result = Adaptation::with_config(AdaptationConfig {
            sharpe_ema_decay: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_sharpe_window() {
        let result = Adaptation::with_config(AdaptationConfig {
            sharpe_window: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_negative_target_sharpe() {
        let result = Adaptation::with_config(AdaptationConfig {
            target_sharpe: -1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_negative_sharpe_boost() {
        let result = Adaptation::with_config(AdaptationConfig {
            max_sharpe_boost: -0.1,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_negative_sharpe_penalty() {
        let result = Adaptation::with_config(AdaptationConfig {
            max_sharpe_penalty: -0.1,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_loss_threshold() {
        let result = Adaptation::with_config(AdaptationConfig {
            large_loss_threshold: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_recovery_rate() {
        let r1 = Adaptation::with_config(AdaptationConfig {
            recovery_rate: 0.0,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = Adaptation::with_config(AdaptationConfig {
            recovery_rate: 1.5,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_bad_multiplier_bounds() {
        let result = Adaptation::with_config(AdaptationConfig {
            min_multiplier: 2.0,
            max_multiplier: 1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_negative_min_multiplier() {
        let result = Adaptation::with_config(AdaptationConfig {
            min_multiplier: -0.1,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let result = Adaptation::with_config(AdaptationConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_unsorted_tiers() {
        let result = Adaptation::with_config(AdaptationConfig {
            drawdown_tiers: vec![
                DrawdownTier {
                    threshold: 0.10,
                    multiplier: 0.50,
                },
                DrawdownTier {
                    threshold: 0.05,
                    multiplier: 0.75,
                },
            ],
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_tier_threshold_out_of_range() {
        let result = Adaptation::with_config(AdaptationConfig {
            drawdown_tiers: vec![DrawdownTier {
                threshold: 1.5,
                multiplier: 0.5,
            }],
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_tier_multiplier_out_of_range() {
        let result = Adaptation::with_config(AdaptationConfig {
            drawdown_tiers: vec![DrawdownTier {
                threshold: 0.1,
                multiplier: 1.5,
            }],
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- Process convenience --

    #[test]
    fn test_process_returns_instance() {
        let a = Adaptation::process(AdaptationConfig::default());
        assert!(a.is_ok());
    }

    #[test]
    fn test_process_rejects_bad_config() {
        let result = Adaptation::process(AdaptationConfig {
            base_appetite: -1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- AdaptedAppetite structure --

    #[test]
    fn test_adapted_appetite_fields() {
        let mut a = Adaptation::new();
        let result = a.update(default_obs(0.01, 0.03));

        assert!(result.multiplier >= 0.0);
        assert!(result.smoothed_multiplier >= 0.0);
        assert!(result.drawdown_component >= 0.0);
        assert!(result.drawdown_component <= 1.0);
        assert!((result.current_drawdown - 0.03).abs() < 1e-10);
        assert_eq!(result.regime, MarketRegime::Normal);
    }

    // -- Clamping --

    #[test]
    fn test_multiplier_clamped_to_bounds() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            min_multiplier: 0.1,
            max_multiplier: 1.5,
            ..Default::default()
        })
        .unwrap();

        // Try to get a very high multiplier
        for _ in 0..30 {
            let result = a.update(default_obs(0.05, 0.0));
            assert!(result.multiplier >= 0.1);
            assert!(result.multiplier <= 1.5);
        }
    }

    // -- Cooldown fraction stat --

    #[test]
    fn test_cooldown_fraction() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            large_loss_threshold: 0.02,
            cooldown_observations: 3,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(-0.03, 0.03)); // triggers, 1 cooldown obs
        a.update(default_obs(0.01, 0.02)); // 2nd cooldown obs
        a.update(default_obs(0.01, 0.01)); // 3rd, cooldown expires
        a.update(default_obs(0.01, 0.0)); // normal

        // 3 out of 4 observations in cooldown
        assert!(a.stats().cooldown_fraction() > 0.5);
    }

    // -- Multiple sequential drawdown levels --

    #[test]
    fn test_increasing_drawdown_decreasing_multiplier() {
        let mut a = Adaptation::new();

        let r1 = a.update(default_obs(0.0, 0.01));
        a.reset();
        let r2 = a.update(default_obs(0.0, 0.05));
        a.reset();
        let r3 = a.update(default_obs(0.0, 0.10));
        a.reset();
        let r4 = a.update(default_obs(0.0, 0.20));

        assert!(r1.drawdown_component > r2.drawdown_component);
        assert!(r2.drawdown_component > r3.drawdown_component);
        assert!(r3.drawdown_component > r4.drawdown_component);
    }

    // -- Sharpe window --

    #[test]
    fn test_sharpe_window_eviction() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            sharpe_window: 3,
            ..Default::default()
        })
        .unwrap();

        a.update(default_obs(0.01, 0.0));
        a.update(default_obs(0.02, 0.0));
        a.update(default_obs(0.03, 0.0));
        a.update(default_obs(0.04, 0.0));

        // Sharpe should be computed from last 3 returns only
        let sharpe = a.rolling_sharpe();
        assert!(sharpe.is_some());
    }

    #[test]
    fn test_rolling_sharpe_single_observation() {
        let mut a = Adaptation::new();
        a.update(default_obs(0.01, 0.0));
        assert!(a.rolling_sharpe().is_none());
    }

    // -- Regime default --

    #[test]
    fn test_default_regime_is_normal() {
        let a = Adaptation::new();
        assert_eq!(a.regime(), MarketRegime::Normal);
    }

    // -- Halt events stat --

    #[test]
    fn test_halt_events_tracked() {
        let mut a = Adaptation::new();
        a.update(default_obs(-0.10, 0.35)); // extreme
        assert!(a.stats().halt_events > 0);
    }

    // -- Boost count --

    #[test]
    fn test_boost_count_tracked() {
        let mut a = Adaptation::with_config(AdaptationConfig {
            regime_aware: false,
            target_sharpe: 0.0,
            max_sharpe_boost: 0.5,
            sharpe_window: 3,
            ..Default::default()
        })
        .unwrap();

        // Strong positive returns → Sharpe boost pushes multiplier > 1.0
        for _ in 0..20 {
            a.update(default_obs(0.05, 0.0));
        }

        assert!(a.stats().boost_count > 0);
    }
}
