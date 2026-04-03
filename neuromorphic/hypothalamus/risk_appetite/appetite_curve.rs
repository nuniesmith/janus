//! Risk Appetite Curve — Utility-theory risk appetite modeling
//!
//! Part of the Hypothalamus region
//! Component: risk_appetite
//!
//! Models risk appetite through classical utility theory curves:
//! - **CARA** (Constant Absolute Risk Aversion): `U(x) = -exp(-α·x) / α`
//! - **CRRA** (Constant Relative Risk Aversion): `U(x) = x^(1-γ) / (1-γ)`
//! - **Prospect Theory**: Asymmetric S-shaped value function with loss aversion
//! - **Linear**: Risk-neutral utility `U(x) = x`
//!
//! Features:
//! - Utility computation and marginal utility (first derivative)
//! - Certainty equivalent calculation for risky prospects
//! - Risk premium estimation
//! - Adaptive risk aversion via EMA of realized outcomes
//! - Regime-aware multipliers for dynamic adjustment
//! - Running statistics and windowed diagnostics

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which utility function family to use
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveType {
    /// Constant Absolute Risk Aversion: U(x) = -exp(-α·x) / α
    Cara,
    /// Constant Relative Risk Aversion: U(x) = x^(1-γ) / (1-γ)  (γ ≠ 1)
    Crra,
    /// Prospect Theory (Kahneman & Tversky):
    ///   gains: x^α          losses: -λ·(-x)^β
    ProspectTheory,
    /// Risk-neutral: U(x) = x
    Linear,
}

impl Default for CurveType {
    fn default() -> Self {
        CurveType::Crra
    }
}

/// Configuration for `AppetiteCurve`
#[derive(Debug, Clone)]
pub struct AppetiteCurveConfig {
    /// Utility curve family
    pub curve_type: CurveType,
    /// Risk-aversion coefficient for CARA (α) or CRRA (γ).
    /// Higher values → more risk-averse.
    pub risk_aversion: f64,
    /// Loss-aversion coefficient for Prospect Theory (λ ≥ 1.0).
    /// Tversky & Kahneman canonical value ≈ 2.25.
    pub loss_aversion: f64,
    /// Gain exponent for Prospect Theory (0 < α ≤ 1)
    pub gain_exponent: f64,
    /// Loss exponent for Prospect Theory (0 < β ≤ 1)
    pub loss_exponent: f64,
    /// EMA decay for adaptive risk-aversion tracking (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum number of outcome observations before adaptation engages
    pub min_samples: usize,
    /// Sliding window size for recent outcome records
    pub window_size: usize,
    /// Weight for blending adapted risk-aversion with base (0..=1)
    pub adaptation_weight: f64,
    /// Floor on adaptive risk-aversion (prevents risk-seeking)
    pub min_risk_aversion: f64,
    /// Ceiling on adaptive risk-aversion
    pub max_risk_aversion: f64,
    /// Reference wealth for CRRA (must be > 0)
    pub reference_wealth: f64,
}

impl Default for AppetiteCurveConfig {
    fn default() -> Self {
        Self {
            curve_type: CurveType::Crra,
            risk_aversion: 2.0,
            loss_aversion: 2.25,
            gain_exponent: 0.88,
            loss_exponent: 0.88,
            ema_decay: 0.05,
            min_samples: 10,
            window_size: 100,
            adaptation_weight: 0.3,
            min_risk_aversion: 0.1,
            max_risk_aversion: 20.0,
            reference_wealth: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Output types
// ---------------------------------------------------------------------------

/// Result of evaluating the utility curve at a point
#[derive(Debug, Clone)]
pub struct UtilityEval {
    /// U(x) — the utility value
    pub utility: f64,
    /// U'(x) — marginal utility (first derivative)
    pub marginal_utility: f64,
    /// Effective risk-aversion coefficient used (may be adapted)
    pub effective_risk_aversion: f64,
    /// Whether the adapted coefficient was used
    pub adapted: bool,
}

/// Result of a certainty-equivalent calculation
#[derive(Debug, Clone)]
pub struct CertaintyEquivalent {
    /// The certain payoff that yields the same expected utility
    pub value: f64,
    /// Risk premium = E[x] - CE(x)
    pub risk_premium: f64,
    /// Risk premium as a fraction of E[x]
    pub risk_premium_ratio: f64,
}

/// An observed trading outcome for adaptive calibration
#[derive(Debug, Clone)]
pub struct OutcomeObservation {
    /// The realized P&L of the trade (can be negative)
    pub pnl: f64,
    /// The expected P&L at entry (signal strength × size)
    pub expected_pnl: f64,
}

/// Sizing recommendation based on the utility curve
#[derive(Debug, Clone)]
pub struct SizingRecommendation {
    /// Multiplier on base position size (0..∞, typically 0.2..2.0)
    pub size_multiplier: f64,
    /// Current effective risk-aversion
    pub effective_risk_aversion: f64,
    /// Marginal utility at the proposed payoff
    pub marginal_utility: f64,
    /// Whether the system is in an adapted state
    pub adapted: bool,
    /// Confidence in the recommendation
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics for the appetite curve
#[derive(Debug, Clone, Default)]
pub struct AppetiteCurveStats {
    /// Total utility evaluations
    pub total_evaluations: u64,
    /// Total outcome observations
    pub total_observations: u64,
    /// Sum of observed PnLs
    pub sum_pnl: f64,
    /// Sum of squared observed PnLs
    pub sum_sq_pnl: f64,
    /// Count of positive outcomes
    pub positive_count: u64,
    /// Count of negative outcomes
    pub negative_count: u64,
    /// Maximum single gain
    pub max_gain: f64,
    /// Maximum single loss (stored as negative)
    pub max_loss: f64,
    /// Peak risk-aversion observed
    pub peak_risk_aversion: f64,
    /// Minimum risk-aversion observed
    pub min_risk_aversion_observed: f64,
    /// Number of times adaptation kicked in
    pub adaptation_count: u64,
}

impl AppetiteCurveStats {
    /// Mean observed PnL
    pub fn mean_pnl(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.sum_pnl / self.total_observations as f64
    }

    /// Variance of observed PnL
    pub fn pnl_variance(&self) -> f64 {
        if self.total_observations < 2 {
            return 0.0;
        }
        let n = self.total_observations as f64;
        let mean = self.sum_pnl / n;
        (self.sum_sq_pnl / n - mean * mean).max(0.0)
    }

    /// Standard deviation of observed PnL
    pub fn pnl_std(&self) -> f64 {
        self.pnl_variance().sqrt()
    }

    /// Win rate (fraction of positive outcomes)
    pub fn win_rate(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.positive_count as f64 / self.total_observations as f64
    }
}

// ---------------------------------------------------------------------------
// Internal record
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct OutcomeRecord {
    pnl: f64,
    expected_pnl: f64,
    surprise: f64, // |pnl - expected_pnl| / max(|expected_pnl|, ε)
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Utility-theory risk appetite curve with adaptive risk-aversion
pub struct AppetiteCurve {
    config: AppetiteCurveConfig,
    /// EMA of realized-vs-expected surprise (drives adaptation)
    ema_surprise: f64,
    ema_initialized: bool,
    /// EMA of realized loss magnitude (drives loss-aware adaptation)
    ema_loss_magnitude: f64,
    loss_ema_initialized: bool,
    /// Count of observations ingested
    observation_count: usize,
    /// Sliding window of recent outcomes
    recent: VecDeque<OutcomeRecord>,
    /// Running statistics
    stats: AppetiteCurveStats,
}

impl Default for AppetiteCurve {
    fn default() -> Self {
        Self::new()
    }
}

impl AppetiteCurve {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(AppetiteCurveConfig::default()).unwrap()
    }

    /// Create with a specific configuration
    pub fn with_config(config: AppetiteCurveConfig) -> Result<Self> {
        // Validate configuration
        if config.risk_aversion < 0.0 {
            return Err(Error::InvalidInput(
                "risk_aversion must be non-negative".into(),
            ));
        }
        if config.loss_aversion < 1.0 {
            return Err(Error::InvalidInput("loss_aversion must be >= 1.0".into()));
        }
        if config.gain_exponent <= 0.0 || config.gain_exponent > 1.0 {
            return Err(Error::InvalidInput(
                "gain_exponent must be in (0, 1]".into(),
            ));
        }
        if config.loss_exponent <= 0.0 || config.loss_exponent > 1.0 {
            return Err(Error::InvalidInput(
                "loss_exponent must be in (0, 1]".into(),
            ));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.adaptation_weight < 0.0 || config.adaptation_weight > 1.0 {
            return Err(Error::InvalidInput(
                "adaptation_weight must be in [0, 1]".into(),
            ));
        }
        if config.min_risk_aversion < 0.0 {
            return Err(Error::InvalidInput(
                "min_risk_aversion must be non-negative".into(),
            ));
        }
        if config.max_risk_aversion <= config.min_risk_aversion {
            return Err(Error::InvalidInput(
                "max_risk_aversion must be > min_risk_aversion".into(),
            ));
        }
        if config.reference_wealth <= 0.0 {
            return Err(Error::InvalidInput("reference_wealth must be > 0".into()));
        }

        Ok(Self {
            config,
            ema_surprise: 0.0,
            ema_initialized: false,
            ema_loss_magnitude: 0.0,
            loss_ema_initialized: false,
            observation_count: 0,
            recent: VecDeque::new(),
            stats: AppetiteCurveStats {
                min_risk_aversion_observed: f64::MAX,
                ..Default::default()
            },
        })
    }

    /// Convenience entry point — validates config then returns self
    pub fn process(config: AppetiteCurveConfig) -> Result<Self> {
        Self::with_config(config)
    }

    // -----------------------------------------------------------------------
    // Core utility functions
    // -----------------------------------------------------------------------

    /// Evaluate U(x) and U'(x) at the given payoff
    pub fn evaluate(&mut self, x: f64) -> UtilityEval {
        let rc = self.effective_risk_aversion();
        let adapted = self.is_adapted();

        let (utility, marginal) = match self.config.curve_type {
            CurveType::Cara => self.cara(x, rc),
            CurveType::Crra => self.crra(x, rc),
            CurveType::ProspectTheory => self.prospect(x),
            CurveType::Linear => (x, 1.0),
        };

        self.stats.total_evaluations += 1;

        UtilityEval {
            utility,
            marginal_utility: marginal,
            effective_risk_aversion: rc,
            adapted,
        }
    }

    /// CARA utility: U(x) = -exp(-α·x)/α,  U'(x) = exp(-α·x)
    fn cara(&self, x: f64, alpha: f64) -> (f64, f64) {
        if alpha.abs() < 1e-12 {
            // Degenerate to linear
            return (x, 1.0);
        }
        let exp_val = (-alpha * x).exp();
        let u = -exp_val / alpha;
        let u_prime = exp_val;
        (u, u_prime)
    }

    /// CRRA utility: U(w) = w^(1-γ)/(1-γ),  w = reference_wealth + x
    /// When γ = 1, U(w) = ln(w)
    fn crra(&self, x: f64, gamma: f64) -> (f64, f64) {
        let w = (self.config.reference_wealth + x).max(1e-12);
        if (gamma - 1.0).abs() < 1e-12 {
            // Log utility
            let u = w.ln();
            let u_prime = 1.0 / w;
            return (u, u_prime);
        }
        let one_minus_gamma = 1.0 - gamma;
        let u = w.powf(one_minus_gamma) / one_minus_gamma;
        let u_prime = w.powf(-gamma);
        (u, u_prime)
    }

    /// Prospect Theory value function
    /// gains (x ≥ 0): v(x) = x^α
    /// losses (x < 0): v(x) = -λ·(-x)^β
    fn prospect(&self, x: f64) -> (f64, f64) {
        let alpha = self.config.gain_exponent;
        let beta = self.config.loss_exponent;
        let lambda = self.config.loss_aversion;

        if x >= 0.0 {
            if x < 1e-15 {
                return (0.0, alpha); // limit as x→0+
            }
            let v = x.powf(alpha);
            let v_prime = alpha * x.powf(alpha - 1.0);
            (v, v_prime)
        } else {
            let abs_x = -x;
            if abs_x < 1e-15 {
                return (0.0, lambda * beta); // limit as x→0-
            }
            let v = -lambda * abs_x.powf(beta);
            let v_prime = lambda * beta * abs_x.powf(beta - 1.0);
            (v, v_prime)
        }
    }

    // -----------------------------------------------------------------------
    // Certainty equivalent & risk premium
    // -----------------------------------------------------------------------

    /// Compute the certainty equivalent for a set of equally-likely payoffs.
    ///
    /// CE = U⁻¹(E[U(x)])
    ///
    /// Returns `None` if payoffs is empty or the inverse cannot be computed.
    pub fn certainty_equivalent(&mut self, payoffs: &[f64]) -> Option<CertaintyEquivalent> {
        if payoffs.is_empty() {
            return None;
        }

        let n = payoffs.len() as f64;
        let expected_payoff: f64 = payoffs.iter().sum::<f64>() / n;

        // Compute E[U(x)]
        let rc = self.effective_risk_aversion();
        let mean_utility: f64 = payoffs
            .iter()
            .map(|&x| match self.config.curve_type {
                CurveType::Cara => self.cara(x, rc).0,
                CurveType::Crra => self.crra(x, rc).0,
                CurveType::ProspectTheory => self.prospect(x).0,
                CurveType::Linear => x,
            })
            .sum::<f64>()
            / n;

        // Invert U to get CE
        let ce = match self.config.curve_type {
            CurveType::Cara => self.cara_inverse(mean_utility, rc),
            CurveType::Crra => self.crra_inverse(mean_utility, rc),
            CurveType::ProspectTheory => self.prospect_inverse(mean_utility),
            CurveType::Linear => Some(mean_utility),
        };

        ce.map(|ce_val| {
            let risk_premium = expected_payoff - ce_val;
            let risk_premium_ratio = if expected_payoff.abs() > 1e-12 {
                risk_premium / expected_payoff.abs()
            } else {
                0.0
            };
            CertaintyEquivalent {
                value: ce_val,
                risk_premium,
                risk_premium_ratio,
            }
        })
    }

    /// Inverse CARA: U⁻¹(y) = -ln(-α·y) / α
    fn cara_inverse(&self, y: f64, alpha: f64) -> Option<f64> {
        if alpha.abs() < 1e-12 {
            return Some(y); // linear
        }
        let inner = -alpha * y;
        if inner <= 0.0 {
            return None;
        }
        Some(-inner.ln() / alpha)
    }

    /// Inverse CRRA: U⁻¹(y) = ((1-γ)·y)^(1/(1-γ)) - w_ref
    fn crra_inverse(&self, y: f64, gamma: f64) -> Option<f64> {
        if (gamma - 1.0).abs() < 1e-12 {
            // Log case: U⁻¹(y) = exp(y) - w_ref
            return Some(y.exp() - self.config.reference_wealth);
        }
        let omg = 1.0 - gamma;
        let inner = omg * y;
        if inner <= 0.0 {
            return None;
        }
        let w = inner.powf(1.0 / omg);
        Some(w - self.config.reference_wealth)
    }

    /// Inverse Prospect Theory (gains branch only; approximate for losses)
    fn prospect_inverse(&self, y: f64) -> Option<f64> {
        let alpha = self.config.gain_exponent;
        let beta = self.config.loss_exponent;
        let lambda = self.config.loss_aversion;

        if y >= 0.0 {
            // v(x) = x^α → x = y^(1/α)
            if alpha.abs() < 1e-12 {
                return None;
            }
            Some(y.powf(1.0 / alpha))
        } else {
            // v(x) = -λ·(-x)^β → (-x)^β = -y/λ → -x = (-y/λ)^(1/β)
            if beta.abs() < 1e-12 || lambda.abs() < 1e-12 {
                return None;
            }
            let abs_y = -y;
            let inner = abs_y / lambda;
            Some(-inner.powf(1.0 / beta))
        }
    }

    // -----------------------------------------------------------------------
    // Sizing recommendation
    // -----------------------------------------------------------------------

    /// Given an expected payoff (signal × base_size), recommend a size multiplier.
    ///
    /// Uses marginal utility: higher marginal utility at the expected payoff → larger size.
    /// Normalised so that marginal utility at x=0 gives multiplier=1.0.
    pub fn recommend_size(&mut self, expected_payoff: f64) -> SizingRecommendation {
        let rc = self.effective_risk_aversion();
        let adapted = self.is_adapted();

        // Marginal utility at x = 0 (the baseline)
        let (_, mu_zero) = match self.config.curve_type {
            CurveType::Cara => self.cara(0.0, rc),
            CurveType::Crra => self.crra(0.0, rc),
            CurveType::ProspectTheory => self.prospect(0.0),
            CurveType::Linear => (0.0, 1.0),
        };

        // Marginal utility at the expected payoff
        let (_, mu_payoff) = match self.config.curve_type {
            CurveType::Cara => self.cara(expected_payoff, rc),
            CurveType::Crra => self.crra(expected_payoff, rc),
            CurveType::ProspectTheory => self.prospect(expected_payoff),
            CurveType::Linear => (expected_payoff, 1.0),
        };

        // Size multiplier: ratio of marginal utilities, clamped
        let raw = if mu_zero.abs() < 1e-15 {
            1.0
        } else {
            (mu_payoff / mu_zero).max(0.0)
        };
        let multiplier = raw.clamp(0.01, 5.0);

        let confidence = self.compute_confidence();

        SizingRecommendation {
            size_multiplier: multiplier,
            effective_risk_aversion: rc,
            marginal_utility: mu_payoff,
            adapted,
            confidence,
        }
    }

    // -----------------------------------------------------------------------
    // Adaptive risk-aversion
    // -----------------------------------------------------------------------

    /// Record a realised outcome for adaptive calibration
    pub fn observe(&mut self, obs: OutcomeObservation) {
        let expected_abs = obs.expected_pnl.abs().max(1e-12);
        let surprise = (obs.pnl - obs.expected_pnl).abs() / expected_abs;

        // Update EMA of surprise
        if !self.ema_initialized {
            self.ema_surprise = surprise;
            self.ema_initialized = true;
        } else {
            self.ema_surprise += self.config.ema_decay * (surprise - self.ema_surprise);
        }

        // Update EMA of loss magnitude (only on losses)
        if obs.pnl < 0.0 {
            let loss_mag = (-obs.pnl) / expected_abs;
            if !self.loss_ema_initialized {
                self.ema_loss_magnitude = loss_mag;
                self.loss_ema_initialized = true;
            } else {
                self.ema_loss_magnitude +=
                    self.config.ema_decay * (loss_mag - self.ema_loss_magnitude);
            }
        }

        self.observation_count += 1;

        // Update stats
        self.stats.total_observations += 1;
        self.stats.sum_pnl += obs.pnl;
        self.stats.sum_sq_pnl += obs.pnl * obs.pnl;
        if obs.pnl >= 0.0 {
            self.stats.positive_count += 1;
            if obs.pnl > self.stats.max_gain {
                self.stats.max_gain = obs.pnl;
            }
        } else {
            self.stats.negative_count += 1;
            if obs.pnl < self.stats.max_loss {
                self.stats.max_loss = obs.pnl;
            }
        }

        // Track adapted risk-aversion stats
        if self.is_adapted() {
            let rc = self.effective_risk_aversion();
            self.stats.adaptation_count += 1;
            if rc > self.stats.peak_risk_aversion {
                self.stats.peak_risk_aversion = rc;
            }
            if rc < self.stats.min_risk_aversion_observed {
                self.stats.min_risk_aversion_observed = rc;
            }
        }

        // Window management
        let record = OutcomeRecord {
            pnl: obs.pnl,
            expected_pnl: obs.expected_pnl,
            surprise,
        };
        self.recent.push_back(record);
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }
    }

    /// Effective risk-aversion coefficient, blending base with adaptive signal
    pub fn effective_risk_aversion(&self) -> f64 {
        if !self.is_adapted() {
            return self.config.risk_aversion;
        }

        // Surprise-driven adjustment: higher surprise → increase risk-aversion
        // Loss-magnitude-driven adjustment: larger losses → increase risk-aversion
        let surprise_factor = 1.0 + self.ema_surprise;
        let loss_factor = if self.loss_ema_initialized {
            1.0 + self.ema_loss_magnitude * 0.5
        } else {
            1.0
        };

        let adaptive_ra = self.config.risk_aversion * surprise_factor * loss_factor;
        let blended = self.config.risk_aversion * (1.0 - self.config.adaptation_weight)
            + adaptive_ra * self.config.adaptation_weight;

        blended.clamp(self.config.min_risk_aversion, self.config.max_risk_aversion)
    }

    /// Whether adaptation has enough data to engage
    pub fn is_adapted(&self) -> bool {
        self.observation_count >= self.config.min_samples
    }

    /// Current EMA surprise value
    pub fn ema_surprise(&self) -> f64 {
        if self.ema_initialized {
            self.ema_surprise
        } else {
            0.0
        }
    }

    /// Current EMA loss magnitude
    pub fn ema_loss_magnitude(&self) -> f64 {
        if self.loss_ema_initialized {
            self.ema_loss_magnitude
        } else {
            0.0
        }
    }

    /// Number of outcome observations recorded
    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    /// Access running statistics
    pub fn stats(&self) -> &AppetiteCurveStats {
        &self.stats
    }

    /// Recent outcome records (read-only)
    #[allow(dead_code)]
    pub(crate) fn recent_outcomes(&self) -> &VecDeque<OutcomeRecord> {
        &self.recent
    }

    /// Windowed mean PnL
    pub fn windowed_mean_pnl(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.pnl).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed win rate
    pub fn windowed_win_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let wins = self.recent.iter().filter(|r| r.pnl >= 0.0).count();
        wins as f64 / self.recent.len() as f64
    }

    /// Windowed mean surprise
    pub fn windowed_mean_surprise(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.surprise).sum();
        sum / self.recent.len() as f64
    }

    /// Detect if risk-aversion is trending upward (performance worsening)
    pub fn is_risk_aversion_increasing(&self) -> bool {
        if self.recent.len() < 4 {
            return false;
        }
        let half = self.recent.len() / 2;
        let first_half_surprise: f64 = self
            .recent
            .iter()
            .take(half)
            .map(|r| r.surprise)
            .sum::<f64>()
            / half as f64;
        let second_half_surprise: f64 = self
            .recent
            .iter()
            .skip(half)
            .map(|r| r.surprise)
            .sum::<f64>()
            / (self.recent.len() - half) as f64;
        second_half_surprise > first_half_surprise * 1.1
    }

    /// Confidence score [0, 1] based on observation count and consistency
    fn compute_confidence(&self) -> f64 {
        if self.observation_count == 0 {
            return 0.0;
        }
        // Ramp up with observation count
        let count_factor =
            (self.observation_count as f64 / self.config.min_samples as f64).min(1.0);

        // Reduce confidence if surprise is very high (unstable)
        let surprise_penalty = if self.ema_initialized {
            1.0 / (1.0 + self.ema_surprise)
        } else {
            0.5
        };

        (count_factor * surprise_penalty).clamp(0.0, 1.0)
    }

    /// Access the current configuration
    pub fn config(&self) -> &AppetiteCurveConfig {
        &self.config
    }

    /// Get the curve type
    pub fn curve_type(&self) -> CurveType {
        self.config.curve_type
    }

    /// Reset all adaptive state, keeping configuration
    pub fn reset(&mut self) {
        self.ema_surprise = 0.0;
        self.ema_initialized = false;
        self.ema_loss_magnitude = 0.0;
        self.loss_ema_initialized = false;
        self.observation_count = 0;
        self.recent.clear();
        self.stats = AppetiteCurveStats {
            min_risk_aversion_observed: f64::MAX,
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

    fn default_obs(pnl: f64, expected: f64) -> OutcomeObservation {
        OutcomeObservation {
            pnl,
            expected_pnl: expected,
        }
    }

    #[test]
    fn test_basic() {
        let instance = AppetiteCurve::new();
        assert!(instance.observation_count() == 0);
    }

    #[test]
    fn test_default_config() {
        let ac = AppetiteCurve::new();
        assert_eq!(ac.curve_type(), CurveType::Crra);
        assert_eq!(ac.config().risk_aversion, 2.0);
    }

    // -- CARA tests --

    #[test]
    fn test_cara_utility_positive() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 1.0,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(1.0);
        let expected_u = -(-1.0_f64).exp() / 1.0;
        assert!((eval.utility - expected_u).abs() < 1e-10);
        assert!(eval.marginal_utility > 0.0);
    }

    #[test]
    fn test_cara_utility_zero() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 1.0,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(0.0);
        assert!((eval.utility - (-1.0)).abs() < 1e-10); // -exp(0)/1 = -1
        assert!((eval.marginal_utility - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cara_concavity() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 1.0,
            ..Default::default()
        })
        .unwrap();

        let e1 = ac.evaluate(0.0);
        let e2 = ac.evaluate(1.0);
        // Marginal utility should decrease (concave function)
        assert!(e2.marginal_utility < e1.marginal_utility);
    }

    #[test]
    fn test_cara_degenerate_zero_alpha() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 0.0,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(5.0);
        assert!((eval.utility - 5.0).abs() < 1e-10); // linear
        assert!((eval.marginal_utility - 1.0).abs() < 1e-10);
    }

    // -- CRRA tests --

    #[test]
    fn test_crra_utility_positive() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Crra,
            risk_aversion: 2.0,
            reference_wealth: 1.0,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(1.0); // w = 1 + 1 = 2
        let expected_u = 2.0_f64.powf(-1.0) / (-1.0); // w^(1-γ)/(1-γ) = 2^(-1)/(-1) = -0.5
        assert!((eval.utility - expected_u).abs() < 1e-10);
    }

    #[test]
    fn test_crra_log_utility() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Crra,
            risk_aversion: 1.0,
            reference_wealth: 1.0,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(1.0); // w = 2, U = ln(2)
        assert!((eval.utility - 2.0_f64.ln()).abs() < 1e-10);
        assert!((eval.marginal_utility - 0.5).abs() < 1e-10); // 1/w = 1/2
    }

    #[test]
    fn test_crra_marginal_decreasing() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Crra,
            risk_aversion: 2.0,
            reference_wealth: 1.0,
            ..Default::default()
        })
        .unwrap();

        let e1 = ac.evaluate(0.5);
        let e2 = ac.evaluate(2.0);
        assert!(e2.marginal_utility < e1.marginal_utility);
    }

    // -- Prospect Theory tests --

    #[test]
    fn test_prospect_gains() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::ProspectTheory,
            gain_exponent: 0.88,
            loss_exponent: 0.88,
            loss_aversion: 2.25,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(1.0);
        assert!((eval.utility - 1.0).abs() < 1e-10); // 1^0.88 = 1
        assert!(eval.marginal_utility > 0.0);
    }

    #[test]
    fn test_prospect_losses_more_painful() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::ProspectTheory,
            gain_exponent: 1.0,
            loss_exponent: 1.0,
            loss_aversion: 2.0,
            ..Default::default()
        })
        .unwrap();

        let gain_eval = ac.evaluate(1.0);
        let loss_eval = ac.evaluate(-1.0);
        // |loss utility| should be larger than gain utility (loss aversion)
        assert!(loss_eval.utility.abs() > gain_eval.utility.abs());
    }

    #[test]
    fn test_prospect_zero() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::ProspectTheory,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(0.0);
        assert!((eval.utility - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_prospect_asymmetry() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::ProspectTheory,
            gain_exponent: 0.88,
            loss_exponent: 0.88,
            loss_aversion: 2.25,
            ..Default::default()
        })
        .unwrap();

        let gain = ac.evaluate(10.0);
        let loss = ac.evaluate(-10.0);
        // Losses should hurt more
        assert!(loss.utility.abs() > gain.utility.abs());
    }

    // -- Linear tests --

    #[test]
    fn test_linear_utility() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Linear,
            ..Default::default()
        })
        .unwrap();

        let eval = ac.evaluate(42.0);
        assert!((eval.utility - 42.0).abs() < 1e-10);
        assert!((eval.marginal_utility - 1.0).abs() < 1e-10);
    }

    // -- Certainty equivalent tests --

    #[test]
    fn test_ce_risk_averse_below_expected() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 1.0,
            ..Default::default()
        })
        .unwrap();

        let payoffs = vec![0.0, 2.0]; // E[x] = 1.0
        let ce = ac.certainty_equivalent(&payoffs).unwrap();
        // For risk-averse agents, CE < E[x]
        assert!(ce.value < 1.0);
        assert!(ce.risk_premium > 0.0);
    }

    #[test]
    fn test_ce_linear_equals_expected() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Linear,
            ..Default::default()
        })
        .unwrap();

        let payoffs = vec![0.0, 2.0]; // E[x] = 1.0
        let ce = ac.certainty_equivalent(&payoffs).unwrap();
        assert!((ce.value - 1.0).abs() < 1e-10);
        assert!(ce.risk_premium.abs() < 1e-10);
    }

    #[test]
    fn test_ce_empty_payoffs() {
        let mut ac = AppetiteCurve::new();
        assert!(ac.certainty_equivalent(&[]).is_none());
    }

    #[test]
    fn test_ce_single_payoff() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 1.0,
            ..Default::default()
        })
        .unwrap();

        let ce = ac.certainty_equivalent(&[5.0]).unwrap();
        // No risk → CE should equal the single payoff
        assert!((ce.value - 5.0).abs() < 1e-8);
    }

    #[test]
    fn test_ce_higher_risk_aversion_lower_ce() {
        let payoffs = vec![-1.0, 3.0]; // E[x] = 1.0

        let mut ac_low = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 0.5,
            ..Default::default()
        })
        .unwrap();

        let mut ac_high = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 3.0,
            ..Default::default()
        })
        .unwrap();

        let ce_low = ac_low.certainty_equivalent(&payoffs).unwrap();
        let ce_high = ac_high.certainty_equivalent(&payoffs).unwrap();
        assert!(ce_high.value < ce_low.value);
    }

    #[test]
    fn test_ce_crra() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Crra,
            risk_aversion: 2.0,
            reference_wealth: 10.0,
            ..Default::default()
        })
        .unwrap();

        let payoffs = vec![0.0, 2.0];
        let ce = ac.certainty_equivalent(&payoffs).unwrap();
        assert!(ce.value < 1.0); // risk-averse → CE < E[x]
        assert!(ce.risk_premium > 0.0);
    }

    // -- Sizing recommendation tests --

    #[test]
    fn test_sizing_neutral_at_zero() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 1.0,
            ..Default::default()
        })
        .unwrap();

        let rec = ac.recommend_size(0.0);
        assert!((rec.size_multiplier - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sizing_smaller_for_large_payoff() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 1.0,
            ..Default::default()
        })
        .unwrap();

        let rec_small = ac.recommend_size(0.1);
        let rec_large = ac.recommend_size(2.0);
        // CARA: marginal utility decreases → size multiplier decreases for larger payoffs
        assert!(rec_large.size_multiplier < rec_small.size_multiplier);
    }

    #[test]
    fn test_sizing_linear_always_one() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Linear,
            ..Default::default()
        })
        .unwrap();

        let rec = ac.recommend_size(100.0);
        assert!((rec.size_multiplier - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sizing_clamped() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::Cara,
            risk_aversion: 0.01, // very low risk aversion
            ..Default::default()
        })
        .unwrap();

        let rec = ac.recommend_size(-100.0); // extreme negative
        assert!(rec.size_multiplier >= 0.01);
        assert!(rec.size_multiplier <= 5.0);
    }

    // -- Adaptation tests --

    #[test]
    fn test_not_adapted_below_min_samples() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_samples: 5,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..4 {
            ac.observe(default_obs(1.0, 1.0));
        }
        assert!(!ac.is_adapted());
    }

    #[test]
    fn test_adapted_at_min_samples() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_samples: 5,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..5 {
            ac.observe(default_obs(1.0, 1.0));
        }
        assert!(ac.is_adapted());
    }

    #[test]
    fn test_surprise_increases_risk_aversion() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_samples: 2,
            ema_decay: 0.5,
            adaptation_weight: 1.0, // full adaptation
            ..Default::default()
        })
        .unwrap();

        let base_ra = ac.config().risk_aversion;

        // Observe outcomes with high surprise
        for _ in 0..5 {
            ac.observe(default_obs(-5.0, 1.0)); // expected +1, got -5
        }

        let adapted_ra = ac.effective_risk_aversion();
        assert!(adapted_ra > base_ra);
    }

    #[test]
    fn test_no_surprise_keeps_base() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_samples: 2,
            ema_decay: 0.5,
            adaptation_weight: 1.0,
            ..Default::default()
        })
        .unwrap();

        let base_ra = ac.config().risk_aversion;

        // Observe outcomes exactly as expected
        for _ in 0..5 {
            ac.observe(default_obs(1.0, 1.0));
        }

        let adapted_ra = ac.effective_risk_aversion();
        // Should be very close to base (surprise ≈ 0)
        assert!((adapted_ra - base_ra).abs() < base_ra * 0.5);
    }

    #[test]
    fn test_losses_increase_risk_aversion() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_samples: 2,
            ema_decay: 0.5,
            adaptation_weight: 1.0,
            ..Default::default()
        })
        .unwrap();

        let base_ra = ac.config().risk_aversion;

        // Large losses
        for _ in 0..10 {
            ac.observe(default_obs(-10.0, 1.0));
        }

        assert!(ac.effective_risk_aversion() > base_ra);
    }

    #[test]
    fn test_risk_aversion_clamped() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_samples: 1,
            max_risk_aversion: 5.0,
            adaptation_weight: 1.0,
            ema_decay: 0.9,
            ..Default::default()
        })
        .unwrap();

        // Extreme surprise
        for _ in 0..50 {
            ac.observe(default_obs(-100.0, 0.1));
        }

        assert!(ac.effective_risk_aversion() <= 5.0);
    }

    // -- Stats tests --

    #[test]
    fn test_stats_tracking() {
        let mut ac = AppetiteCurve::new();
        ac.observe(default_obs(2.0, 1.0));
        ac.observe(default_obs(-1.0, 1.0));
        ac.observe(default_obs(3.0, 2.0));

        assert_eq!(ac.stats().total_observations, 3);
        assert_eq!(ac.stats().positive_count, 2);
        assert_eq!(ac.stats().negative_count, 1);
        assert!((ac.stats().max_gain - 3.0).abs() < 1e-10);
        assert!((ac.stats().max_loss - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_stats_mean_pnl() {
        let mut ac = AppetiteCurve::new();
        ac.observe(default_obs(2.0, 1.0));
        ac.observe(default_obs(4.0, 3.0));

        assert!((ac.stats().mean_pnl() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_win_rate() {
        let mut ac = AppetiteCurve::new();
        ac.observe(default_obs(1.0, 1.0));
        ac.observe(default_obs(-1.0, 1.0));
        ac.observe(default_obs(1.0, 1.0));

        assert!((ac.stats().win_rate() - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = AppetiteCurveStats::default();
        assert_eq!(stats.total_observations, 0);
        assert_eq!(stats.mean_pnl(), 0.0);
        assert_eq!(stats.win_rate(), 0.0);
        assert_eq!(stats.pnl_std(), 0.0);
    }

    // -- Window tests --

    #[test]
    fn test_windowed_mean_pnl() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            window_size: 3,
            ..Default::default()
        })
        .unwrap();

        ac.observe(default_obs(1.0, 1.0));
        ac.observe(default_obs(2.0, 1.0));
        ac.observe(default_obs(3.0, 1.0));
        assert!((ac.windowed_mean_pnl() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_window_eviction() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            window_size: 2,
            ..Default::default()
        })
        .unwrap();

        ac.observe(default_obs(10.0, 1.0));
        ac.observe(default_obs(20.0, 1.0));
        ac.observe(default_obs(30.0, 1.0));

        // First observation (10.0) should be evicted
        assert_eq!(ac.recent_outcomes().len(), 2);
        assert!((ac.windowed_mean_pnl() - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_win_rate() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            window_size: 4,
            ..Default::default()
        })
        .unwrap();

        ac.observe(default_obs(1.0, 1.0));
        ac.observe(default_obs(-1.0, 1.0));
        ac.observe(default_obs(1.0, 1.0));
        ac.observe(default_obs(1.0, 1.0));

        assert!((ac.windowed_win_rate() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_mean_empty() {
        let ac = AppetiteCurve::new();
        assert_eq!(ac.windowed_mean_pnl(), 0.0);
    }

    // -- Evaluation count --

    #[test]
    fn test_evaluation_count() {
        let mut ac = AppetiteCurve::new();
        assert_eq!(ac.stats().total_evaluations, 0);
        ac.evaluate(1.0);
        ac.evaluate(2.0);
        assert_eq!(ac.stats().total_evaluations, 2);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut ac = AppetiteCurve::new();
        for i in 0..20 {
            ac.observe(default_obs(i as f64, 1.0));
        }
        ac.evaluate(1.0);
        assert!(ac.observation_count() > 0);
        assert!(ac.stats().total_evaluations > 0);

        ac.reset();
        assert_eq!(ac.observation_count(), 0);
        assert_eq!(ac.stats().total_observations, 0);
        assert_eq!(ac.stats().total_evaluations, 0);
        assert!(!ac.is_adapted());
        assert!(ac.recent_outcomes().is_empty());
    }

    // -- Confidence --

    #[test]
    fn test_confidence_zero_without_data() {
        let ac = AppetiteCurve::new();
        assert_eq!(ac.compute_confidence(), 0.0);
    }

    #[test]
    fn test_confidence_increases_with_samples() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_samples: 10,
            ..Default::default()
        })
        .unwrap();

        let c0 = ac.compute_confidence();
        for _ in 0..5 {
            ac.observe(default_obs(1.0, 1.0));
        }
        let c5 = ac.compute_confidence();
        for _ in 0..5 {
            ac.observe(default_obs(1.0, 1.0));
        }
        let c10 = ac.compute_confidence();

        assert!(c5 > c0);
        assert!(c10 >= c5);
    }

    // -- Risk aversion trending --

    #[test]
    fn test_is_risk_aversion_increasing() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        // First half: low surprise
        for _ in 0..10 {
            ac.observe(default_obs(1.0, 1.0));
        }
        // Second half: high surprise
        for _ in 0..10 {
            ac.observe(default_obs(-5.0, 1.0));
        }

        assert!(ac.is_risk_aversion_increasing());
    }

    #[test]
    fn test_not_increasing_when_consistent() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            window_size: 20,
            ..Default::default()
        })
        .unwrap();

        for _ in 0..20 {
            ac.observe(default_obs(1.0, 1.0));
        }

        assert!(!ac.is_risk_aversion_increasing());
    }

    #[test]
    fn test_not_increasing_insufficient_data() {
        let mut ac = AppetiteCurve::new();
        ac.observe(default_obs(1.0, 1.0));
        assert!(!ac.is_risk_aversion_increasing());
    }

    // -- Config validation --

    #[test]
    fn test_invalid_config_negative_risk_aversion() {
        let result = AppetiteCurve::with_config(AppetiteCurveConfig {
            risk_aversion: -1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_loss_aversion_below_one() {
        let result = AppetiteCurve::with_config(AppetiteCurveConfig {
            loss_aversion: 0.5,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_gain_exponent() {
        let result = AppetiteCurve::with_config(AppetiteCurveConfig {
            gain_exponent: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());

        let result2 = AppetiteCurve::with_config(AppetiteCurveConfig {
            gain_exponent: 1.5,
            ..Default::default()
        });
        assert!(result2.is_err());
    }

    #[test]
    fn test_invalid_config_bad_loss_exponent() {
        let result = AppetiteCurve::with_config(AppetiteCurveConfig {
            loss_exponent: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_ema_decay() {
        let r1 = AppetiteCurve::with_config(AppetiteCurveConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = AppetiteCurve::with_config(AppetiteCurveConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let result = AppetiteCurve::with_config(AppetiteCurveConfig {
            window_size: 0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_bad_adaptation_weight() {
        let r1 = AppetiteCurve::with_config(AppetiteCurveConfig {
            adaptation_weight: -0.1,
            ..Default::default()
        });
        assert!(r1.is_err());

        let r2 = AppetiteCurve::with_config(AppetiteCurveConfig {
            adaptation_weight: 1.1,
            ..Default::default()
        });
        assert!(r2.is_err());
    }

    #[test]
    fn test_invalid_config_bad_risk_bounds() {
        let result = AppetiteCurve::with_config(AppetiteCurveConfig {
            min_risk_aversion: 5.0,
            max_risk_aversion: 3.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config_zero_reference_wealth() {
        let result = AppetiteCurve::with_config(AppetiteCurveConfig {
            reference_wealth: 0.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- Prospect theory CE --

    #[test]
    fn test_ce_prospect_theory() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            curve_type: CurveType::ProspectTheory,
            gain_exponent: 1.0,
            loss_exponent: 1.0,
            loss_aversion: 2.0,
            ..Default::default()
        })
        .unwrap();

        let payoffs = vec![1.0, -1.0]; // E[x] = 0
        let ce = ac.certainty_equivalent(&payoffs).unwrap();
        // With loss aversion = 2, the expected utility = (1 + (-2))/2 = -0.5
        // CE should be negative (the gamble feels bad)
        assert!(ce.value < 0.0);
    }

    // -- Process convenience --

    #[test]
    fn test_process_returns_instance() {
        let ac = AppetiteCurve::process(AppetiteCurveConfig::default());
        assert!(ac.is_ok());
    }

    #[test]
    fn test_process_rejects_bad_config() {
        let result = AppetiteCurve::process(AppetiteCurveConfig {
            risk_aversion: -1.0,
            ..Default::default()
        });
        assert!(result.is_err());
    }

    // -- Pnl variance --

    #[test]
    fn test_pnl_variance_constant() {
        let mut ac = AppetiteCurve::new();
        for _ in 0..10 {
            ac.observe(default_obs(5.0, 5.0));
        }
        assert!(ac.stats().pnl_variance() < 1e-10);
    }

    #[test]
    fn test_pnl_variance_spread() {
        let mut ac = AppetiteCurve::new();
        ac.observe(default_obs(0.0, 1.0));
        ac.observe(default_obs(10.0, 1.0));
        assert!(ac.stats().pnl_variance() > 0.0);
    }

    // -- EMA accessors --

    #[test]
    fn test_ema_surprise_zero_before_observations() {
        let ac = AppetiteCurve::new();
        assert_eq!(ac.ema_surprise(), 0.0);
    }

    #[test]
    fn test_ema_loss_magnitude_zero_without_losses() {
        let mut ac = AppetiteCurve::new();
        for _ in 0..5 {
            ac.observe(default_obs(1.0, 1.0));
        }
        assert_eq!(ac.ema_loss_magnitude(), 0.0);
    }

    #[test]
    fn test_ema_loss_magnitude_nonzero_with_losses() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            ema_decay: 0.5,
            ..Default::default()
        })
        .unwrap();

        ac.observe(default_obs(-2.0, 1.0));
        assert!(ac.ema_loss_magnitude() > 0.0);
    }

    #[test]
    fn test_windowed_mean_surprise() {
        let mut ac = AppetiteCurve::with_config(AppetiteCurveConfig {
            window_size: 5,
            ..Default::default()
        })
        .unwrap();

        // Outcome exactly as expected → surprise ≈ 0
        for _ in 0..5 {
            ac.observe(default_obs(1.0, 1.0));
        }
        assert!(ac.windowed_mean_surprise() < 0.1);
    }
}
