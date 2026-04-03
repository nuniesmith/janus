//! High-level trading strategy and policy engine
//!
//! Part of the Cortex region
//! Component: manager
//!
//! Determines the overall strategic trading stance based on detected market
//! regimes, risk appetite, and portfolio objectives. Produces allocation
//! policy recommendations that downstream components (goal setting, subgoal
//! generation, planning) consume to guide tactical decisions.
//!
//! Key features:
//! - Market regime classification (bull, bear, ranging, crisis, recovery)
//! - Risk stance determination (aggressive, moderate, conservative, defensive)
//! - Allocation policy generation with per-regime target weights
//! - Strategy blend weighting across multiple strategy types
//! - EMA-smoothed tracking of regime transitions and stance changes
//! - Sliding window of recent policy decisions for stability analysis
//! - Running statistics with regime duration and transition tracking
//! - Configurable regime detection thresholds and hysteresis

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Detected market regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    /// Strong upward trend with low volatility
    Bull,
    /// Sustained downward trend
    Bear,
    /// Sideways movement with no clear direction
    Ranging,
    /// High volatility, sharp drawdowns, correlation spikes
    Crisis,
    /// Transition from bear/crisis back toward normalcy
    Recovery,
}

impl MarketRegime {
    /// Numeric index for array-based tracking
    pub fn index(&self) -> usize {
        match self {
            MarketRegime::Bull => 0,
            MarketRegime::Bear => 1,
            MarketRegime::Ranging => 2,
            MarketRegime::Crisis => 3,
            MarketRegime::Recovery => 4,
        }
    }

    /// From numeric index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => MarketRegime::Bull,
            1 => MarketRegime::Bear,
            2 => MarketRegime::Ranging,
            3 => MarketRegime::Crisis,
            4 => MarketRegime::Recovery,
            _ => MarketRegime::Ranging,
        }
    }

    /// Number of regime variants
    pub const COUNT: usize = 5;

    /// Default risk multiplier for the regime (1.0 = neutral)
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            MarketRegime::Bull => 1.2,
            MarketRegime::Bear => 0.6,
            MarketRegime::Ranging => 0.9,
            MarketRegime::Crisis => 0.3,
            MarketRegime::Recovery => 0.8,
        }
    }

    /// Suggested equity allocation fraction for the regime
    pub fn base_equity_allocation(&self) -> f64 {
        match self {
            MarketRegime::Bull => 0.80,
            MarketRegime::Bear => 0.30,
            MarketRegime::Ranging => 0.50,
            MarketRegime::Crisis => 0.15,
            MarketRegime::Recovery => 0.55,
        }
    }
}

/// Risk stance reflecting overall risk appetite
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RiskStance {
    /// Minimal risk tolerance — emergency defensive posture
    Defensive,
    /// Reduced risk tolerance — capital preservation focus
    Conservative,
    /// Normal risk tolerance — balanced approach
    Moderate,
    /// Maximum risk tolerance — seeking alpha aggressively
    Aggressive,
}

impl RiskStance {
    /// Position sizing multiplier (1.0 = full size)
    pub fn sizing_multiplier(&self) -> f64 {
        match self {
            RiskStance::Aggressive => 1.5,
            RiskStance::Moderate => 1.0,
            RiskStance::Conservative => 0.6,
            RiskStance::Defensive => 0.25,
        }
    }

    /// Maximum drawdown tolerance for this stance
    pub fn max_drawdown_tolerance(&self) -> f64 {
        match self {
            RiskStance::Aggressive => 0.25,
            RiskStance::Moderate => 0.15,
            RiskStance::Conservative => 0.08,
            RiskStance::Defensive => 0.03,
        }
    }

    /// Numeric weight for EMA tracking (higher = more aggressive)
    pub fn weight(&self) -> f64 {
        match self {
            RiskStance::Aggressive => 4.0,
            RiskStance::Moderate => 3.0,
            RiskStance::Conservative => 2.0,
            RiskStance::Defensive => 1.0,
        }
    }
}

/// Strategy type that can be blended in the portfolio
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrategyType {
    /// Trend-following / momentum
    Momentum,
    /// Mean-reversion / statistical arbitrage
    MeanReversion,
    /// Carry / yield harvesting
    Carry,
    /// Volatility trading
    Volatility,
    /// Market-making / liquidity provision
    MarketMaking,
}

impl StrategyType {
    /// Number of strategy variants
    pub const COUNT: usize = 5;

    /// Numeric index
    pub fn index(&self) -> usize {
        match self {
            StrategyType::Momentum => 0,
            StrategyType::MeanReversion => 1,
            StrategyType::Carry => 2,
            StrategyType::Volatility => 3,
            StrategyType::MarketMaking => 4,
        }
    }

    /// From numeric index
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => StrategyType::Momentum,
            1 => StrategyType::MeanReversion,
            2 => StrategyType::Carry,
            3 => StrategyType::Volatility,
            4 => StrategyType::MarketMaking,
            _ => StrategyType::Momentum,
        }
    }
}

/// Regime-dependent strategy weights: how much capital to allocate to each
/// strategy type under a given regime.
#[derive(Debug, Clone)]
pub struct RegimeStrategyWeights {
    /// Weights indexed by StrategyType::index()
    pub weights: [f64; StrategyType::COUNT],
}

impl RegimeStrategyWeights {
    /// Create with equal weights
    pub fn equal() -> Self {
        let w = 1.0 / StrategyType::COUNT as f64;
        Self {
            weights: [w; StrategyType::COUNT],
        }
    }

    /// Normalise weights to sum to 1.0
    pub fn normalise(&mut self) {
        let sum: f64 = self.weights.iter().sum();
        if sum > 1e-15 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }

    /// Get weight for a strategy type
    pub fn get(&self, st: StrategyType) -> f64 {
        self.weights[st.index()]
    }

    /// Set weight for a strategy type
    pub fn set(&mut self, st: StrategyType, value: f64) {
        self.weights[st.index()] = value;
    }
}

/// Market observation snapshot used for regime detection
#[derive(Debug, Clone)]
pub struct MarketObservation {
    /// Trailing return over lookback period (e.g. 20-day)
    pub trailing_return: f64,
    /// Current realised volatility (annualised)
    pub realised_volatility: f64,
    /// Average pairwise asset correlation
    pub avg_correlation: f64,
    /// Current drawdown from peak
    pub drawdown: f64,
    /// Rate of change of volatility (positive = increasing)
    pub vol_rate_of_change: f64,
    /// Momentum signal strength (-1 to 1)
    pub momentum_signal: f64,
    /// Mean-reversion signal strength (-1 to 1)
    pub mean_reversion_signal: f64,
    /// Current VaR estimate
    pub var_estimate: f64,
}

impl Default for MarketObservation {
    fn default() -> Self {
        Self {
            trailing_return: 0.0,
            realised_volatility: 0.15,
            avg_correlation: 0.30,
            drawdown: 0.0,
            vol_rate_of_change: 0.0,
            momentum_signal: 0.0,
            mean_reversion_signal: 0.0,
            var_estimate: 0.02,
        }
    }
}

/// Configuration for the strategic policy engine
#[derive(Debug, Clone)]
pub struct StrategicPolicyConfig {
    /// Volatility threshold for crisis detection (annualised)
    pub crisis_vol_threshold: f64,
    /// Drawdown threshold for crisis detection
    pub crisis_drawdown_threshold: f64,
    /// Correlation threshold for crisis detection
    pub crisis_correlation_threshold: f64,
    /// Minimum positive return to classify as bull
    pub bull_return_threshold: f64,
    /// Maximum volatility for bull classification
    pub bull_max_vol: f64,
    /// Minimum negative return to classify as bear
    pub bear_return_threshold: f64,
    /// Recovery: positive return after a bear/crisis period
    pub recovery_return_threshold: f64,
    /// Hysteresis factor: regime must exceed threshold by this factor to switch
    pub hysteresis: f64,
    /// EMA decay for smoothing regime and stance tracking (0 < decay < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent policy decisions
    pub window_size: usize,
    /// Maximum consecutive same-regime evaluations before forcing re-check
    pub max_regime_duration: u64,
    /// Base risk budget (fraction of portfolio)
    pub base_risk_budget: f64,
    /// Minimum cash reserve fraction
    pub min_cash_reserve: f64,
}

impl Default for StrategicPolicyConfig {
    fn default() -> Self {
        Self {
            crisis_vol_threshold: 0.50,
            crisis_drawdown_threshold: 0.15,
            crisis_correlation_threshold: 0.75,
            bull_return_threshold: 0.02,
            bull_max_vol: 0.25,
            bear_return_threshold: -0.03,
            recovery_return_threshold: 0.01,
            hysteresis: 0.8,
            ema_decay: 0.1,
            window_size: 50,
            max_regime_duration: 1000,
            base_risk_budget: 0.02,
            min_cash_reserve: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Policy decision
// ---------------------------------------------------------------------------

/// A single policy decision produced by the engine
#[derive(Debug, Clone)]
pub struct PolicyDecision {
    /// Detected market regime
    pub regime: MarketRegime,
    /// Determined risk stance
    pub risk_stance: RiskStance,
    /// Overall equity allocation target (0.0–1.0)
    pub equity_allocation: f64,
    /// Cash reserve target (0.0–1.0)
    pub cash_reserve: f64,
    /// Strategy weights for the current regime
    pub strategy_weights: RegimeStrategyWeights,
    /// Position sizing multiplier
    pub sizing_multiplier: f64,
    /// Risk budget (fraction of portfolio value)
    pub risk_budget: f64,
    /// Maximum drawdown tolerance
    pub max_drawdown_tolerance: f64,
    /// Whether a regime transition occurred this step
    pub regime_changed: bool,
    /// Steps in the current regime
    pub regime_duration: u64,
    /// Confidence in the regime classification (0.0–1.0)
    pub regime_confidence: f64,
    /// Step number
    pub step: u64,
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the strategic policy engine
#[derive(Debug, Clone)]
pub struct StrategicPolicyStats {
    /// Total evaluations
    pub total_evaluations: u64,
    /// Total regime transitions
    pub total_transitions: u64,
    /// Steps spent in each regime
    pub regime_durations: [u64; MarketRegime::COUNT],
    /// Number of times each regime was entered
    pub regime_entries: [u64; MarketRegime::COUNT],
    /// EMA of risk stance weight
    pub ema_risk_stance: f64,
    /// EMA of equity allocation
    pub ema_equity_allocation: f64,
    /// EMA of regime confidence
    pub ema_regime_confidence: f64,
    /// Average regime duration (exponentially smoothed)
    pub ema_regime_duration: f64,
    /// Current regime streak length
    pub current_regime_duration: u64,
}

impl Default for StrategicPolicyStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_transitions: 0,
            regime_durations: [0; MarketRegime::COUNT],
            regime_entries: [0; MarketRegime::COUNT],
            ema_risk_stance: 3.0, // Moderate
            ema_equity_allocation: 0.50,
            ema_regime_confidence: 0.50,
            ema_regime_duration: 0.0,
            current_regime_duration: 0,
        }
    }
}

impl StrategicPolicyStats {
    /// Most common regime by total time spent
    pub fn dominant_regime(&self) -> MarketRegime {
        let max_idx = self
            .regime_durations
            .iter()
            .enumerate()
            .max_by_key(|(_, d)| *d)
            .map(|(i, _)| i)
            .unwrap_or(2);
        MarketRegime::from_index(max_idx)
    }

    /// Transition frequency (transitions per evaluation)
    pub fn transition_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_transitions as f64 / self.total_evaluations as f64
    }

    /// Fraction of time spent in each regime
    pub fn regime_fractions(&self) -> [f64; MarketRegime::COUNT] {
        let total: u64 = self.regime_durations.iter().sum();
        let mut fractions = [0.0; MarketRegime::COUNT];
        if total > 0 {
            for (i, d) in self.regime_durations.iter().enumerate() {
                fractions[i] = *d as f64 / total as f64;
            }
        }
        fractions
    }
}

// ---------------------------------------------------------------------------
// Default strategy weights per regime
// ---------------------------------------------------------------------------

fn default_regime_weights(regime: MarketRegime) -> RegimeStrategyWeights {
    let mut w = RegimeStrategyWeights {
        weights: [0.0; StrategyType::COUNT],
    };
    match regime {
        MarketRegime::Bull => {
            w.set(StrategyType::Momentum, 0.40);
            w.set(StrategyType::Carry, 0.25);
            w.set(StrategyType::MeanReversion, 0.15);
            w.set(StrategyType::Volatility, 0.10);
            w.set(StrategyType::MarketMaking, 0.10);
        }
        MarketRegime::Bear => {
            w.set(StrategyType::Momentum, 0.30);
            w.set(StrategyType::MeanReversion, 0.10);
            w.set(StrategyType::Carry, 0.05);
            w.set(StrategyType::Volatility, 0.35);
            w.set(StrategyType::MarketMaking, 0.20);
        }
        MarketRegime::Ranging => {
            w.set(StrategyType::MeanReversion, 0.35);
            w.set(StrategyType::MarketMaking, 0.25);
            w.set(StrategyType::Carry, 0.20);
            w.set(StrategyType::Momentum, 0.10);
            w.set(StrategyType::Volatility, 0.10);
        }
        MarketRegime::Crisis => {
            w.set(StrategyType::Volatility, 0.40);
            w.set(StrategyType::Momentum, 0.25);
            w.set(StrategyType::MeanReversion, 0.05);
            w.set(StrategyType::Carry, 0.05);
            w.set(StrategyType::MarketMaking, 0.25);
        }
        MarketRegime::Recovery => {
            w.set(StrategyType::Momentum, 0.30);
            w.set(StrategyType::MeanReversion, 0.25);
            w.set(StrategyType::Carry, 0.20);
            w.set(StrategyType::Volatility, 0.15);
            w.set(StrategyType::MarketMaking, 0.10);
        }
    }
    w
}

// ---------------------------------------------------------------------------
// StrategicPolicy engine
// ---------------------------------------------------------------------------

/// High-level strategic policy engine.
///
/// Evaluates market observations to detect the current regime, determines
/// the appropriate risk stance, and produces allocation policy decisions
/// with strategy blend weights.
pub struct StrategicPolicy {
    config: StrategicPolicyConfig,
    /// Current detected regime
    current_regime: MarketRegime,
    /// Previous regime (for transition detection)
    previous_regime: MarketRegime,
    /// Current risk stance
    current_stance: RiskStance,
    /// Step counter
    step: u64,
    /// Duration of the current regime in steps
    regime_duration: u64,
    /// EMA initialised flag
    ema_initialized: bool,
    /// Sliding window of recent decisions
    recent: VecDeque<PolicyDecision>,
    /// Running statistics
    stats: StrategicPolicyStats,
    /// Custom regime weights overrides (None = use defaults)
    custom_weights: [Option<RegimeStrategyWeights>; MarketRegime::COUNT],
}

impl Default for StrategicPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl StrategicPolicy {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self {
            config: StrategicPolicyConfig::default(),
            current_regime: MarketRegime::Ranging,
            previous_regime: MarketRegime::Ranging,
            current_stance: RiskStance::Moderate,
            step: 0,
            regime_duration: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: StrategicPolicyStats::default(),
            custom_weights: Default::default(),
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: StrategicPolicyConfig) -> Result<Self> {
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.crisis_vol_threshold <= 0.0 {
            return Err(Error::InvalidInput(
                "crisis_vol_threshold must be > 0".into(),
            ));
        }
        if config.base_risk_budget <= 0.0 || config.base_risk_budget > 1.0 {
            return Err(Error::InvalidInput(
                "base_risk_budget must be in (0, 1]".into(),
            ));
        }
        if config.min_cash_reserve < 0.0 || config.min_cash_reserve >= 1.0 {
            return Err(Error::InvalidInput(
                "min_cash_reserve must be in [0, 1)".into(),
            ));
        }
        if config.hysteresis <= 0.0 || config.hysteresis > 1.0 {
            return Err(Error::InvalidInput("hysteresis must be in (0, 1]".into()));
        }
        Ok(Self {
            config,
            current_regime: MarketRegime::Ranging,
            previous_regime: MarketRegime::Ranging,
            current_stance: RiskStance::Moderate,
            step: 0,
            regime_duration: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: StrategicPolicyStats::default(),
            custom_weights: Default::default(),
        })
    }

    /// Main processing function (no-op entry point for trait conformance)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Regime detection
    // -----------------------------------------------------------------------

    /// Detect the market regime from an observation snapshot.
    ///
    /// Classification priority: Crisis > Bear > Bull > Recovery > Ranging
    pub fn detect_regime(&self, obs: &MarketObservation) -> (MarketRegime, f64) {
        let cfg = &self.config;

        // Crisis detection: high vol + high drawdown OR high correlation
        let crisis_score = {
            let vol_breach = if obs.realised_volatility >= cfg.crisis_vol_threshold {
                (obs.realised_volatility - cfg.crisis_vol_threshold) / cfg.crisis_vol_threshold
            } else {
                0.0
            };
            let dd_breach = if obs.drawdown >= cfg.crisis_drawdown_threshold {
                (obs.drawdown - cfg.crisis_drawdown_threshold) / cfg.crisis_drawdown_threshold
            } else {
                0.0
            };
            let corr_breach = if obs.avg_correlation >= cfg.crisis_correlation_threshold {
                (obs.avg_correlation - cfg.crisis_correlation_threshold)
                    / (1.0 - cfg.crisis_correlation_threshold).max(0.01)
            } else {
                0.0
            };
            (vol_breach + dd_breach + corr_breach) / 3.0
        };

        if crisis_score > 0.0 {
            let confidence = (crisis_score * 2.0).min(1.0);
            // Apply hysteresis: if already in crisis, lower bar to stay
            let eff_threshold = if self.current_regime == MarketRegime::Crisis {
                0.0
            } else {
                1.0 - cfg.hysteresis
            };
            if crisis_score > eff_threshold {
                return (MarketRegime::Crisis, confidence);
            }
        }

        // Bear: sustained negative return
        if obs.trailing_return <= cfg.bear_return_threshold {
            let magnitude = (cfg.bear_return_threshold - obs.trailing_return)
                / cfg.bear_return_threshold.abs().max(0.01);
            let confidence = (magnitude * 0.5 + 0.5).min(1.0);
            let eff_threshold = if self.current_regime == MarketRegime::Bear {
                0.0
            } else {
                1.0 - cfg.hysteresis
            };
            if magnitude > eff_threshold {
                return (MarketRegime::Bear, confidence);
            }
        }

        // Bull: positive return + low volatility
        if obs.trailing_return >= cfg.bull_return_threshold
            && obs.realised_volatility <= cfg.bull_max_vol
        {
            let return_score = (obs.trailing_return - cfg.bull_return_threshold)
                / cfg.bull_return_threshold.abs().max(0.01);
            let vol_score =
                (cfg.bull_max_vol - obs.realised_volatility) / cfg.bull_max_vol.max(0.01);
            let score = (return_score + vol_score) / 2.0;
            let confidence = (score * 0.5 + 0.4).min(1.0);
            return (MarketRegime::Bull, confidence);
        }

        // Recovery: coming from bear/crisis with positive return
        if (self.current_regime == MarketRegime::Bear
            || self.current_regime == MarketRegime::Crisis)
            && obs.trailing_return >= cfg.recovery_return_threshold
        {
            let confidence = 0.5_f64.min(
                (obs.trailing_return - cfg.recovery_return_threshold)
                    / cfg.recovery_return_threshold.abs().max(0.01)
                    * 0.3
                    + 0.4,
            );
            return (MarketRegime::Recovery, confidence.max(0.3));
        }

        // Default: ranging
        let confidence =
            0.4 + 0.3 * (1.0 - obs.realised_volatility / cfg.crisis_vol_threshold).max(0.0);
        (MarketRegime::Ranging, confidence.min(0.8))
    }

    /// Determine risk stance given the detected regime and observation
    pub fn determine_stance(&self, regime: MarketRegime, obs: &MarketObservation) -> RiskStance {
        match regime {
            MarketRegime::Crisis => RiskStance::Defensive,
            MarketRegime::Bear => {
                if obs.drawdown > self.config.crisis_drawdown_threshold * 0.8 {
                    RiskStance::Defensive
                } else {
                    RiskStance::Conservative
                }
            }
            MarketRegime::Recovery => RiskStance::Conservative,
            MarketRegime::Ranging => {
                if obs.realised_volatility > self.config.crisis_vol_threshold * 0.6 {
                    RiskStance::Conservative
                } else {
                    RiskStance::Moderate
                }
            }
            MarketRegime::Bull => {
                if obs.momentum_signal > 0.5
                    && obs.realised_volatility < self.config.bull_max_vol * 0.7
                {
                    RiskStance::Aggressive
                } else {
                    RiskStance::Moderate
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Policy evaluation
    // -----------------------------------------------------------------------

    /// Evaluate a market observation and produce a policy decision.
    ///
    /// This is the main entry point for each evaluation cycle.
    pub fn evaluate(&mut self, obs: &MarketObservation) -> PolicyDecision {
        self.step += 1;

        // Detect regime
        let (new_regime, confidence) = self.detect_regime(obs);
        let regime_changed = new_regime != self.current_regime;

        if regime_changed {
            self.previous_regime = self.current_regime;
            self.current_regime = new_regime;

            // Record transition in stats
            self.stats.total_transitions += 1;
            self.stats.regime_entries[new_regime.index()] += 1;

            // EMA the previous regime duration
            let alpha = self.config.ema_decay;
            if self.regime_duration > 0 {
                if self.ema_initialized {
                    self.stats.ema_regime_duration = alpha * self.regime_duration as f64
                        + (1.0 - alpha) * self.stats.ema_regime_duration;
                } else {
                    self.stats.ema_regime_duration = self.regime_duration as f64;
                }
            }

            self.regime_duration = 0;
        }

        self.regime_duration += 1;
        self.stats.regime_durations[self.current_regime.index()] += 1;
        self.stats.current_regime_duration = self.regime_duration;

        // Determine risk stance
        let risk_stance = self.determine_stance(new_regime, obs);
        self.current_stance = risk_stance;

        // Compute allocation targets
        let base_equity = new_regime.base_equity_allocation();
        let stance_mult = risk_stance.sizing_multiplier();
        let equity_allocation = (base_equity * stance_mult).min(1.0 - self.config.min_cash_reserve);
        let cash_reserve = (1.0 - equity_allocation).max(self.config.min_cash_reserve);

        // Strategy weights
        let strategy_weights = self.custom_weights[new_regime.index()]
            .clone()
            .unwrap_or_else(|| default_regime_weights(new_regime));

        // Risk budget
        let risk_budget = self.config.base_risk_budget * new_regime.risk_multiplier() * stance_mult;

        let decision = PolicyDecision {
            regime: new_regime,
            risk_stance,
            equity_allocation,
            cash_reserve,
            strategy_weights,
            sizing_multiplier: stance_mult,
            risk_budget: risk_budget.min(0.10), // cap at 10%
            max_drawdown_tolerance: risk_stance.max_drawdown_tolerance(),
            regime_changed,
            regime_duration: self.regime_duration,
            regime_confidence: confidence,
            step: self.step,
        };

        // Update EMA stats
        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_risk_stance = risk_stance.weight();
            self.stats.ema_equity_allocation = equity_allocation;
            self.stats.ema_regime_confidence = confidence;
            self.ema_initialized = true;
        } else {
            self.stats.ema_risk_stance =
                alpha * risk_stance.weight() + (1.0 - alpha) * self.stats.ema_risk_stance;
            self.stats.ema_equity_allocation =
                alpha * equity_allocation + (1.0 - alpha) * self.stats.ema_equity_allocation;
            self.stats.ema_regime_confidence =
                alpha * confidence + (1.0 - alpha) * self.stats.ema_regime_confidence;
        }

        self.stats.total_evaluations += 1;

        // Sliding window
        if self.recent.len() >= self.config.window_size {
            self.recent.pop_front();
        }
        self.recent.push_back(decision.clone());

        decision
    }

    // -----------------------------------------------------------------------
    // Custom weight overrides
    // -----------------------------------------------------------------------

    /// Override strategy weights for a specific regime
    pub fn set_regime_weights(&mut self, regime: MarketRegime, weights: RegimeStrategyWeights) {
        self.custom_weights[regime.index()] = Some(weights);
    }

    /// Clear custom weight override for a regime (revert to defaults)
    pub fn clear_regime_weights(&mut self, regime: MarketRegime) {
        self.custom_weights[regime.index()] = None;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Current detected regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Previous regime
    pub fn previous_regime(&self) -> MarketRegime {
        self.previous_regime
    }

    /// Current risk stance
    pub fn current_stance(&self) -> RiskStance {
        self.current_stance
    }

    /// Current step
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Duration in the current regime
    pub fn regime_duration(&self) -> u64 {
        self.regime_duration
    }

    /// Running statistics
    pub fn stats(&self) -> &StrategicPolicyStats {
        &self.stats
    }

    /// Configuration
    pub fn config(&self) -> &StrategicPolicyConfig {
        &self.config
    }

    /// Recent decisions (sliding window)
    pub fn recent_decisions(&self) -> &VecDeque<PolicyDecision> {
        &self.recent
    }

    /// EMA-smoothed equity allocation
    pub fn smoothed_equity_allocation(&self) -> f64 {
        self.stats.ema_equity_allocation
    }

    /// EMA-smoothed risk stance weight
    pub fn smoothed_risk_stance(&self) -> f64 {
        self.stats.ema_risk_stance
    }

    /// EMA-smoothed regime confidence
    pub fn smoothed_regime_confidence(&self) -> f64 {
        self.stats.ema_regime_confidence
    }

    /// Windowed average equity allocation
    pub fn windowed_equity_allocation(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|d| d.equity_allocation).sum();
        sum / self.recent.len() as f64
    }

    /// Windowed regime stability: fraction of window with same regime as current
    pub fn windowed_regime_stability(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let same_count = self
            .recent
            .iter()
            .filter(|d| d.regime == self.current_regime)
            .count();
        same_count as f64 / self.recent.len() as f64
    }

    /// Whether risk stance is becoming more defensive over the window
    pub fn is_becoming_defensive(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_half_avg: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|d| d.risk_stance.weight())
            .sum::<f64>()
            / mid as f64;
        let second_half_avg: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|d| d.risk_stance.weight())
            .sum::<f64>()
            / (n - mid) as f64;
        second_half_avg < first_half_avg * 0.9
    }

    /// Whether risk stance is becoming more aggressive over the window
    pub fn is_becoming_aggressive(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_half_avg: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|d| d.risk_stance.weight())
            .sum::<f64>()
            / mid as f64;
        let second_half_avg: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|d| d.risk_stance.weight())
            .sum::<f64>()
            / (n - mid) as f64;
        second_half_avg > first_half_avg * 1.1
    }

    /// Reset state (keeps config and custom weights)
    pub fn reset(&mut self) {
        self.current_regime = MarketRegime::Ranging;
        self.previous_regime = MarketRegime::Ranging;
        self.current_stance = RiskStance::Moderate;
        self.step = 0;
        self.regime_duration = 0;
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = StrategicPolicyStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn bull_observation() -> MarketObservation {
        MarketObservation {
            trailing_return: 0.05,
            realised_volatility: 0.12,
            avg_correlation: 0.25,
            drawdown: 0.01,
            vol_rate_of_change: -0.01,
            momentum_signal: 0.7,
            mean_reversion_signal: -0.1,
            var_estimate: 0.015,
        }
    }

    fn bear_observation() -> MarketObservation {
        MarketObservation {
            trailing_return: -0.08,
            realised_volatility: 0.30,
            avg_correlation: 0.50,
            drawdown: 0.12,
            vol_rate_of_change: 0.05,
            momentum_signal: -0.6,
            mean_reversion_signal: 0.3,
            var_estimate: 0.04,
        }
    }

    fn crisis_observation() -> MarketObservation {
        MarketObservation {
            trailing_return: -0.15,
            realised_volatility: 0.70,
            avg_correlation: 0.85,
            drawdown: 0.25,
            vol_rate_of_change: 0.20,
            momentum_signal: -0.9,
            mean_reversion_signal: 0.1,
            var_estimate: 0.08,
        }
    }

    fn ranging_observation() -> MarketObservation {
        MarketObservation {
            trailing_return: 0.005,
            realised_volatility: 0.18,
            avg_correlation: 0.35,
            drawdown: 0.03,
            vol_rate_of_change: 0.0,
            momentum_signal: 0.1,
            mean_reversion_signal: 0.2,
            var_estimate: 0.02,
        }
    }

    // -- Basic construction --

    #[test]
    fn test_new_default() {
        let sp = StrategicPolicy::new();
        assert_eq!(sp.current_regime(), MarketRegime::Ranging);
        assert_eq!(sp.current_stance(), RiskStance::Moderate);
        assert_eq!(sp.current_step(), 0);
        assert!(sp.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let sp = StrategicPolicy::with_config(StrategicPolicyConfig::default());
        assert!(sp.is_ok());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.ema_decay = 0.0;
        assert!(StrategicPolicy::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.ema_decay = 1.0;
        assert!(StrategicPolicy::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.window_size = 0;
        assert!(StrategicPolicy::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_crisis_vol() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.crisis_vol_threshold = 0.0;
        assert!(StrategicPolicy::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_risk_budget_zero() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.base_risk_budget = 0.0;
        assert!(StrategicPolicy::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_cash_reserve_one() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.min_cash_reserve = 1.0;
        assert!(StrategicPolicy::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_hysteresis_zero() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.hysteresis = 0.0;
        assert!(StrategicPolicy::with_config(cfg).is_err());
    }

    // -- Regime detection --

    #[test]
    fn test_detect_bull() {
        let sp = StrategicPolicy::new();
        let (regime, confidence) = sp.detect_regime(&bull_observation());
        assert_eq!(regime, MarketRegime::Bull);
        assert!(confidence > 0.0);
        assert!(confidence <= 1.0);
    }

    #[test]
    fn test_detect_bear() {
        let sp = StrategicPolicy::new();
        let (regime, _) = sp.detect_regime(&bear_observation());
        assert_eq!(regime, MarketRegime::Bear);
    }

    #[test]
    fn test_detect_crisis() {
        let sp = StrategicPolicy::new();
        let (regime, confidence) = sp.detect_regime(&crisis_observation());
        assert_eq!(regime, MarketRegime::Crisis);
        assert!(confidence > 0.5);
    }

    #[test]
    fn test_detect_ranging() {
        let sp = StrategicPolicy::new();
        let (regime, _) = sp.detect_regime(&ranging_observation());
        assert_eq!(regime, MarketRegime::Ranging);
    }

    #[test]
    fn test_detect_recovery() {
        // Start from crisis, then provide mild positive return
        let mut sp = StrategicPolicy::new();
        // Force into crisis first
        sp.evaluate(&crisis_observation());
        assert_eq!(sp.current_regime(), MarketRegime::Crisis);

        // Now provide a recovery-like observation
        let recovery_obs = MarketObservation {
            trailing_return: 0.015,
            realised_volatility: 0.35,
            avg_correlation: 0.50,
            drawdown: 0.10,
            vol_rate_of_change: -0.05,
            momentum_signal: 0.2,
            mean_reversion_signal: 0.1,
            var_estimate: 0.03,
        };
        let decision = sp.evaluate(&recovery_obs);
        assert_eq!(decision.regime, MarketRegime::Recovery);
    }

    // -- Risk stance --

    #[test]
    fn test_stance_crisis_is_defensive() {
        let sp = StrategicPolicy::new();
        let stance = sp.determine_stance(MarketRegime::Crisis, &crisis_observation());
        assert_eq!(stance, RiskStance::Defensive);
    }

    #[test]
    fn test_stance_bull_with_strong_momentum_is_aggressive() {
        let sp = StrategicPolicy::new();
        let obs = bull_observation(); // momentum_signal = 0.7, vol = 0.12
        let stance = sp.determine_stance(MarketRegime::Bull, &obs);
        assert_eq!(stance, RiskStance::Aggressive);
    }

    #[test]
    fn test_stance_bear_is_conservative_or_defensive() {
        let sp = StrategicPolicy::new();
        let stance = sp.determine_stance(MarketRegime::Bear, &bear_observation());
        assert!(stance == RiskStance::Conservative || stance == RiskStance::Defensive);
    }

    #[test]
    fn test_stance_ranging_is_moderate_or_conservative() {
        let sp = StrategicPolicy::new();
        let stance = sp.determine_stance(MarketRegime::Ranging, &ranging_observation());
        assert!(stance == RiskStance::Moderate || stance == RiskStance::Conservative);
    }

    // -- Policy evaluation --

    #[test]
    fn test_evaluate_produces_decision() {
        let mut sp = StrategicPolicy::new();
        let decision = sp.evaluate(&bull_observation());
        assert_eq!(decision.step, 1);
        assert!(decision.equity_allocation > 0.0);
        assert!(decision.equity_allocation <= 1.0);
        assert!(decision.cash_reserve >= 0.0);
        assert!(decision.risk_budget > 0.0);
    }

    #[test]
    fn test_evaluate_step_increments() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        sp.evaluate(&bull_observation());
        sp.evaluate(&bull_observation());
        assert_eq!(sp.current_step(), 3);
    }

    #[test]
    fn test_evaluate_regime_transition_detected() {
        let mut sp = StrategicPolicy::new();
        let d1 = sp.evaluate(&bull_observation());
        assert_eq!(d1.regime, MarketRegime::Bull);
        assert!(d1.regime_changed); // from initial Ranging to Bull

        let d2 = sp.evaluate(&crisis_observation());
        assert_eq!(d2.regime, MarketRegime::Crisis);
        assert!(d2.regime_changed);
    }

    #[test]
    fn test_evaluate_no_regime_change() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation()); // Ranging -> Bull (change)
        let d2 = sp.evaluate(&bull_observation()); // Bull -> Bull (no change)
        assert!(!d2.regime_changed);
    }

    #[test]
    fn test_evaluate_regime_duration_increments() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        sp.evaluate(&bull_observation());
        let d3 = sp.evaluate(&bull_observation());
        assert_eq!(d3.regime_duration, 3);
    }

    #[test]
    fn test_evaluate_regime_duration_resets_on_transition() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        sp.evaluate(&bull_observation());
        let d3 = sp.evaluate(&crisis_observation());
        assert_eq!(d3.regime_duration, 1); // reset after transition
    }

    #[test]
    fn test_evaluate_equity_allocation_bounded() {
        let mut sp = StrategicPolicy::new();

        let d_bull = sp.evaluate(&bull_observation());
        assert!(d_bull.equity_allocation >= 0.0);
        assert!(d_bull.equity_allocation <= 1.0 - sp.config().min_cash_reserve);

        sp.reset();
        let d_crisis = sp.evaluate(&crisis_observation());
        assert!(d_crisis.equity_allocation >= 0.0);
        assert!(d_crisis.equity_allocation <= 1.0);
    }

    #[test]
    fn test_evaluate_cash_reserve_meets_minimum() {
        let mut sp = StrategicPolicy::new();
        let d = sp.evaluate(&bull_observation());
        assert!(d.cash_reserve >= sp.config().min_cash_reserve);
    }

    #[test]
    fn test_evaluate_crisis_more_conservative_than_bull() {
        let mut sp = StrategicPolicy::new();
        let d_bull = sp.evaluate(&bull_observation());

        sp.reset();
        let d_crisis = sp.evaluate(&crisis_observation());

        // Crisis should have lower equity, higher cash, lower risk budget
        assert!(d_crisis.equity_allocation < d_bull.equity_allocation);
        assert!(d_crisis.cash_reserve >= d_bull.cash_reserve);
        assert!(d_crisis.risk_budget <= d_bull.risk_budget);
    }

    #[test]
    fn test_evaluate_risk_budget_capped() {
        let mut sp = StrategicPolicy::new();
        let d = sp.evaluate(&bull_observation());
        assert!(d.risk_budget <= 0.10);
    }

    #[test]
    fn test_evaluate_strategy_weights_sum_to_one() {
        let mut sp = StrategicPolicy::new();
        let d = sp.evaluate(&bull_observation());
        let sum: f64 = d.strategy_weights.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // -- Strategy weights per regime --

    #[test]
    fn test_default_regime_weights_bull_favors_momentum() {
        let w = default_regime_weights(MarketRegime::Bull);
        assert!(w.get(StrategyType::Momentum) > w.get(StrategyType::MeanReversion));
    }

    #[test]
    fn test_default_regime_weights_crisis_favors_volatility() {
        let w = default_regime_weights(MarketRegime::Crisis);
        assert!(w.get(StrategyType::Volatility) > w.get(StrategyType::Carry));
    }

    #[test]
    fn test_default_regime_weights_ranging_favors_meanrev() {
        let w = default_regime_weights(MarketRegime::Ranging);
        assert!(w.get(StrategyType::MeanReversion) > w.get(StrategyType::Momentum));
    }

    #[test]
    fn test_default_regime_weights_all_sum_to_one() {
        for i in 0..MarketRegime::COUNT {
            let w = default_regime_weights(MarketRegime::from_index(i));
            let sum: f64 = w.weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Regime {:?} weights sum to {} instead of 1.0",
                MarketRegime::from_index(i),
                sum
            );
        }
    }

    // -- Custom weight overrides --

    #[test]
    fn test_custom_weights_override() {
        let mut sp = StrategicPolicy::new();

        let mut custom = RegimeStrategyWeights::equal();
        custom.set(StrategyType::Momentum, 0.90);
        custom.set(StrategyType::MeanReversion, 0.025);
        custom.set(StrategyType::Carry, 0.025);
        custom.set(StrategyType::Volatility, 0.025);
        custom.set(StrategyType::MarketMaking, 0.025);
        sp.set_regime_weights(MarketRegime::Bull, custom.clone());

        let d = sp.evaluate(&bull_observation());
        assert!((d.strategy_weights.get(StrategyType::Momentum) - 0.90).abs() < 1e-10);
    }

    #[test]
    fn test_clear_custom_weights() {
        let mut sp = StrategicPolicy::new();
        let custom = RegimeStrategyWeights::equal();
        sp.set_regime_weights(MarketRegime::Bull, custom);
        sp.clear_regime_weights(MarketRegime::Bull);

        let d = sp.evaluate(&bull_observation());
        // Should use defaults again
        let default_w = default_regime_weights(MarketRegime::Bull);
        assert!(
            (d.strategy_weights.get(StrategyType::Momentum)
                - default_w.get(StrategyType::Momentum))
            .abs()
                < 1e-10
        );
    }

    // -- Statistics --

    #[test]
    fn test_stats_initial() {
        let sp = StrategicPolicy::new();
        assert_eq!(sp.stats().total_evaluations, 0);
        assert_eq!(sp.stats().total_transitions, 0);
    }

    #[test]
    fn test_stats_after_evaluations() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        sp.evaluate(&bull_observation());
        sp.evaluate(&crisis_observation());

        assert_eq!(sp.stats().total_evaluations, 3);
        assert!(sp.stats().total_transitions >= 1); // at least Ranging->Bull
    }

    #[test]
    fn test_stats_regime_durations() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        sp.evaluate(&bull_observation());
        sp.evaluate(&bull_observation());

        assert!(sp.stats().regime_durations[MarketRegime::Bull.index()] >= 3);
    }

    #[test]
    fn test_stats_dominant_regime() {
        let mut sp = StrategicPolicy::new();
        for _ in 0..10 {
            sp.evaluate(&bull_observation());
        }
        assert_eq!(sp.stats().dominant_regime(), MarketRegime::Bull);
    }

    #[test]
    fn test_stats_transition_rate() {
        let stats = StrategicPolicyStats {
            total_evaluations: 100,
            total_transitions: 5,
            ..Default::default()
        };
        assert!((stats.transition_rate() - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_stats_transition_rate_zero_evals() {
        let stats = StrategicPolicyStats::default();
        assert_eq!(stats.transition_rate(), 0.0);
    }

    #[test]
    fn test_stats_regime_fractions() {
        let stats = StrategicPolicyStats {
            regime_durations: [50, 20, 20, 5, 5],
            ..Default::default()
        };
        let fractions = stats.regime_fractions();
        assert!((fractions[0] - 0.50).abs() < 1e-10);
        let sum: f64 = fractions.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    // -- EMA tracking --

    #[test]
    fn test_ema_initializes_on_first_eval() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        assert!(sp.smoothed_equity_allocation() > 0.0);
        assert!(sp.smoothed_regime_confidence() > 0.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent_evals() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        let eq1 = sp.smoothed_equity_allocation();

        sp.evaluate(&crisis_observation());
        let eq2 = sp.smoothed_equity_allocation();
        // Crisis should pull the EMA down
        assert!(eq2 < eq1);
    }

    #[test]
    fn test_ema_risk_stance_tracked() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&crisis_observation());
        // Defensive has weight 1.0
        assert!(sp.smoothed_risk_stance() <= 2.0);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_decisions_stored() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        assert_eq!(sp.recent_decisions().len(), 1);
    }

    #[test]
    fn test_recent_decisions_windowed() {
        let mut cfg = StrategicPolicyConfig::default();
        cfg.window_size = 3;
        let mut sp = StrategicPolicy::with_config(cfg).unwrap();

        for _ in 0..10 {
            sp.evaluate(&bull_observation());
        }
        assert!(sp.recent_decisions().len() <= 3);
    }

    #[test]
    fn test_windowed_equity_allocation() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        let avg = sp.windowed_equity_allocation();
        assert!(avg > 0.0);
    }

    #[test]
    fn test_windowed_equity_allocation_empty() {
        let sp = StrategicPolicy::new();
        assert_eq!(sp.windowed_equity_allocation(), 0.0);
    }

    #[test]
    fn test_windowed_regime_stability() {
        let mut sp = StrategicPolicy::new();
        for _ in 0..5 {
            sp.evaluate(&bull_observation());
        }
        // All decisions should be Bull
        assert!(sp.windowed_regime_stability() > 0.8);
    }

    #[test]
    fn test_windowed_regime_stability_empty() {
        let sp = StrategicPolicy::new();
        assert_eq!(sp.windowed_regime_stability(), 0.0);
    }

    // -- Trend detection --

    #[test]
    fn test_is_becoming_defensive_insufficient_data() {
        let sp = StrategicPolicy::new();
        assert!(!sp.is_becoming_defensive());
    }

    #[test]
    fn test_is_becoming_aggressive_insufficient_data() {
        let sp = StrategicPolicy::new();
        assert!(!sp.is_becoming_aggressive());
    }

    // -- MarketRegime --

    #[test]
    fn test_regime_index_roundtrip() {
        for i in 0..MarketRegime::COUNT {
            let regime = MarketRegime::from_index(i);
            assert_eq!(regime.index(), i);
        }
    }

    #[test]
    fn test_regime_risk_multiplier() {
        assert!(MarketRegime::Bull.risk_multiplier() > MarketRegime::Crisis.risk_multiplier());
        assert!(MarketRegime::Ranging.risk_multiplier() > MarketRegime::Crisis.risk_multiplier());
    }

    #[test]
    fn test_regime_base_equity_allocation() {
        assert!(
            MarketRegime::Bull.base_equity_allocation()
                > MarketRegime::Crisis.base_equity_allocation()
        );
    }

    // -- RiskStance --

    #[test]
    fn test_stance_ordering() {
        assert!(RiskStance::Aggressive > RiskStance::Moderate);
        assert!(RiskStance::Moderate > RiskStance::Conservative);
        assert!(RiskStance::Conservative > RiskStance::Defensive);
    }

    #[test]
    fn test_stance_sizing_multiplier() {
        assert!(
            RiskStance::Aggressive.sizing_multiplier() > RiskStance::Moderate.sizing_multiplier()
        );
        assert!(
            RiskStance::Moderate.sizing_multiplier() > RiskStance::Conservative.sizing_multiplier()
        );
        assert!(
            RiskStance::Conservative.sizing_multiplier()
                > RiskStance::Defensive.sizing_multiplier()
        );
    }

    #[test]
    fn test_stance_max_drawdown_tolerance() {
        assert!(
            RiskStance::Aggressive.max_drawdown_tolerance()
                > RiskStance::Defensive.max_drawdown_tolerance()
        );
    }

    #[test]
    fn test_stance_weight_ordering() {
        assert!(RiskStance::Aggressive.weight() > RiskStance::Moderate.weight());
        assert!(RiskStance::Moderate.weight() > RiskStance::Conservative.weight());
    }

    // -- StrategyType --

    #[test]
    fn test_strategy_type_index_roundtrip() {
        for i in 0..StrategyType::COUNT {
            let st = StrategyType::from_index(i);
            assert_eq!(st.index(), i);
        }
    }

    // -- RegimeStrategyWeights --

    #[test]
    fn test_equal_weights() {
        let w = RegimeStrategyWeights::equal();
        let expected = 1.0 / StrategyType::COUNT as f64;
        for &wt in &w.weights {
            assert!((wt - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_normalise_weights() {
        let mut w = RegimeStrategyWeights {
            weights: [2.0, 3.0, 5.0, 0.0, 0.0],
        };
        w.normalise();
        let sum: f64 = w.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!((w.weights[0] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_normalise_zero_weights() {
        let mut w = RegimeStrategyWeights {
            weights: [0.0; StrategyType::COUNT],
        };
        w.normalise(); // should not panic
        let sum: f64 = w.weights.iter().sum();
        assert!((sum - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_set_weight() {
        let mut w = RegimeStrategyWeights::equal();
        w.set(StrategyType::Momentum, 0.99);
        assert!((w.get(StrategyType::Momentum) - 0.99).abs() < 1e-10);
    }

    // -- MarketObservation default --

    #[test]
    fn test_market_observation_default() {
        let obs = MarketObservation::default();
        assert!(obs.realised_volatility > 0.0);
        assert_eq!(obs.trailing_return, 0.0);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut sp = StrategicPolicy::new();
        sp.evaluate(&bull_observation());
        sp.evaluate(&crisis_observation());

        assert!(sp.current_step() > 0);
        assert!(sp.stats().total_evaluations > 0);
        assert!(!sp.recent_decisions().is_empty());

        sp.reset();

        assert_eq!(sp.current_step(), 0);
        assert_eq!(sp.stats().total_evaluations, 0);
        assert!(sp.recent_decisions().is_empty());
        assert_eq!(sp.current_regime(), MarketRegime::Ranging);
        assert_eq!(sp.current_stance(), RiskStance::Moderate);
        assert_eq!(sp.regime_duration(), 0);
    }

    // -- Integration-style test --

    #[test]
    fn test_full_lifecycle() {
        let mut sp = StrategicPolicy::new();

        // Phase 1: bull market
        for _ in 0..10 {
            let d = sp.evaluate(&bull_observation());
            assert_eq!(d.regime, MarketRegime::Bull);
            assert!(d.equity_allocation > 0.50);
        }

        // Phase 2: transition to crisis
        let d_crisis = sp.evaluate(&crisis_observation());
        assert_eq!(d_crisis.regime, MarketRegime::Crisis);
        assert!(d_crisis.regime_changed);
        assert!(d_crisis.equity_allocation < 0.50);
        assert_eq!(d_crisis.risk_stance, RiskStance::Defensive);

        // Phase 3: stay in crisis
        for _ in 0..5 {
            let d = sp.evaluate(&crisis_observation());
            assert_eq!(d.regime, MarketRegime::Crisis);
            assert!(!d.regime_changed);
        }

        // Phase 4: recovery
        let recovery_obs = MarketObservation {
            trailing_return: 0.015,
            realised_volatility: 0.30,
            avg_correlation: 0.45,
            drawdown: 0.08,
            vol_rate_of_change: -0.05,
            momentum_signal: 0.2,
            mean_reversion_signal: 0.1,
            var_estimate: 0.025,
        };
        let d_recovery = sp.evaluate(&recovery_obs);
        assert_eq!(d_recovery.regime, MarketRegime::Recovery);

        // Verify stats
        assert!(sp.stats().total_evaluations > 15);
        assert!(sp.stats().total_transitions >= 2); // Bull, Crisis, Recovery
        assert!(sp.stats().regime_durations[MarketRegime::Bull.index()] >= 10);
        assert!(sp.stats().regime_durations[MarketRegime::Crisis.index()] >= 5);

        // EMA should reflect the mix
        assert!(sp.smoothed_equity_allocation() > 0.0);
        assert!(sp.smoothed_equity_allocation() < 1.0);
    }
}
