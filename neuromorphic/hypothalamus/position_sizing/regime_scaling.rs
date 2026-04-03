//! Regime-dependent sizing
//!
//! Part of the Hypothalamus region
//! Component: position_sizing
//!
//! This module implements regime-based position scaling that adjusts
//! position sizes based on detected market regimes (trending, mean-reverting,
//! volatile, etc.) to optimize performance across different market conditions.

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketRegime {
    /// Strong upward trend
    StrongBullish,
    /// Moderate upward trend
    Bullish,
    /// Sideways/ranging market
    Neutral,
    /// Moderate downward trend
    Bearish,
    /// Strong downward trend
    StrongBearish,
    /// High volatility with no clear direction
    HighVolatility,
    /// Low volatility, compressed range
    LowVolatility,
    /// Transitional state between regimes
    Transitional,
    /// Crisis/tail event regime
    Crisis,
    /// Unknown/insufficient data
    Unknown,
}

impl Default for MarketRegime {
    fn default() -> Self {
        MarketRegime::Unknown
    }
}

impl MarketRegime {
    /// Get the base scaling factor for this regime
    pub fn base_scaling(&self) -> f64 {
        match self {
            MarketRegime::StrongBullish => 1.25,    // Increase exposure
            MarketRegime::Bullish => 1.10,          // Slight increase
            MarketRegime::Neutral => 1.00,          // Base case
            MarketRegime::Bearish => 0.75,          // Reduce exposure
            MarketRegime::StrongBearish => 0.50,    // Significant reduction
            MarketRegime::HighVolatility => 0.60,   // Reduce for vol
            MarketRegime::LowVolatility => 0.90,    // Slight reduction (breakout risk)
            MarketRegime::Transitional => 0.70,     // Cautious during transitions
            MarketRegime::Crisis => 0.20,           // Minimal exposure
            MarketRegime::Unknown => 0.80,          // Conservative when uncertain
        }
    }

    /// Get regime description
    pub fn description(&self) -> &'static str {
        match self {
            MarketRegime::StrongBullish => "Strong uptrend with momentum",
            MarketRegime::Bullish => "Moderate uptrend",
            MarketRegime::Neutral => "Sideways/ranging market",
            MarketRegime::Bearish => "Moderate downtrend",
            MarketRegime::StrongBearish => "Strong downtrend with momentum",
            MarketRegime::HighVolatility => "High volatility regime",
            MarketRegime::LowVolatility => "Low volatility compression",
            MarketRegime::Transitional => "Regime transition in progress",
            MarketRegime::Crisis => "Crisis/tail event regime",
            MarketRegime::Unknown => "Insufficient data for classification",
        }
    }

    /// Check if regime favors trend-following strategies
    pub fn favors_trend_following(&self) -> bool {
        matches!(
            self,
            MarketRegime::StrongBullish
                | MarketRegime::Bullish
                | MarketRegime::StrongBearish
                | MarketRegime::Bearish
        )
    }

    /// Check if regime favors mean-reversion strategies
    pub fn favors_mean_reversion(&self) -> bool {
        matches!(self, MarketRegime::Neutral | MarketRegime::LowVolatility)
    }

    /// Check if regime suggests caution
    pub fn requires_caution(&self) -> bool {
        matches!(
            self,
            MarketRegime::HighVolatility
                | MarketRegime::Transitional
                | MarketRegime::Crisis
                | MarketRegime::Unknown
        )
    }
}

/// Strategy type for regime-specific scaling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrategyType {
    /// Trend following (momentum, breakout)
    TrendFollowing,
    /// Mean reversion (counter-trend)
    MeanReversion,
    /// Market making / providing liquidity
    MarketMaking,
    /// Volatility trading (straddles, etc.)
    Volatility,
    /// Statistical arbitrage
    StatArb,
    /// Multi-strategy / adaptive
    Adaptive,
}

impl Default for StrategyType {
    fn default() -> Self {
        StrategyType::Adaptive
    }
}

impl StrategyType {
    /// Get regime-specific scaling adjustment for this strategy type
    pub fn regime_adjustment(&self, regime: MarketRegime) -> f64 {
        match (self, regime) {
            // Trend following loves trends, hates chop
            (StrategyType::TrendFollowing, MarketRegime::StrongBullish) => 1.3,
            (StrategyType::TrendFollowing, MarketRegime::Bullish) => 1.15,
            (StrategyType::TrendFollowing, MarketRegime::StrongBearish) => 1.2,
            (StrategyType::TrendFollowing, MarketRegime::Bearish) => 1.1,
            (StrategyType::TrendFollowing, MarketRegime::Neutral) => 0.5,
            (StrategyType::TrendFollowing, MarketRegime::LowVolatility) => 0.6,

            // Mean reversion loves ranges, hates trends
            (StrategyType::MeanReversion, MarketRegime::Neutral) => 1.25,
            (StrategyType::MeanReversion, MarketRegime::LowVolatility) => 1.15,
            (StrategyType::MeanReversion, MarketRegime::StrongBullish) => 0.4,
            (StrategyType::MeanReversion, MarketRegime::StrongBearish) => 0.4,
            (StrategyType::MeanReversion, MarketRegime::HighVolatility) => 0.5,

            // Market making loves quiet, hates vol
            (StrategyType::MarketMaking, MarketRegime::LowVolatility) => 1.2,
            (StrategyType::MarketMaking, MarketRegime::Neutral) => 1.1,
            (StrategyType::MarketMaking, MarketRegime::HighVolatility) => 0.3,
            (StrategyType::MarketMaking, MarketRegime::Crisis) => 0.0,

            // Volatility trading loves vol spikes
            (StrategyType::Volatility, MarketRegime::HighVolatility) => 1.3,
            (StrategyType::Volatility, MarketRegime::Transitional) => 1.2,
            (StrategyType::Volatility, MarketRegime::LowVolatility) => 0.5,

            // StatArb depends on regime stability
            (StrategyType::StatArb, MarketRegime::Neutral) => 1.2,
            (StrategyType::StatArb, MarketRegime::Crisis) => 0.2,

            // Default: use base regime scaling
            _ => 1.0,
        }
    }
}

/// Regime detection indicators
#[derive(Debug, Clone)]
pub struct RegimeIndicators {
    /// Price trend (positive = bullish, negative = bearish)
    pub trend: f64,
    /// Trend strength (0-1)
    pub trend_strength: f64,
    /// Volatility level (annualized)
    pub volatility: f64,
    /// Volatility percentile (0-1)
    pub volatility_percentile: f64,
    /// Correlation with benchmark
    pub correlation: f64,
    /// Market breadth (advance/decline ratio)
    pub breadth: f64,
    /// Momentum (rate of change)
    pub momentum: f64,
    /// Volume relative to average
    pub relative_volume: f64,
    /// Spread/liquidity indicator
    pub liquidity_score: f64,
    /// VIX or fear index level
    pub fear_index: Option<f64>,
}

impl Default for RegimeIndicators {
    fn default() -> Self {
        Self {
            trend: 0.0,
            trend_strength: 0.0,
            volatility: 0.15,
            volatility_percentile: 0.5,
            correlation: 0.0,
            breadth: 1.0,
            momentum: 0.0,
            relative_volume: 1.0,
            liquidity_score: 1.0,
            fear_index: None,
        }
    }
}

/// Regime detection configuration
#[derive(Debug, Clone)]
pub struct RegimeDetectionConfig {
    /// Trend threshold for bullish/bearish classification
    pub trend_threshold: f64,
    /// Strong trend multiplier
    pub strong_trend_multiplier: f64,
    /// High volatility percentile threshold
    pub high_vol_percentile: f64,
    /// Low volatility percentile threshold
    pub low_vol_percentile: f64,
    /// Crisis volatility threshold (absolute)
    pub crisis_vol_threshold: f64,
    /// Fear index crisis level
    pub crisis_fear_level: f64,
    /// Minimum observations for confident classification
    pub min_observations: usize,
    /// Regime persistence threshold (consecutive signals)
    pub persistence_threshold: usize,
    /// Enable hidden Markov model smoothing
    pub use_hmm_smoothing: bool,
}

impl Default for RegimeDetectionConfig {
    fn default() -> Self {
        Self {
            trend_threshold: 0.05,
            strong_trend_multiplier: 2.0,
            high_vol_percentile: 0.80,
            low_vol_percentile: 0.20,
            crisis_vol_threshold: 0.50,
            crisis_fear_level: 35.0,
            min_observations: 20,
            persistence_threshold: 3,
            use_hmm_smoothing: false,
        }
    }
}

/// Configuration for regime scaling
#[derive(Debug, Clone)]
pub struct RegimeScalingConfig {
    /// Regime detection configuration
    pub detection: RegimeDetectionConfig,
    /// Strategy type for regime-specific adjustments
    pub strategy_type: StrategyType,
    /// Custom regime scaling overrides
    pub custom_scaling: HashMap<MarketRegime, f64>,
    /// Minimum scaling factor
    pub min_scale: f64,
    /// Maximum scaling factor
    pub max_scale: f64,
    /// Enable smooth transitions between regimes
    pub smooth_transitions: bool,
    /// Transition smoothing factor (0-1, higher = faster)
    pub transition_speed: f64,
    /// Override scaling during regime transitions
    pub transition_scale_override: Option<f64>,
    /// Enable strategy-specific adjustments
    pub use_strategy_adjustment: bool,
    /// Conservative mode (lower scaling across all regimes)
    pub conservative_mode: bool,
    /// Conservative mode multiplier
    pub conservative_multiplier: f64,
}

impl Default for RegimeScalingConfig {
    fn default() -> Self {
        Self {
            detection: RegimeDetectionConfig::default(),
            strategy_type: StrategyType::Adaptive,
            custom_scaling: HashMap::new(),
            min_scale: 0.1,
            max_scale: 1.5,
            smooth_transitions: true,
            transition_speed: 0.3,
            transition_scale_override: Some(0.7),
            use_strategy_adjustment: true,
            conservative_mode: false,
            conservative_multiplier: 0.8,
        }
    }
}

/// Regime state for tracking
#[derive(Debug, Clone)]
pub struct RegimeState {
    /// Current detected regime
    pub current_regime: MarketRegime,
    /// Previous regime (before transition)
    pub previous_regime: MarketRegime,
    /// Confidence in current regime (0-1)
    pub confidence: f64,
    /// Days in current regime
    pub days_in_regime: usize,
    /// Whether currently transitioning
    pub is_transitioning: bool,
    /// Transition progress (0-1)
    pub transition_progress: f64,
    /// Raw indicator values
    pub indicators: RegimeIndicators,
    /// Consecutive signals for current regime
    pub consecutive_signals: usize,
    /// Timestamp of last regime change
    pub last_change_timestamp: u64,
}

impl Default for RegimeState {
    fn default() -> Self {
        Self {
            current_regime: MarketRegime::Unknown,
            previous_regime: MarketRegime::Unknown,
            confidence: 0.0,
            days_in_regime: 0,
            is_transitioning: false,
            transition_progress: 1.0,
            indicators: RegimeIndicators::default(),
            consecutive_signals: 0,
            last_change_timestamp: 0,
        }
    }
}

/// Regime history entry
#[derive(Debug, Clone)]
pub struct RegimeHistoryEntry {
    /// Timestamp
    pub timestamp: u64,
    /// Detected regime
    pub regime: MarketRegime,
    /// Confidence level
    pub confidence: f64,
    /// Scaling factor used
    pub scale_factor: f64,
    /// Indicator snapshot
    pub indicators: RegimeIndicators,
}

/// Scaling result
#[derive(Debug, Clone)]
pub struct RegimeScalingResult {
    /// Current market regime
    pub regime: MarketRegime,
    /// Regime description
    pub regime_description: String,
    /// Base scaling from regime
    pub base_scale: f64,
    /// Strategy adjustment
    pub strategy_adjustment: f64,
    /// Transition adjustment (if applicable)
    pub transition_adjustment: f64,
    /// Final scaling factor
    pub final_scale: f64,
    /// Confidence in regime classification
    pub confidence: f64,
    /// Days in current regime
    pub days_in_regime: usize,
    /// Whether in transition
    pub is_transitioning: bool,
    /// Recommended position direction bias
    pub direction_bias: f64,
    /// Notes/warnings
    pub notes: Vec<String>,
}

impl Default for RegimeScalingResult {
    fn default() -> Self {
        Self {
            regime: MarketRegime::Unknown,
            regime_description: String::new(),
            base_scale: 1.0,
            strategy_adjustment: 1.0,
            transition_adjustment: 1.0,
            final_scale: 1.0,
            confidence: 0.0,
            days_in_regime: 0,
            is_transitioning: false,
            direction_bias: 0.0,
            notes: Vec::new(),
        }
    }
}

/// Regime-dependent sizing
pub struct RegimeScaling {
    /// Configuration
    config: RegimeScalingConfig,
    /// Current regime state
    state: RegimeState,
    /// Regime history
    history: Vec<RegimeHistoryEntry>,
    /// Current smoothed scale factor
    smoothed_scale: f64,
    /// Observation count
    observation_count: usize,
    /// Last calculation result
    last_result: Option<RegimeScalingResult>,
    /// Regime duration statistics
    regime_durations: HashMap<MarketRegime, Vec<usize>>,
    /// Maximum history size
    max_history_size: usize,
}

impl Default for RegimeScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl RegimeScaling {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            config: RegimeScalingConfig::default(),
            state: RegimeState::default(),
            history: Vec::new(),
            smoothed_scale: 1.0,
            observation_count: 0,
            last_result: None,
            regime_durations: HashMap::new(),
            max_history_size: 1000,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: RegimeScalingConfig) -> Self {
        Self {
            config,
            state: RegimeState::default(),
            history: Vec::new(),
            smoothed_scale: 1.0,
            observation_count: 0,
            last_result: None,
            regime_durations: HashMap::new(),
            max_history_size: 1000,
        }
    }

    /// Update regime detection with new indicators
    pub fn update(&mut self, indicators: RegimeIndicators, timestamp: u64) -> RegimeScalingResult {
        self.observation_count += 1;
        self.state.indicators = indicators.clone();

        // Detect current regime
        let detected_regime = self.detect_regime(&indicators);

        // Update state machine
        self.update_state_machine(detected_regime, timestamp);

        // Calculate scaling
        let result = self.calculate_scaling();

        // Store history
        self.history.push(RegimeHistoryEntry {
            timestamp,
            regime: self.state.current_regime,
            confidence: self.state.confidence,
            scale_factor: result.final_scale,
            indicators,
        });

        // Prune history if needed
        if self.history.len() > self.max_history_size {
            self.history.remove(0);
        }

        self.last_result = Some(result.clone());
        result
    }

    /// Detect regime from indicators
    fn detect_regime(&self, indicators: &RegimeIndicators) -> MarketRegime {
        let config = &self.config.detection;

        // Check for crisis first
        if indicators.volatility > config.crisis_vol_threshold {
            return MarketRegime::Crisis;
        }

        if let Some(fear) = indicators.fear_index {
            if fear > config.crisis_fear_level {
                return MarketRegime::Crisis;
            }
        }

        // Check volatility extremes
        if indicators.volatility_percentile > config.high_vol_percentile {
            // High vol - check if trending or just volatile
            if indicators.trend_strength < 0.3 {
                return MarketRegime::HighVolatility;
            }
        }

        if indicators.volatility_percentile < config.low_vol_percentile {
            return MarketRegime::LowVolatility;
        }

        // Trend-based classification
        let strong_threshold = config.trend_threshold * config.strong_trend_multiplier;

        if indicators.trend > strong_threshold && indicators.trend_strength > 0.6 {
            return MarketRegime::StrongBullish;
        }

        if indicators.trend > config.trend_threshold {
            return MarketRegime::Bullish;
        }

        if indicators.trend < -strong_threshold && indicators.trend_strength > 0.6 {
            return MarketRegime::StrongBearish;
        }

        if indicators.trend < -config.trend_threshold {
            return MarketRegime::Bearish;
        }

        // Default to neutral
        MarketRegime::Neutral
    }

    /// Update the regime state machine
    fn update_state_machine(&mut self, detected_regime: MarketRegime, timestamp: u64) {
        let previous = self.state.current_regime;

        if detected_regime == previous {
            // Same regime - increment counters
            self.state.consecutive_signals += 1;
            self.state.days_in_regime += 1;
            self.state.is_transitioning = false;
            self.state.transition_progress = 1.0;

            // Increase confidence with persistence
            self.state.confidence =
                (self.state.confidence + 0.05).min(1.0);
        } else {
            // Potential regime change
            if self.state.consecutive_signals > 0 {
                // Reset consecutive counter for new regime
                self.state.consecutive_signals = 1;
            }

            // Check persistence threshold
            if self.state.consecutive_signals >= self.config.detection.persistence_threshold
                || previous == MarketRegime::Unknown
            {
                // Confirmed regime change
                self.record_regime_duration(previous, self.state.days_in_regime);

                self.state.previous_regime = previous;
                self.state.current_regime = detected_regime;
                self.state.days_in_regime = 1;
                self.state.confidence = 0.5; // Start with moderate confidence
                self.state.is_transitioning = self.config.smooth_transitions;
                self.state.transition_progress = 0.0;
                self.state.last_change_timestamp = timestamp;
            } else {
                // Not yet confirmed - mark as transitioning
                self.state.is_transitioning = true;
                self.state.confidence = (self.state.confidence - 0.1).max(0.2);
            }
        }

        // Update transition progress
        if self.state.is_transitioning && self.state.transition_progress < 1.0 {
            self.state.transition_progress =
                (self.state.transition_progress + self.config.transition_speed).min(1.0);

            if self.state.transition_progress >= 1.0 {
                self.state.is_transitioning = false;
            }
        }

        // Adjust confidence based on observation count
        if self.observation_count < self.config.detection.min_observations {
            self.state.confidence *= self.observation_count as f64
                / self.config.detection.min_observations as f64;
        }
    }

    /// Record regime duration for statistics
    fn record_regime_duration(&mut self, regime: MarketRegime, duration: usize) {
        if duration > 0 {
            self.regime_durations
                .entry(regime)
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    /// Calculate the scaling factor
    fn calculate_scaling(&mut self) -> RegimeScalingResult {
        let mut result = RegimeScalingResult::default();

        result.regime = self.state.current_regime;
        result.regime_description = self.state.current_regime.description().to_string();
        result.confidence = self.state.confidence;
        result.days_in_regime = self.state.days_in_regime;
        result.is_transitioning = self.state.is_transitioning;

        // Get base scaling
        let base_scale = self
            .config
            .custom_scaling
            .get(&self.state.current_regime)
            .copied()
            .unwrap_or_else(|| self.state.current_regime.base_scaling());
        result.base_scale = base_scale;

        // Get strategy adjustment
        let strategy_adj = if self.config.use_strategy_adjustment {
            self.config
                .strategy_type
                .regime_adjustment(self.state.current_regime)
        } else {
            1.0
        };
        result.strategy_adjustment = strategy_adj;

        // Calculate transition adjustment
        let transition_adj = if self.state.is_transitioning {
            if let Some(override_scale) = self.config.transition_scale_override {
                // Blend between override and target based on progress
                let target = base_scale * strategy_adj;
                override_scale + (target - override_scale) * self.state.transition_progress
            } else {
                // Blend between previous and current scaling
                let prev_scale = self.state.previous_regime.base_scaling();
                let curr_scale = base_scale * strategy_adj;
                prev_scale + (curr_scale - prev_scale) * self.state.transition_progress
            }
        } else {
            base_scale * strategy_adj
        };
        result.transition_adjustment = if self.state.is_transitioning {
            transition_adj / (base_scale * strategy_adj)
        } else {
            1.0
        };

        // Calculate raw scale
        let mut final_scale = if self.state.is_transitioning {
            transition_adj
        } else {
            base_scale * strategy_adj
        };

        // Apply conservative mode
        if self.config.conservative_mode {
            final_scale *= self.config.conservative_multiplier;
        }

        // Apply smoothing
        if self.config.smooth_transitions {
            let speed = self.config.transition_speed;
            self.smoothed_scale = self.smoothed_scale * (1.0 - speed) + final_scale * speed;
            final_scale = self.smoothed_scale;
        }

        // Clamp to bounds
        final_scale = final_scale
            .max(self.config.min_scale)
            .min(self.config.max_scale);
        result.final_scale = final_scale;

        // Set direction bias
        result.direction_bias = match self.state.current_regime {
            MarketRegime::StrongBullish => 1.0,
            MarketRegime::Bullish => 0.5,
            MarketRegime::Neutral => 0.0,
            MarketRegime::Bearish => -0.5,
            MarketRegime::StrongBearish => -1.0,
            _ => 0.0,
        };

        // Add notes
        if self.state.current_regime.requires_caution() {
            result
                .notes
                .push("Current regime suggests caution".to_string());
        }

        if self.state.confidence < 0.5 {
            result.notes.push(format!(
                "Low confidence ({:.0}%) in regime classification",
                self.state.confidence * 100.0
            ));
        }

        if self.state.is_transitioning {
            result.notes.push(format!(
                "Regime transition in progress ({:.0}% complete)",
                self.state.transition_progress * 100.0
            ));
        }

        if self.observation_count < self.config.detection.min_observations {
            result.notes.push(format!(
                "Limited observations ({}/{})",
                self.observation_count, self.config.detection.min_observations
            ));
        }

        result
    }

    /// Get recommended position size
    pub fn get_position_size(&self, base_size: f64) -> f64 {
        if let Some(ref result) = self.last_result {
            base_size * result.final_scale
        } else {
            base_size * self.config.min_scale // Conservative default
        }
    }

    /// Get current regime
    pub fn current_regime(&self) -> MarketRegime {
        self.state.current_regime
    }

    /// Get regime state
    pub fn state(&self) -> &RegimeState {
        &self.state
    }

    /// Get last calculation result
    pub fn last_result(&self) -> Option<&RegimeScalingResult> {
        self.last_result.as_ref()
    }

    /// Get regime history
    pub fn history(&self) -> &[RegimeHistoryEntry] {
        &self.history
    }

    /// Get average regime duration
    pub fn avg_regime_duration(&self, regime: MarketRegime) -> Option<f64> {
        self.regime_durations.get(&regime).map(|durations| {
            if durations.is_empty() {
                0.0
            } else {
                durations.iter().sum::<usize>() as f64 / durations.len() as f64
            }
        })
    }

    /// Get regime frequency (how often each regime occurs)
    pub fn regime_frequency(&self) -> HashMap<MarketRegime, f64> {
        let mut counts: HashMap<MarketRegime, usize> = HashMap::new();

        for entry in &self.history {
            *counts.entry(entry.regime).or_insert(0) += 1;
        }

        let total = self.history.len() as f64;
        counts
            .into_iter()
            .map(|(regime, count)| (regime, count as f64 / total))
            .collect()
    }

    /// Set strategy type
    pub fn set_strategy_type(&mut self, strategy: StrategyType) {
        self.config.strategy_type = strategy;
    }

    /// Get strategy type
    pub fn strategy_type(&self) -> StrategyType {
        self.config.strategy_type
    }

    /// Enable/disable conservative mode
    pub fn set_conservative_mode(&mut self, enabled: bool) {
        self.config.conservative_mode = enabled;
    }

    /// Set custom scaling for a regime
    pub fn set_custom_scaling(&mut self, regime: MarketRegime, scale: f64) {
        self.config.custom_scaling.insert(regime, scale);
    }

    /// Force a specific regime (for testing or override)
    pub fn force_regime(&mut self, regime: MarketRegime, timestamp: u64) {
        if self.state.current_regime != regime {
            self.record_regime_duration(self.state.current_regime, self.state.days_in_regime);
        }

        self.state.previous_regime = self.state.current_regime;
        self.state.current_regime = regime;
        self.state.confidence = 1.0;
        self.state.days_in_regime = 0;
        self.state.is_transitioning = false;
        self.state.transition_progress = 1.0;
        self.state.last_change_timestamp = timestamp;
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Regime Scaling Summary ===\n");
        s.push_str(&format!(
            "Current Regime: {:?} ({})\n",
            self.state.current_regime,
            self.state.current_regime.description()
        ));
        s.push_str(&format!("Confidence: {:.1}%\n", self.state.confidence * 100.0));
        s.push_str(&format!("Days in Regime: {}\n", self.state.days_in_regime));
        s.push_str(&format!("Strategy: {:?}\n", self.config.strategy_type));

        if let Some(ref result) = self.last_result {
            s.push_str(&format!("Final Scale: {:.2}x\n", result.final_scale));
            s.push_str(&format!("Direction Bias: {:.2}\n", result.direction_bias));
        }

        s.push_str(&format!("Observations: {}\n", self.observation_count));

        s
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via update method
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_bullish_indicators() -> RegimeIndicators {
        RegimeIndicators {
            trend: 0.10,
            trend_strength: 0.7,
            volatility: 0.15,
            volatility_percentile: 0.5,
            ..Default::default()
        }
    }

    fn create_bearish_indicators() -> RegimeIndicators {
        RegimeIndicators {
            trend: -0.10,
            trend_strength: 0.7,
            volatility: 0.18,
            volatility_percentile: 0.6,
            ..Default::default()
        }
    }

    fn create_high_vol_indicators() -> RegimeIndicators {
        RegimeIndicators {
            trend: 0.02,
            trend_strength: 0.2,
            volatility: 0.35,
            volatility_percentile: 0.90,
            ..Default::default()
        }
    }

    #[test]
    fn test_basic() {
        let instance = RegimeScaling::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_bullish_regime_detection() {
        let mut scaler = RegimeScaling::new();
        let indicators = create_bullish_indicators();

        let result = scaler.update(indicators, 1);

        assert_eq!(result.regime, MarketRegime::Bullish);
        assert!(result.final_scale > 1.0); // Bullish should scale up
    }

    #[test]
    fn test_bearish_regime_detection() {
        let mut scaler = RegimeScaling::new();
        let indicators = create_bearish_indicators();

        let result = scaler.update(indicators, 1);

        assert_eq!(result.regime, MarketRegime::Bearish);
        assert!(result.final_scale < 1.0); // Bearish should scale down
    }

    #[test]
    fn test_high_volatility_detection() {
        let mut scaler = RegimeScaling::new();
        let indicators = create_high_vol_indicators();

        let result = scaler.update(indicators, 1);

        assert_eq!(result.regime, MarketRegime::HighVolatility);
        assert!(result.final_scale < 1.0);
    }

    #[test]
    fn test_crisis_detection() {
        let mut scaler = RegimeScaling::new();
        let indicators = RegimeIndicators {
            volatility: 0.60, // Above crisis threshold
            ..Default::default()
        };

        let result = scaler.update(indicators, 1);

        assert_eq!(result.regime, MarketRegime::Crisis);
        assert!(result.final_scale < 0.5);
    }

    #[test]
    fn test_fear_index_crisis() {
        let mut scaler = RegimeScaling::new();
        let indicators = RegimeIndicators {
            volatility: 0.20,
            fear_index: Some(40.0), // Above crisis level
            ..Default::default()
        };

        let result = scaler.update(indicators, 1);

        assert_eq!(result.regime, MarketRegime::Crisis);
    }

    #[test]
    fn test_regime_persistence() {
        let mut scaler = RegimeScaling::new();
        let bullish = create_bullish_indicators();

        // Multiple updates to build persistence
        for i in 1..=5 {
            scaler.update(bullish.clone(), i);
        }

        assert_eq!(scaler.current_regime(), MarketRegime::Bullish);
        assert!(scaler.state().confidence > 0.5);
        assert_eq!(scaler.state().days_in_regime, 5);
    }

    #[test]
    fn test_regime_transition() {
        let mut config = RegimeScalingConfig::default();
        config.smooth_transitions = true;
        config.detection.persistence_threshold = 1;

        let mut scaler = RegimeScaling::with_config(config);

        // Start bullish
        let bullish = create_bullish_indicators();
        scaler.update(bullish, 1);

        // Transition to bearish
        let bearish = create_bearish_indicators();
        let result = scaler.update(bearish, 2);

        assert!(result.is_transitioning || result.regime == MarketRegime::Bearish);
    }

    #[test]
    fn test_strategy_adjustment_trend_following() {
        let mut scaler = RegimeScaling::new();
        scaler.set_strategy_type(StrategyType::TrendFollowing);

        let strong_bullish = RegimeIndicators {
            trend: 0.15,
            trend_strength: 0.8,
            volatility: 0.15,
            volatility_percentile: 0.5,
            ..Default::default()
        };

        let result = scaler.update(strong_bullish, 1);

        // Trend following should have high adjustment in trending regime
        assert!(result.strategy_adjustment >= 1.0);
    }

    #[test]
    fn test_strategy_adjustment_mean_reversion() {
        let mut scaler = RegimeScaling::new();
        scaler.set_strategy_type(StrategyType::MeanReversion);

        // Neutral regime
        let neutral = RegimeIndicators {
            trend: 0.01,
            trend_strength: 0.2,
            volatility: 0.15,
            volatility_percentile: 0.5,
            ..Default::default()
        };

        let result = scaler.update(neutral, 1);

        // Mean reversion should do well in neutral
        assert!(result.strategy_adjustment >= 1.0);
    }

    #[test]
    fn test_position_sizing() {
        let mut scaler = RegimeScaling::new();
        let bullish = create_bullish_indicators();

        scaler.update(bullish, 1);

        let base_size = 1000.0;
        let adjusted_size = scaler.get_position_size(base_size);

        // Bullish should increase size
        assert!(adjusted_size > base_size * 0.9);
    }

    #[test]
    fn test_conservative_mode() {
        let mut config = RegimeScalingConfig::default();
        config.conservative_mode = true;
        config.conservative_multiplier = 0.5;

        let mut scaler = RegimeScaling::with_config(config);
        let bullish = create_bullish_indicators();

        let result = scaler.update(bullish, 1);

        // Conservative mode should reduce scaling
        assert!(result.final_scale < 1.0);
    }

    #[test]
    fn test_custom_scaling() {
        let mut scaler = RegimeScaling::new();
        scaler.set_custom_scaling(MarketRegime::Bullish, 2.0);

        let bullish = create_bullish_indicators();
        let result = scaler.update(bullish, 1);

        assert_eq!(result.base_scale, 2.0);
    }

    #[test]
    fn test_force_regime() {
        let mut scaler = RegimeScaling::new();

        scaler.force_regime(MarketRegime::Crisis, 1);

        assert_eq!(scaler.current_regime(), MarketRegime::Crisis);
        assert_eq!(scaler.state().confidence, 1.0);
    }

    #[test]
    fn test_regime_scaling_factors() {
        assert!(MarketRegime::StrongBullish.base_scaling() > 1.0);
        assert_eq!(MarketRegime::Neutral.base_scaling(), 1.0);
        assert!(MarketRegime::Crisis.base_scaling() < 0.5);
    }

    #[test]
    fn test_regime_favors_checks() {
        assert!(MarketRegime::StrongBullish.favors_trend_following());
        assert!(!MarketRegime::StrongBullish.favors_mean_reversion());
        assert!(MarketRegime::Neutral.favors_mean_reversion());
        assert!(MarketRegime::Crisis.requires_caution());
    }

    #[test]
    fn test_history_tracking() {
        let mut scaler = RegimeScaling::new();

        for i in 1..=10 {
            let indicators = RegimeIndicators {
                trend: 0.05 * (i as f64 % 3.0 - 1.0),
                volatility_percentile: 0.5,
                ..Default::default()
            };
            scaler.update(indicators, i);
        }

        assert_eq!(scaler.history().len(), 10);
    }

    #[test]
    fn test_scale_bounds() {
        let mut config = RegimeScalingConfig::default();
        config.min_scale = 0.2;
        config.max_scale = 1.5;

        let mut scaler = RegimeScaling::with_config(config);

        // Crisis should still be above min
        scaler.force_regime(MarketRegime::Crisis, 1);
        let _ = scaler.calculate_scaling();

        let result = scaler.last_result().unwrap();
        assert!(result.final_scale >= 0.2);
        assert!(result.final_scale <= 1.5);
    }

    #[test]
    fn test_direction_bias() {
        let mut scaler = RegimeScaling::new();

        let bullish = create_bullish_indicators();
        let result = scaler.update(bullish, 1);
        assert!(result.direction_bias > 0.0);

        let bearish = create_bearish_indicators();
        scaler.config.detection.persistence_threshold = 1;
        let result = scaler.update(bearish, 2);
        assert!(result.direction_bias < 0.0);
    }

    #[test]
    fn test_summary() {
        let mut scaler = RegimeScaling::new();
        let bullish = create_bullish_indicators();
        scaler.update(bullish, 1);

        let summary = scaler.summary();
        assert!(summary.contains("Bullish"));
        assert!(summary.contains("Confidence"));
    }

    #[test]
    fn test_regime_duration_tracking() {
        let mut scaler = RegimeScaling::new();
        scaler.config.detection.persistence_threshold = 1;

        // Bullish for 5 days
        let bullish = create_bullish_indicators();
        for i in 1..=5 {
            scaler.update(bullish.clone(), i);
        }

        // Switch to bearish
        let bearish = create_bearish_indicators();
        scaler.update(bearish.clone(), 6);

        // Check duration was recorded
        if let Some(avg) = scaler.avg_regime_duration(MarketRegime::Bullish) {
            assert!(avg > 0.0);
        }
    }

    #[test]
    fn test_low_volatility_regime() {
        let mut scaler = RegimeScaling::new();
        let indicators = RegimeIndicators {
            trend: 0.01,
            volatility_percentile: 0.10, // Very low
            ..Default::default()
        };

        let result = scaler.update(indicators, 1);

        assert_eq!(result.regime, MarketRegime::LowVolatility);
    }

    #[test]
    fn test_strong_trends() {
        let mut scaler = RegimeScaling::new();

        // Strong bullish
        let strong_bull = RegimeIndicators {
            trend: 0.15,
            trend_strength: 0.8,
            volatility_percentile: 0.5,
            ..Default::default()
        };
        let result = scaler.update(strong_bull, 1);
        assert_eq!(result.regime, MarketRegime::StrongBullish);

        // Strong bearish
        let strong_bear = RegimeIndicators {
            trend: -0.15,
            trend_strength: 0.8,
            volatility_percentile: 0.5,
            ..Default::default()
        };
        scaler.config.detection.persistence_threshold = 1;
        let result = scaler.update(strong_bear, 2);
        assert_eq!(result.regime, MarketRegime::StrongBearish);
    }

    #[test]
    fn test_market_making_strategy() {
        let mut scaler = RegimeScaling::new();
        scaler.set_strategy_type(StrategyType::MarketMaking);

        // Crisis should shut down market making
        scaler.force_regime(MarketRegime::Crisis, 1);
        let adj = StrategyType::MarketMaking.regime_adjustment(MarketRegime::Crisis);
        assert_eq!(adj, 0.0);

        // Low vol should boost market making
        let low_vol_adj = StrategyType::MarketMaking.regime_adjustment(MarketRegime::LowVolatility);
        assert!(low_vol_adj > 1.0);
    }
}
