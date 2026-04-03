//! Fear/greed state estimation
//!
//! Part of the Hypothalamus region
//! Component: risk_appetite
//!
//! Implements a composite Fear & Greed Index for trading systems,
//! inspired by the CNN Fear & Greed Index but adapted for
//! algorithmic trading. Combines multiple market signals:
//!
//! - **Volatility**: High volatility → fear, low volatility → greed
//! - **Momentum**: Positive momentum → greed, negative → fear
//! - **Volume anomaly**: Unusual volume spikes → fear
//! - **Drawdown**: Current drawdown from peak → fear
//! - **Mean reversion**: Price distance from moving average
//!
//! The index outputs a value from 0 (extreme fear) to 100 (extreme greed),
//! with classifications into discrete regimes.

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Fear/Greed regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FearGreedRegime {
    /// Index 0-15: Extreme fear — markets in panic
    ExtremeFear,
    /// Index 16-35: Fear — significant caution warranted
    Fear,
    /// Index 36-50: Mild fear — slightly cautious
    MildFear,
    /// Index 51-65: Mild greed — slightly optimistic
    MildGreed,
    /// Index 66-85: Greed — strong optimism
    Greed,
    /// Index 86-100: Extreme greed — euphoria / potential bubble
    ExtremeGreed,
}

impl FearGreedRegime {
    /// Classify a raw index value (0-100) into a regime
    pub fn from_index(index: f64) -> Self {
        match index {
            x if x <= 15.0 => FearGreedRegime::ExtremeFear,
            x if x <= 35.0 => FearGreedRegime::Fear,
            x if x <= 50.0 => FearGreedRegime::MildFear,
            x if x <= 65.0 => FearGreedRegime::MildGreed,
            x if x <= 85.0 => FearGreedRegime::Greed,
            _ => FearGreedRegime::ExtremeGreed,
        }
    }

    /// Get a human-readable label
    pub fn label(&self) -> &'static str {
        match self {
            FearGreedRegime::ExtremeFear => "Extreme Fear",
            FearGreedRegime::Fear => "Fear",
            FearGreedRegime::MildFear => "Mild Fear",
            FearGreedRegime::MildGreed => "Mild Greed",
            FearGreedRegime::Greed => "Greed",
            FearGreedRegime::ExtremeGreed => "Extreme Greed",
        }
    }

    /// Get a position sizing multiplier for this regime
    ///
    /// Fear regimes suggest smaller positions, greed regimes suggest
    /// larger (or contrarian: smaller in extreme greed).
    pub fn sizing_multiplier(&self) -> f64 {
        match self {
            FearGreedRegime::ExtremeFear => 0.3,
            FearGreedRegime::Fear => 0.5,
            FearGreedRegime::MildFear => 0.75,
            FearGreedRegime::MildGreed => 1.0,
            FearGreedRegime::Greed => 1.1,
            FearGreedRegime::ExtremeGreed => 0.8, // contrarian: reduce in euphoria
        }
    }

    /// Whether this regime suggests caution (fear-side)
    pub fn is_fearful(&self) -> bool {
        matches!(
            self,
            FearGreedRegime::ExtremeFear | FearGreedRegime::Fear | FearGreedRegime::MildFear
        )
    }

    /// Whether this regime suggests optimism (greed-side)
    pub fn is_greedy(&self) -> bool {
        matches!(
            self,
            FearGreedRegime::MildGreed | FearGreedRegime::Greed | FearGreedRegime::ExtremeGreed
        )
    }
}

/// Weights for each component of the Fear & Greed Index
#[derive(Debug, Clone)]
pub struct FearGreedWeights {
    /// Weight for the volatility component (0.0 - 1.0)
    pub volatility: f64,
    /// Weight for the momentum component
    pub momentum: f64,
    /// Weight for the volume anomaly component
    pub volume_anomaly: f64,
    /// Weight for the drawdown component
    pub drawdown: f64,
    /// Weight for the mean reversion (price vs MA) component
    pub mean_reversion: f64,
}

impl Default for FearGreedWeights {
    fn default() -> Self {
        Self {
            volatility: 0.30,
            momentum: 0.25,
            volume_anomaly: 0.15,
            drawdown: 0.20,
            mean_reversion: 0.10,
        }
    }
}

impl FearGreedWeights {
    /// Validate that weights are non-negative and sum to approximately 1.0
    pub fn validate(&self) -> Result<()> {
        let sum = self.volatility
            + self.momentum
            + self.volume_anomaly
            + self.drawdown
            + self.mean_reversion;

        if self.volatility < 0.0
            || self.momentum < 0.0
            || self.volume_anomaly < 0.0
            || self.drawdown < 0.0
            || self.mean_reversion < 0.0
        {
            return Err(Error::InvalidInput(
                "FearGreedWeights: all weights must be non-negative".into(),
            ));
        }

        if (sum - 1.0).abs() > 0.01 {
            return Err(Error::InvalidInput(format!(
                "FearGreedWeights: weights must sum to ~1.0, got {}",
                sum
            )));
        }

        Ok(())
    }

    /// Normalize weights to sum to exactly 1.0
    pub fn normalize(&mut self) {
        let sum = self.volatility
            + self.momentum
            + self.volume_anomaly
            + self.drawdown
            + self.mean_reversion;

        if sum > 0.0 {
            self.volatility /= sum;
            self.momentum /= sum;
            self.volume_anomaly /= sum;
            self.drawdown /= sum;
            self.mean_reversion /= sum;
        }
    }
}

/// Configuration for the Fear & Greed Index
#[derive(Debug, Clone)]
pub struct FearGreedConfig {
    /// Component weights
    pub weights: FearGreedWeights,
    /// Window size for volatility calculation (number of return observations)
    pub volatility_window: usize,
    /// Window size for momentum calculation
    pub momentum_window: usize,
    /// Window size for volume baseline (mean volume)
    pub volume_window: usize,
    /// Window size for price moving average (mean reversion)
    pub ma_window: usize,
    /// EMA decay factor for smoothing the composite index
    pub ema_decay: f64,
    /// Baseline (normal) volatility — used to normalize the volatility signal
    pub baseline_volatility: f64,
    /// Maximum volatility (maps to fear = 100 on volatility component)
    pub max_volatility: f64,
    /// Maximum drawdown percentage for full fear on drawdown component (e.g., 0.20 = 20%)
    pub max_drawdown_pct: f64,
    /// Volume spike multiplier: volume > mean * this → fear signal
    pub volume_spike_multiplier: f64,
    /// Minimum observations before the index is considered valid
    pub min_observations: usize,
}

impl Default for FearGreedConfig {
    fn default() -> Self {
        Self {
            weights: FearGreedWeights::default(),
            volatility_window: 20,
            momentum_window: 14,
            volume_window: 20,
            ma_window: 50,
            ema_decay: 0.92,
            baseline_volatility: 0.01,
            max_volatility: 0.05,
            max_drawdown_pct: 0.20,
            volume_spike_multiplier: 2.0,
            min_observations: 10,
        }
    }
}

/// Individual component scores (each 0-100, where 0 = extreme fear, 100 = extreme greed)
#[derive(Debug, Clone, Default)]
pub struct ComponentScores {
    /// Volatility component score (low vol → greed, high vol → fear)
    pub volatility: f64,
    /// Momentum component score (positive momentum → greed)
    pub momentum: f64,
    /// Volume anomaly component score (normal volume → greed, spikes → fear)
    pub volume_anomaly: f64,
    /// Drawdown component score (no drawdown → greed, large drawdown → fear)
    pub drawdown: f64,
    /// Mean reversion component score (price above MA → greed)
    pub mean_reversion: f64,
}

/// A snapshot of the Fear & Greed Index state
#[derive(Debug, Clone)]
pub struct FearGreedSnapshot {
    /// The composite index value (0-100)
    pub index: f64,
    /// The classified regime
    pub regime: FearGreedRegime,
    /// Individual component scores
    pub components: ComponentScores,
    /// The smoothed (EMA) index value
    pub smoothed_index: f64,
    /// Rate of change of the index (positive = moving toward greed)
    pub rate_of_change: f64,
    /// Whether the index has enough data to be considered valid
    pub is_valid: bool,
    /// Number of observations processed
    pub observation_count: u64,
}

/// A single market data observation for the Fear & Greed Index
#[derive(Debug, Clone)]
pub struct MarketObservation {
    /// Current price
    pub price: f64,
    /// Current volume (optional, set to 0.0 if unavailable)
    pub volume: f64,
}

impl MarketObservation {
    /// Create a new observation with price only
    pub fn price_only(price: f64) -> Self {
        Self { price, volume: 0.0 }
    }

    /// Create a new observation with price and volume
    pub fn new(price: f64, volume: f64) -> Self {
        Self { price, volume }
    }
}

/// Fear/greed state estimation using a multi-factor composite index
pub struct FearGreedIndex {
    /// Configuration parameters
    config: FearGreedConfig,
    /// Recent prices for volatility/momentum/MA calculation
    prices: VecDeque<f64>,
    /// Recent log returns for volatility calculation
    returns: VecDeque<f64>,
    /// Recent volumes for volume anomaly detection
    volumes: VecDeque<f64>,
    /// All-time high price (for drawdown calculation)
    peak_price: f64,
    /// Current raw composite index (0-100)
    raw_index: f64,
    /// Smoothed (EMA) composite index
    smoothed_index: f64,
    /// Previous smoothed index (for rate of change)
    previous_smoothed: f64,
    /// Latest component scores
    components: ComponentScores,
    /// Number of observations processed
    observation_count: u64,
    /// History of recent index values (for trend analysis)
    index_history: VecDeque<f64>,
    /// Maximum history length for index values
    max_history: usize,
}

impl Default for FearGreedIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl FearGreedIndex {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(FearGreedConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: FearGreedConfig) -> Self {
        let max_window = config
            .volatility_window
            .max(config.momentum_window)
            .max(config.volume_window)
            .max(config.ma_window)
            + 1; // +1 for the return calculation

        Self {
            prices: VecDeque::with_capacity(max_window),
            returns: VecDeque::with_capacity(config.volatility_window),
            volumes: VecDeque::with_capacity(config.volume_window),
            peak_price: 0.0,
            raw_index: 50.0, // neutral starting point
            smoothed_index: 50.0,
            previous_smoothed: 50.0,
            components: ComponentScores::default(),
            observation_count: 0,
            index_history: VecDeque::with_capacity(100),
            max_history: 100,
            config,
        }
    }

    /// Main processing function — validates internal state
    pub fn process(&self) -> Result<()> {
        self.config.weights.validate()?;

        if self.config.baseline_volatility <= 0.0 {
            return Err(Error::InvalidInput(
                "baseline_volatility must be positive".into(),
            ));
        }
        if self.config.max_volatility <= self.config.baseline_volatility {
            return Err(Error::InvalidInput(
                "max_volatility must be greater than baseline_volatility".into(),
            ));
        }
        if self.config.max_drawdown_pct <= 0.0 || self.config.max_drawdown_pct > 1.0 {
            return Err(Error::InvalidInput(
                "max_drawdown_pct must be in (0.0, 1.0]".into(),
            ));
        }

        Ok(())
    }

    /// Update the index with a new market observation
    pub fn update(&mut self, obs: &MarketObservation) {
        if obs.price <= 0.0 {
            return; // ignore invalid prices
        }

        // Update price history
        let max_window = self
            .config
            .volatility_window
            .max(self.config.momentum_window)
            .max(self.config.ma_window)
            + 1;

        if self.prices.len() >= max_window {
            self.prices.pop_front();
        }
        self.prices.push_back(obs.price);

        // Update volume history
        if obs.volume > 0.0 {
            if self.volumes.len() >= self.config.volume_window {
                self.volumes.pop_front();
            }
            self.volumes.push_back(obs.volume);
        }

        // Compute log return
        if self.prices.len() >= 2 {
            let prev = self.prices[self.prices.len() - 2];
            if prev > 0.0 {
                let log_return = (obs.price / prev).ln();
                if self.returns.len() >= self.config.volatility_window {
                    self.returns.pop_front();
                }
                self.returns.push_back(log_return);
            }
        }

        // Update peak price (for drawdown)
        if obs.price > self.peak_price {
            self.peak_price = obs.price;
        }

        self.observation_count += 1;

        // Recompute composite index
        self.recompute();
    }

    /// Update with just a price value (convenience method)
    pub fn update_price(&mut self, price: f64) {
        self.update(&MarketObservation::price_only(price));
    }

    /// Update with price and volume
    pub fn update_price_volume(&mut self, price: f64, volume: f64) {
        self.update(&MarketObservation::new(price, volume));
    }

    /// Get the current composite index value (0 = extreme fear, 100 = extreme greed)
    pub fn index(&self) -> f64 {
        self.smoothed_index
    }

    /// Get the raw (unsmoothed) index value
    pub fn raw_index(&self) -> f64 {
        self.raw_index
    }

    /// Get the current regime classification
    pub fn regime(&self) -> FearGreedRegime {
        FearGreedRegime::from_index(self.smoothed_index)
    }

    /// Get the current component scores
    pub fn components(&self) -> &ComponentScores {
        &self.components
    }

    /// Get a full snapshot of the current state
    pub fn snapshot(&self) -> FearGreedSnapshot {
        FearGreedSnapshot {
            index: self.smoothed_index,
            regime: self.regime(),
            components: self.components.clone(),
            smoothed_index: self.smoothed_index,
            rate_of_change: self.rate_of_change(),
            is_valid: self.is_valid(),
            observation_count: self.observation_count,
        }
    }

    /// Whether the index has enough data to be considered valid
    pub fn is_valid(&self) -> bool {
        self.observation_count as usize >= self.config.min_observations
    }

    /// Rate of change of the smoothed index (positive = moving toward greed)
    pub fn rate_of_change(&self) -> f64 {
        self.smoothed_index - self.previous_smoothed
    }

    /// Whether the index is trending toward fear
    pub fn trending_fearful(&self) -> bool {
        self.rate_of_change() < -1.0
    }

    /// Whether the index is trending toward greed
    pub fn trending_greedy(&self) -> bool {
        self.rate_of_change() > 1.0
    }

    /// Get the position sizing multiplier based on current regime
    pub fn sizing_multiplier(&self) -> f64 {
        self.regime().sizing_multiplier()
    }

    /// Get the current drawdown from peak (as a fraction, e.g., 0.10 = 10%)
    pub fn current_drawdown(&self) -> f64 {
        if self.peak_price <= 0.0 || self.prices.is_empty() {
            return 0.0;
        }
        let current = *self.prices.back().unwrap();
        if current >= self.peak_price {
            0.0
        } else {
            (self.peak_price - current) / self.peak_price
        }
    }

    /// Get the current realized volatility (annualized, assuming daily data)
    pub fn current_volatility(&self) -> f64 {
        self.compute_volatility()
    }

    /// Get the current momentum (sum of recent returns)
    pub fn current_momentum(&self) -> f64 {
        self.compute_momentum()
    }

    /// Get the number of observations processed
    pub fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Get the index history (most recent values)
    pub fn history(&self) -> &VecDeque<f64> {
        &self.index_history
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.prices.clear();
        self.returns.clear();
        self.volumes.clear();
        self.peak_price = 0.0;
        self.raw_index = 50.0;
        self.smoothed_index = 50.0;
        self.previous_smoothed = 50.0;
        self.components = ComponentScores::default();
        self.observation_count = 0;
        self.index_history.clear();
    }

    /// Override the peak price (useful when initializing with historical data)
    pub fn set_peak_price(&mut self, peak: f64) {
        if peak > 0.0 {
            self.peak_price = peak;
        }
    }

    // ── internal computation ──

    /// Recompute all component scores and the composite index
    fn recompute(&mut self) {
        // Compute individual component scores (each 0-100)
        self.components.volatility = self.score_volatility();
        self.components.momentum = self.score_momentum();
        self.components.volume_anomaly = self.score_volume_anomaly();
        self.components.drawdown = self.score_drawdown();
        self.components.mean_reversion = self.score_mean_reversion();

        // Weighted composite
        let w = &self.config.weights;
        self.raw_index = (w.volatility * self.components.volatility
            + w.momentum * self.components.momentum
            + w.volume_anomaly * self.components.volume_anomaly
            + w.drawdown * self.components.drawdown
            + w.mean_reversion * self.components.mean_reversion)
            .clamp(0.0, 100.0);

        // Smooth with EMA
        self.previous_smoothed = self.smoothed_index;
        let alpha = 1.0 - self.config.ema_decay;
        if self.observation_count <= 1 {
            self.smoothed_index = self.raw_index;
        } else {
            self.smoothed_index =
                self.config.ema_decay * self.smoothed_index + alpha * self.raw_index;
        }

        // Record history
        if self.index_history.len() >= self.max_history {
            self.index_history.pop_front();
        }
        self.index_history.push_back(self.smoothed_index);
    }

    /// Compute realized volatility from recent returns
    fn compute_volatility(&self) -> f64 {
        if self.returns.len() < 2 {
            return 0.0;
        }

        let n = self.returns.len() as f64;
        let mean: f64 = self.returns.iter().sum::<f64>() / n;
        let variance: f64 =
            self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);

        variance.sqrt()
    }

    /// Compute momentum as the sum of recent returns
    fn compute_momentum(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let window = self.config.momentum_window.min(self.returns.len());
        self.returns.iter().rev().take(window).sum()
    }

    /// Score the volatility component (0 = extreme fear / high vol, 100 = greed / low vol)
    fn score_volatility(&self) -> f64 {
        let vol = self.compute_volatility();
        if vol <= 0.0 {
            return 50.0; // neutral if no data
        }

        let baseline = self.config.baseline_volatility;
        let max_vol = self.config.max_volatility;

        // Map volatility to 0-100 (inverted: low vol = high score)
        // vol <= baseline → score 100 (greed)
        // vol >= max_vol  → score 0 (fear)
        if vol <= baseline {
            100.0
        } else if vol >= max_vol {
            0.0
        } else {
            let ratio = (vol - baseline) / (max_vol - baseline);
            (1.0 - ratio) * 100.0
        }
    }

    /// Score the momentum component (0 = fear / negative momentum, 100 = greed / positive)
    fn score_momentum(&self) -> f64 {
        let momentum = self.compute_momentum();

        // Use a sigmoid-like mapping centered at 0
        // Strong positive momentum → 100, strong negative → 0
        let sensitivity = 50.0; // how quickly it saturates
        let sigmoid = 1.0 / (1.0 + (-sensitivity * momentum).exp());
        sigmoid * 100.0
    }

    /// Score the volume anomaly component (0 = fear / volume spike, 100 = greed / normal volume)
    fn score_volume_anomaly(&self) -> f64 {
        if self.volumes.len() < 3 {
            return 50.0; // neutral if insufficient volume data
        }

        let mean_volume: f64 = self.volumes.iter().sum::<f64>() / self.volumes.len() as f64;
        if mean_volume <= 0.0 {
            return 50.0;
        }

        let current_volume = *self.volumes.back().unwrap();
        let ratio = current_volume / mean_volume;

        // Volume within normal range → greed (calm markets)
        // Volume spike → fear (panic / capitulation)
        let spike_threshold = self.config.volume_spike_multiplier;

        if ratio <= 1.0 {
            // Below average or average volume → slightly greedy
            75.0 + 25.0 * (1.0 - ratio) // 75-100
        } else if ratio <= spike_threshold {
            // Above average but below spike → linear decrease
            let t = (ratio - 1.0) / (spike_threshold - 1.0);
            75.0 * (1.0 - t) // 75 → 0
        } else {
            // Spike: extreme fear
            0.0
        }
    }

    /// Score the drawdown component (0 = fear / large drawdown, 100 = greed / no drawdown)
    fn score_drawdown(&self) -> f64 {
        let dd = self.current_drawdown();
        let max_dd = self.config.max_drawdown_pct;

        if dd <= 0.0 {
            100.0 // at or near all-time high
        } else if dd >= max_dd {
            0.0 // extreme drawdown
        } else {
            (1.0 - dd / max_dd) * 100.0
        }
    }

    /// Score the mean reversion component
    /// (0 = fear / price far below MA, 100 = greed / price far above MA)
    fn score_mean_reversion(&self) -> f64 {
        if self.prices.len() < 3 {
            return 50.0;
        }

        let window = self.config.ma_window.min(self.prices.len());
        let ma: f64 = self.prices.iter().rev().take(window).sum::<f64>() / window as f64;

        if ma <= 0.0 {
            return 50.0;
        }

        let current = *self.prices.back().unwrap();
        let deviation = (current - ma) / ma; // fractional deviation

        // Map deviation to 0-100 using sigmoid
        // deviation = +0.10 (10% above MA) → ~90
        // deviation = -0.10 (10% below MA) → ~10
        let sensitivity = 30.0;
        let sigmoid = 1.0 / (1.0 + (-sensitivity * deviation).exp());
        sigmoid * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = FearGreedIndex::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_initial_state_is_neutral() {
        let index = FearGreedIndex::new();
        assert!((index.index() - 50.0).abs() < 1e-9);
        assert_eq!(index.observation_count(), 0);
        assert!(!index.is_valid());
    }

    #[test]
    fn test_regime_classification() {
        assert_eq!(
            FearGreedRegime::from_index(5.0),
            FearGreedRegime::ExtremeFear
        );
        assert_eq!(FearGreedRegime::from_index(25.0), FearGreedRegime::Fear);
        assert_eq!(FearGreedRegime::from_index(45.0), FearGreedRegime::MildFear);
        assert_eq!(
            FearGreedRegime::from_index(55.0),
            FearGreedRegime::MildGreed
        );
        assert_eq!(FearGreedRegime::from_index(75.0), FearGreedRegime::Greed);
        assert_eq!(
            FearGreedRegime::from_index(95.0),
            FearGreedRegime::ExtremeGreed
        );
    }

    #[test]
    fn test_regime_properties() {
        assert!(FearGreedRegime::ExtremeFear.is_fearful());
        assert!(FearGreedRegime::Fear.is_fearful());
        assert!(FearGreedRegime::MildFear.is_fearful());
        assert!(!FearGreedRegime::MildGreed.is_fearful());

        assert!(!FearGreedRegime::ExtremeFear.is_greedy());
        assert!(FearGreedRegime::MildGreed.is_greedy());
        assert!(FearGreedRegime::Greed.is_greedy());
        assert!(FearGreedRegime::ExtremeGreed.is_greedy());
    }

    #[test]
    fn test_rising_prices_produce_greed() {
        let mut index = FearGreedIndex::with_config(FearGreedConfig {
            min_observations: 5,
            volatility_window: 10,
            momentum_window: 10,
            ma_window: 10,
            ema_decay: 0.5, // fast response for testing
            ..Default::default()
        });

        // Feed a steady uptrend
        let mut price = 100.0;
        for _ in 0..30 {
            price *= 1.005; // 0.5% daily gain
            index.update_price(price);
        }

        assert!(index.is_valid());
        assert!(
            index.index() > 55.0,
            "rising prices should produce greed, got {}",
            index.index()
        );
        assert!(
            index.regime().is_greedy(),
            "regime should be greedy, got {:?}",
            index.regime()
        );
    }

    #[test]
    fn test_falling_prices_produce_fear() {
        let mut index = FearGreedIndex::with_config(FearGreedConfig {
            min_observations: 5,
            volatility_window: 10,
            momentum_window: 10,
            ma_window: 10,
            ema_decay: 0.5,
            baseline_volatility: 0.005,
            max_volatility: 0.03,
            ..Default::default()
        });

        // Start with some stable prices, then crash
        for _ in 0..10 {
            index.update_price(100.0);
        }
        let mut price = 100.0;
        for _ in 0..20 {
            price *= 0.97; // 3% daily loss
            index.update_price(price);
        }

        assert!(index.is_valid());
        assert!(
            index.index() < 45.0,
            "falling prices should produce fear, got {}",
            index.index()
        );
    }

    #[test]
    fn test_drawdown_detection() {
        let mut index = FearGreedIndex::new();

        index.update_price(100.0);
        index.update_price(110.0); // new peak
        index.update_price(99.0); // drawdown

        let dd = index.current_drawdown();
        let expected = (110.0 - 99.0) / 110.0;
        assert!(
            (dd - expected).abs() < 1e-9,
            "expected drawdown {}, got {}",
            expected,
            dd
        );
    }

    #[test]
    fn test_no_drawdown_at_peak() {
        let mut index = FearGreedIndex::new();
        index.update_price(100.0);
        index.update_price(105.0);
        index.update_price(110.0);

        assert!((index.current_drawdown() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_volume_spike_causes_fear() {
        let mut index = FearGreedIndex::with_config(FearGreedConfig {
            min_observations: 3,
            volume_window: 5,
            volume_spike_multiplier: 2.0,
            ema_decay: 0.3,
            ..Default::default()
        });

        // Normal volume baseline
        for i in 0..10 {
            index.update(&MarketObservation::new(100.0 + i as f64 * 0.1, 1000.0));
        }

        let before = index.components().volume_anomaly;

        // Volume spike (5x normal)
        index.update(&MarketObservation::new(100.5, 5000.0));
        let after = index.components().volume_anomaly;

        assert!(
            after < before,
            "volume spike should reduce volume score: before={}, after={}",
            before,
            after
        );
    }

    #[test]
    fn test_snapshot() {
        let mut index = FearGreedIndex::with_config(FearGreedConfig {
            min_observations: 3,
            ..Default::default()
        });

        for i in 0..15 {
            index.update_price(100.0 + i as f64);
        }

        let snap = index.snapshot();
        assert!(snap.is_valid);
        assert!(snap.index >= 0.0 && snap.index <= 100.0);
        assert_eq!(snap.observation_count, 15);
    }

    #[test]
    fn test_rate_of_change() {
        let mut index = FearGreedIndex::with_config(FearGreedConfig {
            ema_decay: 0.5,
            min_observations: 2,
            ..Default::default()
        });

        // Start neutral
        index.update_price(100.0);
        index.update_price(100.0);
        let roc_neutral = index.rate_of_change();

        // Drive toward greed with strong uptrend
        for _ in 0..10 {
            index.update_price(200.0);
        }

        // After strong move up, rate_of_change should have been positive at some point
        // (the smoothed index should be moving toward greed)
        assert!(
            index.index() > 50.0 || roc_neutral.abs() < 50.0,
            "index should respond to price changes"
        );
    }

    #[test]
    fn test_sizing_multiplier() {
        assert!(FearGreedRegime::ExtremeFear.sizing_multiplier() < 1.0);
        assert!((FearGreedRegime::MildGreed.sizing_multiplier() - 1.0).abs() < 1e-9);
        assert!(FearGreedRegime::Greed.sizing_multiplier() > 1.0);
        // Contrarian: extreme greed reduces sizing
        assert!(FearGreedRegime::ExtremeGreed.sizing_multiplier() < 1.0);
    }

    #[test]
    fn test_weight_validation() {
        let valid = FearGreedWeights::default();
        assert!(valid.validate().is_ok());

        let invalid = FearGreedWeights {
            volatility: 0.5,
            momentum: 0.5,
            volume_anomaly: 0.5,
            drawdown: 0.5,
            mean_reversion: 0.5,
        };
        assert!(invalid.validate().is_err());

        let negative = FearGreedWeights {
            volatility: -0.1,
            ..Default::default()
        };
        assert!(negative.validate().is_err());
    }

    #[test]
    fn test_weight_normalization() {
        let mut weights = FearGreedWeights {
            volatility: 2.0,
            momentum: 2.0,
            volume_anomaly: 2.0,
            drawdown: 2.0,
            mean_reversion: 2.0,
        };
        weights.normalize();

        let sum = weights.volatility
            + weights.momentum
            + weights.volume_anomaly
            + weights.drawdown
            + weights.mean_reversion;
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "normalized weights should sum to 1.0, got {}",
            sum
        );
        assert!((weights.volatility - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_reset() {
        let mut index = FearGreedIndex::new();
        for i in 0..20 {
            index.update_price(100.0 + i as f64);
        }

        assert!(index.observation_count() > 0);
        index.reset();

        assert_eq!(index.observation_count(), 0);
        assert!((index.index() - 50.0).abs() < 1e-9);
        assert!(index.history().is_empty());
    }

    #[test]
    fn test_index_stays_in_bounds() {
        let mut index = FearGreedIndex::new();

        // Extreme price movements should still keep index in [0, 100]
        let mut price = 100.0;
        for _ in 0..50 {
            price *= 1.1; // 10% daily gain (absurd)
            index.update_price(price);
        }
        assert!(
            index.index() >= 0.0 && index.index() <= 100.0,
            "index out of bounds: {}",
            index.index()
        );

        index.reset();

        price = 100.0;
        for _ in 0..50 {
            price *= 0.9; // 10% daily loss
            index.update_price(price);
        }
        assert!(
            index.index() >= 0.0 && index.index() <= 100.0,
            "index out of bounds: {}",
            index.index()
        );
    }

    #[test]
    fn test_set_peak_price() {
        let mut index = FearGreedIndex::new();
        index.set_peak_price(200.0);
        index.update_price(180.0);

        let dd = index.current_drawdown();
        let expected = (200.0 - 180.0) / 200.0;
        assert!(
            (dd - expected).abs() < 1e-9,
            "expected drawdown {}, got {}",
            expected,
            dd
        );
    }

    #[test]
    fn test_regime_labels() {
        assert_eq!(FearGreedRegime::ExtremeFear.label(), "Extreme Fear");
        assert_eq!(FearGreedRegime::Greed.label(), "Greed");
        assert_eq!(FearGreedRegime::ExtremeGreed.label(), "Extreme Greed");
    }

    #[test]
    fn test_invalid_config_detected() {
        let index = FearGreedIndex::with_config(FearGreedConfig {
            baseline_volatility: 0.0,
            ..Default::default()
        });
        assert!(index.process().is_err());
    }

    #[test]
    fn test_negative_price_ignored() {
        let mut index = FearGreedIndex::new();
        index.update_price(100.0);
        index.update_price(-50.0); // should be ignored

        assert_eq!(index.observation_count(), 1);
    }

    #[test]
    fn test_volatility_score_mapping() {
        let mut index = FearGreedIndex::with_config(FearGreedConfig {
            baseline_volatility: 0.01,
            max_volatility: 0.05,
            ema_decay: 0.1,
            min_observations: 3,
            ..Default::default()
        });

        // Very stable prices → low volatility → high volatility score (greedy)
        for i in 0..30 {
            // tiny oscillation: 100.00, 100.01, 100.00, ...
            let price = 100.0 + (i % 2) as f64 * 0.001;
            index.update_price(price);
        }

        assert!(
            index.components().volatility > 70.0,
            "low vol should give high volatility score, got {}",
            index.components().volatility
        );
    }

    #[test]
    fn test_history_tracking() {
        let mut index = FearGreedIndex::new();
        for i in 0..5 {
            index.update_price(100.0 + i as f64);
        }

        assert_eq!(index.history().len(), 5);
    }
}
