//! # Technical Indicator Analysis Module
//!
//! This module provides comprehensive technical indicator analysis for signal generation.
//! It integrates with the existing `janus-indicators` crate and provides high-level
//! analysis functions for common trading patterns.
//!
//! ## Supported Indicators
//!
//! - **EMA (Exponential Moving Average)**: Trend following
//! - **RSI (Relative Strength Index)**: Momentum oscillator
//! - **MACD (Moving Average Convergence Divergence)**: Trend and momentum
//! - **Bollinger Bands**: Volatility and price levels
//! - **ATR (Average True Range)**: Volatility measurement
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::indicators::{IndicatorAnalyzer, IndicatorConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let analyzer = IndicatorAnalyzer::new(IndicatorConfig::default());
//!
//! // Analyze price data
//! let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0];
//! let analysis = analyzer.analyze_batch(&prices).await?;
//!
//! println!("EMA Fast: {:?}", analysis.ema_fast);
//! println!("RSI: {:?}", analysis.rsi);
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use janus_indicators::{ATR, EMA, ema, macd, rsi, sma};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tracing::{debug, warn};

/// Configuration for indicator analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorConfig {
    /// Fast EMA period (default: 8)
    pub ema_fast_period: usize,

    /// Slow EMA period (default: 21)
    pub ema_slow_period: usize,

    /// RSI period (default: 14)
    pub rsi_period: usize,

    /// MACD fast period (default: 12)
    pub macd_fast_period: usize,

    /// MACD slow period (default: 26)
    pub macd_slow_period: usize,

    /// MACD signal period (default: 9)
    pub macd_signal_period: usize,

    /// ATR period (default: 14)
    pub atr_period: usize,

    /// Bollinger Bands period (default: 20)
    pub bb_period: usize,

    /// Bollinger Bands standard deviation (default: 2.0)
    pub bb_std_dev: f64,
}

impl Default for IndicatorConfig {
    fn default() -> Self {
        Self {
            ema_fast_period: 8,
            ema_slow_period: 21,
            rsi_period: 14,
            macd_fast_period: 12,
            macd_slow_period: 26,
            macd_signal_period: 9,
            atr_period: 14,
            bb_period: 20,
            bb_std_dev: 2.0_f64,
        }
    }
}

/// Results from indicator analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorAnalysis {
    /// Fast EMA value
    pub ema_fast: Option<f64>,

    /// Slow EMA value
    pub ema_slow: Option<f64>,

    /// EMA crossover signal (1.0 = bullish cross, -1.0 = bearish cross, 0.0 = no cross)
    pub ema_cross: f64,

    /// RSI value (0-100)
    pub rsi: Option<f64>,

    /// RSI signal (-1.0 to 1.0, negative = oversold/bullish, positive = overbought/bearish)
    pub rsi_signal: f64,

    /// MACD line value
    pub macd_line: Option<f64>,

    /// MACD signal line value
    pub macd_signal: Option<f64>,

    /// MACD histogram value
    pub macd_histogram: Option<f64>,

    /// MACD crossover signal
    pub macd_cross: f64,

    /// ATR value (volatility)
    pub atr: Option<f64>,

    /// Bollinger Bands upper band
    pub bb_upper: Option<f64>,

    /// Bollinger Bands middle band (SMA)
    pub bb_middle: Option<f64>,

    /// Bollinger Bands lower band
    pub bb_lower: Option<f64>,

    /// Bollinger Bands position (-1.0 = at lower, 0.0 = at middle, 1.0 = at upper)
    pub bb_position: f64,

    /// Overall trend strength (-1.0 to 1.0)
    pub trend_strength: f64,

    /// Volatility level (0.0 to 1.0)
    pub volatility: f64,
}

impl Default for IndicatorAnalysis {
    fn default() -> Self {
        Self {
            ema_fast: None,
            ema_slow: None,
            ema_cross: 0.0,
            rsi: None,
            rsi_signal: 0.0,
            macd_line: None,
            macd_signal: None,
            macd_histogram: None,
            macd_cross: 0.0,
            atr: None,
            bb_upper: None,
            bb_middle: None,
            bb_lower: None,
            bb_position: 0.0,
            trend_strength: 0.0,
            volatility: 0.0,
        }
    }
}

/// Indicator analyzer with state tracking
pub struct IndicatorAnalyzer {
    config: IndicatorConfig,

    // State for incremental updates
    ema_fast: EMA,
    ema_slow: EMA,
    atr_calculator: ATR,

    // Price history for calculations
    price_history: VecDeque<f64>,
    high_history: VecDeque<f64>,
    low_history: VecDeque<f64>,

    // Previous values for crossover detection
    prev_ema_fast: Option<f64>,
    prev_ema_slow: Option<f64>,
    prev_macd_line: Option<f64>,
    prev_macd_signal: Option<f64>,

    // Analysis cache
    last_analysis: Option<IndicatorAnalysis>,
}

impl IndicatorAnalyzer {
    /// Create a new indicator analyzer
    pub fn new(config: IndicatorConfig) -> Self {
        Self {
            ema_fast: EMA::new(config.ema_fast_period),
            ema_slow: EMA::new(config.ema_slow_period),
            atr_calculator: ATR::new(config.atr_period),
            config,
            price_history: VecDeque::with_capacity(100),
            high_history: VecDeque::with_capacity(100),
            low_history: VecDeque::with_capacity(100),
            prev_ema_fast: None,
            prev_ema_slow: None,
            prev_macd_line: None,
            prev_macd_signal: None,
            last_analysis: None,
        }
    }

    /// Update with new price data (close only)
    pub fn update(&mut self, close: f64) -> Result<()> {
        self.update_hlc(close, close, close)
    }

    /// Update with new OHLC data
    pub fn update_hlc(&mut self, high: f64, low: f64, close: f64) -> Result<()> {
        // Update EMAs
        self.ema_fast.update(close);
        self.ema_slow.update(close);

        // Update ATR
        self.atr_calculator.update(high, low, close);

        // Update price history
        self.price_history.push_back(close);
        self.high_history.push_back(high);
        self.low_history.push_back(low);

        // Keep history bounded
        let max_history = self.config.macd_slow_period.max(self.config.bb_period) * 3;
        if self.price_history.len() > max_history {
            self.price_history.pop_front();
            self.high_history.pop_front();
            self.low_history.pop_front();
        }

        Ok(())
    }

    /// Perform full analysis on current state
    pub async fn analyze(&mut self) -> Result<IndicatorAnalysis> {
        let mut analysis = IndicatorAnalysis::default();

        // EMA Analysis
        if self.ema_fast.is_ready() && self.ema_slow.is_ready() {
            let fast = self.ema_fast.value();
            let slow = self.ema_slow.value();

            analysis.ema_fast = Some(fast);
            analysis.ema_slow = Some(slow);

            // Detect crossover
            if let (Some(prev_fast), Some(prev_slow)) = (self.prev_ema_fast, self.prev_ema_slow) {
                if prev_fast <= prev_slow && fast > slow {
                    analysis.ema_cross = 1.0; // Bullish crossover
                    debug!(
                        "EMA bullish crossover detected: fast={:.2}, slow={:.2}",
                        fast, slow
                    );
                } else if prev_fast >= prev_slow && fast < slow {
                    analysis.ema_cross = -1.0; // Bearish crossover
                    debug!(
                        "EMA bearish crossover detected: fast={:.2}, slow={:.2}",
                        fast, slow
                    );
                }
            }

            self.prev_ema_fast = Some(fast);
            self.prev_ema_slow = Some(slow);
        }

        // RSI Analysis
        if self.price_history.len() >= self.config.rsi_period {
            let prices: Vec<f64> = self.price_history.iter().copied().collect();
            match rsi(&prices, self.config.rsi_period) {
                Ok(rsi_values) => {
                    if let Some(&rsi_value) = rsi_values.last() {
                        analysis.rsi = Some(rsi_value);

                        // Generate RSI signal
                        if rsi_value < 30.0 {
                            analysis.rsi_signal = -0.8; // Oversold (bullish)
                        } else if rsi_value < 40.0 {
                            analysis.rsi_signal = -0.4; // Moderately oversold
                        } else if rsi_value > 70.0 {
                            analysis.rsi_signal = 0.8; // Overbought (bearish)
                        } else if rsi_value > 60.0 {
                            analysis.rsi_signal = 0.4; // Moderately overbought
                        }
                    }
                }
                Err(e) => {
                    warn!("RSI calculation failed: {}", e);
                }
            }
        }

        // MACD Analysis
        if self.price_history.len() >= self.config.macd_slow_period + self.config.macd_signal_period
        {
            let prices: Vec<f64> = self.price_history.iter().copied().collect();
            match macd(
                &prices,
                self.config.macd_fast_period,
                self.config.macd_slow_period,
                self.config.macd_signal_period,
            ) {
                Ok((macd_line_vec, signal_line_vec, histogram_vec)) => {
                    let macd_line = *macd_line_vec.last().unwrap_or(&0.0);
                    let signal_line = *signal_line_vec.last().unwrap_or(&0.0);
                    let histogram = *histogram_vec.last().unwrap_or(&0.0);
                    analysis.macd_line = Some(macd_line);
                    analysis.macd_signal = Some(signal_line);
                    analysis.macd_histogram = Some(histogram);

                    // Detect MACD crossover
                    if let (Some(prev_line), Some(prev_signal)) =
                        (self.prev_macd_line, self.prev_macd_signal)
                    {
                        if prev_line <= prev_signal && macd_line > signal_line {
                            analysis.macd_cross = 1.0; // Bullish crossover
                            debug!("MACD bullish crossover detected");
                        } else if prev_line >= prev_signal && macd_line < signal_line {
                            analysis.macd_cross = -1.0; // Bearish crossover
                            debug!("MACD bearish crossover detected");
                        }
                    }

                    self.prev_macd_line = Some(macd_line);
                    self.prev_macd_signal = Some(signal_line);
                }
                Err(e) => {
                    warn!("MACD calculation failed: {}", e);
                }
            }
        }

        // ATR Analysis
        if self.atr_calculator.is_ready() {
            analysis.atr = Some(self.atr_calculator.value());

            // Calculate normalized volatility (0.0 to 1.0)
            if let Some(current_price) = self.price_history.back() {
                let atr_pct = self.atr_calculator.value() / current_price;
                analysis.volatility = (atr_pct * 100.0).min(1.0); // Cap at 1.0
            }
        }

        // Bollinger Bands Analysis
        if self.price_history.len() >= self.config.bb_period {
            let prices: Vec<f64> = self.price_history.iter().copied().collect();
            match self.calculate_bollinger_bands(&prices) {
                Ok((upper, middle, lower)) => {
                    analysis.bb_upper = Some(upper);
                    analysis.bb_middle = Some(middle);
                    analysis.bb_lower = Some(lower);

                    // Calculate position within bands
                    if let Some(current_price) = self.price_history.back() {
                        let band_width = upper - lower;
                        if band_width > 0.0 {
                            let position = (current_price - lower) / band_width;
                            analysis.bb_position = (position * 2.0 - 1.0).clamp(-1.0, 1.0);
                        }
                    }
                }
                Err(e) => {
                    warn!("Bollinger Bands calculation failed: {}", e);
                }
            }
        }

        // Calculate overall trend strength
        analysis.trend_strength = self.calculate_trend_strength(&analysis);

        // Cache the analysis
        self.last_analysis = Some(analysis.clone());

        Ok(analysis)
    }

    /// Analyze a batch of prices (for backtesting or bulk analysis)
    pub async fn analyze_batch(&self, prices: &[f64]) -> Result<IndicatorAnalysis> {
        let mut analysis = IndicatorAnalysis::default();

        if prices.len() < self.config.ema_slow_period {
            return Ok(analysis);
        }

        // Calculate EMAs
        match ema(prices, self.config.ema_fast_period) {
            Ok(fast_values) => {
                if let Some(&fast) = fast_values.last() {
                    analysis.ema_fast = Some(fast);
                }
            }
            Err(e) => warn!("Fast EMA calculation failed: {}", e),
        }

        match ema(prices, self.config.ema_slow_period) {
            Ok(slow_values) => {
                if let Some(&slow) = slow_values.last() {
                    analysis.ema_slow = Some(slow);
                }
            }
            Err(e) => warn!("Slow EMA calculation failed: {}", e),
        }

        // Calculate RSI
        if prices.len() >= self.config.rsi_period {
            match rsi(prices, self.config.rsi_period) {
                Ok(rsi_values) => {
                    if let Some(&rsi_value) = rsi_values.last() {
                        analysis.rsi = Some(rsi_value);
                        if rsi_value < 30.0 {
                            analysis.rsi_signal = -0.8;
                        } else if rsi_value > 70.0 {
                            analysis.rsi_signal = 0.8;
                        }
                    }
                }
                Err(e) => warn!("RSI calculation failed: {}", e),
            }
        }

        // Calculate MACD
        if prices.len() >= self.config.macd_slow_period + self.config.macd_signal_period {
            match macd(
                prices,
                self.config.macd_fast_period,
                self.config.macd_slow_period,
                self.config.macd_signal_period,
            ) {
                Ok((macd_line_vec, signal_line_vec, histogram_vec)) => {
                    let macd_line = *macd_line_vec.last().unwrap_or(&0.0);
                    let signal_line = *signal_line_vec.last().unwrap_or(&0.0);
                    let histogram = *histogram_vec.last().unwrap_or(&0.0);
                    analysis.macd_line = Some(macd_line);
                    analysis.macd_signal = Some(signal_line);
                    analysis.macd_histogram = Some(histogram);

                    if histogram > 0.0 {
                        analysis.macd_cross = 0.5;
                    } else if histogram < 0.0 {
                        analysis.macd_cross = -0.5;
                    }
                }
                Err(e) => warn!("MACD calculation failed: {}", e),
            }
        }

        // Calculate Bollinger Bands
        if prices.len() >= self.config.bb_period {
            match self.calculate_bollinger_bands(prices) {
                Ok((upper, middle, lower)) => {
                    analysis.bb_upper = Some(upper);
                    analysis.bb_middle = Some(middle);
                    analysis.bb_lower = Some(lower);

                    let current_price = prices[prices.len() - 1];
                    let band_width = upper - lower;
                    if band_width > 0.0 {
                        let position = (current_price - lower) / band_width;
                        analysis.bb_position = (position * 2.0 - 1.0).clamp(-1.0, 1.0);
                    }
                }
                Err(e) => warn!("Bollinger Bands calculation failed: {}", e),
            }
        }

        analysis.trend_strength = self.calculate_trend_strength(&analysis);

        Ok(analysis)
    }

    /// Get the last analysis result (cached)
    pub fn last_analysis(&self) -> Option<&IndicatorAnalysis> {
        self.last_analysis.as_ref()
    }

    /// Reset all indicator state
    pub fn reset(&mut self) {
        self.ema_fast = EMA::new(self.config.ema_fast_period);
        self.ema_slow = EMA::new(self.config.ema_slow_period);
        self.atr_calculator = ATR::new(self.config.atr_period);
        self.price_history.clear();
        self.high_history.clear();
        self.low_history.clear();
        self.prev_ema_fast = None;
        self.prev_ema_slow = None;
        self.prev_macd_line = None;
        self.prev_macd_signal = None;
        self.last_analysis = None;
    }

    // Private helper methods

    fn calculate_bollinger_bands(&self, prices: &[f64]) -> Result<(f64, f64, f64)> {
        if prices.len() < self.config.bb_period {
            anyhow::bail!("Insufficient data for Bollinger Bands");
        }

        // Calculate middle band (SMA)
        let sma_values = sma(prices, self.config.bb_period)
            .context("Failed to calculate SMA for Bollinger Bands")?;

        let middle = *sma_values
            .last()
            .context("SMA calculation returned empty result")?;

        // Calculate standard deviation
        let prices_slice = &prices[prices.len() - self.config.bb_period..];
        let variance: f64 = prices_slice
            .iter()
            .map(|&p| {
                let diff = p - middle;
                diff * diff
            })
            .sum::<f64>()
            / self.config.bb_period as f64;

        let std_dev = variance.sqrt();

        // Calculate upper and lower bands
        let upper = middle + (self.config.bb_std_dev * std_dev);
        let lower = middle - (self.config.bb_std_dev * std_dev);

        Ok((upper, middle, lower))
    }

    fn calculate_trend_strength(&self, analysis: &IndicatorAnalysis) -> f64 {
        let mut strength: f64 = 0.0;
        let mut count = 0;

        // EMA trend
        if let (Some(fast), Some(slow)) = (analysis.ema_fast, analysis.ema_slow) {
            let diff = (fast - slow) / slow;
            strength += diff.clamp(-1.0, 1.0);
            count += 1;
        }

        // MACD trend
        if let Some(histogram) = analysis.macd_histogram {
            strength += histogram.signum() * 0.5;
            count += 1;
        }

        // RSI trend (inverted - oversold is bullish)
        strength += -analysis.rsi_signal;
        count += 1;

        if count > 0 {
            strength / count as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indicator_config_default() {
        let config = IndicatorConfig::default();
        assert_eq!(config.ema_fast_period, 8);
        assert_eq!(config.ema_slow_period, 21);
        assert_eq!(config.rsi_period, 14);
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = IndicatorAnalyzer::new(IndicatorConfig::default());
        assert!(analyzer.last_analysis().is_none());
    }

    #[tokio::test]
    async fn test_analyzer_update() {
        let mut analyzer = IndicatorAnalyzer::new(IndicatorConfig::default());

        // Update with some prices
        for price in 100..110 {
            analyzer.update(price as f64).unwrap();
        }

        let analysis = analyzer.analyze().await.unwrap();
        assert!(analysis.ema_fast.is_some() || analysis.ema_slow.is_none());
    }

    #[tokio::test]
    async fn test_batch_analysis() {
        let analyzer = IndicatorAnalyzer::new(IndicatorConfig::default());

        // Generate sample price data (uptrend)
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.5).collect();

        let analysis = analyzer.analyze_batch(&prices).await.unwrap();

        // In uptrend, fast EMA should be above slow EMA
        if let (Some(fast), Some(slow)) = (analysis.ema_fast, analysis.ema_slow) {
            assert!(fast > slow, "Fast EMA should be above slow EMA in uptrend");
        }
    }

    #[tokio::test]
    async fn test_rsi_signals() {
        let analyzer = IndicatorAnalyzer::new(IndicatorConfig::default());

        // Oversold scenario (low prices)
        let mut prices = vec![100.0; 20];
        for i in 0..10 {
            prices.push(100.0 - i as f64 * 2.0);
        }

        let analysis = analyzer.analyze_batch(&prices).await.unwrap();

        if let Some(rsi) = analysis.rsi {
            assert!(rsi < 50.0, "RSI should be low in downtrend");
        }
    }

    #[test]
    fn test_analyzer_reset() {
        let mut analyzer = IndicatorAnalyzer::new(IndicatorConfig::default());

        for price in 100..110 {
            analyzer.update(price as f64).unwrap();
        }

        analyzer.reset();
        assert!(analyzer.last_analysis().is_none());
        assert_eq!(analyzer.price_history.len(), 0);
    }
}
