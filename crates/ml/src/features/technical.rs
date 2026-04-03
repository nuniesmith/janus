//! Technical analysis indicators for feature extraction
//!
//! This module implements common technical indicators used in trading:
//! - SMA (Simple Moving Average)
//! - EMA (Exponential Moving Average)
//! - RSI (Relative Strength Index)
//! - MACD (Moving Average Convergence Divergence)
//! - Bollinger Bands
//! - ATR (Average True Range)
//! - Momentum indicators
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::features::technical::TechnicalIndicators;
//!
//! let indicators = TechnicalIndicators::new()
//!     .with_sma_periods(&[5, 10, 20])
//!     .with_rsi_period(14)
//!     .with_macd(12, 26, 9);
//!
//! let features = indicators.extract_single(&market_event)?;
//! ```

use super::{FeatureExtractor, calculate_ema, calculate_sma, calculate_std};
use crate::error::{MLError, Result};
use janus_core::MarketDataEvent;
use rust_decimal::prelude::ToPrimitive;
use std::collections::VecDeque;

/// Configuration for technical indicators
#[derive(Debug, Clone)]
pub struct TechnicalIndicators {
    /// SMA periods to calculate
    pub sma_periods: Vec<usize>,

    /// EMA periods to calculate
    pub ema_periods: Vec<usize>,

    /// RSI period
    pub rsi_period: Option<usize>,

    /// MACD configuration (fast, slow, signal)
    pub macd_config: Option<(usize, usize, usize)>,

    /// Bollinger Bands configuration (period, std_dev)
    pub bollinger_bands: Option<(usize, f64)>,

    /// ATR period
    pub atr_period: Option<usize>,

    /// Include momentum indicators
    pub include_momentum: bool,

    /// Price history buffer
    price_history: VecDeque<f64>,

    /// High price history
    high_history: VecDeque<f64>,

    /// Low price history
    low_history: VecDeque<f64>,

    /// Volume history
    volume_history: VecDeque<f64>,

    /// Maximum history length needed
    max_history: usize,
}

impl Default for TechnicalIndicators {
    fn default() -> Self {
        Self::new()
    }
}

impl TechnicalIndicators {
    /// Create a new technical indicators extractor with default settings
    pub fn new() -> Self {
        Self {
            sma_periods: vec![5, 10, 20, 50],
            ema_periods: vec![12, 26],
            rsi_period: Some(14),
            macd_config: Some((12, 26, 9)),
            bollinger_bands: Some((20, 2.0)),
            atr_period: Some(14),
            include_momentum: true,
            price_history: VecDeque::new(),
            high_history: VecDeque::new(),
            low_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            max_history: 200,
        }
    }

    /// Set SMA periods
    pub fn with_sma_periods(mut self, periods: &[usize]) -> Self {
        self.sma_periods = periods.to_vec();
        self.update_max_history();
        self
    }

    /// Set EMA periods
    pub fn with_ema_periods(mut self, periods: &[usize]) -> Self {
        self.ema_periods = periods.to_vec();
        self.update_max_history();
        self
    }

    /// Set RSI period
    pub fn with_rsi_period(mut self, period: usize) -> Self {
        self.rsi_period = Some(period);
        self.update_max_history();
        self
    }

    /// Set MACD configuration
    pub fn with_macd(mut self, fast: usize, slow: usize, signal: usize) -> Self {
        self.macd_config = Some((fast, slow, signal));
        self.update_max_history();
        self
    }

    /// Set Bollinger Bands configuration
    pub fn with_bollinger_bands(mut self, period: usize, std_dev: f64) -> Self {
        self.bollinger_bands = Some((period, std_dev));
        self.update_max_history();
        self
    }

    /// Set ATR period
    pub fn with_atr_period(mut self, period: usize) -> Self {
        self.atr_period = Some(period);
        self.update_max_history();
        self
    }

    /// Include or exclude momentum indicators
    pub fn with_momentum(mut self, include: bool) -> Self {
        self.include_momentum = include;
        self
    }

    /// Update the maximum history length based on configured indicators
    fn update_max_history(&mut self) {
        let mut max = 0;

        for &period in &self.sma_periods {
            max = max.max(period);
        }
        for &period in &self.ema_periods {
            max = max.max(period);
        }
        if let Some(period) = self.rsi_period {
            max = max.max(period + 1);
        }
        if let Some((_, slow, signal)) = self.macd_config {
            max = max.max(slow + signal);
        }
        if let Some((period, _)) = self.bollinger_bands {
            max = max.max(period);
        }
        if let Some(period) = self.atr_period {
            max = max.max(period);
        }

        self.max_history = max.max(200); // Minimum 200 for safety
    }

    /// Update price history with new event
    pub fn update(&mut self, event: &MarketDataEvent) -> Result<()> {
        let (price, high, low, volume) = match event {
            MarketDataEvent::Trade(trade) => {
                let price = trade.price.to_f64().ok_or_else(|| {
                    MLError::feature_extraction("Failed to convert trade price to f64")
                })?;
                (price, price, price, trade.quantity.to_f64().unwrap_or(0.0))
            }
            MarketDataEvent::Ticker(ticker) => {
                let price = ticker.last_price.to_f64().ok_or_else(|| {
                    MLError::feature_extraction("Failed to convert ticker price to f64")
                })?;
                let high = ticker.high_24h.and_then(|d| d.to_f64()).unwrap_or(price);
                let low = ticker.low_24h.and_then(|d| d.to_f64()).unwrap_or(price);
                let volume = ticker.volume_24h.to_f64().unwrap_or(0.0);
                (price, high, low, volume)
            }
            MarketDataEvent::Kline(kline) => {
                let price = kline.close.to_f64().ok_or_else(|| {
                    MLError::feature_extraction("Failed to convert kline close to f64")
                })?;
                let high = kline.high.to_f64().unwrap_or(price);
                let low = kline.low.to_f64().unwrap_or(price);
                let volume = kline.volume.to_f64().unwrap_or(0.0);
                (price, high, low, volume)
            }
            _ => {
                return Err(MLError::feature_extraction(
                    "Unsupported event type for technical indicators",
                ));
            }
        };

        self.price_history.push_back(price);
        self.high_history.push_back(high);
        self.low_history.push_back(low);
        self.volume_history.push_back(volume);

        // Trim history to max length
        while self.price_history.len() > self.max_history {
            self.price_history.pop_front();
            self.high_history.pop_front();
            self.low_history.pop_front();
            self.volume_history.pop_front();
        }

        Ok(())
    }

    /// Calculate RSI (Relative Strength Index)
    fn calculate_rsi(&self, period: usize) -> Option<f64> {
        if self.price_history.len() < period + 1 {
            return None;
        }

        let prices: Vec<f64> = self.price_history.iter().copied().collect();
        let mut gains = 0.0;
        let mut losses = 0.0;

        for i in (prices.len() - period)..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }

        let avg_gain = gains / period as f64;
        let avg_loss = losses / period as f64;

        if avg_loss == 0.0 {
            return Some(100.0);
        }

        let rs = avg_gain / avg_loss;
        Some(100.0 - (100.0 / (1.0 + rs)))
    }

    /// Calculate MACD (Moving Average Convergence Divergence)
    fn calculate_macd(&self, fast: usize, slow: usize, _signal: usize) -> Option<(f64, f64, f64)> {
        let prices: Vec<f64> = self.price_history.iter().copied().collect();

        let ema_fast = calculate_ema(&prices, fast)?;
        let ema_slow = calculate_ema(&prices, slow)?;
        let macd_line = ema_fast - ema_slow;

        // For signal line, we need MACD history (simplified here)
        // In production, maintain MACD history
        let signal_line = macd_line; // Simplified
        let histogram = macd_line - signal_line;

        Some((macd_line, signal_line, histogram))
    }

    /// Calculate Bollinger Bands
    fn calculate_bollinger_bands(&self, period: usize, std_dev: f64) -> Option<(f64, f64, f64)> {
        let prices: Vec<f64> = self.price_history.iter().copied().collect();

        if prices.len() < period {
            return None;
        }

        let sma = calculate_sma(&prices, period)?;
        let std = calculate_std(&prices[prices.len() - period..])?;

        let upper = sma + (std_dev * std);
        let lower = sma - (std_dev * std);

        Some((upper, sma, lower))
    }

    /// Calculate ATR (Average True Range)
    fn calculate_atr(&self, period: usize) -> Option<f64> {
        if self.high_history.len() < period + 1 {
            return None;
        }

        let highs: Vec<f64> = self.high_history.iter().copied().collect();
        let lows: Vec<f64> = self.low_history.iter().copied().collect();
        let closes: Vec<f64> = self.price_history.iter().copied().collect();

        let mut true_ranges = Vec::new();

        for i in (highs.len() - period)..highs.len() {
            let high_low = highs[i] - lows[i];
            let high_close = (highs[i] - closes[i - 1]).abs();
            let low_close = (lows[i] - closes[i - 1]).abs();

            let tr = high_low.max(high_close).max(low_close);
            true_ranges.push(tr);
        }

        Some(true_ranges.iter().sum::<f64>() / period as f64)
    }

    /// Calculate momentum (rate of change)
    fn calculate_momentum(&self, period: usize) -> Option<f64> {
        if self.price_history.len() < period + 1 {
            return None;
        }

        let current = *self.price_history.back()?;
        let past = self.price_history[self.price_history.len() - period - 1];

        Some((current - past) / past * 100.0)
    }

    /// Calculate Rate of Change (ROC)
    fn calculate_roc(&self, period: usize) -> Option<f64> {
        self.calculate_momentum(period)
    }
}

impl FeatureExtractor for TechnicalIndicators {
    fn extract_single(&mut self, event: &MarketDataEvent) -> Result<Vec<f64>> {
        // Clone self to update history without mutating
        let mut temp = self.clone();
        temp.update(event)?;

        let mut features = Vec::new();

        // Current price
        let current_price = temp.price_history.back().copied().unwrap_or(0.0);
        features.push(current_price);

        // SMA indicators
        let prices: Vec<f64> = temp.price_history.iter().copied().collect();
        for &period in &temp.sma_periods {
            let sma = calculate_sma(&prices, period).unwrap_or(current_price);
            features.push(sma);
            // Price relative to SMA
            features.push(current_price / sma - 1.0);
        }

        // EMA indicators
        for &period in &temp.ema_periods {
            let ema = calculate_ema(&prices, period).unwrap_or(current_price);
            features.push(ema);
            // Price relative to EMA
            features.push(current_price / ema - 1.0);
        }

        // RSI
        if let Some(period) = temp.rsi_period {
            let rsi = temp.calculate_rsi(period).unwrap_or(50.0);
            features.push(rsi);
            features.push(rsi / 100.0); // Normalized RSI
        }

        // MACD
        if let Some((fast, slow, signal)) = temp.macd_config {
            if let Some((macd, signal_line, histogram)) = temp.calculate_macd(fast, slow, signal) {
                features.push(macd);
                features.push(signal_line);
                features.push(histogram);
            } else {
                features.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
        }

        // Bollinger Bands
        if let Some((period, std_dev)) = temp.bollinger_bands {
            if let Some((upper, middle, lower)) = temp.calculate_bollinger_bands(period, std_dev) {
                features.push(upper);
                features.push(middle);
                features.push(lower);
                // Bollinger Band %B
                let bb_percent = (current_price - lower) / (upper - lower);
                features.push(bb_percent);
                // Bandwidth
                features.push((upper - lower) / middle);
            } else {
                features.extend_from_slice(&[0.0, 0.0, 0.0, 0.5, 0.0]);
            }
        }

        // ATR
        if let Some(period) = temp.atr_period {
            let atr = temp.calculate_atr(period).unwrap_or(0.0);
            features.push(atr);
            // ATR as percentage of price
            features.push(atr / current_price * 100.0);
        }

        // Momentum indicators
        if temp.include_momentum {
            features.push(temp.calculate_momentum(5).unwrap_or(0.0));
            features.push(temp.calculate_momentum(10).unwrap_or(0.0));
            features.push(temp.calculate_roc(20).unwrap_or(0.0));
        }

        // Volume (if available)
        if let Some(&volume) = temp.volume_history.back() {
            features.push(volume);
            // Volume SMA
            let volumes: Vec<f64> = temp.volume_history.iter().copied().collect();
            let vol_sma = calculate_sma(&volumes, 20).unwrap_or(volume);
            features.push(vol_sma);
            features.push(volume / vol_sma.max(1.0)); // Volume ratio
        }

        Ok(features)
    }

    fn num_features(&self) -> usize {
        let mut count = 1; // Current price

        // SMA: value + ratio for each period
        count += self.sma_periods.len() * 2;

        // EMA: value + ratio for each period
        count += self.ema_periods.len() * 2;

        // RSI: value + normalized
        if self.rsi_period.is_some() {
            count += 2;
        }

        // MACD: line + signal + histogram
        if self.macd_config.is_some() {
            count += 3;
        }

        // Bollinger Bands: upper + middle + lower + %B + bandwidth
        if self.bollinger_bands.is_some() {
            count += 5;
        }

        // ATR: value + percentage
        if self.atr_period.is_some() {
            count += 2;
        }

        // Momentum: 3 different periods
        if self.include_momentum {
            count += 3;
        }

        // Volume: current + SMA + ratio
        count += 3;

        count
    }

    fn feature_names(&self) -> Vec<String> {
        let mut names = vec!["price".to_string()];

        for &period in &self.sma_periods {
            names.push(format!("sma_{}", period));
            names.push(format!("price_sma_{}_ratio", period));
        }

        for &period in &self.ema_periods {
            names.push(format!("ema_{}", period));
            names.push(format!("price_ema_{}_ratio", period));
        }

        if let Some(period) = self.rsi_period {
            names.push(format!("rsi_{}", period));
            names.push(format!("rsi_{}_normalized", period));
        }

        if let Some((fast, slow, signal)) = self.macd_config {
            names.push(format!("macd_{}_{}", fast, slow));
            names.push(format!("macd_signal_{}", signal));
            names.push("macd_histogram".to_string());
        }

        if let Some((period, _std_dev)) = self.bollinger_bands {
            names.push(format!("bb_{}_upper", period));
            names.push(format!("bb_{}_middle", period));
            names.push(format!("bb_{}_lower", period));
            names.push(format!("bb_{}_percent_b", period));
            names.push(format!("bb_{}_bandwidth", period));
        }

        if let Some(period) = self.atr_period {
            names.push(format!("atr_{}", period));
            names.push(format!("atr_{}_percent", period));
        }

        if self.include_momentum {
            names.push("momentum_5".to_string());
            names.push("momentum_10".to_string());
            names.push("roc_20".to_string());
        }

        names.push("volume".to_string());
        names.push("volume_sma_20".to_string());
        names.push("volume_ratio".to_string());

        names
    }

    fn min_history_length(&self) -> usize {
        self.max_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Symbol, TradeEvent};
    use rust_decimal::Decimal;

    fn create_test_trade(price: f64, quantity: f64) -> MarketDataEvent {
        MarketDataEvent::Trade(TradeEvent {
            exchange: janus_core::Exchange::Bybit,
            symbol: Symbol::new("BTC", "USD"),
            price: Decimal::from_f64_retain(price).unwrap(),
            quantity: Decimal::from_f64_retain(quantity).unwrap(),
            timestamp: 1234567890,
            received_at: 1234567890,
            trade_id: "test".to_string(),
            side: janus_core::Side::Buy,
            buyer_is_maker: Some(false),
        })
    }

    #[test]
    fn test_technical_indicators_default() {
        let indicators = TechnicalIndicators::new();
        assert!(!indicators.sma_periods.is_empty());
        assert!(indicators.rsi_period.is_some());
    }

    #[test]
    fn test_feature_names_count_matches() {
        let indicators = TechnicalIndicators::new();
        let num_features = indicators.num_features();
        let names = indicators.feature_names();
        assert_eq!(num_features, names.len());
    }

    #[test]
    fn test_extract_single() {
        let mut indicators = TechnicalIndicators::new();
        let event = create_test_trade(50000.0, 1.0);

        let features = indicators.extract_single(&event).unwrap();
        assert_eq!(features.len(), indicators.num_features());
    }

    #[test]
    fn test_update_history() {
        let mut indicators = TechnicalIndicators::new();
        let event = create_test_trade(50000.0, 1.0);

        indicators.update(&event).unwrap();
        assert_eq!(indicators.price_history.len(), 1);
    }

    #[test]
    fn test_rsi_calculation() {
        let mut indicators = TechnicalIndicators::new().with_rsi_period(5);

        // Add some price data
        for price in &[100.0, 102.0, 101.0, 103.0, 104.0, 103.0] {
            let event = create_test_trade(*price, 1.0);
            indicators.update(&event).unwrap();
        }

        let rsi = indicators.calculate_rsi(5);
        assert!(rsi.is_some());
        let rsi_value = rsi.unwrap();
        assert!((0.0..=100.0).contains(&rsi_value));
    }

    #[test]
    fn test_momentum_calculation() {
        let mut indicators = TechnicalIndicators::new();

        for price in &[100.0, 105.0, 110.0, 115.0, 120.0, 125.0] {
            let event = create_test_trade(*price, 1.0);
            indicators.update(&event).unwrap();
        }

        let momentum = indicators.calculate_momentum(5);
        assert!(momentum.is_some());
        assert!(momentum.unwrap() > 0.0); // Price is increasing
    }
}
