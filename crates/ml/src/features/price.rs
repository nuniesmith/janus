//! Price-based features for ML models
//!
//! This module provides price-derived features including:
//! - Returns (simple and log returns)
//! - Volatility (realized volatility, Parkinson, Garman-Klass)
//! - Momentum (rate of change, acceleration)
//! - Range features (high-low ranges, true range)
//! - Price patterns (OHLC relationships)
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::features::price::PriceFeatures;
//!
//! let mut price_features = PriceFeatures::new()
//!     .with_return_periods(&[1, 5, 10])
//!     .with_volatility_windows(&[10, 20, 50]);
//!
//! let features = price_features.extract_single(&market_event)?;
//! ```

use super::FeatureExtractor;
use crate::error::Result;
use janus_core::MarketDataEvent;
use rust_decimal::prelude::ToPrimitive;
use std::collections::VecDeque;

/// Configuration and state for price-based features
#[derive(Debug, Clone)]
pub struct PriceFeatures {
    /// Periods for return calculation
    pub return_periods: Vec<usize>,

    /// Windows for volatility calculation
    pub volatility_windows: Vec<usize>,

    /// Windows for momentum calculation
    pub momentum_windows: Vec<usize>,

    /// Include log returns (in addition to simple returns)
    pub include_log_returns: bool,

    /// Include price acceleration
    pub include_acceleration: bool,

    /// Include range features
    pub include_ranges: bool,

    /// Price history buffer
    price_history: VecDeque<f64>,

    /// High price history
    high_history: VecDeque<f64>,

    /// Low price history
    low_history: VecDeque<f64>,

    /// Close price history
    close_history: VecDeque<f64>,

    /// Open price history
    open_history: VecDeque<f64>,

    /// Maximum history length needed
    max_history: usize,
}

impl Default for PriceFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl PriceFeatures {
    /// Create a new price features extractor with default settings
    pub fn new() -> Self {
        Self {
            return_periods: vec![1, 5, 10, 20],
            volatility_windows: vec![10, 20, 50],
            momentum_windows: vec![5, 10, 20],
            include_log_returns: true,
            include_acceleration: true,
            include_ranges: true,
            price_history: VecDeque::new(),
            high_history: VecDeque::new(),
            low_history: VecDeque::new(),
            close_history: VecDeque::new(),
            open_history: VecDeque::new(),
            max_history: 100,
        }
    }

    /// Set return calculation periods
    pub fn with_return_periods(mut self, periods: &[usize]) -> Self {
        self.return_periods = periods.to_vec();
        self.update_max_history();
        self
    }

    /// Set volatility calculation windows
    pub fn with_volatility_windows(mut self, windows: &[usize]) -> Self {
        self.volatility_windows = windows.to_vec();
        self.update_max_history();
        self
    }

    /// Set momentum calculation windows
    pub fn with_momentum_windows(mut self, windows: &[usize]) -> Self {
        self.momentum_windows = windows.to_vec();
        self.update_max_history();
        self
    }

    /// Include or exclude log returns
    pub fn with_log_returns(mut self, include: bool) -> Self {
        self.include_log_returns = include;
        self
    }

    /// Include or exclude acceleration
    pub fn with_acceleration(mut self, include: bool) -> Self {
        self.include_acceleration = include;
        self
    }

    /// Include or exclude range features
    pub fn with_ranges(mut self, include: bool) -> Self {
        self.include_ranges = include;
        self
    }

    /// Update the maximum history length based on configured features
    fn update_max_history(&mut self) {
        let mut max = 0;

        for &period in &self.return_periods {
            max = max.max(period);
        }

        for &window in &self.volatility_windows {
            max = max.max(window);
        }

        for &window in &self.momentum_windows {
            max = max.max(window);
        }

        self.max_history = max.max(100); // Minimum 100
    }

    /// Update history buffers with new price data
    pub fn update_history(&mut self, event: &MarketDataEvent) {
        match event {
            MarketDataEvent::Trade(trade) => {
                let price = trade.price.to_f64().unwrap_or(0.0);
                self.add_price(price, price, price, price, price);
            }
            MarketDataEvent::Kline(kline) => {
                let open = kline.open.to_f64().unwrap_or(0.0);
                let high = kline.high.to_f64().unwrap_or(0.0);
                let low = kline.low.to_f64().unwrap_or(0.0);
                let close = kline.close.to_f64().unwrap_or(0.0);
                self.add_price(close, open, high, low, close);
            }
            _ => {}
        }
    }

    /// Add price data to history buffers
    fn add_price(&mut self, price: f64, open: f64, high: f64, low: f64, close: f64) {
        self.price_history.push_back(price);
        self.open_history.push_back(open);
        self.high_history.push_back(high);
        self.low_history.push_back(low);
        self.close_history.push_back(close);

        // Trim to max history
        while self.price_history.len() > self.max_history {
            self.price_history.pop_front();
            self.open_history.pop_front();
            self.high_history.pop_front();
            self.low_history.pop_front();
            self.close_history.pop_front();
        }
    }

    /// Calculate simple return over a period
    fn calculate_return(&self, period: usize) -> Option<f64> {
        if self.price_history.len() <= period {
            return None;
        }

        let current = *self.price_history.back()?;
        let past = self.price_history[self.price_history.len() - period - 1];

        if past == 0.0 {
            return None;
        }

        Some((current - past) / past)
    }

    /// Calculate log return over a period
    fn calculate_log_return(&self, period: usize) -> Option<f64> {
        if self.price_history.len() <= period {
            return None;
        }

        let current = *self.price_history.back()?;
        let past = self.price_history[self.price_history.len() - period - 1];

        if past <= 0.0 || current <= 0.0 {
            return None;
        }

        Some((current / past).ln())
    }

    /// Calculate realized volatility over a window
    fn calculate_volatility(&self, window: usize) -> Option<f64> {
        if self.price_history.len() < window {
            return None;
        }

        let start_idx = self.price_history.len() - window;
        let prices: Vec<f64> = self.price_history.iter().skip(start_idx).copied().collect();

        // Calculate returns
        let mut returns = Vec::new();
        for i in 1..prices.len() {
            if prices[i - 1] != 0.0 {
                returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
            }
        }

        if returns.is_empty() {
            return None;
        }

        // Calculate standard deviation
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;

        Some(variance.sqrt())
    }

    /// Calculate Parkinson volatility (uses high-low)
    fn calculate_parkinson_volatility(&self, window: usize) -> Option<f64> {
        if self.high_history.len() < window || self.low_history.len() < window {
            return None;
        }

        let start_idx = self.high_history.len() - window;
        let highs: Vec<f64> = self.high_history.iter().skip(start_idx).copied().collect();
        let lows: Vec<f64> = self.low_history.iter().skip(start_idx).copied().collect();

        let sum: f64 = highs
            .iter()
            .zip(lows.iter())
            .filter_map(|(h, l)| {
                if *h > 0.0 && *l > 0.0 {
                    Some((h / l).ln().powi(2))
                } else {
                    None
                }
            })
            .sum();

        if sum == 0.0 {
            return None;
        }

        // Parkinson constant: 1 / (4 * ln(2))
        let parkinson_const = 0.361;
        Some((parkinson_const * sum / window as f64).sqrt())
    }

    /// Calculate momentum (rate of change)
    fn calculate_momentum(&self, window: usize) -> Option<f64> {
        if self.price_history.len() <= window {
            return None;
        }

        let current = *self.price_history.back()?;
        let past = self.price_history[self.price_history.len() - window - 1];

        if past == 0.0 {
            return None;
        }

        Some((current - past) / past * 100.0)
    }

    /// Calculate price acceleration (change in momentum)
    fn calculate_acceleration(&self, window: usize) -> Option<f64> {
        if self.price_history.len() <= window + 1 {
            return None;
        }

        let mom_current = self.calculate_momentum(window)?;

        // Need to calculate momentum from one period earlier
        let len = self.price_history.len();
        if len <= window + 2 {
            return None;
        }

        let current = self.price_history[len - 2];
        let past = self.price_history[len - window - 2];

        if past == 0.0 {
            return None;
        }

        let mom_previous = (current - past) / past * 100.0;

        Some(mom_current - mom_previous)
    }

    /// Calculate true range
    fn calculate_true_range(&self) -> Option<f64> {
        if self.high_history.len() < 2 || self.low_history.len() < 2 || self.close_history.len() < 2
        {
            return None;
        }

        let high = *self.high_history.back()?;
        let low = *self.low_history.back()?;
        let prev_close = self.close_history[self.close_history.len() - 2];

        let tr1 = high - low;
        let tr2 = (high - prev_close).abs();
        let tr3 = (low - prev_close).abs();

        Some(tr1.max(tr2).max(tr3))
    }

    /// Calculate high-low range percentage
    fn calculate_hl_range(&self) -> Option<f64> {
        let high = *self.high_history.back()?;
        let low = *self.low_history.back()?;
        let close = *self.close_history.back()?;

        if close == 0.0 {
            return None;
        }

        Some((high - low) / close * 100.0)
    }

    /// Calculate close-open range
    fn calculate_co_range(&self) -> Option<f64> {
        let open = *self.open_history.back()?;
        let close = *self.close_history.back()?;

        if open == 0.0 {
            return None;
        }

        Some((close - open) / open * 100.0)
    }

    /// Extract all configured features
    fn extract_features(&mut self, event: &MarketDataEvent) -> Result<Vec<f64>> {
        // Update history first
        self.update_history(event);

        let mut features = Vec::new();

        // Simple returns
        for &period in &self.return_periods {
            features.push(self.calculate_return(period).unwrap_or(0.0));
        }

        // Log returns (if enabled)
        if self.include_log_returns {
            for &period in &self.return_periods {
                features.push(self.calculate_log_return(period).unwrap_or(0.0));
            }
        }

        // Realized volatility
        for &window in &self.volatility_windows {
            features.push(self.calculate_volatility(window).unwrap_or(0.0));
        }

        // Parkinson volatility
        for &window in &self.volatility_windows {
            features.push(self.calculate_parkinson_volatility(window).unwrap_or(0.0));
        }

        // Momentum
        for &window in &self.momentum_windows {
            features.push(self.calculate_momentum(window).unwrap_or(0.0));
        }

        // Acceleration (if enabled)
        if self.include_acceleration {
            for &window in &self.momentum_windows {
                features.push(self.calculate_acceleration(window).unwrap_or(0.0));
            }
        }

        // Range features (if enabled)
        if self.include_ranges {
            features.push(self.calculate_true_range().unwrap_or(0.0));
            features.push(self.calculate_hl_range().unwrap_or(0.0));
            features.push(self.calculate_co_range().unwrap_or(0.0));
        }

        Ok(features)
    }
}

impl FeatureExtractor for PriceFeatures {
    fn extract_single(&mut self, event: &MarketDataEvent) -> Result<Vec<f64>> {
        self.extract_features(event)
    }

    fn num_features(&self) -> usize {
        let mut count = 0;

        // Simple returns
        count += self.return_periods.len();

        // Log returns
        if self.include_log_returns {
            count += self.return_periods.len();
        }

        // Realized volatility
        count += self.volatility_windows.len();

        // Parkinson volatility
        count += self.volatility_windows.len();

        // Momentum
        count += self.momentum_windows.len();

        // Acceleration
        if self.include_acceleration {
            count += self.momentum_windows.len();
        }

        // Range features
        if self.include_ranges {
            count += 3; // true_range, hl_range, co_range
        }

        count
    }

    fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // Simple returns
        for &period in &self.return_periods {
            names.push(format!("return_{}", period));
        }

        // Log returns
        if self.include_log_returns {
            for &period in &self.return_periods {
                names.push(format!("log_return_{}", period));
            }
        }

        // Realized volatility
        for &window in &self.volatility_windows {
            names.push(format!("volatility_{}", window));
        }

        // Parkinson volatility
        for &window in &self.volatility_windows {
            names.push(format!("parkinson_vol_{}", window));
        }

        // Momentum
        for &window in &self.momentum_windows {
            names.push(format!("momentum_{}", window));
        }

        // Acceleration
        if self.include_acceleration {
            for &window in &self.momentum_windows {
                names.push(format!("acceleration_{}", window));
            }
        }

        // Range features
        if self.include_ranges {
            names.push("true_range".to_string());
            names.push("hl_range_pct".to_string());
            names.push("co_range_pct".to_string());
        }

        names
    }

    fn min_history_length(&self) -> usize {
        self.max_history
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, KlineEvent, Side, Symbol, TradeEvent};
    use rust_decimal::Decimal;

    fn create_test_trade(price: f64) -> MarketDataEvent {
        MarketDataEvent::Trade(TradeEvent {
            exchange: Exchange::Bybit,
            symbol: Symbol::new("BTC", "USDT"),
            timestamp: chrono::Utc::now().timestamp_micros(),
            received_at: chrono::Utc::now().timestamp_micros(),
            price: Decimal::from_f64_retain(price).unwrap(),
            quantity: Decimal::new(1, 1),
            side: Side::Buy,
            trade_id: "test".to_string(),
            buyer_is_maker: Some(false),
        })
    }

    fn create_test_kline(open: f64, high: f64, low: f64, close: f64) -> MarketDataEvent {
        MarketDataEvent::Kline(KlineEvent {
            exchange: Exchange::Bybit,
            symbol: Symbol::new("BTC", "USDT"),
            interval: "1m".to_string(),
            open_time: chrono::Utc::now().timestamp_micros(),
            close_time: chrono::Utc::now().timestamp_micros() + 60_000_000,
            open: Decimal::from_f64_retain(open).unwrap(),
            high: Decimal::from_f64_retain(high).unwrap(),
            low: Decimal::from_f64_retain(low).unwrap(),
            close: Decimal::from_f64_retain(close).unwrap(),
            volume: Decimal::new(100, 0),
            quote_volume: Some(Decimal::new(10000, 0)),
            trades: Some(10),
            is_closed: true,
        })
    }

    #[test]
    fn test_price_features_default() {
        let features = PriceFeatures::default();
        assert_eq!(features.return_periods, vec![1, 5, 10, 20]);
        assert_eq!(features.volatility_windows, vec![10, 20, 50]);
        assert!(features.include_log_returns);
        assert!(features.include_acceleration);
    }

    #[test]
    fn test_price_features_builder() {
        let features = PriceFeatures::new()
            .with_return_periods(&[1, 5])
            .with_volatility_windows(&[10, 20])
            .with_momentum_windows(&[5, 10])
            .with_log_returns(false)
            .with_acceleration(false)
            .with_ranges(false);

        assert_eq!(features.return_periods, vec![1, 5]);
        assert_eq!(features.volatility_windows, vec![10, 20]);
        assert_eq!(features.momentum_windows, vec![5, 10]);
        assert!(!features.include_log_returns);
        assert!(!features.include_acceleration);
        assert!(!features.include_ranges);
    }

    #[test]
    fn test_num_features() {
        let features = PriceFeatures::new()
            .with_return_periods(&[1, 5, 10])
            .with_volatility_windows(&[10, 20])
            .with_momentum_windows(&[5, 10]);

        // 3 returns + 3 log_returns + 2 volatility + 2 parkinson + 2 momentum + 2 acceleration + 3 ranges
        assert_eq!(features.num_features(), 17);
    }

    #[test]
    fn test_feature_names() {
        let features = PriceFeatures::new()
            .with_return_periods(&[1, 5])
            .with_volatility_windows(&[10])
            .with_momentum_windows(&[5])
            .with_log_returns(true)
            .with_acceleration(true)
            .with_ranges(true);

        let names = features.feature_names();
        assert!(names.contains(&"return_1".to_string()));
        assert!(names.contains(&"log_return_1".to_string()));
        assert!(names.contains(&"volatility_10".to_string()));
        assert!(names.contains(&"momentum_5".to_string()));
        assert!(names.contains(&"true_range".to_string()));
    }

    #[test]
    fn test_extract_features_trades() {
        let mut features = PriceFeatures::new()
            .with_return_periods(&[1])
            .with_volatility_windows(&[5])
            .with_momentum_windows(&[2]);

        // Add some price history
        for price in [100.0, 101.0, 102.0, 103.0, 104.0, 105.0] {
            let event = create_test_trade(price);
            let result = features.extract_single(&event);
            assert!(result.is_ok());
        }

        let extracted = features.extract_single(&create_test_trade(106.0)).unwrap();
        assert!(!extracted.is_empty());
    }

    #[test]
    fn test_extract_features_klines() {
        let mut features = PriceFeatures::new()
            .with_return_periods(&[1])
            .with_volatility_windows(&[3])
            .with_momentum_windows(&[2]);

        // Add some kline history
        let event1 = create_test_kline(100.0, 105.0, 98.0, 103.0);
        let event2 = create_test_kline(103.0, 107.0, 102.0, 106.0);
        let event3 = create_test_kline(106.0, 110.0, 105.0, 108.0);

        features.extract_single(&event1).unwrap();
        features.extract_single(&event2).unwrap();
        let extracted = features.extract_single(&event3).unwrap();

        assert!(!extracted.is_empty());
    }

    #[test]
    fn test_calculate_return() {
        let mut features = PriceFeatures::new();

        features.add_price(100.0, 100.0, 100.0, 100.0, 100.0);
        features.add_price(105.0, 105.0, 105.0, 105.0, 105.0);

        let ret = features.calculate_return(1);
        assert!(ret.is_some());
        assert!((ret.unwrap() - 0.05).abs() < 0.001); // 5% return
    }

    #[test]
    fn test_calculate_volatility() {
        let mut features = PriceFeatures::new();

        // Add varying prices
        for price in [100.0, 102.0, 99.0, 103.0, 98.0, 104.0] {
            features.add_price(price, price, price, price, price);
        }

        let vol = features.calculate_volatility(5);
        assert!(vol.is_some());
        assert!(vol.unwrap() > 0.0);
    }

    #[test]
    fn test_calculate_momentum() {
        let mut features = PriceFeatures::new();

        features.add_price(100.0, 100.0, 100.0, 100.0, 100.0);
        features.add_price(105.0, 105.0, 105.0, 105.0, 105.0);
        features.add_price(110.0, 110.0, 110.0, 110.0, 110.0);

        let mom = features.calculate_momentum(2);
        assert!(mom.is_some());
        assert!(mom.unwrap() > 0.0); // Positive momentum
    }

    #[test]
    fn test_calculate_true_range() {
        let mut features = PriceFeatures::new();

        features.add_price(100.0, 99.0, 101.0, 98.0, 100.0);
        features.add_price(102.0, 101.0, 104.0, 100.0, 102.0);

        let tr = features.calculate_true_range();
        assert!(tr.is_some());
        assert!(tr.unwrap() > 0.0);
    }
}
