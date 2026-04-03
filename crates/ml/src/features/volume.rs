//! Volume-based features for ML models
//!
//! This module provides volume-derived features including:
//! - VWAP (Volume Weighted Average Price)
//! - Money Flow Index (MFI)
//! - Accumulation/Distribution Line
//! - Volume momentum and trends
//! - On-Balance Volume (OBV)
//! - Volume profile features
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::features::volume::VolumeFeatures;
//!
//! let mut volume_features = VolumeFeatures::new()
//!     .with_vwap_windows(&[10, 20, 50])
//!     .with_mfi_period(14);
//!
//! let features = volume_features.extract_single(&market_event)?;
//! ```

use super::FeatureExtractor;
use crate::error::Result;
use janus_core::MarketDataEvent;
use rust_decimal::prelude::ToPrimitive;
use std::collections::VecDeque;

/// Configuration and state for volume-based features
#[derive(Debug, Clone)]
pub struct VolumeFeatures {
    /// Windows for VWAP calculation
    pub vwap_windows: Vec<usize>,

    /// Period for Money Flow Index
    pub mfi_period: Option<usize>,

    /// Include On-Balance Volume
    pub include_obv: bool,

    /// Include volume momentum
    pub include_volume_momentum: bool,

    /// Windows for volume moving averages
    pub volume_ma_windows: Vec<usize>,

    /// Price history
    price_history: VecDeque<f64>,

    /// Volume history
    volume_history: VecDeque<f64>,

    /// High price history
    high_history: VecDeque<f64>,

    /// Low price history
    low_history: VecDeque<f64>,

    /// Close price history
    close_history: VecDeque<f64>,

    /// Cumulative volume * price (for VWAP)
    vwap_pv_history: VecDeque<f64>,

    /// Cumulative volume (for VWAP)
    vwap_v_history: VecDeque<f64>,

    /// On-Balance Volume cumulative
    obv_cumulative: f64,

    /// Previous close for OBV
    prev_close: Option<f64>,

    /// Maximum history length needed
    max_history: usize,
}

impl Default for VolumeFeatures {
    fn default() -> Self {
        Self::new()
    }
}

impl VolumeFeatures {
    /// Create a new volume features extractor with default settings
    pub fn new() -> Self {
        Self {
            vwap_windows: vec![10, 20, 50],
            mfi_period: Some(14),
            include_obv: true,
            include_volume_momentum: true,
            volume_ma_windows: vec![5, 10, 20],
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            high_history: VecDeque::new(),
            low_history: VecDeque::new(),
            close_history: VecDeque::new(),
            vwap_pv_history: VecDeque::new(),
            vwap_v_history: VecDeque::new(),
            obv_cumulative: 0.0,
            prev_close: None,
            max_history: 100,
        }
    }

    /// Set VWAP calculation windows
    pub fn with_vwap_windows(mut self, windows: &[usize]) -> Self {
        self.vwap_windows = windows.to_vec();
        self.update_max_history();
        self
    }

    /// Set Money Flow Index period
    pub fn with_mfi_period(mut self, period: usize) -> Self {
        self.mfi_period = Some(period);
        self.update_max_history();
        self
    }

    /// Include or exclude OBV
    pub fn with_obv(mut self, include: bool) -> Self {
        self.include_obv = include;
        self
    }

    /// Include or exclude volume momentum
    pub fn with_volume_momentum(mut self, include: bool) -> Self {
        self.include_volume_momentum = include;
        self
    }

    /// Set volume moving average windows
    pub fn with_volume_ma_windows(mut self, windows: &[usize]) -> Self {
        self.volume_ma_windows = windows.to_vec();
        self.update_max_history();
        self
    }

    /// Update the maximum history length based on configured features
    fn update_max_history(&mut self) {
        let mut max = 0;

        for &window in &self.vwap_windows {
            max = max.max(window);
        }

        if let Some(period) = self.mfi_period {
            max = max.max(period);
        }

        for &window in &self.volume_ma_windows {
            max = max.max(window);
        }

        self.max_history = max.max(100); // Minimum 100
    }

    /// Update history buffers with new market data
    pub fn update_history(&mut self, event: &MarketDataEvent) {
        match event {
            MarketDataEvent::Trade(trade) => {
                let price = trade.price.to_f64().unwrap_or(0.0);
                let volume = trade.quantity.to_f64().unwrap_or(0.0);
                self.add_data(price, volume, price, price, price);
            }
            MarketDataEvent::Kline(kline) => {
                let close = kline.close.to_f64().unwrap_or(0.0);
                let volume = kline.volume.to_f64().unwrap_or(0.0);
                let high = kline.high.to_f64().unwrap_or(0.0);
                let low = kline.low.to_f64().unwrap_or(0.0);
                self.add_data(close, volume, high, low, close);
            }
            _ => {}
        }
    }

    /// Add price and volume data to history buffers
    fn add_data(&mut self, price: f64, volume: f64, high: f64, low: f64, close: f64) {
        self.price_history.push_back(price);
        self.volume_history.push_back(volume);
        self.high_history.push_back(high);
        self.low_history.push_back(low);
        self.close_history.push_back(close);

        // Update VWAP components
        let pv = price * volume;
        self.vwap_pv_history.push_back(pv);
        self.vwap_v_history.push_back(volume);

        // Update OBV
        if let Some(prev) = self.prev_close {
            if close > prev {
                self.obv_cumulative += volume;
            } else if close < prev {
                self.obv_cumulative -= volume;
            }
            // No change if close == prev
        }
        self.prev_close = Some(close);

        // Trim to max history
        while self.price_history.len() > self.max_history {
            self.price_history.pop_front();
            self.volume_history.pop_front();
            self.high_history.pop_front();
            self.low_history.pop_front();
            self.close_history.pop_front();
            self.vwap_pv_history.pop_front();
            self.vwap_v_history.pop_front();
        }
    }

    /// Calculate VWAP over a window
    fn calculate_vwap(&self, window: usize) -> Option<f64> {
        if self.vwap_pv_history.len() < window || self.vwap_v_history.len() < window {
            return None;
        }

        let start_idx = self.vwap_pv_history.len() - window;

        let sum_pv: f64 = self.vwap_pv_history.iter().skip(start_idx).sum();
        let sum_v: f64 = self.vwap_v_history.iter().skip(start_idx).sum();

        if sum_v == 0.0 {
            return None;
        }

        Some(sum_pv / sum_v)
    }

    /// Calculate Money Flow Index (MFI)
    fn calculate_mfi(&self, period: usize) -> Option<f64> {
        if self.high_history.len() < period + 1
            || self.low_history.len() < period + 1
            || self.close_history.len() < period + 1
            || self.volume_history.len() < period + 1
        {
            return None;
        }

        let start_idx = self.high_history.len() - period - 1;

        let mut positive_flow = 0.0;
        let mut negative_flow = 0.0;

        for i in start_idx..self.high_history.len() - 1 {
            let typical_price =
                (self.high_history[i + 1] + self.low_history[i + 1] + self.close_history[i + 1])
                    / 3.0;

            let prev_typical =
                (self.high_history[i] + self.low_history[i] + self.close_history[i]) / 3.0;

            let raw_money_flow = typical_price * self.volume_history[i + 1];

            if typical_price > prev_typical {
                positive_flow += raw_money_flow;
            } else if typical_price < prev_typical {
                negative_flow += raw_money_flow;
            }
        }

        if negative_flow == 0.0 {
            return Some(100.0);
        }

        let money_ratio = positive_flow / negative_flow;
        let mfi = 100.0 - (100.0 / (1.0 + money_ratio));

        Some(mfi)
    }

    /// Calculate Accumulation/Distribution Line
    fn calculate_ad_line(&self) -> Option<f64> {
        if self.high_history.is_empty()
            || self.low_history.is_empty()
            || self.close_history.is_empty()
            || self.volume_history.is_empty()
        {
            return None;
        }

        let mut ad_line = 0.0;

        for i in 0..self.high_history.len() {
            let high = self.high_history[i];
            let low = self.low_history[i];
            let close = self.close_history[i];
            let volume = self.volume_history[i];

            let range = high - low;
            if range == 0.0 {
                continue;
            }

            let clv = ((close - low) - (high - close)) / range;
            ad_line += clv * volume;
        }

        Some(ad_line)
    }

    /// Calculate volume moving average
    fn calculate_volume_ma(&self, window: usize) -> Option<f64> {
        if self.volume_history.len() < window {
            return None;
        }

        let start_idx = self.volume_history.len() - window;
        let sum: f64 = self.volume_history.iter().skip(start_idx).sum();

        Some(sum / window as f64)
    }

    /// Calculate volume momentum (rate of change)
    fn calculate_volume_momentum(&self, window: usize) -> Option<f64> {
        if self.volume_history.len() <= window {
            return None;
        }

        let current = *self.volume_history.back()?;
        let past = self.volume_history[self.volume_history.len() - window - 1];

        if past == 0.0 {
            return None;
        }

        Some((current - past) / past * 100.0)
    }

    /// Calculate volume ratio (current vs average)
    fn calculate_volume_ratio(&self, window: usize) -> Option<f64> {
        if self.volume_history.len() < window {
            return None;
        }

        let current = *self.volume_history.back()?;
        let avg = self.calculate_volume_ma(window)?;

        if avg == 0.0 {
            return None;
        }

        Some(current / avg)
    }

    /// Calculate relative volume position
    fn calculate_relative_volume(&self) -> Option<f64> {
        if self.volume_history.len() < 20 {
            return None;
        }

        let current = *self.volume_history.back()?;
        let start_idx = self.volume_history.len() - 20;

        let min_vol = self
            .volume_history
            .iter()
            .skip(start_idx)
            .fold(f64::INFINITY, |a, &b| a.min(b));

        let max_vol = self
            .volume_history
            .iter()
            .skip(start_idx)
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let range = max_vol - min_vol;
        if range == 0.0 {
            return Some(0.5); // Middle if no range
        }

        Some((current - min_vol) / range)
    }

    /// Extract all configured features
    fn extract_features(&mut self, event: &MarketDataEvent) -> Result<Vec<f64>> {
        // Update history first
        self.update_history(event);

        let mut features = Vec::new();

        // VWAP for each window
        for &window in &self.vwap_windows {
            features.push(self.calculate_vwap(window).unwrap_or(0.0));
        }

        // VWAP distance (current price vs VWAP)
        for &window in &self.vwap_windows {
            if let (Some(vwap), Some(&current)) =
                (self.calculate_vwap(window), self.price_history.back())
            {
                if vwap != 0.0 {
                    features.push((current - vwap) / vwap * 100.0);
                } else {
                    features.push(0.0);
                }
            } else {
                features.push(0.0);
            }
        }

        // Money Flow Index
        if let Some(period) = self.mfi_period {
            features.push(self.calculate_mfi(period).unwrap_or(50.0));
        }

        // Accumulation/Distribution Line
        features.push(self.calculate_ad_line().unwrap_or(0.0));

        // On-Balance Volume
        if self.include_obv {
            features.push(self.obv_cumulative);
        }

        // Volume moving averages
        for &window in &self.volume_ma_windows {
            features.push(self.calculate_volume_ma(window).unwrap_or(0.0));
        }

        // Volume ratios (current vs MA)
        for &window in &self.volume_ma_windows {
            features.push(self.calculate_volume_ratio(window).unwrap_or(1.0));
        }

        // Volume momentum
        if self.include_volume_momentum {
            for &window in &self.volume_ma_windows {
                features.push(self.calculate_volume_momentum(window).unwrap_or(0.0));
            }
        }

        // Relative volume position
        features.push(self.calculate_relative_volume().unwrap_or(0.5));

        Ok(features)
    }
}

impl FeatureExtractor for VolumeFeatures {
    fn extract_single(&mut self, event: &MarketDataEvent) -> Result<Vec<f64>> {
        self.extract_features(event)
    }

    fn num_features(&self) -> usize {
        let mut count = 0;

        // VWAP
        count += self.vwap_windows.len();

        // VWAP distance
        count += self.vwap_windows.len();

        // MFI
        if self.mfi_period.is_some() {
            count += 1;
        }

        // AD Line
        count += 1;

        // OBV
        if self.include_obv {
            count += 1;
        }

        // Volume MAs
        count += self.volume_ma_windows.len();

        // Volume ratios
        count += self.volume_ma_windows.len();

        // Volume momentum
        if self.include_volume_momentum {
            count += self.volume_ma_windows.len();
        }

        // Relative volume
        count += 1;

        count
    }

    fn feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        // VWAP
        for &window in &self.vwap_windows {
            names.push(format!("vwap_{}", window));
        }

        // VWAP distance
        for &window in &self.vwap_windows {
            names.push(format!("vwap_dist_{}", window));
        }

        // MFI
        if let Some(period) = self.mfi_period {
            names.push(format!("mfi_{}", period));
        }

        // AD Line
        names.push("ad_line".to_string());

        // OBV
        if self.include_obv {
            names.push("obv".to_string());
        }

        // Volume MAs
        for &window in &self.volume_ma_windows {
            names.push(format!("volume_ma_{}", window));
        }

        // Volume ratios
        for &window in &self.volume_ma_windows {
            names.push(format!("volume_ratio_{}", window));
        }

        // Volume momentum
        if self.include_volume_momentum {
            for &window in &self.volume_ma_windows {
                names.push(format!("volume_momentum_{}", window));
            }
        }

        // Relative volume
        names.push("relative_volume".to_string());

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

    fn create_test_trade(price: f64, volume: f64) -> MarketDataEvent {
        MarketDataEvent::Trade(TradeEvent {
            exchange: Exchange::Bybit,
            symbol: Symbol::new("BTC", "USDT"),
            timestamp: chrono::Utc::now().timestamp_micros(),
            received_at: chrono::Utc::now().timestamp_micros(),
            price: Decimal::from_f64_retain(price).unwrap(),
            quantity: Decimal::from_f64_retain(volume).unwrap(),
            side: Side::Buy,
            trade_id: "test".to_string(),
            buyer_is_maker: Some(false),
        })
    }

    fn create_test_kline(
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> MarketDataEvent {
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
            volume: Decimal::from_f64_retain(volume).unwrap(),
            quote_volume: Some(Decimal::new(10000, 0)),
            trades: Some(10),
            is_closed: true,
        })
    }

    #[test]
    fn test_volume_features_default() {
        let features = VolumeFeatures::default();
        assert_eq!(features.vwap_windows, vec![10, 20, 50]);
        assert_eq!(features.mfi_period, Some(14));
        assert!(features.include_obv);
        assert!(features.include_volume_momentum);
    }

    #[test]
    fn test_volume_features_builder() {
        let features = VolumeFeatures::new()
            .with_vwap_windows(&[10, 20])
            .with_mfi_period(20)
            .with_obv(false)
            .with_volume_momentum(false)
            .with_volume_ma_windows(&[5, 10]);

        assert_eq!(features.vwap_windows, vec![10, 20]);
        assert_eq!(features.mfi_period, Some(20));
        assert!(!features.include_obv);
        assert!(!features.include_volume_momentum);
        assert_eq!(features.volume_ma_windows, vec![5, 10]);
    }

    #[test]
    fn test_num_features() {
        let features = VolumeFeatures::new()
            .with_vwap_windows(&[10, 20])
            .with_mfi_period(14)
            .with_volume_ma_windows(&[5, 10]);

        // 2 vwap + 2 vwap_dist + 1 mfi + 1 ad + 1 obv + 2 vol_ma + 2 vol_ratio + 2 vol_momentum + 1 rel_vol
        assert_eq!(features.num_features(), 14);
    }

    #[test]
    fn test_feature_names() {
        let features = VolumeFeatures::new()
            .with_vwap_windows(&[10])
            .with_mfi_period(14)
            .with_volume_ma_windows(&[5]);

        let names = features.feature_names();
        assert!(names.contains(&"vwap_10".to_string()));
        assert!(names.contains(&"vwap_dist_10".to_string()));
        assert!(names.contains(&"mfi_14".to_string()));
        assert!(names.contains(&"ad_line".to_string()));
        assert!(names.contains(&"obv".to_string()));
        assert!(names.contains(&"volume_ma_5".to_string()));
    }

    #[test]
    fn test_calculate_vwap() {
        let mut features = VolumeFeatures::new();

        features.add_data(100.0, 10.0, 100.0, 100.0, 100.0);
        features.add_data(105.0, 20.0, 105.0, 105.0, 105.0);
        features.add_data(102.0, 15.0, 102.0, 102.0, 102.0);

        let vwap = features.calculate_vwap(3);
        assert!(vwap.is_some());

        // VWAP = (100*10 + 105*20 + 102*15) / (10 + 20 + 15)
        let expected = (100.0 * 10.0 + 105.0 * 20.0 + 102.0 * 15.0) / 45.0;
        assert!((vwap.unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn test_extract_features_trades() {
        let mut features = VolumeFeatures::new()
            .with_vwap_windows(&[3])
            .with_volume_ma_windows(&[2]);

        for (price, volume) in [(100.0, 10.0), (102.0, 15.0), (104.0, 20.0), (103.0, 12.0)] {
            let event = create_test_trade(price, volume);
            let result = features.extract_single(&event);
            assert!(result.is_ok());
        }

        let extracted = features
            .extract_single(&create_test_trade(105.0, 18.0))
            .unwrap();
        assert!(!extracted.is_empty());
    }

    #[test]
    fn test_extract_features_klines() {
        let mut features = VolumeFeatures::new()
            .with_vwap_windows(&[3])
            .with_volume_ma_windows(&[2]);

        let event1 = create_test_kline(100.0, 105.0, 98.0, 103.0, 100.0);
        let event2 = create_test_kline(103.0, 107.0, 102.0, 106.0, 150.0);
        let event3 = create_test_kline(106.0, 110.0, 105.0, 108.0, 120.0);

        features.extract_single(&event1).unwrap();
        features.extract_single(&event2).unwrap();
        let extracted = features.extract_single(&event3).unwrap();

        assert!(!extracted.is_empty());
    }

    #[test]
    fn test_obv_calculation() {
        let mut features = VolumeFeatures::new();

        features.add_data(100.0, 100.0, 100.0, 100.0, 100.0);
        features.add_data(105.0, 150.0, 105.0, 105.0, 105.0); // Price up, add volume
        features.add_data(103.0, 120.0, 103.0, 103.0, 103.0); // Price down, subtract volume

        // OBV should be: 0 + 150 - 120 = 30
        assert!((features.obv_cumulative - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_volume_momentum() {
        let mut features = VolumeFeatures::new();

        features.add_data(100.0, 100.0, 100.0, 100.0, 100.0);
        features.add_data(100.0, 110.0, 100.0, 100.0, 100.0);
        features.add_data(100.0, 120.0, 100.0, 100.0, 100.0);

        let mom = features.calculate_volume_momentum(2);
        assert!(mom.is_some());
        assert!(mom.unwrap() > 0.0); // Volume is increasing
    }
}
