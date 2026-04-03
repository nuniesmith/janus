//! Feature engineering module for financial time-series data.
//!
//! This module provides utilities to compute technical indicators and
//! combine them with raw OHLCV data to create rich feature sets for
//! machine learning models.

use crate::data::csv_loader::OhlcvCandle;
use crate::data::dataset::OhlcvDataset;
use anyhow::Result;
use rayon::prelude::*;

/// Feature configuration for technical indicators
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Include raw OHLCV features
    pub include_ohlcv: bool,
    /// Include returns (price changes)
    pub include_returns: bool,
    /// Include log returns
    pub include_log_returns: bool,
    /// Include volume changes
    pub include_volume_change: bool,
    /// SMA periods to compute
    pub sma_periods: Vec<usize>,
    /// EMA periods to compute
    pub ema_periods: Vec<usize>,
    /// RSI period
    pub rsi_period: Option<usize>,
    /// MACD configuration (fast, slow, signal)
    pub macd_config: Option<(usize, usize, usize)>,
    /// ATR period
    pub atr_period: Option<usize>,
    /// Include Bollinger Bands
    pub bollinger_bands: Option<(usize, f64)>, // (period, std_dev)
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            include_ohlcv: true,
            include_returns: true,
            include_log_returns: false,
            include_volume_change: false,
            sma_periods: vec![],
            ema_periods: vec![],
            rsi_period: None,
            macd_config: None,
            atr_period: None,
            bollinger_bands: None,
        }
    }
}

impl FeatureConfig {
    /// Create a config with common technical indicators
    pub fn with_common_indicators() -> Self {
        Self {
            include_ohlcv: true,
            include_returns: true,
            include_log_returns: false,
            include_volume_change: true,
            sma_periods: vec![10, 20],
            ema_periods: vec![8, 21],
            rsi_period: Some(14),
            macd_config: Some((12, 26, 9)),
            atr_period: Some(14),
            bollinger_bands: Some((20, 2.0)),
        }
    }

    /// Create a minimal config with only OHLCV and returns
    pub fn minimal() -> Self {
        Self {
            include_ohlcv: true,
            include_returns: true,
            ..Default::default()
        }
    }

    /// Calculate total number of features this config will produce
    pub fn num_features(&self) -> usize {
        let mut count = 0;

        if self.include_ohlcv {
            count += 5; // O, H, L, C, V
        }

        if self.include_returns {
            count += 1; // Return
        }

        if self.include_log_returns {
            count += 1; // Log return
        }

        if self.include_volume_change {
            count += 1; // Volume change
        }

        count += self.sma_periods.len(); // SMA for each period
        count += self.ema_periods.len(); // EMA for each period

        if self.rsi_period.is_some() {
            count += 1; // RSI
        }

        if self.macd_config.is_some() {
            count += 3; // MACD line, signal, histogram
        }

        if self.atr_period.is_some() {
            count += 1; // ATR
        }

        if self.bollinger_bands.is_some() {
            count += 3; // Upper, middle, lower bands
        }

        count
    }
}

/// Feature engineer that computes technical indicators
pub struct FeatureEngineer {
    config: FeatureConfig,
}

impl FeatureEngineer {
    /// Create a new feature engineer with the given configuration
    pub fn new(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Compute all features for a sequence of candles
    ///
    /// Returns a matrix of shape [sequence_length, num_features]
    pub fn compute_features(&self, candles: &[OhlcvCandle]) -> Result<Vec<Vec<f64>>> {
        if candles.is_empty() {
            return Ok(Vec::new());
        }

        let sequence_length = candles.len();
        let num_features = self.config.num_features();
        let mut features = vec![vec![0.0; num_features]; sequence_length];

        let mut feature_idx = 0;

        // Extract price arrays for indicator calculations
        let close_prices: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let high_prices: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let low_prices: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        // 1. Raw OHLCV features
        if self.config.include_ohlcv {
            for (i, candle) in candles.iter().enumerate() {
                features[i][feature_idx] = candle.open;
                features[i][feature_idx + 1] = candle.high;
                features[i][feature_idx + 2] = candle.low;
                features[i][feature_idx + 3] = candle.close;
                features[i][feature_idx + 4] = candle.volume;
            }
            feature_idx += 5;
        }

        // 2. Returns
        if self.config.include_returns {
            let returns = self.compute_returns(&close_prices);
            for (i, &ret) in returns.iter().enumerate() {
                features[i][feature_idx] = ret;
            }
            feature_idx += 1;
        }

        // 3. Log returns
        if self.config.include_log_returns {
            let log_returns = self.compute_log_returns(&close_prices);
            for (i, &ret) in log_returns.iter().enumerate() {
                features[i][feature_idx] = ret;
            }
            feature_idx += 1;
        }

        // 4. Volume change
        if self.config.include_volume_change {
            let vol_changes = self.compute_returns(&volumes);
            for (i, &change) in vol_changes.iter().enumerate() {
                features[i][feature_idx] = change;
            }
            feature_idx += 1;
        }

        // 5. SMA indicators
        for &period in &self.config.sma_periods {
            let sma_values = self.compute_sma(&close_prices, period);
            for (i, &val) in sma_values.iter().enumerate() {
                features[i][feature_idx] = val;
            }
            feature_idx += 1;
        }

        // 6. EMA indicators
        for &period in &self.config.ema_periods {
            let ema_values = self.compute_ema(&close_prices, period);
            for (i, &val) in ema_values.iter().enumerate() {
                features[i][feature_idx] = val;
            }
            feature_idx += 1;
        }

        // 7. RSI
        if let Some(period) = self.config.rsi_period {
            let rsi_values = self.compute_rsi(&close_prices, period);
            for (i, &val) in rsi_values.iter().enumerate() {
                features[i][feature_idx] = val;
            }
            feature_idx += 1;
        }

        // 8. MACD
        if let Some((fast, slow, signal)) = self.config.macd_config {
            let (macd_line, signal_line, histogram) =
                self.compute_macd(&close_prices, fast, slow, signal);

            for i in 0..sequence_length {
                features[i][feature_idx] = macd_line[i];
                features[i][feature_idx + 1] = signal_line[i];
                features[i][feature_idx + 2] = histogram[i];
            }
            feature_idx += 3;
        }

        // 9. ATR
        if let Some(period) = self.config.atr_period {
            let atr_values = self.compute_atr(&high_prices, &low_prices, &close_prices, period);
            for (i, &val) in atr_values.iter().enumerate() {
                features[i][feature_idx] = val;
            }
            feature_idx += 1;
        }

        // 10. Bollinger Bands
        if let Some((period, std_dev)) = self.config.bollinger_bands {
            let (upper, middle, lower) =
                self.compute_bollinger_bands(&close_prices, period, std_dev);
            for i in 0..sequence_length {
                features[i][feature_idx] = upper[i];
                features[i][feature_idx + 1] = middle[i];
                features[i][feature_idx + 2] = lower[i];
            }
            // feature_idx += 3; // Not used after this point
        }

        // Forward-fill NaN values (from warmup periods)
        self.forward_fill_nan(&mut features);

        Ok(features)
    }

    /// Compute features for an entire dataset
    pub fn compute_dataset_features(&self, dataset: &OhlcvDataset) -> Result<Vec<Vec<Vec<f64>>>> {
        let mut all_features = Vec::with_capacity(dataset.len());

        for sequence in &dataset.sequences {
            let features = self.compute_features(sequence)?;
            all_features.push(features);
        }

        Ok(all_features)
    }

    /// Compute features for a dataset in parallel using rayon
    ///
    /// This is significantly faster for large datasets with many sequences.
    /// Returns a matrix of shape [num_sequences, sequence_length, num_features]
    pub fn compute_dataset_features_parallel(
        &self,
        dataset: &OhlcvDataset,
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        dataset
            .sequences
            .par_iter()
            .map(|sequence| self.compute_features(sequence))
            .collect()
    }

    /// Compute features for multiple sequences in parallel
    ///
    /// # Arguments
    /// * `sequences` - Slice of sequences to process
    ///
    /// Returns a matrix of shape [num_sequences, sequence_length, num_features]
    pub fn compute_features_parallel(
        &self,
        sequences: &[Vec<OhlcvCandle>],
    ) -> Result<Vec<Vec<Vec<f64>>>> {
        sequences
            .par_iter()
            .map(|sequence| self.compute_features(sequence))
            .collect()
    }

    /// Get the number of features this engineer produces
    pub fn get_num_features(&self) -> usize {
        self.config.num_features()
    }

    // Helper methods for indicator calculations

    fn compute_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut returns = vec![0.0; prices.len()];
        for i in 1..prices.len() {
            returns[i] = (prices[i] - prices[i - 1]) / prices[i - 1];
        }
        returns
    }

    fn compute_log_returns(&self, prices: &[f64]) -> Vec<f64> {
        let mut log_returns = vec![0.0; prices.len()];
        for i in 1..prices.len() {
            log_returns[i] = (prices[i] / prices[i - 1]).ln();
        }
        log_returns
    }

    fn compute_sma(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut sma = vec![f64::NAN; prices.len()];

        if period > prices.len() {
            return sma;
        }

        for i in (period - 1)..prices.len() {
            let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
            sma[i] = sum / period as f64;
        }

        sma
    }

    fn compute_ema(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut ema = vec![f64::NAN; prices.len()];

        if period > prices.len() {
            return ema;
        }

        let alpha = 2.0 / (period as f64 + 1.0);

        // Initialize with SMA
        let first_sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;
        ema[period - 1] = first_sma;

        // Calculate EMA
        for i in period..prices.len() {
            ema[i] = prices[i] * alpha + ema[i - 1] * (1.0 - alpha);
        }

        ema
    }

    fn compute_rsi(&self, prices: &[f64], period: usize) -> Vec<f64> {
        let mut rsi = vec![f64::NAN; prices.len()];

        if prices.len() < period + 1 {
            return rsi;
        }

        // Calculate price changes
        let mut gains = vec![0.0; prices.len()];
        let mut losses = vec![0.0; prices.len()];

        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains[i] = change;
            } else {
                losses[i] = -change;
            }
        }

        // Calculate average gains and losses using EMA
        let avg_gains = self.compute_ema(&gains, period);
        let avg_losses = self.compute_ema(&losses, period);

        // Calculate RSI
        for i in period..prices.len() {
            if !avg_gains[i].is_nan() && !avg_losses[i].is_nan() {
                if avg_losses[i] == 0.0 {
                    rsi[i] = 100.0;
                } else {
                    let rs = avg_gains[i] / avg_losses[i];
                    rsi[i] = 100.0 - (100.0 / (1.0 + rs));
                }
            }
        }

        rsi
    }

    fn compute_macd(
        &self,
        prices: &[f64],
        fast_period: usize,
        slow_period: usize,
        signal_period: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let fast_ema = self.compute_ema(prices, fast_period);
        let slow_ema = self.compute_ema(prices, slow_period);

        // Calculate MACD line
        let mut macd_line = vec![f64::NAN; prices.len()];
        for i in 0..prices.len() {
            if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
                macd_line[i] = fast_ema[i] - slow_ema[i];
            }
        }

        // Calculate signal line
        let signal_line = self.compute_ema(&macd_line, signal_period);

        // Calculate histogram
        let mut histogram = vec![f64::NAN; prices.len()];
        for i in 0..prices.len() {
            if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
                histogram[i] = macd_line[i] - signal_line[i];
            }
        }

        (macd_line, signal_line, histogram)
    }

    fn compute_atr(&self, high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
        let tr = self.compute_true_range(high, low, close);
        self.compute_ema(&tr, period)
    }

    fn compute_true_range(&self, high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
        let mut tr = vec![f64::NAN; high.len()];

        if !high.is_empty() {
            tr[0] = high[0] - low[0];
        }

        for i in 1..high.len() {
            let tr1 = high[i] - low[i];
            let tr2 = (high[i] - close[i - 1]).abs();
            let tr3 = (low[i] - close[i - 1]).abs();

            tr[i] = tr1.max(tr2).max(tr3);
        }

        tr
    }

    fn compute_bollinger_bands(
        &self,
        prices: &[f64],
        period: usize,
        std_dev: f64,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let sma = self.compute_sma(prices, period);
        let mut upper = vec![f64::NAN; prices.len()];
        let mut lower = vec![f64::NAN; prices.len()];

        for i in (period - 1)..prices.len() {
            if !sma[i].is_nan() {
                // Calculate standard deviation for window
                let window = &prices[(i + 1 - period)..=i];
                let mean = sma[i];
                let variance: f64 =
                    window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
                let std = variance.sqrt();

                upper[i] = sma[i] + std_dev * std;
                lower[i] = sma[i] - std_dev * std;
            }
        }

        (upper, sma, lower)
    }

    /// Forward-fill NaN values in feature matrix
    fn forward_fill_nan(&self, features: &mut [Vec<f64>]) {
        if features.is_empty() {
            return;
        }

        let num_features = features[0].len();

        for feature_idx in 0..num_features {
            let mut last_valid: Option<f64> = None;

            for row in features.iter_mut() {
                if row[feature_idx].is_nan() {
                    if let Some(val) = last_valid {
                        row[feature_idx] = val;
                    } else {
                        // If no valid value yet, use 0
                        row[feature_idx] = 0.0;
                    }
                } else {
                    last_valid = Some(row[feature_idx]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_candles(n: usize) -> Vec<OhlcvCandle> {
        (0..n)
            .map(|i| {
                let price = 100.0 + i as f64;
                OhlcvCandle::new(
                    Utc::now(),
                    price,
                    price + 2.0,
                    price - 1.0,
                    price + 0.5,
                    1000.0 + i as f64 * 10.0,
                )
            })
            .collect()
    }

    #[test]
    fn test_feature_config_default() {
        let config = FeatureConfig::default();
        assert!(config.include_ohlcv);
        assert!(config.include_returns);
        assert_eq!(config.num_features(), 6); // OHLCV (5) + returns (1)
    }

    #[test]
    fn test_feature_config_minimal() {
        let config = FeatureConfig::minimal();
        assert_eq!(config.num_features(), 6); // OHLCV + returns
    }

    #[test]
    fn test_feature_config_with_indicators() {
        let config = FeatureConfig::with_common_indicators();
        let expected_features = 5 + // OHLCV
            1 + // returns
            1 + // volume change
            2 + // 2 SMA periods
            2 + // 2 EMA periods
            1 + // RSI
            3 + // MACD (line, signal, histogram)
            1 + // ATR
            3; // Bollinger bands (upper, middle, lower)

        assert_eq!(config.num_features(), expected_features);
    }

    #[test]
    fn test_compute_returns() {
        let engineer = FeatureEngineer::new(FeatureConfig::default());
        let prices = vec![100.0, 102.0, 101.0, 103.0];
        let returns = engineer.compute_returns(&prices);

        assert_eq!(returns.len(), 4);
        assert_eq!(returns[0], 0.0); // First return is 0
        assert!((returns[1] - 0.02).abs() < 1e-10); // (102-100)/100 = 0.02
        assert!((returns[2] - (-1.0 / 102.0)).abs() < 1e-10); // (101-102)/102 = -1/102
    }

    #[test]
    fn test_compute_sma() {
        let engineer = FeatureEngineer::new(FeatureConfig::default());
        let prices = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let sma = engineer.compute_sma(&prices, 3);

        assert_eq!(sma.len(), 5);
        assert!(sma[0].is_nan());
        assert!(sma[1].is_nan());
        assert!((sma[2] - 102.0).abs() < 1e-10); // (100+102+104)/3
        assert!((sma[3] - 104.0).abs() < 1e-10); // (102+104+106)/3
    }

    #[test]
    fn test_compute_ema() {
        let engineer = FeatureEngineer::new(FeatureConfig::default());
        let prices = vec![100.0, 102.0, 104.0, 106.0, 108.0];
        let ema = engineer.compute_ema(&prices, 3);

        assert_eq!(ema.len(), 5);
        assert!(ema[0].is_nan());
        assert!(ema[1].is_nan());
        assert!(!ema[2].is_nan()); // First EMA value (initialized with SMA)
        assert!(!ema[3].is_nan());
    }

    #[test]
    fn test_compute_features_basic() {
        let candles = create_test_candles(50);
        let config = FeatureConfig::minimal();
        let engineer = FeatureEngineer::new(config.clone());

        let features = engineer.compute_features(&candles).unwrap();

        assert_eq!(features.len(), 50); // sequence_length
        assert_eq!(features[0].len(), config.num_features());

        // Check OHLCV values
        assert!((features[0][0] - candles[0].open).abs() < 1e-10);
        assert!((features[0][3] - candles[0].close).abs() < 1e-10);
    }

    #[test]
    fn test_compute_features_with_indicators() {
        let candles = create_test_candles(100);
        let config = FeatureConfig {
            include_ohlcv: true,
            include_returns: true,
            sma_periods: vec![10, 20],
            ema_periods: vec![8],
            rsi_period: Some(14),
            ..Default::default()
        };

        let engineer = FeatureEngineer::new(config.clone());
        let features = engineer.compute_features(&candles).unwrap();

        assert_eq!(features.len(), 100);
        assert_eq!(features[0].len(), config.num_features());

        // Check that all features are valid (no NaN after forward fill)
        for row in &features {
            for &val in row {
                assert!(!val.is_nan(), "Feature contains NaN after forward fill");
            }
        }
    }

    #[test]
    fn test_compute_rsi() {
        let engineer = FeatureEngineer::new(FeatureConfig::default());
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i as f64 * 0.5)).collect();
        let rsi = engineer.compute_rsi(&prices, 14);

        assert_eq!(rsi.len(), 30);
        // RSI values should be between 0 and 100
        for val in rsi.iter().skip(14) {
            if !val.is_nan() {
                assert!(
                    *val >= 0.0 && *val <= 100.0,
                    "RSI value out of range: {}",
                    val
                );
            }
        }
    }

    #[test]
    fn test_compute_macd() {
        let engineer = FeatureEngineer::new(FeatureConfig::default());
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + (i as f64)).collect();
        let (macd_line, signal_line, histogram) = engineer.compute_macd(&prices, 12, 26, 9);

        assert_eq!(macd_line.len(), 50);
        assert_eq!(signal_line.len(), 50);
        assert_eq!(histogram.len(), 50);

        // Check histogram = macd_line - signal_line
        for i in 35..50 {
            if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
                let expected = macd_line[i] - signal_line[i];
                assert!((histogram[i] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_forward_fill_nan() {
        let engineer = FeatureEngineer::new(FeatureConfig::default());
        let mut features = vec![
            vec![1.0, f64::NAN, 3.0],
            vec![2.0, f64::NAN, 4.0],
            vec![3.0, 5.0, 5.0],
            vec![4.0, f64::NAN, 6.0],
        ];

        engineer.forward_fill_nan(&mut features);

        // First NaN should be filled with 0
        assert_eq!(features[0][1], 0.0);
        assert_eq!(features[1][1], 0.0);
        // After valid value appears, forward fill with that
        assert_eq!(features[2][1], 5.0);
        assert_eq!(features[3][1], 5.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let engineer = FeatureEngineer::new(FeatureConfig::default());
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + (i % 5) as f64).collect();
        let (upper, middle, lower) = engineer.compute_bollinger_bands(&prices, 20, 2.0);

        assert_eq!(upper.len(), 30);
        assert_eq!(middle.len(), 30);
        assert_eq!(lower.len(), 30);

        // Check that upper > middle > lower where defined
        for i in 20..30 {
            if !upper[i].is_nan() && !lower[i].is_nan() {
                assert!(
                    upper[i] > middle[i] && middle[i] > lower[i],
                    "Bollinger band ordering incorrect at index {}",
                    i
                );
            }
        }
    }
}
