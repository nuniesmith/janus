//! Technical Indicators Module
//!
//! Lightweight technical analysis indicators for the EMA trading strategy.
//! Implements EMA (Exponential Moving Average) and ATR (Average True Range).

use std::collections::VecDeque;
use std::error::Error;
use std::fmt;

/// Indicator calculation error
#[derive(Debug)]
pub enum IndicatorError {
    InsufficientData { required: usize, available: usize },
    InvalidParameter { name: String, value: f64 },
}

impl fmt::Display for IndicatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndicatorError::InsufficientData {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient data: required {} candles, but only {} available",
                    required, available
                )
            }
            IndicatorError::InvalidParameter { name, value } => {
                write!(f, "Invalid parameter {}: {}", name, value)
            }
        }
    }
}

impl Error for IndicatorError {}

/// Calculate Exponential Moving Average (EMA)
///
/// EMA gives more weight to recent prices, making it more responsive
/// to new information than a simple moving average.
///
/// Formula:
///     EMA_t = Price_t × α + EMA_(t-1) × (1 - α)
///     where α = 2 / (length + 1)
///
/// # Arguments
/// * `prices` - Price series (typically close prices)
/// * `period` - Period for EMA calculation
///
/// # Returns
/// Vector of EMA values (same length as input, with NaN for initial warmup)
pub fn ema(prices: &[f64], period: usize) -> Result<Vec<f64>, IndicatorError> {
    if period == 0 {
        return Err(IndicatorError::InvalidParameter {
            name: "period".to_string(),
            value: period as f64,
        });
    }

    if prices.len() < period {
        return Err(IndicatorError::InsufficientData {
            required: period,
            available: prices.len(),
        });
    }

    let mut result = vec![f64::NAN; prices.len()];
    let alpha = 2.0 / (period as f64 + 1.0);

    // Initialize with SMA for first value
    let first_sma: f64 = prices.iter().take(period).sum::<f64>() / period as f64;
    result[period - 1] = first_sma;

    // Calculate EMA for remaining values
    for i in period..prices.len() {
        result[i] = prices[i] * alpha + result[i - 1] * (1.0 - alpha);
    }

    Ok(result)
}

/// Calculate Simple Moving Average (SMA)
///
/// # Arguments
/// * `prices` - Price series
/// * `period` - Period for SMA calculation
///
/// # Returns
/// Vector of SMA values
#[allow(dead_code)]
pub fn sma(prices: &[f64], period: usize) -> Result<Vec<f64>, IndicatorError> {
    if period == 0 {
        return Err(IndicatorError::InvalidParameter {
            name: "period".to_string(),
            value: period as f64,
        });
    }

    if prices.len() < period {
        return Err(IndicatorError::InsufficientData {
            required: period,
            available: prices.len(),
        });
    }

    let mut result = vec![f64::NAN; prices.len()];

    for i in (period - 1)..prices.len() {
        let sum: f64 = prices[(i + 1 - period)..=i].iter().sum();
        result[i] = sum / period as f64;
    }

    Ok(result)
}

/// Calculate True Range (TR)
///
/// True Range is the greatest of:
/// 1. Current High - Current Low
/// 2. |Current High - Previous Close|
/// 3. |Current Low - Previous Close|
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
///
/// # Returns
/// Vector of True Range values
pub fn true_range(high: &[f64], low: &[f64], close: &[f64]) -> Result<Vec<f64>, IndicatorError> {
    if high.len() != low.len() || high.len() != close.len() {
        return Err(IndicatorError::InsufficientData {
            required: high.len(),
            available: low.len().min(close.len()),
        });
    }

    let mut result = vec![f64::NAN; high.len()];

    // First value is just high - low
    if !high.is_empty() {
        result[0] = high[0] - low[0];
    }

    // Calculate TR for remaining values
    for i in 1..high.len() {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();

        result[i] = tr1.max(tr2).max(tr3);
    }

    Ok(result)
}

/// Calculate Average True Range (ATR)
///
/// ATR is a measure of volatility. Higher ATR means higher volatility.
/// It's the moving average of True Range values.
///
/// # Arguments
/// * `high` - High prices
/// * `low` - Low prices
/// * `close` - Close prices
/// * `period` - Period for ATR calculation
///
/// # Returns
/// Vector of ATR values
pub fn atr(
    high: &[f64],
    low: &[f64],
    close: &[f64],
    period: usize,
) -> Result<Vec<f64>, IndicatorError> {
    let tr = true_range(high, low, close)?;

    // Use EMA smoothing for ATR
    ema(&tr, period)
}

/// Calculate Relative Strength Index (RSI)
///
/// RSI is a momentum oscillator that measures the speed and magnitude
/// of price changes. Values range from 0 to 100.
///
/// # Arguments
/// * `prices` - Price series (typically close prices)
/// * `period` - Period for RSI calculation (default: 14)
///
/// # Returns
/// Vector of RSI values (0-100)
#[allow(dead_code)]
pub fn rsi(prices: &[f64], period: usize) -> Result<Vec<f64>, IndicatorError> {
    if prices.len() < period + 1 {
        return Err(IndicatorError::InsufficientData {
            required: period + 1,
            available: prices.len(),
        });
    }

    let mut result = vec![f64::NAN; prices.len()];

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

    // Calculate average gains and losses
    let avg_gains = ema(&gains, period)?;
    let avg_losses = ema(&losses, period)?;

    // Calculate RSI
    for i in period..prices.len() {
        if avg_losses[i] == 0.0 {
            result[i] = 100.0;
        } else {
            let rs = avg_gains[i] / avg_losses[i];
            result[i] = 100.0 - (100.0 / (1.0 + rs));
        }
    }

    Ok(result)
}

/// Calculate MACD (Moving Average Convergence Divergence)
///
/// # Arguments
/// * `prices` - Price series
/// * `fast_period` - Fast EMA period (default 12)
/// * `slow_period` - Slow EMA period (default 26)
/// * `signal_period` - Signal line period (default 9)
///
/// # Returns
/// Type alias for MACD result tuple (MACD line, Signal line, Histogram)
type MacdResult = Result<(Vec<f64>, Vec<f64>, Vec<f64>), IndicatorError>;

/// Calculate MACD (Moving Average Convergence Divergence)
/// Returns tuple of (MACD line, Signal line, Histogram)
#[allow(dead_code)]
pub fn macd(
    prices: &[f64],
    fast_period: usize,
    slow_period: usize,
    signal_period: usize,
) -> MacdResult {
    let fast_ema = ema(prices, fast_period)?;
    let slow_ema = ema(prices, slow_period)?;

    // Calculate MACD line
    let mut macd_line = vec![f64::NAN; prices.len()];
    for i in 0..prices.len() {
        if !fast_ema[i].is_nan() && !slow_ema[i].is_nan() {
            macd_line[i] = fast_ema[i] - slow_ema[i];
        }
    }

    // Calculate signal line
    let signal_line = ema(&macd_line, signal_period)?;

    // Calculate histogram
    let mut histogram = vec![f64::NAN; prices.len()];
    for i in 0..prices.len() {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    Ok((macd_line, signal_line, histogram))
}

// ============================================================================
// Incremental Indicator Structures
// ============================================================================

/// Incremental EMA calculator
pub struct EMA {
    period: usize,
    alpha: f64,
    value: f64,
    initialized: bool,
    warmup_prices: VecDeque<f64>,
}

impl EMA {
    /// Create new EMA calculator
    pub fn new(period: usize) -> Self {
        let alpha = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            alpha,
            value: 0.0,
            initialized: false,
            warmup_prices: VecDeque::with_capacity(period),
        }
    }

    /// Update with new price
    pub fn update(&mut self, price: f64) {
        if !self.initialized {
            self.warmup_prices.push_back(price);
            if self.warmup_prices.len() >= self.period {
                // Initialize with SMA
                let sum: f64 = self.warmup_prices.iter().sum();
                self.value = sum / self.period as f64;
                self.initialized = true;
                self.warmup_prices.clear();
            }
        } else {
            // Update EMA
            self.value = price * self.alpha + self.value * (1.0 - self.alpha);
        }
    }

    /// Get current EMA value
    pub fn value(&self) -> f64 {
        if self.initialized {
            self.value
        } else {
            f64::NAN
        }
    }

    /// Check if EMA is ready
    pub fn is_ready(&self) -> bool {
        self.initialized
    }
}

/// Incremental ATR calculator
pub struct ATR {
    #[allow(dead_code)]
    period: usize,
    ema: EMA,
    prev_close: Option<f64>,
}

impl ATR {
    /// Create new ATR calculator
    pub fn new(period: usize) -> Self {
        Self {
            period,
            ema: EMA::new(period),
            prev_close: None,
        }
    }

    /// Update with new high, low, close
    pub fn update(&mut self, high: f64, low: f64, close: f64) {
        let tr = if let Some(prev_close) = self.prev_close {
            let hl = high - low;
            let hc = (high - prev_close).abs();
            let lc = (low - prev_close).abs();
            hl.max(hc).max(lc)
        } else {
            high - low
        };

        self.ema.update(tr);
        self.prev_close = Some(close);
    }

    /// Get current ATR value
    pub fn value(&self) -> f64 {
        self.ema.value()
    }

    /// Check if ATR is ready
    pub fn is_ready(&self) -> bool {
        self.ema.is_ready()
    }
}

/// Indicator calculation helper
pub struct IndicatorCalculator {
    pub fast_ema_period: usize,
    pub slow_ema_period: usize,
    pub atr_period: usize,
    fast_ema: EMA,
    slow_ema: EMA,
    atr_calculator: ATR,
    tick_count: usize,
}

impl Default for IndicatorCalculator {
    fn default() -> Self {
        Self {
            fast_ema_period: 8,
            slow_ema_period: 21,
            atr_period: 14,
            fast_ema: EMA::new(8),
            slow_ema: EMA::new(21),
            atr_calculator: ATR::new(14),
            tick_count: 0,
        }
    }
}

impl IndicatorCalculator {
    /// Create a new indicator calculator with custom periods
    pub fn new(fast_ema: usize, slow_ema: usize, atr_period: usize) -> Self {
        Self {
            fast_ema_period: fast_ema,
            slow_ema_period: slow_ema,
            atr_period,
            fast_ema: EMA::new(fast_ema),
            slow_ema: EMA::new(slow_ema),
            atr_calculator: ATR::new(atr_period),
            tick_count: 0,
        }
    }

    /// Update indicators with a new price (incremental)
    pub fn update(&mut self, price: f64) {
        self.fast_ema.update(price);
        self.slow_ema.update(price);
        self.tick_count += 1;
    }

    /// Update ATR with high, low, close
    pub fn update_with_hlc(&mut self, high: f64, low: f64, close: f64) {
        self.fast_ema.update(close);
        self.slow_ema.update(close);
        self.atr_calculator.update(high, low, close);
        self.tick_count += 1;
    }

    /// Get current fast EMA value
    pub fn ema8(&self) -> f64 {
        self.fast_ema.value()
    }

    /// Get current slow EMA value
    pub fn ema21(&self) -> f64 {
        self.slow_ema.value()
    }

    /// Get current ATR value
    pub fn atr(&self) -> f64 {
        self.atr_calculator.value()
    }

    /// Check if indicators are ready (warmed up)
    pub fn is_ready(&self) -> bool {
        self.tick_count >= self.slow_ema_period
    }

    /// Reset all indicators
    pub fn reset(&mut self) {
        self.fast_ema = EMA::new(self.fast_ema_period);
        self.slow_ema = EMA::new(self.slow_ema_period);
        self.atr_calculator = ATR::new(self.atr_period);
        self.tick_count = 0;
    }

    /// Calculate all indicators for the strategy
    pub fn calculate_all(
        &self,
        close: &[f64],
        high: &[f64],
        low: &[f64],
    ) -> Result<StrategyIndicators, IndicatorError> {
        let ema_fast = ema(close, self.fast_ema_period)?;
        let ema_slow = ema(close, self.slow_ema_period)?;
        let atr_values = atr(high, low, close, self.atr_period)?;

        Ok(StrategyIndicators {
            ema_fast,
            ema_slow,
            atr: atr_values,
        })
    }
}

/// Strategy indicators result
#[derive(Debug, Clone)]
pub struct StrategyIndicators {
    pub ema_fast: Vec<f64>,
    pub ema_slow: Vec<f64>,
    pub atr: Vec<f64>,
}

// Incremental indicators for live streaming data

/// Incremental EMA - O(1) update per tick
pub struct IncrementalEma {
    alpha: f64,
    state: f64,
    initialized: bool,
}

impl IncrementalEma {
    pub fn new(period: usize) -> Self {
        Self {
            alpha: 2.0 / (period as f64 + 1.0),
            state: 0.0,
            initialized: false,
        }
    }

    pub fn update(&mut self, price: f64) -> f64 {
        if !self.initialized {
            self.state = price;
            self.initialized = true;
        } else {
            self.state = self.alpha * price + (1.0 - self.alpha) * self.state;
        }
        self.state
    }

    pub fn current(&self) -> Option<f64> {
        if self.initialized {
            Some(self.state)
        } else {
            None
        }
    }
}

/// Incremental ATR calculator
pub struct IncrementalAtr {
    ema: IncrementalEma,
    prev_close: Option<f64>,
}

impl IncrementalAtr {
    pub fn new(period: usize) -> Self {
        Self {
            ema: IncrementalEma::new(period),
            prev_close: None,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let tr = if let Some(prev) = self.prev_close {
            let tr1 = high - low;
            let tr2 = (high - prev).abs();
            let tr3 = (low - prev).abs();
            tr1.max(tr2).max(tr3)
        } else {
            high - low
        };

        self.prev_close = Some(close);
        Some(self.ema.update(tr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sma() {
        let prices = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sma(&prices, 3).unwrap();

        assert!(result[0].is_nan());
        assert!(result[1].is_nan());
        assert_eq!(result[2], 2.0); // (1+2+3)/3
        assert_eq!(result[3], 3.0); // (2+3+4)/3
        assert_eq!(result[4], 4.0); // (3+4+5)/3
    }

    #[test]
    fn test_ema() {
        let prices = vec![
            22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29,
        ];
        let result = ema(&prices, 5).unwrap();

        // First EMA should be SMA
        let first_sma = (22.27 + 22.19 + 22.08 + 22.17 + 22.18) / 5.0;
        assert!((result[4] - first_sma).abs() < 0.001);

        // Subsequent values should be calculated via EMA
        assert!(!result[9].is_nan());
    }

    #[test]
    fn test_true_range() {
        let high = vec![50.0, 52.0, 51.0];
        let low = vec![48.0, 49.0, 48.5];
        let close = vec![49.0, 51.0, 50.0];

        let tr = true_range(&high, &low, &close).unwrap();

        assert_eq!(tr[0], 2.0); // 50 - 48
        assert_eq!(tr[1], 3.0); // max(52-49, |52-49|, |49-49|) = 3
    }

    #[test]
    fn test_atr() {
        let high = vec![50.0, 52.0, 51.0, 53.0, 52.0];
        let low = vec![48.0, 49.0, 48.5, 50.0, 49.0];
        let close = vec![49.0, 51.0, 50.0, 52.0, 51.0];

        let result = atr(&high, &low, &close, 3).unwrap();

        // Should have same length as input
        assert_eq!(result.len(), 5);

        // Later values should not be NaN
        assert!(!result[4].is_nan());
    }

    #[test]
    fn test_insufficient_data() {
        let prices = vec![1.0, 2.0];
        let result = ema(&prices, 5);

        assert!(result.is_err());
    }

    #[test]
    fn test_indicator_calculator() {
        let calc = IndicatorCalculator::default();

        let close = vec![100.0; 50];
        let high = vec![101.0; 50];
        let low = vec![99.0; 50];

        let result = calc.calculate_all(&close, &high, &low).unwrap();

        assert_eq!(result.ema_fast.len(), 50);
        assert_eq!(result.ema_slow.len(), 50);
        assert_eq!(result.atr.len(), 50);
    }
}
