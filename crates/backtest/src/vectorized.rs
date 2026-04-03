//! # Vectorized Indicator Engine
//!
//! High-performance batch indicator calculations using Polars.
//!
//! This module provides vectorized implementations of technical indicators
//! for backtesting scenarios where we process large historical datasets.
//! Unlike the incremental indicators used in live trading, these operate
//! on entire DataFrames at once for maximum performance.
//!
//! ## Performance
//!
//! Vectorized calculations are 10-100x faster than incremental updates
//! when processing historical data, thanks to:
//! - SIMD operations
//! - Cache-friendly memory access patterns
//! - Polars' optimized lazy evaluation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use polars::prelude::*;
//! use janus_backtest::vectorized::VectorizedIndicators;
//!
//! let df = ... // Load tick data
//! let indicators = VectorizedIndicators::new(df);
//! let result = indicators
//!     .add_ema("price", 8, "ema_8")
//!     .add_ema("price", 21, "ema_21")
//!     .add_atr("price", 14, "atr_14")
//!     .compute()?;
//! ```

use polars::prelude::*;
use thiserror::Error;

/// Errors that can occur during vectorized calculations
#[derive(Error, Debug)]
pub enum VectorizedError {
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),

    #[error("Missing column: {0}")]
    MissingColumn(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Builder for vectorized indicator calculations
pub struct VectorizedIndicators {
    df: DataFrame,
    operations: Vec<IndicatorOp>,
}

/// An indicator operation to be applied
#[derive(Clone)]
#[allow(clippy::upper_case_acronyms)]
enum IndicatorOp {
    EMA {
        column: String,
        period: usize,
        output: String,
    },
    SMA {
        column: String,
        period: usize,
        output: String,
    },
    ATR {
        high_col: String,
        low_col: String,
        close_col: String,
        period: usize,
        output: String,
    },
    RSI {
        column: String,
        period: usize,
        output: String,
    },
    MACD {
        column: String,
        fast: usize,
        slow: usize,
        signal: usize,
        output_macd: String,
        output_signal: String,
        output_histogram: String,
    },
    BollingerBands {
        column: String,
        period: usize,
        std_dev: f64,
        output_upper: String,
        output_middle: String,
        output_lower: String,
    },
}

impl VectorizedIndicators {
    /// Create a new vectorized indicator builder
    pub fn new(df: DataFrame) -> Self {
        Self {
            df,
            operations: Vec::new(),
        }
    }

    /// Add Exponential Moving Average
    pub fn add_ema(mut self, column: &str, period: usize, output: &str) -> Self {
        self.operations.push(IndicatorOp::EMA {
            column: column.to_string(),
            period,
            output: output.to_string(),
        });
        self
    }

    /// Add Simple Moving Average
    pub fn add_sma(mut self, column: &str, period: usize, output: &str) -> Self {
        self.operations.push(IndicatorOp::SMA {
            column: column.to_string(),
            period,
            output: output.to_string(),
        });
        self
    }

    /// Add Average True Range
    pub fn add_atr(
        mut self,
        high_col: &str,
        low_col: &str,
        close_col: &str,
        period: usize,
        output: &str,
    ) -> Self {
        self.operations.push(IndicatorOp::ATR {
            high_col: high_col.to_string(),
            low_col: low_col.to_string(),
            close_col: close_col.to_string(),
            period,
            output: output.to_string(),
        });
        self
    }

    /// Add Relative Strength Index
    pub fn add_rsi(mut self, column: &str, period: usize, output: &str) -> Self {
        self.operations.push(IndicatorOp::RSI {
            column: column.to_string(),
            period,
            output: output.to_string(),
        });
        self
    }

    /// Add MACD (Moving Average Convergence Divergence)
    #[allow(clippy::too_many_arguments)]
    pub fn add_macd(
        mut self,
        column: &str,
        fast: usize,
        slow: usize,
        signal: usize,
        output_macd: &str,
        output_signal: &str,
        output_histogram: &str,
    ) -> Self {
        self.operations.push(IndicatorOp::MACD {
            column: column.to_string(),
            fast,
            slow,
            signal,
            output_macd: output_macd.to_string(),
            output_signal: output_signal.to_string(),
            output_histogram: output_histogram.to_string(),
        });
        self
    }

    /// Add Bollinger Bands
    pub fn add_bollinger_bands(
        mut self,
        column: &str,
        period: usize,
        std_dev: f64,
        output_upper: &str,
        output_middle: &str,
        output_lower: &str,
    ) -> Self {
        self.operations.push(IndicatorOp::BollingerBands {
            column: column.to_string(),
            period,
            std_dev,
            output_upper: output_upper.to_string(),
            output_middle: output_middle.to_string(),
            output_lower: output_lower.to_string(),
        });
        self
    }

    /// Compute all indicators and return the enriched DataFrame
    pub fn compute(mut self) -> Result<DataFrame, VectorizedError> {
        for op in &self.operations {
            self.df = self.apply_operation(op.clone())?;
        }
        Ok(self.df)
    }

    fn apply_operation(&self, op: IndicatorOp) -> Result<DataFrame, VectorizedError> {
        match op {
            IndicatorOp::EMA {
                column,
                period,
                output,
            } => self.compute_ema(&column, period, &output),
            IndicatorOp::SMA {
                column,
                period,
                output,
            } => self.compute_sma(&column, period, &output),
            IndicatorOp::ATR {
                high_col,
                low_col,
                close_col,
                period,
                output,
            } => self.compute_atr(&high_col, &low_col, &close_col, period, &output),
            IndicatorOp::RSI {
                column,
                period,
                output,
            } => self.compute_rsi(&column, period, &output),
            IndicatorOp::MACD {
                column,
                fast,
                slow,
                signal,
                output_macd,
                output_signal,
                output_histogram,
            } => self.compute_macd(
                &column,
                fast,
                slow,
                signal,
                &output_macd,
                &output_signal,
                &output_histogram,
            ),
            IndicatorOp::BollingerBands {
                column,
                period,
                std_dev,
                output_upper,
                output_middle,
                output_lower,
            } => self.compute_bollinger_bands(
                &column,
                period,
                std_dev,
                &output_upper,
                &output_middle,
                &output_lower,
            ),
        }
    }

    fn compute_ema(
        &self,
        column: &str,
        period: usize,
        output: &str,
    ) -> Result<DataFrame, VectorizedError> {
        if period == 0 {
            return Err(VectorizedError::InvalidParameter(
                "EMA period must be > 0".to_string(),
            ));
        }

        let series = self
            .df
            .column(column)
            .map_err(|_| VectorizedError::MissingColumn(column.to_string()))?;

        // Calculate EMA using exponential weighted moving average
        let alpha = 2.0 / (period as f64 + 1.0);
        let ema = series
            .cast(&DataType::Float64)?
            .f64()?
            .into_iter()
            .scan(None, |state: &mut Option<f64>, val| {
                if let Some(v) = val {
                    match state {
                        None => {
                            *state = Some(v);
                            Some(Some(v))
                        }
                        Some(prev) => {
                            let new_ema = alpha * v + (1.0 - alpha) * *prev;
                            *state = Some(new_ema);
                            Some(Some(new_ema))
                        }
                    }
                } else {
                    Some(None)
                }
            })
            .collect::<Float64Chunked>();

        let ema_series = ema.into_series().with_name(output.into());
        let mut result = self.df.clone();
        result.with_column(ema_series)?;
        Ok(result)
    }

    fn compute_sma(
        &self,
        column: &str,
        period: usize,
        output: &str,
    ) -> Result<DataFrame, VectorizedError> {
        if period == 0 {
            return Err(VectorizedError::InvalidParameter(
                "SMA period must be > 0".to_string(),
            ));
        }

        let series = self
            .df
            .column(column)
            .map_err(|_| VectorizedError::MissingColumn(column.to_string()))?;

        // Use simple rolling window for SMA
        let series_f64 = series.cast(&DataType::Float64)?;
        let prices = series_f64.f64()?;
        let mut sma_values = Vec::with_capacity(prices.len());

        for i in 0..prices.len() {
            if i + 1 < period {
                sma_values.push(None);
            } else {
                let sum: f64 = (i + 1 - period..=i).filter_map(|idx| prices.get(idx)).sum();
                sma_values.push(Some(sum / period as f64));
            }
        }

        let sma = Float64Chunked::from_slice_options("sma".into(), &sma_values);

        let sma_series = sma.into_series().with_name(output.into());
        let mut result = self.df.clone();
        result.with_column(sma_series)?;
        Ok(result)
    }

    fn compute_atr(
        &self,
        high_col: &str,
        low_col: &str,
        close_col: &str,
        period: usize,
        output: &str,
    ) -> Result<DataFrame, VectorizedError> {
        if period == 0 {
            return Err(VectorizedError::InvalidParameter(
                "ATR period must be > 0".to_string(),
            ));
        }

        let high = self
            .df
            .column(high_col)
            .map_err(|_| VectorizedError::MissingColumn(high_col.to_string()))?
            .cast(&DataType::Float64)?;

        let low = self
            .df
            .column(low_col)
            .map_err(|_| VectorizedError::MissingColumn(low_col.to_string()))?
            .cast(&DataType::Float64)?;

        let close = self
            .df
            .column(close_col)
            .map_err(|_| VectorizedError::MissingColumn(close_col.to_string()))?
            .cast(&DataType::Float64)?;

        // Calculate True Range manually
        let high_values = high.f64()?;
        let low_values = low.f64()?;
        let close_values = close.f64()?;

        let mut tr_values = Vec::with_capacity(high_values.len());

        for i in 0..high_values.len() {
            if i == 0 {
                // First value: just high - low
                match (high_values.get(i), low_values.get(i)) {
                    (Some(h), Some(l)) => tr_values.push(Some(h - l)),
                    _ => tr_values.push(None),
                }
            } else {
                match (
                    high_values.get(i),
                    low_values.get(i),
                    close_values.get(i - 1),
                ) {
                    (Some(h), Some(l), Some(prev_close)) => {
                        let hl = h - l;
                        let hc = (h - prev_close).abs();
                        let lc = (l - prev_close).abs();
                        tr_values.push(Some(hl.max(hc).max(lc)));
                    }
                    _ => tr_values.push(None),
                }
            }
        }

        let tr = Float64Chunked::from_slice_options("tr".into(), &tr_values);

        // Calculate ATR as EMA of TR
        let alpha = 1.0 / period as f64;
        let atr = tr
            .into_iter()
            .scan(None, |state: &mut Option<f64>, val| {
                if let Some(v) = val {
                    match state {
                        None => {
                            *state = Some(v);
                            Some(Some(v))
                        }
                        Some(prev) => {
                            let new_atr = alpha * v + (1.0 - alpha) * *prev;
                            *state = Some(new_atr);
                            Some(Some(new_atr))
                        }
                    }
                } else {
                    Some(None)
                }
            })
            .collect::<Float64Chunked>()
            .with_name("tr".into());

        let atr_series = atr.into_series().with_name(output.into());
        let mut result = self.df.clone();
        result.with_column(atr_series)?;
        Ok(result)
    }

    fn compute_rsi(
        &self,
        column: &str,
        period: usize,
        output: &str,
    ) -> Result<DataFrame, VectorizedError> {
        if period == 0 {
            return Err(VectorizedError::InvalidParameter(
                "RSI period must be > 0".to_string(),
            ));
        }

        let series = self
            .df
            .column(column)
            .map_err(|_| VectorizedError::MissingColumn(column.to_string()))?
            .cast(&DataType::Float64)?;

        let prices = series.f64()?;

        // Calculate price changes
        let changes: Vec<Option<f64>> = prices
            .into_iter()
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| match (w[0], w[1]) {
                (Some(prev), Some(curr)) => Some(curr - prev),
                _ => None,
            })
            .collect();

        // Prepend None to align with original series
        let mut aligned_changes = vec![None];
        aligned_changes.extend(changes);

        // Separate gains and losses
        let gains: Vec<Option<f64>> = aligned_changes
            .iter()
            .map(|c| c.map(|v| if v > 0.0 { v } else { 0.0 }))
            .collect();

        let losses: Vec<Option<f64>> = aligned_changes
            .iter()
            .map(|c| c.map(|v| if v < 0.0 { -v } else { 0.0 }))
            .collect();

        // Calculate EMA of gains and losses
        let alpha = 1.0 / period as f64;

        let avg_gains: Vec<Option<f64>> = gains
            .iter()
            .scan(None, |state: &mut Option<f64>, val| {
                if let Some(v) = val {
                    match state {
                        None => {
                            *state = Some(*v);
                            Some(Some(*v))
                        }
                        Some(prev) => {
                            let new_avg = alpha * v + (1.0 - alpha) * *prev;
                            *state = Some(new_avg);
                            Some(Some(new_avg))
                        }
                    }
                } else {
                    Some(None)
                }
            })
            .collect();

        let avg_losses: Vec<Option<f64>> = losses
            .iter()
            .scan(None, |state: &mut Option<f64>, val| {
                if let Some(v) = val {
                    match state {
                        None => {
                            *state = Some(*v);
                            Some(Some(*v))
                        }
                        Some(prev) => {
                            let new_avg = alpha * v + (1.0 - alpha) * *prev;
                            *state = Some(new_avg);
                            Some(Some(new_avg))
                        }
                    }
                } else {
                    Some(None)
                }
            })
            .collect();

        // Calculate RSI = 100 - (100 / (1 + RS))
        let rsi: Vec<Option<f64>> = avg_gains
            .iter()
            .zip(avg_losses.iter())
            .map(|(gain, loss)| match (gain, loss) {
                (Some(g), Some(l)) if *l > 0.0 => {
                    let rs = g / l;
                    Some(100.0 - (100.0 / (1.0 + rs)))
                }
                (Some(_), Some(l)) if *l == 0.0 => Some(100.0),
                _ => None,
            })
            .collect();

        let rsi_series = Float64Chunked::from_slice_options(output.into(), &rsi).into_series();
        let mut result = self.df.clone();
        result.with_column(rsi_series)?;
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn compute_macd(
        &self,
        column: &str,
        fast: usize,
        slow: usize,
        signal_period: usize,
        output_macd: &str,
        output_signal: &str,
        output_histogram: &str,
    ) -> Result<DataFrame, VectorizedError> {
        // Calculate fast EMA
        let mut temp_df = self.compute_ema(column, fast, "_temp_fast_ema")?;
        // Calculate slow EMA
        temp_df = VectorizedIndicators::new(temp_df)
            .add_ema(column, slow, "_temp_slow_ema")
            .compute()?;

        // MACD = fast EMA - slow EMA
        let fast_ema = temp_df.column("_temp_fast_ema")?;
        let slow_ema = temp_df.column("_temp_slow_ema")?;

        // Compute MACD manually to avoid Result chaining issues
        let fast_values = fast_ema.f64()?;
        let slow_values = slow_ema.f64()?;

        let macd_values: Vec<Option<f64>> = fast_values
            .into_iter()
            .zip(slow_values)
            .map(|(f, s)| match (f, s) {
                (Some(f), Some(s)) => Some(f - s),
                _ => None,
            })
            .collect();

        let macd_series =
            Float64Chunked::from_slice_options(output_macd.into(), &macd_values).into_series();
        temp_df.with_column(macd_series)?;

        // Calculate signal line (EMA of MACD)
        temp_df = VectorizedIndicators::new(temp_df)
            .add_ema(output_macd, signal_period, output_signal)
            .compute()?;

        // Histogram = MACD - Signal
        let macd_col = temp_df.column(output_macd)?;
        let signal_col = temp_df.column(output_signal)?;

        // Compute histogram manually
        let macd_values = macd_col.f64()?;
        let signal_values = signal_col.f64()?;

        let histogram_values: Vec<Option<f64>> = macd_values
            .into_iter()
            .zip(signal_values)
            .map(|(m, s)| match (m, s) {
                (Some(m), Some(s)) => Some(m - s),
                _ => None,
            })
            .collect();

        let histogram_series =
            Float64Chunked::from_slice_options(output_histogram.into(), &histogram_values)
                .into_series();
        temp_df.with_column(histogram_series)?;

        // Drop temporary columns
        let result = temp_df.drop_many(["_temp_fast_ema", "_temp_slow_ema"]);

        Ok(result)
    }

    fn compute_bollinger_bands(
        &self,
        column: &str,
        period: usize,
        std_dev: f64,
        output_upper: &str,
        output_middle: &str,
        output_lower: &str,
    ) -> Result<DataFrame, VectorizedError> {
        if period == 0 {
            return Err(VectorizedError::InvalidParameter(
                "Bollinger period must be > 0".to_string(),
            ));
        }

        let series = self
            .df
            .column(column)
            .map_err(|_| VectorizedError::MissingColumn(column.to_string()))?
            .cast(&DataType::Float64)?;

        // Calculate middle band (SMA)
        // Use simple rolling window for SMA
        let prices = series.f64()?;
        let mut sma_values = Vec::with_capacity(prices.len());

        for i in 0..prices.len() {
            if i + 1 < period {
                sma_values.push(None);
            } else {
                let sum: f64 = (i + 1 - period..=i).filter_map(|idx| prices.get(idx)).sum();
                sma_values.push(Some(sum / period as f64));
            }
        }

        // Calculate standard deviation
        let mut std_values = Vec::with_capacity(prices.len());

        for i in 0..prices.len() {
            if i + 1 < period {
                std_values.push(None);
            } else {
                let window: Vec<f64> = (i + 1 - period..=i)
                    .filter_map(|idx| prices.get(idx))
                    .collect();

                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let variance =
                    window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
                std_values.push(Some(variance.sqrt()));
            }
        }

        // Calculate upper and lower bands
        let mut upper_values = Vec::with_capacity(sma_values.len());
        let mut lower_values = Vec::with_capacity(sma_values.len());

        for i in 0..sma_values.len() {
            match (sma_values[i], std_values[i]) {
                (Some(sma), Some(std)) => {
                    upper_values.push(Some(sma + std_dev * std));
                    lower_values.push(Some(sma - std_dev * std));
                }
                _ => {
                    upper_values.push(None);
                    lower_values.push(None);
                }
            }
        }

        let sma_series =
            Float64Chunked::from_slice_options(output_middle.into(), &sma_values).into_series();
        let upper_series =
            Float64Chunked::from_slice_options(output_upper.into(), &upper_values).into_series();
        let lower_series =
            Float64Chunked::from_slice_options(output_lower.into(), &lower_values).into_series();

        let mut result = self.df.clone();
        result.with_column(sma_series)?;
        result.with_column(upper_series)?;
        result.with_column(lower_series)?;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataframe() -> DataFrame {
        let prices = vec![
            100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.5, 105.0, 106.0, 105.5, 107.0, 108.0,
            107.5, 109.0, 110.0,
        ];

        df! {
            "price" => prices.clone(),
            "high" => prices.iter().map(|p| p + 0.5).collect::<Vec<_>>(),
            "low" => prices.iter().map(|p| p - 0.5).collect::<Vec<_>>(),
            "close" => prices,
        }
        .unwrap()
    }

    #[test]
    fn test_ema_calculation() {
        let df = create_test_dataframe();
        let result = VectorizedIndicators::new(df)
            .add_ema("price", 5, "ema_5")
            .compute()
            .unwrap();

        assert!(result.column("ema_5").is_ok());
        let ema = result.column("ema_5").unwrap();
        assert_eq!(ema.len(), 15);
    }

    #[test]
    fn test_sma_calculation() {
        let df = create_test_dataframe();
        let result = VectorizedIndicators::new(df)
            .add_sma("price", 3, "sma_3")
            .compute()
            .unwrap();

        assert!(result.column("sma_3").is_ok());
    }

    #[test]
    fn test_atr_calculation() {
        let df = create_test_dataframe();
        let result = VectorizedIndicators::new(df)
            .add_atr("high", "low", "close", 14, "atr_14")
            .compute()
            .unwrap();

        assert!(result.column("atr_14").is_ok());
    }

    #[test]
    fn test_rsi_calculation() {
        let df = create_test_dataframe();
        let result = VectorizedIndicators::new(df)
            .add_rsi("price", 14, "rsi_14")
            .compute()
            .unwrap();

        assert!(result.column("rsi_14").is_ok());
        let rsi = result.column("rsi_14").unwrap();
        // RSI should be between 0 and 100
        if let Ok(rsi_f64) = rsi.f64() {
            for val in rsi_f64.into_iter().flatten() {
                assert!((0.0..=100.0).contains(&val));
            }
        }
    }

    #[test]
    fn test_macd_calculation() {
        let df = create_test_dataframe();
        let result = VectorizedIndicators::new(df)
            .add_macd("price", 3, 5, 2, "macd", "signal", "histogram")
            .compute()
            .unwrap();

        assert!(result.column("macd").is_ok());
        assert!(result.column("signal").is_ok());
        assert!(result.column("histogram").is_ok());
    }

    #[test]
    fn test_bollinger_bands() {
        let df = create_test_dataframe();
        let result = VectorizedIndicators::new(df)
            .add_bollinger_bands("price", 5, 2.0, "bb_upper", "bb_middle", "bb_lower")
            .compute()
            .unwrap();

        assert!(result.column("bb_upper").is_ok());
        assert!(result.column("bb_middle").is_ok());
        assert!(result.column("bb_lower").is_ok());

        // Upper should be > middle > lower
        let upper = result.column("bb_upper").unwrap().f64().unwrap();
        let middle = result.column("bb_middle").unwrap().f64().unwrap();
        let lower = result.column("bb_lower").unwrap().f64().unwrap();

        for i in 0..upper.len() {
            if let (Some(u), Some(m), Some(l)) = (upper.get(i), middle.get(i), lower.get(i)) {
                assert!(u >= m);
                assert!(m >= l);
            }
        }
    }

    #[test]
    fn test_chained_indicators() {
        let df = create_test_dataframe();
        let result = VectorizedIndicators::new(df)
            .add_ema("price", 8, "ema_8")
            .add_ema("price", 21, "ema_21")
            .add_atr("high", "low", "close", 14, "atr_14")
            .add_rsi("price", 14, "rsi_14")
            .compute()
            .unwrap();

        assert!(result.column("ema_8").is_ok());
        assert!(result.column("ema_21").is_ok());
        assert!(result.column("atr_14").is_ok());
        assert!(result.column("rsi_14").is_ok());
    }
}
