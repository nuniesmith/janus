//! Volatility-adjusted sizing
//!
//! Part of the Hypothalamus region
//! Component: position_sizing
//!
//! This module implements volatility-based position sizing that adjusts
//! position sizes inversely to volatility, maintaining consistent risk
//! exposure across different market conditions.

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Method for measuring volatility
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityMeasure {
    /// Average True Range
    ATR,
    /// Standard deviation of returns
    StdDev,
    /// Exponentially weighted moving average of squared returns
    EWMA,
    /// Parkinson's volatility (high-low range)
    Parkinson,
    /// Yang-Zhang volatility (comprehensive)
    YangZhang,
    /// Implied volatility from options
    Implied,
}

impl Default for VolatilityMeasure {
    fn default() -> Self {
        VolatilityMeasure::ATR
    }
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    /// Extremely low volatility
    VeryLow,
    /// Below average volatility
    Low,
    /// Normal/average volatility
    Normal,
    /// Above average volatility
    High,
    /// Extremely high volatility (crisis)
    VeryHigh,
    /// Volatility spike detected
    Spike,
}

impl Default for VolatilityRegime {
    fn default() -> Self {
        VolatilityRegime::Normal
    }
}

impl VolatilityRegime {
    /// Get a scaling factor for this regime
    pub fn scaling_factor(&self) -> f64 {
        match self {
            VolatilityRegime::VeryLow => 1.2,  // Slight increase in size
            VolatilityRegime::Low => 1.1,      // Modest increase
            VolatilityRegime::Normal => 1.0,   // Base case
            VolatilityRegime::High => 0.75,    // Reduce size
            VolatilityRegime::VeryHigh => 0.5, // Significant reduction
            VolatilityRegime::Spike => 0.25,   // Minimal size during spikes
        }
    }
}

/// Configuration for volatility scaling
#[derive(Debug, Clone)]
pub struct VolatilityScalingConfig {
    /// Volatility measurement method
    pub measure: VolatilityMeasure,
    /// Target portfolio volatility (annualized, e.g., 0.15 = 15%)
    pub target_volatility: f64,
    /// Lookback period for volatility calculation (days)
    pub lookback_period: usize,
    /// Minimum scaling factor (floor)
    pub min_scale: f64,
    /// Maximum scaling factor (ceiling)
    pub max_scale: f64,
    /// Smoothing factor for EWMA (0-1)
    pub ewma_lambda: f64,
    /// Enable regime-based adjustments
    pub use_regime_adjustment: bool,
    /// Percentile thresholds for regime classification [very_low, low, high, very_high]
    pub regime_thresholds: [f64; 4],
    /// Days for annualization (trading days)
    pub trading_days_per_year: f64,
    /// Enable volatility spike detection
    pub detect_spikes: bool,
    /// Spike threshold (multiple of average volatility)
    pub spike_threshold: f64,
}

impl Default for VolatilityScalingConfig {
    fn default() -> Self {
        Self {
            measure: VolatilityMeasure::ATR,
            target_volatility: 0.15, // 15% annualized
            lookback_period: 20,
            min_scale: 0.1,
            max_scale: 3.0,
            ewma_lambda: 0.94, // RiskMetrics standard
            use_regime_adjustment: true,
            regime_thresholds: [0.10, 0.25, 0.75, 0.90],
            trading_days_per_year: 252.0,
            detect_spikes: true,
            spike_threshold: 2.5,
        }
    }
}

/// OHLCV price bar for volatility calculation
#[derive(Debug, Clone)]
pub struct PriceBar {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub timestamp: u64,
}

impl PriceBar {
    pub fn new(open: f64, high: f64, low: f64, close: f64, volume: f64, timestamp: u64) -> Self {
        Self {
            open,
            high,
            low,
            close,
            volume,
            timestamp,
        }
    }

    /// Calculate true range given previous close
    pub fn true_range(&self, prev_close: f64) -> f64 {
        let high_low = self.high - self.low;
        let high_prev_close = (self.high - prev_close).abs();
        let low_prev_close = (self.low - prev_close).abs();
        high_low.max(high_prev_close).max(low_prev_close)
    }

    /// Calculate return from previous close
    pub fn return_pct(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 {
            (self.close - prev_close) / prev_close
        } else {
            0.0
        }
    }

    /// Calculate log return from previous close
    pub fn log_return(&self, prev_close: f64) -> f64 {
        if prev_close > 0.0 && self.close > 0.0 {
            (self.close / prev_close).ln()
        } else {
            0.0
        }
    }
}

/// Volatility calculation result
#[derive(Debug, Clone)]
pub struct VolatilityResult {
    /// Current volatility estimate (daily)
    pub daily_volatility: f64,
    /// Annualized volatility
    pub annualized_volatility: f64,
    /// Volatility regime
    pub regime: VolatilityRegime,
    /// Scaling factor to apply
    pub scale_factor: f64,
    /// Whether a spike was detected
    pub spike_detected: bool,
    /// Historical volatility percentile (0-1)
    pub percentile: f64,
    /// Average volatility over lookback
    pub average_volatility: f64,
    /// Volatility of volatility (vol clustering indicator)
    pub vol_of_vol: f64,
}

impl Default for VolatilityResult {
    fn default() -> Self {
        Self {
            daily_volatility: 0.0,
            annualized_volatility: 0.0,
            regime: VolatilityRegime::Normal,
            scale_factor: 1.0,
            spike_detected: false,
            percentile: 0.5,
            average_volatility: 0.0,
            vol_of_vol: 0.0,
        }
    }
}

/// Symbol volatility data
#[derive(Debug, Clone)]
pub struct SymbolVolatility {
    /// Symbol identifier
    pub symbol: String,
    /// Price history
    pub price_history: Vec<PriceBar>,
    /// Calculated ATR values
    pub atr_history: Vec<f64>,
    /// Return history
    pub return_history: Vec<f64>,
    /// Current ATR
    pub current_atr: f64,
    /// Current volatility estimate
    pub current_volatility: f64,
    /// EWMA variance estimate
    pub ewma_variance: f64,
    /// Last calculation result
    pub last_result: Option<VolatilityResult>,
}

impl SymbolVolatility {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            price_history: Vec::new(),
            atr_history: Vec::new(),
            return_history: Vec::new(),
            current_atr: 0.0,
            current_volatility: 0.0,
            ewma_variance: 0.0,
            last_result: None,
        }
    }
}

/// Position sizing recommendation
#[derive(Debug, Clone)]
pub struct PositionSizeRecommendation {
    /// Symbol
    pub symbol: String,
    /// Base position size (before volatility adjustment)
    pub base_size: f64,
    /// Volatility-adjusted position size
    pub adjusted_size: f64,
    /// Scale factor applied
    pub scale_factor: f64,
    /// Current volatility used
    pub volatility_used: f64,
    /// Target volatility
    pub target_volatility: f64,
    /// Regime adjustment applied
    pub regime_adjustment: f64,
    /// Notes/warnings
    pub notes: Vec<String>,
}

/// Volatility-adjusted sizing
pub struct VolatilityScaling {
    /// Configuration
    config: VolatilityScalingConfig,
    /// Per-symbol volatility data
    symbol_data: HashMap<String, SymbolVolatility>,
    /// Historical volatility distribution (for percentile calculation)
    volatility_distribution: Vec<f64>,
    /// Portfolio-level volatility estimate
    portfolio_volatility: f64,
    /// Total calculations performed
    calculations_count: usize,
}

impl Default for VolatilityScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl VolatilityScaling {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self {
            config: VolatilityScalingConfig::default(),
            symbol_data: HashMap::new(),
            volatility_distribution: Vec::new(),
            portfolio_volatility: 0.0,
            calculations_count: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: VolatilityScalingConfig) -> Self {
        Self {
            config,
            symbol_data: HashMap::new(),
            volatility_distribution: Vec::new(),
            portfolio_volatility: 0.0,
            calculations_count: 0,
        }
    }

    /// Add a price bar for a symbol
    pub fn add_price_bar(&mut self, symbol: &str, bar: PriceBar) {
        let sym_data = self
            .symbol_data
            .entry(symbol.to_string())
            .or_insert_with(|| SymbolVolatility::new(symbol.to_string()));

        // Calculate return if we have previous data
        if let Some(prev_bar) = sym_data.price_history.last() {
            let ret = bar.return_pct(prev_bar.close);
            sym_data.return_history.push(ret);

            // Calculate true range
            let tr = bar.true_range(prev_bar.close);

            // Update ATR (simple moving average of TR)
            if sym_data.atr_history.len() >= self.config.lookback_period {
                let atr = self.calculate_atr(&sym_data.atr_history, tr);
                sym_data.current_atr = atr;
            }
            sym_data.atr_history.push(tr);

            // Update EWMA variance
            let lambda = self.config.ewma_lambda;
            sym_data.ewma_variance = lambda * sym_data.ewma_variance + (1.0 - lambda) * ret * ret;
        }

        sym_data.price_history.push(bar);

        // Keep history bounded
        let max_history = self.config.lookback_period * 10;
        if sym_data.price_history.len() > max_history {
            sym_data.price_history.remove(0);
        }
        if sym_data.atr_history.len() > max_history {
            sym_data.atr_history.remove(0);
        }
        if sym_data.return_history.len() > max_history {
            sym_data.return_history.remove(0);
        }
    }

    /// Calculate ATR using simple moving average
    fn calculate_atr(&self, tr_history: &[f64], current_tr: f64) -> f64 {
        let period = self.config.lookback_period;
        if tr_history.len() < period {
            // Simple average of available data
            let sum: f64 = tr_history.iter().sum::<f64>() + current_tr;
            return sum / (tr_history.len() + 1) as f64;
        }

        // Use last N-1 values plus current
        let start = tr_history.len().saturating_sub(period - 1);
        let sum: f64 = tr_history[start..].iter().sum::<f64>() + current_tr;
        sum / period as f64
    }

    /// Calculate volatility for a symbol
    pub fn calculate_volatility(&mut self, symbol: &str) -> Result<VolatilityResult> {
        self.calculations_count += 1;

        let sym_data = self
            .symbol_data
            .get_mut(symbol)
            .ok_or_else(|| Error::NotFound(format!("No data found for symbol: {}", symbol)))?;

        if sym_data.return_history.len() < 2 {
            return Err(Error::InvalidInput(
                "Insufficient data for volatility calculation".to_string(),
            ));
        }

        let mut result = VolatilityResult::default();

        // Calculate daily volatility based on configured method
        let daily_vol = match self.config.measure {
            VolatilityMeasure::ATR => {
                // ATR as percentage of price
                if let Some(last_bar) = sym_data.price_history.last() {
                    if last_bar.close > 0.0 {
                        sym_data.current_atr / last_bar.close
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            }
            VolatilityMeasure::StdDev => self.calculate_std_dev(&sym_data.return_history),
            VolatilityMeasure::EWMA => sym_data.ewma_variance.sqrt(),
            VolatilityMeasure::Parkinson => self.calculate_parkinson(&sym_data.price_history),
            VolatilityMeasure::YangZhang => self.calculate_yang_zhang(&sym_data.price_history),
            VolatilityMeasure::Implied => {
                // Placeholder - would require options data
                self.calculate_std_dev(&sym_data.return_history)
            }
        };

        result.daily_volatility = daily_vol;
        result.annualized_volatility = daily_vol * self.config.trading_days_per_year.sqrt();
        sym_data.current_volatility = daily_vol;

        // Calculate average volatility over lookback
        let recent_vols: Vec<f64> = if sym_data.return_history.len() >= self.config.lookback_period
        {
            sym_data
                .return_history
                .windows(self.config.lookback_period)
                .map(|w| self.calculate_std_dev(w))
                .collect()
        } else {
            vec![daily_vol]
        };
        result.average_volatility = recent_vols.iter().sum::<f64>() / recent_vols.len() as f64;

        // Calculate volatility of volatility
        if recent_vols.len() > 1 {
            result.vol_of_vol = self.calculate_std_dev(&recent_vols) / result.average_volatility;
        }

        // Update distribution and calculate percentile
        self.volatility_distribution.push(daily_vol);
        if self.volatility_distribution.len() > 1000 {
            self.volatility_distribution.remove(0);
        }
        result.percentile = self.calculate_percentile(daily_vol);

        // Determine regime
        result.regime = self.classify_regime(result.percentile);

        // Detect spike
        if self.config.detect_spikes && result.average_volatility > 0.0 {
            let spike_ratio = daily_vol / result.average_volatility;
            result.spike_detected = spike_ratio > self.config.spike_threshold;
            if result.spike_detected {
                result.regime = VolatilityRegime::Spike;
            }
        }

        // Calculate scale factor
        let base_scale = if daily_vol > 0.0 {
            self.config.target_volatility
                / (result.annualized_volatility / self.config.trading_days_per_year.sqrt())
        } else {
            1.0
        };

        // Apply regime adjustment if enabled
        let regime_adjustment = if self.config.use_regime_adjustment {
            result.regime.scaling_factor()
        } else {
            1.0
        };

        let final_scale = (base_scale * regime_adjustment)
            .max(self.config.min_scale)
            .min(self.config.max_scale);

        result.scale_factor = final_scale;

        // Store result
        sym_data.last_result = Some(result.clone());

        Ok(result)
    }

    /// Calculate standard deviation of returns
    fn calculate_std_dev(&self, returns: &[f64]) -> f64 {
        if returns.len() < 2 {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate Parkinson volatility (high-low range based)
    fn calculate_parkinson(&self, bars: &[PriceBar]) -> f64 {
        if bars.len() < 2 {
            return 0.0;
        }

        let period = self.config.lookback_period.min(bars.len());
        let recent_bars = &bars[bars.len() - period..];

        let sum: f64 = recent_bars
            .iter()
            .map(|bar| {
                if bar.low > 0.0 {
                    let log_hl = (bar.high / bar.low).ln();
                    log_hl * log_hl
                } else {
                    0.0
                }
            })
            .sum();

        let factor = 1.0 / (4.0 * 2.0_f64.ln());
        (factor * sum / period as f64).sqrt()
    }

    /// Calculate Yang-Zhang volatility (comprehensive estimator)
    fn calculate_yang_zhang(&self, bars: &[PriceBar]) -> f64 {
        if bars.len() < 3 {
            return self.calculate_std_dev(
                &bars
                    .windows(2)
                    .map(|w| w[1].return_pct(w[0].close))
                    .collect::<Vec<_>>(),
            );
        }

        let period = self.config.lookback_period.min(bars.len() - 1);
        let recent_bars = &bars[bars.len() - period - 1..];

        let k = 0.34 / (1.34 + (period as f64 + 1.0) / (period as f64 - 1.0));

        // Overnight variance
        let overnight_returns: Vec<f64> = recent_bars
            .windows(2)
            .map(|w| {
                if w[0].close > 0.0 {
                    (w[1].open / w[0].close).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let overnight_var = self.calculate_variance(&overnight_returns);

        // Open-close variance
        let open_close_returns: Vec<f64> = recent_bars[1..]
            .iter()
            .map(|bar| {
                if bar.open > 0.0 {
                    (bar.close / bar.open).ln()
                } else {
                    0.0
                }
            })
            .collect();
        let open_close_var = self.calculate_variance(&open_close_returns);

        // Rogers-Satchell variance
        let rs_var: f64 = recent_bars[1..]
            .iter()
            .map(|bar| {
                if bar.open > 0.0 && bar.low > 0.0 {
                    let log_ho = (bar.high / bar.open).ln();
                    let log_hc = (bar.high / bar.close).ln();
                    let log_lo = (bar.low / bar.open).ln();
                    let log_lc = (bar.low / bar.close).ln();
                    log_ho * log_hc + log_lo * log_lc
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / period as f64;

        let yang_zhang_var = overnight_var + k * open_close_var + (1.0 - k) * rs_var;
        yang_zhang_var.max(0.0).sqrt()
    }

    /// Calculate variance
    fn calculate_variance(&self, data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }

    /// Calculate percentile of current volatility
    fn calculate_percentile(&self, current_vol: f64) -> f64 {
        if self.volatility_distribution.is_empty() {
            return 0.5;
        }

        let count_below = self
            .volatility_distribution
            .iter()
            .filter(|&&v| v < current_vol)
            .count();

        count_below as f64 / self.volatility_distribution.len() as f64
    }

    /// Classify volatility regime based on percentile
    fn classify_regime(&self, percentile: f64) -> VolatilityRegime {
        let thresholds = &self.config.regime_thresholds;

        if percentile < thresholds[0] {
            VolatilityRegime::VeryLow
        } else if percentile < thresholds[1] {
            VolatilityRegime::Low
        } else if percentile < thresholds[2] {
            VolatilityRegime::Normal
        } else if percentile < thresholds[3] {
            VolatilityRegime::High
        } else {
            VolatilityRegime::VeryHigh
        }
    }

    /// Get position size recommendation
    pub fn get_position_size(
        &mut self,
        symbol: &str,
        base_size: f64,
        price: f64,
    ) -> Result<PositionSizeRecommendation> {
        let vol_result = self.calculate_volatility(symbol)?;

        let mut notes = Vec::new();

        // Calculate adjusted size
        let adjusted_size = base_size * vol_result.scale_factor;

        // Add warnings if applicable
        if vol_result.spike_detected {
            notes.push("Volatility spike detected - size significantly reduced".to_string());
        }

        if vol_result.regime == VolatilityRegime::VeryHigh {
            notes.push("Very high volatility regime - proceed with caution".to_string());
        }

        if vol_result.scale_factor <= self.config.min_scale {
            notes.push(format!(
                "Position capped at minimum scale factor ({:.1}x)",
                self.config.min_scale
            ));
        }

        if vol_result.scale_factor >= self.config.max_scale {
            notes.push(format!(
                "Position capped at maximum scale factor ({:.1}x)",
                self.config.max_scale
            ));
        }

        // Calculate dollar-adjusted values
        let base_notional = base_size * price;
        let adjusted_notional = adjusted_size * price;

        if adjusted_notional > base_notional * 2.0 {
            notes.push("Significant size increase due to low volatility".to_string());
        }

        Ok(PositionSizeRecommendation {
            symbol: symbol.to_string(),
            base_size,
            adjusted_size,
            scale_factor: vol_result.scale_factor,
            volatility_used: vol_result.daily_volatility,
            target_volatility: self.config.target_volatility,
            regime_adjustment: vol_result.regime.scaling_factor(),
            notes,
        })
    }

    /// Get volatility for a symbol (if calculated)
    pub fn get_volatility(&self, symbol: &str) -> Option<&VolatilityResult> {
        self.symbol_data
            .get(symbol)
            .and_then(|s| s.last_result.as_ref())
    }

    /// Get symbol data
    pub fn get_symbol_data(&self, symbol: &str) -> Option<&SymbolVolatility> {
        self.symbol_data.get(symbol)
    }

    /// Get all tracked symbols
    pub fn get_symbols(&self) -> Vec<&String> {
        self.symbol_data.keys().collect()
    }

    /// Set target volatility
    pub fn set_target_volatility(&mut self, target: f64) {
        self.config.target_volatility = target;
    }

    /// Get target volatility
    pub fn target_volatility(&self) -> f64 {
        self.config.target_volatility
    }

    /// Get calculation count
    pub fn calculations_count(&self) -> usize {
        self.calculations_count
    }

    /// Set volatility measure method
    pub fn set_measure(&mut self, measure: VolatilityMeasure) {
        self.config.measure = measure;
    }

    /// Calculate portfolio-level volatility (simple average for now)
    pub fn calculate_portfolio_volatility(&mut self) -> f64 {
        let vols: Vec<f64> = self
            .symbol_data
            .values()
            .filter_map(|s| s.last_result.as_ref().map(|r| r.annualized_volatility))
            .collect();

        if vols.is_empty() {
            return 0.0;
        }

        // Simple average (more sophisticated would include correlations)
        self.portfolio_volatility = vols.iter().sum::<f64>() / vols.len() as f64;
        self.portfolio_volatility
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via calculate methods
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_bars(n: usize, base_price: f64, volatility: f64) -> Vec<PriceBar> {
        let mut bars = Vec::with_capacity(n);
        let mut price = base_price;

        for i in 0..n {
            let change = volatility * if i % 2 == 0 { 1.0 } else { -0.5 };
            price *= 1.0 + change;

            let high = price * (1.0 + volatility.abs());
            let low = price * (1.0 - volatility.abs());

            bars.push(PriceBar::new(price, high, low, price, 1000.0, i as u64));
        }

        bars
    }

    #[test]
    fn test_basic() {
        let instance = VolatilityScaling::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_add_price_bars() {
        let mut scaler = VolatilityScaling::new();
        let bars = create_test_bars(30, 100.0, 0.02);

        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let sym_data = scaler.get_symbol_data("AAPL").unwrap();
        assert_eq!(sym_data.price_history.len(), 30);
        assert!(sym_data.return_history.len() > 0);
    }

    #[test]
    fn test_calculate_volatility() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.lookback_period = 10;

        let bars = create_test_bars(20, 100.0, 0.02);
        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let result = scaler.calculate_volatility("AAPL").unwrap();
        assert!(result.daily_volatility > 0.0);
        assert!(result.annualized_volatility > result.daily_volatility);
        assert!(result.scale_factor > 0.0);
    }

    #[test]
    fn test_volatility_not_found() {
        let mut scaler = VolatilityScaling::new();
        let result = scaler.calculate_volatility("UNKNOWN");
        assert!(result.is_err());
    }

    #[test]
    fn test_regime_classification() {
        let scaler = VolatilityScaling::new();

        assert_eq!(scaler.classify_regime(0.05), VolatilityRegime::VeryLow);
        assert_eq!(scaler.classify_regime(0.15), VolatilityRegime::Low);
        assert_eq!(scaler.classify_regime(0.50), VolatilityRegime::Normal);
        assert_eq!(scaler.classify_regime(0.80), VolatilityRegime::High);
        assert_eq!(scaler.classify_regime(0.95), VolatilityRegime::VeryHigh);
    }

    #[test]
    fn test_regime_scaling_factors() {
        assert!(VolatilityRegime::VeryLow.scaling_factor() > 1.0);
        assert_eq!(VolatilityRegime::Normal.scaling_factor(), 1.0);
        assert!(VolatilityRegime::VeryHigh.scaling_factor() < 1.0);
        assert!(
            VolatilityRegime::Spike.scaling_factor() < VolatilityRegime::VeryHigh.scaling_factor()
        );
    }

    #[test]
    fn test_position_size_recommendation() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.lookback_period = 10;
        scaler.config.target_volatility = 0.15;

        let bars = create_test_bars(20, 100.0, 0.02);
        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let rec = scaler.get_position_size("AAPL", 100.0, 150.0).unwrap();
        assert_eq!(rec.symbol, "AAPL");
        assert_eq!(rec.base_size, 100.0);
        assert!(rec.adjusted_size > 0.0);
        assert!(rec.scale_factor > 0.0);
    }

    #[test]
    fn test_high_volatility_reduces_size() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.lookback_period = 10;
        scaler.config.target_volatility = 0.10; // Low target

        // High volatility bars
        let bars = create_test_bars(20, 100.0, 0.05);
        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let rec = scaler.get_position_size("AAPL", 100.0, 150.0).unwrap();
        // With high vol and low target, scale should be reduced
        assert!(rec.scale_factor < 2.0);
    }

    #[test]
    fn test_std_dev_calculation() {
        let scaler = VolatilityScaling::new();
        let returns = vec![0.01, -0.01, 0.02, -0.02, 0.01];
        let std_dev = scaler.calculate_std_dev(&returns);
        assert!(std_dev > 0.0);
        assert!(std_dev < 0.05); // Reasonable range
    }

    #[test]
    fn test_price_bar_true_range() {
        let bar = PriceBar::new(100.0, 105.0, 95.0, 102.0, 1000.0, 0);

        // Normal case: high-low is max
        let tr = bar.true_range(100.0);
        assert_eq!(tr, 10.0); // 105 - 95

        // Gap up case
        let tr_gap_up = bar.true_range(90.0);
        assert_eq!(tr_gap_up, 15.0); // 105 - 90

        // Gap down case
        let tr_gap_down = bar.true_range(110.0);
        assert_eq!(tr_gap_down, 15.0); // 110 - 95
    }

    #[test]
    fn test_ewma_update() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.measure = VolatilityMeasure::EWMA;

        let bars = create_test_bars(30, 100.0, 0.02);
        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let sym_data = scaler.get_symbol_data("AAPL").unwrap();
        assert!(sym_data.ewma_variance > 0.0);
    }

    #[test]
    fn test_spike_detection() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.lookback_period = 10;
        scaler.config.detect_spikes = true;
        scaler.config.spike_threshold = 2.0;

        // Normal volatility bars
        let bars = create_test_bars(15, 100.0, 0.01);
        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        // Add spike bars
        let spike_bars = create_test_bars(5, 100.0, 0.05);
        for bar in spike_bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let result = scaler.calculate_volatility("AAPL").unwrap();
        // Result should reflect elevated volatility
        assert!(result.daily_volatility > 0.0);
    }

    #[test]
    fn test_scale_factor_bounds() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.min_scale = 0.5;
        scaler.config.max_scale = 2.0;
        scaler.config.lookback_period = 10;

        // Very low volatility (would normally scale up a lot)
        let bars = create_test_bars(20, 100.0, 0.001);
        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let result = scaler.calculate_volatility("AAPL").unwrap();
        assert!(result.scale_factor <= 2.0);
        assert!(result.scale_factor >= 0.5);
    }

    #[test]
    fn test_parkinson_volatility() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.measure = VolatilityMeasure::Parkinson;
        scaler.config.lookback_period = 10;

        let bars = create_test_bars(20, 100.0, 0.02);
        for bar in bars {
            scaler.add_price_bar("AAPL", bar);
        }

        let result = scaler.calculate_volatility("AAPL").unwrap();
        assert!(result.daily_volatility > 0.0);
    }

    #[test]
    fn test_multiple_symbols() {
        let mut scaler = VolatilityScaling::new();
        scaler.config.lookback_period = 10;

        let bars1 = create_test_bars(20, 100.0, 0.02);
        let bars2 = create_test_bars(20, 50.0, 0.03);

        for (b1, b2) in bars1.into_iter().zip(bars2.into_iter()) {
            scaler.add_price_bar("AAPL", b1);
            scaler.add_price_bar("GOOG", b2);
        }

        let symbols = scaler.get_symbols();
        assert_eq!(symbols.len(), 2);

        let _ = scaler.calculate_volatility("AAPL").unwrap();
        let _ = scaler.calculate_volatility("GOOG").unwrap();

        let portfolio_vol = scaler.calculate_portfolio_volatility();
        assert!(portfolio_vol > 0.0);
    }

    #[test]
    fn test_set_target_volatility() {
        let mut scaler = VolatilityScaling::new();
        scaler.set_target_volatility(0.20);
        assert_eq!(scaler.target_volatility(), 0.20);
    }

    #[test]
    fn test_insufficient_data() {
        let mut scaler = VolatilityScaling::new();
        scaler.add_price_bar("AAPL", PriceBar::new(100.0, 101.0, 99.0, 100.0, 1000.0, 0));

        let result = scaler.calculate_volatility("AAPL");
        assert!(result.is_err());
    }

    #[test]
    fn test_return_calculation() {
        let bar = PriceBar::new(100.0, 105.0, 95.0, 110.0, 1000.0, 0);

        let ret = bar.return_pct(100.0);
        assert!((ret - 0.10).abs() < 0.001); // 10% return

        let log_ret = bar.log_return(100.0);
        assert!((log_ret - 0.10_f64.ln_1p()).abs() < 0.001);
    }
}
