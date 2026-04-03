//! Position sizing with volatility adjustments.
//!
//! This module provides advanced position sizing strategies that adjust
//! based on market volatility, account size, and risk parameters.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Volatility estimation window
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityWindow {
    /// 10 periods
    Short,
    /// 20 periods
    Medium,
    /// 50 periods
    Long,
    /// Custom number of periods
    Custom(usize),
}

impl VolatilityWindow {
    /// Get the number of periods
    pub fn periods(&self) -> usize {
        match self {
            VolatilityWindow::Short => 10,
            VolatilityWindow::Medium => 20,
            VolatilityWindow::Long => 50,
            VolatilityWindow::Custom(n) => *n,
        }
    }
}

impl Default for VolatilityWindow {
    fn default() -> Self {
        VolatilityWindow::Medium
    }
}

/// Volatility estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityEstimate {
    /// Historical volatility (standard deviation of returns)
    pub historical_vol: f64,

    /// Average True Range (ATR)
    pub atr: f64,

    /// Current volatility percentile (0-1)
    pub percentile: f64,

    /// Is volatility high relative to history?
    pub is_high: bool,

    /// Number of periods used for calculation
    pub periods: usize,
}

impl VolatilityEstimate {
    /// Create a new volatility estimate
    pub fn new(historical_vol: f64, atr: f64, percentile: f64, periods: usize) -> Self {
        Self {
            historical_vol,
            atr,
            percentile,
            is_high: percentile > 0.75, // High if above 75th percentile
            periods,
        }
    }

    /// Get scaling factor for position size based on volatility
    /// High volatility = smaller positions
    pub fn size_scaling_factor(&self, target_vol: f64) -> f64 {
        if self.historical_vol == 0.0 {
            return 1.0;
        }

        let scale = target_vol / self.historical_vol;
        scale.clamp(0.5, 2.0) // Limit adjustment to 50%-200%
    }
}

/// Volatility adjuster for position sizing
pub struct VolatilityAdjuster {
    /// Target volatility
    target_volatility: f64,

    /// Lookback window
    window: VolatilityWindow,

    /// Price history
    price_history: VecDeque<f64>,

    /// Return history
    return_history: VecDeque<f64>,

    /// ATR history
    atr_history: VecDeque<f64>,

    /// Current volatility estimate
    current_estimate: Option<VolatilityEstimate>,
}

impl VolatilityAdjuster {
    /// Create a new volatility adjuster
    pub fn new(target_volatility: f64, window: VolatilityWindow) -> Self {
        let capacity = window.periods() + 1;
        Self {
            target_volatility,
            window,
            price_history: VecDeque::with_capacity(capacity),
            return_history: VecDeque::with_capacity(capacity),
            atr_history: VecDeque::with_capacity(capacity),
            current_estimate: None,
        }
    }

    /// Update with new price
    pub fn update_price(&mut self, price: f64, high: Option<f64>, low: Option<f64>) {
        // Add to price history
        self.price_history.push_back(price);
        let periods = self.window.periods();

        // Keep only window size
        if self.price_history.len() > periods + 1 {
            self.price_history.pop_front();
        }

        // Calculate return if we have previous price
        if self.price_history.len() >= 2 {
            let prev_price = self.price_history[self.price_history.len() - 2];
            if prev_price > 0.0 {
                let ret = (price - prev_price) / prev_price;
                self.return_history.push_back(ret);

                if self.return_history.len() > periods {
                    self.return_history.pop_front();
                }
            }
        }

        // Calculate ATR if we have high/low
        if let (Some(h), Some(l)) = (high, low) {
            if self.price_history.len() >= 2 {
                let prev_close = self.price_history[self.price_history.len() - 2];
                let tr = (h - l)
                    .max((h - prev_close).abs())
                    .max((l - prev_close).abs());

                self.atr_history.push_back(tr);

                if self.atr_history.len() > periods {
                    self.atr_history.pop_front();
                }
            }
        }

        // Update estimate
        self.calculate_volatility();
    }

    /// Calculate current volatility
    fn calculate_volatility(&mut self) {
        if self.return_history.len() < 5 {
            return; // Need minimum data
        }

        // Calculate historical volatility (standard deviation of returns)
        let mean_return: f64 =
            self.return_history.iter().sum::<f64>() / self.return_history.len() as f64;

        let variance: f64 = self
            .return_history
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (self.return_history.len() - 1) as f64;

        let historical_vol = variance.sqrt();

        // Annualize (assuming daily returns, 252 trading days)
        let annualized_vol = historical_vol * (252.0_f64).sqrt();

        // Calculate ATR
        let atr = if !self.atr_history.is_empty() {
            self.atr_history.iter().sum::<f64>() / self.atr_history.len() as f64
        } else {
            0.0
        };

        // Calculate percentile (simplified - compare to recent history)
        let percentile = if annualized_vol > 0.0 {
            let higher_count = self
                .return_history
                .iter()
                .filter(|&&r| r.abs() > historical_vol)
                .count();
            higher_count as f64 / self.return_history.len() as f64
        } else {
            0.5
        };

        self.current_estimate = Some(VolatilityEstimate::new(
            annualized_vol,
            atr,
            percentile,
            self.return_history.len(),
        ));
    }

    /// Get current volatility estimate
    pub fn current_estimate(&self) -> Option<&VolatilityEstimate> {
        self.current_estimate.as_ref()
    }

    /// Get position size adjustment factor
    pub fn get_adjustment_factor(&self) -> f64 {
        if let Some(ref estimate) = self.current_estimate {
            estimate.size_scaling_factor(self.target_volatility)
        } else {
            1.0 // No adjustment if no data
        }
    }

    /// Reset the adjuster
    pub fn reset(&mut self) {
        self.price_history.clear();
        self.return_history.clear();
        self.atr_history.clear();
        self.current_estimate = None;
    }
}

/// Position sizing method
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SizingMethod {
    /// Fixed percentage of capital
    FixedPercent,
    /// Volatility-scaled
    VolatilityScaled,
    /// ATR-based
    AtrBased,
    /// Risk-based (fixed risk per trade)
    RiskBased,
}

/// Position sizer with multiple strategies
pub struct PositionSizer {
    /// Maximum position size (fraction of capital)
    max_position_size: f64,

    /// Enable volatility scaling
    use_volatility_scaling: bool,

    /// Target volatility for scaling
    target_volatility: f64,

    /// Volatility adjuster
    volatility_adjuster: Option<VolatilityAdjuster>,

    /// Current sizing method
    method: SizingMethod,
}

impl PositionSizer {
    /// Create a new position sizer
    pub fn new(
        max_position_size: f64,
        use_volatility_scaling: bool,
        target_volatility: f64,
        window_periods: usize,
    ) -> Self {
        let volatility_adjuster = if use_volatility_scaling {
            Some(VolatilityAdjuster::new(
                target_volatility,
                VolatilityWindow::Custom(window_periods),
            ))
        } else {
            None
        };

        Self {
            max_position_size,
            use_volatility_scaling,
            target_volatility,
            volatility_adjuster,
            method: if use_volatility_scaling {
                SizingMethod::VolatilityScaled
            } else {
                SizingMethod::FixedPercent
            },
        }
    }

    /// Calculate position size
    pub fn calculate_size(
        &self,
        capital: f64,
        price: f64,
        volatility: Option<f64>,
    ) -> Result<f64, super::RiskError> {
        if capital <= 0.0 || price <= 0.0 {
            return Err(super::RiskError::InvalidConfig(
                "Capital and price must be positive".to_string(),
            ));
        }

        // Base size (max allowed)
        let base_size_value = capital * self.max_position_size;

        // Apply volatility adjustment if enabled
        let adjusted_size_value = if self.use_volatility_scaling {
            if let Some(ref adjuster) = self.volatility_adjuster {
                let adjustment = adjuster.get_adjustment_factor();
                base_size_value * adjustment
            } else if let Some(vol) = volatility {
                // Use provided volatility
                let adjustment = if vol > 0.0 {
                    (self.target_volatility / vol).clamp(0.5, 2.0)
                } else {
                    1.0
                };
                base_size_value * adjustment
            } else {
                base_size_value
            }
        } else {
            base_size_value
        };

        // Convert to quantity
        let size = adjusted_size_value / price;

        Ok(size)
    }

    /// Update volatility adjuster with new price data
    pub fn update_prices(&mut self, price: f64, high: Option<f64>, low: Option<f64>) {
        if let Some(ref mut adjuster) = self.volatility_adjuster {
            adjuster.update_price(price, high, low);
        }
    }

    /// Get current volatility estimate
    pub fn current_volatility(&self) -> Option<&VolatilityEstimate> {
        self.volatility_adjuster
            .as_ref()
            .and_then(|adj| adj.current_estimate())
    }

    /// Set sizing method
    pub fn set_method(&mut self, method: SizingMethod) {
        self.method = method;
    }

    /// Get current sizing method
    pub fn method(&self) -> SizingMethod {
        self.method
    }

    /// Reset volatility calculations
    pub fn reset(&mut self) {
        if let Some(ref mut adjuster) = self.volatility_adjuster {
            adjuster.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_window() {
        assert_eq!(VolatilityWindow::Short.periods(), 10);
        assert_eq!(VolatilityWindow::Medium.periods(), 20);
        assert_eq!(VolatilityWindow::Long.periods(), 50);
        assert_eq!(VolatilityWindow::Custom(30).periods(), 30);
    }

    #[test]
    fn test_volatility_estimate_creation() {
        let estimate = VolatilityEstimate::new(0.20, 100.0, 0.80, 20);
        assert_eq!(estimate.historical_vol, 0.20);
        assert_eq!(estimate.atr, 100.0);
        assert_eq!(estimate.percentile, 0.80);
        assert!(estimate.is_high);
    }

    #[test]
    fn test_volatility_scaling_factor() {
        let estimate = VolatilityEstimate::new(0.30, 100.0, 0.50, 20);

        // Target vol 0.15, current vol 0.30
        // Scale = 0.15 / 0.30 = 0.5
        let scale = estimate.size_scaling_factor(0.15);
        assert!((scale - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_volatility_adjuster_creation() {
        let adjuster = VolatilityAdjuster::new(0.15, VolatilityWindow::Medium);
        assert_eq!(adjuster.target_volatility, 0.15);
        assert!(adjuster.current_estimate().is_none());
    }

    #[test]
    fn test_volatility_adjuster_update() {
        let mut adjuster = VolatilityAdjuster::new(0.15, VolatilityWindow::Short);

        // Add prices
        for i in 0..15 {
            let price = 100.0 + i as f64;
            adjuster.update_price(price, Some(price + 2.0), Some(price - 2.0));
        }

        // Should have estimate now
        assert!(adjuster.current_estimate().is_some());
    }

    #[test]
    fn test_position_sizer_creation() {
        let sizer = PositionSizer::new(0.10, true, 0.15, 20);
        assert_eq!(sizer.max_position_size, 0.10);
        assert!(sizer.use_volatility_scaling);
        assert!(sizer.volatility_adjuster.is_some());
    }

    #[test]
    fn test_position_sizer_calculate_size() {
        let sizer = PositionSizer::new(0.10, false, 0.15, 20);

        let size = sizer.calculate_size(10000.0, 50000.0, None).unwrap();

        // 10% of 10000 = 1000, divided by price 50000 = 0.02
        assert!((size - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_position_sizer_with_volatility() {
        let sizer = PositionSizer::new(0.10, true, 0.15, 20);

        // Calculate size with provided volatility
        let size = sizer.calculate_size(10000.0, 50000.0, Some(0.30)).unwrap();

        // Should successfully calculate a positive size
        assert!(size > 0.0);
        // Size should be reasonable (not exceed max)
        assert!(size <= 0.02); // Max would be 10% of 10000 / 50000 = 0.02
    }

    #[test]
    fn test_position_sizer_invalid_inputs() {
        let sizer = PositionSizer::new(0.10, false, 0.15, 20);

        // Zero capital
        assert!(sizer.calculate_size(0.0, 50000.0, None).is_err());

        // Negative price
        assert!(sizer.calculate_size(10000.0, -50000.0, None).is_err());
    }

    #[test]
    fn test_position_sizer_update_prices() {
        let mut sizer = PositionSizer::new(0.10, true, 0.15, 10);

        for i in 0..15 {
            let price = 100.0 + i as f64;
            sizer.update_prices(price, Some(price + 1.0), Some(price - 1.0));
        }

        assert!(sizer.current_volatility().is_some());
    }

    #[test]
    fn test_position_sizer_reset() {
        let mut sizer = PositionSizer::new(0.10, true, 0.15, 10);

        for i in 0..15 {
            sizer.update_prices(100.0 + i as f64, None, None);
        }

        sizer.reset();
        assert!(sizer.current_volatility().is_none());
    }

    #[test]
    fn test_sizing_method() {
        let mut sizer = PositionSizer::new(0.10, false, 0.15, 20);
        assert_eq!(sizer.method(), SizingMethod::FixedPercent);

        sizer.set_method(SizingMethod::AtrBased);
        assert_eq!(sizer.method(), SizingMethod::AtrBased);
    }
}
