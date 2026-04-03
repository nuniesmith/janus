//! Technical Indicators for Regime Detection
//!
//! Self-contained indicator implementations used by the regime detection system.
//! Provides EMA, ATR, ADX, and Bollinger Bands calculations optimized for
//! market regime classification.
//!
//! These are intentionally kept within the regime crate rather than depending on
//! `janus-indicators`, because:
//! 1. The regime crate needs specific indicator semantics (e.g., ADX with DI crossover)
//! 2. Keeps the crate self-contained with zero internal dependencies
//! 3. `janus-indicators` can later delegate to these if desired

use super::types::TrendDirection;
use std::collections::VecDeque;

// ============================================================================
// Exponential Moving Average (EMA)
// ============================================================================

/// Exponential Moving Average calculator
///
/// Uses the standard EMA formula: EMA_t = price * k + EMA_{t-1} * (1 - k)
/// where k = 2 / (period + 1)
#[derive(Debug, Clone)]
pub struct EMA {
    period: usize,
    multiplier: f64,
    current_value: Option<f64>,
    initialized: bool,
    warmup_count: usize,
}

impl EMA {
    /// Create a new EMA with the given period
    pub fn new(period: usize) -> Self {
        let multiplier = 2.0 / (period as f64 + 1.0);
        Self {
            period,
            multiplier,
            current_value: None,
            initialized: false,
            warmup_count: 0,
        }
    }

    /// Update with a new price value, returning the EMA if warmed up
    pub fn update(&mut self, price: f64) -> Option<f64> {
        self.warmup_count += 1;

        match self.current_value {
            Some(prev_ema) => {
                let new_ema = (price - prev_ema) * self.multiplier + prev_ema;
                self.current_value = Some(new_ema);

                if self.warmup_count >= self.period {
                    self.initialized = true;
                }
            }
            None => {
                self.current_value = Some(price);
            }
        }

        if self.initialized {
            self.current_value
        } else {
            None
        }
    }

    /// Get the current EMA value (None if not yet warmed up)
    pub fn value(&self) -> Option<f64> {
        if self.initialized {
            self.current_value
        } else {
            None
        }
    }

    /// Check if the EMA has enough data to produce valid values
    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Reset the EMA state
    pub fn reset(&mut self) {
        self.current_value = None;
        self.initialized = false;
        self.warmup_count = 0;
    }
}

// ============================================================================
// Average True Range (ATR)
// ============================================================================

/// Average True Range (ATR) calculator
///
/// Uses Wilder's smoothing method for the ATR calculation.
/// True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
    values: VecDeque<f64>,
    prev_close: Option<f64>,
    current_atr: Option<f64>,
}

impl ATR {
    /// Create a new ATR with the given period
    pub fn new(period: usize) -> Self {
        Self {
            period,
            values: VecDeque::with_capacity(period),
            prev_close: None,
            current_atr: None,
        }
    }

    /// Update with OHLC data, returning the ATR if warmed up
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        let true_range = match self.prev_close {
            Some(prev_c) => {
                let hl = high - low;
                let hc = (high - prev_c).abs();
                let lc = (low - prev_c).abs();
                hl.max(hc).max(lc)
            }
            None => high - low,
        };

        self.prev_close = Some(close);
        self.values.push_back(true_range);

        if self.values.len() > self.period {
            self.values.pop_front();
        }

        if self.values.len() >= self.period {
            // Use Wilder's smoothing method
            match self.current_atr {
                Some(prev_atr) => {
                    let new_atr =
                        (prev_atr * (self.period - 1) as f64 + true_range) / self.period as f64;
                    self.current_atr = Some(new_atr);
                }
                None => {
                    let sum: f64 = self.values.iter().sum();
                    self.current_atr = Some(sum / self.period as f64);
                }
            }
        }

        self.current_atr
    }

    /// Get the current ATR value
    pub fn value(&self) -> Option<f64> {
        self.current_atr
    }

    /// Check if the ATR has enough data
    pub fn is_ready(&self) -> bool {
        self.current_atr.is_some()
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Reset the ATR state
    pub fn reset(&mut self) {
        self.values.clear();
        self.prev_close = None;
        self.current_atr = None;
    }
}

// ============================================================================
// Average Directional Index (ADX)
// ============================================================================

/// Average Directional Index (ADX) calculator
///
/// Measures trend strength (not direction). Values above 25 typically indicate
/// a strong trend, while values below 20 suggest a ranging market.
///
/// Also provides +DI and -DI for trend direction via `trend_direction()`.
#[derive(Debug, Clone)]
pub struct ADX {
    period: usize,
    atr: ATR,
    plus_dm_ema: EMA,
    minus_dm_ema: EMA,
    dx_values: VecDeque<f64>,
    prev_high: Option<f64>,
    prev_low: Option<f64>,
    current_adx: Option<f64>,
    plus_di: Option<f64>,
    minus_di: Option<f64>,
}

impl ADX {
    /// Create a new ADX with the given period
    pub fn new(period: usize) -> Self {
        Self {
            period,
            atr: ATR::new(period),
            plus_dm_ema: EMA::new(period),
            minus_dm_ema: EMA::new(period),
            dx_values: VecDeque::with_capacity(period),
            prev_high: None,
            prev_low: None,
            current_adx: None,
            plus_di: None,
            minus_di: None,
        }
    }

    /// Update with HLC data, returning the ADX value if warmed up
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Option<f64> {
        // Calculate directional movement
        let (plus_dm, minus_dm) = match (self.prev_high, self.prev_low) {
            (Some(prev_h), Some(prev_l)) => {
                let up_move = high - prev_h;
                let down_move = prev_l - low;

                let plus = if up_move > down_move && up_move > 0.0 {
                    up_move
                } else {
                    0.0
                };

                let minus = if down_move > up_move && down_move > 0.0 {
                    down_move
                } else {
                    0.0
                };

                (plus, minus)
            }
            _ => (0.0, 0.0),
        };

        self.prev_high = Some(high);
        self.prev_low = Some(low);

        // Update ATR
        let atr = self.atr.update(high, low, close);

        // Smooth directional movement
        let smoothed_plus_dm = self.plus_dm_ema.update(plus_dm);
        let smoothed_minus_dm = self.minus_dm_ema.update(minus_dm);

        // Calculate DI values
        if let (Some(atr_val), Some(plus_dm_smooth), Some(minus_dm_smooth)) =
            (atr, smoothed_plus_dm, smoothed_minus_dm)
            && atr_val > 0.0
        {
            self.plus_di = Some((plus_dm_smooth / atr_val) * 100.0);
            self.minus_di = Some((minus_dm_smooth / atr_val) * 100.0);

            // Calculate DX
            let di_sum = self.plus_di.unwrap() + self.minus_di.unwrap();
            if di_sum > 0.0 {
                let di_diff = (self.plus_di.unwrap() - self.minus_di.unwrap()).abs();
                let dx = (di_diff / di_sum) * 100.0;

                self.dx_values.push_back(dx);
                if self.dx_values.len() > self.period {
                    self.dx_values.pop_front();
                }

                // Calculate ADX as smoothed DX
                if self.dx_values.len() >= self.period {
                    match self.current_adx {
                        Some(prev_adx) => {
                            let new_adx =
                                (prev_adx * (self.period - 1) as f64 + dx) / self.period as f64;
                            self.current_adx = Some(new_adx);
                        }
                        None => {
                            let sum: f64 = self.dx_values.iter().sum();
                            self.current_adx = Some(sum / self.period as f64);
                        }
                    }
                }
            }
        }

        self.current_adx
    }

    /// Get the current ADX value
    pub fn value(&self) -> Option<f64> {
        self.current_adx
    }

    /// Get the +DI value
    pub fn plus_di(&self) -> Option<f64> {
        self.plus_di
    }

    /// Get the -DI value
    pub fn minus_di(&self) -> Option<f64> {
        self.minus_di
    }

    /// Returns trend direction based on DI crossover.
    ///
    /// - `+DI > -DI` → Bullish
    /// - `-DI > +DI` → Bearish
    pub fn trend_direction(&self) -> Option<TrendDirection> {
        match (self.plus_di, self.minus_di) {
            (Some(plus), Some(minus)) => {
                if plus > minus {
                    Some(TrendDirection::Bullish)
                } else {
                    Some(TrendDirection::Bearish)
                }
            }
            _ => None,
        }
    }

    /// Check if the ADX has enough data
    pub fn is_ready(&self) -> bool {
        self.current_adx.is_some()
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Reset the ADX state
    pub fn reset(&mut self) {
        self.atr.reset();
        self.plus_dm_ema.reset();
        self.minus_dm_ema.reset();
        self.dx_values.clear();
        self.prev_high = None;
        self.prev_low = None;
        self.current_adx = None;
        self.plus_di = None;
        self.minus_di = None;
    }
}

// ============================================================================
// Bollinger Bands
// ============================================================================

/// Bollinger Bands output values
#[derive(Debug, Clone, Copy)]
pub struct BollingerBandsValues {
    /// Upper band (SMA + n * σ)
    pub upper: f64,
    /// Middle band (SMA)
    pub middle: f64,
    /// Lower band (SMA - n * σ)
    pub lower: f64,
    /// Band width as percentage of price
    pub width: f64,
    /// Where current width ranks historically (0–100 percentile)
    pub width_percentile: f64,
    /// Where price is within the bands (0.0 = lower, 1.0 = upper)
    pub percent_b: f64,
    /// Standard deviation of prices
    pub std_dev: f64,
}

impl BollingerBandsValues {
    /// Is price overbought (near or above upper band)?
    pub fn is_overbought(&self) -> bool {
        self.percent_b >= 0.95
    }

    /// Is price oversold (near or below lower band)?
    pub fn is_oversold(&self) -> bool {
        self.percent_b <= 0.05
    }

    /// Are bands wide (high volatility)?
    pub fn is_high_volatility(&self, threshold_percentile: f64) -> bool {
        self.width_percentile >= threshold_percentile
    }

    /// Are bands narrow (potential breakout coming)?
    pub fn is_squeeze(&self, threshold_percentile: f64) -> bool {
        self.width_percentile <= threshold_percentile
    }
}

/// Bollinger Bands calculator
///
/// Computes upper, lower, and middle bands along with band width percentile
/// for volatility regime classification.
#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    std_dev_multiplier: f64,
    prices: VecDeque<f64>,
    width_history: VecDeque<f64>,
    width_history_size: usize,
}

impl BollingerBands {
    /// Create a new Bollinger Bands calculator
    ///
    /// # Arguments
    /// * `period` - Lookback period for the SMA (typically 20)
    /// * `std_dev_multiplier` - Standard deviation multiplier (typically 2.0)
    pub fn new(period: usize, std_dev_multiplier: f64) -> Self {
        Self {
            period,
            std_dev_multiplier,
            prices: VecDeque::with_capacity(period),
            width_history: VecDeque::with_capacity(100),
            width_history_size: 100, // Keep 100 periods for percentile calc
        }
    }

    /// Update with a new price, returning band values if warmed up
    pub fn update(&mut self, price: f64) -> Option<BollingerBandsValues> {
        self.prices.push_back(price);
        if self.prices.len() > self.period {
            self.prices.pop_front();
        }

        if self.prices.len() < self.period {
            return None;
        }

        // Calculate SMA (middle band)
        let sum: f64 = self.prices.iter().sum();
        let sma = sum / self.period as f64;

        // Calculate standard deviation
        let variance: f64 =
            self.prices.iter().map(|p| (p - sma).powi(2)).sum::<f64>() / self.period as f64;
        let std_dev = variance.sqrt();

        // Calculate bands
        let upper = sma + (std_dev * self.std_dev_multiplier);
        let lower = sma - (std_dev * self.std_dev_multiplier);
        let width = if sma > 0.0 {
            (upper - lower) / sma * 100.0 // Width as percentage of price
        } else {
            0.0
        };

        // Update width history for percentile calculation
        self.width_history.push_back(width);
        if self.width_history.len() > self.width_history_size {
            self.width_history.pop_front();
        }

        // Calculate width percentile
        let width_percentile = self.calculate_width_percentile(width);

        // Calculate %B (where price is within bands)
        let percent_b = if upper - lower > 0.0 {
            (price - lower) / (upper - lower)
        } else {
            0.5
        };

        Some(BollingerBandsValues {
            upper,
            middle: sma,
            lower,
            width,
            width_percentile,
            percent_b,
            std_dev,
        })
    }

    /// Calculate where the current width ranks in recent history
    fn calculate_width_percentile(&self, current_width: f64) -> f64 {
        if self.width_history.len() < 10 {
            return 50.0; // Not enough data
        }

        let count_below = self
            .width_history
            .iter()
            .filter(|&&w| w < current_width)
            .count();

        (count_below as f64 / self.width_history.len() as f64) * 100.0
    }

    /// Check if the Bollinger Bands have enough data
    pub fn is_ready(&self) -> bool {
        self.prices.len() >= self.period
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Get the standard deviation multiplier
    pub fn std_dev_multiplier(&self) -> f64 {
        self.std_dev_multiplier
    }

    /// Reset the Bollinger Bands state
    pub fn reset(&mut self) {
        self.prices.clear();
        self.width_history.clear();
    }
}

// ============================================================================
// RSI (Relative Strength Index)
// ============================================================================

/// Relative Strength Index (RSI) calculator
///
/// Uses EMA-smoothed gains and losses for a responsive RSI calculation.
/// Values above 70 indicate overbought, below 30 indicate oversold.
#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    gains: EMA,
    losses: EMA,
    prev_close: Option<f64>,
}

impl RSI {
    /// Create a new RSI with the given period (typically 14)
    pub fn new(period: usize) -> Self {
        Self {
            period,
            gains: EMA::new(period),
            losses: EMA::new(period),
            prev_close: None,
        }
    }

    /// Update with a new close price, returning the RSI if warmed up
    pub fn update(&mut self, close: f64) -> Option<f64> {
        if let Some(prev) = self.prev_close {
            let change = close - prev;
            let gain = if change > 0.0 { change } else { 0.0 };
            let loss = if change < 0.0 { -change } else { 0.0 };

            if let (Some(avg_gain), Some(avg_loss)) =
                (self.gains.update(gain), self.losses.update(loss))
            {
                self.prev_close = Some(close);

                if avg_loss == 0.0 {
                    return Some(100.0);
                }

                let rs = avg_gain / avg_loss;
                let rsi = 100.0 - (100.0 / (1.0 + rs));
                return Some(rsi);
            }
        }

        self.prev_close = Some(close);
        None
    }

    /// Get the current RSI value
    pub fn value(&self) -> Option<f64> {
        // RSI doesn't cache, caller should store the last update result
        None
    }

    /// Check if RSI has enough data
    pub fn is_ready(&self) -> bool {
        self.gains.is_ready() && self.losses.is_ready()
    }

    /// Get the period
    pub fn period(&self) -> usize {
        self.period
    }

    /// Reset the RSI state
    pub fn reset(&mut self) {
        self.gains.reset();
        self.losses.reset();
        self.prev_close = None;
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate a Simple Moving Average from a slice of values
pub fn calculate_sma(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }
    prices.iter().sum::<f64>() / prices.len() as f64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- EMA Tests ---

    #[test]
    fn test_ema_creation() {
        let ema = EMA::new(10);
        assert_eq!(ema.period(), 10);
        assert!(!ema.is_ready());
        assert!(ema.value().is_none());
    }

    #[test]
    fn test_ema_warmup() {
        let mut ema = EMA::new(10);

        // Should return None during warmup
        for i in 1..10 {
            let result = ema.update(i as f64 * 10.0);
            assert!(result.is_none(), "Should be None during warmup at step {i}");
        }

        // Should return Some after warmup
        let result = ema.update(100.0);
        assert!(result.is_some(), "Should be ready after {0} updates", 10);
        assert!(ema.is_ready());
    }

    #[test]
    fn test_ema_calculation() {
        let mut ema = EMA::new(10);

        // Warm up
        for i in 1..=10 {
            ema.update(i as f64 * 10.0);
        }

        assert!(ema.is_ready());
        let value = ema.value().unwrap();
        // EMA should be between the min and max input values
        assert!(value > 10.0 && value <= 100.0);
    }

    #[test]
    fn test_ema_tracks_trend() {
        let mut ema = EMA::new(5);

        // Warm up with constant price
        for _ in 0..5 {
            ema.update(100.0);
        }
        let stable = ema.value().unwrap();

        // Feed higher prices
        for _ in 0..10 {
            ema.update(110.0);
        }
        let after_up = ema.value().unwrap();

        assert!(after_up > stable, "EMA should increase with rising prices");
    }

    #[test]
    fn test_ema_reset() {
        let mut ema = EMA::new(5);
        for _ in 0..10 {
            ema.update(100.0);
        }
        assert!(ema.is_ready());

        ema.reset();
        assert!(!ema.is_ready());
        assert!(ema.value().is_none());
    }

    // --- ATR Tests ---

    #[test]
    fn test_atr_creation() {
        let atr = ATR::new(14);
        assert_eq!(atr.period(), 14);
        assert!(!atr.is_ready());
    }

    #[test]
    fn test_atr_warmup() {
        let mut atr = ATR::new(14);

        for i in 1..=14 {
            let base = 100.0 + i as f64;
            let result = atr.update(base + 1.0, base - 1.0, base);
            if i < 14 {
                assert!(result.is_none());
            }
        }

        assert!(atr.is_ready());
    }

    #[test]
    fn test_atr_increases_with_volatility() {
        let mut atr = ATR::new(14);

        // Low volatility warmup
        for i in 1..=14 {
            let base = 100.0 + i as f64 * 0.1;
            atr.update(base + 0.5, base - 0.5, base);
        }
        let low_vol_atr = atr.value().unwrap();

        // High volatility bars
        for i in 0..20 {
            let base = 100.0 + if i % 2 == 0 { 5.0 } else { -5.0 };
            atr.update(base + 3.0, base - 3.0, base);
        }
        let high_vol_atr = atr.value().unwrap();

        assert!(
            high_vol_atr > low_vol_atr,
            "ATR should increase with volatility: {high_vol_atr} vs {low_vol_atr}"
        );
    }

    #[test]
    fn test_atr_reset() {
        let mut atr = ATR::new(14);
        for i in 0..20 {
            let base = 100.0 + i as f64;
            atr.update(base + 1.0, base - 1.0, base);
        }
        assert!(atr.is_ready());

        atr.reset();
        assert!(!atr.is_ready());
        assert!(atr.value().is_none());
    }

    // --- ADX Tests ---

    #[test]
    fn test_adx_creation() {
        let adx = ADX::new(14);
        assert_eq!(adx.period(), 14);
        assert!(!adx.is_ready());
    }

    #[test]
    fn test_adx_trending_detection() {
        let mut adx = ADX::new(14);

        // Simulate strong uptrend (prices going up steadily)
        for i in 1..=50 {
            let high = 100.0 + i as f64 * 2.0;
            let low = 100.0 + i as f64 * 2.0 - 1.0;
            let close = 100.0 + i as f64 * 2.0 - 0.5;
            adx.update(high, low, close);
        }

        if let Some(adx_value) = adx.value() {
            assert!(
                adx_value > 20.0,
                "ADX should indicate trend in strong uptrend: {adx_value}"
            );
        }
    }

    #[test]
    fn test_adx_trend_direction() {
        let mut adx = ADX::new(14);

        // Strong uptrend
        for i in 1..=50 {
            let high = 100.0 + i as f64 * 2.0;
            let low = 100.0 + i as f64 * 2.0 - 1.0;
            let close = 100.0 + i as f64 * 2.0 - 0.5;
            adx.update(high, low, close);
        }

        if let Some(dir) = adx.trend_direction() {
            assert_eq!(
                dir,
                TrendDirection::Bullish,
                "Should detect bullish direction in uptrend"
            );
        }
    }

    #[test]
    fn test_adx_di_values() {
        let mut adx = ADX::new(14);

        for i in 1..=50 {
            let high = 100.0 + i as f64 * 2.0;
            let low = 100.0 + i as f64 * 2.0 - 1.0;
            let close = 100.0 + i as f64 * 2.0 - 0.5;
            adx.update(high, low, close);
        }

        // In an uptrend, +DI should be higher than -DI
        if let (Some(plus), Some(minus)) = (adx.plus_di(), adx.minus_di()) {
            assert!(
                plus > minus,
                "+DI ({plus}) should be > -DI ({minus}) in uptrend"
            );
        }
    }

    #[test]
    fn test_adx_reset() {
        let mut adx = ADX::new(14);
        for i in 1..=50 {
            let base = 100.0 + i as f64;
            adx.update(base + 1.0, base - 1.0, base);
        }
        assert!(adx.is_ready());

        adx.reset();
        assert!(!adx.is_ready());
        assert!(adx.value().is_none());
        assert!(adx.plus_di().is_none());
        assert!(adx.minus_di().is_none());
    }

    // --- Bollinger Bands Tests ---

    #[test]
    fn test_bb_creation() {
        let bb = BollingerBands::new(20, 2.0);
        assert_eq!(bb.period(), 20);
        assert_eq!(bb.std_dev_multiplier(), 2.0);
        assert!(!bb.is_ready());
    }

    #[test]
    fn test_bb_warmup() {
        let mut bb = BollingerBands::new(20, 2.0);

        for i in 1..20 {
            let result = bb.update(100.0 + i as f64 * 0.1);
            assert!(result.is_none());
        }

        let result = bb.update(102.0);
        assert!(result.is_some());
        assert!(bb.is_ready());
    }

    #[test]
    fn test_bb_band_ordering() {
        let mut bb = BollingerBands::new(20, 2.0);

        for i in 1..=25 {
            let price = 100.0 + (i as f64 % 5.0);
            bb.update(price);
        }

        let result = bb.update(102.0).unwrap();
        assert!(
            result.upper > result.middle,
            "Upper band ({}) should be > middle ({})",
            result.upper,
            result.middle
        );
        assert!(
            result.middle > result.lower,
            "Middle ({}) should be > lower ({})",
            result.middle,
            result.lower
        );
    }

    #[test]
    fn test_bb_percent_b() {
        let mut bb = BollingerBands::new(20, 2.0);

        // Build some history with variance
        for i in 1..=20 {
            bb.update(100.0 + (i as f64 % 3.0));
        }

        // Price at middle should give %B near 0.5
        let values = bb.update(100.0 + 1.0);
        if let Some(v) = values {
            // %B should be between 0 and 1 for normal prices
            assert!(
                v.percent_b >= 0.0 && v.percent_b <= 1.0,
                "%B should be in [0,1]: {}",
                v.percent_b
            );
        }
    }

    #[test]
    fn test_bb_squeeze_detection() {
        let mut bb = BollingerBands::new(20, 2.0);

        // First, create wide bands with volatile data
        for i in 0..50 {
            let price = 100.0 + if i % 2 == 0 { 10.0 } else { -10.0 };
            bb.update(price);
        }

        // Then tighten with constant price
        for _ in 0..50 {
            bb.update(100.0);
        }

        let result = bb.update(100.0).unwrap();
        // After constant prices, width percentile should be low
        assert!(
            result.width_percentile < 50.0,
            "Constant prices should produce low width percentile: {}",
            result.width_percentile
        );
    }

    #[test]
    fn test_bb_overbought_oversold() {
        let mut bb = BollingerBands::new(20, 2.0);

        // Build history around 100
        for _ in 0..20 {
            bb.update(100.0);
        }

        // Price far above should be overbought
        let result = bb.update(110.0).unwrap();
        assert!(
            result.is_overbought(),
            "Price far above bands should be overbought, %B = {}",
            result.percent_b
        );
    }

    #[test]
    fn test_bb_reset() {
        let mut bb = BollingerBands::new(20, 2.0);
        for i in 0..25 {
            bb.update(100.0 + i as f64);
        }
        assert!(bb.is_ready());

        bb.reset();
        assert!(!bb.is_ready());
    }

    // --- RSI Tests ---

    #[test]
    fn test_rsi_creation() {
        let rsi = RSI::new(14);
        assert_eq!(rsi.period(), 14);
        assert!(!rsi.is_ready());
    }

    #[test]
    fn test_rsi_bullish_market() {
        let mut rsi = RSI::new(14);

        // Consistently rising prices
        let mut last_rsi = None;
        for i in 0..30 {
            let price = 100.0 + i as f64;
            if let Some(val) = rsi.update(price) {
                last_rsi = Some(val);
            }
        }

        if let Some(val) = last_rsi {
            assert!(
                val > 50.0,
                "RSI should be above 50 in bullish market: {val}"
            );
        }
    }

    #[test]
    fn test_rsi_bearish_market() {
        let mut rsi = RSI::new(14);

        // Consistently falling prices
        let mut last_rsi = None;
        for i in 0..30 {
            let price = 200.0 - i as f64;
            if let Some(val) = rsi.update(price) {
                last_rsi = Some(val);
            }
        }

        if let Some(val) = last_rsi {
            assert!(
                val < 50.0,
                "RSI should be below 50 in bearish market: {val}"
            );
        }
    }

    #[test]
    fn test_rsi_range() {
        let mut rsi = RSI::new(14);

        for i in 0..50 {
            let price = 100.0 + (i as f64 * 0.7).sin() * 10.0;
            if let Some(val) = rsi.update(price) {
                assert!(
                    (0.0..=100.0).contains(&val),
                    "RSI should be in [0, 100]: {val}"
                );
            }
        }
    }

    #[test]
    fn test_rsi_reset() {
        let mut rsi = RSI::new(14);
        for i in 0..30 {
            rsi.update(100.0 + i as f64);
        }
        assert!(rsi.is_ready());

        rsi.reset();
        assert!(!rsi.is_ready());
    }

    // --- SMA Helper Test ---

    #[test]
    fn test_calculate_sma() {
        assert_eq!(calculate_sma(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(calculate_sma(&[100.0]), 100.0);
        assert_eq!(calculate_sma(&[]), 0.0);
    }

    #[test]
    fn test_calculate_sma_precision() {
        let prices = vec![10.0, 20.0, 30.0];
        let sma = calculate_sma(&prices);
        assert!((sma - 20.0).abs() < f64::EPSILON);
    }
}
