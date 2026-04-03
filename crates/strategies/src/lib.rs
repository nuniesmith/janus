//! Trading Strategies Module
//!
//! This module contains all trading strategy implementations for the JANUS
//! trading system.
//!
//! ## Strategies
//!
//! - **EMA Flip** ([`ema_flip`]) — 8/21 EMA Long-Short Flip Strategy with
//!   ATR-based stops. Trades pullbacks to the fast EMA in the direction of
//!   the slow EMA trend.
//!
//! - **Mean Reversion** ([`mean_reversion`]) — Bollinger Bands mean reversion
//!   strategy with RSI confirmation and ATR-based stop losses. Works best in
//!   ranging/mean-reverting market regimes (as detected by `janus-regime`).
//!
//! - **Bollinger Squeeze Breakout** ([`bollinger_squeeze`]) — Detects low-volatility
//!   squeeze periods via Bollinger Band width and generates breakout signals when
//!   price escapes the bands. Uses EMA trend + ADX strength for confirmation.
//!   Best at ranging → trending transitions.
//!
//! - **VWAP Scalper** ([`vwap_scalper`]) — Mean reversion scalping around the
//!   Volume-Weighted Average Price with standard deviation bands. Enters long at
//!   the lower VWAP band, short at the upper band, targeting the VWAP mean.
//!   Best in ranging/mean-reverting markets.
//!
//! - **Opening Range Breakout** ([`opening_range`]) — Defines the first N candles
//!   of a session as the "opening range" and trades breakouts above or below that
//!   range with volume confirmation. Targets 1–2× range extension. Best at regime
//!   transitions from consolidation to trend.
//!
//! - **EMA Ribbon Scalper** ([`ema_ribbon_scalper`]) — 8/13/21 EMA ribbon
//!   pullback scalper with volume confirmation. Enters when price pulls back to
//!   the ribbon in the direction of the trend. ATR-based stops. Best in trending
//!   markets. Ported from Kraken `EmaRibbonScalper`.
//!
//! - **Trend Pullback** ([`trend_pullback`]) — Fibonacci retracement entries
//!   within established trends, confirmed by RSI divergence and candlestick
//!   reversal patterns (pin bars, engulfing). Uses EMA 50/200 for trend detection.
//!   Best in trending markets. Ported from Kraken `TrendPullback`.
//!
//! - **Momentum Surge** ([`momentum_surge`]) — Detects sudden price surges
//!   accompanied by volume spikes and enters on the first pullback within the
//!   surge. Uses candle-count based lookback (adapted from Kraken's
//!   `chrono::DateTime` approach). Best in trending/volatile markets.
//!   Ported from Kraken `MomentumSurge`.
//!
//! - **Multi-Timeframe Trend** ([`multi_tf_trend`]) — Swing-timeframe EMA 50/200
//!   crossover with ADX trend strength and higher-timeframe alignment. Requires
//!   HTF trend bias to be set externally. Best in clearly trending markets.
//!   Ported from Kraken `MultiTimeframeTrend`.
//!
//! ## Regime-Aware Strategy Selection
//!
//! These strategies are designed to be gated by the regime detection system
//! in `janus-regime`. Use the router's `RoutedSignal.strategy` field to decide
//! which strategy to run on each tick:
//!
//! - `ActiveStrategy::TrendFollowing` → use EMA Flip, EMA Ribbon Scalper,
//!   Trend Pullback, Momentum Surge, or Multi-Timeframe Trend
//! - `ActiveStrategy::MeanReversion` → use Mean Reversion, Bollinger Squeeze,
//!   VWAP Scalper, or Opening Range Breakout
//! - `ActiveStrategy::NoTrade` → skip
//!
//! The Bollinger Squeeze strategy complements Mean Reversion: MR profits from
//! oscillations *within* a range, while BB Squeeze captures the *breakout*
//! when the range ends.
//!
//! The EMA Ribbon Scalper and Trend Pullback strategies complement the simple
//! EMA Flip: Ribbon Scalper uses a 3-EMA ribbon for higher-quality pullback
//! entries, while Trend Pullback adds Fibonacci retracement and candlestick
//! pattern confirmation for deeper pullback opportunities.

// Re-export dependencies so downstream code can reference them via this crate
// when needed (e.g. `crate::indicators`, `crate::models`).
pub use janus_indicators as indicators;
pub use janus_models as models;

// ============================================================================
// Strategy affinity & gating
// ============================================================================

pub mod affinity;
pub mod gating;

// ============================================================================
// Strategy modules
// ============================================================================

pub mod bollinger_squeeze;
pub mod ema_flip;
pub mod ema_ribbon_scalper;
pub mod mean_reversion;
pub mod momentum_surge;
pub mod multi_tf_trend;
pub mod opening_range;
pub mod trend_pullback;
pub mod vwap_scalper;

// ============================================================================
// Re-exports for convenience
// ============================================================================

pub use bollinger_squeeze::{
    SqueezeBreakoutConfig, SqueezeBreakoutResult, SqueezeBreakoutSignal, SqueezeBreakoutStrategy,
};
pub use ema_flip::{EmaCrossover, EmaFlipConfig, EmaFlipStrategy};
pub use ema_ribbon_scalper::{
    EmaRibbonConfig, EmaRibbonResult, EmaRibbonScalperStrategy, EmaRibbonSignal,
};
pub use mean_reversion::{
    MeanReversionConfig, MeanReversionResult, MeanReversionSignal, MeanReversionStrategy,
};
pub use momentum_surge::{
    MomentumSurgeConfig, MomentumSurgeResult, MomentumSurgeSignal, MomentumSurgeStrategy,
};
pub use multi_tf_trend::{MultiTfConfig, MultiTfResult, MultiTfSignal, MultiTfTrendStrategy};
pub use opening_range::{OrbConfig, OrbResult, OrbSignal, OrbStrategy};
pub use trend_pullback::{
    TrendPullbackConfig, TrendPullbackResult, TrendPullbackSignal, TrendPullbackStrategy,
};
pub use vwap_scalper::{VwapScalperConfig, VwapScalperResult, VwapScalperStrategy, VwapSignal};

// ============================================================================
// Backward-compatible simple EMA Flip (used by forward service event_loop.rs)
// ============================================================================

/// Signal types from EMA crossover.
///
/// This is the lightweight signal enum consumed by the forward service's event
/// loop. For the full-featured strategy with ATR stops and candle-level
/// analysis, see [`ema_flip::EmaFlipStrategy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Signal {
    Buy,
    Sell,
    Close,
    None,
}

/// Simple EMA Flip Strategy for 8/21 crossover.
///
/// Tracks the previous fast/slow EMA values and detects crossovers.
/// This is the lightweight version used directly by the forward service
/// event loop. For a richer implementation, see [`ema_flip::EmaFlipStrategy`].
pub struct EMAFlipStrategy {
    #[allow(dead_code)]
    fast_period: usize,
    #[allow(dead_code)]
    slow_period: usize,
    last_fast: Option<f64>,
    last_slow: Option<f64>,
}

impl EMAFlipStrategy {
    /// Create a new EMA flip strategy
    pub fn new(fast_period: usize, slow_period: usize) -> Self {
        Self {
            fast_period,
            slow_period,
            last_fast: None,
            last_slow: None,
        }
    }

    /// Check for crossover signal
    pub fn check_signal(&mut self, fast_ema: f64, slow_ema: f64) -> Signal {
        let signal = if let (Some(prev_fast), Some(prev_slow)) = (self.last_fast, self.last_slow) {
            // Bullish crossover: fast crosses above slow
            if prev_fast <= prev_slow && fast_ema > slow_ema {
                Signal::Buy
            }
            // Bearish crossover: fast crosses below slow
            else if prev_fast >= prev_slow && fast_ema < slow_ema {
                Signal::Sell
            } else {
                Signal::None
            }
        } else {
            Signal::None
        };

        // Update state for next check
        self.last_fast = Some(fast_ema);
        self.last_slow = Some(slow_ema);

        signal
    }

    /// Reset strategy state
    pub fn reset(&mut self) {
        self.last_fast = None;
        self.last_slow = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_creation() {
        let strategy = EMAFlipStrategy::new(8, 21);
        assert_eq!(strategy.fast_period, 8);
        assert_eq!(strategy.slow_period, 21);
    }

    #[test]
    fn test_bullish_crossover() {
        let mut strategy = EMAFlipStrategy::new(8, 21);

        // First call - no signal (need history)
        let signal = strategy.check_signal(50.0, 51.0);
        assert_eq!(signal, Signal::None);

        // Fast below slow
        let signal = strategy.check_signal(50.0, 51.0);
        assert_eq!(signal, Signal::None);

        // Bullish crossover - fast crosses above slow
        let signal = strategy.check_signal(52.0, 51.0);
        assert_eq!(signal, Signal::Buy);
    }

    #[test]
    fn test_bearish_crossover() {
        let mut strategy = EMAFlipStrategy::new(8, 21);

        // Initialize with fast above slow
        strategy.check_signal(52.0, 51.0);
        let signal = strategy.check_signal(52.0, 51.0);
        assert_eq!(signal, Signal::None);

        // Bearish crossover - fast crosses below slow
        let signal = strategy.check_signal(50.0, 51.0);
        assert_eq!(signal, Signal::Sell);
    }

    #[test]
    fn test_no_crossover() {
        let mut strategy = EMAFlipStrategy::new(8, 21);

        strategy.check_signal(50.0, 51.0);
        let signal = strategy.check_signal(49.0, 51.0);
        assert_eq!(signal, Signal::None); // Still below, no crossover
    }

    #[test]
    fn test_reset() {
        let mut strategy = EMAFlipStrategy::new(8, 21);

        strategy.check_signal(50.0, 51.0);
        strategy.reset();

        assert_eq!(strategy.last_fast, None);
        assert_eq!(strategy.last_slow, None);
    }
}
