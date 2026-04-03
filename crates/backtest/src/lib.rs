//! # Janus Backtest - Temporal Fortress Backtesting Engine
//!
//! Zero-lookahead backtesting framework for Project JANUS.
//!
//! ## Features
//!
//! - **Temporal Fortress**: Double-buffered state machine preventing lookahead bias
//! - **Event-Driven Replay**: Tick-by-tick or bar-by-bar simulation
//! - **Vectorized Indicators**: High-performance batch calculations using Polars
//! - **Realistic Execution**: Slippage, commission, and partial fill simulation
//! - **Prop Firm Compliance**: Integrated Sheriff validation
//! - **Performance Metrics**: Sharpe ratio, drawdown, win rate, profit factor, etc.
//!
//! ## Architecture
//!
//! ```text
//! Historical Data → Data Loader → Temporal Fortress → Replay Engine → Metrics
//!                                        ↓
//!                                  Zero Lookahead
//!                                        ↓
//!                      Strategy ← Indicators ← Market State
//! ```
//!
//! ## Usage
//!
//! ### Basic Backtest
//!
//! ```rust,ignore
//! use janus_backtest::*;
//! use janus_strategies::EMAFlipStrategy;
//!
//! // Load historical data
//! let config = DataLoaderConfig::new("data/btcusdt_2024.parquet")
//!     .with_symbol("BTCUSD".to_string())
//!     .with_time_range(start_date, end_date);
//!
//! let loader = DataLoader::new(config);
//! let ticks = loader.load()?;
//!
//! // Configure backtest
//! let replay_config = ReplayConfig {
//!     initial_balance: 10_000.0,
//!     symbol: "BTCUSD".to_string(),
//!     slippage_bps: 5.0,
//!     commission_bps: 6.0,
//!     ..Default::default()
//! };
//!
//! // Run backtest
//! let mut engine = ReplayEngine::new(replay_config);
//! let strategy = EMAFlipStrategy::new(8, 21);
//! let metrics = engine.run(ticks, strategy)?;
//!
//! println!("Total Return: {:.2}%", metrics.total_return_pct);
//! println!("Win Rate: {:.2}%", metrics.win_rate);
//! println!("Max Drawdown: {:.2}%", metrics.max_drawdown_pct);
//! println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
//! ```
//!
//! ### Vectorized Analysis
//!
//! For large-scale backtests across multiple parameters:
//!
//! ```rust,ignore
//! use janus_backtest::vectorized::VectorizedIndicators;
//! use polars::prelude::*;
//!
//! let df = loader.load_dataframe()?;
//!
//! let enriched = VectorizedIndicators::new(df)
//!     .add_ema("price", 8, "ema_8")
//!     .add_ema("price", 21, "ema_21")
//!     .add_atr("high", "low", "close", 14, "atr_14")
//!     .add_rsi("price", 14, "rsi_14")
//!     .add_macd("price", 12, 26, 9, "macd", "signal", "histogram")
//!     .compute()?;
//!
//! // Now run custom analysis on enriched DataFrame
//! ```
//!
//! ## Zero-Lookahead Guarantee
//!
//! The temporal fortress ensures that at any point in time `t`, only data
//! with timestamps `<= t` is accessible to the strategy. Attempting to access
//! future data results in a `LookaheadViolation` error.
//!
//! This is critical for avoiding overfitted strategies that would fail in live trading.
//!
//! ## Performance
//!
//! - **Event-driven replay**: ~100k ticks/sec (single-threaded)
//! - **Vectorized indicators**: 10-100x faster than incremental for batch processing
//! - **Memory efficient**: Configurable lookback windows prevent unbounded growth
//!
//! ## Testing Strategies
//!
//! The backtest engine accepts any type implementing the `Strategy` trait:
//!
//! ```rust,ignore
//! use janus_strategies::{Strategy, Signal};
//!
//! struct MyStrategy {
//!     // ... fields
//! }
//!
//! impl Strategy for MyStrategy {
//!     fn generate_signal(&mut self, ticks: &[&Tick]) -> Result<Signal, Box<dyn std::error::Error>> {
//!         // Your logic here
//!         Ok(Signal::None)
//!     }
//! }
//! ```

pub mod data_loader;
pub mod fortress;
pub mod ohlcv_loader;
pub mod replay;
pub mod strategy_backtester;
pub mod vectorized;

// Re-export main types
pub use data_loader::{DataLoader, DataLoaderConfig, DataLoaderError, Side, Tick};
pub use fortress::{FortressError, FortressStats, TemporalFortress, TimestampedEvent};
pub use ohlcv_loader::{
    DataSummary, OhlcvColumnMap, OhlcvLoader, OhlcvLoaderConfig, OhlcvLoaderError, TimestampFormat,
    load_binance_csv, load_kraken_csv, load_ohlcv_csv, load_ohlcv_parquet,
};
pub use replay::{BacktestMetrics, ReplayConfig, ReplayEngine, ReplayError, Trade};
pub use strategy_backtester::{
    BacktestReport, BacktestSignal, CompletedTrade, Direction, ExitReason, OhlcvBar,
    StrategyBacktester, StrategyBacktesterConfig, StrategyId, StrategyMetrics,
    SyntheticDataGenerator,
};
pub use vectorized::{VectorizedError, VectorizedIndicators};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crate_imports() {
        // Smoke test to ensure all modules are accessible
        let _ = std::mem::size_of::<DataLoader>();
        let _ = std::mem::size_of::<TemporalFortress<Tick>>();
        let _ = std::mem::size_of::<ReplayEngine>();
    }
}
