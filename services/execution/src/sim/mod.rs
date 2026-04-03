//! # FKS Simulation Environment
//!
//! Unified simulation framework for backtesting, forward-testing, and live trading.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                        Trading Environment                               │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
//! │  │  Backtest    │  │ Forward Test │  │    Live      │                  │
//! │  │  (Historical)│  │ (Paper/Sim)  │  │  (Kraken)    │                  │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                  │
//! │         │                 │                 │                           │
//! │         └─────────────────┼─────────────────┘                           │
//! │                           │                                              │
//! │                    ┌──────▼──────┐                                      │
//! │                    │  Unified    │                                      │
//! │                    │  DataFeed   │                                      │
//! │                    └──────┬──────┘                                      │
//! │                           │                                              │
//! │  ┌────────────────────────┼────────────────────────┐                   │
//! │  │                        │                        │                    │
//! │  ▼                        ▼                        ▼                    │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │
//! │  │  Historical  │  │    Live      │  │   Recorded   │                  │
//! │  │  Parquet/CSV │  │  WebSocket   │  │   QuestDB    │                  │
//! │  └──────────────┘  └──────────────┘  └──────────────┘                  │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ### Backtesting with Historical Data
//!
//! ```rust,ignore
//! use janus_execution::sim::{SimEnvironment, SimConfig, DataSource};
//!
//! let config = SimConfig::backtest()
//!     .with_data_source(DataSource::Parquet("data/btcusdt_2024.parquet"))
//!     .with_initial_balance(10_000.0)
//!     .with_slippage_bps(5.0)
//!     .with_commission_bps(6.0);
//!
//! let mut env = SimEnvironment::new(config).await?;
//! let results = env.run_backtest(strategy).await?;
//! ```
//!
//! ### Forward Testing with Live Data
//!
//! ```rust,ignore
//! let config = SimConfig::forward_test()
//!     .with_exchanges(vec!["kraken", "binance", "bybit"])
//!     .with_symbols(vec!["BTC/USDT", "ETH/USDT"])
//!     .with_record_data(true); // Record for later replay
//!
//! let mut env = SimEnvironment::new(config).await?;
//! env.run_forward_test(strategy, Duration::from_hours(24)).await?;
//! ```
//!
//! ### Live Trading (Kraken)
//!
//! ```rust,ignore
//! let config = SimConfig::live()
//!     .with_exchange("kraken")
//!     .with_api_credentials(api_key, api_secret);
//!
//! let mut env = SimEnvironment::new(config).await?;
//! env.run_live(strategy).await?;
//! ```

pub mod config;
pub mod data_feed;
pub mod data_recorder;
pub mod environment;
pub mod live_feed_bridge;
pub mod local_fallback;
pub mod metrics;
pub mod metrics_server;
pub mod optimization;
pub mod questdb_backtest;
pub mod replay;

// Re-export main types from config
pub use config::{DataSource, SimConfig, SimMode};

// Re-export main types from data_feed
pub use data_feed::{
    AggregatedDataFeed, DataFeed, DataFeedStats, MarketEvent, TickData, TradeData,
};

// Re-export data recorder
pub use data_recorder::{DataRecorder, RecorderConfig, RecorderError, RecorderStats};

// Re-export metrics
pub use metrics::{
    BridgeMetrics, RecorderMetrics, SimMetricsExporter, global_sim_metrics, sim_prometheus_metrics,
    start_metrics_collector, update_bridge_stats, update_recorder_stats,
};

// Re-export metrics server
pub use metrics_server::{
    MetricsServerError, SimMetricsServer, SimMetricsServerBuilder, SimMetricsServerConfig,
    SimMetricsServerHandle, start_sim_metrics_server, start_sim_metrics_server_with_components,
};

// Re-export local fallback types
pub use local_fallback::{
    FallbackError, FallbackFormat, FallbackStats, LocalFallbackConfig, LocalFallbackWriter,
    ReplayStats as FallbackReplayStats, SerializableEvent, replay_fallback_to_questdb,
};

// Re-export QuestDB backtest types
pub use questdb_backtest::{
    BacktestState, EventBasedStrategy, Position as BacktestPosition, QuestDBBacktestConfig,
    QuestDBBacktestError, QuestDBStrategyEvaluator, QuestDBWalkForwardRunner,
    QuestDBWalkForwardRunnerBuilder, SignalType, SimulatedTrade,
};

// Re-export environment types
pub use environment::{
    Account, OrderSide, Position, Signal, SimEnvironment, SimError, SimResult, Strategy,
    TradeExecution,
};

// Re-export optimization types
pub use optimization::{
    OptimizationConfig, OptimizationDirection, OptimizationError, OptimizationMetric,
    OptimizationResult, OptimizationRunResult, ParameterRange, ParameterSet, ParameterValue,
    StrategyEvaluator, WalkForwardAnalysis, WalkForwardBacktestBuilder, WalkForwardBacktestRunner,
    WalkForwardConfig, WalkForwardResult, WalkForwardWindow,
};

// Re-export replay types
pub use replay::{
    EventTypeFilter, QuestDBLoaderConfig, ReplayConfig, ReplayEngine, ReplayError, ReplaySpeed,
    ReplayState, ReplayStats,
};

// Re-export live feed bridge types
pub use live_feed_bridge::{
    LiveFeedBridge, LiveFeedBridgeConfig, LiveFeedBridgeError, LiveFeedBridgeStats,
};

/// Prelude for common imports
pub mod prelude {
    pub use super::config::{DataSource, SimConfig, SimMode};
    pub use super::data_feed::{DataFeed, MarketEvent, TickData, TradeData};
    pub use super::data_recorder::DataRecorder;
    pub use super::environment::{SimEnvironment, SimResult, Strategy};
    pub use super::live_feed_bridge::{LiveFeedBridge, LiveFeedBridgeConfig};
    pub use super::optimization::{
        OptimizationConfig, OptimizationResult, ParameterRange, WalkForwardAnalysis,
    };
    pub use super::replay::{ReplayConfig, ReplayEngine, ReplaySpeed};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_imports() {
        // Smoke test to ensure all modules compile
        let _ = std::mem::size_of::<SimMode>();
    }
}
