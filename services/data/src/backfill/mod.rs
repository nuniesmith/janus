//! Backfill operations module
//!
//! This module contains functionality for backfilling historical data gaps,
//! including distributed locking to prevent concurrent backfill of the same gap.

pub mod executor;
pub mod gap_integration;
pub mod historical_candles;
pub mod indicator_warmup;
pub mod lock;
pub mod python_data_client;
pub mod scheduler;
pub mod signal_backtest;
pub mod throttle;

pub use executor::{BackfillExecutor, BackfillRequest, BackfillResult, Trade};
pub use gap_integration::{
    GapIntegrationConfig, GapIntegrationManager, IntegrationStats, estimate_gap_trades,
};
pub use historical_candles::{FetchConfig, FetchResult, HistoricalCandleFetcher, deep_warmup};
pub use indicator_warmup::{IndicatorWarmup, WarmupConfig, WarmupResult};
pub use lock::{BackfillLock, LockConfig, LockGuard, LockMetrics};
pub use python_data_client::{
    AssetInfo, AssetsResponse, BarStatusEntry, BarsStatusResponse, CandleFetchRequest,
    DataSourceResponse, FetchCandlesResult, FillJobResponse, GapEntry, GapReportResponse,
    LiveFeedStatus, PythonDataClient, PythonDataClientConfig, SymbolsResponse,
    to_data_service_symbol,
};
pub use scheduler::{BackfillScheduler, GapInfo, SchedulerConfig, SchedulerStats};
pub use signal_backtest::{BacktestConfig, BacktestResults, SignalBacktest, SignalTypeMetrics};
pub use throttle::{
    BackfillThrottle, DiskUsage, ThrottleConfig, ThrottleError, calculate_batches, get_disk_usage,
    should_batch,
};
