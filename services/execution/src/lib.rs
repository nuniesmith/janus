//! FKS Execution Service Library
//!
//! This library provides order execution, position management, and risk controls
//! for the FKS trading system. It supports multiple execution modes:

// ── Clippy suppressions ────────────────────────────────────────────────
// Only genuinely necessary suppressions are kept here.

// Structural — too many arguments in exchange/execution APIs
#![allow(clippy::too_many_arguments)]
// `.map_or(true/false, ...)` used in query filters (e.g. gRPC get_positions)
#![allow(clippy::unnecessary_map_or)]
// Sequential validation checks kept uncollapsed for clarity in order/risk code
#![allow(clippy::collapsible_if)]
// Single-arm matches on websocket messages kept for future message types
#![allow(clippy::single_match)]
// Test code extensively uses `let mut s = T::default(); s.field = val;`
#![allow(clippy::field_reassign_with_default)]
// Some Default impls delegate to Self::new() for consistency
#![allow(clippy::derivable_impls)]
//!
//! - **Simulated**: In-memory execution for backtesting and development
//! - **Paper**: Live market data with simulated execution
//! - **Live**: Real execution on live exchanges
//!
//! # Example
//!
//! ```no_run
//! use janus_execution::{Config, ExecutionEngineFactory};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = Config::from_env()?;
//!     config.validate()?;
//!
//!     let engine = ExecutionEngineFactory::create(
//!         config.execution_mode.mode,
//!         &config
//!     )?;
//!
//!     // Use the engine...
//!     Ok(())
//! }
//! ```

// Re-export protobuf types from centralized fks-proto crate
pub mod generated {
    pub use fks_proto::execution as fks_execution;

    // Re-export at the expected path for backwards compatibility
    pub mod fks {
        pub mod execution {
            pub mod v1 {
                pub use fks_proto::execution::*;
                // NOTE: Side, OrderType, TimeInForce, HealthCheckRequest, and
                // HealthCheckResponse now live in fks_proto::common.  Consumers
                // should import them directly from fks_proto::common::{…}.
            }
        }
    }
}

// Core modules
pub mod config;
pub mod error;
pub mod types;

// Kill switch guard (defense-in-depth for order submission)
pub mod kill_switch_guard;

// Execution engines
pub mod execution;

// Exchange adapters
pub mod exchanges;

// Order management
pub mod orders;

// Position and account management
pub mod positions;

// Execution strategies
pub mod strategies;

// Notifications
pub mod notifications;

// Alertmanager client for pushing trade signals
pub mod alertmanager;

// API modules
pub mod api;

// Simulation environment
pub mod sim;

// State broadcasting (for eliminating HTTP blocking)
pub mod state_broadcaster;

// Registry sync (pull routing rules from the registry service)
pub mod registry_sync;

// Re-exports for convenience
pub use alertmanager::{
    AccountType as AlertAccountType, AlertmanagerAlert, AlertmanagerClient, AlertmanagerError,
    SignalDirection as AlertSignalDirection, TradeSignalAlert,
};
pub use api::{ExecutionServiceImpl, HttpState, create_http_router};
pub use config::{Config, ExecutionMode, KrakenConfig, SimulationConfig};
pub use error::{ExecutionError, Result};
pub use exchanges::kraken::{
    BalanceUpdateEvent, ExecutionEvent, ExecutionType, FillTrackerConfig, KrakenBalance,
    KrakenFillTracker, KrakenOrderResult, KrakenPrivateWebSocket, KrakenPrivateWsConfig,
    KrakenRestClient, KrakenRestConfig, OrderStatusEvent, PrivateWsEvent, ReconciliationResult,
};
pub use exchanges::{
    Balance, Exchange, OrderStatusResponse, OrderUpdate, PositionUpdate, bybit::BybitExchange,
    rate_limit::RateLimiter, router::ExchangeRouter,
};
pub use execution::{
    ExecutionEngine, ExecutionEngineFactory, ExecutionUpdate, KrakenExecutionConfig,
    KrakenExecutionEngine, KrakenExecutionStats, SignalFlowConfig, SignalFlowCoordinator,
    SignalFlowStats, SignalResponse, SignalSide, SignalType, SimulatedExecutionEngine,
    TradingSignal,
};
pub use kill_switch_guard::{
    AtomicOrderGate, KillSwitchGuard, KillSwitchGuardConfig, NoOpKillSwitchGuard, OrderGate,
};
pub use notifications::{DiscordNotifier, NotificationManager};
pub use orders::{
    OrderHistory, OrderManager, OrderStateTransition, OrderTracker, OrderValidator,
    ValidationConfig, ValidationResult,
};
pub use positions::{
    AccountManager, AccountStats, Balance as AccountBalance, MarginAccount,
    Position as PositionInfo, PositionSide as PositionDirection, PositionStats, PositionTracker,
};
pub use strategies::{
    IcebergConfig, IcebergExecutor, IcebergState, IcebergStatus, TwapConfig, TwapExecutor,
    TwapState, TwapStatus, VolumeBucket, VwapConfig, VwapExecutor, VwapState, VwapStatus,
};
pub use types::{
    Account, Fill, Order, OrderSide, OrderStatusEnum, OrderTypeEnum, Position, PositionSide,
};

// State broadcasting exports
pub use state_broadcaster::{
    BroadcasterConfig, ExecutionState, SharedExecutionState, StateBroadcaster, VolatilityEstimate,
    VolatilityRegime,
};

// Re-export simulation environment
pub use sim::{
    AggregatedDataFeed, DataFeed, DataRecorder, MarketEvent, OptimizationConfig,
    OptimizationResult, ParameterRange, ReplayConfig, ReplayEngine, ReplaySpeed, SimConfig,
    SimEnvironment, SimMode, SimResult, Strategy, TickData, TradeData, WalkForwardAnalysis,
    WalkForwardConfig, WalkForwardResult,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Library name
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_name() {
        assert_eq!(NAME, "janus-execution");
    }
}
