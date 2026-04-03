//! JANUS Core - Shared types and state for all modules
//!
//! This library provides:
//! - Common types (Signal, Config, etc.)
//! - Shared application state
//! - Module interface traits
//! - Unified metrics
//! - Inter-module communication channels

pub mod checkpoint_notify;
pub mod config;
pub mod error;
pub mod logging;
pub mod market;
pub mod metrics;
pub mod optimized_params;
pub mod signal;
pub mod state;
pub mod supervisor;

// Re-exports for convenience
pub use checkpoint_notify::{
    CheckpointNotification, CheckpointNotifier, CheckpointNotifierConfig, checkpoint_channel,
};
pub use config::Config;
pub use error::{Error, Result};
pub use logging::{LoggingConfig, LoggingGuard, init_logging};
pub use market::{
    Exchange, FundingRateEvent, KlineEvent, LiquidationEvent, MarketDataBus, MarketDataEvent,
    MarketType, OrderBookEvent, PriceLevel, Side, Symbol, TickerEvent, TradeEvent,
};
pub use optimized_params::{OptimizedParams, ParamManager, ParamNotification};
pub use signal::{Signal, SignalBus, SignalType};
pub use state::{JanusState, LogLevelController, ServiceState};
pub use supervisor::{
    BackoffConfig, JanusService, JanusSupervisor, RestartPolicy, ServicePhase, SupervisorConfig,
    SupervisorMetrics,
};

/// Module interface trait - all modules must implement this
#[async_trait::async_trait]
pub trait Module: Send + Sync {
    /// Module name
    fn name(&self) -> &str;

    /// Start the module
    async fn start(&self, state: std::sync::Arc<JanusState>) -> Result<()>;

    /// Stop the module gracefully
    async fn stop(&self) -> Result<()>;

    /// Check if module is healthy
    async fn health_check(&self) -> bool;
}

/// Module metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModuleInfo {
    pub name: String,
    pub version: String,
    pub status: ModuleStatus,
    pub uptime_seconds: u64,
}

/// Module status
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ModuleStatus {
    Starting,
    Running,
    Stopping,
    Stopped,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_status() {
        let status = ModuleStatus::Running;
        assert_eq!(status, ModuleStatus::Running);
    }
}
