//! Execution engine module for the FKS Execution Service
//!
//! This module provides different execution engines for various trading modes:
//! - Simulated: For backtesting and development
//! - Paper: For paper trading with real market data
//! - Live: For live trading on exchanges (currently supports Kraken)

pub mod arbitrage;
pub mod arbitrage_bridge;
pub mod best_execution;
pub mod histogram;
pub mod kraken;
pub mod metrics;
pub mod retry;
pub mod signal_flow;
pub mod simulated;

use crate::config::ExecutionMode;
use crate::error::Result;
use crate::types::Order;

/// Trait for execution engines
#[async_trait::async_trait]
pub trait ExecutionEngine: Send + Sync {
    /// Submit an order for execution
    async fn submit_order(&self, order: Order) -> Result<String>;

    /// Cancel an order
    async fn cancel_order(&self, order_id: &str) -> Result<()>;

    /// Get order status
    fn get_order(&self, order_id: &str) -> Result<Order>;

    /// Get all active orders
    fn get_active_orders(&self) -> Vec<Order>;

    /// Get execution mode
    fn mode(&self) -> ExecutionMode;
}

/// Factory for creating execution engines
pub struct ExecutionEngineFactory;

impl ExecutionEngineFactory {
    /// Create an execution engine based on the mode
    pub fn create(
        mode: ExecutionMode,
        config: &crate::config::Config,
    ) -> Result<Box<dyn ExecutionEngine>> {
        match mode {
            ExecutionMode::Simulated => {
                let sim_config = config.simulation.as_ref().ok_or_else(|| {
                    crate::error::ExecutionError::Config(
                        "Simulation config required for simulated mode".to_string(),
                    )
                })?;
                Ok(Box::new(SimulatedExecutionEngine::new(sim_config.clone())))
            }
            ExecutionMode::Paper => {
                // Paper trading uses Kraken in dry-run mode
                let mut kraken_config = kraken::KrakenExecutionConfig::from_env();
                kraken_config.dry_run = true; // Force dry run for paper trading
                Ok(Box::new(kraken::KrakenExecutionEngine::new(kraken_config)))
            }
            ExecutionMode::Live => {
                // Live trading on Kraken
                let kraken_config = kraken::KrakenExecutionConfig::from_env();

                // Safety check: ensure we have credentials for live trading
                if !kraken_config.rest_config.has_credentials() {
                    return Err(crate::error::ExecutionError::Config(
                        "Kraken API credentials required for live trading mode".to_string(),
                    ));
                }

                // Warn if dry_run is enabled in live mode (might be intentional)
                if kraken_config.dry_run {
                    tracing::warn!(
                        "Live trading mode with dry_run enabled - orders will be validated but not executed"
                    );
                }

                Ok(Box::new(kraken::KrakenExecutionEngine::new(kraken_config)))
            }
        }
    }

    /// Create a Kraken execution engine directly
    pub fn create_kraken(
        config: kraken::KrakenExecutionConfig,
    ) -> Result<kraken::KrakenExecutionEngine> {
        Ok(kraken::KrakenExecutionEngine::new(config))
    }

    /// Create a simulated execution engine directly
    pub fn create_simulated(
        config: crate::config::SimulationConfig,
    ) -> Result<SimulatedExecutionEngine> {
        Ok(SimulatedExecutionEngine::new(config))
    }
}

/// Simulated execution engine wrapper
pub struct SimulatedExecutionEngine {
    inner: simulated::SimulatedExecutor,
}

impl SimulatedExecutionEngine {
    pub fn new(config: crate::config::SimulationConfig) -> Self {
        Self {
            inner: simulated::SimulatedExecutor::new(config),
        }
    }

    /// Get the underlying simulated executor for additional operations
    pub fn executor(&self) -> &simulated::SimulatedExecutor {
        &self.inner
    }
}

#[async_trait::async_trait]
impl ExecutionEngine for SimulatedExecutionEngine {
    async fn submit_order(&self, order: Order) -> Result<String> {
        self.inner.submit_order(order).await
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        self.inner.cancel_order(order_id).await
    }

    fn get_order(&self, order_id: &str) -> Result<Order> {
        self.inner.get_order(order_id)
    }

    fn get_active_orders(&self) -> Vec<Order> {
        self.inner.get_active_orders()
    }

    fn mode(&self) -> ExecutionMode {
        ExecutionMode::Simulated
    }
}

// Re-export key types for convenience
pub use arbitrage::{
    ArbitrageConfig, ArbitrageEvent, ArbitrageMonitor, ArbitrageOpportunity, ArbitrageSummary,
    Exchange, ExchangePrice, KrakenRecommendation, PriceComparison, RecommendedAction, SpreadStats,
};
pub use best_execution::{
    BestExecutionAnalyzer, ExecutionAnalysis, ExecutionConfig, ExecutionQualityMetrics,
    ExecutionRecommendation, ExecutionRecord, OrderSide as BestExecOrderSide, PriceTrend,
};
pub use kraken::{KrakenExecutionConfig, KrakenExecutionEngine, KrakenExecutionStats};
pub use retry::{
    CircuitBreaker, CircuitBreakerConfig, CircuitState, RetryBuilder, RetryConfig, RetryMetrics,
    retry, retry_with_backoff, retry_with_circuit_breaker,
};
pub use signal_flow::{
    ExecutionUpdate, SignalFlowConfig, SignalFlowCoordinator, SignalFlowStats, SignalResponse,
    SignalSide, SignalType, TradingSignal,
};
pub use simulated::{SimulatedExecutor, SimulatorStats};

// Metrics exports
pub use metrics::{
    ArbitrageMetrics, BestExecutionMetrics, CircuitBreakerState, ExecutionMetricsRegistry,
    RecommendationType, RetryMetrics as ExecutionRetryMetrics, SignalFlowMetrics,
    arbitrage_metrics, best_execution_metrics, execution_prometheus_metrics,
    global_execution_metrics, retry_metrics, signal_flow_metrics, unified_prometheus_metrics,
};
