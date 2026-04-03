//! Arbitrage Bridge Module
//!
//! This module provides a bridge between the `MarketDataAggregator` and `ArbitrageMonitor`,
//! automatically converting market data events (tickers) into arbitrage price updates.
//!
//! # Overview
//!
//! The `ArbitrageBridge` listens to market data events from all connected exchanges
//! and feeds them to the `ArbitrageMonitor` for cross-exchange price comparison and
//! arbitrage opportunity detection.
//!
//! # Example
//!
//! ```ignore
//! use janus_execution::execution::arbitrage_bridge::ArbitrageBridge;
//! use janus_execution::execution::arbitrage::{ArbitrageMonitor, ArbitrageConfig};
//! use janus_execution::exchanges::provider::MarketDataAggregator;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let aggregator = Arc::new(MarketDataAggregator::new());
//!     let monitor = Arc::new(ArbitrageMonitor::new(ArbitrageConfig::default()));
//!
//!     let bridge = ArbitrageBridge::new(aggregator.clone(), monitor.clone());
//!     bridge.start().await;
//!
//!     // Bridge now automatically forwards ticker updates to the arbitrage monitor
//!     Ok(())
//! }
//! ```

use crate::error::Result;
use crate::exchanges::market_data::{ExchangeId, MarketDataEvent, Ticker};
use crate::exchanges::provider::MarketDataAggregator;
use crate::execution::arbitrage::{ArbitrageMonitor, Exchange, ExchangePrice};
use crate::execution::best_execution::BestExecutionAnalyzer;
use crate::execution::metrics::{ArbitrageMetrics, arbitrage_metrics};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Bridge between MarketDataAggregator and ArbitrageMonitor
///
/// This struct manages the connection between market data feeds and the arbitrage
/// detection system, handling the conversion of market data events to arbitrage
/// price updates.
pub struct ArbitrageBridge {
    /// Market data aggregator providing ticker updates
    aggregator: Arc<MarketDataAggregator>,

    /// Arbitrage monitor to receive price updates
    monitor: Arc<ArbitrageMonitor>,

    /// Optional best execution analyzer to also receive price updates
    best_execution: Option<Arc<BestExecutionAnalyzer>>,

    /// Running state
    is_running: Arc<AtomicBool>,

    /// Task handle for the bridge loop
    task_handle: Arc<RwLock<Option<tokio::task::JoinHandle<()>>>>,

    /// Prometheus metrics
    metrics: Arc<ArbitrageMetrics>,
}

impl ArbitrageBridge {
    /// Create a new arbitrage bridge
    pub fn new(aggregator: Arc<MarketDataAggregator>, monitor: Arc<ArbitrageMonitor>) -> Self {
        Self {
            aggregator,
            monitor,
            best_execution: None,
            is_running: Arc::new(AtomicBool::new(false)),
            task_handle: Arc::new(RwLock::new(None)),
            metrics: arbitrage_metrics(),
        }
    }

    /// Create a new arbitrage bridge with best execution analyzer
    pub fn with_best_execution(
        aggregator: Arc<MarketDataAggregator>,
        monitor: Arc<ArbitrageMonitor>,
        best_execution: Arc<BestExecutionAnalyzer>,
    ) -> Self {
        Self {
            aggregator,
            monitor,
            best_execution: Some(best_execution),
            is_running: Arc::new(AtomicBool::new(false)),
            task_handle: Arc::new(RwLock::new(None)),
            metrics: arbitrage_metrics(),
        }
    }

    /// Check if the bridge is running
    pub fn is_running(&self) -> bool {
        self.is_running.load(Ordering::SeqCst)
    }

    /// Start the bridge
    ///
    /// This spawns a background task that listens to market data events
    /// and forwards them to the arbitrage monitor.
    pub async fn start(&self) -> Result<()> {
        if self.is_running.load(Ordering::SeqCst) {
            warn!("Arbitrage bridge is already running");
            return Ok(());
        }

        info!("Starting arbitrage bridge");
        self.is_running.store(true, Ordering::SeqCst);

        let mut event_rx = self.aggregator.subscribe_events();
        let monitor = self.monitor.clone();
        let best_execution = self.best_execution.clone();
        let is_running = self.is_running.clone();
        let metrics = self.metrics.clone();

        let handle = tokio::spawn(async move {
            info!("Arbitrage bridge task started");

            while is_running.load(Ordering::SeqCst) {
                match event_rx.recv().await {
                    Ok(event) => {
                        if let MarketDataEvent::Ticker(ticker) = event {
                            // Convert to ExchangePrice and update monitor
                            if let Some(price) = Self::ticker_to_exchange_price(&ticker) {
                                let symbol = price.symbol.clone();
                                let exchange_name = format!("{}", price.exchange);

                                // Record price update metric
                                metrics.record_price_update(&exchange_name);

                                monitor.update_price(price.clone()).await;

                                // Check for arbitrage opportunities and record metrics
                                if let Some(comparison) = monitor.get_comparison(&symbol).await {
                                    // Record Kraken deviation if available
                                    if let Some(deviation) = comparison.kraken_deviation_pct {
                                        metrics.update_kraken_deviation(&symbol, deviation);
                                    }

                                    // Check for arbitrage opportunity
                                    if let Some(opp) =
                                        comparison.find_arbitrage(rust_decimal::Decimal::ZERO)
                                    {
                                        metrics.record_opportunity(
                                            &opp.symbol,
                                            opp.spread_pct,
                                            opp.actionable,
                                        );

                                        // Record alert if spread is significant
                                        if opp.spread_pct > rust_decimal::Decimal::from(50) {
                                            metrics.record_alert();
                                        }
                                    }

                                    // Also update best execution analyzer if present
                                    if let Some(ref analyzer) = best_execution {
                                        analyzer.update_prices(comparison).await;
                                    }
                                }

                                debug!(
                                    "Forwarded ticker update: {} {} bid={} ask={}",
                                    price.exchange, price.symbol, price.bid, price.ask
                                );
                            }
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => {
                        warn!("Market data event channel closed");
                        break;
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                        warn!("Arbitrage bridge lagged by {} messages", n);
                        // Continue processing
                    }
                }
            }

            info!("Arbitrage bridge task stopped");
        });

        *self.task_handle.write().await = Some(handle);
        Ok(())
    }

    /// Stop the bridge
    pub async fn stop(&self) {
        info!("Stopping arbitrage bridge");
        self.is_running.store(false, Ordering::SeqCst);

        if let Some(handle) = self.task_handle.write().await.take() {
            handle.abort();
        }
    }

    /// Convert a Ticker to an ExchangePrice
    fn ticker_to_exchange_price(ticker: &Ticker) -> Option<ExchangePrice> {
        let exchange = Self::exchange_id_to_exchange(ticker.exchange)?;

        Some(ExchangePrice::new(
            exchange,
            ticker.symbol.clone(),
            ticker.bid,
            ticker.ask,
            ticker.bid_qty,
            ticker.ask_qty,
        ))
    }

    /// Convert ExchangeId to Exchange enum used by arbitrage module
    fn exchange_id_to_exchange(id: ExchangeId) -> Option<Exchange> {
        match id {
            ExchangeId::Kraken => Some(Exchange::Kraken),
            ExchangeId::Bybit => Some(Exchange::Bybit),
            ExchangeId::Binance => Some(Exchange::Binance),
        }
    }
}

/// Builder for ArbitrageBridge with fluent API
pub struct ArbitrageBridgeBuilder {
    aggregator: Option<Arc<MarketDataAggregator>>,
    monitor: Option<Arc<ArbitrageMonitor>>,
    best_execution: Option<Arc<BestExecutionAnalyzer>>,
}

impl ArbitrageBridgeBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            aggregator: None,
            monitor: None,
            best_execution: None,
        }
    }

    /// Set the market data aggregator
    pub fn aggregator(mut self, aggregator: Arc<MarketDataAggregator>) -> Self {
        self.aggregator = Some(aggregator);
        self
    }

    /// Set the arbitrage monitor
    pub fn monitor(mut self, monitor: Arc<ArbitrageMonitor>) -> Self {
        self.monitor = Some(monitor);
        self
    }

    /// Set the best execution analyzer (optional)
    pub fn best_execution(mut self, analyzer: Arc<BestExecutionAnalyzer>) -> Self {
        self.best_execution = Some(analyzer);
        self
    }

    /// Build the bridge
    pub fn build(self) -> Result<ArbitrageBridge> {
        let aggregator = self.aggregator.ok_or_else(|| {
            crate::error::ExecutionError::Config("MarketDataAggregator is required".into())
        })?;

        let monitor = self.monitor.ok_or_else(|| {
            crate::error::ExecutionError::Config("ArbitrageMonitor is required".into())
        })?;

        Ok(ArbitrageBridge {
            aggregator,
            monitor,
            best_execution: self.best_execution,
            is_running: Arc::new(AtomicBool::new(false)),
            task_handle: Arc::new(RwLock::new(None)),
            metrics: arbitrage_metrics(),
        })
    }
}

impl Default for ArbitrageBridgeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::arbitrage::ArbitrageConfig;
    use rust_decimal::Decimal;

    #[test]
    fn test_exchange_id_conversion() {
        assert_eq!(
            ArbitrageBridge::exchange_id_to_exchange(ExchangeId::Kraken),
            Some(Exchange::Kraken)
        );
        assert_eq!(
            ArbitrageBridge::exchange_id_to_exchange(ExchangeId::Bybit),
            Some(Exchange::Bybit)
        );
        assert_eq!(
            ArbitrageBridge::exchange_id_to_exchange(ExchangeId::Binance),
            Some(Exchange::Binance)
        );
    }

    #[test]
    fn test_ticker_to_exchange_price() {
        let ticker = Ticker {
            exchange: ExchangeId::Kraken,
            symbol: "BTC/USDT".to_string(),
            bid: Decimal::from(50000),
            bid_qty: Decimal::from(1),
            ask: Decimal::from(50010),
            ask_qty: Decimal::from(2),
            last: Decimal::from(50005),
            volume_24h: Decimal::from(1000),
            high_24h: Decimal::from(51000),
            low_24h: Decimal::from(49000),
            change_24h: Decimal::from(500),
            change_pct_24h: Decimal::from(1),
            vwap: None,
            timestamp: chrono::Utc::now(),
        };

        let price = ArbitrageBridge::ticker_to_exchange_price(&ticker).unwrap();

        assert_eq!(price.exchange, Exchange::Kraken);
        assert_eq!(price.symbol, "BTC/USDT");
        assert_eq!(price.bid, Decimal::from(50000));
        assert_eq!(price.ask, Decimal::from(50010));
        assert_eq!(price.bid_qty, Decimal::from(1));
        assert_eq!(price.ask_qty, Decimal::from(2));
    }

    #[tokio::test]
    async fn test_bridge_creation() {
        let aggregator = Arc::new(MarketDataAggregator::new());
        let monitor = Arc::new(ArbitrageMonitor::new(ArbitrageConfig::default()));

        let bridge = ArbitrageBridge::new(aggregator, monitor);

        assert!(!bridge.is_running());
    }

    #[tokio::test]
    async fn test_builder() {
        let aggregator = Arc::new(MarketDataAggregator::new());
        let monitor = Arc::new(ArbitrageMonitor::new(ArbitrageConfig::default()));

        let bridge = ArbitrageBridgeBuilder::new()
            .aggregator(aggregator)
            .monitor(monitor)
            .build()
            .unwrap();

        assert!(!bridge.is_running());
    }

    #[tokio::test]
    async fn test_builder_missing_aggregator() {
        let monitor = Arc::new(ArbitrageMonitor::new(ArbitrageConfig::default()));

        let result = ArbitrageBridgeBuilder::new().monitor(monitor).build();

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_builder_missing_monitor() {
        let aggregator = Arc::new(MarketDataAggregator::new());

        let result = ArbitrageBridgeBuilder::new().aggregator(aggregator).build();

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_start_stop() {
        let aggregator = Arc::new(MarketDataAggregator::new());
        let monitor = Arc::new(ArbitrageMonitor::new(ArbitrageConfig::default()));

        let bridge = ArbitrageBridge::new(aggregator, monitor);

        // Start the bridge
        bridge.start().await.unwrap();
        assert!(bridge.is_running());

        // Give it a moment to spawn the task
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

        // Stop the bridge
        bridge.stop().await;
        assert!(!bridge.is_running());
    }
}
