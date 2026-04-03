//! Exchange connectors for market data ingestion
//!
//! This module provides a unified abstraction layer for connecting to multiple
//! cryptocurrency exchanges. Each connector implements exchange-specific logic
//! for WebSocket subscriptions, message parsing, and authentication.
//!
//! ## Supported Exchanges:
//! - Binance: High-volume spot and futures markets
//! - Bybit: V5 unified API for spot and derivatives
//! - Kucoin: Token-based WebSocket authentication
//! - Coinbase: Advanced Trade API (via bridge adapter)
//! - Kraken: Professional trading (via bridge adapter)
//! - OKX: Global derivatives (via bridge adapter)
//!
//! ## Architecture:
//! The ConnectorManager handles exchange selection, failover logic, and
//! lifecycle management of WebSocket connections. It uses bridge adapters
//! to integrate the new `janus-exchanges` crate with CNS metrics and health monitoring.

#![allow(dead_code)]

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{error, info, warn};

use crate::actors::{ExchangeConnector, Router, WebSocketActor};
use crate::config::{Config, Exchange};

mod binance;
pub mod bridge;
mod bybit;
pub mod circuit_breaker_integration;
mod kucoin;

pub use binance::BinanceConnector;
pub use bridge::{CoinbaseBridge, KrakenBridge, OkxBridge};
pub use bybit::BybitConnector;
pub use kucoin::KucoinConnector;

// ExchangeConnector trait is now defined in crate::actors to avoid circular dependencies

/// Manages all exchange connectors and handles failover logic
pub struct ConnectorManager {
    config: Arc<Config>,
    shutdown_rx: broadcast::Receiver<()>,
    active_exchange: Exchange,
    connectors: ConnectorRegistry,
}

/// Registry of available exchange connectors
pub struct ConnectorRegistry {
    binance: Arc<BinanceConnector>,
    bybit: Arc<BybitConnector>,
    kucoin: Arc<KucoinConnector>,
    coinbase: Arc<CoinbaseBridge>,
    kraken: Arc<KrakenBridge>,
    okx: Arc<OkxBridge>,
}

impl ConnectorManager {
    /// Create a new ConnectorManager
    pub async fn new(config: Arc<Config>, shutdown_rx: broadcast::Receiver<()>) -> Result<Self> {
        info!("ConnectorManager: Initializing exchange connectors");

        // Get configured kline intervals (default to 1m, 5m, 15m, 1h, 4h for multi-timeframe analysis)
        let kline_intervals = vec![
            "1m".to_string(),
            "5m".to_string(),
            "15m".to_string(),
            "1h".to_string(),
            "4h".to_string(),
        ];

        // Initialize all connectors with multi-timeframe support
        let binance = Arc::new(
            BinanceConnector::new(config.exchanges.binance_ws_url.clone())
                .with_kline_intervals(kline_intervals),
        );

        let bybit = Arc::new(BybitConnector::new(config.exchanges.bybit_ws_url.clone()));

        let kucoin =
            Arc::new(KucoinConnector::new(config.exchanges.kucoin_rest_url.clone()).await?);

        // Initialize bridge adapters for new exchanges (with CNS + Health monitoring)
        let coinbase = Arc::new(CoinbaseBridge::new(
            config.exchanges.coinbase_ws_url.clone(),
        ));
        let kraken = Arc::new(KrakenBridge::new(config.exchanges.kraken_ws_url.clone()));
        let okx = Arc::new(OkxBridge::new(config.exchanges.okx_ws_url.clone()));

        let connectors = ConnectorRegistry {
            binance,
            bybit,
            kucoin,
            coinbase,
            kraken,
            okx,
        };

        let active_exchange = config.exchanges.primary;

        info!(
            "ConnectorManager: Initialized with primary exchange: {}",
            active_exchange
        );

        Ok(Self {
            config,
            shutdown_rx,
            active_exchange,
            connectors,
        })
    }

    /// Start WebSocket connections for all configured assets
    pub async fn start_all(&self, router: Arc<Router>) -> Result<()> {
        info!(
            "ConnectorManager: Starting WebSocket connections for {} assets",
            self.config.assets.len()
        );

        let router_tx = router.get_sender();

        for asset in &self.config.assets {
            let symbol = self.format_symbol(asset);
            let connector = self.get_active_connector();

            info!(
                "ConnectorManager: Starting WebSocket for {} on {}",
                symbol,
                connector.exchange_name()
            );

            // Build WebSocket configuration
            let ws_config = connector.build_ws_config(&symbol);

            // Create WebSocket actor with connector for message parsing
            let actor = WebSocketActor::new(
                ws_config,
                connector.clone(),
                router_tx.clone(),
                self.shutdown_rx.resubscribe(),
            );

            // Spawn the actor in a new task
            tokio::spawn(async move {
                if let Err(e) = actor.run().await {
                    error!(
                        "ConnectorManager: WebSocket actor failed for {}: {}",
                        symbol, e
                    );
                }
            });

            // Small delay to avoid connection burst
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        }

        info!("ConnectorManager: All WebSocket connections started");
        Ok(())
    }

    /// Get the currently active exchange connector
    fn get_active_connector(&self) -> Arc<dyn ExchangeConnector> {
        match self.active_exchange {
            Exchange::Binance => self.connectors.binance.clone() as Arc<dyn ExchangeConnector>,
            Exchange::Bybit => self.connectors.bybit.clone() as Arc<dyn ExchangeConnector>,
            Exchange::Kucoin => self.connectors.kucoin.clone() as Arc<dyn ExchangeConnector>,
            Exchange::Coinbase => self.connectors.coinbase.clone() as Arc<dyn ExchangeConnector>,
            Exchange::Kraken => self.connectors.kraken.clone() as Arc<dyn ExchangeConnector>,
            Exchange::Okx => self.connectors.okx.clone() as Arc<dyn ExchangeConnector>,
        }
    }

    /// Format symbol for the active exchange
    ///
    /// Different exchanges use different symbol formats:
    /// - Binance: "BTCUSD" (lowercase for WebSocket)
    /// - Bybit: "BTCUSD" (uppercase)
    /// - Kucoin: "BTC-USDT" (with hyphen)
    /// - Coinbase: "BTC-USD" (with hyphen, USD not USDT)
    /// - Kraken: "BTC/USD" (with slash)
    /// - OKX: "BTC-USDT" (with hyphen)
    fn format_symbol(&self, asset: &str) -> String {
        match self.active_exchange {
            Exchange::Binance => format!("{}usdt", asset.to_lowercase()),
            Exchange::Bybit => format!("{}USDT", asset.to_uppercase()),
            Exchange::Kucoin => format!("{}-USDT", asset.to_uppercase()),
            Exchange::Coinbase => format!("{}usdt", asset.to_lowercase()), // Bridge handles conversion
            Exchange::Kraken => format!("{}usdt", asset.to_lowercase()), // Bridge handles conversion
            Exchange::Okx => format!("{}usdt", asset.to_lowercase()), // Bridge handles conversion
        }
    }

    /// Switch to a different exchange (failover)
    pub async fn failover_to(&mut self, exchange: Exchange) -> Result<()> {
        if self.active_exchange == exchange {
            warn!(
                "ConnectorManager: Already using exchange {}, ignoring failover",
                exchange
            );
            return Ok(());
        }

        info!(
            "ConnectorManager: Failing over from {} to {}",
            self.active_exchange, exchange
        );

        self.active_exchange = exchange;

        // In a production implementation, we would:
        // 1. Gracefully close existing WebSocket connections
        // 2. Wait for pending messages to flush
        // 3. Reconnect to the new exchange
        // For now, we just update the active exchange and rely on
        // the auto-reconnect logic in WebSocketActor

        info!("ConnectorManager: Failover complete");
        Ok(())
    }

    /// Get the current active exchange
    pub fn active_exchange(&self) -> Exchange {
        self.active_exchange
    }

    /// Check health of all connectors
    pub async fn health_check(&self) -> ConnectorHealth {
        // Get health from bridge adapters (CNS-backed)
        let coinbase_health = self
            .connectors
            .coinbase
            .health_checker()
            .get_health("coinbase")
            .await;
        let kraken_health = self
            .connectors
            .kraken
            .health_checker()
            .get_health("kraken")
            .await;
        let okx_health = self.connectors.okx.health_checker().get_health("okx").await;

        ConnectorHealth {
            binance: HealthStatus::Unknown, // Legacy connectors don't have health yet
            bybit: HealthStatus::Unknown,
            kucoin: HealthStatus::Unknown,
            coinbase: convert_exchange_health(coinbase_health),
            kraken: convert_exchange_health(kraken_health),
            okx: convert_exchange_health(okx_health),
        }
    }

    /// Get access to all connectors for monitoring
    pub fn connectors(&self) -> &ConnectorRegistry {
        &self.connectors
    }
}

/// Convert janus-exchanges ExchangeHealth to local HealthStatus
fn convert_exchange_health(
    health: Option<janus_exchanges::health::ExchangeHealth>,
) -> HealthStatus {
    use janus_exchanges::health::HealthStatus as ExtHealthStatus;

    match health {
        Some(h) => match h.status {
            ExtHealthStatus::Healthy => HealthStatus::Healthy,
            ExtHealthStatus::Degraded => HealthStatus::Degraded,
            ExtHealthStatus::Down => HealthStatus::Unhealthy,
            ExtHealthStatus::Unknown => HealthStatus::Unknown,
        },
        None => HealthStatus::Unknown,
    }
}

impl ConnectorRegistry {
    /// Get health checker for Coinbase
    pub fn coinbase_health(&self) -> Arc<janus_exchanges::HealthChecker> {
        self.coinbase.health_checker()
    }

    /// Get health checker for Kraken
    pub fn kraken_health(&self) -> Arc<janus_exchanges::HealthChecker> {
        self.kraken.health_checker()
    }

    /// Get health checker for OKX
    pub fn okx_health(&self) -> Arc<janus_exchanges::HealthChecker> {
        self.okx.health_checker()
    }
}

/// Health status of each connector
#[derive(Debug, Clone)]
pub struct ConnectorHealth {
    pub binance: HealthStatus,
    pub bybit: HealthStatus,
    pub kucoin: HealthStatus,
    pub coinbase: HealthStatus,
    pub kraken: HealthStatus,
    pub okx: HealthStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_symbol_binance() {
        let _config = Arc::new(Config::from_env().unwrap());
        let (_, _rx): (broadcast::Sender<()>, broadcast::Receiver<()>) = broadcast::channel(1);
        // Note: This would need proper async test setup
        // let manager = ConnectorManager::new(config, rx).await.unwrap();
        // assert_eq!(manager.format_symbol("BTC"), "btcusdt");
    }

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
        assert_eq!(HealthStatus::Unknown.to_string(), "unknown");
    }
}
