//! Alternative metrics ingestion module
//!
//! This module provides polling-based ingestion for metrics that don't have
//! WebSocket endpoints, including:
//! - Fear & Greed Index (market sentiment)
//! - ETF Net Flows (institutional demand)
//! - Volatility Indices (DVOL, realized vol)
//! - Altcoin Season Index
//!
//! ## Architecture:
//! Each metric has its own poller that runs on a configurable interval.
//! Pollers respect rate limits and implement deduplication to avoid
//! storing redundant data.

use anyhow::Result;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{error, info};

use crate::actors::{PollerBuilder, Router};
use crate::config::Config;

mod etf_flow;
mod fear_greed;
mod volatility;

// P0 Item 5: Prometheus metrics exporter
pub mod prometheus_exporter;

pub use etf_flow::EtfFlowPoller;
pub use fear_greed::FearGreedPoller;
#[allow(unused_imports)]
pub use prometheus_exporter::PrometheusExporter;
pub use volatility::VolatilityPoller;

/// Manages all metric pollers
#[allow(dead_code)]
pub struct MetricsManager {
    config: Arc<Config>,
    shutdown_rx: broadcast::Receiver<()>,
}

impl MetricsManager {
    /// Create a new MetricsManager
    pub async fn new(config: Arc<Config>, shutdown_rx: broadcast::Receiver<()>) -> Result<Self> {
        Ok(Self {
            config,
            shutdown_rx,
        })
    }

    /// Start all enabled metric pollers
    #[allow(dead_code)]
    pub async fn start_all(&self, router: Arc<Router>) -> Result<()> {
        info!("MetricsManager: Starting metric pollers");

        let router_tx = router.get_sender();

        // Initialize Redis client for rate limiting (if configured)
        let redis_client = if !self.config.redis.url.is_empty() {
            match redis::Client::open(self.config.redis.url.as_str()) {
                Ok(client) => Some(client),
                Err(e) => {
                    error!("MetricsManager: Failed to create Redis client: {}", e);
                    None
                }
            }
        } else {
            None
        };

        // Fear & Greed Index Poller
        if self.config.metrics.enable_fear_greed {
            info!("MetricsManager: Starting Fear & Greed Index poller");

            let fear_greed_poller = Arc::new(FearGreedPoller::new(
                self.config.metrics.fear_greed_url.clone(),
            ));

            let poller = PollerBuilder::new("fear_greed")
                .interval_secs(self.config.metrics.poll_interval_secs)
                .enabled(true)
                .with_redis(
                    redis_client
                        .clone()
                        .unwrap_or_else(|| redis::Client::open("redis://localhost:6379").unwrap()),
                )
                .build(
                    move || {
                        let poller = fear_greed_poller.clone();
                        Box::pin(async move { poller.poll().await })
                    },
                    router_tx.clone(),
                    self.shutdown_rx.resubscribe(),
                );

            tokio::spawn(async move {
                if let Err(e) = poller.run().await {
                    error!("MetricsManager: Fear & Greed poller failed: {}", e);
                }
            });
        }

        // ETF Net Flows Poller
        if self.config.metrics.enable_etf_flows {
            info!("MetricsManager: Starting ETF Net Flows poller");

            let etf_flow_poller = Arc::new(EtfFlowPoller::new(
                self.config.metrics.etf_flows_url.clone(),
            ));

            let poller = PollerBuilder::new("etf_flows")
                .interval_secs(self.config.metrics.poll_interval_secs * 2) // Poll less frequently
                .enabled(true)
                .with_redis(
                    redis_client
                        .clone()
                        .unwrap_or_else(|| redis::Client::open("redis://localhost:6379").unwrap()),
                )
                .build(
                    move || {
                        let poller = etf_flow_poller.clone();
                        Box::pin(async move { poller.poll().await })
                    },
                    router_tx.clone(),
                    self.shutdown_rx.resubscribe(),
                );

            tokio::spawn(async move {
                if let Err(e) = poller.run().await {
                    error!("MetricsManager: ETF Flows poller failed: {}", e);
                }
            });
        }

        // Volatility Poller
        if self.config.metrics.enable_volatility {
            info!("MetricsManager: Starting Volatility (DVOL) poller");

            let volatility_poller =
                Arc::new(VolatilityPoller::new(self.config.metrics.dvol_url.clone()));

            let poller = PollerBuilder::new("volatility")
                .interval_secs(60) // Poll every minute for volatility
                .enabled(true)
                .with_redis(
                    redis_client
                        .clone()
                        .unwrap_or_else(|| redis::Client::open("redis://localhost:6379").unwrap()),
                )
                .build(
                    move || {
                        let poller = volatility_poller.clone();
                        Box::pin(async move { poller.poll().await })
                    },
                    router_tx.clone(),
                    self.shutdown_rx.resubscribe(),
                );

            tokio::spawn(async move {
                if let Err(e) = poller.run().await {
                    error!("MetricsManager: Volatility poller failed: {}", e);
                }
            });
        }

        info!("MetricsManager: All metric pollers started");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    #[tokio::test]
    async fn test_metrics_manager_creation() {
        let config =
            Arc::new(Config::from_env().expect("Config::from_env should succeed with defaults"));
        let (_shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);

        let manager = MetricsManager::new(config.clone(), shutdown_rx).await;
        assert!(manager.is_ok(), "MetricsManager::new should succeed");

        let mgr = manager.unwrap();
        // Verify it captured the config we passed in
        assert_eq!(mgr.config.assets, config.assets);
    }

    #[test]
    fn test_metrics_config_defaults() {
        let config = Config::from_env().expect("Config::from_env should succeed with defaults");
        // Verify default metric toggles are sensible
        assert!(config.metrics.enable_fear_greed);
        assert!(config.metrics.enable_etf_flows);
        assert!(config.metrics.enable_volatility);
        assert!(config.metrics.poll_interval_secs > 0);
    }
}
