//! Storage layer for persisting market data to QuestDB and Redis
//!
//! This module provides high-performance persistence using:
//! - **QuestDB**: Time-series database for trade/candle/metric data via ILP
//! - **Redis**: In-memory store for rate limiting, deduplication, and state
//!
//! ## Architecture:
//! - Batched writes to QuestDB using Influx Line Protocol (ILP)
//! - Configurable flush intervals and buffer sizes
//! - Automatic reconnection on connection failures
//! - Zero-copy serialization where possible

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::{Mutex, broadcast};
use tracing::{debug, error, info};

use crate::actors::{CandleData, HealthData, MetricData, TradeData};
use crate::config::Config;

pub mod ilp;
mod redis_ops;

pub use ilp::{IlpWriter, VerificationResult};
pub use redis_ops::RedisManager;

/// Main storage manager coordinating QuestDB and Redis
pub struct StorageManager {
    config: Arc<Config>,

    /// QuestDB writer for time-series data
    pub ilp_writer: Arc<Mutex<IlpWriter>>,

    /// Redis manager for state/cache
    pub redis_manager: Arc<RedisManager>,

    /// Shutdown signal
    shutdown_rx: broadcast::Receiver<()>,

    /// Statistics
    stats: Arc<Mutex<StorageStats>>,
}

impl StorageManager {
    /// Create a new StorageManager
    pub async fn new(
        config: Arc<Config>,
        shutdown_rx: broadcast::Receiver<()>,
    ) -> Result<Arc<Self>> {
        info!("StorageManager: Initializing storage connections");

        // Initialize ILP writer for QuestDB
        let ilp_writer = IlpWriter::new(
            &config.questdb.host,
            config.questdb.ilp_port,
            config.questdb.buffer_size,
            config.questdb.flush_interval_ms,
        )
        .await
        .context("Failed to initialize ILP writer")?;

        info!("StorageManager: QuestDB ILP writer initialized");

        // Initialize Redis manager
        let redis_manager = RedisManager::new(&config.redis.url, &config.redis.key_prefix)
            .await
            .context("Failed to initialize Redis manager")?;

        info!("StorageManager: Redis manager initialized");

        let manager = Arc::new(Self {
            config,
            ilp_writer: Arc::new(Mutex::new(ilp_writer)),
            redis_manager: Arc::new(redis_manager),
            shutdown_rx,
            stats: Arc::new(Mutex::new(StorageStats::default())),
        });

        // Start background flush task
        manager.start_flush_task();

        Ok(manager)
    }

    /// Store a trade to QuestDB
    pub async fn store_trade(&self, trade: TradeData) -> Result<()> {
        debug!(
            "StorageManager: Storing trade {} {} @ {}",
            trade.symbol, trade.side, trade.price
        );

        // Write to ILP buffer
        let mut writer = self.ilp_writer.lock().await;
        writer.write_trade(&trade).await?;

        // Update stats
        let mut stats = self.stats.lock().await;
        stats.trades_written += 1;

        Ok(())
    }

    /// Store a candle to QuestDB
    pub async fn store_candle(&self, candle: CandleData) -> Result<()> {
        debug!(
            "StorageManager: Storing candle {} {} [{}-{}]",
            candle.symbol, candle.interval, candle.open, candle.close
        );

        let mut writer = self.ilp_writer.lock().await;
        writer.write_candle(&candle).await?;

        let mut stats = self.stats.lock().await;
        stats.candles_written += 1;

        Ok(())
    }

    /// Store a metric to QuestDB
    pub async fn store_metric(&self, metric: MetricData) -> Result<()> {
        debug!(
            "StorageManager: Storing metric {} = {} for {}",
            metric.metric_type, metric.value, metric.asset
        );

        // Check for deduplication in Redis
        if self.config.operational.enable_backfill {
            let is_duplicate = self
                .redis_manager
                .is_metric_duplicate(&metric)
                .await
                .unwrap_or(false);

            if is_duplicate {
                debug!(
                    "StorageManager: Skipping duplicate metric {}",
                    metric.metric_type
                );
                return Ok(());
            }
        }

        let mut writer = self.ilp_writer.lock().await;
        writer.write_metric(&metric).await?;

        // Store hash in Redis for deduplication
        if self.config.operational.enable_backfill {
            let _ = self.redis_manager.mark_metric_seen(&metric).await;
        }

        let mut stats = self.stats.lock().await;
        stats.metrics_written += 1;

        Ok(())
    }

    /// Store health check data
    pub async fn store_health(&self, health: HealthData) -> Result<()> {
        debug!(
            "StorageManager: Storing health check from {} - {}",
            health.component, health.status
        );

        // Store in Redis for quick access
        self.redis_manager
            .store_health_status(&health.component, &health.status.to_string())
            .await?;

        // Optionally write to QuestDB for historical analysis
        // (commented out to avoid excessive writes for health checks)
        // let mut writer = self.ilp_writer.lock().await;
        // writer.write_health(&health).await?;

        Ok(())
    }

    /// Write a raw ILP line to QuestDB
    /// This is used for custom tables like signals that don't have dedicated write methods
    pub async fn write_raw_ilp(&self, line: &str) -> Result<()> {
        let mut writer = self.ilp_writer.lock().await;
        writer.write_raw(line).await?;
        Ok(())
    }

    /// Manually flush all pending writes to QuestDB
    pub async fn flush(&self) -> Result<()> {
        debug!("StorageManager: Manual flush requested");

        let mut writer = self.ilp_writer.lock().await;
        writer.flush().await?;

        info!("StorageManager: Manual flush completed");
        Ok(())
    }

    /// Start background task for periodic flushing
    fn start_flush_task(self: &Arc<Self>) {
        let manager = Arc::clone(self);
        let flush_interval = self.config.questdb.flush_interval_ms;
        let mut shutdown_rx = self.shutdown_rx.resubscribe();

        tokio::spawn(async move {
            let mut interval =
                tokio::time::interval(tokio::time::Duration::from_millis(flush_interval));

            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        if let Err(e) = manager.flush().await {
                            error!("StorageManager: Background flush failed: {}", e);
                        }
                    }

                    _ = shutdown_rx.recv() => {
                        info!("StorageManager: Flush task shutting down");
                        // Final flush before shutdown
                        let _ = manager.flush().await;
                        break;
                    }
                }
            }
        });
    }

    /// Get Redis manager for direct access
    pub fn redis(&self) -> &Arc<RedisManager> {
        &self.redis_manager
    }

    /// Get current storage statistics
    pub async fn stats(&self) -> StorageStats {
        self.stats.lock().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        let mut stats = self.stats.lock().await;
        *stats = StorageStats::default();
        info!("StorageManager: Statistics reset");
    }
}

/// Storage statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct StorageStats {
    pub trades_written: u64,
    pub candles_written: u64,
    pub metrics_written: u64,
    pub flushes_completed: u64,
    pub flush_errors: u64,
}

impl StorageStats {
    pub fn total_writes(&self) -> u64 {
        self.trades_written + self.candles_written + self.metrics_written
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_stats_default() {
        let stats = StorageStats::default();
        assert_eq!(stats.trades_written, 0);
        assert_eq!(stats.candles_written, 0);
        assert_eq!(stats.metrics_written, 0);
        assert_eq!(stats.total_writes(), 0);
    }

    #[test]
    fn test_storage_stats_total() {
        let stats = StorageStats {
            trades_written: 100,
            candles_written: 50,
            metrics_written: 25,
            flushes_completed: 10,
            flush_errors: 1,
        };

        assert_eq!(stats.total_writes(), 175);
    }
}
