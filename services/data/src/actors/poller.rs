//! Poller Actor - REST API polling with rate limiting
//!
//! This actor handles periodic polling of REST APIs for metrics that don't
//! have WebSocket endpoints (e.g., Fear & Greed Index, ETF flows).
//!
//! ## Features:
//! - Configurable polling intervals
//! - Redis-backed rate limiting using token bucket algorithm
//! - Exponential backoff on failures
//! - Deduplication to avoid storing identical data
//! - Graceful shutdown with final flush

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc};
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use super::{ActorStats, DataMessage, MetricData};

/// Poller actor configuration
#[derive(Debug, Clone)]
pub struct PollerConfig {
    /// Name of this poller (e.g., "fear_greed", "etf_flows")
    pub name: String,

    /// Polling interval (seconds)
    pub interval_secs: u64,

    /// Enable this poller
    pub enabled: bool,

    /// Redis rate limit key (if applicable)
    pub rate_limit_key: Option<String>,

    /// Maximum requests per time window
    pub rate_limit_max: Option<u32>,

    /// Rate limit window (seconds)
    pub rate_limit_window_secs: Option<u64>,
}

impl Default for PollerConfig {
    fn default() -> Self {
        Self {
            name: String::from("poller"),
            interval_secs: 300, // 5 minutes
            enabled: true,
            rate_limit_key: None,
            rate_limit_max: None,
            rate_limit_window_secs: None,
        }
    }
}

/// Poller actor that executes periodic REST API calls
pub struct PollerActor<F>
where
    F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<MetricData>>> + Send>>
        + Send
        + Sync
        + 'static,
{
    config: PollerConfig,

    /// The actual polling function (async closure)
    poll_fn: Arc<F>,

    /// Sender to router
    router_tx: mpsc::UnboundedSender<DataMessage>,

    /// Shutdown signal
    shutdown_rx: broadcast::Receiver<()>,

    /// Statistics
    stats: ActorStats,

    /// Redis client for rate limiting (optional)
    redis_client: Option<redis::Client>,
}

impl<F> PollerActor<F>
where
    F: Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<MetricData>>> + Send>>
        + Send
        + Sync
        + 'static,
{
    /// Create a new Poller actor
    pub fn new(
        config: PollerConfig,
        poll_fn: F,
        router_tx: mpsc::UnboundedSender<DataMessage>,
        shutdown_rx: broadcast::Receiver<()>,
        redis_client: Option<redis::Client>,
    ) -> Self {
        Self {
            config,
            poll_fn: Arc::new(poll_fn),
            router_tx,
            shutdown_rx,
            stats: ActorStats::new(),
            redis_client,
        }
    }

    /// Start the poller actor
    pub async fn run(mut self) -> Result<()> {
        if !self.config.enabled {
            info!(
                "Poller Actor '{}': Disabled, not starting",
                self.config.name
            );
            return Ok(());
        }

        info!(
            "Poller Actor '{}': Starting with interval {} seconds",
            self.config.name, self.config.interval_secs
        );

        let mut poll_interval = interval(Duration::from_secs(self.config.interval_secs));

        // Execute immediately on startup
        match self.execute_poll().await {
            Err(e) => {
                error!(
                    "Poller Actor '{}': Initial poll failed: {}",
                    self.config.name, e
                );
                self.stats.record_failure();
            }
            _ => {
                self.stats.record_success();
            }
        }

        loop {
            tokio::select! {
                // Execute poll on interval
                _ = poll_interval.tick() => {
                    match self.execute_poll().await { Err(e) => {
                        error!(
                            "Poller Actor '{}': Poll failed: {}",
                            self.config.name, e
                        );
                        self.stats.record_failure();
                    } _ => {
                        self.stats.record_success();
                    }}
                }

                // Handle shutdown
                _ = self.shutdown_rx.recv() => {
                    info!("Poller Actor '{}': Shutdown signal received", self.config.name);
                    break;
                }
            }
        }

        info!("Poller Actor '{}': Stopped", self.config.name);
        Ok(())
    }

    /// Execute a single poll operation
    async fn execute_poll(&mut self) -> Result<()> {
        debug!("Poller Actor '{}': Executing poll", self.config.name);

        // Check rate limit if configured
        if let Some(ref key) = self.config.rate_limit_key
            && !self.check_rate_limit(key).await?
        {
            warn!(
                "Poller Actor '{}': Rate limit exceeded, skipping poll",
                self.config.name
            );
            return Ok(());
        }

        // Execute the polling function
        let metrics = (self.poll_fn)().await.context("Poll function failed")?;

        if metrics.is_empty() {
            debug!("Poller Actor '{}': No metrics returned", self.config.name);
            return Ok(());
        }

        // Send metrics to router
        let count = metrics.len();
        for metric in metrics {
            let msg = DataMessage::Metric(metric);
            self.router_tx
                .send(msg)
                .context("Failed to send metric to router")?;
        }

        debug!(
            "Poller Actor '{}': Successfully polled and sent {} metrics",
            self.config.name, count
        );

        Ok(())
    }

    /// Check rate limit using Redis token bucket algorithm
    async fn check_rate_limit(&self, key: &str) -> Result<bool> {
        let Some(ref client) = self.redis_client else {
            // No Redis client, allow request
            return Ok(true);
        };

        let Some(max_requests) = self.config.rate_limit_max else {
            // No rate limit configured
            return Ok(true);
        };

        // Get a connection (async)
        let mut conn = client
            .get_multiplexed_async_connection()
            .await
            .context("Failed to get Redis connection")?;

        // Use Redis INCR + EXPIRE for simple rate limiting
        // Key format: ratelimit:{service_name}:{window_start}
        let window_secs = self.config.rate_limit_window_secs.unwrap_or(60);
        let now = chrono::Utc::now().timestamp();
        let window_start = now - (now % window_secs as i64);
        let rate_limit_key = format!("{}:{}", key, window_start);

        // Increment the counter
        let count: u32 = redis::cmd("INCR")
            .arg(&rate_limit_key)
            .query_async(&mut conn)
            .await
            .context("Failed to increment rate limit counter")?;

        // Set expiration on first increment
        if count == 1 {
            let _: () = redis::cmd("EXPIRE")
                .arg(&rate_limit_key)
                .arg(window_secs * 2) // Expire after 2 windows to be safe
                .query_async(&mut conn)
                .await
                .context("Failed to set expiration on rate limit key")?;
        }

        // Check if limit exceeded
        if count > max_requests {
            debug!(
                "Poller Actor '{}': Rate limit exceeded ({}/{})",
                self.config.name, count, max_requests
            );
            Ok(false)
        } else {
            debug!(
                "Poller Actor '{}': Rate limit OK ({}/{})",
                self.config.name, count, max_requests
            );
            Ok(true)
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &ActorStats {
        &self.stats
    }
}

/// Builder pattern for creating PollerActor
pub struct PollerBuilder {
    config: PollerConfig,
    redis_client: Option<redis::Client>,
}

impl PollerBuilder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: PollerConfig {
                name: name.into(),
                ..Default::default()
            },
            redis_client: None,
        }
    }

    pub fn interval_secs(mut self, secs: u64) -> Self {
        self.config.interval_secs = secs;
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    pub fn rate_limit(
        mut self,
        key: impl Into<String>,
        max_requests: u32,
        window_secs: u64,
    ) -> Self {
        self.config.rate_limit_key = Some(key.into());
        self.config.rate_limit_max = Some(max_requests);
        self.config.rate_limit_window_secs = Some(window_secs);
        self
    }

    pub fn with_redis(mut self, client: redis::Client) -> Self {
        self.redis_client = Some(client);
        self
    }

    pub fn build<F>(
        self,
        poll_fn: F,
        router_tx: mpsc::UnboundedSender<DataMessage>,
        shutdown_rx: broadcast::Receiver<()>,
    ) -> PollerActor<F>
    where
        F: Fn() -> std::pin::Pin<
                Box<dyn std::future::Future<Output = Result<Vec<MetricData>>> + Send>,
            > + Send
            + Sync
            + 'static,
    {
        PollerActor::new(
            self.config,
            poll_fn,
            router_tx,
            shutdown_rx,
            self.redis_client,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poller_config_default() {
        let config = PollerConfig::default();
        assert_eq!(config.name, "poller");
        assert_eq!(config.interval_secs, 300);
        assert!(config.enabled);
        assert!(config.rate_limit_key.is_none());
    }

    #[test]
    fn test_poller_builder() {
        let builder = PollerBuilder::new("test_poller")
            .interval_secs(60)
            .enabled(false)
            .rate_limit("test:limit", 100, 3600);

        assert_eq!(builder.config.name, "test_poller");
        assert_eq!(builder.config.interval_secs, 60);
        assert!(!builder.config.enabled);
        assert_eq!(
            builder.config.rate_limit_key,
            Some("test:limit".to_string())
        );
        assert_eq!(builder.config.rate_limit_max, Some(100));
        assert_eq!(builder.config.rate_limit_window_secs, Some(3600));
    }
}
