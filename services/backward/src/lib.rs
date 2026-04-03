//! # JANUS Backward Service
//!
//! Historical analysis, persistence, and analytics service.
//! Handles backward-looking operations including:
//! - Database persistence
//! - Historical analytics
//! - Performance metrics
//! - Backtesting
//! - Scheduled jobs
//! - Data aggregation
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  JANUS Backward Service                      │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
//! │  │  Database    │  │  Performance │  │  Analytics   │     │
//! │  │ Persistence  │  │   Metrics    │  │   Engine     │     │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
//! │         │                  │                  │             │
//! │         └──────────────────┼──────────────────┘             │
//! │                            │                                │
//! │                   ┌────────▼────────┐                       │
//! │                   │  Job Scheduler  │                       │
//! │                   └────────┬────────┘                       │
//! │                            │                                │
//! │         ┌──────────────────┴──────────────────┐            │
//! │         │                                      │            │
//! │    ┌────▼────┐                          ┌─────▼─────┐      │
//! │    │Workers  │                          │   REST    │      │
//! │    │ (Tasks) │                          │    API    │      │
//! │    └─────────┘                          └───────────┘      │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```

pub mod http;
pub mod metrics;
pub mod persistence;
pub mod sched;
pub mod tasks;
pub mod worker;

use tokio_util::sync::CancellationToken;

// Re-exports for convenience
pub use http::start_http_server;
pub use metrics::{JanusMetrics, PrometheusExporter, RiskMetricsCollector, SignalMetricsCollector};
pub use persistence::{
    Database, DatabaseConfig, DatabaseError,
    repositories::{
        MetricsRepository, PerformanceRepository, PortfolioRepository, PositionRepository,
        SignalRepository,
    },
};

use anyhow::Result;
use std::sync::Arc;

use tracing::info;

/// Backward service version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Backward service name
pub const SERVICE_NAME: &str = "janus-backward";

/// Service configuration
#[derive(Debug, Clone)]
pub struct BackwardServiceConfig {
    /// Service host
    pub host: String,

    /// HTTP API port
    pub http_port: u16,

    /// Database configuration
    pub database_config: DatabaseConfig,

    /// Enable metrics endpoint
    pub enable_metrics: bool,

    /// Metrics port
    pub metrics_port: u16,

    /// Redis URL for job queue
    pub redis_url: String,

    /// Number of worker threads
    pub worker_threads: usize,

    /// Enable scheduled jobs
    pub enable_scheduler: bool,
}

impl Default for BackwardServiceConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            http_port: 8082,
            database_config: DatabaseConfig::default(),
            enable_metrics: true,
            metrics_port: 9091,
            redis_url: "redis://localhost:6379".to_string(),
            worker_threads: 4,
            enable_scheduler: true,
        }
    }
}

/// Backward service - Historical analysis and persistence
pub struct BackwardService {
    config: BackwardServiceConfig,
    database: Arc<Database>,
    metrics: Arc<JanusMetrics>,
    /// Cancellation token for coordinating graceful shutdown across workers
    cancel: CancellationToken,
    /// Handles to spawned worker tasks so we can await them on shutdown
    worker_handles: Vec<tokio::task::JoinHandle<()>>,
}

impl BackwardService {
    /// Create a new backward service instance
    pub async fn new(config: BackwardServiceConfig) -> Result<Self> {
        info!("Initializing JANUS Backward Service v{}", VERSION);

        // Initialize database
        info!("Connecting to database...");
        let database = Arc::new(Database::connect(config.database_config.clone()).await?);

        // Run migrations
        info!("Running database migrations...");
        database.run_migrations().await?;

        // Initialize metrics
        let metrics = Arc::new(JanusMetrics::new()?);

        Ok(Self {
            config,
            database,
            metrics,
            cancel: CancellationToken::new(),
            worker_handles: Vec::new(),
        })
    }

    /// Get database instance
    pub fn database(&self) -> Arc<Database> {
        Arc::clone(&self.database)
    }

    /// Get metrics instance
    pub fn metrics(&self) -> Arc<JanusMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Get service configuration
    pub fn config(&self) -> &BackwardServiceConfig {
        &self.config
    }

    /// Start the service
    #[tracing::instrument(skip(self), fields(host = %self.config.host, port = self.config.http_port))]
    pub async fn start(&mut self) -> Result<()> {
        info!(
            "Starting JANUS Backward Service on {}:{} (HTTP)",
            self.config.host, self.config.http_port
        );

        // Start scheduler if enabled
        if self.config.enable_scheduler {
            info!("Starting periodic job scheduler");
            let cancel = self.cancel.clone();
            let db = Arc::clone(&self.database);
            let metrics = Arc::clone(&self.metrics);
            let handle = tokio::spawn(async move {
                let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(60));
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            info!("Scheduler received shutdown signal");
                            break;
                        }
                        _ = interval.tick() => {
                            // Periodic health / aggregation tick
                            if let Ok(true) = db.health_check().await.map(|_| true) {
                                // Record a heartbeat via system metrics
                                metrics.system_metrics().record_http_request(0.0);
                            }
                            info!("Scheduler tick completed");
                        }
                    }
                }
                info!("Scheduler exited");
            });
            self.worker_handles.push(handle);
        }

        // Start workers
        info!(
            "Starting {} worker thread(s)...",
            self.config.worker_threads
        );
        for i in 0..self.config.worker_threads {
            let cancel = self.cancel.clone();
            let db = Arc::clone(&self.database);
            let worker_id = i;
            let handle = tokio::spawn(async move {
                info!("Worker {} started", worker_id);
                loop {
                    tokio::select! {
                        _ = cancel.cancelled() => {
                            info!("Worker {} received shutdown signal", worker_id);
                            break;
                        }
                        // Workers idle-poll waiting for jobs; in production this
                        // would read from a Redis/channel-based job queue.
                        _ = tokio::time::sleep(tokio::time::Duration::from_secs(5)) => {
                            // Check database connectivity as a heartbeat
                            if let Err(e) = db.health_check().await {
                                tracing::warn!("Worker {} health-check failed: {}", worker_id, e);
                            }
                        }
                    }
                }
                info!("Worker {} exited", worker_id);
            });
            self.worker_handles.push(handle);
            info!("Worker {} spawned", i);
        }

        // Start HTTP server
        info!("✅ Starting HTTP API server...");
        let http_state = http::HttpState::new();
        http::start_http_server(self.config.http_port, http_state).await?;

        Ok(())
    }

    /// Shutdown the service gracefully
    #[tracing::instrument(skip(self))]
    pub async fn shutdown(mut self) -> Result<()> {
        info!("Shutting down JANUS Backward Service");

        // 1. Signal all workers and the scheduler to stop
        info!(
            "Cancelling {} worker/scheduler tasks...",
            self.worker_handles.len()
        );
        self.cancel.cancel();

        // 2. Wait for all spawned tasks to finish (with a timeout)
        let drain_timeout = tokio::time::Duration::from_secs(10);
        for (i, handle) in self.worker_handles.drain(..).enumerate() {
            match tokio::time::timeout(drain_timeout, handle).await {
                Ok(Ok(())) => info!("Task {} joined cleanly", i),
                Ok(Err(e)) => tracing::warn!("Task {} panicked: {}", i, e),
                Err(_) => tracing::warn!(
                    "Task {} did not finish within {:?}, abandoning",
                    i,
                    drain_timeout
                ),
            }
        }

        // 3. Close database connections
        info!("Database connections will be dropped on service drop");

        info!("JANUS Backward Service shut down successfully");
        Ok(())
    }

    /// Get health status
    #[tracing::instrument(skip(self))]
    pub async fn health_check(&self) -> HealthStatus {
        let db_healthy = self.database.health_check().await.is_ok();

        HealthStatus {
            service: SERVICE_NAME.to_string(),
            version: VERSION.to_string(),
            status: if db_healthy { "healthy" } else { "degraded" }.to_string(),
            database_connected: db_healthy,
            worker_count: self.config.worker_threads,
            scheduler_enabled: self.config.enable_scheduler,
        }
    }

    /// Get signal repository
    pub fn signal_repository(&self) -> SignalRepository {
        SignalRepository::new(self.database.pool().clone())
    }

    /// Get portfolio repository
    pub fn portfolio_repository(&self) -> PortfolioRepository {
        PortfolioRepository::new(self.database.pool().clone())
    }

    /// Get position repository
    pub fn position_repository(&self) -> PositionRepository {
        PositionRepository::new(self.database.pool().clone())
    }

    /// Get performance repository
    pub fn performance_repository(&self) -> PerformanceRepository {
        PerformanceRepository::new(self.database.pool().clone())
    }

    /// Get metrics repository
    pub fn metrics_repository(&self) -> MetricsRepository {
        MetricsRepository::new(self.database.pool().clone())
    }
}

/// Health check status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HealthStatus {
    pub service: String,
    pub version: String,
    pub status: String,
    pub database_connected: bool,
    pub worker_count: usize,
    pub scheduler_enabled: bool,
}

/// Sanitize a database URL for safe logging (redact password)
fn sanitize_db_url(url: &str) -> String {
    // Match postgresql://user:password@host:port/db?params
    if let Some(at_pos) = url.find('@')
        && let Some(colon_pos) = url[..at_pos].rfind(':')
        && let Some(scheme_end) = url.find("://")
        && colon_pos > scheme_end + 3
    {
        let mut sanitized = String::with_capacity(url.len());
        sanitized.push_str(&url[..colon_pos + 1]);
        sanitized.push_str("****");
        sanitized.push_str(&url[at_pos..]);
        return sanitized;
    }
    format!("(unparseable URL, len={})", url.len())
}

/// Start the backward module as part of the unified JANUS system
///
/// This function is called by the unified JANUS binary to start the backward analytics and persistence module.
#[tracing::instrument(name = "backward::start_module", skip(state))]
pub async fn start_module(state: Arc<janus_core::JanusState>) -> janus_core::Result<()> {
    info!("Backward module registered — waiting for start command...");

    state
        .register_module_health("backward", true, Some("standby".to_string()))
        .await;

    // ── Wait for services to be started via API / web interface ──────
    if !state.wait_for_services_start().await {
        info!("Backward module: shutdown requested before services started");
        state
            .register_module_health("backward", false, Some("shutdown_before_start".to_string()))
            .await;
        return Ok(());
    }

    info!("Starting Backward module with unified JANUS integration...");

    state
        .register_module_health("backward", true, Some("starting".to_string()))
        .await;

    // Log diagnostic info about database configuration
    let db_url = &state.config.database.url;
    let sanitized_url = sanitize_db_url(db_url);
    info!("Backward module database config:");
    info!("  URL (sanitized): {}", sanitized_url);
    info!(
        "  Max connections: {}",
        state.config.database.max_connections
    );
    info!(
        "  Connect timeout: {}s",
        state.config.database.connect_timeout_secs
    );

    // Create backward service configuration from JANUS config
    let backward_config = BackwardServiceConfig {
        host: "0.0.0.0".to_string(),
        http_port: state.config.ports.http + 200, // Offset to avoid conflict
        metrics_port: 0,                          // Metrics handled by janus-api
        database_config: DatabaseConfig {
            database_url: state.config.database.url.clone(),
            max_connections: state.config.database.max_connections,
            min_connections: state.config.database.min_connections,
            connect_timeout: state.config.database.connect_timeout_secs,
            idle_timeout: state.config.database.idle_timeout_secs,
            max_lifetime: state.config.database.max_lifetime_secs,
            enable_logging: state.config.database.enable_logging,
        },
        redis_url: state.config.redis.url.clone(),
        worker_threads: state.config.backward.worker_threads,
        enable_metrics: false,
        enable_scheduler: state.config.backward.enable_scheduler,
    };

    // Retry database connection with backoff
    const MAX_RETRIES: u32 = 3;
    const RETRY_DELAY_SECS: u64 = 5;

    let mut service = None;
    let mut last_error = String::new();

    for attempt in 1..=MAX_RETRIES {
        info!(
            "Backward module: database connection attempt {}/{}",
            attempt, MAX_RETRIES
        );

        state
            .register_module_health(
                "backward",
                true,
                Some(format!("connecting (attempt {}/{})", attempt, MAX_RETRIES)),
            )
            .await;

        match BackwardService::new(backward_config.clone()).await {
            Ok(svc) => {
                info!(
                    "✅ Backward module: database connection established on attempt {}",
                    attempt
                );
                service = Some(svc);
                break;
            }
            Err(e) => {
                last_error = format!("{:#}", e);
                tracing::error!(
                    "❌ Backward module: database connection attempt {}/{} failed: {}",
                    attempt,
                    MAX_RETRIES,
                    last_error
                );

                if attempt < MAX_RETRIES {
                    info!("Backward module: retrying in {}s...", RETRY_DELAY_SECS);
                    tokio::time::sleep(tokio::time::Duration::from_secs(RETRY_DELAY_SECS)).await;
                }
            }
        }
    }

    let service = service.ok_or_else(|| {
        tracing::error!(
            "❌ Backward module: all {} connection attempts failed. Last error: {}",
            MAX_RETRIES,
            last_error
        );
        tracing::error!("  Database URL (sanitized): {}", sanitized_url);
        tracing::error!("  💡 Verify POSTGRES_PASSWORD in .env matches the password postgres was initialized with.");
        tracing::error!("  💡 Check: docker exec fks_postgres pg_isready -U fks_user -d fks");
        janus_core::Error::module("backward", format!("Failed to connect to database after {} attempts: {}", MAX_RETRIES, last_error))
    })?;

    state
        .register_module_health("backward", true, Some("running".to_string()))
        .await;

    // Subscribe to signals from signal bus
    let mut signal_rx = state.signal_bus.subscribe();
    let signal_repo = service.signal_repository();

    // Spawn signal persistence loop
    let state_clone = state.clone();
    let persist_task = tokio::spawn(async move {
        let mut batch: Vec<janus_core::Signal> = Vec::new();
        let batch_size = state_clone.config.backward.persistence.batch_size;
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(
            state_clone.config.backward.persistence.flush_interval_secs,
        ));

        loop {
            tokio::select! {
                Ok(signal) = signal_rx.recv() => {
                    info!(
                        "Backward module received signal for persistence: {} {} (confidence: {:.2})",
                        signal.symbol,
                        signal.signal_type,
                        signal.confidence
                    );

                    // Add to batch
                    batch.push(signal);

                    // Persist batch if full
                    if batch.len() >= batch_size {
                        info!("Persisting batch of {} signals", batch.len());
                        let persisted_count = persist_signal_batch(&signal_repo, &batch).await;
                        for _ in 0..persisted_count {
                            state_clone.increment_signals_persisted();
                        }
                        batch.clear();
                    }
                }
                _ = interval.tick() => {
                    // Periodic flush of remaining signals
                    if !batch.is_empty() {
                        info!("Flushing batch of {} signals (periodic)", batch.len());
                        let persisted_count = persist_signal_batch(&signal_repo, &batch).await;
                        for _ in 0..persisted_count {
                            state_clone.increment_signals_persisted();
                        }
                        batch.clear();
                    }

                    // Update health
                    state_clone
                        .register_module_health("backward", true, Some("persisting signals".to_string()))
                        .await;
                }
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    if state_clone.is_shutdown_requested() {
                        // Final flush
                        if !batch.is_empty() {
                            info!("Final flush of {} signals", batch.len());
                            let persisted_count = persist_signal_batch(&signal_repo, &batch).await;
                            for _ in 0..persisted_count {
                                state_clone.increment_signals_persisted();
                            }
                            batch.clear();
                        }
                        break;
                    }
                }
            }
        }
        info!("Backward persistence loop exited");
    });

    // Wait for shutdown
    while !state.is_shutdown_requested() {
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // Update health periodically
        state
            .register_module_health("backward", true, Some("running".to_string()))
            .await;
    }

    info!("Backward module shutting down...");

    // Cancel tasks
    persist_task.abort();

    // Shutdown service
    service
        .shutdown()
        .await
        .map_err(|e| janus_core::Error::module("backward", e.to_string()))?;

    state
        .register_module_health("backward", false, Some("stopped".to_string()))
        .await;

    info!("Backward module exited");
    Ok(())
}

/// Convert a janus_core::Signal to a NewSignal for persistence
fn signal_to_new_signal(signal: &janus_core::Signal) -> persistence::models::NewSignal {
    use uuid::Uuid;

    // Parse UUID from signal ID, or generate new one if invalid
    let signal_id = Uuid::parse_str(&signal.id).unwrap_or_else(|_| Uuid::new_v4());

    // Convert signal type to string
    let signal_type = match signal.signal_type {
        janus_core::SignalType::Buy => "Buy",
        janus_core::SignalType::Sell => "Sell",
        janus_core::SignalType::Hold => "Hold",
        janus_core::SignalType::Close => "Close",
    }
    .to_string();

    // Convert metadata HashMap to JSON
    let metadata = if signal.metadata.is_empty() {
        None
    } else {
        serde_json::to_value(&signal.metadata).ok()
    };

    persistence::models::NewSignal {
        signal_id,
        symbol: signal.symbol.clone(),
        signal_type,
        timeframe: "1m".to_string(), // Default timeframe, could be extracted from metadata
        confidence: signal.confidence,
        strength: signal.confidence, // Use confidence as strength if not separately available
        timestamp: signal.timestamp,
        entry_price: signal.target_price,
        stop_loss: signal.stop_loss,
        take_profit: signal.take_profit,
        position_size: signal.quantity,
        risk_amount: None,
        risk_reward_ratio: calculate_risk_reward(signal),
        source_type: "forward".to_string(),
        source_name: Some(signal.source.clone()),
        strategy_name: signal.strategy_id.clone(),
        strategy_score: None,
        model_name: None,
        model_version: None,
        model_confidence: Some(signal.confidence),
        indicators: None,
        metadata,
        filtered: false,
        is_backtest: false,
    }
}

/// Calculate risk/reward ratio from signal prices
fn calculate_risk_reward(signal: &janus_core::Signal) -> Option<f64> {
    match (signal.target_price, signal.stop_loss, signal.take_profit) {
        (Some(entry), Some(stop), Some(take)) => {
            let risk = (entry - stop).abs();
            let reward = (take - entry).abs();
            if risk > 0.0 {
                Some(reward / risk)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Persist a batch of signals to the database
async fn persist_signal_batch(
    signal_repo: &SignalRepository,
    batch: &[janus_core::Signal],
) -> usize {
    let mut persisted_count = 0;

    for signal in batch {
        let new_signal = signal_to_new_signal(signal);

        match signal_repo.create(new_signal).await {
            Ok(record) => {
                info!(
                    "Persisted signal {} for {} (type: {}, confidence: {:.2})",
                    record.signal_id, record.symbol, record.signal_type, record.confidence
                );
                persisted_count += 1;
            }
            Err(e) => {
                tracing::error!(
                    "Failed to persist signal {} for {}: {}",
                    signal.id,
                    signal.symbol,
                    e
                );
            }
        }
    }

    info!(
        "Persisted {}/{} signals in batch",
        persisted_count,
        batch.len()
    );
    persisted_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = BackwardServiceConfig::default();
        assert_eq!(config.http_port, 8082);
        assert_eq!(config.metrics_port, 9091);
        assert_eq!(config.worker_threads, 4);
        assert!(config.enable_scheduler);
    }

    #[test]
    fn test_service_constants() {
        assert_eq!(SERVICE_NAME, "janus-backward");
        assert!(!VERSION.is_empty());
    }
}
