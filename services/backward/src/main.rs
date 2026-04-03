//! # JANUS Backward Service - Main Entry Point
//!
//! Historical analysis, persistence, and analytics service.
//! Handles backward-looking operations including:
//! - Database persistence
//! - Historical analytics
//! - Performance metrics
//! - Backtesting
//! - Scheduled jobs

use anyhow::Result;
use janus_backward::{BackwardService, BackwardServiceConfig};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info,janus_backward=debug".to_string()),
        )
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true)
        .init();

    info!("╔═══════════════════════════════════════════════════════════╗");
    info!("║       JANUS BACKWARD SERVICE - Historical Analytics      ║");
    info!("║   Database • Performance • Metrics • Scheduled Jobs      ║");
    info!("╚═══════════════════════════════════════════════════════════╝");

    // Load configuration from environment
    let config = load_config()?;

    // Create service
    info!("Initializing Backward Service...");
    let mut service = BackwardService::new(config).await?;

    // Handle graceful shutdown
    let service_handle = tokio::spawn(async move {
        if let Err(e) = service.start().await {
            tracing::error!("Backward service error: {}", e);
        }
    });

    // Wait for shutdown signal
    tokio::signal::ctrl_c().await?;
    info!("Shutdown signal received, stopping service...");

    // Wait for service to stop
    service_handle.await?;

    info!("✅ JANUS Backward Service stopped successfully");
    Ok(())
}

/// Load configuration from environment variables
fn load_config() -> Result<BackwardServiceConfig> {
    use janus_backward::DatabaseConfig;

    // Build database URL from individual components or use DATABASE_URL if provided
    let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
        let host = std::env::var("DB_HOST").unwrap_or_else(|_| "localhost".to_string());
        let port = std::env::var("DB_PORT").unwrap_or_else(|_| "5432".to_string());
        let database = std::env::var("DB_NAME").unwrap_or_else(|_| "janus".to_string());
        let username = std::env::var("DB_USER").unwrap_or_else(|_| "postgres".to_string());
        let password = std::env::var("DB_PASSWORD").unwrap_or_else(|_| "postgres".to_string());
        format!(
            "postgresql://{}:{}@{}:{}/{}",
            username, password, host, port, database
        )
    });

    let database_config = DatabaseConfig {
        database_url,
        max_connections: std::env::var("DB_MAX_CONNECTIONS")
            .unwrap_or_else(|_| "10".to_string())
            .parse()
            .unwrap_or(10),
        min_connections: std::env::var("DB_MIN_CONNECTIONS")
            .unwrap_or_else(|_| "2".to_string())
            .parse()
            .unwrap_or(2),
        connect_timeout: std::env::var("DB_CONNECT_TIMEOUT")
            .unwrap_or_else(|_| "30".to_string())
            .parse()
            .unwrap_or(30),
        idle_timeout: std::env::var("DB_IDLE_TIMEOUT")
            .unwrap_or_else(|_| "600".to_string())
            .parse()
            .unwrap_or(600),
        max_lifetime: std::env::var("DB_MAX_LIFETIME")
            .unwrap_or_else(|_| "1800".to_string())
            .parse()
            .unwrap_or(1800),
        enable_logging: std::env::var("DB_ENABLE_LOGGING")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false),
    };

    let config = BackwardServiceConfig {
        host: std::env::var("BACKWARD_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
        http_port: std::env::var("BACKWARD_HTTP_PORT")
            .unwrap_or_else(|_| "8082".to_string())
            .parse()?,
        metrics_port: std::env::var("BACKWARD_METRICS_PORT")
            .unwrap_or_else(|_| "9091".to_string())
            .parse()?,
        redis_url: std::env::var("REDIS_URL")
            .unwrap_or_else(|_| "redis://localhost:6379".to_string()),
        worker_threads: std::env::var("BACKWARD_WORKERS")
            .unwrap_or_else(|_| "4".to_string())
            .parse()
            .unwrap_or(4),
        enable_metrics: std::env::var("BACKWARD_ENABLE_METRICS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true),
        enable_scheduler: std::env::var("BACKWARD_ENABLE_SCHEDULER")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true),
        database_config,
    };

    Ok(config)
}
