//! FKS Execution Service - Main Entry Point
//!
//! This service handles order execution, position management, and risk controls.
//! It provides both gRPC (for service-to-service) and HTTP (for admin/monitoring) interfaces.

use janus_execution::{
    Config,
    api::{grpc::ExecutionServiceImpl, http},
    kill_switch_guard::{KillSwitchGuard, KillSwitchGuardConfig},
    notifications::NotificationManager,
    orders::OrderManager,
    state_broadcaster::{BroadcasterConfig, SharedExecutionState, StateBroadcaster},
};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::signal;
use tonic::transport::Server;
use tracing::{error, info, warn};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

// Redis for state broadcasting and kill switch guard

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing/logging
    init_logging();

    info!(
        "Starting JANUS Execution Service v{}",
        janus_execution::VERSION
    );

    // Load configuration
    let config = match Config::from_env() {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            return Err(e.into());
        }
    };

    // Validate configuration
    if let Err(e) = config.validate() {
        error!("Configuration validation failed: {}", e);
        return Err(e.into());
    }

    info!("Configuration loaded successfully");
    info!("Execution mode: {:?}", config.execution_mode.mode);
    info!("gRPC port: {}", config.service.grpc_port);
    info!("HTTP port: {}", config.service.http_port);

    // Connect to Redis for state broadcasting
    let redis_url = &config.redis.url;
    info!("Connecting to Redis at: {}", redis_url);

    let redis_client = redis::Client::open(redis_url.as_str()).map_err(|e| {
        error!("Failed to create Redis client: {}", e);
        e
    })?;

    let redis_conn = redis::aio::ConnectionManager::new(redis_client)
        .await
        .map_err(|e| {
            error!("Failed to connect to Redis: {}", e);
            e
        })?;

    info!("Redis connection established");

    // Initialize shared execution state
    let initial_equity = std::env::var("INITIAL_EQUITY")
        .unwrap_or_else(|_| "10000.0".to_string())
        .parse::<f64>()
        .unwrap_or(10000.0);

    let execution_state = SharedExecutionState::new(initial_equity);
    info!(
        "Initialized execution state with equity: ${:.2}",
        initial_equity
    );

    // Configure and start state broadcaster
    let broadcaster_config = BroadcasterConfig {
        interval: std::time::Duration::from_millis(100), // 10Hz updates
        channel_prefix: "janus.state".to_string(),
        verbose: cfg!(debug_assertions), // Verbose in debug mode only
    };

    let broadcaster = StateBroadcaster::new(
        redis_conn.clone(),
        execution_state.clone(),
        broadcaster_config,
    );

    info!("Starting state broadcaster (10Hz to Redis)");
    tokio::spawn(async move {
        if let Err(e) = broadcaster.run().await {
            error!("State broadcaster crashed: {}", e);
        }
    });

    // Initialize QuestDB connection for order history
    let questdb_host =
        std::env::var("QUESTDB_HOST").unwrap_or_else(|_| "localhost:9009".to_string());
    info!("QuestDB host: {}", questdb_host);

    // ── Kill Switch Guard (defense-in-depth) ────────────────────────────
    // Reads the shared `janus:kill_switch` key from Redis and blocks ALL
    // order submission when the kill switch is active. This is independent
    // of the forward service / brain pipeline kill switch — even if a
    // caller bypasses the brain and submits orders directly to execution,
    // they will be blocked.
    let kill_switch_config = KillSwitchGuardConfig::from_env();
    let kill_switch_guard = KillSwitchGuard::new(redis_conn.clone(), kill_switch_config);

    // Spawn the background sync task that polls Redis for state changes.
    // `spawn_sync_task` internally calls `tokio::spawn` and returns a JoinHandle.
    let _kill_switch_sync_handle = kill_switch_guard.spawn_sync_task();

    let kill_switch_gate: Arc<dyn janus_execution::kill_switch_guard::OrderGate> =
        Arc::new(kill_switch_guard.clone());

    info!(
        "Kill switch guard initialized (key='{}', poll={}ms, fail_closed={})",
        kill_switch_guard.is_active(), // log initial state
        std::env::var("EXEC_KILL_SWITCH_POLL_MS").unwrap_or_else(|_| "250".to_string()),
        std::env::var("EXEC_KILL_SWITCH_FAIL_CLOSED").unwrap_or_else(|_| "true".to_string()),
    );

    // Create order manager
    let order_manager = match OrderManager::new(questdb_host).await {
        Ok(mut mgr) => {
            // Wire the kill switch guard into the order manager
            mgr.set_order_gate(kill_switch_gate.clone());
            info!("Order manager initialized successfully (with kill switch guard)");
            Arc::new(mgr)
        }
        Err(e) => {
            warn!("Failed to initialize order manager with QuestDB: {}", e);
            warn!("Service will continue but order persistence may be unavailable");
            // In production, you might want to fail here, but for development we continue
            return Err(e.into());
        }
    };

    // Initialize notification manager
    let notification_manager = NotificationManager::from_env();
    if notification_manager.is_enabled() {
        info!("Discord notifications enabled");
    } else {
        info!("Discord notifications disabled (set DISCORD_ENABLE_NOTIFICATIONS=true to enable)");
    }
    let notification_manager = Arc::new(notification_manager);

    // Create gRPC service
    let grpc_service = ExecutionServiceImpl::new(order_manager.clone());
    let grpc_server = grpc_service.into_server();

    // Create HTTP router
    let http_state = http::HttpState::new(order_manager.clone())
        .with_notification_manager(notification_manager.clone());
    let http_router = http::create_router(http_state);

    // Parse server addresses
    let grpc_addr: SocketAddr = format!("0.0.0.0:{}", config.service.grpc_port)
        .parse()
        .expect("Invalid gRPC address");
    let http_addr: SocketAddr = format!("0.0.0.0:{}", config.service.http_port)
        .parse()
        .expect("Invalid HTTP address");

    info!("gRPC server starting on {}", grpc_addr);
    info!("HTTP server starting on {}", http_addr);

    // Start both servers concurrently
    let grpc_handle = tokio::spawn(async move {
        info!("gRPC server listening on {}", grpc_addr);
        if let Err(e) = Server::builder()
            .add_service(grpc_server)
            .serve(grpc_addr)
            .await
        {
            error!("gRPC server error: {}", e);
        }
    });

    let http_handle = tokio::spawn(async move {
        info!("HTTP server listening on {}", http_addr);
        let listener = tokio::net::TcpListener::bind(http_addr)
            .await
            .expect("Failed to bind HTTP server");

        if let Err(e) = axum::serve(listener, http_router).await {
            error!("HTTP server error: {}", e);
        }
    });

    info!("FKS Execution Service is ready");
    info!("  gRPC API:        http://{}", grpc_addr);
    info!("  HTTP API:        http://{}", http_addr);
    info!("  Health:          http://{}/health", http_addr);
    info!("  Metrics:         http://{}/metrics", http_addr);
    info!("  Redis Channels:  janus.state.equity, janus.state.volatility, janus.state.full");
    info!("  State Updates:   Broadcasting at 10Hz (every 100ms)");

    // Wait for shutdown signal
    tokio::select! {
        _ = signal::ctrl_c() => {
            info!("Received shutdown signal (Ctrl+C)");
        }
        _ = shutdown_signal() => {
            info!("Received termination signal");
        }
    }

    info!("Shutting down FKS Execution Service");

    // Give servers time to finish current requests
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Abort servers
    grpc_handle.abort();
    http_handle.abort();

    info!("FKS Execution Service stopped");

    Ok(())
}

/// Initialize logging with tracing
fn init_logging() {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new("info,fks_execution=debug"))
        .unwrap();

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_thread_ids(true)
                .with_level(true)
                .with_ansi(true)
                .compact(),
        )
        .init();
}

/// Wait for termination signal (SIGTERM)
#[cfg(unix)]
async fn shutdown_signal() {
    use tokio::signal::unix::{SignalKind, signal};

    let mut sigterm = signal(SignalKind::terminate()).expect("Failed to setup SIGTERM handler");
    sigterm.recv().await;
}

#[cfg(not(unix))]
async fn shutdown_signal() {
    // On non-Unix platforms, just wait forever (Ctrl+C is the only way)
    std::future::pending::<()>().await;
}
