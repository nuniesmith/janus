//! Integration Tests for Parameter Hot-Reload Flow
//!
//! These tests verify the end-to-end flow of:
//! 1. Optimizer publishing params to Redis
//! 2. Forward service subscribing to updates
//! 3. ParamAppliers processing the updates
//! 4. SignalGenerator and RiskManager using the new params
//!
//! # Test Requirements
//!
//! - Redis must be running locally on `redis://127.0.0.1:6379`
//! - Set `REDIS_URL` environment variable to override
//!
//! # Running Tests
//!
//! ```bash
//! # Run with Redis available
//! cargo test -p janus-forward --test param_reload_integration
//!
//! # Run with custom Redis URL
//! REDIS_URL=redis://custom:6379 cargo test -p janus-forward --test param_reload_integration
//! ```
//!
//! # Test Coverage
//!
//! - Unit tests (no Redis): Applier conversions, handle creation, component wiring
//! - Integration tests (Redis): Pub/sub flow, stats tracking, concurrent updates
//! - Edge cases: Rapid updates, isolation, position limits, strategy constraints
//! - HA scenarios: Multiple managers subscribing to same instance

use janus_core::optimized_params::OptimizedParams;
use janus_forward::ParamQueryHandle;
use janus_forward::param_reload::{
    IndicatorParamApplier, ParamApplier, ParamReloadConfig, ParamReloadManager, RiskParamApplier,
    StrategyParamApplier,
};
use janus_forward::risk::RiskManager;
use janus_forward::signal::{SignalGenerator, SignalGeneratorConfig};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::timeout;

/// Get Redis URL from environment or use default
fn get_redis_url() -> String {
    std::env::var("REDIS_URL").unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string())
}

/// Generate a unique instance ID for test isolation
fn test_instance_id() -> String {
    format!("test_{}", &uuid::Uuid::new_v4().to_string()[..8])
}

/// Create test OptimizedParams for an asset
fn create_test_params(asset: &str) -> OptimizedParams {
    OptimizedParams {
        asset: asset.to_string(),
        ema_fast_period: 12,
        ema_slow_period: 26,
        atr_length: 14,
        atr_multiplier: 2.5,
        min_ema_spread_pct: 0.25,
        min_trailing_stop_pct: 1.5,
        take_profit_pct: 3.0,
        min_profit_pct: 0.5,
        max_position_size_usd: 5000.0,
        trade_cooldown_seconds: 300,
        enabled: true,
        require_htf_alignment: false,
        htf_timeframe_minutes: 240,
        min_hold_minutes: 5,
        prefer_trailing_stop_exit: true,
        optimized_at: chrono::Utc::now().to_rfc3339(),
        optimization_score: 0.75,
        backtest_result: Default::default(),
    }
}

/// Create test params with trading disabled
fn create_disabled_params(asset: &str) -> OptimizedParams {
    OptimizedParams {
        asset: asset.to_string(),
        ema_fast_period: 8,
        ema_slow_period: 21,
        atr_length: 14,
        atr_multiplier: 2.0,
        min_ema_spread_pct: 0.2,
        min_trailing_stop_pct: 1.0,
        take_profit_pct: 2.0,
        min_profit_pct: 0.3,
        max_position_size_usd: 1000.0,
        trade_cooldown_seconds: 600,
        enabled: false, // Trading disabled
        require_htf_alignment: false,
        htf_timeframe_minutes: 240,
        min_hold_minutes: 5,
        prefer_trailing_stop_exit: true,
        optimized_at: chrono::Utc::now().to_rfc3339(),
        optimization_score: 0.50,
        backtest_result: Default::default(),
    }
}

/// Helper to check if Redis is available
async fn redis_available() -> bool {
    let url = get_redis_url();
    match redis::Client::open(url) {
        Ok(client) => client.get_multiplexed_async_connection().await.is_ok(),
        Err(_) => false,
    }
}

/// Macro to skip tests if Redis is not available
macro_rules! require_redis {
    () => {
        if !redis_available().await {
            eprintln!("Skipping test: Redis not available at {}", get_redis_url());
            return;
        }
    };
}

// ============================================================================
// Unit Tests (No Redis Required)
// ============================================================================

#[tokio::test]
async fn test_param_applier_indicator_config_conversion() {
    let applier = IndicatorParamApplier::new();

    // Initially no config
    assert!(applier.get_config("BTC").await.is_none());

    // Apply params
    let params = create_test_params("BTC");
    applier.apply_params(&params).await.unwrap();

    // Now config should exist
    let config = applier.get_config("BTC").await;
    assert!(config.is_some());

    let config = config.unwrap();
    assert_eq!(config.ema_fast_period, 12);
    assert_eq!(config.ema_slow_period, 26);
    assert_eq!(config.atr_period, 14);
}

#[tokio::test]
async fn test_param_applier_risk_config_extraction() {
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let applier = RiskParamApplier::new(risk_manager);

    // Apply params
    let params = create_test_params("ETH");
    applier.apply_params(&params).await.unwrap();

    // Check trading enabled
    assert!(applier.is_trading_enabled("ETH").await);

    // Check max position size
    let max_pos = applier.get_max_position_size("ETH").await;
    assert_eq!(max_pos, Some(5000.0));

    // Check full config
    let config = applier.get_asset_config("ETH").await;
    assert!(config.is_some());
    let config = config.unwrap();
    assert_eq!(config.atr_multiplier, 2.5);
    assert_eq!(config.take_profit_pct, 3.0);
}

#[tokio::test]
async fn test_param_applier_strategy_constraints() {
    let applier = StrategyParamApplier::new();

    // Apply params with min_ema_spread_pct = 0.25
    let params = create_test_params("SOL");
    applier.apply_params(&params).await.unwrap();

    // Spread above minimum should pass
    assert!(applier.check_min_ema_spread("SOL", 0.30).await);

    // Spread below minimum should fail
    assert!(!applier.check_min_ema_spread("SOL", 0.20).await);

    // Exact minimum should pass
    assert!(applier.check_min_ema_spread("SOL", 0.25).await);
}

#[tokio::test]
async fn test_param_applier_trading_disabled() {
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let applier = RiskParamApplier::new(risk_manager);

    // Apply disabled params
    let params = create_disabled_params("DOGE");
    applier.apply_params(&params).await.unwrap();

    // Trading should be disabled
    assert!(!applier.is_trading_enabled("DOGE").await);
}

#[tokio::test]
async fn test_param_query_handle_creation() {
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    let query_handle = ParamQueryHandle {
        indicator_applier: indicator_applier.clone(),
        risk_applier: risk_applier.clone(),
        strategy_applier: strategy_applier.clone(),
    };

    // Apply some params directly to appliers
    let params = create_test_params("BTC");
    indicator_applier.apply_params(&params).await.unwrap();
    risk_applier.apply_params(&params).await.unwrap();
    strategy_applier.apply_params(&params).await.unwrap();

    // Query through handle
    let config = query_handle.get_indicator_config("BTC").await;
    assert!(config.is_some());

    assert!(query_handle.is_trading_enabled("BTC").await);
    assert!(query_handle.check_min_ema_spread("BTC", 0.30).await);
}

#[tokio::test]
async fn test_signal_generator_with_param_handle() {
    // Create components
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager.clone()));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    // Apply params with trading enabled
    let params = create_test_params("BTC");
    indicator_applier.apply_params(&params).await.unwrap();
    risk_applier.apply_params(&params).await.unwrap();
    strategy_applier.apply_params(&params).await.unwrap();

    // Create query handle
    let query_handle = ParamQueryHandle {
        indicator_applier,
        risk_applier,
        strategy_applier,
    };

    // Create signal generator with param handle
    let config = SignalGeneratorConfig::default();
    let mut signal_gen = SignalGenerator::new(config);
    signal_gen.set_param_query_handle(query_handle);

    // Verify param reload is configured
    assert!(signal_gen.has_param_reload());

    // Verify trading is enabled
    assert!(signal_gen.is_trading_enabled("BTC").await);

    // Verify indicator config is available
    let indicator_config = signal_gen.get_optimized_indicator_config("BTC").await;
    assert!(indicator_config.is_some());
}

#[tokio::test]
async fn test_signal_generator_blocks_disabled_asset() {
    // Create components
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager.clone()));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    // Apply params with trading DISABLED
    let params = create_disabled_params("SHIB");
    indicator_applier.apply_params(&params).await.unwrap();
    risk_applier.apply_params(&params).await.unwrap();
    strategy_applier.apply_params(&params).await.unwrap();

    // Create query handle
    let query_handle = ParamQueryHandle {
        indicator_applier,
        risk_applier,
        strategy_applier,
    };

    // Create signal generator with param handle
    let config = SignalGeneratorConfig::default();
    let mut signal_gen = SignalGenerator::new(config);
    signal_gen.set_param_query_handle(query_handle);

    // Trading should be disabled for SHIB
    assert!(!signal_gen.is_trading_enabled("SHIB").await);
}

#[tokio::test]
async fn test_risk_manager_with_param_handle() {
    // Create components
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager_inner = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager_inner.clone()));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    // Apply params
    let params = create_test_params("ETH");
    risk_applier.apply_params(&params).await.unwrap();

    // Create query handle
    let query_handle = ParamQueryHandle {
        indicator_applier,
        risk_applier,
        strategy_applier,
    };

    // Set param handle on risk manager
    {
        let mut rm = risk_manager_inner.write().await;
        rm.set_param_query_handle(query_handle);
    }

    // Verify risk manager uses optimized params
    {
        let rm = risk_manager_inner.read().await;
        assert!(rm.has_param_reload());
        assert!(rm.is_trading_enabled("ETH").await);

        let max_pos = rm.get_max_position_size_for_asset("ETH").await;
        assert_eq!(max_pos, 5000.0);
    }
}

#[tokio::test]
async fn test_multiple_asset_params() {
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    // Apply params for multiple assets
    let btc_params = create_test_params("BTC");
    let eth_params = OptimizedParams {
        asset: "ETH".to_string(),
        ema_fast_period: 10,
        ema_slow_period: 30,
        min_ema_spread_pct: 0.3,
        max_position_size_usd: 3000.0,
        ..create_test_params("ETH")
    };
    let disabled_params = create_disabled_params("DOGE");

    indicator_applier.apply_params(&btc_params).await.unwrap();
    indicator_applier.apply_params(&eth_params).await.unwrap();
    indicator_applier
        .apply_params(&disabled_params)
        .await
        .unwrap();

    risk_applier.apply_params(&btc_params).await.unwrap();
    risk_applier.apply_params(&eth_params).await.unwrap();
    risk_applier.apply_params(&disabled_params).await.unwrap();

    strategy_applier.apply_params(&btc_params).await.unwrap();
    strategy_applier.apply_params(&eth_params).await.unwrap();
    strategy_applier
        .apply_params(&disabled_params)
        .await
        .unwrap();

    // Verify each asset has correct config
    let btc_config = indicator_applier.get_config("BTC").await.unwrap();
    assert_eq!(btc_config.ema_fast_period, 12);

    let eth_config = indicator_applier.get_config("ETH").await.unwrap();
    assert_eq!(eth_config.ema_fast_period, 10);

    // Verify trading status
    assert!(risk_applier.is_trading_enabled("BTC").await);
    assert!(risk_applier.is_trading_enabled("ETH").await);
    assert!(!risk_applier.is_trading_enabled("DOGE").await);

    // Verify different spread thresholds
    assert!(strategy_applier.check_min_ema_spread("BTC", 0.25).await);
    assert!(!strategy_applier.check_min_ema_spread("ETH", 0.25).await); // ETH needs 0.3
    assert!(strategy_applier.check_min_ema_spread("ETH", 0.35).await);
}

// ============================================================================
// Integration Tests (Redis Required)
// ============================================================================

#[tokio::test]
async fn test_param_reload_manager_creation() {
    require_redis!();

    let instance_id = test_instance_id();
    let config = ParamReloadConfig {
        redis_url: get_redis_url(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 1000,
        max_reconnect_attempts: 3,
    };

    let manager = ParamReloadManager::new(config);

    // Register appliers
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    manager.register_applier(indicator_applier).await;

    // Stats should be empty
    let stats = manager.stats().await;
    assert_eq!(stats.total_received, 0);
    assert_eq!(stats.successful_applies, 0);
}

#[tokio::test]
async fn test_publish_and_receive_params() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // Create publisher
    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    // Create reload manager
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    // Register appliers
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    manager.register_applier(indicator_applier.clone()).await;
    manager.register_applier(risk_applier.clone()).await;
    manager.register_applier(strategy_applier.clone()).await;

    // Start background task
    let _handle = manager.clone().start_background_task().await.unwrap();

    // Give the subscriber time to connect
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Publish params
    let params = create_test_params("BTC");
    publisher.publish_params(&params).await.unwrap();

    // Wait for params to be received and applied
    let result = timeout(Duration::from_secs(5), async {
        loop {
            if indicator_applier.get_config("BTC").await.is_some() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(result.is_ok(), "Timeout waiting for params to be applied");

    // Verify params were applied correctly
    let config = indicator_applier.get_config("BTC").await.unwrap();
    assert_eq!(config.ema_fast_period, 12);
    assert_eq!(config.ema_slow_period, 26);

    // Verify risk applier
    assert!(risk_applier.is_trading_enabled("BTC").await);
    assert_eq!(
        risk_applier.get_max_position_size("BTC").await,
        Some(5000.0)
    );

    // Clean up
    manager.stop().await;

    // Clean up Redis keys
    let _ = publisher.delete_params("BTC").await;
}

#[tokio::test]
async fn test_load_initial_params_from_redis() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // First, publish some params to Redis
    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    let btc_params = create_test_params("BTC");
    let eth_params = create_test_params("ETH");

    publisher.publish_params(&btc_params).await.unwrap();
    publisher.publish_params(&eth_params).await.unwrap();

    // Now create a reload manager and load initial params
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    // Register appliers
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    manager.register_applier(indicator_applier.clone()).await;

    // Load initial params
    let count = manager.load_initial().await.unwrap();
    assert_eq!(count, 2); // BTC and ETH

    // Verify params were loaded
    assert!(indicator_applier.get_config("BTC").await.is_some());
    assert!(indicator_applier.get_config("ETH").await.is_some());

    // Clean up
    let _ = publisher.delete_params("BTC").await;
    let _ = publisher.delete_params("ETH").await;
}

#[tokio::test]
async fn test_param_update_propagates_to_signal_generator() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // Create publisher
    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    // Create reload manager
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    // Create appliers
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    manager.register_applier(indicator_applier.clone()).await;
    manager.register_applier(risk_applier.clone()).await;
    manager.register_applier(strategy_applier.clone()).await;

    // Create query handle
    let query_handle = ParamQueryHandle {
        indicator_applier: indicator_applier.clone(),
        risk_applier: risk_applier.clone(),
        strategy_applier: strategy_applier.clone(),
    };

    // Create signal generator with param handle
    let sg_config = SignalGeneratorConfig::default();
    let mut signal_gen = SignalGenerator::new(sg_config);
    signal_gen.set_param_query_handle(query_handle);

    // Initially no config for SOL
    assert!(
        signal_gen
            .get_optimized_indicator_config("SOL")
            .await
            .is_none()
    );

    // Start background task
    let _handle = manager.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Publish params for SOL
    let params = create_test_params("SOL");
    publisher.publish_params(&params).await.unwrap();

    // Wait for params to propagate
    let result = timeout(Duration::from_secs(5), async {
        loop {
            if signal_gen
                .get_optimized_indicator_config("SOL")
                .await
                .is_some()
            {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(
        result.is_ok(),
        "Timeout waiting for params to propagate to signal generator"
    );

    // Verify signal generator can now access the config
    let config = signal_gen
        .get_optimized_indicator_config("SOL")
        .await
        .unwrap();
    assert_eq!(config.ema_fast_period, 12);

    // Verify trading enabled check works
    assert!(signal_gen.is_trading_enabled("SOL").await);

    // Clean up
    manager.stop().await;
    let _ = publisher.delete_params("SOL").await;
}

#[tokio::test]
async fn test_disable_trading_via_param_update() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // Create publisher
    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    // Create reload manager
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    // Create appliers
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    manager.register_applier(indicator_applier.clone()).await;
    manager.register_applier(risk_applier.clone()).await;
    manager.register_applier(strategy_applier.clone()).await;

    // Create query handle and signal generator
    let query_handle = ParamQueryHandle {
        indicator_applier: indicator_applier.clone(),
        risk_applier: risk_applier.clone(),
        strategy_applier: strategy_applier.clone(),
    };

    let sg_config = SignalGeneratorConfig::default();
    let mut signal_gen = SignalGenerator::new(sg_config);
    signal_gen.set_param_query_handle(query_handle);

    // Start background task
    let _handle = manager.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // First publish enabled params
    let enabled_params = create_test_params("LINK");
    publisher.publish_params(&enabled_params).await.unwrap();

    // Wait for params
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert!(signal_gen.is_trading_enabled("LINK").await);

    // Now publish disabled params
    let disabled_params = create_disabled_params("LINK");
    publisher.publish_params(&disabled_params).await.unwrap();

    // Wait for update
    let result = timeout(Duration::from_secs(5), async {
        loop {
            if !signal_gen.is_trading_enabled("LINK").await {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(result.is_ok(), "Timeout waiting for trading to be disabled");

    // Clean up
    manager.stop().await;
    let _ = publisher.delete_params("LINK").await;
}

#[tokio::test]
async fn test_stats_tracking() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // Create publisher
    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    // Create reload manager
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    // Register appliers
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    manager.register_applier(indicator_applier.clone()).await;

    // Start background task
    let _handle = manager.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Initial stats
    let stats = manager.stats().await;
    assert_eq!(stats.total_received, 0);

    // Publish multiple params
    for asset in &["BTC", "ETH", "SOL"] {
        let params = create_test_params(asset);
        publisher.publish_params(&params).await.unwrap();
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // Wait for all to be processed
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Check stats
    let stats = manager.stats().await;
    assert!(
        stats.total_received >= 3,
        "Expected at least 3 received, got {}",
        stats.total_received
    );
    assert!(
        stats.successful_applies >= 3,
        "Expected at least 3 applies, got {}",
        stats.successful_applies
    );

    // Clean up
    manager.stop().await;
    for asset in &["BTC", "ETH", "SOL"] {
        let _ = publisher.delete_params(asset).await;
    }
}

// ============================================================================
// End-to-End Flow Test
// ============================================================================

#[tokio::test]
async fn test_full_hot_reload_flow() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // === Step 1: Create publisher (simulating optimizer) ===
    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    // === Step 2: Create Forward service components ===
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager.clone()));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    manager.register_applier(indicator_applier.clone()).await;
    manager.register_applier(risk_applier.clone()).await;
    manager.register_applier(strategy_applier.clone()).await;

    // === Step 3: Create query handle and wire to components ===
    let query_handle = ParamQueryHandle {
        indicator_applier: indicator_applier.clone(),
        risk_applier: risk_applier.clone(),
        strategy_applier: strategy_applier.clone(),
    };

    // Wire to signal generator
    let sg_config = SignalGeneratorConfig::default();
    let mut signal_gen = SignalGenerator::new(sg_config);
    signal_gen.set_param_query_handle(query_handle.clone());

    // Wire to risk manager
    {
        let mut rm = risk_manager.write().await;
        rm.set_param_query_handle(query_handle);
    }

    // === Step 4: Start background subscription ===
    let _handle = manager.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // === Step 5: Simulate optimizer publishing results ===

    // Notify optimization started
    publisher
        .notify_optimization_started(&["BTC".to_string(), "ETH".to_string()])
        .await
        .unwrap();

    // Publish BTC params
    let btc_params = OptimizedParams {
        asset: "BTC".to_string(),
        ema_fast_period: 8,
        ema_slow_period: 21,
        atr_length: 14,
        atr_multiplier: 2.0,
        min_ema_spread_pct: 0.15,
        min_trailing_stop_pct: 1.0,
        take_profit_pct: 2.5,
        min_profit_pct: 0.3,
        max_position_size_usd: 10000.0,
        trade_cooldown_seconds: 180,
        enabled: true,
        require_htf_alignment: false,
        htf_timeframe_minutes: 240,
        min_hold_minutes: 5,
        prefer_trailing_stop_exit: true,
        optimized_at: chrono::Utc::now().to_rfc3339(),
        optimization_score: 0.80,
        backtest_result: Default::default(),
    };
    publisher.publish_params(&btc_params).await.unwrap();

    // Publish ETH params
    let eth_params = OptimizedParams {
        asset: "ETH".to_string(),
        ema_fast_period: 10,
        ema_slow_period: 30,
        atr_length: 20,
        atr_multiplier: 2.5,
        min_ema_spread_pct: 0.20,
        min_trailing_stop_pct: 1.5,
        take_profit_pct: 3.0,
        min_profit_pct: 0.5,
        max_position_size_usd: 5000.0,
        trade_cooldown_seconds: 300,
        enabled: true,
        require_htf_alignment: false,
        htf_timeframe_minutes: 240,
        min_hold_minutes: 5,
        prefer_trailing_stop_exit: true,
        optimized_at: chrono::Utc::now().to_rfc3339(),
        optimization_score: 0.75,
        backtest_result: Default::default(),
    };
    publisher.publish_params(&eth_params).await.unwrap();

    // Notify optimization complete (create a batch result for the API)
    let batch_result = janus_optimizer::BatchPublishResult {
        total: 2,
        successful: vec!["BTC".to_string(), "ETH".to_string()],
        failed: vec![],
    };
    publisher
        .notify_optimization_complete(&batch_result)
        .await
        .unwrap();

    // === Step 6: Wait for params to be received ===
    let result = timeout(Duration::from_secs(5), async {
        loop {
            let btc_ready = indicator_applier.get_config("BTC").await.is_some();
            let eth_ready = indicator_applier.get_config("ETH").await.is_some();
            if btc_ready && eth_ready {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(result.is_ok(), "Timeout waiting for params");

    // === Step 7: Verify signal generator uses new params ===

    // Check BTC config
    let btc_config = signal_gen
        .get_optimized_indicator_config("BTC")
        .await
        .unwrap();
    assert_eq!(btc_config.ema_fast_period, 8);
    assert_eq!(btc_config.ema_slow_period, 21);

    // Check ETH config
    let eth_config = signal_gen
        .get_optimized_indicator_config("ETH")
        .await
        .unwrap();
    assert_eq!(eth_config.ema_fast_period, 10);
    assert_eq!(eth_config.ema_slow_period, 30);

    // Check trading enabled
    assert!(signal_gen.is_trading_enabled("BTC").await);
    assert!(signal_gen.is_trading_enabled("ETH").await);

    // Check EMA spread constraints
    assert!(signal_gen.check_strategy_constraints("BTC", 0.15).await);
    assert!(!signal_gen.check_strategy_constraints("BTC", 0.10).await);

    // === Step 8: Verify risk manager uses new params ===
    {
        let rm = risk_manager.read().await;

        // BTC max position
        let btc_max = rm.get_max_position_size_for_asset("BTC").await;
        assert_eq!(btc_max, 10000.0);

        // ETH max position
        let eth_max = rm.get_max_position_size_for_asset("ETH").await;
        assert_eq!(eth_max, 5000.0);
    }

    // === Step 9: Clean up ===
    manager.stop().await;
    let _ = publisher.delete_params("BTC").await;
    let _ = publisher.delete_params("ETH").await;

    println!("✅ Full hot-reload flow test passed!");
}

// ============================================================================
// Additional Edge Case Tests
// ============================================================================

#[tokio::test]
async fn test_concurrent_param_updates() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // Create publisher
    let publisher = Arc::new(
        janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
            .await
            .expect("Failed to create publisher"),
    );

    // Create reload manager
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager.clone()));

    manager.register_applier(indicator_applier.clone()).await;
    manager.register_applier(risk_applier.clone()).await;

    // Start background task
    let _handle = manager.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Publish updates concurrently for multiple assets
    let assets = vec!["BTC", "ETH", "SOL", "AVAX", "LINK"];
    let mut handles = vec![];

    for asset in &assets {
        let publisher_clone = publisher.clone();
        let asset_str = asset.to_string();
        let handle = tokio::spawn(async move {
            let params = OptimizedParams {
                asset: asset_str.clone(),
                ema_fast_period: 12,
                ema_slow_period: 26,
                atr_length: 14,
                atr_multiplier: 2.5,
                min_ema_spread_pct: 0.25,
                min_trailing_stop_pct: 1.5,
                take_profit_pct: 3.0,
                min_profit_pct: 0.5,
                max_position_size_usd: 5000.0,
                trade_cooldown_seconds: 300,
                enabled: true,
                require_htf_alignment: false,
                htf_timeframe_minutes: 240,
                min_hold_minutes: 5,
                prefer_trailing_stop_exit: true,
                optimized_at: chrono::Utc::now().to_rfc3339(),
                optimization_score: 0.70,
                backtest_result: Default::default(),
            };
            publisher_clone.publish_params(&params).await
        });
        handles.push(handle);
    }

    // Wait for all publishes
    for handle in handles {
        handle.await.unwrap().unwrap();
    }

    // Wait for all params to be applied
    let result = timeout(Duration::from_secs(10), async {
        loop {
            let mut all_ready = true;
            for asset in &assets {
                if indicator_applier.get_config(asset).await.is_none() {
                    all_ready = false;
                    break;
                }
            }
            if all_ready {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(result.is_ok(), "Timeout waiting for concurrent params");

    // Verify all assets have configs
    for asset in &assets {
        let config = indicator_applier.get_config(asset).await;
        assert!(config.is_some(), "Missing config for {}", asset);
    }

    // Clean up
    manager.stop().await;
    for asset in &assets {
        let _ = publisher.delete_params(asset).await;
    }

    println!("✅ Concurrent param updates test passed!");
}

#[tokio::test]
async fn test_rapid_param_version_updates() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    // Create publisher
    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    // Create reload manager
    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    manager.register_applier(indicator_applier.clone()).await;

    // Start background task
    let _handle = manager.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Rapidly publish multiple versions of params
    for version in 1..=5u32 {
        let params = OptimizedParams {
            asset: "BTC".to_string(),
            ema_fast_period: 8 + version,
            ema_slow_period: 21 + version,
            atr_length: 14,
            atr_multiplier: 2.0,
            min_ema_spread_pct: 0.15,
            min_trailing_stop_pct: 1.0,
            take_profit_pct: 2.5,
            min_profit_pct: 0.3,
            max_position_size_usd: 10000.0,
            trade_cooldown_seconds: 180,
            enabled: true,
            require_htf_alignment: false,
            htf_timeframe_minutes: 240,
            min_hold_minutes: 5,
            prefer_trailing_stop_exit: true,
            optimized_at: chrono::Utc::now().to_rfc3339(),
            optimization_score: 0.70 + (version as f64 * 0.02),
            backtest_result: Default::default(),
        };
        publisher.publish_params(&params).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Wait for final version to be applied
    let result = timeout(Duration::from_secs(5), async {
        loop {
            if let Some(config) = indicator_applier.get_config("BTC").await {
                // Final version should have ema_fast_period = 8 + 5 = 13
                if config.ema_fast_period == 13 {
                    return true;
                }
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(result.is_ok(), "Timeout waiting for final version");

    // Clean up
    manager.stop().await;
    let _ = publisher.delete_params("BTC").await;

    println!("✅ Rapid version updates test passed!");
}

#[tokio::test]
async fn test_param_applier_isolation() {
    // Test that appliers are isolated and don't interfere with each other
    let indicator_applier1 = Arc::new(IndicatorParamApplier::new());
    let indicator_applier2 = Arc::new(IndicatorParamApplier::new());

    // Apply params to applier1 only
    let params = create_test_params("BTC");
    indicator_applier1.apply_params(&params).await.unwrap();

    // Applier1 should have the config
    assert!(indicator_applier1.get_config("BTC").await.is_some());

    // Applier2 should NOT have the config
    assert!(indicator_applier2.get_config("BTC").await.is_none());

    println!("✅ Param applier isolation test passed!");
}

#[tokio::test]
async fn test_signal_generator_without_param_handle() {
    // SignalGenerator should work without param handle (uses defaults)
    let config = SignalGeneratorConfig::default();
    let signal_gen = SignalGenerator::new(config);

    // Without param handle, these should return defaults
    assert!(!signal_gen.has_param_reload());

    // Trading should be enabled by default (no restrictions)
    assert!(signal_gen.is_trading_enabled("BTC").await);

    // No optimized config available
    assert!(
        signal_gen
            .get_optimized_indicator_config("BTC")
            .await
            .is_none()
    );

    println!("✅ SignalGenerator without param handle test passed!");
}

#[tokio::test]
async fn test_risk_manager_position_limits() {
    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    let risk_manager_inner = Arc::new(RwLock::new(RiskManager::new(Default::default())));
    let risk_applier = Arc::new(RiskParamApplier::new(risk_manager_inner.clone()));
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    // Apply params with specific position limits
    let btc_params = OptimizedParams {
        asset: "BTC".to_string(),
        max_position_size_usd: 10000.0,
        ..create_test_params("BTC")
    };

    let eth_params = OptimizedParams {
        asset: "ETH".to_string(),
        max_position_size_usd: 5000.0,
        ..create_test_params("ETH")
    };

    risk_applier.apply_params(&btc_params).await.unwrap();
    risk_applier.apply_params(&eth_params).await.unwrap();

    // Create query handle
    let query_handle = ParamQueryHandle {
        indicator_applier,
        risk_applier: risk_applier.clone(),
        strategy_applier,
    };

    // Set param handle on risk manager
    {
        let mut rm = risk_manager_inner.write().await;
        rm.set_param_query_handle(query_handle);
    }

    // Verify different limits per asset
    {
        let rm = risk_manager_inner.read().await;
        assert_eq!(rm.get_max_position_size_for_asset("BTC").await, 10000.0);
        assert_eq!(rm.get_max_position_size_for_asset("ETH").await, 5000.0);

        // Unknown asset should use default
        let unknown_max = rm.get_max_position_size_for_asset("UNKNOWN").await;
        // Default is from RiskConfig
        assert!(unknown_max > 0.0);
    }

    println!("✅ Risk manager position limits test passed!");
}

#[tokio::test]
async fn test_strategy_constraints_edge_cases() {
    let strategy_applier = Arc::new(StrategyParamApplier::new());

    // Apply params with various spread requirements
    let tight_spread = OptimizedParams {
        asset: "BTC".to_string(),
        min_ema_spread_pct: 0.10,
        ..create_test_params("BTC")
    };

    let wide_spread = OptimizedParams {
        asset: "SHIB".to_string(),
        min_ema_spread_pct: 0.50,
        ..create_test_params("SHIB")
    };

    strategy_applier.apply_params(&tight_spread).await.unwrap();
    strategy_applier.apply_params(&wide_spread).await.unwrap();

    // BTC with tight spread requirement
    assert!(strategy_applier.check_min_ema_spread("BTC", 0.10).await); // Exactly at min
    assert!(strategy_applier.check_min_ema_spread("BTC", 0.15).await); // Above min
    assert!(!strategy_applier.check_min_ema_spread("BTC", 0.09).await); // Below min
    assert!(!strategy_applier.check_min_ema_spread("BTC", 0.0).await); // Zero

    // SHIB with wide spread requirement
    assert!(!strategy_applier.check_min_ema_spread("SHIB", 0.40).await); // Below
    assert!(strategy_applier.check_min_ema_spread("SHIB", 0.50).await); // At min
    assert!(strategy_applier.check_min_ema_spread("SHIB", 1.00).await); // Well above

    // Unknown asset uses default (0.2% minimum spread)
    assert!(!strategy_applier.check_min_ema_spread("UNKNOWN", 0.10).await); // Below default
    assert!(strategy_applier.check_min_ema_spread("UNKNOWN", 0.20).await); // At default
    assert!(strategy_applier.check_min_ema_spread("UNKNOWN", 0.50).await); // Above default

    println!("✅ Strategy constraints edge cases test passed!");
}

#[tokio::test]
async fn test_param_reload_manager_stats_accuracy() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    let config = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager = Arc::new(ParamReloadManager::new(config));

    let indicator_applier = Arc::new(IndicatorParamApplier::new());
    manager.register_applier(indicator_applier.clone()).await;

    // Initial stats should be zero
    let initial_stats = manager.stats().await;
    assert_eq!(initial_stats.total_received, 0);
    assert_eq!(initial_stats.successful_applies, 0);
    assert_eq!(initial_stats.failed_applies, 0);

    // Start background task
    let _handle = manager.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Publish valid params
    let params = create_test_params("BTC");
    publisher.publish_params(&params).await.unwrap();
    tokio::time::sleep(Duration::from_millis(500)).await;

    // Stats should show the received and applied param
    let after_stats = manager.stats().await;
    assert!(
        after_stats.total_received >= 1,
        "Expected at least 1 received"
    );
    assert!(
        after_stats.successful_applies >= 1,
        "Expected at least 1 successful apply"
    );

    // Clean up
    manager.stop().await;
    let _ = publisher.delete_params("BTC").await;

    println!("✅ Param reload manager stats accuracy test passed!");
}

#[tokio::test]
async fn test_multiple_managers_same_instance() {
    require_redis!();

    let instance_id = test_instance_id();
    let redis_url = get_redis_url();

    let publisher = janus_optimizer::ParamPublisher::new(&redis_url, &instance_id)
        .await
        .expect("Failed to create publisher");

    // Create two managers with the same instance ID (simulating HA setup)
    let config1 = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let config2 = ParamReloadConfig {
        redis_url: redis_url.clone(),
        instance_id: instance_id.clone(),
        enabled: true,
        reconnect_delay_ms: 100,
        max_reconnect_attempts: 3,
    };

    let manager1 = Arc::new(ParamReloadManager::new(config1));
    let manager2 = Arc::new(ParamReloadManager::new(config2));

    let applier1 = Arc::new(IndicatorParamApplier::new());
    let applier2 = Arc::new(IndicatorParamApplier::new());

    manager1.register_applier(applier1.clone()).await;
    manager2.register_applier(applier2.clone()).await;

    // Start both managers
    let _handle1 = manager1.clone().start_background_task().await.unwrap();
    let _handle2 = manager2.clone().start_background_task().await.unwrap();
    tokio::time::sleep(Duration::from_millis(300)).await;

    // Publish params
    let params = create_test_params("ETH");
    publisher.publish_params(&params).await.unwrap();

    // Both managers should receive the update
    let result = timeout(Duration::from_secs(5), async {
        loop {
            let m1_ready = applier1.get_config("ETH").await.is_some();
            let m2_ready = applier2.get_config("ETH").await.is_some();
            if m1_ready && m2_ready {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    })
    .await;

    assert!(result.is_ok(), "Both managers should receive params");

    // Verify both have the same config
    let config1 = applier1.get_config("ETH").await.unwrap();
    let config2 = applier2.get_config("ETH").await.unwrap();
    assert_eq!(config1.ema_fast_period, config2.ema_fast_period);

    // Clean up
    manager1.stop().await;
    manager2.stop().await;
    let _ = publisher.delete_params("ETH").await;

    println!("✅ Multiple managers same instance test passed!");
}
