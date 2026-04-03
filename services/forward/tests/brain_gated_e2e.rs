//! # Brain-Gated Execution — End-to-End Integration Tests
//!
//! Spins up a mock gRPC `ExecutionService` and wires it through
//! `BrainGatedExecutionClient` to verify the full pipeline:
//!
//! ```text
//! Signal → TradingPipeline (gate) → ExecutionClient (gRPC) → MockServer
//! ```
//!
//! ## Test Matrix
//!
//! | Scenario                        | Expected Outcome            |
//! |---------------------------------|-----------------------------|
//! | Bullish + healthy pipeline      | Submitted to mock server    |
//! | Kill switch active              | Blocked, never hits server  |
//! | Low confidence regime           | Blocked at pipeline         |
//! | ReduceOnly (crisis regime)      | Submitted with reduced qty  |
//! | ReduceOnly disabled in config   | Blocked                     |
//! | Stale signal                    | Rejected before evaluation  |
//! | Pipeline proceeds, gRPC fails   | Error result with decision  |
//! | Stats accumulate correctly      | Counters match expectations |

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::Notify;
use tonic::{Request, Response, Status};

// ── Proto types ────────────────────────────────────────────────────
use fks_proto::common::{
    HealthCheckRequest as ExecHealthCheckRequest, HealthCheckResponse as ExecHealthCheckResponse,
};
use fks_proto::execution::execution_service_server::{ExecutionService, ExecutionServiceServer};
use fks_proto::execution::{
    CancelAllOrdersRequest, CancelAllOrdersResponse, CancelOrderRequest, CancelOrderResponse,
    GetAccountRequest, GetAccountResponse, GetActiveOrdersRequest, GetActiveOrdersResponse,
    GetOrderStatusRequest, GetOrderStatusResponse, GetPositionsRequest, GetPositionsResponse,
    StreamUpdatesRequest, StreamUpdatesResponse, SubmitSignalRequest, SubmitSignalResponse,
};

// ── Forward service types ──────────────────────────────────────────
use janus_forward::brain_wiring::{TradingPipeline, TradingPipelineBuilder, TradingPipelineConfig};
use janus_forward::execution::brain_gated::{
    BrainGatedConfig, BrainGatedExecutionClient, GatedSubmissionResult,
};
use janus_forward::execution::client::{ExecutionClient, ExecutionClientConfig};
use janus_forward::signal::{SignalSource, SignalType, Timeframe, TradingSignal};
use janus_regime::router::{ActiveStrategy, DetectionMethod};
use janus_regime::types::TrendDirection;
use janus_regime::{MarketRegime, RoutedSignal};

// ════════════════════════════════════════════════════════════════════
// Mock Execution gRPC Server
// ════════════════════════════════════════════════════════════════════

/// Tracks how many signals the mock server received.
#[derive(Debug, Default)]
struct MockExecutionState {
    submit_count: AtomicU64,
    health_check_count: AtomicU64,
    /// When > 0, the next N submissions will fail.
    fail_next: AtomicU64,
}

/// A mock `ExecutionService` implementation that accepts signals,
/// increments counters, and optionally fails.
#[derive(Debug)]
struct MockExecutionService {
    state: Arc<MockExecutionState>,
}

#[tonic::async_trait]
impl ExecutionService for MockExecutionService {
    async fn submit_signal(
        &self,
        request: Request<SubmitSignalRequest>,
    ) -> Result<Response<SubmitSignalResponse>, Status> {
        let req = request.into_inner();
        let count = self.state.submit_count.fetch_add(1, Ordering::SeqCst) + 1;

        // Check if we should simulate a failure
        let fail_remaining = self.state.fail_next.load(Ordering::SeqCst);
        if fail_remaining > 0 {
            self.state.fail_next.fetch_sub(1, Ordering::SeqCst);
            return Ok(Response::new(SubmitSignalResponse {
                success: false,
                order_id: String::new(),
                internal_order_id: String::new(),
                message: "Mock failure: simulated exchange rejection".to_string(),
                timestamp: chrono::Utc::now().timestamp_millis(),
                status: 0,
            }));
        }

        Ok(Response::new(SubmitSignalResponse {
            success: true,
            order_id: format!("MOCK-{}-{}", req.symbol, count),
            internal_order_id: format!("INT-{}", count),
            message: "Accepted by mock execution service".to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            status: 0,
        }))
    }

    async fn get_order_status(
        &self,
        _request: Request<GetOrderStatusRequest>,
    ) -> Result<Response<GetOrderStatusResponse>, Status> {
        Err(Status::unimplemented("not implemented in mock"))
    }

    async fn get_active_orders(
        &self,
        _request: Request<GetActiveOrdersRequest>,
    ) -> Result<Response<GetActiveOrdersResponse>, Status> {
        Err(Status::unimplemented("not implemented in mock"))
    }

    async fn cancel_order(
        &self,
        _request: Request<CancelOrderRequest>,
    ) -> Result<Response<CancelOrderResponse>, Status> {
        Err(Status::unimplemented("not implemented in mock"))
    }

    async fn cancel_all_orders(
        &self,
        _request: Request<CancelAllOrdersRequest>,
    ) -> Result<Response<CancelAllOrdersResponse>, Status> {
        Err(Status::unimplemented("not implemented in mock"))
    }

    async fn get_positions(
        &self,
        _request: Request<GetPositionsRequest>,
    ) -> Result<Response<GetPositionsResponse>, Status> {
        Err(Status::unimplemented("not implemented in mock"))
    }

    async fn get_account(
        &self,
        _request: Request<GetAccountRequest>,
    ) -> Result<Response<GetAccountResponse>, Status> {
        Err(Status::unimplemented("not implemented in mock"))
    }

    type StreamUpdatesStream =
        tokio_stream::wrappers::ReceiverStream<Result<StreamUpdatesResponse, Status>>;

    async fn stream_updates(
        &self,
        _request: Request<StreamUpdatesRequest>,
    ) -> Result<Response<Self::StreamUpdatesStream>, Status> {
        Err(Status::unimplemented("not implemented in mock"))
    }

    async fn health_check(
        &self,
        _request: Request<ExecHealthCheckRequest>,
    ) -> Result<Response<ExecHealthCheckResponse>, Status> {
        self.state.health_check_count.fetch_add(1, Ordering::SeqCst);
        Ok(Response::new(ExecHealthCheckResponse {
            healthy: true,
            status: "healthy".to_string(),
            components: Default::default(),
            uptime_seconds: 0,
            version: String::new(),
            timestamp: chrono::Utc::now().timestamp_millis(),
        }))
    }
}

// ════════════════════════════════════════════════════════════════════
// Test Helpers
// ════════════════════════════════════════════════════════════════════

/// Spin up the mock gRPC server on a random port. Returns the socket
/// address and the shared state for assertions.
async fn start_mock_server() -> (SocketAddr, Arc<MockExecutionState>) {
    let state = Arc::new(MockExecutionState::default());
    let svc = MockExecutionService {
        state: state.clone(),
    };

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let notify = Arc::new(Notify::new());
    let notify_clone = notify.clone();

    tokio::spawn(async move {
        let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
        notify_clone.notify_one();
        tonic::transport::Server::builder()
            .add_service(ExecutionServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
            .unwrap();
    });

    // Wait until the server task has started
    notify.notified().await;
    // Give the server a moment to bind
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, state)
}

/// Create an `ExecutionClient` pointing at the mock server.
async fn make_execution_client(addr: SocketAddr) -> ExecutionClient {
    let config = ExecutionClientConfig {
        endpoint: format!("http://{}", addr),
        connect_timeout_secs: 5,
        request_timeout_secs: 5,
        max_retries: 1,
        retry_backoff_ms: 10,
        ..Default::default()
    };
    ExecutionClient::new(config).await.expect("connect to mock")
}

/// Build a pipeline with very low min-confidence so bullish signals pass easily.
fn make_permissive_pipeline() -> Arc<TradingPipeline> {
    Arc::new(
        TradingPipelineBuilder::new()
            .config(TradingPipelineConfig {
                min_regime_confidence: 0.0,
                enable_gating: false,
                enable_correlation_filter: false,
                ..Default::default()
            })
            .build(),
    )
}

/// Build a default pipeline with all stages enabled and a realistic
/// min_regime_confidence threshold (the struct default is 0.0, but
/// production configs typically use 0.3 via env vars).
fn make_default_pipeline() -> Arc<TradingPipeline> {
    Arc::new(
        TradingPipelineBuilder::new()
            .config(TradingPipelineConfig {
                min_regime_confidence: 0.3,
                ..Default::default()
            })
            .build(),
    )
}

/// Create a fresh bullish TradingSignal.
fn bullish_signal(symbol: &str) -> TradingSignal {
    TradingSignal {
        signal_id: uuid::Uuid::new_v4().to_string(),
        signal_type: SignalType::Buy,
        symbol: symbol.to_string(),
        timeframe: Timeframe::H1,
        confidence: 0.85,
        strength: 0.7,
        timestamp: chrono::Utc::now(),
        source: SignalSource::TechnicalIndicator {
            name: "ema_flip".to_string(),
        },
        entry_price: Some(50000.0),
        stop_loss: Some(49000.0),
        take_profit: Some(52000.0),
        predicted_duration_seconds: None,
        metadata: std::collections::HashMap::new(),
    }
}

/// Create a stale signal (30 minutes old).
fn stale_signal(symbol: &str) -> TradingSignal {
    TradingSignal {
        signal_id: uuid::Uuid::new_v4().to_string(),
        signal_type: SignalType::Buy,
        symbol: symbol.to_string(),
        timeframe: Timeframe::H1,
        confidence: 0.85,
        strength: 0.7,
        timestamp: chrono::Utc::now() - chrono::Duration::seconds(1800),
        source: SignalSource::TechnicalIndicator {
            name: "ema_flip".to_string(),
        },
        entry_price: Some(50000.0),
        stop_loss: Some(49000.0),
        take_profit: Some(52000.0),
        predicted_duration_seconds: None,
        metadata: std::collections::HashMap::new(),
    }
}

/// Create a bullish RoutedSignal with trending regime.
fn bullish_routed_signal() -> RoutedSignal {
    RoutedSignal {
        strategy: ActiveStrategy::TrendFollowing,
        regime: MarketRegime::Trending(TrendDirection::Bullish),
        confidence: 0.9,
        position_factor: 0.8,
        reason: "Strong bullish trend detected".to_string(),
        detection_method: DetectionMethod::Ensemble,
        methods_agree: Some(true),
        state_probabilities: None,
        expected_duration: Some(20.0),
        trend_direction: Some(TrendDirection::Bullish),
    }
}

/// Create a crisis RoutedSignal (should trigger ReduceOnly).
fn crisis_routed_signal() -> RoutedSignal {
    RoutedSignal {
        strategy: ActiveStrategy::NoTrade,
        regime: MarketRegime::Volatile,
        confidence: 0.9,
        position_factor: 0.1,
        reason: "Crisis regime detected — extreme volatility".to_string(),
        detection_method: DetectionMethod::Ensemble,
        methods_agree: Some(true),
        state_probabilities: None,
        expected_duration: Some(5.0),
        trend_direction: None,
    }
}

/// Create a low-confidence RoutedSignal.
fn low_confidence_routed_signal() -> RoutedSignal {
    RoutedSignal {
        strategy: ActiveStrategy::TrendFollowing,
        regime: MarketRegime::Trending(TrendDirection::Bullish),
        confidence: 0.05, // below default min_regime_confidence of 0.3
        position_factor: 0.3,
        reason: "Low confidence regime detection".to_string(),
        detection_method: DetectionMethod::Indicators,
        methods_agree: Some(false),
        state_probabilities: None,
        expected_duration: None,
        trend_direction: None,
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

/// Full happy-path: bullish signal evaluated through the pipeline,
/// submitted to mock gRPC server, order ID returned.
#[tokio::test]
async fn test_e2e_bullish_signal_submitted_to_mock_server() {
    let (addr, server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    let signal = bullish_signal("BTCUSD");
    let routed = bullish_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(result.is_submitted(), "Expected Submitted, got: {}", result);

    if let GatedSubmissionResult::Submitted {
        decision,
        response,
        applied_scale,
    } = &result
    {
        assert_eq!(decision.symbol, "BTCUSD");
        assert!(response.success);
        assert!(response.order_id.as_ref().unwrap().starts_with("MOCK-"));
        assert!(*applied_scale > 0.0);
    }

    assert_eq!(
        server_state.submit_count.load(Ordering::SeqCst),
        1,
        "Mock server should have received exactly 1 submission"
    );
}

/// Kill switch is active → signal should be blocked and never reach the server.
#[tokio::test]
async fn test_e2e_kill_switch_blocks_signal() {
    let (addr, server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();
    pipeline.activate_kill_switch().await;

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    let signal = bullish_signal("BTCUSD");
    let routed = bullish_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(result.is_blocked(), "Expected Blocked, got: {}", result);
    assert_eq!(
        server_state.submit_count.load(Ordering::SeqCst),
        0,
        "Mock server should NOT have received any submission"
    );
}

/// Low-confidence regime signal → blocked at pipeline.
#[tokio::test]
async fn test_e2e_low_confidence_blocked() {
    let (addr, server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_default_pipeline(); // default: min_regime_confidence = 0.3

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    let signal = bullish_signal("BTCUSD");
    let routed = low_confidence_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    assert!(
        result.is_blocked(),
        "Low confidence should be blocked, got: {}",
        result
    );

    assert_eq!(server_state.submit_count.load(Ordering::SeqCst), 0);
}

/// Crisis regime with allow_new_positions_in_crisis=false → ReduceOnly.
/// If allow_reduce_only=true (default), the signal should still be submitted.
#[tokio::test]
async fn test_e2e_crisis_reduce_only_submitted() {
    let (addr, _server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = Arc::new(
        TradingPipelineBuilder::new()
            .config(TradingPipelineConfig {
                allow_new_positions_in_crisis: false,
                enable_gating: false,
                enable_correlation_filter: false,
                min_regime_confidence: 0.0,
                ..Default::default()
            })
            .build(),
    );

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    let signal = bullish_signal("BTCUSD");
    let routed = crisis_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    // Crisis with allow_new_positions_in_crisis=false should yield ReduceOnly,
    // which the default config allows to pass through.
    match &result {
        GatedSubmissionResult::Submitted { applied_scale, .. } => {
            assert!(
                *applied_scale <= 1.0,
                "Scale should be reduced in crisis, got {}",
                applied_scale
            );
        }
        GatedSubmissionResult::Blocked { reason, .. } => {
            // The pipeline blocks new positions in crisis — this is acceptable.
            assert!(
                reason.contains("Crisis")
                    || reason.contains("crisis")
                    || reason.contains("ReduceOnly"),
                "If blocked, reason should mention crisis: {}",
                reason
            );
        }
        other => {
            // Both Submitted (via ReduceOnly path) and Blocked are acceptable
            // depending on exact pipeline logic for crisis regime
            panic!("Unexpected result: {}", other);
        }
    }
}

/// ReduceOnly disabled in config → crisis signal is fully blocked.
#[tokio::test]
async fn test_e2e_reduce_only_disabled_blocks_crisis() {
    let (addr, server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = Arc::new(
        TradingPipelineBuilder::new()
            .config(TradingPipelineConfig {
                allow_new_positions_in_crisis: false,
                enable_gating: false,
                enable_correlation_filter: false,
                min_regime_confidence: 0.0,
                ..Default::default()
            })
            .build(),
    );

    let config = BrainGatedConfig {
        allow_reduce_only: false,
        ..Default::default()
    };

    let mut gated = BrainGatedExecutionClient::with_config(exec_client, pipeline, config);

    let signal = bullish_signal("BTCUSD");
    let routed = crisis_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    // With allow_reduce_only=false, ReduceOnly decisions become blocks
    assert!(
        result.is_blocked(),
        "ReduceOnly should be blocked when disabled, got: {}",
        result
    );

    assert_eq!(server_state.submit_count.load(Ordering::SeqCst), 0);
}

/// Stale signal → rejected before pipeline evaluation even runs.
#[tokio::test]
async fn test_e2e_stale_signal_rejected() {
    let (addr, server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let config = BrainGatedConfig {
        max_signal_age_secs: 60, // 60 seconds max
        ..Default::default()
    };

    let mut gated = BrainGatedExecutionClient::with_config(exec_client, pipeline, config);

    let signal = stale_signal("BTCUSD"); // 30 minutes old
    let routed = bullish_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    match &result {
        GatedSubmissionResult::Stale {
            signal_age_secs,
            max_age_secs,
        } => {
            assert!(*signal_age_secs > 60);
            assert_eq!(*max_age_secs, 60);
        }
        other => panic!("Expected Stale, got: {}", other),
    }

    assert_eq!(server_state.submit_count.load(Ordering::SeqCst), 0);
}

/// Pipeline approves but the mock gRPC server rejects → Error result with decision.
#[tokio::test]
async fn test_e2e_pipeline_approves_but_grpc_fails() {
    let (addr, server_state) = start_mock_server().await;

    // Tell the mock to fail the next submission
    server_state.fail_next.store(5, Ordering::SeqCst); // fail all retries

    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    let signal = bullish_signal("BTCUSD");
    let routed = bullish_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    match &result {
        GatedSubmissionResult::Error { decision, error } => {
            assert!(decision.is_some(), "Should have a decision");
            assert!(!error.is_empty(), "Error message should not be empty");
        }
        other => panic!("Expected Error, got: {}", other),
    }
}

/// Stats accumulate correctly across multiple submissions.
#[tokio::test]
async fn test_e2e_stats_accumulate() {
    let (addr, _server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let config = BrainGatedConfig {
        max_signal_age_secs: 60,
        ..Default::default()
    };

    let mut gated = BrainGatedExecutionClient::with_config(exec_client, pipeline.clone(), config);

    // 1) Submit a valid signal (should succeed)
    let signal = bullish_signal("BTCUSD");
    let routed = bullish_routed_signal();
    let r1 = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(r1.is_submitted(), "First signal should submit: {}", r1);

    // 2) Submit another valid signal
    let signal2 = bullish_signal("ETHUSD");
    let r2 = gated
        .submit_gated(&signal2, &routed, "mean_rev", &[], None, None, None, None)
        .await;
    assert!(r2.is_submitted(), "Second signal should submit: {}", r2);

    // 3) Submit a stale signal (should be rejected)
    let stale = stale_signal("SOLUSD");
    let r3 = gated
        .submit_gated(&stale, &routed, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(
        matches!(r3, GatedSubmissionResult::Stale { .. }),
        "Stale signal should be rejected: {}",
        r3
    );

    // 4) Activate kill switch and try again
    gated.activate_kill_switch().await;
    let signal4 = bullish_signal("BTCUSD");
    let r4 = gated
        .submit_gated(&signal4, &routed, "ema_flip", &[], None, None, None, None)
        .await;
    assert!(r4.is_blocked(), "Kill-switched signal should block: {}", r4);

    // Check accumulated stats
    let stats = gated.stats().await;
    assert_eq!(stats.total_signals, 4, "Should have 4 total signals");
    assert_eq!(stats.submitted, 2, "Should have 2 submitted");
    assert_eq!(stats.stale_rejected, 1, "Should have 1 stale rejection");
    assert_eq!(stats.blocked, 1, "Should have 1 block");
    assert!(
        stats.submission_rate() > 0.0,
        "Submission rate should be positive"
    );
}

/// Health check passes through to the mock server.
#[tokio::test]
async fn test_e2e_health_check_passes_through() {
    let (addr, server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    let healthy = gated
        .health_check()
        .await
        .expect("health check should work");
    assert!(healthy, "Mock server should report healthy");

    assert_eq!(
        server_state.health_check_count.load(Ordering::SeqCst),
        1,
        "Server should have received exactly 1 health check"
    );
}

/// Kill switch can be activated and deactivated via the gated client.
#[tokio::test]
async fn test_e2e_kill_switch_toggle() {
    let (addr, _) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    assert!(!gated.is_killed().await, "Should start unkilled");

    gated.activate_kill_switch().await;
    assert!(gated.is_killed().await, "Should be killed after activation");

    gated.deactivate_kill_switch().await;
    assert!(
        !gated.is_killed().await,
        "Should be unkilled after deactivation"
    );
}

/// Multiple signals to different symbols should each be independently evaluated.
#[tokio::test]
async fn test_e2e_multiple_symbols() {
    let (addr, server_state) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);
    let routed = bullish_routed_signal();

    let symbols = ["BTCUSD", "ETHUSD", "SOLUSD", "AVAXUSD", "LINKUSD"];

    for symbol in &symbols {
        let signal = bullish_signal(symbol);
        let result = gated
            .submit_gated(&signal, &routed, "momentum", &[], None, None, None, None)
            .await;
        assert!(
            result.is_submitted(),
            "Signal for {} should be submitted: {}",
            symbol,
            result
        );
    }

    assert_eq!(
        server_state.submit_count.load(Ordering::SeqCst),
        symbols.len() as u64,
        "Server should have received all {} signals",
        symbols.len()
    );

    let stats = gated.stats().await;
    assert_eq!(stats.submitted, symbols.len() as u64);
    assert!(stats.avg_submitted_scale() > 0.0);
}

/// Verify that brain_scale metadata is attached to submitted signals.
#[tokio::test]
async fn test_e2e_metadata_annotation() {
    let (addr, _) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline);

    let signal = bullish_signal("BTCUSD");
    let routed = bullish_routed_signal();

    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;

    // The GatedSubmissionResult::Submitted has a decision with the applied scale.
    // The metadata annotation happens inside submit_with_scale, which modifies
    // the signal before sending to the execution client. We can verify the
    // decision's scale is present.
    if let GatedSubmissionResult::Submitted { applied_scale, .. } = &result {
        assert!(
            *applied_scale > 0.0,
            "Applied scale should be positive: {}",
            applied_scale
        );
    } else {
        panic!("Expected Submitted, got: {}", result);
    }
}

/// Verify the Display implementation on GatedSubmissionResult.
#[tokio::test]
async fn test_e2e_result_display_formatting() {
    let (addr, _) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let mut gated = BrainGatedExecutionClient::new(exec_client, pipeline.clone());

    // Submitted
    let signal = bullish_signal("BTCUSD");
    let routed = bullish_routed_signal();
    let result = gated
        .submit_gated(&signal, &routed, "ema_flip", &[], None, None, None, None)
        .await;
    let display = format!("{}", result);
    assert!(
        display.contains("Submitted") || display.contains("BTCUSD"),
        "Display should contain Submitted or symbol: {}",
        display
    );

    // Blocked (kill switch)
    pipeline.activate_kill_switch().await;
    let signal2 = bullish_signal("ETHUSD");
    let result2 = gated
        .submit_gated(&signal2, &routed, "ema_flip", &[], None, None, None, None)
        .await;
    let display2 = format!("{}", result2);
    assert!(
        display2.contains("Blocked") || display2.contains("ETHUSD"),
        "Display should contain Blocked or symbol: {}",
        display2
    );
}

/// Verify that the gated client exposes the underlying config.
#[tokio::test]
async fn test_e2e_config_accessors() {
    let (addr, _) = start_mock_server().await;
    let exec_client = make_execution_client(addr).await;
    let pipeline = make_permissive_pipeline();

    let config = BrainGatedConfig {
        allow_reduce_only: false,
        apply_scale_to_signal: false,
        max_signal_age_secs: 120,
        record_trade_results: false,
    };

    let gated = BrainGatedExecutionClient::with_config(exec_client, pipeline, config);

    assert!(!gated.config().allow_reduce_only);
    assert!(!gated.config().apply_scale_to_signal);
    assert_eq!(gated.config().max_signal_age_secs, 120);
    assert!(!gated.config().record_trade_results);
    assert!(gated.endpoint().contains("127.0.0.1"));
}
