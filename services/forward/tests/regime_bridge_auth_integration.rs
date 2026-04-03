//! # End-to-End Authentication Integration Tests for Regime Bridge gRPC
//!
//! These tests stand up a real tonic gRPC server with the `AuthInterceptor`
//! enabled and make actual gRPC calls through the auth layer to verify:
//!
//! - Push with valid token → accepted
//! - Push with invalid token → UNAUTHENTICATED
//! - Push with missing token → UNAUTHENTICATED
//! - GetCurrentRegime with valid/invalid/missing token
//! - StreamRegimeUpdates with valid/invalid/missing token
//! - Multi-token: any valid token in the set is accepted
//! - Runtime token rotation: add/revoke mid-flight
//! - Batch push through auth layer
//! - Allow-all interceptor passes everything (backward compat)

use std::net::SocketAddr;
use std::time::Duration;

use tokio::sync::broadcast;
use tokio::time::timeout;
use tonic::transport::{Channel, Server};
use tonic::{Code, Request};

use janus_forward::regime_bridge::BridgedRegimeState;
use janus_forward::regime_bridge_auth::{AuthConfig, AuthInterceptor, inject_auth_metadata};
use janus_forward::regime_bridge_proto::regime_bridge_service_client::RegimeBridgeServiceClient;
use janus_forward::regime_bridge_proto::regime_bridge_service_server::RegimeBridgeServiceServer;
use janus_forward::regime_bridge_proto::{
    AmygdalaRegime, GetCurrentRegimeRequest, HypothalamusRegime, PushRegimeStateBatchRequest,
    PushRegimeStateRequest, RegimeIndicators, RegimeState, StreamRegimeUpdatesRequest,
};
use janus_forward::regime_bridge_server::RegimeBridgeServer;

// ============================================================================
// Test helpers
// ============================================================================

/// Start an authenticated gRPC server on an ephemeral port.
/// Returns (server address, interceptor handle for runtime rotation, join handle).
async fn start_test_server(
    interceptor: AuthInterceptor,
) -> (SocketAddr, AuthInterceptor, tokio::task::JoinHandle<()>) {
    let (tx, _rx) = broadcast::channel::<BridgedRegimeState>(64);
    let bridge_server = RegimeBridgeServer::new(tx);
    let _updater = bridge_server.spawn_snapshot_updater();

    // Bind to port 0 for an OS-assigned ephemeral port
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    let interceptor_clone = interceptor.clone();

    let svc = RegimeBridgeServiceServer::with_interceptor(bridge_server, interceptor);

    let handle = tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
            .await
            .unwrap();
    });

    // Give the server a moment to start
    tokio::time::sleep(Duration::from_millis(50)).await;

    (addr, interceptor_clone, handle)
}

/// Connect a gRPC client to the given address.
async fn connect_client(addr: SocketAddr) -> RegimeBridgeServiceClient<Channel> {
    let endpoint = format!("http://{}", addr);
    RegimeBridgeServiceClient::connect(endpoint)
        .await
        .expect("Failed to connect to test server")
}

/// Build a sample `RegimeState` for testing.
fn sample_regime_state(symbol: &str) -> RegimeState {
    RegimeState {
        symbol: symbol.to_string(),
        hypothalamus_regime: HypothalamusRegime::Bullish.into(),
        amygdala_regime: AmygdalaRegime::LowVolTrending.into(),
        position_scale: 1.1,
        is_high_risk: false,
        confidence: 0.85,
        indicators: Some(RegimeIndicators {
            trend: 0.7,
            trend_strength: 0.6,
            volatility: 500.0,
            volatility_percentile: 0.3,
            correlation: 0.5,
            breadth: 0.5,
            momentum: 0.4,
            relative_volume: 1.5,
            liquidity_score: 0.9,
            fear_index: 0.0,
        }),
        timestamp_us: 1_700_000_000_000_000,
        sequence: 0,
        is_transition: false,
        previous_hypothalamus_regime: HypothalamusRegime::Unspecified.into(),
        previous_amygdala_regime: AmygdalaRegime::Unspecified.into(),
    }
}

/// Build a `PushRegimeStateRequest` with an auth token injected into metadata.
fn push_request_with_token(symbol: &str, token: Option<&str>) -> Request<PushRegimeStateRequest> {
    let payload = PushRegimeStateRequest {
        state: Some(sample_regime_state(symbol)),
        source_id: "test-client".to_string(),
    };
    let mut request = Request::new(payload);
    if let Some(t) = token {
        inject_auth_metadata(&mut request, t).unwrap();
    }
    request
}

fn get_current_request_with_token(
    symbol: &str,
    token: Option<&str>,
) -> Request<GetCurrentRegimeRequest> {
    let payload = GetCurrentRegimeRequest {
        symbols: vec![symbol.to_string()],
    };
    let mut request = Request::new(payload);
    if let Some(t) = token {
        inject_auth_metadata(&mut request, t).unwrap();
    }
    request
}

fn stream_request_with_token(token: Option<&str>) -> Request<StreamRegimeUpdatesRequest> {
    let payload = StreamRegimeUpdatesRequest {
        symbols: vec![],
        transitions_only: false,
        min_confidence: 0.0,
        client_id: "test-stream-client".to_string(),
    };
    let mut request = Request::new(payload);
    if let Some(t) = token {
        inject_auth_metadata(&mut request, t).unwrap();
    }
    request
}

fn batch_request_with_token(
    symbols: &[&str],
    token: Option<&str>,
) -> Request<PushRegimeStateBatchRequest> {
    let states: Vec<RegimeState> = symbols.iter().map(|s| sample_regime_state(s)).collect();
    let payload = PushRegimeStateBatchRequest {
        states,
        source_id: "test-batch-client".to_string(),
    };
    let mut request = Request::new(payload);
    if let Some(t) = token {
        inject_auth_metadata(&mut request, t).unwrap();
    }
    request
}

// ============================================================================
// PushRegimeState auth tests
// ============================================================================

#[tokio::test]
async fn test_push_with_valid_token_accepted() {
    let interceptor = AuthInterceptor::with_token("test-secret-42");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("test-secret-42")))
        .await;

    assert!(resp.is_ok(), "Expected OK, got: {:?}", resp.err());
    let inner = resp.unwrap().into_inner();
    assert!(inner.accepted);
    assert_eq!(inner.message, "accepted");
}

#[tokio::test]
async fn test_push_with_invalid_token_rejected() {
    let interceptor = AuthInterceptor::with_token("correct-token");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("wrong-token")))
        .await;

    assert!(resp.is_err());
    let status = resp.unwrap_err();
    assert_eq!(status.code(), Code::Unauthenticated);
    assert!(
        status.message().contains("Invalid authentication token"),
        "Unexpected message: {}",
        status.message()
    );
}

#[tokio::test]
async fn test_push_with_missing_token_rejected() {
    let interceptor = AuthInterceptor::with_token("my-secret");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // No token injected
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", None))
        .await;

    assert!(resp.is_err());
    let status = resp.unwrap_err();
    assert_eq!(status.code(), Code::Unauthenticated);
    assert!(
        status.message().contains("Missing authorization header"),
        "Unexpected message: {}",
        status.message()
    );
}

// ============================================================================
// GetCurrentRegime auth tests
// ============================================================================

#[tokio::test]
async fn test_get_current_regime_with_valid_token() {
    let interceptor = AuthInterceptor::with_token("get-secret");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // First push a state so there's something to get
    let push_resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("get-secret")))
        .await;
    assert!(push_resp.is_ok());

    // Small delay for snapshot updater
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Now get current regime with valid token
    let resp = client
        .get_current_regime(get_current_request_with_token("BTCUSD", Some("get-secret")))
        .await;

    assert!(resp.is_ok(), "Expected OK, got: {:?}", resp.err());
}

#[tokio::test]
async fn test_get_current_regime_with_invalid_token() {
    let interceptor = AuthInterceptor::with_token("correct");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .get_current_regime(get_current_request_with_token("BTCUSD", Some("wrong")))
        .await;

    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);
}

#[tokio::test]
async fn test_get_current_regime_with_missing_token() {
    let interceptor = AuthInterceptor::with_token("secret");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .get_current_regime(get_current_request_with_token("BTCUSD", None))
        .await;

    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);
}

// ============================================================================
// StreamRegimeUpdates auth tests
// ============================================================================

#[tokio::test]
async fn test_stream_with_valid_token_opens() {
    let interceptor = AuthInterceptor::with_token("stream-secret");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // Open a stream with valid token — should succeed
    let resp = client
        .stream_regime_updates(stream_request_with_token(Some("stream-secret")))
        .await;

    assert!(
        resp.is_ok(),
        "Expected stream to open, got: {:?}",
        resp.err()
    );
}

#[tokio::test]
async fn test_stream_with_invalid_token_rejected() {
    let interceptor = AuthInterceptor::with_token("correct");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .stream_regime_updates(stream_request_with_token(Some("incorrect")))
        .await;

    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);
}

#[tokio::test]
async fn test_stream_with_missing_token_rejected() {
    let interceptor = AuthInterceptor::with_token("secret");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .stream_regime_updates(stream_request_with_token(None))
        .await;

    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);
}

// ============================================================================
// Multi-token tests
// ============================================================================

#[tokio::test]
async fn test_multi_token_any_valid_accepted() {
    let config = AuthConfig::with_tokens(vec![
        "token-alpha".into(),
        "token-beta".into(),
        "token-gamma".into(),
    ]);
    let interceptor = AuthInterceptor::new(config);
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // All three tokens should work
    for token in &["token-alpha", "token-beta", "token-gamma"] {
        let resp = client
            .push_regime_state(push_request_with_token("BTCUSD", Some(token)))
            .await;
        assert!(
            resp.is_ok(),
            "Token '{}' should be accepted, got: {:?}",
            token,
            resp.err()
        );
        assert!(resp.unwrap().into_inner().accepted);
    }
}

#[tokio::test]
async fn test_multi_token_invalid_rejected() {
    let config = AuthConfig::with_tokens(vec!["alpha".into(), "beta".into()]);
    let interceptor = AuthInterceptor::new(config);
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("gamma")))
        .await;

    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);
}

// ============================================================================
// Runtime token rotation tests
// ============================================================================

#[tokio::test]
async fn test_runtime_add_token_allows_new_clients() {
    let interceptor = AuthInterceptor::with_token("original-token");
    let (addr, shared_int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // New token should be rejected initially
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("new-token")))
        .await;
    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);

    // Add the new token at runtime
    shared_int.add_token("new-token").await;

    // Now the new token should work
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("new-token")))
        .await;
    assert!(resp.is_ok(), "New token should be accepted after add_token");
    assert!(resp.unwrap().into_inner().accepted);

    // Original token should still work
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("original-token")))
        .await;
    assert!(resp.is_ok());
}

#[tokio::test]
async fn test_runtime_revoke_token_rejects_old_clients() {
    let config = AuthConfig::with_tokens(vec!["keep-me".into(), "revoke-me".into()]);
    let interceptor = AuthInterceptor::new(config);
    let (addr, shared_int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // Both tokens work initially
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("revoke-me")))
        .await;
    assert!(resp.is_ok());

    // Revoke one
    shared_int.revoke_token("revoke-me").await;

    // Revoked token should now be rejected
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("revoke-me")))
        .await;
    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);

    // Non-revoked token still works
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("keep-me")))
        .await;
    assert!(resp.is_ok());
}

#[tokio::test]
async fn test_runtime_replace_tokens_atomic_swap() {
    let interceptor = AuthInterceptor::with_token("old-token");
    let (addr, shared_int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // Old token works
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("old-token")))
        .await;
    assert!(resp.is_ok());

    // Atomically replace all tokens
    shared_int
        .replace_tokens(vec!["fresh-1".into(), "fresh-2".into()])
        .await;

    // Old token immediately rejected
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("old-token")))
        .await;
    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);

    // New tokens work
    for token in &["fresh-1", "fresh-2"] {
        let resp = client
            .push_regime_state(push_request_with_token("BTCUSD", Some(token)))
            .await;
        assert!(resp.is_ok(), "Token '{}' should work after replace", token);
    }
}

// ============================================================================
// Batch push auth tests
// ============================================================================

#[tokio::test]
async fn test_batch_push_with_valid_token() {
    let interceptor = AuthInterceptor::with_token("batch-secret");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .push_regime_state_batch(batch_request_with_token(
            &["BTCUSD", "ETHUSD"],
            Some("batch-secret"),
        ))
        .await;

    assert!(resp.is_ok(), "Batch push should succeed with valid token");
    let inner = resp.unwrap().into_inner();
    assert_eq!(inner.accepted_count, 2);
    assert_eq!(inner.rejected_count, 0);
}

#[tokio::test]
async fn test_batch_push_with_invalid_token() {
    let interceptor = AuthInterceptor::with_token("correct");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .push_regime_state_batch(batch_request_with_token(
            &["BTCUSD", "ETHUSD"],
            Some("wrong"),
        ))
        .await;

    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);
}

#[tokio::test]
async fn test_batch_push_with_missing_token() {
    let interceptor = AuthInterceptor::with_token("secret");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .push_regime_state_batch(batch_request_with_token(&["BTCUSD"], None))
        .await;

    assert!(resp.is_err());
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);
}

// ============================================================================
// Allow-all (backward compatibility) tests
// ============================================================================

#[tokio::test]
async fn test_allow_all_no_token_passes() {
    let interceptor = AuthInterceptor::allow_all();
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // No token provided — should pass through
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", None))
        .await;

    assert!(resp.is_ok(), "Allow-all should pass without token");
    assert!(resp.unwrap().into_inner().accepted);
}

#[tokio::test]
async fn test_allow_all_with_token_also_passes() {
    let interceptor = AuthInterceptor::allow_all();
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // Token provided but unnecessary — should still pass
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("any-token")))
        .await;

    assert!(resp.is_ok());
}

#[tokio::test]
async fn test_allow_all_get_current_regime_passes() {
    let interceptor = AuthInterceptor::allow_all();
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .get_current_regime(get_current_request_with_token("BTCUSD", None))
        .await;

    // Should succeed (might return "not found" status but NOT unauthenticated)
    match resp {
        Ok(_) => {} // fine — no state pushed yet
        Err(status) => {
            assert_ne!(
                status.code(),
                Code::Unauthenticated,
                "Allow-all should never return Unauthenticated"
            );
        }
    }
}

#[tokio::test]
async fn test_allow_all_stream_opens() {
    let interceptor = AuthInterceptor::allow_all();
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let resp = client
        .stream_regime_updates(stream_request_with_token(None))
        .await;

    assert!(resp.is_ok(), "Allow-all stream should open without token");
}

// ============================================================================
// Interceptor counter verification through real gRPC
// ============================================================================

#[tokio::test]
async fn test_interceptor_counters_through_grpc() {
    let interceptor = AuthInterceptor::with_token("counter-test");
    let (addr, shared_int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    assert_eq!(shared_int.accepted_count(), 0);
    assert_eq!(shared_int.rejected_count(), 0);

    // 2 valid pushes
    for _ in 0..2 {
        let _ = client
            .push_regime_state(push_request_with_token("BTCUSD", Some("counter-test")))
            .await;
    }

    // 1 invalid push
    let _ = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("wrong")))
        .await;

    // 1 missing token push
    let _ = client
        .push_regime_state(push_request_with_token("BTCUSD", None))
        .await;

    assert_eq!(shared_int.accepted_count(), 2);
    assert_eq!(shared_int.rejected_count(), 2);
}

// ============================================================================
// Stream receives pushed data through auth
// ============================================================================

#[tokio::test]
async fn test_stream_receives_pushed_data_through_auth() {
    let interceptor = AuthInterceptor::with_token("e2e-token");
    let (addr, _int, _handle) = start_test_server(interceptor).await;

    let mut push_client = connect_client(addr).await;
    let mut stream_client = connect_client(addr).await;

    // Open stream
    let stream_resp = stream_client
        .stream_regime_updates(stream_request_with_token(Some("e2e-token")))
        .await
        .expect("Stream should open");

    let mut stream = stream_resp.into_inner();

    // Small delay to let stream establish
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Push a state
    let push_resp = push_client
        .push_regime_state(push_request_with_token("BTCUSD", Some("e2e-token")))
        .await
        .expect("Push should succeed");
    assert!(push_resp.into_inner().accepted);

    // The stream should receive the pushed state
    let result = timeout(Duration::from_secs(2), stream.message()).await;

    match result {
        Ok(Ok(Some(state))) => {
            assert_eq!(state.symbol, "BTCUSD");
            assert_eq!(
                state.hypothalamus_regime,
                i32::from(HypothalamusRegime::Bullish)
            );
        }
        Ok(Ok(None)) => panic!("Stream ended unexpectedly"),
        Ok(Err(e)) => panic!("Stream error: {:?}", e),
        Err(_) => panic!("Timed out waiting for stream message"),
    }
}

// ============================================================================
// Zero-downtime rotation end-to-end
// ============================================================================

#[tokio::test]
async fn test_zero_downtime_rotation_e2e() {
    // Full rotation workflow through real gRPC:
    //   1. Start with v1 token
    //   2. Add v2 token (overlap window — both valid)
    //   3. Revoke v1 token (only v2 valid)

    let interceptor = AuthInterceptor::with_token("v1-secret");
    let (addr, shared_int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // Step 1: v1 works, v2 doesn't
    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("v1-secret")))
        .await;
    assert!(resp.is_ok());

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("v2-secret")))
        .await;
    assert!(resp.is_err());

    // Step 2: Add v2 — both work
    shared_int.add_token("v2-secret").await;

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("v1-secret")))
        .await;
    assert!(resp.is_ok(), "v1 should still work during overlap");

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("v2-secret")))
        .await;
    assert!(resp.is_ok(), "v2 should work after adding");

    // Step 3: Revoke v1 — only v2 works
    shared_int.revoke_token("v1-secret").await;

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("v1-secret")))
        .await;
    assert!(resp.is_err(), "v1 should be rejected after revocation");
    assert_eq!(resp.unwrap_err().code(), Code::Unauthenticated);

    let resp = client
        .push_regime_state(push_request_with_token("BTCUSD", Some("v2-secret")))
        .await;
    assert!(resp.is_ok(), "v2 should still work after v1 revocation");

    // Verify rotation count
    assert_eq!(shared_int.rotation_count(), 2); // 1 add + 1 revoke
}

// ============================================================================
// Case sensitivity: bearer prefix
// ============================================================================

#[tokio::test]
async fn test_lowercase_bearer_prefix_accepted() {
    let interceptor = AuthInterceptor::with_token("case-test");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    // Manually inject lowercase "bearer" prefix
    let payload = PushRegimeStateRequest {
        state: Some(sample_regime_state("BTCUSD")),
        source_id: "test".to_string(),
    };
    let mut request = Request::new(payload);
    request
        .metadata_mut()
        .insert("authorization", "bearer case-test".parse().unwrap());

    let resp = client.push_regime_state(request).await;
    assert!(resp.is_ok(), "Lowercase 'bearer' should be accepted");
}

#[tokio::test]
async fn test_wrong_auth_scheme_rejected() {
    let interceptor = AuthInterceptor::with_token("scheme-test");
    let (addr, _int, _handle) = start_test_server(interceptor).await;
    let mut client = connect_client(addr).await;

    let payload = PushRegimeStateRequest {
        state: Some(sample_regime_state("BTCUSD")),
        source_id: "test".to_string(),
    };
    let mut request = Request::new(payload);
    request
        .metadata_mut()
        .insert("authorization", "Token scheme-test".parse().unwrap());

    let resp = client.push_regime_state(request).await;
    assert!(resp.is_err());
    let status = resp.unwrap_err();
    assert_eq!(status.code(), Code::Unauthenticated);
    assert!(
        status.message().contains("Bearer scheme"),
        "Expected Bearer scheme error, got: {}",
        status.message()
    );
}
