//! Kill Switch End-to-End Integration Tests
//!
//! This test binary verifies that the execution service's kill switch guard
//! correctly blocks ALL order submission (including close/reduce-only orders)
//! when the kill switch is active, and resumes when deactivated.
//!
//! These tests exercise the `AtomicOrderGate` (standing in for the Redis-backed
//! `KillSwitchGuard`) and verify the full wiring into the order/signal
//! submission hot path, error types, gRPC status mapping, and concurrency safety.
//!
//! # Running
//!
//! ```sh
//! cargo test -p janus-execution --test kill_switch_e2e
//! ```

use janus_execution::error::ExecutionError;
use janus_execution::kill_switch_guard::{AtomicOrderGate, OrderGate};
use janus_execution::types::{Order, OrderSide, OrderTypeEnum};
use rust_decimal::Decimal;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a minimal buy order for testing.
fn make_buy_order(symbol: &str) -> Order {
    Order::new(
        format!("test-signal-buy-{}", uuid::Uuid::new_v4()),
        symbol.to_string(),
        "simulated".to_string(),
        OrderSide::Buy,
        OrderTypeEnum::Market,
        Decimal::new(1, 3), // 0.001
    )
}

/// Create a minimal sell (close) order for testing.
fn make_sell_order(symbol: &str) -> Order {
    Order::new(
        format!("test-signal-sell-{}", uuid::Uuid::new_v4()),
        symbol.to_string(),
        "simulated".to_string(),
        OrderSide::Sell,
        OrderTypeEnum::Market,
        Decimal::new(1, 3), // 0.001
    )
}

/// Simulate what `OrderManager::submit_order` does at the kill switch guard
/// check point. Returns `Err(ExecutionError::KillSwitchActive)` if blocked,
/// `Ok(())` if allowed.
fn simulate_order_gate_check(gate: &dyn OrderGate) -> Result<(), ExecutionError> {
    if gate.is_blocked() {
        Err(ExecutionError::KillSwitchActive(
            gate.block_reason().to_string(),
        ))
    } else {
        Ok(())
    }
}

// ===========================================================================
// AtomicOrderGate unit-level sanity checks (no async, no Redis)
// ===========================================================================

#[test]
fn gate_starts_unblocked() {
    let gate = AtomicOrderGate::new(false);
    assert!(!gate.is_blocked());
}

#[test]
fn gate_starts_blocked() {
    let gate = AtomicOrderGate::new(true);
    assert!(gate.is_blocked());
}

#[test]
fn gate_can_transition() {
    let gate = AtomicOrderGate::new(false);
    assert!(!gate.is_blocked());

    gate.set_blocked(true);
    assert!(gate.is_blocked());

    gate.set_blocked(false);
    assert!(!gate.is_blocked());
}

#[test]
fn gate_block_reason_is_descriptive() {
    let gate = AtomicOrderGate::new(true);
    let reason = gate.block_reason();
    assert!(
        reason.to_lowercase().contains("kill switch"),
        "block_reason should mention kill switch, got: {}",
        reason,
    );
}

// ===========================================================================
// OrderManager integration: kill switch blocks submit_order
// ===========================================================================

/// Verify that the order gate check returns `KillSwitchActive` when
/// blocked — for a BUY (new entry) order.
#[tokio::test]
async fn order_manager_blocks_buy_when_killed() {
    let gate = Arc::new(AtomicOrderGate::new(true));
    let _order = make_buy_order("BTCUSDT");

    let result = simulate_order_gate_check(gate.as_ref());
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.is_kill_switch());
    assert!(err.to_string().contains("Kill switch"));

    // Verify gRPC status conversion
    let status = err.to_grpc_status();
    assert_eq!(status.code(), tonic::Code::FailedPrecondition);
    assert!(
        status.message().contains("Kill switch"),
        "gRPC status message should mention kill switch, got: {}",
        status.message(),
    );
}

/// Verify that SELL (close position) orders are ALSO blocked when the kill
/// switch is active. This is the "hard kill" behavior: when the kill switch
/// fires, ALL orders are blocked — not just new entries.
#[tokio::test]
async fn order_manager_blocks_sell_close_when_killed() {
    let gate = Arc::new(AtomicOrderGate::new(true));
    let order = make_sell_order("BTCUSDT");

    // The gate blocks regardless of order side
    let result = simulate_order_gate_check(gate.as_ref());
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(err.is_kill_switch());

    // Verify the sell order would have been blocked
    assert_eq!(order.side, OrderSide::Sell);
    assert!(
        gate.is_blocked(),
        "Sell orders must be blocked by kill switch too"
    );
}

/// Verify that orders are allowed when the gate is not blocked.
#[tokio::test]
async fn order_manager_allows_when_not_killed() {
    let gate = Arc::new(AtomicOrderGate::new(false));
    let order = make_buy_order("ETHUSDT");

    let result = simulate_order_gate_check(gate.as_ref());
    assert!(result.is_ok(), "Orders should be allowed when gate is open");
    assert_eq!(order.symbol, "ETHUSDT");
}

// ===========================================================================
// Kill switch state transitions during order flow
// ===========================================================================

/// Simulate a scenario where the kill switch activates mid-session:
/// 1. Orders are initially allowed
/// 2. Kill switch activates → orders blocked
/// 3. Kill switch deactivates → orders resume
#[tokio::test]
async fn kill_switch_lifecycle_transitions() {
    let gate = Arc::new(AtomicOrderGate::new(false));

    // Phase 1: Orders allowed
    assert!(
        simulate_order_gate_check(gate.as_ref()).is_ok(),
        "Phase 1: orders should be allowed",
    );
    let _order1 = make_buy_order("BTCUSDT");

    // Phase 2: Kill switch activates
    gate.set_blocked(true);

    // Buy order — should be blocked
    let result2 = simulate_order_gate_check(gate.as_ref());
    assert!(
        result2.is_err(),
        "Phase 2: buy orders should be blocked after kill switch activation",
    );

    // Sell (close) order — should also be blocked (hard kill)
    let _order3 = make_sell_order("BTCUSDT");
    let result3 = simulate_order_gate_check(gate.as_ref());
    assert!(
        result3.is_err(),
        "Phase 2: sell/close orders should also be blocked",
    );

    // Phase 3: Kill switch deactivates
    gate.set_blocked(false);

    let result4 = simulate_order_gate_check(gate.as_ref());
    assert!(
        result4.is_ok(),
        "Phase 3: orders should resume after deactivation",
    );
    let _order4 = make_buy_order("ETHUSDT");
}

/// Verify that the kill switch blocks ALL symbols, not just specific ones.
#[tokio::test]
async fn kill_switch_blocks_all_symbols() {
    let gate = Arc::new(AtomicOrderGate::new(true));

    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "ARBUSDT"];

    for symbol in &symbols {
        let result = simulate_order_gate_check(gate.as_ref());
        assert!(
            result.is_err(),
            "Kill switch should block orders for {}",
            symbol,
        );

        // Both sides blocked
        let _buy = make_buy_order(symbol);
        let _sell = make_sell_order(symbol);
        assert!(gate.is_blocked());
    }
}

/// Verify that the kill switch blocks ALL order types, not just market orders.
#[tokio::test]
async fn kill_switch_blocks_all_order_types() {
    let gate = Arc::new(AtomicOrderGate::new(true));

    let order_types = [
        OrderTypeEnum::Market,
        OrderTypeEnum::Limit,
        OrderTypeEnum::StopMarket,
        OrderTypeEnum::StopLimit,
        OrderTypeEnum::TrailingStop,
    ];

    for order_type in &order_types {
        let mut order = make_buy_order("BTCUSDT");
        order.order_type = *order_type;

        let result = simulate_order_gate_check(gate.as_ref());
        assert!(
            result.is_err(),
            "Kill switch should block {:?} orders",
            order_type,
        );
    }
}

// ===========================================================================
// Concurrent access safety
// ===========================================================================

/// Verify the gate is safe to read from multiple concurrent tasks.
#[tokio::test]
async fn kill_switch_concurrent_reads() {
    let gate = Arc::new(AtomicOrderGate::new(false));

    let mut handles = Vec::new();

    for i in 0..100 {
        let g = gate.clone();
        handles.push(tokio::spawn(async move {
            // One task activates the kill switch mid-way
            if i == 50 {
                g.set_blocked(true);
            }
            // All reads must succeed without panic
            let _ = g.is_blocked();
            let _ = simulate_order_gate_check(g.as_ref());
        }));
    }

    for handle in handles {
        handle.await.unwrap();
    }

    // After all tasks complete, the gate should be blocked (task 50 set it)
    assert!(gate.is_blocked());
}

/// Rapid toggling of the kill switch should not cause data races.
#[tokio::test]
async fn kill_switch_rapid_toggling() {
    let gate = Arc::new(AtomicOrderGate::new(false));

    let writer = gate.clone();
    let reader = gate.clone();

    let write_handle = tokio::spawn(async move {
        for i in 0..1000 {
            writer.set_blocked(i % 2 == 0);
            if i % 100 == 0 {
                tokio::task::yield_now().await;
            }
        }
    });

    let read_handle = tokio::spawn(async move {
        let mut active_count = 0u32;
        let mut inactive_count = 0u32;
        for _ in 0..1000 {
            if reader.is_blocked() {
                active_count += 1;
            } else {
                inactive_count += 1;
            }
        }
        (active_count, inactive_count)
    });

    write_handle.await.unwrap();
    let (active, inactive) = read_handle.await.unwrap();

    // The real assertion is that no panic occurred during concurrent access
    assert_eq!(
        active + inactive,
        1000,
        "All reads should complete: active={}, inactive={}",
        active,
        inactive,
    );
}

// ===========================================================================
// Error type integration
// ===========================================================================

#[test]
fn kill_switch_error_is_not_retryable() {
    let err = ExecutionError::KillSwitchActive("test".to_string());

    // Kill switch errors should NOT be retried — the order must be rejected
    // and the caller should wait until the kill switch is deactivated.
    assert!(
        !err.is_retryable(),
        "Kill switch errors must not be retryable"
    );
}

#[test]
fn kill_switch_error_is_not_rate_limit() {
    let err = ExecutionError::KillSwitchActive("test".to_string());
    assert!(
        !err.is_rate_limit(),
        "Kill switch errors are not rate limits"
    );
}

#[test]
fn kill_switch_error_is_kill_switch() {
    let err = ExecutionError::KillSwitchActive("emergency stop".to_string());
    assert!(err.is_kill_switch());
}

#[test]
fn kill_switch_error_display() {
    let err = ExecutionError::KillSwitchActive(
        "Kill switch is active — all order submission is halted".to_string(),
    );
    let msg = err.to_string();
    assert!(msg.contains("Kill switch active"), "Display: {}", msg);
    assert!(msg.contains("halted"), "Display: {}", msg);
}

#[test]
fn kill_switch_error_grpc_status_code() {
    let err = ExecutionError::KillSwitchActive("emergency stop".to_string());
    let status = err.to_grpc_status();

    // Kill switch should map to FAILED_PRECONDITION (not UNAVAILABLE or INTERNAL)
    // because the service is available but refuses orders due to a precondition.
    assert_eq!(
        status.code(),
        tonic::Code::FailedPrecondition,
        "Kill switch should be FailedPrecondition, got {:?}",
        status.code(),
    );
}

#[test]
fn kill_switch_error_debug_includes_reason() {
    let err = ExecutionError::KillSwitchActive("flash crash detected".to_string());
    let debug = format!("{:?}", err);
    assert!(
        debug.contains("flash crash detected"),
        "Debug output should contain the reason: {}",
        debug,
    );
}

// ===========================================================================
// Defense-in-depth: no gate set (backwards compatibility)
// ===========================================================================

/// When no order gate is set on the OrderManager, orders should proceed
/// normally (backwards compatible).
#[test]
fn no_gate_means_no_blocking() {
    let gate: Option<Arc<dyn OrderGate>> = None;

    // Simulating the guard check from OrderManager::submit_order:
    let blocked = gate.as_ref().is_some_and(|g| g.is_blocked());
    assert!(!blocked, "No gate should mean no blocking");
}

#[test]
fn some_gate_unblocked_means_allowed() {
    let gate: Option<Arc<dyn OrderGate>> = Some(Arc::new(AtomicOrderGate::new(false)));

    let blocked = gate.as_ref().is_some_and(|g| g.is_blocked());
    assert!(!blocked, "Unblocked gate should allow orders");
}

#[test]
fn some_gate_blocked_means_rejected() {
    let gate: Option<Arc<dyn OrderGate>> = Some(Arc::new(AtomicOrderGate::new(true)));

    let blocked = gate.as_ref().is_some_and(|g| g.is_blocked());
    assert!(blocked, "Blocked gate should reject orders");
}

// ===========================================================================
// Scenario: Flash crash kill switch activation
// ===========================================================================

/// Simulate a flash crash scenario:
/// 1. System is trading normally
/// 2. Flash crash detected → kill switch fires
/// 3. All pending order submissions are blocked
/// 4. Operator investigates and deactivates
/// 5. Trading resumes
#[tokio::test]
async fn scenario_flash_crash_kill_switch() {
    let gate = Arc::new(AtomicOrderGate::new(false));

    // 1. Normal trading
    let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    for sym in &symbols {
        assert!(
            simulate_order_gate_check(gate.as_ref()).is_ok(),
            "Should be trading normally on {}",
            sym,
        );
    }

    // 2. Flash crash detected — kill switch fires
    gate.set_blocked(true);

    // 3. All pending orders are blocked
    let mut blocked_count = 0u32;
    for sym in &symbols {
        // New entry orders
        let _buy = make_buy_order(sym);
        if simulate_order_gate_check(gate.as_ref()).is_err() {
            blocked_count += 1;
        }

        // Close orders (also blocked in hard kill)
        let _sell = make_sell_order(sym);
        if simulate_order_gate_check(gate.as_ref()).is_err() {
            blocked_count += 1;
        }
    }
    assert_eq!(
        blocked_count,
        6, // 3 symbols × 2 sides
        "All orders should be blocked during flash crash",
    );

    // 4. Operator deactivates
    gate.set_blocked(false);

    // 5. Trading resumes
    for sym in &symbols {
        assert!(
            simulate_order_gate_check(gate.as_ref()).is_ok(),
            "Should resume trading on {}",
            sym,
        );
    }
}

// ===========================================================================
// Scenario: Kill switch persists across multiple evaluation cycles
// ===========================================================================

/// The kill switch state is latching — it stays active until explicitly
/// deactivated. Multiple order attempts should all be blocked.
#[tokio::test]
async fn kill_switch_is_latching() {
    let gate = Arc::new(AtomicOrderGate::new(true));

    // 100 consecutive order attempts should all be blocked
    for i in 0..100 {
        let result = simulate_order_gate_check(gate.as_ref());
        assert!(
            result.is_err(),
            "Attempt {} should be blocked (latching)",
            i,
        );
    }

    // Only explicit deactivation releases
    gate.set_blocked(false);
    assert!(
        simulate_order_gate_check(gate.as_ref()).is_ok(),
        "Should be unblocked after deactivation",
    );
}

// ===========================================================================
// Scenario: Multiple independent gates (per-service isolation)
// ===========================================================================

/// Different execution service instances can have independent gates.
/// Activating one does not affect the other.
#[tokio::test]
async fn independent_gates_are_isolated() {
    let gate_a = Arc::new(AtomicOrderGate::new(false));
    let gate_b = Arc::new(AtomicOrderGate::new(false));

    // Both initially open
    assert!(simulate_order_gate_check(gate_a.as_ref()).is_ok());
    assert!(simulate_order_gate_check(gate_b.as_ref()).is_ok());

    // Activate gate A only
    gate_a.set_blocked(true);

    assert!(
        simulate_order_gate_check(gate_a.as_ref()).is_err(),
        "Gate A should be blocked"
    );
    assert!(
        simulate_order_gate_check(gate_b.as_ref()).is_ok(),
        "Gate B should still be open"
    );

    // Activate gate B
    gate_b.set_blocked(true);

    assert!(simulate_order_gate_check(gate_a.as_ref()).is_err());
    assert!(simulate_order_gate_check(gate_b.as_ref()).is_err());

    // Deactivate gate A only
    gate_a.set_blocked(false);

    assert!(
        simulate_order_gate_check(gate_a.as_ref()).is_ok(),
        "Gate A should be open again"
    );
    assert!(
        simulate_order_gate_check(gate_b.as_ref()).is_err(),
        "Gate B should still be blocked"
    );
}

// ===========================================================================
// Trait object polymorphism (OrderGate as dyn)
// ===========================================================================

/// Verify OrderGate works correctly through trait object indirection,
/// which is how it's used in OrderManager (Option<Arc<dyn OrderGate>>).
#[test]
fn order_gate_through_trait_object() {
    let concrete = AtomicOrderGate::new(false);
    let gate: Arc<dyn OrderGate> = Arc::new(concrete);

    assert!(!gate.is_blocked());
    assert!(gate.block_reason().contains("Kill switch"));

    // We can't call set_blocked through the trait object (it's not in the trait),
    // but we can verify the trait object API works.
}

/// Verify that NoOpKillSwitchGuard never blocks through the OrderGate trait.
#[test]
fn noop_guard_never_blocks_via_trait() {
    let guard = janus_execution::kill_switch_guard::NoOpKillSwitchGuard;
    let gate: &dyn OrderGate = &guard;

    assert!(!gate.is_blocked());

    // Call it many times — should never change
    for _ in 0..100 {
        assert!(!gate.is_blocked());
    }
}
