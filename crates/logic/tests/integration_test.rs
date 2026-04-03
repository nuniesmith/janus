//! Integration tests for janus-logic crate.
//!
//! Tests the full flow of signal validation → risk validation → order generation

use common::{Order, OrderSide, OrderType, Price, Signal, Volume};
use logic::{
    AssetClass, ComprehensiveRiskEngine, PositionSizer, PositionTracker, PropFirmType, RiskContext,
    SignalValidator, TradingMode, TypedOrder, Unchecked, UserProfile,
};

/// Create a test risk engine with default configuration
fn create_test_risk_engine() -> ComprehensiveRiskEngine {
    use logic::{
        GlobalSafety, PerTradeRisk, PropFirmConstraints, PropFirmRules, PropFirmType, RiskRules,
        RiskValidator,
    };
    let mut validator = RiskValidator::new();

    // Add test prop firm rules
    let test_rules = PropFirmRules {
        account_size: 100000.0,
        constraints: PropFirmConstraints {
            max_daily_drawdown_limit: 5000.0,
            max_total_loss_limit: 10000.0,
            account_size: 100000.0,
        },
    };
    validator.add_prop_firm_rules(PropFirmType::GenericChallenge, test_rules);

    // Add test risk rules
    let risk_rules = RiskRules {
        global_safety: GlobalSafety {
            max_leverage: 10.0,
            max_open_positions: 5,
            kill_switch_enabled: false,
        },
        per_trade_risk: PerTradeRisk {
            max_risk_per_trade_percent: 100.0,
        },
    };
    validator.set_risk_rules(risk_rules);

    ComprehensiveRiskEngine::new(
        validator, 100.0,   // max_position_size
        -1000.0, // max_daily_loss
        3600,    // wash_sale_window
    )
}

/// Create a test signal
fn create_test_signal() -> Signal {
    Signal {
        symbol: "BTC/USD".to_string(),
        side: OrderSide::Buy,
        strength: 0.8,
        confidence: 0.9,
        predicted_duration_seconds: Some(3600), // 1 hour
    }
}

/// Create a test user profile
fn create_test_profile() -> UserProfile {
    UserProfile {
        user_id: "test_user".to_string(),
        risk_tolerance: 0.7,
        asset_class: AssetClass::Crypto,
        trading_mode: TradingMode::Swing,
        allowed_symbols: vec!["BTC/USD".to_string(), "ETH/USD".to_string()],
    }
}

#[test]
fn test_signal_validation_flow() {
    let signal = create_test_signal();
    let profile = create_test_profile();

    // Signal should pass validation
    assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());
}

#[test]
fn test_signal_rejected_for_scalp_mode() {
    let mut signal = create_test_signal();
    signal.predicted_duration_seconds = Some(960); // 16 minutes

    let mut profile = create_test_profile();
    profile.trading_mode = TradingMode::Scalp;

    // Signal should be rejected (duration > 15 min for SCALP)
    assert!(SignalValidator::validate_signal(&signal, &profile).is_err());
}

#[test]
fn test_position_sizing() {
    let sizer = PositionSizer::new(100000.0, 0.015); // $100k, 1.5% risk

    let entry_price = Price(50000.0);
    let stop_loss = Price(49000.0); // $1000 stop loss

    let (units, _usd, risk) = sizer
        .calculate_position_size(entry_price, stop_loss, None)
        .unwrap();

    // Risk should be 1.5% of $100k = $1500
    assert!((risk.value() - 1500.0).abs() < 1.0);

    // Position size should be $1500 / $1000 = 1.5 units
    assert!((units.value() - 1.5).abs() < 0.01);
}

#[test]
fn test_position_tracking() {
    let mut tracker = PositionTracker::new();

    // Add a trade
    tracker
        .update_from_trade("BTC/USD", OrderSide::Buy, Volume(1.0), Price(50000.0))
        .unwrap();

    // Update price
    tracker.update_price("BTC/USD", Price(51000.0));

    // Check PnL
    let position = tracker.get_position("BTC/USD").unwrap();
    let pnl = position.unrealized_pnl().unwrap();
    assert_eq!(pnl.value(), 1000.0); // $1000 profit
}

#[test]
fn test_full_order_validation_flow() {
    let risk_engine = create_test_risk_engine();
    let position_tracker = PositionTracker::new();
    let position_sizer = PositionSizer::new(100000.0, 0.015);

    // 1. Create signal
    let signal = create_test_signal();
    let profile = create_test_profile();

    // 2. Validate signal
    assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());

    // 3. Calculate position size
    let entry_price = Price(50000.0);
    let stop_loss = Price(49000.0);
    let (units, _usd, _risk) = position_sizer
        .calculate_position_size(entry_price, stop_loss, None)
        .unwrap();

    // 4. Create order
    let order = Order::new(
        signal.symbol.clone(),
        signal.side,
        OrderType::Market,
        units,
        Some(entry_price),
    );

    // 5. Build risk context
    let risk_context = RiskContext {
        current_daily_pnl: 0.0,
        total_pnl: 0.0,
        account_size: position_sizer.portfolio_value(),
        prop_firm_type: PropFirmType::GenericChallenge,
        current_leverage: 0.0,
        current_open_positions: position_tracker.open_positions_count(),
        current_position: 0.0,
    };

    // 6. Validate order with context
    let recent_trades = vec![];
    assert!(
        risk_engine
            .validate_order_with_context(&order, &risk_context, &recent_trades)
            .is_ok()
    );

    // 7. Final verification with typed order
    let unchecked = TypedOrder::<Unchecked>::new(order);
    let verified = unchecked.verify(&risk_engine).unwrap();

    // Order is verified and ready
    assert!(verified.order().symbol == "BTC/USD");
}

#[test]
fn test_order_rejected_by_risk_limits() {
    let risk_engine = create_test_risk_engine();

    // Create an order that would exceed position limits
    let large_order = Order::new(
        "BTC/USD".to_string(),
        OrderSide::Buy,
        OrderType::Market,
        Volume(100.0), // Very large position
        Some(Price(50000.0)),
    );

    let risk_context = RiskContext {
        current_daily_pnl: 0.0,
        total_pnl: 0.0,
        account_size: 100000.0,
        prop_firm_type: PropFirmType::GenericChallenge,
        current_leverage: 0.0,
        current_open_positions: 0,
        current_position: 0.0,
    };

    // This should fail because position size exceeds max_position_size (100.0)
    // Note: The actual validation depends on the risk_engine's max_position_size
    // Since we set it to 100.0 in create_test_risk_engine, a position of 100.0
    // at $50k = $5M, which is way over the limit
    let recent_trades = vec![];
    let result =
        risk_engine.validate_order_with_context(&large_order, &risk_context, &recent_trades);

    // Should reject due to position size limits
    assert!(result.is_err());
}

#[test]
fn test_position_tracker_with_multiple_trades() {
    let mut tracker = PositionTracker::new();

    // Buy 1 BTC at $50k
    tracker
        .update_from_trade("BTC/USD", OrderSide::Buy, Volume(1.0), Price(50000.0))
        .unwrap();

    // Buy another 0.5 BTC at $51k (should average entry price)
    tracker
        .update_from_trade("BTC/USD", OrderSide::Buy, Volume(0.5), Price(51000.0))
        .unwrap();

    let position = tracker.get_position("BTC/USD").unwrap();

    // Total quantity should be 1.5
    assert!((position.quantity.value() - 1.5).abs() < 0.01);

    // Average entry should be weighted average
    // (1.0 * 50000 + 0.5 * 51000) / 1.5 = 50333.33
    let expected_avg = (1.0 * 50000.0 + 0.5 * 51000.0) / 1.5;
    assert!((position.average_entry_price.value() - expected_avg).abs() < 1.0);
}

#[test]
fn test_signal_validation_with_bitcoin_asset_class() {
    let mut signal = create_test_signal();
    signal.symbol = "ETH/USD".to_string(); // Wrong symbol for Bitcoin-only profile

    let mut profile = create_test_profile();
    profile.asset_class = AssetClass::Bitcoin;

    // Should reject ETH symbol when profile is Bitcoin-only
    assert!(SignalValidator::validate_signal(&signal, &profile).is_err());

    // BTC should work
    signal.symbol = "BTC/USD".to_string();
    assert!(SignalValidator::validate_signal(&signal, &profile).is_ok());
}

#[test]
fn test_risk_context_building() {
    let mut tracker = PositionTracker::new();
    let sizer = PositionSizer::new(100000.0, 0.015);

    // Add some positions
    tracker
        .update_from_trade("BTC/USD", OrderSide::Buy, Volume(1.0), Price(50000.0))
        .unwrap();
    tracker
        .update_from_trade("ETH/USD", OrderSide::Buy, Volume(10.0), Price(3000.0))
        .unwrap();

    // Update prices
    tracker.update_price("BTC/USD", Price(51000.0));
    tracker.update_price("ETH/USD", Price(3100.0));

    // Build risk context
    let context = RiskContext {
        current_daily_pnl: 0.0,
        total_pnl: 0.0,
        account_size: sizer.portfolio_value(),
        prop_firm_type: PropFirmType::GenericChallenge,
        current_leverage: calculate_leverage(&tracker, &sizer),
        current_open_positions: tracker.open_positions_count(),
        current_position: tracker
            .get_position("BTC/USD")
            .map(|p| p.quantity.value())
            .unwrap_or(0.0),
    };

    assert_eq!(context.current_open_positions, 2);
    assert_eq!(context.current_position, 1.0);
}

/// Helper function to calculate leverage
fn calculate_leverage(tracker: &PositionTracker, sizer: &PositionSizer) -> f64 {
    let total_notional: f64 = tracker
        .all_positions()
        .values()
        .map(|p| p.notional_value().value())
        .sum();

    let portfolio_value = sizer.portfolio_value();

    if portfolio_value > 0.0 {
        total_notional / portfolio_value
    } else {
        0.0
    }
}
