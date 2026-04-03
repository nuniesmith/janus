//! Strategy inference orchestrator actor.
//!
//! Coordinates vision, logic, and execution components.
//!
//! This actor integrates:
//! - Signal validation (user profiles, trading modes)
//! - Risk validation (prop firm rules, risk rules, position limits)
//! - Position tracking and sizing
//! - Order verification using the typestate pattern

use anyhow::Result;
use common::{Order, OrderType, Price, Signal, Trade};
use logic::{
    ComprehensiveRiskEngine, PositionSizer, PositionTracker, PropFirmType, RiskContext,
    SignalValidator, TradingMode, TypedOrder, Unchecked, UserProfile,
};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{error, info, warn};

/// Strategy actor - orchestrates the inference pipeline
#[allow(dead_code)]
pub struct StrategyActor {
    risk_engine: Arc<ComprehensiveRiskEngine>,
    position_tracker: Arc<Mutex<PositionTracker>>,
    position_sizer: Arc<Mutex<PositionSizer>>,
    user_profile: UserProfile,
    realized_pnl: f64,
}

impl StrategyActor {
    /// Create a new strategy actor with risk engine
    pub fn new(
        risk_engine: ComprehensiveRiskEngine,
        portfolio_value: f64,
        user_id: String,
        trading_mode: TradingMode,
    ) -> Self {
        Self {
            risk_engine: Arc::new(risk_engine),
            position_tracker: Arc::new(Mutex::new(PositionTracker::new())),
            position_sizer: Arc::new(Mutex::new(PositionSizer::new(portfolio_value, 0.015))),
            user_profile: UserProfile::new(user_id, trading_mode),
            realized_pnl: 0.0,
        }
    }

    /// Process a signal and generate an order if valid
    pub async fn process_signal(&self, signal: Signal) -> Result<Option<Order>> {
        info!("Processing signal: {} {:?}", signal.symbol, signal.side);

        // 1. Validate signal against user profile
        match SignalValidator::validate_signal(&signal, &self.user_profile) {
            Ok(()) => {
                info!("Signal validated for user {}", self.user_profile.user_id);
            }
            Err(e) => {
                warn!("Signal rejected by validation: {}", e);
                return Ok(None);
            }
        }

        // 2. Get current position state
        let (current_position, total_unrealized_pnl, open_positions) = {
            let tracker = self.position_tracker.lock().await;
            let current_pos = tracker
                .get_position(&signal.symbol)
                .map(|p| p.quantity.value())
                .unwrap_or(0.0);
            let unrealized = tracker.total_unrealized_pnl().value();
            let open = tracker.open_positions_count();
            (current_pos, unrealized, open)
        };

        // 3. Calculate position size using the last known market price from the
        //    position tracker and a default 2% stop distance. The brain-gated
        //    execution path (BrainGatedExecutionClient) is the preferred
        //    production path; this actor serves as a fallback.
        let last_price = {
            let tracker = self.position_tracker.lock().await;
            tracker
                .get_position(&signal.symbol)
                .map(|p| p.current_price.value())
                .unwrap_or(0.0)
        };
        let market_price = if last_price > 0.0 {
            last_price
        } else {
            // No tracked price available — use a conservative fallback.
            // In practice the EventLoop path (not this actor) handles
            // live trading with real-time prices from the WebSocket feed.
            50000.0
        };
        let stop_distance = market_price * 0.02; // 2% default stop distance
        let entry_price = Price(market_price);
        let stop_loss = Price(market_price - stop_distance);

        let (units, usd, risk) = {
            let sizer = self.position_sizer.lock().await;
            sizer.calculate_position_size(entry_price, stop_loss, None)?
        };

        info!(
            "Calculated position: {} units (${:.2}), risk: ${:.2}",
            units.value(),
            usd.value(),
            risk.value()
        );

        // 4. Create order
        let order = Order::new(
            signal.symbol.clone(),
            signal.side,
            OrderType::Market,
            units,
            None, // Market order
        );

        // 5. Build risk context
        let portfolio_value = {
            let sizer = self.position_sizer.lock().await;
            sizer.portfolio_value()
        };

        let risk_context = RiskContext {
            current_daily_pnl: self.realized_pnl + total_unrealized_pnl,
            total_pnl: self.realized_pnl + total_unrealized_pnl,
            account_size: portfolio_value,
            prop_firm_type: PropFirmType::GenericChallenge,
            current_leverage: self.calculate_leverage().await,
            current_open_positions: open_positions,
            current_position,
        };

        // 6. Validate order with full context
        // Trade history is tracked by the position tracker; pass current
        // positions as a proxy for recent trade activity.
        let recent_trades = vec![];
        match self
            .risk_engine
            .validate_order_with_context(&order, &risk_context, &recent_trades)
        {
            Ok(compliance_score) => {
                info!(
                    "Order validated with compliance score: {:.2}",
                    compliance_score
                );
            }
            Err(e) => {
                warn!("Order rejected by risk validation: {}", e);
                return Ok(None);
            }
        }

        // 7. Final verification with typed order pattern
        let unchecked = TypedOrder::<Unchecked>::new(order);
        let verified = match unchecked.verify(&*self.risk_engine) {
            Ok(v) => v,
            Err(e) => {
                error!("Order verification failed: {}", e);
                return Ok(None);
            }
        };

        info!("Order fully verified and ready for execution");
        Ok(Some(verified.into_order()))
    }

    /// Handle executed trade - update positions
    pub async fn handle_trade(&self, trade: Trade) -> Result<()> {
        info!(
            "Handling executed trade: {} {:?} {}",
            trade.symbol,
            trade.side,
            trade.quantity.value()
        );

        let mut tracker = self.position_tracker.lock().await;
        tracker.update_from_trade(&trade.symbol, trade.side, trade.quantity, trade.price)?;

        // Update realized PnL if position was closed (simplified)
        // In production, you'd track this more accurately
        Ok(())
    }

    /// Update market price for position PnL calculation
    pub async fn update_price(&self, symbol: &str, price: Price) {
        let mut tracker = self.position_tracker.lock().await;
        tracker.update_price(symbol, price);
    }

    /// Calculate current leverage
    async fn calculate_leverage(&self) -> f64 {
        let tracker = self.position_tracker.lock().await;
        let sizer = self.position_sizer.lock().await;

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

    /// Get current state summary
    pub async fn get_state_summary(&self) -> StateSummary {
        let tracker = self.position_tracker.lock().await;
        let sizer = self.position_sizer.lock().await;

        // Calculate leverage inline to avoid deadlock
        let total_notional: f64 = tracker
            .all_positions()
            .values()
            .map(|p| p.notional_value().value())
            .sum();

        let portfolio_value = sizer.portfolio_value();
        let leverage = if portfolio_value > 0.0 {
            total_notional / portfolio_value
        } else {
            0.0
        };

        StateSummary {
            open_positions: tracker.open_positions_count(),
            total_unrealized_pnl: tracker.total_unrealized_pnl().value(),
            realized_pnl: self.realized_pnl,
            leverage,
            portfolio_value,
        }
    }
}

/// State summary for monitoring
#[derive(Debug, Clone)]
pub struct StateSummary {
    pub open_positions: usize,
    pub total_unrealized_pnl: f64,
    pub realized_pnl: f64,
    pub leverage: f64,
    pub portfolio_value: f64,
}

impl Default for StrategyActor {
    fn default() -> Self {
        // Create a default risk engine (will need config in production)
        let validator = logic::RiskValidator::new();
        let risk_engine = ComprehensiveRiskEngine::new(
            validator, 100.0,   // max_position_size
            -1000.0, // max_daily_loss
            3600,    // wash_sale_window
        );

        Self::new(
            risk_engine,
            100000.0, // portfolio_value
            "default_user".to_string(),
            TradingMode::Swing,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_strategy_actor_creation() {
        let actor = StrategyActor::default();
        let summary = actor.get_state_summary().await;
        assert_eq!(summary.open_positions, 0);
    }
}
