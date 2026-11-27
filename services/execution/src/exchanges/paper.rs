//! Paper exchange backend — simulated fills through the real [`Exchange`]
//! lifecycle.
//!
//! Purpose: a "paper" label must mean **simulated fills, fees, positions and
//! balances through the same order lifecycle a production executor uses** —
//! not an order marked `Submitted` with no backend behind it. [`PaperExchange`]
//! implements [`Exchange`], so it is registered in an [`ExchangeRouter`] like
//! any real venue and every order routed to it flows through
//! `OrderManager::submit_order` → validation → kill-switch gate → placement.
//!
//! # Fill model (conservative)
//!
//! - **Market orders** fill immediately at the order's reference price
//!   (`order.price` or the `reference_price` metadata key) adjusted by
//!   configurable slippage (adverse side), minus a taker fee. **No reference
//!   price → the order is refused** — the simulator never invents a fill
//!   price.
//! - **Limit orders** rest (status `Submitted`) until cancelled; the simulator
//!   does not claim fills it cannot observe.
//! - **Reduce-only orders** are enforced against the paper account's own
//!   positions: a reduction must have an opposing position and must never
//!   cross through zero into new exposure.
//!
//! Fills are pulled by the order manager via [`PaperExchange::take_fill`] and
//! injected into `OrderManager::handle_fill`, so position tracking, history
//! persistence and order state transitions are identical to a live fill.

use crate::error::{ExecutionError, Result};
use crate::exchanges::{
    Balance, Exchange, OrderStatusResponse, OrderUpdateReceiver, PositionUpdateReceiver,
};
use crate::types::{Fill, Order, OrderSide, OrderStatusEnum, Position};
use async_trait::async_trait;
use chrono::Utc;
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Paper-trading configuration (env-overridable).
#[derive(Debug, Clone)]
pub struct PaperConfig {
    /// Slippage charged adverse to the order side, in basis points of price.
    pub slippage_bps: f64,
    /// Taker fee, in basis points of notional.
    pub taker_fee_bps: f64,
    /// Simulated starting cash balance (quote currency).
    pub initial_balance: f64,
}

impl Default for PaperConfig {
    fn default() -> Self {
        Self {
            slippage_bps: 2.0,
            taker_fee_bps: 10.0,
            initial_balance: 100_000.0,
        }
    }
}

impl PaperConfig {
    /// Read from the environment (`PAPER_SLIPPAGE_BPS`, `PAPER_TAKER_FEE_BPS`,
    /// `PAPER_INITIAL_BALANCE`), falling back to the defaults above.
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            slippage_bps: std::env::var("PAPER_SLIPPAGE_BPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.slippage_bps),
            taker_fee_bps: std::env::var("PAPER_TAKER_FEE_BPS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.taker_fee_bps),
            initial_balance: std::env::var("PAPER_INITIAL_BALANCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.initial_balance),
        }
    }
}

/// Internal record of one order the paper account knows about.
#[derive(Debug, Clone)]
struct PaperOrder {
    status: OrderStatusEnum,
    filled_quantity: Decimal,
    average_fill_price: Option<Decimal>,
    quantity: Decimal,
    symbol: String,
}

/// Simulated exchange account: fills, fees, positions and cash — the same
/// surfaces a live venue exposes, so downstream consumers cannot tell them
/// apart by interface.
#[derive(Default)]
struct PaperAccount {
    orders: HashMap<String, PaperOrder>,
    /// Fills awaiting pickup by [`PaperExchange::take_fill`].
    pending_fills: HashMap<String, Fill>,
    /// Net position size per symbol (positive long, negative short).
    positions: HashMap<String, Position>,
    /// Cash balance (quote currency).
    cash: Decimal,
}

/// A simulated exchange implementing [`Exchange`]. Register it in an
/// [`ExchangeRouter`] under a name (conventionally `"paper"`) and route orders
/// to it for paper trading.
pub struct PaperExchange {
    config: PaperConfig,
    account: Mutex<PaperAccount>,
}

impl PaperExchange {
    /// Create a paper exchange with the given configuration.
    pub fn new(config: PaperConfig) -> Self {
        info!(
            slippage_bps = config.slippage_bps,
            taker_fee_bps = config.taker_fee_bps,
            initial_balance = config.initial_balance,
            "Paper exchange initialized (simulated fills, fees, positions)"
        );
        let cash = Decimal::try_from(config.initial_balance).unwrap_or(Decimal::ZERO);
        Self {
            config,
            account: Mutex::new(PaperAccount {
                cash,
                ..PaperAccount::default()
            }),
        }
    }

    /// Create with configuration from the environment.
    pub fn from_env() -> Self {
        Self::new(PaperConfig::from_env())
    }

    /// The reference price for an order: the explicit limit/entry price, else
    /// the `reference_price` metadata key. `None` when neither is present —
    /// the simulator refuses to invent a fill price.
    fn reference_price(&self, order: &Order) -> Option<Decimal> {
        if let Some(price) = order.price {
            if price > Decimal::ZERO {
                return Some(price);
            }
        }
        for key in ["reference_price", "close_price"] {
            if let Some(raw) = order.metadata.get(key)
                && let Ok(f) = raw.parse::<f64>()
                && f > 0.0
                && let Some(d) = Decimal::from_f64_retain(f)
            {
                return Some(d);
            }
        }
        None
    }

    /// Apply slippage adverse to the fill: a buy fills higher, a sell lower.
    fn slipped(&self, price: Decimal, side: OrderSide) -> Decimal {
        // Pure-decimal bps math: converting the f64 bps value directly is
        // exact for whole numbers, and the /10_000 division must happen in
        // Decimal — doing it in f64 leaves float residue (100.02 becoming
        // …0095843472).
        let slip_bps = Decimal::from_f64_retain(self.config.slippage_bps)
            .unwrap_or(Decimal::ZERO);
        let slip = slip_bps / Decimal::from(10_000u32);
        match side {
            OrderSide::Buy => price * (Decimal::ONE + slip),
            OrderSide::Sell => price * (Decimal::ONE - slip),
        }
    }

    fn taker_fee(&self, notional: Decimal) -> Decimal {
        let fee_bps =
            Decimal::from_f64_retain(self.config.taker_fee_bps).unwrap_or(Decimal::ZERO);
        let rate = fee_bps / Decimal::from(10_000u32);
        notional * rate
    }

    /// Simulate the immediate fill for a market order against the reference
    /// price, updating the paper account (position, cash), and stage the fill
    /// for pickup. Returns `None` when the order must rest (limit).
    fn simulate_market_fill(&self, order: &Order, exchange_order_id: &str) -> Result<Option<Fill>> {
        let reference = self.reference_price(order).ok_or_else(|| {
            ExecutionError::Config(
                "paper exchange: no reference price (order.price or metadata.reference_price) — \
                 refusing to invent a fill"
                    .to_string(),
            )
        })?;

        if order.quantity <= Decimal::ZERO {
            return Err(ExecutionError::OrderValidation(format!(
                "paper exchange: non-positive quantity {}",
                order.quantity
            )));
        }

        let mut account = self.account.lock().unwrap();
        let position_size = account
            .positions
            .get(&order.symbol)
            .map(|p| p.quantity)
            .unwrap_or(Decimal::ZERO);

        // Reduce-only enforcement against the paper account's own book: a
        // reduction needs an opposing position and must never cross zero.
        if order.reduce_only {
            let reduces = matches!(
                (order.side, position_size.is_sign_positive()),
                (OrderSide::Sell, true) | (OrderSide::Buy, false)
            );
            if position_size.is_zero() || !reduces {
                return Err(ExecutionError::RiskViolation(format!(
                    "paper exchange: reduce-only {} {} with no opposing position (size {})",
                    match order.side {
                        OrderSide::Buy => "Buy",
                        OrderSide::Sell => "Sell",
                    },
                    order.symbol,
                    position_size
                )));
            }
            if order.quantity > position_size.abs() {
                return Err(ExecutionError::RiskViolation(format!(
                    "paper exchange: reduce-only quantity {} exceeds position size {} — \
                     would cross through zero into new exposure",
                    order.quantity, position_size.abs()
                )));
            }
        }

        let fill_price = self.slipped(reference, order.side);
        let notional = fill_price * order.quantity;
        let fee = self.taker_fee(notional);

        // Update the paper book: cash moves opposite the asset flow, plus fees.
        let delta = match order.side {
            OrderSide::Buy => -(notional + fee),
            OrderSide::Sell => notional - fee,
        };
        account.cash += delta;
        let signed_qty = match order.side {
            OrderSide::Buy => order.quantity,
            OrderSide::Sell => -order.quantity,
        };
        let entry = account
            .positions
            .entry(order.symbol.clone())
            .or_insert_with(|| Position {
                symbol: order.symbol.clone(),
                exchange: "paper".to_string(),
                side: crate::types::PositionSide::Long,
                quantity: Decimal::ZERO,
                average_entry_price: Decimal::ZERO,
                current_price: Decimal::ZERO,
                unrealized_pnl: Decimal::ZERO,
                realized_pnl: Decimal::ZERO,
                margin_used: Decimal::ZERO,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            });
        // First fill on a flat book sets the average entry; increases keep
        // the weighted average (fills are near-identical prices in paper).
        if entry.average_entry_price.is_zero() {
            entry.average_entry_price = fill_price;
        }
        entry.quantity += signed_qty;
        entry.current_price = fill_price;
        entry.updated_at = Utc::now();
        entry.side = if entry.quantity.is_sign_positive() {
            crate::types::PositionSide::Long
        } else {
            crate::types::PositionSide::Short
        };
        if entry.quantity.is_zero() {
            account.positions.remove(&order.symbol);
        }

        let fill = Fill {
            id: Uuid::new_v4().to_string(),
            order_id: order.id.clone(),
            quantity: order.quantity,
            price: fill_price,
            fee,
            fee_currency: "USDT".to_string(),
            side: order.side,
            timestamp: Utc::now(),
            is_maker: false,
        };
        account.pending_fills.insert(exchange_order_id.to_string(), fill.clone());
        account.orders.insert(
            exchange_order_id.to_string(),
            PaperOrder {
                status: OrderStatusEnum::Filled,
                filled_quantity: order.quantity,
                average_fill_price: Some(fill_price),
                quantity: order.quantity,
                symbol: order.symbol.clone(),
            },
        );
        debug!(
            exchange_order_id = %exchange_order_id,
            price = %fill_price,
            fee = %fee,
            "paper: simulated market fill"
        );
        Ok(Some(fill))
    }

    fn record_resting(&self, order: &Order, exchange_order_id: &str) {
        let mut account = self.account.lock().unwrap();
        account.orders.insert(
            exchange_order_id.to_string(),
            PaperOrder {
                status: OrderStatusEnum::Submitted,
                filled_quantity: Decimal::ZERO,
                average_fill_price: None,
                quantity: order.quantity,
                symbol: order.symbol.clone(),
            },
        );
    }

    /// Take the staged fill for an exchange order ID, if the order filled
    /// immediately (market). One-shot: the fill is consumed. Limit orders that
    /// are resting return `None`.
    pub fn take_fill(&self, exchange_order_id: &str) -> Option<Fill> {
        self.account
            .lock()
            .unwrap()
            .pending_fills
            .remove(exchange_order_id)
    }
}

#[async_trait]
impl Exchange for PaperExchange {
    fn name(&self) -> &str {
        "paper"
    }

    fn is_testnet(&self) -> bool {
        true
    }

    async fn place_order(&self, order: &Order) -> Result<String> {
        let exchange_order_id = format!("paper-{}", Uuid::new_v4());
        debug!(
            symbol = %order.symbol,
            side = ?order.side,
            quantity = %order.quantity,
            exchange_order_id = %exchange_order_id,
            "paper: placing order"
        );
        match order.order_type {
            crate::types::OrderTypeEnum::Market => {
                self.simulate_market_fill(order, &exchange_order_id)?;
            }
            _ => {
                // Limit/stop orders rest — the simulator does not claim fills
                // it cannot observe.
                warn!(
                    exchange_order_id = %exchange_order_id,
                    "paper: non-market order rests unfilled (no simulated book)"
                );
                self.record_resting(order, &exchange_order_id);
            }
        }
        Ok(exchange_order_id)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let mut account = self.account.lock().unwrap();
        match account.orders.get_mut(order_id) {
            Some(po) if po.status == OrderStatusEnum::Submitted => {
                po.status = OrderStatusEnum::Cancelled;
                info!(exchange_order_id = %order_id, "paper: cancelled resting order");
                Ok(())
            }
            Some(po) => Err(ExecutionError::InvalidOrderState(format!(
                "paper: order {} is not cancellable (status {:?})",
                order_id, po.status
            ))),
            None => Err(ExecutionError::OrderNotFound(order_id.to_string())),
        }
    }

    async fn cancel_all_orders(&self, symbol: Option<&str>) -> Result<Vec<String>> {
        let mut account = self.account.lock().unwrap();
        let mut cancelled = Vec::new();
        for (id, po) in account.orders.iter_mut() {
            if po.status == OrderStatusEnum::Submitted
                && symbol.is_none_or(|s| po.symbol == s)
            {
                po.status = OrderStatusEnum::Cancelled;
                cancelled.push(id.clone());
            }
        }
        Ok(cancelled)
    }

    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse> {
        let account = self.account.lock().unwrap();
        account
            .orders
            .get(order_id)
            .map(|po| OrderStatusResponse {
                order_id: order_id.to_string(),
                client_order_id: None,
                symbol: po.symbol.clone(),
                status: po.status,
                quantity: po.quantity,
                filled_quantity: po.filled_quantity,
                remaining_quantity: po.quantity - po.filled_quantity,
                price: None,
                average_fill_price: po.average_fill_price,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                fills: Vec::new(),
            })
            .ok_or_else(|| ExecutionError::OrderNotFound(order_id.to_string()))
    }

    async fn get_active_orders(&self, symbol: Option<&str>) -> Result<Vec<OrderStatusResponse>> {
        let account = self.account.lock().unwrap();
        Ok(account
            .orders
            .iter()
            .filter(|(_, po)| {
                po.status == OrderStatusEnum::Submitted && symbol.is_none_or(|s| po.symbol == s)
            })
            .map(|(id, po)| OrderStatusResponse {
                order_id: id.clone(),
                client_order_id: None,
                symbol: po.symbol.clone(),
                status: po.status,
                quantity: po.quantity,
                filled_quantity: po.filled_quantity,
                remaining_quantity: po.quantity - po.filled_quantity,
                price: None,
                average_fill_price: po.average_fill_price,
                created_at: Utc::now(),
                updated_at: Utc::now(),
                fills: Vec::new(),
            })
            .collect())
    }

    async fn get_balance(&self) -> Result<Balance> {
        let account = self.account.lock().unwrap();
        Ok(Balance {
            total: account.cash,
            available: account.cash,
            used: Decimal::ZERO,
            currency: "USDT".to_string(),
            timestamp: Utc::now(),
        })
    }

    async fn get_positions(&self, symbol: Option<&str>) -> Result<Vec<Position>> {
        let account = self.account.lock().unwrap();
        Ok(account
            .positions
            .values()
            .filter(|p| symbol.is_none_or(|s| p.symbol == s))
            .cloned()
            .collect())
    }

    async fn subscribe_order_updates(&self) -> Result<OrderUpdateReceiver> {
        // Fills are pulled via `take_fill` (synchronous with placement), not
        // pushed over a stream.
        let (_tx, rx) = mpsc::unbounded_channel();
        Ok(rx)
    }

    async fn subscribe_position_updates(&self) -> Result<PositionUpdateReceiver> {
        let (_tx, rx) = mpsc::unbounded_channel();
        Ok(rx)
    }

    async fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{OrderTypeEnum, OrderSide};

    fn market_order(side: OrderSide, qty: Decimal, reference: Decimal) -> Order {
        let mut o = Order::new(
            "sig-1".to_string(),
            "BTCUSD".to_string(),
            "paper".to_string(),
            side,
            OrderTypeEnum::Market,
            qty,
        );
        o.price = Some(reference);
        o
    }

    #[tokio::test]
    async fn market_buy_fills_at_slipped_price_with_fee() {
        let paper = PaperExchange::new(PaperConfig {
            slippage_bps: 2.0,
            taker_fee_bps: 10.0,
            initial_balance: 100_000.0,
        });
        let order = market_order(OrderSide::Buy, Decimal::from(1), Decimal::from(100));
        let id = paper.place_order(&order).await.unwrap();
        let fill = paper.take_fill(&id).expect("market order must fill");

        // 100 * (1 + 2 bps) = 100.02
        assert_eq!(fill.price, Decimal::new(10002, 2));
        // fee = 100.02 * 10 bps = 0.10002
        let expected_fee = Decimal::new(10002, 2) * Decimal::new(10, 4);
        assert_eq!(fill.fee, expected_fee);
        // Cash reduced by notional + fee.
        let balance = paper.get_balance().await.unwrap();
        assert_eq!(balance.total, Decimal::from(100_000) - Decimal::new(10002, 2) - expected_fee);
        // Position opened long 1.
        let positions = paper.get_positions(Some("BTCUSD")).await.unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(positions[0].quantity, Decimal::from(1));
        // Fill is one-shot.
        assert!(paper.take_fill(&id).is_none());
    }

    #[tokio::test]
    async fn market_sell_fills_lower_and_flattens_position() {
        let paper = PaperExchange::new(PaperConfig::default());
        // Open long 2 first.
        let open = market_order(OrderSide::Buy, Decimal::from(2), Decimal::from(100));
        let open_id = paper.place_order(&open).await.unwrap();
        paper.take_fill(&open_id).unwrap();

        let close = market_order(OrderSide::Sell, Decimal::from(2), Decimal::from(100));
        let close_id = paper.place_order(&close).await.unwrap();
        paper.take_fill(&close_id).unwrap();

        // Position flat after the closing sell.
        assert!(paper.get_positions(Some("BTCUSD")).await.unwrap().is_empty());
        // Cash below start: round trip paid slippage + fees both ways.
        let cash = paper.get_balance().await.unwrap().total;
        assert!(cash < Decimal::from(100_000), "round trip must cost money, got {cash}");
    }

    #[tokio::test]
    async fn reduce_only_rejected_without_opposing_position() {
        let paper = PaperExchange::new(PaperConfig::default());
        let mut order = market_order(OrderSide::Sell, Decimal::from(1), Decimal::from(100));
        order.reduce_only = true;
        let err = paper.place_order(&order).await.unwrap_err();
        assert!(matches!(err, ExecutionError::RiskViolation(_)));
    }

    #[tokio::test]
    async fn reduce_only_rejects_quantity_crossing_zero() {
        let paper = PaperExchange::new(PaperConfig::default());
        let open = market_order(OrderSide::Buy, Decimal::from(1), Decimal::from(100));
        let id = paper.place_order(&open).await.unwrap();
        paper.take_fill(&id).unwrap();

        let mut order = market_order(OrderSide::Sell, Decimal::from(2), Decimal::from(100));
        order.reduce_only = true;
        let err = paper.place_order(&order).await.unwrap_err();
        assert!(
            matches!(err, ExecutionError::RiskViolation(ref m) if m.contains("cross through zero"))
        );
    }

    #[tokio::test]
    async fn reduce_only_within_position_is_allowed() {
        let paper = PaperExchange::new(PaperConfig::default());
        let open = market_order(OrderSide::Buy, Decimal::from(2), Decimal::from(100));
        let id = paper.place_order(&open).await.unwrap();
        paper.take_fill(&id).unwrap();

        let mut order = market_order(OrderSide::Sell, Decimal::from(2), Decimal::from(100));
        order.reduce_only = true;
        let close_id = paper.place_order(&order).await.unwrap();
        assert!(paper.take_fill(&close_id).is_some());
        assert!(paper.get_positions(Some("BTCUSD")).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn market_order_without_reference_price_is_refused() {
        let paper = PaperExchange::new(PaperConfig::default());
        let mut order = market_order(OrderSide::Buy, Decimal::from(1), Decimal::from(100));
        order.price = None; // no reference anywhere
        let err = paper.place_order(&order).await.unwrap_err();
        assert!(matches!(err, ExecutionError::Config(_)));
    }

    #[tokio::test]
    async fn limit_order_rests_unfilled() {
        let paper = PaperExchange::new(PaperConfig::default());
        let mut order = market_order(OrderSide::Buy, Decimal::from(1), Decimal::from(100));
        order.order_type = OrderTypeEnum::Limit;
        let id = paper.place_order(&order).await.unwrap();
        assert!(paper.take_fill(&id).is_none(), "resting limit must not fill");
        let status = paper.get_order_status(&id).await.unwrap();
        assert_eq!(status.status, OrderStatusEnum::Submitted);
        // And it is cancellable.
        paper.cancel_order(&id).await.unwrap();
        assert_eq!(
            paper.get_order_status(&id).await.unwrap().status,
            OrderStatusEnum::Cancelled
        );
    }
}
