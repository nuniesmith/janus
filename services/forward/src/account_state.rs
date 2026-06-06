//! Live Bybit account state, reconciled from the private user-data feed.
//!
//! This folds the order / fill / position / balance events produced by
//! [`crate::bybit_compat::BybitPrivateWebSocket`] into an in-memory account
//! picture for observability and (future) risk enforcement.
//!
//! ## NO ORDER ENTRY
//! This module is **read-only with respect to trading**. It must never initiate
//! orders: nothing here imports the execution client, the brain-gated client,
//! or `bybit_compat::BybitRestClient`, and it never calls
//! `SignalGenerator::submit_signal_to_execution`. It only acquires a write lock
//! on a [`LiveAccount`] and mutates that model. This preserves janus's
//! "manual trading co-pilot — all signal flow terminates at a human decision
//! point" principle: the private feed is purely an inbound state source.
//!
//! ## Reconciliation model
//! Exchange **position snapshots** (`PositionUpdate`) are authoritative — they
//! overwrite the position. **Fills** (`ExecutionUpdate`) fold in as best-effort
//! signed deltas (average-entry + realised-PnL maths) between snapshots; any
//! drift is corrected by the next snapshot.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::bybit_compat::{BybitCredentials, BybitPrivateWebSocket, WsMessage};

/// Credentials + flags for the optional Bybit private user-data feed.
#[derive(Clone)]
pub struct BybitPrivateFeedConfig {
    pub key: String,
    pub secret: String,
    pub testnet: bool,
}

impl std::fmt::Debug for BybitPrivateFeedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never leak credentials through Debug.
        f.debug_struct("BybitPrivateFeedConfig")
            .field("key", &"***REDACTED***")
            .field("secret", &"***REDACTED***")
            .field("testnet", &self.testnet)
            .finish()
    }
}

/// A single open position. `qty` is signed: positive = long, negative = short.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LivePosition {
    pub symbol: String,
    pub qty: f64,
    pub avg_entry: f64,
    pub unrealized_pnl: f64,
    pub realized_pnl: f64,
}

/// A resting / partially-filled order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OrderState {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub price: f64,
    pub qty: f64,
    pub remaining: f64,
    pub status: String,
}

/// Wallet balance for one currency.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Balance {
    pub available: f64,
    pub hold: f64,
}

/// In-memory account picture reconciled from the Bybit private feed.
#[derive(Debug, Clone, Default)]
pub struct LiveAccount {
    /// Open positions by symbol.
    pub positions: HashMap<String, LivePosition>,
    /// Resting / partially-filled orders by exchange order id.
    pub open_orders: HashMap<String, OrderState>,
    /// Wallet balances by currency.
    pub balances: HashMap<String, Balance>,
    /// Most recent event timestamp applied (ms; `receipt_ts` preferred).
    pub last_update_ms: i64,
    /// Count of fills folded in (observability).
    pub fills_applied: u64,
}

impl LiveAccount {
    /// Order-state update: upsert resting orders, drop terminal ones.
    pub fn apply_order(&mut self, v: &Value) {
        self.touch(v);
        let order_id = text(v, "order_id");
        if order_id.is_empty() {
            return;
        }
        let status = text(v, "status");
        if is_terminal(&status) {
            self.open_orders.remove(&order_id);
            return;
        }
        self.open_orders.insert(
            order_id.clone(),
            OrderState {
                order_id,
                symbol: text(v, "symbol"),
                side: text(v, "side"),
                price: num(v, "price"),
                qty: num(v, "size"),
                remaining: num(v, "remaining_size"),
                status,
            },
        );
    }

    /// A fill: fold into the position via average-entry / realised-PnL maths.
    /// Best-effort — a subsequent [`Self::apply_position`] snapshot is truth.
    pub fn apply_fill(&mut self, v: &Value) {
        self.touch(v);
        let symbol = text(v, "symbol");
        let (price, size) = match (
            v.get("match_price").and_then(Value::as_f64),
            v.get("match_size").and_then(Value::as_f64),
        ) {
            (Some(p), Some(s)) if s > 0.0 => (p, s),
            _ => return, // an order-state frame, not a fill
        };
        if symbol.is_empty() {
            return;
        }
        let delta = if side_is_buy(v) { size } else { -size };
        let pos = self.positions.entry(symbol.clone()).or_default();
        pos.symbol = symbol;
        apply_fill_to_position(pos, delta, price, num(v, "fee"));
        self.fills_applied += 1;
    }

    /// Authoritative position snapshot from the exchange — overwrites the model.
    pub fn apply_position(&mut self, v: &Value) {
        self.touch(v);
        let symbol = text(v, "symbol");
        if symbol.is_empty() {
            return;
        }
        let qty = num(v, "current_qty");
        if is_zero(qty) {
            self.positions.remove(&symbol);
            return;
        }
        let pos = self.positions.entry(symbol.clone()).or_default();
        pos.symbol = symbol;
        pos.qty = qty;
        pos.avg_entry = num(v, "avg_entry_price");
        pos.unrealized_pnl = num(v, "unrealised_pnl");
        pos.realized_pnl = num(v, "realised_pnl");
    }

    /// Wallet balance update for one currency.
    pub fn apply_balance(&mut self, v: &Value) {
        self.touch(v);
        let currency = text(v, "currency");
        if currency.is_empty() {
            return;
        }
        self.balances.insert(
            currency,
            Balance {
                available: num(v, "available_balance"),
                hold: num(v, "hold_balance"),
            },
        );
    }

    /// Net unrealised PnL across all open positions.
    pub fn net_unrealized_pnl(&self) -> f64 {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }

    fn touch(&mut self, v: &Value) {
        let ts = v
            .get("receipt_ts")
            .or_else(|| v.get("exchange_ts"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        if ts > self.last_update_ms {
            self.last_update_ms = ts;
        }
    }
}

/// Route one feed message into the account model. Factored out (pure w.r.t. the
/// account) for unit testing; non-private variants (`Tick`/`Trade`/`Kline`) are
/// ignored.
pub fn apply_ws_message(account: &mut LiveAccount, msg: WsMessage) {
    match msg {
        WsMessage::OrderUpdate(v) => account.apply_order(&v),
        WsMessage::ExecutionUpdate(v) => account.apply_fill(&v),
        WsMessage::PositionUpdate(v) => account.apply_position(&v),
        WsMessage::BalanceUpdate(v) => account.apply_balance(&v),
        _ => {}
    }
}

/// Drive the Bybit private feed, reconciling every event into `account`, with
/// reconnect + exponential backoff. Runs until the task is dropped; transient
/// disconnects are retried. **Never places orders.**
pub async fn run_private_feed(config: BybitPrivateFeedConfig, account: Arc<RwLock<LiveAccount>>) {
    let mut backoff = Duration::from_secs(1);
    let max_backoff = Duration::from_secs(60);
    loop {
        let creds = BybitCredentials::new(config.key.clone(), config.secret.clone());
        // `connect()` consumes the driver, so it's rebuilt each attempt.
        let (ws, mut rx) = BybitPrivateWebSocket::new(creds, config.testnet);

        let acct = Arc::clone(&account);
        let consumer = tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                let mut a = acct.write().await;
                apply_ws_message(&mut a, msg);
            }
        });

        info!(
            testnet = config.testnet,
            "📒 Bybit private feed: connecting"
        );
        match ws.connect().await {
            Ok(()) => warn!("Bybit private feed closed; reconnecting"),
            Err(e) => warn!(error = %e, "Bybit private feed error; reconnecting"),
        }
        // The feed dropped its sender, so the consumer drains and ends; abort is
        // a belt-and-suspenders guard before the next attempt.
        consumer.abort();
        tokio::time::sleep(backoff).await;
        backoff = (backoff * 2).min(max_backoff);
    }
}

// ── helpers ────────────────────────────────────────────────────────────────

/// Fold a signed `delta` fill at `price` (paying `fee`) into a position,
/// maintaining signed qty, weighted-average entry, and realised PnL.
fn apply_fill_to_position(pos: &mut LivePosition, delta: f64, price: f64, fee: f64) {
    pos.realized_pnl -= fee; // fees always reduce realised PnL
    if is_zero(pos.qty) || same_sign(pos.qty, delta) {
        // Opening or increasing — weight the average entry by absolute size.
        let denom = pos.qty.abs() + delta.abs();
        if denom > 0.0 {
            pos.avg_entry = (pos.avg_entry * pos.qty.abs() + price * delta.abs()) / denom;
        }
        pos.qty += delta;
    } else {
        // Reducing / closing / flipping — realise PnL on the closed portion.
        let closing = delta.abs().min(pos.qty.abs());
        let dir = if pos.qty > 0.0 { 1.0 } else { -1.0 };
        pos.realized_pnl += (price - pos.avg_entry) * closing * dir;
        let new_qty = pos.qty + delta;
        if is_zero(new_qty) {
            pos.qty = 0.0;
            pos.avg_entry = 0.0;
        } else if same_sign(new_qty, pos.qty) {
            pos.qty = new_qty; // partial close — entry unchanged
        } else {
            pos.qty = new_qty; // flip — remainder opens at the fill price
            pos.avg_entry = price;
        }
    }
}

/// Bybit order statuses that mean the order is no longer resting.
fn is_terminal(status: &str) -> bool {
    matches!(
        status,
        "Filled" | "Cancelled" | "Canceled" | "Rejected" | "Deactivated" | "Closed"
    )
}

/// `exchange-apiws` serialises `TradeSide` as `"Buy"`/`"Sell"`.
fn side_is_buy(v: &Value) -> bool {
    matches!(
        v.get("side").and_then(Value::as_str),
        Some("Buy") | Some("buy")
    )
}

fn num(v: &Value, key: &str) -> f64 {
    v.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn text(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn is_zero(x: f64) -> bool {
    x.abs() < 1e-9
}

fn same_sign(a: f64, b: f64) -> bool {
    (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn fill_maths_open_add_partial_close_and_flip() {
        let mut pos = LivePosition::default();

        // Open long 1 @ 100, fee 0.1.
        apply_fill_to_position(&mut pos, 1.0, 100.0, 0.1);
        assert!(approx(pos.qty, 1.0) && approx(pos.avg_entry, 100.0));
        assert!(approx(pos.realized_pnl, -0.1));

        // Add 1 @ 200 → qty 2, avg 150.
        apply_fill_to_position(&mut pos, 1.0, 200.0, 0.0);
        assert!(approx(pos.qty, 2.0) && approx(pos.avg_entry, 150.0));

        // Partial close: sell 1 @ 200 → realise (200-150)*1 = +50, qty 1, avg 150.
        apply_fill_to_position(&mut pos, -1.0, 200.0, 0.0);
        assert!(approx(pos.qty, 1.0) && approx(pos.avg_entry, 150.0));
        assert!(approx(pos.realized_pnl, -0.1 + 50.0));

        // Flip: sell 3 @ 200 → close last 1 (+50), open short 2 @ 200.
        apply_fill_to_position(&mut pos, -3.0, 200.0, 0.0);
        assert!(approx(pos.qty, -2.0) && approx(pos.avg_entry, 200.0));
        assert!(approx(pos.realized_pnl, -0.1 + 100.0));
    }

    #[test]
    fn fill_maths_short_pnl() {
        let mut pos = LivePosition::default();
        // Open short 2 @ 100.
        apply_fill_to_position(&mut pos, -2.0, 100.0, 0.0);
        assert!(approx(pos.qty, -2.0) && approx(pos.avg_entry, 100.0));
        // Cover 1 @ 80 → short profits (100-80)*1 = +20, qty -1.
        apply_fill_to_position(&mut pos, 1.0, 80.0, 0.0);
        assert!(approx(pos.qty, -1.0) && approx(pos.avg_entry, 100.0));
        assert!(approx(pos.realized_pnl, 20.0));
    }

    #[test]
    fn apply_fill_reads_execution_json() {
        let mut acct = LiveAccount::default();
        // Mirrors the shape translate_private emits for a Bybit fill.
        acct.apply_fill(&json!({
            "symbol": "BTCUSDT", "side": "Buy",
            "match_price": 30000.0, "match_size": 0.5, "fee": 0.0,
            "receipt_ts": 42
        }));
        let pos = acct.positions.get("BTCUSDT").expect("position opened");
        assert!(approx(pos.qty, 0.5) && approx(pos.avg_entry, 30000.0));
        assert_eq!(acct.fills_applied, 1);
        assert_eq!(acct.last_update_ms, 42);
        // A frame without match_price is order-state, not a fill → ignored here.
        acct.apply_fill(&json!({ "symbol": "BTCUSDT", "side": "Buy" }));
        assert_eq!(acct.fills_applied, 1);
    }

    #[test]
    fn position_snapshot_is_authoritative() {
        let mut acct = LiveAccount::default();
        acct.apply_fill(&json!({
            "symbol": "ETHUSDT", "side": "Buy", "match_price": 2000.0, "match_size": 3.0
        }));
        // Snapshot overrides the fill-derived view.
        acct.apply_position(&json!({
            "symbol": "ETHUSDT", "current_qty": -5, "avg_entry_price": 2100.0,
            "unrealised_pnl": 12.0, "realised_pnl": 3.0
        }));
        let pos = acct.positions.get("ETHUSDT").unwrap();
        assert!(approx(pos.qty, -5.0) && approx(pos.avg_entry, 2100.0));
        assert!(approx(pos.unrealized_pnl, 12.0) && approx(pos.realized_pnl, 3.0));
        // current_qty 0 closes the position.
        acct.apply_position(&json!({ "symbol": "ETHUSDT", "current_qty": 0 }));
        assert!(!acct.positions.contains_key("ETHUSDT"));
    }

    #[test]
    fn orders_upsert_then_terminal_removes() {
        let mut acct = LiveAccount::default();
        acct.apply_order(&json!({
            "order_id": "o-1", "symbol": "BTCUSDT", "side": "Buy",
            "price": 100.0, "size": 1.0, "remaining_size": 1.0, "status": "New"
        }));
        assert_eq!(acct.open_orders.len(), 1);
        assert_eq!(acct.open_orders["o-1"].status, "New");
        // Terminal status drops it.
        acct.apply_order(&json!({ "order_id": "o-1", "status": "Filled" }));
        assert!(acct.open_orders.is_empty());
    }

    #[test]
    fn balance_update_sets_currency() {
        let mut acct = LiveAccount::default();
        acct.apply_balance(&json!({
            "currency": "USDT", "available_balance": 1234.5, "hold_balance": 6.0
        }));
        let b = acct.balances.get("USDT").unwrap();
        assert!(approx(b.available, 1234.5) && approx(b.hold, 6.0));
    }

    #[test]
    fn apply_ws_message_routes_each_variant() {
        let mut acct = LiveAccount::default();
        apply_ws_message(
            &mut acct,
            WsMessage::PositionUpdate(
                json!({ "symbol": "BTCUSDT", "current_qty": 2, "avg_entry_price": 100.0 }),
            ),
        );
        apply_ws_message(
            &mut acct,
            WsMessage::BalanceUpdate(json!({ "currency": "USDT", "available_balance": 50.0 })),
        );
        // A non-private variant is a no-op.
        apply_ws_message(&mut acct, WsMessage::Kline(json!({})));
        assert_eq!(acct.positions.len(), 1);
        assert_eq!(acct.balances.len(), 1);
    }

    #[test]
    fn config_debug_redacts_secrets() {
        let cfg = BybitPrivateFeedConfig {
            key: "AKIA-super-secret".into(),
            secret: "shhhh".into(),
            testnet: true,
        };
        let dbg = format!("{cfg:?}");
        assert!(!dbg.contains("super-secret") && !dbg.contains("shhhh"));
        assert!(dbg.contains("REDACTED") && dbg.contains("testnet: true"));
    }
}
