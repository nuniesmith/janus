//! Position Tracker
//!
//! Tracks positions across exchanges with real-time P&L calculations,
//! aggregation, and risk metrics.

use crate::error::{ExecutionError, Result};
use crate::types::OrderSide;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};

/// Position for a single symbol on an exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol
    pub symbol: String,

    /// Exchange name
    pub exchange: String,

    /// Current position size (positive = long, negative = short, 0 = flat)
    pub size: Decimal,

    /// Average entry price
    pub entry_price: Decimal,

    /// Current market price
    pub mark_price: Decimal,

    /// Leverage multiplier
    pub leverage: Decimal,

    /// Liquidation price (if applicable)
    pub liquidation_price: Option<Decimal>,

    /// Unrealized P&L in quote currency
    pub unrealized_pnl: Decimal,

    /// Unrealized P&L percentage
    pub unrealized_pnl_pct: Decimal,

    /// Realized P&L (from closed trades)
    pub realized_pnl: Decimal,

    /// Total P&L (realized + unrealized)
    pub total_pnl: Decimal,

    /// Position value in quote currency
    pub position_value: Decimal,

    /// Initial margin required
    pub initial_margin: Decimal,

    /// Maintenance margin required
    pub maintenance_margin: Decimal,

    /// Timestamp of last update
    pub updated_at: chrono::DateTime<chrono::Utc>,

    /// Position side (Long, Short, or Flat)
    pub side: PositionSide,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

impl Position {
    /// Create a new empty position
    pub fn new(symbol: String, exchange: String) -> Self {
        Self {
            symbol,
            exchange,
            size: Decimal::ZERO,
            entry_price: Decimal::ZERO,
            mark_price: Decimal::ZERO,
            leverage: Decimal::ONE,
            liquidation_price: None,
            unrealized_pnl: Decimal::ZERO,
            unrealized_pnl_pct: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            total_pnl: Decimal::ZERO,
            position_value: Decimal::ZERO,
            initial_margin: Decimal::ZERO,
            maintenance_margin: Decimal::ZERO,
            updated_at: chrono::Utc::now(),
            side: PositionSide::Flat,
        }
    }

    /// Update position from a fill
    pub fn apply_fill(
        &mut self,
        side: OrderSide,
        fill_qty: Decimal,
        fill_price: Decimal,
    ) -> Result<()> {
        if fill_qty <= Decimal::ZERO {
            return Err(ExecutionError::InvalidQuantity(
                "Fill quantity must be positive".to_string(),
            ));
        }

        let old_size = self.size;
        let signed_qty = match side {
            OrderSide::Buy => fill_qty,
            OrderSide::Sell => -fill_qty,
        };

        let new_size = old_size + signed_qty;

        // Helper functions for sign checking
        let is_positive = |x: Decimal| x > Decimal::ZERO;
        let is_negative = |x: Decimal| x < Decimal::ZERO;
        let same_sign = |a: Decimal, b: Decimal| {
            (is_positive(a) && is_positive(b)) || (is_negative(a) && is_negative(b))
        };

        // Calculate realized P&L if closing or reducing position
        // This happens when the position size decreases in absolute value
        let old_abs = old_size.abs();
        let new_abs = new_size.abs();

        if old_size != Decimal::ZERO && new_abs < old_abs {
            // Position is being reduced or fully closed
            let closed_qty = old_abs - new_abs;

            if closed_qty > Decimal::ZERO && self.entry_price > Decimal::ZERO {
                let pnl_per_unit = if old_size > Decimal::ZERO {
                    // Closing long
                    fill_price - self.entry_price
                } else {
                    // Closing short
                    self.entry_price - fill_price
                };

                let realized = pnl_per_unit * closed_qty;
                self.realized_pnl += realized;

                debug!(
                    "Realized P&L: {} (closed {} @ {} from entry {})",
                    realized, closed_qty, fill_price, self.entry_price
                );
            }
        }

        // Update position size and entry price
        if same_sign(new_size, old_size) || old_size == Decimal::ZERO {
            // Adding to position or opening new
            let total_cost = old_size.abs() * self.entry_price + fill_qty * fill_price;
            self.size = new_size;
            if new_size != Decimal::ZERO {
                self.entry_price = total_cost / new_size.abs();
            }
        } else {
            // Position was reduced or flipped
            self.size = new_size;
            if !same_sign(new_size, old_size) && new_size != Decimal::ZERO {
                // Flipped - new entry price is the fill price
                self.entry_price = fill_price;
            }
        }

        // Update side
        self.side = if self.size > Decimal::ZERO {
            PositionSide::Long
        } else if self.size < Decimal::ZERO {
            PositionSide::Short
        } else {
            PositionSide::Flat
        };

        self.updated_at = chrono::Utc::now();
        self.recalculate_pnl();

        Ok(())
    }

    /// Update mark price and recalculate P&L
    pub fn update_mark_price(&mut self, mark_price: Decimal) {
        self.mark_price = mark_price;
        self.recalculate_pnl();
        self.updated_at = chrono::Utc::now();
    }

    /// Recalculate unrealized P&L and other metrics
    fn recalculate_pnl(&mut self) {
        if self.size == Decimal::ZERO {
            self.unrealized_pnl = Decimal::ZERO;
            self.unrealized_pnl_pct = Decimal::ZERO;
            self.position_value = Decimal::ZERO;
            return;
        }

        // Position value
        self.position_value = self.size.abs() * self.mark_price;

        // Unrealized P&L
        if self.entry_price > Decimal::ZERO && self.mark_price > Decimal::ZERO {
            let pnl_per_unit = if self.size > Decimal::ZERO {
                // Long position
                self.mark_price - self.entry_price
            } else {
                // Short position
                self.entry_price - self.mark_price
            };

            self.unrealized_pnl = pnl_per_unit * self.size.abs();

            // Unrealized P&L percentage
            let cost_basis = self.entry_price * self.size.abs();
            if cost_basis > Decimal::ZERO {
                self.unrealized_pnl_pct = (self.unrealized_pnl / cost_basis) * Decimal::from(100);
            }
        }

        // Total P&L
        self.total_pnl = self.realized_pnl + self.unrealized_pnl;

        // Margin calculations
        if self.leverage > Decimal::ZERO {
            self.initial_margin = self.position_value / self.leverage;
            self.maintenance_margin = self.initial_margin * Decimal::from_str_exact("0.5").unwrap();
            // Example: 50% of initial
        }
    }

    /// Set leverage
    pub fn set_leverage(&mut self, leverage: Decimal) {
        self.leverage = leverage;
        self.recalculate_pnl();
    }

    /// Set liquidation price
    pub fn set_liquidation_price(&mut self, liq_price: Decimal) {
        self.liquidation_price = Some(liq_price);
    }

    /// Is position flat?
    pub fn is_flat(&self) -> bool {
        self.size == Decimal::ZERO
    }

    /// Is position long?
    pub fn is_long(&self) -> bool {
        self.size > Decimal::ZERO
    }

    /// Is position short?
    pub fn is_short(&self) -> bool {
        self.size < Decimal::ZERO
    }

    /// Get position direction
    pub fn direction(&self) -> &str {
        match self.side {
            PositionSide::Long => "LONG",
            PositionSide::Short => "SHORT",
            PositionSide::Flat => "FLAT",
        }
    }
}

/// Key for position map (exchange + symbol)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PositionKey {
    exchange: String,
    symbol: String,
}

/// Position tracker managing all positions across exchanges
pub struct PositionTracker {
    /// All positions by exchange and symbol
    positions: Arc<RwLock<HashMap<PositionKey, Position>>>,

    /// Global statistics
    stats: Arc<RwLock<PositionStats>>,
}

/// Aggregated position statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PositionStats {
    /// Total number of positions
    pub total_positions: usize,

    /// Number of long positions
    pub long_positions: usize,

    /// Number of short positions
    pub short_positions: usize,

    /// Total unrealized P&L across all positions
    pub total_unrealized_pnl: Decimal,

    /// Total realized P&L across all positions
    pub total_realized_pnl: Decimal,

    /// Total P&L (realized + unrealized)
    pub total_pnl: Decimal,

    /// Total position value
    pub total_position_value: Decimal,

    /// Total initial margin
    pub total_initial_margin: Decimal,

    /// Total maintenance margin
    pub total_maintenance_margin: Decimal,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl PositionTracker {
    /// Create a new position tracker
    pub fn new() -> Self {
        Self {
            positions: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(PositionStats::default())),
        }
    }

    /// Get or create a position
    pub async fn get_position(&self, exchange: &str, symbol: String) -> Position {
        let key = PositionKey {
            exchange: exchange.to_string(),
            symbol: symbol.clone(),
        };

        let positions = self.positions.read().await;
        positions
            .get(&key)
            .cloned()
            .unwrap_or_else(|| Position::new(symbol, exchange.to_string()))
    }

    /// Get all positions
    pub async fn get_all_positions(&self) -> Vec<Position> {
        let positions = self.positions.read().await;
        positions.values().cloned().collect()
    }

    /// Get positions for a specific exchange
    pub async fn get_positions_by_exchange(&self, exchange: &str) -> Vec<Position> {
        let positions = self.positions.read().await;
        positions
            .values()
            .filter(|p| p.exchange == exchange)
            .cloned()
            .collect()
    }

    /// Get positions for a specific symbol (across all exchanges)
    pub async fn get_positions_by_symbol(&self, symbol: &str) -> Vec<Position> {
        let positions = self.positions.read().await;
        positions
            .values()
            .filter(|p| p.symbol == symbol)
            .cloned()
            .collect()
    }

    /// Update position from a fill
    pub async fn apply_fill(
        &self,
        exchange: &str,
        symbol: String,
        side: OrderSide,
        fill_qty: Decimal,
        fill_price: Decimal,
    ) -> Result<Position> {
        let key = PositionKey {
            exchange: exchange.to_string(),
            symbol: symbol.clone(),
        };

        let mut positions = self.positions.write().await;
        let position = positions
            .entry(key)
            .or_insert_with(|| Position::new(symbol, exchange.to_string()));

        position.apply_fill(side, fill_qty, fill_price)?;

        let updated = position.clone();
        drop(positions);

        // Update stats
        self.update_stats().await;

        info!(
            "Position updated: {} {} {} @ {} (size: {}, pnl: {})",
            exchange, updated.symbol, side, fill_price, updated.size, updated.total_pnl
        );

        Ok(updated)
    }

    /// Update mark price for a position
    pub async fn update_mark_price(
        &self,
        exchange: &str,
        symbol: String,
        mark_price: Decimal,
    ) -> Result<()> {
        let key = PositionKey {
            exchange: exchange.to_string(),
            symbol,
        };

        let mut positions = self.positions.write().await;
        if let Some(position) = positions.get_mut(&key) {
            position.update_mark_price(mark_price);
            debug!(
                "Mark price updated: {} {} @ {} (unrealized pnl: {})",
                exchange, position.symbol, mark_price, position.unrealized_pnl
            );
        }
        drop(positions);

        // Update stats
        self.update_stats().await;

        Ok(())
    }

    /// Set leverage for a position
    pub async fn set_leverage(
        &self,
        exchange: &str,
        symbol: String,
        leverage: Decimal,
    ) -> Result<()> {
        let key = PositionKey {
            exchange: exchange.to_string(),
            symbol: symbol.clone(),
        };

        let mut positions = self.positions.write().await;
        let position = positions
            .entry(key)
            .or_insert_with(|| Position::new(symbol, exchange.to_string()));

        position.set_leverage(leverage);
        info!(
            "Leverage set: {} {} = {}x",
            exchange, position.symbol, leverage
        );

        Ok(())
    }

    /// Update statistics
    async fn update_stats(&self) {
        let positions = self.positions.read().await;

        let mut stats = PositionStats {
            total_positions: 0,
            long_positions: 0,
            short_positions: 0,
            total_unrealized_pnl: Decimal::ZERO,
            total_realized_pnl: Decimal::ZERO,
            total_pnl: Decimal::ZERO,
            total_position_value: Decimal::ZERO,
            total_initial_margin: Decimal::ZERO,
            total_maintenance_margin: Decimal::ZERO,
            updated_at: chrono::Utc::now(),
        };

        for position in positions.values() {
            if !position.is_flat() {
                stats.total_positions += 1;

                if position.is_long() {
                    stats.long_positions += 1;
                } else if position.is_short() {
                    stats.short_positions += 1;
                }

                stats.total_unrealized_pnl += position.unrealized_pnl;
                stats.total_realized_pnl += position.realized_pnl;
                stats.total_position_value += position.position_value;
                stats.total_initial_margin += position.initial_margin;
                stats.total_maintenance_margin += position.maintenance_margin;
            }
        }

        stats.total_pnl = stats.total_realized_pnl + stats.total_unrealized_pnl;

        *self.stats.write().await = stats;
    }

    /// Get aggregated statistics
    pub async fn get_stats(&self) -> PositionStats {
        self.stats.read().await.clone()
    }

    /// Close a position (set size to zero)
    pub async fn close_position(&self, exchange: &str, symbol: String) -> Result<()> {
        let key = PositionKey {
            exchange: exchange.to_string(),
            symbol: symbol.clone(),
        };

        let mut positions = self.positions.write().await;
        if let Some(position) = positions.get_mut(&key) {
            position.size = Decimal::ZERO;
            position.side = PositionSide::Flat;
            position.recalculate_pnl();
            info!("Position closed: {} {}", exchange, symbol);
        }
        drop(positions);

        self.update_stats().await;
        Ok(())
    }

    /// Clear all positions (for testing/reset)
    pub async fn clear_all(&self) {
        let mut positions = self.positions.write().await;
        positions.clear();
        drop(positions);

        *self.stats.write().await = PositionStats::default();
        info!("All positions cleared");
    }
}

impl Default for PositionTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let pos = Position::new("BTCUSD".to_string(), "bybit".to_string());
        assert_eq!(pos.size, Decimal::ZERO);
        assert_eq!(pos.side, PositionSide::Flat);
        assert!(pos.is_flat());
    }

    #[test]
    fn test_open_long_position() {
        let mut pos = Position::new("BTCUSD".to_string(), "bybit".to_string());

        // Buy 1 BTC at 50000
        pos.apply_fill(OrderSide::Buy, Decimal::ONE, Decimal::from(50000))
            .unwrap();

        assert_eq!(pos.size, Decimal::ONE);
        assert_eq!(pos.entry_price, Decimal::from(50000));
        assert_eq!(pos.side, PositionSide::Long);
        assert!(pos.is_long());
    }

    #[test]
    fn test_open_short_position() {
        let mut pos = Position::new("BTCUSD".to_string(), "bybit".to_string());

        // Sell 1 BTC at 50000
        pos.apply_fill(OrderSide::Sell, Decimal::ONE, Decimal::from(50000))
            .unwrap();

        assert_eq!(pos.size, Decimal::from(-1));
        assert_eq!(pos.entry_price, Decimal::from(50000));
        assert_eq!(pos.side, PositionSide::Short);
        assert!(pos.is_short());
    }

    #[test]
    fn test_add_to_long_position() {
        let mut pos = Position::new("BTCUSD".to_string(), "bybit".to_string());

        // Buy 1 BTC at 50000
        pos.apply_fill(OrderSide::Buy, Decimal::ONE, Decimal::from(50000))
            .unwrap();

        // Buy 1 more BTC at 51000
        pos.apply_fill(OrderSide::Buy, Decimal::ONE, Decimal::from(51000))
            .unwrap();

        assert_eq!(pos.size, Decimal::from(2));
        assert_eq!(pos.entry_price, Decimal::from(50500)); // Average
    }

    #[test]
    fn test_unrealized_pnl_long() {
        let mut pos = Position::new("BTCUSD".to_string(), "bybit".to_string());

        // Buy 1 BTC at 50000
        pos.apply_fill(OrderSide::Buy, Decimal::ONE, Decimal::from(50000))
            .unwrap();

        // Mark price rises to 52000
        pos.update_mark_price(Decimal::from(52000));

        assert_eq!(pos.unrealized_pnl, Decimal::from(2000));
        assert_eq!(pos.position_value, Decimal::from(52000));
    }

    #[test]
    fn test_unrealized_pnl_short() {
        let mut pos = Position::new("BTCUSD".to_string(), "bybit".to_string());

        // Sell 1 BTC at 50000
        pos.apply_fill(OrderSide::Sell, Decimal::ONE, Decimal::from(50000))
            .unwrap();

        // Mark price falls to 48000
        pos.update_mark_price(Decimal::from(48000));

        assert_eq!(pos.unrealized_pnl, Decimal::from(2000)); // Profit on short
    }

    #[test]
    fn test_close_long_position() {
        let mut pos = Position::new("BTCUSD".to_string(), "bybit".to_string());

        // Buy 1 BTC at 50000
        pos.apply_fill(OrderSide::Buy, Decimal::ONE, Decimal::from(50000))
            .unwrap();

        // Sell 1 BTC at 52000 (close)
        pos.apply_fill(OrderSide::Sell, Decimal::ONE, Decimal::from(52000))
            .unwrap();

        assert_eq!(pos.size, Decimal::ZERO);
        assert_eq!(pos.realized_pnl, Decimal::from(2000));
        assert_eq!(pos.side, PositionSide::Flat);
    }

    #[test]
    fn test_partial_close() {
        let mut pos = Position::new("BTCUSD".to_string(), "bybit".to_string());

        // Buy 2 BTC at 50000
        pos.apply_fill(OrderSide::Buy, Decimal::from(2), Decimal::from(50000))
            .unwrap();

        // Sell 1 BTC at 52000 (partial close)
        pos.apply_fill(OrderSide::Sell, Decimal::ONE, Decimal::from(52000))
            .unwrap();

        assert_eq!(pos.size, Decimal::ONE);
        assert_eq!(pos.realized_pnl, Decimal::from(2000));
    }

    #[tokio::test]
    async fn test_position_tracker() {
        let tracker = PositionTracker::new();

        // Open position
        tracker
            .apply_fill(
                "bybit",
                "BTCUSD".to_string(),
                OrderSide::Buy,
                Decimal::ONE,
                Decimal::from(50000),
            )
            .await
            .unwrap();

        let pos = tracker.get_position("bybit", "BTCUSD".to_string()).await;
        assert_eq!(pos.size, Decimal::ONE);

        let stats = tracker.get_stats().await;
        assert_eq!(stats.total_positions, 1);
        assert_eq!(stats.long_positions, 1);
    }

    #[tokio::test]
    async fn test_multiple_positions() {
        let tracker = PositionTracker::new();

        // Open BTC long
        tracker
            .apply_fill(
                "bybit",
                "BTCUSD".to_string(),
                OrderSide::Buy,
                Decimal::ONE,
                Decimal::from(50000),
            )
            .await
            .unwrap();

        // Open ETH short
        tracker
            .apply_fill(
                "bybit",
                "ETHUSDT".to_string(),
                OrderSide::Sell,
                Decimal::from(10),
                Decimal::from(3000),
            )
            .await
            .unwrap();

        let stats = tracker.get_stats().await;
        assert_eq!(stats.total_positions, 2);
        assert_eq!(stats.long_positions, 1);
        assert_eq!(stats.short_positions, 1);
    }
}
