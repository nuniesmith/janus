//! Core types for the FKS Execution Service

use chrono::{DateTime, Utc};

use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// Re-export generated protobuf types
pub use crate::generated::fks::execution::v1::*;

/// Internal order representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Internal unique order ID
    pub id: String,

    /// Exchange order ID (once submitted)
    pub exchange_order_id: Option<String>,

    /// Client order ID (for exchange correlation)
    pub client_order_id: Option<String>,

    /// Original signal ID from JANUS
    pub signal_id: String,

    /// Trading symbol (e.g., "BTCUSD")
    pub symbol: String,

    /// Exchange name (e.g., "bybit", "binance")
    pub exchange: String,

    /// Buy or Sell
    pub side: OrderSide,

    /// Order type (Market, Limit, etc.)
    pub order_type: OrderTypeEnum,

    /// Order quantity
    pub quantity: Decimal,

    /// Filled quantity
    pub filled_quantity: Decimal,

    /// Remaining quantity
    pub remaining_quantity: Decimal,

    /// Limit price (for limit orders)
    pub price: Option<Decimal>,

    /// Stop price (for stop orders)
    pub stop_price: Option<Decimal>,

    /// Average fill price
    pub average_fill_price: Option<Decimal>,

    /// Time in force
    pub time_in_force: TimeInForceEnum,

    /// Execution strategy
    pub strategy: ExecutionStrategyEnum,

    /// Current order status
    pub status: OrderStatusEnum,

    /// List of fills for this order
    pub fills: Vec<Fill>,

    /// Order creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Order {
    /// Create a new order
    pub fn new(
        signal_id: String,
        symbol: String,
        exchange: String,
        side: OrderSide,
        order_type: OrderTypeEnum,
        quantity: Decimal,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4().to_string(),
            exchange_order_id: None,
            client_order_id: None,
            signal_id,
            symbol,
            exchange,
            side,
            order_type,
            quantity,
            filled_quantity: Decimal::ZERO,
            remaining_quantity: quantity,
            price: None,
            stop_price: None,
            average_fill_price: None,
            time_in_force: TimeInForceEnum::Gtc,
            strategy: ExecutionStrategyEnum::Immediate,
            status: OrderStatusEnum::New,
            fills: Vec::new(),
            created_at: now,
            updated_at: now,
            metadata: HashMap::new(),
        }
    }

    /// Check if order is terminal (completed/cancelled/rejected)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.status,
            OrderStatusEnum::Filled
                | OrderStatusEnum::Cancelled
                | OrderStatusEnum::Rejected
                | OrderStatusEnum::Expired
        )
    }

    /// Check if order is active (can be filled or cancelled)
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            OrderStatusEnum::New | OrderStatusEnum::Submitted | OrderStatusEnum::PartiallyFilled
        )
    }

    /// Add a fill to this order
    pub fn add_fill(&mut self, fill: Fill) {
        self.filled_quantity += fill.quantity;
        self.remaining_quantity = self.quantity - self.filled_quantity;

        // Update average fill price
        if let Some(avg_price) = self.average_fill_price {
            let total_cost = avg_price * self.filled_quantity;
            let fill_cost = fill.price * fill.quantity;
            let new_filled = self.filled_quantity;
            if new_filled > Decimal::ZERO {
                self.average_fill_price = Some((total_cost + fill_cost) / new_filled);
            }
        } else {
            self.average_fill_price = Some(fill.price);
        }

        // Update status
        if self.remaining_quantity <= Decimal::ZERO {
            self.status = OrderStatusEnum::Filled;
        } else {
            self.status = OrderStatusEnum::PartiallyFilled;
        }

        self.fills.push(fill);
        self.updated_at = Utc::now();
    }
}

/// Fill/Trade representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// Unique fill ID
    pub id: String,

    /// Associated order ID
    pub order_id: String,

    /// Filled quantity
    pub quantity: Decimal,

    /// Fill price
    pub price: Decimal,

    /// Fee amount
    pub fee: Decimal,

    /// Fee currency
    pub fee_currency: String,

    /// Side (Buy/Sell)
    pub side: OrderSide,

    /// Fill timestamp
    pub timestamp: DateTime<Utc>,

    /// Is this a maker fill (vs taker)?
    pub is_maker: bool,
}

/// Position representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Trading symbol
    pub symbol: String,

    /// Exchange name
    pub exchange: String,

    /// Position side (Long/Short)
    pub side: PositionSide,

    /// Position quantity
    pub quantity: Decimal,

    /// Average entry price
    pub average_entry_price: Decimal,

    /// Current market price
    pub current_price: Decimal,

    /// Unrealized P&L
    pub unrealized_pnl: Decimal,

    /// Realized P&L (from closed portions)
    pub realized_pnl: Decimal,

    /// Margin used for this position
    pub margin_used: Decimal,

    /// Liquidation price
    pub liquidation_price: Option<Decimal>,

    /// Position opened timestamp
    pub opened_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

impl Position {
    /// Calculate unrealized P&L based on current price
    pub fn calculate_unrealized_pnl(&mut self) {
        let pnl = match self.side {
            PositionSide::Long => (self.current_price - self.average_entry_price) * self.quantity,
            PositionSide::Short => (self.average_entry_price - self.current_price) * self.quantity,
        };

        self.unrealized_pnl = pnl;
    }

    /// Update position with a fill
    pub fn update_with_fill(&mut self, fill: &Fill) {
        let fill_qty = fill.quantity;

        match (&self.side, &fill.side) {
            // Adding to position
            (PositionSide::Long, OrderSide::Buy) | (PositionSide::Short, OrderSide::Sell) => {
                let current_cost = self.average_entry_price * self.quantity;
                let fill_cost = fill.price * fill_qty;
                let new_qty = self.quantity + fill_qty;

                if new_qty > Decimal::ZERO {
                    self.average_entry_price = (current_cost + fill_cost) / new_qty;
                    self.quantity = new_qty;
                }
            }
            // Reducing position
            (PositionSide::Long, OrderSide::Sell) | (PositionSide::Short, OrderSide::Buy) => {
                let realized = self.calculate_realized_pnl(fill);
                self.realized_pnl += realized;
                self.quantity -= fill_qty;

                // If position is closed or flipped
                if self.quantity <= Decimal::ZERO {
                    let remaining = -self.quantity;
                    if remaining > Decimal::ZERO {
                        // Position flipped
                        self.side = if self.side == PositionSide::Long {
                            PositionSide::Short
                        } else {
                            PositionSide::Long
                        };
                        self.quantity = remaining;
                        self.average_entry_price = fill.price;
                    }
                }
            }
        }

        self.updated_at = Utc::now();
        self.calculate_unrealized_pnl();
    }

    /// Calculate realized P&L from a closing fill
    fn calculate_realized_pnl(&self, fill: &Fill) -> Decimal {
        match self.side {
            PositionSide::Long => (fill.price - self.average_entry_price) * fill.quantity,
            PositionSide::Short => (self.average_entry_price - fill.price) * fill.quantity,
        }
    }
}

/// Account balance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    /// Total account balance
    pub balance: Decimal,

    /// Available balance for trading
    pub available_balance: Decimal,

    /// Margin used
    pub margin_used: Decimal,

    /// Total unrealized P&L
    pub unrealized_pnl: Decimal,

    /// Total realized P&L (session)
    pub realized_pnl: Decimal,

    /// Per-exchange balances
    pub balances_by_exchange: HashMap<String, Decimal>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// Enums
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "BUY"),
            OrderSide::Sell => write!(f, "SELL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
}

impl std::fmt::Display for PositionSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PositionSide::Long => write!(f, "LONG"),
            PositionSide::Short => write!(f, "SHORT"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderTypeEnum {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    TrailingStop,
    /// Stop loss (market) - Binance
    StopLoss,
    /// Stop loss limit - Binance
    StopLossLimit,
    /// Take profit (market) - Binance
    TakeProfit,
    /// Take profit limit - Binance
    TakeProfitLimit,
    /// Limit maker (post-only) - Binance
    LimitMaker,
}

impl std::fmt::Display for OrderTypeEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderTypeEnum::Market => write!(f, "MARKET"),
            OrderTypeEnum::Limit => write!(f, "LIMIT"),
            OrderTypeEnum::StopMarket => write!(f, "STOP_MARKET"),
            OrderTypeEnum::StopLimit => write!(f, "STOP_LIMIT"),
            OrderTypeEnum::TrailingStop => write!(f, "TRAILING_STOP"),
            OrderTypeEnum::StopLoss => write!(f, "STOP_LOSS"),
            OrderTypeEnum::StopLossLimit => write!(f, "STOP_LOSS_LIMIT"),
            OrderTypeEnum::TakeProfit => write!(f, "TAKE_PROFIT"),
            OrderTypeEnum::TakeProfitLimit => write!(f, "TAKE_PROFIT_LIMIT"),
            OrderTypeEnum::LimitMaker => write!(f, "LIMIT_MAKER"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatusEnum {
    New,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
    PendingCancel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeInForceEnum {
    Gtc, // Good Till Cancel
    Ioc, // Immediate or Cancel
    Fok, // Fill or Kill
    Gtd, // Good Till Date
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategyEnum {
    Immediate,
    Twap,
    Vwap,
    AlmgrenChriss,
    Iceberg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionMode {
    Simulated,
    Paper,
    Live,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::new(
            "sig123".to_string(),
            "BTCUSD".to_string(),
            "bybit".to_string(),
            OrderSide::Buy,
            OrderTypeEnum::Market,
            Decimal::from(1),
        );

        assert_eq!(order.status, OrderStatusEnum::New);
        assert_eq!(order.filled_quantity, Decimal::ZERO);
        assert_eq!(order.remaining_quantity, Decimal::from(1));
    }

    #[test]
    fn test_order_is_terminal() {
        let mut order = Order::new(
            "sig123".to_string(),
            "BTCUSD".to_string(),
            "bybit".to_string(),
            OrderSide::Buy,
            OrderTypeEnum::Market,
            Decimal::from(1),
        );

        assert!(!order.is_terminal());
        order.status = OrderStatusEnum::Filled;
        assert!(order.is_terminal());
    }

    #[test]
    fn test_position_pnl_long() {
        let mut pos = Position {
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: PositionSide::Long,
            quantity: Decimal::from(1),
            average_entry_price: Decimal::from(50000),
            current_price: Decimal::from(51000),
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            margin_used: Decimal::from(5000),
            liquidation_price: None,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
        };

        pos.calculate_unrealized_pnl();
        assert_eq!(pos.unrealized_pnl, Decimal::from(1000));
    }

    #[test]
    fn test_position_pnl_short() {
        let mut pos = Position {
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: PositionSide::Short,
            quantity: Decimal::from(1),
            average_entry_price: Decimal::from(50000),
            current_price: Decimal::from(49000),
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            margin_used: Decimal::from(5000),
            liquidation_price: None,
            opened_at: Utc::now(),
            updated_at: Utc::now(),
        };

        pos.calculate_unrealized_pnl();
        assert_eq!(pos.unrealized_pnl, Decimal::from(1000));
    }
}
