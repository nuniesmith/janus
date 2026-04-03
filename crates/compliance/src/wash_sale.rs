//! # Wash Sale Detection Module
//!
//! Implements IRS wash sale rule detection and tracking for tax compliance.
//!
//! ## IRS Wash Sale Rule (IRC Section 1091)
//!
//! A wash sale occurs when you sell a security at a loss and purchase a
//! "substantially identical" security within 30 days before or after the sale.
//! The loss is disallowed for tax purposes and added to the cost basis of the
//! replacement security.
//!
//! ## Rule Details
//!
//! - **Window:** 61 days total (30 days before + sale day + 30 days after)
//! - **Substantially Identical:** Same or substantially similar security
//! - **Disallowed Loss:** Added to replacement security's cost basis
//! - **Holding Period:** Inherited from original security

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;

/// Number of days before the sale to check for replacement purchases
const LOOKBACK_DAYS: i64 = 30;

/// Number of days after the sale to check for replacement purchases
const LOOKFORWARD_DAYS: i64 = 30;

/// Wash sale detection errors
#[derive(Error, Debug)]
pub enum WashSaleError {
    #[error("Invalid trade data: {0}")]
    InvalidTrade(String),

    #[error("Symbol mismatch: expected {expected}, got {actual}")]
    SymbolMismatch { expected: String, actual: String },

    #[error("Calculation error: {0}")]
    CalculationError(String),

    #[error("No trades found for symbol: {0}")]
    NoTradesFound(String),
}

/// Result type for wash sale operations
pub type Result<T> = std::result::Result<T, WashSaleError>;

/// Trade action type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeAction {
    /// Buy/Long position
    Buy,
    /// Sell/Close position
    Sell,
}

/// Individual trade record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Unique trade identifier
    pub id: String,

    /// Trading symbol
    pub symbol: String,

    /// Trade action (Buy/Sell)
    pub action: TradeAction,

    /// Number of shares/contracts
    pub quantity: f64,

    /// Price per share/contract
    pub price: f64,

    /// Trade execution timestamp
    pub timestamp: DateTime<Utc>,

    /// Commission and fees
    pub commission: f64,

    /// Whether this trade is part of a wash sale
    pub is_wash_sale: bool,

    /// Reference to the wash sale this trade is involved in
    pub wash_sale_id: Option<String>,
}

impl Trade {
    /// Calculate total trade value (price * quantity + commission)
    pub fn total_value(&self) -> f64 {
        (self.price * self.quantity.abs()) + self.commission
    }

    /// Calculate net proceeds for a sale (negative for cost)
    pub fn net_proceeds(&self) -> f64 {
        match self.action {
            TradeAction::Sell => (self.price * self.quantity.abs()) - self.commission,
            TradeAction::Buy => -((self.price * self.quantity.abs()) + self.commission),
        }
    }

    /// Check if this is a loss-producing sale
    pub fn is_loss_sale(&self, cost_basis: f64) -> bool {
        matches!(self.action, TradeAction::Sell) && self.net_proceeds() < cost_basis
    }
}

/// Wash sale detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WashSaleResult {
    /// Unique identifier for this wash sale
    pub id: String,

    /// Symbol involved
    pub symbol: String,

    /// The sale trade that triggered the wash sale
    pub sale_trade_id: String,

    /// Replacement purchase trades
    pub replacement_trade_ids: Vec<String>,

    /// Original loss amount (disallowed)
    pub disallowed_loss: f64,

    /// Adjusted cost basis for replacement security
    pub adjusted_basis: f64,

    /// Quantity affected by wash sale
    pub affected_quantity: f64,

    /// Date of the loss sale
    pub sale_date: DateTime<Utc>,

    /// Earliest replacement purchase date
    pub earliest_replacement: DateTime<Utc>,

    /// Latest replacement purchase date
    pub latest_replacement: DateTime<Utc>,
}

/// Position tracking for wash sale calculations
#[derive(Debug, Clone)]
struct Position {
    /// Symbol (reserved for future use in detailed tracking)
    #[allow(dead_code)]
    symbol: String,
    total_quantity: f64,
    total_cost: f64,
    trades: Vec<Trade>,
}

impl Position {
    fn new(symbol: String) -> Self {
        Self {
            symbol,
            total_quantity: 0.0,
            total_cost: 0.0,
            trades: Vec::new(),
        }
    }

    fn average_cost_basis(&self) -> f64 {
        if self.total_quantity > 0.0 {
            self.total_cost / self.total_quantity
        } else {
            0.0
        }
    }

    fn add_trade(&mut self, trade: &Trade) {
        match trade.action {
            TradeAction::Buy => {
                self.total_cost += trade.total_value();
                self.total_quantity += trade.quantity;
            }
            TradeAction::Sell => {
                let basis = self.average_cost_basis() * trade.quantity;
                self.total_cost -= basis;
                self.total_quantity -= trade.quantity;
            }
        }
        self.trades.push(trade.clone());
    }
}

/// Wash sale detector
pub struct WashSaleDetector {
    /// Trade history by symbol
    trade_history: HashMap<String, VecDeque<Trade>>,

    /// Detected wash sales
    wash_sales: Vec<WashSaleResult>,

    /// Current positions
    positions: HashMap<String, Position>,

    /// Enable strict mode (more conservative detection)
    strict_mode: bool,

    /// Maximum history to maintain (in days)
    max_history_days: i64,
}

impl Default for WashSaleDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl WashSaleDetector {
    /// Create a new wash sale detector
    pub fn new() -> Self {
        Self {
            trade_history: HashMap::new(),
            wash_sales: Vec::new(),
            positions: HashMap::new(),
            strict_mode: false,
            max_history_days: 365, // Keep 1 year of history
        }
    }

    /// Create a detector with strict mode enabled
    pub fn with_strict_mode(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Set maximum history retention period
    pub fn with_max_history_days(mut self, days: i64) -> Self {
        self.max_history_days = days;
        self
    }

    /// Add a trade to the detector
    pub fn add_trade(&mut self, trade: Trade) -> Result<()> {
        // Validate trade
        if trade.quantity <= 0.0 {
            return Err(WashSaleError::InvalidTrade(
                "Quantity must be positive".to_string(),
            ));
        }

        if trade.price <= 0.0 {
            return Err(WashSaleError::InvalidTrade(
                "Price must be positive".to_string(),
            ));
        }

        // Update position
        let position = self
            .positions
            .entry(trade.symbol.clone())
            .or_insert_with(|| Position::new(trade.symbol.clone()));
        position.add_trade(&trade);

        // Add to history
        let history = self.trade_history.entry(trade.symbol.clone()).or_default();
        history.push_back(trade.clone());

        // Check for wash sale if this is a sale
        if matches!(trade.action, TradeAction::Sell) {
            self.check_wash_sale(&trade)?;
        }

        // Also check if this buy triggers a wash sale on a recent loss sale
        if matches!(trade.action, TradeAction::Buy) {
            self.check_wash_sale_on_buy(&trade)?;
        }

        // Clean up old trades
        self.cleanup_old_trades();

        Ok(())
    }

    /// Check if a sale triggers a wash sale
    fn check_wash_sale(&mut self, sale: &Trade) -> Result<()> {
        let history = self
            .trade_history
            .get(&sale.symbol)
            .ok_or_else(|| WashSaleError::NoTradesFound(sale.symbol.clone()))?;

        let position = self
            .positions
            .get(&sale.symbol)
            .ok_or_else(|| WashSaleError::NoTradesFound(sale.symbol.clone()))?;

        // Calculate cost basis for this sale
        let cost_basis = position.average_cost_basis() * sale.quantity;

        // Check if it's a loss
        if !sale.is_loss_sale(cost_basis) {
            return Ok(()); // Not a loss, no wash sale possible
        }

        let loss = cost_basis - sale.net_proceeds();

        // Find replacement purchases within wash sale window
        let window_start = sale.timestamp - Duration::days(LOOKBACK_DAYS);
        let window_end = sale.timestamp + Duration::days(LOOKFORWARD_DAYS);

        let replacements: Vec<Trade> = history
            .iter()
            .filter(|t| {
                // Must be a buy
                matches!(t.action, TradeAction::Buy)
                    // Within the wash sale window
                    && t.timestamp >= window_start
                    && t.timestamp <= window_end
                    // Not the same trade
                    && t.id != sale.id
                    // Exclude sales on the same day if strict mode
                    && (!self.strict_mode || t.timestamp.date_naive() != sale.timestamp.date_naive())
            })
            .cloned()
            .collect();

        if replacements.is_empty() {
            return Ok(()); // No replacement purchases, not a wash sale
        }

        // Calculate affected quantity (minimum of sale quantity and total replacement quantity)
        let replacement_quantity: f64 = replacements.iter().map(|t| t.quantity).sum();
        let affected_quantity = sale.quantity.min(replacement_quantity);

        // Calculate disallowed loss (proportional to affected quantity)
        let disallowed_loss = loss * (affected_quantity / sale.quantity);

        // Calculate adjusted basis for replacement securities
        let total_replacement_cost: f64 = replacements.iter().map(|t| t.total_value()).sum();
        let adjusted_basis = total_replacement_cost + disallowed_loss;

        // Find earliest and latest replacement
        let earliest = replacements
            .iter()
            .map(|t| t.timestamp)
            .min()
            .unwrap_or(sale.timestamp);
        let latest = replacements
            .iter()
            .map(|t| t.timestamp)
            .max()
            .unwrap_or(sale.timestamp);

        // Create wash sale result
        let wash_sale = WashSaleResult {
            id: format!("WS-{}-{}", sale.symbol, sale.timestamp.timestamp()),
            symbol: sale.symbol.clone(),
            sale_trade_id: sale.id.clone(),
            replacement_trade_ids: replacements.iter().map(|t| t.id.clone()).collect(),
            disallowed_loss,
            adjusted_basis,
            affected_quantity,
            sale_date: sale.timestamp,
            earliest_replacement: earliest,
            latest_replacement: latest,
        };

        self.wash_sales.push(wash_sale);

        Ok(())
    }

    /// Check if a buy triggers a wash sale on a recent loss sale
    fn check_wash_sale_on_buy(&mut self, buy: &Trade) -> Result<()> {
        let history = match self.trade_history.get(&buy.symbol) {
            Some(h) => h,
            None => return Ok(()), // No history, nothing to check
        };

        // Find recent loss sales that could be affected by this buy
        let window_start = buy.timestamp - Duration::days(LOOKFORWARD_DAYS);
        let window_end = buy.timestamp + Duration::days(LOOKBACK_DAYS);

        // Find buys that occurred before each potential loss sale to calculate cost basis
        let buys_before_sale: Vec<&Trade> = history
            .iter()
            .filter(|t| matches!(t.action, TradeAction::Buy) && t.id != buy.id)
            .collect();

        // Find potential loss sales
        let loss_sales: Vec<(Trade, f64)> = history
            .iter()
            .filter(|t| {
                // Must be a sell
                matches!(t.action, TradeAction::Sell)
                    // Within the wash sale window
                    && t.timestamp >= window_start
                    && t.timestamp <= window_end
                    // Not already detected as wash sale
                    && !self.wash_sales.iter().any(|ws| ws.sale_trade_id == t.id)
            })
            .filter_map(|sale| {
                // Calculate cost basis at time of sale from prior buys
                let prior_buys: Vec<&&Trade> = buys_before_sale
                    .iter()
                    .filter(|b| b.timestamp < sale.timestamp)
                    .collect();

                if prior_buys.is_empty() {
                    return None;
                }

                // Calculate average cost basis from prior buys
                let total_cost: f64 = prior_buys.iter().map(|b| b.total_value()).sum();
                let total_qty: f64 = prior_buys.iter().map(|b| b.quantity).sum();
                let avg_cost_basis = if total_qty > 0.0 {
                    total_cost / total_qty
                } else {
                    return None;
                };

                let cost_basis = avg_cost_basis * sale.quantity;

                // Check if it's a loss
                if sale.is_loss_sale(cost_basis) {
                    Some((sale.clone(), cost_basis))
                } else {
                    None
                }
            })
            .collect();

        for (sale, cost_basis) in loss_sales {
            // Calculate loss
            let loss = cost_basis - sale.net_proceeds();

            // Calculate affected quantity
            let affected_quantity = sale.quantity.min(buy.quantity);

            // Calculate disallowed loss (proportional to affected quantity)
            let disallowed_loss = loss * (affected_quantity / sale.quantity);

            // Calculate adjusted basis
            let adjusted_basis = buy.total_value() + disallowed_loss;

            // Create wash sale result
            let wash_sale = WashSaleResult {
                id: format!("WS-{}-{}", sale.symbol, sale.timestamp.timestamp()),
                symbol: sale.symbol.clone(),
                sale_trade_id: sale.id.clone(),
                replacement_trade_ids: vec![buy.id.clone()],
                disallowed_loss,
                adjusted_basis,
                affected_quantity,
                sale_date: sale.timestamp,
                earliest_replacement: buy.timestamp,
                latest_replacement: buy.timestamp,
            };

            self.wash_sales.push(wash_sale);
        }

        Ok(())
    }

    /// Get all detected wash sales
    pub fn get_wash_sales(&self) -> &[WashSaleResult] {
        &self.wash_sales
    }

    /// Get wash sales for a specific symbol
    pub fn get_wash_sales_for_symbol(&self, symbol: &str) -> Vec<&WashSaleResult> {
        self.wash_sales
            .iter()
            .filter(|ws| ws.symbol == symbol)
            .collect()
    }

    /// Get total disallowed losses
    pub fn total_disallowed_losses(&self) -> f64 {
        self.wash_sales.iter().map(|ws| ws.disallowed_loss).sum()
    }

    /// Get total disallowed losses for a symbol
    pub fn total_disallowed_losses_for_symbol(&self, symbol: &str) -> f64 {
        self.wash_sales
            .iter()
            .filter(|ws| ws.symbol == symbol)
            .map(|ws| ws.disallowed_loss)
            .sum()
    }

    /// Check if a proposed sale would trigger a wash sale
    pub fn would_trigger_wash_sale(
        &self,
        symbol: &str,
        quantity: f64,
        price: f64,
        proposed_date: DateTime<Utc>,
    ) -> Result<Option<f64>> {
        let history = self
            .trade_history
            .get(symbol)
            .ok_or_else(|| WashSaleError::NoTradesFound(symbol.to_string()))?;

        let position = self
            .positions
            .get(symbol)
            .ok_or_else(|| WashSaleError::NoTradesFound(symbol.to_string()))?;

        // Calculate potential cost basis
        let cost_basis = position.average_cost_basis() * quantity;
        let net_proceeds = (price * quantity) - 0.0; // Assume no commission for check

        // Check if it would be a loss
        if net_proceeds >= cost_basis {
            return Ok(None); // Not a loss
        }

        let loss = cost_basis - net_proceeds;

        // Check for replacement purchases
        let window_start = proposed_date - Duration::days(LOOKBACK_DAYS);
        let window_end = proposed_date + Duration::days(LOOKFORWARD_DAYS);

        let has_replacements = history.iter().any(|t| {
            matches!(t.action, TradeAction::Buy)
                && t.timestamp >= window_start
                && t.timestamp <= window_end
        });

        if has_replacements {
            Ok(Some(loss))
        } else {
            Ok(None)
        }
    }

    /// Clean up old trade history
    fn cleanup_old_trades(&mut self) {
        let cutoff = Utc::now() - Duration::days(self.max_history_days);

        for history in self.trade_history.values_mut() {
            while let Some(trade) = history.front() {
                if trade.timestamp < cutoff {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.trade_history.clear();
        self.wash_sales.clear();
        self.positions.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> WashSaleStats {
        let total_symbols = self.trade_history.len();
        let total_trades: usize = self.trade_history.values().map(|h| h.len()).sum();
        let total_wash_sales = self.wash_sales.len();
        let total_disallowed = self.total_disallowed_losses();

        WashSaleStats {
            total_symbols,
            total_trades,
            total_wash_sales,
            total_disallowed_losses: total_disallowed,
        }
    }
}

/// Wash sale detection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WashSaleStats {
    pub total_symbols: usize,
    pub total_trades: usize,
    pub total_wash_sales: usize,
    pub total_disallowed_losses: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_trade(
        id: &str,
        symbol: &str,
        action: TradeAction,
        quantity: f64,
        price: f64,
        days_offset: i64,
    ) -> Trade {
        Trade {
            id: id.to_string(),
            symbol: symbol.to_string(),
            action,
            quantity,
            price,
            timestamp: Utc::now() + Duration::days(days_offset),
            commission: 0.0,
            is_wash_sale: false,
            wash_sale_id: None,
        }
    }

    #[test]
    fn test_simple_wash_sale() {
        let mut detector = WashSaleDetector::new();

        // Buy 100 shares at $50
        let buy = create_trade("T1", "AAPL", TradeAction::Buy, 100.0, 50.0, -60);
        detector.add_trade(buy).unwrap();

        // Sell at a loss ($40)
        let sell = create_trade("T2", "AAPL", TradeAction::Sell, 100.0, 40.0, -30);
        detector.add_trade(sell.clone()).unwrap();

        // Repurchase within 30 days
        let repurchase = create_trade("T3", "AAPL", TradeAction::Buy, 100.0, 42.0, -15);
        detector.add_trade(repurchase).unwrap();

        // Should detect wash sale
        let wash_sales = detector.get_wash_sales();
        assert_eq!(wash_sales.len(), 1);

        let ws = &wash_sales[0];
        assert_eq!(ws.symbol, "AAPL");
        assert_eq!(ws.sale_trade_id, "T2");
        assert!(ws.disallowed_loss > 0.0);
    }

    #[test]
    fn test_no_wash_sale_on_gain() {
        let mut detector = WashSaleDetector::new();

        // Buy at $50
        let buy = create_trade("T1", "TSLA", TradeAction::Buy, 100.0, 50.0, -60);
        detector.add_trade(buy).unwrap();

        // Sell at a gain ($60)
        let sell = create_trade("T2", "TSLA", TradeAction::Sell, 100.0, 60.0, -30);
        detector.add_trade(sell).unwrap();

        // Repurchase
        let repurchase = create_trade("T3", "TSLA", TradeAction::Buy, 100.0, 62.0, -15);
        detector.add_trade(repurchase).unwrap();

        // No wash sale (not a loss)
        assert_eq!(detector.get_wash_sales().len(), 0);
    }

    #[test]
    fn test_no_wash_sale_outside_window() {
        let mut detector = WashSaleDetector::new();

        // Buy at $50
        let buy = create_trade("T1", "MSFT", TradeAction::Buy, 100.0, 50.0, -100);
        detector.add_trade(buy).unwrap();

        // Sell at a loss
        let sell = create_trade("T2", "MSFT", TradeAction::Sell, 100.0, 40.0, -60);
        detector.add_trade(sell).unwrap();

        // Repurchase OUTSIDE 30-day window (35 days later)
        let repurchase = create_trade("T3", "MSFT", TradeAction::Buy, 100.0, 42.0, -25);
        detector.add_trade(repurchase).unwrap();

        // No wash sale (outside window)
        assert_eq!(detector.get_wash_sales().len(), 0);
    }

    #[test]
    fn test_partial_wash_sale() {
        let mut detector = WashSaleDetector::new();

        // Buy 100 shares at $50
        let buy = create_trade("T1", "GOOG", TradeAction::Buy, 100.0, 50.0, -60);
        detector.add_trade(buy).unwrap();

        // Sell 100 shares at a loss
        let sell = create_trade("T2", "GOOG", TradeAction::Sell, 100.0, 40.0, -30);
        detector.add_trade(sell).unwrap();

        // Only repurchase 50 shares
        let repurchase = create_trade("T3", "GOOG", TradeAction::Buy, 50.0, 42.0, -15);
        detector.add_trade(repurchase).unwrap();

        let wash_sales = detector.get_wash_sales();
        assert_eq!(wash_sales.len(), 1);

        // Only 50 shares should be affected
        assert_eq!(wash_sales[0].affected_quantity, 50.0);
    }

    #[test]
    fn test_total_disallowed_losses() {
        let mut detector = WashSaleDetector::new();

        // Create multiple wash sales
        let buy1 = create_trade("T1", "AAPL", TradeAction::Buy, 100.0, 50.0, -60);
        detector.add_trade(buy1).unwrap();

        let sell1 = create_trade("T2", "AAPL", TradeAction::Sell, 100.0, 40.0, -30);
        detector.add_trade(sell1).unwrap();

        let repurchase1 = create_trade("T3", "AAPL", TradeAction::Buy, 100.0, 42.0, -15);
        detector.add_trade(repurchase1).unwrap();

        let total = detector.total_disallowed_losses();
        assert!(total > 0.0);
    }

    #[test]
    fn test_would_trigger_wash_sale() {
        let mut detector = WashSaleDetector::new();

        // Buy at $50
        let buy = create_trade("T1", "NVDA", TradeAction::Buy, 100.0, 50.0, -20);
        detector.add_trade(buy).unwrap();

        // Check if selling at a loss would trigger wash sale
        let proposed_date = Utc::now() - Duration::days(10);
        let result = detector
            .would_trigger_wash_sale("NVDA", 100.0, 40.0, proposed_date)
            .unwrap();

        assert!(result.is_some()); // Would trigger wash sale
    }

    #[test]
    fn test_stats() {
        let mut detector = WashSaleDetector::new();

        let buy = create_trade("T1", "META", TradeAction::Buy, 100.0, 50.0, -60);
        detector.add_trade(buy).unwrap();

        let sell = create_trade("T2", "META", TradeAction::Sell, 100.0, 40.0, -30);
        detector.add_trade(sell).unwrap();

        let repurchase = create_trade("T3", "META", TradeAction::Buy, 100.0, 42.0, -15);
        detector.add_trade(repurchase).unwrap();

        let stats = detector.stats();
        assert_eq!(stats.total_symbols, 1);
        assert_eq!(stats.total_trades, 3);
        assert_eq!(stats.total_wash_sales, 1);
        assert!(stats.total_disallowed_losses > 0.0);
    }
}
