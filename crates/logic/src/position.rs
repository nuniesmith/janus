//! Position tracking and sizing logic.
//!
//! Provides position tracking and risk-based position sizing calculations.

use common::{JanusError, OrderSide, Price, Result, Volume};
use std::collections::HashMap;

/// Position information for a symbol
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: Volume,
    pub average_entry_price: Price,
    pub current_price: Option<Price>,
    pub side: PositionSide,
}

/// Position side (Long or Short)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionSide {
    Long,
    Short,
}

impl From<OrderSide> for PositionSide {
    fn from(side: OrderSide) -> Self {
        match side {
            OrderSide::Buy => PositionSide::Long,
            OrderSide::Sell => PositionSide::Short,
        }
    }
}

impl Position {
    /// Create a new position
    pub fn new(symbol: String, quantity: Volume, entry_price: Price, side: PositionSide) -> Self {
        Self {
            symbol,
            quantity,
            average_entry_price: entry_price,
            current_price: None,
            side,
        }
    }

    /// Update current price
    pub fn update_price(&mut self, price: Price) {
        self.current_price = Some(price);
    }

    /// Get unrealized PnL
    pub fn unrealized_pnl(&self) -> Option<Price> {
        self.current_price.map(|current| {
            let entry = self.average_entry_price.value();
            let current_val = current.value();
            let quantity = self.quantity.value();

            let pnl = match self.side {
                PositionSide::Long => (current_val - entry) * quantity,
                PositionSide::Short => (entry - current_val) * quantity,
            };

            Price(pnl)
        })
    }

    /// Get position value (notional)
    pub fn notional_value(&self) -> Price {
        let price = self.current_price.unwrap_or(self.average_entry_price);
        Price(self.quantity.value() * price.value())
    }

    /// Add to position (average entry price calculation)
    pub fn add(&mut self, quantity: Volume, price: Price) {
        let total_quantity = self.quantity.value() + quantity.value();
        let total_value = (self.quantity.value() * self.average_entry_price.value())
            + (quantity.value() * price.value());

        self.quantity = Volume(total_quantity);
        self.average_entry_price = Price(total_value / total_quantity);
    }

    /// Reduce position
    pub fn reduce(&mut self, quantity: Volume) -> Result<()> {
        if quantity.value() > self.quantity.value() {
            return Err(JanusError::Internal(format!(
                "Cannot reduce position by {} when position is {}",
                quantity.value(),
                self.quantity.value()
            )));
        }

        self.quantity = Volume(self.quantity.value() - quantity.value());
        Ok(())
    }

    /// Check if position is flat (zero quantity)
    pub fn is_flat(&self) -> bool {
        self.quantity.value().abs() < f64::EPSILON
    }
}

/// Position tracker that maintains positions across symbols
#[derive(Debug, Clone)]
pub struct PositionTracker {
    positions: HashMap<String, Position>,
}

impl PositionTracker {
    /// Create a new position tracker
    pub fn new() -> Self {
        Self {
            positions: HashMap::new(),
        }
    }

    /// Get position for a symbol
    pub fn get_position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get position for a symbol (mutable)
    pub fn get_position_mut(&mut self, symbol: &str) -> Option<&mut Position> {
        self.positions.get_mut(symbol)
    }

    /// Update position from a trade
    pub fn update_from_trade(
        &mut self,
        symbol: &str,
        side: OrderSide,
        quantity: Volume,
        price: Price,
    ) -> Result<()> {
        let position_side = PositionSide::from(side);

        if let Some(position) = self.positions.get_mut(symbol) {
            // Existing position
            if position.side == position_side {
                // Same direction - add to position
                position.add(quantity, price);
            } else {
                // Opposite direction - reduce or flip position
                if quantity.value() >= position.quantity.value() {
                    // Flip position
                    let excess = quantity.value() - position.quantity.value();
                    position.quantity = Volume(excess);
                    position.average_entry_price = price;
                    position.side = position_side;
                } else {
                    // Reduce position
                    position.reduce(quantity)?;
                }
            }
        } else {
            // New position
            let position = Position::new(symbol.to_string(), quantity, price, position_side);
            self.positions.insert(symbol.to_string(), position);
        }

        Ok(())
    }

    /// Update price for a symbol
    pub fn update_price(&mut self, symbol: &str, price: Price) {
        if let Some(position) = self.positions.get_mut(symbol) {
            position.update_price(price);
        }
    }

    /// Get total unrealized PnL across all positions
    pub fn total_unrealized_pnl(&self) -> Price {
        let total: f64 = self
            .positions
            .values()
            .filter_map(|p| p.unrealized_pnl())
            .map(|p| p.value())
            .sum();
        Price(total)
    }

    /// Get count of open positions
    pub fn open_positions_count(&self) -> usize {
        self.positions.values().filter(|p| !p.is_flat()).count()
    }

    /// Get all positions
    pub fn all_positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Remove flat positions
    pub fn cleanup_flat_positions(&mut self) {
        self.positions.retain(|_, p| !p.is_flat());
    }
}

impl Default for PositionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Position sizer that calculates position sizes based on risk
pub struct PositionSizer {
    portfolio_value: f64,
    default_risk_pct: f64,
}

impl PositionSizer {
    /// Create a new position sizer
    pub fn new(portfolio_value: f64, default_risk_pct: f64) -> Self {
        Self {
            portfolio_value,
            default_risk_pct: default_risk_pct.clamp(0.01, 0.02), // 1-2%
        }
    }

    /// Calculate position size based on risk percentage
    ///
    /// Args:
    /// - entry_price: Entry price for the trade
    /// - stop_loss: Stop loss price
    /// - risk_pct: Risk percentage (0.01 = 1%, 0.02 = 2%)
    ///
    /// Returns:
    /// - Position size in units (Volume)
    /// - Position size in USD (Price)
    /// - Actual risk amount (Price)
    pub fn calculate_position_size(
        &self,
        entry_price: Price,
        stop_loss: Price,
        risk_pct: Option<f64>,
    ) -> Result<(Volume, Price, Price)> {
        let risk_pct = risk_pct.unwrap_or(self.default_risk_pct);

        if risk_pct < 0.005 || risk_pct > 0.02 {
            return Err(JanusError::RiskViolation(format!(
                "Risk percentage {} outside 0.5%-2% range",
                risk_pct * 100.0
            )));
        }

        let entry = entry_price.value();
        let stop = stop_loss.value();

        if entry <= 0.0 || stop <= 0.0 {
            return Err(JanusError::Internal(
                "Invalid entry or stop loss price".to_string(),
            ));
        }

        // Risk per unit
        let risk_per_unit = (entry - stop).abs();
        if risk_per_unit <= 0.0 {
            return Err(JanusError::Internal(
                "Entry and stop loss prices are too close".to_string(),
            ));
        }

        // Calculate risk amount and position size
        let risk_amount = self.portfolio_value * risk_pct;
        let position_size_units = risk_amount / risk_per_unit;
        let position_size_usd = position_size_units * entry;

        Ok((
            Volume(position_size_units),
            Price(position_size_usd),
            Price(risk_amount),
        ))
    }

    /// Calculate position size with additional constraints
    pub fn calculate_position_size_with_limits(
        &self,
        entry_price: Price,
        stop_loss: Price,
        risk_pct: Option<f64>,
        max_position_pct: Option<f64>,
        max_position_usd: Option<f64>,
    ) -> Result<(Volume, Price, Price)> {
        let (_units, usd, risk) = self.calculate_position_size(entry_price, stop_loss, risk_pct)?;

        // Apply max position percentage limit
        let max_usd_by_pct = max_position_pct
            .map(|pct| self.portfolio_value * pct)
            .unwrap_or(f64::INFINITY);

        // Apply max position USD limit
        let max_usd = max_position_usd
            .unwrap_or(f64::INFINITY)
            .min(max_usd_by_pct);

        let final_usd = usd.value().min(max_usd);
        let final_units = final_usd / entry_price.value();

        Ok((Volume(final_units), Price(final_usd), risk))
    }

    /// Update portfolio value
    pub fn update_portfolio_value(&mut self, value: f64) {
        self.portfolio_value = value;
    }

    /// Get current portfolio value
    pub fn portfolio_value(&self) -> f64 {
        self.portfolio_value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_creation() {
        let position = Position::new(
            "BTC/USD".to_string(),
            Volume(1.0),
            Price(50000.0),
            PositionSide::Long,
        );

        assert_eq!(position.symbol, "BTC/USD");
        assert_eq!(position.quantity.value(), 1.0);
        assert!(!position.is_flat());
    }

    #[test]
    fn test_position_pnl() {
        let mut position = Position::new(
            "BTC/USD".to_string(),
            Volume(1.0),
            Price(50000.0),
            PositionSide::Long,
        );

        // No PnL without current price
        assert!(position.unrealized_pnl().is_none());

        // Update price and check PnL
        position.update_price(Price(51000.0));
        let pnl = position.unrealized_pnl().unwrap();
        assert_eq!(pnl.value(), 1000.0); // $1000 profit
    }

    #[test]
    fn test_position_tracker() {
        let mut tracker = PositionTracker::new();

        // Add a trade
        tracker
            .update_from_trade("BTC/USD", OrderSide::Buy, Volume(1.0), Price(50000.0))
            .unwrap();

        let position = tracker.get_position("BTC/USD").unwrap();
        assert_eq!(position.quantity.value(), 1.0);
        assert_eq!(tracker.open_positions_count(), 1);
    }

    #[test]
    fn test_position_sizer() {
        let sizer = PositionSizer::new(100000.0, 0.015); // $100k, 1.5% risk

        let (units, _usd, risk) = sizer
            .calculate_position_size(
                Price(50000.0),
                Price(49000.0), // $1000 stop loss
                None,
            )
            .unwrap();

        // Risk amount should be 1.5% of $100k = $1500
        assert!((risk.value() - 1500.0).abs() < 1.0);

        // Position size should be $1500 / $1000 = 1.5 units
        assert!((units.value() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_position_sizer_with_limits() {
        let sizer = PositionSizer::new(100000.0, 0.02); // 2% risk

        let (_units, usd, _risk) = sizer
            .calculate_position_size_with_limits(
                Price(50000.0),
                Price(49000.0),
                None,
                Some(0.01), // Max 1% of portfolio
                None,
            )
            .unwrap();

        // Should be capped at 1% of $100k = $1000
        assert!(usd.value() <= 1000.0);
    }
}
