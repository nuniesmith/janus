//! Order book validator for market data

use async_trait::async_trait;
use janus_core::{MarketDataEvent, OrderBookEvent};
use rust_decimal::Decimal;

use super::{OrderBookValidatorConfig, Validator};
use crate::{Result, SharedState, ValidationResult};

/// Order book validator
pub struct OrderBookValidator {
    config: OrderBookValidatorConfig,
    #[allow(dead_code)]
    state: SharedState,
}

impl OrderBookValidator {
    /// Create a new order book validator
    pub fn new(config: OrderBookValidatorConfig, state: SharedState) -> Self {
        Self { config, state }
    }

    /// Validate order book
    fn validate_orderbook_internal(&self, orderbook: &OrderBookEvent) -> ValidationResult {
        if !self.config.enabled {
            return ValidationResult::success(self.name());
        }

        let mut result = ValidationResult::success(self.name());

        // Check bid/ask spread
        if self.config.check_bid_ask_spread
            && !orderbook.bids.is_empty()
            && !orderbook.asks.is_empty()
        {
            let best_bid = orderbook.bids[0].price;
            let best_ask = orderbook.asks[0].price;

            if best_bid >= best_ask {
                return ValidationResult::failure(
                    self.name(),
                    format!("Bid {} >= Ask {} (crossed book)", best_bid, best_ask),
                );
            }

            // Check spread percentage
            let spread_pct = ((best_ask - best_bid) / best_ask).abs();
            let min_spread = Decimal::try_from(self.config.min_spread_pct).unwrap_or(Decimal::ZERO);
            let max_spread = Decimal::try_from(self.config.max_spread_pct).unwrap_or(Decimal::ONE);

            if spread_pct < min_spread {
                result = result.with_warning(format!(
                    "Spread {:.4}% below minimum {:.4}%",
                    spread_pct * Decimal::from(100),
                    self.config.min_spread_pct * 100.0
                ));
            }

            if spread_pct > max_spread {
                result = result.with_warning(format!(
                    "Spread {:.4}% above maximum {:.4}%",
                    spread_pct * Decimal::from(100),
                    self.config.max_spread_pct * 100.0
                ));
            }
        }

        // Validate bids
        if self.config.check_sorted_levels {
            for i in 1..orderbook.bids.len().min(self.config.max_levels) {
                if orderbook.bids[i].price >= orderbook.bids[i - 1].price {
                    return ValidationResult::failure(
                        self.name(),
                        format!(
                            "Bids not sorted descending: level {} ({}) >= level {} ({})",
                            i,
                            orderbook.bids[i].price,
                            i - 1,
                            orderbook.bids[i - 1].price
                        ),
                    );
                }
            }
        }

        // Validate asks
        if self.config.check_sorted_levels {
            for i in 1..orderbook.asks.len().min(self.config.max_levels) {
                if orderbook.asks[i].price <= orderbook.asks[i - 1].price {
                    return ValidationResult::failure(
                        self.name(),
                        format!(
                            "Asks not sorted ascending: level {} ({}) <= level {} ({})",
                            i,
                            orderbook.asks[i].price,
                            i - 1,
                            orderbook.asks[i - 1].price
                        ),
                    );
                }
            }
        }

        // Check positive sizes
        if self.config.check_positive_sizes {
            for level in orderbook.bids.iter().take(self.config.max_levels) {
                if level.quantity.is_sign_negative() || level.quantity.is_zero() {
                    return ValidationResult::failure(
                        self.name(),
                        format!("Bid at {} has invalid size {}", level.price, level.quantity),
                    );
                }
            }

            for level in orderbook.asks.iter().take(self.config.max_levels) {
                if level.quantity.is_sign_negative() || level.quantity.is_zero() {
                    return ValidationResult::failure(
                        self.name(),
                        format!("Ask at {} has invalid size {}", level.price, level.quantity),
                    );
                }
            }
        }

        result
    }
}

#[async_trait]
impl Validator for OrderBookValidator {
    fn name(&self) -> &str {
        "orderbook_validator"
    }

    async fn validate(&self, event: &MarketDataEvent) -> Result<ValidationResult> {
        match event {
            MarketDataEvent::OrderBook(orderbook) => self.validate_orderbook(orderbook).await,
            _ => Ok(ValidationResult::success(self.name())),
        }
    }

    async fn validate_orderbook(&self, orderbook: &OrderBookEvent) -> Result<ValidationResult> {
        Ok(self.validate_orderbook_internal(orderbook))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Symbol};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn create_validator() -> OrderBookValidator {
        let config = OrderBookValidatorConfig::default();
        let state = Arc::new(RwLock::new(crate::MarketDataState::new()));
        OrderBookValidator::new(config, state)
    }

    #[tokio::test]
    async fn test_validate_valid_orderbook() {
        use janus_core::PriceLevel;
        let validator = create_validator();
        let orderbook = OrderBookEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            bids: vec![
                PriceLevel::new(Decimal::new(50000, 0), Decimal::new(100, 2)),
                PriceLevel::new(Decimal::new(49990, 0), Decimal::new(200, 2)),
            ],
            asks: vec![
                PriceLevel::new(Decimal::new(50010, 0), Decimal::new(100, 2)),
                PriceLevel::new(Decimal::new(50020, 0), Decimal::new(200, 2)),
            ],
            timestamp: 1234567890,
            sequence: 1,
            is_snapshot: true,
        };

        let result = validator.validate_orderbook(&orderbook).await.unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_validate_crossed_book() {
        use janus_core::PriceLevel;
        let validator = create_validator();
        let orderbook = OrderBookEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            bids: vec![PriceLevel::new(
                Decimal::new(50010, 0),
                Decimal::new(100, 2),
            )],
            asks: vec![PriceLevel::new(
                Decimal::new(50000, 0),
                Decimal::new(100, 2),
            )],
            timestamp: 1234567890,
            sequence: 1,
            is_snapshot: true,
        };

        let result = validator.validate_orderbook(&orderbook).await.unwrap();
        assert!(!result.is_valid);
        assert!(result.errors[0].contains("crossed"));
    }

    #[tokio::test]
    async fn test_validate_unsorted_bids() {
        use janus_core::PriceLevel;
        let validator = create_validator();
        let orderbook = OrderBookEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            bids: vec![
                PriceLevel::new(Decimal::new(49990, 0), Decimal::new(100, 2)),
                PriceLevel::new(Decimal::new(50000, 0), Decimal::new(200, 2)), // Wrong order
            ],
            asks: vec![PriceLevel::new(
                Decimal::new(50010, 0),
                Decimal::new(100, 2),
            )],
            timestamp: 1234567890,
            sequence: 1,
            is_snapshot: true,
        };

        let result = validator.validate_orderbook(&orderbook).await.unwrap();
        assert!(!result.is_valid);
    }
}
