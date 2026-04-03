//! Price validator for market data

use async_trait::async_trait;
use janus_core::{KlineEvent, MarketDataEvent, TickerEvent, TradeEvent};
use rust_decimal::Decimal;

use super::{PriceValidatorConfig, Validator};
use crate::{Result, SharedState, ValidationResult};

/// Price validator
pub struct PriceValidator {
    config: PriceValidatorConfig,
    state: SharedState,
}

impl PriceValidator {
    /// Create a new price validator
    pub fn new(config: PriceValidatorConfig, state: SharedState) -> Self {
        Self { config, state }
    }

    /// Validate a price value
    fn validate_price(&self, price: Decimal, last_price: Option<Decimal>) -> ValidationResult {
        if !self.config.enabled {
            return ValidationResult::success(self.name());
        }

        let mut result = ValidationResult::success(self.name());

        // Check for zero
        if self.config.reject_zero_prices && price.is_zero() {
            return ValidationResult::failure(self.name(), "Price is zero");
        }

        // Check for negative
        if self.config.reject_negative_prices && price.is_sign_negative() {
            return ValidationResult::failure(self.name(), "Price is negative");
        }

        // Check min/max bounds
        if let Some(min_price) = self.config.min_price
            && price < min_price
        {
            return ValidationResult::failure(
                self.name(),
                format!("Price {} below minimum {}", price, min_price),
            );
        }

        if let Some(max_price) = self.config.max_price
            && price > max_price
        {
            return ValidationResult::failure(
                self.name(),
                format!("Price {} above maximum {}", price, max_price),
            );
        }

        // Check decimal places
        let scale = price.scale();
        if scale > self.config.max_decimals {
            result = result.with_warning(format!(
                "Price has {} decimals, max is {}",
                scale, self.config.max_decimals
            ));
        }

        // Spike detection
        if self.config.spike_detection_enabled
            && let Some(last) = last_price
            && !last.is_zero()
        {
            let change = ((price - last) / last).abs();
            let max_change =
                Decimal::try_from(self.config.max_price_change_pct).unwrap_or(Decimal::new(20, 2)); // 0.20

            if change > max_change {
                result = result.with_warning(format!(
                    "Price spike detected: {:.2}% change from {} to {}",
                    change * Decimal::from(100),
                    last,
                    price
                ));
            }
        }

        result
    }
}

#[async_trait]
impl Validator for PriceValidator {
    fn name(&self) -> &str {
        "price_validator"
    }

    async fn validate(&self, event: &MarketDataEvent) -> Result<ValidationResult> {
        match event {
            MarketDataEvent::Trade(trade) => self.validate_trade(trade).await,
            MarketDataEvent::Ticker(ticker) => self.validate_ticker(ticker).await,
            MarketDataEvent::Kline(kline) => self.validate_kline(kline).await,
            _ => Ok(ValidationResult::success(self.name())),
        }
    }

    async fn validate_ticker(&self, ticker: &TickerEvent) -> Result<ValidationResult> {
        let state = self.state.read().await;
        let last_price = state
            .price_stats
            .get(&ticker.symbol)
            .and_then(|window| window.data.last().copied())
            .map(|p| Decimal::try_from(p).unwrap_or(ticker.last_price));

        drop(state);

        // Validate last price
        let price_result = self.validate_price(ticker.last_price, last_price);
        if !price_result.is_valid {
            return Ok(price_result);
        }

        // Validate best bid and ask if available
        if let (Some(bid), Some(ask)) = (ticker.best_bid, ticker.best_ask) {
            let bid_result = self.validate_price(bid, last_price);
            if !bid_result.is_valid {
                return Ok(bid_result);
            }

            let ask_result = self.validate_price(ask, last_price);
            if !ask_result.is_valid {
                return Ok(ask_result);
            }

            // Check bid < ask
            if bid >= ask {
                return Ok(ValidationResult::failure(
                    self.name(),
                    format!("Bid price {} >= Ask price {}", bid, ask),
                ));
            }
        }

        // Update state
        let mut state = self.state.write().await;
        let window = state
            .price_stats
            .entry(ticker.symbol.clone())
            .or_insert_with(|| crate::RollingWindow::new(100));

        if let Ok(price_f64) = ticker.last_price.to_string().parse::<f64>() {
            window.add(price_f64);
        }

        Ok(ValidationResult::success(self.name()))
    }

    async fn validate_trade(&self, trade: &TradeEvent) -> Result<ValidationResult> {
        let state = self.state.read().await;
        let last_price = state
            .price_stats
            .get(&trade.symbol)
            .and_then(|window| window.data.last().copied())
            .map(|p| Decimal::try_from(p).unwrap_or(trade.price));

        drop(state);

        let result = self.validate_price(trade.price, last_price);

        // Update state if valid
        if result.is_valid {
            let mut state = self.state.write().await;
            let window = state
                .price_stats
                .entry(trade.symbol.clone())
                .or_insert_with(|| crate::RollingWindow::new(100));

            if let Ok(price_f64) = trade.price.to_string().parse::<f64>() {
                window.add(price_f64);
            }
        }

        Ok(result)
    }

    async fn validate_kline(&self, kline: &KlineEvent) -> Result<ValidationResult> {
        // Validate OHLC relationships
        if kline.high < kline.low {
            return Ok(ValidationResult::failure(
                self.name(),
                format!("High {} < Low {}", kline.high, kline.low),
            ));
        }

        if kline.open < kline.low || kline.open > kline.high {
            return Ok(ValidationResult::failure(
                self.name(),
                format!(
                    "Open {} outside range [{}, {}]",
                    kline.open, kline.low, kline.high
                ),
            ));
        }

        if kline.close < kline.low || kline.close > kline.high {
            return Ok(ValidationResult::failure(
                self.name(),
                format!(
                    "Close {} outside range [{}, {}]",
                    kline.close, kline.low, kline.high
                ),
            ));
        }

        Ok(ValidationResult::success(self.name()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Side, Symbol};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn create_validator() -> PriceValidator {
        let config = PriceValidatorConfig::default();
        let state = Arc::new(RwLock::new(crate::MarketDataState::new()));
        PriceValidator::new(config, state)
    }

    #[tokio::test]
    async fn test_validate_zero_price() {
        let validator = create_validator();
        let trade = TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::ZERO,
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: 1234567890,
            received_at: 1234567890,
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        };

        let result = validator.validate_trade(&trade).await.unwrap();
        assert!(!result.is_valid);
        assert!(result.errors[0].contains("zero"));
    }

    #[tokio::test]
    async fn test_validate_negative_price() {
        let validator = create_validator();
        let trade = TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::new(-1000, 0),
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: 1234567890,
            received_at: 1234567890,
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        };

        let result = validator.validate_trade(&trade).await.unwrap();
        assert!(!result.is_valid);
        assert!(result.errors[0].contains("negative"));
    }

    #[tokio::test]
    async fn test_validate_valid_trade() {
        let validator = create_validator();
        let trade = TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::new(50000, 0),
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: 1234567890,
            received_at: 1234567890,
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        };

        let result = validator.validate_trade(&trade).await.unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_validate_kline() {
        let validator = create_validator();
        let kline = KlineEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            interval: "1m".to_string(),
            open_time: 1234567890,
            close_time: 1234567950,
            open: Decimal::new(50000, 0),
            high: Decimal::new(50100, 0),
            low: Decimal::new(49900, 0),
            close: Decimal::new(50050, 0),
            volume: Decimal::new(100, 0),
            quote_volume: Some(Decimal::new(5000000, 0)),
            trades: Some(50),
            is_closed: true,
        };

        let result = validator.validate_kline(&kline).await.unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_validate_invalid_kline() {
        let validator = create_validator();
        let kline = KlineEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            interval: "1m".to_string(),
            open_time: 1234567890,
            close_time: 1234567950,
            open: Decimal::new(50000, 0),
            high: Decimal::new(49900, 0), // High < Low (invalid)
            low: Decimal::new(50100, 0),
            close: Decimal::new(50050, 0),
            volume: Decimal::new(100, 0),
            quote_volume: Some(Decimal::new(5000000, 0)),
            trades: Some(50),
            is_closed: true,
        };

        let result = validator.validate_kline(&kline).await.unwrap();
        assert!(!result.is_valid);
    }
}
