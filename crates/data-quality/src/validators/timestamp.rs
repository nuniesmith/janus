//! Timestamp validator for market data

use async_trait::async_trait;
use janus_core::MarketDataEvent;

use super::{TimestampValidatorConfig, Validator};
use crate::{Result, SharedState, ValidationResult};

/// Timestamp validator
pub struct TimestampValidator {
    config: TimestampValidatorConfig,
    state: SharedState,
}

impl TimestampValidator {
    /// Create a new timestamp validator
    pub fn new(config: TimestampValidatorConfig, state: SharedState) -> Self {
        Self { config, state }
    }

    /// Validate a timestamp
    async fn validate_timestamp(
        &self,
        exchange: janus_core::Exchange,
        symbol: &janus_core::Symbol,
        timestamp: i64,
    ) -> ValidationResult {
        if !self.config.enabled {
            return ValidationResult::success(self.name());
        }

        let now = chrono::Utc::now().timestamp_micros();
        let mut result = ValidationResult::success(self.name());

        // Check for future timestamps
        let future_diff = timestamp - now;
        if future_diff > self.config.max_future_us {
            return ValidationResult::failure(
                self.name(),
                format!(
                    "Timestamp {} is too far in the future ({} µs ahead)",
                    timestamp, future_diff
                ),
            );
        }

        // Check for past timestamps
        let past_diff = now - timestamp;
        if past_diff > self.config.max_past_us {
            result = result.with_warning(format!(
                "Timestamp {} is {} µs in the past",
                timestamp, past_diff
            ));
        }

        // Check monotonicity
        if self.config.check_monotonicity {
            let state = self.state.read().await;
            if let Some(last_ts) = state.get_last_timestamp(exchange, symbol) {
                if timestamp <= last_ts && self.config.detect_duplicates {
                    return ValidationResult::failure(
                        self.name(),
                        format!("Non-monotonic timestamp: {} <= last {}", timestamp, last_ts),
                    );
                } else if timestamp < last_ts {
                    result = result.with_warning(format!(
                        "Timestamp {} is before last timestamp {}",
                        timestamp, last_ts
                    ));
                }
            }
        }

        result
    }
}

#[async_trait]
impl Validator for TimestampValidator {
    fn name(&self) -> &str {
        "timestamp_validator"
    }

    async fn validate(&self, event: &MarketDataEvent) -> Result<ValidationResult> {
        let (exchange, symbol, timestamp) = match event {
            MarketDataEvent::Trade(t) => (t.exchange, &t.symbol, t.timestamp),
            MarketDataEvent::Ticker(t) => (t.exchange, &t.symbol, t.timestamp),
            MarketDataEvent::Kline(k) => (k.exchange, &k.symbol, k.open_time),
            MarketDataEvent::OrderBook(ob) => (ob.exchange, &ob.symbol, ob.timestamp),
            MarketDataEvent::Liquidation(l) => (l.exchange, &l.symbol, l.timestamp),
            MarketDataEvent::FundingRate(f) => (f.exchange, &f.symbol, f.timestamp),
        };

        let result = self.validate_timestamp(exchange, symbol, timestamp).await;

        // Update state if valid
        if result.is_valid {
            let mut state = self.state.write().await;
            state.update_timestamp(exchange, symbol.clone(), timestamp);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Side, Symbol, TradeEvent};
    use rust_decimal::Decimal;
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn create_validator() -> TimestampValidator {
        let config = TimestampValidatorConfig::default();
        let state = Arc::new(RwLock::new(crate::MarketDataState::new()));
        TimestampValidator::new(config, state)
    }

    #[tokio::test]
    async fn test_validate_current_timestamp() {
        let validator = create_validator();
        let now = chrono::Utc::now().timestamp_micros();
        let trade = TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::new(50000, 0),
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: now,
            received_at: now,
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        };

        let event = MarketDataEvent::Trade(trade);
        let result = validator.validate(&event).await.unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_validate_future_timestamp() {
        let validator = create_validator();
        let future = chrono::Utc::now().timestamp_micros() + 10_000_000; // 10s future
        let trade = TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::new(50000, 0),
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: future,
            received_at: future,
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        };

        let event = MarketDataEvent::Trade(trade);
        let result = validator.validate(&event).await.unwrap();
        assert!(!result.is_valid);
    }
}
