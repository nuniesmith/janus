//! Volume validator for market data

use async_trait::async_trait;
use janus_core::{KlineEvent, MarketDataEvent, TradeEvent};
use rust_decimal::Decimal;

use super::{Validator, VolumeValidatorConfig};
use crate::{Result, SharedState, ValidationResult};

/// Volume validator
pub struct VolumeValidator {
    config: VolumeValidatorConfig,
    state: SharedState,
}

impl VolumeValidator {
    /// Create a new volume validator
    pub fn new(config: VolumeValidatorConfig, state: SharedState) -> Self {
        Self { config, state }
    }

    /// Validate a volume value
    fn validate_volume(&self, volume: Decimal, last_avg: Option<f64>) -> ValidationResult {
        if !self.config.enabled {
            return ValidationResult::success(self.name());
        }

        let mut result = ValidationResult::success(self.name());

        // Check for zero
        if self.config.reject_zero_volumes && volume.is_zero() {
            return ValidationResult::failure(self.name(), "Volume is zero");
        }

        // Check for negative
        if self.config.reject_negative_volumes && volume.is_sign_negative() {
            return ValidationResult::failure(self.name(), "Volume is negative");
        }

        // Check min/max bounds
        if let Some(min_volume) = self.config.min_volume
            && volume < min_volume
        {
            return ValidationResult::failure(
                self.name(),
                format!("Volume {} below minimum {}", volume, min_volume),
            );
        }

        if let Some(max_volume) = self.config.max_volume
            && volume > max_volume
        {
            return ValidationResult::failure(
                self.name(),
                format!("Volume {} above maximum {}", volume, max_volume),
            );
        }

        // Volume spike detection
        if let Some(avg) = last_avg
            && avg > 0.0
            && let Ok(vol_f64) = volume.to_string().parse::<f64>()
        {
            let ratio = vol_f64 / avg;
            if ratio > self.config.max_volume_spike_ratio {
                result = result.with_warning(format!(
                    "Volume spike detected: {:.2}x average (volume: {}, avg: {:.2})",
                    ratio, volume, avg
                ));
            }
        }

        result
    }
}

#[async_trait]
impl Validator for VolumeValidator {
    fn name(&self) -> &str {
        "volume_validator"
    }

    async fn validate(&self, event: &MarketDataEvent) -> Result<ValidationResult> {
        match event {
            MarketDataEvent::Trade(trade) => self.validate_trade(trade).await,
            MarketDataEvent::Kline(kline) => self.validate_kline(kline).await,
            _ => Ok(ValidationResult::success(self.name())),
        }
    }

    async fn validate_trade(&self, trade: &TradeEvent) -> Result<ValidationResult> {
        let state = self.state.read().await;
        let stats = state
            .volume_stats
            .get(&trade.symbol)
            .map(|w| w.statistics());

        let avg = stats.map(|s| s.mean);
        drop(state);

        let result = self.validate_volume(trade.quantity, avg);

        // Update state if valid
        if result.is_valid {
            let mut state = self.state.write().await;
            let window = state
                .volume_stats
                .entry(trade.symbol.clone())
                .or_insert_with(|| crate::RollingWindow::new(100));

            if let Ok(vol_f64) = trade.quantity.to_string().parse::<f64>() {
                window.add(vol_f64);
            }
        }

        Ok(result)
    }

    async fn validate_kline(&self, kline: &KlineEvent) -> Result<ValidationResult> {
        let state = self.state.read().await;
        let stats = state
            .volume_stats
            .get(&kline.symbol)
            .map(|w| w.statistics());

        let avg = stats.map(|s| s.mean);
        drop(state);

        let result = self.validate_volume(kline.volume, avg);

        // Update state if valid
        if result.is_valid {
            let mut state = self.state.write().await;
            let window = state
                .volume_stats
                .entry(kline.symbol.clone())
                .or_insert_with(|| crate::RollingWindow::new(100));

            if let Ok(vol_f64) = kline.volume.to_string().parse::<f64>() {
                window.add(vol_f64);
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Side, Symbol};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn create_validator() -> VolumeValidator {
        let config = VolumeValidatorConfig::default();
        let state = Arc::new(RwLock::new(crate::MarketDataState::new()));
        VolumeValidator::new(config, state)
    }

    #[tokio::test]
    async fn test_validate_negative_volume() {
        let validator = create_validator();
        let trade = TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::new(50000, 0),
            quantity: Decimal::new(-1, 0),
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
    async fn test_validate_valid_volume() {
        let validator = create_validator();
        let trade = TradeEvent {
            symbol: Symbol::new("BTC", "USD"),
            exchange: Exchange::Coinbase,
            price: Decimal::new(50000, 0),
            quantity: Decimal::new(100, 2), // 1.00
            side: Side::Buy,
            timestamp: 1234567890,
            received_at: 1234567890,
            trade_id: "123".to_string(),
            buyer_is_maker: None,
        };

        let result = validator.validate_trade(&trade).await.unwrap();
        assert!(result.is_valid);
    }
}
