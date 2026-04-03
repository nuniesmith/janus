//! Price and Volume Normalization
//!
//! This module provides utilities for normalizing price and volume data
//! across exchanges with different precision levels and quote currencies.

use janus_core::{Exchange, Symbol};
use rust_decimal::Decimal;
use std::collections::HashMap;

/// Price normalizer for cross-exchange comparison
pub struct PriceNormalizer {
    /// Exchange-specific price scales
    price_scales: HashMap<(Exchange, Symbol), Decimal>,
    /// Volume scales
    volume_scales: HashMap<(Exchange, Symbol), Decimal>,
}

impl PriceNormalizer {
    /// Create a new price normalizer
    pub fn new() -> Self {
        Self {
            price_scales: HashMap::new(),
            volume_scales: HashMap::new(),
        }
    }

    /// Set price scale for an exchange/symbol pair
    pub fn set_price_scale(&mut self, exchange: Exchange, symbol: Symbol, scale: Decimal) {
        self.price_scales.insert((exchange, symbol), scale);
    }

    /// Set volume scale for an exchange/symbol pair
    pub fn set_volume_scale(&mut self, exchange: Exchange, symbol: Symbol, scale: Decimal) {
        self.volume_scales.insert((exchange, symbol), scale);
    }

    /// Normalize price to standard precision (8 decimal places)
    pub fn normalize_price(&self, price: Decimal, exchange: Exchange, symbol: &Symbol) -> Decimal {
        let scale = self
            .price_scales
            .get(&(exchange, symbol.clone()))
            .copied()
            .unwrap_or(Decimal::ONE);

        price * scale
    }

    /// Normalize volume to standard precision
    pub fn normalize_volume(
        &self,
        volume: Decimal,
        exchange: Exchange,
        symbol: &Symbol,
    ) -> Decimal {
        let scale = self
            .volume_scales
            .get(&(exchange, symbol.clone()))
            .copied()
            .unwrap_or(Decimal::ONE);

        volume * scale
    }

    /// Convert price between quote currencies
    pub fn convert_quote(
        &self,
        price: Decimal,
        from_quote: &str,
        to_quote: &str,
        conversion_rate: Decimal,
    ) -> Decimal {
        if from_quote == to_quote {
            return price;
        }

        price * conversion_rate
    }
}

impl Default for PriceNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_price_normalization() {
        let mut normalizer = PriceNormalizer::new();
        let symbol = Symbol::new("BTC", "USDT");

        normalizer.set_price_scale(Exchange::Binance, symbol.clone(), Decimal::ONE);

        let normalized =
            normalizer.normalize_price(Decimal::from(50000), Exchange::Binance, &symbol);

        assert_eq!(normalized, Decimal::from(50000));
    }

    #[test]
    fn test_quote_conversion() {
        let normalizer = PriceNormalizer::new();

        // Convert BTC price from USD to EUR (example rate)
        let btc_usd = Decimal::from(50000);
        let usd_to_eur = Decimal::new(85, 2); // 0.85

        let btc_eur = normalizer.convert_quote(btc_usd, "USD", "EUR", usd_to_eur);

        assert_eq!(btc_eur, Decimal::from(42500));
    }

    #[test]
    fn test_same_quote_conversion() {
        let normalizer = PriceNormalizer::new();

        let price = Decimal::from(50000);
        let result = normalizer.convert_quote(price, "USD", "USD", Decimal::ONE);

        assert_eq!(result, price);
    }
}
