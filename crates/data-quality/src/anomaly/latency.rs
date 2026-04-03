//! Latency anomaly detector for detecting excessive ingestion delays

use janus_core::MarketDataEvent;

use super::{AnomalyDetector, AnomalyResult, AnomalySeverity};
use crate::Result;

/// Latency anomaly detector
pub struct LatencyAnomalyDetector {
    /// Maximum acceptable latency (microseconds)
    max_latency_us: i64,
}

impl LatencyAnomalyDetector {
    /// Create a new latency anomaly detector
    pub fn new(max_latency_us: i64) -> Self {
        Self { max_latency_us }
    }
}

impl AnomalyDetector for LatencyAnomalyDetector {
    fn name(&self) -> &str {
        "latency_anomaly_detector"
    }

    fn detect(&mut self, event: &MarketDataEvent) -> Result<AnomalyResult> {
        let (symbol, event_timestamp) = match event {
            MarketDataEvent::Trade(t) => (&t.symbol, t.timestamp),
            MarketDataEvent::Ticker(t) => (&t.symbol, t.timestamp),
            MarketDataEvent::Kline(k) => (&k.symbol, k.open_time),
            MarketDataEvent::OrderBook(ob) => (&ob.symbol, ob.timestamp),
            MarketDataEvent::Liquidation(l) => (&l.symbol, l.timestamp),
            MarketDataEvent::FundingRate(f) => (&f.symbol, f.timestamp),
        };

        let now = chrono::Utc::now().timestamp_micros();
        let latency = now - event_timestamp;

        if latency > self.max_latency_us {
            let severity = if latency > self.max_latency_us * 10 {
                AnomalySeverity::High
            } else if latency > self.max_latency_us * 5 {
                AnomalySeverity::Medium
            } else {
                AnomalySeverity::Low
            };

            Ok(AnomalyResult::detected(
                self.name(),
                symbol.clone(),
                severity,
                format!(
                    "High latency: {:.2}ms (threshold: {:.2}ms)",
                    latency as f64 / 1000.0,
                    self.max_latency_us as f64 / 1000.0
                ),
            )
            .with_score(latency as f64))
        } else {
            Ok(AnomalyResult::none(self.name(), symbol.clone()))
        }
    }

    fn reset(&mut self) {
        // No state to reset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Side, Symbol, TradeEvent};
    use rust_decimal::Decimal;

    #[test]
    fn test_latency_detection() {
        let mut detector = LatencyAnomalyDetector::new(1_000_000); // 1 second

        let symbol = Symbol::new("BTC", "USD");

        // Current timestamp - should not trigger
        let now = chrono::Utc::now().timestamp_micros();
        let trade = TradeEvent {
            symbol: symbol.clone(),
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
        let result = detector.detect(&event).unwrap();
        assert!(!result.is_anomaly);

        // Old timestamp - should trigger high severity
        let old = now - 15_000_000; // 15 seconds ago
        let trade = TradeEvent {
            symbol: symbol.clone(),
            exchange: Exchange::Coinbase,
            price: Decimal::new(50000, 0),
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: old,
            received_at: now + 2_000_000,
            trade_id: "124".to_string(),
            buyer_is_maker: None,
        };
        let event = MarketDataEvent::Trade(trade);
        let result = detector.detect(&event).unwrap();
        assert!(result.is_anomaly);
        assert_eq!(result.severity, AnomalySeverity::High);
    }
}
