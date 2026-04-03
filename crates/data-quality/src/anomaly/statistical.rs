//! Statistical anomaly detector using z-score analysis

use janus_core::{MarketDataEvent, Symbol};
use std::collections::HashMap;

use super::{AnomalyDetector, AnomalyResult, AnomalySeverity};
use crate::{Result, RollingWindow};

/// Statistical anomaly detector
pub struct StatisticalAnomalyDetector {
    /// Z-score threshold for anomaly detection
    threshold: f64,

    /// Window size for statistics
    window_size: usize,

    /// Price windows per symbol
    price_windows: HashMap<Symbol, RollingWindow>,

    /// Volume windows per symbol
    volume_windows: HashMap<Symbol, RollingWindow>,
}

impl StatisticalAnomalyDetector {
    /// Create a new statistical anomaly detector
    pub fn new(threshold: f64, window_size: usize) -> Self {
        Self {
            threshold,
            window_size,
            price_windows: HashMap::new(),
            volume_windows: HashMap::new(),
        }
    }

    /// Detect price anomalies
    fn detect_price_anomaly(&mut self, symbol: &Symbol, price: f64) -> Option<AnomalyResult> {
        let window = self
            .price_windows
            .entry(symbol.clone())
            .or_insert_with(|| RollingWindow::new(self.window_size));

        if window.len() < 10 {
            // Need minimum data points
            window.add(price);
            return None;
        }

        let stats = window.statistics();
        let z_score = stats.z_score(price);

        window.add(price);

        if z_score.abs() > self.threshold {
            let severity = if z_score.abs() > self.threshold * 2.0 {
                AnomalySeverity::High
            } else if z_score.abs() > self.threshold * 1.5 {
                AnomalySeverity::Medium
            } else {
                AnomalySeverity::Low
            };

            Some(
                AnomalyResult::detected(
                    self.name(),
                    symbol.clone(),
                    severity,
                    format!(
                        "Price anomaly: z-score {:.2} (price: {:.2}, mean: {:.2}, stddev: {:.2})",
                        z_score, price, stats.mean, stats.std_dev
                    ),
                )
                .with_score(z_score.abs()),
            )
        } else {
            None
        }
    }

    /// Detect volume anomalies
    fn detect_volume_anomaly(&mut self, symbol: &Symbol, volume: f64) -> Option<AnomalyResult> {
        let window = self
            .volume_windows
            .entry(symbol.clone())
            .or_insert_with(|| RollingWindow::new(self.window_size));

        if window.len() < 10 {
            window.add(volume);
            return None;
        }

        let stats = window.statistics();
        let z_score = stats.z_score(volume);

        window.add(volume);

        if z_score.abs() > self.threshold {
            let severity = if z_score.abs() > self.threshold * 2.0 {
                AnomalySeverity::High
            } else {
                AnomalySeverity::Medium
            };

            Some(
                AnomalyResult::detected(
                    self.name(),
                    symbol.clone(),
                    severity,
                    format!(
                        "Volume anomaly: z-score {:.2} (volume: {:.2}, mean: {:.2})",
                        z_score, volume, stats.mean
                    ),
                )
                .with_score(z_score.abs()),
            )
        } else {
            None
        }
    }
}

impl AnomalyDetector for StatisticalAnomalyDetector {
    fn name(&self) -> &str {
        "statistical_anomaly_detector"
    }

    fn detect(&mut self, event: &MarketDataEvent) -> Result<AnomalyResult> {
        match event {
            MarketDataEvent::Trade(trade) => {
                let price = trade.price.to_string().parse::<f64>().unwrap_or(0.0);
                let volume = trade.quantity.to_string().parse::<f64>().unwrap_or(0.0);

                if let Some(price_anomaly) = self.detect_price_anomaly(&trade.symbol, price) {
                    return Ok(price_anomaly);
                }

                if let Some(volume_anomaly) = self.detect_volume_anomaly(&trade.symbol, volume) {
                    return Ok(volume_anomaly);
                }

                Ok(AnomalyResult::none(self.name(), trade.symbol.clone()))
            }
            MarketDataEvent::Ticker(ticker) => {
                let price = ticker.last_price.to_string().parse::<f64>().unwrap_or(0.0);

                if let Some(anomaly) = self.detect_price_anomaly(&ticker.symbol, price) {
                    return Ok(anomaly);
                }

                Ok(AnomalyResult::none(self.name(), ticker.symbol.clone()))
            }
            MarketDataEvent::Kline(kline) => {
                let price = kline.close.to_string().parse::<f64>().unwrap_or(0.0);
                let volume = kline.volume.to_string().parse::<f64>().unwrap_or(0.0);

                if let Some(price_anomaly) = self.detect_price_anomaly(&kline.symbol, price) {
                    return Ok(price_anomaly);
                }

                if let Some(volume_anomaly) = self.detect_volume_anomaly(&kline.symbol, volume) {
                    return Ok(volume_anomaly);
                }

                Ok(AnomalyResult::none(self.name(), kline.symbol.clone()))
            }
            _ => Ok(AnomalyResult::none(
                self.name(),
                Symbol::new("UNKNOWN", "UNKNOWN"),
            )),
        }
    }

    fn reset(&mut self) {
        self.price_windows.clear();
        self.volume_windows.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Side, TradeEvent};
    use rust_decimal::Decimal;

    #[test]
    fn test_price_anomaly_detection() {
        let mut detector = StatisticalAnomalyDetector::new(3.0, 100);

        let symbol = Symbol::new("BTC", "USD");
        let exchange = Exchange::Coinbase;

        // Add normal data
        for i in 0..20 {
            let trade = TradeEvent {
                symbol: symbol.clone(),
                exchange,
                price: Decimal::new(50000 + i, 0),
                quantity: Decimal::ONE,
                side: Side::Buy,
                timestamp: 1234567890 + i,
                received_at: 1234567890 + i,
                trade_id: format!("{}", i),
                buyer_is_maker: None,
            };
            let event = MarketDataEvent::Trade(trade);
            let _ = detector.detect(&event);
        }

        // Add anomalous price
        let trade = TradeEvent {
            symbol: symbol.clone(),
            exchange,
            price: Decimal::new(60000, 0), // Spike
            quantity: Decimal::ONE,
            side: Side::Buy,
            timestamp: 1234567910,
            received_at: 1234567910,
            trade_id: "20".to_string(),
            buyer_is_maker: None,
        };
        let event = MarketDataEvent::Trade(trade);
        let result = detector.detect(&event).unwrap();

        assert!(result.is_anomaly);
        assert!(result.score.is_some());
    }
}
