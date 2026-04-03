//! CNS (Central Nervous System) Reporter for Exchange Metrics
//!
//! This module provides a reporter that sends exchange health and performance
//! metrics to the CNS Prometheus metrics system.
//!
//! ## Usage
//!
//! ```rust
//! use janus_exchanges::CNSReporter;
//! use std::time::Duration;
//!
//! let reporter = CNSReporter::new("binance");
//!
//! // Record successful message
//! reporter.record_message("trades", "BTC-USDT");
//!
//! // Record parse error
//! reporter.record_parse_error("invalid_json");
//!
//! // Record latency
//! reporter.record_latency("trades", Duration::from_millis(5));
//!
//! // Update health status
//! use janus_exchanges::health::ExchangeHealthStatus;
//! reporter.update_health(ExchangeHealthStatus::Healthy);
//! ```

use crate::health::ExchangeHealthStatus;
use std::time::Duration;

/// CNS Reporter for exchange metrics
///
/// Sends exchange health and performance metrics to the CNS Prometheus registry.
/// This allows exchange adapter health to be monitored via Grafana dashboards.
pub struct CNSReporter {
    /// Exchange name (e.g., "binance", "coinbase")
    exchange: String,
}

impl CNSReporter {
    /// Create a new CNS reporter for an exchange
    ///
    /// # Arguments
    ///
    /// * `exchange` - Exchange name (should match Exchange enum string representation)
    ///
    /// # Example
    ///
    /// ```
    /// use janus_exchanges::CNSReporter;
    ///
    /// let reporter = CNSReporter::new("coinbase");
    /// ```
    pub fn new(exchange: &str) -> Self {
        Self {
            exchange: exchange.to_lowercase(),
        }
    }

    /// Record a successfully parsed message
    ///
    /// Increments the `janus_exchange_message_total` counter with labels:
    /// - `exchange`: Exchange name
    /// - `channel`: Channel/stream name (e.g., "trades", "ticker", "orderbook")
    /// - `symbol`: Trading pair symbol (e.g., "BTC-USDT")
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel or stream name
    /// * `symbol` - Trading pair symbol
    ///
    /// # Example
    ///
    /// ```
    /// # use janus_exchanges::CNSReporter;
    /// let reporter = CNSReporter::new("coinbase");
    /// reporter.record_message("trades", "BTC-USD");
    /// reporter.record_message("level2", "ETH-USD");
    /// ```
    pub fn record_message(&self, channel: &str, symbol: &str) {
        #[cfg(feature = "cns-metrics")]
        {
            use janus_cns::metrics::METRICS_REGISTRY;

            METRICS_REGISTRY
                .exchange_message_total
                .with_label_values(&[self.exchange.as_str(), channel, symbol])
                .inc();
        }

        // If CNS metrics feature is disabled, this is a no-op
        #[cfg(not(feature = "cns-metrics"))]
        {
            let _ = (channel, symbol);
        }
    }

    /// Record a message parse error
    ///
    /// Increments the `janus_exchange_message_parse_errors_total` counter with labels:
    /// - `exchange`: Exchange name
    /// - `reason`: Error reason (e.g., "invalid_json", "missing_field", "unknown_type")
    ///
    /// # Arguments
    ///
    /// * `reason` - Error reason/category
    ///
    /// # Example
    ///
    /// ```
    /// # use janus_exchanges::CNSReporter;
    /// let reporter = CNSReporter::new("kraken");
    /// reporter.record_parse_error("invalid_json");
    /// reporter.record_parse_error("missing_field");
    /// ```
    pub fn record_parse_error(&self, reason: &str) {
        #[cfg(feature = "cns-metrics")]
        {
            use janus_cns::metrics::METRICS_REGISTRY;

            METRICS_REGISTRY
                .exchange_message_parse_errors
                .with_label_values(&[self.exchange.as_str(), reason])
                .inc();
        }

        #[cfg(not(feature = "cns-metrics"))]
        {
            let _ = reason;
        }
    }

    /// Record message processing latency
    ///
    /// Observes latency in the `janus_exchange_latency_seconds` histogram with labels:
    /// - `exchange`: Exchange name
    /// - `channel`: Channel/stream name
    ///
    /// # Arguments
    ///
    /// * `channel` - Channel or stream name
    /// * `duration` - Processing duration
    ///
    /// # Example
    ///
    /// ```
    /// # use janus_exchanges::CNSReporter;
    /// # use std::time::{Duration, Instant};
    /// let reporter = CNSReporter::new("okx");
    /// let start = Instant::now();
    /// // ... process message ...
    /// reporter.record_latency("trades", start.elapsed());
    /// ```
    pub fn record_latency(&self, channel: &str, duration: Duration) {
        #[cfg(feature = "cns-metrics")]
        {
            use janus_cns::metrics::METRICS_REGISTRY;

            METRICS_REGISTRY
                .exchange_latency_seconds
                .with_label_values(&[self.exchange.as_str(), channel])
                .observe(duration.as_secs_f64());
        }

        #[cfg(not(feature = "cns-metrics"))]
        {
            let _ = (channel, duration);
        }
    }

    /// Update exchange health status
    ///
    /// Sets the `janus_exchange_health_status` gauge with label:
    /// - `exchange`: Exchange name
    ///
    /// Values:
    /// - `1.0` = Healthy
    /// - `0.5` = Degraded
    /// - `0.0` = Down
    ///
    /// # Arguments
    ///
    /// * `status` - Exchange health status
    ///
    /// # Example
    ///
    /// ```
    /// # use janus_exchanges::{CNSReporter, health::ExchangeHealthStatus};
    /// let reporter = CNSReporter::new("binance");
    /// reporter.update_health(ExchangeHealthStatus::Healthy);
    /// reporter.update_health(ExchangeHealthStatus::Degraded);
    /// reporter.update_health(ExchangeHealthStatus::Down);
    /// ```
    pub fn update_health(&self, status: ExchangeHealthStatus) {
        #[cfg(feature = "cns-metrics")]
        {
            use janus_cns::metrics::METRICS_REGISTRY;

            let value = match status {
                ExchangeHealthStatus::Healthy => 1.0,
                ExchangeHealthStatus::Degraded => 0.5,
                ExchangeHealthStatus::Down => 0.0,
                ExchangeHealthStatus::Unknown => 0.25,
            };

            METRICS_REGISTRY
                .exchange_health_status
                .with_label_values(&[self.exchange.as_str()])
                .set(value);
        }

        #[cfg(not(feature = "cns-metrics"))]
        {
            let _ = status;
        }
    }

    /// Get the exchange name for this reporter
    pub fn exchange(&self) -> &str {
        &self.exchange
    }
}

impl Clone for CNSReporter {
    fn clone(&self) -> Self {
        Self {
            exchange: self.exchange.clone(),
        }
    }
}

impl std::fmt::Debug for CNSReporter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CNSReporter")
            .field("exchange", &self.exchange)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reporter_creation() {
        let reporter = CNSReporter::new("binance");
        assert_eq!(reporter.exchange(), "binance");

        let reporter = CNSReporter::new("COINBASE");
        assert_eq!(reporter.exchange(), "coinbase");
    }

    #[test]
    fn test_reporter_clone() {
        let reporter1 = CNSReporter::new("kraken");
        let reporter2 = reporter1.clone();
        assert_eq!(reporter1.exchange(), reporter2.exchange());
    }

    #[test]
    fn test_record_message_no_panic() {
        let reporter = CNSReporter::new("okx");
        // Should not panic even without CNS feature enabled
        reporter.record_message("trades", "BTC-USDT");
        reporter.record_message("ticker", "ETH-USDT");
    }

    #[test]
    fn test_record_parse_error_no_panic() {
        let reporter = CNSReporter::new("binance");
        reporter.record_parse_error("invalid_json");
        reporter.record_parse_error("missing_field");
    }

    #[test]
    fn test_record_latency_no_panic() {
        let reporter = CNSReporter::new("coinbase");
        reporter.record_latency("level2", Duration::from_millis(5));
        reporter.record_latency("trades", Duration::from_micros(500));
    }

    #[test]
    fn test_update_health_no_panic() {
        let reporter = CNSReporter::new("kraken");
        reporter.update_health(ExchangeHealthStatus::Healthy);
        reporter.update_health(ExchangeHealthStatus::Degraded);
        reporter.update_health(ExchangeHealthStatus::Down);
    }

    #[test]
    fn test_debug_format() {
        let reporter = CNSReporter::new("okx");
        let debug_str = format!("{:?}", reporter);
        assert!(debug_str.contains("CNSReporter"));
        assert!(debug_str.contains("okx"));
    }
}
