//! Exchange WebSocket Metrics for Prometheus
//!
//! This module provides Prometheus-compatible metrics for tracking the health
//! and performance of WebSocket connections to multiple exchanges.
//!
//! # Metrics Exported
//!
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | exchange_ws_connected | Gauge | Connection status (1=connected, 0=disconnected) |
//! | exchange_ws_messages_total | Counter | Total messages received |
//! | exchange_ws_errors_total | Counter | Total errors encountered |
//! | exchange_ws_reconnects_total | Counter | Total reconnection attempts |
//! | exchange_ws_latency_ms | Gauge | Last message latency in milliseconds |
//! | exchange_ws_subscriptions | Gauge | Number of active subscriptions |
//! | exchange_ws_last_message_timestamp | Gauge | Unix timestamp of last message |

use crate::exchanges::market_data::ExchangeId;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// Exchange Metrics
// ============================================================================

/// Metrics for a single exchange WebSocket connection
#[derive(Debug)]
pub struct ExchangeMetrics {
    /// Exchange identifier
    exchange: ExchangeId,

    /// Connection status (1 = connected, 0 = disconnected)
    connected: AtomicU64,

    /// Total messages received
    messages_total: AtomicU64,

    /// Total errors encountered
    errors_total: AtomicU64,

    /// Total reconnection attempts
    reconnects_total: AtomicU64,

    /// Last message latency in milliseconds
    latency_ms: AtomicU64,

    /// Number of active subscriptions
    subscriptions: AtomicU64,

    /// Timestamp of last message (unix millis)
    last_message_timestamp: AtomicU64,

    /// Messages per second (rolling window)
    messages_per_second: RwLock<f64>,

    /// Connection start time
    connection_start: RwLock<Option<Instant>>,

    /// Subscription details: channel -> symbols count
    subscription_details: RwLock<HashMap<String, usize>>,
}

impl ExchangeMetrics {
    /// Create new metrics for an exchange
    pub fn new(exchange: ExchangeId) -> Self {
        Self {
            exchange,
            connected: AtomicU64::new(0),
            messages_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            reconnects_total: AtomicU64::new(0),
            latency_ms: AtomicU64::new(0),
            subscriptions: AtomicU64::new(0),
            last_message_timestamp: AtomicU64::new(0),
            messages_per_second: RwLock::new(0.0),
            connection_start: RwLock::new(None),
            subscription_details: RwLock::new(HashMap::new()),
        }
    }

    /// Get exchange identifier
    pub fn exchange(&self) -> ExchangeId {
        self.exchange
    }

    /// Record connection established
    pub fn record_connected(&self) {
        self.connected.store(1, Ordering::SeqCst);
        *self.connection_start.write() = Some(Instant::now());
    }

    /// Record disconnection
    pub fn record_disconnected(&self) {
        self.connected.store(0, Ordering::SeqCst);
        *self.connection_start.write() = None;
    }

    /// Record a message received
    pub fn record_message(&self) {
        self.messages_total.fetch_add(1, Ordering::SeqCst);
        self.last_message_timestamp.store(
            chrono::Utc::now().timestamp_millis() as u64,
            Ordering::SeqCst,
        );
    }

    /// Record a message with latency
    pub fn record_message_with_latency(&self, latency_ms: u64) {
        self.record_message();
        self.latency_ms.store(latency_ms, Ordering::SeqCst);
    }

    /// Record an error
    pub fn record_error(&self) {
        self.errors_total.fetch_add(1, Ordering::SeqCst);
    }

    /// Record a reconnection attempt
    pub fn record_reconnect(&self) {
        self.reconnects_total.fetch_add(1, Ordering::SeqCst);
    }

    /// Update subscription count
    pub fn set_subscription_count(&self, count: u64) {
        self.subscriptions.store(count, Ordering::SeqCst);
    }

    /// Add subscription details
    pub fn add_subscription(&self, channel: &str, symbol_count: usize) {
        let mut details = self.subscription_details.write();
        details.insert(channel.to_string(), symbol_count);
        let total: usize = details.values().sum();
        self.subscriptions.store(total as u64, Ordering::SeqCst);
    }

    /// Remove subscription
    pub fn remove_subscription(&self, channel: &str) {
        let mut details = self.subscription_details.write();
        details.remove(channel);
        let total: usize = details.values().sum();
        self.subscriptions.store(total as u64, Ordering::SeqCst);
    }

    /// Update messages per second (call periodically)
    pub fn update_rate(&self, messages_in_window: u64, window_seconds: f64) {
        if window_seconds > 0.0 {
            let rate = messages_in_window as f64 / window_seconds;
            *self.messages_per_second.write() = rate;
        }
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst) == 1
    }

    /// Get total messages
    pub fn total_messages(&self) -> u64 {
        self.messages_total.load(Ordering::SeqCst)
    }

    /// Get total errors
    pub fn total_errors(&self) -> u64 {
        self.errors_total.load(Ordering::SeqCst)
    }

    /// Get total reconnects
    pub fn total_reconnects(&self) -> u64 {
        self.reconnects_total.load(Ordering::SeqCst)
    }

    /// Get current latency
    pub fn current_latency_ms(&self) -> u64 {
        self.latency_ms.load(Ordering::SeqCst)
    }

    /// Get subscription count
    pub fn subscription_count(&self) -> u64 {
        self.subscriptions.load(Ordering::SeqCst)
    }

    /// Get last message timestamp
    pub fn last_message_timestamp(&self) -> u64 {
        self.last_message_timestamp.load(Ordering::SeqCst)
    }

    /// Get messages per second
    pub fn messages_per_second(&self) -> f64 {
        *self.messages_per_second.read()
    }

    /// Get connection uptime in seconds
    pub fn uptime_seconds(&self) -> Option<f64> {
        self.connection_start
            .read()
            .map(|start| start.elapsed().as_secs_f64())
    }

    /// Export metrics in Prometheus format
    pub fn to_prometheus(&self) -> String {
        let exchange = self.exchange.name().to_lowercase();
        let mut output = String::new();

        // Connection status
        output.push_str(&format!(
            "exchange_ws_connected{{exchange=\"{}\"}} {}\n",
            exchange,
            self.connected.load(Ordering::SeqCst)
        ));

        // Messages total
        output.push_str(&format!(
            "exchange_ws_messages_total{{exchange=\"{}\"}} {}\n",
            exchange,
            self.messages_total.load(Ordering::SeqCst)
        ));

        // Errors total
        output.push_str(&format!(
            "exchange_ws_errors_total{{exchange=\"{}\"}} {}\n",
            exchange,
            self.errors_total.load(Ordering::SeqCst)
        ));

        // Reconnects total
        output.push_str(&format!(
            "exchange_ws_reconnects_total{{exchange=\"{}\"}} {}\n",
            exchange,
            self.reconnects_total.load(Ordering::SeqCst)
        ));

        // Latency
        output.push_str(&format!(
            "exchange_ws_latency_ms{{exchange=\"{}\"}} {}\n",
            exchange,
            self.latency_ms.load(Ordering::SeqCst)
        ));

        // Subscriptions
        output.push_str(&format!(
            "exchange_ws_subscriptions{{exchange=\"{}\"}} {}\n",
            exchange,
            self.subscriptions.load(Ordering::SeqCst)
        ));

        // Last message timestamp
        output.push_str(&format!(
            "exchange_ws_last_message_timestamp{{exchange=\"{}\"}} {}\n",
            exchange,
            self.last_message_timestamp.load(Ordering::SeqCst)
        ));

        // Messages per second
        output.push_str(&format!(
            "exchange_ws_messages_per_second{{exchange=\"{}\"}} {:.2}\n",
            exchange,
            self.messages_per_second()
        ));

        // Uptime
        if let Some(uptime) = self.uptime_seconds() {
            output.push_str(&format!(
                "exchange_ws_uptime_seconds{{exchange=\"{}\"}} {:.2}\n",
                exchange, uptime
            ));
        }

        output
    }
}

// ============================================================================
// Aggregated Metrics
// ============================================================================

/// Aggregated metrics for all exchanges
#[derive(Debug, Default)]
pub struct ExchangeMetricsRegistry {
    /// Metrics per exchange
    exchanges: RwLock<HashMap<ExchangeId, Arc<ExchangeMetrics>>>,
}

impl ExchangeMetricsRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            exchanges: RwLock::new(HashMap::new()),
        }
    }

    /// Register an exchange and get its metrics handle
    pub fn register(&self, exchange: ExchangeId) -> Arc<ExchangeMetrics> {
        let mut exchanges = self.exchanges.write();

        if let Some(metrics) = exchanges.get(&exchange) {
            return metrics.clone();
        }

        let metrics = Arc::new(ExchangeMetrics::new(exchange));
        exchanges.insert(exchange, metrics.clone());
        metrics
    }

    /// Get metrics for an exchange
    pub fn get(&self, exchange: ExchangeId) -> Option<Arc<ExchangeMetrics>> {
        self.exchanges.read().get(&exchange).cloned()
    }

    /// Get all registered exchanges
    pub fn exchanges(&self) -> Vec<ExchangeId> {
        self.exchanges.read().keys().copied().collect()
    }

    /// Get total connected exchanges
    pub fn connected_count(&self) -> usize {
        self.exchanges
            .read()
            .values()
            .filter(|m| m.is_connected())
            .count()
    }

    /// Get total messages across all exchanges
    pub fn total_messages(&self) -> u64 {
        self.exchanges
            .read()
            .values()
            .map(|m| m.total_messages())
            .sum()
    }

    /// Get total errors across all exchanges
    pub fn total_errors(&self) -> u64 {
        self.exchanges
            .read()
            .values()
            .map(|m| m.total_errors())
            .sum()
    }

    /// Get aggregated health status
    pub fn health_summary(&self) -> HealthSummary {
        let exchanges = self.exchanges.read();
        let total = exchanges.len();
        let connected = exchanges.values().filter(|m| m.is_connected()).count();

        let mut exchange_status = Vec::new();
        for (id, metrics) in exchanges.iter() {
            exchange_status.push(ExchangeHealthStatus {
                exchange: *id,
                connected: metrics.is_connected(),
                messages_total: metrics.total_messages(),
                errors_total: metrics.total_errors(),
                reconnects_total: metrics.total_reconnects(),
                latency_ms: metrics.current_latency_ms(),
                subscriptions: metrics.subscription_count(),
                last_message_timestamp: metrics.last_message_timestamp(),
                uptime_seconds: metrics.uptime_seconds(),
            });
        }

        HealthSummary {
            total_exchanges: total,
            connected_exchanges: connected,
            all_connected: total > 0 && connected == total,
            exchange_status,
        }
    }

    /// Export all metrics in Prometheus format
    pub fn to_prometheus(&self) -> String {
        let mut output = String::new();

        // Metric descriptions (only once)
        output.push_str("# HELP exchange_ws_connected WebSocket connection status (1=connected, 0=disconnected)\n");
        output.push_str("# TYPE exchange_ws_connected gauge\n");
        output.push_str("# HELP exchange_ws_messages_total Total WebSocket messages received\n");
        output.push_str("# TYPE exchange_ws_messages_total counter\n");
        output.push_str("# HELP exchange_ws_errors_total Total WebSocket errors\n");
        output.push_str("# TYPE exchange_ws_errors_total counter\n");
        output.push_str(
            "# HELP exchange_ws_reconnects_total Total WebSocket reconnection attempts\n",
        );
        output.push_str("# TYPE exchange_ws_reconnects_total counter\n");
        output.push_str("# HELP exchange_ws_latency_ms Last message latency in milliseconds\n");
        output.push_str("# TYPE exchange_ws_latency_ms gauge\n");
        output.push_str("# HELP exchange_ws_subscriptions Number of active subscriptions\n");
        output.push_str("# TYPE exchange_ws_subscriptions gauge\n");
        output
            .push_str("# HELP exchange_ws_last_message_timestamp Unix timestamp of last message\n");
        output.push_str("# TYPE exchange_ws_last_message_timestamp gauge\n");
        output.push_str("# HELP exchange_ws_messages_per_second Messages received per second\n");
        output.push_str("# TYPE exchange_ws_messages_per_second gauge\n");
        output.push_str("# HELP exchange_ws_uptime_seconds Connection uptime in seconds\n");
        output.push_str("# TYPE exchange_ws_uptime_seconds gauge\n");
        output.push('\n');

        // Export metrics for each exchange
        for metrics in self.exchanges.read().values() {
            output.push_str(&metrics.to_prometheus());
            output.push('\n');
        }

        // Aggregate metrics
        let health = self.health_summary();
        output
            .push_str("# HELP exchange_ws_total_exchanges Total number of registered exchanges\n");
        output.push_str("# TYPE exchange_ws_total_exchanges gauge\n");
        output.push_str(&format!(
            "exchange_ws_total_exchanges {}\n",
            health.total_exchanges
        ));
        output.push('\n');

        output.push_str("# HELP exchange_ws_connected_exchanges Number of connected exchanges\n");
        output.push_str("# TYPE exchange_ws_connected_exchanges gauge\n");
        output.push_str(&format!(
            "exchange_ws_connected_exchanges {}\n",
            health.connected_exchanges
        ));
        output.push('\n');

        output.push_str("# HELP exchange_ws_all_healthy All exchanges connected and healthy\n");
        output.push_str("# TYPE exchange_ws_all_healthy gauge\n");
        output.push_str(&format!(
            "exchange_ws_all_healthy {}\n",
            if health.all_connected { 1 } else { 0 }
        ));

        output
    }
}

/// Health summary for all exchanges
#[derive(Debug, Clone)]
pub struct HealthSummary {
    /// Total registered exchanges
    pub total_exchanges: usize,
    /// Number of connected exchanges
    pub connected_exchanges: usize,
    /// Whether all exchanges are connected
    pub all_connected: bool,
    /// Per-exchange status
    pub exchange_status: Vec<ExchangeHealthStatus>,
}

/// Health status for a single exchange
#[derive(Debug, Clone)]
pub struct ExchangeHealthStatus {
    /// Exchange identifier
    pub exchange: ExchangeId,
    /// Whether connected
    pub connected: bool,
    /// Total messages received
    pub messages_total: u64,
    /// Total errors
    pub errors_total: u64,
    /// Total reconnection attempts
    pub reconnects_total: u64,
    /// Current latency in ms
    pub latency_ms: u64,
    /// Active subscription count
    pub subscriptions: u64,
    /// Last message timestamp
    pub last_message_timestamp: u64,
    /// Connection uptime
    pub uptime_seconds: Option<f64>,
}

// ============================================================================
// Global Registry (Singleton)
// ============================================================================

use std::sync::OnceLock;

/// Global metrics registry
static GLOBAL_REGISTRY: OnceLock<ExchangeMetricsRegistry> = OnceLock::new();

/// Get the global metrics registry
pub fn global_registry() -> &'static ExchangeMetricsRegistry {
    GLOBAL_REGISTRY.get_or_init(ExchangeMetricsRegistry::new)
}

/// Register an exchange with the global registry
pub fn register_exchange(exchange: ExchangeId) -> Arc<ExchangeMetrics> {
    global_registry().register(exchange)
}

/// Get metrics for an exchange from the global registry
pub fn get_exchange_metrics(exchange: ExchangeId) -> Option<Arc<ExchangeMetrics>> {
    global_registry().get(exchange)
}

/// Export all metrics in Prometheus format
pub fn prometheus_metrics() -> String {
    global_registry().to_prometheus()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange_metrics_creation() {
        let metrics = ExchangeMetrics::new(ExchangeId::Bybit);
        assert_eq!(metrics.exchange(), ExchangeId::Bybit);
        assert!(!metrics.is_connected());
        assert_eq!(metrics.total_messages(), 0);
    }

    #[test]
    fn test_connection_tracking() {
        let metrics = ExchangeMetrics::new(ExchangeId::Kraken);

        metrics.record_connected();
        assert!(metrics.is_connected());
        assert!(metrics.uptime_seconds().is_some());

        metrics.record_disconnected();
        assert!(!metrics.is_connected());
        assert!(metrics.uptime_seconds().is_none());
    }

    #[test]
    fn test_message_tracking() {
        let metrics = ExchangeMetrics::new(ExchangeId::Binance);

        metrics.record_message();
        metrics.record_message();
        metrics.record_message();

        assert_eq!(metrics.total_messages(), 3);
        assert!(metrics.last_message_timestamp() > 0);
    }

    #[test]
    fn test_message_with_latency() {
        let metrics = ExchangeMetrics::new(ExchangeId::Bybit);

        metrics.record_message_with_latency(42);

        assert_eq!(metrics.total_messages(), 1);
        assert_eq!(metrics.current_latency_ms(), 42);
    }

    #[test]
    fn test_error_tracking() {
        let metrics = ExchangeMetrics::new(ExchangeId::Kraken);

        metrics.record_error();
        metrics.record_error();

        assert_eq!(metrics.total_errors(), 2);
    }

    #[test]
    fn test_reconnect_tracking() {
        let metrics = ExchangeMetrics::new(ExchangeId::Binance);

        metrics.record_reconnect();
        metrics.record_reconnect();
        metrics.record_reconnect();

        assert_eq!(metrics.total_reconnects(), 3);
    }

    #[test]
    fn test_subscription_tracking() {
        let metrics = ExchangeMetrics::new(ExchangeId::Bybit);

        metrics.add_subscription("ticker", 3);
        assert_eq!(metrics.subscription_count(), 3);

        metrics.add_subscription("trades", 2);
        assert_eq!(metrics.subscription_count(), 5);

        metrics.remove_subscription("ticker");
        assert_eq!(metrics.subscription_count(), 2);
    }

    #[test]
    fn test_rate_calculation() {
        let metrics = ExchangeMetrics::new(ExchangeId::Kraken);

        metrics.update_rate(100, 10.0);
        assert!((metrics.messages_per_second() - 10.0).abs() < 0.01);

        metrics.update_rate(0, 0.0); // Should not panic
    }

    #[test]
    fn test_prometheus_output() {
        let metrics = ExchangeMetrics::new(ExchangeId::Binance);
        metrics.record_connected();
        metrics.record_message();
        metrics.add_subscription("ticker", 2);

        let output = metrics.to_prometheus();

        assert!(output.contains("exchange_ws_connected{exchange=\"binance\"} 1"));
        assert!(output.contains("exchange_ws_messages_total{exchange=\"binance\"} 1"));
        assert!(output.contains("exchange_ws_subscriptions{exchange=\"binance\"} 2"));
    }

    #[test]
    fn test_registry_creation() {
        let registry = ExchangeMetricsRegistry::new();

        let bybit = registry.register(ExchangeId::Bybit);
        let kraken = registry.register(ExchangeId::Kraken);

        assert_eq!(registry.exchanges().len(), 2);
        assert!(registry.get(ExchangeId::Bybit).is_some());
        assert!(registry.get(ExchangeId::Kraken).is_some());
        assert!(registry.get(ExchangeId::Binance).is_none());

        // Same exchange returns same Arc
        let bybit2 = registry.register(ExchangeId::Bybit);
        assert!(Arc::ptr_eq(&bybit, &bybit2));

        // Test connected count
        assert_eq!(registry.connected_count(), 0);

        bybit.record_connected();
        kraken.record_connected();
        assert_eq!(registry.connected_count(), 2);
    }

    #[test]
    fn test_health_summary() {
        let registry = ExchangeMetricsRegistry::new();

        let bybit = registry.register(ExchangeId::Bybit);
        let kraken = registry.register(ExchangeId::Kraken);
        let binance = registry.register(ExchangeId::Binance);

        bybit.record_connected();
        kraken.record_connected();
        // binance disconnected

        let health = registry.health_summary();

        assert_eq!(health.total_exchanges, 3);
        assert_eq!(health.connected_exchanges, 2);
        assert!(!health.all_connected);
        assert_eq!(health.exchange_status.len(), 3);

        binance.record_connected();
        let health2 = registry.health_summary();
        assert!(health2.all_connected);
    }

    #[test]
    fn test_aggregate_metrics() {
        let registry = ExchangeMetricsRegistry::new();

        let bybit = registry.register(ExchangeId::Bybit);
        let kraken = registry.register(ExchangeId::Kraken);

        bybit.record_message();
        bybit.record_message();
        kraken.record_message();

        bybit.record_error();
        kraken.record_error();
        kraken.record_error();

        assert_eq!(registry.total_messages(), 3);
        assert_eq!(registry.total_errors(), 3);
    }

    #[test]
    fn test_registry_prometheus_output() {
        let registry = ExchangeMetricsRegistry::new();

        let bybit = registry.register(ExchangeId::Bybit);
        let kraken = registry.register(ExchangeId::Kraken);

        bybit.record_connected();
        kraken.record_connected();

        let output = registry.to_prometheus();

        // Check metric descriptions
        assert!(output.contains("# HELP exchange_ws_connected"));
        assert!(output.contains("# TYPE exchange_ws_connected gauge"));

        // Check exchange-specific metrics
        assert!(output.contains("exchange=\"bybit\""));
        assert!(output.contains("exchange=\"kraken\""));

        // Check aggregate metrics
        assert!(output.contains("exchange_ws_total_exchanges 2"));
        assert!(output.contains("exchange_ws_connected_exchanges 2"));
        assert!(output.contains("exchange_ws_all_healthy 1"));
    }

    #[test]
    fn test_global_registry() {
        // This test uses the global singleton
        let bybit = register_exchange(ExchangeId::Bybit);
        bybit.record_message();

        let retrieved = get_exchange_metrics(ExchangeId::Bybit);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().total_messages(), bybit.total_messages());

        let prometheus_output = prometheus_metrics();
        assert!(prometheus_output.contains("exchange_ws"));
    }
}
