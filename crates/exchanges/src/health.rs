//! Exchange Health Monitoring
//!
//! This module provides health checking capabilities for exchange connections,
//! tracking connectivity, latency, and message processing statistics.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Health status for an exchange connection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Connection is healthy and receiving data
    Healthy,
    /// Connection is degraded (high latency, packet loss)
    Degraded,
    /// Connection is down or not responding
    Down,
    /// Connection status is unknown (just started)
    Unknown,
}

/// Type alias for CNSReporter compatibility
pub type ExchangeHealthStatus = HealthStatus;

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Down => write!(f, "down"),
            HealthStatus::Unknown => write!(f, "unknown"),
        }
    }
}

/// Health metrics for an exchange
#[derive(Debug, Clone)]
pub struct ExchangeHealth {
    /// Current health status
    pub status: HealthStatus,
    /// Last time data was received
    pub last_message_at: Option<Instant>,
    /// Average message latency (milliseconds)
    pub avg_latency_ms: f64,
    /// Messages received in last minute
    pub messages_per_minute: u64,
    /// Total messages received
    pub total_messages: u64,
    /// Total parse errors
    pub parse_errors: u64,
    /// Connection uptime
    pub uptime: Duration,
    /// Last error message
    pub last_error: Option<String>,
}

impl Default for ExchangeHealth {
    fn default() -> Self {
        Self {
            status: HealthStatus::Unknown,
            last_message_at: None,
            avg_latency_ms: 0.0,
            messages_per_minute: 0,
            total_messages: 0,
            parse_errors: 0,
            uptime: Duration::from_secs(0),
            last_error: None,
        }
    }
}

impl ExchangeHealth {
    /// Calculate overall health score (0.0 to 1.0)
    pub fn health_score(&self) -> f64 {
        match self.status {
            HealthStatus::Healthy => 1.0,
            HealthStatus::Degraded => 0.5,
            HealthStatus::Down => 0.0,
            HealthStatus::Unknown => 0.25,
        }
    }

    /// Check if connection is considered stale
    pub fn is_stale(&self, threshold: Duration) -> bool {
        match self.last_message_at {
            Some(last) => last.elapsed() > threshold,
            None => true,
        }
    }

    /// Get error rate (errors per 1000 messages)
    pub fn error_rate(&self) -> f64 {
        if self.total_messages == 0 {
            return 0.0;
        }
        (self.parse_errors as f64 / self.total_messages as f64) * 1000.0
    }
}

/// Health checker for exchange connections
pub struct HealthChecker {
    exchanges: Arc<RwLock<HashMap<String, ExchangeHealth>>>,
    stale_threshold: Duration,
    degraded_latency_ms: f64,
    start_time: Instant,
}

impl HealthChecker {
    /// Create a new health checker
    pub fn new() -> Self {
        Self {
            exchanges: Arc::new(RwLock::new(HashMap::new())),
            stale_threshold: Duration::from_secs(10),
            degraded_latency_ms: 100.0,
            start_time: Instant::now(),
        }
    }

    /// Create health checker with custom thresholds
    pub fn with_thresholds(stale_threshold: Duration, degraded_latency_ms: f64) -> Self {
        Self {
            exchanges: Arc::new(RwLock::new(HashMap::new())),
            stale_threshold,
            degraded_latency_ms,
            start_time: Instant::now(),
        }
    }

    /// Record a successful message received
    pub async fn record_message(&self, exchange: &str, latency_ms: Option<f64>) {
        let mut exchanges = self.exchanges.write().await;
        let health = exchanges.entry(exchange.to_string()).or_default();

        health.last_message_at = Some(Instant::now());
        health.total_messages += 1;
        health.uptime = self.start_time.elapsed();

        // Update average latency with exponential moving average
        if let Some(lat) = latency_ms {
            if health.total_messages == 1 {
                health.avg_latency_ms = lat;
            } else {
                let alpha = 0.1; // Smoothing factor
                health.avg_latency_ms = alpha * lat + (1.0 - alpha) * health.avg_latency_ms;
            }
        }

        // Update health status
        health.status = if health.is_stale(self.stale_threshold) {
            HealthStatus::Down
        } else if health.avg_latency_ms > self.degraded_latency_ms {
            HealthStatus::Degraded
        } else if health.error_rate() > 10.0 {
            // More than 10 errors per 1000 messages
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        };
    }

    /// Record a parse error
    pub async fn record_error(&self, exchange: &str, error: String) {
        let mut exchanges = self.exchanges.write().await;
        let health = exchanges.entry(exchange.to_string()).or_default();

        health.parse_errors += 1;
        health.last_error = Some(error);

        // Update status based on error rate
        if health.error_rate() > 10.0 {
            health.status = HealthStatus::Degraded;
        }
    }

    /// Get health status for an exchange
    pub async fn get_health(&self, exchange: &str) -> Option<ExchangeHealth> {
        let exchanges = self.exchanges.read().await;
        exchanges.get(exchange).cloned()
    }

    /// Get health status for all exchanges
    pub async fn get_all_health(&self) -> HashMap<String, ExchangeHealth> {
        let exchanges = self.exchanges.read().await;
        exchanges.clone()
    }

    /// Check if all exchanges are healthy
    pub async fn all_healthy(&self) -> bool {
        let exchanges = self.exchanges.read().await;
        exchanges
            .values()
            .all(|h| h.status == HealthStatus::Healthy)
    }

    /// Get overall system health score (0.0 to 1.0)
    pub async fn system_health_score(&self) -> f64 {
        let exchanges = self.exchanges.read().await;
        if exchanges.is_empty() {
            return 0.0;
        }

        let total_score: f64 = exchanges.values().map(|h| h.health_score()).sum();
        total_score / exchanges.len() as f64
    }

    /// Update stale connections status
    pub async fn update_stale_status(&self) {
        let mut exchanges = self.exchanges.write().await;
        for health in exchanges.values_mut() {
            if health.is_stale(self.stale_threshold) {
                health.status = HealthStatus::Down;
            }
        }
    }

    /// Reset health stats for an exchange
    pub async fn reset(&self, exchange: &str) {
        let mut exchanges = self.exchanges.write().await;
        exchanges.remove(exchange);
    }

    /// Reset all health stats
    pub async fn reset_all(&self) {
        let mut exchanges = self.exchanges.write().await;
        exchanges.clear();
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_checker() {
        let checker = HealthChecker::new();

        // Record a message
        checker.record_message("binance", Some(5.0)).await;

        let health = checker.get_health("binance").await.unwrap();
        assert_eq!(health.status, HealthStatus::Healthy);
        assert_eq!(health.total_messages, 1);
        assert_eq!(health.avg_latency_ms, 5.0);
    }

    #[tokio::test]
    async fn test_degraded_latency() {
        let checker = HealthChecker::with_thresholds(Duration::from_secs(10), 10.0);

        // Record high latency message
        checker.record_message("binance", Some(50.0)).await;

        let health = checker.get_health("binance").await.unwrap();
        assert_eq!(health.status, HealthStatus::Degraded);
    }

    #[tokio::test]
    async fn test_error_tracking() {
        let checker = HealthChecker::new();

        // Record messages and errors
        for _ in 0..100 {
            checker.record_message("binance", Some(5.0)).await;
        }

        for _ in 0..2 {
            checker
                .record_error("binance", "Parse error".to_string())
                .await;
        }

        let health = checker.get_health("binance").await.unwrap();
        assert_eq!(health.parse_errors, 2);
        assert!(health.error_rate() > 0.0);
    }

    #[tokio::test]
    async fn test_system_health_score() {
        let checker = HealthChecker::new();

        checker.record_message("binance", Some(5.0)).await;
        checker.record_message("coinbase", Some(5.0)).await;

        let score = checker.system_health_score().await;
        assert_eq!(score, 1.0); // Both healthy
    }

    #[test]
    fn test_health_score_calculation() {
        let health = ExchangeHealth {
            status: HealthStatus::Healthy,
            ..Default::default()
        };
        assert_eq!(health.health_score(), 1.0);

        let health = ExchangeHealth {
            status: HealthStatus::Degraded,
            ..Default::default()
        };
        assert_eq!(health.health_score(), 0.5);

        let health = ExchangeHealth {
            status: HealthStatus::Down,
            ..Default::default()
        };
        assert_eq!(health.health_score(), 0.0);
    }
}
