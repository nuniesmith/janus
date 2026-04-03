//! Structured Logging with Correlation IDs
//!
//! This module provides structured logging utilities with correlation ID support
//! for tracing requests through the backfill orchestration system.
//!
//! ## Features
//!
//! - **Correlation IDs**: Track requests across gap detection → scheduler → backfill → QuestDB
//! - **Structured Fields**: Consistent field names for exchange, symbol, timestamps, etc.
//! - **Context Propagation**: Thread-local storage for correlation IDs
//! - **JSON Formatting**: Machine-parsable logs for aggregation
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_data::logging::{CorrelationId, log_backfill_started, log_backfill_completed};
//!
//! async fn process_backfill() {
//!     let correlation_id = CorrelationId::new();
//!
//!     log_backfill_started(
//!         &correlation_id,
//!         "binance",
//!         "BTCUSD",
//!         start_time,
//!         end_time,
//!         5000,
//!     );
//!
//!     // ... perform backfill ...
//!
//!     log_backfill_completed(
//!         &correlation_id,
//!         "binance",
//!         "BTCUSD",
//!         duration,
//!         trades_filled,
//!     );
//! }
//! ```

#![allow(dead_code)] // Public API - functions used by consumers

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use tracing::{error, info, warn};
use uuid::Uuid;

// ============================================================================
// Correlation ID
// ============================================================================

/// Correlation ID for tracking requests through the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CorrelationId(Uuid);

impl CorrelationId {
    /// Create a new random correlation ID
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Parse a correlation ID from a string
    pub fn parse(s: &str) -> Result<Self, uuid::Error> {
        Ok(Self(Uuid::parse_str(s)?))
    }

    /// Get the underlying UUID
    pub fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Get the correlation ID as a string
    pub fn as_str(&self) -> String {
        self.0.to_string()
    }
}

impl Default for CorrelationId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CorrelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for CorrelationId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

// ============================================================================
// Structured Log Fields
// ============================================================================

/// Standard structured fields for logging
#[derive(Debug, Clone, Serialize)]
pub struct LogFields {
    pub correlation_id: String,
    pub exchange: String,
    pub symbol: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trades_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

// ============================================================================
// Gap Detection Logging
// ============================================================================

/// Log when a gap is detected
pub fn log_gap_detected(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    estimated_trades: usize,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        start_time = %start_time,
        end_time = %end_time,
        estimated_trades = estimated_trades,
        gap_duration_seconds = (end_time - start_time).num_seconds(),
        "Gap detected"
    );
}

/// Log when a gap is filtered (not submitted to backfill)
pub fn log_gap_filtered(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    reason: &str,
    gap_duration_seconds: i64,
    estimated_trades: usize,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        reason = reason,
        gap_duration_seconds = gap_duration_seconds,
        estimated_trades = estimated_trades,
        "Gap filtered (not submitted for backfill)"
    );
}

/// Log when a gap is deduplicated
pub fn log_gap_deduplicated(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    start_time: DateTime<Utc>,
) {
    warn!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        start_time = %start_time,
        "Gap deduplicated (already in queue)"
    );
}

// ============================================================================
// Backfill Scheduler Logging
// ============================================================================

/// Log when a backfill is submitted to the scheduler
pub fn log_backfill_submitted(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    priority: u64,
    queue_size: usize,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        priority = priority,
        queue_size = queue_size,
        "Backfill submitted to scheduler"
    );
}

/// Log when a backfill is started
pub fn log_backfill_started(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
    estimated_trades: usize,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        start_time = %start_time,
        end_time = %end_time,
        estimated_trades = estimated_trades,
        time_range_seconds = (end_time - start_time).num_seconds(),
        "Backfill started"
    );
}

/// Log when a backfill is completed successfully
pub fn log_backfill_completed(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    duration_ms: u64,
    trades_filled: usize,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        duration_ms = duration_ms,
        trades_filled = trades_filled,
        throughput_trades_per_sec = (trades_filled as f64 / (duration_ms as f64 / 1000.0)),
        "Backfill completed successfully"
    );
}

/// Log when a backfill fails
pub fn log_backfill_failed(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    error: &str,
) {
    error!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        error = error,
        "Backfill failed"
    );
}

/// Log when a backfill is retried
pub fn log_backfill_retry(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    retry_count: u32,
    backoff_ms: u64,
) {
    warn!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        retry_count = retry_count,
        backoff_ms = backoff_ms,
        "Backfill retry scheduled"
    );
}

/// Log when a backfill exceeds max retries
pub fn log_backfill_max_retries_exceeded(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    retry_count: u32,
    last_error: &str,
) {
    error!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        retry_count = retry_count,
        last_error = last_error,
        "Backfill exceeded max retries - giving up"
    );
}

// ============================================================================
// Throttle & Lock Logging
// ============================================================================

/// Log when a backfill is throttled
pub fn log_backfill_throttled(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    reason: &str,
    running_backfills: usize,
    max_concurrent: usize,
) {
    warn!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        reason = reason,
        running_backfills = running_backfills,
        max_concurrent = max_concurrent,
        "Backfill throttled"
    );
}

/// Log when a backfill lock is acquired
pub fn log_lock_acquired(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    lock_key: &str,
    ttl_seconds: u64,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        lock_key = lock_key,
        ttl_seconds = ttl_seconds,
        "Distributed lock acquired"
    );
}

/// Log when a backfill lock acquisition fails
pub fn log_lock_failed(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    lock_key: &str,
    reason: &str,
) {
    warn!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        lock_key = lock_key,
        reason = reason,
        "Failed to acquire distributed lock"
    );
}

/// Log when a backfill lock is released
pub fn log_lock_released(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    lock_key: &str,
    duration_ms: u64,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        lock_key = lock_key,
        duration_ms = duration_ms,
        "Distributed lock released"
    );
}

// ============================================================================
// QuestDB Write Logging
// ============================================================================

/// Log when trades are written to QuestDB
pub fn log_questdb_write(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    trades_count: usize,
    write_duration_ms: u64,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        trades_count = trades_count,
        write_duration_ms = write_duration_ms,
        throughput_trades_per_sec = (trades_count as f64 / (write_duration_ms as f64 / 1000.0)),
        "Trades written to QuestDB"
    );
}

/// Log when a QuestDB write fails
pub fn log_questdb_write_error(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    trades_count: usize,
    error: &str,
) {
    error!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        trades_count = trades_count,
        error = error,
        "QuestDB write failed"
    );
}

// ============================================================================
// Circuit Breaker Logging
// ============================================================================

/// Log when a circuit breaker opens
pub fn log_circuit_breaker_opened(
    correlation_id: &CorrelationId,
    exchange: &str,
    failure_count: usize,
    threshold: usize,
) {
    error!(
        correlation_id = %correlation_id,
        exchange = exchange,
        failure_count = failure_count,
        threshold = threshold,
        "Circuit breaker opened"
    );
}

/// Log when a circuit breaker transitions to half-open
pub fn log_circuit_breaker_half_open(correlation_id: &CorrelationId, exchange: &str) {
    warn!(
        correlation_id = %correlation_id,
        exchange = exchange,
        "Circuit breaker half-open (testing recovery)"
    );
}

/// Log when a circuit breaker closes
pub fn log_circuit_breaker_closed(correlation_id: &CorrelationId, exchange: &str) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        "Circuit breaker closed (recovered)"
    );
}

/// Log when circuit breaker state is checked
pub fn log_circuit_breaker_checked(
    correlation_id: &CorrelationId,
    exchange: &str,
    state: janus_rate_limiter::circuit_breaker::CircuitState,
) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        state = ?state,
        "Circuit breaker state checked"
    );
}

/// Log when an exchange request is made
pub fn log_exchange_request(correlation_id: &CorrelationId, exchange: &str, symbol: &str) {
    info!(
        correlation_id = %correlation_id,
        exchange = exchange,
        symbol = symbol,
        "Exchange API request initiated"
    );
}

/// Log when verification is completed
pub fn log_verification_completed(
    correlation_id: &CorrelationId,
    exchange: &str,
    symbol: &str,
    verified: bool,
    rows_found: usize,
) {
    if verified {
        info!(
            correlation_id = %correlation_id,
            exchange = exchange,
            symbol = symbol,
            rows_found = rows_found,
            "Post-backfill verification completed successfully"
        );
    } else {
        warn!(
            correlation_id = %correlation_id,
            exchange = exchange,
            symbol = symbol,
            rows_found = rows_found,
            "Post-backfill verification FAILED"
        );
    }
}

// ============================================================================
// WebSocket Connection Logging
// ============================================================================

/// Log when a WebSocket connection is established
pub fn log_websocket_connected(exchange: &str, symbol: &str, reconnect_count: usize) {
    info!(
        exchange = exchange,
        symbol = symbol,
        reconnect_count = reconnect_count,
        "WebSocket connected"
    );
}

/// Log when a WebSocket connection is lost
pub fn log_websocket_disconnected(
    exchange: &str,
    symbol: &str,
    reason: &str,
    will_reconnect: bool,
) {
    warn!(
        exchange = exchange,
        symbol = symbol,
        reason = reason,
        will_reconnect = will_reconnect,
        "WebSocket disconnected"
    );
}

/// Log when a WebSocket reconnect is attempted
pub fn log_websocket_reconnecting(exchange: &str, symbol: &str, attempt: usize, backoff_ms: u64) {
    info!(
        exchange = exchange,
        symbol = symbol,
        attempt = attempt,
        backoff_ms = backoff_ms,
        "WebSocket reconnecting"
    );
}

// ============================================================================
// Performance Logging
// ============================================================================

/// Log ingestion latency metrics
pub fn log_ingestion_latency(
    exchange: &str,
    symbol: &str,
    latency_ms: u64,
    exchange_timestamp: DateTime<Utc>,
    ingestion_timestamp: DateTime<Utc>,
) {
    if latency_ms > 1000 {
        warn!(
            exchange = exchange,
            symbol = symbol,
            latency_ms = latency_ms,
            exchange_timestamp = %exchange_timestamp,
            ingestion_timestamp = %ingestion_timestamp,
            "High ingestion latency detected"
        );
    }
}

/// Log throughput metrics
pub fn log_throughput(exchange: &str, symbol: &str, trades_per_second: f64, window_seconds: u64) {
    info!(
        exchange = exchange,
        symbol = symbol,
        trades_per_second = trades_per_second,
        window_seconds = window_seconds,
        "Throughput measurement"
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correlation_id_creation() {
        let id1 = CorrelationId::new();
        let id2 = CorrelationId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_correlation_id_parse() {
        let id = CorrelationId::new();
        let id_str = id.to_string();
        let parsed = CorrelationId::parse(&id_str).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_correlation_id_display() {
        let id = CorrelationId::new();
        let display = format!("{}", id);
        assert!(!display.is_empty());
        assert!(Uuid::parse_str(&display).is_ok());
    }
}
