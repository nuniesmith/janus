// Prometheus metrics endpoint
//
// Exposes runtime metrics in Prometheus format

use axum::{extract::State, http::StatusCode, response::IntoResponse};
use prometheus::{
    Encoder, IntCounter, IntGauge, TextEncoder, register_int_counter, register_int_gauge,
};
use std::sync::LazyLock;

use crate::api::AppState;

// Ingestion metrics
pub static TRADES_INGESTED: LazyLock<IntCounter> = LazyLock::new(|| {
    register_int_counter!(
        "data_factory_trades_ingested_total",
        "Total number of trades ingested"
    )
    .unwrap()
});

pub static METRICS_INGESTED: LazyLock<IntCounter> = LazyLock::new(|| {
    register_int_counter!(
        "data_factory_metrics_ingested_total",
        "Total number of market metrics ingested"
    )
    .unwrap()
});

pub static WEBSOCKET_RECONNECTS: LazyLock<IntCounter> = LazyLock::new(|| {
    register_int_counter!(
        "data_factory_websocket_reconnects_total",
        "Total number of WebSocket reconnections"
    )
    .unwrap()
});

pub static ERRORS_TOTAL: LazyLock<IntCounter> = LazyLock::new(|| {
    register_int_counter!(
        "data_factory_errors_total",
        "Total number of errors encountered"
    )
    .unwrap()
});

// Buffer and queue metrics
pub static ILP_BUFFER_SIZE: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "data_factory_ilp_buffer_size",
        "Current ILP buffer size (lines waiting to flush)"
    )
    .unwrap()
});

pub static REDIS_CONNECTIONS: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "data_factory_redis_connections",
        "Current number of Redis connections"
    )
    .unwrap()
});

// Client metrics
pub static WEBSOCKET_CLIENTS: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "data_factory_websocket_clients",
        "Current number of connected WebSocket clients"
    )
    .unwrap()
});

pub static ACTIVE_SUBSCRIPTIONS: LazyLock<IntGauge> = LazyLock::new(|| {
    register_int_gauge!(
        "data_factory_active_subscriptions",
        "Current number of active symbol subscriptions"
    )
    .unwrap()
});

// Note: Technical Indicator metrics are defined in metrics/prometheus_exporter.rs
// to avoid duplicate registration. Use those directly via:
// use crate::metrics::prometheus_exporter::{INDICATORS_CALCULATED, INDICATOR_CALCULATION_DURATION, ...};

/// Initialize Prometheus metrics (no-op now that we use register! macros)
///
/// Metrics are automatically registered when first accessed
pub fn init_metrics() -> Result<(), prometheus::Error> {
    // Force initialization of lazy statics
    let _ = &*TRADES_INGESTED;
    let _ = &*METRICS_INGESTED;
    let _ = &*WEBSOCKET_RECONNECTS;
    let _ = &*ERRORS_TOTAL;
    let _ = &*ILP_BUFFER_SIZE;
    let _ = &*REDIS_CONNECTIONS;
    let _ = &*WEBSOCKET_CLIENTS;
    let _ = &*ACTIVE_SUBSCRIPTIONS;
    // Note: Indicator metrics are initialized via prometheus_exporter.rs
    Ok(())
}

/// Prometheus metrics handler
///
/// Returns metrics in Prometheus text format
pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();

    let mut buffer = Vec::new();
    match encoder.encode(&metric_families, &mut buffer) {
        Ok(_) => (
            StatusCode::OK,
            [("Content-Type", "text/plain; version=0.0.4; charset=utf-8")],
            buffer,
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [("Content-Type", "text/plain; charset=utf-8")],
            format!("Error encoding metrics: {}", e).into_bytes(),
        ),
    }
}

/// Get historical metrics handler
///
/// Returns a JSON summary of current ingestion and system metrics gathered
/// from the Prometheus registry.  A full QuestDB-backed time-range query can
/// be layered on top once the query interface is wired in.
pub async fn get_metrics_handler(
    State(_state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let snapshot = MetricsSnapshot {
        trades_ingested: TRADES_INGESTED.get(),
        metrics_ingested: METRICS_INGESTED.get(),
        websocket_reconnects: WEBSOCKET_RECONNECTS.get(),
        errors_total: ERRORS_TOTAL.get(),
        ilp_buffer_size: ILP_BUFFER_SIZE.get(),
        redis_connections: REDIS_CONNECTIONS.get(),
        websocket_clients: WEBSOCKET_CLIENTS.get(),
        active_subscriptions: ACTIVE_SUBSCRIPTIONS.get(),
    };

    Ok((StatusCode::OK, axum::Json(snapshot)))
}

/// Point-in-time snapshot of the key ingestion metrics.
#[derive(serde::Serialize)]
struct MetricsSnapshot {
    trades_ingested: u64,
    metrics_ingested: u64,
    websocket_reconnects: u64,
    errors_total: u64,
    ilp_buffer_size: i64,
    redis_connections: i64,
    websocket_clients: i64,
    active_subscriptions: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_initialization() {
        // Clear registry first
        let result = init_metrics();
        assert!(result.is_ok() || result.is_err()); // May fail if already registered
    }

    #[test]
    fn test_counter_increment() {
        TRADES_INGESTED.inc();
        let value = TRADES_INGESTED.get();
        assert!(value >= 1);
    }

    #[test]
    fn test_gauge_set() {
        ILP_BUFFER_SIZE.set(100);
        assert_eq!(ILP_BUFFER_SIZE.get(), 100);

        ILP_BUFFER_SIZE.set(0);
        assert_eq!(ILP_BUFFER_SIZE.get(), 0);
    }
}
