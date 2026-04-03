//! # JANUS API Module
//!
//! This module contains the gRPC and REST API implementations for the JANUS service.
//! Full implementation will be completed in Week 4.
//!
//! ## API Endpoints (Planned)
//!
//! ### gRPC
//! - `GenerateSignal`: Generate a single signal from market data
//! - `GenerateSignalBatch`: Generate multiple signals in batch
//! - `GetSignalHistory`: Retrieve historical signals
//! - `StreamSignals`: Real-time signal streaming
//!
//! ### REST
//! - `POST /api/v1/signals/generate`: Generate signal
//! - `POST /api/v1/signals/batch`: Batch signal generation
//! - `GET /api/v1/signals/history`: Get signal history
//! - `GET /api/v1/health`: Health check
//! - `GET /api/v1/metrics`: Prometheus metrics

pub mod brain_rest;
pub mod feedback_grpc;
pub mod grpc;
pub mod rest;
pub mod risk_rest;
pub mod server;

use crate::signal::{SignalBatch, TradingSignal};
use serde::{Deserialize, Serialize};

/// API request for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateSignalRequest {
    pub symbol: String,
    pub timeframe: String,
    pub indicators: IndicatorInput,
}

/// Indicator input for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndicatorInput {
    pub ema_fast: Option<f64>,
    pub ema_slow: Option<f64>,
    pub rsi: Option<f64>,
    pub macd_line: Option<f64>,
    pub macd_signal: Option<f64>,
    pub macd_histogram: Option<f64>,
    pub atr: Option<f64>,
}

/// API response for signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateSignalResponse {
    pub success: bool,
    pub signal: Option<TradingSignal>,
    pub error: Option<String>,
}

/// API request for batch signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateBatchRequest {
    pub requests: Vec<GenerateSignalRequest>,
}

/// API response for batch signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateBatchResponse {
    pub success: bool,
    pub batch: Option<SignalBatch>,
    pub error: Option<String>,
}

/// Health check response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: String,
    pub service: String,
    pub version: String,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_signal_request_serialization() {
        let request = GenerateSignalRequest {
            symbol: "BTC/USD".to_string(),
            timeframe: "1h".to_string(),
            indicators: IndicatorInput {
                ema_fast: Some(51.0),
                ema_slow: Some(50.0),
                rsi: Some(35.0),
                macd_line: None,
                macd_signal: None,
                macd_histogram: None,
                atr: Some(100.0),
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("BTC/USD"));
        assert!(json.contains("1h"));
    }

    #[test]
    fn test_health_check_response() {
        let response = HealthCheckResponse {
            status: "healthy".to_string(),
            service: "janus".to_string(),
            version: "0.1.0".to_string(),
            uptime_seconds: 3600,
        };

        assert_eq!(response.status, "healthy");
        assert_eq!(response.uptime_seconds, 3600);
    }
}
