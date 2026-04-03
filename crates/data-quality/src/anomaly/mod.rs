//! Anomaly detection for market data

use janus_core::{MarketDataEvent, Symbol};
use serde::{Deserialize, Serialize};

pub mod latency;
pub mod sequence;
pub mod statistical;

pub use latency::LatencyAnomalyDetector;
pub use sequence::SequenceAnomalyDetector;
pub use statistical::StatisticalAnomalyDetector;

use crate::Result;

/// Anomaly severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalySeverity {
    /// Low severity - informational
    Low,
    /// Medium severity - warning
    Medium,
    /// High severity - critical
    High,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize)]
pub struct AnomalyResult {
    /// Whether an anomaly was detected
    pub is_anomaly: bool,

    /// Severity of the anomaly
    pub severity: AnomalySeverity,

    /// Description of the anomaly
    pub description: String,

    /// Detector that found the anomaly
    pub detector: String,

    /// Symbol where anomaly was detected
    pub symbol: Symbol,

    /// Timestamp when anomaly was detected
    pub detected_at: i64,

    /// Anomaly score (optional, detector-specific)
    pub score: Option<f64>,
}

impl AnomalyResult {
    /// Create a result indicating no anomaly
    pub fn none(detector: impl Into<String>, symbol: Symbol) -> Self {
        Self {
            is_anomaly: false,
            severity: AnomalySeverity::Low,
            description: String::new(),
            detector: detector.into(),
            symbol,
            detected_at: chrono::Utc::now().timestamp_micros(),
            score: None,
        }
    }

    /// Create a result indicating an anomaly was detected
    pub fn detected(
        detector: impl Into<String>,
        symbol: Symbol,
        severity: AnomalySeverity,
        description: impl Into<String>,
    ) -> Self {
        Self {
            is_anomaly: true,
            severity,
            description: description.into(),
            detector: detector.into(),
            symbol,
            detected_at: chrono::Utc::now().timestamp_micros(),
            score: None,
        }
    }

    /// Add a score to the result
    pub fn with_score(mut self, score: f64) -> Self {
        self.score = Some(score);
        self
    }
}

/// Trait for anomaly detectors
pub trait AnomalyDetector: Send + Sync {
    /// Name of the detector
    fn name(&self) -> &str;

    /// Detect anomalies in a market data event
    fn detect(&mut self, event: &MarketDataEvent) -> Result<AnomalyResult>;

    /// Reset detector state
    fn reset(&mut self);
}
