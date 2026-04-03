//! Alertmanager client for pushing trade signals as alerts.
//!
//! Janus-execution formats trade signals into Alertmanager-compatible
//! alert payloads and POSTs them to the Alertmanager API. This is the
//! handoff point between Janus (signal generation) and the Python side
//! (signal monitoring + execution routing).
//!
//! Alert format:
//!   Labels (for routing):
//!     - alertname: "TradeSignal"
//!     - category: "trade-signal"
//!     - signal_id: unique identifier
//!     - symbol: e.g. "BTCUSD", "MES"
//!     - signal_direction: "LONG" | "SHORT" | "EXIT"
//!     - account_type: "prop-firm" | "personal-crypto" | "hardware-wallet"
//!     - strategy: strategy name
//!     - severity: "info" (so it doesn't trigger ops alerting)
//!
//!   Annotations (for display):
//!     - confidence, strength, entry_price, stop_loss, take_profit_1/2/3
//!     - position_size, regime, reasoning, timeframe

use reqwest::Client;
use serde::Serialize;
use std::collections::HashMap;
use std::time::Duration;
use tracing::{error, info, warn};

/// Default Alertmanager push endpoint
const DEFAULT_ALERTMANAGER_URL: &str = "http://fks_alertmanager:9093";

/// Alert payload matching Alertmanager's POST /api/v2/alerts format
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AlertmanagerAlert {
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub starts_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ends_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator_url: Option<String>,
}

/// Trade signal data to be formatted as an Alertmanager alert
#[derive(Debug, Clone)]
pub struct TradeSignalAlert {
    pub signal_id: String,
    pub symbol: String,
    pub direction: SignalDirection,
    pub account_type: AccountType,
    pub confidence: f64,
    pub strength: f64,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit_1: f64,
    pub take_profit_2: f64,
    pub take_profit_3: f64,
    pub position_size: f64,
    pub strategy: String,
    pub regime: String,
    pub reasoning: String,
    pub timeframe: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalDirection {
    Long,
    Short,
    Exit,
    Neutral,
}

impl std::fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Long => write!(f, "LONG"),
            Self::Short => write!(f, "SHORT"),
            Self::Exit => write!(f, "EXIT"),
            Self::Neutral => write!(f, "NEUTRAL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccountType {
    PropFirm,
    PersonalCrypto,
    HardwareWallet,
}

impl std::fmt::Display for AccountType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PropFirm => write!(f, "prop-firm"),
            Self::PersonalCrypto => write!(f, "personal-crypto"),
            Self::HardwareWallet => write!(f, "hardware-wallet"),
        }
    }
}

/// Client for pushing trade signals to Alertmanager
pub struct AlertmanagerClient {
    client: Client,
    base_url: String,
}

impl AlertmanagerClient {
    /// Create a new Alertmanager client.
    ///
    /// Resolution order for the base URL:
    ///   1. Explicit `base_url` argument
    ///   2. `ALERTMANAGER_URL` environment variable
    ///   3. [`DEFAULT_ALERTMANAGER_URL`] constant
    pub fn new(base_url: Option<String>) -> Self {
        let url = base_url.unwrap_or_else(|| {
            std::env::var("ALERTMANAGER_URL")
                .unwrap_or_else(|_| DEFAULT_ALERTMANAGER_URL.to_string())
        });

        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to create HTTP client");

        info!(url = %url, "AlertmanagerClient initialized");

        Self {
            client,
            base_url: url,
        }
    }

    /// Push a trade signal as an alert to Alertmanager.
    ///
    /// The signal is formatted into the Alertmanager v2 alert schema and
    /// POSTed to `/api/v2/alerts`. On success the alert will be routed
    /// through the Alertmanager pipeline where the Python-side receiver
    /// picks it up for execution routing.
    pub async fn push_signal(&self, signal: &TradeSignalAlert) -> Result<(), AlertmanagerError> {
        let alert = self.format_alert(signal);
        let url = format!("{}/api/v2/alerts", self.base_url);

        info!(
            signal_id = %signal.signal_id,
            symbol = %signal.symbol,
            direction = %signal.direction,
            account_type = %signal.account_type,
            "Pushing trade signal to Alertmanager"
        );

        let response = self
            .client
            .post(&url)
            .json(&vec![alert])
            .send()
            .await
            .map_err(|e| AlertmanagerError::HttpError(e.to_string()))?;

        let status = response.status();
        if status.is_success() {
            info!(
                signal_id = %signal.signal_id,
                "Trade signal pushed to Alertmanager successfully"
            );
            Ok(())
        } else {
            let body = response.text().await.unwrap_or_default();
            error!(
                signal_id = %signal.signal_id,
                status = %status,
                body = %body,
                "Alertmanager rejected trade signal"
            );
            Err(AlertmanagerError::ApiError {
                status: status.as_u16(),
                body,
            })
        }
    }

    /// Resolve (close) a previously-fired trade signal alert.
    ///
    /// Alertmanager uses matching labels to correlate the resolve with the
    /// original alert. We set `endsAt` to *now* so the alert is silenced
    /// immediately.
    pub async fn resolve_signal(
        &self,
        signal_id: &str,
        symbol: &str,
    ) -> Result<(), AlertmanagerError> {
        let mut labels = HashMap::new();
        labels.insert("alertname".to_string(), "TradeSignal".to_string());
        labels.insert("category".to_string(), "trade-signal".to_string());
        labels.insert("signal_id".to_string(), signal_id.to_string());
        labels.insert("symbol".to_string(), symbol.to_string());

        let now = chrono::Utc::now();
        let alert = AlertmanagerAlert {
            labels,
            annotations: HashMap::new(),
            starts_at: None,
            ends_at: Some(now.to_rfc3339()),
            generator_url: None,
        };

        let url = format!("{}/api/v2/alerts", self.base_url);
        let response = self
            .client
            .post(&url)
            .json(&vec![alert])
            .send()
            .await
            .map_err(|e| AlertmanagerError::HttpError(e.to_string()))?;

        let status = response.status();
        if status.is_success() {
            info!(signal_id = %signal_id, "Trade signal resolved in Alertmanager");
            Ok(())
        } else {
            let body = response.text().await.unwrap_or_default();
            warn!(
                signal_id = %signal_id,
                status = %status,
                body = %body,
                "Failed to resolve trade signal"
            );
            Err(AlertmanagerError::ApiError {
                status: status.as_u16(),
                body,
            })
        }
    }

    /// Check Alertmanager health via `GET /-/healthy`.
    pub async fn health_check(&self) -> bool {
        let url = format!("{}/-/healthy", self.base_url);
        match self.client.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(_) => false,
        }
    }

    /// Format a [`TradeSignalAlert`] into an Alertmanager-compatible alert.
    fn format_alert(&self, signal: &TradeSignalAlert) -> AlertmanagerAlert {
        let mut labels = HashMap::new();
        labels.insert("alertname".to_string(), "TradeSignal".to_string());
        labels.insert("category".to_string(), "trade-signal".to_string());
        labels.insert("signal_id".to_string(), signal.signal_id.clone());
        labels.insert("symbol".to_string(), signal.symbol.clone());
        labels.insert("signal_direction".to_string(), signal.direction.to_string());
        labels.insert("account_type".to_string(), signal.account_type.to_string());
        labels.insert("strategy".to_string(), signal.strategy.clone());
        labels.insert("severity".to_string(), "info".to_string());

        let mut annotations = HashMap::new();
        annotations.insert(
            "confidence".to_string(),
            format!("{:.4}", signal.confidence),
        );
        annotations.insert("strength".to_string(), format!("{:.4}", signal.strength));
        annotations.insert(
            "entry_price".to_string(),
            format!("{:.8}", signal.entry_price),
        );
        annotations.insert("stop_loss".to_string(), format!("{:.8}", signal.stop_loss));
        annotations.insert(
            "take_profit_1".to_string(),
            format!("{:.8}", signal.take_profit_1),
        );
        annotations.insert(
            "take_profit_2".to_string(),
            format!("{:.8}", signal.take_profit_2),
        );
        annotations.insert(
            "take_profit_3".to_string(),
            format!("{:.8}", signal.take_profit_3),
        );
        annotations.insert(
            "position_size".to_string(),
            format!("{:.4}", signal.position_size),
        );
        annotations.insert("regime".to_string(), signal.regime.clone());
        annotations.insert("reasoning".to_string(), signal.reasoning.clone());
        annotations.insert("timeframe".to_string(), signal.timeframe.clone());
        annotations.insert(
            "summary".to_string(),
            format!(
                "{} {} @ {:.2} (SL={:.2}, TP={:.2}) [{}]",
                signal.direction,
                signal.symbol,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit_1,
                signal.strategy,
            ),
        );

        AlertmanagerAlert {
            labels,
            annotations,
            starts_at: Some(chrono::Utc::now().to_rfc3339()),
            ends_at: None,
            generator_url: Some("http://fks_janus:8080".to_string()),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AlertmanagerError {
    #[error("HTTP error: {0}")]
    HttpError(String),

    #[error("Alertmanager API error (status={status}): {body}")]
    ApiError { status: u16, body: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a sample signal for tests.
    fn sample_signal() -> TradeSignalAlert {
        TradeSignalAlert {
            signal_id: "sig-001".to_string(),
            symbol: "BTCUSD".to_string(),
            direction: SignalDirection::Long,
            account_type: AccountType::PersonalCrypto,
            confidence: 0.85,
            strength: 0.72,
            entry_price: 67500.0,
            stop_loss: 66800.0,
            take_profit_1: 68500.0,
            take_profit_2: 69500.0,
            take_profit_3: 71000.0,
            position_size: 0.05,
            strategy: "MomentumSurge".to_string(),
            regime: "TRENDING".to_string(),
            reasoning: "Strong breakout above resistance with volume confirmation".to_string(),
            timeframe: "5m".to_string(),
        }
    }

    #[test]
    fn test_format_alert_labels() {
        let client = AlertmanagerClient::new(Some("http://localhost:9093".to_string()));
        let signal = sample_signal();
        let alert = client.format_alert(&signal);

        assert_eq!(alert.labels["alertname"], "TradeSignal");
        assert_eq!(alert.labels["category"], "trade-signal");
        assert_eq!(alert.labels["signal_id"], "sig-001");
        assert_eq!(alert.labels["symbol"], "BTCUSD");
        assert_eq!(alert.labels["signal_direction"], "LONG");
        assert_eq!(alert.labels["account_type"], "personal-crypto");
        assert_eq!(alert.labels["strategy"], "MomentumSurge");
        assert_eq!(alert.labels["severity"], "info");
    }

    #[test]
    fn test_format_alert_annotations() {
        let client = AlertmanagerClient::new(Some("http://localhost:9093".to_string()));
        let signal = sample_signal();
        let alert = client.format_alert(&signal);

        assert!(alert.annotations.contains_key("confidence"));
        assert!(alert.annotations.contains_key("strength"));
        assert!(alert.annotations.contains_key("entry_price"));
        assert!(alert.annotations.contains_key("stop_loss"));
        assert!(alert.annotations.contains_key("take_profit_1"));
        assert!(alert.annotations.contains_key("take_profit_2"));
        assert!(alert.annotations.contains_key("take_profit_3"));
        assert!(alert.annotations.contains_key("position_size"));
        assert!(alert.annotations.contains_key("regime"));
        assert!(alert.annotations.contains_key("reasoning"));
        assert!(alert.annotations.contains_key("timeframe"));
        assert!(alert.annotations.contains_key("summary"));

        // Verify the summary has the expected shape
        let summary = &alert.annotations["summary"];
        assert!(summary.contains("LONG"));
        assert!(summary.contains("BTCUSD"));
        assert!(summary.contains("MomentumSurge"));
    }

    #[test]
    fn test_format_alert_timestamps() {
        let client = AlertmanagerClient::new(Some("http://localhost:9093".to_string()));
        let signal = sample_signal();
        let alert = client.format_alert(&signal);

        assert!(alert.starts_at.is_some(), "startsAt should be set");
        assert!(alert.ends_at.is_none(), "endsAt should not be set on fire");
        assert!(alert.generator_url.is_some());
    }

    #[test]
    fn test_signal_direction_display() {
        assert_eq!(SignalDirection::Long.to_string(), "LONG");
        assert_eq!(SignalDirection::Short.to_string(), "SHORT");
        assert_eq!(SignalDirection::Exit.to_string(), "EXIT");
        assert_eq!(SignalDirection::Neutral.to_string(), "NEUTRAL");
    }

    #[test]
    fn test_account_type_display() {
        assert_eq!(AccountType::PropFirm.to_string(), "prop-firm");
        assert_eq!(AccountType::PersonalCrypto.to_string(), "personal-crypto");
        assert_eq!(AccountType::HardwareWallet.to_string(), "hardware-wallet");
    }

    #[test]
    fn test_alert_serialization_skips_none_fields() {
        let alert = AlertmanagerAlert {
            labels: HashMap::from([("alertname".to_string(), "TradeSignal".to_string())]),
            annotations: HashMap::new(),
            starts_at: None,
            ends_at: None,
            generator_url: None,
        };

        let json = serde_json::to_string(&alert).expect("serialization failed");
        assert!(!json.contains("startsAt"));
        assert!(!json.contains("endsAt"));
        assert!(!json.contains("generatorUrl"));
        // Labels should always be present
        assert!(json.contains("labels"));
    }

    #[test]
    fn test_alert_serialization_includes_present_fields() {
        let client = AlertmanagerClient::new(Some("http://localhost:9093".to_string()));
        let signal = sample_signal();
        let alert = client.format_alert(&signal);

        let json = serde_json::to_string(&alert).expect("serialization failed");
        assert!(json.contains("startsAt"));
        assert!(json.contains("generatorUrl"));
        // camelCase rename check
        assert!(!json.contains("starts_at"));
        assert!(!json.contains("generator_url"));
    }

    #[test]
    fn test_alertmanager_error_display() {
        let http_err = AlertmanagerError::HttpError("connection refused".to_string());
        assert!(http_err.to_string().contains("connection refused"));

        let api_err = AlertmanagerError::ApiError {
            status: 400,
            body: "bad request".to_string(),
        };
        let msg = api_err.to_string();
        assert!(msg.contains("400"));
        assert!(msg.contains("bad request"));
    }
}
