//! Fear & Greed Index Poller
//!
//! Fetches the Crypto Fear & Greed Index from Alternative.me API.
//!
//! ## API Documentation:
//! - Endpoint: https://api.alternative.me/fng/
//! - Rate Limit: None (free public API)
//! - Update Frequency: Once daily (around 00:00 UTC)
//!
//! ## Response Format:
//! ```json
//! {
//!   "name": "Fear and Greed Index",
//!   "data": [
//!     {
//!       "value": "45",
//!       "value_classification": "Neutral",
//!       "timestamp": "1672531200",
//!       "time_until_update": "86400"
//!     }
//!   ],
//!   "metadata": {
//!     "error": null
//!   }
//! }
//! ```
//!
//! ## Index Scale:
//! - 0-24: Extreme Fear
//! - 25-49: Fear
//! - 50-74: Greed
//! - 75-100: Extreme Greed

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::actors::MetricData;

/// Fear & Greed Index poller
#[derive(Clone)]
pub struct FearGreedPoller {
    url: String,
    client: reqwest::Client,
}

impl FearGreedPoller {
    /// Create a new Fear & Greed poller
    pub fn new(url: String) -> Self {
        Self {
            url,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Poll the Fear & Greed Index
    pub async fn poll(&self) -> Result<Vec<MetricData>> {
        debug!("FearGreedPoller: Fetching Fear & Greed Index");

        let response = self
            .client
            .get(&self.url)
            .send()
            .await
            .context("Failed to fetch Fear & Greed Index")?;

        if !response.status().is_success() {
            anyhow::bail!("Fear & Greed API returned status: {}", response.status());
        }

        let fg_response: FearGreedResponse = response
            .json()
            .await
            .context("Failed to parse Fear & Greed response")?;

        // Check for API errors
        if let Some(error) = fg_response.metadata.error {
            anyhow::bail!("Fear & Greed API error: {}", error);
        }

        // Extract the most recent data point
        let data = fg_response
            .data
            .first()
            .context("No data in Fear & Greed response")?;

        // Parse value
        let value: f64 = data
            .value
            .parse()
            .context("Failed to parse Fear & Greed value")?;

        // Parse timestamp (seconds to milliseconds)
        let timestamp: i64 = data
            .timestamp
            .parse::<i64>()
            .context("Failed to parse timestamp")?
            * 1000;

        info!(
            "FearGreedPoller: Fetched Fear & Greed Index = {} ({})",
            value, data.value_classification
        );

        // Create metric data
        let metric = MetricData {
            metric_type: "fear_greed".to_string(),
            asset: "GLOBAL".to_string(),
            source: "alternative_me".to_string(),
            value,
            meta: Some(data.value_classification.clone()),
            timestamp,
        };

        Ok(vec![metric])
    }
}

/// Fear & Greed API response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FearGreedResponse {
    name: String,
    data: Vec<FearGreedData>,
    metadata: FearGreedMetadata,
}

/// Fear & Greed data point
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FearGreedData {
    /// Index value (0-100)
    value: String,

    /// Classification (e.g., "Extreme Fear", "Neutral", "Greed")
    value_classification: String,

    /// Unix timestamp (seconds)
    timestamp: String,

    /// Time until next update (seconds)
    #[serde(default)]
    time_until_update: Option<String>,
}

/// Fear & Greed metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FearGreedMetadata {
    error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_fear_greed_response() {
        let json = r#"{
            "name": "Fear and Greed Index",
            "data": [
                {
                    "value": "45",
                    "value_classification": "Neutral",
                    "timestamp": "1672531200",
                    "time_until_update": "86400"
                }
            ],
            "metadata": {
                "error": null
            }
        }"#;

        let response: FearGreedResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.name, "Fear and Greed Index");
        assert_eq!(response.data[0].value, "45");
        assert_eq!(response.data[0].value_classification, "Neutral");
        assert!(response.metadata.error.is_none());
    }

    #[test]
    fn test_parse_value() {
        let value_str = "45";
        let value: f64 = value_str.parse().unwrap();
        assert_eq!(value, 45.0);
    }

    #[tokio::test]
    async fn test_fear_greed_poller_creation() {
        let poller = FearGreedPoller::new("https://api.alternative.me/fng/".to_string());
        assert_eq!(poller.url, "https://api.alternative.me/fng/");
    }
}
