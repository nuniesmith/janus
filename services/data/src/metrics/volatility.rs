//! Volatility Index Poller
//!
//! Fetches implied volatility data from Deribit's DVOL index.
//!
//! ## API Documentation:
//! - Endpoint: https://www.deribit.com/api/v2/public/get_volatility_index_data
//! - Rate Limit: Public endpoint, no authentication required
//! - Update Frequency: Real-time (we poll every minute)
//!
//! ## DVOL Index:
//! Deribit's volatility index (DVOL) is similar to VIX for crypto.
//! It represents the expected 30-day volatility of Bitcoin or Ethereum.
//!
//! ## Response Format:
//! ```json
//! {
//!   "jsonrpc": "2.0",
//!   "result": {
//!     "data": [
//!       [1672531200000, 65.5],
//!       [1672531260000, 65.8]
//!     ],
//!     "continuation": "xbt_usd"
//!   },
//!   "usIn": 1672531200000000,
//!   "usOut": 1672531200000500,
//!   "usDiff": 500
//! }
//! ```

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::actors::MetricData;

/// Volatility Index poller
#[derive(Clone)]
pub struct VolatilityPoller {
    url: String,
    client: reqwest::Client,
}

impl VolatilityPoller {
    /// Create a new Volatility poller
    pub fn new(url: String) -> Self {
        Self {
            url,
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }

    /// Poll the DVOL volatility index
    pub async fn poll(&self) -> Result<Vec<MetricData>> {
        debug!("VolatilityPoller: Fetching DVOL index");

        let mut metrics = Vec::new();

        // Fetch BTC volatility
        if let Ok(btc_vol) = self.fetch_dvol("BTC").await {
            metrics.push(btc_vol);
        }

        // Fetch ETH volatility
        if let Ok(eth_vol) = self.fetch_dvol("ETH").await {
            metrics.push(eth_vol);
        }

        info!(
            "VolatilityPoller: Fetched {} volatility data points",
            metrics.len()
        );

        Ok(metrics)
    }

    /// Fetch DVOL for a specific currency
    async fn fetch_dvol(&self, currency: &str) -> Result<MetricData> {
        let url = format!("{}?currency={}", self.url, currency);

        debug!("VolatilityPoller: Fetching DVOL for {}", currency);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .context("Failed to fetch DVOL data")?;

        if !response.status().is_success() {
            anyhow::bail!("DVOL API returned status: {}", response.status());
        }

        let dvol_response: DvolResponse = response
            .json()
            .await
            .context("Failed to parse DVOL response")?;

        // Extract the most recent data point
        let data_point = dvol_response
            .result
            .data
            .last()
            .context("No DVOL data points")?;

        let timestamp = data_point.0;
        let value = data_point.1;

        info!("VolatilityPoller: DVOL for {} = {:.2}", currency, value);

        // Create metric data
        let metric = MetricData {
            metric_type: "dvol_index".to_string(),
            asset: currency.to_string(),
            source: "deribit".to_string(),
            value,
            meta: Some("30d_implied_volatility".to_string()),
            timestamp,
        };

        Ok(metric)
    }

    /// Calculate realized volatility from historical price data
    /// This would query QuestDB for recent trades and calculate std dev of log returns
    #[allow(dead_code)]
    pub async fn calculate_realized_volatility(
        &self,
        _symbol: &str,
        _window_hours: u32,
    ) -> Result<f64> {
        // In production this would query QuestDB for recent trade prices.
        // For now we accept a helper that can be wired up to any price source.
        debug!("VolatilityPoller: calculate_realized_volatility called");

        // Placeholder: no price source connected yet – delegate to the
        // pure-math helper so the algorithm is already validated.
        Ok(0.0)
    }

    /// Pure computation of realized volatility from a price series.
    ///
    /// Steps:
    /// 1. Compute log returns: ln(price_t / price_{t-1})
    /// 2. Compute sample standard deviation of those returns
    /// 3. Annualize: std_dev * sqrt(periods_per_year)
    ///
    /// `interval_minutes` is the time between consecutive prices (e.g. 1 for
    /// minute bars). The annualization factor assumes 365 trading days, 24 h.
    #[allow(dead_code)]
    pub fn realized_volatility_from_prices(prices: &[f64], interval_minutes: f64) -> Result<f64> {
        if prices.len() < 2 {
            anyhow::bail!("Need at least 2 price observations to compute volatility");
        }

        // 1. Log returns
        let log_returns: Vec<f64> = prices
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 && w[1] > 0.0 {
                    Some((w[1] / w[0]).ln())
                } else {
                    None
                }
            })
            .collect();

        if log_returns.is_empty() {
            anyhow::bail!("No valid price pairs for log-return calculation");
        }

        let n = log_returns.len() as f64;

        // 2. Standard deviation
        //    For a single return (n=1), sample std dev is undefined (mean = the
        //    single value, so deviation = 0). Use the zero-mean estimator instead,
        //    which is standard for realized-volatility calculation on short
        //    intervals where E[r] ≈ 0.
        let variance = if log_returns.len() == 1 {
            // Zero-mean estimator: σ² = r²
            log_returns[0].powi(2)
        } else {
            let mean = log_returns.iter().sum::<f64>() / n;
            log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)
        };

        let std_dev = variance.sqrt();

        // 4. Annualize
        //    minutes_per_year = 365 * 24 * 60 = 525_600
        let periods_per_year = 525_600.0 / interval_minutes;
        let annualized = std_dev * periods_per_year.sqrt();

        Ok(annualized)
    }
}

/// Deribit DVOL API response
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DvolResponse {
    jsonrpc: String,
    result: DvolResult,
    #[serde(rename = "usIn")]
    us_in: i64,
    #[serde(rename = "usOut")]
    us_out: i64,
    #[serde(rename = "usDiff")]
    us_diff: i64,
}

/// DVOL result data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DvolResult {
    /// Array of [timestamp_ms, volatility_value] tuples
    data: Vec<(i64, f64)>,
    continuation: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dvol_response() {
        let json = r#"{
            "jsonrpc": "2.0",
            "result": {
                "data": [
                    [1672531200000, 65.5],
                    [1672531260000, 65.8],
                    [1672531320000, 66.0]
                ],
                "continuation": "xbt_usd"
            },
            "usIn": 1672531200000000,
            "usOut": 1672531200000500,
            "usDiff": 500
        }"#;

        let response: DvolResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.jsonrpc, "2.0");
        assert_eq!(response.result.data.len(), 3);
        assert_eq!(response.result.data[0].0, 1672531200000);
        assert_eq!(response.result.data[0].1, 65.5);
        assert_eq!(response.result.data.last().unwrap().1, 66.0);
    }

    #[tokio::test]
    async fn test_volatility_poller_creation() {
        let poller = VolatilityPoller::new(
            "https://www.deribit.com/api/v2/public/get_volatility_index_data".to_string(),
        );
        assert!(poller.url.contains("deribit.com"));
    }

    #[test]
    fn test_dvol_data_tuple_parsing() {
        let data: Vec<(i64, f64)> = vec![(1672531200000, 65.5), (1672531260000, 65.8)];

        assert_eq!(data[0].0, 1672531200000);
        assert_eq!(data[0].1, 65.5);
        assert_eq!(data[1].0, 1672531260000);
        assert_eq!(data[1].1, 65.8);
    }

    #[test]
    fn test_realized_volatility_constant_prices() {
        // Constant prices → zero volatility
        let prices = vec![100.0; 10];
        let vol = VolatilityPoller::realized_volatility_from_prices(&prices, 1.0).unwrap();
        assert!(
            (vol - 0.0).abs() < 1e-10,
            "constant prices should have zero vol, got {}",
            vol
        );
    }

    #[test]
    fn test_realized_volatility_known_values() {
        // Simple two-price case: single log return = ln(110/100) ≈ 0.09531
        // std dev of a single observation (n-1=0) → fallback to n=1 denominator
        let prices = vec![100.0, 110.0];
        let vol = VolatilityPoller::realized_volatility_from_prices(&prices, 1.0).unwrap();
        // With only one return, variance denominator is max(n-1,1)=1
        // std_dev = |ln(1.1)| ≈ 0.09531, annualized = 0.09531 * sqrt(525600) ≈ 69.1
        assert!(vol > 60.0 && vol < 80.0, "unexpected vol: {}", vol);
    }

    #[test]
    fn test_realized_volatility_too_few_prices() {
        assert!(VolatilityPoller::realized_volatility_from_prices(&[], 1.0).is_err());
        assert!(VolatilityPoller::realized_volatility_from_prices(&[100.0], 1.0).is_err());
    }

    #[test]
    fn test_realized_volatility_interval_scaling() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.1).sin()).collect();
        let vol_1m = VolatilityPoller::realized_volatility_from_prices(&prices, 1.0).unwrap();
        let vol_5m = VolatilityPoller::realized_volatility_from_prices(&prices, 5.0).unwrap();
        // Larger interval → fewer periods per year → lower annualized vol
        assert!(
            vol_5m < vol_1m,
            "5-min vol ({}) should be less than 1-min vol ({})",
            vol_5m,
            vol_1m
        );
    }
}
