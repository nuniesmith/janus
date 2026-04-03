//! Kraken REST API Client
//!
//! This module provides authenticated REST API access to Kraken exchange
//! for account management and order execution.
//!
//! # Authentication
//!
//! Kraken uses HMAC-SHA512 signatures for private endpoints:
//! 1. Decode API secret from base64
//! 2. SHA256 hash of (nonce + POST data)
//! 3. HMAC-SHA512 of (endpoint + hash) using API secret
//! 4. Base64 encode the signature
//!
//! # Endpoints
//!
//! | Endpoint | Description | Auth Required |
//! |----------|-------------|---------------|
//! | /0/public/Time | Server time | No |
//! | /0/private/Balance | Account balances | Yes |
//! | /0/private/AddOrder | Place order | Yes |
//! | /0/private/CancelOrder | Cancel order | Yes |
//! | /0/private/OpenOrders | Open orders | Yes |
//! | /0/private/QueryOrders | Query orders | Yes |
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::exchanges::kraken::rest::{KrakenRestClient, KrakenRestConfig};
//!
//! let config = KrakenRestConfig {
//!     api_key: "your_key".to_string(),
//!     api_secret: "your_secret".to_string(),
//!     ..Default::default()
//! };
//!
//! let client = KrakenRestClient::new(config);
//!
//! // Get account balance
//! let balances = client.get_balance().await?;
//!
//! // Place a market order
//! let order = client.place_market_order("BTC/USD", OrderSide::Buy, 0.001, false).await?;
//! ```

use base64::{Engine as _, engine::general_purpose};
use chrono::Utc;
use hmac::{Hmac, Mac};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use sha2::{Digest, Sha256, Sha512};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::{Duration, Instant};
use tracing::{debug, error, info};

use crate::error::{ExecutionError, Result};
use crate::execution::histogram::global_latency_histograms;
use crate::execution::metrics::retry_metrics;
use crate::types::OrderSide;

type HmacSha512 = Hmac<Sha512>;

/// Kraken REST API base URL
pub const REST_API_URL: &str = "https://api.kraken.com";

/// Kraken Demo/Futures REST API URL
pub const REST_DEMO_URL: &str = "https://demo-futures.kraken.com";

/// Kraken REST client configuration
#[derive(Debug, Clone)]
pub struct KrakenRestConfig {
    /// API key for authentication
    pub api_key: String,
    /// API secret (base64 encoded) for signing requests
    pub api_secret: String,
    /// Use demo/testnet environment
    pub testnet: bool,
    /// Custom API URL (overrides testnet setting)
    pub custom_url: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl Default for KrakenRestConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_secret: String::new(),
            testnet: false,
            custom_url: None,
            timeout_secs: 30,
        }
    }
}

impl KrakenRestConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let api_key = std::env::var("KRAKEN_API_KEY").unwrap_or_default();
        let api_secret = std::env::var("KRAKEN_API_SECRET").unwrap_or_default();
        let testnet = std::env::var("KRAKEN_TESTNET")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);
        let custom_url = std::env::var("KRAKEN_API_URL").ok();
        let timeout_secs = std::env::var("KRAKEN_TIMEOUT_SECS")
            .unwrap_or_else(|_| "30".to_string())
            .parse()
            .unwrap_or(30);

        Self {
            api_key,
            api_secret,
            testnet,
            custom_url,
            timeout_secs,
        }
    }

    /// Get the API base URL
    pub fn get_url(&self) -> &str {
        if let Some(ref url) = self.custom_url {
            url.as_str()
        } else if self.testnet {
            REST_DEMO_URL
        } else {
            REST_API_URL
        }
    }

    /// Check if the config has valid credentials
    pub fn has_credentials(&self) -> bool {
        !self.api_key.is_empty() && !self.api_secret.is_empty()
    }
}

/// Kraken API response wrapper
#[derive(Debug, Deserialize)]
struct KrakenResponse<T> {
    error: Vec<String>,
    result: Option<T>,
}

/// Server time response
#[derive(Debug, Deserialize)]
struct ServerTimeResult {
    unixtime: u64,
    #[allow(dead_code)]
    rfc1123: String,
}

/// WebSocket token response
#[derive(Debug, Deserialize)]
struct WebSocketTokenResult {
    token: String,
    #[allow(dead_code)]
    expires: Option<u64>,
}

/// Balance response
#[derive(Debug, Deserialize)]
struct BalanceResult {
    #[serde(flatten)]
    balances: HashMap<String, String>,
}

/// Add order response
#[derive(Debug, Deserialize)]
struct AddOrderResult {
    descr: OrderDescription,
    txid: Vec<String>,
}

/// Order description from response
#[derive(Debug, Deserialize)]
struct OrderDescription {
    order: String,
    #[allow(dead_code)]
    close: Option<String>,
}

/// Open orders response
#[derive(Debug, Deserialize)]
struct OpenOrdersResult {
    open: HashMap<String, KrakenOrderInfo>,
}

/// Order information from Kraken
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOrderInfo {
    #[serde(default)]
    pub status: String,
    pub opentm: f64,
    #[serde(default)]
    pub starttm: f64,
    #[serde(default)]
    pub expiretm: f64,
    pub descr: KrakenOrderDescr,
    pub vol: String,
    pub vol_exec: String,
    #[serde(default)]
    pub cost: String,
    #[serde(default)]
    pub fee: String,
    #[serde(default)]
    pub price: String,
    #[serde(default)]
    pub stopprice: String,
    #[serde(default)]
    pub limitprice: String,
    #[serde(default)]
    pub misc: String,
    #[serde(default)]
    pub oflags: String,
}

/// Order description details
#[derive(Debug, Clone, Deserialize)]
pub struct KrakenOrderDescr {
    pub pair: String,
    #[serde(rename = "type")]
    pub side: String,
    pub ordertype: String,
    pub price: String,
    #[serde(default)]
    pub price2: String,
    pub leverage: String,
    pub order: String,
    #[serde(default)]
    pub close: String,
}

/// Account balance entry
#[derive(Debug, Clone)]
pub struct KrakenBalance {
    /// Currency/asset name
    pub currency: String,
    /// Available balance
    pub available: Decimal,
    /// Total balance (may include reserved)
    pub total: Decimal,
}

/// Order placement result
#[derive(Debug, Clone)]
pub struct KrakenOrderResult {
    /// Transaction ID(s)
    pub txid: Vec<String>,
    /// Order description
    pub description: String,
}

/// Kraken REST API client
pub struct KrakenRestClient {
    config: KrakenRestConfig,
    client: Client,
    base_url: String,
}

impl KrakenRestClient {
    /// Create a new Kraken REST client
    pub fn new(config: KrakenRestConfig) -> Self {
        let base_url = config.get_url().to_string();
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .expect("Failed to create HTTP client");

        info!(
            "Kraken REST client created (testnet: {}, has_creds: {})",
            config.testnet,
            config.has_credentials()
        );

        Self {
            config,
            client,
            base_url,
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Self {
        Self::new(KrakenRestConfig::from_env())
    }

    /// Check if the client has valid credentials for private endpoints
    pub fn has_credentials(&self) -> bool {
        self.config.has_credentials()
    }

    // ==================== Public Endpoints ====================

    /// Get server time (public endpoint)
    pub async fn get_server_time(&self) -> Result<u64> {
        let url = format!("{}/0/public/Time", self.base_url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Network error: {}", e)))?;

        let result: KrakenResponse<ServerTimeResult> = response
            .json()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Parse error: {}", e)))?;

        if !result.error.is_empty() {
            return Err(ExecutionError::exchange("kraken", result.error.join(", ")));
        }

        Ok(result.result.unwrap().unixtime)
    }

    // ==================== Private Endpoints ====================

    /// Get account balance (private endpoint)
    pub async fn get_balance(&self) -> Result<Vec<KrakenBalance>> {
        if !self.has_credentials() {
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        let endpoint = "/0/private/Balance";
        let params = HashMap::new();

        let response = self.private_request(endpoint, &params).await?;

        let result: KrakenResponse<BalanceResult> = response
            .json()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Parse error: {}", e)))?;

        if !result.error.is_empty() {
            return Err(ExecutionError::exchange("kraken", result.error.join(", ")));
        }

        let balances = result
            .result
            .unwrap()
            .balances
            .into_iter()
            .filter_map(|(currency, amount)| {
                let total = Decimal::from_str(&amount).ok()?;
                if total > Decimal::ZERO {
                    Some(KrakenBalance {
                        currency: normalize_currency(&currency),
                        available: total,
                        total,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(balances)
    }

    /// Get USD balance specifically
    pub async fn get_usd_balance(&self) -> Result<Decimal> {
        let balances = self.get_balance().await?;

        // Kraken uses ZUSD for USD
        let usd = balances
            .iter()
            .find(|b| b.currency == "USD" || b.currency == "ZUSD")
            .map(|b| b.available)
            .unwrap_or(Decimal::ZERO);

        Ok(usd)
    }

    /// Place a market order
    pub async fn place_market_order(
        &self,
        pair: &str,
        side: OrderSide,
        volume: Decimal,
        validate: bool,
    ) -> Result<KrakenOrderResult> {
        let metrics = retry_metrics("kraken_market_order");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        if !self.has_credentials() {
            metrics.record_failure(1, start.elapsed().as_millis() as u64);
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        info!(
            "Placing {} market order for {} {} (validate: {})",
            match side {
                OrderSide::Buy => "BUY",
                OrderSide::Sell => "SELL",
            },
            volume,
            pair,
            validate
        );

        let mut params = HashMap::new();
        params.insert("ordertype".to_string(), "market".to_string());
        params.insert(
            "type".to_string(),
            match side {
                OrderSide::Buy => "buy",
                OrderSide::Sell => "sell",
            }
            .to_string(),
        );
        params.insert("volume".to_string(), volume.to_string());
        params.insert("pair".to_string(), to_kraken_pair(pair));

        if validate {
            params.insert("validate".to_string(), "true".to_string());
        }

        let endpoint = "/0/private/AddOrder";
        let response = match self.private_request(endpoint, &params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_placement("kraken", duration_ms as f64);
                histograms.record_api_call("kraken_market_order", duration_ms as f64);
                if e.is_rate_limit() {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let result: KrakenResponse<AddOrderResult> = response.json().await.map_err(|e| {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            histograms.record_order_placement("kraken", duration_ms as f64);
            histograms.record_api_call("kraken_market_order", duration_ms as f64);
            ExecutionError::exchange("kraken", format!("Parse error: {}", e))
        })?;

        if !result.error.is_empty() {
            let duration_ms = start.elapsed().as_millis() as u64;
            let error_msg = result.error.join(", ");
            metrics.record_failure(1, duration_ms);
            histograms.record_order_placement("kraken", duration_ms as f64);
            histograms.record_api_call("kraken_market_order", duration_ms as f64);
            if error_msg.contains("Rate limit") || error_msg.contains("EAPI:Rate limit") {
                metrics.record_rate_limit();
            }
            return Err(ExecutionError::exchange("kraken", error_msg));
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_order_placement("kraken", duration_ms as f64);
        histograms.record_api_call("kraken_market_order", duration_ms as f64);

        let order_result = result.result.unwrap();
        let txid = order_result.txid.clone();
        let description = order_result.descr.order;

        info!(
            "Order placed successfully: {:?} - {} ({}ms)",
            txid, description, duration_ms
        );

        Ok(KrakenOrderResult { txid, description })
    }

    /// Place a limit order
    pub async fn place_limit_order(
        &self,
        pair: &str,
        side: OrderSide,
        price: Decimal,
        volume: Decimal,
        validate: bool,
    ) -> Result<KrakenOrderResult> {
        let metrics = retry_metrics("kraken_limit_order");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        if !self.has_credentials() {
            metrics.record_failure(1, start.elapsed().as_millis() as u64);
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        info!(
            "Placing {} limit order: {} {} @ {}",
            match side {
                OrderSide::Buy => "BUY",
                OrderSide::Sell => "SELL",
            },
            volume,
            pair,
            price
        );

        let mut params = HashMap::new();
        params.insert("ordertype".to_string(), "limit".to_string());
        params.insert(
            "type".to_string(),
            match side {
                OrderSide::Buy => "buy",
                OrderSide::Sell => "sell",
            }
            .to_string(),
        );
        params.insert("price".to_string(), price.to_string());
        params.insert("volume".to_string(), volume.to_string());
        params.insert("pair".to_string(), to_kraken_pair(pair));

        if validate {
            params.insert("validate".to_string(), "true".to_string());
        }

        let endpoint = "/0/private/AddOrder";
        let response = match self.private_request(endpoint, &params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_placement("kraken", duration_ms as f64);
                histograms.record_api_call("kraken_limit_order", duration_ms as f64);
                if e.is_rate_limit() {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let result: KrakenResponse<AddOrderResult> = response.json().await.map_err(|e| {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            histograms.record_order_placement("kraken", duration_ms as f64);
            histograms.record_api_call("kraken_limit_order", duration_ms as f64);
            ExecutionError::exchange("kraken", format!("Parse error: {}", e))
        })?;

        if !result.error.is_empty() {
            let duration_ms = start.elapsed().as_millis() as u64;
            let error_msg = result.error.join(", ");
            metrics.record_failure(1, duration_ms);
            histograms.record_order_placement("kraken", duration_ms as f64);
            histograms.record_api_call("kraken_limit_order", duration_ms as f64);
            if error_msg.contains("Rate limit") || error_msg.contains("EAPI:Rate limit") {
                metrics.record_rate_limit();
            }
            return Err(ExecutionError::exchange("kraken", error_msg));
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_order_placement("kraken", duration_ms as f64);
        histograms.record_api_call("kraken_limit_order", duration_ms as f64);

        let order_result = result.result.unwrap();
        let txid = order_result.txid.clone();
        let description = order_result.descr.order;

        info!(
            "Limit order placed: {:?} - {} ({}ms)",
            txid, description, duration_ms
        );

        Ok(KrakenOrderResult { txid, description })
    }

    /// Place a stop-loss order
    pub async fn place_stop_loss_order(
        &self,
        pair: &str,
        side: OrderSide,
        stop_price: Decimal,
        volume: Decimal,
        validate: bool,
    ) -> Result<KrakenOrderResult> {
        if !self.has_credentials() {
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        info!(
            "Placing {} stop-loss order for {} {} @ {} (validate: {})",
            match side {
                OrderSide::Buy => "BUY",
                OrderSide::Sell => "SELL",
            },
            volume,
            pair,
            stop_price,
            validate
        );

        let mut params = HashMap::new();
        params.insert("ordertype".to_string(), "stop-loss".to_string());
        params.insert(
            "type".to_string(),
            match side {
                OrderSide::Buy => "buy",
                OrderSide::Sell => "sell",
            }
            .to_string(),
        );
        params.insert("price".to_string(), stop_price.to_string());
        params.insert("volume".to_string(), volume.to_string());
        params.insert("pair".to_string(), to_kraken_pair(pair));

        if validate {
            params.insert("validate".to_string(), "true".to_string());
        }

        let endpoint = "/0/private/AddOrder";
        let response = self.private_request(endpoint, &params).await?;

        let result: KrakenResponse<AddOrderResult> = response
            .json()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Parse error: {}", e)))?;

        if !result.error.is_empty() {
            return Err(ExecutionError::exchange("kraken", result.error.join(", ")));
        }

        let order_result = result.result.unwrap();
        let txid = order_result.txid.clone();
        let description = order_result.descr.order;

        info!(
            "Stop-loss order placed successfully: {:?} - {}",
            txid, description
        );

        Ok(KrakenOrderResult { txid, description })
    }

    /// Place a take-profit order
    pub async fn place_take_profit_order(
        &self,
        pair: &str,
        side: OrderSide,
        take_profit_price: Decimal,
        volume: Decimal,
        validate: bool,
    ) -> Result<KrakenOrderResult> {
        if !self.has_credentials() {
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        info!(
            "Placing {} take-profit order for {} {} @ {} (validate: {})",
            match side {
                OrderSide::Buy => "BUY",
                OrderSide::Sell => "SELL",
            },
            volume,
            pair,
            take_profit_price,
            validate
        );

        let mut params = HashMap::new();
        params.insert("ordertype".to_string(), "take-profit".to_string());
        params.insert(
            "type".to_string(),
            match side {
                OrderSide::Buy => "buy",
                OrderSide::Sell => "sell",
            }
            .to_string(),
        );
        params.insert("price".to_string(), take_profit_price.to_string());
        params.insert("volume".to_string(), volume.to_string());
        params.insert("pair".to_string(), to_kraken_pair(pair));

        if validate {
            params.insert("validate".to_string(), "true".to_string());
        }

        let endpoint = "/0/private/AddOrder";
        let response = self.private_request(endpoint, &params).await?;

        let result: KrakenResponse<AddOrderResult> = response
            .json()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Parse error: {}", e)))?;

        if !result.error.is_empty() {
            return Err(ExecutionError::exchange("kraken", result.error.join(", ")));
        }

        let order_result = result.result.unwrap();
        let txid = order_result.txid.clone();
        let description = order_result.descr.order;

        info!(
            "Take-profit order placed successfully: {:?} - {}",
            txid, description
        );

        Ok(KrakenOrderResult { txid, description })
    }

    /// Cancel an order
    pub async fn cancel_order(&self, txid: &str) -> Result<bool> {
        let metrics = retry_metrics("kraken_cancel_order");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        if !self.has_credentials() {
            metrics.record_failure(1, start.elapsed().as_millis() as u64);
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        info!("Cancelling order: {}", txid);

        let mut params = HashMap::new();
        params.insert("txid".to_string(), txid.to_string());

        let endpoint = "/0/private/CancelOrder";
        let response = match self.private_request(endpoint, &params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_cancellation("kraken", duration_ms as f64);
                histograms.record_api_call("kraken_cancel_order", duration_ms as f64);
                if e.is_rate_limit() {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let result: KrakenResponse<serde_json::Value> = response.json().await.map_err(|e| {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            histograms.record_order_cancellation("kraken", duration_ms as f64);
            histograms.record_api_call("kraken_cancel_order", duration_ms as f64);
            ExecutionError::exchange("kraken", format!("Parse error: {}", e))
        })?;

        if !result.error.is_empty() {
            let duration_ms = start.elapsed().as_millis() as u64;
            let error_msg = result.error.join(", ");
            metrics.record_failure(1, duration_ms);
            histograms.record_order_cancellation("kraken", duration_ms as f64);
            histograms.record_api_call("kraken_cancel_order", duration_ms as f64);
            if error_msg.contains("Rate limit") || error_msg.contains("EAPI:Rate limit") {
                metrics.record_rate_limit();
            }
            return Err(ExecutionError::exchange("kraken", error_msg));
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_order_cancellation("kraken", duration_ms as f64);
        histograms.record_api_call("kraken_cancel_order", duration_ms as f64);

        info!("Order {} cancelled successfully ({}ms)", txid, duration_ms);
        Ok(true)
    }

    /// Cancel all orders (optionally for a specific pair)
    pub async fn cancel_all_orders(&self, pair: Option<&str>) -> Result<u32> {
        if !self.has_credentials() {
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        info!("Canceling all orders (pair: {:?})", pair);

        let endpoint = "/0/private/CancelAll";
        let mut params = HashMap::new();

        if let Some(p) = pair {
            params.insert("pair".to_string(), to_kraken_pair(p));
        }

        let response = self.private_request(endpoint, &params).await?;

        let result: KrakenResponse<serde_json::Value> = response
            .json()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Parse error: {}", e)))?;

        if !result.error.is_empty() {
            return Err(ExecutionError::exchange("kraken", result.error.join(", ")));
        }

        let count = result
            .result
            .and_then(|v| v.get("count").and_then(|c| c.as_u64()))
            .unwrap_or(0) as u32;

        info!("Canceled {} orders", count);
        Ok(count)
    }

    /// Get open orders
    pub async fn get_open_orders(&self) -> Result<HashMap<String, KrakenOrderInfo>> {
        let metrics = retry_metrics("kraken_get_orders");
        let start = Instant::now();

        if !self.has_credentials() {
            metrics.record_failure(1, start.elapsed().as_millis() as u64);
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        let params = HashMap::new();
        let endpoint = "/0/private/OpenOrders";
        let response = match self.private_request(endpoint, &params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                if e.is_rate_limit() {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let result: KrakenResponse<OpenOrdersResult> = response.json().await.map_err(|e| {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            ExecutionError::exchange("kraken", format!("Parse error: {}", e))
        })?;

        if !result.error.is_empty() {
            let duration_ms = start.elapsed().as_millis() as u64;
            let error_msg = result.error.join(", ");
            metrics.record_failure(1, duration_ms);
            if error_msg.contains("Rate limit") || error_msg.contains("EAPI:Rate limit") {
                metrics.record_rate_limit();
            }
            return Err(ExecutionError::exchange("kraken", error_msg));
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);

        Ok(result.result.map(|r| r.open).unwrap_or_default())
    }

    /// Query specific orders by txid
    pub async fn query_orders(&self, txids: &[&str]) -> Result<HashMap<String, KrakenOrderInfo>> {
        if !self.has_credentials() {
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        let endpoint = "/0/private/QueryOrders";
        let mut params = HashMap::new();
        params.insert("txid".to_string(), txids.join(","));

        let response = self.private_request(endpoint, &params).await?;

        let result: KrakenResponse<HashMap<String, KrakenOrderInfo>> = response
            .json()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Parse error: {}", e)))?;

        if !result.error.is_empty() {
            return Err(ExecutionError::exchange("kraken", result.error.join(", ")));
        }

        Ok(result.result.unwrap_or_default())
    }

    /// Get WebSocket authentication token
    ///
    /// Returns a token that can be used to authenticate with Kraken's private WebSocket API.
    /// The token is valid for 15 minutes.
    pub async fn get_websocket_token(&self) -> Result<String> {
        if !self.has_credentials() {
            return Err(ExecutionError::Auth(
                "API credentials not configured".to_string(),
            ));
        }

        info!("Requesting WebSocket authentication token...");

        let endpoint = "/0/private/GetWebSocketsToken";
        let params = HashMap::new();

        let response = self.private_request(endpoint, &params).await?;

        let result: KrakenResponse<WebSocketTokenResult> = response
            .json()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Parse error: {}", e)))?;

        if !result.error.is_empty() {
            return Err(ExecutionError::exchange("kraken", result.error.join(", ")));
        }

        let token_result = result.result.ok_or_else(|| {
            ExecutionError::exchange("kraken", "No token in response".to_string())
        })?;

        info!("WebSocket token obtained successfully");
        Ok(token_result.token)
    }

    // ==================== Private Request Helper ====================

    /// Make a private (authenticated) API request
    async fn private_request(
        &self,
        endpoint: &str,
        params: &HashMap<String, String>,
    ) -> Result<reqwest::Response> {
        let url = format!("{}{}", self.base_url, endpoint);
        let nonce = Utc::now().timestamp_millis().to_string();

        let mut form_data = params.clone();
        form_data.insert("nonce".to_string(), nonce.clone());

        // Create POST data string
        let post_data = form_data
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&");

        // Generate signature
        let signature = self.generate_signature(endpoint, &nonce, &post_data)?;

        debug!("Making private request to: {}", endpoint);

        let response = self
            .client
            .post(&url)
            .header("API-Key", &self.config.api_key)
            .header("API-Sign", signature)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(post_data)
            .send()
            .await
            .map_err(|e| ExecutionError::exchange("kraken", format!("Network error: {}", e)))?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            error!("Kraken API error: {} - {}", status, error_text);
            return Err(ExecutionError::exchange(
                "kraken",
                format!("HTTP {}: {}", status, error_text),
            ));
        }

        Ok(response)
    }

    /// Generate HMAC-SHA512 signature for authenticated requests
    fn generate_signature(&self, endpoint: &str, nonce: &str, post_data: &str) -> Result<String> {
        // Decode API secret from base64
        let api_secret = general_purpose::STANDARD
            .decode(&self.config.api_secret)
            .map_err(|e| ExecutionError::Auth(format!("Invalid API secret encoding: {}", e)))?;

        // Create SHA256 hash of (nonce + POST data)
        let mut hasher = Sha256::new();
        hasher.update(nonce.as_bytes());
        hasher.update(post_data.as_bytes());
        let hash = hasher.finalize();

        // Create message: endpoint path + hash
        let mut message = endpoint.as_bytes().to_vec();
        message.extend_from_slice(&hash);

        // Create HMAC-SHA512
        let mut mac = HmacSha512::new_from_slice(&api_secret)
            .map_err(|e| ExecutionError::Auth(format!("HMAC error: {}", e)))?;
        mac.update(&message);
        let signature = mac.finalize().into_bytes();

        // Encode signature as base64
        Ok(general_purpose::STANDARD.encode(signature))
    }
}

// ==================== Helper Functions ====================

/// Convert a normalized pair (BTC/USDT) to Kraken format (BTCUSD)
fn to_kraken_pair(pair: &str) -> String {
    // Remove slash and convert USDT to USD
    pair.replace('/', "").replace("USDT", "USD").to_uppercase()
}

/// Normalize Kraken currency codes
fn normalize_currency(kraken_currency: &str) -> String {
    match kraken_currency {
        "ZUSD" | "USD" => "USD".to_string(),
        "XXBT" | "XBT" => "BTC".to_string(),
        "XETH" | "ETH" => "ETH".to_string(),
        "XXRP" | "XRP" => "XRP".to_string(),
        "XLTC" | "LTC" => "LTC".to_string(),
        "XXLM" | "XLM" => "XLM".to_string(),
        "XXDG" | "XDG" => "DOGE".to_string(),
        "ZEUR" | "EUR" => "EUR".to_string(),
        "ZGBP" | "GBP" => "GBP".to_string(),
        "ZCAD" | "CAD" => "CAD".to_string(),
        "ZJPY" | "JPY" => "JPY".to_string(),
        other => {
            // Remove leading X or Z if present
            if other.starts_with('X') || other.starts_with('Z') {
                other[1..].to_string()
            } else {
                other.to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = KrakenRestConfig::default();
        assert!(!config.has_credentials());
        assert_eq!(config.get_url(), REST_API_URL);
        assert!(!config.testnet);
    }

    #[test]
    fn test_config_with_credentials() {
        let config = KrakenRestConfig {
            api_key: "test_key".to_string(),
            api_secret: "test_secret".to_string(),
            ..Default::default()
        };
        assert!(config.has_credentials());
    }

    #[test]
    fn test_config_testnet() {
        let config = KrakenRestConfig {
            testnet: true,
            ..Default::default()
        };
        assert_eq!(config.get_url(), REST_DEMO_URL);
    }

    #[test]
    fn test_config_custom_url() {
        let config = KrakenRestConfig {
            custom_url: Some("https://custom.api.com".to_string()),
            testnet: true, // Should be ignored
            ..Default::default()
        };
        assert_eq!(config.get_url(), "https://custom.api.com");
    }

    #[test]
    fn test_to_kraken_pair() {
        assert_eq!(to_kraken_pair("BTC/USDT"), "BTCUSD");
        assert_eq!(to_kraken_pair("ETH/USDT"), "ETHUSD");
        assert_eq!(to_kraken_pair("BTC/USD"), "BTCUSD");
        assert_eq!(to_kraken_pair("ETH/EUR"), "ETHEUR");
    }

    #[test]
    fn test_normalize_currency() {
        assert_eq!(normalize_currency("ZUSD"), "USD");
        assert_eq!(normalize_currency("XXBT"), "BTC");
        assert_eq!(normalize_currency("XETH"), "ETH");
        assert_eq!(normalize_currency("SOL"), "SOL");
        assert_eq!(normalize_currency("ZEUR"), "EUR");
    }

    #[test]
    fn test_client_creation() {
        let config = KrakenRestConfig::default();
        let client = KrakenRestClient::new(config);
        assert!(!client.has_credentials());
    }

    #[tokio::test]
    async fn test_server_time() {
        // This test requires network access
        // In CI, you might want to skip or mock this
        let config = KrakenRestConfig::default();
        let client = KrakenRestClient::new(config);

        match client.get_server_time().await {
            Ok(time) => {
                assert!(time > 0);
                println!("Server time: {}", time);
            }
            Err(e) => {
                // Network might not be available in test env
                println!("Server time test skipped: {}", e);
            }
        }
    }
}
