//! Binance exchange adapter implementation
//!
//! This module implements the Exchange trait for Binance, providing order
//! execution, position management, and market data integration.
//!
//! # Authentication
//!
//! Binance uses HMAC-SHA256 signatures for authenticated endpoints.
//! All private endpoints require API key in header and signature in query params.
//!
//! # API Endpoints
//!
//! - **REST API**: `https://api.binance.com`
//! - **Testnet**: `https://testnet.binance.vision`
//!
//! # Symbol Format
//!
//! Binance uses uppercase concatenated format (e.g., `BTCUSD`, `ETHUSDT`)

use crate::error::{ExecutionError, Result};
use crate::exchanges::{
    Balance, Exchange, OrderStatusResponse, OrderUpdateReceiver, PositionUpdateReceiver,
};
use crate::execution::histogram::global_latency_histograms;
use crate::execution::metrics::retry_metrics;
use crate::types::{Order, OrderSide, OrderStatusEnum, OrderTypeEnum, Position};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use rust_decimal::Decimal;
use serde::Deserialize;
use sha2::Sha256;
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

type HmacSha256 = Hmac<Sha256>;

// Binance API endpoints
const BINANCE_REST_MAINNET: &str = "https://api.binance.com";
const BINANCE_REST_TESTNET: &str = "https://testnet.binance.vision";

// API Endpoints
const ENDPOINT_ORDER: &str = "/api/v3/order";
const ENDPOINT_OPEN_ORDERS: &str = "/api/v3/openOrders";
#[allow(dead_code)]
const ENDPOINT_ALL_ORDERS: &str = "/api/v3/allOrders";
const ENDPOINT_ACCOUNT: &str = "/api/v3/account";
#[allow(dead_code)]
const ENDPOINT_EXCHANGE_INFO: &str = "/api/v3/exchangeInfo";
const ENDPOINT_TICKER_PRICE: &str = "/api/v3/ticker/price";
const ENDPOINT_SERVER_TIME: &str = "/api/v3/time";

/// Binance order response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinanceOrderResponse {
    pub symbol: String,
    pub order_id: i64,
    pub order_list_id: Option<i64>,
    pub client_order_id: String,
    pub transact_time: Option<i64>,
    pub price: String,
    pub orig_qty: String,
    pub executed_qty: String,
    pub cummulative_quote_qty: String,
    pub status: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
    #[serde(default)]
    pub fills: Vec<BinanceFill>,
}

/// Binance fill information
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinanceFill {
    pub price: String,
    pub qty: String,
    pub commission: String,
    pub commission_asset: String,
    #[serde(default)]
    pub trade_id: Option<i64>,
}

/// Binance account information
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinanceAccountInfo {
    pub maker_commission: i32,
    pub taker_commission: i32,
    pub buyer_commission: i32,
    pub seller_commission: i32,
    pub can_trade: bool,
    pub can_withdraw: bool,
    pub can_deposit: bool,
    pub update_time: i64,
    pub account_type: String,
    pub balances: Vec<BinanceBalance>,
}

/// Binance balance entry
#[derive(Debug, Clone, Deserialize)]
pub struct BinanceBalance {
    pub asset: String,
    pub free: String,
    pub locked: String,
}

/// Binance order status response
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BinanceOrderStatus {
    pub symbol: String,
    pub order_id: i64,
    pub order_list_id: Option<i64>,
    pub client_order_id: String,
    pub price: String,
    pub orig_qty: String,
    pub executed_qty: String,
    pub cummulative_quote_qty: String,
    pub status: String,
    pub time_in_force: String,
    #[serde(rename = "type")]
    pub order_type: String,
    pub side: String,
    pub stop_price: Option<String>,
    pub iceberg_qty: Option<String>,
    pub time: i64,
    pub update_time: i64,
    pub is_working: bool,
    pub orig_quote_order_qty: Option<String>,
}

/// Binance exchange adapter
pub struct BinanceExchange {
    /// API credentials
    api_key: String,
    api_secret: String,

    /// HTTP client
    client: Client,

    /// Base URL for API requests
    base_url: String,

    /// Whether this is testnet
    testnet: bool,
}

impl BinanceExchange {
    /// Create a new Binance exchange adapter
    ///
    /// # Arguments
    /// * `api_key` - Binance API key
    /// * `api_secret` - Binance API secret
    /// * `testnet` - Whether to use testnet (true) or mainnet (false)
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let base_url = if testnet {
            BINANCE_REST_TESTNET.to_string()
        } else {
            BINANCE_REST_MAINNET.to_string()
        };

        info!(
            "Creating Binance exchange adapter (testnet: {}, base_url: {})",
            testnet, base_url
        );

        Self {
            api_key,
            api_secret,
            client: Client::new(),
            base_url,
            testnet,
        }
    }

    /// Create from environment variables
    ///
    /// Reads:
    /// - `BINANCE_API_KEY`
    /// - `BINANCE_API_SECRET`
    /// - `BINANCE_TESTNET` (optional, defaults to false)
    pub fn from_env() -> Self {
        let api_key = std::env::var("BINANCE_API_KEY").unwrap_or_default();
        let api_secret = std::env::var("BINANCE_API_SECRET").unwrap_or_default();
        let testnet = std::env::var("BINANCE_TESTNET")
            .unwrap_or_else(|_| "false".to_string())
            .parse()
            .unwrap_or(false);

        Self::new(api_key, api_secret, testnet)
    }

    /// Generate HMAC-SHA256 signature for API request
    fn sign(&self, query_string: &str) -> String {
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(query_string.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Get current timestamp in milliseconds
    fn timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Build query string from parameters
    fn build_query_string(params: &[(String, String)]) -> String {
        params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&")
    }

    /// Make a signed POST request to Binance API
    async fn post_signed(
        &self,
        endpoint: &str,
        mut params: Vec<(String, String)>,
    ) -> Result<serde_json::Value> {
        let timestamp = Self::timestamp_ms();
        params.push(("timestamp".to_string(), timestamp.to_string()));

        let query_string = Self::build_query_string(&params);
        let signature = self.sign(&query_string);
        params.push(("signature".to_string(), signature));

        let url = format!("{}{}", self.base_url, endpoint);
        let full_query = Self::build_query_string(&params);

        debug!("POST {}?{}", url, full_query);

        let response = self
            .client
            .post(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(full_query)
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        if !status.is_success() {
            error!("Binance API error: {} - {}", status, body);
            return Err(ExecutionError::exchange(
                "binance",
                format!("HTTP {}: {}", status, body),
            ));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        // Check for Binance error response
        if let Some(code) = json.get("code").and_then(|v| v.as_i64()) {
            if code != 0 {
                let msg = json
                    .get("msg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                error!("Binance error code {}: {}", code, msg);
                return Err(ExecutionError::exchange(
                    "binance",
                    format!("Error {}: {}", code, msg),
                ));
            }
        }

        Ok(json)
    }

    /// Make a signed GET request to Binance API
    async fn get_signed(
        &self,
        endpoint: &str,
        mut params: Vec<(String, String)>,
    ) -> Result<serde_json::Value> {
        let timestamp = Self::timestamp_ms();
        params.push(("timestamp".to_string(), timestamp.to_string()));

        let query_string = Self::build_query_string(&params);
        let signature = self.sign(&query_string);
        params.push(("signature".to_string(), signature));

        let url = format!("{}{}", self.base_url, endpoint);
        let full_query = Self::build_query_string(&params);

        debug!("GET {}?{}", url, full_query);

        let response = self
            .client
            .get(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .query(&params)
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        if !status.is_success() {
            error!("Binance API error: {} - {}", status, body);
            return Err(ExecutionError::exchange(
                "binance",
                format!("HTTP {}: {}", status, body),
            ));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        // Check for Binance error response
        if let Some(code) = json.get("code").and_then(|v| v.as_i64()) {
            let msg = json
                .get("msg")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            error!("Binance error code {}: {}", code, msg);
            return Err(ExecutionError::exchange(
                "binance",
                format!("Error {}: {}", code, msg),
            ));
        }

        Ok(json)
    }

    /// Make a signed DELETE request to Binance API
    async fn delete_signed(
        &self,
        endpoint: &str,
        mut params: Vec<(String, String)>,
    ) -> Result<serde_json::Value> {
        let timestamp = Self::timestamp_ms();
        params.push(("timestamp".to_string(), timestamp.to_string()));

        let query_string = Self::build_query_string(&params);
        let signature = self.sign(&query_string);
        params.push(("signature".to_string(), signature));

        let url = format!("{}{}", self.base_url, endpoint);
        let full_query = Self::build_query_string(&params);

        debug!("DELETE {}?{}", url, full_query);

        let response = self
            .client
            .delete(&url)
            .header("X-MBX-APIKEY", &self.api_key)
            .query(&params)
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        if !status.is_success() {
            error!("Binance API error: {} - {}", status, body);
            return Err(ExecutionError::exchange(
                "binance",
                format!("HTTP {}: {}", status, body),
            ));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        Ok(json)
    }

    /// Make an unsigned GET request to Binance API (for public endpoints)
    async fn get_public(&self, endpoint: &str) -> Result<serde_json::Value> {
        let url = format!("{}{}", self.base_url, endpoint);

        debug!("GET {}", url);

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        if !status.is_success() {
            error!("Binance API error: {} - {}", status, body);
            return Err(ExecutionError::exchange(
                "binance",
                format!("HTTP {}: {}", status, body),
            ));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        Ok(json)
    }

    /// Convert normalized symbol to Binance format
    ///
    /// Binance uses uppercase concatenated format: `BTCUSD`
    pub fn to_binance_symbol(normalized: &str) -> String {
        normalized.to_uppercase().replace("/", "")
    }

    /// Convert Binance symbol to normalized format
    ///
    /// Assumes common quote currencies (USDT, USDC, BTC, ETH, BNB)
    pub fn from_binance_symbol(binance_symbol: &str) -> String {
        let upper = binance_symbol.to_uppercase();
        let quotes = ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"];

        for quote in quotes {
            if upper.ends_with(quote) {
                let base = &upper[..upper.len() - quote.len()];
                if !base.is_empty() {
                    return format!("{}/{}", base, quote);
                }
            }
        }

        upper
    }

    /// Convert internal OrderType to Binance order type string
    fn order_type_to_binance(order_type: &OrderTypeEnum) -> &'static str {
        match order_type {
            OrderTypeEnum::Market => "MARKET",
            OrderTypeEnum::Limit => "LIMIT",
            OrderTypeEnum::StopLoss => "STOP_LOSS",
            OrderTypeEnum::StopLossLimit => "STOP_LOSS_LIMIT",
            OrderTypeEnum::TakeProfit => "TAKE_PROFIT",
            OrderTypeEnum::TakeProfitLimit => "TAKE_PROFIT_LIMIT",
            OrderTypeEnum::LimitMaker => "LIMIT_MAKER",
            OrderTypeEnum::StopMarket => "STOP_LOSS",
            OrderTypeEnum::StopLimit => "STOP_LOSS_LIMIT",
            OrderTypeEnum::TrailingStop => "TRAILING_STOP_MARKET",
        }
    }

    /// Convert internal OrderSide to Binance side string
    fn order_side_to_binance(side: &OrderSide) -> &'static str {
        match side {
            OrderSide::Buy => "BUY",
            OrderSide::Sell => "SELL",
        }
    }

    /// Parse Binance order status to internal status
    fn parse_order_status(status: &str) -> OrderStatusEnum {
        match status {
            "NEW" => OrderStatusEnum::New,
            "PARTIALLY_FILLED" => OrderStatusEnum::PartiallyFilled,
            "FILLED" => OrderStatusEnum::Filled,
            "CANCELED" => OrderStatusEnum::Cancelled,
            "PENDING_CANCEL" => OrderStatusEnum::PendingCancel,
            "REJECTED" => OrderStatusEnum::Rejected,
            "EXPIRED" => OrderStatusEnum::Expired,
            _ => OrderStatusEnum::New,
        }
    }

    /// Parse Binance side string to internal OrderSide
    #[allow(dead_code)]
    fn parse_order_side(side: &str) -> OrderSide {
        match side {
            "BUY" => OrderSide::Buy,
            "SELL" => OrderSide::Sell,
            _ => OrderSide::Buy,
        }
    }

    /// Convert timestamp (ms) to DateTime<Utc>
    fn timestamp_to_datetime(ts_ms: i64) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(ts_ms).unwrap_or_else(Utc::now)
    }

    /// Get server time (useful for debugging time sync issues)
    pub async fn get_server_time(&self) -> Result<i64> {
        let response = self.get_public(ENDPOINT_SERVER_TIME).await?;
        response
            .get("serverTime")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| {
                ExecutionError::exchange("binance", "Invalid server time response".to_string())
            })
    }

    /// Get current price for a symbol
    pub async fn get_price(&self, symbol: &str) -> Result<Decimal> {
        let binance_symbol = Self::to_binance_symbol(symbol);
        let url = format!("{}?symbol={}", ENDPOINT_TICKER_PRICE, binance_symbol);
        let response = self.get_public(&url).await?;

        let price_str = response
            .get("price")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ExecutionError::exchange("binance", "Invalid price response".to_string())
            })?;

        Decimal::from_str(price_str).map_err(|e| {
            ExecutionError::exchange("binance", format!("Failed to parse price: {}", e))
        })
    }

    /// Get account information
    pub async fn get_account_info(&self) -> Result<BinanceAccountInfo> {
        let response = self.get_signed(ENDPOINT_ACCOUNT, vec![]).await?;
        serde_json::from_value(response).map_err(|e| ExecutionError::Serialization(e.to_string()))
    }
}

#[async_trait]
impl Exchange for BinanceExchange {
    fn name(&self) -> &str {
        "binance"
    }

    fn is_testnet(&self) -> bool {
        self.testnet
    }

    async fn place_order(&self, order: &Order) -> Result<String> {
        let metrics = retry_metrics("binance_place_order");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        let binance_symbol = Self::to_binance_symbol(&order.symbol);

        let mut params = vec![
            ("symbol".to_string(), binance_symbol),
            (
                "side".to_string(),
                Self::order_side_to_binance(&order.side).to_string(),
            ),
            (
                "type".to_string(),
                Self::order_type_to_binance(&order.order_type).to_string(),
            ),
            ("quantity".to_string(), order.quantity.to_string()),
        ];

        // Add price for limit orders
        if matches!(
            order.order_type,
            OrderTypeEnum::Limit
                | OrderTypeEnum::StopLossLimit
                | OrderTypeEnum::TakeProfitLimit
                | OrderTypeEnum::LimitMaker
        ) {
            if let Some(price) = order.price {
                params.push(("price".to_string(), price.to_string()));
                params.push(("timeInForce".to_string(), "GTC".to_string()));
            }
        }

        // Add client order ID if provided
        if let Some(ref client_order_id) = order.client_order_id {
            params.push(("newClientOrderId".to_string(), client_order_id.clone()));
        }

        // Request full response to get fill information
        params.push(("newOrderRespType".to_string(), "FULL".to_string()));

        info!(
            "Placing {} {} order for {} {} at {:?}",
            order.side, order.order_type, order.quantity, order.symbol, order.price
        );

        let response = match self.post_signed(ENDPOINT_ORDER, params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_placement("binance", duration_ms as f64);
                histograms.record_api_call("binance_place_order", duration_ms as f64);
                let error_str = e.to_string();
                if error_str.contains("429")
                    || error_str.contains("rate limit")
                    || error_str.contains("-1015")
                {
                    metrics.record_rate_limit();
                } else if error_str.contains("Network") || error_str.contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let order_response: BinanceOrderResponse = match serde_json::from_value(response) {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_placement("binance", duration_ms as f64);
                histograms.record_api_call("binance_place_order", duration_ms as f64);
                return Err(ExecutionError::Serialization(e.to_string()));
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_order_placement("binance", duration_ms as f64);
        histograms.record_api_call("binance_place_order", duration_ms as f64);

        info!(
            "Order placed successfully: orderId={}, status={}, fills={} ({}ms)",
            order_response.order_id,
            order_response.status,
            order_response.fills.len(),
            duration_ms
        );

        Ok(order_response.order_id.to_string())
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let metrics = retry_metrics("binance_cancel_order");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        // Binance requires symbol for cancellation, but we only have order_id
        // This is a limitation - in practice, you'd store symbol with order_id
        // For now, we'll try to get order info first or require symbol in order_id format

        // Try parsing as "SYMBOL:ORDER_ID" format
        let (symbol, id) = if order_id.contains(':') {
            let parts: Vec<&str> = order_id.split(':').collect();
            (parts[0].to_string(), parts[1].to_string())
        } else {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            return Err(ExecutionError::exchange(
                "binance",
                "Order ID must be in format 'SYMBOL:ORDER_ID' for cancellation".to_string(),
            ));
        };

        let params = vec![("symbol".to_string(), symbol), ("orderId".to_string(), id)];

        match self.delete_signed(ENDPOINT_ORDER, params).await {
            Ok(_) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_success(1, duration_ms);
                histograms.record_order_cancellation("binance", duration_ms as f64);
                histograms.record_api_call("binance_cancel_order", duration_ms as f64);
                info!("Order cancelled: {} ({}ms)", order_id, duration_ms);
                Ok(())
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_cancellation("binance", duration_ms as f64);
                histograms.record_api_call("binance_cancel_order", duration_ms as f64);
                let error_str = e.to_string();
                if error_str.contains("429")
                    || error_str.contains("rate limit")
                    || error_str.contains("-1015")
                {
                    metrics.record_rate_limit();
                } else if error_str.contains("Network") || error_str.contains("timeout") {
                    metrics.record_network_error();
                }
                Err(e)
            }
        }
    }

    async fn cancel_all_orders(&self, symbol: Option<&str>) -> Result<Vec<String>> {
        let metrics = retry_metrics("binance_cancel_all_orders");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        let symbol = symbol.ok_or_else(|| {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            ExecutionError::exchange(
                "binance",
                "Symbol is required for cancel_all_orders".to_string(),
            )
        })?;

        let binance_symbol = Self::to_binance_symbol(symbol);
        let params = vec![("symbol".to_string(), binance_symbol)];

        let response = match self.delete_signed(ENDPOINT_OPEN_ORDERS, params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("binance_cancel_all_orders", duration_ms as f64);
                let error_str = e.to_string();
                if error_str.contains("429")
                    || error_str.contains("rate limit")
                    || error_str.contains("-1015")
                {
                    metrics.record_rate_limit();
                } else if error_str.contains("Network") || error_str.contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let cancelled: Vec<BinanceOrderResponse> = match serde_json::from_value(response) {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("binance_cancel_all_orders", duration_ms as f64);
                return Err(ExecutionError::Serialization(e.to_string()));
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("binance_cancel_all_orders", duration_ms as f64);

        let cancelled_ids: Vec<String> = cancelled.iter().map(|o| o.order_id.to_string()).collect();

        info!(
            "Cancelled {} orders for {} ({}ms)",
            cancelled_ids.len(),
            symbol,
            duration_ms
        );

        Ok(cancelled_ids)
    }

    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse> {
        let metrics = retry_metrics("binance_get_order_status");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        // Try parsing as "SYMBOL:ORDER_ID" format
        let (symbol, id) = if order_id.contains(':') {
            let parts: Vec<&str> = order_id.split(':').collect();
            (parts[0].to_string(), parts[1].to_string())
        } else {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            return Err(ExecutionError::exchange(
                "binance",
                "Order ID must be in format 'SYMBOL:ORDER_ID'".to_string(),
            ));
        };

        let params = vec![("symbol".to_string(), symbol), ("orderId".to_string(), id)];

        let response = match self.get_signed(ENDPOINT_ORDER, params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("binance_get_order_status", duration_ms as f64);
                let error_str = e.to_string();
                if error_str.contains("429")
                    || error_str.contains("rate limit")
                    || error_str.contains("-1015")
                {
                    metrics.record_rate_limit();
                } else if error_str.contains("Network") || error_str.contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let order: BinanceOrderStatus = match serde_json::from_value(response) {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("binance_get_order_status", duration_ms as f64);
                return Err(ExecutionError::Serialization(e.to_string()));
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("binance_get_order_status", duration_ms as f64);

        let quantity = Decimal::from_str(&order.orig_qty).unwrap_or_default();
        let filled_quantity = Decimal::from_str(&order.executed_qty).unwrap_or_default();
        let price = Decimal::from_str(&order.price).ok();

        let avg_fill_price = if filled_quantity > Decimal::ZERO {
            Decimal::from_str(&order.cummulative_quote_qty)
                .ok()
                .map(|quote_qty| quote_qty / filled_quantity)
        } else {
            None
        };

        Ok(OrderStatusResponse {
            order_id: order.order_id.to_string(),
            client_order_id: Some(order.client_order_id),
            symbol: Self::from_binance_symbol(&order.symbol),
            status: Self::parse_order_status(&order.status),
            quantity,
            filled_quantity,
            remaining_quantity: quantity - filled_quantity,
            price,
            average_fill_price: avg_fill_price,
            created_at: Self::timestamp_to_datetime(order.time),
            updated_at: Self::timestamp_to_datetime(order.update_time),
            fills: vec![],
        })
    }

    async fn get_active_orders(&self, symbol: Option<&str>) -> Result<Vec<OrderStatusResponse>> {
        let metrics = retry_metrics("binance_get_active_orders");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        let mut params = vec![];
        if let Some(sym) = symbol {
            params.push(("symbol".to_string(), Self::to_binance_symbol(sym)));
        }

        let response = match self.get_signed(ENDPOINT_OPEN_ORDERS, params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("binance_get_active_orders", duration_ms as f64);
                let error_str = e.to_string();
                if error_str.contains("429")
                    || error_str.contains("rate limit")
                    || error_str.contains("-1015")
                {
                    metrics.record_rate_limit();
                } else if error_str.contains("Network") || error_str.contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let orders: Vec<BinanceOrderStatus> = match serde_json::from_value(response) {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("binance_get_active_orders", duration_ms as f64);
                return Err(ExecutionError::Serialization(e.to_string()));
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("binance_get_active_orders", duration_ms as f64);

        Ok(orders
            .into_iter()
            .map(|order| {
                let quantity = Decimal::from_str(&order.orig_qty).unwrap_or_default();
                let filled_quantity = Decimal::from_str(&order.executed_qty).unwrap_or_default();
                let price = Decimal::from_str(&order.price).ok();

                let avg_fill_price = if filled_quantity > Decimal::ZERO {
                    Decimal::from_str(&order.cummulative_quote_qty)
                        .ok()
                        .map(|quote_qty| quote_qty / filled_quantity)
                } else {
                    None
                };

                OrderStatusResponse {
                    order_id: order.order_id.to_string(),
                    client_order_id: Some(order.client_order_id),
                    symbol: Self::from_binance_symbol(&order.symbol),
                    status: Self::parse_order_status(&order.status),
                    quantity,
                    filled_quantity,
                    remaining_quantity: quantity - filled_quantity,
                    price,
                    average_fill_price: avg_fill_price,
                    created_at: Self::timestamp_to_datetime(order.time),
                    updated_at: Self::timestamp_to_datetime(order.update_time),
                    fills: vec![],
                }
            })
            .collect())
    }

    async fn get_balance(&self) -> Result<Balance> {
        let metrics = retry_metrics("binance_get_balance");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        let account_info = match self.get_account_info().await {
            Ok(info) => info,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("binance_get_balance", duration_ms as f64);
                let error_str = e.to_string();
                if error_str.contains("429")
                    || error_str.contains("rate limit")
                    || error_str.contains("-1015")
                {
                    metrics.record_rate_limit();
                } else if error_str.contains("Network") || error_str.contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("binance_get_balance", duration_ms as f64);

        // Find USDT balance (primary trading currency)
        let usdt_balance = account_info
            .balances
            .iter()
            .find(|b| b.asset == "USDT")
            .map(|b| {
                let free = Decimal::from_str(&b.free).unwrap_or_default();
                let locked = Decimal::from_str(&b.locked).unwrap_or_default();
                Balance {
                    total: free + locked,
                    available: free,
                    used: locked,
                    currency: "USDT".to_string(),
                    timestamp: Utc::now(),
                }
            })
            .unwrap_or(Balance {
                total: Decimal::ZERO,
                available: Decimal::ZERO,
                used: Decimal::ZERO,
                currency: "USDT".to_string(),
                timestamp: Utc::now(),
            });

        Ok(usdt_balance)
    }

    async fn get_positions(&self, _symbol: Option<&str>) -> Result<Vec<Position>> {
        // Binance Spot doesn't have traditional "positions" like futures
        // Return empty for spot trading
        // For futures, would need to use /fapi/v2/positionRisk endpoint
        Ok(vec![])
    }

    async fn subscribe_order_updates(&self) -> Result<OrderUpdateReceiver> {
        // Note: For real-time order updates, you'd need to:
        // 1. Create a listen key via POST /api/v3/userDataStream
        // 2. Connect to wss://stream.binance.com:9443/ws/<listenKey>
        // 3. Keep the listen key alive with PUT /api/v3/userDataStream
        //
        // This is a placeholder that returns an unused channel
        warn!("subscribe_order_updates not fully implemented - use private WebSocket");
        let (_tx, rx) = mpsc::unbounded_channel();
        Ok(rx)
    }

    async fn subscribe_position_updates(&self) -> Result<PositionUpdateReceiver> {
        // Same as order updates - requires user data stream WebSocket
        warn!("subscribe_position_updates not fully implemented - use private WebSocket");
        let (_tx, rx) = mpsc::unbounded_channel();
        Ok(rx)
    }

    async fn health_check(&self) -> Result<()> {
        let metrics = retry_metrics("binance_health_check");
        let start = Instant::now();

        // Check server time to verify connectivity
        match self.get_server_time().await {
            Ok(_) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_success(1, duration_ms);
                Ok(())
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                let error_str = e.to_string();
                if error_str.contains("Network") || error_str.contains("timeout") {
                    metrics.record_network_error();
                }
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binance_creation() {
        let exchange =
            BinanceExchange::new("test_key".to_string(), "test_secret".to_string(), true);

        assert_eq!(exchange.name(), "binance");
        assert!(exchange.is_testnet());
    }

    #[test]
    fn test_symbol_conversion() {
        assert_eq!(BinanceExchange::to_binance_symbol("BTC/USDT"), "BTCUSDT");
        assert_eq!(BinanceExchange::to_binance_symbol("ETH/USDT"), "ETHUSDT");
        assert_eq!(BinanceExchange::to_binance_symbol("sol/usdt"), "SOLUSDT");
    }

    #[test]
    fn test_symbol_normalization() {
        assert_eq!(BinanceExchange::from_binance_symbol("BTCUSDT"), "BTC/USDT");
        assert_eq!(BinanceExchange::from_binance_symbol("ETHUSDC"), "ETH/USDC");
        assert_eq!(BinanceExchange::from_binance_symbol("SOLBNB"), "SOL/BNB");
    }

    #[test]
    fn test_order_type_conversion() {
        assert_eq!(
            BinanceExchange::order_type_to_binance(&OrderTypeEnum::Market),
            "MARKET"
        );
        assert_eq!(
            BinanceExchange::order_type_to_binance(&OrderTypeEnum::Limit),
            "LIMIT"
        );
        assert_eq!(
            BinanceExchange::order_type_to_binance(&OrderTypeEnum::StopLoss),
            "STOP_LOSS"
        );
    }

    #[test]
    fn test_order_side_conversion() {
        assert_eq!(
            BinanceExchange::order_side_to_binance(&OrderSide::Buy),
            "BUY"
        );
        assert_eq!(
            BinanceExchange::order_side_to_binance(&OrderSide::Sell),
            "SELL"
        );
    }

    #[test]
    fn test_status_parsing() {
        assert_eq!(
            BinanceExchange::parse_order_status("NEW"),
            OrderStatusEnum::New
        );
        assert_eq!(
            BinanceExchange::parse_order_status("FILLED"),
            OrderStatusEnum::Filled
        );
        assert_eq!(
            BinanceExchange::parse_order_status("CANCELED"),
            OrderStatusEnum::Cancelled
        );
        assert_eq!(
            BinanceExchange::parse_order_status("PARTIALLY_FILLED"),
            OrderStatusEnum::PartiallyFilled
        );
    }

    #[test]
    fn test_signature_generation() {
        let exchange = BinanceExchange::new(
            "vmPUZE6mv9SD5VNHk4HlWFsOr6aKE2zvsw0MuIgwCIPy6utIco14y7Ju91duEh8A".to_string(),
            "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j".to_string(),
            false,
        );

        // Test that signature is generated (exact value depends on timestamp)
        let query = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559";
        let signature = exchange.sign(query);

        // Should be hex string
        assert_eq!(signature.len(), 64);
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_query_string_builder() {
        let params = vec![
            ("symbol".to_string(), "BTCUSD".to_string()),
            ("side".to_string(), "BUY".to_string()),
        ];

        let query = BinanceExchange::build_query_string(&params);
        assert_eq!(query, "symbol=BTCUSD&side=BUY");
    }

    #[test]
    fn test_timestamp_conversion() {
        let ts = 1699999999999_i64;
        let dt = BinanceExchange::timestamp_to_datetime(ts);
        assert_eq!(dt.timestamp_millis(), ts);
    }
}
