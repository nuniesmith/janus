//! Bybit exchange adapter implementation
//!
//! This module implements the Exchange trait for Bybit, providing order
//! execution, position management, and market data integration.

use crate::error::{ExecutionError, Result};
use crate::exchanges::{
    Balance, Exchange, OrderStatusResponse, OrderUpdateReceiver, PositionUpdateReceiver,
};
use crate::execution::histogram::global_latency_histograms;
use crate::execution::metrics::retry_metrics;
use crate::types::{Order, OrderSide, OrderStatusEnum, OrderTypeEnum, Position, PositionSide};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use hmac::{Hmac, Mac};
use reqwest::Client;
use rust_decimal::Decimal;

use serde_json::json;
use sha2::Sha256;

use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

type HmacSha256 = Hmac<Sha256>;

// Bybit API endpoints
const BYBIT_REST_MAINNET: &str = "https://api.bybit.com";
const BYBIT_REST_TESTNET: &str = "https://api-testnet.bybit.com";

/// Bybit exchange adapter
pub struct BybitExchange {
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

impl BybitExchange {
    /// Create a new Bybit exchange adapter
    ///
    /// # Arguments
    /// * `api_key` - Bybit API key
    /// * `api_secret` - Bybit API secret
    /// * `testnet` - Whether to use testnet (true) or mainnet (false)
    pub fn new(api_key: String, api_secret: String, testnet: bool) -> Self {
        let base_url = if testnet {
            BYBIT_REST_TESTNET.to_string()
        } else {
            BYBIT_REST_MAINNET.to_string()
        };

        Self {
            api_key,
            api_secret,
            client: Client::new(),
            base_url,
            testnet,
        }
    }

    /// Generate HMAC-SHA256 signature for API request
    fn sign(&self, timestamp: u64, params: &str) -> String {
        let sign_str = format!("{}{}{}", timestamp, &self.api_key, params);
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())
            .expect("HMAC can take key of any size");
        mac.update(sign_str.as_bytes());
        hex::encode(mac.finalize().into_bytes())
    }

    /// Get current timestamp in milliseconds
    fn timestamp_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Make a signed POST request to Bybit API
    async fn post_signed(
        &self,
        endpoint: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let timestamp = Self::timestamp_ms();
        let params_str = params.to_string();
        let signature = self.sign(timestamp, &params_str);

        let url = format!("{}{}", self.base_url, endpoint);

        debug!("POST {}: {}", url, params_str);

        let response = self
            .client
            .post(&url)
            .header("X-BAPI-API-KEY", &self.api_key)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-SIGN", signature)
            .json(&params)
            .send()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| ExecutionError::Network(e.to_string()))?;

        if !status.is_success() {
            error!("Bybit API error: {} - {}", status, body);
            return Err(ExecutionError::exchange(
                "bybit",
                format!("HTTP {}: {}", status, body),
            ));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        // Check Bybit response code
        if let Some(ret_code) = json.get("retCode").and_then(|v| v.as_i64()) {
            if ret_code != 0 {
                let msg = json
                    .get("retMsg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                error!("Bybit error code {}: {}", ret_code, msg);
                return Err(ExecutionError::exchange(
                    "bybit",
                    format!("Error {}: {}", ret_code, msg),
                ));
            }
        }

        Ok(json)
    }

    /// Make a signed GET request to Bybit API
    async fn get_signed(
        &self,
        endpoint: &str,
        params: Vec<(String, String)>,
    ) -> Result<serde_json::Value> {
        let timestamp = Self::timestamp_ms();

        // Build query string
        let mut query_params = params.clone();
        let query_string = if query_params.is_empty() {
            String::new()
        } else {
            query_params.sort_by(|a, b| a.0.cmp(&b.0));
            query_params
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join("&")
        };

        let signature = self.sign(timestamp, &query_string);

        let url = format!("{}{}", self.base_url, endpoint);

        debug!("GET {}?{}", url, query_string);

        let response = self
            .client
            .get(&url)
            .header("X-BAPI-API-KEY", &self.api_key)
            .header("X-BAPI-TIMESTAMP", timestamp.to_string())
            .header("X-BAPI-SIGN", signature)
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
            error!("Bybit API error: {} - {}", status, body);
            return Err(ExecutionError::exchange(
                "bybit",
                format!("HTTP {}: {}", status, body),
            ));
        }

        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        // Check Bybit response code
        if let Some(ret_code) = json.get("retCode").and_then(|v| v.as_i64()) {
            if ret_code != 0 {
                let msg = json
                    .get("retMsg")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                error!("Bybit error code {}: {}", ret_code, msg);
                return Err(ExecutionError::exchange(
                    "bybit",
                    format!("Error {}: {}", ret_code, msg),
                ));
            }
        }

        Ok(json)
    }

    /// Convert internal OrderType to Bybit order type string
    fn order_type_to_bybit(order_type: &OrderTypeEnum) -> &'static str {
        match order_type {
            OrderTypeEnum::Market => "Market",
            OrderTypeEnum::Limit => "Limit",
            _ => "Limit", // Default to limit for unsupported types
        }
    }

    /// Convert internal OrderSide to Bybit side string
    fn order_side_to_bybit(side: &OrderSide) -> &'static str {
        match side {
            OrderSide::Buy => "Buy",
            OrderSide::Sell => "Sell",
        }
    }

    /// Parse Bybit order status to internal status
    fn parse_order_status(status: &str) -> OrderStatusEnum {
        match status {
            "New" => OrderStatusEnum::New,
            "PartiallyFilled" => OrderStatusEnum::PartiallyFilled,
            "Filled" => OrderStatusEnum::Filled,
            "Cancelled" => OrderStatusEnum::Cancelled,
            "Rejected" => OrderStatusEnum::Rejected,
            "PendingCancel" => OrderStatusEnum::PendingCancel,
            _ => OrderStatusEnum::New,
        }
    }

    /// Parse Bybit side string to internal OrderSide
    #[allow(dead_code)]
    fn parse_order_side(side: &str) -> OrderSide {
        match side {
            "Buy" => OrderSide::Buy,
            "Sell" => OrderSide::Sell,
            _ => OrderSide::Buy,
        }
    }
}

#[async_trait]
impl Exchange for BybitExchange {
    fn name(&self) -> &str {
        "bybit"
    }

    fn is_testnet(&self) -> bool {
        self.testnet
    }

    async fn place_order(&self, order: &Order) -> Result<String> {
        let metrics = retry_metrics("bybit_place_order");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        info!(
            "Placing order on Bybit: {} {} {} @ {:?}",
            order.side, order.quantity, order.symbol, order.price
        );

        let params = json!({
            "category": "linear",
            "symbol": order.symbol,
            "side": Self::order_side_to_bybit(&order.side),
            "orderType": Self::order_type_to_bybit(&order.order_type),
            "qty": order.quantity.to_string(),
            "price": order.price.map(|p| p.to_string()),
            "timeInForce": "GTC",
        });

        let response = match self.post_signed("/v5/order/create", params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_placement("bybit", duration_ms as f64);
                histograms.record_api_call("bybit_place_order", duration_ms as f64);
                if e.to_string().contains("rate limit") || e.to_string().contains("Rate limit") {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let order_id = response
            .get("result")
            .and_then(|r| r.get("orderId"))
            .and_then(|o| o.as_str())
            .ok_or_else(|| {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_placement("bybit", duration_ms as f64);
                histograms.record_api_call("bybit_place_order", duration_ms as f64);
                ExecutionError::exchange("bybit", "No order ID in response")
            })?
            .to_string();

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_order_placement("bybit", duration_ms as f64);
        histograms.record_api_call("bybit_place_order", duration_ms as f64);

        info!(
            "Order placed successfully: {} ({}ms)",
            order_id, duration_ms
        );

        Ok(order_id)
    }

    async fn cancel_order(&self, order_id: &str) -> Result<()> {
        let metrics = retry_metrics("bybit_cancel_order");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        info!("Cancelling order on Bybit: {}", order_id);

        let params = json!({
            "category": "linear",
            "orderId": order_id,
        });

        match self.post_signed("/v5/order/cancel", params).await {
            Ok(_) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_success(1, duration_ms);
                histograms.record_order_cancellation("bybit", duration_ms as f64);
                histograms.record_api_call("bybit_cancel_order", duration_ms as f64);
                info!(
                    "Order cancelled successfully: {} ({}ms)",
                    order_id, duration_ms
                );
                Ok(())
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_order_cancellation("bybit", duration_ms as f64);
                histograms.record_api_call("bybit_cancel_order", duration_ms as f64);
                if e.to_string().contains("rate limit") || e.to_string().contains("Rate limit") {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                Err(e)
            }
        }
    }

    async fn cancel_all_orders(&self, symbol: Option<&str>) -> Result<Vec<String>> {
        let metrics = retry_metrics("bybit_cancel_all_orders");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        info!("Cancelling all orders on Bybit for symbol: {:?}", symbol);

        let mut params = json!({
            "category": "linear",
        });

        if let Some(symbol) = symbol {
            params["symbol"] = json!(symbol);
        }

        let response = match self.post_signed("/v5/order/cancel-all", params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("bybit_cancel_all_orders", duration_ms as f64);
                if e.to_string().contains("rate limit") || e.to_string().contains("Rate limit") {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        // Bybit returns list of cancelled order IDs
        let cancelled_ids: Vec<String> = response
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|item| item.get("orderId").and_then(|id| id.as_str()))
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("bybit_cancel_all_orders", duration_ms as f64);

        info!(
            "Cancelled {} orders ({}ms)",
            cancelled_ids.len(),
            duration_ms
        );

        Ok(cancelled_ids)
    }

    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatusResponse> {
        let metrics = retry_metrics("bybit_get_order_status");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        debug!("Getting order status from Bybit: {}", order_id);

        let params = vec![
            ("category".to_string(), "linear".to_string()),
            ("orderId".to_string(), order_id.to_string()),
        ];

        let response = match self.get_signed("/v5/order/realtime", params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("bybit_get_order_status", duration_ms as f64);
                if e.to_string().contains("rate limit") || e.to_string().contains("Rate limit") {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let order_data = response
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                ExecutionError::OrderNotFound(order_id.to_string())
            })?;

        // Parse order details
        let symbol = order_data
            .get("symbol")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let status_str = order_data
            .get("orderStatus")
            .and_then(|v| v.as_str())
            .unwrap_or("New");

        let qty_str = order_data
            .get("qty")
            .and_then(|v| v.as_str())
            .unwrap_or("0");
        let filled_str = order_data
            .get("cumExecQty")
            .and_then(|v| v.as_str())
            .unwrap_or("0");

        let quantity = Decimal::from_str_exact(qty_str)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;
        let filled_quantity = Decimal::from_str_exact(filled_str)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        let price = order_data
            .get("price")
            .and_then(|v| v.as_str())
            .and_then(|s| Decimal::from_str_exact(s).ok());

        let avg_price = order_data
            .get("avgPrice")
            .and_then(|v| v.as_str())
            .and_then(|s| Decimal::from_str_exact(s).ok());

        let created_at = order_data
            .get("createdTime")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .and_then(DateTime::from_timestamp_millis)
            .unwrap_or_else(Utc::now);

        let updated_at = order_data
            .get("updatedTime")
            .and_then(|v| v.as_str())
            .and_then(|s| s.parse::<i64>().ok())
            .and_then(DateTime::from_timestamp_millis)
            .unwrap_or_else(Utc::now);

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("bybit_get_order_status", duration_ms as f64);

        Ok(OrderStatusResponse {
            order_id: order_id.to_string(),
            client_order_id: order_data
                .get("orderLinkId")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            symbol,
            status: Self::parse_order_status(status_str),
            quantity,
            filled_quantity,
            remaining_quantity: quantity - filled_quantity,
            price,
            average_fill_price: avg_price,
            created_at,
            updated_at,
            fills: vec![], // Bybit doesn't return fills in order status
        })
    }

    async fn get_active_orders(&self, symbol: Option<&str>) -> Result<Vec<OrderStatusResponse>> {
        let metrics = retry_metrics("bybit_get_active_orders");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        debug!("Getting active orders from Bybit: {:?}", symbol);

        let mut params = vec![("category".to_string(), "linear".to_string())];

        if let Some(symbol) = symbol {
            params.push(("symbol".to_string(), symbol.to_string()));
        }

        let response = match self.get_signed("/v5/order/realtime", params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("bybit_get_active_orders", duration_ms as f64);
                if e.to_string().contains("rate limit") || e.to_string().contains("Rate limit") {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let orders = response
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .ok_or_else(|| {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                ExecutionError::Internal("No orders in response".to_string())
            })?;

        let mut result = Vec::new();

        for order_data in orders {
            if let Some(order_id) = order_data.get("orderId").and_then(|v| v.as_str()) {
                // Reuse get_order_status parsing logic
                match self.get_order_status(order_id).await {
                    Ok(order) => result.push(order),
                    Err(e) => warn!("Failed to parse order {}: {}", order_id, e),
                }
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("bybit_get_active_orders", duration_ms as f64);

        Ok(result)
    }

    async fn get_balance(&self) -> Result<Balance> {
        let metrics = retry_metrics("bybit_get_balance");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        debug!("Getting balance from Bybit");

        let params = vec![("accountType".to_string(), "CONTRACT".to_string())];

        let response = match self.get_signed("/v5/account/wallet-balance", params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("bybit_get_balance", duration_ms as f64);
                if e.to_string().contains("rate limit") || e.to_string().contains("Rate limit") {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("bybit_get_balance", duration_ms as f64);

        let account = response
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .and_then(|arr| arr.first())
            .ok_or_else(|| ExecutionError::Internal("No account data in response".to_string()))?;

        // Get USDT coin balance
        let coin = account
            .get("coin")
            .and_then(|c| c.as_array())
            .and_then(|arr| {
                arr.iter()
                    .find(|c| c.get("coin").and_then(|v| v.as_str()) == Some("USDT"))
            })
            .ok_or_else(|| ExecutionError::Internal("No USDT balance found".to_string()))?;

        let total_str = coin
            .get("walletBalance")
            .and_then(|v| v.as_str())
            .unwrap_or("0");
        let available_str = coin
            .get("availableToWithdraw")
            .and_then(|v| v.as_str())
            .unwrap_or("0");

        let total = Decimal::from_str_exact(total_str)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;
        let available = Decimal::from_str_exact(available_str)
            .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

        Ok(Balance {
            total,
            available,
            used: total - available,
            currency: "USDT".to_string(),
            timestamp: Utc::now(),
        })
    }

    async fn get_positions(&self, symbol: Option<&str>) -> Result<Vec<Position>> {
        let metrics = retry_metrics("bybit_get_positions");
        let histograms = global_latency_histograms();
        let start = Instant::now();

        debug!("Getting positions from Bybit: {:?}", symbol);

        let mut params = vec![
            ("category".to_string(), "linear".to_string()),
            ("settleCoin".to_string(), "USDT".to_string()),
        ];

        if let Some(symbol) = symbol {
            params.push(("symbol".to_string(), symbol.to_string()));
        }

        let response = match self.get_signed("/v5/position/list", params).await {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                histograms.record_api_call("bybit_get_positions", duration_ms as f64);
                if e.to_string().contains("rate limit") || e.to_string().contains("Rate limit") {
                    metrics.record_rate_limit();
                } else if e.to_string().contains("Network") || e.to_string().contains("timeout") {
                    metrics.record_network_error();
                }
                return Err(e);
            }
        };

        let duration_ms = start.elapsed().as_millis() as u64;
        metrics.record_success(1, duration_ms);
        histograms.record_api_call("bybit_get_positions", duration_ms as f64);

        let positions = response
            .get("result")
            .and_then(|r| r.get("list"))
            .and_then(|l| l.as_array())
            .ok_or_else(|| ExecutionError::Internal("No positions in response".to_string()))?;

        let mut result = Vec::new();

        for pos_data in positions {
            let symbol = pos_data
                .get("symbol")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();

            let size_str = pos_data.get("size").and_then(|v| v.as_str()).unwrap_or("0");
            let size = Decimal::from_str_exact(size_str)
                .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

            // Skip zero positions
            if size == Decimal::ZERO {
                continue;
            }

            let side_str = pos_data
                .get("side")
                .and_then(|v| v.as_str())
                .unwrap_or("Buy");
            let side = if side_str == "Buy" {
                PositionSide::Long
            } else {
                PositionSide::Short
            };

            let avg_price_str = pos_data
                .get("avgPrice")
                .and_then(|v| v.as_str())
                .unwrap_or("0");
            let mark_price_str = pos_data
                .get("markPrice")
                .and_then(|v| v.as_str())
                .unwrap_or("0");
            let unrealized_pnl_str = pos_data
                .get("unrealisedPnl")
                .and_then(|v| v.as_str())
                .unwrap_or("0");

            let average_entry_price = Decimal::from_str_exact(avg_price_str)
                .map_err(|e| ExecutionError::Serialization(e.to_string()))?;
            let current_price = Decimal::from_str_exact(mark_price_str)
                .map_err(|e| ExecutionError::Serialization(e.to_string()))?;
            let unrealized_pnl = Decimal::from_str_exact(unrealized_pnl_str)
                .map_err(|e| ExecutionError::Serialization(e.to_string()))?;

            result.push(Position {
                symbol,
                exchange: "bybit".to_string(),
                side,
                quantity: size,
                average_entry_price,
                current_price,
                unrealized_pnl,
                realized_pnl: Decimal::ZERO,
                margin_used: Decimal::ZERO,
                liquidation_price: None,
                opened_at: Utc::now(),
                updated_at: Utc::now(),
            });
        }

        Ok(result)
    }

    async fn subscribe_order_updates(&self) -> Result<OrderUpdateReceiver> {
        // WebSocket implementation would go here
        // For now, return a channel that never receives updates
        let (_tx, rx) = mpsc::unbounded_channel();
        warn!("WebSocket order updates not yet implemented for Bybit");
        Ok(rx)
    }

    async fn subscribe_position_updates(&self) -> Result<PositionUpdateReceiver> {
        // WebSocket implementation would go here
        // For now, return a channel that never receives updates
        let (_tx, rx) = mpsc::unbounded_channel();
        warn!("WebSocket position updates not yet implemented for Bybit");
        Ok(rx)
    }

    async fn health_check(&self) -> Result<()> {
        let metrics = retry_metrics("bybit_health_check");
        let start = Instant::now();

        debug!("Performing health check for Bybit");

        // Try to get server time as a simple health check
        let response = match self
            .client
            .get(format!("{}/v5/market/time", self.base_url))
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                let duration_ms = start.elapsed().as_millis() as u64;
                metrics.record_failure(1, duration_ms);
                metrics.record_network_error();
                return Err(ExecutionError::Network(e.to_string()));
            }
        };

        if response.status().is_success() {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_success(1, duration_ms);
            Ok(())
        } else {
            let duration_ms = start.elapsed().as_millis() as u64;
            metrics.record_failure(1, duration_ms);
            Err(ExecutionError::Internal("Health check failed".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bybit_creation() {
        let exchange = BybitExchange::new("test_key".to_string(), "test_secret".to_string(), true);

        assert_eq!(exchange.name(), "bybit");
        assert!(exchange.is_testnet());
    }

    #[test]
    fn test_order_type_conversion() {
        assert_eq!(
            BybitExchange::order_type_to_bybit(&OrderTypeEnum::Market),
            "Market"
        );
        assert_eq!(
            BybitExchange::order_type_to_bybit(&OrderTypeEnum::Limit),
            "Limit"
        );
    }

    #[test]
    fn test_order_side_conversion() {
        assert_eq!(BybitExchange::order_side_to_bybit(&OrderSide::Buy), "Buy");
        assert_eq!(BybitExchange::order_side_to_bybit(&OrderSide::Sell), "Sell");
    }

    #[test]
    fn test_status_parsing() {
        assert_eq!(
            BybitExchange::parse_order_status("New"),
            OrderStatusEnum::New
        );
        assert_eq!(
            BybitExchange::parse_order_status("Filled"),
            OrderStatusEnum::Filled
        );
        assert_eq!(
            BybitExchange::parse_order_status("Cancelled"),
            OrderStatusEnum::Cancelled
        );
    }

    #[test]
    fn test_signature_generation() {
        let exchange = BybitExchange::new("test_key".to_string(), "test_secret".to_string(), true);

        let timestamp = 1234567890;
        let params = "test_params";
        let signature = exchange.sign(timestamp, params);

        // Signature should be a hex string
        assert!(!signature.is_empty());
        assert!(signature.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
