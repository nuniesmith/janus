//! gRPC Service Implementation for FKS Execution Service
//!
//! This module implements the ExecutionService gRPC interface defined in execution.proto.
//! It provides methods for order submission, status queries, position management, and streaming updates.

use crate::execution::histogram::global_latency_histograms;
use crate::orders::OrderManager;
use crate::types::{
    ExecutionStrategyEnum, Order, OrderSide, OrderStatusEnum, OrderTypeEnum, TimeInForceEnum,
};
use chrono::Utc;
use fks_proto::common::{HealthCheckRequest, HealthCheckResponse, OrderType, Side, TimeInForce};
use fks_proto::execution::{
    CancelAllOrdersRequest, CancelAllOrdersResponse, CancelOrderRequest, CancelOrderResponse,
    ExecutionStrategy, Fill, GetAccountRequest, GetAccountResponse, GetActiveOrdersRequest,
    GetActiveOrdersResponse, GetOrderStatusRequest, GetOrderStatusResponse, GetPositionsRequest,
    GetPositionsResponse, OrderStatus, OrderUpdate, StreamUpdatesRequest, StreamUpdatesResponse,
    SubmitSignalRequest, SubmitSignalResponse, UpdateType,
    execution_service_server::{ExecutionService, ExecutionServiceServer},
    stream_updates_response,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::broadcast;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

/// gRPC service implementation
pub struct ExecutionServiceImpl {
    /// Order manager
    order_manager: Arc<OrderManager>,

    /// Update broadcast channel
    update_tx: broadcast::Sender<StreamUpdatesResponse>,
}

impl ExecutionServiceImpl {
    /// Create a new gRPC service
    pub fn new(order_manager: Arc<OrderManager>) -> Self {
        let (update_tx, _) = broadcast::channel(1000);

        Self {
            order_manager,
            update_tx,
        }
    }

    /// Create a gRPC server
    pub fn into_server(self) -> ExecutionServiceServer<Self> {
        ExecutionServiceServer::new(self)
    }

    /// Broadcast an execution update
    fn broadcast_update(&self, update: StreamUpdatesResponse) {
        if let Err(e) = self.update_tx.send(update) {
            debug!("No active stream subscribers, skipping broadcast: {}", e);
        }
    }

    /// Convert internal order to gRPC GetOrderStatusResponse
    fn order_to_response(&self, order: &Order) -> GetOrderStatusResponse {
        GetOrderStatusResponse {
            order_id: order.exchange_order_id.clone().unwrap_or_default(),
            internal_order_id: order.id.clone(),
            symbol: order.symbol.clone(),
            exchange: order.exchange.clone(),
            side: order_side_to_proto(order.side) as i32,
            order_type: order_type_to_proto(order.order_type) as i32,
            status: order_status_to_proto(order.status) as i32,
            quantity: decimal_to_f64(order.quantity),
            filled_quantity: decimal_to_f64(order.filled_quantity),
            remaining_quantity: decimal_to_f64(order.remaining_quantity),
            price: order.price.map(decimal_to_f64),
            average_fill_price: order.average_fill_price.map(decimal_to_f64),
            created_at: order.created_at.timestamp_millis(),
            updated_at: Some(order.updated_at.timestamp_millis()),
            fills: order
                .fills
                .iter()
                .map(|f| Fill {
                    fill_id: f.id.clone(),
                    order_id: f.order_id.clone(),
                    quantity: decimal_to_f64(f.quantity),
                    price: decimal_to_f64(f.price),
                    fee: decimal_to_f64(f.fee),
                    fee_currency: f.fee_currency.clone(),
                    side: order_side_to_proto(f.side) as i32,
                    timestamp: f.timestamp.timestamp_millis(),
                    is_maker: f.is_maker,
                })
                .collect(),
        }
    }
}

#[tonic::async_trait]
impl ExecutionService for ExecutionServiceImpl {
    /// Submit a trading signal for execution
    async fn submit_signal(
        &self,
        request: Request<SubmitSignalRequest>,
    ) -> std::result::Result<Response<SubmitSignalResponse>, Status> {
        let start = Instant::now();
        let histograms = global_latency_histograms();
        let req = request.into_inner();

        info!(
            signal_id = %req.signal_id,
            symbol = %req.symbol,
            side = ?req.side,
            quantity = req.quantity,
            "Received signal"
        );

        // Convert protobuf to internal types
        let side = match Side::try_from(req.side) {
            Ok(Side::Buy) => OrderSide::Buy,
            Ok(Side::Sell) => OrderSide::Sell,
            _ => {
                return Err(Status::invalid_argument("Invalid side"));
            }
        };

        let order_type = match OrderType::try_from(req.order_type) {
            Ok(OrderType::Market) => OrderTypeEnum::Market,
            Ok(OrderType::Limit) => OrderTypeEnum::Limit,
            Ok(OrderType::StopMarket) => OrderTypeEnum::StopMarket,
            Ok(OrderType::StopLimit) => OrderTypeEnum::StopLimit,
            Ok(OrderType::TrailingStop) => OrderTypeEnum::TrailingStop,
            _ => {
                return Err(Status::invalid_argument("Invalid order type"));
            }
        };

        let time_in_force = match TimeInForce::try_from(req.time_in_force) {
            Ok(TimeInForce::Gtc) => TimeInForceEnum::Gtc,
            Ok(TimeInForce::Ioc) => TimeInForceEnum::Ioc,
            Ok(TimeInForce::Fok) => TimeInForceEnum::Fok,
            Ok(TimeInForce::Gtd) => TimeInForceEnum::Gtd,
            _ => TimeInForceEnum::Gtc,
        };

        let strategy = match ExecutionStrategy::try_from(req.strategy) {
            Ok(ExecutionStrategy::Immediate) => ExecutionStrategyEnum::Immediate,
            Ok(ExecutionStrategy::Twap) => ExecutionStrategyEnum::Twap,
            Ok(ExecutionStrategy::Vwap) => ExecutionStrategyEnum::Vwap,
            Ok(ExecutionStrategy::AlmgrenChriss) => ExecutionStrategyEnum::AlmgrenChriss,
            Ok(ExecutionStrategy::Iceberg) => ExecutionStrategyEnum::Iceberg,
            _ => ExecutionStrategyEnum::Immediate,
        };

        // Create order
        let mut order = Order::new(
            req.signal_id,
            req.symbol,
            req.exchange,
            side,
            order_type,
            Decimal::from_f64_retain(req.quantity)
                .ok_or_else(|| Status::invalid_argument("Invalid quantity"))?,
        );

        order.time_in_force = time_in_force;
        order.strategy = strategy;

        if let Some(price) = req.price {
            order.price = Some(
                Decimal::from_f64_retain(price)
                    .ok_or_else(|| Status::invalid_argument("Invalid price"))?,
            );
        }

        if let Some(stop_price) = req.stop_price {
            order.stop_price = Some(
                Decimal::from_f64_retain(stop_price)
                    .ok_or_else(|| Status::invalid_argument("Invalid stop price"))?,
            );
        }

        // Add metadata
        order.metadata = req.metadata;

        // Submit order
        match self.order_manager.submit_order(order.clone()).await {
            Ok(order_id) => {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                histograms.record_order_placement("grpc", duration_ms);
                histograms.record_api_call("grpc_submit_signal", duration_ms);
                info!(order_id = %order_id, latency_ms = duration_ms, "Order submitted successfully");

                // Broadcast update
                let update = StreamUpdatesResponse {
                    r#type: UpdateType::Order as i32,
                    timestamp: Utc::now().timestamp_millis(),
                    update: Some(stream_updates_response::Update::OrderUpdate(OrderUpdate {
                        order_id: order.exchange_order_id.clone().unwrap_or_default(),
                        internal_order_id: order_id.clone(),
                        symbol: order.symbol.clone(),
                        old_status: OrderStatus::New as i32,
                        new_status: OrderStatus::Submitted as i32,
                        message: "Order submitted".to_string(),
                    })),
                };
                self.broadcast_update(update);

                Ok(Response::new(SubmitSignalResponse {
                    success: true,
                    order_id: order.exchange_order_id.unwrap_or_default(),
                    internal_order_id: order_id,
                    message: "Order submitted successfully".to_string(),
                    timestamp: Utc::now().timestamp_millis(),
                    status: OrderStatus::Submitted as i32,
                }))
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                histograms.record_order_placement("grpc", duration_ms);
                histograms.record_api_call("grpc_submit_signal", duration_ms);
                error!(error = %e, latency_ms = duration_ms, "Failed to submit order");
                Err(e.to_grpc_status())
            }
        }
    }

    /// Get order status
    async fn get_order_status(
        &self,
        request: Request<GetOrderStatusRequest>,
    ) -> std::result::Result<Response<GetOrderStatusResponse>, Status> {
        let start = Instant::now();
        let histograms = global_latency_histograms();
        let req = request.into_inner();

        debug!(order_id = %req.order_id, "Getting order status");

        let result = match self.order_manager.get_order(&req.order_id) {
            Ok(order) => Ok(Response::new(self.order_to_response(&order))),
            Err(e) => {
                warn!(order_id = %req.order_id, error = %e, "Order not found");
                Err(e.to_grpc_status())
            }
        };

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        histograms.record_api_call("grpc_get_order_status", duration_ms);
        result
    }

    /// Get all active orders
    async fn get_active_orders(
        &self,
        request: Request<GetActiveOrdersRequest>,
    ) -> std::result::Result<Response<GetActiveOrdersResponse>, Status> {
        let start = Instant::now();
        let histograms = global_latency_histograms();
        let req = request.into_inner();

        debug!(
            symbol = ?req.symbol,
            exchange = ?req.exchange,
            "Getting active orders"
        );

        let mut orders = self.order_manager.get_active_orders();

        // Filter by symbol if provided
        if let Some(ref symbol) = req.symbol {
            orders.retain(|o| o.symbol == *symbol);
        }

        // Filter by exchange if provided
        if let Some(ref exchange) = req.exchange {
            orders.retain(|o| o.exchange == *exchange);
        }

        let response = GetActiveOrdersResponse {
            orders: orders.iter().map(|o| self.order_to_response(o)).collect(),
        };

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        histograms.record_api_call("grpc_get_active_orders", duration_ms);
        Ok(Response::new(response))
    }

    /// Cancel an order
    async fn cancel_order(
        &self,
        request: Request<CancelOrderRequest>,
    ) -> std::result::Result<Response<CancelOrderResponse>, Status> {
        let start = Instant::now();
        let histograms = global_latency_histograms();
        let req = request.into_inner();

        info!(order_id = %req.order_id, "Cancelling order");

        match self.order_manager.cancel_order(&req.order_id).await {
            Ok(()) => {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                histograms.record_order_cancellation("grpc", duration_ms);
                histograms.record_api_call("grpc_cancel_order", duration_ms);
                // Broadcast update
                let update = StreamUpdatesResponse {
                    r#type: UpdateType::Order as i32,
                    timestamp: Utc::now().timestamp_millis(),
                    update: Some(stream_updates_response::Update::OrderUpdate(OrderUpdate {
                        order_id: req.order_id.clone(),
                        internal_order_id: req.order_id.clone(),
                        symbol: String::new(),
                        old_status: OrderStatus::Submitted as i32,
                        new_status: OrderStatus::Cancelled as i32,
                        message: "Order cancelled".to_string(),
                    })),
                };
                self.broadcast_update(update);

                Ok(Response::new(CancelOrderResponse {
                    success: true,
                    order_id: req.order_id,
                    message: "Order cancelled successfully".to_string(),
                    timestamp: Utc::now().timestamp_millis(),
                }))
            }
            Err(e) => {
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                histograms.record_order_cancellation("grpc", duration_ms);
                histograms.record_api_call("grpc_cancel_order", duration_ms);
                error!(order_id = %req.order_id, error = %e, latency_ms = duration_ms, "Failed to cancel order");
                Err(e.to_grpc_status())
            }
        }
    }

    /// Cancel all orders for a symbol
    async fn cancel_all_orders(
        &self,
        request: Request<CancelAllOrdersRequest>,
    ) -> std::result::Result<Response<CancelAllOrdersResponse>, Status> {
        let start = Instant::now();
        let histograms = global_latency_histograms();
        let req = request.into_inner();

        info!(
            symbol = ?req.symbol,
            exchange = ?req.exchange,
            "Cancelling all orders"
        );

        let mut orders = self.order_manager.get_active_orders();

        // Filter by symbol if provided
        if let Some(ref symbol) = req.symbol {
            orders.retain(|o| o.symbol == *symbol);
        }

        // Filter by exchange if provided
        if let Some(ref exchange) = req.exchange {
            orders.retain(|o| o.exchange == *exchange);
        }

        let mut cancelled_count = 0;
        let mut cancelled_order_ids = Vec::new();

        for order in orders {
            match self.order_manager.cancel_order(&order.id).await {
                Ok(()) => {
                    cancelled_count += 1;
                    cancelled_order_ids.push(order.id);
                }
                Err(e) => {
                    warn!(order_id = %order.id, error = %e, "Failed to cancel order");
                }
            }
        }

        let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
        histograms.record_api_call("grpc_cancel_all_orders", duration_ms);

        Ok(Response::new(CancelAllOrdersResponse {
            success: true,
            cancelled_count,
            message: format!("Cancelled {} orders", cancelled_count),
            cancelled_order_ids,
        }))
    }

    /// Get current positions
    async fn get_positions(
        &self,
        request: Request<GetPositionsRequest>,
    ) -> std::result::Result<Response<GetPositionsResponse>, Status> {
        let req = request.into_inner();

        debug!(
            exchange = ?req.exchange,
            symbol = ?req.symbol,
            "Getting positions"
        );

        // Get positions from order manager's position tracker
        let all_positions = self.order_manager.get_all_positions().await;

        // Filter by exchange and/or symbol if specified
        let filtered_positions: Vec<_> = all_positions
            .into_iter()
            .filter(|pos| {
                let exchange_match = req.exchange.as_ref().map_or(true, |e| pos.exchange == *e);
                let symbol_match = req.symbol.as_ref().map_or(true, |s| pos.symbol == *s);
                exchange_match && symbol_match
            })
            .collect();

        // Convert to protobuf format
        let positions: Vec<fks_proto::execution::Position> = filtered_positions
            .iter()
            .map(|pos| fks_proto::execution::Position {
                symbol: pos.symbol.clone(),
                exchange: pos.exchange.clone(),
                side: match pos.side {
                    crate::positions::PositionSide::Long => Side::Buy as i32,
                    crate::positions::PositionSide::Short => Side::Sell as i32,
                    crate::positions::PositionSide::Flat => 0,
                },
                quantity: decimal_to_f64(pos.size.abs()),
                average_entry_price: decimal_to_f64(pos.entry_price),
                current_price: decimal_to_f64(pos.mark_price),
                unrealized_pnl: decimal_to_f64(pos.unrealized_pnl),
                realized_pnl: decimal_to_f64(pos.realized_pnl),
                margin_used: decimal_to_f64(pos.initial_margin),
                liquidation_price: pos.liquidation_price.map(decimal_to_f64).unwrap_or(0.0),
                opened_at: pos.updated_at.timestamp_millis(),
                updated_at: Some(pos.updated_at.timestamp_millis()),
            })
            .collect();

        // Calculate total P&L
        let total_pnl: f64 = filtered_positions
            .iter()
            .map(|pos| decimal_to_f64(pos.unrealized_pnl + pos.realized_pnl))
            .sum();

        info!(
            "Returning {} positions with total PnL: {:.2}",
            positions.len(),
            total_pnl
        );

        Ok(Response::new(GetPositionsResponse {
            positions,
            total_pnl,
        }))
    }

    /// Get account information
    async fn get_account(
        &self,
        request: Request<GetAccountRequest>,
    ) -> std::result::Result<Response<GetAccountResponse>, Status> {
        let req = request.into_inner();

        debug!(exchange = ?req.exchange, "Getting account information");

        // Get all positions to calculate account-level metrics
        let all_positions = self.order_manager.get_all_positions().await;

        // Filter by exchange if specified
        let filtered_positions: Vec<crate::positions::Position> = match &req.exchange {
            Some(exchange) if !exchange.is_empty() => all_positions
                .into_iter()
                .filter(|pos| pos.exchange == *exchange)
                .collect(),
            _ => all_positions,
        };

        // Calculate aggregated metrics
        let mut total_unrealized_pnl = Decimal::ZERO;
        let mut total_realized_pnl = Decimal::ZERO;
        let mut total_margin_used = Decimal::ZERO;
        let mut balances_by_exchange: HashMap<String, f64> = HashMap::new();

        for pos in &filtered_positions {
            total_unrealized_pnl += pos.unrealized_pnl;
            total_realized_pnl += pos.realized_pnl;
            total_margin_used += pos.initial_margin;

            // Track position value by exchange
            let entry = balances_by_exchange
                .entry(pos.exchange.clone())
                .or_insert(0.0);
            *entry += decimal_to_f64(pos.position_value);
        }

        // Get total balance from order manager (this would come from exchange APIs in production)
        // For now, estimate based on position values and PnL
        let total_position_value: Decimal =
            filtered_positions.iter().map(|p| p.position_value).sum();

        // Estimate balance (in production, this comes from exchange balance queries)
        let estimated_balance = total_position_value + total_unrealized_pnl + total_realized_pnl;
        let available_balance = estimated_balance - total_margin_used;

        info!(
            "Account info: balance={:.2}, available={:.2}, margin={:.2}, unrealized_pnl={:.2}",
            decimal_to_f64(estimated_balance),
            decimal_to_f64(available_balance),
            decimal_to_f64(total_margin_used),
            decimal_to_f64(total_unrealized_pnl)
        );

        Ok(Response::new(GetAccountResponse {
            balance: decimal_to_f64(estimated_balance),
            available_balance: decimal_to_f64(available_balance),
            margin_used: decimal_to_f64(total_margin_used),
            unrealized_pnl: decimal_to_f64(total_unrealized_pnl),
            realized_pnl: decimal_to_f64(total_realized_pnl),
            balances_by_exchange,
        }))
    }

    /// Associated type for streaming updates
    type StreamUpdatesStream = std::pin::Pin<
        Box<
            dyn tokio_stream::Stream<Item = std::result::Result<StreamUpdatesResponse, Status>>
                + Send,
        >,
    >;

    /// Stream execution updates
    async fn stream_updates(
        &self,
        request: Request<StreamUpdatesRequest>,
    ) -> std::result::Result<Response<Self::StreamUpdatesStream>, Status> {
        let req = request.into_inner();

        info!(
            symbols = ?req.symbols,
            types = ?req.types,
            "Starting update stream"
        );

        let rx = self.update_tx.subscribe();
        let stream = BroadcastStream::new(rx).filter_map(|result| match result {
            Ok(update) => Some(Ok(update)),
            Err(e) => {
                warn!("Broadcast error: {}", e);
                None
            }
        });

        Ok(Response::new(Box::pin(stream)))
    }

    /// Health check
    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> std::result::Result<Response<HealthCheckResponse>, Status> {
        let mut components = HashMap::new();

        // Check order manager
        let stats = self.order_manager.get_statistics();
        components.insert(
            "order_manager".to_string(),
            format!("OK ({} active orders)", stats.active_orders),
        );

        // Overall health
        let healthy = true;
        let status = if healthy { "OK" } else { "DEGRADED" };

        Ok(Response::new(HealthCheckResponse {
            healthy,
            status: status.to_string(),
            components,
            uptime_seconds: 0,
            version: String::new(),
            timestamp: Utc::now().timestamp_millis(),
        }))
    }
}

// ============================================================================
// Conversion Functions
// ============================================================================

fn order_side_to_proto(side: OrderSide) -> Side {
    match side {
        OrderSide::Buy => Side::Buy,
        OrderSide::Sell => Side::Sell,
    }
}

fn order_type_to_proto(order_type: OrderTypeEnum) -> OrderType {
    match order_type {
        OrderTypeEnum::Market => OrderType::Market,
        OrderTypeEnum::Limit => OrderType::Limit,
        OrderTypeEnum::StopMarket => OrderType::StopMarket,
        OrderTypeEnum::StopLimit => OrderType::StopLimit,
        OrderTypeEnum::TrailingStop => OrderType::TrailingStop,
        // Map Binance-specific types to closest equivalents
        OrderTypeEnum::StopLoss => OrderType::StopMarket,
        OrderTypeEnum::StopLossLimit => OrderType::StopLimit,
        OrderTypeEnum::TakeProfit => OrderType::StopMarket,
        OrderTypeEnum::TakeProfitLimit => OrderType::StopLimit,
        OrderTypeEnum::LimitMaker => OrderType::Limit,
    }
}

fn order_status_to_proto(status: OrderStatusEnum) -> OrderStatus {
    match status {
        OrderStatusEnum::New => OrderStatus::New,
        OrderStatusEnum::Submitted => OrderStatus::Submitted,
        OrderStatusEnum::PartiallyFilled => OrderStatus::PartiallyFilled,
        OrderStatusEnum::Filled => OrderStatus::Filled,
        OrderStatusEnum::Cancelled => OrderStatus::Cancelled,
        OrderStatusEnum::Rejected => OrderStatus::Rejected,
        OrderStatusEnum::Expired => OrderStatus::Expired,
        OrderStatusEnum::PendingCancel => OrderStatus::PendingCancel,
    }
}

fn decimal_to_f64(value: Decimal) -> f64 {
    value.to_f64().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_side_conversion() {
        assert_eq!(order_side_to_proto(OrderSide::Buy), Side::Buy);
        assert_eq!(order_side_to_proto(OrderSide::Sell), Side::Sell);
    }

    #[test]
    fn test_order_type_conversion() {
        assert_eq!(
            order_type_to_proto(OrderTypeEnum::Market),
            OrderType::Market
        );
        assert_eq!(order_type_to_proto(OrderTypeEnum::Limit), OrderType::Limit);
    }

    #[test]
    fn test_order_status_conversion() {
        assert_eq!(
            order_status_to_proto(OrderStatusEnum::New),
            OrderStatus::New
        );
        assert_eq!(
            order_status_to_proto(OrderStatusEnum::Filled),
            OrderStatus::Filled
        );
    }

    #[test]
    fn test_decimal_to_f64() {
        assert_eq!(decimal_to_f64(Decimal::from(100)), 100.0);
        assert_eq!(decimal_to_f64(Decimal::new(5, 1)), 0.5);
    }
}
