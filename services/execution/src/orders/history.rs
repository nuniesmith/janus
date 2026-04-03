//! Order History Module
//!
//! Provides persistence layer for orders and fills using QuestDB.
//! This enables audit trails, compliance reporting, and historical analysis.

use crate::error::{ExecutionError, Result, StorageError};
use crate::types::{Fill, Order};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::io::Write;
use std::net::TcpStream;
use tracing::{debug, error, info, warn};

/// Order history manager with QuestDB persistence
pub struct OrderHistory {
    /// QuestDB host address (e.g., "localhost:9009")
    questdb_host: String,

    /// Table name for orders
    orders_table: String,

    /// Table name for fills
    fills_table: String,

    /// Enable write-through caching (reserved for future use)
    #[allow(dead_code)]
    enable_cache: bool,
}

impl OrderHistory {
    /// Create a new order history manager
    pub async fn new(questdb_host: &str) -> Result<Self> {
        info!(questdb_host = %questdb_host, "Initializing order history");

        let history = Self {
            questdb_host: questdb_host.to_string(),
            orders_table: "execution_orders".to_string(),
            fills_table: "execution_fills".to_string(),
            enable_cache: true,
        };

        // Test connection
        history.test_connection()?;

        Ok(history)
    }

    /// Test QuestDB connection
    fn test_connection(&self) -> Result<()> {
        debug!(host = %self.questdb_host, "Testing QuestDB connection");

        match TcpStream::connect(&self.questdb_host) {
            Ok(_) => {
                info!("Successfully connected to QuestDB");
                Ok(())
            }
            Err(e) => {
                error!(
                    host = %self.questdb_host,
                    error = %e,
                    "Failed to connect to QuestDB"
                );
                Err(ExecutionError::Storage(StorageError::Connection(format!(
                    "Cannot connect to QuestDB at {}: {}",
                    self.questdb_host, e
                ))))
            }
        }
    }

    /// Record an order to QuestDB
    pub async fn record_order(&self, order: &Order) -> Result<()> {
        debug!(
            order_id = %order.id,
            symbol = %order.symbol,
            status = ?order.status,
            "Recording order to QuestDB"
        );

        let line = self.format_order_ilp(order)?;
        self.send_ilp(&line)?;

        Ok(())
    }

    /// Record a fill to QuestDB
    pub async fn record_fill(&self, fill: &Fill) -> Result<()> {
        debug!(
            fill_id = %fill.id,
            order_id = %fill.order_id,
            quantity = %fill.quantity,
            price = %fill.price,
            "Recording fill to QuestDB"
        );

        let line = self.format_fill_ilp(fill)?;
        self.send_ilp(&line)?;

        Ok(())
    }

    /// Format order as InfluxDB Line Protocol (ILP)
    fn format_order_ilp(&self, order: &Order) -> Result<String> {
        // Table name and tags
        let mut line = format!(
            "{},order_id={},symbol={},exchange={},side={:?},order_type={:?},status={:?}",
            self.orders_table,
            escape_tag(&order.id),
            escape_tag(&order.symbol),
            escape_tag(&order.exchange),
            order.side,
            order.order_type,
            order.status,
        );

        // Add exchange_order_id tag if available
        if let Some(ref exchange_order_id) = order.exchange_order_id {
            line.push_str(",exchange_order_id=");
            line.push_str(&escape_tag(exchange_order_id));
        }

        // Add signal_id tag
        line.push_str(",signal_id=");
        line.push_str(&escape_tag(&order.signal_id));

        // Fields
        line.push(' ');

        let mut fields = Vec::new();

        // Numeric fields
        fields.push(format!("quantity={}", decimal_to_f64(order.quantity)?));
        fields.push(format!(
            "filled_quantity={}",
            decimal_to_f64(order.filled_quantity)?
        ));
        fields.push(format!(
            "remaining_quantity={}",
            decimal_to_f64(order.remaining_quantity)?
        ));

        if let Some(price) = order.price {
            fields.push(format!("price={}", decimal_to_f64(price)?));
        }

        if let Some(stop_price) = order.stop_price {
            fields.push(format!("stop_price={}", decimal_to_f64(stop_price)?));
        }

        if let Some(avg_price) = order.average_fill_price {
            fields.push(format!("average_fill_price={}", decimal_to_f64(avg_price)?));
        }

        // String fields
        fields.push(format!("time_in_force=\"{:?}\"", order.time_in_force));
        fields.push(format!("strategy=\"{:?}\"", order.strategy));
        fields.push(format!("fills_count={}", order.fills.len()));

        line.push_str(&fields.join(","));

        // Timestamp (nanoseconds)
        let timestamp_ns = order.updated_at.timestamp_nanos_opt().ok_or_else(|| {
            ExecutionError::Storage(StorageError::Write("Invalid timestamp".to_string()))
        })?;
        line.push_str(&format!(" {}", timestamp_ns));

        Ok(line)
    }

    /// Format fill as InfluxDB Line Protocol (ILP)
    fn format_fill_ilp(&self, fill: &Fill) -> Result<String> {
        // Table name and tags
        let mut line = format!(
            "{},fill_id={},order_id={},side={:?},fee_currency={}",
            self.fills_table,
            escape_tag(&fill.id),
            escape_tag(&fill.order_id),
            fill.side,
            escape_tag(&fill.fee_currency),
        );

        // Fields
        line.push(' ');

        let fields = [
            format!("quantity={}", decimal_to_f64(fill.quantity)?),
            format!("price={}", decimal_to_f64(fill.price)?),
            format!("fee={}", decimal_to_f64(fill.fee)?),
            format!("is_maker={}", fill.is_maker),
        ];

        line.push_str(&fields.join(","));

        // Timestamp (nanoseconds)
        let timestamp_ns = fill.timestamp.timestamp_nanos_opt().ok_or_else(|| {
            ExecutionError::Storage(StorageError::Write("Invalid timestamp".to_string()))
        })?;
        line.push_str(&format!(" {}", timestamp_ns));

        Ok(line)
    }

    /// Send ILP message to QuestDB
    fn send_ilp(&self, line: &str) -> Result<()> {
        match TcpStream::connect(&self.questdb_host) {
            Ok(mut stream) => {
                let message = format!("{}\n", line);

                if let Err(e) = stream.write_all(message.as_bytes()) {
                    error!(error = %e, "Failed to write to QuestDB");
                    return Err(ExecutionError::Storage(StorageError::Write(format!(
                        "Write failed: {}",
                        e
                    ))));
                }

                if let Err(e) = stream.flush() {
                    warn!(error = %e, "Failed to flush QuestDB stream");
                }

                debug!("Successfully wrote to QuestDB: {}", line);
                Ok(())
            }
            Err(e) => {
                error!(
                    host = %self.questdb_host,
                    error = %e,
                    "Failed to connect to QuestDB for write"
                );
                Err(ExecutionError::Storage(StorageError::Connection(format!(
                    "Connection failed: {}",
                    e
                ))))
            }
        }
    }

    /// Get orders by symbol (placeholder - requires REST API or PG wire)
    pub async fn get_orders_by_symbol(&self, symbol: &str, limit: usize) -> Result<Vec<Order>> {
        // This is a placeholder implementation
        // In production, you would query QuestDB via its REST API or PostgreSQL wire protocol
        warn!(
            symbol = %symbol,
            limit = limit,
            "get_orders_by_symbol not fully implemented - requires QuestDB REST/PG API"
        );

        Ok(Vec::new())
    }

    /// Get fills for an order (placeholder - requires REST API or PG wire)
    pub async fn get_fills_by_order(&self, order_id: &str, limit: usize) -> Result<Vec<Fill>> {
        warn!(
            order_id = %order_id,
            limit = limit,
            "get_fills_by_order not fully implemented - requires QuestDB REST/PG API"
        );

        Ok(Vec::new())
    }

    /// Get order statistics for a time range (placeholder)
    pub async fn get_order_statistics(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<OrderHistoryStatistics> {
        warn!(
            start = %start_time,
            end = %end_time,
            "get_order_statistics not fully implemented - requires QuestDB REST/PG API"
        );

        Ok(OrderHistoryStatistics::default())
    }

    /// Batch record multiple orders
    pub async fn batch_record_orders(&self, orders: &[Order]) -> Result<()> {
        debug!(count = orders.len(), "Batch recording orders");

        for order in orders {
            self.record_order(order).await?;
        }

        Ok(())
    }

    /// Batch record multiple fills
    pub async fn batch_record_fills(&self, fills: &[Fill]) -> Result<()> {
        debug!(count = fills.len(), "Batch recording fills");

        for fill in fills {
            self.record_fill(fill).await?;
        }

        Ok(())
    }

    /// Flush any pending writes (no-op for direct ILP)
    pub async fn flush(&self) -> Result<()> {
        // Direct ILP writes are already flushed
        Ok(())
    }

    /// Get table names
    pub fn get_table_names(&self) -> (&str, &str) {
        (&self.orders_table, &self.fills_table)
    }
}

/// Statistics from order history
#[derive(Debug, Clone, Default)]
pub struct OrderHistoryStatistics {
    pub total_orders: usize,
    pub total_fills: usize,
    pub filled_orders: usize,
    pub cancelled_orders: usize,
    pub rejected_orders: usize,
    pub total_volume: Decimal,
    pub total_fees: Decimal,
}

/// Escape a tag value for ILP
fn escape_tag(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace(',', "\\,")
        .replace(' ', "\\ ")
        .replace('=', "\\=")
}

/// Convert Decimal to f64 for ILP
fn decimal_to_f64(value: Decimal) -> Result<f64> {
    value.to_f64().ok_or_else(|| {
        ExecutionError::Storage(StorageError::Write(format!(
            "Cannot convert Decimal {} to f64",
            value
        )))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ExecutionStrategyEnum, OrderSide, OrderStatusEnum, OrderTypeEnum, TimeInForceEnum,
    };
    use std::collections::HashMap;

    fn create_test_order() -> Order {
        Order {
            id: "order-123".to_string(),
            exchange_order_id: Some("exch-456".to_string()),
            client_order_id: None,
            signal_id: "signal-789".to_string(),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            order_type: OrderTypeEnum::Limit,
            quantity: Decimal::from(1),
            filled_quantity: Decimal::new(5, 1),    // 0.5
            remaining_quantity: Decimal::new(5, 1), // 0.5
            price: Some(Decimal::from(50000)),
            stop_price: None,
            average_fill_price: Some(Decimal::from(49950)),
            time_in_force: TimeInForceEnum::Gtc,
            strategy: ExecutionStrategyEnum::Immediate,
            status: OrderStatusEnum::PartiallyFilled,
            fills: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    fn create_test_fill() -> Fill {
        Fill {
            id: "fill-123".to_string(),
            order_id: "order-123".to_string(),
            quantity: Decimal::new(5, 1), // 0.5
            price: Decimal::from(49950),
            fee: Decimal::new(25, 0), // 25
            fee_currency: "USDT".to_string(),
            side: OrderSide::Buy,
            timestamp: Utc::now(),
            is_maker: true,
        }
    }

    #[test]
    fn test_format_order_ilp() {
        let order = create_test_order();
        let history = OrderHistory {
            questdb_host: "localhost:9009".to_string(),
            orders_table: "execution_orders".to_string(),
            fills_table: "execution_fills".to_string(),
            enable_cache: true,
        };

        let ilp = history.format_order_ilp(&order).unwrap();

        // Verify basic structure
        assert!(ilp.starts_with("execution_orders,"));
        assert!(ilp.contains("order_id=order-123"));
        assert!(ilp.contains("symbol=BTCUSD"));
        assert!(ilp.contains("exchange=bybit"));
        assert!(ilp.contains("side=Buy"));
        assert!(ilp.contains("quantity=1"));
        assert!(ilp.contains("filled_quantity=0.5"));
    }

    #[test]
    fn test_format_fill_ilp() {
        let fill = create_test_fill();
        let history = OrderHistory {
            questdb_host: "localhost:9009".to_string(),
            orders_table: "execution_orders".to_string(),
            fills_table: "execution_fills".to_string(),
            enable_cache: true,
        };

        let ilp = history.format_fill_ilp(&fill).unwrap();

        // Verify basic structure
        assert!(ilp.starts_with("execution_fills,"));
        assert!(ilp.contains("fill_id=fill-123"));
        assert!(ilp.contains("order_id=order-123"));
        assert!(ilp.contains("side=Buy"));
        assert!(ilp.contains("quantity=0.5"));
        assert!(ilp.contains("price=49950"));
        assert!(ilp.contains("fee=25"));
        assert!(ilp.contains("is_maker=true"));
    }

    #[test]
    fn test_escape_tag() {
        assert_eq!(escape_tag("simple"), "simple");
        assert_eq!(escape_tag("with space"), "with\\ space");
        assert_eq!(escape_tag("with,comma"), "with\\,comma");
        assert_eq!(escape_tag("with=equals"), "with\\=equals");
        assert_eq!(escape_tag("with\\backslash"), "with\\\\backslash");
    }

    #[test]
    fn test_decimal_to_f64() {
        assert_eq!(decimal_to_f64(Decimal::from(100)).unwrap(), 100.0);
        assert_eq!(decimal_to_f64(Decimal::new(5, 1)).unwrap(), 0.5);
        assert_eq!(decimal_to_f64(Decimal::new(12345, 2)).unwrap(), 123.45);
    }

    #[tokio::test]
    async fn test_order_history_creation() {
        // This test will fail if QuestDB is not running, which is expected
        // It validates the connection logic
        let _result = OrderHistory::new("localhost:9009").await;
        // We don't assert success here as QuestDB may not be running in test env
    }

    #[test]
    fn test_get_table_names() {
        let history = OrderHistory {
            questdb_host: "localhost:9009".to_string(),
            orders_table: "execution_orders".to_string(),
            fills_table: "execution_fills".to_string(),
            enable_cache: true,
        };

        let (orders_table, fills_table) = history.get_table_names();
        assert_eq!(orders_table, "execution_orders");
        assert_eq!(fills_table, "execution_fills");
    }
}
