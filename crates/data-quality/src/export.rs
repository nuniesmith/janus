//! Parquet Export Module
//!
//! Provides functionality for exporting market data quality metrics
//! and validated data to Parquet files for long-term storage and analytics.
//!
//! # Features
//!
//! - Buffered batch writing for efficiency
//! - Schema management for market data types
//! - Compression support (Snappy, Gzip, Zstd)
//! - Partitioning by date/symbol/exchange
//! - Automatic file rotation
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_data_quality::export::{ParquetExporter, ExportConfig};
//!
//! let config = ExportConfig::builder()
//!     .output_dir("/data/exports")
//!     .partition_by_date(true)
//!     .compression(Compression::Snappy)
//!     .build();
//!
//! let exporter = ParquetExporter::new(config)?;
//! exporter.write_trade(&trade_event).await?;
//! exporter.flush().await?;
//! ```

use std::collections::HashMap;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, BooleanBuilder, Float64Builder, StringBuilder, TimestampMicrosecondBuilder,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use chrono::{DateTime, Datelike, Utc};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;

use janus_core::{MarketDataEvent, OrderBookEvent, TradeEvent};

use crate::ValidationResult;
use crate::error::{DataQualityError, Result};

/// Configuration for Parquet export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Base output directory
    pub output_dir: PathBuf,

    /// Enable partitioning by date
    pub partition_by_date: bool,

    /// Enable partitioning by symbol
    pub partition_by_symbol: bool,

    /// Enable partitioning by exchange
    pub partition_by_exchange: bool,

    /// Compression type
    pub compression: CompressionType,

    /// Maximum rows per file before rotation
    pub max_rows_per_file: usize,

    /// Maximum file size in bytes before rotation
    pub max_file_size_bytes: u64,

    /// Buffer size for batch writing
    pub buffer_size: usize,

    /// File prefix
    pub file_prefix: String,

    /// Include validation results in export
    pub include_validation: bool,
}

/// Compression type for Parquet files
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum CompressionType {
    /// No compression
    None,
    /// Snappy compression (fast, moderate compression)
    #[default]
    Snappy,
    /// Gzip compression (slower, better compression)
    Gzip,
    /// Zstd compression (good balance)
    Zstd,
    /// LZ4 compression (very fast)
    Lz4,
}

impl From<CompressionType> for Compression {
    fn from(ct: CompressionType) -> Self {
        match ct {
            CompressionType::None => Compression::UNCOMPRESSED,
            CompressionType::Snappy => Compression::SNAPPY,
            CompressionType::Gzip => Compression::GZIP(Default::default()),
            CompressionType::Zstd => Compression::ZSTD(Default::default()),
            CompressionType::Lz4 => Compression::LZ4,
        }
    }
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./exports"),
            partition_by_date: true,
            partition_by_symbol: false,
            partition_by_exchange: true,
            compression: CompressionType::Snappy,
            max_rows_per_file: 1_000_000,
            max_file_size_bytes: 100 * 1024 * 1024, // 100 MB
            buffer_size: 10_000,
            file_prefix: "market_data".to_string(),
            include_validation: true,
        }
    }
}

impl ExportConfig {
    /// Create a new configuration builder
    pub fn builder() -> ExportConfigBuilder {
        ExportConfigBuilder::default()
    }
}

/// Builder for ExportConfig
#[derive(Debug, Default)]
pub struct ExportConfigBuilder {
    config: ExportConfig,
}

impl ExportConfigBuilder {
    /// Set the output directory
    pub fn output_dir(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.output_dir = path.into();
        self
    }

    /// Enable/disable date partitioning
    pub fn partition_by_date(mut self, enabled: bool) -> Self {
        self.config.partition_by_date = enabled;
        self
    }

    /// Enable/disable symbol partitioning
    pub fn partition_by_symbol(mut self, enabled: bool) -> Self {
        self.config.partition_by_symbol = enabled;
        self
    }

    /// Enable/disable exchange partitioning
    pub fn partition_by_exchange(mut self, enabled: bool) -> Self {
        self.config.partition_by_exchange = enabled;
        self
    }

    /// Set compression type
    pub fn compression(mut self, compression: CompressionType) -> Self {
        self.config.compression = compression;
        self
    }

    /// Set maximum rows per file
    pub fn max_rows_per_file(mut self, rows: usize) -> Self {
        self.config.max_rows_per_file = rows;
        self
    }

    /// Set maximum file size
    pub fn max_file_size_bytes(mut self, bytes: u64) -> Self {
        self.config.max_file_size_bytes = bytes;
        self
    }

    /// Set buffer size
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.config.buffer_size = size;
        self
    }

    /// Set file prefix
    pub fn file_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.file_prefix = prefix.into();
        self
    }

    /// Include validation results
    pub fn include_validation(mut self, include: bool) -> Self {
        self.config.include_validation = include;
        self
    }

    /// Build the configuration
    pub fn build(self) -> ExportConfig {
        self.config
    }
}

/// Trade record for buffering before Parquet write
#[derive(Debug, Clone)]
struct TradeRecord {
    timestamp: i64,
    received_at: i64,
    exchange: String,
    symbol: String,
    price: f64,
    quantity: f64,
    side: String,
    trade_id: String,
    buyer_is_maker: Option<bool>,
    is_valid: Option<bool>,
    validation_errors: Option<String>,
}

/// Order book record for buffering
#[derive(Debug, Clone)]
struct OrderBookRecord {
    timestamp: i64,
    exchange: String,
    symbol: String,
    best_bid_price: f64,
    best_bid_qty: f64,
    best_ask_price: f64,
    best_ask_qty: f64,
    spread: f64,
    mid_price: f64,
    bid_depth: f64,
    ask_depth: f64,
    is_valid: Option<bool>,
    validation_errors: Option<String>,
}

/// Parquet exporter for market data
pub struct ParquetExporter {
    config: ExportConfig,
    trade_buffers: Arc<Mutex<HashMap<String, Vec<TradeRecord>>>>,
    order_book_buffers: Arc<Mutex<HashMap<String, Vec<OrderBookRecord>>>>,
    trade_schema: Arc<Schema>,
    order_book_schema: Arc<Schema>,
}

impl ParquetExporter {
    /// Create a new Parquet exporter
    pub fn new(config: ExportConfig) -> Result<Self> {
        // Ensure output directory exists
        fs::create_dir_all(&config.output_dir).map_err(|e| {
            DataQualityError::IoError(format!(
                "Failed to create output directory {:?}: {}",
                config.output_dir, e
            ))
        })?;

        let trade_schema = Self::create_trade_schema(config.include_validation);
        let order_book_schema = Self::create_order_book_schema(config.include_validation);

        Ok(Self {
            config,
            trade_buffers: Arc::new(Mutex::new(HashMap::new())),
            order_book_buffers: Arc::new(Mutex::new(HashMap::new())),
            trade_schema: Arc::new(trade_schema),
            order_book_schema: Arc::new(order_book_schema),
        })
    }

    /// Create schema for trade data
    fn create_trade_schema(include_validation: bool) -> Schema {
        let mut fields = vec![
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "received_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("exchange", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
            Field::new("quantity", DataType::Float64, false),
            Field::new("side", DataType::Utf8, false),
            Field::new("trade_id", DataType::Utf8, false),
            Field::new("buyer_is_maker", DataType::Boolean, true),
        ];

        if include_validation {
            fields.push(Field::new("is_valid", DataType::Boolean, true));
            fields.push(Field::new("validation_errors", DataType::Utf8, true));
        }

        Schema::new(fields)
    }

    /// Create schema for order book data
    fn create_order_book_schema(include_validation: bool) -> Schema {
        let mut fields = vec![
            Field::new(
                "timestamp",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "received_at",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("exchange", DataType::Utf8, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("best_bid_price", DataType::Float64, false),
            Field::new("best_bid_qty", DataType::Float64, false),
            Field::new("best_ask_price", DataType::Float64, false),
            Field::new("best_ask_qty", DataType::Float64, false),
            Field::new("spread", DataType::Float64, false),
            Field::new("mid_price", DataType::Float64, false),
            Field::new("bid_depth", DataType::Float64, false),
            Field::new("ask_depth", DataType::Float64, false),
        ];

        if include_validation {
            fields.push(Field::new("is_valid", DataType::Boolean, true));
            fields.push(Field::new("validation_errors", DataType::Utf8, true));
        }

        Schema::new(fields)
    }

    /// Generate partition path for a given timestamp, exchange, and symbol
    fn partition_path(
        &self,
        timestamp: DateTime<Utc>,
        exchange: &str,
        symbol: &str,
        data_type: &str,
    ) -> PathBuf {
        let mut path = self.config.output_dir.clone();
        path.push(data_type);

        if self.config.partition_by_exchange {
            path.push(format!("exchange={}", exchange.to_lowercase()));
        }

        if self.config.partition_by_symbol {
            path.push(format!("symbol={}", symbol.replace("/", "_")));
        }

        if self.config.partition_by_date {
            path.push(format!(
                "year={}/month={:02}/day={:02}",
                timestamp.year(),
                timestamp.month(),
                timestamp.day()
            ));
        }

        path
    }

    /// Generate a unique filename for a Parquet file
    fn generate_filename(&self, data_type: &str) -> String {
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S_%3f");
        let uuid = uuid::Uuid::new_v4().to_string()[..8].to_string();
        format!(
            "{}_{}_{}_{}.parquet",
            self.config.file_prefix, data_type, timestamp, uuid
        )
    }

    /// Get the partition key for buffering
    fn get_partition_key(&self, timestamp: i64, exchange: &str, symbol: &str) -> String {
        let dt = DateTime::from_timestamp_micros(timestamp).unwrap_or_else(Utc::now);

        let mut parts = vec![];

        if self.config.partition_by_exchange {
            parts.push(exchange.to_lowercase());
        }

        if self.config.partition_by_symbol {
            parts.push(symbol.replace("/", "_"));
        }

        if self.config.partition_by_date {
            parts.push(format!("{}", dt.format("%Y-%m-%d")));
        }

        if parts.is_empty() {
            "default".to_string()
        } else {
            parts.join("_")
        }
    }

    /// Write a trade event to the buffer
    pub async fn write_trade(
        &self,
        trade: &TradeEvent,
        validation: Option<&ValidationResult>,
    ) -> Result<()> {
        let record = TradeRecord {
            timestamp: trade.timestamp,
            received_at: trade.received_at,
            exchange: format!("{:?}", trade.exchange),
            symbol: trade.symbol.to_string(),
            price: trade.price.to_string().parse().unwrap_or(0.0),
            quantity: trade.quantity.to_string().parse().unwrap_or(0.0),
            side: format!("{:?}", trade.side),
            trade_id: trade.trade_id.clone(),
            buyer_is_maker: trade.buyer_is_maker,
            is_valid: validation.map(|v| v.is_valid),
            validation_errors: validation.map(|v| v.errors.join("; ")),
        };

        let partition_key =
            self.get_partition_key(trade.timestamp, &record.exchange, &record.symbol);

        let mut buffers = self.trade_buffers.lock().await;
        let buffer = buffers
            .entry(partition_key.clone())
            .or_insert_with(Vec::new);
        buffer.push(record);

        // Check if buffer needs flushing
        if buffer.len() >= self.config.buffer_size {
            let records = std::mem::take(buffer);
            drop(buffers);
            self.flush_trade_buffer(&partition_key, records).await?;
        }

        Ok(())
    }

    /// Write an order book event to the buffer
    pub async fn write_order_book(
        &self,
        order_book: &OrderBookEvent,
        validation: Option<&ValidationResult>,
    ) -> Result<()> {
        let (best_bid_price, best_bid_qty) = order_book
            .bids
            .first()
            .map(|l| {
                (
                    l.price.to_string().parse().unwrap_or(0.0),
                    l.quantity.to_string().parse().unwrap_or(0.0),
                )
            })
            .unwrap_or((0.0, 0.0));

        let (best_ask_price, best_ask_qty) = order_book
            .asks
            .first()
            .map(|l| {
                (
                    l.price.to_string().parse().unwrap_or(0.0),
                    l.quantity.to_string().parse().unwrap_or(0.0),
                )
            })
            .unwrap_or((0.0, 0.0));

        let spread = best_ask_price - best_bid_price;
        let mid_price = (best_ask_price + best_bid_price) / 2.0;

        let bid_depth: f64 = order_book
            .bids
            .iter()
            .map(|l| l.quantity.to_string().parse::<f64>().unwrap_or(0.0))
            .sum();

        let ask_depth: f64 = order_book
            .asks
            .iter()
            .map(|l| l.quantity.to_string().parse::<f64>().unwrap_or(0.0))
            .sum();

        let record = OrderBookRecord {
            timestamp: order_book.timestamp,
            exchange: format!("{:?}", order_book.exchange),
            symbol: order_book.symbol.to_string(),
            best_bid_price,
            best_bid_qty,
            best_ask_price,
            best_ask_qty,
            spread,
            mid_price,
            bid_depth,
            ask_depth,
            is_valid: validation.map(|v| v.is_valid),
            validation_errors: validation.map(|v| v.errors.join("; ")),
        };

        let partition_key =
            self.get_partition_key(order_book.timestamp, &record.exchange, &record.symbol);

        let mut buffers = self.order_book_buffers.lock().await;
        let buffer = buffers
            .entry(partition_key.clone())
            .or_insert_with(Vec::new);
        buffer.push(record);

        // Check if buffer needs flushing
        if buffer.len() >= self.config.buffer_size {
            let records = std::mem::take(buffer);
            drop(buffers);
            self.flush_order_book_buffer(&partition_key, records)
                .await?;
        }

        Ok(())
    }

    /// Write a market data event (dispatches to appropriate method)
    pub async fn write_event(
        &self,
        event: &MarketDataEvent,
        validation: Option<&ValidationResult>,
    ) -> Result<()> {
        match event {
            MarketDataEvent::Trade(trade) => self.write_trade(trade, validation).await,
            MarketDataEvent::OrderBook(ob) => self.write_order_book(ob, validation).await,
            _ => Ok(()), // Other event types not exported for now
        }
    }

    /// Flush trade buffer to Parquet file
    async fn flush_trade_buffer(
        &self,
        partition_key: &str,
        records: Vec<TradeRecord>,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        tracing::debug!(
            "Flushing {} trade records for partition {}",
            records.len(),
            partition_key
        );

        // Build arrays
        let mut timestamp_builder = TimestampMicrosecondBuilder::with_capacity(records.len());
        let mut received_at_builder = TimestampMicrosecondBuilder::with_capacity(records.len());
        let mut exchange_builder = StringBuilder::new();
        let mut symbol_builder = StringBuilder::new();
        let mut price_builder = Float64Builder::with_capacity(records.len());
        let mut quantity_builder = Float64Builder::with_capacity(records.len());
        let mut side_builder = StringBuilder::new();
        let mut trade_id_builder = StringBuilder::new();
        let mut buyer_is_maker_builder = BooleanBuilder::with_capacity(records.len());

        let mut is_valid_builder: Option<BooleanBuilder> = if self.config.include_validation {
            Some(BooleanBuilder::with_capacity(records.len()))
        } else {
            None
        };
        let mut validation_errors_builder: Option<StringBuilder> = if self.config.include_validation
        {
            Some(StringBuilder::new())
        } else {
            None
        };

        let first_record = records.first().unwrap();
        let timestamp =
            DateTime::from_timestamp_micros(first_record.timestamp).unwrap_or_else(Utc::now);

        for record in &records {
            timestamp_builder.append_value(record.timestamp);
            received_at_builder.append_value(record.received_at);
            exchange_builder.append_value(&record.exchange);
            symbol_builder.append_value(&record.symbol);
            price_builder.append_value(record.price);
            quantity_builder.append_value(record.quantity);
            side_builder.append_value(&record.side);
            trade_id_builder.append_value(&record.trade_id);

            match record.buyer_is_maker {
                Some(v) => buyer_is_maker_builder.append_value(v),
                None => buyer_is_maker_builder.append_null(),
            }

            if let Some(ref mut builder) = is_valid_builder {
                match record.is_valid {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }

            if let Some(ref mut builder) = validation_errors_builder {
                match &record.validation_errors {
                    Some(v) if !v.is_empty() => builder.append_value(v),
                    _ => builder.append_null(),
                }
            }
        }

        // Create arrays
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(timestamp_builder.finish()),
            Arc::new(received_at_builder.finish()),
            Arc::new(exchange_builder.finish()),
            Arc::new(symbol_builder.finish()),
            Arc::new(price_builder.finish()),
            Arc::new(quantity_builder.finish()),
            Arc::new(side_builder.finish()),
            Arc::new(trade_id_builder.finish()),
            Arc::new(buyer_is_maker_builder.finish()),
        ];

        if let Some(mut builder) = is_valid_builder {
            columns.push(Arc::new(builder.finish()));
        }

        if let Some(mut builder) = validation_errors_builder {
            columns.push(Arc::new(builder.finish()));
        }

        // Create record batch
        let batch = RecordBatch::try_new(self.trade_schema.clone(), columns).map_err(|e| {
            DataQualityError::IoError(format!("Failed to create record batch: {}", e))
        })?;

        // Write to file
        let output_path = self.partition_path(
            timestamp,
            &first_record.exchange,
            &first_record.symbol,
            "trades",
        );
        self.write_batch_to_file(&output_path, &batch, &self.trade_schema)
            .await
    }

    /// Flush order book buffer to Parquet file
    async fn flush_order_book_buffer(
        &self,
        partition_key: &str,
        records: Vec<OrderBookRecord>,
    ) -> Result<()> {
        if records.is_empty() {
            return Ok(());
        }

        tracing::debug!(
            "Flushing {} order book records for partition {}",
            records.len(),
            partition_key
        );

        // Build arrays
        let mut timestamp_builder = TimestampMicrosecondBuilder::with_capacity(records.len());
        let mut exchange_builder = StringBuilder::new();
        let mut symbol_builder = StringBuilder::new();
        let mut best_bid_price_builder = Float64Builder::with_capacity(records.len());
        let mut best_bid_qty_builder = Float64Builder::with_capacity(records.len());
        let mut best_ask_price_builder = Float64Builder::with_capacity(records.len());
        let mut best_ask_qty_builder = Float64Builder::with_capacity(records.len());
        let mut spread_builder = Float64Builder::with_capacity(records.len());
        let mut mid_price_builder = Float64Builder::with_capacity(records.len());
        let mut bid_depth_builder = Float64Builder::with_capacity(records.len());
        let mut ask_depth_builder = Float64Builder::with_capacity(records.len());

        let mut is_valid_builder: Option<BooleanBuilder> = if self.config.include_validation {
            Some(BooleanBuilder::with_capacity(records.len()))
        } else {
            None
        };
        let mut validation_errors_builder: Option<StringBuilder> = if self.config.include_validation
        {
            Some(StringBuilder::new())
        } else {
            None
        };

        let first_record = records.first().unwrap();
        let timestamp =
            DateTime::from_timestamp_micros(first_record.timestamp).unwrap_or_else(Utc::now);

        for record in &records {
            timestamp_builder.append_value(record.timestamp);
            exchange_builder.append_value(&record.exchange);
            symbol_builder.append_value(&record.symbol);
            best_bid_price_builder.append_value(record.best_bid_price);
            best_bid_qty_builder.append_value(record.best_bid_qty);
            best_ask_price_builder.append_value(record.best_ask_price);
            best_ask_qty_builder.append_value(record.best_ask_qty);
            spread_builder.append_value(record.spread);
            mid_price_builder.append_value(record.mid_price);
            bid_depth_builder.append_value(record.bid_depth);
            ask_depth_builder.append_value(record.ask_depth);

            if let Some(ref mut builder) = is_valid_builder {
                match record.is_valid {
                    Some(v) => builder.append_value(v),
                    None => builder.append_null(),
                }
            }

            if let Some(ref mut builder) = validation_errors_builder {
                match &record.validation_errors {
                    Some(v) if !v.is_empty() => builder.append_value(v),
                    _ => builder.append_null(),
                }
            }
        }

        // Create arrays
        let mut columns: Vec<ArrayRef> = vec![
            Arc::new(timestamp_builder.finish()),
            Arc::new(exchange_builder.finish()),
            Arc::new(symbol_builder.finish()),
            Arc::new(best_bid_price_builder.finish()),
            Arc::new(best_bid_qty_builder.finish()),
            Arc::new(best_ask_price_builder.finish()),
            Arc::new(best_ask_qty_builder.finish()),
            Arc::new(spread_builder.finish()),
            Arc::new(mid_price_builder.finish()),
            Arc::new(bid_depth_builder.finish()),
            Arc::new(ask_depth_builder.finish()),
        ];

        if let Some(mut builder) = is_valid_builder {
            columns.push(Arc::new(builder.finish()));
        }

        if let Some(mut builder) = validation_errors_builder {
            columns.push(Arc::new(builder.finish()));
        }

        // Create record batch
        let batch = RecordBatch::try_new(self.order_book_schema.clone(), columns).map_err(|e| {
            DataQualityError::IoError(format!("Failed to create record batch: {}", e))
        })?;

        // Write to file
        let output_path = self.partition_path(
            timestamp,
            &first_record.exchange,
            &first_record.symbol,
            "order_books",
        );
        self.write_batch_to_file(&output_path, &batch, &self.order_book_schema)
            .await
    }

    /// Write a record batch to a Parquet file
    async fn write_batch_to_file(
        &self,
        output_dir: &Path,
        batch: &RecordBatch,
        schema: &Arc<Schema>,
    ) -> Result<()> {
        // Create directory if needed
        fs::create_dir_all(output_dir).map_err(|e| {
            DataQualityError::IoError(format!(
                "Failed to create directory {:?}: {}",
                output_dir, e
            ))
        })?;

        // Generate filename
        let filename = self.generate_filename(if Arc::ptr_eq(schema, &self.trade_schema) {
            "trades"
        } else {
            "order_books"
        });
        let file_path = output_dir.join(&filename);

        tracing::info!("Writing Parquet file: {:?}", file_path);

        // Create file
        let file = File::create(&file_path).map_err(|e| {
            DataQualityError::IoError(format!("Failed to create file {:?}: {}", file_path, e))
        })?;

        // Configure writer properties
        let compression: Compression = self.config.compression.into();
        let props = WriterProperties::builder()
            .set_compression(compression)
            .set_created_by("janus-data-quality".to_string())
            .build();

        // Create Arrow writer
        let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props)).map_err(|e| {
            DataQualityError::IoError(format!("Failed to create Parquet writer: {}", e))
        })?;

        // Write batch
        writer
            .write(batch)
            .map_err(|e| DataQualityError::IoError(format!("Failed to write batch: {}", e)))?;

        // Close writer
        let _metadata = writer.close().map_err(|e| {
            DataQualityError::IoError(format!("Failed to close Parquet writer: {}", e))
        })?;

        tracing::info!(
            "Successfully wrote {} rows to {:?}",
            batch.num_rows(),
            file_path
        );

        Ok(())
    }

    /// Flush all buffers to disk
    pub async fn flush(&self) -> Result<()> {
        // Flush trade buffers
        let mut trade_buffers = self.trade_buffers.lock().await;
        let trade_partitions: Vec<_> = trade_buffers.keys().cloned().collect();
        for partition_key in trade_partitions {
            if let Some(records) = trade_buffers.remove(&partition_key)
                && !records.is_empty()
            {
                drop(trade_buffers);
                self.flush_trade_buffer(&partition_key, records).await?;
                trade_buffers = self.trade_buffers.lock().await;
            }
        }
        drop(trade_buffers);

        // Flush order book buffers
        let mut ob_buffers = self.order_book_buffers.lock().await;
        let ob_partitions: Vec<_> = ob_buffers.keys().cloned().collect();
        for partition_key in ob_partitions {
            if let Some(records) = ob_buffers.remove(&partition_key)
                && !records.is_empty()
            {
                drop(ob_buffers);
                self.flush_order_book_buffer(&partition_key, records)
                    .await?;
                ob_buffers = self.order_book_buffers.lock().await;
            }
        }

        Ok(())
    }

    /// Get statistics about current buffer state
    pub async fn buffer_stats(&self) -> BufferStats {
        let trade_buffers = self.trade_buffers.lock().await;
        let ob_buffers = self.order_book_buffers.lock().await;

        let trade_records: usize = trade_buffers.values().map(|v| v.len()).sum();
        let ob_records: usize = ob_buffers.values().map(|v| v.len()).sum();

        BufferStats {
            trade_partitions: trade_buffers.len(),
            trade_records,
            order_book_partitions: ob_buffers.len(),
            order_book_records: ob_records,
        }
    }
}

/// Statistics about the exporter's buffer state
#[derive(Debug, Clone)]
pub struct BufferStats {
    /// Number of trade partitions
    pub trade_partitions: usize,
    /// Total trade records in buffer
    pub trade_records: usize,
    /// Number of order book partitions
    pub order_book_partitions: usize,
    /// Total order book records in buffer
    pub order_book_records: usize,
}

impl BufferStats {
    /// Total records across all buffers
    pub fn total_records(&self) -> usize {
        self.trade_records + self.order_book_records
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_core::{Exchange, Side, Symbol};
    use rust_decimal::Decimal;

    fn create_test_trade() -> TradeEvent {
        TradeEvent {
            exchange: Exchange::Coinbase,
            symbol: Symbol::new("BTC", "USD"),
            timestamp: chrono::Utc::now().timestamp_micros(),
            received_at: chrono::Utc::now().timestamp_micros(),
            price: Decimal::new(50000, 0),
            quantity: Decimal::new(1, 1),
            side: Side::Buy,
            trade_id: "test-123".to_string(),
            buyer_is_maker: Some(false),
        }
    }

    #[test]
    fn test_export_config_builder() {
        let config = ExportConfig::builder()
            .output_dir("/tmp/test")
            .partition_by_date(true)
            .partition_by_symbol(true)
            .compression(CompressionType::Zstd)
            .buffer_size(5000)
            .build();

        assert_eq!(config.output_dir, PathBuf::from("/tmp/test"));
        assert!(config.partition_by_date);
        assert!(config.partition_by_symbol);
        assert_eq!(config.buffer_size, 5000);
    }

    #[test]
    fn test_exporter_creation() {
        let temp_dir = std::env::temp_dir().join(format!("test_exporter_{}", uuid::Uuid::new_v4()));
        let config = ExportConfig::builder().output_dir(&temp_dir).build();

        let exporter = ParquetExporter::new(config);
        assert!(exporter.is_ok());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_partition_key_generation() {
        let temp_dir =
            std::env::temp_dir().join(format!("test_partition_{}", uuid::Uuid::new_v4()));
        let config = ExportConfig::builder()
            .output_dir(&temp_dir)
            .partition_by_date(true)
            .partition_by_exchange(true)
            .partition_by_symbol(false)
            .build();

        let exporter = ParquetExporter::new(config).unwrap();
        let key = exporter.get_partition_key(
            chrono::Utc::now().timestamp_micros(),
            "Coinbase",
            "BTC/USD",
        );

        assert!(key.contains("coinbase"));
        assert!(key.contains("-")); // Date separator

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_write_trade_to_buffer() {
        let temp_dir = std::env::temp_dir().join(format!("test_buffer_{}", uuid::Uuid::new_v4()));
        let config = ExportConfig::builder()
            .output_dir(&temp_dir)
            .buffer_size(100) // Large buffer to avoid auto-flush
            .build();

        let exporter = ParquetExporter::new(config).unwrap();
        let trade = create_test_trade();

        exporter.write_trade(&trade, None).await.unwrap();

        let stats = exporter.buffer_stats().await;
        assert_eq!(stats.trade_records, 1);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_flush_creates_file() {
        let temp_dir = std::env::temp_dir().join(format!("test_flush_{}", uuid::Uuid::new_v4()));
        let config = ExportConfig::builder()
            .output_dir(&temp_dir)
            .partition_by_date(false)
            .partition_by_symbol(false)
            .partition_by_exchange(false)
            .buffer_size(100)
            .build();

        let exporter = ParquetExporter::new(config).unwrap();

        // Write some trades
        for _ in 0..5 {
            let trade = create_test_trade();
            exporter.write_trade(&trade, None).await.unwrap();
        }

        // Flush
        exporter.flush().await.unwrap();

        // Check that file was created
        let stats = exporter.buffer_stats().await;
        assert_eq!(stats.trade_records, 0); // Buffer should be empty after flush

        // Check that parquet file exists
        let trades_dir = temp_dir.join("trades");
        assert!(trades_dir.exists());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[tokio::test]
    async fn test_write_with_validation() {
        let temp_dir =
            std::env::temp_dir().join(format!("test_validation_{}", uuid::Uuid::new_v4()));
        let config = ExportConfig::builder()
            .output_dir(&temp_dir)
            .include_validation(true)
            .buffer_size(100)
            .build();

        let exporter = ParquetExporter::new(config).unwrap();
        let trade = create_test_trade();

        let validation = ValidationResult::success("test_validator");
        exporter
            .write_trade(&trade, Some(&validation))
            .await
            .unwrap();

        let stats = exporter.buffer_stats().await;
        assert_eq!(stats.trade_records, 1);

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
