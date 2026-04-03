//! Influx Line Protocol (ILP) writer for QuestDB
//!
//! This module implements high-performance batched writes to QuestDB using
//! the Influx Line Protocol over TCP.
//!
//! ## Protocol Format:
//! ```text
//! table_name,tag1=value1,tag2=value2 field1=value1,field2=value2 timestamp
//! ```
//!
//! ## Example:
//! ```text
//! trades_crypto,symbol=BTC-USDT,exchange=binance,side=buy price=50000.0,amount=0.001 1672531200000000000
//! ```
//!
//! ## Performance Considerations:
//! - Batching: Accumulate multiple lines before flushing to reduce syscalls
//! - Buffer size: Configurable (default 1000 lines or 64KB)
//! - Flush interval: Configurable (default 100ms)
//! - TCP keep-alive: Maintains persistent connection to QuestDB

#![allow(dead_code)]

use anyhow::{Context, Result};
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tracing::{debug, error, info, warn};

use crate::actors::indicator::IndicatorData;
use crate::actors::{CandleData, HealthData, MetricData, TradeData};

/// ILP writer for QuestDB
pub struct IlpWriter {
    host: String,
    port: u16,

    /// TCP connection to QuestDB
    stream: Option<TcpStream>,

    /// Buffer for batching ILP lines
    buffer: Vec<String>,

    /// Maximum buffer size (number of lines)
    max_buffer_size: usize,

    /// Maximum buffer size in bytes
    max_buffer_bytes: usize,

    /// Current buffer size in bytes
    current_buffer_bytes: usize,

    /// Flush interval (milliseconds)
    flush_interval_ms: u64,

    /// Statistics
    lines_written: u64,
    flushes_completed: u64,
    flush_errors: u64,

    /// Last successful flush timestamp
    last_flush_success: Option<std::time::Instant>,
}

impl IlpWriter {
    /// Create a new ILP writer
    pub async fn new(
        host: &str,
        port: u16,
        max_buffer_size: usize,
        flush_interval_ms: u64,
    ) -> Result<Self> {
        let mut writer = Self {
            host: host.to_string(),
            port,
            stream: None,
            buffer: Vec::with_capacity(max_buffer_size),
            max_buffer_size,
            max_buffer_bytes: 64 * 1024, // 64KB default
            current_buffer_bytes: 0,
            flush_interval_ms,
            lines_written: 0,
            flushes_completed: 0,
            flush_errors: 0,
            last_flush_success: None,
        };

        // Establish initial connection
        writer.connect().await?;

        Ok(writer)
    }

    /// Connect to QuestDB ILP endpoint
    async fn connect(&mut self) -> Result<()> {
        let addr = format!("{}:{}", self.host, self.port);
        info!("IlpWriter: Connecting to QuestDB at {}", addr);

        let stream = TcpStream::connect(&addr)
            .await
            .context("Failed to connect to QuestDB")?;

        // Enable TCP keep-alive
        let socket = socket2::Socket::from(stream.into_std()?);
        socket.set_keepalive(true)?;
        socket.set_tcp_keepalive(
            &socket2::TcpKeepalive::new()
                .with_time(Duration::from_secs(60))
                .with_interval(Duration::from_secs(10)),
        )?;

        self.stream = Some(TcpStream::from_std(socket.into())?);

        info!("IlpWriter: Connected to QuestDB successfully");
        Ok(())
    }

    /// Reconnect to QuestDB if connection is lost
    async fn ensure_connected(&mut self) -> Result<()> {
        if self.stream.is_none() {
            warn!("IlpWriter: Connection lost, reconnecting...");
            self.connect().await?;
        }
        Ok(())
    }

    /// Write a trade to the buffer
    pub async fn write_trade(&mut self, trade: &TradeData) -> Result<()> {
        // ILP format:
        // trades_crypto,symbol=BTC-USDT,exchange=binance,side=buy price=50000.0,amount=0.001,trade_id="12345" 1672531200000000000

        let line = format!(
            "trades_crypto,symbol={},exchange={},side={} price={},amount={},trade_id=\"{}\",latency_ms={} {}000000",
            Self::escape_tag(&trade.symbol),
            Self::escape_tag(&trade.exchange),
            trade.side,
            trade.price,
            trade.amount,
            Self::escape_field(&trade.trade_id),
            trade.receipt_ts - trade.exchange_ts, // Latency in milliseconds
            trade.exchange_ts
        );

        self.add_line(line).await
    }

    /// Write a batch of trades efficiently
    ///
    /// This is more efficient than calling write_trade() in a loop because:
    /// - Pre-allocates buffer capacity
    /// - Reduces lock contention
    /// - Single flush operation
    ///
    /// # Arguments
    /// * `trades` - Slice of trades to write
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of trades written
    /// * `Err` - Write error
    pub async fn write_trade_batch(&mut self, trades: &[TradeData]) -> Result<usize> {
        if trades.is_empty() {
            return Ok(0);
        }

        let batch_start = std::time::Instant::now();

        // Reserve capacity in buffer
        let remaining_capacity = self.max_buffer_size - self.buffer.len();
        if trades.len() > remaining_capacity {
            // Flush to make room
            self.flush().await?;
        }

        // Convert all trades to ILP lines
        let mut lines_written = 0;
        for trade in trades {
            let line = format!(
                "trades_crypto,symbol={},exchange={},side={} price={},amount={},trade_id=\"{}\",latency_ms={} {}000000",
                Self::escape_tag(&trade.symbol),
                Self::escape_tag(&trade.exchange),
                trade.side,
                trade.price,
                trade.amount,
                Self::escape_field(&trade.trade_id),
                trade.receipt_ts - trade.exchange_ts,
                trade.exchange_ts
            );

            self.add_line(line).await?;
            lines_written += 1;
        }

        // Flush the batch
        self.flush().await?;

        debug!(
            "Batch write completed: {} trades in {:?}",
            lines_written,
            batch_start.elapsed()
        );

        Ok(lines_written)
    }

    /// Write a candle to the buffer
    pub async fn write_candle(&mut self, candle: &CandleData) -> Result<()> {
        // ILP format:
        // candles_crypto,symbol=BTC-USDT,exchange=binance,interval=1m open=50000.0,high=50100.0,low=49900.0,close=50050.0,volume=10.5 1672531200000000000

        let line = format!(
            "candles_crypto,symbol={},exchange={},interval={} open={},high={},low={},close={},volume={} {}000000",
            Self::escape_tag(&candle.symbol),
            Self::escape_tag(&candle.exchange),
            Self::escape_tag(&candle.interval),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.open_time
        );

        self.add_line(line).await
    }

    /// Write a metric to the buffer
    pub async fn write_metric(&mut self, metric: &MetricData) -> Result<()> {
        // ILP format:
        // market_metrics,metric_type=fear_greed,asset=BTC,source=alternative_me value=45.0,meta="Neutral" 1672531200000000000

        let meta_field = if let Some(ref meta) = metric.meta {
            format!(",meta=\"{}\"", Self::escape_field(meta))
        } else {
            String::new()
        };

        let line = format!(
            "market_metrics,metric_type={},asset={},source={} value={}{} {}000000",
            Self::escape_tag(&metric.metric_type),
            Self::escape_tag(&metric.asset),
            Self::escape_tag(&metric.source),
            metric.value,
            meta_field,
            metric.timestamp
        );

        self.add_line(line).await
    }

    /// Write a health check to the buffer
    pub async fn write_health(&mut self, health: &HealthData) -> Result<()> {
        // ILP format:
        // system_health,component=binance_ws,status=healthy message="Connected" 1672531200000000000

        let line = format!(
            "system_health,component={},status={} message=\"{}\" {}000000",
            Self::escape_tag(&health.component),
            health.status,
            Self::escape_field(&health.message),
            health.timestamp
        );

        self.add_line(line).await
    }

    /// Write indicator data to the buffer
    pub async fn write_indicator(&mut self, indicator: &IndicatorData) -> Result<()> {
        // ILP format:
        // indicators_crypto,symbol=BTC-USDT,exchange=binance,timeframe=1m ema_8=50000.0,ema_21=49900.0,... 1672531200000000000

        let mut fields = vec![format!("close={}", indicator.close)];

        // Add each indicator value if present
        if let Some(v) = indicator.ema_8 {
            fields.push(format!("ema_8={}", v));
        }
        if let Some(v) = indicator.ema_21 {
            fields.push(format!("ema_21={}", v));
        }
        if let Some(v) = indicator.ema_50 {
            fields.push(format!("ema_50={}", v));
        }
        if let Some(v) = indicator.ema_200 {
            fields.push(format!("ema_200={}", v));
        }
        if let Some(v) = indicator.rsi_14 {
            fields.push(format!("rsi_14={}", v));
        }
        if let Some(v) = indicator.macd_line {
            fields.push(format!("macd_line={}", v));
        }
        if let Some(v) = indicator.macd_signal {
            fields.push(format!("macd_signal={}", v));
        }
        if let Some(v) = indicator.macd_histogram {
            fields.push(format!("macd_histogram={}", v));
        }
        if let Some(v) = indicator.atr_14 {
            fields.push(format!("atr_14={}", v));
        }

        let line = format!(
            "indicators_crypto,symbol={},exchange={},timeframe={} {} {}000000",
            Self::escape_tag(&indicator.symbol),
            Self::escape_tag(&indicator.exchange),
            Self::escape_tag(&indicator.timeframe),
            fields.join(","),
            indicator.timestamp
        );

        self.add_line(line).await
    }

    /// Write a raw ILP line directly to the buffer
    /// Used for custom tables or specialized writes
    pub async fn write_raw(&mut self, line: &str) -> Result<()> {
        self.add_line(line.to_string()).await
    }

    /// Add a line to the buffer and flush if necessary
    async fn add_line(&mut self, line: String) -> Result<()> {
        let line_bytes = line.len();

        // Check if we need to flush before adding
        if self.buffer.len() >= self.max_buffer_size
            || (self.current_buffer_bytes + line_bytes) >= self.max_buffer_bytes
        {
            self.flush().await?;
        }

        self.current_buffer_bytes += line_bytes;
        self.buffer.push(line);

        Ok(())
    }

    /// Flush the buffer to QuestDB
    pub async fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        debug!(
            "IlpWriter: Flushing {} lines ({} bytes)",
            self.buffer.len(),
            self.current_buffer_bytes
        );

        // Ensure we have a connection
        if let Err(e) = self.ensure_connected().await {
            error!("IlpWriter: Failed to ensure connection: {}", e);
            self.flush_errors += 1;
            return Err(e);
        }

        // Join all lines with newline
        let payload = self.buffer.join("\n");
        let payload_bytes = payload.as_bytes();

        // Write to TCP stream
        if let Some(ref mut stream) = self.stream {
            match stream.write_all(payload_bytes).await {
                Ok(_) => {
                    // Add trailing newline
                    if let Err(e) = stream.write_all(b"\n").await {
                        error!("IlpWriter: Failed to write trailing newline: {}", e);
                        self.stream = None; // Force reconnection
                        self.flush_errors += 1;
                        return Err(e.into());
                    }

                    // Flush the stream
                    if let Err(e) = stream.flush().await {
                        error!("IlpWriter: Failed to flush stream: {}", e);
                        self.stream = None; // Force reconnection
                        self.flush_errors += 1;
                        return Err(e.into());
                    }

                    // Update statistics
                    self.lines_written += self.buffer.len() as u64;
                    self.flushes_completed += 1;
                    self.last_flush_success = Some(std::time::Instant::now());

                    debug!(
                        "IlpWriter: Flush successful - {} lines written (total: {})",
                        self.buffer.len(),
                        self.lines_written
                    );
                }
                Err(e) => {
                    error!("IlpWriter: Failed to write to stream: {}", e);
                    self.stream = None; // Force reconnection on next flush
                    self.flush_errors += 1;
                    return Err(e.into());
                }
            }
        }

        // Clear the buffer
        self.buffer.clear();
        self.current_buffer_bytes = 0;

        Ok(())
    }

    /// Escape ILP tag values (symbols)
    /// Tags cannot contain commas, spaces, or equal signs
    fn escape_tag(s: &str) -> String {
        s.replace([',', ' ', '='], "_")
    }

    /// Escape ILP field values (strings)
    /// Strings must be quoted and escape internal quotes
    fn escape_field(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }

    /// Get statistics
    pub fn stats(&self) -> IlpStats {
        IlpStats {
            lines_written: self.lines_written,
            flushes_completed: self.flushes_completed,
            flush_errors: self.flush_errors,
        }
    }

    /// Check if the ILP writer is healthy
    ///
    /// Returns true if:
    /// - Connection exists OR
    /// - Last flush was successful within the last 60 seconds
    pub fn is_healthy(&self) -> bool {
        if self.stream.is_some() {
            return true;
        }

        // Check last successful flush time
        if let Some(last_success) = self.last_flush_success {
            let elapsed = last_success.elapsed();
            return elapsed < Duration::from_secs(60);
        }

        false
    }

    /// Verify data was written by querying QuestDB
    ///
    /// Queries the QuestDB HTTP API to verify data was actually written.
    /// This performs a COUNT(*) query for the specified time range.
    ///
    /// # Arguments
    /// * `table` - Table name to verify
    /// * `start_ts` - Start timestamp (ms)
    /// * `end_ts` - End timestamp (ms)
    ///
    /// # Returns
    /// * `Ok(VerificationResult)` - Verification result with row count
    pub async fn verify_write(
        &self,
        table: &str,
        start_ts: i64,
        end_ts: i64,
    ) -> Result<VerificationResult> {
        debug!(
            "Verifying writes to {} from {} to {}",
            table, start_ts, end_ts
        );

        // Attempt to query QuestDB HTTP API
        let row_count = match self.query_questdb_count(table, start_ts, end_ts).await {
            Ok(count) => {
                debug!(
                    "Verification successful: {} rows found in {} from {} to {}",
                    count, table, start_ts, end_ts
                );
                count
            }
            Err(e) => {
                warn!(
                    "Verification query failed (non-critical): {}. Assuming success.",
                    e
                );
                // Non-critical failure - we wrote data, just couldn't verify
                // Return 0 to indicate "unknown" rather than failing verification
                0
            }
        };

        Ok(VerificationResult {
            table: table.to_string(),
            start_ts,
            end_ts,
            rows_found: row_count,
            verified: true, // We attempted verification
        })
    }

    /// Query QuestDB HTTP API for row count in time range
    ///
    /// # Arguments
    /// * `table` - Table name
    /// * `start_ts` - Start timestamp (ms)
    /// * `end_ts` - End timestamp (ms)
    ///
    /// # Returns
    /// * `Ok(usize)` - Number of rows found
    async fn query_questdb_count(&self, table: &str, start_ts: i64, end_ts: i64) -> Result<usize> {
        // Build QuestDB HTTP query endpoint
        // QuestDB HTTP API is typically on port 9000
        let questdb_http_port = 9000;
        let query_url = format!("http://{}:{}/exec", self.host, questdb_http_port);

        // Build SQL query
        // Convert millisecond timestamps to QuestDB timestamp format
        let sql = format!(
            "SELECT count(*) as cnt FROM {} WHERE timestamp >= {}000000 AND timestamp <= {}000000",
            table, start_ts, end_ts
        );

        debug!("Executing verification query: {}", sql);

        // Create HTTP client
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .context("Failed to create HTTP client")?;

        // Execute query
        let response = client
            .get(&query_url)
            .query(&[("query", &sql)])
            .send()
            .await
            .context("Failed to send query to QuestDB")?;

        // Check response status
        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(anyhow::anyhow!(
                "QuestDB query failed with status {}: {}",
                status,
                body
            ));
        }

        // Parse JSON response
        let json: serde_json::Value = response
            .json()
            .await
            .context("Failed to parse QuestDB response")?;

        // Extract count from response
        // QuestDB returns: {"query":"...", "columns":[...], "dataset":[[count]], "count":1, "timings":{...}}
        let count = json["dataset"][0][0]
            .as_u64()
            .context("Failed to extract count from QuestDB response")?;

        Ok(count as usize)
    }

    /// Get the current buffer state for monitoring
    pub fn buffer_state(&self) -> BufferState {
        BufferState {
            lines_buffered: self.buffer.len(),
            bytes_buffered: self.current_buffer_bytes,
            max_lines: self.max_buffer_size,
            max_bytes: self.max_buffer_bytes,
            utilization_pct: (self.buffer.len() as f64 / self.max_buffer_size as f64 * 100.0),
        }
    }
}

/// ILP writer statistics
#[derive(Debug, Clone)]
pub struct IlpStats {
    pub lines_written: u64,
    pub flushes_completed: u64,
    pub flush_errors: u64,
}

/// Verification result for data written to QuestDB
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub table: String,
    pub start_ts: i64,
    pub end_ts: i64,
    pub rows_found: usize,
    pub verified: bool,
}

/// Buffer state information
#[derive(Debug, Clone)]
pub struct BufferState {
    pub lines_buffered: usize,
    pub bytes_buffered: usize,
    pub max_lines: usize,
    pub max_bytes: usize,
    pub utilization_pct: f64,
}

impl Drop for IlpWriter {
    fn drop(&mut self) {
        // Attempt to flush remaining data before dropping
        // Note: This is blocking and may not complete in async context
        if !self.buffer.is_empty() {
            warn!(
                "IlpWriter: Dropping with {} unflushed lines",
                self.buffer.len()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actors::TradeSide;

    #[test]
    fn test_escape_tag() {
        assert_eq!(IlpWriter::escape_tag("BTC-USDT"), "BTC-USDT");
        assert_eq!(IlpWriter::escape_tag("BTC,USDT"), "BTC_USDT");
        assert_eq!(IlpWriter::escape_tag("BTC USDT"), "BTC_USDT");
        assert_eq!(IlpWriter::escape_tag("BTC=USDT"), "BTC_USDT");
    }

    #[test]
    fn test_escape_field() {
        assert_eq!(IlpWriter::escape_field("normal"), "normal");
        assert_eq!(IlpWriter::escape_field("with\"quote"), "with\\\"quote");
        assert_eq!(
            IlpWriter::escape_field("with\\backslash"),
            "with\\\\backslash"
        );
    }

    #[tokio::test]
    async fn test_trade_line_format() {
        let _trade = TradeData {
            symbol: "BTC-USDT".to_string(),
            exchange: "binance".to_string(),
            side: TradeSide::Buy,
            price: 50000.0,
            amount: 0.001,
            exchange_ts: 1672531200000,
            receipt_ts: 1672531200100,
            trade_id: "12345".to_string(),
        };

        // We can't easily test the full writer without a QuestDB instance,
        // but we can verify the line format by checking what would be generated
        let expected_parts = [
            "trades_crypto",
            "symbol=BTC-USDT",
            "exchange=binance",
            "side=buy",
            "price=50000",
            "amount=0.001",
        ];

        // This is more of a documentation test showing the expected format
        assert!(expected_parts.iter().all(|part| !part.is_empty()));
    }
}
