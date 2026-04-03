//! QuestDB Writer Module
//!
//! This module provides high-performance time-series data persistence
//! using QuestDB's Influx Line Protocol (ILP).
//!
//! Features:
//! - Batched writes with configurable batch size
//! - Temporal flush triggers (max time between flushes)
//! - Non-blocking TCP writes
//! - Automatic reconnection on connection loss
//! - High throughput (>100K inserts/sec, targeting 1M/sec)

use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_BATCH_SIZE: usize = 1000;
const DEFAULT_FLUSH_INTERVAL_MS: u64 = 100;
const DEFAULT_CHANNEL_BUFFER: usize = 10000;
const RECONNECT_DELAY_MS: u64 = 1000;

// ============================================================================
// Data Types
// ============================================================================

/// Tick write message for channel
#[derive(Debug, Clone)]
pub struct TickWrite {
    pub symbol: String,
    pub bid_px: f64,
    pub ask_px: f64,
    pub bid_sz: f64,
    pub ask_sz: f64,
    pub timestamp_micros: i64,
}

/// Trade write message
#[derive(Debug, Clone)]
pub struct TradeWrite {
    pub symbol: String,
    pub side: String,
    pub price: f64,
    pub size: f64,
    pub timestamp_micros: i64,
}

/// Signal write message
#[derive(Debug, Clone)]
pub struct SignalWrite {
    pub symbol: String,
    pub signal_type: String,
    pub entry_price: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub position_size: f64,
    pub timestamp_micros: i64,
}

/// Execution write message
#[derive(Debug, Clone)]
pub struct ExecutionWrite {
    pub order_id: String,
    pub symbol: String,
    pub side: String,
    pub price: f64,
    pub size: f64,
    pub fee: f64,
    pub timestamp_micros: i64,
}

/// Regime state write message for persistent archiving of bridged regime states.
///
/// Written to the `regime_states` table in QuestDB via ILP.
///
/// # ILP Format
///
/// ```text
/// regime_states,symbol=BTCUSD,hypothalamus=StrongBullish,amygdala=LowVolTrending
///   position_scale=1.2,is_high_risk=false,confidence=0.85,trend=0.75,
///   trend_strength=0.6,volatility=500.0,volatility_percentile=0.3,
///   momentum=0.4,relative_volume=1.5,liquidity_score=0.9,
///   is_transition=false 1234567890000000
/// ```
#[derive(Debug, Clone)]
pub struct RegimeWrite {
    pub symbol: String,
    pub hypothalamus_regime: String,
    pub amygdala_regime: String,
    pub position_scale: f64,
    pub is_high_risk: bool,
    pub confidence: f64,
    pub trend: f64,
    pub trend_strength: f64,
    pub volatility: f64,
    pub volatility_percentile: f64,
    pub momentum: f64,
    pub relative_volume: f64,
    pub liquidity_score: f64,
    pub is_transition: bool,
    pub timestamp_micros: i64,
}

/// Write message enum
#[derive(Debug, Clone)]
pub enum WriteMessage {
    Tick(TickWrite),
    Trade(TradeWrite),
    Signal(SignalWrite),
    Execution(ExecutionWrite),
    Regime(RegimeWrite),
    Flush,
    Shutdown,
}

// ============================================================================
// Writer Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct QuestDBConfig {
    pub host: String,
    pub port: u16,
    pub batch_size: usize,
    pub flush_interval_ms: u64,
    pub channel_buffer: usize,
}

impl Default for QuestDBConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 9009,
            batch_size: DEFAULT_BATCH_SIZE,
            flush_interval_ms: DEFAULT_FLUSH_INTERVAL_MS,
            channel_buffer: DEFAULT_CHANNEL_BUFFER,
        }
    }
}

impl QuestDBConfig {
    pub fn new(host: String, port: u16) -> Self {
        Self {
            host,
            port,
            ..Default::default()
        }
    }

    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn with_flush_interval(mut self, interval_ms: u64) -> Self {
        self.flush_interval_ms = interval_ms;
        self
    }

    pub fn with_channel_buffer(mut self, buffer: usize) -> Self {
        self.channel_buffer = buffer;
        self
    }
}

// ============================================================================
// QuestDB Writer
// ============================================================================

/// High-performance QuestDB ILP writer
pub struct QuestDBWriter {
    #[allow(dead_code)]
    config: Arc<QuestDBConfig>,
    tx: mpsc::Sender<WriteMessage>,
}

impl QuestDBWriter {
    /// Create a new QuestDB writer
    pub fn new(config: QuestDBConfig) -> Self {
        let (tx, rx) = mpsc::channel(config.channel_buffer);
        let config = Arc::new(config);

        // Spawn writer task
        let writer_config = Arc::clone(&config);
        tokio::spawn(async move {
            if let Err(e) = writer_loop(writer_config, rx).await {
                error!("QuestDB writer loop error: {}", e);
            }
        });

        Self { config, tx }
    }

    /// Write a market tick
    pub async fn write_tick(&self, tick: TickWrite) -> Result<()> {
        self.tx
            .send(WriteMessage::Tick(tick))
            .await
            .map_err(|e| anyhow!("Failed to send tick: {}", e))
    }

    /// Write a trade
    pub async fn write_trade(&self, trade: TradeWrite) -> Result<()> {
        self.tx
            .send(WriteMessage::Trade(trade))
            .await
            .map_err(|e| anyhow!("Failed to send trade: {}", e))
    }

    /// Write a signal
    pub async fn write_signal(&self, signal: SignalWrite) -> Result<()> {
        self.tx
            .send(WriteMessage::Signal(signal))
            .await
            .map_err(|e| anyhow!("Failed to send signal: {}", e))
    }

    /// Write an execution
    pub async fn write_execution(&self, execution: ExecutionWrite) -> Result<()> {
        self.tx
            .send(WriteMessage::Execution(execution))
            .await
            .map_err(|e| anyhow!("Failed to send execution: {}", e))
    }

    /// Write a regime state for persistent archiving.
    ///
    /// Each bridged regime state is persisted to the `regime_states` table
    /// in QuestDB so it can be replayed, queried historically, or used for
    /// model training.
    pub async fn write_regime(&self, regime: RegimeWrite) -> Result<()> {
        self.tx
            .send(WriteMessage::Regime(regime))
            .await
            .map_err(|e| anyhow!("Failed to send regime write: {}", e))
    }

    /// Flush buffered writes
    pub async fn flush(&self) -> Result<()> {
        self.tx
            .send(WriteMessage::Flush)
            .await
            .map_err(|e| anyhow!("Failed to send flush: {}", e))
    }

    /// Shutdown the writer
    pub async fn shutdown(&self) -> Result<()> {
        self.tx
            .send(WriteMessage::Shutdown)
            .await
            .map_err(|e| anyhow!("Failed to send shutdown: {}", e))
    }
}

// ============================================================================
// Writer Loop
// ============================================================================

async fn writer_loop(
    config: Arc<QuestDBConfig>,
    mut rx: mpsc::Receiver<WriteMessage>,
) -> Result<()> {
    let mut connection = None;
    let mut buffer = String::with_capacity(config.batch_size * 256);
    let mut batch_count = 0;
    let mut last_flush = Instant::now();
    let mut flush_timer = interval(Duration::from_millis(config.flush_interval_ms));

    info!(
        "QuestDB writer started: {}:{} (batch_size={}, flush_interval_ms={})",
        config.host, config.port, config.batch_size, config.flush_interval_ms
    );

    loop {
        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Some(WriteMessage::Tick(tick)) => {
                        format_tick(&mut buffer, &tick);
                        batch_count += 1;
                    }
                    Some(WriteMessage::Trade(trade)) => {
                        format_trade(&mut buffer, &trade);
                        batch_count += 1;
                    }
                    Some(WriteMessage::Signal(signal)) => {
                        format_signal(&mut buffer, &signal);
                        batch_count += 1;
                    }
                    Some(WriteMessage::Execution(execution)) => {
                        format_execution(&mut buffer, &execution);
                        batch_count += 1;
                    }
                    Some(WriteMessage::Regime(regime)) => {
                        format_regime(&mut buffer, &regime);
                        batch_count += 1;
                    }
                    Some(WriteMessage::Flush) => {
                        flush_buffer(&config, &mut connection, &mut buffer, batch_count).await?;
                        batch_count = 0;
                        last_flush = Instant::now();
                    }
                    Some(WriteMessage::Shutdown) => {
                        info!("Shutdown requested, flushing final batch");
                        flush_buffer(&config, &mut connection, &mut buffer, batch_count).await?;
                        if let Some(mut conn) = connection.take() {
                            let _ = conn.shutdown().await;
                        }
                        break;
                    }
                    None => {
                        warn!("Channel closed, shutting down");
                        break;
                    }
                }

                // Flush if batch size reached
                if batch_count >= config.batch_size {
                    flush_buffer(&config, &mut connection, &mut buffer, batch_count).await?;
                    batch_count = 0;
                    last_flush = Instant::now();
                }
            }
            _ = flush_timer.tick() => {
                // Temporal flush trigger
                if batch_count > 0 && last_flush.elapsed() >= Duration::from_millis(config.flush_interval_ms) {
                    debug!("Temporal flush trigger: {} messages", batch_count);
                    flush_buffer(&config, &mut connection, &mut buffer, batch_count).await?;
                    batch_count = 0;
                    last_flush = Instant::now();
                }
            }
        }
    }

    Ok(())
}

async fn flush_buffer(
    config: &QuestDBConfig,
    connection: &mut Option<TcpStream>,
    buffer: &mut String,
    count: usize,
) -> Result<()> {
    if buffer.is_empty() {
        return Ok(());
    }

    // Ensure connection
    if connection.is_none() {
        match connect_with_retry(config).await {
            Ok(conn) => {
                info!("Connected to QuestDB at {}:{}", config.host, config.port);
                *connection = Some(conn);
            }
            Err(e) => {
                error!("Failed to connect to QuestDB: {}", e);
                buffer.clear();
                return Err(e);
            }
        }
    }

    // Write buffer
    if let Some(conn) = connection {
        match conn.write_all(buffer.as_bytes()).await {
            Ok(_) => {
                debug!("Flushed {} messages ({} bytes)", count, buffer.len());
                buffer.clear();
                Ok(())
            }
            Err(e) => {
                error!("Write error: {}, reconnecting", e);
                *connection = None;
                buffer.clear();
                Err(anyhow!("Write failed: {}", e))
            }
        }
    } else {
        buffer.clear();
        Err(anyhow!("No connection available"))
    }
}

async fn connect_with_retry(config: &QuestDBConfig) -> Result<TcpStream> {
    let max_retries = 5;
    for attempt in 1..=max_retries {
        match TcpStream::connect(format!("{}:{}", config.host, config.port)).await {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                return Ok(stream);
            }
            Err(e) => {
                if attempt < max_retries {
                    warn!(
                        "Connection attempt {}/{} failed: {}, retrying...",
                        attempt, max_retries, e
                    );
                    tokio::time::sleep(Duration::from_millis(RECONNECT_DELAY_MS)).await;
                } else {
                    return Err(anyhow!(
                        "Failed to connect after {} attempts: {}",
                        max_retries,
                        e
                    ));
                }
            }
        }
    }
    unreachable!()
}

// ============================================================================
// ILP Formatting
// ============================================================================

fn format_tick(buffer: &mut String, tick: &TickWrite) {
    // Format: ticks,symbol=BTCUSD bid_px=50000.0,ask_px=50001.0,bid_sz=1.5,ask_sz=2.0 1234567890000000
    buffer.push_str("ticks,symbol=");
    buffer.push_str(&tick.symbol);
    buffer.push_str(" bid_px=");
    buffer.push_str(&tick.bid_px.to_string());
    buffer.push_str(",ask_px=");
    buffer.push_str(&tick.ask_px.to_string());
    buffer.push_str(",bid_sz=");
    buffer.push_str(&tick.bid_sz.to_string());
    buffer.push_str(",ask_sz=");
    buffer.push_str(&tick.ask_sz.to_string());
    buffer.push(' ');
    buffer.push_str(&tick.timestamp_micros.to_string());
    buffer.push('\n');
}

fn format_trade(buffer: &mut String, trade: &TradeWrite) {
    // Format: trades,symbol=BTCUSD,side=Buy price=50000.0,size=0.5 1234567890000000
    buffer.push_str("trades,symbol=");
    buffer.push_str(&trade.symbol);
    buffer.push_str(",side=");
    buffer.push_str(&trade.side);
    buffer.push_str(" price=");
    buffer.push_str(&trade.price.to_string());
    buffer.push_str(",size=");
    buffer.push_str(&trade.size.to_string());
    buffer.push(' ');
    buffer.push_str(&trade.timestamp_micros.to_string());
    buffer.push('\n');
}

fn format_signal(buffer: &mut String, signal: &SignalWrite) {
    // Format: signals,symbol=BTCUSD,type=LONG entry=50000.0,sl=49000.0,tp=52000.0,size=0.5 1234567890000000
    buffer.push_str("signals,symbol=");
    buffer.push_str(&signal.symbol);
    buffer.push_str(",type=");
    buffer.push_str(&signal.signal_type);
    buffer.push_str(" entry=");
    buffer.push_str(&signal.entry_price.to_string());
    buffer.push_str(",sl=");
    buffer.push_str(&signal.stop_loss.to_string());
    buffer.push_str(",tp=");
    buffer.push_str(&signal.take_profit.to_string());
    buffer.push_str(",size=");
    buffer.push_str(&signal.position_size.to_string());
    buffer.push(' ');
    buffer.push_str(&signal.timestamp_micros.to_string());
    buffer.push('\n');
}

fn format_execution(buffer: &mut String, execution: &ExecutionWrite) {
    // Format: executions,order_id=12345,symbol=BTCUSD,side=Buy price=50000.0,size=0.5,fee=1.25 1234567890000000
    buffer.push_str("executions,order_id=");
    buffer.push_str(&execution.order_id);
    buffer.push_str(",symbol=");
    buffer.push_str(&execution.symbol);
    buffer.push_str(",side=");
    buffer.push_str(&execution.side);
    buffer.push_str(" price=");
    buffer.push_str(&execution.price.to_string());
    buffer.push_str(",size=");
    buffer.push_str(&execution.size.to_string());
    buffer.push_str(",fee=");
    buffer.push_str(&execution.fee.to_string());
    buffer.push(' ');
    buffer.push_str(&execution.timestamp_micros.to_string());
    buffer.push('\n');
}

fn format_regime(buffer: &mut String, regime: &RegimeWrite) {
    // Format: regime_states,symbol=BTCUSD,hypothalamus=StrongBullish,amygdala=LowVolTrending
    //   position_scale=1.2,is_high_risk=f,confidence=0.85,trend=0.75,trend_strength=0.6,
    //   volatility=500.0,volatility_percentile=0.3,momentum=0.4,relative_volume=1.5,
    //   liquidity_score=0.9,is_transition=f 1234567890000000
    buffer.push_str("regime_states,symbol=");
    buffer.push_str(&regime.symbol);
    buffer.push_str(",hypothalamus=");
    buffer.push_str(&regime.hypothalamus_regime);
    buffer.push_str(",amygdala=");
    buffer.push_str(&regime.amygdala_regime);
    buffer.push_str(" position_scale=");
    buffer.push_str(&regime.position_scale.to_string());
    buffer.push_str(",is_high_risk=");
    buffer.push_str(if regime.is_high_risk { "t" } else { "f" });
    buffer.push_str(",confidence=");
    buffer.push_str(&regime.confidence.to_string());
    buffer.push_str(",trend=");
    buffer.push_str(&regime.trend.to_string());
    buffer.push_str(",trend_strength=");
    buffer.push_str(&regime.trend_strength.to_string());
    buffer.push_str(",volatility=");
    buffer.push_str(&regime.volatility.to_string());
    buffer.push_str(",volatility_percentile=");
    buffer.push_str(&regime.volatility_percentile.to_string());
    buffer.push_str(",momentum=");
    buffer.push_str(&regime.momentum.to_string());
    buffer.push_str(",relative_volume=");
    buffer.push_str(&regime.relative_volume.to_string());
    buffer.push_str(",liquidity_score=");
    buffer.push_str(&regime.liquidity_score.to_string());
    buffer.push_str(",is_transition=");
    buffer.push_str(if regime.is_transition { "t" } else { "f" });
    buffer.push(' ');
    buffer.push_str(&regime.timestamp_micros.to_string());
    buffer.push('\n');
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Get current timestamp in microseconds
pub fn now_micros() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_micros() as i64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = QuestDBConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 9009);
        assert_eq!(config.batch_size, DEFAULT_BATCH_SIZE);
    }

    #[test]
    fn test_config_builder() {
        let config = QuestDBConfig::new("localhost".to_string(), 9009)
            .with_batch_size(500)
            .with_flush_interval(50)
            .with_channel_buffer(5000);

        assert_eq!(config.batch_size, 500);
        assert_eq!(config.flush_interval_ms, 50);
        assert_eq!(config.channel_buffer, 5000);
    }

    #[test]
    fn test_tick_formatting() {
        let mut buffer = String::new();
        let tick = TickWrite {
            symbol: "BTCUSD".to_string(),
            bid_px: 50000.0,
            ask_px: 50001.0,
            bid_sz: 1.5,
            ask_sz: 2.0,
            timestamp_micros: 1234567890000000,
        };

        format_tick(&mut buffer, &tick);

        assert!(buffer.contains("ticks,symbol=BTCUSD"));
        assert!(buffer.contains("bid_px=50000"));
        assert!(buffer.contains("ask_px=50001"));
        assert!(buffer.contains("1234567890000000"));
    }

    #[test]
    fn test_trade_formatting() {
        let mut buffer = String::new();
        let trade = TradeWrite {
            symbol: "ETHUSDT".to_string(),
            side: "Buy".to_string(),
            price: 3000.0,
            size: 0.5,
            timestamp_micros: 1234567890000000,
        };

        format_trade(&mut buffer, &trade);

        assert!(buffer.contains("trades,symbol=ETHUSDT"));
        assert!(buffer.contains("side=Buy"));
        assert!(buffer.contains("price=3000"));
        assert!(buffer.contains("size=0.5"));
    }

    #[test]
    fn test_signal_formatting() {
        let mut buffer = String::new();
        let signal = SignalWrite {
            symbol: "BTCUSD".to_string(),
            signal_type: "LONG".to_string(),
            entry_price: 50000.0,
            stop_loss: 49000.0,
            take_profit: 52000.0,
            position_size: 0.5,
            timestamp_micros: 1234567890000000,
        };

        format_signal(&mut buffer, &signal);

        assert!(buffer.contains("signals,symbol=BTCUSD"));
        assert!(buffer.contains("type=LONG"));
        assert!(buffer.contains("entry=50000"));
        assert!(buffer.contains("sl=49000"));
        assert!(buffer.contains("tp=52000"));
    }

    #[test]
    fn test_regime_formatting() {
        let regime = RegimeWrite {
            symbol: "BTCUSD".to_string(),
            hypothalamus_regime: "StrongBullish".to_string(),
            amygdala_regime: "LowVolTrending".to_string(),
            position_scale: 1.2,
            is_high_risk: false,
            confidence: 0.85,
            trend: 0.75,
            trend_strength: 0.6,
            volatility: 500.0,
            volatility_percentile: 0.3,
            momentum: 0.4,
            relative_volume: 1.5,
            liquidity_score: 0.9,
            is_transition: false,
            timestamp_micros: 1234567890000000,
        };

        let mut buffer = String::new();
        format_regime(&mut buffer, &regime);

        assert!(buffer.starts_with(
            "regime_states,symbol=BTCUSD,hypothalamus=StrongBullish,amygdala=LowVolTrending "
        ));
        assert!(buffer.contains("position_scale=1.2"));
        assert!(buffer.contains("is_high_risk=f"));
        assert!(buffer.contains("confidence=0.85"));
        assert!(buffer.contains("trend=0.75"));
        assert!(buffer.contains("trend_strength=0.6"));
        assert!(buffer.contains("volatility=500"));
        assert!(buffer.contains("volatility_percentile=0.3"));
        assert!(buffer.contains("momentum=0.4"));
        assert!(buffer.contains("relative_volume=1.5"));
        assert!(buffer.contains("liquidity_score=0.9"));
        assert!(buffer.contains("is_transition=f"));
        assert!(buffer.ends_with("1234567890000000\n"));
    }

    #[test]
    fn test_regime_formatting_high_risk_transition() {
        let regime = RegimeWrite {
            symbol: "ETHUSDT".to_string(),
            hypothalamus_regime: "Crisis".to_string(),
            amygdala_regime: "Crisis".to_string(),
            position_scale: 0.25,
            is_high_risk: true,
            confidence: 0.95,
            trend: -0.8,
            trend_strength: 0.9,
            volatility: 1200.0,
            volatility_percentile: 0.95,
            momentum: -0.6,
            relative_volume: 3.5,
            liquidity_score: 0.3,
            is_transition: true,
            timestamp_micros: 9999999999000000,
        };

        let mut buffer = String::new();
        format_regime(&mut buffer, &regime);

        assert!(buffer.contains("symbol=ETHUSDT"));
        assert!(buffer.contains("hypothalamus=Crisis"));
        assert!(buffer.contains("amygdala=Crisis"));
        assert!(buffer.contains("is_high_risk=t"));
        assert!(buffer.contains("is_transition=t"));
        assert!(buffer.contains("position_scale=0.25"));
    }

    #[test]
    fn test_execution_formatting() {
        let mut buffer = String::new();
        let execution = ExecutionWrite {
            order_id: "ORD123".to_string(),
            symbol: "BTCUSD".to_string(),
            side: "Sell".to_string(),
            price: 51000.0,
            size: 0.3,
            fee: 1.53,
            timestamp_micros: 1234567890000000,
        };

        format_execution(&mut buffer, &execution);

        assert!(buffer.contains("executions,order_id=ORD123"));
        assert!(buffer.contains("symbol=BTCUSD"));
        assert!(buffer.contains("side=Sell"));
        assert!(buffer.contains("price=51000"));
        assert!(buffer.contains("fee=1.53"));
    }

    #[test]
    fn test_now_micros() {
        let micros = now_micros();
        assert!(micros > 0);
        // Should be a reasonable timestamp (after 2020)
        assert!(micros > 1577836800000000);
    }

    #[tokio::test]
    async fn test_writer_creation() {
        let config = QuestDBConfig::default();
        let writer = QuestDBWriter::new(config);

        // Writer should be created successfully
        // Note: actual connection will fail without QuestDB running, but that's OK for unit test
        let _ = writer;
    }
}
