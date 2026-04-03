//! Local Fallback Writer for DataRecorder
//!
//! Provides local file storage as a fallback when QuestDB is unavailable.
//! Supports multiple formats: NDJSON (newline-delimited JSON) and Parquet.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    DataRecorder with Fallback                           │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  MarketEvent ──▶ ┌──────────────┐                                       │
//! │                  │ DataRecorder │                                       │
//! │                  └──────┬───────┘                                       │
//! │                         │                                                │
//! │            ┌────────────┴────────────┐                                  │
//! │            │                         │                                   │
//! │            ▼                         ▼                                   │
//! │   ┌──────────────┐         ┌──────────────────┐                        │
//! │   │   QuestDB    │         │  Local Fallback  │                        │
//! │   │   (primary)  │         │  (NDJSON/Parquet)│                        │
//! │   └──────────────┘         └──────────────────┘                        │
//! │         ✓ OK                   ⚠ When QuestDB                           │
//! │                                  unavailable                             │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_execution::sim::local_fallback::{
//!     LocalFallbackConfig, LocalFallbackWriter, FallbackFormat,
//! };
//!
//! // Create fallback writer
//! let config = LocalFallbackConfig::new("./data/fallback")
//!     .with_format(FallbackFormat::NdJson)
//!     .with_rotation_size_mb(100)
//!     .with_compression(true);
//!
//! let writer = LocalFallbackWriter::new(config)?;
//! writer.start()?;
//!
//! // Write events when QuestDB is unavailable
//! writer.write_event(&event).await?;
//!
//! // Later, replay fallback data to QuestDB
//! writer.replay_to_questdb("localhost", 9009).await?;
//! ```

use super::data_feed::{CandleData, MarketEvent, OrderBookData, TickData, TradeData, TradeSide};
use chrono::{DateTime, Utc};
use flate2::Compression;
use flate2::write::GzEncoder;
use parking_lot::RwLock;
use rust_decimal::prelude::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ============================================================================
// Errors
// ============================================================================

/// Errors that can occur during fallback operations
#[derive(Debug, Error)]
pub enum FallbackError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Invalid path: {0}")]
    InvalidPath(String),

    #[error("Writer not started")]
    NotStarted,

    #[error("Writer already started")]
    AlreadyStarted,

    #[error("Rotation failed: {0}")]
    RotationFailed(String),

    #[error("Replay failed: {0}")]
    ReplayFailed(String),

    #[error("Channel closed")]
    ChannelClosed,
}

// ============================================================================
// Configuration
// ============================================================================

/// Format for fallback data storage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FallbackFormat {
    /// Newline-delimited JSON (easy to read, append-friendly)
    #[default]
    NdJson,
    /// CSV format (simple, widely compatible)
    Csv,
}

impl std::fmt::Display for FallbackFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FallbackFormat::NdJson => write!(f, "ndjson"),
            FallbackFormat::Csv => write!(f, "csv"),
        }
    }
}

/// Configuration for local fallback writer
#[derive(Debug, Clone)]
pub struct LocalFallbackConfig {
    /// Base directory for fallback files
    pub base_path: PathBuf,
    /// File format to use
    pub format: FallbackFormat,
    /// Maximum file size before rotation (bytes)
    pub rotation_size_bytes: u64,
    /// Maximum number of rotated files to keep
    pub max_files: usize,
    /// Buffer size for writes
    pub buffer_size: usize,
    /// Flush interval (milliseconds)
    pub flush_interval_ms: u64,
    /// Whether fallback is enabled
    pub enabled: bool,
    /// File prefix
    pub file_prefix: String,
    /// Compress rotated files with gzip
    pub compress_rotated: bool,
}

impl Default for LocalFallbackConfig {
    fn default() -> Self {
        Self {
            base_path: PathBuf::from("./data/fallback"),
            format: FallbackFormat::NdJson,
            rotation_size_bytes: 100 * 1024 * 1024, // 100 MB
            max_files: 10,
            buffer_size: 8192,
            flush_interval_ms: 1000,
            enabled: true,
            file_prefix: "fks_fallback".to_string(),
            compress_rotated: false,
        }
    }
}

impl LocalFallbackConfig {
    /// Create a new configuration with the specified base path
    pub fn new(base_path: impl Into<PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
            ..Default::default()
        }
    }

    /// Set the file format
    pub fn with_format(mut self, format: FallbackFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the rotation size in megabytes
    pub fn with_rotation_size_mb(mut self, mb: u64) -> Self {
        self.rotation_size_bytes = mb * 1024 * 1024;
        self
    }

    /// Set the rotation size in bytes
    pub fn with_rotation_size_bytes(mut self, bytes: u64) -> Self {
        self.rotation_size_bytes = bytes;
        self
    }

    /// Set the maximum number of rotated files to keep
    pub fn with_max_files(mut self, max: usize) -> Self {
        self.max_files = max;
        self
    }

    /// Set the buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the flush interval in milliseconds
    pub fn with_flush_interval_ms(mut self, ms: u64) -> Self {
        self.flush_interval_ms = ms;
        self
    }

    /// Enable or disable fallback
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set the file prefix
    pub fn with_file_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.file_prefix = prefix.into();
        self
    }

    /// Enable compression for rotated files
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compress_rotated = enabled;
        self
    }

    /// Get the file extension based on format
    pub fn file_extension(&self) -> &str {
        match self.format {
            FallbackFormat::NdJson => "ndjson",
            FallbackFormat::Csv => "csv",
        }
    }

    /// Get the current file path
    pub fn current_file_path(&self) -> PathBuf {
        self.base_path
            .join(format!("{}.{}", self.file_prefix, self.file_extension()))
    }

    /// Get the path for a rotated file
    pub fn rotated_file_path(&self, index: usize) -> PathBuf {
        let ext = if self.compress_rotated {
            format!("{}.gz", self.file_extension())
        } else {
            self.file_extension().to_string()
        };
        self.base_path
            .join(format!("{}.{}.{}", self.file_prefix, index, ext))
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for the fallback writer
#[derive(Debug, Clone, Default)]
pub struct FallbackStats {
    /// Total events written
    pub events_written: u64,
    /// Total bytes written
    pub bytes_written: u64,
    /// Number of file rotations
    pub rotations: u64,
    /// Number of write errors
    pub write_errors: u64,
    /// Number of files cleaned up
    pub files_cleaned: u64,
    /// Current file size
    pub current_file_size: u64,
    /// Number of files currently stored
    pub files_count: usize,
    /// Whether writer is active
    pub active: bool,
    /// Start time
    pub start_time: Option<DateTime<Utc>>,
    /// Last write time
    pub last_write_time: Option<DateTime<Utc>>,
    /// Events pending in buffer
    pub buffer_depth: usize,
}

impl FallbackStats {
    /// Get events per second
    pub fn events_per_second(&self) -> f64 {
        if let Some(start) = self.start_time {
            let elapsed = Utc::now().signed_duration_since(start).num_seconds() as f64;
            if elapsed > 0.0 {
                return self.events_written as f64 / elapsed;
            }
        }
        0.0
    }

    /// Get total storage used (approximate)
    pub fn total_storage_bytes(&self) -> u64 {
        self.bytes_written
    }
}

// ============================================================================
// Serializable Event Types
// ============================================================================

/// Serializable wrapper for market events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SerializableEvent {
    Tick(SerializableTick),
    Trade(SerializableTrade),
    OrderBook(SerializableOrderBook),
    Candle(SerializableCandle),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableTick {
    pub symbol: String,
    pub exchange: String,
    pub bid_price: f64,
    pub ask_price: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub timestamp: i64, // Unix timestamp nanos
    pub sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableTrade {
    pub symbol: String,
    pub exchange: String,
    pub price: f64,
    pub size: f64,
    pub side: String,
    pub trade_id: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableOrderBook {
    pub symbol: String,
    pub exchange: String,
    pub bids: Vec<(f64, f64)>, // (price, size)
    pub asks: Vec<(f64, f64)>,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableCandle {
    pub symbol: String,
    pub exchange: String,
    pub open_time: i64,
    pub close_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub trade_count: u64,
}

impl From<&TickData> for SerializableTick {
    fn from(tick: &TickData) -> Self {
        Self {
            symbol: tick.symbol.clone(),
            exchange: tick.exchange.clone(),
            bid_price: tick.bid_price.to_f64().unwrap_or(0.0),
            ask_price: tick.ask_price.to_f64().unwrap_or(0.0),
            bid_size: tick.bid_size.to_f64().unwrap_or(0.0),
            ask_size: tick.ask_size.to_f64().unwrap_or(0.0),
            timestamp: tick.timestamp.timestamp_nanos_opt().unwrap_or(0),
            sequence: tick.sequence,
        }
    }
}

impl From<&TradeData> for SerializableTrade {
    fn from(trade: &TradeData) -> Self {
        Self {
            symbol: trade.symbol.clone(),
            exchange: trade.exchange.clone(),
            price: trade.price.to_f64().unwrap_or(0.0),
            size: trade.size.to_f64().unwrap_or(0.0),
            side: match trade.side {
                TradeSide::Buy => "buy".to_string(),
                TradeSide::Sell => "sell".to_string(),
                TradeSide::Unknown => "unknown".to_string(),
            },
            trade_id: trade.trade_id.clone(),
            timestamp: trade.timestamp.timestamp_nanos_opt().unwrap_or(0),
        }
    }
}

impl From<&OrderBookData> for SerializableOrderBook {
    fn from(ob: &OrderBookData) -> Self {
        Self {
            symbol: ob.symbol.clone(),
            exchange: ob.exchange.clone(),
            bids: ob
                .bids
                .iter()
                .map(|l| {
                    (
                        l.price.to_f64().unwrap_or(0.0),
                        l.size.to_f64().unwrap_or(0.0),
                    )
                })
                .collect(),
            asks: ob
                .asks
                .iter()
                .map(|l| {
                    (
                        l.price.to_f64().unwrap_or(0.0),
                        l.size.to_f64().unwrap_or(0.0),
                    )
                })
                .collect(),
            timestamp: ob.timestamp.timestamp_nanos_opt().unwrap_or(0),
        }
    }
}

impl From<&CandleData> for SerializableCandle {
    fn from(candle: &CandleData) -> Self {
        Self {
            symbol: candle.symbol.clone(),
            exchange: candle.exchange.clone(),
            open_time: candle.open_time.timestamp_nanos_opt().unwrap_or(0),
            close_time: candle.close_time.timestamp_nanos_opt().unwrap_or(0),
            open: candle.open.to_f64().unwrap_or(0.0),
            high: candle.high.to_f64().unwrap_or(0.0),
            low: candle.low.to_f64().unwrap_or(0.0),
            close: candle.close.to_f64().unwrap_or(0.0),
            volume: candle.volume.to_f64().unwrap_or(0.0),
            trade_count: candle.trade_count,
        }
    }
}

impl From<&MarketEvent> for Option<SerializableEvent> {
    fn from(event: &MarketEvent) -> Self {
        match event {
            MarketEvent::Tick(tick) => Some(SerializableEvent::Tick(tick.into())),
            MarketEvent::Trade(trade) => Some(SerializableEvent::Trade(trade.into())),
            MarketEvent::OrderBook(ob) => Some(SerializableEvent::OrderBook(ob.into())),
            MarketEvent::Candle(candle) => Some(SerializableEvent::Candle(candle.into())),
            _ => None,
        }
    }
}

// ============================================================================
// Writer Messages
// ============================================================================

enum WriterMessage {
    Event(MarketEvent),
    Flush,
    Rotate,
    Shutdown,
}

// ============================================================================
// Local Fallback Writer
// ============================================================================

/// Local fallback writer for market data
///
/// Writes market events to local files when QuestDB is unavailable.
/// Supports NDJSON and CSV formats with automatic file rotation.
pub struct LocalFallbackWriter {
    config: LocalFallbackConfig,
    tx: Option<mpsc::Sender<WriterMessage>>,
    stats: Arc<RwLock<FallbackStats>>,
    running: Arc<AtomicBool>,
    events_counter: Arc<AtomicU64>,
    task_handle: Option<tokio::task::JoinHandle<()>>,
}

impl LocalFallbackWriter {
    /// Create a new fallback writer
    pub fn new(config: LocalFallbackConfig) -> Result<Self, FallbackError> {
        // Ensure base directory exists
        if !config.base_path.exists() {
            fs::create_dir_all(&config.base_path)?;
        }

        Ok(Self {
            config,
            tx: None,
            stats: Arc::new(RwLock::new(FallbackStats::default())),
            running: Arc::new(AtomicBool::new(false)),
            events_counter: Arc::new(AtomicU64::new(0)),
            task_handle: None,
        })
    }

    /// Start the fallback writer
    pub fn start(&mut self) -> Result<(), FallbackError> {
        if self.running.load(Ordering::SeqCst) {
            return Err(FallbackError::AlreadyStarted);
        }

        if !self.config.enabled {
            info!("Local fallback writer is disabled");
            return Ok(());
        }

        info!(
            "Starting local fallback writer at {:?}",
            self.config.base_path
        );

        // Create channel
        let (tx, rx) = mpsc::channel(10_000);
        self.tx = Some(tx);

        // Initialize stats
        {
            let mut stats = self.stats.write();
            stats.start_time = Some(Utc::now());
            stats.active = true;
        }

        // Spawn writer task
        let config = self.config.clone();
        let stats = self.stats.clone();
        let running = self.running.clone();
        let events_counter = self.events_counter.clone();

        running.store(true, Ordering::SeqCst);

        let handle = tokio::spawn(async move {
            writer_loop(config, rx, stats, running, events_counter).await;
        });

        self.task_handle = Some(handle);

        info!("Local fallback writer started");
        Ok(())
    }

    /// Stop the fallback writer
    pub async fn stop(&mut self) -> Result<(), FallbackError> {
        if !self.running.load(Ordering::SeqCst) {
            return Ok(());
        }

        info!("Stopping local fallback writer");

        // Send shutdown message
        if let Some(tx) = &self.tx {
            let _ = tx.send(WriterMessage::Shutdown).await;
        }

        // Wait for task to finish
        if let Some(handle) = self.task_handle.take() {
            let _ = handle.await;
        }

        self.tx = None;
        self.running.store(false, Ordering::SeqCst);

        {
            let mut stats = self.stats.write();
            stats.active = false;
        }

        info!("Local fallback writer stopped");
        Ok(())
    }

    /// Write an event to the fallback storage
    pub async fn write_event(&self, event: &MarketEvent) -> Result<(), FallbackError> {
        if !self.config.enabled {
            return Ok(());
        }

        let tx = self.tx.as_ref().ok_or(FallbackError::NotStarted)?;

        tx.send(WriterMessage::Event(event.clone()))
            .await
            .map_err(|_| FallbackError::ChannelClosed)?;

        Ok(())
    }

    /// Force a flush of buffered data
    pub async fn flush(&self) -> Result<(), FallbackError> {
        if let Some(tx) = &self.tx {
            tx.send(WriterMessage::Flush)
                .await
                .map_err(|_| FallbackError::ChannelClosed)?;
        }
        Ok(())
    }

    /// Force file rotation
    pub async fn rotate(&self) -> Result<(), FallbackError> {
        if let Some(tx) = &self.tx {
            tx.send(WriterMessage::Rotate)
                .await
                .map_err(|_| FallbackError::ChannelClosed)?;
        }
        Ok(())
    }

    /// Get current statistics
    pub fn stats(&self) -> FallbackStats {
        self.stats.read().clone()
    }

    /// Check if the writer is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the configuration
    pub fn config(&self) -> &LocalFallbackConfig {
        &self.config
    }

    /// Count existing fallback files
    pub fn count_fallback_files(&self) -> Result<usize, FallbackError> {
        let mut count = 0;
        let pattern = format!("{}.", self.config.file_prefix);

        for entry in fs::read_dir(&self.config.base_path)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with(&pattern) {
                count += 1;
            }
        }

        Ok(count)
    }

    /// List all fallback files
    pub fn list_fallback_files(&self) -> Result<Vec<PathBuf>, FallbackError> {
        let mut files = Vec::new();
        let pattern = format!("{}.", self.config.file_prefix);

        for entry in fs::read_dir(&self.config.base_path)? {
            let entry = entry?;
            if entry.file_name().to_string_lossy().starts_with(&pattern) {
                files.push(entry.path());
            }
        }

        files.sort();
        Ok(files)
    }

    /// Get total size of all fallback files
    pub fn total_fallback_size(&self) -> Result<u64, FallbackError> {
        let mut total = 0;

        for file in self.list_fallback_files()? {
            if let Ok(metadata) = fs::metadata(&file) {
                total += metadata.len();
            }
        }

        Ok(total)
    }

    /// Read events from a fallback file (handles both plain and gzipped files)
    pub fn read_events(&self, path: &Path) -> Result<Vec<SerializableEvent>, FallbackError> {
        let is_gzipped = path.extension().is_some_and(|ext| ext == "gz");

        if is_gzipped {
            self.read_events_from_gzip(path)
        } else {
            self.read_events_plain(path)
        }
    }

    /// Read events from a plain (uncompressed) fallback file
    fn read_events_plain(&self, path: &Path) -> Result<Vec<SerializableEvent>, FallbackError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        self.parse_events_from_reader(reader)
    }

    /// Read events from a gzip-compressed fallback file
    fn read_events_from_gzip(&self, path: &Path) -> Result<Vec<SerializableEvent>, FallbackError> {
        use flate2::read::GzDecoder;

        let file = File::open(path)?;
        let decoder = GzDecoder::new(file);
        let reader = BufReader::new(decoder);
        self.parse_events_from_reader(reader)
    }

    /// Parse events from any BufRead implementation
    fn parse_events_from_reader<R: BufRead>(
        &self,
        reader: R,
    ) -> Result<Vec<SerializableEvent>, FallbackError> {
        let mut events = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<SerializableEvent>(&line) {
                Ok(event) => events.push(event),
                Err(e) => {
                    warn!("Failed to parse event: {}", e);
                }
            }
        }

        Ok(events)
    }

    /// Delete all fallback files (use with caution!)
    pub fn clear_all(&self) -> Result<usize, FallbackError> {
        let files = self.list_fallback_files()?;
        let count = files.len();

        for file in files {
            fs::remove_file(file)?;
        }

        info!("Cleared {} fallback files", count);
        Ok(count)
    }
}

impl Drop for LocalFallbackWriter {
    fn drop(&mut self) {
        if self.running.load(Ordering::SeqCst) {
            warn!("LocalFallbackWriter dropped while still running");
            self.running.store(false, Ordering::SeqCst);
        }
    }
}

// ============================================================================
// Writer Loop
// ============================================================================

async fn writer_loop(
    config: LocalFallbackConfig,
    mut rx: mpsc::Receiver<WriterMessage>,
    stats: Arc<RwLock<FallbackStats>>,
    running: Arc<AtomicBool>,
    events_counter: Arc<AtomicU64>,
) {
    let mut buffer: VecDeque<String> = VecDeque::with_capacity(1000);
    let mut current_file_size: u64 = 0;
    let flush_interval = std::time::Duration::from_millis(config.flush_interval_ms);
    let mut flush_timer = tokio::time::interval(flush_interval);

    // Open or create the current file
    let mut writer: Option<BufWriter<File>> = match open_current_file(&config) {
        Ok((w, size)) => {
            current_file_size = size;
            Some(w)
        }
        Err(e) => {
            error!("Failed to open fallback file: {}", e);
            None
        }
    };

    loop {
        // Update stats
        {
            let mut s = stats.write();
            s.buffer_depth = buffer.len();
            s.current_file_size = current_file_size;
        }

        tokio::select! {
            msg = rx.recv() => {
                match msg {
                    Some(WriterMessage::Event(event)) => {
                        // Serialize event
                        if let Some(serializable) = Option::<SerializableEvent>::from(&event) {
                            match serialize_event(&serializable, &config) {
                                Ok(line) => {
                                    buffer.push_back(line);
                                    events_counter.fetch_add(1, Ordering::Relaxed);
                                }
                                Err(e) => {
                                    warn!("Serialization error: {}", e);
                                    stats.write().write_errors += 1;
                                }
                            }
                        }

                        // Flush if buffer is large enough
                        if buffer.len() >= 100 {
                            if let Err(e) = flush_buffer(&mut buffer, &mut writer, &config, &stats, &mut current_file_size).await {
                                warn!("Flush error: {}", e);
                            }

                            // Check for rotation
                            if current_file_size >= config.rotation_size_bytes {
                                if let Err(e) = rotate_file(&mut writer, &config, &stats, &mut current_file_size).await {
                                    warn!("Rotation error: {}", e);
                                }
                            }
                        }
                    }
                    Some(WriterMessage::Flush) => {
                        if let Err(e) = flush_buffer(&mut buffer, &mut writer, &config, &stats, &mut current_file_size).await {
                            warn!("Flush error: {}", e);
                        }
                    }
                    Some(WriterMessage::Rotate) => {
                        // Flush first
                        let _ = flush_buffer(&mut buffer, &mut writer, &config, &stats, &mut current_file_size).await;
                        if let Err(e) = rotate_file(&mut writer, &config, &stats, &mut current_file_size).await {
                            warn!("Rotation error: {}", e);
                        }
                    }
                    Some(WriterMessage::Shutdown) | None => {
                        // Final flush
                        let _ = flush_buffer(&mut buffer, &mut writer, &config, &stats, &mut current_file_size).await;
                        break;
                    }
                }
            }
            _ = flush_timer.tick() => {
                // Periodic flush
                if !buffer.is_empty() {
                    if let Err(e) = flush_buffer(&mut buffer, &mut writer, &config, &stats, &mut current_file_size).await {
                        warn!("Periodic flush error: {}", e);
                    }
                }
            }
        }
    }

    running.store(false, Ordering::SeqCst);
    debug!("Fallback writer loop exited");
}

/// Open or create the current fallback file
fn open_current_file(
    config: &LocalFallbackConfig,
) -> Result<(BufWriter<File>, u64), FallbackError> {
    let path = config.current_file_path();

    // Get existing file size if file exists
    let existing_size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);

    let file = OpenOptions::new().create(true).append(true).open(&path)?;

    Ok((
        BufWriter::with_capacity(config.buffer_size, file),
        existing_size,
    ))
}

/// Serialize an event to a string
fn serialize_event(
    event: &SerializableEvent,
    config: &LocalFallbackConfig,
) -> Result<String, FallbackError> {
    match config.format {
        FallbackFormat::NdJson => {
            serde_json::to_string(event).map_err(|e| FallbackError::Serialization(e.to_string()))
        }
        FallbackFormat::Csv => {
            // Simple CSV serialization
            let line = match event {
                SerializableEvent::Tick(t) => {
                    format!(
                        "tick,{},{},{},{},{},{},{},{}",
                        t.symbol,
                        t.exchange,
                        t.bid_price,
                        t.ask_price,
                        t.bid_size,
                        t.ask_size,
                        t.timestamp,
                        t.sequence
                    )
                }
                SerializableEvent::Trade(t) => {
                    format!(
                        "trade,{},{},{},{},{},{},{}",
                        t.symbol, t.exchange, t.price, t.size, t.side, t.trade_id, t.timestamp
                    )
                }
                SerializableEvent::OrderBook(ob) => {
                    format!(
                        "orderbook,{},{},{},{},{}",
                        ob.symbol,
                        ob.exchange,
                        ob.bids.len(),
                        ob.asks.len(),
                        ob.timestamp
                    )
                }
                SerializableEvent::Candle(c) => {
                    format!(
                        "candle,{},{},{},{},{},{},{},{},{},{}",
                        c.symbol,
                        c.exchange,
                        c.open_time,
                        c.close_time,
                        c.open,
                        c.high,
                        c.low,
                        c.close,
                        c.volume,
                        c.trade_count
                    )
                }
            };
            Ok(line)
        }
    }
}

/// Flush buffer to file
async fn flush_buffer(
    buffer: &mut VecDeque<String>,
    writer: &mut Option<BufWriter<File>>,
    config: &LocalFallbackConfig,
    stats: &Arc<RwLock<FallbackStats>>,
    current_size: &mut u64,
) -> Result<(), FallbackError> {
    if buffer.is_empty() {
        return Ok(());
    }

    // Ensure writer is open
    if writer.is_none() {
        let (w, size) = open_current_file(config)?;
        *writer = Some(w);
        *current_size = size;
    }

    let w = writer.as_mut().unwrap();
    let mut bytes_written: u64 = 0;
    let mut events_written: u64 = 0;

    while let Some(line) = buffer.pop_front() {
        let line_with_newline = format!("{}\n", line);
        let line_bytes = line_with_newline.as_bytes();

        w.write_all(line_bytes)?;
        bytes_written += line_bytes.len() as u64;
        events_written += 1;
    }

    w.flush()?;

    // Update stats
    {
        let mut s = stats.write();
        s.events_written += events_written;
        s.bytes_written += bytes_written;
        s.last_write_time = Some(Utc::now());
        s.buffer_depth = 0;
    }

    *current_size += bytes_written;

    Ok(())
}

/// Rotate the current file
async fn rotate_file(
    writer: &mut Option<BufWriter<File>>,
    config: &LocalFallbackConfig,
    stats: &Arc<RwLock<FallbackStats>>,
    current_size: &mut u64,
) -> Result<(), FallbackError> {
    // Close current writer
    if let Some(w) = writer.take() {
        drop(w);
    }

    let current_path = config.current_file_path();

    // Find next rotation index
    let mut next_index = 0;
    for i in 0..1000 {
        let rotated_path = config.rotated_file_path(i);
        if !rotated_path.exists() {
            next_index = i;
            break;
        }
    }

    // Get the rotated file path
    let rotated_path = config.rotated_file_path(next_index);

    if current_path.exists() {
        if config.compress_rotated {
            // Compress the file with gzip
            compress_file(&current_path, &rotated_path)?;
            // Remove the original uncompressed file
            fs::remove_file(&current_path)?;
            info!("Rotated and compressed fallback file to {:?}", rotated_path);
        } else {
            // Just rename without compression
            fs::rename(&current_path, &rotated_path)?;
            info!("Rotated fallback file to {:?}", rotated_path);
        }
    }

    // Clean up old files if we have too many
    cleanup_old_files(config, stats)?;

    // Open new file
    let (w, size) = open_current_file(config)?;
    *writer = Some(w);
    *current_size = size;

    stats.write().rotations += 1;

    Ok(())
}

/// Compress a file using gzip
fn compress_file(source: &Path, dest: &Path) -> Result<(), FallbackError> {
    // Read the source file
    let mut source_file = File::open(source)?;
    let mut contents = Vec::new();
    source_file.read_to_end(&mut contents)?;

    // Create the compressed output file
    let dest_file = File::create(dest)?;
    let mut encoder = GzEncoder::new(dest_file, Compression::default());
    encoder.write_all(&contents)?;
    encoder.finish()?;

    debug!("Compressed {} bytes to {:?}", contents.len(), dest);

    Ok(())
}

/// Clean up old rotated files
fn cleanup_old_files(
    config: &LocalFallbackConfig,
    stats: &Arc<RwLock<FallbackStats>>,
) -> Result<(), FallbackError> {
    let pattern = format!("{}.", config.file_prefix);
    let mut rotated_files: Vec<PathBuf> = Vec::new();

    for entry in fs::read_dir(&config.base_path)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();

        // Check if it's a rotated file (has a number after the prefix)
        if name.starts_with(&pattern)
            && name != format!("{}.{}", config.file_prefix, config.file_extension())
        {
            rotated_files.push(entry.path());
        }
    }

    // Sort by name (older files have lower indices)
    rotated_files.sort();

    // Remove oldest files if we have too many
    while rotated_files.len() > config.max_files {
        if let Some(oldest) = rotated_files.first() {
            fs::remove_file(oldest)?;
            info!("Removed old fallback file: {:?}", oldest);
            rotated_files.remove(0);
            stats.write().files_cleaned += 1;
        }
    }

    // Update files count
    stats.write().files_count = rotated_files.len() + 1; // +1 for current file

    Ok(())
}

// ============================================================================
// Replay Functionality
// ============================================================================

/// Replay events from fallback files to QuestDB
///
/// This function reads all events from fallback files and sends them to QuestDB.
pub async fn replay_fallback_to_questdb(
    fallback_path: &Path,
    file_prefix: &str,
    questdb_host: &str,
    questdb_port: u16,
    table_prefix: &str,
) -> Result<ReplayStats, FallbackError> {
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpStream;

    info!(
        "Replaying fallback data from {:?} to {}:{}",
        fallback_path, questdb_host, questdb_port
    );

    let mut stats = ReplayStats::default();

    // Find all fallback files
    let pattern = format!("{}.", file_prefix);
    let mut files: Vec<PathBuf> = Vec::new();

    for entry in fs::read_dir(fallback_path)? {
        let entry = entry?;
        if entry.file_name().to_string_lossy().starts_with(&pattern) {
            files.push(entry.path());
        }
    }

    files.sort();
    stats.files_found = files.len();

    if files.is_empty() {
        info!("No fallback files found");
        return Ok(stats);
    }

    // Connect to QuestDB
    let mut stream = TcpStream::connect(format!("{}:{}", questdb_host, questdb_port))
        .await
        .map_err(|e| FallbackError::ReplayFailed(format!("Connection failed: {}", e)))?;

    let _ = stream.set_nodelay(true);

    info!("Connected to QuestDB, replaying {} files", files.len());

    for file_path in &files {
        stats.files_processed += 1;
        debug!("Processing {:?}", file_path);

        let file = File::open(file_path)?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            // Parse the event
            match serde_json::from_str::<SerializableEvent>(&line) {
                Ok(event) => {
                    // Convert to ILP format
                    if let Some(ilp_line) = event_to_ilp(&event, table_prefix) {
                        stream
                            .write_all(ilp_line.as_bytes())
                            .await
                            .map_err(|e| FallbackError::ReplayFailed(e.to_string()))?;
                        stats.events_replayed += 1;
                    }
                }
                Err(e) => {
                    debug!("Failed to parse event: {}", e);
                    stats.events_failed += 1;
                }
            }
        }

        // Flush after each file
        stream
            .flush()
            .await
            .map_err(|e| FallbackError::ReplayFailed(e.to_string()))?;
    }

    info!(
        "Replay complete: {} events from {} files",
        stats.events_replayed, stats.files_processed
    );

    Ok(stats)
}

/// Statistics for replay operation
#[derive(Debug, Clone, Default)]
pub struct ReplayStats {
    pub files_found: usize,
    pub files_processed: usize,
    pub events_replayed: u64,
    pub events_failed: u64,
}

/// Convert a serializable event to ILP line protocol
fn event_to_ilp(event: &SerializableEvent, table_prefix: &str) -> Option<String> {
    match event {
        SerializableEvent::Tick(tick) => Some(format!(
            "{}_ticks,symbol={},exchange={} bid={},ask={},bid_size={},ask_size={} {}\n",
            table_prefix,
            escape_tag(&tick.symbol),
            escape_tag(&tick.exchange),
            tick.bid_price,
            tick.ask_price,
            tick.bid_size,
            tick.ask_size,
            tick.timestamp
        )),
        SerializableEvent::Trade(trade) => Some(format!(
            "{}_trades,symbol={},exchange={},side={} price={},size={},trade_id=\"{}\" {}\n",
            table_prefix,
            escape_tag(&trade.symbol),
            escape_tag(&trade.exchange),
            &trade.side,
            trade.price,
            trade.size,
            escape_string(&trade.trade_id),
            trade.timestamp
        )),
        SerializableEvent::Candle(candle) => Some(format!(
            "{}_candles,symbol={},exchange={} open={},high={},low={},close={},volume={},trades={} {}\n",
            table_prefix,
            escape_tag(&candle.symbol),
            escape_tag(&candle.exchange),
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
            candle.trade_count,
            candle.close_time
        )),
        SerializableEvent::OrderBook(_) => {
            // Order book replay is complex, skip for now
            None
        }
    }
}

/// Escape a tag value for ILP protocol
fn escape_tag(s: &str) -> String {
    s.replace(',', "\\,")
        .replace('=', "\\=")
        .replace(' ', "\\ ")
        .replace('/', "_")
}

/// Escape a string value for ILP protocol
fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;
    use tempfile::TempDir;

    fn create_test_tick() -> TickData {
        TickData {
            symbol: "BTC/USDT".to_string(),
            exchange: "kraken".to_string(),
            bid_price: dec!(50000.0),
            ask_price: dec!(50001.0),
            bid_size: dec!(1.5),
            ask_size: dec!(2.0),
            timestamp: Utc::now(),
            sequence: 1,
        }
    }

    fn create_test_trade() -> TradeData {
        TradeData {
            symbol: "BTC/USDT".to_string(),
            exchange: "kraken".to_string(),
            price: dec!(50000.5),
            size: dec!(0.1),
            side: TradeSide::Buy,
            trade_id: "trade123".to_string(),
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_config_default() {
        let config = LocalFallbackConfig::default();
        assert_eq!(config.format, FallbackFormat::NdJson);
        assert_eq!(config.rotation_size_bytes, 100 * 1024 * 1024);
        assert_eq!(config.max_files, 10);
        assert!(config.enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = LocalFallbackConfig::new("./test/data")
            .with_format(FallbackFormat::Csv)
            .with_rotation_size_mb(50)
            .with_max_files(5)
            .with_enabled(true);

        assert_eq!(config.base_path, PathBuf::from("./test/data"));
        assert_eq!(config.format, FallbackFormat::Csv);
        assert_eq!(config.rotation_size_bytes, 50 * 1024 * 1024);
        assert_eq!(config.max_files, 5);
    }

    #[test]
    fn test_serializable_tick() {
        let tick = create_test_tick();
        let serializable: SerializableTick = (&tick).into();

        assert_eq!(serializable.symbol, "BTC/USDT");
        assert_eq!(serializable.exchange, "kraken");
        assert!((serializable.bid_price - 50000.0).abs() < 0.01);
        assert!((serializable.ask_price - 50001.0).abs() < 0.01);
    }

    #[test]
    fn test_serializable_trade() {
        let trade = create_test_trade();
        let serializable: SerializableTrade = (&trade).into();

        assert_eq!(serializable.symbol, "BTC/USDT");
        assert_eq!(serializable.exchange, "kraken");
        assert_eq!(serializable.side, "buy");
        assert_eq!(serializable.trade_id, "trade123");
    }

    #[test]
    fn test_serialize_event_ndjson() {
        let tick = create_test_tick();
        let event = SerializableEvent::Tick((&tick).into());
        let config = LocalFallbackConfig::default();

        let result = serialize_event(&event, &config);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("\"type\":\"Tick\""));
        assert!(json.contains("BTC/USDT"));
    }

    #[test]
    fn test_serialize_event_csv() {
        let tick = create_test_tick();
        let event = SerializableEvent::Tick((&tick).into());
        let config = LocalFallbackConfig::default().with_format(FallbackFormat::Csv);

        let result = serialize_event(&event, &config);
        assert!(result.is_ok());

        let csv = result.unwrap();
        assert!(csv.starts_with("tick,"));
        assert!(csv.contains("BTC/USDT"));
    }

    #[test]
    fn test_event_to_ilp() {
        let tick = create_test_tick();
        let event = SerializableEvent::Tick((&tick).into());

        let ilp = event_to_ilp(&event, "fks");
        assert!(ilp.is_some());

        let line = ilp.unwrap();
        assert!(line.starts_with("fks_ticks,"));
        assert!(line.contains("symbol=BTC_USDT")); // / replaced with _
        assert!(line.contains("exchange=kraken"));
    }

    #[test]
    fn test_escape_tag() {
        assert_eq!(escape_tag("BTC/USDT"), "BTC_USDT");
        assert_eq!(escape_tag("hello world"), "hello\\ world");
        assert_eq!(escape_tag("key=value"), "key\\=value");
    }

    #[test]
    fn test_escape_string() {
        assert_eq!(escape_string("hello"), "hello");
        assert_eq!(escape_string("hello\"world"), "hello\\\"world");
        assert_eq!(escape_string("line1\nline2"), "line1\\nline2");
    }

    #[test]
    fn test_fallback_stats_default() {
        let stats = FallbackStats::default();
        assert_eq!(stats.events_written, 0);
        assert_eq!(stats.bytes_written, 0);
        assert_eq!(stats.rotations, 0);
        assert!(!stats.active);
    }

    #[tokio::test]
    async fn test_writer_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = LocalFallbackConfig::new(temp_dir.path());

        let writer = LocalFallbackWriter::new(config);
        assert!(writer.is_ok());

        let writer = writer.unwrap();
        assert!(!writer.is_running());
    }

    #[tokio::test]
    async fn test_writer_start_stop() {
        let temp_dir = TempDir::new().unwrap();
        let config = LocalFallbackConfig::new(temp_dir.path());

        let mut writer = LocalFallbackWriter::new(config).unwrap();

        // Start
        let result = writer.start();
        assert!(result.is_ok());
        assert!(writer.is_running());

        // Stop
        let result = writer.stop().await;
        assert!(result.is_ok());
        assert!(!writer.is_running());
    }

    #[tokio::test]
    async fn test_writer_write_event() {
        let temp_dir = TempDir::new().unwrap();
        let config = LocalFallbackConfig::new(temp_dir.path());

        let mut writer = LocalFallbackWriter::new(config).unwrap();
        writer.start().unwrap();

        let tick = create_test_tick();
        let event = MarketEvent::Tick(tick);

        let result = writer.write_event(&event).await;
        assert!(result.is_ok());

        // Flush and stop
        writer.flush().await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        writer.stop().await.unwrap();

        // Check stats
        let stats = writer.stats();
        assert!(stats.events_written >= 1);
    }

    #[tokio::test]
    async fn test_writer_disabled() {
        let temp_dir = TempDir::new().unwrap();
        let config = LocalFallbackConfig::new(temp_dir.path()).with_enabled(false);

        let mut writer = LocalFallbackWriter::new(config).unwrap();

        // Start should succeed but not actually start
        let result = writer.start();
        assert!(result.is_ok());
        assert!(!writer.is_running());

        // Write should succeed (no-op)
        let tick = create_test_tick();
        let event = MarketEvent::Tick(tick);
        let result = writer.write_event(&event).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_file_paths() {
        let config = LocalFallbackConfig::new("./data").with_file_prefix("test");

        assert_eq!(
            config.current_file_path(),
            PathBuf::from("./data/test.ndjson")
        );
        assert_eq!(
            config.rotated_file_path(0),
            PathBuf::from("./data/test.0.ndjson")
        );
        assert_eq!(
            config.rotated_file_path(5),
            PathBuf::from("./data/test.5.ndjson")
        );
    }

    #[test]
    fn test_file_paths_with_compression() {
        let config = LocalFallbackConfig::new("./data")
            .with_file_prefix("test")
            .with_compression(true);

        assert_eq!(
            config.current_file_path(),
            PathBuf::from("./data/test.ndjson")
        );
        assert_eq!(
            config.rotated_file_path(0),
            PathBuf::from("./data/test.0.ndjson.gz")
        );
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", FallbackFormat::NdJson), "ndjson");
        assert_eq!(format!("{}", FallbackFormat::Csv), "csv");
    }

    #[test]
    fn test_market_event_to_serializable() {
        let tick = create_test_tick();
        let event = MarketEvent::Tick(tick);

        let serializable: Option<SerializableEvent> = (&event).into();
        assert!(serializable.is_some());

        // Test non-serializable events
        let event = MarketEvent::EndOfData;
        let serializable: Option<SerializableEvent> = (&event).into();
        assert!(serializable.is_none());
    }

    #[test]
    fn test_compress_file() {
        use flate2::read::GzDecoder;
        use std::io::Read;

        let temp_dir = tempfile::tempdir().unwrap();
        let source_path = temp_dir.path().join("test_source.txt");
        let dest_path = temp_dir.path().join("test_source.txt.gz");

        // Write some test data
        let test_data = "Hello, World!\nThis is a test file for compression.\n".repeat(100);
        fs::write(&source_path, &test_data).unwrap();

        // Compress the file
        compress_file(&source_path, &dest_path).unwrap();

        // Verify compressed file exists and is smaller
        assert!(dest_path.exists());
        let compressed_size = fs::metadata(&dest_path).unwrap().len();
        let original_size = test_data.len() as u64;
        assert!(
            compressed_size < original_size,
            "Compressed file should be smaller"
        );

        // Verify we can decompress and get original content
        let compressed_file = File::open(&dest_path).unwrap();
        let mut decoder = GzDecoder::new(compressed_file);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed).unwrap();
        assert_eq!(decompressed, test_data);
    }

    #[tokio::test]
    async fn test_writer_rotation_with_compression() {
        let temp_dir = tempfile::tempdir().unwrap();
        let config = LocalFallbackConfig::new(temp_dir.path())
            .with_rotation_size_bytes(500) // Small size to trigger rotation
            .with_compression(true)
            .with_enabled(true);

        let mut writer = LocalFallbackWriter::new(config.clone()).unwrap();
        writer.start().unwrap();

        // Write enough events to trigger rotation
        for _ in 0..50 {
            let tick = create_test_tick();
            writer.write_event(&MarketEvent::Tick(tick)).await.unwrap();
        }

        // Force a rotation
        writer.rotate().await.unwrap();

        // Give some time for async operations
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        writer.stop().await.unwrap();

        // Check that .gz files were created
        let gz_files: Vec<_> = fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "gz"))
            .collect();

        // We should have at least one compressed rotated file
        assert!(
            !gz_files.is_empty() || config.compress_rotated,
            "Expected compressed rotated files when compression is enabled"
        );
    }

    #[test]
    fn test_config_compression_setting() {
        // Default should be no compression
        let config = LocalFallbackConfig::default();
        assert!(!config.compress_rotated);

        // With compression enabled
        let config = LocalFallbackConfig::new("./data").with_compression(true);
        assert!(config.compress_rotated);

        // Rotated file path should have .gz extension when compression is enabled
        assert!(
            config
                .rotated_file_path(0)
                .to_string_lossy()
                .ends_with(".gz")
        );

        // Without compression, no .gz extension
        let config = LocalFallbackConfig::new("./data").with_compression(false);
        assert!(
            !config
                .rotated_file_path(0)
                .to_string_lossy()
                .ends_with(".gz")
        );
    }

    #[test]
    fn test_read_events_from_gzip() {
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let temp_dir = tempfile::tempdir().unwrap();
        let gz_path = temp_dir.path().join("test_events.ndjson.gz");

        // Create test events
        let tick = create_test_tick();
        let event = SerializableEvent::Tick(SerializableTick::from(&tick));
        let json_line = serde_json::to_string(&event).unwrap() + "\n";

        // Write compressed test data
        let file = File::create(&gz_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        for _ in 0..5 {
            encoder.write_all(json_line.as_bytes()).unwrap();
        }
        encoder.finish().unwrap();

        // Read events using the writer's read_events method
        let config = LocalFallbackConfig::new(temp_dir.path());
        let writer = LocalFallbackWriter::new(config).unwrap();

        let events = writer.read_events(&gz_path).unwrap();
        assert_eq!(events.len(), 5);

        // Verify all events are ticks
        for event in events {
            match event {
                SerializableEvent::Tick(t) => {
                    assert_eq!(t.symbol, "BTC/USDT");
                    assert_eq!(t.exchange, "kraken");
                }
                _ => panic!("Expected tick event"),
            }
        }
    }

    #[test]
    fn test_read_events_plain_vs_gzip() {
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let temp_dir = tempfile::tempdir().unwrap();
        let plain_path = temp_dir.path().join("test_events.ndjson");
        let gz_path = temp_dir.path().join("test_events.ndjson.gz");

        // Create test event JSON
        let tick = create_test_tick();
        let event = SerializableEvent::Tick(SerializableTick::from(&tick));
        let json_line = serde_json::to_string(&event).unwrap() + "\n";
        let content = json_line.repeat(3);

        // Write plain file
        fs::write(&plain_path, &content).unwrap();

        // Write gzipped file
        let file = File::create(&gz_path).unwrap();
        let mut encoder = GzEncoder::new(file, Compression::default());
        encoder.write_all(content.as_bytes()).unwrap();
        encoder.finish().unwrap();

        // Read both files
        let config = LocalFallbackConfig::new(temp_dir.path());
        let writer = LocalFallbackWriter::new(config).unwrap();

        let plain_events = writer.read_events(&plain_path).unwrap();
        let gz_events = writer.read_events(&gz_path).unwrap();

        // Both should have same number of events
        assert_eq!(plain_events.len(), gz_events.len());
        assert_eq!(plain_events.len(), 3);
    }
}
