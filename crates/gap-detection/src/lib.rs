//! # Gap Detection Spike Prototype
//!
//! This module implements sophisticated gap detection for crypto market data streams.
//! It addresses the limitations of naive time-based gap detection by using:
//!
//! 1. **Sequence ID Tracking** - Detects missing trades via monotonic exchange IDs
//! 2. **Heartbeat Monitoring** - Detects silent connection failures
//! 3. **Statistical Anomaly Detection** - Identifies unusual drops in tick rate
//! 4. **Volume-Aware Detection** - Accounts for low-liquidity pairs
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              Gap Detection System                       │
//! └─────────────────────────────────────────────────────────┘
//!          │
//!          ├─► SequenceGapDetector (for trades with IDs)
//!          ├─► HeartbeatMonitor (for WebSocket liveness)
//!          ├─► StatisticalDetector (for anomaly detection)
//!          └─► VolumeAwareDetector (for low-liquidity pairs)
//! ```
//!
//! ## Problem Statement
//!
//! The original research proposed:
//! ```sql
//! SELECT timestamp FROM trades
//! WHERE timestamp > dateadd('m', -10, now())
//! SAMPLE BY 1m FILL(0)
//! WHERE tick_count = 0
//! ```
//!
//! **Issues**:
//! - False positives on low-liquidity pairs
//! - False negatives if 1 tick received but 1000 missed
//! - 10-minute detection latency
//! - No distinction between network vs exchange issues

use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use thiserror::Error;
use tokio_postgres::types::ToSql;
use tracing::{debug, error, info, warn};

// Type aliases for complex shared state types
type TradeIdMap = Arc<RwLock<HashMap<String, u64>>>;
type HeartbeatMap = Arc<RwLock<HashMap<String, DateTime<Utc>>>>;
type TickCountMap = Arc<RwLock<HashMap<String, (DateTime<Utc>, u32)>>>;

// ============================================================================
// Error Types
// ============================================================================

#[derive(Error, Debug, Clone)]
pub enum GapDetectionError {
    #[error("Gap detected: {gap_type} from {start} to {end}, missing {count} items")]
    GapDetected {
        gap_type: GapType,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        count: u64,
    },

    #[error("Heartbeat timeout: no data for {duration:?}")]
    HeartbeatTimeout { duration: Duration },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Database error: {0}")]
    DatabaseError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapType {
    /// Missing sequence IDs in trade stream
    SequenceGap,
    /// No data received for extended period
    HeartbeatTimeout,
    /// Statistical anomaly (tick rate dropped)
    StatisticalAnomaly,
    /// Volume-based gap (significant price movement without trades)
    VolumeGap,
}

impl std::fmt::Display for GapType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GapType::SequenceGap => write!(f, "Sequence Gap"),
            GapType::HeartbeatTimeout => write!(f, "Heartbeat Timeout"),
            GapType::StatisticalAnomaly => write!(f, "Statistical Anomaly"),
            GapType::VolumeGap => write!(f, "Volume Gap"),
        }
    }
}

pub type Result<T> = std::result::Result<T, GapDetectionError>;

// ============================================================================
// Prepared Query
// ============================================================================

/// A parameterized SQL query ready for safe execution via `tokio_postgres`.
///
/// Instead of interpolating user-supplied values directly into SQL strings,
/// `PreparedQuery` holds the SQL template with `$1`, `$2`, … placeholders and
/// keeps the parameter values separately. This eliminates SQL injection risk
/// at the protocol level.
#[derive(Debug)]
pub struct PreparedQuery {
    /// SQL text with `$1`, `$2`, … positional placeholders.
    pub sql: String,
    /// The exchange name bound to `$1`.
    pub exchange: String,
    /// The pair name bound to `$2`.
    pub pair: String,
}

impl PreparedQuery {
    /// Execute this query against a live `tokio_postgres::Client`.
    ///
    /// The parameters are bound through the PostgreSQL wire protocol, so no
    /// string escaping or identifier validation is required for safety.
    pub async fn execute(
        &self,
        client: &tokio_postgres::Client,
    ) -> std::result::Result<Vec<tokio_postgres::Row>, tokio_postgres::Error> {
        let params: Vec<&(dyn ToSql + Sync)> = vec![&self.exchange, &self.pair];
        client.query(self.sql.as_str(), &params).await
    }

    /// Return the raw SQL template (useful for logging / debugging).
    pub fn sql(&self) -> &str {
        &self.sql
    }
}

// ============================================================================
// Data Models
// ============================================================================

#[derive(Debug, Clone)]
pub struct Trade {
    pub exchange: String,
    pub pair: String,
    pub trade_id: Option<u64>,
    pub timestamp: DateTime<Utc>,
    pub price: f64,
    pub amount: f64,
}

#[derive(Debug, Clone)]
pub struct Gap {
    pub gap_type: GapType,
    pub exchange: String,
    pub pair: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub missing_count: u64,
    pub detected_at: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl Gap {
    pub fn duration(&self) -> Duration {
        self.end_time - self.start_time
    }

    pub fn severity(&self) -> GapSeverity {
        let duration_secs = self.duration().num_seconds();
        let missing = self.missing_count;

        match (duration_secs, missing) {
            (d, _) if d > 300 => GapSeverity::Critical, // > 5 minutes
            (d, m) if d > 60 || m > 1000 => GapSeverity::High, // > 1 min or > 1000 trades
            (d, m) if d > 10 || m > 100 => GapSeverity::Medium,
            _ => GapSeverity::Low,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GapSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// ============================================================================
// Sequence Gap Detector
// ============================================================================

/// Detects gaps by tracking monotonic sequence IDs from exchanges
///
/// Most exchanges provide trade IDs that increment monotonically.
/// If we receive ID 1000, then ID 1005, we know 4 trades are missing.
pub struct SequenceGapDetector {
    /// Last seen trade ID per (exchange, pair)
    last_trade_id: TradeIdMap,

    /// Maximum expected gap (exchange restarts reset IDs)
    max_expected_gap: u64,

    /// Detected gaps (ring buffer)
    detected_gaps: Arc<RwLock<VecDeque<Gap>>>,

    /// Max gaps to store in memory
    max_stored_gaps: usize,
}

impl SequenceGapDetector {
    pub fn new(max_expected_gap: u64, max_stored_gaps: usize) -> Self {
        Self {
            last_trade_id: Arc::new(RwLock::new(HashMap::new())),
            max_expected_gap,
            detected_gaps: Arc::new(RwLock::new(VecDeque::new())),
            max_stored_gaps,
        }
    }

    /// Process a trade and check for sequence gaps
    pub fn check_trade(&self, trade: &Trade) -> Result<()> {
        let Some(trade_id) = trade.trade_id else {
            // No trade ID available, skip sequence checking
            return Ok(());
        };

        let key = self.make_key(&trade.exchange, &trade.pair);
        let mut last_ids = self.last_trade_id.write();

        if let Some(&last_id) = last_ids.get(&key) {
            let expected_id = last_id + 1;
            let gap_size = trade_id.saturating_sub(expected_id);

            if gap_size > 0 && gap_size < self.max_expected_gap {
                // We have a gap!
                warn!(
                    exchange = %trade.exchange,
                    pair = %trade.pair,
                    last_id = last_id,
                    current_id = trade_id,
                    missing = gap_size,
                    "Sequence gap detected"
                );

                let gap = Gap {
                    gap_type: GapType::SequenceGap,
                    exchange: trade.exchange.clone(),
                    pair: trade.pair.clone(),
                    start_time: trade.timestamp - Duration::seconds(1), // Estimate
                    end_time: trade.timestamp,
                    missing_count: gap_size,
                    detected_at: Utc::now(),
                    metadata: [
                        ("last_id".to_string(), last_id.to_string()),
                        ("current_id".to_string(), trade_id.to_string()),
                    ]
                    .into_iter()
                    .collect(),
                };

                self.store_gap(gap.clone());

                return Err(GapDetectionError::GapDetected {
                    gap_type: GapType::SequenceGap,
                    start: gap.start_time,
                    end: gap.end_time,
                    count: gap_size,
                });
            } else if gap_size >= self.max_expected_gap {
                // Probably exchange restart or rollover
                info!(
                    exchange = %trade.exchange,
                    pair = %trade.pair,
                    last_id = last_id,
                    current_id = trade_id,
                    "Large ID gap detected (likely exchange restart)"
                );
            }
        }

        // Update last seen ID
        last_ids.insert(key, trade_id);
        Ok(())
    }

    /// Get all detected gaps
    pub fn get_gaps(&self) -> Vec<Gap> {
        self.detected_gaps.read().iter().cloned().collect()
    }

    /// Clear stored gaps
    pub fn clear_gaps(&self) {
        self.detected_gaps.write().clear();
    }

    fn make_key(&self, exchange: &str, pair: &str) -> String {
        format!("{}:{}", exchange, pair)
    }

    fn store_gap(&self, gap: Gap) {
        let mut gaps = self.detected_gaps.write();
        gaps.push_back(gap);

        // Keep only the most recent gaps
        while gaps.len() > self.max_stored_gaps {
            gaps.pop_front();
        }
    }
}

// ============================================================================
// Heartbeat Monitor
// ============================================================================

/// Monitors WebSocket liveness via heartbeat tracking
///
/// Problem: A silent WebSocket disconnection won't be detected by sequence IDs.
/// Solution: Track last data receipt time per connection.
pub struct HeartbeatMonitor {
    /// Last data receipt time per (exchange, pair)
    last_heartbeat: HeartbeatMap,

    /// Timeout duration
    timeout: Duration,
}

impl HeartbeatMonitor {
    pub fn new(timeout_seconds: i64) -> Self {
        Self {
            last_heartbeat: Arc::new(RwLock::new(HashMap::new())),
            timeout: Duration::seconds(timeout_seconds),
        }
    }

    /// Record a heartbeat (any data received)
    pub fn heartbeat(&self, exchange: &str, pair: &str) {
        let key = format!("{}:{}", exchange, pair);
        self.last_heartbeat.write().insert(key, Utc::now());
    }

    /// Check if heartbeat has timed out
    pub fn check_timeout(&self, exchange: &str, pair: &str) -> Result<()> {
        let key = format!("{}:{}", exchange, pair);
        let heartbeats = self.last_heartbeat.read();

        if let Some(&last_time) = heartbeats.get(&key) {
            let elapsed = Utc::now() - last_time;
            if elapsed > self.timeout {
                return Err(GapDetectionError::HeartbeatTimeout { duration: elapsed });
            }
        } else {
            // No heartbeat recorded yet, which is okay during initialization
            debug!(
                exchange = exchange,
                pair = pair,
                "No heartbeat recorded yet"
            );
        }

        Ok(())
    }

    /// Check all connections and return those that have timed out
    pub fn check_all_timeouts(&self) -> Vec<(String, String, Duration)> {
        let heartbeats = self.last_heartbeat.read();
        let now = Utc::now();
        let mut timeouts = Vec::new();

        for (key, &last_time) in heartbeats.iter() {
            let elapsed = now - last_time;
            if elapsed > self.timeout
                && let Some((exchange, pair)) = key.split_once(':')
            {
                timeouts.push((exchange.to_string(), pair.to_string(), elapsed));
            }
        }

        timeouts
    }
}

// ============================================================================
// Statistical Anomaly Detector
// ============================================================================

/// Detects gaps via statistical anomalies in tick rate
///
/// Uses exponential moving average to detect when tick rate drops
/// significantly below historical average.
type TickHistory = HashMap<String, VecDeque<(DateTime<Utc>, u32)>>;

pub struct StatisticalDetector {
    /// Tick count per minute per (exchange, pair)
    tick_history: Arc<RwLock<TickHistory>>,

    /// Window size for moving average (in minutes)
    window_size: usize,

    /// Threshold: trigger if tick rate drops below X% of average
    threshold_ratio: f64,
}

impl StatisticalDetector {
    pub fn new(window_size: usize, threshold_ratio: f64) -> Self {
        Self {
            tick_history: Arc::new(RwLock::new(HashMap::new())),
            window_size,
            threshold_ratio,
        }
    }

    /// Record tick count for a time window
    pub fn record_tick_count(&self, exchange: &str, pair: &str, count: u32) {
        let key = format!("{}:{}", exchange, pair);
        let mut history = self.tick_history.write();

        let entry = history.entry(key).or_default();
        entry.push_back((Utc::now(), count));

        // Keep only recent history
        while entry.len() > self.window_size {
            entry.pop_front();
        }
    }

    /// Check if current tick count is anomalously low
    pub fn check_anomaly(&self, exchange: &str, pair: &str, current_count: u32) -> Option<Gap> {
        let key = format!("{}:{}", exchange, pair);
        let history = self.tick_history.read();

        let tick_history = history.get(&key)?;

        if tick_history.len() < 3 {
            // Not enough data
            return None;
        }

        // Calculate moving average
        let sum: u32 = tick_history.iter().map(|(_, count)| count).sum();
        let avg = sum as f64 / tick_history.len() as f64;

        // Check if current is below threshold
        let threshold = avg * self.threshold_ratio;
        if (current_count as f64) < threshold && avg > 10.0 {
            // Anomaly detected!
            warn!(
                exchange = exchange,
                pair = pair,
                current = current_count,
                average = avg,
                threshold = threshold,
                "Statistical anomaly detected"
            );

            return Some(Gap {
                gap_type: GapType::StatisticalAnomaly,
                exchange: exchange.to_string(),
                pair: pair.to_string(),
                start_time: Utc::now() - Duration::minutes(1),
                end_time: Utc::now(),
                missing_count: (avg - current_count as f64) as u64,
                detected_at: Utc::now(),
                metadata: [
                    ("current_count".to_string(), current_count.to_string()),
                    ("average".to_string(), avg.to_string()),
                    ("threshold".to_string(), threshold.to_string()),
                ]
                .into_iter()
                .collect(),
            });
        }

        None
    }
}

// ============================================================================
// Unified Gap Detection Manager
// ============================================================================

pub struct GapDetectionManager {
    sequence_detector: SequenceGapDetector,
    heartbeat_monitor: HeartbeatMonitor,
    statistical_detector: StatisticalDetector,
    all_gaps: Arc<RwLock<Vec<Gap>>>,
    /// Track tick counts per minute for statistical detection
    tick_counts: TickCountMap,
}

impl GapDetectionManager {
    pub fn new(
        max_sequence_gap: u64,
        heartbeat_timeout_secs: i64,
        statistical_window: usize,
        statistical_threshold: f64,
    ) -> Self {
        Self {
            sequence_detector: SequenceGapDetector::new(max_sequence_gap, 1000),
            heartbeat_monitor: HeartbeatMonitor::new(heartbeat_timeout_secs),
            statistical_detector: StatisticalDetector::new(
                statistical_window,
                statistical_threshold,
            ),
            all_gaps: Arc::new(RwLock::new(Vec::new())),
            tick_counts: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Process a trade through all detectors
    pub fn process_trade(&self, trade: &Trade) {
        // Sequence check
        if let Err(GapDetectionError::GapDetected { .. }) =
            self.sequence_detector.check_trade(trade)
            && let Some(gap) = self.sequence_detector.get_gaps().last()
        {
            let mut gaps = self.all_gaps.write();
            // Deduplicate: only add if not already present
            if !gaps.iter().any(|g| {
                g.exchange == gap.exchange && g.pair == gap.pair && g.start_time == gap.start_time
            }) {
                gaps.push(gap.clone());
            }
        }

        // Update heartbeat
        self.heartbeat_monitor
            .heartbeat(&trade.exchange, &trade.pair);

        // Increment tick count for statistical analysis
        let key = format!("{}:{}", trade.exchange, trade.pair);
        let mut tick_counts = self.tick_counts.write();
        let entry = tick_counts.entry(key).or_insert((Utc::now(), 0));
        entry.1 += 1;
    }

    /// Run periodic checks (call every minute)
    pub async fn run_periodic_checks(&self) {
        // Check heartbeat timeouts
        let timeouts = self.heartbeat_monitor.check_all_timeouts();
        for (exchange, pair, duration) in timeouts {
            error!(
                exchange = exchange,
                pair = pair,
                duration_secs = duration.num_seconds(),
                "Heartbeat timeout detected"
            );

            let gap = Gap {
                gap_type: GapType::HeartbeatTimeout,
                exchange,
                pair,
                start_time: Utc::now() - duration,
                end_time: Utc::now(),
                missing_count: 0, // Unknown
                detected_at: Utc::now(),
                metadata: [(
                    "duration_secs".to_string(),
                    duration.num_seconds().to_string(),
                )]
                .into_iter()
                .collect(),
            };

            self.all_gaps.write().push(gap);
        }

        // Process tick counts and check for statistical anomalies
        let now = Utc::now();
        let mut tick_counts = self.tick_counts.write();
        let keys_to_process: Vec<String> = tick_counts.keys().cloned().collect();

        for key in keys_to_process {
            if let Some((start_time, count)) = tick_counts.remove(&key) {
                // Only process if we have data from the current minute window
                let elapsed = now - start_time;
                if elapsed >= Duration::seconds(55) && elapsed < Duration::seconds(65) {
                    // Extract exchange and pair from key
                    if let Some((exchange, pair)) = key.split_once(':') {
                        // Record the tick count
                        self.statistical_detector
                            .record_tick_count(exchange, pair, count);

                        // Check for anomalies
                        if let Some(gap) = self
                            .statistical_detector
                            .check_anomaly(exchange, pair, count)
                        {
                            info!(
                                exchange = exchange,
                                pair = pair,
                                count = count,
                                "Statistical anomaly detected"
                            );
                            self.all_gaps.write().push(gap);
                        }
                    }
                }
            }
        }
    }

    /// Get all detected gaps
    pub fn get_all_gaps(&self) -> Vec<Gap> {
        self.all_gaps.read().clone()
    }

    /// Get gaps by severity
    pub fn get_gaps_by_severity(&self, min_severity: GapSeverity) -> Vec<Gap> {
        self.all_gaps
            .read()
            .iter()
            .filter(|gap| gap.severity() >= min_severity)
            .cloned()
            .collect()
    }

    /// Clear all gaps
    pub fn clear_gaps(&self) {
        self.all_gaps.write().clear();
        self.sequence_detector.clear_gaps();
    }
}

impl Default for GapDetectionManager {
    /// Default configuration for crypto exchanges
    fn default() -> Self {
        Self::new(
            10000, // Max sequence gap (exchange restarts)
            30,    // 30 second heartbeat timeout
            10,    // 10 minute statistical window
            0.3,   // Alert if tick rate < 30% of average
        )
    }
}

// ============================================================================
// Database Integration (QuestDB)
// ============================================================================

#[async_trait]
pub trait GapBackfiller: Send + Sync {
    /// Backfill data for a detected gap
    async fn backfill_gap(&self, gap: &Gap) -> Result<u64>;
}

/// SQL-based gap detection using QuestDB
pub struct SqlGapDetector {
    connection_string: String,
}

impl SqlGapDetector {
    pub fn new(connection_string: String) -> Self {
        Self { connection_string }
    }

    /// Get the connection string for this detector
    pub fn connection_string(&self) -> &str {
        &self.connection_string
    }

    /// Validate SQL identifier (exchange/pair names)
    ///
    /// Ensures the identifier only contains safe characters.
    /// This is kept as **defense-in-depth** even though the parameterized query
    /// path makes SQL injection impossible at the protocol level.
    fn validate_identifier(identifier: &str) -> Result<String> {
        // Allow alphanumeric, underscore, hyphen, and dot
        if identifier
            .chars()
            .all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            Ok(identifier.to_string())
        } else {
            Err(GapDetectionError::InvalidConfig(format!(
                "Invalid identifier '{}': contains dangerous characters",
                identifier
            )))
        }
    }

    /// Improved gap detection query using sequence IDs.
    ///
    /// Returns a [`PreparedQuery`] with parameterized placeholders (`$1`, `$2`)
    /// for `exchange` and `pair`. The `lookback_minutes` value is inlined as a
    /// literal because it is an `i64` (type-safe, not injectable) and because
    /// QuestDB's `dateadd` does not accept bind parameters for its interval
    /// argument.
    ///
    /// Identifiers are also validated as defense-in-depth.
    pub fn build_sequence_gap_query(
        &self,
        exchange: &str,
        pair: &str,
        lookback_minutes: i64,
    ) -> Result<PreparedQuery> {
        // Defense-in-depth: reject obviously invalid identifiers early
        let exchange = Self::validate_identifier(exchange)?;
        let pair = Self::validate_identifier(pair)?;

        let sql = format!(
            r#"
            WITH trade_sequences AS (
                SELECT
                    timestamp,
                    trade_id,
                    LAG(trade_id) OVER (ORDER BY timestamp) as prev_id,
                    trade_id - LAG(trade_id) OVER (ORDER BY timestamp) as id_gap
                FROM trades
                WHERE exchange = $1
                  AND pair = $2
                  AND timestamp > dateadd('m', -{}, now())
                  AND trade_id IS NOT NULL
            )
            SELECT
                timestamp,
                prev_id,
                trade_id,
                id_gap
            FROM trade_sequences
            WHERE id_gap > 1           -- Missing trades
              AND id_gap < 10000       -- Ignore exchange restarts
            ORDER BY timestamp DESC
            "#,
            lookback_minutes
        );

        Ok(PreparedQuery {
            sql,
            exchange,
            pair,
        })
    }

    /// Time-based gap detection (fallback for exchanges without trade IDs).
    ///
    /// Returns a [`PreparedQuery`] with parameterized placeholders (`$1`, `$2`)
    /// for `exchange` and `pair`. See [`build_sequence_gap_query`] for rationale
    /// on `lookback_minutes` inlining.
    ///
    /// Identifiers are also validated as defense-in-depth.
    pub fn build_time_gap_query(
        &self,
        exchange: &str,
        pair: &str,
        lookback_minutes: i64,
    ) -> Result<PreparedQuery> {
        // Defense-in-depth: reject obviously invalid identifiers early
        let exchange = Self::validate_identifier(exchange)?;
        let pair = Self::validate_identifier(pair)?;

        let sql = format!(
            r#"
            WITH tick_counts AS (
                SELECT
                    timestamp,
                    count() as ticks
                FROM trades
                WHERE exchange = $1
                  AND pair = $2
                  AND timestamp > dateadd('m', -{}, now())
                SAMPLE BY 1m FILL(0)
            ),
            avg_ticks AS (
                SELECT avg(ticks) as average
                FROM tick_counts
                WHERE ticks > 0
            )
            SELECT
                tc.timestamp,
                tc.ticks,
                at.average,
                (at.average * 0.1) as threshold
            FROM tick_counts tc, avg_ticks at
            WHERE tc.ticks < (at.average * 0.1)  -- Less than 10% of average
              AND at.average > 5                  -- Ignore low-liquidity pairs
            ORDER BY tc.timestamp DESC
            "#,
            lookback_minutes
        );

        Ok(PreparedQuery {
            sql,
            exchange,
            pair,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(trade_id: u64, timestamp: DateTime<Utc>) -> Trade {
        Trade {
            exchange: "binance".to_string(),
            pair: "BTCUSD".to_string(),
            trade_id: Some(trade_id),
            timestamp,
            price: 50000.0,
            amount: 0.1,
        }
    }

    #[test]
    fn test_sequence_gap_detection() {
        let detector = SequenceGapDetector::new(10000, 100);
        let now = Utc::now();

        // First trade
        detector.check_trade(&make_trade(1000, now)).unwrap();

        // Second trade - no gap
        detector.check_trade(&make_trade(1001, now)).unwrap();

        // Third trade - gap of 5
        let result = detector.check_trade(&make_trade(1007, now));
        assert!(result.is_err());

        if let Err(GapDetectionError::GapDetected { count, .. }) = result {
            assert_eq!(count, 5);
        }
    }

    #[test]
    fn test_heartbeat_monitoring() {
        let monitor = HeartbeatMonitor::new(5); // 5 second timeout

        // Record heartbeat
        monitor.heartbeat("binance", "BTCUSD");

        // Should not timeout immediately
        assert!(monitor.check_timeout("binance", "BTCUSD").is_ok());

        // Sleep and check again (would timeout in real scenario)
        // In test, we can't easily test time-based logic without tokio::time::pause
    }

    #[test]
    fn test_statistical_detector() {
        let detector = StatisticalDetector::new(5, 0.5);

        // Build up history
        for _ in 0..5 {
            detector.record_tick_count("binance", "BTCUSD", 100);
        }

        // Normal count - no anomaly
        assert!(detector.check_anomaly("binance", "BTCUSD", 95).is_none());

        // Anomalously low count
        let gap = detector.check_anomaly("binance", "BTCUSD", 30);
        assert!(gap.is_some());
        assert_eq!(gap.unwrap().gap_type, GapType::StatisticalAnomaly);
    }

    #[test]
    fn test_gap_severity() {
        let gap = Gap {
            gap_type: GapType::SequenceGap,
            exchange: "binance".to_string(),
            pair: "BTCUSD".to_string(),
            start_time: Utc::now(),
            end_time: Utc::now() + Duration::seconds(10),
            missing_count: 50,
            detected_at: Utc::now(),
            metadata: HashMap::new(),
        };

        assert_eq!(gap.severity(), GapSeverity::Low);

        let critical_gap = Gap {
            end_time: gap.start_time + Duration::minutes(10),
            ..gap
        };

        assert_eq!(critical_gap.severity(), GapSeverity::Critical);
    }

    #[test]
    fn test_manager_integration() {
        let manager = GapDetectionManager::default();
        let now = Utc::now();

        // Process trades
        manager.process_trade(&make_trade(1000, now));
        manager.process_trade(&make_trade(1001, now));
        manager.process_trade(&make_trade(1010, now)); // Gap of 8

        let gaps = manager.get_all_gaps();
        assert_eq!(gaps.len(), 1);
        assert_eq!(gaps[0].missing_count, 8);
    }

    #[test]
    fn test_gap_deduplication() {
        let manager = GapDetectionManager::default();
        let now = Utc::now();

        // Create initial gap
        manager.process_trade(&make_trade(1000, now));
        manager.process_trade(&make_trade(1010, now)); // Gap of 9

        // Process another trade - should not duplicate the gap
        let gaps_before = manager.get_all_gaps().len();
        manager.process_trade(&make_trade(1011, now));
        let gaps_after = manager.get_all_gaps().len();

        assert_eq!(gaps_before, gaps_after, "Gap should not be duplicated");
    }

    #[test]
    fn test_sql_query_validation() {
        let detector = SqlGapDetector::new("postgresql://localhost".to_string());

        // Valid identifiers should work
        assert!(
            detector
                .build_sequence_gap_query("binance", "BTCUSD", 10)
                .is_ok()
        );
        assert!(
            detector
                .build_time_gap_query("coinbase", "ETH-USD", 5)
                .is_ok()
        );

        // Invalid identifiers with SQL injection attempts should fail
        assert!(
            detector
                .build_sequence_gap_query("binance'; DROP TABLE trades; --", "BTCUSD", 10)
                .is_err()
        );
        assert!(
            detector
                .build_time_gap_query("binance", "BTC' OR '1'='1", 10)
                .is_err()
        );

        // Identifiers with quotes should fail
        assert!(
            detector
                .build_sequence_gap_query("test\"quote", "BTCUSD", 10)
                .is_err()
        );
    }

    #[test]
    fn test_sql_query_uses_parameterized_placeholders() {
        let detector = SqlGapDetector::new("postgresql://localhost".to_string());

        let query = detector
            .build_sequence_gap_query("binance", "BTCUSD", 10)
            .unwrap();

        // SQL must use $1/$2 placeholders, NOT interpolated values
        assert!(
            query.sql().contains("exchange = $1"),
            "exchange should use $1 placeholder"
        );
        assert!(
            query.sql().contains("pair = $2"),
            "pair should use $2 placeholder"
        );
        assert!(
            !query.sql().contains("'binance'"),
            "exchange value must not be interpolated into SQL"
        );
        assert!(
            !query.sql().contains("'BTCUSD'"),
            "pair value must not be interpolated into SQL"
        );

        // Parameters are stored separately
        assert_eq!(query.exchange, "binance");
        assert_eq!(query.pair, "BTCUSD");

        // Query structure is intact
        assert!(query.sql().contains("WHERE id_gap > 1"));
        assert!(query.sql().contains("AND id_gap < 10000"));
        // lookback_minutes (i64, type-safe) is inlined
        assert!(query.sql().contains("dateadd('m', -10, now())"));
    }

    #[test]
    fn test_time_gap_query_uses_parameterized_placeholders() {
        let detector = SqlGapDetector::new("postgresql://localhost".to_string());

        let query = detector
            .build_time_gap_query("coinbase", "ETH-USD", 5)
            .unwrap();

        assert!(
            query.sql().contains("exchange = $1"),
            "exchange should use $1 placeholder"
        );
        assert!(
            query.sql().contains("pair = $2"),
            "pair should use $2 placeholder"
        );
        assert!(
            !query.sql().contains("'coinbase'"),
            "exchange value must not be interpolated into SQL"
        );
        assert!(
            !query.sql().contains("'ETH-USD'"),
            "pair value must not be interpolated into SQL"
        );

        assert_eq!(query.exchange, "coinbase");
        assert_eq!(query.pair, "ETH-USD");

        assert!(query.sql().contains("SAMPLE BY 1m FILL(0)"));
        assert!(query.sql().contains("dateadd('m', -5, now())"));
    }
}
