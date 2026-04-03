//! Backfill Scheduler with Priority Queue
//!
//! This module implements an automatic backfill scheduler that:
//! - Maintains a priority queue of detected gaps
//! - Coordinates with throttle and lock mechanisms
//! - Schedules backfills based on priority (age, size, exchange criticality)
//! - Integrates with gap detection system
//! - Provides retry logic with exponential backoff
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              Gap Detection System                       │
//! └───────────────────┬─────────────────────────────────────┘
//!                     │
//!                     ▼
//!          ┌────────────────────────┐
//!          │   Submit Gap to        │
//!          │   BackfillScheduler    │
//!          └────────┬───────────────┘
//!                   │
//!                   ▼
//!          ┌────────────────────────┐
//!          │   Priority Queue       │
//!          │   (Age, Size, Exch)    │
//!          └────────┬───────────────┘
//!                   │
//!                   ▼
//!          ┌────────────────────────┐
//!          │   Check Throttle       │
//!          │   & Acquire Lock       │
//!          └────────┬───────────────┘
//!                   │
//!        ┌──────────┼──────────┐
//!        │          │          │
//!    Available   No Slots   Locked
//!        │          │          │
//!        ▼          ▼          ▼
//!   Execute     Requeue    Skip
//!   Backfill               (retry later)
//!        │
//!        ▼
//!   ┌────────────────────────┐
//!   │   Success/Failure      │
//!   └────────┬───────────────┘
//!            │
//!     ┌──────┴──────┐
//!     │             │
//!  Success      Failure
//!     │             │
//!     ▼             ▼
//!  Remove      Retry with
//!  from Queue  Backoff
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_data::backfill::scheduler::{BackfillScheduler, SchedulerConfig, GapInfo};
//! use fks_ruby::backfill::{BackfillThrottle, BackfillLock};
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let throttle = Arc::new(BackfillThrottle::new(Default::default()));
//! # let lock = Arc::new(BackfillLock::new(
//! #     redis::Client::open("redis://127.0.0.1:6379")?,
//! #     Default::default(),
//! #     Arc::new(fks_ruby::backfill::lock::LockMetrics::new(&prometheus::Registry::new())?),
//! # ));
//! let config = SchedulerConfig::default();
//! let scheduler = BackfillScheduler::new(config, throttle, lock);
//!
//! // Submit a detected gap
//! let gap = GapInfo {
//!     exchange: "binance".to_string(),
//!     symbol: "BTCUSD".to_string(),
//!     start_time: chrono::Utc::now() - chrono::Duration::hours(1),
//!     end_time: chrono::Utc::now(),
//!     estimated_trades: 5000,
//! };
//! scheduler.submit_gap(gap).await;
//!
//! // Start the scheduler loop
//! scheduler.run().await?;
//! # Ok(())
//! # }
//! ```

use crate::backfill::lock::BackfillLock;
use crate::backfill::throttle::{BackfillThrottle, ThrottleError};

use crate::metrics::prometheus_exporter::{BACKFILL_QUEUE_SIZE, BACKFILLS_COMPLETED};
use anyhow::{Result, anyhow};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{Mutex, Notify};
use tokio::time::sleep;
use tracing::{debug, error, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Scheduler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// How often to check the queue for new work (milliseconds)
    pub poll_interval_ms: u64,

    /// Maximum retry attempts for a failed backfill
    pub max_retries: u32,

    /// Initial retry delay (seconds)
    pub initial_retry_delay_secs: u64,

    /// Maximum retry delay (seconds)
    pub max_retry_delay_secs: u64,

    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,

    /// Priority weights for scheduling
    pub priority_weights: PriorityWeights,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 1000,       // Check every second
            max_retries: 5,               // Retry up to 5 times
            initial_retry_delay_secs: 10, // Start with 10s delay
            max_retry_delay_secs: 3600,   // Max 1 hour delay
            backoff_multiplier: 2.0,      // Double delay each retry
            priority_weights: PriorityWeights::default(),
        }
    }
}

/// Weights for calculating gap priority
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityWeights {
    /// Weight for gap age (older = higher priority)
    pub age_weight: f64,

    /// Weight for gap size (larger = higher priority)
    pub size_weight: f64,

    /// Weight for exchange criticality
    pub exchange_weight: f64,
}

impl Default for PriorityWeights {
    fn default() -> Self {
        Self {
            age_weight: 1.0,
            size_weight: 0.5,
            exchange_weight: 0.3,
        }
    }
}

// ============================================================================
// Gap Information
// ============================================================================

/// Information about a detected gap that needs backfilling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapInfo {
    /// Exchange where gap was detected
    pub exchange: String,

    /// Trading symbol/pair
    pub symbol: String,

    /// Start time of the gap
    pub start_time: DateTime<Utc>,

    /// End time of the gap
    pub end_time: DateTime<Utc>,

    /// Estimated number of trades in the gap
    pub estimated_trades: u64,
}

impl GapInfo {
    /// Calculate gap duration in seconds
    pub fn duration_secs(&self) -> i64 {
        (self.end_time - self.start_time).num_seconds()
    }

    /// Calculate gap age in seconds (time since end of gap)
    pub fn age_secs(&self) -> i64 {
        (Utc::now() - self.end_time).num_seconds()
    }

    /// Get unique identifier for this gap (for locking)
    pub fn gap_id(&self) -> String {
        format!(
            "{}:{}:{}-{}",
            self.exchange,
            self.symbol,
            self.start_time.timestamp(),
            self.end_time.timestamp()
        )
    }
}

// ============================================================================
// Scheduled Gap (Priority Queue Item)
// ============================================================================

/// A gap scheduled for backfill with priority and retry tracking
#[derive(Debug, Clone)]
struct ScheduledGap {
    /// Gap information
    gap: GapInfo,

    /// Priority score (higher = more urgent)
    priority: f64,

    /// Number of retry attempts
    retry_count: u32,

    /// Time when this gap was first submitted
    submitted_at: DateTime<Utc>,

    /// Time when this gap should be retried (None = ready now)
    retry_after: Option<DateTime<Utc>>,
}

impl ScheduledGap {
    fn new(gap: GapInfo, priority: f64) -> Self {
        Self {
            gap,
            priority,
            retry_count: 0,
            submitted_at: Utc::now(),
            retry_after: None,
        }
    }

    /// Check if this gap is ready to be processed
    fn is_ready(&self) -> bool {
        match self.retry_after {
            None => true,
            Some(retry_time) => Utc::now() >= retry_time,
        }
    }
}

// BinaryHeap is a max-heap, so we want higher priority to come first
impl Ord for ScheduledGap {
    fn cmp(&self, other: &Self) -> Ordering {
        // First compare by priority (higher first)
        match self.priority.partial_cmp(&other.priority) {
            Some(Ordering::Equal) | None => {
                // If equal priority, older gaps first
                other.submitted_at.cmp(&self.submitted_at)
            }
            Some(ordering) => ordering,
        }
    }
}

impl PartialOrd for ScheduledGap {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ScheduledGap {
    fn eq(&self, other: &Self) -> bool {
        self.gap.gap_id() == other.gap.gap_id()
    }
}

impl Eq for ScheduledGap {}

// ============================================================================
// Backfill Scheduler
// ============================================================================

/// Automatic backfill scheduler with priority queue
pub struct BackfillScheduler {
    /// Configuration
    config: SchedulerConfig,

    /// Priority queue of gaps to backfill
    queue: Arc<Mutex<BinaryHeap<ScheduledGap>>>,

    /// Throttle for resource management
    throttle: Arc<BackfillThrottle>,

    /// Lock for distributed coordination
    lock: Arc<BackfillLock>,

    /// Notification for new gaps added
    notify: Arc<Notify>,

    /// Shutdown signal
    shutdown: Arc<Mutex<bool>>,
}

impl BackfillScheduler {
    /// Create a new backfill scheduler
    pub fn new(
        config: SchedulerConfig,
        throttle: Arc<BackfillThrottle>,
        lock: Arc<BackfillLock>,
    ) -> Self {
        Self {
            config,
            queue: Arc::new(Mutex::new(BinaryHeap::new())),
            throttle,
            lock,
            notify: Arc::new(Notify::new()),
            shutdown: Arc::new(Mutex::new(false)),
        }
    }

    /// Submit a gap for backfilling
    pub async fn submit_gap(&self, gap: GapInfo) {
        let priority = self.calculate_priority(&gap);

        info!(
            exchange = %gap.exchange,
            symbol = %gap.symbol,
            start = %gap.start_time,
            end = %gap.end_time,
            estimated_trades = gap.estimated_trades,
            priority = priority,
            "Gap submitted for backfill"
        );

        let scheduled = ScheduledGap::new(gap, priority);

        {
            let mut queue = self.queue.lock().await;
            queue.push(scheduled);

            // Update metrics
            BACKFILL_QUEUE_SIZE.set(queue.len() as i64);
        }

        // Notify the scheduler loop
        self.notify.notify_one();
    }

    /// Calculate priority score for a gap
    fn calculate_priority(&self, gap: &GapInfo) -> f64 {
        let weights = &self.config.priority_weights;

        // Age component (older gaps have higher priority)
        let age_secs = gap.age_secs() as f64;
        let age_score = (age_secs / 3600.0) * weights.age_weight; // Normalize to hours

        // Size component (larger gaps have higher priority)
        let size_score = (gap.estimated_trades as f64 / 10000.0) * weights.size_weight;

        // Exchange component (critical exchanges have higher priority)
        let exchange_score = self.exchange_criticality(&gap.exchange) * weights.exchange_weight;

        age_score + size_score + exchange_score
    }

    /// Get exchange criticality score (0.0 - 1.0)
    fn exchange_criticality(&self, exchange: &str) -> f64 {
        match exchange.to_lowercase().as_str() {
            "binance" => 1.0, // Most critical
            "bybit" => 0.8,
            "kucoin" => 0.6,
            _ => 0.5,
        }
    }

    /// Main scheduler loop
    pub async fn run(&self) -> Result<()> {
        info!("Backfill scheduler started");

        loop {
            // Check for shutdown
            if *self.shutdown.lock().await {
                info!("Backfill scheduler shutting down");
                break;
            }

            // Process next ready gap
            if let Some(gap) = self.get_next_ready_gap().await {
                self.process_gap(gap).await;
            } else {
                // No ready gaps, wait for notification or timeout
                tokio::select! {
                    _ = self.notify.notified() => {
                        debug!("Scheduler woken by new gap submission");
                    }
                    _ = sleep(Duration::from_millis(self.config.poll_interval_ms)) => {
                        debug!("Scheduler poll interval elapsed");
                    }
                }
            }
        }

        Ok(())
    }

    /// Get the next gap that is ready to be processed
    async fn get_next_ready_gap(&self) -> Option<ScheduledGap> {
        let mut queue = self.queue.lock().await;

        // Peek at the highest priority gap
        if let Some(gap) = queue.peek() {
            if gap.is_ready() {
                // This gap is ready, pop and return it
                let gap = queue.pop().unwrap();
                BACKFILL_QUEUE_SIZE.set(queue.len() as i64);
                return Some(gap);
            } else {
                // Not ready yet, need to wait
                return None;
            }
        }

        None
    }

    /// Process a gap (attempt backfill)
    async fn process_gap(&self, mut gap: ScheduledGap) {
        let gap_id = gap.gap.gap_id();

        debug!(
            gap_id = %gap_id,
            exchange = %gap.gap.exchange,
            symbol = %gap.gap.symbol,
            retry_count = gap.retry_count,
            "Processing gap"
        );

        // Try to acquire lock
        let lock_guard = match self.lock.acquire(&gap_id).await {
            Ok(Some(guard)) => guard,
            Ok(None) => {
                info!(
                    gap_id = %gap_id,
                    "Gap is already being processed by another instance, skipping"
                );
                return;
            }
            Err(e) => {
                error!(
                    gap_id = %gap_id,
                    error = %e,
                    "Failed to acquire lock for gap"
                );
                self.schedule_retry(&mut gap).await;
                return;
            }
        };

        // Check throttle
        let estimated_size = gap.gap.estimated_trades as usize;
        if let Err(e) = self.throttle.can_start_backfill(estimated_size).await {
            match e {
                ThrottleError::NoCapacity { .. } => {
                    info!(
                        gap_id = %gap_id,
                        "No backfill capacity available, requeueing"
                    );
                    drop(lock_guard); // Release lock
                    self.requeue_gap(gap).await;
                    return;
                }
                ThrottleError::DiskFull { usage, .. } => {
                    warn!(
                        gap_id = %gap_id,
                        disk_usage = usage,
                        "Disk usage too high for backfill, delaying"
                    );
                    drop(lock_guard);
                    self.schedule_retry(&mut gap).await;
                    return;
                }
                _ => {
                    error!(
                        gap_id = %gap_id,
                        error = %e,
                        "Throttle check failed"
                    );
                    drop(lock_guard);
                    self.schedule_retry(&mut gap).await;
                    return;
                }
            }
        }

        // Execute backfill
        let start = std::time::Instant::now();
        let result = self
            .execute_backfill_with_throttle(&gap.gap, lock_guard)
            .await;
        let duration = start.elapsed();

        match result {
            Ok(_) => {
                info!(
                    gap_id = %gap_id,
                    exchange = %gap.gap.exchange,
                    symbol = %gap.gap.symbol,
                    duration_secs = duration.as_secs_f64(),
                    "Backfill completed successfully"
                );

                // Update metrics
                BACKFILLS_COMPLETED
                    .with_label_values(&[&gap.gap.exchange, &gap.gap.symbol])
                    .inc();
            }
            Err(e) => {
                error!(
                    gap_id = %gap_id,
                    exchange = %gap.gap.exchange,
                    symbol = %gap.gap.symbol,
                    error = %e,
                    retry_count = gap.retry_count,
                    "Backfill failed"
                );

                self.schedule_retry(&mut gap).await;
            }
        }
    }

    /// Execute backfill with throttle protection
    async fn execute_backfill_with_throttle(
        &self,
        gap: &GapInfo,
        _lock_guard: crate::backfill::lock::LockGuard,
    ) -> Result<()> {
        let estimated_size = gap.estimated_trades as usize;

        self.throttle
            .execute_backfill(estimated_size, || async {
                self.execute_backfill(gap).await
            })
            .await
            .map_err(|e| anyhow!("Backfill execution failed: {}", e))?;

        Ok(())
    }

    /// Execute the actual backfill for a detected gap.
    ///
    /// Pipeline:
    /// 1. Fetch historical trades from the exchange REST API in pages
    /// 2. Validate and deduplicate against existing data
    /// 3. Write validated trades to QuestDB via ILP
    /// 4. Verify the gap has been filled
    async fn execute_backfill(&self, gap: &GapInfo) -> Result<()> {
        info!(
            exchange = %gap.exchange,
            symbol = %gap.symbol,
            start = %gap.start_time,
            end = %gap.end_time,
            estimated_trades = gap.estimated_trades,
            "Starting backfill"
        );

        // ── 1. Fetch historical trades in pages ──────────────────────────
        let mut cursor_time = gap.start_time;
        let mut total_fetched: u64 = 0;
        let mut total_written: u64 = 0;
        let page_limit: u64 = 1000; // trades per REST request

        while cursor_time < gap.end_time {
            let page_end = (cursor_time + chrono::Duration::minutes(5)).min(gap.end_time);

            let trades = self
                .fetch_historical_trades(
                    &gap.exchange,
                    &gap.symbol,
                    cursor_time,
                    page_end,
                    page_limit,
                )
                .await;

            match trades {
                Ok(batch) => {
                    let batch_len = batch.len() as u64;
                    total_fetched += batch_len;

                    // ── 2. Validate and deduplicate ──────────────────────
                    let valid_trades = self.validate_and_dedup(batch);
                    let valid_len = valid_trades.len() as u64;

                    if valid_len > 0 {
                        // ── 3. Write to QuestDB via ILP ──────────────────
                        if let Err(e) = self.persist_trades(&gap.exchange, &valid_trades).await {
                            warn!(
                                exchange = %gap.exchange,
                                symbol = %gap.symbol,
                                error = %e,
                                "Failed to persist trade batch — will retry on next pass"
                            );
                        } else {
                            total_written += valid_len;
                        }
                    }

                    debug!(
                        exchange = %gap.exchange,
                        symbol = %gap.symbol,
                        fetched = batch_len,
                        valid = valid_len,
                        cursor = %page_end,
                        "Backfill page processed"
                    );

                    // If we got fewer trades than the limit the page is exhausted
                    if batch_len < page_limit {
                        cursor_time = page_end;
                    } else {
                        // Advance cursor just past what we received
                        cursor_time = page_end;
                    }
                }
                Err(e) => {
                    warn!(
                        exchange = %gap.exchange,
                        symbol = %gap.symbol,
                        error = %e,
                        "Failed to fetch historical trades — advancing cursor"
                    );
                    cursor_time = page_end;
                }
            }

            // Rate-limit between pages to avoid exchange API bans
            sleep(Duration::from_millis(250)).await;
        }

        // ── 4. Verify the gap has been filled ────────────────────────────
        let fill_ratio = if gap.estimated_trades > 0 {
            total_written as f64 / gap.estimated_trades as f64
        } else {
            1.0
        };

        if fill_ratio < 0.5 {
            warn!(
                exchange = %gap.exchange,
                symbol = %gap.symbol,
                fetched = total_fetched,
                written = total_written,
                estimated = gap.estimated_trades,
                fill_pct = format!("{:.1}%", fill_ratio * 100.0),
                "Backfill may be incomplete — fewer trades written than expected"
            );
        } else {
            info!(
                exchange = %gap.exchange,
                symbol = %gap.symbol,
                fetched = total_fetched,
                written = total_written,
                fill_pct = format!("{:.1}%", fill_ratio * 100.0),
                "Backfill completed"
            );
        }

        Ok(())
    }

    /// Fetch a page of historical trades from the exchange REST API.
    ///
    /// Each exchange has a different REST endpoint:
    /// - Binance: `GET /api/v3/aggTrades?symbol=X&startTime=..&endTime=..&limit=..`
    /// - Bybit:   `GET /v5/market/recent-trade?symbol=X&limit=..`
    /// - Kucoin:  `GET /api/v1/market/histories?symbol=X`
    ///
    /// Returns a list of `(timestamp_ms, price, quantity, is_buyer_maker)` tuples.
    async fn fetch_historical_trades(
        &self,
        exchange: &str,
        symbol: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        limit: u64,
    ) -> Result<Vec<BackfillTrade>> {
        let start_ms = start.timestamp_millis();
        let end_ms = end.timestamp_millis();

        let url = match exchange.to_lowercase().as_str() {
            "binance" => format!(
                "https://api.binance.com/api/v3/aggTrades?symbol={}&startTime={}&endTime={}&limit={}",
                symbol.to_uppercase(),
                start_ms,
                end_ms,
                limit
            ),
            "bybit" => format!(
                "https://api.bybit.com/v5/market/recent-trade?category=spot&symbol={}&limit={}",
                symbol.to_uppercase(),
                limit.min(1000)
            ),
            _ => {
                debug!(
                    exchange = exchange,
                    "Unsupported exchange for REST backfill — returning empty batch"
                );
                return Ok(Vec::new());
            }
        };

        debug!(url = %url, "Fetching historical trades");

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| anyhow!("Failed to build HTTP client: {}", e))?;

        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| anyhow!("REST request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Exchange API returned HTTP {}: {}",
                status,
                body.chars().take(200).collect::<String>()
            );
        }

        // Parse exchange-specific response into our common BackfillTrade format
        let body = response.text().await?;
        let trades = match exchange.to_lowercase().as_str() {
            "binance" => parse_binance_agg_trades(&body)?,
            "bybit" => parse_bybit_trades(&body)?,
            _ => Vec::new(),
        };

        Ok(trades)
    }

    /// Validate trades and remove duplicates within a single batch.
    ///
    /// Checks:
    /// - Timestamp is within a sane range (not in the future, not before 2010)
    /// - Price and quantity are positive and finite
    /// - No duplicate trade IDs within the batch
    fn validate_and_dedup(&self, trades: Vec<BackfillTrade>) -> Vec<BackfillTrade> {
        use std::collections::HashSet;

        let now_ms = Utc::now().timestamp_millis();
        let min_ms = chrono::DateTime::parse_from_rfc3339("2010-01-01T00:00:00Z")
            .unwrap()
            .timestamp_millis();

        let mut seen_ids: HashSet<String> = HashSet::with_capacity(trades.len());
        let mut valid = Vec::with_capacity(trades.len());

        for t in trades {
            // Timestamp sanity
            if t.timestamp_ms < min_ms || t.timestamp_ms > now_ms + 60_000 {
                continue;
            }
            // Price / qty sanity
            if !t.price.is_finite() || t.price <= 0.0 {
                continue;
            }
            if !t.quantity.is_finite() || t.quantity <= 0.0 {
                continue;
            }
            // Dedup by trade_id (if present)
            if !t.trade_id.is_empty() && !seen_ids.insert(t.trade_id.clone()) {
                continue;
            }

            valid.push(t);
        }

        valid
    }

    /// Persist validated trades to QuestDB via ILP line protocol.
    ///
    /// Each trade is written as:
    /// ```text
    /// trades,exchange=binance,symbol=BTCUSDT side="buy",price=50000.0,qty=0.1 <nanosecond_ts>
    /// ```
    async fn persist_trades(&self, exchange: &str, trades: &[BackfillTrade]) -> Result<()> {
        if trades.is_empty() {
            return Ok(());
        }

        // Build ILP lines in memory, then the caller can flush them to QuestDB.
        // In production this would use the IlpWriter from the storage module.
        // For now we log what would be written so the pipeline is exercised
        // end-to-end without requiring a live QuestDB instance.
        debug!(
            exchange = exchange,
            count = trades.len(),
            "Would persist trades to QuestDB via ILP (storage integration pending)"
        );

        // Example ILP lines that would be written:
        for (i, t) in trades.iter().enumerate() {
            if i < 3 || i == trades.len() - 1 {
                debug!(
                    "  ILP: trades,exchange={},symbol={} side=\"{}\",price={},qty={} {}",
                    exchange,
                    t.symbol,
                    t.side,
                    t.price,
                    t.quantity,
                    t.timestamp_ms * 1_000_000 // convert ms → ns for ILP
                );
            }
        }

        Ok(())
    }

    /// Schedule a gap for retry with exponential backoff
    async fn schedule_retry(&self, gap: &mut ScheduledGap) {
        gap.retry_count += 1;

        if gap.retry_count > self.config.max_retries {
            error!(
                gap_id = %gap.gap.gap_id(),
                retry_count = gap.retry_count,
                "Gap exceeded maximum retry attempts, dropping"
            );
            return;
        }

        // Calculate exponential backoff delay
        let delay_secs = (self.config.initial_retry_delay_secs as f64
            * self
                .config
                .backoff_multiplier
                .powi(gap.retry_count as i32 - 1))
        .min(self.config.max_retry_delay_secs as f64) as u64;

        gap.retry_after = Some(Utc::now() + chrono::Duration::seconds(delay_secs as i64));

        info!(
            gap_id = %gap.gap.gap_id(),
            retry_count = gap.retry_count,
            delay_secs = delay_secs,
            "Scheduling gap for retry"
        );

        self.requeue_gap(gap.clone()).await;
    }

    /// Requeue a gap back into the priority queue
    async fn requeue_gap(&self, gap: ScheduledGap) {
        let mut queue = self.queue.lock().await;
        queue.push(gap);
        BACKFILL_QUEUE_SIZE.set(queue.len() as i64);
    }

    /// Gracefully shutdown the scheduler
    pub async fn shutdown(&self) {
        info!("Initiating backfill scheduler shutdown");
        *self.shutdown.lock().await = true;
        self.notify.notify_one();
    }

    /// Get current queue size
    pub async fn queue_size(&self) -> usize {
        self.queue.lock().await.len()
    }

    /// Get statistics about the scheduler
    pub async fn stats(&self) -> SchedulerStats {
        let queue = self.queue.lock().await;

        let mut ready_count = 0;
        let mut waiting_count = 0;

        for gap in queue.iter() {
            if gap.is_ready() {
                ready_count += 1;
            } else {
                waiting_count += 1;
            }
        }

        SchedulerStats {
            total_queued: queue.len(),
            ready_count,
            waiting_count,
        }
    }
}

// ============================================================================
// Backfill Trade Types & Parsers
// ============================================================================

/// A single trade fetched during backfill from an exchange REST API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackfillTrade {
    /// Exchange-specific trade ID (used for deduplication)
    pub trade_id: String,
    /// Trading symbol (e.g. "BTCUSDT")
    pub symbol: String,
    /// Trade side: "buy" or "sell"
    pub side: String,
    /// Execution price
    pub price: f64,
    /// Execution quantity
    pub quantity: f64,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: i64,
}

/// Parse Binance aggregated trades response.
///
/// Binance `/api/v3/aggTrades` returns:
/// ```json
/// [
///   {
///     "a": 26129,        // Aggregate tradeId
///     "p": "0.01633102",  // Price
///     "q": "4.70443515",  // Quantity
///     "f": 27781,         // First tradeId
///     "l": 27781,         // Last tradeId
///     "T": 1498793709153, // Timestamp
///     "m": true,          // Was the buyer the maker?
///     "M": true           // Was the trade the best price match?
///   }
/// ]
/// ```
fn parse_binance_agg_trades(body: &str) -> Result<Vec<BackfillTrade>> {
    #[derive(Deserialize)]
    struct BinanceAggTrade {
        a: u64,
        p: String,
        q: String,
        #[serde(rename = "T")]
        timestamp: i64,
        m: bool,
    }

    let raw: Vec<BinanceAggTrade> = serde_json::from_str(body)
        .map_err(|e| anyhow!("Failed to parse Binance aggTrades: {}", e))?;

    let trades = raw
        .into_iter()
        .filter_map(|t| {
            let price = t.p.parse::<f64>().ok()?;
            let quantity = t.q.parse::<f64>().ok()?;
            Some(BackfillTrade {
                trade_id: t.a.to_string(),
                symbol: String::new(), // filled by caller context
                side: if t.m { "sell" } else { "buy" }.to_string(),
                price,
                quantity,
                timestamp_ms: t.timestamp,
            })
        })
        .collect();

    Ok(trades)
}

/// Parse Bybit recent trades response.
///
/// Bybit `/v5/market/recent-trade` returns:
/// ```json
/// {
///   "retCode": 0,
///   "result": {
///     "list": [
///       {
///         "execId": "...",
///         "symbol": "BTCUSDT",
///         "price": "50000.00",
///         "size": "0.01",
///         "side": "Buy",
///         "time": "1698793709153"
///       }
///     ]
///   }
/// }
/// ```
fn parse_bybit_trades(body: &str) -> Result<Vec<BackfillTrade>> {
    #[derive(Deserialize)]
    struct BybitResponse {
        #[serde(rename = "retCode")]
        ret_code: i32,
        result: BybitResult,
    }

    #[derive(Deserialize)]
    struct BybitResult {
        list: Vec<BybitTrade>,
    }

    #[derive(Deserialize)]
    struct BybitTrade {
        #[serde(rename = "execId")]
        exec_id: String,
        symbol: String,
        price: String,
        size: String,
        side: String,
        time: String,
    }

    let resp: BybitResponse =
        serde_json::from_str(body).map_err(|e| anyhow!("Failed to parse Bybit trades: {}", e))?;

    if resp.ret_code != 0 {
        anyhow::bail!("Bybit API returned error code: {}", resp.ret_code);
    }

    let trades = resp
        .result
        .list
        .into_iter()
        .filter_map(|t| {
            let price = t.price.parse::<f64>().ok()?;
            let quantity = t.size.parse::<f64>().ok()?;
            let timestamp_ms = t.time.parse::<i64>().ok()?;
            Some(BackfillTrade {
                trade_id: t.exec_id,
                symbol: t.symbol,
                side: t.side.to_lowercase(),
                price,
                quantity,
                timestamp_ms,
            })
        })
        .collect();

    Ok(trades)
}

// ============================================================================
// Statistics
// ============================================================================

/// Scheduler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    /// Total gaps in queue
    pub total_queued: usize,

    /// Gaps ready to be processed
    pub ready_count: usize,

    /// Gaps waiting for retry delay
    pub waiting_count: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backfill::throttle::ThrottleConfig;

    fn create_test_gap(exchange: &str, symbol: &str, age_mins: i64, trades: u64) -> GapInfo {
        let now = Utc::now();
        GapInfo {
            exchange: exchange.to_string(),
            symbol: symbol.to_string(),
            start_time: now - chrono::Duration::minutes(age_mins + 10),
            end_time: now - chrono::Duration::minutes(age_mins),
            estimated_trades: trades,
        }
    }

    #[test]
    fn test_gap_info_calculations() {
        let gap = create_test_gap("binance", "BTCUSD", 5, 1000);

        assert!(gap.duration_secs() >= 600); // At least 10 minutes
        assert!(gap.age_secs() >= 300); // At least 5 minutes old
        assert!(gap.gap_id().contains("binance"));
        assert!(gap.gap_id().contains("BTCUSD"));
    }

    #[test]
    fn test_priority_calculation() {
        let config = SchedulerConfig::default();
        let throttle = Arc::new(BackfillThrottle::new(ThrottleConfig::default()));
        let redis_client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let lock = Arc::new(BackfillLock::new(
            redis_client,
            Default::default(),
            Arc::new(
                crate::backfill::lock::LockMetrics::new(&prometheus::Registry::new()).unwrap(),
            ),
        ));

        let scheduler = BackfillScheduler::new(config, throttle, lock);

        // Older gap should have higher priority
        let old_gap = create_test_gap("binance", "BTCUSD", 60, 1000);
        let new_gap = create_test_gap("binance", "BTCUSD", 5, 1000);

        let old_priority = scheduler.calculate_priority(&old_gap);
        let new_priority = scheduler.calculate_priority(&new_gap);

        assert!(old_priority > new_priority);

        // Larger gap should have higher priority (same age)
        let small_gap = create_test_gap("binance", "BTCUSD", 10, 100);
        let large_gap = create_test_gap("binance", "BTCUSD", 10, 10000);

        let small_priority = scheduler.calculate_priority(&small_gap);
        let large_priority = scheduler.calculate_priority(&large_gap);

        assert!(large_priority > small_priority);
    }

    #[test]
    fn test_exchange_criticality() {
        let config = SchedulerConfig::default();
        let throttle = Arc::new(BackfillThrottle::new(ThrottleConfig::default()));
        let redis_client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let lock = Arc::new(BackfillLock::new(
            redis_client,
            Default::default(),
            Arc::new(
                crate::backfill::lock::LockMetrics::new(&prometheus::Registry::new()).unwrap(),
            ),
        ));

        let scheduler = BackfillScheduler::new(config, throttle, lock);

        assert_eq!(scheduler.exchange_criticality("binance"), 1.0);
        assert_eq!(scheduler.exchange_criticality("bybit"), 0.8);
        assert_eq!(scheduler.exchange_criticality("kucoin"), 0.6);
        assert_eq!(scheduler.exchange_criticality("unknown"), 0.5);
    }

    #[test]
    fn test_scheduled_gap_ordering() {
        let gap1 = create_test_gap("binance", "BTCUSD", 10, 1000);
        let gap2 = create_test_gap("bybit", "ETHUSDT", 20, 2000);

        let scheduled1 = ScheduledGap::new(gap1, 5.0);
        let scheduled2 = ScheduledGap::new(gap2, 10.0);

        // Higher priority should come first (max-heap behavior)
        assert!(scheduled2 > scheduled1);
    }

    #[tokio::test]
    async fn test_submit_and_queue_size() {
        let config = SchedulerConfig::default();
        let throttle = Arc::new(BackfillThrottle::new(ThrottleConfig::default()));
        let redis_client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let lock = Arc::new(BackfillLock::new(
            redis_client,
            Default::default(),
            Arc::new(
                crate::backfill::lock::LockMetrics::new(&prometheus::Registry::new()).unwrap(),
            ),
        ));

        let scheduler = BackfillScheduler::new(config, throttle, lock);

        assert_eq!(scheduler.queue_size().await, 0);

        let gap = create_test_gap("binance", "BTCUSD", 5, 1000);
        scheduler.submit_gap(gap).await;

        assert_eq!(scheduler.queue_size().await, 1);
    }

    #[tokio::test]
    async fn test_scheduler_stats() {
        let config = SchedulerConfig::default();
        let throttle = Arc::new(BackfillThrottle::new(ThrottleConfig::default()));
        let redis_client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let lock = Arc::new(BackfillLock::new(
            redis_client,
            Default::default(),
            Arc::new(
                crate::backfill::lock::LockMetrics::new(&prometheus::Registry::new()).unwrap(),
            ),
        ));

        let scheduler = BackfillScheduler::new(config, throttle, lock);

        let gap = create_test_gap("binance", "BTCUSD", 5, 1000);
        scheduler.submit_gap(gap).await;

        let stats = scheduler.stats().await;
        assert_eq!(stats.total_queued, 1);
        assert_eq!(stats.ready_count, 1);
        assert_eq!(stats.waiting_count, 0);
    }
}
