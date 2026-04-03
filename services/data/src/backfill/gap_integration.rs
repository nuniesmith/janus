//! Gap Detection Integration
//!
//! This module integrates the gap detection system with the backfill scheduler,
//! automatically submitting detected gaps for backfilling.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │         Gap Detection System (JANUS crate)              │
//! │  - Sequence ID tracking                                 │
//! │  - Heartbeat monitoring                                 │
//! │  - Statistical anomaly detection                        │
//! └───────────────────┬─────────────────────────────────────┘
//!                     │
//!                     ▼
//!          ┌────────────────────────┐
//!          │  GapIntegrationManager │
//!          │  (This Module)         │
//!          └────────┬───────────────┘
//!                   │
//!          ┌────────┼────────┐
//!          │        │        │
//!      Filter   Convert  Submit
//!          │        │        │
//!          ▼        ▼        ▼
//!     ┌────────────────────────┐
//!     │  BackfillScheduler     │
//!     │  (Priority Queue)      │
//!     └────────────────────────┘
//! ```
//!
//! ## Features
//!
//! - Automatic gap detection monitoring
//! - Gap filtering (ignore very small gaps)
//! - Deduplication (prevent duplicate gap submissions)
//! - Metrics tracking (gaps detected, filtered, submitted)
//! - Configurable thresholds
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_data::backfill::gap_integration::{GapIntegrationManager, GapIntegrationConfig};
//! use fks_ruby::backfill::BackfillScheduler;
//! use std::sync::Arc;
//!
//! # async fn example() -> anyhow::Result<()> {
//! # let scheduler = Arc::new(BackfillScheduler::new(
//! #     Default::default(),
//! #     Arc::new(fks_ruby::backfill::BackfillThrottle::new(Default::default())),
//! #     Arc::new(fks_ruby::backfill::BackfillLock::new(
//! #         redis::Client::open("redis://127.0.0.1:6379")?,
//! #         Default::default(),
//! #         Arc::new(fks_ruby::backfill::lock::LockMetrics::new(&prometheus::Registry::new())?),
//! #     )),
//! # ));
//! let config = GapIntegrationConfig::default();
//! let manager = GapIntegrationManager::new(config, scheduler);
//!
//! // Monitor for gaps and auto-submit to scheduler
//! // Typically called from gap detection callbacks
//! # Ok(())
//! # }
//! ```

use crate::backfill::scheduler::{BackfillScheduler, GapInfo};
use crate::metrics::prometheus_exporter::GAPS_DETECTED;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

// ============================================================================
// Configuration
// ============================================================================

/// Gap integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapIntegrationConfig {
    /// Minimum gap duration (seconds) to submit for backfill
    pub min_gap_duration_secs: i64,

    /// Minimum estimated trades to submit for backfill
    pub min_gap_trades: u64,

    /// Maximum gap size (trades) to attempt backfill
    /// (larger gaps may overwhelm the system)
    pub max_gap_trades: u64,

    /// How long to remember seen gaps for deduplication (seconds)
    pub dedup_window_secs: i64,

    /// Auto-submit gaps to scheduler (if false, gaps are only logged)
    pub auto_submit: bool,
}

impl Default for GapIntegrationConfig {
    fn default() -> Self {
        Self {
            min_gap_duration_secs: 10, // Ignore gaps < 10 seconds
            min_gap_trades: 10,        // Ignore gaps < 10 trades
            max_gap_trades: 1_000_000, // Don't backfill > 1M trades
            dedup_window_secs: 3600,   // Remember gaps for 1 hour
            auto_submit: true,         // Auto-submit by default
        }
    }
}

// ============================================================================
// Gap Integration Manager
// ============================================================================

/// Manager for integrating gap detection with backfill scheduler
pub struct GapIntegrationManager {
    /// Configuration
    config: GapIntegrationConfig,

    /// Reference to backfill scheduler
    scheduler: Arc<BackfillScheduler>,

    /// Deduplication set (gap IDs we've recently seen)
    seen_gaps: Arc<RwLock<HashSet<String>>>,

    /// Statistics
    stats: Arc<RwLock<IntegrationStats>>,
}

impl GapIntegrationManager {
    /// Create a new gap integration manager
    pub fn new(config: GapIntegrationConfig, scheduler: Arc<BackfillScheduler>) -> Self {
        Self {
            config,
            scheduler,
            seen_gaps: Arc::new(RwLock::new(HashSet::new())),
            stats: Arc::new(RwLock::new(IntegrationStats::default())),
        }
    }

    /// Handle a detected gap from the gap detection system
    pub async fn handle_gap(
        &self,
        exchange: String,
        symbol: String,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        estimated_trades: u64,
    ) {
        // Record gap detection
        GAPS_DETECTED.with_label_values(&[&exchange, &symbol]).inc();

        self.stats.write().await.total_detected += 1;

        // Create gap info
        let gap = GapInfo {
            exchange: exchange.clone(),
            symbol: symbol.clone(),
            start_time,
            end_time,
            estimated_trades,
        };

        // Check if we should process this gap
        if !self.should_process_gap(&gap).await {
            return;
        }

        // Check deduplication
        if self.is_duplicate(&gap).await {
            debug!(
                exchange = %exchange,
                symbol = %symbol,
                gap_id = %gap.gap_id(),
                "Gap already seen recently, skipping"
            );
            self.stats.write().await.duplicates += 1;
            return;
        }

        // Mark as seen
        self.mark_seen(&gap).await;

        // Submit to scheduler
        if self.config.auto_submit {
            info!(
                exchange = %exchange,
                symbol = %symbol,
                start = %start_time,
                end = %end_time,
                estimated_trades = estimated_trades,
                "Submitting gap to backfill scheduler"
            );

            self.scheduler.submit_gap(gap).await;
            self.stats.write().await.submitted += 1;
        } else {
            info!(
                exchange = %exchange,
                symbol = %symbol,
                start = %start_time,
                end = %end_time,
                estimated_trades = estimated_trades,
                "Gap detected (auto_submit disabled)"
            );
            self.stats.write().await.logged_only += 1;
        }
    }

    /// Check if a gap should be processed
    async fn should_process_gap(&self, gap: &GapInfo) -> bool {
        let duration_secs = gap.duration_secs();

        // Check minimum duration
        if duration_secs < self.config.min_gap_duration_secs {
            debug!(
                exchange = %gap.exchange,
                symbol = %gap.symbol,
                duration_secs = duration_secs,
                min_duration = self.config.min_gap_duration_secs,
                "Gap too short, filtering"
            );
            self.stats.write().await.filtered_too_small += 1;
            return false;
        }

        // Check minimum trades
        if gap.estimated_trades < self.config.min_gap_trades {
            debug!(
                exchange = %gap.exchange,
                symbol = %gap.symbol,
                estimated_trades = gap.estimated_trades,
                min_trades = self.config.min_gap_trades,
                "Gap has too few trades, filtering"
            );
            self.stats.write().await.filtered_too_small += 1;
            return false;
        }

        // Check maximum trades
        if gap.estimated_trades > self.config.max_gap_trades {
            warn!(
                exchange = %gap.exchange,
                symbol = %gap.symbol,
                estimated_trades = gap.estimated_trades,
                max_trades = self.config.max_gap_trades,
                "Gap too large to backfill safely, filtering"
            );
            self.stats.write().await.filtered_too_large += 1;
            return false;
        }

        true
    }

    /// Check if a gap has been seen recently (deduplication)
    async fn is_duplicate(&self, gap: &GapInfo) -> bool {
        let gap_id = gap.gap_id();
        let seen = self.seen_gaps.read().await;
        seen.contains(&gap_id)
    }

    /// Mark a gap as seen for deduplication
    async fn mark_seen(&self, gap: &GapInfo) {
        let gap_id = gap.gap_id();
        let mut seen = self.seen_gaps.write().await;
        seen.insert(gap_id);

        // Note: In production, you'd want a cleanup task to remove old entries
        // after dedup_window_secs to prevent unbounded growth
    }

    /// Cleanup old deduplication entries (should be called periodically)
    pub async fn cleanup_dedup_cache(&self) {
        let mut seen = self.seen_gaps.write().await;
        let initial_size = seen.len();

        // In a real implementation, you'd track timestamps and remove old entries
        // For now, if the set gets too large, clear it
        if initial_size > 10000 {
            warn!(
                size = initial_size,
                "Deduplication cache too large, clearing"
            );
            seen.clear();
        }
    }

    /// Get integration statistics
    pub async fn get_stats(&self) -> IntegrationStats {
        self.stats.read().await.clone()
    }

    /// Reset statistics
    pub async fn reset_stats(&self) {
        *self.stats.write().await = IntegrationStats::default();
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Integration statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntegrationStats {
    /// Total gaps detected
    pub total_detected: u64,

    /// Gaps filtered (too small)
    pub filtered_too_small: u64,

    /// Gaps filtered (too large)
    pub filtered_too_large: u64,

    /// Duplicate gaps (already seen)
    pub duplicates: u64,

    /// Gaps submitted to scheduler
    pub submitted: u64,

    /// Gaps only logged (auto_submit disabled)
    pub logged_only: u64,
}

impl IntegrationStats {
    /// Get total gaps processed (submitted + filtered + duplicates)
    pub fn total_processed(&self) -> u64 {
        self.submitted
            + self.filtered_too_small
            + self.filtered_too_large
            + self.duplicates
            + self.logged_only
    }

    /// Get submission rate (submitted / total_processed)
    pub fn submission_rate(&self) -> f64 {
        let total = self.total_processed();
        if total == 0 {
            0.0
        } else {
            self.submitted as f64 / total as f64
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Estimate number of trades in a gap based on duration and typical rate
///
/// This is a fallback when exact trade counts aren't available
pub fn estimate_gap_trades(duration_secs: i64, exchange: &str, symbol: &str) -> u64 {
    // Rough estimates based on typical exchange volumes
    let trades_per_second = match (exchange.to_lowercase().as_str(), symbol) {
        ("binance", s) if s.contains("BTC") || s.contains("ETH") => 5.0,
        ("binance", _) => 2.0,
        ("bybit", s) if s.contains("BTC") || s.contains("ETH") => 3.0,
        ("bybit", _) => 1.0,
        ("kucoin", _) => 0.5,
        _ => 1.0,
    };

    (duration_secs as f64 * trades_per_second) as u64
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backfill::BackfillLock;
    use crate::backfill::throttle::{BackfillThrottle, ThrottleConfig};
    use chrono::Duration;

    fn create_test_scheduler() -> Arc<BackfillScheduler> {
        let throttle = Arc::new(BackfillThrottle::new(ThrottleConfig::default()));
        let redis_client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let lock = Arc::new(BackfillLock::new(
            redis_client,
            Default::default(),
            Arc::new(
                crate::backfill::lock::LockMetrics::new(&prometheus::Registry::new()).unwrap(),
            ),
        ));

        Arc::new(BackfillScheduler::new(Default::default(), throttle, lock))
    }

    #[tokio::test]
    async fn test_gap_integration_creation() {
        let scheduler = create_test_scheduler();
        let config = GapIntegrationConfig::default();
        let manager = GapIntegrationManager::new(config, scheduler);

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_detected, 0);
    }

    #[tokio::test]
    async fn test_gap_filtering_too_small() {
        let scheduler = create_test_scheduler();
        let config = GapIntegrationConfig {
            min_gap_duration_secs: 60,
            min_gap_trades: 100,
            ..Default::default()
        };

        let manager = GapIntegrationManager::new(config, scheduler);

        let now = Utc::now();
        manager
            .handle_gap(
                "binance".to_string(),
                "BTCUSD".to_string(),
                now - Duration::seconds(5), // Only 5 seconds
                now,
                50, // Only 50 trades
            )
            .await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_detected, 1);
        assert_eq!(stats.filtered_too_small, 1);
        assert_eq!(stats.submitted, 0);
    }

    #[tokio::test]
    async fn test_gap_filtering_too_large() {
        let scheduler = create_test_scheduler();
        let config = GapIntegrationConfig {
            max_gap_trades: 1000,
            ..Default::default()
        };

        let manager = GapIntegrationManager::new(config, scheduler);

        let now = Utc::now();
        manager
            .handle_gap(
                "binance".to_string(),
                "BTCUSD".to_string(),
                now - Duration::minutes(60),
                now,
                100_000, // Too many trades
            )
            .await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_detected, 1);
        assert_eq!(stats.filtered_too_large, 1);
        assert_eq!(stats.submitted, 0);
    }

    #[tokio::test]
    async fn test_gap_submission() {
        let scheduler = create_test_scheduler();
        let config = GapIntegrationConfig::default();
        let manager = GapIntegrationManager::new(config, scheduler.clone());

        let now = Utc::now();
        manager
            .handle_gap(
                "binance".to_string(),
                "BTCUSD".to_string(),
                now - Duration::minutes(10),
                now,
                1000,
            )
            .await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_detected, 1);
        assert_eq!(stats.submitted, 1);
        assert_eq!(stats.filtered_too_small, 0);

        // Verify it was added to scheduler
        assert_eq!(scheduler.queue_size().await, 1);
    }

    #[tokio::test]
    async fn test_gap_deduplication() {
        let scheduler = create_test_scheduler();
        let config = GapIntegrationConfig::default();
        let manager = GapIntegrationManager::new(config, scheduler.clone());

        let now = Utc::now();
        let start = now - Duration::minutes(10);
        let end = now;

        // Submit same gap twice
        manager
            .handle_gap(
                "binance".to_string(),
                "BTCUSD".to_string(),
                start,
                end,
                1000,
            )
            .await;

        manager
            .handle_gap(
                "binance".to_string(),
                "BTCUSD".to_string(),
                start,
                end,
                1000,
            )
            .await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_detected, 2);
        assert_eq!(stats.submitted, 1); // Only submitted once
        assert_eq!(stats.duplicates, 1);

        // Only one gap in scheduler
        assert_eq!(scheduler.queue_size().await, 1);
    }

    #[tokio::test]
    async fn test_auto_submit_disabled() {
        let scheduler = create_test_scheduler();
        let config = GapIntegrationConfig {
            auto_submit: false,
            ..Default::default()
        };

        let manager = GapIntegrationManager::new(config, scheduler.clone());

        let now = Utc::now();
        manager
            .handle_gap(
                "binance".to_string(),
                "BTCUSD".to_string(),
                now - Duration::minutes(10),
                now,
                1000,
            )
            .await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.total_detected, 1);
        assert_eq!(stats.submitted, 0);
        assert_eq!(stats.logged_only, 1);

        // Nothing in scheduler
        assert_eq!(scheduler.queue_size().await, 0);
    }

    #[test]
    fn test_estimate_gap_trades() {
        // Binance BTC should be high volume
        let btc_trades = estimate_gap_trades(60, "binance", "BTCUSD");
        assert!(btc_trades >= 60 * 5); // At least 5 trades/sec

        // KuCoin should be lower
        let kucoin_trades = estimate_gap_trades(60, "kucoin", "BTCUSD");
        assert!(kucoin_trades < btc_trades);
    }

    #[test]
    fn test_integration_stats_calculations() {
        let stats = IntegrationStats {
            total_detected: 100,
            filtered_too_small: 20,
            filtered_too_large: 5,
            duplicates: 10,
            submitted: 60,
            logged_only: 5,
        };

        assert_eq!(stats.total_processed(), 100);
        assert_eq!(stats.submission_rate(), 0.6);
    }
}
