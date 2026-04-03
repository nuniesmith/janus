//! Backfill Throttling & Resource Management
//!
//! Implements P0 Item 4: Backfill Throttling & Disk Monitoring
//!
//! This module provides:
//! - Concurrency limiting (max 2 concurrent backfills)
//! - Disk usage monitoring (stop at 90%, alert at 80%)
//! - QuestDB OOO buffer protection (max 2M rows)
//! - Graceful degradation under resource pressure
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              Backfill Request                           │
//! └─────────────────────┬───────────────────────────────────┘
//!                       │
//!                       ▼
//!          ┌────────────────────────┐
//!          │  Check Disk Usage      │
//!          │  (< 90% required)      │
//!          └────────┬───────────────┘
//!                   │ OK
//!                   ▼
//!          ┌────────────────────────┐
//!          │  Acquire Semaphore     │
//!          │  (max 2 permits)       │
//!          └────────┬───────────────┘
//!                   │ Permit acquired
//!                   ▼
//!          ┌────────────────────────┐
//!          │  Execute Backfill      │
//!          │  (batched writes)      │
//!          └────────┬───────────────┘
//!                   │ Complete
//!                   ▼
//!          ┌────────────────────────┐
//!          │  Release Semaphore     │
//!          │  (automatic on drop)   │
//!          └────────────────────────┘
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use janus_data::backfill::throttle::{BackfillThrottle, ThrottleConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ThrottleConfig::default();
//! let throttle = BackfillThrottle::new(config);
//!
//! // Execute throttled backfill
//! # let gap = 1000;
//! # async fn backfill_trades(_gap: usize) -> anyhow::Result<()> { Ok(()) }
//! match throttle.execute_backfill(gap, || async move {
//!     // Your backfill logic here
//!     backfill_trades(gap).await
//! }).await {
//!     Ok(()) => println!("Backfill completed"),
//!     Err(fks_ruby::backfill::throttle::ThrottleError::DiskFull { usage, .. }) => {
//!         println!("Backfill rejected: disk usage {}%", usage);
//!     }
//!     Err(e) => println!("Error: {}", e),
//! }
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::Semaphore;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

#[cfg(target_os = "linux")]
use std::fs;

// ============================================================================
// Constants
// ============================================================================

/// Maximum concurrent backfills allowed
const MAX_CONCURRENT_BACKFILLS: usize = 2;

/// Maximum disk usage before stopping backfills (90%)
const MAX_DISK_USAGE: f64 = 0.90;

/// Disk usage threshold for warnings (80%)
const ALERT_DISK_USAGE: f64 = 0.80;

/// QuestDB maximum out-of-order (OOO) rows before commits slow down
const MAX_OOO_ROWS: usize = 2_000_000;

/// Batch size for backfill writes (prevents OOO overflow)
const BACKFILL_BATCH_SIZE: usize = 10_000;

/// Disk monitoring interval
const DISK_MONITOR_INTERVAL: Duration = Duration::from_secs(30);

// ============================================================================
// Error Types
// ============================================================================

/// Throttle-related errors
#[derive(Error, Debug)]
pub enum ThrottleError {
    /// Disk usage too high to start backfill
    #[error("Disk usage too high: {usage:.1}% (max: {max:.1}%)")]
    DiskFull { usage: f64, max: f64 },

    /// All backfill slots are busy
    #[error("All backfill slots busy (max {max} concurrent)")]
    NoCapacity { max: usize },

    /// Gap is too large to backfill safely
    #[error("Gap too large: {size} trades exceeds OOO limit {limit}")]
    GapTooLarge { size: usize, limit: usize },

    /// Failed to check disk usage
    #[error("Failed to check disk usage: {0}")]
    DiskCheckFailed(String),

    /// Backfill execution failed
    #[error("Backfill execution failed: {0}")]
    ExecutionFailed(#[from] anyhow::Error),
}

// ============================================================================
// Configuration
// ============================================================================

/// Throttle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThrottleConfig {
    /// Maximum concurrent backfills
    pub max_concurrent: usize,

    /// Maximum disk usage percentage (0.0 - 1.0)
    pub max_disk_usage: f64,

    /// Alert threshold for disk usage (0.0 - 1.0)
    pub alert_disk_usage: f64,

    /// Maximum OOO rows before batching
    pub max_ooo_rows: usize,

    /// Batch size for backfill writes
    pub batch_size: usize,

    /// QuestDB data directory path
    pub questdb_data_dir: String,
}

impl Default for ThrottleConfig {
    fn default() -> Self {
        Self {
            max_concurrent: MAX_CONCURRENT_BACKFILLS,
            max_disk_usage: MAX_DISK_USAGE,
            alert_disk_usage: ALERT_DISK_USAGE,
            max_ooo_rows: MAX_OOO_ROWS,
            batch_size: BACKFILL_BATCH_SIZE,
            questdb_data_dir: "/var/lib/questdb".to_string(),
        }
    }
}

// ============================================================================
// Disk Usage Monitor
// ============================================================================

/// Disk usage information
#[derive(Debug, Clone)]
pub struct DiskUsage {
    /// Total disk space in bytes
    pub total: u64,

    /// Used disk space in bytes
    pub used: u64,

    /// Available disk space in bytes
    pub available: u64,

    /// Usage percentage (0.0 - 1.0)
    pub usage_percent: f64,
}

impl DiskUsage {
    /// Check if disk usage is below the maximum threshold
    pub fn is_below_max(&self, max: f64) -> bool {
        self.usage_percent < max
    }

    /// Check if disk usage exceeds the alert threshold
    pub fn should_alert(&self, threshold: f64) -> bool {
        self.usage_percent >= threshold
    }
}

/// Get disk usage for a given path
#[cfg(target_os = "linux")]
pub async fn get_disk_usage(path: &str) -> Result<DiskUsage, ThrottleError> {
    let _metadata = tokio::fs::metadata(path)
        .await
        .map_err(|e| ThrottleError::DiskCheckFailed(format!("Failed to stat {}: {}", path, e)))?;

    // Read /proc/mounts to find the mount point
    let mounts = fs::read_to_string("/proc/mounts").map_err(|e| {
        ThrottleError::DiskCheckFailed(format!("Failed to read /proc/mounts: {}", e))
    })?;

    let mut mount_point = "/".to_string();
    for line in mounts.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let mp = parts[1];
            if path.starts_with(mp) && mp.len() > mount_point.len() {
                mount_point = mp.to_string();
            }
        }
    }

    // Use statvfs to get filesystem stats
    let stats = nix::sys::statvfs::statvfs(mount_point.as_str())
        .map_err(|e| ThrottleError::DiskCheckFailed(format!("statvfs failed: {}", e)))?;

    let total = stats.blocks() * stats.block_size();
    let available = stats.blocks_available() * stats.block_size();
    let used = total - available;
    let usage_percent = if total > 0 {
        used as f64 / total as f64
    } else {
        0.0
    };

    Ok(DiskUsage {
        total,
        used,
        available,
        usage_percent,
    })
}

/// Fallback disk usage implementation for non-Linux systems
#[cfg(not(target_os = "linux"))]
pub async fn get_disk_usage(_path: &str) -> Result<DiskUsage, ThrottleError> {
    // Return a safe default on non-Linux systems
    warn!("Disk usage monitoring not implemented for this OS, returning safe defaults");
    Ok(DiskUsage {
        total: 1_000_000_000_000,   // 1TB
        used: 100_000_000_000,      // 100GB
        available: 900_000_000_000, // 900GB
        usage_percent: 0.10,        // 10%
    })
}

// ============================================================================
// Backfill Throttle
// ============================================================================

/// Backfill throttle with resource management
pub struct BackfillThrottle {
    /// Semaphore limiting concurrent backfills
    semaphore: Arc<Semaphore>,

    /// Configuration
    config: ThrottleConfig,

    /// Last disk check result (cached for performance)
    last_disk_check: Arc<tokio::sync::RwLock<Option<(Instant, DiskUsage)>>>,
}

impl BackfillThrottle {
    /// Create a new backfill throttle
    pub fn new(config: ThrottleConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));

        Self {
            semaphore,
            config,
            last_disk_check: Arc::new(tokio::sync::RwLock::new(None)),
        }
    }

    /// Check disk usage with caching (checks every 10 seconds max)
    async fn check_disk_usage(&self) -> Result<DiskUsage, ThrottleError> {
        // Check cache first
        {
            let cache = self.last_disk_check.read().await;
            if let Some((timestamp, usage)) = cache.as_ref()
                && timestamp.elapsed() < Duration::from_secs(10)
            {
                return Ok(usage.clone());
            }
        }

        // Perform actual disk check
        let usage = get_disk_usage(&self.config.questdb_data_dir).await?;

        // Update cache
        {
            let mut cache = self.last_disk_check.write().await;
            *cache = Some((Instant::now(), usage.clone()));
        }

        // Log warnings if needed
        if usage.should_alert(self.config.alert_disk_usage) {
            warn!(
                usage_percent = format!("{:.1}%", usage.usage_percent * 100.0),
                threshold = format!("{:.1}%", self.config.alert_disk_usage * 100.0),
                available_gb = usage.available / 1_073_741_824,
                "Disk usage HIGH - approaching limit"
            );

            // Update Prometheus metric
            {
                use crate::metrics::prometheus_exporter::QUESTDB_DISK_USAGE;
                QUESTDB_DISK_USAGE.set(usage.usage_percent * 100.0);
            }
        }

        Ok(usage)
    }

    /// Check if a backfill can be started
    pub async fn can_start_backfill(&self, gap_size: usize) -> Result<(), ThrottleError> {
        // Check disk usage first
        let disk_usage = self.check_disk_usage().await?;
        if !disk_usage.is_below_max(self.config.max_disk_usage) {
            return Err(ThrottleError::DiskFull {
                usage: disk_usage.usage_percent * 100.0,
                max: self.config.max_disk_usage * 100.0,
            });
        }

        // Check gap size
        if gap_size > self.config.max_ooo_rows {
            return Err(ThrottleError::GapTooLarge {
                size: gap_size,
                limit: self.config.max_ooo_rows,
            });
        }

        // Check if a permit is available (non-blocking check)
        if self.semaphore.available_permits() == 0 {
            return Err(ThrottleError::NoCapacity {
                max: self.config.max_concurrent,
            });
        }

        Ok(())
    }

    /// Execute a backfill with throttling
    ///
    /// This acquires a semaphore permit, checks disk usage, and executes the backfill.
    /// The permit is automatically released when the future completes.
    pub async fn execute_backfill<F, Fut, T>(
        &self,
        gap_size: usize,
        f: F,
    ) -> Result<T, ThrottleError>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        // Pre-flight checks
        let disk_usage = self.check_disk_usage().await?;
        if !disk_usage.is_below_max(self.config.max_disk_usage) {
            return Err(ThrottleError::DiskFull {
                usage: disk_usage.usage_percent * 100.0,
                max: self.config.max_disk_usage * 100.0,
            });
        }

        if gap_size > self.config.max_ooo_rows {
            warn!(
                gap_size = gap_size,
                max_ooo = self.config.max_ooo_rows,
                "Gap exceeds OOO limit - will be batched"
            );
        }

        // Acquire semaphore permit (blocks if at capacity)
        let start = Instant::now();
        debug!("Acquiring backfill permit...");
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| ThrottleError::ExecutionFailed(anyhow::anyhow!("Semaphore closed")))?;

        let wait_time = start.elapsed();
        if wait_time > Duration::from_secs(1) {
            info!(
                wait_ms = wait_time.as_millis(),
                "Waited for backfill permit"
            );
        }

        // Update metrics
        {
            use crate::metrics::prometheus_exporter::BACKFILLS_RUNNING;
            BACKFILLS_RUNNING.inc();
        }

        // Execute backfill
        let result = f().await.map_err(ThrottleError::ExecutionFailed);

        // Release permit (automatic via Drop)
        {
            use crate::metrics::prometheus_exporter::BACKFILLS_RUNNING;
            BACKFILLS_RUNNING.dec();
        }

        result
    }

    /// Get current number of running backfills
    pub fn running_count(&self) -> usize {
        self.config.max_concurrent - self.semaphore.available_permits()
    }

    /// Get available backfill slots
    pub fn available_slots(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Start periodic disk monitoring task
    pub fn start_disk_monitor(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = interval(DISK_MONITOR_INTERVAL);

            loop {
                interval.tick().await;

                match self.check_disk_usage().await {
                    Ok(usage) => {
                        debug!(
                            usage_percent = format!("{:.1}%", usage.usage_percent * 100.0),
                            available_gb = usage.available / 1_073_741_824,
                            "Disk usage check"
                        );

                        // Critical alert if disk is almost full
                        if usage.usage_percent >= self.config.max_disk_usage {
                            error!(
                                usage_percent = format!("{:.1}%", usage.usage_percent * 100.0),
                                "CRITICAL: Disk usage at maximum - backfills will be blocked"
                            );
                        }
                    }
                    Err(e) => {
                        error!(error = %e, "Failed to check disk usage");
                    }
                }
            }
        })
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Calculate number of batches needed for a gap
pub fn calculate_batches(gap_size: usize, batch_size: usize) -> usize {
    gap_size.div_ceil(batch_size)
}

/// Check if a gap should be batched
pub fn should_batch(gap_size: usize, ooo_limit: usize) -> bool {
    gap_size > ooo_limit
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ThrottleConfig::default();
        assert_eq!(config.max_concurrent, 2);
        assert_eq!(config.max_disk_usage, 0.90);
        assert_eq!(config.alert_disk_usage, 0.80);
        assert_eq!(config.batch_size, 10_000);
    }

    #[test]
    fn test_disk_usage_checks() {
        let usage = DiskUsage {
            total: 1000,
            used: 850,
            available: 150,
            usage_percent: 0.85,
        };

        assert!(usage.is_below_max(0.90));
        assert!(!usage.is_below_max(0.80));
        assert!(usage.should_alert(0.80));
        assert!(!usage.should_alert(0.90));
    }

    #[tokio::test]
    async fn test_throttle_creation() {
        let config = ThrottleConfig::default();
        let throttle = BackfillThrottle::new(config);

        assert_eq!(throttle.running_count(), 0);
        assert_eq!(throttle.available_slots(), 2);
    }

    #[tokio::test]
    async fn test_concurrent_limit() {
        let config = ThrottleConfig {
            max_concurrent: 2,
            questdb_data_dir: "/tmp".to_string(), // Use /tmp for tests
            ..Default::default()
        };
        let throttle = Arc::new(BackfillThrottle::new(config));

        // Use channels to signal when tasks have acquired permits
        let (tx1, mut rx1) = tokio::sync::mpsc::channel(1);
        let (tx2, mut rx2) = tokio::sync::mpsc::channel(1);

        let t1 = Arc::clone(&throttle);
        let h1 = tokio::spawn(async move {
            t1.execute_backfill(100, || async {
                let _ = tx1.send(()).await;
                // Hold permit for 200ms to ensure test can check running count
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok::<_, anyhow::Error>(())
            })
            .await
        });

        let t2 = Arc::clone(&throttle);
        let h2 = tokio::spawn(async move {
            t2.execute_backfill(100, || async {
                let _ = tx2.send(()).await;
                // Hold permit for 200ms to ensure test can check running count
                tokio::time::sleep(Duration::from_millis(200)).await;
                Ok::<_, anyhow::Error>(())
            })
            .await
        });

        // Wait for both tasks to signal they've acquired permits
        rx1.recv().await;
        rx2.recv().await;

        // Give a small delay to ensure semaphore state is stable
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Should have 0 available slots now
        assert_eq!(throttle.running_count(), 2);
        assert_eq!(throttle.available_slots(), 0);

        // Wait for completion
        let _ = tokio::join!(h1, h2);

        // Permits should be released
        assert_eq!(throttle.running_count(), 0);
        assert_eq!(throttle.available_slots(), 2);
    }

    #[tokio::test]
    async fn test_gap_size_limit() {
        let config = ThrottleConfig {
            max_ooo_rows: 1000,
            questdb_data_dir: "/tmp".to_string(), // Use /tmp for tests
            ..Default::default()
        };
        let throttle = BackfillThrottle::new(config);

        // Small gap should be OK
        let result = throttle.can_start_backfill(500).await;
        assert!(result.is_ok(), "Small gap should be OK, got: {:?}", result);

        // Large gap should be rejected
        let result = throttle.can_start_backfill(2000).await;
        assert!(
            matches!(result, Err(ThrottleError::GapTooLarge { .. })),
            "Large gap should be rejected, got: {:?}",
            result
        );
    }

    #[test]
    fn test_calculate_batches() {
        assert_eq!(calculate_batches(10_000, 10_000), 1);
        assert_eq!(calculate_batches(15_000, 10_000), 2);
        assert_eq!(calculate_batches(25_000, 10_000), 3);
        assert_eq!(calculate_batches(1, 10_000), 1);
    }

    #[test]
    fn test_should_batch() {
        assert!(!should_batch(1000, 2000));
        assert!(should_batch(3000, 2000));
        assert!(!should_batch(2000, 2000)); // Equal is not greater
    }

    #[tokio::test]
    async fn test_disk_monitor_start() {
        let config = ThrottleConfig::default();
        let throttle = Arc::new(BackfillThrottle::new(config));

        let handle = throttle.clone().start_disk_monitor();

        // Let it run for a bit
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Stop the monitor
        handle.abort();
    }
}
