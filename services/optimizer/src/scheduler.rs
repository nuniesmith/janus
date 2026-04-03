//! Optimization Scheduler
//!
//! Provides scheduled execution of optimization cycles with configurable intervals.
//! Supports both fixed intervals (e.g., "6h") and cron expressions for more complex schedules.
//!
//! # Features
//!
//! - Fixed interval scheduling (e.g., every 6 hours)
//! - Graceful shutdown on service termination
//! - Skip optimization if previous run is still in progress
//! - Configurable retry on failure
//!
//! # Usage
//!
//! ```rust,ignore
//! let scheduler = OptimizationScheduler::new(
//!     service,
//!     state,
//!     "6h".to_string(),
//! );
//!
//! // Run scheduler (blocks until shutdown)
//! scheduler.run().await;
//! ```

use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio::time::{Instant, interval_at};
use tracing::{debug, error, info, warn};

use crate::AppState;
use crate::config::parse_interval;
use crate::service::OptimizerService;

/// Scheduler state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerState {
    /// Scheduler is idle, waiting for next run
    Idle,
    /// Optimization is currently running
    Running,
    /// Scheduler is shutting down
    ShuttingDown,
    /// Scheduler has stopped
    Stopped,
}

/// Statistics about scheduler runs
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total number of scheduled runs
    pub total_runs: u64,

    /// Successful optimization runs
    pub successful_runs: u64,

    /// Failed optimization runs
    pub failed_runs: u64,

    /// Skipped runs (previous still running)
    pub skipped_runs: u64,

    /// Last run timestamp
    pub last_run_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Next scheduled run timestamp
    pub next_run_at: Option<chrono::DateTime<chrono::Utc>>,

    /// Last run duration in seconds
    pub last_run_duration_secs: Option<f64>,
}

/// Optimization scheduler
pub struct OptimizationScheduler {
    /// Optimizer service
    service: Arc<RwLock<OptimizerService>>,

    /// Application state
    state: Arc<AppState>,

    /// Optimization interval string (e.g., "6h", "30m")
    interval_str: String,

    /// Parsed interval duration
    interval: Duration,

    /// Current scheduler state
    scheduler_state: Arc<RwLock<SchedulerState>>,

    /// Scheduler statistics
    stats: Arc<RwLock<SchedulerStats>>,

    /// Maximum consecutive failures before backing off
    max_consecutive_failures: u32,

    /// Backoff multiplier for failures
    backoff_multiplier: f64,

    /// Maximum backoff duration
    max_backoff: Duration,
}

impl OptimizationScheduler {
    /// Create a new scheduler
    pub fn new(
        service: Arc<RwLock<OptimizerService>>,
        state: Arc<AppState>,
        interval_str: String,
    ) -> Self {
        let interval = parse_interval(&interval_str).unwrap_or(Duration::from_secs(6 * 3600));

        info!(
            "Scheduler configured with interval: {} ({:?})",
            interval_str, interval
        );

        Self {
            service,
            state,
            interval_str,
            interval,
            scheduler_state: Arc::new(RwLock::new(SchedulerState::Idle)),
            stats: Arc::new(RwLock::new(SchedulerStats::default())),
            max_consecutive_failures: 3,
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(3600), // 1 hour max backoff
        }
    }

    /// Create scheduler with custom failure handling
    #[allow(dead_code)]
    pub fn with_failure_config(
        mut self,
        max_consecutive_failures: u32,
        backoff_multiplier: f64,
        max_backoff: Duration,
    ) -> Self {
        self.max_consecutive_failures = max_consecutive_failures;
        self.backoff_multiplier = backoff_multiplier;
        self.max_backoff = max_backoff;
        self
    }

    /// Get current scheduler state
    #[allow(dead_code)]
    pub async fn get_state(&self) -> SchedulerState {
        *self.scheduler_state.read().await
    }

    /// Get scheduler statistics
    #[allow(dead_code)]
    pub async fn get_stats(&self) -> SchedulerStats {
        self.stats.read().await.clone()
    }

    /// Run the scheduler (main loop)
    pub async fn run(&self) {
        info!(
            "Starting optimization scheduler with interval: {}",
            self.interval_str
        );

        let mut shutdown_rx = self.state.shutdown_tx.subscribe();
        let mut consecutive_failures = 0u32;
        let mut current_interval = self.interval;

        // Calculate first tick time (start of next interval)
        let start = Instant::now() + self.interval;
        let mut ticker = interval_at(start, current_interval);

        // Update next run time
        self.update_next_run_time(current_interval).await;

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    // Check if we should run
                    let current_state = *self.scheduler_state.read().await;

                    match current_state {
                        SchedulerState::Running => {
                            warn!("Skipping scheduled optimization - previous run still in progress");
                            self.record_skipped().await;
                            continue;
                        }
                        SchedulerState::ShuttingDown | SchedulerState::Stopped => {
                            info!("Scheduler stopping");
                            break;
                        }
                        SchedulerState::Idle => {
                            // Proceed with optimization
                        }
                    }

                    // Run optimization
                    let success = self.run_optimization_cycle().await;

                    if success {
                        consecutive_failures = 0;
                        // Reset interval to normal if we were backed off
                        if current_interval != self.interval {
                            current_interval = self.interval;
                            ticker = interval_at(Instant::now() + current_interval, current_interval);
                            info!("Resetting scheduler interval to {:?}", current_interval);
                        }
                    } else {
                        consecutive_failures += 1;

                        if consecutive_failures >= self.max_consecutive_failures {
                            // Apply backoff
                            let backoff_factor = self.backoff_multiplier.powi(
                                (consecutive_failures - self.max_consecutive_failures + 1) as i32
                            );
                            let backed_off = Duration::from_secs_f64(
                                self.interval.as_secs_f64() * backoff_factor
                            );
                            current_interval = backed_off.min(self.max_backoff);

                            warn!(
                                "Backing off scheduler after {} consecutive failures. New interval: {:?}",
                                consecutive_failures, current_interval
                            );

                            ticker = interval_at(Instant::now() + current_interval, current_interval);
                        }
                    }

                    // Update next run time
                    self.update_next_run_time(current_interval).await;
                }

                _ = shutdown_rx.recv() => {
                    info!("Received shutdown signal, stopping scheduler");
                    *self.scheduler_state.write().await = SchedulerState::ShuttingDown;
                    break;
                }
            }
        }

        *self.scheduler_state.write().await = SchedulerState::Stopped;
        info!("Optimization scheduler stopped");
    }

    /// Run a single optimization cycle
    async fn run_optimization_cycle(&self) -> bool {
        *self.scheduler_state.write().await = SchedulerState::Running;

        let start = std::time::Instant::now();
        let start_time = chrono::Utc::now();

        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        info!("Scheduled optimization starting at {}", start_time);
        info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        let result = {
            let mut service = self.service.write().await;
            service.run_optimization_cycle().await
        };

        let duration = start.elapsed();

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.total_runs += 1;
            stats.last_run_at = Some(start_time);
            stats.last_run_duration_secs = Some(duration.as_secs_f64());
        }

        let success = match result {
            Ok(cycle_result) => {
                {
                    let mut stats = self.stats.write().await;
                    if cycle_result.failed == 0 {
                        stats.successful_runs += 1;
                    } else if cycle_result.successful == 0 {
                        stats.failed_runs += 1;
                    } else {
                        // Partial success
                        stats.successful_runs += 1;
                    }
                }

                self.state
                    .metrics
                    .record_scheduled_run(cycle_result.successful > 0, duration.as_secs_f64());

                info!(
                    "Scheduled optimization complete: {} successful, {} failed in {:.1}s",
                    cycle_result.successful,
                    cycle_result.failed,
                    duration.as_secs_f64()
                );

                cycle_result.successful > 0
            }
            Err(e) => {
                {
                    let mut stats = self.stats.write().await;
                    stats.failed_runs += 1;
                }

                self.state
                    .metrics
                    .record_scheduled_run(false, duration.as_secs_f64());

                error!("Scheduled optimization failed: {}", e);
                false
            }
        };

        *self.scheduler_state.write().await = SchedulerState::Idle;
        success
    }

    /// Record a skipped run
    async fn record_skipped(&self) {
        let mut stats = self.stats.write().await;
        stats.skipped_runs += 1;
    }

    /// Update the next scheduled run time
    async fn update_next_run_time(&self, interval: Duration) {
        let next = chrono::Utc::now() + chrono::Duration::from_std(interval).unwrap_or_default();
        let mut stats = self.stats.write().await;
        stats.next_run_at = Some(next);

        debug!("Next scheduled optimization at {}", next);
    }

    /// Stop the scheduler gracefully
    #[allow(dead_code)]
    pub async fn stop(&self) {
        info!("Requesting scheduler stop");
        *self.scheduler_state.write().await = SchedulerState::ShuttingDown;
    }

    /// Check if scheduler is running
    #[allow(dead_code)]
    pub async fn is_running(&self) -> bool {
        matches!(
            *self.scheduler_state.read().await,
            SchedulerState::Idle | SchedulerState::Running
        )
    }

    /// Get the configured interval
    #[allow(dead_code)]
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Get the interval string
    #[allow(dead_code)]
    pub fn interval_str(&self) -> &str {
        &self.interval_str
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduler_state() {
        assert_eq!(SchedulerState::Idle, SchedulerState::Idle);
        assert_ne!(SchedulerState::Idle, SchedulerState::Running);
    }

    #[test]
    fn test_scheduler_stats_default() {
        let stats = SchedulerStats::default();
        assert_eq!(stats.total_runs, 0);
        assert_eq!(stats.successful_runs, 0);
        assert_eq!(stats.failed_runs, 0);
        assert_eq!(stats.skipped_runs, 0);
        assert!(stats.last_run_at.is_none());
        assert!(stats.next_run_at.is_none());
    }

    #[test]
    fn test_backoff_calculation() {
        let base_interval = Duration::from_secs(3600); // 1 hour
        let multiplier: f64 = 2.0;
        let max_backoff = Duration::from_secs(14400); // 4 hours

        // First failure (after max consecutive)
        let backoff_1 = Duration::from_secs_f64(base_interval.as_secs_f64() * multiplier.powi(1));
        assert_eq!(backoff_1, Duration::from_secs(7200)); // 2 hours

        // Second failure
        let backoff_2 = Duration::from_secs_f64(base_interval.as_secs_f64() * multiplier.powi(2));
        assert_eq!(backoff_2, Duration::from_secs(14400)); // 4 hours

        // Third failure (should be capped)
        let backoff_3 = Duration::from_secs_f64(base_interval.as_secs_f64() * multiplier.powi(3))
            .min(max_backoff);
        assert_eq!(backoff_3, max_backoff); // Capped at 4 hours
    }
}
