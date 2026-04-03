//! Track portfolio balance
//!
//! Part of the Hypothalamus region
//! Component: homeostasis
//!
//! This module tracks portfolio balance metrics over time, maintains historical
//! snapshots, and provides trend analysis for detecting portfolio health changes.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Portfolio balance snapshot
#[derive(Debug, Clone)]
pub struct BalanceSnapshot {
    /// Timestamp (epoch millis)
    pub timestamp: u64,
    /// Total portfolio value
    pub total_value: f64,
    /// Cash balance
    pub cash_balance: f64,
    /// Total position value
    pub position_value: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Realized PnL (cumulative)
    pub realized_pnl: f64,
    /// Number of open positions
    pub position_count: usize,
    /// Gross exposure (long + short)
    pub gross_exposure: f64,
    /// Net exposure (long - short)
    pub net_exposure: f64,
    /// Long exposure
    pub long_exposure: f64,
    /// Short exposure
    pub short_exposure: f64,
    /// Margin used
    pub margin_used: f64,
    /// Available margin
    pub available_margin: f64,
}

impl BalanceSnapshot {
    /// Create a new snapshot
    pub fn new(timestamp: u64, total_value: f64) -> Self {
        Self {
            timestamp,
            total_value,
            cash_balance: total_value,
            position_value: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            position_count: 0,
            gross_exposure: 0.0,
            net_exposure: 0.0,
            long_exposure: 0.0,
            short_exposure: 0.0,
            margin_used: 0.0,
            available_margin: total_value,
        }
    }

    /// Calculate leverage ratio
    pub fn leverage(&self) -> f64 {
        if self.total_value > 0.0 {
            self.gross_exposure / self.total_value
        } else {
            0.0
        }
    }

    /// Calculate long/short ratio
    pub fn long_short_ratio(&self) -> Option<f64> {
        if self.short_exposure > 0.0 {
            Some(self.long_exposure / self.short_exposure)
        } else if self.long_exposure > 0.0 {
            Some(f64::INFINITY)
        } else {
            None
        }
    }

    /// Calculate margin utilization
    pub fn margin_utilization(&self) -> f64 {
        if self.total_value > 0.0 {
            self.margin_used / self.total_value
        } else {
            0.0
        }
    }
}

/// Balance change event
#[derive(Debug, Clone)]
pub struct BalanceChange {
    /// Timestamp
    pub timestamp: u64,
    /// Change type
    pub change_type: ChangeType,
    /// Amount changed
    pub amount: f64,
    /// New total value after change
    pub new_total: f64,
    /// Description
    pub description: String,
}

/// Type of balance change
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Deposit into account
    Deposit,
    /// Withdrawal from account
    Withdrawal,
    /// Realized profit
    RealizedProfit,
    /// Realized loss
    RealizedLoss,
    /// Fee/commission
    Fee,
    /// Interest/funding
    Interest,
    /// Dividend received
    Dividend,
    /// Position mark-to-market change
    MarkToMarket,
    /// Other adjustment
    Other,
}

/// Trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrendDirection {
    /// Strongly increasing
    StrongUp,
    /// Moderately increasing
    Up,
    /// Roughly flat
    Flat,
    /// Moderately decreasing
    Down,
    /// Strongly decreasing
    StrongDown,
}

/// Trend analysis result
#[derive(Debug, Clone)]
pub struct TrendAnalysis {
    /// Overall trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 - 1.0)
    pub strength: f64,
    /// Linear regression slope (per period)
    pub slope: f64,
    /// R-squared of the trend
    pub r_squared: f64,
    /// Period over which trend was calculated
    pub period_count: usize,
    /// Volatility of balance over period
    pub volatility: f64,
    /// Maximum drawdown over period
    pub max_drawdown: f64,
    /// Peak value in period
    pub peak_value: f64,
    /// Trough value in period
    pub trough_value: f64,
}

/// Health status of portfolio balance
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceHealth {
    /// Excellent - growing steadily
    Excellent,
    /// Good - stable or slight growth
    Good,
    /// Warning - declining or volatile
    Warning,
    /// Critical - significant decline
    Critical,
}

/// Configuration for balance tracker
#[derive(Debug, Clone)]
pub struct BalanceTrackerConfig {
    /// Maximum snapshots to retain
    pub max_snapshots: usize,
    /// Maximum change events to retain
    pub max_changes: usize,
    /// Snapshot interval (milliseconds)
    pub snapshot_interval: u64,
    /// Short-term trend window (number of snapshots)
    pub short_term_window: usize,
    /// Long-term trend window (number of snapshots)
    pub long_term_window: usize,
    /// Drawdown warning threshold (percentage)
    pub drawdown_warning_threshold: f64,
    /// Drawdown critical threshold (percentage)
    pub drawdown_critical_threshold: f64,
    /// Minimum change amount to record
    pub min_change_amount: f64,
}

impl Default for BalanceTrackerConfig {
    fn default() -> Self {
        Self {
            max_snapshots: 10000,
            max_changes: 5000,
            snapshot_interval: 60_000, // 1 minute
            short_term_window: 60,     // 1 hour of minute snapshots
            long_term_window: 1440,    // 24 hours of minute snapshots
            drawdown_warning_threshold: 5.0,
            drawdown_critical_threshold: 10.0,
            min_change_amount: 0.01,
        }
    }
}

/// Track portfolio balance
pub struct BalanceTracker {
    /// Configuration
    config: BalanceTrackerConfig,
    /// Historical snapshots
    snapshots: VecDeque<BalanceSnapshot>,
    /// Balance change history
    changes: VecDeque<BalanceChange>,
    /// Current snapshot (latest)
    current: BalanceSnapshot,
    /// High water mark (peak value)
    high_water_mark: f64,
    /// High water mark timestamp
    high_water_mark_time: u64,
    /// Low water mark (trough value)
    low_water_mark: f64,
    /// Starting balance
    starting_balance: f64,
    /// Last snapshot timestamp
    last_snapshot_time: u64,
    /// Daily snapshots for EOD tracking
    daily_snapshots: HashMap<String, BalanceSnapshot>,
    /// Cached short-term trend
    short_term_trend: Option<TrendAnalysis>,
    /// Cached long-term trend
    long_term_trend: Option<TrendAnalysis>,
}

impl Default for BalanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BalanceTracker {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(BalanceTrackerConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: BalanceTrackerConfig) -> Self {
        Self {
            config,
            snapshots: VecDeque::new(),
            changes: VecDeque::new(),
            current: BalanceSnapshot::new(0, 0.0),
            high_water_mark: 0.0,
            high_water_mark_time: 0,
            low_water_mark: f64::MAX,
            starting_balance: 0.0,
            last_snapshot_time: 0,
            daily_snapshots: HashMap::new(),
            short_term_trend: None,
            long_term_trend: None,
        }
    }

    /// Initialize with starting balance
    pub fn initialize(&mut self, starting_balance: f64, timestamp: u64) {
        self.starting_balance = starting_balance;
        self.high_water_mark = starting_balance;
        self.high_water_mark_time = timestamp;
        self.low_water_mark = starting_balance;
        self.current = BalanceSnapshot::new(timestamp, starting_balance);
        self.last_snapshot_time = timestamp;

        // Take initial snapshot
        self.take_snapshot(timestamp);
    }

    /// Update current balance state
    pub fn update(&mut self, snapshot: BalanceSnapshot) {
        let timestamp = snapshot.timestamp;
        let total_value = snapshot.total_value;

        // Update high water mark
        if total_value > self.high_water_mark {
            self.high_water_mark = total_value;
            self.high_water_mark_time = timestamp;
        }

        // Update low water mark
        if total_value < self.low_water_mark {
            self.low_water_mark = total_value;
        }

        self.current = snapshot;

        // Take periodic snapshot if interval has passed
        if timestamp >= self.last_snapshot_time + self.config.snapshot_interval {
            self.take_snapshot(timestamp);
        }

        // Invalidate cached trends
        self.short_term_trend = None;
        self.long_term_trend = None;
    }

    /// Take a snapshot
    fn take_snapshot(&mut self, timestamp: u64) {
        let mut snapshot = self.current.clone();
        snapshot.timestamp = timestamp;

        self.snapshots.push_back(snapshot);
        self.last_snapshot_time = timestamp;

        // Trim old snapshots
        while self.snapshots.len() > self.config.max_snapshots {
            self.snapshots.pop_front();
        }
    }

    /// Record a balance change event
    pub fn record_change(
        &mut self,
        change_type: ChangeType,
        amount: f64,
        timestamp: u64,
        description: &str,
    ) {
        if amount.abs() < self.config.min_change_amount {
            return;
        }

        let change = BalanceChange {
            timestamp,
            change_type,
            amount,
            new_total: self.current.total_value,
            description: description.to_string(),
        };

        self.changes.push_back(change);

        // Trim old changes
        while self.changes.len() > self.config.max_changes {
            self.changes.pop_front();
        }
    }

    /// Record end-of-day snapshot
    pub fn record_daily_snapshot(&mut self, date: &str) {
        self.daily_snapshots
            .insert(date.to_string(), self.current.clone());
    }

    /// Get current balance snapshot
    pub fn current_balance(&self) -> &BalanceSnapshot {
        &self.current
    }

    /// Get current drawdown from high water mark
    pub fn current_drawdown(&self) -> f64 {
        if self.high_water_mark > 0.0 {
            (self.high_water_mark - self.current.total_value) / self.high_water_mark * 100.0
        } else {
            0.0
        }
    }

    /// Get return since inception
    pub fn total_return(&self) -> f64 {
        if self.starting_balance > 0.0 {
            (self.current.total_value - self.starting_balance) / self.starting_balance * 100.0
        } else {
            0.0
        }
    }

    /// Calculate trend analysis over a window
    fn calculate_trend(&self, window: usize) -> Option<TrendAnalysis> {
        if self.snapshots.len() < 2 {
            return None;
        }

        let window_size = window.min(self.snapshots.len());
        let start_idx = self.snapshots.len() - window_size;
        let window_data: Vec<_> = self.snapshots.iter().skip(start_idx).collect();

        if window_data.is_empty() {
            return None;
        }

        // Extract values
        let values: Vec<f64> = window_data.iter().map(|s| s.total_value).collect();
        let n = values.len() as f64;

        // Calculate linear regression
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;
        let mut ss_tot = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
            ss_tot += (y - y_mean).powi(2);
        }

        let slope = if denominator != 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        // Calculate R-squared
        let mut ss_res = 0.0;
        for (i, &y) in values.iter().enumerate() {
            let y_pred = y_mean + slope * (i as f64 - x_mean);
            ss_res += (y - y_pred).powi(2);
        }

        let r_squared = if ss_tot != 0.0 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        // Calculate volatility
        let variance = if n > 1.0 {
            values.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / (n - 1.0)
        } else {
            0.0
        };
        let volatility = variance.sqrt();

        // Calculate max drawdown in window
        let mut peak = values[0];
        let mut max_drawdown = 0.0;
        let mut trough = values[0];

        for &value in &values {
            if value > peak {
                peak = value;
            }
            let dd = if peak > 0.0 {
                (peak - value) / peak * 100.0
            } else {
                0.0
            };
            if dd > max_drawdown {
                max_drawdown = dd;
            }
            if value < trough {
                trough = value;
            }
        }

        // Determine trend direction
        let normalized_slope = if y_mean != 0.0 {
            slope / y_mean * 100.0 // As percentage per period
        } else {
            0.0
        };

        let direction = if normalized_slope > 0.5 {
            TrendDirection::StrongUp
        } else if normalized_slope > 0.1 {
            TrendDirection::Up
        } else if normalized_slope < -0.5 {
            TrendDirection::StrongDown
        } else if normalized_slope < -0.1 {
            TrendDirection::Down
        } else {
            TrendDirection::Flat
        };

        let strength = (normalized_slope.abs() / 1.0).min(1.0) * r_squared;

        Some(TrendAnalysis {
            direction,
            strength,
            slope,
            r_squared,
            period_count: values.len(),
            volatility,
            max_drawdown,
            peak_value: peak,
            trough_value: trough,
        })
    }

    /// Get short-term trend analysis
    pub fn short_term_trend(&mut self) -> Option<TrendAnalysis> {
        if self.short_term_trend.is_none() {
            self.short_term_trend = self.calculate_trend(self.config.short_term_window);
        }
        self.short_term_trend.clone()
    }

    /// Get long-term trend analysis
    pub fn long_term_trend(&mut self) -> Option<TrendAnalysis> {
        if self.long_term_trend.is_none() {
            self.long_term_trend = self.calculate_trend(self.config.long_term_window);
        }
        self.long_term_trend.clone()
    }

    /// Get balance health assessment
    pub fn health_status(&mut self) -> BalanceHealth {
        let drawdown = self.current_drawdown();

        if drawdown >= self.config.drawdown_critical_threshold {
            return BalanceHealth::Critical;
        }

        if drawdown >= self.config.drawdown_warning_threshold {
            return BalanceHealth::Warning;
        }

        // Check trends
        if let Some(trend) = self.short_term_trend() {
            match trend.direction {
                TrendDirection::StrongDown => return BalanceHealth::Warning,
                TrendDirection::Down if trend.strength > 0.5 => return BalanceHealth::Warning,
                TrendDirection::StrongUp => return BalanceHealth::Excellent,
                TrendDirection::Up if trend.strength > 0.5 => return BalanceHealth::Excellent,
                _ => {}
            }
        }

        BalanceHealth::Good
    }

    /// Get recent snapshots
    pub fn recent_snapshots(&self, count: usize) -> Vec<&BalanceSnapshot> {
        self.snapshots.iter().rev().take(count).collect()
    }

    /// Get recent changes
    pub fn recent_changes(&self, count: usize) -> Vec<&BalanceChange> {
        self.changes.iter().rev().take(count).collect()
    }

    /// Get changes by type
    pub fn changes_by_type(&self, change_type: ChangeType) -> Vec<&BalanceChange> {
        self.changes
            .iter()
            .filter(|c| c.change_type == change_type)
            .collect()
    }

    /// Get total by change type
    pub fn total_by_change_type(&self, change_type: ChangeType) -> f64 {
        self.changes
            .iter()
            .filter(|c| c.change_type == change_type)
            .map(|c| c.amount)
            .sum()
    }

    /// Get daily snapshot for a date
    pub fn daily_snapshot(&self, date: &str) -> Option<&BalanceSnapshot> {
        self.daily_snapshots.get(date)
    }

    /// Get statistics
    pub fn stats(&mut self) -> BalanceTrackerStats {
        let trend = self.short_term_trend();

        BalanceTrackerStats {
            current_value: self.current.total_value,
            starting_balance: self.starting_balance,
            high_water_mark: self.high_water_mark,
            low_water_mark: self.low_water_mark,
            current_drawdown: self.current_drawdown(),
            total_return: self.total_return(),
            snapshot_count: self.snapshots.len(),
            change_count: self.changes.len(),
            health: self.health_status(),
            trend_direction: trend.as_ref().map(|t| t.direction),
            trend_strength: trend.as_ref().map(|t| t.strength),
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics for balance tracker
#[derive(Debug, Clone)]
pub struct BalanceTrackerStats {
    pub current_value: f64,
    pub starting_balance: f64,
    pub high_water_mark: f64,
    pub low_water_mark: f64,
    pub current_drawdown: f64,
    pub total_return: f64,
    pub snapshot_count: usize,
    pub change_count: usize,
    pub health: BalanceHealth,
    pub trend_direction: Option<TrendDirection>,
    pub trend_strength: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = BalanceTracker::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_initialize() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        assert_eq!(tracker.starting_balance, 100_000.0);
        assert_eq!(tracker.high_water_mark, 100_000.0);
        assert_eq!(tracker.current.total_value, 100_000.0);
    }

    #[test]
    fn test_update_balance() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        // Simulate balance increase
        let mut snapshot = BalanceSnapshot::new(2000, 105_000.0);
        snapshot.unrealized_pnl = 5_000.0;
        tracker.update(snapshot);

        assert_eq!(tracker.high_water_mark, 105_000.0);
        assert_eq!(tracker.current.total_value, 105_000.0);
    }

    #[test]
    fn test_drawdown_calculation() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        // Set high water mark
        tracker.update(BalanceSnapshot::new(2000, 110_000.0));

        // Decline
        tracker.update(BalanceSnapshot::new(3000, 99_000.0));

        let drawdown = tracker.current_drawdown();
        // Drawdown from 110k to 99k = 10%
        assert!((drawdown - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_total_return() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        tracker.update(BalanceSnapshot::new(2000, 115_000.0));

        let return_pct = tracker.total_return();
        assert!((return_pct - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_record_change() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        tracker.record_change(ChangeType::RealizedProfit, 500.0, 2000, "Closed BTC trade");
        tracker.record_change(ChangeType::Fee, -10.0, 2000, "Commission");

        assert_eq!(tracker.changes.len(), 2);

        let profits = tracker.total_by_change_type(ChangeType::RealizedProfit);
        assert_eq!(profits, 500.0);
    }

    #[test]
    fn test_trend_analysis() {
        let config = BalanceTrackerConfig {
            snapshot_interval: 1, // Take snapshot on every update
            short_term_window: 10,
            ..Default::default()
        };
        let mut tracker = BalanceTracker::with_config(config);
        tracker.initialize(100_000.0, 0);

        // Create upward trend
        for i in 1..=20 {
            tracker.update(BalanceSnapshot::new(i * 1000, 100_000.0 + i as f64 * 500.0));
        }

        let trend = tracker.short_term_trend();
        assert!(trend.is_some());
        let trend = trend.unwrap();
        assert!(matches!(
            trend.direction,
            TrendDirection::Up | TrendDirection::StrongUp
        ));
        assert!(trend.slope > 0.0);
    }

    #[test]
    fn test_health_status() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        // Healthy state
        tracker.update(BalanceSnapshot::new(2000, 102_000.0));
        assert_eq!(tracker.health_status(), BalanceHealth::Good);

        // Warning state (drawdown)
        tracker.update(BalanceSnapshot::new(3000, 95_000.0)); // 5% from HWM
        let health = tracker.health_status();
        assert!(matches!(
            health,
            BalanceHealth::Warning | BalanceHealth::Good
        ));
    }

    #[test]
    fn test_leverage_calculation() {
        let mut snapshot = BalanceSnapshot::new(1000, 100_000.0);
        snapshot.gross_exposure = 250_000.0;

        assert!((snapshot.leverage() - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_long_short_ratio() {
        let mut snapshot = BalanceSnapshot::new(1000, 100_000.0);
        snapshot.long_exposure = 150_000.0;
        snapshot.short_exposure = 50_000.0;

        let ratio = snapshot.long_short_ratio().unwrap();
        assert!((ratio - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_margin_utilization() {
        let mut snapshot = BalanceSnapshot::new(1000, 100_000.0);
        snapshot.margin_used = 40_000.0;

        assert!((snapshot.margin_utilization() - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_daily_snapshot() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        tracker.record_daily_snapshot("2024-01-15");

        let daily = tracker.daily_snapshot("2024-01-15");
        assert!(daily.is_some());
        assert_eq!(daily.unwrap().total_value, 100_000.0);
    }

    #[test]
    fn test_recent_snapshots() {
        let config = BalanceTrackerConfig {
            snapshot_interval: 1,
            ..Default::default()
        };
        let mut tracker = BalanceTracker::with_config(config);
        tracker.initialize(100_000.0, 0);

        for i in 1..=10 {
            tracker.update(BalanceSnapshot::new(i * 1000, 100_000.0 + i as f64 * 100.0));
        }

        let recent = tracker.recent_snapshots(5);
        assert_eq!(recent.len(), 5);
    }

    #[test]
    fn test_stats() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        tracker.update(BalanceSnapshot::new(2000, 110_000.0));
        tracker.record_change(ChangeType::RealizedProfit, 500.0, 2000, "test");

        let stats = tracker.stats();
        assert_eq!(stats.current_value, 110_000.0);
        assert_eq!(stats.starting_balance, 100_000.0);
        assert_eq!(stats.high_water_mark, 110_000.0);
        assert!((stats.total_return - 10.0).abs() < 0.1);
        assert_eq!(stats.change_count, 1);
    }

    #[test]
    fn test_changes_by_type() {
        let mut tracker = BalanceTracker::new();
        tracker.initialize(100_000.0, 1000);

        tracker.record_change(ChangeType::RealizedProfit, 500.0, 2000, "trade1");
        tracker.record_change(ChangeType::Fee, -10.0, 2000, "fee1");
        tracker.record_change(ChangeType::RealizedProfit, 300.0, 3000, "trade2");
        tracker.record_change(ChangeType::Fee, -5.0, 3000, "fee2");

        let profits = tracker.changes_by_type(ChangeType::RealizedProfit);
        assert_eq!(profits.len(), 2);

        let fees = tracker.changes_by_type(ChangeType::Fee);
        assert_eq!(fees.len(), 2);

        let total_fees = tracker.total_by_change_type(ChangeType::Fee);
        assert_eq!(total_fees, -15.0);
    }

    #[test]
    fn test_min_change_filter() {
        let config = BalanceTrackerConfig {
            min_change_amount: 1.0,
            ..Default::default()
        };
        let mut tracker = BalanceTracker::with_config(config);
        tracker.initialize(100_000.0, 1000);

        // This should be filtered out
        tracker.record_change(ChangeType::Fee, 0.005, 2000, "tiny fee");

        // This should be recorded
        tracker.record_change(ChangeType::Fee, 1.50, 2000, "normal fee");

        assert_eq!(tracker.changes.len(), 1);
    }
}
