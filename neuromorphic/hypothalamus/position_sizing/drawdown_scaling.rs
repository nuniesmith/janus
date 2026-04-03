//! Reduce size during drawdown
//!
//! Part of the Hypothalamus region
//! Component: position_sizing
//!
//! This module implements drawdown-based position scaling that automatically
//! reduces position sizes during drawdown periods to protect capital and
//! accelerate recovery.

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Drawdown severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DrawdownLevel {
    /// No significant drawdown (< 5%)
    None,
    /// Minor drawdown (5-10%)
    Minor,
    /// Moderate drawdown (10-20%)
    Moderate,
    /// Significant drawdown (20-30%)
    Significant,
    /// Severe drawdown (30-40%)
    Severe,
    /// Critical drawdown (> 40%)
    Critical,
}

impl Default for DrawdownLevel {
    fn default() -> Self {
        DrawdownLevel::None
    }
}

impl DrawdownLevel {
    /// Get the scaling factor for this drawdown level
    pub fn scaling_factor(&self) -> f64 {
        match self {
            DrawdownLevel::None => 1.0,
            DrawdownLevel::Minor => 0.85,
            DrawdownLevel::Moderate => 0.65,
            DrawdownLevel::Significant => 0.45,
            DrawdownLevel::Severe => 0.25,
            DrawdownLevel::Critical => 0.10,
        }
    }

    /// Get the drawdown percentage range for this level
    pub fn threshold_range(&self) -> (f64, f64) {
        match self {
            DrawdownLevel::None => (0.0, 0.05),
            DrawdownLevel::Minor => (0.05, 0.10),
            DrawdownLevel::Moderate => (0.10, 0.20),
            DrawdownLevel::Significant => (0.20, 0.30),
            DrawdownLevel::Severe => (0.30, 0.40),
            DrawdownLevel::Critical => (0.40, 1.0),
        }
    }

    /// Check if this level requires trading halt consideration
    pub fn requires_halt_consideration(&self) -> bool {
        matches!(self, DrawdownLevel::Severe | DrawdownLevel::Critical)
    }
}

/// Recovery phase classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecoveryPhase {
    /// Still in drawdown, not recovering
    InDrawdown,
    /// Initial recovery (< 25% recovered)
    EarlyRecovery,
    /// Mid recovery (25-50% recovered)
    MidRecovery,
    /// Late recovery (50-75% recovered)
    LateRecovery,
    /// Near full recovery (> 75% recovered)
    NearRecovery,
    /// Fully recovered, at new high
    FullyRecovered,
}

impl Default for RecoveryPhase {
    fn default() -> Self {
        RecoveryPhase::FullyRecovered
    }
}

impl RecoveryPhase {
    /// Get scaling adjustment for recovery phase
    /// Gradually increases position size as recovery progresses
    pub fn scaling_adjustment(&self) -> f64 {
        match self {
            RecoveryPhase::InDrawdown => 0.0,     // Use drawdown level scaling
            RecoveryPhase::EarlyRecovery => 0.15, // Slight increase
            RecoveryPhase::MidRecovery => 0.30,   // Moderate increase
            RecoveryPhase::LateRecovery => 0.50,  // Larger increase
            RecoveryPhase::NearRecovery => 0.75,  // Almost full
            RecoveryPhase::FullyRecovered => 1.0, // Full size
        }
    }
}

/// Configuration for drawdown scaling
#[derive(Debug, Clone)]
pub struct DrawdownScalingConfig {
    /// Custom drawdown thresholds [minor, moderate, significant, severe, critical]
    pub thresholds: [f64; 5],
    /// Custom scaling factors for each level [none, minor, moderate, significant, severe, critical]
    pub scaling_factors: [f64; 6],
    /// Enable gradual recovery (vs immediate return to full size)
    pub gradual_recovery: bool,
    /// Number of consecutive positive days required to start recovery
    pub recovery_confirmation_days: usize,
    /// Minimum days in reduced size mode before increasing
    pub min_drawdown_duration_days: usize,
    /// Use time-weighted recovery (slower recovery for longer drawdowns)
    pub time_weighted_recovery: bool,
    /// Recovery speed multiplier (higher = faster recovery)
    pub recovery_speed: f64,
    /// Halt trading at this drawdown level
    pub halt_threshold: Option<f64>,
    /// Track intraday drawdowns
    pub track_intraday: bool,
    /// Use max drawdown over trailing period (days) instead of from peak
    pub trailing_period: Option<usize>,
    /// Psychological comfort threshold - alert when approaching this
    pub comfort_threshold: f64,
}

impl Default for DrawdownScalingConfig {
    fn default() -> Self {
        Self {
            thresholds: [0.05, 0.10, 0.20, 0.30, 0.40],
            scaling_factors: [1.0, 0.85, 0.65, 0.45, 0.25, 0.10],
            gradual_recovery: true,
            recovery_confirmation_days: 3,
            min_drawdown_duration_days: 5,
            time_weighted_recovery: true,
            recovery_speed: 1.0,
            halt_threshold: Some(0.50),
            track_intraday: true,
            trailing_period: None,
            comfort_threshold: 0.15,
        }
    }
}

/// Equity snapshot for tracking
#[derive(Debug, Clone)]
pub struct EquitySnapshot {
    /// Timestamp (Unix seconds)
    pub timestamp: u64,
    /// Equity value
    pub equity: f64,
    /// Daily high
    pub daily_high: f64,
    /// Daily low
    pub daily_low: f64,
    /// Drawdown from peak at this point
    pub drawdown: f64,
    /// Notes (e.g., "New high", "Recovery started")
    pub notes: Option<String>,
}

impl EquitySnapshot {
    pub fn new(timestamp: u64, equity: f64) -> Self {
        Self {
            timestamp,
            equity,
            daily_high: equity,
            daily_low: equity,
            drawdown: 0.0,
            notes: None,
        }
    }
}

/// Drawdown event record
#[derive(Debug, Clone)]
pub struct DrawdownEvent {
    /// Event ID
    pub id: u64,
    /// Start timestamp
    pub start_time: u64,
    /// End timestamp (None if ongoing)
    pub end_time: Option<u64>,
    /// Peak equity before drawdown
    pub peak_equity: f64,
    /// Trough equity (lowest point)
    pub trough_equity: f64,
    /// Maximum drawdown percentage
    pub max_drawdown: f64,
    /// Current drawdown percentage
    pub current_drawdown: f64,
    /// Duration in days
    pub duration_days: usize,
    /// Deepest level reached
    pub deepest_level: DrawdownLevel,
    /// Recovery phase
    pub recovery_phase: RecoveryPhase,
    /// Recovery percentage (0-1)
    pub recovery_pct: f64,
    /// Days spent at each level
    pub days_at_levels: [usize; 6],
}

impl DrawdownEvent {
    pub fn new(id: u64, start_time: u64, peak_equity: f64) -> Self {
        Self {
            id,
            start_time,
            end_time: None,
            peak_equity,
            trough_equity: peak_equity,
            max_drawdown: 0.0,
            current_drawdown: 0.0,
            duration_days: 0,
            deepest_level: DrawdownLevel::None,
            recovery_phase: RecoveryPhase::InDrawdown,
            recovery_pct: 0.0,
            days_at_levels: [0; 6],
        }
    }

    /// Check if drawdown is ongoing
    pub fn is_active(&self) -> bool {
        self.end_time.is_none()
    }

    /// Calculate time to recovery at current rate (estimated days)
    pub fn estimated_recovery_days(&self, avg_daily_return: f64) -> Option<f64> {
        if avg_daily_return <= 0.0 || self.current_drawdown <= 0.0 {
            return None;
        }

        // Amount needed to recover (as multiple)
        let recovery_needed = 1.0 / (1.0 - self.current_drawdown) - 1.0;

        // Days = ln(1 + recovery_needed) / ln(1 + daily_return)
        Some((1.0 + recovery_needed).ln() / (1.0 + avg_daily_return).ln())
    }
}

/// Scaling result
#[derive(Debug, Clone)]
pub struct DrawdownScalingResult {
    /// Current drawdown percentage
    pub current_drawdown: f64,
    /// Current drawdown level
    pub level: DrawdownLevel,
    /// Recovery phase
    pub recovery_phase: RecoveryPhase,
    /// Base scaling factor (from drawdown level)
    pub base_scale: f64,
    /// Recovery adjustment
    pub recovery_adjustment: f64,
    /// Final scaling factor
    pub final_scale: f64,
    /// Whether trading should be halted
    pub halt_trading: bool,
    /// Days in current drawdown
    pub days_in_drawdown: usize,
    /// Approaching comfort threshold warning
    pub comfort_warning: bool,
    /// Peak equity
    pub peak_equity: f64,
    /// Current equity
    pub current_equity: f64,
    /// Amount to recover
    pub amount_to_recover: f64,
    /// Recovery percentage
    pub recovery_pct: f64,
    /// Notes/warnings
    pub notes: Vec<String>,
}

impl Default for DrawdownScalingResult {
    fn default() -> Self {
        Self {
            current_drawdown: 0.0,
            level: DrawdownLevel::None,
            recovery_phase: RecoveryPhase::FullyRecovered,
            base_scale: 1.0,
            recovery_adjustment: 1.0,
            final_scale: 1.0,
            halt_trading: false,
            days_in_drawdown: 0,
            comfort_warning: false,
            peak_equity: 0.0,
            current_equity: 0.0,
            amount_to_recover: 0.0,
            recovery_pct: 0.0,
            notes: Vec::new(),
        }
    }
}

/// Reduce size during drawdown
pub struct DrawdownScaling {
    /// Configuration
    config: DrawdownScalingConfig,
    /// Current peak equity (high-water mark)
    peak_equity: f64,
    /// Current equity
    current_equity: f64,
    /// Equity history
    equity_history: VecDeque<EquitySnapshot>,
    /// Current drawdown event (if any)
    current_event: Option<DrawdownEvent>,
    /// Historical drawdown events
    event_history: Vec<DrawdownEvent>,
    /// Event counter
    event_counter: u64,
    /// Consecutive positive days
    consecutive_positive_days: usize,
    /// Days in current drawdown
    days_in_drawdown: usize,
    /// Last result
    last_result: Option<DrawdownScalingResult>,
    /// Total calculations
    calculations_count: usize,
    /// Maximum history size
    max_history_size: usize,
}

impl Default for DrawdownScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl DrawdownScaling {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            config: DrawdownScalingConfig::default(),
            peak_equity: 0.0,
            current_equity: 0.0,
            equity_history: VecDeque::new(),
            current_event: None,
            event_history: Vec::new(),
            event_counter: 0,
            consecutive_positive_days: 0,
            days_in_drawdown: 0,
            last_result: None,
            calculations_count: 0,
            max_history_size: 1000,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: DrawdownScalingConfig) -> Self {
        Self {
            config,
            peak_equity: 0.0,
            current_equity: 0.0,
            equity_history: VecDeque::new(),
            current_event: None,
            event_history: Vec::new(),
            event_counter: 0,
            consecutive_positive_days: 0,
            days_in_drawdown: 0,
            last_result: None,
            calculations_count: 0,
            max_history_size: 1000,
        }
    }

    /// Initialize with starting equity
    pub fn initialize(&mut self, equity: f64, timestamp: u64) {
        self.peak_equity = equity;
        self.current_equity = equity;
        self.equity_history
            .push_back(EquitySnapshot::new(timestamp, equity));
    }

    /// Update equity and calculate scaling
    pub fn update_equity(&mut self, equity: f64, timestamp: u64) -> DrawdownScalingResult {
        self.calculations_count += 1;

        let previous_equity = self.current_equity;
        self.current_equity = equity;

        // Check if this is a positive day
        if equity > previous_equity {
            self.consecutive_positive_days += 1;
        } else {
            self.consecutive_positive_days = 0;
        }

        // Update peak if we have a new high
        let mut snapshot = EquitySnapshot::new(timestamp, equity);
        if equity > self.peak_equity {
            self.peak_equity = equity;
            snapshot.notes = Some("New high".to_string());

            // Close any active drawdown event
            if let Some(ref mut event) = self.current_event {
                event.end_time = Some(timestamp);
                event.recovery_phase = RecoveryPhase::FullyRecovered;
                event.recovery_pct = 1.0;
                self.event_history.push(event.clone());
            }
            self.current_event = None;
            self.days_in_drawdown = 0;
            self.consecutive_positive_days = 0;
        }

        // Calculate current drawdown
        let drawdown = if self.peak_equity > 0.0 {
            (self.peak_equity - equity) / self.peak_equity
        } else {
            0.0
        };
        snapshot.drawdown = drawdown;

        // Determine drawdown level
        let level = self.classify_drawdown(drawdown);

        // Handle drawdown event tracking
        if drawdown > self.config.thresholds[0] {
            self.days_in_drawdown += 1;

            if self.current_event.is_none() {
                // Start new drawdown event
                self.event_counter += 1;
                self.current_event = Some(DrawdownEvent::new(
                    self.event_counter,
                    timestamp,
                    self.peak_equity,
                ));
            }

            if let Some(ref mut event) = self.current_event {
                // Update trough if deeper
                if equity < event.trough_equity {
                    event.trough_equity = equity;
                    event.max_drawdown = (event.peak_equity - equity) / event.peak_equity;
                }
                event.current_drawdown = drawdown;
                event.duration_days = self.days_in_drawdown;

                // Update deepest level
                if level > event.deepest_level {
                    event.deepest_level = level;
                }

                // Track days at each level
                event.days_at_levels[level as usize] += 1;

                // Calculate recovery percentage
                if event.max_drawdown > 0.0 {
                    let recovered = event.max_drawdown - drawdown;
                    event.recovery_pct = recovered / event.max_drawdown;
                }

                // Determine recovery phase
                event.recovery_phase = self.classify_recovery(event.recovery_pct, drawdown);
            }
        }

        // Store snapshot
        self.equity_history.push_back(snapshot);
        if self.equity_history.len() > self.max_history_size {
            self.equity_history.pop_front();
        }

        // Calculate scaling
        self.calculate_scaling(drawdown, level)
    }

    /// Classify drawdown level based on percentage
    fn classify_drawdown(&self, drawdown: f64) -> DrawdownLevel {
        if drawdown < self.config.thresholds[0] {
            DrawdownLevel::None
        } else if drawdown < self.config.thresholds[1] {
            DrawdownLevel::Minor
        } else if drawdown < self.config.thresholds[2] {
            DrawdownLevel::Moderate
        } else if drawdown < self.config.thresholds[3] {
            DrawdownLevel::Significant
        } else if drawdown < self.config.thresholds[4] {
            DrawdownLevel::Severe
        } else {
            DrawdownLevel::Critical
        }
    }

    /// Classify recovery phase
    fn classify_recovery(&self, recovery_pct: f64, current_drawdown: f64) -> RecoveryPhase {
        // Check if we're actually recovering (consecutive positive days)
        if self.consecutive_positive_days < self.config.recovery_confirmation_days {
            return RecoveryPhase::InDrawdown;
        }

        // Must be in a meaningful drawdown to be "recovering"
        if current_drawdown < self.config.thresholds[0] {
            return RecoveryPhase::FullyRecovered;
        }

        if recovery_pct < 0.25 {
            RecoveryPhase::EarlyRecovery
        } else if recovery_pct < 0.50 {
            RecoveryPhase::MidRecovery
        } else if recovery_pct < 0.75 {
            RecoveryPhase::LateRecovery
        } else if recovery_pct < 1.0 {
            RecoveryPhase::NearRecovery
        } else {
            RecoveryPhase::FullyRecovered
        }
    }

    /// Calculate the scaling factor
    fn calculate_scaling(&mut self, drawdown: f64, level: DrawdownLevel) -> DrawdownScalingResult {
        let mut result = DrawdownScalingResult::default();
        result.current_drawdown = drawdown;
        result.level = level;
        result.peak_equity = self.peak_equity;
        result.current_equity = self.current_equity;
        result.amount_to_recover = self.peak_equity - self.current_equity;
        result.days_in_drawdown = self.days_in_drawdown;

        // Check halt threshold
        if let Some(halt) = self.config.halt_threshold {
            if drawdown >= halt {
                result.halt_trading = true;
                result.final_scale = 0.0;
                result.notes.push(format!(
                    "TRADING HALTED: Drawdown {:.1}% exceeds halt threshold {:.1}%",
                    drawdown * 100.0,
                    halt * 100.0
                ));
                self.last_result = Some(result.clone());
                return result;
            }
        }

        // Get base scaling factor
        let base_scale = self.config.scaling_factors[level as usize];
        result.base_scale = base_scale;

        // Determine recovery phase and adjustment
        let recovery_phase = if let Some(ref event) = self.current_event {
            result.recovery_pct = event.recovery_pct;
            event.recovery_phase
        } else {
            result.recovery_pct = 1.0;
            RecoveryPhase::FullyRecovered
        };
        result.recovery_phase = recovery_phase;

        // Calculate recovery adjustment
        let mut recovery_adjustment = 1.0;
        if self.config.gradual_recovery && recovery_phase != RecoveryPhase::InDrawdown {
            let base_adjustment = recovery_phase.scaling_adjustment();

            // Apply time-weighted recovery if enabled
            if self.config.time_weighted_recovery && self.days_in_drawdown > 0 {
                // Longer drawdowns recover more slowly
                let time_factor = 1.0 / (1.0 + (self.days_in_drawdown as f64 / 30.0).ln());
                recovery_adjustment = base_adjustment * time_factor * self.config.recovery_speed;
            } else {
                recovery_adjustment = base_adjustment * self.config.recovery_speed;
            }
        }
        result.recovery_adjustment = recovery_adjustment;

        // Calculate final scale
        // If recovering, blend between drawdown scale and full scale based on recovery
        if recovery_phase != RecoveryPhase::InDrawdown {
            let recovery_blend = result.recovery_pct.min(1.0);
            result.final_scale =
                base_scale + (1.0 - base_scale) * recovery_blend * recovery_adjustment;
        } else {
            result.final_scale = base_scale;
        }

        // Ensure minimum duration in drawdown mode
        if self.days_in_drawdown < self.config.min_drawdown_duration_days
            && self.days_in_drawdown > 0
        {
            // Don't increase scale too quickly
            result.final_scale = result.final_scale.min(base_scale);
            result.notes.push(format!(
                "Maintaining reduced size for {} more days (min duration)",
                self.config.min_drawdown_duration_days - self.days_in_drawdown
            ));
        }

        // Check comfort threshold warning
        if drawdown >= self.config.comfort_threshold && drawdown < self.config.thresholds[2] {
            result.comfort_warning = true;
            result.notes.push(format!(
                "Approaching comfort threshold ({:.1}%)",
                self.config.comfort_threshold * 100.0
            ));
        }

        // Add level-specific notes
        if level.requires_halt_consideration() {
            result.notes.push(format!(
                "Drawdown level {:?} - consider additional risk reduction",
                level
            ));
        }

        // Clamp final scale
        result.final_scale = result.final_scale.clamp(0.0, 1.0);

        self.last_result = Some(result.clone());
        result
    }

    /// Get recommended position size
    pub fn get_position_size(&self, base_size: f64) -> f64 {
        if let Some(ref result) = self.last_result {
            base_size * result.final_scale
        } else {
            base_size
        }
    }

    /// Get current drawdown percentage
    pub fn current_drawdown(&self) -> f64 {
        if self.peak_equity > 0.0 {
            (self.peak_equity - self.current_equity) / self.peak_equity
        } else {
            0.0
        }
    }

    /// Get current drawdown level
    pub fn current_level(&self) -> DrawdownLevel {
        self.classify_drawdown(self.current_drawdown())
    }

    /// Get peak equity (high-water mark)
    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    /// Get current equity
    pub fn current_equity(&self) -> f64 {
        self.current_equity
    }

    /// Get days in current drawdown
    pub fn days_in_drawdown(&self) -> usize {
        self.days_in_drawdown
    }

    /// Get current event details
    pub fn current_event(&self) -> Option<&DrawdownEvent> {
        self.current_event.as_ref()
    }

    /// Get historical drawdown events
    pub fn event_history(&self) -> &[DrawdownEvent] {
        &self.event_history
    }

    /// Get last calculation result
    pub fn last_result(&self) -> Option<&DrawdownScalingResult> {
        self.last_result.as_ref()
    }

    /// Get maximum historical drawdown
    pub fn max_historical_drawdown(&self) -> f64 {
        self.event_history
            .iter()
            .map(|e| e.max_drawdown)
            .fold(0.0, f64::max)
            .max(
                self.current_event
                    .as_ref()
                    .map(|e| e.max_drawdown)
                    .unwrap_or(0.0),
            )
    }

    /// Get average drawdown duration (in days)
    pub fn avg_drawdown_duration(&self) -> f64 {
        if self.event_history.is_empty() {
            return 0.0;
        }

        let total_days: usize = self.event_history.iter().map(|e| e.duration_days).sum();
        total_days as f64 / self.event_history.len() as f64
    }

    /// Get equity curve statistics
    pub fn equity_statistics(&self) -> EquityStatistics {
        let equities: Vec<f64> = self.equity_history.iter().map(|s| s.equity).collect();

        if equities.len() < 2 {
            return EquityStatistics::default();
        }

        let returns: Vec<f64> = equities.windows(2).map(|w| (w[1] - w[0]) / w[0]).collect();

        let avg_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - avg_return).powi(2))
            .sum::<f64>()
            / (returns.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let positive_returns = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = positive_returns as f64 / returns.len() as f64;

        let max_drawdown = self
            .equity_history
            .iter()
            .map(|s| s.drawdown)
            .fold(0.0, f64::max);

        EquityStatistics {
            avg_daily_return: avg_return,
            daily_std_dev: std_dev,
            win_rate,
            max_drawdown,
            total_return: if equities.first().map(|e| *e > 0.0).unwrap_or(false) {
                (equities.last().unwrap_or(&0.0) - equities.first().unwrap_or(&1.0))
                    / equities.first().unwrap_or(&1.0)
            } else {
                0.0
            },
            sharpe_ratio: if std_dev > 0.0 {
                avg_return / std_dev * (252.0_f64).sqrt()
            } else {
                0.0
            },
            calmar_ratio: if max_drawdown > 0.0 {
                (avg_return * 252.0) / max_drawdown
            } else {
                0.0
            },
        }
    }

    /// Reset drawdown tracking (e.g., for new trading period)
    pub fn reset(&mut self, starting_equity: f64, timestamp: u64) {
        self.peak_equity = starting_equity;
        self.current_equity = starting_equity;
        self.equity_history.clear();
        self.current_event = None;
        self.consecutive_positive_days = 0;
        self.days_in_drawdown = 0;
        self.last_result = None;
        self.equity_history
            .push_back(EquitySnapshot::new(timestamp, starting_equity));
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via update_equity
        Ok(())
    }
}

/// Equity curve statistics
#[derive(Debug, Clone, Default)]
pub struct EquityStatistics {
    pub avg_daily_return: f64,
    pub daily_std_dev: f64,
    pub win_rate: f64,
    pub max_drawdown: f64,
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub calmar_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = DrawdownScaling::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_initialization() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        assert_eq!(scaler.peak_equity(), 100_000.0);
        assert_eq!(scaler.current_equity(), 100_000.0);
        assert_eq!(scaler.current_drawdown(), 0.0);
    }

    #[test]
    fn test_drawdown_detection() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // 10% drawdown
        let result = scaler.update_equity(90_000.0, 1);

        assert!((result.current_drawdown - 0.10).abs() < 0.001);
        assert_eq!(result.level, DrawdownLevel::Moderate);
    }

    #[test]
    fn test_drawdown_levels() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // Test each level
        let _ = scaler.update_equity(97_000.0, 1);
        assert_eq!(scaler.current_level(), DrawdownLevel::None);

        let _ = scaler.update_equity(93_000.0, 2);
        assert_eq!(scaler.current_level(), DrawdownLevel::Minor);

        let _ = scaler.update_equity(85_000.0, 3);
        assert_eq!(scaler.current_level(), DrawdownLevel::Moderate);

        let _ = scaler.update_equity(75_000.0, 4);
        assert_eq!(scaler.current_level(), DrawdownLevel::Significant);

        let _ = scaler.update_equity(65_000.0, 5);
        assert_eq!(scaler.current_level(), DrawdownLevel::Severe);

        let _ = scaler.update_equity(55_000.0, 6);
        assert_eq!(scaler.current_level(), DrawdownLevel::Critical);
    }

    #[test]
    fn test_scaling_factors() {
        assert_eq!(DrawdownLevel::None.scaling_factor(), 1.0);
        assert!(DrawdownLevel::Minor.scaling_factor() < 1.0);
        assert!(DrawdownLevel::Critical.scaling_factor() < DrawdownLevel::Severe.scaling_factor());
    }

    #[test]
    fn test_new_high_resets_drawdown() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // Drawdown
        let _ = scaler.update_equity(90_000.0, 1);
        assert!(scaler.current_drawdown() > 0.0);

        // New high
        let result = scaler.update_equity(110_000.0, 2);

        assert_eq!(result.current_drawdown, 0.0);
        assert_eq!(result.level, DrawdownLevel::None);
        assert_eq!(scaler.peak_equity(), 110_000.0);
        assert_eq!(scaler.days_in_drawdown(), 0);
    }

    #[test]
    fn test_position_sizing() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // No drawdown - full size
        let _ = scaler.update_equity(100_000.0, 1);
        assert_eq!(scaler.get_position_size(1000.0), 1000.0);

        // 15% drawdown (moderate)
        let _ = scaler.update_equity(85_000.0, 2);
        let size = scaler.get_position_size(1000.0);
        assert!(size < 1000.0);
        assert!(size > 500.0);
    }

    #[test]
    fn test_halt_threshold() {
        let mut config = DrawdownScalingConfig::default();
        config.halt_threshold = Some(0.30);

        let mut scaler = DrawdownScaling::with_config(config);
        scaler.initialize(100_000.0, 0);

        // 35% drawdown - should halt
        let result = scaler.update_equity(65_000.0, 1);

        assert!(result.halt_trading);
        assert_eq!(result.final_scale, 0.0);
    }

    #[test]
    fn test_drawdown_event_tracking() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // Enter drawdown
        let _ = scaler.update_equity(90_000.0, 1);
        assert!(scaler.current_event().is_some());

        // Deeper drawdown
        let _ = scaler.update_equity(80_000.0, 2);

        let event = scaler.current_event().unwrap();
        assert!(event.is_active());
        assert!((event.max_drawdown - 0.20).abs() < 0.01);
        assert_eq!(event.trough_equity, 80_000.0);

        // Recovery to new high
        let _ = scaler.update_equity(110_000.0, 3);
        assert!(scaler.current_event().is_none());
        assert_eq!(scaler.event_history().len(), 1);
    }

    #[test]
    fn test_recovery_phases() {
        let mut config = DrawdownScalingConfig::default();
        config.recovery_confirmation_days = 1; // Quick recovery for test

        let mut scaler = DrawdownScaling::with_config(config);
        scaler.initialize(100_000.0, 0);

        // Create 20% drawdown
        let _ = scaler.update_equity(80_000.0, 1);

        // Start recovering
        let result = scaler.update_equity(85_000.0, 2);

        // Should be in early recovery
        assert!(result.recovery_pct > 0.0);
    }

    #[test]
    fn test_days_in_drawdown() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        for i in 1..=5 {
            let _ = scaler.update_equity(90_000.0, i);
        }

        assert_eq!(scaler.days_in_drawdown(), 5);
    }

    #[test]
    fn test_comfort_threshold_warning() {
        let mut config = DrawdownScalingConfig::default();
        config.comfort_threshold = 0.08;

        let mut scaler = DrawdownScaling::with_config(config);
        scaler.initialize(100_000.0, 0);

        // Just past comfort threshold but not moderate
        let result = scaler.update_equity(91_000.0, 1);

        assert!(result.comfort_warning);
    }

    #[test]
    fn test_max_historical_drawdown() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // First drawdown: 15%
        let _ = scaler.update_equity(85_000.0, 1);
        let _ = scaler.update_equity(110_000.0, 2); // Recover

        // Second drawdown: 25%
        let _ = scaler.update_equity(82_500.0, 3);
        let _ = scaler.update_equity(120_000.0, 4); // Recover

        let max_dd = scaler.max_historical_drawdown();
        assert!((max_dd - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_equity_statistics() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // Simulate some equity changes
        let _ = scaler.update_equity(101_000.0, 1);
        let _ = scaler.update_equity(102_000.0, 2);
        let _ = scaler.update_equity(100_500.0, 3);
        let _ = scaler.update_equity(103_000.0, 4);

        let stats = scaler.equity_statistics();

        assert!(stats.avg_daily_return > 0.0);
        assert!(stats.win_rate > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);
        let _ = scaler.update_equity(90_000.0, 1);

        scaler.reset(50_000.0, 100);

        assert_eq!(scaler.peak_equity(), 50_000.0);
        assert_eq!(scaler.current_equity(), 50_000.0);
        assert_eq!(scaler.days_in_drawdown(), 0);
        assert!(scaler.current_event().is_none());
    }

    #[test]
    fn test_level_comparison() {
        assert!(DrawdownLevel::Critical > DrawdownLevel::Severe);
        assert!(DrawdownLevel::Severe > DrawdownLevel::None);
    }

    #[test]
    fn test_estimated_recovery_days() {
        let event = DrawdownEvent {
            id: 1,
            start_time: 0,
            end_time: None,
            peak_equity: 100_000.0,
            trough_equity: 80_000.0,
            max_drawdown: 0.20,
            current_drawdown: 0.10, // Half recovered
            duration_days: 10,
            deepest_level: DrawdownLevel::Moderate,
            recovery_phase: RecoveryPhase::MidRecovery,
            recovery_pct: 0.50,
            days_at_levels: [0; 6],
        };

        // With 1% daily return, should estimate some recovery time
        let days = event.estimated_recovery_days(0.01);
        assert!(days.is_some());
        assert!(days.unwrap() > 0.0);
    }

    #[test]
    fn test_gradual_recovery_disabled() {
        let mut config = DrawdownScalingConfig::default();
        config.gradual_recovery = false;

        let mut scaler = DrawdownScaling::with_config(config);
        scaler.initialize(100_000.0, 0);

        // Create drawdown
        let _ = scaler.update_equity(85_000.0, 1);

        // Partial recovery
        let result = scaler.update_equity(90_000.0, 2);

        // Without gradual recovery, should still use drawdown level scaling
        assert_eq!(result.recovery_adjustment, 1.0);
    }

    #[test]
    fn test_days_at_levels_tracking() {
        let mut scaler = DrawdownScaling::new();
        scaler.initialize(100_000.0, 0);

        // Spend time at moderate level
        for i in 1..=5 {
            let _ = scaler.update_equity(88_000.0, i);
        }

        let event = scaler.current_event().unwrap();
        assert!(event.days_at_levels[DrawdownLevel::Moderate as usize] > 0);
    }
}
