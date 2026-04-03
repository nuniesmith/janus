//! Adaptive homeostatic setpoint tracking
//!
//! Part of the Hypothalamus region
//! Component: homeostasis
//!
//! Implements adaptive setpoint management for portfolio homeostasis.
//! A setpoint represents the desired "resting state" of a regulated
//! variable (e.g., target portfolio volatility, target exposure,
//! target drawdown limit). The setpoint adapts over time based on:
//!
//! - Recent performance (good performance → relax setpoint slightly)
//! - Market regime changes (high volatility → tighten setpoint)
//! - Operator overrides (manual setpoint adjustments)
//! - Mean reversion toward a long-term baseline
//!
//! This mirrors biological homeostasis where setpoints (e.g., body
//! temperature) can shift under sustained stress or adaptation.

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// The type of regulated variable this setpoint controls
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SetpointType {
    /// Target portfolio volatility (annualized, e.g., 0.15 = 15%)
    Volatility,
    /// Target net exposure as a fraction of capital (e.g., 0.80 = 80%)
    Exposure,
    /// Maximum acceptable drawdown (e.g., 0.10 = 10%)
    MaxDrawdown,
    /// Target cash reserve fraction (e.g., 0.20 = 20%)
    CashReserve,
    /// Target position concentration limit (max fraction in single asset)
    Concentration,
    /// Target Sharpe ratio (performance floor)
    SharpeFloor,
    /// Custom user-defined setpoint
    Custom,
}

/// Direction of setpoint adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdjustmentDirection {
    /// Tighten the setpoint (more conservative)
    Tighten,
    /// Relax the setpoint (more aggressive)
    Relax,
    /// No change
    Hold,
}

/// A record of a setpoint adjustment for audit trail
#[derive(Debug, Clone)]
pub struct SetpointAdjustment {
    /// The step at which the adjustment occurred
    pub step: u64,
    /// Previous setpoint value
    pub previous_value: f64,
    /// New setpoint value
    pub new_value: f64,
    /// Direction of the adjustment
    pub direction: AdjustmentDirection,
    /// Reason for the adjustment
    pub reason: String,
}

/// Configuration for setpoint adaptation
#[derive(Debug, Clone)]
pub struct SetpointConfig {
    /// The type of regulated variable
    pub setpoint_type: SetpointType,
    /// Initial setpoint value
    pub initial_value: f64,
    /// Long-term baseline value (setpoint reverts toward this over time)
    pub baseline_value: f64,
    /// Hard minimum for the setpoint (can never go below this)
    pub hard_min: f64,
    /// Hard maximum for the setpoint (can never exceed this)
    pub hard_max: f64,
    /// Adaptation rate: how quickly the setpoint adjusts (0.0 - 1.0)
    pub adaptation_rate: f64,
    /// Mean reversion rate toward baseline (0.0 - 1.0)
    pub mean_reversion_rate: f64,
    /// Tightening multiplier applied when performance is bad
    pub tightening_multiplier: f64,
    /// Relaxation multiplier applied when performance is good
    pub relaxation_multiplier: f64,
    /// Number of recent observations to consider for adaptation
    pub observation_window: usize,
    /// Performance threshold: below this, tighten; above, relax
    pub performance_threshold: f64,
    /// Cooldown steps between automatic adjustments
    pub cooldown_steps: u64,
    /// Maximum adjustment per step (absolute value)
    pub max_step_adjustment: f64,
    /// Whether to enable automatic adaptation (can be disabled for manual-only)
    pub auto_adapt: bool,
    /// Whether the setpoint represents a ceiling (true) or a floor (false)
    /// Ceiling: actual value should stay below setpoint (e.g., max drawdown)
    /// Floor: actual value should stay above setpoint (e.g., Sharpe floor)
    pub is_ceiling: bool,
}

impl Default for SetpointConfig {
    fn default() -> Self {
        Self {
            setpoint_type: SetpointType::Volatility,
            initial_value: 0.15,
            baseline_value: 0.15,
            hard_min: 0.05,
            hard_max: 0.40,
            adaptation_rate: 0.02,
            mean_reversion_rate: 0.005,
            tightening_multiplier: 1.5,
            relaxation_multiplier: 0.5,
            observation_window: 30,
            performance_threshold: 0.0,
            cooldown_steps: 5,
            max_step_adjustment: 0.01,
            auto_adapt: true,
            is_ceiling: true,
        }
    }
}

/// Current status of the setpoint relative to the actual value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetpointStatus {
    /// Actual value is within acceptable range of setpoint
    InRange,
    /// Actual value has breached the setpoint (violation)
    Breached,
    /// Actual value is approaching the setpoint (warning zone)
    Warning,
    /// Not enough data to determine status
    Unknown,
}

/// Snapshot of the current setpoint state
#[derive(Debug, Clone)]
pub struct SetpointSnapshot {
    /// Current setpoint value
    pub value: f64,
    /// The long-term baseline
    pub baseline: f64,
    /// Deviation of current setpoint from baseline
    pub deviation_from_baseline: f64,
    /// Status relative to last observed actual value
    pub status: SetpointStatus,
    /// Current actual value (last observed)
    pub actual_value: f64,
    /// Error: difference between actual and setpoint
    pub error: f64,
    /// Number of breaches detected
    pub breach_count: u64,
    /// Total number of observations
    pub observation_count: u64,
    /// Number of adjustments made
    pub adjustment_count: usize,
    /// Whether auto-adaptation is enabled
    pub auto_adapt: bool,
}

/// Adaptive homeostatic setpoint for portfolio regulation
pub struct Setpoint {
    /// Configuration parameters
    config: SetpointConfig,
    /// Current setpoint value
    value: f64,
    /// Recent observed actual values
    observations: VecDeque<f64>,
    /// Recent performance values (e.g., returns) for adaptation
    performance_history: VecDeque<f64>,
    /// Last observed actual value
    last_actual: f64,
    /// Current time step
    current_step: u64,
    /// Step at which the last adjustment occurred
    last_adjustment_step: u64,
    /// Total number of observations
    observation_count: u64,
    /// Number of times the actual value breached the setpoint
    breach_count: u64,
    /// Consecutive breach count (for escalation)
    consecutive_breaches: u64,
    /// History of adjustments (audit trail)
    adjustments: Vec<SetpointAdjustment>,
    /// Maximum adjustments to keep in history
    max_adjustment_history: usize,
    /// Accumulated error integral (for integral-like adaptation)
    error_integral: f64,
    /// Error integral decay factor
    error_integral_decay: f64,
}

impl Default for Setpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl Setpoint {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(SetpointConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: SetpointConfig) -> Self {
        let (effective_min, effective_max) = if config.hard_min <= config.hard_max {
            (config.hard_min, config.hard_max)
        } else {
            // Store as-is for process() validation to catch, but use
            // a safe clamping range for initialization
            (config.hard_max, config.hard_min)
        };
        let initial = config.initial_value.clamp(effective_min, effective_max);
        Self {
            observations: VecDeque::with_capacity(config.observation_window),
            performance_history: VecDeque::with_capacity(config.observation_window),
            value: initial,
            last_actual: 0.0,
            current_step: 0,
            last_adjustment_step: 0,
            observation_count: 0,
            breach_count: 0,
            consecutive_breaches: 0,
            adjustments: Vec::new(),
            max_adjustment_history: 200,
            error_integral: 0.0,
            error_integral_decay: 0.95,
            config,
        }
    }

    /// Create a volatility setpoint with specified target
    pub fn volatility(target: f64) -> Self {
        Self::with_config(SetpointConfig {
            setpoint_type: SetpointType::Volatility,
            initial_value: target,
            baseline_value: target,
            hard_min: target * 0.3,
            hard_max: target * 3.0,
            is_ceiling: true,
            ..Default::default()
        })
    }

    /// Create a max drawdown setpoint
    pub fn max_drawdown(limit: f64) -> Self {
        Self::with_config(SetpointConfig {
            setpoint_type: SetpointType::MaxDrawdown,
            initial_value: limit,
            baseline_value: limit,
            hard_min: 0.01,
            hard_max: 0.50,
            is_ceiling: true,
            tightening_multiplier: 2.0,
            relaxation_multiplier: 0.3,
            ..Default::default()
        })
    }

    /// Create an exposure setpoint
    pub fn exposure(target: f64) -> Self {
        Self::with_config(SetpointConfig {
            setpoint_type: SetpointType::Exposure,
            initial_value: target,
            baseline_value: target,
            hard_min: 0.0,
            hard_max: 2.0,
            is_ceiling: false, // exposure is neither purely ceiling nor floor
            ..Default::default()
        })
    }

    /// Main processing function — validates state
    pub fn process(&self) -> Result<()> {
        if self.config.hard_min > self.config.hard_max {
            return Err(Error::InvalidInput(
                "Setpoint hard_min must be <= hard_max".into(),
            ));
        }
        if self.config.adaptation_rate < 0.0 || self.config.adaptation_rate > 1.0 {
            return Err(Error::InvalidInput(
                "Setpoint adaptation_rate must be in [0.0, 1.0]".into(),
            ));
        }
        if self.value < self.config.hard_min || self.value > self.config.hard_max {
            return Err(Error::InvalidState(format!(
                "Setpoint value {} is outside bounds [{}, {}]",
                self.value, self.config.hard_min, self.config.hard_max
            )));
        }
        Ok(())
    }

    /// Observe the current actual value of the regulated variable
    ///
    /// This updates the internal state and triggers automatic adaptation
    /// if enabled and cooldown has elapsed.
    pub fn observe(&mut self, actual_value: f64) {
        // Store observation
        if self.observations.len() >= self.config.observation_window {
            self.observations.pop_front();
        }
        self.observations.push_back(actual_value);
        self.last_actual = actual_value;
        self.observation_count += 1;

        // Check for breach
        let breached = self.is_breached_value(actual_value);
        if breached {
            self.breach_count += 1;
            self.consecutive_breaches += 1;
        } else {
            self.consecutive_breaches = 0;
        }

        // Update error integral
        let error = self.compute_error(actual_value);
        self.error_integral = self.error_integral * self.error_integral_decay + error;

        // Auto-adapt if enabled
        if self.config.auto_adapt {
            self.try_auto_adapt();
        }

        self.current_step += 1;
    }

    /// Observe a performance metric (e.g., recent return) for adaptation logic
    pub fn observe_performance(&mut self, performance: f64) {
        if self.performance_history.len() >= self.config.observation_window {
            self.performance_history.pop_front();
        }
        self.performance_history.push_back(performance);
    }

    /// Manually adjust the setpoint value
    pub fn set_value(&mut self, new_value: f64, reason: &str) {
        let clamped = new_value.clamp(self.config.hard_min, self.config.hard_max);
        let direction = if clamped > self.value {
            if self.config.is_ceiling {
                AdjustmentDirection::Relax
            } else {
                AdjustmentDirection::Tighten
            }
        } else if clamped < self.value {
            if self.config.is_ceiling {
                AdjustmentDirection::Tighten
            } else {
                AdjustmentDirection::Relax
            }
        } else {
            AdjustmentDirection::Hold
        };

        self.record_adjustment(clamped, direction, reason.to_string());
        self.value = clamped;
        self.last_adjustment_step = self.current_step;
    }

    /// Tighten the setpoint by a given amount (absolute adjustment)
    pub fn tighten(&mut self, amount: f64) {
        let adjustment = amount.abs().min(self.config.max_step_adjustment);
        let new_value = if self.config.is_ceiling {
            self.value - adjustment // lower ceiling = tighter
        } else {
            self.value + adjustment // raise floor = tighter
        };
        self.set_value(new_value, "manual tighten");
    }

    /// Relax the setpoint by a given amount (absolute adjustment)
    pub fn relax(&mut self, amount: f64) {
        let adjustment = amount.abs().min(self.config.max_step_adjustment);
        let new_value = if self.config.is_ceiling {
            self.value + adjustment // raise ceiling = more relaxed
        } else {
            self.value - adjustment // lower floor = more relaxed
        };
        self.set_value(new_value, "manual relax");
    }

    /// Reset the setpoint to its baseline value
    pub fn reset_to_baseline(&mut self) {
        self.set_value(self.config.baseline_value, "reset to baseline");
    }

    /// Get the current setpoint value
    pub fn value(&self) -> f64 {
        self.value
    }

    /// Get the baseline value
    pub fn baseline(&self) -> f64 {
        self.config.baseline_value
    }

    /// Get the deviation from baseline (current - baseline)
    pub fn deviation_from_baseline(&self) -> f64 {
        self.value - self.config.baseline_value
    }

    /// Get the deviation as a percentage of baseline
    pub fn deviation_pct(&self) -> f64 {
        if self.config.baseline_value == 0.0 {
            return 0.0;
        }
        (self.value - self.config.baseline_value) / self.config.baseline_value.abs()
    }

    /// Get the current error (actual - setpoint, or setpoint - actual for ceiling)
    pub fn error(&self) -> f64 {
        self.compute_error(self.last_actual)
    }

    /// Get the accumulated error integral
    pub fn error_integral(&self) -> f64 {
        self.error_integral
    }

    /// Whether the last observed actual value breached the setpoint
    pub fn is_breached(&self) -> bool {
        if self.observation_count == 0 {
            return false;
        }
        self.is_breached_value(self.last_actual)
    }

    /// Get the current status
    pub fn status(&self) -> SetpointStatus {
        if self.observation_count == 0 {
            return SetpointStatus::Unknown;
        }

        if self.is_breached() {
            SetpointStatus::Breached
        } else {
            // Check if approaching (within 20% of the setpoint range)
            let margin = (self.config.hard_max - self.config.hard_min) * 0.10;
            let distance = if self.config.is_ceiling {
                self.value - self.last_actual
            } else {
                self.last_actual - self.value
            };

            if distance < margin && distance >= 0.0 {
                SetpointStatus::Warning
            } else {
                SetpointStatus::InRange
            }
        }
    }

    /// Get the total number of breaches
    pub fn breach_count(&self) -> u64 {
        self.breach_count
    }

    /// Get the consecutive breach count
    pub fn consecutive_breaches(&self) -> u64 {
        self.consecutive_breaches
    }

    /// Get the total number of observations
    pub fn observation_count(&self) -> u64 {
        self.observation_count
    }

    /// Get the last observed actual value
    pub fn last_actual(&self) -> f64 {
        self.last_actual
    }

    /// Get the setpoint type
    pub fn setpoint_type(&self) -> SetpointType {
        self.config.setpoint_type
    }

    /// Get the adjustment history
    pub fn adjustments(&self) -> &[SetpointAdjustment] {
        &self.adjustments
    }

    /// Get a snapshot of the current state
    pub fn snapshot(&self) -> SetpointSnapshot {
        SetpointSnapshot {
            value: self.value,
            baseline: self.config.baseline_value,
            deviation_from_baseline: self.deviation_from_baseline(),
            status: self.status(),
            actual_value: self.last_actual,
            error: self.error(),
            breach_count: self.breach_count,
            observation_count: self.observation_count,
            adjustment_count: self.adjustments.len(),
            auto_adapt: self.config.auto_adapt,
        }
    }

    /// Get the breach rate (breaches / observations)
    pub fn breach_rate(&self) -> f64 {
        if self.observation_count == 0 {
            return 0.0;
        }
        self.breach_count as f64 / self.observation_count as f64
    }

    /// Get the mean of recent observations
    pub fn mean_actual(&self) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.observations.iter().sum();
        sum / self.observations.len() as f64
    }

    /// Get the mean of recent performance values
    pub fn mean_performance(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.performance_history.iter().sum();
        sum / self.performance_history.len() as f64
    }

    /// Enable or disable auto-adaptation
    pub fn set_auto_adapt(&mut self, enabled: bool) {
        self.config.auto_adapt = enabled;
    }

    /// Whether auto-adaptation is enabled
    pub fn auto_adapt_enabled(&self) -> bool {
        self.config.auto_adapt
    }

    /// Reset all state (observations, breaches, etc.) but keep the current setpoint value
    pub fn reset_observations(&mut self) {
        self.observations.clear();
        self.performance_history.clear();
        self.last_actual = 0.0;
        self.observation_count = 0;
        self.breach_count = 0;
        self.consecutive_breaches = 0;
        self.error_integral = 0.0;
        self.current_step = 0;
        self.last_adjustment_step = 0;
    }

    /// Full reset including setpoint value back to initial
    pub fn reset(&mut self) {
        self.reset_observations();
        self.value = self
            .config
            .initial_value
            .clamp(self.config.hard_min, self.config.hard_max);
        self.adjustments.clear();
    }

    // ── internal ──

    /// Compute the error signal for a given actual value
    fn compute_error(&self, actual: f64) -> f64 {
        if self.config.is_ceiling {
            // For a ceiling, error is positive when actual exceeds setpoint (bad)
            actual - self.value
        } else {
            // For a floor, error is positive when actual is below setpoint (bad)
            self.value - actual
        }
    }

    /// Check if a given actual value breaches the setpoint
    fn is_breached_value(&self, actual: f64) -> bool {
        if self.config.is_ceiling {
            actual > self.value
        } else {
            actual < self.value
        }
    }

    /// Try to auto-adapt the setpoint based on recent observations and performance
    fn try_auto_adapt(&mut self) {
        // Check cooldown
        let steps_since_last = self.current_step.saturating_sub(self.last_adjustment_step);
        if steps_since_last < self.config.cooldown_steps {
            return;
        }

        // Need enough observations
        if self.observations.len() < 3 {
            return;
        }

        let mut adjustment = 0.0_f64;
        let mut reason_parts: Vec<&str> = Vec::new();

        // Component 1: Performance-based adaptation
        if !self.performance_history.is_empty() {
            let avg_perf = self.mean_performance();
            if avg_perf < self.config.performance_threshold {
                // Bad performance → tighten
                let delta = (self.config.performance_threshold - avg_perf).abs()
                    * self.config.adaptation_rate
                    * self.config.tightening_multiplier;
                if self.config.is_ceiling {
                    adjustment -= delta; // lower ceiling
                } else {
                    adjustment += delta; // raise floor
                }
                reason_parts.push("poor performance");
            } else {
                // Good performance → relax (more cautiously)
                let delta = (avg_perf - self.config.performance_threshold).abs()
                    * self.config.adaptation_rate
                    * self.config.relaxation_multiplier;
                if self.config.is_ceiling {
                    adjustment += delta; // raise ceiling
                } else {
                    adjustment -= delta; // lower floor
                }
                reason_parts.push("good performance");
            }
        }

        // Component 2: Consecutive breach escalation
        if self.consecutive_breaches >= 3 {
            let escalation = self.consecutive_breaches as f64 * 0.002;
            if self.config.is_ceiling {
                adjustment -= escalation; // lower ceiling (tighter) after repeated breaches
            } else {
                adjustment += escalation; // raise floor (tighter) after repeated breaches
            }
            reason_parts.push("consecutive breaches");
        }

        // Component 3: Mean reversion toward baseline
        let baseline_pull =
            (self.config.baseline_value - self.value) * self.config.mean_reversion_rate;
        adjustment += baseline_pull;
        if baseline_pull.abs() > 1e-6 {
            reason_parts.push("mean reversion");
        }

        // Clamp adjustment magnitude
        adjustment = adjustment.clamp(
            -self.config.max_step_adjustment,
            self.config.max_step_adjustment,
        );

        // Apply if meaningful
        if adjustment.abs() > 1e-8 {
            let new_value =
                (self.value + adjustment).clamp(self.config.hard_min, self.config.hard_max);

            if (new_value - self.value).abs() > 1e-8 {
                let direction = if new_value > self.value {
                    if self.config.is_ceiling {
                        AdjustmentDirection::Relax
                    } else {
                        AdjustmentDirection::Tighten
                    }
                } else if self.config.is_ceiling {
                    AdjustmentDirection::Tighten
                } else {
                    AdjustmentDirection::Relax
                };

                let reason = if reason_parts.is_empty() {
                    "auto adapt".to_string()
                } else {
                    format!("auto adapt: {}", reason_parts.join(", "))
                };

                self.record_adjustment(new_value, direction, reason);
                self.value = new_value;
                self.last_adjustment_step = self.current_step;
            }
        }
    }

    /// Record an adjustment in the history
    fn record_adjustment(
        &mut self,
        new_value: f64,
        direction: AdjustmentDirection,
        reason: String,
    ) {
        let adj = SetpointAdjustment {
            step: self.current_step,
            previous_value: self.value,
            new_value,
            direction,
            reason,
        };

        self.adjustments.push(adj);

        // Trim history if too long
        if self.adjustments.len() > self.max_adjustment_history {
            let excess = self.adjustments.len() - self.max_adjustment_history;
            self.adjustments.drain(0..excess);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Setpoint::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_initial_value() {
        let sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.20,
            baseline_value: 0.20,
            ..Default::default()
        });

        assert!((sp.value() - 0.20).abs() < 1e-9);
        assert!((sp.baseline() - 0.20).abs() < 1e-9);
        assert!((sp.deviation_from_baseline()).abs() < 1e-9);
    }

    #[test]
    fn test_initial_value_clamped_to_bounds() {
        let sp = Setpoint::with_config(SetpointConfig {
            initial_value: 100.0,
            hard_max: 1.0,
            hard_min: 0.0,
            ..Default::default()
        });

        assert!(
            (sp.value() - 1.0).abs() < 1e-9,
            "should be clamped to hard_max"
        );
    }

    #[test]
    fn test_breach_detection_ceiling() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            is_ceiling: true,
            auto_adapt: false,
            ..Default::default()
        });

        sp.observe(0.10); // below ceiling → no breach
        assert!(!sp.is_breached());
        assert_eq!(sp.status(), SetpointStatus::InRange);

        sp.observe(0.20); // above ceiling → breach
        assert!(sp.is_breached());
        assert_eq!(sp.status(), SetpointStatus::Breached);
        assert_eq!(sp.breach_count(), 1);
    }

    #[test]
    fn test_breach_detection_floor() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.50,
            baseline_value: 0.50,
            hard_min: 0.0,
            hard_max: 2.0,
            is_ceiling: false,
            auto_adapt: false,
            ..Default::default()
        });

        sp.observe(0.60); // above floor → no breach
        assert!(!sp.is_breached());

        sp.observe(0.30); // below floor → breach
        assert!(sp.is_breached());
        assert_eq!(sp.status(), SetpointStatus::Breached);
    }

    #[test]
    fn test_consecutive_breaches() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            is_ceiling: true,
            auto_adapt: false,
            ..Default::default()
        });

        sp.observe(0.20); // breach 1
        sp.observe(0.22); // breach 2
        sp.observe(0.18); // breach 3
        assert_eq!(sp.consecutive_breaches(), 3);

        sp.observe(0.10); // no breach
        assert_eq!(sp.consecutive_breaches(), 0);
        assert_eq!(sp.breach_count(), 3);
    }

    #[test]
    fn test_manual_set_value() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            hard_min: 0.05,
            hard_max: 0.40,
            auto_adapt: false,
            ..Default::default()
        });

        sp.set_value(0.20, "operator adjustment");
        assert!((sp.value() - 0.20).abs() < 1e-9);
        assert_eq!(sp.adjustments().len(), 1);
        assert_eq!(sp.adjustments()[0].reason, "operator adjustment");
    }

    #[test]
    fn test_set_value_clamped() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            hard_min: 0.05,
            hard_max: 0.40,
            auto_adapt: false,
            ..Default::default()
        });

        sp.set_value(0.60, "try to exceed max");
        assert!(
            (sp.value() - 0.40).abs() < 1e-9,
            "should be clamped to hard_max"
        );

        sp.set_value(0.01, "try to go below min");
        assert!(
            (sp.value() - 0.05).abs() < 1e-9,
            "should be clamped to hard_min"
        );
    }

    #[test]
    fn test_tighten_ceiling() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            is_ceiling: true,
            max_step_adjustment: 0.05,
            auto_adapt: false,
            ..Default::default()
        });

        let before = sp.value();
        sp.tighten(0.02);
        assert!(
            sp.value() < before,
            "tightening a ceiling should lower the value"
        );
        assert!((sp.value() - (before - 0.02)).abs() < 1e-9);
    }

    #[test]
    fn test_relax_ceiling() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            is_ceiling: true,
            max_step_adjustment: 0.05,
            auto_adapt: false,
            ..Default::default()
        });

        let before = sp.value();
        sp.relax(0.02);
        assert!(
            sp.value() > before,
            "relaxing a ceiling should raise the value"
        );
    }

    #[test]
    fn test_tighten_floor() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.50,
            hard_min: 0.0,
            hard_max: 2.0,
            is_ceiling: false,
            max_step_adjustment: 0.10,
            auto_adapt: false,
            ..Default::default()
        });

        let before = sp.value();
        sp.tighten(0.05);
        assert!(
            sp.value() > before,
            "tightening a floor should raise the value"
        );
    }

    #[test]
    fn test_reset_to_baseline() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            baseline_value: 0.15,
            auto_adapt: false,
            ..Default::default()
        });

        sp.set_value(0.25, "move away");
        assert!((sp.value() - 0.25).abs() < 1e-9);

        sp.reset_to_baseline();
        assert!((sp.value() - 0.15).abs() < 1e-9);
    }

    #[test]
    fn test_deviation_pct() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.10,
            baseline_value: 0.10,
            auto_adapt: false,
            ..Default::default()
        });

        sp.set_value(0.12, "shift");
        let dev = sp.deviation_pct();
        assert!(
            (dev - 0.2).abs() < 1e-9,
            "expected 20% deviation, got {}",
            dev
        );
    }

    #[test]
    fn test_error_signal_ceiling() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            is_ceiling: true,
            auto_adapt: false,
            ..Default::default()
        });

        sp.observe(0.20);
        // Error for ceiling: actual - setpoint = 0.20 - 0.15 = 0.05 (positive = bad)
        assert!((sp.error() - 0.05).abs() < 1e-9);

        sp.observe(0.10);
        // Error: 0.10 - 0.15 = -0.05 (negative = good, within ceiling)
        assert!((sp.error() - (-0.05)).abs() < 1e-9);
    }

    #[test]
    fn test_auto_adapt_tightens_on_bad_performance() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            baseline_value: 0.15,
            is_ceiling: true,
            auto_adapt: true,
            adaptation_rate: 0.10,
            tightening_multiplier: 2.0,
            relaxation_multiplier: 0.5,
            performance_threshold: 0.0,
            cooldown_steps: 0,
            mean_reversion_rate: 0.0,
            max_step_adjustment: 0.05,
            ..Default::default()
        });

        let initial = sp.value();

        // Feed bad performance data
        for _ in 0..10 {
            sp.observe_performance(-0.05);
        }

        // Trigger adaptation via observation
        for _ in 0..5 {
            sp.observe(0.10);
        }

        // Setpoint should have tightened (lowered for ceiling)
        assert!(
            sp.value() < initial,
            "should tighten on bad performance: {} >= {}",
            sp.value(),
            initial
        );
    }

    #[test]
    fn test_auto_adapt_relaxes_on_good_performance() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            baseline_value: 0.20,
            is_ceiling: true,
            auto_adapt: true,
            adaptation_rate: 0.10,
            tightening_multiplier: 1.0,
            relaxation_multiplier: 1.0,
            performance_threshold: 0.0,
            cooldown_steps: 0,
            mean_reversion_rate: 0.0,
            max_step_adjustment: 0.05,
            ..Default::default()
        });

        let initial = sp.value();

        for _ in 0..10 {
            sp.observe_performance(0.10);
        }

        for _ in 0..5 {
            sp.observe(0.10);
        }

        assert!(
            sp.value() > initial,
            "should relax on good performance: {} <= {}",
            sp.value(),
            initial
        );
    }

    #[test]
    fn test_mean_reversion() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.30,
            baseline_value: 0.15,
            is_ceiling: true,
            auto_adapt: true,
            adaptation_rate: 0.0,
            mean_reversion_rate: 0.10,
            cooldown_steps: 0,
            max_step_adjustment: 0.05,
            ..Default::default()
        });

        let initial = sp.value();

        for _ in 0..10 {
            sp.observe(0.10);
        }

        // Should revert toward baseline (0.15), meaning it should decrease from 0.30
        assert!(
            sp.value() < initial,
            "should revert toward baseline: {} >= {}",
            sp.value(),
            initial
        );
    }

    #[test]
    fn test_cooldown_prevents_rapid_adaptation() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            baseline_value: 0.20,
            is_ceiling: true,
            auto_adapt: true,
            adaptation_rate: 0.50,
            mean_reversion_rate: 0.50,
            cooldown_steps: 100,
            max_step_adjustment: 0.05,
            ..Default::default()
        });

        let _initial = sp.value();

        // Only 3 observations — within cooldown
        for _ in 0..3 {
            sp.observe_performance(0.10);
            sp.observe(0.10);
        }

        // After the first adjustment, cooldown should prevent further adjustments
        let adj_count = sp.adjustments().len();
        assert!(
            adj_count <= 1,
            "cooldown should limit adjustments, got {}",
            adj_count
        );
    }

    #[test]
    fn test_breach_rate() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            is_ceiling: true,
            auto_adapt: false,
            ..Default::default()
        });

        sp.observe(0.10); // ok
        sp.observe(0.20); // breach
        sp.observe(0.10); // ok
        sp.observe(0.25); // breach

        assert!((sp.breach_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_mean_actual() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            auto_adapt: false,
            ..Default::default()
        });

        sp.observe(10.0);
        sp.observe(20.0);
        sp.observe(30.0);

        assert!((sp.mean_actual() - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_factory_methods() {
        let vol_sp = Setpoint::volatility(0.20);
        assert!((vol_sp.value() - 0.20).abs() < 1e-9);
        assert_eq!(vol_sp.setpoint_type(), SetpointType::Volatility);

        let dd_sp = Setpoint::max_drawdown(0.10);
        assert!((dd_sp.value() - 0.10).abs() < 1e-9);
        assert_eq!(dd_sp.setpoint_type(), SetpointType::MaxDrawdown);

        let exp_sp = Setpoint::exposure(0.80);
        assert!((exp_sp.value() - 0.80).abs() < 1e-9);
        assert_eq!(exp_sp.setpoint_type(), SetpointType::Exposure);
    }

    #[test]
    fn test_snapshot() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            baseline_value: 0.15,
            auto_adapt: false,
            ..Default::default()
        });

        sp.observe(0.12);
        sp.observe(0.18);

        let snap = sp.snapshot();
        assert!((snap.value - 0.15).abs() < 1e-9);
        assert_eq!(snap.observation_count, 2);
        assert_eq!(snap.breach_count, 1); // 0.18 > 0.15 ceiling
    }

    #[test]
    fn test_warning_status() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            hard_min: 0.05,
            hard_max: 0.40,
            is_ceiling: true,
            auto_adapt: false,
            ..Default::default()
        });

        // 10% of range (0.40-0.05=0.35) is 0.035
        // Just below ceiling by 0.01 → within warning zone
        sp.observe(0.14);
        assert_eq!(sp.status(), SetpointStatus::Warning);

        // Well below ceiling
        sp.observe(0.05);
        assert_eq!(sp.status(), SetpointStatus::InRange);
    }

    #[test]
    fn test_reset() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            auto_adapt: false,
            ..Default::default()
        });

        sp.set_value(0.30, "change");
        sp.observe(0.40);

        sp.reset();
        assert!((sp.value() - 0.15).abs() < 1e-9);
        assert_eq!(sp.observation_count(), 0);
        assert_eq!(sp.breach_count(), 0);
        assert!(sp.adjustments().is_empty());
    }

    #[test]
    fn test_reset_observations_keeps_setpoint() {
        let mut sp = Setpoint::with_config(SetpointConfig {
            initial_value: 0.15,
            auto_adapt: false,
            ..Default::default()
        });

        sp.set_value(0.25, "adjusted");
        sp.observe(0.30);

        sp.reset_observations();
        assert!(
            (sp.value() - 0.25).abs() < 1e-9,
            "setpoint should be preserved"
        );
        assert_eq!(sp.observation_count(), 0);
    }

    #[test]
    fn test_process_validates_bounds() {
        let sp = Setpoint::with_config(SetpointConfig {
            hard_min: 0.50,
            hard_max: 0.10, // invalid: min > max
            initial_value: 0.15,
            auto_adapt: false,
            ..Default::default()
        });

        // process() should detect the invalid bounds and return an error
        assert!(sp.process().is_err());
    }

    #[test]
    fn test_auto_adapt_toggle() {
        let mut sp = Setpoint::new();
        assert!(sp.auto_adapt_enabled());

        sp.set_auto_adapt(false);
        assert!(!sp.auto_adapt_enabled());
    }
}
