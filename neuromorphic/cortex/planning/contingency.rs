//! Contingency planning engine for adverse market event response
//!
//! Part of the Cortex region
//! Component: planning
//!
//! Manages emergency response plans that activate when market conditions
//! breach predefined trigger thresholds. Each contingency plan defines
//! trigger conditions, severity classification, and a sequence of fallback
//! actions (position reduction, hedging, liquidity preservation) to execute
//! when triggered.
//!
//! Key features:
//! - Configurable trigger conditions (drawdown, volatility spike, correlation break)
//! - Multi-level severity classification (watch, warning, critical, emergency)
//! - Ordered fallback action sequences with position reduction schedules
//! - Hedge ratio computation for tail-risk protection
//! - Plan activation / deactivation lifecycle with cooldown
//! - EMA-smoothed tracking of trigger frequency and action effectiveness
//! - Sliding window of recent activations for trend analysis
//! - Running statistics for audit and post-mortem review

use crate::common::{Error, Result};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Type of trigger condition that can activate a contingency plan
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerType {
    /// Portfolio drawdown exceeds threshold (e.g. 0.10 = 10%)
    MaxDrawdown,
    /// Realised volatility exceeds threshold (annualised)
    VolatilitySpike,
    /// Correlation breakdown: average pairwise correlation exceeds threshold
    CorrelationBreak,
    /// Value-at-Risk breach: actual loss exceeds VaR estimate
    VarBreach,
    /// Liquidity crisis: spread or slippage exceeds threshold
    LiquidityCrisis,
    /// Custom trigger evaluated externally
    Custom,
}

/// Severity level of a contingency event (ordered from least to most severe)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertLevel {
    /// Monitoring — no action yet, elevated awareness
    Watch,
    /// Conditions deteriorating — prepare fallback actions
    Warning,
    /// Active stress — begin executing fallback actions
    Critical,
    /// Extreme event — immediate full-scale response
    Emergency,
}

impl AlertLevel {
    /// Numeric severity weight (higher = worse)
    pub fn weight(&self) -> f64 {
        match self {
            AlertLevel::Watch => 1.0,
            AlertLevel::Warning => 2.0,
            AlertLevel::Critical => 3.0,
            AlertLevel::Emergency => 4.0,
        }
    }

    /// Whether this level requires immediate action
    pub fn requires_action(&self) -> bool {
        matches!(self, AlertLevel::Critical | AlertLevel::Emergency)
    }
}

/// A single fallback action to execute when a plan is triggered
#[derive(Debug, Clone, PartialEq)]
pub enum FallbackAction {
    /// Reduce position by the given fraction (0.0–1.0) of current exposure
    ReducePosition { fraction: f64 },
    /// Add a hedge with the given notional ratio relative to portfolio value
    AddHedge { hedge_ratio: f64 },
    /// Tighten stop-losses by the given factor (e.g. 0.5 = halve distance)
    TightenStops { factor: f64 },
    /// Increase cash allocation to the given target fraction
    RaiseCash { target_fraction: f64 },
    /// Cancel all open orders
    CancelOpenOrders,
    /// Halt new entries until conditions improve
    HaltNewEntries,
    /// Send alert notification (for human review)
    Notify { message: String },
}

impl FallbackAction {
    /// Estimated impact on portfolio risk (negative = reduces risk)
    pub fn risk_impact_estimate(&self) -> f64 {
        match self {
            FallbackAction::ReducePosition { fraction } => -fraction,
            FallbackAction::AddHedge { hedge_ratio } => -hedge_ratio * 0.8,
            FallbackAction::TightenStops { factor } => -0.1 * (1.0 - factor),
            FallbackAction::RaiseCash { target_fraction } => -target_fraction * 0.5,
            FallbackAction::CancelOpenOrders => -0.05,
            FallbackAction::HaltNewEntries => -0.02,
            FallbackAction::Notify { .. } => 0.0,
        }
    }
}

/// Definition of a single trigger condition
#[derive(Debug, Clone)]
pub struct TriggerCondition {
    /// Type of trigger
    pub trigger_type: TriggerType,
    /// Threshold value that activates this trigger
    pub threshold: f64,
    /// Alert level assigned when this trigger fires
    pub alert_level: AlertLevel,
    /// Human-readable description
    pub description: String,
}

/// A complete contingency plan: trigger → actions
#[derive(Debug, Clone)]
pub struct ContingencyPlan {
    /// Unique plan identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Trigger conditions (any one can activate the plan)
    pub triggers: Vec<TriggerCondition>,
    /// Ordered sequence of actions to execute
    pub actions: Vec<FallbackAction>,
    /// Whether the plan is currently enabled
    pub enabled: bool,
    /// Cooldown period in steps after deactivation before re-triggering
    pub cooldown_steps: u64,
}

/// Market state snapshot used to evaluate triggers
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Current portfolio drawdown from peak (0.0–1.0)
    pub drawdown: f64,
    /// Current realised volatility (annualised)
    pub volatility: f64,
    /// Average pairwise correlation
    pub avg_correlation: f64,
    /// Current VaR estimate
    pub var_estimate: f64,
    /// Actual realised loss (for VaR breach comparison)
    pub realised_loss: f64,
    /// Current bid-ask spread as fraction of mid price
    pub spread: f64,
    /// Portfolio value
    pub portfolio_value: f64,
    /// Current cash fraction
    pub cash_fraction: f64,
    /// Current gross exposure
    pub gross_exposure: f64,
}

impl Default for MarketState {
    fn default() -> Self {
        Self {
            drawdown: 0.0,
            volatility: 0.15,
            avg_correlation: 0.3,
            var_estimate: 0.02,
            realised_loss: 0.0,
            spread: 0.001,
            portfolio_value: 1_000_000.0,
            cash_fraction: 0.05,
            gross_exposure: 0.95,
        }
    }
}

/// Configuration for the contingency engine
#[derive(Debug, Clone)]
pub struct ContingencyConfig {
    /// Maximum number of plans the engine can hold
    pub max_plans: usize,
    /// Default cooldown steps when not specified per plan
    pub default_cooldown: u64,
    /// EMA decay factor for tracking trigger frequency (0 < decay < 1)
    pub ema_decay: f64,
    /// Sliding window size for recent activations
    pub window_size: usize,
    /// Maximum simultaneous active plans
    pub max_active_plans: usize,
    /// Whether to auto-escalate if multiple plans trigger simultaneously
    pub auto_escalate: bool,
    /// Escalation threshold: number of simultaneous triggers to escalate
    pub escalation_threshold: usize,
}

impl Default for ContingencyConfig {
    fn default() -> Self {
        Self {
            max_plans: 50,
            default_cooldown: 10,
            ema_decay: 0.1,
            window_size: 100,
            max_active_plans: 5,
            auto_escalate: true,
            escalation_threshold: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Activation record
// ---------------------------------------------------------------------------

/// Record of a plan activation event
#[derive(Debug, Clone)]
pub struct ActivationRecord {
    /// Plan ID that was activated
    pub plan_id: String,
    /// Alert level at activation
    pub alert_level: AlertLevel,
    /// Trigger type that fired
    pub trigger_type: TriggerType,
    /// Trigger threshold
    pub trigger_threshold: f64,
    /// Actual observed value that breached the trigger
    pub observed_value: f64,
    /// Number of actions executed
    pub actions_executed: usize,
    /// Estimated total risk impact of all actions
    pub estimated_risk_impact: f64,
    /// Step number when activation occurred
    pub step: u64,
}

/// Outcome of evaluating a single trigger against current market state
#[derive(Debug, Clone)]
pub struct TriggerResult {
    /// Whether the trigger fired
    pub fired: bool,
    /// Alert level if fired
    pub alert_level: AlertLevel,
    /// Observed value
    pub observed: f64,
    /// Threshold
    pub threshold: f64,
    /// Breach magnitude: (observed - threshold) / threshold
    pub breach_magnitude: f64,
}

/// Summary of a contingency evaluation cycle
#[derive(Debug, Clone)]
pub struct EvaluationSummary {
    /// Total plans evaluated
    pub plans_evaluated: usize,
    /// Plans that triggered
    pub plans_triggered: usize,
    /// Highest alert level observed
    pub max_alert_level: AlertLevel,
    /// Total actions recommended
    pub total_actions: usize,
    /// Aggregate estimated risk impact
    pub aggregate_risk_impact: f64,
    /// Whether escalation was applied
    pub escalated: bool,
    /// Activation records for this cycle
    pub activations: Vec<ActivationRecord>,
}

// ---------------------------------------------------------------------------
// Running statistics
// ---------------------------------------------------------------------------

/// Running statistics for the contingency engine
#[derive(Debug, Clone)]
pub struct ContingencyStats {
    /// Total evaluation cycles
    pub total_evaluations: u64,
    /// Total activations across all plans
    pub total_activations: u64,
    /// Total escalations applied
    pub total_escalations: u64,
    /// EMA of activation rate (activations per evaluation)
    pub ema_activation_rate: f64,
    /// EMA of average alert level weight when triggered
    pub ema_alert_severity: f64,
    /// EMA of aggregate risk impact per activation
    pub ema_risk_impact: f64,
    /// Most common trigger type (by count)
    pub trigger_counts: [u64; 6], // indexed by TriggerType discriminant
    /// Peak simultaneous active plans
    pub peak_active_plans: usize,
}

impl Default for ContingencyStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_activations: 0,
            total_escalations: 0,
            ema_activation_rate: 0.0,
            ema_alert_severity: 0.0,
            ema_risk_impact: 0.0,
            trigger_counts: [0; 6],
            peak_active_plans: 0,
        }
    }
}

impl ContingencyStats {
    /// Dominant trigger type (most frequent)
    pub fn dominant_trigger(&self) -> TriggerType {
        let max_idx = self
            .trigger_counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0);
        trigger_type_from_index(max_idx)
    }

    /// Average activations per evaluation
    pub fn activation_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_activations as f64 / self.total_evaluations as f64
    }

    /// Escalation rate
    pub fn escalation_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_escalations as f64 / self.total_evaluations as f64
    }
}

fn trigger_type_index(tt: &TriggerType) -> usize {
    match tt {
        TriggerType::MaxDrawdown => 0,
        TriggerType::VolatilitySpike => 1,
        TriggerType::CorrelationBreak => 2,
        TriggerType::VarBreach => 3,
        TriggerType::LiquidityCrisis => 4,
        TriggerType::Custom => 5,
    }
}

fn trigger_type_from_index(idx: usize) -> TriggerType {
    match idx {
        0 => TriggerType::MaxDrawdown,
        1 => TriggerType::VolatilitySpike,
        2 => TriggerType::CorrelationBreak,
        3 => TriggerType::VarBreach,
        4 => TriggerType::LiquidityCrisis,
        _ => TriggerType::Custom,
    }
}

// ---------------------------------------------------------------------------
// Contingency Engine
// ---------------------------------------------------------------------------

/// Contingency planning engine.
///
/// Holds a registry of contingency plans, evaluates them against market state
/// snapshots, tracks cooldowns, and produces activation records with ordered
/// fallback action sequences.
pub struct Contingency {
    config: ContingencyConfig,
    plans: Vec<ContingencyPlan>,
    /// Cooldown counters per plan (indexed same as `plans`)
    cooldowns: Vec<u64>,
    /// Currently active plan IDs
    active_plan_ids: Vec<String>,
    /// Step counter
    step: u64,
    /// EMA initialized flag
    ema_initialized: bool,
    /// Sliding window of recent activation records
    recent: VecDeque<ActivationRecord>,
    /// Running statistics
    stats: ContingencyStats,
}

impl Default for Contingency {
    fn default() -> Self {
        Self::new()
    }
}

impl Contingency {
    /// Create a new contingency engine with default configuration
    pub fn new() -> Self {
        Self {
            config: ContingencyConfig::default(),
            plans: Vec::new(),
            cooldowns: Vec::new(),
            active_plan_ids: Vec::new(),
            step: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: ContingencyStats::default(),
        }
    }

    /// Create with explicit configuration
    pub fn with_config(config: ContingencyConfig) -> Result<Self> {
        if config.max_plans == 0 {
            return Err(Error::InvalidInput("max_plans must be > 0".into()));
        }
        if config.ema_decay <= 0.0 || config.ema_decay >= 1.0 {
            return Err(Error::InvalidInput("ema_decay must be in (0, 1)".into()));
        }
        if config.window_size == 0 {
            return Err(Error::InvalidInput("window_size must be > 0".into()));
        }
        if config.max_active_plans == 0 {
            return Err(Error::InvalidInput("max_active_plans must be > 0".into()));
        }
        if config.escalation_threshold == 0 {
            return Err(Error::InvalidInput(
                "escalation_threshold must be > 0".into(),
            ));
        }
        Ok(Self {
            config,
            plans: Vec::new(),
            cooldowns: Vec::new(),
            active_plan_ids: Vec::new(),
            step: 0,
            ema_initialized: false,
            recent: VecDeque::new(),
            stats: ContingencyStats::default(),
        })
    }

    /// Main processing function (no-op entry point for trait conformance)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Plan management
    // -----------------------------------------------------------------------

    /// Register a new contingency plan
    pub fn add_plan(&mut self, plan: ContingencyPlan) -> Result<()> {
        if self.plans.len() >= self.config.max_plans {
            return Err(Error::ResourceExhausted(format!(
                "maximum plan count ({}) reached",
                self.config.max_plans
            )));
        }
        if plan.triggers.is_empty() {
            return Err(Error::InvalidInput(
                "plan must have at least one trigger".into(),
            ));
        }
        if plan.actions.is_empty() {
            return Err(Error::InvalidInput(
                "plan must have at least one action".into(),
            ));
        }
        // Check for duplicate ID
        if self.plans.iter().any(|p| p.id == plan.id) {
            return Err(Error::InvalidInput(format!(
                "plan with id '{}' already exists",
                plan.id
            )));
        }
        self.cooldowns.push(0);
        self.plans.push(plan);
        Ok(())
    }

    /// Remove a plan by ID
    pub fn remove_plan(&mut self, plan_id: &str) -> Result<()> {
        let idx = self
            .plans
            .iter()
            .position(|p| p.id == plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;
        self.plans.remove(idx);
        self.cooldowns.remove(idx);
        self.active_plan_ids.retain(|id| id != plan_id);
        Ok(())
    }

    /// Enable or disable a plan
    pub fn set_plan_enabled(&mut self, plan_id: &str, enabled: bool) -> Result<()> {
        let plan = self
            .plans
            .iter_mut()
            .find(|p| p.id == plan_id)
            .ok_or_else(|| Error::NotFound(format!("plan '{}' not found", plan_id)))?;
        plan.enabled = enabled;
        if !enabled {
            self.active_plan_ids.retain(|id| id != plan_id);
        }
        Ok(())
    }

    /// Number of registered plans
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Currently active plan IDs
    pub fn active_plans(&self) -> &[String] {
        &self.active_plan_ids
    }

    /// Get a plan by ID
    pub fn get_plan(&self, plan_id: &str) -> Option<&ContingencyPlan> {
        self.plans.iter().find(|p| p.id == plan_id)
    }

    // -----------------------------------------------------------------------
    // Trigger evaluation
    // -----------------------------------------------------------------------

    /// Evaluate a single trigger condition against market state
    pub fn evaluate_trigger(trigger: &TriggerCondition, state: &MarketState) -> TriggerResult {
        let observed = match trigger.trigger_type {
            TriggerType::MaxDrawdown => state.drawdown,
            TriggerType::VolatilitySpike => state.volatility,
            TriggerType::CorrelationBreak => state.avg_correlation,
            TriggerType::VarBreach => {
                // Breach if realised loss exceeds VaR estimate
                if state.var_estimate.abs() < 1e-15 {
                    0.0
                } else {
                    state.realised_loss / state.var_estimate
                }
            }
            TriggerType::LiquidityCrisis => state.spread,
            TriggerType::Custom => 0.0, // Custom triggers evaluated externally
        };

        let fired = observed >= trigger.threshold;
        let breach_magnitude = if trigger.threshold.abs() > 1e-15 {
            (observed - trigger.threshold) / trigger.threshold
        } else {
            0.0
        };

        TriggerResult {
            fired,
            alert_level: trigger.alert_level,
            observed,
            threshold: trigger.threshold,
            breach_magnitude: if fired { breach_magnitude } else { 0.0 },
        }
    }

    /// Evaluate all plans against market state and return an evaluation summary.
    ///
    /// This is the main entry point for each evaluation cycle. It:
    /// 1. Decrements cooldowns
    /// 2. Evaluates each enabled, non-cooling-down plan
    /// 3. Collects activation records for triggered plans
    /// 4. Applies escalation if configured
    /// 5. Updates statistics
    pub fn evaluate(&mut self, state: &MarketState) -> EvaluationSummary {
        self.step += 1;

        // Decrement cooldowns
        for cd in self.cooldowns.iter_mut() {
            if *cd > 0 {
                *cd -= 1;
            }
        }

        let mut activations = Vec::new();
        let mut max_alert = AlertLevel::Watch;
        let mut total_actions = 0usize;
        let mut aggregate_risk_impact = 0.0f64;
        let plans_evaluated = self.plans.len();

        for (i, plan) in self.plans.iter().enumerate() {
            if !plan.enabled || self.cooldowns[i] > 0 {
                continue;
            }

            // Evaluate all triggers for this plan — fire on the worst match
            let mut worst_trigger: Option<TriggerResult> = None;
            let mut worst_trigger_type: Option<&TriggerCondition> = None;

            for trigger in &plan.triggers {
                let result = Self::evaluate_trigger(trigger, state);
                if result.fired {
                    let dominated = match &worst_trigger {
                        None => true,
                        Some(prev) => {
                            result.alert_level > prev.alert_level
                                || (result.alert_level == prev.alert_level
                                    && result.breach_magnitude > prev.breach_magnitude)
                        }
                    };
                    if dominated {
                        worst_trigger = Some(result);
                        worst_trigger_type = Some(trigger);
                    }
                }
            }

            if let (Some(trigger_result), Some(trigger_cond)) = (worst_trigger, worst_trigger_type)
            {
                // Compute total risk impact of actions
                let risk_impact: f64 = plan.actions.iter().map(|a| a.risk_impact_estimate()).sum();

                let record = ActivationRecord {
                    plan_id: plan.id.clone(),
                    alert_level: trigger_result.alert_level,
                    trigger_type: trigger_cond.trigger_type.clone(),
                    trigger_threshold: trigger_result.threshold,
                    observed_value: trigger_result.observed,
                    actions_executed: plan.actions.len(),
                    estimated_risk_impact: risk_impact,
                    step: self.step,
                };

                if trigger_result.alert_level > max_alert {
                    max_alert = trigger_result.alert_level;
                }
                total_actions += plan.actions.len();
                aggregate_risk_impact += risk_impact;

                // Set cooldown
                let cooldown = if plan.cooldown_steps > 0 {
                    plan.cooldown_steps
                } else {
                    self.config.default_cooldown
                };
                self.cooldowns[i] = cooldown;

                // Track active plan
                if !self.active_plan_ids.contains(&plan.id) {
                    if self.active_plan_ids.len() < self.config.max_active_plans {
                        self.active_plan_ids.push(plan.id.clone());
                    }
                }

                // Update trigger counts
                let tidx = trigger_type_index(&trigger_cond.trigger_type);
                self.stats.trigger_counts[tidx] += 1;

                activations.push(record);
            }
        }

        // Auto-escalation
        let escalated =
            self.config.auto_escalate && activations.len() >= self.config.escalation_threshold;

        if escalated {
            max_alert = AlertLevel::Emergency;
            self.stats.total_escalations += 1;
        }

        let plans_triggered = activations.len();

        // Update stats
        self.stats.total_evaluations += 1;
        self.stats.total_activations += plans_triggered as u64;

        if self.active_plan_ids.len() > self.stats.peak_active_plans {
            self.stats.peak_active_plans = self.active_plan_ids.len();
        }

        // EMA updates
        let activation_rate = plans_triggered as f64 / plans_evaluated.max(1) as f64;
        let avg_severity = if plans_triggered > 0 {
            activations
                .iter()
                .map(|a| a.alert_level.weight())
                .sum::<f64>()
                / plans_triggered as f64
        } else {
            0.0
        };
        let avg_risk_impact = if plans_triggered > 0 {
            aggregate_risk_impact / plans_triggered as f64
        } else {
            0.0
        };

        let alpha = self.config.ema_decay;
        if !self.ema_initialized {
            self.stats.ema_activation_rate = activation_rate;
            self.stats.ema_alert_severity = avg_severity;
            self.stats.ema_risk_impact = avg_risk_impact;
            self.ema_initialized = true;
        } else {
            self.stats.ema_activation_rate =
                alpha * activation_rate + (1.0 - alpha) * self.stats.ema_activation_rate;
            self.stats.ema_alert_severity =
                alpha * avg_severity + (1.0 - alpha) * self.stats.ema_alert_severity;
            self.stats.ema_risk_impact =
                alpha * avg_risk_impact + (1.0 - alpha) * self.stats.ema_risk_impact;
        }

        // Store activations in sliding window
        for record in &activations {
            if self.recent.len() >= self.config.window_size {
                self.recent.pop_front();
            }
            self.recent.push_back(record.clone());
        }

        // Clean up deactivated plans (those past cooldown with no re-trigger)
        self.active_plan_ids.retain(|id| {
            activations.iter().any(|a| &a.plan_id == id)
                || self
                    .plans
                    .iter()
                    .zip(self.cooldowns.iter())
                    .any(|(p, cd)| p.id == *id && *cd > 0)
        });

        EvaluationSummary {
            plans_evaluated,
            plans_triggered,
            max_alert_level: max_alert,
            total_actions,
            aggregate_risk_impact,
            escalated,
            activations,
        }
    }

    // -----------------------------------------------------------------------
    // Position reduction schedule
    // -----------------------------------------------------------------------

    /// Compute a position reduction schedule given an alert level.
    ///
    /// Returns the fraction of current exposure to reduce.
    /// Watch → 0%, Warning → 10%, Critical → 40%, Emergency → 75%
    pub fn reduction_fraction(level: AlertLevel) -> f64 {
        match level {
            AlertLevel::Watch => 0.0,
            AlertLevel::Warning => 0.10,
            AlertLevel::Critical => 0.40,
            AlertLevel::Emergency => 0.75,
        }
    }

    /// Compute notional hedge ratio for a given alert level and portfolio VaR.
    ///
    /// Hedge ratio increases with severity and scales with the VaR estimate.
    pub fn hedge_ratio(level: AlertLevel, var_estimate: f64) -> f64 {
        let base = match level {
            AlertLevel::Watch => 0.0,
            AlertLevel::Warning => 0.25,
            AlertLevel::Critical => 0.50,
            AlertLevel::Emergency => 1.0,
        };
        // Scale by VaR (larger VaR → larger hedge)
        base * var_estimate.abs().min(1.0)
    }

    /// Compute recommended cash target fraction for a given alert level
    pub fn cash_target(level: AlertLevel, current_cash: f64) -> f64 {
        let target: f64 = match level {
            AlertLevel::Watch => 0.05,
            AlertLevel::Warning => 0.15,
            AlertLevel::Critical => 0.30,
            AlertLevel::Emergency => 0.50,
        };
        target.max(current_cash) // never decrease cash
    }

    /// Build a complete action sequence for a given alert level and market state
    pub fn build_response_actions(level: AlertLevel, state: &MarketState) -> Vec<FallbackAction> {
        let mut actions = Vec::new();

        if !level.requires_action() && level == AlertLevel::Watch {
            return actions; // No actions for watch level
        }

        // Warning: notify + halt new entries
        if level >= AlertLevel::Warning {
            actions.push(FallbackAction::Notify {
                message: format!(
                    "Alert level {:?}: drawdown={:.2}%, vol={:.2}%",
                    level,
                    state.drawdown * 100.0,
                    state.volatility * 100.0
                ),
            });
            actions.push(FallbackAction::HaltNewEntries);
        }

        // Critical: add position reduction + tighten stops
        if level >= AlertLevel::Critical {
            let reduction = Self::reduction_fraction(level);
            actions.push(FallbackAction::ReducePosition {
                fraction: reduction,
            });
            actions.push(FallbackAction::TightenStops { factor: 0.5 });
        }

        // Emergency: cancel orders + hedge + raise cash
        if level >= AlertLevel::Emergency {
            actions.push(FallbackAction::CancelOpenOrders);
            let hr = Self::hedge_ratio(level, state.var_estimate);
            if hr > 0.0 {
                actions.push(FallbackAction::AddHedge { hedge_ratio: hr });
            }
            let cash_target = Self::cash_target(level, state.cash_fraction);
            actions.push(FallbackAction::RaiseCash {
                target_fraction: cash_target,
            });
        }

        actions
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Running statistics
    pub fn stats(&self) -> &ContingencyStats {
        &self.stats
    }

    /// Current step counter
    pub fn current_step(&self) -> u64 {
        self.step
    }

    /// Configuration reference
    pub fn config(&self) -> &ContingencyConfig {
        &self.config
    }

    /// Recent activation records (sliding window)
    pub fn recent_activations(&self) -> &VecDeque<ActivationRecord> {
        &self.recent
    }

    /// EMA-smoothed activation rate
    pub fn smoothed_activation_rate(&self) -> f64 {
        self.stats.ema_activation_rate
    }

    /// EMA-smoothed alert severity
    pub fn smoothed_alert_severity(&self) -> f64 {
        self.stats.ema_alert_severity
    }

    /// EMA-smoothed risk impact
    pub fn smoothed_risk_impact(&self) -> f64 {
        self.stats.ema_risk_impact
    }

    /// Windowed activation rate (activations in window / window length)
    pub fn windowed_activation_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        self.recent.len() as f64 / self.config.window_size as f64
    }

    /// Windowed average alert severity
    pub fn windowed_avg_severity(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.alert_level.weight()).sum();
        sum / self.recent.len() as f64
    }

    /// Whether alert severity is increasing based on recent window trend
    pub fn is_severity_increasing(&self) -> bool {
        let n = self.recent.len();
        if n < 4 {
            return false;
        }
        let mid = n / 2;
        let first_half_avg: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.alert_level.weight())
            .sum::<f64>()
            / mid as f64;
        let second_half_avg: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.alert_level.weight())
            .sum::<f64>()
            / (n - mid) as f64;
        second_half_avg > first_half_avg * 1.1
    }

    /// Reset engine state (plans are kept but cooldowns and stats are cleared)
    pub fn reset(&mut self) {
        self.step = 0;
        self.cooldowns = vec![0; self.plans.len()];
        self.active_plan_ids.clear();
        self.ema_initialized = false;
        self.recent.clear();
        self.stats = ContingencyStats::default();
    }
}

// ---------------------------------------------------------------------------
// Preset plans
// ---------------------------------------------------------------------------

/// Create a standard drawdown-based contingency plan
pub fn preset_drawdown_plan() -> ContingencyPlan {
    ContingencyPlan {
        id: "drawdown_standard".into(),
        name: "Standard Drawdown Protection".into(),
        triggers: vec![
            TriggerCondition {
                trigger_type: TriggerType::MaxDrawdown,
                threshold: 0.05,
                alert_level: AlertLevel::Warning,
                description: "5% drawdown warning".into(),
            },
            TriggerCondition {
                trigger_type: TriggerType::MaxDrawdown,
                threshold: 0.10,
                alert_level: AlertLevel::Critical,
                description: "10% drawdown critical".into(),
            },
            TriggerCondition {
                trigger_type: TriggerType::MaxDrawdown,
                threshold: 0.20,
                alert_level: AlertLevel::Emergency,
                description: "20% drawdown emergency".into(),
            },
        ],
        actions: vec![
            FallbackAction::Notify {
                message: "Drawdown protection triggered".into(),
            },
            FallbackAction::ReducePosition { fraction: 0.25 },
            FallbackAction::TightenStops { factor: 0.5 },
        ],
        enabled: true,
        cooldown_steps: 5,
    }
}

/// Create a standard volatility-spike contingency plan
pub fn preset_volatility_plan() -> ContingencyPlan {
    ContingencyPlan {
        id: "vol_spike_standard".into(),
        name: "Standard Volatility Spike Protection".into(),
        triggers: vec![
            TriggerCondition {
                trigger_type: TriggerType::VolatilitySpike,
                threshold: 0.40,
                alert_level: AlertLevel::Warning,
                description: "40% annualised vol warning".into(),
            },
            TriggerCondition {
                trigger_type: TriggerType::VolatilitySpike,
                threshold: 0.80,
                alert_level: AlertLevel::Critical,
                description: "80% annualised vol critical".into(),
            },
        ],
        actions: vec![
            FallbackAction::HaltNewEntries,
            FallbackAction::ReducePosition { fraction: 0.30 },
            FallbackAction::AddHedge { hedge_ratio: 0.20 },
        ],
        enabled: true,
        cooldown_steps: 10,
    }
}

/// Create a liquidity crisis contingency plan
pub fn preset_liquidity_plan() -> ContingencyPlan {
    ContingencyPlan {
        id: "liquidity_crisis".into(),
        name: "Liquidity Crisis Response".into(),
        triggers: vec![
            TriggerCondition {
                trigger_type: TriggerType::LiquidityCrisis,
                threshold: 0.01,
                alert_level: AlertLevel::Warning,
                description: "1% spread warning".into(),
            },
            TriggerCondition {
                trigger_type: TriggerType::LiquidityCrisis,
                threshold: 0.05,
                alert_level: AlertLevel::Emergency,
                description: "5% spread emergency".into(),
            },
        ],
        actions: vec![
            FallbackAction::CancelOpenOrders,
            FallbackAction::HaltNewEntries,
            FallbackAction::RaiseCash {
                target_fraction: 0.30,
            },
        ],
        enabled: true,
        cooldown_steps: 20,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Helpers --

    fn simple_plan(
        id: &str,
        trigger_type: TriggerType,
        threshold: f64,
        level: AlertLevel,
    ) -> ContingencyPlan {
        ContingencyPlan {
            id: id.into(),
            name: format!("Test plan {}", id),
            triggers: vec![TriggerCondition {
                trigger_type,
                threshold,
                alert_level: level,
                description: "test trigger".into(),
            }],
            actions: vec![FallbackAction::ReducePosition { fraction: 0.25 }],
            enabled: true,
            cooldown_steps: 2,
        }
    }

    fn stressed_state() -> MarketState {
        MarketState {
            drawdown: 0.15,
            volatility: 0.60,
            avg_correlation: 0.80,
            var_estimate: 0.05,
            realised_loss: 0.08,
            spread: 0.02,
            portfolio_value: 900_000.0,
            cash_fraction: 0.03,
            gross_exposure: 0.97,
        }
    }

    fn calm_state() -> MarketState {
        MarketState {
            drawdown: 0.01,
            volatility: 0.12,
            avg_correlation: 0.25,
            var_estimate: 0.02,
            realised_loss: 0.005,
            spread: 0.0005,
            portfolio_value: 1_050_000.0,
            cash_fraction: 0.10,
            gross_exposure: 0.90,
        }
    }

    // -- Basic construction --

    #[test]
    fn test_new_default() {
        let c = Contingency::new();
        assert_eq!(c.plan_count(), 0);
        assert_eq!(c.current_step(), 0);
        assert!(c.process().is_ok());
    }

    #[test]
    fn test_with_config() {
        let c = Contingency::with_config(ContingencyConfig::default());
        assert!(c.is_ok());
    }

    #[test]
    fn test_invalid_config_zero_max_plans() {
        let mut cfg = ContingencyConfig::default();
        cfg.max_plans = 0;
        assert!(Contingency::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_zero() {
        let mut cfg = ContingencyConfig::default();
        cfg.ema_decay = 0.0;
        assert!(Contingency::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_ema_decay_one() {
        let mut cfg = ContingencyConfig::default();
        cfg.ema_decay = 1.0;
        assert!(Contingency::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_window() {
        let mut cfg = ContingencyConfig::default();
        cfg.window_size = 0;
        assert!(Contingency::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_max_active() {
        let mut cfg = ContingencyConfig::default();
        cfg.max_active_plans = 0;
        assert!(Contingency::with_config(cfg).is_err());
    }

    #[test]
    fn test_invalid_config_zero_escalation_threshold() {
        let mut cfg = ContingencyConfig::default();
        cfg.escalation_threshold = 0;
        assert!(Contingency::with_config(cfg).is_err());
    }

    // -- Plan management --

    #[test]
    fn test_add_plan() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        assert!(c.add_plan(plan).is_ok());
        assert_eq!(c.plan_count(), 1);
    }

    #[test]
    fn test_add_plan_duplicate_id() {
        let mut c = Contingency::new();
        let p1 = simple_plan("dup", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        let p2 = simple_plan(
            "dup",
            TriggerType::VolatilitySpike,
            0.50,
            AlertLevel::Warning,
        );
        assert!(c.add_plan(p1).is_ok());
        assert!(c.add_plan(p2).is_err());
    }

    #[test]
    fn test_add_plan_no_triggers() {
        let mut c = Contingency::new();
        let plan = ContingencyPlan {
            id: "empty".into(),
            name: "Empty".into(),
            triggers: vec![],
            actions: vec![FallbackAction::HaltNewEntries],
            enabled: true,
            cooldown_steps: 1,
        };
        assert!(c.add_plan(plan).is_err());
    }

    #[test]
    fn test_add_plan_no_actions() {
        let mut c = Contingency::new();
        let plan = ContingencyPlan {
            id: "no_action".into(),
            name: "No actions".into(),
            triggers: vec![TriggerCondition {
                trigger_type: TriggerType::MaxDrawdown,
                threshold: 0.10,
                alert_level: AlertLevel::Critical,
                description: "test".into(),
            }],
            actions: vec![],
            enabled: true,
            cooldown_steps: 1,
        };
        assert!(c.add_plan(plan).is_err());
    }

    #[test]
    fn test_add_plan_exceeds_max() {
        let mut cfg = ContingencyConfig::default();
        cfg.max_plans = 1;
        let mut c = Contingency::with_config(cfg).unwrap();
        let p1 = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        let p2 = simple_plan(
            "p2",
            TriggerType::VolatilitySpike,
            0.50,
            AlertLevel::Warning,
        );
        assert!(c.add_plan(p1).is_ok());
        assert!(c.add_plan(p2).is_err());
    }

    #[test]
    fn test_remove_plan() {
        let mut c = Contingency::new();
        let plan = simple_plan("rm", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        assert_eq!(c.plan_count(), 1);
        assert!(c.remove_plan("rm").is_ok());
        assert_eq!(c.plan_count(), 0);
    }

    #[test]
    fn test_remove_plan_not_found() {
        let mut c = Contingency::new();
        assert!(c.remove_plan("nonexistent").is_err());
    }

    #[test]
    fn test_enable_disable_plan() {
        let mut c = Contingency::new();
        let plan = simple_plan(
            "toggle",
            TriggerType::MaxDrawdown,
            0.10,
            AlertLevel::Critical,
        );
        c.add_plan(plan).unwrap();
        assert!(c.set_plan_enabled("toggle", false).is_ok());
        let p = c.get_plan("toggle").unwrap();
        assert!(!p.enabled);
        assert!(c.set_plan_enabled("toggle", true).is_ok());
        let p = c.get_plan("toggle").unwrap();
        assert!(p.enabled);
    }

    #[test]
    fn test_enable_disable_not_found() {
        let mut c = Contingency::new();
        assert!(c.set_plan_enabled("nope", true).is_err());
    }

    // -- Trigger evaluation --

    #[test]
    fn test_trigger_fires_drawdown() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::MaxDrawdown,
            threshold: 0.10,
            alert_level: AlertLevel::Critical,
            description: "dd".into(),
        };
        let state = stressed_state(); // drawdown = 0.15
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(result.fired);
        assert_eq!(result.alert_level, AlertLevel::Critical);
        assert!(result.breach_magnitude > 0.0);
    }

    #[test]
    fn test_trigger_does_not_fire() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::MaxDrawdown,
            threshold: 0.20,
            alert_level: AlertLevel::Critical,
            description: "dd".into(),
        };
        let state = calm_state(); // drawdown = 0.01
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(!result.fired);
        assert_eq!(result.breach_magnitude, 0.0);
    }

    #[test]
    fn test_trigger_volatility_spike() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::VolatilitySpike,
            threshold: 0.40,
            alert_level: AlertLevel::Warning,
            description: "vol".into(),
        };
        let state = stressed_state(); // vol = 0.60
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(result.fired);
    }

    #[test]
    fn test_trigger_var_breach() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::VarBreach,
            threshold: 1.0, // fire if realised_loss / var_estimate >= 1.0
            alert_level: AlertLevel::Critical,
            description: "var breach".into(),
        };
        let state = stressed_state(); // 0.08 / 0.05 = 1.6 >= 1.0
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(result.fired);
        assert!(result.observed > 1.0);
    }

    #[test]
    fn test_trigger_liquidity_crisis() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::LiquidityCrisis,
            threshold: 0.01,
            alert_level: AlertLevel::Emergency,
            description: "liq".into(),
        };
        let state = stressed_state(); // spread = 0.02
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(result.fired);
    }

    #[test]
    fn test_trigger_correlation_break() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::CorrelationBreak,
            threshold: 0.70,
            alert_level: AlertLevel::Warning,
            description: "corr".into(),
        };
        let state = stressed_state(); // avg_correlation = 0.80
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(result.fired);
    }

    #[test]
    fn test_trigger_exact_threshold() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::MaxDrawdown,
            threshold: 0.15,
            alert_level: AlertLevel::Warning,
            description: "exact".into(),
        };
        let state = stressed_state(); // drawdown = 0.15
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(result.fired); // >= threshold
    }

    #[test]
    fn test_trigger_zero_var_estimate() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::VarBreach,
            threshold: 1.0,
            alert_level: AlertLevel::Critical,
            description: "zero var".into(),
        };
        let mut state = calm_state();
        state.var_estimate = 0.0;
        let result = Contingency::evaluate_trigger(&trigger, &state);
        assert!(!result.fired); // observed = 0.0 < 1.0
    }

    // -- Full evaluation cycle --

    #[test]
    fn test_evaluate_no_plans() {
        let mut c = Contingency::new();
        let summary = c.evaluate(&calm_state());
        assert_eq!(summary.plans_evaluated, 0);
        assert_eq!(summary.plans_triggered, 0);
        assert!(!summary.escalated);
    }

    #[test]
    fn test_evaluate_calm_no_trigger() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        let summary = c.evaluate(&calm_state());
        assert_eq!(summary.plans_evaluated, 1);
        assert_eq!(summary.plans_triggered, 0);
        assert_eq!(summary.total_actions, 0);
    }

    #[test]
    fn test_evaluate_stressed_triggers() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        let summary = c.evaluate(&stressed_state());
        assert_eq!(summary.plans_triggered, 1);
        assert_eq!(summary.activations[0].plan_id, "p1");
        assert_eq!(summary.activations[0].alert_level, AlertLevel::Critical);
        assert!(summary.activations[0].estimated_risk_impact < 0.0);
    }

    #[test]
    fn test_evaluate_cooldown_prevents_retrigger() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();

        let s1 = c.evaluate(&stressed_state());
        assert_eq!(s1.plans_triggered, 1);

        // Second evaluation: cooldown should prevent re-trigger
        let s2 = c.evaluate(&stressed_state());
        assert_eq!(s2.plans_triggered, 0);
    }

    #[test]
    fn test_evaluate_cooldown_expires() {
        let mut c = Contingency::new();
        // cooldown_steps = 2
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();

        // Step 1: triggers, cooldown set to 2
        c.evaluate(&stressed_state());
        // Step 2: cooldown = 1 (decremented), not fired
        c.evaluate(&stressed_state());
        // Step 3: cooldown = 0 (decremented), can fire again
        let s3 = c.evaluate(&stressed_state());
        assert_eq!(s3.plans_triggered, 1);
    }

    #[test]
    fn test_evaluate_disabled_plan_not_triggered() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.set_plan_enabled("p1", false).unwrap();

        let summary = c.evaluate(&stressed_state());
        assert_eq!(summary.plans_triggered, 0);
    }

    #[test]
    fn test_evaluate_multiple_plans() {
        let mut c = Contingency::new();
        let p1 = simple_plan("dd", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        let p2 = simple_plan(
            "vol",
            TriggerType::VolatilitySpike,
            0.40,
            AlertLevel::Warning,
        );
        c.add_plan(p1).unwrap();
        c.add_plan(p2).unwrap();

        let summary = c.evaluate(&stressed_state());
        assert_eq!(summary.plans_triggered, 2);
    }

    #[test]
    fn test_evaluate_escalation() {
        let mut cfg = ContingencyConfig::default();
        cfg.escalation_threshold = 2;
        let mut c = Contingency::with_config(cfg).unwrap();

        let p1 = simple_plan("dd", TriggerType::MaxDrawdown, 0.10, AlertLevel::Warning);
        let p2 = simple_plan(
            "vol",
            TriggerType::VolatilitySpike,
            0.40,
            AlertLevel::Warning,
        );
        c.add_plan(p1).unwrap();
        c.add_plan(p2).unwrap();

        let summary = c.evaluate(&stressed_state());
        assert!(summary.escalated);
        assert_eq!(summary.max_alert_level, AlertLevel::Emergency);
    }

    #[test]
    fn test_evaluate_no_escalation_below_threshold() {
        let mut cfg = ContingencyConfig::default();
        cfg.escalation_threshold = 5;
        let mut c = Contingency::with_config(cfg).unwrap();

        let p1 = simple_plan("dd", TriggerType::MaxDrawdown, 0.10, AlertLevel::Warning);
        c.add_plan(p1).unwrap();

        let summary = c.evaluate(&stressed_state());
        assert!(!summary.escalated);
    }

    #[test]
    fn test_evaluate_worst_trigger_selected() {
        let mut c = Contingency::new();
        // Plan with two triggers — both will fire on stressed state
        let plan = ContingencyPlan {
            id: "multi".into(),
            name: "Multi trigger".into(),
            triggers: vec![
                TriggerCondition {
                    trigger_type: TriggerType::MaxDrawdown,
                    threshold: 0.10,
                    alert_level: AlertLevel::Warning,
                    description: "dd warning".into(),
                },
                TriggerCondition {
                    trigger_type: TriggerType::VolatilitySpike,
                    threshold: 0.40,
                    alert_level: AlertLevel::Critical,
                    description: "vol critical".into(),
                },
            ],
            actions: vec![FallbackAction::HaltNewEntries],
            enabled: true,
            cooldown_steps: 2,
        };
        c.add_plan(plan).unwrap();

        let summary = c.evaluate(&stressed_state());
        assert_eq!(summary.plans_triggered, 1);
        // The worst trigger (Critical) should be selected
        assert_eq!(summary.activations[0].alert_level, AlertLevel::Critical);
    }

    // -- Statistics tracking --

    #[test]
    fn test_stats_initial() {
        let c = Contingency::new();
        assert_eq!(c.stats().total_evaluations, 0);
        assert_eq!(c.stats().total_activations, 0);
    }

    #[test]
    fn test_stats_after_evaluation() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());

        assert_eq!(c.stats().total_evaluations, 1);
        assert_eq!(c.stats().total_activations, 1);
        assert!(c.stats().trigger_counts[0] > 0); // MaxDrawdown index = 0
    }

    #[test]
    fn test_stats_trigger_counts() {
        let mut c = Contingency::new();
        let p1 = simple_plan("dd", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        let p2 = simple_plan(
            "vol",
            TriggerType::VolatilitySpike,
            0.40,
            AlertLevel::Warning,
        );
        c.add_plan(p1).unwrap();
        c.add_plan(p2).unwrap();
        c.evaluate(&stressed_state());

        assert_eq!(c.stats().trigger_counts[0], 1); // MaxDrawdown
        assert_eq!(c.stats().trigger_counts[1], 1); // VolatilitySpike
    }

    #[test]
    fn test_stats_dominant_trigger() {
        let stats = ContingencyStats {
            trigger_counts: [5, 2, 0, 0, 1, 0],
            ..Default::default()
        };
        assert_eq!(stats.dominant_trigger(), TriggerType::MaxDrawdown);
    }

    #[test]
    fn test_stats_activation_rate() {
        let stats = ContingencyStats {
            total_evaluations: 10,
            total_activations: 3,
            ..Default::default()
        };
        let rate = stats.activation_rate();
        assert!((rate - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_stats_activation_rate_zero_evals() {
        let stats = ContingencyStats::default();
        assert_eq!(stats.activation_rate(), 0.0);
    }

    #[test]
    fn test_stats_escalation_rate() {
        let stats = ContingencyStats {
            total_evaluations: 20,
            total_escalations: 4,
            ..Default::default()
        };
        assert!((stats.escalation_rate() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_stats_peak_active_plans() {
        let mut cfg = ContingencyConfig::default();
        cfg.escalation_threshold = 100; // disable escalation for this test
        let mut c = Contingency::with_config(cfg).unwrap();
        let p1 = simple_plan("dd", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        let p2 = simple_plan(
            "vol",
            TriggerType::VolatilitySpike,
            0.40,
            AlertLevel::Warning,
        );
        c.add_plan(p1).unwrap();
        c.add_plan(p2).unwrap();
        c.evaluate(&stressed_state());
        assert!(c.stats().peak_active_plans >= 2);
    }

    // -- EMA tracking --

    #[test]
    fn test_ema_initializes_on_first_eval() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());
        // After first eval with 1 plan triggered out of 1: rate = 1.0
        assert!(c.smoothed_activation_rate() > 0.0);
    }

    #[test]
    fn test_ema_blends_on_subsequent_evals() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();

        // First: triggers (rate = 1.0)
        c.evaluate(&stressed_state());
        let rate1 = c.smoothed_activation_rate();

        // Second: cooldown, does not trigger (rate = 0.0)
        c.evaluate(&calm_state());
        let rate2 = c.smoothed_activation_rate();
        assert!(rate2 < rate1);
    }

    #[test]
    fn test_ema_alert_severity() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Emergency);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());
        assert!(c.smoothed_alert_severity() > 0.0);
    }

    #[test]
    fn test_ema_risk_impact() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());
        // Risk impact should be negative (risk reducing)
        assert!(c.smoothed_risk_impact() < 0.0);
    }

    // -- Sliding window --

    #[test]
    fn test_recent_activations_stored() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());
        assert_eq!(c.recent_activations().len(), 1);
    }

    #[test]
    fn test_recent_activations_windowed() {
        let mut cfg = ContingencyConfig::default();
        cfg.window_size = 3;
        let mut c = Contingency::with_config(cfg).unwrap();

        // Plan with cooldown 1 so it can re-trigger every other step
        let plan = ContingencyPlan {
            id: "p1".into(),
            name: "Test".into(),
            triggers: vec![TriggerCondition {
                trigger_type: TriggerType::MaxDrawdown,
                threshold: 0.10,
                alert_level: AlertLevel::Critical,
                description: "dd".into(),
            }],
            actions: vec![FallbackAction::HaltNewEntries],
            enabled: true,
            cooldown_steps: 1,
        };
        c.add_plan(plan).unwrap();

        // Trigger many times; window should cap at 3
        for _ in 0..10 {
            c.evaluate(&stressed_state());
        }

        assert!(c.recent_activations().len() <= 3);
    }

    #[test]
    fn test_windowed_activation_rate() {
        let mut cfg = ContingencyConfig::default();
        cfg.window_size = 10;
        let mut c = Contingency::with_config(cfg).unwrap();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());
        // 1 activation in window of size 10
        assert!((c.windowed_activation_rate() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_avg_severity() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());
        // Critical weight = 3.0
        assert!((c.windowed_avg_severity() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_windowed_empty() {
        let c = Contingency::new();
        assert_eq!(c.windowed_activation_rate(), 0.0);
        assert_eq!(c.windowed_avg_severity(), 0.0);
    }

    // -- Severity trend --

    #[test]
    fn test_is_severity_increasing_insufficient_data() {
        let c = Contingency::new();
        assert!(!c.is_severity_increasing());
    }

    // -- Position reduction & hedge --

    #[test]
    fn test_reduction_fraction_watch() {
        assert_eq!(Contingency::reduction_fraction(AlertLevel::Watch), 0.0);
    }

    #[test]
    fn test_reduction_fraction_warning() {
        assert!((Contingency::reduction_fraction(AlertLevel::Warning) - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_reduction_fraction_critical() {
        assert!((Contingency::reduction_fraction(AlertLevel::Critical) - 0.40).abs() < 1e-10);
    }

    #[test]
    fn test_reduction_fraction_emergency() {
        assert!((Contingency::reduction_fraction(AlertLevel::Emergency) - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_hedge_ratio_watch() {
        assert_eq!(Contingency::hedge_ratio(AlertLevel::Watch, 0.05), 0.0);
    }

    #[test]
    fn test_hedge_ratio_emergency() {
        let hr = Contingency::hedge_ratio(AlertLevel::Emergency, 0.05);
        assert!(hr > 0.0);
        assert!((hr - 0.05).abs() < 1e-10); // 1.0 * 0.05
    }

    #[test]
    fn test_hedge_ratio_capped_var() {
        let hr = Contingency::hedge_ratio(AlertLevel::Emergency, 2.0);
        // var capped at 1.0, so ratio = 1.0 * 1.0 = 1.0
        assert!((hr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cash_target_watch() {
        assert!((Contingency::cash_target(AlertLevel::Watch, 0.03) - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_cash_target_never_decreases() {
        // If current cash is already 0.60, target should stay at 0.60 even for Warning (0.15)
        assert!((Contingency::cash_target(AlertLevel::Warning, 0.60) - 0.60).abs() < 1e-10);
    }

    #[test]
    fn test_cash_target_emergency() {
        let target = Contingency::cash_target(AlertLevel::Emergency, 0.03);
        assert!((target - 0.50).abs() < 1e-10);
    }

    // -- Build response actions --

    #[test]
    fn test_build_response_watch() {
        let actions = Contingency::build_response_actions(AlertLevel::Watch, &calm_state());
        assert!(actions.is_empty());
    }

    #[test]
    fn test_build_response_warning() {
        let actions = Contingency::build_response_actions(AlertLevel::Warning, &stressed_state());
        assert!(actions.len() >= 2);
        // Should include Notify and HaltNewEntries
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, FallbackAction::Notify { .. }))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, FallbackAction::HaltNewEntries))
        );
    }

    #[test]
    fn test_build_response_critical() {
        let actions = Contingency::build_response_actions(AlertLevel::Critical, &stressed_state());
        // Should include position reduction and tighten stops in addition to warning actions
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, FallbackAction::ReducePosition { .. }))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, FallbackAction::TightenStops { .. }))
        );
    }

    #[test]
    fn test_build_response_emergency() {
        let actions = Contingency::build_response_actions(AlertLevel::Emergency, &stressed_state());
        // Emergency should have all action types
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, FallbackAction::CancelOpenOrders))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, FallbackAction::RaiseCash { .. }))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, FallbackAction::ReducePosition { .. }))
        );
    }

    // -- Fallback action risk impact --

    #[test]
    fn test_reduce_position_risk_impact() {
        let a = FallbackAction::ReducePosition { fraction: 0.50 };
        assert!((a.risk_impact_estimate() - (-0.50)).abs() < 1e-10);
    }

    #[test]
    fn test_add_hedge_risk_impact() {
        let a = FallbackAction::AddHedge { hedge_ratio: 0.25 };
        assert!(a.risk_impact_estimate() < 0.0);
    }

    #[test]
    fn test_notify_zero_risk_impact() {
        let a = FallbackAction::Notify {
            message: "test".into(),
        };
        assert_eq!(a.risk_impact_estimate(), 0.0);
    }

    // -- Alert level --

    #[test]
    fn test_alert_level_ordering() {
        assert!(AlertLevel::Watch < AlertLevel::Warning);
        assert!(AlertLevel::Warning < AlertLevel::Critical);
        assert!(AlertLevel::Critical < AlertLevel::Emergency);
    }

    #[test]
    fn test_alert_level_requires_action() {
        assert!(!AlertLevel::Watch.requires_action());
        assert!(!AlertLevel::Warning.requires_action());
        assert!(AlertLevel::Critical.requires_action());
        assert!(AlertLevel::Emergency.requires_action());
    }

    #[test]
    fn test_alert_level_weight() {
        assert!(AlertLevel::Emergency.weight() > AlertLevel::Critical.weight());
        assert!(AlertLevel::Critical.weight() > AlertLevel::Warning.weight());
        assert!(AlertLevel::Warning.weight() > AlertLevel::Watch.weight());
    }

    // -- Preset plans --

    #[test]
    fn test_preset_drawdown_plan() {
        let plan = preset_drawdown_plan();
        assert!(!plan.triggers.is_empty());
        assert!(!plan.actions.is_empty());
        assert!(plan.enabled);
        assert_eq!(plan.id, "drawdown_standard");
    }

    #[test]
    fn test_preset_volatility_plan() {
        let plan = preset_volatility_plan();
        assert!(!plan.triggers.is_empty());
        assert!(!plan.actions.is_empty());
        assert!(plan.enabled);
    }

    #[test]
    fn test_preset_liquidity_plan() {
        let plan = preset_liquidity_plan();
        assert!(!plan.triggers.is_empty());
        assert!(
            plan.triggers
                .iter()
                .any(|t| t.trigger_type == TriggerType::LiquidityCrisis)
        );
    }

    #[test]
    fn test_preset_plans_can_be_added() {
        let mut c = Contingency::new();
        assert!(c.add_plan(preset_drawdown_plan()).is_ok());
        assert!(c.add_plan(preset_volatility_plan()).is_ok());
        assert!(c.add_plan(preset_liquidity_plan()).is_ok());
        assert_eq!(c.plan_count(), 3);
    }

    #[test]
    fn test_preset_plans_trigger_on_stress() {
        let mut c = Contingency::new();
        c.add_plan(preset_drawdown_plan()).unwrap();
        c.add_plan(preset_volatility_plan()).unwrap();
        c.add_plan(preset_liquidity_plan()).unwrap();

        let summary = c.evaluate(&stressed_state());
        // All three should trigger on the stressed state
        assert_eq!(summary.plans_triggered, 3);
    }

    // -- Reset --

    #[test]
    fn test_reset() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        c.evaluate(&stressed_state());

        assert!(c.current_step() > 0);
        assert!(c.stats().total_evaluations > 0);
        assert!(!c.recent_activations().is_empty());

        c.reset();

        assert_eq!(c.current_step(), 0);
        assert_eq!(c.stats().total_evaluations, 0);
        assert!(c.recent_activations().is_empty());
        assert!(c.active_plans().is_empty());
        // Plans are preserved
        assert_eq!(c.plan_count(), 1);
    }

    // -- Step counter --

    #[test]
    fn test_step_increments() {
        let mut c = Contingency::new();
        assert_eq!(c.current_step(), 0);
        c.evaluate(&calm_state());
        assert_eq!(c.current_step(), 1);
        c.evaluate(&calm_state());
        assert_eq!(c.current_step(), 2);
    }

    // -- Active plan tracking --

    #[test]
    fn test_active_plans_populated() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();
        assert!(c.active_plans().is_empty());

        c.evaluate(&stressed_state());
        assert!(c.active_plans().contains(&"p1".to_string()));
    }

    #[test]
    fn test_active_plans_cleared_after_cooldown() {
        let mut c = Contingency::new();
        let plan = simple_plan("p1", TriggerType::MaxDrawdown, 0.10, AlertLevel::Critical);
        c.add_plan(plan).unwrap();

        c.evaluate(&stressed_state()); // triggers, cooldown = 2
        assert!(!c.active_plans().is_empty());

        // Calm evals to let cooldown expire and plan deactivate
        c.evaluate(&calm_state()); // cooldown = 1, still active
        c.evaluate(&calm_state()); // cooldown = 0, no trigger, deactivated
        c.evaluate(&calm_state()); // another calm eval
        assert!(c.active_plans().is_empty());
    }

    // -- Trigger type indexing --

    #[test]
    fn test_trigger_type_index_roundtrip() {
        let types = [
            TriggerType::MaxDrawdown,
            TriggerType::VolatilitySpike,
            TriggerType::CorrelationBreak,
            TriggerType::VarBreach,
            TriggerType::LiquidityCrisis,
            TriggerType::Custom,
        ];
        for (i, tt) in types.iter().enumerate() {
            assert_eq!(trigger_type_index(tt), i);
            assert_eq!(trigger_type_from_index(i), *tt);
        }
    }

    // -- MarketState default --

    #[test]
    fn test_market_state_default() {
        let ms = MarketState::default();
        assert!(ms.portfolio_value > 0.0);
        assert!(ms.volatility > 0.0);
        assert!(ms.drawdown >= 0.0);
    }

    // -- Edge case: custom trigger never fires on its own --

    #[test]
    fn test_custom_trigger_does_not_fire_automatically() {
        let trigger = TriggerCondition {
            trigger_type: TriggerType::Custom,
            threshold: 0.5,
            alert_level: AlertLevel::Warning,
            description: "custom".into(),
        };
        let result = Contingency::evaluate_trigger(&trigger, &stressed_state());
        // Custom observed = 0.0 < 0.5
        assert!(!result.fired);
    }

    // -- Integration-style test --

    #[test]
    fn test_full_lifecycle() {
        let mut c = Contingency::new();

        // Add presets
        c.add_plan(preset_drawdown_plan()).unwrap();
        c.add_plan(preset_volatility_plan()).unwrap();
        c.add_plan(preset_liquidity_plan()).unwrap();

        // Phase 1: calm market — nothing triggers
        for _ in 0..5 {
            let summary = c.evaluate(&calm_state());
            assert_eq!(summary.plans_triggered, 0);
        }

        // Phase 2: stress hits
        let summary = c.evaluate(&stressed_state());
        assert!(summary.plans_triggered > 0);
        assert!(summary.aggregate_risk_impact < 0.0);
        assert!(c.stats().total_activations > 0);

        // Phase 3: cooldown period — same stress but plans cooling
        let summary2 = c.evaluate(&stressed_state());
        assert!(summary2.plans_triggered < summary.plans_triggered);

        // Phase 4: back to calm after cooldown
        for _ in 0..25 {
            c.evaluate(&calm_state());
        }

        // Verify stats accumulated correctly
        assert!(c.stats().total_evaluations > 30);
        assert!(c.smoothed_activation_rate() < 1.0);
        assert!(c.windowed_activation_rate() >= 0.0);
    }
}
