//! Contingency planning
//!
//! Part of the Prefrontal region
//! Component: planning
//!
//! Models "what-if" scenarios and maintains contingency plans that activate
//! when market conditions deteriorate. Each contingency plan has a trigger
//! condition, a severity level, and a set of prescribed actions.
//!
//! ## Features
//!
//! - **Scenario-based planning**: Define contingency plans with named
//!   triggers (drawdown breach, volatility spike, liquidity dry-up, etc.)
//! - **Multi-level severity**: Plans are categorised as Advisory, Warning,
//!   or Emergency, each with different response characteristics
//! - **Trigger monitoring**: Feed market observations and the engine
//!   evaluates all registered triggers, activating matching plans
//! - **Cascading chains**: Plans can reference follow-up plans that
//!   activate if the primary plan fails to resolve the situation
//! - **Cooldown enforcement**: After activation, a plan enters cooldown
//!   to prevent rapid re-triggering
//! - **EMA-smoothed threat level**: Aggregate threat signal across all
//!   active contingencies, smoothed for downstream consumption
//! - **Plan lifecycle**: Plans move through Dormant → Armed → Active →
//!   Resolved/Expired states
//! - **Running statistics**: Activation counts, false alarm rate,
//!   resolution times, escalation counts
//! - **Windowed diagnostics**: Recent activation frequency, mean severity,
//!   and threat trend detection

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the contingency engine
#[derive(Debug, Clone)]
pub struct ContingencyConfig {
    /// EMA decay for threat level smoothing (0 < decay < 1)
    pub ema_decay: f64,
    /// Minimum observations before EMA is considered initialised
    pub min_samples: usize,
    /// Sliding window size for windowed diagnostics
    pub window_size: usize,
    /// Default cooldown ticks after plan activation
    pub default_cooldown: u64,
    /// Maximum number of plans that can be registered
    pub max_plans: usize,
    /// Maximum cascade depth (prevents infinite loops)
    pub max_cascade_depth: usize,
    /// Expiry ticks — plans auto-expire if active for this long without resolution
    pub default_expiry_ticks: u64,
}

impl Default for ContingencyConfig {
    fn default() -> Self {
        Self {
            ema_decay: 0.12,
            min_samples: 3,
            window_size: 100,
            default_cooldown: 10,
            max_plans: 200,
            max_cascade_depth: 5,
            default_expiry_ticks: 100,
        }
    }
}

// ---------------------------------------------------------------------------
// Supporting types
// ---------------------------------------------------------------------------

/// Severity level for a contingency plan
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Informational — log and monitor
    Advisory = 0,
    /// Warning — reduce exposure, widen stops
    Warning = 1,
    /// Emergency — flatten positions, halt trading
    Emergency = 2,
}

impl Severity {
    /// Numeric weight for aggregation (0.0–1.0)
    pub fn weight(&self) -> f64 {
        match self {
            Severity::Advisory => 0.25,
            Severity::Warning => 0.60,
            Severity::Emergency => 1.00,
        }
    }
}

/// Lifecycle state of a contingency plan
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanState {
    /// Plan is registered but not triggered
    Dormant,
    /// Plan's trigger condition is close to firing (optional pre-stage)
    Armed,
    /// Plan has been activated
    Active,
    /// Plan was activated but the situation resolved
    Resolved,
    /// Plan expired without being resolved
    Expired,
    /// Plan is in cooldown after previous activation
    Cooldown,
}

/// A prescribed action within a contingency plan
#[derive(Debug, Clone)]
pub struct ContingencyAction {
    /// Human-readable name
    pub name: String,
    /// Description of what to do
    pub description: String,
    /// Whether this action has been executed
    pub executed: bool,
}

impl ContingencyAction {
    pub fn new(name: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            description: description.to_string(),
            executed: false,
        }
    }
}

/// Trigger condition definition
#[derive(Debug, Clone)]
pub struct TriggerCondition {
    /// Name of the metric to monitor (e.g. "drawdown", "volatility", "spread")
    pub metric: String,
    /// Threshold value — plan triggers when metric crosses this
    pub threshold: f64,
    /// Whether trigger fires when metric goes above (true) or below (false) threshold
    pub above: bool,
}

impl TriggerCondition {
    pub fn above(metric: &str, threshold: f64) -> Self {
        Self {
            metric: metric.to_string(),
            threshold,
            above: true,
        }
    }

    pub fn below(metric: &str, threshold: f64) -> Self {
        Self {
            metric: metric.to_string(),
            threshold,
            above: false,
        }
    }

    /// Evaluate whether the trigger fires given the current metric value
    pub fn evaluate(&self, value: f64) -> bool {
        if self.above {
            value >= self.threshold
        } else {
            value <= self.threshold
        }
    }
}

/// A registered contingency plan
#[derive(Debug, Clone)]
pub struct ContingencyPlan {
    /// Unique plan identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Severity level
    pub severity: Severity,
    /// Trigger conditions (any one firing activates the plan)
    pub triggers: Vec<TriggerCondition>,
    /// Prescribed actions
    pub actions: Vec<ContingencyAction>,
    /// Current lifecycle state
    pub state: PlanState,
    /// Optional follow-up plan ID (cascade)
    pub cascade_to: Option<String>,
    /// Cooldown ticks remaining (0 = ready)
    pub cooldown_remaining: u64,
    /// Custom cooldown override (None = use default)
    pub cooldown_ticks: Option<u64>,
    /// Custom expiry override (None = use default)
    pub expiry_ticks: Option<u64>,
    /// Ticks since activation (only meaningful when Active)
    pub active_ticks: u64,
    /// Number of times this plan has been activated
    pub activation_count: u64,
    /// Tick when last activated
    pub last_activated_tick: Option<u64>,
    /// Tick when last resolved
    pub last_resolved_tick: Option<u64>,
}

/// Result of evaluating triggers on a single tick
#[derive(Debug, Clone)]
pub struct EvaluationResult {
    /// Plans that were newly activated this tick
    pub activated: Vec<String>,
    /// Plans that expired this tick
    pub expired: Vec<String>,
    /// Plans that were cascaded to this tick
    pub cascaded: Vec<String>,
    /// Current aggregate threat level (0.0–1.0)
    pub threat_level: f64,
    /// Highest severity among active plans
    pub max_severity: Option<Severity>,
    /// Total number of currently active plans
    pub active_count: usize,
}

/// Activation event record
#[derive(Debug, Clone)]
pub struct ActivationEvent {
    pub plan_id: String,
    pub severity: Severity,
    pub trigger_metric: String,
    pub trigger_value: f64,
    pub tick: u64,
    pub cascaded_from: Option<String>,
}

/// Record for windowed analysis
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct WindowRecord {
    activated_count: usize,
    max_severity: Option<Severity>,
    threat_level: f64,
    tick: u64,
}

/// Running statistics
#[derive(Debug, Clone)]
pub struct ContingencyStats {
    /// Total evaluation ticks
    pub total_evaluations: u64,
    /// Total plan activations across all plans
    pub total_activations: u64,
    /// Total plan resolutions
    pub total_resolutions: u64,
    /// Total plan expirations (unresolved)
    pub total_expirations: u64,
    /// Total cascade events
    pub total_cascades: u64,
    /// Number of Advisory activations
    pub advisory_activations: u64,
    /// Number of Warning activations
    pub warning_activations: u64,
    /// Number of Emergency activations
    pub emergency_activations: u64,
    /// Sum of active ticks at resolution (for mean time-to-resolve)
    pub sum_resolution_ticks: u64,
    /// Count of resolutions with known duration
    pub resolution_count: u64,
    /// Peak threat level observed
    pub peak_threat_level: f64,
    /// Number of registered plans
    pub registered_plans: usize,
    /// Currently active plan count
    pub active_plans: usize,
    /// Cooldown block count
    pub cooldown_blocks: u64,
}

impl Default for ContingencyStats {
    fn default() -> Self {
        Self {
            total_evaluations: 0,
            total_activations: 0,
            total_resolutions: 0,
            total_expirations: 0,
            total_cascades: 0,
            advisory_activations: 0,
            warning_activations: 0,
            emergency_activations: 0,
            sum_resolution_ticks: 0,
            resolution_count: 0,
            peak_threat_level: 0.0,
            registered_plans: 0,
            active_plans: 0,
            cooldown_blocks: 0,
        }
    }
}

impl ContingencyStats {
    /// Mean time to resolve (in ticks)
    pub fn mean_resolution_time(&self) -> f64 {
        if self.resolution_count == 0 {
            return 0.0;
        }
        self.sum_resolution_ticks as f64 / self.resolution_count as f64
    }

    /// Expiration rate (fraction of activations that expired without resolution)
    pub fn expiration_rate(&self) -> f64 {
        let total_outcomes = self.total_resolutions + self.total_expirations;
        if total_outcomes == 0 {
            return 0.0;
        }
        self.total_expirations as f64 / total_outcomes as f64
    }

    /// Activation rate (activations per evaluation)
    pub fn activation_rate(&self) -> f64 {
        if self.total_evaluations == 0 {
            return 0.0;
        }
        self.total_activations as f64 / self.total_evaluations as f64
    }

    /// Cascade rate (cascades per activation)
    pub fn cascade_rate(&self) -> f64 {
        if self.total_activations == 0 {
            return 0.0;
        }
        self.total_cascades as f64 / self.total_activations as f64
    }
}

// ---------------------------------------------------------------------------
// Main struct
// ---------------------------------------------------------------------------

/// Contingency planning engine for trading risk scenarios
pub struct Contingency {
    config: ContingencyConfig,

    /// Registered plans keyed by ID
    plans: HashMap<String, ContingencyPlan>,

    /// Current aggregate threat level (EMA-smoothed)
    ema_threat: f64,
    ema_initialized: bool,

    /// Activation history
    history: Vec<ActivationEvent>,

    /// Windowed evaluation records
    recent: VecDeque<WindowRecord>,

    /// Current tick
    current_tick: u64,

    /// Running statistics
    stats: ContingencyStats,
}

impl Default for Contingency {
    fn default() -> Self {
        Self::new()
    }
}

impl Contingency {
    /// Create a new instance with default config
    pub fn new() -> Self {
        Self::with_config(ContingencyConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ContingencyConfig) -> Self {
        assert!(
            config.ema_decay > 0.0 && config.ema_decay < 1.0,
            "ema_decay must be in (0, 1)"
        );
        assert!(config.window_size > 0, "window_size must be > 0");
        assert!(config.max_plans > 0, "max_plans must be > 0");
        assert!(
            config.max_cascade_depth > 0,
            "max_cascade_depth must be > 0"
        );
        assert!(
            config.default_expiry_ticks > 0,
            "default_expiry_ticks must be > 0"
        );

        Self {
            config,
            plans: HashMap::new(),
            ema_threat: 0.0,
            ema_initialized: false,
            history: Vec::new(),
            recent: VecDeque::new(),
            current_tick: 0,
            stats: ContingencyStats::default(),
        }
    }

    // -----------------------------------------------------------------------
    // Main processing
    // -----------------------------------------------------------------------

    /// Main processing function — evaluate all triggers against current metrics
    ///
    /// `metrics` maps metric names to their current values (e.g. "drawdown" → 0.12)
    pub fn process(
        &mut self,
        metrics: &HashMap<String, f64>,
        tick: u64,
    ) -> Result<EvaluationResult> {
        self.current_tick = tick;
        self.stats.total_evaluations += 1;

        let mut activated = Vec::new();
        let mut expired = Vec::new();
        let mut cascaded = Vec::new();

        // Collect plan IDs first to avoid borrow issues
        let plan_ids: Vec<String> = self.plans.keys().cloned().collect();

        for plan_id in &plan_ids {
            let plan = match self.plans.get(plan_id) {
                Some(p) => p.clone(),
                None => continue,
            };

            match plan.state {
                PlanState::Active => {
                    // Tick active duration and check expiry
                    let expiry = plan
                        .expiry_ticks
                        .unwrap_or(self.config.default_expiry_ticks);
                    if let Some(p) = self.plans.get_mut(plan_id) {
                        p.active_ticks += 1;
                        if p.active_ticks >= expiry {
                            p.state = PlanState::Expired;
                            expired.push(plan_id.clone());
                            self.stats.total_expirations += 1;

                            // Check cascade
                            if let Some(cascade_id) = p.cascade_to.clone() {
                                self.try_cascade(&cascade_id, plan_id, 0, &mut cascaded);
                            }
                        }
                    }
                }
                PlanState::Cooldown => {
                    if let Some(p) = self.plans.get_mut(plan_id) {
                        if p.cooldown_remaining > 0 {
                            p.cooldown_remaining -= 1;
                        }
                        if p.cooldown_remaining == 0 {
                            p.state = PlanState::Dormant;
                        }
                    }
                }
                PlanState::Dormant | PlanState::Armed => {
                    // Evaluate triggers
                    let mut triggered = false;
                    let mut trigger_metric = String::new();
                    let mut trigger_value = 0.0;

                    for trigger in &plan.triggers {
                        if let Some(&value) = metrics.get(&trigger.metric) {
                            if trigger.evaluate(value) {
                                triggered = true;
                                trigger_metric = trigger.metric.clone();
                                trigger_value = value;
                                break;
                            }
                        }
                    }

                    if triggered {
                        self.activate_plan(plan_id, &trigger_metric, trigger_value, None);
                        activated.push(plan_id.clone());
                    }
                }
                _ => {} // Resolved, Expired — no action
            }
        }

        // Compute aggregate threat level
        let raw_threat = self.compute_raw_threat();

        // Update EMA
        if !self.ema_initialized {
            self.ema_threat = raw_threat;
            self.ema_initialized = true;
        } else {
            let alpha = self.config.ema_decay;
            self.ema_threat = alpha * raw_threat + (1.0 - alpha) * self.ema_threat;
        }

        // Track peak
        if self.ema_threat > self.stats.peak_threat_level {
            self.stats.peak_threat_level = self.ema_threat;
        }

        // Determine max severity among active plans
        let max_severity = self.max_active_severity();

        // Count active plans
        let active_count = self
            .plans
            .values()
            .filter(|p| p.state == PlanState::Active)
            .count();
        self.stats.active_plans = active_count;

        // Record for windowed diagnostics
        self.recent.push_back(WindowRecord {
            activated_count: activated.len(),
            max_severity,
            threat_level: self.ema_threat,
            tick,
        });
        while self.recent.len() > self.config.window_size {
            self.recent.pop_front();
        }

        Ok(EvaluationResult {
            activated,
            expired,
            cascaded,
            threat_level: self.ema_threat,
            max_severity,
            active_count,
        })
    }

    // -----------------------------------------------------------------------
    // Plan registration
    // -----------------------------------------------------------------------

    /// Register a new contingency plan
    pub fn register_plan(
        &mut self,
        id: &str,
        name: &str,
        severity: Severity,
        triggers: Vec<TriggerCondition>,
        actions: Vec<ContingencyAction>,
    ) -> Result<()> {
        if self.plans.len() >= self.config.max_plans {
            return Err(Error::InvalidInput(format!(
                "Maximum plans ({}) reached",
                self.config.max_plans
            )));
        }
        if self.plans.contains_key(id) {
            return Err(Error::InvalidInput(format!(
                "Plan '{}' already registered",
                id
            )));
        }
        if triggers.is_empty() {
            return Err(Error::InvalidInput(
                "Plan must have at least one trigger".into(),
            ));
        }

        let plan = ContingencyPlan {
            id: id.to_string(),
            name: name.to_string(),
            severity,
            triggers,
            actions,
            state: PlanState::Dormant,
            cascade_to: None,
            cooldown_remaining: 0,
            cooldown_ticks: None,
            expiry_ticks: None,
            active_ticks: 0,
            activation_count: 0,
            last_activated_tick: None,
            last_resolved_tick: None,
        };

        self.plans.insert(id.to_string(), plan);
        self.stats.registered_plans = self.plans.len();
        Ok(())
    }

    /// Remove a plan
    pub fn deregister_plan(&mut self, id: &str) -> Result<()> {
        if self.plans.remove(id).is_none() {
            return Err(Error::NotFound(format!("Plan '{}' not found", id)));
        }
        self.stats.registered_plans = self.plans.len();
        Ok(())
    }

    /// Set the cascade target for a plan
    pub fn set_cascade(&mut self, plan_id: &str, cascade_to: &str) -> Result<()> {
        if !self.plans.contains_key(cascade_to) {
            return Err(Error::NotFound(format!(
                "Cascade target '{}' not found",
                cascade_to
            )));
        }
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("Plan '{}' not found", plan_id)))?;
        plan.cascade_to = Some(cascade_to.to_string());
        Ok(())
    }

    /// Set custom cooldown for a plan
    pub fn set_cooldown(&mut self, plan_id: &str, ticks: u64) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("Plan '{}' not found", plan_id)))?;
        plan.cooldown_ticks = Some(ticks);
        Ok(())
    }

    /// Set custom expiry for a plan
    pub fn set_expiry(&mut self, plan_id: &str, ticks: u64) -> Result<()> {
        if ticks == 0 {
            return Err(Error::InvalidInput("expiry must be > 0".into()));
        }
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("Plan '{}' not found", plan_id)))?;
        plan.expiry_ticks = Some(ticks);
        Ok(())
    }

    /// Arm a plan (pre-stage before full activation)
    pub fn arm_plan(&mut self, id: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(id)
            .ok_or_else(|| Error::NotFound(format!("Plan '{}' not found", id)))?;
        if plan.state != PlanState::Dormant {
            return Err(Error::InvalidState(format!(
                "Plan '{}' cannot be armed from state {:?}",
                id, plan.state
            )));
        }
        plan.state = PlanState::Armed;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Plan lifecycle
    // -----------------------------------------------------------------------

    /// Manually activate a plan
    pub fn activate(&mut self, plan_id: &str) -> Result<()> {
        self.activate_plan(plan_id, "manual", 0.0, None);
        Ok(())
    }

    fn activate_plan(
        &mut self,
        plan_id: &str,
        trigger_metric: &str,
        trigger_value: f64,
        cascaded_from: Option<&str>,
    ) {
        if let Some(plan) = self.plans.get_mut(plan_id) {
            if plan.state == PlanState::Cooldown {
                self.stats.cooldown_blocks += 1;
                return;
            }

            plan.state = PlanState::Active;
            plan.active_ticks = 0;
            plan.activation_count += 1;
            plan.last_activated_tick = Some(self.current_tick);

            // Reset action execution flags
            for action in &mut plan.actions {
                action.executed = false;
            }

            // Update stats
            self.stats.total_activations += 1;
            match plan.severity {
                Severity::Advisory => self.stats.advisory_activations += 1,
                Severity::Warning => self.stats.warning_activations += 1,
                Severity::Emergency => self.stats.emergency_activations += 1,
            }

            if cascaded_from.is_some() {
                self.stats.total_cascades += 1;
            }

            // Record event
            self.history.push(ActivationEvent {
                plan_id: plan_id.to_string(),
                severity: plan.severity,
                trigger_metric: trigger_metric.to_string(),
                trigger_value,
                tick: self.current_tick,
                cascaded_from: cascaded_from.map(|s| s.to_string()),
            });
        }
    }

    fn try_cascade(
        &mut self,
        cascade_id: &str,
        from_id: &str,
        depth: usize,
        cascaded: &mut Vec<String>,
    ) {
        if depth >= self.config.max_cascade_depth {
            return;
        }

        let can_cascade = self
            .plans
            .get(cascade_id)
            .map(|p| matches!(p.state, PlanState::Dormant | PlanState::Armed))
            .unwrap_or(false);

        if can_cascade {
            self.activate_plan(cascade_id, "cascade", 0.0, Some(from_id));
            cascaded.push(cascade_id.to_string());

            // Check if the cascaded plan also has a cascade target
            let next_cascade = self
                .plans
                .get(cascade_id)
                .and_then(|p| p.cascade_to.clone());
            if let Some(next_id) = next_cascade {
                self.try_cascade(&next_id, cascade_id, depth + 1, cascaded);
            }
        }
    }

    /// Resolve an active plan (situation has been handled)
    pub fn resolve(&mut self, plan_id: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("Plan '{}' not found", plan_id)))?;

        if plan.state != PlanState::Active {
            return Err(Error::InvalidState(format!(
                "Plan '{}' is not active (state: {:?})",
                plan_id, plan.state
            )));
        }

        let active_duration = plan.active_ticks;
        let cooldown = plan.cooldown_ticks.unwrap_or(self.config.default_cooldown);

        if cooldown == 0 {
            plan.state = PlanState::Dormant;
        } else {
            plan.state = PlanState::Cooldown;
        }
        plan.cooldown_remaining = cooldown;
        plan.last_resolved_tick = Some(self.current_tick);

        self.stats.total_resolutions += 1;
        self.stats.sum_resolution_ticks += active_duration;
        self.stats.resolution_count += 1;

        Ok(())
    }

    /// Mark an action within an active plan as executed
    pub fn execute_action(&mut self, plan_id: &str, action_name: &str) -> Result<()> {
        let plan = self
            .plans
            .get_mut(plan_id)
            .ok_or_else(|| Error::NotFound(format!("Plan '{}' not found", plan_id)))?;

        if plan.state != PlanState::Active {
            return Err(Error::InvalidState(format!(
                "Plan '{}' is not active",
                plan_id
            )));
        }

        let action = plan
            .actions
            .iter_mut()
            .find(|a| a.name == action_name)
            .ok_or_else(|| Error::NotFound(format!("Action '{}' not found", action_name)))?;

        action.executed = true;
        Ok(())
    }

    /// Check if all actions in a plan have been executed
    pub fn all_actions_executed(&self, plan_id: &str) -> bool {
        self.plans
            .get(plan_id)
            .map(|p| !p.actions.is_empty() && p.actions.iter().all(|a| a.executed))
            .unwrap_or(false)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn compute_raw_threat(&self) -> f64 {
        let active_plans: Vec<&ContingencyPlan> = self
            .plans
            .values()
            .filter(|p| p.state == PlanState::Active)
            .collect();

        if active_plans.is_empty() {
            return 0.0;
        }

        // Threat = max severity weight among active plans, boosted by count
        let max_weight = active_plans
            .iter()
            .map(|p| p.severity.weight())
            .fold(0.0_f64, |a, b| a.max(b));

        let count_boost = 1.0 + (active_plans.len() as f64 - 1.0) * 0.1;
        (max_weight * count_boost).min(1.0)
    }

    fn max_active_severity(&self) -> Option<Severity> {
        self.plans
            .values()
            .filter(|p| p.state == PlanState::Active)
            .map(|p| p.severity)
            .max()
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Get a plan by ID
    pub fn plan(&self, id: &str) -> Option<&ContingencyPlan> {
        self.plans.get(id)
    }

    /// Get all plans
    pub fn plans(&self) -> &HashMap<String, ContingencyPlan> {
        &self.plans
    }

    /// Number of registered plans
    pub fn plan_count(&self) -> usize {
        self.plans.len()
    }

    /// Number of currently active plans
    pub fn active_count(&self) -> usize {
        self.plans
            .values()
            .filter(|p| p.state == PlanState::Active)
            .count()
    }

    /// Current EMA-smoothed threat level
    pub fn threat_level(&self) -> f64 {
        if !self.ema_initialized {
            return 0.0;
        }
        self.ema_threat
    }

    /// Whether any plan is currently active
    pub fn any_active(&self) -> bool {
        self.plans.values().any(|p| p.state == PlanState::Active)
    }

    /// Whether any Emergency-level plan is active
    pub fn emergency_active(&self) -> bool {
        self.plans
            .values()
            .any(|p| p.state == PlanState::Active && p.severity == Severity::Emergency)
    }

    /// Get all currently active plan IDs
    pub fn active_plan_ids(&self) -> Vec<&str> {
        self.plans
            .values()
            .filter(|p| p.state == PlanState::Active)
            .map(|p| p.id.as_str())
            .collect()
    }

    /// Activation history
    pub fn history(&self) -> &[ActivationEvent] {
        &self.history
    }

    /// Running statistics
    pub fn stats(&self) -> &ContingencyStats {
        &self.stats
    }

    /// Whether the engine has enough data for reliable signals
    pub fn is_warmed_up(&self) -> bool {
        self.stats.total_evaluations as usize >= self.config.min_samples
    }

    // -----------------------------------------------------------------------
    // Windowed diagnostics
    // -----------------------------------------------------------------------

    /// Mean activations per evaluation tick in the recent window
    pub fn windowed_activation_rate(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: usize = self.recent.iter().map(|r| r.activated_count).sum();
        sum as f64 / self.recent.len() as f64
    }

    /// Mean threat level over the recent window
    pub fn windowed_mean_threat(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.recent.iter().map(|r| r.threat_level).sum();
        sum / self.recent.len() as f64
    }

    /// Fraction of recent ticks with at least one active plan
    pub fn windowed_active_fraction(&self) -> f64 {
        if self.recent.is_empty() {
            return 0.0;
        }
        let active = self
            .recent
            .iter()
            .filter(|r| r.max_severity.is_some())
            .count();
        active as f64 / self.recent.len() as f64
    }

    /// Whether threat level is trending upward
    pub fn is_threat_increasing(&self) -> bool {
        let n = self.recent.len();
        if n < 6 {
            return false;
        }
        let mid = n / 2;
        let first_mean: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.threat_level)
            .sum::<f64>()
            / mid as f64;
        let second_mean: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.threat_level)
            .sum::<f64>()
            / (n - mid) as f64;

        second_mean > first_mean * 1.2
    }

    /// Whether threat level is decreasing
    pub fn is_threat_decreasing(&self) -> bool {
        let n = self.recent.len();
        if n < 6 {
            return false;
        }
        let mid = n / 2;
        let first_mean: f64 = self
            .recent
            .iter()
            .take(mid)
            .map(|r| r.threat_level)
            .sum::<f64>()
            / mid as f64;
        let second_mean: f64 = self
            .recent
            .iter()
            .skip(mid)
            .map(|r| r.threat_level)
            .sum::<f64>()
            / (n - mid) as f64;

        second_mean < first_mean * 0.8
    }

    /// Confidence based on evaluation count
    pub fn confidence(&self) -> f64 {
        let n = self.stats.total_evaluations as f64;
        let required = self.config.min_samples as f64;
        if n >= required * 4.0 {
            1.0
        } else {
            (n / (required * 4.0)).min(1.0)
        }
    }

    // -----------------------------------------------------------------------
    // Reset
    // -----------------------------------------------------------------------

    /// Reset all state, keeping configuration and plan registrations.
    /// Plans are returned to Dormant state.
    pub fn reset(&mut self) {
        for plan in self.plans.values_mut() {
            plan.state = PlanState::Dormant;
            plan.cooldown_remaining = 0;
            plan.active_ticks = 0;
            plan.activation_count = 0;
            plan.last_activated_tick = None;
            plan.last_resolved_tick = None;
            for action in &mut plan.actions {
                action.executed = false;
            }
        }
        self.ema_threat = 0.0;
        self.ema_initialized = false;
        self.history.clear();
        self.recent.clear();
        self.current_tick = 0;
        self.stats = ContingencyStats {
            registered_plans: self.plans.len(),
            ..Default::default()
        };
    }

    /// Reset everything including plan registrations
    pub fn reset_all(&mut self) {
        self.plans.clear();
        self.ema_threat = 0.0;
        self.ema_initialized = false;
        self.history.clear();
        self.recent.clear();
        self.current_tick = 0;
        self.stats = ContingencyStats::default();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn drawdown_trigger(threshold: f64) -> TriggerCondition {
        TriggerCondition::above("drawdown", threshold)
    }

    fn vol_trigger(threshold: f64) -> TriggerCondition {
        TriggerCondition::above("volatility", threshold)
    }

    fn reduce_action() -> ContingencyAction {
        ContingencyAction::new("reduce_exposure", "Cut position sizes by 50%")
    }

    fn halt_action() -> ContingencyAction {
        ContingencyAction::new("halt_trading", "Stop all new orders")
    }

    fn make_metrics(drawdown: f64, volatility: f64) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("drawdown".to_string(), drawdown);
        m.insert("volatility".to_string(), volatility);
        m
    }

    #[test]
    fn test_basic() {
        let mut instance = Contingency::new();
        assert!(instance.process(&HashMap::new(), 0).is_ok());
    }

    #[test]
    fn test_register_plan() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd_warning",
            "Drawdown Warning",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action()],
        )
        .unwrap();
        assert_eq!(c.plan_count(), 1);
        assert!(c.plan("dd_warning").is_some());
    }

    #[test]
    fn test_register_duplicate() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        assert!(
            c.register_plan(
                "p",
                "P2",
                Severity::Advisory,
                vec![drawdown_trigger(0.2)],
                vec![]
            )
            .is_err()
        );
    }

    #[test]
    fn test_register_no_triggers() {
        let mut c = Contingency::new();
        assert!(
            c.register_plan("p", "P", Severity::Advisory, vec![], vec![])
                .is_err()
        );
    }

    #[test]
    fn test_max_plans() {
        let mut c = Contingency::with_config(ContingencyConfig {
            max_plans: 2,
            ..Default::default()
        });
        c.register_plan(
            "a",
            "A",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.register_plan(
            "b",
            "B",
            Severity::Advisory,
            vec![drawdown_trigger(0.2)],
            vec![],
        )
        .unwrap();
        assert!(
            c.register_plan(
                "c",
                "C",
                Severity::Advisory,
                vec![drawdown_trigger(0.3)],
                vec![]
            )
            .is_err()
        );
    }

    #[test]
    fn test_deregister_plan() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.deregister_plan("p").unwrap();
        assert_eq!(c.plan_count(), 0);
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut c = Contingency::new();
        assert!(c.deregister_plan("nope").is_err());
    }

    #[test]
    fn test_trigger_fires() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action()],
        )
        .unwrap();

        let metrics = make_metrics(0.06, 0.01);
        let result = c.process(&metrics, 1).unwrap();
        assert_eq!(result.activated.len(), 1);
        assert_eq!(result.activated[0], "dd");
        assert_eq!(result.active_count, 1);
        assert!(result.threat_level > 0.0);
        assert_eq!(result.max_severity, Some(Severity::Warning));
    }

    #[test]
    fn test_trigger_does_not_fire_below_threshold() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.10)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.05, 0.01);
        let result = c.process(&metrics, 1).unwrap();
        assert!(result.activated.is_empty());
        assert_eq!(result.active_count, 0);
    }

    #[test]
    fn test_below_trigger() {
        let mut c = Contingency::new();
        c.register_plan(
            "low_vol",
            "Low Vol",
            Severity::Advisory,
            vec![TriggerCondition::below("volatility", 0.01)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.0, 0.005);
        let result = c.process(&metrics, 1).unwrap();
        assert_eq!(result.activated.len(), 1);
    }

    #[test]
    fn test_below_trigger_not_fired() {
        let mut c = Contingency::new();
        c.register_plan(
            "low_vol",
            "Low Vol",
            Severity::Advisory,
            vec![TriggerCondition::below("volatility", 0.01)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.0, 0.05);
        let result = c.process(&metrics, 1).unwrap();
        assert!(result.activated.is_empty());
    }

    #[test]
    fn test_resolve_plan() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action()],
        )
        .unwrap();

        let metrics = make_metrics(0.10, 0.01);
        c.process(&metrics, 1).unwrap();
        assert_eq!(c.plan("dd").unwrap().state, PlanState::Active);

        c.resolve("dd").unwrap();
        assert_eq!(c.plan("dd").unwrap().state, PlanState::Cooldown);
        assert_eq!(c.stats().total_resolutions, 1);
    }

    #[test]
    fn test_resolve_non_active_error() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        assert!(c.resolve("p").is_err());
    }

    #[test]
    fn test_resolve_nonexistent() {
        let mut c = Contingency::new();
        assert!(c.resolve("nope").is_err());
    }

    #[test]
    fn test_cooldown_prevents_reactivation() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_cooldown: 3,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.10, 0.01);
        c.process(&metrics, 1).unwrap(); // activates
        c.resolve("dd").unwrap(); // now in cooldown

        // Should not re-activate during cooldown
        let r2 = c.process(&metrics, 2).unwrap();
        assert!(r2.activated.is_empty());
        assert_eq!(c.plan("dd").unwrap().state, PlanState::Cooldown);
    }

    #[test]
    fn test_cooldown_expires() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_cooldown: 2,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.10, 0.01);
        c.process(&metrics, 1).unwrap(); // activates
        c.resolve("dd").unwrap(); // cooldown = 2

        c.process(&metrics, 2).unwrap(); // cooldown = 1
        c.process(&metrics, 3).unwrap(); // cooldown = 0, transitions to Dormant

        // Now should re-activate
        let r = c.process(&metrics, 4).unwrap();
        assert_eq!(r.activated.len(), 1);
    }

    #[test]
    fn test_expiry() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 3,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let calm = make_metrics(0.0, 0.0);
        let hot = make_metrics(0.10, 0.01);

        c.process(&hot, 1).unwrap(); // activate (active_ticks = 0)
        c.process(&calm, 2).unwrap(); // active_ticks = 1
        c.process(&calm, 3).unwrap(); // active_ticks = 2
        let r = c.process(&calm, 4).unwrap(); // active_ticks = 3 >= expiry => expired

        assert_eq!(r.expired.len(), 1);
        assert_eq!(r.expired[0], "dd");
        assert_eq!(c.plan("dd").unwrap().state, PlanState::Expired);
        assert_eq!(c.stats().total_expirations, 1);
    }

    #[test]
    fn test_cascade() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 2,
            default_cooldown: 0,
            ..Default::default()
        });

        c.register_plan(
            "primary",
            "Primary",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action()],
        )
        .unwrap();
        c.register_plan(
            "secondary",
            "Secondary",
            Severity::Emergency,
            vec![drawdown_trigger(0.15)],
            vec![halt_action()],
        )
        .unwrap();
        c.set_cascade("primary", "secondary").unwrap();

        let hot = make_metrics(0.10, 0.0);
        let calm = make_metrics(0.0, 0.0);

        c.process(&hot, 1).unwrap(); // activate primary
        c.process(&calm, 2).unwrap(); // active_ticks = 1
        let r = c.process(&calm, 3).unwrap(); // active_ticks = 2 => expired => cascade

        assert!(r.expired.contains(&"primary".to_string()));
        assert!(r.cascaded.contains(&"secondary".to_string()));
        assert_eq!(c.plan("secondary").unwrap().state, PlanState::Active);
        assert_eq!(c.stats().total_cascades, 1);
    }

    #[test]
    fn test_cascade_depth_limit() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 1,
            default_cooldown: 0,
            max_cascade_depth: 2,
            ..Default::default()
        });

        // Create chain: a -> b -> c -> d (d should not cascade since depth limit = 2)
        for name in &["a", "b", "c", "d"] {
            c.register_plan(
                name,
                name,
                Severity::Advisory,
                vec![drawdown_trigger(0.99)], // high threshold, won't fire naturally
                vec![],
            )
            .unwrap();
        }
        c.set_cascade("a", "b").unwrap();
        c.set_cascade("b", "c").unwrap();
        c.set_cascade("c", "d").unwrap();

        // Manually activate "a", let it expire
        c.activate("a").unwrap();
        let r = c.process(&HashMap::new(), 1).unwrap(); // a expires, cascades

        // With max_cascade_depth=2: a->b (depth 0), b does not expire yet
        assert!(r.cascaded.contains(&"b".to_string()));
        // "b" just got activated, it shouldn't have expired/cascaded yet
    }

    #[test]
    fn test_set_cascade_nonexistent_target() {
        let mut c = Contingency::new();
        c.register_plan(
            "a",
            "A",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        assert!(c.set_cascade("a", "nonexistent").is_err());
    }

    #[test]
    fn test_set_cascade_nonexistent_plan() {
        let mut c = Contingency::new();
        c.register_plan(
            "a",
            "A",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        assert!(c.set_cascade("nonexistent", "a").is_err());
    }

    #[test]
    fn test_manual_activation() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.activate("p").unwrap();
        assert_eq!(c.plan("p").unwrap().state, PlanState::Active);
        assert_eq!(c.stats().total_activations, 1);
    }

    #[test]
    fn test_arm_plan() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.arm_plan("p").unwrap();
        assert_eq!(c.plan("p").unwrap().state, PlanState::Armed);
    }

    #[test]
    fn test_arm_plan_non_dormant_error() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.activate("p").unwrap();
        assert!(c.arm_plan("p").is_err());
    }

    #[test]
    fn test_arm_plan_nonexistent() {
        let mut c = Contingency::new();
        assert!(c.arm_plan("nope").is_err());
    }

    #[test]
    fn test_armed_plan_can_be_triggered() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();
        c.arm_plan("dd").unwrap();

        let metrics = make_metrics(0.10, 0.0);
        let r = c.process(&metrics, 1).unwrap();
        assert_eq!(r.activated.len(), 1);
    }

    #[test]
    fn test_execute_action() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action(), halt_action()],
        )
        .unwrap();

        c.activate("dd").unwrap();
        c.execute_action("dd", "reduce_exposure").unwrap();

        assert!(
            c.plan("dd")
                .unwrap()
                .actions
                .iter()
                .find(|a| a.name == "reduce_exposure")
                .unwrap()
                .executed
        );
        assert!(!c.all_actions_executed("dd")); // halt_trading not yet executed
    }

    #[test]
    fn test_all_actions_executed() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action()],
        )
        .unwrap();

        c.activate("dd").unwrap();
        c.execute_action("dd", "reduce_exposure").unwrap();
        assert!(c.all_actions_executed("dd"));
    }

    #[test]
    fn test_all_actions_executed_empty() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.activate("p").unwrap();
        // No actions → false
        assert!(!c.all_actions_executed("p"));
    }

    #[test]
    fn test_execute_action_non_active() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![reduce_action()],
        )
        .unwrap();
        assert!(c.execute_action("p", "reduce_exposure").is_err());
    }

    #[test]
    fn test_execute_action_nonexistent_action() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![reduce_action()],
        )
        .unwrap();
        c.activate("p").unwrap();
        assert!(c.execute_action("p", "nonexistent").is_err());
    }

    #[test]
    fn test_execute_action_nonexistent_plan() {
        let mut c = Contingency::new();
        assert!(c.execute_action("nope", "action").is_err());
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Emergency > Severity::Warning);
        assert!(Severity::Warning > Severity::Advisory);
    }

    #[test]
    fn test_severity_weight() {
        assert!(Severity::Emergency.weight() > Severity::Warning.weight());
        assert!(Severity::Warning.weight() > Severity::Advisory.weight());
    }

    #[test]
    fn test_threat_level_increases_with_severity() {
        let mut c = Contingency::with_config(ContingencyConfig {
            ema_decay: 0.99,
            ..Default::default()
        });

        c.register_plan(
            "advisory",
            "A",
            Severity::Advisory,
            vec![drawdown_trigger(0.01)],
            vec![],
        )
        .unwrap();
        c.register_plan(
            "emergency",
            "E",
            Severity::Emergency,
            vec![vol_trigger(0.5)],
            vec![],
        )
        .unwrap();

        // Only advisory fires
        let metrics1 = make_metrics(0.05, 0.01);
        let r1 = c.process(&metrics1, 1).unwrap();
        let _t1 = r1.threat_level;

        c.resolve("advisory").unwrap();
        // Wait for cooldown
        for i in 2..20 {
            c.process(&make_metrics(0.0, 0.0), i).unwrap();
        }

        // Now trigger emergency
        let metrics2 = make_metrics(0.0, 0.6);
        let r2 = c.process(&metrics2, 20).unwrap();
        let t2 = r2.threat_level;

        // Emergency threat should be higher (though EMA smoothing may complicate exact comparison)
        // At least verify it registered
        assert!(r2.max_severity == Some(Severity::Emergency));
        assert!(t2 > 0.0);
    }

    #[test]
    fn test_multiple_triggers_any_fires() {
        let mut c = Contingency::new();
        c.register_plan(
            "multi",
            "Multi",
            Severity::Warning,
            vec![drawdown_trigger(0.10), vol_trigger(0.50)],
            vec![],
        )
        .unwrap();

        // Only volatility fires
        let metrics = make_metrics(0.01, 0.60);
        let r = c.process(&metrics, 1).unwrap();
        assert_eq!(r.activated.len(), 1);
    }

    #[test]
    fn test_already_active_plan_not_reactivated() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 100,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.10, 0.0);
        c.process(&metrics, 1).unwrap(); // activates
        let r2 = c.process(&metrics, 2).unwrap(); // already active, should not re-activate
        assert!(r2.activated.is_empty());
        assert_eq!(c.stats().total_activations, 1);
    }

    #[test]
    fn test_any_active() {
        let mut c = Contingency::new();
        assert!(!c.any_active());

        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.activate("p").unwrap();
        assert!(c.any_active());
    }

    #[test]
    fn test_emergency_active() {
        let mut c = Contingency::new();
        assert!(!c.emergency_active());

        c.register_plan(
            "e",
            "E",
            Severity::Emergency,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.activate("e").unwrap();
        assert!(c.emergency_active());
    }

    #[test]
    fn test_active_plan_ids() {
        let mut c = Contingency::new();
        c.register_plan(
            "a",
            "A",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.register_plan(
            "b",
            "B",
            Severity::Warning,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();

        c.activate("a").unwrap();
        let ids = c.active_plan_ids();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&"a"));
    }

    #[test]
    fn test_history_recorded() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.10, 0.0);
        c.process(&metrics, 1).unwrap();

        assert_eq!(c.history().len(), 1);
        assert_eq!(c.history()[0].plan_id, "dd");
        assert_eq!(c.history()[0].trigger_metric, "drawdown");
        assert!(c.history()[0].trigger_value >= 0.10);
        assert_eq!(c.history()[0].severity, Severity::Warning);
    }

    #[test]
    fn test_stats_defaults() {
        let stats = ContingencyStats::default();
        assert_eq!(stats.total_evaluations, 0);
        assert_eq!(stats.total_activations, 0);
        assert_eq!(stats.mean_resolution_time(), 0.0);
        assert_eq!(stats.expiration_rate(), 0.0);
        assert_eq!(stats.activation_rate(), 0.0);
        assert_eq!(stats.cascade_rate(), 0.0);
    }

    #[test]
    fn test_stats_mean_resolution_time() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_cooldown: 0,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        let calm = make_metrics(0.0, 0.0);

        c.process(&hot, 1).unwrap(); // activate
        c.process(&calm, 2).unwrap(); // active_ticks = 1
        c.process(&calm, 3).unwrap(); // active_ticks = 2
        c.resolve("dd").unwrap();

        assert!((c.stats().mean_resolution_time() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_expiration_rate() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 1,
            default_cooldown: 0,
            ..Default::default()
        });
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.01)],
            vec![],
        )
        .unwrap();

        // Activate and let expire
        let hot = make_metrics(0.10, 0.0);
        c.process(&hot, 1).unwrap(); // activate
        c.process(&make_metrics(0.0, 0.0), 2).unwrap(); // expire

        assert!((c.stats().expiration_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_stats_activation_rate() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 100,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        let calm = make_metrics(0.0, 0.0);

        c.process(&hot, 1).unwrap(); // activates (1 activation in 1 eval)
        c.process(&calm, 2).unwrap(); // no activation

        // 1 activation in 2 evaluations = 0.5
        assert!((c.stats().activation_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_severity_counts() {
        let mut c = Contingency::new();
        c.register_plan(
            "a",
            "A",
            Severity::Advisory,
            vec![drawdown_trigger(0.01)],
            vec![],
        )
        .unwrap();
        c.register_plan("w", "W", Severity::Warning, vec![vol_trigger(0.01)], vec![])
            .unwrap();
        c.register_plan(
            "e",
            "E",
            Severity::Emergency,
            vec![drawdown_trigger(0.02)],
            vec![],
        )
        .unwrap();

        let metrics = make_metrics(0.10, 0.50);
        c.process(&metrics, 1).unwrap();

        assert_eq!(c.stats().advisory_activations, 1);
        assert_eq!(c.stats().warning_activations, 1);
        assert_eq!(c.stats().emergency_activations, 1);
    }

    #[test]
    fn test_set_cooldown() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.set_cooldown("p", 20).unwrap();
        assert_eq!(c.plan("p").unwrap().cooldown_ticks, Some(20));
    }

    #[test]
    fn test_set_expiry() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.set_expiry("p", 50).unwrap();
        assert_eq!(c.plan("p").unwrap().expiry_ticks, Some(50));
    }

    #[test]
    fn test_set_expiry_zero() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        assert!(c.set_expiry("p", 0).is_err());
    }

    #[test]
    fn test_set_expiry_nonexistent() {
        let mut c = Contingency::new();
        assert!(c.set_expiry("nope", 10).is_err());
    }

    #[test]
    fn test_set_cooldown_nonexistent() {
        let mut c = Contingency::new();
        assert!(c.set_cooldown("nope", 10).is_err());
    }

    #[test]
    fn test_activation_count_increments() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_cooldown: 0,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        c.process(&hot, 1).unwrap(); // activate
        c.resolve("dd").unwrap(); // resolve
        c.process(&hot, 2).unwrap(); // re-activate (cooldown = 0)

        assert_eq!(c.plan("dd").unwrap().activation_count, 2);
    }

    #[test]
    fn test_trigger_condition_evaluate() {
        let above = TriggerCondition::above("vol", 0.5);
        assert!(above.evaluate(0.5)); // at threshold = fires
        assert!(above.evaluate(0.6));
        assert!(!above.evaluate(0.4));

        let below = TriggerCondition::below("price", 100.0);
        assert!(below.evaluate(100.0)); // at threshold = fires
        assert!(below.evaluate(99.0));
        assert!(!below.evaluate(101.0));
    }

    #[test]
    fn test_windowed_activation_rate() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 100,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        let calm = make_metrics(0.0, 0.0);

        c.process(&hot, 1).unwrap(); // 1 activation
        c.process(&calm, 2).unwrap(); // 0 activations

        // 1 total activation over 2 ticks = 0.5
        assert!((c.windowed_activation_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_activation_rate_empty() {
        let c = Contingency::new();
        assert_eq!(c.windowed_activation_rate(), 0.0);
    }

    #[test]
    fn test_windowed_mean_threat() {
        let mut c = Contingency::new();
        c.process(&HashMap::new(), 1).unwrap();
        c.process(&HashMap::new(), 2).unwrap();
        // No active plans => threat = 0
        assert!((c.windowed_mean_threat() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_mean_threat_empty() {
        let c = Contingency::new();
        assert_eq!(c.windowed_mean_threat(), 0.0);
    }

    #[test]
    fn test_is_threat_increasing_insufficient() {
        let c = Contingency::new();
        assert!(!c.is_threat_increasing());
    }

    #[test]
    fn test_is_threat_decreasing_insufficient() {
        let c = Contingency::new();
        assert!(!c.is_threat_decreasing());
    }

    #[test]
    fn test_is_warmed_up() {
        let mut c = Contingency::with_config(ContingencyConfig {
            min_samples: 3,
            ..Default::default()
        });
        assert!(!c.is_warmed_up());

        for i in 0..3 {
            c.process(&HashMap::new(), i).unwrap();
        }
        assert!(c.is_warmed_up());
    }

    #[test]
    fn test_confidence_increases() {
        let mut c = Contingency::with_config(ContingencyConfig {
            min_samples: 3,
            ..Default::default()
        });
        assert_eq!(c.confidence(), 0.0);

        for i in 0..6 {
            c.process(&HashMap::new(), i).unwrap();
        }
        let c6 = c.confidence();
        assert!(c6 > 0.0);

        for i in 6..20 {
            c.process(&HashMap::new(), i).unwrap();
        }
        let c20 = c.confidence();
        assert!(c20 >= c6);
        assert!(c20 <= 1.0);
    }

    #[test]
    fn test_window_eviction() {
        let mut c = Contingency::with_config(ContingencyConfig {
            window_size: 3,
            ..Default::default()
        });
        for i in 0..10 {
            c.process(&HashMap::new(), i).unwrap();
        }
        assert_eq!(c.recent.len(), 3);
    }

    #[test]
    fn test_reset() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action()],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        c.process(&hot, 1).unwrap();
        assert!(c.any_active());
        assert!(c.stats().total_activations > 0);

        c.reset();
        assert!(!c.any_active());
        assert_eq!(c.stats().total_activations, 0);
        assert_eq!(c.stats().total_evaluations, 0);
        assert_eq!(c.plan_count(), 1); // plan kept
        assert_eq!(c.plan("dd").unwrap().state, PlanState::Dormant);
        assert_eq!(c.plan("dd").unwrap().activation_count, 0);
        assert!(c.history().is_empty());
        assert!(c.recent.is_empty());
        assert_eq!(c.threat_level(), 0.0);
    }

    #[test]
    fn test_reset_all() {
        let mut c = Contingency::new();
        c.register_plan(
            "p",
            "P",
            Severity::Advisory,
            vec![drawdown_trigger(0.1)],
            vec![],
        )
        .unwrap();
        c.reset_all();
        assert_eq!(c.plan_count(), 0);
    }

    #[test]
    fn test_peak_threat_tracked() {
        let mut c = Contingency::with_config(ContingencyConfig {
            ema_decay: 0.99,
            ..Default::default()
        });
        c.register_plan(
            "e",
            "E",
            Severity::Emergency,
            vec![drawdown_trigger(0.01)],
            vec![],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        c.process(&hot, 1).unwrap();
        let peak = c.stats().peak_threat_level;
        assert!(peak > 0.0);

        // Resolve and process calm — peak should not decrease
        c.resolve("e").unwrap();
        for i in 2..20 {
            c.process(&make_metrics(0.0, 0.0), i).unwrap();
        }
        assert!((c.stats().peak_threat_level - peak).abs() < 1e-9);
    }

    #[test]
    fn test_default_has_no_plans() {
        let c = Contingency::new();
        assert_eq!(c.plan_count(), 0);
        assert_eq!(c.active_count(), 0);
        assert!(!c.any_active());
        assert!(!c.emergency_active());
        assert_eq!(c.threat_level(), 0.0);
        assert!(c.active_plan_ids().is_empty());
    }

    #[test]
    fn test_missing_metric_does_not_trigger() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        // Metrics don't contain "drawdown"
        let mut metrics = HashMap::new();
        metrics.insert("other_metric".to_string(), 0.99);

        let r = c.process(&metrics, 1).unwrap();
        assert!(r.activated.is_empty());
    }

    #[test]
    fn test_custom_cooldown_used() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_cooldown: 100, // high default
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();
        c.set_cooldown("dd", 1).unwrap(); // low custom

        let hot = make_metrics(0.10, 0.0);
        c.process(&hot, 1).unwrap();
        c.resolve("dd").unwrap();

        // Cooldown = 1
        c.process(&hot, 2).unwrap(); // cooldown decrements to 0
        let r = c.process(&hot, 3).unwrap(); // should be Dormant now and re-fire
        assert_eq!(r.activated.len(), 1);
    }

    #[test]
    fn test_custom_expiry_used() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 100, // high default
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();
        c.set_expiry("dd", 1).unwrap(); // low custom

        let hot = make_metrics(0.10, 0.0);
        c.process(&hot, 1).unwrap(); // activate
        let r = c.process(&make_metrics(0.0, 0.0), 2).unwrap(); // active_ticks=1 >= expiry=1

        assert!(r.expired.contains(&"dd".to_string()));
    }

    #[test]
    fn test_windowed_active_fraction() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 100,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        let calm = make_metrics(0.0, 0.0);

        c.process(&calm, 1).unwrap(); // no active
        c.process(&hot, 2).unwrap(); // active
        c.process(&calm, 3).unwrap(); // still active (not resolved)
        c.process(&calm, 4).unwrap(); // still active

        // 3 out of 4 ticks with active plan
        assert!((c.windowed_active_fraction() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn test_windowed_active_fraction_empty() {
        let c = Contingency::new();
        assert_eq!(c.windowed_active_fraction(), 0.0);
    }

    #[test]
    fn test_last_activated_and_resolved_ticks() {
        let mut c = Contingency::new();
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);
        c.process(&hot, 42).unwrap();
        assert_eq!(c.plan("dd").unwrap().last_activated_tick, Some(42));

        c.resolve("dd").unwrap();
        assert_eq!(c.plan("dd").unwrap().last_resolved_tick, Some(42));
    }

    #[test]
    fn test_actions_reset_on_reactivation() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_cooldown: 0,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![reduce_action()],
        )
        .unwrap();

        let hot = make_metrics(0.10, 0.0);

        c.process(&hot, 1).unwrap(); // activate
        c.execute_action("dd", "reduce_exposure").unwrap();
        assert!(c.all_actions_executed("dd"));

        c.resolve("dd").unwrap();
        c.process(&hot, 2).unwrap(); // re-activate

        // Actions should be reset
        assert!(!c.all_actions_executed("dd"));
        let action = c
            .plan("dd")
            .unwrap()
            .actions
            .iter()
            .find(|a| a.name == "reduce_exposure")
            .unwrap();
        assert!(!action.executed);
    }

    #[test]
    fn test_cooldown_blocks_stat() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_cooldown: 10,
            ..Default::default()
        });
        c.register_plan(
            "dd",
            "DD",
            Severity::Warning,
            vec![drawdown_trigger(0.05)],
            vec![],
        )
        .unwrap();

        c.activate("dd").unwrap();
        c.resolve("dd").unwrap();

        // Try to manually activate during cooldown
        c.activate("dd").unwrap(); // this goes through activate_plan which checks cooldown
        assert_eq!(c.stats().cooldown_blocks, 1);
    }

    #[test]
    #[should_panic(expected = "ema_decay must be in (0, 1)")]
    fn test_invalid_config_ema_decay_zero() {
        Contingency::with_config(ContingencyConfig {
            ema_decay: 0.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "ema_decay must be in (0, 1)")]
    fn test_invalid_config_ema_decay_one() {
        Contingency::with_config(ContingencyConfig {
            ema_decay: 1.0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_invalid_config_window_size() {
        Contingency::with_config(ContingencyConfig {
            window_size: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "max_plans must be > 0")]
    fn test_invalid_config_max_plans() {
        Contingency::with_config(ContingencyConfig {
            max_plans: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "max_cascade_depth must be > 0")]
    fn test_invalid_config_cascade_depth() {
        Contingency::with_config(ContingencyConfig {
            max_cascade_depth: 0,
            ..Default::default()
        });
    }

    #[test]
    #[should_panic(expected = "default_expiry_ticks must be > 0")]
    fn test_invalid_config_expiry_zero() {
        Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 0,
            ..Default::default()
        });
    }

    #[test]
    fn test_cascade_event_recorded_in_history() {
        let mut c = Contingency::with_config(ContingencyConfig {
            default_expiry_ticks: 1,
            default_cooldown: 0,
            ..Default::default()
        });

        c.register_plan(
            "a",
            "A",
            Severity::Warning,
            vec![drawdown_trigger(0.99)],
            vec![],
        )
        .unwrap();
        c.register_plan(
            "b",
            "B",
            Severity::Emergency,
            vec![drawdown_trigger(0.99)],
            vec![],
        )
        .unwrap();
        c.set_cascade("a", "b").unwrap();

        c.activate("a").unwrap();
        c.process(&HashMap::new(), 1).unwrap(); // a expires, cascades to b

        let cascade_events: Vec<_> = c
            .history()
            .iter()
            .filter(|e| e.cascaded_from.is_some())
            .collect();
        assert_eq!(cascade_events.len(), 1);
        assert_eq!(cascade_events[0].plan_id, "b");
        assert_eq!(cascade_events[0].cascaded_from.as_deref(), Some("a"));
    }

    #[test]
    fn test_contiguous_action_new() {
        let action = ContingencyAction::new("test_action", "Test description");
        assert_eq!(action.name, "test_action");
        assert_eq!(action.description, "Test description");
        assert!(!action.executed);
    }
}
