//! Low-level execution tactics
//!
//! Part of the Hippocampus region
//! Component: worker
//!
//! This module implements tactical policies for order execution, including
//! timing strategies, order type selection, and adaptive execution algorithms.

use crate::common::Result;
use std::collections::HashMap;

/// Execution tactic type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TacticType {
    /// Aggressive - prioritize fill speed
    Aggressive,
    /// Passive - prioritize price improvement
    Passive,
    /// Balanced - balance speed and price
    Balanced,
    /// TWAP - Time-weighted average price
    TWAP,
    /// VWAP - Volume-weighted average price
    VWAP,
    /// POV - Percentage of volume
    POV,
    /// Implementation Shortfall
    IS,
    /// Iceberg - hidden quantity
    Iceberg,
    /// Sniper - wait for favorable conditions
    Sniper,
    /// Adaptive - dynamically adjust
    Adaptive,
}

impl Default for TacticType {
    fn default() -> Self {
        TacticType::Balanced
    }
}

impl TacticType {
    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            TacticType::Aggressive => "Prioritize speed, cross spread if needed",
            TacticType::Passive => "Prioritize price, post limit orders",
            TacticType::Balanced => "Balance fill probability and price",
            TacticType::TWAP => "Spread order evenly over time",
            TacticType::VWAP => "Track volume-weighted average price",
            TacticType::POV => "Participate as percentage of volume",
            TacticType::IS => "Minimize implementation shortfall",
            TacticType::Iceberg => "Hide order size, show partial",
            TacticType::Sniper => "Wait for favorable liquidity",
            TacticType::Adaptive => "Dynamically adjust tactics",
        }
    }

    /// Get urgency level (0-1)
    pub fn urgency(&self) -> f64 {
        match self {
            TacticType::Aggressive => 1.0,
            TacticType::Sniper => 0.2,
            TacticType::Passive => 0.3,
            TacticType::Iceberg => 0.4,
            TacticType::TWAP => 0.5,
            TacticType::VWAP => 0.5,
            TacticType::POV => 0.5,
            TacticType::Balanced => 0.6,
            TacticType::IS => 0.7,
            TacticType::Adaptive => 0.5,
        }
    }

    /// Get market impact expectation (0-1)
    pub fn expected_impact(&self) -> f64 {
        match self {
            TacticType::Aggressive => 0.9,
            TacticType::Passive => 0.2,
            TacticType::Balanced => 0.5,
            TacticType::TWAP => 0.4,
            TacticType::VWAP => 0.3,
            TacticType::POV => 0.4,
            TacticType::IS => 0.5,
            TacticType::Iceberg => 0.3,
            TacticType::Sniper => 0.1,
            TacticType::Adaptive => 0.4,
        }
    }
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

/// Order type recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    /// Market order - immediate fill
    Market,
    /// Limit order at specific price
    Limit,
    /// Limit order that cancels if not filled immediately
    IOC,
    /// Fill or kill - all or nothing immediate
    FOK,
    /// Post only - maker only, cancel if would take
    PostOnly,
    /// Pegged to mid/bid/ask
    Pegged,
    /// Stop market
    StopMarket,
    /// Stop limit
    StopLimit,
}

impl Default for OrderType {
    fn default() -> Self {
        OrderType::Limit
    }
}

/// Market conditions for tactic selection
#[derive(Debug, Clone)]
pub struct MarketConditions {
    /// Current bid price
    pub bid: f64,
    /// Current ask price
    pub ask: f64,
    /// Spread as percentage
    pub spread_pct: f64,
    /// Current volatility (annualized)
    pub volatility: f64,
    /// Volume relative to average
    pub relative_volume: f64,
    /// Order book imbalance (-1 to 1)
    pub book_imbalance: f64,
    /// Recent price momentum
    pub momentum: f64,
    /// Liquidity score (0-1)
    pub liquidity_score: f64,
    /// Time of day factor (market open/close effects)
    pub time_factor: f64,
    /// Market regime indicator
    pub regime: String,
}

impl Default for MarketConditions {
    fn default() -> Self {
        Self {
            bid: 100.0,
            ask: 100.01,
            spread_pct: 0.01,
            volatility: 0.20,
            relative_volume: 1.0,
            book_imbalance: 0.0,
            momentum: 0.0,
            liquidity_score: 0.5,
            time_factor: 1.0,
            regime: "normal".to_string(),
        }
    }
}

impl MarketConditions {
    /// Get mid price
    pub fn mid_price(&self) -> f64 {
        (self.bid + self.ask) / 2.0
    }

    /// Get spread in dollars
    pub fn spread(&self) -> f64 {
        self.ask - self.bid
    }

    /// Check if conditions favor aggressive execution
    pub fn favors_aggressive(&self) -> bool {
        self.momentum.abs() > 0.5 || self.volatility > 0.3 || self.book_imbalance.abs() > 0.5
    }

    /// Check if conditions favor passive execution
    pub fn favors_passive(&self) -> bool {
        self.spread_pct > 0.1 && self.liquidity_score > 0.5 && self.momentum.abs() < 0.2
    }
}

/// Order execution parameters
#[derive(Debug, Clone)]
pub struct ExecutionParams {
    /// Target quantity
    pub quantity: f64,
    /// Order side
    pub side: OrderSide,
    /// Maximum allowed slippage (percentage)
    pub max_slippage: f64,
    /// Time horizon for execution (seconds)
    pub time_horizon: u64,
    /// Urgency level (0-1)
    pub urgency: f64,
    /// Risk aversion (0-1)
    pub risk_aversion: f64,
    /// Allow crossing spread
    pub allow_crossing: bool,
    /// Maximum participation rate
    pub max_participation: f64,
    /// Limit price (optional)
    pub limit_price: Option<f64>,
    /// Stop price (optional)
    pub stop_price: Option<f64>,
}

impl Default for ExecutionParams {
    fn default() -> Self {
        Self {
            quantity: 100.0,
            side: OrderSide::Buy,
            max_slippage: 0.005,
            time_horizon: 300,
            urgency: 0.5,
            risk_aversion: 0.5,
            allow_crossing: true,
            max_participation: 0.1,
            limit_price: None,
            stop_price: None,
        }
    }
}

/// Tactical decision/recommendation
#[derive(Debug, Clone)]
pub struct TacticalDecision {
    /// Selected tactic
    pub tactic: TacticType,
    /// Recommended order type
    pub order_type: OrderType,
    /// Recommended price (limit price)
    pub price: Option<f64>,
    /// Recommended quantity for this slice
    pub quantity: f64,
    /// Number of slices to divide order
    pub num_slices: usize,
    /// Time between slices (seconds)
    pub slice_interval: u64,
    /// Confidence in decision (0-1)
    pub confidence: f64,
    /// Expected fill probability
    pub fill_probability: f64,
    /// Expected market impact
    pub expected_impact: f64,
    /// Expected slippage
    pub expected_slippage: f64,
    /// Reasoning/notes
    pub reasoning: Vec<String>,
    /// Alternative tactics considered
    pub alternatives: Vec<TacticType>,
}

impl Default for TacticalDecision {
    fn default() -> Self {
        Self {
            tactic: TacticType::Balanced,
            order_type: OrderType::Limit,
            price: None,
            quantity: 0.0,
            num_slices: 1,
            slice_interval: 0,
            confidence: 0.5,
            fill_probability: 0.5,
            expected_impact: 0.0,
            expected_slippage: 0.0,
            reasoning: Vec::new(),
            alternatives: Vec::new(),
        }
    }
}

/// Execution slice for multi-part orders
#[derive(Debug, Clone)]
pub struct ExecutionSlice {
    /// Slice number
    pub slice_num: usize,
    /// Quantity for this slice
    pub quantity: f64,
    /// Target execution time
    pub target_time: u64,
    /// Price limit
    pub price_limit: Option<f64>,
    /// Order type for this slice
    pub order_type: OrderType,
    /// Status
    pub status: SliceStatus,
    /// Filled quantity
    pub filled_quantity: f64,
    /// Average fill price
    pub avg_fill_price: Option<f64>,
}

/// Status of an execution slice
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceStatus {
    Pending,
    Active,
    PartiallyFilled,
    Filled,
    Cancelled,
    Failed,
}

impl Default for SliceStatus {
    fn default() -> Self {
        SliceStatus::Pending
    }
}

/// Configuration for tactical policy
#[derive(Debug, Clone)]
pub struct TacticalPolicyConfig {
    /// Default tactic to use
    pub default_tactic: TacticType,
    /// Enable adaptive tactic selection
    pub enable_adaptive: bool,
    /// Spread threshold for passive tactics (percentage)
    pub passive_spread_threshold: f64,
    /// Volume threshold for aggressive tactics
    pub aggressive_volume_threshold: f64,
    /// Maximum order slices
    pub max_slices: usize,
    /// Minimum slice size
    pub min_slice_size: f64,
    /// Price improvement target (basis points)
    pub price_improvement_target: f64,
    /// Fill probability minimum
    pub min_fill_probability: f64,
    /// Enable iceberg detection
    pub detect_iceberg: bool,
    /// Learning rate for policy updates
    pub learning_rate: f64,
}

impl Default for TacticalPolicyConfig {
    fn default() -> Self {
        Self {
            default_tactic: TacticType::Balanced,
            enable_adaptive: true,
            passive_spread_threshold: 0.05,
            aggressive_volume_threshold: 1.5,
            max_slices: 10,
            min_slice_size: 1.0,
            price_improvement_target: 5.0,
            min_fill_probability: 0.8,
            detect_iceberg: true,
            learning_rate: 0.01,
        }
    }
}

/// Execution outcome for learning
#[derive(Debug, Clone)]
pub struct ExecutionOutcome {
    /// Tactic used
    pub tactic: TacticType,
    /// Order type used
    pub order_type: OrderType,
    /// Target quantity
    pub target_quantity: f64,
    /// Filled quantity
    pub filled_quantity: f64,
    /// Target price (decision price)
    pub target_price: f64,
    /// Average execution price
    pub avg_price: f64,
    /// Total execution time (seconds)
    pub execution_time: u64,
    /// Market conditions at decision
    pub conditions: MarketConditions,
    /// Slippage realized
    pub slippage: f64,
    /// Fill rate (filled / target)
    pub fill_rate: f64,
    /// Market impact observed
    pub market_impact: f64,
    /// Success flag
    pub success: bool,
}

impl ExecutionOutcome {
    /// Calculate implementation shortfall
    pub fn implementation_shortfall(&self) -> f64 {
        let direction = if self.avg_price > self.target_price {
            1.0
        } else {
            -1.0
        };
        direction * (self.avg_price - self.target_price).abs() / self.target_price
    }
}

/// Tactic performance statistics
#[derive(Debug, Clone, Default)]
pub struct TacticStats {
    /// Number of times used
    pub usage_count: usize,
    /// Number of successful executions
    pub success_count: usize,
    /// Average fill rate
    pub avg_fill_rate: f64,
    /// Average slippage
    pub avg_slippage: f64,
    /// Average market impact
    pub avg_impact: f64,
    /// Average implementation shortfall
    pub avg_shortfall: f64,
    /// Total quantity executed
    pub total_quantity: f64,
}

impl TacticStats {
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.usage_count == 0 {
            return 0.0;
        }
        self.success_count as f64 / self.usage_count as f64
    }

    /// Update statistics with new outcome
    pub fn update(&mut self, outcome: &ExecutionOutcome) {
        self.usage_count += 1;
        if outcome.success {
            self.success_count += 1;
        }

        // Rolling averages
        let n = self.usage_count as f64;
        self.avg_fill_rate = ((self.avg_fill_rate * (n - 1.0)) + outcome.fill_rate) / n;
        self.avg_slippage = ((self.avg_slippage * (n - 1.0)) + outcome.slippage) / n;
        self.avg_impact = ((self.avg_impact * (n - 1.0)) + outcome.market_impact) / n;
        self.avg_shortfall =
            ((self.avg_shortfall * (n - 1.0)) + outcome.implementation_shortfall()) / n;
        self.total_quantity += outcome.filled_quantity;
    }
}

/// Low-level execution tactics
pub struct TacticalPolicy {
    /// Configuration
    config: TacticalPolicyConfig,
    /// Tactic performance statistics
    tactic_stats: HashMap<TacticType, TacticStats>,
    /// Recent execution outcomes
    outcome_history: Vec<ExecutionOutcome>,
    /// Maximum history size
    max_history: usize,
    /// Total decisions made
    total_decisions: usize,
    /// Q-values for tactic selection (state-action values)
    q_values: HashMap<String, HashMap<TacticType, f64>>,
}

impl Default for TacticalPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl TacticalPolicy {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            config: TacticalPolicyConfig::default(),
            tactic_stats: HashMap::new(),
            outcome_history: Vec::new(),
            max_history: 1000,
            total_decisions: 0,
            q_values: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TacticalPolicyConfig) -> Self {
        Self {
            config,
            tactic_stats: HashMap::new(),
            outcome_history: Vec::new(),
            max_history: 1000,
            total_decisions: 0,
            q_values: HashMap::new(),
        }
    }

    /// Select optimal tactic for given conditions
    pub fn select_tactic(
        &mut self,
        conditions: &MarketConditions,
        params: &ExecutionParams,
    ) -> TacticalDecision {
        self.total_decisions += 1;

        let mut decision = TacticalDecision::default();
        decision.quantity = params.quantity;

        // Get state key for Q-learning
        let state_key = self.discretize_state(conditions, params);

        // Select tactic
        let tactic = if self.config.enable_adaptive {
            self.select_adaptive_tactic(&state_key, conditions, params)
        } else {
            self.select_rule_based_tactic(conditions, params)
        };

        decision.tactic = tactic;
        decision.confidence = self.calculate_confidence(tactic, conditions);

        // Determine order type and price
        self.configure_order(&mut decision, conditions, params);

        // Calculate execution plan
        self.plan_execution(&mut decision, conditions, params);

        // Add reasoning
        self.add_reasoning(&mut decision, conditions, params);

        // Record alternatives
        decision.alternatives = self.get_alternatives(tactic, conditions);

        decision
    }

    /// Discretize state for Q-learning
    fn discretize_state(&self, conditions: &MarketConditions, params: &ExecutionParams) -> String {
        let spread_bucket = if conditions.spread_pct < 0.01 {
            "tight"
        } else if conditions.spread_pct < 0.05 {
            "normal"
        } else {
            "wide"
        };

        let vol_bucket = if conditions.volatility < 0.15 {
            "low"
        } else if conditions.volatility < 0.30 {
            "med"
        } else {
            "high"
        };

        let urgency_bucket = if params.urgency < 0.3 {
            "low"
        } else if params.urgency < 0.7 {
            "med"
        } else {
            "high"
        };

        let size_bucket = if params.quantity < 100.0 {
            "small"
        } else if params.quantity < 1000.0 {
            "med"
        } else {
            "large"
        };

        format!(
            "{}_{}_{}_{}_{}",
            spread_bucket, vol_bucket, urgency_bucket, size_bucket, conditions.regime
        )
    }

    /// Select tactic using adaptive Q-learning approach
    fn select_adaptive_tactic(
        &self,
        state_key: &str,
        conditions: &MarketConditions,
        params: &ExecutionParams,
    ) -> TacticType {
        // Get Q-values for this state
        let state_q = self.q_values.get(state_key);

        // Epsilon-greedy selection
        let epsilon = 0.1;
        let mut rng = rand::rng();
        use rand::RngExt;

        if rng.random::<f64>() < epsilon || state_q.is_none() {
            // Explore or no data: use rule-based
            self.select_rule_based_tactic(conditions, params)
        } else {
            // Exploit: select best known tactic
            state_q
                .unwrap()
                .iter()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(t, _)| *t)
                .unwrap_or_else(|| self.select_rule_based_tactic(conditions, params))
        }
    }

    /// Select tactic using rule-based logic
    fn select_rule_based_tactic(
        &self,
        conditions: &MarketConditions,
        params: &ExecutionParams,
    ) -> TacticType {
        // High urgency -> aggressive
        if params.urgency > 0.8 {
            return TacticType::Aggressive;
        }

        // Very low urgency and wide spread -> passive
        if params.urgency < 0.3 && conditions.spread_pct > self.config.passive_spread_threshold {
            return TacticType::Passive;
        }

        // Large order relative to volume -> TWAP/VWAP
        if params.quantity > 1000.0 && params.time_horizon > 300 {
            if conditions.relative_volume > 1.0 {
                return TacticType::VWAP;
            } else {
                return TacticType::TWAP;
            }
        }

        // High volatility with momentum -> sniper or IS
        if conditions.volatility > 0.25 && conditions.momentum.abs() > 0.3 {
            if params.risk_aversion > 0.6 {
                return TacticType::IS;
            } else {
                return TacticType::Sniper;
            }
        }

        // Good liquidity and want to hide size -> iceberg
        if params.quantity > 500.0 && conditions.liquidity_score > 0.6 {
            return TacticType::Iceberg;
        }

        // High participation concern -> POV
        if params.max_participation < 0.05 {
            return TacticType::POV;
        }

        // Default based on conditions
        if conditions.favors_aggressive() {
            TacticType::Aggressive
        } else if conditions.favors_passive() {
            TacticType::Passive
        } else {
            TacticType::Balanced
        }
    }

    /// Configure order type and price
    fn configure_order(
        &self,
        decision: &mut TacticalDecision,
        conditions: &MarketConditions,
        params: &ExecutionParams,
    ) {
        let mid = conditions.mid_price();

        match decision.tactic {
            TacticType::Aggressive => {
                decision.order_type = if params.allow_crossing {
                    OrderType::Market
                } else {
                    OrderType::IOC
                };
                decision.price = match params.side {
                    OrderSide::Buy => Some(conditions.ask),
                    OrderSide::Sell => Some(conditions.bid),
                };
                decision.fill_probability = 0.95;
                decision.expected_impact = conditions.spread_pct * 0.5;
            }
            TacticType::Passive => {
                decision.order_type = OrderType::PostOnly;
                decision.price = match params.side {
                    OrderSide::Buy => Some(conditions.bid),
                    OrderSide::Sell => Some(conditions.ask),
                };
                decision.fill_probability = 0.4;
                decision.expected_impact = 0.0;
            }
            TacticType::Balanced => {
                decision.order_type = OrderType::Limit;
                let offset = conditions.spread() * 0.25;
                decision.price = match params.side {
                    OrderSide::Buy => Some(mid + offset),
                    OrderSide::Sell => Some(mid - offset),
                };
                decision.fill_probability = 0.7;
                decision.expected_impact = conditions.spread_pct * 0.25;
            }
            TacticType::Sniper => {
                decision.order_type = OrderType::Limit;
                // More aggressive limit price but wait for liquidity
                decision.price = match params.side {
                    OrderSide::Buy => Some(conditions.bid + conditions.spread() * 0.1),
                    OrderSide::Sell => Some(conditions.ask - conditions.spread() * 0.1),
                };
                decision.fill_probability = 0.5;
                decision.expected_impact = 0.0;
            }
            TacticType::Iceberg => {
                decision.order_type = OrderType::Limit;
                decision.price = Some(mid);
                decision.fill_probability = 0.75;
                decision.expected_impact = conditions.spread_pct * 0.1;
            }
            _ => {
                // TWAP, VWAP, POV, IS - use limit orders at mid
                decision.order_type = OrderType::Limit;
                decision.price = Some(mid);
                decision.fill_probability = 0.7;
                decision.expected_impact = conditions.spread_pct * 0.2;
            }
        }

        // Apply limit price constraint
        if let Some(limit) = params.limit_price {
            if let Some(ref mut price) = decision.price {
                match params.side {
                    OrderSide::Buy => *price = price.min(limit),
                    OrderSide::Sell => *price = price.max(limit),
                }
            }
        }

        // Calculate expected slippage
        decision.expected_slippage =
            decision.expected_impact + (1.0 - decision.fill_probability) * conditions.spread_pct;
    }

    /// Plan execution (slicing)
    fn plan_execution(
        &self,
        decision: &mut TacticalDecision,
        _conditions: &MarketConditions,
        params: &ExecutionParams,
    ) {
        match decision.tactic {
            TacticType::TWAP => {
                // Divide evenly over time
                let num_slices = (params.time_horizon / 60)
                    .max(1)
                    .min(self.config.max_slices as u64);
                decision.num_slices = num_slices as usize;
                decision.slice_interval = params.time_horizon / num_slices;
                decision.quantity = params.quantity / num_slices as f64;
            }
            TacticType::VWAP => {
                // Adjust slices based on expected volume profile
                let num_slices = (params.time_horizon / 60)
                    .max(1)
                    .min(self.config.max_slices as u64);
                decision.num_slices = num_slices as usize;
                decision.slice_interval = params.time_horizon / num_slices;
                // In practice, slice sizes would vary with volume profile
                decision.quantity = params.quantity / num_slices as f64;
            }
            TacticType::POV => {
                // Slices determined by volume participation
                decision.num_slices = self.config.max_slices;
                decision.slice_interval = 60; // Check every minute
                decision.quantity =
                    params.quantity * params.max_participation / self.config.max_slices as f64;
            }
            TacticType::Iceberg => {
                // Show small portion, replenish on fill
                let show_size = (params.quantity * 0.1).max(self.config.min_slice_size);
                decision.num_slices = (params.quantity / show_size).ceil() as usize;
                decision.slice_interval = 0; // Replenish immediately
                decision.quantity = show_size;
            }
            _ => {
                // Single order or minimal slicing
                if params.quantity > 1000.0 && decision.tactic != TacticType::Aggressive {
                    decision.num_slices = 2;
                    decision.slice_interval = params.time_horizon / 2;
                    decision.quantity = params.quantity / 2.0;
                } else {
                    decision.num_slices = 1;
                    decision.slice_interval = 0;
                }
            }
        }
    }

    /// Add reasoning to decision
    fn add_reasoning(
        &self,
        decision: &mut TacticalDecision,
        conditions: &MarketConditions,
        params: &ExecutionParams,
    ) {
        decision
            .reasoning
            .push(format!("Selected tactic: {:?}", decision.tactic));
        decision.reasoning.push(format!(
            "Spread: {:.2}% ({})",
            conditions.spread_pct * 100.0,
            if conditions.spread_pct < 0.02 {
                "tight"
            } else {
                "wide"
            }
        ));
        decision.reasoning.push(format!(
            "Volatility: {:.1}% ({})",
            conditions.volatility * 100.0,
            if conditions.volatility < 0.20 {
                "low"
            } else {
                "elevated"
            }
        ));
        decision
            .reasoning
            .push(format!("Urgency: {:.0}%", params.urgency * 100.0));
        decision.reasoning.push(format!(
            "Expected fill probability: {:.0}%",
            decision.fill_probability * 100.0
        ));
        decision.reasoning.push(format!(
            "Expected slippage: {:.2}%",
            decision.expected_slippage * 100.0
        ));
    }

    /// Get alternative tactics
    fn get_alternatives(
        &self,
        selected: TacticType,
        conditions: &MarketConditions,
    ) -> Vec<TacticType> {
        let mut alternatives = Vec::new();

        match selected {
            TacticType::Aggressive => {
                alternatives.push(TacticType::Balanced);
                alternatives.push(TacticType::IS);
            }
            TacticType::Passive => {
                alternatives.push(TacticType::Balanced);
                alternatives.push(TacticType::Sniper);
            }
            TacticType::Balanced => {
                if conditions.favors_aggressive() {
                    alternatives.push(TacticType::Aggressive);
                } else {
                    alternatives.push(TacticType::Passive);
                }
                alternatives.push(TacticType::TWAP);
            }
            _ => {
                alternatives.push(TacticType::Balanced);
            }
        }

        alternatives
    }

    /// Calculate confidence in tactic selection
    fn calculate_confidence(&self, tactic: TacticType, conditions: &MarketConditions) -> f64 {
        let mut confidence = 0.5;

        // Boost confidence if we have good historical data
        if let Some(stats) = self.tactic_stats.get(&tactic) {
            if stats.usage_count > 10 {
                confidence += stats.success_rate() * 0.3;
            }
        }

        // Adjust based on market clarity
        if conditions.book_imbalance.abs() > 0.3 {
            confidence += 0.1; // Clear direction
        }
        if conditions.liquidity_score > 0.7 {
            confidence += 0.1; // Good liquidity
        }

        confidence.min(1.0)
    }

    /// Record execution outcome for learning
    pub fn record_outcome(&mut self, outcome: ExecutionOutcome) {
        // Update tactic statistics
        let stats = self.tactic_stats.entry(outcome.tactic).or_default();
        stats.update(&outcome);

        // Update Q-values
        let state_key = self.discretize_state(&outcome.conditions, &ExecutionParams::default());
        let reward = self.calculate_reward(&outcome);

        let state_q = self.q_values.entry(state_key).or_default();
        let current_q = state_q.get(&outcome.tactic).copied().unwrap_or(0.0);
        let new_q = current_q + self.config.learning_rate * (reward - current_q);
        state_q.insert(outcome.tactic, new_q);

        // Store in history
        self.outcome_history.push(outcome);
        if self.outcome_history.len() > self.max_history {
            self.outcome_history.remove(0);
        }
    }

    /// Calculate reward for Q-learning
    fn calculate_reward(&self, outcome: &ExecutionOutcome) -> f64 {
        let fill_component = outcome.fill_rate * 0.4;
        let slippage_component = (1.0 - outcome.slippage.abs().min(0.05) / 0.05) * 0.3;
        let impact_component = (1.0 - outcome.market_impact.min(0.05) / 0.05) * 0.2;
        let success_component = if outcome.success { 0.1 } else { 0.0 };

        fill_component + slippage_component + impact_component + success_component
    }

    /// Get statistics for a tactic
    pub fn get_tactic_stats(&self, tactic: TacticType) -> Option<&TacticStats> {
        self.tactic_stats.get(&tactic)
    }

    /// Get all tactic statistics
    pub fn all_tactic_stats(&self) -> &HashMap<TacticType, TacticStats> {
        &self.tactic_stats
    }

    /// Get total decisions made
    pub fn total_decisions(&self) -> usize {
        self.total_decisions
    }

    /// Get outcome history
    pub fn outcome_history(&self) -> &[ExecutionOutcome] {
        &self.outcome_history
    }

    /// Generate execution slices
    pub fn generate_slices(
        &self,
        decision: &TacticalDecision,
        params: &ExecutionParams,
        start_time: u64,
    ) -> Vec<ExecutionSlice> {
        let mut slices = Vec::with_capacity(decision.num_slices);

        for i in 0..decision.num_slices {
            slices.push(ExecutionSlice {
                slice_num: i + 1,
                quantity: if i == decision.num_slices - 1 {
                    // Last slice gets remainder
                    params.quantity - (decision.quantity * i as f64)
                } else {
                    decision.quantity
                },
                target_time: start_time + (decision.slice_interval * i as u64),
                price_limit: decision.price,
                order_type: decision.order_type,
                status: SliceStatus::Pending,
                filled_quantity: 0.0,
                avg_fill_price: None,
            });
        }

        slices
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via select_tactic
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_conditions() -> MarketConditions {
        MarketConditions {
            bid: 100.0,
            ask: 100.05,
            spread_pct: 0.0005,
            volatility: 0.20,
            relative_volume: 1.0,
            book_imbalance: 0.0,
            momentum: 0.0,
            liquidity_score: 0.5,
            time_factor: 1.0,
            regime: "normal".to_string(),
        }
    }

    fn create_test_params() -> ExecutionParams {
        ExecutionParams {
            quantity: 100.0,
            side: OrderSide::Buy,
            max_slippage: 0.005,
            time_horizon: 300,
            urgency: 0.5,
            risk_aversion: 0.5,
            allow_crossing: true,
            max_participation: 0.1,
            limit_price: None,
            stop_price: None,
        }
    }

    #[test]
    fn test_basic() {
        let instance = TacticalPolicy::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_select_tactic() {
        let mut policy = TacticalPolicy::new();
        let conditions = create_test_conditions();
        let params = create_test_params();

        let decision = policy.select_tactic(&conditions, &params);

        assert!(decision.confidence > 0.0);
        assert!(decision.quantity > 0.0);
        assert!(!decision.reasoning.is_empty());
    }

    #[test]
    fn test_aggressive_selection() {
        let mut policy = TacticalPolicy::new();
        let conditions = create_test_conditions();
        let mut params = create_test_params();
        params.urgency = 0.9; // High urgency

        let decision = policy.select_tactic(&conditions, &params);

        assert_eq!(decision.tactic, TacticType::Aggressive);
    }

    #[test]
    fn test_passive_selection() {
        let mut policy = TacticalPolicy::new();
        let mut conditions = create_test_conditions();
        conditions.spread_pct = 0.10; // Wide spread

        let mut params = create_test_params();
        params.urgency = 0.2; // Low urgency

        let decision = policy.select_tactic(&conditions, &params);

        assert_eq!(decision.tactic, TacticType::Passive);
    }

    #[test]
    fn test_twap_selection() {
        let mut policy = TacticalPolicy::new();
        let conditions = create_test_conditions();
        let mut params = create_test_params();
        params.quantity = 5000.0; // Large order
        params.time_horizon = 600;
        params.urgency = 0.4;

        let decision = policy.select_tactic(&conditions, &params);

        assert!(matches!(
            decision.tactic,
            TacticType::TWAP | TacticType::VWAP | TacticType::Iceberg
        ));
    }

    #[test]
    fn test_order_configuration() {
        let mut policy = TacticalPolicy::new();
        let conditions = create_test_conditions();
        let params = create_test_params();

        let decision = policy.select_tactic(&conditions, &params);

        assert!(decision.price.is_some());
        assert!(decision.fill_probability >= 0.0 && decision.fill_probability <= 1.0);
    }

    #[test]
    fn test_execution_slices() {
        let policy = TacticalPolicy::new();

        let mut decision = TacticalDecision::default();
        decision.tactic = TacticType::TWAP;
        decision.num_slices = 3;
        decision.slice_interval = 100;
        decision.quantity = 100.0;
        decision.price = Some(100.0);
        decision.order_type = OrderType::Limit;

        let params = ExecutionParams {
            quantity: 300.0,
            ..Default::default()
        };

        let slices = policy.generate_slices(&decision, &params, 1000);

        assert_eq!(slices.len(), 3);
        assert_eq!(slices[0].target_time, 1000);
        assert_eq!(slices[1].target_time, 1100);
        assert_eq!(slices[2].target_time, 1200);
    }

    #[test]
    fn test_record_outcome() {
        let mut policy = TacticalPolicy::new();

        let outcome = ExecutionOutcome {
            tactic: TacticType::Balanced,
            order_type: OrderType::Limit,
            target_quantity: 100.0,
            filled_quantity: 95.0,
            target_price: 100.0,
            avg_price: 100.02,
            execution_time: 60,
            conditions: create_test_conditions(),
            slippage: 0.0002,
            fill_rate: 0.95,
            market_impact: 0.001,
            success: true,
        };

        policy.record_outcome(outcome);

        let stats = policy.get_tactic_stats(TacticType::Balanced).unwrap();
        assert_eq!(stats.usage_count, 1);
        assert_eq!(stats.success_count, 1);
    }

    #[test]
    fn test_tactic_type_properties() {
        assert!(TacticType::Aggressive.urgency() > TacticType::Passive.urgency());
        assert!(TacticType::Aggressive.expected_impact() > TacticType::Sniper.expected_impact());
    }

    #[test]
    fn test_market_conditions() {
        let mut conditions = create_test_conditions();

        // Test favors_aggressive
        conditions.momentum = 0.7;
        assert!(conditions.favors_aggressive());

        // Test favors_passive
        conditions.momentum = 0.0;
        conditions.spread_pct = 0.15;
        conditions.liquidity_score = 0.7;
        assert!(conditions.favors_passive());
    }

    #[test]
    fn test_implementation_shortfall() {
        let outcome = ExecutionOutcome {
            tactic: TacticType::Balanced,
            order_type: OrderType::Limit,
            target_quantity: 100.0,
            filled_quantity: 100.0,
            target_price: 100.0,
            avg_price: 100.10,
            execution_time: 60,
            conditions: create_test_conditions(),
            slippage: 0.001,
            fill_rate: 1.0,
            market_impact: 0.001,
            success: true,
        };

        let is = outcome.implementation_shortfall();
        assert!((is - 0.001).abs() < 0.0001);
    }

    #[test]
    fn test_tactic_stats_update() {
        let mut stats = TacticStats::default();

        let outcome = ExecutionOutcome {
            tactic: TacticType::Balanced,
            order_type: OrderType::Limit,
            target_quantity: 100.0,
            filled_quantity: 90.0,
            target_price: 100.0,
            avg_price: 100.05,
            execution_time: 60,
            conditions: create_test_conditions(),
            slippage: 0.0005,
            fill_rate: 0.9,
            market_impact: 0.001,
            success: true,
        };

        stats.update(&outcome);

        assert_eq!(stats.usage_count, 1);
        assert_eq!(stats.success_rate(), 1.0);
        assert_eq!(stats.avg_fill_rate, 0.9);
    }

    #[test]
    fn test_limit_price_constraint() {
        let mut policy = TacticalPolicy::new();
        let conditions = create_test_conditions();
        let mut params = create_test_params();
        params.limit_price = Some(99.95);
        params.side = OrderSide::Buy;

        let decision = policy.select_tactic(&conditions, &params);

        if let Some(price) = decision.price {
            assert!(price <= 99.95);
        }
    }

    #[test]
    fn test_alternatives() {
        let mut policy = TacticalPolicy::new();
        let conditions = create_test_conditions();
        let params = create_test_params();

        let decision = policy.select_tactic(&conditions, &params);

        assert!(!decision.alternatives.is_empty());
        assert!(!decision.alternatives.contains(&decision.tactic));
    }
}
