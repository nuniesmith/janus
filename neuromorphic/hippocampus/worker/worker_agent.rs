//! Feudal RL worker implementation
//!
//! Part of the Hippocampus region
//! Component: worker
//!
//! This module implements a feudal reinforcement learning worker agent that
//! coordinates between the skill library and procedural memory to execute
//! trading actions. The worker receives goals from a manager and selects
//! appropriate skills and procedures to achieve them.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Goal type from manager
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GoalType {
    /// Enter a position
    EnterPosition,
    /// Exit a position
    ExitPosition,
    /// Adjust position size
    AdjustPosition,
    /// Manage risk
    ManageRisk,
    /// Optimize execution
    OptimizeExecution,
    /// Hedge exposure
    Hedge,
    /// Rebalance portfolio
    Rebalance,
    /// Emergency action
    Emergency,
    /// Wait/observe
    Observe,
}

impl Default for GoalType {
    fn default() -> Self {
        GoalType::Observe
    }
}

impl GoalType {
    /// Get priority level (higher = more urgent)
    pub fn priority(&self) -> u8 {
        match self {
            GoalType::Emergency => 10,
            GoalType::ManageRisk => 9,
            GoalType::ExitPosition => 7,
            GoalType::EnterPosition => 6,
            GoalType::AdjustPosition => 5,
            GoalType::Hedge => 4,
            GoalType::OptimizeExecution => 3,
            GoalType::Rebalance => 2,
            GoalType::Observe => 1,
        }
    }

    /// Get default timeout in milliseconds
    pub fn default_timeout_ms(&self) -> u64 {
        match self {
            GoalType::Emergency => 1000,
            GoalType::ManageRisk => 5000,
            GoalType::ExitPosition => 10000,
            GoalType::EnterPosition => 30000,
            GoalType::AdjustPosition => 15000,
            GoalType::Hedge => 20000,
            GoalType::OptimizeExecution => 60000,
            GoalType::Rebalance => 120000,
            GoalType::Observe => u64::MAX,
        }
    }
}

/// A goal assigned by the manager
#[derive(Debug, Clone)]
pub struct Goal {
    /// Unique goal ID
    pub id: String,
    /// Goal type
    pub goal_type: GoalType,
    /// Target symbol (if applicable)
    pub symbol: Option<String>,
    /// Target direction (-1 short, 0 neutral, 1 long)
    pub direction: i8,
    /// Target magnitude (e.g., position size)
    pub magnitude: f64,
    /// Urgency (0.0 to 1.0)
    pub urgency: f64,
    /// Time limit in milliseconds
    pub timeout_ms: u64,
    /// Additional parameters
    pub params: HashMap<String, f64>,
    /// Timestamp when goal was set
    pub set_at: u64,
    /// Reward signal from manager
    pub intrinsic_reward: f64,
}

impl Goal {
    /// Create a new goal
    pub fn new(goal_type: GoalType) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        Self {
            id: format!("goal_{}_{}", goal_type as u8, now),
            goal_type,
            symbol: None,
            direction: 0,
            magnitude: 0.0,
            urgency: 0.5,
            timeout_ms: goal_type.default_timeout_ms(),
            params: HashMap::new(),
            set_at: now,
            intrinsic_reward: 0.0,
        }
    }

    /// Set symbol
    pub fn with_symbol(mut self, symbol: &str) -> Self {
        self.symbol = Some(symbol.to_string());
        self
    }

    /// Set direction
    pub fn with_direction(mut self, direction: i8) -> Self {
        self.direction = direction;
        self
    }

    /// Set magnitude
    pub fn with_magnitude(mut self, magnitude: f64) -> Self {
        self.magnitude = magnitude;
        self
    }

    /// Set urgency
    pub fn with_urgency(mut self, urgency: f64) -> Self {
        self.urgency = urgency.clamp(0.0, 1.0);
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Set parameter
    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.params.insert(key.to_string(), value);
        self
    }

    /// Check if goal is expired
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        // Use >= so that timeout_ms=0 means immediately expired
        now.saturating_sub(self.set_at) >= self.timeout_ms
    }

    /// Get remaining time
    pub fn remaining_ms(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let elapsed = now.saturating_sub(self.set_at);
        self.timeout_ms.saturating_sub(elapsed)
    }
}

/// Current state of the worker
#[derive(Debug, Clone, Default)]
pub struct WorkerState {
    /// Current market price
    pub price: f64,
    /// Current bid
    pub bid: f64,
    /// Current ask
    pub ask: f64,
    /// Current volatility
    pub volatility: f64,
    /// Current spread percentage
    pub spread_pct: f64,
    /// Current position size
    pub position: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Realized PnL
    pub realized_pnl: f64,
    /// Open orders count
    pub open_orders: usize,
    /// Pending fills
    pub pending_fills: usize,
    /// Risk utilization (0.0 to 1.0)
    pub risk_utilization: f64,
    /// Market regime
    pub regime: MarketRegime,
    /// Timestamp
    pub timestamp: u64,
}

/// Market regime
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MarketRegime {
    #[default]
    Unknown,
    Trending,
    Ranging,
    Volatile,
    Quiet,
    Crisis,
}

impl WorkerState {
    /// Create from current market data
    pub fn new() -> Self {
        Self {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis() as u64,
            ..Default::default()
        }
    }

    /// Update price
    pub fn with_price(mut self, bid: f64, ask: f64) -> Self {
        self.bid = bid;
        self.ask = ask;
        self.price = (bid + ask) / 2.0;
        self.spread_pct = if self.price > 0.0 {
            (ask - bid) / self.price
        } else {
            0.0
        };
        self
    }

    /// Update position
    pub fn with_position(mut self, position: f64) -> Self {
        self.position = position;
        self
    }

    /// Update PnL
    pub fn with_pnl(mut self, unrealized: f64, realized: f64) -> Self {
        self.unrealized_pnl = unrealized;
        self.realized_pnl = realized;
        self
    }

    /// Update volatility
    pub fn with_volatility(mut self, volatility: f64) -> Self {
        self.volatility = volatility;
        self
    }

    /// Update regime
    pub fn with_regime(mut self, regime: MarketRegime) -> Self {
        self.regime = regime;
        self
    }
}

/// Action output from worker
#[derive(Debug, Clone)]
pub struct WorkerAction {
    /// Action type
    pub action_type: ActionType,
    /// Target symbol
    pub symbol: String,
    /// Quantity/size
    pub quantity: f64,
    /// Price (for limit orders)
    pub price: Option<f64>,
    /// Confidence in this action (0.0 to 1.0)
    pub confidence: f64,
    /// Skill that generated this action
    pub skill_id: Option<String>,
    /// Procedure being executed
    pub procedure_id: Option<String>,
    /// Goal this action is for
    pub goal_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionType {
    /// Do nothing
    NoOp,
    /// Place market buy
    MarketBuy,
    /// Place market sell
    MarketSell,
    /// Place limit buy
    LimitBuy,
    /// Place limit sell
    LimitSell,
    /// Cancel specific order
    CancelOrder,
    /// Cancel all orders
    CancelAll,
    /// Adjust stop loss
    AdjustStop,
    /// Adjust take profit
    AdjustTakeProfit,
    /// Scale position
    ScalePosition,
    /// Flatten position
    Flatten,
}

impl Default for ActionType {
    fn default() -> Self {
        ActionType::NoOp
    }
}

impl WorkerAction {
    /// Create a new action
    pub fn new(action_type: ActionType, symbol: &str, goal_id: &str) -> Self {
        Self {
            action_type,
            symbol: symbol.to_string(),
            quantity: 0.0,
            price: None,
            confidence: 0.5,
            skill_id: None,
            procedure_id: None,
            goal_id: goal_id.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_millis() as u64,
            metadata: HashMap::new(),
        }
    }

    /// Set quantity
    pub fn with_quantity(mut self, quantity: f64) -> Self {
        self.quantity = quantity;
        self
    }

    /// Set price
    pub fn with_price(mut self, price: f64) -> Self {
        self.price = Some(price);
        self
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set skill ID
    pub fn with_skill(mut self, skill_id: &str) -> Self {
        self.skill_id = Some(skill_id.to_string());
        self
    }

    /// Set procedure ID
    pub fn with_procedure(mut self, procedure_id: &str) -> Self {
        self.procedure_id = Some(procedure_id.to_string());
        self
    }
}

/// Result of action execution
#[derive(Debug, Clone)]
pub struct ActionResult {
    /// Action that was executed
    pub action: WorkerAction,
    /// Whether execution succeeded
    pub success: bool,
    /// Fill price (if applicable)
    pub fill_price: Option<f64>,
    /// Filled quantity
    pub filled_quantity: f64,
    /// Slippage
    pub slippage: f64,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Error message if failed
    pub error: Option<String>,
}

impl ActionResult {
    /// Create successful result
    pub fn success(action: WorkerAction) -> Self {
        Self {
            action,
            success: true,
            fill_price: None,
            filled_quantity: 0.0,
            slippage: 0.0,
            execution_time_ms: 0,
            error: None,
        }
    }

    /// Create failed result
    pub fn failure(action: WorkerAction, error: &str) -> Self {
        Self {
            action,
            success: false,
            fill_price: None,
            filled_quantity: 0.0,
            slippage: 0.0,
            execution_time_ms: 0,
            error: Some(error.to_string()),
        }
    }

    /// Set fill details
    pub fn with_fill(mut self, price: f64, quantity: f64, slippage: f64) -> Self {
        self.fill_price = Some(price);
        self.filled_quantity = quantity;
        self.slippage = slippage;
        self
    }

    /// Set execution time
    pub fn with_time(mut self, time_ms: u64) -> Self {
        self.execution_time_ms = time_ms;
        self
    }
}

/// Worker agent configuration
#[derive(Debug, Clone)]
pub struct WorkerAgentConfig {
    /// Maximum concurrent goals
    pub max_goals: usize,
    /// Maximum action history
    pub max_history: usize,
    /// Exploration rate (epsilon)
    pub exploration_rate: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub discount_factor: f64,
    /// Minimum confidence to execute
    pub min_confidence: f64,
    /// Enable learning
    pub enable_learning: bool,
    /// Default symbol
    pub default_symbol: String,
}

impl Default for WorkerAgentConfig {
    fn default() -> Self {
        Self {
            max_goals: 10,
            max_history: 1000,
            exploration_rate: 0.1,
            learning_rate: 0.01,
            discount_factor: 0.99,
            min_confidence: 0.3,
            enable_learning: true,
            default_symbol: "BTC".to_string(),
        }
    }
}

/// Q-value entry for state-action pairs
#[derive(Debug, Clone, Default)]
pub struct QValue {
    pub value: f64,
    pub visits: u64,
    pub avg_reward: f64,
}

impl QValue {
    /// Update Q-value with new experience
    pub fn update(&mut self, reward: f64, learning_rate: f64) {
        self.visits += 1;
        self.value += learning_rate * (reward - self.value);
        let n = self.visits as f64;
        self.avg_reward = ((self.avg_reward * (n - 1.0)) + reward) / n;
    }
}

/// Statistics for the worker agent
#[derive(Debug, Clone, Default)]
pub struct WorkerStats {
    pub total_goals: u64,
    pub completed_goals: u64,
    pub failed_goals: u64,
    pub total_actions: u64,
    pub successful_actions: u64,
    pub avg_action_confidence: f64,
    pub avg_execution_time: f64,
    pub total_reward: f64,
    pub goals_by_type: HashMap<GoalType, u64>,
}

impl WorkerStats {
    /// Calculate goal success rate
    pub fn goal_success_rate(&self) -> f64 {
        if self.total_goals == 0 {
            return 0.0;
        }
        self.completed_goals as f64 / self.total_goals as f64
    }

    /// Calculate action success rate
    pub fn action_success_rate(&self) -> f64 {
        if self.total_actions == 0 {
            return 0.0;
        }
        self.successful_actions as f64 / self.total_actions as f64
    }
}

/// Feudal RL worker implementation
pub struct WorkerAgent {
    /// Configuration
    config: WorkerAgentConfig,
    /// Current goals (priority queue by urgency)
    goals: VecDeque<Goal>,
    /// Current state
    state: WorkerState,
    /// Q-values for goal-action pairs
    q_values: HashMap<String, HashMap<ActionType, QValue>>,
    /// Action history
    action_history: VecDeque<ActionResult>,
    /// Goal history
    goal_history: VecDeque<(Goal, bool)>,
    /// Statistics
    stats: WorkerStats,
    /// Whether agent is active
    active: bool,
    /// Last action timestamp
    last_action_time: u64,
    /// Random state for exploration
    rng_state: u64,
}

impl Default for WorkerAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerAgent {
    /// Create a new instance
    pub fn new() -> Self {
        Self {
            config: WorkerAgentConfig::default(),
            goals: VecDeque::new(),
            state: WorkerState::new(),
            q_values: HashMap::new(),
            action_history: VecDeque::new(),
            goal_history: VecDeque::new(),
            stats: WorkerStats::default(),
            active: true,
            last_action_time: 0,
            rng_state: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: WorkerAgentConfig) -> Self {
        Self {
            config,
            goals: VecDeque::new(),
            state: WorkerState::new(),
            q_values: HashMap::new(),
            action_history: VecDeque::new(),
            goal_history: VecDeque::new(),
            stats: WorkerStats::default(),
            active: true,
            last_action_time: 0,
            rng_state: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or(Duration::ZERO)
                .as_nanos() as u64,
        }
    }

    /// Simple pseudo-random number generator
    fn random(&mut self) -> f64 {
        // xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f64) / (u64::MAX as f64)
    }

    /// Set a new goal from manager
    pub fn set_goal(&mut self, goal: Goal) -> Result<()> {
        if self.goals.len() >= self.config.max_goals {
            // Remove lowest priority expired goal
            self.goals.retain(|g| !g.is_expired());

            if self.goals.len() >= self.config.max_goals {
                return Err(Error::InvalidState("Maximum goals reached".to_string()));
            }
        }

        self.stats.total_goals += 1;
        *self.stats.goals_by_type.entry(goal.goal_type).or_insert(0) += 1;

        // Insert by priority (higher priority first)
        let priority = goal.goal_type.priority();
        let pos = self
            .goals
            .iter()
            .position(|g| g.goal_type.priority() < priority)
            .unwrap_or(self.goals.len());

        self.goals.insert(pos, goal);

        Ok(())
    }

    /// Get current highest priority goal
    pub fn current_goal(&self) -> Option<&Goal> {
        self.goals.front()
    }

    /// Update state
    pub fn update_state(&mut self, state: WorkerState) {
        self.state = state;
    }

    /// Get current state
    pub fn state(&self) -> &WorkerState {
        &self.state
    }

    /// Select action for current goal
    pub fn select_action(&mut self) -> Option<WorkerAction> {
        let goal = self.goals.front()?.clone();

        if goal.is_expired() {
            return None;
        }

        let state_key = self.discretize_state(&goal);
        let action_type = self.select_action_type(&state_key, &goal);

        let symbol = goal
            .symbol
            .clone()
            .unwrap_or_else(|| self.config.default_symbol.clone());

        let mut action = WorkerAction::new(action_type, &symbol, &goal.id);

        // Set quantity based on goal magnitude
        action.quantity = self.calculate_quantity(&goal, action_type);

        // Set price for limit orders
        if matches!(action_type, ActionType::LimitBuy | ActionType::LimitSell) {
            action.price = Some(self.calculate_limit_price(action_type));
        }

        // Set confidence
        action.confidence = self.calculate_confidence(&state_key, action_type);

        // Check minimum confidence
        if action.confidence < self.config.min_confidence && action_type != ActionType::NoOp {
            action.action_type = ActionType::NoOp;
            action.confidence = 1.0;
        }

        Some(action)
    }

    /// Discretize state for Q-learning
    fn discretize_state(&self, goal: &Goal) -> String {
        let pos_bucket = if self.state.position.abs() < 0.01 {
            "flat"
        } else if self.state.position > 0.0 {
            "long"
        } else {
            "short"
        };

        let vol_bucket = if self.state.volatility < 0.01 {
            "low"
        } else if self.state.volatility < 0.03 {
            "med"
        } else {
            "high"
        };

        let pnl_bucket = if self.state.unrealized_pnl > 0.02 {
            "profit"
        } else if self.state.unrealized_pnl < -0.02 {
            "loss"
        } else {
            "neutral"
        };

        format!(
            "{:?}_{}_{}_{}",
            goal.goal_type, pos_bucket, vol_bucket, pnl_bucket
        )
    }

    /// Select action type using epsilon-greedy policy
    fn select_action_type(&mut self, state_key: &str, goal: &Goal) -> ActionType {
        // Get available actions for this goal type
        let available_actions = self.get_available_actions(goal);

        if available_actions.is_empty() {
            return ActionType::NoOp;
        }

        // Epsilon-greedy exploration
        if self.random() < self.config.exploration_rate {
            // Random action
            let idx = (self.random() * available_actions.len() as f64) as usize;
            return available_actions[idx.min(available_actions.len() - 1)];
        }

        // Greedy action
        let q_values = self.q_values.get(state_key);

        available_actions
            .into_iter()
            .max_by(|a, b| {
                let qa = q_values
                    .and_then(|q| q.get(a))
                    .map(|v| v.value)
                    .unwrap_or(0.0);
                let qb = q_values
                    .and_then(|q| q.get(b))
                    .map(|v| v.value)
                    .unwrap_or(0.0);
                qa.partial_cmp(&qb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(ActionType::NoOp)
    }

    /// Get available actions for a goal type
    fn get_available_actions(&self, goal: &Goal) -> Vec<ActionType> {
        match goal.goal_type {
            GoalType::EnterPosition => {
                if goal.direction > 0 {
                    vec![
                        ActionType::MarketBuy,
                        ActionType::LimitBuy,
                        ActionType::NoOp,
                    ]
                } else if goal.direction < 0 {
                    vec![
                        ActionType::MarketSell,
                        ActionType::LimitSell,
                        ActionType::NoOp,
                    ]
                } else {
                    vec![ActionType::NoOp]
                }
            }
            GoalType::ExitPosition => {
                if self.state.position > 0.0 {
                    vec![
                        ActionType::MarketSell,
                        ActionType::LimitSell,
                        ActionType::Flatten,
                        ActionType::NoOp,
                    ]
                } else if self.state.position < 0.0 {
                    vec![
                        ActionType::MarketBuy,
                        ActionType::LimitBuy,
                        ActionType::Flatten,
                        ActionType::NoOp,
                    ]
                } else {
                    vec![ActionType::NoOp]
                }
            }
            GoalType::AdjustPosition => vec![ActionType::ScalePosition, ActionType::NoOp],
            GoalType::ManageRisk => vec![
                ActionType::AdjustStop,
                ActionType::ScalePosition,
                ActionType::Flatten,
                ActionType::NoOp,
            ],
            GoalType::OptimizeExecution => vec![
                ActionType::CancelOrder,
                ActionType::LimitBuy,
                ActionType::LimitSell,
                ActionType::NoOp,
            ],
            GoalType::Emergency => vec![ActionType::CancelAll, ActionType::Flatten],
            GoalType::Hedge | GoalType::Rebalance => {
                vec![
                    ActionType::MarketBuy,
                    ActionType::MarketSell,
                    ActionType::LimitBuy,
                    ActionType::LimitSell,
                    ActionType::NoOp,
                ]
            }
            GoalType::Observe => vec![ActionType::NoOp],
        }
    }

    /// Calculate quantity for action
    fn calculate_quantity(&self, goal: &Goal, action_type: ActionType) -> f64 {
        match action_type {
            ActionType::Flatten => self.state.position.abs(),
            ActionType::ScalePosition => goal.magnitude * self.state.position.abs().max(1.0),
            _ => goal.magnitude,
        }
    }

    /// Calculate limit price
    fn calculate_limit_price(&self, action_type: ActionType) -> f64 {
        match action_type {
            ActionType::LimitBuy => {
                // Bid slightly below mid for better fill
                self.state.bid + (self.state.ask - self.state.bid) * 0.25
            }
            ActionType::LimitSell => {
                // Ask slightly above mid
                self.state.ask - (self.state.ask - self.state.bid) * 0.25
            }
            _ => self.state.price,
        }
    }

    /// Calculate confidence for action
    fn calculate_confidence(&self, state_key: &str, action_type: ActionType) -> f64 {
        let q_values = self.q_values.get(state_key);

        let q_value = q_values
            .and_then(|q| q.get(&action_type))
            .cloned()
            .unwrap_or_default();

        // Base confidence on visits and value
        let visit_factor = 1.0 - (-0.01 * q_value.visits as f64).exp();
        let value_factor = (q_value.value + 1.0).clamp(0.0, 2.0) / 2.0;

        (visit_factor * 0.3 + value_factor * 0.7).clamp(0.1, 1.0)
    }

    /// Record action result and learn
    pub fn record_result(&mut self, result: ActionResult) {
        self.stats.total_actions += 1;
        if result.success {
            self.stats.successful_actions += 1;
        }

        // Update average confidence
        let n = self.stats.total_actions as f64;
        self.stats.avg_action_confidence =
            ((self.stats.avg_action_confidence * (n - 1.0)) + result.action.confidence) / n;

        // Update average execution time
        self.stats.avg_execution_time =
            ((self.stats.avg_execution_time * (n - 1.0)) + result.execution_time_ms as f64) / n;

        // Learn from result
        if self.config.enable_learning {
            self.learn_from_result(&result);
        }

        // Update last action time
        self.last_action_time = result.action.timestamp;

        // Store in history
        self.action_history.push_front(result);
        while self.action_history.len() > self.config.max_history {
            self.action_history.pop_back();
        }
    }

    /// Learn from action result
    fn learn_from_result(&mut self, result: &ActionResult) {
        // Find corresponding goal
        let goal = self.goals.iter().find(|g| g.id == result.action.goal_id);

        if let Some(goal) = goal {
            let state_key = self.discretize_state(goal);

            // Calculate reward
            let reward = self.calculate_reward(result, goal);
            self.stats.total_reward += reward;

            // Update Q-value
            let q_entry = self
                .q_values
                .entry(state_key)
                .or_insert_with(HashMap::new)
                .entry(result.action.action_type)
                .or_insert_with(QValue::default);

            q_entry.update(reward, self.config.learning_rate);
        }
    }

    /// Calculate reward for action
    fn calculate_reward(&self, result: &ActionResult, goal: &Goal) -> f64 {
        let mut reward = goal.intrinsic_reward;

        // Success bonus
        if result.success {
            reward += 0.1;
        } else {
            reward -= 0.1;
        }

        // Slippage penalty
        reward -= result.slippage * 10.0;

        // Fill rate bonus
        if result.action.quantity > 0.0 {
            let fill_rate = result.filled_quantity / result.action.quantity;
            reward += fill_rate * 0.1;
        }

        // Execution time penalty (relative to goal urgency)
        let time_penalty = (result.execution_time_ms as f64 / 1000.0) * goal.urgency * 0.01;
        reward -= time_penalty;

        reward.clamp(-1.0, 1.0)
    }

    /// Complete current goal
    pub fn complete_goal(&mut self, success: bool) {
        if let Some(goal) = self.goals.pop_front() {
            if success {
                self.stats.completed_goals += 1;
            } else {
                self.stats.failed_goals += 1;
            }

            self.goal_history.push_front((goal, success));
            while self.goal_history.len() > self.config.max_history {
                self.goal_history.pop_back();
            }
        }
    }

    /// Clean up expired goals
    pub fn cleanup_expired(&mut self) {
        let expired: Vec<Goal> = self
            .goals
            .iter()
            .filter(|g| g.is_expired())
            .cloned()
            .collect();

        self.goals.retain(|g| !g.is_expired());

        for goal in expired {
            self.stats.failed_goals += 1;
            self.goal_history.push_front((goal, false));
        }

        while self.goal_history.len() > self.config.max_history {
            self.goal_history.pop_back();
        }
    }

    /// Set exploration rate
    pub fn set_exploration_rate(&mut self, rate: f64) {
        self.config.exploration_rate = rate.clamp(0.0, 1.0);
    }

    /// Decay exploration rate
    pub fn decay_exploration(&mut self, decay: f64) {
        self.config.exploration_rate *= decay;
        self.config.exploration_rate = self.config.exploration_rate.max(0.01);
    }

    /// Get exploration rate
    pub fn exploration_rate(&self) -> f64 {
        self.config.exploration_rate
    }

    /// Get statistics
    pub fn stats(&self) -> &WorkerStats {
        &self.stats
    }

    /// Get action history
    pub fn action_history(&self) -> &VecDeque<ActionResult> {
        &self.action_history
    }

    /// Get goal history
    pub fn goal_history(&self) -> &VecDeque<(Goal, bool)> {
        &self.goal_history
    }

    /// Get pending goals count
    pub fn pending_goals(&self) -> usize {
        self.goals.len()
    }

    /// Check if agent is active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Activate agent
    pub fn activate(&mut self) {
        self.active = true;
    }

    /// Deactivate agent
    pub fn deactivate(&mut self) {
        self.active = false;
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = WorkerAgent::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_set_goal() {
        let mut agent = WorkerAgent::new();

        let goal = Goal::new(GoalType::EnterPosition)
            .with_symbol("BTC")
            .with_direction(1)
            .with_magnitude(1.0);

        assert!(agent.set_goal(goal).is_ok());
        assert_eq!(agent.pending_goals(), 1);
    }

    #[test]
    fn test_goal_priority_ordering() {
        let mut agent = WorkerAgent::new();

        // Set low priority goal first
        let observe = Goal::new(GoalType::Observe);
        agent.set_goal(observe).unwrap();

        // Set high priority goal
        let emergency = Goal::new(GoalType::Emergency);
        agent.set_goal(emergency).unwrap();

        // Emergency should be first
        let current = agent.current_goal().unwrap();
        assert_eq!(current.goal_type, GoalType::Emergency);
    }

    #[test]
    fn test_select_action() {
        let mut agent = WorkerAgent::new();

        let goal = Goal::new(GoalType::EnterPosition)
            .with_symbol("BTC")
            .with_direction(1)
            .with_magnitude(1.0);

        agent.set_goal(goal).unwrap();

        let action = agent.select_action();
        assert!(action.is_some());

        let action = action.unwrap();
        assert!(matches!(
            action.action_type,
            ActionType::MarketBuy | ActionType::LimitBuy | ActionType::NoOp
        ));
    }

    #[test]
    fn test_update_state() {
        let mut agent = WorkerAgent::new();

        let state = WorkerState::new()
            .with_price(100.0, 100.1)
            .with_position(1.0)
            .with_volatility(0.02);

        agent.update_state(state);

        assert_eq!(agent.state().bid, 100.0);
        assert_eq!(agent.state().ask, 100.1);
        assert_eq!(agent.state().position, 1.0);
    }

    #[test]
    fn test_record_result() {
        let mut agent = WorkerAgent::new();

        let action = WorkerAction::new(ActionType::MarketBuy, "BTC", "goal_1");
        let result = ActionResult::success(action).with_fill(100.0, 1.0, 0.001);

        agent.record_result(result);

        assert_eq!(agent.stats().total_actions, 1);
        assert_eq!(agent.stats().successful_actions, 1);
    }

    #[test]
    fn test_complete_goal() {
        let mut agent = WorkerAgent::new();

        let goal = Goal::new(GoalType::EnterPosition);
        agent.set_goal(goal).unwrap();

        agent.complete_goal(true);

        assert_eq!(agent.pending_goals(), 0);
        assert_eq!(agent.stats().completed_goals, 1);
    }

    #[test]
    fn test_goal_expiration() {
        let mut agent = WorkerAgent::new();

        // Create goal with 0 timeout (immediately expired)
        let goal = Goal::new(GoalType::Observe).with_timeout(0);
        agent.set_goal(goal).unwrap();

        // Clean up expired
        agent.cleanup_expired();

        assert_eq!(agent.pending_goals(), 0);
        assert_eq!(agent.stats().failed_goals, 1);
    }

    #[test]
    fn test_exploration_decay() {
        let mut agent = WorkerAgent::new();
        let initial = agent.exploration_rate();

        agent.decay_exploration(0.9);
        assert!(agent.exploration_rate() < initial);

        agent.decay_exploration(0.9);
        assert!(agent.exploration_rate() >= 0.01); // Should not go below minimum
    }

    #[test]
    fn test_activate_deactivate() {
        let mut agent = WorkerAgent::new();

        assert!(agent.is_active());

        agent.deactivate();
        assert!(!agent.is_active());

        agent.activate();
        assert!(agent.is_active());
    }

    #[test]
    fn test_goal_builder() {
        let goal = Goal::new(GoalType::EnterPosition)
            .with_symbol("ETH")
            .with_direction(1)
            .with_magnitude(2.5)
            .with_urgency(0.8)
            .with_timeout(5000)
            .with_param("stop_loss", 0.02);

        assert_eq!(goal.symbol, Some("ETH".to_string()));
        assert_eq!(goal.direction, 1);
        assert_eq!(goal.magnitude, 2.5);
        assert_eq!(goal.urgency, 0.8);
        assert_eq!(goal.timeout_ms, 5000);
        assert_eq!(goal.params.get("stop_loss"), Some(&0.02));
    }

    #[test]
    fn test_action_builder() {
        let action = WorkerAction::new(ActionType::LimitBuy, "BTC", "goal_1")
            .with_quantity(1.5)
            .with_price(50000.0)
            .with_confidence(0.9)
            .with_skill("momentum_entry");

        assert_eq!(action.quantity, 1.5);
        assert_eq!(action.price, Some(50000.0));
        assert_eq!(action.confidence, 0.9);
        assert_eq!(action.skill_id, Some("momentum_entry".to_string()));
    }

    #[test]
    fn test_state_builder() {
        let state = WorkerState::new()
            .with_price(100.0, 101.0)
            .with_position(-1.5)
            .with_pnl(0.05, 0.1)
            .with_volatility(0.025)
            .with_regime(MarketRegime::Trending);

        assert_eq!(state.bid, 100.0);
        assert_eq!(state.ask, 101.0);
        assert_eq!(state.position, -1.5);
        assert_eq!(state.unrealized_pnl, 0.05);
        assert_eq!(state.realized_pnl, 0.1);
        assert_eq!(state.volatility, 0.025);
        assert_eq!(state.regime, MarketRegime::Trending);
    }

    #[test]
    fn test_max_goals_limit() {
        let config = WorkerAgentConfig {
            max_goals: 2,
            ..Default::default()
        };
        let mut agent = WorkerAgent::with_config(config);

        agent.set_goal(Goal::new(GoalType::EnterPosition)).unwrap();
        agent.set_goal(Goal::new(GoalType::ExitPosition)).unwrap();

        // Third goal should fail
        let result = agent.set_goal(Goal::new(GoalType::Observe));
        assert!(result.is_err());
    }

    #[test]
    fn test_emergency_actions() {
        let mut agent = WorkerAgent::new();

        let goal = Goal::new(GoalType::Emergency);
        agent.set_goal(goal).unwrap();

        let action = agent.select_action().unwrap();

        // Emergency should only allow CancelAll or Flatten
        assert!(matches!(
            action.action_type,
            ActionType::CancelAll | ActionType::Flatten
        ));
    }

    #[test]
    fn test_stats() {
        let mut agent = WorkerAgent::new();

        agent.set_goal(Goal::new(GoalType::EnterPosition)).unwrap();
        agent.complete_goal(true);

        agent.set_goal(Goal::new(GoalType::ExitPosition)).unwrap();
        agent.complete_goal(false);

        let stats = agent.stats();
        assert_eq!(stats.total_goals, 2);
        assert_eq!(stats.completed_goals, 1);
        assert_eq!(stats.failed_goals, 1);
        assert_eq!(stats.goal_success_rate(), 0.5);
    }

    #[test]
    fn test_q_value_update() {
        let mut q = QValue::default();
        assert_eq!(q.value, 0.0);

        q.update(1.0, 0.1);
        assert!(q.value > 0.0);
        assert_eq!(q.visits, 1);

        q.update(-0.5, 0.1);
        assert_eq!(q.visits, 2);
    }

    #[test]
    fn test_exit_position_actions() {
        let mut agent = WorkerAgent::new();

        // Set state with long position
        agent.update_state(WorkerState::new().with_position(1.0));

        let goal = Goal::new(GoalType::ExitPosition);
        agent.set_goal(goal).unwrap();

        let action = agent.select_action().unwrap();

        // Should select sell actions for exiting long
        assert!(matches!(
            action.action_type,
            ActionType::MarketSell | ActionType::LimitSell | ActionType::Flatten | ActionType::NoOp
        ));
    }

    #[test]
    fn test_learning_disabled() {
        let config = WorkerAgentConfig {
            enable_learning: false,
            ..Default::default()
        };
        let mut agent = WorkerAgent::with_config(config);

        let goal = Goal::new(GoalType::EnterPosition);
        agent.set_goal(goal).unwrap();

        let action = WorkerAction::new(ActionType::MarketBuy, "BTC", "goal_1");
        let result = ActionResult::success(action);

        agent.record_result(result);

        // Q-values should not be updated
        assert!(agent.q_values.is_empty());
    }
}
