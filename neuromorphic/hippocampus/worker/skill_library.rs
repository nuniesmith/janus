//! Learned micro-strategies
//!
//! Part of the Hippocampus region
//! Component: worker
//!
//! This module implements a skill library that stores and manages learned
//! micro-strategies for trading. Skills are reusable patterns that encode
//! successful trading behaviors discovered through experience.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Category of trading skill
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkillCategory {
    /// Entry timing skills
    EntryTiming,
    /// Exit timing skills
    ExitTiming,
    /// Position sizing skills
    PositionSizing,
    /// Risk management skills
    RiskManagement,
    /// Order execution skills
    OrderExecution,
    /// Market reading skills
    MarketReading,
    /// Pattern recognition skills
    PatternRecognition,
    /// Regime detection skills
    RegimeDetection,
    /// Volatility handling skills
    VolatilityHandling,
    /// Correlation trading skills
    CorrelationTrading,
}

impl Default for SkillCategory {
    fn default() -> Self {
        SkillCategory::OrderExecution
    }
}

impl SkillCategory {
    /// Get description of the skill category
    pub fn description(&self) -> &'static str {
        match self {
            SkillCategory::EntryTiming => "Skills for optimal entry timing",
            SkillCategory::ExitTiming => "Skills for optimal exit timing",
            SkillCategory::PositionSizing => "Skills for position size determination",
            SkillCategory::RiskManagement => "Skills for managing risk exposure",
            SkillCategory::OrderExecution => "Skills for efficient order execution",
            SkillCategory::MarketReading => "Skills for reading market conditions",
            SkillCategory::PatternRecognition => "Skills for recognizing chart patterns",
            SkillCategory::RegimeDetection => "Skills for detecting market regimes",
            SkillCategory::VolatilityHandling => "Skills for handling volatility",
            SkillCategory::CorrelationTrading => "Skills for correlation-based trading",
        }
    }
}

/// Complexity level of a skill
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SkillComplexity {
    /// Basic, fundamental skill
    Basic,
    /// Intermediate skill building on basics
    Intermediate,
    /// Advanced skill requiring experience
    Advanced,
    /// Expert-level skill
    Expert,
}

impl Default for SkillComplexity {
    fn default() -> Self {
        SkillComplexity::Basic
    }
}

impl SkillComplexity {
    /// Get minimum experience required
    pub fn min_experience(&self) -> u64 {
        match self {
            SkillComplexity::Basic => 0,
            SkillComplexity::Intermediate => 100,
            SkillComplexity::Advanced => 500,
            SkillComplexity::Expert => 2000,
        }
    }

    /// Get learning rate modifier
    pub fn learning_rate_modifier(&self) -> f64 {
        match self {
            SkillComplexity::Basic => 1.0,
            SkillComplexity::Intermediate => 0.7,
            SkillComplexity::Advanced => 0.4,
            SkillComplexity::Expert => 0.2,
        }
    }
}

/// Market condition for skill applicability
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MarketCondition {
    /// Any market condition
    Any,
    /// Trending up
    TrendingUp,
    /// Trending down
    TrendingDown,
    /// Range-bound
    Ranging,
    /// High volatility
    HighVolatility,
    /// Low volatility
    LowVolatility,
    /// High liquidity
    HighLiquidity,
    /// Low liquidity
    LowLiquidity,
    /// Market open
    MarketOpen,
    /// Market close
    MarketClose,
    /// News event
    NewsEvent,
}

impl Default for MarketCondition {
    fn default() -> Self {
        MarketCondition::Any
    }
}

/// Skill trigger condition
#[derive(Debug, Clone)]
pub struct SkillTrigger {
    /// Trigger name
    pub name: String,
    /// Trigger type
    pub trigger_type: TriggerType,
    /// Threshold value
    pub threshold: f64,
    /// Comparison operator
    pub comparison: Comparison,
}

impl SkillTrigger {
    /// Create a new trigger
    pub fn new(
        name: &str,
        trigger_type: TriggerType,
        threshold: f64,
        comparison: Comparison,
    ) -> Self {
        Self {
            name: name.to_string(),
            trigger_type,
            threshold,
            comparison,
        }
    }

    /// Evaluate the trigger
    pub fn evaluate(&self, value: f64) -> bool {
        match self.comparison {
            Comparison::GreaterThan => value > self.threshold,
            Comparison::LessThan => value < self.threshold,
            Comparison::GreaterOrEqual => value >= self.threshold,
            Comparison::LessOrEqual => value <= self.threshold,
            Comparison::Equal => (value - self.threshold).abs() < 0.0001,
            Comparison::NotEqual => (value - self.threshold).abs() >= 0.0001,
        }
    }
}

/// Trigger types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerType {
    /// Price-based trigger
    Price,
    /// Volume-based trigger
    Volume,
    /// Volatility-based trigger
    Volatility,
    /// Momentum-based trigger
    Momentum,
    /// Time-based trigger
    Time,
    /// Spread-based trigger
    Spread,
    /// Position-based trigger
    Position,
    /// PnL-based trigger
    PnL,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
    NotEqual,
}

/// Skill action to perform
#[derive(Debug, Clone)]
pub enum SkillAction {
    /// Adjust position
    AdjustPosition {
        direction: i8, // -1, 0, 1
        magnitude: f64,
    },
    /// Adjust stop loss
    AdjustStopLoss {
        offset_type: OffsetType,
        offset: f64,
    },
    /// Adjust take profit
    AdjustTakeProfit {
        offset_type: OffsetType,
        offset: f64,
    },
    /// Change order type
    ChangeOrderType { new_type: String },
    /// Delay action
    Delay { duration_ms: u64 },
    /// Split order
    SplitOrder { num_parts: usize, interval_ms: u64 },
    /// Signal generation
    GenerateSignal { signal_type: String, strength: f64 },
    /// Wait for condition
    WaitFor { condition: String, timeout_ms: u64 },
    /// Custom action
    Custom {
        action_id: String,
        params: HashMap<String, f64>,
    },
}

/// Offset type for price adjustments
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OffsetType {
    /// Absolute price offset
    Absolute,
    /// Percentage offset
    Percentage,
    /// ATR multiple
    ATRMultiple,
    /// Standard deviation multiple
    StdDevMultiple,
}

/// A learned trading skill
#[derive(Debug, Clone)]
pub struct Skill {
    /// Unique skill ID
    pub id: String,
    /// Skill name
    pub name: String,
    /// Skill description
    pub description: String,
    /// Category
    pub category: SkillCategory,
    /// Complexity level
    pub complexity: SkillComplexity,
    /// Applicable market conditions
    pub conditions: Vec<MarketCondition>,
    /// Triggers that activate this skill
    pub triggers: Vec<SkillTrigger>,
    /// Actions to perform
    pub actions: Vec<SkillAction>,
    /// Required prerequisite skills
    pub prerequisites: Vec<String>,
    /// Success rate from historical use
    pub success_rate: f64,
    /// Average reward when used
    pub avg_reward: f64,
    /// Number of times used
    pub usage_count: u64,
    /// Number of successful uses
    pub success_count: u64,
    /// Confidence in this skill
    pub confidence: f64,
    /// When this skill was created
    pub created_at: u64,
    /// When this skill was last used
    pub last_used: Option<u64>,
    /// Whether this is a built-in skill
    pub builtin: bool,
    /// Whether this skill is active
    pub active: bool,
    /// Parameters that can be tuned
    pub parameters: HashMap<String, f64>,
    /// Performance by condition
    pub condition_performance: HashMap<MarketCondition, ConditionStats>,
}

/// Performance statistics for a specific condition
#[derive(Debug, Clone, Default)]
pub struct ConditionStats {
    pub usage_count: u64,
    pub success_count: u64,
    pub avg_reward: f64,
    pub total_reward: f64,
}

impl ConditionStats {
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.usage_count == 0 {
            return 0.5;
        }
        self.success_count as f64 / self.usage_count as f64
    }

    /// Record usage
    pub fn record(&mut self, success: bool, reward: f64) {
        self.usage_count += 1;
        if success {
            self.success_count += 1;
        }
        self.total_reward += reward;
        self.avg_reward = self.total_reward / self.usage_count as f64;
    }
}

impl Skill {
    /// Create a new skill
    pub fn new(id: &str, name: &str, category: SkillCategory) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: String::new(),
            category,
            complexity: SkillComplexity::Basic,
            conditions: vec![MarketCondition::Any],
            triggers: Vec::new(),
            actions: Vec::new(),
            prerequisites: Vec::new(),
            success_rate: 0.5,
            avg_reward: 0.0,
            usage_count: 0,
            success_count: 0,
            confidence: 0.5,
            created_at: now,
            last_used: None,
            builtin: false,
            active: true,
            parameters: HashMap::new(),
            condition_performance: HashMap::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Set complexity
    pub fn with_complexity(mut self, complexity: SkillComplexity) -> Self {
        self.complexity = complexity;
        self
    }

    /// Add market condition
    pub fn with_condition(mut self, condition: MarketCondition) -> Self {
        // If adding a specific condition, remove the default Any condition
        if condition != MarketCondition::Any {
            self.conditions.retain(|c| *c != MarketCondition::Any);
        }
        if !self.conditions.contains(&condition) {
            self.conditions.push(condition);
        }
        self
    }

    /// Add trigger
    pub fn with_trigger(mut self, trigger: SkillTrigger) -> Self {
        self.triggers.push(trigger);
        self
    }

    /// Add action
    pub fn with_action(mut self, action: SkillAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Add prerequisite
    pub fn with_prerequisite(mut self, skill_id: &str) -> Self {
        if !self.prerequisites.contains(&skill_id.to_string()) {
            self.prerequisites.push(skill_id.to_string());
        }
        self
    }

    /// Set parameter
    pub fn with_parameter(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    /// Mark as built-in
    pub fn as_builtin(mut self) -> Self {
        self.builtin = true;
        self
    }

    /// Record usage outcome
    pub fn record_usage(&mut self, success: bool, reward: f64, condition: MarketCondition) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        self.last_used = Some(now);
        self.usage_count += 1;
        if success {
            self.success_count += 1;
        }

        // Update success rate with exponential moving average
        let alpha = 0.1;
        let success_val = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * success_val + (1.0 - alpha) * self.success_rate;

        // Update average reward
        let n = self.usage_count as f64;
        self.avg_reward = ((self.avg_reward * (n - 1.0)) + reward) / n;

        // Update confidence
        self.confidence = self.calculate_confidence();

        // Update condition-specific stats
        self.condition_performance
            .entry(condition)
            .or_default()
            .record(success, reward);
    }

    /// Calculate confidence based on usage and success
    fn calculate_confidence(&self) -> f64 {
        if self.usage_count == 0 {
            return 0.5;
        }

        // More usage = more confidence in the estimate
        let usage_factor = 1.0 - (-0.01 * self.usage_count as f64).exp();

        // High success rate = more confidence
        let success_factor = self.success_rate;

        // Combined confidence
        (usage_factor * 0.4 + success_factor * 0.6).clamp(0.0, 1.0)
    }

    /// Check if skill is applicable to condition
    pub fn is_applicable(&self, condition: MarketCondition) -> bool {
        self.conditions.contains(&MarketCondition::Any) || self.conditions.contains(&condition)
    }

    /// Check if triggers are satisfied
    pub fn check_triggers(&self, values: &HashMap<TriggerType, f64>) -> bool {
        if self.triggers.is_empty() {
            return true;
        }

        self.triggers.iter().all(|trigger| {
            values
                .get(&trigger.trigger_type)
                .map(|v| trigger.evaluate(*v))
                .unwrap_or(false)
        })
    }

    /// Get performance score for ranking
    pub fn performance_score(&self) -> f64 {
        let usage_weight = (self.usage_count as f64).ln().max(1.0);
        let success_weight = self.success_rate;
        let reward_weight = self.avg_reward.clamp(0.0, 1.0);
        let confidence_weight = self.confidence;

        (usage_weight * 0.1 + success_weight * 0.4 + reward_weight * 0.3 + confidence_weight * 0.2)
            .clamp(0.0, 1.0)
    }
}

/// Skill usage record
#[derive(Debug, Clone)]
pub struct SkillUsage {
    /// Skill ID
    pub skill_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Market condition at time of use
    pub condition: MarketCondition,
    /// Whether usage was successful
    pub success: bool,
    /// Reward received
    pub reward: f64,
    /// Context data
    pub context: HashMap<String, f64>,
}

/// Configuration for the skill library
#[derive(Debug, Clone)]
pub struct SkillLibraryConfig {
    /// Maximum skills to store
    pub max_skills: usize,
    /// Maximum usage history per skill
    pub max_history_per_skill: usize,
    /// Enable skill learning
    pub enable_learning: bool,
    /// Minimum success rate to keep skill active
    pub min_success_rate: f64,
    /// Minimum usage before pruning consideration
    pub min_usage_for_pruning: u64,
    /// Enable skill combination
    pub enable_combination: bool,
    /// Learning rate for skill updates
    pub learning_rate: f64,
}

impl Default for SkillLibraryConfig {
    fn default() -> Self {
        Self {
            max_skills: 500,
            max_history_per_skill: 100,
            enable_learning: true,
            min_success_rate: 0.4,
            min_usage_for_pruning: 20,
            enable_combination: true,
            learning_rate: 0.1,
        }
    }
}

/// Learned micro-strategies
pub struct SkillLibrary {
    /// Configuration
    config: SkillLibraryConfig,
    /// All skills indexed by ID
    skills: HashMap<String, Skill>,
    /// Skills indexed by category
    by_category: HashMap<SkillCategory, Vec<String>>,
    /// Skills indexed by condition
    by_condition: HashMap<MarketCondition, Vec<String>>,
    /// Skill usage history
    usage_history: VecDeque<SkillUsage>,
    /// Total usage history
    max_history: usize,
    /// Skill counter for ID generation
    skill_counter: u64,
    /// Unlocked skills (by experience level)
    unlocked_complexity: SkillComplexity,
    /// Current experience points
    experience: u64,
    /// Active skill combinations
    combinations: Vec<SkillCombination>,
}

/// A combination of skills
#[derive(Debug, Clone)]
pub struct SkillCombination {
    /// Combination ID
    pub id: String,
    /// Skill IDs in this combination
    pub skill_ids: Vec<String>,
    /// Performance of this combination
    pub success_rate: f64,
    /// Usage count
    pub usage_count: u64,
}

impl Default for SkillLibrary {
    fn default() -> Self {
        Self::new()
    }
}

impl SkillLibrary {
    /// Create a new instance
    pub fn new() -> Self {
        let mut library = Self {
            config: SkillLibraryConfig::default(),
            skills: HashMap::new(),
            by_category: HashMap::new(),
            by_condition: HashMap::new(),
            usage_history: VecDeque::new(),
            max_history: 10000,
            skill_counter: 0,
            unlocked_complexity: SkillComplexity::Basic,
            experience: 0,
            combinations: Vec::new(),
        };

        library.register_default_skills();
        library
    }

    /// Create with custom configuration
    pub fn with_config(config: SkillLibraryConfig) -> Self {
        let mut library = Self {
            config,
            skills: HashMap::new(),
            by_category: HashMap::new(),
            by_condition: HashMap::new(),
            usage_history: VecDeque::new(),
            max_history: 10000,
            skill_counter: 0,
            unlocked_complexity: SkillComplexity::Basic,
            experience: 0,
            combinations: Vec::new(),
        };

        library.register_default_skills();
        library
    }

    /// Register default built-in skills
    fn register_default_skills(&mut self) {
        // Entry timing skills
        let momentum_entry = Skill::new(
            "momentum_entry",
            "Momentum Entry",
            SkillCategory::EntryTiming,
        )
        .with_description("Enter on strong momentum confirmation")
        .with_complexity(SkillComplexity::Basic)
        .with_condition(MarketCondition::TrendingUp)
        .with_condition(MarketCondition::TrendingDown)
        .with_trigger(SkillTrigger::new(
            "momentum",
            TriggerType::Momentum,
            0.5,
            Comparison::GreaterThan,
        ))
        .with_action(SkillAction::GenerateSignal {
            signal_type: "entry".to_string(),
            strength: 0.7,
        })
        .with_parameter("momentum_threshold", 0.5)
        .as_builtin();
        self.register_skill(momentum_entry);

        let breakout_entry = Skill::new(
            "breakout_entry",
            "Breakout Entry",
            SkillCategory::EntryTiming,
        )
        .with_description("Enter on price breakout with volume confirmation")
        .with_complexity(SkillComplexity::Intermediate)
        .with_condition(MarketCondition::Ranging)
        .with_trigger(SkillTrigger::new(
            "volume",
            TriggerType::Volume,
            1.5,
            Comparison::GreaterThan,
        ))
        .with_action(SkillAction::GenerateSignal {
            signal_type: "breakout".to_string(),
            strength: 0.8,
        })
        .with_parameter("volume_multiplier", 1.5)
        .as_builtin();
        self.register_skill(breakout_entry);

        // Exit timing skills
        let trailing_exit = Skill::new("trailing_exit", "Trailing Exit", SkillCategory::ExitTiming)
            .with_description("Exit using trailing stop")
            .with_complexity(SkillComplexity::Basic)
            .with_condition(MarketCondition::Any)
            .with_action(SkillAction::AdjustStopLoss {
                offset_type: OffsetType::ATRMultiple,
                offset: 2.0,
            })
            .with_parameter("atr_multiple", 2.0)
            .as_builtin();
        self.register_skill(trailing_exit);

        let volatility_exit = Skill::new(
            "volatility_exit",
            "Volatility Exit",
            SkillCategory::ExitTiming,
        )
        .with_description("Exit when volatility spikes")
        .with_complexity(SkillComplexity::Intermediate)
        .with_condition(MarketCondition::HighVolatility)
        .with_trigger(SkillTrigger::new(
            "volatility",
            TriggerType::Volatility,
            0.3,
            Comparison::GreaterThan,
        ))
        .with_action(SkillAction::AdjustPosition {
            direction: 0,
            magnitude: 1.0,
        })
        .as_builtin();
        self.register_skill(volatility_exit);

        // Position sizing skills
        let volatility_sizing = Skill::new(
            "volatility_sizing",
            "Volatility Sizing",
            SkillCategory::PositionSizing,
        )
        .with_description("Size positions inversely to volatility")
        .with_complexity(SkillComplexity::Basic)
        .with_condition(MarketCondition::Any)
        .with_parameter("base_risk", 0.02)
        .with_parameter("vol_divisor", 20.0)
        .as_builtin();
        self.register_skill(volatility_sizing);

        let kelly_sizing = Skill::new(
            "kelly_sizing",
            "Kelly Sizing",
            SkillCategory::PositionSizing,
        )
        .with_description("Size positions using Kelly criterion")
        .with_complexity(SkillComplexity::Advanced)
        .with_condition(MarketCondition::Any)
        .with_parameter("kelly_fraction", 0.25)
        .as_builtin();
        self.register_skill(kelly_sizing);

        // Order execution skills
        let iceberg_execution = Skill::new(
            "iceberg_execution",
            "Iceberg Orders",
            SkillCategory::OrderExecution,
        )
        .with_description("Split large orders into smaller chunks")
        .with_complexity(SkillComplexity::Intermediate)
        .with_condition(MarketCondition::LowLiquidity)
        .with_trigger(SkillTrigger::new(
            "position",
            TriggerType::Position,
            1000.0,
            Comparison::GreaterThan,
        ))
        .with_action(SkillAction::SplitOrder {
            num_parts: 5,
            interval_ms: 1000,
        })
        .with_parameter("chunk_size", 0.2)
        .as_builtin();
        self.register_skill(iceberg_execution);

        let passive_execution = Skill::new(
            "passive_execution",
            "Passive Execution",
            SkillCategory::OrderExecution,
        )
        .with_description("Use limit orders for price improvement")
        .with_complexity(SkillComplexity::Basic)
        .with_condition(MarketCondition::HighLiquidity)
        .with_trigger(SkillTrigger::new(
            "spread",
            TriggerType::Spread,
            0.001,
            Comparison::LessThan,
        ))
        .with_action(SkillAction::ChangeOrderType {
            new_type: "limit".to_string(),
        })
        .as_builtin();
        self.register_skill(passive_execution);

        // Risk management skills
        let stop_tightening = Skill::new(
            "stop_tightening",
            "Stop Tightening",
            SkillCategory::RiskManagement,
        )
        .with_description("Tighten stops in profit")
        .with_complexity(SkillComplexity::Basic)
        .with_condition(MarketCondition::Any)
        .with_trigger(SkillTrigger::new(
            "pnl",
            TriggerType::PnL,
            0.02,
            Comparison::GreaterThan,
        ))
        .with_action(SkillAction::AdjustStopLoss {
            offset_type: OffsetType::Percentage,
            offset: 0.005,
        })
        .as_builtin();
        self.register_skill(stop_tightening);

        let max_loss_exit = Skill::new(
            "max_loss_exit",
            "Maximum Loss Exit",
            SkillCategory::RiskManagement,
        )
        .with_description("Exit when max loss reached")
        .with_complexity(SkillComplexity::Basic)
        .with_condition(MarketCondition::Any)
        .with_trigger(SkillTrigger::new(
            "pnl",
            TriggerType::PnL,
            -0.02,
            Comparison::LessThan,
        ))
        .with_action(SkillAction::AdjustPosition {
            direction: 0,
            magnitude: 1.0,
        })
        .with_parameter("max_loss", 0.02)
        .as_builtin();
        self.register_skill(max_loss_exit);
    }

    /// Register a skill
    pub fn register_skill(&mut self, skill: Skill) {
        let id = skill.id.clone();
        let category = skill.category;
        let conditions = skill.conditions.clone();

        self.skills.insert(id.clone(), skill);

        self.by_category
            .entry(category)
            .or_insert_with(Vec::new)
            .push(id.clone());

        for condition in conditions {
            self.by_condition
                .entry(condition)
                .or_insert_with(Vec::new)
                .push(id.clone());
        }
    }

    /// Get skill by ID
    pub fn get_skill(&self, id: &str) -> Option<&Skill> {
        self.skills.get(id)
    }

    /// Get mutable skill by ID
    pub fn get_skill_mut(&mut self, id: &str) -> Option<&mut Skill> {
        self.skills.get_mut(id)
    }

    /// Get skills by category
    pub fn get_by_category(&self, category: SkillCategory) -> Vec<&Skill> {
        self.by_category
            .get(&category)
            .map(|ids| ids.iter().filter_map(|id| self.skills.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get applicable skills for condition
    pub fn get_applicable(&self, condition: MarketCondition) -> Vec<&Skill> {
        let mut skills: Vec<&Skill> = self
            .skills
            .values()
            .filter(|s| {
                s.active && s.is_applicable(condition) && s.complexity <= self.unlocked_complexity
            })
            .collect();

        // Sort by performance score
        skills.sort_by(|a, b| {
            b.performance_score()
                .partial_cmp(&a.performance_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        skills
    }

    /// Get best skill for category and condition
    pub fn get_best_skill(
        &self,
        category: SkillCategory,
        condition: MarketCondition,
    ) -> Option<&Skill> {
        self.skills
            .values()
            .filter(|s| {
                s.active
                    && s.category == category
                    && s.is_applicable(condition)
                    && s.complexity <= self.unlocked_complexity
            })
            .max_by(|a, b| {
                a.performance_score()
                    .partial_cmp(&b.performance_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Find skills with satisfied triggers
    pub fn find_triggered_skills(
        &self,
        condition: MarketCondition,
        values: &HashMap<TriggerType, f64>,
    ) -> Vec<&Skill> {
        self.get_applicable(condition)
            .into_iter()
            .filter(|s| s.check_triggers(values))
            .collect()
    }

    /// Record skill usage
    pub fn record_usage(
        &mut self,
        skill_id: &str,
        success: bool,
        reward: f64,
        condition: MarketCondition,
        context: HashMap<String, f64>,
    ) -> Result<()> {
        let skill = self
            .skills
            .get_mut(skill_id)
            .ok_or_else(|| Error::NotFound(format!("Skill {} not found", skill_id)))?;

        skill.record_usage(success, reward, condition);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let usage = SkillUsage {
            skill_id: skill_id.to_string(),
            timestamp: now,
            condition,
            success,
            reward,
            context,
        };

        self.usage_history.push_front(usage);
        while self.usage_history.len() > self.max_history {
            self.usage_history.pop_back();
        }

        // Gain experience
        self.add_experience(if success { 10 } else { 2 });

        Ok(())
    }

    /// Add experience points
    pub fn add_experience(&mut self, amount: u64) {
        self.experience += amount;
        self.update_unlocked_complexity();
    }

    /// Update unlocked complexity level
    fn update_unlocked_complexity(&mut self) {
        self.unlocked_complexity = if self.experience >= SkillComplexity::Expert.min_experience() {
            SkillComplexity::Expert
        } else if self.experience >= SkillComplexity::Advanced.min_experience() {
            SkillComplexity::Advanced
        } else if self.experience >= SkillComplexity::Intermediate.min_experience() {
            SkillComplexity::Intermediate
        } else {
            SkillComplexity::Basic
        };
    }

    /// Learn a new skill
    pub fn learn_skill(
        &mut self,
        name: &str,
        category: SkillCategory,
        complexity: SkillComplexity,
    ) -> Result<String> {
        if !self.config.enable_learning {
            return Err(Error::InvalidState(
                "Skill learning is disabled".to_string(),
            ));
        }

        if self.skills.len() >= self.config.max_skills {
            return Err(Error::InvalidState("Maximum skills reached".to_string()));
        }

        self.skill_counter += 1;
        let id = format!("learned_{}_{}", category as u8, self.skill_counter);

        let skill = Skill::new(&id, name, category).with_complexity(complexity);

        self.register_skill(skill);

        Ok(id)
    }

    /// Prune underperforming skills
    pub fn prune_skills(&mut self) {
        let min_rate = self.config.min_success_rate;
        let min_usage = self.config.min_usage_for_pruning;

        let to_remove: Vec<String> = self
            .skills
            .iter()
            .filter(|(_, s)| !s.builtin && s.usage_count >= min_usage && s.success_rate < min_rate)
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            self.remove_skill(&id);
        }
    }

    /// Remove a skill
    fn remove_skill(&mut self, id: &str) {
        if let Some(skill) = self.skills.remove(id) {
            // Remove from category index
            if let Some(ids) = self.by_category.get_mut(&skill.category) {
                ids.retain(|i| i != id);
            }

            // Remove from condition indexes
            for condition in &skill.conditions {
                if let Some(ids) = self.by_condition.get_mut(condition) {
                    ids.retain(|i| i != id);
                }
            }
        }
    }

    /// Create skill combination
    pub fn create_combination(&mut self, skill_ids: Vec<String>) -> Result<String> {
        if !self.config.enable_combination {
            return Err(Error::InvalidState(
                "Skill combination is disabled".to_string(),
            ));
        }

        // Verify all skills exist
        for id in &skill_ids {
            if !self.skills.contains_key(id) {
                return Err(Error::NotFound(format!("Skill {} not found", id)));
            }
        }

        let combo_id = format!("combo_{}", self.combinations.len());

        self.combinations.push(SkillCombination {
            id: combo_id.clone(),
            skill_ids,
            success_rate: 0.5,
            usage_count: 0,
        });

        Ok(combo_id)
    }

    /// Get usage history
    pub fn usage_history(&self) -> &VecDeque<SkillUsage> {
        &self.usage_history
    }

    /// Get recent usage for a skill
    pub fn recent_usage(&self, skill_id: &str, limit: usize) -> Vec<&SkillUsage> {
        self.usage_history
            .iter()
            .filter(|u| u.skill_id == skill_id)
            .take(limit)
            .collect()
    }

    /// Get experience level
    pub fn experience(&self) -> u64 {
        self.experience
    }

    /// Get unlocked complexity
    pub fn unlocked_complexity(&self) -> SkillComplexity {
        self.unlocked_complexity
    }

    /// Get all skills
    pub fn all_skills(&self) -> impl Iterator<Item = &Skill> {
        self.skills.values()
    }

    /// Get active skill count
    pub fn active_count(&self) -> usize {
        self.skills.values().filter(|s| s.active).count()
    }

    /// Get total skill count
    pub fn len(&self) -> usize {
        self.skills.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
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
        let instance = SkillLibrary::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_default_skills() {
        let library = SkillLibrary::new();
        assert!(!library.is_empty());
        assert!(library.len() >= 10);

        // Check some defaults exist
        assert!(library.get_skill("momentum_entry").is_some());
        assert!(library.get_skill("trailing_exit").is_some());
    }

    #[test]
    fn test_get_by_category() {
        let library = SkillLibrary::new();

        let entry_skills = library.get_by_category(SkillCategory::EntryTiming);
        assert!(!entry_skills.is_empty());

        let risk_skills = library.get_by_category(SkillCategory::RiskManagement);
        assert!(!risk_skills.is_empty());
    }

    #[test]
    fn test_get_applicable() {
        let library = SkillLibrary::new();

        let applicable = library.get_applicable(MarketCondition::TrendingUp);
        assert!(!applicable.is_empty());

        // Should include momentum entry
        assert!(applicable.iter().any(|s| s.id == "momentum_entry"));
    }

    #[test]
    fn test_get_best_skill() {
        let library = SkillLibrary::new();

        let best = library.get_best_skill(SkillCategory::ExitTiming, MarketCondition::Any);
        assert!(best.is_some());
    }

    #[test]
    fn test_record_usage() {
        let mut library = SkillLibrary::new();

        let result = library.record_usage(
            "momentum_entry",
            true,
            0.05,
            MarketCondition::TrendingUp,
            HashMap::new(),
        );
        assert!(result.is_ok());

        let skill = library.get_skill("momentum_entry").unwrap();
        assert_eq!(skill.usage_count, 1);
        assert_eq!(skill.success_count, 1);
    }

    #[test]
    fn test_experience_progression() {
        let mut library = SkillLibrary::new();

        assert_eq!(library.unlocked_complexity(), SkillComplexity::Basic);

        // Add enough experience for intermediate
        library.add_experience(100);
        assert_eq!(library.unlocked_complexity(), SkillComplexity::Intermediate);

        // Add enough for advanced
        library.add_experience(400);
        assert_eq!(library.unlocked_complexity(), SkillComplexity::Advanced);
    }

    #[test]
    fn test_learn_skill() {
        let mut library = SkillLibrary::new();
        let initial_count = library.len();

        let result = library.learn_skill(
            "Custom Entry",
            SkillCategory::EntryTiming,
            SkillComplexity::Basic,
        );
        assert!(result.is_ok());
        assert_eq!(library.len(), initial_count + 1);
    }

    #[test]
    fn test_skill_triggers() {
        let skill =
            Skill::new("test", "Test", SkillCategory::EntryTiming).with_trigger(SkillTrigger::new(
                "momentum",
                TriggerType::Momentum,
                0.5,
                Comparison::GreaterThan,
            ));

        let mut values = HashMap::new();
        values.insert(TriggerType::Momentum, 0.6);

        assert!(skill.check_triggers(&values));

        values.insert(TriggerType::Momentum, 0.4);
        assert!(!skill.check_triggers(&values));
    }

    #[test]
    fn test_find_triggered_skills() {
        let library = SkillLibrary::new();

        let mut values = HashMap::new();
        values.insert(TriggerType::Momentum, 0.6);
        values.insert(TriggerType::Volume, 2.0);

        let triggered = library.find_triggered_skills(MarketCondition::TrendingUp, &values);
        assert!(!triggered.is_empty());
    }

    #[test]
    fn test_skill_performance_score() {
        let mut skill = Skill::new("test", "Test", SkillCategory::EntryTiming);

        // Initial score
        let initial = skill.performance_score();

        // Record some successes
        for _ in 0..10 {
            skill.record_usage(true, 0.05, MarketCondition::Any);
        }

        // Score should improve
        assert!(skill.performance_score() > initial);
    }

    #[test]
    fn test_condition_stats() {
        let mut skill = Skill::new("test", "Test", SkillCategory::EntryTiming);

        skill.record_usage(true, 0.05, MarketCondition::TrendingUp);
        skill.record_usage(true, 0.03, MarketCondition::TrendingUp);
        skill.record_usage(false, -0.02, MarketCondition::Ranging);

        assert!(
            skill
                .condition_performance
                .contains_key(&MarketCondition::TrendingUp)
        );
        assert!(
            skill
                .condition_performance
                .contains_key(&MarketCondition::Ranging)
        );

        let trending_stats = skill
            .condition_performance
            .get(&MarketCondition::TrendingUp)
            .unwrap();
        assert_eq!(trending_stats.usage_count, 2);
        assert_eq!(trending_stats.success_count, 2);
    }

    #[test]
    fn test_skill_builder() {
        let skill = Skill::new("test", "Test Skill", SkillCategory::OrderExecution)
            .with_description("A test skill")
            .with_complexity(SkillComplexity::Advanced)
            .with_condition(MarketCondition::HighVolatility)
            .with_parameter("threshold", 0.5)
            .as_builtin();

        assert_eq!(skill.description, "A test skill");
        assert_eq!(skill.complexity, SkillComplexity::Advanced);
        assert!(skill.conditions.contains(&MarketCondition::HighVolatility));
        assert!(skill.builtin);
        assert_eq!(skill.parameters.get("threshold"), Some(&0.5));
    }

    #[test]
    fn test_trigger_evaluate() {
        let trigger = SkillTrigger::new("test", TriggerType::Price, 100.0, Comparison::GreaterThan);

        assert!(trigger.evaluate(101.0));
        assert!(!trigger.evaluate(99.0));
        assert!(!trigger.evaluate(100.0));

        let trigger_eq = SkillTrigger::new("test", TriggerType::Price, 100.0, Comparison::Equal);
        assert!(trigger_eq.evaluate(100.0));
        assert!(!trigger_eq.evaluate(100.001));
    }

    #[test]
    fn test_create_combination() {
        let mut library = SkillLibrary::new();

        let result = library.create_combination(vec![
            "momentum_entry".to_string(),
            "trailing_exit".to_string(),
        ]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_prune_skills() {
        let mut library = SkillLibrary::new();

        // Learn a skill
        let skill_id = library
            .learn_skill(
                "Bad Skill",
                SkillCategory::EntryTiming,
                SkillComplexity::Basic,
            )
            .unwrap();

        // Record many failures
        for _ in 0..25 {
            library
                .record_usage(
                    &skill_id,
                    false,
                    -0.01,
                    MarketCondition::Any,
                    HashMap::new(),
                )
                .unwrap();
        }

        // Prune
        library.prune_skills();

        // Skill should be removed
        assert!(library.get_skill(&skill_id).is_none());
    }

    #[test]
    fn test_recent_usage() {
        let mut library = SkillLibrary::new();

        for _ in 0..5 {
            library
                .record_usage(
                    "momentum_entry",
                    true,
                    0.01,
                    MarketCondition::Any,
                    HashMap::new(),
                )
                .unwrap();
        }

        let recent = library.recent_usage("momentum_entry", 3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_complexity_unlocking() {
        let library = SkillLibrary::new();

        // With basic complexity, should not see advanced skills
        let applicable = library.get_applicable(MarketCondition::Any);
        assert!(
            applicable
                .iter()
                .all(|s| s.complexity <= SkillComplexity::Basic)
        );
    }

    #[test]
    fn test_skill_applicability() {
        let skill = Skill::new("test", "Test", SkillCategory::EntryTiming)
            .with_condition(MarketCondition::TrendingUp)
            .with_condition(MarketCondition::TrendingDown);

        assert!(skill.is_applicable(MarketCondition::TrendingUp));
        assert!(skill.is_applicable(MarketCondition::TrendingDown));
        assert!(!skill.is_applicable(MarketCondition::Ranging));

        // Skill with Any condition should match everything
        let any_skill = Skill::new("test2", "Test 2", SkillCategory::EntryTiming);
        assert!(any_skill.is_applicable(MarketCondition::Ranging));
    }
}
