//! Procedural memory (how to trade)
//!
//! Part of the Hippocampus region
//! Component: worker
//!
//! This module implements procedural memory - the "how to trade" knowledge that
//! encodes learned sequences of actions, motor patterns, and execution routines.
//! It stores and retrieves procedural knowledge about order execution, timing,
//! and trading workflows.

use crate::common::{Error, Result};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Type of procedural memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProcedureType {
    /// Order entry sequence
    OrderEntry,
    /// Order exit sequence
    OrderExit,
    /// Position scaling procedure
    PositionScaling,
    /// Risk management routine
    RiskManagement,
    /// Stop adjustment procedure
    StopAdjustment,
    /// Take profit adjustment
    TakeProfitAdjustment,
    /// Hedging procedure
    Hedging,
    /// Rebalancing routine
    Rebalancing,
    /// Emergency exit procedure
    EmergencyExit,
    /// Market making routine
    MarketMaking,
}

impl Default for ProcedureType {
    fn default() -> Self {
        ProcedureType::OrderEntry
    }
}

impl ProcedureType {
    /// Get the typical duration for this procedure type
    pub fn typical_duration_ms(&self) -> u64 {
        match self {
            ProcedureType::OrderEntry => 500,
            ProcedureType::OrderExit => 300,
            ProcedureType::PositionScaling => 1000,
            ProcedureType::RiskManagement => 200,
            ProcedureType::StopAdjustment => 150,
            ProcedureType::TakeProfitAdjustment => 150,
            ProcedureType::Hedging => 2000,
            ProcedureType::Rebalancing => 5000,
            ProcedureType::EmergencyExit => 100,
            ProcedureType::MarketMaking => 50,
        }
    }

    /// Get priority level (higher = more urgent)
    pub fn priority(&self) -> u8 {
        match self {
            ProcedureType::EmergencyExit => 10,
            ProcedureType::RiskManagement => 9,
            ProcedureType::StopAdjustment => 7,
            ProcedureType::OrderExit => 6,
            ProcedureType::OrderEntry => 5,
            ProcedureType::TakeProfitAdjustment => 4,
            ProcedureType::PositionScaling => 3,
            ProcedureType::Hedging => 3,
            ProcedureType::Rebalancing => 2,
            ProcedureType::MarketMaking => 1,
        }
    }
}

/// Execution context for procedures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionContext {
    /// Normal market conditions
    Normal,
    /// High volatility environment
    HighVolatility,
    /// Low liquidity conditions
    LowLiquidity,
    /// Trending market
    Trending,
    /// Range-bound market
    Ranging,
    /// News event period
    NewsEvent,
    /// Market open/close
    MarketTransition,
    /// After hours trading
    AfterHours,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        ExecutionContext::Normal
    }
}

/// A single step in a procedure
#[derive(Debug, Clone)]
pub struct ProcedureStep {
    /// Step sequence number
    pub sequence: u32,
    /// Step name/description
    pub name: String,
    /// Action to perform
    pub action: StepAction,
    /// Preconditions that must be met
    pub preconditions: Vec<Condition>,
    /// Expected duration in milliseconds
    pub expected_duration_ms: u64,
    /// Whether this step is critical (must succeed)
    pub critical: bool,
    /// Retry count on failure
    pub max_retries: u8,
    /// Fallback step on failure
    pub fallback: Option<Box<ProcedureStep>>,
}

impl ProcedureStep {
    /// Create a new procedure step
    pub fn new(sequence: u32, name: &str, action: StepAction) -> Self {
        Self {
            sequence,
            name: name.to_string(),
            action,
            preconditions: Vec::new(),
            expected_duration_ms: 100,
            critical: false,
            max_retries: 3,
            fallback: None,
        }
    }

    /// Mark step as critical
    pub fn critical(mut self) -> Self {
        self.critical = true;
        self
    }

    /// Add precondition
    pub fn with_precondition(mut self, condition: Condition) -> Self {
        self.preconditions.push(condition);
        self
    }

    /// Set expected duration
    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.expected_duration_ms = duration_ms;
        self
    }

    /// Set max retries
    pub fn with_retries(mut self, retries: u8) -> Self {
        self.max_retries = retries;
        self
    }

    /// Set fallback step
    pub fn with_fallback(mut self, fallback: ProcedureStep) -> Self {
        self.fallback = Some(Box::new(fallback));
        self
    }
}

/// Actions that can be performed in a procedure step
#[derive(Debug, Clone, PartialEq)]
pub enum StepAction {
    /// Check market conditions
    CheckMarket {
        symbol: String,
        checks: Vec<MarketCheck>,
    },
    /// Place an order
    PlaceOrder {
        order_type: OrderActionType,
        size_pct: f64,
        price_offset: Option<f64>,
    },
    /// Modify existing order
    ModifyOrder { modification: OrderModification },
    /// Cancel order(s)
    CancelOrders { scope: CancelScope },
    /// Wait for condition
    WaitFor {
        condition: Condition,
        timeout_ms: u64,
    },
    /// Adjust position
    AdjustPosition { adjustment: PositionAdjustment },
    /// Update stops/limits
    UpdateStops {
        stop_type: StopType,
        new_level: PriceLevel,
    },
    /// Log/record information
    Record { data_type: RecordType },
    /// Notify/alert
    Notify {
        alert_level: AlertLevel,
        message: String,
    },
    /// Custom action (extensibility)
    Custom {
        action_id: String,
        params: HashMap<String, String>,
    },
}

/// Order action types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderActionType {
    MarketBuy,
    MarketSell,
    LimitBuy,
    LimitSell,
    StopBuy,
    StopSell,
    StopLimitBuy,
    StopLimitSell,
}

/// Order modification types
#[derive(Debug, Clone, PartialEq)]
pub enum OrderModification {
    ChangePrice(f64),
    ChangeSize(f64),
    ChangeType(OrderActionType),
    ChangeTIF(TimeInForce),
}

/// Time in force options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    GTC,
    IOC,
    FOK,
    Day,
    GTD(u64),
}

/// Cancel scope
#[derive(Debug, Clone, PartialEq)]
pub enum CancelScope {
    Single(String),
    AllForSymbol(String),
    AllBuys,
    AllSells,
    All,
}

/// Position adjustment types
#[derive(Debug, Clone, PartialEq)]
pub enum PositionAdjustment {
    ScaleIn(f64),
    ScaleOut(f64),
    Flatten,
    Reverse,
    HedgeWith(String),
}

/// Stop types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopType {
    StopLoss,
    TrailingStop,
    TakeProfit,
    BreakevenStop,
}

/// Price level specification
#[derive(Debug, Clone, PartialEq)]
pub enum PriceLevel {
    Absolute(f64),
    RelativeOffset(f64),
    PercentOffset(f64),
    ATRMultiple(f64),
    SwingLevel,
}

/// Record types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordType {
    StateSnapshot,
    Decision,
    Execution,
    Error,
    Performance,
}

/// Alert levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

/// Market checks
#[derive(Debug, Clone, PartialEq)]
pub enum MarketCheck {
    SpreadBelow(f64),
    VolumeAbove(f64),
    VolatilityBelow(f64),
    LiquidityAbove(f64),
    NoHalt,
    WithinTradingHours,
}

/// Conditions for preconditions and wait steps
#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    /// Order is filled
    OrderFilled(String),
    /// Order is partially filled
    OrderPartiallyFilled(String),
    /// Position exists
    HasPosition(String),
    /// No open orders
    NoOpenOrders(String),
    /// Price crosses level
    PriceCrossed(f64),
    /// Time elapsed
    TimeElapsed(u64),
    /// Market condition met
    MarketCondition(MarketCheck),
    /// Custom condition
    Custom(String),
    /// All conditions met
    All(Vec<Condition>),
    /// Any condition met
    Any(Vec<Condition>),
}

/// A complete procedure definition
#[derive(Debug, Clone)]
pub struct Procedure {
    /// Unique procedure ID
    pub id: String,
    /// Procedure type
    pub procedure_type: ProcedureType,
    /// Human-readable name
    pub name: String,
    /// Description
    pub description: String,
    /// Applicable contexts
    pub contexts: Vec<ExecutionContext>,
    /// Ordered list of steps
    pub steps: Vec<ProcedureStep>,
    /// Parameters that can be customized
    pub parameters: HashMap<String, f64>,
    /// Success rate from past executions
    pub success_rate: f64,
    /// Average execution time in milliseconds
    pub avg_execution_time: u64,
    /// Number of times executed
    pub execution_count: u64,
    /// Last execution timestamp
    pub last_executed: Option<u64>,
    /// Version number
    pub version: u32,
    /// Whether this is a learned procedure
    pub learned: bool,
}

impl Procedure {
    /// Create a new procedure
    pub fn new(id: &str, procedure_type: ProcedureType, name: &str) -> Self {
        Self {
            id: id.to_string(),
            procedure_type,
            name: name.to_string(),
            description: String::new(),
            contexts: vec![ExecutionContext::Normal],
            steps: Vec::new(),
            parameters: HashMap::new(),
            success_rate: 1.0,
            avg_execution_time: procedure_type.typical_duration_ms(),
            execution_count: 0,
            last_executed: None,
            version: 1,
            learned: false,
        }
    }

    /// Add a step
    pub fn add_step(&mut self, step: ProcedureStep) {
        self.steps.push(step);
    }

    /// Set description
    pub fn with_description(mut self, description: &str) -> Self {
        self.description = description.to_string();
        self
    }

    /// Add applicable context
    pub fn with_context(mut self, context: ExecutionContext) -> Self {
        if !self.contexts.contains(&context) {
            self.contexts.push(context);
        }
        self
    }

    /// Set parameter
    pub fn with_parameter(mut self, key: &str, value: f64) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    /// Mark as learned
    pub fn as_learned(mut self) -> Self {
        self.learned = true;
        self
    }

    /// Record execution result
    pub fn record_execution(&mut self, success: bool, duration_ms: u64) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        self.last_executed = Some(now);
        self.execution_count += 1;

        // Update success rate with exponential moving average
        let alpha = 0.1;
        let success_val = if success { 1.0 } else { 0.0 };
        self.success_rate = alpha * success_val + (1.0 - alpha) * self.success_rate;

        // Update average execution time
        let n = self.execution_count as f64;
        self.avg_execution_time =
            ((self.avg_execution_time as f64 * (n - 1.0) + duration_ms as f64) / n) as u64;
    }

    /// Get estimated duration
    pub fn estimated_duration(&self) -> u64 {
        if self.execution_count > 10 {
            self.avg_execution_time
        } else {
            self.steps.iter().map(|s| s.expected_duration_ms).sum()
        }
    }

    /// Check if procedure is applicable in context
    pub fn is_applicable(&self, context: ExecutionContext) -> bool {
        self.contexts.contains(&context) || self.contexts.contains(&ExecutionContext::Normal)
    }
}

/// Execution state for a running procedure
#[derive(Debug, Clone)]
pub struct ProcedureExecution {
    /// Execution ID
    pub id: String,
    /// Procedure being executed
    pub procedure_id: String,
    /// Current step index
    pub current_step: usize,
    /// Step retry count
    pub retry_count: u8,
    /// Start timestamp
    pub start_time: u64,
    /// Last step completion time
    pub last_step_time: Option<u64>,
    /// Execution status
    pub status: ExecutionStatus,
    /// Step results
    pub step_results: Vec<StepResult>,
    /// Parameters for this execution
    pub parameters: HashMap<String, f64>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStatus {
    Pending,
    Running,
    WaitingForCondition,
    Completed,
    Failed,
    Cancelled,
    TimedOut,
}

impl Default for ExecutionStatus {
    fn default() -> Self {
        ExecutionStatus::Pending
    }
}

/// Result of a step execution
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Step sequence number
    pub sequence: u32,
    /// Whether step succeeded
    pub success: bool,
    /// Execution time in milliseconds
    pub duration_ms: u64,
    /// Number of retries used
    pub retries: u8,
    /// Output/result data
    pub output: Option<String>,
    /// Error if failed
    pub error: Option<String>,
}

impl ProcedureExecution {
    /// Create new execution
    pub fn new(procedure_id: &str) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static EXEC_COUNTER: AtomicU64 = AtomicU64::new(0);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        let counter = EXEC_COUNTER.fetch_add(1, Ordering::SeqCst);

        Self {
            id: format!("exec_{}_{}_{}", procedure_id, now, counter),
            procedure_id: procedure_id.to_string(),
            current_step: 0,
            retry_count: 0,
            start_time: now,
            last_step_time: None,
            status: ExecutionStatus::Pending,
            step_results: Vec::new(),
            parameters: HashMap::new(),
            error: None,
        }
    }

    /// Start execution
    pub fn start(&mut self) {
        self.status = ExecutionStatus::Running;
    }

    /// Record step completion
    pub fn complete_step(&mut self, result: StepResult) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;

        self.step_results.push(result);
        self.last_step_time = Some(now);
        self.current_step += 1;
        self.retry_count = 0;
    }

    /// Record step failure
    pub fn fail_step(&mut self, error: &str, can_retry: bool) {
        if can_retry {
            self.retry_count += 1;
        } else {
            self.status = ExecutionStatus::Failed;
            self.error = Some(error.to_string());
        }
    }

    /// Complete execution
    pub fn complete(&mut self) {
        self.status = ExecutionStatus::Completed;
    }

    /// Fail execution
    pub fn fail(&mut self, error: &str) {
        self.status = ExecutionStatus::Failed;
        self.error = Some(error.to_string());
    }

    /// Cancel execution
    pub fn cancel(&mut self) {
        self.status = ExecutionStatus::Cancelled;
    }

    /// Get elapsed time
    pub fn elapsed_ms(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_millis() as u64;
        now.saturating_sub(self.start_time)
    }

    /// Check if execution is active
    pub fn is_active(&self) -> bool {
        matches!(
            self.status,
            ExecutionStatus::Running | ExecutionStatus::WaitingForCondition
        )
    }

    /// Get success rate of steps
    pub fn step_success_rate(&self) -> f64 {
        if self.step_results.is_empty() {
            return 1.0;
        }
        let successes = self.step_results.iter().filter(|r| r.success).count();
        successes as f64 / self.step_results.len() as f64
    }
}

/// Configuration for procedural memory
#[derive(Debug, Clone)]
pub struct ProceduralConfig {
    /// Maximum procedures to store
    pub max_procedures: usize,
    /// Maximum execution history per procedure
    pub max_history: usize,
    /// Enable procedure learning
    pub enable_learning: bool,
    /// Minimum success rate to keep learned procedure
    pub min_success_rate: f64,
    /// Enable procedure adaptation
    pub enable_adaptation: bool,
    /// Maximum concurrent executions
    pub max_concurrent: usize,
    /// Default timeout in milliseconds
    pub default_timeout_ms: u64,
}

impl Default for ProceduralConfig {
    fn default() -> Self {
        Self {
            max_procedures: 1000,
            max_history: 100,
            enable_learning: true,
            min_success_rate: 0.6,
            enable_adaptation: true,
            max_concurrent: 10,
            default_timeout_ms: 30000,
        }
    }
}

/// Procedural memory (how to trade)
pub struct Procedural {
    /// Configuration
    config: ProceduralConfig,
    /// Stored procedures indexed by ID
    procedures: HashMap<String, Procedure>,
    /// Procedures indexed by type
    by_type: HashMap<ProcedureType, Vec<String>>,
    /// Active executions
    active_executions: HashMap<String, ProcedureExecution>,
    /// Execution history
    execution_history: VecDeque<ProcedureExecution>,
    /// Procedure counter for ID generation
    procedure_counter: u64,
    /// Current execution context
    current_context: ExecutionContext,
    /// Statistics
    stats: ProceduralStats,
}

/// Statistics for procedural memory
#[derive(Debug, Clone, Default)]
pub struct ProceduralStats {
    pub total_procedures: usize,
    pub learned_procedures: usize,
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub avg_execution_time: f64,
    pub executions_by_type: HashMap<ProcedureType, u64>,
}

impl Default for Procedural {
    fn default() -> Self {
        Self::new()
    }
}

impl Procedural {
    /// Create a new instance
    pub fn new() -> Self {
        let mut procedural = Self {
            config: ProceduralConfig::default(),
            procedures: HashMap::new(),
            by_type: HashMap::new(),
            active_executions: HashMap::new(),
            execution_history: VecDeque::new(),
            procedure_counter: 0,
            current_context: ExecutionContext::Normal,
            stats: ProceduralStats::default(),
        };

        // Register default procedures
        procedural.register_default_procedures();

        procedural
    }

    /// Create with custom configuration
    pub fn with_config(config: ProceduralConfig) -> Self {
        let mut procedural = Self {
            config,
            procedures: HashMap::new(),
            by_type: HashMap::new(),
            active_executions: HashMap::new(),
            execution_history: VecDeque::new(),
            procedure_counter: 0,
            current_context: ExecutionContext::Normal,
            stats: ProceduralStats::default(),
        };

        procedural.register_default_procedures();

        procedural
    }

    /// Register default built-in procedures
    fn register_default_procedures(&mut self) {
        // Simple market entry procedure
        let mut market_entry = Procedure::new(
            "default_market_entry",
            ProcedureType::OrderEntry,
            "Market Entry",
        );
        market_entry.description = "Simple market order entry".to_string();
        market_entry.add_step(
            ProcedureStep::new(
                1,
                "Check market conditions",
                StepAction::CheckMarket {
                    symbol: "{{symbol}}".to_string(),
                    checks: vec![MarketCheck::NoHalt, MarketCheck::WithinTradingHours],
                },
            )
            .critical(),
        );
        market_entry.add_step(
            ProcedureStep::new(
                2,
                "Place market order",
                StepAction::PlaceOrder {
                    order_type: OrderActionType::MarketBuy,
                    size_pct: 1.0,
                    price_offset: None,
                },
            )
            .critical()
            .with_retries(3),
        );
        market_entry.add_step(ProcedureStep::new(
            3,
            "Record execution",
            StepAction::Record {
                data_type: RecordType::Execution,
            },
        ));
        self.register_procedure(market_entry);

        // Limit entry with checks
        let mut limit_entry = Procedure::new(
            "default_limit_entry",
            ProcedureType::OrderEntry,
            "Limit Entry",
        );
        limit_entry.description = "Limit order entry with spread check".to_string();
        limit_entry.add_step(
            ProcedureStep::new(
                1,
                "Check spread",
                StepAction::CheckMarket {
                    symbol: "{{symbol}}".to_string(),
                    checks: vec![MarketCheck::SpreadBelow(0.05), MarketCheck::NoHalt],
                },
            )
            .critical(),
        );
        limit_entry.add_step(
            ProcedureStep::new(
                2,
                "Place limit order",
                StepAction::PlaceOrder {
                    order_type: OrderActionType::LimitBuy,
                    size_pct: 1.0,
                    price_offset: Some(-0.001), // Slightly below mid
                },
            )
            .with_retries(2),
        );
        limit_entry.add_step(ProcedureStep::new(
            3,
            "Wait for fill",
            StepAction::WaitFor {
                condition: Condition::OrderFilled("{{order_id}}".to_string()),
                timeout_ms: 5000,
            },
        ));
        self.register_procedure(limit_entry);

        // Emergency exit procedure
        let mut emergency_exit = Procedure::new(
            "default_emergency_exit",
            ProcedureType::EmergencyExit,
            "Emergency Exit",
        );
        emergency_exit.description = "Immediate position liquidation".to_string();
        emergency_exit.contexts = vec![
            ExecutionContext::Normal,
            ExecutionContext::HighVolatility,
            ExecutionContext::NewsEvent,
        ];
        emergency_exit.add_step(
            ProcedureStep::new(
                1,
                "Cancel all orders",
                StepAction::CancelOrders {
                    scope: CancelScope::All,
                },
            )
            .critical()
            .with_duration(50),
        );
        emergency_exit.add_step(
            ProcedureStep::new(
                2,
                "Market exit all",
                StepAction::AdjustPosition {
                    adjustment: PositionAdjustment::Flatten,
                },
            )
            .critical()
            .with_duration(100),
        );
        emergency_exit.add_step(ProcedureStep::new(
            3,
            "Alert",
            StepAction::Notify {
                alert_level: AlertLevel::Critical,
                message: "Emergency exit executed".to_string(),
            },
        ));
        self.register_procedure(emergency_exit);

        // Scale-in procedure
        let mut scale_in = Procedure::new(
            "default_scale_in",
            ProcedureType::PositionScaling,
            "Scale In",
        );
        scale_in.description = "Add to position on favorable move".to_string();
        scale_in.add_step(
            ProcedureStep::new(
                1,
                "Verify position exists",
                StepAction::CheckMarket {
                    symbol: "{{symbol}}".to_string(),
                    checks: vec![MarketCheck::NoHalt],
                },
            )
            .with_precondition(Condition::HasPosition("{{symbol}}".to_string())),
        );
        scale_in.add_step(ProcedureStep::new(
            2,
            "Place scale order",
            StepAction::PlaceOrder {
                order_type: OrderActionType::LimitBuy,
                size_pct: 0.5,             // Half of original size
                price_offset: Some(0.002), // Slightly above current
            },
        ));
        scale_in.parameters.insert("scale_factor".to_string(), 0.5);
        self.register_procedure(scale_in);

        // Stop adjustment procedure
        let mut trailing_stop = Procedure::new(
            "default_trailing_stop",
            ProcedureType::StopAdjustment,
            "Trailing Stop Update",
        );
        trailing_stop.description = "Update trailing stop to lock in profits".to_string();
        trailing_stop.add_step(
            ProcedureStep::new(
                1,
                "Check position",
                StepAction::CheckMarket {
                    symbol: "{{symbol}}".to_string(),
                    checks: vec![],
                },
            )
            .with_precondition(Condition::HasPosition("{{symbol}}".to_string())),
        );
        trailing_stop.add_step(ProcedureStep::new(
            2,
            "Update stop",
            StepAction::UpdateStops {
                stop_type: StopType::TrailingStop,
                new_level: PriceLevel::ATRMultiple(2.0),
            },
        ));
        self.register_procedure(trailing_stop);
    }

    /// Register a procedure
    pub fn register_procedure(&mut self, procedure: Procedure) {
        let id = procedure.id.clone();
        let proc_type = procedure.procedure_type;

        self.procedures.insert(id.clone(), procedure);
        self.by_type
            .entry(proc_type)
            .or_insert_with(Vec::new)
            .push(id);

        self.update_stats();
    }

    /// Get procedure by ID
    pub fn get_procedure(&self, id: &str) -> Option<&Procedure> {
        self.procedures.get(id)
    }

    /// Get mutable procedure by ID
    pub fn get_procedure_mut(&mut self, id: &str) -> Option<&mut Procedure> {
        self.procedures.get_mut(id)
    }

    /// Get procedures by type
    pub fn get_by_type(&self, proc_type: ProcedureType) -> Vec<&Procedure> {
        self.by_type
            .get(&proc_type)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.procedures.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get best procedure for context
    pub fn get_best_procedure(
        &self,
        proc_type: ProcedureType,
        context: ExecutionContext,
    ) -> Option<&Procedure> {
        let candidates = self.get_by_type(proc_type);

        candidates
            .into_iter()
            .filter(|p| p.is_applicable(context))
            .max_by(|a, b| {
                // Prioritize by success rate, then by execution count
                let score_a = a.success_rate * 100.0 + (a.execution_count as f64).ln();
                let score_b = b.success_rate * 100.0 + (b.execution_count as f64).ln();
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Start executing a procedure
    pub fn start_execution(
        &mut self,
        procedure_id: &str,
        parameters: HashMap<String, f64>,
    ) -> Result<String> {
        // Check if procedure exists
        if !self.procedures.contains_key(procedure_id) {
            return Err(Error::NotFound(format!(
                "Procedure {} not found",
                procedure_id
            )));
        }

        // Check concurrent execution limit
        if self.active_executions.len() >= self.config.max_concurrent {
            return Err(Error::InvalidState(
                "Maximum concurrent executions reached".to_string(),
            ));
        }

        // Create execution
        let mut execution = ProcedureExecution::new(procedure_id);
        execution.parameters = parameters;
        execution.start();

        let exec_id = execution.id.clone();
        self.active_executions.insert(exec_id.clone(), execution);

        self.stats.total_executions += 1;
        *self
            .stats
            .executions_by_type
            .entry(self.procedures[procedure_id].procedure_type)
            .or_insert(0) += 1;

        Ok(exec_id)
    }

    /// Get active execution
    pub fn get_execution(&self, exec_id: &str) -> Option<&ProcedureExecution> {
        self.active_executions.get(exec_id)
    }

    /// Get active execution mutably
    pub fn get_execution_mut(&mut self, exec_id: &str) -> Option<&mut ProcedureExecution> {
        self.active_executions.get_mut(exec_id)
    }

    /// Advance execution to next step
    pub fn advance_execution(&mut self, exec_id: &str, step_result: StepResult) -> Result<bool> {
        let execution = self
            .active_executions
            .get_mut(exec_id)
            .ok_or_else(|| Error::NotFound(format!("Execution {} not found", exec_id)))?;

        let procedure = self
            .procedures
            .get(&execution.procedure_id)
            .ok_or_else(|| Error::NotFound("Procedure not found".to_string()))?;

        // Record step result
        execution.complete_step(step_result);

        // Check if complete
        if execution.current_step >= procedure.steps.len() {
            execution.complete();
            return Ok(true);
        }

        Ok(false)
    }

    /// Complete an execution
    pub fn complete_execution(&mut self, exec_id: &str, success: bool) -> Result<()> {
        let execution = self
            .active_executions
            .remove(exec_id)
            .ok_or_else(|| Error::NotFound(format!("Execution {} not found", exec_id)))?;

        let duration = execution.elapsed_ms();

        // Update procedure statistics
        if let Some(procedure) = self.procedures.get_mut(&execution.procedure_id) {
            procedure.record_execution(success, duration);
        }

        // Update global statistics
        if success {
            self.stats.successful_executions += 1;
        } else {
            self.stats.failed_executions += 1;
        }

        // Update average execution time
        let total = self.stats.successful_executions + self.stats.failed_executions;
        if total > 0 {
            self.stats.avg_execution_time = (self.stats.avg_execution_time * (total - 1) as f64
                + duration as f64)
                / total as f64;
        }

        // Store in history
        self.execution_history.push_front(execution);
        while self.execution_history.len() > self.config.max_history {
            self.execution_history.pop_back();
        }

        Ok(())
    }

    /// Cancel an execution
    pub fn cancel_execution(&mut self, exec_id: &str) -> Result<()> {
        let mut execution = self
            .active_executions
            .remove(exec_id)
            .ok_or_else(|| Error::NotFound(format!("Execution {} not found", exec_id)))?;

        execution.cancel();
        self.execution_history.push_front(execution);

        Ok(())
    }

    /// Set current execution context
    pub fn set_context(&mut self, context: ExecutionContext) {
        self.current_context = context;
    }

    /// Get current context
    pub fn current_context(&self) -> ExecutionContext {
        self.current_context
    }

    /// Learn a new procedure from execution history
    pub fn learn_procedure(
        &mut self,
        name: &str,
        proc_type: ProcedureType,
        steps: Vec<ProcedureStep>,
    ) -> Result<String> {
        if !self.config.enable_learning {
            return Err(Error::InvalidState("Learning is disabled".to_string()));
        }

        self.procedure_counter += 1;
        let id = format!("learned_{}_{}", proc_type as u8, self.procedure_counter);

        let mut procedure = Procedure::new(&id, proc_type, name);
        procedure.steps = steps;
        procedure.learned = true;

        self.register_procedure(procedure);

        Ok(id)
    }

    /// Adapt a procedure based on execution results
    pub fn adapt_procedure(&mut self, procedure_id: &str) -> Result<()> {
        if !self.config.enable_adaptation {
            return Err(Error::InvalidState("Adaptation is disabled".to_string()));
        }

        let procedure = self
            .procedures
            .get_mut(procedure_id)
            .ok_or_else(|| Error::NotFound("Procedure not found".to_string()))?;

        // Only adapt learned procedures
        if !procedure.learned {
            return Err(Error::InvalidArgument(
                "Cannot adapt built-in procedures".to_string(),
            ));
        }

        // Increment version
        procedure.version += 1;

        // Could implement more sophisticated adaptation logic here
        // For now, just adjust timing estimates based on actual performance

        Ok(())
    }

    /// Prune underperforming learned procedures
    pub fn prune_procedures(&mut self) {
        let min_rate = self.config.min_success_rate;
        let min_executions = 10;

        let to_remove: Vec<String> = self
            .procedures
            .iter()
            .filter(|(_, p)| {
                p.learned && p.execution_count >= min_executions && p.success_rate < min_rate
            })
            .map(|(id, _)| id.clone())
            .collect();

        for id in to_remove {
            if let Some(proc) = self.procedures.remove(&id) {
                if let Some(ids) = self.by_type.get_mut(&proc.procedure_type) {
                    ids.retain(|i| i != &id);
                }
            }
        }

        self.update_stats();
    }

    /// Get all active executions
    pub fn active_executions(&self) -> &HashMap<String, ProcedureExecution> {
        &self.active_executions
    }

    /// Get execution history
    pub fn execution_history(&self) -> &VecDeque<ProcedureExecution> {
        &self.execution_history
    }

    /// Get statistics
    pub fn stats(&self) -> &ProceduralStats {
        &self.stats
    }

    /// Update statistics
    fn update_stats(&mut self) {
        self.stats.total_procedures = self.procedures.len();
        self.stats.learned_procedures = self.procedures.values().filter(|p| p.learned).count();
    }

    /// Get number of procedures
    pub fn len(&self) -> usize {
        self.procedures.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.procedures.is_empty()
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing logic - check for timed out executions, etc.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = Procedural::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_default_procedures() {
        let procedural = Procedural::new();
        assert!(!procedural.is_empty());
        assert!(procedural.len() >= 5);

        // Check default procedures exist
        assert!(procedural.get_procedure("default_market_entry").is_some());
        assert!(procedural.get_procedure("default_emergency_exit").is_some());
    }

    #[test]
    fn test_get_by_type() {
        let procedural = Procedural::new();

        let entry_procs = procedural.get_by_type(ProcedureType::OrderEntry);
        assert!(!entry_procs.is_empty());

        let emergency_procs = procedural.get_by_type(ProcedureType::EmergencyExit);
        assert!(!emergency_procs.is_empty());
    }

    #[test]
    fn test_start_execution() {
        let mut procedural = Procedural::new();

        let result = procedural.start_execution("default_market_entry", HashMap::new());
        assert!(result.is_ok());

        let exec_id = result.unwrap();
        let execution = procedural.get_execution(&exec_id);
        assert!(execution.is_some());
        assert!(execution.unwrap().is_active());
    }

    #[test]
    fn test_complete_execution() {
        let mut procedural = Procedural::new();

        let exec_id = procedural
            .start_execution("default_market_entry", HashMap::new())
            .unwrap();

        // Complete it
        let result = procedural.complete_execution(&exec_id, true);
        assert!(result.is_ok());

        // Should be in history now
        assert!(!procedural.execution_history().is_empty());
        assert_eq!(procedural.stats().successful_executions, 1);
    }

    #[test]
    fn test_cancel_execution() {
        let mut procedural = Procedural::new();

        let exec_id = procedural
            .start_execution("default_market_entry", HashMap::new())
            .unwrap();

        let result = procedural.cancel_execution(&exec_id);
        assert!(result.is_ok());

        // Should not be active
        assert!(procedural.get_execution(&exec_id).is_none());
    }

    #[test]
    fn test_execution_not_found() {
        let mut procedural = Procedural::new();

        let result = procedural.complete_execution("nonexistent", true);
        assert!(result.is_err());
    }

    #[test]
    fn test_procedure_not_found() {
        let mut procedural = Procedural::new();

        let result = procedural.start_execution("nonexistent", HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_learn_procedure() {
        let mut procedural = Procedural::new();
        let initial_count = procedural.len();

        let steps = vec![ProcedureStep::new(
            1,
            "Test step",
            StepAction::Record {
                data_type: RecordType::StateSnapshot,
            },
        )];

        let result = procedural.learn_procedure("Test Procedure", ProcedureType::OrderEntry, steps);
        assert!(result.is_ok());
        assert_eq!(procedural.len(), initial_count + 1);

        let proc = procedural.get_procedure(&result.unwrap());
        assert!(proc.is_some());
        assert!(proc.unwrap().learned);
    }

    #[test]
    fn test_procedure_execution_flow() {
        let mut procedural = Procedural::new();

        // Start execution
        let exec_id = procedural
            .start_execution("default_market_entry", HashMap::new())
            .unwrap();

        // Advance through steps
        let step1 = StepResult {
            sequence: 1,
            success: true,
            duration_ms: 50,
            retries: 0,
            output: None,
            error: None,
        };
        procedural.advance_execution(&exec_id, step1).unwrap();

        let step2 = StepResult {
            sequence: 2,
            success: true,
            duration_ms: 100,
            retries: 1,
            output: Some("Order placed".to_string()),
            error: None,
        };
        procedural.advance_execution(&exec_id, step2).unwrap();

        let step3 = StepResult {
            sequence: 3,
            success: true,
            duration_ms: 10,
            retries: 0,
            output: None,
            error: None,
        };
        let complete = procedural.advance_execution(&exec_id, step3).unwrap();

        // Should be complete now
        assert!(complete);
    }

    #[test]
    fn test_get_best_procedure() {
        let procedural = Procedural::new();

        let best =
            procedural.get_best_procedure(ProcedureType::OrderEntry, ExecutionContext::Normal);
        assert!(best.is_some());
    }

    #[test]
    fn test_context_handling() {
        let mut procedural = Procedural::new();

        procedural.set_context(ExecutionContext::HighVolatility);
        assert_eq!(
            procedural.current_context(),
            ExecutionContext::HighVolatility
        );

        // Emergency exit should be applicable in high volatility
        let emergency = procedural.get_procedure("default_emergency_exit").unwrap();
        assert!(emergency.is_applicable(ExecutionContext::HighVolatility));
    }

    #[test]
    fn test_procedure_step_builder() {
        let step = ProcedureStep::new(
            1,
            "Test",
            StepAction::Record {
                data_type: RecordType::Decision,
            },
        )
        .critical()
        .with_duration(200)
        .with_retries(5)
        .with_precondition(Condition::NoOpenOrders("BTC".to_string()));

        assert!(step.critical);
        assert_eq!(step.expected_duration_ms, 200);
        assert_eq!(step.max_retries, 5);
        assert!(!step.preconditions.is_empty());
    }

    #[test]
    fn test_procedure_record_execution() {
        let mut procedure = Procedure::new("test", ProcedureType::OrderEntry, "Test");

        procedure.record_execution(true, 100);
        assert_eq!(procedure.execution_count, 1);

        procedure.record_execution(false, 200);
        assert_eq!(procedure.execution_count, 2);
        assert!(procedure.success_rate < 1.0);
    }

    #[test]
    fn test_execution_elapsed_time() {
        let execution = ProcedureExecution::new("test");
        // Elapsed time should be very small but non-negative
        assert!(execution.elapsed_ms() < 1000);
    }

    #[test]
    fn test_max_concurrent_executions() {
        let mut procedural = Procedural::with_config(ProceduralConfig {
            max_concurrent: 2,
            ..Default::default()
        });

        // Start first two - should succeed
        assert!(
            procedural
                .start_execution("default_market_entry", HashMap::new())
                .is_ok()
        );
        assert!(
            procedural
                .start_execution("default_market_entry", HashMap::new())
                .is_ok()
        );

        // Third should fail
        let result = procedural.start_execution("default_market_entry", HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_procedure_type_properties() {
        assert!(ProcedureType::EmergencyExit.priority() > ProcedureType::OrderEntry.priority());
        assert!(
            ProcedureType::EmergencyExit.typical_duration_ms()
                < ProcedureType::Rebalancing.typical_duration_ms()
        );
    }

    #[test]
    fn test_prune_procedures() {
        let mut procedural = Procedural::new();

        // Learn a bad procedure
        let steps = vec![ProcedureStep::new(
            1,
            "Test",
            StepAction::Record {
                data_type: RecordType::StateSnapshot,
            },
        )];
        let proc_id = procedural
            .learn_procedure("Bad Procedure", ProcedureType::OrderEntry, steps)
            .unwrap();

        // Simulate many failures
        if let Some(proc) = procedural.get_procedure_mut(&proc_id) {
            for _ in 0..15 {
                proc.record_execution(false, 100);
            }
        }

        // Prune should remove it
        procedural.prune_procedures();
        assert!(procedural.get_procedure(&proc_id).is_none());
    }

    #[test]
    fn test_stats() {
        let procedural = Procedural::new();
        let stats = procedural.stats();

        assert!(stats.total_procedures > 0);
        assert_eq!(stats.total_executions, 0);
    }
}
