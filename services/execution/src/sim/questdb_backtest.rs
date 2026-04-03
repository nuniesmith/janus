//! QuestDB-Backed Walk-Forward Backtest Integration
//!
//! This module provides integration between the Walk-Forward optimization framework
//! and QuestDB recorded market data. It enables running backtests and parameter
//! optimization directly on historical data stored in QuestDB.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                    QuestDB Walk-Forward Pipeline                        │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
//! │  │   QuestDB        │───▶│  ReplayEngine    │───▶│  Strategy        │  │
//! │  │   (recorded data)│    │  (time-filtered) │    │  (evaluation)    │  │
//! │  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
//! │                                                           │              │
//! │                                                           ▼              │
//! │  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
//! │  │  Walk-Forward    │◀───│  Optimization    │◀───│  Metrics         │  │
//! │  │  Result          │    │  Runner          │    │  (Sharpe, etc.)  │  │
//! │  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ### Basic Walk-Forward with QuestDB
//!
//! ```rust,ignore
//! use janus_execution::sim::questdb_backtest::{
//!     QuestDBBacktestConfig, QuestDBWalkForwardRunner,
//! };
//! use janus_execution::sim::optimization::{
//!     ParameterRange, OptimizationMetric, WalkForwardConfig,
//! };
//!
//! // Configure QuestDB connection
//! let config = QuestDBBacktestConfig::new("localhost", 9000)
//!     .with_ticks_table("fks_ticks")
//!     .with_trades_table("fks_trades")
//!     .with_symbols(vec!["BTC/USDT".to_string()]);
//!
//! // Create runner with walk-forward configuration
//! let runner = QuestDBWalkForwardRunner::builder()
//!     .questdb_config(config)
//!     .windows(5)
//!     .in_sample_pct(0.7)
//!     .parameter(ParameterRange::int("fast_period", 5, 20, 5))
//!     .parameter(ParameterRange::int("slow_period", 20, 50, 10))
//!     .metric(OptimizationMetric::SharpeRatio)
//!     .build()?;
//!
//! // Run walk-forward optimization
//! let result = runner.run(&my_strategy).await?;
//!
//! if result.is_robust(0.7) {
//!     println!("Strategy passes walk-forward validation!");
//! }
//! ```
//!
//! ### Using the Event-Based Strategy Interface
//!
//! ```rust,ignore
//! use janus_execution::sim::questdb_backtest::{
//!     EventBasedStrategy, BacktestState, SignalType,
//! };
//!
//! struct MyStrategy {
//!     fast_period: usize,
//!     slow_period: usize,
//! }
//!
//! impl EventBasedStrategy for MyStrategy {
//!     fn on_tick(&mut self, tick: &TickData, state: &mut BacktestState) -> Option<SignalType> {
//!         // Update indicators
//!         state.update_price(tick.symbol.clone(), tick.mid_price());
//!
//!         // Generate signals
//!         if self.should_buy(state) {
//!             Some(SignalType::Buy { size: dec!(0.1) })
//!         } else if self.should_sell(state) {
//!             Some(SignalType::Sell { size: dec!(0.1) })
//!         } else {
//!             None
//!         }
//!     }
//!
//!     fn on_trade(&mut self, _trade: &TradeData, _state: &mut BacktestState) -> Option<SignalType> {
//!         None
//!     }
//! }
//! ```

use super::data_feed::{MarketEvent, TickData, TradeData};
use super::optimization::{
    OptimizationConfig, OptimizationDirection, OptimizationError, OptimizationMetric,
    OptimizationRunResult, ParameterRange, ParameterSet, StrategyEvaluator,
    WalkForwardBacktestRunner, WalkForwardConfig, WalkForwardResult,
};
use super::replay::{QuestDBLoaderConfig, ReplayConfig, ReplayEngine, ReplayError};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info};

// ============================================================================
// Errors
// ============================================================================

/// Errors specific to QuestDB backtest operations
#[derive(Debug, Error)]
pub enum QuestDBBacktestError {
    #[error("QuestDB connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Data loading failed: {0}")]
    LoadFailed(String),

    #[error("No data available for time range {0} to {1}")]
    NoDataInRange(DateTime<Utc>, DateTime<Utc>),

    #[error("Optimization error: {0}")]
    Optimization(#[from] OptimizationError),

    #[error("Replay error: {0}")]
    Replay(#[from] ReplayError),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Strategy error: {0}")]
    StrategyError(String),
}

// ============================================================================
// QuestDB Backtest Configuration
// ============================================================================

/// Configuration for QuestDB-backed backtesting
#[derive(Debug, Clone)]
pub struct QuestDBBacktestConfig {
    /// QuestDB host
    pub host: String,
    /// QuestDB HTTP port
    pub port: u16,
    /// Ticks table name
    pub ticks_table: String,
    /// Trades table name
    pub trades_table: String,
    /// Symbols to include
    pub symbols: Option<Vec<String>>,
    /// Exchanges to include
    pub exchanges: Option<Vec<String>>,
    /// Whether to load tick data
    pub load_ticks: bool,
    /// Whether to load trade data
    pub load_trades: bool,
    /// Cache loaded data for reuse across windows
    pub cache_data: bool,
    /// Initial account balance for backtesting
    pub initial_balance: Decimal,
    /// Commission per trade (in basis points)
    pub commission_bps: Decimal,
    /// Slippage estimate (in basis points)
    pub slippage_bps: Decimal,
}

impl Default for QuestDBBacktestConfig {
    fn default() -> Self {
        Self {
            host: "localhost".to_string(),
            port: 9000,
            ticks_table: "fks_ticks".to_string(),
            trades_table: "fks_trades".to_string(),
            symbols: None,
            exchanges: None,
            load_ticks: true,
            load_trades: true,
            cache_data: true,
            initial_balance: Decimal::from(10_000),
            commission_bps: Decimal::new(10, 2), // 0.10%
            slippage_bps: Decimal::new(5, 2),    // 0.05%
        }
    }
}

impl QuestDBBacktestConfig {
    /// Create a new configuration with QuestDB connection details
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            ..Default::default()
        }
    }

    /// Set the ticks table name
    pub fn with_ticks_table(mut self, table: impl Into<String>) -> Self {
        self.ticks_table = table.into();
        self
    }

    /// Set the trades table name
    pub fn with_trades_table(mut self, table: impl Into<String>) -> Self {
        self.trades_table = table.into();
        self
    }

    /// Set symbols to filter
    pub fn with_symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Set exchanges to filter
    pub fn with_exchanges(mut self, exchanges: Vec<String>) -> Self {
        self.exchanges = Some(exchanges);
        self
    }

    /// Enable/disable tick data loading
    pub fn with_ticks(mut self, enabled: bool) -> Self {
        self.load_ticks = enabled;
        self
    }

    /// Enable/disable trade data loading
    pub fn with_trades(mut self, enabled: bool) -> Self {
        self.load_trades = enabled;
        self
    }

    /// Enable/disable data caching
    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_data = enabled;
        self
    }

    /// Set initial account balance
    pub fn with_initial_balance(mut self, balance: Decimal) -> Self {
        self.initial_balance = balance;
        self
    }

    /// Set commission in basis points
    pub fn with_commission_bps(mut self, bps: Decimal) -> Self {
        self.commission_bps = bps;
        self
    }

    /// Set slippage in basis points
    pub fn with_slippage_bps(mut self, bps: Decimal) -> Self {
        self.slippage_bps = bps;
        self
    }

    /// Convert to QuestDB loader config
    pub fn to_loader_config(&self) -> QuestDBLoaderConfig {
        let mut config = QuestDBLoaderConfig::new(&self.host, self.port)
            .with_ticks_table(&self.ticks_table)
            .with_trades_table(&self.trades_table);

        if let Some(ref symbols) = self.symbols {
            config = config.with_symbols(symbols.clone());
        }

        if let Some(ref exchanges) = self.exchanges {
            config = config.with_exchanges(exchanges.clone());
        }

        config
    }
}

// ============================================================================
// Backtest State and Signals
// ============================================================================

/// Signal types that a strategy can generate
#[derive(Debug, Clone)]
pub enum SignalType {
    /// Buy signal with optional size
    Buy { size: Decimal },
    /// Sell signal with optional size
    Sell { size: Decimal },
    /// Close all positions
    CloseAll,
    /// No action
    None,
}

/// A simulated trade execution
#[derive(Debug, Clone)]
pub struct SimulatedTrade {
    /// Symbol traded
    pub symbol: String,
    /// Execution price
    pub price: Decimal,
    /// Trade size (positive for buy, negative for sell)
    pub size: Decimal,
    /// Commission paid
    pub commission: Decimal,
    /// Slippage incurred
    pub slippage: Decimal,
    /// Timestamp of execution
    pub timestamp: DateTime<Utc>,
    /// Realized P&L (for closing trades)
    pub realized_pnl: Option<Decimal>,
}

/// Position tracking
#[derive(Debug, Clone, Default)]
pub struct Position {
    /// Symbol
    pub symbol: String,
    /// Current position size (positive = long, negative = short)
    pub size: Decimal,
    /// Average entry price
    pub avg_entry_price: Decimal,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Last update time
    pub last_update: Option<DateTime<Utc>>,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            ..Default::default()
        }
    }

    /// Update position with a trade
    pub fn apply_trade(&mut self, size: Decimal, price: Decimal) -> Decimal {
        let mut realized_pnl = Decimal::ZERO;

        if self.size == Decimal::ZERO {
            // Opening new position
            self.avg_entry_price = price;
            self.size = size;
        } else if (self.size > Decimal::ZERO && size > Decimal::ZERO)
            || (self.size < Decimal::ZERO && size < Decimal::ZERO)
        {
            // Adding to position - calculate new average
            let total_cost = self.avg_entry_price * self.size.abs() + price * size.abs();
            let new_size = self.size + size;
            self.avg_entry_price = total_cost / new_size.abs();
            self.size = new_size;
        } else {
            // Reducing or reversing position
            let close_size = size.abs().min(self.size.abs());
            if self.size > Decimal::ZERO {
                // Long position, selling
                realized_pnl = (price - self.avg_entry_price) * close_size;
            } else {
                // Short position, buying
                realized_pnl = (self.avg_entry_price - price) * close_size;
            }

            let new_size = self.size + size;
            if new_size.abs() < Decimal::new(1, 10) {
                // Position closed
                self.size = Decimal::ZERO;
                self.avg_entry_price = Decimal::ZERO;
            } else if (new_size > Decimal::ZERO) != (self.size > Decimal::ZERO) {
                // Position reversed
                self.avg_entry_price = price;
                self.size = new_size;
            } else {
                self.size = new_size;
            }
        }

        realized_pnl
    }

    /// Update unrealized P&L based on current price
    pub fn update_unrealized(&mut self, current_price: Decimal) {
        if self.size == Decimal::ZERO {
            self.unrealized_pnl = Decimal::ZERO;
        } else if self.size > Decimal::ZERO {
            self.unrealized_pnl = (current_price - self.avg_entry_price) * self.size;
        } else {
            self.unrealized_pnl = (self.avg_entry_price - current_price) * self.size.abs();
        }
    }
}

/// Backtest state tracking
#[derive(Debug, Clone)]
pub struct BacktestState {
    /// Current positions by symbol
    pub positions: HashMap<String, Position>,
    /// Account balance (cash)
    pub balance: Decimal,
    /// Initial balance
    pub initial_balance: Decimal,
    /// Total realized P&L
    pub realized_pnl: Decimal,
    /// Trade history
    pub trades: Vec<SimulatedTrade>,
    /// Current prices by symbol
    pub current_prices: HashMap<String, Decimal>,
    /// Price history (for indicators)
    pub price_history: HashMap<String, Vec<(DateTime<Utc>, Decimal)>>,
    /// Maximum price history length per symbol
    pub max_history_len: usize,
    /// High water mark for drawdown calculation
    pub high_water_mark: Decimal,
    /// Maximum drawdown observed
    pub max_drawdown: Decimal,
    /// Commission rate (basis points)
    pub commission_bps: Decimal,
    /// Slippage rate (basis points)
    pub slippage_bps: Decimal,
    /// Current timestamp
    pub current_time: Option<DateTime<Utc>>,
    /// Number of winning trades
    pub winning_trades: u64,
    /// Number of losing trades
    pub losing_trades: u64,
    /// Sum of winning trade P&L
    pub gross_profit: Decimal,
    /// Sum of losing trade P&L (absolute)
    pub gross_loss: Decimal,
}

impl BacktestState {
    /// Create a new backtest state
    pub fn new(initial_balance: Decimal, commission_bps: Decimal, slippage_bps: Decimal) -> Self {
        Self {
            positions: HashMap::new(),
            balance: initial_balance,
            initial_balance,
            realized_pnl: Decimal::ZERO,
            trades: Vec::new(),
            current_prices: HashMap::new(),
            price_history: HashMap::new(),
            max_history_len: 1000,
            high_water_mark: initial_balance,
            max_drawdown: Decimal::ZERO,
            commission_bps,
            slippage_bps,
            current_time: None,
            winning_trades: 0,
            losing_trades: 0,
            gross_profit: Decimal::ZERO,
            gross_loss: Decimal::ZERO,
        }
    }

    /// Update current price for a symbol
    pub fn update_price(&mut self, symbol: String, price: Decimal) {
        self.current_prices.insert(symbol.clone(), price);

        // Update price history
        if let Some(time) = self.current_time {
            let history = self.price_history.entry(symbol.clone()).or_default();
            history.push((time, price));
            if history.len() > self.max_history_len {
                history.remove(0);
            }
        }

        // Update position unrealized P&L
        if let Some(pos) = self.positions.get_mut(&symbol) {
            pos.update_unrealized(price);
        }

        // Update drawdown
        self.update_drawdown();
    }

    /// Execute a trade
    pub fn execute_trade(
        &mut self,
        symbol: &str,
        size: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Option<SimulatedTrade> {
        let price = *self.current_prices.get(symbol)?;

        // Apply slippage (adverse)
        let slippage_mult = self.slippage_bps / Decimal::from(10_000);
        let exec_price = if size > Decimal::ZERO {
            price * (Decimal::ONE + slippage_mult) // Buy higher
        } else {
            price * (Decimal::ONE - slippage_mult) // Sell lower
        };

        // Calculate commission
        let notional = exec_price * size.abs();
        let commission = notional * self.commission_bps / Decimal::from(10_000);

        // Update position
        let position = self
            .positions
            .entry(symbol.to_string())
            .or_insert_with(|| Position::new(symbol));
        let realized_pnl = position.apply_trade(size, exec_price);

        // Update balance
        self.balance -= commission;
        self.balance += realized_pnl;
        self.realized_pnl += realized_pnl;

        // Track win/loss
        if realized_pnl > Decimal::ZERO {
            self.winning_trades += 1;
            self.gross_profit += realized_pnl;
        } else if realized_pnl < Decimal::ZERO {
            self.losing_trades += 1;
            self.gross_loss += realized_pnl.abs();
        }

        let trade = SimulatedTrade {
            symbol: symbol.to_string(),
            price: exec_price,
            size,
            commission,
            slippage: (exec_price - price).abs() * size.abs(),
            timestamp,
            realized_pnl: if realized_pnl != Decimal::ZERO {
                Some(realized_pnl)
            } else {
                None
            },
        };

        self.trades.push(trade.clone());
        self.update_drawdown();

        Some(trade)
    }

    /// Close all positions
    pub fn close_all_positions(&mut self, timestamp: DateTime<Utc>) -> Vec<SimulatedTrade> {
        let mut trades = Vec::new();
        let symbols: Vec<String> = self.positions.keys().cloned().collect();

        for symbol in symbols {
            let size = self
                .positions
                .get(&symbol)
                .map(|p| -p.size)
                .unwrap_or_default();
            if size != Decimal::ZERO {
                if let Some(trade) = self.execute_trade(&symbol, size, timestamp) {
                    trades.push(trade);
                }
            }
        }

        trades
    }

    /// Update drawdown tracking
    fn update_drawdown(&mut self) {
        let equity = self.equity();
        if equity > self.high_water_mark {
            self.high_water_mark = equity;
        }

        let drawdown = (self.high_water_mark - equity) / self.high_water_mark;
        if drawdown > self.max_drawdown {
            self.max_drawdown = drawdown;
        }
    }

    /// Get total equity (balance + unrealized P&L)
    pub fn equity(&self) -> Decimal {
        let unrealized: Decimal = self.positions.values().map(|p| p.unrealized_pnl).sum();
        self.balance + unrealized
    }

    /// Get total return as a decimal (0.1 = 10%)
    pub fn total_return(&self) -> Decimal {
        if self.initial_balance == Decimal::ZERO {
            return Decimal::ZERO;
        }
        (self.equity() - self.initial_balance) / self.initial_balance
    }

    /// Get win rate
    pub fn win_rate(&self) -> f64 {
        let total = self.winning_trades + self.losing_trades;
        if total == 0 {
            return 0.0;
        }
        self.winning_trades as f64 / total as f64
    }

    /// Get profit factor
    pub fn profit_factor(&self) -> f64 {
        if self.gross_loss == Decimal::ZERO {
            return if self.gross_profit > Decimal::ZERO {
                f64::INFINITY
            } else {
                0.0
            };
        }
        self.gross_profit.to_f64().unwrap_or(0.0) / self.gross_loss.to_f64().unwrap_or(1.0)
    }

    /// Get recent prices for a symbol (for indicator calculation)
    pub fn get_prices(&self, symbol: &str, count: usize) -> Vec<Decimal> {
        self.price_history
            .get(symbol)
            .map(|h| {
                h.iter()
                    .rev()
                    .take(count)
                    .map(|(_, p)| *p)
                    .collect::<Vec<_>>()
                    .into_iter()
                    .rev()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Calculate simple moving average
    pub fn sma(&self, symbol: &str, period: usize) -> Option<Decimal> {
        let prices = self.get_prices(symbol, period);
        if prices.len() < period {
            return None;
        }
        let sum: Decimal = prices.iter().sum();
        Some(sum / Decimal::from(period))
    }

    /// Reset state for new backtest run
    pub fn reset(&mut self) {
        self.positions.clear();
        self.balance = self.initial_balance;
        self.realized_pnl = Decimal::ZERO;
        self.trades.clear();
        self.current_prices.clear();
        self.price_history.clear();
        self.high_water_mark = self.initial_balance;
        self.max_drawdown = Decimal::ZERO;
        self.current_time = None;
        self.winning_trades = 0;
        self.losing_trades = 0;
        self.gross_profit = Decimal::ZERO;
        self.gross_loss = Decimal::ZERO;
    }
}

// ============================================================================
// Event-Based Strategy Interface
// ============================================================================

/// Trait for event-based strategy implementations
pub trait EventBasedStrategy: Send + Sync {
    /// Called when a tick event is received
    fn on_tick(&mut self, tick: &TickData, state: &mut BacktestState) -> Option<SignalType>;

    /// Called when a trade event is received
    fn on_trade(&mut self, trade: &TradeData, state: &mut BacktestState) -> Option<SignalType>;

    /// Called at the end of the backtest to finalize
    fn on_finish(&mut self, _state: &mut BacktestState) {}

    /// Create a new instance with given parameters
    fn with_params(&self, params: &ParameterSet) -> Box<dyn EventBasedStrategy>;

    /// Get the name of the strategy
    fn name(&self) -> &str {
        "UnnamedStrategy"
    }
}

// ============================================================================
// QuestDB Strategy Evaluator
// ============================================================================

/// Strategy evaluator that loads data from QuestDB
pub struct QuestDBStrategyEvaluator<S: EventBasedStrategy + Clone> {
    /// QuestDB configuration
    config: QuestDBBacktestConfig,
    /// Strategy template (cloned for each evaluation)
    strategy_template: S,
    /// Cached market events (if caching enabled)
    cached_events: Arc<RwLock<Option<CachedEvents>>>,
}

/// Cached events for reuse
struct CachedEvents {
    events: Vec<MarketEvent>,
    start_time: DateTime<Utc>,
    end_time: DateTime<Utc>,
}

impl<S: EventBasedStrategy + Clone> QuestDBStrategyEvaluator<S> {
    /// Create a new evaluator
    pub fn new(config: QuestDBBacktestConfig, strategy: S) -> Self {
        Self {
            config,
            strategy_template: strategy,
            cached_events: Arc::new(RwLock::new(None)),
        }
    }

    /// Load events from QuestDB for a time range
    async fn load_events(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<MarketEvent>, QuestDBBacktestError> {
        // Check cache first
        if self.config.cache_data {
            let cache = self.cached_events.read();
            if let Some(ref cached) = *cache {
                // Check if cached data covers our range
                if cached.start_time <= start_time && cached.end_time >= end_time {
                    // Filter events to our range
                    let filtered: Vec<MarketEvent> = cached
                        .events
                        .iter()
                        .filter(|e| {
                            e.timestamp()
                                .map(|t| t >= start_time && t <= end_time)
                                .unwrap_or(false)
                        })
                        .cloned()
                        .collect();
                    debug!(
                        "Using {} cached events for range {} to {}",
                        filtered.len(),
                        start_time,
                        end_time
                    );
                    return Ok(filtered);
                }
            }
        }

        // Load from QuestDB
        let mut replay_config = ReplayConfig::fast();
        replay_config.event_types.ticks = self.config.load_ticks;
        replay_config.event_types.trades = self.config.load_trades;

        let mut engine = ReplayEngine::new(replay_config);

        let count = engine
            .load_all_from_questdb(
                &self.config.host,
                self.config.port,
                &self.config.ticks_table,
                &self.config.trades_table,
                Some(start_time),
                Some(end_time),
                self.config.symbols.clone(),
            )
            .await?;

        if count == 0 {
            return Err(QuestDBBacktestError::NoDataInRange(start_time, end_time));
        }

        // Extract events from engine
        let events: Vec<MarketEvent> = (0..engine.total_events())
            .filter_map(|i| engine.get_event(i).cloned())
            .collect();

        info!(
            "Loaded {} events from QuestDB for range {} to {}",
            events.len(),
            start_time,
            end_time
        );

        // Update cache if enabled
        if self.config.cache_data {
            let mut cache = self.cached_events.write();
            *cache = Some(CachedEvents {
                events: events.clone(),
                start_time,
                end_time,
            });
        }

        Ok(events)
    }

    /// Run backtest on events with given parameters
    fn run_backtest(
        &self,
        events: &[MarketEvent],
        params: &ParameterSet,
    ) -> Result<BacktestState, QuestDBBacktestError> {
        let mut strategy = self.strategy_template.with_params(params);
        let mut state = BacktestState::new(
            self.config.initial_balance,
            self.config.commission_bps,
            self.config.slippage_bps,
        );

        for event in events {
            // Update time
            if let Some(ts) = event.timestamp() {
                state.current_time = Some(ts);
            }

            // Process event and get signal
            let signal = match event {
                MarketEvent::Tick(tick) => {
                    state.update_price(tick.symbol.clone(), tick.mid_price());
                    strategy.on_tick(tick, &mut state)
                }
                MarketEvent::Trade(trade) => {
                    state.update_price(trade.symbol.clone(), trade.price);
                    strategy.on_trade(trade, &mut state)
                }
                _ => None,
            };

            // Execute signal
            if let Some(signal) = signal {
                let ts = state.current_time.unwrap_or_else(Utc::now);
                match signal {
                    SignalType::Buy { size } => {
                        if let Some(symbol) = event.symbol() {
                            state.execute_trade(symbol, size, ts);
                        }
                    }
                    SignalType::Sell { size } => {
                        if let Some(symbol) = event.symbol() {
                            state.execute_trade(symbol, -size, ts);
                        }
                    }
                    SignalType::CloseAll => {
                        state.close_all_positions(ts);
                    }
                    SignalType::None => {}
                }
            }
        }

        // Finalize
        strategy.on_finish(&mut state);

        // Close remaining positions
        if let Some(ts) = state.current_time {
            state.close_all_positions(ts);
        }

        Ok(state)
    }

    /// Calculate Sharpe ratio from daily returns
    fn calculate_sharpe(&self, state: &BacktestState) -> f64 {
        if state.trades.is_empty() {
            return 0.0;
        }

        // Group trades by day and calculate daily returns
        let mut daily_pnl: HashMap<String, Decimal> = HashMap::new();
        for trade in &state.trades {
            if let Some(pnl) = trade.realized_pnl {
                let day = trade.timestamp.format("%Y-%m-%d").to_string();
                *daily_pnl.entry(day).or_default() += pnl;
            }
        }

        if daily_pnl.is_empty() {
            return 0.0;
        }

        let returns: Vec<f64> = daily_pnl
            .values()
            .map(|pnl| pnl.to_f64().unwrap_or(0.0) / state.initial_balance.to_f64().unwrap_or(1.0))
            .collect();

        if returns.is_empty() {
            return 0.0;
        }

        let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev < 1e-10 {
            return 0.0;
        }

        // Annualized Sharpe (assuming 252 trading days)
        (mean / std_dev) * 252.0_f64.sqrt()
    }

    /// Clear cached data
    pub fn clear_cache(&self) {
        let mut cache = self.cached_events.write();
        *cache = None;
    }
}

impl<S: EventBasedStrategy + Clone + 'static> StrategyEvaluator for QuestDBStrategyEvaluator<S> {
    fn evaluate(
        &self,
        params: &ParameterSet,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<OptimizationRunResult, OptimizationError> {
        // We need to block on the async load - use a runtime handle
        let events = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.load_events(start_time, end_time))
        })
        .map_err(|e| OptimizationError::Failed(e.to_string()))?;

        let start = std::time::Instant::now();

        // Run backtest
        let state = self
            .run_backtest(&events, params)
            .map_err(|e| OptimizationError::Failed(e.to_string()))?;

        let duration_ms = start.elapsed().as_millis() as u64;

        // Calculate metrics
        let sharpe = self.calculate_sharpe(&state);
        let total_return = state.total_return().to_f64().unwrap_or(0.0);
        let win_rate = state.win_rate();
        let profit_factor = state.profit_factor();
        let max_drawdown = state.max_drawdown.to_f64().unwrap_or(0.0);
        let total_trades = state.trades.len();

        // Build result (metric_value is Sharpe by default)
        let mut result = OptimizationRunResult::new(params.clone(), sharpe);
        result.sharpe_ratio = Some(sharpe);
        result.total_trades = total_trades as u64;
        result.win_rate = win_rate;
        result.max_drawdown = max_drawdown;
        result.profit_factor = Some(profit_factor);
        result.run_duration_ms = duration_ms;

        // Store additional metrics
        result
            .metrics
            .insert("total_return".to_string(), total_return);
        result.metrics.insert(
            "gross_profit".to_string(),
            state.gross_profit.to_f64().unwrap_or(0.0),
        );
        result.metrics.insert(
            "gross_loss".to_string(),
            state.gross_loss.to_f64().unwrap_or(0.0),
        );
        result
            .metrics
            .insert("num_winning".to_string(), state.winning_trades as f64);
        result
            .metrics
            .insert("num_losing".to_string(), state.losing_trades as f64);

        Ok(result)
    }
}

// ============================================================================
// QuestDB Walk-Forward Runner
// ============================================================================

/// Walk-forward runner with QuestDB data source
pub struct QuestDBWalkForwardRunner {
    /// QuestDB configuration
    questdb_config: QuestDBBacktestConfig,
    /// Walk-forward configuration
    wf_config: WalkForwardConfig,
    /// Data time range (discovered or provided)
    data_start: Option<DateTime<Utc>>,
    data_end: Option<DateTime<Utc>>,
    /// Verbose logging
    verbose: bool,
}

impl QuestDBWalkForwardRunner {
    /// Create a builder for the runner
    pub fn builder() -> QuestDBWalkForwardRunnerBuilder {
        QuestDBWalkForwardRunnerBuilder::new()
    }

    /// Create a new runner with configurations
    pub fn new(questdb_config: QuestDBBacktestConfig, wf_config: WalkForwardConfig) -> Self {
        Self {
            questdb_config,
            wf_config,
            data_start: None,
            data_end: None,
            verbose: false,
        }
    }

    /// Set the data time range explicitly
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.data_start = Some(start);
        self.data_end = Some(end);
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Discover the data time range from QuestDB
    pub async fn discover_time_range(
        &mut self,
    ) -> Result<(DateTime<Utc>, DateTime<Utc>), QuestDBBacktestError> {
        let query = format!(
            "SELECT min(timestamp) as min_ts, max(timestamp) as max_ts FROM {}",
            self.questdb_config.ticks_table
        );

        let url = format!(
            "http://{}:{}/exec?query={}",
            self.questdb_config.host,
            self.questdb_config.port,
            urlencoding::encode(&query)
        );

        let client = reqwest::Client::new();
        let response = client
            .get(&url)
            .send()
            .await
            .map_err(|e| QuestDBBacktestError::ConnectionFailed(e.to_string()))?;

        let text = response
            .text()
            .await
            .map_err(|e| QuestDBBacktestError::LoadFailed(e.to_string()))?;

        // Parse response (simple JSON parsing)
        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| QuestDBBacktestError::LoadFailed(format!("JSON parse error: {}", e)))?;

        let dataset = json
            .get("dataset")
            .and_then(|d| d.as_array())
            .and_then(|a| a.first())
            .and_then(|row| row.as_array())
            .ok_or_else(|| {
                QuestDBBacktestError::LoadFailed("Invalid response format".to_string())
            })?;

        let min_ts = dataset
            .first()
            .and_then(|v| v.as_str())
            .ok_or_else(|| QuestDBBacktestError::LoadFailed("Missing min timestamp".to_string()))?;

        let max_ts = dataset
            .get(1)
            .and_then(|v| v.as_str())
            .ok_or_else(|| QuestDBBacktestError::LoadFailed("Missing max timestamp".to_string()))?;

        let start = DateTime::parse_from_rfc3339(min_ts)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| QuestDBBacktestError::LoadFailed(format!("Parse error: {}", e)))?;

        let end = DateTime::parse_from_rfc3339(max_ts)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| QuestDBBacktestError::LoadFailed(format!("Parse error: {}", e)))?;

        self.data_start = Some(start);
        self.data_end = Some(end);

        info!("Discovered data range: {} to {}", start, end);

        Ok((start, end))
    }

    /// Run walk-forward optimization with an event-based strategy
    pub async fn run<S: EventBasedStrategy + Clone + 'static>(
        &mut self,
        strategy: &S,
    ) -> Result<WalkForwardResult, QuestDBBacktestError> {
        // Ensure we have time range
        let (start, end) = match (self.data_start, self.data_end) {
            (Some(s), Some(e)) => (s, e),
            _ => self.discover_time_range().await?,
        };

        info!(
            "Starting QuestDB walk-forward: {} windows, data range {} to {}",
            self.wf_config.num_windows, start, end
        );

        // Create evaluator
        let evaluator =
            QuestDBStrategyEvaluator::new(self.questdb_config.clone(), strategy.clone());

        // Create walk-forward runner
        let runner = WalkForwardBacktestRunner::new(self.wf_config.clone(), start, end)?
            .with_verbose(self.verbose);

        // Run walk-forward
        let result = runner.run(&evaluator).await?;

        Ok(result)
    }

    /// Run walk-forward synchronously (blocks)
    pub fn run_sync<S: EventBasedStrategy + Clone + 'static>(
        &mut self,
        strategy: &S,
    ) -> Result<WalkForwardResult, QuestDBBacktestError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(self.run(strategy))
        })
    }
}

// ============================================================================
// Builder
// ============================================================================

/// Builder for QuestDBWalkForwardRunner
pub struct QuestDBWalkForwardRunnerBuilder {
    questdb_host: String,
    questdb_port: u16,
    ticks_table: String,
    trades_table: String,
    symbols: Option<Vec<String>>,
    exchanges: Option<Vec<String>>,
    load_ticks: bool,
    load_trades: bool,
    initial_balance: Decimal,
    commission_bps: Decimal,
    slippage_bps: Decimal,
    num_windows: usize,
    in_sample_pct: f64,
    min_trades: usize,
    anchored: bool,
    parameters: Vec<ParameterRange>,
    metric: OptimizationMetric,
    direction: OptimizationDirection,
    parallel: bool,
    verbose: bool,
    data_start: Option<DateTime<Utc>>,
    data_end: Option<DateTime<Utc>>,
}

impl Default for QuestDBWalkForwardRunnerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QuestDBWalkForwardRunnerBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self {
            questdb_host: "localhost".to_string(),
            questdb_port: 9000,
            ticks_table: "fks_ticks".to_string(),
            trades_table: "fks_trades".to_string(),
            symbols: None,
            exchanges: None,
            load_ticks: true,
            load_trades: true,
            initial_balance: Decimal::from(10_000),
            commission_bps: Decimal::new(10, 2),
            slippage_bps: Decimal::new(5, 2),
            num_windows: 5,
            in_sample_pct: 0.7,
            min_trades: 10,
            anchored: false,
            parameters: Vec::new(),
            metric: OptimizationMetric::SharpeRatio,
            direction: OptimizationDirection::Maximize,
            parallel: true,
            verbose: false,
            data_start: None,
            data_end: None,
        }
    }

    /// Set QuestDB configuration
    pub fn questdb_config(mut self, config: QuestDBBacktestConfig) -> Self {
        self.questdb_host = config.host;
        self.questdb_port = config.port;
        self.ticks_table = config.ticks_table;
        self.trades_table = config.trades_table;
        self.symbols = config.symbols;
        self.exchanges = config.exchanges;
        self.load_ticks = config.load_ticks;
        self.load_trades = config.load_trades;
        self.initial_balance = config.initial_balance;
        self.commission_bps = config.commission_bps;
        self.slippage_bps = config.slippage_bps;
        self
    }

    /// Set QuestDB host
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.questdb_host = host.into();
        self
    }

    /// Set QuestDB port
    pub fn port(mut self, port: u16) -> Self {
        self.questdb_port = port;
        self
    }

    /// Set ticks table name
    pub fn ticks_table(mut self, table: impl Into<String>) -> Self {
        self.ticks_table = table.into();
        self
    }

    /// Set trades table name
    pub fn trades_table(mut self, table: impl Into<String>) -> Self {
        self.trades_table = table.into();
        self
    }

    /// Set symbols to filter
    pub fn symbols(mut self, symbols: Vec<String>) -> Self {
        self.symbols = Some(symbols);
        self
    }

    /// Set exchanges to filter
    pub fn exchanges(mut self, exchanges: Vec<String>) -> Self {
        self.exchanges = Some(exchanges);
        self
    }

    /// Set initial balance
    pub fn initial_balance(mut self, balance: Decimal) -> Self {
        self.initial_balance = balance;
        self
    }

    /// Set commission in basis points
    pub fn commission_bps(mut self, bps: Decimal) -> Self {
        self.commission_bps = bps;
        self
    }

    /// Set slippage in basis points
    pub fn slippage_bps(mut self, bps: Decimal) -> Self {
        self.slippage_bps = bps;
        self
    }

    /// Set number of walk-forward windows
    pub fn windows(mut self, n: usize) -> Self {
        self.num_windows = n;
        self
    }

    /// Set in-sample percentage
    pub fn in_sample_pct(mut self, pct: f64) -> Self {
        self.in_sample_pct = pct;
        self
    }

    /// Set minimum trades per window
    pub fn min_trades(mut self, n: usize) -> Self {
        self.min_trades = n;
        self
    }

    /// Use anchored windows
    pub fn anchored(mut self) -> Self {
        self.anchored = true;
        self
    }

    /// Use rolling windows
    pub fn rolling(mut self) -> Self {
        self.anchored = false;
        self
    }

    /// Add a parameter range
    pub fn parameter(mut self, param: ParameterRange) -> Self {
        self.parameters.push(param);
        self
    }

    /// Set the optimization metric
    pub fn metric(mut self, metric: OptimizationMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Set optimization direction
    pub fn direction(mut self, direction: OptimizationDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Enable/disable parallel execution
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Enable/disable verbose logging
    pub fn verbose(mut self, enabled: bool) -> Self {
        self.verbose = enabled;
        self
    }

    /// Set data time range
    pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.data_start = Some(start);
        self.data_end = Some(end);
        self
    }

    /// Build the runner
    pub fn build(self) -> Result<QuestDBWalkForwardRunner, QuestDBBacktestError> {
        if self.parameters.is_empty() {
            return Err(QuestDBBacktestError::InvalidConfig(
                "At least one parameter range must be specified".to_string(),
            ));
        }

        // Build QuestDB config
        let questdb_config = QuestDBBacktestConfig {
            host: self.questdb_host,
            port: self.questdb_port,
            ticks_table: self.ticks_table,
            trades_table: self.trades_table,
            symbols: self.symbols,
            exchanges: self.exchanges,
            load_ticks: self.load_ticks,
            load_trades: self.load_trades,
            cache_data: true,
            initial_balance: self.initial_balance,
            commission_bps: self.commission_bps,
            slippage_bps: self.slippage_bps,
        };

        // Build optimization config
        let mut opt_config = OptimizationConfig::new()
            .with_metric(self.metric)
            .with_direction(self.direction)
            .with_parallel(self.parallel)
            .with_verbose(self.verbose);

        for param in self.parameters {
            opt_config = opt_config.with_parameter(param);
        }

        // Build walk-forward config
        let wf_config = WalkForwardConfig::new(self.num_windows)
            .with_in_sample_pct(self.in_sample_pct)
            .with_min_trades(self.min_trades)
            .with_optimization(opt_config);

        let wf_config = if self.anchored {
            wf_config.anchored()
        } else {
            wf_config.rolling()
        };

        let mut runner = QuestDBWalkForwardRunner::new(questdb_config, wf_config);
        runner.data_start = self.data_start;
        runner.data_end = self.data_end;
        runner.verbose = self.verbose;

        Ok(runner)
    }
}

// ============================================================================
// URL Encoding Helper
// ============================================================================

mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut result = String::new();
        for c in s.chars() {
            match c {
                'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => result.push(c),
                ' ' => result.push('+'),
                _ => {
                    for byte in c.to_string().bytes() {
                        result.push_str(&format!("%{:02X}", byte));
                    }
                }
            }
        }
        result
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_questdb_config() {
        let config = QuestDBBacktestConfig::new("localhost", 9000)
            .with_ticks_table("test_ticks")
            .with_trades_table("test_trades")
            .with_symbols(vec!["BTC/USDT".to_string()])
            .with_initial_balance(Decimal::from(50_000));

        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 9000);
        assert_eq!(config.ticks_table, "test_ticks");
        assert_eq!(config.trades_table, "test_trades");
        assert_eq!(config.symbols, Some(vec!["BTC/USDT".to_string()]));
        assert_eq!(config.initial_balance, Decimal::from(50_000));
    }

    #[test]
    fn test_backtest_state_creation() {
        let state = BacktestState::new(
            Decimal::from(10_000),
            Decimal::new(10, 2),
            Decimal::new(5, 2),
        );

        assert_eq!(state.balance, Decimal::from(10_000));
        assert_eq!(state.initial_balance, Decimal::from(10_000));
        assert_eq!(state.realized_pnl, Decimal::ZERO);
        assert!(state.trades.is_empty());
        assert!(state.positions.is_empty());
    }

    #[test]
    fn test_position_apply_trade() {
        let mut pos = Position::new("BTC/USDT");

        // Open long position
        let pnl = pos.apply_trade(Decimal::from(1), Decimal::from(50_000));
        assert_eq!(pnl, Decimal::ZERO);
        assert_eq!(pos.size, Decimal::from(1));
        assert_eq!(pos.avg_entry_price, Decimal::from(50_000));

        // Close long position at profit
        let pnl = pos.apply_trade(Decimal::from(-1), Decimal::from(51_000));
        assert_eq!(pnl, Decimal::from(1_000)); // (51000 - 50000) * 1
        assert_eq!(pos.size, Decimal::ZERO);
    }

    #[test]
    fn test_position_add_to_position() {
        let mut pos = Position::new("BTC/USDT");

        // Open long position
        pos.apply_trade(Decimal::from(1), Decimal::from(50_000));

        // Add to position
        pos.apply_trade(Decimal::from(1), Decimal::from(52_000));
        assert_eq!(pos.size, Decimal::from(2));
        // Average: (50000 + 52000) / 2 = 51000
        assert_eq!(pos.avg_entry_price, Decimal::from(51_000));
    }

    #[test]
    fn test_backtest_state_equity() {
        let mut state = BacktestState::new(
            Decimal::from(10_000),
            Decimal::ZERO, // No commission for test
            Decimal::ZERO, // No slippage for test
        );

        // Update price
        state.update_price("BTC/USDT".to_string(), Decimal::from(50_000));

        // Execute buy
        state.current_time = Some(Utc::now());
        state.execute_trade("BTC/USDT", Decimal::from(1), Utc::now());

        // Check position exists
        assert!(state.positions.contains_key("BTC/USDT"));
        assert_eq!(state.positions["BTC/USDT"].size, Decimal::from(1));

        // Price increases
        state.update_price("BTC/USDT".to_string(), Decimal::from(51_000));

        // Unrealized P&L should be +1000
        assert_eq!(
            state.positions["BTC/USDT"].unrealized_pnl,
            Decimal::from(1_000)
        );

        // Equity = balance + unrealized
        assert_eq!(state.equity(), Decimal::from(11_000));
    }

    #[test]
    fn test_backtest_state_win_rate() {
        let mut state = BacktestState::new(Decimal::from(10_000), Decimal::ZERO, Decimal::ZERO);

        // Simulate wins and losses
        state.winning_trades = 7;
        state.losing_trades = 3;

        assert!((state.win_rate() - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_backtest_state_profit_factor() {
        let mut state = BacktestState::new(Decimal::from(10_000), Decimal::ZERO, Decimal::ZERO);

        state.gross_profit = Decimal::from(2_000);
        state.gross_loss = Decimal::from(1_000);

        assert!((state.profit_factor() - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_builder_validation() {
        // Should fail without parameters
        let result = QuestDBWalkForwardRunner::builder()
            .host("localhost")
            .port(9000)
            .build();

        assert!(result.is_err());
        match result {
            Err(QuestDBBacktestError::InvalidConfig(msg)) => {
                assert!(msg.contains("parameter"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_builder_success() {
        let result = QuestDBWalkForwardRunner::builder()
            .host("localhost")
            .port(9000)
            .ticks_table("my_ticks")
            .symbols(vec!["BTC/USDT".to_string()])
            .windows(5)
            .in_sample_pct(0.7)
            .parameter(ParameterRange::int("period", 10, 50, 10))
            .metric(OptimizationMetric::SharpeRatio)
            .build();

        assert!(result.is_ok());
        let runner = result.unwrap();
        assert_eq!(runner.questdb_config.host, "localhost");
        assert_eq!(runner.questdb_config.ticks_table, "my_ticks");
        assert_eq!(runner.wf_config.num_windows, 5);
    }

    #[test]
    fn test_url_encoding() {
        assert_eq!(urlencoding::encode("hello world"), "hello+world");
        assert_eq!(urlencoding::encode("a=b"), "a%3Db");
        assert_eq!(urlencoding::encode("test-123"), "test-123");
    }

    #[test]
    fn test_backtest_state_sma() {
        let mut state = BacktestState::new(Decimal::from(10_000), Decimal::ZERO, Decimal::ZERO);

        // Add price history
        let symbol = "BTC/USDT".to_string();
        state.current_time = Some(Utc::now());

        for i in 1..=5 {
            state.update_price(symbol.clone(), Decimal::from(100 * i));
        }

        // SMA(5) = (100 + 200 + 300 + 400 + 500) / 5 = 300
        let sma = state.sma(&symbol, 5);
        assert_eq!(sma, Some(Decimal::from(300)));

        // SMA(10) should be None (not enough data)
        let sma = state.sma(&symbol, 10);
        assert_eq!(sma, None);
    }

    #[test]
    fn test_backtest_state_reset() {
        let mut state = BacktestState::new(Decimal::from(10_000), Decimal::ZERO, Decimal::ZERO);

        // Add some data
        state.update_price("BTC/USDT".to_string(), Decimal::from(50_000));
        state.winning_trades = 5;
        state.gross_profit = Decimal::from(1_000);

        // Reset
        state.reset();

        assert!(state.positions.is_empty());
        assert!(state.trades.is_empty());
        assert!(state.current_prices.is_empty());
        assert_eq!(state.balance, Decimal::from(10_000));
        assert_eq!(state.winning_trades, 0);
        assert_eq!(state.gross_profit, Decimal::ZERO);
    }

    #[test]
    fn test_close_all_positions() {
        let mut state = BacktestState::new(Decimal::from(10_000), Decimal::ZERO, Decimal::ZERO);

        state.current_time = Some(Utc::now());
        state.update_price("BTC/USDT".to_string(), Decimal::from(50_000));
        state.update_price("ETH/USDT".to_string(), Decimal::from(3_000));

        // Open positions
        state.execute_trade("BTC/USDT", Decimal::from(1), Utc::now());
        state.execute_trade("ETH/USDT", Decimal::from(2), Utc::now());

        assert_eq!(state.positions.len(), 2);

        // Close all
        let trades = state.close_all_positions(Utc::now());

        // Should have 2 closing trades
        assert_eq!(trades.len(), 2);

        // Positions should be flat
        for pos in state.positions.values() {
            assert_eq!(pos.size, Decimal::ZERO);
        }
    }
}
