//! Simulation Environment - Main Entry Point
//!
//! Provides a unified interface for running strategies across different modes:
//! - Backtest: Historical data replay with zero-lookahead
//! - Forward Test: Live data with simulated execution (paper trading)
//! - Live: Real execution via exchange APIs (Kraken for Canada)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                          SimEnvironment                                  │
//! ├─────────────────────────────────────────────────────────────────────────┤
//! │                                                                          │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
//! │  │   Strategy  │───▶│   Executor  │───▶│   Exchange  │                 │
//! │  │  (Signals)  │    │  (Sim/Live) │    │ (API/Mock)  │                 │
//! │  └─────────────┘    └─────────────┘    └─────────────┘                 │
//! │         ▲                                      │                         │
//! │         │                                      │                         │
//! │         │           ┌─────────────┐           │                         │
//! │         └───────────│  DataFeed   │◀──────────┘                         │
//! │                     │ (Tick/Trade)│                                      │
//! │                     └─────────────┘                                      │
//! │                                                                          │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use rust_decimal::Decimal;
use rust_decimal::prelude::{Signed, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use thiserror::Error;
use tokio::sync::broadcast;
use tracing::{debug, info, warn};

use super::config::{SimConfig, SimMode};
use super::data_feed::{AggregatedDataFeed, DataFeed, MarketEvent};
use super::data_recorder::{DataRecorder, RecorderConfig};
use super::live_feed_bridge::{LiveFeedBridge, LiveFeedBridgeConfig};
use crate::exchanges::MarketDataAggregator;

/// Errors that can occur in the simulation environment
#[derive(Debug, Error)]
pub enum SimError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Data feed error: {0}")]
    DataFeed(String),

    #[error("Execution error: {0}")]
    Execution(String),

    #[error("Strategy error: {0}")]
    Strategy(String),

    #[error("Exchange error: {0}")]
    Exchange(String),

    #[error("Not initialized: {0}")]
    NotInitialized(String),

    #[error("Already running")]
    AlreadyRunning,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Trading signal from strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Signal {
    /// Buy signal with target position size
    Buy {
        symbol: String,
        size: Decimal,
        price: Option<Decimal>,
        stop_loss: Option<Decimal>,
        take_profit: Option<Decimal>,
    },
    /// Sell signal with target position size
    Sell {
        symbol: String,
        size: Decimal,
        price: Option<Decimal>,
        stop_loss: Option<Decimal>,
        take_profit: Option<Decimal>,
    },
    /// Close existing position
    Close { symbol: String },
    /// No action
    None,
}

impl Signal {
    /// Check if this is an actionable signal
    pub fn is_actionable(&self) -> bool {
        !matches!(self, Signal::None)
    }

    /// Get the symbol if applicable
    pub fn symbol(&self) -> Option<&str> {
        match self {
            Signal::Buy { symbol, .. } => Some(symbol),
            Signal::Sell { symbol, .. } => Some(symbol),
            Signal::Close { symbol } => Some(symbol),
            Signal::None => None,
        }
    }
}

/// Trait for trading strategies
pub trait Strategy: Send + Sync {
    /// Process a market event and generate trading signals
    fn on_event(&mut self, event: &MarketEvent) -> Vec<Signal>;

    /// Called when the simulation starts
    fn on_start(&mut self) {}

    /// Called when the simulation stops
    fn on_stop(&mut self) {}

    /// Get strategy name
    fn name(&self) -> &str;

    /// Reset strategy state
    fn reset(&mut self) {}
}

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl std::fmt::Display for OrderSide {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderSide::Buy => write!(f, "buy"),
            OrderSide::Sell => write!(f, "sell"),
        }
    }
}

/// A simulated or real trade execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeExecution {
    /// Unique trade ID
    pub id: String,
    /// Trading symbol
    pub symbol: String,
    /// Order side
    pub side: OrderSide,
    /// Executed quantity
    pub quantity: Decimal,
    /// Execution price
    pub price: Decimal,
    /// Commission paid
    pub commission: Decimal,
    /// Slippage from requested price
    pub slippage: Decimal,
    /// Execution timestamp
    pub timestamp: DateTime<Utc>,
    /// Exchange where executed
    pub exchange: String,
}

impl TradeExecution {
    /// Calculate the total cost (for buys) or proceeds (for sells)
    pub fn total_value(&self) -> Decimal {
        self.quantity * self.price
    }

    /// Calculate net value after commission
    pub fn net_value(&self) -> Decimal {
        match self.side {
            OrderSide::Buy => self.total_value() + self.commission,
            OrderSide::Sell => self.total_value() - self.commission,
        }
    }
}

/// Position tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Trading symbol
    pub symbol: String,
    /// Position size (positive = long, negative = short)
    pub size: Decimal,
    /// Average entry price
    pub entry_price: Decimal,
    /// Current market price
    pub current_price: Decimal,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Realized P&L for this position
    pub realized_pnl: Decimal,
    /// Entry timestamp
    pub entry_time: DateTime<Utc>,
    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

impl Position {
    /// Create a new position
    pub fn new(symbol: &str, size: Decimal, entry_price: Decimal) -> Self {
        Self {
            symbol: symbol.to_string(),
            size,
            entry_price,
            current_price: entry_price,
            unrealized_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            entry_time: Utc::now(),
            last_update: Utc::now(),
        }
    }

    /// Update position with new market price
    pub fn update_price(&mut self, price: Decimal) {
        self.current_price = price;
        self.unrealized_pnl = (price - self.entry_price) * self.size;
        self.last_update = Utc::now();
    }

    /// Check if position is long
    pub fn is_long(&self) -> bool {
        self.size > Decimal::ZERO
    }

    /// Check if position is short
    pub fn is_short(&self) -> bool {
        self.size < Decimal::ZERO
    }

    /// Get absolute position size
    pub fn abs_size(&self) -> Decimal {
        self.size.abs()
    }
}

/// Account state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    /// Current cash balance
    pub balance: Decimal,
    /// Initial balance
    pub initial_balance: Decimal,
    /// Open positions
    pub positions: HashMap<String, Position>,
    /// Total realized P&L
    pub realized_pnl: Decimal,
    /// Total unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Total commissions paid
    pub total_commission: Decimal,
    /// Peak equity (for drawdown calculation)
    pub peak_equity: Decimal,
    /// Current drawdown
    pub current_drawdown: Decimal,
    /// Maximum drawdown seen
    pub max_drawdown: Decimal,
}

impl Account {
    /// Create a new account with initial balance
    pub fn new(initial_balance: Decimal) -> Self {
        Self {
            balance: initial_balance,
            initial_balance,
            positions: HashMap::new(),
            realized_pnl: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            peak_equity: initial_balance,
            current_drawdown: Decimal::ZERO,
            max_drawdown: Decimal::ZERO,
        }
    }

    /// Calculate total equity (balance + unrealized P&L)
    pub fn equity(&self) -> Decimal {
        self.balance + self.unrealized_pnl
    }

    /// Update unrealized P&L from positions
    pub fn update_unrealized_pnl(&mut self) {
        self.unrealized_pnl = self.positions.values().map(|p| p.unrealized_pnl).sum();

        // Update drawdown tracking
        let equity = self.equity();
        if equity > self.peak_equity {
            self.peak_equity = equity;
        }

        if self.peak_equity > Decimal::ZERO {
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity;
            if self.current_drawdown > self.max_drawdown {
                self.max_drawdown = self.current_drawdown;
            }
        }
    }

    /// Get return percentage
    pub fn return_pct(&self) -> Decimal {
        if self.initial_balance > Decimal::ZERO {
            (self.equity() - self.initial_balance) / self.initial_balance * Decimal::from(100)
        } else {
            Decimal::ZERO
        }
    }
}

/// Simulation result metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimResult {
    /// Simulation mode
    pub mode: String,
    /// Strategy name
    pub strategy_name: String,
    /// Start time
    pub start_time: DateTime<Utc>,
    /// End time
    pub end_time: DateTime<Utc>,
    /// Duration in seconds
    pub duration_seconds: i64,
    /// Initial balance
    pub initial_balance: Decimal,
    /// Final balance
    pub final_balance: Decimal,
    /// Final equity (balance + unrealized P&L)
    pub final_equity: Decimal,
    /// Total return percentage
    pub total_return_pct: Decimal,
    /// Total P&L
    pub total_pnl: Decimal,
    /// Realized P&L
    pub realized_pnl: Decimal,
    /// Unrealized P&L
    pub unrealized_pnl: Decimal,
    /// Total trades executed
    pub total_trades: u64,
    /// Winning trades
    pub winning_trades: u64,
    /// Losing trades
    pub losing_trades: u64,
    /// Win rate percentage
    pub win_rate: f64,
    /// Maximum drawdown percentage
    pub max_drawdown_pct: Decimal,
    /// Total commission paid
    pub total_commission: Decimal,
    /// Sharpe ratio (if calculable)
    pub sharpe_ratio: Option<f64>,
    /// Profit factor (gross profit / gross loss)
    pub profit_factor: Option<f64>,
    /// Average win
    pub avg_win: Option<Decimal>,
    /// Average loss
    pub avg_loss: Option<Decimal>,
    /// Largest win
    pub largest_win: Option<Decimal>,
    /// Largest loss
    pub largest_loss: Option<Decimal>,
    /// Total market events processed
    pub events_processed: u64,
    /// Symbols traded
    pub symbols_traded: Vec<String>,
    /// Final positions
    pub final_positions: Vec<Position>,
    /// All trade executions
    pub trades: Vec<TradeExecution>,
}

impl Default for SimResult {
    fn default() -> Self {
        Self {
            mode: String::new(),
            strategy_name: String::new(),
            start_time: Utc::now(),
            end_time: Utc::now(),
            duration_seconds: 0,
            initial_balance: Decimal::ZERO,
            final_balance: Decimal::ZERO,
            final_equity: Decimal::ZERO,
            total_return_pct: Decimal::ZERO,
            total_pnl: Decimal::ZERO,
            realized_pnl: Decimal::ZERO,
            unrealized_pnl: Decimal::ZERO,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            max_drawdown_pct: Decimal::ZERO,
            total_commission: Decimal::ZERO,
            sharpe_ratio: None,
            profit_factor: None,
            avg_win: None,
            avg_loss: None,
            largest_win: None,
            largest_loss: None,
            events_processed: 0,
            symbols_traded: Vec::new(),
            final_positions: Vec::new(),
            trades: Vec::new(),
        }
    }
}

impl SimResult {
    /// Create a summary string
    pub fn summary(&self) -> String {
        format!(
            "SimResult {{ mode: {}, strategy: {}, duration: {}s, trades: {}, return: {:.2}%, max_dd: {:.2}%, win_rate: {:.1}% }}",
            self.mode,
            self.strategy_name,
            self.duration_seconds,
            self.total_trades,
            self.total_return_pct.to_f64().unwrap_or(0.0),
            self.max_drawdown_pct.to_f64().unwrap_or(0.0) * 100.0,
            self.win_rate
        )
    }
}

/// Simulation environment - main entry point
pub struct SimEnvironment {
    /// Configuration
    config: SimConfig,
    /// Account state
    account: Arc<RwLock<Account>>,
    /// Data feed
    data_feed: Arc<RwLock<AggregatedDataFeed>>,
    /// Data recorder (optional)
    recorder: Option<DataRecorder>,
    /// Trade history
    trades: Arc<RwLock<Vec<TradeExecution>>>,
    /// Running flag
    running: Arc<AtomicBool>,
    /// Events processed counter
    events_processed: Arc<AtomicU64>,
    /// Simulation start time
    start_time: Option<DateTime<Utc>>,
    /// Trade counter for ID generation
    trade_counter: Arc<AtomicU64>,
}

impl SimEnvironment {
    /// Create a new simulation environment
    pub async fn new(config: SimConfig) -> Result<Self, SimError> {
        // Validate configuration
        config
            .validate()
            .map_err(|e| SimError::Config(e.to_string()))?;

        let initial_balance = config.initial_balance;

        // Initialize data feed
        let data_feed = Arc::new(RwLock::new(AggregatedDataFeed::new(&format!(
            "{}_feed",
            config.mode
        ))));

        // Initialize recorder if enabled
        let recorder = if config.recording.enabled {
            let rec_config = RecorderConfig::new(
                &config.recording.questdb_host,
                config.recording.questdb_port,
            )
            .with_record_ticks(config.recording.record_ticks)
            .with_record_trades(config.recording.record_trades)
            .with_record_orderbook(config.recording.record_orderbook);

            Some(DataRecorder::new(rec_config))
        } else {
            None
        };

        Ok(Self {
            config,
            account: Arc::new(RwLock::new(Account::new(initial_balance))),
            data_feed,
            recorder,
            trades: Arc::new(RwLock::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            events_processed: Arc::new(AtomicU64::new(0)),
            start_time: None,
            trade_counter: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Initialize the environment (connect data feeds, etc.)
    pub async fn initialize(&mut self) -> Result<(), SimError> {
        info!(
            "Initializing simulation environment in {:?} mode",
            self.config.mode
        );

        // Start recorder if enabled
        if let Some(ref mut recorder) = self.recorder {
            recorder
                .start()
                .await
                .map_err(|e| SimError::DataFeed(e.to_string()))?;
            info!("Data recorder started");
        }

        // Initialize data feed based on mode
        match self.config.mode {
            SimMode::Backtest => {
                info!("Backtest mode - data will be loaded from source");
                // Data loading happens during run_backtest
            }
            SimMode::ForwardTest | SimMode::Live => {
                info!("Live/Forward test mode - connecting to exchanges");
                // Exchange connections will be set up externally
            }
        }

        Ok(())
    }

    /// Run a backtest with the given strategy
    pub async fn run_backtest<S: Strategy>(
        &mut self,
        strategy: &mut S,
    ) -> Result<SimResult, SimError> {
        if self.config.mode != SimMode::Backtest {
            return Err(SimError::Config("Not in backtest mode".to_string()));
        }

        info!("Starting backtest with strategy: {}", strategy.name());

        self.running.store(true, Ordering::SeqCst);
        self.start_time = Some(Utc::now());
        strategy.on_start();

        // The actual backtest implementation would load data from the data source
        // and replay it through the strategy. For now, this is a skeleton.

        let result = self.build_result(strategy.name()).await;

        strategy.on_stop();
        self.running.store(false, Ordering::SeqCst);

        info!("Backtest completed: {}", result.summary());
        Ok(result)
    }

    /// Run forward testing (paper trading) with live data
    pub async fn run_forward_test<S: Strategy>(
        &mut self,
        strategy: &mut S,
        duration: std::time::Duration,
    ) -> Result<SimResult, SimError> {
        if self.config.mode != SimMode::ForwardTest {
            return Err(SimError::Config("Not in forward test mode".to_string()));
        }

        info!(
            "Starting forward test with strategy: {}, duration: {:?}",
            strategy.name(),
            duration
        );

        self.running.store(true, Ordering::SeqCst);
        self.start_time = Some(Utc::now());
        strategy.on_start();

        // Subscribe to data feed events
        let mut event_rx = self.data_feed.read().subscribe();
        let deadline = tokio::time::Instant::now() + duration;

        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => {
                    info!("Forward test duration reached");
                    break;
                }
                event = event_rx.recv() => {
                    match event {
                        Ok(market_event) => {
                            self.process_event(&market_event, strategy).await?;
                        }
                        Err(broadcast::error::RecvError::Lagged(n)) => {
                            warn!("Lagged {} events", n);
                        }
                        Err(broadcast::error::RecvError::Closed) => {
                            info!("Data feed closed");
                            break;
                        }
                    }
                }
            }

            if !self.running.load(Ordering::SeqCst) {
                break;
            }
        }

        let result = self.build_result(strategy.name()).await;

        strategy.on_stop();
        self.running.store(false, Ordering::SeqCst);

        info!("Forward test completed: {}", result.summary());
        Ok(result)
    }

    /// Process a market event through the strategy
    async fn process_event<S: Strategy>(
        &self,
        event: &MarketEvent,
        strategy: &mut S,
    ) -> Result<(), SimError> {
        self.events_processed.fetch_add(1, Ordering::Relaxed);

        // Update position prices if this is a tick
        if let MarketEvent::Tick(tick) = event {
            self.update_position_price(&tick.symbol, tick.mid_price());
        }

        // Record event if recorder is enabled
        if let Some(ref recorder) = self.recorder {
            let _ = recorder.record_event(event).await;
        }

        // Get signals from strategy
        let signals = strategy.on_event(event);

        // Execute signals
        for signal in signals {
            if signal.is_actionable() {
                self.execute_signal(&signal, event).await?;
            }
        }

        Ok(())
    }

    /// Execute a trading signal
    async fn execute_signal(&self, signal: &Signal, event: &MarketEvent) -> Result<(), SimError> {
        match signal {
            Signal::Buy {
                symbol,
                size,
                price,
                ..
            } => {
                let exec_price = price.unwrap_or_else(|| {
                    // Use current ask price
                    if let MarketEvent::Tick(tick) = event {
                        tick.ask_price
                    } else {
                        Decimal::ZERO
                    }
                });

                if exec_price > Decimal::ZERO {
                    self.execute_trade(symbol, OrderSide::Buy, *size, exec_price)
                        .await?;
                }
            }
            Signal::Sell {
                symbol,
                size,
                price,
                ..
            } => {
                let exec_price = price.unwrap_or_else(|| {
                    // Use current bid price
                    if let MarketEvent::Tick(tick) = event {
                        tick.bid_price
                    } else {
                        Decimal::ZERO
                    }
                });

                if exec_price > Decimal::ZERO {
                    self.execute_trade(symbol, OrderSide::Sell, *size, exec_price)
                        .await?;
                }
            }
            Signal::Close { symbol } => {
                self.close_position(symbol).await?;
            }
            Signal::None => {}
        }

        Ok(())
    }

    /// Execute a trade
    async fn execute_trade(
        &self,
        symbol: &str,
        side: OrderSide,
        size: Decimal,
        price: Decimal,
    ) -> Result<TradeExecution, SimError> {
        // Calculate slippage
        let slippage = if self.config.execution.enable_slippage {
            let slippage_bps =
                Decimal::try_from(self.config.execution.slippage_bps).unwrap_or(Decimal::ZERO);
            price * slippage_bps / Decimal::from(10_000)
        } else {
            Decimal::ZERO
        };

        // Apply slippage (worse price for the trader)
        let exec_price = match side {
            OrderSide::Buy => price + slippage,
            OrderSide::Sell => price - slippage,
        };

        // Calculate commission
        let commission_bps =
            Decimal::try_from(self.config.execution.commission_bps).unwrap_or(Decimal::ZERO);
        let commission = (size * exec_price) * commission_bps / Decimal::from(10_000);

        // Create trade execution
        let trade_id = self.trade_counter.fetch_add(1, Ordering::SeqCst);
        let execution = TradeExecution {
            id: format!("trade_{}", trade_id),
            symbol: symbol.to_string(),
            side,
            quantity: size,
            price: exec_price,
            commission,
            slippage,
            timestamp: Utc::now(),
            exchange: self.config.exchanges.first().cloned().unwrap_or_default(),
        };

        // Update account
        {
            let mut account = self.account.write();

            // Deduct/add balance
            let value = execution.net_value();
            match side {
                OrderSide::Buy => {
                    account.balance -= value;
                }
                OrderSide::Sell => {
                    account.balance += value - commission;
                }
            }

            account.total_commission += commission;

            // Update or create position
            let position_delta = match side {
                OrderSide::Buy => size,
                OrderSide::Sell => -size,
            };

            // First, compute what changes we need to make based on current position state
            enum PositionAction {
                None,
                Create,
                Flip {
                    realized: Decimal,
                    new_size: Decimal,
                },
                Close {
                    realized: Decimal,
                },
                Increase {
                    new_entry: Decimal,
                    new_size: Decimal,
                },
                Reduce {
                    realized: Decimal,
                    new_size: Decimal,
                },
            }

            let action = if let Some(pos) = account.positions.get(symbol) {
                let old_size = pos.size;
                let new_size = old_size + position_delta;

                if new_size.signum() != old_size.signum() && new_size != Decimal::ZERO {
                    // Position flipped - realize old P&L
                    let realized = (exec_price - pos.entry_price) * old_size;
                    PositionAction::Flip { realized, new_size }
                } else if new_size == Decimal::ZERO {
                    // Position closed
                    let realized = (exec_price - pos.entry_price) * old_size;
                    PositionAction::Close { realized }
                } else {
                    // Position increased/decreased
                    if position_delta.signum() == old_size.signum() {
                        // Adding to position - update average price
                        let total_cost = (pos.entry_price * old_size.abs())
                            + (exec_price * position_delta.abs());
                        let new_entry = total_cost / new_size.abs();
                        PositionAction::Increase {
                            new_entry,
                            new_size,
                        }
                    } else {
                        // Reducing position - realize partial P&L
                        let realized = (exec_price - pos.entry_price) * position_delta.abs();
                        PositionAction::Reduce { realized, new_size }
                    }
                }
            } else if position_delta != Decimal::ZERO {
                PositionAction::Create
            } else {
                PositionAction::None
            };

            // Now apply the action with separate borrows
            match action {
                PositionAction::None => {}
                PositionAction::Create => {
                    account.positions.insert(
                        symbol.to_string(),
                        Position::new(symbol, position_delta, exec_price),
                    );
                }
                PositionAction::Flip { realized, new_size } => {
                    account.realized_pnl += realized;
                    if let Some(pos) = account.positions.get_mut(symbol) {
                        pos.realized_pnl += realized;
                        pos.size = new_size;
                        pos.entry_price = exec_price;
                    }
                }
                PositionAction::Close { realized } => {
                    account.realized_pnl += realized;
                    account.positions.remove(symbol);
                }
                PositionAction::Increase {
                    new_entry,
                    new_size,
                } => {
                    if let Some(pos) = account.positions.get_mut(symbol) {
                        pos.entry_price = new_entry;
                        pos.size = new_size;
                    }
                }
                PositionAction::Reduce { realized, new_size } => {
                    account.realized_pnl += realized;
                    if let Some(pos) = account.positions.get_mut(symbol) {
                        pos.realized_pnl += realized;
                        pos.size = new_size;
                    }
                }
            }

            account.update_unrealized_pnl();
        }

        // Store trade
        self.trades.write().push(execution.clone());

        debug!(
            "Executed trade: {} {} {} @ {} (commission: {}, slippage: {})",
            execution.side, execution.quantity, symbol, execution.price, commission, slippage
        );

        Ok(execution)
    }

    /// Close a position
    async fn close_position(&self, symbol: &str) -> Result<Option<TradeExecution>, SimError> {
        let position = {
            let account = self.account.read();
            account.positions.get(symbol).cloned()
        };

        if let Some(pos) = position {
            let side = if pos.is_long() {
                OrderSide::Sell
            } else {
                OrderSide::Buy
            };

            let execution = self
                .execute_trade(symbol, side, pos.abs_size(), pos.current_price)
                .await?;
            Ok(Some(execution))
        } else {
            Ok(None)
        }
    }

    /// Update position price
    fn update_position_price(&self, symbol: &str, price: Decimal) {
        let mut account = self.account.write();
        if let Some(pos) = account.positions.get_mut(symbol) {
            pos.update_price(price);
        }
        account.update_unrealized_pnl();
    }

    /// Build the simulation result
    async fn build_result(&self, strategy_name: &str) -> SimResult {
        let account = self.account.read();
        let trades = self.trades.read();

        let end_time = Utc::now();
        let start_time = self.start_time.unwrap_or(end_time);
        let duration = end_time - start_time;

        // Calculate trade statistics
        // Note: These are placeholders - a more sophisticated implementation would
        // track P&L per round-trip trade
        let winning_trades = 0u64;
        let losing_trades = 0u64;
        let gross_profit = Decimal::ZERO;
        let gross_loss = Decimal::ZERO;
        let largest_win: Option<Decimal> = None;
        let largest_loss: Option<Decimal> = None;
        let returns: Vec<f64> = Vec::new();

        let total_trades = trades.len() as u64;
        let win_rate = if total_trades > 0 {
            (winning_trades as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        let profit_factor = if gross_loss > Decimal::ZERO {
            Some((gross_profit / gross_loss).to_f64().unwrap_or(0.0))
        } else if gross_profit > Decimal::ZERO {
            Some(f64::INFINITY)
        } else {
            None
        };

        // Calculate Sharpe ratio (simplified)
        let sharpe_ratio = if returns.len() > 1 {
            let mean: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                / (returns.len() - 1) as f64;
            let std_dev = variance.sqrt();
            if std_dev > 0.0 {
                Some(mean / std_dev * (252.0_f64).sqrt()) // Annualized
            } else {
                None
            }
        } else {
            None
        };

        let symbols_traded: Vec<String> = trades
            .iter()
            .map(|t| t.symbol.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        SimResult {
            mode: self.config.mode.to_string(),
            strategy_name: strategy_name.to_string(),
            start_time,
            end_time,
            duration_seconds: duration.num_seconds(),
            initial_balance: account.initial_balance,
            final_balance: account.balance,
            final_equity: account.equity(),
            total_return_pct: account.return_pct(),
            total_pnl: account.equity() - account.initial_balance,
            realized_pnl: account.realized_pnl,
            unrealized_pnl: account.unrealized_pnl,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            max_drawdown_pct: account.max_drawdown,
            total_commission: account.total_commission,
            sharpe_ratio,
            profit_factor,
            avg_win: if winning_trades > 0 {
                Some(gross_profit / Decimal::from(winning_trades))
            } else {
                None
            },
            avg_loss: if losing_trades > 0 {
                Some(gross_loss / Decimal::from(losing_trades))
            } else {
                None
            },
            largest_win,
            largest_loss,
            events_processed: self.events_processed.load(Ordering::Relaxed),
            symbols_traded,
            final_positions: account.positions.values().cloned().collect(),
            trades: trades.clone(),
        }
    }

    /// Get current account state
    pub fn account(&self) -> Account {
        self.account.read().clone()
    }

    /// Get data feed for external event publishing
    pub fn data_feed(&self) -> Arc<RwLock<AggregatedDataFeed>> {
        self.data_feed.clone()
    }

    /// Publish a market event to the data feed
    pub fn publish_event(&self, event: MarketEvent) {
        self.data_feed.read().publish(event);
    }

    /// Check if environment is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Stop the environment
    pub async fn stop(&mut self) -> Result<(), SimError> {
        info!("Stopping simulation environment");
        self.running.store(false, Ordering::SeqCst);

        if let Some(ref mut recorder) = self.recorder {
            recorder
                .stop()
                .await
                .map_err(|e| SimError::DataFeed(e.to_string()))?;
        }

        Ok(())
    }

    /// Get configuration
    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    /// Get trade history
    pub fn trades(&self) -> Vec<TradeExecution> {
        self.trades.read().clone()
    }

    /// Get events processed count
    pub fn events_processed(&self) -> u64 {
        self.events_processed.load(Ordering::Relaxed)
    }

    /// Reset the environment for a new run
    pub fn reset(&mut self) {
        *self.account.write() = Account::new(self.config.initial_balance);
        self.trades.write().clear();
        self.events_processed.store(0, Ordering::Relaxed);
        self.start_time = None;
        self.trade_counter.store(0, Ordering::Relaxed);
    }

    /// Connect to live exchange data via a MarketDataAggregator
    ///
    /// This creates a LiveFeedBridge that converts `MarketDataEvent` from exchange
    /// providers to `MarketEvent` for the simulation environment, enabling
    /// forward testing with real market data.
    ///
    /// # Arguments
    ///
    /// * `aggregator` - The MarketDataAggregator with connected exchange providers
    /// * `bridge_config` - Optional configuration for the LiveFeedBridge
    ///
    /// # Returns
    ///
    /// Returns the LiveFeedBridge for monitoring statistics and stopping
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use janus_execution::exchanges::{MarketDataAggregator, create_kraken_provider};
    /// use janus_execution::sim::{SimEnvironment, SimConfig, LiveFeedBridgeConfig};
    ///
    /// // Create and configure aggregator
    /// let mut aggregator = MarketDataAggregator::new();
    /// aggregator.add_provider(create_kraken_provider(None)?);
    /// aggregator.connect_all().await?;
    /// aggregator.subscribe_ticker_all(&["BTC/USDT"]).await?;
    ///
    /// // Create environment and connect to live data
    /// let config = SimConfig::forward_test();
    /// let mut env = SimEnvironment::new(config).await?;
    /// let bridge = env.connect_live_data(&aggregator, None).await?;
    ///
    /// // Run forward test with real market data
    /// let result = env.run_forward_test(&mut strategy, Duration::from_secs(60)).await?;
    ///
    /// // Check bridge statistics
    /// let stats = bridge.stats();
    /// println!("Events received: {}", stats.events_received);
    /// ```
    pub async fn connect_live_data(
        &self,
        aggregator: &MarketDataAggregator,
        bridge_config: Option<LiveFeedBridgeConfig>,
    ) -> Result<Arc<LiveFeedBridge>, SimError> {
        if self.config.mode != SimMode::ForwardTest && self.config.mode != SimMode::Live {
            return Err(SimError::Config(
                "connect_live_data requires ForwardTest or Live mode".to_string(),
            ));
        }

        info!("Connecting to live market data via LiveFeedBridge");

        // Create bridge with config or defaults
        let config = bridge_config.unwrap_or_else(|| {
            LiveFeedBridgeConfig::default()
                .with_normalize_symbols(true)
                .with_filter_invalid_ticks(true)
        });

        // Create a new AggregatedDataFeed that shares the same broadcast channel
        // We need to create a bridge that publishes to our existing data_feed
        let bridge_feed = {
            // Get a reference to the actual feed and create a wrapper
            let feed_read = self.data_feed.read();
            // We'll create the bridge with a reference to our feed's sender
            Arc::new(AggregatedDataFeed::with_sender(feed_read.sender()))
        };

        let bridge = Arc::new(LiveFeedBridge::with_config(bridge_feed, config));

        // Connect the bridge to the aggregator
        bridge
            .connect_to_aggregator(aggregator)
            .await
            .map_err(|e| SimError::Exchange(e.to_string()))?;

        info!("LiveFeedBridge connected to MarketDataAggregator");

        Ok(bridge)
    }

    /// Connect to live exchange data with a custom LiveFeedBridge
    ///
    /// This allows for more advanced setups where you want to manage
    /// the LiveFeedBridge lifecycle separately.
    ///
    /// # Arguments
    ///
    /// * `bridge` - A pre-configured LiveFeedBridge
    ///
    /// # Returns
    ///
    /// Returns the bridge for monitoring (same instance passed in)
    pub fn attach_live_feed_bridge(&self, bridge: Arc<LiveFeedBridge>) -> Arc<LiveFeedBridge> {
        info!("Attached external LiveFeedBridge to SimEnvironment");
        bridge
    }

    /// Create a LiveFeedBridge that publishes to this environment's data feed
    ///
    /// Use this when you want to connect to individual providers rather than
    /// through a MarketDataAggregator.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let bridge = env.create_live_feed_bridge(None);
    /// bridge.connect_to_provider(kraken_provider).await?;
    /// bridge.connect_to_provider(binance_provider).await?;
    /// ```
    pub fn create_live_feed_bridge(
        &self,
        config: Option<LiveFeedBridgeConfig>,
    ) -> Arc<LiveFeedBridge> {
        let bridge_config = config.unwrap_or_else(|| {
            LiveFeedBridgeConfig::default()
                .with_normalize_symbols(true)
                .with_filter_invalid_ticks(true)
        });

        // Create a feed that shares our broadcast channel
        let bridge_feed = {
            let feed_read = self.data_feed.read();
            Arc::new(AggregatedDataFeed::with_sender(feed_read.sender()))
        };

        Arc::new(LiveFeedBridge::with_config(bridge_feed, bridge_config))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[allow(dead_code)]
    struct TestStrategy {
        name: String,
        signals: Vec<Signal>,
        signal_index: usize,
    }

    impl TestStrategy {
        #[allow(dead_code)]
        fn new(name: &str, signals: Vec<Signal>) -> Self {
            Self {
                name: name.to_string(),
                signals,
                signal_index: 0,
            }
        }
    }

    impl Strategy for TestStrategy {
        fn on_event(&mut self, _event: &MarketEvent) -> Vec<Signal> {
            if self.signal_index < self.signals.len() {
                let signal = self.signals[self.signal_index].clone();
                self.signal_index += 1;
                vec![signal]
            } else {
                vec![Signal::None]
            }
        }

        fn name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_position() {
        let mut pos = Position::new("BTC/USDT", dec!(1.0), dec!(50000.0));

        assert!(pos.is_long());
        assert!(!pos.is_short());
        assert_eq!(pos.unrealized_pnl, Decimal::ZERO);

        pos.update_price(dec!(51000.0));
        assert_eq!(pos.unrealized_pnl, dec!(1000.0));

        pos.update_price(dec!(49000.0));
        assert_eq!(pos.unrealized_pnl, dec!(-1000.0));
    }

    #[test]
    fn test_account() {
        let mut account = Account::new(dec!(10000.0));

        assert_eq!(account.equity(), dec!(10000.0));
        assert_eq!(account.return_pct(), Decimal::ZERO);

        // Add a position
        account.positions.insert(
            "BTC/USDT".to_string(),
            Position::new("BTC/USDT", dec!(0.1), dec!(50000.0)),
        );

        // Update price
        if let Some(pos) = account.positions.get_mut("BTC/USDT") {
            pos.update_price(dec!(51000.0));
        }
        account.update_unrealized_pnl();

        assert_eq!(account.unrealized_pnl, dec!(100.0));
        assert_eq!(account.equity(), dec!(10100.0));
    }

    #[test]
    fn test_signal() {
        let buy = Signal::Buy {
            symbol: "BTC/USDT".to_string(),
            size: dec!(1.0),
            price: Some(dec!(50000.0)),
            stop_loss: None,
            take_profit: None,
        };

        assert!(buy.is_actionable());
        assert_eq!(buy.symbol(), Some("BTC/USDT"));

        assert!(!Signal::None.is_actionable());
        assert_eq!(Signal::None.symbol(), None);
    }

    #[test]
    fn test_trade_execution() {
        let trade = TradeExecution {
            id: "test_1".to_string(),
            symbol: "BTC/USDT".to_string(),
            side: OrderSide::Buy,
            quantity: dec!(0.1),
            price: dec!(50000.0),
            commission: dec!(5.0),
            slippage: dec!(2.5),
            timestamp: Utc::now(),
            exchange: "kraken".to_string(),
        };

        assert_eq!(trade.total_value(), dec!(5000.0));
        assert_eq!(trade.net_value(), dec!(5005.0)); // Buy includes commission
    }

    #[test]
    fn test_sim_result_summary() {
        let mut result = SimResult::default();
        result.mode = "backtest".to_string();
        result.strategy_name = "test".to_string();
        result.duration_seconds = 3600;
        result.total_trades = 10;
        result.total_return_pct = dec!(5.5);
        result.max_drawdown_pct = dec!(0.02);
        result.win_rate = 60.0;

        let summary = result.summary();
        assert!(summary.contains("backtest"));
        assert!(summary.contains("test"));
        assert!(summary.contains("10"));
    }

    #[tokio::test]
    async fn test_sim_environment_creation() {
        use std::path::PathBuf;

        let config = SimConfig::backtest()
            .with_data_source(super::super::config::DataSource::Parquet(PathBuf::from(
                "test.parquet",
            )))
            .with_initial_balance(10000.0)
            .with_symbols(vec!["BTC/USDT".to_string()])
            .build_unchecked();

        let env = SimEnvironment::new(config).await.unwrap();

        assert!(!env.is_running());
        assert_eq!(env.account().initial_balance, dec!(10000));
    }

    #[tokio::test]
    async fn test_connect_live_data_requires_forward_test_mode() {
        use std::path::PathBuf;

        // Create a backtest config - should fail connect_live_data
        let config = SimConfig::backtest()
            .with_data_source(super::super::config::DataSource::Parquet(PathBuf::from(
                "test.parquet",
            )))
            .with_initial_balance(10000.0)
            .build_unchecked();

        let env = SimEnvironment::new(config).await.unwrap();
        let aggregator = MarketDataAggregator::new();

        // Should fail because we're in backtest mode
        let result = env.connect_live_data(&aggregator, None).await;
        assert!(result.is_err());
        if let Err(SimError::Config(msg)) = result {
            assert!(msg.contains("ForwardTest or Live mode"));
        } else {
            panic!("Expected Config error");
        }
    }

    #[tokio::test]
    async fn test_create_live_feed_bridge() {
        let config = SimConfig::forward_test()
            .with_initial_balance(10000.0)
            .with_symbols(vec!["BTC/USDT".to_string()])
            .build_unchecked();

        let env = SimEnvironment::new(config).await.unwrap();

        // Should be able to create a bridge
        let bridge = env.create_live_feed_bridge(None);
        assert!(!bridge.is_running());
    }

    #[tokio::test]
    async fn test_create_live_feed_bridge_with_config() {
        let config = SimConfig::forward_test()
            .with_initial_balance(10000.0)
            .with_symbols(vec!["BTC/USDT".to_string()])
            .build_unchecked();

        let env = SimEnvironment::new(config).await.unwrap();

        // Create bridge with custom config
        let bridge_config = LiveFeedBridgeConfig::default()
            .with_normalize_symbols(true)
            .with_filter_invalid_ticks(true)
            .with_tick_debounce_ms(100);

        let bridge = env.create_live_feed_bridge(Some(bridge_config));
        assert!(!bridge.is_running());
    }

    #[tokio::test]
    async fn test_connect_live_data_forward_test_mode() {
        let config = SimConfig::forward_test()
            .with_initial_balance(10000.0)
            .with_symbols(vec!["BTC/USDT".to_string()])
            .build_unchecked();

        let env = SimEnvironment::new(config).await.unwrap();
        let aggregator = MarketDataAggregator::new();

        // Should succeed in forward test mode (even with no providers)
        let result = env.connect_live_data(&aggregator, None).await;
        assert!(result.is_ok());

        let bridge = result.unwrap();
        assert!(bridge.is_running());

        // Clean up
        bridge.stop().await;
    }
}
