//! Signal Flow Coordinator
//!
//! This module coordinates the flow of trading signals from Janus to execution,
//! integrating with the Kraken execution engine and fill tracker for real-time
//! order management.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌──────────────────────┐     ┌─────────────────────┐
//! │   Janus     │────▶│  SignalFlowCoordinator│────▶│  KrakenExecution    │
//! │  (Forward)  │     │                      │     │     Engine          │
//! └─────────────┘     └──────────────────────┘     └─────────────────────┘
//!                              │                            │
//!                              │                            ▼
//!                              │                   ┌─────────────────────┐
//!                              │                   │  KrakenFillTracker  │
//!                              │                   │   (Private WS)      │
//!                              ◀───────────────────┴─────────────────────┘
//!                           (Fill Updates)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_execution::execution::signal_flow::{SignalFlowCoordinator, SignalFlowConfig};
//!
//! let config = SignalFlowConfig::from_env();
//! let coordinator = SignalFlowCoordinator::new(config).await?;
//!
//! // Start the coordinator
//! coordinator.start().await?;
//!
//! // Submit a signal
//! let response = coordinator.submit_signal(signal).await?;
//! println!("Order submitted: {}", response.order_id);
//!
//! // Subscribe to execution updates
//! let mut rx = coordinator.subscribe_updates();
//! while let Ok(update) = rx.recv().await {
//!     println!("Update: {:?}", update);
//! }
//! ```

use crate::error::{ExecutionError, Result};
use crate::exchanges::kraken::fill_tracker::{FillTrackerConfig, KrakenFillTracker};
use crate::execution::best_execution::{
    BestExecutionAnalyzer, ExecutionConfig as BestExecutionConfig, ExecutionRecommendation,
    OrderSide as BestExecOrderSide,
};
use crate::execution::histogram::global_latency_histograms;
use crate::execution::kraken::{KrakenExecutionConfig, KrakenExecutionEngine};
use crate::execution::metrics::{RecommendationType, SignalFlowMetrics, signal_flow_metrics};
use crate::types::{Fill, Order, OrderSide, OrderStatusEnum, OrderTypeEnum, Position};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

// ============================================================================
// Configuration
// ============================================================================

/// Signal flow coordinator configuration
#[derive(Debug, Clone)]
pub struct SignalFlowConfig {
    /// Kraken execution engine configuration
    pub execution_config: KrakenExecutionConfig,
    /// Fill tracker configuration
    pub fill_tracker_config: FillTrackerConfig,
    /// Best execution analyzer configuration
    pub best_execution_config: BestExecutionConfig,
    /// Enable real-time fill tracking via WebSocket
    pub enable_fill_tracking: bool,
    /// Enable best execution analysis before order submission
    pub enable_best_execution: bool,
    /// Default exchange for orders
    pub default_exchange: String,
    /// Minimum signal confidence threshold
    pub min_confidence: f64,
    /// Minimum signal strength threshold
    pub min_strength: f64,
    /// Maximum position size per symbol
    pub max_position_size: Decimal,
    /// Risk limit per trade as percentage of balance
    pub risk_limit_pct: Decimal,
    /// Block orders when best execution recommends Avoid
    pub block_on_avoid: bool,
    /// Use limit orders when best execution recommends UseLimitOrder
    pub use_limit_orders: bool,
}

impl Default for SignalFlowConfig {
    fn default() -> Self {
        Self {
            execution_config: KrakenExecutionConfig::default(),
            fill_tracker_config: FillTrackerConfig::default(),
            best_execution_config: BestExecutionConfig::default(),
            enable_fill_tracking: true,
            enable_best_execution: true,
            default_exchange: "kraken".to_string(),
            min_confidence: 0.5,
            min_strength: 0.5,
            max_position_size: Decimal::from(10),
            risk_limit_pct: Decimal::from(2), // 2% per trade
            block_on_avoid: true,
            use_limit_orders: true,
        }
    }
}

impl SignalFlowConfig {
    /// Create configuration from environment variables
    pub fn from_env() -> Self {
        let execution_config = KrakenExecutionConfig::from_env();
        let fill_tracker_config = FillTrackerConfig::from_env();
        let best_execution_config = BestExecutionConfig::from_env();

        let enable_fill_tracking = std::env::var("ENABLE_FILL_TRACKING")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let enable_best_execution = std::env::var("ENABLE_BEST_EXECUTION")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let default_exchange =
            std::env::var("DEFAULT_EXCHANGE").unwrap_or_else(|_| "kraken".to_string());

        let min_confidence = std::env::var("MIN_SIGNAL_CONFIDENCE")
            .unwrap_or_else(|_| "0.5".to_string())
            .parse()
            .unwrap_or(0.5);

        let min_strength = std::env::var("MIN_SIGNAL_STRENGTH")
            .unwrap_or_else(|_| "0.5".to_string())
            .parse()
            .unwrap_or(0.5);

        let max_position_size = std::env::var("MAX_POSITION_SIZE")
            .unwrap_or_else(|_| "10".to_string())
            .parse::<i64>()
            .map(Decimal::from)
            .unwrap_or(Decimal::from(10));

        let risk_limit_pct = std::env::var("RISK_LIMIT_PCT")
            .unwrap_or_else(|_| "2".to_string())
            .parse::<i64>()
            .map(Decimal::from)
            .unwrap_or(Decimal::from(2));

        let block_on_avoid = std::env::var("BLOCK_ON_AVOID")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        let use_limit_orders = std::env::var("USE_LIMIT_ORDERS")
            .unwrap_or_else(|_| "true".to_string())
            .parse()
            .unwrap_or(true);

        Self {
            execution_config,
            fill_tracker_config,
            best_execution_config,
            enable_fill_tracking,
            enable_best_execution,
            default_exchange,
            min_confidence,
            min_strength,
            max_position_size,
            risk_limit_pct,
            block_on_avoid,
            use_limit_orders,
        }
    }
}

// ============================================================================
// Signal Types
// ============================================================================

/// Trading signal from Janus
#[derive(Debug, Clone)]
pub struct TradingSignal {
    /// Unique signal ID
    pub signal_id: String,
    /// Trading symbol
    pub symbol: String,
    /// Signal side (buy/sell)
    pub side: SignalSide,
    /// Signal type (entry, exit, etc.)
    pub signal_type: SignalType,
    /// Signal confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Signal strength (0.0 to 1.0)
    pub strength: f64,
    /// Suggested entry price
    pub entry_price: Option<Decimal>,
    /// Suggested stop loss price
    pub stop_loss: Option<Decimal>,
    /// Suggested take profit price
    pub take_profit: Option<Decimal>,
    /// Suggested position size
    pub quantity: Option<Decimal>,
    /// Signal source/strategy
    pub source: String,
    /// Signal timestamp
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TradingSignal {
    /// Create a new trading signal
    pub fn new(symbol: String, side: SignalSide, signal_type: SignalType) -> Self {
        Self {
            signal_id: Uuid::new_v4().to_string(),
            symbol,
            side,
            signal_type,
            confidence: 0.5,
            strength: 0.5,
            entry_price: None,
            stop_loss: None,
            take_profit: None,
            quantity: None,
            source: "unknown".to_string(),
            timestamp: Utc::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set confidence
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set strength
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set entry price
    pub fn with_entry_price(mut self, price: Decimal) -> Self {
        self.entry_price = Some(price);
        self
    }

    /// Set stop loss
    pub fn with_stop_loss(mut self, price: Decimal) -> Self {
        self.stop_loss = Some(price);
        self
    }

    /// Set take profit
    pub fn with_take_profit(mut self, price: Decimal) -> Self {
        self.take_profit = Some(price);
        self
    }

    /// Set quantity
    pub fn with_quantity(mut self, qty: Decimal) -> Self {
        self.quantity = Some(qty);
        self
    }

    /// Set source
    pub fn with_source(mut self, source: String) -> Self {
        self.source = source;
        self
    }

    /// Calculate risk/reward ratio if prices are set
    pub fn risk_reward_ratio(&self) -> Option<f64> {
        match (self.entry_price, self.stop_loss, self.take_profit) {
            (Some(entry), Some(sl), Some(tp)) => {
                let risk = (entry - sl).abs();
                let reward = (tp - entry).abs();
                if risk > Decimal::ZERO {
                    Some(
                        reward
                            .checked_div(risk)
                            .map(|d| d.to_string().parse().unwrap_or(0.0))
                            .unwrap_or(0.0),
                    )
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Signal side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalSide {
    Buy,
    Sell,
}

impl From<SignalSide> for OrderSide {
    fn from(side: SignalSide) -> Self {
        match side {
            SignalSide::Buy => OrderSide::Buy,
            SignalSide::Sell => OrderSide::Sell,
        }
    }
}

/// Signal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalType {
    /// Entry signal
    Entry,
    /// Exit signal
    Exit,
    /// Add to position
    AddToPosition,
    /// Reduce position
    ReducePosition,
    /// Close position
    ClosePosition,
    /// Stop loss triggered
    StopLoss,
    /// Take profit triggered
    TakeProfit,
}

// ============================================================================
// Signal Response
// ============================================================================

/// Response from signal submission
#[derive(Debug, Clone)]
pub struct SignalResponse {
    /// Signal ID
    pub signal_id: String,
    /// Whether the signal was accepted
    pub accepted: bool,
    /// Internal order ID (if order was created)
    pub order_id: Option<String>,
    /// Exchange order ID (if submitted to exchange)
    pub exchange_order_id: Option<String>,
    /// Status message
    pub message: String,
    /// Response timestamp
    pub timestamp: DateTime<Utc>,
}

impl SignalResponse {
    /// Create an accepted response
    pub fn accepted(signal_id: String, order_id: String) -> Self {
        Self {
            signal_id,
            accepted: true,
            order_id: Some(order_id),
            exchange_order_id: None,
            message: "Signal accepted and order submitted".to_string(),
            timestamp: Utc::now(),
        }
    }

    /// Create a rejected response
    pub fn rejected(signal_id: String, reason: String) -> Self {
        Self {
            signal_id,
            accepted: false,
            order_id: None,
            exchange_order_id: None,
            message: reason,
            timestamp: Utc::now(),
        }
    }
}

// ============================================================================
// Execution Updates
// ============================================================================

/// Execution update event
#[derive(Debug, Clone)]
pub enum ExecutionUpdate {
    /// Order submitted to exchange
    OrderSubmitted {
        order_id: String,
        exchange_order_id: String,
        symbol: String,
        side: OrderSide,
        quantity: Decimal,
        timestamp: DateTime<Utc>,
    },
    /// Order filled (partial or full)
    OrderFilled {
        order_id: String,
        fill: Fill,
        remaining: Decimal,
        timestamp: DateTime<Utc>,
    },
    /// Order status changed
    OrderStatusChanged {
        order_id: String,
        old_status: OrderStatusEnum,
        new_status: OrderStatusEnum,
        timestamp: DateTime<Utc>,
    },
    /// Order cancelled
    OrderCancelled {
        order_id: String,
        reason: String,
        timestamp: DateTime<Utc>,
    },
    /// Order rejected
    OrderRejected {
        order_id: String,
        reason: String,
        timestamp: DateTime<Utc>,
    },
    /// Position updated
    PositionUpdated {
        symbol: String,
        position: Position,
        timestamp: DateTime<Utc>,
    },
    /// Balance updated
    BalanceUpdated {
        currency: String,
        available: Decimal,
        total: Decimal,
        timestamp: DateTime<Utc>,
    },
}

// ============================================================================
// Signal Flow Coordinator
// ============================================================================

/// Coordinates signal flow from Janus to execution with fill tracking
pub struct SignalFlowCoordinator {
    /// Configuration
    config: SignalFlowConfig,
    /// Kraken execution engine
    execution_engine: Arc<KrakenExecutionEngine>,
    /// Fill tracker (optional, for real-time fill updates)
    fill_tracker: Option<Arc<KrakenFillTracker>>,
    /// Best execution analyzer (optional, for execution quality analysis)
    best_execution: Option<Arc<BestExecutionAnalyzer>>,
    /// Execution update broadcaster
    update_tx: broadcast::Sender<ExecutionUpdate>,
    /// Is the coordinator running
    is_running: Arc<RwLock<bool>>,
    /// Signal statistics
    stats: Arc<RwLock<SignalFlowStats>>,
    /// Pending signals (signal_id -> order_id)
    pending_signals: Arc<RwLock<HashMap<String, String>>>,
    /// Prometheus metrics
    metrics: Arc<SignalFlowMetrics>,
    /// Kill switch guard — blocks ALL signal submission when active.
    /// Defense-in-depth: even if the brain pipeline's kill switch fires,
    /// the execution service independently refuses signals/orders.
    order_gate: Option<Arc<dyn crate::kill_switch_guard::OrderGate>>,
    /// Alertmanager client for publishing approved signals (fire-and-forget).
    /// `None` when `ALERTMANAGER_URL` is not set in the environment.
    alertmanager_client: Option<Arc<crate::alertmanager::AlertmanagerClient>>,
    /// Account-type label embedded in every Alertmanager alert.
    /// Controlled by `EXEC_ACCOUNT_TYPE` env var (default: `personal-crypto`).
    alertmanager_account_type: crate::alertmanager::AccountType,
}

impl SignalFlowCoordinator {
    /// Create a new signal flow coordinator
    pub async fn new(config: SignalFlowConfig) -> Result<Self> {
        info!("Creating signal flow coordinator...");

        // Create execution engine
        let execution_engine =
            Arc::new(KrakenExecutionEngine::new(config.execution_config.clone()));

        // Create fill tracker if enabled
        let fill_tracker =
            if config.enable_fill_tracking && config.fill_tracker_config.has_credentials() {
                Some(Arc::new(KrakenFillTracker::new(
                    config.fill_tracker_config.clone(),
                )))
            } else {
                if config.enable_fill_tracking {
                    warn!("Fill tracking enabled but no credentials configured");
                }
                None
            };

        // Create best execution analyzer if enabled
        let best_execution = if config.enable_best_execution {
            info!("Best execution analysis enabled");
            Some(Arc::new(BestExecutionAnalyzer::new(
                config.best_execution_config.clone(),
            )))
        } else {
            None
        };

        let (update_tx, _) = broadcast::channel(10000);

        // Get global metrics
        let metrics = signal_flow_metrics();

        // Alertmanager client — opt-in via ALERTMANAGER_URL env var.
        let alertmanager_client = std::env::var("ALERTMANAGER_URL").ok().map(|url| {
            info!(url = %url, "AlertmanagerClient enabled for signal push");
            Arc::new(crate::alertmanager::AlertmanagerClient::new(Some(url)))
        });

        // Account type for Alertmanager labels — EXEC_ACCOUNT_TYPE env var.
        let alertmanager_account_type = match std::env::var("EXEC_ACCOUNT_TYPE")
            .unwrap_or_else(|_| "personal-crypto".to_string())
            .as_str()
        {
            "prop-firm" => crate::alertmanager::AccountType::PropFirm,
            "hardware-wallet" => crate::alertmanager::AccountType::HardwareWallet,
            _ => crate::alertmanager::AccountType::PersonalCrypto,
        };

        Ok(Self {
            config,
            execution_engine,
            fill_tracker,
            best_execution,
            update_tx,
            is_running: Arc::new(RwLock::new(false)),
            stats: Arc::new(RwLock::new(SignalFlowStats::default())),
            pending_signals: Arc::new(RwLock::new(HashMap::new())),
            metrics,
            order_gate: None,
            alertmanager_client,
            alertmanager_account_type,
        })
    }

    /// Set the kill switch order gate.
    ///
    /// When set, every call to `submit_signal` will check this gate
    /// **before** any signal validation or best-execution analysis.
    /// If the gate reports blocked, the signal is immediately rejected
    /// with an error.
    pub fn set_order_gate(&mut self, gate: Arc<dyn crate::kill_switch_guard::OrderGate>) {
        self.order_gate = Some(gate);
    }

    /// Create from environment variables
    pub async fn from_env() -> Result<Self> {
        Self::new(SignalFlowConfig::from_env()).await
    }

    /// Subscribe to execution updates
    pub fn subscribe_updates(&self) -> broadcast::Receiver<ExecutionUpdate> {
        self.update_tx.subscribe()
    }

    /// Check if the coordinator is running
    pub async fn is_running(&self) -> bool {
        *self.is_running.read().await
    }

    /// Start the coordinator
    pub async fn start(&self) -> Result<()> {
        let mut is_running = self.is_running.write().await;
        if *is_running {
            return Err(ExecutionError::Internal(
                "Signal flow coordinator already running".to_string(),
            ));
        }

        info!("Starting signal flow coordinator...");

        // Initialize execution engine
        self.execution_engine.initialize().await?;

        // Start fill tracker if available
        if let Some(ref tracker) = self.fill_tracker {
            info!("Starting fill tracker...");
            tracker.start().await?;

            // Spawn fill event processor
            let tracker_clone = tracker.clone();
            let update_tx = self.update_tx.clone();
            let pending_signals = self.pending_signals.clone();
            let stats = self.stats.clone();
            let metrics = self.metrics.clone();

            tokio::spawn(async move {
                Self::process_fill_events(
                    tracker_clone,
                    update_tx,
                    pending_signals,
                    stats,
                    metrics,
                )
                .await;
            });
        }

        *is_running = true;
        info!("Signal flow coordinator started successfully");

        Ok(())
    }

    /// Stop the coordinator
    pub async fn stop(&self) {
        info!("Stopping signal flow coordinator...");

        *self.is_running.write().await = false;

        if let Some(ref tracker) = self.fill_tracker {
            tracker.stop().await;
        }

        info!("Signal flow coordinator stopped");
    }

    /// Submit a trading signal for execution
    pub async fn submit_signal(&self, signal: TradingSignal) -> Result<SignalResponse> {
        let start = Instant::now();
        let signal_id = signal.signal_id.clone();

        // ── Kill switch guard (defense-in-depth) ───────────────────────
        // This check runs BEFORE any validation, analysis, or exchange I/O.
        // It is a single atomic load — negligible latency on the hot path.
        if let Some(gate) = &self.order_gate {
            if gate.is_blocked() {
                error!(
                    signal_id = %signal_id,
                    symbol = %signal.symbol,
                    side = ?signal.side,
                    "🚨 SIGNAL BLOCKED BY KILL SWITCH"
                );
                let mut stats = self.stats.write().await;
                stats.signals_received += 1;
                stats.signals_rejected += 1;
                self.metrics
                    .record_signal_received(&format!("{:?}", signal.signal_type));
                self.metrics.record_signal_rejected();
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                global_latency_histograms().record_signal_processing(duration_ms);
                return Ok(SignalResponse::rejected(
                    signal_id,
                    gate.block_reason().to_string(),
                ));
            }
        }

        info!(
            signal_id = %signal_id,
            symbol = %signal.symbol,
            side = ?signal.side,
            confidence = signal.confidence,
            strength = signal.strength,
            "Processing trading signal"
        );

        // Update stats and metrics
        {
            let mut stats = self.stats.write().await;
            stats.signals_received += 1;
        }
        self.metrics
            .record_signal_received(&format!("{:?}", signal.signal_type));

        // Validate signal
        if let Err(reason) = self.validate_signal(&signal).await {
            warn!(signal_id = %signal_id, reason = %reason, "Signal rejected");
            let mut stats = self.stats.write().await;
            stats.signals_rejected += 1;
            self.metrics.record_signal_rejected();
            // Record signal processing latency even for rejections
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
            global_latency_histograms().record_signal_processing(duration_ms);
            return Ok(SignalResponse::rejected(signal_id, reason));
        }

        // Perform best execution analysis if enabled
        let (order_type_override, limit_price_override) = if let Some(ref analyzer) =
            self.best_execution
        {
            match self.analyze_best_execution(&signal, analyzer).await {
                Ok(Some((order_type, limit_price))) => (Some(order_type), limit_price),
                Ok(None) => {
                    // Analysis recommends avoiding the trade
                    warn!(signal_id = %signal_id, "Best execution analysis recommends avoiding trade");
                    let mut stats = self.stats.write().await;
                    stats.signals_rejected += 1;
                    self.metrics.record_signal_rejected();
                    // Record signal processing latency for avoided trades
                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                    global_latency_histograms().record_signal_processing(duration_ms);
                    return Ok(SignalResponse::rejected(
                        signal_id,
                        "Best execution analysis recommends avoiding this trade".to_string(),
                    ));
                }
                Err(e) => {
                    warn!(signal_id = %signal_id, error = %e, "Best execution analysis failed, proceeding with defaults");
                    (None, None)
                }
            }
        } else {
            (None, None)
        };

        // Convert signal to order (with potential overrides from best execution)
        let mut order = self.signal_to_order(&signal).await?;

        // Apply best execution overrides
        if let Some(order_type) = order_type_override {
            order.order_type = order_type;
        }
        if let Some(limit_price) = limit_price_override {
            order.price = Some(limit_price);
        }
        let order_id = order.id.clone();

        // Register signal -> order mapping
        self.pending_signals
            .write()
            .await
            .insert(signal_id.clone(), order_id.clone());

        // ── Alertmanager push (fire-and-forget) ────────────────────────────
        // Notify the Python-side monitor that a signal has been approved and
        // is being submitted to the exchange.  This is non-blocking: a failed
        // push is logged at WARN but never prevents order submission.
        if let Some(ref am_client) = self.alertmanager_client {
            let am_signal = crate::alertmanager::TradeSignalAlert {
                signal_id: signal_id.clone(),
                symbol: signal.symbol.clone(),
                direction: match signal.side {
                    SignalSide::Buy => crate::alertmanager::SignalDirection::Long,
                    SignalSide::Sell => crate::alertmanager::SignalDirection::Short,
                },
                account_type: self.alertmanager_account_type,
                confidence: signal.confidence,
                strength: signal.strength,
                entry_price: signal
                    .entry_price
                    .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0),
                stop_loss: signal
                    .stop_loss
                    .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0),
                take_profit_1: signal
                    .take_profit
                    .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0),
                take_profit_2: signal
                    .take_profit
                    .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0),
                take_profit_3: signal
                    .take_profit
                    .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0),
                position_size: signal
                    .quantity
                    .map(|d| d.to_string().parse::<f64>().unwrap_or(0.0))
                    .unwrap_or(0.0),
                strategy: signal
                    .metadata
                    .get("strategy")
                    .cloned()
                    .unwrap_or_else(|| signal.source.clone()),
                regime: signal
                    .metadata
                    .get("regime")
                    .cloned()
                    .unwrap_or_else(|| "UNKNOWN".to_string()),
                reasoning: signal
                    .metadata
                    .get("reasoning")
                    .cloned()
                    .unwrap_or_default(),
                timeframe: signal
                    .metadata
                    .get("timeframe")
                    .cloned()
                    .unwrap_or_else(|| "5m".to_string()),
            };
            let client_arc = Arc::clone(am_client);
            tokio::spawn(async move {
                if let Err(e) = client_arc.push_signal(&am_signal).await {
                    warn!("Alertmanager push failed (non-fatal): {}", e);
                }
            });
        }

        // Submit order to execution engine
        match self.execution_engine.submit_order(order.clone()).await {
            Ok(submitted_order_id) => {
                info!(
                    signal_id = %signal_id,
                    order_id = %submitted_order_id,
                    "Order submitted successfully"
                );

                // Register with fill tracker if available
                if let Some(ref tracker) = self.fill_tracker {
                    if let Some(exchange_id) = self
                        .execution_engine
                        .get_order(&submitted_order_id)
                        .ok()
                        .and_then(|o| o.exchange_order_id)
                    {
                        tracker
                            .register_order(&submitted_order_id, &exchange_id)
                            .await;
                    }
                    // Record submission time for fill latency tracking
                    tracker.record_order_submission(&submitted_order_id).await;
                }

                // Update stats and metrics
                {
                    let mut stats = self.stats.write().await;
                    stats.signals_executed += 1;
                    stats.orders_submitted += 1;
                }
                self.metrics.record_signal_executed();
                self.metrics.record_order_submitted();

                // Broadcast update
                let _ = self.update_tx.send(ExecutionUpdate::OrderSubmitted {
                    order_id: submitted_order_id.clone(),
                    exchange_order_id: order.exchange_order_id.clone().unwrap_or_default(),
                    symbol: order.symbol.clone(),
                    side: order.side,
                    quantity: order.quantity,
                    timestamp: Utc::now(),
                });

                // Record signal processing latency for successful submissions
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                global_latency_histograms().record_signal_processing(duration_ms);
                debug!(signal_id = %signal_id, latency_ms = duration_ms, "Signal processed successfully");

                let mut response = SignalResponse::accepted(signal_id, submitted_order_id);
                response.exchange_order_id = order.exchange_order_id;
                Ok(response)
            }
            Err(e) => {
                error!(
                    signal_id = %signal_id,
                    error = %e,
                    "Failed to submit order"
                );

                // Update stats and metrics
                {
                    let mut stats = self.stats.write().await;
                    stats.signals_rejected += 1;
                    stats.orders_rejected += 1;
                }
                self.metrics.record_signal_rejected();
                self.metrics.record_order_rejected();

                // Broadcast rejection
                let _ = self.update_tx.send(ExecutionUpdate::OrderRejected {
                    order_id,
                    reason: e.to_string(),
                    timestamp: Utc::now(),
                });

                // Record signal processing latency for rejections
                let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                global_latency_histograms().record_signal_processing(duration_ms);

                Ok(SignalResponse::rejected(signal_id, e.to_string()))
            }
        }
    }

    /// Analyze best execution for a signal
    /// Returns Ok(Some((order_type, limit_price))) if order should proceed
    /// Returns Ok(None) if order should be blocked (Avoid recommendation)
    async fn analyze_best_execution(
        &self,
        signal: &TradingSignal,
        analyzer: &BestExecutionAnalyzer,
    ) -> Result<Option<(OrderTypeEnum, Option<Decimal>)>> {
        use crate::execution::metrics::best_execution_metrics;

        let quantity = signal.quantity.unwrap_or(Decimal::ONE);
        let best_exec_side = match signal.side {
            SignalSide::Buy => BestExecOrderSide::Buy,
            SignalSide::Sell => BestExecOrderSide::Sell,
        };

        let analysis = match analyzer
            .analyze(&signal.symbol, best_exec_side, quantity)
            .await
        {
            Some(a) => a,
            None => {
                // No analysis available (no price data), proceed with default
                warn!(
                    signal_id = %signal.signal_id,
                    "No best execution analysis available (missing price data)"
                );
                return Ok(Some((OrderTypeEnum::Market, None)));
            }
        };

        info!(
            signal_id = %signal.signal_id,
            symbol = %signal.symbol,
            recommendation = ?analysis.recommendation,
            score = %analysis.score,
            slippage_bps = %analysis.estimated_slippage_bps,
            "Best execution analysis complete"
        );

        // Record metrics
        let metrics = best_execution_metrics();
        let rec_type = match analysis.recommendation {
            ExecutionRecommendation::ExecuteNow => RecommendationType::ExecuteNow,
            ExecutionRecommendation::Wait => RecommendationType::Wait,
            ExecutionRecommendation::Acceptable => RecommendationType::Acceptable,
            ExecutionRecommendation::UseLimitOrder => RecommendationType::UseLimitOrder,
            ExecutionRecommendation::Avoid => RecommendationType::Avoid,
        };
        metrics.record_analysis(
            &signal.symbol,
            rec_type,
            analysis.score,
            analysis.estimated_slippage_bps,
        );

        match analysis.recommendation {
            ExecutionRecommendation::Avoid => {
                if self.config.block_on_avoid {
                    Ok(None) // Block the order
                } else {
                    warn!(
                        signal_id = %signal.signal_id,
                        "Best execution recommends Avoid but block_on_avoid is disabled"
                    );
                    Ok(Some((OrderTypeEnum::Market, None)))
                }
            }
            ExecutionRecommendation::UseLimitOrder => {
                if self.config.use_limit_orders {
                    let limit_price = analysis.suggested_limit_price;
                    Ok(Some((OrderTypeEnum::Limit, limit_price)))
                } else {
                    Ok(Some((OrderTypeEnum::Market, None)))
                }
            }
            ExecutionRecommendation::Wait => {
                // For Wait, we still proceed but log a warning
                warn!(
                    signal_id = %signal.signal_id,
                    "Best execution recommends waiting, but proceeding with order"
                );
                Ok(Some((OrderTypeEnum::Market, None)))
            }
            ExecutionRecommendation::ExecuteNow | ExecutionRecommendation::Acceptable => {
                Ok(Some((OrderTypeEnum::Market, None)))
            }
        }
    }

    /// Validate a trading signal
    async fn validate_signal(&self, signal: &TradingSignal) -> std::result::Result<(), String> {
        // Check confidence threshold
        if signal.confidence < self.config.min_confidence {
            return Err(format!(
                "Signal confidence {} below threshold {}",
                signal.confidence, self.config.min_confidence
            ));
        }

        // Check strength threshold
        if signal.strength < self.config.min_strength {
            return Err(format!(
                "Signal strength {} below threshold {}",
                signal.strength, self.config.min_strength
            ));
        }

        // Check symbol format
        if signal.symbol.is_empty() {
            return Err("Symbol is required".to_string());
        }

        // Check quantity if provided
        if let Some(qty) = signal.quantity {
            if qty <= Decimal::ZERO {
                return Err("Quantity must be positive".to_string());
            }
            if qty > self.config.max_position_size {
                return Err(format!(
                    "Quantity {} exceeds max position size {}",
                    qty, self.config.max_position_size
                ));
            }
        }

        // Check risk/reward if stop loss and take profit are set
        if let Some(rr) = signal.risk_reward_ratio() {
            if rr < 1.0 {
                warn!(
                    signal_id = %signal.signal_id,
                    risk_reward = rr,
                    "Signal has unfavorable risk/reward ratio"
                );
            }
        }

        Ok(())
    }

    /// Convert a trading signal to an order
    async fn signal_to_order(&self, signal: &TradingSignal) -> Result<Order> {
        let side = signal.side.into();

        // Determine order type
        let order_type = if signal.entry_price.is_some() {
            OrderTypeEnum::Limit
        } else {
            OrderTypeEnum::Market
        };

        // Determine quantity
        let quantity = signal.quantity.unwrap_or(Decimal::from(1));

        let mut order = Order::new(
            signal.signal_id.clone(),
            signal.symbol.clone(),
            self.config.default_exchange.clone(),
            side,
            order_type,
            quantity,
        );

        // Set price for limit orders
        order.price = signal.entry_price;

        // Set stop price if this is a stop loss order
        if signal.signal_type == SignalType::StopLoss {
            order.stop_price = signal.stop_loss;
            order.order_type = OrderTypeEnum::StopMarket;
        }

        // Add signal metadata to order
        order
            .metadata
            .insert("signal_id".to_string(), signal.signal_id.clone());
        order
            .metadata
            .insert("signal_source".to_string(), signal.source.clone());
        order.metadata.insert(
            "signal_confidence".to_string(),
            signal.confidence.to_string(),
        );
        order
            .metadata
            .insert("signal_strength".to_string(), signal.strength.to_string());

        if let Some(sl) = signal.stop_loss {
            order
                .metadata
                .insert("stop_loss".to_string(), sl.to_string());
        }
        if let Some(tp) = signal.take_profit {
            order
                .metadata
                .insert("take_profit".to_string(), tp.to_string());
        }

        Ok(order)
    }

    /// Process fill events from the fill tracker
    async fn process_fill_events(
        tracker: Arc<KrakenFillTracker>,
        update_tx: broadcast::Sender<ExecutionUpdate>,
        pending_signals: Arc<RwLock<HashMap<String, String>>>,
        stats: Arc<RwLock<SignalFlowStats>>,
        metrics: Arc<SignalFlowMetrics>,
    ) {
        let mut fill_rx = tracker.subscribe_fills();
        let mut status_rx = tracker.subscribe_order_status();

        loop {
            tokio::select! {
                Ok(fill) = fill_rx.recv() => {
                    info!(
                        order_id = %fill.order_id,
                        quantity = %fill.quantity,
                        price = %fill.price,
                        "Fill received from tracker"
                    );

                    // Update stats and metrics
                    {
                        let mut s = stats.write().await;
                        s.fills_received += 1;
                    }
                    metrics.record_fill_received();
                    metrics.record_order_filled(fill.quantity * fill.price);

                    // Broadcast fill update
                    let _ = update_tx.send(ExecutionUpdate::OrderFilled {
                        order_id: fill.order_id.clone(),
                        fill: fill.clone(),
                        remaining: Decimal::ZERO, // Would need to track this
                        timestamp: Utc::now(),
                    });
                }
                Ok((order_id, status)) = status_rx.recv() => {
                    debug!(
                        order_id = %order_id,
                        status = ?status,
                        "Order status update from tracker"
                    );

                    // Broadcast status update
                    let _ = update_tx.send(ExecutionUpdate::OrderStatusChanged {
                        order_id: order_id.clone(),
                        old_status: OrderStatusEnum::Submitted, // Would need to track previous
                        new_status: status,
                        timestamp: Utc::now(),
                    });

                    // Update metrics based on status
                    match status {
                        OrderStatusEnum::Cancelled => metrics.record_order_cancelled(),
                        OrderStatusEnum::Rejected => metrics.record_order_rejected(),
                        _ => {}
                    }

                    // Clean up pending signals if order is terminal
                    if matches!(
                        status,
                        OrderStatusEnum::Filled
                            | OrderStatusEnum::Cancelled
                            | OrderStatusEnum::Rejected
                            | OrderStatusEnum::Expired
                    ) {
                        let mut pending = pending_signals.write().await;
                        pending.retain(|_, v| v != &order_id);
                    }
                }
                else => {
                    // Channel closed
                    break;
                }
            }
        }

        info!("Fill event processing stopped");
    }

    /// Get execution engine reference
    pub fn execution_engine(&self) -> &KrakenExecutionEngine {
        &self.execution_engine
    }

    /// Get fill tracker reference
    pub fn fill_tracker(&self) -> Option<&KrakenFillTracker> {
        self.fill_tracker.as_ref().map(|t| t.as_ref())
    }

    /// Get best execution analyzer reference
    pub fn best_execution_analyzer(&self) -> Option<&BestExecutionAnalyzer> {
        self.best_execution.as_ref().map(|a| a.as_ref())
    }

    /// Get the metrics reference
    pub fn metrics(&self) -> &SignalFlowMetrics {
        &self.metrics
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> SignalFlowStats {
        self.stats.read().await.clone()
    }

    /// Get all positions
    pub fn get_positions(&self) -> Vec<Position> {
        self.execution_engine.get_all_positions()
    }

    /// Get a specific position
    pub fn get_position(&self, symbol: &str) -> Option<Position> {
        self.execution_engine.get_position(symbol)
    }

    /// Get current balance
    pub fn get_balance(&self) -> Decimal {
        self.execution_engine.get_balance()
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        self.execution_engine.cancel_order(order_id).await?;

        let _ = self.update_tx.send(ExecutionUpdate::OrderCancelled {
            order_id: order_id.to_string(),
            reason: "User requested".to_string(),
            timestamp: Utc::now(),
        });

        // Update stats
        {
            let mut stats = self.stats.write().await;
            stats.orders_cancelled += 1;
        }

        Ok(())
    }

    /// Close a position
    pub async fn close_position(&self, symbol: &str) -> Result<SignalResponse> {
        if let Some(position) = self.get_position(symbol) {
            // Create exit signal
            let side = match position.side {
                crate::types::PositionSide::Long => SignalSide::Sell,
                crate::types::PositionSide::Short => SignalSide::Buy,
            };

            let signal = TradingSignal::new(symbol.to_string(), side, SignalType::ClosePosition)
                .with_quantity(position.quantity)
                .with_confidence(1.0)
                .with_strength(1.0)
                .with_source("manual_close".to_string());

            self.submit_signal(signal).await
        } else {
            Ok(SignalResponse::rejected(
                Uuid::new_v4().to_string(),
                format!("No position found for {}", symbol),
            ))
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Signal flow statistics
#[derive(Debug, Clone, Default)]
pub struct SignalFlowStats {
    /// Total signals received
    pub signals_received: u64,
    /// Signals that passed validation and were executed
    pub signals_executed: u64,
    /// Signals rejected (validation failed or execution failed)
    pub signals_rejected: u64,
    /// Orders submitted to exchange
    pub orders_submitted: u64,
    /// Orders filled (partial or full)
    pub orders_filled: u64,
    /// Orders cancelled
    pub orders_cancelled: u64,
    /// Orders rejected by exchange
    pub orders_rejected: u64,
    /// Fills received from exchange
    pub fills_received: u64,
}

impl SignalFlowStats {
    /// Calculate acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.signals_received == 0 {
            0.0
        } else {
            self.signals_executed as f64 / self.signals_received as f64
        }
    }

    /// Calculate fill rate
    pub fn fill_rate(&self) -> f64 {
        if self.orders_submitted == 0 {
            0.0
        } else {
            self.orders_filled as f64 / self.orders_submitted as f64
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = SignalFlowConfig::default();
        assert!(config.enable_fill_tracking);
        assert!(config.enable_best_execution);
        assert_eq!(config.min_confidence, 0.5);
        assert_eq!(config.min_strength, 0.5);
        assert!(config.block_on_avoid);
        assert!(config.use_limit_orders);
    }

    #[test]
    fn test_trading_signal_creation() {
        let signal = TradingSignal::new("BTC/USD".to_string(), SignalSide::Buy, SignalType::Entry)
            .with_confidence(0.8)
            .with_strength(0.7)
            .with_source("ema_crossover".to_string());

        assert_eq!(signal.symbol, "BTC/USD");
        assert_eq!(signal.side, SignalSide::Buy);
        assert_eq!(signal.confidence, 0.8);
        assert_eq!(signal.strength, 0.7);
        assert_eq!(signal.source, "ema_crossover");
    }

    #[test]
    fn test_signal_with_prices() {
        let signal = TradingSignal::new("ETH/USD".to_string(), SignalSide::Buy, SignalType::Entry)
            .with_entry_price(Decimal::from(2000))
            .with_stop_loss(Decimal::from(1900))
            .with_take_profit(Decimal::from(2200));

        assert_eq!(signal.entry_price, Some(Decimal::from(2000)));
        assert_eq!(signal.stop_loss, Some(Decimal::from(1900)));
        assert_eq!(signal.take_profit, Some(Decimal::from(2200)));

        // Risk: 100, Reward: 200, R/R = 2.0
        let rr = signal.risk_reward_ratio().unwrap();
        assert!((rr - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_signal_response_accepted() {
        let response = SignalResponse::accepted("sig-123".to_string(), "order-456".to_string());
        assert!(response.accepted);
        assert_eq!(response.order_id, Some("order-456".to_string()));
    }

    #[test]
    fn test_signal_response_rejected() {
        let response =
            SignalResponse::rejected("sig-123".to_string(), "Low confidence".to_string());
        assert!(!response.accepted);
        assert!(response.order_id.is_none());
        assert!(response.message.contains("Low confidence"));
    }

    #[test]
    fn test_signal_side_conversion() {
        let buy: OrderSide = SignalSide::Buy.into();
        assert_eq!(buy, OrderSide::Buy);

        let sell: OrderSide = SignalSide::Sell.into();
        assert_eq!(sell, OrderSide::Sell);
    }

    #[test]
    fn test_stats_default() {
        let stats = SignalFlowStats::default();
        assert_eq!(stats.signals_received, 0);
        assert_eq!(stats.acceptance_rate(), 0.0);
        assert_eq!(stats.fill_rate(), 0.0);
    }

    #[test]
    fn test_stats_rates() {
        let stats = SignalFlowStats {
            signals_received: 100,
            signals_executed: 80,
            signals_rejected: 20,
            orders_submitted: 80,
            orders_filled: 70,
            orders_cancelled: 5,
            orders_rejected: 5,
            fills_received: 70,
        };

        assert_eq!(stats.acceptance_rate(), 0.8);
        assert!((stats.fill_rate() - 0.875).abs() < 0.001);
    }
}
