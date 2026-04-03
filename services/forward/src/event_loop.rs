//! Production Event Loop for JANUS Forward Service
//!
//! This module implements the high-frequency event loop that processes market data
//! in real-time, updating indicators, generating signals, validating through Sheriff,
//! and executing orders via Bybit.
//!
//! Pipeline:
//! 1. Bybit WebSocket → Market Ticks
//! 2. Incremental Indicator Updates (EMA 8/21, ATR)
//! 3. Strategy Signal Generation (EMA Flip + MR + Squeeze + VWAP Scalper + ORB
//!    + EMA Ribbon + Trend Pullback + Momentum Surge + Multi-TF Trend)
//! 4. Regime Detection → Strategy Gating (MR, Squeeze, VWAP, ORB when MeanReverting;
//!    EMA Ribbon, Trend Pullback, Momentum Surge, Multi-TF when TrendFollowing)
//! 5. Compliance Sheriff Validation (Prop Firm Rules)
//! 6. Position Sizing & Risk Management
//! 7. Order Execution via Bybit REST
//! 8. Data Persistence to QuestDB

use anyhow::Result;
use janus_bybit_client::{
    BybitCredentials, BybitRestClient, BybitWebSocket, OrderRequest, OrderSide, OrderType,
    WsMessage,
};
use janus_forward::brain_wiring::{TradeAction, TradingPipeline};
use janus_forward::metrics::JanusMetrics;
use janus_forward::regime::{RegimeManager, RegimeManagerConfig};
use janus_forward::regime_bridge::{BridgedRegimeState, bridge_regime_signal};
use janus_indicators::IndicatorCalculator;
use janus_models::prop_firm::ChallengeType;
use janus_models::prop_firm::PropFirmValidator;
use janus_questdb_writer::{
    ExecutionWrite, QuestDBConfig, QuestDBWriter, SignalWrite, TickWrite, TradeWrite, now_micros,
};
use janus_regime::ActiveStrategy;
use janus_regime::types::TrendDirection;
use janus_risk::PositionSizer;
use janus_strategies::EMAFlipStrategy;
use janus_strategies::bollinger_squeeze::{
    SqueezeBreakoutConfig, SqueezeBreakoutSignal, SqueezeBreakoutStrategy,
};
use janus_strategies::ema_ribbon_scalper::{
    EmaRibbonConfig, EmaRibbonScalperStrategy, EmaRibbonSignal,
};
use janus_strategies::mean_reversion::{
    MeanReversionConfig, MeanReversionSignal, MeanReversionStrategy,
};
use janus_strategies::momentum_surge::{
    MomentumSurgeConfig, MomentumSurgeSignal, MomentumSurgeStrategy,
};
use janus_strategies::multi_tf_trend::{MultiTfConfig, MultiTfSignal, MultiTfTrendStrategy};
use janus_strategies::opening_range::{OrbConfig, OrbSignal, OrbStrategy};
use janus_strategies::trend_pullback::{
    TrendPullbackConfig, TrendPullbackSignal, TrendPullbackStrategy,
};
use janus_strategies::vwap_scalper::{VwapScalperConfig, VwapScalperStrategy, VwapSignal};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use tracing::{debug, error, info, warn};

// ============================================================================
// Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct EventLoopConfig {
    pub symbol: String,
    pub bybit_testnet: bool,
    pub bybit_api_key: String,
    pub bybit_api_secret: String,
    pub questdb_host: String,
    pub questdb_port: u16,
    pub account_size: f64,
    pub challenge_type: ChallengeType,
    pub trading_enabled: bool,
    pub max_risk_per_trade: f64,
    /// UTC hours at which to trigger session boundaries for ORB and VWAP.
    /// For example, `vec![0, 8, 13]` triggers at UTC midnight, London open,
    /// and NY open. Default is `vec![0]` (daily reset at midnight UTC).
    pub session_start_hours_utc: Vec<u8>,
    /// Optional path to `regime.toml` configuration file.
    /// When set, the regime manager loads its config from this file.
    /// When `None`, falls back to `RegimeManagerConfig::default()`.
    pub regime_toml_path: Option<String>,
    /// Taker fee rate applied to each execution for P&L tracking.
    /// Kraken default: 0.0026 (0.26%). Bybit linear: 0.00055 (0.055%).
    /// Set via `TAKER_FEE_RATE` env var.
    pub taker_fee_rate: f64,
    /// When true, the event loop will close any open position on shutdown.
    /// Set via `CLOSE_ON_SHUTDOWN` env var. Default: false (paper trading safe).
    pub close_on_shutdown: bool,
}

impl Default for EventLoopConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSD".to_string(),
            bybit_testnet: true,
            bybit_api_key: std::env::var("BYBIT_API_KEY").unwrap_or_default(),
            bybit_api_secret: std::env::var("BYBIT_API_SECRET").unwrap_or_default(),
            questdb_host: "127.0.0.1".to_string(),
            questdb_port: 9009,
            account_size: 10000.0,
            challenge_type: ChallengeType::OneStep,
            trading_enabled: false,   // Start in monitoring mode
            max_risk_per_trade: 0.01, // 1% risk per trade
            session_start_hours_utc: vec![0, 8, 13], // Midnight UTC, London, NY
            regime_toml_path: None,
            taker_fee_rate: std::env::var("TAKER_FEE_RATE")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0026), // Kraken default taker fee: 0.26%
            close_on_shutdown: std::env::var("CLOSE_ON_SHUTDOWN")
                .ok()
                .and_then(|v| v.parse::<bool>().ok())
                .unwrap_or(false),
        }
    }
}

// ============================================================================
// Strategy State
// ============================================================================

/// Higher-timeframe candle aggregator.
///
/// Accumulates N low-timeframe (LTF) candles into one HTF candle, then
/// computes a simple EMA-cross trend direction that is fed into HTF-aware
/// strategies (Multi-TF Trend, EMA Ribbon Scalper, Trend Pullback).
struct HtfAggregator {
    /// How many LTF candles to aggregate into one HTF candle.
    ratio: usize,
    /// Running count of LTF candles within the current HTF bar.
    count: usize,
    /// Aggregated OHLCV for the current HTF bar.
    agg_open: f64,
    agg_high: f64,
    agg_low: f64,
    agg_close: f64,
    agg_volume: f64,
    /// Short EMA on HTF closes for trend detection.
    ema_short: janus_regime::indicators::EMA,
    /// Long EMA on HTF closes for trend detection.
    ema_long: janus_regime::indicators::EMA,
    /// Number of completed HTF candles (for warmup gating).
    candle_count: usize,
    /// Minimum HTF candles before trend is emitted.
    warmup: usize,
}

impl HtfAggregator {
    /// Create a new aggregator.
    ///
    /// * `ratio` — LTF candles per HTF candle (e.g. 15).
    /// * `ema_short` — short EMA period on HTF (e.g. 20).
    /// * `ema_long` — long EMA period on HTF (e.g. 50).
    fn new(ratio: usize, ema_short_period: usize, ema_long_period: usize) -> Self {
        Self {
            ratio,
            count: 0,
            agg_open: 0.0,
            agg_high: f64::NEG_INFINITY,
            agg_low: f64::INFINITY,
            agg_close: 0.0,
            agg_volume: 0.0,
            ema_short: janus_regime::indicators::EMA::new(ema_short_period),
            ema_long: janus_regime::indicators::EMA::new(ema_long_period),
            candle_count: 0,
            warmup: ema_long_period + 5,
        }
    }

    /// Feed one LTF candle. Returns `Some(TrendDirection)` when an HTF candle
    /// completes and the EMAs are warmed up; `None` otherwise.
    fn feed(
        &mut self,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Option<TrendDirection> {
        if self.count == 0 {
            self.agg_open = open;
            self.agg_high = high;
            self.agg_low = low;
        } else {
            if high > self.agg_high {
                self.agg_high = high;
            }
            if low < self.agg_low {
                self.agg_low = low;
            }
        }
        self.agg_close = close;
        self.agg_volume += volume;
        self.count += 1;

        if self.count >= self.ratio {
            // HTF candle complete — update EMAs
            let htf_close = self.agg_close;
            self.ema_short.update(htf_close);
            self.ema_long.update(htf_close);
            self.candle_count += 1;

            // Reset for next HTF bar
            self.count = 0;
            self.agg_high = f64::NEG_INFINITY;
            self.agg_low = f64::INFINITY;
            self.agg_volume = 0.0;

            // Emit trend once warmed up
            if self.candle_count >= self.warmup {
                let short_val = self.ema_short.value();
                let long_val = self.ema_long.value();
                if short_val > long_val {
                    return Some(TrendDirection::Bullish);
                } else {
                    return Some(TrendDirection::Bearish);
                }
            }
        }
        None
    }
}

use janus_forward::regime::RegimeConfigHolder;

struct StrategyState {
    indicators: IndicatorCalculator,
    strategy: EMAFlipStrategy,
    last_signal: Option<String>, // "BUY", "SELL", or "NONE"
    position: Option<Position>,
    tick_count: u64,
    last_price: f64,
    /// Regime detection manager — classifies market state and recommends strategy
    regime: RegimeManager,
    /// The last recommended strategy from the regime router
    last_active_strategy: Option<ActiveStrategy>,
    /// Mean reversion strategy instance — activated when regime recommends MeanReversion
    mean_reversion: MeanReversionStrategy,
    /// Last signal produced by the mean reversion strategy (to avoid duplicate signals)
    last_mr_signal: Option<String>,
    /// Bollinger Squeeze Breakout strategy — runs alongside MR when regime is MeanReverting
    squeeze_breakout: SqueezeBreakoutStrategy,
    /// Last signal produced by the squeeze breakout strategy (to avoid duplicate signals)
    last_squeeze_signal: Option<String>,
    /// VWAP Scalper strategy — mean reversion scalping around session VWAP
    vwap_scalper: VwapScalperStrategy,
    /// Last signal produced by the VWAP Scalper (duplicate-signal guard)
    last_vwap_signal: Option<String>,
    /// Opening Range Breakout strategy — breakout from session opening range
    orb: OrbStrategy,
    /// Last signal produced by ORB (duplicate-signal guard)
    last_orb_signal: Option<String>,
    /// EMA Ribbon Scalper — 8/13/21 ribbon pullback scalper (TrendFollowing)
    ema_ribbon: EmaRibbonScalperStrategy,
    /// Last signal produced by EMA Ribbon Scalper (duplicate-signal guard)
    last_ema_ribbon_signal: Option<String>,
    /// Trend Pullback — Fibonacci retracement + candlestick patterns (TrendFollowing)
    trend_pullback: TrendPullbackStrategy,
    /// Last signal produced by Trend Pullback (duplicate-signal guard)
    last_trend_pullback_signal: Option<String>,
    /// Momentum Surge — sudden price surges with volume spikes (TrendFollowing/Volatile)
    momentum_surge: MomentumSurgeStrategy,
    /// Last signal produced by Momentum Surge (duplicate-signal guard)
    last_momentum_surge_signal: Option<String>,
    /// Multi-Timeframe Trend — EMA 50/200 + ADX with HTF alignment (TrendFollowing only)
    multi_tf_trend: MultiTfTrendStrategy,
    /// Last signal produced by Multi-TF Trend (duplicate-signal guard)
    last_multi_tf_signal: Option<String>,
    /// Last position factor from the regime router (used for position sizing)
    last_position_factor: f64,
    /// Last regime confidence (used for mean reversion size scaling)
    last_regime_confidence: f64,
    /// Higher-timeframe candle aggregator for feeding HTF trend to strategies
    htf_aggregator: HtfAggregator,
}

impl StrategyState {
    fn new(
        symbol: &str,
        metrics: Option<&Arc<JanusMetrics>>,
        regime_toml_path: Option<&str>,
    ) -> Self {
        // Load regime config from TOML file if provided, otherwise use defaults
        let regime_config = match regime_toml_path {
            Some(path) => match RegimeManagerConfig::from_toml_file(path) {
                Ok(cfg) => {
                    info!("✅ Regime config loaded from {}", path);
                    cfg
                }
                Err(e) => {
                    warn!(
                        "Failed to load regime config from {} ({}), using defaults",
                        path, e
                    );
                    RegimeManagerConfig::default()
                }
            },
            None => {
                info!("ℹ️ No regime.toml path configured, using default regime config");
                RegimeManagerConfig::default()
            }
        };

        let regime = match metrics {
            Some(m) => match RegimeManager::with_metrics(regime_config.clone(), &m.registry()) {
                Ok(mut mgr) => {
                    mgr.register_asset(symbol);
                    info!(
                        "✅ Regime manager created with Prometheus metrics for {}",
                        symbol
                    );
                    mgr
                }
                Err(e) => {
                    warn!(
                        "Failed to create regime manager with metrics ({}), falling back to unmetered",
                        e
                    );
                    let mut mgr = RegimeManager::new(regime_config);
                    mgr.register_asset(symbol);
                    mgr
                }
            },
            None => {
                let mut mgr = RegimeManager::new(regime_config);
                mgr.register_asset(symbol);
                mgr
            }
        };

        // Create ORB with crypto-aggressive config; session must be started
        // explicitly by the session timer before breakout signals fire.
        let orb = OrbStrategy::new(OrbConfig::crypto_aggressive());

        Self {
            indicators: IndicatorCalculator::new(8, 21, 14),
            strategy: EMAFlipStrategy::new(8, 21),
            last_signal: None,
            position: None,
            tick_count: 0,
            last_price: 0.0,
            regime,
            last_active_strategy: None,
            mean_reversion: MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive()),
            last_mr_signal: None,
            squeeze_breakout: SqueezeBreakoutStrategy::new(
                SqueezeBreakoutConfig::crypto_aggressive(),
            ),
            last_squeeze_signal: None,
            vwap_scalper: VwapScalperStrategy::new(VwapScalperConfig::crypto_aggressive()),
            last_vwap_signal: None,
            orb,
            last_orb_signal: None,
            ema_ribbon: EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive()),
            last_ema_ribbon_signal: None,
            trend_pullback: TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive()),
            last_trend_pullback_signal: None,
            momentum_surge: MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive()),
            last_momentum_surge_signal: None,
            multi_tf_trend: MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive()),
            last_multi_tf_signal: None,
            last_position_factor: 1.0,
            last_regime_confidence: 0.0,
            // HTF aggregator: 15 LTF candles → 1 HTF candle, EMA(20)/EMA(50) for trend
            htf_aggregator: HtfAggregator::new(15, 20, 50),
        }
    }

    fn is_ready(&self) -> bool {
        self.indicators.is_ready()
    }
}

impl RegimeConfigHolder for StrategyState {
    fn regime_manager_mut(&mut self) -> &mut RegimeManager {
        &mut self.regime
    }
}

impl StrategyState {
    /// Reset session state for VWAP Scalper, ORB, and trend-following strategies.
    /// Called by the session timer at configured UTC boundaries.
    fn start_session(&mut self) {
        self.vwap_scalper.reset_session();
        self.orb.start_session();
        self.last_vwap_signal = None;
        self.last_orb_signal = None;
        // Reset trend-following duplicate-signal guards so new session signals fire
        self.last_ema_ribbon_signal = None;
        self.last_trend_pullback_signal = None;
        self.last_momentum_surge_signal = None;
        self.last_multi_tf_signal = None;
        info!("📅 Session boundary: VWAP reset + ORB opening range started + TF guards cleared");
    }
}

// ============================================================================
// Position Tracking
// ============================================================================

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Position {
    /// Trading symbol
    symbol: String,
    /// "Buy" or "Sell"
    side: String,
    /// Entry price
    entry_price: f64,
    /// Position size in base asset
    size: f64,
    /// Stop-loss price
    stop_loss: f64,
    /// Take-profit price
    take_profit: f64,
    /// Entry timestamp (microseconds)
    timestamp: i64,
    /// Which strategy opened this position
    source_strategy: PositionSource,
}

/// Tracks which strategy opened a position so we can route exit logic correctly.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PositionSource {
    EmaFlip,
    MeanReversion,
    SqueezeBreakout,
    VwapScalper,
    Orb,
    EmaRibbon,
    TrendPullback,
    MomentumSurge,
    MultiTfTrend,
}

// ============================================================================
// Event Loop
// ============================================================================

pub struct EventLoop {
    config: EventLoopConfig,
    state: Arc<RwLock<StrategyState>>,
    validator: Arc<RwLock<PropFirmValidator>>,
    /// Position sizer for calculating order quantities
    #[allow(dead_code)]
    position_sizer: Arc<PositionSizer>,
    questdb: Arc<QuestDBWriter>,
    bybit_rest: Arc<BybitRestClient>,
    /// Prometheus metrics — shared with the regime manager
    metrics: Option<Arc<JanusMetrics>>,
    /// Broadcast channel for bridged regime states.
    /// Neuromorphic consumers (hypothalamus, amygdala) subscribe to this
    /// channel to receive regime updates translated into their native types.
    regime_bridge_tx: tokio::sync::broadcast::Sender<BridgedRegimeState>,
    /// Optional brain-inspired trading pipeline for gating order execution.
    /// When set, every new position entry is evaluated through the full
    /// regime → hypothalamus → amygdala → gating → correlation pipeline
    /// before the order is sent to Bybit.
    brain_pipeline: Option<Arc<TradingPipeline>>,
}

impl EventLoop {
    /// Create a new event loop
    pub fn new(config: EventLoopConfig) -> Result<Self> {
        // Initialize QuestDB writer
        let questdb_config = QuestDBConfig::new(config.questdb_host.clone(), config.questdb_port)
            .with_batch_size(1000)
            .with_flush_interval(100)
            .with_channel_buffer(10000);
        let questdb = Arc::new(QuestDBWriter::new(questdb_config));

        // Initialize PropFirmValidator
        let validator = Arc::new(RwLock::new(PropFirmValidator::new(
            config.account_size,
            config.challenge_type,
        )));

        // Initialize position sizer
        let position_sizer = Arc::new(PositionSizer::new(config.account_size, 1.0));

        // Initialize Bybit REST client
        let creds = BybitCredentials::new(
            config.bybit_api_key.clone(),
            config.bybit_api_secret.clone(),
        );
        let bybit_rest = Arc::new(BybitRestClient::new(creds, config.bybit_testnet));

        // Initialize Prometheus metrics
        let metrics = match JanusMetrics::new() {
            Ok(m) => {
                let m = Arc::new(m);
                info!("✅ Prometheus metrics initialized");
                Some(m)
            }
            Err(e) => {
                warn!(
                    "Failed to initialize Prometheus metrics: {}. Continuing without metrics.",
                    e
                );
                None
            }
        };

        let state = Arc::new(RwLock::new(StrategyState::new(
            &config.symbol,
            metrics.as_ref(),
            config.regime_toml_path.as_deref(),
        )));

        // Create broadcast channel for bridged regime states.
        // Buffer of 64 is generous — consumers that fall behind will
        // receive a Lagged error and can skip to the latest state.
        let (regime_bridge_tx, _) = tokio::sync::broadcast::channel::<BridgedRegimeState>(64);

        Ok(Self {
            config: config.clone(),
            state,
            validator,
            position_sizer,
            questdb,
            bybit_rest,
            metrics,
            regime_bridge_tx,
            brain_pipeline: None,
        })
    }

    /// Attach a brain-inspired trading pipeline for gating order execution.
    ///
    /// When set, every new position entry is evaluated through the pipeline
    /// before the order is sent to Bybit. If the pipeline blocks the trade,
    /// the order is skipped. If it approves with a scale < 1.0, the position
    /// size is reduced accordingly.
    pub fn set_brain_pipeline(&mut self, pipeline: Arc<TradingPipeline>) {
        info!("🧠 EventLoop: brain pipeline attached for order gating");
        self.brain_pipeline = Some(pipeline);
    }

    /// Check whether a brain pipeline is attached.
    pub fn has_brain_pipeline(&self) -> bool {
        self.brain_pipeline.is_some()
    }

    /// Evaluate the brain pipeline gate for a new position entry.
    ///
    /// Returns `Some(scale)` if the trade should proceed (scale may be < 1.0
    /// to reduce position size), or `None` if the pipeline blocks the trade.
    ///
    /// When no pipeline is attached, always returns `Some(1.0)` (pass-through).
    async fn brain_gate_check(&self, strategy_name: &str) -> Option<f64> {
        let pipeline = self.brain_pipeline.as_ref()?;

        // Build a RoutedSignal from the current state
        let state = self.state.read().await;
        let regime = state
            .regime
            .current_regime(&self.config.symbol)
            .unwrap_or(janus_regime::MarketRegime::Uncertain);
        let confidence = state.last_regime_confidence;
        let position_factor = state.last_position_factor;
        let active_strategy = state
            .last_active_strategy
            .unwrap_or(janus_regime::ActiveStrategy::EmaFlip);

        // Gather indicator values from regime manager
        let adx_val = state.regime.adx_value(&self.config.symbol);
        let bb_width_pct = state
            .regime
            .last_regime_confidence(&self.config.symbol)
            .map(|rc| rc.bb_width_percentile / 100.0);
        let atr_val = state.regime.atr_value(&self.config.symbol);
        let rel_vol = state.regime.relative_volume(&self.config.symbol);

        // Current positions list (just our symbol if we have a position)
        let current_positions: Vec<String> = if state.position.is_some() {
            vec![self.config.symbol.clone()]
        } else {
            vec![]
        };
        drop(state);

        let routed = janus_regime::RoutedSignal {
            strategy: active_strategy,
            regime,
            confidence,
            position_factor,
            reason: "event-loop-brain-gate".to_string(),
            detection_method: janus_regime::DetectionMethod::Statistical,
            methods_agree: None,
            state_probabilities: None,
            expected_duration: None,
            trend_direction: None,
        };

        let decision = pipeline
            .evaluate(
                &self.config.symbol,
                &routed,
                strategy_name,
                &current_positions,
                adx_val,
                bb_width_pct,
                atr_val,
                rel_vol,
            )
            .await;

        match &decision.action {
            TradeAction::Proceed { scale } => {
                info!(
                    "🧠✅ Brain gate APPROVED {} {} (scale={:.2})",
                    strategy_name, self.config.symbol, scale
                );
                Some(*scale)
            }
            TradeAction::ReduceOnly { scale, reason } => {
                info!(
                    "🧠⚠️  Brain gate REDUCE-ONLY {} {} (scale={:.2}): {}",
                    strategy_name, self.config.symbol, scale, reason
                );
                // Allow reduce-only for closing positions; for new entries
                // treat as a scaled-down proceed
                Some(*scale)
            }
            TradeAction::Block { reason, stage } => {
                info!(
                    "🧠🚫 Brain gate BLOCKED {} {} at [{}]: {}",
                    strategy_name, self.config.symbol, stage, reason
                );
                None
            }
        }
    }

    /// Subscribe to the regime bridge broadcast channel.
    ///
    /// Each call returns a new `Receiver` that will receive all future
    /// `BridgedRegimeState` updates. Neuromorphic consumers (hypothalamus,
    /// amygdala) should call this at startup and process events from the
    /// returned receiver.
    pub fn subscribe_regime_bridge(&self) -> tokio::sync::broadcast::Receiver<BridgedRegimeState> {
        self.regime_bridge_tx.subscribe()
    }

    /// Get a clone of the regime bridge broadcast sender.
    ///
    /// Use this to share the sender with the regime bridge gRPC server so
    /// that `StreamRegimeUpdates` subscribers receive events directly from
    /// the event loop's broadcast channel — no Redis hop required.
    pub fn regime_bridge_sender(&self) -> tokio::sync::broadcast::Sender<BridgedRegimeState> {
        self.regime_bridge_tx.clone()
    }

    /// Start the event loop
    pub async fn run(self: Arc<Self>) -> Result<()> {
        info!("Starting Event Loop for {}", self.config.symbol);
        info!(
            "Mode: {}",
            if self.config.trading_enabled {
                "LIVE TRADING"
            } else {
                "MONITORING ONLY"
            }
        );
        info!(
            "Session boundaries (UTC hours): {:?}",
            self.config.session_start_hours_utc
        );

        // Create WebSocket connection
        let _creds = BybitCredentials::new(
            self.config.bybit_api_key.clone(),
            self.config.bybit_api_secret.clone(),
        );
        let (ws, mut tick_rx) = BybitWebSocket::new_public(self.config.bybit_testnet);

        // Subscribe to orderbook for the symbol
        let subscription = format!("orderbook.50.{}", self.config.symbol);
        info!("Subscribing to: {}", subscription);

        // Spawn WebSocket connection task
        tokio::spawn(async move {
            if let Err(e) = ws.connect(vec![subscription]).await {
                error!("WebSocket connection error: {}", e);
            }
        });

        // Spawn periodic tasks
        let self_for_stats = Arc::clone(&self);
        tokio::spawn(async move {
            let mut stats_timer = interval(Duration::from_secs(10));
            loop {
                stats_timer.tick().await;
                if let Err(e) = self_for_stats.print_stats().await {
                    error!("Error printing stats: {}", e);
                }
            }
        });

        // Spawn daily reset task (for PropFirmValidator)
        let self_for_reset = Arc::clone(&self);
        tokio::spawn(async move {
            let mut reset_timer = interval(Duration::from_secs(86400)); // 24 hours
            loop {
                reset_timer.tick().await;
                info!("Daily reset triggered");
                if let Err(e) = self_for_reset.daily_reset().await {
                    error!("Error in daily reset: {}", e);
                }
            }
        });

        // Spawn session boundary timer for ORB + VWAP session management.
        // Checks every 60s if the current UTC hour matches a configured session
        // start hour. When it does, calls start_session() on ORB and
        // reset_session() on VWAP Scalper.
        let self_for_session = Arc::clone(&self);
        let session_hours = self.config.session_start_hours_utc.clone();
        tokio::spawn(async move {
            let mut session_timer = interval(Duration::from_secs(60));
            let mut last_triggered_hour: Option<u8> = None;
            loop {
                session_timer.tick().await;
                let now = chrono::Utc::now();
                let current_hour = now.format("%H").to_string().parse::<u8>().unwrap_or(255);
                let current_minute = now.format("%M").to_string().parse::<u8>().unwrap_or(255);

                // Trigger within the first minute of a session hour, once per hour
                if session_hours.contains(&current_hour)
                    && current_minute < 1
                    && last_triggered_hour != Some(current_hour)
                {
                    last_triggered_hour = Some(current_hour);
                    info!(
                        "📅 Session boundary reached: {:02}:00 UTC — resetting VWAP + ORB",
                        current_hour
                    );
                    let mut state = self_for_session.state.write().await;
                    state.start_session();
                }

                // Clear the guard once we've moved past minute 0
                if current_minute >= 1 {
                    if last_triggered_hour == Some(current_hour) {
                        // still in the same hour, keep guard
                    } else {
                        last_triggered_hour = None;
                    }
                }
            }
        });

        // Trigger an initial session start so ORB is active immediately
        {
            let mut state = self.state.write().await;
            state.start_session();
        }

        // Main event loop
        info!("Event loop started, waiting for ticks...");
        while let Some(msg) = tick_rx.recv().await {
            match msg {
                WsMessage::Tick(tick) => {
                    if let Err(e) = self.process_tick(tick).await {
                        error!("Error processing tick: {}", e);
                    }
                }
                WsMessage::Trade(trade) => {
                    debug!("Trade: {} {} @ {}", trade.symbol, trade.size, trade.price);

                    // Accumulate trade volume into the regime manager's candle
                    // aggregator so completed candles include volume data for
                    // VWAP Scalper and ORB.
                    {
                        let mut state = self.state.write().await;
                        state.regime.on_trade_volume(&trade.symbol, trade.size);
                    }

                    // Persist trade to QuestDB
                    if let Err(e) = self
                        .questdb
                        .write_trade(TradeWrite {
                            symbol: trade.symbol.clone(),
                            side: trade.side.clone(),
                            price: trade.price,
                            size: trade.size,
                            timestamp_micros: trade.timestamp as i64,
                        })
                        .await
                    {
                        warn!("Failed to write trade to QuestDB: {}", e);
                    }
                }
                WsMessage::OrderUpdate(data) => {
                    info!("Order update: {:?}", data);
                }
                WsMessage::PositionUpdate(data) => {
                    info!("Position update: {:?}", data);
                }
                WsMessage::ExecutionUpdate(data) => {
                    info!("Execution update: {:?}", data);
                }
                _ => {}
            }
        }

        warn!("Event loop terminated - WebSocket channel closed");
        Ok(())
    }

    // ========================================================================
    // Tick Processing
    // ========================================================================

    async fn process_tick(&self, tick: janus_bybit_client::BybitTick) -> Result<()> {
        let mid_price = (tick.bid_price + tick.ask_price) / 2.0;
        let symbol = tick.symbol.clone();

        // Persist tick to QuestDB
        if let Err(e) = self
            .questdb
            .write_tick(TickWrite {
                symbol: tick.symbol.clone(),
                bid_px: tick.bid_price,
                ask_px: tick.ask_price,
                bid_sz: tick.bid_size,
                ask_sz: tick.ask_size,
                timestamp_micros: tick.timestamp as i64,
            })
            .await
        {
            warn!("Failed to write tick to QuestDB: {}", e);
        }

        // Update state
        let mut state = self.state.write().await;
        state.tick_count += 1;
        state.last_price = mid_price;

        // Update indicators
        state.indicators.update(mid_price);

        // Update regime detection — feed tick to the regime manager
        if let Some(routed_signal) = state
            .regime
            .on_tick(&symbol, tick.bid_price, tick.ask_price)
        {
            // Cache regime metadata for position sizing
            state.last_position_factor = routed_signal.position_factor;
            state.last_regime_confidence = routed_signal.confidence;
            state.last_active_strategy = Some(routed_signal.strategy);

            // ── Bridge regime signal to neuromorphic types ────────────
            // Translate the janus-regime RoutedSignal into hypothalamus /
            // amygdala native types and broadcast to subscribers.
            //
            // Extract real indicator values from the regime detector via
            // the clean accessor API on RegimeManager.
            let adx_val = state.regime.adx_value(&symbol);
            let bb_width_pct = state
                .regime
                .last_regime_confidence(&symbol)
                .map(|rc| rc.bb_width_percentile / 100.0); // normalize 0–100 → 0.0–1.0
            let atr_val = state.regime.atr_value(&symbol);

            let rel_vol = state.regime.relative_volume(&symbol);

            let bridged = bridge_regime_signal(
                &symbol,
                &routed_signal,
                adx_val,
                bb_width_pct,
                atr_val,
                rel_vol,
            );
            // Non-blocking send: if no subscribers are listening, the
            // message is silently dropped. Subscribers that lag behind
            // receive a Lagged error and can skip to the latest state.
            let _ = self.regime_bridge_tx.send(bridged.clone());

            // Record bridge state in Prometheus metrics
            if self.metrics.is_some()
                && let Some(regime_metrics) = state.regime.metrics()
            {
                regime_metrics.record_bridge_state(
                    &symbol,
                    bridged.hypothalamus_regime.to_prometheus_i64(),
                    bridged.amygdala_regime.to_prometheus_i64(),
                    bridged.position_scale,
                );
            }

            // Log regime state periodically
            if state.tick_count % 500 == 0 {
                info!(
                    "📊 Regime: {} | Strategy: {} | Conf: {:.0}% | Size: {:.0}%{}",
                    routed_signal.regime,
                    routed_signal.strategy,
                    routed_signal.confidence * 100.0,
                    routed_signal.position_factor * 100.0,
                    routed_signal
                        .methods_agree
                        .map(|a| format!(" | Agree: {}", if a { "✓" } else { "✗" }))
                        .unwrap_or_default()
                );
                info!(
                    "🧠 Bridge: {} | Scale: {:.0}% | Risk: {}",
                    bridged.hypothalamus_regime,
                    bridged.position_scale * 100.0,
                    if bridged.is_high_risk { "HIGH" } else { "low" },
                );
            }

            // ── Mean Reversion: feed completed candle to the MR strategy ──
            // The regime manager just completed a candle (that's why we got a
            // RoutedSignal). Grab the candle and update MeanReversionStrategy.
            if let Some(candle) = state.regime.last_candle(&symbol) {
                let mr_result = state.mean_reversion.update_hlc_with_reason(
                    candle.high,
                    candle.low,
                    candle.close,
                );

                // Only act on MR signals when the regime recommends MeanReversion
                if routed_signal.strategy == ActiveStrategy::MeanReversion {
                    match mr_result.signal {
                        MeanReversionSignal::Buy => {
                            let mr_sig_str = "MR_BUY".to_string();
                            if state.last_mr_signal.as_ref() != Some(&mr_sig_str) {
                                state.last_mr_signal = Some(mr_sig_str);

                                // Compute regime-scaled position factor
                                let regime_factor = MeanReversionStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.mean_reversion.last_atr().unwrap_or(mid_price * 0.02);
                                let stop = state
                                    .mean_reversion
                                    .stop_loss()
                                    .unwrap_or(mid_price - 2.0 * atr);
                                let tp = state
                                    .mean_reversion
                                    .take_profit()
                                    .unwrap_or(mid_price + 3.0 * atr);

                                info!(
                                    "🟢 MR BUY SIGNAL: {} | BB %B={:.2} | RSI={:.1} | Factor={:.0}%",
                                    mr_result.reason,
                                    mr_result
                                        .bb_values
                                        .as_ref()
                                        .map(|b| b.percent_b)
                                        .unwrap_or(0.0),
                                    mr_result.rsi.unwrap_or(0.0),
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "mean_reversion",
                                        "BUY",
                                        routed_signal.confidence,
                                    );
                                }

                                // Release lock before async order handling
                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_mean_reversion_buy(price, atr, stop, tp, regime_factor)
                                    .await
                                {
                                    error!("Error handling MR buy signal: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        MeanReversionSignal::Sell => {
                            let mr_sig_str = "MR_SELL".to_string();
                            if state.last_mr_signal.as_ref() != Some(&mr_sig_str) {
                                state.last_mr_signal = Some(mr_sig_str);

                                let regime_factor = MeanReversionStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.mean_reversion.last_atr().unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 MR SELL SIGNAL: {} | BB %B={:.2} | RSI={:.1} | Factor={:.0}%",
                                    mr_result.reason,
                                    mr_result
                                        .bb_values
                                        .as_ref()
                                        .map(|b| b.percent_b)
                                        .unwrap_or(0.0),
                                    mr_result.rsi.unwrap_or(0.0),
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "mean_reversion",
                                        "SELL",
                                        routed_signal.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_mean_reversion_sell(price, atr).await {
                                    error!("Error handling MR sell signal: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        MeanReversionSignal::Hold => {
                            // Reset duplicate-signal guard so the next real signal fires
                            state.last_mr_signal = None;
                        }
                    }
                } else {
                    // Regime is NOT MeanReversion — clear the MR signal guard
                    state.last_mr_signal = None;
                }

                // ── VWAP Scalper: feed same candle ────────────────────────
                // VWAP Scalper runs when regime is MeanReverting or Uncertain.
                // It uses volume data from aggregated trade messages.
                if VwapScalperStrategy::should_trade_in_regime(&routed_signal.regime) {
                    let vwap_result = state.vwap_scalper.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );

                    match vwap_result.signal {
                        VwapSignal::Buy => {
                            let vwap_sig_str = "VWAP_BUY".to_string();
                            if state.last_vwap_signal.as_ref() != Some(&vwap_sig_str) {
                                state.last_vwap_signal = Some(vwap_sig_str);

                                let regime_factor = VwapScalperStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state.vwap_scalper.last_atr().unwrap_or(mid_price * 0.02);
                                let stop = vwap_result.stop_loss.unwrap_or(mid_price - 2.0 * atr);
                                let tp = vwap_result.take_profit.unwrap_or(mid_price + 3.0 * atr);

                                info!(
                                    "🟢 VWAP BUY: {} | VWAP={:.2} | Bands=[{:.2}, {:.2}] | Conf={:.0}% | Factor={:.0}%",
                                    vwap_result.reason,
                                    vwap_result.vwap.unwrap_or(0.0),
                                    vwap_result.lower_band.unwrap_or(0.0),
                                    vwap_result.upper_band.unwrap_or(0.0),
                                    vwap_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "vwap_scalper",
                                        "BUY",
                                        vwap_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_vwap_buy(price, atr, stop, tp, regime_factor)
                                    .await
                                {
                                    error!("Error handling VWAP buy signal: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        VwapSignal::Sell => {
                            let vwap_sig_str = "VWAP_SELL".to_string();
                            if state.last_vwap_signal.as_ref() != Some(&vwap_sig_str) {
                                state.last_vwap_signal = Some(vwap_sig_str);

                                let regime_factor = VwapScalperStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state.vwap_scalper.last_atr().unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 VWAP SELL: {} | VWAP={:.2} | Conf={:.0}% | Factor={:.0}%",
                                    vwap_result.reason,
                                    vwap_result.vwap.unwrap_or(0.0),
                                    vwap_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "vwap_scalper",
                                        "SELL",
                                        vwap_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_vwap_sell(price, atr).await {
                                    error!("Error handling VWAP sell signal: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        VwapSignal::Hold => {
                            state.last_vwap_signal = None;
                        }
                    }
                } else {
                    state.last_vwap_signal = None;
                }

                // ── Opening Range Breakout: feed same candle ─────────────
                // ORB runs when regime is MeanReverting or Uncertain. It
                // requires start_session() to have been called by the
                // session timer before breakout signals fire.
                if OrbStrategy::should_trade_in_regime(&routed_signal.regime) {
                    let orb_result = state.orb.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );

                    match orb_result.signal {
                        OrbSignal::BuyBreakout => {
                            let orb_sig_str = "ORB_BUY".to_string();
                            if state.last_orb_signal.as_ref() != Some(&orb_sig_str) {
                                state.last_orb_signal = Some(orb_sig_str);

                                let regime_factor = OrbStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state.orb.last_atr().unwrap_or(mid_price * 0.02);
                                let stop = orb_result.stop_loss.unwrap_or(mid_price - 2.0 * atr);
                                let tp = orb_result.take_profit.unwrap_or(mid_price + 4.0 * atr);

                                info!(
                                    "🟢 ORB BUY BREAKOUT: {} | Range=[{:.2}, {:.2}] | Size={:.2} | Conf={:.0}% | Factor={:.0}%",
                                    orb_result.reason,
                                    orb_result.range_low.unwrap_or(0.0),
                                    orb_result.range_high.unwrap_or(0.0),
                                    orb_result.range_size.unwrap_or(0.0),
                                    orb_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "orb",
                                        "BUY",
                                        orb_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_orb_buy(price, atr, stop, tp, regime_factor)
                                    .await
                                {
                                    error!("Error handling ORB buy breakout: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        OrbSignal::SellBreakout => {
                            let orb_sig_str = "ORB_SELL".to_string();
                            if state.last_orb_signal.as_ref() != Some(&orb_sig_str) {
                                state.last_orb_signal = Some(orb_sig_str);

                                let regime_factor = OrbStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state.orb.last_atr().unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 ORB SELL BREAKOUT: {} | Range=[{:.2}, {:.2}] | Conf={:.0}% | Factor={:.0}%",
                                    orb_result.reason,
                                    orb_result.range_low.unwrap_or(0.0),
                                    orb_result.range_high.unwrap_or(0.0),
                                    orb_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "orb",
                                        "SELL",
                                        orb_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_orb_sell(price, atr).await {
                                    error!("Error handling ORB sell breakout: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        OrbSignal::Forming => {
                            // ORB range is still forming — log periodically
                            if state.tick_count % 500 == 0 {
                                debug!(
                                    "🔷 ORB forming: range [{:.2}, {:.2}], candle {}/{}",
                                    orb_result.range_low.unwrap_or(0.0),
                                    orb_result.range_high.unwrap_or(0.0),
                                    state.orb.candle_count(),
                                    state.orb.config().opening_range_candles,
                                );
                            }
                            state.last_orb_signal = None;
                        }
                        OrbSignal::Hold => {
                            state.last_orb_signal = None;
                        }
                    }
                } else {
                    state.last_orb_signal = None;
                }

                // ── Bollinger Squeeze Breakout: feed same candle ──────────
                // BB Squeeze complements MR: MR profits from oscillations
                // within the range, while BB Squeeze captures the breakout
                // when the range ends. Both run when regime is MeanReverting.
                if SqueezeBreakoutStrategy::should_trade_in_regime(&routed_signal.regime) {
                    let sq_result = state.squeeze_breakout.update_ohlc(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                    );

                    match sq_result.signal {
                        SqueezeBreakoutSignal::BuyBreakout => {
                            let sq_sig_str = "SQ_BUY".to_string();
                            if state.last_squeeze_signal.as_ref() != Some(&sq_sig_str) {
                                state.last_squeeze_signal = Some(sq_sig_str);

                                let regime_factor = SqueezeBreakoutStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state
                                    .squeeze_breakout
                                    .last_atr()
                                    .unwrap_or(mid_price * 0.02);
                                let stop = sq_result.stop_loss.unwrap_or(mid_price - 2.0 * atr);
                                let tp = sq_result.take_profit.unwrap_or(mid_price + 4.0 * atr);

                                info!(
                                    "🟢 SQUEEZE BUY BREAKOUT: {} | Squeeze={} bars | ADX={:.1} | Conf={:.0}% | Factor={:.0}%",
                                    sq_result.reason,
                                    sq_result.squeeze_duration,
                                    sq_result.adx.unwrap_or(0.0),
                                    sq_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "squeeze_breakout",
                                        "BUY",
                                        sq_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_squeeze_breakout_buy(
                                        price,
                                        atr,
                                        stop,
                                        tp,
                                        regime_factor,
                                    )
                                    .await
                                {
                                    error!("Error handling squeeze breakout buy: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        SqueezeBreakoutSignal::SellBreakout => {
                            let sq_sig_str = "SQ_SELL".to_string();
                            if state.last_squeeze_signal.as_ref() != Some(&sq_sig_str) {
                                state.last_squeeze_signal = Some(sq_sig_str);

                                let regime_factor = SqueezeBreakoutStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state
                                    .squeeze_breakout
                                    .last_atr()
                                    .unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 SQUEEZE SELL BREAKOUT: {} | Squeeze={} bars | ADX={:.1} | Conf={:.0}% | Factor={:.0}%",
                                    sq_result.reason,
                                    sq_result.squeeze_duration,
                                    sq_result.adx.unwrap_or(0.0),
                                    sq_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "squeeze_breakout",
                                        "SELL",
                                        sq_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_squeeze_breakout_sell(price, atr).await
                                {
                                    error!("Error handling squeeze breakout sell: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        SqueezeBreakoutSignal::Squeeze => {
                            // In squeeze — log periodically
                            if state.tick_count % 500 == 0 {
                                debug!(
                                    "🔶 In squeeze: {} candles | BB width={:.2}%",
                                    sq_result.squeeze_duration,
                                    sq_result.bb_values.as_ref().map(|b| b.width).unwrap_or(0.0),
                                );
                            }
                            // Reset duplicate guard so next breakout fires
                            state.last_squeeze_signal = None;
                        }
                        SqueezeBreakoutSignal::Hold => {
                            state.last_squeeze_signal = None;
                        }
                    }
                } else {
                    // Regime not suitable for squeeze — clear signal guard
                    state.last_squeeze_signal = None;
                }

                // ════════════════════════════════════════════════════════════
                // TREND-FOLLOWING STRATEGIES (Kraken-ported)
                // ════════════════════════════════════════════════════════════

                // ── EMA Ribbon Scalper: 8/13/21 ribbon pullback ───────────
                if EmaRibbonScalperStrategy::should_trade_in_regime(&routed_signal.regime) {
                    let ribbon_result = state.ema_ribbon.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );

                    match ribbon_result.signal {
                        EmaRibbonSignal::Buy => {
                            let sig_str = "RIBBON_BUY".to_string();
                            if state.last_ema_ribbon_signal.as_ref() != Some(&sig_str) {
                                state.last_ema_ribbon_signal = Some(sig_str);

                                let regime_factor = EmaRibbonScalperStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state.ema_ribbon.last_atr().unwrap_or(mid_price * 0.02);
                                let stop = ribbon_result.stop_loss.unwrap_or(mid_price - 2.0 * atr);
                                let tp = ribbon_result.take_profit.unwrap_or(mid_price + 4.0 * atr);

                                info!(
                                    "🟢 RIBBON BUY: {} | EMA8={:.2} EMA13={:.2} EMA21={:.2} | Vol={:.1}x | Conf={:.0}% | Factor={:.0}%",
                                    ribbon_result.reason,
                                    ribbon_result.fast_ema.unwrap_or(0.0),
                                    ribbon_result.mid_ema.unwrap_or(0.0),
                                    ribbon_result.slow_ema.unwrap_or(0.0),
                                    ribbon_result.volume_multiplier,
                                    ribbon_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "ema_ribbon",
                                        "BUY",
                                        ribbon_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_ema_ribbon_buy(price, atr, stop, tp, regime_factor)
                                    .await
                                {
                                    error!("Error handling EMA Ribbon buy: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        EmaRibbonSignal::Sell => {
                            let sig_str = "RIBBON_SELL".to_string();
                            if state.last_ema_ribbon_signal.as_ref() != Some(&sig_str) {
                                state.last_ema_ribbon_signal = Some(sig_str);

                                let regime_factor = EmaRibbonScalperStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr = state.ema_ribbon.last_atr().unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 RIBBON SELL: {} | Conf={:.0}% | Factor={:.0}%",
                                    ribbon_result.reason,
                                    ribbon_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "ema_ribbon",
                                        "SELL",
                                        ribbon_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_ema_ribbon_sell(price, atr).await {
                                    error!("Error handling EMA Ribbon sell: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        EmaRibbonSignal::Hold => {
                            state.last_ema_ribbon_signal = None;
                        }
                    }
                } else {
                    state.last_ema_ribbon_signal = None;
                }

                // ── Trend Pullback: Fibonacci + candlestick patterns ─────
                if TrendPullbackStrategy::should_trade_in_regime(&routed_signal.regime) {
                    let tp_result = state.trend_pullback.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );

                    match tp_result.signal {
                        TrendPullbackSignal::Buy => {
                            let sig_str = "TP_BUY".to_string();
                            if state.last_trend_pullback_signal.as_ref() != Some(&sig_str) {
                                state.last_trend_pullback_signal = Some(sig_str);

                                let regime_factor = TrendPullbackStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.trend_pullback.last_atr().unwrap_or(mid_price * 0.02);
                                let stop = tp_result.stop_loss.unwrap_or(mid_price - 2.0 * atr);
                                let tp = tp_result.take_profit.unwrap_or(mid_price + 4.0 * atr);

                                info!(
                                    "🟢 PULLBACK BUY: {} | EMA50={:.2} EMA200={:.2} | RSI={:.1} | Fib={} | Pattern={} | Conf={:.0}% | Factor={:.0}%",
                                    tp_result.reason,
                                    tp_result.ema_short.unwrap_or(0.0),
                                    tp_result.ema_long.unwrap_or(0.0),
                                    tp_result.rsi.unwrap_or(0.0),
                                    tp_result
                                        .fib_level_hit
                                        .map(|f| format!("{:.3}", f))
                                        .as_deref()
                                        .unwrap_or("none"),
                                    tp_result.candle_pattern.as_deref().unwrap_or("none"),
                                    tp_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "trend_pullback",
                                        "BUY",
                                        tp_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_trend_pullback_buy(price, atr, stop, tp, regime_factor)
                                    .await
                                {
                                    error!("Error handling Trend Pullback buy: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        TrendPullbackSignal::Sell => {
                            let sig_str = "TP_SELL".to_string();
                            if state.last_trend_pullback_signal.as_ref() != Some(&sig_str) {
                                state.last_trend_pullback_signal = Some(sig_str);

                                let regime_factor = TrendPullbackStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.trend_pullback.last_atr().unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 PULLBACK SELL: {} | Conf={:.0}% | Factor={:.0}%",
                                    tp_result.reason,
                                    tp_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "trend_pullback",
                                        "SELL",
                                        tp_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_trend_pullback_sell(price, atr).await {
                                    error!("Error handling Trend Pullback sell: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        TrendPullbackSignal::Hold => {
                            state.last_trend_pullback_signal = None;
                        }
                    }
                } else {
                    state.last_trend_pullback_signal = None;
                }

                // ── Momentum Surge: sudden price moves + volume spikes ───
                if MomentumSurgeStrategy::should_trade_in_regime(&routed_signal.regime) {
                    let ms_result = state.momentum_surge.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );

                    match ms_result.signal {
                        MomentumSurgeSignal::Buy => {
                            let sig_str = "SURGE_BUY".to_string();
                            if state.last_momentum_surge_signal.as_ref() != Some(&sig_str) {
                                state.last_momentum_surge_signal = Some(sig_str);

                                let regime_factor = MomentumSurgeStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.momentum_surge.last_atr().unwrap_or(mid_price * 0.02);
                                let stop = ms_result.stop_loss.unwrap_or(mid_price - 2.0 * atr);
                                let tp = ms_result.take_profit.unwrap_or(mid_price + 4.0 * atr);

                                info!(
                                    "🟢 SURGE BUY: {} | Δ={:.2}% | Vol={:.1}x | Conf={:.0}% | Factor={:.0}%",
                                    ms_result.reason,
                                    ms_result.price_change_pct.unwrap_or(0.0),
                                    ms_result.volume_multiplier,
                                    ms_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "momentum_surge",
                                        "BUY",
                                        ms_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_momentum_surge_buy(price, atr, stop, tp, regime_factor)
                                    .await
                                {
                                    error!("Error handling Momentum Surge buy: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        MomentumSurgeSignal::Sell => {
                            let sig_str = "SURGE_SELL".to_string();
                            if state.last_momentum_surge_signal.as_ref() != Some(&sig_str) {
                                state.last_momentum_surge_signal = Some(sig_str);

                                let regime_factor = MomentumSurgeStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.momentum_surge.last_atr().unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 SURGE SELL: {} | Δ={:.2}% | Conf={:.0}% | Factor={:.0}%",
                                    ms_result.reason,
                                    ms_result.price_change_pct.unwrap_or(0.0),
                                    ms_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "momentum_surge",
                                        "SELL",
                                        ms_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_momentum_surge_sell(price, atr).await {
                                    error!("Error handling Momentum Surge sell: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        MomentumSurgeSignal::Hold => {
                            state.last_momentum_surge_signal = None;
                        }
                    }
                } else {
                    state.last_momentum_surge_signal = None;
                }

                // ── Multi-TF Trend: EMA 50/200 + ADX (Trending only) ─────
                // ── HTF Aggregator: feed LTF candle for trend detection ──
                // Aggregates N LTF candles into one HTF candle, then computes
                // an EMA-cross trend direction. The result is fed into
                // Multi-TF Trend, EMA Ribbon Scalper, and Trend Pullback.
                if let Some(htf_trend) = state.htf_aggregator.feed(
                    candle.open,
                    candle.high,
                    candle.low,
                    candle.close,
                    candle.volume,
                ) {
                    state.multi_tf_trend.set_htf_trend(htf_trend);
                    state.ema_ribbon.set_htf_trend(htf_trend);
                    state.trend_pullback.set_htf_trend(htf_trend);
                    if state.tick_count % 500 == 0 {
                        info!(
                            "📈 HTF trend updated: {:?} (HTF candle #{})",
                            htf_trend, state.htf_aggregator.candle_count,
                        );
                    }
                }

                // ── Multi-TF Trend: EMA 50/200 + ADX (Trending only) ─────
                if MultiTfTrendStrategy::should_trade_in_regime(&routed_signal.regime) {
                    let mtf_result = state.multi_tf_trend.update_ohlcv(
                        candle.open,
                        candle.high,
                        candle.low,
                        candle.close,
                        candle.volume,
                    );

                    match mtf_result.signal {
                        MultiTfSignal::Buy => {
                            let sig_str = "MTF_BUY".to_string();
                            if state.last_multi_tf_signal.as_ref() != Some(&sig_str) {
                                state.last_multi_tf_signal = Some(sig_str);

                                let regime_factor = MultiTfTrendStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.multi_tf_trend.last_atr().unwrap_or(mid_price * 0.02);
                                let stop = mtf_result.stop_loss.unwrap_or(mid_price - 2.0 * atr);
                                let tp = mtf_result.take_profit.unwrap_or(mid_price + 4.0 * atr);

                                info!(
                                    "🟢 MTF BUY: {} | EMA50={:.2} EMA200={:.2} | ADX={:.1} | HTF={} | Conf={:.0}% | Factor={:.0}%",
                                    mtf_result.reason,
                                    mtf_result.ema_short.unwrap_or(0.0),
                                    mtf_result.ema_long.unwrap_or(0.0),
                                    mtf_result.adx.unwrap_or(0.0),
                                    mtf_result
                                        .htf_trend
                                        .as_ref()
                                        .map(|t| format!("{:?}", t))
                                        .as_deref()
                                        .unwrap_or("none"),
                                    mtf_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "multi_tf_trend",
                                        "BUY",
                                        mtf_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self
                                    .handle_multi_tf_buy(price, atr, stop, tp, regime_factor)
                                    .await
                                {
                                    error!("Error handling Multi-TF buy: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        MultiTfSignal::Sell => {
                            let sig_str = "MTF_SELL".to_string();
                            if state.last_multi_tf_signal.as_ref() != Some(&sig_str) {
                                state.last_multi_tf_signal = Some(sig_str);

                                let regime_factor = MultiTfTrendStrategy::regime_size_factor(
                                    &routed_signal.regime,
                                    routed_signal.confidence,
                                )
                                .unwrap_or(routed_signal.position_factor);

                                let atr =
                                    state.multi_tf_trend.last_atr().unwrap_or(mid_price * 0.02);

                                info!(
                                    "🔴 MTF SELL: {} | ADX={:.1} | Conf={:.0}% | Factor={:.0}%",
                                    mtf_result.reason,
                                    mtf_result.adx.unwrap_or(0.0),
                                    mtf_result.confidence * 100.0,
                                    regime_factor * 100.0,
                                );

                                // Record signal generation metric
                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics().record_strategy_signal_generated(
                                        "multi_tf_trend",
                                        "SELL",
                                        mtf_result.confidence,
                                    );
                                }

                                let price = mid_price;
                                drop(state);
                                if let Err(e) = self.handle_multi_tf_sell(price, atr).await {
                                    error!("Error handling Multi-TF sell: {}", e);
                                }
                                return Ok(());
                            }
                        }
                        MultiTfSignal::Hold => {
                            state.last_multi_tf_signal = None;
                        }
                    }
                } else {
                    state.last_multi_tf_signal = None;
                }
            }
        }

        // Skip if indicators not ready
        if !state.is_ready() {
            if state.tick_count % 100 == 0 {
                debug!("Warming up indicators... {}/21 candles", state.tick_count);
            }
            return Ok(());
        }

        // Get indicator values
        let ema8 = state.indicators.ema8();
        let ema21 = state.indicators.ema21();
        let atr = state.indicators.atr();

        // Check for signal from EMA strategy
        let signal = state.strategy.check_signal(ema8, ema21);

        // Gate strategy execution based on regime classification.
        // If the regime router recommends NoTrade, skip signal processing.
        // If it recommends MeanReversion, the EMA Flip signal is suppressed
        // because the mean reversion path above already handles it.
        let regime_allows_ema = match &state.last_active_strategy {
            Some(ActiveStrategy::TrendFollowing) => true,
            Some(ActiveStrategy::NoTrade) => false,
            Some(ActiveStrategy::MeanReversion) => false,
            None => true, // Regime not ready yet — allow EMA as fallback
        };

        // Only act on new signals (avoid repeated signals)
        let signal_str = format!("{:?}", signal);
        if state.last_signal.as_ref() != Some(&signal_str) {
            state.last_signal = Some(signal_str.clone());

            match signal {
                janus_strategies::Signal::Buy => {
                    if regime_allows_ema {
                        info!(
                            "🟢 BUY SIGNAL: EMA8 ({:.2}) crossed above EMA21 ({:.2})",
                            ema8, ema21
                        );
                        // Record signal generation metric
                        if let Some(m) = self.metrics.as_ref() {
                            m.signal_metrics().record_strategy_signal_generated(
                                "ema_flip",
                                "BUY",
                                state.last_regime_confidence,
                            );
                        }
                        drop(state); // Release lock before async calls
                        if let Err(e) = self.handle_buy_signal(mid_price, atr).await {
                            error!("Error handling buy signal: {}", e);
                        }
                    } else {
                        info!(
                            "🚫 BUY SIGNAL suppressed by regime (strategy: {})",
                            state
                                .last_active_strategy
                                .as_ref()
                                .map_or("n/a".to_string(), |s| s.to_string())
                        );
                    }
                }
                janus_strategies::Signal::Sell => {
                    if regime_allows_ema {
                        info!(
                            "🔴 SELL SIGNAL: EMA8 ({:.2}) crossed below EMA21 ({:.2})",
                            ema8, ema21
                        );
                        // Record signal generation metric
                        if let Some(m) = self.metrics.as_ref() {
                            m.signal_metrics().record_strategy_signal_generated(
                                "ema_flip",
                                "SELL",
                                state.last_regime_confidence,
                            );
                        }
                        drop(state); // Release lock before async calls
                        if let Err(e) = self.handle_sell_signal(mid_price, atr).await {
                            error!("Error handling sell signal: {}", e);
                        }
                    } else {
                        info!(
                            "🚫 SELL SIGNAL suppressed by regime (strategy: {})",
                            state
                                .last_active_strategy
                                .as_ref()
                                .map_or("n/a".to_string(), |s| s.to_string())
                        );
                    }
                }
                janus_strategies::Signal::Close => {
                    let position = state.position.clone();
                    let source = position.as_ref().map(|p| p.source_strategy.clone());
                    drop(state);

                    if let Some(pos) = position {
                        let strategy_label =
                            format!("{:?}", source.unwrap_or(PositionSource::EmaFlip));
                        info!(
                            "🟡 CLOSE SIGNAL: Closing {} position for {} @ {:.2}",
                            pos.side, pos.symbol, mid_price
                        );

                        if self.config.trading_enabled {
                            let close_side = if pos.side == "Buy" {
                                OrderSide::Sell
                            } else {
                                OrderSide::Buy
                            };

                            if let Err(e) = self
                                .execute_order(close_side, pos.size, None, 0.0, 0.0)
                                .await
                            {
                                error!("Failed to close position on Close signal: {}", e);
                            } else {
                                let pnl = if pos.side == "Buy" {
                                    (mid_price - pos.entry_price) * pos.size
                                } else {
                                    (pos.entry_price - mid_price) * pos.size
                                };

                                info!("💰 Position closed via Close signal — P&L: ${:.2}", pnl);

                                if let Some(m) = self.metrics.as_ref() {
                                    m.signal_metrics()
                                        .record_strategy_position_closed(&strategy_label, pnl);
                                }

                                let mut validator = self.validator.write().await;
                                let new_balance = validator.account.current_balance + pnl;
                                validator.update_balance(new_balance);
                                drop(validator);

                                let mut state = self.state.write().await;
                                state.position = None;
                            }
                        } else {
                            info!("📊 MONITORING MODE - Close signal not executed");
                        }
                    } else {
                        debug!("Close signal received but no open position");
                    }
                }
                janus_strategies::Signal::None => {
                    // No signal, just log indicators occasionally
                    if state.tick_count % 500 == 0 {
                        debug!(
                            "Price: {:.2} | EMA8: {:.2} | EMA21: {:.2} | ATR: {:.2}",
                            mid_price, ema8, ema21, atr
                        );
                    }
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // EMA Flip Signal Handlers
    // ========================================================================

    /// Handle buy signal from EMA Flip strategy
    async fn handle_buy_signal(&self, price: f64, atr: f64) -> Result<()> {
        // Check if we already have a position
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring buy signal");
            return Ok(());
        }
        drop(state);

        // Calculate stop loss and take profit
        let stop_loss = price - (2.0 * atr);
        let take_profit = price + (3.0 * atr);

        // Calculate position size
        let risk_amount = self.config.account_size * self.config.max_risk_per_trade;
        let risk_per_unit = price - stop_loss;
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        // Validate with PropFirmValidator
        let validator = self.validator.read().await;
        let result = validator.validate_trade(
            &self.config.symbol,
            price,
            stop_loss,
            position_size,
            1, // leverage
        );

        if !result.compliant {
            warn!("❌ Validator REJECTED buy signal: {:?}", result.violations);
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("ema_flip");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("ema_flip");
        }
        info!("✅ Validator APPROVED buy signal");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4}",
            price, stop_loss, take_profit, position_size
        );

        // Persist signal to QuestDB
        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write signal to QuestDB: {}", e);
        }

        // Execute order if trading enabled
        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("ema_flip").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("ema_flip");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute buy order: {}", e);
                return Err(e);
            }

            // Update state with position
            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::EmaFlip,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("ema_flip");
            }
        } else {
            info!("📊 MONITORING MODE - Order not executed");
        }

        Ok(())
    }

    /// Handle sell signal from EMA Flip strategy
    async fn handle_sell_signal(&self, price: f64, _atr: f64) -> Result<()> {
        // Check if we have a position to close
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            info!("Closing position: {} @ {:.2}", pos.side, price);

            // Close the position
            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close position: {}", e);
                    return Err(e);
                }

                // Calculate P&L
                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 Position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("ema_flip", pnl);
                }

                // Update validator with new balance
                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                // Clear position
                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - Position close not executed");
            }
        } else {
            info!("No position to close on sell signal");
        }

        Ok(())
    }

    // ========================================================================
    // Mean Reversion Signal Handlers
    // ========================================================================

    /// Handle buy signal from Mean Reversion strategy.
    ///
    /// Uses regime-scaled position sizing: the `regime_factor` (0.0–1.0) is
    /// derived from `MeanReversionStrategy::regime_size_factor()` and scales
    /// the base position size down when confidence is low.
    async fn handle_mean_reversion_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring MR buy signal");
            return Ok(());
        }
        drop(state);

        // Calculate position size with regime scaling
        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("MR buy: computed zero position size, skipping");
            return Ok(());
        }

        // Validate with PropFirmValidator
        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED MR buy signal: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("mean_reversion");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("mean_reversion");
        }
        info!("✅ Validator APPROVED MR buy signal");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        // Persist signal to QuestDB
        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "MR_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write MR signal to QuestDB: {}", e);
        }

        // Execute order if trading enabled
        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("mean_reversion").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("mean_reversion");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute MR buy order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::MeanReversion,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("mean_reversion");
            }
        } else {
            info!("📊 MONITORING MODE - MR buy order not executed");
        }

        Ok(())
    }

    /// Handle sell signal from Mean Reversion strategy.
    ///
    /// If we hold a MR-sourced long position, this closes it. Otherwise it logs
    /// the signal. (Short entries from MR are not yet implemented — the strategy
    /// currently only opens longs at the lower Bollinger Band and exits at the
    /// middle/upper band.)
    async fn handle_mean_reversion_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            // Only close if the position was opened by mean reversion
            if pos.source_strategy != PositionSource::MeanReversion {
                info!(
                    "🚫 MR sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing MR position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close MR position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 MR position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("mean_reversion", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - MR position close not executed");
            }
        } else {
            info!("No position to close on MR sell signal");
        }

        Ok(())
    }

    // ========================================================================
    // Order Execution
    // ========================================================================

    // ========================================================================
    // Bollinger Squeeze Breakout Signal Handlers
    // ========================================================================

    /// Handle buy breakout signal from the Bollinger Squeeze strategy.
    ///
    /// Uses regime-scaled position sizing identical to the MR handler.
    async fn handle_squeeze_breakout_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring squeeze breakout buy signal");
            return Ok(());
        }
        drop(state);

        // Calculate position size with regime scaling
        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("Squeeze buy: computed zero position size, skipping");
            return Ok(());
        }

        // Validate with PropFirmValidator
        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED squeeze breakout buy: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("squeeze_breakout");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("squeeze_breakout");
        }
        info!("✅ Validator APPROVED squeeze breakout buy");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        // Persist signal to QuestDB
        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "SQ_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write squeeze signal to QuestDB: {}", e);
        }

        // Execute order if trading enabled
        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("squeeze_breakout").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("squeeze_breakout");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute squeeze breakout buy order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::SqueezeBreakout,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("squeeze_breakout");
            }
        } else {
            info!("📊 MONITORING MODE - Squeeze breakout buy order not executed");
        }

        Ok(())
    }

    /// Handle sell breakout signal from the Bollinger Squeeze strategy.
    ///
    /// If we hold a SqueezeBreakout-sourced long position, this closes it.
    /// A sell breakout can also open a short position in the future.
    async fn handle_squeeze_breakout_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            // Only close if the position was opened by squeeze breakout
            if pos.source_strategy != PositionSource::SqueezeBreakout {
                info!(
                    "🚫 Squeeze sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing squeeze position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close squeeze position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 Squeeze position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("squeeze_breakout", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - Squeeze position close not executed");
            }
        } else {
            info!("No position to close on squeeze sell breakout signal");
        }

        Ok(())
    }

    // ========================================================================
    // VWAP Scalper Signal Handlers
    // ========================================================================

    /// Handle buy signal from the VWAP Scalper strategy.
    ///
    /// Uses regime-scaled position sizing identical to the MR/Squeeze handlers.
    async fn handle_vwap_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring VWAP buy signal");
            return Ok(());
        }
        drop(state);

        // Calculate position size with regime scaling
        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("VWAP buy: computed zero position size, skipping");
            return Ok(());
        }

        // Validate with PropFirmValidator
        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED VWAP buy signal: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("vwap_scalper");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("vwap_scalper");
        }
        info!("✅ Validator APPROVED VWAP buy signal");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        // Persist signal to QuestDB
        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "VWAP_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write VWAP signal to QuestDB: {}", e);
        }

        // Execute order if trading enabled
        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("vwap_scalper").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("vwap_scalper");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute VWAP buy order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::VwapScalper,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("vwap_scalper");
            }
        } else {
            info!("📊 MONITORING MODE - VWAP buy order not executed");
        }

        Ok(())
    }

    /// Handle sell signal from the VWAP Scalper strategy.
    ///
    /// If we hold a VwapScalper-sourced long position, this closes it.
    async fn handle_vwap_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            // Only close if the position was opened by VWAP scalper
            if pos.source_strategy != PositionSource::VwapScalper {
                info!(
                    "🚫 VWAP sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing VWAP position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close VWAP position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 VWAP position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("vwap_scalper", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - VWAP position close not executed");
            }
        } else {
            info!("No position to close on VWAP sell signal");
        }

        Ok(())
    }

    // ========================================================================
    // Opening Range Breakout Signal Handlers
    // ========================================================================

    /// Handle buy breakout signal from the ORB strategy.
    ///
    /// Uses regime-scaled position sizing. ORB targets range extension as TP.
    async fn handle_orb_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring ORB buy breakout");
            return Ok(());
        }
        drop(state);

        // Calculate position size with regime scaling
        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("ORB buy: computed zero position size, skipping");
            return Ok(());
        }

        // Validate with PropFirmValidator
        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED ORB buy breakout: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics().record_strategy_signal_rejected("orb");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics().record_strategy_signal_approved("orb");
        }
        info!("✅ Validator APPROVED ORB buy breakout");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        // Persist signal to QuestDB
        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "ORB_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write ORB signal to QuestDB: {}", e);
        }

        // Execute order if trading enabled
        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("orb").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics().record_strategy_signal_rejected("orb");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute ORB buy breakout order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::Orb,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics().record_strategy_position_opened("orb");
            }
        } else {
            info!("📊 MONITORING MODE - ORB buy breakout order not executed");
        }

        Ok(())
    }

    /// Handle sell breakout signal from the ORB strategy.
    ///
    /// If we hold an ORB-sourced long position, this closes it.
    /// A sell breakout can also open a short position in the future.
    async fn handle_orb_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            // Only close if the position was opened by ORB
            if pos.source_strategy != PositionSource::Orb {
                info!(
                    "🚫 ORB sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing ORB position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close ORB position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 ORB position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("orb", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - ORB position close not executed");
            }
        } else {
            info!("No position to close on ORB sell breakout signal");
        }

        Ok(())
    }

    // ========================================================================
    // EMA Ribbon Scalper Handlers
    // ========================================================================

    async fn handle_ema_ribbon_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring EMA Ribbon buy signal");
            return Ok(());
        }
        drop(state);

        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("EMA Ribbon buy: computed zero position size, skipping");
            return Ok(());
        }

        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED EMA Ribbon buy: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("ema_ribbon");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("ema_ribbon");
        }
        info!("✅ Validator APPROVED EMA Ribbon buy");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "RIBBON_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write EMA Ribbon signal to QuestDB: {}", e);
        }

        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("ema_ribbon").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("ema_ribbon");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute EMA Ribbon buy order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::EmaRibbon,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("ema_ribbon");
            }
        } else {
            info!("📊 MONITORING MODE - EMA Ribbon buy order not executed");
        }

        Ok(())
    }

    async fn handle_ema_ribbon_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            if pos.source_strategy != PositionSource::EmaRibbon {
                info!(
                    "🚫 EMA Ribbon sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing EMA Ribbon position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close EMA Ribbon position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 EMA Ribbon position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("ema_ribbon", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - EMA Ribbon position close not executed");
            }
        } else {
            info!("No position to close on EMA Ribbon sell signal");
        }

        Ok(())
    }

    // ========================================================================
    // Trend Pullback Handlers
    // ========================================================================

    async fn handle_trend_pullback_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring Trend Pullback buy signal");
            return Ok(());
        }
        drop(state);

        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("Trend Pullback buy: computed zero position size, skipping");
            return Ok(());
        }

        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED Trend Pullback buy: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("trend_pullback");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("trend_pullback");
        }
        info!("✅ Validator APPROVED Trend Pullback buy");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "TP_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write Trend Pullback signal to QuestDB: {}", e);
        }

        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("trend_pullback").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("trend_pullback");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute Trend Pullback buy order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::TrendPullback,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("trend_pullback");
            }
        } else {
            info!("📊 MONITORING MODE - Trend Pullback buy order not executed");
        }

        Ok(())
    }

    async fn handle_trend_pullback_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            if pos.source_strategy != PositionSource::TrendPullback {
                info!(
                    "🚫 Trend Pullback sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing Trend Pullback position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close Trend Pullback position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 Trend Pullback position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("trend_pullback", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - Trend Pullback position close not executed");
            }
        } else {
            info!("No position to close on Trend Pullback sell signal");
        }

        Ok(())
    }

    // ========================================================================
    // Momentum Surge Handlers
    // ========================================================================

    async fn handle_momentum_surge_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring Momentum Surge buy signal");
            return Ok(());
        }
        drop(state);

        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("Momentum Surge buy: computed zero position size, skipping");
            return Ok(());
        }

        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED Momentum Surge buy: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("momentum_surge");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("momentum_surge");
        }
        info!("✅ Validator APPROVED Momentum Surge buy");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "SURGE_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write Momentum Surge signal to QuestDB: {}", e);
        }

        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("momentum_surge").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("momentum_surge");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute Momentum Surge buy order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::MomentumSurge,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("momentum_surge");
            }
        } else {
            info!("📊 MONITORING MODE - Momentum Surge buy order not executed");
        }

        Ok(())
    }

    async fn handle_momentum_surge_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            if pos.source_strategy != PositionSource::MomentumSurge {
                info!(
                    "🚫 Momentum Surge sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing Momentum Surge position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close Momentum Surge position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 Momentum Surge position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("momentum_surge", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - Momentum Surge position close not executed");
            }
        } else {
            info!("No position to close on Momentum Surge sell signal");
        }

        Ok(())
    }

    // ========================================================================
    // Multi-TF Trend Handlers
    // ========================================================================

    async fn handle_multi_tf_buy(
        &self,
        price: f64,
        atr: f64,
        stop_loss: f64,
        take_profit: f64,
        regime_factor: f64,
    ) -> Result<()> {
        let state = self.state.read().await;
        if state.position.is_some() {
            warn!("Already in position, ignoring Multi-TF buy signal");
            return Ok(());
        }
        drop(state);

        let risk_amount = self.config.account_size * self.config.max_risk_per_trade * regime_factor;
        let risk_per_unit = (price - stop_loss).abs();
        let position_size = if risk_per_unit > 0.0 {
            risk_amount / risk_per_unit
        } else {
            0.0
        };

        if position_size <= 0.0 {
            warn!("Multi-TF buy: computed zero position size, skipping");
            return Ok(());
        }

        let validator = self.validator.read().await;
        let result =
            validator.validate_trade(&self.config.symbol, price, stop_loss, position_size, 1);

        if !result.compliant {
            warn!(
                "❌ Validator REJECTED Multi-TF buy: {:?}",
                result.violations
            );
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_signal_rejected("multi_tf_trend");
            }
            return Ok(());
        }
        drop(validator);

        if let Some(m) = self.metrics.as_ref() {
            m.signal_metrics()
                .record_strategy_signal_approved("multi_tf_trend");
        }
        info!("✅ Validator APPROVED Multi-TF buy");
        info!(
            "  Entry: ${:.2} | SL: ${:.2} | TP: ${:.2} | Size: {:.4} | ATR: {:.2} | Factor: {:.0}%",
            price,
            stop_loss,
            take_profit,
            position_size,
            atr,
            regime_factor * 100.0,
        );

        if let Err(e) = self
            .questdb
            .write_signal(SignalWrite {
                symbol: self.config.symbol.clone(),
                signal_type: "MTF_BUY".to_string(),
                entry_price: price,
                stop_loss,
                take_profit,
                position_size,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write Multi-TF signal to QuestDB: {}", e);
        }

        if self.config.trading_enabled {
            // Brain pipeline gate check
            let brain_scale = self.brain_gate_check("multi_tf_trend").await;
            if brain_scale.is_none() && self.brain_pipeline.is_some() {
                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_signal_rejected("multi_tf_trend");
                }
                return Ok(());
            }
            let position_size = position_size * brain_scale.unwrap_or(1.0);

            if let Err(e) = self
                .execute_order(
                    OrderSide::Buy,
                    position_size,
                    Some(price),
                    stop_loss,
                    take_profit,
                )
                .await
            {
                error!("Failed to execute Multi-TF buy order: {}", e);
                return Err(e);
            }

            let mut state = self.state.write().await;
            state.position = Some(Position {
                symbol: self.config.symbol.clone(),
                side: "Buy".to_string(),
                entry_price: price,
                size: position_size,
                stop_loss,
                take_profit,
                timestamp: now_micros(),
                source_strategy: PositionSource::MultiTfTrend,
            });
            if let Some(m) = self.metrics.as_ref() {
                m.signal_metrics()
                    .record_strategy_position_opened("multi_tf_trend");
            }
        } else {
            info!("📊 MONITORING MODE - Multi-TF buy order not executed");
        }

        Ok(())
    }

    async fn handle_multi_tf_sell(&self, price: f64, _atr: f64) -> Result<()> {
        let state = self.state.read().await;
        let position = state.position.clone();
        drop(state);

        if let Some(pos) = position {
            if pos.source_strategy != PositionSource::MultiTfTrend {
                info!(
                    "🚫 Multi-TF sell signal ignored — current position opened by {:?}",
                    pos.source_strategy
                );
                return Ok(());
            }

            info!(
                "Closing Multi-TF position: {} @ {:.2} (entry was {:.2})",
                pos.side, price, pos.entry_price
            );

            if self.config.trading_enabled {
                let close_side = if pos.side == "Buy" {
                    OrderSide::Sell
                } else {
                    OrderSide::Buy
                };

                if let Err(e) = self
                    .execute_order(close_side, pos.size, Some(price), 0.0, 0.0)
                    .await
                {
                    error!("Failed to close Multi-TF position: {}", e);
                    return Err(e);
                }

                let pnl = if pos.side == "Buy" {
                    (price - pos.entry_price) * pos.size
                } else {
                    (pos.entry_price - price) * pos.size
                };

                info!("💰 Multi-TF position closed - P&L: ${:.2}", pnl);

                if let Some(m) = self.metrics.as_ref() {
                    m.signal_metrics()
                        .record_strategy_position_closed("multi_tf_trend", pnl);
                }

                let mut validator = self.validator.write().await;
                let new_balance = validator.account.current_balance + pnl;
                validator.update_balance(new_balance);
                drop(validator);

                let mut state = self.state.write().await;
                state.position = None;
            } else {
                info!("📊 MONITORING MODE - Multi-TF position close not executed");
            }
        } else {
            info!("No position to close on Multi-TF sell signal");
        }

        Ok(())
    }

    // ========================================================================
    // Order Execution
    // ========================================================================

    async fn execute_order(
        &self,
        side: OrderSide,
        quantity: f64,
        price: Option<f64>,
        _stop_loss: f64,
        _take_profit: f64,
    ) -> Result<()> {
        // ── Defense-in-depth: kill switch guard ────────────────────────
        // Even if the brain_gate_check was bypassed (e.g. close/reduce-only
        // orders, or a code path that doesn't call the gate), block order
        // submission when the pipeline kill switch is active.
        if let Some(ref pipeline) = self.brain_pipeline {
            if pipeline.is_killed().await {
                warn!(
                    "🛑 execute_order BLOCKED — pipeline kill switch is active (side={:?}, qty={:.4})",
                    side, quantity
                );
                return Ok(());
            }
        }

        let order_type = if price.is_some() {
            OrderType::Limit
        } else {
            OrderType::Market
        };

        let order = OrderRequest {
            category: "linear".to_string(),
            symbol: self.config.symbol.clone(),
            side,
            order_type,
            qty: format!("{:.4}", quantity),
            price: price.map(|p| format!("{:.2}", p)),
            time_in_force: Some("GTC".to_string()),
            order_link_id: Some(format!("janus_{}", now_micros())),
            reduce_only: Some(false),
            close_on_trigger: Some(false),
        };

        info!("📤 Placing order: {:?}", order);

        let response = self.bybit_rest.place_order(order).await?;

        info!("✅ Order placed: {}", response.order_id);

        // Persist execution to QuestDB
        if let Err(e) = self
            .questdb
            .write_execution(ExecutionWrite {
                order_id: response.order_id.clone(),
                symbol: self.config.symbol.clone(),
                side: format!("{:?}", side),
                price: price.unwrap_or(0.0),
                size: quantity,
                fee: price.unwrap_or(0.0) * quantity * self.config.taker_fee_rate,
                timestamp_micros: now_micros(),
            })
            .await
        {
            warn!("Failed to write execution to QuestDB: {}", e);
        }

        Ok(())
    }

    // ========================================================================
    // Diagnostics & Maintenance
    // ========================================================================

    /// Print statistics
    async fn print_stats(&self) -> Result<()> {
        let state = self.state.read().await;
        let _validator = self.validator.read().await;

        info!("─────────────────────────────────────────");
        info!("📊 Event Loop Statistics");
        info!("  Ticks processed: {}", state.tick_count);
        info!("  Last price: ${:.2}", state.last_price);
        info!("  Indicators ready: {}", state.is_ready());

        if state.is_ready() {
            info!("  EMA8: {:.2}", state.indicators.ema8());
            info!("  EMA21: {:.2}", state.indicators.ema21());
            info!("  ATR: {:.2}", state.indicators.atr());
        }

        // Regime detection status
        let regime_summaries = state.regime.summary();
        for summary in &regime_summaries {
            info!(
                "  Regime [{}]: {} | Strategy: {} | Ready: {}",
                summary.symbol,
                summary.regime.as_ref().map_or(
                    "warming up".to_string(),
                    |r: &janus_regime::MarketRegime| r.to_string()
                ),
                summary
                    .strategy
                    .as_ref()
                    .map_or("n/a".to_string(), |s: &janus_regime::ActiveStrategy| s
                        .to_string()),
                summary.is_ready
            );
        }

        // Mean Reversion strategy status
        if state.mean_reversion.is_ready() {
            info!(
                "  MR: ready={} | in_pos={} | BB={} | RSI={} | ATR={}",
                state.mean_reversion.is_ready(),
                state.mean_reversion.is_in_position(),
                state
                    .mean_reversion
                    .last_bb_values()
                    .map(|b| format!("mid={:.2} w={:.1}%", b.middle, b.width))
                    .unwrap_or_else(|| "n/a".to_string()),
                state
                    .mean_reversion
                    .last_rsi()
                    .map(|r| format!("{:.1}", r))
                    .unwrap_or_else(|| "n/a".to_string()),
                state
                    .mean_reversion
                    .last_atr()
                    .map(|a| format!("{:.2}", a))
                    .unwrap_or_else(|| "n/a".to_string()),
            );
        }

        // VWAP Scalper strategy status
        info!(
            "  VWAP Scalper: ready={} | in_pos={} | VWAP={} | bands={} | ATR={}",
            state.vwap_scalper.is_ready(),
            state.vwap_scalper.is_in_position(),
            state
                .vwap_scalper
                .last_vwap()
                .map(|v| format!("{:.2}", v))
                .unwrap_or_else(|| "n/a".to_string()),
            state
                .vwap_scalper
                .vwap_levels()
                .map(|(_, lower, upper)| format!("[{:.2}, {:.2}]", lower, upper))
                .unwrap_or_else(|| "n/a".to_string()),
            state
                .vwap_scalper
                .last_atr()
                .map(|a| format!("{:.2}", a))
                .unwrap_or_else(|| "n/a".to_string()),
        );

        // ORB strategy status
        info!(
            "  ORB: session={} | range_done={} | in_pos={} | range={} | breakout_fired={}",
            state.orb.is_session_active(),
            state.orb.is_range_complete(),
            state.orb.is_in_position(),
            if state.orb.is_range_complete() {
                format!(
                    "[{:.2}, {:.2}] size={:.2}",
                    state.orb.range_low().unwrap_or(0.0),
                    state.orb.range_high().unwrap_or(0.0),
                    state.orb.range_size().unwrap_or(0.0),
                )
            } else {
                "forming".to_string()
            },
            state.orb.breakout_fired(),
        );

        // Active strategy
        info!(
            "  Active strategy: {}",
            state
                .last_active_strategy
                .as_ref()
                .map_or("n/a (fallback EMA)".to_string(), |s| s.to_string())
        );

        if let Some(pos) = &state.position {
            let unrealized_pnl = if pos.side == "Buy" {
                (state.last_price - pos.entry_price) * pos.size
            } else {
                (pos.entry_price - state.last_price) * pos.size
            };
            info!(
                "  Position: {} {:.4} @ {:.2} (via {:?})",
                pos.side, pos.size, pos.entry_price, pos.source_strategy
            );
            info!("  Unrealized P&L: ${:.2}", unrealized_pnl);
        } else {
            info!("  Position: None");
        }

        let validator = self.validator.read().await;
        let account = validator.get_account_summary();
        info!("  Account balance: ${:.2}", account.current_balance);
        info!(
            "  Daily P&L: ${:.2} ({:.2}%)",
            account.daily_pnl, account.daily_pnl_percent
        );

        // Metrics status
        if self.metrics.is_some() {
            info!("  Prometheus: ✅ connected");
        } else {
            info!("  Prometheus: ❌ not configured");
        }

        info!("─────────────────────────────────────────");

        Ok(())
    }

    /// Daily reset (PropFirmValidator balance reset)
    async fn daily_reset(&self) -> Result<()> {
        let mut validator = self.validator.write().await;
        validator.reset_daily_balance();
        validator.increment_trading_day();
        info!("PropFirmValidator daily reset complete");
        Ok(())
    }

    /// Get a reference to the Prometheus metrics (if configured).
    ///
    /// Useful for exposing the `/metrics` HTTP endpoint in the API server.
    #[allow(dead_code)]
    pub fn metrics(&self) -> Option<&Arc<JanusMetrics>> {
        self.metrics.as_ref()
    }

    /// Get a clone of the QuestDB writer handle.
    ///
    /// Used by the regime bridge consumer to persist bridged regime states
    /// to the `regime_states` table for historical replay and analysis.
    pub fn questdb(&self) -> Arc<QuestDBWriter> {
        Arc::clone(&self.questdb)
    }

    /// Spawn a background task that watches the `regime.toml` file for changes
    /// and hot-reloads tunable parameters (volume lookback overrides, min
    /// confidence, volatile position factor) into the running `RegimeManager`.
    ///
    /// Non-reloadable fields (e.g. `ticks_per_candle`, `detection_method`) are
    /// logged as warnings — a full restart is needed for those.
    ///
    /// Returns `None` if `path` is `None` (no config file configured).
    ///
    /// # Arguments
    ///
    /// * `path`          — Path to the `regime.toml` file (from `REGIME_TOML_PATH`).
    /// * `poll_interval` — How often to check for file changes (e.g. 30 seconds).
    pub fn spawn_config_watcher(
        &self,
        path: Option<&str>,
        poll_interval: std::time::Duration,
    ) -> Option<tokio::task::JoinHandle<()>> {
        let path = path?;
        let owned_path = std::path::PathBuf::from(path);
        let state = Arc::clone(&self.state);
        let handle =
            janus_forward::regime::spawn_regime_config_watcher(owned_path, state, poll_interval);
        Some(handle)
    }

    /// Graceful shutdown
    pub async fn shutdown(self: Arc<Self>) -> Result<()> {
        info!("Shutting down event loop...");

        // Close open position if configured
        if self.config.close_on_shutdown {
            let state = self.state.read().await;
            let position = state.position.clone();
            let last_price = state.last_price;
            drop(state);

            if let Some(pos) = position {
                info!(
                    "🔻 close_on_shutdown: closing {} {} position @ ~{:.2}",
                    pos.symbol, pos.side, last_price
                );

                if self.config.trading_enabled {
                    let close_side = if pos.side == "Buy" {
                        OrderSide::Sell
                    } else {
                        OrderSide::Buy
                    };

                    match self
                        .execute_order(close_side, pos.size, None, 0.0, 0.0)
                        .await
                    {
                        Ok(()) => {
                            let pnl = if pos.side == "Buy" {
                                (last_price - pos.entry_price) * pos.size
                            } else {
                                (pos.entry_price - last_price) * pos.size
                            };
                            info!("💰 Shutdown position closed — P&L: ${:.2}", pnl);

                            if let Some(m) = self.metrics.as_ref() {
                                let label = format!("{:?}", pos.source_strategy);
                                m.signal_metrics()
                                    .record_strategy_position_closed(&label, pnl);
                            }

                            let mut validator = self.validator.write().await;
                            let new_balance = validator.account.current_balance + pnl;
                            validator.update_balance(new_balance);
                            drop(validator);

                            let mut state = self.state.write().await;
                            state.position = None;
                        }
                        Err(e) => {
                            error!("Failed to close position on shutdown: {}", e);
                        }
                    }
                } else {
                    info!("📊 MONITORING MODE — shutdown close not executed");
                }
            } else {
                info!("No open position to close on shutdown");
            }
        }

        // Flush QuestDB
        if let Err(e) = self.questdb.flush().await {
            error!("Error flushing QuestDB: {}", e);
        }

        info!("Event loop shutdown complete");
        Ok(())
    }
}
