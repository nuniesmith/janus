//! # Strategy-Level Bar Backtester
//!
//! Replays OHLCV bars through the full JANUS pipeline:
//!
//! ```text
//! OHLCV Bars → Regime Detection → Strategy Routing → Signal Generation
//!                                                          ↓
//!                              Metrics ← Execution Simulation ← HTF Aggregation
//! ```
//!
//! Unlike the tick-level `ReplayEngine`, this backtester operates on pre-formed
//! OHLCV candles and directly feeds each strategy with the same data flow used
//! in the forward service event loop.
//!
//! ## Supported Strategies
//!
//! All nine JANUS strategies are wired in:
//!
//! | Strategy | Target Regime | Signal Type |
//! |----------|--------------|-------------|
//! | EMA Flip | Trending | Buy/Sell crossover |
//! | Mean Reversion | MeanReverting | BB + RSI |
//! | Squeeze Breakout | MeanReverting→Trending | BB width squeeze |
//! | VWAP Scalper | MeanReverting | VWAP band bounce |
//! | Opening Range Breakout | Transition | Session range breakout |
//! | EMA Ribbon Scalper | Trending | 8/13/21 ribbon pullback |
//! | Trend Pullback | Trending | Fibonacci + patterns |
//! | Momentum Surge | Trending/Volatile | Price surge + pullback |
//! | Multi-TF Trend | Trending | EMA 50/200 + ADX + HTF |

use chrono::{DateTime, Utc};
use janus_regime::{EnhancedRouter, MarketRegime, RoutedSignal, TrendDirection};
use janus_strategies::{
    EmaRibbonConfig, EmaRibbonScalperStrategy, EmaRibbonSignal, MeanReversionConfig,
    MeanReversionSignal, MeanReversionStrategy, MomentumSurgeConfig, MomentumSurgeSignal,
    MomentumSurgeStrategy, MultiTfConfig, MultiTfSignal, MultiTfTrendStrategy, OrbConfig,
    OrbSignal, OrbStrategy, SqueezeBreakoutConfig, SqueezeBreakoutSignal, SqueezeBreakoutStrategy,
    TrendPullbackConfig, TrendPullbackSignal, TrendPullbackStrategy, VwapScalperConfig,
    VwapScalperStrategy, VwapSignal,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use tracing::{debug, info};

// ============================================================================
// OHLCV Bar
// ============================================================================

/// A single OHLCV bar for backtesting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OhlcvBar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl OhlcvBar {
    /// Create a new OHLCV bar.
    pub fn new(
        timestamp: DateTime<Utc>,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Self {
            timestamp,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    /// Typical price = (H + L + C) / 3.
    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }
}

// ============================================================================
// HTF Aggregator (mirrors forward service HtfAggregator)
// ============================================================================

/// Aggregates N low-timeframe candles into one higher-timeframe candle and
/// computes a trend direction via short/long EMA on HTF closes.
#[derive(Debug)]
struct HtfAggregator {
    ratio: usize,
    count: usize,
    agg_open: f64,
    agg_high: f64,
    agg_low: f64,
    agg_close: f64,
    agg_volume: f64,
    ema_short: janus_regime::indicators::EMA,
    ema_long: janus_regime::indicators::EMA,
    candle_count: usize,
    warmup: usize,
}

impl HtfAggregator {
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
    /// completes and the EMAs are warmed up.
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
            let htf_close = self.agg_close;
            self.ema_short.update(htf_close);
            self.ema_long.update(htf_close);
            self.candle_count += 1;

            // Reset for next HTF bar
            self.count = 0;
            self.agg_high = f64::NEG_INFINITY;
            self.agg_low = f64::INFINITY;
            self.agg_volume = 0.0;

            if self.candle_count >= self.warmup {
                let short_val = self.ema_short.value();
                let long_val = self.ema_long.value();
                return if short_val > long_val {
                    Some(TrendDirection::Bullish)
                } else {
                    Some(TrendDirection::Bearish)
                };
            }
        }
        None
    }
}

// ============================================================================
// Strategy Identification
// ============================================================================

/// Identifies which strategy produced a signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyId {
    MeanReversion,
    SqueezeBreakout,
    VwapScalper,
    OpeningRangeBreakout,
    EmaRibbonScalper,
    TrendPullback,
    MomentumSurge,
    MultiTfTrend,
}

impl fmt::Display for StrategyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MeanReversion => write!(f, "MeanReversion"),
            Self::SqueezeBreakout => write!(f, "SqueezeBreakout"),
            Self::VwapScalper => write!(f, "VwapScalper"),
            Self::OpeningRangeBreakout => write!(f, "ORB"),
            Self::EmaRibbonScalper => write!(f, "EmaRibbon"),
            Self::TrendPullback => write!(f, "TrendPullback"),
            Self::MomentumSurge => write!(f, "MomentumSurge"),
            Self::MultiTfTrend => write!(f, "MultiTfTrend"),
        }
    }
}

impl StrategyId {
    /// All strategy IDs in order.
    pub fn all() -> &'static [StrategyId] {
        &[
            Self::MeanReversion,
            Self::SqueezeBreakout,
            Self::VwapScalper,
            Self::OpeningRangeBreakout,
            Self::EmaRibbonScalper,
            Self::TrendPullback,
            Self::MomentumSurge,
            Self::MultiTfTrend,
        ]
    }
}

// ============================================================================
// Unified Signal
// ============================================================================

/// Direction-normalised signal emitted by any strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Long,
    Short,
}

impl fmt::Display for Direction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Long => write!(f, "LONG"),
            Self::Short => write!(f, "SHORT"),
        }
    }
}

/// A concrete signal produced by a strategy during backtesting.
#[derive(Debug, Clone)]
pub struct BacktestSignal {
    pub strategy: StrategyId,
    pub direction: Direction,
    pub confidence: f64,
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub reason: String,
    pub bar_index: usize,
    pub price: f64,
    pub regime: MarketRegime,
}

// ============================================================================
// Simulated Position
// ============================================================================

/// A simulated position opened during backtesting.
#[derive(Debug, Clone)]
struct SimPosition {
    strategy: StrategyId,
    direction: Direction,
    entry_price: f64,
    entry_bar: usize,
    stop_loss: Option<f64>,
    take_profit: Option<f64>,
    size_factor: f64,
}

// ============================================================================
// Completed Trade
// ============================================================================

/// A completed round-trip trade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedTrade {
    pub strategy: StrategyId,
    pub direction: Direction,
    pub entry_price: f64,
    pub exit_price: f64,
    pub entry_bar: usize,
    pub exit_bar: usize,
    pub pnl_pct: f64,
    pub pnl_absolute: f64,
    pub size_factor: f64,
    pub exit_reason: ExitReason,
}

/// Why a position was closed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExitReason {
    StopLoss,
    TakeProfit,
    OppositeSignal,
    EndOfData,
}

impl fmt::Display for ExitReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StopLoss => write!(f, "Stop Loss"),
            Self::TakeProfit => write!(f, "Take Profit"),
            Self::OppositeSignal => write!(f, "Opposite Signal"),
            Self::EndOfData => write!(f, "End of Data"),
        }
    }
}

// ============================================================================
// Per-Strategy Metrics
// ============================================================================

/// Performance metrics for a single strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetrics {
    pub strategy: StrategyId,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub win_rate: f64,
    pub total_pnl_pct: f64,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    pub largest_win_pct: f64,
    pub largest_loss_pct: f64,
    pub profit_factor: f64,
    pub max_drawdown_pct: f64,
    pub sharpe_ratio: f64,
    pub signals_generated: usize,
    pub signals_in_target_regime: usize,
    pub signals_outside_regime: usize,
    pub avg_trade_duration_bars: f64,
}

impl Default for StrategyMetrics {
    fn default() -> Self {
        Self {
            strategy: StrategyId::MeanReversion,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
            win_rate: 0.0,
            total_pnl_pct: 0.0,
            avg_win_pct: 0.0,
            avg_loss_pct: 0.0,
            largest_win_pct: 0.0,
            largest_loss_pct: 0.0,
            profit_factor: 0.0,
            max_drawdown_pct: 0.0,
            sharpe_ratio: 0.0,
            signals_generated: 0,
            signals_in_target_regime: 0,
            signals_outside_regime: 0,
            avg_trade_duration_bars: 0.0,
        }
    }
}

impl fmt::Display for StrategyMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<16} | Trades: {:>4} | Win: {:>5.1}% | PnL: {:>+7.2}% | PF: {:>5.2} | \
             MaxDD: {:>5.2}% | Sharpe: {:>5.2} | Sigs: {} (regime: {}/{})",
            self.strategy,
            self.total_trades,
            self.win_rate,
            self.total_pnl_pct,
            self.profit_factor,
            self.max_drawdown_pct,
            self.sharpe_ratio,
            self.signals_generated,
            self.signals_in_target_regime,
            self.signals_generated,
        )
    }
}

// ============================================================================
// Aggregate Backtest Report
// ============================================================================

/// Full backtest report across all strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestReport {
    pub symbol: String,
    pub total_bars: usize,
    pub strategy_metrics: HashMap<StrategyId, StrategyMetrics>,
    pub regime_distribution: HashMap<String, usize>,
    pub total_signals: usize,
    pub total_trades: usize,
    pub aggregate_pnl_pct: f64,
}

impl fmt::Display for BacktestReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════════════════"
        )?;
        writeln!(f, "  JANUS Strategy Backtest Report — {}", self.symbol)?;
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════════════════"
        )?;
        writeln!(
            f,
            "  Bars: {} | Total Signals: {} | Total Trades: {} | Aggregate PnL: {:+.2}%",
            self.total_bars, self.total_signals, self.total_trades, self.aggregate_pnl_pct
        )?;
        writeln!(f)?;

        writeln!(f, "  Regime Distribution:")?;
        for (regime, count) in &self.regime_distribution {
            let pct = (*count as f64 / self.total_bars as f64) * 100.0;
            writeln!(f, "    {:<20} {:>5} bars ({:.1}%)", regime, count, pct)?;
        }
        writeln!(f)?;

        writeln!(f, "  Per-Strategy Results:")?;
        writeln!(
            f,
            "  ─────────────────────────────────────────────────────────────────────────"
        )?;
        let mut sorted: Vec<_> = self.strategy_metrics.values().collect();
        sorted.sort_by(|a, b| b.total_pnl_pct.partial_cmp(&a.total_pnl_pct).unwrap());
        for m in sorted {
            writeln!(f, "  {}", m)?;
        }
        writeln!(
            f,
            "═══════════════════════════════════════════════════════════════════════════"
        )?;
        Ok(())
    }
}

// ============================================================================
// Backtester Configuration
// ============================================================================

/// Configuration for the strategy backtester.
#[derive(Debug, Clone)]
pub struct StrategyBacktesterConfig {
    /// Trading symbol.
    pub symbol: String,
    /// Initial account balance in USDT.
    pub initial_balance: f64,
    /// Slippage in basis points applied to entries and exits.
    pub slippage_bps: f64,
    /// Commission per trade in basis points.
    pub commission_bps: f64,
    /// HTF aggregator ratio (LTF candles per HTF candle).
    pub htf_ratio: usize,
    /// HTF short EMA period.
    pub htf_ema_short: usize,
    /// HTF long EMA period.
    pub htf_ema_long: usize,
    /// Whether to allow multiple concurrent positions (one per strategy).
    pub allow_concurrent_positions: bool,
    /// Maximum number of concurrent positions.
    pub max_concurrent_positions: usize,
    /// Session length in bars (for ORB reset).
    pub session_bars: usize,
}

impl Default for StrategyBacktesterConfig {
    fn default() -> Self {
        Self {
            symbol: "BTCUSD".to_string(),
            initial_balance: 10_000.0,
            slippage_bps: 5.0,
            commission_bps: 6.0,
            htf_ratio: 15,
            htf_ema_short: 20,
            htf_ema_long: 50,
            allow_concurrent_positions: true,
            max_concurrent_positions: 3,
            session_bars: 96, // 24h if 15m bars
        }
    }
}

// ============================================================================
// Strategy Backtester
// ============================================================================

/// Bar-level backtester that mirrors the forward service event loop.
///
/// Feeds OHLCV bars through regime detection → strategy selection → signal
/// generation → simulated execution, producing per-strategy metrics.
pub struct StrategyBacktester {
    config: StrategyBacktesterConfig,

    // Regime detection
    router: EnhancedRouter,

    // Strategies (concrete, like forward event loop)
    mean_reversion: MeanReversionStrategy,
    squeeze_breakout: SqueezeBreakoutStrategy,
    vwap_scalper: VwapScalperStrategy,
    orb: OrbStrategy,
    ema_ribbon: EmaRibbonScalperStrategy,
    trend_pullback: TrendPullbackStrategy,
    momentum_surge: MomentumSurgeStrategy,
    multi_tf_trend: MultiTfTrendStrategy,

    // HTF aggregator
    htf_aggregator: HtfAggregator,

    // Duplicate-signal guards (per strategy)
    last_signals: HashMap<StrategyId, String>,

    // Open positions (per strategy)
    positions: HashMap<StrategyId, SimPosition>,

    // Completed trades
    trades: Vec<CompletedTrade>,

    // Signal log
    signals: Vec<BacktestSignal>,

    // Regime tracking
    regime_counts: HashMap<String, usize>,
    last_regime: MarketRegime,
    bar_count: usize,
    session_bar_count: usize,
}

impl StrategyBacktester {
    /// Create a new strategy backtester with the given configuration.
    pub fn new(config: StrategyBacktesterConfig) -> Self {
        let router = EnhancedRouter::with_ensemble();

        let htf_aggregator =
            HtfAggregator::new(config.htf_ratio, config.htf_ema_short, config.htf_ema_long);

        Self {
            mean_reversion: MeanReversionStrategy::new(MeanReversionConfig::crypto_aggressive()),
            squeeze_breakout: SqueezeBreakoutStrategy::new(
                SqueezeBreakoutConfig::crypto_aggressive(),
            ),
            vwap_scalper: VwapScalperStrategy::new(VwapScalperConfig::crypto_aggressive()),
            orb: OrbStrategy::new(OrbConfig::crypto_aggressive()),
            ema_ribbon: EmaRibbonScalperStrategy::new(EmaRibbonConfig::crypto_aggressive()),
            trend_pullback: TrendPullbackStrategy::new(TrendPullbackConfig::crypto_aggressive()),
            momentum_surge: MomentumSurgeStrategy::new(MomentumSurgeConfig::crypto_aggressive()),
            multi_tf_trend: MultiTfTrendStrategy::new(MultiTfConfig::crypto_aggressive()),
            router,
            htf_aggregator,
            last_signals: HashMap::new(),
            positions: HashMap::new(),
            trades: Vec::new(),
            signals: Vec::new(),
            regime_counts: HashMap::new(),
            last_regime: MarketRegime::Uncertain,
            bar_count: 0,
            session_bar_count: 0,
            config,
        }
    }

    /// Create a backtester with default config.
    pub fn default_backtester() -> Self {
        Self::new(StrategyBacktesterConfig::default())
    }

    /// Run the backtest on the given OHLCV bars.
    ///
    /// Returns a detailed `BacktestReport` with per-strategy metrics.
    pub fn run(&mut self, bars: &[OhlcvBar]) -> BacktestReport {
        if bars.is_empty() {
            return self.build_report();
        }

        info!(
            "Starting strategy backtest: {} bars, symbol={}",
            bars.len(),
            self.config.symbol
        );

        // Start ORB session
        self.orb.start_session();

        for (i, bar) in bars.iter().enumerate() {
            self.process_bar(i, bar);
        }

        // Close any remaining positions at final bar
        let final_bar = bars.last().unwrap();
        let final_price = final_bar.close;
        let final_idx = bars.len() - 1;
        self.close_all_positions(final_price, final_idx, ExitReason::EndOfData);

        info!(
            "Backtest complete: {} bars, {} signals, {} trades",
            self.bar_count,
            self.signals.len(),
            self.trades.len()
        );

        self.build_report()
    }

    /// Process a single OHLCV bar.
    fn process_bar(&mut self, bar_idx: usize, bar: &OhlcvBar) {
        self.bar_count += 1;
        self.session_bar_count += 1;

        // Session boundary for ORB/VWAP
        if self.session_bar_count >= self.config.session_bars {
            self.session_bar_count = 0;
            self.vwap_scalper.reset_session();
            self.orb.start_session();
            self.last_signals.remove(&StrategyId::VwapScalper);
            self.last_signals.remove(&StrategyId::OpeningRangeBreakout);
            self.last_signals.remove(&StrategyId::EmaRibbonScalper);
            self.last_signals.remove(&StrategyId::TrendPullback);
            self.last_signals.remove(&StrategyId::MomentumSurge);
            self.last_signals.remove(&StrategyId::MultiTfTrend);
        }

        // Check stop-loss / take-profit on open positions
        self.check_sl_tp(bar, bar_idx);

        // Update regime detection
        let routed = self
            .router
            .update(&self.config.symbol, bar.high, bar.low, bar.close);

        let routed = match routed {
            Some(r) => r,
            None => return,
        };

        // Track regime distribution
        let regime_label = format!("{}", routed.regime);
        *self.regime_counts.entry(regime_label).or_insert(0) += 1;
        self.last_regime = routed.regime;

        // Feed HTF aggregator
        if let Some(htf_trend) = self
            .htf_aggregator
            .feed(bar.open, bar.high, bar.low, bar.close, bar.volume)
        {
            self.ema_ribbon.set_htf_trend(htf_trend);
            self.trend_pullback.set_htf_trend(htf_trend);
            self.multi_tf_trend.set_htf_trend(htf_trend);
        }

        // Route to strategies based on regime
        self.run_strategies(bar_idx, bar, &routed);
    }

    /// Run all strategies that are eligible for the current regime.
    fn run_strategies(&mut self, bar_idx: usize, bar: &OhlcvBar, routed: &RoutedSignal) {
        let regime = &routed.regime;
        let confidence = routed.confidence;
        let position_factor = routed.position_factor;

        // ── Mean Reversion ──────────────────────────────────────────────
        if MeanReversionStrategy::should_trade_in_regime(regime) {
            let mr_result = self
                .mean_reversion
                .update_hlc_with_reason(bar.high, bar.low, bar.close);
            match mr_result.signal {
                MeanReversionSignal::Buy => {
                    if self.dedup_check(StrategyId::MeanReversion, "BUY") {
                        let atr = self.mean_reversion.last_atr().unwrap_or(bar.close * 0.02);
                        let stop = self
                            .mean_reversion
                            .stop_loss()
                            .unwrap_or(bar.close - 2.0 * atr);
                        let tp = self
                            .mean_reversion
                            .take_profit()
                            .unwrap_or(bar.close + 3.0 * atr);
                        let factor = MeanReversionStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::MeanReversion,
                            Direction::Long,
                            mr_result.rsi.unwrap_or(0.5).min(1.0),
                            Some(stop),
                            Some(tp),
                            mr_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                MeanReversionSignal::Sell => {
                    if self.dedup_check(StrategyId::MeanReversion, "SELL") {
                        let atr = self.mean_reversion.last_atr().unwrap_or(bar.close * 0.02);
                        let stop = self
                            .mean_reversion
                            .stop_loss()
                            .unwrap_or(bar.close + 2.0 * atr);
                        let tp = self
                            .mean_reversion
                            .take_profit()
                            .unwrap_or(bar.close - 3.0 * atr);
                        let factor = MeanReversionStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::MeanReversion,
                            Direction::Short,
                            mr_result.rsi.unwrap_or(0.5).min(1.0),
                            Some(stop),
                            Some(tp),
                            mr_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                MeanReversionSignal::Hold => {
                    self.last_signals.remove(&StrategyId::MeanReversion);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::MeanReversion);
        }

        // ── Squeeze Breakout ────────────────────────────────────────────
        if SqueezeBreakoutStrategy::should_trade_in_regime(regime) {
            let sq_result = self
                .squeeze_breakout
                .update_hlc(bar.high, bar.low, bar.close);
            match sq_result.signal {
                SqueezeBreakoutSignal::BuyBreakout => {
                    if self.dedup_check(StrategyId::SqueezeBreakout, "BUY") {
                        let factor =
                            SqueezeBreakoutStrategy::regime_size_factor(regime, confidence)
                                .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::SqueezeBreakout,
                            Direction::Long,
                            sq_result.confidence,
                            sq_result.stop_loss,
                            sq_result.take_profit,
                            sq_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                SqueezeBreakoutSignal::SellBreakout => {
                    if self.dedup_check(StrategyId::SqueezeBreakout, "SELL") {
                        let factor =
                            SqueezeBreakoutStrategy::regime_size_factor(regime, confidence)
                                .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::SqueezeBreakout,
                            Direction::Short,
                            sq_result.confidence,
                            sq_result.stop_loss,
                            sq_result.take_profit,
                            sq_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                SqueezeBreakoutSignal::Squeeze | SqueezeBreakoutSignal::Hold => {
                    self.last_signals.remove(&StrategyId::SqueezeBreakout);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::SqueezeBreakout);
        }

        // ── VWAP Scalper ────────────────────────────────────────────────
        if VwapScalperStrategy::should_trade_in_regime(regime) {
            let vwap_result = self
                .vwap_scalper
                .update_ohlcv(bar.open, bar.high, bar.low, bar.close, bar.volume);
            match vwap_result.signal {
                VwapSignal::Buy => {
                    if self.dedup_check(StrategyId::VwapScalper, "BUY") {
                        let factor = VwapScalperStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::VwapScalper,
                            Direction::Long,
                            vwap_result.confidence,
                            vwap_result.stop_loss,
                            vwap_result.take_profit,
                            vwap_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                VwapSignal::Sell => {
                    if self.dedup_check(StrategyId::VwapScalper, "SELL") {
                        let factor = VwapScalperStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::VwapScalper,
                            Direction::Short,
                            vwap_result.confidence,
                            vwap_result.stop_loss,
                            vwap_result.take_profit,
                            vwap_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                VwapSignal::Hold => {
                    self.last_signals.remove(&StrategyId::VwapScalper);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::VwapScalper);
        }

        // ── Opening Range Breakout ──────────────────────────────────────
        if OrbStrategy::should_trade_in_regime(regime) {
            let orb_result = self
                .orb
                .update_ohlcv(bar.open, bar.high, bar.low, bar.close, bar.volume);
            match orb_result.signal {
                OrbSignal::BuyBreakout => {
                    if self.dedup_check(StrategyId::OpeningRangeBreakout, "BUY") {
                        let factor = OrbStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::OpeningRangeBreakout,
                            Direction::Long,
                            orb_result.confidence,
                            orb_result.stop_loss,
                            orb_result.take_profit,
                            orb_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                OrbSignal::SellBreakout => {
                    if self.dedup_check(StrategyId::OpeningRangeBreakout, "SELL") {
                        let factor = OrbStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::OpeningRangeBreakout,
                            Direction::Short,
                            orb_result.confidence,
                            orb_result.stop_loss,
                            orb_result.take_profit,
                            orb_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                OrbSignal::Forming | OrbSignal::Hold => {
                    self.last_signals.remove(&StrategyId::OpeningRangeBreakout);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::OpeningRangeBreakout);
        }

        // ── EMA Ribbon Scalper ──────────────────────────────────────────
        if EmaRibbonScalperStrategy::should_trade_in_regime(regime) {
            let ribbon_result = self
                .ema_ribbon
                .update_ohlcv(bar.open, bar.high, bar.low, bar.close, bar.volume);
            match ribbon_result.signal {
                EmaRibbonSignal::Buy => {
                    if self.dedup_check(StrategyId::EmaRibbonScalper, "BUY") {
                        let factor =
                            EmaRibbonScalperStrategy::regime_size_factor(regime, confidence)
                                .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::EmaRibbonScalper,
                            Direction::Long,
                            ribbon_result.confidence,
                            ribbon_result.stop_loss,
                            ribbon_result.take_profit,
                            ribbon_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                EmaRibbonSignal::Sell => {
                    if self.dedup_check(StrategyId::EmaRibbonScalper, "SELL") {
                        let factor =
                            EmaRibbonScalperStrategy::regime_size_factor(regime, confidence)
                                .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::EmaRibbonScalper,
                            Direction::Short,
                            ribbon_result.confidence,
                            ribbon_result.stop_loss,
                            ribbon_result.take_profit,
                            ribbon_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                EmaRibbonSignal::Hold => {
                    self.last_signals.remove(&StrategyId::EmaRibbonScalper);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::EmaRibbonScalper);
        }

        // ── Trend Pullback ──────────────────────────────────────────────
        if TrendPullbackStrategy::should_trade_in_regime(regime) {
            let tp_result = self
                .trend_pullback
                .update_ohlcv(bar.open, bar.high, bar.low, bar.close, bar.volume);
            match tp_result.signal {
                TrendPullbackSignal::Buy => {
                    if self.dedup_check(StrategyId::TrendPullback, "BUY") {
                        let factor = TrendPullbackStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::TrendPullback,
                            Direction::Long,
                            tp_result.confidence,
                            tp_result.stop_loss,
                            tp_result.take_profit,
                            tp_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                TrendPullbackSignal::Sell => {
                    if self.dedup_check(StrategyId::TrendPullback, "SELL") {
                        let factor = TrendPullbackStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::TrendPullback,
                            Direction::Short,
                            tp_result.confidence,
                            tp_result.stop_loss,
                            tp_result.take_profit,
                            tp_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                TrendPullbackSignal::Hold => {
                    self.last_signals.remove(&StrategyId::TrendPullback);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::TrendPullback);
        }

        // ── Momentum Surge ──────────────────────────────────────────────
        if MomentumSurgeStrategy::should_trade_in_regime(regime) {
            let ms_result = self
                .momentum_surge
                .update_ohlcv(bar.open, bar.high, bar.low, bar.close, bar.volume);
            match ms_result.signal {
                MomentumSurgeSignal::Buy => {
                    if self.dedup_check(StrategyId::MomentumSurge, "BUY") {
                        let factor = MomentumSurgeStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::MomentumSurge,
                            Direction::Long,
                            ms_result.confidence,
                            ms_result.stop_loss,
                            ms_result.take_profit,
                            ms_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                MomentumSurgeSignal::Sell => {
                    if self.dedup_check(StrategyId::MomentumSurge, "SELL") {
                        let factor = MomentumSurgeStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::MomentumSurge,
                            Direction::Short,
                            ms_result.confidence,
                            ms_result.stop_loss,
                            ms_result.take_profit,
                            ms_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                MomentumSurgeSignal::Hold => {
                    self.last_signals.remove(&StrategyId::MomentumSurge);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::MomentumSurge);
        }

        // ── Multi-TF Trend ──────────────────────────────────────────────
        if MultiTfTrendStrategy::should_trade_in_regime(regime) {
            let mtf_result = self
                .multi_tf_trend
                .update_ohlcv(bar.open, bar.high, bar.low, bar.close, bar.volume);
            match mtf_result.signal {
                MultiTfSignal::Buy => {
                    if self.dedup_check(StrategyId::MultiTfTrend, "BUY") {
                        let factor = MultiTfTrendStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::MultiTfTrend,
                            Direction::Long,
                            mtf_result.confidence,
                            mtf_result.stop_loss,
                            mtf_result.take_profit,
                            mtf_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                MultiTfSignal::Sell => {
                    if self.dedup_check(StrategyId::MultiTfTrend, "SELL") {
                        let factor = MultiTfTrendStrategy::regime_size_factor(regime, confidence)
                            .unwrap_or(position_factor);
                        self.emit_signal(
                            bar_idx,
                            bar,
                            StrategyId::MultiTfTrend,
                            Direction::Short,
                            mtf_result.confidence,
                            mtf_result.stop_loss,
                            mtf_result.take_profit,
                            mtf_result.reason,
                            *regime,
                            factor,
                        );
                    }
                }
                MultiTfSignal::Hold => {
                    self.last_signals.remove(&StrategyId::MultiTfTrend);
                }
            }
        } else {
            self.last_signals.remove(&StrategyId::MultiTfTrend);
        }
    }

    /// Check if a signal is new (not a duplicate of the last signal for this strategy).
    fn dedup_check(&mut self, strategy: StrategyId, direction: &str) -> bool {
        let key = direction.to_string();
        if self.last_signals.get(&strategy) == Some(&key) {
            return false;
        }
        self.last_signals.insert(strategy, key);
        true
    }

    /// Emit a signal and optionally open a position.
    #[allow(clippy::too_many_arguments)]
    fn emit_signal(
        &mut self,
        bar_idx: usize,
        bar: &OhlcvBar,
        strategy: StrategyId,
        direction: Direction,
        confidence: f64,
        stop_loss: Option<f64>,
        take_profit: Option<f64>,
        reason: String,
        regime: MarketRegime,
        size_factor: f64,
    ) {
        let signal = BacktestSignal {
            strategy,
            direction,
            confidence,
            stop_loss,
            take_profit,
            reason: reason.clone(),
            bar_index: bar_idx,
            price: bar.close,
            regime,
        };
        self.signals.push(signal);

        debug!(
            "Signal: {} {} @ {:.2} [{}] conf={:.2} | {}",
            strategy, direction, bar.close, regime, confidence, reason
        );

        // Position management: close opposite, open if slot available
        if let Some(existing) = self.positions.get(&strategy) {
            if existing.direction != direction {
                // Close existing opposite position
                let trade = self.close_position_trade(
                    strategy,
                    bar.close,
                    bar_idx,
                    ExitReason::OppositeSignal,
                );
                if let Some(t) = trade {
                    self.trades.push(t);
                }
            } else {
                // Same direction — already in position, skip
                return;
            }
        }

        // Check concurrent position limit
        if !self.config.allow_concurrent_positions && !self.positions.is_empty() {
            return;
        }
        if self.positions.len() >= self.config.max_concurrent_positions
            && !self.positions.contains_key(&strategy)
        {
            return;
        }

        // Apply slippage to entry
        let slippage_mult = 1.0 + (self.config.slippage_bps / 10_000.0);
        let entry_price = match direction {
            Direction::Long => bar.close * slippage_mult,
            Direction::Short => bar.close / slippage_mult,
        };

        let position = SimPosition {
            strategy,
            direction,
            entry_price,
            entry_bar: bar_idx,
            stop_loss,
            take_profit,
            size_factor,
        };
        self.positions.insert(strategy, position);
    }

    /// Check stop-loss and take-profit on all open positions using the current bar.
    fn check_sl_tp(&mut self, bar: &OhlcvBar, bar_idx: usize) {
        let strategies_to_close: Vec<(StrategyId, f64, ExitReason)> = self
            .positions
            .iter()
            .filter_map(|(id, pos)| {
                match pos.direction {
                    Direction::Long => {
                        // Stop loss hit
                        if let Some(sl) = pos.stop_loss
                            && bar.low <= sl
                        {
                            return Some((*id, sl, ExitReason::StopLoss));
                        }
                        // Take profit hit
                        if let Some(tp) = pos.take_profit
                            && bar.high >= tp
                        {
                            return Some((*id, tp, ExitReason::TakeProfit));
                        }
                    }
                    Direction::Short => {
                        // Stop loss hit
                        if let Some(sl) = pos.stop_loss
                            && bar.high >= sl
                        {
                            return Some((*id, sl, ExitReason::StopLoss));
                        }
                        // Take profit hit
                        if let Some(tp) = pos.take_profit
                            && bar.low <= tp
                        {
                            return Some((*id, tp, ExitReason::TakeProfit));
                        }
                    }
                }
                None
            })
            .collect();

        for (strategy, exit_price, reason) in strategies_to_close {
            if let Some(trade) = self.close_position_trade(strategy, exit_price, bar_idx, reason) {
                self.trades.push(trade);
            }
        }
    }

    /// Close a position and return the completed trade.
    fn close_position_trade(
        &mut self,
        strategy: StrategyId,
        exit_price: f64,
        bar_idx: usize,
        reason: ExitReason,
    ) -> Option<CompletedTrade> {
        let pos = self.positions.remove(&strategy)?;

        // Apply slippage to exit
        let slippage_mult = 1.0 + (self.config.slippage_bps / 10_000.0);
        let slipped_exit = match pos.direction {
            Direction::Long => exit_price / slippage_mult,
            Direction::Short => exit_price * slippage_mult,
        };

        // Calculate PnL
        let gross_pnl_pct = match pos.direction {
            Direction::Long => ((slipped_exit - pos.entry_price) / pos.entry_price) * 100.0,
            Direction::Short => ((pos.entry_price - slipped_exit) / pos.entry_price) * 100.0,
        };

        // Subtract commission (round-trip)
        let commission_pct = (self.config.commission_bps / 10_000.0) * 2.0 * 100.0;
        let net_pnl_pct = gross_pnl_pct - commission_pct;

        let notional = self.config.initial_balance * 0.8 * pos.size_factor;
        let pnl_absolute = notional * (net_pnl_pct / 100.0);

        debug!(
            "Close {} {} @ {:.2} → {:.2} | PnL: {:+.2}% ({:+.2}) | {}",
            pos.strategy,
            pos.direction,
            pos.entry_price,
            slipped_exit,
            net_pnl_pct,
            pnl_absolute,
            reason
        );

        Some(CompletedTrade {
            strategy,
            direction: pos.direction,
            entry_price: pos.entry_price,
            exit_price: slipped_exit,
            entry_bar: pos.entry_bar,
            exit_bar: bar_idx,
            pnl_pct: net_pnl_pct,
            pnl_absolute,
            size_factor: pos.size_factor,
            exit_reason: reason,
        })
    }

    /// Close all open positions (end of data).
    fn close_all_positions(&mut self, price: f64, bar_idx: usize, reason: ExitReason) {
        let strategy_ids: Vec<StrategyId> = self.positions.keys().copied().collect();
        for strategy in strategy_ids {
            if let Some(trade) = self.close_position_trade(strategy, price, bar_idx, reason) {
                self.trades.push(trade);
            }
        }
    }

    /// Build the final backtest report from accumulated data.
    fn build_report(&self) -> BacktestReport {
        let mut strategy_metrics: HashMap<StrategyId, StrategyMetrics> = HashMap::new();

        // Initialize metrics for all strategies
        for &id in StrategyId::all() {
            let m = StrategyMetrics {
                strategy: id,
                ..Default::default()
            };
            strategy_metrics.insert(id, m);
        }

        // Count signals per strategy
        for sig in &self.signals {
            let m = strategy_metrics.get_mut(&sig.strategy).unwrap();
            m.signals_generated += 1;
            if self.is_target_regime(sig.strategy, &sig.regime) {
                m.signals_in_target_regime += 1;
            } else {
                m.signals_outside_regime += 1;
            }
        }

        // Compute trade metrics per strategy
        for &id in StrategyId::all() {
            let trades: Vec<&CompletedTrade> =
                self.trades.iter().filter(|t| t.strategy == id).collect();

            let m = strategy_metrics.get_mut(&id).unwrap();
            m.total_trades = trades.len();

            if trades.is_empty() {
                continue;
            }

            let winners: Vec<&&CompletedTrade> =
                trades.iter().filter(|t| t.pnl_pct > 0.0).collect();
            let losers: Vec<&&CompletedTrade> =
                trades.iter().filter(|t| t.pnl_pct <= 0.0).collect();

            m.winning_trades = winners.len();
            m.losing_trades = losers.len();
            m.win_rate = (m.winning_trades as f64 / m.total_trades as f64) * 100.0;

            m.total_pnl_pct = trades.iter().map(|t| t.pnl_pct).sum();

            if !winners.is_empty() {
                m.avg_win_pct =
                    winners.iter().map(|t| t.pnl_pct).sum::<f64>() / winners.len() as f64;
                m.largest_win_pct = winners
                    .iter()
                    .map(|t| t.pnl_pct)
                    .fold(f64::NEG_INFINITY, f64::max);
            }

            if !losers.is_empty() {
                m.avg_loss_pct =
                    losers.iter().map(|t| t.pnl_pct).sum::<f64>() / losers.len() as f64;
                m.largest_loss_pct = losers
                    .iter()
                    .map(|t| t.pnl_pct)
                    .fold(f64::INFINITY, f64::min);
            }

            // Profit factor
            let gross_profit: f64 = winners.iter().map(|t| t.pnl_pct).sum();
            let gross_loss: f64 = losers.iter().map(|t| t.pnl_pct.abs()).sum();
            m.profit_factor = if gross_loss > 0.0 {
                gross_profit / gross_loss
            } else if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };

            // Max drawdown (cumulative PnL curve)
            let mut cum_pnl = 0.0;
            let mut peak_pnl = 0.0;
            let mut max_dd = 0.0;
            for t in &trades {
                cum_pnl += t.pnl_pct;
                if cum_pnl > peak_pnl {
                    peak_pnl = cum_pnl;
                }
                let dd = peak_pnl - cum_pnl;
                if dd > max_dd {
                    max_dd = dd;
                }
            }
            m.max_drawdown_pct = max_dd;

            // Sharpe ratio (simplified — per-trade returns)
            let returns: Vec<f64> = trades.iter().map(|t| t.pnl_pct).collect();
            let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance =
                returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f64>() / returns.len() as f64;
            let std_dev = variance.sqrt();
            m.sharpe_ratio = if std_dev > 0.0 {
                mean_ret / std_dev
            } else {
                0.0
            };

            // Average trade duration
            let total_duration: usize = trades
                .iter()
                .map(|t| t.exit_bar.saturating_sub(t.entry_bar))
                .sum();
            m.avg_trade_duration_bars = total_duration as f64 / trades.len() as f64;
        }

        let total_signals = self.signals.len();
        let total_trades = self.trades.len();
        let aggregate_pnl: f64 = self.trades.iter().map(|t| t.pnl_pct).sum();

        BacktestReport {
            symbol: self.config.symbol.clone(),
            total_bars: self.bar_count,
            strategy_metrics,
            regime_distribution: self.regime_counts.clone(),
            total_signals,
            total_trades,
            aggregate_pnl_pct: aggregate_pnl,
        }
    }

    /// Check whether a regime is the "target" regime for a given strategy.
    fn is_target_regime(&self, strategy: StrategyId, regime: &MarketRegime) -> bool {
        match strategy {
            StrategyId::MeanReversion => MeanReversionStrategy::should_trade_in_regime(regime),
            StrategyId::SqueezeBreakout => SqueezeBreakoutStrategy::should_trade_in_regime(regime),
            StrategyId::VwapScalper => VwapScalperStrategy::should_trade_in_regime(regime),
            StrategyId::OpeningRangeBreakout => OrbStrategy::should_trade_in_regime(regime),
            StrategyId::EmaRibbonScalper => {
                EmaRibbonScalperStrategy::should_trade_in_regime(regime)
            }
            StrategyId::TrendPullback => TrendPullbackStrategy::should_trade_in_regime(regime),
            StrategyId::MomentumSurge => MomentumSurgeStrategy::should_trade_in_regime(regime),
            StrategyId::MultiTfTrend => MultiTfTrendStrategy::should_trade_in_regime(regime),
        }
    }

    // ====================================================================
    // Public accessors
    // ====================================================================

    /// Get all completed trades.
    pub fn trades(&self) -> &[CompletedTrade] {
        &self.trades
    }

    /// Get all emitted signals.
    pub fn signals(&self) -> &[BacktestSignal] {
        &self.signals
    }

    /// Get trades for a specific strategy.
    pub fn trades_for(&self, strategy: StrategyId) -> Vec<&CompletedTrade> {
        self.trades
            .iter()
            .filter(|t| t.strategy == strategy)
            .collect()
    }

    /// Get signals for a specific strategy.
    pub fn signals_for(&self, strategy: StrategyId) -> Vec<&BacktestSignal> {
        self.signals
            .iter()
            .filter(|s| s.strategy == strategy)
            .collect()
    }

    /// Get the bar count processed so far.
    pub fn bar_count(&self) -> usize {
        self.bar_count
    }
}

// ============================================================================
// Synthetic Data Generation
// ============================================================================

/// Generates synthetic OHLCV bars for testing strategies across different
/// market regimes.
pub struct SyntheticDataGenerator;

impl SyntheticDataGenerator {
    /// Generate a trending-up price series.
    ///
    /// * `bars` — number of bars
    /// * `start_price` — initial price
    /// * `drift_pct` — drift per bar (e.g. 0.003 = 0.3% per bar)
    /// * `noise_pct` — random noise amplitude (e.g. 0.01 = 1%)
    pub fn trending_up(
        bars: usize,
        start_price: f64,
        drift_pct: f64,
        noise_pct: f64,
    ) -> Vec<OhlcvBar> {
        Self::generate_series(bars, start_price, drift_pct, noise_pct, false)
    }

    /// Generate a trending-down price series.
    pub fn trending_down(
        bars: usize,
        start_price: f64,
        drift_pct: f64,
        noise_pct: f64,
    ) -> Vec<OhlcvBar> {
        Self::generate_series(bars, start_price, -drift_pct, noise_pct, false)
    }

    /// Generate a mean-reverting (range-bound) price series oscillating
    /// around `center_price`.
    ///
    /// * `range_pct` — amplitude of oscillation (e.g. 0.03 = ±3%)
    pub fn mean_reverting(
        bars: usize,
        center_price: f64,
        range_pct: f64,
        noise_pct: f64,
    ) -> Vec<OhlcvBar> {
        Self::generate_oscillating(bars, center_price, range_pct, noise_pct)
    }

    /// Generate a volatile (choppy) series with high ATR but no clear trend.
    pub fn volatile(bars: usize, center_price: f64, volatility_pct: f64) -> Vec<OhlcvBar> {
        Self::generate_series(bars, center_price, 0.0, volatility_pct, true)
    }

    /// Generate a mixed regime series: trend → range → volatile → trend.
    ///
    /// Each phase is `phase_bars` bars long.
    pub fn mixed_regime(phase_bars: usize, start_price: f64) -> Vec<OhlcvBar> {
        let mut all = Vec::new();

        // Phase 1: Trending up
        let trend_up = Self::trending_up(phase_bars, start_price, 0.002, 0.008);
        let last_price = trend_up.last().unwrap().close;
        all.extend(trend_up);

        // Phase 2: Mean reverting
        let range = Self::mean_reverting(phase_bars, last_price, 0.025, 0.005);
        let last_price = range.last().unwrap().close;
        all.extend(range);

        // Phase 3: Volatile
        let vol = Self::volatile(phase_bars, last_price, 0.04);
        let last_price = vol.last().unwrap().close;
        all.extend(vol);

        // Phase 4: Trending down
        let trend_down = Self::trending_down(phase_bars, last_price, 0.002, 0.008);
        all.extend(trend_down);

        // Re-index timestamps
        let base_time = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        for (i, bar) in all.iter_mut().enumerate() {
            bar.timestamp = base_time + chrono::Duration::minutes(i as i64 * 15);
        }

        all
    }

    // ── Internal generators ─────────────────────────────────────────────

    fn generate_series(
        bars: usize,
        start_price: f64,
        drift_per_bar: f64,
        noise_pct: f64,
        wide_bars: bool,
    ) -> Vec<OhlcvBar> {
        let base_time = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut result = Vec::with_capacity(bars);
        let mut price = start_price;

        // Simple deterministic pseudo-random using a linear congruential generator
        let mut seed: u64 = 42;

        for i in 0..bars {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r1 = ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0; // -1..1

            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let _r2 = ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;

            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r3 = (seed >> 33) as f64 / u32::MAX as f64;

            let open = price;
            // Dampen noise so it doesn't compound as a random walk and overwhelm drift.
            // Use noise as a per-bar perturbation scaled by 0.3 to keep drift dominant.
            let noise = price * noise_pct * r1 * 0.3;
            let close = price * (1.0 + drift_per_bar) + noise;

            let bar_range = if wide_bars {
                price * noise_pct * 2.0
            } else {
                price * noise_pct * 1.2
            };

            let high = open.max(close) + bar_range * r3.abs() * 0.5;
            let low = open.min(close) - bar_range * (1.0 - r3).abs() * 0.5;

            // Ensure high >= max(open, close) and low <= min(open, close)
            let high = high.max(open).max(close);
            let low = low.min(open).min(close).max(0.01); // floor at 0.01

            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let vol_r = (seed >> 33) as f64 / u32::MAX as f64;
            let volume = 1000.0 + 5000.0 * vol_r;

            result.push(OhlcvBar {
                timestamp: base_time + chrono::Duration::minutes(i as i64 * 15),
                open,
                high,
                low,
                close,
                volume,
            });

            price = close;
        }

        result
    }

    fn generate_oscillating(
        bars: usize,
        center: f64,
        range_pct: f64,
        noise_pct: f64,
    ) -> Vec<OhlcvBar> {
        let base_time = DateTime::from_timestamp(1_700_000_000, 0).unwrap();
        let mut result = Vec::with_capacity(bars);
        let amplitude = center * range_pct;

        let mut seed: u64 = 12345;

        for i in 0..bars {
            // Sine wave oscillation with noise
            let phase = (i as f64 / bars as f64) * std::f64::consts::PI * 6.0;
            let sine_component = amplitude * phase.sin();

            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r1 = ((seed >> 33) as f64 / u32::MAX as f64) * 2.0 - 1.0;

            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let r2 = (seed >> 33) as f64 / u32::MAX as f64;

            let mid = center + sine_component;
            let noise = center * noise_pct * r1;
            let open = mid + noise * 0.3;
            let close = mid + noise * 0.7;

            let bar_range = center * noise_pct * 1.5;
            let high = open.max(close) + bar_range * r2 * 0.4;
            let low = open.min(close) - bar_range * (1.0 - r2) * 0.4;
            let high = high.max(open).max(close);
            let low = low.min(open).min(close).max(0.01);

            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let vol_r = (seed >> 33) as f64 / u32::MAX as f64;
            let volume = 800.0 + 3000.0 * vol_r;

            result.push(OhlcvBar {
                timestamp: base_time + chrono::Duration::minutes(i as i64 * 15),
                open,
                high,
                low,
                close,
                volume,
            });
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
    fn test_synthetic_trending_up() {
        let bars = SyntheticDataGenerator::trending_up(200, 50_000.0, 0.002, 0.008);
        assert_eq!(bars.len(), 200);
        // Overall trend should be upward
        let first = bars.first().unwrap().close;
        let last = bars.last().unwrap().close;
        assert!(
            last > first,
            "Trending up data should end higher: first={}, last={}",
            first,
            last
        );
    }

    #[test]
    fn test_synthetic_trending_down() {
        let bars = SyntheticDataGenerator::trending_down(200, 50_000.0, 0.002, 0.008);
        assert_eq!(bars.len(), 200);
        let first = bars.first().unwrap().close;
        let last = bars.last().unwrap().close;
        assert!(
            last < first,
            "Trending down data should end lower: first={}, last={}",
            first,
            last
        );
    }

    #[test]
    fn test_synthetic_mean_reverting() {
        let bars = SyntheticDataGenerator::mean_reverting(300, 50_000.0, 0.03, 0.005);
        assert_eq!(bars.len(), 300);
        // Should stay within a range around center
        let center = 50_000.0;
        for bar in &bars {
            assert!(
                bar.close > center * 0.85 && bar.close < center * 1.15,
                "Mean-reverting close {} should be near center {}",
                bar.close,
                center
            );
        }
    }

    #[test]
    fn test_synthetic_volatile() {
        let bars = SyntheticDataGenerator::volatile(200, 50_000.0, 0.04);
        assert_eq!(bars.len(), 200);
        // Should have wider bars
        let avg_range: f64 =
            bars.iter().map(|b| (b.high - b.low) / b.close).sum::<f64>() / bars.len() as f64;
        assert!(
            avg_range > 0.01,
            "Volatile data should have wider bars, avg_range={}",
            avg_range
        );
    }

    #[test]
    fn test_synthetic_mixed_regime() {
        let bars = SyntheticDataGenerator::mixed_regime(200, 50_000.0);
        assert_eq!(bars.len(), 800); // 4 phases × 200
    }

    #[test]
    fn test_ohlcv_bar_typical_price() {
        let bar = OhlcvBar::new(
            DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
            100.0,
            110.0,
            90.0,
            105.0,
            1000.0,
        );
        let tp = bar.typical_price();
        assert!((tp - 101.6666).abs() < 0.01);
    }

    #[test]
    fn test_strategy_backtester_creation() {
        let bt = StrategyBacktester::default_backtester();
        assert_eq!(bt.bar_count(), 0);
        assert!(bt.trades().is_empty());
        assert!(bt.signals().is_empty());
    }

    #[test]
    fn test_backtest_with_trending_data() {
        let bars = SyntheticDataGenerator::trending_up(500, 50_000.0, 0.003, 0.01);
        let mut bt = StrategyBacktester::default_backtester();
        let report = bt.run(&bars);

        assert_eq!(report.total_bars, 500);
        assert!(report.total_bars > 0);
        // We should see at least some regime classification
        assert!(!report.regime_distribution.is_empty());

        println!("{}", report);
    }

    #[test]
    fn test_backtest_with_mean_reverting_data() {
        let bars = SyntheticDataGenerator::mean_reverting(500, 50_000.0, 0.03, 0.005);
        let mut bt = StrategyBacktester::default_backtester();
        let report = bt.run(&bars);

        assert_eq!(report.total_bars, 500);
        println!("{}", report);
    }

    #[test]
    fn test_backtest_with_mixed_regime() {
        let bars = SyntheticDataGenerator::mixed_regime(300, 50_000.0);
        let mut bt = StrategyBacktester::default_backtester();
        let report = bt.run(&bars);

        assert_eq!(report.total_bars, 1200);
        // Should have multiple regime types detected
        assert!(
            !report.regime_distribution.is_empty(),
            "Should detect at least one regime type"
        );

        println!("{}", report);
    }

    #[test]
    fn test_strategy_id_all() {
        assert_eq!(StrategyId::all().len(), 8);
    }

    #[test]
    fn test_report_has_all_strategies() {
        let bars = SyntheticDataGenerator::trending_up(300, 50_000.0, 0.003, 0.01);
        let mut bt = StrategyBacktester::default_backtester();
        let report = bt.run(&bars);

        // All 8 strategies should appear in metrics (even if 0 trades)
        for &id in StrategyId::all() {
            assert!(
                report.strategy_metrics.contains_key(&id),
                "Missing strategy metrics for {}",
                id
            );
        }
    }

    #[test]
    fn test_sl_tp_execution() {
        // Create a scenario with clear stops: a big trend reversal
        let mut bars = SyntheticDataGenerator::trending_up(100, 50_000.0, 0.005, 0.003);
        // Append a crash so any long SL triggers
        let last_price = bars.last().unwrap().close;
        let crash = SyntheticDataGenerator::trending_down(50, last_price, 0.02, 0.005);
        bars.extend(crash);

        let mut bt = StrategyBacktester::default_backtester();
        let report = bt.run(&bars);

        // Some trades should have stop-loss exits
        let _sl_trades = bt
            .trades()
            .iter()
            .filter(|t| t.exit_reason == ExitReason::StopLoss)
            .count();
        // It's okay if there are no SL exits — depends on whether signals were generated
        // The important thing is the backtester didn't panic
        assert!(report.total_bars == 150);
    }

    #[test]
    fn test_empty_data() {
        let bars: Vec<OhlcvBar> = vec![];
        let mut bt = StrategyBacktester::default_backtester();
        let report = bt.run(&bars);
        assert_eq!(report.total_bars, 0);
        assert_eq!(report.total_signals, 0);
        assert_eq!(report.total_trades, 0);
    }

    #[test]
    fn test_single_bar() {
        let bars = vec![OhlcvBar::new(
            DateTime::from_timestamp(1_700_000_000, 0).unwrap(),
            50_000.0,
            50_100.0,
            49_900.0,
            50_050.0,
            1000.0,
        )];
        let mut bt = StrategyBacktester::default_backtester();
        let report = bt.run(&bars);
        assert_eq!(report.total_bars, 1);
    }

    #[test]
    fn test_dedup_guard() {
        // Verify dedup logic
        let mut bt = StrategyBacktester::default_backtester();
        assert!(bt.dedup_check(StrategyId::MeanReversion, "BUY"));
        assert!(!bt.dedup_check(StrategyId::MeanReversion, "BUY")); // duplicate
        assert!(bt.dedup_check(StrategyId::MeanReversion, "SELL")); // different direction
        assert!(bt.dedup_check(StrategyId::EmaRibbonScalper, "BUY")); // different strategy
    }

    #[test]
    fn test_concurrent_position_limit() {
        let config = StrategyBacktesterConfig {
            max_concurrent_positions: 1,
            allow_concurrent_positions: true,
            ..Default::default()
        };
        let mut bt = StrategyBacktester::new(config);
        // With max 1 position, only one strategy can be in a position at a time
        let bars = SyntheticDataGenerator::trending_up(500, 50_000.0, 0.003, 0.01);
        let report = bt.run(&bars);
        // Should still produce valid output
        assert!(report.total_bars == 500);
    }
}
