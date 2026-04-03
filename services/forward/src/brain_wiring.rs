//! # Brain Wiring — Phase 5 Integration Layer
//!
//! This module wires the brain-inspired subsystems together into a unified
//! `TradingPipeline` that processes every tick through the full
//! regime → hypothalamus → amygdala → gating → correlation → execution chain.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────┐    ┌───────────────┐    ┌──────────────┐    ┌──────────────┐
//! │  Regime  │───▶│  Regime       │───▶│ Hypothalamus │───▶│   Amygdala   │
//! │ Detector │    │  Bridge       │    │ (pos scale)  │    │ (threat det) │
//! └──────────┘    └───────────────┘    └──────────────┘    └──────┬───────┘
//!                                                                 │
//!                 ┌───────────────┐    ┌──────────────┐          │
//!                 │  Correlation  │◀───│   Strategy   │◀─────────┘
//!                 │   Tracker     │    │    Gate      │
//!                 └───────┬───────┘    └──────────────┘
//!                         │
//!                   ┌─────▼─────┐
//!                   │ Execution │
//!                   │ Decision  │
//!                   └───────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::brain_wiring::{TradingPipeline, TradingPipelineConfig};
//!
//! let pipeline = TradingPipeline::new(TradingPipelineConfig::default());
//!
//! // On each tick:
//! let decision = pipeline.evaluate(
//!     "BTCUSDT",
//!     &regime_signal,
//!     "ema_flip",
//!     &current_positions,
//!     Some(adx), Some(bb_width), Some(atr), Some(rel_vol),
//! );
//!
//! match decision.action {
//!     TradeAction::Proceed { scale } => { /* execute at `scale` */ },
//!     TradeAction::Block { reason } => { /* log and skip */ },
//!     TradeAction::ReduceOnly { scale, reason } => { /* only close */ },
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use janus_regime::{DetectionMethod, MarketRegime, RoutedSignal};
use janus_risk::{CorrelationConfig, CorrelationTracker};
use janus_strategies::affinity::StrategyAffinityTracker;
use janus_strategies::gating::{StrategyGate, StrategyGatingConfig};

use crate::regime_bridge::{AmygdalaRegime, BridgedRegimeState, bridge_regime_signal};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the full trading pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingPipelineConfig {
    /// Enable/disable the hypothalamus position scaling.
    #[serde(default = "default_true")]
    pub enable_hypothalamus_scaling: bool,

    /// Enable/disable the amygdala threat filter.
    #[serde(default = "default_true")]
    pub enable_amygdala_filter: bool,

    /// Enable/disable the strategy affinity gating.
    #[serde(default = "default_true")]
    pub enable_gating: bool,

    /// Enable/disable correlation-based position limits.
    #[serde(default = "default_true")]
    pub enable_correlation_filter: bool,

    /// Maximum position scale (cap for hypothalamus output).
    #[serde(default = "default_max_scale")]
    pub max_position_scale: f64,

    /// Minimum position scale below which we skip trading entirely.
    #[serde(default = "default_min_scale")]
    pub min_position_scale: f64,

    /// Scale multiplier applied during high-risk (amygdala) states.
    #[serde(default = "default_high_risk_scale")]
    pub high_risk_scale_factor: f64,

    /// Whether to allow new positions during crisis regime.
    #[serde(default)]
    pub allow_new_positions_in_crisis: bool,

    /// Minimum confidence from regime detector to act on.
    #[serde(default = "default_min_confidence")]
    pub min_regime_confidence: f64,

    /// Correlation tracker configuration.
    #[serde(default)]
    pub correlation: CorrelationConfig,

    /// Strategy gating configuration.
    #[serde(default)]
    pub gating: StrategyGatingConfig,

    /// Minimum trades for affinity confidence.
    #[serde(default = "default_min_trades")]
    pub affinity_min_trades: usize,
}

fn default_true() -> bool {
    true
}
fn default_max_scale() -> f64 {
    2.0
}
fn default_min_scale() -> f64 {
    0.05
}
fn default_high_risk_scale() -> f64 {
    0.5
}
fn default_min_confidence() -> f64 {
    0.0
}
fn default_min_trades() -> usize {
    10
}

impl Default for TradingPipelineConfig {
    fn default() -> Self {
        Self {
            enable_hypothalamus_scaling: true,
            enable_amygdala_filter: true,
            enable_gating: true,
            enable_correlation_filter: true,
            max_position_scale: default_max_scale(),
            min_position_scale: default_min_scale(),
            high_risk_scale_factor: default_high_risk_scale(),
            allow_new_positions_in_crisis: false,
            min_regime_confidence: default_min_confidence(),
            correlation: CorrelationConfig::default(),
            gating: StrategyGatingConfig::default(),
            affinity_min_trades: default_min_trades(),
        }
    }
}

// ============================================================================
// Trade Action / Decision
// ============================================================================

/// The final output action from the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub enum TradeAction {
    /// The trade is allowed to proceed at the given scale factor.
    Proceed {
        /// Position scale factor (1.0 = normal, <1.0 = reduced, >1.0 = increased).
        scale: f64,
    },

    /// The trade is blocked entirely.
    Block {
        /// Human-readable reason for the block.
        reason: String,
        /// Which stage blocked it.
        stage: PipelineStage,
    },

    /// Only reduce/close existing positions (no new entries).
    ReduceOnly {
        /// Scale factor for the reduction.
        scale: f64,
        /// Reason for reduce-only mode.
        reason: String,
    },
}

impl TradeAction {
    /// Returns `true` if the action allows trading (Proceed or ReduceOnly).
    pub fn is_actionable(&self) -> bool {
        !matches!(self, TradeAction::Block { .. })
    }

    /// Returns the scale factor, or 0.0 if blocked.
    pub fn scale(&self) -> f64 {
        match self {
            TradeAction::Proceed { scale } => *scale,
            TradeAction::ReduceOnly { scale, .. } => *scale,
            TradeAction::Block { .. } => 0.0,
        }
    }
}

impl fmt::Display for TradeAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TradeAction::Proceed { scale } => write!(f, "PROCEED(scale={scale:.3})"),
            TradeAction::Block { reason, stage } => write!(f, "BLOCK@{stage}({reason})"),
            TradeAction::ReduceOnly { scale, reason } => {
                write!(f, "REDUCE_ONLY(scale={scale:.3}, {reason})")
            }
        }
    }
}

/// Which stage of the pipeline produced a decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineStage {
    RegimeBridge,
    Hypothalamus,
    Amygdala,
    StrategyGate,
    CorrelationFilter,
    KillSwitch,
}

impl fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PipelineStage::RegimeBridge => write!(f, "RegimeBridge"),
            PipelineStage::Hypothalamus => write!(f, "Hypothalamus"),
            PipelineStage::Amygdala => write!(f, "Amygdala"),
            PipelineStage::StrategyGate => write!(f, "StrategyGate"),
            PipelineStage::CorrelationFilter => write!(f, "CorrelationFilter"),
            PipelineStage::KillSwitch => write!(f, "KillSwitch"),
        }
    }
}

/// Full pipeline evaluation result with diagnostics.
#[derive(Debug, Clone)]
pub struct TradingDecision {
    /// The symbol evaluated.
    pub symbol: String,

    /// The strategy evaluated.
    pub strategy: String,

    /// The final trade action.
    pub action: TradeAction,

    /// Bridged regime state from the regime bridge.
    pub bridged_regime: BridgedRegimeState,

    /// Position scale from hypothalamus (before amygdala/gating adjustments).
    pub raw_hypothalamus_scale: f64,

    /// Whether the amygdala flagged this as high-risk.
    pub amygdala_high_risk: bool,

    /// Whether the strategy gate approved this strategy.
    pub gate_approved: bool,

    /// Whether correlation filter passed.
    pub correlation_passed: bool,

    /// Timestamp of the decision.
    pub timestamp: chrono::DateTime<Utc>,

    /// Duration of the pipeline evaluation in microseconds.
    pub evaluation_us: u64,
}

impl TradingDecision {
    /// Whether the decision allows trading.
    pub fn is_actionable(&self) -> bool {
        self.action.is_actionable()
    }

    /// Convenience: get the final scale factor.
    pub fn scale(&self) -> f64 {
        self.action.scale()
    }
}

impl fmt::Display for TradingDecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Decision[{}/{}: {} | regime={} | hypo_scale={:.3} | risk={} | gate={} | corr={}]",
            self.symbol,
            self.strategy,
            self.action,
            self.bridged_regime.hypothalamus_regime,
            self.raw_hypothalamus_scale,
            if self.amygdala_high_risk {
                "HIGH"
            } else {
                "low"
            },
            if self.gate_approved { "pass" } else { "fail" },
            if self.correlation_passed {
                "pass"
            } else {
                "fail"
            },
        )
    }
}

// ============================================================================
// Pipeline Metrics
// ============================================================================

/// Counters and gauges for pipeline health monitoring.
#[derive(Debug, Default)]
pub struct PipelineMetrics {
    /// Total evaluations.
    pub total_evaluations: u64,
    /// Evaluations that resulted in Proceed.
    pub proceed_count: u64,
    /// Evaluations that resulted in Block.
    pub block_count: u64,
    /// Evaluations that resulted in ReduceOnly.
    pub reduce_only_count: u64,
    /// Blocks by stage.
    pub blocks_by_stage: HashMap<PipelineStage, u64>,
    /// Sum of evaluation durations (for computing average).
    pub total_evaluation_us: u64,
    /// Last evaluation result.
    pub last_action: Option<TradeAction>,
}

impl PipelineMetrics {
    /// Average evaluation time in microseconds.
    pub fn avg_evaluation_us(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            self.total_evaluation_us as f64 / self.total_evaluations as f64
        }
    }

    /// Block rate as a percentage.
    pub fn block_rate_pct(&self) -> f64 {
        if self.total_evaluations == 0 {
            0.0
        } else {
            (self.block_count as f64 / self.total_evaluations as f64) * 100.0
        }
    }

    fn record(&mut self, decision: &TradingDecision) {
        self.total_evaluations += 1;
        self.total_evaluation_us += decision.evaluation_us;
        self.last_action = Some(decision.action.clone());
        match &decision.action {
            TradeAction::Proceed { .. } => self.proceed_count += 1,
            TradeAction::Block { stage, .. } => {
                self.block_count += 1;
                *self.blocks_by_stage.entry(*stage).or_insert(0) += 1;
            }
            TradeAction::ReduceOnly { .. } => self.reduce_only_count += 1,
        }
    }
}

// ============================================================================
// Trading Pipeline
// ============================================================================

/// The main trading pipeline. Thread-safe and designed for concurrent access.
///
/// Wraps the regime bridge, hypothalamus scaling, amygdala threat filter,
/// strategy gating, and correlation tracker into a single evaluation path.
pub struct TradingPipeline {
    config: TradingPipelineConfig,
    correlation_tracker: RwLock<CorrelationTracker>,
    strategy_gate: RwLock<StrategyGate>,
    metrics: RwLock<PipelineMetrics>,
    killed: RwLock<bool>,
}

impl TradingPipeline {
    /// Create a new trading pipeline with the given configuration.
    pub fn new(config: TradingPipelineConfig) -> Self {
        let correlation_tracker = CorrelationTracker::new(config.correlation.clone());
        let affinity_tracker = StrategyAffinityTracker::new(config.affinity_min_trades);
        let strategy_gate = StrategyGate::new(config.gating.clone(), affinity_tracker);

        Self {
            config,
            correlation_tracker: RwLock::new(correlation_tracker),
            strategy_gate: RwLock::new(strategy_gate),
            metrics: RwLock::new(PipelineMetrics::default()),
            killed: RwLock::new(false),
        }
    }

    /// Create a pipeline with externally-constructed components.
    pub fn with_components(
        config: TradingPipelineConfig,
        correlation_tracker: CorrelationTracker,
        strategy_gate: StrategyGate,
    ) -> Self {
        Self {
            config,
            correlation_tracker: RwLock::new(correlation_tracker),
            strategy_gate: RwLock::new(strategy_gate),
            metrics: RwLock::new(PipelineMetrics::default()),
            killed: RwLock::new(false),
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Core evaluation
    // ────────────────────────────────────────────────────────────────────

    /// Evaluate whether a strategy should trade on a symbol given the current
    /// regime signal and open positions.
    ///
    /// This is the primary entry point — called on every tick/signal.
    #[allow(clippy::too_many_arguments)]
    pub async fn evaluate(
        &self,
        symbol: &str,
        signal: &RoutedSignal,
        strategy_name: &str,
        current_positions: &[String],
        adx_value: Option<f64>,
        bb_width_percentile: Option<f64>,
        atr_value: Option<f64>,
        relative_volume: Option<f64>,
    ) -> TradingDecision {
        let start = std::time::Instant::now();

        // ── Stage 0: Kill switch ───────────────────────────────────────
        if *self.killed.read().await {
            let bridged = bridge_regime_signal(
                symbol,
                signal,
                adx_value,
                bb_width_percentile,
                atr_value,
                relative_volume,
            );
            let decision = TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: "Kill switch is active".to_string(),
                    stage: PipelineStage::KillSwitch,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale: 0.0,
                amygdala_high_risk: true,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
            self.metrics.write().await.record(&decision);
            return decision;
        }

        // ── Stage 1: Regime Bridge ─────────────────────────────────────
        let bridged = bridge_regime_signal(
            symbol,
            signal,
            adx_value,
            bb_width_percentile,
            atr_value,
            relative_volume,
        );

        // Check minimum confidence
        if bridged.confidence < self.config.min_regime_confidence {
            let decision = TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Regime confidence {:.1}% below minimum {:.1}%",
                        bridged.confidence * 100.0,
                        self.config.min_regime_confidence * 100.0,
                    ),
                    stage: PipelineStage::RegimeBridge,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale: 0.0,
                amygdala_high_risk: false,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
            self.metrics.write().await.record(&decision);
            return decision;
        }

        debug!(
            symbol,
            regime = %bridged.hypothalamus_regime,
            confidence = bridged.confidence,
            "Regime bridge complete"
        );

        // ── Stage 2: Hypothalamus (position scaling) ───────────────────
        let mut scale = if self.config.enable_hypothalamus_scaling {
            bridged.position_scale
        } else {
            1.0
        };

        let raw_hypothalamus_scale = scale;

        // Clamp to configured bounds
        scale = scale.clamp(
            self.config.min_position_scale,
            self.config.max_position_scale,
        );

        // If scale is too small, don't bother trading
        if scale < self.config.min_position_scale {
            let decision = TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Hypothalamus scale {scale:.3} below minimum {:.3}",
                        self.config.min_position_scale,
                    ),
                    stage: PipelineStage::Hypothalamus,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale,
                amygdala_high_risk: false,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
            self.metrics.write().await.record(&decision);
            return decision;
        }

        // ── Stage 3: Amygdala (threat detection) ───────────────────────
        let amygdala_high_risk = bridged.is_high_risk;

        if self.config.enable_amygdala_filter && amygdala_high_risk {
            // Apply high-risk scaling
            scale *= self.config.high_risk_scale_factor;

            // Crisis regime: block new positions entirely if configured
            if bridged.amygdala_regime == AmygdalaRegime::Crisis
                && !self.config.allow_new_positions_in_crisis
            {
                let decision = TradingDecision {
                    symbol: symbol.to_string(),
                    strategy: strategy_name.to_string(),
                    action: TradeAction::ReduceOnly {
                        scale,
                        reason: format!(
                            "Crisis regime detected (amygdala={})",
                            bridged.amygdala_regime,
                        ),
                    },
                    bridged_regime: bridged,
                    raw_hypothalamus_scale,
                    amygdala_high_risk: true,
                    gate_approved: false,
                    correlation_passed: false,
                    timestamp: Utc::now(),
                    evaluation_us: start.elapsed().as_micros() as u64,
                };
                self.metrics.write().await.record(&decision);
                return decision;
            }

            debug!(
                symbol,
                amygdala = %bridged.amygdala_regime,
                new_scale = scale,
                "Amygdala applied high-risk scaling"
            );
        }

        // ── Stage 4: Strategy Gating ───────────────────────────────────
        let gate_approved = if self.config.enable_gating {
            let gate = self.strategy_gate.read().await;
            gate.should_run(strategy_name, symbol, &signal.regime)
        } else {
            true
        };

        if !gate_approved {
            let decision = TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Strategy '{}' not approved for '{}' in regime {}",
                        strategy_name, symbol, signal.regime,
                    ),
                    stage: PipelineStage::StrategyGate,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale,
                amygdala_high_risk,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
            self.metrics.write().await.record(&decision);
            return decision;
        }

        // ── Stage 5: Correlation Filter ────────────────────────────────
        let correlation_passed = if self.config.enable_correlation_filter {
            let tracker = self.correlation_tracker.read().await;
            !tracker.would_exceed_correlation_limit(symbol, current_positions)
        } else {
            true
        };

        if !correlation_passed {
            let decision = TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Correlation limit exceeded: too many correlated positions with '{}'",
                        symbol,
                    ),
                    stage: PipelineStage::CorrelationFilter,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale,
                amygdala_high_risk,
                gate_approved: true,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
            self.metrics.write().await.record(&decision);
            return decision;
        }

        // ── All stages passed — proceed ────────────────────────────────
        let decision = TradingDecision {
            symbol: symbol.to_string(),
            strategy: strategy_name.to_string(),
            action: TradeAction::Proceed { scale },
            bridged_regime: bridged,
            raw_hypothalamus_scale,
            amygdala_high_risk,
            gate_approved: true,
            correlation_passed: true,
            timestamp: Utc::now(),
            evaluation_us: start.elapsed().as_micros() as u64,
        };

        debug!(
            symbol,
            strategy = strategy_name,
            scale,
            "Pipeline: all stages passed"
        );

        self.metrics.write().await.record(&decision);
        decision
    }

    // ────────────────────────────────────────────────────────────────────
    // Synchronous evaluation (for contexts without async runtime)
    // ────────────────────────────────────────────────────────────────────

    /// Evaluate the pipeline synchronously using the provided mutable refs.
    ///
    /// This is useful for testing or single-threaded contexts where you
    /// already hold the locks. It avoids the async overhead.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_sync(
        config: &TradingPipelineConfig,
        symbol: &str,
        signal: &RoutedSignal,
        strategy_name: &str,
        current_positions: &[String],
        correlation_tracker: &CorrelationTracker,
        strategy_gate: &StrategyGate,
        killed: bool,
        adx_value: Option<f64>,
        bb_width_percentile: Option<f64>,
        atr_value: Option<f64>,
        relative_volume: Option<f64>,
    ) -> TradingDecision {
        let start = std::time::Instant::now();

        // Kill switch
        if killed {
            let bridged = bridge_regime_signal(
                symbol,
                signal,
                adx_value,
                bb_width_percentile,
                atr_value,
                relative_volume,
            );
            return TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: "Kill switch is active".to_string(),
                    stage: PipelineStage::KillSwitch,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale: 0.0,
                amygdala_high_risk: true,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
        }

        // Regime bridge
        let bridged = bridge_regime_signal(
            symbol,
            signal,
            adx_value,
            bb_width_percentile,
            atr_value,
            relative_volume,
        );

        if bridged.confidence < config.min_regime_confidence {
            return TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Regime confidence {:.1}% below minimum {:.1}%",
                        bridged.confidence * 100.0,
                        config.min_regime_confidence * 100.0,
                    ),
                    stage: PipelineStage::RegimeBridge,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale: 0.0,
                amygdala_high_risk: false,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
        }

        // Hypothalamus
        let mut scale = if config.enable_hypothalamus_scaling {
            bridged.position_scale
        } else {
            1.0
        };
        let raw_hypothalamus_scale = scale;
        scale = scale.clamp(config.min_position_scale, config.max_position_scale);

        if scale < config.min_position_scale {
            return TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Hypothalamus scale {scale:.3} below minimum {:.3}",
                        config.min_position_scale,
                    ),
                    stage: PipelineStage::Hypothalamus,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale,
                amygdala_high_risk: false,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
        }

        // Amygdala
        let amygdala_high_risk = bridged.is_high_risk;
        if config.enable_amygdala_filter && amygdala_high_risk {
            scale *= config.high_risk_scale_factor;
            if bridged.amygdala_regime == AmygdalaRegime::Crisis
                && !config.allow_new_positions_in_crisis
            {
                return TradingDecision {
                    symbol: symbol.to_string(),
                    strategy: strategy_name.to_string(),
                    action: TradeAction::ReduceOnly {
                        scale,
                        reason: format!(
                            "Crisis regime detected (amygdala={})",
                            bridged.amygdala_regime,
                        ),
                    },
                    bridged_regime: bridged,
                    raw_hypothalamus_scale,
                    amygdala_high_risk: true,
                    gate_approved: false,
                    correlation_passed: false,
                    timestamp: Utc::now(),
                    evaluation_us: start.elapsed().as_micros() as u64,
                };
            }
        }

        // Strategy gate
        let gate_approved = if config.enable_gating {
            strategy_gate.should_run(strategy_name, symbol, &signal.regime)
        } else {
            true
        };

        if !gate_approved {
            return TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Strategy '{}' not approved for '{}' in regime {}",
                        strategy_name, symbol, signal.regime,
                    ),
                    stage: PipelineStage::StrategyGate,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale,
                amygdala_high_risk,
                gate_approved: false,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
        }

        // Correlation
        let correlation_passed = if config.enable_correlation_filter {
            !correlation_tracker.would_exceed_correlation_limit(symbol, current_positions)
        } else {
            true
        };

        if !correlation_passed {
            return TradingDecision {
                symbol: symbol.to_string(),
                strategy: strategy_name.to_string(),
                action: TradeAction::Block {
                    reason: format!(
                        "Correlation limit exceeded: too many correlated positions with '{}'",
                        symbol,
                    ),
                    stage: PipelineStage::CorrelationFilter,
                },
                bridged_regime: bridged,
                raw_hypothalamus_scale,
                amygdala_high_risk,
                gate_approved: true,
                correlation_passed: false,
                timestamp: Utc::now(),
                evaluation_us: start.elapsed().as_micros() as u64,
            };
        }

        // All clear
        TradingDecision {
            symbol: symbol.to_string(),
            strategy: strategy_name.to_string(),
            action: TradeAction::Proceed { scale },
            bridged_regime: bridged,
            raw_hypothalamus_scale,
            amygdala_high_risk,
            gate_approved: true,
            correlation_passed: true,
            timestamp: Utc::now(),
            evaluation_us: start.elapsed().as_micros() as u64,
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Price feed (correlation tracker)
    // ────────────────────────────────────────────────────────────────────

    /// Feed a price update to the correlation tracker.
    pub async fn update_price(&self, asset: &str, price: f64) {
        self.correlation_tracker.write().await.update(asset, price);
    }

    /// Feed a batch of price updates to the correlation tracker.
    pub async fn update_prices_batch(&self, prices: &[(&str, f64)]) {
        self.correlation_tracker.write().await.update_batch(prices);
    }

    // ────────────────────────────────────────────────────────────────────
    // Trade result recording (affinity)
    // ────────────────────────────────────────────────────────────────────

    /// Record a completed trade result for affinity tracking.
    pub async fn record_trade_result(
        &self,
        strategy: &str,
        asset: &str,
        pnl: f64,
        is_winner: bool,
    ) {
        self.strategy_gate
            .write()
            .await
            .tracker_mut()
            .record_trade_result(strategy, asset, pnl, is_winner);
    }

    /// Record a completed trade result with risk-reward ratio.
    pub async fn record_trade_result_with_rr(
        &self,
        strategy: &str,
        asset: &str,
        pnl: f64,
        is_winner: bool,
        rr_ratio: Option<f64>,
    ) {
        self.strategy_gate
            .write()
            .await
            .tracker_mut()
            .record_trade_result_with_rr(strategy, asset, pnl, is_winner, rr_ratio);
    }

    // ────────────────────────────────────────────────────────────────────
    // Kill switch
    // ────────────────────────────────────────────────────────────────────

    /// Activate the kill switch — all subsequent evaluations will be blocked.
    pub async fn activate_kill_switch(&self) {
        let mut killed = self.killed.write().await;
        if !*killed {
            warn!("🛑 Kill switch activated — all trading halted");
            *killed = true;
        }
    }

    /// Deactivate the kill switch — resume normal evaluation.
    pub async fn deactivate_kill_switch(&self) {
        let mut killed = self.killed.write().await;
        if *killed {
            info!("✅ Kill switch deactivated — trading resumed");
            *killed = false;
        }
    }

    /// Check if the kill switch is active.
    pub async fn is_killed(&self) -> bool {
        *self.killed.read().await
    }

    // ────────────────────────────────────────────────────────────────────
    // Access / inspection
    // ────────────────────────────────────────────────────────────────────

    /// Get a read reference to the correlation tracker.
    pub async fn correlation_tracker(
        &self,
    ) -> tokio::sync::RwLockReadGuard<'_, CorrelationTracker> {
        self.correlation_tracker.read().await
    }

    /// Get a read reference to the strategy gate.
    pub async fn strategy_gate(&self) -> tokio::sync::RwLockReadGuard<'_, StrategyGate> {
        self.strategy_gate.read().await
    }

    /// Get a mutable reference to the strategy gate (e.g. to replace the tracker).
    pub async fn strategy_gate_mut(&self) -> tokio::sync::RwLockWriteGuard<'_, StrategyGate> {
        self.strategy_gate.write().await
    }

    /// Get a snapshot of the pipeline metrics.
    pub async fn metrics_snapshot(&self) -> PipelineMetrics {
        let m = self.metrics.read().await;
        PipelineMetrics {
            total_evaluations: m.total_evaluations,
            proceed_count: m.proceed_count,
            block_count: m.block_count,
            reduce_only_count: m.reduce_only_count,
            blocks_by_stage: m.blocks_by_stage.clone(),
            total_evaluation_us: m.total_evaluation_us,
            last_action: m.last_action.clone(),
        }
    }

    /// Get a reference to the pipeline configuration.
    pub fn config(&self) -> &TradingPipelineConfig {
        &self.config
    }

    /// Get the list of strategies that the gate would allow for an asset/regime.
    pub async fn enabled_strategies_for<'a>(
        &self,
        asset: &str,
        regime: &MarketRegime,
        all_strategies: &'a [String],
    ) -> Vec<&'a str> {
        let gate = self.strategy_gate.read().await;
        gate.enabled_strategies(asset, regime, all_strategies)
    }

    /// Get the affinity weight for a strategy on an asset.
    pub async fn affinity_weight(&self, strategy: &str, asset: &str) -> f64 {
        let gate = self.strategy_gate.read().await;
        gate.tracker().weight_for(strategy, asset)
    }

    /// Get highly correlated pairs from the correlation tracker.
    pub async fn highly_correlated_pairs(&self) -> Vec<(String, String, f64)> {
        let tracker = self.correlation_tracker.read().await;
        tracker
            .highly_correlated_pairs()
            .into_iter()
            .map(|(a, b, c)| (a.to_string(), b.to_string(), c))
            .collect()
    }
}

// ============================================================================
// Pipeline Builder
// ============================================================================

/// Builder for constructing a `TradingPipeline` with custom components.
pub struct TradingPipelineBuilder {
    config: TradingPipelineConfig,
    correlation_tracker: Option<CorrelationTracker>,
    strategy_gate: Option<StrategyGate>,
}

impl TradingPipelineBuilder {
    pub fn new() -> Self {
        Self {
            config: TradingPipelineConfig::default(),
            correlation_tracker: None,
            strategy_gate: None,
        }
    }

    pub fn config(mut self, config: TradingPipelineConfig) -> Self {
        self.config = config;
        self
    }

    pub fn correlation_tracker(mut self, tracker: CorrelationTracker) -> Self {
        self.correlation_tracker = Some(tracker);
        self
    }

    pub fn strategy_gate(mut self, gate: StrategyGate) -> Self {
        self.strategy_gate = Some(gate);
        self
    }

    pub fn build(self) -> TradingPipeline {
        let correlation_tracker = self
            .correlation_tracker
            .unwrap_or_else(|| CorrelationTracker::new(self.config.correlation.clone()));
        let strategy_gate = self.strategy_gate.unwrap_or_else(|| {
            let tracker = StrategyAffinityTracker::new(self.config.affinity_min_trades);
            StrategyGate::new(self.config.gating.clone(), tracker)
        });

        TradingPipeline::with_components(self.config, correlation_tracker, strategy_gate)
    }
}

impl Default for TradingPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Convenience: create a pipeline-ready RoutedSignal for testing
// ============================================================================

/// Helper to create a `RoutedSignal` for testing/simulation.
pub fn make_test_signal(regime: MarketRegime, confidence: f64) -> RoutedSignal {
    use janus_regime::ActiveStrategy;

    let strategy = match regime {
        MarketRegime::Trending(_) => ActiveStrategy::TrendFollowing,
        MarketRegime::MeanReverting => ActiveStrategy::MeanReversion,
        _ => ActiveStrategy::NoTrade,
    };

    RoutedSignal {
        regime,
        strategy,
        confidence,
        trend_direction: match regime {
            MarketRegime::Trending(dir) => Some(dir),
            _ => None,
        },
        position_factor: 1.0,
        expected_duration: Some(10.0),
        methods_agree: Some(confidence > 0.7),
        reason: String::new(),
        detection_method: DetectionMethod::Ensemble,
        state_probabilities: None,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use janus_regime::{ActiveStrategy, MarketRegime, TrendDirection};

    fn bullish_trending_signal(confidence: f64) -> RoutedSignal {
        RoutedSignal {
            regime: MarketRegime::Trending(TrendDirection::Bullish),
            strategy: ActiveStrategy::TrendFollowing,
            confidence,
            trend_direction: Some(TrendDirection::Bullish),
            position_factor: 1.0,
            expected_duration: Some(10.0),
            methods_agree: Some(true),
            reason: String::new(),
            detection_method: DetectionMethod::Ensemble,
            state_probabilities: None,
        }
    }

    fn bearish_trending_signal(confidence: f64) -> RoutedSignal {
        RoutedSignal {
            regime: MarketRegime::Trending(TrendDirection::Bearish),
            strategy: ActiveStrategy::TrendFollowing,
            confidence,
            trend_direction: Some(TrendDirection::Bearish),
            position_factor: 1.0,
            expected_duration: Some(10.0),
            methods_agree: Some(true),
            reason: String::new(),
            detection_method: DetectionMethod::Ensemble,
            state_probabilities: None,
        }
    }

    fn volatile_signal(confidence: f64) -> RoutedSignal {
        RoutedSignal {
            regime: MarketRegime::Volatile,
            strategy: ActiveStrategy::NoTrade,
            confidence,
            trend_direction: None,
            position_factor: 0.5,
            expected_duration: Some(5.0),
            methods_agree: Some(true),
            reason: String::new(),
            detection_method: DetectionMethod::Ensemble,
            state_probabilities: None,
        }
    }

    fn mean_reverting_signal(confidence: f64) -> RoutedSignal {
        RoutedSignal {
            regime: MarketRegime::MeanReverting,
            strategy: ActiveStrategy::MeanReversion,
            confidence,
            trend_direction: None,
            position_factor: 1.0,
            expected_duration: Some(8.0),
            methods_agree: Some(true),
            reason: String::new(),
            detection_method: DetectionMethod::Ensemble,
            state_probabilities: None,
        }
    }

    fn uncertain_signal(confidence: f64) -> RoutedSignal {
        RoutedSignal {
            regime: MarketRegime::Uncertain,
            strategy: ActiveStrategy::NoTrade,
            confidence,
            trend_direction: None,
            position_factor: 0.3,
            expected_duration: Some(3.0),
            methods_agree: Some(false),
            reason: String::new(),
            detection_method: DetectionMethod::Ensemble,
            state_probabilities: None,
        }
    }

    // ── Config tests ───────────────────────────────────────────────────

    #[test]
    fn test_default_config() {
        let config = TradingPipelineConfig::default();
        assert!(config.enable_hypothalamus_scaling);
        assert!(config.enable_amygdala_filter);
        assert!(config.enable_gating);
        assert!(config.enable_correlation_filter);
        assert!(!config.allow_new_positions_in_crisis);
        assert_eq!(config.max_position_scale, 2.0);
        assert_eq!(config.min_position_scale, 0.05);
        assert_eq!(config.high_risk_scale_factor, 0.5);
        assert_eq!(config.affinity_min_trades, 10);
    }

    // ── TradeAction tests ──────────────────────────────────────────────

    #[test]
    fn test_trade_action_proceed_is_actionable() {
        let action = TradeAction::Proceed { scale: 1.0 };
        assert!(action.is_actionable());
        assert_eq!(action.scale(), 1.0);
    }

    #[test]
    fn test_trade_action_block_is_not_actionable() {
        let action = TradeAction::Block {
            reason: "test".into(),
            stage: PipelineStage::Amygdala,
        };
        assert!(!action.is_actionable());
        assert_eq!(action.scale(), 0.0);
    }

    #[test]
    fn test_trade_action_reduce_only_is_actionable() {
        let action = TradeAction::ReduceOnly {
            scale: 0.3,
            reason: "crisis".into(),
        };
        assert!(action.is_actionable());
        assert_eq!(action.scale(), 0.3);
    }

    #[test]
    fn test_trade_action_display() {
        let action = TradeAction::Proceed { scale: 1.2 };
        assert!(format!("{action}").contains("PROCEED"));

        let action = TradeAction::Block {
            reason: "test".into(),
            stage: PipelineStage::KillSwitch,
        };
        assert!(format!("{action}").contains("BLOCK@KillSwitch"));
    }

    // ── Pipeline stage display ─────────────────────────────────────────

    #[test]
    fn test_pipeline_stage_display() {
        assert_eq!(format!("{}", PipelineStage::RegimeBridge), "RegimeBridge");
        assert_eq!(format!("{}", PipelineStage::Hypothalamus), "Hypothalamus");
        assert_eq!(format!("{}", PipelineStage::Amygdala), "Amygdala");
        assert_eq!(format!("{}", PipelineStage::StrategyGate), "StrategyGate");
        assert_eq!(
            format!("{}", PipelineStage::CorrelationFilter),
            "CorrelationFilter"
        );
        assert_eq!(format!("{}", PipelineStage::KillSwitch), "KillSwitch");
    }

    // ── Sync evaluation tests ──────────────────────────────────────────

    #[test]
    fn test_sync_bullish_trending_proceeds() {
        let config = TradingPipelineConfig {
            enable_gating: false, // disable gating for this test
            enable_correlation_filter: false,
            ..Default::default()
        };
        let signal = bullish_trending_signal(0.85);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        assert!(decision.is_actionable());
        assert!(decision.scale() > 1.0, "Strong bullish should scale up");
        assert!(!decision.amygdala_high_risk);
        assert!(matches!(decision.action, TradeAction::Proceed { .. }));
    }

    #[test]
    fn test_sync_crisis_triggers_reduce_only() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        // Volatile + high confidence → Crisis
        let signal = volatile_signal(0.9);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        assert!(decision.amygdala_high_risk);
        assert!(
            matches!(decision.action, TradeAction::ReduceOnly { .. }),
            "Crisis should trigger ReduceOnly, got: {:?}",
            decision.action
        );
    }

    #[test]
    fn test_sync_kill_switch_blocks_everything() {
        let config = TradingPipelineConfig::default();
        let signal = bullish_trending_signal(0.9);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            true, // killed
            None,
            None,
            None,
            None,
        );

        assert!(!decision.is_actionable());
        assert!(matches!(
            decision.action,
            TradeAction::Block {
                stage: PipelineStage::KillSwitch,
                ..
            }
        ));
    }

    #[test]
    fn test_sync_low_confidence_blocked() {
        let config = TradingPipelineConfig {
            min_regime_confidence: 0.5,
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let signal = uncertain_signal(0.2);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        assert!(!decision.is_actionable());
        assert!(matches!(
            decision.action,
            TradeAction::Block {
                stage: PipelineStage::RegimeBridge,
                ..
            }
        ));
    }

    #[test]
    fn test_sync_high_risk_scales_down() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            high_risk_scale_factor: 0.3,
            allow_new_positions_in_crisis: true, // allow crisis so we get Proceed
            ..Default::default()
        };
        // High vol trending → is_high_risk = true
        let signal = volatile_signal(0.6); // moderate confidence → HighVolMeanReverting, still high_risk
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        // The regime should be high-risk, so scale should be reduced
        if decision.amygdala_high_risk {
            assert!(
                decision.scale() < decision.raw_hypothalamus_scale,
                "High-risk should reduce scale: got {} vs raw {}",
                decision.scale(),
                decision.raw_hypothalamus_scale,
            );
        }
    }

    #[test]
    fn test_sync_correlation_blocks_excess_positions() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: true,
            correlation: CorrelationConfig {
                window: 10,
                correlation_threshold: 0.5,
                max_correlated_positions: 2,
                min_observations: 5,
                monitored_pairs: Vec::new(),
            },
            ..Default::default()
        };

        let signal = bullish_trending_signal(0.85);

        // Create correlation tracker and feed highly correlated prices
        let mut ct = CorrelationTracker::new(config.correlation.clone());

        // Feed perfectly correlated series for BTC and ETH
        for i in 0..20 {
            let price = 100.0 + i as f64;
            ct.update("BTCUSDT", price);
            ct.update("ETHUSDT", price * 2.0);
            ct.update("SOLUSDT", price * 0.5);
        }

        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        // We already hold ETHUSDT and SOLUSDT which are correlated with BTCUSDT
        let positions = vec!["ETHUSDT".to_string(), "SOLUSDT".to_string()];

        let decision = TradingPipeline::evaluate_sync(
            &config, "BTCUSDT", &signal, "ema_flip", &positions, &ct, &gate, false, None, None,
            None, None,
        );

        // All three assets are perfectly correlated. We have 2 positions,
        // adding BTCUSDT would make 3 correlated positions, and max is 2.
        assert!(!decision.is_actionable());
        assert!(matches!(
            decision.action,
            TradeAction::Block {
                stage: PipelineStage::CorrelationFilter,
                ..
            }
        ));
    }

    #[test]
    fn test_sync_mean_reverting_proceeds() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let signal = mean_reverting_signal(0.7);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "mean_reversion",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        assert!(decision.is_actionable());
        // Mean reverting maps to Neutral → scale should be ~1.0
        assert!(
            (decision.scale() - 1.0).abs() < 0.01,
            "Mean reverting should have ~1.0 scale, got {}",
            decision.scale()
        );
    }

    #[test]
    fn test_sync_bearish_trending_scales_appropriately() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let signal = bearish_trending_signal(0.85);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        assert!(decision.is_actionable());
        // StrongBearish should scale down
        let raw = decision.raw_hypothalamus_scale;
        assert!(
            raw < 1.0,
            "Bearish should scale down, got raw_scale={}",
            raw
        );
    }

    // ── Async pipeline tests ───────────────────────────────────────────

    #[tokio::test]
    async fn test_pipeline_proceeds_for_bullish() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let pipeline = TradingPipeline::new(config);
        let signal = bullish_trending_signal(0.85);

        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;

        assert!(decision.is_actionable());
        assert!(decision.scale() > 1.0);
    }

    #[tokio::test]
    async fn test_pipeline_kill_switch() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let pipeline = TradingPipeline::new(config);

        assert!(!pipeline.is_killed().await);

        pipeline.activate_kill_switch().await;
        assert!(pipeline.is_killed().await);

        let signal = bullish_trending_signal(0.9);
        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
        assert!(!decision.is_actionable());

        // Deactivate and verify
        pipeline.deactivate_kill_switch().await;
        assert!(!pipeline.is_killed().await);

        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
        assert!(decision.is_actionable());
    }

    #[tokio::test]
    async fn test_pipeline_records_metrics() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let pipeline = TradingPipeline::new(config);
        let signal = bullish_trending_signal(0.85);

        pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
        pipeline
            .evaluate("ETHUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;

        let metrics = pipeline.metrics_snapshot().await;
        assert_eq!(metrics.total_evaluations, 2);
        assert_eq!(metrics.proceed_count, 2);
        assert_eq!(metrics.block_count, 0);
    }

    #[tokio::test]
    async fn test_pipeline_records_trade_results() {
        let config = TradingPipelineConfig::default();
        let pipeline = TradingPipeline::new(config);

        // Record some trades
        pipeline
            .record_trade_result("ema_flip", "BTCUSDT", 100.0, true)
            .await;
        pipeline
            .record_trade_result("ema_flip", "BTCUSDT", -30.0, false)
            .await;

        let weight = pipeline.affinity_weight("ema_flip", "BTCUSDT").await;
        // With only 2 trades (less than min_trades_for_confidence=10), weight should be neutral
        assert_eq!(weight, 0.5);
    }

    #[tokio::test]
    async fn test_pipeline_price_updates() {
        let config = TradingPipelineConfig::default();
        let pipeline = TradingPipeline::new(config);

        pipeline.update_price("BTCUSDT", 50000.0).await;
        pipeline.update_price("BTCUSDT", 50100.0).await;
        pipeline.update_price("ETHUSDT", 3000.0).await;
        pipeline.update_price("ETHUSDT", 3010.0).await;

        let tracker = pipeline.correlation_tracker().await;
        assert_eq!(tracker.tracked_assets().len(), 2);
    }

    #[tokio::test]
    async fn test_pipeline_batch_price_updates() {
        let config = TradingPipelineConfig::default();
        let pipeline = TradingPipeline::new(config);

        // Need at least 2 prices per asset to have returns
        pipeline.update_price("BTCUSDT", 50000.0).await;
        pipeline.update_price("ETHUSDT", 3000.0).await;

        pipeline
            .update_prices_batch(&[("BTCUSDT", 50100.0), ("ETHUSDT", 3010.0)])
            .await;

        let tracker = pipeline.correlation_tracker().await;
        assert!(tracker.tracked_assets().len() >= 2);
    }

    #[tokio::test]
    async fn test_pipeline_crisis_reduce_only() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            allow_new_positions_in_crisis: false,
            ..Default::default()
        };
        let pipeline = TradingPipeline::new(config);
        let signal = volatile_signal(0.9); // High confidence volatile → Crisis

        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;

        assert!(decision.amygdala_high_risk);
        assert!(matches!(decision.action, TradeAction::ReduceOnly { .. }));
    }

    #[tokio::test]
    async fn test_pipeline_crisis_allowed_when_configured() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            allow_new_positions_in_crisis: true, // allow it
            ..Default::default()
        };
        let pipeline = TradingPipeline::new(config);
        let signal = volatile_signal(0.9); // Crisis

        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;

        // Should proceed (with reduced scale) even in crisis
        assert!(decision.amygdala_high_risk);
        assert!(matches!(decision.action, TradeAction::Proceed { .. }));
        assert!(
            decision.scale() < 1.0,
            "Crisis should still reduce scale even when allowed"
        );
    }

    // ── Metrics tests ──────────────────────────────────────────────────

    #[test]
    fn test_pipeline_metrics_block_rate() {
        let mut metrics = PipelineMetrics::default();
        assert_eq!(metrics.block_rate_pct(), 0.0);
        assert_eq!(metrics.avg_evaluation_us(), 0.0);

        metrics.total_evaluations = 100;
        metrics.block_count = 25;
        assert!((metrics.block_rate_pct() - 25.0).abs() < f64::EPSILON);
    }

    // ── Builder tests ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_pipeline_builder_default() {
        let pipeline = TradingPipelineBuilder::new().build();
        let signal = bullish_trending_signal(0.85);

        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;

        // Default config has gating enabled but allow_untested=true,
        // so untested strategies should pass
        assert!(decision.is_actionable());
    }

    #[tokio::test]
    async fn test_pipeline_builder_custom_config() {
        let config = TradingPipelineConfig {
            min_regime_confidence: 0.8,
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let pipeline = TradingPipelineBuilder::new().config(config).build();

        // Low confidence should be blocked
        let signal = bullish_trending_signal(0.5);
        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
        assert!(!decision.is_actionable());

        // High confidence should pass
        let signal = bullish_trending_signal(0.9);
        let decision = pipeline
            .evaluate("BTCUSDT", &signal, "ema_flip", &[], None, None, None, None)
            .await;
        assert!(decision.is_actionable());
    }

    // ── make_test_signal tests ─────────────────────────────────────────

    #[test]
    fn test_make_test_signal_trending() {
        let signal = make_test_signal(MarketRegime::Trending(TrendDirection::Bullish), 0.8);
        assert_eq!(
            signal.regime,
            MarketRegime::Trending(TrendDirection::Bullish)
        );
        assert_eq!(signal.confidence, 0.8);
        assert_eq!(signal.strategy, ActiveStrategy::TrendFollowing);
        assert_eq!(signal.trend_direction, Some(TrendDirection::Bullish));
    }

    #[test]
    fn test_make_test_signal_mean_reverting() {
        let signal = make_test_signal(MarketRegime::MeanReverting, 0.6);
        assert_eq!(signal.regime, MarketRegime::MeanReverting);
        assert_eq!(signal.strategy, ActiveStrategy::MeanReversion);
    }

    #[test]
    fn test_make_test_signal_volatile() {
        let signal = make_test_signal(MarketRegime::Volatile, 0.9);
        assert_eq!(signal.regime, MarketRegime::Volatile);
        assert_eq!(signal.strategy, ActiveStrategy::NoTrade);
    }

    // ── TradingDecision display ────────────────────────────────────────

    #[test]
    fn test_decision_display() {
        let config = TradingPipelineConfig {
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let signal = bullish_trending_signal(0.85);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        let display = format!("{decision}");
        assert!(display.contains("BTCUSDT"));
        assert!(display.contains("ema_flip"));
        assert!(display.contains("PROCEED"));
    }

    // ── Disabled feature tests ─────────────────────────────────────────

    #[test]
    fn test_sync_all_stages_disabled() {
        let config = TradingPipelineConfig {
            enable_hypothalamus_scaling: false,
            enable_amygdala_filter: false,
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let signal = volatile_signal(0.9); // Would normally trigger crisis
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        // With everything disabled, should proceed at scale 1.0
        assert!(decision.is_actionable());
        assert_eq!(decision.scale(), 1.0);
    }

    #[test]
    fn test_sync_hypothalamus_disabled_uses_1x_scale() {
        let config = TradingPipelineConfig {
            enable_hypothalamus_scaling: false,
            enable_amygdala_filter: false,
            enable_gating: false,
            enable_correlation_filter: false,
            ..Default::default()
        };
        let signal = bullish_trending_signal(0.9);
        let ct = CorrelationTracker::with_defaults();
        let gate = StrategyGate::new(
            StrategyGatingConfig::default(),
            StrategyAffinityTracker::new(10),
        );

        let decision = TradingPipeline::evaluate_sync(
            &config,
            "BTCUSDT",
            &signal,
            "ema_flip",
            &[],
            &ct,
            &gate,
            false,
            None,
            None,
            None,
            None,
        );

        // With hypothalamus disabled, scale should be exactly 1.0
        assert_eq!(decision.scale(), 1.0);
    }

    // ── Enabled strategies query test ──────────────────────────────────

    #[tokio::test]
    async fn test_pipeline_enabled_strategies_for() {
        let config = TradingPipelineConfig {
            enable_gating: true,
            ..Default::default()
        };
        let pipeline = TradingPipeline::new(config);

        let all = vec![
            "ema_flip".to_string(),
            "mean_reversion".to_string(),
            "vwap_scalper".to_string(),
        ];

        // With default gating (allow_untested=true), all should pass
        let enabled = pipeline
            .enabled_strategies_for(
                "BTCUSDT",
                &MarketRegime::Trending(TrendDirection::Bullish),
                &all,
            )
            .await;
        assert_eq!(enabled.len(), 3);
    }
}
