//! # Brain-Gated Execution Client
//!
//! Wraps the standard `ExecutionClient` with a `TradingPipeline` gate.
//! Every signal submission is first evaluated through the full
//! regime → hypothalamus → amygdala → gating → correlation pipeline.
//! Only signals that receive `TradeAction::Proceed` (or `ReduceOnly`
//! when configured) are forwarded to the execution service.
//!
//! ## Architecture
//!
//! ```text
//! ┌────────────┐     ┌──────────────────┐     ┌─────────────────┐
//! │  Incoming   │────▶│  TradingPipeline  │────▶│ ExecutionClient │
//! │   Signal    │     │  (brain gate)     │     │   (gRPC)        │
//! └────────────┘     └──────────────────┘     └─────────────────┘
//!                         │                          │
//!                    Block / ReduceOnly          Submit order
//!                    → skip / adjust             → exchange
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use janus_forward::execution::BrainGatedExecutionClient;
//!
//! let gated = BrainGatedExecutionClient::new(exec_client, pipeline)
//!     .with_metrics(brain_metrics)
//!     .allow_reduce_only(true);
//!
//! // This will evaluate the signal through the brain pipeline first.
//! let result = gated.submit_signal(&signal, &routed_signal, "ema_flip", &positions).await;
//! ```

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::brain_wiring::{TradeAction, TradingDecision, TradingPipeline};
use crate::execution::client::{Exchange, ExecutionClient, SubmitSignalResponse};
use crate::metrics::BrainPipelineMetricsCollector;
use crate::signal::TradingSignal;
use janus_regime::RoutedSignal;

// ════════════════════════════════════════════════════════════════════
// Configuration
// ════════════════════════════════════════════════════════════════════

/// Configuration for the brain-gated execution client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainGatedConfig {
    /// When true, `ReduceOnly` decisions are forwarded to the execution
    /// client with reduced quantity. When false, `ReduceOnly` decisions
    /// are treated as blocks.
    #[serde(default = "default_true")]
    pub allow_reduce_only: bool,

    /// When true, the original signal's quantity/confidence is scaled by
    /// the pipeline's position scale factor before submission.
    #[serde(default = "default_true")]
    pub apply_scale_to_signal: bool,

    /// Maximum age (in seconds) for a signal before it is rejected.
    /// 0 = no staleness check.
    #[serde(default)]
    pub max_signal_age_secs: u64,

    /// When true, trade results are automatically fed back into the
    /// pipeline's affinity tracker after execution.
    #[serde(default = "default_true")]
    pub record_trade_results: bool,
}

fn default_true() -> bool {
    true
}

impl Default for BrainGatedConfig {
    fn default() -> Self {
        Self {
            allow_reduce_only: true,
            apply_scale_to_signal: true,
            max_signal_age_secs: 0,
            record_trade_results: true,
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Gated submission result
// ════════════════════════════════════════════════════════════════════

/// Outcome of a brain-gated signal submission.
#[derive(Debug, Clone)]
pub enum GatedSubmissionResult {
    /// Signal passed the pipeline and was submitted to the execution service.
    Submitted {
        decision: TradingDecision,
        response: SubmitSignalResponse,
        applied_scale: f64,
    },
    /// Signal was blocked by the pipeline and not submitted.
    Blocked {
        decision: TradingDecision,
        reason: String,
    },
    /// Signal was too old and was rejected before pipeline evaluation.
    Stale {
        signal_age_secs: i64,
        max_age_secs: u64,
    },
    /// An error occurred during pipeline evaluation or submission.
    Error {
        decision: Option<TradingDecision>,
        error: String,
    },
}

impl GatedSubmissionResult {
    /// Returns true if the signal was successfully submitted.
    pub fn is_submitted(&self) -> bool {
        matches!(self, GatedSubmissionResult::Submitted { .. })
    }

    /// Returns true if the signal was blocked by the pipeline.
    pub fn is_blocked(&self) -> bool {
        matches!(self, GatedSubmissionResult::Blocked { .. })
    }

    /// Returns the trading decision, if one was produced.
    pub fn decision(&self) -> Option<&TradingDecision> {
        match self {
            GatedSubmissionResult::Submitted { decision, .. } => Some(decision),
            GatedSubmissionResult::Blocked { decision, .. } => Some(decision),
            GatedSubmissionResult::Error { decision, .. } => decision.as_ref(),
            GatedSubmissionResult::Stale { .. } => None,
        }
    }
}

impl std::fmt::Display for GatedSubmissionResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GatedSubmissionResult::Submitted {
                decision,
                applied_scale,
                ..
            } => write!(
                f,
                "Submitted {} {} (scale={:.2})",
                decision.symbol, decision.strategy, applied_scale
            ),
            GatedSubmissionResult::Blocked { decision, reason } => {
                write!(
                    f,
                    "Blocked {} {} — {}",
                    decision.symbol, decision.strategy, reason
                )
            }
            GatedSubmissionResult::Stale {
                signal_age_secs,
                max_age_secs,
            } => write!(
                f,
                "Stale signal (age={}s, max={}s)",
                signal_age_secs, max_age_secs
            ),
            GatedSubmissionResult::Error { error, .. } => write!(f, "Error: {}", error),
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// Aggregate stats
// ════════════════════════════════════════════════════════════════════

/// Running statistics for the gated execution client.
#[derive(Debug, Clone, Default)]
pub struct GatedExecutionStats {
    pub total_signals: u64,
    pub submitted: u64,
    pub blocked: u64,
    pub stale_rejected: u64,
    pub errors: u64,
    pub total_scale_sum: f64,
}

impl GatedExecutionStats {
    /// Average scale factor of submitted trades.
    pub fn avg_submitted_scale(&self) -> f64 {
        if self.submitted == 0 {
            0.0
        } else {
            self.total_scale_sum / self.submitted as f64
        }
    }

    /// Submission rate (submitted / total).
    pub fn submission_rate(&self) -> f64 {
        if self.total_signals == 0 {
            0.0
        } else {
            self.submitted as f64 / self.total_signals as f64
        }
    }
}

// ════════════════════════════════════════════════════════════════════
// BrainGatedExecutionClient
// ════════════════════════════════════════════════════════════════════

/// An execution client that gates every signal submission through the
/// brain-inspired `TradingPipeline` before forwarding to the underlying
/// `ExecutionClient`.
pub struct BrainGatedExecutionClient {
    /// The underlying gRPC execution client.
    execution_client: ExecutionClient,
    /// The brain-inspired trading pipeline used as a gate.
    pipeline: Arc<TradingPipeline>,
    /// Configuration.
    config: BrainGatedConfig,
    /// Optional Prometheus metrics collector.
    metrics: Option<Arc<BrainPipelineMetricsCollector>>,
    /// Running statistics.
    stats: tokio::sync::RwLock<GatedExecutionStats>,
}

impl BrainGatedExecutionClient {
    /// Create a new brain-gated execution client.
    pub fn new(execution_client: ExecutionClient, pipeline: Arc<TradingPipeline>) -> Self {
        Self {
            execution_client,
            pipeline,
            config: BrainGatedConfig::default(),
            metrics: None,
            stats: tokio::sync::RwLock::new(GatedExecutionStats::default()),
        }
    }

    /// Create with a custom configuration.
    pub fn with_config(
        execution_client: ExecutionClient,
        pipeline: Arc<TradingPipeline>,
        config: BrainGatedConfig,
    ) -> Self {
        Self {
            execution_client,
            pipeline,
            config,
            metrics: None,
            stats: tokio::sync::RwLock::new(GatedExecutionStats::default()),
        }
    }

    /// Attach a Prometheus metrics collector.
    pub fn with_metrics(mut self, metrics: Arc<BrainPipelineMetricsCollector>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// Set whether ReduceOnly decisions are forwarded.
    pub fn allow_reduce_only(mut self, allow: bool) -> Self {
        self.config.allow_reduce_only = allow;
        self
    }

    // ────────────────────────────────────────────────────────────────
    // Core: gated signal submission
    // ────────────────────────────────────────────────────────────────

    /// Evaluate a signal through the brain pipeline and, if approved,
    /// submit it to the execution service.
    ///
    /// # Parameters
    ///
    /// * `signal` — The `TradingSignal` to submit.
    /// * `routed_signal` — The regime `RoutedSignal` for pipeline context.
    /// * `strategy_name` — Name of the strategy that produced the signal.
    /// * `current_positions` — List of currently open position symbols.
    /// * `adx_value` — Optional ADX indicator value.
    /// * `bb_width_percentile` — Optional Bollinger Band width percentile.
    /// * `atr_value` — Optional ATR value.
    /// * `relative_volume` — Optional relative volume.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_gated(
        &mut self,
        signal: &TradingSignal,
        routed_signal: &RoutedSignal,
        strategy_name: &str,
        current_positions: &[String],
        adx_value: Option<f64>,
        bb_width_percentile: Option<f64>,
        atr_value: Option<f64>,
        relative_volume: Option<f64>,
    ) -> GatedSubmissionResult {
        // ── Staleness check ────────────────────────────────────────
        if self.config.max_signal_age_secs > 0 {
            let age = signal.age_seconds();
            if age > self.config.max_signal_age_secs as i64 {
                let mut stats = self.stats.write().await;
                stats.total_signals += 1;
                stats.stale_rejected += 1;
                return GatedSubmissionResult::Stale {
                    signal_age_secs: age,
                    max_age_secs: self.config.max_signal_age_secs,
                };
            }
        }

        // ── Pipeline evaluation ────────────────────────────────────
        let decision = self
            .pipeline
            .evaluate(
                &signal.symbol,
                routed_signal,
                strategy_name,
                current_positions,
                adx_value,
                bb_width_percentile,
                atr_value,
                relative_volume,
            )
            .await;

        // Record into Prometheus metrics
        if let Some(metrics) = &self.metrics {
            metrics.record_decision(&decision);
            metrics.set_kill_switch(self.pipeline.is_killed().await);
        }

        // ── Gate the decision ──────────────────────────────────────
        let mut stats = self.stats.write().await;
        stats.total_signals += 1;

        match &decision.action {
            TradeAction::Proceed { scale } => {
                let applied_scale = *scale;
                drop(stats); // release lock before async call

                let result = self.submit_with_scale(signal, applied_scale).await;

                let mut stats = self.stats.write().await;
                match result {
                    Ok(response) => {
                        stats.submitted += 1;
                        stats.total_scale_sum += applied_scale;

                        info!(
                            "✅ Brain-gated submission: {} {} scale={:.2} → order={}",
                            decision.symbol,
                            decision.strategy,
                            applied_scale,
                            response.order_id.as_deref().unwrap_or("N/A"),
                        );

                        // Record trade result into affinity tracker
                        if self.config.record_trade_results {
                            // We record a successful submission (not yet a fill)
                            // Actual P&L recording should happen when the fill is received.
                            debug!(
                                "Trade result will be recorded when fill is received for {}",
                                signal.symbol
                            );
                        }

                        GatedSubmissionResult::Submitted {
                            decision,
                            response,
                            applied_scale,
                        }
                    }
                    Err(e) => {
                        stats.errors += 1;
                        warn!(
                            "❌ Brain-gated submission approved but execution failed: {}",
                            e
                        );
                        GatedSubmissionResult::Error {
                            decision: Some(decision),
                            error: e.to_string(),
                        }
                    }
                }
            }

            TradeAction::ReduceOnly { scale, reason } => {
                if self.config.allow_reduce_only {
                    let applied_scale = *scale;
                    let reason_clone = reason.clone();
                    drop(stats);

                    info!(
                        "⚠️  Brain-gated ReduceOnly: {} {} scale={:.2} — {}",
                        decision.symbol, decision.strategy, applied_scale, reason_clone,
                    );

                    let result = self.submit_with_scale(signal, applied_scale).await;

                    let mut stats = self.stats.write().await;
                    match result {
                        Ok(response) => {
                            stats.submitted += 1;
                            stats.total_scale_sum += applied_scale;

                            GatedSubmissionResult::Submitted {
                                decision,
                                response,
                                applied_scale,
                            }
                        }
                        Err(e) => {
                            stats.errors += 1;
                            GatedSubmissionResult::Error {
                                decision: Some(decision),
                                error: e.to_string(),
                            }
                        }
                    }
                } else {
                    stats.blocked += 1;
                    let reason_str = format!("ReduceOnly not allowed: {}", reason);
                    info!(
                        "🚫 Brain-gated block (ReduceOnly disabled): {} {} — {}",
                        decision.symbol, decision.strategy, reason_str,
                    );
                    GatedSubmissionResult::Blocked {
                        decision,
                        reason: reason_str,
                    }
                }
            }

            TradeAction::Block { reason, stage } => {
                stats.blocked += 1;
                let reason_str = format!("[{}] {}", stage, reason);
                info!(
                    "🚫 Brain-gated block: {} {} — {}",
                    decision.symbol, decision.strategy, reason_str,
                );
                GatedSubmissionResult::Blocked {
                    decision,
                    reason: reason_str,
                }
            }
        }
    }

    /// Submit a signal to a specific exchange, with brain gating.
    #[allow(clippy::too_many_arguments)]
    pub async fn submit_gated_to_exchange(
        &mut self,
        signal: &TradingSignal,
        exchange: Exchange,
        routed_signal: &RoutedSignal,
        strategy_name: &str,
        current_positions: &[String],
        adx_value: Option<f64>,
        bb_width_percentile: Option<f64>,
        atr_value: Option<f64>,
        relative_volume: Option<f64>,
    ) -> GatedSubmissionResult {
        // Inject exchange into signal metadata
        let mut modified_signal = signal.clone();
        modified_signal
            .metadata
            .insert("exchange".to_string(), exchange.as_str().to_string());

        self.submit_gated(
            &modified_signal,
            routed_signal,
            strategy_name,
            current_positions,
            adx_value,
            bb_width_percentile,
            atr_value,
            relative_volume,
        )
        .await
    }

    // ────────────────────────────────────────────────────────────────
    // Internal: scaled submission
    // ────────────────────────────────────────────────────────────────

    /// Submit a signal with the given scale factor applied to its
    /// confidence (which the execution client uses for quantity sizing).
    async fn submit_with_scale(
        &mut self,
        signal: &TradingSignal,
        scale: f64,
    ) -> Result<SubmitSignalResponse> {
        if self.config.apply_scale_to_signal {
            let mut scaled_signal = signal.clone();
            // Scale confidence which is used by ExecutionClient for quantity
            scaled_signal.confidence = (signal.confidence * scale).clamp(0.0, 1.0);
            // Annotate the metadata with the brain scale factor
            scaled_signal
                .metadata
                .insert("brain_scale".to_string(), format!("{:.4}", scale));
            scaled_signal
                .metadata
                .insert("brain_gated".to_string(), "true".to_string());
            scaled_signal
                .metadata
                .insert("brain_gated_at".to_string(), Utc::now().to_rfc3339());
            self.execution_client.submit_signal(&scaled_signal).await
        } else {
            let mut annotated_signal = signal.clone();
            annotated_signal
                .metadata
                .insert("brain_scale".to_string(), format!("{:.4}", scale));
            annotated_signal
                .metadata
                .insert("brain_gated".to_string(), "true".to_string());
            self.execution_client.submit_signal(&annotated_signal).await
        }
    }

    // ────────────────────────────────────────────────────────────────
    // Accessors
    // ────────────────────────────────────────────────────────────────

    /// Get the underlying pipeline reference.
    pub fn pipeline(&self) -> &Arc<TradingPipeline> {
        &self.pipeline
    }

    /// Get a snapshot of the running statistics.
    pub async fn stats(&self) -> GatedExecutionStats {
        self.stats.read().await.clone()
    }

    /// Get the underlying execution client configuration.
    pub fn execution_config(&self) -> &crate::execution::ExecutionClientConfig {
        self.execution_client.config()
    }

    /// Get the underlying execution client's endpoint.
    pub fn endpoint(&self) -> &str {
        self.execution_client.endpoint()
    }

    /// Get the brain-gated config.
    pub fn config(&self) -> &BrainGatedConfig {
        &self.config
    }

    /// Perform a health check on the underlying execution service.
    pub async fn health_check(&mut self) -> Result<bool> {
        self.execution_client.health_check().await
    }

    /// Activate the pipeline kill switch (blocks all future trades).
    pub async fn activate_kill_switch(&self) {
        self.pipeline.activate_kill_switch().await;
        if let Some(metrics) = &self.metrics {
            metrics.set_kill_switch(true);
        }
    }

    /// Deactivate the pipeline kill switch.
    pub async fn deactivate_kill_switch(&self) {
        self.pipeline.deactivate_kill_switch().await;
        if let Some(metrics) = &self.metrics {
            metrics.set_kill_switch(false);
        }
    }

    /// Check if the pipeline kill switch is active.
    pub async fn is_killed(&self) -> bool {
        self.pipeline.is_killed().await
    }
}

// ════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brain_wiring::TradingPipelineConfig;

    // NOTE: Full integration tests require a running execution service.
    // These unit tests validate the gating logic, config, and stats only.

    #[test]
    fn test_default_config() {
        let config = BrainGatedConfig::default();
        assert!(config.allow_reduce_only);
        assert!(config.apply_scale_to_signal);
        assert_eq!(config.max_signal_age_secs, 0);
        assert!(config.record_trade_results);
    }

    #[test]
    fn test_gated_submission_result_display() {
        use crate::brain_wiring::{PipelineStage, TradeAction, TradingDecision, make_test_signal};
        use crate::regime_bridge::bridge_regime_signal;
        use janus_regime::MarketRegime;

        // Build a real BridgedRegimeState via the bridge function
        let routed = make_test_signal(MarketRegime::Uncertain, 0.3);
        let bridged = bridge_regime_signal("BTCUSDT", &routed, None, None, None, None);

        let decision = TradingDecision {
            symbol: "BTCUSDT".to_string(),
            strategy: "ema_flip".to_string(),
            action: TradeAction::Block {
                reason: "Low confidence".to_string(),
                stage: PipelineStage::RegimeBridge,
            },
            bridged_regime: bridged,
            raw_hypothalamus_scale: 0.0,
            amygdala_high_risk: false,
            gate_approved: false,
            correlation_passed: false,
            timestamp: Utc::now(),
            evaluation_us: 42,
        };

        let result = GatedSubmissionResult::Blocked {
            decision,
            reason: "[RegimeBridge] Low confidence".to_string(),
        };

        assert!(result.is_blocked());
        assert!(!result.is_submitted());
        assert!(result.decision().is_some());

        let display = format!("{}", result);
        assert!(display.contains("Blocked"));
        assert!(display.contains("BTCUSDT"));
    }

    #[test]
    fn test_stale_result_display() {
        let result = GatedSubmissionResult::Stale {
            signal_age_secs: 120,
            max_age_secs: 60,
        };

        assert!(!result.is_submitted());
        assert!(!result.is_blocked());
        assert!(result.decision().is_none());

        let display = format!("{}", result);
        assert!(display.contains("Stale"));
        assert!(display.contains("120"));
    }

    #[test]
    fn test_gated_execution_stats() {
        let mut stats = GatedExecutionStats::default();
        assert_eq!(stats.submission_rate(), 0.0);
        assert_eq!(stats.avg_submitted_scale(), 0.0);

        stats.total_signals = 10;
        stats.submitted = 6;
        stats.blocked = 3;
        stats.errors = 1;
        stats.total_scale_sum = 4.8;

        assert!((stats.submission_rate() - 0.6).abs() < 0.001);
        assert!((stats.avg_submitted_scale() - 0.8).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_pipeline_kill_switch_blocks_via_gated_client() {
        // This test validates that when the pipeline kill switch is active,
        // the pipeline will produce a Block decision. We can't call
        // submit_gated without a real execution client, but we can verify
        // the pipeline's behavior directly.
        let pipeline = Arc::new(TradingPipeline::new(TradingPipelineConfig::default()));
        pipeline.activate_kill_switch().await;
        assert!(pipeline.is_killed().await);

        // Create a test signal and evaluate directly through the pipeline
        use janus_regime::{MarketRegime, TrendDirection};
        let signal = crate::brain_wiring::make_test_signal(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.9,
        );
        let decision = pipeline
            .evaluate(
                "BTCUSDT",
                &signal,
                "test_strategy",
                &[],
                None,
                None,
                None,
                None,
            )
            .await;

        assert!(!decision.is_actionable());
        match &decision.action {
            TradeAction::Block { stage, .. } => {
                assert!(matches!(
                    stage,
                    crate::brain_wiring::PipelineStage::KillSwitch
                ));
            }
            _ => panic!("Expected Block from kill switch"),
        }
    }
}
