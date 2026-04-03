//! # Signal Generation Module
//!
//! This module provides signal generation with support for dynamic parameter
//! hot-reload from the optimizer. When a `ParamReloadHandle` is provided,
//! the signal generator will query per-asset indicator configurations
//! that have been optimized and published via Redis.
//!
//! Core signal generation functionality for the JANUS service.
//! This module consolidates all signal-related types and generation logic.

pub mod types;

pub use types::{SignalBatch, SignalSource, SignalType, Timeframe, TradingSignal};

use crate::execution::{
    BrainGatedExecutionClient, ExecutionClient, ExecutionClientConfig, GatedSubmissionResult,
};
use crate::features::{FeatureConfig, FeatureEngineering};
use crate::indicators::IndicatorAnalysis;
use crate::inference::{ModelCache, ModelInference};
use anyhow::{Result, anyhow};
use janus_regime::{ActiveStrategy, DetectionMethod, MarketRegime, RoutedSignal};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn};

/// Signal generator configuration
#[derive(Debug, Clone)]
pub struct SignalGeneratorConfig {
    /// Minimum confidence threshold (0.0 to 1.0)
    pub min_confidence: f64,

    /// Minimum strength threshold (0.0 to 1.0)
    pub min_strength: f64,

    /// Maximum signal age in seconds before considered stale
    pub max_age_seconds: i64,

    /// Enable quality filtering
    pub enable_quality_filter: bool,

    /// Batch size for signal processing
    pub batch_size: usize,

    /// Enable ML model inference
    pub enable_ml_inference: bool,

    /// ML model weight in signal fusion (0.0 to 1.0)
    pub ml_weight: f64,

    /// Feature engineering configuration
    pub feature_config: FeatureConfig,
}

impl Default for SignalGeneratorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            min_strength: 0.5,
            max_age_seconds: 300, // 5 minutes
            enable_quality_filter: true,
            batch_size: 100,
            enable_ml_inference: false,
            ml_weight: 0.5,
            feature_config: FeatureConfig::default(),
        }
    }
}

/// Core signal generator
///
/// Responsible for generating trading signals from various sources:
/// - Technical indicators (EMA, RSI, MACD, etc.)
/// - ML models (ONNX inference)
/// - Combined strategies
pub struct SignalGenerator {
    config: SignalGeneratorConfig,
    signal_cache: Arc<RwLock<Vec<TradingSignal>>>,
    metrics: Arc<SignalMetrics>,
    feature_engineering: Arc<RwLock<FeatureEngineering>>,
    model_inference: Option<Arc<ModelInference>>,
    execution_client: Option<Arc<Mutex<ExecutionClient>>>,
    /// Optional brain-gated execution client.
    /// When set, all signal submissions are routed through the brain pipeline
    /// before reaching the execution service. The raw `execution_client` is
    /// bypassed in favor of this gated wrapper.
    brain_gated_client: Option<Arc<Mutex<BrainGatedExecutionClient>>>,
    /// Optional param query handle for dynamic config updates
    param_query_handle: Option<crate::ParamQueryHandle>,
}

impl SignalGenerator {
    /// Create a new signal generator
    pub fn new(config: SignalGeneratorConfig) -> Self {
        let feature_engineering = FeatureEngineering::new(config.feature_config.clone());

        Self {
            config,
            signal_cache: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(SignalMetrics::default()),
            feature_engineering: Arc::new(RwLock::new(feature_engineering)),
            model_inference: None,
            execution_client: None,
            brain_gated_client: None,
            param_query_handle: None,
        }
    }

    /// Create a new signal generator with execution client
    pub async fn new_with_execution(
        config: SignalGeneratorConfig,
        execution_config: Option<ExecutionClientConfig>,
    ) -> Result<Self> {
        let feature_engineering = FeatureEngineering::new(config.feature_config.clone());

        let execution_client = if let Some(exec_cfg) = execution_config {
            match ExecutionClient::new(exec_cfg).await {
                Ok(client) => {
                    info!("✅ Execution client connected");
                    Some(Arc::new(Mutex::new(client)))
                }
                Err(e) => {
                    warn!("Failed to connect execution client: {}", e);
                    warn!("Signals will be generated but not executed");
                    None
                }
            }
        } else {
            info!("Execution client not configured - signals will not be auto-executed");
            None
        };

        Ok(Self {
            config,
            signal_cache: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(SignalMetrics::default()),
            feature_engineering: Arc::new(RwLock::new(feature_engineering)),
            model_inference: None,
            execution_client,
            brain_gated_client: None,
            param_query_handle: None,
        })
    }

    /// Create a new signal generator with ML inference enabled
    pub fn with_ml_inference(mut config: SignalGeneratorConfig) -> Self {
        config.enable_ml_inference = true;
        let feature_engineering = FeatureEngineering::new(config.feature_config.clone());
        let model_cache = ModelCache::new();
        let model_inference = ModelInference::new(model_cache);

        Self {
            config,
            signal_cache: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(SignalMetrics::default()),
            feature_engineering: Arc::new(RwLock::new(feature_engineering)),
            model_inference: Some(Arc::new(model_inference)),
            execution_client: None,
            brain_gated_client: None,
            param_query_handle: None,
        }
    }

    /// Set the parameter query handle for dynamic config updates
    ///
    /// When set, the signal generator will query optimized indicator configs
    /// per-asset before generating signals.
    pub fn set_param_query_handle(&mut self, handle: crate::ParamQueryHandle) {
        info!("Signal generator: param query handle attached");
        self.param_query_handle = Some(handle);
    }

    /// Attach a brain-gated execution client.
    ///
    /// When set, all signal submissions (`submit_to_execution`,
    /// `submit_signal_to_execution`) are routed through the brain pipeline
    /// gate before reaching the execution service. The raw `execution_client`
    /// is bypassed.
    ///
    /// Call this after constructing the `SignalGenerator` and after
    /// `BrainRuntime::boot()` has produced a `TradingPipeline`.
    pub fn set_brain_gated_client(&mut self, client: BrainGatedExecutionClient) {
        info!("🧠 Signal generator: brain-gated execution client attached");
        self.brain_gated_client = Some(Arc::new(Mutex::new(client)));
    }

    /// Check whether the brain-gated execution path is active.
    pub fn has_brain_gated_client(&self) -> bool {
        self.brain_gated_client.is_some()
    }

    /// Get a reference to the brain-gated client (if set).
    pub fn brain_gated_client(&self) -> Option<&Arc<Mutex<BrainGatedExecutionClient>>> {
        self.brain_gated_client.as_ref()
    }

    /// Get the current param query handle (if set)
    pub fn param_query_handle(&self) -> Option<&crate::ParamQueryHandle> {
        self.param_query_handle.as_ref()
    }

    /// Check if trading is enabled for a given asset via param reload
    ///
    /// Returns `true` if:
    /// - No param query handle is set (default allow)
    /// - Param query handle says trading is enabled for this asset
    pub async fn is_trading_enabled(&self, asset: &str) -> bool {
        if let Some(ref handle) = self.param_query_handle {
            handle.is_trading_enabled(asset).await
        } else {
            true // Default: allow trading if no param query handle
        }
    }

    /// Get optimized indicator config for an asset
    ///
    /// Returns the optimized config if available, otherwise None
    pub async fn get_optimized_indicator_config(
        &self,
        asset: &str,
    ) -> Option<crate::indicators::IndicatorConfig> {
        if let Some(ref handle) = self.param_query_handle {
            handle.get_indicator_config(asset).await
        } else {
            None
        }
    }

    /// Check strategy constraints from optimized params
    pub async fn check_strategy_constraints(&self, asset: &str, ema_spread: f64) -> bool {
        if let Some(ref handle) = self.param_query_handle {
            handle.check_min_ema_spread(asset, ema_spread).await
        } else {
            true // Default: pass if no param query handle
        }
    }

    /// Submit signal to execution service if client is available.
    ///
    /// When a brain-gated execution client is attached, the signal is routed
    /// through the brain pipeline first. A synthetic `RoutedSignal` is built
    /// from the signal's metadata (confidence, symbol) so the pipeline can
    /// evaluate regime / gating / correlation checks.
    async fn submit_to_execution(&self, signal: &TradingSignal) -> Result<()> {
        // ── Brain-gated path ───────────────────────────────────────
        if let Some(gated) = &self.brain_gated_client {
            let routed = Self::synthetic_routed_signal(signal.confidence);
            let strategy_name = signal
                .metadata
                .get("strategy")
                .map(|s| s.as_str())
                .unwrap_or("signal_generator");
            let positions: Vec<String> = Vec::new(); // no position context here

            let mut client = gated.lock().await;
            let result = client
                .submit_gated(
                    signal,
                    &routed,
                    strategy_name,
                    &positions,
                    None, // adx
                    None, // bb_width
                    None, // atr
                    None, // relative_volume
                )
                .await;

            match &result {
                GatedSubmissionResult::Submitted {
                    response,
                    applied_scale,
                    ..
                } => {
                    info!(
                        "🧠✅ Signal {} brain-gated submitted (scale={:.2}, order: {})",
                        signal.signal_id,
                        applied_scale,
                        response.order_id.as_deref().unwrap_or("N/A")
                    );
                }
                GatedSubmissionResult::Blocked { reason, .. } => {
                    info!(
                        "🧠🚫 Signal {} blocked by brain pipeline: {}",
                        signal.signal_id, reason
                    );
                }
                GatedSubmissionResult::Stale {
                    signal_age_secs,
                    max_age_secs,
                } => {
                    warn!(
                        "🧠⏰ Signal {} stale (age={}s, max={}s)",
                        signal.signal_id, signal_age_secs, max_age_secs
                    );
                }
                GatedSubmissionResult::Error { error, .. } => {
                    error!(
                        "🧠❌ Signal {} brain-gated error: {}",
                        signal.signal_id, error
                    );
                }
            }

            if result.is_submitted() {
                Ok(())
            } else if result.is_blocked() {
                // Blocked is not an error — the pipeline decided not to trade
                Ok(())
            } else {
                // Stale / Error — return error so callers can react
                Err(anyhow!("Brain-gated submission failed: {}", result))
            }
        }
        // ── Direct (non-gated) path ────────────────────────────────
        else if let Some(client) = &self.execution_client {
            let mut client = client.lock().await;
            match client.submit_signal(signal).await {
                Ok(response) => {
                    info!(
                        "✅ Signal {} submitted to execution (order: {})",
                        signal.signal_id,
                        response.order_id.as_deref().unwrap_or("N/A")
                    );
                    Ok(())
                }
                Err(e) => {
                    error!(
                        "❌ Failed to submit signal {} to execution: {}",
                        signal.signal_id, e
                    );
                    Err(e)
                }
            }
        } else {
            debug!(
                "Execution client not configured, signal {} not submitted",
                signal.signal_id
            );
            Ok(())
        }
    }

    /// Submit a janus_core::Signal to execution service (public interface for unified binary).
    ///
    /// When a brain-gated execution client is attached, the signal is converted
    /// to a `TradingSignal` and routed through the brain pipeline.
    pub async fn submit_signal_to_execution(&self, signal: &janus_core::Signal) -> Result<()> {
        // Convert janus_core::Signal to TradingSignal for submission
        let signal_type = match signal.signal_type {
            janus_core::SignalType::Buy => SignalType::Buy,
            janus_core::SignalType::Sell => SignalType::Sell,
            janus_core::SignalType::Hold => SignalType::Hold,
            janus_core::SignalType::Close => SignalType::Sell, // Map Close to Sell for execution
        };

        let strategy_name = signal
            .strategy_id
            .clone()
            .unwrap_or_else(|| "unified".to_string());

        let mut trading_signal = TradingSignal::new(
            signal.symbol.clone(),
            signal_type,
            Timeframe::M15, // Default timeframe
            signal.confidence,
            SignalSource::TechnicalIndicator {
                name: strategy_name.clone(),
            },
        );
        trading_signal
            .metadata
            .insert("strategy".to_string(), strategy_name);

        // ── Brain-gated path ───────────────────────────────────────
        if self.brain_gated_client.is_some() {
            return self.submit_to_execution(&trading_signal).await;
        }

        // ── Direct (non-gated) path ────────────────────────────────
        if let Some(client) = &self.execution_client {
            let mut client = client.lock().await;

            match client.submit_signal(&trading_signal).await {
                Ok(response) => {
                    info!(
                        "✅ Signal {} submitted to execution (order: {})",
                        signal.id,
                        response.order_id.as_deref().unwrap_or("N/A")
                    );
                    Ok(())
                }
                Err(e) => {
                    error!(
                        "❌ Failed to submit signal {} to execution: {}",
                        signal.id, e
                    );
                    Err(e)
                }
            }
        } else {
            debug!(
                "Execution client not configured, signal {} not submitted",
                signal.id
            );
            Ok(())
        }
    }

    /// Build a synthetic `RoutedSignal` for pipeline evaluation when
    /// the caller doesn't have full regime context. Uses the signal's
    /// confidence and defaults for other fields.
    fn synthetic_routed_signal(confidence: f64) -> RoutedSignal {
        RoutedSignal {
            strategy: ActiveStrategy::NoTrade,
            regime: MarketRegime::Uncertain,
            confidence,
            position_factor: 1.0,
            reason: "synthetic-for-brain-gate".to_string(),
            detection_method: DetectionMethod::Ensemble,
            methods_agree: None,
            state_probabilities: None,
            expected_duration: None,
            trend_direction: None,
        }
    }

    /// Load an ML model for inference
    pub async fn load_model(&self, name: &str, path: impl AsRef<std::path::Path>) -> Result<()> {
        if let Some(ref inference) = self.model_inference {
            inference.load_model(name, path).await?;
            info!("Loaded ML model '{}' for signal generation", name);
            Ok(())
        } else {
            Err(anyhow!("ML inference not enabled"))
        }
    }

    /// Generate signal from indicator analysis with optional ML fusion
    pub async fn generate_from_analysis(
        &self,
        symbol: String,
        timeframe: Timeframe,
        analysis: &IndicatorAnalysis,
        current_price: f64,
    ) -> Result<Option<TradingSignal>> {
        debug!(
            "Generating signal from analysis for {} on {}",
            symbol,
            timeframe.as_str()
        );

        // Extract asset name from symbol (e.g., "BTC/USD" -> "BTC")
        let asset = symbol.split('/').next().unwrap_or(&symbol);

        // Check if trading is enabled for this asset via param reload
        if !self.is_trading_enabled(asset).await {
            debug!(
                "Trading disabled for asset {} via optimized params, skipping signal",
                asset
            );
            return Ok(None);
        }

        // Check EMA spread constraint if we have optimized params
        if let (Some(ema_fast), Some(ema_slow)) = (analysis.ema_fast, analysis.ema_slow) {
            let ema_spread = ((ema_fast - ema_slow) / ema_slow * 100.0).abs();
            if !self.check_strategy_constraints(asset, ema_spread).await {
                debug!(
                    "EMA spread {:.4}% below minimum for asset {}, skipping signal",
                    ema_spread, asset
                );
                return Ok(None);
            }
        }

        // Update price history
        {
            let mut fe = self.feature_engineering.write().await;
            fe.update_price(current_price);
        }

        // Extract features
        let features = {
            let fe = self.feature_engineering.read().await;
            fe.extract_features(analysis, current_price)?
        };

        // Get indicator-based signal
        let indicator_signal_type = self.analyze_from_analysis(analysis)?;
        let indicator_confidence = self.calculate_analysis_confidence(analysis);

        // If ML inference is enabled, fuse predictions
        let (final_signal_type, final_confidence) = if self.config.enable_ml_inference {
            if let Some(ref inference) = self.model_inference {
                match inference.predict("signal_classifier", &features).await {
                    Ok(prediction) => {
                        debug!(
                            "ML prediction: {:?} (confidence: {:.3})",
                            prediction.signal_type, prediction.confidence
                        );

                        // Fuse indicator and ML predictions
                        self.fuse_predictions(
                            indicator_signal_type,
                            indicator_confidence,
                            prediction.signal_type,
                            prediction.confidence,
                        )
                    }
                    Err(e) => {
                        warn!("ML inference failed, falling back to indicators: {}", e);
                        (indicator_signal_type, indicator_confidence)
                    }
                }
            } else {
                (indicator_signal_type, indicator_confidence)
            }
        } else {
            (indicator_signal_type, indicator_confidence)
        };

        if final_signal_type == SignalType::Hold {
            return Ok(None);
        }

        // Create signal with appropriate source
        let source = if self.config.enable_ml_inference {
            SignalSource::ModelInference {
                model_name: "signal_classifier".to_string(),
                version: "v1".to_string(),
            }
        } else {
            SignalSource::TechnicalIndicator {
                name: "COMBINED".to_string(),
            }
        };

        let signal = TradingSignal::new(
            symbol,
            final_signal_type,
            timeframe,
            final_confidence,
            source,
        )
        .with_strength(final_confidence);

        // Apply quality filter
        if self.config.enable_quality_filter
            && !signal.meets_threshold(self.config.min_confidence, self.config.min_strength)
        {
            debug!("Signal filtered out due to quality threshold");
            self.metrics
                .filtered_signals
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(None);
        }

        self.metrics
            .generated_signals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        info!(
            "Generated signal: {:?} for {} (confidence: {:.3})",
            signal.signal_type, signal.symbol, signal.confidence
        );

        // Submit to execution service if available
        if let Err(e) = self.submit_to_execution(&signal).await {
            warn!(
                "Signal {} generated but execution submission failed: {}",
                signal.signal_id, e
            );
            // Continue - signal is still valid even if submission failed
        }

        Ok(Some(signal))
    }

    /// Fuse indicator and ML predictions
    fn fuse_predictions(
        &self,
        indicator_type: SignalType,
        indicator_confidence: f64,
        ml_type: SignalType,
        ml_confidence: f64,
    ) -> (SignalType, f64) {
        let ml_weight = self.config.ml_weight;
        let indicator_weight = 1.0 - ml_weight;

        // If predictions agree, boost confidence
        if indicator_type == ml_type {
            let fused_confidence =
                (indicator_confidence * indicator_weight + ml_confidence * ml_weight) * 1.2;
            return (indicator_type, fused_confidence.min(1.0));
        }

        // If predictions conflict, use weighted voting
        let indicator_score = indicator_confidence * indicator_weight;
        let ml_score = ml_confidence * ml_weight;

        if ml_score > indicator_score {
            (ml_type, ml_score)
        } else {
            (indicator_type, indicator_score)
        }
    }

    /// Analyze indicators from IndicatorAnalysis
    fn analyze_from_analysis(&self, analysis: &IndicatorAnalysis) -> Result<SignalType> {
        let mut score = 0.0;

        // EMA crossover
        if analysis.ema_cross != 0.0 {
            score += analysis.ema_cross * 0.3;
        }

        // RSI logic
        if let Some(rsi) = analysis.rsi {
            if rsi < 30.0 {
                score += 0.4; // Oversold
            } else if rsi > 70.0 {
                score -= 0.4; // Overbought
            }
        }

        // MACD logic
        if let Some(histogram) = analysis.macd_histogram {
            if histogram > 0.0 {
                score += 0.3;
            } else {
                score -= 0.3;
            }
        }

        // Bollinger Bands
        if let (Some(_upper), Some(_lower)) = (analysis.bb_upper, analysis.bb_lower) {
            // Price near lower band is bullish, near upper is bearish
            // We'd need current price for this - skip for now
        }

        Ok(SignalType::from_numeric(score))
    }

    /// Calculate confidence from IndicatorAnalysis
    fn calculate_analysis_confidence(&self, analysis: &IndicatorAnalysis) -> f64 {
        let mut confidence = 0.0;
        let mut count = 0;

        if analysis.ema_fast.is_some() {
            confidence += 0.2;
            count += 1;
        }

        if analysis.rsi.is_some() {
            confidence += 0.3;
            count += 1;
        }

        if analysis.macd_line.is_some() && analysis.macd_signal.is_some() {
            confidence += 0.3;
            count += 1;
        }

        if analysis.bb_upper.is_some() && analysis.bb_lower.is_some() {
            confidence += 0.2;
            count += 1;
        }

        if count > 0 {
            confidence * (count as f64 / 4.0)
        } else {
            0.5
        }
    }

    /// Generate a signal from technical indicators
    pub async fn generate_from_indicators(
        &self,
        symbol: String,
        timeframe: Timeframe,
        indicator_values: IndicatorValues,
    ) -> Result<Option<TradingSignal>> {
        debug!(
            "Generating signal from indicators for {} on {}",
            symbol,
            timeframe.as_str()
        );

        // Determine signal type based on indicator values
        let signal_type = self.analyze_indicators(&indicator_values)?;

        if signal_type == SignalType::Hold {
            return Ok(None);
        }

        // Calculate confidence based on indicator alignment
        let confidence = self.calculate_indicator_confidence(&indicator_values);

        // Create signal
        let signal = TradingSignal::new(
            symbol,
            signal_type,
            timeframe,
            confidence,
            SignalSource::TechnicalIndicator {
                name: "COMBINED".to_string(),
            },
        )
        .with_strength(indicator_values.strength);

        // Apply quality filter
        if self.config.enable_quality_filter
            && !signal.meets_threshold(self.config.min_confidence, self.config.min_strength)
        {
            debug!("Signal filtered out due to quality threshold");
            self.metrics
                .filtered_signals
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok(None);
        }

        self.metrics
            .generated_signals
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        info!(
            "Generated signal: {:?} for {}",
            signal.signal_type, signal.symbol
        );

        Ok(Some(signal))
    }

    /// Generate a batch of signals
    pub async fn generate_batch(&self, signals: Vec<TradingSignal>) -> Result<SignalBatch> {
        let mut batch = SignalBatch::new(signals);

        if self.config.enable_quality_filter {
            batch.filter_by_threshold(self.config.min_confidence, self.config.min_strength);
            batch.remove_stale(self.config.max_age_seconds);
        }

        info!("Generated batch with {} signals", batch.signals.len());
        Ok(batch)
    }

    /// Cache a signal for future retrieval
    pub async fn cache_signal(&self, signal: TradingSignal) -> Result<()> {
        let mut cache = self.signal_cache.write().await;
        cache.push(signal);

        // Limit cache size
        if cache.len() > self.config.batch_size * 2 {
            cache.drain(0..self.config.batch_size);
        }

        Ok(())
    }

    /// Get cached signals
    pub async fn get_cached_signals(&self) -> Vec<TradingSignal> {
        let cache = self.signal_cache.read().await;
        cache.clone()
    }

    /// Clear stale signals from cache
    pub async fn cleanup_cache(&self) -> Result<usize> {
        let mut cache = self.signal_cache.write().await;
        let initial_len = cache.len();
        cache.retain(|s| !s.is_stale(self.config.max_age_seconds));
        let removed = initial_len - cache.len();

        if removed > 0 {
            debug!("Removed {} stale signals from cache", removed);
        }

        Ok(removed)
    }

    /// Get signal generation metrics
    pub fn metrics(&self) -> &SignalMetrics {
        &self.metrics
    }

    /// Check if param reload is configured
    pub fn has_param_reload(&self) -> bool {
        self.param_query_handle.is_some()
    }

    /// Get ML inference metrics (if enabled)
    pub async fn ml_metrics(&self) -> Option<crate::inference::ModelMetrics> {
        if let Some(ref inference) = self.model_inference {
            Some(inference.metrics().await)
        } else {
            None
        }
    }

    // Private helper methods

    fn analyze_indicators(&self, indicators: &IndicatorValues) -> Result<SignalType> {
        // Simple logic for now - will be enhanced in Week 2
        let mut score = 0.0;

        // EMA crossover logic
        if let (Some(fast_ema), Some(slow_ema)) = (indicators.ema_fast, indicators.ema_slow) {
            if fast_ema > slow_ema {
                score += 0.3;
            } else {
                score -= 0.3;
            }
        }

        // RSI logic
        if let Some(rsi) = indicators.rsi {
            if rsi < 30.0 {
                score += 0.4; // Oversold
            } else if rsi > 70.0 {
                score -= 0.4; // Overbought
            }
        }

        // MACD logic
        if let Some(macd_histogram) = indicators.macd_histogram {
            if macd_histogram > 0.0 {
                score += 0.3;
            } else {
                score -= 0.3;
            }
        }

        Ok(SignalType::from_numeric(score))
    }

    fn calculate_indicator_confidence(&self, indicators: &IndicatorValues) -> f64 {
        let mut confidence = 0.0;
        let mut count = 0;

        if indicators.ema_fast.is_some() && indicators.ema_slow.is_some() {
            confidence += 0.3;
            count += 1;
        }

        if indicators.rsi.is_some() {
            confidence += 0.4;
            count += 1;
        }

        if indicators.macd_histogram.is_some() {
            confidence += 0.3;
            count += 1;
        }

        if count > 0 {
            confidence * (count as f64 / 3.0)
        } else {
            0.5 // Default confidence
        }
    }
}

/// Indicator values for signal generation
#[derive(Debug, Clone, Default)]
pub struct IndicatorValues {
    pub ema_fast: Option<f64>,
    pub ema_slow: Option<f64>,
    pub rsi: Option<f64>,
    pub macd_line: Option<f64>,
    pub macd_signal: Option<f64>,
    pub macd_histogram: Option<f64>,
    pub atr: Option<f64>,
    pub strength: f64,
}

impl IndicatorValues {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_ema(mut self, fast: f64, slow: f64) -> Self {
        self.ema_fast = Some(fast);
        self.ema_slow = Some(slow);
        self
    }

    pub fn with_rsi(mut self, rsi: f64) -> Self {
        self.rsi = Some(rsi);
        self
    }

    pub fn with_macd(mut self, macd_line: f64, signal: f64, histogram: f64) -> Self {
        self.macd_line = Some(macd_line);
        self.macd_signal = Some(signal);
        self.macd_histogram = Some(histogram);
        self
    }

    pub fn with_atr(mut self, atr: f64) -> Self {
        self.atr = Some(atr);
        self
    }

    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }
}

/// Signal generation metrics
#[derive(Debug, Default)]
pub struct SignalMetrics {
    pub generated_signals: std::sync::atomic::AtomicU64,
    pub filtered_signals: std::sync::atomic::AtomicU64,
    pub cached_signals: std::sync::atomic::AtomicU64,
}

impl SignalMetrics {
    pub fn total_generated(&self) -> u64 {
        self.generated_signals
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn total_filtered(&self) -> u64 {
        self.filtered_signals
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn filter_rate(&self) -> f64 {
        let generated = self.total_generated() as f64;
        let filtered = self.total_filtered() as f64;

        if generated > 0.0 {
            filtered / generated
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_signal_generator_creation() {
        let config = SignalGeneratorConfig::default();
        let generator = SignalGenerator::new(config);

        assert_eq!(generator.config.min_confidence, 0.6);
        assert_eq!(generator.config.min_strength, 0.5);
    }

    #[tokio::test]
    async fn test_generate_from_indicators_bullish() {
        // Use lower thresholds for testing
        let config = SignalGeneratorConfig {
            min_confidence: 0.3,
            min_strength: 0.3,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let indicators = IndicatorValues::new()
            .with_ema(51.0, 50.0) // Fast > Slow (bullish)
            .with_rsi(25.0) // Oversold (bullish)
            .with_strength(0.8);

        let result = generator
            .generate_from_indicators("BTC/USD".to_string(), Timeframe::H1, indicators)
            .await
            .unwrap();

        assert!(result.is_some());
        let signal = result.unwrap();
        assert!(signal.signal_type.is_bullish());
    }

    #[tokio::test]
    async fn test_generate_from_indicators_bearish() {
        // Use lower thresholds for testing
        let config = SignalGeneratorConfig {
            min_confidence: 0.3,
            min_strength: 0.3,
            ..Default::default()
        };
        let generator = SignalGenerator::new(config);

        let indicators = IndicatorValues::new()
            .with_ema(49.0, 50.0) // Fast < Slow (bearish)
            .with_rsi(75.0) // Overbought (bearish)
            .with_strength(0.7);

        let result = generator
            .generate_from_indicators("BTC/USD".to_string(), Timeframe::H1, indicators)
            .await
            .unwrap();

        assert!(result.is_some());
        let signal = result.unwrap();
        assert!(signal.signal_type.is_bearish());
    }

    #[tokio::test]
    async fn test_signal_filtering() {
        let config = SignalGeneratorConfig {
            min_confidence: 0.9, // Very high threshold
            ..Default::default()
        };

        let generator = SignalGenerator::new(config);

        let indicators = IndicatorValues::new()
            .with_ema(51.0, 50.0)
            .with_strength(0.5);

        let result = generator
            .generate_from_indicators("BTC/USD".to_string(), Timeframe::H1, indicators)
            .await
            .unwrap();

        // Should be filtered due to high confidence threshold
        assert!(result.is_none());
        assert!(generator.metrics().total_filtered() > 0);
    }

    #[tokio::test]
    async fn test_signal_caching() {
        let generator = SignalGenerator::new(SignalGeneratorConfig::default());

        let signal = TradingSignal::new(
            "BTC/USD".to_string(),
            SignalType::Buy,
            Timeframe::H1,
            0.8,
            SignalSource::TechnicalIndicator {
                name: "EMA".to_string(),
            },
        );

        generator.cache_signal(signal.clone()).await.unwrap();

        let cached = generator.get_cached_signals().await;
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].symbol, "BTC/USD");
    }

    #[tokio::test]
    async fn test_batch_generation() {
        let generator = SignalGenerator::new(SignalGeneratorConfig::default());

        let signals = vec![
            TradingSignal::new(
                "BTC/USD".to_string(),
                SignalType::Buy,
                Timeframe::H1,
                0.8,
                SignalSource::TechnicalIndicator {
                    name: "EMA".to_string(),
                },
            )
            .with_strength(0.7),
            TradingSignal::new(
                "ETH/USD".to_string(),
                SignalType::Sell,
                Timeframe::M15,
                0.5,
                SignalSource::TechnicalIndicator {
                    name: "RSI".to_string(),
                },
            )
            .with_strength(0.4), // Below threshold
        ];

        let batch = generator.generate_batch(signals).await.unwrap();

        // Only high-quality signal should pass
        assert_eq!(batch.signals.len(), 1);
        assert_eq!(batch.signals[0].symbol, "BTC/USD");
    }
}
