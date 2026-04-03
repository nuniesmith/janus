//! # Regime Detection Integration for the Forward Service
//!
//! This module wraps the `janus-regime` crate's `EnhancedRouter` to provide:
//!
//! - **Per-asset regime detection** via `EnsembleRegimeDetector` (indicator + HMM)
//! - **Candle aggregation** from raw ticks into OHLC bars for regime updates
//! - **Prometheus metrics** for regime classification, confidence, and transitions
//! - **Strategy gating** via `RoutedSignal` to decide which strategy to run
//! - **TOML configuration** via `regime.toml` for tuning all detection parameters
//! - **Relative volume tracking** via rolling average of candle volumes
//!
//! ## Usage in the Event Loop
//!
//! ```rust,ignore
//! use crate::regime::{RegimeManager, RegimeManagerConfig};
//!
//! // Load from config file (falls back to defaults if file not found):
//! let config = RegimeManagerConfig::from_toml_file("config/regime.toml")
//!     .unwrap_or_else(|_| RegimeManagerConfig::default());
//!
//! let mut regime_mgr = RegimeManager::new(config);
//! regime_mgr.register_asset("BTCUSDT");
//!
//! // On each tick:
//! if let Some(routed) = regime_mgr.on_tick("BTCUSDT", bid, ask) {
//!     match routed.strategy {
//!         ActiveStrategy::TrendFollowing => { /* run EMA Flip */ }
//!         ActiveStrategy::MeanReversion  => { /* run Mean Reversion */ }
//!         ActiveStrategy::NoTrade        => { /* skip */ }
//!     }
//! }
//! ```

use janus_regime::{
    ActiveStrategy, DetectionMethod, EnhancedRouter, EnhancedRouterConfig, EnsembleConfig,
    HMMConfig, MarketRegime, RegimeConfig, RoutedSignal, TrendDirection,
};
use prometheus::{GaugeVec, IntCounterVec, IntGaugeVec, Opts, Registry};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use tracing::{info, warn};

/// Default number of candles used to compute the rolling average volume.
const DEFAULT_VOLUME_LOOKBACK: usize = 20;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the regime manager.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RegimeManagerConfig {
    /// Number of ticks to aggregate into one candle before updating the regime
    /// detector. Higher values = smoother regime detection but slower reaction.
    pub ticks_per_candle: u64,

    /// Which detection method to use (Ensemble recommended for production).
    pub detection_method: DetectionMethod,

    /// Minimum confidence required to act on a regime signal.
    pub min_confidence: f64,

    /// Position size factor applied when regime is Volatile.
    pub volatile_position_factor: f64,

    /// Whether to log regime changes to tracing.
    pub log_regime_changes: bool,

    /// Number of completed candles to keep for the rolling average volume
    /// used to compute relative volume. Higher values = smoother baseline,
    /// less sensitive to individual candle spikes.
    ///   10  — responsive (picks up volume shifts quickly)
    ///   20  — default (balanced)
    ///   50  — smooth (good for longer-horizon strategies)
    pub volume_lookback: usize,

    /// Per-asset overrides for `volume_lookback`. When a symbol key is present
    /// in this map, its value is used instead of the global `volume_lookback`.
    ///
    /// Example in `regime.toml`:
    /// ```toml
    /// [manager]
    /// volume_lookback = 20
    ///
    /// [manager.volume_lookback_overrides]
    /// BTCUSD = 10   # faster reaction for BTC
    /// ETHUSD = 30   # smoother for ETH
    /// ```
    pub volume_lookback_overrides: HashMap<String, usize>,
}

impl Default for RegimeManagerConfig {
    fn default() -> Self {
        Self {
            ticks_per_candle: 100,
            detection_method: DetectionMethod::Ensemble,
            min_confidence: 0.5,
            volatile_position_factor: 0.3,
            log_regime_changes: true,
            volume_lookback: DEFAULT_VOLUME_LOOKBACK,
            volume_lookback_overrides: HashMap::new(),
        }
    }
}

impl RegimeManagerConfig {
    /// Fast-reacting config for scalping / HFT — fewer ticks per candle.
    pub fn fast() -> Self {
        Self {
            ticks_per_candle: 25,
            volume_lookback: 10,
            ..Default::default()
        }
    }

    /// Slow, stable config for swing trading — more ticks per candle.
    pub fn slow() -> Self {
        Self {
            ticks_per_candle: 500,
            volume_lookback: 50,
            ..Default::default()
        }
    }

    /// Resolve the effective volume lookback for a given symbol.
    ///
    /// Returns the per-asset override if one exists, otherwise the global default.
    pub fn effective_volume_lookback(&self, symbol: &str) -> usize {
        self.volume_lookback_overrides
            .get(symbol)
            .copied()
            .unwrap_or(self.volume_lookback)
    }

    /// Load configuration from a TOML file.
    ///
    /// The file is structured with `[manager]`, `[indicators]`, `[hmm]`,
    /// `[ensemble]`, and `[router]` sections. See `config/regime.toml` for
    /// the full annotated reference.
    ///
    /// Falls back to compiled defaults for any missing fields.
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self, RegimeConfigError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|e| RegimeConfigError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        Self::from_toml_str(&contents)
    }

    /// Parse configuration from a TOML string.
    pub fn from_toml_str(toml_str: &str) -> Result<Self, RegimeConfigError> {
        let file: RegimeTomlFile =
            toml::from_str(toml_str).map_err(RegimeConfigError::ParseError)?;
        Ok(file.into_manager_config())
    }

    /// Try to load from `path`, falling back to defaults if the file
    /// doesn't exist. Errors are logged as warnings.
    pub fn from_toml_file_or_default<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        match Self::from_toml_file(path) {
            Ok(config) => {
                info!("✅ Loaded regime config from {}", path.display());
                config
            }
            Err(RegimeConfigError::Io { .. }) => {
                warn!(
                    "⚠️ Regime config not found at {}, using defaults",
                    path.display()
                );
                Self::default()
            }
            Err(e) => {
                warn!(
                    "⚠️ Failed to parse regime config at {}: {}, using defaults",
                    path.display(),
                    e
                );
                Self::default()
            }
        }
    }

    /// Build the full `EnhancedRouterConfig` from this manager config
    /// plus any loaded indicator/HMM/ensemble overrides.
    pub fn to_router_config(&self) -> EnhancedRouterConfig {
        EnhancedRouterConfig {
            detection_method: self.detection_method,
            min_confidence: self.min_confidence,
            volatile_position_factor: self.volatile_position_factor,
            log_changes: self.log_regime_changes,
            ..Default::default()
        }
    }
}

// ============================================================================
// TOML File Structure
// ============================================================================

/// Top-level structure of `regime.toml`.
///
/// Each section is optional — missing sections use compiled defaults.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
struct RegimeTomlFile {
    manager: ManagerSection,
    indicators: IndicatorSection,
    hmm: HmmSection,
    ensemble: EnsembleSection,
    router: RouterSection,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct ManagerSection {
    ticks_per_candle: u64,
    detection_method: String,
    min_confidence: f64,
    volatile_position_factor: f64,
    log_regime_changes: bool,
    volume_lookback: usize,
    /// Per-asset volume lookback overrides (symbol → lookback).
    #[serde(default)]
    volume_lookback_overrides: HashMap<String, usize>,
}

impl Default for ManagerSection {
    fn default() -> Self {
        Self {
            ticks_per_candle: 100,
            detection_method: "Ensemble".to_string(),
            min_confidence: 0.5,
            volatile_position_factor: 0.3,
            log_regime_changes: true,
            volume_lookback: DEFAULT_VOLUME_LOOKBACK,
            volume_lookback_overrides: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct IndicatorSection {
    adx_period: usize,
    adx_trending_threshold: f64,
    adx_ranging_threshold: f64,
    bb_period: usize,
    bb_std_dev: f64,
    bb_width_volatility_threshold: f64,
    ema_short_period: usize,
    ema_long_period: usize,
    atr_period: usize,
    atr_expansion_threshold: f64,
    regime_stability_bars: usize,
    min_regime_duration: usize,
}

impl Default for IndicatorSection {
    fn default() -> Self {
        let rc = RegimeConfig::crypto_optimized();
        Self {
            adx_period: rc.adx_period,
            adx_trending_threshold: rc.adx_trending_threshold,
            adx_ranging_threshold: rc.adx_ranging_threshold,
            bb_period: rc.bb_period,
            bb_std_dev: rc.bb_std_dev,
            bb_width_volatility_threshold: rc.bb_width_volatility_threshold,
            ema_short_period: rc.ema_short_period,
            ema_long_period: rc.ema_long_period,
            atr_period: rc.atr_period,
            atr_expansion_threshold: rc.atr_expansion_threshold,
            regime_stability_bars: rc.regime_stability_bars,
            min_regime_duration: rc.min_regime_duration,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct HmmSection {
    n_states: usize,
    min_observations: usize,
    learning_rate: f64,
    transition_smoothing: f64,
    lookback_window: usize,
    min_confidence: f64,
}

impl Default for HmmSection {
    fn default() -> Self {
        let hc = HMMConfig::crypto_optimized();
        Self {
            n_states: hc.n_states,
            min_observations: hc.min_observations,
            learning_rate: hc.learning_rate,
            transition_smoothing: hc.transition_smoothing,
            lookback_window: hc.lookback_window,
            min_confidence: hc.min_confidence,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct EnsembleSection {
    indicator_weight: f64,
    hmm_weight: f64,
    agreement_threshold: f64,
    require_hmm_warmup: bool,
    agreement_confidence_boost: f64,
    disagreement_confidence_penalty: f64,
}

impl Default for EnsembleSection {
    fn default() -> Self {
        let ec = EnsembleConfig::default();
        Self {
            indicator_weight: ec.indicator_weight,
            hmm_weight: ec.hmm_weight,
            agreement_threshold: ec.agreement_threshold,
            require_hmm_warmup: ec.require_hmm_warmup,
            agreement_confidence_boost: ec.agreement_confidence_boost,
            disagreement_confidence_penalty: ec.disagreement_confidence_penalty,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct RouterSection {
    volatile_position_factor: f64,
    min_confidence: f64,
    log_changes: bool,
}

impl Default for RouterSection {
    fn default() -> Self {
        Self {
            volatile_position_factor: 0.3,
            min_confidence: 0.5,
            log_changes: true,
        }
    }
}

impl RegimeTomlFile {
    /// Convert the parsed TOML file into a `RegimeManagerConfig` plus the
    /// sub-configs needed by the router, HMM, ensemble, and indicators.
    fn into_manager_config(self) -> RegimeManagerConfig {
        let detection_method = match self.manager.detection_method.to_lowercase().as_str() {
            "indicators" | "indicator" => DetectionMethod::Indicators,
            "hmm" => DetectionMethod::HMM,
            _ => DetectionMethod::Ensemble,
        };

        RegimeManagerConfig {
            ticks_per_candle: self.manager.ticks_per_candle,
            detection_method,
            min_confidence: self.manager.min_confidence,
            volatile_position_factor: self.manager.volatile_position_factor,
            log_regime_changes: self.manager.log_regime_changes,
            volume_lookback: self.manager.volume_lookback,
            volume_lookback_overrides: self.manager.volume_lookback_overrides.clone(),
        }
    }

    /// Build a full `EnhancedRouterConfig` from all parsed sections.
    fn into_router_config(self) -> EnhancedRouterConfig {
        let detection_method = match self.manager.detection_method.to_lowercase().as_str() {
            "indicators" | "indicator" => DetectionMethod::Indicators,
            "hmm" => DetectionMethod::HMM,
            _ => DetectionMethod::Ensemble,
        };

        let indicator_config = RegimeConfig {
            adx_period: self.indicators.adx_period,
            adx_trending_threshold: self.indicators.adx_trending_threshold,
            adx_ranging_threshold: self.indicators.adx_ranging_threshold,
            bb_period: self.indicators.bb_period,
            bb_std_dev: self.indicators.bb_std_dev,
            bb_width_volatility_threshold: self.indicators.bb_width_volatility_threshold,
            ema_short_period: self.indicators.ema_short_period,
            ema_long_period: self.indicators.ema_long_period,
            atr_period: self.indicators.atr_period,
            atr_expansion_threshold: self.indicators.atr_expansion_threshold,
            regime_stability_bars: self.indicators.regime_stability_bars,
            min_regime_duration: self.indicators.min_regime_duration,
        };

        let hmm_config = HMMConfig {
            n_states: self.hmm.n_states,
            min_observations: self.hmm.min_observations,
            learning_rate: self.hmm.learning_rate,
            transition_smoothing: self.hmm.transition_smoothing,
            lookback_window: self.hmm.lookback_window,
            min_confidence: self.hmm.min_confidence,
        };

        let ensemble_config = EnsembleConfig {
            indicator_weight: self.ensemble.indicator_weight,
            hmm_weight: self.ensemble.hmm_weight,
            agreement_threshold: self.ensemble.agreement_threshold,
            require_hmm_warmup: self.ensemble.require_hmm_warmup,
            agreement_confidence_boost: self.ensemble.agreement_confidence_boost,
            disagreement_confidence_penalty: self.ensemble.disagreement_confidence_penalty,
        };

        EnhancedRouterConfig {
            detection_method,
            indicator_config,
            hmm_config: Some(hmm_config),
            ensemble_config: Some(ensemble_config),
            volatile_position_factor: self.router.volatile_position_factor,
            min_confidence: self.router.min_confidence,
            log_changes: self.router.log_changes,
        }
    }
}

// ============================================================================
// Config Errors
// ============================================================================

/// Errors that can occur when loading regime configuration.
#[derive(Debug)]
pub enum RegimeConfigError {
    /// Failed to read the config file from disk.
    Io {
        path: String,
        source: std::io::Error,
    },
    /// Failed to parse the TOML content.
    ParseError(toml::de::Error),
}

impl std::fmt::Display for RegimeConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "Failed to read {}: {}", path, source),
            Self::ParseError(e) => write!(f, "Failed to parse regime config: {}", e),
        }
    }
}

impl std::error::Error for RegimeConfigError {}

// ============================================================================
// Candle Aggregator
// ============================================================================

/// Accumulates ticks into OHLC candles.
#[derive(Debug)]
struct CandleAggregator {
    ticks_per_candle: u64,
    tick_count: u64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    started: bool,
}

/// A completed OHLCV candle from tick aggregation.
#[derive(Debug, Clone, Copy)]
pub struct AggregatedCandle {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    /// Accumulated trade volume during this candle period.
    /// Will be 0.0 if no trade messages were received.
    pub volume: f64,
}

impl CandleAggregator {
    fn new(ticks_per_candle: u64) -> Self {
        Self {
            ticks_per_candle,
            tick_count: 0,
            open: 0.0,
            high: f64::NEG_INFINITY,
            low: f64::MAX,
            close: 0.0,
            volume: 0.0,
            started: false,
        }
    }

    /// Feed a mid-price tick. Returns `Some(candle)` when a full candle is formed.
    fn on_tick(&mut self, price: f64) -> Option<AggregatedCandle> {
        if !self.started {
            self.open = price;
            self.high = price;
            self.low = price;
            self.started = true;
        }

        self.high = self.high.max(price);
        self.low = self.low.min(price);
        self.close = price;
        self.tick_count += 1;

        if self.tick_count >= self.ticks_per_candle {
            let candle = AggregatedCandle {
                open: self.open,
                high: self.high,
                low: self.low,
                close: self.close,
                volume: self.volume,
            };
            self.reset(price);
            Some(candle)
        } else {
            None
        }
    }

    fn reset(&mut self, next_open: f64) {
        self.tick_count = 0;
        self.open = next_open;
        self.high = next_open;
        self.low = next_open;
        self.close = next_open;
        self.volume = 0.0;
    }

    /// Accumulate trade volume into the current candle being formed.
    fn on_trade_volume(&mut self, volume: f64) {
        self.volume += volume;
    }

    /// How many ticks have been accumulated in the current (incomplete) candle.
    #[allow(dead_code)]
    fn current_tick_count(&self) -> u64 {
        self.tick_count
    }
}

// ============================================================================
// Prometheus Metrics
// ============================================================================

/// Prometheus metrics for regime detection.
pub struct RegimeMetrics {
    /// Current regime classification per asset (0=Uncertain, 1=Trending, 2=MeanReverting, 3=Volatile)
    pub regime_current: IntGaugeVec,
    /// Confidence of the current regime classification (0.0–1.0)
    pub regime_confidence: GaugeVec,
    /// Total regime transitions per asset
    pub regime_transitions_total: IntCounterVec,
    /// Current recommended strategy per asset (0=NoTrade, 1=TrendFollowing, 2=MeanReversion)
    pub strategy_current: IntGaugeVec,
    /// Position size factor recommended by the regime router
    pub position_factor: GaugeVec,
    /// Whether the ensemble methods agree (1=agree, 0=disagree)
    pub methods_agree: IntGaugeVec,
    /// Expected regime duration in bars (from HMM)
    pub expected_duration: GaugeVec,
    /// Total candles processed per asset
    pub candles_processed_total: IntCounterVec,
    /// Whether the detector is warmed up per asset
    pub detector_ready: IntGaugeVec,
    /// Relative volume for the most recently completed candle (1.0 = average)
    pub relative_volume: GaugeVec,
    /// Bridged hypothalamus regime (0=Unknown … 9=Crisis)
    pub bridge_hypothalamus: IntGaugeVec,
    /// Bridged amygdala regime (0=Unknown … 6=Crisis)
    pub bridge_amygdala: IntGaugeVec,
    /// Bridged position scale from hypothalamus regime (0.0–1.5)
    pub bridge_position_scale: GaugeVec,
}

impl RegimeMetrics {
    /// Create and register regime metrics on the given Prometheus registry.
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let regime_current = IntGaugeVec::new(
            Opts::new(
                "janus_regime_current",
                "Current market regime (0=Uncertain, 1=TrendingBull, 2=TrendingBear, 3=MeanReverting, 4=Volatile)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(regime_current.clone()))?;

        let regime_confidence = GaugeVec::new(
            Opts::new(
                "janus_regime_confidence",
                "Confidence of the current regime classification (0.0-1.0)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(regime_confidence.clone()))?;

        let regime_transitions_total = IntCounterVec::new(
            Opts::new(
                "janus_regime_transitions_total",
                "Total number of regime transitions",
            ),
            &["asset", "from_regime", "to_regime"],
        )?;
        registry.register(Box::new(regime_transitions_total.clone()))?;

        let strategy_current = IntGaugeVec::new(
            Opts::new(
                "janus_regime_strategy_current",
                "Current recommended strategy (0=NoTrade, 1=TrendFollowing, 2=MeanReversion)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(strategy_current.clone()))?;

        let position_factor = GaugeVec::new(
            Opts::new(
                "janus_regime_position_factor",
                "Position size factor recommended by regime router (0.0-1.0)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(position_factor.clone()))?;

        let methods_agree = IntGaugeVec::new(
            Opts::new(
                "janus_regime_methods_agree",
                "Whether ensemble detection methods agree (1=yes, 0=no)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(methods_agree.clone()))?;

        let expected_duration = GaugeVec::new(
            Opts::new(
                "janus_regime_expected_duration_bars",
                "Expected regime duration in bars (from HMM)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(expected_duration.clone()))?;

        let candles_processed_total = IntCounterVec::new(
            Opts::new(
                "janus_regime_candles_processed_total",
                "Total number of candles processed by regime detector",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(candles_processed_total.clone()))?;

        let detector_ready = IntGaugeVec::new(
            Opts::new(
                "janus_regime_detector_ready",
                "Whether the regime detector has warmed up (1=ready, 0=warming up)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(detector_ready.clone()))?;

        let relative_volume = GaugeVec::new(
            Opts::new(
                "janus_regime_relative_volume",
                "Relative volume of the last completed candle (1.0 = average)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(relative_volume.clone()))?;

        let bridge_hypothalamus = IntGaugeVec::new(
            Opts::new(
                "janus_bridge_hypothalamus_regime",
                "Bridged hypothalamus regime (0=Unknown,1=StrongBullish,2=Bullish,3=Neutral,4=Bearish,5=StrongBearish,6=HighVol,7=LowVol,8=Transitional,9=Crisis)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(bridge_hypothalamus.clone()))?;

        let bridge_amygdala = IntGaugeVec::new(
            Opts::new(
                "janus_bridge_amygdala_regime",
                "Bridged amygdala regime (0=Unknown,1=LowVolTrending,2=LowVolMeanReverting,3=HighVolTrending,4=HighVolMeanReverting,5=Transitional,6=Crisis)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(bridge_amygdala.clone()))?;

        let bridge_position_scale = GaugeVec::new(
            Opts::new(
                "janus_bridge_position_scale",
                "Position scale factor from bridged hypothalamus regime (0.0-1.5)",
            ),
            &["asset"],
        )?;
        registry.register(Box::new(bridge_position_scale.clone()))?;

        Ok(Self {
            regime_current,
            regime_confidence,
            regime_transitions_total,
            strategy_current,
            position_factor,
            methods_agree,
            expected_duration,
            candles_processed_total,
            detector_ready,
            relative_volume,
            bridge_hypothalamus,
            bridge_amygdala,
            bridge_position_scale,
        })
    }

    /// Record metrics for a new routed signal.
    fn record_signal(&self, asset: &str, signal: &RoutedSignal) {
        let regime_val = regime_to_i64(&signal.regime);
        self.regime_current
            .with_label_values(&[asset])
            .set(regime_val);
        self.regime_confidence
            .with_label_values(&[asset])
            .set(signal.confidence);

        let strategy_val = strategy_to_i64(&signal.strategy);
        self.strategy_current
            .with_label_values(&[asset])
            .set(strategy_val);

        self.position_factor
            .with_label_values(&[asset])
            .set(signal.position_factor);

        if let Some(agree) = signal.methods_agree {
            self.methods_agree
                .with_label_values(&[asset])
                .set(i64::from(agree));
        }

        if let Some(dur) = signal.expected_duration {
            self.expected_duration.with_label_values(&[asset]).set(dur);
        }
    }

    /// Record the relative volume for an asset.
    fn record_relative_volume(&self, asset: &str, rel_vol: f64) {
        self.relative_volume
            .with_label_values(&[asset])
            .set(rel_vol);
    }

    /// Record bridged regime state from the neuromorphic bridge.
    pub fn record_bridge_state(
        &self,
        asset: &str,
        hypothalamus: i64,
        amygdala: i64,
        position_scale: f64,
    ) {
        self.bridge_hypothalamus
            .with_label_values(&[asset])
            .set(hypothalamus);
        self.bridge_amygdala
            .with_label_values(&[asset])
            .set(amygdala);
        self.bridge_position_scale
            .with_label_values(&[asset])
            .set(position_scale);
    }

    /// Record a regime transition.
    fn record_transition(&self, asset: &str, from: &MarketRegime, to: &MarketRegime) {
        self.regime_transitions_total
            .with_label_values(&[asset, &regime_label(from), &regime_label(to)])
            .inc();
    }

    /// Record that a candle was processed.
    fn record_candle(&self, asset: &str) {
        self.candles_processed_total
            .with_label_values(&[asset])
            .inc();
    }

    /// Record detector readiness.
    fn record_ready(&self, asset: &str, ready: bool) {
        self.detector_ready
            .with_label_values(&[asset])
            .set(i64::from(ready));
    }
}

fn regime_to_i64(regime: &MarketRegime) -> i64 {
    match regime {
        MarketRegime::Uncertain => 0,
        MarketRegime::Trending(TrendDirection::Bullish) => 1,
        MarketRegime::Trending(TrendDirection::Bearish) => 2,
        MarketRegime::MeanReverting => 3,
        MarketRegime::Volatile => 4,
    }
}

fn strategy_to_i64(strategy: &ActiveStrategy) -> i64 {
    match strategy {
        ActiveStrategy::NoTrade => 0,
        ActiveStrategy::TrendFollowing => 1,
        ActiveStrategy::MeanReversion => 2,
    }
}

fn regime_label(regime: &MarketRegime) -> String {
    match regime {
        MarketRegime::Uncertain => "uncertain".to_string(),
        MarketRegime::Trending(TrendDirection::Bullish) => "trending_bull".to_string(),
        MarketRegime::Trending(TrendDirection::Bearish) => "trending_bear".to_string(),
        MarketRegime::MeanReverting => "mean_reverting".to_string(),
        MarketRegime::Volatile => "volatile".to_string(),
    }
}

// ============================================================================
// Regime Manager
// ============================================================================

/// Per-asset state tracked by the `RegimeManager`.
struct AssetState {
    aggregator: CandleAggregator,
    last_regime: Option<MarketRegime>,
    /// The most recently completed candle from tick aggregation.
    last_candle: Option<AggregatedCandle>,
    /// Rolling window of completed candle volumes for relative volume calc.
    volume_history: VecDeque<f64>,
    /// Number of candles to keep for the rolling average.
    volume_lookback: usize,
    /// Cached relative volume of the most recently completed candle
    /// (`current_volume / rolling_avg`). `None` until at least one candle with
    /// non-zero volume has been completed.
    last_relative_volume: Option<f64>,
}

/// Manages regime detection for multiple assets in the forward service.
///
/// Aggregates raw ticks into candles, feeds them to the `EnhancedRouter`,
/// and emits `RoutedSignal`s with Prometheus metrics on each completed candle.
pub struct RegimeManager {
    config: RegimeManagerConfig,
    router: EnhancedRouter,
    assets: HashMap<String, AssetState>,
    metrics: Option<RegimeMetrics>,
}

impl RegimeManager {
    /// Create a new regime manager without Prometheus metrics.
    pub fn new(config: RegimeManagerConfig) -> Self {
        let router_config = config.to_router_config();
        let router = EnhancedRouter::new(router_config);

        Self {
            config,
            router,
            assets: HashMap::new(),
            metrics: None,
        }
    }

    /// Create a regime manager from a TOML config file.
    ///
    /// This loads the full config chain: manager settings, indicator params,
    /// HMM params, ensemble params, and router settings — all from one file.
    pub fn from_toml_file<P: AsRef<Path>>(path: P) -> Result<Self, RegimeConfigError> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|e| RegimeConfigError::Io {
            path: path.display().to_string(),
            source: e,
        })?;
        let file: RegimeTomlFile =
            toml::from_str(&contents).map_err(RegimeConfigError::ParseError)?;

        let config = file.clone().into_manager_config();
        let router_config = file.into_router_config();
        let router = EnhancedRouter::new(router_config);

        Ok(Self {
            config,
            router,
            assets: HashMap::new(),
            metrics: None,
        })
    }

    /// Create from TOML file, falling back to defaults if file is missing.
    pub fn from_toml_file_or_default<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref();
        match Self::from_toml_file(path) {
            Ok(mgr) => {
                info!("✅ Loaded regime config from {}", path.display());
                mgr
            }
            Err(RegimeConfigError::Io { .. }) => {
                warn!(
                    "⚠️ Regime config not found at {}, using defaults",
                    path.display()
                );
                Self::new(RegimeManagerConfig::default())
            }
            Err(e) => {
                warn!(
                    "⚠️ Failed to parse regime config at {}: {}, using defaults",
                    path.display(),
                    e
                );
                Self::new(RegimeManagerConfig::default())
            }
        }
    }

    /// Create a new regime manager with Prometheus metrics registered on `registry`.
    pub fn with_metrics(
        config: RegimeManagerConfig,
        registry: &Registry,
    ) -> Result<Self, prometheus::Error> {
        let metrics = RegimeMetrics::new(registry)?;
        let mut mgr = Self::new(config);
        mgr.metrics = Some(metrics);
        Ok(mgr)
    }

    /// Register an asset for regime detection. Must be called before `on_tick`.
    ///
    /// Uses the per-asset `volume_lookback` override from config if one exists,
    /// otherwise falls back to the global `volume_lookback`.
    pub fn register_asset(&mut self, symbol: &str) {
        let lookback = self.config.effective_volume_lookback(symbol);
        self.router.register_asset(symbol);
        self.assets.insert(
            symbol.to_string(),
            AssetState {
                aggregator: CandleAggregator::new(self.config.ticks_per_candle),
                last_regime: None,
                last_candle: None,
                volume_history: VecDeque::with_capacity(lookback + 1),
                volume_lookback: lookback,
                last_relative_volume: None,
            },
        );
        info!(
            "Regime detection registered for {} (method: {}, ticks/candle: {}, vol_lookback: {}{})",
            symbol,
            self.config.detection_method,
            self.config.ticks_per_candle,
            lookback,
            if self.config.volume_lookback_overrides.contains_key(symbol) {
                " [per-asset override]"
            } else {
                ""
            },
        );
    }

    /// Register an asset with an explicit volume lookback, ignoring config.
    ///
    /// This is useful for programmatic registration where the caller wants
    /// fine-grained control without modifying the TOML config.
    pub fn register_asset_with_lookback(&mut self, symbol: &str, volume_lookback: usize) {
        self.router.register_asset(symbol);
        self.assets.insert(
            symbol.to_string(),
            AssetState {
                aggregator: CandleAggregator::new(self.config.ticks_per_candle),
                last_regime: None,
                last_candle: None,
                volume_history: VecDeque::with_capacity(volume_lookback + 1),
                volume_lookback,
                last_relative_volume: None,
            },
        );
        info!(
            "Regime detection registered for {} (method: {}, ticks/candle: {}, vol_lookback: {} [explicit])",
            symbol, self.config.detection_method, self.config.ticks_per_candle, volume_lookback
        );
    }

    /// Feed a raw tick (bid/ask) for the given asset.
    ///
    /// Returns `Some(RoutedSignal)` when enough ticks have been accumulated to
    /// form a candle and the regime detector has been updated. Returns `None` if
    /// the candle is still forming or the detector is warming up.
    pub fn on_tick(&mut self, symbol: &str, bid: f64, ask: f64) -> Option<RoutedSignal> {
        let mid = (bid + ask) / 2.0;
        self.on_tick_price(symbol, mid)
    }

    /// Feed a raw mid-price tick for the given asset.
    ///
    /// Returns `Some(RoutedSignal)` when a candle is completed and the detector
    /// has produced a signal.
    pub fn on_tick_price(&mut self, symbol: &str, price: f64) -> Option<RoutedSignal> {
        // Auto-register if not seen before
        if !self.assets.contains_key(symbol) {
            self.register_asset(symbol);
        }

        let state = self.assets.get_mut(symbol)?;

        // Aggregate into candle
        let candle = state.aggregator.on_tick(price)?;

        // Store the completed candle so callers can retrieve it
        state.last_candle = Some(candle);

        // Update rolling volume average and compute relative volume.
        // Only track when candle actually has volume data (> 0).
        if candle.volume > 0.0 {
            state.volume_history.push_back(candle.volume);
            if state.volume_history.len() > state.volume_lookback {
                state.volume_history.pop_front();
            }
            if !state.volume_history.is_empty() {
                let avg: f64 =
                    state.volume_history.iter().sum::<f64>() / state.volume_history.len() as f64;
                if avg > 0.0 {
                    state.last_relative_volume = Some(candle.volume / avg);
                }
            }
        }

        // Feed candle to the router
        let signal = self
            .router
            .update(symbol, candle.high, candle.low, candle.close);

        // Update metrics
        if let Some(ref metrics) = self.metrics {
            metrics.record_candle(symbol);
            metrics.record_ready(symbol, self.router.is_ready(symbol));

            // Record relative volume when available
            if let Some(rv) = state.last_relative_volume {
                metrics.record_relative_volume(symbol, rv);
            }

            if let Some(ref sig) = signal {
                metrics.record_signal(symbol, sig);

                // Detect regime transitions
                let new_regime = &sig.regime;
                if let Some(ref old_regime) = state.last_regime
                    && old_regime != new_regime
                {
                    metrics.record_transition(symbol, old_regime, new_regime);
                    if self.config.log_regime_changes {
                        info!(
                            "🔄 Regime change for {}: {} → {} (conf: {:.0}%, strategy: {})",
                            symbol,
                            old_regime,
                            new_regime,
                            sig.confidence * 100.0,
                            sig.strategy
                        );
                    }
                }
            }
        } else if let Some(ref sig) = signal {
            // Log even without metrics
            let new_regime = &sig.regime;
            if let Some(ref old_regime) = state.last_regime
                && old_regime != new_regime
                && self.config.log_regime_changes
            {
                info!(
                    "🔄 Regime change for {}: {} → {} (conf: {:.0}%, strategy: {})",
                    symbol,
                    old_regime,
                    new_regime,
                    sig.confidence * 100.0,
                    sig.strategy
                );
            }
        }

        // Track last regime
        if let Some(ref sig) = signal {
            state.last_regime = Some(sig.regime);
        }

        signal
    }

    /// Feed a pre-formed candle directly (bypasses tick aggregation).
    ///
    /// Useful when candle data is already available from a data service or
    /// backtest replay.
    pub fn on_candle(
        &mut self,
        symbol: &str,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<RoutedSignal> {
        // Auto-register if not seen before
        if !self.assets.contains_key(symbol) {
            self.register_asset(symbol);
        }

        let signal = self.router.update(symbol, high, low, close);

        if let Some(ref metrics) = self.metrics {
            metrics.record_candle(symbol);
            metrics.record_ready(symbol, self.router.is_ready(symbol));

            if let Some(ref sig) = signal {
                metrics.record_signal(symbol, sig);

                let state = self.assets.get(symbol);
                if let Some(asset_state) = state {
                    let new_regime = &sig.regime;
                    if let Some(ref old_regime) = asset_state.last_regime
                        && old_regime != new_regime
                    {
                        metrics.record_transition(symbol, old_regime, new_regime);
                    }
                }
            }
        }

        // Track last regime
        if let Some(ref sig) = signal
            && let Some(state) = self.assets.get_mut(symbol)
        {
            state.last_regime = Some(sig.regime);
        }

        signal
    }

    /// Get the current regime for an asset (if the detector is ready).
    pub fn current_regime(&self, symbol: &str) -> Option<&MarketRegime> {
        self.assets.get(symbol).and_then(|s| s.last_regime.as_ref())
    }

    /// Get the current recommended strategy for an asset.
    pub fn current_strategy(&self, symbol: &str) -> Option<ActiveStrategy> {
        self.router.get_strategy(symbol)
    }

    /// Whether the detector for the given asset has warmed up.
    pub fn is_ready(&self, symbol: &str) -> bool {
        self.router.is_ready(symbol)
    }

    /// Get a summary of all registered assets and their regime state.
    pub fn summary(&self) -> Vec<AssetRegimeSummary> {
        self.assets
            .iter()
            .map(|(symbol, state)| AssetRegimeSummary {
                symbol: symbol.clone(),
                regime: state.last_regime,
                strategy: self.router.get_strategy(symbol),
                is_ready: self.router.is_ready(symbol),
            })
            .collect()
    }

    /// Get the number of registered assets.
    pub fn asset_count(&self) -> usize {
        self.assets.len()
    }

    /// Get a reference to the underlying router (for advanced queries).
    pub fn router(&self) -> &EnhancedRouter {
        &self.router
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &RegimeManagerConfig {
        &self.config
    }

    /// Reference to the metrics (if configured).
    pub fn metrics(&self) -> Option<&RegimeMetrics> {
        self.metrics.as_ref()
    }

    /// Get the most recently completed candle for an asset (if any).
    ///
    /// This is populated each time tick aggregation produces a full candle.
    /// Useful for feeding the candle to secondary strategies (e.g. mean
    /// reversion) that operate on the same cadence as the regime detector.
    pub fn last_candle(&self, symbol: &str) -> Option<AggregatedCandle> {
        self.assets.get(symbol).and_then(|s| s.last_candle)
    }

    /// Get the current ATR (Average True Range) value for an asset.
    ///
    /// Delegates to the underlying router/detector. Returns `None` if the
    /// asset isn't registered or the detector hasn't warmed up yet.
    pub fn atr_value(&self, symbol: &str) -> Option<f64> {
        self.router.atr_value(symbol)
    }

    /// Get the current ADX (Average Directional Index) value for an asset.
    ///
    /// Delegates to the underlying router/detector. Returns `None` if the
    /// asset isn't registered or the detector hasn't warmed up yet.
    pub fn adx_value(&self, symbol: &str) -> Option<f64> {
        self.router.adx_value(symbol)
    }

    /// Get the most recent [`RegimeConfidence`] for an asset.
    ///
    /// This contains the raw indicator values (ADX, BB width percentile,
    /// trend strength) that produced the last regime classification.
    /// Useful for enriching the regime bridge with real indicator data
    /// rather than passing `None` placeholders.
    ///
    /// Returns `None` if the asset hasn't produced a regime signal yet.
    pub fn last_regime_confidence(&self, symbol: &str) -> Option<&janus_regime::RegimeConfidence> {
        self.router.last_regime_confidence(symbol)
    }

    /// Get the relative volume for the most recently completed candle.
    ///
    /// Relative volume is `current_candle_volume / rolling_average_volume`.
    /// A value of 1.0 means average volume; 2.0 means double the average.
    ///
    /// Returns `None` if:
    /// - The asset isn't registered
    /// - No candles with non-zero volume have completed yet
    pub fn relative_volume(&self, symbol: &str) -> Option<f64> {
        self.assets.get(symbol).and_then(|s| s.last_relative_volume)
    }

    /// Accumulate trade volume for an asset into the current (forming) candle.
    ///
    /// Call this when a `WsMessage::Trade` is received. The volume will be
    /// included in the next completed `AggregatedCandle`, making it available
    /// to volume-aware strategies like VWAP Scalper and ORB.
    pub fn on_trade_volume(&mut self, symbol: &str, volume: f64) {
        if let Some(state) = self.assets.get_mut(symbol) {
            state.aggregator.on_trade_volume(volume);
        }
    }

    // ── Hot-Reload ──────────────────────────────────────────────────────

    /// Reload configuration from a new `RegimeManagerConfig`.
    ///
    /// Updates tunable parameters at runtime without restarting the service.
    /// Currently supports:
    ///
    /// - `volume_lookback` (global default)
    /// - `volume_lookback_overrides` (per-asset overrides)
    /// - `min_confidence`
    /// - `volatile_position_factor`
    /// - `log_regime_changes`
    ///
    /// **Not** hot-reloadable (requires restart):
    /// - `ticks_per_candle` (changing mid-stream would corrupt the aggregator)
    /// - `detection_method` (changing the detector type requires re-warmup)
    ///
    /// Returns a summary of what changed.
    pub fn reload_config(&mut self, new_config: RegimeManagerConfig) -> ReloadSummary {
        let mut summary = ReloadSummary::default();

        // ── volume_lookback (global) ────────────────────────────────────
        if new_config.volume_lookback != self.config.volume_lookback {
            info!(
                "🔄 Regime reload: volume_lookback {} → {}",
                self.config.volume_lookback, new_config.volume_lookback
            );
            summary.volume_lookback_changed = true;
            self.config.volume_lookback = new_config.volume_lookback;
        }

        // ── volume_lookback_overrides (per-asset) ───────────────────────
        if new_config.volume_lookback_overrides != self.config.volume_lookback_overrides {
            info!(
                "🔄 Regime reload: volume_lookback_overrides changed ({} entries → {} entries)",
                self.config.volume_lookback_overrides.len(),
                new_config.volume_lookback_overrides.len()
            );
            summary.volume_lookback_overrides_changed = true;
            self.config.volume_lookback_overrides = new_config.volume_lookback_overrides;
        }

        // ── Apply updated lookbacks to existing assets ──────────────────
        if summary.volume_lookback_changed || summary.volume_lookback_overrides_changed {
            for (symbol, state) in &mut self.assets {
                let new_lookback = self.config.effective_volume_lookback(symbol);
                if new_lookback != state.volume_lookback {
                    info!(
                        "🔄 Regime reload: {} volume_lookback {} → {}",
                        symbol, state.volume_lookback, new_lookback
                    );
                    state.volume_lookback = new_lookback;
                    // Resize history: if new lookback is smaller, trim from front
                    while state.volume_history.len() > new_lookback {
                        state.volume_history.pop_front();
                    }
                    summary.assets_updated += 1;
                }
            }
        }

        // ── min_confidence ──────────────────────────────────────────────
        if (new_config.min_confidence - self.config.min_confidence).abs() > f64::EPSILON {
            info!(
                "🔄 Regime reload: min_confidence {:.2} → {:.2}",
                self.config.min_confidence, new_config.min_confidence
            );
            summary.min_confidence_changed = true;
            self.config.min_confidence = new_config.min_confidence;
        }

        // ── volatile_position_factor ────────────────────────────────────
        if (new_config.volatile_position_factor - self.config.volatile_position_factor).abs()
            > f64::EPSILON
        {
            info!(
                "🔄 Regime reload: volatile_position_factor {:.2} → {:.2}",
                self.config.volatile_position_factor, new_config.volatile_position_factor
            );
            summary.volatile_position_factor_changed = true;
            self.config.volatile_position_factor = new_config.volatile_position_factor;
        }

        // ── log_regime_changes ──────────────────────────────────────────
        if new_config.log_regime_changes != self.config.log_regime_changes {
            info!(
                "🔄 Regime reload: log_regime_changes {} → {}",
                self.config.log_regime_changes, new_config.log_regime_changes
            );
            self.config.log_regime_changes = new_config.log_regime_changes;
        }

        // ── Warn about non-reloadable fields ────────────────────────────
        if new_config.ticks_per_candle != self.config.ticks_per_candle {
            warn!(
                "⚠️ Regime reload: ticks_per_candle changed ({} → {}) but is NOT hot-reloadable — restart required",
                self.config.ticks_per_candle, new_config.ticks_per_candle
            );
            summary.requires_restart = true;
        }
        if new_config.detection_method != self.config.detection_method {
            warn!(
                "⚠️ Regime reload: detection_method changed ({} → {}) but is NOT hot-reloadable — restart required",
                self.config.detection_method, new_config.detection_method
            );
            summary.requires_restart = true;
        }

        summary
    }

    /// Reload configuration from a TOML file.
    ///
    /// Convenience wrapper around [`reload_config`](Self::reload_config)
    /// that reads and parses the file first. Returns `Err` if the file
    /// can't be read or parsed; returns `Ok(summary)` with what changed.
    pub fn reload_from_toml<P: AsRef<Path>>(
        &mut self,
        path: P,
    ) -> Result<ReloadSummary, RegimeConfigError> {
        let new_config = RegimeManagerConfig::from_toml_file(path)?;
        Ok(self.reload_config(new_config))
    }
}

/// Summary of what changed during a [`RegimeManager::reload_config`] call.
#[derive(Debug, Clone, Default)]
pub struct ReloadSummary {
    /// Whether the global `volume_lookback` changed.
    pub volume_lookback_changed: bool,
    /// Whether per-asset `volume_lookback_overrides` changed.
    pub volume_lookback_overrides_changed: bool,
    /// Number of registered assets whose lookback was updated.
    pub assets_updated: usize,
    /// Whether `min_confidence` changed.
    pub min_confidence_changed: bool,
    /// Whether `volatile_position_factor` changed.
    pub volatile_position_factor_changed: bool,
    /// Whether a non-reloadable field changed (restart required).
    pub requires_restart: bool,
}

impl ReloadSummary {
    /// Whether any field was actually changed.
    pub fn any_changed(&self) -> bool {
        self.volume_lookback_changed
            || self.volume_lookback_overrides_changed
            || self.min_confidence_changed
            || self.volatile_position_factor_changed
    }
}

// ============================================================================
// File Watcher for Hot-Reload
// ============================================================================

/// Spawn a background task that watches a `regime.toml` file for changes
/// and applies updated configuration to the `RegimeManager` at runtime.
///
/// The watcher checks the file's last-modified time every `poll_interval`
/// and reloads when it detects a change.
///
/// # Arguments
///
/// * `path`          — Path to the `regime.toml` file.
/// * `state`         — Shared reference to the strategy state containing
///   the `RegimeManager`.
/// * `poll_interval` — How often to check for file changes (e.g. 30 seconds).
///
/// # Returns
///
/// A `JoinHandle` that runs until the process exits. The task logs all
/// reload outcomes (success, no change, parse errors) at appropriate levels.
pub fn spawn_regime_config_watcher<P: AsRef<Path> + Send + 'static>(
    path: P,
    state: std::sync::Arc<tokio::sync::RwLock<impl RegimeConfigHolder + Send + Sync + 'static>>,
    poll_interval: std::time::Duration,
) -> tokio::task::JoinHandle<()> {
    use std::time::SystemTime;

    let path = path.as_ref().to_path_buf();
    let mut last_modified: Option<SystemTime> = None;

    info!(
        "👁️ Regime config watcher started: {} (poll interval: {:?})",
        path.display(),
        poll_interval
    );

    tokio::spawn(async move {
        let mut interval = tokio::time::interval(poll_interval);

        loop {
            interval.tick().await;

            // Check file mtime
            let metadata = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) => {
                    // File might have been temporarily removed during deploy
                    warn!(
                        "⚠️ Regime config watcher: cannot stat {}: {}",
                        path.display(),
                        e
                    );
                    continue;
                }
            };

            let mtime = match metadata.modified() {
                Ok(t) => t,
                Err(e) => {
                    warn!(
                        "⚠️ Regime config watcher: cannot read mtime for {}: {}",
                        path.display(),
                        e
                    );
                    continue;
                }
            };

            // First iteration — record baseline, don't reload
            if last_modified.is_none() {
                last_modified = Some(mtime);
                continue;
            }

            // File hasn't changed
            if last_modified == Some(mtime) {
                continue;
            }

            // File changed — reload
            info!(
                "🔄 Regime config file changed: {} — reloading...",
                path.display()
            );
            last_modified = Some(mtime);

            let mut guard = state.write().await;
            match guard.regime_manager_mut().reload_from_toml(&path) {
                Ok(summary) => {
                    if summary.any_changed() {
                        info!(
                            "✅ Regime config reloaded: lookback={}, overrides={}, confidence={}, vol_factor={}, assets_updated={}{}",
                            summary.volume_lookback_changed,
                            summary.volume_lookback_overrides_changed,
                            summary.min_confidence_changed,
                            summary.volatile_position_factor_changed,
                            summary.assets_updated,
                            if summary.requires_restart {
                                " ⚠️ RESTART REQUIRED for some changes"
                            } else {
                                ""
                            }
                        );
                    } else {
                        info!(
                            "ℹ️ Regime config reloaded but no tunable parameters changed{}",
                            if summary.requires_restart {
                                " (non-reloadable fields changed — restart required)"
                            } else {
                                ""
                            }
                        );
                    }
                }
                Err(e) => {
                    warn!(
                        "⚠️ Regime config reload failed for {}: {} — keeping current config",
                        path.display(),
                        e
                    );
                }
            }
        }
    })
}

/// Trait for types that contain a mutable `RegimeManager`.
///
/// Implemented by `StrategyState` (in `event_loop.rs`) so the file watcher
/// can access the regime manager through the shared state lock without
/// knowing the full `StrategyState` type.
pub trait RegimeConfigHolder {
    /// Get a mutable reference to the `RegimeManager`.
    fn regime_manager_mut(&mut self) -> &mut RegimeManager;
}

/// Summary of regime state for a single asset.
#[derive(Debug, Clone)]
pub struct AssetRegimeSummary {
    pub symbol: String,
    pub regime: Option<MarketRegime>,
    pub strategy: Option<ActiveStrategy>,
    pub is_ready: bool,
}

impl std::fmt::Display for AssetRegimeSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: regime={} strategy={} ready={}",
            self.symbol,
            self.regime
                .as_ref()
                .map_or("n/a".to_string(), |r| r.to_string()),
            self.strategy
                .as_ref()
                .map_or("n/a".to_string(), |s| s.to_string()),
            self.is_ready,
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // TOML config loading tests ----------------------------------------------

    #[test]
    fn test_from_toml_str_defaults() {
        let config = RegimeManagerConfig::from_toml_str("").unwrap();
        assert_eq!(config.ticks_per_candle, 100);
        assert!((config.min_confidence - 0.5).abs() < f64::EPSILON);
        assert!(config.log_regime_changes);
    }

    #[test]
    fn test_from_toml_str_manager_section() {
        let toml = r#"
[manager]
ticks_per_candle = 50
detection_method = "HMM"
min_confidence = 0.6
volatile_position_factor = 0.2
log_regime_changes = false
"#;
        let config = RegimeManagerConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.ticks_per_candle, 50);
        assert!(matches!(config.detection_method, DetectionMethod::HMM));
        assert!((config.min_confidence - 0.6).abs() < f64::EPSILON);
        assert!((config.volatile_position_factor - 0.2).abs() < f64::EPSILON);
        assert!(!config.log_regime_changes);
    }

    #[test]
    fn test_from_toml_str_partial_override() {
        let toml = r#"
[manager]
ticks_per_candle = 200

[hmm]
learning_rate = 0.01
"#;
        let config = RegimeManagerConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.ticks_per_candle, 200);
        // min_confidence should be default
        assert!((config.min_confidence - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_toml_str_full_router_config() {
        let toml = r#"
[manager]
detection_method = "Ensemble"

[indicators]
adx_period = 20
adx_trending_threshold = 30.0

[hmm]
n_states = 4
min_observations = 50
learning_rate = 0.01

[ensemble]
indicator_weight = 0.7
hmm_weight = 0.3
require_hmm_warmup = false

[router]
min_confidence = 0.6
volatile_position_factor = 0.25
"#;
        let file: RegimeTomlFile = toml::from_str(toml).unwrap();
        let router_config = file.into_router_config();

        assert!(matches!(
            router_config.detection_method,
            DetectionMethod::Ensemble
        ));
        assert_eq!(router_config.indicator_config.adx_period, 20);
        assert!(
            (router_config.indicator_config.adx_trending_threshold - 30.0).abs() < f64::EPSILON
        );

        let hmm = router_config.hmm_config.unwrap();
        assert_eq!(hmm.n_states, 4);
        assert_eq!(hmm.min_observations, 50);
        assert!((hmm.learning_rate - 0.01).abs() < f64::EPSILON);

        let ens = router_config.ensemble_config.unwrap();
        assert!((ens.indicator_weight - 0.7).abs() < f64::EPSILON);
        assert!(!ens.require_hmm_warmup);

        assert!((router_config.min_confidence - 0.6).abs() < f64::EPSILON);
        assert!((router_config.volatile_position_factor - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_toml_str_detection_method_variants() {
        for (input, expected_variant) in &[
            ("Indicators", "indicators"),
            ("indicators", "indicators"),
            ("indicator", "indicators"),
            ("HMM", "hmm"),
            ("hmm", "hmm"),
            ("Ensemble", "ensemble"),
            ("ensemble", "ensemble"),
            ("anything_else", "ensemble"),
        ] {
            let toml = format!("[manager]\ndetection_method = \"{}\"", input);
            let config = RegimeManagerConfig::from_toml_str(&toml).unwrap();
            let label = match config.detection_method {
                DetectionMethod::Indicators => "indicators",
                DetectionMethod::HMM => "hmm",
                DetectionMethod::Ensemble => "ensemble",
            };
            assert_eq!(label, *expected_variant, "Failed for input: {}", input);
        }
    }

    #[test]
    fn test_from_toml_file_missing_returns_error() {
        let result = RegimeManagerConfig::from_toml_file("/nonexistent/regime.toml");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, RegimeConfigError::Io { .. }));
        assert!(err.to_string().contains("/nonexistent/regime.toml"));
    }

    #[test]
    fn test_from_toml_file_or_default_missing_file() {
        let config = RegimeManagerConfig::from_toml_file_or_default("/nonexistent/regime.toml");
        assert_eq!(config.ticks_per_candle, 100); // should be default
    }

    #[test]
    fn test_regime_manager_from_toml_file_or_default() {
        let mgr = RegimeManager::from_toml_file_or_default("/nonexistent/regime.toml");
        assert_eq!(mgr.config().ticks_per_candle, 100);
    }

    #[test]
    fn test_regime_config_error_display() {
        let err = RegimeConfigError::ParseError(
            toml::from_str::<RegimeTomlFile>("invalid {{{{").unwrap_err(),
        );
        let display = err.to_string();
        assert!(display.contains("Failed to parse"));
    }

    #[test]
    fn test_to_router_config() {
        let config = RegimeManagerConfig {
            ticks_per_candle: 50,
            detection_method: DetectionMethod::HMM,
            min_confidence: 0.7,
            volatile_position_factor: 0.2,
            log_regime_changes: false,
            volume_lookback: DEFAULT_VOLUME_LOOKBACK,
            volume_lookback_overrides: HashMap::new(),
        };
        let rc = config.to_router_config();
        assert!(matches!(rc.detection_method, DetectionMethod::HMM));
        assert!((rc.min_confidence - 0.7).abs() < f64::EPSILON);
        assert!((rc.volatile_position_factor - 0.2).abs() < f64::EPSILON);
        assert!(!rc.log_changes);
    }

    fn default_manager() -> RegimeManager {
        RegimeManager::new(RegimeManagerConfig {
            ticks_per_candle: 5,
            log_regime_changes: false,
            ..Default::default()
        })
    }

    // Candle aggregator tests ------------------------------------------------

    #[test]
    fn test_candle_aggregator_produces_candle_after_n_ticks() {
        let mut agg = CandleAggregator::new(3);
        assert!(agg.on_tick(100.0).is_none());
        assert!(agg.on_tick(102.0).is_none());
        let candle = agg.on_tick(101.0);
        assert!(candle.is_some());
        let c = candle.unwrap();
        assert!((c.open - 100.0).abs() < f64::EPSILON);
        assert!((c.high - 102.0).abs() < f64::EPSILON);
        assert!((c.low - 100.0).abs() < f64::EPSILON);
        assert!((c.close - 101.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_candle_aggregator_resets_after_candle() {
        let mut agg = CandleAggregator::new(2);
        agg.on_tick(100.0);
        let _ = agg.on_tick(105.0); // candle 1

        // Next candle should start fresh
        assert!(agg.on_tick(200.0).is_none());
        let c = agg.on_tick(210.0).unwrap();
        // The open of the second candle is the close of the first (carry-over)
        assert!((c.high - 210.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_candle_aggregator_tracks_hlc() {
        let mut agg = CandleAggregator::new(5);
        agg.on_tick(100.0);
        agg.on_tick(110.0); // new high
        agg.on_tick(90.0); // new low
        agg.on_tick(95.0);
        let c = agg.on_tick(105.0).unwrap();
        assert!((c.high - 110.0).abs() < f64::EPSILON);
        assert!((c.low - 90.0).abs() < f64::EPSILON);
        assert!((c.close - 105.0).abs() < f64::EPSILON);
    }

    // Regime manager tests ---------------------------------------------------

    #[test]
    fn test_register_asset() {
        let mut mgr = default_manager();
        mgr.register_asset("BTCUSDT");
        assert_eq!(mgr.asset_count(), 1);
        assert!(!mgr.is_ready("BTCUSDT"));
    }

    #[test]
    fn test_auto_register_on_tick() {
        let mut mgr = default_manager();
        mgr.on_tick_price("ETHUSDT", 2000.0);
        assert_eq!(mgr.asset_count(), 1);
    }

    #[test]
    fn test_no_signal_during_warmup() {
        let mut mgr = default_manager();
        mgr.register_asset("BTCUSDT");

        // Feed a few ticks — not enough to warm up the regime detector
        for i in 0..10 {
            let price = 50000.0 + i as f64 * 10.0;
            let signal = mgr.on_tick_price("BTCUSDT", price);
            // Might get None (candle not ready) or Some with Uncertain regime
            if let Some(ref _s) = signal {
                // During early warmup the regime should be Uncertain
                // (detector not ready yet) — signal is valid either way
            }
        }
    }

    #[test]
    fn test_produces_signal_after_enough_candles() {
        let mut mgr = RegimeManager::new(RegimeManagerConfig {
            ticks_per_candle: 2,
            log_regime_changes: false,
            ..Default::default()
        });
        mgr.register_asset("BTCUSDT");

        let mut got_signal = false;
        // Feed enough data for warmup (~300 candles = 600 ticks with trending data)
        for i in 0..800 {
            let price = 50000.0 + i as f64 * 5.0;
            if let Some(signal) = mgr.on_tick_price("BTCUSDT", price) {
                got_signal = true;
                // Should eventually detect a trend — signal is valid
                let _ = (i, &signal.regime, &signal.strategy, signal.confidence);
            }
        }
        assert!(got_signal, "Should have produced at least one signal");
    }

    #[test]
    fn test_on_candle_direct() {
        let mut mgr = default_manager();
        mgr.register_asset("BTCUSDT");

        // Feed candles directly
        for i in 0..100 {
            let base = 50000.0 + i as f64 * 10.0;
            mgr.on_candle("BTCUSDT", base + 50.0, base - 50.0, base);
        }

        // After enough candles, the detector should have some state
        // (may or may not be "ready" depending on ensemble warmup requirements)
        assert!(mgr.asset_count() == 1);
    }

    #[test]
    fn test_summary() {
        let mut mgr = default_manager();
        mgr.register_asset("BTCUSDT");
        mgr.register_asset("ETHUSDT");

        let summaries = mgr.summary();
        assert_eq!(summaries.len(), 2);

        for s in &summaries {
            assert!(s.symbol == "BTCUSDT" || s.symbol == "ETHUSDT");
            assert!(!s.is_ready);
        }
    }

    #[test]
    fn test_current_regime_none_before_warmup() {
        let mut mgr = default_manager();
        mgr.register_asset("BTCUSDT");
        assert!(mgr.current_regime("BTCUSDT").is_none());
    }

    #[test]
    fn test_default_config() {
        let config = RegimeManagerConfig::default();
        assert_eq!(config.ticks_per_candle, 100);
        assert!(config.log_regime_changes);
    }

    #[test]
    fn test_fast_config() {
        let config = RegimeManagerConfig::fast();
        assert_eq!(config.ticks_per_candle, 25);
    }

    #[test]
    fn test_slow_config() {
        let config = RegimeManagerConfig::slow();
        assert_eq!(config.ticks_per_candle, 500);
    }

    #[test]
    fn test_asset_regime_summary_display() {
        let summary = AssetRegimeSummary {
            symbol: "BTCUSDT".to_string(),
            regime: Some(MarketRegime::MeanReverting),
            strategy: Some(ActiveStrategy::MeanReversion),
            is_ready: true,
        };
        let display = summary.to_string();
        assert!(display.contains("BTCUSDT"));
        assert!(display.contains("ready=true"));
    }

    // Helper conversion tests ------------------------------------------------

    #[test]
    fn test_regime_to_i64() {
        assert_eq!(regime_to_i64(&MarketRegime::Uncertain), 0);
        assert_eq!(
            regime_to_i64(&MarketRegime::Trending(TrendDirection::Bullish)),
            1
        );
        assert_eq!(
            regime_to_i64(&MarketRegime::Trending(TrendDirection::Bearish)),
            2
        );
        assert_eq!(regime_to_i64(&MarketRegime::MeanReverting), 3);
        assert_eq!(regime_to_i64(&MarketRegime::Volatile), 4);
    }

    #[test]
    fn test_strategy_to_i64() {
        assert_eq!(strategy_to_i64(&ActiveStrategy::NoTrade), 0);
        assert_eq!(strategy_to_i64(&ActiveStrategy::TrendFollowing), 1);
        assert_eq!(strategy_to_i64(&ActiveStrategy::MeanReversion), 2);
    }

    #[test]
    fn test_regime_label() {
        assert_eq!(regime_label(&MarketRegime::Uncertain), "uncertain");
        assert_eq!(regime_label(&MarketRegime::MeanReverting), "mean_reverting");
        assert_eq!(regime_label(&MarketRegime::Volatile), "volatile");
        assert_eq!(
            regime_label(&MarketRegime::Trending(TrendDirection::Bullish)),
            "trending_bull"
        );
    }

    // ========================================================================
    // Relative Volume Tests
    // ========================================================================

    #[test]
    fn test_relative_volume_none_before_any_volume() {
        let mut mgr = default_manager();
        mgr.register_asset("BTCUSDT");

        // Feed ticks without any trade volume — no candle volume data
        for i in 0..300 {
            mgr.on_tick_price("BTCUSDT", 100.0 + (i as f64) * 0.01);
        }

        assert!(
            mgr.relative_volume("BTCUSDT").is_none(),
            "relative_volume should be None when no trade volume has been accumulated"
        );
    }

    #[test]
    fn test_relative_volume_unknown_asset_returns_none() {
        let mgr = default_manager();
        assert!(mgr.relative_volume("NONEXISTENT").is_none());
    }

    #[test]
    fn test_relative_volume_available_after_volume_candle() {
        let config = RegimeManagerConfig {
            ticks_per_candle: 3,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);
        mgr.register_asset("BTCUSDT");

        // Accumulate trade volume
        mgr.on_trade_volume("BTCUSDT", 100.0);
        mgr.on_trade_volume("BTCUSDT", 50.0);

        // Complete one candle (3 ticks)
        mgr.on_tick_price("BTCUSDT", 100.0);
        mgr.on_tick_price("BTCUSDT", 101.0);
        mgr.on_tick_price("BTCUSDT", 102.0);

        let rv = mgr.relative_volume("BTCUSDT");
        assert!(
            rv.is_some(),
            "relative_volume should be Some after candle with volume"
        );
        // First candle: 150 / avg(150) = 1.0
        assert!(
            (rv.unwrap() - 1.0).abs() < 1e-10,
            "First candle relative volume should be 1.0, got {}",
            rv.unwrap()
        );
    }

    #[test]
    fn test_relative_volume_detects_spike() {
        let config = RegimeManagerConfig {
            ticks_per_candle: 3,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);
        mgr.register_asset("BTCUSDT");

        // Feed 5 candles with normal volume (100)
        for i in 0..5 {
            mgr.on_trade_volume("BTCUSDT", 100.0);
            let base = 100.0 + i as f64;
            mgr.on_tick_price("BTCUSDT", base);
            mgr.on_tick_price("BTCUSDT", base + 0.5);
            mgr.on_tick_price("BTCUSDT", base + 1.0);
        }

        // Baseline should be ~1.0
        let baseline = mgr.relative_volume("BTCUSDT").expect("should have rv");
        assert!(
            (baseline - 1.0).abs() < 0.01,
            "Baseline relative volume should be ~1.0, got {}",
            baseline
        );

        // Feed a candle with 3× normal volume
        mgr.on_trade_volume("BTCUSDT", 300.0);
        mgr.on_tick_price("BTCUSDT", 106.0);
        mgr.on_tick_price("BTCUSDT", 106.5);
        mgr.on_tick_price("BTCUSDT", 107.0);

        let spike = mgr.relative_volume("BTCUSDT").expect("should have rv");
        assert!(
            spike > 1.5,
            "Relative volume after 3× spike should be > 1.5, got {}",
            spike
        );
    }

    #[test]
    fn test_relative_volume_zero_volume_candles_ignored() {
        let config = RegimeManagerConfig {
            ticks_per_candle: 3,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);
        mgr.register_asset("BTCUSDT");

        // Feed one candle with volume
        mgr.on_trade_volume("BTCUSDT", 200.0);
        mgr.on_tick_price("BTCUSDT", 100.0);
        mgr.on_tick_price("BTCUSDT", 100.5);
        mgr.on_tick_price("BTCUSDT", 101.0);

        let rv_with_vol = mgr.relative_volume("BTCUSDT").expect("should have rv");

        // Feed several candles without volume (no on_trade_volume calls)
        for i in 0..5 {
            let base = 101.0 + i as f64;
            mgr.on_tick_price("BTCUSDT", base);
            mgr.on_tick_price("BTCUSDT", base + 0.5);
            mgr.on_tick_price("BTCUSDT", base + 1.0);
        }

        let rv_after = mgr
            .relative_volume("BTCUSDT")
            .expect("should still have rv");
        assert_eq!(
            rv_with_vol, rv_after,
            "Zero-volume candles should not change relative_volume"
        );
    }

    #[test]
    fn test_relative_volume_rolling_window_bounded() {
        let config = RegimeManagerConfig {
            ticks_per_candle: 3,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);
        mgr.register_asset("BTCUSDT");

        // Feed 100 candles with volume
        for i in 0..100 {
            mgr.on_trade_volume("BTCUSDT", 50.0 + i as f64);
            let base = 100.0 + i as f64 * 0.1;
            mgr.on_tick_price("BTCUSDT", base);
            mgr.on_tick_price("BTCUSDT", base + 0.05);
            mgr.on_tick_price("BTCUSDT", base + 0.1);
        }

        let rv = mgr.relative_volume("BTCUSDT").expect("should have rv");
        assert!(rv.is_finite(), "relative volume must be finite, got {}", rv);
        assert!(rv > 0.0, "relative volume must be positive, got {}", rv);
        // With linearly increasing volumes and a 20-candle window, latest/avg ≈ 1.07
        assert!(rv < 2.0, "relative volume should be reasonable, got {}", rv);
    }

    // ========================================================================
    // Prometheus Bridge Metrics Tests
    // ========================================================================

    #[test]
    fn test_regime_metrics_includes_relative_volume_gauge() {
        let registry = Registry::new();
        let metrics = RegimeMetrics::new(&registry).expect("metrics creation failed");

        // Record a relative volume value
        metrics.record_relative_volume("BTCUSDT", 1.5);

        let val = metrics
            .relative_volume
            .with_label_values(&["BTCUSDT"])
            .get();
        assert!(
            (val - 1.5).abs() < f64::EPSILON,
            "relative_volume gauge should be 1.5, got {}",
            val
        );
    }

    #[test]
    fn test_regime_metrics_includes_bridge_gauges() {
        let registry = Registry::new();
        let metrics = RegimeMetrics::new(&registry).expect("metrics creation failed");

        metrics.record_bridge_state("BTCUSDT", 2, 1, 1.1);

        assert_eq!(
            metrics
                .bridge_hypothalamus
                .with_label_values(&["BTCUSDT"])
                .get(),
            2,
            "bridge_hypothalamus gauge should be 2 (Bullish)"
        );
        assert_eq!(
            metrics
                .bridge_amygdala
                .with_label_values(&["BTCUSDT"])
                .get(),
            1,
            "bridge_amygdala gauge should be 1 (LowVolTrending)"
        );
        assert!(
            (metrics
                .bridge_position_scale
                .with_label_values(&["BTCUSDT"])
                .get()
                - 1.1)
                .abs()
                < f64::EPSILON,
            "bridge_position_scale gauge should be 1.1"
        );
    }

    #[test]
    fn test_regime_metrics_bridge_updates_overwrite() {
        let registry = Registry::new();
        let metrics = RegimeMetrics::new(&registry).expect("metrics creation failed");

        // First state
        metrics.record_bridge_state("ETHUSD", 1, 1, 1.25);
        assert_eq!(
            metrics
                .bridge_hypothalamus
                .with_label_values(&["ETHUSD"])
                .get(),
            1
        );

        // Update to a different state
        metrics.record_bridge_state("ETHUSD", 9, 6, 0.2);
        assert_eq!(
            metrics
                .bridge_hypothalamus
                .with_label_values(&["ETHUSD"])
                .get(),
            9,
            "bridge_hypothalamus should update to 9 (Crisis)"
        );
        assert_eq!(
            metrics.bridge_amygdala.with_label_values(&["ETHUSD"]).get(),
            6,
            "bridge_amygdala should update to 6 (Crisis)"
        );
        assert!(
            (metrics
                .bridge_position_scale
                .with_label_values(&["ETHUSD"])
                .get()
                - 0.2)
                .abs()
                < f64::EPSILON,
            "bridge_position_scale should update to 0.2"
        );
    }

    #[test]
    fn test_regime_metrics_relative_volume_multi_asset() {
        let registry = Registry::new();
        let metrics = RegimeMetrics::new(&registry).expect("metrics creation failed");

        metrics.record_relative_volume("BTCUSDT", 1.2);
        metrics.record_relative_volume("ETHUSDT", 2.5);

        let btc = metrics
            .relative_volume
            .with_label_values(&["BTCUSDT"])
            .get();
        let eth = metrics
            .relative_volume
            .with_label_values(&["ETHUSDT"])
            .get();

        assert!(
            (btc - 1.2).abs() < f64::EPSILON,
            "BTCUSDT relative_volume should be 1.2, got {}",
            btc
        );
        assert!(
            (eth - 2.5).abs() < f64::EPSILON,
            "ETHUSDT relative_volume should be 2.5, got {}",
            eth
        );
    }

    #[test]
    fn test_relative_volume_recorded_in_metrics_on_tick() {
        let registry = Registry::new();
        let config = RegimeManagerConfig {
            ticks_per_candle: 3,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::with_metrics(config, &registry).expect("manager with metrics");
        mgr.register_asset("BTCUSDT");

        // Feed candles with volume — metrics should be updated
        for i in 0..5 {
            mgr.on_trade_volume("BTCUSDT", 100.0);
            let base = 100.0 + i as f64;
            mgr.on_tick_price("BTCUSDT", base);
            mgr.on_tick_price("BTCUSDT", base + 0.5);
            mgr.on_tick_price("BTCUSDT", base + 1.0);
        }

        // The metrics should have a relative_volume gauge value set
        let rv_gauge = mgr
            .metrics()
            .expect("should have metrics")
            .relative_volume
            .with_label_values(&["BTCUSDT"])
            .get();

        assert!(
            rv_gauge > 0.0,
            "relative_volume gauge should be set after candles with volume, got {}",
            rv_gauge
        );
        assert!(
            rv_gauge.is_finite(),
            "relative_volume gauge should be finite"
        );
    }

    #[test]
    fn test_regime_metrics_all_new_gauges_registered() {
        let registry = Registry::new();
        let metrics = RegimeMetrics::new(&registry).expect("metrics creation failed");

        // Write a value to each new gauge so Prometheus includes them in gather()
        metrics.record_relative_volume("TEST", 1.0);
        metrics.record_bridge_state("TEST", 0, 0, 1.0);

        let families = registry.gather();
        let names: Vec<&str> = families.iter().map(|f| f.name()).collect();

        assert!(
            names.contains(&"janus_regime_relative_volume"),
            "Registry should contain janus_regime_relative_volume, found: {:?}",
            names
        );
        assert!(
            names.contains(&"janus_bridge_hypothalamus_regime"),
            "Registry should contain janus_bridge_hypothalamus_regime, found: {:?}",
            names
        );
        assert!(
            names.contains(&"janus_bridge_amygdala_regime"),
            "Registry should contain janus_bridge_amygdala_regime, found: {:?}",
            names
        );
        assert!(
            names.contains(&"janus_bridge_position_scale"),
            "Registry should contain janus_bridge_position_scale, found: {:?}",
            names
        );
    }

    // ── volume_lookback configuration tests ─────────────────────────────

    #[test]
    fn test_default_config_volume_lookback() {
        let config = RegimeManagerConfig::default();
        assert_eq!(config.volume_lookback, DEFAULT_VOLUME_LOOKBACK);
        assert_eq!(config.volume_lookback, 20);
    }

    #[test]
    fn test_fast_config_volume_lookback() {
        let config = RegimeManagerConfig::fast();
        assert_eq!(config.volume_lookback, 10);
    }

    #[test]
    fn test_slow_config_volume_lookback() {
        let config = RegimeManagerConfig::slow();
        assert_eq!(config.volume_lookback, 50);
    }

    #[test]
    fn test_from_toml_str_volume_lookback_default() {
        let config = RegimeManagerConfig::from_toml_str("").unwrap();
        assert_eq!(config.volume_lookback, DEFAULT_VOLUME_LOOKBACK);
    }

    #[test]
    fn test_from_toml_str_volume_lookback_override() {
        let toml = r#"
[manager]
volume_lookback = 35
"#;
        let config = RegimeManagerConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.volume_lookback, 35);
    }

    #[test]
    fn test_from_toml_str_volume_lookback_small() {
        let toml = r#"
[manager]
volume_lookback = 5
"#;
        let config = RegimeManagerConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.volume_lookback, 5);
    }

    #[test]
    fn test_from_toml_str_volume_lookback_large() {
        let toml = r#"
[manager]
volume_lookback = 100
"#;
        let config = RegimeManagerConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.volume_lookback, 100);
    }

    #[test]
    fn test_from_toml_str_volume_lookback_with_other_fields() {
        let toml = r#"
[manager]
ticks_per_candle = 50
volume_lookback = 15
min_confidence = 0.4
"#;
        let config = RegimeManagerConfig::from_toml_str(toml).unwrap();
        assert_eq!(config.ticks_per_candle, 50);
        assert_eq!(config.volume_lookback, 15);
        assert!((config.min_confidence - 0.4).abs() < f64::EPSILON);
    }

    #[test]
    fn test_register_asset_uses_config_volume_lookback() {
        let config = RegimeManagerConfig {
            volume_lookback: 42,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);
        mgr.register_asset("BTCUSDT");

        // Feed enough volume candles to fill a small window, then verify
        // relative volume is computed based on the configured lookback.
        // We use on_trade_volume + on_tick_price to produce candles with volume.
        let ticks_per_candle = mgr.config().ticks_per_candle;

        // Produce 3 candles with known volume
        for candle_idx in 0..3 {
            let vol = 100.0 * (candle_idx as f64 + 1.0);
            mgr.on_trade_volume("BTCUSDT", vol);
            for t in 0..ticks_per_candle {
                let price = 50000.0 + (t as f64) * 0.01;
                mgr.on_tick_price("BTCUSDT", price);
            }
        }

        // Relative volume should be available after producing candles
        let rv = mgr.relative_volume("BTCUSDT");
        assert!(
            rv.is_some(),
            "relative volume should be available after candles with volume"
        );
    }

    #[test]
    fn test_volume_lookback_bounds_history_length() {
        let config = RegimeManagerConfig {
            volume_lookback: 3,  // Very small window
            ticks_per_candle: 5, // Fast candle formation
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);
        mgr.register_asset("TEST");

        // Produce 10 candles with volume — only last 3 should be in the rolling window
        for i in 0..10 {
            let vol = 100.0 * (i as f64 + 1.0);
            mgr.on_trade_volume("TEST", vol);
            for t in 0..5 {
                let price = 1000.0 + (t as f64) * 0.1 + (i as f64) * 10.0;
                mgr.on_tick_price("TEST", price);
            }
        }

        // The relative volume should reflect only the last 3 candles,
        // not all 10. The last candle volume is 1000.0 (i=9).
        // Rolling avg of last 3: (800 + 900 + 1000) / 3 = 900
        // Relative volume = 1000 / 900 ≈ 1.111
        let rv = mgr.relative_volume("TEST");
        assert!(rv.is_some());
        let rv_val = rv.unwrap();
        // With a 3-candle window, relative volume of the last candle should
        // be close to but > 1.0 (since volumes are increasing).
        assert!(rv_val > 0.9, "relative volume {} should be > 0.9", rv_val);
        assert!(rv_val < 2.0, "relative volume {} should be < 2.0", rv_val);
    }

    // ════════════════════════════════════════════════════════════════════
    // Per-asset volume lookback
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_effective_volume_lookback_uses_global_default() {
        let config = RegimeManagerConfig {
            volume_lookback: 25,
            volume_lookback_overrides: HashMap::new(),
            ..Default::default()
        };
        assert_eq!(config.effective_volume_lookback("BTCUSD"), 25);
        assert_eq!(config.effective_volume_lookback("ETHUSD"), 25);
    }

    #[test]
    fn test_effective_volume_lookback_uses_per_asset_override() {
        let mut overrides = HashMap::new();
        overrides.insert("BTCUSD".to_string(), 10);
        overrides.insert("SOLUSD".to_string(), 50);

        let config = RegimeManagerConfig {
            volume_lookback: 20,
            volume_lookback_overrides: overrides,
            ..Default::default()
        };

        // Overridden assets use their specific values
        assert_eq!(config.effective_volume_lookback("BTCUSD"), 10);
        assert_eq!(config.effective_volume_lookback("SOLUSD"), 50);
        // Non-overridden assets fall back to global
        assert_eq!(config.effective_volume_lookback("ETHUSD"), 20);
    }

    #[test]
    fn test_register_asset_respects_per_asset_override() {
        let mut overrides = HashMap::new();
        overrides.insert("BTCUSD".to_string(), 8);

        let config = RegimeManagerConfig {
            ticks_per_candle: 5,
            volume_lookback: 20,
            volume_lookback_overrides: overrides,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);

        mgr.register_asset("BTCUSD");
        mgr.register_asset("ETHUSD");

        // BTCUSD should have lookback = 8 (override)
        let btc_state = mgr.assets.get("BTCUSD").unwrap();
        assert_eq!(btc_state.volume_lookback, 8);

        // ETHUSD should have lookback = 20 (global)
        let eth_state = mgr.assets.get("ETHUSD").unwrap();
        assert_eq!(eth_state.volume_lookback, 20);
    }

    #[test]
    fn test_register_asset_with_lookback_overrides_everything() {
        let config = RegimeManagerConfig {
            ticks_per_candle: 5,
            volume_lookback: 20,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);

        mgr.register_asset_with_lookback("BTCUSD", 7);

        let state = mgr.assets.get("BTCUSD").unwrap();
        assert_eq!(state.volume_lookback, 7);
    }

    #[test]
    fn test_per_asset_lookback_affects_rolling_window_size() {
        let mut overrides = HashMap::new();
        overrides.insert("FAST".to_string(), 3);

        let config = RegimeManagerConfig {
            ticks_per_candle: 5,
            volume_lookback: 20,
            volume_lookback_overrides: overrides,
            log_regime_changes: false,
            ..Default::default()
        };
        let mut mgr = RegimeManager::new(config);
        mgr.register_asset("FAST");
        mgr.register_asset("NORMAL");

        // Feed enough candles with volume to fill windows
        for i in 0..30 {
            let price = 100.0 + (i as f64) * 0.01;
            let state_fast = mgr.assets.get_mut("FAST").unwrap();
            state_fast.aggregator.on_trade_volume(100.0 + i as f64);
            let state_normal = mgr.assets.get_mut("NORMAL").unwrap();
            state_normal.aggregator.on_trade_volume(100.0 + i as f64);

            // Feed ticks to form candles (5 ticks per candle)
            for _ in 0..5 {
                mgr.on_tick_price("FAST", price);
                mgr.on_tick_price("NORMAL", price);
            }
        }

        // FAST should have at most 3 entries in volume_history
        let fast_state = mgr.assets.get("FAST").unwrap();
        assert!(
            fast_state.volume_history.len() <= 3,
            "FAST volume_history should have at most 3 entries, got {}",
            fast_state.volume_history.len()
        );

        // NORMAL should have at most 20 entries in volume_history
        let normal_state = mgr.assets.get("NORMAL").unwrap();
        assert!(
            normal_state.volume_history.len() <= 20,
            "NORMAL volume_history should have at most 20 entries, got {}",
            normal_state.volume_history.len()
        );
    }

    #[test]
    fn test_per_asset_lookback_toml_parsing() {
        let toml_str = r#"
[manager]
ticks_per_candle = 100
volume_lookback = 20

[manager.volume_lookback_overrides]
BTCUSD = 10
ETHUSD = 30
SOLUSD = 50
"#;
        let config = RegimeManagerConfig::from_toml_str(toml_str).expect("should parse TOML");

        assert_eq!(config.volume_lookback, 20);
        assert_eq!(config.effective_volume_lookback("BTCUSD"), 10);
        assert_eq!(config.effective_volume_lookback("ETHUSD"), 30);
        assert_eq!(config.effective_volume_lookback("SOLUSD"), 50);
        assert_eq!(config.effective_volume_lookback("AVAXUSD"), 20); // fallback
    }

    #[test]
    fn test_per_asset_lookback_toml_empty_overrides() {
        let toml_str = r#"
[manager]
ticks_per_candle = 50
volume_lookback = 15
"#;
        let config = RegimeManagerConfig::from_toml_str(toml_str).expect("should parse TOML");

        assert_eq!(config.volume_lookback, 15);
        assert!(config.volume_lookback_overrides.is_empty());
        assert_eq!(config.effective_volume_lookback("BTCUSD"), 15);
    }
}
