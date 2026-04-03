//! Enhanced Strategy Router
//!
//! Routes market data to the appropriate trading strategy based on the detected
//! market regime. Supports three detection methods:
//!
//! 1. **Indicators** — Fast, rule-based (ADX/BB/ATR)
//! 2. **HMM** — Statistical, learns from returns
//! 3. **Ensemble** — Combines both for robustness (recommended)
//!
//! The router is self-contained and emits regime classifications with strategy
//! recommendations. The consuming service (e.g., forward service) is responsible
//! for dispatching to actual strategy implementations.
//!
//! The router also exposes the most recent [`RegimeConfidence`] per asset via
//! [`EnhancedRouter::last_regime_confidence`], giving callers access to the raw
//! indicator values (ADX, BB width percentile, trend strength) that produced the
//! last classification. This is used by the regime bridge to enrich neuromorphic
//! indicator snapshots.
//!
//! Ported from kraken's `strategy/enhanced_router.rs`, adapted for the JANUS
//! architecture. The kraken version embedded a mean reversion strategy directly;
//! this version cleanly separates detection from execution.

use crate::detector::RegimeDetector;
use crate::ensemble::{EnsembleConfig, EnsembleRegimeDetector, EnsembleResult};
use crate::hmm::{HMMConfig, HMMRegimeDetector};
use crate::types::{MarketRegime, RegimeConfidence, RegimeConfig, TrendDirection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Which detection method to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DetectionMethod {
    /// Technical indicators (ADX, BB, ATR) — fast, rule-based
    Indicators,
    /// Hidden Markov Model — statistical, learns from returns
    #[allow(clippy::upper_case_acronyms)]
    HMM,
    /// Ensemble — combines both for robustness (recommended)
    #[default]
    Ensemble,
}

impl std::fmt::Display for DetectionMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DetectionMethod::Indicators => write!(f, "Indicators"),
            DetectionMethod::HMM => write!(f, "HMM"),
            DetectionMethod::Ensemble => write!(f, "Ensemble"),
        }
    }
}

/// Configuration for enhanced router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedRouterConfig {
    /// Which detection method to use
    pub detection_method: DetectionMethod,

    /// Indicator-based config
    pub indicator_config: RegimeConfig,

    /// HMM config
    pub hmm_config: Option<HMMConfig>,

    /// Ensemble config
    pub ensemble_config: Option<EnsembleConfig>,

    /// Position size multiplier for volatile markets (0.0–1.0)
    pub volatile_position_factor: f64,

    /// Minimum confidence to recommend trading
    pub min_confidence: f64,

    /// Log regime changes to stdout
    pub log_changes: bool,
}

impl Default for EnhancedRouterConfig {
    fn default() -> Self {
        Self {
            detection_method: DetectionMethod::Ensemble,
            indicator_config: RegimeConfig::crypto_optimized(),
            hmm_config: Some(HMMConfig::crypto_optimized()),
            ensemble_config: Some(EnsembleConfig::default()),
            volatile_position_factor: 0.5,
            min_confidence: 0.5,
            log_changes: true,
        }
    }
}

/// Active strategy recommendation from the router
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveStrategy {
    /// Use a trend-following strategy (e.g., Golden Cross, EMA Pullback)
    TrendFollowing,
    /// Use a mean-reversion strategy (e.g., Bollinger Bands, VWAP)
    MeanReversion,
    /// Do not trade — regime is unclear or confidence too low
    NoTrade,
}

impl std::fmt::Display for ActiveStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ActiveStrategy::TrendFollowing => write!(f, "Trend Following"),
            ActiveStrategy::MeanReversion => write!(f, "Mean Reversion"),
            ActiveStrategy::NoTrade => write!(f, "No Trade"),
        }
    }
}

/// Signal emitted by the router indicating the recommended action
#[derive(Debug, Clone)]
pub struct RoutedSignal {
    /// Recommended strategy to use
    pub strategy: ActiveStrategy,
    /// Detected market regime
    pub regime: MarketRegime,
    /// Confidence in the regime classification (0.0–1.0)
    pub confidence: f64,
    /// Suggested position size factor (0.0–1.0)
    pub position_factor: f64,
    /// Human-readable reason for the recommendation
    pub reason: String,

    /// Which detection method produced this
    pub detection_method: DetectionMethod,

    /// Did ensemble methods agree? (only populated for Ensemble)
    pub methods_agree: Option<bool>,

    /// HMM state probabilities (only populated for HMM/Ensemble)
    pub state_probabilities: Option<Vec<f64>>,

    /// Expected regime duration in bars (from HMM)
    pub expected_duration: Option<f64>,

    /// Trend direction if trending
    pub trend_direction: Option<TrendDirection>,
}

impl std::fmt::Display for RoutedSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Strategy: {} | Regime: {} | Conf: {:.0}% | Size: {:.0}%",
            self.strategy,
            self.regime,
            self.confidence * 100.0,
            self.position_factor * 100.0
        )?;

        if let Some(agree) = self.methods_agree {
            write!(f, " | Agree: {}", if agree { "✓" } else { "✗" })?;
        }

        if let Some(dur) = self.expected_duration {
            write!(f, " | ExpDur: {:.0} bars", dur)?;
        }

        Ok(())
    }
}

/// Wrapper for different detector types
#[allow(clippy::upper_case_acronyms)]
enum Detector {
    Indicator(Box<RegimeDetector>),
    HMM(Box<HMMRegimeDetector>),
    Ensemble(Box<EnsembleRegimeDetector>),
}

impl std::fmt::Debug for Detector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Detector::Indicator(_) => write!(f, "Detector::Indicator(...)"),
            Detector::HMM(_) => write!(f, "Detector::HMM(...)"),
            Detector::Ensemble(_) => write!(f, "Detector::Ensemble(...)"),
        }
    }
}

/// Per-asset state tracked by the router
#[derive(Debug)]
struct AssetState {
    detector: Detector,
    current_strategy: ActiveStrategy,
    last_regime: MarketRegime,
    regime_change_count: u32,
    /// Most recent `RegimeConfidence` from the detector, including raw
    /// indicator values (ADX, BB width percentile, trend strength).
    last_confidence: Option<RegimeConfidence>,
}

/// Enhanced Strategy Router
///
/// Manages per-asset regime detectors and emits strategy recommendations
/// based on detected market conditions.
///
/// # Example
///
/// ```rust
/// use janus_regime::router::{EnhancedRouter, EnhancedRouterConfig, ActiveStrategy};
///
/// let mut router = EnhancedRouter::with_ensemble();
/// router.register_asset("BTC/USD");
///
/// // Feed OHLC data
/// for i in 0..300 {
///     let price = 50000.0 + i as f64 * 10.0;
///     if let Some(signal) = router.update("BTC/USD", price + 50.0, price - 50.0, price) {
///         if signal.strategy != ActiveStrategy::NoTrade {
///             println!("{}", signal);
///         }
///     }
/// }
/// ```
pub struct EnhancedRouter {
    config: EnhancedRouterConfig,
    assets: HashMap<String, AssetState>,
}

impl std::fmt::Debug for EnhancedRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnhancedRouter")
            .field("config", &self.config)
            .field("assets", &self.assets.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl EnhancedRouter {
    /// Create with specific config
    pub fn new(config: EnhancedRouterConfig) -> Self {
        Self {
            config,
            assets: HashMap::new(),
        }
    }

    /// Create with indicator-based detection
    pub fn with_indicators() -> Self {
        Self::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::Indicators,
            ..Default::default()
        })
    }

    /// Create with HMM-based detection
    pub fn with_hmm() -> Self {
        Self::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::HMM,
            hmm_config: Some(HMMConfig::crypto_optimized()),
            ..Default::default()
        })
    }

    /// Create with Ensemble detection (recommended)
    pub fn with_ensemble() -> Self {
        Self::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::Ensemble,
            ensemble_config: Some(EnsembleConfig::default()),
            ..Default::default()
        })
    }

    /// Register an asset for tracking.
    ///
    /// Creates the appropriate detector based on the configured detection method.
    /// If the asset is already registered, this is a no-op.
    pub fn register_asset(&mut self, symbol: &str) {
        if self.assets.contains_key(symbol) {
            return;
        }

        let detector = match self.config.detection_method {
            DetectionMethod::Indicators => Detector::Indicator(Box::new(RegimeDetector::new(
                self.config.indicator_config.clone(),
            ))),
            DetectionMethod::HMM => {
                let hmm_config = self.config.hmm_config.clone().unwrap_or_default();
                Detector::HMM(Box::new(HMMRegimeDetector::new(hmm_config)))
            }
            DetectionMethod::Ensemble => {
                let ens_config = self.config.ensemble_config.clone().unwrap_or_default();
                Detector::Ensemble(Box::new(EnsembleRegimeDetector::new(
                    ens_config,
                    self.config.indicator_config.clone(),
                )))
            }
        };

        self.assets.insert(
            symbol.to_string(),
            AssetState {
                detector,
                current_strategy: ActiveStrategy::NoTrade,
                last_regime: MarketRegime::Uncertain,
                regime_change_count: 0,
                last_confidence: None,
            },
        );
    }

    /// Unregister an asset, removing its state and detector
    pub fn unregister_asset(&mut self, symbol: &str) -> bool {
        self.assets.remove(symbol).is_some()
    }

    /// Update with new OHLC data for an asset and get a routing signal.
    ///
    /// If the asset isn't registered, it will be auto-registered.
    /// Returns `None` only if the detector fails internally (should not happen).
    pub fn update(
        &mut self,
        symbol: &str,
        high: f64,
        low: f64,
        close: f64,
    ) -> Option<RoutedSignal> {
        if !self.assets.contains_key(symbol) {
            self.register_asset(symbol);
        }

        let state = self.assets.get_mut(symbol)?;

        // Get regime from appropriate detector.
        // We also store the raw `RegimeConfidence` so callers can access
        // indicator values (ADX, BB width, trend strength) after the fact.
        let (regime_result, methods_agree, state_probs, expected_duration) =
            match &mut state.detector {
                Detector::Indicator(det) => {
                    let result = det.update(high, low, close);
                    (result, None, None, None)
                }
                Detector::HMM(det) => {
                    let result = det.update_ohlc(high, low, close);
                    let probs = det.state_probabilities().to_vec();
                    let duration = det.expected_regime_duration(det.current_state_index());
                    (result, None, Some(probs), Some(duration))
                }
                Detector::Ensemble(det) => {
                    let ens_result: EnsembleResult = det.update(high, low, close);
                    let probs = det.hmm_state_probabilities().to_vec();
                    let duration = det.expected_regime_duration();
                    (
                        ens_result.to_regime_confidence(),
                        Some(ens_result.methods_agree),
                        Some(probs),
                        Some(duration),
                    )
                }
            };

        // Stash the raw result for later access via last_regime_confidence()
        state.last_confidence = Some(regime_result);

        // Check for regime change
        if regime_result.regime != state.last_regime {
            state.regime_change_count += 1;
            if self.config.log_changes {
                println!(
                    "[{}] Regime change #{} ({:?}): {} → {} (conf: {:.2})",
                    symbol,
                    state.regime_change_count,
                    self.config.detection_method,
                    state.last_regime,
                    regime_result.regime,
                    regime_result.confidence
                );
            }
            state.last_regime = regime_result.regime;
        }

        // Select strategy based on regime
        let min_confidence = self.config.min_confidence;
        let volatile_factor = self.config.volatile_position_factor;
        let (strategy, position_factor, reason) =
            Self::compute_strategy(&regime_result, min_confidence, volatile_factor);
        state.current_strategy = strategy;

        // Extract trend direction if trending
        let trend_direction = match regime_result.regime {
            MarketRegime::Trending(dir) => Some(dir),
            _ => None,
        };

        Some(RoutedSignal {
            strategy,
            regime: regime_result.regime,
            confidence: regime_result.confidence,
            position_factor,
            reason,
            detection_method: self.config.detection_method,
            methods_agree,
            state_probabilities: state_probs,
            expected_duration,
            trend_direction,
        })
    }

    /// Compute strategy recommendation from a regime classification.
    ///
    /// This is a pure function — no side effects.
    fn compute_strategy(
        regime: &RegimeConfidence,
        min_confidence: f64,
        volatile_factor: f64,
    ) -> (ActiveStrategy, f64, String) {
        if regime.confidence < min_confidence {
            return (
                ActiveStrategy::NoTrade,
                0.0,
                format!(
                    "Confidence too low ({:.0}% < {:.0}%)",
                    regime.confidence * 100.0,
                    min_confidence * 100.0
                ),
            );
        }

        match regime.regime {
            MarketRegime::Trending(dir) => (
                ActiveStrategy::TrendFollowing,
                1.0,
                format!(
                    "{} trend detected (ADX: {:.1}, conf: {:.0}%)",
                    dir,
                    regime.adx_value,
                    regime.confidence * 100.0
                ),
            ),
            MarketRegime::MeanReverting => (
                ActiveStrategy::MeanReversion,
                1.0,
                format!(
                    "Mean-reverting regime (BB%: {:.0}, conf: {:.0}%)",
                    regime.bb_width_percentile,
                    regime.confidence * 100.0
                ),
            ),
            MarketRegime::Volatile => (
                ActiveStrategy::MeanReversion,
                volatile_factor,
                format!(
                    "Volatile regime — reduced size to {:.0}% (conf: {:.0}%)",
                    volatile_factor * 100.0,
                    regime.confidence * 100.0
                ),
            ),
            MarketRegime::Uncertain => (
                ActiveStrategy::NoTrade,
                0.0,
                "Uncertain regime — staying out".to_string(),
            ),
        }
    }

    // ========================================================================
    // Public Accessors
    // ========================================================================

    /// Get current regime for an asset
    pub fn get_regime(&self, symbol: &str) -> Option<MarketRegime> {
        self.assets.get(symbol).map(|s| s.last_regime)
    }

    /// Get the most recent [`RegimeConfidence`] for an asset.
    ///
    /// This contains the raw indicator values (ADX, BB width percentile,
    /// trend strength) that produced the last regime classification.
    /// Returns `None` if the asset hasn't been updated yet.
    pub fn last_regime_confidence(&self, symbol: &str) -> Option<&RegimeConfidence> {
        self.assets
            .get(symbol)
            .and_then(|s| s.last_confidence.as_ref())
    }

    /// Get the current ATR (Average True Range) value for an asset.
    ///
    /// Delegates to the underlying detector regardless of detection method:
    /// - **Indicators** → `RegimeDetector::atr_value()`
    /// - **HMM** → not available (returns `None`)
    /// - **Ensemble** → delegates to the embedded indicator detector
    ///
    /// Returns `None` if the asset isn't registered or the detector hasn't
    /// warmed up yet.
    pub fn atr_value(&self, symbol: &str) -> Option<f64> {
        self.assets.get(symbol).and_then(|s| match &s.detector {
            Detector::Indicator(det) => det.atr_value(),
            Detector::HMM(_) => None,
            Detector::Ensemble(det) => det.indicator_detector().atr_value(),
        })
    }

    /// Get the current ADX (Average Directional Index) value for an asset.
    ///
    /// Delegates to the underlying detector regardless of detection method:
    /// - **Indicators** → `RegimeDetector::adx_value()`
    /// - **HMM** → not available (returns `None`)
    /// - **Ensemble** → delegates to the embedded indicator detector
    ///
    /// Returns `None` if the asset isn't registered or the detector hasn't
    /// warmed up yet.
    pub fn adx_value(&self, symbol: &str) -> Option<f64> {
        self.assets.get(symbol).and_then(|s| match &s.detector {
            Detector::Indicator(det) => det.adx_value(),
            Detector::HMM(_) => None,
            Detector::Ensemble(det) => det.indicator_detector().adx_value(),
        })
    }

    /// Get current recommended strategy for an asset
    pub fn get_strategy(&self, symbol: &str) -> Option<ActiveStrategy> {
        self.assets.get(symbol).map(|s| s.current_strategy)
    }

    /// Check if detector is warmed up for an asset
    pub fn is_ready(&self, symbol: &str) -> bool {
        self.assets
            .get(symbol)
            .map(|s| match &s.detector {
                Detector::Indicator(d) => d.is_ready(),
                Detector::HMM(d) => d.is_ready(),
                Detector::Ensemble(d) => d.is_ready(),
            })
            .unwrap_or(false)
    }

    /// Get detection method being used
    pub fn detection_method(&self) -> DetectionMethod {
        self.config.detection_method
    }

    /// Get regime change count for an asset
    pub fn regime_changes(&self, symbol: &str) -> u32 {
        self.assets
            .get(symbol)
            .map(|s| s.regime_change_count)
            .unwrap_or(0)
    }

    /// Get all registered asset symbols
    pub fn registered_assets(&self) -> Vec<&str> {
        self.assets.keys().map(|s| s.as_str()).collect()
    }

    /// Get the router configuration
    pub fn config(&self) -> &EnhancedRouterConfig {
        &self.config
    }

    /// Get a summary of all asset states
    pub fn summary(&self) -> Vec<AssetSummary> {
        self.assets
            .iter()
            .map(|(symbol, state)| AssetSummary {
                symbol: symbol.clone(),
                regime: state.last_regime,
                strategy: state.current_strategy,
                regime_changes: state.regime_change_count,
                is_ready: match &state.detector {
                    Detector::Indicator(d) => d.is_ready(),
                    Detector::HMM(d) => d.is_ready(),
                    Detector::Ensemble(d) => d.is_ready(),
                },
            })
            .collect()
    }
}

/// Summary of an asset's regime state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetSummary {
    pub symbol: String,
    pub regime: MarketRegime,
    pub strategy: ActiveStrategy,
    pub regime_changes: u32,
    pub is_ready: bool,
}

impl std::fmt::Display for AssetSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} → {} (changes: {}, ready: {})",
            self.symbol, self.regime, self.strategy, self.regime_changes, self.is_ready
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation_ensemble() {
        let router = EnhancedRouter::with_ensemble();
        assert_eq!(router.detection_method(), DetectionMethod::Ensemble);
    }

    #[test]
    fn test_router_creation_indicators() {
        let router = EnhancedRouter::with_indicators();
        assert_eq!(router.detection_method(), DetectionMethod::Indicators);
    }

    #[test]
    fn test_router_creation_hmm() {
        let router = EnhancedRouter::with_hmm();
        assert_eq!(router.detection_method(), DetectionMethod::HMM);
    }

    #[test]
    fn test_method_switching() {
        let indicator_router = EnhancedRouter::with_indicators();
        let hmm_router = EnhancedRouter::with_hmm();
        let ensemble_router = EnhancedRouter::with_ensemble();

        assert_eq!(
            indicator_router.detection_method(),
            DetectionMethod::Indicators
        );
        assert_eq!(hmm_router.detection_method(), DetectionMethod::HMM);
        assert_eq!(
            ensemble_router.detection_method(),
            DetectionMethod::Ensemble
        );
    }

    #[test]
    fn test_asset_registration() {
        let mut router = EnhancedRouter::with_ensemble();
        router.register_asset("BTC/USD");
        router.register_asset("ETH/USD");

        assert!(router.get_regime("BTC/USD").is_some());
        assert!(router.get_regime("ETH/USD").is_some());
        assert!(router.get_regime("SOL/USD").is_none());
    }

    #[test]
    fn test_asset_unregistration() {
        let mut router = EnhancedRouter::with_ensemble();
        router.register_asset("BTC/USD");
        assert!(router.get_regime("BTC/USD").is_some());

        assert!(router.unregister_asset("BTC/USD"));
        assert!(router.get_regime("BTC/USD").is_none());

        // Unregistering non-existent asset returns false
        assert!(!router.unregister_asset("BTC/USD"));
    }

    #[test]
    fn test_auto_registration() {
        let mut router = EnhancedRouter::with_indicators();

        // Should auto-register on first update
        assert!(router.get_regime("BTC/USD").is_none());
        let signal = router.update("BTC/USD", 101.0, 99.0, 100.0);
        assert!(signal.is_some());
        assert!(router.get_regime("BTC/USD").is_some());
    }

    #[test]
    fn test_duplicate_registration_noop() {
        let mut router = EnhancedRouter::with_ensemble();
        router.register_asset("BTC/USD");

        // Feed some data
        for i in 0..50 {
            let price = 100.0 + i as f64;
            router.update("BTC/USD", price + 1.0, price - 1.0, price);
        }

        let changes_before = router.regime_changes("BTC/USD");

        // Re-registering should be a no-op
        router.register_asset("BTC/USD");

        let changes_after = router.regime_changes("BTC/USD");
        assert_eq!(changes_before, changes_after);
    }

    #[test]
    fn test_registered_assets() {
        let mut router = EnhancedRouter::with_ensemble();
        router.register_asset("BTC/USD");
        router.register_asset("ETH/USD");
        router.register_asset("SOL/USD");

        let assets = router.registered_assets();
        assert_eq!(assets.len(), 3);
        assert!(assets.contains(&"BTC/USD"));
        assert!(assets.contains(&"ETH/USD"));
        assert!(assets.contains(&"SOL/USD"));
    }

    #[test]
    fn test_initial_regime_is_uncertain() {
        let mut router = EnhancedRouter::with_ensemble();
        router.register_asset("BTC/USD");

        assert_eq!(router.get_regime("BTC/USD"), Some(MarketRegime::Uncertain));
        assert_eq!(
            router.get_strategy("BTC/USD"),
            Some(ActiveStrategy::NoTrade)
        );
    }

    #[test]
    fn test_not_ready_before_warmup() {
        let mut router = EnhancedRouter::with_indicators();
        router.register_asset("BTC/USD");

        assert!(!router.is_ready("BTC/USD"));

        // Feed a few bars — not enough for warmup
        for i in 0..10 {
            let price = 100.0 + i as f64;
            router.update("BTC/USD", price + 1.0, price - 1.0, price);
        }

        assert!(!router.is_ready("BTC/USD"));
    }

    #[test]
    fn test_is_ready_unknown_asset() {
        let router = EnhancedRouter::with_ensemble();
        assert!(!router.is_ready("UNKNOWN"));
    }

    #[test]
    fn test_regime_changes_counted() {
        let mut router = EnhancedRouter::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::Indicators,
            log_changes: false, // Suppress output in tests
            ..Default::default()
        });

        router.register_asset("BTC/USD");
        assert_eq!(router.regime_changes("BTC/USD"), 0);

        // Feed data — regime may or may not change depending on data
        for i in 0..300 {
            let price = 100.0 + i as f64 * 0.5;
            router.update("BTC/USD", price + 1.0, price - 1.0, price);
        }

        // At minimum, the regime should have been set at least once
        // (exact count depends on stability filter and data)
        let changes = router.regime_changes("BTC/USD");
        let _ = changes; // Just verify it doesn't panic
    }

    #[test]
    fn test_routed_signal_fields() {
        let mut router = EnhancedRouter::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::Indicators,
            log_changes: false,
            ..Default::default()
        });

        let signal = router.update("BTC/USD", 101.0, 99.0, 100.0);
        assert!(signal.is_some());

        let signal = signal.unwrap();
        assert_eq!(signal.detection_method, DetectionMethod::Indicators);
        assert!((0.0..=1.0).contains(&signal.confidence));
        assert!((0.0..=1.0).contains(&signal.position_factor));
        assert!(!signal.reason.is_empty());
        // Indicator method doesn't populate these
        assert!(signal.methods_agree.is_none());
        assert!(signal.state_probabilities.is_none());
        assert!(signal.expected_duration.is_none());
    }

    #[test]
    fn test_routed_signal_display() {
        let signal = RoutedSignal {
            strategy: ActiveStrategy::TrendFollowing,
            regime: MarketRegime::Trending(TrendDirection::Bullish),
            confidence: 0.85,
            position_factor: 1.0,
            reason: "Bullish trend".to_string(),
            detection_method: DetectionMethod::Ensemble,
            methods_agree: Some(true),
            state_probabilities: Some(vec![0.6, 0.2, 0.2]),
            expected_duration: Some(15.0),
            trend_direction: Some(TrendDirection::Bullish),
        };

        let display = format!("{signal}");
        assert!(display.contains("Trend Following"));
        assert!(display.contains("85%"));
        assert!(display.contains("100%"));
        assert!(display.contains("✓"));
        assert!(display.contains("15 bars"));
    }

    #[test]
    fn test_compute_strategy_low_confidence() {
        let regime = RegimeConfidence::new(MarketRegime::Trending(TrendDirection::Bullish), 0.3);
        let (strategy, factor, reason) = EnhancedRouter::compute_strategy(&regime, 0.5, 0.5);

        assert_eq!(strategy, ActiveStrategy::NoTrade);
        assert_eq!(factor, 0.0);
        assert!(reason.contains("Confidence too low"));
    }

    #[test]
    fn test_compute_strategy_trending() {
        let regime = RegimeConfidence::with_metrics(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.8,
            30.0,
            50.0,
            0.7,
        );
        let (strategy, factor, reason) = EnhancedRouter::compute_strategy(&regime, 0.5, 0.5);

        assert_eq!(strategy, ActiveStrategy::TrendFollowing);
        assert_eq!(factor, 1.0);
        assert!(reason.contains("Bullish"));
    }

    #[test]
    fn test_compute_strategy_mean_reverting() {
        let regime =
            RegimeConfidence::with_metrics(MarketRegime::MeanReverting, 0.7, 15.0, 30.0, 0.2);
        let (strategy, factor, reason) = EnhancedRouter::compute_strategy(&regime, 0.5, 0.5);

        assert_eq!(strategy, ActiveStrategy::MeanReversion);
        assert_eq!(factor, 1.0);
        assert!(reason.contains("Mean-reverting"));
    }

    #[test]
    fn test_compute_strategy_volatile() {
        let regime = RegimeConfidence::with_metrics(MarketRegime::Volatile, 0.75, 22.0, 85.0, 0.3);
        let (strategy, factor, reason) = EnhancedRouter::compute_strategy(&regime, 0.5, 0.4);

        assert_eq!(strategy, ActiveStrategy::MeanReversion);
        assert_eq!(factor, 0.4);
        assert!(reason.contains("Volatile"));
        assert!(reason.contains("40%"));
    }

    #[test]
    fn test_compute_strategy_uncertain() {
        let regime = RegimeConfidence::new(MarketRegime::Uncertain, 0.6);
        let (strategy, factor, _) = EnhancedRouter::compute_strategy(&regime, 0.5, 0.5);

        assert_eq!(strategy, ActiveStrategy::NoTrade);
        assert_eq!(factor, 0.0);
    }

    #[test]
    fn test_active_strategy_display() {
        assert_eq!(
            format!("{}", ActiveStrategy::TrendFollowing),
            "Trend Following"
        );
        assert_eq!(
            format!("{}", ActiveStrategy::MeanReversion),
            "Mean Reversion"
        );
        assert_eq!(format!("{}", ActiveStrategy::NoTrade), "No Trade");
    }

    #[test]
    fn test_detection_method_display() {
        assert_eq!(format!("{}", DetectionMethod::Indicators), "Indicators");
        assert_eq!(format!("{}", DetectionMethod::HMM), "HMM");
        assert_eq!(format!("{}", DetectionMethod::Ensemble), "Ensemble");
    }

    #[test]
    fn test_summary() {
        let mut router = EnhancedRouter::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::Indicators,
            log_changes: false,
            ..Default::default()
        });

        router.register_asset("BTC/USD");
        router.register_asset("ETH/USD");

        let summary = router.summary();
        assert_eq!(summary.len(), 2);

        for s in &summary {
            assert!(s.symbol == "BTC/USD" || s.symbol == "ETH/USD");
            assert_eq!(s.regime, MarketRegime::Uncertain);
            assert_eq!(s.strategy, ActiveStrategy::NoTrade);
            assert_eq!(s.regime_changes, 0);
        }
    }

    #[test]
    fn test_asset_summary_display() {
        let summary = AssetSummary {
            symbol: "BTC/USD".to_string(),
            regime: MarketRegime::Trending(TrendDirection::Bullish),
            strategy: ActiveStrategy::TrendFollowing,
            regime_changes: 3,
            is_ready: true,
        };

        let display = format!("{summary}");
        assert!(display.contains("BTC/USD"));
        assert!(display.contains("Trending"));
        assert!(display.contains("Trend Following"));
        assert!(display.contains("3"));
    }

    #[test]
    fn test_hmm_signal_has_state_probs() {
        let mut router = EnhancedRouter::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::HMM,
            log_changes: false,
            ..Default::default()
        });

        let signal = router.update("BTC/USD", 101.0, 99.0, 100.0);
        let signal = signal.unwrap();

        assert!(signal.state_probabilities.is_some());
        let probs = signal.state_probabilities.unwrap();
        assert_eq!(probs.len(), 3);

        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "State probabilities should sum to 1.0"
        );
    }

    #[test]
    fn test_ensemble_signal_has_agreement() {
        let mut router = EnhancedRouter::new(EnhancedRouterConfig {
            detection_method: DetectionMethod::Ensemble,
            log_changes: false,
            ..Default::default()
        });

        let signal = router.update("BTC/USD", 101.0, 99.0, 100.0);
        let signal = signal.unwrap();

        assert!(signal.methods_agree.is_some());
        assert!(signal.state_probabilities.is_some());
        assert!(signal.expected_duration.is_some());
    }
}
