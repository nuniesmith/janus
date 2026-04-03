//! Ensemble Regime Detector
//!
//! Combines multiple regime detection methods for more robust classification:
//! 1. **Technical Indicators** (ADX, Bollinger Bands, ATR) — Fast, rule-based
//! 2. **Hidden Markov Model** — Statistical, learns from returns
//!
//! The ensemble approach provides more robust regime detection by:
//! - Reducing false positives when methods disagree
//! - Increasing confidence when methods agree
//! - Leveraging different strengths of each approach
//!
//! Ported from kraken's `regime/ensemble.rs`, adapted for the JANUS type system.

use super::detector::RegimeDetector;
use super::hmm::HMMRegimeDetector;
use super::types::{MarketRegime, RegimeConfidence, RegimeConfig};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for ensemble detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Weight for technical indicator detector (0.0 - 1.0)
    pub indicator_weight: f64,
    /// Weight for HMM detector (0.0 - 1.0)
    pub hmm_weight: f64,
    /// Minimum agreement threshold to declare a regime
    pub agreement_threshold: f64,
    /// Use HMM only after warmup (more conservative)
    pub require_hmm_warmup: bool,
    /// Boost confidence when both methods agree
    pub agreement_confidence_boost: f64,
    /// Reduce confidence when methods disagree
    pub disagreement_confidence_penalty: f64,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            indicator_weight: 0.6, // Slightly favor indicators (faster response)
            hmm_weight: 0.4,
            agreement_threshold: 0.5,
            require_hmm_warmup: true,
            agreement_confidence_boost: 0.15,
            disagreement_confidence_penalty: 0.2,
        }
    }
}

impl EnsembleConfig {
    /// Equal weighting between methods
    pub fn balanced() -> Self {
        Self {
            indicator_weight: 0.5,
            hmm_weight: 0.5,
            ..Default::default()
        }
    }

    /// Favor HMM (more statistical)
    pub fn hmm_focused() -> Self {
        Self {
            indicator_weight: 0.3,
            hmm_weight: 0.7,
            agreement_threshold: 0.6,
            ..Default::default()
        }
    }

    /// Favor indicators (faster response)
    pub fn indicator_focused() -> Self {
        Self {
            indicator_weight: 0.7,
            hmm_weight: 0.3,
            agreement_threshold: 0.4,
            ..Default::default()
        }
    }
}

/// Result from ensemble detection
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    /// Final regime determination
    pub regime: MarketRegime,
    /// Combined confidence
    pub confidence: f64,
    /// Whether methods agree on regime category
    pub methods_agree: bool,
    /// Indicator-based result
    pub indicator_result: RegimeConfidence,
    /// HMM-based result
    pub hmm_result: RegimeConfidence,
    /// Individual method regimes for debugging
    pub indicator_regime: MarketRegime,
    pub hmm_regime: MarketRegime,
}

impl EnsembleResult {
    /// Convert to standard `RegimeConfidence`
    pub fn to_regime_confidence(&self) -> RegimeConfidence {
        RegimeConfidence::new(self.regime, self.confidence)
    }
}

impl std::fmt::Display for EnsembleResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Ensemble: {} (conf: {:.0}%, agree: {})",
            self.regime,
            self.confidence * 100.0,
            if self.methods_agree { "✓" } else { "✗" }
        )
    }
}

/// Ensemble regime detector combining indicator-based and HMM methods.
///
/// Feeds the same OHLC data to both detectors simultaneously and combines
/// their outputs using weighted averaging with agreement bonuses/penalties.
///
/// # Example
///
/// ```rust
/// use janus_regime::{EnsembleRegimeDetector, EnsembleConfig, RegimeConfig, MarketRegime};
///
/// let mut ensemble = EnsembleRegimeDetector::default_config();
///
/// // Feed OHLC bars
/// for i in 0..300 {
///     let price = 100.0 + i as f64 * 0.5;
///     let result = ensemble.update(price + 1.0, price - 1.0, price);
///     if ensemble.is_ready() {
///         println!("{}", result);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct EnsembleRegimeDetector {
    config: EnsembleConfig,

    /// Technical indicator-based detector
    indicator_detector: RegimeDetector,

    /// Hidden Markov Model detector
    hmm_detector: HMMRegimeDetector,

    /// Current ensemble regime
    current_regime: MarketRegime,

    /// Track agreement history
    agreement_history: VecDeque<bool>,
}

impl EnsembleRegimeDetector {
    /// Create with specific configs for both the ensemble and the indicator detector
    pub fn new(ensemble_config: EnsembleConfig, indicator_config: RegimeConfig) -> Self {
        Self {
            config: ensemble_config,
            indicator_detector: RegimeDetector::new(indicator_config),
            hmm_detector: HMMRegimeDetector::crypto_optimized(),
            current_regime: MarketRegime::Uncertain,
            agreement_history: VecDeque::with_capacity(100),
        }
    }

    /// Create with default configs (indicator-weighted, crypto-optimized)
    pub fn default_config() -> Self {
        Self::new(EnsembleConfig::default(), RegimeConfig::crypto_optimized())
    }

    /// Create balanced ensemble (equal weighting)
    pub fn balanced() -> Self {
        Self::new(EnsembleConfig::balanced(), RegimeConfig::crypto_optimized())
    }

    /// Create indicator-focused ensemble
    pub fn indicator_focused() -> Self {
        Self::new(
            EnsembleConfig::indicator_focused(),
            RegimeConfig::crypto_optimized(),
        )
    }

    /// Create HMM-focused ensemble
    pub fn hmm_focused() -> Self {
        Self::new(
            EnsembleConfig::hmm_focused(),
            RegimeConfig::crypto_optimized(),
        )
    }

    /// Update with new OHLC data and get the ensemble result.
    ///
    /// Both detectors are updated with the same data. The ensemble then
    /// combines their outputs, adjusting confidence based on agreement.
    pub fn update(&mut self, high: f64, low: f64, close: f64) -> EnsembleResult {
        // Update both detectors
        let indicator_result = self.indicator_detector.update(high, low, close);
        let hmm_result = self.hmm_detector.update_ohlc(high, low, close);

        // Get individual regimes
        let indicator_regime = indicator_result.regime;
        let hmm_regime = hmm_result.regime;

        // Check if HMM is warmed up
        let hmm_ready = self.hmm_detector.is_ready();

        // Determine if methods agree
        let methods_agree = Self::regimes_agree(indicator_regime, hmm_regime);

        // Track agreement
        self.agreement_history.push_back(methods_agree);
        if self.agreement_history.len() > 100 {
            self.agreement_history.pop_front();
        }

        // Calculate combined regime and confidence
        let (regime, confidence) = if self.config.require_hmm_warmup && !hmm_ready {
            // Use only indicators until HMM is ready
            (indicator_regime, indicator_result.confidence)
        } else {
            self.combine_results(
                indicator_regime,
                indicator_result.confidence,
                hmm_regime,
                hmm_result.confidence,
                methods_agree,
            )
        };

        self.current_regime = regime;

        EnsembleResult {
            regime,
            confidence,
            methods_agree,
            indicator_result,
            hmm_result,
            indicator_regime,
            hmm_regime,
        }
    }

    /// Check if two regimes agree (same category, direction may differ)
    fn regimes_agree(r1: MarketRegime, r2: MarketRegime) -> bool {
        matches!(
            (r1, r2),
            (MarketRegime::Trending(_), MarketRegime::Trending(_))
                | (MarketRegime::MeanReverting, MarketRegime::MeanReverting)
                | (MarketRegime::Volatile, MarketRegime::Volatile)
                | (MarketRegime::Uncertain, MarketRegime::Uncertain)
        )
    }

    /// Check if regimes agree on direction too (stricter)
    fn regimes_agree_direction(r1: MarketRegime, r2: MarketRegime) -> bool {
        match (r1, r2) {
            (MarketRegime::Trending(d1), MarketRegime::Trending(d2)) => d1 == d2,
            (MarketRegime::MeanReverting, MarketRegime::MeanReverting) => true,
            (MarketRegime::Volatile, MarketRegime::Volatile) => true,
            (MarketRegime::Uncertain, MarketRegime::Uncertain) => true,
            _ => false,
        }
    }

    /// Combine results from both methods using weighted averaging
    fn combine_results(
        &self,
        indicator_regime: MarketRegime,
        indicator_conf: f64,
        hmm_regime: MarketRegime,
        hmm_conf: f64,
        agree: bool,
    ) -> (MarketRegime, f64) {
        let w_ind = self.config.indicator_weight;
        let w_hmm = self.config.hmm_weight;

        // Weighted confidence
        let mut combined_conf = w_ind * indicator_conf + w_hmm * hmm_conf;

        // Adjust confidence based on agreement
        if agree {
            // Boost confidence when methods agree
            combined_conf += self.config.agreement_confidence_boost;

            // Extra boost if they agree on direction too
            if Self::regimes_agree_direction(indicator_regime, hmm_regime) {
                combined_conf += 0.05;
            }
        } else {
            // Penalty when methods disagree
            combined_conf -= self.config.disagreement_confidence_penalty;
        }

        combined_conf = combined_conf.clamp(0.0, 1.0);

        // Determine final regime
        let regime = if agree {
            // Use the regime they agree on (prefer indicator's direction if trending)
            indicator_regime
        } else if combined_conf < self.config.agreement_threshold {
            // Low confidence due to disagreement - be conservative
            MarketRegime::Uncertain
        } else {
            // Use higher-weighted method's regime
            if w_ind >= w_hmm {
                indicator_regime
            } else {
                hmm_regime
            }
        };

        (regime, combined_conf)
    }

    // ========================================================================
    // Public Accessors
    // ========================================================================

    /// Get current regime
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get agreement rate over recent history (0.0 to 1.0)
    pub fn agreement_rate(&self) -> f64 {
        if self.agreement_history.is_empty() {
            return 0.0;
        }
        let agrees = self.agreement_history.iter().filter(|&&a| a).count();
        agrees as f64 / self.agreement_history.len() as f64
    }

    /// Check if both detectors are ready.
    ///
    /// When `require_hmm_warmup` is true, both must be ready.
    /// Otherwise, only the indicator detector needs to be ready.
    pub fn is_ready(&self) -> bool {
        self.indicator_detector.is_ready()
            && (!self.config.require_hmm_warmup || self.hmm_detector.is_ready())
    }

    /// Check if only the indicator detector is ready (HMM may still be warming up)
    pub fn indicator_ready(&self) -> bool {
        self.indicator_detector.is_ready()
    }

    /// Check if the HMM detector is ready
    pub fn hmm_ready(&self) -> bool {
        self.hmm_detector.is_ready()
    }

    /// Get HMM state probabilities
    pub fn hmm_state_probabilities(&self) -> &[f64] {
        self.hmm_detector.state_probabilities()
    }

    /// Get HMM expected regime duration
    pub fn expected_regime_duration(&self) -> f64 {
        self.hmm_detector
            .expected_regime_duration(self.hmm_detector.current_state_index())
    }

    /// Get detailed status for monitoring
    pub fn status(&self) -> EnsembleStatus {
        EnsembleStatus {
            current_regime: self.current_regime,
            indicator_ready: self.indicator_detector.is_ready(),
            hmm_ready: self.hmm_detector.is_ready(),
            agreement_rate: self.agreement_rate(),
            hmm_state_probs: self.hmm_detector.state_probabilities().to_vec(),
            expected_duration: self.expected_regime_duration(),
        }
    }

    /// Get a reference to the underlying indicator detector
    pub fn indicator_detector(&self) -> &RegimeDetector {
        &self.indicator_detector
    }

    /// Get a reference to the underlying HMM detector
    pub fn hmm_detector(&self) -> &HMMRegimeDetector {
        &self.hmm_detector
    }

    /// Get the ensemble configuration
    pub fn config(&self) -> &EnsembleConfig {
        &self.config
    }
}

/// Status information for monitoring / dashboards
#[derive(Debug, Clone)]
pub struct EnsembleStatus {
    pub current_regime: MarketRegime,
    pub indicator_ready: bool,
    pub hmm_ready: bool,
    pub agreement_rate: f64,
    pub hmm_state_probs: Vec<f64>,
    pub expected_duration: f64,
}

impl std::fmt::Display for EnsembleStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Regime: {} | Agreement: {:.1}% | HMM Ready: {} | Expected Duration: {:.1} bars",
            self.current_regime,
            self.agreement_rate * 100.0,
            self.hmm_ready,
            self.expected_duration
        )
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrendDirection;

    #[test]
    fn test_ensemble_creation() {
        let ensemble = EnsembleRegimeDetector::default_config();
        assert!(!ensemble.is_ready());
        assert_eq!(ensemble.current_regime(), MarketRegime::Uncertain);
    }

    #[test]
    fn test_balanced_creation() {
        let ensemble = EnsembleRegimeDetector::balanced();
        assert!(!ensemble.is_ready());
        assert_eq!(ensemble.config().indicator_weight, 0.5);
        assert_eq!(ensemble.config().hmm_weight, 0.5);
    }

    #[test]
    fn test_indicator_focused_creation() {
        let ensemble = EnsembleRegimeDetector::indicator_focused();
        assert!(ensemble.config().indicator_weight > ensemble.config().hmm_weight);
    }

    #[test]
    fn test_hmm_focused_creation() {
        let ensemble = EnsembleRegimeDetector::hmm_focused();
        assert!(ensemble.config().hmm_weight > ensemble.config().indicator_weight);
    }

    #[test]
    fn test_regimes_agree_same_category() {
        // Same category should agree
        assert!(EnsembleRegimeDetector::regimes_agree(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::Trending(TrendDirection::Bearish)
        ));

        assert!(EnsembleRegimeDetector::regimes_agree(
            MarketRegime::MeanReverting,
            MarketRegime::MeanReverting
        ));

        assert!(EnsembleRegimeDetector::regimes_agree(
            MarketRegime::Volatile,
            MarketRegime::Volatile
        ));

        assert!(EnsembleRegimeDetector::regimes_agree(
            MarketRegime::Uncertain,
            MarketRegime::Uncertain
        ));
    }

    #[test]
    fn test_regimes_disagree_different_category() {
        assert!(!EnsembleRegimeDetector::regimes_agree(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::MeanReverting
        ));

        assert!(!EnsembleRegimeDetector::regimes_agree(
            MarketRegime::Volatile,
            MarketRegime::Trending(TrendDirection::Bearish)
        ));

        assert!(!EnsembleRegimeDetector::regimes_agree(
            MarketRegime::Uncertain,
            MarketRegime::MeanReverting
        ));
    }

    #[test]
    fn test_regimes_agree_direction() {
        assert!(EnsembleRegimeDetector::regimes_agree_direction(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::Trending(TrendDirection::Bullish)
        ));

        assert!(!EnsembleRegimeDetector::regimes_agree_direction(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::Trending(TrendDirection::Bearish)
        ));

        assert!(EnsembleRegimeDetector::regimes_agree_direction(
            MarketRegime::MeanReverting,
            MarketRegime::MeanReverting
        ));

        assert!(!EnsembleRegimeDetector::regimes_agree_direction(
            MarketRegime::Trending(TrendDirection::Bullish),
            MarketRegime::MeanReverting
        ));
    }

    #[test]
    fn test_agreement_rate_empty() {
        let ensemble = EnsembleRegimeDetector::default_config();
        assert_eq!(ensemble.agreement_rate(), 0.0);
    }

    #[test]
    fn test_agreement_rate_tracked() {
        let mut ensemble = EnsembleRegimeDetector::default_config();

        // Simulate some updates
        let mut price = 100.0;
        for i in 0..50 {
            price *= if i % 2 == 0 { 1.01 } else { 0.99 };
            ensemble.update(price * 1.01, price * 0.99, price);
        }

        // Should have some agreement rate between 0 and 1
        let rate = ensemble.agreement_rate();
        assert!(
            (0.0..=1.0).contains(&rate),
            "Agreement rate should be in [0, 1]: {rate}"
        );
    }

    #[test]
    fn test_bull_market_agreement() {
        let mut ensemble = EnsembleRegimeDetector::default_config();

        // Strong bull market - both methods should eventually agree
        let mut price = 100.0;
        for _ in 0..300 {
            price *= 1.005; // Consistent upward
            let high = price * 1.002;
            let low = price * 0.998;
            ensemble.update(high, low, price);
        }

        let result = ensemble.update(price * 1.002, price * 0.998, price);

        // In a strong trend, agreement rate should be reasonable
        assert!(
            ensemble.agreement_rate() > 0.2,
            "Agreement rate should be > 0.2 in consistent bull market: {}",
            ensemble.agreement_rate()
        );

        // Result should be valid
        assert!(
            (0.0..=1.0).contains(&result.confidence),
            "Confidence should be in [0, 1]: {}",
            result.confidence
        );
    }

    #[test]
    fn test_ensemble_result_display() {
        let result = EnsembleResult {
            regime: MarketRegime::Trending(TrendDirection::Bullish),
            confidence: 0.85,
            methods_agree: true,
            indicator_result: RegimeConfidence::new(
                MarketRegime::Trending(TrendDirection::Bullish),
                0.8,
            ),
            hmm_result: RegimeConfidence::new(MarketRegime::Trending(TrendDirection::Bullish), 0.9),
            indicator_regime: MarketRegime::Trending(TrendDirection::Bullish),
            hmm_regime: MarketRegime::Trending(TrendDirection::Bullish),
        };

        let display = format!("{result}");
        assert!(display.contains("Trending (Bullish)"));
        assert!(display.contains("85%"));
        assert!(display.contains("✓"));
    }

    #[test]
    fn test_ensemble_result_disagreement_display() {
        let result = EnsembleResult {
            regime: MarketRegime::Uncertain,
            confidence: 0.3,
            methods_agree: false,
            indicator_result: RegimeConfidence::new(
                MarketRegime::Trending(TrendDirection::Bullish),
                0.6,
            ),
            hmm_result: RegimeConfidence::new(MarketRegime::MeanReverting, 0.5),
            indicator_regime: MarketRegime::Trending(TrendDirection::Bullish),
            hmm_regime: MarketRegime::MeanReverting,
        };

        let display = format!("{result}");
        assert!(display.contains("✗"));
    }

    #[test]
    fn test_ensemble_to_regime_confidence() {
        let result = EnsembleResult {
            regime: MarketRegime::MeanReverting,
            confidence: 0.72,
            methods_agree: true,
            indicator_result: RegimeConfidence::new(MarketRegime::MeanReverting, 0.7),
            hmm_result: RegimeConfidence::new(MarketRegime::MeanReverting, 0.75),
            indicator_regime: MarketRegime::MeanReverting,
            hmm_regime: MarketRegime::MeanReverting,
        };

        let rc = result.to_regime_confidence();
        assert_eq!(rc.regime, MarketRegime::MeanReverting);
        assert!((rc.confidence - 0.72).abs() < f64::EPSILON);
    }

    #[test]
    fn test_status_display() {
        let status = EnsembleStatus {
            current_regime: MarketRegime::Volatile,
            indicator_ready: true,
            hmm_ready: false,
            agreement_rate: 0.65,
            hmm_state_probs: vec![0.3, 0.3, 0.4],
            expected_duration: 8.5,
        };

        let display = format!("{status}");
        assert!(display.contains("Volatile"));
        assert!(display.contains("65.0%"));
        assert!(display.contains("false"));
    }

    #[test]
    fn test_ready_state() {
        let mut ensemble = EnsembleRegimeDetector::default_config();

        // Initially not ready
        assert!(!ensemble.is_ready());
        assert!(!ensemble.indicator_ready());
        assert!(!ensemble.hmm_ready());

        // Feed data
        let mut price = 100.0;
        for _ in 0..300 {
            price *= 1.001;
            ensemble.update(price * 1.01, price * 0.99, price);
        }

        // After enough data, should be ready
        assert!(ensemble.indicator_ready());
        // HMM readiness depends on min_observations config
    }

    #[test]
    fn test_hmm_state_probabilities_accessible() {
        let mut ensemble = EnsembleRegimeDetector::default_config();

        let mut price = 100.0;
        for _ in 0..100 {
            price *= 1.001;
            ensemble.update(price * 1.01, price * 0.99, price);
        }

        let probs = ensemble.hmm_state_probabilities();
        assert_eq!(probs.len(), 3, "Should have 3 HMM states");

        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "HMM state probs should sum to 1.0: {sum}"
        );
    }

    #[test]
    fn test_expected_regime_duration() {
        let ensemble = EnsembleRegimeDetector::default_config();
        let duration = ensemble.expected_regime_duration();
        assert!(duration > 0.0, "Duration should be > 0: {duration}");
    }

    #[test]
    fn test_detector_accessors() {
        let ensemble = EnsembleRegimeDetector::default_config();

        // Should be able to access underlying detectors
        assert!(!ensemble.indicator_detector().is_ready());
        assert!(!ensemble.hmm_detector().is_ready());
    }

    #[test]
    fn test_combine_results_agreement_boosts_confidence() {
        let ensemble = EnsembleRegimeDetector::default_config();

        let (_, conf_agree) = ensemble.combine_results(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.7,
            MarketRegime::Trending(TrendDirection::Bullish),
            0.7,
            true,
        );

        let (_, conf_disagree) = ensemble.combine_results(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.7,
            MarketRegime::MeanReverting,
            0.7,
            false,
        );

        assert!(
            conf_agree > conf_disagree,
            "Agreement should boost confidence: agree={conf_agree} vs disagree={conf_disagree}"
        );
    }

    #[test]
    fn test_combine_results_disagreement_returns_uncertain_at_low_conf() {
        let config = EnsembleConfig {
            agreement_threshold: 0.8,
            disagreement_confidence_penalty: 0.5,
            ..Default::default()
        };
        let ensemble = EnsembleRegimeDetector::new(config, RegimeConfig::default());

        let (regime, _) = ensemble.combine_results(
            MarketRegime::Trending(TrendDirection::Bullish),
            0.4,
            MarketRegime::MeanReverting,
            0.4,
            false,
        );

        assert_eq!(
            regime,
            MarketRegime::Uncertain,
            "Low confidence + disagreement should produce Uncertain"
        );
    }
}
