//! Hidden Markov Model Regime Detection
//!
//! Implements HMM-based regime detection as described in:
//! - Hamilton, J.D. (1989) "A New Approach to the Economic Analysis of Nonstationary Time Series"
//!
//! The HMM approach learns regime distributions directly from returns data,
//! making no assumptions about what indicators define each regime.
//!
//! Ported from kraken's `regime/hmm.rs`, adapted for the JANUS type system.
//! Uses stable Rust only (no nightly features).

use super::types::{MarketRegime, RegimeConfidence, TrendDirection};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for HMM regime detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HMMConfig {
    /// Number of hidden states (regimes)
    pub n_states: usize,
    /// Minimum observations before making predictions
    pub min_observations: usize,
    /// Learning rate for online updates (0 = no online learning)
    pub learning_rate: f64,
    /// Smoothing factor for transition probabilities
    pub transition_smoothing: f64,
    /// Window size for return calculations
    pub lookback_window: usize,
    /// Confidence threshold for regime classification
    pub min_confidence: f64,
}

impl Default for HMMConfig {
    fn default() -> Self {
        Self {
            n_states: 3, // Bull, Bear, High-Vol
            min_observations: 100,
            learning_rate: 0.01,
            transition_smoothing: 0.1,
            lookback_window: 252, // ~1 year of daily data
            min_confidence: 0.6,
        }
    }
}

impl HMMConfig {
    /// Config optimized for crypto (faster regime changes)
    pub fn crypto_optimized() -> Self {
        Self {
            n_states: 3,
            min_observations: 50,
            learning_rate: 0.02, // Faster adaptation
            transition_smoothing: 0.05,
            lookback_window: 100,
            min_confidence: 0.5,
        }
    }

    /// Conservative config (more stable regimes)
    pub fn conservative() -> Self {
        Self {
            n_states: 2, // Just bull/bear
            min_observations: 150,
            learning_rate: 0.005,
            transition_smoothing: 0.15,
            lookback_window: 500,
            min_confidence: 0.7,
        }
    }
}

/// Gaussian parameters for a single hidden state
#[derive(Debug, Clone)]
struct GaussianState {
    mean: f64,
    variance: f64,
    /// Running statistics for online updates
    sum: f64,
    sum_sq: f64,
    count: usize,
}

impl GaussianState {
    fn new(mean: f64, variance: f64) -> Self {
        Self {
            mean,
            variance,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
        }
    }

    /// Probability density function
    fn pdf(&self, x: f64) -> f64 {
        let diff = x - self.mean;
        let exponent = -0.5 * diff * diff / self.variance;
        let normalizer = (2.0 * std::f64::consts::PI * self.variance).sqrt();
        exponent.exp() / normalizer
    }

    /// Update statistics with new observation
    fn update(&mut self, x: f64, weight: f64, learning_rate: f64) {
        if learning_rate > 0.0 {
            // Online update using exponential moving average
            self.mean = (1.0 - learning_rate * weight) * self.mean + learning_rate * weight * x;
            let new_var = (x - self.mean).powi(2);
            self.variance =
                (1.0 - learning_rate * weight) * self.variance + learning_rate * weight * new_var;
            self.variance = self.variance.max(1e-8); // Prevent zero variance
        }

        // Also track running stats
        self.sum += x * weight;
        self.sum_sq += x * x * weight;
        self.count += 1;
    }
}

/// Hidden Markov Model for regime detection.
///
/// Uses a 3-state HMM (by default) to model market regimes:
/// - State 0: Bull market (positive returns, low volatility)
/// - State 1: Bear market (negative returns, medium volatility)
/// - State 2: High volatility (any direction, high volatility)
///
/// The model uses the forward algorithm for online filtering and periodically
/// re-estimates parameters using the Baum-Welch algorithm.
///
/// # Example
///
/// ```rust
/// use janus_regime::{HMMRegimeDetector, HMMConfig, MarketRegime};
///
/// let mut detector = HMMRegimeDetector::crypto_optimized();
///
/// // Feed close prices
/// for i in 0..200 {
///     let price = 100.0 * (1.0 + 0.001 * i as f64); // gentle uptrend
///     let result = detector.update(price);
///     if detector.is_ready() {
///         println!("HMM regime: {} (conf: {:.0}%)", result.regime, result.confidence * 100.0);
///     }
/// }
/// ```
#[derive(Debug)]
pub struct HMMRegimeDetector {
    config: HMMConfig,

    /// Gaussian emission distributions for each state
    states: Vec<GaussianState>,

    /// Transition probability matrix A[i][j] = P(state_j | state_i)
    transition_matrix: Vec<Vec<f64>>,

    /// Initial state probabilities
    initial_probs: Vec<f64>,

    /// Current state probabilities (filtered)
    state_probs: Vec<f64>,

    /// History of returns for batch updates
    returns_history: VecDeque<f64>,

    /// History of prices for return calculation
    prices: VecDeque<f64>,

    /// Current most likely state
    current_state: usize,

    /// Confidence in current state
    current_confidence: f64,

    /// Total observations processed
    n_observations: usize,

    /// Last detected regime
    #[allow(dead_code)]
    last_regime: MarketRegime,
}

impl HMMRegimeDetector {
    /// Create a new HMM detector with the given configuration
    pub fn new(config: HMMConfig) -> Self {
        let n = config.n_states;

        // Initialize states with reasonable priors for financial returns
        // State 0: Bull (positive returns, low vol)
        // State 1: Bear (negative returns, higher vol)
        // State 2: High Vol (any direction, high vol)
        let states = match n {
            2 => vec![
                GaussianState::new(0.001, 0.0001),  // Bull: ~0.1% daily, low vol
                GaussianState::new(-0.001, 0.0004), // Bear: -0.1% daily, higher vol
            ],
            3 => vec![
                GaussianState::new(0.001, 0.0001),  // Bull: positive, low vol
                GaussianState::new(-0.001, 0.0002), // Bear: negative, medium vol
                GaussianState::new(0.0, 0.0009),    // High Vol: neutral, high vol
            ],
            _ => (0..n)
                .map(|i| {
                    let mean = (i as f64 - n as f64 / 2.0) * 0.001;
                    let var = 0.0001 * (1.0 + i as f64);
                    GaussianState::new(mean, var)
                })
                .collect(),
        };

        // Initialize transition matrix with slight persistence
        // Higher diagonal = states tend to persist
        let mut transition_matrix = vec![vec![0.0; n]; n];
        for (i, row) in transition_matrix.iter_mut().enumerate().take(n) {
            for (j, cell) in row.iter_mut().enumerate().take(n) {
                if i == j {
                    *cell = 0.9; // 90% stay in same state
                } else {
                    *cell = 0.1 / (n - 1) as f64;
                }
            }
        }

        // Equal initial probabilities
        let initial_probs = vec![1.0 / n as f64; n];
        let state_probs = initial_probs.clone();

        Self {
            config: config.clone(),
            states,
            transition_matrix,
            initial_probs,
            state_probs,
            returns_history: VecDeque::with_capacity(config.lookback_window),
            prices: VecDeque::with_capacity(10),
            current_state: 0,
            current_confidence: 0.0,
            n_observations: 0,
            last_regime: MarketRegime::Uncertain,
        }
    }

    /// Create with default config
    pub fn default_config() -> Self {
        Self::new(HMMConfig::default())
    }

    /// Create optimized for crypto
    pub fn crypto_optimized() -> Self {
        Self::new(HMMConfig::crypto_optimized())
    }

    /// Create with conservative config
    pub fn conservative() -> Self {
        Self::new(HMMConfig::conservative())
    }

    /// Update with new close price and get regime.
    ///
    /// Calculates log return from the previous close, then runs the forward
    /// algorithm step and optional parameter updates.
    pub fn update(&mut self, close: f64) -> RegimeConfidence {
        // Calculate log return
        if let Some(&prev_close) = self.prices.back()
            && prev_close > 0.0
        {
            let log_return = (close / prev_close).ln();
            self.process_return(log_return);
        }

        // Store price
        self.prices.push_back(close);
        if self.prices.len() > 10 {
            self.prices.pop_front();
        }

        // Return current regime
        self.get_regime_confidence()
    }

    /// Update with OHLC data (uses close price for HMM)
    pub fn update_ohlc(&mut self, _high: f64, _low: f64, close: f64) -> RegimeConfidence {
        self.update(close)
    }

    /// Process a single return observation
    fn process_return(&mut self, ret: f64) {
        self.n_observations += 1;

        // Store return
        self.returns_history.push_back(ret);
        if self.returns_history.len() > self.config.lookback_window {
            self.returns_history.pop_front();
        }

        // Forward algorithm step (filtering)
        self.forward_step(ret);

        // Update state parameters if we have enough data
        if self.n_observations > self.config.min_observations && self.config.learning_rate > 0.0 {
            self.online_parameter_update(ret);
        }

        // Periodically re-estimate with Baum-Welch if we have enough data
        let reestimate_interval = self.config.lookback_window / 2;
        if self.n_observations > 0
            && reestimate_interval > 0
            && self.n_observations.is_multiple_of(reestimate_interval)
            && self.returns_history.len() >= self.config.min_observations
        {
            self.baum_welch_update();
        }
    }

    /// Forward algorithm step - update state probabilities given new observation
    fn forward_step(&mut self, ret: f64) {
        let n = self.config.n_states;
        let mut new_probs = vec![0.0; n];

        // Calculate emission probabilities
        let emissions: Vec<f64> = self.states.iter().map(|s| s.pdf(ret)).collect();

        // Forward step: P(state_j | obs) ∝ P(obs | state_j) * Σᵢ P(state_j | state_i) * P(state_i)
        for j in 0..n {
            let mut sum = 0.0;
            for i in 0..n {
                sum += self.transition_matrix[i][j] * self.state_probs[i];
            }
            new_probs[j] = emissions[j] * sum;
        }

        // Normalize
        let total: f64 = new_probs.iter().sum();
        if total > 1e-300 {
            for p in &mut new_probs {
                *p /= total;
            }
        } else {
            // Reset to uniform if probabilities collapse
            new_probs = vec![1.0 / n as f64; n];
        }

        self.state_probs = new_probs;

        // Update current state and confidence
        let (max_idx, max_prob) = self
            .state_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        self.current_state = max_idx;
        self.current_confidence = *max_prob;
    }

    /// Online parameter update using soft assignments
    fn online_parameter_update(&mut self, ret: f64) {
        let lr = self.config.learning_rate;

        for (i, state) in self.states.iter_mut().enumerate() {
            let weight = self.state_probs[i];
            state.update(ret, weight, lr);
        }

        // Update transition matrix (soft transitions)
        // This is a simplified online update
        let smoothing = self.config.transition_smoothing;
        for i in 0..self.config.n_states {
            for j in 0..self.config.n_states {
                let target = if i == j {
                    0.9
                } else {
                    0.1 / (self.config.n_states - 1) as f64
                };
                self.transition_matrix[i][j] =
                    (1.0 - smoothing) * self.transition_matrix[i][j] + smoothing * target;
            }
        }
    }

    /// Baum-Welch algorithm for batch parameter re-estimation.
    ///
    /// Runs the full forward-backward algorithm on the returns history
    /// to re-estimate emission parameters. Uses blending with existing
    /// parameters to prevent sudden jumps.
    fn baum_welch_update(&mut self) {
        let returns: Vec<f64> = self.returns_history.iter().copied().collect();
        if returns.len() < self.config.min_observations {
            return;
        }

        let n = self.config.n_states;
        let t = returns.len();

        // Forward pass
        let mut alpha = vec![vec![0.0; n]; t];

        // Initialize
        for (j, alpha_val) in alpha[0].iter_mut().enumerate().take(n) {
            *alpha_val = self.initial_probs[j] * self.states[j].pdf(returns[0]);
        }
        Self::normalize_vec(&mut alpha[0]);

        // Forward
        for time in 1..t {
            for j in 0..n {
                let mut sum = 0.0;
                for (i, alpha_prev) in alpha[time - 1].iter().enumerate().take(n) {
                    sum += alpha_prev * self.transition_matrix[i][j];
                }
                alpha[time][j] = sum * self.states[j].pdf(returns[time]);
            }
            Self::normalize_vec(&mut alpha[time]);
        }

        // Backward pass
        let mut beta = vec![vec![1.0; n]; t];

        for time in (0..t - 1).rev() {
            for i in 0..n {
                let mut sum = 0.0;
                for (j, beta_next) in beta[time + 1].iter().enumerate().take(n) {
                    sum += self.transition_matrix[i][j]
                        * self.states[j].pdf(returns[time + 1])
                        * beta_next;
                }
                beta[time][i] = sum;
            }
            Self::normalize_vec(&mut beta[time]);
        }

        // Compute gamma (state occupancy probabilities)
        let mut gamma = vec![vec![0.0; n]; t];
        for time in 0..t {
            let mut sum = 0.0;
            for (j, gamma_val) in gamma[time].iter_mut().enumerate().take(n) {
                *gamma_val = alpha[time][j] * beta[time][j];
                sum += *gamma_val;
            }
            if sum > 1e-300 {
                for gamma_val in gamma[time].iter_mut().take(n) {
                    *gamma_val /= sum;
                }
            }
        }

        // Re-estimate emission parameters
        for (j, state) in self.states.iter_mut().enumerate().take(n) {
            let mut weight_sum = 0.0;
            let mut mean_sum = 0.0;
            let mut var_sum = 0.0;

            for time in 0..t {
                let w = gamma[time][j];
                weight_sum += w;
                mean_sum += w * returns[time];
            }

            if weight_sum > 1e-8 {
                let new_mean = mean_sum / weight_sum;

                for time in 0..t {
                    let w = gamma[time][j];
                    var_sum += w * (returns[time] - new_mean).powi(2);
                }

                let new_var = (var_sum / weight_sum).max(1e-8);

                // Blend with existing parameters (prevents sudden jumps)
                let blend = 0.3;
                state.mean = (1.0 - blend) * state.mean + blend * new_mean;
                state.variance = (1.0 - blend) * state.variance + blend * new_var;
            }
        }
    }

    /// Helper to normalize a probability vector
    fn normalize_vec(vec: &mut [f64]) {
        let sum: f64 = vec.iter().sum();
        if sum > 1e-300 {
            for v in vec.iter_mut() {
                *v /= sum;
            }
        }
    }

    /// Get current regime with confidence
    pub fn get_regime_confidence(&self) -> RegimeConfidence {
        if self.n_observations < self.config.min_observations {
            return RegimeConfidence::new(MarketRegime::Uncertain, 0.0);
        }

        let regime = self.state_to_regime(self.current_state);
        let confidence = self.current_confidence;

        RegimeConfidence::with_metrics(
            regime,
            confidence,
            self.states[self.current_state].mean * 100.0 * 252.0, // Annualized return %
            self.states[self.current_state].variance.sqrt() * 100.0 * 252.0_f64.sqrt(), // Annualized vol %
            0.0, // No trend strength in HMM
        )
    }

    /// Map state index to `MarketRegime` based on learned parameters.
    ///
    /// Classification is based on the Gaussian emission parameters:
    /// - High variance → Volatile
    /// - Positive mean → Trending(Bullish)
    /// - Negative mean → Trending(Bearish)
    /// - Low variance, neutral mean → MeanReverting
    fn state_to_regime(&self, state: usize) -> MarketRegime {
        let state_params = &self.states[state];
        let mean = state_params.mean;
        let vol = state_params.variance.sqrt();

        // Classify based on learned parameters
        let is_high_vol = vol > 0.02; // > 2% daily vol
        let is_positive = mean > 0.0005; // > 0.05% daily
        let is_negative = mean < -0.0005;

        if is_high_vol {
            MarketRegime::Volatile
        } else if is_positive {
            MarketRegime::Trending(TrendDirection::Bullish)
        } else if is_negative {
            MarketRegime::Trending(TrendDirection::Bearish)
        } else {
            MarketRegime::MeanReverting // Low vol, neutral returns = ranging
        }
    }

    // ========================================================================
    // Public Accessors
    // ========================================================================

    /// Get state probabilities
    pub fn state_probabilities(&self) -> &[f64] {
        &self.state_probs
    }

    /// Get state parameters (mean, variance) for inspection
    pub fn state_parameters(&self) -> Vec<(f64, f64)> {
        self.states.iter().map(|s| (s.mean, s.variance)).collect()
    }

    /// Get transition matrix
    pub fn transition_matrix(&self) -> &[Vec<f64>] {
        &self.transition_matrix
    }

    /// Get current state index
    pub fn current_state_index(&self) -> usize {
        self.current_state
    }

    /// Check if model is warmed up (has enough observations)
    pub fn is_ready(&self) -> bool {
        self.n_observations >= self.config.min_observations
    }

    /// Get expected regime duration (from transition matrix).
    ///
    /// Expected duration = 1 / (1 - P(stay in state))
    pub fn expected_regime_duration(&self, state: usize) -> f64 {
        if state < self.config.n_states {
            1.0 / (1.0 - self.transition_matrix[state][state])
        } else {
            0.0
        }
    }

    /// Predict most likely next state
    pub fn predict_next_state(&self) -> (usize, f64) {
        let mut next_probs = vec![0.0; self.config.n_states];

        for (j, next_prob) in next_probs.iter_mut().enumerate().take(self.config.n_states) {
            for i in 0..self.config.n_states {
                *next_prob += self.transition_matrix[i][j] * self.state_probs[i];
            }
        }

        let (max_idx, max_prob) = next_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        (max_idx, *max_prob)
    }

    /// Get the total number of observations processed
    pub fn n_observations(&self) -> usize {
        self.n_observations
    }

    /// Get the current confidence score
    pub fn current_confidence(&self) -> f64 {
        self.current_confidence
    }

    /// Get the configuration
    pub fn config(&self) -> &HMMConfig {
        &self.config
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_initialization() {
        let detector = HMMRegimeDetector::default_config();
        assert!(!detector.is_ready());
        assert_eq!(detector.state_probabilities().len(), 3);
    }

    #[test]
    fn test_hmm_crypto_config() {
        let detector = HMMRegimeDetector::crypto_optimized();
        assert_eq!(detector.config().n_states, 3);
        assert_eq!(detector.config().min_observations, 50);
    }

    #[test]
    fn test_hmm_conservative_config() {
        let detector = HMMRegimeDetector::conservative();
        assert_eq!(detector.config().n_states, 2);
        assert_eq!(detector.config().min_observations, 150);
        assert_eq!(detector.state_probabilities().len(), 2);
    }

    #[test]
    fn test_hmm_warmup() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        // Feed fewer than min_observations
        for i in 0..49 {
            let price = 100.0 + (i as f64) * 0.01;
            let result = detector.update(price);
            assert_eq!(
                result.regime,
                MarketRegime::Uncertain,
                "Should be Uncertain during warmup at step {i}"
            );
        }

        assert!(!detector.is_ready());
    }

    #[test]
    fn test_hmm_becomes_ready() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        for i in 0..60 {
            let price = 100.0 + (i as f64) * 0.01;
            detector.update(price);
        }

        assert!(detector.is_ready(), "Should be ready after 60 observations");
    }

    #[test]
    fn test_bull_market_detection() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        // Strong consistent uptrend
        let mut price = 100.0;
        for _ in 0..200 {
            price *= 1.005; // 0.5% daily gain
            let result = detector.update(price);
            if detector.is_ready() {
                // After warmup, regime should be trending or at least not uncertain
                assert_ne!(result.regime, MarketRegime::Uncertain);
            }
        }

        let final_result = detector.get_regime_confidence();
        // In a strong bull market, we expect bullish trending
        assert!(
            matches!(
                final_result.regime,
                MarketRegime::Trending(TrendDirection::Bullish)
            ),
            "Expected Bullish trend, got: {:?}",
            final_result.regime
        );
    }

    #[test]
    fn test_volatile_market_detection() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        // High volatility: large alternating swings
        let mut price = 100.0;
        for i in 0..200 {
            if i % 2 == 0 {
                price *= 1.05; // 5% up
            } else {
                price *= 0.95; // 5% down
            }
            detector.update(price);
        }

        let result = detector.get_regime_confidence();
        // With large swings, should detect volatile or at least not a clean trend
        assert!(
            matches!(
                result.regime,
                MarketRegime::Volatile | MarketRegime::MeanReverting
            ),
            "Expected Volatile or MeanReverting for choppy data, got: {:?}",
            result.regime
        );
    }

    #[test]
    fn test_state_probabilities_sum_to_one() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        let mut price = 100.0;
        for _ in 0..100 {
            price *= 1.001;
            detector.update(price);

            let probs = detector.state_probabilities();
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "State probabilities should sum to 1.0, got: {sum}"
            );
        }
    }

    #[test]
    fn test_transition_matrix_rows_sum_to_one() {
        let detector = HMMRegimeDetector::default_config();
        let tm = detector.transition_matrix();

        for (i, row) in tm.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Transition matrix row {i} should sum to 1.0, got: {sum}"
            );
        }
    }

    #[test]
    fn test_expected_regime_duration() {
        let detector = HMMRegimeDetector::default_config();

        // With 0.9 persistence, expected duration = 1 / (1 - 0.9) = 10
        let duration = detector.expected_regime_duration(0);
        assert!(
            (duration - 10.0).abs() < 1e-6,
            "Expected duration should be ~10 with 0.9 persistence, got: {duration}"
        );
    }

    #[test]
    fn test_predict_next_state() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        let mut price = 100.0;
        for _ in 0..100 {
            price *= 1.002;
            detector.update(price);
        }

        let (next_state, prob) = detector.predict_next_state();
        assert!(next_state < detector.config().n_states);
        assert!(
            (0.0..=1.0).contains(&prob),
            "Predicted probability should be in [0, 1]: {prob}"
        );
    }

    #[test]
    fn test_state_parameters() {
        let detector = HMMRegimeDetector::default_config();
        let params = detector.state_parameters();

        assert_eq!(params.len(), 3, "Should have 3 state parameters");

        for (mean, variance) in &params {
            assert!(variance > &0.0, "Variance should be positive: {variance}");
            assert!(mean.is_finite(), "Mean should be finite: {mean}");
        }
    }

    #[test]
    fn test_update_ohlc_uses_close() {
        let mut det1 = HMMRegimeDetector::crypto_optimized();
        let mut det2 = HMMRegimeDetector::crypto_optimized();

        // Both should produce identical results since OHLC just uses close
        for i in 0..100 {
            let close = 100.0 + i as f64 * 0.1;
            let r1 = det1.update(close);
            let r2 = det2.update_ohlc(close * 1.01, close * 0.99, close);

            assert_eq!(
                r1.regime, r2.regime,
                "update and update_ohlc should produce same regime"
            );
        }
    }

    #[test]
    fn test_n_observations_tracking() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        assert_eq!(detector.n_observations(), 0);

        for i in 0..50 {
            detector.update(100.0 + i as f64);
        }

        // n_observations counts returns, so it's prices - 1
        assert_eq!(detector.n_observations(), 49);
    }

    #[test]
    fn test_confidence_range() {
        let mut detector = HMMRegimeDetector::crypto_optimized();

        let mut price = 100.0;
        for _ in 0..200 {
            price *= 1.002;
            detector.update(price);
        }

        let confidence = detector.current_confidence();
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Confidence should be in [0, 1]: {confidence}"
        );
    }
}
