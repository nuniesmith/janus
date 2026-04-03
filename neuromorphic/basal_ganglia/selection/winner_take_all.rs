//! Competitive action selection
//!
//! Part of the Basal Ganglia region
//! Component: selection
//!
//! Implements a winner-take-all (WTA) network for competitive action
//! selection. Multiple candidate actions compete, and the one with
//! the strongest activation wins. Features:
//! - Softmax selection with configurable temperature
//! - Lateral inhibition between competing actions
//! - Optional noise injection for exploration
//! - History tracking to detect action perseveration

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Result of a winner-take-all competition
#[derive(Debug, Clone)]
pub struct WTAResult {
    /// Index of the winning action
    pub winner: usize,
    /// Activation level of the winner after competition
    pub winning_activation: f64,
    /// Softmax probability of the winner
    pub winning_probability: f64,
    /// All activations after competition (post-inhibition)
    pub final_activations: Vec<f64>,
    /// All softmax probabilities
    pub probabilities: Vec<f64>,
    /// Margin between winner and runner-up
    pub margin: f64,
    /// Whether the competition was decisive (margin > decisiveness threshold)
    pub decisive: bool,
}

/// Configuration for the WTA network
#[derive(Debug, Clone)]
pub struct WTAConfig {
    /// Number of competing actions
    pub num_actions: usize,
    /// Softmax temperature (lower = more greedy, higher = more exploratory)
    pub temperature: f64,
    /// Lateral inhibition strength (0.0 = none, 1.0 = full suppression)
    pub inhibition_strength: f64,
    /// Number of inhibition iterations per competition round
    pub inhibition_iterations: usize,
    /// Noise standard deviation for exploration (0.0 = deterministic)
    pub noise_std: f64,
    /// Minimum margin to consider a decision "decisive"
    pub decisiveness_threshold: f64,
    /// History window for perseveration detection
    pub history_window: usize,
    /// Perseveration penalty applied when the same action wins repeatedly
    pub perseveration_penalty: f64,
    /// Minimum temperature floor (prevents division by near-zero)
    pub min_temperature: f64,
    /// Temperature annealing rate per competition (0.0 = no annealing)
    pub annealing_rate: f64,
}

impl Default for WTAConfig {
    fn default() -> Self {
        Self {
            num_actions: 4,
            temperature: 1.0,
            inhibition_strength: 0.3,
            inhibition_iterations: 3,
            noise_std: 0.0,
            decisiveness_threshold: 0.15,
            history_window: 20,
            perseveration_penalty: 0.1,
            min_temperature: 0.01,
            annealing_rate: 0.0,
        }
    }
}

/// Simple linear congruential PRNG for deterministic noise generation
/// (avoids pulling in rand as a dependency)
struct SimplePrng {
    state: u64,
}

impl SimplePrng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generate a roughly normal-distributed value using Box-Muller approximation
    fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u2 = (self.next_u64() as f64) / (u64::MAX as f64);
        let u1 = u1.max(1e-15); // avoid log(0)
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Competitive action selection via winner-take-all dynamics
pub struct WinnerTakeAll {
    /// Configuration parameters
    config: WTAConfig,
    /// Current effective temperature
    effective_temperature: f64,
    /// Baseline activations (persistent biases per action)
    biases: Vec<f64>,
    /// History of winning actions (for perseveration detection)
    history: VecDeque<usize>,
    /// Total number of competitions run
    competition_count: u64,
    /// Win counts per action (for statistics)
    win_counts: Vec<u64>,
    /// PRNG for noise injection
    rng: SimplePrng,
}

impl Default for WinnerTakeAll {
    fn default() -> Self {
        Self::new()
    }
}

impl WinnerTakeAll {
    /// Create a new instance with default configuration (4 actions)
    pub fn new() -> Self {
        Self::with_config(WTAConfig::default())
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: WTAConfig) -> Self {
        let n = config.num_actions.max(1);
        Self {
            effective_temperature: config.temperature,
            biases: vec![0.0; n],
            history: VecDeque::with_capacity(config.history_window),
            competition_count: 0,
            win_counts: vec![0; n],
            rng: SimplePrng::new(42),
            config,
        }
    }

    /// Create a new instance with a specific number of actions
    pub fn with_actions(num_actions: usize) -> Self {
        Self::with_config(WTAConfig {
            num_actions,
            ..Default::default()
        })
    }

    /// Main processing function — validates state
    pub fn process(&self) -> Result<()> {
        if self.config.num_actions == 0 {
            return Err(Error::InvalidInput(
                "WinnerTakeAll requires at least 1 action".into(),
            ));
        }
        if self.effective_temperature < 0.0 {
            return Err(Error::InvalidState(
                "Temperature must be non-negative".into(),
            ));
        }
        Ok(())
    }

    /// Run a competition with the given raw activations.
    ///
    /// `activations` must have length equal to `config.num_actions`.
    /// Returns the result of the competition including winner, probabilities, etc.
    pub fn compete(&mut self, activations: &[f64]) -> Result<WTAResult> {
        if activations.len() != self.config.num_actions {
            return Err(Error::InvalidInput(format!(
                "Expected {} activations, got {}",
                self.config.num_actions,
                activations.len()
            )));
        }

        let _n = activations.len();

        // Step 1: Add biases and noise
        let mut acts: Vec<f64> = activations
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let noise = if self.config.noise_std > 0.0 {
                    self.rng.next_normal() * self.config.noise_std
                } else {
                    0.0
                };
                a + self.biases[i] + noise
            })
            .collect();

        // Step 2: Apply perseveration penalty
        if self.config.perseveration_penalty > 0.0 && !self.history.is_empty() {
            let recent_counts = self.recent_action_counts();
            let max_count = *recent_counts.iter().max().unwrap_or(&0) as f64;
            if max_count > 0.0 {
                for (i, act) in acts.iter_mut().enumerate() {
                    let freq = recent_counts[i] as f64 / max_count;
                    *act -= freq * self.config.perseveration_penalty;
                }
            }
        }

        // Step 3: Lateral inhibition iterations
        for _ in 0..self.config.inhibition_iterations {
            acts = self.apply_inhibition(&acts);
        }

        // Step 4: Compute softmax probabilities
        let temp = self.effective_temperature.max(self.config.min_temperature);
        let probabilities = Self::softmax(&acts, temp);

        // Step 5: Determine winner (argmax of probabilities)
        let (winner, &winning_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        // Step 6: Compute margin (difference between top two)
        let mut sorted_probs = probabilities.clone();
        sorted_probs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let margin = if sorted_probs.len() >= 2 {
            sorted_probs[0] - sorted_probs[1]
        } else {
            1.0
        };

        let decisive = margin >= self.config.decisiveness_threshold;

        // Step 7: Update history and stats
        if self.history.len() >= self.config.history_window {
            self.history.pop_front();
        }
        self.history.push_back(winner);
        self.win_counts[winner] += 1;
        self.competition_count += 1;

        // Step 8: Anneal temperature
        if self.config.annealing_rate > 0.0 {
            self.effective_temperature = (self.effective_temperature - self.config.annealing_rate)
                .max(self.config.min_temperature);
        }

        Ok(WTAResult {
            winner,
            winning_activation: acts[winner],
            winning_probability: winning_prob,
            final_activations: acts,
            probabilities,
            margin,
            decisive,
        })
    }

    /// Set a bias for a specific action (persistent activation offset)
    pub fn set_bias(&mut self, action: usize, bias: f64) -> Result<()> {
        if action >= self.config.num_actions {
            return Err(Error::InvalidInput(format!(
                "Action index {} out of range (max {})",
                action,
                self.config.num_actions - 1
            )));
        }
        self.biases[action] = bias;
        Ok(())
    }

    /// Set biases for all actions at once
    pub fn set_biases(&mut self, biases: &[f64]) -> Result<()> {
        if biases.len() != self.config.num_actions {
            return Err(Error::InvalidInput(format!(
                "Expected {} biases, got {}",
                self.config.num_actions,
                biases.len()
            )));
        }
        self.biases.copy_from_slice(biases);
        Ok(())
    }

    /// Set the temperature (overrides annealing)
    pub fn set_temperature(&mut self, temperature: f64) {
        self.effective_temperature = temperature.max(self.config.min_temperature);
    }

    /// Get the current effective temperature
    pub fn temperature(&self) -> f64 {
        self.effective_temperature
    }

    /// Get win counts per action
    pub fn win_counts(&self) -> &[u64] {
        &self.win_counts
    }

    /// Get the total number of competitions run
    pub fn competition_count(&self) -> u64 {
        self.competition_count
    }

    /// Get the win rate for a specific action
    pub fn win_rate(&self, action: usize) -> f64 {
        if action >= self.config.num_actions || self.competition_count == 0 {
            return 0.0;
        }
        self.win_counts[action] as f64 / self.competition_count as f64
    }

    /// Check if the system is perseverating (same action winning excessively)
    pub fn is_perseverating(&self) -> bool {
        if self.history.len() < 3 {
            return false;
        }
        let last = *self.history.back().unwrap();
        let streak = self
            .history
            .iter()
            .rev()
            .take_while(|&&a| a == last)
            .count();
        streak as f64 / self.history.len() as f64 > 0.8
    }

    /// Get the most recently winning action
    pub fn last_winner(&self) -> Option<usize> {
        self.history.back().copied()
    }

    /// Reset all state (history, counts, temperature)
    pub fn reset(&mut self) {
        self.history.clear();
        self.win_counts.fill(0);
        self.competition_count = 0;
        self.effective_temperature = self.config.temperature;
        self.biases.fill(0.0);
    }

    /// Seed the internal PRNG for reproducibility
    pub fn seed(&mut self, seed: u64) {
        self.rng = SimplePrng::new(seed);
    }

    // ── internal ──

    /// Apply one round of lateral inhibition
    fn apply_inhibition(&self, activations: &[f64]) -> Vec<f64> {
        let n = activations.len();
        if n <= 1 || self.config.inhibition_strength <= 0.0 {
            return activations.to_vec();
        }

        // Find the current maximum
        let _max_act = activations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // Each unit inhibits others proportionally to its activation
        let total: f64 = activations.iter().sum();
        let mean = total / n as f64;

        activations
            .iter()
            .map(|&a| {
                // Winner gets less inhibition, losers get more
                let inhibition = (mean - a) * self.config.inhibition_strength;
                let new_val = a - inhibition;
                // ReLU: activations can't go below 0 after inhibition
                new_val.max(0.0)
            })
            .collect()
    }

    /// Compute softmax probabilities with temperature
    fn softmax(values: &[f64], temperature: f64) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }

        let temp = temperature.max(1e-10);

        // Subtract max for numerical stability
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let exps: Vec<f64> = values
            .iter()
            .map(|&v| ((v - max_val) / temp).exp())
            .collect();
        let sum: f64 = exps.iter().sum();

        if sum <= 0.0 || !sum.is_finite() {
            // Fallback: uniform distribution
            let uniform = 1.0 / values.len() as f64;
            return vec![uniform; values.len()];
        }

        exps.iter().map(|&e| e / sum).collect()
    }

    /// Count recent action selections within the history window
    fn recent_action_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.config.num_actions];
        for &action in &self.history {
            if action < counts.len() {
                counts[action] += 1;
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = WinnerTakeAll::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_clear_winner() {
        let mut wta = WinnerTakeAll::with_actions(3);
        let result = wta.compete(&[0.1, 0.9, 0.2]).unwrap();
        assert_eq!(result.winner, 1);
        assert!(result.winning_probability > 0.4);
        assert!(result.margin > 0.0);
    }

    #[test]
    fn test_softmax_temperature() {
        // Low temperature → more decisive
        let probs_low = WinnerTakeAll::softmax(&[1.0, 2.0, 0.5], 0.1);
        // High temperature → more uniform
        let probs_high = WinnerTakeAll::softmax(&[1.0, 2.0, 0.5], 10.0);

        // Winner probability should be higher with low temperature
        let max_low = probs_low.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_high = probs_high.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_low > max_high,
            "Low temp max {} should be > high temp max {}",
            max_low,
            max_high
        );
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let probs = WinnerTakeAll::softmax(&[0.3, 0.5, 0.2, 0.8], 1.0);
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "probabilities should sum to 1, got {}",
            sum
        );
    }

    #[test]
    fn test_inhibition_amplifies_winner() {
        let wta = WinnerTakeAll::with_config(WTAConfig {
            num_actions: 3,
            inhibition_strength: 0.5,
            inhibition_iterations: 5,
            temperature: 1.0,
            noise_std: 0.0,
            ..Default::default()
        });

        let activations = vec![0.5, 0.9, 0.3];
        let mut result = activations.clone();
        for _ in 0..5 {
            result = wta.apply_inhibition(&result);
        }

        // The strongest should remain strongest
        let max_idx = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_idx, 1, "action 1 should still win after inhibition");
    }

    #[test]
    fn test_bias_affects_outcome() {
        let mut wta = WinnerTakeAll::with_config(WTAConfig {
            num_actions: 3,
            temperature: 0.1,
            noise_std: 0.0,
            inhibition_strength: 0.0,
            perseveration_penalty: 0.0,
            ..Default::default()
        });

        // Without bias, action 0 wins
        let result = wta.compete(&[1.0, 0.5, 0.5]).unwrap();
        assert_eq!(result.winner, 0);

        wta.reset();

        // With a strong bias on action 2, it should win
        wta.set_bias(2, 2.0).unwrap();
        let result = wta.compete(&[1.0, 0.5, 0.5]).unwrap();
        assert_eq!(result.winner, 2);
    }

    #[test]
    fn test_win_counts_and_rates() {
        let mut wta = WinnerTakeAll::with_config(WTAConfig {
            num_actions: 2,
            temperature: 0.01,
            noise_std: 0.0,
            inhibition_strength: 0.0,
            perseveration_penalty: 0.0,
            ..Default::default()
        });

        for _ in 0..5 {
            wta.compete(&[1.0, 0.1]).unwrap();
        }
        for _ in 0..5 {
            wta.compete(&[0.1, 1.0]).unwrap();
        }

        assert_eq!(wta.competition_count(), 10);
        assert_eq!(wta.win_counts()[0], 5);
        assert_eq!(wta.win_counts()[1], 5);
        assert!((wta.win_rate(0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_wrong_activation_count_returns_error() {
        let mut wta = WinnerTakeAll::with_actions(3);
        let result = wta.compete(&[0.1, 0.2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_annealing() {
        let mut wta = WinnerTakeAll::with_config(WTAConfig {
            num_actions: 2,
            temperature: 1.0,
            annealing_rate: 0.1,
            min_temperature: 0.2,
            noise_std: 0.0,
            inhibition_strength: 0.0,
            perseveration_penalty: 0.0,
            ..Default::default()
        });

        let initial_temp = wta.temperature();
        for _ in 0..5 {
            wta.compete(&[0.5, 0.5]).unwrap();
        }

        assert!(
            wta.temperature() < initial_temp,
            "temperature should decrease"
        );
        assert!(
            wta.temperature() >= 0.2,
            "temperature should not go below min"
        );
    }

    #[test]
    fn test_reset() {
        let mut wta = WinnerTakeAll::with_config(WTAConfig {
            num_actions: 2,
            annealing_rate: 0.1,
            noise_std: 0.0,
            inhibition_strength: 0.0,
            perseveration_penalty: 0.0,
            ..Default::default()
        });

        for _ in 0..5 {
            wta.compete(&[0.8, 0.2]).unwrap();
        }

        wta.reset();
        assert_eq!(wta.competition_count(), 0);
        assert_eq!(wta.win_counts(), &[0, 0]);
        assert_eq!(wta.last_winner(), None);
        assert!((wta.temperature() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_decisive_flag() {
        let mut wta = WinnerTakeAll::with_config(WTAConfig {
            num_actions: 2,
            temperature: 0.1,
            decisiveness_threshold: 0.3,
            noise_std: 0.0,
            inhibition_strength: 0.0,
            perseveration_penalty: 0.0,
            ..Default::default()
        });

        // Very different activations → decisive
        let result = wta.compete(&[0.1, 1.0]).unwrap();
        assert!(result.decisive, "large gap should be decisive");
    }

    #[test]
    fn test_single_action() {
        let mut wta = WinnerTakeAll::with_actions(1);
        let result = wta.compete(&[0.5]).unwrap();
        assert_eq!(result.winner, 0);
        assert!((result.winning_probability - 1.0).abs() < 1e-9);
    }
}
