//! Fear Network - Fear-conditioned inhibition using FNI-RL
//!
//! Implements fear-conditioned neural inhibition for risk-aware trading decisions.
//! The network learns to associate market conditions with potential losses,
//! creating learned fear responses that can inhibit risky actions.
//!
//! Key concepts:
//! - Fear conditioning: Learning to associate market patterns with losses
//! - Fear extinction: Gradually reducing fear when conditions improve
//! - Fear generalization: Applying learned fears to similar situations
//! - Fear inhibition: Using learned fears to modulate trading decisions

use crate::common::Result;
use std::collections::HashMap;

/// Configuration for the fear network
#[derive(Debug, Clone)]
pub struct FearNetworkConfig {
    /// Learning rate for fear acquisition
    pub acquisition_rate: f64,
    /// Learning rate for fear extinction
    pub extinction_rate: f64,
    /// Decay rate for fear memories over time
    pub memory_decay: f64,
    /// Threshold for fear activation (0.0 - 1.0)
    pub activation_threshold: f64,
    /// Maximum fear level (caps fear response)
    pub max_fear: f64,
    /// Generalization width (how similar patterns trigger fear)
    pub generalization_sigma: f64,
    /// Number of fear memory slots
    pub memory_capacity: usize,
    /// Minimum fear level to maintain (prevents complete extinction)
    pub floor_fear: f64,
}

impl Default for FearNetworkConfig {
    fn default() -> Self {
        Self {
            acquisition_rate: 0.3,
            extinction_rate: 0.1,
            memory_decay: 0.995,
            activation_threshold: 0.3,
            max_fear: 1.0,
            generalization_sigma: 0.2,
            memory_capacity: 1000,
            floor_fear: 0.01,
        }
    }
}

/// A fear memory associating a pattern with fear level
#[derive(Debug, Clone)]
pub struct FearMemory {
    /// Feature pattern that triggered fear
    pub pattern: Vec<f64>,
    /// Current fear level (0.0 - 1.0)
    pub fear_level: f64,
    /// Number of times this fear was reinforced
    pub reinforcement_count: u32,
    /// Timestamp of last update
    pub last_update: i64,
    /// Original loss that created this fear
    pub original_loss: f64,
    /// Category/label for this fear memory
    pub category: String,
}

impl FearMemory {
    pub fn new(pattern: Vec<f64>, initial_fear: f64, loss: f64, category: String) -> Self {
        Self {
            pattern,
            fear_level: initial_fear,
            reinforcement_count: 1,
            last_update: chrono::Utc::now().timestamp_millis(),
            original_loss: loss,
            category,
        }
    }

    /// Calculate similarity to another pattern using Gaussian kernel
    pub fn similarity(&self, other: &[f64], sigma: f64) -> f64 {
        if self.pattern.len() != other.len() {
            return 0.0;
        }

        let squared_dist: f64 = self
            .pattern
            .iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();

        (-squared_dist / (2.0 * sigma * sigma)).exp()
    }
}

/// Fear response from the network
#[derive(Debug, Clone)]
pub struct FearResponse {
    /// Overall fear level (0.0 - 1.0)
    pub fear_level: f64,
    /// Whether fear threshold is exceeded
    pub is_activated: bool,
    /// Top contributing fear memories
    pub contributing_memories: Vec<(String, f64)>,
    /// Recommended inhibition strength (0.0 - 1.0)
    pub inhibition_strength: f64,
    /// Fear trend (positive = increasing, negative = decreasing)
    pub trend: f64,
}

impl Default for FearResponse {
    fn default() -> Self {
        Self {
            fear_level: 0.0,
            is_activated: false,
            contributing_memories: Vec::new(),
            inhibition_strength: 0.0,
            trend: 0.0,
        }
    }
}

/// Market state input for fear processing
#[derive(Debug, Clone)]
pub struct MarketState {
    /// Feature vector representing current market
    pub features: Vec<f64>,
    /// Current volatility
    pub volatility: f64,
    /// Recent return (negative = loss)
    pub recent_return: f64,
    /// Drawdown from peak
    pub drawdown: f64,
    /// Timestamp
    pub timestamp: i64,
}

/// Fear conditioning event
#[derive(Debug, Clone)]
pub struct ConditioningEvent {
    /// Market state when loss occurred
    pub state: MarketState,
    /// Loss magnitude (positive value)
    pub loss: f64,
    /// Category/label for this event
    pub category: String,
}

/// Fear-conditioned inhibition neural network
pub struct FearNetwork {
    config: FearNetworkConfig,
    /// Stored fear memories
    memories: Vec<FearMemory>,
    /// Recent fear levels for trend calculation
    fear_history: std::collections::VecDeque<f64>,
    /// Global fear baseline (ambient fear level)
    baseline_fear: f64,
    /// Total conditioning events
    conditioning_count: u64,
    /// Total extinction events
    extinction_count: u64,
    /// Fear by category
    category_fear: HashMap<String, f64>,
}

impl Default for FearNetwork {
    fn default() -> Self {
        Self::new()
    }
}

impl FearNetwork {
    /// Create a new FearNetwork with default configuration
    pub fn new() -> Self {
        Self::with_config(FearNetworkConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: FearNetworkConfig) -> Self {
        Self {
            memories: Vec::with_capacity(config.memory_capacity),
            fear_history: std::collections::VecDeque::with_capacity(100),
            baseline_fear: 0.0,
            conditioning_count: 0,
            extinction_count: 0,
            category_fear: HashMap::new(),
            config,
        }
    }

    /// Condition the network with a fear-inducing event (loss occurred)
    pub fn condition(&mut self, event: ConditioningEvent) {
        let loss = event.loss.abs();
        let initial_fear = (loss * self.config.acquisition_rate).min(self.config.max_fear);

        // Check if similar memory exists
        let mut found_similar = false;
        for memory in &mut self.memories {
            let similarity =
                memory.similarity(&event.state.features, self.config.generalization_sigma);
            if similarity > 0.8 {
                // Reinforce existing memory
                memory.fear_level =
                    (memory.fear_level + initial_fear * similarity).min(self.config.max_fear);
                memory.reinforcement_count += 1;
                memory.last_update = chrono::Utc::now().timestamp_millis();
                found_similar = true;
                break;
            }
        }

        // Create new memory if no similar one exists
        if !found_similar {
            let memory = FearMemory::new(
                event.state.features.clone(),
                initial_fear,
                loss,
                event.category.clone(),
            );

            // Manage capacity
            if self.memories.len() >= self.config.memory_capacity {
                // Remove oldest, least reinforced memory
                if let Some(idx) = self.find_weakest_memory() {
                    self.memories.remove(idx);
                }
            }

            self.memories.push(memory);
        }

        // Update category fear
        let category_fear = self.category_fear.entry(event.category).or_insert(0.0);
        *category_fear = (*category_fear + initial_fear * 0.5).min(self.config.max_fear);

        // Update baseline
        self.baseline_fear = (self.baseline_fear + initial_fear * 0.1).min(0.5);

        self.conditioning_count += 1;
    }

    /// Extinguish fear when positive outcome occurs (no loss despite fear)
    pub fn extinguish(&mut self, state: &MarketState, strength: f64) {
        let extinction_amount = strength * self.config.extinction_rate;

        for memory in &mut self.memories {
            let similarity = memory.similarity(&state.features, self.config.generalization_sigma);
            if similarity > 0.3 {
                memory.fear_level = (memory.fear_level - extinction_amount * similarity)
                    .max(self.config.floor_fear);
                memory.last_update = chrono::Utc::now().timestamp_millis();
            }
        }

        // Reduce baseline fear
        self.baseline_fear = (self.baseline_fear - extinction_amount * 0.1).max(0.0);

        // Reduce category fears
        for fear in self.category_fear.values_mut() {
            *fear = (*fear - extinction_amount * 0.1).max(0.0);
        }

        self.extinction_count += 1;
    }

    /// Evaluate current fear level given market state
    pub fn evaluate(&mut self, state: &MarketState) -> FearResponse {
        // Apply memory decay
        self.decay_memories();

        // Calculate fear from generalized memories
        let mut total_fear = self.baseline_fear;
        let mut contributing: Vec<(String, f64)> = Vec::new();
        let mut total_weight = 1.0; // Start with baseline

        for memory in &self.memories {
            let similarity = memory.similarity(&state.features, self.config.generalization_sigma);
            if similarity > 0.1 {
                let contribution = memory.fear_level * similarity;
                total_fear += contribution;
                total_weight += similarity;

                if contribution > 0.05 {
                    contributing.push((memory.category.clone(), contribution));
                }
            }
        }

        // Normalize by weights
        if total_weight > 0.0 {
            total_fear /= total_weight;
        }

        // Add situational fear adjustments
        total_fear = self.adjust_for_conditions(total_fear, state);

        // Clamp to valid range
        total_fear = total_fear.clamp(0.0, self.config.max_fear);

        // Calculate trend
        let trend = self.calculate_trend(total_fear);

        // Sort contributions
        contributing.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        contributing.truncate(5);

        // Calculate inhibition strength (non-linear transformation)
        let inhibition_strength = self.calculate_inhibition(total_fear);

        // Update history
        if self.fear_history.len() >= 100 {
            self.fear_history.pop_front();
        }
        self.fear_history.push_back(total_fear);

        FearResponse {
            fear_level: total_fear,
            is_activated: total_fear >= self.config.activation_threshold,
            contributing_memories: contributing,
            inhibition_strength,
            trend,
        }
    }

    /// Adjust fear based on current market conditions
    fn adjust_for_conditions(&self, base_fear: f64, state: &MarketState) -> f64 {
        let mut fear = base_fear;

        // High volatility increases fear
        if state.volatility > 0.02 {
            fear *= 1.0 + (state.volatility - 0.02) * 10.0;
        }

        // Drawdown increases fear
        if state.drawdown > 0.05 {
            fear *= 1.0 + state.drawdown * 2.0;
        }

        // Recent losses increase fear
        if state.recent_return < -0.01 {
            fear *= 1.0 + state.recent_return.abs() * 5.0;
        }

        fear
    }

    /// Calculate inhibition strength from fear level
    fn calculate_inhibition(&self, fear: f64) -> f64 {
        // Sigmoid-like transformation
        // Low fear = low inhibition, high fear = strong inhibition
        let normalized = (fear - self.config.activation_threshold)
            / (self.config.max_fear - self.config.activation_threshold);

        if normalized <= 0.0 {
            0.0
        } else {
            // Smooth sigmoid curve
            let x = normalized * 6.0 - 3.0; // Map to [-3, 3] for sigmoid
            1.0 / (1.0 + (-x).exp())
        }
    }

    /// Calculate fear trend from history
    fn calculate_trend(&self, current: f64) -> f64 {
        if self.fear_history.len() < 10 {
            return 0.0;
        }

        let recent_avg: f64 = self.fear_history.iter().rev().take(5).sum::<f64>() / 5.0;
        let older_avg: f64 = self.fear_history.iter().rev().skip(5).take(5).sum::<f64>() / 5.0;

        (current - recent_avg) + (recent_avg - older_avg) * 0.5
    }

    /// Apply time-based decay to memories
    fn decay_memories(&mut self) {
        let now = chrono::Utc::now().timestamp_millis();

        for memory in &mut self.memories {
            let age_ms = now - memory.last_update;
            let age_days = age_ms as f64 / (24.0 * 60.0 * 60.0 * 1000.0);

            // Exponential decay based on age
            let decay = self.config.memory_decay.powf(age_days);
            memory.fear_level = (memory.fear_level * decay).max(self.config.floor_fear);
        }

        // Decay baseline
        self.baseline_fear *= self.config.memory_decay;
    }

    /// Find index of weakest memory for removal
    fn find_weakest_memory(&self) -> Option<usize> {
        self.memories
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let score_a = a.fear_level * (a.reinforcement_count as f64).sqrt();
                let score_b = b.fear_level * (b.reinforcement_count as f64).sqrt();
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(idx, _)| idx)
    }

    /// Get total number of fear memories
    pub fn memory_count(&self) -> usize {
        self.memories.len()
    }

    /// Get current baseline fear
    pub fn baseline_fear(&self) -> f64 {
        self.baseline_fear
    }

    /// Get fear level for a specific category
    pub fn category_fear(&self, category: &str) -> f64 {
        self.category_fear.get(category).copied().unwrap_or(0.0)
    }

    /// Get all category fears
    pub fn all_category_fears(&self) -> &HashMap<String, f64> {
        &self.category_fear
    }

    /// Get statistics about the network
    pub fn statistics(&self) -> FearNetworkStats {
        let avg_fear = if self.memories.is_empty() {
            0.0
        } else {
            self.memories.iter().map(|m| m.fear_level).sum::<f64>() / self.memories.len() as f64
        };

        let max_fear = self
            .memories
            .iter()
            .map(|m| m.fear_level)
            .fold(0.0, f64::max);

        let total_reinforcements: u32 = self.memories.iter().map(|m| m.reinforcement_count).sum();

        FearNetworkStats {
            memory_count: self.memories.len(),
            average_fear: avg_fear,
            max_fear,
            baseline_fear: self.baseline_fear,
            conditioning_count: self.conditioning_count,
            extinction_count: self.extinction_count,
            total_reinforcements,
            category_count: self.category_fear.len(),
        }
    }

    /// Reset the network
    pub fn reset(&mut self) {
        self.memories.clear();
        self.fear_history.clear();
        self.baseline_fear = 0.0;
        self.conditioning_count = 0;
        self.extinction_count = 0;
        self.category_fear.clear();
    }

    /// Main processing function (compatibility with neuromorphic interface)
    pub fn process(&self) -> Result<()> {
        Ok(())
    }
}

/// Statistics about the fear network
#[derive(Debug, Clone)]
pub struct FearNetworkStats {
    pub memory_count: usize,
    pub average_fear: f64,
    pub max_fear: f64,
    pub baseline_fear: f64,
    pub conditioning_count: u64,
    pub extinction_count: u64,
    pub total_reinforcements: u32,
    pub category_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_state(features: Vec<f64>) -> MarketState {
        MarketState {
            features,
            volatility: 0.01,
            recent_return: 0.0,
            drawdown: 0.0,
            timestamp: chrono::Utc::now().timestamp_millis(),
        }
    }

    #[test]
    fn test_basic_creation() {
        let network = FearNetwork::new();
        assert_eq!(network.memory_count(), 0);
        assert_eq!(network.baseline_fear(), 0.0);
    }

    #[test]
    fn test_conditioning() {
        let mut network = FearNetwork::new();

        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.5, 0.3]),
            loss: 0.5,
            category: "drawdown".to_string(),
        };

        network.condition(event);

        assert_eq!(network.memory_count(), 1);
        assert!(network.baseline_fear() > 0.0);
    }

    #[test]
    fn test_fear_evaluation() {
        let mut network = FearNetwork::new();

        // Condition on a pattern
        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.5, 0.3]),
            loss: 0.5,
            category: "drawdown".to_string(),
        };
        network.condition(event);

        // Evaluate similar pattern - should have fear
        let similar_state = create_state(vec![0.9, 0.5, 0.3]);
        let response = network.evaluate(&similar_state);
        assert!(response.fear_level > 0.0);

        // Evaluate dissimilar pattern - should have less fear
        let dissimilar_state = create_state(vec![0.0, 0.0, 0.0]);
        let response2 = network.evaluate(&dissimilar_state);
        assert!(response2.fear_level < response.fear_level);
    }

    #[test]
    fn test_extinction() {
        let mut network = FearNetwork::new();

        // Condition
        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.5, 0.3]),
            loss: 0.5,
            category: "drawdown".to_string(),
        };
        network.condition(event);

        let initial_response = network.evaluate(&create_state(vec![1.0, 0.5, 0.3]));

        // Extinguish
        for _ in 0..10 {
            network.extinguish(&create_state(vec![1.0, 0.5, 0.3]), 1.0);
        }

        let final_response = network.evaluate(&create_state(vec![1.0, 0.5, 0.3]));

        assert!(final_response.fear_level < initial_response.fear_level);
    }

    #[test]
    fn test_fear_activation() {
        let mut network = FearNetwork::with_config(FearNetworkConfig {
            activation_threshold: 0.2,
            acquisition_rate: 0.5,
            ..Default::default()
        });

        // Strong conditioning
        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.5, 0.3]),
            loss: 1.0,
            category: "crash".to_string(),
        };
        network.condition(event);

        let response = network.evaluate(&create_state(vec![1.0, 0.5, 0.3]));
        assert!(response.is_activated);
        assert!(response.inhibition_strength > 0.0);
    }

    #[test]
    fn test_generalization() {
        let mut network = FearNetwork::with_config(FearNetworkConfig {
            generalization_sigma: 0.5,
            ..Default::default()
        });

        // Condition on one pattern
        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.0, 0.0]),
            loss: 0.5,
            category: "test".to_string(),
        };
        network.condition(event);

        // Similar pattern should have fear
        let response = network.evaluate(&create_state(vec![0.8, 0.1, 0.0]));
        assert!(response.fear_level > 0.0);
    }

    #[test]
    fn test_category_fear() {
        let mut network = FearNetwork::new();

        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.5, 0.3]),
            loss: 0.5,
            category: "volatility_spike".to_string(),
        };
        network.condition(event);

        assert!(network.category_fear("volatility_spike") > 0.0);
        assert_eq!(network.category_fear("unknown"), 0.0);
    }

    #[test]
    fn test_memory_similarity() {
        let memory = FearMemory::new(vec![1.0, 0.0, 0.0], 0.5, 0.1, "test".to_string());

        // Identical pattern
        let sim_identical = memory.similarity(&[1.0, 0.0, 0.0], 0.2);
        assert!((sim_identical - 1.0).abs() < 0.001);

        // Different pattern
        let sim_different = memory.similarity(&[0.0, 1.0, 0.0], 0.2);
        assert!(sim_different < 0.5);
    }

    #[test]
    fn test_condition_adjustments() {
        let mut network = FearNetwork::new();

        // Condition first
        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.5, 0.3]),
            loss: 0.3,
            category: "test".to_string(),
        };
        network.condition(event);

        // High volatility should increase fear
        let high_vol_state = MarketState {
            features: vec![1.0, 0.5, 0.3],
            volatility: 0.05,
            recent_return: 0.0,
            drawdown: 0.0,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let response = network.evaluate(&high_vol_state);

        // Compare with low volatility
        let low_vol_state = MarketState {
            features: vec![1.0, 0.5, 0.3],
            volatility: 0.01,
            recent_return: 0.0,
            drawdown: 0.0,
            timestamp: chrono::Utc::now().timestamp_millis(),
        };

        let response2 = network.evaluate(&low_vol_state);

        assert!(response.fear_level >= response2.fear_level);
    }

    #[test]
    fn test_statistics() {
        let mut network = FearNetwork::new();

        // Add some conditioning events
        for i in 0..5 {
            let event = ConditioningEvent {
                state: create_state(vec![i as f64 * 0.2, 0.5, 0.3]),
                loss: 0.3,
                category: format!("category_{}", i),
            };
            network.condition(event);
        }

        let stats = network.statistics();
        assert_eq!(stats.memory_count, 5);
        assert_eq!(stats.conditioning_count, 5);
        assert!(stats.average_fear > 0.0);
    }

    #[test]
    fn test_reset() {
        let mut network = FearNetwork::new();

        let event = ConditioningEvent {
            state: create_state(vec![1.0, 0.5, 0.3]),
            loss: 0.5,
            category: "test".to_string(),
        };
        network.condition(event);

        assert!(network.memory_count() > 0);

        network.reset();

        assert_eq!(network.memory_count(), 0);
        assert_eq!(network.baseline_fear(), 0.0);
    }

    #[test]
    fn test_reinforcement() {
        let mut network = FearNetwork::with_config(FearNetworkConfig {
            generalization_sigma: 0.1, // High similarity required
            ..Default::default()
        });

        // Condition multiple times on same pattern
        for _ in 0..5 {
            let event = ConditioningEvent {
                state: create_state(vec![1.0, 0.5, 0.3]),
                loss: 0.3,
                category: "repeated".to_string(),
            };
            network.condition(event);
        }

        // Should only have one memory (reinforced)
        assert_eq!(network.memory_count(), 1);

        let stats = network.statistics();
        assert!(stats.total_reinforcements > 1);
    }

    #[test]
    fn test_inhibition_calculation() {
        let network = FearNetwork::with_config(FearNetworkConfig {
            activation_threshold: 0.3,
            max_fear: 1.0,
            ..Default::default()
        });

        // Below threshold = no inhibition
        let low_inhibition = network.calculate_inhibition(0.1);
        assert_eq!(low_inhibition, 0.0);

        // Above threshold = some inhibition
        let mid_inhibition = network.calculate_inhibition(0.5);
        assert!(mid_inhibition > 0.0 && mid_inhibition < 1.0);

        // High fear = strong inhibition
        let high_inhibition = network.calculate_inhibition(0.9);
        assert!(high_inhibition > mid_inhibition);
    }

    #[test]
    fn test_process() {
        let network = FearNetwork::new();
        assert!(network.process().is_ok());
    }
}
