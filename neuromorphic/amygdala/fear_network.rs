//! Fear Network with Integrated Reinforcement Learning (FNI-RL)
//!
//! Maintains a memory of dangerous market conditions and learns from outcomes.
//! Uses reinforcement learning to adapt threat detection based on experience.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

/// Threat signature with learned parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatSignature {
    /// Name of the threat pattern
    pub name: String,

    /// Requires volatility spike
    pub volatility_spike: bool,

    /// Requires volume spike
    pub volume_spike: bool,

    /// Requires price gap
    pub price_gap: bool,

    /// Base danger score (0.0 to 1.0)
    pub danger_score: f32,

    /// Learned weight adjustment (-1.0 to 1.0)
    pub learned_weight: f32,

    /// Number of times this threat was detected
    pub detection_count: u64,

    /// Number of true positives (correctly predicted danger)
    pub true_positives: u64,

    /// Number of false alarms (predicted danger but was safe)
    pub false_alarms: u64,

    /// Average loss when threat occurred
    pub avg_loss_on_threat: f32,

    /// Average loss when false alarm
    pub avg_loss_on_false_alarm: f32,

    /// Last updated timestamp
    pub last_updated: i64,
}

impl ThreatSignature {
    /// Create a new threat signature
    pub fn new(
        name: String,
        volatility_spike: bool,
        volume_spike: bool,
        price_gap: bool,
        danger_score: f32,
    ) -> Self {
        Self {
            name,
            volatility_spike,
            volume_spike,
            price_gap,
            danger_score: danger_score.clamp(0.0, 1.0),
            learned_weight: 0.0,
            detection_count: 0,
            true_positives: 0,
            false_alarms: 0,
            avg_loss_on_threat: 0.0,
            avg_loss_on_false_alarm: 0.0,
            last_updated: chrono::Utc::now().timestamp(),
        }
    }

    /// Get the effective danger score after RL adjustments
    pub fn effective_danger_score(&self) -> f32 {
        (self.danger_score + self.learned_weight).clamp(0.0, 1.0)
    }

    /// Get precision (true positives / total detections)
    pub fn precision(&self) -> f32 {
        if self.detection_count == 0 {
            return 0.5; // No data, assume neutral
        }
        self.true_positives as f32 / self.detection_count as f32
    }

    /// Update based on outcome (reinforcement learning)
    pub fn update_from_outcome(&mut self, was_dangerous: bool, loss: f32) {
        self.detection_count += 1;
        self.last_updated = chrono::Utc::now().timestamp();

        if was_dangerous {
            self.true_positives += 1;
            // Update average loss on real threat
            let n = self.true_positives as f32;
            self.avg_loss_on_threat = ((self.avg_loss_on_threat * (n - 1.0)) + loss) / n;

            // Increase weight if threat was real and caused loss
            if loss > 0.0 {
                self.learned_weight += 0.05 * (loss / 1000.0).min(1.0);
            }
        } else {
            self.false_alarms += 1;
            // Update average loss on false alarm
            let n = self.false_alarms as f32;
            self.avg_loss_on_false_alarm = ((self.avg_loss_on_false_alarm * (n - 1.0)) + loss) / n;

            // Decrease weight for false alarms (but not too much)
            self.learned_weight -= 0.03;
        }

        // Clamp learned weight
        self.learned_weight = self.learned_weight.clamp(-0.3, 0.3);
    }
}

/// Fear Network with Integrated Reinforcement Learning
pub struct FearNetwork {
    /// Learned threat signatures
    fear_memories: HashMap<String, ThreatSignature>,

    /// Base threshold for triggering fear response
    base_threshold: f32,

    /// Learning rate for RL updates
    learning_rate: f32,

    /// Recent outcomes for batch learning
    recent_outcomes: Vec<FearOutcome>,

    /// Max recent outcomes to keep
    max_recent_outcomes: usize,
}

/// Outcome of a fear response for learning
#[derive(Debug, Clone)]
pub struct FearOutcome {
    /// Which threat signature was triggered
    pub threat_name: String,

    /// Market conditions at the time
    pub conditions: MarketConditions,

    /// Whether the threat was real (true positive) or false alarm
    pub was_dangerous: bool,

    /// Actual loss/gain that occurred
    pub outcome_value: f32,

    /// Timestamp
    pub timestamp: i64,
}

impl FearNetwork {
    /// Create a new fear network with RL
    pub fn new(threshold: f32) -> Self {
        let mut network = Self {
            fear_memories: HashMap::new(),
            base_threshold: threshold,
            learning_rate: 0.1,
            recent_outcomes: Vec::new(),
            max_recent_outcomes: 100,
        };

        // Pre-load common threat patterns
        network.add_fear_memory(ThreatSignature::new(
            "flash_crash".to_string(),
            true, // volatility spike
            true, // volume spike
            true, // price gap
            0.95,
        ));

        network.add_fear_memory(ThreatSignature::new(
            "extreme_volatility".to_string(),
            true,  // volatility spike
            false, // volume spike
            false, // price gap
            0.75,
        ));

        network.add_fear_memory(ThreatSignature::new(
            "liquidity_crisis".to_string(),
            false, // volatility spike
            true,  // volume spike
            false, // price gap
            0.70,
        ));

        network.add_fear_memory(ThreatSignature::new(
            "gap_down".to_string(),
            false, // volatility spike
            false, // volume spike
            true,  // price gap
            0.65,
        ));

        network
    }

    /// Add a new fear memory pattern
    pub fn add_fear_memory(&mut self, signature: ThreatSignature) {
        info!("Adding fear pattern: {}", signature.name);
        self.fear_memories.insert(signature.name.clone(), signature);
    }

    /// Detect if current conditions match a threat pattern
    pub fn detect_threat(&self, conditions: &MarketConditions) -> Option<&ThreatSignature> {
        self.fear_memories
            .values()
            .filter(|signature| {
                self.matches_threat(conditions, signature)
                    && signature.effective_danger_score() >= self.base_threshold
            })
            .max_by(|a, b| {
                a.effective_danger_score()
                    .partial_cmp(&b.effective_danger_score())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Check if conditions match a specific threat pattern
    fn matches_threat(&self, conditions: &MarketConditions, signature: &ThreatSignature) -> bool {
        let mut score = 0;
        let mut total = 0;

        if signature.volatility_spike {
            total += 1;
            if conditions.volatility > conditions.avg_volatility * 2.0 {
                score += 1;
            }
        }

        if signature.volume_spike {
            total += 1;
            if conditions.volume > conditions.avg_volume * 3.0 {
                score += 1;
            }
        }

        if signature.price_gap {
            total += 1;
            if conditions.price_change_pct.abs() > 0.05 {
                // 5% gap
                score += 1;
            }
        }

        // Require at least one condition to match AND score must be at least half of total
        total > 0 && score > 0 && score * 2 >= total
    }

    /// Record an outcome for reinforcement learning
    pub fn record_outcome(
        &mut self,
        threat_name: String,
        conditions: MarketConditions,
        was_dangerous: bool,
        outcome_value: f32,
    ) {
        debug!(
            "Recording outcome for {}: was_dangerous={}, value={:.2}",
            threat_name, was_dangerous, outcome_value
        );

        // Store outcome
        let outcome = FearOutcome {
            threat_name: threat_name.clone(),
            conditions,
            was_dangerous,
            outcome_value,
            timestamp: chrono::Utc::now().timestamp(),
        };

        self.recent_outcomes.push(outcome);

        // Prune if needed
        if self.recent_outcomes.len() > self.max_recent_outcomes {
            self.recent_outcomes.remove(0);
        }

        // Update the threat signature
        if let Some(signature) = self.fear_memories.get_mut(&threat_name) {
            signature.update_from_outcome(was_dangerous, outcome_value.abs());

            info!(
                "Updated {} - precision: {:.2}%, effective_score: {:.2}, weight: {:.3}",
                threat_name,
                signature.precision() * 100.0,
                signature.effective_danger_score(),
                signature.learned_weight
            );
        }
    }

    /// Batch update from recent outcomes (for periodic learning)
    pub fn batch_update(&mut self) {
        if self.recent_outcomes.is_empty() {
            return;
        }

        info!(
            "Performing batch update on {} outcomes",
            self.recent_outcomes.len()
        );

        // Group outcomes by threat
        let mut by_threat: HashMap<String, Vec<&FearOutcome>> = HashMap::new();
        for outcome in &self.recent_outcomes {
            by_threat
                .entry(outcome.threat_name.clone())
                .or_default()
                .push(outcome);
        }

        // Update each threat signature
        for (threat_name, outcomes) in by_threat {
            if let Some(signature) = self.fear_memories.get_mut(&threat_name) {
                let true_pos = outcomes.iter().filter(|o| o.was_dangerous).count();
                let total = outcomes.len();
                let precision = true_pos as f32 / total as f32;

                // Adjust learned weight based on precision
                if precision < 0.5 {
                    // Too many false alarms, decrease weight
                    signature.learned_weight -= self.learning_rate * (0.5 - precision);
                } else if precision > 0.7 {
                    // Good precision, increase weight slightly
                    signature.learned_weight += self.learning_rate * (precision - 0.7);
                }

                signature.learned_weight = signature.learned_weight.clamp(-0.3, 0.3);
            }
        }

        // Clear processed outcomes
        self.recent_outcomes.clear();

        info!("Batch update complete");
    }

    /// Get statistics about fear network learning
    pub fn get_stats(&self) -> FearNetworkStats {
        let total_detections: u64 = self.fear_memories.values().map(|s| s.detection_count).sum();
        let total_true_positives: u64 = self.fear_memories.values().map(|s| s.true_positives).sum();
        let total_false_alarms: u64 = self.fear_memories.values().map(|s| s.false_alarms).sum();

        let overall_precision = if total_detections > 0 {
            total_true_positives as f32 / total_detections as f32
        } else {
            0.0
        };

        FearNetworkStats {
            num_patterns: self.fear_memories.len(),
            total_detections,
            total_true_positives,
            total_false_alarms,
            overall_precision,
            pending_outcomes: self.recent_outcomes.len(),
        }
    }

    /// Get a specific threat signature (mutable for external updates)
    pub fn get_threat_mut(&mut self, name: &str) -> Option<&mut ThreatSignature> {
        self.fear_memories.get_mut(name)
    }

    /// Get all threat signatures
    pub fn get_all_threats(&self) -> Vec<&ThreatSignature> {
        self.fear_memories.values().collect()
    }

    /// Adjust base threshold
    pub fn set_threshold(&mut self, threshold: f32) {
        self.base_threshold = threshold.clamp(0.0, 1.0);
        info!("Fear threshold set to {:.2}", self.base_threshold);
    }

    /// Get current threshold
    pub fn threshold(&self) -> f32 {
        self.base_threshold
    }
}

/// Statistics about the fear network
#[derive(Debug, Clone)]
pub struct FearNetworkStats {
    pub num_patterns: usize,
    pub total_detections: u64,
    pub total_true_positives: u64,
    pub total_false_alarms: u64,
    pub overall_precision: f32,
    pub pending_outcomes: usize,
}

/// Market conditions for threat detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f32,
    pub avg_volatility: f32,
    pub volume: f32,
    pub avg_volume: f32,
    pub price_change_pct: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threat_signature_creation() {
        let sig = ThreatSignature::new("test".to_string(), true, true, true, 0.8);
        assert_eq!(sig.name, "test");
        assert_eq!(sig.danger_score, 0.8);
        assert_eq!(sig.learned_weight, 0.0);
        assert_eq!(sig.effective_danger_score(), 0.8);
    }

    #[test]
    fn test_rl_update_true_positive() {
        let mut sig = ThreatSignature::new("test".to_string(), true, true, true, 0.7);

        // Record a true positive with loss
        sig.update_from_outcome(true, 500.0);

        assert_eq!(sig.true_positives, 1);
        assert_eq!(sig.detection_count, 1);
        assert!(sig.learned_weight > 0.0); // Should increase weight
        assert_eq!(sig.precision(), 1.0);
    }

    #[test]
    fn test_rl_update_false_alarm() {
        let mut sig = ThreatSignature::new("test".to_string(), true, true, true, 0.7);

        // Record a false alarm
        sig.update_from_outcome(false, 0.0);

        assert_eq!(sig.false_alarms, 1);
        assert_eq!(sig.detection_count, 1);
        assert!(sig.learned_weight < 0.0); // Should decrease weight
        assert_eq!(sig.precision(), 0.0);
    }

    #[test]
    fn test_threat_detection() {
        let network = FearNetwork::new(0.7);

        let flash_crash_conditions = MarketConditions {
            volatility: 0.08,
            avg_volatility: 0.02,
            volume: 10000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.06,
        };

        let threat = network.detect_threat(&flash_crash_conditions);
        assert!(threat.is_some());

        if let Some(t) = threat {
            assert!(t.effective_danger_score() >= 0.7);
            assert_eq!(t.name, "flash_crash");
        }
    }

    #[test]
    fn test_no_threat() {
        let network = FearNetwork::new(0.7);

        let normal_conditions = MarketConditions {
            volatility: 0.02,
            avg_volatility: 0.02,
            volume: 2000.0,
            avg_volume: 2000.0,
            price_change_pct: 0.01,
        };

        let threat = network.detect_threat(&normal_conditions);
        assert!(threat.is_none());
    }

    #[test]
    fn test_learning_from_outcomes() {
        let mut network = FearNetwork::new(0.7);

        let conditions = MarketConditions {
            volatility: 0.08,
            avg_volatility: 0.02,
            volume: 10000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.06,
        };

        // Record several true positives
        for _ in 0..5 {
            network.record_outcome("flash_crash".to_string(), conditions.clone(), true, 1000.0);
        }

        // Record a false alarm
        network.record_outcome("flash_crash".to_string(), conditions.clone(), false, 0.0);

        let stats = network.get_stats();
        assert_eq!(stats.total_detections, 6);
        assert_eq!(stats.total_true_positives, 5);
        assert_eq!(stats.total_false_alarms, 1);
        assert!((stats.overall_precision - 5.0 / 6.0).abs() < 0.01);
    }

    #[test]
    fn test_batch_update() {
        let mut network = FearNetwork::new(0.7);

        let conditions = MarketConditions {
            volatility: 0.05,
            avg_volatility: 0.02,
            volume: 5000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.03,
        };

        // Record outcomes without immediate updates
        for i in 0..10 {
            let is_dangerous = i % 3 == 0; // 1/3 are real threats
            network.record_outcome(
                "extreme_volatility".to_string(),
                conditions.clone(),
                is_dangerous,
                100.0,
            );
        }

        let initial_weight = network
            .fear_memories
            .get("extreme_volatility")
            .unwrap()
            .learned_weight;

        network.batch_update();

        // Should have adjusted weight based on outcomes
        let final_weight = network
            .fear_memories
            .get("extreme_volatility")
            .unwrap()
            .learned_weight;
        assert_ne!(initial_weight, final_weight);
    }

    #[test]
    fn test_threat_priority() {
        let mut network = FearNetwork::new(0.5);

        // Add a learned high-priority threat
        let mut high_priority =
            ThreatSignature::new("learned_danger".to_string(), true, false, false, 0.6);
        high_priority.learned_weight = 0.3; // Boosted by learning
        network.add_fear_memory(high_priority);

        let conditions = MarketConditions {
            volatility: 0.08,
            avg_volatility: 0.02,
            volume: 10000.0,
            avg_volume: 2000.0,
            price_change_pct: -0.06,
        };

        // Both flash_crash (0.95) and learned_danger (0.6 + 0.3 = 0.9) match
        // flash_crash should still win
        let threat = network.detect_threat(&conditions);
        assert!(threat.is_some());
        assert_eq!(threat.unwrap().name, "flash_crash");
    }

    #[test]
    fn test_get_stats() {
        let mut network = FearNetwork::new(0.7);

        network.record_outcome(
            "flash_crash".to_string(),
            MarketConditions {
                volatility: 0.08,
                avg_volatility: 0.02,
                volume: 10000.0,
                avg_volume: 2000.0,
                price_change_pct: -0.06,
            },
            true,
            1000.0,
        );

        let stats = network.get_stats();
        assert_eq!(stats.num_patterns, 4); // 4 pre-loaded patterns
        assert_eq!(stats.total_detections, 1);
        assert_eq!(stats.total_true_positives, 1);
        assert_eq!(stats.overall_precision, 1.0);
    }
}
