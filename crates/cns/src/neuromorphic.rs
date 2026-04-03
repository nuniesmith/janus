//! Neuromorphic Brain Region Initialization and Monitoring
//!
//! This module provides initialization, health checking, and coordination
//! for the JANUS neuromorphic brain architecture.

use crate::signals::{ComponentType, ProbeStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

/// Neuromorphic brain region initialization status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainRegionStatus {
    /// Component type (brain region)
    pub region: ComponentType,

    /// Initialization status
    pub initialized: bool,

    /// Health status
    pub health: ProbeStatus,

    /// Region description
    pub description: String,

    /// Dependencies (other regions this depends on)
    pub dependencies: Vec<ComponentType>,

    /// Initialization time in milliseconds
    pub init_time_ms: Option<u64>,

    /// Error message if initialization failed
    pub error: Option<String>,
}

/// Neuromorphic brain coordinator
#[derive(Debug)]
pub struct NeuromorphicBrain {
    /// Region statuses
    regions: HashMap<ComponentType, BrainRegionStatus>,

    /// Overall brain status
    active: bool,
}

impl NeuromorphicBrain {
    /// Create a new neuromorphic brain coordinator
    pub fn new() -> Self {
        let mut regions = HashMap::new();

        // Initialize all brain regions with metadata
        regions.insert(
            ComponentType::Cortex,
            BrainRegionStatus {
                region: ComponentType::Cortex,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Strategic planning & long-term memory (Manager Agent)".to_string(),
                dependencies: vec![],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::Hippocampus,
            BrainRegionStatus {
                region: ComponentType::Hippocampus,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Episodic memory & experience replay (Worker Agent)".to_string(),
                dependencies: vec![ComponentType::Cortex],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::BasalGanglia,
            BrainRegionStatus {
                region: ComponentType::BasalGanglia,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Action selection & reinforcement learning (Actor-Critic)".to_string(),
                dependencies: vec![ComponentType::Hippocampus],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::Thalamus,
            BrainRegionStatus {
                region: ComponentType::Thalamus,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Attention & multimodal fusion (Sensory Relay)".to_string(),
                dependencies: vec![],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::Prefrontal,
            BrainRegionStatus {
                region: ComponentType::Prefrontal,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Logic, planning & compliance (Executive Control)".to_string(),
                dependencies: vec![ComponentType::Cortex],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::Amygdala,
            BrainRegionStatus {
                region: ComponentType::Amygdala,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Fear, threat detection & circuit breakers (Emotional Response)"
                    .to_string(),
                dependencies: vec![ComponentType::Thalamus],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::Hypothalamus,
            BrainRegionStatus {
                region: ComponentType::Hypothalamus,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Homeostasis & risk appetite (Internal Regulation)".to_string(),
                dependencies: vec![ComponentType::Amygdala],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::Cerebellum,
            BrainRegionStatus {
                region: ComponentType::Cerebellum,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Motor control & execution (Fine-tuned Actions)".to_string(),
                dependencies: vec![ComponentType::BasalGanglia, ComponentType::Hypothalamus],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::VisualCortex,
            BrainRegionStatus {
                region: ComponentType::VisualCortex,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Pattern recognition & vision (Visual Processing)".to_string(),
                dependencies: vec![ComponentType::Thalamus],
                init_time_ms: None,
                error: None,
            },
        );

        regions.insert(
            ComponentType::Integration,
            BrainRegionStatus {
                region: ComponentType::Integration,
                initialized: false,
                health: ProbeStatus::Unknown,
                description: "Brainstem & global coordination (System Integration)".to_string(),
                // Integration depends on ALL other regions - must initialize last
                dependencies: vec![
                    ComponentType::Cortex,
                    ComponentType::Hippocampus,
                    ComponentType::BasalGanglia,
                    ComponentType::Thalamus,
                    ComponentType::Prefrontal,
                    ComponentType::Amygdala,
                    ComponentType::Hypothalamus,
                    ComponentType::Cerebellum,
                    ComponentType::VisualCortex,
                ],
                init_time_ms: None,
                error: None,
            },
        );

        Self {
            regions,
            active: false,
        }
    }

    /// Get initialization order based on dependencies
    pub fn initialization_order(&self) -> Vec<ComponentType> {
        let mut order = Vec::new();
        let mut initialized = std::collections::HashSet::new();

        // Topological sort based on dependencies
        while initialized.len() < self.regions.len() {
            let mut added = false;

            for (region_type, status) in &self.regions {
                if initialized.contains(region_type) {
                    continue;
                }

                // Check if all dependencies are initialized
                let deps_ready = status
                    .dependencies
                    .iter()
                    .all(|dep| initialized.contains(dep));

                if deps_ready {
                    order.push(*region_type);
                    initialized.insert(*region_type);
                    added = true;
                }
            }

            if !added {
                // Circular dependency or error - add remaining in arbitrary order
                warn!("Circular dependency detected in brain regions");
                for region_type in self.regions.keys() {
                    if !initialized.contains(region_type) {
                        order.push(*region_type);
                        initialized.insert(*region_type);
                    }
                }
                break;
            }
        }

        order
    }

    /// Update region health status
    pub fn update_health(&mut self, region: ComponentType, status: ProbeStatus) {
        if let Some(region_status) = self.regions.get_mut(&region) {
            region_status.health = status;
            debug!("Updated {} health: {}", region, status);
        }
    }

    /// Mark region as initialized
    pub fn mark_initialized(&mut self, region: ComponentType, init_time_ms: u64) {
        if let Some(region_status) = self.regions.get_mut(&region) {
            region_status.initialized = true;
            region_status.init_time_ms = Some(init_time_ms);
            info!(
                "🧠 Brain region {} initialized in {}ms",
                region, init_time_ms
            );
        }
    }

    /// Mark region initialization as failed
    pub fn mark_failed(&mut self, region: ComponentType, error: String) {
        if let Some(region_status) = self.regions.get_mut(&region) {
            region_status.error = Some(error.clone());
            error!(
                "🧠 Brain region {} initialization failed: {}",
                region, error
            );
        }
    }

    /// Get region status
    pub fn get_status(&self, region: ComponentType) -> Option<&BrainRegionStatus> {
        self.regions.get(&region)
    }

    /// Get all region statuses
    pub fn get_all_statuses(&self) -> Vec<&BrainRegionStatus> {
        self.regions.values().collect()
    }

    /// Check if all regions are initialized
    pub fn is_fully_initialized(&self) -> bool {
        self.regions.values().all(|s| s.initialized)
    }

    /// Check if brain is active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Activate the brain
    pub fn activate(&mut self) {
        self.active = true;
        info!("🧠 Neuromorphic brain activated");
    }

    /// Deactivate the brain
    pub fn deactivate(&mut self) {
        self.active = false;
        info!("🧠 Neuromorphic brain deactivated");
    }

    /// Get brain health summary
    pub fn health_summary(&self) -> BrainHealthSummary {
        let total_regions = self.regions.len();
        let initialized_count = self.regions.values().filter(|s| s.initialized).count();
        let healthy_count = self
            .regions
            .values()
            .filter(|s| s.health == ProbeStatus::Up)
            .count();
        let degraded_count = self
            .regions
            .values()
            .filter(|s| s.health == ProbeStatus::Degraded)
            .count();
        let down_count = self
            .regions
            .values()
            .filter(|s| s.health == ProbeStatus::Down)
            .count();

        let initialization_progress = (initialized_count as f64 / total_regions as f64) * 100.0;
        let health_score =
            self.regions.values().map(|s| s.health.score()).sum::<f64>() / total_regions as f64;

        BrainHealthSummary {
            total_regions,
            initialized_count,
            healthy_count,
            degraded_count,
            down_count,
            initialization_progress,
            health_score,
            active: self.active,
        }
    }

    /// Log brain status
    pub fn log_status(&self) {
        let summary = self.health_summary();

        info!(
            "🧠 Brain Status: {}/{} initialized ({:.1}%), Health: {:.2}, Active: {}",
            summary.initialized_count,
            summary.total_regions,
            summary.initialization_progress,
            summary.health_score,
            summary.active
        );

        // Log individual region statuses
        for status in self.get_all_statuses() {
            let init_status = if status.initialized { "✓" } else { "✗" };
            let time_str = status
                .init_time_ms
                .map(|t| format!(" ({}ms)", t))
                .unwrap_or_default();

            debug!(
                "  {} {} [{}]{} - {}",
                init_status, status.region, status.health, time_str, status.description
            );

            if let Some(err) = &status.error {
                warn!("    Error: {}", err);
            }
        }
    }
}

impl Default for NeuromorphicBrain {
    fn default() -> Self {
        Self::new()
    }
}

/// Brain health summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainHealthSummary {
    /// Total number of brain regions
    pub total_regions: usize,

    /// Number of initialized regions
    pub initialized_count: usize,

    /// Number of healthy regions
    pub healthy_count: usize,

    /// Number of degraded regions
    pub degraded_count: usize,

    /// Number of down regions
    pub down_count: usize,

    /// Initialization progress percentage
    pub initialization_progress: f64,

    /// Overall health score (0.0 - 1.0)
    pub health_score: f64,

    /// Whether brain is active
    pub active: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brain_initialization_order() {
        let brain = NeuromorphicBrain::new();
        let order = brain.initialization_order();

        // Cortex and Thalamus should be first (no dependencies)
        assert!(
            order
                .iter()
                .position(|&r| r == ComponentType::Cortex)
                .unwrap()
                < order
                    .iter()
                    .position(|&r| r == ComponentType::Hippocampus)
                    .unwrap()
        );

        // Integration should be last (depends on many others)
        assert_eq!(order.last().unwrap(), &ComponentType::Integration);
    }

    #[test]
    fn test_brain_health_tracking() {
        let mut brain = NeuromorphicBrain::new();

        brain.update_health(ComponentType::Cortex, ProbeStatus::Up);
        brain.mark_initialized(ComponentType::Cortex, 100);

        let status = brain.get_status(ComponentType::Cortex).unwrap();
        assert!(status.initialized);
        assert_eq!(status.health, ProbeStatus::Up);
        assert_eq!(status.init_time_ms, Some(100));
    }

    #[test]
    fn test_brain_health_summary() {
        let mut brain = NeuromorphicBrain::new();

        // Initialize half the regions
        for (i, region) in brain
            .regions
            .keys()
            .take(5)
            .cloned()
            .collect::<Vec<_>>()
            .iter()
            .enumerate()
        {
            brain.update_health(*region, ProbeStatus::Up);
            brain.mark_initialized(*region, (i as u64 + 1) * 100);
        }

        let summary = brain.health_summary();
        assert_eq!(summary.initialized_count, 5);
        assert_eq!(summary.total_regions, 10);
        assert_eq!(summary.initialization_progress, 50.0);
    }
}
