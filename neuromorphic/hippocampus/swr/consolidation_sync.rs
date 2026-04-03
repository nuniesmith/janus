//! Sync with cortex during sleep
//!
//! Part of the Hippocampus region
//! Component: swr
//!
//! This module implements memory consolidation synchronization that transfers
//! learned patterns from hippocampus (episodic memory) to cortex (semantic/procedural)
//! during low-activity "sleep" periods, similar to biological memory consolidation.

use crate::common::Result;
use std::collections::{HashMap, VecDeque};

/// Consolidation phase (analogous to sleep stages)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsolidationPhase {
    /// Active trading - no consolidation
    Awake,
    /// Light consolidation (can be interrupted)
    Light,
    /// Deep consolidation (replay and transfer)
    Deep,
    /// REM-like phase (pattern integration)
    Integration,
    /// Transitional between phases
    Transition,
}

impl Default for ConsolidationPhase {
    fn default() -> Self {
        ConsolidationPhase::Awake
    }
}

impl ConsolidationPhase {
    /// Get description
    pub fn description(&self) -> &'static str {
        match self {
            ConsolidationPhase::Awake => "Active state - normal operations",
            ConsolidationPhase::Light => "Light consolidation - quick pattern refresh",
            ConsolidationPhase::Deep => "Deep consolidation - memory replay and transfer",
            ConsolidationPhase::Integration => "Integration phase - pattern synthesis",
            ConsolidationPhase::Transition => "Transitioning between phases",
        }
    }

    /// Check if consolidation is active
    pub fn is_consolidating(&self) -> bool {
        matches!(
            self,
            ConsolidationPhase::Light | ConsolidationPhase::Deep | ConsolidationPhase::Integration
        )
    }

    /// Get consolidation intensity (0-1)
    pub fn intensity(&self) -> f64 {
        match self {
            ConsolidationPhase::Awake => 0.0,
            ConsolidationPhase::Light => 0.3,
            ConsolidationPhase::Deep => 0.8,
            ConsolidationPhase::Integration => 0.6,
            ConsolidationPhase::Transition => 0.2,
        }
    }
}

/// Memory type for consolidation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Episodic - specific trade instances
    Episodic,
    /// Semantic - general market knowledge
    Semantic,
    /// Procedural - trading skills/strategies
    Procedural,
    /// Emotional - risk/reward associations
    Emotional,
}

impl Default for MemoryType {
    fn default() -> Self {
        MemoryType::Episodic
    }
}

impl MemoryType {
    /// Get consolidation priority
    pub fn priority(&self) -> f64 {
        match self {
            MemoryType::Emotional => 1.0, // Highest - critical for risk management
            MemoryType::Procedural => 0.85, // High - trading skills
            MemoryType::Semantic => 0.70, // Medium - market knowledge
            MemoryType::Episodic => 0.50, // Lower - specific instances
        }
    }
}

/// A memory trace to be consolidated
#[derive(Debug, Clone)]
pub struct MemoryTrace {
    /// Unique identifier
    pub id: u64,
    /// Memory type
    pub memory_type: MemoryType,
    /// Content representation (embedding or features)
    pub content: Vec<f32>,
    /// Associated context
    pub context: Vec<f32>,
    /// Strength of the memory (0-1)
    pub strength: f64,
    /// Number of times replayed
    pub replay_count: usize,
    /// Last replay timestamp
    pub last_replay: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Associated reward signal
    pub reward_signal: f64,
    /// Emotional valence (-1 to 1)
    pub emotional_valence: f64,
    /// Tags/categories
    pub tags: Vec<String>,
    /// Whether successfully consolidated to cortex
    pub consolidated: bool,
    /// Consolidation confidence
    pub consolidation_confidence: f64,
    /// Source episode/experience ID
    pub source_id: Option<u64>,
}

impl MemoryTrace {
    pub fn new(id: u64, memory_type: MemoryType, content: Vec<f32>, created_at: u64) -> Self {
        Self {
            id,
            memory_type,
            content,
            context: Vec::new(),
            strength: 0.5,
            replay_count: 0,
            last_replay: 0,
            created_at,
            reward_signal: 0.0,
            emotional_valence: 0.0,
            tags: Vec::new(),
            consolidated: false,
            consolidation_confidence: 0.0,
            source_id: None,
        }
    }

    /// Set context
    pub fn with_context(mut self, context: Vec<f32>) -> Self {
        self.context = context;
        self
    }

    /// Set strength
    pub fn with_strength(mut self, strength: f64) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set reward signal
    pub fn with_reward(mut self, reward: f64) -> Self {
        self.reward_signal = reward;
        self
    }

    /// Set emotional valence
    pub fn with_emotion(mut self, valence: f64) -> Self {
        self.emotional_valence = valence.clamp(-1.0, 1.0);
        self
    }

    /// Add tag
    pub fn with_tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Set source ID
    pub fn with_source(mut self, source_id: u64) -> Self {
        self.source_id = Some(source_id);
        self
    }

    /// Record a replay
    pub fn record_replay(&mut self, timestamp: u64) {
        self.replay_count += 1;
        self.last_replay = timestamp;
        // Strengthen memory with each replay (diminishing returns)
        self.strength = (self.strength + 0.1 * (1.0 - self.strength)).min(1.0);
    }

    /// Calculate consolidation priority
    pub fn consolidation_priority(&self) -> f64 {
        let type_priority = self.memory_type.priority();
        let emotion_factor = 1.0 + self.emotional_valence.abs() * 0.5;
        let reward_factor = 1.0 + self.reward_signal.abs() * 0.3;
        let replay_factor = 1.0 + (self.replay_count as f64 * 0.1).min(1.0);
        let recency_factor = 1.0 / (1.0 + (self.replay_count as f64 * 0.05));

        type_priority
            * self.strength
            * emotion_factor
            * reward_factor
            * replay_factor
            * recency_factor
    }
}

/// Cortex target for consolidation
#[derive(Debug, Clone)]
pub struct CortexTarget {
    /// Target name/identifier
    pub name: String,
    /// Memory types this target accepts
    pub accepted_types: Vec<MemoryType>,
    /// Current capacity utilization (0-1)
    pub utilization: f64,
    /// Processing rate (traces per cycle)
    pub processing_rate: usize,
    /// Consolidation threshold (minimum strength to accept)
    pub threshold: f64,
    /// Total traces consolidated
    pub total_consolidated: usize,
    /// Whether target is available
    pub available: bool,
}

impl CortexTarget {
    pub fn new(name: &str, accepted_types: Vec<MemoryType>) -> Self {
        Self {
            name: name.to_string(),
            accepted_types,
            utilization: 0.0,
            processing_rate: 10,
            threshold: 0.3,
            total_consolidated: 0,
            available: true,
        }
    }

    /// Check if target accepts a memory type
    pub fn accepts(&self, memory_type: MemoryType) -> bool {
        self.accepted_types.contains(&memory_type)
    }

    /// Check if target can accept more memories
    pub fn can_accept(&self) -> bool {
        self.available && self.utilization < 0.95
    }
}

/// Consolidation event record
#[derive(Debug, Clone)]
pub struct ConsolidationEvent {
    /// Event ID
    pub id: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Phase during consolidation
    pub phase: ConsolidationPhase,
    /// Memory traces processed
    pub traces_processed: usize,
    /// Traces successfully consolidated
    pub traces_consolidated: usize,
    /// Target cortex regions
    pub targets: Vec<String>,
    /// Duration in milliseconds
    pub duration_ms: u64,
    /// Average trace strength
    pub avg_strength: f64,
    /// Notes
    pub notes: Vec<String>,
}

/// Configuration for consolidation sync
#[derive(Debug, Clone)]
pub struct ConsolidationSyncConfig {
    /// Minimum strength for consolidation
    pub min_strength: f64,
    /// Maximum traces per consolidation cycle
    pub max_traces_per_cycle: usize,
    /// Strength decay rate per cycle (for non-consolidated)
    pub decay_rate: f64,
    /// Replay boost to strength
    pub replay_boost: f64,
    /// Enable automatic phase transitions
    pub auto_phase_transition: bool,
    /// Deep consolidation trigger threshold (buffer fullness)
    pub deep_trigger_threshold: f64,
    /// Integration phase duration (cycles)
    pub integration_duration: usize,
    /// Priority weight for emotional memories
    pub emotional_weight: f64,
    /// Priority weight for reward-associated memories
    pub reward_weight: f64,
    /// Enable memory pruning
    pub enable_pruning: bool,
    /// Pruning threshold (remove below this strength)
    pub pruning_threshold: f64,
    /// Maximum buffer size
    pub max_buffer_size: usize,
}

impl Default for ConsolidationSyncConfig {
    fn default() -> Self {
        Self {
            min_strength: 0.3,
            max_traces_per_cycle: 50,
            decay_rate: 0.01,
            replay_boost: 0.15,
            auto_phase_transition: true,
            deep_trigger_threshold: 0.7,
            integration_duration: 5,
            emotional_weight: 1.5,
            reward_weight: 1.3,
            enable_pruning: true,
            pruning_threshold: 0.1,
            max_buffer_size: 10000,
        }
    }
}

/// Consolidation statistics
#[derive(Debug, Clone, Default)]
pub struct ConsolidationStats {
    /// Total consolidation cycles
    pub total_cycles: usize,
    /// Total traces processed
    pub total_processed: usize,
    /// Total traces consolidated
    pub total_consolidated: usize,
    /// Total traces pruned
    pub total_pruned: usize,
    /// Average consolidation rate
    pub avg_consolidation_rate: f64,
    /// Time spent in each phase
    pub phase_time: HashMap<String, u64>,
    /// Consolidation by memory type
    pub by_type: HashMap<MemoryType, usize>,
    /// Current buffer size
    pub buffer_size: usize,
    /// Buffer utilization
    pub buffer_utilization: f64,
}

/// Sync result
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Whether sync was successful
    pub success: bool,
    /// Traces processed
    pub traces_processed: usize,
    /// Traces consolidated
    pub traces_consolidated: usize,
    /// Traces failed
    pub traces_failed: usize,
    /// Current phase
    pub phase: ConsolidationPhase,
    /// Messages/notes
    pub messages: Vec<String>,
}

impl Default for SyncResult {
    fn default() -> Self {
        Self {
            success: true,
            traces_processed: 0,
            traces_consolidated: 0,
            traces_failed: 0,
            phase: ConsolidationPhase::Awake,
            messages: Vec::new(),
        }
    }
}

/// Sync with cortex during sleep
pub struct ConsolidationSync {
    /// Configuration
    config: ConsolidationSyncConfig,
    /// Memory trace buffer (hippocampus)
    trace_buffer: VecDeque<MemoryTrace>,
    /// Cortex targets
    cortex_targets: Vec<CortexTarget>,
    /// Current consolidation phase
    current_phase: ConsolidationPhase,
    /// Cycles in current phase
    cycles_in_phase: usize,
    /// Consolidation event history
    event_history: Vec<ConsolidationEvent>,
    /// Statistics
    stats: ConsolidationStats,
    /// Trace ID counter
    trace_counter: u64,
    /// Event ID counter
    event_counter: u64,
    /// Last consolidation timestamp
    last_consolidation: u64,
    /// Phase start timestamp
    phase_start: u64,
}

impl Default for ConsolidationSync {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsolidationSync {
    /// Create a new instance
    pub fn new() -> Self {
        let mut instance = Self {
            config: ConsolidationSyncConfig::default(),
            trace_buffer: VecDeque::new(),
            cortex_targets: Vec::new(),
            current_phase: ConsolidationPhase::Awake,
            cycles_in_phase: 0,
            event_history: Vec::new(),
            stats: ConsolidationStats::default(),
            trace_counter: 0,
            event_counter: 0,
            last_consolidation: 0,
            phase_start: 0,
        };

        // Initialize default cortex targets
        instance.add_cortex_target(CortexTarget::new(
            "semantic_cortex",
            vec![MemoryType::Semantic],
        ));
        instance.add_cortex_target(CortexTarget::new(
            "procedural_cortex",
            vec![MemoryType::Procedural],
        ));
        instance.add_cortex_target(CortexTarget::new(
            "emotional_cortex",
            vec![MemoryType::Emotional],
        ));
        instance.add_cortex_target(CortexTarget::new(
            "episodic_archive",
            vec![MemoryType::Episodic],
        ));

        instance
    }

    /// Create with custom configuration
    pub fn with_config(config: ConsolidationSyncConfig) -> Self {
        let mut instance = Self::new();
        instance.config = config;
        instance
    }

    /// Add a cortex target
    pub fn add_cortex_target(&mut self, target: CortexTarget) {
        self.cortex_targets.push(target);
    }

    /// Add a memory trace to the buffer
    pub fn add_trace(&mut self, trace: MemoryTrace) {
        // Enforce buffer limit
        while self.trace_buffer.len() >= self.config.max_buffer_size {
            // Remove weakest trace
            if let Some(idx) = self.find_weakest_trace() {
                self.trace_buffer.remove(idx);
                self.stats.total_pruned += 1;
            } else {
                self.trace_buffer.pop_front();
            }
        }

        self.trace_buffer.push_back(trace);
        self.update_buffer_stats();
    }

    /// Create and add a new trace
    pub fn create_trace(
        &mut self,
        memory_type: MemoryType,
        content: Vec<f32>,
        timestamp: u64,
    ) -> u64 {
        self.trace_counter += 1;
        let trace = MemoryTrace::new(self.trace_counter, memory_type, content, timestamp);
        self.add_trace(trace);
        self.trace_counter
    }

    /// Find index of weakest trace
    fn find_weakest_trace(&self) -> Option<usize> {
        self.trace_buffer
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.strength.partial_cmp(&b.strength).unwrap())
            .map(|(idx, _)| idx)
    }

    /// Update buffer statistics
    fn update_buffer_stats(&mut self) {
        self.stats.buffer_size = self.trace_buffer.len();
        self.stats.buffer_utilization =
            self.trace_buffer.len() as f64 / self.config.max_buffer_size as f64;
    }

    /// Transition to a new phase
    pub fn transition_to(&mut self, phase: ConsolidationPhase, timestamp: u64) {
        if self.current_phase != phase {
            // Record time in previous phase
            let phase_name = format!("{:?}", self.current_phase);
            let duration = timestamp.saturating_sub(self.phase_start);
            *self.stats.phase_time.entry(phase_name).or_insert(0) += duration;

            self.current_phase = phase;
            self.cycles_in_phase = 0;
            self.phase_start = timestamp;
        }
    }

    /// Check if should auto-transition phases
    fn check_auto_transition(&mut self, timestamp: u64) {
        if !self.config.auto_phase_transition {
            return;
        }

        match self.current_phase {
            ConsolidationPhase::Awake => {
                // Trigger consolidation if buffer is getting full
                if self.stats.buffer_utilization >= self.config.deep_trigger_threshold {
                    self.transition_to(ConsolidationPhase::Light, timestamp);
                }
            }
            ConsolidationPhase::Light => {
                // Transition to deep after initial processing
                if self.cycles_in_phase >= 2 {
                    self.transition_to(ConsolidationPhase::Deep, timestamp);
                }
            }
            ConsolidationPhase::Deep => {
                // Transition to integration after deep processing
                if self.cycles_in_phase >= 3 {
                    self.transition_to(ConsolidationPhase::Integration, timestamp);
                }
            }
            ConsolidationPhase::Integration => {
                // Return to awake after integration
                if self.cycles_in_phase >= self.config.integration_duration {
                    self.transition_to(ConsolidationPhase::Awake, timestamp);
                }
            }
            ConsolidationPhase::Transition => {
                // Brief transition period
                if self.cycles_in_phase >= 1 {
                    self.transition_to(ConsolidationPhase::Light, timestamp);
                }
            }
        }
    }

    /// Run a consolidation cycle
    pub fn consolidate(&mut self, timestamp: u64) -> SyncResult {
        self.check_auto_transition(timestamp);

        let mut result = SyncResult {
            phase: self.current_phase,
            ..Default::default()
        };

        if !self.current_phase.is_consolidating() {
            result
                .messages
                .push("Not in consolidation phase".to_string());
            return result;
        }

        self.cycles_in_phase += 1;
        self.stats.total_cycles += 1;
        self.last_consolidation = timestamp;

        // Get traces to process based on phase
        let intensity = self.current_phase.intensity();
        let max_traces = (self.config.max_traces_per_cycle as f64 * intensity) as usize;

        // Sort traces by consolidation priority
        let mut trace_priorities: Vec<(usize, f64)> = self
            .trace_buffer
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.consolidated && t.strength >= self.config.min_strength)
            .map(|(idx, t)| (idx, t.consolidation_priority()))
            .collect();

        trace_priorities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Process top traces
        let traces_to_process: Vec<usize> = trace_priorities
            .iter()
            .take(max_traces)
            .map(|(idx, _)| *idx)
            .collect();

        result.traces_processed = traces_to_process.len();

        // Consolidate each trace
        for idx in &traces_to_process {
            // Clone the trace to avoid borrow conflicts
            let (consolidated, memory_type) = {
                if let Some(trace) = self.trace_buffer.get(*idx) {
                    let trace_clone = trace.clone();
                    let memory_type = trace.memory_type;
                    let _ = trace;
                    let mut mutable_trace = trace_clone;
                    let consolidated = self.consolidate_trace(&mut mutable_trace, timestamp);
                    // Update the trace in the buffer
                    if let Some(original) = self.trace_buffer.get_mut(*idx) {
                        *original = mutable_trace;
                    }
                    (consolidated, Some(memory_type))
                } else {
                    (false, None)
                }
            };

            if let Some(mem_type) = memory_type {
                if consolidated {
                    result.traces_consolidated += 1;
                    *self.stats.by_type.entry(mem_type).or_insert(0) += 1;
                } else {
                    result.traces_failed += 1;
                }
            }
        }

        // Apply decay to non-processed traces
        self.apply_decay();

        // Prune weak traces if enabled
        if self.config.enable_pruning {
            let pruned = self.prune_weak_traces();
            if pruned > 0 {
                result
                    .messages
                    .push(format!("Pruned {} weak traces", pruned));
            }
        }

        // Record event
        self.record_event(timestamp, &result);

        // Update statistics
        self.stats.total_processed += result.traces_processed;
        self.stats.total_consolidated += result.traces_consolidated;
        self.update_buffer_stats();

        if result.traces_processed > 0 {
            self.stats.avg_consolidation_rate =
                self.stats.total_consolidated as f64 / self.stats.total_processed as f64;
        }

        result.success = result.traces_failed == 0 || result.traces_consolidated > 0;
        result
    }

    /// Consolidate a single trace to cortex
    fn consolidate_trace(&mut self, trace: &mut MemoryTrace, timestamp: u64) -> bool {
        // Find suitable cortex target
        let target = self.cortex_targets.iter_mut().find(|t| {
            t.accepts(trace.memory_type) && t.can_accept() && trace.strength >= t.threshold
        });

        if let Some(target) = target {
            // Simulate transfer to cortex
            trace.consolidated = true;
            trace.consolidation_confidence = trace.strength * 0.9;
            trace.record_replay(timestamp);

            target.total_consolidated += 1;
            target.utilization = (target.utilization + 0.01).min(1.0);

            true
        } else {
            // Boost strength slightly for retry
            trace.strength = (trace.strength + self.config.replay_boost * 0.5).min(1.0);
            false
        }
    }

    /// Apply decay to non-consolidated traces
    fn apply_decay(&mut self) {
        for trace in &mut self.trace_buffer {
            if !trace.consolidated {
                trace.strength *= 1.0 - self.config.decay_rate;
            }
        }
    }

    /// Prune weak traces
    fn prune_weak_traces(&mut self) -> usize {
        let before = self.trace_buffer.len();
        self.trace_buffer
            .retain(|t| t.consolidated || t.strength >= self.config.pruning_threshold);
        let pruned = before - self.trace_buffer.len();
        self.stats.total_pruned += pruned;
        pruned
    }

    /// Record a consolidation event
    fn record_event(&mut self, timestamp: u64, result: &SyncResult) {
        self.event_counter += 1;

        let event = ConsolidationEvent {
            id: self.event_counter,
            timestamp,
            phase: self.current_phase,
            traces_processed: result.traces_processed,
            traces_consolidated: result.traces_consolidated,
            targets: self.cortex_targets.iter().map(|t| t.name.clone()).collect(),
            duration_ms: 0, // Would be measured in real implementation
            avg_strength: self.average_trace_strength(),
            notes: result.messages.clone(),
        };

        self.event_history.push(event);

        // Keep limited history
        if self.event_history.len() > 1000 {
            self.event_history.remove(0);
        }
    }

    /// Calculate average trace strength
    fn average_trace_strength(&self) -> f64 {
        if self.trace_buffer.is_empty() {
            return 0.0;
        }
        self.trace_buffer.iter().map(|t| t.strength).sum::<f64>() / self.trace_buffer.len() as f64
    }

    /// Replay a memory trace (strengthen it)
    pub fn replay_trace(&mut self, trace_id: u64, timestamp: u64) -> bool {
        if let Some(trace) = self.trace_buffer.iter_mut().find(|t| t.id == trace_id) {
            trace.record_replay(timestamp);
            trace.strength = (trace.strength + self.config.replay_boost).min(1.0);
            true
        } else {
            false
        }
    }

    /// Get traces by type
    pub fn get_traces_by_type(&self, memory_type: MemoryType) -> Vec<&MemoryTrace> {
        self.trace_buffer
            .iter()
            .filter(|t| t.memory_type == memory_type)
            .collect()
    }

    /// Get consolidated traces
    pub fn get_consolidated(&self) -> Vec<&MemoryTrace> {
        self.trace_buffer
            .iter()
            .filter(|t| t.consolidated)
            .collect()
    }

    /// Get unconsolidated traces
    pub fn get_unconsolidated(&self) -> Vec<&MemoryTrace> {
        self.trace_buffer
            .iter()
            .filter(|t| !t.consolidated)
            .collect()
    }

    /// Get trace by ID
    pub fn get_trace(&self, trace_id: u64) -> Option<&MemoryTrace> {
        self.trace_buffer.iter().find(|t| t.id == trace_id)
    }

    /// Get current phase
    pub fn current_phase(&self) -> ConsolidationPhase {
        self.current_phase
    }

    /// Get buffer size
    pub fn buffer_size(&self) -> usize {
        self.trace_buffer.len()
    }

    /// Get statistics
    pub fn stats(&self) -> &ConsolidationStats {
        &self.stats
    }

    /// Get event history
    pub fn event_history(&self) -> &[ConsolidationEvent] {
        &self.event_history
    }

    /// Get cortex targets
    pub fn cortex_targets(&self) -> &[CortexTarget] {
        &self.cortex_targets
    }

    /// Force immediate consolidation
    pub fn force_consolidation(&mut self, timestamp: u64) -> SyncResult {
        let previous_phase = self.current_phase;
        self.transition_to(ConsolidationPhase::Deep, timestamp);
        let result = self.consolidate(timestamp);
        self.transition_to(previous_phase, timestamp);
        result
    }

    /// Clear all traces
    pub fn clear(&mut self) {
        self.trace_buffer.clear();
        self.update_buffer_stats();
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Consolidation Sync Summary ===\n");
        s.push_str(&format!("Current Phase: {:?}\n", self.current_phase));
        s.push_str(&format!(
            "Buffer Size: {} / {}\n",
            self.trace_buffer.len(),
            self.config.max_buffer_size
        ));
        s.push_str(&format!(
            "Buffer Utilization: {:.1}%\n",
            self.stats.buffer_utilization * 100.0
        ));
        s.push_str(&format!("Total Cycles: {}\n", self.stats.total_cycles));
        s.push_str(&format!(
            "Total Consolidated: {}\n",
            self.stats.total_consolidated
        ));
        s.push_str(&format!(
            "Consolidation Rate: {:.1}%\n",
            self.stats.avg_consolidation_rate * 100.0
        ));
        s.push_str(&format!("Total Pruned: {}\n", self.stats.total_pruned));
        s.push_str(&format!(
            "Avg Trace Strength: {:.3}\n",
            self.average_trace_strength()
        ));

        s.push_str("\nBy Memory Type:\n");
        for (mem_type, count) in &self.stats.by_type {
            s.push_str(&format!("  {:?}: {}\n", mem_type, count));
        }

        s.push_str("\nCortex Targets:\n");
        for target in &self.cortex_targets {
            s.push_str(&format!(
                "  {}: {} consolidated, {:.1}% utilized\n",
                target.name,
                target.total_consolidated,
                target.utilization * 100.0
            ));
        }

        s
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via consolidate method
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trace(id: u64, memory_type: MemoryType, strength: f64) -> MemoryTrace {
        MemoryTrace::new(id, memory_type, vec![1.0, 2.0, 3.0], 0).with_strength(strength)
    }

    #[test]
    fn test_basic() {
        let instance = ConsolidationSync::new();
        assert!(instance.process().is_ok());
        assert_eq!(instance.buffer_size(), 0);
    }

    #[test]
    fn test_add_trace() {
        let mut sync = ConsolidationSync::new();

        let trace = create_test_trace(1, MemoryType::Episodic, 0.5);
        sync.add_trace(trace);

        assert_eq!(sync.buffer_size(), 1);
    }

    #[test]
    fn test_create_trace() {
        let mut sync = ConsolidationSync::new();

        let id = sync.create_trace(MemoryType::Semantic, vec![1.0, 2.0], 100);

        assert!(id > 0);
        assert_eq!(sync.buffer_size(), 1);
        assert!(sync.get_trace(id).is_some());
    }

    #[test]
    fn test_phase_transitions() {
        let mut sync = ConsolidationSync::new();

        assert_eq!(sync.current_phase(), ConsolidationPhase::Awake);

        sync.transition_to(ConsolidationPhase::Light, 100);
        assert_eq!(sync.current_phase(), ConsolidationPhase::Light);
        assert!(sync.current_phase().is_consolidating());

        sync.transition_to(ConsolidationPhase::Deep, 200);
        assert_eq!(sync.current_phase(), ConsolidationPhase::Deep);
    }

    #[test]
    fn test_consolidation_cycle() {
        let mut sync = ConsolidationSync::new();

        // Add some traces
        for i in 0..10 {
            let trace = create_test_trace(i, MemoryType::Semantic, 0.5 + i as f64 * 0.05);
            sync.add_trace(trace);
        }

        // Enter consolidation phase
        sync.transition_to(ConsolidationPhase::Deep, 100);

        let result = sync.consolidate(200);

        assert!(result.traces_processed > 0);
        assert_eq!(result.phase, ConsolidationPhase::Deep);
    }

    #[test]
    fn test_memory_type_priority() {
        assert!(MemoryType::Emotional.priority() > MemoryType::Procedural.priority());
        assert!(MemoryType::Procedural.priority() > MemoryType::Semantic.priority());
        assert!(MemoryType::Semantic.priority() > MemoryType::Episodic.priority());
    }

    #[test]
    fn test_trace_consolidation_priority() {
        let trace1 = create_test_trace(1, MemoryType::Emotional, 0.8).with_reward(1.0);
        let trace2 = create_test_trace(2, MemoryType::Episodic, 0.4);

        assert!(trace1.consolidation_priority() > trace2.consolidation_priority());
    }

    #[test]
    fn test_replay_trace() {
        let mut sync = ConsolidationSync::new();

        let trace = create_test_trace(1, MemoryType::Procedural, 0.5);
        sync.add_trace(trace);

        let initial_strength = sync.get_trace(1).unwrap().strength;
        sync.replay_trace(1, 100);

        let new_strength = sync.get_trace(1).unwrap().strength;
        assert!(new_strength > initial_strength);
    }

    #[test]
    fn test_get_traces_by_type() {
        let mut sync = ConsolidationSync::new();

        sync.add_trace(create_test_trace(1, MemoryType::Semantic, 0.5));
        sync.add_trace(create_test_trace(2, MemoryType::Semantic, 0.6));
        sync.add_trace(create_test_trace(3, MemoryType::Procedural, 0.7));

        let semantic = sync.get_traces_by_type(MemoryType::Semantic);
        assert_eq!(semantic.len(), 2);

        let procedural = sync.get_traces_by_type(MemoryType::Procedural);
        assert_eq!(procedural.len(), 1);
    }

    #[test]
    fn test_cortex_target() {
        let target = CortexTarget::new("test", vec![MemoryType::Semantic, MemoryType::Episodic]);

        assert!(target.accepts(MemoryType::Semantic));
        assert!(target.accepts(MemoryType::Episodic));
        assert!(!target.accepts(MemoryType::Emotional));
        assert!(target.can_accept());
    }

    #[test]
    fn test_auto_phase_transition() {
        let mut config = ConsolidationSyncConfig::default();
        config.auto_phase_transition = true;
        config.deep_trigger_threshold = 0.5;
        config.max_buffer_size = 10;

        let mut sync = ConsolidationSync::with_config(config);

        // Fill buffer past threshold
        for i in 0..6 {
            sync.add_trace(create_test_trace(i, MemoryType::Semantic, 0.5));
        }

        sync.check_auto_transition(100);

        // Should have transitioned from Awake
        assert_ne!(sync.current_phase(), ConsolidationPhase::Awake);
    }

    #[test]
    fn test_pruning() {
        let mut config = ConsolidationSyncConfig::default();
        config.enable_pruning = true;
        config.pruning_threshold = 0.3;

        let mut sync = ConsolidationSync::with_config(config);

        // Add weak and strong traces
        sync.add_trace(create_test_trace(1, MemoryType::Semantic, 0.1)); // Will be pruned
        sync.add_trace(create_test_trace(2, MemoryType::Semantic, 0.5)); // Will survive

        sync.transition_to(ConsolidationPhase::Deep, 100);
        sync.consolidate(200);

        // Weak trace should be pruned
        assert!(sync.get_trace(1).is_none() || sync.get_trace(1).unwrap().strength >= 0.3);
    }

    #[test]
    fn test_force_consolidation() {
        let mut sync = ConsolidationSync::new();

        for i in 0..5 {
            sync.add_trace(create_test_trace(i, MemoryType::Semantic, 0.6));
        }

        let result = sync.force_consolidation(100);

        assert!(result.traces_processed > 0);
    }

    #[test]
    fn test_buffer_limit() {
        let mut config = ConsolidationSyncConfig::default();
        config.max_buffer_size = 5;

        let mut sync = ConsolidationSync::with_config(config);

        // Add more than limit
        for i in 0..10 {
            sync.add_trace(create_test_trace(i, MemoryType::Semantic, 0.5));
        }

        assert!(sync.buffer_size() <= 5);
    }

    #[test]
    fn test_statistics() {
        let mut sync = ConsolidationSync::new();

        for i in 0..5 {
            sync.add_trace(create_test_trace(i, MemoryType::Semantic, 0.5));
        }

        sync.transition_to(ConsolidationPhase::Deep, 100);
        sync.consolidate(200);

        let stats = sync.stats();
        assert!(stats.total_cycles > 0);
        assert!(stats.buffer_size > 0);
    }

    #[test]
    fn test_event_history() {
        let mut sync = ConsolidationSync::new();

        for i in 0..5 {
            sync.add_trace(create_test_trace(i, MemoryType::Semantic, 0.5));
        }

        sync.transition_to(ConsolidationPhase::Deep, 100);
        sync.consolidate(200);
        sync.consolidate(300);

        let history = sync.event_history();
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_phase_intensity() {
        assert_eq!(ConsolidationPhase::Awake.intensity(), 0.0);
        assert!(ConsolidationPhase::Deep.intensity() > ConsolidationPhase::Light.intensity());
    }

    #[test]
    fn test_memory_trace_builders() {
        let trace = MemoryTrace::new(1, MemoryType::Emotional, vec![1.0], 0)
            .with_context(vec![2.0])
            .with_strength(0.8)
            .with_reward(1.5)
            .with_emotion(0.7)
            .with_tag("important")
            .with_source(42);

        assert_eq!(trace.context, vec![2.0]);
        assert_eq!(trace.strength, 0.8);
        assert_eq!(trace.reward_signal, 1.5);
        assert_eq!(trace.emotional_valence, 0.7);
        assert!(trace.tags.contains(&"important".to_string()));
        assert_eq!(trace.source_id, Some(42));
    }

    #[test]
    fn test_summary() {
        let mut sync = ConsolidationSync::new();

        for i in 0..3 {
            sync.add_trace(create_test_trace(i, MemoryType::Semantic, 0.5));
        }

        let summary = sync.summary();
        assert!(summary.contains("Consolidation Sync"));
        assert!(summary.contains("Buffer Size"));
    }

    #[test]
    fn test_clear() {
        let mut sync = ConsolidationSync::new();

        for i in 0..5 {
            sync.add_trace(create_test_trace(i, MemoryType::Semantic, 0.5));
        }

        assert_eq!(sync.buffer_size(), 5);

        sync.clear();

        assert_eq!(sync.buffer_size(), 0);
    }

    #[test]
    fn test_decay() {
        let mut sync = ConsolidationSync::new();
        sync.config.decay_rate = 0.1;

        sync.add_trace(create_test_trace(1, MemoryType::Semantic, 0.5));

        let initial = sync.get_trace(1).unwrap().strength;

        sync.transition_to(ConsolidationPhase::Light, 100);
        sync.apply_decay();

        let after = sync.get_trace(1).unwrap().strength;
        assert!(after < initial);
    }

    #[test]
    fn test_consolidated_vs_unconsolidated() {
        let mut sync = ConsolidationSync::new();

        for i in 0..5 {
            let mut trace = create_test_trace(i, MemoryType::Semantic, 0.6);
            if i < 2 {
                trace.consolidated = true;
            }
            sync.add_trace(trace);
        }

        let consolidated = sync.get_consolidated();
        let unconsolidated = sync.get_unconsolidated();

        assert_eq!(consolidated.len(), 2);
        assert_eq!(unconsolidated.len(), 3);
    }
}
