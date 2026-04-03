//! Immediate threat response
//!
//! Part of the Amygdala region
//! Component: fear
//!
//! Handles immediate responses to detected threats including:
//! - Response type selection based on threat severity
//! - Action coordination and prioritization
//! - Response timing and urgency management
//! - Escalation pathways for severe threats
//! - Response history and effectiveness tracking

use crate::common::Result;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Threat severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ThreatSeverity {
    /// Minor threat - monitor only
    Low,
    /// Moderate threat - reduce exposure
    Medium,
    /// Significant threat - hedge positions
    High,
    /// Severe threat - exit positions
    Critical,
    /// Emergency - immediate halt
    Emergency,
}

impl ThreatSeverity {
    /// Get multiplier for response urgency
    pub fn urgency_multiplier(&self) -> f64 {
        match self {
            Self::Low => 0.2,
            Self::Medium => 0.4,
            Self::High => 0.7,
            Self::Critical => 0.9,
            Self::Emergency => 1.0,
        }
    }

    /// Maximum response delay allowed
    pub fn max_delay_ms(&self) -> u64 {
        match self {
            Self::Low => 5000,
            Self::Medium => 2000,
            Self::High => 500,
            Self::Critical => 100,
            Self::Emergency => 10,
        }
    }
}

/// Types of threat responses
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResponseType {
    /// Continue monitoring without action
    Monitor,
    /// Reduce position size by percentage
    ReduceExposure { reduction_pct: u8 },
    /// Add hedging positions
    Hedge { hedge_ratio: u8 },
    /// Close specific positions
    ClosePositions { symbols: Vec<String> },
    /// Halt new order entry
    HaltNewOrders,
    /// Cancel all pending orders
    CancelPendingOrders,
    /// Full position liquidation
    Liquidate,
    /// Trigger kill switch
    KillSwitch,
    /// Custom response action
    Custom { action: String },
}

impl ResponseType {
    /// Priority level of the response (higher = more urgent)
    pub fn priority(&self) -> u8 {
        match self {
            Self::Monitor => 1,
            Self::ReduceExposure { .. } => 3,
            Self::Hedge { .. } => 4,
            Self::ClosePositions { .. } => 5,
            Self::HaltNewOrders => 6,
            Self::CancelPendingOrders => 7,
            Self::Liquidate => 9,
            Self::KillSwitch => 10,
            Self::Custom { .. } => 2,
        }
    }

    /// Whether this response requires confirmation
    pub fn requires_confirmation(&self) -> bool {
        matches!(
            self,
            Self::Liquidate | Self::KillSwitch | Self::ClosePositions { .. }
        )
    }
}

/// A detected threat requiring response
#[derive(Debug, Clone)]
pub struct Threat {
    /// Unique threat identifier
    pub id: u64,
    /// Source of the threat detection
    pub source: String,
    /// Type/category of threat
    pub threat_type: String,
    /// Severity level
    pub severity: ThreatSeverity,
    /// Confidence in threat detection (0.0 - 1.0)
    pub confidence: f64,
    /// Affected symbols
    pub affected_symbols: Vec<String>,
    /// Estimated time to impact in milliseconds
    pub time_to_impact_ms: Option<u64>,
    /// Additional threat context
    pub context: HashMap<String, f64>,
    /// Timestamp when threat was detected
    pub detected_at: Instant,
}

impl Threat {
    /// Calculate threat score combining severity and confidence
    pub fn threat_score(&self) -> f64 {
        self.severity.urgency_multiplier() * self.confidence
    }

    /// Check if threat is time-critical
    pub fn is_time_critical(&self) -> bool {
        self.time_to_impact_ms.is_some_and(|t| t < 1000)
    }
}

/// Response action to execute
#[derive(Debug, Clone)]
pub struct ResponseAction {
    /// Unique action identifier
    pub id: u64,
    /// Associated threat ID
    pub threat_id: u64,
    /// Type of response
    pub response_type: ResponseType,
    /// Priority level
    pub priority: u8,
    /// Whether action was executed
    pub executed: bool,
    /// Execution timestamp
    pub executed_at: Option<Instant>,
    /// Execution result message
    pub result_message: Option<String>,
}

/// Record of a response for tracking effectiveness
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ResponseRecord {
    threat_severity: ThreatSeverity,
    response_type: ResponseType,
    response_time_ms: u64,
    was_effective: bool,
    timestamp: Instant,
}

/// Configuration for threat response system
#[derive(Debug, Clone)]
pub struct ThreatResponseConfig {
    /// Maximum pending responses to track
    pub max_pending_responses: usize,
    /// History size for effectiveness tracking
    pub history_size: usize,
    /// Enable automatic escalation
    pub auto_escalate: bool,
    /// Time before escalating unresolved threats (ms)
    pub escalation_delay_ms: u64,
    /// Minimum confidence to trigger response
    pub min_confidence_threshold: f64,
    /// Response cool-down period (ms)
    pub cooldown_ms: u64,
}

impl Default for ThreatResponseConfig {
    fn default() -> Self {
        Self {
            max_pending_responses: 100,
            history_size: 1000,
            auto_escalate: true,
            escalation_delay_ms: 5000,
            min_confidence_threshold: 0.5,
            cooldown_ms: 1000,
        }
    }
}

/// Response statistics
#[derive(Debug, Clone, Default)]
pub struct ResponseStats {
    pub total_threats_received: u64,
    pub total_responses_generated: u64,
    pub total_responses_executed: u64,
    pub avg_response_time_ms: f64,
    pub escalations_triggered: u64,
    pub effectiveness_rate: f64,
}

/// Immediate threat response coordinator
pub struct ThreatResponse {
    /// Configuration
    config: ThreatResponseConfig,
    /// Active threats awaiting response
    active_threats: HashMap<u64, Threat>,
    /// Pending response actions
    pending_actions: VecDeque<ResponseAction>,
    /// Response history for effectiveness tracking
    response_history: VecDeque<ResponseRecord>,
    /// Action ID counter
    next_action_id: u64,
    /// Last response time by threat type (for cooldown)
    last_response_time: HashMap<String, Instant>,
    /// Statistics
    stats: ResponseStats,
    /// Response time accumulator for averaging
    response_time_sum_ms: u64,
}

impl Default for ThreatResponse {
    fn default() -> Self {
        Self::new()
    }
}

impl ThreatResponse {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self::with_config(ThreatResponseConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: ThreatResponseConfig) -> Self {
        Self {
            config,
            active_threats: HashMap::new(),
            pending_actions: VecDeque::new(),
            response_history: VecDeque::new(),
            next_action_id: 1,
            last_response_time: HashMap::new(),
            stats: ResponseStats::default(),
            response_time_sum_ms: 0,
        }
    }

    /// Main processing function - process all active threats
    pub fn process(&mut self) -> Result<()> {
        self.check_escalations();
        self.cleanup_old_threats();
        Ok(())
    }

    /// Register a new threat and generate response
    pub fn register_threat(&mut self, threat: Threat) -> Vec<ResponseAction> {
        self.stats.total_threats_received += 1;

        // Check confidence threshold
        if threat.confidence < self.config.min_confidence_threshold {
            return vec![];
        }

        // Check cooldown for this threat type
        if let Some(last_time) = self.last_response_time.get(&threat.threat_type) {
            if last_time.elapsed().as_millis() < self.config.cooldown_ms as u128 {
                return vec![];
            }
        }

        // Generate response based on threat
        let actions = self.generate_response(&threat);

        // Store threat for tracking
        self.active_threats.insert(threat.id, threat.clone());

        // Update cooldown
        self.last_response_time
            .insert(threat.threat_type.clone(), Instant::now());

        // Queue actions
        for action in &actions {
            if self.pending_actions.len() < self.config.max_pending_responses {
                self.pending_actions.push_back(action.clone());
            }
        }

        self.stats.total_responses_generated += actions.len() as u64;
        actions
    }

    /// Generate appropriate response for a threat
    fn generate_response(&mut self, threat: &Threat) -> Vec<ResponseAction> {
        let mut actions = Vec::new();

        let response_types = self.select_response_types(threat);

        for response_type in response_types {
            let action = ResponseAction {
                id: self.next_action_id,
                threat_id: threat.id,
                response_type,
                priority: threat.severity as u8,
                executed: false,
                executed_at: None,
                result_message: None,
            };
            self.next_action_id += 1;
            actions.push(action);
        }

        // Sort by priority (highest first)
        actions.sort_by(|a, b| b.priority.cmp(&a.priority));

        actions
    }

    /// Select appropriate response types based on threat characteristics
    fn select_response_types(&self, threat: &Threat) -> Vec<ResponseType> {
        let mut responses = Vec::new();

        match threat.severity {
            ThreatSeverity::Low => {
                responses.push(ResponseType::Monitor);
            }
            ThreatSeverity::Medium => {
                responses.push(ResponseType::ReduceExposure { reduction_pct: 25 });
            }
            ThreatSeverity::High => {
                responses.push(ResponseType::ReduceExposure { reduction_pct: 50 });
                responses.push(ResponseType::Hedge { hedge_ratio: 30 });
                responses.push(ResponseType::HaltNewOrders);
            }
            ThreatSeverity::Critical => {
                if !threat.affected_symbols.is_empty() {
                    responses.push(ResponseType::ClosePositions {
                        symbols: threat.affected_symbols.clone(),
                    });
                }
                responses.push(ResponseType::CancelPendingOrders);
                responses.push(ResponseType::HaltNewOrders);
            }
            ThreatSeverity::Emergency => {
                responses.push(ResponseType::KillSwitch);
                responses.push(ResponseType::Liquidate);
                responses.push(ResponseType::CancelPendingOrders);
            }
        }

        // Adjust for time-critical threats
        if threat.is_time_critical() && threat.severity >= ThreatSeverity::High {
            if !responses.contains(&ResponseType::CancelPendingOrders) {
                responses.insert(0, ResponseType::CancelPendingOrders);
            }
        }

        responses
    }

    /// Get the next pending action to execute
    pub fn next_action(&mut self) -> Option<ResponseAction> {
        self.pending_actions.pop_front()
    }

    /// Get all pending actions sorted by priority
    pub fn pending_actions(&self) -> Vec<&ResponseAction> {
        let mut actions: Vec<_> = self.pending_actions.iter().collect();
        actions.sort_by(|a, b| b.priority.cmp(&a.priority));
        actions
    }

    /// Mark an action as executed
    pub fn mark_executed(
        &mut self,
        action_id: u64,
        success: bool,
        message: Option<String>,
    ) -> bool {
        // Find in pending actions
        for action in self.pending_actions.iter_mut() {
            if action.id == action_id {
                action.executed = true;
                action.executed_at = Some(Instant::now());
                action.result_message = message.clone();

                self.stats.total_responses_executed += 1;

                // Record for effectiveness tracking
                if let Some(threat) = self.active_threats.get(&action.threat_id) {
                    let response_time_ms = threat.detected_at.elapsed().as_millis() as u64;
                    self.response_time_sum_ms += response_time_ms;

                    let record = ResponseRecord {
                        threat_severity: threat.severity,
                        response_type: action.response_type.clone(),
                        response_time_ms,
                        was_effective: success,
                        timestamp: Instant::now(),
                    };

                    self.response_history.push_back(record);
                    while self.response_history.len() > self.config.history_size {
                        self.response_history.pop_front();
                    }

                    // Update statistics
                    self.update_stats();
                }

                return true;
            }
        }
        false
    }

    /// Resolve a threat (remove from active tracking)
    pub fn resolve_threat(&mut self, threat_id: u64) -> Option<Threat> {
        self.active_threats.remove(&threat_id)
    }

    /// Check for threats requiring escalation
    fn check_escalations(&mut self) {
        if !self.config.auto_escalate {
            return;
        }

        let escalation_threshold = Duration::from_millis(self.config.escalation_delay_ms);
        let threats_to_escalate: Vec<u64> = self
            .active_threats
            .iter()
            .filter(|(_, threat)| {
                threat.detected_at.elapsed() > escalation_threshold
                    && threat.severity < ThreatSeverity::Emergency
            })
            .map(|(id, _)| *id)
            .collect();

        for threat_id in threats_to_escalate {
            if let Some(threat) = self.active_threats.get_mut(&threat_id) {
                // Escalate severity
                threat.severity = match threat.severity {
                    ThreatSeverity::Low => ThreatSeverity::Medium,
                    ThreatSeverity::Medium => ThreatSeverity::High,
                    ThreatSeverity::High => ThreatSeverity::Critical,
                    ThreatSeverity::Critical => ThreatSeverity::Emergency,
                    ThreatSeverity::Emergency => ThreatSeverity::Emergency,
                };

                self.stats.escalations_triggered += 1;
            }

            // Generate new responses for escalated threat (clone to avoid borrow conflict)
            if let Some(threat) = self.active_threats.get(&threat_id).cloned() {
                let new_actions = self.generate_response(&threat);
                for action in new_actions {
                    if self.pending_actions.len() < self.config.max_pending_responses {
                        self.pending_actions.push_back(action);
                    }
                }
            }
        }
    }

    /// Remove threats that have been resolved or are too old
    fn cleanup_old_threats(&mut self) {
        let max_age = Duration::from_secs(300); // 5 minutes
        self.active_threats
            .retain(|_, threat| threat.detected_at.elapsed() < max_age);
    }

    /// Update running statistics
    fn update_stats(&mut self) {
        if self.stats.total_responses_executed > 0 {
            self.stats.avg_response_time_ms =
                self.response_time_sum_ms as f64 / self.stats.total_responses_executed as f64;
        }

        if !self.response_history.is_empty() {
            let effective_count = self
                .response_history
                .iter()
                .filter(|r| r.was_effective)
                .count();
            self.stats.effectiveness_rate =
                effective_count as f64 / self.response_history.len() as f64;
        }
    }

    /// Get effectiveness rate for a specific response type
    pub fn response_effectiveness(&self, response_type: &ResponseType) -> Option<f64> {
        let relevant: Vec<_> = self
            .response_history
            .iter()
            .filter(|r| &r.response_type == response_type)
            .collect();

        if relevant.is_empty() {
            return None;
        }

        let effective = relevant.iter().filter(|r| r.was_effective).count();
        Some(effective as f64 / relevant.len() as f64)
    }

    /// Get average response time for a severity level
    pub fn avg_response_time_for_severity(&self, severity: ThreatSeverity) -> Option<f64> {
        let relevant: Vec<_> = self
            .response_history
            .iter()
            .filter(|r| r.threat_severity == severity)
            .collect();

        if relevant.is_empty() {
            return None;
        }

        let sum: u64 = relevant.iter().map(|r| r.response_time_ms).sum();
        Some(sum as f64 / relevant.len() as f64)
    }

    /// Get current statistics
    pub fn stats(&self) -> &ResponseStats {
        &self.stats
    }

    /// Get count of active threats
    pub fn active_threat_count(&self) -> usize {
        self.active_threats.len()
    }

    /// Get count of active threats by severity
    pub fn threats_by_severity(&self) -> HashMap<ThreatSeverity, usize> {
        let mut counts = HashMap::new();
        for threat in self.active_threats.values() {
            *counts.entry(threat.severity).or_insert(0) += 1;
        }
        counts
    }

    /// Clear all state (for testing or reset)
    pub fn reset(&mut self) {
        self.active_threats.clear();
        self.pending_actions.clear();
        self.response_history.clear();
        self.last_response_time.clear();
        self.stats = ResponseStats::default();
        self.response_time_sum_ms = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_threat(severity: ThreatSeverity) -> Threat {
        Threat {
            id: 1,
            source: "test_detector".to_string(),
            threat_type: "test_threat".to_string(),
            severity,
            confidence: 0.8,
            affected_symbols: vec!["BTC-USD".to_string()],
            time_to_impact_ms: Some(5000),
            context: HashMap::new(),
            detected_at: Instant::now(),
        }
    }

    #[test]
    fn test_basic() {
        let mut instance = ThreatResponse::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_low_severity_response() {
        let mut response = ThreatResponse::new();
        let threat = create_test_threat(ThreatSeverity::Low);

        let actions = response.register_threat(threat);
        assert!(!actions.is_empty());
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.response_type, ResponseType::Monitor))
        );
    }

    #[test]
    fn test_critical_severity_response() {
        let mut response = ThreatResponse::new();
        let threat = create_test_threat(ThreatSeverity::Critical);

        let actions = response.register_threat(threat);
        assert!(actions.len() >= 2);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.response_type, ResponseType::CancelPendingOrders))
        );
    }

    #[test]
    fn test_emergency_response() {
        let mut response = ThreatResponse::new();
        let threat = create_test_threat(ThreatSeverity::Emergency);

        let actions = response.register_threat(threat);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.response_type, ResponseType::KillSwitch))
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a.response_type, ResponseType::Liquidate))
        );
    }

    #[test]
    fn test_confidence_threshold() {
        let mut response = ThreatResponse::new();
        let mut threat = create_test_threat(ThreatSeverity::High);
        threat.confidence = 0.2; // Below threshold

        let actions = response.register_threat(threat);
        assert!(actions.is_empty());
    }

    #[test]
    fn test_cooldown() {
        let config = ThreatResponseConfig {
            cooldown_ms: 10000, // 10 second cooldown
            ..Default::default()
        };
        let mut response = ThreatResponse::with_config(config);

        let threat1 = create_test_threat(ThreatSeverity::Medium);
        let actions1 = response.register_threat(threat1);
        assert!(!actions1.is_empty());

        // Same threat type should be blocked by cooldown
        let mut threat2 = create_test_threat(ThreatSeverity::Medium);
        threat2.id = 2;
        let actions2 = response.register_threat(threat2);
        assert!(actions2.is_empty());
    }

    #[test]
    fn test_mark_executed() {
        let mut response = ThreatResponse::new();
        let threat = create_test_threat(ThreatSeverity::Medium);

        let actions = response.register_threat(threat);
        assert!(!actions.is_empty());

        let action_id = actions[0].id;
        assert!(response.mark_executed(action_id, true, Some("Success".to_string())));
        assert_eq!(response.stats().total_responses_executed, 1);
    }

    #[test]
    fn test_threat_score() {
        let mut threat = create_test_threat(ThreatSeverity::High);
        threat.confidence = 0.9;

        let score = threat.threat_score();
        assert!(score > 0.5);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_time_critical_detection() {
        let mut threat = create_test_threat(ThreatSeverity::High);
        threat.time_to_impact_ms = Some(500);
        assert!(threat.is_time_critical());

        threat.time_to_impact_ms = Some(2000);
        assert!(!threat.is_time_critical());
    }

    #[test]
    fn test_response_priority() {
        assert!(ResponseType::KillSwitch.priority() > ResponseType::Monitor.priority());
        assert!(
            ResponseType::Liquidate.priority()
                > ResponseType::ReduceExposure { reduction_pct: 50 }.priority()
        );
    }

    #[test]
    fn test_threats_by_severity() {
        let mut response = ThreatResponse::new();

        let mut threat1 = create_test_threat(ThreatSeverity::Low);
        threat1.id = 1;
        threat1.threat_type = "type1".to_string();
        response.register_threat(threat1);

        let mut threat2 = create_test_threat(ThreatSeverity::Low);
        threat2.id = 2;
        threat2.threat_type = "type2".to_string();
        response.register_threat(threat2);

        let mut threat3 = create_test_threat(ThreatSeverity::High);
        threat3.id = 3;
        threat3.threat_type = "type3".to_string();
        response.register_threat(threat3);

        let by_severity = response.threats_by_severity();
        assert_eq!(by_severity.get(&ThreatSeverity::Low), Some(&2));
        assert_eq!(by_severity.get(&ThreatSeverity::High), Some(&1));
    }

    #[test]
    fn test_resolve_threat() {
        let mut response = ThreatResponse::new();
        let threat = create_test_threat(ThreatSeverity::Medium);
        let threat_id = threat.id;

        response.register_threat(threat);
        assert_eq!(response.active_threat_count(), 1);

        let resolved = response.resolve_threat(threat_id);
        assert!(resolved.is_some());
        assert_eq!(response.active_threat_count(), 0);
    }

    #[test]
    fn test_reset() {
        let mut response = ThreatResponse::new();
        let threat = create_test_threat(ThreatSeverity::Medium);
        response.register_threat(threat);

        response.reset();
        assert_eq!(response.active_threat_count(), 0);
        assert!(response.pending_actions().is_empty());
    }
}
