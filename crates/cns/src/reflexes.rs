//! # Reflexes Module
//!
//! Implements auto-recovery mechanisms and circuit breakers for the JANUS CNS.
//! These act as "reflex arcs" - automatic responses to system health issues.

use crate::alerts::{AlertConfig, AlertManager};
use crate::signals::{ComponentHealth, ComponentType, ProbeStatus};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tracing::{info, warn};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Circuit is closed, requests flow normally
    Closed,
    /// Circuit is open, requests are blocked
    Open,
    /// Circuit is half-open, testing if service recovered
    HalfOpen,
}

impl CircuitState {
    /// Convert to integer for metrics
    pub fn to_int(&self) -> i64 {
        match self {
            CircuitState::Closed => 0,
            CircuitState::Open => 1,
            CircuitState::HalfOpen => 2,
        }
    }
}

/// Configuration for circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitBreakerConfig {
    /// Number of failures before opening circuit
    pub failure_threshold: u32,
    /// Time window for counting failures (seconds)
    pub failure_window_secs: i64,
    /// Time to wait before attempting recovery (seconds)
    pub recovery_timeout_secs: i64,
    /// Number of successful requests needed to close circuit from half-open
    pub success_threshold: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            failure_window_secs: 60,
            recovery_timeout_secs: 30,
            success_threshold: 3,
        }
    }
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    component: ComponentType,
    config: CircuitBreakerConfig,
    state: Arc<RwLock<CircuitBreakerState>>,
}

#[derive(Debug)]
struct CircuitBreakerState {
    current_state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<DateTime<Utc>>,
    last_state_change: DateTime<Utc>,
    trip_count: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    pub fn new(component: ComponentType, config: CircuitBreakerConfig) -> Self {
        Self {
            component,
            config,
            state: Arc::new(RwLock::new(CircuitBreakerState {
                current_state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
                last_state_change: Utc::now(),
                trip_count: 0,
            })),
        }
    }

    /// Get current circuit state
    pub fn state(&self) -> CircuitState {
        self.state.read().unwrap().current_state
    }

    /// Get number of times circuit has tripped
    pub fn trip_count(&self) -> u64 {
        self.state.read().unwrap().trip_count
    }

    /// Record a successful operation
    pub fn record_success(&self) {
        let mut state = self.state.write().unwrap();

        match state.current_state {
            CircuitState::Closed => {
                // Reset failure count on success
                state.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                state.success_count += 1;
                if state.success_count >= self.config.success_threshold {
                    info!(
                        component = %self.component,
                        "Circuit breaker closing after {} successes",
                        state.success_count
                    );
                    state.current_state = CircuitState::Closed;
                    state.failure_count = 0;
                    state.success_count = 0;
                    state.last_state_change = Utc::now();
                }
            }
            CircuitState::Open => {
                // Check if recovery timeout has elapsed
                if self.should_attempt_recovery(&state) {
                    info!(component = %self.component, "Circuit breaker entering half-open state");
                    state.current_state = CircuitState::HalfOpen;
                    state.success_count = 1;
                    state.last_state_change = Utc::now();
                }
            }
        }
    }

    /// Record a failed operation
    pub fn record_failure(&self) {
        let mut state = self.state.write().unwrap();
        state.last_failure_time = Some(Utc::now());

        match state.current_state {
            CircuitState::Closed => {
                // Check if failure is within time window
                if self.is_within_failure_window(&state) {
                    state.failure_count += 1;
                } else {
                    // Reset count if outside window
                    state.failure_count = 1;
                }

                if state.failure_count >= self.config.failure_threshold {
                    warn!(
                        component = %self.component,
                        failures = state.failure_count,
                        "Circuit breaker tripping to OPEN state"
                    );
                    state.current_state = CircuitState::Open;
                    state.trip_count += 1;
                    state.last_state_change = Utc::now();
                }
            }
            CircuitState::HalfOpen => {
                warn!(
                    component = %self.component,
                    "Circuit breaker returning to OPEN state after failure in half-open"
                );
                state.current_state = CircuitState::Open;
                state.failure_count = 1;
                state.success_count = 0;
                state.trip_count += 1;
                state.last_state_change = Utc::now();
            }
            CircuitState::Open => {
                // Already open, just track the failure
                state.failure_count += 1;
            }
        }
    }

    /// Check if operation is allowed
    pub fn is_call_permitted(&self) -> bool {
        let state = self.state.read().unwrap();
        match state.current_state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                drop(state);
                if self.should_attempt_recovery_now() {
                    let mut state = self.state.write().unwrap();
                    if state.current_state == CircuitState::Open {
                        state.current_state = CircuitState::HalfOpen;
                        state.last_state_change = Utc::now();
                        true
                    } else {
                        state.current_state == CircuitState::HalfOpen
                    }
                } else {
                    false
                }
            }
        }
    }

    fn is_within_failure_window(&self, state: &CircuitBreakerState) -> bool {
        if let Some(last_failure) = state.last_failure_time {
            let window = Duration::seconds(self.config.failure_window_secs);
            Utc::now().signed_duration_since(last_failure) < window
        } else {
            false
        }
    }

    fn should_attempt_recovery(&self, state: &CircuitBreakerState) -> bool {
        let timeout = Duration::seconds(self.config.recovery_timeout_secs);
        Utc::now().signed_duration_since(state.last_state_change) >= timeout
    }

    fn should_attempt_recovery_now(&self) -> bool {
        let state = self.state.read().unwrap();
        self.should_attempt_recovery(&state)
    }
}

/// Reflex action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefexAction {
    /// Log a warning
    LogWarning { message: String },
    /// Send an alert
    SendAlert {
        severity: AlertSeverity,
        message: String,
    },
    /// Restart a component
    RestartComponent { component: ComponentType },
    /// Throttle requests to a component
    ThrottleComponent {
        component: ComponentType,
        rate_limit: u32,
    },
    /// Open circuit breaker
    OpenCircuitBreaker { component: ComponentType },
    /// Execute custom command
    ExecuteCommand { command: String },
    /// Trigger graceful shutdown
    GracefulShutdown,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Reflex rule - condition and action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflexRule {
    /// Rule ID
    pub id: String,
    /// Rule description
    pub description: String,
    /// Condition to trigger the reflex
    pub condition: ReflexCondition,
    /// Action to execute
    pub action: RefexAction,
    /// Cooldown period in seconds (prevent repeated actions)
    pub cooldown_secs: i64,
}

/// Reflex conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReflexCondition {
    /// Component is down
    ComponentDown { component: ComponentType },
    /// Component is degraded
    ComponentDegraded { component: ComponentType },
    /// Response time exceeds threshold (ms)
    ResponseTimeSlow {
        component: ComponentType,
        threshold_ms: u64,
    },
    /// Health score below threshold
    HealthScoreBelow { threshold: f64 },
    /// Multiple components down
    MultipleComponentsDown { count: usize },
    /// Custom condition (for future extension)
    Custom { name: String },
}

impl ReflexCondition {
    /// Check if condition is met
    pub fn is_met(&self, health: &ComponentHealth) -> bool {
        match self {
            ReflexCondition::ComponentDown { component } => {
                health.component_type == *component && health.status == ProbeStatus::Down
            }
            ReflexCondition::ComponentDegraded { component } => {
                health.component_type == *component && health.status == ProbeStatus::Degraded
            }
            ReflexCondition::ResponseTimeSlow {
                component,
                threshold_ms,
            } => {
                health.component_type == *component
                    && health.response_time_ms.unwrap_or(0) > *threshold_ms
            }
            ReflexCondition::HealthScoreBelow { threshold } => health.status.score() < *threshold,
            _ => false, // Other conditions need system-level data
        }
    }
}

/// Reflex system - manages automatic recovery actions
pub struct Reflex {
    rules: Vec<ReflexRule>,
    circuit_breakers: Arc<RwLock<HashMap<ComponentType, CircuitBreaker>>>,
    last_action_times: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    alert_manager: Option<AlertManager>,
}

impl Reflex {
    /// Create a new reflex system
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            last_action_times: Arc::new(RwLock::new(HashMap::new())),
            alert_manager: None,
        }
    }

    /// Create a reflex system with alert manager
    pub fn with_alerts(mut self, config: AlertConfig) -> Self {
        self.alert_manager = Some(AlertManager::new(config));
        self
    }

    /// Set alert manager
    pub fn set_alert_manager(&mut self, config: AlertConfig) {
        self.alert_manager = Some(AlertManager::new(config));
    }

    /// Add a reflex rule
    pub fn add_rule(&mut self, rule: ReflexRule) {
        self.rules.push(rule);
    }

    /// Add a circuit breaker for a component
    pub fn add_circuit_breaker(&self, component: ComponentType, config: CircuitBreakerConfig) {
        let breaker = CircuitBreaker::new(component, config);
        self.circuit_breakers
            .write()
            .unwrap()
            .insert(component, breaker);
    }

    /// Get circuit breaker for a component
    pub fn get_circuit_breaker(&self, component: ComponentType) -> Option<CircuitState> {
        self.circuit_breakers
            .read()
            .unwrap()
            .get(&component)
            .map(|b| b.state())
    }

    /// Process health update and execute reflexes if needed
    pub fn process_health(&self, health: &ComponentHealth) -> Vec<RefexAction> {
        let mut actions = Vec::new();

        // Update circuit breaker
        if let Some(breaker) = self
            .circuit_breakers
            .read()
            .unwrap()
            .get(&health.component_type)
        {
            match health.status {
                ProbeStatus::Up => breaker.record_success(),
                ProbeStatus::Down => breaker.record_failure(),
                _ => {}
            }
        }

        // Check reflex rules
        for rule in &self.rules {
            if rule.condition.is_met(health)
                && self.should_execute_action(&rule.id, rule.cooldown_secs)
            {
                info!(
                    "Reflex triggered: {} for {:?}",
                    rule.description, health.component_type
                );
                actions.push(rule.action.clone());
                self.record_action_time(&rule.id);

                // Send alert if action is SendAlert and alert manager is configured
                if let RefexAction::SendAlert { severity, message } = &rule.action {
                    self.send_alert_async(*severity, &rule.description, message);
                }
            }
        }

        actions
    }

    /// Send alert asynchronously (non-blocking)
    fn send_alert_async(&self, severity: AlertSeverity, title: &str, message: &str) {
        if let Some(ref alert_manager) = self.alert_manager {
            let alert_manager = alert_manager.clone();
            let title = title.to_string();
            let message = message.to_string();

            tokio::spawn(async move {
                match alert_manager.send_alert(severity, title, message).await {
                    Ok(result) => {
                        if result.any_success() {
                            info!("Alert sent successfully");
                        } else {
                            warn!("Failed to send alert to any destination");
                        }
                    }
                    Err(e) => {
                        warn!("Error sending alert: {}", e);
                    }
                }
            });
        }
    }

    /// Check if action should be executed (respecting cooldown)
    fn should_execute_action(&self, rule_id: &str, cooldown_secs: i64) -> bool {
        let last_times = self.last_action_times.read().unwrap();
        if let Some(last_time) = last_times.get(rule_id) {
            let elapsed = Utc::now().signed_duration_since(*last_time);
            elapsed >= Duration::seconds(cooldown_secs)
        } else {
            true
        }
    }

    /// Record action execution time
    fn record_action_time(&self, rule_id: &str) {
        self.last_action_times
            .write()
            .unwrap()
            .insert(rule_id.to_string(), Utc::now());
    }

    /// Get default reflex rules for JANUS
    pub fn default_rules() -> Vec<ReflexRule> {
        vec![
            ReflexRule {
                id: "forward_down_alert".to_string(),
                description: "Alert when Forward service is down".to_string(),
                condition: ReflexCondition::ComponentDown {
                    component: ComponentType::ForwardService,
                },
                action: RefexAction::SendAlert {
                    severity: AlertSeverity::Critical,
                    message: "Forward service is DOWN - trading halted".to_string(),
                },
                cooldown_secs: 300, // 5 minutes
            },
            ReflexRule {
                id: "backward_down_alert".to_string(),
                description: "Alert when Backward service is down".to_string(),
                condition: ReflexCondition::ComponentDown {
                    component: ComponentType::BackwardService,
                },
                action: RefexAction::SendAlert {
                    severity: AlertSeverity::Error,
                    message: "Backward service is DOWN - training/memory disabled".to_string(),
                },
                cooldown_secs: 300,
            },
            ReflexRule {
                id: "redis_down_circuit".to_string(),
                description: "Open circuit breaker when Redis is down".to_string(),
                condition: ReflexCondition::ComponentDown {
                    component: ComponentType::Redis,
                },
                action: RefexAction::OpenCircuitBreaker {
                    component: ComponentType::Redis,
                },
                cooldown_secs: 60,
            },
            ReflexRule {
                id: "qdrant_slow_warning".to_string(),
                description: "Warn when Qdrant is slow".to_string(),
                condition: ReflexCondition::ResponseTimeSlow {
                    component: ComponentType::Qdrant,
                    threshold_ms: 1000,
                },
                action: RefexAction::LogWarning {
                    message: "Qdrant response time exceeds 1000ms".to_string(),
                },
                cooldown_secs: 120,
            },
        ]
    }
}

impl Default for Reflex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_breaker_closed_to_open() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(ComponentType::Redis, config);

        assert_eq!(breaker.state(), CircuitState::Closed);
        assert!(breaker.is_call_permitted());

        breaker.record_failure();
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);

        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Open);
        assert!(!breaker.is_call_permitted());
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(ComponentType::Redis, config);

        breaker.record_failure();
        breaker.record_failure();
        breaker.record_success();

        // Success should reset failure count
        breaker.record_failure();
        assert_eq!(breaker.state(), CircuitState::Closed);
    }

    #[test]
    fn test_reflex_condition_component_down() {
        let condition = ReflexCondition::ComponentDown {
            component: ComponentType::Redis,
        };

        let health_down = ComponentHealth::unhealthy(ComponentType::Redis, "error");
        assert!(condition.is_met(&health_down));

        let health_up = ComponentHealth::healthy(ComponentType::Redis);
        assert!(!condition.is_met(&health_up));
    }

    #[test]
    fn test_reflex_cooldown() {
        let reflex = Reflex::new();
        let rule_id = "test_rule";

        // First execution should be allowed
        assert!(reflex.should_execute_action(rule_id, 60));

        // Record execution
        reflex.record_action_time(rule_id);

        // Second execution should be blocked (cooldown)
        assert!(!reflex.should_execute_action(rule_id, 60));
    }
}
