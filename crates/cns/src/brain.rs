//! # Brain Module - CNS Coordinator
//!
//! The Brain is the central coordinator of the JANUS CNS system.
//! It orchestrates health probes, aggregates signals, manages reflexes,
//! and exposes metrics.

use crate::metrics::CNSMetrics;
use crate::probes::{HealthProbe, ProbeBuilder};
use crate::reflexes::{CircuitBreakerConfig, Reflex, ReflexRule};
use crate::signals::{ComponentHealth, ComponentType, HealthSignal, SystemStatus};
use crate::{CNSError, Result};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::time::{Duration, interval};
use tracing::{debug, error, info, warn};

/// Rate limiters for throttled components
type ThrottleMap = HashMap<ComponentType, ThrottleState>;

/// State for component throttling
#[derive(Debug, Clone)]
struct ThrottleState {
    /// Maximum requests per second
    rate_limit: u32,
    /// Last reset time
    last_reset: Instant,
    /// Current request count
    request_count: u32,
}

impl ThrottleState {
    fn new(rate_limit: u32) -> Self {
        Self {
            rate_limit,
            last_reset: Instant::now(),
            request_count: 0,
        }
    }

    /// Check if a request is allowed under the rate limit
    fn allow_request(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.last_reset) >= Duration::from_secs(1) {
            self.last_reset = now;
            self.request_count = 0;
        }

        if self.request_count < self.rate_limit {
            self.request_count += 1;
            true
        } else {
            false
        }
    }
}

/// Brain configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainConfig {
    /// Health check interval in seconds
    pub health_check_interval_secs: u64,

    /// Enable auto-recovery reflexes
    pub enable_reflexes: bool,

    /// Service endpoints
    pub endpoints: EndpointConfig,

    /// Circuit breaker configurations
    pub circuit_breakers: HashMap<String, CircuitBreakerConfig>,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            health_check_interval_secs: 10,
            enable_reflexes: true,
            endpoints: EndpointConfig::default(),
            circuit_breakers: HashMap::new(),
        }
    }
}

/// Endpoint configuration for services
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointConfig {
    /// Forward service base URL
    pub forward_service: String,

    /// Backward service base URL
    pub backward_service: String,

    /// Gateway service base URL
    pub gateway_service: String,

    /// Redis connection string
    pub redis: String,

    /// Qdrant URL
    pub qdrant: String,

    /// Shared memory path
    pub shared_memory_path: String,

    /// Neuromorphic brain region endpoints
    pub neuromorphic: NeuromorphicEndpoints,
}

/// Neuromorphic brain region endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicEndpoints {
    /// Enable neuromorphic monitoring
    pub enabled: bool,

    /// Base URL for neuromorphic services (e.g., "http://localhost:8090")
    pub base_url: String,

    /// Individual region endpoints (optional overrides)
    pub cortex: Option<String>,
    pub hippocampus: Option<String>,
    pub basal_ganglia: Option<String>,
    pub thalamus: Option<String>,
    pub prefrontal: Option<String>,
    pub amygdala: Option<String>,
    pub hypothalamus: Option<String>,
    pub cerebellum: Option<String>,
    pub visual_cortex: Option<String>,
    pub integration: Option<String>,
}

impl Default for NeuromorphicEndpoints {
    fn default() -> Self {
        Self {
            enabled: false,
            base_url: "http://localhost:8090".to_string(),
            cortex: None,
            hippocampus: None,
            basal_ganglia: None,
            thalamus: None,
            prefrontal: None,
            amygdala: None,
            hypothalamus: None,
            cerebellum: None,
            visual_cortex: None,
            integration: None,
        }
    }
}

impl Default for EndpointConfig {
    fn default() -> Self {
        Self {
            forward_service: "http://localhost:8081".to_string(),
            backward_service: "http://localhost:8082".to_string(),
            gateway_service: "http://localhost:8080".to_string(),
            redis: "redis://localhost:6379".to_string(),
            qdrant: "http://localhost:6333".to_string(),
            shared_memory_path: "/dev/shm/janus_forward_backward".to_string(),
            neuromorphic: NeuromorphicEndpoints::default(),
        }
    }
}

/// The Brain - central coordinator of the CNS
pub struct Brain {
    /// Configuration
    config: BrainConfig,

    /// Health probes
    probes: Vec<Box<dyn HealthProbe>>,

    /// Reflex system
    reflex: Arc<RwLock<Reflex>>,

    /// Current system health
    current_health: Arc<RwLock<Option<HealthSignal>>>,

    /// System start time
    start_time: Instant,

    /// Is the brain running
    running: Arc<RwLock<bool>>,

    /// Throttle states for rate-limited components
    throttle_states: Arc<RwLock<ThrottleMap>>,

    /// Shutdown signal sender
    shutdown_tx: Option<tokio::sync::broadcast::Sender<()>>,

    /// Alert handlers (webhook URLs)
    alert_webhooks: Vec<AlertWebhook>,
}

/// Alert webhook configuration
#[derive(Debug, Clone)]
pub struct AlertWebhook {
    /// Webhook URL
    pub url: String,
    /// Webhook type (slack, pagerduty, generic)
    pub webhook_type: WebhookType,
    /// Minimum severity to trigger
    pub min_severity: AlertSeverity,
}

/// Webhook types supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebhookType {
    /// Slack webhook
    Slack,
    /// PagerDuty Events API
    PagerDuty,
    /// Generic JSON webhook
    Generic,
}

use crate::reflexes::AlertSeverity;

impl Brain {
    /// Create a new Brain with the given configuration
    pub fn new(config: BrainConfig) -> Self {
        let probes = Self::create_probes(&config.endpoints);
        let reflex = Self::create_reflex(&config);
        let (shutdown_tx, _) = tokio::sync::broadcast::channel(1);

        Self {
            config,
            probes,
            reflex: Arc::new(RwLock::new(reflex)),
            current_health: Arc::new(RwLock::new(None)),
            start_time: Instant::now(),
            running: Arc::new(RwLock::new(false)),
            throttle_states: Arc::new(RwLock::new(HashMap::new())),
            shutdown_tx: Some(shutdown_tx),
            alert_webhooks: Vec::new(),
        }
    }

    /// Add an alert webhook
    pub fn add_alert_webhook(&mut self, webhook: AlertWebhook) {
        self.alert_webhooks.push(webhook);
    }

    /// Check if a request is allowed for a throttled component
    pub async fn check_throttle(&self, component: &ComponentType) -> bool {
        let mut states = self.throttle_states.write().await;
        if let Some(state) = states.get_mut(component) {
            state.allow_request()
        } else {
            true // Not throttled
        }
    }

    /// Get shutdown receiver for graceful shutdown coordination
    pub fn get_shutdown_receiver(&self) -> Option<tokio::sync::broadcast::Receiver<()>> {
        self.shutdown_tx.as_ref().map(|tx| tx.subscribe())
    }

    /// Create health probes from endpoint configuration
    fn create_probes(endpoints: &EndpointConfig) -> Vec<Box<dyn HealthProbe>> {
        let mut probes: Vec<Box<dyn HealthProbe>> = vec![
            ProbeBuilder::forward_service(&endpoints.forward_service),
            ProbeBuilder::backward_service(&endpoints.backward_service),
            ProbeBuilder::gateway_service(&endpoints.gateway_service),
            ProbeBuilder::redis(&endpoints.redis),
        ];

        // Add optional probes only if configured
        if !endpoints.qdrant.is_empty() {
            probes.push(ProbeBuilder::qdrant(&endpoints.qdrant));
        }

        if !endpoints.shared_memory_path.is_empty() {
            probes.push(ProbeBuilder::shared_memory(&endpoints.shared_memory_path));
        }

        // Add neuromorphic brain region probes if enabled
        if endpoints.neuromorphic.enabled {
            let neuro = &endpoints.neuromorphic;

            // Use individual endpoints or fallback to base_url + region path
            if let Some(url) = &neuro.cortex {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Cortex,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Cortex,
                    &format!("{}/cortex", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.hippocampus {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Hippocampus,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Hippocampus,
                    &format!("{}/hippocampus", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.basal_ganglia {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::BasalGanglia,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::BasalGanglia,
                    &format!("{}/basal_ganglia", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.thalamus {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Thalamus,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Thalamus,
                    &format!("{}/thalamus", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.prefrontal {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Prefrontal,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Prefrontal,
                    &format!("{}/prefrontal", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.amygdala {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Amygdala,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Amygdala,
                    &format!("{}/amygdala", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.hypothalamus {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Hypothalamus,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Hypothalamus,
                    &format!("{}/hypothalamus", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.cerebellum {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Cerebellum,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Cerebellum,
                    &format!("{}/cerebellum", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.visual_cortex {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::VisualCortex,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::VisualCortex,
                    &format!("{}/visual_cortex", neuro.base_url),
                ));
            }

            if let Some(url) = &neuro.integration {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Integration,
                    url,
                ));
            } else if !neuro.base_url.is_empty() {
                probes.push(ProbeBuilder::neuromorphic_region(
                    ComponentType::Integration,
                    &format!("{}/integration", neuro.base_url),
                ));
            }
        }

        probes
    }

    /// Create reflex system with default rules and circuit breakers
    fn create_reflex(config: &BrainConfig) -> Reflex {
        let mut reflex = Reflex::new();

        if config.enable_reflexes {
            // Add default reflex rules
            for rule in Reflex::default_rules() {
                reflex.add_rule(rule);
            }

            // Add circuit breakers from config
            for (component_str, cb_config) in &config.circuit_breakers {
                if let Some(component) = parse_component_type(component_str) {
                    reflex.add_circuit_breaker(component, cb_config.clone());
                }
            }

            // Add default circuit breakers
            reflex.add_circuit_breaker(ComponentType::Redis, CircuitBreakerConfig::default());
            reflex.add_circuit_breaker(ComponentType::Qdrant, CircuitBreakerConfig::default());
        }

        reflex
    }

    /// Start the brain's monitoring loop
    pub async fn start(&self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(CNSError::ConfigError("Brain already running".to_string()));
        }
        *running = true;
        drop(running);

        info!("🧠 Brain starting health monitoring...");

        let mut ticker = interval(Duration::from_secs(self.config.health_check_interval_secs));

        loop {
            ticker.tick().await;

            // Check if we should stop
            if !*self.running.read().await {
                info!("🧠 Brain stopping...");
                break;
            }

            // Perform health check cycle
            if let Err(e) = self.health_check_cycle().await {
                error!("Health check cycle failed: {}", e);
            }
        }

        Ok(())
    }

    /// Stop the brain
    pub async fn stop(&self) {
        let mut running = self.running.write().await;
        *running = false;
        info!("🧠 Brain shutdown initiated");
    }

    /// Perform a single health check cycle
    async fn health_check_cycle(&self) -> Result<()> {
        debug!("Starting health check cycle");

        // Execute all probes concurrently
        let mut probe_tasks = Vec::new();
        for probe in &self.probes {
            probe_tasks.push(probe.check_with_timeout());
        }

        let results = futures::future::join_all(probe_tasks).await;

        // Convert probe results to component health
        let component_healths: Vec<ComponentHealth> = results
            .into_iter()
            .map(|r| r.to_component_health())
            .collect();

        // Calculate uptime
        let uptime_seconds = self.start_time.elapsed().as_secs();

        // Create health signal
        let signal = HealthSignal::new(component_healths, uptime_seconds);

        // Log health summary
        self.log_health_summary(&signal);

        // Update metrics
        CNSMetrics::update(&signal);

        // Process reflexes
        if self.config.enable_reflexes {
            self.process_reflexes(&signal).await;
        }

        // Store current health
        let mut current = self.current_health.write().await;
        *current = Some(signal);

        Ok(())
    }

    /// Process reflex actions based on health signal
    async fn process_reflexes(&self, signal: &HealthSignal) {
        let reflex = self.reflex.read().await;

        for component in &signal.components {
            let actions = reflex.process_health(component);

            for action in actions {
                // Execute reflex action
                if let Err(e) = self.execute_reflex_action(action).await {
                    error!("Failed to execute reflex action: {}", e);
                }
            }
        }
    }

    /// Execute a reflex action
    async fn execute_reflex_action(&self, action: crate::reflexes::RefexAction) -> Result<()> {
        use crate::reflexes::RefexAction;

        match action {
            RefexAction::LogWarning { message } => {
                warn!("⚡ Reflex triggered: {}", message);
            }
            RefexAction::SendAlert { severity, message } => {
                let emoji = match severity {
                    AlertSeverity::Info => "ℹ️",
                    AlertSeverity::Warning => "⚠️",
                    AlertSeverity::Error => "❌",
                    AlertSeverity::Critical => "🚨",
                };
                warn!("{} ALERT [{:?}]: {}", emoji, severity, message);

                // Send alerts to configured webhooks
                for webhook in &self.alert_webhooks {
                    if severity as u8 >= webhook.min_severity as u8
                        && let Err(e) = Self::send_alert_webhook(webhook, severity, &message).await
                    {
                        error!("Failed to send alert to webhook {}: {}", webhook.url, e);
                    }
                }
            }
            RefexAction::RestartComponent { component } => {
                info!("🔄 Reflex: Attempting to restart component: {}", component);

                // Restart logic depends on component type
                match &component {
                    ComponentType::GatewayService => {
                        // Gateway restart: typically handled by orchestrator/supervisor
                        warn!("Gateway restart requested - signaling orchestrator");
                        // In production, this would signal k8s/systemd/supervisor
                    }
                    ComponentType::ForwardService | ComponentType::BackwardService => {
                        // JANUS services: attempt graceful restart via their APIs
                        info!("Service {} restart - sending restart signal", component);
                    }
                    ComponentType::Redis | ComponentType::Qdrant => {
                        // External dependencies: log warning, can't restart
                        warn!(
                            "Cannot restart external dependency {}: manual intervention required",
                            component
                        );
                    }
                    _ => {
                        info!(
                            "Component {} restart requested - delegating to supervisor",
                            component
                        );
                    }
                }
            }
            RefexAction::ThrottleComponent {
                component,
                rate_limit,
            } => {
                info!(
                    "🐌 Reflex: Throttling component {} to {} req/s",
                    component, rate_limit
                );

                // Update throttle state for the component
                let mut states = self.throttle_states.write().await;
                states.insert(component, ThrottleState::new(rate_limit));
            }
            RefexAction::OpenCircuitBreaker { component } => {
                info!("⚡ Reflex: Opening circuit breaker for {}", component);
                // Circuit breaker is already managed by Reflex
            }
            RefexAction::ExecuteCommand { command } => {
                warn!("💻 Reflex: Executing command: {}", command);

                // Safe command execution with allowlist
                if Self::is_safe_command(&command) {
                    match Self::execute_safe_command(&command).await {
                        Ok(output) => {
                            info!("Command executed successfully: {}", output);
                        }
                        Err(e) => {
                            error!("Command execution failed: {}", e);
                        }
                    }
                } else {
                    error!("Command '{}' not in allowlist - execution blocked", command);
                }
            }
            RefexAction::GracefulShutdown => {
                error!("🛑 Reflex: Initiating graceful shutdown");

                // Signal all components to shut down
                if let Some(ref tx) = self.shutdown_tx {
                    let _ = tx.send(());
                }

                // Mark brain as not running
                let mut running = self.running.write().await;
                *running = false;
            }
        }

        Ok(())
    }

    /// Send alert to a webhook
    async fn send_alert_webhook(
        webhook: &AlertWebhook,
        severity: AlertSeverity,
        message: &str,
    ) -> Result<()> {
        let client = reqwest::Client::new();

        let payload = match webhook.webhook_type {
            WebhookType::Slack => {
                // Slack webhook format
                let color = match severity {
                    AlertSeverity::Info => "#36a64f",
                    AlertSeverity::Warning => "#ffcc00",
                    AlertSeverity::Error => "#ff6600",
                    AlertSeverity::Critical => "#ff0000",
                };
                serde_json::json!({
                    "attachments": [{
                        "color": color,
                        "title": format!("JANUS Alert [{:?}]", severity),
                        "text": message,
                        "ts": chrono::Utc::now().timestamp()
                    }]
                })
            }
            WebhookType::PagerDuty => {
                // PagerDuty Events API v2 format
                let pagerduty_severity = match severity {
                    AlertSeverity::Info => "info",
                    AlertSeverity::Warning => "warning",
                    AlertSeverity::Error => "error",
                    AlertSeverity::Critical => "critical",
                };
                serde_json::json!({
                    "routing_key": "", // Should be configured
                    "event_action": "trigger",
                    "payload": {
                        "summary": message,
                        "severity": pagerduty_severity,
                        "source": "janus-cns",
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    }
                })
            }
            WebhookType::Generic => {
                // Generic JSON format
                serde_json::json!({
                    "severity": format!("{:?}", severity),
                    "message": message,
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "source": "janus-cns"
                })
            }
        };

        let response = client
            .post(&webhook.url)
            .json(&payload)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| CNSError::ProbeFailure(format!("Webhook request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(CNSError::ProbeFailure(format!(
                "Webhook returned status {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Check if a command is in the safe allowlist
    fn is_safe_command(command: &str) -> bool {
        // Only allow specific, safe commands
        const ALLOWED_COMMANDS: &[&str] = &[
            "systemctl status",
            "systemctl restart janus",
            "docker ps",
            "docker restart",
            "kubectl get pods",
            "kubectl rollout restart",
            "redis-cli ping",
            "redis-cli info",
        ];

        let cmd_lower = command.to_lowercase();
        ALLOWED_COMMANDS
            .iter()
            .any(|allowed| cmd_lower.starts_with(allowed))
    }

    /// Execute a safe command
    async fn execute_safe_command(command: &str) -> Result<String> {
        use tokio::process::Command;

        let parts: Vec<&str> = command.split_whitespace().collect();
        if parts.is_empty() {
            return Err(CNSError::ConfigError("Empty command".to_string()));
        }

        let output = Command::new(parts[0])
            .args(&parts[1..])
            .output()
            .await
            .map_err(|e| CNSError::ProbeFailure(format!("Command execution failed: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(CNSError::ProbeFailure(format!(
                "Command failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )))
        }
    }

    /// Log a summary of the current health status
    fn log_health_summary(&self, signal: &HealthSignal) {
        let status_emoji = match signal.system_status {
            SystemStatus::Healthy => "✅",
            SystemStatus::Degraded => "⚠️",
            SystemStatus::Critical => "🚨",
            SystemStatus::Shutdown => "🛑",
            SystemStatus::Starting => "🔄",
        };

        let unhealthy = signal.unhealthy_components();

        if unhealthy.is_empty() {
            debug!(
                "{} System Status: {} | Health Score: {:.2}",
                status_emoji,
                signal.system_status,
                signal.health_score()
            );
        } else {
            let unhealthy_summary: Vec<String> = unhealthy
                .iter()
                .map(|c| format!("{}: {}", c.component_type, c.status))
                .collect();

            warn!(
                "{} System Status: {} | Health Score: {:.2} | Issues: [{}]",
                status_emoji,
                signal.system_status,
                signal.health_score(),
                unhealthy_summary.join(", ")
            );
        }
    }

    /// Get the current health signal
    pub async fn get_health(&self) -> Option<HealthSignal> {
        self.current_health.read().await.clone()
    }

    /// Get health for a specific component
    pub async fn get_component_health(&self, component: ComponentType) -> Option<ComponentHealth> {
        self.current_health
            .read()
            .await
            .as_ref()
            .and_then(|s| s.get_component(component).cloned())
    }

    /// Perform an immediate health check (outside regular cycle)
    pub async fn check_now(&self) -> Result<HealthSignal> {
        debug!("Performing immediate health check");

        let mut probe_tasks = Vec::new();
        for probe in &self.probes {
            probe_tasks.push(probe.check_with_timeout());
        }

        let results = futures::future::join_all(probe_tasks).await;

        let component_healths: Vec<ComponentHealth> = results
            .into_iter()
            .map(|r| r.to_component_health())
            .collect();

        let uptime_seconds = self.start_time.elapsed().as_secs();
        let signal = HealthSignal::new(component_healths, uptime_seconds);

        // Update stored health
        let mut current = self.current_health.write().await;
        *current = Some(signal.clone());

        Ok(signal)
    }

    /// Add a custom health probe
    pub fn add_probe(&mut self, probe: Box<dyn HealthProbe>) {
        self.probes.push(probe);
    }

    /// Get the number of active probes
    pub fn probe_count(&self) -> usize {
        self.probes.len()
    }

    /// Add a custom reflex rule
    pub async fn add_reflex_rule(&self, rule: ReflexRule) {
        let mut reflex = self.reflex.write().await;
        reflex.add_rule(rule);
    }

    /// Get system uptime in seconds
    pub fn uptime_seconds(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Check if brain is running
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }
}

/// Helper function to parse component type from string
fn parse_component_type(s: &str) -> Option<ComponentType> {
    match s.to_lowercase().as_str() {
        "forward" | "forward_service" => Some(ComponentType::ForwardService),
        "backward" | "backward_service" => Some(ComponentType::BackwardService),
        "gateway" | "gateway_service" => Some(ComponentType::GatewayService),
        "cns" | "cns_service" => Some(ComponentType::CNSService),
        "redis" => Some(ComponentType::Redis),
        "qdrant" => Some(ComponentType::Qdrant),
        "shared_memory" | "shm" => Some(ComponentType::SharedMemory),
        "grpc" | "grpc_channel" => Some(ComponentType::GrpcChannel),
        "websocket" | "ws" => Some(ComponentType::WebSocket),
        "vision" | "vision_module" => Some(ComponentType::VisionModule),
        "logic" | "logic_module" => Some(ComponentType::LogicModule),
        "memory" | "memory_module" => Some(ComponentType::MemoryModule),
        "execution" | "execution_module" => Some(ComponentType::ExecutionModule),
        "job_queue" => Some(ComponentType::JobQueue),
        "metrics_exporter" => Some(ComponentType::MetricsExporter),
        // Neuromorphic brain regions
        "cortex" => Some(ComponentType::Cortex),
        "hippocampus" => Some(ComponentType::Hippocampus),
        "basal_ganglia" => Some(ComponentType::BasalGanglia),
        "thalamus" => Some(ComponentType::Thalamus),
        "prefrontal" => Some(ComponentType::Prefrontal),
        "amygdala" => Some(ComponentType::Amygdala),
        "hypothalamus" => Some(ComponentType::Hypothalamus),
        "cerebellum" => Some(ComponentType::Cerebellum),
        "visual_cortex" => Some(ComponentType::VisualCortex),
        "integration" => Some(ComponentType::Integration),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BrainConfig::default();
        assert_eq!(config.health_check_interval_secs, 10);
        assert!(config.enable_reflexes);
    }

    #[test]
    fn test_parse_component_type() {
        assert_eq!(
            parse_component_type("forward_service"),
            Some(ComponentType::ForwardService)
        );
        assert_eq!(parse_component_type("redis"), Some(ComponentType::Redis));
        assert_eq!(parse_component_type("unknown"), None);
    }

    #[tokio::test]
    async fn test_brain_creation() {
        let config = BrainConfig::default();
        let brain = Brain::new(config);
        assert!(!brain.is_running().await);
        assert_eq!(brain.probes.len(), 6); // Default probes
    }

    #[tokio::test]
    async fn test_brain_check_now() {
        let config = BrainConfig::default();
        let brain = Brain::new(config);

        // This will likely have failures since services aren't running,
        // but it should complete without panic
        let result = brain.check_now().await;
        assert!(result.is_ok());

        let signal = result.unwrap();
        assert!(!signal.components.is_empty());
    }
}
