//! # Component Restart Module
//!
//! Implements automatic component restart logic for the JANUS CNS.
//! Supports both Docker-based and systemd-based component management.

use crate::Result;
use crate::signals::ComponentType;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Configuration for component restart behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestartConfig {
    /// Maximum number of restart attempts before giving up
    pub max_restart_attempts: u32,

    /// Time window for counting restart attempts (seconds)
    pub restart_window_secs: u64,

    /// Delay between restart attempts (milliseconds)
    pub restart_delay_ms: u64,

    /// Exponential backoff multiplier (1.0 = no backoff)
    pub backoff_multiplier: f64,

    /// Maximum backoff delay (milliseconds)
    pub max_backoff_ms: u64,

    /// Enable Docker-based restart
    pub enable_docker: bool,

    /// Enable systemd-based restart
    pub enable_systemd: bool,

    /// Custom restart commands per component
    pub custom_commands: HashMap<String, String>,
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            max_restart_attempts: 3,
            restart_window_secs: 300, // 5 minutes
            restart_delay_ms: 1000,   // 1 second
            backoff_multiplier: 2.0,
            max_backoff_ms: 30000, // 30 seconds
            enable_docker: true,
            enable_systemd: false,
            custom_commands: HashMap::new(),
        }
    }
}

/// Restart strategy for a component
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartStrategy {
    /// Use Docker container restart
    Docker,
    /// Use systemd service restart
    Systemd,
    /// Use custom command
    Custom,
    /// Manual restart required
    Manual,
}

/// Result of a restart attempt
#[derive(Debug, Clone)]
pub struct RestartResult {
    pub component: ComponentType,
    pub success: bool,
    pub strategy_used: RestartStrategy,
    pub attempts: u32,
    pub duration_ms: u64,
    pub error_message: Option<String>,
}

/// Component restart manager
pub struct RestartManager {
    config: RestartConfig,
    restart_history: HashMap<ComponentType, Vec<Instant>>,
}

impl RestartManager {
    /// Create a new restart manager
    pub fn new(config: RestartConfig) -> Self {
        Self {
            config,
            restart_history: HashMap::new(),
        }
    }

    /// Attempt to restart a component
    pub async fn restart_component(&mut self, component: ComponentType) -> Result<RestartResult> {
        let start_time = Instant::now();

        // Check if we've exceeded restart limits
        if !self.should_attempt_restart(component) {
            warn!("Component {:?} has exceeded restart limits", component);
            return Ok(RestartResult {
                component,
                success: false,
                strategy_used: RestartStrategy::Manual,
                attempts: self.get_recent_restart_count(component),
                duration_ms: start_time.elapsed().as_millis() as u64,
                error_message: Some("Exceeded restart attempt limit".to_string()),
            });
        }

        // Record this restart attempt
        self.record_restart_attempt(component);
        let attempt_number = self.get_recent_restart_count(component);

        info!(
            "Attempting to restart {:?} (attempt {}/{})",
            component, attempt_number, self.config.max_restart_attempts
        );

        // Determine restart strategy
        let strategy = self.determine_restart_strategy(component);

        // Apply backoff delay if not the first attempt
        if attempt_number > 1 {
            let delay = self.calculate_backoff_delay(attempt_number);
            debug!("Applying backoff delay of {}ms", delay.as_millis());
            tokio::time::sleep(delay).await;
        }

        // Execute restart
        let result = match strategy {
            RestartStrategy::Docker => self.restart_docker_container(component).await,
            RestartStrategy::Systemd => self.restart_systemd_service(component).await,
            RestartStrategy::Custom => self.restart_custom_command(component).await,
            RestartStrategy::Manual => {
                Err(anyhow::anyhow!("Manual restart required for {:?}", component).into())
            }
        };

        let success = result.is_ok();
        let error_message = result.err().map(|e| e.to_string());

        if success {
            info!(
                "Successfully restarted {:?} using {:?}",
                component, strategy
            );
        } else {
            error!(
                "Failed to restart {:?}: {}",
                component,
                error_message.as_ref().unwrap()
            );
        }

        Ok(RestartResult {
            component,
            success,
            strategy_used: strategy,
            attempts: attempt_number,
            duration_ms: start_time.elapsed().as_millis() as u64,
            error_message,
        })
    }

    /// Check if we should attempt restart based on limits
    fn should_attempt_restart(&self, component: ComponentType) -> bool {
        let recent_count = self.get_recent_restart_count(component);
        recent_count < self.config.max_restart_attempts
    }

    /// Get count of recent restart attempts within the time window
    fn get_recent_restart_count(&self, component: ComponentType) -> u32 {
        let window = Duration::from_secs(self.config.restart_window_secs);
        let cutoff = Instant::now() - window;

        self.restart_history
            .get(&component)
            .map(|history| history.iter().filter(|&&time| time > cutoff).count() as u32)
            .unwrap_or(0)
    }

    /// Record a restart attempt
    fn record_restart_attempt(&mut self, component: ComponentType) {
        self.restart_history
            .entry(component)
            .or_default()
            .push(Instant::now());
    }

    /// Determine the best restart strategy for a component
    fn determine_restart_strategy(&self, component: ComponentType) -> RestartStrategy {
        let component_name = self.component_to_name(component);

        // Check for custom command
        if self.config.custom_commands.contains_key(&component_name) {
            return RestartStrategy::Custom;
        }

        // Check Docker
        if self.config.enable_docker && self.is_docker_container(&component_name) {
            return RestartStrategy::Docker;
        }

        // Check systemd
        if self.config.enable_systemd && self.is_systemd_service(&component_name) {
            return RestartStrategy::Systemd;
        }

        // Default to manual
        RestartStrategy::Manual
    }

    /// Calculate exponential backoff delay
    fn calculate_backoff_delay(&self, attempt: u32) -> Duration {
        let base_delay = self.config.restart_delay_ms as f64;
        let multiplier = self.config.backoff_multiplier;
        let exponent = (attempt - 1) as f64;

        let delay_ms = base_delay * multiplier.powf(exponent);
        let capped_delay_ms = delay_ms.min(self.config.max_backoff_ms as f64);

        Duration::from_millis(capped_delay_ms as u64)
    }

    /// Restart a Docker container
    async fn restart_docker_container(&self, component: ComponentType) -> Result<()> {
        let container_name = self.component_to_name(component);

        debug!("Restarting Docker container: {}", container_name);

        let output = Command::new("docker")
            .args(["restart", &container_name])
            .output()
            .context("Failed to execute docker restart command")?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("Docker restart failed: {}", stderr).into())
        }
    }

    /// Restart a systemd service
    async fn restart_systemd_service(&self, component: ComponentType) -> Result<()> {
        let service_name = format!("{}.service", self.component_to_name(component));

        debug!("Restarting systemd service: {}", service_name);

        let output = Command::new("systemctl")
            .args(["restart", &service_name])
            .output()
            .context("Failed to execute systemctl restart command")?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("Systemd restart failed: {}", stderr).into())
        }
    }

    /// Execute custom restart command
    async fn restart_custom_command(&self, component: ComponentType) -> Result<()> {
        let component_name = self.component_to_name(component);
        let command = self
            .config
            .custom_commands
            .get(&component_name)
            .context("No custom command configured")?;

        debug!("Executing custom restart command: {}", command);

        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .context("Failed to execute custom restart command")?;

        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("Custom restart command failed: {}", stderr).into())
        }
    }

    /// Check if a container exists in Docker
    fn is_docker_container(&self, name: &str) -> bool {
        Command::new("docker")
            .args([
                "ps",
                "-a",
                "--filter",
                &format!("name={}", name),
                "--format",
                "{{.Names}}",
            ])
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.trim() == name
            })
            .unwrap_or(false)
    }

    /// Check if a systemd service exists
    fn is_systemd_service(&self, name: &str) -> bool {
        let service_name = format!("{}.service", name);
        Command::new("systemctl")
            .args(["list-unit-files", &service_name])
            .output()
            .map(|output| {
                let stdout = String::from_utf8_lossy(&output.stdout);
                stdout.contains(&service_name)
            })
            .unwrap_or(false)
    }

    /// Convert ComponentType to container/service name
    fn component_to_name(&self, component: ComponentType) -> String {
        match component {
            ComponentType::ForwardService => "janus-forward".to_string(),
            ComponentType::BackwardService => "janus-backward".to_string(),
            ComponentType::GatewayService => "janus-gateway".to_string(),
            ComponentType::Redis => "redis".to_string(),
            ComponentType::Qdrant => "qdrant".to_string(),
            ComponentType::SharedMemory => "janus-shm".to_string(),
            ComponentType::CNSService => "janus-cns".to_string(),
            ComponentType::GrpcChannel => "janus-grpc".to_string(),
            ComponentType::WebSocket => "janus-ws".to_string(),
            ComponentType::VisionModule => "janus-vision".to_string(),
            ComponentType::LogicModule => "janus-logic".to_string(),
            ComponentType::MemoryModule => "janus-memory".to_string(),
            ComponentType::ExecutionModule => "janus-execution".to_string(),
            ComponentType::JobQueue => "janus-queue".to_string(),
            ComponentType::MetricsExporter => "janus-metrics".to_string(),
            // Neuromorphic brain regions
            ComponentType::Cortex => "janus-cortex".to_string(),
            ComponentType::Hippocampus => "janus-hippocampus".to_string(),
            ComponentType::BasalGanglia => "janus-basal-ganglia".to_string(),
            ComponentType::Thalamus => "janus-thalamus".to_string(),
            ComponentType::Prefrontal => "janus-prefrontal".to_string(),
            ComponentType::Amygdala => "janus-amygdala".to_string(),
            ComponentType::Hypothalamus => "janus-hypothalamus".to_string(),
            ComponentType::Cerebellum => "janus-cerebellum".to_string(),
            ComponentType::VisualCortex => "janus-visual-cortex".to_string(),
            ComponentType::Integration => "janus-integration".to_string(),
        }
    }

    /// Clean up old restart history to prevent memory growth
    pub fn cleanup_old_history(&mut self) {
        let window = Duration::from_secs(self.config.restart_window_secs);
        let cutoff = Instant::now() - window;

        for history in self.restart_history.values_mut() {
            history.retain(|&time| time > cutoff);
        }
    }

    /// Get restart statistics for a component
    pub fn get_stats(&self, component: ComponentType) -> RestartStats {
        let recent_count = self.get_recent_restart_count(component);
        let total_count = self
            .restart_history
            .get(&component)
            .map(|h| h.len() as u32)
            .unwrap_or(0);

        RestartStats {
            component,
            recent_restarts: recent_count,
            total_restarts: total_count,
            can_restart: self.should_attempt_restart(component),
        }
    }
}

/// Restart statistics for a component
#[derive(Debug, Clone)]
pub struct RestartStats {
    pub component: ComponentType,
    pub recent_restarts: u32,
    pub total_restarts: u32,
    pub can_restart: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restart_config_default() {
        let config = RestartConfig::default();
        assert_eq!(config.max_restart_attempts, 3);
        assert_eq!(config.restart_window_secs, 300);
        assert!(config.enable_docker);
    }

    #[test]
    fn test_backoff_calculation() {
        let config = RestartConfig {
            restart_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_backoff_ms: 30000,
            ..Default::default()
        };

        let manager = RestartManager::new(config);

        // First attempt: 1000ms
        assert_eq!(manager.calculate_backoff_delay(1).as_millis(), 1000);

        // Second attempt: 2000ms (1000 * 2^1)
        assert_eq!(manager.calculate_backoff_delay(2).as_millis(), 2000);

        // Third attempt: 4000ms (1000 * 2^2)
        assert_eq!(manager.calculate_backoff_delay(3).as_millis(), 4000);

        // Should cap at max_backoff_ms
        assert!(manager.calculate_backoff_delay(10).as_millis() <= 30000);
    }

    #[test]
    fn test_component_to_name() {
        let manager = RestartManager::new(RestartConfig::default());

        assert_eq!(
            manager.component_to_name(ComponentType::ForwardService),
            "janus-forward"
        );
        assert_eq!(
            manager.component_to_name(ComponentType::BackwardService),
            "janus-backward"
        );
        assert_eq!(
            manager.component_to_name(ComponentType::GatewayService),
            "janus-gateway"
        );
    }

    #[test]
    fn test_restart_attempt_tracking() {
        let mut manager = RestartManager::new(RestartConfig::default());

        assert_eq!(
            manager.get_recent_restart_count(ComponentType::ForwardService),
            0
        );

        manager.record_restart_attempt(ComponentType::ForwardService);
        assert_eq!(
            manager.get_recent_restart_count(ComponentType::ForwardService),
            1
        );

        manager.record_restart_attempt(ComponentType::ForwardService);
        manager.record_restart_attempt(ComponentType::ForwardService);
        assert_eq!(
            manager.get_recent_restart_count(ComponentType::ForwardService),
            3
        );
    }

    #[test]
    fn test_should_attempt_restart() {
        let mut manager = RestartManager::new(RestartConfig {
            max_restart_attempts: 2,
            ..Default::default()
        });

        assert!(manager.should_attempt_restart(ComponentType::ForwardService));

        manager.record_restart_attempt(ComponentType::ForwardService);
        assert!(manager.should_attempt_restart(ComponentType::ForwardService));

        manager.record_restart_attempt(ComponentType::ForwardService);
        assert!(!manager.should_attempt_restart(ComponentType::ForwardService));
    }

    #[test]
    fn test_get_stats() {
        let mut manager = RestartManager::new(RestartConfig::default());

        manager.record_restart_attempt(ComponentType::ForwardService);
        manager.record_restart_attempt(ComponentType::ForwardService);

        let stats = manager.get_stats(ComponentType::ForwardService);
        assert_eq!(stats.recent_restarts, 2);
        assert_eq!(stats.total_restarts, 2);
        assert!(stats.can_restart);
    }
}
