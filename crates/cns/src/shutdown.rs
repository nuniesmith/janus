//! # Graceful Shutdown Module
//!
//! Implements coordinated graceful shutdown for all JANUS components.
//! Ensures proper cleanup order and state persistence during system shutdown.

use crate::Result;
use crate::signals::ComponentType;
use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, broadcast};
use tracing::{debug, error, info, warn};

/// Shutdown signal types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShutdownSignal {
    /// Graceful shutdown requested
    Graceful,
    /// Immediate shutdown (emergency)
    Immediate,
    /// Shutdown due to fatal error
    Fatal,
}

/// Shutdown phase during the shutdown sequence
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ShutdownPhase {
    /// Initial phase - stop accepting new requests
    StopAcceptingRequests = 0,
    /// Drain in-flight requests
    DrainRequests = 1,
    /// Persist critical state
    PersistState = 2,
    /// Close connections
    CloseConnections = 3,
    /// Shutdown background tasks
    ShutdownTasks = 4,
    /// Final cleanup
    FinalCleanup = 5,
    /// Shutdown complete
    Complete = 6,
}

/// Configuration for graceful shutdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownConfig {
    /// Maximum time to wait for graceful shutdown per phase (seconds)
    pub phase_timeout_secs: u64,

    /// Maximum total shutdown time before force kill (seconds)
    pub total_timeout_secs: u64,

    /// Shutdown order priority (lower number = shutdown first)
    pub component_priority: HashMap<String, u32>,

    /// Whether to persist state before shutdown
    pub persist_state: bool,

    /// Directory for state persistence
    pub state_dir: Option<String>,

    /// Broadcast shutdown signal to all components
    pub broadcast_signal: bool,
}

impl Default for ShutdownConfig {
    fn default() -> Self {
        let mut component_priority = HashMap::new();

        // Define shutdown order (reverse of startup)
        component_priority.insert("gateway".to_string(), 1); // Stop accepting requests first
        component_priority.insert("forward".to_string(), 2); // Forward processing
        component_priority.insert("backward".to_string(), 3); // Backward processing
        component_priority.insert("cns".to_string(), 4); // Monitoring
        component_priority.insert("shared_memory".to_string(), 5); // Shared resources
        component_priority.insert("redis".to_string(), 6); // Redis last
        component_priority.insert("qdrant".to_string(), 6); // Qdrant last

        Self {
            phase_timeout_secs: 30,
            total_timeout_secs: 180, // 3 minutes total
            component_priority,
            persist_state: true,
            state_dir: Some("/var/lib/janus/state".to_string()),
            broadcast_signal: true,
        }
    }
}

/// Shutdown coordinator
pub struct ShutdownCoordinator {
    config: ShutdownConfig,
    shutdown_flag: Arc<AtomicBool>,
    shutdown_tx: broadcast::Sender<ShutdownSignal>,
    current_phase: Arc<RwLock<ShutdownPhase>>,
    component_states: Arc<RwLock<HashMap<ComponentType, ComponentShutdownState>>>,
}

/// State of a component during shutdown
#[derive(Debug, Clone)]
pub struct ComponentShutdownState {
    pub component: ComponentType,
    pub phase: ShutdownPhase,
    pub started_at: Option<Instant>,
    pub completed_at: Option<Instant>,
    pub error: Option<String>,
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    pub fn new(config: ShutdownConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(100);

        Self {
            config,
            shutdown_flag: Arc::new(AtomicBool::new(false)),
            shutdown_tx,
            current_phase: Arc::new(RwLock::new(ShutdownPhase::StopAcceptingRequests)),
            component_states: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get a shutdown receiver for listening to shutdown signals
    pub fn subscribe(&self) -> broadcast::Receiver<ShutdownSignal> {
        self.shutdown_tx.subscribe()
    }

    /// Check if shutdown has been initiated
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown_flag.load(Ordering::Relaxed)
    }

    /// Initiate graceful shutdown
    pub async fn initiate_shutdown(&self, signal: ShutdownSignal) -> Result<ShutdownReport> {
        if self.shutdown_flag.swap(true, Ordering::Relaxed) {
            warn!("Shutdown already in progress");
            return Err(anyhow::anyhow!("Shutdown already in progress").into());
        }

        info!("Initiating {:?} shutdown", signal);
        let start_time = Instant::now();

        // Broadcast shutdown signal
        if self.config.broadcast_signal
            && let Err(e) = self.shutdown_tx.send(signal)
        {
            warn!("Failed to broadcast shutdown signal: {}", e);
        }

        // Execute shutdown sequence
        let report = match signal {
            ShutdownSignal::Graceful => self.graceful_shutdown().await,
            ShutdownSignal::Immediate => self.immediate_shutdown().await,
            ShutdownSignal::Fatal => self.fatal_shutdown().await,
        };

        info!(
            "Shutdown completed in {}ms",
            start_time.elapsed().as_millis()
        );

        report
    }

    /// Execute graceful shutdown sequence
    async fn graceful_shutdown(&self) -> Result<ShutdownReport> {
        let start_time = Instant::now();
        let mut report = ShutdownReport::new(ShutdownSignal::Graceful);

        // Phase 1: Stop accepting new requests
        if let Err(e) = self
            .execute_phase(ShutdownPhase::StopAcceptingRequests, || async {
                info!("Stopping request acceptance...");
                self.broadcast_phase(ShutdownPhase::StopAcceptingRequests)
                    .await;
                tokio::time::sleep(Duration::from_millis(500)).await;
                Ok(())
            })
            .await
        {
            report.add_error("StopAcceptingRequests", e);
        }

        // Phase 2: Drain in-flight requests
        if let Err(e) = self
            .execute_phase(ShutdownPhase::DrainRequests, || async {
                info!("Draining in-flight requests...");
                self.wait_for_drain().await?;
                Ok(())
            })
            .await
        {
            report.add_error("DrainRequests", e);
        }

        // Phase 3: Persist state
        if self.config.persist_state
            && let Err(e) = self
                .execute_phase(ShutdownPhase::PersistState, || async {
                    info!("Persisting system state...");
                    self.persist_system_state().await?;
                    Ok(())
                })
                .await
        {
            report.add_error("PersistState", e);
        }

        // Phase 4: Close connections
        if let Err(e) = self
            .execute_phase(ShutdownPhase::CloseConnections, || async {
                info!("Closing connections...");
                self.close_all_connections().await?;
                Ok(())
            })
            .await
        {
            report.add_error("CloseConnections", e);
        }

        // Phase 5: Shutdown background tasks
        if let Err(e) = self
            .execute_phase(ShutdownPhase::ShutdownTasks, || async {
                info!("Shutting down background tasks...");
                self.shutdown_background_tasks().await?;
                Ok(())
            })
            .await
        {
            report.add_error("ShutdownTasks", e);
        }

        // Phase 6: Final cleanup
        if let Err(e) = self
            .execute_phase(ShutdownPhase::FinalCleanup, || async {
                info!("Performing final cleanup...");
                self.final_cleanup().await?;
                Ok(())
            })
            .await
        {
            report.add_error("FinalCleanup", e);
        }

        // Mark as complete
        *self.current_phase.write().await = ShutdownPhase::Complete;
        report.duration_ms = start_time.elapsed().as_millis() as u64;
        report.success = report.errors.is_empty();

        Ok(report)
    }

    /// Execute immediate shutdown (skip some phases)
    async fn immediate_shutdown(&self) -> Result<ShutdownReport> {
        let start_time = Instant::now();
        let mut report = ShutdownReport::new(ShutdownSignal::Immediate);

        info!("Executing immediate shutdown");

        // Only persist state and close connections
        if let Err(e) = self.persist_system_state().await {
            report.add_error("PersistState", e);
        }

        if let Err(e) = self.close_all_connections().await {
            report.add_error("CloseConnections", e);
        }

        report.duration_ms = start_time.elapsed().as_millis() as u64;
        report.success = report.errors.is_empty();

        Ok(report)
    }

    /// Execute fatal shutdown (minimal cleanup)
    async fn fatal_shutdown(&self) -> Result<ShutdownReport> {
        let start_time = Instant::now();
        let mut report = ShutdownReport::new(ShutdownSignal::Fatal);

        error!("Executing fatal shutdown");

        // Only try to persist state
        if let Err(e) = self.persist_system_state().await {
            report.add_error("PersistState", e);
        }

        report.duration_ms = start_time.elapsed().as_millis() as u64;
        report.success = false; // Fatal shutdown is never "successful"

        Ok(report)
    }

    /// Execute a shutdown phase with timeout
    async fn execute_phase<F, Fut>(&self, phase: ShutdownPhase, f: F) -> Result<()>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        debug!("Entering shutdown phase: {:?}", phase);
        *self.current_phase.write().await = phase;

        let timeout = Duration::from_secs(self.config.phase_timeout_secs);

        match tokio::time::timeout(timeout, f()).await {
            Ok(Ok(())) => {
                debug!("Shutdown phase {:?} completed successfully", phase);
                Ok(())
            }
            Ok(Err(e)) => {
                warn!("Shutdown phase {:?} failed: {}", phase, e);
                Err(e)
            }
            Err(_) => {
                warn!("Shutdown phase {:?} timed out", phase);
                Err(anyhow::anyhow!("Phase {:?} timed out", phase).into())
            }
        }
    }

    /// Broadcast current phase to all components
    async fn broadcast_phase(&self, phase: ShutdownPhase) {
        debug!("Broadcasting shutdown phase: {:?}", phase);
        // In a real implementation, this would notify all components
        // For now, just update internal state
    }

    /// Wait for in-flight requests to drain
    async fn wait_for_drain(&self) -> Result<()> {
        let max_wait = Duration::from_secs(30);
        let check_interval = Duration::from_millis(100);
        let start = Instant::now();

        while start.elapsed() < max_wait {
            // In a real implementation, check actual request counters
            // For now, just wait a bit
            tokio::time::sleep(check_interval).await;

            // Simulate checking if requests are drained
            if self.are_requests_drained().await {
                return Ok(());
            }
        }

        warn!("Request drain timed out");
        Ok(()) // Don't fail, just continue
    }

    /// Check if all requests are drained (stub)
    async fn are_requests_drained(&self) -> bool {
        // In a real implementation, check metrics or counters
        true
    }

    /// Persist system state to disk
    async fn persist_system_state(&self) -> Result<()> {
        if let Some(ref state_dir) = self.config.state_dir {
            debug!("Persisting state to: {}", state_dir);

            // Ensure state directory exists
            std::fs::create_dir_all(state_dir).context("Failed to create state directory")?;

            // Persist component states (convert to serializable format)
            let states = self.component_states.read().await;
            let state_file = format!("{}/shutdown_state.json", state_dir);

            // Convert to a serializable format (phase and error info only)
            let serializable_states: HashMap<String, (ShutdownPhase, Option<String>)> = states
                .iter()
                .map(|(k, v)| (format!("{:?}", k), (v.phase, v.error.clone())))
                .collect();

            let state_json = serde_json::to_string_pretty(&serializable_states)
                .context("Failed to serialize state")?;

            std::fs::write(&state_file, state_json).context("Failed to write state file")?;

            info!("System state persisted to: {}", state_file);
        }

        Ok(())
    }

    /// Close all connections
    async fn close_all_connections(&self) -> Result<()> {
        debug!("Closing all connections");
        // In a real implementation:
        // - Close Redis connections
        // - Close gRPC channels
        // - Close HTTP connections
        // - Close shared memory
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    /// Shutdown background tasks
    async fn shutdown_background_tasks(&self) -> Result<()> {
        debug!("Shutting down background tasks");
        // In a real implementation:
        // - Cancel tokio tasks
        // - Wait for task completion
        // - Clean up task resources
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    /// Final cleanup
    async fn final_cleanup(&self) -> Result<()> {
        debug!("Performing final cleanup");
        // In a real implementation:
        // - Remove PID files
        // - Clean up temporary files
        // - Release file locks
        tokio::time::sleep(Duration::from_millis(100)).await;
        Ok(())
    }

    /// Get current shutdown phase
    pub async fn current_phase(&self) -> ShutdownPhase {
        *self.current_phase.read().await
    }

    /// Update component shutdown state
    pub async fn update_component_state(
        &self,
        component: ComponentType,
        phase: ShutdownPhase,
        error: Option<String>,
    ) {
        let mut states = self.component_states.write().await;

        states
            .entry(component)
            .and_modify(|state| {
                state.phase = phase;
                if phase == ShutdownPhase::Complete {
                    state.completed_at = Some(Instant::now());
                }
                if let Some(err) = error.clone() {
                    state.error = Some(err);
                }
            })
            .or_insert_with(|| ComponentShutdownState {
                component,
                phase,
                started_at: Some(Instant::now()),
                completed_at: if phase == ShutdownPhase::Complete {
                    Some(Instant::now())
                } else {
                    None
                },
                error,
            });
    }

    /// Get shutdown report for all components
    pub async fn get_component_states(&self) -> HashMap<ComponentType, ComponentShutdownState> {
        self.component_states.read().await.clone()
    }
}

/// Report of shutdown execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownReport {
    pub signal: ShutdownSignal,
    pub success: bool,
    pub duration_ms: u64,
    pub errors: Vec<ShutdownError>,
}

impl ShutdownReport {
    fn new(signal: ShutdownSignal) -> Self {
        Self {
            signal,
            success: true,
            duration_ms: 0,
            errors: Vec::new(),
        }
    }

    fn add_error(&mut self, phase: &str, error: crate::CNSError) {
        self.errors.push(ShutdownError {
            phase: phase.to_string(),
            message: error.to_string(),
        });
        self.success = false;
    }
}

/// Error during shutdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShutdownError {
    pub phase: String,
    pub message: String,
}

// ============================================================================
// Signal Handler Integration
// ============================================================================

/// Setup signal handlers for graceful shutdown
pub fn setup_signal_handlers(coordinator: Arc<ShutdownCoordinator>) {
    tokio::spawn(async move {
        use tokio::signal;

        #[cfg(unix)]
        {
            let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
                .expect("Failed to setup SIGTERM handler");
            let mut sigint = signal::unix::signal(signal::unix::SignalKind::interrupt())
                .expect("Failed to setup SIGINT handler");

            tokio::select! {
                _ = sigterm.recv() => {
                    info!("Received SIGTERM, initiating graceful shutdown");
                    let _ = coordinator.initiate_shutdown(ShutdownSignal::Graceful).await;
                }
                _ = sigint.recv() => {
                    info!("Received SIGINT (Ctrl+C), initiating graceful shutdown");
                    let _ = coordinator.initiate_shutdown(ShutdownSignal::Graceful).await;
                }
            }
        }

        #[cfg(not(unix))]
        {
            signal::ctrl_c().await.expect("Failed to listen for Ctrl+C");
            info!("Received Ctrl+C, initiating graceful shutdown");
            let _ = coordinator
                .initiate_shutdown(ShutdownSignal::Graceful)
                .await;
        }
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shutdown_config_default() {
        let config = ShutdownConfig::default();
        assert_eq!(config.phase_timeout_secs, 30);
        assert_eq!(config.total_timeout_secs, 180);
        assert!(config.persist_state);
        assert!(config.broadcast_signal);
    }

    #[tokio::test]
    async fn test_shutdown_coordinator_creation() {
        let config = ShutdownConfig::default();
        let coordinator = ShutdownCoordinator::new(config);
        assert!(!coordinator.is_shutting_down());
    }

    #[tokio::test]
    async fn test_shutdown_flag() {
        let coordinator = ShutdownCoordinator::new(ShutdownConfig::default());
        assert!(!coordinator.is_shutting_down());

        coordinator.shutdown_flag.store(true, Ordering::Relaxed);
        assert!(coordinator.is_shutting_down());
    }

    #[tokio::test]
    async fn test_subscribe_to_shutdown() {
        let coordinator = ShutdownCoordinator::new(ShutdownConfig::default());
        let mut rx = coordinator.subscribe();

        coordinator
            .shutdown_tx
            .send(ShutdownSignal::Graceful)
            .unwrap();

        let signal = rx.recv().await.unwrap();
        assert_eq!(signal, ShutdownSignal::Graceful);
    }

    #[tokio::test]
    async fn test_current_phase() {
        let coordinator = ShutdownCoordinator::new(ShutdownConfig::default());
        let phase = coordinator.current_phase().await;
        assert_eq!(phase, ShutdownPhase::StopAcceptingRequests);
    }

    #[tokio::test]
    async fn test_update_component_state() {
        let coordinator = ShutdownCoordinator::new(ShutdownConfig::default());

        coordinator
            .update_component_state(
                ComponentType::ForwardService,
                ShutdownPhase::DrainRequests,
                None,
            )
            .await;

        let states = coordinator.get_component_states().await;
        assert!(states.contains_key(&ComponentType::ForwardService));

        let state = &states[&ComponentType::ForwardService];
        assert_eq!(state.phase, ShutdownPhase::DrainRequests);
        assert!(state.error.is_none());
    }

    #[test]
    fn test_shutdown_phases_ordering() {
        assert!(ShutdownPhase::StopAcceptingRequests < ShutdownPhase::DrainRequests);
        assert!(ShutdownPhase::DrainRequests < ShutdownPhase::PersistState);
        assert!(ShutdownPhase::FinalCleanup < ShutdownPhase::Complete);
    }

    #[test]
    fn test_shutdown_report_creation() {
        let report = ShutdownReport::new(ShutdownSignal::Graceful);
        assert!(report.success);
        assert_eq!(report.duration_ms, 0);
        assert!(report.errors.is_empty());
    }
}
