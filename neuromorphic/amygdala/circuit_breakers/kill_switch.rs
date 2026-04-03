//! # KillSwitch Circuit Breaker
//!
//! The ultimate emergency stop - immediately halts all trading activity and
//! shuts down the system. This is the "nuclear option" for critical situations.
//!
//! ## Use Cases
//!
//! - Critical system malfunction detected
//! - Severe market anomaly (flash crash, circuit breaker)
//! - Security breach detected
//! - Catastrophic loss threshold exceeded
//! - Regulatory emergency
//! - Manual emergency intervention
//!
//! ## Safety Features
//!
//! - Requires confirmation for non-automated triggers
//! - Immediately cancels all open orders
//! - Closes all open positions (optional)
//! - Notifies all stakeholders
//! - Persists system state before shutdown
//! - Creates audit trail

use crate::common::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

use super::cancel_all::CancelAll;

/// Trait for closing all open positions during an emergency.
///
/// Implement this trait on your position manager / execution client
/// and pass it to [`KillSwitch::set_position_closer`] so that the
/// kill switch can close positions instead of no-oping.
#[async_trait::async_trait]
pub trait PositionCloser: Send + Sync {
    /// Close all open positions at market. Returns the number of
    /// positions successfully closed.
    async fn close_all_positions(&self) -> Result<usize>;
}

/// Trait for persisting system state during an emergency shutdown.
#[async_trait::async_trait]
pub trait StatePersister: Send + Sync {
    /// Persist current system state (positions, balances, etc.).
    async fn persist(&self) -> Result<()>;
}

/// Trait for sending emergency notifications.
#[async_trait::async_trait]
pub trait EmergencyNotifier: Send + Sync {
    /// Send an emergency notification. Returns the number of
    /// recipients notified.
    async fn notify(&self, reason: &KillReason) -> Result<usize>;
}

/// Kill switch activation reason
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KillReason {
    /// Manual intervention by operator
    Manual { operator: String, reason: String },

    /// Automated trigger due to loss threshold
    LossThreshold { current_loss: f64, threshold: f64 },

    /// System malfunction detected
    SystemMalfunction { component: String, error: String },

    /// Market anomaly detected
    MarketAnomaly { description: String },

    /// Security breach
    SecurityBreach { threat: String },

    /// Regulatory requirement
    Regulatory { requirement: String },

    /// Custom reason
    Custom { description: String },
}

/// Kill switch activation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchResult {
    /// When the kill switch was activated
    pub activated_at: DateTime<Utc>,

    /// Reason for activation
    pub reason: KillReason,

    /// Whether activation was successful
    pub success: bool,

    /// Orders cancelled
    pub orders_cancelled: usize,

    /// Positions closed
    pub positions_closed: usize,

    /// Time taken to complete shutdown (ms)
    pub shutdown_duration_ms: u64,

    /// Errors encountered during shutdown
    pub errors: Vec<String>,

    /// State persistence status
    pub state_persisted: bool,

    /// Notifications sent
    pub notifications_sent: usize,
}

/// Kill switch configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchConfig {
    /// Whether to close all positions on activation
    pub close_positions: bool,

    /// Maximum time to wait for position closure (seconds)
    pub position_close_timeout_secs: u64,

    /// Whether to persist state before shutdown
    pub persist_state: bool,

    /// Require confirmation for manual triggers
    pub require_confirmation: bool,

    /// Send notifications on activation
    pub send_notifications: bool,

    /// Emergency contacts to notify
    pub emergency_contacts: Vec<String>,

    /// Maximum shutdown time before force kill (seconds)
    pub max_shutdown_time_secs: u64,
}

impl Default for KillSwitchConfig {
    fn default() -> Self {
        Self {
            close_positions: true,
            position_close_timeout_secs: 60,
            persist_state: true,
            require_confirmation: true,
            send_notifications: true,
            emergency_contacts: Vec::new(),
            max_shutdown_time_secs: 120,
        }
    }
}

/// Kill switch state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KillSwitchState {
    /// Normal operation
    Armed,

    /// Pending confirmation
    PendingConfirmation,

    /// Currently executing shutdown
    Activating,

    /// Shutdown complete, system halted
    Activated,
}

/// Emergency kill switch.
///
/// The kill switch delegates order cancellation to a [`CancelAll`] circuit
/// breaker (if one has been registered via [`KillSwitch::set_cancel_all`]).
/// Position closing is delegated to an optional [`PositionCloser`]
/// implementation (register via [`KillSwitch::set_position_closer`]).
///
/// When no delegate is registered, the corresponding step logs a warning
/// and returns 0 (no orders cancelled / positions closed).
pub struct KillSwitch {
    config: KillSwitchConfig,
    state: Arc<RwLock<KillSwitchState>>,
    is_activated: Arc<AtomicBool>,
    activation_time: Arc<RwLock<Option<DateTime<Utc>>>>,
    activation_count: Arc<RwLock<u64>>,
    last_result: Arc<RwLock<Option<KillSwitchResult>>>,
    pending_reason: Arc<RwLock<Option<KillReason>>>,
    /// Optional CancelAll circuit breaker for order cancellation.
    cancel_all: Arc<RwLock<Option<Arc<CancelAll>>>>,
    /// Optional position closer for emergency position liquidation.
    position_closer: Arc<RwLock<Option<Arc<dyn PositionCloser>>>>,
    /// Optional state persister for saving system state on shutdown.
    state_persister: Arc<RwLock<Option<Arc<dyn StatePersister>>>>,
    /// Optional emergency notifier for alerting stakeholders.
    emergency_notifier: Arc<RwLock<Option<Arc<dyn EmergencyNotifier>>>>,
}

impl Default for KillSwitch {
    fn default() -> Self {
        Self::new(KillSwitchConfig::default())
    }
}

impl KillSwitch {
    /// Create a new kill switch
    pub fn new(config: KillSwitchConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(KillSwitchState::Armed)),
            is_activated: Arc::new(AtomicBool::new(false)),
            activation_time: Arc::new(RwLock::new(None)),
            activation_count: Arc::new(RwLock::new(0)),
            last_result: Arc::new(RwLock::new(None)),
            pending_reason: Arc::new(RwLock::new(None)),
            cancel_all: Arc::new(RwLock::new(None)),
            position_closer: Arc::new(RwLock::new(None)),
            state_persister: Arc::new(RwLock::new(None)),
            emergency_notifier: Arc::new(RwLock::new(None)),
        }
    }

    /// Register a [`CancelAll`] circuit breaker for order cancellation.
    ///
    /// When set, [`activate_internal`] will delegate order cancellation to
    /// this instance instead of no-oping.
    pub async fn set_cancel_all(&self, cancel_all: Arc<CancelAll>) {
        let mut guard = self.cancel_all.write().await;
        *guard = Some(cancel_all);
        info!("KillSwitch: CancelAll circuit breaker registered");
    }

    /// Register a [`PositionCloser`] for emergency position liquidation.
    pub async fn set_position_closer(&self, closer: Arc<dyn PositionCloser>) {
        let mut guard = self.position_closer.write().await;
        *guard = Some(closer);
        info!("KillSwitch: PositionCloser registered");
    }

    /// Register a [`StatePersister`] for saving system state on emergency shutdown.
    pub async fn set_state_persister(&self, persister: Arc<dyn StatePersister>) {
        let mut guard = self.state_persister.write().await;
        *guard = Some(persister);
        info!("KillSwitch: StatePersister registered");
    }

    /// Register an [`EmergencyNotifier`] for alerting stakeholders.
    pub async fn set_emergency_notifier(&self, notifier: Arc<dyn EmergencyNotifier>) {
        let mut guard = self.emergency_notifier.write().await;
        *guard = Some(notifier);
        info!("KillSwitch: EmergencyNotifier registered");
    }

    /// Request kill switch activation (may require confirmation)
    pub async fn request_activation(&self, reason: KillReason) -> Result<KillSwitchState> {
        if self.is_activated() {
            warn!("Kill switch already activated");
            return Ok(KillSwitchState::Activated);
        }

        info!("🚨 KILL SWITCH ACTIVATION REQUESTED: {:?}", reason);

        // Check if confirmation is required
        if self.config.require_confirmation && matches!(reason, KillReason::Manual { .. }) {
            let mut state = self.state.write().await;
            *state = KillSwitchState::PendingConfirmation;

            let mut pending = self.pending_reason.write().await;
            *pending = Some(reason);

            info!("⏳ Kill switch activation pending confirmation");
            Ok(KillSwitchState::PendingConfirmation)
        } else {
            // Activate immediately for automated triggers
            self.activate_internal(reason).await?;
            Ok(KillSwitchState::Activated)
        }
    }

    /// Confirm pending activation
    pub async fn confirm_activation(&self) -> Result<KillSwitchResult> {
        let current_state = *self.state.read().await;

        if current_state != KillSwitchState::PendingConfirmation {
            return Err(anyhow::anyhow!("No pending activation to confirm").into());
        }

        let reason = self
            .pending_reason
            .write()
            .await
            .take()
            .ok_or_else(|| anyhow::anyhow!("No pending reason found"))?;

        info!("✅ Kill switch activation CONFIRMED");
        self.activate_internal(reason).await
    }

    /// Cancel pending activation
    pub async fn cancel_pending(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if *state == KillSwitchState::PendingConfirmation {
            *state = KillSwitchState::Armed;

            let mut pending = self.pending_reason.write().await;
            *pending = None;

            info!("❌ Kill switch activation CANCELLED");
            Ok(())
        } else {
            Err(anyhow::anyhow!("No pending activation to cancel").into())
        }
    }

    /// Internal activation logic
    async fn activate_internal(&self, reason: KillReason) -> Result<KillSwitchResult> {
        let start = std::time::Instant::now();

        // Update state
        {
            let mut state = self.state.write().await;
            *state = KillSwitchState::Activating;
        }

        error!("🔴 KILL SWITCH ACTIVATED: {:?}", reason);

        // Set activation flag
        self.is_activated.store(true, Ordering::SeqCst);

        // Record activation time
        {
            let mut activation_time = self.activation_time.write().await;
            *activation_time = Some(Utc::now());

            let mut count = self.activation_count.write().await;
            *count += 1;
        }

        let mut result = KillSwitchResult {
            activated_at: Utc::now(),
            reason: reason.clone(),
            success: false,
            orders_cancelled: 0,
            positions_closed: 0,
            shutdown_duration_ms: 0,
            errors: Vec::new(),
            state_persisted: false,
            notifications_sent: 0,
        };

        // Step 1: Cancel all open orders
        info!("Step 1: Cancelling all open orders...");
        match self.cancel_all_orders().await {
            Ok(count) => {
                result.orders_cancelled = count;
                info!("✅ Cancelled {} orders", count);
            }
            Err(e) => {
                let error_msg = format!("Failed to cancel orders: {}", e);
                error!("{}", error_msg);
                result.errors.push(error_msg);
            }
        }

        // Step 2: Close all positions (if configured)
        if self.config.close_positions {
            info!("Step 2: Closing all positions...");
            match self.close_all_positions().await {
                Ok(count) => {
                    result.positions_closed = count;
                    info!("✅ Closed {} positions", count);
                }
                Err(e) => {
                    let error_msg = format!("Failed to close positions: {}", e);
                    error!("{}", error_msg);
                    result.errors.push(error_msg);
                }
            }
        }

        // Step 3: Persist system state
        if self.config.persist_state {
            info!("Step 3: Persisting system state...");
            match self.persist_state().await {
                Ok(_) => {
                    result.state_persisted = true;
                    info!("✅ State persisted");
                }
                Err(e) => {
                    let error_msg = format!("Failed to persist state: {}", e);
                    error!("{}", error_msg);
                    result.errors.push(error_msg);
                }
            }
        }

        // Step 4: Send notifications
        if self.config.send_notifications {
            info!("Step 4: Sending emergency notifications...");
            match self.send_notifications(&reason).await {
                Ok(count) => {
                    result.notifications_sent = count;
                    info!("✅ Sent {} notifications", count);
                }
                Err(e) => {
                    let error_msg = format!("Failed to send notifications: {}", e);
                    error!("{}", error_msg);
                    result.errors.push(error_msg);
                }
            }
        }

        // Complete
        result.shutdown_duration_ms = start.elapsed().as_millis() as u64;
        result.success = result.errors.is_empty();

        {
            let mut state = self.state.write().await;
            *state = KillSwitchState::Activated;
        }

        {
            let mut last_result = self.last_result.write().await;
            *last_result = Some(result.clone());
        }

        if result.success {
            error!(
                "🛑 SYSTEM SHUTDOWN COMPLETE ({}ms)",
                result.shutdown_duration_ms
            );
        } else {
            error!(
                "⚠️  SYSTEM SHUTDOWN COMPLETE WITH ERRORS ({}ms): {:?}",
                result.shutdown_duration_ms, result.errors
            );
        }

        Ok(result)
    }

    /// Cancel all open orders via the registered [`CancelAll`] circuit breaker.
    ///
    /// If no `CancelAll` has been registered, logs a warning and returns 0.
    async fn cancel_all_orders(&self) -> Result<usize> {
        let guard = self.cancel_all.read().await;
        if let Some(ref cancel_all) = *guard {
            info!("KillSwitch: delegating order cancellation to CancelAll circuit breaker");
            let result = cancel_all
                .trigger("KillSwitch emergency activation")
                .await?;
            if !result.errors.is_empty() {
                for err in &result.errors {
                    error!("CancelAll error: {}", err);
                }
            }
            info!(
                "CancelAll completed: {} cancelled, {} failed ({}ms)",
                result.cancelled_count, result.failed_count, result.duration_ms
            );
            Ok(result.cancelled_count)
        } else {
            warn!(
                "KillSwitch: no CancelAll circuit breaker registered — \
                 cannot cancel orders. Register one via set_cancel_all()."
            );
            Ok(0)
        }
    }

    /// Close all open positions via the registered [`PositionCloser`].
    ///
    /// If no `PositionCloser` has been registered, logs a warning and returns 0.
    async fn close_all_positions(&self) -> Result<usize> {
        let guard = self.position_closer.read().await;
        if let Some(ref closer) = *guard {
            info!("KillSwitch: delegating position closing to registered PositionCloser");
            let count = closer.close_all_positions().await?;
            info!("PositionCloser completed: {} positions closed", count);
            Ok(count)
        } else {
            warn!(
                "KillSwitch: no PositionCloser registered — \
                 cannot close positions. Register one via set_position_closer()."
            );
            Ok(0)
        }
    }

    /// Persist system state via the registered [`StatePersister`].
    ///
    /// If no `StatePersister` has been registered, logs a warning and succeeds.
    async fn persist_state(&self) -> Result<()> {
        let guard = self.state_persister.read().await;
        if let Some(ref persister) = *guard {
            info!("KillSwitch: persisting system state via registered StatePersister");
            persister.persist().await?;
            info!("StatePersister completed successfully");
        } else {
            warn!(
                "KillSwitch: no StatePersister registered — \
                 system state will NOT be persisted. Register one via set_state_persister()."
            );
        }
        Ok(())
    }

    /// Send emergency notifications via the registered [`EmergencyNotifier`].
    ///
    /// Falls back to logging if no notifier is registered.
    async fn send_notifications(&self, reason: &KillReason) -> Result<usize> {
        let guard = self.emergency_notifier.read().await;
        if let Some(ref notifier) = *guard {
            info!("KillSwitch: sending emergency notifications via registered EmergencyNotifier");
            let count = notifier.notify(reason).await?;
            info!("EmergencyNotifier completed: {} recipients notified", count);
            Ok(count)
        } else {
            // Fallback: log the notification intent
            warn!(
                "KillSwitch: no EmergencyNotifier registered — \
                 logging emergency notification only. Reason: {:?}",
                reason
            );
            info!(
                "📢 EMERGENCY NOTIFICATION (log-only): {:?} — \
                 {} contacts configured but no notifier wired",
                reason,
                self.config.emergency_contacts.len()
            );
            Ok(0)
        }
    }

    /// Check if kill switch is activated
    pub fn is_activated(&self) -> bool {
        self.is_activated.load(Ordering::SeqCst)
    }

    /// Get current state
    pub async fn state(&self) -> KillSwitchState {
        *self.state.read().await
    }

    /// Get last activation result
    pub async fn last_result(&self) -> Option<KillSwitchResult> {
        self.last_result.read().await.clone()
    }

    /// Get activation statistics
    pub async fn stats(&self) -> KillSwitchStats {
        let activation_count = *self.activation_count.read().await;
        let activation_time = *self.activation_time.read().await;
        let current_state = *self.state.read().await;

        KillSwitchStats {
            activation_count,
            last_activation: activation_time,
            current_state,
            is_activated: self.is_activated(),
        }
    }

    /// Reset kill switch (use with extreme caution!)
    pub async fn reset(&self) -> Result<()> {
        warn!("⚠️  RESETTING KILL SWITCH - USE WITH CAUTION");

        self.is_activated.store(false, Ordering::SeqCst);

        let mut state = self.state.write().await;
        *state = KillSwitchState::Armed;

        let mut activation_time = self.activation_time.write().await;
        *activation_time = None;

        info!("Kill switch reset to Armed state");
        Ok(())
    }
}

/// Kill switch statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KillSwitchStats {
    pub activation_count: u64,
    pub last_activation: Option<DateTime<Utc>>,
    pub current_state: KillSwitchState,
    pub is_activated: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kill_switch_creation() {
        let kill_switch = KillSwitch::default();
        assert!(!kill_switch.is_activated());
        assert_eq!(kill_switch.state().await, KillSwitchState::Armed);
    }

    #[tokio::test]
    async fn test_automatic_activation() {
        let config = KillSwitchConfig {
            require_confirmation: false,
            close_positions: false,
            persist_state: false,
            send_notifications: false,
            ..Default::default()
        };

        let kill_switch = KillSwitch::new(config);

        let reason = KillReason::LossThreshold {
            current_loss: 10000.0,
            threshold: 5000.0,
        };

        let state = kill_switch.request_activation(reason).await.unwrap();
        assert_eq!(state, KillSwitchState::Activated);
        assert!(kill_switch.is_activated());
    }

    #[tokio::test]
    async fn test_manual_activation_with_confirmation() {
        let config = KillSwitchConfig {
            require_confirmation: true,
            close_positions: false,
            persist_state: false,
            send_notifications: false,
            ..Default::default()
        };

        let kill_switch = KillSwitch::new(config);

        let reason = KillReason::Manual {
            operator: "admin".to_string(),
            reason: "Testing".to_string(),
        };

        // Request should go to pending
        let state = kill_switch.request_activation(reason).await.unwrap();
        assert_eq!(state, KillSwitchState::PendingConfirmation);
        assert!(!kill_switch.is_activated());

        // Confirm activation
        let _result = kill_switch.confirm_activation().await.unwrap();
        assert!(kill_switch.is_activated());
        assert_eq!(kill_switch.state().await, KillSwitchState::Activated);
    }

    #[tokio::test]
    async fn test_cancel_pending_activation() {
        let config = KillSwitchConfig {
            require_confirmation: true,
            ..Default::default()
        };

        let kill_switch = KillSwitch::new(config);

        let reason = KillReason::Manual {
            operator: "admin".to_string(),
            reason: "Testing".to_string(),
        };

        kill_switch.request_activation(reason).await.unwrap();
        assert_eq!(
            kill_switch.state().await,
            KillSwitchState::PendingConfirmation
        );

        // Cancel
        kill_switch.cancel_pending().await.unwrap();
        assert_eq!(kill_switch.state().await, KillSwitchState::Armed);
        assert!(!kill_switch.is_activated());
    }

    #[tokio::test]
    async fn test_activation_tracking() {
        let config = KillSwitchConfig {
            require_confirmation: false,
            close_positions: false,
            persist_state: false,
            send_notifications: false,
            ..Default::default()
        };

        let kill_switch = KillSwitch::new(config);

        let stats = kill_switch.stats().await;
        assert_eq!(stats.activation_count, 0);
        assert!(stats.last_activation.is_none());

        let reason = KillReason::SystemMalfunction {
            component: "test".to_string(),
            error: "test error".to_string(),
        };

        kill_switch.request_activation(reason).await.unwrap();

        let stats = kill_switch.stats().await;
        assert_eq!(stats.activation_count, 1);
        assert!(stats.last_activation.is_some());
    }

    #[tokio::test]
    async fn test_reset() {
        let config = KillSwitchConfig {
            require_confirmation: false,
            close_positions: false,
            persist_state: false,
            send_notifications: false,
            ..Default::default()
        };

        let kill_switch = KillSwitch::new(config);

        let reason = KillReason::Custom {
            description: "Test".to_string(),
        };

        kill_switch.request_activation(reason).await.unwrap();
        assert!(kill_switch.is_activated());

        kill_switch.reset().await.unwrap();
        assert!(!kill_switch.is_activated());
        assert_eq!(kill_switch.state().await, KillSwitchState::Armed);
    }

    #[tokio::test]
    async fn test_double_activation() {
        let config = KillSwitchConfig {
            require_confirmation: false,
            close_positions: false,
            persist_state: false,
            send_notifications: false,
            ..Default::default()
        };

        let kill_switch = KillSwitch::new(config);

        let reason = KillReason::Custom {
            description: "First".to_string(),
        };

        kill_switch.request_activation(reason).await.unwrap();
        assert!(kill_switch.is_activated());

        // Second activation should be ignored
        let reason2 = KillReason::Custom {
            description: "Second".to_string(),
        };

        let state = kill_switch.request_activation(reason2).await.unwrap();
        assert_eq!(state, KillSwitchState::Activated);
    }

    #[test]
    fn test_kill_reason_variants() {
        let manual = KillReason::Manual {
            operator: "admin".to_string(),
            reason: "test".to_string(),
        };
        assert!(matches!(manual, KillReason::Manual { .. }));

        let loss = KillReason::LossThreshold {
            current_loss: 100.0,
            threshold: 50.0,
        };
        assert!(matches!(loss, KillReason::LossThreshold { .. }));
    }
}
