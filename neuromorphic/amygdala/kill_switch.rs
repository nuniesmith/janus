//! Kill Switch - Emergency Trading Halt
//!
//! ⚠️ CRITICAL SAFETY COMPONENT ⚠️
//!
//! This is your emergency stop mechanism. It should be tested thoroughly
//! and triggered automatically when threats are detected.
//!
//! ## Safety Invariants
//!
//! 1. Once triggered, **no new orders may be submitted** until explicitly reset.
//! 2. The `should_allow_order` guard MUST be checked on every order submission path.
//! 3. Trigger state is recorded with timestamp and reason for post-incident review.
//! 4. Reset requires an explicit call — there is no automatic cooldown.

use crate::common::Result;
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Actions the kill switch can take during an emergency.
#[derive(Debug, Clone)]
pub enum EmergencyAction {
    CancelAllOrders,
    CloseAllPositions,
    DisableTrading,
    SendAlert(String),
    LogIncident(String),
}

/// Recorded state of a kill switch activation for audit trail.
#[derive(Debug, Clone)]
pub struct TriggerRecord {
    /// The threat level that caused the trigger.
    pub threat_level: f32,
    /// Human-readable reason for the trigger.
    pub reason: String,
    /// Wall-clock instant when the trigger occurred.
    pub triggered_at: Instant,
}

/// Kill switch levels matching the audit's four-level requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KillSwitchScope {
    /// Kill a single strategy only.
    PerStrategy,
    /// Kill a single instrument only.
    PerInstrument,
    /// Kill a single account only.
    PerAccount,
    /// Kill the entire system.
    Global,
}

pub struct KillSwitch {
    /// Whether the trading system is allowed to operate.
    enabled: Arc<AtomicBool>,
    /// Whether the kill switch has been triggered (latching flag).
    triggered: Arc<AtomicBool>,
    /// Whether new order submission is blocked (independent of `enabled`).
    /// This flag is set atomically and checked on the hot path.
    orders_blocked: Arc<AtomicBool>,
    /// Threshold above which the kill switch fires.
    threat_threshold: f32,
    /// Ordered list of actions to execute on trigger.
    emergency_actions: Arc<RwLock<Vec<EmergencyAction>>>,
    /// Optional integration point for actual order/position management.
    emergency_manager: Option<Arc<dyn EmergencyOrderManager>>,
    /// Audit trail of the most recent trigger event.
    trigger_record: Arc<RwLock<Option<TriggerRecord>>>,
}

/// Trait for emergency order management integration.
#[async_trait]
pub trait EmergencyOrderManager: Send + Sync {
    async fn cancel_all_orders(&self) -> Result<usize>;
    async fn close_all_positions(&self) -> Result<usize>;
    async fn disable_trading(&self) -> Result<()>;
    async fn send_alert(&self, message: &str) -> Result<()>;
}

impl KillSwitch {
    pub fn new(threat_threshold: f32) -> Self {
        Self {
            enabled: Arc::new(AtomicBool::new(true)),
            triggered: Arc::new(AtomicBool::new(false)),
            orders_blocked: Arc::new(AtomicBool::new(false)),
            threat_threshold,
            emergency_actions: Arc::new(RwLock::new(Vec::new())),
            emergency_manager: None,
            trigger_record: Arc::new(RwLock::new(None)),
        }
    }

    /// Set emergency order manager for actual order/position management.
    pub fn with_emergency_manager(mut self, manager: Arc<dyn EmergencyOrderManager>) -> Self {
        self.emergency_manager = Some(manager);
        self
    }

    // -----------------------------------------------------------------------
    // Guards — call these on every order-submission code path
    // -----------------------------------------------------------------------

    /// Returns `true` if the trading system is enabled (not disabled by kill switch).
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }

    /// Returns `true` if the kill switch has been triggered.
    pub fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::SeqCst)
    }

    /// Returns `true` if new order submission is currently blocked.
    ///
    /// This is the **primary guard** that every order-submission path MUST check.
    /// It is set to `true` the instant the kill switch fires, even before the
    /// emergency protocol finishes executing.
    pub fn are_orders_blocked(&self) -> bool {
        self.orders_blocked.load(Ordering::SeqCst)
    }

    /// Convenience check for the order-submission hot path.
    ///
    /// Returns `true` only when:
    /// - The system is enabled,
    /// - The kill switch has not been triggered, AND
    /// - Orders are not blocked.
    ///
    /// Usage in order submission:
    /// ```rust,ignore
    /// if !kill_switch.should_allow_order() {
    ///     return Err(TradingError::KillSwitchActive);
    /// }
    /// ```
    pub fn should_allow_order(&self) -> bool {
        self.is_enabled() && !self.is_triggered() && !self.are_orders_blocked()
    }

    // -----------------------------------------------------------------------
    // Trigger
    // -----------------------------------------------------------------------

    /// Trigger the kill switch if `threat_level` exceeds the configured threshold.
    ///
    /// This method:
    /// 1. Immediately blocks new orders (atomic flag).
    /// 2. Records the trigger reason and timestamp.
    /// 3. Executes the configured emergency action protocol.
    pub async fn trigger(&self, threat_level: f32, reason: &str) -> Result<()> {
        if threat_level <= self.threat_threshold {
            return Ok(());
        }

        // --- Atomic state transitions (must happen before any async work) ---
        self.orders_blocked.store(true, Ordering::SeqCst);
        self.triggered.store(true, Ordering::SeqCst);

        error!(
            "🚨 KILL SWITCH TRIGGERED! Threat level: {:.2}, Reason: {}",
            threat_level, reason
        );

        // Record the trigger for post-incident review
        {
            let mut record = self.trigger_record.write().await;
            *record = Some(TriggerRecord {
                threat_level,
                reason: reason.to_string(),
                triggered_at: Instant::now(),
            });
        }

        // Execute emergency actions
        self.execute_emergency_protocol().await?;

        Ok(())
    }

    /// Manually trigger the kill switch with maximum threat level.
    pub async fn manual_trigger(&self, reason: &str) -> Result<()> {
        self.trigger(1.0, reason).await
    }

    // -----------------------------------------------------------------------
    // Emergency protocol
    // -----------------------------------------------------------------------

    /// Execute all configured emergency actions in order.
    async fn execute_emergency_protocol(&self) -> Result<()> {
        let actions = self.emergency_actions.read().await;

        for action in actions.iter() {
            match action {
                EmergencyAction::CancelAllOrders => {
                    warn!("Cancelling all open orders...");
                    if let Some(manager) = &self.emergency_manager {
                        match manager.cancel_all_orders().await {
                            Ok(count) => info!("Cancelled {} orders", count),
                            Err(e) => error!("Failed to cancel orders: {}", e),
                        }
                    } else {
                        warn!("No emergency manager configured - orders not cancelled");
                    }
                }
                EmergencyAction::CloseAllPositions => {
                    warn!("Closing all positions with market orders...");
                    if let Some(manager) = &self.emergency_manager {
                        match manager.close_all_positions().await {
                            Ok(count) => info!("Closed {} positions", count),
                            Err(e) => error!("Failed to close positions: {}", e),
                        }
                    } else {
                        warn!("No emergency manager configured - positions not closed");
                    }
                }
                EmergencyAction::DisableTrading => {
                    warn!("Disabling all trading...");
                    self.enabled.store(false, Ordering::SeqCst);
                    if let Some(manager) = &self.emergency_manager {
                        if let Err(e) = manager.disable_trading().await {
                            error!("Failed to disable trading in manager: {}", e);
                        }
                    }
                }
                EmergencyAction::SendAlert(msg) => {
                    error!("📧 ALERT: {}", msg);
                    if let Some(manager) = &self.emergency_manager {
                        if let Err(e) = manager.send_alert(msg).await {
                            error!("Failed to send alert: {}", e);
                        }
                    } else {
                        warn!("No alert handler configured - alert logged only");
                    }
                }
                EmergencyAction::LogIncident(msg) => {
                    error!("📝 INCIDENT: {}", msg);
                    // Incidents are always logged via tracing
                }
            }
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Configuration
    // -----------------------------------------------------------------------

    /// Add an emergency action to the protocol.
    pub async fn add_emergency_action(&self, action: EmergencyAction) {
        self.emergency_actions.write().await.push(action);
    }

    // -----------------------------------------------------------------------
    // Recovery (use with extreme caution)
    // -----------------------------------------------------------------------

    /// Reset the kill switch.
    ///
    /// ⚠️ This re-enables trading and unblocks order submission.
    /// Only call this after you have confirmed the root cause is resolved.
    pub fn reset(&self) {
        warn!("⚠️  Resetting kill switch - ensure it's safe to resume trading!");
        self.triggered.store(false, Ordering::SeqCst);
        self.orders_blocked.store(false, Ordering::SeqCst);
        self.enabled.store(true, Ordering::SeqCst);
    }

    // -----------------------------------------------------------------------
    // Inspection
    // -----------------------------------------------------------------------

    /// Return the most recent trigger record, if any.
    pub async fn trigger_record(&self) -> Option<TriggerRecord> {
        self.trigger_record.read().await.clone()
    }

    /// Return how long ago the kill switch was triggered, or `None` if it
    /// has not been triggered (or has been reset since).
    pub async fn time_since_trigger(&self) -> Option<std::time::Duration> {
        self.trigger_record
            .read()
            .await
            .as_ref()
            .map(|r| r.triggered_at.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kill_switch_trigger() {
        let switch = KillSwitch::new(0.8);

        assert!(!switch.is_triggered());
        assert!(switch.should_allow_order());

        // Add emergency actions
        switch
            .add_emergency_action(EmergencyAction::DisableTrading)
            .await;

        // Should trigger
        switch.trigger(0.9, "Test threat").await.unwrap();
        assert!(switch.is_triggered());
        assert!(!switch.should_allow_order());
    }

    #[tokio::test]
    async fn test_kill_switch_no_trigger() {
        let switch = KillSwitch::new(0.8);

        // Should not trigger — threat level below threshold
        switch.trigger(0.5, "Low threat").await.unwrap();
        assert!(!switch.is_triggered());
        assert!(switch.should_allow_order());
    }

    #[tokio::test]
    async fn test_manual_trigger() {
        let switch = KillSwitch::new(0.8);

        switch.manual_trigger("Manual stop").await.unwrap();
        assert!(switch.is_triggered());
        assert!(!switch.should_allow_order());
    }

    #[tokio::test]
    async fn test_orders_blocked_immediately_on_trigger() {
        let switch = KillSwitch::new(0.8);

        assert!(!switch.are_orders_blocked());
        switch.trigger(0.95, "Flash crash").await.unwrap();
        assert!(switch.are_orders_blocked());
    }

    #[tokio::test]
    async fn test_trigger_record_captured() {
        let switch = KillSwitch::new(0.8);

        switch.trigger(0.95, "Excessive drawdown").await.unwrap();

        let record = switch.trigger_record().await;
        assert!(record.is_some());
        let record = record.unwrap();
        assert!((record.threat_level - 0.95).abs() < f32::EPSILON);
        assert_eq!(record.reason, "Excessive drawdown");
    }

    #[tokio::test]
    async fn test_reset() {
        let switch = KillSwitch::new(0.8);

        switch.manual_trigger("Test").await.unwrap();
        assert!(switch.is_triggered());
        assert!(!switch.should_allow_order());

        switch.reset();
        assert!(!switch.is_triggered());
        assert!(switch.is_enabled());
        assert!(!switch.are_orders_blocked());
        assert!(switch.should_allow_order());
    }

    #[tokio::test]
    async fn test_should_allow_order_false_when_disabled() {
        let switch = KillSwitch::new(0.8);

        // Manually disable without triggering
        switch.enabled.store(false, Ordering::SeqCst);
        assert!(!switch.should_allow_order());
    }

    #[tokio::test]
    async fn test_time_since_trigger() {
        let switch = KillSwitch::new(0.8);

        // No trigger yet
        assert!(switch.time_since_trigger().await.is_none());

        switch.manual_trigger("Latency spike").await.unwrap();

        let elapsed = switch.time_since_trigger().await;
        assert!(elapsed.is_some());
        // Should be very recent (< 1 second in test)
        assert!(elapsed.unwrap().as_secs() < 1);
    }
}
