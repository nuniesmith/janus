//! # PositionFreeze Circuit Breaker
//!
//! Prevents any modifications to existing positions during periods of high
//! volatility or uncertainty. Positions are locked in their current state.
//!
//! ## Use Cases
//!
//! - High market volatility detected
//! - Uncertain market conditions
//! - News event pending
//! - System performing self-diagnostics
//! - Temporary risk management hold
//! - Pre-market close position review
//!
//! ## Features
//!
//! - Prevents position increases
//! - Prevents position decreases (optional)
//! - Allows emergency exits (optional)
//! - Time-based auto-unfreeze
//! - Per-symbol or global freeze
//! - Audit trail of freeze events

use crate::common::Result;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

/// Freeze level determines what operations are blocked
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreezeLevel {
    /// Allow emergency exits only (sell to close)
    EmergencyExitOnly,

    /// Prevent all position increases
    NoIncrease,

    /// Prevent all position changes (complete freeze)
    Complete,

    /// Allow reduction only (no increases)
    ReductionOnly,
}

/// Freeze scope - what gets frozen
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreezeScope {
    /// Freeze all positions globally
    Global,

    /// Freeze specific symbols only
    Symbols(HashSet<String>),

    /// Freeze specific exchanges only
    Exchanges(HashSet<String>),
}

/// Reason for position freeze
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FreezeReason {
    /// High volatility detected
    HighVolatility { vix_level: f64 },

    /// News event pending
    NewsEvent { event: String, time: DateTime<Utc> },

    /// Manual intervention
    Manual { operator: String, reason: String },

    /// System diagnostics in progress
    SystemDiagnostics,

    /// Risk threshold breached
    RiskThreshold { metric: String, value: f64 },

    /// Market hours transition (pre-close, pre-open)
    MarketTransition { transition: String },

    /// Custom reason
    Custom { description: String },
}

/// Active freeze record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveFreeze {
    /// Unique freeze ID
    pub id: String,

    /// When the freeze was activated
    pub activated_at: DateTime<Utc>,

    /// Freeze level
    pub level: FreezeLevel,

    /// What is frozen
    pub scope: FreezeScope,

    /// Reason for freeze
    pub reason: FreezeReason,

    /// When to auto-unfreeze (if set)
    pub auto_unfreeze_at: Option<DateTime<Utc>>,

    /// Number of blocked operations
    pub blocked_operations: u64,
}

/// Result of a freeze operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreezeResult {
    pub freeze_id: String,
    pub activated_at: DateTime<Utc>,
    pub scope: FreezeScope,
    pub level: FreezeLevel,
}

/// Configuration for position freeze
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionFreezeConfig {
    /// Default freeze level
    pub default_level: FreezeLevel,

    /// Default freeze duration (seconds, None = manual unfreeze)
    pub default_duration_secs: Option<u64>,

    /// Allow emergency exits during complete freeze
    pub allow_emergency_exits: bool,

    /// Send alerts on freeze activation
    pub send_alerts: bool,

    /// Maximum concurrent freezes
    pub max_concurrent_freezes: usize,
}

impl Default for PositionFreezeConfig {
    fn default() -> Self {
        Self {
            default_level: FreezeLevel::NoIncrease,
            default_duration_secs: Some(300), // 5 minutes
            allow_emergency_exits: true,
            send_alerts: true,
            max_concurrent_freezes: 10,
        }
    }
}

/// Position freeze circuit breaker
pub struct PositionFreeze {
    config: PositionFreezeConfig,
    active_freezes: Arc<RwLock<HashMap<String, ActiveFreeze>>>,
    freeze_history: Arc<RwLock<Vec<ActiveFreeze>>>,
    total_freezes: Arc<RwLock<u64>>,
}

impl Default for PositionFreeze {
    fn default() -> Self {
        Self::new(PositionFreezeConfig::default())
    }
}

impl PositionFreeze {
    /// Create a new position freeze circuit breaker
    pub fn new(config: PositionFreezeConfig) -> Self {
        Self {
            config,
            active_freezes: Arc::new(RwLock::new(HashMap::new())),
            freeze_history: Arc::new(RwLock::new(Vec::new())),
            total_freezes: Arc::new(RwLock::new(0)),
        }
    }

    /// Activate a freeze
    pub async fn freeze(
        &self,
        level: FreezeLevel,
        scope: FreezeScope,
        reason: FreezeReason,
        duration_secs: Option<u64>,
    ) -> Result<FreezeResult> {
        // Check max concurrent freezes
        let active_count = self.active_freezes.read().await.len();
        if active_count >= self.config.max_concurrent_freezes {
            return Err(anyhow::anyhow!(
                "Maximum concurrent freezes ({}) reached",
                self.config.max_concurrent_freezes
            )
            .into());
        }

        let freeze_id = self.generate_freeze_id().await;
        let activated_at = Utc::now();

        let auto_unfreeze_at = duration_secs
            .or(self.config.default_duration_secs)
            .map(|secs| activated_at + Duration::seconds(secs as i64));

        let freeze = ActiveFreeze {
            id: freeze_id.clone(),
            activated_at,
            level,
            scope: scope.clone(),
            reason: reason.clone(),
            auto_unfreeze_at,
            blocked_operations: 0,
        };

        info!("🔒 POSITION FREEZE ACTIVATED: {} - {:?}", freeze_id, reason);

        // Store active freeze
        {
            let mut freezes = self.active_freezes.write().await;
            freezes.insert(freeze_id.clone(), freeze.clone());
        }

        // Update stats
        {
            let mut total = self.total_freezes.write().await;
            *total += 1;
        }

        Ok(FreezeResult {
            freeze_id,
            activated_at,
            scope,
            level,
        })
    }

    /// Activate a global freeze
    pub async fn freeze_all(
        &self,
        level: FreezeLevel,
        reason: FreezeReason,
        duration_secs: Option<u64>,
    ) -> Result<FreezeResult> {
        self.freeze(level, FreezeScope::Global, reason, duration_secs)
            .await
    }

    /// Freeze specific symbols
    pub async fn freeze_symbols(
        &self,
        symbols: HashSet<String>,
        level: FreezeLevel,
        reason: FreezeReason,
        duration_secs: Option<u64>,
    ) -> Result<FreezeResult> {
        self.freeze(level, FreezeScope::Symbols(symbols), reason, duration_secs)
            .await
    }

    /// Unfreeze by ID
    pub async fn unfreeze(&self, freeze_id: &str) -> Result<()> {
        let mut freezes = self.active_freezes.write().await;

        if let Some(freeze) = freezes.remove(freeze_id) {
            info!("🔓 POSITION FREEZE DEACTIVATED: {}", freeze_id);

            // Archive to history
            let mut history = self.freeze_history.write().await;
            history.push(freeze);

            Ok(())
        } else {
            Err(anyhow::anyhow!("Freeze ID not found: {}", freeze_id).into())
        }
    }

    /// Unfreeze all active freezes
    pub async fn unfreeze_all(&self) -> Result<usize> {
        let mut freezes = self.active_freezes.write().await;
        let count = freezes.len();

        // Move all to history
        {
            let mut history = self.freeze_history.write().await;
            for freeze in freezes.values() {
                history.push(freeze.clone());
            }
        }

        freezes.clear();

        info!("🔓 ALL POSITION FREEZES DEACTIVATED ({})", count);
        Ok(count)
    }

    /// Check if an operation is allowed
    pub async fn is_operation_allowed(
        &self,
        symbol: &str,
        exchange: &str,
        operation: PositionOperation,
    ) -> OperationAllowance {
        self.cleanup_expired_freezes().await;

        let freezes = self.active_freezes.read().await;

        // Check all active freezes
        for freeze in freezes.values() {
            // Check if this freeze applies to this symbol/exchange
            let applies = match &freeze.scope {
                FreezeScope::Global => true,
                FreezeScope::Symbols(symbols) => symbols.contains(symbol),
                FreezeScope::Exchanges(exchanges) => exchanges.contains(exchange),
            };

            if !applies {
                continue;
            }

            // Check if operation is allowed under this freeze level
            let allowed = match freeze.level {
                FreezeLevel::EmergencyExitOnly => {
                    matches!(operation, PositionOperation::EmergencyExit)
                }
                FreezeLevel::NoIncrease => !matches!(operation, PositionOperation::Increase),
                FreezeLevel::Complete => {
                    self.config.allow_emergency_exits
                        && matches!(operation, PositionOperation::EmergencyExit)
                }
                FreezeLevel::ReductionOnly => matches!(
                    operation,
                    PositionOperation::Decrease | PositionOperation::EmergencyExit
                ),
            };

            if !allowed {
                return OperationAllowance::Blocked {
                    freeze_id: freeze.id.clone(),
                    reason: format!("Blocked by {:?} freeze: {:?}", freeze.level, freeze.reason),
                };
            }
        }

        OperationAllowance::Allowed
    }

    /// Record a blocked operation
    pub async fn record_blocked_operation(&self, freeze_id: &str) {
        let mut freezes = self.active_freezes.write().await;
        if let Some(freeze) = freezes.get_mut(freeze_id) {
            freeze.blocked_operations += 1;
        }
    }

    /// Clean up expired auto-unfreeze entries
    async fn cleanup_expired_freezes(&self) {
        let now = Utc::now();
        let mut freezes = self.active_freezes.write().await;
        let mut to_remove = Vec::new();

        for (id, freeze) in freezes.iter() {
            if let Some(unfreeze_at) = freeze.auto_unfreeze_at {
                if now >= unfreeze_at {
                    to_remove.push(id.clone());
                }
            }
        }

        if !to_remove.is_empty() {
            let mut history = self.freeze_history.write().await;
            for id in to_remove {
                if let Some(freeze) = freezes.remove(&id) {
                    info!("🔓 Auto-unfroze expired freeze: {}", id);
                    history.push(freeze);
                }
            }
        }
    }

    /// Generate a unique freeze ID
    async fn generate_freeze_id(&self) -> String {
        let count = *self.total_freezes.read().await;
        format!("FREEZE-{}-{}", Utc::now().timestamp(), count + 1)
    }

    /// Get all active freezes
    pub async fn get_active_freezes(&self) -> Vec<ActiveFreeze> {
        self.cleanup_expired_freezes().await;
        self.active_freezes.read().await.values().cloned().collect()
    }

    /// Get freeze by ID
    pub async fn get_freeze(&self, freeze_id: &str) -> Option<ActiveFreeze> {
        self.active_freezes.read().await.get(freeze_id).cloned()
    }

    /// Get statistics
    pub async fn stats(&self) -> PositionFreezeStats {
        self.cleanup_expired_freezes().await;

        let active_freezes = self.active_freezes.read().await;
        let history = self.freeze_history.read().await;
        let total = *self.total_freezes.read().await;

        let total_blocked_ops: u64 = active_freezes.values().map(|f| f.blocked_operations).sum();

        PositionFreezeStats {
            active_freezes: active_freezes.len(),
            total_freezes: total,
            total_blocked_operations: total_blocked_ops,
            history_count: history.len(),
        }
    }

    /// Check if globally frozen
    pub async fn is_globally_frozen(&self) -> bool {
        self.cleanup_expired_freezes().await;
        let freezes = self.active_freezes.read().await;
        freezes
            .values()
            .any(|f| matches!(f.scope, FreezeScope::Global))
    }

    /// Check if symbol is frozen
    pub async fn is_symbol_frozen(&self, symbol: &str) -> bool {
        self.cleanup_expired_freezes().await;
        let freezes = self.active_freezes.read().await;
        freezes.values().any(|f| match &f.scope {
            FreezeScope::Global => true,
            FreezeScope::Symbols(symbols) => symbols.contains(symbol),
            _ => false,
        })
    }
}

/// Position operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionOperation {
    /// Increase position size
    Increase,
    /// Decrease position size
    Decrease,
    /// Emergency exit (sell all at market)
    EmergencyExit,
    /// Modify stop/limit (no size change)
    Modify,
}

/// Operation allowance result
#[derive(Debug, Clone)]
pub enum OperationAllowance {
    /// Operation is allowed
    Allowed,
    /// Operation is blocked
    Blocked { freeze_id: String, reason: String },
}

/// Position freeze statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionFreezeStats {
    pub active_freezes: usize,
    pub total_freezes: u64,
    pub total_blocked_operations: u64,
    pub history_count: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_freeze_creation() {
        let freeze = PositionFreeze::default();
        let stats = freeze.stats().await;
        assert_eq!(stats.active_freezes, 0);
        assert_eq!(stats.total_freezes, 0);
    }

    #[tokio::test]
    async fn test_global_freeze() {
        let freeze = PositionFreeze::default();

        let reason = FreezeReason::HighVolatility { vix_level: 40.0 };
        let result = freeze
            .freeze_all(FreezeLevel::NoIncrease, reason, Some(60))
            .await
            .unwrap();

        assert!(!result.freeze_id.is_empty());
        assert!(matches!(result.scope, FreezeScope::Global));

        let stats = freeze.stats().await;
        assert_eq!(stats.active_freezes, 1);
    }

    #[tokio::test]
    async fn test_symbol_freeze() {
        let freeze = PositionFreeze::default();

        let mut symbols = HashSet::new();
        symbols.insert("AAPL".to_string());
        symbols.insert("MSFT".to_string());

        let reason = FreezeReason::NewsEvent {
            event: "Earnings".to_string(),
            time: Utc::now(),
        };

        freeze
            .freeze_symbols(symbols, FreezeLevel::Complete, reason, Some(300))
            .await
            .unwrap();

        assert!(freeze.is_symbol_frozen("AAPL").await);
        assert!(freeze.is_symbol_frozen("MSFT").await);
        assert!(!freeze.is_symbol_frozen("TSLA").await);
    }

    #[tokio::test]
    async fn test_operation_allowance() {
        let freeze = PositionFreeze::default();

        let reason = FreezeReason::Manual {
            operator: "admin".to_string(),
            reason: "Test".to_string(),
        };

        freeze
            .freeze_all(FreezeLevel::NoIncrease, reason, None)
            .await
            .unwrap();

        // Increase should be blocked
        let result = freeze
            .is_operation_allowed("AAPL", "NYSE", PositionOperation::Increase)
            .await;
        assert!(matches!(result, OperationAllowance::Blocked { .. }));

        // Decrease should be allowed
        let result = freeze
            .is_operation_allowed("AAPL", "NYSE", PositionOperation::Decrease)
            .await;
        assert!(matches!(result, OperationAllowance::Allowed));

        // Emergency exit should be allowed
        let result = freeze
            .is_operation_allowed("AAPL", "NYSE", PositionOperation::EmergencyExit)
            .await;
        assert!(matches!(result, OperationAllowance::Allowed));
    }

    #[tokio::test]
    async fn test_complete_freeze() {
        let config = PositionFreezeConfig {
            allow_emergency_exits: true,
            ..Default::default()
        };
        let freeze = PositionFreeze::new(config);

        let reason = FreezeReason::SystemDiagnostics;
        freeze
            .freeze_all(FreezeLevel::Complete, reason, None)
            .await
            .unwrap();

        // All operations should be blocked except emergency exit
        let result = freeze
            .is_operation_allowed("AAPL", "NYSE", PositionOperation::Increase)
            .await;
        assert!(matches!(result, OperationAllowance::Blocked { .. }));

        let result = freeze
            .is_operation_allowed("AAPL", "NYSE", PositionOperation::Decrease)
            .await;
        assert!(matches!(result, OperationAllowance::Blocked { .. }));

        // Emergency exit should still be allowed
        let result = freeze
            .is_operation_allowed("AAPL", "NYSE", PositionOperation::EmergencyExit)
            .await;
        assert!(matches!(result, OperationAllowance::Allowed));
    }

    #[tokio::test]
    async fn test_unfreeze() {
        let freeze = PositionFreeze::default();

        let reason = FreezeReason::Custom {
            description: "Test".to_string(),
        };
        let result = freeze
            .freeze_all(FreezeLevel::NoIncrease, reason, None)
            .await
            .unwrap();

        assert_eq!(freeze.stats().await.active_freezes, 1);

        freeze.unfreeze(&result.freeze_id).await.unwrap();

        assert_eq!(freeze.stats().await.active_freezes, 0);
        assert_eq!(freeze.stats().await.history_count, 1);
    }

    #[tokio::test]
    async fn test_unfreeze_all() {
        let freeze = PositionFreeze::default();

        let reason = FreezeReason::Custom {
            description: "Test1".to_string(),
        };
        freeze
            .freeze_all(FreezeLevel::NoIncrease, reason, None)
            .await
            .unwrap();

        let reason2 = FreezeReason::Custom {
            description: "Test2".to_string(),
        };
        let mut symbols = HashSet::new();
        symbols.insert("AAPL".to_string());
        freeze
            .freeze_symbols(symbols, FreezeLevel::Complete, reason2, None)
            .await
            .unwrap();

        assert_eq!(freeze.stats().await.active_freezes, 2);

        let count = freeze.unfreeze_all().await.unwrap();
        assert_eq!(count, 2);
        assert_eq!(freeze.stats().await.active_freezes, 0);
    }

    #[tokio::test]
    async fn test_auto_unfreeze() {
        let freeze = PositionFreeze::default();

        let reason = FreezeReason::Custom {
            description: "Auto".to_string(),
        };

        // Freeze for 1 second
        freeze
            .freeze_all(FreezeLevel::NoIncrease, reason, Some(1))
            .await
            .unwrap();

        assert_eq!(freeze.stats().await.active_freezes, 1);

        // Wait for expiration
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

        // Should be auto-unfrozen
        let stats = freeze.stats().await;
        assert_eq!(stats.active_freezes, 0);
    }

    #[tokio::test]
    async fn test_max_concurrent_freezes() {
        let config = PositionFreezeConfig {
            max_concurrent_freezes: 2,
            ..Default::default()
        };
        let freeze = PositionFreeze::new(config);

        let reason = FreezeReason::Custom {
            description: "Test".to_string(),
        };

        // First two should succeed
        freeze
            .freeze_all(FreezeLevel::NoIncrease, reason.clone(), None)
            .await
            .unwrap();

        let mut symbols = HashSet::new();
        symbols.insert("AAPL".to_string());
        freeze
            .freeze_symbols(symbols, FreezeLevel::Complete, reason.clone(), None)
            .await
            .unwrap();

        // Third should fail
        let result = freeze
            .freeze_all(FreezeLevel::NoIncrease, reason, None)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_freeze_levels() {
        assert!(matches!(
            FreezeLevel::EmergencyExitOnly,
            FreezeLevel::EmergencyExitOnly
        ));
        assert!(matches!(FreezeLevel::NoIncrease, FreezeLevel::NoIncrease));
        assert!(matches!(FreezeLevel::Complete, FreezeLevel::Complete));
        assert!(matches!(
            FreezeLevel::ReductionOnly,
            FreezeLevel::ReductionOnly
        ));
    }
}
