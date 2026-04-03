//! # SafeMode Circuit Breaker
//!
//! Enters a conservative trading mode with strict limits and enhanced monitoring.
//! Unlike a complete shutdown, SafeMode allows limited trading under tight controls.
//!
//! ## Use Cases
//!
//! - Recovery from near-miss incident
//! - High market uncertainty
//! - New strategy testing
//! - Post-malfunction validation
//! - Gradual system restart after kill switch
//! - Training/demo mode
//!
//! ## Features
//!
//! - Reduced position size limits
//! - Restricted order types (market/limit only)
//! - Lower risk parameters
//! - Enhanced logging and monitoring
//! - Gradual exit capability
//! - Whitelist-based symbol trading

use crate::common::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Safe mode restriction level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafeModeLevel {
    /// Level 1: Minimal restrictions (80% of normal)
    Minimal,

    /// Level 2: Moderate restrictions (50% of normal)
    Moderate,

    /// Level 3: Strict restrictions (25% of normal)
    Strict,

    /// Level 4: Ultra-conservative (10% of normal)
    UltraConservative,
}

impl SafeModeLevel {
    /// Get position size multiplier for this level
    pub fn position_multiplier(&self) -> f64 {
        match self {
            SafeModeLevel::Minimal => 0.8,
            SafeModeLevel::Moderate => 0.5,
            SafeModeLevel::Strict => 0.25,
            SafeModeLevel::UltraConservative => 0.1,
        }
    }

    /// Get risk multiplier for this level
    pub fn risk_multiplier(&self) -> f64 {
        match self {
            SafeModeLevel::Minimal => 0.7,
            SafeModeLevel::Moderate => 0.4,
            SafeModeLevel::Strict => 0.2,
            SafeModeLevel::UltraConservative => 0.05,
        }
    }
}

/// Allowed order types in safe mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllowedOrderType {
    /// Market orders only
    MarketOnly,

    /// Market and limit orders
    MarketAndLimit,

    /// All standard orders (no exotic types)
    Standard,
}

/// Reason for entering safe mode
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafeModeReason {
    /// Recovering from incident
    PostIncident { incident: String },

    /// High market uncertainty
    MarketUncertainty { metric: String, value: f64 },

    /// Testing new strategy
    StrategyTest { strategy: String },

    /// Manual activation
    Manual { operator: String, reason: String },

    /// System validation
    SystemValidation,

    /// Gradual restart after shutdown
    PostShutdown { shutdown_reason: String },

    /// Custom reason
    Custom { description: String },
}

/// Safe mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeModeConfig {
    /// Restriction level
    pub level: SafeModeLevel,

    /// Maximum position size (USD)
    pub max_position_size: f64,

    /// Maximum order size (USD)
    pub max_order_size: f64,

    /// Maximum total exposure (USD)
    pub max_total_exposure: f64,

    /// Allowed order types
    pub allowed_order_types: AllowedOrderType,

    /// Whitelisted symbols (empty = all allowed with restrictions)
    pub symbol_whitelist: HashSet<String>,

    /// Maximum concurrent positions
    pub max_concurrent_positions: usize,

    /// Require confirmations for orders above threshold
    pub confirmation_threshold: Option<f64>,

    /// Enhanced logging
    pub enhanced_logging: bool,

    /// Auto-exit after duration (seconds)
    pub auto_exit_after_secs: Option<u64>,
}

impl Default for SafeModeConfig {
    fn default() -> Self {
        Self {
            level: SafeModeLevel::Moderate,
            max_position_size: 10000.0,
            max_order_size: 5000.0,
            max_total_exposure: 50000.0,
            allowed_order_types: AllowedOrderType::MarketAndLimit,
            symbol_whitelist: HashSet::new(),
            max_concurrent_positions: 5,
            confirmation_threshold: Some(2000.0),
            enhanced_logging: true,
            auto_exit_after_secs: Some(3600), // 1 hour
        }
    }
}

/// Safe mode state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafeModeState {
    /// Safe mode not active
    Inactive,

    /// Safe mode active
    Active,

    /// Gradual exit in progress
    Exiting,
}

/// Order validation result
#[derive(Debug, Clone)]
pub enum OrderValidation {
    /// Order is allowed
    Allowed,

    /// Order requires confirmation
    RequiresConfirmation { reason: String },

    /// Order is rejected
    Rejected { reason: String },
}

/// Safe mode circuit breaker
pub struct SafeMode {
    config: Arc<RwLock<SafeModeConfig>>,
    state: Arc<RwLock<SafeModeState>>,
    activation_time: Arc<RwLock<Option<DateTime<Utc>>>>,
    activation_reason: Arc<RwLock<Option<SafeModeReason>>>,
    rejected_orders: Arc<RwLock<u64>>,
    confirmed_orders: Arc<RwLock<u64>>,
    current_exposure: Arc<RwLock<f64>>,
    active_positions: Arc<RwLock<HashMap<String, f64>>>,
}

impl Default for SafeMode {
    fn default() -> Self {
        Self::new(SafeModeConfig::default())
    }
}

impl SafeMode {
    /// Create a new safe mode circuit breaker
    pub fn new(config: SafeModeConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            state: Arc::new(RwLock::new(SafeModeState::Inactive)),
            activation_time: Arc::new(RwLock::new(None)),
            activation_reason: Arc::new(RwLock::new(None)),
            rejected_orders: Arc::new(RwLock::new(0)),
            confirmed_orders: Arc::new(RwLock::new(0)),
            current_exposure: Arc::new(RwLock::new(0.0)),
            active_positions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Enter safe mode
    pub async fn enter(&self, reason: SafeModeReason) -> Result<()> {
        let mut state = self.state.write().await;

        if *state != SafeModeState::Inactive {
            return Err(anyhow::anyhow!("Safe mode already active").into());
        }

        info!("🛡️  ENTERING SAFE MODE: {:?}", reason);

        *state = SafeModeState::Active;

        {
            let mut activation_time = self.activation_time.write().await;
            *activation_time = Some(Utc::now());
        }

        {
            let mut activation_reason = self.activation_reason.write().await;
            *activation_reason = Some(reason);
        }

        // Reset counters
        {
            let mut rejected = self.rejected_orders.write().await;
            *rejected = 0;
            let mut confirmed = self.confirmed_orders.write().await;
            *confirmed = 0;
        }

        info!("✅ Safe mode activated");
        Ok(())
    }

    /// Exit safe mode
    pub async fn exit(&self) -> Result<()> {
        let mut state = self.state.write().await;

        if *state == SafeModeState::Inactive {
            return Err(anyhow::anyhow!("Safe mode not active").into());
        }

        info!("🛡️  EXITING SAFE MODE");

        *state = SafeModeState::Inactive;

        {
            let mut activation_time = self.activation_time.write().await;
            *activation_time = None;
        }

        {
            let mut activation_reason = self.activation_reason.write().await;
            *activation_reason = None;
        }

        info!("✅ Safe mode deactivated");
        Ok(())
    }

    /// Begin gradual exit from safe mode
    ///
    /// Spawns a background task that progressively relaxes trading restrictions
    /// over `steps` intervals, each lasting `step_duration_secs` seconds.
    ///
    /// At each step, position limits, order limits, and exposure caps are
    /// linearly interpolated from their safe-mode values back toward the
    /// original (pre-safe-mode) defaults. Once all steps complete, safe mode
    /// is fully deactivated.
    ///
    /// If any step fails or the state is externally changed, the gradual exit
    /// is aborted and the system remains in its current state.
    pub async fn begin_gradual_exit(&self, steps: u32, step_duration_secs: u64) -> Result<()> {
        if steps == 0 {
            return Err(anyhow::anyhow!("Steps must be > 0").into());
        }

        {
            let mut state = self.state.write().await;

            if *state != SafeModeState::Active {
                return Err(anyhow::anyhow!("Safe mode not active").into());
            }

            info!(
                "🛡️  Beginning gradual exit from safe mode ({} steps, {}s per step)",
                steps, step_duration_secs
            );

            *state = SafeModeState::Exiting;
        }

        // Capture the current restricted config as the starting point
        let starting_config = self.config.read().await.clone();

        // Define normal operating limits to interpolate toward
        // These represent the fully-relaxed (non-safe-mode) defaults
        let target_max_position_size = 100_000.0_f64;
        let target_max_order_size = 50_000.0_f64;
        let target_max_total_exposure = 500_000.0_f64;
        let target_max_concurrent_positions: usize = 50;

        let config = Arc::clone(&self.config);
        let state = Arc::clone(&self.state);
        let activation_time = Arc::clone(&self.activation_time);
        let activation_reason = Arc::clone(&self.activation_reason);

        tokio::spawn(async move {
            let step_duration = std::time::Duration::from_secs(step_duration_secs);

            for step in 1..=steps {
                // Wait for the step interval before relaxing further
                tokio::time::sleep(step_duration).await;

                // Verify we're still in the Exiting state (could have been
                // manually overridden by enter() or exit())
                {
                    let current_state = state.read().await;
                    if *current_state != SafeModeState::Exiting {
                        warn!(
                            "🛡️  Gradual exit aborted at step {}/{} — state changed to {:?}",
                            step, steps, *current_state
                        );
                        return;
                    }
                }

                let progress = step as f64 / steps as f64;

                // Linearly interpolate limits from safe-mode values toward normal
                let new_max_position = starting_config.max_position_size
                    + (target_max_position_size - starting_config.max_position_size) * progress;
                let new_max_order = starting_config.max_order_size
                    + (target_max_order_size - starting_config.max_order_size) * progress;
                let new_max_exposure = starting_config.max_total_exposure
                    + (target_max_total_exposure - starting_config.max_total_exposure) * progress;
                let new_max_positions = starting_config.max_concurrent_positions
                    + ((target_max_concurrent_positions as f64
                        - starting_config.max_concurrent_positions as f64)
                        * progress) as usize;

                // Apply relaxed limits
                {
                    let mut cfg = config.write().await;
                    cfg.max_position_size = new_max_position;
                    cfg.max_order_size = new_max_order;
                    cfg.max_total_exposure = new_max_exposure;
                    cfg.max_concurrent_positions = new_max_positions;

                    // At 50% progress, widen allowed order types
                    if progress >= 0.5 {
                        cfg.allowed_order_types = AllowedOrderType::Standard;
                    }

                    // At 75% progress, remove confirmation requirement
                    if progress >= 0.75 {
                        cfg.confirmation_threshold = None;
                    }

                    info!(
                        "🛡️  Safe mode exit step {}/{} ({:.0}%): position_limit={:.0}, \
                         order_limit={:.0}, exposure_limit={:.0}, max_positions={}",
                        step,
                        steps,
                        progress * 100.0,
                        new_max_position,
                        new_max_order,
                        new_max_exposure,
                        new_max_positions
                    );
                }
            }

            // All steps complete — fully deactivate safe mode
            {
                let mut s = state.write().await;
                *s = SafeModeState::Inactive;
            }
            {
                let mut t = activation_time.write().await;
                *t = None;
            }
            {
                let mut r = activation_reason.write().await;
                *r = None;
            }

            info!("🛡️  ✅ Safe mode fully deactivated after gradual exit");
        });

        Ok(())
    }

    /// Validate an order against safe mode restrictions
    pub async fn validate_order(
        &self,
        symbol: &str,
        order_value: f64,
        order_type: &str,
    ) -> OrderValidation {
        let state = *self.state.read().await;

        if state == SafeModeState::Inactive {
            return OrderValidation::Allowed;
        }

        let config = self.config.read().await;

        // Check symbol whitelist
        if !config.symbol_whitelist.is_empty() && !config.symbol_whitelist.contains(symbol) {
            return OrderValidation::Rejected {
                reason: format!("Symbol {} not in safe mode whitelist", symbol),
            };
        }

        // Check order type
        let allowed = match config.allowed_order_types {
            AllowedOrderType::MarketOnly => order_type == "MARKET",
            AllowedOrderType::MarketAndLimit => order_type == "MARKET" || order_type == "LIMIT",
            AllowedOrderType::Standard => !matches!(
                order_type,
                "STOP_MARKET" | "TRAILING_STOP" | "OCO" | "ICEBERG"
            ),
        };

        if !allowed {
            return OrderValidation::Rejected {
                reason: format!("Order type {} not allowed in safe mode", order_type),
            };
        }

        // Check order size
        if order_value > config.max_order_size {
            return OrderValidation::Rejected {
                reason: format!(
                    "Order value ${:.2} exceeds safe mode limit ${:.2}",
                    order_value, config.max_order_size
                ),
            };
        }

        // Check position size
        let positions = self.active_positions.read().await;
        let current_position = positions.get(symbol).unwrap_or(&0.0);
        let new_position = current_position + order_value;

        if new_position > config.max_position_size {
            return OrderValidation::Rejected {
                reason: format!(
                    "Position would exceed safe mode limit ${:.2}",
                    config.max_position_size
                ),
            };
        }

        // Check total exposure
        let current_exposure = *self.current_exposure.read().await;
        if current_exposure + order_value > config.max_total_exposure {
            return OrderValidation::Rejected {
                reason: format!(
                    "Total exposure would exceed safe mode limit ${:.2}",
                    config.max_total_exposure
                ),
            };
        }

        // Check concurrent positions
        if !positions.contains_key(symbol) && positions.len() >= config.max_concurrent_positions {
            return OrderValidation::Rejected {
                reason: format!(
                    "Maximum concurrent positions ({}) reached",
                    config.max_concurrent_positions
                ),
            };
        }

        // Check confirmation threshold
        if let Some(threshold) = config.confirmation_threshold {
            if order_value > threshold {
                return OrderValidation::RequiresConfirmation {
                    reason: format!("Order value ${:.2} requires confirmation", order_value),
                };
            }
        }

        OrderValidation::Allowed
    }

    /// Record an order execution (update exposure tracking)
    pub async fn record_order(&self, symbol: String, value: f64) {
        let mut positions = self.active_positions.write().await;
        *positions.entry(symbol).or_insert(0.0) += value;

        let mut exposure = self.current_exposure.write().await;
        *exposure += value;
    }

    /// Record a position close
    pub async fn close_position(&self, symbol: &str) {
        let mut positions = self.active_positions.write().await;
        if let Some(value) = positions.remove(symbol) {
            let mut exposure = self.current_exposure.write().await;
            *exposure -= value;
        }
    }

    /// Record a rejected order
    pub async fn record_rejection(&self) {
        let mut rejected = self.rejected_orders.write().await;
        *rejected += 1;
    }

    /// Record a confirmed order
    pub async fn record_confirmation(&self) {
        let mut confirmed = self.confirmed_orders.write().await;
        *confirmed += 1;
    }

    /// Update safe mode level
    pub async fn set_level(&self, level: SafeModeLevel) -> Result<()> {
        let state = *self.state.read().await;
        if state == SafeModeState::Inactive {
            return Err(anyhow::anyhow!("Safe mode not active").into());
        }

        let mut config = self.config.write().await;
        let old_level = config.level;
        config.level = level;

        // Adjust limits based on new level
        let multiplier = level.position_multiplier();
        config.max_position_size *= multiplier / old_level.position_multiplier();
        config.max_order_size *= multiplier / old_level.position_multiplier();
        config.max_total_exposure *= multiplier / old_level.position_multiplier();

        info!("Safe mode level changed: {:?} -> {:?}", old_level, level);
        Ok(())
    }

    /// Check if safe mode is active
    pub async fn is_active(&self) -> bool {
        *self.state.read().await != SafeModeState::Inactive
    }

    /// Get current state
    pub async fn state(&self) -> SafeModeState {
        *self.state.read().await
    }

    /// Check for auto-exit
    pub async fn check_auto_exit(&self) -> bool {
        let state = *self.state.read().await;
        if state != SafeModeState::Active {
            return false;
        }

        let config = self.config.read().await;
        if let Some(auto_exit_secs) = config.auto_exit_after_secs {
            if let Some(activation_time) = *self.activation_time.read().await {
                let elapsed = Utc::now().signed_duration_since(activation_time);
                if elapsed.num_seconds() >= auto_exit_secs as i64 {
                    return true;
                }
            }
        }

        false
    }

    /// Get statistics
    pub async fn stats(&self) -> SafeModeStats {
        let state = *self.state.read().await;
        let activation_time = *self.activation_time.read().await;
        let rejected = *self.rejected_orders.read().await;
        let confirmed = *self.confirmed_orders.read().await;
        let exposure = *self.current_exposure.read().await;
        let active_positions = self.active_positions.read().await.len();
        let config = self.config.read().await;

        SafeModeStats {
            state,
            level: config.level,
            activation_time,
            rejected_orders: rejected,
            confirmed_orders: confirmed,
            current_exposure: exposure,
            active_positions,
            max_exposure: config.max_total_exposure,
        }
    }
}

/// Safe mode statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeModeStats {
    pub state: SafeModeState,
    pub level: SafeModeLevel,
    pub activation_time: Option<DateTime<Utc>>,
    pub rejected_orders: u64,
    pub confirmed_orders: u64,
    pub current_exposure: f64,
    pub active_positions: usize,
    pub max_exposure: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_safe_mode_creation() {
        let safe_mode = SafeMode::default();
        assert!(!safe_mode.is_active().await);
        assert_eq!(safe_mode.state().await, SafeModeState::Inactive);
    }

    #[tokio::test]
    async fn test_enter_safe_mode() {
        let safe_mode = SafeMode::default();

        let reason = SafeModeReason::Manual {
            operator: "admin".to_string(),
            reason: "Testing".to_string(),
        };

        safe_mode.enter(reason).await.unwrap();
        assert!(safe_mode.is_active().await);
        assert_eq!(safe_mode.state().await, SafeModeState::Active);

        let stats = safe_mode.stats().await;
        assert!(stats.activation_time.is_some());
    }

    #[tokio::test]
    async fn test_exit_safe_mode() {
        let safe_mode = SafeMode::default();

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();
        assert!(safe_mode.is_active().await);

        safe_mode.exit().await.unwrap();
        assert!(!safe_mode.is_active().await);
    }

    #[tokio::test]
    async fn test_order_validation_allowed() {
        let safe_mode = SafeMode::default();

        let reason = SafeModeReason::StrategyTest {
            strategy: "test".to_string(),
        };
        safe_mode.enter(reason).await.unwrap();

        let validation = safe_mode.validate_order("AAPL", 1000.0, "MARKET").await;
        assert!(matches!(validation, OrderValidation::Allowed));
    }

    #[tokio::test]
    async fn test_order_validation_size_limit() {
        let safe_mode = SafeMode::default();

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();

        // Order exceeds max_order_size (default 5000.0)
        let validation = safe_mode.validate_order("AAPL", 10000.0, "MARKET").await;
        assert!(matches!(validation, OrderValidation::Rejected { .. }));
    }

    #[tokio::test]
    async fn test_order_validation_order_type() {
        let config = SafeModeConfig {
            allowed_order_types: AllowedOrderType::MarketOnly,
            ..Default::default()
        };
        let safe_mode = SafeMode::new(config);

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();

        let validation = safe_mode.validate_order("AAPL", 1000.0, "LIMIT").await;
        assert!(matches!(validation, OrderValidation::Rejected { .. }));
    }

    #[tokio::test]
    async fn test_symbol_whitelist() {
        let mut whitelist = HashSet::new();
        whitelist.insert("AAPL".to_string());
        whitelist.insert("MSFT".to_string());

        let config = SafeModeConfig {
            symbol_whitelist: whitelist,
            ..Default::default()
        };
        let safe_mode = SafeMode::new(config);

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();

        // AAPL should be allowed
        let validation = safe_mode.validate_order("AAPL", 1000.0, "MARKET").await;
        assert!(matches!(validation, OrderValidation::Allowed));

        // TSLA should be rejected
        let validation = safe_mode.validate_order("TSLA", 1000.0, "MARKET").await;
        assert!(matches!(validation, OrderValidation::Rejected { .. }));
    }

    #[tokio::test]
    async fn test_confirmation_threshold() {
        let config = SafeModeConfig {
            confirmation_threshold: Some(2000.0),
            ..Default::default()
        };
        let safe_mode = SafeMode::new(config);

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();

        // Below threshold
        let validation = safe_mode.validate_order("AAPL", 1500.0, "MARKET").await;
        assert!(matches!(validation, OrderValidation::Allowed));

        // Above threshold
        let validation = safe_mode.validate_order("AAPL", 3000.0, "MARKET").await;
        assert!(matches!(
            validation,
            OrderValidation::RequiresConfirmation { .. }
        ));
    }

    #[tokio::test]
    async fn test_exposure_tracking() {
        let safe_mode = SafeMode::default();

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();

        safe_mode.record_order("AAPL".to_string(), 5000.0).await;
        safe_mode.record_order("MSFT".to_string(), 3000.0).await;

        let stats = safe_mode.stats().await;
        assert_eq!(stats.current_exposure, 8000.0);
        assert_eq!(stats.active_positions, 2);
    }

    #[tokio::test]
    async fn test_level_change() {
        let safe_mode = SafeMode::default();

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();

        let stats = safe_mode.stats().await;
        assert_eq!(stats.level, SafeModeLevel::Moderate);

        safe_mode.set_level(SafeModeLevel::Strict).await.unwrap();

        let stats = safe_mode.stats().await;
        assert_eq!(stats.level, SafeModeLevel::Strict);
    }

    #[tokio::test]
    async fn test_rejection_tracking() {
        let safe_mode = SafeMode::default();

        let reason = SafeModeReason::SystemValidation;
        safe_mode.enter(reason).await.unwrap();

        safe_mode.record_rejection().await;
        safe_mode.record_rejection().await;
        safe_mode.record_confirmation().await;

        let stats = safe_mode.stats().await;
        assert_eq!(stats.rejected_orders, 2);
        assert_eq!(stats.confirmed_orders, 1);
    }

    #[test]
    fn test_level_multipliers() {
        assert_eq!(SafeModeLevel::Minimal.position_multiplier(), 0.8);
        assert_eq!(SafeModeLevel::Moderate.position_multiplier(), 0.5);
        assert_eq!(SafeModeLevel::Strict.position_multiplier(), 0.25);
        assert_eq!(SafeModeLevel::UltraConservative.position_multiplier(), 0.1);
    }
}
