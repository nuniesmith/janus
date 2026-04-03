//! Leverage management
//!
//! Part of the Hypothalamus region
//! Component: energy
//!
//! This module implements dynamic leverage control, monitoring margin utilization,
//! and triggering deleveraging when risk thresholds are exceeded.

use crate::common::{Error, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Leverage tier configuration
#[derive(Debug, Clone)]
pub struct LeverageTier {
    /// Tier name (e.g., "conservative", "moderate", "aggressive")
    pub name: String,
    /// Maximum leverage allowed in this tier
    pub max_leverage: f64,
    /// Margin call threshold (margin utilization %)
    pub margin_call_threshold: f64,
    /// Liquidation threshold (margin utilization %)
    pub liquidation_threshold: f64,
    /// Warning threshold (margin utilization %)
    pub warning_threshold: f64,
}

impl Default for LeverageTier {
    fn default() -> Self {
        Self {
            name: "moderate".to_string(),
            max_leverage: 3.0,
            margin_call_threshold: 80.0,
            liquidation_threshold: 90.0,
            warning_threshold: 70.0,
        }
    }
}

/// Configuration for leverage control
#[derive(Debug, Clone)]
pub struct LeverageControlConfig {
    /// Available leverage tiers
    pub tiers: Vec<LeverageTier>,
    /// Current active tier index
    pub active_tier: usize,
    /// Maximum account-wide leverage
    pub max_account_leverage: f64,
    /// Per-symbol maximum leverage
    pub per_symbol_max_leverage: f64,
    /// Auto-deleverage when margin exceeds threshold
    pub auto_deleverage: bool,
    /// Deleverage target (margin utilization %)
    pub deleverage_target: f64,
    /// Cooldown between deleverage operations
    pub deleverage_cooldown_secs: u64,
    /// Minimum position size to maintain
    pub min_position_size: f64,
    /// Volatility scaling factor for dynamic leverage
    pub volatility_scaling: bool,
    /// Base volatility for scaling (annualized %)
    pub base_volatility: f64,
}

impl Default for LeverageControlConfig {
    fn default() -> Self {
        Self {
            tiers: vec![
                LeverageTier {
                    name: "conservative".to_string(),
                    max_leverage: 1.5,
                    margin_call_threshold: 70.0,
                    liquidation_threshold: 80.0,
                    warning_threshold: 60.0,
                },
                LeverageTier {
                    name: "moderate".to_string(),
                    max_leverage: 3.0,
                    margin_call_threshold: 80.0,
                    liquidation_threshold: 90.0,
                    warning_threshold: 70.0,
                },
                LeverageTier {
                    name: "aggressive".to_string(),
                    max_leverage: 5.0,
                    margin_call_threshold: 85.0,
                    liquidation_threshold: 95.0,
                    warning_threshold: 75.0,
                },
            ],
            active_tier: 1, // moderate
            max_account_leverage: 5.0,
            per_symbol_max_leverage: 10.0,
            auto_deleverage: true,
            deleverage_target: 60.0,
            deleverage_cooldown_secs: 300,
            min_position_size: 100.0,
            volatility_scaling: true,
            base_volatility: 20.0,
        }
    }
}

/// Position with leverage information
#[derive(Debug, Clone)]
pub struct LeveragedPosition {
    /// Symbol
    pub symbol: String,
    /// Position size (notional value)
    pub notional_value: f64,
    /// Margin used
    pub margin_used: f64,
    /// Current leverage
    pub leverage: f64,
    /// Unrealized PnL
    pub unrealized_pnl: f64,
    /// Liquidation price
    pub liquidation_price: Option<f64>,
}

/// Margin status for the account
#[derive(Debug, Clone)]
pub struct MarginStatus {
    /// Total account equity
    pub equity: f64,
    /// Total margin used
    pub margin_used: f64,
    /// Available margin
    pub available_margin: f64,
    /// Margin utilization (%)
    pub margin_utilization: f64,
    /// Current account leverage
    pub account_leverage: f64,
    /// Margin level indicator
    pub level: MarginLevel,
    /// Distance to margin call (%)
    pub distance_to_margin_call: f64,
    /// Distance to liquidation (%)
    pub distance_to_liquidation: f64,
}

/// Margin level indicator
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarginLevel {
    /// Safe - plenty of margin available
    Safe,
    /// Warning - approaching margin limits
    Warning,
    /// MarginCall - margin call threshold reached
    MarginCall,
    /// Critical - near liquidation
    Critical,
}

/// Deleverage instruction
#[derive(Debug, Clone)]
pub struct DeleverageInstruction {
    /// Symbol to reduce
    pub symbol: String,
    /// Amount to reduce (notional value)
    pub reduce_amount: f64,
    /// Reduce percentage (0-100)
    pub reduce_percentage: f64,
    /// Priority (higher = reduce first)
    pub priority: u8,
    /// Reason for deleverage
    pub reason: DeleverageReason,
}

/// Reason for deleveraging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleverageReason {
    /// Margin utilization too high
    MarginCall,
    /// Near liquidation
    Liquidation,
    /// Volatility spike
    VolatilitySpike,
    /// Account leverage limit
    AccountLeverage,
    /// Symbol leverage limit
    SymbolLeverage,
    /// Risk reduction
    RiskReduction,
    /// Manual trigger
    Manual,
}

/// Leverage check result
#[derive(Debug, Clone)]
pub struct LeverageCheckResult {
    /// Whether the leverage is acceptable
    pub acceptable: bool,
    /// Current leverage
    pub current_leverage: f64,
    /// Maximum allowed leverage
    pub max_allowed: f64,
    /// Reason if not acceptable
    pub rejection_reason: Option<String>,
    /// Adjusted leverage if volatility scaling applied
    pub adjusted_max: Option<f64>,
}

/// Leverage management
pub struct LeverageControl {
    /// Configuration
    config: LeverageControlConfig,
    /// Current positions
    positions: HashMap<String, LeveragedPosition>,
    /// Account equity
    account_equity: f64,
    /// Total margin used
    total_margin_used: f64,
    /// Last deleverage time
    last_deleverage: Option<Instant>,
    /// Deleverage history
    deleverage_history: Vec<(Instant, DeleverageInstruction)>,
    /// Current volatility by symbol
    symbol_volatility: HashMap<String, f64>,
    /// Warning callbacks triggered
    warnings_triggered: u64,
    /// Margin calls triggered
    margin_calls_triggered: u64,
}

impl Default for LeverageControl {
    fn default() -> Self {
        Self::new()
    }
}

impl LeverageControl {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(LeverageControlConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: LeverageControlConfig) -> Self {
        Self {
            config,
            positions: HashMap::new(),
            account_equity: 0.0,
            total_margin_used: 0.0,
            last_deleverage: None,
            deleverage_history: Vec::new(),
            symbol_volatility: HashMap::new(),
            warnings_triggered: 0,
            margin_calls_triggered: 0,
        }
    }

    /// Update account equity
    pub fn update_equity(&mut self, equity: f64) {
        self.account_equity = equity;
    }

    /// Update position
    pub fn update_position(&mut self, position: LeveragedPosition) {
        self.total_margin_used -= self
            .positions
            .get(&position.symbol)
            .map(|p| p.margin_used)
            .unwrap_or(0.0);
        self.total_margin_used += position.margin_used;
        self.positions.insert(position.symbol.clone(), position);
    }

    /// Remove position
    pub fn remove_position(&mut self, symbol: &str) {
        if let Some(pos) = self.positions.remove(symbol) {
            self.total_margin_used -= pos.margin_used;
        }
    }

    /// Update symbol volatility
    pub fn update_volatility(&mut self, symbol: &str, volatility: f64) {
        self.symbol_volatility
            .insert(symbol.to_string(), volatility);
    }

    /// Get current margin status
    pub fn get_margin_status(&self) -> MarginStatus {
        let margin_utilization = if self.account_equity > 0.0 {
            (self.total_margin_used / self.account_equity) * 100.0
        } else {
            0.0
        };

        let account_leverage = if self.account_equity > 0.0 {
            let total_notional: f64 = self
                .positions
                .values()
                .map(|p| p.notional_value.abs())
                .sum();
            total_notional / self.account_equity
        } else {
            0.0
        };

        let tier = self.get_active_tier();

        let level = if margin_utilization >= tier.liquidation_threshold {
            MarginLevel::Critical
        } else if margin_utilization >= tier.margin_call_threshold {
            MarginLevel::MarginCall
        } else if margin_utilization >= tier.warning_threshold {
            MarginLevel::Warning
        } else {
            MarginLevel::Safe
        };

        MarginStatus {
            equity: self.account_equity,
            margin_used: self.total_margin_used,
            available_margin: (self.account_equity - self.total_margin_used).max(0.0),
            margin_utilization,
            account_leverage,
            level,
            distance_to_margin_call: tier.margin_call_threshold - margin_utilization,
            distance_to_liquidation: tier.liquidation_threshold - margin_utilization,
        }
    }

    /// Get active leverage tier
    pub fn get_active_tier(&self) -> &LeverageTier {
        self.config
            .tiers
            .get(self.config.active_tier)
            .unwrap_or(&self.config.tiers[0])
    }

    /// Set active tier by index
    pub fn set_active_tier(&mut self, tier_index: usize) -> Result<()> {
        if tier_index >= self.config.tiers.len() {
            return Err(Error::InvalidArgument(format!(
                "Invalid tier index: {}",
                tier_index
            )));
        }
        self.config.active_tier = tier_index;
        Ok(())
    }

    /// Set active tier by name
    pub fn set_active_tier_by_name(&mut self, name: &str) -> Result<()> {
        let tier_index = self
            .config
            .tiers
            .iter()
            .position(|t| t.name == name)
            .ok_or_else(|| Error::NotFound(format!("Tier not found: {}", name)))?;
        self.config.active_tier = tier_index;
        Ok(())
    }

    /// Check if proposed leverage is acceptable
    pub fn check_leverage(&self, symbol: &str, proposed_leverage: f64) -> LeverageCheckResult {
        let tier = self.get_active_tier();
        let mut max_allowed = tier.max_leverage.min(self.config.per_symbol_max_leverage);

        // Apply volatility scaling if enabled
        let adjusted_max = if self.config.volatility_scaling {
            if let Some(&vol) = self.symbol_volatility.get(symbol) {
                // Scale down leverage when volatility is high
                let vol_ratio = self.config.base_volatility / vol.max(1.0);
                let scaled_max = max_allowed * vol_ratio.clamp(0.5, 1.5);
                max_allowed = scaled_max;
                Some(scaled_max)
            } else {
                None
            }
        } else {
            None
        };

        let acceptable = proposed_leverage <= max_allowed;
        let rejection_reason = if !acceptable {
            Some(format!(
                "Proposed leverage {:.2}x exceeds maximum allowed {:.2}x",
                proposed_leverage, max_allowed
            ))
        } else {
            None
        };

        LeverageCheckResult {
            acceptable,
            current_leverage: proposed_leverage,
            max_allowed,
            rejection_reason,
            adjusted_max,
        }
    }

    /// Check if new position would exceed account leverage limits
    pub fn check_account_leverage(&self, additional_notional: f64) -> LeverageCheckResult {
        let total_notional: f64 = self
            .positions
            .values()
            .map(|p| p.notional_value.abs())
            .sum();
        let new_total = total_notional + additional_notional.abs();

        let proposed_leverage = if self.account_equity > 0.0 {
            new_total / self.account_equity
        } else {
            f64::INFINITY
        };

        let max_allowed = self.config.max_account_leverage;
        let acceptable = proposed_leverage <= max_allowed;

        LeverageCheckResult {
            acceptable,
            current_leverage: proposed_leverage,
            max_allowed,
            rejection_reason: if !acceptable {
                Some(format!(
                    "Account leverage {:.2}x would exceed maximum {:.2}x",
                    proposed_leverage, max_allowed
                ))
            } else {
                None
            },
            adjusted_max: None,
        }
    }

    /// Generate deleverage instructions if needed
    pub fn generate_deleverage_instructions(&mut self) -> Vec<DeleverageInstruction> {
        let mut instructions = Vec::new();
        let status = self.get_margin_status();

        // Check if we need to deleverage
        let needs_deleverage = match status.level {
            MarginLevel::Critical | MarginLevel::MarginCall => true,
            MarginLevel::Warning => false, // Only warn, don't auto-deleverage
            MarginLevel::Safe => false,
        };

        if !needs_deleverage || !self.config.auto_deleverage {
            return instructions;
        }

        // Check cooldown
        if let Some(last) = self.last_deleverage {
            if last.elapsed() < Duration::from_secs(self.config.deleverage_cooldown_secs) {
                return instructions;
            }
        }

        // Calculate how much margin we need to free
        let target_utilization = self.config.deleverage_target;
        let target_margin = self.account_equity * (target_utilization / 100.0);
        let margin_to_free = self.total_margin_used - target_margin;

        if margin_to_free <= 0.0 {
            return instructions;
        }

        // Sort positions by priority (largest PnL loss first, then by size)
        let mut position_list: Vec<_> = self.positions.values().cloned().collect();
        position_list.sort_by(|a, b| {
            // Prioritize losing positions
            a.unrealized_pnl
                .partial_cmp(&b.unrealized_pnl)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    // Then by size (larger first)
                    b.notional_value
                        .abs()
                        .partial_cmp(&a.notional_value.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let mut margin_freed = 0.0;
        let reason = if status.level == MarginLevel::Critical {
            DeleverageReason::Liquidation
        } else {
            DeleverageReason::MarginCall
        };

        for (priority, pos) in position_list.iter().enumerate() {
            if margin_freed >= margin_to_free {
                break;
            }

            // Calculate how much to reduce this position
            let remaining_to_free = margin_to_free - margin_freed;
            let reduce_amount = remaining_to_free.min(pos.margin_used);
            let reduce_percentage = (reduce_amount / pos.margin_used) * 100.0;

            // Don't reduce below minimum size
            let new_notional = pos.notional_value * (1.0 - reduce_percentage / 100.0);
            if new_notional.abs() < self.config.min_position_size && reduce_percentage < 100.0 {
                // Close entirely instead
                instructions.push(DeleverageInstruction {
                    symbol: pos.symbol.clone(),
                    reduce_amount: pos.notional_value.abs(),
                    reduce_percentage: 100.0,
                    priority: priority as u8,
                    reason,
                });
                margin_freed += pos.margin_used;
            } else {
                instructions.push(DeleverageInstruction {
                    symbol: pos.symbol.clone(),
                    reduce_amount: pos.notional_value.abs() * (reduce_percentage / 100.0),
                    reduce_percentage,
                    priority: priority as u8,
                    reason,
                });
                margin_freed += reduce_amount;
            }
        }

        if !instructions.is_empty() {
            self.last_deleverage = Some(Instant::now());
            self.margin_calls_triggered += 1;
            for inst in &instructions {
                self.deleverage_history.push((Instant::now(), inst.clone()));
            }
        }

        instructions
    }

    /// Process margin check and return any warnings/actions needed
    pub fn process(&mut self) -> Result<Option<Vec<DeleverageInstruction>>> {
        let status = self.get_margin_status();

        // Track warnings
        if status.level == MarginLevel::Warning {
            self.warnings_triggered += 1;
        }

        // Generate deleverage instructions if needed
        let instructions = self.generate_deleverage_instructions();

        if instructions.is_empty() {
            Ok(None)
        } else {
            Ok(Some(instructions))
        }
    }

    /// Get maximum position size given current margin
    pub fn get_max_position_size(&self, symbol: &str) -> f64 {
        let tier = self.get_active_tier();
        let status = self.get_margin_status();

        // Available margin determines base max size
        let base_max = status.available_margin * tier.max_leverage;

        // Apply volatility scaling if enabled
        if self.config.volatility_scaling {
            if let Some(&vol) = self.symbol_volatility.get(symbol) {
                let vol_ratio = self.config.base_volatility / vol.max(1.0);
                return base_max * vol_ratio.clamp(0.5, 1.5);
            }
        }

        base_max
    }

    /// Get statistics
    pub fn get_stats(&self) -> LeverageControlStats {
        let status = self.get_margin_status();
        LeverageControlStats {
            account_leverage: status.account_leverage,
            margin_utilization: status.margin_utilization,
            margin_level: status.level,
            active_tier: self.get_active_tier().name.clone(),
            position_count: self.positions.len(),
            total_notional: self
                .positions
                .values()
                .map(|p| p.notional_value.abs())
                .sum(),
            warnings_triggered: self.warnings_triggered,
            margin_calls_triggered: self.margin_calls_triggered,
            deleverage_events: self.deleverage_history.len(),
        }
    }
}

/// Leverage control statistics
#[derive(Debug, Clone)]
pub struct LeverageControlStats {
    pub account_leverage: f64,
    pub margin_utilization: f64,
    pub margin_level: MarginLevel,
    pub active_tier: String,
    pub position_count: usize,
    pub total_notional: f64,
    pub warnings_triggered: u64,
    pub margin_calls_triggered: u64,
    pub deleverage_events: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = LeverageControl::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_margin_status_safe() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        let position = LeveragedPosition {
            symbol: "BTC-USD".to_string(),
            notional_value: 50_000.0,
            margin_used: 10_000.0,
            leverage: 5.0,
            unrealized_pnl: 500.0,
            liquidation_price: Some(45_000.0),
        };
        control.update_position(position);

        let status = control.get_margin_status();
        assert_eq!(status.level, MarginLevel::Safe);
        assert!((status.margin_utilization - 10.0).abs() < 0.01);
        assert!((status.account_leverage - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_margin_status_warning() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        // Add position that uses 75% margin (warning threshold is 70%)
        let position = LeveragedPosition {
            symbol: "BTC-USD".to_string(),
            notional_value: 375_000.0,
            margin_used: 75_000.0,
            leverage: 5.0,
            unrealized_pnl: -1000.0,
            liquidation_price: Some(42_000.0),
        };
        control.update_position(position);

        let status = control.get_margin_status();
        assert_eq!(status.level, MarginLevel::Warning);
    }

    #[test]
    fn test_margin_status_margin_call() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        // Add position that uses 85% margin (margin call threshold is 80%)
        let position = LeveragedPosition {
            symbol: "BTC-USD".to_string(),
            notional_value: 425_000.0,
            margin_used: 85_000.0,
            leverage: 5.0,
            unrealized_pnl: -5000.0,
            liquidation_price: Some(40_000.0),
        };
        control.update_position(position);

        let status = control.get_margin_status();
        assert_eq!(status.level, MarginLevel::MarginCall);
    }

    #[test]
    fn test_leverage_check() {
        let control = LeverageControl::new();

        // Moderate tier has max leverage of 3.0
        let check = control.check_leverage("BTC-USD", 2.5);
        assert!(check.acceptable);
        assert_eq!(check.max_allowed, 3.0);

        let check = control.check_leverage("BTC-USD", 5.0);
        assert!(!check.acceptable);
        assert!(check.rejection_reason.is_some());
    }

    #[test]
    fn test_volatility_scaling() {
        let mut control = LeverageControl::new();

        // Set high volatility (2x base)
        control.update_volatility("BTC-USD", 40.0); // base is 20%

        let check = control.check_leverage("BTC-USD", 2.0);
        // Max should be scaled down: 3.0 * (20/40) = 1.5
        assert!(!check.acceptable);
        assert!(check.adjusted_max.is_some());
        assert!((check.adjusted_max.unwrap() - 1.5).abs() < 0.01);

        // Lower volatility should allow higher leverage
        control.update_volatility("ETH-USD", 10.0);
        let check = control.check_leverage("ETH-USD", 3.0);
        // Max should be scaled up but capped: 3.0 * (20/10) = 6.0, but capped at 1.5x = 4.5
        assert!(check.acceptable);
    }

    #[test]
    fn test_account_leverage_check() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        // Max account leverage is 5.0x
        let check = control.check_account_leverage(400_000.0);
        assert!(check.acceptable);
        assert!((check.current_leverage - 4.0).abs() < 0.01);

        let check = control.check_account_leverage(600_000.0);
        assert!(!check.acceptable);
    }

    #[test]
    fn test_deleverage_generation() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        // Add positions that exceed margin call threshold
        control.update_position(LeveragedPosition {
            symbol: "BTC-USD".to_string(),
            notional_value: 300_000.0,
            margin_used: 50_000.0,
            leverage: 6.0,
            unrealized_pnl: -2000.0,
            liquidation_price: Some(35_000.0),
        });

        control.update_position(LeveragedPosition {
            symbol: "ETH-USD".to_string(),
            notional_value: 200_000.0,
            margin_used: 35_000.0,
            leverage: 5.7,
            unrealized_pnl: -500.0,
            liquidation_price: Some(2_800.0),
        });

        // Total margin: 85,000 = 85% utilization (margin call)
        let instructions = control.generate_deleverage_instructions();
        assert!(!instructions.is_empty());

        // Losing positions should be reduced first
        assert_eq!(instructions[0].symbol, "BTC-USD");
        assert_eq!(instructions[0].reason, DeleverageReason::MarginCall);
    }

    #[test]
    fn test_tier_switching() {
        let mut control = LeverageControl::new();

        assert_eq!(control.get_active_tier().name, "moderate");

        control.set_active_tier_by_name("conservative").unwrap();
        assert_eq!(control.get_active_tier().name, "conservative");
        assert_eq!(control.get_active_tier().max_leverage, 1.5);

        control.set_active_tier_by_name("aggressive").unwrap();
        assert_eq!(control.get_active_tier().name, "aggressive");
        assert_eq!(control.get_active_tier().max_leverage, 5.0);

        assert!(control.set_active_tier_by_name("invalid").is_err());
    }

    #[test]
    fn test_max_position_size() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        // With no positions, available margin is full equity
        let max_size = control.get_max_position_size("BTC-USD");
        // Max leverage 3.0 * 100,000 = 300,000
        assert!((max_size - 300_000.0).abs() < 0.01);

        // Add position using some margin
        control.update_position(LeveragedPosition {
            symbol: "ETH-USD".to_string(),
            notional_value: 100_000.0,
            margin_used: 20_000.0,
            leverage: 5.0,
            unrealized_pnl: 0.0,
            liquidation_price: None,
        });

        let max_size = control.get_max_position_size("BTC-USD");
        // Available: 80,000 * 3.0 = 240,000
        assert!((max_size - 240_000.0).abs() < 0.01);
    }

    #[test]
    fn test_remove_position() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        control.update_position(LeveragedPosition {
            symbol: "BTC-USD".to_string(),
            notional_value: 50_000.0,
            margin_used: 10_000.0,
            leverage: 5.0,
            unrealized_pnl: 0.0,
            liquidation_price: None,
        });

        assert_eq!(control.positions.len(), 1);
        assert!((control.total_margin_used - 10_000.0).abs() < 0.01);

        control.remove_position("BTC-USD");

        assert_eq!(control.positions.len(), 0);
        assert!(control.total_margin_used.abs() < 0.01);
    }

    #[test]
    fn test_stats() {
        let mut control = LeverageControl::new();
        control.update_equity(100_000.0);

        control.update_position(LeveragedPosition {
            symbol: "BTC-USD".to_string(),
            notional_value: 150_000.0,
            margin_used: 30_000.0,
            leverage: 5.0,
            unrealized_pnl: 1000.0,
            liquidation_price: None,
        });

        let stats = control.get_stats();
        assert_eq!(stats.position_count, 1);
        assert!((stats.total_notional - 150_000.0).abs() < 0.01);
        assert!((stats.account_leverage - 1.5).abs() < 0.01);
        assert!((stats.margin_utilization - 30.0).abs() < 0.01);
        assert_eq!(stats.active_tier, "moderate");
    }
}
