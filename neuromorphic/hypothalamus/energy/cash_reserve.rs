//! Cash Reserve Management
//!
//! Maintains cash reserves for the trading system, implementing a tiered
//! reserve structure with emergency buffers, dynamic adjustment based on
//! market conditions, and reserve-to-exposure ratio management.
//!
//! Part of the Hypothalamus region (homeostatic regulation)
//! Component: energy

use crate::common::{Error, Result};
use std::collections::VecDeque;

/// Reserve tier configuration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReserveTier {
    /// Minimum operating reserve - never touched except for emergencies
    Emergency,
    /// Safety buffer for unexpected drawdowns
    Safety,
    /// Operational reserve for normal trading
    Operational,
    /// Excess cash available for deployment
    Deployable,
}

/// Reserve status indicating current health
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveStatus {
    /// Reserves are critically low
    Critical,
    /// Reserves are below target
    Low,
    /// Reserves are at healthy levels
    Healthy,
    /// Reserves are above target (excess cash)
    Excess,
}

/// Configuration for cash reserve management
#[derive(Debug, Clone)]
pub struct CashReserveConfig {
    /// Emergency reserve as percentage of total capital (default: 5%)
    pub emergency_reserve_pct: f64,
    /// Safety reserve as percentage of total capital (default: 10%)
    pub safety_reserve_pct: f64,
    /// Operational reserve as percentage of total capital (default: 15%)
    pub operational_reserve_pct: f64,
    /// Target reserve-to-exposure ratio (default: 0.25)
    pub target_reserve_ratio: f64,
    /// Minimum reserve-to-exposure ratio before triggering alerts (default: 0.15)
    pub min_reserve_ratio: f64,
    /// Maximum reserve-to-exposure ratio before suggesting deployment (default: 0.40)
    pub max_reserve_ratio: f64,
    /// Window size for tracking reserve history
    pub history_window: usize,
    /// Enable dynamic reserve adjustment based on volatility
    pub dynamic_adjustment: bool,
    /// Volatility multiplier for dynamic adjustment (default: 1.5)
    pub volatility_multiplier: f64,
}

impl Default for CashReserveConfig {
    fn default() -> Self {
        Self {
            emergency_reserve_pct: 0.05,
            safety_reserve_pct: 0.10,
            operational_reserve_pct: 0.15,
            target_reserve_ratio: 0.25,
            min_reserve_ratio: 0.15,
            max_reserve_ratio: 0.40,
            history_window: 100,
            dynamic_adjustment: true,
            volatility_multiplier: 1.5,
        }
    }
}

/// Reserve level snapshot
#[derive(Debug, Clone)]
pub struct ReserveSnapshot {
    pub timestamp: u64,
    pub total_capital: f64,
    pub cash_balance: f64,
    pub total_exposure: f64,
    pub reserve_ratio: f64,
    pub status: ReserveStatus,
}

/// Recommendation for reserve action
#[derive(Debug, Clone)]
pub struct ReserveRecommendation {
    /// Recommended action
    pub action: ReserveAction,
    /// Amount involved in the action
    pub amount: f64,
    /// Priority of the recommendation (0.0 - 1.0)
    pub priority: f64,
    /// Reason for the recommendation
    pub reason: String,
}

/// Possible reserve actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveAction {
    /// No action needed
    None,
    /// Reduce exposure to build reserves
    ReduceExposure,
    /// Deploy excess reserves
    DeployReserves,
    /// Emergency: halt new positions
    EmergencyHalt,
    /// Rebalance between tiers
    Rebalance,
}

/// Cash reserve manager
pub struct CashReserve {
    config: CashReserveConfig,
    /// Current cash balance
    cash_balance: f64,
    /// Total capital (cash + positions at cost)
    total_capital: f64,
    /// Current total exposure (sum of position values)
    total_exposure: f64,
    /// Current market volatility estimate (0.0 - 1.0)
    current_volatility: f64,
    /// Historical reserve snapshots
    history: VecDeque<ReserveSnapshot>,
    /// Timestamp counter for snapshots
    timestamp_counter: u64,
    /// Emergency reserve amount (absolute)
    emergency_reserve: f64,
    /// Safety reserve amount (absolute)
    safety_reserve: f64,
    /// Operational reserve amount (absolute)
    operational_reserve: f64,
}

impl Default for CashReserve {
    fn default() -> Self {
        Self::new()
    }
}

impl CashReserve {
    /// Create a new cash reserve manager with default configuration
    pub fn new() -> Self {
        Self::with_config(CashReserveConfig::default())
    }

    /// Create a new cash reserve manager with custom configuration
    pub fn with_config(config: CashReserveConfig) -> Self {
        Self {
            config,
            cash_balance: 0.0,
            total_capital: 0.0,
            total_exposure: 0.0,
            current_volatility: 0.2, // Default moderate volatility
            history: VecDeque::new(),
            timestamp_counter: 0,
            emergency_reserve: 0.0,
            safety_reserve: 0.0,
            operational_reserve: 0.0,
        }
    }

    /// Initialize with starting capital
    pub fn initialize(&mut self, starting_capital: f64) -> Result<()> {
        if starting_capital <= 0.0 {
            return Err(Error::InvalidInput(
                "Starting capital must be positive".to_string(),
            ));
        }

        self.total_capital = starting_capital;
        self.cash_balance = starting_capital;
        self.total_exposure = 0.0;
        self.recalculate_tiers();
        self.record_snapshot();

        Ok(())
    }

    /// Update current state with new values
    pub fn update(
        &mut self,
        cash_balance: f64,
        total_exposure: f64,
        total_capital: f64,
    ) -> Result<()> {
        if cash_balance < 0.0 {
            return Err(Error::InvalidInput(
                "Cash balance cannot be negative".to_string(),
            ));
        }

        self.cash_balance = cash_balance;
        self.total_exposure = total_exposure;
        self.total_capital = total_capital;
        self.recalculate_tiers();
        self.record_snapshot();

        Ok(())
    }

    /// Update volatility estimate for dynamic adjustment
    pub fn update_volatility(&mut self, volatility: f64) {
        self.current_volatility = volatility.clamp(0.0, 1.0);
        if self.config.dynamic_adjustment {
            self.recalculate_tiers();
        }
    }

    /// Get current reserve ratio (cash / exposure)
    pub fn reserve_ratio(&self) -> f64 {
        if self.total_exposure <= 0.0 {
            return f64::INFINITY;
        }
        self.cash_balance / self.total_exposure
    }

    /// Get current reserve status
    pub fn status(&self) -> ReserveStatus {
        let ratio = self.reserve_ratio();
        let effective_min = self.effective_min_ratio();
        let effective_max = self.effective_max_ratio();

        if self.cash_balance <= self.emergency_reserve {
            ReserveStatus::Critical
        } else if ratio < effective_min {
            ReserveStatus::Low
        } else if ratio > effective_max {
            ReserveStatus::Excess
        } else {
            ReserveStatus::Healthy
        }
    }

    /// Get the effective minimum reserve ratio (adjusted for volatility)
    fn effective_min_ratio(&self) -> f64 {
        if self.config.dynamic_adjustment {
            self.config.min_reserve_ratio
                * (1.0 + self.current_volatility * self.config.volatility_multiplier)
        } else {
            self.config.min_reserve_ratio
        }
    }

    /// Get the effective maximum reserve ratio (adjusted for volatility)
    fn effective_max_ratio(&self) -> f64 {
        if self.config.dynamic_adjustment {
            self.config.max_reserve_ratio
                * (1.0 + self.current_volatility * self.config.volatility_multiplier)
        } else {
            self.config.max_reserve_ratio
        }
    }

    /// Get available cash for new positions (above operational reserve)
    pub fn available_for_deployment(&self) -> f64 {
        let minimum_reserve = self.emergency_reserve + self.safety_reserve + self.operational_reserve;
        (self.cash_balance - minimum_reserve).max(0.0)
    }

    /// Get available cash in a specific tier
    pub fn available_in_tier(&self, tier: ReserveTier) -> f64 {
        match tier {
            ReserveTier::Emergency => self.cash_balance.min(self.emergency_reserve),
            ReserveTier::Safety => {
                let above_emergency = (self.cash_balance - self.emergency_reserve).max(0.0);
                above_emergency.min(self.safety_reserve)
            }
            ReserveTier::Operational => {
                let above_safety =
                    (self.cash_balance - self.emergency_reserve - self.safety_reserve).max(0.0);
                above_safety.min(self.operational_reserve)
            }
            ReserveTier::Deployable => self.available_for_deployment(),
        }
    }

    /// Check if a withdrawal amount is safe
    pub fn can_withdraw(&self, amount: f64) -> bool {
        if amount <= 0.0 {
            return true;
        }
        self.cash_balance - amount >= self.emergency_reserve + self.safety_reserve
    }

    /// Check if a withdrawal amount would breach emergency reserves
    pub fn would_breach_emergency(&self, amount: f64) -> bool {
        self.cash_balance - amount < self.emergency_reserve
    }

    /// Get recommendation based on current state
    pub fn get_recommendation(&self) -> ReserveRecommendation {
        let status = self.status();
        let ratio = self.reserve_ratio();
        let target = self.config.target_reserve_ratio;

        match status {
            ReserveStatus::Critical => {
                let deficit = self.emergency_reserve - self.cash_balance;
                ReserveRecommendation {
                    action: ReserveAction::EmergencyHalt,
                    amount: deficit.max(0.0),
                    priority: 1.0,
                    reason: "Cash reserves critically low - halt all new positions".to_string(),
                }
            }
            ReserveStatus::Low => {
                // Calculate how much exposure to reduce to reach target ratio
                let target_cash = self.total_exposure * target;
                let needed = target_cash - self.cash_balance;
                let priority = 1.0 - (ratio / self.effective_min_ratio()).min(1.0);

                ReserveRecommendation {
                    action: ReserveAction::ReduceExposure,
                    amount: needed.max(0.0),
                    priority: 0.5 + priority * 0.5,
                    reason: format!(
                        "Reserve ratio {:.1}% below minimum {:.1}%",
                        ratio * 100.0,
                        self.effective_min_ratio() * 100.0
                    ),
                }
            }
            ReserveStatus::Excess => {
                let excess = self.cash_balance - self.total_exposure * target;
                let priority = ((ratio - self.effective_max_ratio())
                    / (1.0 - self.effective_max_ratio()))
                    .min(1.0);

                ReserveRecommendation {
                    action: ReserveAction::DeployReserves,
                    amount: excess.max(0.0),
                    priority: priority * 0.5, // Lower priority than safety actions
                    reason: format!(
                        "Reserve ratio {:.1}% above target {:.1}%",
                        ratio * 100.0,
                        target * 100.0
                    ),
                }
            }
            ReserveStatus::Healthy => ReserveRecommendation {
                action: ReserveAction::None,
                amount: 0.0,
                priority: 0.0,
                reason: "Reserves at healthy levels".to_string(),
            },
        }
    }

    /// Simulate the effect of a position change on reserves
    pub fn simulate_position_change(
        &self,
        cash_delta: f64,
        exposure_delta: f64,
    ) -> (ReserveStatus, f64) {
        let new_cash = self.cash_balance + cash_delta;
        let new_exposure = self.total_exposure + exposure_delta;

        let new_ratio = if new_exposure <= 0.0 {
            f64::INFINITY
        } else {
            new_cash / new_exposure
        };

        let new_status = if new_cash <= self.emergency_reserve {
            ReserveStatus::Critical
        } else if new_ratio < self.effective_min_ratio() {
            ReserveStatus::Low
        } else if new_ratio > self.effective_max_ratio() {
            ReserveStatus::Excess
        } else {
            ReserveStatus::Healthy
        };

        (new_status, new_ratio)
    }

    /// Get reserve utilization metrics
    pub fn utilization(&self) -> ReserveUtilization {
        let total_reserve_target =
            self.emergency_reserve + self.safety_reserve + self.operational_reserve;

        ReserveUtilization {
            cash_balance: self.cash_balance,
            total_capital: self.total_capital,
            total_exposure: self.total_exposure,
            reserve_ratio: self.reserve_ratio(),
            emergency_utilization: if self.emergency_reserve > 0.0 {
                (self.cash_balance.min(self.emergency_reserve) / self.emergency_reserve).min(1.0)
            } else {
                1.0
            },
            safety_utilization: if self.safety_reserve > 0.0 {
                (self.available_in_tier(ReserveTier::Safety) / self.safety_reserve).min(1.0)
            } else {
                1.0
            },
            operational_utilization: if self.operational_reserve > 0.0 {
                (self.available_in_tier(ReserveTier::Operational) / self.operational_reserve)
                    .min(1.0)
            } else {
                1.0
            },
            deployable_amount: self.available_for_deployment(),
            buffer_from_emergency: (self.cash_balance - self.emergency_reserve).max(0.0),
            reserve_coverage: if total_reserve_target > 0.0 {
                self.cash_balance / total_reserve_target
            } else {
                f64::INFINITY
            },
        }
    }

    /// Get recent history
    pub fn recent_history(&self, count: usize) -> Vec<&ReserveSnapshot> {
        self.history.iter().rev().take(count).collect()
    }

    /// Main processing function - analyzes state and returns recommendation
    pub fn process(&self) -> Result<ReserveRecommendation> {
        Ok(self.get_recommendation())
    }

    /// Recalculate tier amounts based on current capital and volatility
    fn recalculate_tiers(&mut self) {
        let vol_adjustment = if self.config.dynamic_adjustment {
            1.0 + self.current_volatility * self.config.volatility_multiplier
        } else {
            1.0
        };

        self.emergency_reserve = self.total_capital * self.config.emergency_reserve_pct * vol_adjustment;
        self.safety_reserve = self.total_capital * self.config.safety_reserve_pct * vol_adjustment;
        self.operational_reserve =
            self.total_capital * self.config.operational_reserve_pct * vol_adjustment;
    }

    /// Record current state as a snapshot
    fn record_snapshot(&mut self) {
        let snapshot = ReserveSnapshot {
            timestamp: self.timestamp_counter,
            total_capital: self.total_capital,
            cash_balance: self.cash_balance,
            total_exposure: self.total_exposure,
            reserve_ratio: self.reserve_ratio(),
            status: self.status(),
        };

        self.history.push_back(snapshot);
        self.timestamp_counter += 1;

        // Trim history if needed
        while self.history.len() > self.config.history_window {
            self.history.pop_front();
        }
    }
}

/// Reserve utilization metrics
#[derive(Debug, Clone)]
pub struct ReserveUtilization {
    pub cash_balance: f64,
    pub total_capital: f64,
    pub total_exposure: f64,
    pub reserve_ratio: f64,
    pub emergency_utilization: f64,
    pub safety_utilization: f64,
    pub operational_utilization: f64,
    pub deployable_amount: f64,
    pub buffer_from_emergency: f64,
    pub reserve_coverage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = CashReserve::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_initialization() {
        let mut reserve = CashReserve::new();
        assert!(reserve.initialize(100_000.0).is_ok());
        assert_eq!(reserve.cash_balance, 100_000.0);
        assert_eq!(reserve.total_capital, 100_000.0);
        assert_eq!(reserve.total_exposure, 0.0);
    }

    #[test]
    fn test_initialization_invalid() {
        let mut reserve = CashReserve::new();
        assert!(reserve.initialize(-1000.0).is_err());
        assert!(reserve.initialize(0.0).is_err());
    }

    #[test]
    fn test_reserve_status_healthy() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        // 30k cash, 100k exposure = 30% reserve ratio (within healthy range)
        reserve.update(30_000.0, 100_000.0, 100_000.0).unwrap();
        assert_eq!(reserve.status(), ReserveStatus::Healthy);
    }

    #[test]
    fn test_reserve_status_low() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        // 10k cash, 100k exposure = 10% reserve ratio (below min of 15%)
        reserve.update(10_000.0, 100_000.0, 100_000.0).unwrap();
        assert_eq!(reserve.status(), ReserveStatus::Low);
    }

    #[test]
    fn test_reserve_status_critical() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        // 3k cash (below emergency reserve of 5k)
        reserve.update(3_000.0, 97_000.0, 100_000.0).unwrap();
        assert_eq!(reserve.status(), ReserveStatus::Critical);
    }

    #[test]
    fn test_reserve_status_excess() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        // 60k cash, 50k exposure = 120% reserve ratio (above max)
        reserve.update(60_000.0, 50_000.0, 100_000.0).unwrap();
        assert_eq!(reserve.status(), ReserveStatus::Excess);
    }

    #[test]
    fn test_available_for_deployment() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        // With default config: emergency=5%, safety=10%, operational=15% = 30% total
        // So with 50k cash and 100k capital, deployable = 50k - 30k = 20k
        reserve.update(50_000.0, 50_000.0, 100_000.0).unwrap();
        let deployable = reserve.available_for_deployment();
        assert!((deployable - 20_000.0).abs() < 100.0); // Allow for volatility adjustment
    }

    #[test]
    fn test_can_withdraw() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(30_000.0, 70_000.0, 100_000.0).unwrap();

        // Should be able to withdraw small amount
        assert!(reserve.can_withdraw(1_000.0));

        // Should not be able to withdraw most of cash (would breach safety)
        assert!(!reserve.can_withdraw(20_000.0));
    }

    #[test]
    fn test_would_breach_emergency() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(10_000.0, 90_000.0, 100_000.0).unwrap();

        // Emergency reserve is ~5k, so withdrawing 6k would breach
        assert!(reserve.would_breach_emergency(6_000.0));
        assert!(!reserve.would_breach_emergency(3_000.0));
    }

    #[test]
    fn test_recommendation_critical() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(3_000.0, 97_000.0, 100_000.0).unwrap();

        let rec = reserve.get_recommendation();
        assert_eq!(rec.action, ReserveAction::EmergencyHalt);
        assert_eq!(rec.priority, 1.0);
    }

    #[test]
    fn test_recommendation_healthy() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(25_000.0, 75_000.0, 100_000.0).unwrap();

        let rec = reserve.get_recommendation();
        assert_eq!(rec.action, ReserveAction::None);
    }

    #[test]
    fn test_volatility_adjustment() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(30_000.0, 70_000.0, 100_000.0).unwrap();

        let before_emergency = reserve.emergency_reserve;

        // Increase volatility
        reserve.update_volatility(0.8);

        // Emergency reserve should increase with higher volatility
        assert!(reserve.emergency_reserve > before_emergency);
    }

    #[test]
    fn test_simulate_position_change() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(30_000.0, 70_000.0, 100_000.0).unwrap();

        // Simulate opening a large position
        let (status, ratio) = reserve.simulate_position_change(-20_000.0, 20_000.0);
        // New: 10k cash, 90k exposure = 11% ratio = Low
        assert_eq!(status, ReserveStatus::Low);
        assert!((ratio - 0.111).abs() < 0.01);
    }

    #[test]
    fn test_tier_availability() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(50_000.0, 50_000.0, 100_000.0).unwrap();

        // Emergency should be fully funded
        assert!(reserve.available_in_tier(ReserveTier::Emergency) > 0.0);

        // Safety should be fully funded
        assert!(reserve.available_in_tier(ReserveTier::Safety) > 0.0);

        // Operational should be fully funded
        assert!(reserve.available_in_tier(ReserveTier::Operational) > 0.0);

        // Should have deployable funds
        assert!(reserve.available_in_tier(ReserveTier::Deployable) > 0.0);
    }

    #[test]
    fn test_utilization_metrics() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        reserve.update(40_000.0, 60_000.0, 100_000.0).unwrap();

        let util = reserve.utilization();
        assert_eq!(util.cash_balance, 40_000.0);
        assert_eq!(util.total_exposure, 60_000.0);
        assert!((util.reserve_ratio - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_history_tracking() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();

        // Make several updates
        for i in 0..5 {
            reserve
                .update(
                    50_000.0 - (i as f64 * 5_000.0),
                    50_000.0 + (i as f64 * 5_000.0),
                    100_000.0,
                )
                .unwrap();
        }

        let history = reserve.recent_history(3);
        assert_eq!(history.len(), 3);
    }

    #[test]
    fn test_no_exposure_handling() {
        let mut reserve = CashReserve::new();
        reserve.initialize(100_000.0).unwrap();
        // No exposure = infinite reserve ratio = excess
        assert_eq!(reserve.reserve_ratio(), f64::INFINITY);
        assert_eq!(reserve.status(), ReserveStatus::Excess);
    }

    #[test]
    fn test_custom_config() {
        let config = CashReserveConfig {
            emergency_reserve_pct: 0.10,
            safety_reserve_pct: 0.15,
            operational_reserve_pct: 0.20,
            target_reserve_ratio: 0.30,
            min_reserve_ratio: 0.20,
            max_reserve_ratio: 0.50,
            history_window: 50,
            dynamic_adjustment: false,
            volatility_multiplier: 2.0,
        };

        let mut reserve = CashReserve::with_config(config);
        reserve.initialize(100_000.0).unwrap();

        // Emergency should be 10% of 100k = 10k
        assert!((reserve.emergency_reserve - 10_000.0).abs() < 1.0);
    }
}
