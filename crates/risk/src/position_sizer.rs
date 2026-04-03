//! Position Sizing and Risk Management
//!
//! This module handles position sizing calculations for different account sizes,
//! risk management, and profit allocation including the wife tax.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Position sizing calculator
pub struct PositionSizer {
    pub account_size: f64,
    pub max_risk_percent: f64,
    pub use_kelly: bool,
    pub kelly_fraction: f64,
}

impl PositionSizer {
    /// Create a new position sizer
    pub fn new(account_size: f64, max_risk_percent: f64) -> Self {
        Self {
            account_size,
            max_risk_percent,
            use_kelly: false,
            kelly_fraction: 0.25,
        }
    }

    /// Calculate position size based on risk amount
    ///
    /// # Arguments
    /// * `entry_price` - Entry price for the trade
    /// * `stop_loss` - Stop loss price
    /// * `risk_percent` - Risk percentage (default to max if None)
    ///
    /// # Returns
    /// Position size in USD
    pub fn calculate_position_size(
        &self,
        entry_price: f64,
        stop_loss: f64,
        risk_percent: Option<f64>,
    ) -> PositionSizeResult {
        let risk_pct = risk_percent.unwrap_or(self.max_risk_percent);
        let risk_amount = self.account_size * (risk_pct / 100.0);

        // Calculate stop loss distance in percentage
        let stop_distance_pct = ((entry_price - stop_loss).abs() / entry_price) * 100.0;

        // Position size = Risk Amount / Stop Distance %
        let position_size = if stop_distance_pct > 0.0 {
            risk_amount / (stop_distance_pct / 100.0)
        } else {
            0.0
        };

        // Calculate leverage needed (if position > account)
        let leverage = if position_size > self.account_size {
            (position_size / self.account_size).ceil() as i32
        } else {
            1
        };

        PositionSizeResult {
            position_size_usd: position_size,
            position_size_percent: (position_size / self.account_size) * 100.0,
            risk_amount,
            risk_percent: risk_pct,
            stop_loss_distance_pct: stop_distance_pct,
            leverage,
            contracts: position_size / entry_price,
        }
    }

    /// Calculate position size using Kelly Criterion
    ///
    /// Kelly % = W - [(1 - W) / R]
    /// Where W = Win rate, R = Average Win / Average Loss
    pub fn calculate_kelly_position(&self, win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
        if avg_loss == 0.0 {
            return self.max_risk_percent;
        }

        let r = avg_win / avg_loss;
        let kelly_pct = win_rate - ((1.0 - win_rate) / r);

        // Apply Kelly fraction for safety
        let adjusted_kelly = kelly_pct * self.kelly_fraction;

        // Clamp to max risk
        adjusted_kelly.max(0.0).min(self.max_risk_percent)
    }

    /// Calculate take profit levels based on risk:reward ratios
    pub fn calculate_take_profits(
        &self,
        entry_price: f64,
        stop_loss: f64,
        direction: Direction,
        rr_ratios: &[f64],
    ) -> Vec<f64> {
        let risk = (entry_price - stop_loss).abs();

        rr_ratios
            .iter()
            .map(|rr| match direction {
                Direction::Long => entry_price + (risk * rr),
                Direction::Short => entry_price - (risk * rr),
            })
            .collect()
    }

    /// Scale position size based on account size tiers
    pub fn scale_for_account_tier(&self) -> AccountTierInfo {
        let tier = match self.account_size as i64 {
            0..=5000 => AccountTier::Starter,
            5001..=10000 => AccountTier::Basic,
            10001..=25000 => AccountTier::Intermediate,
            25001..=50000 => AccountTier::Advanced,
            50001..=100000 => AccountTier::Professional,
            _ => AccountTier::Institutional,
        };

        let suggested_risk = match tier {
            AccountTier::Starter => 2.0, // More conservative for small accounts
            AccountTier::Basic => 2.5,
            AccountTier::Intermediate => 3.0, // Full 3% at this level
            AccountTier::Advanced => 3.0,
            AccountTier::Professional => 2.5, // Reduce risk as account grows
            AccountTier::Institutional => 2.0, // Very conservative for large accounts
        };

        let max_concurrent_trades = match tier {
            AccountTier::Starter => 1,
            AccountTier::Basic => 2,
            AccountTier::Intermediate => 3,
            AccountTier::Advanced => 4,
            AccountTier::Professional => 5,
            AccountTier::Institutional => 6,
        };

        AccountTierInfo {
            tier,
            account_size: self.account_size,
            suggested_risk_percent: suggested_risk,
            max_concurrent_trades,
        }
    }
}

/// Direction for TP calculation
#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Long,
    Short,
}

/// Position size calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizeResult {
    pub position_size_usd: f64,
    pub position_size_percent: f64,
    pub risk_amount: f64,
    pub risk_percent: f64,
    pub stop_loss_distance_pct: f64,
    pub leverage: i32,
    pub contracts: f64,
}

/// Account tier classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccountTier {
    Starter,       // $5K
    Basic,         // $10K
    Intermediate,  // $25K
    Advanced,      // $50K
    Professional,  // $100K
    Institutional, // $200K+
}

/// Account tier information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountTierInfo {
    pub tier: AccountTier,
    pub account_size: f64,
    pub suggested_risk_percent: f64,
    pub max_concurrent_trades: i32,
}

/// Profit allocation calculator
pub struct ProfitAllocator {
    pub wife_tax_percent: f64,
    pub personal_pay_percent: f64,
    pub crypto_exchange_percent: f64,
    pub hardware_wallet_percent: f64,
    pub expand_accounts_percent: f64,
}

impl Default for ProfitAllocator {
    fn default() -> Self {
        Self {
            wife_tax_percent: 1.0,
            personal_pay_percent: 30.0,
            crypto_exchange_percent: 30.0,
            hardware_wallet_percent: 20.0,
            expand_accounts_percent: 19.0,
        }
    }
}

impl ProfitAllocator {
    /// Allocate profits according to the strategy
    pub fn allocate(&self, profit: f64) -> ProfitAllocation {
        let wife_tax = profit * (self.wife_tax_percent / 100.0);
        let personal_pay = profit * (self.personal_pay_percent / 100.0);
        let crypto_exchanges = profit * (self.crypto_exchange_percent / 100.0);
        let hardware_wallet = profit * (self.hardware_wallet_percent / 100.0);
        let expand_accounts = profit * (self.expand_accounts_percent / 100.0);

        // Distribute crypto exchange funds across exchanges
        let per_exchange = crypto_exchanges / 3.0; // 3 exchanges

        let mut exchange_allocations = HashMap::new();
        exchange_allocations.insert("crypto_com".to_string(), per_exchange);
        exchange_allocations.insert("kraken".to_string(), per_exchange);
        exchange_allocations.insert("binance".to_string(), per_exchange);

        ProfitAllocation {
            total_profit: profit,
            wife_tax,
            personal_pay,
            crypto_exchanges: exchange_allocations,
            hardware_wallet,
            expand_accounts,
        }
    }

    /// Calculate when we can add another prop firm account
    pub fn calculate_account_scaling(
        &self,
        current_accounts: i32,
        total_balance: f64,
    ) -> ScalingRecommendation {
        // Conservative: Only scale when we have 2x the account fee saved
        let next_account_fee = match current_accounts {
            0..=2 => 599.0, // $10K two-step
            3..=5 => 999.0, // $25K two-step
            _ => 1399.0,    // $50K+ two-step
        };

        let required_buffer = next_account_fee * 2.0;
        let can_scale = total_balance >= required_buffer;

        ScalingRecommendation {
            current_accounts,
            can_add_account: can_scale,
            required_balance: required_buffer,
            current_balance: total_balance,
            recommended_account_size: if can_scale {
                match current_accounts {
                    0..=2 => 10000.0,
                    3..=5 => 25000.0,
                    _ => 50000.0,
                }
            } else {
                0.0
            },
        }
    }
}

/// Profit allocation breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfitAllocation {
    pub total_profit: f64,
    pub wife_tax: f64,
    pub personal_pay: f64,
    pub crypto_exchanges: HashMap<String, f64>,
    pub hardware_wallet: f64,
    pub expand_accounts: f64,
}

/// Account scaling recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingRecommendation {
    pub current_accounts: i32,
    pub can_add_account: bool,
    pub required_balance: f64,
    pub current_balance: f64,
    pub recommended_account_size: f64,
}

/// Multi-TP/SL strategy
#[allow(dead_code)]
pub struct MultiTpSlStrategy {
    pub tp_percentages: Vec<f64>, // % of position to close at each TP
    pub tp_rr_ratios: Vec<f64>,   // Risk:Reward ratios for each TP
    pub trailing_stop_after_tp: Option<i32>, // Start trailing after TP X
}

impl Default for MultiTpSlStrategy {
    fn default() -> Self {
        Self {
            tp_percentages: vec![25.0, 50.0, 75.0, 100.0],
            tp_rr_ratios: vec![1.0, 2.0, 3.0, 5.0],
            trailing_stop_after_tp: Some(1), // Start trailing after TP1
        }
    }
}

impl MultiTpSlStrategy {
    /// Calculate all TP levels and exit percentages
    #[allow(dead_code)]
    pub fn calculate_levels(
        &self,
        entry_price: f64,
        stop_loss: f64,
        direction: Direction,
    ) -> Vec<TpLevel> {
        let risk = (entry_price - stop_loss).abs();

        self.tp_rr_ratios
            .iter()
            .zip(&self.tp_percentages)
            .enumerate()
            .map(|(i, (rr, pct))| {
                let price = match direction {
                    Direction::Long => entry_price + (risk * rr),
                    Direction::Short => entry_price - (risk * rr),
                };

                TpLevel {
                    level: i + 1,
                    price,
                    close_percent: *pct,
                    rr_ratio: *rr,
                }
            })
            .collect()
    }

    /// Calculate trailing stop distance after TP hit
    #[allow(dead_code)]
    pub fn calculate_trailing_stop(&self, _entry_price: f64, atr: f64, tp_level: i32) -> f64 {
        // Tighten trailing stop as we hit more TPs
        let multiplier = match tp_level {
            1 => 1.5,
            2 => 1.0,
            3 => 0.75,
            _ => 0.5,
        };

        atr * multiplier
    }
}

/// Take profit level
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct TpLevel {
    pub level: usize,
    pub price: f64,
    pub close_percent: f64,
    pub rr_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_sizing() {
        let sizer = PositionSizer::new(5000.0, 3.0);
        let result = sizer.calculate_position_size(50000.0, 48500.0, None);

        // 3% of $5000 = $150 risk
        // Stop distance = 3%
        // Position = $150 / 0.03 = $5000
        assert!((result.position_size_usd - 5000.0).abs() < 1.0);
        assert_eq!(result.leverage, 1);
    }

    #[test]
    fn test_take_profit_calculation() {
        let sizer = PositionSizer::new(5000.0, 3.0);
        let tps = sizer.calculate_take_profits(50000.0, 48500.0, Direction::Long, &[1.0, 2.0, 3.0]);

        assert_eq!(tps.len(), 3);
        assert_eq!(tps[0], 51500.0); // Entry + 1R
        assert_eq!(tps[1], 53000.0); // Entry + 2R
        assert_eq!(tps[2], 54500.0); // Entry + 3R
    }

    #[test]
    fn test_profit_allocation() {
        let allocator = ProfitAllocator::default();
        let allocation = allocator.allocate(10000.0);

        assert_eq!(allocation.wife_tax, 100.0); // 1%
        assert_eq!(allocation.personal_pay, 3000.0); // 30%
        assert_eq!(allocation.hardware_wallet, 2000.0); // 20%
        assert_eq!(allocation.expand_accounts, 1900.0); // 19%

        let total_crypto = allocation.crypto_exchanges.values().sum::<f64>();
        assert!((total_crypto - 3000.0).abs() < 0.1); // 30%
    }

    #[test]
    fn test_account_tier_scaling() {
        let sizer = PositionSizer::new(5000.0, 3.0);
        let tier_info = sizer.scale_for_account_tier();
        assert_eq!(tier_info.tier, AccountTier::Starter);
        assert_eq!(tier_info.suggested_risk_percent, 2.0);

        let sizer = PositionSizer::new(50000.0, 3.0);
        let tier_info = sizer.scale_for_account_tier();
        assert_eq!(tier_info.tier, AccountTier::Advanced);
        assert_eq!(tier_info.max_concurrent_trades, 4);
    }

    #[test]
    fn test_multi_tp_strategy() {
        let strategy = MultiTpSlStrategy::default();
        let levels = strategy.calculate_levels(50000.0, 48500.0, Direction::Long);

        assert_eq!(levels.len(), 4);
        assert_eq!(levels[0].close_percent, 25.0);
        assert_eq!(levels[0].price, 51500.0);
        assert_eq!(levels[3].close_percent, 100.0);
    }
}
