//! HyroTrader Compliance Module - The Sheriff

pub mod wash_sale;

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};

// Re-exports
pub use wash_sale::{
    Trade, TradeAction, WashSaleDetector, WashSaleError, WashSaleResult, WashSaleStats,
};

#[derive(Debug, Clone)]
pub struct HyroTraderRules {
    pub initial_balance: f64,
    pub daily_loss_limit_pct: f64,
    pub max_loss_pct: f64,
    pub min_trading_days: usize,
    pub profit_target_pct: f64,
}

impl HyroTraderRules {
    pub fn one_step_10k() -> Self {
        Self {
            initial_balance: 10_000.0,
            daily_loss_limit_pct: 5.0,
            max_loss_pct: 10.0,
            min_trading_days: 5,
            profit_target_pct: 10.0,
        }
    }
}

pub struct ComplianceSheriff {
    rules: HyroTraderRules,
    sod_balance: f64,
    #[allow(dead_code)]
    sod_timestamp: DateTime<Utc>,
}

impl ComplianceSheriff {
    pub fn new(rules: HyroTraderRules, current_balance: f64) -> Self {
        Self {
            rules,
            sod_balance: current_balance,
            sod_timestamp: Utc::now(),
        }
    }

    pub fn validate_order(
        &self,
        current_equity: f64,
        _order_risk: f64,
        stop_loss: Option<f64>,
    ) -> Result<()> {
        if stop_loss.is_none() {
            bail!("❌ Sheriff REJECT: Stop loss is MANDATORY");
        }

        let daily_loss = self.sod_balance - current_equity;
        let max_daily_loss = self.rules.initial_balance * (self.rules.daily_loss_limit_pct / 100.0);

        if daily_loss >= max_daily_loss {
            bail!("❌ Sheriff REJECT: Daily loss limit BREACHED");
        }

        Ok(())
    }
}
