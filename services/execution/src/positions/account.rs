//! Account Balance Management
//!
//! Tracks account balances, margin requirements, and risk metrics
//! across exchanges with real-time updates.

use crate::error::Result;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Account balance for a single currency on an exchange
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    /// Currency/coin symbol (e.g., "USDT", "BTC")
    pub currency: String,

    /// Exchange name
    pub exchange: String,

    /// Total balance (available + locked)
    pub total: Decimal,

    /// Available balance (can be used for trading)
    pub available: Decimal,

    /// Locked balance (in open orders, margin)
    pub locked: Decimal,

    /// Balance in margin/positions
    pub in_margin: Decimal,

    /// Equity (total + unrealized P&L)
    pub equity: Decimal,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl Balance {
    /// Create a new balance
    pub fn new(currency: String, exchange: String) -> Self {
        Self {
            currency,
            exchange,
            total: Decimal::ZERO,
            available: Decimal::ZERO,
            locked: Decimal::ZERO,
            in_margin: Decimal::ZERO,
            equity: Decimal::ZERO,
            updated_at: chrono::Utc::now(),
        }
    }

    /// Update balance values
    pub fn update(&mut self, total: Decimal, available: Decimal, locked: Decimal) {
        self.total = total;
        self.available = available;
        self.locked = locked;
        self.updated_at = chrono::Utc::now();
    }

    /// Set equity (includes unrealized P&L)
    pub fn set_equity(&mut self, equity: Decimal) {
        self.equity = equity;
        self.updated_at = chrono::Utc::now();
    }

    /// Get unrealized P&L
    pub fn unrealized_pnl(&self) -> Decimal {
        self.equity - self.total
    }
}

/// Margin account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarginAccount {
    /// Exchange name
    pub exchange: String,

    /// Account type (e.g., "UNIFIED", "CONTRACT", "SPOT")
    pub account_type: String,

    /// Total equity (wallet balance + unrealized P&L)
    pub total_equity: Decimal,

    /// Total wallet balance
    pub total_wallet_balance: Decimal,

    /// Total available balance
    pub total_available_balance: Decimal,

    /// Total margin balance
    pub total_margin_balance: Decimal,

    /// Total unrealized P&L
    pub total_unrealized_pnl: Decimal,

    /// Total initial margin (required for positions)
    pub total_initial_margin: Decimal,

    /// Total maintenance margin (required to avoid liquidation)
    pub total_maintenance_margin: Decimal,

    /// Margin ratio (equity / initial_margin)
    pub margin_ratio: Decimal,

    /// Account health ratio (for risk assessment)
    pub health_ratio: Decimal,

    /// Balances by currency
    pub balances: HashMap<String, Balance>,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl MarginAccount {
    /// Create a new margin account
    pub fn new(exchange: String, account_type: String) -> Self {
        Self {
            exchange,
            account_type,
            total_equity: Decimal::ZERO,
            total_wallet_balance: Decimal::ZERO,
            total_available_balance: Decimal::ZERO,
            total_margin_balance: Decimal::ZERO,
            total_unrealized_pnl: Decimal::ZERO,
            total_initial_margin: Decimal::ZERO,
            total_maintenance_margin: Decimal::ZERO,
            margin_ratio: Decimal::ZERO,
            health_ratio: Decimal::ZERO,
            balances: HashMap::new(),
            updated_at: chrono::Utc::now(),
        }
    }

    /// Update margin account totals
    #[allow(clippy::too_many_arguments)]
    pub fn update_totals(
        &mut self,
        total_equity: Decimal,
        total_wallet_balance: Decimal,
        total_available_balance: Decimal,
        total_margin_balance: Decimal,
        total_unrealized_pnl: Decimal,
        total_initial_margin: Decimal,
        total_maintenance_margin: Decimal,
    ) {
        self.total_equity = total_equity;
        self.total_wallet_balance = total_wallet_balance;
        self.total_available_balance = total_available_balance;
        self.total_margin_balance = total_margin_balance;
        self.total_unrealized_pnl = total_unrealized_pnl;
        self.total_initial_margin = total_initial_margin;
        self.total_maintenance_margin = total_maintenance_margin;

        // Calculate margin ratio (higher is better)
        if self.total_initial_margin > Decimal::ZERO {
            self.margin_ratio = self.total_equity / self.total_initial_margin;
        } else {
            self.margin_ratio = Decimal::MAX;
        }

        // Calculate health ratio (distance from liquidation)
        if self.total_maintenance_margin > Decimal::ZERO {
            self.health_ratio = self.total_equity / self.total_maintenance_margin;
        } else {
            self.health_ratio = Decimal::MAX;
        }

        self.updated_at = chrono::Utc::now();
    }

    /// Update a specific currency balance
    pub fn update_balance(&mut self, currency: String, balance: Balance) {
        self.balances.insert(currency, balance);
        self.updated_at = chrono::Utc::now();
    }

    /// Get balance for a currency
    pub fn get_balance(&self, currency: &str) -> Option<&Balance> {
        self.balances.get(currency)
    }

    /// Check if account is healthy (not near liquidation)
    pub fn is_healthy(&self) -> bool {
        // Health ratio > 1.2 is considered safe
        self.health_ratio > Decimal::from_str_exact("1.2").unwrap()
    }

    /// Check if account is at risk of liquidation
    pub fn is_at_risk(&self) -> bool {
        // Health ratio < 1.1 is considered risky
        self.health_ratio < Decimal::from_str_exact("1.1").unwrap()
    }

    /// Get available buying power
    pub fn buying_power(&self, leverage: Decimal) -> Decimal {
        self.total_available_balance * leverage
    }
}

/// Account manager tracking balances across all exchanges
pub struct AccountManager {
    /// Margin accounts by exchange
    accounts: Arc<RwLock<HashMap<String, MarginAccount>>>,

    /// Global account statistics
    stats: Arc<RwLock<AccountStats>>,
}

/// Global account statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccountStats {
    /// Total equity across all exchanges
    pub total_equity: Decimal,

    /// Total wallet balance across all exchanges
    pub total_wallet_balance: Decimal,

    /// Total available balance
    pub total_available_balance: Decimal,

    /// Total unrealized P&L
    pub total_unrealized_pnl: Decimal,

    /// Total initial margin
    pub total_initial_margin: Decimal,

    /// Total maintenance margin
    pub total_maintenance_margin: Decimal,

    /// Global margin ratio
    pub global_margin_ratio: Decimal,

    /// Global health ratio
    pub global_health_ratio: Decimal,

    /// Number of exchanges
    pub num_exchanges: usize,

    /// Number of currencies
    pub num_currencies: usize,

    /// Last update timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl AccountManager {
    /// Create a new account manager
    pub fn new() -> Self {
        Self {
            accounts: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AccountStats::default())),
        }
    }

    /// Get or create a margin account
    pub async fn get_account(&self, exchange: &str) -> Option<MarginAccount> {
        let accounts = self.accounts.read().await;
        accounts.get(exchange).cloned()
    }

    /// Get all accounts
    pub async fn get_all_accounts(&self) -> Vec<MarginAccount> {
        let accounts = self.accounts.read().await;
        accounts.values().cloned().collect()
    }

    /// Update a margin account
    pub async fn update_account(&self, account: MarginAccount) -> Result<()> {
        let exchange = account.exchange.clone();

        let mut accounts = self.accounts.write().await;
        accounts.insert(exchange.clone(), account);
        drop(accounts);

        // Update global stats
        self.update_stats().await;

        info!("Account updated: {}", exchange);
        Ok(())
    }

    /// Update a balance for an exchange
    pub async fn update_balance(
        &self,
        exchange: &str,
        currency: String,
        total: Decimal,
        available: Decimal,
        locked: Decimal,
    ) -> Result<()> {
        let mut accounts = self.accounts.write().await;

        let account = accounts
            .entry(exchange.to_string())
            .or_insert_with(|| MarginAccount::new(exchange.to_string(), "UNIFIED".to_string()));

        let mut balance = Balance::new(currency.clone(), exchange.to_string());
        balance.update(total, available, locked);
        account.update_balance(currency.clone(), balance);

        drop(accounts);

        self.update_stats().await;

        debug!("Balance updated: {} {} = {}", exchange, currency, total);
        Ok(())
    }

    /// Update margin metrics for an exchange
    #[allow(clippy::too_many_arguments)]
    pub async fn update_margin_metrics(
        &self,
        exchange: &str,
        total_equity: Decimal,
        total_wallet_balance: Decimal,
        total_available_balance: Decimal,
        total_margin_balance: Decimal,
        total_unrealized_pnl: Decimal,
        total_initial_margin: Decimal,
        total_maintenance_margin: Decimal,
    ) -> Result<()> {
        let health_ratio = {
            let mut accounts = self.accounts.write().await;

            let account = accounts
                .entry(exchange.to_string())
                .or_insert_with(|| MarginAccount::new(exchange.to_string(), "UNIFIED".to_string()));

            account.update_totals(
                total_equity,
                total_wallet_balance,
                total_available_balance,
                total_margin_balance,
                total_unrealized_pnl,
                total_initial_margin,
                total_maintenance_margin,
            );

            account.health_ratio
        };

        self.update_stats().await;

        info!(
            "Margin metrics updated: {} (equity: {}, health: {})",
            exchange, total_equity, health_ratio
        );

        Ok(())
    }

    /// Update global statistics
    async fn update_stats(&self) {
        let accounts = self.accounts.read().await;

        let mut stats = AccountStats {
            total_equity: Decimal::ZERO,
            total_wallet_balance: Decimal::ZERO,
            total_available_balance: Decimal::ZERO,
            total_unrealized_pnl: Decimal::ZERO,
            total_initial_margin: Decimal::ZERO,
            total_maintenance_margin: Decimal::ZERO,
            global_margin_ratio: Decimal::ZERO,
            global_health_ratio: Decimal::ZERO,
            num_exchanges: accounts.len(),
            num_currencies: 0,
            updated_at: chrono::Utc::now(),
        };

        let mut currencies = std::collections::HashSet::new();

        for account in accounts.values() {
            stats.total_equity += account.total_equity;
            stats.total_wallet_balance += account.total_wallet_balance;
            stats.total_available_balance += account.total_available_balance;
            stats.total_unrealized_pnl += account.total_unrealized_pnl;
            stats.total_initial_margin += account.total_initial_margin;
            stats.total_maintenance_margin += account.total_maintenance_margin;

            for currency in account.balances.keys() {
                currencies.insert(currency.clone());
            }
        }

        stats.num_currencies = currencies.len();

        // Calculate global ratios
        if stats.total_initial_margin > Decimal::ZERO {
            stats.global_margin_ratio = stats.total_equity / stats.total_initial_margin;
        } else {
            stats.global_margin_ratio = Decimal::MAX;
        }

        if stats.total_maintenance_margin > Decimal::ZERO {
            stats.global_health_ratio = stats.total_equity / stats.total_maintenance_margin;
        } else {
            stats.global_health_ratio = Decimal::MAX;
        }

        *self.stats.write().await = stats;
    }

    /// Get global statistics
    pub async fn get_stats(&self) -> AccountStats {
        self.stats.read().await.clone()
    }

    /// Check if any account is at risk
    pub async fn check_risk(&self) -> Vec<String> {
        let accounts = self.accounts.read().await;
        let mut at_risk = Vec::new();

        for (exchange, account) in accounts.iter() {
            if account.is_at_risk() {
                at_risk.push(format!(
                    "{} (health ratio: {:.2})",
                    exchange, account.health_ratio
                ));
                warn!(
                    "Account at risk: {} - health ratio: {}",
                    exchange, account.health_ratio
                );
            }
        }

        at_risk
    }

    /// Get total available buying power (with leverage)
    pub async fn total_buying_power(&self, leverage: Decimal) -> Decimal {
        let accounts = self.accounts.read().await;
        accounts.values().map(|a| a.buying_power(leverage)).sum()
    }

    /// Clear all accounts (for testing)
    pub async fn clear_all(&self) {
        let mut accounts = self.accounts.write().await;
        accounts.clear();
        drop(accounts);

        *self.stats.write().await = AccountStats::default();
        info!("All accounts cleared");
    }
}

impl Default for AccountManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balance_creation() {
        let balance = Balance::new("USDT".to_string(), "bybit".to_string());
        assert_eq!(balance.currency, "USDT");
        assert_eq!(balance.total, Decimal::ZERO);
    }

    #[test]
    fn test_balance_update() {
        let mut balance = Balance::new("USDT".to_string(), "bybit".to_string());
        balance.update(
            Decimal::from(10000),
            Decimal::from(9000),
            Decimal::from(1000),
        );

        assert_eq!(balance.total, Decimal::from(10000));
        assert_eq!(balance.available, Decimal::from(9000));
        assert_eq!(balance.locked, Decimal::from(1000));
    }

    #[test]
    fn test_margin_account_creation() {
        let account = MarginAccount::new("bybit".to_string(), "UNIFIED".to_string());
        assert_eq!(account.exchange, "bybit");
        assert_eq!(account.total_equity, Decimal::ZERO);
    }

    #[test]
    fn test_margin_ratio_calculation() {
        let mut account = MarginAccount::new("bybit".to_string(), "UNIFIED".to_string());

        account.update_totals(
            Decimal::from(10000), // equity
            Decimal::from(10000), // wallet balance
            Decimal::from(8000),  // available
            Decimal::from(10000), // margin balance
            Decimal::ZERO,        // unrealized pnl
            Decimal::from(2000),  // initial margin
            Decimal::from(1000),  // maintenance margin
        );

        // Margin ratio = 10000 / 2000 = 5
        assert_eq!(account.margin_ratio, Decimal::from(5));

        // Health ratio = 10000 / 1000 = 10
        assert_eq!(account.health_ratio, Decimal::from(10));
        assert!(account.is_healthy());
        assert!(!account.is_at_risk());
    }

    #[test]
    fn test_account_at_risk() {
        let mut account = MarginAccount::new("bybit".to_string(), "UNIFIED".to_string());

        account.update_totals(
            Decimal::from(1050), // equity
            Decimal::from(1000), // wallet balance
            Decimal::from(50),   // available
            Decimal::from(1050), // margin balance
            Decimal::from(50),   // unrealized pnl
            Decimal::from(950),  // initial margin
            Decimal::from(1000), // maintenance margin
        );

        // Health ratio = 1050 / 1000 = 1.05 (below 1.1 threshold)
        assert!(account.is_at_risk());
        assert!(!account.is_healthy());
    }

    #[test]
    fn test_buying_power() {
        let mut account = MarginAccount::new("bybit".to_string(), "UNIFIED".to_string());

        account.update_totals(
            Decimal::from(10000),
            Decimal::from(10000),
            Decimal::from(8000), // available
            Decimal::from(10000),
            Decimal::ZERO,
            Decimal::from(2000),
            Decimal::from(1000),
        );

        // Buying power with 10x leverage = 8000 * 10 = 80000
        let buying_power = account.buying_power(Decimal::from(10));
        assert_eq!(buying_power, Decimal::from(80000));
    }

    #[tokio::test]
    async fn test_account_manager() {
        let manager = AccountManager::new();

        manager
            .update_balance(
                "bybit",
                "USDT".to_string(),
                Decimal::from(10000),
                Decimal::from(9000),
                Decimal::from(1000),
            )
            .await
            .unwrap();

        let account = manager.get_account("bybit").await.unwrap();
        let balance = account.get_balance("USDT").unwrap();
        assert_eq!(balance.total, Decimal::from(10000));

        let stats = manager.get_stats().await;
        assert_eq!(stats.num_exchanges, 1);
    }

    #[tokio::test]
    async fn test_multiple_exchanges() {
        let manager = AccountManager::new();

        manager
            .update_balance(
                "bybit",
                "USDT".to_string(),
                Decimal::from(10000),
                Decimal::from(9000),
                Decimal::from(1000),
            )
            .await
            .unwrap();

        manager
            .update_balance(
                "binance",
                "USDT".to_string(),
                Decimal::from(5000),
                Decimal::from(4500),
                Decimal::from(500),
            )
            .await
            .unwrap();

        let stats = manager.get_stats().await;
        assert_eq!(stats.num_exchanges, 2);

        let accounts = manager.get_all_accounts().await;
        assert_eq!(accounts.len(), 2);
    }

    #[tokio::test]
    async fn test_risk_detection() {
        let manager = AccountManager::new();

        // Create a risky account
        let mut account = MarginAccount::new("bybit".to_string(), "UNIFIED".to_string());
        account.update_totals(
            Decimal::from(1050),
            Decimal::from(1000),
            Decimal::from(50),
            Decimal::from(1050),
            Decimal::from(50),
            Decimal::from(950),
            Decimal::from(1000),
        );

        manager.update_account(account).await.unwrap();

        let at_risk = manager.check_risk().await;
        assert_eq!(at_risk.len(), 1);
        assert!(at_risk[0].contains("bybit"));
    }
}
