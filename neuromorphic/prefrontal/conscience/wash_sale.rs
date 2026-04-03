//! Wash sale prevention
//!
//! Part of the Prefrontal region
//! Component: conscience
//!
//! Implements wash sale rule compliance by tracking sales at a loss and preventing
//! repurchases of substantially identical securities within the 30-day window
//! (30 days before and 30 days after a loss sale).

use crate::common::{Error, Result};
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Wash sale rule window in days (30 days before and after)
const WASH_SALE_WINDOW_DAYS: u64 = 30;
/// Milliseconds in a day
const MS_PER_DAY: u64 = 24 * 60 * 60 * 1000;
/// Wash sale window in milliseconds
const WASH_SALE_WINDOW_MS: u64 = WASH_SALE_WINDOW_DAYS * MS_PER_DAY;

/// Result of a wash sale check
#[derive(Debug, Clone)]
pub struct WashSaleCheckResult {
    /// Whether the trade is allowed
    pub allowed: bool,
    /// Whether this would trigger a wash sale
    pub is_wash_sale: bool,
    /// Disallowed loss amount (if wash sale)
    pub disallowed_loss: f64,
    /// Related loss sale that triggered the wash sale
    pub related_loss_sale: Option<LossSale>,
    /// Warning message
    pub message: String,
    /// Days until the restriction expires
    pub days_until_clear: Option<u64>,
}

impl WashSaleCheckResult {
    pub fn allowed() -> Self {
        Self {
            allowed: true,
            is_wash_sale: false,
            disallowed_loss: 0.0,
            related_loss_sale: None,
            message: "Trade allowed".to_string(),
            days_until_clear: None,
        }
    }

    pub fn blocked(loss_sale: LossSale, disallowed_loss: f64, days_until_clear: u64) -> Self {
        Self {
            allowed: false,
            is_wash_sale: true,
            disallowed_loss,
            related_loss_sale: Some(loss_sale.clone()),
            message: format!(
                "Wash sale violation: repurchasing within 30 days of loss sale on {}. \
                 Disallowed loss: ${:.2}. Clear in {} days.",
                loss_sale.symbol, disallowed_loss, days_until_clear
            ),
            days_until_clear: Some(days_until_clear),
        }
    }
}

/// A recorded sale at a loss
#[derive(Debug, Clone)]
pub struct LossSale {
    /// Unique ID
    pub id: u64,
    /// Symbol sold
    pub symbol: String,
    /// Quantity sold
    pub quantity: f64,
    /// Sale price
    pub sale_price: f64,
    /// Cost basis
    pub cost_basis: f64,
    /// Loss amount (positive value)
    pub loss_amount: f64,
    /// Timestamp of sale
    pub sale_time: u64,
    /// Window start (30 days before sale)
    pub window_start: u64,
    /// Window end (30 days after sale)
    pub window_end: u64,
    /// Whether this loss has been disallowed due to wash sale
    pub is_disallowed: bool,
    /// Replacement purchase that triggered disallowance
    pub replacement_purchase_id: Option<u64>,
    /// Related symbols (substantially identical)
    pub related_symbols: Vec<String>,
}

impl LossSale {
    pub fn new(
        id: u64,
        symbol: &str,
        quantity: f64,
        sale_price: f64,
        cost_basis: f64,
        sale_time: u64,
    ) -> Self {
        let loss_amount = (cost_basis - sale_price * quantity).max(0.0);
        Self {
            id,
            symbol: symbol.to_string(),
            quantity,
            sale_price,
            cost_basis,
            loss_amount,
            sale_time,
            window_start: sale_time.saturating_sub(WASH_SALE_WINDOW_MS),
            window_end: sale_time.saturating_add(WASH_SALE_WINDOW_MS),
            is_disallowed: false,
            replacement_purchase_id: None,
            related_symbols: Vec::new(),
        }
    }

    /// Check if a timestamp is within the wash sale window
    pub fn is_in_window(&self, timestamp: u64) -> bool {
        timestamp >= self.window_start && timestamp <= self.window_end
    }

    /// Days until the window expires
    pub fn days_until_clear(&self, current_time: u64) -> u64 {
        if current_time >= self.window_end {
            0
        } else {
            (self.window_end - current_time) / MS_PER_DAY
        }
    }
}

/// A purchase that may trigger a wash sale
#[derive(Debug, Clone)]
pub struct Purchase {
    /// Unique ID
    pub id: u64,
    /// Symbol purchased
    pub symbol: String,
    /// Quantity purchased
    pub quantity: f64,
    /// Purchase price
    pub purchase_price: f64,
    /// Total cost
    pub total_cost: f64,
    /// Timestamp of purchase
    pub purchase_time: u64,
    /// Whether this triggered a wash sale
    pub triggered_wash_sale: bool,
    /// Related loss sale ID if wash sale triggered
    pub related_loss_sale_id: Option<u64>,
    /// Adjusted cost basis (includes disallowed loss)
    pub adjusted_cost_basis: f64,
}

/// Substantially identical security mapping
#[derive(Debug, Clone)]
pub struct SubstantiallyIdenticalGroup {
    /// Group name
    pub name: String,
    /// Symbols that are substantially identical
    pub symbols: Vec<String>,
    /// Description of why they're identical
    pub description: String,
}

/// Configuration for wash sale prevention
#[derive(Debug, Clone)]
pub struct WashSaleConfig {
    /// Enable wash sale prevention
    pub enabled: bool,
    /// Block trades that would trigger wash sales
    pub block_violations: bool,
    /// Warn only (don't block)
    pub warn_only: bool,
    /// Include substantially identical securities
    pub check_substantially_identical: bool,
    /// Minimum loss amount to track
    pub min_loss_amount: f64,
    /// Maximum records to keep
    pub max_records: usize,
    /// Auto-adjust cost basis for wash sales
    pub auto_adjust_basis: bool,
}

impl Default for WashSaleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            block_violations: true,
            warn_only: false,
            check_substantially_identical: true,
            min_loss_amount: 1.0,
            max_records: 10_000,
            auto_adjust_basis: true,
        }
    }
}

/// Wash sale prevention and tracking
pub struct WashSale {
    /// Configuration
    config: WashSaleConfig,
    /// Loss sales indexed by ID
    loss_sales: HashMap<u64, LossSale>,
    /// Loss sales indexed by symbol
    loss_sales_by_symbol: HashMap<String, Vec<u64>>,
    /// Purchases indexed by ID
    purchases: HashMap<u64, Purchase>,
    /// Substantially identical groups
    identical_groups: Vec<SubstantiallyIdenticalGroup>,
    /// Symbol to group mapping
    symbol_to_group: HashMap<String, usize>,
    /// Next loss sale ID
    next_loss_id: u64,
    /// Next purchase ID
    next_purchase_id: u64,
    /// Wash sale violations count
    violation_count: u64,
    /// Total disallowed losses
    total_disallowed_losses: f64,
    /// Current timestamp
    current_time: u64,
}

impl Default for WashSale {
    fn default() -> Self {
        Self::new()
    }
}

impl WashSale {
    /// Create a new instance
    pub fn new() -> Self {
        Self::with_config(WashSaleConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: WashSaleConfig) -> Self {
        Self {
            config,
            loss_sales: HashMap::new(),
            loss_sales_by_symbol: HashMap::new(),
            purchases: HashMap::new(),
            identical_groups: Vec::new(),
            symbol_to_group: HashMap::new(),
            next_loss_id: 1,
            next_purchase_id: 1,
            violation_count: 0,
            total_disallowed_losses: 0.0,
            current_time: 0,
        }
    }

    /// Set current timestamp
    pub fn set_time(&mut self, timestamp: u64) {
        self.current_time = timestamp;
    }

    /// Register a group of substantially identical securities
    pub fn register_identical_group(&mut self, group: SubstantiallyIdenticalGroup) {
        let group_idx = self.identical_groups.len();
        for symbol in &group.symbols {
            self.symbol_to_group.insert(symbol.clone(), group_idx);
        }
        self.identical_groups.push(group);
    }

    /// Get all symbols that are substantially identical to the given symbol
    pub fn get_identical_symbols(&self, symbol: &str) -> Vec<String> {
        if !self.config.check_substantially_identical {
            return vec![symbol.to_string()];
        }

        if let Some(&group_idx) = self.symbol_to_group.get(symbol) {
            self.identical_groups[group_idx].symbols.clone()
        } else {
            vec![symbol.to_string()]
        }
    }

    /// Record a sale at a loss
    pub fn record_loss_sale(
        &mut self,
        symbol: &str,
        quantity: f64,
        sale_price: f64,
        cost_basis: f64,
    ) -> Result<u64> {
        let loss_amount = cost_basis - sale_price * quantity;
        if loss_amount < self.config.min_loss_amount {
            return Err(Error::InvalidInput(
                "Sale is not at a loss or loss is below minimum".to_string(),
            ));
        }

        let id = self.next_loss_id;
        self.next_loss_id += 1;

        let mut loss_sale = LossSale::new(
            id,
            symbol,
            quantity,
            sale_price,
            cost_basis,
            self.current_time,
        );

        // Add related symbols
        loss_sale.related_symbols = self.get_identical_symbols(symbol);

        // Index by symbol and related symbols
        for sym in &loss_sale.related_symbols {
            self.loss_sales_by_symbol
                .entry(sym.clone())
                .or_default()
                .push(id);
        }

        self.loss_sales.insert(id, loss_sale);

        info!(
            loss_sale_id = id,
            symbol = symbol,
            loss_amount = loss_amount,
            "Recorded loss sale"
        );

        // Prune old records
        self.prune_old_records();

        Ok(id)
    }

    /// Check if a proposed purchase would trigger a wash sale
    pub fn check_purchase(&self, symbol: &str, timestamp: u64) -> WashSaleCheckResult {
        if !self.config.enabled {
            return WashSaleCheckResult::allowed();
        }

        // Get all related symbols
        let related_symbols = self.get_identical_symbols(symbol);

        // Check for any loss sales in the window
        for sym in &related_symbols {
            if let Some(loss_ids) = self.loss_sales_by_symbol.get(sym) {
                for &loss_id in loss_ids {
                    if let Some(loss_sale) = self.loss_sales.get(&loss_id) {
                        // Skip already disallowed losses
                        if loss_sale.is_disallowed {
                            continue;
                        }

                        // Check if purchase is within the wash sale window
                        if loss_sale.is_in_window(timestamp) {
                            let days_until_clear = loss_sale.days_until_clear(timestamp);

                            if self.config.block_violations && !self.config.warn_only {
                                return WashSaleCheckResult::blocked(
                                    loss_sale.clone(),
                                    loss_sale.loss_amount,
                                    days_until_clear,
                                );
                            } else {
                                // Warn only
                                let mut result = WashSaleCheckResult::allowed();
                                result.is_wash_sale = true;
                                result.disallowed_loss = loss_sale.loss_amount;
                                result.related_loss_sale = Some(loss_sale.clone());
                                result.days_until_clear = Some(days_until_clear);
                                result.message = format!(
                                    "Warning: This purchase would trigger a wash sale. \
                                     Disallowed loss: ${:.2}",
                                    loss_sale.loss_amount
                                );
                                return result;
                            }
                        }
                    }
                }
            }
        }

        WashSaleCheckResult::allowed()
    }

    /// Record a purchase (may trigger wash sale)
    pub fn record_purchase(
        &mut self,
        symbol: &str,
        quantity: f64,
        purchase_price: f64,
    ) -> Result<(u64, Option<WashSaleCheckResult>)> {
        let check_result = self.check_purchase(symbol, self.current_time);

        if !check_result.allowed {
            warn!(
                symbol = symbol,
                message = %check_result.message,
                "Purchase blocked due to wash sale"
            );
            return Err(Error::RiskViolation(check_result.message.clone()));
        }

        let id = self.next_purchase_id;
        self.next_purchase_id += 1;

        let total_cost = quantity * purchase_price;
        let mut purchase = Purchase {
            id,
            symbol: symbol.to_string(),
            quantity,
            purchase_price,
            total_cost,
            purchase_time: self.current_time,
            triggered_wash_sale: check_result.is_wash_sale,
            related_loss_sale_id: check_result.related_loss_sale.as_ref().map(|ls| ls.id),
            adjusted_cost_basis: total_cost,
        };

        // Handle wash sale if it occurred (warn_only mode)
        let wash_sale_result = if check_result.is_wash_sale {
            if let Some(ref loss_sale) = check_result.related_loss_sale {
                // Mark the loss sale as disallowed
                if let Some(ls) = self.loss_sales.get_mut(&loss_sale.id) {
                    ls.is_disallowed = true;
                    ls.replacement_purchase_id = Some(id);
                }

                // Adjust cost basis of new purchase
                if self.config.auto_adjust_basis {
                    purchase.adjusted_cost_basis = total_cost + check_result.disallowed_loss;
                }

                self.violation_count += 1;
                self.total_disallowed_losses += check_result.disallowed_loss;

                info!(
                    purchase_id = id,
                    loss_sale_id = loss_sale.id,
                    disallowed_loss = check_result.disallowed_loss,
                    adjusted_basis = purchase.adjusted_cost_basis,
                    "Wash sale recorded"
                );

                Some(check_result)
            } else {
                None
            }
        } else {
            None
        };

        self.purchases.insert(id, purchase);

        Ok((id, wash_sale_result))
    }

    /// Get active loss sales (still in window)
    pub fn get_active_loss_sales(&self, symbol: Option<&str>) -> Vec<&LossSale> {
        self.loss_sales
            .values()
            .filter(|ls| {
                ls.window_end > self.current_time
                    && !ls.is_disallowed
                    && symbol.map_or(true, |s| {
                        ls.symbol == s || ls.related_symbols.contains(&s.to_string())
                    })
            })
            .collect()
    }

    /// Get all restricted symbols (cannot purchase due to recent loss sales)
    pub fn get_restricted_symbols(&self) -> Vec<String> {
        let mut restricted = Vec::new();
        for ls in self.loss_sales.values() {
            if ls.window_end > self.current_time && !ls.is_disallowed {
                for sym in &ls.related_symbols {
                    if !restricted.contains(sym) {
                        restricted.push(sym.clone());
                    }
                }
            }
        }
        restricted
    }

    /// Get restriction info for a symbol
    pub fn get_restriction_info(&self, symbol: &str) -> Option<RestrictionInfo> {
        let related_symbols = self.get_identical_symbols(symbol);

        for sym in &related_symbols {
            if let Some(loss_ids) = self.loss_sales_by_symbol.get(sym) {
                for &loss_id in loss_ids {
                    if let Some(ls) = self.loss_sales.get(&loss_id) {
                        if ls.window_end > self.current_time && !ls.is_disallowed {
                            return Some(RestrictionInfo {
                                symbol: symbol.to_string(),
                                restricted_until: ls.window_end,
                                days_remaining: ls.days_until_clear(self.current_time),
                                loss_sale_id: ls.id,
                                loss_amount: ls.loss_amount,
                                related_symbols: ls.related_symbols.clone(),
                            });
                        }
                    }
                }
            }
        }

        None
    }

    /// Prune old records outside the wash sale window
    fn prune_old_records(&mut self) {
        // Remove loss sales outside the window
        let cutoff = self.current_time.saturating_sub(WASH_SALE_WINDOW_MS);
        let old_ids: Vec<u64> = self
            .loss_sales
            .iter()
            .filter(|(_, ls)| ls.window_end < cutoff)
            .map(|(id, _)| *id)
            .collect();

        for id in old_ids {
            if let Some(ls) = self.loss_sales.remove(&id) {
                // Remove from symbol index
                for sym in &ls.related_symbols {
                    if let Some(ids) = self.loss_sales_by_symbol.get_mut(sym) {
                        ids.retain(|&i| i != id);
                    }
                }
            }
        }

        // Limit total records
        while self.loss_sales.len() > self.config.max_records {
            if let Some((&oldest_id, _)) = self.loss_sales.iter().min_by_key(|(_, ls)| ls.sale_time)
            {
                if let Some(ls) = self.loss_sales.remove(&oldest_id) {
                    for sym in &ls.related_symbols {
                        if let Some(ids) = self.loss_sales_by_symbol.get_mut(sym) {
                            ids.retain(|&i| i != oldest_id);
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    /// Get statistics
    pub fn stats(&self) -> WashSaleStats {
        let active_restrictions = self.get_restricted_symbols().len();
        let pending_loss_sales = self
            .loss_sales
            .values()
            .filter(|ls| ls.window_end > self.current_time && !ls.is_disallowed)
            .count();

        WashSaleStats {
            total_loss_sales_tracked: self.loss_sales.len(),
            active_restrictions,
            pending_loss_sales,
            violation_count: self.violation_count,
            total_disallowed_losses: self.total_disallowed_losses,
            identical_groups: self.identical_groups.len(),
        }
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        debug!(
            loss_sales = self.loss_sales.len(),
            violations = self.violation_count,
            "Wash sale process called"
        );
        Ok(())
    }
}

/// Restriction information for a symbol
#[derive(Debug, Clone)]
pub struct RestrictionInfo {
    pub symbol: String,
    pub restricted_until: u64,
    pub days_remaining: u64,
    pub loss_sale_id: u64,
    pub loss_amount: f64,
    pub related_symbols: Vec<String>,
}

/// Wash sale statistics
#[derive(Debug, Clone)]
pub struct WashSaleStats {
    pub total_loss_sales_tracked: usize,
    pub active_restrictions: usize,
    pub pending_loss_sales: usize,
    pub violation_count: u64,
    pub total_disallowed_losses: f64,
    pub identical_groups: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    const DAY_MS: u64 = 24 * 60 * 60 * 1000;

    fn create_test_wash_sale() -> WashSale {
        let mut ws = WashSale::new();
        ws.set_time(30 * DAY_MS); // Start at day 30
        ws
    }

    #[test]
    fn test_basic() {
        let instance = WashSale::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_record_loss_sale() {
        let mut ws = create_test_wash_sale();

        // Sell 100 shares at $90, cost basis $100/share = $1000 loss
        let id = ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        assert!(id > 0);
        assert_eq!(ws.loss_sales.len(), 1);

        let ls = ws.loss_sales.get(&id).unwrap();
        assert_eq!(ls.symbol, "AAPL");
        assert!((ls.loss_amount - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_wash_sale_detection() {
        let mut ws = create_test_wash_sale();

        // Record loss sale
        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        // Try to buy within 30 days - should be blocked
        let check = ws.check_purchase("AAPL", ws.current_time + 10 * DAY_MS);
        assert!(!check.allowed);
        assert!(check.is_wash_sale);
        assert!(check.days_until_clear.is_some());
    }

    #[test]
    fn test_wash_sale_after_window() {
        let mut ws = create_test_wash_sale();

        // Record loss sale
        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        // Buy after 31 days - should be allowed
        let check = ws.check_purchase("AAPL", ws.current_time + 31 * DAY_MS);
        assert!(check.allowed);
        assert!(!check.is_wash_sale);
    }

    #[test]
    fn test_substantially_identical() {
        let mut ws = create_test_wash_sale();

        // Register SPY and VOO as substantially identical
        ws.register_identical_group(SubstantiallyIdenticalGroup {
            name: "SP500 ETFs".to_string(),
            symbols: vec!["SPY".to_string(), "VOO".to_string(), "IVV".to_string()],
            description: "S&P 500 index ETFs".to_string(),
        });

        // Record loss sale on SPY
        ws.record_loss_sale("SPY", 100.0, 400.0, 45000.0).unwrap();

        // Try to buy VOO - should be blocked (substantially identical)
        let check = ws.check_purchase("VOO", ws.current_time + 10 * DAY_MS);
        assert!(!check.allowed);
        assert!(check.is_wash_sale);
    }

    #[test]
    fn test_warn_only_mode() {
        let config = WashSaleConfig {
            warn_only: true,
            ..Default::default()
        };
        let mut ws = WashSale::with_config(config);
        ws.set_time(30 * DAY_MS);

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        // Check returns allowed but warns
        let check = ws.check_purchase("AAPL", ws.current_time + 10 * DAY_MS);
        assert!(check.allowed); // Allowed in warn-only mode
        assert!(check.is_wash_sale); // But still flagged as wash sale
    }

    #[test]
    fn test_record_purchase_with_wash_sale() {
        let config = WashSaleConfig {
            warn_only: true,
            auto_adjust_basis: true,
            ..Default::default()
        };
        let mut ws = WashSale::with_config(config);
        ws.set_time(30 * DAY_MS);

        // Loss sale: $1000 loss
        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        // Purchase within window
        ws.set_time(35 * DAY_MS);
        let (id, result) = ws.record_purchase("AAPL", 100.0, 95.0).unwrap();

        assert!(id > 0);
        assert!(result.is_some());

        let purchase = ws.purchases.get(&id).unwrap();
        assert!(purchase.triggered_wash_sale);
        // Adjusted basis = 100 * 95 + 1000 disallowed loss = 10500
        assert!((purchase.adjusted_cost_basis - 10500.0).abs() < 0.01);

        assert_eq!(ws.violation_count, 1);
    }

    #[test]
    fn test_restricted_symbols() {
        let mut ws = create_test_wash_sale();

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();
        ws.record_loss_sale("GOOGL", 50.0, 100.0, 6000.0).unwrap();

        let restricted = ws.get_restricted_symbols();
        assert_eq!(restricted.len(), 2);
        assert!(restricted.contains(&"AAPL".to_string()));
        assert!(restricted.contains(&"GOOGL".to_string()));
    }

    #[test]
    fn test_restriction_info() {
        let mut ws = create_test_wash_sale();

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        let info = ws.get_restriction_info("AAPL").unwrap();
        assert_eq!(info.symbol, "AAPL");
        assert!(info.days_remaining <= 30);
        assert!((info.loss_amount - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_no_restriction_after_window() {
        let mut ws = create_test_wash_sale();

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        // Move time past window
        ws.set_time(ws.current_time + 31 * DAY_MS);

        let info = ws.get_restriction_info("AAPL");
        assert!(info.is_none());
    }

    #[test]
    fn test_min_loss_amount() {
        let config = WashSaleConfig {
            min_loss_amount: 100.0,
            ..Default::default()
        };
        let mut ws = WashSale::with_config(config);
        ws.set_time(30 * DAY_MS);

        // Small loss below threshold
        let result = ws.record_loss_sale("AAPL", 10.0, 99.0, 1000.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_purchase_blocked() {
        let mut ws = create_test_wash_sale();

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        ws.set_time(ws.current_time + 10 * DAY_MS);
        let result = ws.record_purchase("AAPL", 100.0, 95.0);

        assert!(result.is_err());
    }

    #[test]
    fn test_identical_symbols_lookup() {
        let mut ws = create_test_wash_sale();

        ws.register_identical_group(SubstantiallyIdenticalGroup {
            name: "Test Group".to_string(),
            symbols: vec!["A".to_string(), "B".to_string(), "C".to_string()],
            description: "Test".to_string(),
        });

        let identical = ws.get_identical_symbols("A");
        assert_eq!(identical.len(), 3);
        assert!(identical.contains(&"B".to_string()));
        assert!(identical.contains(&"C".to_string()));

        // Non-grouped symbol
        let single = ws.get_identical_symbols("AAPL");
        assert_eq!(single.len(), 1);
        assert_eq!(single[0], "AAPL");
    }

    #[test]
    fn test_active_loss_sales() {
        let mut ws = create_test_wash_sale();

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();
        ws.record_loss_sale("GOOGL", 50.0, 100.0, 6000.0).unwrap();

        let active = ws.get_active_loss_sales(None);
        assert_eq!(active.len(), 2);

        let apple_active = ws.get_active_loss_sales(Some("AAPL"));
        assert_eq!(apple_active.len(), 1);
    }

    #[test]
    fn test_stats() {
        let mut ws = create_test_wash_sale();

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        let stats = ws.stats();
        assert_eq!(stats.total_loss_sales_tracked, 1);
        assert_eq!(stats.active_restrictions, 1);
        assert_eq!(stats.pending_loss_sales, 1);
    }

    #[test]
    fn test_disabled() {
        let config = WashSaleConfig {
            enabled: false,
            ..Default::default()
        };
        let mut ws = WashSale::with_config(config);
        ws.set_time(30 * DAY_MS);

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        // Should be allowed even within window when disabled
        let check = ws.check_purchase("AAPL", ws.current_time + 10 * DAY_MS);
        assert!(check.allowed);
    }

    #[test]
    fn test_days_until_clear() {
        let mut ws = create_test_wash_sale();

        ws.record_loss_sale("AAPL", 100.0, 90.0, 10000.0).unwrap();

        // Check at 10 days after sale
        let check = ws.check_purchase("AAPL", ws.current_time + 10 * DAY_MS);
        assert!(check.days_until_clear.is_some());
        // Should be ~20 days remaining
        let days = check.days_until_clear.unwrap();
        assert!((19..=21).contains(&days));
    }
}
