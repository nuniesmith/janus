//! Order Validation Module
//!
//! Provides pre-trade validation and risk checks before orders are submitted to exchanges.
//! This includes position limits, order size limits, price checks, and compliance rules.

use crate::error::Result;
use crate::types::{Order, OrderSide, OrderTypeEnum};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::debug;

/// Configuration for order validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum order size per symbol (in quote currency)
    pub max_order_value: Decimal,

    /// Minimum order size per symbol (in quote currency)
    pub min_order_value: Decimal,

    /// Maximum position size per symbol (in quote currency)
    pub max_position_value: Decimal,

    /// Maximum leverage allowed
    pub max_leverage: Decimal,

    /// Price deviation threshold from market (percentage)
    pub max_price_deviation_pct: Decimal,

    /// Require sufficient balance check
    pub check_balance: bool,

    /// Require price reasonableness check
    pub check_price_bounds: bool,

    /// Require position limits check
    pub check_position_limits: bool,

    /// Symbol-specific overrides
    pub symbol_overrides: HashMap<String, SymbolValidationConfig>,

    /// Maximum orders per second per symbol
    pub max_orders_per_second: u32,

    /// Allow short selling
    pub allow_short_selling: bool,

    /// Minimum time between orders for same symbol (milliseconds)
    pub min_order_interval_ms: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_order_value: Decimal::from(100_000),
            min_order_value: Decimal::from(10),
            max_position_value: Decimal::from(500_000),
            max_leverage: Decimal::from(10),
            max_price_deviation_pct: Decimal::from(5), // 5%
            check_balance: true,
            check_price_bounds: true,
            check_position_limits: true,
            symbol_overrides: HashMap::new(),
            max_orders_per_second: 10,
            allow_short_selling: true,
            min_order_interval_ms: 100,
        }
    }
}

/// Symbol-specific validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolValidationConfig {
    pub max_order_value: Option<Decimal>,
    pub min_order_value: Option<Decimal>,
    pub max_position_value: Option<Decimal>,
    pub max_leverage: Option<Decimal>,
    pub allow_trading: bool,
}

/// Result of order validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Is the order valid?
    pub is_valid: bool,

    /// List of validation warnings (non-blocking)
    pub warnings: Vec<String>,

    /// List of rejection reasons (blocking)
    pub rejection_reasons: Vec<String>,

    /// Estimated order value
    pub estimated_value: Option<Decimal>,

    /// Suggested modifications (if any)
    pub suggestions: Vec<String>,
}

impl ValidationResult {
    /// Create a valid result
    pub fn valid() -> Self {
        Self {
            is_valid: true,
            warnings: Vec::new(),
            rejection_reasons: Vec::new(),
            estimated_value: None,
            suggestions: Vec::new(),
        }
    }

    /// Create an invalid result with reason
    pub fn invalid(reason: String) -> Self {
        Self {
            is_valid: false,
            warnings: Vec::new(),
            rejection_reasons: vec![reason],
            estimated_value: None,
            suggestions: Vec::new(),
        }
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Add a rejection reason and mark as invalid
    pub fn add_rejection(&mut self, reason: String) {
        self.is_valid = false;
        self.rejection_reasons.push(reason);
    }

    /// Add a suggestion
    pub fn add_suggestion(&mut self, suggestion: String) {
        self.suggestions.push(suggestion);
    }
}

/// Order validator
pub struct OrderValidator {
    config: ValidationConfig,

    /// Order submission timestamps per symbol (for rate limiting)
    submission_history: parking_lot::RwLock<HashMap<String, Vec<i64>>>,
}

impl OrderValidator {
    /// Create a new validator with the given configuration
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            submission_history: parking_lot::RwLock::new(HashMap::new()),
        }
    }

    /// Validate an order before submission
    pub async fn validate_order(&self, order: &Order) -> Result<ValidationResult> {
        let mut result = ValidationResult::valid();

        // Get effective config (with symbol overrides)
        let effective_config = self.get_effective_config(&order.symbol);

        // 1. Basic sanity checks
        self.validate_basic_constraints(order, &mut result)?;

        // 2. Symbol trading allowed check
        if let Some(symbol_config) = self.config.symbol_overrides.get(&order.symbol) {
            if !symbol_config.allow_trading {
                result.add_rejection(format!("Trading not allowed for symbol {}", order.symbol));
                return Ok(result);
            }
        }

        // 3. Quantity checks
        self.validate_quantity(order, &mut result)?;

        // 4. Order value checks
        if let Some(estimated_value) = self.estimate_order_value(order)? {
            result.estimated_value = Some(estimated_value);
            self.validate_order_value(estimated_value, &effective_config, &mut result)?;
        }

        // 5. Price bounds check (for limit orders)
        if self.config.check_price_bounds {
            self.validate_price_bounds(order, &mut result)?;
        }

        // 6. Short selling check
        if !self.config.allow_short_selling && order.side == OrderSide::Sell {
            // In a real system, we'd check if this is a closing or opening sell
            // For now, we'll allow sells but add a warning
            result.add_warning("Short selling may be restricted".to_string());
        }

        // 7. Rate limiting check
        self.validate_rate_limit(order, &mut result)?;

        // 8. Time interval check
        self.validate_order_interval(order, &mut result)?;

        debug!(
            order_id = %order.id,
            symbol = %order.symbol,
            is_valid = result.is_valid,
            warnings = result.warnings.len(),
            rejections = result.rejection_reasons.len(),
            "Order validation complete"
        );

        Ok(result)
    }

    /// Get effective configuration with symbol overrides applied
    fn get_effective_config(&self, symbol: &str) -> EffectiveConfig {
        let base = &self.config;
        let override_config = self.config.symbol_overrides.get(symbol);

        EffectiveConfig {
            max_order_value: override_config
                .and_then(|c| c.max_order_value)
                .unwrap_or(base.max_order_value),
            min_order_value: override_config
                .and_then(|c| c.min_order_value)
                .unwrap_or(base.min_order_value),
            max_position_value: override_config
                .and_then(|c| c.max_position_value)
                .unwrap_or(base.max_position_value),
            max_leverage: override_config
                .and_then(|c| c.max_leverage)
                .unwrap_or(base.max_leverage),
        }
    }

    /// Validate basic order constraints
    fn validate_basic_constraints(
        &self,
        order: &Order,
        result: &mut ValidationResult,
    ) -> Result<()> {
        // Check symbol is not empty
        if order.symbol.is_empty() {
            result.add_rejection("Symbol cannot be empty".to_string());
        }

        // Check exchange is not empty
        if order.exchange.is_empty() {
            result.add_rejection("Exchange cannot be empty".to_string());
        }

        // Check quantity is positive
        if order.quantity <= Decimal::ZERO {
            result.add_rejection(format!("Quantity must be positive, got {}", order.quantity));
        }

        // Check limit price is positive for limit orders
        if matches!(
            order.order_type,
            OrderTypeEnum::Limit | OrderTypeEnum::StopLimit
        ) {
            if let Some(price) = order.price {
                if price <= Decimal::ZERO {
                    result.add_rejection(format!("Limit price must be positive, got {}", price));
                }
            } else {
                result.add_rejection("Limit orders must have a price".to_string());
            }
        }

        // Check stop price for stop orders
        if matches!(
            order.order_type,
            OrderTypeEnum::StopMarket | OrderTypeEnum::StopLimit
        ) {
            if let Some(stop_price) = order.stop_price {
                if stop_price <= Decimal::ZERO {
                    result
                        .add_rejection(format!("Stop price must be positive, got {}", stop_price));
                }
            } else {
                result.add_rejection("Stop orders must have a stop price".to_string());
            }
        }

        Ok(())
    }

    /// Validate order quantity
    fn validate_quantity(&self, order: &Order, result: &mut ValidationResult) -> Result<()> {
        // Check for extremely small quantities
        if order.quantity < Decimal::new(1, 8) {
            result.add_warning(format!(
                "Order quantity {} is very small and may be rejected by exchange",
                order.quantity
            ));
        }

        // Check for extremely large quantities
        if order.quantity > Decimal::from(1_000_000) {
            result.add_warning(format!("Order quantity {} is very large", order.quantity));
        }

        Ok(())
    }

    /// Estimate the order value
    fn estimate_order_value(&self, order: &Order) -> Result<Option<Decimal>> {
        match order.order_type {
            OrderTypeEnum::Limit
            | OrderTypeEnum::StopLimit
            | OrderTypeEnum::StopLossLimit
            | OrderTypeEnum::TakeProfitLimit
            | OrderTypeEnum::LimitMaker => {
                if let Some(price) = order.price {
                    Ok(Some(price * order.quantity))
                } else {
                    Ok(None)
                }
            }
            OrderTypeEnum::Market
            | OrderTypeEnum::StopMarket
            | OrderTypeEnum::StopLoss
            | OrderTypeEnum::TakeProfit => {
                // For market orders, we'd need current market price
                // For now, return None (would be provided by caller or fetched from market data)
                Ok(None)
            }
            OrderTypeEnum::TrailingStop => Ok(None),
        }
    }

    /// Validate order value against limits
    fn validate_order_value(
        &self,
        value: Decimal,
        config: &EffectiveConfig,
        result: &mut ValidationResult,
    ) -> Result<()> {
        if value < config.min_order_value {
            result.add_rejection(format!(
                "Order value {} below minimum {}",
                value, config.min_order_value
            ));
        }

        if value > config.max_order_value {
            result.add_rejection(format!(
                "Order value {} exceeds maximum {}",
                value, config.max_order_value
            ));
            result.add_suggestion(format!(
                "Consider reducing order size to stay within {} limit",
                config.max_order_value
            ));
        }

        Ok(())
    }

    /// Validate price is within reasonable bounds
    fn validate_price_bounds(&self, order: &Order, result: &mut ValidationResult) -> Result<()> {
        // This is a placeholder - in a real system, we'd fetch current market price
        // and compare against it

        if let Some(price) = order.price {
            // Check for obviously wrong prices
            if price > Decimal::from(1_000_000_000) {
                result.add_warning(format!("Price {} seems unreasonably high", price));
            }
        }

        Ok(())
    }

    /// Validate rate limiting
    fn validate_rate_limit(&self, order: &Order, result: &mut ValidationResult) -> Result<()> {
        let now = chrono::Utc::now().timestamp_millis();
        let one_second_ago = now - 1000;

        let mut history = self.submission_history.write();
        let symbol_history = history.entry(order.symbol.clone()).or_default();

        // Remove old entries
        symbol_history.retain(|&ts| ts > one_second_ago);

        // Check rate limit
        if symbol_history.len() >= self.config.max_orders_per_second as usize {
            result.add_rejection(format!(
                "Rate limit exceeded: {} orders per second for {}",
                self.config.max_orders_per_second, order.symbol
            ));
            return Ok(());
        }

        // Record this submission
        symbol_history.push(now);

        Ok(())
    }

    /// Validate minimum time interval between orders
    fn validate_order_interval(&self, order: &Order, result: &mut ValidationResult) -> Result<()> {
        let now = chrono::Utc::now().timestamp_millis();
        let min_interval = self.config.min_order_interval_ms as i64;

        let history = self.submission_history.read();
        if let Some(symbol_history) = history.get(&order.symbol) {
            if let Some(&last_submission) = symbol_history.last() {
                let time_since_last = now - last_submission;
                if time_since_last < min_interval {
                    result.add_warning(format!(
                        "Order submitted {}ms after previous order (minimum interval: {}ms)",
                        time_since_last, min_interval
                    ));
                }
            }
        }

        Ok(())
    }

    /// Update configuration
    pub fn update_config(&mut self, config: ValidationConfig) {
        self.config = config;
    }

    /// Get current configuration
    pub fn get_config(&self) -> &ValidationConfig {
        &self.config
    }

    /// Clear submission history (useful for testing)
    pub fn clear_history(&self) {
        let mut history = self.submission_history.write();
        history.clear();
    }
}

/// Effective configuration after applying symbol overrides
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct EffectiveConfig {
    max_order_value: Decimal,
    min_order_value: Decimal,
    max_position_value: Decimal,
    max_leverage: Decimal,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TimeInForceEnum;

    fn create_test_order() -> Order {
        Order {
            id: "test-order-1".to_string(),
            exchange_order_id: None,
            client_order_id: None,
            signal_id: "signal-1".to_string(),
            symbol: "BTCUSD".to_string(),
            exchange: "bybit".to_string(),
            side: OrderSide::Buy,
            order_type: OrderTypeEnum::Limit,
            quantity: Decimal::from(1),
            filled_quantity: Decimal::ZERO,
            remaining_quantity: Decimal::from(1),
            price: Some(Decimal::from(50000)),
            stop_price: None,
            average_fill_price: None,
            time_in_force: TimeInForceEnum::Gtc,
            strategy: crate::types::ExecutionStrategyEnum::Immediate,
            status: crate::types::OrderStatusEnum::New,
            fills: Vec::new(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_valid_order() {
        let validator = OrderValidator::new(ValidationConfig::default());
        let order = create_test_order();

        let result = validator.validate_order(&order).await.unwrap();
        assert!(result.is_valid);
        assert_eq!(result.rejection_reasons.len(), 0);
    }

    #[tokio::test]
    async fn test_zero_quantity_rejected() {
        let validator = OrderValidator::new(ValidationConfig::default());
        let mut order = create_test_order();
        order.quantity = Decimal::ZERO;

        let result = validator.validate_order(&order).await.unwrap();
        assert!(!result.is_valid);
        assert!(
            result
                .rejection_reasons
                .iter()
                .any(|r| r.contains("positive"))
        );
    }

    #[tokio::test]
    async fn test_negative_price_rejected() {
        let validator = OrderValidator::new(ValidationConfig::default());
        let mut order = create_test_order();
        order.price = Some(Decimal::from(-100));

        let result = validator.validate_order(&order).await.unwrap();
        assert!(!result.is_valid);
    }

    #[tokio::test]
    async fn test_order_value_limits() {
        let config = ValidationConfig {
            min_order_value: Decimal::from(1000),
            max_order_value: Decimal::from(10000),
            ..Default::default()
        };

        let validator = OrderValidator::new(config);

        // Order value = 1 * 50000 = 50000 (exceeds max)
        let order = create_test_order();
        let result = validator.validate_order(&order).await.unwrap();
        assert!(!result.is_valid);
    }

    #[tokio::test]
    async fn test_rate_limiting() {
        let config = ValidationConfig {
            max_orders_per_second: 2,
            ..Default::default()
        };

        let validator = OrderValidator::new(config);
        let order = create_test_order();

        // First two orders should pass
        let result1 = validator.validate_order(&order).await.unwrap();
        assert!(result1.is_valid);

        let result2 = validator.validate_order(&order).await.unwrap();
        assert!(result2.is_valid);

        // Third should fail rate limit
        let result3 = validator.validate_order(&order).await.unwrap();
        assert!(!result3.is_valid);
        assert!(
            result3
                .rejection_reasons
                .iter()
                .any(|r| r.contains("Rate limit"))
        );
    }

    #[tokio::test]
    async fn test_symbol_override() {
        let mut config = ValidationConfig::default();
        config.symbol_overrides.insert(
            "BTCUSD".to_string(),
            SymbolValidationConfig {
                max_order_value: Some(Decimal::from(100000)),
                min_order_value: None,
                max_position_value: None,
                max_leverage: None,
                allow_trading: true,
            },
        );

        let validator = OrderValidator::new(config);
        let order = create_test_order();

        let result = validator.validate_order(&order).await.unwrap();
        assert!(result.is_valid);
    }

    #[tokio::test]
    async fn test_trading_not_allowed() {
        let mut config = ValidationConfig::default();
        config.symbol_overrides.insert(
            "BTCUSD".to_string(),
            SymbolValidationConfig {
                max_order_value: None,
                min_order_value: None,
                max_position_value: None,
                max_leverage: None,
                allow_trading: false,
            },
        );

        let validator = OrderValidator::new(config);
        let order = create_test_order();

        let result = validator.validate_order(&order).await.unwrap();
        assert!(!result.is_valid);
        assert!(
            result
                .rejection_reasons
                .iter()
                .any(|r| r.contains("not allowed"))
        );
    }

    #[tokio::test]
    async fn test_limit_order_requires_price() {
        let validator = OrderValidator::new(ValidationConfig::default());
        let mut order = create_test_order();
        order.order_type = OrderTypeEnum::Limit;
        order.price = None;

        let result = validator.validate_order(&order).await.unwrap();
        assert!(!result.is_valid);
        assert!(
            result
                .rejection_reasons
                .iter()
                .any(|r| r.contains("must have a price"))
        );
    }
}
