//! Execution Analytics
//!
//! This module provides comprehensive analytics for order execution quality,
//! including slippage tracking, implementation shortfall calculation, and
//! execution performance metrics.
//!
//! # Example
//!
//! ```rust
//! use vision::execution::analytics::{ExecutionAnalytics, ExecutionRecord};
//!
//! let mut analytics = ExecutionAnalytics::new();
//!
//! // Record executions
//! analytics.record_execution(ExecutionRecord {
//!     order_id: "ORD001".to_string(),
//!     quantity: 1000.0,
//!     execution_price: 100.5,
//!     benchmark_price: 100.0,
//!     timestamp: std::time::Instant::now(),
//!     side: Side::Buy,
//!     venue: "NYSE".to_string(),
//! });
//!
//! // Get analytics
//! let report = analytics.generate_report();
//! println!("Avg Slippage: {:.2} bps", report.average_slippage_bps);
//! println!("Total Cost: ${:.2}", report.total_cost);
//! ```

use std::collections::HashMap;
use std::time::Instant;

/// Order side
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

/// Execution record
#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    /// Order identifier
    pub order_id: String,
    /// Executed quantity
    pub quantity: f64,
    /// Actual execution price
    pub execution_price: f64,
    /// Benchmark price (e.g., arrival price, VWAP)
    pub benchmark_price: f64,
    /// Execution timestamp
    pub timestamp: Instant,
    /// Buy or sell side
    pub side: Side,
    /// Execution venue
    pub venue: String,
}

impl ExecutionRecord {
    /// Calculate slippage in basis points
    pub fn slippage_bps(&self) -> f64 {
        let price_diff = match self.side {
            Side::Buy => self.execution_price - self.benchmark_price,
            Side::Sell => self.benchmark_price - self.execution_price,
        };
        (price_diff / self.benchmark_price) * 10000.0
    }

    /// Calculate absolute cost (positive = cost, negative = savings)
    pub fn cost(&self) -> f64 {
        let price_diff = match self.side {
            Side::Buy => self.execution_price - self.benchmark_price,
            Side::Sell => self.benchmark_price - self.execution_price,
        };
        price_diff * self.quantity
    }

    /// Calculate cost in basis points
    pub fn cost_bps(&self) -> f64 {
        self.slippage_bps()
    }
}

/// Execution analytics tracker
pub struct ExecutionAnalytics {
    executions: Vec<ExecutionRecord>,
    orders: HashMap<String, Vec<ExecutionRecord>>,
}

impl ExecutionAnalytics {
    /// Create a new execution analytics tracker
    pub fn new() -> Self {
        Self {
            executions: Vec::new(),
            orders: HashMap::new(),
        }
    }

    /// Record an execution
    pub fn record_execution(&mut self, record: ExecutionRecord) {
        let order_id = record.order_id.clone();
        self.executions.push(record.clone());
        self.orders
            .entry(order_id)
            .or_insert_with(Vec::new)
            .push(record);
    }

    /// Get all executions
    pub fn executions(&self) -> &[ExecutionRecord] {
        &self.executions
    }

    /// Get executions for a specific order
    pub fn executions_for_order(&self, order_id: &str) -> Option<&[ExecutionRecord]> {
        self.orders.get(order_id).map(|v| v.as_slice())
    }

    /// Calculate average slippage in basis points
    pub fn average_slippage_bps(&self) -> f64 {
        if self.executions.is_empty() {
            return 0.0;
        }

        let total: f64 = self.executions.iter().map(|e| e.slippage_bps()).sum();
        total / self.executions.len() as f64
    }

    /// Calculate volume-weighted average slippage
    pub fn vwap_slippage_bps(&self) -> f64 {
        let total_value: f64 = self
            .executions
            .iter()
            .map(|e| e.slippage_bps() * e.quantity)
            .sum();

        let total_quantity: f64 = self.executions.iter().map(|e| e.quantity).sum();

        if total_quantity > 0.0 {
            total_value / total_quantity
        } else {
            0.0
        }
    }

    /// Calculate total implementation cost
    pub fn total_cost(&self) -> f64 {
        self.executions.iter().map(|e| e.cost()).sum()
    }

    /// Calculate implementation shortfall percentage
    pub fn implementation_shortfall_pct(&self) -> f64 {
        let total_notional: f64 = self
            .executions
            .iter()
            .map(|e| e.quantity * e.benchmark_price)
            .sum();

        if total_notional > 0.0 {
            (self.total_cost() / total_notional) * 100.0
        } else {
            0.0
        }
    }

    /// Calculate execution quality score (0-100, higher is better)
    pub fn quality_score(&self) -> f64 {
        if self.executions.is_empty() {
            return 50.0; // Neutral score
        }

        let slippage = self.vwap_slippage_bps().abs();

        // Score based on slippage (lower is better)
        // 0 bps = 100, 50 bps = 50, 100+ bps = 0
        let score = (100.0 - slippage.min(100.0)).max(0.0);

        score
    }

    /// Get statistics by venue
    pub fn venue_statistics(&self) -> HashMap<String, VenueStats> {
        let mut stats: HashMap<String, VenueStats> = HashMap::new();

        for execution in &self.executions {
            let venue_stats = stats
                .entry(execution.venue.clone())
                .or_insert_with(|| VenueStats {
                    venue: execution.venue.clone(),
                    execution_count: 0,
                    total_quantity: 0.0,
                    total_cost: 0.0,
                    average_slippage_bps: 0.0,
                });

            venue_stats.execution_count += 1;
            venue_stats.total_quantity += execution.quantity;
            venue_stats.total_cost += execution.cost();
        }

        // Calculate average slippage per venue
        for (venue, stat) in &mut stats {
            let venue_executions: Vec<&ExecutionRecord> = self
                .executions
                .iter()
                .filter(|e| &e.venue == venue)
                .collect();

            if !venue_executions.is_empty() {
                let total_slippage: f64 = venue_executions
                    .iter()
                    .map(|e| e.slippage_bps() * e.quantity)
                    .sum();
                stat.average_slippage_bps = total_slippage / stat.total_quantity;
            }
        }

        stats
    }

    /// Generate comprehensive execution report
    pub fn generate_report(&self) -> ExecutionReport {
        let total_executions = self.executions.len();
        let total_quantity: f64 = self.executions.iter().map(|e| e.quantity).sum();

        let buy_executions = self
            .executions
            .iter()
            .filter(|e| e.side == Side::Buy)
            .count();
        let sell_executions = total_executions - buy_executions;

        let average_slippage_bps = self.average_slippage_bps();
        let vwap_slippage_bps = self.vwap_slippage_bps();
        let total_cost = self.total_cost();
        let implementation_shortfall = self.implementation_shortfall_pct();
        let quality_score = self.quality_score();

        // Calculate min/max slippage
        let mut min_slippage_bps = f64::INFINITY;
        let mut max_slippage_bps = f64::NEG_INFINITY;

        for execution in &self.executions {
            let slippage = execution.slippage_bps();
            min_slippage_bps = min_slippage_bps.min(slippage);
            max_slippage_bps = max_slippage_bps.max(slippage);
        }

        if self.executions.is_empty() {
            min_slippage_bps = 0.0;
            max_slippage_bps = 0.0;
        }

        let venue_stats = self.venue_statistics();

        ExecutionReport {
            total_executions,
            buy_executions,
            sell_executions,
            total_quantity,
            average_slippage_bps,
            vwap_slippage_bps,
            min_slippage_bps,
            max_slippage_bps,
            total_cost,
            implementation_shortfall,
            quality_score,
            venue_stats,
        }
    }

    /// Reset analytics
    pub fn reset(&mut self) {
        self.executions.clear();
        self.orders.clear();
    }

    /// Get number of unique orders
    pub fn unique_orders(&self) -> usize {
        self.orders.len()
    }

    /// Calculate average execution price for an order
    pub fn average_execution_price(&self, order_id: &str) -> Option<f64> {
        let executions = self.executions_for_order(order_id)?;

        if executions.is_empty() {
            return None;
        }

        let total_value: f64 = executions
            .iter()
            .map(|e| e.execution_price * e.quantity)
            .sum();
        let total_quantity: f64 = executions.iter().map(|e| e.quantity).sum();

        if total_quantity > 0.0 {
            Some(total_value / total_quantity)
        } else {
            None
        }
    }

    /// Get executions within time range
    pub fn executions_in_range(&self, start: Instant, end: Instant) -> Vec<&ExecutionRecord> {
        self.executions
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .collect()
    }
}

impl Default for ExecutionAnalytics {
    fn default() -> Self {
        Self::new()
    }
}

/// Venue-specific statistics
#[derive(Debug, Clone)]
pub struct VenueStats {
    pub venue: String,
    pub execution_count: usize,
    pub total_quantity: f64,
    pub total_cost: f64,
    pub average_slippage_bps: f64,
}

/// Comprehensive execution report
#[derive(Debug, Clone)]
pub struct ExecutionReport {
    /// Total number of executions
    pub total_executions: usize,
    /// Number of buy executions
    pub buy_executions: usize,
    /// Number of sell executions
    pub sell_executions: usize,
    /// Total quantity executed
    pub total_quantity: f64,
    /// Average slippage in basis points
    pub average_slippage_bps: f64,
    /// Volume-weighted average slippage in bps
    pub vwap_slippage_bps: f64,
    /// Minimum slippage observed
    pub min_slippage_bps: f64,
    /// Maximum slippage observed
    pub max_slippage_bps: f64,
    /// Total implementation cost
    pub total_cost: f64,
    /// Implementation shortfall percentage
    pub implementation_shortfall: f64,
    /// Execution quality score (0-100)
    pub quality_score: f64,
    /// Statistics by venue
    pub venue_stats: HashMap<String, VenueStats>,
}

impl ExecutionReport {
    /// Print a formatted report
    pub fn print(&self) {
        println!("=== Execution Analytics Report ===");
        println!("Total Executions: {}", self.total_executions);
        println!("  Buy: {}", self.buy_executions);
        println!("  Sell: {}", self.sell_executions);
        println!("Total Quantity: {:.2}", self.total_quantity);
        println!();
        println!("Slippage Analysis:");
        println!("  Average: {:.2} bps", self.average_slippage_bps);
        println!("  VWAP: {:.2} bps", self.vwap_slippage_bps);
        println!("  Min: {:.2} bps", self.min_slippage_bps);
        println!("  Max: {:.2} bps", self.max_slippage_bps);
        println!();
        println!("Cost Analysis:");
        println!("  Total Cost: ${:.2}", self.total_cost);
        println!(
            "  Implementation Shortfall: {:.4}%",
            self.implementation_shortfall
        );
        println!();
        println!("Quality Score: {:.1}/100", self.quality_score);
        println!();

        if !self.venue_stats.is_empty() {
            println!("Venue Statistics:");
            let mut venues: Vec<_> = self.venue_stats.values().collect();
            venues.sort_by(|a, b| a.venue.cmp(&b.venue));

            for venue in venues {
                println!("  {}:", venue.venue);
                println!("    Executions: {}", venue.execution_count);
                println!("    Quantity: {:.2}", venue.total_quantity);
                println!("    Avg Slippage: {:.2} bps", venue.average_slippage_bps);
                println!("    Total Cost: ${:.2}", venue.total_cost);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn create_test_execution(
        order_id: &str,
        quantity: f64,
        execution_price: f64,
        benchmark_price: f64,
        side: Side,
        venue: &str,
    ) -> ExecutionRecord {
        ExecutionRecord {
            order_id: order_id.to_string(),
            quantity,
            execution_price,
            benchmark_price,
            timestamp: Instant::now(),
            side,
            venue: venue.to_string(),
        }
    }

    #[test]
    fn test_slippage_calculation_buy() {
        let record = create_test_execution("ORD1", 100.0, 100.5, 100.0, Side::Buy, "NYSE");

        // Buy at 100.5 vs benchmark 100 = 0.5 / 100 * 10000 = 50 bps
        assert!((record.slippage_bps() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_slippage_calculation_sell() {
        let record = create_test_execution("ORD1", 100.0, 99.5, 100.0, Side::Sell, "NYSE");

        // Sell at 99.5 vs benchmark 100 = 50 bps adverse
        assert!((record.slippage_bps() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_calculation() {
        let record = create_test_execution("ORD1", 1000.0, 100.5, 100.0, Side::Buy, "NYSE");

        // Cost = (100.5 - 100.0) * 1000 = 500
        assert!((record.cost() - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_analytics_average_slippage() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            100.0,
            100.5,
            100.0,
            Side::Buy,
            "NYSE",
        ));
        analytics.record_execution(create_test_execution(
            "ORD2",
            100.0,
            101.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let avg_slippage = analytics.average_slippage_bps();
        // (50 + 100) / 2 = 75 bps
        assert!((avg_slippage - 75.0).abs() < 0.01);
    }

    #[test]
    fn test_vwap_slippage() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            100.0,
            100.5,
            100.0,
            Side::Buy,
            "NYSE",
        ));
        analytics.record_execution(create_test_execution(
            "ORD2",
            300.0,
            101.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let vwap_slippage = analytics.vwap_slippage_bps();
        // (50 * 100 + 100 * 300) / 400 = 35000 / 400 = 87.5 bps
        assert!((vwap_slippage - 87.5).abs() < 0.01);
    }

    #[test]
    fn test_total_cost() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            1000.0,
            100.5,
            100.0,
            Side::Buy,
            "NYSE",
        ));
        analytics.record_execution(create_test_execution(
            "ORD2",
            2000.0,
            101.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let total_cost = analytics.total_cost();
        // 500 + 2000 = 2500
        assert!((total_cost - 2500.0).abs() < 0.01);
    }

    #[test]
    fn test_implementation_shortfall() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            1000.0,
            101.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let shortfall = analytics.implementation_shortfall_pct();
        // Cost = 1000, Notional = 100000, Shortfall = 1%
        assert!((shortfall - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quality_score() {
        let mut analytics = ExecutionAnalytics::new();

        // Perfect execution
        analytics.record_execution(create_test_execution(
            "ORD1",
            1000.0,
            100.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let score = analytics.quality_score();
        assert!((score - 100.0).abs() < 0.01);

        analytics.reset();

        // 50 bps slippage
        analytics.record_execution(create_test_execution(
            "ORD1",
            1000.0,
            100.5,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let score = analytics.quality_score();
        assert!((score - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_venue_statistics() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            1000.0,
            100.5,
            100.0,
            Side::Buy,
            "NYSE",
        ));
        analytics.record_execution(create_test_execution(
            "ORD2",
            500.0,
            101.0,
            100.0,
            Side::Buy,
            "NASDAQ",
        ));
        analytics.record_execution(create_test_execution(
            "ORD3",
            1500.0,
            100.2,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let venue_stats = analytics.venue_statistics();

        assert_eq!(venue_stats.len(), 2);
        assert!(venue_stats.contains_key("NYSE"));
        assert!(venue_stats.contains_key("NASDAQ"));

        let nyse_stats = &venue_stats["NYSE"];
        assert_eq!(nyse_stats.execution_count, 2);
        assert_eq!(nyse_stats.total_quantity, 2500.0);
    }

    #[test]
    fn test_generate_report() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            1000.0,
            100.5,
            100.0,
            Side::Buy,
            "NYSE",
        ));
        analytics.record_execution(create_test_execution(
            "ORD2",
            500.0,
            99.5,
            100.0,
            Side::Sell,
            "NYSE",
        ));

        let report = analytics.generate_report();

        assert_eq!(report.total_executions, 2);
        assert_eq!(report.buy_executions, 1);
        assert_eq!(report.sell_executions, 1);
        assert_eq!(report.total_quantity, 1500.0);
        assert!(report.quality_score > 0.0 && report.quality_score <= 100.0);
    }

    #[test]
    fn test_reset() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            1000.0,
            100.5,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        assert_eq!(analytics.executions().len(), 1);

        analytics.reset();

        assert_eq!(analytics.executions().len(), 0);
        assert_eq!(analytics.unique_orders(), 0);
    }

    #[test]
    fn test_average_execution_price() {
        let mut analytics = ExecutionAnalytics::new();

        analytics.record_execution(create_test_execution(
            "ORD1",
            100.0,
            100.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));
        analytics.record_execution(create_test_execution(
            "ORD1",
            100.0,
            102.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let avg_price = analytics.average_execution_price("ORD1").unwrap();
        assert!((avg_price - 101.0).abs() < 0.01);
    }

    #[test]
    fn test_executions_in_range() {
        let mut analytics = ExecutionAnalytics::new();

        let start = Instant::now();
        analytics.record_execution(create_test_execution(
            "ORD1",
            100.0,
            100.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        std::thread::sleep(Duration::from_millis(10));
        let mid = Instant::now();

        analytics.record_execution(create_test_execution(
            "ORD2",
            100.0,
            100.0,
            100.0,
            Side::Buy,
            "NYSE",
        ));

        let end = Instant::now();

        let in_range = analytics.executions_in_range(start, mid);
        assert_eq!(in_range.len(), 1);

        let all = analytics.executions_in_range(start, end);
        assert_eq!(all.len(), 2);
    }
}
