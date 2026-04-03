//! Kelly criterion position sizing
//!
//! Part of the Hypothalamus region
//! Component: position_sizing
//!
//! The Kelly criterion determines optimal bet sizing based on edge and odds.
//! This implementation supports fractional Kelly for more conservative sizing,
//! multiple bet calculation methods, and bankroll management.

use crate::common::{Error, Result};
use std::collections::HashMap;

/// Kelly fraction configuration (how much of full Kelly to use)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KellyFraction {
    /// Full Kelly (maximum growth, high volatility)
    Full,
    /// Half Kelly (recommended for most traders)
    Half,
    /// Quarter Kelly (conservative)
    Quarter,
    /// Custom fraction (0.0 to 1.0)
    Custom(f64),
}

impl Default for KellyFraction {
    fn default() -> Self {
        KellyFraction::Half
    }
}

impl KellyFraction {
    /// Get the numeric fraction value
    pub fn value(&self) -> f64 {
        match self {
            KellyFraction::Full => 1.0,
            KellyFraction::Half => 0.5,
            KellyFraction::Quarter => 0.25,
            KellyFraction::Custom(f) => f.clamp(0.0, 1.0),
        }
    }
}

/// Trade statistics for Kelly calculation
#[derive(Debug, Clone)]
pub struct TradeStats {
    /// Total number of trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Average winning trade (as multiple of risk)
    pub avg_win: f64,
    /// Average losing trade (as multiple of risk, should be positive)
    pub avg_loss: f64,
    /// Optional: explicitly provided win rate (overrides calculation)
    pub explicit_win_rate: Option<f64>,
}

impl Default for TradeStats {
    fn default() -> Self {
        Self {
            total_trades: 0,
            winning_trades: 0,
            avg_win: 1.0,
            avg_loss: 1.0,
            explicit_win_rate: None,
        }
    }
}

impl TradeStats {
    /// Create from raw trade data
    pub fn new(total_trades: usize, winning_trades: usize, avg_win: f64, avg_loss: f64) -> Self {
        Self {
            total_trades,
            winning_trades,
            avg_win,
            avg_loss,
            explicit_win_rate: None,
        }
    }

    /// Create with explicit win rate
    pub fn with_win_rate(win_rate: f64, avg_win: f64, avg_loss: f64) -> Self {
        Self {
            total_trades: 100, // Placeholder for explicit rate
            winning_trades: (win_rate * 100.0) as usize,
            avg_win,
            avg_loss,
            explicit_win_rate: Some(win_rate),
        }
    }

    /// Calculate win rate
    pub fn win_rate(&self) -> f64 {
        if let Some(rate) = self.explicit_win_rate {
            return rate;
        }
        if self.total_trades == 0 {
            return 0.0;
        }
        self.winning_trades as f64 / self.total_trades as f64
    }

    /// Calculate loss rate
    pub fn loss_rate(&self) -> f64 {
        1.0 - self.win_rate()
    }

    /// Calculate profit factor (gross profits / gross losses)
    pub fn profit_factor(&self) -> f64 {
        if self.avg_loss == 0.0 || self.winning_trades == 0 {
            return 0.0;
        }
        let gross_profits = self.winning_trades as f64 * self.avg_win;
        let gross_losses = (self.total_trades - self.winning_trades) as f64 * self.avg_loss;
        if gross_losses == 0.0 {
            return f64::INFINITY;
        }
        gross_profits / gross_losses
    }

    /// Calculate expectancy (expected value per trade)
    pub fn expectancy(&self) -> f64 {
        let win_rate = self.win_rate();
        (win_rate * self.avg_win) - ((1.0 - win_rate) * self.avg_loss)
    }
}

/// Kelly criterion calculation result
#[derive(Debug, Clone)]
pub struct KellyResult {
    /// Full Kelly percentage (0.0 to 1.0+)
    pub full_kelly: f64,
    /// Adjusted Kelly (after applying fraction)
    pub adjusted_kelly: f64,
    /// Recommended position size as fraction of bankroll
    pub recommended_fraction: f64,
    /// Edge (expected value per unit risked)
    pub edge: f64,
    /// Odds (average win / average loss)
    pub odds: f64,
    /// Win probability used
    pub win_probability: f64,
    /// Whether the edge is positive
    pub has_edge: bool,
    /// Confidence level (based on sample size)
    pub confidence: f64,
    /// Warning messages if any
    pub warnings: Vec<String>,
}

impl Default for KellyResult {
    fn default() -> Self {
        Self {
            full_kelly: 0.0,
            adjusted_kelly: 0.0,
            recommended_fraction: 0.0,
            edge: 0.0,
            odds: 0.0,
            win_probability: 0.0,
            has_edge: false,
            confidence: 0.0,
            warnings: Vec::new(),
        }
    }
}

/// Configuration for Kelly criterion
#[derive(Debug, Clone)]
pub struct KellyConfig {
    /// Kelly fraction to use
    pub fraction: KellyFraction,
    /// Minimum number of trades for reliable calculation
    pub min_trades: usize,
    /// Maximum position size as fraction of bankroll
    pub max_position: f64,
    /// Minimum position size as fraction of bankroll
    pub min_position: f64,
    /// Apply additional safety margin
    pub safety_margin: f64,
    /// Adjust for correlation between positions
    pub correlation_adjustment: bool,
}

impl Default for KellyConfig {
    fn default() -> Self {
        Self {
            fraction: KellyFraction::Half,
            min_trades: 30,
            max_position: 0.25, // Max 25% of bankroll per position
            min_position: 0.01, // Min 1% of bankroll
            safety_margin: 0.9, // 10% safety buffer
            correlation_adjustment: true,
        }
    }
}

/// Symbol-specific Kelly data
#[derive(Debug, Clone)]
pub struct SymbolKelly {
    /// Symbol identifier
    pub symbol: String,
    /// Trade statistics for this symbol
    pub stats: TradeStats,
    /// Last calculated Kelly result
    pub last_result: Option<KellyResult>,
    /// Historical Kelly values (for trend analysis)
    pub kelly_history: Vec<f64>,
    /// Number of concurrent positions with this symbol
    pub concurrent_positions: usize,
}

impl SymbolKelly {
    pub fn new(symbol: String) -> Self {
        Self {
            symbol,
            stats: TradeStats::default(),
            last_result: None,
            kelly_history: Vec::new(),
            concurrent_positions: 0,
        }
    }
}

/// Kelly criterion position sizing
pub struct KellyCriterion {
    /// Configuration
    config: KellyConfig,
    /// Current bankroll
    bankroll: f64,
    /// Overall trade statistics
    overall_stats: TradeStats,
    /// Per-symbol statistics
    symbol_stats: HashMap<String, SymbolKelly>,
    /// Last overall calculation
    last_result: Option<KellyResult>,
    /// Total calculations performed
    calculations_count: usize,
}

impl Default for KellyCriterion {
    fn default() -> Self {
        Self::new()
    }
}

impl KellyCriterion {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self {
            config: KellyConfig::default(),
            bankroll: 100_000.0,
            overall_stats: TradeStats::default(),
            symbol_stats: HashMap::new(),
            last_result: None,
            calculations_count: 0,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: KellyConfig) -> Self {
        Self {
            config,
            bankroll: 100_000.0,
            overall_stats: TradeStats::default(),
            symbol_stats: HashMap::new(),
            last_result: None,
            calculations_count: 0,
        }
    }

    /// Set the current bankroll
    pub fn set_bankroll(&mut self, bankroll: f64) {
        self.bankroll = bankroll;
    }

    /// Get the current bankroll
    pub fn bankroll(&self) -> f64 {
        self.bankroll
    }

    /// Update overall trade statistics
    pub fn update_stats(&mut self, stats: TradeStats) {
        self.overall_stats = stats;
    }

    /// Record a completed trade
    pub fn record_trade(&mut self, symbol: &str, is_win: bool, win_amount: f64, loss_amount: f64) {
        // Update overall stats
        self.overall_stats.total_trades += 1;
        if is_win {
            self.overall_stats.winning_trades += 1;
            // Update rolling average win
            let n = self.overall_stats.winning_trades as f64;
            self.overall_stats.avg_win =
                ((self.overall_stats.avg_win * (n - 1.0)) + win_amount) / n;
        } else {
            // Update rolling average loss
            let losing_trades = self.overall_stats.total_trades - self.overall_stats.winning_trades;
            let n = losing_trades as f64;
            if n > 0.0 {
                self.overall_stats.avg_loss =
                    ((self.overall_stats.avg_loss * (n - 1.0)) + loss_amount) / n;
            }
        }

        // Update symbol-specific stats
        let symbol_kelly = self
            .symbol_stats
            .entry(symbol.to_string())
            .or_insert_with(|| SymbolKelly::new(symbol.to_string()));

        symbol_kelly.stats.total_trades += 1;
        if is_win {
            symbol_kelly.stats.winning_trades += 1;
            let n = symbol_kelly.stats.winning_trades as f64;
            symbol_kelly.stats.avg_win =
                ((symbol_kelly.stats.avg_win * (n - 1.0)) + win_amount) / n;
        } else {
            let losing_trades = symbol_kelly.stats.total_trades - symbol_kelly.stats.winning_trades;
            let n = losing_trades as f64;
            if n > 0.0 {
                symbol_kelly.stats.avg_loss =
                    ((symbol_kelly.stats.avg_loss * (n - 1.0)) + loss_amount) / n;
            }
        }
    }

    /// Calculate Kelly criterion from trade statistics
    pub fn calculate(&mut self, stats: &TradeStats) -> KellyResult {
        self.calculations_count += 1;
        let mut result = KellyResult::default();
        let mut warnings = Vec::new();

        // Check minimum trades
        if stats.total_trades < self.config.min_trades {
            warnings.push(format!(
                "Insufficient trades ({} < {}). Results may be unreliable.",
                stats.total_trades, self.config.min_trades
            ));
        }

        // Calculate win probability
        let win_prob = stats.win_rate();
        result.win_probability = win_prob;

        // Calculate odds (b in Kelly formula)
        // b = average_win / average_loss
        let odds = if stats.avg_loss > 0.0 {
            stats.avg_win / stats.avg_loss
        } else {
            warnings.push("Average loss is zero, cannot calculate odds".to_string());
            1.0
        };
        result.odds = odds;

        // Calculate edge
        // edge = (win_prob * odds) - (1 - win_prob)
        // edge = (win_prob * (1 + odds)) - 1
        let edge = (win_prob * (1.0 + odds)) - 1.0;
        result.edge = edge;
        result.has_edge = edge > 0.0;

        if !result.has_edge {
            warnings.push("No positive edge detected. Kelly suggests no bet.".to_string());
            result.warnings = warnings;
            return result;
        }

        // Calculate full Kelly
        // Kelly % = (p * b - q) / b
        // where p = win probability, q = loss probability, b = odds
        let q = 1.0 - win_prob;
        let full_kelly = if odds > 0.0 {
            (win_prob * odds - q) / odds
        } else {
            0.0
        };
        result.full_kelly = full_kelly;

        // Apply fraction
        let fraction = self.config.fraction.value();
        let adjusted = full_kelly * fraction;
        result.adjusted_kelly = adjusted;

        // Apply safety margin
        let with_safety = adjusted * self.config.safety_margin;

        // Clamp to min/max
        let recommended =
            with_safety
                .max(0.0)
                .min(self.config.max_position)
                .max(if with_safety > 0.0 {
                    self.config.min_position
                } else {
                    0.0
                });
        result.recommended_fraction = recommended;

        // Calculate confidence based on sample size
        // Using sqrt(n) / 10 as a simple confidence metric
        result.confidence = ((stats.total_trades as f64).sqrt() / 10.0).min(1.0);

        // Additional warnings
        if full_kelly > 0.5 {
            warnings.push(format!(
                "Full Kelly ({:.1}%) is very high. Using fractional Kelly is strongly recommended.",
                full_kelly * 100.0
            ));
        }

        if odds < 1.0 && win_prob < 0.6 {
            warnings
                .push("Low odds with moderate win rate. Consider improving strategy.".to_string());
        }

        result.warnings = warnings;
        self.last_result = Some(result.clone());
        result
    }

    /// Calculate Kelly for overall statistics
    pub fn calculate_overall(&mut self) -> KellyResult {
        let stats = self.overall_stats.clone();
        self.calculate(&stats)
    }

    /// Calculate Kelly for a specific symbol
    pub fn calculate_for_symbol(&mut self, symbol: &str) -> Result<KellyResult> {
        let symbol_kelly = self.symbol_stats.get_mut(symbol).ok_or_else(|| {
            Error::NotFound(format!("No statistics found for symbol: {}", symbol))
        })?;

        let stats = symbol_kelly.stats.clone();
        let result = self.calculate(&stats);

        // Store in symbol history
        symbol_kelly.last_result = Some(result.clone());
        symbol_kelly.kelly_history.push(result.adjusted_kelly);

        // Keep history bounded
        if symbol_kelly.kelly_history.len() > 100 {
            symbol_kelly.kelly_history.remove(0);
        }

        Ok(result)
    }

    /// Get recommended position size in dollars
    pub fn get_position_size(&self, kelly_result: &KellyResult) -> f64 {
        self.bankroll * kelly_result.recommended_fraction
    }

    /// Get recommended position size for a given risk amount
    pub fn get_position_size_for_risk(
        &self,
        kelly_result: &KellyResult,
        risk_per_unit: f64,
    ) -> f64 {
        if risk_per_unit <= 0.0 {
            return 0.0;
        }
        let dollar_risk = self.bankroll * kelly_result.recommended_fraction;
        dollar_risk / risk_per_unit
    }

    /// Calculate Kelly for a specific trade setup
    pub fn calculate_for_setup(
        &mut self,
        win_probability: f64,
        reward_risk_ratio: f64,
    ) -> KellyResult {
        // Create synthetic stats for this setup
        let stats = TradeStats::with_win_rate(win_probability, reward_risk_ratio, 1.0);
        self.calculate(&stats)
    }

    /// Adjust Kelly for portfolio of correlated positions
    pub fn adjust_for_correlation(&self, kelly_values: &[f64], correlations: &[f64]) -> Vec<f64> {
        if !self.config.correlation_adjustment || correlations.is_empty() {
            return kelly_values.to_vec();
        }

        // Calculate average correlation
        let avg_correlation: f64 = correlations.iter().sum::<f64>() / correlations.len() as f64;

        // Reduce position sizes based on correlation
        // Higher correlation = lower individual position sizes
        let adjustment = 1.0 / (1.0 + avg_correlation * (kelly_values.len() as f64 - 1.0));

        kelly_values.iter().map(|k| k * adjustment).collect()
    }

    /// Get statistics for a symbol
    pub fn get_symbol_stats(&self, symbol: &str) -> Option<&SymbolKelly> {
        self.symbol_stats.get(symbol)
    }

    /// Get all tracked symbols
    pub fn get_symbols(&self) -> Vec<&String> {
        self.symbol_stats.keys().collect()
    }

    /// Get last calculation result
    pub fn last_result(&self) -> Option<&KellyResult> {
        self.last_result.as_ref()
    }

    /// Get overall trade statistics
    pub fn overall_stats(&self) -> &TradeStats {
        &self.overall_stats
    }

    /// Get calculation count
    pub fn calculations_count(&self) -> usize {
        self.calculations_count
    }

    /// Main processing function
    pub fn process(&self) -> Result<()> {
        // Processing is done on-demand via calculate methods
        Ok(())
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();
        report.push_str("=== Kelly Criterion Summary ===\n");
        report.push_str(&format!("Bankroll: ${:.2}\n", self.bankroll));
        report.push_str(&format!(
            "Kelly Fraction: {:.0}%\n",
            self.config.fraction.value() * 100.0
        ));
        report.push_str(&format!(
            "Total Trades: {}\n",
            self.overall_stats.total_trades
        ));
        report.push_str(&format!(
            "Win Rate: {:.1}%\n",
            self.overall_stats.win_rate() * 100.0
        ));
        report.push_str(&format!("Avg Win: {:.2}\n", self.overall_stats.avg_win));
        report.push_str(&format!("Avg Loss: {:.2}\n", self.overall_stats.avg_loss));
        report.push_str(&format!(
            "Expectancy: {:.3}\n",
            self.overall_stats.expectancy()
        ));

        if let Some(ref result) = self.last_result {
            report.push_str(&format!("Full Kelly: {:.1}%\n", result.full_kelly * 100.0));
            report.push_str(&format!(
                "Recommended: {:.1}%\n",
                result.recommended_fraction * 100.0
            ));
            report.push_str(&format!(
                "Position Size: ${:.2}\n",
                self.get_position_size(result)
            ));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let instance = KellyCriterion::new();
        assert!(instance.process().is_ok());
    }

    #[test]
    fn test_kelly_fraction_values() {
        assert_eq!(KellyFraction::Full.value(), 1.0);
        assert_eq!(KellyFraction::Half.value(), 0.5);
        assert_eq!(KellyFraction::Quarter.value(), 0.25);
        assert_eq!(KellyFraction::Custom(0.33).value(), 0.33);
        // Test clamping
        assert_eq!(KellyFraction::Custom(1.5).value(), 1.0);
        assert_eq!(KellyFraction::Custom(-0.5).value(), 0.0);
    }

    #[test]
    fn test_trade_stats_win_rate() {
        let stats = TradeStats::new(100, 55, 2.0, 1.0);
        assert_eq!(stats.win_rate(), 0.55);
        assert_eq!(stats.loss_rate(), 0.45);
    }

    #[test]
    fn test_trade_stats_expectancy() {
        // 55% win rate, 2:1 reward/risk
        let stats = TradeStats::new(100, 55, 2.0, 1.0);
        // Expected: (0.55 * 2.0) - (0.45 * 1.0) = 1.1 - 0.45 = 0.65
        assert!((stats.expectancy() - 0.65).abs() < 0.001);
    }

    #[test]
    fn test_kelly_calculation_positive_edge() {
        let mut kelly = KellyCriterion::new();
        kelly.config.fraction = KellyFraction::Full;
        kelly.config.min_trades = 0; // Allow small sample for test

        // 60% win rate, 1:1 reward/risk
        let stats = TradeStats::new(100, 60, 1.0, 1.0);
        let result = kelly.calculate(&stats);

        assert!(result.has_edge);
        // Full Kelly = (0.6 * 1 - 0.4) / 1 = 0.2 = 20%
        assert!((result.full_kelly - 0.20).abs() < 0.01);
    }

    #[test]
    fn test_kelly_calculation_no_edge() {
        let mut kelly = KellyCriterion::new();
        kelly.config.min_trades = 0;

        // 40% win rate, 1:1 reward/risk (negative expectancy)
        let stats = TradeStats::new(100, 40, 1.0, 1.0);
        let result = kelly.calculate(&stats);

        assert!(!result.has_edge);
        assert_eq!(result.recommended_fraction, 0.0);
    }

    #[test]
    fn test_kelly_calculation_coin_flip() {
        let mut kelly = KellyCriterion::new();
        kelly.config.min_trades = 0;

        // 50% win rate, 1:1 (no edge)
        let stats = TradeStats::new(100, 50, 1.0, 1.0);
        let result = kelly.calculate(&stats);

        assert!(!result.has_edge);
        assert!((result.full_kelly).abs() < 0.01); // Should be ~0
    }

    #[test]
    fn test_kelly_with_high_odds() {
        let mut kelly = KellyCriterion::new();
        kelly.config.fraction = KellyFraction::Full;
        kelly.config.min_trades = 0;

        // 40% win rate but 3:1 reward/risk
        let stats = TradeStats::new(100, 40, 3.0, 1.0);
        let result = kelly.calculate(&stats);

        assert!(result.has_edge);
        // Expected edge = (0.4 * 4) - 1 = 0.6
        assert!((result.edge - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_fractional_kelly() {
        let mut kelly = KellyCriterion::new();
        kelly.config.min_trades = 0;
        kelly.config.safety_margin = 1.0; // No safety margin for test

        // 60% win rate, 1:1 = 20% full Kelly
        let stats = TradeStats::new(100, 60, 1.0, 1.0);

        // Test half Kelly
        kelly.config.fraction = KellyFraction::Half;
        let result = kelly.calculate(&stats);
        assert!((result.adjusted_kelly - 0.10).abs() < 0.01); // 10%

        // Test quarter Kelly
        kelly.config.fraction = KellyFraction::Quarter;
        let result = kelly.calculate(&stats);
        assert!((result.adjusted_kelly - 0.05).abs() < 0.01); // 5%
    }

    #[test]
    fn test_position_size_clamping() {
        let mut kelly = KellyCriterion::new();
        kelly.config.fraction = KellyFraction::Full;
        kelly.config.min_trades = 0;
        kelly.config.max_position = 0.10; // Max 10%
        kelly.config.safety_margin = 1.0;

        // 80% win rate, 2:1 = very high Kelly
        let stats = TradeStats::new(100, 80, 2.0, 1.0);
        let result = kelly.calculate(&stats);

        // Should be clamped to max_position
        assert!(result.recommended_fraction <= 0.10);
    }

    #[test]
    fn test_bankroll_position_size() {
        let mut kelly = KellyCriterion::new();
        kelly.set_bankroll(100_000.0);
        kelly.config.min_trades = 0;
        kelly.config.safety_margin = 1.0;
        kelly.config.fraction = KellyFraction::Half;

        let stats = TradeStats::new(100, 60, 1.0, 1.0);
        let result = kelly.calculate(&stats);

        let position_size = kelly.get_position_size(&result);
        // Half of 20% = 10% of $100k = $10,000
        assert!((position_size - 10_000.0).abs() < 100.0);
    }

    #[test]
    fn test_record_trade() {
        let mut kelly = KellyCriterion::new();

        kelly.record_trade("AAPL", true, 2.0, 0.0);
        kelly.record_trade("AAPL", true, 1.5, 0.0);
        kelly.record_trade("AAPL", false, 0.0, 1.0);

        assert_eq!(kelly.overall_stats.total_trades, 3);
        assert_eq!(kelly.overall_stats.winning_trades, 2);

        let aapl_stats = kelly.get_symbol_stats("AAPL").unwrap();
        assert_eq!(aapl_stats.stats.total_trades, 3);
    }

    #[test]
    fn test_calculate_for_setup() {
        let mut kelly = KellyCriterion::new();
        kelly.config.min_trades = 0;
        kelly.config.fraction = KellyFraction::Full;
        kelly.config.safety_margin = 1.0;

        // 60% win probability, 2:1 reward/risk
        let result = kelly.calculate_for_setup(0.60, 2.0);

        assert!(result.has_edge);
        assert!((result.win_probability - 0.60).abs() < 0.01);
        assert!((result.odds - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_correlation_adjustment() {
        let kelly = KellyCriterion::new();

        let kelly_values = vec![0.10, 0.10, 0.10];
        let correlations = vec![0.5, 0.5, 0.5]; // 50% correlation

        let adjusted = kelly.adjust_for_correlation(&kelly_values, &correlations);

        // With correlation, individual positions should be smaller
        for adj in &adjusted {
            assert!(*adj < 0.10);
        }
    }

    #[test]
    fn test_profit_factor() {
        let stats = TradeStats::new(100, 60, 1.5, 1.0);
        // Gross profits = 60 * 1.5 = 90
        // Gross losses = 40 * 1.0 = 40
        // PF = 90 / 40 = 2.25
        assert!((stats.profit_factor() - 2.25).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let mut kelly = KellyCriterion::new();
        kelly.set_bankroll(50_000.0);
        kelly.overall_stats = TradeStats::new(100, 55, 2.0, 1.0);
        kelly.calculate_overall();

        let summary = kelly.summary();
        assert!(summary.contains("50000"));
        assert!(summary.contains("100")); // Total trades
    }

    #[test]
    fn test_explicit_win_rate() {
        let stats = TradeStats::with_win_rate(0.65, 1.5, 1.0);
        assert_eq!(stats.win_rate(), 0.65);
    }

    #[test]
    fn test_edge_calculation() {
        let mut kelly = KellyCriterion::new();
        kelly.config.min_trades = 0;

        // 50% win rate, 2:1 odds
        // Edge = (0.5 * 3) - 1 = 0.5
        let stats = TradeStats::with_win_rate(0.50, 2.0, 1.0);
        let result = kelly.calculate(&stats);

        assert!((result.edge - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_confidence_scaling() {
        let mut kelly = KellyCriterion::new();

        // Low sample size = low confidence
        let stats1 = TradeStats::new(10, 6, 1.0, 1.0);
        let result1 = kelly.calculate(&stats1);
        assert!(result1.confidence < 0.5);

        // High sample size = high confidence
        let stats2 = TradeStats::new(1000, 600, 1.0, 1.0);
        let result2 = kelly.calculate(&stats2);
        assert!(result2.confidence > 0.9);
    }
}
