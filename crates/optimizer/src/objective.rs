//! Objective Function and Scoring
//!
//! This module defines the multi-objective scoring function used to evaluate
//! trading strategy performance during optimization.
//!
//! The scoring function balances multiple objectives:
//! - Total return (higher is better)
//! - Sharpe ratio (risk-adjusted return)
//! - Win rate (consistency)
//! - Maximum drawdown (risk, lower is better)
//! - Profit factor (reward/risk ratio)
//! - Trade frequency (reasonable activity level)

use crate::backtester::BacktestResult;
use serde::{Deserialize, Serialize};

/// Default scoring weights
pub const DEFAULT_RETURN_WEIGHT: f64 = 1.5;
pub const DEFAULT_SHARPE_WEIGHT: f64 = 2.0;
pub const DEFAULT_WIN_RATE_WEIGHT: f64 = 0.5;
pub const DEFAULT_DRAWDOWN_WEIGHT: f64 = 0.8;
pub const DEFAULT_PROFIT_FACTOR_WEIGHT: f64 = 1.0;
pub const DEFAULT_FREQUENCY_WEIGHT: f64 = 0.3;

/// Minimum trades required for valid scoring
pub const MIN_TRADES_FOR_SCORING: usize = 10;

/// Ideal trades per day range
pub const MIN_TRADES_PER_DAY: f64 = 0.5;
pub const MAX_TRADES_PER_DAY: f64 = 5.0;

/// Weights for multi-objective scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringWeights {
    /// Weight for total return component
    pub return_weight: f64,

    /// Weight for Sharpe ratio component
    pub sharpe_weight: f64,

    /// Weight for win rate component (penalizes <50%)
    pub win_rate_weight: f64,

    /// Weight for drawdown penalty (negative contribution)
    pub drawdown_weight: f64,

    /// Weight for profit factor bonus
    pub profit_factor_weight: f64,

    /// Weight for trade frequency component
    pub frequency_weight: f64,

    /// Minimum required trades for valid score
    pub min_trades: usize,

    /// Preferred minimum trades per day
    pub min_trades_per_day: f64,

    /// Preferred maximum trades per day
    pub max_trades_per_day: f64,

    /// Maximum consecutive losses threshold
    pub max_consecutive_losses: usize,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            return_weight: DEFAULT_RETURN_WEIGHT,
            sharpe_weight: DEFAULT_SHARPE_WEIGHT,
            win_rate_weight: DEFAULT_WIN_RATE_WEIGHT,
            drawdown_weight: DEFAULT_DRAWDOWN_WEIGHT,
            profit_factor_weight: DEFAULT_PROFIT_FACTOR_WEIGHT,
            frequency_weight: DEFAULT_FREQUENCY_WEIGHT,
            min_trades: MIN_TRADES_FOR_SCORING,
            min_trades_per_day: MIN_TRADES_PER_DAY,
            max_trades_per_day: MAX_TRADES_PER_DAY,
            max_consecutive_losses: 5,
        }
    }
}

impl ScoringWeights {
    /// Create new scoring weights
    pub fn new() -> Self {
        Self::default()
    }

    /// Create weights focused on return maximization
    pub fn return_focused() -> Self {
        Self {
            return_weight: 3.0,
            sharpe_weight: 1.0,
            win_rate_weight: 0.3,
            drawdown_weight: 0.5,
            profit_factor_weight: 0.5,
            frequency_weight: 0.2,
            ..Default::default()
        }
    }

    /// Create weights focused on risk-adjusted returns
    pub fn risk_adjusted() -> Self {
        Self {
            return_weight: 1.0,
            sharpe_weight: 3.0,
            win_rate_weight: 0.5,
            drawdown_weight: 1.5,
            profit_factor_weight: 1.5,
            frequency_weight: 0.3,
            ..Default::default()
        }
    }

    /// Create weights focused on consistency
    pub fn consistency_focused() -> Self {
        Self {
            return_weight: 1.0,
            sharpe_weight: 1.5,
            win_rate_weight: 2.0,
            drawdown_weight: 1.0,
            profit_factor_weight: 2.0,
            frequency_weight: 0.5,
            max_consecutive_losses: 3,
            ..Default::default()
        }
    }

    /// Create balanced weights (default but explicitly named)
    pub fn balanced() -> Self {
        Self::default()
    }

    /// Builder-style method to set return weight
    pub fn with_return_weight(mut self, weight: f64) -> Self {
        self.return_weight = weight;
        self
    }

    /// Builder-style method to set sharpe weight
    pub fn with_sharpe_weight(mut self, weight: f64) -> Self {
        self.sharpe_weight = weight;
        self
    }

    /// Builder-style method to set win rate weight
    pub fn with_win_rate_weight(mut self, weight: f64) -> Self {
        self.win_rate_weight = weight;
        self
    }

    /// Builder-style method to set drawdown weight
    pub fn with_drawdown_weight(mut self, weight: f64) -> Self {
        self.drawdown_weight = weight;
        self
    }

    /// Builder-style method to set profit factor weight
    pub fn with_profit_factor_weight(mut self, weight: f64) -> Self {
        self.profit_factor_weight = weight;
        self
    }

    /// Builder-style method to set frequency weight
    pub fn with_frequency_weight(mut self, weight: f64) -> Self {
        self.frequency_weight = weight;
        self
    }

    /// Builder-style method to set min trades
    pub fn with_min_trades(mut self, min: usize) -> Self {
        self.min_trades = min;
        self
    }

    /// Builder-style method to set trades per day range
    pub fn with_trades_per_day_range(mut self, min: f64, max: f64) -> Self {
        self.min_trades_per_day = min;
        self.max_trades_per_day = max;
        self
    }
}

/// Objective function for evaluating backtest results
#[derive(Debug, Clone)]
pub struct ObjectiveFunction {
    /// Scoring weights
    weights: ScoringWeights,
}

impl Default for ObjectiveFunction {
    fn default() -> Self {
        Self::new(ScoringWeights::default())
    }
}

impl ObjectiveFunction {
    /// Create a new objective function with the given weights
    pub fn new(weights: ScoringWeights) -> Self {
        Self { weights }
    }

    /// Create with return-focused weights
    pub fn return_focused() -> Self {
        Self::new(ScoringWeights::return_focused())
    }

    /// Create with risk-adjusted weights
    pub fn risk_adjusted() -> Self {
        Self::new(ScoringWeights::risk_adjusted())
    }

    /// Create with consistency-focused weights
    pub fn consistency_focused() -> Self {
        Self::new(ScoringWeights::consistency_focused())
    }

    /// Get the scoring weights
    pub fn weights(&self) -> &ScoringWeights {
        &self.weights
    }

    /// Calculate the objective score for a backtest result
    ///
    /// Returns a single score where higher is better.
    /// Invalid results (too few trades, etc.) return large negative values.
    pub fn score(&self, result: &BacktestResult) -> f64 {
        // Check minimum trade requirement
        if result.total_trades < self.weights.min_trades {
            return -500.0;
        }

        // Calculate individual components
        let return_score = self.return_score(result);
        let sharpe_score = self.sharpe_score(result);
        let win_rate_score = self.win_rate_score(result);
        let drawdown_penalty = self.drawdown_penalty(result);
        let profit_factor_score = self.profit_factor_score(result);
        let frequency_score = self.frequency_score(result);
        let consistency_bonus = self.consistency_bonus(result);

        // Combine scores
        return_score
            + sharpe_score
            + win_rate_score
            + drawdown_penalty // Already negative
            + profit_factor_score
            + frequency_score
            + consistency_bonus
    }

    /// Calculate return component of score
    fn return_score(&self, result: &BacktestResult) -> f64 {
        result.total_pnl_pct * self.weights.return_weight
    }

    /// Calculate Sharpe ratio component
    fn sharpe_score(&self, result: &BacktestResult) -> f64 {
        // Sharpe can be negative, but we cap the penalty
        let capped_sharpe = result.sharpe_ratio.clamp(-2.0, 5.0);
        capped_sharpe * self.weights.sharpe_weight * 10.0 // Scale up for impact
    }

    /// Calculate win rate component
    fn win_rate_score(&self, result: &BacktestResult) -> f64 {
        // Penalize win rate below 50%, bonus above
        let win_rate_delta = result.win_rate - 50.0;
        win_rate_delta * self.weights.win_rate_weight
    }

    /// Calculate drawdown penalty (negative contribution)
    fn drawdown_penalty(&self, result: &BacktestResult) -> f64 {
        // Higher drawdown = more negative score
        -result.max_drawdown_pct * self.weights.drawdown_weight
    }

    /// Calculate profit factor bonus
    fn profit_factor_score(&self, result: &BacktestResult) -> f64 {
        let pf = result.profit_factor;

        if pf > 1.5 {
            // Strong profit factor bonus
            (pf - 1.0).min(2.0) * self.weights.profit_factor_weight * 10.0
        } else if pf > 1.0 {
            // Slight bonus for profitable
            (pf - 1.0) * self.weights.profit_factor_weight * 5.0
        } else {
            // Penalty for losing strategy
            (pf - 1.0) * self.weights.profit_factor_weight * 20.0
        }
    }

    /// Calculate trade frequency component
    fn frequency_score(&self, result: &BacktestResult) -> f64 {
        let tpd = result.trades_per_day;

        if tpd >= self.weights.min_trades_per_day && tpd <= self.weights.max_trades_per_day {
            // Ideal range bonus
            self.weights.frequency_weight * 10.0
        } else if tpd < self.weights.min_trades_per_day * 0.5 {
            // Too few trades penalty
            -self.weights.frequency_weight * 30.0
        } else if tpd > self.weights.max_trades_per_day * 2.0 {
            // Too many trades penalty (overtrading)
            -self.weights.frequency_weight * 20.0
        } else {
            // Slightly outside range
            0.0
        }
    }

    /// Calculate consistency bonus based on consecutive losses
    fn consistency_bonus(&self, result: &BacktestResult) -> f64 {
        if result.max_consecutive_losses <= self.weights.max_consecutive_losses {
            15.0 // Good consistency bonus
        } else if result.max_consecutive_losses > self.weights.max_consecutive_losses * 2 {
            -15.0 // Poor consistency penalty
        } else {
            0.0
        }
    }

    /// Get a breakdown of score components for debugging/analysis
    pub fn score_breakdown(&self, result: &BacktestResult) -> ScoreBreakdown {
        ScoreBreakdown {
            total: self.score(result),
            return_component: self.return_score(result),
            sharpe_component: self.sharpe_score(result),
            win_rate_component: self.win_rate_score(result),
            drawdown_component: self.drawdown_penalty(result),
            profit_factor_component: self.profit_factor_score(result),
            frequency_component: self.frequency_score(result),
            consistency_component: self.consistency_bonus(result),
        }
    }
}

/// Breakdown of score components for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    /// Total combined score
    pub total: f64,

    /// Return component contribution
    pub return_component: f64,

    /// Sharpe ratio component contribution
    pub sharpe_component: f64,

    /// Win rate component contribution
    pub win_rate_component: f64,

    /// Drawdown component contribution (usually negative)
    pub drawdown_component: f64,

    /// Profit factor component contribution
    pub profit_factor_component: f64,

    /// Trade frequency component contribution
    pub frequency_component: f64,

    /// Consistency component contribution
    pub consistency_component: f64,
}

impl ScoreBreakdown {
    /// Get the largest positive contributor
    pub fn largest_positive(&self) -> (&'static str, f64) {
        let components = [
            ("return", self.return_component),
            ("sharpe", self.sharpe_component),
            ("win_rate", self.win_rate_component),
            ("profit_factor", self.profit_factor_component),
            ("frequency", self.frequency_component),
            ("consistency", self.consistency_component),
        ];

        components
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or(("none", 0.0))
    }

    /// Get the largest negative contributor
    pub fn largest_negative(&self) -> (&'static str, f64) {
        let components = [
            ("return", self.return_component),
            ("sharpe", self.sharpe_component),
            ("win_rate", self.win_rate_component),
            ("drawdown", self.drawdown_component),
            ("profit_factor", self.profit_factor_component),
            ("frequency", self.frequency_component),
            ("consistency", self.consistency_component),
        ];

        components
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap_or(("none", 0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_good_result() -> BacktestResult {
        BacktestResult {
            total_trades: 50,
            winning_trades: 30,
            losing_trades: 20,
            total_pnl_pct: 15.0,
            max_drawdown_pct: 5.0,
            win_rate: 60.0,
            profit_factor: 1.8,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            trades_per_day: 2.0,
            avg_trade_duration_minutes: 45.0,
            max_consecutive_wins: 5,
            max_consecutive_losses: 3,
            avg_winner_pct: 2.0,
            avg_loser_pct: 1.5,
            largest_winner_pct: 5.0,
            largest_loser_pct: 3.0,
        }
    }

    fn create_poor_result() -> BacktestResult {
        BacktestResult {
            total_trades: 50,
            winning_trades: 15,
            losing_trades: 35,
            total_pnl_pct: -10.0,
            max_drawdown_pct: 20.0,
            win_rate: 30.0,
            profit_factor: 0.5,
            sharpe_ratio: -0.5,
            sortino_ratio: -0.3,
            trades_per_day: 15.0, // Overtrading
            avg_trade_duration_minutes: 10.0,
            max_consecutive_wins: 2,
            max_consecutive_losses: 12,
            avg_winner_pct: 1.0,
            avg_loser_pct: 2.0,
            largest_winner_pct: 2.0,
            largest_loser_pct: 8.0,
        }
    }

    fn create_few_trades_result() -> BacktestResult {
        BacktestResult {
            total_trades: 5, // Below minimum
            ..create_good_result()
        }
    }

    #[test]
    fn test_default_weights() {
        let weights = ScoringWeights::default();
        assert_eq!(weights.return_weight, DEFAULT_RETURN_WEIGHT);
        assert_eq!(weights.sharpe_weight, DEFAULT_SHARPE_WEIGHT);
        assert_eq!(weights.min_trades, MIN_TRADES_FOR_SCORING);
    }

    #[test]
    fn test_preset_weights() {
        let return_focused = ScoringWeights::return_focused();
        assert!(return_focused.return_weight > return_focused.sharpe_weight);

        let risk_adjusted = ScoringWeights::risk_adjusted();
        assert!(risk_adjusted.sharpe_weight > risk_adjusted.return_weight);

        let consistency = ScoringWeights::consistency_focused();
        assert!(consistency.win_rate_weight > consistency.return_weight);
    }

    #[test]
    fn test_weights_builder() {
        let weights = ScoringWeights::new()
            .with_return_weight(5.0)
            .with_sharpe_weight(3.0)
            .with_min_trades(20);

        assert_eq!(weights.return_weight, 5.0);
        assert_eq!(weights.sharpe_weight, 3.0);
        assert_eq!(weights.min_trades, 20);
    }

    #[test]
    fn test_good_result_positive_score() {
        let objective = ObjectiveFunction::default();
        let result = create_good_result();
        let score = objective.score(&result);

        assert!(
            score > 0.0,
            "Good result should have positive score: {}",
            score
        );
    }

    #[test]
    fn test_poor_result_lower_score() {
        let objective = ObjectiveFunction::default();
        let good = create_good_result();
        let poor = create_poor_result();

        let good_score = objective.score(&good);
        let poor_score = objective.score(&poor);

        assert!(
            good_score > poor_score,
            "Good result ({}) should score higher than poor result ({})",
            good_score,
            poor_score
        );
    }

    #[test]
    fn test_few_trades_penalty() {
        let objective = ObjectiveFunction::default();
        let result = create_few_trades_result();
        let score = objective.score(&result);

        assert_eq!(score, -500.0, "Few trades should return penalty score");
    }

    #[test]
    fn test_score_breakdown() {
        let objective = ObjectiveFunction::default();
        let result = create_good_result();
        let breakdown = objective.score_breakdown(&result);

        assert!(breakdown.return_component > 0.0);
        assert!(breakdown.sharpe_component > 0.0);
        assert!(breakdown.win_rate_component > 0.0); // 60% > 50%
        assert!(breakdown.drawdown_component < 0.0); // Penalty
        assert!(breakdown.profit_factor_component > 0.0);
        assert!(breakdown.consistency_component > 0.0); // 3 consecutive losses is good

        // Total should match score
        let score = objective.score(&result);
        assert!(
            (breakdown.total - score).abs() < 0.001,
            "Breakdown total should match score"
        );
    }

    #[test]
    fn test_largest_contributors() {
        let objective = ObjectiveFunction::default();
        let result = create_good_result();
        let breakdown = objective.score_breakdown(&result);

        let (_, largest_pos) = breakdown.largest_positive();
        assert!(largest_pos > 0.0);

        let (name, largest_neg) = breakdown.largest_negative();
        assert_eq!(name, "drawdown"); // Drawdown is typically negative
        assert!(largest_neg < 0.0);
    }

    #[test]
    fn test_different_objective_functions() {
        let result = create_good_result();

        let default_score = ObjectiveFunction::default().score(&result);
        let return_score = ObjectiveFunction::return_focused().score(&result);
        let risk_score = ObjectiveFunction::risk_adjusted().score(&result);

        // All should be positive for good result
        assert!(default_score > 0.0);
        assert!(return_score > 0.0);
        assert!(risk_score > 0.0);

        // Scores should differ based on weights
        assert!(
            (default_score - return_score).abs() > 0.1,
            "Different weights should produce different scores"
        );
    }

    #[test]
    fn test_overtrading_penalty() {
        let objective = ObjectiveFunction::default();
        let mut result = create_good_result();

        // Normal frequency
        result.trades_per_day = 2.0;
        let normal_score = objective.score(&result);

        // Overtrading
        result.trades_per_day = 15.0;
        let overtrading_score = objective.score(&result);

        assert!(
            normal_score > overtrading_score,
            "Overtrading should be penalized"
        );
    }

    #[test]
    fn test_undertrading_penalty() {
        let objective = ObjectiveFunction::default();
        let mut result = create_good_result();

        // Normal frequency
        result.trades_per_day = 2.0;
        let normal_score = objective.score(&result);

        // Undertrading
        result.trades_per_day = 0.1;
        let undertrading_score = objective.score(&result);

        assert!(
            normal_score > undertrading_score,
            "Undertrading should be penalized"
        );
    }

    #[test]
    fn test_consecutive_losses_impact() {
        let objective = ObjectiveFunction::default();
        let mut result = create_good_result();

        // Few consecutive losses
        result.max_consecutive_losses = 2;
        let few_losses_score = objective.score(&result);

        // Many consecutive losses
        result.max_consecutive_losses = 15;
        let many_losses_score = objective.score(&result);

        assert!(
            few_losses_score > many_losses_score,
            "More consecutive losses should lower score"
        );
    }
}
