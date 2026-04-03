//! Optimization Results
//!
//! This module defines the data structures for storing optimization results,
//! including individual trial results and the overall optimization outcome.

use crate::backtester::BacktestResult;
use crate::config::OptimizerConfig;
use crate::constraints::SampledParams;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result of a single optimization trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Trial number (0-indexed)
    pub trial_number: usize,

    /// Sampled parameters for this trial
    pub params: SampledParams,

    /// Backtest result
    pub backtest_result: BacktestResult,

    /// Objective score (higher is better)
    pub score: f64,

    /// When this trial completed
    pub timestamp: DateTime<Utc>,
}

impl TrialResult {
    /// Create a new trial result
    pub fn new(
        trial_number: usize,
        params: SampledParams,
        backtest_result: BacktestResult,
        score: f64,
    ) -> Self {
        Self {
            trial_number,
            params,
            backtest_result,
            score,
            timestamp: Utc::now(),
        }
    }

    /// Check if this trial is better than another
    pub fn is_better_than(&self, other: &TrialResult) -> bool {
        self.score > other.score
    }

    /// Check if this trial is profitable
    pub fn is_profitable(&self) -> bool {
        self.backtest_result.is_profitable()
    }

    /// Get a parameter value by name
    pub fn get_param(&self, name: &str) -> Option<f64> {
        self.params.get(name)
    }

    /// Get a parameter as integer
    pub fn get_param_int(&self, name: &str) -> Option<i32> {
        self.params.get_int(name)
    }

    /// Get a parameter as boolean
    pub fn get_param_bool(&self, name: &str) -> Option<bool> {
        self.params.get_bool(name)
    }
}

/// Complete result of an optimization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Asset symbol that was optimized
    pub asset: String,

    /// Best parameters found
    pub best_params: SampledParams,

    /// Best objective score achieved
    pub best_score: f64,

    /// Backtest result for the best parameters
    pub best_backtest: BacktestResult,

    /// All trial results
    pub all_trials: Vec<TrialResult>,

    /// Total number of trials run
    pub total_trials: usize,

    /// Optimization duration in seconds
    pub duration_seconds: f64,

    /// When optimization started
    pub started_at: DateTime<Utc>,

    /// When optimization completed
    pub completed_at: DateTime<Utc>,

    /// Configuration used for this optimization
    #[serde(skip)]
    pub config: OptimizerConfig,
}

impl OptimizationResult {
    /// Get the improvement from worst to best trial
    pub fn score_improvement(&self) -> f64 {
        if self.all_trials.is_empty() {
            return 0.0;
        }

        let worst = self
            .all_trials
            .iter()
            .map(|t| t.score)
            .fold(f64::INFINITY, f64::min);

        self.best_score - worst
    }

    /// Get the number of profitable trials
    pub fn profitable_trials(&self) -> usize {
        self.all_trials.iter().filter(|t| t.is_profitable()).count()
    }

    /// Get the profitability rate
    pub fn profitability_rate(&self) -> f64 {
        if self.all_trials.is_empty() {
            return 0.0;
        }
        self.profitable_trials() as f64 / self.all_trials.len() as f64
    }

    /// Get statistics about the score distribution
    pub fn score_statistics(&self) -> ScoreStatistics {
        if self.all_trials.is_empty() {
            return ScoreStatistics::default();
        }

        let scores: Vec<f64> = self.all_trials.iter().map(|t| t.score).collect();
        let n = scores.len() as f64;

        let mean = scores.iter().sum::<f64>() / n;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let median = if sorted.len().is_multiple_of(2) {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let p25_idx = (n * 0.25) as usize;
        let p75_idx = (n * 0.75) as usize;
        let percentile_25 = sorted.get(p25_idx).copied().unwrap_or(min);
        let percentile_75 = sorted.get(p75_idx).copied().unwrap_or(max);

        ScoreStatistics {
            min,
            max,
            mean,
            median,
            std_dev,
            percentile_25,
            percentile_75,
        }
    }

    /// Get the top N trials
    pub fn top_trials(&self, n: usize) -> Vec<&TrialResult> {
        let mut trials: Vec<_> = self.all_trials.iter().collect();
        trials.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        trials.into_iter().take(n).collect()
    }

    /// Get trials per second throughput
    pub fn throughput(&self) -> f64 {
        if self.duration_seconds > 0.0 {
            self.total_trials as f64 / self.duration_seconds
        } else {
            0.0
        }
    }

    /// Convert best params to a HashMap for easy access
    pub fn best_params_map(&self) -> &HashMap<String, f64> {
        &self.best_params.values
    }

    /// Convert to janus-core OptimizedParams format
    pub fn to_optimized_params(&self) -> janus_core::optimized_params::OptimizedParams {
        let params = &self.best_params;
        let backtest = &self.best_backtest;

        janus_core::optimized_params::OptimizedParams {
            asset: self.asset.clone(),
            ema_fast_period: params.get_int("ema_fast_period").unwrap_or(9) as u32,
            ema_slow_period: params.get_int("ema_slow_period").unwrap_or(28) as u32,
            atr_length: params.get_int("atr_length").unwrap_or(14) as u32,
            atr_multiplier: params.get("atr_multiplier").unwrap_or(2.0),
            min_trailing_stop_pct: params.get("min_trailing_stop_pct").unwrap_or(0.5),
            min_ema_spread_pct: params.get("min_ema_spread_pct").unwrap_or(0.20),
            min_profit_pct: params.get("min_profit_pct").unwrap_or(0.40),
            take_profit_pct: params.get("take_profit_pct").unwrap_or(5.0),
            trade_cooldown_seconds: (params.get_int("cooldown_bars").unwrap_or(5) * 300) as u64,
            require_htf_alignment: params.get_bool("require_htf_alignment").unwrap_or(true),
            htf_timeframe_minutes: params.get_int("htf_timeframe_minutes").unwrap_or(15) as u32,
            max_position_size_usd: 20.0,
            enabled: true,
            min_hold_minutes: 15,
            prefer_trailing_stop_exit: true,
            optimized_at: self.completed_at.to_rfc3339(),
            optimization_score: self.best_score,
            backtest_result: janus_core::optimized_params::BacktestResultSummary {
                total_trades: backtest.total_trades as u32,
                winning_trades: backtest.winning_trades as u32,
                losing_trades: backtest.losing_trades as u32,
                total_pnl_pct: backtest.total_pnl_pct,
                max_drawdown_pct: backtest.max_drawdown_pct,
                win_rate: backtest.win_rate,
                profit_factor: backtest.profit_factor,
                sharpe_ratio: backtest.sharpe_ratio,
                trades_per_day: backtest.trades_per_day,
            },
        }
    }

    /// Create a summary report
    pub fn summary(&self) -> OptimizationSummary {
        OptimizationSummary {
            asset: self.asset.clone(),
            total_trials: self.total_trials,
            duration_seconds: self.duration_seconds,
            throughput: self.throughput(),
            best_score: self.best_score,
            best_return_pct: self.best_backtest.total_pnl_pct,
            best_win_rate: self.best_backtest.win_rate,
            best_sharpe: self.best_backtest.sharpe_ratio,
            best_max_drawdown: self.best_backtest.max_drawdown_pct,
            best_trades: self.best_backtest.total_trades,
            profitable_trials: self.profitable_trials(),
            profitability_rate: self.profitability_rate(),
            score_stats: self.score_statistics(),
        }
    }
}

/// Statistics about the score distribution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScoreStatistics {
    /// Minimum score
    pub min: f64,

    /// Maximum score
    pub max: f64,

    /// Mean score
    pub mean: f64,

    /// Median score
    pub median: f64,

    /// Standard deviation of scores
    pub std_dev: f64,

    /// 25th percentile
    pub percentile_25: f64,

    /// 75th percentile
    pub percentile_75: f64,
}

impl ScoreStatistics {
    /// Get the interquartile range
    pub fn iqr(&self) -> f64 {
        self.percentile_75 - self.percentile_25
    }

    /// Get the coefficient of variation
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() > 1e-10 {
            self.std_dev / self.mean.abs()
        } else {
            0.0
        }
    }
}

/// Summary of an optimization run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    /// Asset symbol
    pub asset: String,

    /// Total trials run
    pub total_trials: usize,

    /// Duration in seconds
    pub duration_seconds: f64,

    /// Trials per second
    pub throughput: f64,

    /// Best objective score
    pub best_score: f64,

    /// Best total return percentage
    pub best_return_pct: f64,

    /// Best win rate
    pub best_win_rate: f64,

    /// Best Sharpe ratio
    pub best_sharpe: f64,

    /// Best max drawdown
    pub best_max_drawdown: f64,

    /// Best number of trades
    pub best_trades: usize,

    /// Number of profitable trials
    pub profitable_trials: usize,

    /// Fraction of profitable trials
    pub profitability_rate: f64,

    /// Score statistics
    pub score_stats: ScoreStatistics,
}

impl std::fmt::Display for OptimizationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Optimization Summary: {} ===", self.asset)?;
        writeln!(
            f,
            "Trials: {} in {:.1}s ({:.1} trials/sec)",
            self.total_trials, self.duration_seconds, self.throughput
        )?;
        writeln!(
            f,
            "Profitable: {} ({:.1}%)",
            self.profitable_trials,
            self.profitability_rate * 100.0
        )?;
        writeln!(f, "\nBest Result:")?;
        writeln!(f, "  Score:       {:.2}", self.best_score)?;
        writeln!(f, "  Return:      {:.2}%", self.best_return_pct)?;
        writeln!(f, "  Win Rate:    {:.1}%", self.best_win_rate)?;
        writeln!(f, "  Sharpe:      {:.2}", self.best_sharpe)?;
        writeln!(f, "  Max DD:      {:.2}%", self.best_max_drawdown)?;
        writeln!(f, "  Trades:      {}", self.best_trades)?;
        writeln!(f, "\nScore Distribution:")?;
        writeln!(
            f,
            "  Range:  [{:.2}, {:.2}]",
            self.score_stats.min, self.score_stats.max
        )?;
        writeln!(
            f,
            "  Mean:   {:.2} ± {:.2}",
            self.score_stats.mean, self.score_stats.std_dev
        )?;
        writeln!(f, "  Median: {:.2}", self.score_stats.median)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_trial(num: usize, score: f64, pnl: f64) -> TrialResult {
        let mut values = HashMap::new();
        values.insert("ema_fast_period".to_string(), 9.0);
        values.insert("ema_slow_period".to_string(), 28.0);

        TrialResult {
            trial_number: num,
            params: SampledParams::new(values),
            backtest_result: BacktestResult {
                total_trades: 20,
                winning_trades: 12,
                losing_trades: 8,
                total_pnl_pct: pnl,
                max_drawdown_pct: 5.0,
                win_rate: 60.0,
                profit_factor: if pnl > 0.0 { 1.5 } else { 0.8 },
                sharpe_ratio: 1.2,
                sortino_ratio: 1.5,
                trades_per_day: 2.0,
                avg_trade_duration_minutes: 45.0,
                max_consecutive_wins: 4,
                max_consecutive_losses: 3,
                avg_winner_pct: 2.0,
                avg_loser_pct: 1.5,
                largest_winner_pct: 5.0,
                largest_loser_pct: 3.0,
            },
            score,
            timestamp: Utc::now(),
        }
    }

    #[test]
    fn test_trial_result_comparison() {
        let trial1 = create_test_trial(0, 10.0, 5.0);
        let trial2 = create_test_trial(1, 15.0, 8.0);

        assert!(trial2.is_better_than(&trial1));
        assert!(!trial1.is_better_than(&trial2));
    }

    #[test]
    fn test_trial_result_profitable() {
        let profitable = create_test_trial(0, 10.0, 5.0);
        let unprofitable = create_test_trial(1, -5.0, -3.0);

        assert!(profitable.is_profitable());
        assert!(!unprofitable.is_profitable());
    }

    #[test]
    fn test_trial_result_get_params() {
        let trial = create_test_trial(0, 10.0, 5.0);

        assert_eq!(trial.get_param("ema_fast_period"), Some(9.0));
        assert_eq!(trial.get_param_int("ema_fast_period"), Some(9));
        assert_eq!(trial.get_param("nonexistent"), None);
    }

    #[test]
    fn test_optimization_result_score_improvement() {
        let trials = vec![
            create_test_trial(0, 5.0, 2.0),
            create_test_trial(1, 15.0, 8.0),
            create_test_trial(2, 10.0, 5.0),
        ];

        let result = OptimizationResult {
            asset: "BTC".to_string(),
            best_params: trials[1].params.clone(),
            best_score: 15.0,
            best_backtest: trials[1].backtest_result.clone(),
            all_trials: trials,
            total_trials: 3,
            duration_seconds: 10.0,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            config: OptimizerConfig::default(),
        };

        assert_eq!(result.score_improvement(), 10.0); // 15 - 5
    }

    #[test]
    fn test_optimization_result_profitable_trials() {
        let trials = vec![
            create_test_trial(0, 10.0, 5.0),  // profitable
            create_test_trial(1, -5.0, -3.0), // not profitable
            create_test_trial(2, 15.0, 8.0),  // profitable
        ];

        let result = OptimizationResult {
            asset: "BTC".to_string(),
            best_params: trials[2].params.clone(),
            best_score: 15.0,
            best_backtest: trials[2].backtest_result.clone(),
            all_trials: trials,
            total_trials: 3,
            duration_seconds: 10.0,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            config: OptimizerConfig::default(),
        };

        assert_eq!(result.profitable_trials(), 2);
        assert!((result.profitability_rate() - 0.6667).abs() < 0.01);
    }

    #[test]
    fn test_score_statistics() {
        let trials = vec![
            create_test_trial(0, 10.0, 5.0),
            create_test_trial(1, 20.0, 10.0),
            create_test_trial(2, 15.0, 7.0),
            create_test_trial(3, 5.0, 2.0),
            create_test_trial(4, 25.0, 12.0),
        ];

        let result = OptimizationResult {
            asset: "BTC".to_string(),
            best_params: trials[4].params.clone(),
            best_score: 25.0,
            best_backtest: trials[4].backtest_result.clone(),
            all_trials: trials,
            total_trials: 5,
            duration_seconds: 10.0,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            config: OptimizerConfig::default(),
        };

        let stats = result.score_statistics();

        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 25.0);
        assert_eq!(stats.mean, 15.0);
        assert_eq!(stats.median, 15.0);
    }

    #[test]
    fn test_top_trials() {
        let trials = vec![
            create_test_trial(0, 10.0, 5.0),
            create_test_trial(1, 25.0, 12.0),
            create_test_trial(2, 15.0, 7.0),
        ];

        let result = OptimizationResult {
            asset: "BTC".to_string(),
            best_params: trials[1].params.clone(),
            best_score: 25.0,
            best_backtest: trials[1].backtest_result.clone(),
            all_trials: trials,
            total_trials: 3,
            duration_seconds: 10.0,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            config: OptimizerConfig::default(),
        };

        let top2 = result.top_trials(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].score, 25.0);
        assert_eq!(top2[1].score, 15.0);
    }

    #[test]
    fn test_throughput() {
        let trials = vec![create_test_trial(0, 10.0, 5.0)];

        let result = OptimizationResult {
            asset: "BTC".to_string(),
            best_params: trials[0].params.clone(),
            best_score: 10.0,
            best_backtest: trials[0].backtest_result.clone(),
            all_trials: trials,
            total_trials: 100,
            duration_seconds: 10.0,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            config: OptimizerConfig::default(),
        };

        assert_eq!(result.throughput(), 10.0); // 100 trials / 10 seconds
    }

    #[test]
    fn test_summary_display() {
        let trials = vec![create_test_trial(0, 10.0, 5.0)];

        let result = OptimizationResult {
            asset: "BTC".to_string(),
            best_params: trials[0].params.clone(),
            best_score: 10.0,
            best_backtest: trials[0].backtest_result.clone(),
            all_trials: trials,
            total_trials: 100,
            duration_seconds: 10.0,
            started_at: Utc::now(),
            completed_at: Utc::now(),
            config: OptimizerConfig::default(),
        };

        let summary = result.summary();
        let output = format!("{}", summary);

        assert!(output.contains("BTC"));
        assert!(output.contains("Score:"));
        assert!(output.contains("Return:"));
    }

    #[test]
    fn test_score_statistics_iqr() {
        let stats = ScoreStatistics {
            min: 0.0,
            max: 100.0,
            mean: 50.0,
            median: 50.0,
            std_dev: 25.0,
            percentile_25: 25.0,
            percentile_75: 75.0,
        };

        assert_eq!(stats.iqr(), 50.0);
        assert_eq!(stats.coefficient_of_variation(), 0.5);
    }
}
