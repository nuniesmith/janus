//! Evaluation metrics for ML models
//!
//! This module provides comprehensive metrics for evaluating model performance:
//! - Regression metrics (MSE, RMSE, MAE, R²)
//! - Classification metrics (Accuracy, Precision, Recall, F1)
//! - Trading-specific metrics (Sharpe, Win Rate, Max Drawdown)
//! - Performance tracking and reporting
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                  Evaluation Pipeline                     │
//! ├─────────────────────────────────────────────────────────┤
//! │                                                          │
//! │  Model Predictions + Ground Truth                       │
//! │         │                                                │
//! │         ▼                                                │
//! │  MetricsCalculator                                      │
//! │         │                                                │
//! │         ├─► Regression Metrics (MSE, RMSE, MAE, R²)     │
//! │         ├─► Classification Metrics (Acc, P, R, F1)      │
//! │         └─► Trading Metrics (Sharpe, Win Rate, DD)      │
//! │         │                                                │
//! │         ▼                                                │
//! │  EvaluationReport                                       │
//! │                                                          │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::evaluation::{MetricsCalculator, RegressionMetrics};
//!
//! let predictions = vec![1.0, 2.0, 3.0, 4.0];
//! let targets = vec![1.1, 2.2, 2.9, 4.1];
//!
//! let calculator = MetricsCalculator::new();
//! let metrics = calculator.regression_metrics(&predictions, &targets)?;
//!
//! println!("RMSE: {:.4}", metrics.rmse);
//! println!("R²: {:.4}", metrics.r_squared);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error::{MLError, Result};

/// Regression metrics for continuous prediction tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionMetrics {
    /// Mean Squared Error
    pub mse: f64,

    /// Root Mean Squared Error
    pub rmse: f64,

    /// Mean Absolute Error
    pub mae: f64,

    /// Mean Absolute Percentage Error (%)
    pub mape: f64,

    /// R² (coefficient of determination)
    pub r_squared: f64,

    /// Number of samples evaluated
    pub num_samples: usize,
}

impl RegressionMetrics {
    /// Calculate regression metrics from predictions and targets
    pub fn calculate(predictions: &[f64], targets: &[f64]) -> Result<Self> {
        if predictions.len() != targets.len() {
            return Err(MLError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        if predictions.is_empty() {
            return Err(MLError::InvalidInput("Empty predictions".to_string()));
        }

        let n = predictions.len();

        // MSE and RMSE
        let mse: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64;

        let rmse = mse.sqrt();

        // MAE
        let mae: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f64>()
            / n as f64;

        // MAPE
        let mape: f64 = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(_, t)| t.abs() > 1e-10) // Avoid division by zero
            .map(|(p, t)| ((p - t) / t).abs())
            .sum::<f64>()
            / n as f64
            * 100.0;

        // R²
        let target_mean: f64 = targets.iter().sum::<f64>() / n as f64;
        let ss_tot: f64 = targets.iter().map(|t| (t - target_mean).powi(2)).sum();
        let ss_res: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (t - p).powi(2))
            .sum();

        let r_squared = if ss_tot > 1e-10 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };

        Ok(Self {
            mse,
            rmse,
            mae,
            mape,
            r_squared,
            num_samples: n,
        })
    }

    /// Print metrics in a formatted way
    pub fn print(&self) {
        println!("Regression Metrics (n={}):", self.num_samples);
        println!("  MSE:       {:.6}", self.mse);
        println!("  RMSE:      {:.6}", self.rmse);
        println!("  MAE:       {:.6}", self.mae);
        println!("  MAPE:      {:.2}%", self.mape);
        println!("  R²:        {:.6}", self.r_squared);
    }
}

/// Classification metrics for discrete prediction tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationMetrics {
    /// Accuracy (correct / total)
    pub accuracy: f64,

    /// Precision (TP / (TP + FP))
    pub precision: f64,

    /// Recall (TP / (TP + FN))
    pub recall: f64,

    /// F1 Score (harmonic mean of precision and recall)
    pub f1_score: f64,

    /// True Positives
    pub true_positives: usize,

    /// True Negatives
    pub true_negatives: usize,

    /// False Positives
    pub false_positives: usize,

    /// False Negatives
    pub false_negatives: usize,

    /// Number of samples evaluated
    pub num_samples: usize,
}

impl ClassificationMetrics {
    /// Calculate binary classification metrics
    pub fn calculate_binary(predictions: &[bool], targets: &[bool]) -> Result<Self> {
        if predictions.len() != targets.len() {
            return Err(MLError::InvalidInput(
                "Predictions and targets must have same length".to_string(),
            ));
        }

        if predictions.is_empty() {
            return Err(MLError::InvalidInput("Empty predictions".to_string()));
        }

        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_ = 0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            match (pred, target) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
            }
        }

        let n = predictions.len();
        let accuracy = (tp + tn) as f64 / n as f64;

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };

        let f1_score = if precision + recall > 1e-10 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        Ok(Self {
            accuracy,
            precision,
            recall,
            f1_score,
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives: fn_,
            num_samples: n,
        })
    }

    /// Print metrics in a formatted way
    pub fn print(&self) {
        println!("Classification Metrics (n={}):", self.num_samples);
        println!("  Accuracy:  {:.4}", self.accuracy);
        println!("  Precision: {:.4}", self.precision);
        println!("  Recall:    {:.4}", self.recall);
        println!("  F1 Score:  {:.4}", self.f1_score);
        println!("\nConfusion Matrix:");
        println!(
            "  TP: {}  FP: {}",
            self.true_positives, self.false_positives
        );
        println!(
            "  FN: {}  TN: {}",
            self.false_negatives, self.true_negatives
        );
    }
}

/// Trading-specific metrics for backtesting model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    /// Sharpe Ratio (risk-adjusted return)
    pub sharpe_ratio: f64,

    /// Sortino Ratio (downside risk-adjusted return)
    pub sortino_ratio: f64,

    /// Maximum Drawdown (%)
    pub max_drawdown: f64,

    /// Win Rate (% of profitable trades)
    pub win_rate: f64,

    /// Average Win / Average Loss ratio
    pub profit_factor: f64,

    /// Total Return (%)
    pub total_return: f64,

    /// Number of trades
    pub num_trades: usize,

    /// Number of winning trades
    pub num_wins: usize,

    /// Number of losing trades
    pub num_losses: usize,
}

impl TradingMetrics {
    /// Calculate trading metrics from returns
    ///
    /// # Arguments
    /// * `returns` - Array of per-period returns (e.g., daily returns)
    /// * `risk_free_rate` - Risk-free rate for Sharpe calculation (annualized)
    pub fn calculate(returns: &[f64], risk_free_rate: f64) -> Result<Self> {
        if returns.is_empty() {
            return Err(MLError::InvalidInput("Empty returns".to_string()));
        }

        let n = returns.len();

        // Calculate mean and std of returns
        let mean_return: f64 = returns.iter().sum::<f64>() / n as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / n as f64;
        let std_return = variance.sqrt();

        // Sharpe Ratio (assuming daily returns, annualize)
        let sharpe_ratio = if std_return > 1e-10 {
            ((mean_return - risk_free_rate / 252.0) / std_return) * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Sortino Ratio (only downside deviation)
        let downside_returns: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let downside_std = if !downside_returns.is_empty() {
            let downside_var: f64 = downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                / downside_returns.len() as f64;
            downside_var.sqrt()
        } else {
            1e-10
        };

        let sortino_ratio = if downside_std > 1e-10 {
            ((mean_return - risk_free_rate / 252.0) / downside_std) * (252.0_f64).sqrt()
        } else {
            0.0
        };

        // Maximum Drawdown
        let mut cumulative = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for &ret in returns {
            cumulative *= 1.0 + ret;
            if cumulative > peak {
                peak = cumulative;
            }
            let drawdown = (peak - cumulative) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        // Win/Loss stats
        let wins: Vec<f64> = returns.iter().filter(|&&r| r > 0.0).copied().collect();
        let losses: Vec<f64> = returns.iter().filter(|&&r| r < 0.0).copied().collect();

        let num_wins = wins.len();
        let num_losses = losses.len();
        let win_rate = if n > 0 {
            num_wins as f64 / n as f64
        } else {
            0.0
        };

        let avg_win = if !wins.is_empty() {
            wins.iter().sum::<f64>() / wins.len() as f64
        } else {
            0.0
        };

        let avg_loss = if !losses.is_empty() {
            losses.iter().sum::<f64>().abs() / losses.len() as f64
        } else {
            1e-10
        };

        let profit_factor = if avg_loss > 1e-10 {
            avg_win / avg_loss
        } else {
            0.0
        };

        // Total return
        let total_return = returns.iter().fold(1.0, |acc, &r| acc * (1.0 + r)) - 1.0;

        Ok(Self {
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: max_dd * 100.0, // Convert to percentage
            win_rate: win_rate * 100.0,   // Convert to percentage
            profit_factor,
            total_return: total_return * 100.0, // Convert to percentage
            num_trades: n,
            num_wins,
            num_losses,
        })
    }

    /// Print metrics in a formatted way
    pub fn print(&self) {
        println!("Trading Metrics (n={}):", self.num_trades);
        println!("  Total Return:   {:.2}%", self.total_return);
        println!("  Sharpe Ratio:   {:.4}", self.sharpe_ratio);
        println!("  Sortino Ratio:  {:.4}", self.sortino_ratio);
        println!("  Max Drawdown:   {:.2}%", self.max_drawdown);
        println!("  Win Rate:       {:.2}%", self.win_rate);
        println!("  Profit Factor:  {:.4}", self.profit_factor);
        println!("  Wins/Losses:    {}/{}", self.num_wins, self.num_losses);
    }
}

/// Combined evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Model name
    pub model_name: String,

    /// Evaluation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Regression metrics (if applicable)
    pub regression: Option<RegressionMetrics>,

    /// Classification metrics (if applicable)
    pub classification: Option<ClassificationMetrics>,

    /// Trading metrics (if applicable)
    pub trading: Option<TradingMetrics>,

    /// Custom metrics
    pub custom: HashMap<String, f64>,
}

impl EvaluationReport {
    /// Create a new evaluation report
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            timestamp: chrono::Utc::now(),
            regression: None,
            classification: None,
            trading: None,
            custom: HashMap::new(),
        }
    }

    /// Add regression metrics
    pub fn with_regression(mut self, metrics: RegressionMetrics) -> Self {
        self.regression = Some(metrics);
        self
    }

    /// Add classification metrics
    pub fn with_classification(mut self, metrics: ClassificationMetrics) -> Self {
        self.classification = Some(metrics);
        self
    }

    /// Add trading metrics
    pub fn with_trading(mut self, metrics: TradingMetrics) -> Self {
        self.trading = Some(metrics);
        self
    }

    /// Add a custom metric
    pub fn add_custom_metric(&mut self, name: String, value: f64) {
        self.custom.insert(name, value);
    }

    /// Print the full report
    pub fn print(&self) {
        println!("\n========================================");
        println!("Evaluation Report: {}", self.model_name);
        println!("Timestamp: {}", self.timestamp);
        println!("========================================\n");

        if let Some(ref reg) = self.regression {
            reg.print();
            println!();
        }

        if let Some(ref cls) = self.classification {
            cls.print();
            println!();
        }

        if let Some(ref trd) = self.trading {
            trd.print();
            println!();
        }

        if !self.custom.is_empty() {
            println!("Custom Metrics:");
            for (name, value) in &self.custom {
                println!("  {}: {:.6}", name, value);
            }
            println!();
        }
    }

    /// Save report to JSON file
    pub fn save_json<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load report from JSON file
    pub fn load_json<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let report = serde_json::from_str(&json)?;
        Ok(report)
    }
}

/// Metrics calculator utility
pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Create a new metrics calculator
    pub fn new() -> Self {
        Self
    }

    /// Calculate regression metrics
    pub fn regression_metrics(
        &self,
        predictions: &[f64],
        targets: &[f64],
    ) -> Result<RegressionMetrics> {
        RegressionMetrics::calculate(predictions, targets)
    }

    /// Calculate classification metrics
    pub fn classification_metrics(
        &self,
        predictions: &[bool],
        targets: &[bool],
    ) -> Result<ClassificationMetrics> {
        ClassificationMetrics::calculate_binary(predictions, targets)
    }

    /// Calculate trading metrics
    pub fn trading_metrics(&self, returns: &[f64], risk_free_rate: f64) -> Result<TradingMetrics> {
        TradingMetrics::calculate(returns, risk_free_rate)
    }

    /// Create a full evaluation report
    pub fn create_report(
        &self,
        model_name: String,
        predictions: &[f64],
        targets: &[f64],
    ) -> Result<EvaluationReport> {
        let reg_metrics = self.regression_metrics(predictions, targets)?;

        // Convert to trading returns (simple percentage change)
        let returns: Vec<f64> = predictions
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0].max(1e-10))
            .collect();

        let trading_metrics = if !returns.is_empty() {
            Some(self.trading_metrics(&returns, 0.02)?)
        } else {
            None
        };

        let mut report = EvaluationReport::new(model_name).with_regression(reg_metrics);

        if let Some(tm) = trading_metrics {
            report = report.with_trading(tm);
        }

        Ok(report)
    }
}

impl Default for MetricsCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regression_metrics() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.1, 2.2, 2.9, 4.1, 4.8];

        let metrics = RegressionMetrics::calculate(&predictions, &targets).unwrap();

        assert!(metrics.mse > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.r_squared > 0.9); // Should be very high for this data
        assert_eq!(metrics.num_samples, 5);
    }

    #[test]
    fn test_perfect_regression() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let metrics = RegressionMetrics::calculate(&predictions, &targets).unwrap();

        assert_eq!(metrics.mse, 0.0);
        assert_eq!(metrics.rmse, 0.0);
        assert_eq!(metrics.mae, 0.0);
        assert!((metrics.r_squared - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_classification_metrics() {
        let predictions = vec![true, true, false, false, true, false, true, true];
        let targets = vec![true, false, false, true, true, false, false, true];

        let metrics = ClassificationMetrics::calculate_binary(&predictions, &targets).unwrap();

        assert_eq!(metrics.true_positives, 3);
        assert_eq!(metrics.false_positives, 2);
        assert_eq!(metrics.true_negatives, 2);
        assert_eq!(metrics.false_negatives, 1);
        assert_eq!(metrics.accuracy, 0.625);
        assert_eq!(metrics.num_samples, 8);
    }

    #[test]
    fn test_perfect_classification() {
        let predictions = vec![true, true, false, false];
        let targets = vec![true, true, false, false];

        let metrics = ClassificationMetrics::calculate_binary(&predictions, &targets).unwrap();

        assert_eq!(metrics.accuracy, 1.0);
        assert_eq!(metrics.precision, 1.0);
        assert_eq!(metrics.recall, 1.0);
        assert_eq!(metrics.f1_score, 1.0);
    }

    #[test]
    fn test_trading_metrics() {
        // Simulate 10 days of returns
        let returns = vec![
            0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.02, 0.03, -0.015, 0.01,
        ];

        let metrics = TradingMetrics::calculate(&returns, 0.02).unwrap();

        assert!(metrics.sharpe_ratio != 0.0);
        assert!(metrics.max_drawdown >= 0.0);
        assert!(metrics.win_rate >= 0.0 && metrics.win_rate <= 100.0);
        assert_eq!(metrics.num_trades, 10);
        assert_eq!(metrics.num_wins, 6);
        assert_eq!(metrics.num_losses, 4);
    }

    #[test]
    fn test_positive_returns_trading_metrics() {
        let returns = vec![0.01, 0.02, 0.015, 0.03];

        let metrics = TradingMetrics::calculate(&returns, 0.02).unwrap();

        assert_eq!(metrics.num_wins, 4);
        assert_eq!(metrics.num_losses, 0);
        assert_eq!(metrics.win_rate, 100.0);
        assert!(metrics.total_return > 0.0);
        assert!(metrics.max_drawdown < 1e-6); // Should be near zero
    }

    #[test]
    fn test_evaluation_report() {
        let mut report = EvaluationReport::new("test_model".to_string());

        let reg_metrics = RegressionMetrics::calculate(&[1.0, 2.0], &[1.1, 2.0]).unwrap();
        report = report.with_regression(reg_metrics);

        report.add_custom_metric("inference_time_ms".to_string(), 2.5);

        assert!(report.regression.is_some());
        assert_eq!(report.custom.len(), 1);
        assert_eq!(report.custom.get("inference_time_ms"), Some(&2.5));
    }

    #[test]
    fn test_metrics_calculator() {
        let calc = MetricsCalculator::new();

        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.1, 2.0, 2.9];

        let reg_metrics = calc.regression_metrics(&predictions, &targets).unwrap();
        assert!(reg_metrics.rmse > 0.0);

        let pred_binary = vec![true, false, true, false];
        let target_binary = vec![true, false, false, true];

        let cls_metrics = calc
            .classification_metrics(&pred_binary, &target_binary)
            .unwrap();
        assert!(cls_metrics.accuracy >= 0.0);
    }

    #[test]
    fn test_empty_predictions_error() {
        let result = RegressionMetrics::calculate(&[], &[]);
        assert!(result.is_err());

        let result = ClassificationMetrics::calculate_binary(&[], &[]);
        assert!(result.is_err());

        let result = TradingMetrics::calculate(&[], 0.02);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_lengths_error() {
        let result = RegressionMetrics::calculate(&[1.0], &[1.0, 2.0]);
        assert!(result.is_err());

        let result = ClassificationMetrics::calculate_binary(&[true], &[true, false]);
        assert!(result.is_err());
    }
}
