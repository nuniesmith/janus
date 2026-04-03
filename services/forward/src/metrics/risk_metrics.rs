//! # Risk Metrics Collector
//!
//! Prometheus metrics for risk management and portfolio tracking.

use prometheus::{
    Gauge, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge, Opts, Registry,
};
use std::sync::Arc;

/// Performance metrics update parameters
pub struct PerformanceMetrics {
    pub total: u64,
    pub wins: u64,
    pub losses: u64,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub expected_value: f64,
}

/// Risk metrics collector
pub struct RiskMetricsCollector {
    // Position sizing metrics
    pub position_sizes_calculated_total: IntCounter,
    pub position_size_avg: Gauge,
    pub position_size_histogram: Histogram,
    pub position_value_histogram: Histogram,

    // Risk amount metrics
    pub risk_amount_avg: Gauge,
    pub risk_amount_histogram: Histogram,
    pub risk_percentage_avg: Gauge,
    pub risk_percentage_histogram: Histogram,

    // Stop loss metrics
    pub stop_losses_calculated_total: IntCounter,
    pub stop_loss_distance_avg: Gauge,
    pub stop_loss_distance_histogram: Histogram,

    // Take profit metrics
    pub take_profits_calculated_total: IntCounter,
    pub take_profit_distance_avg: Gauge,
    pub take_profit_distance_histogram: Histogram,

    // Risk/Reward metrics
    pub risk_reward_ratio_avg: Gauge,
    pub risk_reward_ratio_histogram: Histogram,

    // Portfolio metrics
    pub portfolio_heat: Gauge,
    pub portfolio_exposure: Gauge,
    pub portfolio_exposure_percentage: Gauge,
    pub portfolio_position_count: IntGauge,
    pub portfolio_concentration_risk: Gauge,
    pub portfolio_diversification_score: Gauge,

    // Position limits
    pub position_limit_violations: IntCounterVec,
    pub daily_loss_limit_violations: IntCounter,
    pub portfolio_exposure_limit_violations: IntCounter,
    pub symbol_exposure_limit_violations: IntCounterVec,

    // Performance metrics
    pub total_trades: IntCounter,
    pub winning_trades: IntCounter,
    pub losing_trades: IntCounter,
    pub win_rate: Gauge,
    pub profit_factor: Gauge,
    pub avg_win: Gauge,
    pub avg_loss: Gauge,
    pub expected_value: Gauge,

    // Drawdown metrics
    pub current_drawdown: Gauge,
    pub current_drawdown_percentage: Gauge,
    pub max_drawdown: Gauge,
    pub max_drawdown_percentage: Gauge,
    pub drawdown_duration: IntGauge,

    // Kelly criterion
    pub kelly_fraction: Gauge,

    // Sharpe ratio
    pub sharpe_ratio: Gauge,

    // Risk calculation duration
    pub position_sizing_duration: Histogram,
    pub stop_calculation_duration: Histogram,
    pub risk_validation_duration: Histogram,

    // Risk errors
    pub risk_calculation_errors: IntCounter,
    pub risk_validation_errors: IntCounterVec,
}

impl RiskMetricsCollector {
    /// Create new risk metrics collector
    pub fn new(registry: Arc<Registry>) -> Result<Self, prometheus::Error> {
        // Position sizing metrics
        let position_sizes_calculated_total = IntCounter::with_opts(Opts::new(
            "janus_position_sizes_calculated_total",
            "Total number of position sizes calculated",
        ))?;
        registry.register(Box::new(position_sizes_calculated_total.clone()))?;

        let position_size_avg = Gauge::with_opts(Opts::new(
            "janus_position_size_avg",
            "Average position size (quantity)",
        ))?;
        registry.register(Box::new(position_size_avg.clone()))?;

        let position_size_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_position_size",
            "Distribution of position sizes",
        ))?;
        registry.register(Box::new(position_size_histogram.clone()))?;

        let position_value_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_position_value",
            "Distribution of position values in dollars",
        ))?;
        registry.register(Box::new(position_value_histogram.clone()))?;

        // Risk amount metrics
        let risk_amount_avg = Gauge::with_opts(Opts::new(
            "janus_risk_amount_avg",
            "Average risk amount in dollars",
        ))?;
        registry.register(Box::new(risk_amount_avg.clone()))?;

        let risk_amount_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_risk_amount",
            "Distribution of risk amounts",
        ))?;
        registry.register(Box::new(risk_amount_histogram.clone()))?;

        let risk_percentage_avg = Gauge::with_opts(Opts::new(
            "janus_risk_percentage_avg",
            "Average risk as percentage of account",
        ))?;
        registry.register(Box::new(risk_percentage_avg.clone()))?;

        let risk_percentage_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_risk_percentage",
            "Distribution of risk percentages",
        ))?;
        registry.register(Box::new(risk_percentage_histogram.clone()))?;

        // Stop loss metrics
        let stop_losses_calculated_total = IntCounter::with_opts(Opts::new(
            "janus_stop_losses_calculated_total",
            "Total number of stop losses calculated",
        ))?;
        registry.register(Box::new(stop_losses_calculated_total.clone()))?;

        let stop_loss_distance_avg = Gauge::with_opts(Opts::new(
            "janus_stop_loss_distance_avg",
            "Average stop loss distance from entry",
        ))?;
        registry.register(Box::new(stop_loss_distance_avg.clone()))?;

        let stop_loss_distance_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_stop_loss_distance",
            "Distribution of stop loss distances",
        ))?;
        registry.register(Box::new(stop_loss_distance_histogram.clone()))?;

        // Take profit metrics
        let take_profits_calculated_total = IntCounter::with_opts(Opts::new(
            "janus_take_profits_calculated_total",
            "Total number of take profits calculated",
        ))?;
        registry.register(Box::new(take_profits_calculated_total.clone()))?;

        let take_profit_distance_avg = Gauge::with_opts(Opts::new(
            "janus_take_profit_distance_avg",
            "Average take profit distance from entry",
        ))?;
        registry.register(Box::new(take_profit_distance_avg.clone()))?;

        let take_profit_distance_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_take_profit_distance",
            "Distribution of take profit distances",
        ))?;
        registry.register(Box::new(take_profit_distance_histogram.clone()))?;

        // Risk/Reward metrics
        let risk_reward_ratio_avg = Gauge::with_opts(Opts::new(
            "janus_risk_reward_ratio_avg",
            "Average risk/reward ratio",
        ))?;
        registry.register(Box::new(risk_reward_ratio_avg.clone()))?;

        let risk_reward_ratio_histogram = Histogram::with_opts(HistogramOpts::new(
            "janus_risk_reward_ratio",
            "Distribution of risk/reward ratios",
        ))?;
        registry.register(Box::new(risk_reward_ratio_histogram.clone()))?;

        // Portfolio metrics
        let portfolio_heat = Gauge::with_opts(Opts::new(
            "janus_portfolio_heat",
            "Portfolio heat (total risk as fraction of account)",
        ))?;
        registry.register(Box::new(portfolio_heat.clone()))?;

        let portfolio_exposure = Gauge::with_opts(Opts::new(
            "janus_portfolio_exposure",
            "Total portfolio exposure in dollars",
        ))?;
        registry.register(Box::new(portfolio_exposure.clone()))?;

        let portfolio_exposure_percentage = Gauge::with_opts(Opts::new(
            "janus_portfolio_exposure_percentage",
            "Portfolio exposure as percentage of account",
        ))?;
        registry.register(Box::new(portfolio_exposure_percentage.clone()))?;

        let portfolio_position_count = IntGauge::with_opts(Opts::new(
            "janus_portfolio_position_count",
            "Number of open positions in portfolio",
        ))?;
        registry.register(Box::new(portfolio_position_count.clone()))?;

        let portfolio_concentration_risk = Gauge::with_opts(Opts::new(
            "janus_portfolio_concentration_risk",
            "Portfolio concentration risk (largest position as % of total)",
        ))?;
        registry.register(Box::new(portfolio_concentration_risk.clone()))?;

        let portfolio_diversification_score = Gauge::with_opts(Opts::new(
            "janus_portfolio_diversification_score",
            "Portfolio diversification score (0.0 to 1.0)",
        ))?;
        registry.register(Box::new(portfolio_diversification_score.clone()))?;

        // Position limits
        let position_limit_violations = IntCounterVec::new(
            Opts::new(
                "janus_position_limit_violations_total",
                "Total position limit violations by type",
            ),
            &["limit_type"],
        )?;
        registry.register(Box::new(position_limit_violations.clone()))?;

        let daily_loss_limit_violations = IntCounter::with_opts(Opts::new(
            "janus_daily_loss_limit_violations_total",
            "Total daily loss limit violations",
        ))?;
        registry.register(Box::new(daily_loss_limit_violations.clone()))?;

        let portfolio_exposure_limit_violations = IntCounter::with_opts(Opts::new(
            "janus_portfolio_exposure_limit_violations_total",
            "Total portfolio exposure limit violations",
        ))?;
        registry.register(Box::new(portfolio_exposure_limit_violations.clone()))?;

        let symbol_exposure_limit_violations = IntCounterVec::new(
            Opts::new(
                "janus_symbol_exposure_limit_violations_total",
                "Total symbol exposure limit violations",
            ),
            &["symbol"],
        )?;
        registry.register(Box::new(symbol_exposure_limit_violations.clone()))?;

        // Performance metrics
        let total_trades = IntCounter::with_opts(Opts::new(
            "janus_total_trades",
            "Total number of trades executed",
        ))?;
        registry.register(Box::new(total_trades.clone()))?;

        let winning_trades = IntCounter::with_opts(Opts::new(
            "janus_winning_trades",
            "Total number of winning trades",
        ))?;
        registry.register(Box::new(winning_trades.clone()))?;

        let losing_trades = IntCounter::with_opts(Opts::new(
            "janus_losing_trades",
            "Total number of losing trades",
        ))?;
        registry.register(Box::new(losing_trades.clone()))?;

        let win_rate = Gauge::with_opts(Opts::new(
            "janus_win_rate",
            "Win rate (percentage of winning trades)",
        ))?;
        registry.register(Box::new(win_rate.clone()))?;

        let profit_factor = Gauge::with_opts(Opts::new(
            "janus_profit_factor",
            "Profit factor (total profit / total loss)",
        ))?;
        registry.register(Box::new(profit_factor.clone()))?;

        let avg_win = Gauge::with_opts(Opts::new("janus_avg_win", "Average winning trade"))?;
        registry.register(Box::new(avg_win.clone()))?;

        let avg_loss = Gauge::with_opts(Opts::new("janus_avg_loss", "Average losing trade"))?;
        registry.register(Box::new(avg_loss.clone()))?;

        let expected_value = Gauge::with_opts(Opts::new(
            "janus_expected_value",
            "Expected value per trade",
        ))?;
        registry.register(Box::new(expected_value.clone()))?;

        // Drawdown metrics
        let current_drawdown = Gauge::with_opts(Opts::new(
            "janus_current_drawdown",
            "Current drawdown amount",
        ))?;
        registry.register(Box::new(current_drawdown.clone()))?;

        let current_drawdown_percentage = Gauge::with_opts(Opts::new(
            "janus_current_drawdown_percentage",
            "Current drawdown as percentage",
        ))?;
        registry.register(Box::new(current_drawdown_percentage.clone()))?;

        let max_drawdown = Gauge::with_opts(Opts::new(
            "janus_max_drawdown",
            "Maximum drawdown ever experienced",
        ))?;
        registry.register(Box::new(max_drawdown.clone()))?;

        let max_drawdown_percentage = Gauge::with_opts(Opts::new(
            "janus_max_drawdown_percentage",
            "Maximum drawdown percentage",
        ))?;
        registry.register(Box::new(max_drawdown_percentage.clone()))?;

        let drawdown_duration = IntGauge::with_opts(Opts::new(
            "janus_drawdown_duration",
            "Current drawdown duration in periods",
        ))?;
        registry.register(Box::new(drawdown_duration.clone()))?;

        // Kelly criterion
        let kelly_fraction = Gauge::with_opts(Opts::new(
            "janus_kelly_fraction",
            "Kelly criterion optimal bet size fraction",
        ))?;
        registry.register(Box::new(kelly_fraction.clone()))?;

        // Sharpe ratio
        let sharpe_ratio = Gauge::with_opts(Opts::new(
            "janus_sharpe_ratio",
            "Sharpe ratio (risk-adjusted returns)",
        ))?;
        registry.register(Box::new(sharpe_ratio.clone()))?;

        // Risk calculation duration
        let position_sizing_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_position_sizing_duration_seconds",
            "Time taken to calculate position size",
        ))?;
        registry.register(Box::new(position_sizing_duration.clone()))?;

        let stop_calculation_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_stop_calculation_duration_seconds",
            "Time taken to calculate stop loss/take profit",
        ))?;
        registry.register(Box::new(stop_calculation_duration.clone()))?;

        let risk_validation_duration = Histogram::with_opts(HistogramOpts::new(
            "janus_risk_validation_duration_seconds",
            "Time taken to validate risk limits",
        ))?;
        registry.register(Box::new(risk_validation_duration.clone()))?;

        // Risk errors
        let risk_calculation_errors = IntCounter::with_opts(Opts::new(
            "janus_risk_calculation_errors_total",
            "Total risk calculation errors",
        ))?;
        registry.register(Box::new(risk_calculation_errors.clone()))?;

        let risk_validation_errors = IntCounterVec::new(
            Opts::new(
                "janus_risk_validation_errors_total",
                "Total risk validation errors by type",
            ),
            &["error_type"],
        )?;
        registry.register(Box::new(risk_validation_errors.clone()))?;

        Ok(Self {
            position_sizes_calculated_total,
            position_size_avg,
            position_size_histogram,
            position_value_histogram,
            risk_amount_avg,
            risk_amount_histogram,
            risk_percentage_avg,
            risk_percentage_histogram,
            stop_losses_calculated_total,
            stop_loss_distance_avg,
            stop_loss_distance_histogram,
            take_profits_calculated_total,
            take_profit_distance_avg,
            take_profit_distance_histogram,
            risk_reward_ratio_avg,
            risk_reward_ratio_histogram,
            portfolio_heat,
            portfolio_exposure,
            portfolio_exposure_percentage,
            portfolio_position_count,
            portfolio_concentration_risk,
            portfolio_diversification_score,
            position_limit_violations,
            daily_loss_limit_violations,
            portfolio_exposure_limit_violations,
            symbol_exposure_limit_violations,
            total_trades,
            winning_trades,
            losing_trades,
            win_rate,
            profit_factor,
            avg_win,
            avg_loss,
            expected_value,
            current_drawdown,
            current_drawdown_percentage,
            max_drawdown,
            max_drawdown_percentage,
            drawdown_duration,
            kelly_fraction,
            sharpe_ratio,
            position_sizing_duration,
            stop_calculation_duration,
            risk_validation_duration,
            risk_calculation_errors,
            risk_validation_errors,
        })
    }

    /// Record position size calculation
    pub fn record_position_size(
        &self,
        quantity: f64,
        value: f64,
        risk_amount: f64,
        risk_percentage: f64,
        duration_secs: f64,
    ) {
        self.position_sizes_calculated_total.inc();
        self.position_size_histogram.observe(quantity);
        self.position_value_histogram.observe(value);
        self.risk_amount_histogram.observe(risk_amount);
        self.risk_percentage_histogram.observe(risk_percentage);
        self.position_sizing_duration.observe(duration_secs);
    }

    /// Update average position size and risk
    pub fn update_position_averages(
        &self,
        avg_size: f64,
        avg_risk_amount: f64,
        avg_risk_percentage: f64,
    ) {
        self.position_size_avg.set(avg_size);
        self.risk_amount_avg.set(avg_risk_amount);
        self.risk_percentage_avg.set(avg_risk_percentage);
    }

    /// Record stop loss calculation
    pub fn record_stop_loss(&self, distance: f64, duration_secs: f64) {
        self.stop_losses_calculated_total.inc();
        self.stop_loss_distance_histogram.observe(distance);
        self.stop_calculation_duration.observe(duration_secs);
    }

    /// Record take profit calculation
    pub fn record_take_profit(&self, distance: f64) {
        self.take_profits_calculated_total.inc();
        self.take_profit_distance_histogram.observe(distance);
    }

    /// Update stop loss average
    pub fn update_stop_loss_avg(&self, avg_distance: f64) {
        self.stop_loss_distance_avg.set(avg_distance);
    }

    /// Update take profit average
    pub fn update_take_profit_avg(&self, avg_distance: f64) {
        self.take_profit_distance_avg.set(avg_distance);
    }

    /// Record risk/reward ratio
    pub fn record_risk_reward(&self, ratio: f64) {
        self.risk_reward_ratio_histogram.observe(ratio);
    }

    /// Update average risk/reward ratio
    pub fn update_risk_reward_avg(&self, avg_ratio: f64) {
        self.risk_reward_ratio_avg.set(avg_ratio);
    }

    /// Update portfolio metrics
    pub fn update_portfolio_metrics(
        &self,
        heat: f64,
        exposure: f64,
        exposure_pct: f64,
        position_count: i64,
        concentration: f64,
        diversification: f64,
    ) {
        self.portfolio_heat.set(heat);
        self.portfolio_exposure.set(exposure);
        self.portfolio_exposure_percentage.set(exposure_pct);
        self.portfolio_position_count.set(position_count);
        self.portfolio_concentration_risk.set(concentration);
        self.portfolio_diversification_score.set(diversification);
    }

    /// Record position limit violation
    pub fn record_position_limit_violation(&self, limit_type: &str) {
        self.position_limit_violations
            .with_label_values(&[limit_type])
            .inc();
    }

    /// Record daily loss limit violation
    pub fn record_daily_loss_limit_violation(&self) {
        self.daily_loss_limit_violations.inc();
    }

    /// Record portfolio exposure limit violation
    pub fn record_portfolio_exposure_violation(&self) {
        self.portfolio_exposure_limit_violations.inc();
    }

    /// Record symbol exposure limit violation
    pub fn record_symbol_exposure_violation(&self, symbol: &str) {
        self.symbol_exposure_limit_violations
            .with_label_values(&[symbol])
            .inc();
    }

    /// Update performance metrics
    pub fn update_performance_metrics(&self, metrics: PerformanceMetrics) {
        self.total_trades.inc_by(metrics.total);
        self.winning_trades.inc_by(metrics.wins);
        self.losing_trades.inc_by(metrics.losses);
        self.win_rate.set(metrics.win_rate);
        self.profit_factor.set(metrics.profit_factor);
        self.avg_win.set(metrics.avg_win);
        self.avg_loss.set(metrics.avg_loss);
        self.expected_value.set(metrics.expected_value);
    }

    /// Update drawdown metrics
    pub fn update_drawdown_metrics(
        &self,
        current_dd: f64,
        current_dd_pct: f64,
        max_dd: f64,
        max_dd_pct: f64,
        duration: i64,
    ) {
        self.current_drawdown.set(current_dd);
        self.current_drawdown_percentage.set(current_dd_pct);
        self.max_drawdown.set(max_dd);
        self.max_drawdown_percentage.set(max_dd_pct);
        self.drawdown_duration.set(duration);
    }

    /// Update Kelly fraction
    pub fn update_kelly_fraction(&self, fraction: f64) {
        self.kelly_fraction.set(fraction);
    }

    /// Update Sharpe ratio
    pub fn update_sharpe_ratio(&self, ratio: f64) {
        self.sharpe_ratio.set(ratio);
    }

    /// Record risk validation
    pub fn record_risk_validation(&self, duration_secs: f64) {
        self.risk_validation_duration.observe(duration_secs);
    }

    /// Record risk calculation error
    pub fn record_risk_calculation_error(&self) {
        self.risk_calculation_errors.inc();
    }

    /// Record risk validation error
    pub fn record_risk_validation_error(&self, error_type: &str) {
        self.risk_validation_errors
            .with_label_values(&[error_type])
            .inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_metrics_creation() {
        let registry = Arc::new(Registry::new());
        let metrics = RiskMetricsCollector::new(registry);
        assert!(metrics.is_ok());
    }

    #[test]
    fn test_position_size_recording() {
        let registry = Arc::new(Registry::new());
        let metrics = RiskMetricsCollector::new(registry).unwrap();

        metrics.record_position_size(0.1, 5000.0, 100.0, 0.01, 0.0001);
        assert_eq!(metrics.position_sizes_calculated_total.get(), 1);
    }

    #[test]
    fn test_stop_loss_recording() {
        let registry = Arc::new(Registry::new());
        let metrics = RiskMetricsCollector::new(registry).unwrap();

        metrics.record_stop_loss(1000.0, 0.0001);
        assert_eq!(metrics.stop_losses_calculated_total.get(), 1);
    }

    #[test]
    fn test_portfolio_metrics_update() {
        let registry = Arc::new(Registry::new());
        let metrics = RiskMetricsCollector::new(registry).unwrap();

        metrics.update_portfolio_metrics(0.05, 5000.0, 0.5, 3, 0.4, 0.8);
        assert_eq!(metrics.portfolio_heat.get(), 0.05);
        assert_eq!(metrics.portfolio_position_count.get(), 3);
    }

    #[test]
    fn test_limit_violation_recording() {
        let registry = Arc::new(Registry::new());
        let metrics = RiskMetricsCollector::new(registry).unwrap();

        metrics.record_position_limit_violation("max_position_size");
        metrics.record_daily_loss_limit_violation();
        metrics.record_portfolio_exposure_violation();

        assert_eq!(metrics.daily_loss_limit_violations.get(), 1);
        assert_eq!(metrics.portfolio_exposure_limit_violations.get(), 1);
    }

    #[test]
    fn test_performance_metrics_update() {
        let registry = Arc::new(Registry::new());
        let metrics = RiskMetricsCollector::new(registry).unwrap();

        let perf_metrics = PerformanceMetrics {
            total: 10,
            wins: 6,
            losses: 4,
            win_rate: 0.6,
            profit_factor: 1.5,
            avg_win: 200.0,
            avg_loss: 100.0,
            expected_value: 80.0,
        };
        metrics.update_performance_metrics(perf_metrics);
        assert_eq!(metrics.win_rate.get(), 0.6);
        assert_eq!(metrics.profit_factor.get(), 1.5);
    }
}
