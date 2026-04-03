//! Signal Backtesting Tool
//!
//! Analyzes historical signal performance by fetching signals from QuestDB
//! and calculating win rates, average returns, and other performance metrics.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use signal_backtest::{SignalBacktest, BacktestConfig};
//!
//! let config = BacktestConfig {
//!     lookforward_minutes: 60,
//!     profit_target_pct: 1.0,
//!     stop_loss_pct: 0.5,
//! };
//!
//! let backtest = SignalBacktest::new(storage, config);
//! let results = backtest.run("BTCUSD", "1m", None, None).await?;
//! println!("Win rate: {}%", results.win_rate);
//! ```

use anyhow::{Context, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::storage::StorageManager;

/// Backtesting configuration
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// How many minutes to look forward after signal to measure performance
    pub lookforward_minutes: i64,

    /// Profit target percentage (e.g., 1.0 = 1%)
    pub profit_target_pct: f64,

    /// Stop loss percentage (e.g., 0.5 = 0.5%)
    pub stop_loss_pct: f64,

    /// Minimum number of signals required for statistical significance
    pub min_signals: usize,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            lookforward_minutes: 60, // 1 hour lookforward
            profit_target_pct: 1.0,  // 1% profit target
            stop_loss_pct: 0.5,      // 0.5% stop loss
            min_signals: 10,         // At least 10 signals
        }
    }
}

/// Individual signal result
#[derive(Debug, Clone, Serialize)]
pub struct SignalResult {
    pub signal_type: String,
    pub direction: String,
    pub timestamp: String,
    pub entry_price: f64,
    pub exit_price: f64,
    pub return_pct: f64,
    pub outcome: TradeOutcome,
    pub duration_minutes: i64,
}

/// Trade outcome classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TradeOutcome {
    /// Hit profit target
    Win,
    /// Hit stop loss
    Loss,
    /// Neither target hit within lookforward period
    Neutral,
}

/// Backtesting results for a symbol/timeframe
#[derive(Debug, Clone, Serialize)]
pub struct BacktestResults {
    pub symbol: String,
    pub timeframe: String,
    pub total_signals: usize,
    pub total_trades: usize,
    pub wins: usize,
    pub losses: usize,
    pub neutral: usize,
    pub win_rate: f64,
    pub loss_rate: f64,
    pub avg_return: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub max_win: f64,
    pub max_loss: f64,
    pub profit_factor: f64,
    pub sharpe_ratio: f64,
    pub results_by_type: HashMap<String, SignalTypeMetrics>,
    pub individual_results: Vec<SignalResult>,
}

/// Performance metrics per signal type
#[derive(Debug, Clone, Serialize)]
pub struct SignalTypeMetrics {
    pub signal_type: String,
    pub count: usize,
    pub wins: usize,
    pub losses: usize,
    pub win_rate: f64,
    pub avg_return: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
}

/// Historical signal from QuestDB
#[derive(Debug)]
struct HistoricalSignal {
    signal_type: String,
    direction: String,
    timestamp: String,
    price: f64,
}

/// Price data point
#[derive(Debug)]
struct PricePoint {
    #[allow(dead_code)]
    timestamp: String,
    price: f64,
}

/// Signal backtesting engine
pub struct SignalBacktest {
    #[allow(dead_code)]
    storage: Arc<StorageManager>,
    config: BacktestConfig,
}

impl SignalBacktest {
    /// Create a new backtesting engine
    pub fn new(storage: Arc<StorageManager>, config: BacktestConfig) -> Self {
        Self { storage, config }
    }

    /// Run backtest for a symbol/timeframe
    ///
    /// # Arguments
    /// * `symbol` - Trading pair (e.g., "BTCUSD")
    /// * `timeframe` - Candle interval (e.g., "1m")
    /// * `start_time` - Optional start time filter (ISO 8601)
    /// * `end_time` - Optional end time filter (ISO 8601)
    pub async fn run(
        &self,
        symbol: &str,
        timeframe: &str,
        start_time: Option<&str>,
        end_time: Option<&str>,
    ) -> Result<BacktestResults> {
        info!(
            "Starting backtest for {}:{} (lookforward: {}m)",
            symbol, timeframe, self.config.lookforward_minutes
        );

        // Fetch historical signals
        let signals = self
            .fetch_signals(symbol, timeframe, start_time, end_time)
            .await?;

        if signals.is_empty() {
            warn!("No signals found for {}:{}", symbol, timeframe);
            return Ok(BacktestResults::empty(symbol, timeframe));
        }

        info!("Found {} signals to backtest", signals.len());

        // Analyze each signal
        let mut individual_results = Vec::new();
        for signal in &signals {
            match self.analyze_signal(symbol, timeframe, signal).await {
                Ok(result) => individual_results.push(result),
                Err(e) => {
                    warn!("Failed to analyze signal: {}", e);
                    continue;
                }
            }
        }

        // Calculate aggregate metrics
        let results = self.calculate_metrics(symbol, timeframe, signals.len(), individual_results);

        info!(
            "Backtest complete: {} trades, {:.2}% win rate, {:.3}% avg return",
            results.total_trades, results.win_rate, results.avg_return
        );

        Ok(results)
    }

    /// Fetch historical signals from QuestDB
    async fn fetch_signals(
        &self,
        symbol: &str,
        timeframe: &str,
        start_time: Option<&str>,
        end_time: Option<&str>,
    ) -> Result<Vec<HistoricalSignal>> {
        let questdb_host = std::env::var("QUESTDB_HOST").unwrap_or_else(|_| "questdb".to_string());
        let questdb_http_port =
            std::env::var("QUESTDB_HTTP_PORT").unwrap_or_else(|_| "9000".to_string());
        let url = format!("http://{}:{}/exec", questdb_host, questdb_http_port);

        // Build query
        let mut sql = format!(
            "SELECT signal_type, direction, timestamp, price FROM signals_crypto WHERE symbol = '{}' AND timeframe = '{}'",
            symbol, timeframe
        );

        if let Some(start) = start_time {
            sql.push_str(&format!(" AND timestamp >= '{}'", start));
        }

        if let Some(end) = end_time {
            sql.push_str(&format!(" AND timestamp < '{}'", end));
        }

        sql.push_str(" ORDER BY timestamp ASC");

        debug!("Fetching signals: {}", sql);

        // Execute query
        let client = reqwest::Client::new();
        let response = client.get(&url).query(&[("query", &sql)]).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("QuestDB query failed: {}", error_text));
        }

        let json: serde_json::Value = response.json().await?;
        let dataset = json["dataset"]
            .as_array()
            .context("Missing dataset in response")?;

        let mut signals = Vec::new();
        for row in dataset {
            let row_array = row.as_array().context("Invalid row format")?;
            if row_array.len() >= 4 {
                signals.push(HistoricalSignal {
                    signal_type: row_array[0].as_str().unwrap_or_default().to_string(),
                    direction: row_array[1].as_str().unwrap_or_default().to_string(),
                    timestamp: row_array[2].as_str().unwrap_or_default().to_string(),
                    price: row_array[3].as_f64().unwrap_or(0.0),
                });
            }
        }

        Ok(signals)
    }

    /// Fetch price data after a signal
    async fn fetch_price_data(
        &self,
        symbol: &str,
        timeframe: &str,
        signal_time: &str,
    ) -> Result<Vec<PricePoint>> {
        let questdb_host = std::env::var("QUESTDB_HOST").unwrap_or_else(|_| "questdb".to_string());
        let questdb_http_port =
            std::env::var("QUESTDB_HTTP_PORT").unwrap_or_else(|_| "9000".to_string());
        let url = format!("http://{}:{}/exec", questdb_host, questdb_http_port);

        // Calculate end time (signal time + lookforward)
        let sql = format!(
            "SELECT timestamp, close FROM candles_crypto WHERE symbol = '{}' AND interval = '{}' AND timestamp > '{}' ORDER BY timestamp ASC LIMIT {}",
            symbol,
            timeframe,
            signal_time,
            self.config.lookforward_minutes + 1
        );

        debug!("Fetching price data: {}", sql);

        let client = reqwest::Client::new();
        let response = client.get(&url).query(&[("query", &sql)]).send().await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(anyhow::anyhow!("QuestDB query failed: {}", error_text));
        }

        let json: serde_json::Value = response.json().await?;
        let dataset = json["dataset"]
            .as_array()
            .context("Missing dataset in response")?;

        let mut prices = Vec::new();
        for row in dataset {
            let row_array = row.as_array().context("Invalid row format")?;
            if row_array.len() >= 2 {
                prices.push(PricePoint {
                    timestamp: row_array[0].as_str().unwrap_or_default().to_string(),
                    price: row_array[1].as_f64().unwrap_or(0.0),
                });
            }
        }

        Ok(prices)
    }

    /// Analyze a single signal's performance
    async fn analyze_signal(
        &self,
        symbol: &str,
        timeframe: &str,
        signal: &HistoricalSignal,
    ) -> Result<SignalResult> {
        let entry_price = signal.price;

        // Fetch subsequent price data
        let prices = self
            .fetch_price_data(symbol, timeframe, &signal.timestamp)
            .await?;

        if prices.is_empty() {
            // No data after signal - mark as neutral
            return Ok(SignalResult {
                signal_type: signal.signal_type.clone(),
                direction: signal.direction.clone(),
                timestamp: signal.timestamp.clone(),
                entry_price,
                exit_price: entry_price,
                return_pct: 0.0,
                outcome: TradeOutcome::Neutral,
                duration_minutes: 0,
            });
        }

        // Determine profit target and stop loss based on direction
        let (profit_target, stop_loss) = if signal.direction == "bullish" {
            (
                entry_price * (1.0 + self.config.profit_target_pct / 100.0),
                entry_price * (1.0 - self.config.stop_loss_pct / 100.0),
            )
        } else {
            (
                entry_price * (1.0 - self.config.profit_target_pct / 100.0),
                entry_price * (1.0 + self.config.stop_loss_pct / 100.0),
            )
        };

        // Scan price action for outcome
        let mut outcome = TradeOutcome::Neutral;
        let mut exit_price = entry_price;
        let mut duration_minutes = 0;

        for (i, price_point) in prices.iter().enumerate() {
            duration_minutes = i as i64;

            if signal.direction == "bullish" {
                // Long position
                if price_point.price >= profit_target {
                    outcome = TradeOutcome::Win;
                    exit_price = profit_target;
                    break;
                } else if price_point.price <= stop_loss {
                    outcome = TradeOutcome::Loss;
                    exit_price = stop_loss;
                    break;
                }
            } else {
                // Short position
                if price_point.price <= profit_target {
                    outcome = TradeOutcome::Win;
                    exit_price = profit_target;
                    break;
                } else if price_point.price >= stop_loss {
                    outcome = TradeOutcome::Loss;
                    exit_price = stop_loss;
                    break;
                }
            }
        }

        // If no exit triggered, use last price
        if outcome == TradeOutcome::Neutral && !prices.is_empty() {
            exit_price = prices.last().unwrap().price;
        }

        // Calculate return
        let return_pct = if signal.direction == "bullish" {
            ((exit_price - entry_price) / entry_price) * 100.0
        } else {
            ((entry_price - exit_price) / entry_price) * 100.0
        };

        Ok(SignalResult {
            signal_type: signal.signal_type.clone(),
            direction: signal.direction.clone(),
            timestamp: signal.timestamp.clone(),
            entry_price,
            exit_price,
            return_pct,
            outcome,
            duration_minutes,
        })
    }

    /// Calculate aggregate metrics from individual results
    fn calculate_metrics(
        &self,
        symbol: &str,
        timeframe: &str,
        total_signals: usize,
        results: Vec<SignalResult>,
    ) -> BacktestResults {
        let total_trades = results.len();

        if total_trades == 0 {
            return BacktestResults::empty(symbol, timeframe);
        }

        let mut wins = 0;
        let mut losses = 0;
        let mut neutral = 0;
        let mut total_return = 0.0;
        let mut total_win_return = 0.0;
        let mut total_loss_return = 0.0;
        let mut max_win = f64::MIN;
        let mut max_loss = f64::MAX;

        // Per-signal-type metrics
        let mut type_metrics: HashMap<String, Vec<f64>> = HashMap::new();
        let mut type_outcomes: HashMap<String, (usize, usize, usize)> = HashMap::new();

        for result in &results {
            match result.outcome {
                TradeOutcome::Win => {
                    wins += 1;
                    total_win_return += result.return_pct;
                    max_win = max_win.max(result.return_pct);

                    let (w, _l, _n) = type_outcomes
                        .entry(result.signal_type.clone())
                        .or_insert((0, 0, 0));
                    *w += 1;
                }
                TradeOutcome::Loss => {
                    losses += 1;
                    total_loss_return += result.return_pct;
                    max_loss = max_loss.min(result.return_pct);

                    let (_w, l, _n) = type_outcomes
                        .entry(result.signal_type.clone())
                        .or_insert((0, 0, 0));
                    *l += 1;
                }
                TradeOutcome::Neutral => {
                    neutral += 1;

                    let (_w, _l, n) = type_outcomes
                        .entry(result.signal_type.clone())
                        .or_insert((0, 0, 0));
                    *n += 1;
                }
            }

            total_return += result.return_pct;
            type_metrics
                .entry(result.signal_type.clone())
                .or_default()
                .push(result.return_pct);
        }

        let win_rate = if total_trades > 0 {
            (wins as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        let loss_rate = if total_trades > 0 {
            (losses as f64 / total_trades as f64) * 100.0
        } else {
            0.0
        };

        let avg_return = total_return / total_trades as f64;
        let avg_win = if wins > 0 {
            total_win_return / wins as f64
        } else {
            0.0
        };
        let avg_loss = if losses > 0 {
            total_loss_return / losses as f64
        } else {
            0.0
        };

        // Profit factor: total wins / total losses (absolute)
        let profit_factor = if total_loss_return.abs() > 0.0 {
            total_win_return.abs() / total_loss_return.abs()
        } else {
            0.0
        };

        // Sharpe ratio (simplified): avg return / std dev
        let mean = avg_return;
        let variance = results
            .iter()
            .map(|r| (r.return_pct - mean).powi(2))
            .sum::<f64>()
            / total_trades as f64;
        let std_dev = variance.sqrt();
        let sharpe_ratio = if std_dev > 0.0 { mean / std_dev } else { 0.0 };

        // Build per-type metrics
        let mut results_by_type = HashMap::new();
        for (signal_type, returns) in type_metrics {
            let (w, l, _n) = type_outcomes.get(&signal_type).unwrap_or(&(0, 0, 0));
            let count = returns.len();
            let type_avg = returns.iter().sum::<f64>() / count as f64;
            let wins_only: Vec<f64> = returns.iter().copied().filter(|r| *r > 0.0).collect();
            let losses_only: Vec<f64> = returns.iter().copied().filter(|r| *r < 0.0).collect();

            let type_avg_win = if !wins_only.is_empty() {
                wins_only.iter().sum::<f64>() / wins_only.len() as f64
            } else {
                0.0
            };

            let type_avg_loss = if !losses_only.is_empty() {
                losses_only.iter().sum::<f64>() / losses_only.len() as f64
            } else {
                0.0
            };

            let type_win_rate = if count > 0 {
                (*w as f64 / count as f64) * 100.0
            } else {
                0.0
            };

            results_by_type.insert(
                signal_type.clone(),
                SignalTypeMetrics {
                    signal_type,
                    count,
                    wins: *w,
                    losses: *l,
                    win_rate: type_win_rate,
                    avg_return: type_avg,
                    avg_win: type_avg_win,
                    avg_loss: type_avg_loss,
                },
            );
        }

        BacktestResults {
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            total_signals,
            total_trades,
            wins,
            losses,
            neutral,
            win_rate,
            loss_rate,
            avg_return,
            avg_win,
            avg_loss,
            max_win,
            max_loss,
            profit_factor,
            sharpe_ratio,
            results_by_type,
            individual_results: results,
        }
    }
}

impl BacktestResults {
    /// Create empty results
    fn empty(symbol: &str, timeframe: &str) -> Self {
        Self {
            symbol: symbol.to_string(),
            timeframe: timeframe.to_string(),
            total_signals: 0,
            total_trades: 0,
            wins: 0,
            losses: 0,
            neutral: 0,
            win_rate: 0.0,
            loss_rate: 0.0,
            avg_return: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            max_win: 0.0,
            max_loss: 0.0,
            profit_factor: 0.0,
            sharpe_ratio: 0.0,
            results_by_type: HashMap::new(),
            individual_results: Vec::new(),
        }
    }

    /// Print human-readable summary
    pub fn print_summary(&self) {
        println!("\n╔════════════════════════════════════════════════════════╗");
        println!("║           SIGNAL BACKTEST RESULTS                     ║");
        println!("╚════════════════════════════════════════════════════════╝");
        println!();
        println!("  Symbol:         {}", self.symbol);
        println!("  Timeframe:      {}", self.timeframe);
        println!("  Total Signals:  {}", self.total_signals);
        println!("  Total Trades:   {}", self.total_trades);
        println!();
        println!("  Wins:           {}", self.wins);
        println!("  Losses:         {}", self.losses);
        println!("  Neutral:        {}", self.neutral);
        println!("  Win Rate:       {:.2}%", self.win_rate);
        println!("  Loss Rate:      {:.2}%", self.loss_rate);
        println!();
        println!("  Avg Return:     {:.3}%", self.avg_return);
        println!("  Avg Win:        {:.3}%", self.avg_win);
        println!("  Avg Loss:       {:.3}%", self.avg_loss);
        println!("  Max Win:        {:.3}%", self.max_win);
        println!("  Max Loss:       {:.3}%", self.max_loss);
        println!();
        println!("  Profit Factor:  {:.2}", self.profit_factor);
        println!("  Sharpe Ratio:   {:.2}", self.sharpe_ratio);
        println!();

        if !self.results_by_type.is_empty() {
            println!("  Per-Signal-Type Metrics:");
            println!("  ════════════════════════");
            for (signal_type, metrics) in &self.results_by_type {
                println!(
                    "    {}: {} trades, {:.1}% win rate, {:.3}% avg return",
                    signal_type, metrics.count, metrics.win_rate, metrics.avg_return
                );
            }
            println!();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BacktestConfig::default();
        assert_eq!(config.lookforward_minutes, 60);
        assert_eq!(config.profit_target_pct, 1.0);
        assert_eq!(config.stop_loss_pct, 0.5);
        assert_eq!(config.min_signals, 10);
    }

    #[test]
    fn test_empty_results() {
        let results = BacktestResults::empty("BTCUSD", "1m");
        assert_eq!(results.symbol, "BTCUSD");
        assert_eq!(results.timeframe, "1m");
        assert_eq!(results.total_trades, 0);
        assert_eq!(results.win_rate, 0.0);
    }

    #[test]
    fn test_trade_outcome_serialization() {
        let outcome = TradeOutcome::Win;
        let json = serde_json::to_string(&outcome).unwrap();
        assert_eq!(json, r#""win""#);
    }
}
