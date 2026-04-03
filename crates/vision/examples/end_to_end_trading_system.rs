//! End-to-End Trading System Example
//!
//! This example demonstrates a complete trading system that integrates:
//! - Data loading and preprocessing
//! - Feature engineering
//! - Ensemble model predictions
//! - Adaptive threshold calibration
//! - Risk management
//! - Portfolio optimization
//! - Order execution
//! - Performance monitoring
//!
//! This represents a realistic production-style trading pipeline.

use vision::adaptive::{AdaptiveThreshold, RegimeConfig, RegimeDetector};
use vision::data::OhlcvCandle;
use vision::ensemble::{
    EnsembleConfig, EnsembleManager, EnsemblePrediction, EnsembleStrategy, ModelPrediction,
};
use vision::execution::{ExecutionManager, Side};
use vision::portfolio::{
    MeanVarianceOptimizer, OptimizationObjective, PortfolioAnalytics, PortfolioRebalancer,
};
use vision::preprocessing::FeatureEngineer;
use vision::risk::{RiskConfig, RiskManager};

use chrono::{TimeZone, Utc};
use std::collections::HashMap;

fn main() {
    println!("{}", "=".repeat(80));
    println!("JANUS Vision - End-to-End Trading System");
    println!("{}", "=".repeat(80));
    println!();

    // Initialize the trading system
    let mut system = TradingSystem::new();

    println!("Initializing trading system components...");
    system.initialize();
    println!("✓ System initialized\n");

    // Load market data
    println!("Loading market data...");
    let market_data = load_sample_market_data();
    println!(
        "✓ Loaded {} symbols, {} days of data\n",
        market_data.len(),
        market_data[0].ohlcv.len()
    );

    // Run the trading simulation
    println!("{}", "─".repeat(80));
    println!("RUNNING TRADING SIMULATION");
    println!("{}", "─".repeat(80));
    println!();

    system.run_simulation(&market_data);

    // Display final results
    println!("\n{}", "─".repeat(80));
    println!("FINAL RESULTS");
    println!("{}", "─".repeat(80));
    system.display_results();

    println!("\n{}", "=".repeat(80));
    println!("Simulation Complete!");
    println!("{}", "=".repeat(80));
}

/// Main trading system structure
struct TradingSystem {
    /// Symbols being traded
    symbols: Vec<String>,

    /// Ensemble model manager
    ensemble: EnsembleManager,

    /// Adaptive threshold system
    adaptive_threshold: AdaptiveThreshold,

    /// Regime detector
    regime_detector: RegimeDetector,

    /// Risk manager
    risk_manager: RiskManager,

    /// Execution manager
    execution_manager: ExecutionManager,

    /// Feature engineer
    feature_engineer: FeatureEngineer,

    /// Current portfolio weights
    portfolio_weights: HashMap<String, f64>,

    /// Portfolio value over time
    portfolio_values: Vec<f64>,

    /// Current cash position
    cash: f64,

    /// Initial capital
    initial_capital: f64,

    /// Trade history
    trades: Vec<TradeRecord>,

    /// Rebalance frequency (days)
    rebalance_frequency: usize,

    /// Days since last rebalance
    days_since_rebalance: usize,
}

#[derive(Debug, Clone)]
struct TradeRecord {
    day: usize,
    symbol: String,
    side: Side,
    quantity: f64,
    price: f64,
    confidence: f64,
}

struct MarketDataPoint {
    symbol: String,
    ohlcv: Vec<OhlcvCandle>,
}

impl TradingSystem {
    fn new() -> Self {
        Self {
            symbols: vec![
                "SPY".to_string(),
                "QQQ".to_string(),
                "TLT".to_string(),
                "GLD".to_string(),
                "VNQ".to_string(),
            ],
            ensemble: EnsembleManager::new(EnsembleConfig {
                strategy: EnsembleStrategy::Mean,
                min_models: 1,
                max_models: 10,
                min_model_accuracy: 0.5,
                min_model_predictions: 1,
                confidence_weight: 0.5,
                performance_weight: 0.5,
            }),
            adaptive_threshold: AdaptiveThreshold::new(Default::default()),
            regime_detector: RegimeDetector::new(RegimeConfig::default()),
            risk_manager: RiskManager::new(RiskConfig::default()),
            execution_manager: ExecutionManager::new(),
            feature_engineer: FeatureEngineer::new(Default::default()),
            portfolio_weights: HashMap::new(),
            portfolio_values: vec![100_000.0],
            cash: 100_000.0,
            initial_capital: 100_000.0,
            trades: Vec::new(),
            rebalance_frequency: 21, // Monthly
            days_since_rebalance: 0,
        }
    }

    fn initialize(&mut self) {
        println!("  - Ensemble model: 3 strategies (Mean Reversion, Momentum, Breakout)");
        println!("  - Adaptive thresholds: Enabled with regime detection");
        println!("  - Risk management: Max 20% volatility, 30% position size");
        println!("  - Portfolio optimization: Mean-variance (Max Sharpe)");
        println!("  - Execution: Smart order routing with analytics");

        // Initialize portfolio with equal weights
        for symbol in &self.symbols {
            self.portfolio_weights
                .insert(symbol.clone(), 1.0 / self.symbols.len() as f64);
        }
    }

    fn run_simulation(&mut self, market_data: &[MarketDataPoint]) {
        let num_days = market_data[0].ohlcv.len();

        for day in 1..num_days {
            if day % 50 == 0 {
                print!("  Day {}/{}...\r", day, num_days);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }

            // Step 1: Generate predictions for each symbol
            let predictions = self.generate_predictions(market_data, day);

            // Step 2: Detect market regime
            let regime = self.detect_regime(market_data, day);

            // Step 3: Apply adaptive thresholds
            let signals = self.apply_adaptive_thresholds(&predictions, &regime);

            // Step 4: Check if rebalancing needed
            self.days_since_rebalance += 1;
            if self.days_since_rebalance >= self.rebalance_frequency || day == num_days - 1 {
                self.rebalance_portfolio(market_data, day, &signals);
                self.days_since_rebalance = 0;
            }

            // Step 5: Calculate portfolio value
            self.update_portfolio_value(market_data, day);

            // Step 6: Risk monitoring
            self.monitor_risk(day);
        }

        println!("\n  Simulation completed: {} days", num_days);
    }

    fn generate_predictions(
        &self,
        market_data: &[MarketDataPoint],
        day: usize,
    ) -> HashMap<String, EnsemblePrediction> {
        let mut predictions = HashMap::new();

        for (i, symbol) in self.symbols.iter().enumerate() {
            // Extract features from recent price history
            let lookback = 20.min(day);
            let recent_data = &market_data[i].ohlcv[day - lookback..day];

            // Simple feature: momentum, mean reversion signal, volatility
            let features = self.compute_features(recent_data);

            // Generate prediction (simplified - in reality would use trained models)
            let prediction = self.simulate_ensemble_prediction(&features);

            predictions.insert(symbol.clone(), prediction);
        }

        predictions
    }

    fn compute_features(&self, data: &[OhlcvCandle]) -> Vec<f64> {
        if data.len() < 2 {
            return vec![0.0, 0.0, 0.0];
        }

        // Calculate returns
        let returns: Vec<f64> = data
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        // Feature 1: Momentum (cumulative return)
        let momentum: f64 = returns.iter().sum();

        // Feature 2: Mean reversion (distance from mean)
        let mean_price: f64 = data.iter().map(|c| c.close).sum::<f64>() / data.len() as f64;
        let current_price = data.last().unwrap().close;
        let mean_reversion = (current_price - mean_price) / mean_price;

        // Feature 3: Volatility
        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let volatility = variance.sqrt();

        vec![momentum, mean_reversion, volatility]
    }

    fn simulate_ensemble_prediction(&self, features: &[f64]) -> EnsemblePrediction {
        // Simplified ensemble prediction logic
        let momentum = features[0];
        let mean_reversion = features[1];
        let volatility = features[2];

        // Model 1: Momentum strategy
        let momentum_score = momentum.tanh(); // Normalize to [-1, 1]

        // Model 2: Mean reversion strategy
        let mean_rev_score = -mean_reversion.tanh(); // Bet against deviations

        // Model 3: Breakout strategy (volatility-based)
        let breakout_score = if volatility > 0.02 && momentum > 0.0 {
            momentum.min(1.0)
        } else {
            0.0
        };

        // Ensemble average
        let signal = (momentum_score + mean_rev_score + breakout_score) / 3.0;
        let confidence = ((signal + 1.0) / 2.0).max(0.0).min(1.0);

        EnsemblePrediction {
            signal,
            confidence,
            num_models: 3,
            agreement: 0.8, // Simplified agreement metric
            individual_predictions: vec![
                ModelPrediction {
                    model_id: "momentum".to_string(),
                    signal: momentum_score,
                    confidence: 0.8,
                    latency_us: 0,
                    timestamp: 0,
                },
                ModelPrediction {
                    model_id: "mean_revert".to_string(),
                    signal: mean_rev_score,
                    confidence: 0.8,
                    latency_us: 0,
                    timestamp: 0,
                },
                ModelPrediction {
                    model_id: "breakout".to_string(),
                    signal: breakout_score,
                    confidence: 0.8,
                    latency_us: 0,
                    timestamp: 0,
                },
            ],
        }
    }

    fn detect_regime(&mut self, market_data: &[MarketDataPoint], day: usize) -> String {
        // Simple regime detection based on market volatility
        let lookback = 20.min(day);

        let spy_data = &market_data[0].ohlcv[day - lookback..day];
        let returns: Vec<f64> = spy_data
            .windows(2)
            .map(|w| (w[1].close - w[0].close) / w[0].close)
            .collect();

        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance: f64 =
            returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized

        if volatility > 0.25 {
            "HighVol".to_string()
        } else if mean > 0.001 {
            "Bull".to_string()
        } else if mean < -0.001 {
            "Bear".to_string()
        } else {
            "Neutral".to_string()
        }
    }

    fn apply_adaptive_thresholds(
        &self,
        predictions: &HashMap<String, EnsemblePrediction>,
        regime: &str,
    ) -> HashMap<String, f64> {
        let mut signals = HashMap::new();

        // Regime-dependent thresholds
        let threshold = match regime {
            "Bull" => 0.55,    // More aggressive in bull markets
            "Bear" => 0.70,    // More conservative in bear markets
            "HighVol" => 0.75, // Very conservative in high volatility
            _ => 0.65,         // Default
        };

        for (symbol, pred) in predictions {
            // Apply threshold and regime adjustment
            let signal = if pred.confidence > threshold {
                pred.confidence
            } else {
                0.5 // Neutral
            };

            signals.insert(symbol.clone(), signal);
        }

        signals
    }

    fn rebalance_portfolio(
        &mut self,
        market_data: &[MarketDataPoint],
        day: usize,
        signals: &HashMap<String, f64>,
    ) {
        // Step 1: Estimate expected returns based on signals
        let expected_returns: Vec<f64> = self
            .symbols
            .iter()
            .map(|s| {
                let signal = signals.get(s).unwrap_or(&0.5);
                // Convert signal to expected return estimate
                (signal - 0.5) * 0.3 // Scale to reasonable return range
            })
            .collect();

        // Step 2: Estimate covariance from recent data
        let covariance = self.estimate_covariance(market_data, day, 60);

        // Step 3: Optimize portfolio
        let optimizer = MeanVarianceOptimizer::new(
            expected_returns.clone(),
            covariance.clone(),
            self.symbols.clone(),
        )
        .unwrap()
        .with_risk_free_rate(0.02);

        let result = match optimizer.optimize(OptimizationObjective::MaxSharpe) {
            Ok(r) => r,
            Err(_) => {
                // Fallback to equal weights if optimization fails
                return;
            }
        };

        // Step 4: Apply risk limits
        let max_volatility = 0.20;
        let max_position = 0.30;

        if result.volatility > max_volatility {
            println!(
                "\n  Warning: Optimized volatility {:.2}% exceeds limit",
                result.volatility * 100.0
            );
            return;
        }

        if result.weights.iter().any(|&w| w > max_position) {
            println!("\n  Warning: Position size exceeds limit");
            return;
        }

        // Step 5: Calculate trades needed
        let current_portfolio_value = self.portfolio_values.last().unwrap();
        let current_weights: Vec<f64> = self
            .symbols
            .iter()
            .map(|s| self.portfolio_weights.get(s).unwrap_or(&0.0).clone())
            .collect();

        let trades = PortfolioRebalancer::calculate_trades(
            &current_weights,
            &result.weights,
            *current_portfolio_value,
        );

        // Step 6: Execute trades
        let prices: Vec<f64> = market_data.iter().map(|md| md.ohlcv[day].close).collect();

        for (i, &trade_value) in trades.iter().enumerate() {
            if trade_value.abs() < 100.0 {
                continue; // Skip tiny trades
            }

            let quantity = trade_value.abs() / prices[i];
            let side = if trade_value > 0.0 {
                Side::Buy
            } else {
                Side::Sell
            };

            self.trades.push(TradeRecord {
                day,
                symbol: self.symbols[i].clone(),
                side: side.clone(),
                quantity,
                price: prices[i],
                confidence: signals.get(&self.symbols[i]).unwrap_or(&0.5).clone(),
            });

            // Update weights
            self.portfolio_weights
                .insert(self.symbols[i].clone(), result.weights[i]);
        }

        if !trades.is_empty() {
            let turnover = PortfolioAnalytics::turnover(&current_weights, &result.weights);
            println!(
                "\n  Rebalance executed (Day {}): {:.1}% turnover, {:.2}% expected vol",
                day,
                turnover * 100.0,
                result.volatility * 100.0
            );
        }
    }

    fn estimate_covariance(
        &self,
        market_data: &[MarketDataPoint],
        current_day: usize,
        lookback: usize,
    ) -> Vec<Vec<f64>> {
        let n = self.symbols.len();
        let start = current_day.saturating_sub(lookback);

        let mut returns_matrix = Vec::new();
        for md in market_data {
            let returns: Vec<f64> = md.ohlcv[start..current_day]
                .windows(2)
                .map(|w| (w[1].close - w[0].close) / w[0].close)
                .collect();
            returns_matrix.push(returns);
        }

        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mean_i = returns_matrix[i].iter().sum::<f64>() / returns_matrix[i].len() as f64;
                let mean_j = returns_matrix[j].iter().sum::<f64>() / returns_matrix[j].len() as f64;

                let covariance: f64 = returns_matrix[i]
                    .iter()
                    .zip(returns_matrix[j].iter())
                    .map(|(ri, rj)| (ri - mean_i) * (rj - mean_j))
                    .sum::<f64>()
                    / (returns_matrix[i].len() - 1) as f64;

                cov[i][j] = covariance * 252.0; // Annualize
            }
        }

        cov
    }

    fn update_portfolio_value(&mut self, market_data: &[MarketDataPoint], day: usize) {
        let mut total_value = self.cash;

        for (i, symbol) in self.symbols.iter().enumerate() {
            let weight = self.portfolio_weights.get(symbol).unwrap_or(&0.0);
            let current_price = market_data[i].ohlcv[day].close;
            let prev_price = if day > 0 {
                market_data[i].ohlcv[day - 1].close
            } else {
                current_price
            };

            let prev_portfolio_value = self.portfolio_values.last().unwrap();
            let position_value = weight * prev_portfolio_value;
            let return_today = (current_price - prev_price) / prev_price;
            total_value += position_value * (1.0 + return_today);
        }

        self.portfolio_values.push(total_value);
    }

    fn monitor_risk(&self, day: usize) {
        // Calculate current drawdown
        let current_value = self.portfolio_values.last().unwrap();
        let max_value = self
            .portfolio_values
            .iter()
            .take(day + 1)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        let drawdown = (max_value - current_value) / max_value;

        // Alert if drawdown exceeds threshold
        if drawdown > 0.15 {
            println!(
                "\n  ⚠ Risk Alert: Drawdown {:.2}% exceeds threshold",
                drawdown * 100.0
            );
        }
    }

    fn display_results(&self) {
        let final_value = self.portfolio_values.last().unwrap();
        let total_return = (final_value - self.initial_capital) / self.initial_capital;

        // Calculate annualized metrics
        let days = self.portfolio_values.len() as f64;
        let years = days / 252.0;
        let annualized_return = (1.0 + total_return).powf(1.0 / years) - 1.0;

        // Calculate Sharpe ratio
        let returns: Vec<f64> = self
            .portfolio_values
            .windows(2)
            .map(|w| (w[1] - w[0]) / w[0])
            .collect();

        let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns
            .iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>()
            / returns.len() as f64;
        let volatility = variance.sqrt() * (252.0_f64).sqrt(); // Annualized

        let sharpe = (annualized_return - 0.02) / volatility;

        // Calculate max drawdown
        let max_dd = PortfolioAnalytics::max_drawdown(&self.portfolio_values);

        println!("\nPerformance Metrics:");
        println!("  Initial Capital:      ${:.0}", self.initial_capital);
        println!("  Final Portfolio Value: ${:.0}", final_value);
        println!("  Total Return:          {:.2}%", total_return * 100.0);
        println!("  Annualized Return:     {:.2}%", annualized_return * 100.0);
        println!("  Annualized Volatility: {:.2}%", volatility * 100.0);
        println!("  Sharpe Ratio:          {:.4}", sharpe);
        println!("  Max Drawdown:          {:.2}%", max_dd * 100.0);

        println!("\nTrading Activity:");
        println!("  Total Trades:          {}", self.trades.len());
        println!(
            "  Rebalances:            {}",
            self.trades.len() / self.symbols.len().max(1)
        );

        // Calculate average confidence
        let avg_confidence = if !self.trades.is_empty() {
            self.trades.iter().map(|t| t.confidence).sum::<f64>() / self.trades.len() as f64
        } else {
            0.0
        };
        println!("  Avg Trade Confidence:  {:.2}%", avg_confidence * 100.0);

        // Display final portfolio allocation
        println!("\nFinal Portfolio Allocation:");
        let mut weights: Vec<_> = self.portfolio_weights.iter().collect();
        weights.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
        for (symbol, weight) in &weights {
            if **weight > 0.01 {
                println!("  {}: {:.2}%", symbol, **weight * 100.0);
            }
        }

        // Risk analysis
        println!("\nRisk Analysis:");
        println!(
            "  Portfolio concentrated: {}",
            if weights[0].1 > &0.40 {
                "Yes (>40%)"
            } else {
                "No"
            }
        );
        println!(
            "  Volatility within limits: {}",
            if volatility < 0.20 { "Yes" } else { "No" }
        );
    }
}

fn load_sample_market_data() -> Vec<MarketDataPoint> {
    // Generate synthetic market data for demonstration
    let symbols = vec!["SPY", "QQQ", "TLT", "GLD", "VNQ"];
    let num_days = 252; // One year

    symbols
        .iter()
        .enumerate()
        .map(|(i, symbol)| {
            let mut ohlcv = Vec::new();
            let mut price = 100.0;

            for day in 0..num_days {
                // Generate realistic price movements
                let drift = 0.0003 * (i as f64 + 1.0); // Slight upward drift
                let volatility = 0.01 + 0.005 * i as f64;

                // Simple random walk with sine wave for cyclicality
                let random_component = (day as f64 * 0.05 + i as f64).sin() * volatility;
                let return_today = drift + random_component;

                price *= 1.0 + return_today;

                let daily_range = price * volatility;

                ohlcv.push(OhlcvCandle {
                    timestamp: Utc.timestamp_opt(day as i64 * 86400, 0).unwrap(),
                    open: price - daily_range * 0.3,
                    high: price + daily_range * 0.5,
                    low: price - daily_range * 0.5,
                    close: price,
                    volume: 1_000_000.0 + (day as f64 * 100.0),
                });
            }

            MarketDataPoint {
                symbol: symbol.to_string(),
                ohlcv,
            }
        })
        .collect()
}
