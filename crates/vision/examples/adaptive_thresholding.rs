//! Adaptive Thresholding and Dynamic Calibration Example
//!
//! This example demonstrates how to use the adaptive thresholding system to:
//! 1. Detect market regimes (trending, ranging, volatile)
//! 2. Dynamically adjust confidence thresholds based on performance
//! 3. Calibrate model predictions to actual probabilities
//! 4. Combine all components for adaptive trading decisions

use vision::adaptive::{
    AdaptiveSystem, CalibrationConfig, MarketRegime, RegimeAdjuster, RegimeConfig, RegimeDetector,
    RiskLevel, ThresholdConfig,
};

fn main() {
    println!("=== Adaptive Thresholding & Dynamic Calibration Demo ===\n");

    // Example 1: Market Regime Detection
    regime_detection_example();

    // Example 2: Dynamic Threshold Calibration
    threshold_calibration_example();

    // Example 3: Confidence Calibration
    confidence_calibration_example();

    // Example 4: Complete Adaptive System
    complete_adaptive_system_example();

    // Example 5: Real-world Scenario
    realistic_trading_scenario();
}

fn regime_detection_example() {
    println!("--- Example 1: Market Regime Detection ---");

    let config = RegimeConfig {
        trend_window: 20,
        volatility_window: 20,
        adx_threshold: 25.0,
        volatility_threshold: 0.7,
        min_data_points: 20,
    };

    let mut detector = RegimeDetector::new(config);

    // Simulate different market conditions

    // 1. Uptrend
    println!("\n1. Simulating uptrend...");
    for i in 0..30 {
        let price = 100.0 + i as f64 * 2.0;
        let regime = detector.update(price);
        if i >= 20 {
            println!("  Price: {:.2}, Regime: {:?}", price, regime);
        }
    }

    // 2. Ranging market
    println!("\n2. Simulating ranging market...");
    detector.reset();
    for i in 0..30 {
        let price = 100.0 + ((i as f64 * 0.5).sin() * 3.0);
        let regime = detector.update(price);
        if i >= 20 {
            println!("  Price: {:.2}, Regime: {:?}", price, regime);
        }
    }

    // 3. Volatile market
    println!("\n3. Simulating volatile market...");
    detector.reset();
    for i in 0..30 {
        let price = 100.0 + ((i as f64 * 0.2).sin() * 10.0);
        let regime = detector.update(price);
        if i >= 20 {
            println!("  Price: {:.2}, Regime: {:?}", price, regime);
        }
    }

    println!();
}

fn threshold_calibration_example() {
    println!("--- Example 2: Dynamic Threshold Calibration ---");

    let config = ThresholdConfig {
        base_confidence: 0.7,
        lookback_window: 100,
        min_samples: 20,
        max_adjustment: 0.3,
        target_win_rate: 0.55,
        learning_rate: 0.01,
    };

    let mut calibrator = vision::adaptive::ThresholdCalibrator::new(config.clone());

    println!("\nInitial threshold: {:.3}", calibrator.current_threshold());

    // Simulate trades with varying win rates
    println!("\nSimulating trades with 70% win rate...");
    for i in 0..50 {
        let confidence = 0.6 + (i as f64 / 100.0);
        let profit = if i % 10 < 7 { 100.0 } else { -50.0 };
        calibrator.update(confidence, profit);

        if i % 10 == 9 {
            let stats = calibrator.statistics();
            println!(
                "  Trades: {}, Win Rate: {:.2}%, Threshold: {:.3}",
                stats.sample_count,
                stats.win_rate * 100.0,
                stats.current_threshold
            );
        }
    }

    println!("\nSimulating trades with 30% win rate...");
    calibrator.reset();
    for i in 0..50 {
        let confidence = 0.6 + (i as f64 / 100.0);
        let profit = if i % 10 < 3 { 100.0 } else { -50.0 };
        calibrator.update(confidence, profit);

        if i % 10 == 9 {
            let stats = calibrator.statistics();
            println!(
                "  Trades: {}, Win Rate: {:.2}%, Threshold: {:.3}",
                stats.sample_count,
                stats.win_rate * 100.0,
                stats.current_threshold
            );
        }
    }

    // Demonstrate volatility adjustment
    println!("\nVolatility-adjusted thresholds:");
    calibrator.reset();
    for _ in 0..30 {
        calibrator.update(0.7, 100.0);
    }

    calibrator.update_volatility(0.02); // Normal volatility
    println!(
        "  Normal volatility: {:.3}",
        calibrator.volatility_adjusted_threshold()
    );

    for _ in 0..10 {
        calibrator.update_volatility(0.05); // High volatility
    }
    println!(
        "  High volatility: {:.3}",
        calibrator.volatility_adjusted_threshold()
    );

    println!();
}

fn confidence_calibration_example() {
    println!("--- Example 3: Confidence Calibration ---");

    let config = CalibrationConfig {
        num_bins: 10,
        min_samples_per_bin: 10,
        lookback_window: 1000,
        regularization: 0.01,
    };

    let mut calibrator = vision::adaptive::CombinedCalibrator::new(config);

    println!("\nAdding calibration samples...");
    println!("(Simulating: high confidence → high success rate)");

    // Simulate calibration data where higher confidence correlates with success
    for i in 0..100 {
        let confidence = i as f64 / 100.0;
        let outcome = rand::random::<f64>() < confidence; // Success rate matches confidence
        calibrator.add_sample(confidence, outcome);
    }

    if calibrator.is_fitted() {
        println!("\nCalibration results:");
        println!("  Predicted | Calibrated");
        println!("  ----------|----------");
        for pred in [0.3, 0.5, 0.7, 0.9] {
            let calibrated = calibrator.calibrate(pred);
            println!("    {:.2}    |   {:.3}", pred, calibrated);
        }

        let metrics = calibrator.get_metrics();
        println!("\nCalibration metrics:");
        println!("  Brier Score: {:.4}", metrics.brier_score);
        println!("  Log Loss: {:.4}", metrics.log_loss);
        println!("  Samples: {}", metrics.sample_count);
    }

    println!();
}

fn complete_adaptive_system_example() {
    println!("--- Example 4: Complete Adaptive System ---");

    let mut system = AdaptiveSystem::new();

    println!("\nSimulating market with adaptive decision-making...\n");

    // Simulate 100 trading periods
    for day in 0..100 {
        // Update market data (simulate trending market)
        let price = 100.0 + day as f64 * 0.5 + (day as f64 * 0.1).sin() * 2.0;
        let volatility = 0.02 + ((day as f64 * 0.05).sin().abs() * 0.03);

        let regime = system.update_market(price, volatility);

        // Simulate model prediction
        let raw_confidence = 0.6 + (day as f64 * 0.003);

        // Process through adaptive system
        let (calibrated_confidence, threshold, should_trade) =
            system.process_prediction(raw_confidence);

        // Simulate trade outcome
        if should_trade {
            let profit = if calibrated_confidence > 0.7 {
                100.0 // Good trades
            } else {
                -50.0 // Marginal trades
            };

            system.update_trade(raw_confidence, profit, volatility);

            if day % 20 == 0 {
                println!("Day {}: Trading Decision", day);
                println!("  Regime: {:?}", regime);
                println!("  Raw Confidence: {:.3}", raw_confidence);
                println!("  Calibrated: {:.3}", calibrated_confidence);
                println!("  Threshold: {:.3}", threshold);
                println!("  Trade: YES (Profit: {:.0})", profit);
            }
        } else if day % 20 == 0 {
            println!("Day {}: No Trade", day);
            println!("  Regime: {:?}", regime);
            println!(
                "  Calibrated: {:.3} < Threshold: {:.3}",
                calibrated_confidence, threshold
            );
        }
    }

    // Final statistics
    println!("\n--- Final System Statistics ---");
    let stats = system.get_stats();
    println!("Regime: {:?}", stats.regime);
    println!("\nThreshold Stats:");
    println!("  Current: {:.3}", stats.threshold_stats.current_threshold);
    println!("  Base: {:.3}", stats.threshold_stats.base_threshold);
    println!("  Win Rate: {:.2}%", stats.threshold_stats.win_rate * 100.0);
    println!("  Avg Profit: {:.2}", stats.threshold_stats.avg_profit);
    println!("  Trades: {}", stats.threshold_stats.sample_count);

    println!("\nCalibration Metrics:");
    println!(
        "  Brier Score: {:.4}",
        stats.calibration_metrics.brier_score
    );
    println!("  Samples: {}", stats.calibration_metrics.sample_count);

    println!();
}

fn realistic_trading_scenario() {
    println!("--- Example 5: Realistic Trading Scenario ---");

    // Create system with custom configurations
    let regime_config = RegimeConfig {
        trend_window: 20,
        volatility_window: 20,
        adx_threshold: 25.0,
        volatility_threshold: 0.7,
        min_data_points: 20,
    };

    let threshold_config = ThresholdConfig {
        base_confidence: 0.65,
        target_win_rate: 0.55,
        learning_rate: 0.02,
        ..Default::default()
    };

    let calibration_config = CalibrationConfig {
        num_bins: 10,
        min_samples_per_bin: 10,
        ..Default::default()
    };

    let mut system =
        AdaptiveSystem::with_configs(regime_config, threshold_config, calibration_config);

    let regime_adjuster = RegimeAdjuster::default();

    println!("\nSimulating realistic multi-regime market conditions...\n");

    let mut total_profit = 0.0;
    let mut trades_taken = 0;
    let mut trades_by_regime: std::collections::HashMap<String, (i32, f64)> =
        std::collections::HashMap::new();

    // Simulate different market phases
    for phase in 0..3 {
        println!("=== Phase {} ===", phase + 1);

        let (phase_name, price_fn): (&str, Box<dyn Fn(usize) -> f64>) = match phase {
            0 => ("Bull Trend", Box::new(|i| 100.0 + i as f64 * 1.5)),
            1 => (
                "Ranging/Volatile",
                Box::new(|i| 100.0 + (i as f64 * 0.3).sin() * 8.0),
            ),
            _ => ("Bear Trend", Box::new(|i| 200.0 - i as f64 * 1.2)),
        };

        println!("Market condition: {}\n", phase_name);

        for i in 0..50 {
            let price = price_fn(i);
            let volatility = 0.02 + ((i as f64 * 0.1).sin().abs() * 0.03);

            let regime = system.update_market(price, volatility);

            // Simulate model predictions (better in trends)
            let base_confidence = if regime.is_trending() {
                0.7 + (i as f64 * 0.002)
            } else {
                0.6 + (i as f64 * 0.001)
            };

            let (calibrated_confidence, threshold, should_trade) =
                system.process_prediction(base_confidence);

            if should_trade {
                // Base position size
                let base_size = 1000.0;
                let adjusted_size = regime_adjuster.adjust_position_size(base_size, regime);

                // Simulate profit (trends are more profitable)
                let profit_per_unit = if regime.is_trending() {
                    if calibrated_confidence > 0.75 {
                        0.15
                    } else {
                        0.05
                    }
                } else {
                    if calibrated_confidence > 0.75 {
                        0.08
                    } else {
                        -0.03
                    }
                };

                let profit = adjusted_size * profit_per_unit;
                total_profit += profit;
                trades_taken += 1;

                // Track by regime
                let regime_key = format!("{:?}", regime);
                let entry = trades_by_regime.entry(regime_key).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += profit;

                system.update_trade(base_confidence, profit, volatility);

                if i % 20 == 0 {
                    println!(
                        "  Trade #{}: Regime={:?}, Confidence={:.3}, Size={:.0}, P/L={:.2}",
                        trades_taken, regime, calibrated_confidence, adjusted_size, profit
                    );
                }
            }
        }

        println!();
    }

    // Final report
    println!("=== Trading Performance Summary ===");
    println!("Total Trades: {}", trades_taken);
    println!("Total Profit: ${:.2}", total_profit);
    println!(
        "Avg Profit per Trade: ${:.2}",
        total_profit / trades_taken as f64
    );

    println!("\n--- Performance by Regime ---");
    let mut regime_vec: Vec<_> = trades_by_regime.iter().collect();
    regime_vec.sort_by_key(|(k, _)| k.to_string());

    for (regime, (count, profit)) in regime_vec {
        println!(
            "{:20} : {} trades, ${:8.2} total, ${:6.2} avg",
            regime,
            count,
            profit,
            profit / *count as f64
        );
    }

    let final_stats = system.get_stats();
    println!("\n--- Final Adaptive System State ---");
    println!("Current Regime: {:?}", final_stats.regime);
    println!(
        "Threshold: {:.3}",
        final_stats.threshold_stats.current_threshold
    );
    println!(
        "Win Rate: {:.1}%",
        final_stats.threshold_stats.win_rate * 100.0
    );

    println!();
}
