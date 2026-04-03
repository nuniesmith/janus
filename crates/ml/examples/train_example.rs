//! Example: Training an LSTM model on synthetic market data
//!
//! This example demonstrates the complete Phase 3 training pipeline:
//! 1. Generate synthetic market data samples
//! 2. Create a windowed dataset for time series modeling
//! 3. Split into train/validation/test sets
//! 4. Configure and run training with early stopping
//! 5. Evaluate the model and generate metrics report
//!
//! Run with:
//! ```bash
//! cargo run --example train_example
//! ```

use janus_ml::{
    backend::cpu_backend,
    dataset::{MarketDataSample, MarketDataset, SampleMetadata, WindowConfig},
    evaluation::MetricsCalculator,
    models::LstmConfig,
    optimizer::OptimizerType,
    training::{Trainer, TrainingConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    println!("═══════════════════════════════════════════════════════");
    println!("  JANUS ML Training Pipeline - Phase 3 Example");
    println!("═══════════════════════════════════════════════════════\n");

    // 1. Generate synthetic market data
    println!("📊 Generating synthetic market data...");
    let num_samples = 1000;
    let samples = generate_synthetic_data(num_samples);
    println!("   ✓ Generated {} samples\n", num_samples);

    // 2. Create dataset
    println!("🗂️  Creating dataset...");
    let dataset = MarketDataset::from_samples(samples)?;
    println!("   ✓ Dataset size: {}", dataset.len());
    println!("   ✓ Feature dimension: {}\n", dataset.feature_dim());

    // 3. Create windowed dataset for time series
    println!("🪟 Creating sliding windows...");
    let window_config = WindowConfig::new(50, 1) // 50 timesteps lookback, predict 1 ahead
        .with_stride(1);
    let windowed = dataset.into_windowed(window_config)?;
    println!("   ✓ Created {} windows\n", windowed.len());

    // 4. Split into train/validation/test
    println!("✂️  Splitting dataset...");
    let (train, val, test) = windowed.split(0.7, 0.15, 0.15)?;
    println!("   ✓ Train set: {} windows", train.len());
    println!("   ✓ Validation set: {} windows", val.len());
    println!("   ✓ Test set: {} windows\n", test.len());

    // 5. Configure training
    println!("⚙️  Configuring training...");
    let train_config = TrainingConfig::new()
        .epochs(20)
        .batch_size(16)
        .learning_rate(0.001)
        .weight_decay(0.0001)
        .optimizer(OptimizerType::AdamW)
        .warmup_steps(50)
        .cosine_schedule(true)
        .early_stopping_patience(5)
        .device(cpu_backend().device().clone())
        .seed(42);

    println!("   ✓ Epochs: {}", train_config.epochs);
    println!("   ✓ Batch size: {}", train_config.batch_size);
    println!("   ✓ Learning rate: {}", train_config.learning_rate);
    println!("   ✓ Optimizer: {:?}", train_config.optimizer_type);
    println!("   ✓ LR Schedule: warmup + cosine");
    println!(
        "   ✓ Early stopping patience: {:?}\n",
        train_config.early_stopping_patience
    );

    // 6. Configure model
    println!("🧠 Configuring LSTM model...");
    let model_config = LstmConfig::new(5, 32, 1) // 5 features, 32 hidden units, 1 output
        .with_num_layers(2)
        .with_dropout(0.2);

    println!("   ✓ Input size: {}", model_config.input_size);
    println!("   ✓ Hidden size: {}", model_config.hidden_size);
    println!("   ✓ Num layers: {}", model_config.num_layers);
    println!("   ✓ Dropout: {}\n", model_config.dropout);

    // 7. Create trainer
    println!("🏋️  Creating trainer...");
    let mut trainer = Trainer::new(model_config, train_config)?;
    println!("   ✓ Trainer initialized\n");

    // 8. Train the model
    println!("🚀 Starting training...");
    println!("─────────────────────────────────────────────────────\n");

    let history = trainer.fit(train, Some(val))?;

    println!("\n─────────────────────────────────────────────────────");
    println!("✅ Training complete!\n");

    // 9. Print training summary
    history.print_summary();

    // Print optimizer stats
    let opt_stats = trainer.optimizer().stats();
    println!("\nOptimizer Statistics:");
    println!("  Total steps: {}", opt_stats.step);
    println!("  Final LR: {:.6}", opt_stats.learning_rate);

    // 10. Evaluate on test set
    println!("\n📈 Evaluating on test set...");
    let report = trainer.evaluate(test)?;
    println!();
    report.print();

    // 11. Demonstrate metrics calculator
    println!("\n💡 Additional Metrics Examples:");
    println!("─────────────────────────────────────────────────────\n");

    let calc = MetricsCalculator::new();

    // Example: Perfect predictions
    let perfect_preds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let perfect_targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let perfect_metrics = calc.regression_metrics(&perfect_preds, &perfect_targets)?;
    println!("Perfect Predictions:");
    println!("  RMSE: {:.6} (should be 0.0)", perfect_metrics.rmse);
    println!("  R²: {:.6} (should be 1.0)\n", perfect_metrics.r_squared);

    // Example: Trading metrics
    let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.02, 0.03];
    let trading_metrics = calc.trading_metrics(&returns, 0.02)?;
    println!("Trading Metrics:");
    println!("  Sharpe Ratio: {:.4}", trading_metrics.sharpe_ratio);
    println!("  Win Rate: {:.2}%", trading_metrics.win_rate);
    println!("  Max Drawdown: {:.2}%\n", trading_metrics.max_drawdown);

    println!("═══════════════════════════════════════════════════════");
    println!("  Example Complete! ✨");
    println!("═══════════════════════════════════════════════════════");

    Ok(())
}

/// Generate synthetic market data for demonstration
///
/// Creates a simple trend + noise pattern with 5 features:
/// - Price (with upward trend)
/// - Volume (random walk)
/// - RSI (bounded oscillator)
/// - MACD (trend indicator)
/// - Bollinger Band width
fn generate_synthetic_data(n: usize) -> Vec<MarketDataSample> {
    use std::f64::consts::PI;

    (0..n)
        .map(|i| {
            let t = i as f64;

            // Base price with trend + seasonality + noise
            let trend = 50000.0 + t * 10.0;
            let seasonality = 1000.0 * (2.0 * PI * t / 100.0).sin();
            let noise = (t * 0.1).sin() * 100.0;
            let price = trend + seasonality + noise;

            // Volume (random walk bounded)
            let volume = 100.0 + 50.0 * ((t * 0.05).sin() + 1.0);

            // RSI (0-100 oscillator)
            let rsi = 50.0 + 30.0 * (t * 0.1).sin();

            // MACD (centered around 0)
            let macd = 10.0 * (t * 0.08).sin();

            // Bollinger Band Width
            let bb_width = 200.0 + 100.0 * (t * 0.06).cos();

            // Features vector
            let features = vec![
                price / 50000.0,      // Normalize price
                volume / 100.0,       // Normalize volume
                rsi / 100.0,          // RSI already 0-100
                (macd + 20.0) / 40.0, // Normalize MACD
                bb_width / 300.0,     // Normalize BB width
            ];

            // Target: next price (normalized)
            let next_price = 50000.0
                + (t + 1.0) * 10.0
                + 1000.0 * (2.0 * PI * (t + 1.0) / 100.0).sin()
                + ((t + 1.0) * 0.1).sin() * 100.0;
            let target = next_price / 50000.0;

            MarketDataSample {
                timestamp: (i as i64) * 60_000_000, // 1-minute intervals
                features,
                target,
                metadata: SampleMetadata {
                    symbol: "BTCUSD".to_string(),
                    exchange: "binance".to_string(),
                    price,
                    volume,
                },
            }
        })
        .collect()
}
