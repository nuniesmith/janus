//! Gradient Clipping Demo
//!
//! This example demonstrates the gradient clipping feature and its impact
//! on training stability. It compares training with and without gradient clipping.
//!
//! Run with:
//! ```bash
//! cargo run --example gradient_clipping_demo
//! ```

use janus_ml::dataset::{MarketDataSample, MarketDataset, WindowConfig, WindowedDataset};
use janus_ml::models::trainable::TrainableLstmConfig;
use janus_ml::training_autodiff::{AutodiffTrainer, AutodiffTrainingConfig, GradientClipping};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n{}", "=".repeat(70));
    println!("    GRADIENT CLIPPING DEMONSTRATION");
    println!("{}\n", "=".repeat(70));

    // Create challenging dataset (prone to gradient explosion)
    println!("📊 Generating challenging synthetic dataset...");
    let (train_dataset, val_dataset) = create_challenging_dataset()?;
    println!("   Training samples: {}", train_dataset.len());
    println!("   Validation samples: {}", val_dataset.len());
    println!();

    // Model configuration
    let model_config = TrainableLstmConfig::new(6, 64, 1)
        .with_num_layers(2)
        .with_dropout(0.1);

    println!("🧠 Model: LSTM (6 → 64 → 1), 2 layers, dropout=0.1");
    println!();

    // Run different scenarios
    let scenarios = vec![
        ("No Clipping", None),
        ("Norm Clip (1.0)", Some(GradientClipping::ByNorm(1.0))),
        ("Norm Clip (0.5)", Some(GradientClipping::ByNorm(0.5))),
        ("Value Clip (5.0)", Some(GradientClipping::ByValue(5.0))),
    ];

    let mut results = Vec::new();

    for (name, clip_strategy) in scenarios {
        println!("{}", "─".repeat(70));
        println!("Scenario: {}", name);
        println!("{}", "─".repeat(70));

        // Configure training
        let mut config = AutodiffTrainingConfig::default()
            .epochs(10)
            .batch_size(16)
            .learning_rate(0.01) // Higher LR to stress test
            .warmup_epochs(2)
            .early_stopping_patience(5)
            .seed(42);

        if let Some(clip) = clip_strategy {
            config = config.grad_clip(clip);
        }

        println!("  Epochs: {}", config.epochs);
        println!("  Batch size: {}", config.batch_size);
        println!("  Learning rate: {}", config.learning_rate);
        println!(
            "  Gradient clipping: {}",
            if let Some(clip) = &config.grad_clip {
                clip.description()
            } else {
                "Disabled".to_string()
            }
        );
        println!();

        // Train
        let start = Instant::now();
        let mut trainer = AutodiffTrainer::new(model_config.clone(), config.clone())?;
        let history = trainer.fit(train_dataset.clone(), Some(val_dataset.clone()))?;
        let duration = start.elapsed();

        // Print summary
        print_training_summary(&history, duration);
        println!();

        results.push((name, history, duration));
    }

    // Comparison table
    println!("{}", "=".repeat(70));
    println!("COMPARISON SUMMARY");
    println!("{}", "=".repeat(70));
    println!();
    println!(
        "{:<20} | {:>12} | {:>12} | {:>10} | {:>8}",
        "Strategy", "Final Loss", "Best Loss", "Time (s)", "Epochs"
    );
    println!("{}", "─".repeat(70));

    for (name, history, duration) in &results {
        let final_loss = history
            .val_loss
            .last()
            .map(|v| format!("{:.6}", v))
            .unwrap_or_else(|| "N/A".to_string());

        let best_loss = format!("{:.6}", history.best_val_loss);
        let time = format!("{:.2}", duration.as_secs_f64());
        let epochs = history.num_epochs();

        println!(
            "{:<20} | {:>12} | {:>12} | {:>10} | {:>8}",
            name, final_loss, best_loss, time, epochs
        );
    }

    println!();
    println!("Key Observations:");
    println!("  • Gradient clipping helps stabilize training with high learning rates");
    println!("  • Norm-based clipping preserves gradient direction better than value clipping");
    println!("  • Conservative clipping (0.5) may slow convergence but increases stability");
    println!("  • Choose threshold based on loss behavior and convergence requirements");
    println!();
    println!("✅ Gradient clipping demonstration completed!");
    println!();

    Ok(())
}

fn create_challenging_dataset()
-> Result<(WindowedDataset, WindowedDataset), Box<dyn std::error::Error>> {
    // Create synthetic data with large values and high variance
    // This makes training challenging and prone to gradient issues
    let num_samples = 500;
    let mut samples = Vec::new();

    for i in 0..num_samples {
        let t = i as f64 * 0.05;

        // Base price with trend and volatility
        let trend = t * 0.5;
        let seasonal = 10.0 * (t * 0.3).sin();
        let noise = ((i * 17) % 100) as f64 * 0.5 - 25.0;

        // Occasional large spikes to stress gradients
        let spike = if i % 30 == 0 { 50.0 } else { 0.0 };

        let close_price = 100.0 + trend + seasonal + noise + spike;
        let open_price = close_price - ((i % 7) as f64 - 3.5) * 2.0;
        let high_price = close_price.max(open_price) + ((i % 5) as f64) * 1.5;
        let low_price = close_price.min(open_price) - ((i % 5) as f64) * 1.5;

        let volume = 5000.0 + ((i * 23) % 3000) as f64;
        let mid_price = (high_price + low_price) / 2.0;

        // Features: [open, high, low, close, volume, mid_price]
        let features = vec![
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            mid_price,
        ];

        // Target: next close price
        let target = close_price + ((i % 13) as f64 - 6.5);

        samples.push(MarketDataSample {
            timestamp: i as i64,
            features,
            target,
            metadata: Default::default(),
        });
    }

    // Create dataset
    let mut dataset = MarketDataset::new();
    dataset.samples = samples;
    dataset.metadata.num_samples = num_samples;
    dataset.metadata.num_features = 6;

    // Split 80/20
    let split_idx = (num_samples as f32 * 0.8) as usize;
    let train_samples = dataset.samples[..split_idx].to_vec();
    let val_samples = dataset.samples[split_idx..].to_vec();

    let mut train_ds = MarketDataset::new();
    train_ds.samples = train_samples;
    train_ds.metadata.num_samples = train_ds.samples.len();
    train_ds.metadata.num_features = 6;

    let mut val_ds = MarketDataset::new();
    val_ds.samples = val_samples;
    val_ds.metadata.num_samples = val_ds.samples.len();
    val_ds.metadata.num_features = 6;

    // Create windowed datasets
    let window_config = WindowConfig {
        window_size: 10,
        horizon: 1,
        stride: 1,
        min_samples: 10,
    };

    let train_windowed = WindowedDataset::from_dataset(train_ds, window_config.clone())?;
    let val_windowed = WindowedDataset::from_dataset(val_ds, window_config)?;

    Ok((train_windowed, val_windowed))
}

fn print_training_summary(
    history: &janus_ml::training_autodiff::AutodiffTrainingHistory,
    duration: std::time::Duration,
) {
    if !history.train_loss.is_empty() {
        let improvement = if history.train_loss[0] > 0.0 {
            ((history.train_loss[0] - history.train_loss.last().unwrap()) / history.train_loss[0])
                * 100.0
        } else {
            0.0
        };

        println!("  Training:");
        println!("    Initial loss: {:.6}", history.train_loss[0]);
        println!(
            "    Final loss:   {:.6} ({:.1}% improvement)",
            history.train_loss.last().unwrap(),
            improvement
        );
    }

    if !history.val_loss.is_empty() {
        println!("  Validation:");
        println!("    Initial loss: {:.6}", history.val_loss[0]);
        println!("    Final loss:   {:.6}", history.val_loss.last().unwrap());
        println!("    Best loss:    {:.6}", history.best_val_loss);
        println!("    Best epoch:   {}", history.best_epoch);
    }

    println!("  Time: {:.2}s", duration.as_secs_f64());
}
