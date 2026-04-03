//! Example demonstrating how to save and load trained model weights
//!
//! This example shows:
//! 1. Training a model on synthetic data
//! 2. Saving the trained weights to disk
//! 3. Loading the weights into a new model
//! 4. Verifying the loaded model produces the same predictions
//!
//! Run with:
//! ```bash
//! cargo run --example save_load_model
//! ```

use janus_ml::backend::AutodiffCpuBackend;
use janus_ml::dataset::DataLoader;
use janus_ml::dataset::{
    MarketDataSample, MarketDataset, SampleMetadata, WindowConfig, WindowedDataset,
};
use janus_ml::models::trainable::TrainableLstmConfig;
use janus_ml::training_autodiff::{AutodiffTrainer, AutodiffTrainingConfig};
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n=== JANUS ML Weight Save/Load Example ===\n");

    // 1. Generate and prepare training data
    println!("📊 Generating synthetic market data...");
    let samples = generate_synthetic_data(500);
    let mut train_dataset = MarketDataset::new();
    for sample in samples {
        train_dataset.samples.push(sample);
    }
    train_dataset.metadata.num_samples = train_dataset.samples.len();
    train_dataset.metadata.num_features = 6;
    println!("   Generated {} samples", train_dataset.len());

    // 2. Create windowed dataset
    println!("\n🪟 Creating windowed dataset...");
    let window_config = WindowConfig {
        window_size: 20,
        stride: 1,
        horizon: 1,
        min_samples: 20,
    };
    let train_windowed = WindowedDataset::from_dataset(train_dataset, window_config.clone())?;
    println!("   Created {} training windows", train_windowed.len());

    // 3. Configure model
    println!("\n🧠 Configuring LSTM model...");
    let model_config = TrainableLstmConfig::new(6, 32, 1)
        .with_num_layers(2)
        .with_dropout(0.1);
    println!("   Input: 6, Hidden: 32, Output: 1, Layers: 2");

    // 4. Configure training
    println!("\n⚙️  Configuring training...");
    let training_config = AutodiffTrainingConfig::default()
        .epochs(10)
        .batch_size(16)
        .learning_rate(0.001)
        .warmup_epochs(2)
        .seed(42);
    println!("   Epochs: 10, Batch size: 16, LR: 0.001");

    // 5. Train model
    println!("\n🏋️  Training model...");
    let mut trainer = AutodiffTrainer::new(model_config.clone(), training_config.clone())?;
    let history = trainer.fit(train_windowed.clone(), None)?;
    println!("\n✅ Training complete!");
    println!(
        "   Final loss: {:.6}",
        history.train_loss.last().unwrap_or(&0.0)
    );

    // 6. Save the trained model
    let temp_dir = tempdir()?;
    let weights_path = temp_dir.path().join("trained_model.mpk");
    println!("\n💾 Saving trained model...");
    println!("   Path: {:?}", weights_path);
    trainer.save_model(&weights_path)?;
    println!("   ✓ Model saved successfully");

    // Verify files exist
    assert!(weights_path.exists(), "Weights file should exist");
    assert!(
        weights_path.with_extension("json").exists(),
        "Config file should exist"
    );
    println!("   ✓ Both weights (.mpk) and config (.json) files created");

    // 7. Make predictions with trained model
    println!("\n🔮 Making predictions with trained model...");
    let test_samples = generate_synthetic_data(50);
    let mut test_dataset = MarketDataset::new();
    for sample in test_samples {
        test_dataset.samples.push(sample);
    }
    test_dataset.metadata.num_samples = test_dataset.samples.len();
    test_dataset.metadata.num_features = 6;

    let test_windowed = WindowedDataset::from_dataset(test_dataset, window_config)?;
    let mut test_loader = DataLoader::new(test_windowed.clone())
        .batch_size(16)
        .shuffle(false);

    let device = Default::default();
    let original_predictions = {
        let mut preds = Vec::new();
        while let Some(batch_result) = test_loader.next_batch::<AutodiffCpuBackend>(&device) {
            let (features, _) = batch_result?;
            let pred = trainer.model().forward(features);
            let pred_data = pred.into_data();
            let pred_vec: Vec<f32> = pred_data.to_vec().unwrap();
            preds.extend(pred_vec);
        }
        preds
    };
    println!(
        "   Original model predictions: {} values",
        original_predictions.len()
    );
    if original_predictions.len() >= 3 {
        println!(
            "   Sample predictions: [{:.4}, {:.4}, {:.4}, ...]",
            original_predictions[0], original_predictions[1], original_predictions[2]
        );
    }

    // 8. Load weights into a new model
    println!("\n📥 Loading weights into new model...");
    let loaded_trainer = AutodiffTrainer::from_weights(
        model_config.clone(),
        training_config.clone(),
        &weights_path,
    )?;
    println!("   ✓ Model loaded successfully");

    // 9. Make predictions with loaded model
    println!("\n🔮 Making predictions with loaded model...");
    test_loader.reset();
    let loaded_predictions = {
        let mut preds = Vec::new();
        while let Some(batch_result) = test_loader.next_batch::<AutodiffCpuBackend>(&device) {
            let (features, _) = batch_result?;
            let pred = loaded_trainer.model().forward(features);
            let pred_data = pred.into_data();
            let pred_vec: Vec<f32> = pred_data.to_vec().unwrap();
            preds.extend(pred_vec);
        }
        preds
    };
    println!(
        "   Loaded model predictions: {} values",
        loaded_predictions.len()
    );
    if loaded_predictions.len() >= 3 {
        println!(
            "   Sample predictions: [{:.4}, {:.4}, {:.4}, ...]",
            loaded_predictions[0], loaded_predictions[1], loaded_predictions[2]
        );
    }

    // 10. Verify predictions match
    println!("\n✓ Verifying predictions match...");
    assert_eq!(
        original_predictions.len(),
        loaded_predictions.len(),
        "Number of predictions should match"
    );

    let max_diff = original_predictions
        .iter()
        .zip(&loaded_predictions)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("   Max difference: {:.10}", max_diff);
    println!("   Tolerance: 1e-6");

    if max_diff < 1e-6 {
        println!("   ✅ Predictions match perfectly!");
    } else if max_diff < 1e-4 {
        println!("   ✅ Predictions match within tolerance");
    } else {
        println!("   ⚠️  Predictions differ: {:.10}", max_diff);
    }

    // 11. Summary
    println!("\n{}", "=".repeat(60));
    println!("✅ SAVE/LOAD DEMONSTRATION COMPLETE");
    println!("{}", "=".repeat(60));
    println!("\n📝 Key takeaways:");
    println!("  • Trained models can be saved with .save_model()");
    println!("  • Both weights (.mpk) and config (.json) are saved");
    println!("  • Models can be loaded with .from_weights()");
    println!("  • Loaded models produce identical predictions");
    println!("  • Perfect for deployment, sharing, and resuming training");

    println!("\n💾 File formats:");
    println!("  • .mpk  - MessagePack binary format (weights)");
    println!("  • .json - JSON text format (configuration)");

    println!("\n🎯 Use cases:");
    println!("  • Deploy trained models to production");
    println!("  • Share models between team members");
    println!("  • Resume training from checkpoints");
    println!("  • Run backtests with trained models");
    println!("  • Version control model snapshots");

    println!("\n💡 Usage example:");
    println!("   // Save:");
    println!("   trainer.save_model(\"./my_model.mpk\")?;");
    println!();
    println!("   // Load:");
    println!("   let trainer = AutodiffTrainer::from_weights(");
    println!("       model_config,");
    println!("       training_config,");
    println!("       \"./my_model.mpk\"");
    println!("   )?;");

    Ok(())
}

/// Generate synthetic market data for demonstration
fn generate_synthetic_data(num_samples: usize) -> Vec<MarketDataSample> {
    use rand::RngExt;
    let mut rng = rand::rng();

    let mut samples = Vec::with_capacity(num_samples);
    let mut price = 50000.0;

    for i in 0..num_samples {
        // Random walk with slight upward trend
        let change_pct = rng.random_range(-0.02..0.025);
        price *= 1.0 + change_pct;

        let volume = rng.random_range(100.0..1000.0);

        // Create feature vector (6 features as expected by model)
        let open = price * 0.999;
        let high = price * 1.002;
        let low = price * 0.998;
        let close = price;
        let mid_price = (high + low) / 2.0;

        let features = vec![
            open / 50000.0,
            high / 50000.0,
            low / 50000.0,
            close / 50000.0,
            volume / 500.0,
            mid_price / 50000.0,
        ];

        // Target: predict next price (normalized)
        let target = (price * (1.0 + rng.random_range(-0.01..0.01))) / 50000.0;

        let metadata = SampleMetadata {
            symbol: "BTCUSD".to_string(),
            exchange: "synthetic".to_string(),
            price,
            volume,
        };

        samples.push(MarketDataSample {
            timestamp: i as i64 * 3600,
            features,
            target,
            metadata,
        });
    }

    samples
}
