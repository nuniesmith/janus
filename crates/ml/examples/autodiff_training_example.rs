//! Example demonstrating autodiff-enabled training with gradient updates
//!
//! This example shows how to:
//! 1. Create synthetic market data
//! 2. Configure a trainable LSTM model with autodiff backend
//! 3. Train the model with actual gradient descent
//! 4. Monitor training metrics and learning rate scheduling
//! 5. Observe weight updates and loss reduction
//!
//! Run with:
//! ```bash
//! cargo run --example autodiff_training_example
//! ```

use janus_ml::dataset::{MarketDataSample, MarketDataset, WindowConfig, WindowedDataset};
use janus_ml::models::trainable::TrainableLstmConfig;
use janus_ml::training_autodiff::{AutodiffTrainer, AutodiffTrainingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n=== JANUS ML Autodiff Training Example ===\n");

    // Step 1: Generate synthetic market data
    println!("📊 Generating synthetic market data...");
    let samples = generate_synthetic_data(1000);
    println!("   Generated {} samples", samples.len());

    // Step 2: Create dataset and split
    let mut dataset = MarketDataset::new();
    for sample in samples {
        dataset.samples.push(sample);
    }
    dataset.metadata.num_samples = dataset.samples.len();
    dataset.metadata.num_features = 6;
    let (train_data, val_data, _test_data) = dataset.split(0.7, 0.15, 0.15)?;

    println!(
        "   Train: {} samples, Val: {} samples",
        train_data.len(),
        val_data.len()
    );

    // Step 3: Create windowed datasets
    println!("\n🪟 Creating windowed datasets...");
    let window_config = WindowConfig {
        window_size: 20,
        stride: 1,
        horizon: 1,
        min_samples: 20,
    };

    let train_windowed = WindowedDataset::from_dataset(train_data, window_config.clone())?;
    let val_windowed = WindowedDataset::from_dataset(val_data, window_config)?;

    println!(
        "   Train windows: {}, Val windows: {}",
        train_windowed.len(),
        val_windowed.len()
    );

    // Step 4: Configure model
    println!("\n🧠 Configuring LSTM model...");
    let model_config = TrainableLstmConfig::new(
        6,  // input_size: 6 features (open, high, low, close, volume, mid_price)
        32, // hidden_size
        1,  // output_size: predict next price
    )
    .with_num_layers(2)
    .with_dropout(0.1);

    println!("   Input size: {}", model_config.input_size);
    println!("   Hidden size: {}", model_config.hidden_size);
    println!("   Num layers: {}", model_config.num_layers);
    println!("   Dropout: {}", model_config.dropout);

    // Step 5: Configure training with autodiff
    println!("\n⚙️  Configuring training...");
    let training_config = AutodiffTrainingConfig::default()
        .epochs(50)
        .batch_size(16)
        .learning_rate(0.001)
        .warmup_epochs(5)
        .cosine_schedule(true)
        .early_stopping_patience(10)
        .seed(42);

    println!("   Epochs: {}", training_config.epochs);
    println!("   Batch size: {}", training_config.batch_size);
    println!("   Learning rate: {}", training_config.learning_rate);
    println!("   Warmup epochs: {:?}", training_config.warmup_epochs);
    println!(
        "   Cosine schedule: {}",
        training_config.use_cosine_schedule
    );
    println!(
        "   Early stopping patience: {:?}",
        training_config.early_stopping_patience
    );

    // Step 6: Create trainer
    println!("\n🚀 Creating autodiff trainer...");
    let mut trainer = AutodiffTrainer::new(model_config, training_config)?;
    println!("   Trainer initialized with gradient tracking enabled");

    // Step 7: Train the model
    println!("\n🏋️  Training model with gradient descent...");
    println!("   (Watch for decreasing loss - proof that weights are updating!)\n");

    let history = trainer.fit(train_windowed, Some(val_windowed))?;

    // Step 8: Analyze results
    println!("\n📈 Training Results:");
    println!("   ─────────────────────────────────────");
    println!("   Total epochs: {}", history.num_epochs());
    println!("   Best epoch: {}", history.best_epoch);
    println!("   Best validation loss: {:.6}", history.best_val_loss);

    if let (Some(&first_train), Some(&last_train)) =
        (history.train_loss.first(), history.train_loss.last())
    {
        let improvement = ((first_train - last_train) / first_train) * 100.0;
        println!("   Initial train loss: {:.6}", first_train);
        println!("   Final train loss: {:.6}", last_train);
        println!("   Loss reduction: {:.2}%", improvement);
    }

    if let (Some(&first_val), Some(&last_val)) = (history.val_loss.first(), history.val_loss.last())
    {
        let improvement = ((first_val - last_val) / first_val) * 100.0;
        println!("   Initial val loss: {:.6}", first_val);
        println!("   Final val loss: {:.6}", last_val);
        println!("   Val improvement: {:.2}%", improvement);
    }

    // Step 9: Show learning rate schedule
    println!("\n📊 Learning Rate Schedule:");
    println!("   ─────────────────────────────────────");
    for (i, &lr) in history.learning_rates.iter().enumerate() {
        if i % 10 == 0 || i == history.learning_rates.len() - 1 {
            println!("   Epoch {:3}: LR = {:.6}", i, lr);
        }
    }

    // Step 10: Save training history
    println!("\n💾 Saving training history...");
    let history_path = "/tmp/autodiff_training_history.json";
    history.save_json(history_path)?;
    println!("   Saved to: {}", history_path);

    println!("\n✅ Autodiff training example completed successfully!");
    println!("\n🎉 Key Achievement: Model weights were updated using backpropagation!");
    println!("   The decreasing loss proves that gradients were computed and applied.\n");

    Ok(())
}

/// Generate synthetic market data for demonstration
fn generate_synthetic_data(num_samples: usize) -> Vec<MarketDataSample> {
    let mut samples = Vec::new();
    let base_price = 100.0;

    for i in 0..num_samples {
        let t = i as f64 * 0.1;

        // Generate price with trend and noise
        let trend = t * 0.01;
        let seasonal = 5.0 * (t * 0.5).sin();
        let noise = ((i * 7) % 100) as f64 * 0.1 - 5.0;

        let close_price = base_price + trend + seasonal + noise;
        let open_price = close_price - ((i % 5) as f64 - 2.5) * 0.5;
        let high_price = close_price.max(open_price) + ((i % 3) as f64) * 0.3;
        let low_price = close_price.min(open_price) - ((i % 3) as f64) * 0.3;

        let volume = 1000.0 + ((i * 13) % 500) as f64;
        let mid_price = (high_price + low_price) / 2.0;

        // Create feature vector (6 features)
        let features = vec![
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            mid_price,
        ];

        // Target is next close price (for simplicity, use current + small change)
        let target = close_price + ((i % 10) as f64 - 5.0) * 0.1;

        samples.push(MarketDataSample {
            timestamp: i as i64,
            features,
            target,
            metadata: Default::default(),
        });
    }

    samples
}
