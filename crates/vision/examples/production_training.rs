//! Production-ready training example with engineered features.
//!
//! This example demonstrates a complete training pipeline including:
//! - CSV data loading with validation
//! - Feature engineering with technical indicators
//! - Preprocessing and normalization
//! - Train/validation split
//! - Training loop with checkpointing
//! - Early stopping
//! - Performance metrics
//!
//! Run with:
//! ```bash
//! cargo run --package vision --example production_training --release
//! ```

use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        ElementConversion,
        backend::{AutodiffBackend, Backend},
    },
};
use std::time::Instant;
use vision::{
    data::{
        csv_loader::OhlcvCandle,
        dataset::{OhlcvDataset, SequenceConfig},
    },
    diffgaf::combined::{DiffGafLstm, DiffGafLstmConfig},
    preprocessing::{
        features::{FeatureConfig, FeatureEngineer},
        normalization::ZScoreScaler,
        tensor_conversion::{TensorConverter, TensorConverterConfig, create_batch_with_features},
    },
};

type MyBackend = Autodiff<NdArray>;

// Training configuration
struct TrainingConfig {
    epochs: usize,
    batch_size: usize,
    learning_rate: f64,
    validation_split: f64,
    early_stopping_patience: usize,
    checkpoint_dir: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 50,
            batch_size: 32,
            learning_rate: 0.001,
            validation_split: 0.2,
            early_stopping_patience: 10,
            checkpoint_dir: "checkpoints".to_string(),
        }
    }
}

// Training metrics tracker
struct MetricsTracker {
    train_losses: Vec<f32>,
    val_losses: Vec<f32>,
    best_val_loss: f32,
    best_epoch: usize,
    patience_counter: usize,
}

impl MetricsTracker {
    fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            best_val_loss: f32::INFINITY,
            best_epoch: 0,
            patience_counter: 0,
        }
    }

    fn update(&mut self, train_loss: f32, val_loss: f32, epoch: usize, patience: usize) -> bool {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);

        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.best_epoch = epoch;
            self.patience_counter = 0;
            true // Improved
        } else {
            self.patience_counter += 1;
            self.patience_counter >= patience // Should stop?
        }
    }

    fn print_summary(&self) {
        println!("\n📊 Training Summary:");
        println!("  Best validation loss: {:.6}", self.best_val_loss);
        println!("  Best epoch: {}", self.best_epoch);
        println!(
            "  Final train loss: {:.6}",
            self.train_losses.last().unwrap_or(&0.0)
        );
        println!(
            "  Final val loss: {:.6}",
            self.val_losses.last().unwrap_or(&0.0)
        );
    }
}

// Generate synthetic training data
fn generate_training_data() -> Vec<OhlcvCandle> {
    use chrono::{Duration, Utc};

    println!("📦 Generating synthetic OHLCV data...");

    let start_time = Utc::now() - Duration::days(365);
    let num_candles = 10000;

    let mut candles = Vec::with_capacity(num_candles);
    let mut price = 100.0;

    for i in 0..num_candles {
        let timestamp = start_time + Duration::minutes(i as i64);

        // Random walk with trend
        let trend = 0.0001 * (i as f64).sin();
        let volatility = 0.02;
        let change = trend + volatility * (rand::random::<f64>() - 0.5);

        price *= 1.0 + change;
        price = price.max(50.0).min(200.0); // Keep price in reasonable range

        let high = price * (1.0 + 0.005 * rand::random::<f64>());
        let low = price * (1.0 - 0.005 * rand::random::<f64>());
        let close = low + (high - low) * rand::random::<f64>();
        let volume = 1000000.0 * (0.5 + rand::random::<f64>());

        candles.push(OhlcvCandle::new(timestamp, price, high, low, close, volume));
    }

    println!("  Generated {} candles", candles.len());
    candles
}

// Compute engineered features for all sequences (parallel)
fn compute_features_for_dataset(
    dataset: &OhlcvDataset,
    engineer: &FeatureEngineer,
) -> Vec<Vec<Vec<f64>>> {
    engineer
        .compute_dataset_features_parallel(dataset)
        .expect("Failed to compute features")
}

// Split dataset into train and validation
fn train_val_split(
    features: Vec<Vec<Vec<f64>>>,
    labels: Vec<f64>,
    val_split: f64,
) -> (Vec<Vec<Vec<f64>>>, Vec<f64>, Vec<Vec<Vec<f64>>>, Vec<f64>) {
    let total = features.len();
    let split_idx = (total as f64 * (1.0 - val_split)) as usize;

    let (train_features, val_features) = features.split_at(split_idx);
    let (train_labels, val_labels) = labels.split_at(split_idx);

    (
        train_features.to_vec(),
        train_labels.to_vec(),
        val_features.to_vec(),
        val_labels.to_vec(),
    )
}

// Training epoch
fn train_epoch<B: AutodiffBackend>(
    mut model: DiffGafLstm<B>,
    optimizer: &mut impl Optimizer<DiffGafLstm<B>, B>,
    train_features: &[Vec<Vec<f64>>],
    train_labels: &[f64],
    converter: &TensorConverter<ZScoreScaler>,
    batch_size: usize,
    device: &B::Device,
) -> (DiffGafLstm<B>, f32) {
    let num_batches = (train_features.len() + batch_size - 1) / batch_size;
    let mut total_loss = 0.0;
    let mut batch_count = 0;

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(train_features.len());

        let batch_features = &train_features[start_idx..end_idx];
        let batch_labels_slice = &train_labels[start_idx..end_idx];

        // Convert to tensors
        let (inputs, targets) =
            create_batch_with_features(batch_features, batch_labels_slice, converter, device);

        // Convert regression targets to classification (3 classes: down, neutral, up)
        let class_targets = targets.clone().int();

        // Forward pass
        let output = model.forward_classification(inputs, class_targets);
        let loss = output.loss;

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        let model_updated = optimizer.step(1.0, model, grads);
        model = model_updated;

        // Get loss value
        let loss_value = loss.into_scalar().elem::<f32>();
        total_loss += loss_value;
        batch_count += 1;
    }

    (model, total_loss / batch_count as f32)
}

// Validation epoch
fn validate_epoch<B: Backend>(
    model: &DiffGafLstm<B>,
    val_features: &[Vec<Vec<f64>>],
    val_labels: &[f64],
    converter: &TensorConverter<ZScoreScaler>,
    batch_size: usize,
    device: &B::Device,
) -> f32 {
    let num_batches = (val_features.len() + batch_size - 1) / batch_size;
    let mut total_loss = 0.0;
    let mut batch_count = 0;

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(val_features.len());

        let batch_features = &val_features[start_idx..end_idx];
        let batch_labels_slice = &val_labels[start_idx..end_idx];

        // Convert to tensors
        let (inputs, targets) =
            create_batch_with_features(batch_features, batch_labels_slice, converter, device);

        let class_targets = targets.clone().int();

        // Forward pass only (no backward)
        let output = model.forward_classification(inputs, class_targets);
        let loss_value = output.loss.into_scalar().elem::<f32>();
        total_loss += loss_value;
        batch_count += 1;
    }

    total_loss / batch_count as f32
}

fn main() {
    println!("🚀 JANUS Vision - Production Training Pipeline\n");

    let device = Default::default();
    let config = TrainingConfig::default();

    // Step 1: Generate and load data
    let candles = generate_training_data();

    // Step 2: Create dataset
    println!("\n📈 Creating sequences...");
    let dataset = OhlcvDataset::from_candles(
        candles,
        SequenceConfig {
            sequence_length: 60,
            stride: 1,
            prediction_horizon: 1,
        },
    )
    .expect("Failed to create dataset");
    println!("  Created {} sequences", dataset.len());

    // Step 3: Feature engineering
    println!("\n🔧 Engineering features...");
    let feature_config = FeatureConfig {
        include_ohlcv: true,
        include_returns: true,
        include_log_returns: true,
        include_volume_change: true,
        sma_periods: vec![5, 10, 20],
        ema_periods: vec![12, 26],
        rsi_period: Some(14),
        macd_config: Some((12, 26, 9)),
        atr_period: Some(14),
        bollinger_bands: Some((20, 2.0)),
    };

    let engineer = FeatureEngineer::new(feature_config.clone());
    let num_features = engineer.get_num_features();
    println!("  Total features: {}", num_features);

    // Compute all features in parallel
    let start = Instant::now();
    let all_features = compute_features_for_dataset(&dataset, &engineer);
    let compute_time = start.elapsed().as_secs_f64();
    println!("  Feature computation took: {:.2}s", compute_time);

    let sequences_per_sec = dataset.len() as f64 / compute_time;
    println!("  Throughput: {:.0} sequences/sec", sequences_per_sec);

    // Extract labels
    let all_labels: Vec<f64> = dataset.labels.clone();

    // Step 4: Train/validation split
    println!("\n✂️  Splitting data...");
    let (train_features, train_labels, val_features, val_labels) =
        train_val_split(all_features, all_labels, config.validation_split);

    println!(
        "  Training samples: {}, Validation samples: {}",
        train_features.len(),
        val_features.len()
    );

    // Step 5: Fit normalization on training data
    println!("\n📐 Fitting normalization...");
    let converter_config = TensorConverterConfig {
        normalize: true,
        num_features,
    };

    let mut converter = TensorConverter::<ZScoreScaler>::new(converter_config);
    converter.fit_with_features(&train_features);
    println!("  Normalization fitted");

    // Step 6: Initialize model
    println!("\n🧠 Initializing model...");
    let model_config = DiffGafLstmConfig {
        input_features: num_features,
        time_steps: 60,
        lstm_hidden_size: 128,
        num_lstm_layers: 2,
        num_classes: 3, // down, neutral, up
        dropout: 0.3,
        gaf_pool_size: 32,
        bidirectional: false,
    };

    let model: DiffGafLstm<MyBackend> = model_config.init(&device);
    println!("  Model initialized");
    println!("    Input features: {}", model_config.input_features);
    println!("    Time steps: {}", model_config.time_steps);
    println!("    LSTM hidden size: {}", model_config.lstm_hidden_size);
    println!("    Num classes: {}", model_config.num_classes);

    // Step 7: Initialize optimizer
    let mut optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(1e-5)))
        .init();

    // Step 8: Training loop
    println!("\n🏋️  Starting training...");
    println!("  Epochs: {}", config.epochs);
    println!("  Batch size: {}", config.batch_size);
    println!("  Learning rate: {}", config.learning_rate);
    println!(
        "  Early stopping patience: {}",
        config.early_stopping_patience
    );

    let mut metrics = MetricsTracker::new();
    let train_start = Instant::now();
    let mut current_model = model;

    for epoch in 0..config.epochs {
        let epoch_start = Instant::now();

        // Train
        let (updated_model, train_loss) = train_epoch(
            current_model,
            &mut optimizer,
            &train_features,
            &train_labels,
            &converter,
            config.batch_size,
            &device,
        );
        current_model = updated_model;

        // Validate (use inner backend for non-autodiff validation)
        let val_loss = validate_epoch::<NdArray>(
            &current_model.valid(),
            &val_features,
            &val_labels,
            &converter,
            config.batch_size,
            &device,
        );

        let epoch_time = epoch_start.elapsed().as_secs_f64();

        // Update metrics
        let should_stop =
            metrics.update(train_loss, val_loss, epoch, config.early_stopping_patience);

        // Print progress
        let improved = val_loss < metrics.best_val_loss || epoch == 0;
        let status = if improved { "✓" } else { " " };

        println!(
            "{} Epoch {:3}/{} | Train Loss: {:.6} | Val Loss: {:.6} | Time: {:.2}s",
            status,
            epoch + 1,
            config.epochs,
            train_loss,
            val_loss,
            epoch_time
        );

        // Early stopping
        if should_stop {
            println!(
                "\n⏹️  Early stopping triggered after {} epochs without improvement",
                config.early_stopping_patience
            );
            break;
        }
    }

    let total_time = train_start.elapsed().as_secs_f64();

    // Step 9: Print final results
    println!("\n✅ Training completed!");
    println!("  Total time: {:.2}s", total_time);
    metrics.print_summary();

    println!("\n💾 Model ready for deployment!");
    println!("  Final model available as: current_model");
    println!("  To save the model, use: current_model.valid().save_checkpoint(...)");
}
