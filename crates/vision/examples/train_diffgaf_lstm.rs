//! Training example for DiffGAF-LSTM model
//!
//! This example demonstrates:
//! - Creating synthetic time series data
//! - Configuring and initializing the model
//! - Training loop with metrics tracking
//! - Checkpoint management
//! - Validation and early stopping
//!
//! Run with:
//! ```bash
//! cargo run --package vision --example train_diffgaf_lstm
//! ```

use burn::backend::ndarray::NdArrayDevice;
use burn::backend::{Autodiff, NdArray};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{ElementConversion, Tensor, backend::Backend};
use std::path::Path;
use vision::{BestModelTracker, CheckpointMetadata, DiffGafLstm, DiffGafLstmConfig};

type MyBackend = Autodiff<NdArray>;

/// Synthetic dataset for training
struct SyntheticDataset {
    inputs: Vec<Vec<Vec<f32>>>, // [samples, time, features]
    targets: Vec<usize>,        // [samples]
}

impl SyntheticDataset {
    /// Generate synthetic time series data
    fn generate(num_samples: usize, time_steps: usize, num_features: usize) -> Self {
        use rand::{Rng, RngExt};
        let mut rng = rand::rng();

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for _ in 0..num_samples {
            let mut sample = Vec::new();

            // Generate sinusoidal patterns with noise
            for t in 0..time_steps {
                let mut timestep = Vec::new();
                for f in 0..num_features {
                    let phase = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
                    let freq = 0.1 + (f as f32) * 0.05;
                    let value =
                        (freq * (t as f32) + phase).sin() + rng.random::<f32>() * 0.1 - 0.05;
                    timestep.push(value);
                }
                sample.push(timestep);
            }

            // Classify based on average trend
            let avg_trend: f32 = sample.iter().flat_map(|t| t.iter()).sum::<f32>()
                / (time_steps * num_features) as f32;

            let target = if avg_trend > 0.2 {
                0 // Strong uptrend
            } else if avg_trend < -0.2 {
                2 // Strong downtrend
            } else {
                1 // Neutral
            };

            inputs.push(sample);
            targets.push(target);
        }

        Self { inputs, targets }
    }

    /// Split dataset into train and validation sets
    fn split(self, train_ratio: f32) -> (Self, Self) {
        let split_idx = (self.inputs.len() as f32 * train_ratio) as usize;

        let train = Self {
            inputs: self.inputs[..split_idx].to_vec(),
            targets: self.targets[..split_idx].to_vec(),
        };

        let val = Self {
            inputs: self.inputs[split_idx..].to_vec(),
            targets: self.targets[split_idx..].to_vec(),
        };

        (train, val)
    }

    /// Get batch of data as tensors
    fn get_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 1, burn::tensor::Int>) {
        let batch_size = indices.len();
        let time_steps = self.inputs[0].len();
        let num_features = self.inputs[0][0].len();

        // Create input tensor [batch, time, features]
        let mut batch_inputs = Vec::new();
        for &idx in indices {
            batch_inputs.push(self.inputs[idx].clone());
        }

        // Flatten to 1D for Tensor creation
        let mut flat_data = Vec::new();
        for sample in &batch_inputs {
            for timestep in sample {
                for &value in timestep {
                    flat_data.push(value);
                }
            }
        }

        // Create tensor with shape information
        let shape = burn::tensor::Shape::new([batch_size, time_steps, num_features]);
        let data = burn::tensor::TensorData::new(flat_data, shape);
        let inputs = Tensor::from_data(data.convert::<f32>(), device);

        // Create target tensor
        let target_data: Vec<i64> = indices
            .iter()
            .map(|&idx| self.targets[idx] as i64)
            .collect();
        let target_shape = burn::tensor::Shape::new([batch_size]);
        let target_tensor_data = burn::tensor::TensorData::new(target_data, target_shape);
        let targets = Tensor::from_data(target_tensor_data.convert::<i64>(), device);

        (inputs, targets)
    }

    fn len(&self) -> usize {
        self.inputs.len()
    }
}

/// Training metrics tracker
struct MetricsTracker {
    train_losses: Vec<f64>,
    val_losses: Vec<f64>,
    val_accuracies: Vec<f64>,
}

impl MetricsTracker {
    fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            val_accuracies: Vec::new(),
        }
    }

    fn record(&mut self, train_loss: f64, val_loss: f64, val_accuracy: f64) {
        self.train_losses.push(train_loss);
        self.val_losses.push(val_loss);
        self.val_accuracies.push(val_accuracy);
    }

    fn print_summary(&self) {
        if self.train_losses.is_empty() {
            return;
        }

        println!("\n=== Training Summary ===");
        println!("Total epochs: {}", self.train_losses.len());
        println!("Final train loss: {:.4}", self.train_losses.last().unwrap());
        println!("Final val loss: {:.4}", self.val_losses.last().unwrap());
        println!(
            "Best val loss: {:.4}",
            self.val_losses.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        );
        println!(
            "Final val accuracy: {:.2}%",
            self.val_accuracies.last().unwrap() * 100.0
        );
        println!(
            "Best val accuracy: {:.2}%",
            self.val_accuracies.iter().fold(0.0_f64, |a, &b| a.max(b)) * 100.0
        );
    }
}

/// Early stopping tracker
struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    counter: usize,
    best_loss: Option<f64>,
}

impl EarlyStopping {
    fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            counter: 0,
            best_loss: None,
        }
    }

    fn check(&mut self, val_loss: f64) -> bool {
        let improved = match self.best_loss {
            Some(best) => val_loss < best - self.min_delta,
            None => true,
        };

        if improved {
            self.best_loss = Some(val_loss);
            self.counter = 0;
            false // Don't stop
        } else {
            self.counter += 1;
            self.counter >= self.patience // Stop if patience exceeded
        }
    }
}

/// Train for one epoch
fn train_epoch(
    model: &mut DiffGafLstm<MyBackend>,
    dataset: &SyntheticDataset,
    optimizer: &mut impl Optimizer<DiffGafLstm<MyBackend>, MyBackend>,
    batch_size: usize,
    device: &<MyBackend as Backend>::Device,
) -> f64 {
    let num_batches = (dataset.len() + batch_size - 1) / batch_size;
    let mut total_loss = 0.0;

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(dataset.len());
        let indices: Vec<usize> = (start_idx..end_idx).collect();

        let (inputs, targets) = dataset.get_batch(&indices, device);

        // Forward pass
        let output = model.forward_classification(inputs, targets);
        let loss = output.loss.clone();

        // Backward pass
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, model);
        *model = optimizer.step(1.0, model.clone(), grads);

        // Accumulate loss
        let loss_item: f32 = loss.clone().into_scalar().elem();
        total_loss += loss_item as f64;
    }

    total_loss / num_batches as f64
}

/// Validate the model
fn validate(
    model: &DiffGafLstm<MyBackend>,
    dataset: &SyntheticDataset,
    batch_size: usize,
    device: &<MyBackend as Backend>::Device,
) -> (f64, f64) {
    let num_batches = (dataset.len() + batch_size - 1) / batch_size;
    let mut total_loss = 0.0;
    let mut correct = 0;
    let mut total = 0;

    for batch_idx in 0..num_batches {
        let start_idx = batch_idx * batch_size;
        let end_idx = (start_idx + batch_size).min(dataset.len());
        let indices: Vec<usize> = (start_idx..end_idx).collect();

        let (inputs, targets) = dataset.get_batch(&indices, device);

        // Forward pass (no gradients needed)
        let output = model.forward_classification(inputs, targets.clone());
        let loss = output.loss;
        let logits = output.output;

        // Accumulate loss
        let loss_item: f32 = loss.clone().into_scalar().elem();
        total_loss += loss_item as f64;

        // Calculate accuracy
        let predictions = logits.argmax(1);
        let targets_data = targets.into_data();
        let preds_data = predictions.into_data();

        // Convert to vectors for comparison
        let target_vec: Vec<i64> = targets_data.to_vec().unwrap();
        let pred_vec: Vec<i64> = preds_data.to_vec().unwrap();

        for (pred, target) in pred_vec.iter().zip(target_vec.iter()) {
            if pred == target {
                correct += 1;
            }
            total += 1;
        }
    }

    let avg_loss = total_loss / num_batches as f64;
    let accuracy = correct as f64 / total as f64;

    (avg_loss, accuracy)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 DiffGAF-LSTM Training Example\n");

    // Configuration
    let num_samples = 1000;
    let time_steps = 60;
    let num_features = 5;
    let num_classes = 3;
    let batch_size = 32;
    let num_epochs = 50;

    let device = NdArrayDevice::Cpu;

    println!("📊 Dataset Configuration:");
    println!("  Samples: {}", num_samples);
    println!("  Time steps: {}", time_steps);
    println!("  Features: {}", num_features);
    println!("  Classes: {}", num_classes);
    println!("  Batch size: {}\n", batch_size);

    // Generate synthetic dataset
    println!("🔄 Generating synthetic dataset...");
    let dataset = SyntheticDataset::generate(num_samples, time_steps, num_features);
    let (train_data, val_data) = dataset.split(0.8);
    println!("  Train samples: {}", train_data.len());
    println!("  Val samples: {}\n", val_data.len());

    // Create model
    println!("🏗️  Creating model...");
    let config = DiffGafLstmConfig {
        input_features: num_features,
        time_steps,
        lstm_hidden_size: 64,
        num_lstm_layers: 2,
        num_classes,
        dropout: 0.2,
        gaf_pool_size: 16,
        bidirectional: false,
    };

    let mut model = config.init::<MyBackend>(&device);
    println!("  Model initialized with config:");
    println!("    LSTM hidden size: {}", config.lstm_hidden_size);
    println!("    LSTM layers: {}", config.num_lstm_layers);
    println!("    GAF pool size: {}", config.gaf_pool_size);
    println!("    Dropout: {}\n", config.dropout);

    // Create optimizer
    println!("⚙️  Setting up optimizer...");
    let mut optimizer = AdamConfig::new().init();
    println!("  Adam optimizer\n");

    // Setup tracking
    let mut metrics = MetricsTracker::new();
    let mut early_stopping = EarlyStopping::new(10, 0.001);
    let mut best_tracker = BestModelTracker::new("checkpoints/best_model");

    // Create checkpoints directory
    std::fs::create_dir_all("checkpoints")?;

    println!("🎯 Starting training for {} epochs...\n", num_epochs);
    println!("{:-<80}", "");

    // Training loop
    for epoch in 1..=num_epochs {
        // Train
        let train_loss = train_epoch(&mut model, &train_data, &mut optimizer, batch_size, &device);

        // Validate
        let (val_loss, val_accuracy) = validate(&model, &val_data, batch_size, &device);

        // Record metrics
        metrics.record(train_loss, val_loss, val_accuracy);

        // Print progress
        print!(
            "Epoch {:3}/{}: train_loss={:.4}, val_loss={:.4}, val_acc={:.2}%",
            epoch,
            num_epochs,
            train_loss,
            val_loss,
            val_accuracy * 100.0
        );

        // Check for best model
        let is_best = best_tracker
            .update(&model, epoch, train_loss, val_loss, config.clone())
            .unwrap_or(false);

        if is_best {
            print!(" ⭐ [Best model saved!]");
        }

        println!();

        // Save periodic checkpoints
        if epoch % 10 == 0 {
            let metadata =
                CheckpointMetadata::new(epoch, train_loss, Some(val_loss), config.clone());
            let checkpoint_path = format!("checkpoints/epoch_{}", epoch);
            if let Err(e) = model.save_checkpoint(&checkpoint_path, metadata) {
                eprintln!("Warning: Failed to save checkpoint: {}", e);
            } else {
                println!("  💾 Checkpoint saved: {}", checkpoint_path);
            }
        }

        // Check early stopping
        if early_stopping.check(val_loss) {
            println!("\n⏸️  Early stopping triggered at epoch {}", epoch);
            println!("  No improvement for {} epochs", early_stopping.patience);
            break;
        }
    }

    println!("{:-<80}", "");
    metrics.print_summary();

    println!("\n✅ Training complete!");
    println!("📁 Checkpoints saved in: ./checkpoints/");
    println!("   - best_model.bin (best validation loss)");
    println!("   - epoch_*.bin (periodic checkpoints)");

    // Demonstrate loading checkpoint
    println!("\n🔄 Testing checkpoint loading...");
    if Path::new("checkpoints/best_model.bin").exists() {
        match DiffGafLstm::load_checkpoint("checkpoints/best_model", &device) {
            Ok((loaded_model, metadata)) => {
                println!("✅ Successfully loaded best model:");
                println!("   Epoch: {}", metadata.epoch);
                println!("   Train loss: {:.4}", metadata.train_loss);
                println!("   Val loss: {:.4}", metadata.val_loss.unwrap_or(0.0));

                // Quick validation with loaded model
                let (val_loss, val_accuracy) =
                    validate(&loaded_model, &val_data, batch_size, &device);
                println!("   Verified val loss: {:.4}", val_loss);
                println!("   Verified val accuracy: {:.2}%", val_accuracy * 100.0);
            }
            Err(e) => {
                eprintln!("❌ Failed to load checkpoint: {}", e);
            }
        }
    }

    println!("\n🎉 Example complete!");

    Ok(())
}
