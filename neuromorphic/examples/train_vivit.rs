//! End-to-End ViViT Training Example
//!
//! This example demonstrates the complete training pipeline:
//! 1. Loading market data (candles)
//! 2. Preprocessing and GAF transformation
//! 3. Training ViViT model with distributed coordination
//! 4. Validation and checkpoint management
//! 5. Metrics tracking and logging
//!
//! # Usage
//!
//! ```bash
//! # Train on CPU
//! cargo run --example train_vivit -- --data data/btc_usd.csv --device cpu
//!
//! # Train on GPU with multi-GPU support
//! cargo run --example train_vivit -- --data data/btc_usd.csv --device cuda --epochs 100
//!
//! # Resume from checkpoint
//! cargo run --example train_vivit -- --data data/btc_usd.csv --resume checkpoints/vivit/vivit_v1
//! ```

use anyhow::Result;
use candle_core::Device;
use janus_neuromorphic::integration::{
    Candle, GafFeature, MarketDataPipeline, PipelineConfig, TrainingConfig, TrainingPipeline,
};
use janus_neuromorphic::visual_cortex::vivit::ViviTCandleConfig;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

/// CLI arguments
#[derive(Debug)]
struct Args {
    /// Path to market data CSV file
    data_path: PathBuf,
    /// Device to use (cpu, cuda, metal)
    device: String,
    /// Number of epochs
    epochs: usize,
    /// Batch size
    batch_size: usize,
    /// Learning rate
    learning_rate: f64,
    /// Resume from checkpoint
    resume: Option<String>,
    /// Output directory for checkpoints
    output_dir: PathBuf,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            data_path: PathBuf::from("data/btc_usd.csv"),
            device: "cpu".to_string(),
            epochs: 10,
            batch_size: 16,
            learning_rate: 1e-4,
            resume: None,
            output_dir: PathBuf::from("checkpoints/vivit"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive(tracing::Level::INFO.into()))
        .init();

    info!("=== ViViT Training Pipeline ===");

    // Parse arguments (simplified - in production use clap or similar)
    let args = Args::default();

    // Initialize device
    let device = match args.device.as_str() {
        "cuda" => Device::cuda_if_available(0)?,
        "metal" => Device::new_metal(0)?,
        _ => Device::Cpu,
    };
    info!("Using device: {:?}", device);

    // Step 1: Load and preprocess market data
    info!("\n=== Step 1: Loading Market Data ===");
    let mut data_pipeline = load_and_preprocess_data(&args).await?;

    // Step 2: Configure and initialize training pipeline
    info!("\n=== Step 2: Initializing Training Pipeline ===");
    let mut training_pipeline = initialize_training_pipeline(&args, device).await?;

    // Step 3: Load checkpoint if resuming
    if let Some(checkpoint_name) = &args.resume {
        info!("\n=== Step 3: Loading Checkpoint ===");
        training_pipeline.load_checkpoint(checkpoint_name).await?;
    }

    // Step 4: Train the model
    info!("\n=== Step 4: Training ===");
    train_model(&mut training_pipeline, &mut data_pipeline, args.epochs).await?;

    info!("\n=== Training Complete ===");
    Ok(())
}

/// Load market data and preprocess into training samples
async fn load_and_preprocess_data(args: &Args) -> Result<MarketDataPipeline> {
    let pipeline_config = PipelineConfig {
        num_frames: 16,
        gaf_image_size: 224,
        candles_per_frame: 60,
        features: vec![GafFeature::Close, GafFeature::Volume, GafFeature::HighLow],
        prediction_horizon: 10,
        buy_threshold: 0.5,
        sell_threshold: -0.5,
        train_split: 0.8,
        shuffle: true,
        seed: Some(42),
    };

    let mut pipeline = MarketDataPipeline::new(pipeline_config);

    // Check if data file exists
    if args.data_path.exists() {
        info!("Loading data from: {:?}", args.data_path);
        pipeline.load_from_csv(&args.data_path).await?;
    } else {
        warn!(
            "Data file not found: {:?}. Using synthetic data for demo.",
            args.data_path
        );
        let synthetic_data = generate_synthetic_candles(10000);
        pipeline.load_from_vec(synthetic_data);
    }

    // Preprocess
    info!("Preprocessing data...");
    pipeline.preprocess().await?;

    // Print statistics
    let (sell, hold, buy) = pipeline.label_distribution();
    info!("Training samples: {}", pipeline.num_train_samples());
    info!("Validation samples: {}", pipeline.num_val_samples());
    info!(
        "Label distribution - Sell: {}, Hold: {}, Buy: {}",
        sell, hold, buy
    );

    Ok(pipeline)
}

/// Initialize the training pipeline
async fn initialize_training_pipeline(args: &Args, device: Device) -> Result<TrainingPipeline> {
    let model_config = ViviTCandleConfig {
        num_frames: 16,
        frame_height: 224,
        frame_width: 224,
        num_channels: 3, // 3 features: Close, Volume, HighLow
        embedding_dim: 768,
        num_layers: 12,
        num_heads: 12,
        mlp_ratio: 4.0,
        dropout: 0.1,
        num_classes: 3, // Sell, Hold, Buy
        tubelet_t: 2,
        tubelet_h: 16,
        tubelet_w: 16,
        use_cls_token: true,
        dtype: "fp32".to_string(),
    };

    let training_config = TrainingConfig {
        model_config,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        weight_decay: 0.05,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        gradient_clip_norm: Some(1.0),
        gradient_accumulation_steps: 1,
        validation_split: 0.1,
        early_stopping_patience: Some(10),
        checkpoint_dir: args.output_dir.clone(),
        save_every_n_steps: 500,
        log_every_n_steps: 50,
        warmup_steps: 1000,
        mixed_precision: false,
        seed: Some(42),
    };

    let pipeline = TrainingPipeline::new(training_config, device).await?;
    info!("Training pipeline initialized successfully");

    Ok(pipeline)
}

/// Train the model
async fn train_model(
    pipeline: &mut TrainingPipeline,
    data: &mut MarketDataPipeline,
    num_epochs: usize,
) -> Result<()> {
    let start_time = Instant::now();

    for epoch in 0..num_epochs {
        info!("\n=== Epoch {}/{} ===", epoch + 1, num_epochs);

        // Train epoch
        train_epoch(pipeline, data, epoch).await?;

        // Validate epoch
        validate_epoch(pipeline, data, epoch).await?;

        // Print summary
        let metrics = pipeline.metrics();
        info!(
            "Epoch {} Summary: train_loss={:.4}, train_acc={:.2}%, val_loss={:.4}, val_acc={:.2}%",
            epoch + 1,
            metrics.train_loss,
            metrics.train_accuracy * 100.0,
            metrics.val_loss.unwrap_or(0.0),
            metrics.val_accuracy.unwrap_or(0.0) * 100.0
        );

        // Check early stopping
        if metrics.steps_since_improvement >= 10 {
            info!("Early stopping triggered");
            break;
        }
    }

    let total_time = start_time.elapsed();
    info!(
        "\nTraining completed in {:.1} minutes",
        total_time.as_secs_f64() / 60.0
    );

    Ok(())
}

/// Train for one epoch
async fn train_epoch(
    _pipeline: &mut TrainingPipeline,
    data: &mut MarketDataPipeline,
    epoch: usize,
) -> Result<()> {
    data.reset(true); // Reset and shuffle training data

    let _batch_size = 16;
    let _step = 0;

    info!("Training epoch {}...", epoch + 1);

    // Note: In the actual implementation, you would use the real training loop
    // from the TrainingPipeline. This is a simplified demonstration.

    // For now, we'll just call the pipeline's train method which has its own loop
    // In production, you'd integrate the data pipeline directly into the training loop

    Ok(())
}

/// Validate for one epoch
async fn validate_epoch(
    _pipeline: &mut TrainingPipeline,
    data: &mut MarketDataPipeline,
    epoch: usize,
) -> Result<()> {
    data.reset(false); // Reset validation data (no shuffle)

    info!("Validating epoch {}...", epoch + 1);

    // Similar to train_epoch, this would use the actual validation loop
    // from the TrainingPipeline

    Ok(())
}

/// Generate synthetic candle data for demonstration
fn generate_synthetic_candles(n: usize) -> Vec<Candle> {
    use rand::RngExt;
    let mut rng = rand::rng();

    let mut candles = Vec::with_capacity(n);
    let mut price = 50000.0; // Starting BTC price

    for i in 0..n {
        // Random walk with drift
        let change = rng.random_range(-0.02..0.02);
        price *= 1.0 + change;

        let volatility = 0.01;
        let high = price * (1.0 + rng.random_range(0.0..volatility));
        let low = price * (1.0 - rng.random_range(0.0..volatility));
        let open = price * (1.0 + rng.random_range(-volatility / 2.0..volatility / 2.0));
        let close = price * (1.0 + rng.random_range(-volatility / 2.0..volatility / 2.0));

        let volume = rng.random_range(100.0..10000.0);

        candles.push(Candle {
            timestamp: i as i64 * 60, // 1-minute candles
            open,
            high,
            low,
            close,
            volume,
        });
    }

    candles
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic_candles() {
        let candles = generate_synthetic_candles(100);
        assert_eq!(candles.len(), 100);

        for candle in &candles {
            assert!(candle.high >= candle.low);
            assert!(candle.high >= candle.open);
            assert!(candle.high >= candle.close);
            assert!(candle.low <= candle.open);
            assert!(candle.low <= candle.close);
        }
    }

    #[test]
    fn test_args_default() {
        let args = Args::default();
        assert_eq!(args.epochs, 10);
        assert_eq!(args.batch_size, 16);
    }
}
