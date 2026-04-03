//! End-to-end example: Training DiffGAF-LSTM with real OHLCV data
//!
//! This example demonstrates:
//! 1. Loading OHLCV data from CSV
//! 2. Data validation and quality checks
//! 3. Creating sequences with temporal splits
//! 4. Preprocessing and normalization
//! 5. Tensor conversion for Burn
//! 6. Training a DiffGAF-LSTM model
//!
//! Run with: cargo run --example train_with_real_data --release

use anyhow::Result;
use burn::backend::NdArray;
use std::io::Write;
use std::path::Path;
use vision::{
    DiffGafLstmConfig, FeatureConfig, FeatureEngineer, MinMaxScaler, OhlcvDataset, SequenceConfig,
    TensorConverter, TensorConverterConfig, TrainValSplit, load_ohlcv_csv,
};

type MyBackend = NdArray<f32>;

/// Create sample OHLCV CSV data for demonstration
fn create_sample_csv(path: &Path) -> Result<()> {
    let mut file = std::fs::File::create(path)?;

    // Write CSV header
    writeln!(file, "timestamp,open,high,low,close,volume")?;

    // Generate 1000 candles with realistic price movements
    let mut price = 50000.0;
    let base_time = 1704067200000i64; // 2024-01-01 00:00:00 UTC in milliseconds

    for i in 0..1000 {
        let timestamp = base_time + (i * 60000); // 1-minute candles

        // Simulate price movement with random walk
        let change = (((i * 17) % 100) as f64 - 50.0) / 50.0;
        price += change * 10.0;
        price = price.max(40000.0).min(60000.0); // Keep price in reasonable range

        let volatility = 0.002;
        let open = price;
        let high = price * (1.0 + volatility);
        let low = price * (1.0 - volatility);
        let close = price + (((i * 23) % 100) as f64 - 50.0) / 100.0;
        let volume = 100.0 + (((i * 13) % 500) as f64);

        writeln!(
            file,
            "{},{:.2},{:.2},{:.2},{:.2},{:.2}",
            timestamp, open, high, low, close, volume
        )?;

        price = close;
    }

    Ok(())
}

fn main() -> Result<()> {
    println!("=== DiffGAF-LSTM Training with Real Data Pipeline ===\n");

    // Step 1: Create sample data
    println!("Step 1: Creating sample CSV data...");
    let csv_path = Path::new("sample_ohlcv.csv");
    create_sample_csv(csv_path)?;
    println!("  ✓ Created {} with 1000 candles\n", csv_path.display());

    // Step 2: Load and validate data
    println!("Step 2: Loading and validating data...");
    let candles = load_ohlcv_csv(csv_path)?;
    println!("  ✓ Loaded {} candles", candles.len());

    let validation = vision::validate_ohlcv(&candles);
    println!("  ✓ Validation report:");
    println!("    - Total candles: {}", validation.total_candles);
    println!("    - Errors: {}", validation.errors.len());
    println!("    - Warnings: {}", validation.warnings.len());
    println!("    - Is valid: {}\n", validation.is_valid());

    // Step 3: Create sequences
    println!("Step 3: Creating sequences...");
    let seq_config = SequenceConfig {
        sequence_length: 60,
        stride: 5,
        prediction_horizon: 1,
    };

    let dataset = OhlcvDataset::from_candles(candles, seq_config)?;
    println!("  ✓ Created {} sequences", dataset.len());

    let stats = dataset.stats();
    println!("  ✓ Dataset statistics:");
    println!("    - Mean return: {:.6}", stats.mean_return);
    println!("    - Std return: {:.6}", stats.std_return);
    println!("    - Min return: {:.6}", stats.min_return);
    println!("    - Max return: {:.6}\n", stats.max_return);

    // Step 4: Split into train/val/test
    println!("Step 4: Splitting dataset...");
    let split_config = TrainValSplit::default();
    let (train_dataset, val_dataset, test_dataset) = dataset.split(split_config)?;
    println!("  ✓ Train: {} sequences", train_dataset.len());
    println!("  ✓ Val:   {} sequences", val_dataset.len());
    println!("  ✓ Test:  {} sequences\n", test_dataset.len());

    // Step 5: Feature engineering (optional - demonstrates technical indicators)
    println!("Step 5: Feature engineering with technical indicators...");

    // Option 1: Use raw OHLCV features only (5 features)
    let use_indicators = false; // Set to true to enable technical indicators

    let num_features = if use_indicators {
        let feature_config = FeatureConfig::with_common_indicators();
        println!(
            "  ✓ Using enhanced features: {} total",
            feature_config.num_features()
        );
        println!("    - OHLCV: 5");
        println!("    - Returns + Volume change: 2");
        println!("    - SMA (10, 20): 2");
        println!("    - EMA (8, 21): 2");
        println!("    - RSI: 1");
        println!("    - MACD (line, signal, histogram): 3");
        println!("    - ATR: 1");
        println!("    - Bollinger Bands (upper, mid, lower): 3");

        let engineer = FeatureEngineer::new(feature_config.clone());

        // Compute features for a sample sequence to demonstrate
        if let Some((seq, _)) = train_dataset.get(0) {
            let features = engineer.compute_features(seq).unwrap();
            println!(
                "  ✓ Sample features computed: {}x{}",
                features.len(),
                features[0].len()
            );
        }

        feature_config.num_features()
    } else {
        println!("  ✓ Using raw OHLCV features only: 5");
        5
    };
    println!();

    // Step 6: Setup preprocessing
    println!("Step 6: Setting up normalization...");
    let converter_config = TensorConverterConfig {
        normalize: true,
        num_features,
    };

    let mut converter = TensorConverter::<MinMaxScaler>::new(converter_config);
    converter.fit(&train_dataset);
    println!("  ✓ Fitted MinMax scaler on training data");
    println!("  ✓ Converter is fitted: {}\n", converter.is_fitted());

    // Step 7: Convert sample sequences to tensors
    println!("Step 7: Converting samples to tensors...");
    let device = Default::default();

    // Convert first training sequence
    if let Some((seq, label)) = train_dataset.get(0) {
        let tensor = converter.sequence_to_tensor::<MyBackend>(seq, &device);
        println!("  ✓ Sample tensor shape: {:?}", tensor.shape());
        println!("  ✓ Sample label: {:.6}\n", label);
    }

    // Step 8: Initialize model (without training)
    println!("Step 8: Initializing DiffGAF-LSTM model...");
    let model_config = DiffGafLstmConfig {
        input_features: num_features,
        time_steps: 60,
        lstm_hidden_size: 64,
        num_lstm_layers: 2,
        num_classes: 3,
        dropout: 0.2,
        gaf_pool_size: 4,
        bidirectional: false,
    };

    println!("  ✓ Model configuration:");
    println!("    - Input features: {}", model_config.input_features);
    println!("    - Time steps: {}", model_config.time_steps);
    println!("    - LSTM hidden size: {}", model_config.lstm_hidden_size);
    println!("    - LSTM layers: {}", model_config.num_lstm_layers);
    println!("    - Classes: {}", model_config.num_classes);
    println!("    - Dropout: {}", model_config.dropout);
    println!("    - GAF pool size: {}", model_config.gaf_pool_size);
    println!("    - Bidirectional: {}\n", model_config.bidirectional);

    // Step 9: Classification labels
    println!("Step 9: Generating classification labels...");
    let threshold = 0.001; // 0.1% threshold
    let train_labels = train_dataset.to_classification_labels(threshold);
    let val_labels = val_dataset.to_classification_labels(threshold);

    let mut class_counts = [0usize; 3];
    for &label in &train_labels {
        class_counts[label] += 1;
    }

    println!("  ✓ Training set class distribution:");
    println!(
        "    - Down (0):    {} ({:.1}%)",
        class_counts[0],
        100.0 * class_counts[0] as f64 / train_labels.len() as f64
    );
    println!(
        "    - Neutral (1): {} ({:.1}%)",
        class_counts[1],
        100.0 * class_counts[1] as f64 / train_labels.len() as f64
    );
    println!(
        "    - Up (2):      {} ({:.1}%)\n",
        class_counts[2],
        100.0 * class_counts[2] as f64 / train_labels.len() as f64
    );

    // Step 10: Batch iteration example
    println!("Step 10: Demonstrating batch iteration...");
    use vision::BatchIterator;
    let batch_size = 16;
    let mut batch_iter = BatchIterator::new(&train_dataset, batch_size);

    println!("  ✓ Total batches: {}", batch_iter.num_batches());

    let mut batch_count = 0;
    while let Some((indices, is_last)) = batch_iter.next_batch() {
        batch_count += 1;
        if batch_count == 1 {
            println!("  ✓ First batch size: {}", indices.len());
        }
        if is_last {
            println!("  ✓ Last batch size: {}", indices.len());
            break;
        }
    }
    println!();

    // Cleanup
    if csv_path.exists() {
        std::fs::remove_file(csv_path)?;
        println!("Cleaned up sample data file.");
    }

    println!("\n=== Pipeline Demo Complete ===");
    println!("\n✓ Successfully demonstrated:");
    println!("  1. CSV data loading");
    println!("  2. Data validation");
    println!("  3. Sequence generation");
    println!("  4. Train/val/test splitting");
    println!("  5. Feature engineering (optional)");
    println!("  6. Feature normalization");
    println!("  7. Tensor conversion");
    println!("  8. Model initialization");
    println!("  9. Classification label generation");
    println!("  10. Batch iteration");

    println!("\nNext steps to implement actual training:");
    println!("  1. Enable feature engineering (set use_indicators = true)");
    println!("  2. Add training loop with forward/backward passes");
    println!("  3. Implement validation metrics (accuracy, F1-score)");
    println!("  4. Add model checkpointing and early stopping");
    println!("  5. Integrate with real market data sources");
    println!("  6. Add visualization of GAF images");
    println!("  7. Experiment with different feature configurations");

    Ok(())
}
