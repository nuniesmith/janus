//! Signal Generation Example
//!
//! This example demonstrates the complete pipeline for generating trading signals
//! from a DiffGAF-LSTM model, including:
//!
//! 1. Model initialization
//! 2. Inference on sample data
//! 3. Signal generation with confidence scoring
//! 4. Signal filtering for quality control
//!
//! # Usage
//!
//! ```bash
//! cargo run --example signal_generation --release
//! ```

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::tensor::{Shape, Tensor, TensorData};

use vision::diffgaf::combined::DiffGafLstmConfig;
use vision::signals::{
    ConfidenceConfig, FilterConfig, GeneratorConfig, PipelineBuilder, SignalType,
};

type Backend = NdArray;

/// Generate random input data for demonstration
fn generate_sample_inputs(
    batch_size: usize,
    time_steps: usize,
    num_features: usize,
) -> Tensor<Backend, 3> {
    let device = NdArrayDevice::Cpu;

    // Generate random-like data using a simple PRNG
    let total_size = batch_size * time_steps * num_features;
    let mut rng = 12345u64;
    let mut data = Vec::with_capacity(total_size);

    for _ in 0..total_size {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        let value = ((rng % 1000) as f32) / 1000.0; // 0.0 to 1.0
        data.push(value);
    }

    let shape = Shape::new([batch_size, time_steps, num_features]);
    let tensor_data = TensorData::new(data, shape);
    Tensor::<Backend, 3>::from_data(tensor_data.convert::<f32>(), &device)
}

fn main() {
    println!("\n=== DiffGAF Vision: Signal Generation Pipeline ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Configuration
    // -------------------------------------------------------------------------
    println!("📋 Step 1: Configuration\n");

    let batch_size = 5;
    let time_steps = 60; // 60 timesteps (e.g., 60 minutes of data)
    let num_features = 10; // 10 technical indicators per timestep
    let num_classes = 3; // Buy, Hold, Sell

    println!("  Model Configuration:");
    println!("    - Input features: {}", num_features);
    println!("    - Time steps: {}", time_steps);
    println!("    - Output classes: {} (Buy, Hold, Sell)", num_classes);

    // Create model configuration
    let model_config = DiffGafLstmConfig {
        input_features: num_features,
        time_steps,
        num_classes,
        lstm_hidden_size: 64,
        num_lstm_layers: 2,
        dropout: 0.2,
        gaf_pool_size: 16,
        bidirectional: false,
    };

    println!("\n  Signal Generation:");
    println!("    - Confidence threshold: 65%");
    println!("    - Default position size: 10%");
    println!("    - Generate Hold signals: No (only actionable signals)");

    println!("\n  Signal Filtering:");
    println!("    - Min confidence: 65%");
    println!("    - Max positions: 3");
    println!("    - Max signal age: 60 seconds");
    println!("    - Min interval between signals: 300 seconds");

    // -------------------------------------------------------------------------
    // Step 2: Initialize Model
    // -------------------------------------------------------------------------
    println!("\n🤖 Step 2: Model Initialization\n");

    let device = NdArrayDevice::Cpu;
    let model = model_config.init(&device);

    println!("  ✓ DiffGAF-LSTM model initialized");
    println!(
        "  ✓ Model parameters: ~{} layers",
        model_config.num_lstm_layers
    );
    println!("\n  ⚠️  Note: Using untrained model for demonstration");
    println!("     In production: Load trained weights from checkpoint");

    // -------------------------------------------------------------------------
    // Step 3: Create Signal Pipeline
    // -------------------------------------------------------------------------
    println!("\n⚙️  Step 3: Signal Pipeline Setup\n");

    // Configure signal generation
    let generator_config = GeneratorConfig {
        confidence: ConfidenceConfig::with_threshold(0.65),
        class_mapping: [0, 1, 2], // Buy=0, Hold=1, Sell=2
        default_position_size: Some(0.1),
        include_probabilities: true,
        generate_hold_signals: false,
    };

    // Configure signal filtering
    let filter_config = FilterConfig {
        min_confidence: 0.65,
        max_positions: 3,
        max_signal_age_seconds: 60,
        min_signal_interval_seconds: 300,
        max_position_size: 0.15,
        allowed_signal_types: vec![SignalType::Buy, SignalType::Sell, SignalType::Close],
        allowed_assets: vec![],
        blocked_assets: vec![],
    };

    // Build integrated pipeline
    let pipeline = PipelineBuilder::new()
        .model(model)
        .generator_config(generator_config)
        .filter_config(filter_config)
        .use_softmax(true)
        .build()
        .expect("Failed to build signal pipeline");

    println!("  ✓ Signal generator configured");
    println!("  ✓ Signal filter configured");
    println!("  ✓ Integrated pipeline ready");

    // -------------------------------------------------------------------------
    // Step 4: Prepare Sample Data
    // -------------------------------------------------------------------------
    println!("\n📊 Step 4: Sample Data\n");

    let assets = vec!["BTCUSD", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"];
    let inputs = generate_sample_inputs(batch_size, time_steps, num_features);

    println!("  Generated sample data for {} assets:", assets.len());
    for asset in &assets {
        println!("    - {}", asset);
    }
    println!(
        "\n  Input shape: [{}, {}, {}]",
        batch_size, time_steps, num_features
    );

    // -------------------------------------------------------------------------
    // Step 5: Run Inference & Generate Signals
    // -------------------------------------------------------------------------
    println!("\n🔮 Step 5: Inference & Signal Generation\n");

    let signals = match pipeline.process_batch(&inputs, &assets) {
        Ok(sigs) => {
            println!("  ✓ Inference complete");
            println!("  ✓ Signals generated and filtered");
            println!("  ✓ {} signals passed all filters\n", sigs.len());
            sigs
        }
        Err(e) => {
            eprintln!("  ✗ Error: {}", e);
            eprintln!("\n  This may occur with mismatched tensor dimensions.");
            eprintln!("  Ensure model config matches input data shape.");
            return;
        }
    };

    // -------------------------------------------------------------------------
    // Step 6: Display Results
    // -------------------------------------------------------------------------
    println!("═══════════════════════════════════════════════════════");
    println!("                  TRADING SIGNALS                      ");
    println!("═══════════════════════════════════════════════════════\n");

    if signals.is_empty() {
        println!("  No signals passed filters.\n");
        println!("  Reasons:");
        println!("    • Confidence below 65% threshold");
        println!("    • Hold signals filtered out (not actionable)");
        println!("    • Position limits reached\n");
        println!("  ℹ️  This is expected with an untrained model.");
        println!("     Train the model first for meaningful signals.\n");
    } else {
        for (i, signal) in signals.iter().enumerate() {
            println!("Signal #{}", i + 1);
            println!("  Asset:      {}", signal.asset);
            println!("  Action:     {:?}", signal.signal_type);
            println!("  Confidence: {:.1}%", signal.confidence * 100.0);
            println!("  Strength:   {:?}", signal.strength);

            if let Some(size) = signal.suggested_size {
                println!("  Position:   {:.1}% of capital", size * 100.0);
            }

            if let Some(ref probs) = signal.class_probabilities {
                println!("  Probabilities:");
                println!("    Buy:  {:.1}%", probs[0] * 100.0);
                println!("    Hold: {:.1}%", probs[1] * 100.0);
                println!("    Sell: {:.1}%", probs[2] * 100.0);
            }

            println!();
        }

        // Summary statistics
        let buy_count = signals
            .iter()
            .filter(|s| s.signal_type == SignalType::Buy)
            .count();
        let sell_count = signals
            .iter()
            .filter(|s| s.signal_type == SignalType::Sell)
            .count();
        let close_count = signals
            .iter()
            .filter(|s| s.signal_type == SignalType::Close)
            .count();

        let avg_confidence = if !signals.is_empty() {
            signals.iter().map(|s| s.confidence).sum::<f64>() / signals.len() as f64
        } else {
            0.0
        };

        println!("─────────────────────────────────────────────────────");
        println!("Summary:");
        println!("  Total signals:     {}", signals.len());
        println!("    Buy signals:     {}", buy_count);
        println!("    Sell signals:    {}", sell_count);
        println!("    Close signals:   {}", close_count);
        println!("  Avg confidence:    {:.1}%", avg_confidence * 100.0);
        println!("─────────────────────────────────────────────────────\n");
    }

    // -------------------------------------------------------------------------
    // Step 7: Usage Guide
    // -------------------------------------------------------------------------
    println!("═══════════════════════════════════════════════════════");
    println!("                   NEXT STEPS                          ");
    println!("═══════════════════════════════════════════════════════\n");

    println!("1️⃣  Train a Model:");
    println!("   cargo run --example production_training --release\n");

    println!("2️⃣  Load Trained Weights:");
    println!("   let model = DiffGafLstm::load_weights(");
    println!("       &config,");
    println!("       \"checkpoints/best_model.bin\",");
    println!("       &device");
    println!("   )?;\n");

    println!("3️⃣  Connect Live Data:");
    println!("   • WebSocket feed from exchange");
    println!("   • Real-time feature engineering");
    println!("   • Streaming inference → signals\n");

    println!("4️⃣  Execute Signals:");
    println!("   • Validate with backtesting service");
    println!("   • Risk management checks");
    println!("   • Forward to execution engine\n");

    println!("5️⃣  Monitor Performance:");
    println!("   • Track signal accuracy");
    println!("   • Measure PnL and Sharpe ratio");
    println!("   • Adjust thresholds dynamically\n");

    println!("═══════════════════════════════════════════════════════");
    println!("              Signal Generation Complete               ");
    println!("═══════════════════════════════════════════════════════\n");
}
