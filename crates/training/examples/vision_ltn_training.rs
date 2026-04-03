//! # Vision + LTN Training Example
//!
//! This example demonstrates end-to-end training with:
//! - DiffGAF + ViViT vision pipeline for market data encoding
//! - Logic Tensor Networks for neuro-symbolic constraints
//! - Prioritized replay buffer with SWR sampling
//! - AdamW optimizer with warmup+cosine scheduling
//! - Checkpointing and metrics tracking
//!
//! ## Requirements
//!
//! This example requires the `vision` feature, which is only available on macOS
//! due to dependencies on `font-kit` and `core-graphics`.
//!
//! ## Usage
//!
//! ```bash
//! # CPU training (macOS only)
//! cargo run --example vision_ltn_training --features vision
//!
//! # GPU training (CUDA, macOS only)
//! cargo run --example vision_ltn_training --features cuda,vision
//! ```
//!
//! ## Note
//!
//! This example will not compile on Linux or Windows unless the `vision` crate
//! dependencies are replaced with cross-platform alternatives.

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use logic::{DiffLTN, Grounding, RuleBuilder, TNormType};
use training::{
    LRSchedulerConfig, OptimizerConfig, ReplayBatch, ReplayBufferConfig, TrainingCallback,
    TrainingConfig, TrainingLoop, create_experience,
};
use vision::{VisionPipeline, VisionPipelineConfig};

/// Custom callback that prints detailed metrics
struct DetailedMetricsCallback;

impl TrainingCallback for DetailedMetricsCallback {
    fn on_step_end(&mut self, metrics: &training::StepMetrics) -> Result<()> {
        if metrics.step % 10 == 0 {
            println!("\n{}", "=".repeat(80));
            println!("{}", metrics.format());
            println!("  📊 Metrics Breakdown:");
            println!("     - Task Loss:     {:.6}", metrics.task_loss);
            println!("     - Logic Loss:    {:.6}", metrics.logic_loss);
            println!("     - Total Loss:    {:.6}", metrics.total_loss);
            println!("     - Learning Rate: {:.2e}", metrics.learning_rate);
            println!(
                "     - Grad Norm:     {:.6}",
                metrics.grad_norm.unwrap_or(0.0)
            );
            println!("     - Buffer Size:   {}", metrics.replay_buffer_size);
            println!("     - Duration:      {}ms", metrics.step_duration_ms);
            println!("{}", "=".repeat(80));
        }
        Ok(())
    }

    fn on_validation_end(&mut self, metrics: &training::ValidationMetrics) -> Result<()> {
        println!("\n{}", "🔍 VALIDATION ".to_string() + &"=".repeat(65));
        println!("{}", metrics.format());
        println!("{}", "=".repeat(80));
        Ok(())
    }

    fn on_checkpoint_saved(
        &mut self,
        path: &std::path::Path,
        metadata: &training::CheckpointMetadata,
    ) -> Result<()> {
        println!("\n💾 Checkpoint saved:");
        println!("   Path: {}", path.display());
        println!("   Step: {}", metadata.step);
        println!("   Loss: {:.6}", metadata.metrics.total_loss);
        Ok(())
    }

    fn on_early_stopping(&mut self, step: usize) -> Result<()> {
        println!("\n⚠️  Early stopping triggered at step {}", step);
        Ok(())
    }
}

/// Create a task loss function for predicting next market state
fn create_task_loss_fn(
    _vision_pipeline: VisionPipeline,
) -> impl FnMut(&ReplayBatch<Tensor, Tensor>, &VarMap, &Device) -> Result<Tensor> {
    move |batch: &ReplayBatch<Tensor, Tensor>, _var_map: &VarMap, device: &Device| {
        // Extract states from batch
        // In a real scenario, experiences would contain market data sequences
        let batch_size = batch.experiences.len();

        // For this example, we'll create dummy predictions
        // In production, you would:
        // 1. Extract state tensors from batch.experiences
        // 2. Run vision_pipeline.forward(state_sequences)
        // 3. Make predictions (e.g., next price movement)
        // 4. Compute MSE or classification loss

        // Dummy implementation:
        let predictions = Tensor::randn(0f32, 1.0, (batch_size, 10), device)?;
        let targets = Tensor::randn(0f32, 1.0, (batch_size, 10), device)?;

        // MSE loss
        let diff = predictions.sub(&targets)?;
        let squared = diff.sqr()?;
        let loss = squared.mean_all()?;

        Ok(loss)
    }
}

/// Create a logic loss function with trading constraints
fn create_logic_loss_fn()
-> impl FnMut(&ReplayBatch<Tensor, Tensor>, &VarMap, &Device) -> Result<Tensor> {
    move |batch: &ReplayBatch<Tensor, Tensor>, _var_map: &VarMap, device: &Device| {
        // Create LTN with Łukasiewicz t-norm (recommended for stability)
        let mut ltn = DiffLTN::new(TNormType::Lukasiewicz);

        // Define trading logic rules:
        // 1. High confidence → Action allowed
        // 2. Low volatility ∧ High confidence → Position valid
        // 3. Risk exceeded → No action
        // 4. Position valid → Stop loss active

        ltn.add_rule(
            RuleBuilder::new("confidence_action_rule")
                .implies("high_confidence", "action_allowed")
                .weight(2.0) // Higher weight = more important
                .build(),
        );

        ltn.add_rule(
            RuleBuilder::new("safe_position_rule")
                .and("low_volatility", "high_confidence")
                .weight(1.5)
                .build(),
        );

        ltn.add_rule(
            RuleBuilder::new("risk_constraint")
                .implies("risk_exceeded", "no_action")
                .weight(3.0) // Very important safety rule
                .build(),
        );

        ltn.add_rule(
            RuleBuilder::new("stop_loss_requirement")
                .implies("position_valid", "stop_loss_active")
                .weight(2.5)
                .build(),
        );

        // Create grounding from neural network outputs
        let batch_size = batch.experiences.len();

        // In production, these would come from:
        // - Vision pipeline embeddings → MLP → predicates
        // For now, we'll create dummy groundings

        let mut grounding = Grounding::new();

        // Dummy predicate values (would come from neural predictions)
        grounding.set(
            "high_confidence",
            Tensor::randn(0.7f32, 0.2, (batch_size,), device)?,
        );
        grounding.set(
            "action_allowed",
            Tensor::randn(0.6f32, 0.2, (batch_size,), device)?,
        );
        grounding.set(
            "low_volatility",
            Tensor::randn(0.5f32, 0.2, (batch_size,), device)?,
        );
        grounding.set(
            "position_valid",
            Tensor::randn(0.6f32, 0.2, (batch_size,), device)?,
        );
        grounding.set(
            "risk_exceeded",
            Tensor::randn(0.2f32, 0.15, (batch_size,), device)?,
        );
        grounding.set(
            "stop_loss_active",
            Tensor::randn(0.8f32, 0.15, (batch_size,), device)?,
        );
        grounding.set(
            "no_action",
            Tensor::randn(0.3f32, 0.15, (batch_size,), device)?,
        );

        // Compute satisfaction loss
        // Returns (1 - satisfaction) so we minimize to maximize rule satisfaction
        let logic_loss = ltn.satisfaction_loss(&grounding)?;

        Ok(logic_loss)
    }
}

fn main() -> Result<()> {
    println!("\n🚀 Project JANUS: Vision + LTN Training Example\n");

    // 1. Setup device (use CUDA if available)
    let device = Device::cuda_if_available(0)?;
    println!("📱 Device: {:?}", device);

    // 2. Configure vision pipeline
    let vision_config = VisionPipelineConfig::small();
    let (vision_pipeline, _vision_var_map) = VisionPipeline::new(vision_config.clone(), &device)?;
    println!("👁️  Vision pipeline created:");
    println!(
        "   - Image size: {}x{}",
        vision_config.gaf.output_size, vision_config.gaf.output_size
    );
    println!("   - Num frames: {}", vision_config.num_frames);
    println!("   - Embed dim: {}", vision_config.vivit.embed_dim);

    // 3. Configure training
    let train_config = TrainingConfig {
        num_steps: 100,
        batch_size: 16,
        logic_weight: 0.5,
        checkpoint_every: 50,
        validate_every: 20,
        log_every: 5,
        checkpoint_dir: std::path::PathBuf::from("checkpoints/vision_ltn_example"),
        max_checkpoints: 5,
        early_stopping_patience: Some(5),
        device: device.clone(),
        ..Default::default()
    };

    println!("\n🎯 Training configuration:");
    println!("   - Steps: {}", train_config.num_steps);
    println!("   - Batch size: {}", train_config.batch_size);
    println!("   - Logic weight: {}", train_config.logic_weight);
    println!(
        "   - Checkpoint dir: {}",
        train_config.checkpoint_dir.display()
    );

    // 4. Configure optimizer (AdamW with weight decay)
    let opt_config = OptimizerConfig::adamw()
        .learning_rate(1e-4)
        .weight_decay(0.01)
        .epsilon(1e-8)
        .build();

    println!("\n⚙️  Optimizer: AdamW");
    println!("   - Learning rate: {:.2e}", opt_config.learning_rate);
    println!("   - Weight decay: {}", opt_config.weight_decay);

    // 5. Configure learning rate scheduler (warmup + cosine)
    let sched_config = LRSchedulerConfig::warmup_cosine()
        .warmup_steps(10)
        .total_steps(train_config.num_steps)
        .min_lr(1e-5)
        .build();

    println!("\n📈 LR Scheduler: Warmup + Cosine");
    println!("   - Warmup steps: 10");
    println!("   - Total steps: {}", train_config.num_steps);

    // 6. Configure replay buffer
    let replay_config = ReplayBufferConfig {
        capacity: 10_000,
        alpha: 0.6, // Prioritization exponent
        beta: 0.4,  // Importance sampling
        beta_anneal_steps: train_config.num_steps,
        priority_epsilon: 1e-6,
        min_priority: 0.01,
    };

    println!("\n💾 Replay Buffer:");
    println!("   - Capacity: {}", replay_config.capacity);
    println!("   - Alpha (prioritization): {}", replay_config.alpha);
    println!("   - Beta (IS): {}", replay_config.beta);

    // 7. Create training loop
    let mut training_loop =
        TrainingLoop::new(train_config, opt_config, sched_config, replay_config)?;

    // 8. Add callbacks
    training_loop.add_callback(Box::new(DetailedMetricsCallback));

    println!("\n📚 Populating replay buffer with synthetic experiences...");

    // 9. Populate replay buffer with dummy experiences
    for i in 0..200 {
        let state = Tensor::randn(0f32, 1.0, (128, 5), &device)?; // 128 timesteps, 5 features (OHLCV)
        let action = Tensor::randn(0f32, 1.0, (10,), &device)?; // 10-dim action space
        let reward = ((i as f64 * 0.01) % 1.0) as f32; // Dummy reward
        let next_state = Tensor::randn(0f32, 1.0, (128, 5), &device)?;
        let done = i % 50 == 0;

        let experience = create_experience(state, action, reward, next_state, done);
        training_loop.add_experience(experience);
    }

    println!("   ✓ Added 200 synthetic experiences");

    // 10. Create loss functions
    let task_loss_fn = create_task_loss_fn(vision_pipeline);
    let logic_loss_fn = create_logic_loss_fn();

    // 11. Run training
    println!("\n🏋️  Starting training loop...\n");

    let final_metrics = training_loop.run(
        task_loss_fn,
        logic_loss_fn,
        None, // No validation data for this example
        None::<fn(&[training::Experience<Tensor, Tensor>], &VarMap, &Device) -> Result<Tensor>>,
        None::<fn(&[training::Experience<Tensor, Tensor>], &VarMap, &Device) -> Result<Tensor>>,
    )?;

    // 12. Print final results
    println!("\n\n✅ Training completed!");
    println!("{}", "=".repeat(80));
    println!("📊 Final Metrics:");
    println!("   - Total Loss:    {:.6}", final_metrics.total_loss);
    println!("   - Task Loss:     {:.6}", final_metrics.task_loss);
    println!("   - Logic Loss:    {:.6}", final_metrics.logic_loss);
    println!("   - Learning Rate: {:.2e}", final_metrics.learning_rate);
    println!("   - Steps:         {}", final_metrics.step);
    println!("{}", "=".repeat(80));

    println!("\n💡 Next steps:");
    println!("   1. Load checkpoints for inference");
    println!("   2. Deploy to Forward service for live trading");
    println!("   3. Monitor with Prometheus/Grafana");
    println!("   4. Enable CUDA for GPU training (--features cuda)");
    println!("   5. Implement Wake/Sleep scheduling for VRAM management");

    Ok(())
}
