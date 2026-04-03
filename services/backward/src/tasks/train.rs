//! Training task — gradient-based DQN weight updates with target-network
//! soft-updates.
//!
//! This task is responsible for:
//! 1. Sampling batches from the Prioritized Experience Replay (PER) buffer
//! 2. Computing Double-DQN TD targets via the online + target Q-networks
//! 3. Performing an importance-sampling-weighted gradient step on the online
//!    network (Adam optimiser, real backpropagation)
//! 4. Polyak (τ-based) soft-updating the target network towards the online
//!    network
//! 5. Updating replay-buffer priorities with fresh TD errors
//! 6. Checkpointing the trained model as an inference-ready [`LstmPredictor`]
//!
//! ## ML Integration
//!
//! This module uses [`janus_ml::dqn::DqnOnlineModel`] for gradient-based
//! training and [`janus_ml::dqn::compute_double_dqn_targets`] for stable
//! target computation, replacing the earlier inference-only forward-pass
//! pipeline.

use crate::worker::TrainJob;
use anyhow::{Context, Result};
use common::{Action, ActionType, Experience, State, StateMetadata};
use janus_core::checkpoint_notify::{CheckpointNotification, CheckpointNotifier};
use memory::SumTree;
use tracing::{debug, info, warn};

use janus_ml::backend::{BackendDevice, CpuBackend};
use janus_ml::dqn::{DqnOnlineModel, DqnStepResult, compute_double_dqn_targets};
use janus_ml::models::trainable::TrainableLstmConfig;
use janus_ml::models::{LstmConfig, LstmPredictor};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for a training run.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Maximum capacity of the replay buffer.
    pub replay_capacity: usize,
    /// PER alpha parameter — controls how much prioritization is used (0 = uniform, 1 = full).
    pub per_alpha: f64,
    /// PER beta parameter — importance-sampling correction (annealed from initial to 1.0).
    pub per_beta: f64,
    /// Learning rate for Adam weight updates.
    pub learning_rate: f64,
    /// L2 weight decay for Adam.
    pub weight_decay: f64,
    /// Discount factor (gamma) for TD target computation.
    pub gamma: f64,
    /// Number of steps between model checkpoint saves.
    pub checkpoint_interval: usize,
    /// Target network soft-update coefficient (tau).
    pub tau: f64,

    // ── ML model configuration ────────────────────────────────────────
    /// Input feature dimension for the Q-network LSTM.
    pub model_input_size: usize,
    /// Hidden state dimension for the Q-network LSTM.
    pub model_hidden_size: usize,
    /// Output dimension (number of discrete actions: Buy, Sell, Hold, Close).
    pub model_output_size: usize,
    /// Number of LSTM layers.
    pub model_num_layers: usize,
    /// Dropout probability during training.
    pub model_dropout: f64,
    /// Directory for saving model checkpoints.
    pub checkpoint_dir: String,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            replay_capacity: 100_000,
            per_alpha: 0.6,
            per_beta: 0.4,
            learning_rate: 3e-4,
            weight_decay: 1e-5,
            gamma: 0.99,
            checkpoint_interval: 1000,
            tau: 0.005,
            // Default model dimensions — should be overridden from service config
            model_input_size: 9,
            model_hidden_size: 64,
            model_output_size: 4, // Buy, Sell, Hold, Close
            model_num_layers: 2,
            model_dropout: 0.2,
            checkpoint_dir: "checkpoints/backward".to_string(),
        }
    }
}

// ─── Metrics ──────────────────────────────────────────────────────────────────

/// Outcome metrics from a single training step.
#[derive(Debug, Clone, Default)]
pub struct TrainStepMetrics {
    /// Mean TD error across the batch.
    pub mean_td_error: f64,
    /// Mean loss value (IS-weighted MSE).
    pub mean_loss: f64,
    /// Max absolute TD error in the batch (useful for monitoring).
    pub max_td_error: f64,
    /// Number of experiences sampled.
    pub batch_size: usize,
    /// Current replay buffer utilization (0.0–1.0).
    pub buffer_utilization: f64,
    /// Mean Q-value predicted by the online network for the taken actions.
    pub mean_q_value: f64,
    /// Mean TD target value.
    pub mean_target_q_value: f64,
    /// Whether a gradient step was actually performed.
    pub gradient_step_performed: bool,
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Create a placeholder experience for seeding the replay buffer.
fn create_placeholder_experience() -> Experience {
    let metadata = StateMetadata::new("PLACEHOLDER".to_string());
    let state = State::from_flat_gaf(vec![0.0_f32; 9], vec![], metadata.clone());
    let next_state = State::from_flat_gaf(vec![0.0_f32; 9], vec![], metadata);
    let action = Action::new(ActionType::Hold, "PLACEHOLDER".to_string(), 0.0);
    Experience::new(state, action, 0.0, next_state, false)
}

/// Build an [`LstmConfig`] (inference) from a [`TrainingConfig`].
fn build_inference_config(config: &TrainingConfig) -> LstmConfig {
    LstmConfig::new(
        config.model_input_size,
        config.model_hidden_size,
        config.model_output_size,
    )
    .with_num_layers(config.model_num_layers)
    .with_dropout(config.model_dropout)
}

/// Build a [`TrainableLstmConfig`] from a [`TrainingConfig`].
fn build_trainable_config(config: &TrainingConfig) -> TrainableLstmConfig {
    TrainableLstmConfig::new(
        config.model_input_size,
        config.model_hidden_size,
        config.model_output_size,
    )
    .with_num_layers(config.model_num_layers)
    .with_dropout(config.model_dropout)
}

/// Convert an experience's GAF features into a padded/truncated `f32` vector
/// of length `input_size`, suitable for feeding into the model.
fn state_to_features(state: &State, input_size: usize) -> Vec<f32> {
    let mut features: Vec<f32> = state.gaf_features_flat.clone();
    features.resize(input_size, 0.0);
    features
}

/// Map an [`ActionType`] to a Q-value output index.
fn action_to_index(action_type: &ActionType) -> usize {
    match action_type {
        ActionType::Buy => 0,
        ActionType::Sell => 1,
        ActionType::Hold => 2,
        ActionType::Close => 3,
    }
}

/// Run the LSTM forward pass and return Q-values for a batch of states.
///
/// Each state is independently fed through the model as a `[1, 1, input_size]`
/// tensor.  Returns a `Vec<Vec<f64>>` where each inner vec has `output_size`
/// Q-values.
fn batch_predict(
    model: &LstmPredictor<CpuBackend>,
    states: &[&State],
    input_size: usize,
) -> Vec<Vec<f64>> {
    use janus_ml::tensor::{Backend, Tensor, TensorData};
    let device = <CpuBackend as Backend>::Device::default();

    states
        .iter()
        .map(|state| {
            let features = state_to_features(state, input_size);
            let tensor = Tensor::<CpuBackend, 3>::from_data(
                TensorData::new(features, [1, 1, input_size]),
                &device,
            );
            match model.forward(tensor) {
                Ok(output) => {
                    let vals: Vec<f32> = output.to_data().to_vec().unwrap();
                    vals.into_iter().map(|v| v as f64).collect()
                }
                Err(e) => {
                    warn!("Model forward pass failed: {}", e);
                    vec![0.0_f64; model.config().output_size]
                }
            }
        })
        .collect()
}

fn replay_buffer_utilization(current: usize, capacity: usize) -> f64 {
    if capacity == 0 {
        0.0
    } else {
        current as f64 / capacity as f64
    }
}

/// Save an inference model checkpoint to disk.
fn save_checkpoint(
    model: &LstmPredictor<CpuBackend>,
    checkpoint_dir: &std::path::Path,
) -> Result<()> {
    std::fs::create_dir_all(checkpoint_dir)?;
    let model_path = checkpoint_dir.join("latest_model.bin");
    model
        .save(&model_path)
        .map_err(|e| anyhow::anyhow!("Failed to save model checkpoint: {}", e))?;
    info!(path = %model_path.display(), "Model checkpoint saved");
    Ok(())
}

// ─── Main training entry point ────────────────────────────────────────────────

/// Handle a training job dispatched by the scheduler / job queue.
///
/// This function orchestrates a complete training step:
/// 1. Initialise the online Q-network (trainable, with gradients) and the
///    target Q-network (inference-only, for stable TD targets)
/// 2. Initialise (or re-attach to) the PER buffer
/// 3. Sample a prioritized batch
/// 4. Compute Double-DQN TD targets via online+target forward passes
/// 5. Perform an IS-weighted gradient step on the online network
/// 6. Soft-update the target network with Polyak averaging (τ)
/// 7. Update priorities in the replay buffer
/// 8. Checkpoint the trained model as an inference predictor
#[allow(dead_code)]
pub async fn handle_training(job: TrainJob, notifier: Option<&CheckpointNotifier>) -> Result<()> {
    let config = TrainingConfig::default();

    info!(
        batch_size = job.batch_size,
        replay_capacity = config.replay_capacity,
        alpha = config.per_alpha,
        beta = config.per_beta,
        lr = config.learning_rate,
        tau = config.tau,
        gamma = config.gamma,
        model_input = config.model_input_size,
        model_hidden = config.model_hidden_size,
        model_output = config.model_output_size,
        "Starting gradient-based training job"
    );

    // ── 1. Build Q-networks ───────────────────────────────────────────────
    let trainable_cfg = build_trainable_config(&config);
    let inference_cfg = build_inference_config(&config);

    // Online network — the network we're training (with autodiff + Adam)
    let mut online = DqnOnlineModel::new(trainable_cfg, config.learning_rate, config.weight_decay)
        .context("Failed to create online DQN model")?;

    // Target network — a lagging copy used for stable TD targets.
    // In production this would be loaded from the last checkpoint; for now
    // it is initialised fresh and immediately soft-updated from the online
    // model so they start from the same weights.
    let mut target_model = LstmPredictor::new(inference_cfg, BackendDevice::cpu());
    online.soft_update_target(&mut target_model, 1.0); // full copy to start

    info!(
        online_steps = online.step_count,
        "Q-networks initialised (online=trainable, target=inference)"
    );

    // ── 2. Initialise PER buffer ──────────────────────────────────────────
    let mut replay_buffer = SumTree::new(config.replay_capacity, config.per_alpha, config.per_beta);

    let buffer_size = replay_buffer.len();
    if buffer_size < job.batch_size {
        warn!(
            buffer_size = buffer_size,
            required = job.batch_size,
            "Replay buffer has fewer experiences than batch size — seeding with placeholders"
        );
        let needed = job.batch_size.saturating_sub(buffer_size);
        for i in 0..needed {
            let exp = create_placeholder_experience();
            replay_buffer.add(exp, 1.0);
            debug!(index = i, "Seeded placeholder experience");
        }
    }

    // ── 3. Sample a prioritized batch ─────────────────────────────────────
    let samples = replay_buffer
        .sample(job.batch_size)
        .map_err(|e| anyhow::anyhow!("Failed to sample batch from PER buffer: {}", e))
        .context("Sampling from replay buffer")?;

    info!(
        sampled = samples.len(),
        "Sampled prioritized batch from replay buffer"
    );

    let mut indices = Vec::with_capacity(samples.len());
    let mut is_weights = Vec::with_capacity(samples.len());
    let mut experiences = Vec::with_capacity(samples.len());

    for (exp, idx, weight) in &samples {
        experiences.push(exp.clone());
        indices.push(*idx);
        is_weights.push(*weight);
    }

    // Normalise IS weights so the max weight is 1.0
    let max_weight = is_weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_weight > 0.0 {
        for w in &mut is_weights {
            *w /= max_weight;
        }
    }

    // ── 4. Compute Double-DQN TD targets ──────────────────────────────────
    let next_states: Vec<Vec<f32>> = experiences
        .iter()
        .map(|e| state_to_features(&e.next_state, config.model_input_size))
        .collect();
    let rewards: Vec<f64> = experiences.iter().map(|e| e.reward as f64).collect();
    let dones: Vec<bool> = experiences.iter().map(|e| e.done).collect();

    let td_targets = compute_double_dqn_targets(
        &online,
        &target_model,
        &next_states,
        &rewards,
        &dones,
        config.gamma,
        config.model_input_size,
        1, // seq_len
    );

    let mean_target = td_targets.iter().sum::<f64>() / td_targets.len() as f64;

    // ── 5. Gradient step on the online network ────────────────────────────
    let states: Vec<Vec<f32>> = experiences
        .iter()
        .map(|e| state_to_features(&e.state, config.model_input_size))
        .collect();
    let actions: Vec<usize> = experiences
        .iter()
        .map(|e| action_to_index(&e.action.action_type))
        .collect();

    let step_result: DqnStepResult = online
        .gradient_step(
            &states,
            &actions,
            &td_targets,
            &is_weights,
            config.model_input_size,
            1, // seq_len
        )
        .context("Gradient step failed")?;

    info!(
        loss = format!("{:.6}", step_result.loss),
        mean_q = format!("{:.4}", step_result.mean_q),
        max_td = format!("{:.6}", step_result.max_td_error),
        mean_target = format!("{:.4}", mean_target),
        step = online.step_count,
        "Gradient step completed"
    );

    // ── 6. Soft-update the target network ─────────────────────────────────
    online.soft_update_target(&mut target_model, config.tau);

    debug!(
        tau = config.tau,
        "Target network soft-updated (Polyak averaging)"
    );

    // ── 7. Update priorities in the replay buffer ─────────────────────────
    //
    // Recompute per-sample TD errors using the *updated* online network so
    // that priorities reflect the latest model state.
    let epsilon = 1e-6_f64;
    let td_errors = compute_td_errors(&online, &target_model, &experiences, &config);

    for (i, &idx) in indices.iter().enumerate() {
        let new_priority = td_errors[i].abs() + epsilon;
        replay_buffer.update_priority(idx, new_priority);
    }

    info!(updated = indices.len(), "Updated replay buffer priorities");

    // ── 8. Checkpoint ─────────────────────────────────────────────────────
    let checkpoint_dir = std::path::Path::new(&config.checkpoint_dir);
    match online.to_inference(BackendDevice::cpu()) {
        Ok(predictor) => {
            if let Err(e) = save_checkpoint(&predictor, checkpoint_dir) {
                warn!(error = %e, "Failed to save model checkpoint (non-fatal)");
            } else {
                debug!(dir = %checkpoint_dir.display(), "Model checkpoint saved");

                // ── 8b. Notify downstream services of new checkpoint ──
                if let Some(n) = notifier {
                    let model_path = checkpoint_dir
                        .join("latest_model.bin")
                        .to_string_lossy()
                        .to_string();

                    let notification = CheckpointNotification::new(&model_path, "lstm_dqn_v1")
                        .with_version(online.step_count as u64)
                        .with_training_step(online.step_count as u64)
                        .with_metadata("loss", format!("{:.6}", step_result.loss))
                        .with_metadata("mean_q", format!("{:.4}", step_result.mean_q));

                    if let Err(e) = n.publish(&notification).await {
                        warn!(
                            error = %e,
                            "Failed to publish checkpoint notification (non-fatal)"
                        );
                    }
                }
            }
        }
        Err(e) => {
            warn!(error = %e, "Failed to convert online model to inference for checkpoint");
        }
    }

    // ── Aggregate metrics ─────────────────────────────────────────────────
    let mean_td_error = td_errors.iter().map(|e| e.abs()).sum::<f64>() / td_errors.len() as f64;
    let max_td_error_abs = td_errors.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);

    let metrics = TrainStepMetrics {
        mean_td_error,
        mean_loss: step_result.loss,
        max_td_error: max_td_error_abs,
        batch_size: experiences.len(),
        buffer_utilization: replay_buffer_utilization(replay_buffer.len(), config.replay_capacity),
        mean_q_value: step_result.mean_q,
        mean_target_q_value: mean_target,
        gradient_step_performed: true,
    };

    info!(
        mean_loss = format!("{:.6}", metrics.mean_loss),
        mean_td = format!("{:.6}", metrics.mean_td_error),
        max_td = format!("{:.6}", metrics.max_td_error),
        mean_q = format!("{:.4}", metrics.mean_q_value),
        mean_target_q = format!("{:.4}", metrics.mean_target_q_value),
        batch = metrics.batch_size,
        grad_step = metrics.gradient_step_performed,
        "Training job completed successfully"
    );

    Ok(())
}

// ─── Core computation (kept for priority-update recomputation) ─────────────

/// Compute per-experience TD errors using the online and target Q-networks.
///
/// For each experience `(s, a, r, s', done)`:
///
///   `td_error = r + γ * Q_target(s', argmax_a Q_online(s', a)) - Q_online(s, a)`
///
/// When `done == true` the target is simply `r`.
///
/// This uses the *inference* copies of both models (no gradients) so it can
/// be called after the gradient step to recompute priorities.
fn compute_td_errors(
    online: &DqnOnlineModel,
    target: &LstmPredictor<CpuBackend>,
    experiences: &[Experience],
    config: &TrainingConfig,
) -> Vec<f64> {
    let input_size = config.model_input_size;

    // Convert the online model to an inference predictor for forward passes
    let online_predictor = match online.to_inference(BackendDevice::cpu()) {
        Ok(p) => p,
        Err(e) => {
            warn!(
                "Failed to convert online to inference for TD-error recomputation: {}",
                e
            );
            // Fall back to zero TD errors
            return vec![0.0; experiences.len()];
        }
    };

    // Collect state / next-state references
    let states: Vec<&State> = experiences.iter().map(|e| &e.state).collect();
    let next_states: Vec<&State> = experiences.iter().map(|e| &e.next_state).collect();

    // Online Q(s, ·) for all experiences
    let q_values = batch_predict(&online_predictor, &states, input_size);

    // Online Q(s', ·) — used for argmax (Double DQN)
    let next_q_online = batch_predict(&online_predictor, &next_states, input_size);

    // Target Q(s', ·) — evaluated at the online-argmax action
    let next_q_target = batch_predict(target, &next_states, input_size);

    experiences
        .iter()
        .enumerate()
        .map(|(i, exp)| {
            let action_idx = action_to_index(&exp.action.action_type);
            let q_sa = q_values[i].get(action_idx).copied().unwrap_or(0.0);

            if exp.done {
                (exp.reward as f64) - q_sa
            } else {
                // Double DQN: pick best action according to *online*, evaluate
                // with *target*
                let best_next_action = next_q_online[i]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let target_q = next_q_target[i]
                    .get(best_next_action)
                    .copied()
                    .unwrap_or(0.0);

                (exp.reward as f64) + config.gamma * target_q - q_sa
            }
        })
        .collect()
}

/// Compute training-step metrics using real model forward passes (inference
/// only — no gradients).
///
/// This is retained for diagnostic use and backwards compatibility.  The
/// actual training is performed by [`DqnOnlineModel::gradient_step`].
#[allow(dead_code)]
fn compute_training_step(
    online: &LstmPredictor<CpuBackend>,
    target: &LstmPredictor<CpuBackend>,
    experiences: &[Experience],
    is_weights: &[f64],
    config: &TrainingConfig,
) -> Result<TrainStepMetrics> {
    let batch_size = experiences.len();
    if batch_size == 0 {
        anyhow::bail!("Empty batch — nothing to train on");
    }

    let input_size = config.model_input_size;

    // Q(s, a) for the taken actions
    let states: Vec<&State> = experiences.iter().map(|e| &e.state).collect();
    let next_states: Vec<&State> = experiences.iter().map(|e| &e.next_state).collect();

    let q_values = batch_predict(online, &states, input_size);
    let next_q_online = batch_predict(online, &next_states, input_size);
    let next_q_target = batch_predict(target, &next_states, input_size);

    let mut td_errors = Vec::with_capacity(batch_size);
    let mut q_taken = Vec::with_capacity(batch_size);
    let mut q_target_vals = Vec::with_capacity(batch_size);

    for (i, exp) in experiences.iter().enumerate() {
        let action_idx = action_to_index(&exp.action.action_type);
        let q_sa = q_values[i].get(action_idx).copied().unwrap_or(0.0);
        q_taken.push(q_sa);

        if exp.done {
            td_errors.push((exp.reward as f64) - q_sa);
            q_target_vals.push(exp.reward as f64);
        } else {
            let best_next = next_q_online[i]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            let tgt = next_q_target[i].get(best_next).copied().unwrap_or(0.0);
            let td_target = (exp.reward as f64) + config.gamma * tgt;
            td_errors.push(td_target - q_sa);
            q_target_vals.push(td_target);
        }
    }

    // IS-weighted squared TD error
    let weighted_losses: Vec<f64> = td_errors
        .iter()
        .zip(is_weights.iter())
        .map(|(&td, &w)| w * td * td)
        .collect();

    let mean_loss = weighted_losses.iter().sum::<f64>() / batch_size as f64;
    let mean_td_error = td_errors.iter().map(|e| e.abs()).sum::<f64>() / batch_size as f64;
    let max_td_error = td_errors.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);

    let mean_q_value = q_taken.iter().sum::<f64>() / batch_size as f64;
    let mean_target_q_value = q_target_vals.iter().sum::<f64>() / batch_size as f64;

    Ok(TrainStepMetrics {
        mean_td_error,
        mean_loss,
        max_td_error,
        batch_size,
        buffer_utilization: replay_buffer_utilization(batch_size, config.replay_capacity),
        mean_q_value,
        mean_target_q_value,
        gradient_step_performed: false,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_defaults() {
        let cfg = TrainingConfig::default();
        assert_eq!(cfg.replay_capacity, 100_000);
        assert!((cfg.per_alpha - 0.6).abs() < f64::EPSILON);
        assert!((cfg.per_beta - 0.4).abs() < f64::EPSILON);
        assert!((cfg.gamma - 0.99).abs() < f64::EPSILON);
        assert!((cfg.tau - 0.005).abs() < f64::EPSILON);
        assert!((cfg.learning_rate - 3e-4).abs() < 1e-10);
        assert_eq!(cfg.model_input_size, 9);
        assert_eq!(cfg.model_output_size, 4);
    }

    #[test]
    fn test_placeholder_experience_creation() {
        let exp = create_placeholder_experience();
        assert_eq!(exp.reward, 0.0);
        assert!(!exp.done);
        assert_eq!(exp.action.action_type, ActionType::Hold);
    }

    #[test]
    fn test_action_to_index_mapping() {
        assert_eq!(action_to_index(&ActionType::Buy), 0);
        assert_eq!(action_to_index(&ActionType::Sell), 1);
        assert_eq!(action_to_index(&ActionType::Hold), 2);
        assert_eq!(action_to_index(&ActionType::Close), 3);
    }

    #[test]
    fn test_state_to_features_padding() {
        let metadata = StateMetadata::new("TEST".to_string());
        let state = State::from_flat_gaf(vec![1.0, 2.0, 3.0], vec![], metadata);

        let features = state_to_features(&state, 9);
        assert_eq!(features.len(), 9);
        assert_eq!(features[0], 1.0);
        assert_eq!(features[2], 3.0);
        // Padded positions should be 0.0
        assert_eq!(features[3], 0.0);
        assert_eq!(features[8], 0.0);
    }

    #[test]
    fn test_state_to_features_truncation() {
        let metadata = StateMetadata::new("TEST".to_string());
        let state = State::from_flat_gaf(vec![1.0; 20], vec![], metadata);

        let features = state_to_features(&state, 5);
        assert_eq!(features.len(), 5);
    }

    #[test]
    fn test_build_trainable_config() {
        let cfg = TrainingConfig::default();
        let tcfg = build_trainable_config(&cfg);
        assert_eq!(tcfg.input_size, 9);
        assert_eq!(tcfg.hidden_size, 64);
        assert_eq!(tcfg.output_size, 4);
        assert_eq!(tcfg.num_layers, 2);
    }

    #[test]
    fn test_build_inference_config() {
        let cfg = TrainingConfig::default();
        let icfg = build_inference_config(&cfg);
        assert_eq!(icfg.input_size, 9);
        assert_eq!(icfg.hidden_size, 64);
        assert_eq!(icfg.output_size, 4);
        assert_eq!(icfg.num_layers, 2);
    }

    #[test]
    fn test_batch_predict_returns_correct_shape() {
        let config = TrainingConfig::default();
        let model_cfg = build_inference_config(&config);
        let bdev = BackendDevice::cpu();
        let model = LstmPredictor::new(model_cfg, bdev);

        let metadata = StateMetadata::new("TEST".to_string());
        let state = State::from_flat_gaf(vec![0.1_f32; 9], vec![], metadata);

        let predictions = batch_predict(&model, &[&state], config.model_input_size);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].len(), config.model_output_size);
    }

    #[test]
    fn test_dqn_online_model_gradient_step() {
        let config = TrainingConfig::default();
        let tcfg = build_trainable_config(&config);

        let mut online = DqnOnlineModel::new(tcfg, config.learning_rate, config.weight_decay)
            .expect("Failed to create DQN online model");

        let states = vec![vec![0.1_f32; config.model_input_size]; 4];
        let actions = vec![0, 1, 2, 3];
        let td_targets = vec![1.0, -0.5, 0.3, 0.0];
        let is_weights = vec![1.0; 4];

        let result = online.gradient_step(
            &states,
            &actions,
            &td_targets,
            &is_weights,
            config.model_input_size,
            1,
        );
        assert!(result.is_ok(), "Gradient step failed: {:?}", result.err());

        let metrics = result.unwrap();
        assert!(metrics.loss.is_finite());
        assert!(metrics.mean_q.is_finite());
        assert_eq!(online.step_count, 1);
    }

    #[test]
    fn test_soft_update_target_network() {
        let config = TrainingConfig::default();
        let tcfg = build_trainable_config(&config);
        let icfg = build_inference_config(&config);

        let online = DqnOnlineModel::new(tcfg, config.learning_rate, config.weight_decay)
            .expect("Failed to create DQN online model");

        let mut target = LstmPredictor::new(icfg, BackendDevice::cpu());
        let weights_before = target.extract_weights_pub();

        // Full copy (tau=1.0)
        online.soft_update_target(&mut target, 1.0);
        let weights_after = target.extract_weights_pub();

        // Weights should have changed (online and target were initialised
        // independently with random weights)
        let any_changed = weights_before
            .iter()
            .zip(weights_after.iter())
            .any(|(a, b)| {
                a.data
                    .iter()
                    .zip(b.data.iter())
                    .any(|(x, y)| (x - y).abs() > 1e-9)
            });
        assert!(
            any_changed,
            "Target weights should change after soft update with tau=1.0"
        );
    }

    #[test]
    fn test_compute_td_errors_non_terminal() {
        let config = TrainingConfig::default();
        let tcfg = build_trainable_config(&config);
        let icfg = build_inference_config(&config);

        let online = DqnOnlineModel::new(tcfg, config.learning_rate, config.weight_decay)
            .expect("Failed to create DQN online model");
        let target = LstmPredictor::new(icfg, BackendDevice::cpu());

        let experiences: Vec<Experience> =
            (0..4).map(|_| create_placeholder_experience()).collect();

        let td_errors = compute_td_errors(&online, &target, &experiences, &config);
        assert_eq!(td_errors.len(), 4);
        // TD errors should be finite
        for td in &td_errors {
            assert!(td.is_finite(), "TD error should be finite: {}", td);
        }
    }

    #[test]
    fn test_compute_td_errors_terminal() {
        let config = TrainingConfig::default();
        let tcfg = build_trainable_config(&config);
        let icfg = build_inference_config(&config);

        let online = DqnOnlineModel::new(tcfg, config.learning_rate, config.weight_decay)
            .expect("Failed to create DQN online model");
        let target = LstmPredictor::new(icfg, BackendDevice::cpu());

        let metadata = StateMetadata::new("TEST".to_string());
        let state = State::from_flat_gaf(vec![0.0_f32; 9], vec![], metadata.clone());
        let next_state = State::from_flat_gaf(vec![0.0_f32; 9], vec![], metadata);
        let action = Action::new(ActionType::Buy, "TEST".to_string(), 1.0);
        let exp = Experience::new(state, action, 5.0, next_state, true);

        let td_errors = compute_td_errors(&online, &target, &[exp], &config);
        assert_eq!(td_errors.len(), 1);
        // For terminal state: td = reward - Q(s,a)
        assert!(td_errors[0].is_finite());
    }

    #[test]
    fn test_compute_training_step_returns_valid_metrics() {
        let config = TrainingConfig::default();
        let model_cfg = build_inference_config(&config);
        let bdev = BackendDevice::cpu();
        let online = LstmPredictor::new(model_cfg.clone(), bdev.clone());
        let target = LstmPredictor::new(model_cfg, bdev);

        let experiences: Vec<Experience> =
            (0..8).map(|_| create_placeholder_experience()).collect();
        let is_weights = vec![1.0; 8];

        let metrics =
            compute_training_step(&online, &target, &experiences, &is_weights, &config).unwrap();

        assert_eq!(metrics.batch_size, 8);
        assert!(metrics.mean_loss >= 0.0);
        assert!(metrics.mean_td_error >= 0.0);
        assert!(metrics.max_td_error >= 0.0);
        assert!(metrics.mean_q_value.is_finite());
        assert!(metrics.mean_target_q_value.is_finite());
        assert!(!metrics.gradient_step_performed);
    }

    #[test]
    fn test_priority_update_round_trip() {
        let mut buffer = SumTree::new(100, 0.6, 0.4);
        for _ in 0..10 {
            let exp = create_placeholder_experience();
            buffer.add(exp, 1.0);
        }
        assert_eq!(buffer.len(), 10);

        let samples = buffer.sample(5).unwrap();
        for (_, idx, _) in &samples {
            buffer.update_priority(*idx, 2.0);
        }

        assert_eq!(buffer.len(), 10);
    }

    #[tokio::test]
    async fn test_handle_training_runs_without_error() {
        let job = TrainJob::default();
        let result = handle_training(job, None).await;
        assert!(result.is_ok(), "Training failed: {:?}", result.err());
    }

    #[test]
    fn test_replay_buffer_utilization() {
        assert!((replay_buffer_utilization(50, 100) - 0.5).abs() < f64::EPSILON);
        assert!((replay_buffer_utilization(0, 100) - 0.0).abs() < f64::EPSILON);
        assert!((replay_buffer_utilization(0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_save_checkpoint_creates_file() {
        let config = TrainingConfig::default();
        let model_cfg = build_inference_config(&config);
        let bdev = BackendDevice::cpu();
        let model = LstmPredictor::new(model_cfg, bdev);

        let tmp_dir = std::env::temp_dir().join("janus_train_test_checkpoint_v2");
        let result = save_checkpoint(&model, &tmp_dir);
        assert!(result.is_ok(), "Checkpoint save failed: {:?}", result.err());

        let model_path = tmp_dir.join("latest_model.bin");
        assert!(model_path.exists(), "Checkpoint file should exist");

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_online_to_inference_roundtrip() {
        let config = TrainingConfig::default();
        let tcfg = build_trainable_config(&config);

        let online = DqnOnlineModel::new(tcfg, config.learning_rate, config.weight_decay)
            .expect("Failed to create DQN online model");

        let predictor = online.to_inference(BackendDevice::cpu());
        assert!(
            predictor.is_ok(),
            "to_inference failed: {:?}",
            predictor.err()
        );

        let predictor = predictor.unwrap();
        assert_eq!(predictor.config().input_size, config.model_input_size);
        assert_eq!(predictor.config().hidden_size, config.model_hidden_size);
        assert_eq!(predictor.config().output_size, config.model_output_size);
    }

    #[test]
    fn test_gradient_step_changes_online_weights() {
        let config = TrainingConfig::default();
        let tcfg = build_trainable_config(&config);

        let mut online =
            DqnOnlineModel::new(tcfg, 1e-2, 0.0).expect("Failed to create DQN online model");

        let w0 = online.online_weights();

        // Run several steps with large targets to force weight changes
        for _ in 0..5 {
            let states = vec![vec![1.0_f32; config.model_input_size]; 8];
            let actions = vec![0; 8];
            let td_targets = vec![10.0; 8];
            let is_weights = vec![1.0; 8];
            let _ = online.gradient_step(
                &states,
                &actions,
                &td_targets,
                &is_weights,
                config.model_input_size,
                1,
            );
        }

        let w1 = online.online_weights();

        let any_changed = w0.iter().zip(w1.iter()).any(|(a, b)| {
            a.data
                .iter()
                .zip(b.data.iter())
                .any(|(x, y)| (x - y).abs() > 1e-9)
        });
        assert!(
            any_changed,
            "Online weights should change after gradient steps"
        );
    }
}
