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

use crate::persistence::{ExperienceStore, ExperienceStoreConfig, TrainingSample};
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
use janus_vision::gaf_features_from_closes;

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

    // ── GAF seeding (ML Phase 1) ──────────────────────────────────────
    /// Optional real close-price series used to seed the replay buffer with
    /// real GAF-feature experiences instead of zero placeholders. `None` keeps
    /// the historical placeholder behaviour (the default), so this is opt-in.
    pub seed_closes: Option<Vec<f32>>,
    /// Window length for GAF seeding.
    pub gaf_window: usize,
    /// GAF image size for seeding.
    pub gaf_image_size: usize,

    // ── Qdrant training source (Phase 2 — EXPERIENCE_PIPELINE.md §7) ───
    /// When `true`, fill the replay buffer from live-collected experiences in
    /// Qdrant *in addition to* the seed path. Defaults to `false` and is
    /// normally driven by the [`TRAIN_FROM_QDRANT_ENV`] env var in
    /// [`handle_training`]; the seed/placeholder path is unchanged when off.
    pub train_from_qdrant: bool,
    /// Maximum number of experiences to scroll from Qdrant per training job.
    pub qdrant_sample_limit: usize,
    /// Optional lower bound (epoch ms) on `timestamp_ms` when loading from
    /// Qdrant — `None` loads the most recent `qdrant_sample_limit` points.
    pub qdrant_since_ms: Option<i64>,
}

/// Environment variable gating the Qdrant training-data path. Absent or any
/// value other than `1/true/yes/on` (case-insensitive) keeps the live default:
/// training seeds from `seed_closes`/placeholders exactly as before.
pub const TRAIN_FROM_QDRANT_ENV: &str = "JANUS_TRAIN_FROM_QDRANT";

/// Whether the Qdrant training path is enabled via [`TRAIN_FROM_QDRANT_ENV`].
/// Default OFF.
fn train_from_qdrant_enabled() -> bool {
    parse_train_from_qdrant(std::env::var(TRAIN_FROM_QDRANT_ENV).ok().as_deref())
}

/// Pure env-value parser for the Qdrant training gate (kept separate from
/// [`train_from_qdrant_enabled`] so it is testable without mutating process
/// env). Only `1/true/yes/on` (case-insensitive, trimmed) enable it; anything
/// else — including `None` (unset) — leaves it OFF.
fn parse_train_from_qdrant(val: Option<&str>) -> bool {
    match val {
        Some(v) => matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
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
            seed_closes: None,
            gaf_window: 32,
            gaf_image_size: 16,
            train_from_qdrant: false,
            qdrant_sample_limit: 10_000,
            qdrant_since_ms: None,
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

/// Build seed experiences for the replay buffer. Returns real GAF-feature
/// experiences when `config.seed_closes` is set, otherwise an empty vec (the
/// caller then falls back to [`create_placeholder_experience`]).
fn build_seed_experiences(config: &TrainingConfig) -> Result<Vec<Experience>> {
    match &config.seed_closes {
        Some(closes) => {
            experiences_from_series(closes, config.gaf_window, config.gaf_image_size, "SEED")
        }
        None => Ok(Vec::new()),
    }
}

/// Build a *real* training experience from a price window, computing GAF
/// features via DiffGAF instead of the zero placeholder used by
/// [`create_placeholder_experience`].
///
/// `closes` is the current window and `next_closes` the window advanced one bar
/// (the resulting next state). Once a real OHLCV source feeds the training loop
/// (next increment), the replay buffer seeds from these real-feature
/// experiences instead of zeros — closing the train-on-zeros gap documented in
/// `docs/architecture/ML_VISION_SCOPE.md`.
pub fn experience_from_closes(
    closes: &[f32],
    next_closes: &[f32],
    action_type: ActionType,
    reward: f32,
    done: bool,
    symbol: &str,
    image_size: usize,
) -> Result<Experience> {
    let device = candle_core::Device::Cpu;
    let gaf = gaf_features_from_closes(closes, image_size, &device)
        .context("computing GAF features for state")?;
    let next_gaf = gaf_features_from_closes(next_closes, image_size, &device)
        .context("computing GAF features for next state")?;

    let meta = StateMetadata::new(symbol.to_string());
    let state = State::from_flat_gaf(gaf.flat, closes.to_vec(), meta.clone());
    let next_state = State::from_flat_gaf(next_gaf.flat, next_closes.to_vec(), meta);
    let action = Action::new(action_type, symbol.to_string(), 0.0);

    Ok(Experience::new(state, action, reward, next_state, done))
}

/// Build a batch of real training experiences by sliding a `window`-length
/// window over `closes`.
///
/// Each step yields one experience whose features come from DiffGAF (real, not
/// the zero placeholder) and whose action/reward use a simple directional
/// scheme: the action is `Buy` when the next bar rises and `Sell` when it
/// falls, rewarded by the realized next-bar return. This is a deliberately
/// simple off-policy seed for the DQN replay buffer — the reward/label scheme
/// is exactly the kind of thing the eventual feasibility probe will tune.
///
/// Returns an empty vec when `closes` is shorter than `window + 1`.
pub fn experiences_from_series(
    closes: &[f32],
    window: usize,
    image_size: usize,
    symbol: &str,
) -> Result<Vec<Experience>> {
    let mut experiences = Vec::new();
    if window == 0 || closes.len() < window + 1 {
        return Ok(experiences);
    }
    for start in 0..(closes.len() - window) {
        let cur = &closes[start..start + window];
        let next = &closes[start + 1..start + 1 + window];
        // Realized return of the bar just past the current window.
        let last = closes[start + window - 1];
        let nxt = closes[start + window];
        let ret = (nxt - last) / last.abs().max(f32::EPSILON);
        let action = if ret >= 0.0 {
            ActionType::Buy
        } else {
            ActionType::Sell
        };
        experiences.push(experience_from_closes(
            cur, next, action, ret, false, symbol, image_size,
        )?);
    }
    Ok(experiences)
}

// ─── Qdrant experience loading (Phase 2) ────────────────────────────────────────

/// Map the canonical stored action index (design §3.2: `0=Buy, 1=Sell,
/// 2=Hold, 3=Close`) back to an [`ActionType`]. Inverse of [`action_to_index`].
/// Unknown indices fall back to `Hold` (the neutral action).
fn action_from_index(index: u8) -> ActionType {
    match index {
        0 => ActionType::Buy,
        1 => ActionType::Sell,
        3 => ActionType::Close,
        _ => ActionType::Hold,
    }
}

/// Convert a Qdrant-recovered [`TrainingSample`] into a replay-buffer
/// [`Experience`].
///
/// The stored GAF vectors drop straight into `State::from_flat_gaf` — the
/// replay buffer only ever reads `gaf_features_flat` (via `state_to_features`,
/// which pads/truncates to `model_input_size`), so a dim mismatch degrades to
/// pad/truncate rather than a panic. The raw-feature channel is left empty
/// (`state_raw` is optional in the record and unused by the LSTM path today).
fn experience_from_training_sample(sample: &TrainingSample) -> Experience {
    let meta = StateMetadata::new("QDRANT".to_string());
    let state = State::from_flat_gaf(sample.state_vector.clone(), vec![], meta.clone());
    let next_state = State::from_flat_gaf(sample.next_state_vector.clone(), vec![], meta);
    let action = Action::new(action_from_index(sample.action), "QDRANT".to_string(), 0.0);
    Experience::new(state, action, sample.reward, next_state, sample.done)
}

/// Summary of the reward signal across a batch of loaded experiences, logged so
/// a training run from live data is observable (requirement 3).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RewardDistribution {
    pub count: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
    pub positive: usize,
    pub negative: usize,
    pub zero: usize,
}

impl RewardDistribution {
    /// Compute reward summary statistics over `experiences`. Returns the
    /// all-zero default for an empty slice.
    fn from_experiences(experiences: &[Experience]) -> Self {
        if experiences.is_empty() {
            return Self::default();
        }
        let rewards: Vec<f32> = experiences.iter().map(|e| e.reward).collect();
        let count = rewards.len();
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let (mut positive, mut negative, mut zero) = (0usize, 0usize, 0usize);
        for &r in &rewards {
            min = min.min(r);
            max = max.max(r);
            sum += r as f64;
            if r > 0.0 {
                positive += 1;
            } else if r < 0.0 {
                negative += 1;
            } else {
                zero += 1;
            }
        }
        let mean = (sum / count as f64) as f32;
        let var = rewards
            .iter()
            .map(|&r| {
                let d = r as f64 - mean as f64;
                d * d
            })
            .sum::<f64>()
            / count as f64;
        Self {
            count,
            min,
            max,
            mean,
            std: var.sqrt() as f32,
            positive,
            negative,
            zero,
        }
    }
}

/// Load live-collected experiences from Qdrant and map them into replay-buffer
/// [`Experience`]s (EXPERIENCE_PIPELINE.md §7 Phase 2 item 2).
///
/// Constructs an [`ExperienceStore`] from the process env (`QDRANT_URL` etc.)
/// and scrolls the newest `qdrant_sample_limit` points (optionally bounded by
/// `qdrant_since_ms`) via [`ExperienceStore::sample_for_training`]. Any store
/// or scroll error propagates to the caller, which logs and falls back to the
/// seed path — a Qdrant outage must never abort a training job.
async fn load_qdrant_experiences(config: &TrainingConfig) -> Result<Vec<Experience>> {
    let store_cfg = ExperienceStoreConfig::from_env();
    let store = ExperienceStore::new(store_cfg)
        .await
        .context("connecting to Qdrant for training experiences")?;
    let samples = store
        .sample_for_training(config.qdrant_sample_limit, config.qdrant_since_ms)
        .await
        .context("sampling experiences from Qdrant")?;
    Ok(samples
        .iter()
        .map(experience_from_training_sample)
        .collect())
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

// ─── Replay-buffer construction ───────────────────────────────────────────────

/// Build the PER replay buffer for a training run.
///
/// Creates a [`SumTree`] of `config.replay_capacity`, optionally fills it from
/// live Qdrant experiences (Phase 2, env-gated via `config.train_from_qdrant`),
/// then tops up any shortfall to `batch_size` with GAF-seed / placeholder
/// experiences. Shared verbatim by [`handle_training`] and
/// [`run_training_session`] so both build the buffer identically.
///
/// A Qdrant load error is logged and degrades to the seed path rather than
/// aborting — a Qdrant outage must never abort training.
async fn build_training_buffer(config: &TrainingConfig, batch_size: usize) -> Result<SumTree> {
    // ── Initialise PER buffer ─────────────────────────────────────────────
    let mut replay_buffer = SumTree::new(config.replay_capacity, config.per_alpha, config.per_beta);

    // ── Optionally fill from live Qdrant experiences (Phase 2) ────────────
    //
    // Env-gated OFF by default: when `JANUS_TRAIN_FROM_QDRANT` is unset, this
    // block is skipped entirely and the seed/placeholder path below is the
    // sole source — live training behaviour is unchanged. When enabled, the
    // buffer is filled from live-collected experiences *in addition to* the
    // seed fallback (which still tops up any shortfall to `batch_size`).
    if config.train_from_qdrant {
        match load_qdrant_experiences(config).await {
            Ok(exps) => {
                let dist = RewardDistribution::from_experiences(&exps);
                info!(
                    loaded = dist.count,
                    reward_min = dist.min,
                    reward_max = dist.max,
                    reward_mean = dist.mean,
                    reward_std = dist.std,
                    reward_positive = dist.positive,
                    reward_negative = dist.negative,
                    reward_zero = dist.zero,
                    limit = config.qdrant_sample_limit,
                    since_ms = config.qdrant_since_ms,
                    "Loaded live experiences from Qdrant for training"
                );
                for exp in exps {
                    replay_buffer.add(exp, 1.0);
                }
            }
            Err(e) => {
                warn!(
                    error = %e,
                    "Failed to load experiences from Qdrant — falling back to seed path"
                );
            }
        }
    }

    let buffer_size = replay_buffer.len();
    if buffer_size < batch_size {
        warn!(
            buffer_size = buffer_size,
            required = batch_size,
            "Replay buffer has fewer experiences than batch size — seeding with placeholders"
        );
        let needed = batch_size.saturating_sub(buffer_size);
        // Seed with real GAF-feature experiences when configured (ML Phase 1),
        // otherwise zero placeholders. Default config has `seed_closes: None`,
        // so behaviour is unchanged unless real seed data is supplied.
        let seed_exps = match build_seed_experiences(config) {
            Ok(exps) => exps,
            Err(e) => {
                warn!(error = %e, "GAF seeding failed; falling back to placeholders");
                Vec::new()
            }
        };
        for i in 0..needed {
            let exp = if seed_exps.is_empty() {
                create_placeholder_experience()
            } else {
                seed_exps[i % seed_exps.len()].clone()
            };
            replay_buffer.add(exp, 1.0);
            debug!(index = i, "Seeded experience");
        }
    }

    Ok(replay_buffer)
}

// ─── Single training step ─────────────────────────────────────────────────────

/// Outcome of one gradient step over a sampled PER batch.
struct StepOutcome {
    /// Raw result of the gradient step (loss, mean Q, max TD error).
    step_result: DqnStepResult,
    /// Mean TD target across the batch.
    mean_target: f64,
    /// Mean absolute TD error (recomputed post-update).
    mean_td_error: f64,
    /// Max absolute TD error (recomputed post-update).
    max_td_error: f64,
    /// Number of experiences in the batch.
    batch_size: usize,
}

/// Run ONE training step: sample a prioritized batch, compute Double-DQN TD
/// targets, take an IS-weighted gradient step on the online network,
/// soft-update the target network, and refresh replay priorities.
///
/// Mutates `online`, `target_model` and `replay_buffer` in place, so repeated
/// calls with the same models + buffer + optimiser accumulate learning across
/// steps (used by [`run_training_session`]). [`handle_training`] calls it once.
fn train_one_step(
    online: &mut DqnOnlineModel,
    target_model: &mut LstmPredictor<CpuBackend>,
    replay_buffer: &mut SumTree,
    config: &TrainingConfig,
    batch_size: usize,
) -> Result<StepOutcome> {
    // ── Sample a prioritized batch ────────────────────────────────────────
    let samples = replay_buffer
        .sample(batch_size)
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

    // ── Compute Double-DQN TD targets ─────────────────────────────────────
    let next_states: Vec<Vec<f32>> = experiences
        .iter()
        .map(|e| state_to_features(&e.next_state, config.model_input_size))
        .collect();
    let rewards: Vec<f64> = experiences.iter().map(|e| e.reward as f64).collect();
    let dones: Vec<bool> = experiences.iter().map(|e| e.done).collect();

    let td_targets = compute_double_dqn_targets(
        online,
        target_model,
        &next_states,
        &rewards,
        &dones,
        config.gamma,
        config.model_input_size,
        1, // seq_len
    );

    let mean_target = td_targets.iter().sum::<f64>() / td_targets.len() as f64;

    // ── Gradient step on the online network ───────────────────────────────
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

    // ── Soft-update the target network ────────────────────────────────────
    online.soft_update_target(target_model, config.tau);

    debug!(
        tau = config.tau,
        "Target network soft-updated (Polyak averaging)"
    );

    // ── Update priorities in the replay buffer ────────────────────────────
    //
    // Recompute per-sample TD errors using the *updated* online network so
    // that priorities reflect the latest model state.
    let epsilon = 1e-6_f64;
    let td_errors = compute_td_errors(online, target_model, &experiences, config);

    for (i, &idx) in indices.iter().enumerate() {
        let new_priority = td_errors[i].abs() + epsilon;
        replay_buffer.update_priority(idx, new_priority);
    }

    info!(updated = indices.len(), "Updated replay buffer priorities");

    let mean_td_error = td_errors.iter().map(|e| e.abs()).sum::<f64>() / td_errors.len() as f64;
    let max_td_error = td_errors.iter().map(|e| e.abs()).fold(0.0_f64, f64::max);

    Ok(StepOutcome {
        step_result,
        mean_target,
        mean_td_error,
        max_td_error,
        batch_size: experiences.len(),
    })
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
    let config = TrainingConfig {
        // Env-gated Phase 2 path; default OFF leaves the seed path untouched.
        train_from_qdrant: train_from_qdrant_enabled(),
        ..TrainingConfig::default()
    };

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

    // ── 2. Build the replay buffer (Qdrant + seed/placeholder fallback) ───
    let mut replay_buffer = build_training_buffer(&config, job.batch_size).await?;

    // ── 3–7. One gradient step over a prioritized batch ───────────────────
    let outcome = train_one_step(
        &mut online,
        &mut target_model,
        &mut replay_buffer,
        &config,
        job.batch_size,
    )?;
    let step_result = outcome.step_result;

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
    let metrics = TrainStepMetrics {
        mean_td_error: outcome.mean_td_error,
        mean_loss: step_result.loss,
        max_td_error: outcome.max_td_error,
        batch_size: outcome.batch_size,
        buffer_utilization: replay_buffer_utilization(replay_buffer.len(), config.replay_capacity),
        mean_q_value: step_result.mean_q,
        mean_target_q_value: outcome.mean_target,
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

// ─── Multi-step challenger training session ───────────────────────────────────

/// Default per-step batch size for a scheduled (challenger) training session.
/// Mirrors the default [`TrainJob`] batch size used by [`handle_training`].
const SESSION_BATCH_SIZE: usize = 32;

/// Metrics summarising a multi-step training session (challenger training).
#[derive(Debug, Clone)]
pub struct TrainSessionMetrics {
    /// Number of gradient steps actually performed.
    pub steps: usize,
    /// Mean loss across all steps.
    pub mean_loss: f64,
    /// Loss of the final step.
    pub final_loss: f64,
    /// Mean Q-value (of taken actions) across all steps.
    pub mean_q: f64,
    /// Replay-buffer size the session trained on.
    pub buffer_size: usize,
}

/// Run a multi-step DQN training session from a FRESH online + target model over
/// a single replay buffer, checkpointing the result to `config.checkpoint_dir`.
///
/// The models, buffer and optimiser are created once and reused across all
/// `steps` gradient steps so learning accumulates within the session.
///
/// ## SAFETY
///
/// This deliberately takes **no** [`CheckpointNotifier`]: a scheduled session
/// must NOT publish a checkpoint notification (which could hot-load the model
/// into forward inference — safety invariant 3). The caller
/// ([`run_training_scheduler`]) points `config.checkpoint_dir` at the
/// *challenger* directory, never the live champion dir (invariant 2), so a
/// scheduled session can never affect the live signal model.
pub async fn run_training_session(
    config: &TrainingConfig,
    steps: usize,
) -> Result<TrainSessionMetrics> {
    info!(
        steps,
        checkpoint_dir = %config.checkpoint_dir,
        train_from_qdrant = config.train_from_qdrant,
        "Starting multi-step challenger training session"
    );

    // ── Build fresh Q-networks (same construction as handle_training) ─────
    let trainable_cfg = build_trainable_config(config);
    let inference_cfg = build_inference_config(config);

    let mut online = DqnOnlineModel::new(trainable_cfg, config.learning_rate, config.weight_decay)
        .context("Failed to create online DQN model")?;
    let mut target_model = LstmPredictor::new(inference_cfg, BackendDevice::cpu());
    online.soft_update_target(&mut target_model, 1.0); // full copy to start

    // ── Build the replay buffer ONCE — steps reuse it so priorities evolve ─
    let mut replay_buffer = build_training_buffer(config, SESSION_BATCH_SIZE).await?;
    let buffer_size = replay_buffer.len();

    // ── Loop the per-step core, accumulating learning across steps ────────
    let mut total_loss = 0.0_f64;
    let mut total_q = 0.0_f64;
    let mut final_loss = 0.0_f64;
    let mut completed = 0usize;

    for step in 0..steps {
        let outcome = train_one_step(
            &mut online,
            &mut target_model,
            &mut replay_buffer,
            config,
            SESSION_BATCH_SIZE,
        )
        .with_context(|| format!("training session failed at step {step}"))?;
        total_loss += outcome.step_result.loss;
        total_q += outcome.step_result.mean_q;
        final_loss = outcome.step_result.loss;
        completed += 1;
    }

    let mean_loss = if completed > 0 {
        total_loss / completed as f64
    } else {
        0.0
    };
    let mean_q = if completed > 0 {
        total_q / completed as f64
    } else {
        0.0
    };

    // ── Checkpoint to the (challenger) dir — NO notifier is passed anywhere,
    //    so no CheckpointNotification is ever published (invariant 3). ─────
    let predictor = online
        .to_inference(BackendDevice::cpu())
        .context("Failed to convert online model to inference for checkpoint")?;
    save_checkpoint(&predictor, std::path::Path::new(&config.checkpoint_dir))
        .context("Failed to save challenger checkpoint")?;

    let metrics = TrainSessionMetrics {
        steps: completed,
        mean_loss,
        final_loss,
        mean_q,
        buffer_size,
    };

    info!(
        steps = metrics.steps,
        mean_loss = format!("{:.6}", metrics.mean_loss),
        final_loss = format!("{:.6}", metrics.final_loss),
        mean_q = format!("{:.4}", metrics.mean_q),
        buffer_size = metrics.buffer_size,
        checkpoint_dir = %config.checkpoint_dir,
        "Challenger training session completed (challenger checkpoint written)"
    );

    Ok(metrics)
}

// ─── Scheduled challenger training (SAFE, gated) ──────────────────────────────

/// Env var gating the periodic training scheduler. Default OFF: while unset or
/// not truthy the scheduler logs "disabled" and returns. Deploying this code
/// changes NOTHING until this is explicitly set truthy (safety invariant 1).
pub const TRAIN_SCHEDULE_ENABLED_ENV: &str = "JANUS_TRAIN_SCHEDULE_ENABLED";
/// Env var: seconds between scheduled sessions (default 900, floored to 60).
pub const TRAIN_INTERVAL_SECS_ENV: &str = "JANUS_TRAIN_INTERVAL_SECS";
/// Env var: gradient steps per scheduled session (default 200, floored to 1).
pub const TRAIN_STEPS_PER_SESSION_ENV: &str = "JANUS_TRAIN_STEPS_PER_SESSION";
/// Env var: challenger checkpoint dir (default [`DEFAULT_CHALLENGER_DIR`]).
pub const TRAIN_CHALLENGER_DIR_ENV: &str = "JANUS_TRAIN_CHALLENGER_DIR";

/// The LIVE champion checkpoint dir that forward inference hot-loads from.
/// Scheduled training must NEVER write here (safety invariants 2 & 4).
const CHAMPION_CHECKPOINT_DIR: &str = "checkpoints/backward";
/// Default challenger checkpoint dir — a *sub*directory of the champion dir so
/// it can never collide with the champion's own `latest_model.bin`.
const DEFAULT_CHALLENGER_DIR: &str = "checkpoints/backward/challenger";

/// Lexically normalize a path (no filesystem access): drop `.`/empty components
/// and resolve `..`. Enough to fold trailing-slash / `./` / `..` variants that
/// an exact-string compare would miss.
fn lexical_normalize(p: &str) -> std::path::PathBuf {
    use std::path::{Component, PathBuf};
    let mut out = PathBuf::new();
    for comp in std::path::Path::new(p).components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// True if scheduled training into `challenger_dir` would write the SAME
/// `latest_model.bin` as the live champion (safety invariant 4). Compares
/// lexically-normalized paths — and canonical forms when both resolve on disk —
/// so a trailing slash, `./`, `..`, or a relative-vs-absolute spelling of the
/// champion dir cannot slip past what used to be an exact-string guard.
fn collides_with_champion(challenger_dir: &str) -> bool {
    let chal = lexical_normalize(challenger_dir);
    let champ = lexical_normalize(CHAMPION_CHECKPOINT_DIR);
    if chal == champ {
        return true;
    }
    // Relative-vs-absolute (or symlinked) spellings of the same real directory:
    // compare canonical forms when both actually exist on disk.
    match (std::fs::canonicalize(&chal), std::fs::canonicalize(&champ)) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// Configuration for the periodic challenger-training scheduler, resolved from
/// the process environment. Every field defaults to a safe/OFF value.
#[derive(Debug, Clone)]
struct TrainScheduleConfig {
    /// Master gate — false unless [`TRAIN_SCHEDULE_ENABLED_ENV`] is truthy.
    enabled: bool,
    /// Seconds between sessions (>= 60).
    interval_secs: u64,
    /// Gradient steps per session (>= 1).
    steps_per_session: usize,
    /// Challenger checkpoint dir (never the champion dir).
    challenger_dir: String,
}

impl TrainScheduleConfig {
    /// Resolve the scheduler config from the environment. Defaults are safe:
    /// disabled, 900s interval, 200 steps, challenger dir.
    fn from_env() -> Self {
        // Reuse the same truthy parser as the Qdrant gate so `1/true/yes/on`
        // (case/whitespace tolerant) enable it and everything else — including
        // unset — leaves it OFF (invariant 1).
        let enabled =
            parse_train_from_qdrant(std::env::var(TRAIN_SCHEDULE_ENABLED_ENV).ok().as_deref());
        let interval_secs = std::env::var(TRAIN_INTERVAL_SECS_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(900)
            .max(60);
        let steps_per_session = std::env::var(TRAIN_STEPS_PER_SESSION_ENV)
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(200)
            .max(1);
        let challenger_dir = std::env::var(TRAIN_CHALLENGER_DIR_ENV)
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_CHALLENGER_DIR.to_string());
        Self {
            enabled,
            interval_secs,
            steps_per_session,
            challenger_dir,
        }
    }
}

/// Periodic, SAFE, gated challenger-training scheduler.
///
/// ## Safety invariants (enforced here)
///
/// 1. **Default OFF.** Returns immediately (logging "disabled") unless
///    [`TRAIN_SCHEDULE_ENABLED_ENV`] is explicitly truthy — deploying this code
///    changes nothing until an operator opts in.
/// 2. **Challenger dir only.** Each session's [`TrainingConfig::checkpoint_dir`]
///    is set to the challenger dir, never the champion dir.
/// 3. **No notifier.** It calls [`run_training_session`], which takes no
///    [`CheckpointNotifier`], so no checkpoint notification is ever published
///    and forward inference can never hot-load a scheduled checkpoint.
/// 4. **Belt-and-braces guard.** If the resolved challenger dir equals the live
///    champion dir it logs an error and refuses to run.
///
/// The loop is *sleep-first* (never trains at the boot instant) and never
/// propagates a session error — a failed session is logged and the scheduler
/// survives to retry on the next interval.
pub async fn run_training_scheduler(cancel: tokio_util::sync::CancellationToken) {
    let cfg = TrainScheduleConfig::from_env();

    // ── Invariant 1: default-OFF master gate ──────────────────────────────
    if !cfg.enabled {
        info!(
            env = TRAIN_SCHEDULE_ENABLED_ENV,
            "training scheduler disabled — set JANUS_TRAIN_SCHEDULE_ENABLED=true to enable; no scheduled training will run"
        );
        return;
    }

    // ── Invariant 4: refuse to ever train into the live champion dir ──────
    // Path-aware (not exact-string): folds trailing-slash / `./` / `..` /
    // relative-vs-absolute spellings so a misconfigured challenger dir can't
    // resolve onto the champion's latest_model.bin.
    if collides_with_champion(&cfg.challenger_dir) {
        tracing::error!(
            challenger_dir = %cfg.challenger_dir,
            champion_dir = CHAMPION_CHECKPOINT_DIR,
            "training scheduler refusing to run: challenger dir resolves onto the live champion checkpoint dir — scheduled training must never write the live model"
        );
        return;
    }

    let interval = tokio::time::Duration::from_secs(cfg.interval_secs);
    info!(
        interval_secs = cfg.interval_secs,
        steps_per_session = cfg.steps_per_session,
        challenger_dir = %cfg.challenger_dir,
        "training scheduler enabled — periodic challenger training will run"
    );

    loop {
        tokio::select! {
            _ = cancel.cancelled() => {
                info!("training scheduler received shutdown signal");
                break;
            }
            // Sleep-first: never train at the boot instant.
            _ = tokio::time::sleep(interval) => {
                let config = TrainingConfig {
                    train_from_qdrant: train_from_qdrant_enabled(),
                    // Invariant 2: challenger dir only — never the champion.
                    checkpoint_dir: cfg.challenger_dir.clone(),
                    ..TrainingConfig::default()
                };
                // Invariant 3: no notifier is passed on this path.
                match run_training_session(&config, cfg.steps_per_session).await {
                    Ok(m) => info!(
                        steps = m.steps,
                        mean_loss = format!("{:.6}", m.mean_loss),
                        final_loss = format!("{:.6}", m.final_loss),
                        mean_q = format!("{:.4}", m.mean_q),
                        buffer_size = m.buffer_size,
                        challenger_dir = %cfg.challenger_dir,
                        "scheduled challenger training session completed"
                    ),
                    Err(e) => warn!(
                        error = %e,
                        "scheduled challenger training session failed — scheduler will retry next interval"
                    ),
                }
            }
        }
    }

    info!("training scheduler exited");
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
    fn test_experience_from_closes_has_real_features() {
        // Real GAF features, in contrast to create_placeholder_experience's zeros.
        let closes: Vec<f32> = (0..32).map(|t| 100.0 + (t as f32 * 0.2).sin()).collect();
        let next: Vec<f32> = (1..33).map(|t| 100.0 + (t as f32 * 0.2).sin()).collect();

        let exp = experience_from_closes(&closes, &next, ActionType::Buy, 1.0, false, "BTCUSD", 16)
            .unwrap();

        assert!(!exp.state.gaf_features_flat.is_empty());
        assert!(exp.state.gaf_features_flat.iter().any(|&v| v != 0.0));
        assert!(exp.state.gaf_features_flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_experiences_from_series_produces_real_experiences() {
        let closes: Vec<f32> = (0..50).map(|t| 100.0 + (t as f32 * 0.15).sin()).collect();
        let window = 16;
        let exps = experiences_from_series(&closes, window, 16, "BTCUSD").unwrap();

        assert_eq!(exps.len(), closes.len() - window);
        for e in &exps {
            assert!(e.state.gaf_features_flat.iter().any(|&v| v != 0.0));
            assert!(e.state.gaf_features_flat.iter().all(|v| v.is_finite()));
            assert!(e.reward.is_finite());
        }
    }

    #[test]
    fn test_experiences_from_series_handles_short_input() {
        let exps = experiences_from_series(&[1.0, 2.0, 3.0], 16, 16, "X").unwrap();
        assert!(exps.is_empty());
    }

    // ── Phase 2: Qdrant training path (env-gated OFF) ─────────────────────

    #[test]
    fn test_train_from_qdrant_default_off() {
        // The gate must default OFF so live training behaviour is unchanged
        // unless the operator opts in.
        assert!(!TrainingConfig::default().train_from_qdrant);
    }

    #[test]
    fn test_parse_train_from_qdrant_gate() {
        // Unset / empty / arbitrary values leave the gate OFF.
        assert!(!parse_train_from_qdrant(None));
        assert!(!parse_train_from_qdrant(Some("")));
        assert!(!parse_train_from_qdrant(Some("0")));
        assert!(!parse_train_from_qdrant(Some("false")));
        assert!(!parse_train_from_qdrant(Some("nope")));
        // Only the canonical truthy tokens enable it (case/whitespace tolerant).
        assert!(parse_train_from_qdrant(Some("1")));
        assert!(parse_train_from_qdrant(Some("true")));
        assert!(parse_train_from_qdrant(Some(" TRUE ")));
        assert!(parse_train_from_qdrant(Some("Yes")));
        assert!(parse_train_from_qdrant(Some("on")));
    }

    #[test]
    fn test_action_from_index_round_trips_action_to_index() {
        for at in [
            ActionType::Buy,
            ActionType::Sell,
            ActionType::Hold,
            ActionType::Close,
        ] {
            let idx = action_to_index(&at) as u8;
            assert_eq!(action_from_index(idx), at);
        }
        // Out-of-range indices fall back to the neutral Hold action.
        assert_eq!(action_from_index(9), ActionType::Hold);
    }

    #[test]
    fn test_experience_from_training_sample_maps_replay_tuple() {
        let sample = TrainingSample {
            state_vector: vec![0.1, 0.2, 0.3],
            action: 1, // Sell
            reward: -0.0007,
            next_state_vector: vec![0.4, 0.5, 0.6],
            done: true,
        };
        let exp = experience_from_training_sample(&sample);

        assert_eq!(exp.state.gaf_features_flat, vec![0.1, 0.2, 0.3]);
        assert_eq!(exp.next_state.gaf_features_flat, vec![0.4, 0.5, 0.6]);
        assert_eq!(exp.action.action_type, ActionType::Sell);
        assert_eq!(exp.reward, -0.0007);
        assert!(exp.done);
        // The mapped experience must feed the model without a shape panic: the
        // features pad/truncate to model_input_size.
        let feats = state_to_features(&exp.state, 9);
        assert_eq!(feats.len(), 9);
        assert_eq!(&feats[0..3], &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_reward_distribution_summary() {
        let make = |reward: f32| {
            let meta = StateMetadata::new("T".to_string());
            let s = State::from_flat_gaf(vec![0.0; 9], vec![], meta.clone());
            let ns = State::from_flat_gaf(vec![0.0; 9], vec![], meta);
            let a = Action::new(ActionType::Hold, "T".to_string(), 0.0);
            Experience::new(s, a, reward, ns, false)
        };
        let exps = vec![make(1.0), make(-1.0), make(0.0), make(2.0)];
        let dist = RewardDistribution::from_experiences(&exps);
        assert_eq!(dist.count, 4);
        assert_eq!(dist.positive, 2);
        assert_eq!(dist.negative, 1);
        assert_eq!(dist.zero, 1);
        assert_eq!(dist.min, -1.0);
        assert_eq!(dist.max, 2.0);
        assert!((dist.mean - 0.5).abs() < 1e-6);
        assert!(dist.std.is_finite() && dist.std > 0.0);

        // Empty input is the all-zero default (no NaNs).
        let empty = RewardDistribution::from_experiences(&[]);
        assert_eq!(empty, RewardDistribution::default());
    }

    #[test]
    fn test_seed_path_unchanged_when_qdrant_off() {
        // With the gate off (default) and no seed_closes, seeding still yields
        // an empty vec so the placeholder fallback path is untouched.
        let cfg = TrainingConfig::default();
        assert!(!cfg.train_from_qdrant);
        assert!(build_seed_experiences(&cfg).unwrap().is_empty());
    }

    #[test]
    fn test_build_seed_experiences_real_vs_placeholder() {
        // Default (no seed_closes) -> empty, so seeding falls back to placeholders.
        assert!(
            build_seed_experiences(&TrainingConfig::default())
                .unwrap()
                .is_empty()
        );

        // With a real close series -> real-feature experiences.
        let cfg = TrainingConfig {
            gaf_window: 16,
            gaf_image_size: 16,
            seed_closes: Some((0..40).map(|t| 100.0 + (t as f32 * 0.2).sin()).collect()),
            ..TrainingConfig::default()
        };
        let exps = build_seed_experiences(&cfg).unwrap();
        assert!(!exps.is_empty());
        assert!(exps[0].state.gaf_features_flat.iter().any(|&v| v != 0.0));
    }

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

    // ── Challenger training scheduler (SAFE, gated) ───────────────────────

    #[test]
    fn test_train_one_step_returns_finite_loss() {
        let config = TrainingConfig::default();
        let tcfg = build_trainable_config(&config);
        let icfg = build_inference_config(&config);

        let mut online = DqnOnlineModel::new(tcfg, config.learning_rate, config.weight_decay)
            .expect("Failed to create DQN online model");
        let mut target = LstmPredictor::new(icfg, BackendDevice::cpu());
        online.soft_update_target(&mut target, 1.0);

        let batch_size = 8;
        let mut buffer = SumTree::new(config.replay_capacity, config.per_alpha, config.per_beta);
        for _ in 0..batch_size {
            buffer.add(create_placeholder_experience(), 1.0);
        }

        let outcome = train_one_step(&mut online, &mut target, &mut buffer, &config, batch_size)
            .expect("train_one_step failed");
        assert!(
            outcome.step_result.loss.is_finite(),
            "loss should be finite: {}",
            outcome.step_result.loss
        );
        assert_eq!(outcome.batch_size, batch_size);
        assert_eq!(online.step_count, 1, "one step should advance the counter");
    }

    #[tokio::test]
    async fn test_run_training_session_writes_challenger_only() {
        // Point the session at an absolute tempdir; it must write ONLY there
        // and never touch the live champion path.
        let tmp = tempfile::tempdir().expect("create tempdir");
        let challenger = tmp.path().join("challenger");
        let config = TrainingConfig {
            checkpoint_dir: challenger.to_string_lossy().to_string(),
            ..TrainingConfig::default()
        };

        let metrics = run_training_session(&config, 3)
            .await
            .expect("training session failed");

        assert_eq!(metrics.steps, 3, "should complete all 3 steps");
        assert!(metrics.mean_loss.is_finite());
        assert!(metrics.final_loss.is_finite());

        // The challenger checkpoint WAS written.
        assert!(
            challenger.join("latest_model.bin").exists(),
            "challenger checkpoint should exist"
        );

        // The live champion path was NOT created anywhere in the sandbox — the
        // session must never write "checkpoints/backward/latest_model.bin".
        assert!(
            !tmp.path()
                .join("checkpoints/backward/latest_model.bin")
                .exists(),
            "the live champion checkpoint must remain untouched"
        );
    }

    #[test]
    fn test_train_schedule_config_from_env_default_safe() {
        // With the schedule env vars unset in the test process, the resolved
        // config must be safe: disabled and NEVER the champion dir.
        let cfg = TrainScheduleConfig::from_env();
        assert!(!cfg.enabled, "scheduler must default OFF");
        assert_ne!(
            cfg.challenger_dir, CHAMPION_CHECKPOINT_DIR,
            "challenger dir must never default to the champion dir"
        );
        assert_eq!(cfg.challenger_dir, DEFAULT_CHALLENGER_DIR);
        assert!(cfg.interval_secs >= 60);
        assert!(cfg.steps_per_session >= 1);
    }

    #[test]
    fn test_collides_with_champion_folds_path_variants() {
        // Invariant 4 must be path-aware, not exact-string: every spelling that
        // resolves onto the champion dir must be rejected.
        assert!(collides_with_champion(CHAMPION_CHECKPOINT_DIR));
        assert!(collides_with_champion("checkpoints/backward/")); // trailing slash
        assert!(collides_with_champion("./checkpoints/backward")); // leading ./
        assert!(collides_with_champion("checkpoints/./backward")); // interior .
        assert!(collides_with_champion("checkpoints/backward/../backward")); // ..
        // The safe default (a subdirectory) and unrelated dirs must NOT collide.
        assert!(!collides_with_champion(DEFAULT_CHALLENGER_DIR));
        assert!(!collides_with_champion("checkpoints/backward/challenger/"));
        assert!(!collides_with_champion("/tmp/janus-challenger"));
        assert!(!collides_with_champion("checkpoints/forward"));
    }

    #[test]
    fn test_train_schedule_enabled_default_off() {
        // Invariant 1: deploying the scheduler must change NOTHING until the
        // operator opts in. The gate value for an unset env var (None) is OFF.
        assert_eq!(TRAIN_SCHEDULE_ENABLED_ENV, "JANUS_TRAIN_SCHEDULE_ENABLED");
        assert!(!parse_train_from_qdrant(None));
        assert!(!parse_train_from_qdrant(Some("false")));
        assert!(!parse_train_from_qdrant(Some("0")));
        // Only explicit truthy tokens enable it.
        assert!(parse_train_from_qdrant(Some("true")));
        assert!(parse_train_from_qdrant(Some("1")));
    }

    #[tokio::test]
    async fn test_run_training_scheduler_disabled_returns_promptly() {
        // With the scheduler disabled (env unset), it must log "disabled" and
        // return immediately rather than entering the sleep/train loop.
        let cancel = tokio_util::sync::CancellationToken::new();
        let res = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            run_training_scheduler(cancel),
        )
        .await;
        assert!(
            res.is_ok(),
            "disabled scheduler should return promptly, not block"
        );
    }
}
