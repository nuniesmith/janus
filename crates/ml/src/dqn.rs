//! DQN (Deep Q-Network) training utilities.
//!
//! This module provides a high-level wrapper around a [`TrainableLstm`]
//! (or [`TrainableMlp`]) that encapsulates:
//!
//! - An Adam optimiser initialised for the online Q-network
//! - A single-step gradient update given pre-computed TD targets
//! - Polyak (τ-based) soft-update of a target network
//! - Conversion of the trained online model to an inference-ready
//!   [`LstmPredictor`] for checkpointing
//!
//! # Design rationale
//!
//! The backward service needs to perform gradient-based weight updates on a
//! `TrainableLstm<AutodiffCpuBackend>` without depending on Burn crates
//! directly.  This module re-exports all the necessary Burn machinery through
//! a simple, opinionated API so that downstream consumers only need to
//! depend on `janus-ml`.
//!
//! # Example
//!
//! ```rust,ignore
//! use janus_ml::dqn::DqnOnlineModel;
//! use janus_ml::models::trainable::TrainableLstmConfig;
//! use janus_ml::backend::BackendDevice;
//!
//! let model_cfg = TrainableLstmConfig::new(9, 64, 4).with_num_layers(2);
//! let mut online = DqnOnlineModel::new(model_cfg, 3e-4, 1e-5).unwrap();
//!
//! // ... sample a batch and compute TD targets with the target network ...
//!
//! let loss = online.gradient_step(
//!     &states,       // Vec<Vec<f32>>
//!     &actions,      // Vec<usize>
//!     &td_targets,   // Vec<f64>
//!     &is_weights,   // Vec<f64>
//!     input_size,
//!     seq_len,
//! ).unwrap();
//!
//! // Soft-update the target predictor
//! online.soft_update_target(&mut target_predictor, 0.005);
//!
//! // Checkpoint the online model as an inference predictor
//! let predictor = online.to_inference(BackendDevice::cpu()).unwrap();
//! predictor.save("checkpoint.bin").unwrap();
//! ```

use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn_core::module::{AutodiffModule, Param};
use burn_core::tensor::{ElementConversion, Tensor, TensorData};

use crate::backend::{AutodiffCpuBackend, BackendDevice, CpuBackend};
use crate::error::Result;
use crate::models::convert::{
    soft_update_lstm_target, trainable_lstm_config_to_inference, trainable_lstm_to_predictor,
};
use crate::models::trainable::{TrainableLstm, TrainableLstmConfig};
use crate::models::{LstmPredictor, WeightMap};

// ---------------------------------------------------------------------------
// DqnOnlineModel
// ---------------------------------------------------------------------------

/// A wrapper around a [`TrainableLstm`] and its Adam optimiser that provides
/// a single-method API for performing one DQN gradient step.
pub struct DqnOnlineModel {
    /// The online Q-network (with autodiff).
    model: TrainableLstm<AutodiffCpuBackend>,

    /// The model configuration (kept for conversions).
    model_config: TrainableLstmConfig,

    /// Adam optimiser bound to the online model.
    optimizer: OptimizerAdaptor<Adam, TrainableLstm<AutodiffCpuBackend>, AutodiffCpuBackend>,

    /// Current learning rate (can be scheduled externally).
    learning_rate: f64,

    /// Cumulative number of gradient steps performed.
    pub step_count: u64,
}

impl std::fmt::Debug for DqnOnlineModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DqnOnlineModel")
            .field("model_config", &self.model_config)
            .field("learning_rate", &self.learning_rate)
            .field("step_count", &self.step_count)
            .finish()
    }
}

/// Metrics returned from a single gradient step.
#[derive(Debug, Clone, Default)]
pub struct DqnStepResult {
    /// The scalar loss value **before** the weight update.
    pub loss: f64,

    /// Mean Q-value of the taken actions (before update).
    pub mean_q: f64,

    /// Max absolute TD error in the batch.
    pub max_td_error: f64,
}

impl DqnOnlineModel {
    /// Create a new online model with a freshly initialised network and
    /// Adam optimiser.
    ///
    /// # Arguments
    ///
    /// * `model_config`  – Architecture of the LSTM Q-network.
    /// * `learning_rate` – Initial learning rate for Adam.
    /// * `weight_decay`  – L2 weight decay coefficient.
    pub fn new(
        model_config: TrainableLstmConfig,
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        let device = <AutodiffCpuBackend as burn_core::tensor::backend::Backend>::Device::default();
        let model: TrainableLstm<AutodiffCpuBackend> = model_config.init(&device);

        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                weight_decay as f32,
            )))
            .init::<AutodiffCpuBackend, TrainableLstm<AutodiffCpuBackend>>();

        Ok(Self {
            model,
            model_config,
            optimizer,
            learning_rate,
            step_count: 0,
        })
    }

    /// Create a `DqnOnlineModel` and immediately load weights from an
    /// existing [`LstmPredictor`] checkpoint.
    ///
    /// This is useful when resuming training from a saved inference model.
    pub fn from_predictor(
        predictor: &LstmPredictor<CpuBackend>,
        model_config: TrainableLstmConfig,
        learning_rate: f64,
        weight_decay: f64,
    ) -> Result<Self> {
        let mut online = Self::new(model_config, learning_rate, weight_decay)?;

        // Extract weights from the inference model, then apply to the
        // trainable model via the round-trip:
        //   predictor weights → inference WeightMap
        //   → build a fresh predictor → extract from trainable (identity)
        // Actually, we can go the other way: build an inference predictor
        // from our fresh online model, then overwrite its weights from the
        // checkpoint predictor, then copy back.  But the simplest route is
        // to load the predictor's weights into a new inference model and
        // re-init the online model from that weight map.
        let weight_map = predictor.extract_weights_pub();
        online.apply_inference_weights(&weight_map);

        Ok(online)
    }

    // -- Gradient step -------------------------------------------------------

    /// Perform one gradient step on the online Q-network.
    ///
    /// The loss is the importance-sampling-weighted MSE between the online
    /// network's Q-values for the taken actions and the externally computed
    /// TD targets:
    ///
    /// ```text
    /// L = (1/N) Σ  w_i · (Q_online(s_i, a_i) − target_i)²
    /// ```
    ///
    /// # Arguments
    ///
    /// * `states`      – Feature vectors for each experience; each inner
    ///   `Vec<f32>` has length `input_size * seq_len`.
    /// * `actions`     – Action index taken in each experience.
    /// * `td_targets`  – Pre-computed TD target values (detached from the
    ///   computation graph).
    /// * `is_weights`  – Per-sample importance-sampling weights (normalised).
    /// * `input_size`  – Feature dimension per timestep.
    /// * `seq_len`     – Number of timesteps per sample (use 1 for single-step).
    ///
    /// # Returns
    ///
    /// A [`DqnStepResult`] containing the loss value and diagnostic metrics.
    pub fn gradient_step(
        &mut self,
        states: &[Vec<f32>],
        actions: &[usize],
        td_targets: &[f64],
        is_weights: &[f64],
        input_size: usize,
        seq_len: usize,
    ) -> Result<DqnStepResult> {
        let batch_size = states.len();
        if batch_size == 0 {
            return Err(crate::error::MLError::InvalidInput(
                "Empty batch".to_string(),
            ));
        }

        let device = <AutodiffCpuBackend as burn_core::tensor::backend::Backend>::Device::default();

        // ── 1. Build input tensor [batch, seq_len, input_size] ────────────
        let flat_input: Vec<f32> = states
            .iter()
            .flat_map(|s| {
                let mut v = s.clone();
                v.resize(input_size * seq_len, 0.0);
                v
            })
            .collect();

        let input = Tensor::<AutodiffCpuBackend, 3>::from_data(
            TensorData::new(flat_input, [batch_size, seq_len, input_size]),
            &device,
        );

        // ── 2. Forward pass (with gradient tracking) ──────────────────────
        let q_all = self.model.forward(input); // [batch, output_size]

        let output_size = self.model_config.output_size;

        // ── 3. Gather Q(s, a) for the taken actions ───────────────────────
        // Build a one-hot mask and use element-wise multiply + sum to select
        // the Q-value for the action taken.
        let mut mask_data = vec![0.0f32; batch_size * output_size];
        for (i, &a) in actions.iter().enumerate() {
            let idx = a.min(output_size - 1);
            mask_data[i * output_size + idx] = 1.0;
        }
        let mask = Tensor::<AutodiffCpuBackend, 2>::from_data(
            TensorData::new(mask_data, [batch_size, output_size]),
            &device,
        );

        // q_taken: [batch, 1]
        let q_taken = (q_all.clone() * mask).sum_dim(1);

        // ── 4. Build target tensor (detached) ─────────────────────────────
        let target_data: Vec<f32> = td_targets.iter().map(|&v| v as f32).collect();
        let targets = Tensor::<AutodiffCpuBackend, 2>::from_data(
            TensorData::new(target_data, [batch_size, 1]),
            &device,
        );

        // ── 5. IS-weighted MSE loss ───────────────────────────────────────
        let weights_data: Vec<f32> = is_weights.iter().map(|&w| w as f32).collect();
        let weights_tensor = Tensor::<AutodiffCpuBackend, 2>::from_data(
            TensorData::new(weights_data, [batch_size, 1]),
            &device,
        );

        let td_error = q_taken.clone() - targets;
        let weighted_sq = td_error.clone().powf_scalar(2.0) * weights_tensor;
        let loss = weighted_sq.mean();

        // ── 6. Extract scalar metrics before backward ─────────────────────
        let loss_value = loss.clone().into_scalar().elem::<f32>() as f64;

        let q_taken_data: Vec<f32> = q_taken.inner().into_data().to_vec().unwrap();
        let mean_q = q_taken_data.iter().map(|&v| v as f64).sum::<f64>() / batch_size as f64;

        let td_err_data: Vec<f32> = td_error.inner().into_data().to_vec().unwrap();
        let max_td_error = td_err_data
            .iter()
            .map(|v| (*v as f64).abs())
            .fold(0.0_f64, f64::max);

        // ── 7. Backward pass ──────────────────────────────────────────────
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.model);

        // ── 8. Optimiser step ─────────────────────────────────────────────
        self.model = self
            .optimizer
            .step(self.learning_rate, self.model.clone(), grads_params);

        self.step_count += 1;

        Ok(DqnStepResult {
            loss: loss_value,
            mean_q,
            max_td_error,
        })
    }

    // -- Target network update -----------------------------------------------

    /// Polyak soft-update of a target [`LstmPredictor`] towards the online
    /// model's current weights.
    ///
    /// `θ_target ← τ · θ_online + (1 − τ) · θ_target`
    ///
    /// Typical values for `tau` are `0.005` or `0.001`.
    pub fn soft_update_target(&self, target: &mut LstmPredictor<CpuBackend>, tau: f64) {
        let inner_model = self.model.valid();
        soft_update_lstm_target(&inner_model, target, tau);
    }

    // -- Inference conversion ------------------------------------------------

    /// Convert the current online model into an inference-ready
    /// [`LstmPredictor`] for serving or checkpointing.
    pub fn to_inference(&self, device: BackendDevice) -> Result<LstmPredictor<CpuBackend>> {
        let inner_model = self.model.valid();
        let inference_config = trainable_lstm_config_to_inference(&self.model_config);
        trainable_lstm_to_predictor(&inner_model, inference_config, device)
    }

    /// Extract the current online model's weights as an inference-compatible
    /// [`WeightMap`].  This can be used for custom serialisation or inspection.
    pub fn online_weights(&self) -> WeightMap {
        let inner = self.model.valid();
        crate::models::convert::extract_trainable_lstm_weights(&inner)
    }

    // -- Learning rate -------------------------------------------------------

    /// Get the current learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    /// Set the learning rate (for external scheduling).
    pub fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr;
    }

    // -- Accessors -----------------------------------------------------------

    /// Read-only access to the underlying trainable model.
    pub fn model(&self) -> &TrainableLstm<AutodiffCpuBackend> {
        &self.model
    }

    /// Read-only access to the model configuration.
    pub fn model_config(&self) -> &TrainableLstmConfig {
        &self.model_config
    }

    // -- Private helpers -----------------------------------------------------

    /// Apply an inference-model [`WeightMap`] to the online trainable model.
    ///
    /// This converts the online model to an inference predictor, applies the
    /// weight map, then extracts and re-applies back to the trainable model.
    /// It's used when bootstrapping from a saved checkpoint.
    fn apply_inference_weights(&mut self, weight_map: &WeightMap) {
        use burn::module::Module;
        use std::collections::HashMap;

        // Index weights by name for O(1) lookup
        let weight_dict: HashMap<String, &crate::models::SerializedTensor> =
            weight_map.iter().map(|t| (t.name.clone(), t)).collect();

        let device = <AutodiffCpuBackend as burn_core::tensor::backend::Backend>::Device::default();

        // --- Helpers for loading records ---

        // Helper: load a raw tensor from the map
        let load_tensor_2d = |name: &str, shape: &[usize]| -> Tensor<AutodiffCpuBackend, 2> {
            let t = weight_dict
                .get(name)
                .unwrap_or_else(|| panic!("Missing 2D weight in checkpoint: {}", name));
            // Verify shape consistency if needed, but for now trust the config match
            let data = TensorData::new(t.data.clone(), shape);
            Tensor::from_data(data, &device)
        };

        let load_tensor_1d = |name: &str, shape: &[usize]| -> Tensor<AutodiffCpuBackend, 1> {
            let t = weight_dict
                .get(name)
                .unwrap_or_else(|| panic!("Missing 1D weight in checkpoint: {}", name));
            let data = TensorData::new(t.data.clone(), shape);
            Tensor::from_data(data, &device)
        };

        // Helper: construct a LinearRecord from weights
        let load_linear_record = |prefix: &str,
                                  linear: &burn_nn::Linear<AutodiffCpuBackend>|
         -> burn_nn::LinearRecord<AutodiffCpuBackend> {
            let weight_name = format!("{}.weight", prefix);
            let bias_name = format!("{}.bias", prefix);

            let weight = load_tensor_2d(&weight_name, &linear.weight.shape().dims);

            let bias = linear
                .bias
                .as_ref()
                .map(|b| Param::from_tensor(load_tensor_1d(&bias_name, &b.shape().dims)));

            burn_nn::LinearRecord {
                weight: Param::from_tensor(weight),
                bias,
            }
        };

        // --- Apply Weights ---

        // 1. Output Layer
        let output_record = load_linear_record("output", &self.model.output);
        self.model.output = self.model.output.clone().load_record(output_record);

        // 2. LSTM Layers
        for (i, lstm) in self.model.lstm_layers.iter_mut().enumerate() {
            let layer_prefix = format!("lstm_layers.{}", i);

            // Helper to load a specific gate (input/forget/output/cell)
            let load_gate_record =
                |gate_name: &str,
                 controller: &burn_nn::GateController<AutodiffCpuBackend>|
                 -> burn_nn::GateControllerRecord<AutodiffCpuBackend> {
                    let gate_prefix = format!("{}.{}", layer_prefix, gate_name);

                    let input_transform = load_linear_record(
                        &format!("{}.input_transform", gate_prefix),
                        &controller.input_transform,
                    );

                    let hidden_transform = load_linear_record(
                        &format!("{}.hidden_transform", gate_prefix),
                        &controller.hidden_transform,
                    );

                    burn_nn::GateControllerRecord {
                        input_transform,
                        hidden_transform,
                    }
                };

            let record = burn_nn::lstm::LstmRecord {
                input_gate: load_gate_record("input_gate", &lstm.input_gate),
                forget_gate: load_gate_record("forget_gate", &lstm.forget_gate),
                output_gate: load_gate_record("output_gate", &lstm.output_gate),
                cell_gate: load_gate_record("cell_gate", &lstm.cell_gate),
                d_hidden: burn_core::module::ConstantRecord,
            };

            *lstm = lstm.clone().load_record(record);
        }
    }
}

// ===========================================================================
// Batch helper — compute TD targets from online + target models
// ===========================================================================

/// Pre-compute Double DQN TD targets for a batch of experiences.
///
/// For each transition `(s, a, r, s', done)`:
///
/// ```text
/// a* = argmax_a Q_online(s', a)          # action selection by online
/// target = r + γ · Q_target(s', a*)      # evaluation by target
///        = r                              # if done
/// ```
///
/// # Arguments
///
/// * `online`       – The online Q-network (for argmax action selection in s').
/// * `target`       – The target Q-network (for evaluating Q(s', a*)).
/// * `next_states`  – Feature vectors for the next states.
/// * `rewards`      – Scalar rewards.
/// * `dones`        – Episode-terminal flags.
/// * `gamma`        – Discount factor.
/// * `input_size`   – Feature dimension per timestep.
/// * `seq_len`      – Sequence length (use 1 for single-step).
///
/// # Returns
///
/// A `Vec<f64>` of TD target values, one per experience.
#[allow(clippy::too_many_arguments)]
pub fn compute_double_dqn_targets(
    online: &DqnOnlineModel,
    target: &LstmPredictor<CpuBackend>,
    next_states: &[Vec<f32>],
    rewards: &[f64],
    dones: &[bool],
    gamma: f64,
    input_size: usize,
    seq_len: usize,
) -> Vec<f64> {
    use burn_core::tensor::backend::Backend;

    let batch_size = next_states.len();
    let device_cpu = <CpuBackend as Backend>::Device::default();

    // ── Online Q(s', ·) for argmax (use inner model, no gradients) ────────
    let inner_model = online.model.valid();
    let online_q_next: Vec<Vec<f32>> = next_states
        .iter()
        .map(|ns| {
            let mut features = ns.clone();
            features.resize(input_size * seq_len, 0.0);
            let t = Tensor::<CpuBackend, 3>::from_data(
                TensorData::new(features, [1, seq_len, input_size]),
                &device_cpu,
            );
            let out = inner_model.forward(t);
            out.to_data().to_vec::<f32>().unwrap()
        })
        .collect();

    // ── Target Q(s', ·) for evaluation ────────────────────────────────────
    let target_q_next: Vec<Vec<f32>> = next_states
        .iter()
        .map(|ns| {
            let mut features = ns.clone();
            features.resize(input_size * seq_len, 0.0);
            let t = Tensor::<CpuBackend, 3>::from_data(
                TensorData::new(features, [1, seq_len, input_size]),
                &device_cpu,
            );
            match target.forward(t) {
                Ok(out) => out.to_data().to_vec::<f32>().unwrap(),
                Err(_) => vec![0.0f32; online.model_config.output_size],
            }
        })
        .collect();

    // ── Compute targets ───────────────────────────────────────────────────
    (0..batch_size)
        .map(|i| {
            if dones[i] {
                rewards[i]
            } else {
                // argmax over online Q-values
                let best_action = online_q_next[i]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                let target_q = target_q_next[i].get(best_action).copied().unwrap_or(0.0) as f64;

                rewards[i] + gamma * target_q
            }
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::BackendDevice;

    fn small_config() -> TrainableLstmConfig {
        TrainableLstmConfig::new(4, 8, 3).with_num_layers(1)
    }

    #[test]
    fn test_dqn_online_model_creation() {
        let online = DqnOnlineModel::new(small_config(), 1e-3, 1e-5);
        assert!(online.is_ok());
        let online = online.unwrap();
        assert_eq!(online.step_count, 0);
        assert!((online.learning_rate() - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_step_updates_step_count() {
        let mut online = DqnOnlineModel::new(small_config(), 1e-3, 0.0).unwrap();

        let states = vec![vec![0.1f32; 4]; 4];
        let actions = vec![0, 1, 2, 0];
        let td_targets = vec![1.0, -0.5, 0.3, 0.0];
        let is_weights = vec![1.0; 4];

        let result = online.gradient_step(&states, &actions, &td_targets, &is_weights, 4, 1);
        assert!(result.is_ok(), "gradient_step failed: {:?}", result.err());

        let metrics = result.unwrap();
        assert!(metrics.loss.is_finite());
        assert!(metrics.mean_q.is_finite());
        assert!(metrics.max_td_error >= 0.0);
        assert_eq!(online.step_count, 1);
    }

    #[test]
    fn test_gradient_step_empty_batch_errors() {
        let mut online = DqnOnlineModel::new(small_config(), 1e-3, 0.0).unwrap();
        let result = online.gradient_step(&[], &[], &[], &[], 4, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_gradient_steps_change_weights() {
        let mut online = DqnOnlineModel::new(small_config(), 1e-2, 0.0).unwrap();

        let w0 = online.online_weights();

        // Run several steps with non-trivial targets
        for _ in 0..5 {
            let states = vec![vec![1.0f32; 4]; 8];
            let actions = vec![0; 8];
            let td_targets = vec![10.0; 8]; // large target to force updates
            let is_weights = vec![1.0; 8];
            let _ = online.gradient_step(&states, &actions, &td_targets, &is_weights, 4, 1);
        }

        let w1 = online.online_weights();

        // At least some weights should have changed
        let any_changed = w0.iter().zip(w1.iter()).any(|(a, b)| {
            a.data
                .iter()
                .zip(b.data.iter())
                .any(|(x, y)| (x - y).abs() > 1e-9)
        });
        assert!(any_changed, "Weights should change after gradient steps");
    }

    #[test]
    fn test_soft_update_target() {
        let cfg = small_config();
        let online = DqnOnlineModel::new(cfg.clone(), 1e-3, 0.0).unwrap();

        let inference_cfg = trainable_lstm_config_to_inference(&cfg);
        let mut target = LstmPredictor::new(inference_cfg, BackendDevice::cpu());

        let target_w_before = target.extract_weights_pub();

        online.soft_update_target(&mut target, 0.5);

        let target_w_after = target.extract_weights_pub();

        // With tau=0.5, weights should be exactly the average of online and
        // original target.  At minimum they should have changed.
        let any_changed = target_w_before
            .iter()
            .zip(target_w_after.iter())
            .any(|(a, b)| {
                a.data
                    .iter()
                    .zip(b.data.iter())
                    .any(|(x, y)| (x - y).abs() > 1e-9)
            });
        assert!(
            any_changed,
            "Target weights should change after soft update"
        );
    }

    #[test]
    fn test_to_inference() {
        let online = DqnOnlineModel::new(small_config(), 1e-3, 0.0).unwrap();
        let predictor = online.to_inference(BackendDevice::cpu());
        assert!(predictor.is_ok());

        let predictor = predictor.unwrap();
        assert_eq!(predictor.config().input_size, 4);
        assert_eq!(predictor.config().hidden_size, 8);
        assert_eq!(predictor.config().output_size, 3);
    }

    #[test]
    fn test_set_learning_rate() {
        let mut online = DqnOnlineModel::new(small_config(), 1e-3, 0.0).unwrap();
        assert!((online.learning_rate() - 1e-3).abs() < 1e-9);
        online.set_learning_rate(5e-4);
        assert!((online.learning_rate() - 5e-4).abs() < 1e-9);
    }

    #[test]
    fn test_compute_double_dqn_targets_terminal() {
        let cfg = small_config();
        let online = DqnOnlineModel::new(cfg.clone(), 1e-3, 0.0).unwrap();
        let inference_cfg = trainable_lstm_config_to_inference(&cfg);
        let target = LstmPredictor::new(inference_cfg, BackendDevice::cpu());

        let next_states = vec![vec![0.0f32; 4]; 2];
        let rewards = vec![5.0, 3.0];
        let dones = vec![true, true];

        let targets = compute_double_dqn_targets(
            &online,
            &target,
            &next_states,
            &rewards,
            &dones,
            0.99,
            4,
            1,
        );

        assert_eq!(targets.len(), 2);
        assert!((targets[0] - 5.0).abs() < 1e-6);
        assert!((targets[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_double_dqn_targets_non_terminal() {
        let cfg = small_config();
        let online = DqnOnlineModel::new(cfg.clone(), 1e-3, 0.0).unwrap();
        let inference_cfg = trainable_lstm_config_to_inference(&cfg);
        let target = LstmPredictor::new(inference_cfg, BackendDevice::cpu());

        let next_states = vec![vec![0.1f32; 4]; 3];
        let rewards = vec![1.0, 2.0, 0.5];
        let dones = vec![false, false, false];

        let targets = compute_double_dqn_targets(
            &online,
            &target,
            &next_states,
            &rewards,
            &dones,
            0.99,
            4,
            1,
        );

        assert_eq!(targets.len(), 3);
        for t in &targets {
            assert!(t.is_finite(), "Target should be finite: {}", t);
        }
    }
}
