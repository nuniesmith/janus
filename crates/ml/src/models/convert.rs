//! Weight conversion utilities between trainable and inference models.
//!
//! This module bridges the training and inference pipelines by providing
//! functions to:
//!
//! 1. Extract a [`WeightMap`] from a trained `TrainableLstm<B>` or
//!    `TrainableMlp<B>` model.
//! 2. Load that [`WeightMap`] into the corresponding inference model
//!    (`LstmPredictor` or `MlpClassifier`).
//! 3. Perform end-to-end conversion from a trainable model directly into
//!    an inference model.
//!
//! # Typical workflow
//!
//! ```rust,ignore
//! use janus_ml::models::convert;
//! use janus_ml::models::trainable::{TrainableLstmConfig, TrainableLstm};
//! use janus_ml::models::{LstmConfig, LstmPredictor};
//! use janus_ml::backend::{AutodiffCpuBackend, BackendDevice};
//!
//! // After training …
//! let trained_model: TrainableLstm<AutodiffCpuBackend> = /* … */;
//!
//! // Strip the autodiff wrapper
//! let inner_model = trained_model.valid();
//!
//! // Build a matching inference config
//! let inference_config = LstmConfig::new(50, 64, 1).with_num_layers(2);
//!
//! // Convert directly
//! let predictor = convert::trainable_lstm_to_predictor(
//!     &inner_model,
//!     inference_config,
//!     BackendDevice::cpu(),
//! );
//! ```

use burn_core::tensor::{Tensor, backend::Backend};
use burn_nn::Linear;
use burn_nn::lstm::Lstm;

use super::{
    LstmConfig, LstmPredictor, MlpClassifier, MlpConfig, Model, SerializedTensor, WeightMap,
};
use crate::backend::{BackendDevice, CpuBackend};
use crate::error::Result;
use crate::models::trainable::{
    TrainableLstm, TrainableLstmConfig, TrainableMlp, TrainableMlpConfig,
};

// ---------------------------------------------------------------------------
// Tensor serialisation helpers (same signatures as in lstm.rs / mlp.rs,
// duplicated here to avoid leaking private helpers across module boundaries)
// ---------------------------------------------------------------------------

/// Extract a 2-D tensor into a [`SerializedTensor`].
fn serialize_tensor_2d<B: Backend>(name: &str, t: &Tensor<B, 2>) -> SerializedTensor {
    let shape: Vec<usize> = t.dims().to_vec();
    let data: Vec<f32> = t.to_data().to_vec().unwrap();
    SerializedTensor {
        name: name.to_string(),
        shape,
        data,
    }
}

/// Extract a 1-D tensor into a [`SerializedTensor`].
fn serialize_tensor_1d<B: Backend>(name: &str, t: &Tensor<B, 1>) -> SerializedTensor {
    let shape: Vec<usize> = t.dims().to_vec();
    let data: Vec<f32> = t.to_data().to_vec().unwrap();
    SerializedTensor {
        name: name.to_string(),
        shape,
        data,
    }
}

// ---------------------------------------------------------------------------
// Low-level extraction helpers
// ---------------------------------------------------------------------------

/// Extract weight and (optional) bias from a [`Linear`] layer.
fn extract_linear_weights<B: Backend>(prefix: &str, linear: &Linear<B>) -> Vec<SerializedTensor> {
    let mut out = Vec::new();
    out.push(serialize_tensor_2d(
        &format!("{}.weight", prefix),
        &linear.weight.val(),
    ));
    if let Some(ref bias) = linear.bias {
        out.push(serialize_tensor_1d(
            &format!("{}.bias", prefix),
            &bias.val(),
        ));
    }
    out
}

/// Gate names in a Burn 0.19 [`Lstm`] layer, in canonical order.
///
/// This **must** match the ordering used by `lstm.rs::extract_uni_lstm_weights`
/// so that inference-model `restore_weights` can find the tensors by name.
const LSTM_GATE_NAMES: &[&str] = &["input_gate", "forget_gate", "output_gate", "cell_gate"];

/// Extract all gate weights from a single unidirectional Burn [`Lstm`].
fn extract_uni_lstm_weights<B: Backend>(prefix: &str, lstm: &Lstm<B>) -> Vec<SerializedTensor> {
    let gates = [
        (&lstm.input_gate, LSTM_GATE_NAMES[0]),
        (&lstm.forget_gate, LSTM_GATE_NAMES[1]),
        (&lstm.output_gate, LSTM_GATE_NAMES[2]),
        (&lstm.cell_gate, LSTM_GATE_NAMES[3]),
    ];

    let mut out = Vec::new();
    for (gate, name) in &gates {
        let gate_prefix = format!("{}.{}", prefix, name);
        out.extend(extract_linear_weights(
            &format!("{}.input_transform", gate_prefix),
            &gate.input_transform,
        ));
        out.extend(extract_linear_weights(
            &format!("{}.hidden_transform", gate_prefix),
            &gate.hidden_transform,
        ));
    }
    out
}

// ===========================================================================
// Public API — TrainableLstm
// ===========================================================================

/// Extract a [`WeightMap`] from a [`TrainableLstm`].
///
/// The resulting weight map uses exactly the same tensor names as those
/// produced by `LstmPredictor::extract_weights`, so it can be fed directly
/// into `LstmPredictor::restore_weights`.
///
/// > **Note:** The trainable model must use the *inner* (non-autodiff) backend.
/// > Call `.valid()` on an `Autodiff<B>` model first if you trained with
/// > autodiff.
pub fn extract_trainable_lstm_weights<B: Backend>(model: &TrainableLstm<B>) -> WeightMap {
    let mut weights = Vec::new();

    for (i, lstm) in model.lstm_layers.iter().enumerate() {
        weights.extend(extract_uni_lstm_weights(
            &format!("lstm_layers.{}", i),
            lstm,
        ));
    }

    weights.extend(extract_linear_weights("output", &model.output));

    weights
}

/// Build an [`LstmConfig`] from a [`TrainableLstmConfig`].
///
/// This produces an inference-side config whose layer sizes and flags match
/// the trainable config so that weight restoration succeeds.
pub fn trainable_lstm_config_to_inference(cfg: &TrainableLstmConfig) -> LstmConfig {
    LstmConfig::new(cfg.input_size, cfg.hidden_size, cfg.output_size)
        .with_num_layers(cfg.num_layers)
        .with_dropout(cfg.dropout)
        .with_bidirectional(cfg.bidirectional)
}

/// One-shot conversion: extract weights from a [`TrainableLstm`] and load
/// them into a fresh [`LstmPredictor`].
///
/// The `inference_config` **must** have compatible layer dimensions
/// (input_size, hidden_size, output_size, num_layers, bidirectional) with the
/// trained model. You can use [`trainable_lstm_config_to_inference`] to derive
/// it automatically.
///
/// # Errors
///
/// Returns an error if the inference model cannot be constructed or if the
/// weight map is structurally incompatible.
pub fn trainable_lstm_to_predictor<B: Backend>(
    model: &TrainableLstm<B>,
    inference_config: LstmConfig,
    device: BackendDevice,
) -> Result<LstmPredictor<CpuBackend>> {
    let weights = extract_trainable_lstm_weights(model);
    let mut predictor = LstmPredictor::new(inference_config, device);
    predictor.apply_weight_map(&weights);
    Ok(predictor)
}

/// Convenience: convert a [`TrainableLstm`] to an [`LstmPredictor`],
/// automatically deriving the inference config from the training config.
pub fn trainable_lstm_to_predictor_auto<B: Backend>(
    model: &TrainableLstm<B>,
    training_config: &TrainableLstmConfig,
    device: BackendDevice,
) -> Result<LstmPredictor<CpuBackend>> {
    let inference_config = trainable_lstm_config_to_inference(training_config);
    trainable_lstm_to_predictor(model, inference_config, device)
}

// ===========================================================================
// Public API — TrainableMlp
// ===========================================================================

/// Extract a [`WeightMap`] from a [`TrainableMlp`].
///
/// The resulting weight map uses the same tensor naming convention as
/// `MlpClassifier::extract_weights`:
///
/// - `hidden_layers.{i}.linear.weight`
/// - `hidden_layers.{i}.linear.bias`
/// - `output.weight`
/// - `output.bias`
///
/// > **Note:** `TrainableMlp` does not support batch-norm, so no batch-norm
/// > tensors are emitted.
pub fn extract_trainable_mlp_weights<B: Backend>(model: &TrainableMlp<B>) -> WeightMap {
    let mut weights = Vec::new();

    for (i, linear) in model.layers.iter().enumerate() {
        weights.extend(extract_linear_weights(
            &format!("hidden_layers.{}.linear", i),
            linear,
        ));
    }

    weights.extend(extract_linear_weights("output", &model.output));

    weights
}

/// Build an [`MlpConfig`] from a [`TrainableMlpConfig`].
pub fn trainable_mlp_config_to_inference(cfg: &TrainableMlpConfig) -> MlpConfig {
    let mut mlp_cfg = MlpConfig::new(cfg.input_size, cfg.hidden_sizes.clone(), cfg.output_size)
        .with_dropout(cfg.dropout);

    // TrainableMlp does not use batch-norm; ensure the inference model
    // matches so it doesn't try to restore batch-norm tensors that don't exist.
    mlp_cfg = mlp_cfg.with_batch_norm(false);

    mlp_cfg
}

/// One-shot conversion: extract weights from a [`TrainableMlp`] and load
/// them into a fresh [`MlpClassifier`].
pub fn trainable_mlp_to_classifier<B: Backend>(
    model: &TrainableMlp<B>,
    inference_config: MlpConfig,
    device: BackendDevice,
) -> Result<MlpClassifier<CpuBackend>> {
    let weights = extract_trainable_mlp_weights(model);
    let mut classifier = MlpClassifier::new(inference_config, device);
    classifier.apply_weight_map(&weights);
    Ok(classifier)
}

/// Convenience: convert a [`TrainableMlp`] to an [`MlpClassifier`],
/// automatically deriving the inference config from the training config.
pub fn trainable_mlp_to_classifier_auto<B: Backend>(
    model: &TrainableMlp<B>,
    training_config: &TrainableMlpConfig,
    device: BackendDevice,
) -> Result<MlpClassifier<CpuBackend>> {
    let inference_config = trainable_mlp_config_to_inference(training_config);
    trainable_mlp_to_classifier(model, inference_config, device)
}

// ===========================================================================
// Polyak (τ-based) soft-update utilities
// ===========================================================================

/// Blend two [`WeightMap`]s using Polyak averaging (soft-update).
///
/// For every tensor present in both maps:
///
///   `w_out = τ * w_online + (1 − τ) * w_target`
///
/// Tensors that exist only in one map are copied through unchanged.
///
/// # Arguments
///
/// * `online_weights`  – Weights from the online (actively trained) network.
/// * `target_weights`  – Weights from the target (slowly tracking) network.
/// * `tau`             – Interpolation coefficient in `[0, 1]`. Typical values
///   are `0.005` or `0.001`.
///
/// # Panics
///
/// Panics if a tensor with the same name has a different shape in the two maps.
pub fn soft_update_weight_maps(
    online_weights: &WeightMap,
    target_weights: &WeightMap,
    tau: f64,
) -> WeightMap {
    use std::collections::HashMap;

    let target_map: HashMap<&str, &SerializedTensor> = target_weights
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    let mut result: Vec<SerializedTensor> = Vec::with_capacity(online_weights.len());

    for online_t in online_weights {
        if let Some(target_t) = target_map.get(online_t.name.as_str()) {
            assert_eq!(
                online_t.shape, target_t.shape,
                "Shape mismatch for '{}': online {:?} vs target {:?}",
                online_t.name, online_t.shape, target_t.shape,
            );

            let blended_data: Vec<f32> = online_t
                .data
                .iter()
                .zip(target_t.data.iter())
                .map(|(&o, &t)| (tau as f32) * o + (1.0 - tau as f32) * t)
                .collect();

            result.push(SerializedTensor {
                name: online_t.name.clone(),
                shape: online_t.shape.clone(),
                data: blended_data,
            });
        } else {
            // Tensor only in online — copy through
            result.push(online_t.clone());
        }
    }

    // Append any tensors that exist only in the target map
    let online_names: std::collections::HashSet<&str> =
        online_weights.iter().map(|t| t.name.as_str()).collect();
    for target_t in target_weights {
        if !online_names.contains(target_t.name.as_str()) {
            result.push(target_t.clone());
        }
    }

    result
}

/// Perform a Polyak soft-update of a target [`LstmPredictor`] towards the
/// weights of a [`TrainableLstm`] (the online network).
///
/// This is the standard DQN target-network update:
///
///   `θ_target ← τ · θ_online + (1 − τ) · θ_target`
///
/// The function extracts weights from both models, blends them with
/// [`soft_update_weight_maps`], and restores the blended weights into the
/// target predictor.
///
/// # Arguments
///
/// * `online` – The online (trainable) model whose weights are being tracked.
///   Must use the *inner* (non-autodiff) backend — call `.valid()`
///   first when working with `Autodiff<B>`.
/// * `target` – The target inference model to update in-place.
/// * `tau`    – Soft-update coefficient in `[0, 1]`.
pub fn soft_update_lstm_target<B: Backend>(
    online: &TrainableLstm<B>,
    target: &mut LstmPredictor<CpuBackend>,
    tau: f64,
) {
    let online_weights = extract_trainable_lstm_weights(online);
    let target_weights = target.extract_weights_pub();
    let blended = soft_update_weight_maps(&online_weights, &target_weights, tau);
    target.apply_weight_map(&blended);
}

/// Perform a Polyak soft-update of a target [`MlpClassifier`] towards the
/// weights of a [`TrainableMlp`].
///
/// See [`soft_update_lstm_target`] for details — this is the MLP equivalent.
pub fn soft_update_mlp_target<B: Backend>(
    online: &TrainableMlp<B>,
    target: &mut MlpClassifier<CpuBackend>,
    tau: f64,
) {
    let online_weights = extract_trainable_mlp_weights(online);
    let target_weights = target.extract_weights_pub();
    let blended = soft_update_weight_maps(&online_weights, &target_weights, tau);
    target.apply_weight_map(&blended);
}

// ===========================================================================
// Weight map validation
// ===========================================================================

/// Validation result for a weight-map compatibility check.
#[derive(Debug, Clone)]
pub struct WeightMapValidation {
    /// Tensor names that exist in the weight map and were expected.
    pub matched: Vec<String>,
    /// Tensor names expected by the model but missing from the weight map.
    pub missing: Vec<String>,
    /// Tensor names in the weight map that the model does not expect.
    pub extra: Vec<String>,
    /// Shape mismatches: `(name, expected_shape, actual_shape)`.
    pub shape_mismatches: Vec<(String, Vec<usize>, Vec<usize>)>,
}

impl WeightMapValidation {
    /// Returns `true` when the weight map is fully compatible (no missing
    /// tensors and no shape mismatches).
    pub fn is_ok(&self) -> bool {
        self.missing.is_empty() && self.shape_mismatches.is_empty()
    }

    /// Build a human-readable summary string.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Matched: {}, Missing: {}, Extra: {}, Shape mismatches: {}",
            self.matched.len(),
            self.missing.len(),
            self.extra.len(),
            self.shape_mismatches.len(),
        ));
        for name in &self.missing {
            lines.push(format!("  MISSING: {}", name));
        }
        for (name, expected, actual) in &self.shape_mismatches {
            lines.push(format!(
                "  SHAPE MISMATCH: {} expected {:?} got {:?}",
                name, expected, actual
            ));
        }
        for name in &self.extra {
            lines.push(format!("  EXTRA (unused): {}", name));
        }
        lines.join("\n")
    }
}

/// Validate that a [`WeightMap`] is compatible with an [`LstmPredictor`]
/// built from the given config (without actually constructing the model).
///
/// This is useful as a pre-flight check before attempting to restore weights.
pub fn validate_lstm_weight_map(
    weight_map: &WeightMap,
    config: &LstmConfig,
) -> WeightMapValidation {
    let expected_names = build_expected_lstm_tensor_names(config);
    validate_weight_map_inner(weight_map, &expected_names)
}

/// Validate that a [`WeightMap`] is compatible with an [`MlpClassifier`]
/// built from the given config.
pub fn validate_mlp_weight_map(weight_map: &WeightMap, config: &MlpConfig) -> WeightMapValidation {
    let expected_names = build_expected_mlp_tensor_names(config);
    validate_weight_map_inner(weight_map, &expected_names)
}

// ---------------------------------------------------------------------------
// Internal helpers for validation
// ---------------------------------------------------------------------------

fn validate_weight_map_inner(
    weight_map: &WeightMap,
    expected_names: &[String],
) -> WeightMapValidation {
    use std::collections::HashSet;

    let wm_names: HashSet<&str> = weight_map.iter().map(|st| st.name.as_str()).collect();
    let exp_set: HashSet<&str> = expected_names.iter().map(|s| s.as_str()).collect();

    let matched: Vec<String> = expected_names
        .iter()
        .filter(|n| wm_names.contains(n.as_str()))
        .cloned()
        .collect();

    let missing: Vec<String> = expected_names
        .iter()
        .filter(|n| !wm_names.contains(n.as_str()))
        .cloned()
        .collect();

    let extra: Vec<String> = weight_map
        .iter()
        .filter(|st| !exp_set.contains(st.name.as_str()))
        .map(|st| st.name.clone())
        .collect();

    // We cannot check shapes without building the model, so leave empty.
    let shape_mismatches = Vec::new();

    WeightMapValidation {
        matched,
        missing,
        extra,
        shape_mismatches,
    }
}

/// Produce the list of tensor names that an `LstmPredictor` with the given
/// config would expect.
fn build_expected_lstm_tensor_names(config: &LstmConfig) -> Vec<String> {
    let mut names = Vec::new();

    for layer_idx in 0..config.num_layers {
        let layer_prefix = format!("lstm_layers.{}", layer_idx);

        if config.bidirectional {
            for dir in &["forward", "reverse"] {
                let dir_prefix = format!("{}.{}", layer_prefix, dir);
                names.extend(lstm_gate_tensor_names(&dir_prefix));
            }
        } else {
            names.extend(lstm_gate_tensor_names(&layer_prefix));
        }
    }

    // Output linear layer
    names.push("output.weight".to_string());
    names.push("output.bias".to_string());

    names
}

/// Tensor names for a single unidirectional LSTM (4 gates × 2 transforms × weight+bias).
fn lstm_gate_tensor_names(prefix: &str) -> Vec<String> {
    let mut names = Vec::new();
    for gate in LSTM_GATE_NAMES {
        for transform in &["input_transform", "hidden_transform"] {
            names.push(format!("{}.{}.{}.weight", prefix, gate, transform));
            names.push(format!("{}.{}.{}.bias", prefix, gate, transform));
        }
    }
    names
}

/// Produce the list of tensor names that an `MlpClassifier` with the given
/// config would expect (batch-norm tensors only if `config.batch_norm` is
/// true).
fn build_expected_mlp_tensor_names(config: &MlpConfig) -> Vec<String> {
    let mut names = Vec::new();

    for i in 0..config.hidden_sizes.len() {
        names.push(format!("hidden_layers.{}.linear.weight", i));
        names.push(format!("hidden_layers.{}.linear.bias", i));

        if config.batch_norm {
            names.push(format!("hidden_layers.{}.batch_norm.gamma", i));
            names.push(format!("hidden_layers.{}.batch_norm.beta", i));
            names.push(format!("hidden_layers.{}.batch_norm.running_mean", i));
            names.push(format!("hidden_layers.{}.batch_norm.running_var", i));
        }
    }

    names.push("output.weight".to_string());
    names.push("output.bias".to_string());

    names
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_core::module::AutodiffModule;
    use burn_core::tensor::TensorData;
    use burn_ndarray::NdArray;

    type InnerBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<InnerBackend>;

    // ── LSTM conversion tests ─────────────────────────────────────────

    #[test]
    fn test_extract_trainable_lstm_weights_produces_nonempty_map() {
        let config = TrainableLstmConfig::new(10, 16, 1).with_num_layers(1);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableLstm<InnerBackend> = config.init(&device);

        let weights = extract_trainable_lstm_weights(&model);

        // 1 layer × 4 gates × 2 transforms × 2 (weight+bias) + output weight + bias
        // = 16 + 2 = 18
        assert!(!weights.is_empty(), "Weight map should not be empty");
        assert_eq!(
            weights.len(),
            18,
            "Expected 18 tensors for a 1-layer uni-LSTM"
        );
    }

    #[test]
    fn test_extract_trainable_lstm_weights_multi_layer() {
        let config = TrainableLstmConfig::new(10, 16, 1).with_num_layers(3);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableLstm<InnerBackend> = config.init(&device);

        let weights = extract_trainable_lstm_weights(&model);

        // 3 layers × 16 + 2 = 50
        assert_eq!(weights.len(), 50);
    }

    #[test]
    fn test_trainable_lstm_config_to_inference_roundtrip() {
        let tcfg = TrainableLstmConfig::new(50, 64, 1)
            .with_num_layers(2)
            .with_dropout(0.3)
            .with_bidirectional(true);

        let icfg = trainable_lstm_config_to_inference(&tcfg);

        assert_eq!(icfg.input_size, 50);
        assert_eq!(icfg.hidden_size, 64);
        assert_eq!(icfg.output_size, 1);
        assert_eq!(icfg.num_layers, 2);
        assert!((icfg.dropout - 0.3).abs() < f64::EPSILON);
        assert!(icfg.bidirectional);
    }

    #[test]
    fn test_trainable_lstm_to_predictor_forward_matches() {
        let tcfg = TrainableLstmConfig::new(10, 16, 1).with_num_layers(1);
        let device_inner = <InnerBackend as Backend>::Device::default();
        let model: TrainableLstm<InnerBackend> = tcfg.init(&device_inner);

        // Run forward through the trainable model
        let input_data: Vec<f32> = (0..60).map(|i| i as f32 * 0.01).collect();
        let input_t = burn_core::tensor::Tensor::<InnerBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [2, 3, 10]),
            &device_inner,
        );
        let trainable_output = model.forward(input_t);
        let trainable_vals: Vec<f32> = trainable_output.to_data().to_vec().unwrap();

        // Convert to inference model
        let backend_device = BackendDevice::cpu();
        let predictor = trainable_lstm_to_predictor_auto(&model, &tcfg, backend_device)
            .expect("Conversion should succeed");

        // Run forward through the inference model
        let cpu_device = <CpuBackend as Backend>::Device::default();
        let input_cpu = burn_core::tensor::Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [2, 3, 10]),
            &cpu_device,
        );
        let inference_output = predictor
            .forward(input_cpu)
            .expect("Forward pass should succeed");
        let inference_vals: Vec<f32> = inference_output.to_data().to_vec().unwrap();

        // Outputs should be very close (NdArray == CpuBackend in our setup)
        assert_eq!(
            trainable_vals.len(),
            inference_vals.len(),
            "Output shapes must match"
        );
        for (a, b) in trainable_vals.iter().zip(inference_vals.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Output mismatch: trainable={} vs inference={}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_trainable_lstm_autodiff_to_predictor() {
        // Train with autodiff backend, then convert via .valid()
        let tcfg = TrainableLstmConfig::new(8, 12, 1).with_num_layers(1);
        let device_ad = <TestAutodiffBackend as Backend>::Device::default();
        let ad_model: TrainableLstm<TestAutodiffBackend> = tcfg.init(&device_ad);

        // Strip autodiff
        let inner_model = ad_model.valid();

        let backend_device = BackendDevice::cpu();
        let predictor = trainable_lstm_to_predictor_auto(&inner_model, &tcfg, backend_device)
            .expect("Conversion should succeed");

        // Sanity: forward pass should work
        let cpu_device = <CpuBackend as Backend>::Device::default();
        let input = burn_core::tensor::Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.1_f32; 24], [1, 3, 8]),
            &cpu_device,
        );
        let output = predictor.forward(input).expect("Forward should work");
        assert_eq!(output.dims(), [1, 1]);
    }

    // ── MLP conversion tests ──────────────────────────────────────────

    #[test]
    fn test_extract_trainable_mlp_weights_produces_nonempty_map() {
        let config = TrainableMlpConfig::new(10, vec![16, 8], 3);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableMlp<InnerBackend> = config.init(&device);

        let weights = extract_trainable_mlp_weights(&model);

        // 2 hidden layers × 2 (weight+bias) + output layer × 2 = 6
        assert_eq!(weights.len(), 6);
    }

    #[test]
    fn test_trainable_mlp_config_to_inference() {
        let tcfg = TrainableMlpConfig::new(50, vec![128, 64], 3).with_dropout(0.4);
        let icfg = trainable_mlp_config_to_inference(&tcfg);

        assert_eq!(icfg.input_size, 50);
        assert_eq!(icfg.hidden_sizes, vec![128, 64]);
        assert_eq!(icfg.output_size, 3);
        assert!((icfg.dropout - 0.4).abs() < f64::EPSILON);
        assert!(
            !icfg.batch_norm,
            "Batch norm should be off for trainable→inference"
        );
    }

    #[test]
    fn test_trainable_mlp_to_classifier_forward_matches() {
        let tcfg = TrainableMlpConfig::new(10, vec![16], 3);
        let device_inner = <InnerBackend as Backend>::Device::default();
        let model: TrainableMlp<InnerBackend> = tcfg.init(&device_inner);

        // Forward through trainable
        let input_data: Vec<f32> = (0..20).map(|i| i as f32 * 0.05).collect();
        let input_t = burn_core::tensor::Tensor::<InnerBackend, 2>::from_data(
            TensorData::new(input_data.clone(), [2, 10]),
            &device_inner,
        );
        let trainable_output = model.forward(input_t);
        let trainable_vals: Vec<f32> = trainable_output.to_data().to_vec().unwrap();

        // Convert to inference
        let backend_device = BackendDevice::cpu();
        let classifier = trainable_mlp_to_classifier_auto(&model, &tcfg, backend_device)
            .expect("Conversion should succeed");

        // Forward through inference (MlpClassifier::forward takes 3-D input)
        let cpu_device = <CpuBackend as Backend>::Device::default();
        let input_2d = burn_core::tensor::Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(input_data, [2, 10]),
            &cpu_device,
        );
        let inference_output = classifier.forward_2d(input_2d);
        let inference_vals: Vec<f32> = inference_output.to_data().to_vec().unwrap();

        assert_eq!(trainable_vals.len(), inference_vals.len());
        for (a, b) in trainable_vals.iter().zip(inference_vals.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "Output mismatch: trainable={} vs inference={}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_trainable_mlp_autodiff_to_classifier() {
        let tcfg = TrainableMlpConfig::new(8, vec![12], 2);
        let device_ad = <TestAutodiffBackend as Backend>::Device::default();
        let ad_model: TrainableMlp<TestAutodiffBackend> = tcfg.init(&device_ad);

        let inner_model = ad_model.valid();

        let backend_device = BackendDevice::cpu();
        let classifier = trainable_mlp_to_classifier_auto(&inner_model, &tcfg, backend_device)
            .expect("Conversion should succeed");

        let cpu_device = <CpuBackend as Backend>::Device::default();
        let input = burn_core::tensor::Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(vec![0.1_f32; 8], [1, 8]),
            &cpu_device,
        );
        let output = classifier.forward_2d(input);
        assert_eq!(output.dims(), [1, 2]);
    }

    // ── Validation tests ──────────────────────────────────────────────

    #[test]
    fn test_validate_lstm_weight_map_ok() {
        let tcfg = TrainableLstmConfig::new(10, 16, 1).with_num_layers(1);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableLstm<InnerBackend> = tcfg.init(&device);

        let weights = extract_trainable_lstm_weights(&model);
        let icfg = trainable_lstm_config_to_inference(&tcfg);
        let validation = validate_lstm_weight_map(&weights, &icfg);

        assert!(
            validation.is_ok(),
            "Validation should pass: {}",
            validation.summary()
        );
        assert!(validation.missing.is_empty());
        assert!(validation.extra.is_empty());
    }

    #[test]
    fn test_validate_lstm_weight_map_missing() {
        let icfg = LstmConfig::new(10, 16, 1).with_num_layers(2);
        let empty_weights: WeightMap = Vec::new();
        let validation = validate_lstm_weight_map(&empty_weights, &icfg);

        assert!(!validation.is_ok());
        assert!(!validation.missing.is_empty());
    }

    #[test]
    fn test_validate_mlp_weight_map_ok() {
        let tcfg = TrainableMlpConfig::new(10, vec![16, 8], 3);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableMlp<InnerBackend> = tcfg.init(&device);

        let weights = extract_trainable_mlp_weights(&model);
        let icfg = trainable_mlp_config_to_inference(&tcfg);
        let validation = validate_mlp_weight_map(&weights, &icfg);

        assert!(
            validation.is_ok(),
            "Validation should pass: {}",
            validation.summary()
        );
    }

    #[test]
    fn test_validate_mlp_weight_map_with_extra_tensors() {
        let tcfg = TrainableMlpConfig::new(10, vec![16], 3);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableMlp<InnerBackend> = tcfg.init(&device);

        let mut weights = extract_trainable_mlp_weights(&model);
        // Add an extra tensor that the model won't expect
        weights.push(SerializedTensor {
            name: "phantom.weight".to_string(),
            shape: vec![1],
            data: vec![0.0],
        });

        let icfg = trainable_mlp_config_to_inference(&tcfg);
        let validation = validate_mlp_weight_map(&weights, &icfg);

        // Should still be "ok" (extra tensors are a warning, not a failure)
        assert!(validation.is_ok());
        assert_eq!(validation.extra.len(), 1);
        assert_eq!(validation.extra[0], "phantom.weight");
    }

    // ── Roundtrip: save converted model, reload, verify ───────────────

    #[test]
    fn test_lstm_convert_save_load_roundtrip() {
        let tcfg = TrainableLstmConfig::new(6, 8, 1).with_num_layers(1);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableLstm<InnerBackend> = tcfg.init(&device);

        let backend_device = BackendDevice::cpu();
        let predictor = trainable_lstm_to_predictor_auto(&model, &tcfg, backend_device.clone())
            .expect("Conversion should succeed");

        // Forward pass before save
        let cpu_device = <CpuBackend as Backend>::Device::default();
        let input_data: Vec<f32> = (0..18).map(|i| i as f32 * 0.1).collect();
        let input = burn_core::tensor::Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data.clone(), [1, 3, 6]),
            &cpu_device,
        );
        let output_before: Vec<f32> = predictor
            .forward(input)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();

        // Save
        let tmp = std::env::temp_dir().join("janus_ml_convert_test_lstm.bin");
        predictor.save(&tmp).expect("Save should succeed");

        // Load into a fresh predictor
        let loaded = LstmPredictor::load(&tmp, backend_device).expect("Load should succeed");

        let input2 = burn_core::tensor::Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(input_data, [1, 3, 6]),
            &cpu_device,
        );
        let output_after: Vec<f32> = loaded.forward(input2).unwrap().to_data().to_vec().unwrap();

        for (a, b) in output_before.iter().zip(output_after.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Save/load roundtrip mismatch: {} vs {}",
                a,
                b
            );
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_mlp_convert_save_load_roundtrip() {
        let tcfg = TrainableMlpConfig::new(6, vec![8], 2);
        let device = <InnerBackend as Backend>::Device::default();
        let model: TrainableMlp<InnerBackend> = tcfg.init(&device);

        let backend_device = BackendDevice::cpu();
        let classifier = trainable_mlp_to_classifier_auto(&model, &tcfg, backend_device.clone())
            .expect("Conversion should succeed");

        let cpu_device = <CpuBackend as Backend>::Device::default();
        let input_data: Vec<f32> = (0..6).map(|i| i as f32 * 0.2).collect();
        let input = burn_core::tensor::Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(input_data.clone(), [1, 6]),
            &cpu_device,
        );
        let output_before: Vec<f32> = classifier.forward_2d(input).to_data().to_vec().unwrap();

        let tmp = std::env::temp_dir().join("janus_ml_convert_test_mlp.bin");
        classifier.save(&tmp).expect("Save should succeed");

        let loaded = MlpClassifier::load(&tmp, backend_device).expect("Load should succeed");

        let input2 = burn_core::tensor::Tensor::<CpuBackend, 2>::from_data(
            TensorData::new(input_data, [1, 6]),
            &cpu_device,
        );
        let output_after: Vec<f32> = loaded.forward_2d(input2).to_data().to_vec().unwrap();

        for (a, b) in output_before.iter().zip(output_after.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Save/load roundtrip mismatch: {} vs {}",
                a,
                b
            );
        }

        let _ = std::fs::remove_file(&tmp);
    }

    // ───────────────────────────────────────────────────────────────────
    // Soft-update tests
    // ───────────────────────────────────────────────────────────────────

    #[test]
    fn test_soft_update_weight_maps_tau_zero_keeps_target() {
        // tau=0 means the result should equal the target weights exactly.
        let online = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![2, 3],
            data: vec![10.0; 6],
        }];
        let target = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![2, 3],
            data: vec![1.0; 6],
        }];

        let result = soft_update_weight_maps(&online, &target, 0.0);
        assert_eq!(result.len(), 1);
        for v in &result[0].data {
            assert!(
                (*v - 1.0).abs() < 1e-7,
                "tau=0 should keep target: got {}",
                v
            );
        }
    }

    #[test]
    fn test_soft_update_weight_maps_tau_one_copies_online() {
        // tau=1 means the result should equal the online weights exactly.
        let online = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![2, 3],
            data: vec![10.0; 6],
        }];
        let target = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![2, 3],
            data: vec![1.0; 6],
        }];

        let result = soft_update_weight_maps(&online, &target, 1.0);
        assert_eq!(result.len(), 1);
        for v in &result[0].data {
            assert!(
                (*v - 10.0).abs() < 1e-5,
                "tau=1 should copy online: got {}",
                v
            );
        }
    }

    #[test]
    fn test_soft_update_weight_maps_tau_half_averages() {
        let online = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![4],
            data: vec![0.0, 2.0, 4.0, 6.0],
        }];
        let target = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![4],
            data: vec![10.0, 8.0, 6.0, 4.0],
        }];

        let result = soft_update_weight_maps(&online, &target, 0.5);
        let expected = [5.0, 5.0, 5.0, 5.0];
        for (got, exp) in result[0].data.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-5,
                "tau=0.5 should average: got {} expected {}",
                got,
                exp
            );
        }
    }

    #[test]
    fn test_soft_update_weight_maps_preserves_extra_tensors() {
        let online = vec![
            SerializedTensor {
                name: "shared".to_string(),
                shape: vec![2],
                data: vec![1.0, 1.0],
            },
            SerializedTensor {
                name: "online_only".to_string(),
                shape: vec![1],
                data: vec![99.0],
            },
        ];
        let target = vec![
            SerializedTensor {
                name: "shared".to_string(),
                shape: vec![2],
                data: vec![0.0, 0.0],
            },
            SerializedTensor {
                name: "target_only".to_string(),
                shape: vec![1],
                data: vec![42.0],
            },
        ];

        let result = soft_update_weight_maps(&online, &target, 0.5);

        let names: Vec<&str> = result.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"shared"));
        assert!(names.contains(&"online_only"));
        assert!(names.contains(&"target_only"));
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_soft_update_weight_maps_shape_mismatch_panics() {
        let online = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![2, 3],
            data: vec![1.0; 6],
        }];
        let target = vec![SerializedTensor {
            name: "w".to_string(),
            shape: vec![3, 2],
            data: vec![1.0; 6],
        }];

        let _ = soft_update_weight_maps(&online, &target, 0.5);
    }

    #[test]
    fn test_soft_update_lstm_target_changes_weights() {
        let device = <InnerBackend as burn_core::tensor::backend::Backend>::Device::default();
        let cfg = TrainableLstmConfig::new(4, 8, 2).with_num_layers(1);

        let trainable: TrainableLstm<InnerBackend> = cfg.init(&device);

        let inference_cfg = trainable_lstm_config_to_inference(&cfg);
        let bdev = crate::backend::BackendDevice::cpu();
        let mut target = LstmPredictor::new(inference_cfg, bdev);

        let weights_before = target.extract_weights_pub();

        soft_update_lstm_target(&trainable, &mut target, 0.5);

        let weights_after = target.extract_weights_pub();

        // With tau=0.5 and independently initialised models, at least some
        // weights should differ.
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
            "Target weights should change after soft_update_lstm_target"
        );
    }

    #[test]
    fn test_soft_update_lstm_target_tau_zero_preserves_target() {
        let device = <InnerBackend as burn_core::tensor::backend::Backend>::Device::default();
        let cfg = TrainableLstmConfig::new(4, 8, 2).with_num_layers(1);

        let trainable: TrainableLstm<InnerBackend> = cfg.init(&device);

        let inference_cfg = trainable_lstm_config_to_inference(&cfg);
        let bdev = crate::backend::BackendDevice::cpu();
        let mut target = LstmPredictor::new(inference_cfg, bdev);

        let weights_before = target.extract_weights_pub();

        soft_update_lstm_target(&trainable, &mut target, 0.0);

        let weights_after = target.extract_weights_pub();

        // tau=0 should leave target weights completely unchanged
        for (a, b) in weights_before.iter().zip(weights_after.iter()) {
            assert_eq!(a.name, b.name);
            for (x, y) in a.data.iter().zip(b.data.iter()) {
                assert!(
                    (x - y).abs() < 1e-7,
                    "tau=0 should preserve target weights for '{}': {} vs {}",
                    a.name,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn test_soft_update_lstm_target_tau_one_copies_online() {
        let device = <InnerBackend as burn_core::tensor::backend::Backend>::Device::default();
        let cfg = TrainableLstmConfig::new(4, 8, 2).with_num_layers(1);

        let trainable: TrainableLstm<InnerBackend> = cfg.init(&device);

        let inference_cfg = trainable_lstm_config_to_inference(&cfg);
        let bdev = crate::backend::BackendDevice::cpu();
        let mut target = LstmPredictor::new(inference_cfg, bdev);

        // tau=1.0 → target should become identical to the online model
        soft_update_lstm_target(&trainable, &mut target, 1.0);

        let online_w = extract_trainable_lstm_weights(&trainable);
        let target_w = target.extract_weights_pub();

        for (ow, tw) in online_w.iter().zip(target_w.iter()) {
            assert_eq!(ow.name, tw.name);
            for (x, y) in ow.data.iter().zip(tw.data.iter()) {
                assert!(
                    (x - y).abs() < 1e-5,
                    "tau=1 should copy online→target for '{}': {} vs {}",
                    ow.name,
                    x,
                    y
                );
            }
        }
    }
}
