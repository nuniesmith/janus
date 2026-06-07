//! `MasterCnn` — burn-native reimplementation of the Python `MasterCNN`.
//!
//! Portfolio-level risk aggregator (RUST_MIGRATION.md Phase 2, second model).
//! Mirrors `fks-full/src/ruby/src/ml/model.py::MasterCNN`: it encodes one
//! feature window per tracked asset through the **shared** [`AssetEncoder`]
//! (same backbone as `PerAssetCnn`, separate weights), lets the asset
//! embeddings attend to each other via cross-asset multi-head attention, then
//! reduces to a single portfolio risk logit.
//!
//! ```text
//! input (B, n_assets, n_features=20, window=60)
//!   per-asset AssetEncoder (shared weights) → (B, n_assets, embedding_dim)
//!   CrossAssetAttention:
//!     y = x + MHA(LayerNorm(x))            # self-attention over assets
//!     y = y + FFN(LayerNorm2(y))           # Linear→GELU→Linear
//!   flatten (B, n_assets*embedding_dim)
//!   risk head: Linear(→96)→ReLU→Linear(→24)→ReLU→Linear(→1)
//! output risk logit (B, 1)  →  sigmoid → risk score [0,1]
//! ```
//!
//! Dropout is a no-op at inference and omitted. The cross-attention is built
//! from `burn` primitives (separate q/k/v/out `Linear` layers, scaled dot
//! product, softmax) so it matches PyTorch `nn.MultiheadAttention` semantics
//! exactly and the PyTorch→burn weight mapping is a simple `in_proj` split.
//!
//! Parity: [`reference_risk`] is an independent raw-`f32` implementation of the
//! whole forward pass (reusing [`reference_encoder`]); `tests::burn_matches_reference`
//! differential-tests the `burn` model against it.

use std::collections::HashMap;

use burn_core::tensor::{Tensor, activation, backend::Backend};
use burn_nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use serde::{Deserialize, Serialize};

use super::per_asset_cnn::{
    AssetEncoder, load_record, reference_encoder, save_record, set_linear_b, set_linear_w,
    set_param1, st1, st2,
};
use super::{SerializedTensor, WeightMap};
use crate::error::Result;
use std::path::Path;

/// LayerNorm / numerical epsilon (matches PyTorch / burn default).
const LN_EPS: f32 = 1e-5;

/// Configuration for [`MasterCnn`]. Mirrors the Python constructor args.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MasterCnnConfig {
    /// Number of tracked assets (slots; pass zeros for inactive ones).
    pub n_assets: usize,
    /// Input feature channels per asset window.
    pub n_features: usize,
    /// Candle window length.
    pub window: usize,
    /// Encoder embedding width (must be divisible by `n_heads`).
    pub embedding_dim: usize,
    /// Cross-asset attention heads.
    pub n_heads: usize,
}

impl Default for MasterCnnConfig {
    fn default() -> Self {
        Self {
            n_assets: 5,
            n_features: 20,
            window: 60,
            embedding_dim: 64,
            n_heads: 4,
        }
    }
}

impl MasterCnnConfig {
    fn combined_dim(&self) -> usize {
        self.n_assets * self.embedding_dim
    }
}

/// Exact (erf-based) GELU, computed explicitly so the raw-`f32` reference can
/// reproduce it bit-closely. Matches PyTorch `nn.GELU()` default.
fn gelu_exact<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let half = x.clone().mul_scalar(0.5);
    let inner = x.div_scalar(std::f32::consts::SQRT_2).erf().add_scalar(1.0);
    half.mul(inner)
}

// ---------------------------------------------------------------------------
// Cross-asset multi-head self-attention (pre-norm transformer block)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct CrossAssetAttention<B: Backend> {
    norm: LayerNorm<B>,
    q: Linear<B>,
    k: Linear<B>,
    v: Linear<B>,
    out: Linear<B>,
    norm2: LayerNorm<B>,
    ffn_fc1: Linear<B>, // (emb -> 2*emb)
    ffn_fc2: Linear<B>, // (2*emb -> emb)
    n_heads: usize,
}

impl<B: Backend> CrossAssetAttention<B> {
    fn new(embedding_dim: usize, n_heads: usize, device: &B::Device) -> Self {
        let lin = |i, o| LinearConfig::new(i, o).init(device);
        Self {
            norm: LayerNormConfig::new(embedding_dim).init(device),
            q: lin(embedding_dim, embedding_dim),
            k: lin(embedding_dim, embedding_dim),
            v: lin(embedding_dim, embedding_dim),
            out: lin(embedding_dim, embedding_dim),
            norm2: LayerNormConfig::new(embedding_dim).init(device),
            ffn_fc1: lin(embedding_dim, embedding_dim * 2),
            ffn_fc2: lin(embedding_dim * 2, embedding_dim),
            n_heads,
        }
    }

    /// `x`: (batch, n_assets, embedding_dim) → attended (same shape).
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, e] = x.dims();
        let h = self.n_heads;
        let dh = e / h;

        // self-attention on the pre-norm input
        let normed = self.norm.forward(x.clone());
        let split = |t: Tensor<B, 3>| t.reshape([b, s, h, dh]).swap_dims(1, 2); // (b,h,s,dh)
        let q = split(self.q.forward(normed.clone()));
        let k = split(self.k.forward(normed.clone()));
        let v = split(self.v.forward(normed));
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mul_scalar(1.0 / (dh as f32).sqrt()); // (b,h,s,s)
        let attn = activation::softmax(scores, 3);
        let context = attn.matmul(v).swap_dims(1, 2).reshape([b, s, e]); // (b,s,e)
        let x = x.add(self.out.forward(context)); // residual

        // position-wise feed-forward on the pre-norm input
        let ffn = self.ffn_fc2.forward(gelu_exact(
            self.ffn_fc1.forward(self.norm2.forward(x.clone())),
        ));
        x.add(ffn) // residual
    }
}

// ---------------------------------------------------------------------------
// MasterCnn
// ---------------------------------------------------------------------------

/// Portfolio-level risk aggregator (`burn`-native).
#[derive(Debug)]
pub struct MasterCnn<B: Backend> {
    encoder: AssetEncoder<B>,
    cross_attn: CrossAssetAttention<B>,
    risk_fc1: Linear<B>, // (n_assets*emb -> 96)
    risk_fc2: Linear<B>, // (96 -> 24)
    risk_fc3: Linear<B>, // (24 -> 1)
    config: MasterCnnConfig,
}

impl<B: Backend> MasterCnn<B> {
    /// Build a freshly-initialised model on `device`.
    pub fn new(config: MasterCnnConfig, device: &B::Device) -> Self {
        assert!(
            config.embedding_dim.is_multiple_of(config.n_heads),
            "embedding_dim ({}) must be divisible by n_heads ({})",
            config.embedding_dim,
            config.n_heads
        );
        let encoder = AssetEncoder::new(config.n_features, config.embedding_dim, device);
        let cross_attn = CrossAssetAttention::new(config.embedding_dim, config.n_heads, device);
        Self {
            encoder,
            cross_attn,
            risk_fc1: LinearConfig::new(config.combined_dim(), 96).init(device),
            risk_fc2: LinearConfig::new(96, 24).init(device),
            risk_fc3: LinearConfig::new(24, 1).init(device),
            config,
        }
    }

    /// Model configuration.
    pub fn config(&self) -> &MasterCnnConfig {
        &self.config
    }

    /// Forward pass. `input`: (batch, n_assets, n_features, window) → risk
    /// logits (batch, 1). Apply sigmoid for the [0,1] risk score.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [b, a, nf, w] = input.dims();
        let emb_dim = self.config.embedding_dim;
        // Encode every asset slice with the shared encoder (BatchNorm inference
        // is batch-independent, so a flat (B*A) pass equals the per-asset loop).
        let flat = input.reshape([b * a, nf, w]);
        let embeddings = self.encoder.forward(flat).reshape([b, a, emb_dim]); // (B,A,E)
        let attended = self.cross_attn.forward(embeddings); // (B,A,E)
        let combined = attended.reshape([b, a * emb_dim]); // (B, A*E)
        let hidden = activation::relu(self.risk_fc1.forward(combined));
        let hidden = activation::relu(self.risk_fc2.forward(hidden));
        self.risk_fc3.forward(hidden) // (B,1)
    }

    /// Sigmoid risk score(s) in [0,1].
    pub fn predict_risk(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        activation::sigmoid(self.forward(input))
    }

    // -- weight extract / inject (burn-native layout) ------------------------

    /// Serialise every learnable tensor into a [`WeightMap`].
    pub fn extract_weights(&self) -> WeightMap {
        let mut w = Vec::new();
        self.encoder.extract_into(&mut w);

        let ca = &self.cross_attn;
        w.push(st1("cross_attn.norm.gamma", &ca.norm.gamma.val()));
        w.push(st1("cross_attn.norm.beta", &ca.norm.beta.val()));
        for (name, lin) in [("q", &ca.q), ("k", &ca.k), ("v", &ca.v), ("out", &ca.out)] {
            w.push(st2(&format!("cross_attn.{name}.weight"), &lin.weight.val()));
            w.push(st1(
                &format!("cross_attn.{name}.bias"),
                &lin.bias.as_ref().unwrap().val(),
            ));
        }
        w.push(st1("cross_attn.norm2.gamma", &ca.norm2.gamma.val()));
        w.push(st1("cross_attn.norm2.beta", &ca.norm2.beta.val()));
        for (name, lin) in [("fc1", &ca.ffn_fc1), ("fc2", &ca.ffn_fc2)] {
            w.push(st2(
                &format!("cross_attn.ffn.{name}.weight"),
                &lin.weight.val(),
            ));
            w.push(st1(
                &format!("cross_attn.ffn.{name}.bias"),
                &lin.bias.as_ref().unwrap().val(),
            ));
        }
        for (name, lin) in [
            ("fc1", &self.risk_fc1),
            ("fc2", &self.risk_fc2),
            ("fc3", &self.risk_fc3),
        ] {
            w.push(st2(&format!("risk_head.{name}.weight"), &lin.weight.val()));
            w.push(st1(
                &format!("risk_head.{name}.bias"),
                &lin.bias.as_ref().unwrap().val(),
            ));
        }
        w
    }

    /// Overwrite parameters from a [`WeightMap`] in burn-native layout.
    pub fn apply_weights(&mut self, weights: &WeightMap, device: &B::Device) {
        let map: HashMap<&str, &SerializedTensor> =
            weights.iter().map(|st| (st.name.as_str(), st)).collect();
        self.encoder.apply(&map, device);

        let ca = &mut self.cross_attn;
        set_param1(&mut ca.norm.gamma, map.get("cross_attn.norm.gamma"), device);
        set_param1(&mut ca.norm.beta, map.get("cross_attn.norm.beta"), device);
        for (name, lin) in [
            ("q", &mut ca.q),
            ("k", &mut ca.k),
            ("v", &mut ca.v),
            ("out", &mut ca.out),
        ] {
            set_linear_w(
                lin,
                map.get(format!("cross_attn.{name}.weight").as_str()),
                device,
            );
            set_linear_b(
                lin,
                map.get(format!("cross_attn.{name}.bias").as_str()),
                device,
            );
        }
        set_param1(
            &mut ca.norm2.gamma,
            map.get("cross_attn.norm2.gamma"),
            device,
        );
        set_param1(&mut ca.norm2.beta, map.get("cross_attn.norm2.beta"), device);
        for (name, lin) in [("fc1", &mut ca.ffn_fc1), ("fc2", &mut ca.ffn_fc2)] {
            set_linear_w(
                lin,
                map.get(format!("cross_attn.ffn.{name}.weight").as_str()),
                device,
            );
            set_linear_b(
                lin,
                map.get(format!("cross_attn.ffn.{name}.bias").as_str()),
                device,
            );
        }
        for (name, lin) in [
            ("fc1", &mut self.risk_fc1),
            ("fc2", &mut self.risk_fc2),
            ("fc3", &mut self.risk_fc3),
        ] {
            set_linear_w(
                lin,
                map.get(format!("risk_head.{name}.weight").as_str()),
                device,
            );
            set_linear_b(
                lin,
                map.get(format!("risk_head.{name}.bias").as_str()),
                device,
            );
        }
    }

    /// Persist config + weights to `path` (postcard `ModelRecord`), loadable by
    /// the forward service the same way as `PerAssetCnn` / `LstmPredictor`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        save_record(
            path.as_ref(),
            &self.config,
            &self.extract_weights(),
            "master_cnn",
            self.config.n_features,
            1,
        )
    }

    /// Load a model persisted by [`save`](Self::save).
    pub fn load<P: AsRef<Path>>(path: P, device: &B::Device) -> Result<Self> {
        let (config, weights) = load_record::<MasterCnnConfig>(path.as_ref())?;
        let mut model = Self::new(config, device);
        if !weights.is_empty() {
            model.apply_weights(&weights, device);
        }
        Ok(model)
    }
}

// ---------------------------------------------------------------------------
// Independent reference forward pass (raw f32) — the parity oracle
// ---------------------------------------------------------------------------

/// High-accuracy erf (Abramowitz & Stegun 7.1.26, |err| < 1.5e-7).
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = ((((1.061_405_4 * t - 1.453_152) * t + 1.421_413_7) * t - 0.284_496_73) * t
        + 0.254_829_6)
        * t;
    (1.0 - poly * (-x * x).exp()).copysign(x)
}
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

/// Linear (burn layout `w` = `[in, out]`): `y[o] = b[o] + Σ_i x[i]·w[i*out+o]`.
fn linear(x: &[f32], w: &[f32], b: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
    let mut y = vec![0f32; n_out];
    for (o, slot) in y.iter_mut().enumerate() {
        let mut acc = b[o];
        for i in 0..n_in {
            acc += x[i] * w[i * n_out + o];
        }
        *slot = acc;
    }
    y
}

/// LayerNorm over a single `e`-length row (biased variance, affine).
fn layer_norm(x: &[f32], gamma: &[f32], beta: &[f32], e: usize) -> Vec<f32> {
    let mean = x.iter().sum::<f32>() / e as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / e as f32;
    let inv = 1.0 / (var + LN_EPS).sqrt();
    (0..e)
        .map(|i| (x[i] - mean) * inv * gamma[i] + beta[i])
        .collect()
}

/// Independent raw-`f32` reference for the whole `MasterCnn` forward on a single
/// sample. `input`: row-major `(n_assets, n_features, window)`. Returns the
/// scalar risk logit (pre-sigmoid).
pub fn reference_risk(weights: &WeightMap, input: &[f32], cfg: &MasterCnnConfig) -> f32 {
    let map: HashMap<&str, &SerializedTensor> =
        weights.iter().map(|st| (st.name.as_str(), st)).collect();
    let get = |name: &str| -> &[f32] {
        map.get(name)
            .unwrap_or_else(|| panic!("missing weight {name}"))
            .data
            .as_slice()
    };
    let (a, e, h) = (cfg.n_assets, cfg.embedding_dim, cfg.n_heads);
    let dh = e / h;
    let sample_len = cfg.n_features * cfg.window;

    // 1) encode each asset via the shared backbone → (a, e)
    let mut tokens: Vec<f32> = Vec::with_capacity(a * e);
    for i in 0..a {
        let slice = &input[i * sample_len..(i + 1) * sample_len];
        tokens.extend(reference_encoder(
            &map,
            slice,
            cfg.n_features,
            cfg.window,
            e,
        ));
    }

    // 2a) self-attention on LayerNorm(tokens)
    let (gq, bq) = (get("cross_attn.q.weight"), get("cross_attn.q.bias"));
    let (gk, bk) = (get("cross_attn.k.weight"), get("cross_attn.k.bias"));
    let (gv, bv) = (get("cross_attn.v.weight"), get("cross_attn.v.bias"));
    let (go, bo) = (get("cross_attn.out.weight"), get("cross_attn.out.bias"));
    let (n1g, n1b) = (get("cross_attn.norm.gamma"), get("cross_attn.norm.beta"));

    let normed: Vec<Vec<f32>> = (0..a)
        .map(|s| layer_norm(&tokens[s * e..(s + 1) * e], n1g, n1b, e))
        .collect();
    let q: Vec<Vec<f32>> = normed.iter().map(|r| linear(r, gq, bq, e, e)).collect();
    let k: Vec<Vec<f32>> = normed.iter().map(|r| linear(r, gk, bk, e, e)).collect();
    let v: Vec<Vec<f32>> = normed.iter().map(|r| linear(r, gv, bv, e, e)).collect();

    // context[s] over all heads
    let mut context = vec![0f32; a * e];
    let scale = 1.0 / (dh as f32).sqrt();
    for head in 0..h {
        let off = head * dh;
        for s in 0..a {
            // scaled scores against every token, then softmax
            let mut scores = vec![0f32; a];
            for (t, score) in scores.iter_mut().enumerate() {
                let dot: f32 = (0..dh).map(|d| q[s][off + d] * k[t][off + d]).sum();
                *score = dot * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|v| (v - mx).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for d in 0..dh {
                let mut acc = 0f32;
                for t in 0..a {
                    acc += (exps[t] / denom) * v[t][off + d];
                }
                context[s * e + off + d] = acc;
            }
        }
    }

    // out projection + residual
    let mut x: Vec<f32> = vec![0f32; a * e];
    for s in 0..a {
        let proj = linear(&context[s * e..(s + 1) * e], go, bo, e, e);
        for d in 0..e {
            x[s * e + d] = tokens[s * e + d] + proj[d];
        }
    }

    // 2b) feed-forward on LayerNorm2(x) + residual
    let (n2g, n2b) = (get("cross_attn.norm2.gamma"), get("cross_attn.norm2.beta"));
    let (f1w, f1b) = (
        get("cross_attn.ffn.fc1.weight"),
        get("cross_attn.ffn.fc1.bias"),
    );
    let (f2w, f2b) = (
        get("cross_attn.ffn.fc2.weight"),
        get("cross_attn.ffn.fc2.bias"),
    );
    for s in 0..a {
        let normed2 = layer_norm(&x[s * e..(s + 1) * e], n2g, n2b, e);
        let mut hidden = linear(&normed2, f1w, f1b, e, 2 * e);
        for v in hidden.iter_mut() {
            *v = gelu(*v);
        }
        let ffn = linear(&hidden, f2w, f2b, 2 * e, e);
        for d in 0..e {
            x[s * e + d] += ffn[d];
        }
    }

    // 3) risk head on the flattened tokens
    let combined_dim = a * e;
    let h1 = linear(
        &x,
        get("risk_head.fc1.weight"),
        get("risk_head.fc1.bias"),
        combined_dim,
        96,
    )
    .iter()
    .map(|v| v.max(0.0))
    .collect::<Vec<_>>();
    let h2 = linear(
        &h1,
        get("risk_head.fc2.weight"),
        get("risk_head.fc2.bias"),
        96,
        24,
    )
    .iter()
    .map(|v| v.max(0.0))
    .collect::<Vec<_>>();
    linear(
        &h2,
        get("risk_head.fc3.weight"),
        get("risk_head.fc3.bias"),
        24,
        1,
    )[0]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use burn_core::tensor::TensorData;

    fn make_input(b: usize, a: usize, nf: usize, w: usize) -> Vec<f32> {
        (0..b * a * nf * w)
            .map(|i| ((i * 2_654_435_761usize % 1009) as f32 / 1009.0) - 0.5)
            .collect()
    }

    #[test]
    fn save_load_roundtrip() {
        let device = Default::default();
        let cfg = MasterCnnConfig::default();
        let model = MasterCnn::<CpuBackend>::new(cfg.clone(), &device);
        let input = Tensor::<CpuBackend, 4>::from_data(
            TensorData::new(
                make_input(2, cfg.n_assets, cfg.n_features, cfg.window),
                [2, cfg.n_assets, cfg.n_features, cfg.window],
            ),
            &device,
        );
        let before = model
            .forward(input.clone())
            .to_data()
            .to_vec::<f32>()
            .unwrap();

        let path = std::env::temp_dir().join("master_cnn_roundtrip.bin");
        model.save(&path).unwrap();
        let loaded = MasterCnn::<CpuBackend>::load(&path, &device).unwrap();
        let after = loaded.forward(input).to_data().to_vec::<f32>().unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.config(), model.config());
        for (a, b) in before.iter().zip(after.iter()) {
            assert!((a - b).abs() < 1e-6, "save/load mismatch {a} vs {b}");
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let device = Default::default();
        let cfg = MasterCnnConfig::default();
        let model = MasterCnn::<CpuBackend>::new(cfg.clone(), &device);
        let input = Tensor::<CpuBackend, 4>::from_data(
            TensorData::new(
                make_input(2, cfg.n_assets, cfg.n_features, cfg.window),
                [2, cfg.n_assets, cfg.n_features, cfg.window],
            ),
            &device,
        );
        let out = model.forward(input);
        assert_eq!(out.dims(), [2, 1]);
        for v in out.to_data().to_vec::<f32>().unwrap() {
            assert!(v.is_finite());
        }
        // sigmoid risk score in [0,1]
        for v in model
            .predict_risk(Tensor::<CpuBackend, 4>::from_data(
                TensorData::new(
                    make_input(2, cfg.n_assets, cfg.n_features, cfg.window),
                    [2, cfg.n_assets, cfg.n_features, cfg.window],
                ),
                &device,
            ))
            .to_data()
            .to_vec::<f32>()
            .unwrap()
        {
            assert!((0.0..=1.0).contains(&v), "risk {v} out of [0,1]");
        }
    }

    #[test]
    fn weight_injection_roundtrip() {
        let device = Default::default();
        let cfg = MasterCnnConfig::default();
        let model_a = MasterCnn::<CpuBackend>::new(cfg.clone(), &device);
        let mut model_b = MasterCnn::<CpuBackend>::new(cfg.clone(), &device);
        model_b.apply_weights(&model_a.extract_weights(), &device);

        let input = Tensor::<CpuBackend, 4>::from_data(
            TensorData::new(
                make_input(2, cfg.n_assets, cfg.n_features, cfg.window),
                [2, cfg.n_assets, cfg.n_features, cfg.window],
            ),
            &device,
        );
        let oa = model_a
            .forward(input.clone())
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let ob = model_b.forward(input).to_data().to_vec::<f32>().unwrap();
        for (a, b) in oa.iter().zip(ob.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip mismatch {a} vs {b}");
        }
    }

    /// The heart of the port: the `burn` model must agree with the independent
    /// raw-`f32` reference (encoder + cross-attention + risk head) on the same
    /// weights.
    #[test]
    fn burn_matches_reference() {
        let device = Default::default();
        let cfg = MasterCnnConfig::default();
        let model = MasterCnn::<CpuBackend>::new(cfg.clone(), &device);
        let weights = model.extract_weights();

        let batch = 3;
        let data = make_input(batch, cfg.n_assets, cfg.n_features, cfg.window);
        let input = Tensor::<CpuBackend, 4>::from_data(
            TensorData::new(
                data.clone(),
                [batch, cfg.n_assets, cfg.n_features, cfg.window],
            ),
            &device,
        );
        let burn_out = model.forward(input).to_data().to_vec::<f32>().unwrap();

        let sample_len = cfg.n_assets * cfg.n_features * cfg.window;
        for b in 0..batch {
            let sample = &data[b * sample_len..(b + 1) * sample_len];
            let reference = reference_risk(&weights, sample, &cfg);
            let diff = (burn_out[b] - reference).abs();
            assert!(
                diff < 2e-3,
                "risk logit mismatch (sample {b}): burn {} vs ref {reference} (Δ{diff})",
                burn_out[b]
            );
        }
    }
}
