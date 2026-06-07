//! `TrainablePerAssetCnn` — autodiff-trainable twin of [`super::per_asset_cnn::PerAssetCnn`].
//!
//! Burn-native training (RUST_MIGRATION.md Phase 4). The merged inference
//! `PerAssetCnn` is a plain struct (its `config` field can't derive `Module`),
//! so this is a separate `#[derive(Module)]` twin with the **identical**
//! architecture, trained on the [`Autodiff`](burn_autodiff::Autodiff) backend
//! and bridged back to the inference model by weight transfer:
//!
//! ```text
//! TrainablePerAssetCnn (Autodiff) --train--> extract_weights() -> WeightMap
//!                                                     |
//!                                       PerAssetCnn::apply_weights()  (inference)
//! ```
//!
//! [`extract_weights`](TrainablePerAssetCnn::extract_weights) emits the exact
//! same keys as `PerAssetCnn::extract_weights`, so a trained twin loads straight
//! into the inference model. Dropout is omitted (as in the inference twin); at
//! inference `BatchNorm` uses running stats, at training (Autodiff) batch stats —
//! burn switches automatically by backend.

use burn::module::{Ignored, Module};
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer, adaptor::OptimizerAdaptor};
use burn_core::tensor::backend::Backend;
use burn_core::tensor::{ElementConversion, Int, Tensor, TensorData, activation};
use burn_nn::conv::{Conv1d, Conv1dConfig};
use burn_nn::loss::CrossEntropyLossConfig;
use burn_nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig1d};
use serde::{Deserialize, Serialize};

use crate::backend::AutodiffCpuBackend;
use crate::models::per_asset_cnn::{PerAssetCnn, PerAssetCnnConfig};
use crate::models::{SerializedTensor, WeightMap};

const C1: usize = 64;
const C2: usize = 128;

fn se_mid(channels: usize) -> usize {
    (channels / 4).max(8)
}
fn same_pad(kernel: usize, dilation: usize) -> usize {
    dilation * (kernel - 1) / 2
}

/// `Conv1d` via im2col (unfold) + `matmul`. Autodiff-friendly on `burn-ndarray`
/// — the native SIMD conv *backward* is bugged for kernel-3 conv1d — and
/// numerically equivalent to `Conv1d::forward` up to summation order. `weight`
/// is `(out, in, k)` (no bias; all our convs are bias-free).
fn conv1d_im2col<B: Backend>(
    x: Tensor<B, 3>,
    weight: Tensor<B, 3>,
    dilation: usize,
    pad: usize,
) -> Tensor<B, 3> {
    let [bsz, cin, l] = x.dims();
    let [cout, _cin, k] = weight.dims();
    let device = x.device();
    let x_pad = if pad > 0 {
        let zeros = Tensor::<B, 3>::zeros([bsz, cin, pad], &device);
        Tensor::cat(vec![zeros.clone(), x, zeros], 2)
    } else {
        x
    };
    let mut acc: Option<Tensor<B, 3>> = None;
    for kk in 0..k {
        let start = kk * dilation;
        let shifted = x_pad.clone().slice([0..bsz, 0..cin, start..start + l]); // (B,Cin,L)
        let wk = weight
            .clone()
            .slice([0..cout, 0..cin, kk..kk + 1])
            .reshape([1, cout, cin]);
        let contrib = wk.matmul(shifted); // (1,Cout,Cin)x(B,Cin,L) → (B,Cout,L)
        acc = Some(match acc {
            Some(a) => a.add(contrib),
            None => contrib,
        });
    }
    acc.unwrap()
}

/// Configuration for [`TrainablePerAssetCnn`] (defaults match `PerAssetCnnConfig`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainablePerAssetCnnConfig {
    /// Input feature channels.
    pub n_features: usize,
    /// Output classes.
    pub n_classes: usize,
    /// Encoder embedding width.
    pub embedding_dim: usize,
}

impl Default for TrainablePerAssetCnnConfig {
    fn default() -> Self {
        Self {
            n_features: 20,
            n_classes: 4,
            embedding_dim: 64,
        }
    }
}

impl TrainablePerAssetCnnConfig {
    /// The inference config corresponding to this trainable config.
    pub fn inference(&self, window: usize) -> PerAssetCnnConfig {
        PerAssetCnnConfig {
            n_features: self.n_features,
            window,
            n_classes: self.n_classes,
            embedding_dim: self.embedding_dim,
        }
    }

    /// Build a freshly-initialised trainable model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrainablePerAssetCnn<B> {
        let conv = |i, o, k, d: usize, bias| {
            Conv1dConfig::new(i, o, k)
                .with_padding(PaddingConfig1d::Explicit(same_pad(k, d)))
                .with_dilation(d)
                .with_bias(bias)
                .init(device)
        };
        let block = |i, o, d| TConvBlock {
            conv: conv(i, o, 3, d, false),
            dilation: Ignored(d),
            bn: BatchNormConfig::new(o).init(device),
            skip: if i != o {
                Some(conv(i, o, 1, 1, false))
            } else {
                None
            },
        };
        let se = |c| {
            let mid = se_mid(c);
            TSeBlock {
                fc1: LinearConfig::new(c, mid).with_bias(false).init(device),
                fc2: LinearConfig::new(mid, c).with_bias(false).init(device),
            }
        };
        let emb = self.embedding_dim;
        TrainablePerAssetCnn {
            encoder: TAssetEncoder {
                block1: block(self.n_features, C1, 1),
                se1: se(C1),
                block2: block(C1, C2, 2),
                se2: se(C2),
                block3: block(C2, emb, 4),
                se3: se(emb),
                block4: block(emb, emb, 8),
                se4: se(emb),
                pool_proj: LinearConfig::new(emb * 2, emb)
                    .with_bias(false)
                    .init(device),
            },
            head_fc1: LinearConfig::new(emb, 32).init(device),
            head_fc2: LinearConfig::new(32, self.n_classes).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct TSeBlock<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

impl<B: Backend> TSeBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, c, _l] = x.dims();
        let squeeze = x.clone().mean_dim(2).reshape([b, c]);
        let excite = activation::relu(self.fc1.forward(squeeze));
        let excite = activation::sigmoid(self.fc2.forward(excite));
        x.mul(excite.reshape([b, c, 1]))
    }
}

#[derive(Module, Debug)]
struct TConvBlock<B: Backend> {
    conv: Conv1d<B>,
    dilation: Ignored<usize>,
    bn: BatchNorm<B>,
    skip: Option<Conv1d<B>>,
}

impl<B: Backend> TConvBlock<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let d = self.dilation.0;
        let conv_out = conv1d_im2col(x.clone(), self.conv.weight.val(), d, same_pad(3, d));
        let main = self.bn.forward(conv_out);
        let identity = match &self.skip {
            Some(skip) => conv1d_im2col(x, skip.weight.val(), 1, 0),
            None => x,
        };
        activation::relu(main.add(identity))
    }
}

#[derive(Module, Debug)]
struct TAssetEncoder<B: Backend> {
    block1: TConvBlock<B>,
    se1: TSeBlock<B>,
    block2: TConvBlock<B>,
    se2: TSeBlock<B>,
    block3: TConvBlock<B>,
    se3: TSeBlock<B>,
    block4: TConvBlock<B>,
    se4: TSeBlock<B>,
    pool_proj: Linear<B>,
}

impl<B: Backend> TAssetEncoder<B> {
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = self.se1.forward(self.block1.forward(x));
        let x = self.se2.forward(self.block2.forward(x));
        let x = self.se3.forward(self.block3.forward(x));
        let x = self.se4.forward(self.block4.forward(x));
        let [b, c, _l] = x.dims();
        let avg = x.clone().mean_dim(2).reshape([b, c]);
        let max = x.max_dim(2).reshape([b, c]);
        let pooled = Tensor::cat(vec![avg, max], 1);
        activation::relu(self.pool_proj.forward(pooled))
    }
}

/// Autodiff-trainable `PerAssetCnn` twin.
#[derive(Module, Debug)]
pub struct TrainablePerAssetCnn<B: Backend> {
    encoder: TAssetEncoder<B>,
    head_fc1: Linear<B>,
    head_fc2: Linear<B>,
}

impl<B: Backend> TrainablePerAssetCnn<B> {
    /// Forward pass. `input`: (batch, n_features, window) → logits (batch, n_classes).
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let emb = self.encoder.forward(input);
        let hidden = activation::relu(self.head_fc1.forward(emb));
        self.head_fc2.forward(hidden)
    }

    /// Serialise weights into a [`WeightMap`] with the **same keys** as
    /// `PerAssetCnn::extract_weights`, so the inference model loads them directly.
    pub fn extract_weights(&self) -> WeightMap {
        let mut w = Vec::new();
        let e = &self.encoder;
        let blocks = [
            (&e.block1, &e.se1, 1usize),
            (&e.block2, &e.se2, 2),
            (&e.block3, &e.se3, 3),
            (&e.block4, &e.se4, 4),
        ];
        for (block, se, i) in blocks {
            w.push(st3(
                &format!("encoder.block{i}.conv.weight"),
                &block.conv.weight.val(),
            ));
            w.push(st1(
                &format!("encoder.block{i}.bn.gamma"),
                &block.bn.gamma.val(),
            ));
            w.push(st1(
                &format!("encoder.block{i}.bn.beta"),
                &block.bn.beta.val(),
            ));
            w.push(st1(
                &format!("encoder.block{i}.bn.running_mean"),
                &block.bn.running_mean.value(),
            ));
            w.push(st1(
                &format!("encoder.block{i}.bn.running_var"),
                &block.bn.running_var.value(),
            ));
            if let Some(skip) = &block.skip {
                w.push(st3(
                    &format!("encoder.block{i}.skip.weight"),
                    &skip.weight.val(),
                ));
            }
            w.push(st2(
                &format!("encoder.se{i}.fc1.weight"),
                &se.fc1.weight.val(),
            ));
            w.push(st2(
                &format!("encoder.se{i}.fc2.weight"),
                &se.fc2.weight.val(),
            ));
        }
        w.push(st2("encoder.pool_proj.weight", &e.pool_proj.weight.val()));
        w.push(st2("head.fc1.weight", &self.head_fc1.weight.val()));
        w.push(st1(
            "head.fc1.bias",
            &self.head_fc1.bias.as_ref().unwrap().val(),
        ));
        w.push(st2("head.fc2.weight", &self.head_fc2.weight.val()));
        w.push(st1(
            "head.fc2.bias",
            &self.head_fc2.bias.as_ref().unwrap().val(),
        ));
        w
    }
}

fn st1<B: Backend>(name: &str, t: &Tensor<B, 1>) -> SerializedTensor {
    SerializedTensor {
        name: name.into(),
        shape: t.dims().to_vec(),
        data: t.to_data().to_vec().unwrap(),
    }
}
fn st2<B: Backend>(name: &str, t: &Tensor<B, 2>) -> SerializedTensor {
    SerializedTensor {
        name: name.into(),
        shape: t.dims().to_vec(),
        data: t.to_data().to_vec().unwrap(),
    }
}
fn st3<B: Backend>(name: &str, t: &Tensor<B, 3>) -> SerializedTensor {
    SerializedTensor {
        name: name.into(),
        shape: t.dims().to_vec(),
        data: t.to_data().to_vec().unwrap(),
    }
}

// ---------------------------------------------------------------------------
// Trainer (AdamW + weighted cross-entropy) — mirrors the LSTM AutodiffTrainer.
// ---------------------------------------------------------------------------

/// Minimal autodiff trainer for [`TrainablePerAssetCnn`] on the CPU backend.
pub struct CnnTrainer {
    model: TrainablePerAssetCnn<AutodiffCpuBackend>,
    optimizer: OptimizerAdaptor<Adam, TrainablePerAssetCnn<AutodiffCpuBackend>, AutodiffCpuBackend>,
    lr: f64,
    class_weights: Option<Vec<f32>>,
}

impl CnnTrainer {
    /// Create a trainer for a freshly-initialised model.
    pub fn new(
        config: &TrainablePerAssetCnnConfig,
        lr: f64,
        weight_decay: f64,
        class_weights: Option<Vec<f32>>,
    ) -> Self {
        let device = Default::default();
        let model = config.init::<AutodiffCpuBackend>(&device);
        let optimizer = AdamConfig::new()
            .with_weight_decay(Some(burn::optim::decay::WeightDecayConfig::new(
                weight_decay as f32,
            )))
            .init::<AutodiffCpuBackend, TrainablePerAssetCnn<AutodiffCpuBackend>>();
        Self {
            model,
            optimizer,
            lr,
            class_weights,
        }
    }

    /// Run one optimisation step on a batch; returns the (weighted) CE loss.
    ///
    /// * `features`: `(batch, n_features, window)`.
    /// * `labels`: one class index per sample.
    pub fn step(&mut self, features: Tensor<AutodiffCpuBackend, 3>, labels: &[i64]) -> f32 {
        let device = Default::default();
        let targets = Tensor::<AutodiffCpuBackend, 1, Int>::from_data(
            TensorData::new(labels.to_vec(), [labels.len()]),
            &device,
        );
        let loss_fn = CrossEntropyLossConfig::new()
            .with_weights(self.class_weights.clone())
            .init(&device);

        let logits = self.model.forward(features);
        let loss = loss_fn.forward(logits, targets);
        let loss_value = loss.clone().into_scalar().elem::<f32>();

        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = self.optimizer.step(self.lr, self.model.clone(), grads);
        loss_value
    }

    /// The current trained model.
    pub fn model(&self) -> &TrainablePerAssetCnn<AutodiffCpuBackend> {
        &self.model
    }

    /// Transfer the trained weights into a ready-to-serve inference model.
    pub fn to_inference(&self, window: usize) -> PerAssetCnn<crate::backend::CpuBackend> {
        let cfg = TrainablePerAssetCnnConfig {
            n_features: 20,
            n_classes: 4,
            embedding_dim: 64,
        }
        .inference(window);
        let device = Default::default();
        let mut inf = PerAssetCnn::<crate::backend::CpuBackend>::new(cfg, &device);
        inf.apply_weights(&self.model.extract_weights(), &device);
        inf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{AutodiffCpuBackend, CpuBackend};

    fn input<B: Backend>(batch: usize, nf: usize, w: usize, device: &B::Device) -> Tensor<B, 3> {
        let data: Vec<f32> = (0..batch * nf * w)
            .map(|i| ((i * 7 % 101) as f32 / 101.0) - 0.5)
            .collect();
        Tensor::<B, 3>::from_data(TensorData::new(data, [batch, nf, w]), device)
    }

    #[test]
    fn forward_shape_on_autodiff_backend() {
        let device = Default::default();
        let model = TrainablePerAssetCnnConfig::default().init::<AutodiffCpuBackend>(&device);
        let out = model.forward(input::<AutodiffCpuBackend>(3, 20, 60, &device));
        assert_eq!(out.dims(), [3, 4]);
    }

    /// Weight transfer + architecture equivalence: a trainable twin on the plain
    /// (inference-mode) backend must produce the same forward as the inference
    /// `PerAssetCnn` after loading its extracted weights.
    #[test]
    fn extract_weights_bridges_to_inference() {
        let device = Default::default();
        let cfg = TrainablePerAssetCnnConfig::default();
        let trainable = cfg.init::<CpuBackend>(&device);
        let mut inference = PerAssetCnn::<CpuBackend>::new(cfg.inference(60), &device);
        inference.apply_weights(&trainable.extract_weights(), &device);

        let x = input::<CpuBackend>(2, 20, 60, &device);
        let a = trainable
            .forward(x.clone())
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let b = inference.forward(x).to_data().to_vec::<f32>().unwrap();
        for (p, q) in a.iter().zip(b.iter()) {
            // im2col conv vs burn's SIMD conv differ only by summation order.
            assert!(
                (p - q).abs() < 2e-3,
                "trainable vs inference mismatch {p} vs {q}"
            );
        }
    }

    /// The core Phase-4 proof: AdamW + cross-entropy actually reduces the loss
    /// on a learnable synthetic task (label depends on a feature's sign). The
    /// convs run through the im2col path, so autodiff works on the CPU backend
    /// (burn-ndarray's native SIMD conv backward is bugged for kernel-3 conv1d).
    #[test]
    fn training_reduces_loss() {
        let device = Default::default();
        let mut trainer = CnnTrainer::new(&TrainablePerAssetCnnConfig::default(), 1e-3, 1e-4, None);

        let (batch, nf, w) = (16usize, 20usize, 60usize);
        // label = 1 (long) when channel-0 mean > 0 else 2 (short); deterministic.
        let mk = |seed: usize| -> (Tensor<AutodiffCpuBackend, 3>, Vec<i64>) {
            let mut data = vec![0f32; batch * nf * w];
            let mut labels = vec![0i64; batch];
            for s in 0..batch {
                let positive = (s + seed).is_multiple_of(2);
                for c in 0..nf {
                    for t in 0..w {
                        let base = if c == 0 {
                            if positive { 0.4 } else { -0.4 }
                        } else {
                            0.0
                        };
                        data[(s * nf + c) * w + t] =
                            base + (((s + c + t + seed) * 13 % 17) as f32 / 17.0 - 0.5) * 0.05;
                    }
                }
                labels[s] = if positive { 1 } else { 2 };
            }
            (
                Tensor::from_data(TensorData::new(data, [batch, nf, w]), &device),
                labels,
            )
        };

        let (x0, y0) = mk(0);
        let first = trainer.step(x0, &y0);
        let mut last = first;
        for epoch in 1..40 {
            let (x, y) = mk(epoch);
            last = trainer.step(x, &y);
        }
        assert!(last.is_finite() && first.is_finite());
        assert!(last < first, "loss should fall: first {first}, last {last}");

        // The trained twin bridges to a usable inference model.
        let inf = trainer.to_inference(w);
        let out = inf.forward(input::<CpuBackend>(2, nf, w, &device));
        assert_eq!(out.dims(), [2, 4]);
    }
}
