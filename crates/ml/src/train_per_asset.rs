//! `train_per_asset` — end-to-end "train a `PerAssetCnn` champion from OHLCV".
//!
//! The Phase-4 capstone (RUST_MIGRATION.md): orchestrates the whole burn-native
//! training pipeline into a single call —
//!
//! ```text
//! OHLCV → precompute_features → generate_labels_breakout → make_windows
//!       → CnnTrainer (AdamW + weighted CE, LR warmup, early stopping)
//!       → to_inference → PerAssetCnn   (ready to serve)
//! ```
//!
//! Mirrors the Python `scripts/train.sh` per-asset pipeline. The training-time
//! channels 7–9 use the inference-path constant broadcasts (not the Python
//! train-mode random walks — a known refinement); everything else matches the
//! champion contract.

use burn_core::tensor::{Tensor, TensorData};

use crate::backend::AutodiffCpuBackend;
use crate::backend::CpuBackend;
use crate::features::per_asset_cnn::{LiveState, precompute_features};
use crate::labeler::{BreakoutLabelConfig, generate_labels_breakout, label_counts};
use crate::models::per_asset_cnn::PerAssetCnn;
use crate::models::trainable_per_asset_cnn::{CnnTrainer, TrainablePerAssetCnnConfig};
use crate::per_asset_dataset::{class_weights, make_windows};

/// Configuration for [`train_champion`].
#[derive(Debug, Clone)]
pub struct TrainChampionConfig {
    /// Candle window (time dimension of each sample).
    pub window: usize,
    /// Max training epochs.
    pub epochs: usize,
    /// Samples per optimisation step.
    pub batch_size: usize,
    /// Base learning rate.
    pub lr: f64,
    /// AdamW weight decay.
    pub weight_decay: f64,
    /// Use inverse-frequency class weights in the loss.
    pub use_class_weights: bool,
    /// Linear LR warmup over the first N epochs.
    pub warmup_epochs: usize,
    /// Stop after this many epochs without an improvement in epoch loss.
    pub early_stopping_patience: usize,
    /// Shuffle seed.
    pub seed: u64,
    /// Labeler configuration.
    pub label: BreakoutLabelConfig,
    /// Model configuration.
    pub model: TrainablePerAssetCnnConfig,
}

impl Default for TrainChampionConfig {
    fn default() -> Self {
        Self {
            window: 60,
            epochs: 60,
            batch_size: 64,
            lr: 1e-4,
            weight_decay: 1e-4,
            use_class_weights: true,
            warmup_epochs: 3,
            early_stopping_patience: 12,
            seed: 42,
            label: BreakoutLabelConfig::default(),
            model: TrainablePerAssetCnnConfig::default(),
        }
    }
}

/// Outcome of a training run.
#[derive(Debug, Clone)]
pub struct TrainReport {
    /// Epochs actually run (≤ `epochs`; fewer if early-stopped).
    pub epochs_run: usize,
    /// Number of training samples.
    pub samples: usize,
    /// Per-epoch mean loss.
    pub loss_history: Vec<f32>,
    /// Best (lowest) epoch loss.
    pub best_loss: f32,
    /// Label distribution `[flat, long, short, loss]` over the samples.
    pub label_counts: [usize; 4],
}

/// Deterministic in-place Fisher–Yates shuffle (xorshift64, no deps).
fn shuffle(order: &mut [usize], seed: u64) {
    let mut s = seed | 1;
    for i in (1..order.len()).rev() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let j = (s % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
}

/// Train a `PerAssetCnn` champion end-to-end from an OHLCV series.
///
/// Returns the trained **inference** model + a [`TrainReport`], or `None` when
/// there aren't enough bars to produce any training samples.
pub fn train_champion(
    open: &[f32],
    high: &[f32],
    low: &[f32],
    close: &[f32],
    volume: &[f32],
    cfg: &TrainChampionConfig,
) -> Option<(PerAssetCnn<CpuBackend>, TrainReport)> {
    let nf = cfg.model.n_features;
    let n_bars = close.len();

    // 1) features (full series) + 2) labels.
    let features = precompute_features(
        open,
        high,
        low,
        close,
        volume,
        &LiveState::default(),
        cfg.window,
    )?;
    let labels = generate_labels_breakout(high, low, close, &cfg.label);

    // 3) sliding-window samples.
    let samples = make_windows(&features, nf, n_bars, &labels, cfg.window, None);
    if samples.is_empty() {
        return None;
    }
    let sample_labels: Vec<i64> = samples.iter().map(|s| s.label).collect();
    let counts = label_counts(&sample_labels);

    // 4) trainer (optionally class-weighted).
    let weights = cfg.use_class_weights.then(|| {
        class_weights(&sample_labels, 25.0)
            .iter()
            .map(|&w| w as f32)
            .collect::<Vec<f32>>()
    });
    let mut trainer = CnnTrainer::new(&cfg.model, cfg.lr, cfg.weight_decay, weights);

    // 5) training loop with LR warmup + early stopping.
    let device = Default::default();
    let mut order: Vec<usize> = (0..samples.len()).collect();
    let mut history = Vec::with_capacity(cfg.epochs);
    let mut best = f32::INFINITY;
    let mut patience = 0usize;

    for epoch in 0..cfg.epochs {
        let lr = if epoch < cfg.warmup_epochs {
            cfg.lr * (epoch + 1) as f64 / cfg.warmup_epochs.max(1) as f64
        } else {
            cfg.lr
        };
        trainer.set_lr(lr);
        shuffle(&mut order, cfg.seed.wrapping_add(epoch as u64));

        let (mut epoch_loss, mut batches) = (0.0f32, 0usize);
        for batch in order.chunks(cfg.batch_size) {
            let bsz = batch.len();
            let mut data = Vec::with_capacity(bsz * nf * cfg.window);
            let mut lbls = Vec::with_capacity(bsz);
            for &idx in batch {
                data.extend_from_slice(&samples[idx].features);
                lbls.push(samples[idx].label);
            }
            let x = Tensor::<AutodiffCpuBackend, 3>::from_data(
                TensorData::new(data, [bsz, nf, cfg.window]),
                &device,
            );
            epoch_loss += trainer.step(x, &lbls);
            batches += 1;
        }

        let avg = epoch_loss / batches.max(1) as f32;
        history.push(avg);
        if avg < best - 1e-4 {
            best = avg;
            patience = 0;
        } else {
            patience += 1;
            if patience >= cfg.early_stopping_patience {
                break;
            }
        }
    }

    let model = trainer.to_inference(cfg.window);
    let report = TrainReport {
        epochs_run: history.len(),
        samples: samples.len(),
        loss_history: history,
        best_loss: best,
        label_counts: counts,
    };
    Some((model, report))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_ohlcv(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // A few clean consolidation→breakout episodes so labels aren't all flat.
        let mut close = vec![100.0f32; n];
        for i in 0..n {
            let phase = i / 80;
            let base = 100.0 + phase as f32 * 4.0; // step up each episode → breakouts
            close[i] = base + ((i % 80) as f32 * 0.15).sin() * 0.3;
        }
        let high: Vec<f32> = close.iter().map(|c| c + 0.3).collect();
        let low: Vec<f32> = close.iter().map(|c| c - 0.3).collect();
        let open: Vec<f32> = close.iter().map(|c| c - 0.05).collect();
        let volume: Vec<f32> = (0..n).map(|i| 1000.0 + (i % 11) as f32 * 30.0).collect();
        (open, high, low, close, volume)
    }

    #[test]
    fn too_few_bars_returns_none() {
        let cfg = TrainChampionConfig::default();
        let n = 80; // well below window + label/dataset warmup
        let s = vec![100.0f32; n];
        assert!(train_champion(&s, &s, &s, &s, &s, &cfg).is_none());
    }

    /// End-to-end smoke: the pipeline runs (features → labels → windows →
    /// train → bridge) and yields a usable inference model. Kept small (few
    /// epochs) since im2col conv training is CPU-bound.
    #[test]
    fn trains_end_to_end_and_yields_inference_model() {
        let (o, h, l, c, v) = synth_ohlcv(400);
        let cfg = TrainChampionConfig {
            epochs: 2,
            batch_size: 64,
            warmup_epochs: 1,
            early_stopping_patience: 99,
            ..TrainChampionConfig::default()
        };
        let (model, report) = train_champion(&o, &h, &l, &c, &v, &cfg).expect("should train");

        assert_eq!(report.epochs_run, 2);
        assert!(report.samples > 0);
        assert_eq!(report.loss_history.len(), 2);
        assert!(report.loss_history.iter().all(|x| x.is_finite()));
        assert_eq!(report.label_counts.iter().sum::<usize>(), report.samples);

        // The trained champion serves valid inference.
        let device = Default::default();
        let window = cfg.window;
        let input = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(vec![0.1f32; 20 * window], [1, 20, window]),
            &device,
        );
        let probs = model
            .predict_proba(input)
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(probs.len(), 4);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "softmax sums to 1, got {sum}");
    }
}
