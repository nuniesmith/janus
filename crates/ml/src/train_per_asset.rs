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
use crate::per_asset_dataset::{Sample, class_weights, make_windows};

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

    // 3) sliding-window samples → train on all of them.
    let samples = make_windows(&features, nf, n_bars, &labels, cfg.window, None);
    train_on_samples(&samples, cfg)
}

/// Train a `PerAssetCnn` on a prepared sample set — the shared training core of
/// [`train_champion`] and [`train_champion_with_holdout`]. `None` if empty.
fn train_on_samples(
    samples: &[Sample],
    cfg: &TrainChampionConfig,
) -> Option<(PerAssetCnn<CpuBackend>, TrainReport)> {
    if samples.is_empty() {
        return None;
    }
    let nf = cfg.model.n_features;
    let sample_labels: Vec<i64> = samples.iter().map(|s| s.label).collect();
    let counts = label_counts(&sample_labels);

    // trainer (optionally class-weighted).
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

/// Display names for the 4 label classes (index == class id).
pub const CLASS_NAMES: [&str; 4] = ["flat", "long", "short", "loss"];

/// Per-class score from a validation split.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassScore {
    /// `TP / (TP + FP)` — of the bars predicted this class, share that were right.
    pub precision: f64,
    /// `TP / (TP + FN)` — of the bars truly this class, share that were caught.
    pub recall: f64,
    /// Validation bars truly of this class.
    pub support: usize,
    /// `support / n_val` — this class's own base rate (the precision a random
    /// guesser would get). Precision above this is real skill on the class.
    pub base_rate: f64,
}

/// Out-of-sample metrics from a leakage-safe, time-ordered holdout.
#[derive(Debug, Clone)]
pub struct ValMetrics {
    /// Training samples after the purge.
    pub n_train: usize,
    /// Validation samples evaluated.
    pub n_val: usize,
    /// Overall accuracy on the validation split.
    pub accuracy: f64,
    /// Largest single-class share of the validation set — the trivial
    /// "always predict the majority class" accuracy to beat.
    pub majority_baseline: f64,
    /// Confusion `[true][pred]` over classes 0=flat 1=long 2=short 3=loss.
    pub confusion: [[usize; 4]; 4],
    /// Per-class precision / recall / support / base rate.
    pub per_class: [ClassScore; 4],
    /// Terminal bar splitting train (`< split_bar`) from validation (`>=`).
    pub split_bar: usize,
    /// Purge gap (bars) dropped between splits so no training label sees a
    /// validation-region outcome.
    pub purge: usize,
}

/// Run the trained model over the validation samples and tally per-class metrics.
fn evaluate(
    model: &PerAssetCnn<CpuBackend>,
    val: &[Sample],
    cfg: &TrainChampionConfig,
    n_train: usize,
    split_bar: usize,
    purge: usize,
) -> ValMetrics {
    let nf = cfg.model.n_features;
    let device = Default::default();
    let mut confusion = [[0usize; 4]; 4];
    let mut correct = 0usize;

    for chunk in val.chunks(256) {
        let bsz = chunk.len();
        let mut data = Vec::with_capacity(bsz * nf * cfg.window);
        for s in chunk {
            data.extend_from_slice(&s.features);
        }
        let x = Tensor::<CpuBackend, 3>::from_data(
            TensorData::new(data, [bsz, nf, cfg.window]),
            &device,
        );
        let logits = model.forward(x).to_data().to_vec::<f32>().unwrap();
        for (i, s) in chunk.iter().enumerate() {
            let row = &logits[i * 4..i * 4 + 4];
            let pred = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap();
            let truth = (s.label as usize).min(3);
            confusion[truth][pred] += 1;
            if pred == truth {
                correct += 1;
            }
        }
    }

    let n_val = val.len().max(1);
    let mut per_class = [ClassScore::default(); 4];
    let mut max_support = 0usize;
    for c in 0..4 {
        let tp = confusion[c][c];
        let predicted_c: usize = (0..4).map(|t| confusion[t][c]).sum();
        let support: usize = (0..4).map(|p| confusion[c][p]).sum();
        max_support = max_support.max(support);
        per_class[c] = ClassScore {
            precision: if predicted_c > 0 {
                tp as f64 / predicted_c as f64
            } else {
                0.0
            },
            recall: if support > 0 {
                tp as f64 / support as f64
            } else {
                0.0
            },
            support,
            base_rate: support as f64 / n_val as f64,
        };
    }

    ValMetrics {
        n_train,
        n_val: val.len(),
        accuracy: correct as f64 / n_val as f64,
        majority_baseline: max_support as f64 / n_val as f64,
        confusion,
        per_class,
        split_bar,
        purge,
    }
}

/// Train a champion on a leakage-safe, time-ordered holdout and measure
/// out-of-sample generalization.
///
/// The RAW series is split by time at `1 - val_frac`. A training sample whose
/// forward label horizon (`max_breakout_wait + max_hold_bars`) would reach into
/// the validation region is PURGED — so no training label sees a
/// validation-region outcome (the standard purged split for look-ahead-labeled
/// financial series). Features look only backward, so validation samples reusing
/// pre-split bars for their own warmup is not leakage.
///
/// Returns the model trained on the purged train split plus its validation
/// [`ValMetrics`]. `None` when `val_frac ∉ (0, 0.9)` or the series is too short
/// to form both splits.
pub fn train_champion_with_holdout(
    open: &[f32],
    high: &[f32],
    low: &[f32],
    close: &[f32],
    volume: &[f32],
    cfg: &TrainChampionConfig,
    val_frac: f64,
) -> Option<(PerAssetCnn<CpuBackend>, TrainReport, ValMetrics)> {
    if !(val_frac > 0.0 && val_frac < 0.9) {
        return None;
    }
    let nf = cfg.model.n_features;
    let n_bars = close.len();

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
    let all = make_windows(&features, nf, n_bars, &labels, cfg.window, None);
    if all.is_empty() {
        return None;
    }

    // all[k] is the window ending at bar `first_bar + k` (mirrors make_windows).
    let first_bar = (cfg.window + crate::per_asset_dataset::WARMUP_EXTRA).max(cfg.window - 1);
    let horizon = cfg.label.max_breakout_wait + cfg.label.max_hold_bars;
    let split_bar = ((n_bars as f64) * (1.0 - val_frac)) as usize;
    if split_bar <= first_bar + horizon || split_bar >= n_bars {
        return None;
    }

    let (mut train_samples, mut val_samples) = (Vec::new(), Vec::new());
    for (k, s) in all.iter().enumerate() {
        let t = first_bar + k; // terminal (labeled) bar
        if t + horizon < split_bar {
            train_samples.push(s.clone()); // label stays strictly before the split
        } else if t >= split_bar {
            val_samples.push(s.clone()); // in the validation region
        }
        // else: purge zone [split_bar - horizon, split_bar) — dropped
    }
    if train_samples.is_empty() || val_samples.is_empty() {
        return None;
    }

    let n_train = train_samples.len();
    let (model, report) = train_on_samples(&train_samples, cfg)?;
    let metrics = evaluate(&model, &val_samples, cfg, n_train, split_bar, horizon);
    Some((model, report, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::type_complexity)]
    fn synth_ohlcv(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        // A few clean consolidation→breakout episodes so labels aren't all flat.
        let mut close = vec![100.0f32; n];
        for (i, c) in close.iter_mut().enumerate() {
            let phase = i / 80;
            let base = 100.0 + phase as f32 * 4.0; // step up each episode → breakouts
            *c = base + ((i % 80) as f32 * 0.15).sin() * 0.3;
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

    /// The holdout split must be time-ordered and leakage-safe: purge equals the
    /// label horizon, both splits are populated, the purge drops a horizon-sized
    /// band (so the splits don't cover every window), and metrics are well-formed.
    #[test]
    fn holdout_is_leakage_safe_and_reports_metrics() {
        let (o, h, l, c, v) = synth_ohlcv(1200);
        let cfg = TrainChampionConfig {
            epochs: 2,
            warmup_epochs: 1,
            early_stopping_patience: 99,
            ..TrainChampionConfig::default()
        };
        let (_model, report, m) =
            train_champion_with_holdout(&o, &h, &l, &c, &v, &cfg, 0.3).expect("should validate");

        let horizon = cfg.label.max_breakout_wait + cfg.label.max_hold_bars;
        assert_eq!(
            m.purge, horizon,
            "purge must equal the forward label horizon"
        );
        assert!(
            m.split_bar > 1200 * 6 / 10 && m.split_bar < 1200 * 8 / 10,
            "split placed ~70% through: {}",
            m.split_bar
        );
        assert!(m.n_train > 0 && m.n_val > 0, "both splits populated");
        assert_eq!(report.samples, m.n_train, "report reflects the train split");

        // The purge dropped a horizon-sized band → splits don't cover all windows.
        let total_windows = 1200 - (cfg.window + crate::per_asset_dataset::WARMUP_EXTRA);
        assert!(
            m.n_train + m.n_val <= total_windows.saturating_sub(horizon) + 2,
            "purge must drop ~horizon samples: train {} + val {} vs {} windows",
            m.n_train,
            m.n_val,
            total_windows
        );

        // Metrics well-formed.
        assert!((0.0..=1.0).contains(&m.accuracy));
        assert!((0.0..=1.0).contains(&m.majority_baseline));
        assert_eq!(
            m.per_class.iter().map(|p| p.support).sum::<usize>(),
            m.n_val,
            "per-class support partitions the validation set"
        );
        for p in &m.per_class {
            assert!((0.0..=1.0).contains(&p.precision) && (0.0..=1.0).contains(&p.recall));
        }

        // Out-of-range val_frac → None.
        assert!(train_champion_with_holdout(&o, &h, &l, &c, &v, &cfg, 0.0).is_none());
        assert!(train_champion_with_holdout(&o, &h, &l, &c, &v, &cfg, 0.95).is_none());
    }
}
