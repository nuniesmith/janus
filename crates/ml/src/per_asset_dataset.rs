//! `per_asset_dataset` — training-data preparation for `PerAssetCnn`.
//!
//! Phase-4 building block (RUST_MIGRATION.md): turns a precomputed feature
//! matrix + the [`crate::labeler`] targets into `(window, label)` training
//! samples, plus inverse-frequency class weights for the heavy flat-class
//! imbalance. Pure data logic (no autodiff) — the burn training loop consumes
//! these samples.
//!
//! Mirrors the Python `ml/dataset.py::AssetDataset`: each sample is a
//! `(N_FEATURES, window)` window **ending at** the labeled bar (channel-major,
//! same layout as [`crate::features::per_asset_cnn::extract_features`]), and the
//! label is the labeler's outcome at that bar. The first usable bar is
//! `window + WARMUP_EXTRA` (AO-slow SMA warmup), matching the Python default.

/// Extra warmup bars beyond `window` before the first sample (AO-slow warmup).
pub const WARMUP_EXTRA: usize = 34;

/// One training sample: a flattened `(n_features, window)` feature window
/// (channel-major) paired with its integer class label.
#[derive(Debug, Clone, PartialEq)]
pub struct Sample {
    /// `n_features * window` values, channel-major (`channel * window + bar`).
    pub features: Vec<f32>,
    /// Class label at the window's last bar (0 flat / 1 long / 2 short / 3 loss).
    pub label: i64,
}

/// Build sliding-window training samples from a precomputed feature matrix.
///
/// * `features`: `n_features * n_bars`, channel-major (`channel * n_bars + bar`),
///   as produced by `precompute_all_features` / repeated `extract_features`.
/// * `labels`: one label per bar (length `n_bars`).
/// * `window`: time dimension of each sample.
/// * `warmup`: first usable bar index; defaults to `window + WARMUP_EXTRA`.
///
/// Returns one [`Sample`] per bar in `[warmup, n_bars)` (window ending at the
/// labeled bar). Empty when the shapes disagree or there aren't enough bars.
pub fn make_windows(
    features: &[f32],
    n_features: usize,
    n_bars: usize,
    labels: &[i64],
    window: usize,
    warmup: Option<usize>,
) -> Vec<Sample> {
    if window == 0
        || n_features == 0
        || features.len() != n_features * n_bars
        || labels.len() != n_bars
    {
        return Vec::new();
    }
    let warmup = warmup.unwrap_or(window + WARMUP_EXTRA).max(window - 1);
    if warmup >= n_bars {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(n_bars - warmup);
    for bar in warmup..n_bars {
        let start = bar + 1 - window; // window ends at `bar` (inclusive)
        let mut sample = vec![0.0f32; n_features * window];
        for c in 0..n_features {
            let src = c * n_bars + start;
            sample[c * window..(c + 1) * window].copy_from_slice(&features[src..src + window]);
        }
        out.push(Sample {
            features: sample,
            label: labels[bar],
        });
    }
    out
}

/// Inverse-frequency class weights for `[flat, long, short, loss]`, balancing
/// the heavy flat majority. Port of `ml/labeler.py::class_weights`: weights are
/// `total / (n_classes * count)`, normalised to a min of 1, then clamped to a
/// cap that tightens (10.0) when the rarest non-flat class is < 5% of samples.
pub fn class_weights(labels: &[i64], max_weight: f64) -> [f64; 4] {
    let mut counts = [0.0f64; 4];
    for &l in labels {
        if (0..4).contains(&l) {
            counts[l as usize] += 1.0;
        }
    }
    for c in counts.iter_mut() {
        *c = c.max(1.0);
    }
    let total: f64 = counts.iter().sum();
    let minority = counts[1..].iter().copied().fold(f64::INFINITY, f64::min);
    let dynamic_cap = if minority / total < 0.05 {
        10.0
    } else {
        max_weight.min(25.0)
    };

    let mut w = [0.0f64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = total / (4.0 * counts[i]);
    }
    let wmin = w.iter().copied().fold(f64::INFINITY, f64::min);
    for wi in w.iter_mut() {
        *wi = (*wi / wmin).clamp(0.0, dynamic_cap);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_align_to_last_bar() {
        // features[c * n_bars + t] = c*1000 + t, so we can verify the slice exactly.
        let (n_features, n_bars, window) = (2usize, 30usize, 5usize);
        let features: Vec<f32> = (0..n_features)
            .flat_map(|c| (0..n_bars).map(move |t| (c * 1000 + t) as f32))
            .collect();
        let labels: Vec<i64> = (0..n_bars as i64).map(|t| t % 4).collect();

        let samples = make_windows(
            &features,
            n_features,
            n_bars,
            &labels,
            window,
            Some(window - 1),
        );
        // first usable bar = window-1; samples for bars (window-1)..n_bars
        assert_eq!(samples.len(), n_bars - (window - 1));

        // Sample for bar = window-1=4: window [0..=4]. label = labels[4].
        let s0 = &samples[0];
        assert_eq!(s0.label, labels[window - 1]);
        // channel 0, bars 0..5 → [0,1,2,3,4]; channel 1 → [1000..1005]
        assert_eq!(&s0.features[0..window], &[0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            &s0.features[window..2 * window],
            &[1000.0, 1001.0, 1002.0, 1003.0, 1004.0]
        );

        // Last sample ends at the final bar.
        let last = samples.last().unwrap();
        assert_eq!(last.label, labels[n_bars - 1]);
        assert_eq!(last.features[window - 1], (n_bars - 1) as f32); // ch0 last value
    }

    #[test]
    fn shape_mismatch_and_short_inputs_yield_nothing() {
        assert!(make_windows(&[1.0, 2.0, 3.0], 2, 2, &[0, 0], 2, None).is_empty()); // len 3 != 4
        let feats = vec![0.0f32; 2 * 10];
        assert!(make_windows(&feats, 2, 10, &[0; 10], 2, Some(50)).is_empty()); // warmup >= n_bars
    }

    #[test]
    fn class_weights_balanced_is_unit() {
        let labels: Vec<i64> = (0..40).map(|i| i % 4).collect(); // 10 each
        let w = class_weights(&labels, 50.0);
        for x in w {
            assert!((x - 1.0).abs() < 1e-9, "balanced → all 1.0, got {x}");
        }
    }

    #[test]
    fn class_weights_upweights_rare_classes() {
        // 90 flat, 6 long, 3 short, 1 loss → minorities far rarer than flat.
        let mut labels = vec![0i64; 90];
        labels.extend(std::iter::repeat_n(1, 6));
        labels.extend(std::iter::repeat_n(2, 3));
        labels.push(3);
        let w = class_weights(&labels, 50.0);
        assert!(
            (w[0] - 1.0).abs() < 1e-9,
            "flat (majority) normalises to 1.0"
        );
        assert!(
            w[3] >= w[2] && w[2] >= w[1] && w[1] > 1.0,
            "rarer class → larger weight"
        );
        // rarest non-flat class (loss, 1/100 < 5%) → cap tightens to 10.0
        assert!(
            w.iter().all(|&x| x <= 10.0 + 1e-9),
            "tight cap applies when a class < 5%"
        );
    }
}
