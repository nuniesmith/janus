//! Parity between `common::gaf_flat` (the canonical, dependency-free pooled
//! GASF used by the experience pipeline) and this crate's Candle-based
//! DiffGAF flat path (`gaf_features_from_closes`), per
//! `docs/architecture/EXPERIENCE_PIPELINE.md` §3.1.
//!
//! # Why this is approximate parity, not equivalence
//!
//! True equivalence does not exist **by construction**. Both paths compute
//! the same GASF (`G[i][j] = cos(φ_i + φ_j)` over a normalized series) and
//! then spatially mean-pool, but they differ in two ways:
//!
//! 1. **Pooling.** Vision's "flat" output is a *single* per-channel spatial
//!    mean of the whole GASF image — i.e. it corresponds to `gaf_flat` with
//!    `k = 1` (one global pooling cell), not the `k×k` grid. So only the
//!    `k = 1` case is comparable at all.
//! 2. **Normalization.** `gaf_flat` min-max normalizes the window to
//!    `[-1, 1]` (the classical GAF construction); DiffGAF standardizes over
//!    time and squashes with `tanh` (`x̃ = tanh((x − μ)/σ)`, learnable-norm
//!    disabled). These map the same window to different points in `[-1, 1]`,
//!    so the pooled GASF means agree only approximately.
//!
//! The tests therefore assert `k = 1` agreement within a *documented,
//! empirically calibrated* tolerance (observed gap ≲ 0.1 on realistic close
//! windows; asserted at 0.15) on series where the two normalizations are
//! comparable, plus exact structural invariants (bounds) that hold for both.
//! The resize step is bypassed by setting `image_size = closes.len()`.
//!
//! Dependency direction: vision already depends on `common`, so this test
//! lives here — `common` must never gain a vision/Candle dependency.

use candle_core::Device;
use common::gaf::gaf_flat;
use vision::gaf_features_from_closes;

/// Vision's flat GAF for a closes-only window, with the internal resize
/// disabled (`image_size == window length`) so both paths see the same
/// samples.
fn vision_flat(closes: &[f32]) -> f32 {
    let gaf = gaf_features_from_closes(closes, closes.len(), &Device::Cpu)
        .expect("vision GAF should succeed on finite input");
    assert_eq!(gaf.flat.len(), 1, "closes-only series has one channel");
    gaf.flat[0]
}

/// Tolerance for the k=1 parity check. The residual is attributable to the
/// normalization difference (min-max vs tanh(z-score)) documented above;
/// observed gaps on the fixtures below are ≲ 0.1.
const TOLERANCE: f32 = 0.15;

fn fixtures() -> Vec<(&'static str, Vec<f32>)> {
    vec![
        (
            "sine around 100",
            (0..64).map(|t| 100.0 + (t as f32 * 0.3).sin()).collect(),
        ),
        (
            "btc-like trend + oscillation",
            (0..64)
                .map(|t| 42_000.0 + t as f32 * 15.0 + (t as f32 * 0.7).sin() * 120.0)
                .collect(),
        ),
        (
            "mean-reverting chop",
            (0..48)
                .map(|t| 1.35 + (t as f32 * 1.1).sin() * 0.01 + (t as f32 * 0.23).cos() * 0.004)
                .collect(),
        ),
    ]
}

#[test]
fn k1_pooled_gasf_agrees_with_vision_flat_within_tolerance() {
    for (name, closes) in fixtures() {
        let ours = gaf_flat(&closes, 1);
        assert_eq!(ours.len(), 1, "{name}: k=1 must yield one value");
        let theirs = vision_flat(&closes);
        let gap = (ours[0] - theirs).abs();
        assert!(
            gap < TOLERANCE,
            "{name}: common k=1 GASF mean {} vs vision flat {} (gap {gap} ≥ {TOLERANCE})",
            ours[0],
            theirs,
        );
    }
}

#[test]
fn both_paths_stay_in_gasf_bounds() {
    // Exact invariant shared by both constructions: pooled GASF means are
    // cosine averages, so they live in [-1, 1].
    for (name, closes) in fixtures() {
        let ours = gaf_flat(&closes, 1)[0];
        let theirs = vision_flat(&closes);
        for (path, v) in [("common", ours), ("vision", theirs)] {
            assert!(
                (-1.0..=1.0).contains(&v),
                "{name}/{path}: {v} outside [-1, 1]"
            );
        }
    }
}
