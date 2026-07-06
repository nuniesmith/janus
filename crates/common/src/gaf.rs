//! Dependency-free pooled Gramian Angular Summation Field (GASF) features.
//!
//! This is the canonical `state_gaf` feature function for the experience
//! pipeline (see `docs/architecture/EXPERIENCE_PIPELINE.md` §3.1): **both**
//! the forward writer and backward's trainer/seeder must call [`gaf_flat`]
//! so train/serve feature parity holds by construction. It is pure `f32`
//! math — no Candle, no ndarray — so it is safe to link into the live
//! trading binary.
//!
//! # Pipeline
//!
//! 1. **Min-max normalize** the close window to `[-1, 1]`:
//!    `x̃_i = 2·(x_i − min) / (max − min) − 1`.
//!    A *flat window* (`max == min`) deterministically normalizes to all
//!    zeros, so its GASF is uniformly `cos(π/2 + π/2) = −1` (every pooled
//!    output value is exactly `−1.0`).
//! 2. **Polar encoding**: `φ_i = arccos(x̃_i)`.
//! 3. **GASF**: `G[i][j] = cos(φ_i + φ_j)`, computed via the exact identity
//!    `cos(φ_i + φ_j) = x̃_i·x̃_j − √(1−x̃_i²)·√(1−x̃_j²)` (no trig calls).
//! 4. **k×k spatial mean-pool** over the `n×n` matrix, flattened row-major
//!    to `k²` values, each in `[-1, 1]`.
//!
//! # Pooling partition (ceil/floor block partition)
//!
//! When `n` is not divisible by `k`, remainder cells are distributed with the
//! standard floor-boundary partition: block `b` (0-indexed, `b ∈ 0..k`)
//! covers indices `[b·n/k, (b+1)·n/k)` using integer (floor) division. Block
//! lengths therefore differ by at most one, and the remainder blocks are
//! spread evenly across the window rather than all leading or trailing
//! (e.g. `n=10, k=3` → block lengths `3, 3, 4`; `n=5, k=2` → `2, 3`).
//!
//! # Relationship to `crates/vision` (DiffGAF)
//!
//! `jflow-vision`'s `gaf_features_from_series` computes the same GASF
//! construction but normalizes with `tanh(z-score)` instead of min-max and
//! pools with a single per-channel spatial mean (equivalent to `k = 1`
//! here). The two paths therefore agree approximately — not bitwise — for
//! `k = 1`; a parity test with a documented tolerance lives in
//! `crates/vision/tests/gaf_parity.rs`.

/// Pooled, flattened GASF of a close-price window.
///
/// Returns `k²` values in `[-1, 1]`, row-major (see module docs for the
/// exact construction and the pooling partition).
///
/// # Minimum window
///
/// The window must contain at least `k` samples so that every pooling block
/// covers at least one element. Returns an **empty vec** when
/// `closes.len() < k` or `k == 0`; callers must treat `is_empty()` as
/// "window not warm yet".
///
/// Non-finite inputs (NaN/±inf) are not screened and propagate into the
/// output.
///
/// # Examples
///
/// ```
/// use common::gaf::gaf_flat;
///
/// // k = 1 is the global GASF mean.
/// let v = gaf_flat(&[1.0, 2.0, 3.0], 1);
/// assert_eq!(v.len(), 1);
/// assert!((v[0] - (-1.0 / 9.0)).abs() < 1e-6);
///
/// // Window shorter than k → empty (not warm).
/// assert!(gaf_flat(&[1.0, 2.0], 3).is_empty());
/// ```
#[must_use]
pub fn gaf_flat(closes: &[f32], k: usize) -> Vec<f32> {
    let n = closes.len();
    if k == 0 || n < k {
        return Vec::new();
    }

    // 1. Min-max normalize to [-1, 1]; flat window → all zeros.
    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    for &c in closes {
        min = min.min(c);
        max = max.max(c);
    }
    let range = max - min;
    let normalized: Vec<f32> = if range == 0.0 {
        vec![0.0; n]
    } else {
        closes
            .iter()
            .map(|&c| (2.0 * (c - min) / range - 1.0).clamp(-1.0, 1.0))
            .collect()
    };

    // 2./3. GASF[i][j] = cos(φ_i + φ_j) = x_i·x_j − s_i·s_j with
    // s_i = sin(φ_i) = √(1 − x_i²) (clamped ≥ 0 for float safety).
    let sin_phi: Vec<f32> = normalized
        .iter()
        .map(|&x| (1.0 - x * x).max(0.0).sqrt())
        .collect();

    // 4. k×k mean-pool. The GASF is separable per pooling cell:
    //    Σ_{i∈r, j∈c} (x_i·x_j − s_i·s_j) = X_r·X_c − S_r·S_c
    // with X_b = Σ_{i∈b} x_i and S_b = Σ_{i∈b} s_i, so the pooled cell mean
    // is computed exactly from per-block sums without materializing the
    // n×n matrix. f64 accumulation keeps long-window sums stable.
    let bounds: Vec<usize> = (0..=k).map(|b| b * n / k).collect();
    let mut x_sum = vec![0.0f64; k];
    let mut s_sum = vec![0.0f64; k];
    for b in 0..k {
        for i in bounds[b]..bounds[b + 1] {
            x_sum[b] += f64::from(normalized[i]);
            s_sum[b] += f64::from(sin_phi[i]);
        }
    }

    let mut out = Vec::with_capacity(k * k);
    for r in 0..k {
        let len_r = (bounds[r + 1] - bounds[r]) as f64;
        for c in 0..k {
            let len_c = (bounds[c + 1] - bounds[c]) as f64;
            let mean = (x_sum[r] * x_sum[c] - s_sum[r] * s_sum[c]) / (len_r * len_c);
            out.push((mean as f32).clamp(-1.0, 1.0));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    /// Naive reference: materialize the n×n GASF and mean-pool it with the
    /// documented floor-boundary partition.
    fn gaf_flat_reference(closes: &[f32], k: usize) -> Vec<f32> {
        let n = closes.len();
        if k == 0 || n < k {
            return Vec::new();
        }
        let (min, max) = closes
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &c| {
                (lo.min(c), hi.max(c))
            });
        let range = max - min;
        let x: Vec<f64> = closes
            .iter()
            .map(|&c| {
                if range == 0.0 {
                    0.0
                } else {
                    f64::from((2.0 * (c - min) / range - 1.0).clamp(-1.0, 1.0))
                }
            })
            .collect();
        // Full GASF via arccos/cos — the transparent formulation.
        let phi: Vec<f64> = x.iter().map(|&v| v.acos()).collect();
        let mut out = Vec::with_capacity(k * k);
        for r in 0..k {
            let (r0, r1) = (r * n / k, (r + 1) * n / k);
            for c in 0..k {
                let (c0, c1) = (c * n / k, (c + 1) * n / k);
                let mut acc = 0.0f64;
                for i in r0..r1 {
                    for j in c0..c1 {
                        acc += (phi[i] + phi[j]).cos();
                    }
                }
                out.push((acc / ((r1 - r0) * (c1 - c0)) as f64) as f32);
            }
        }
        out
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() < EPS, "index {i}: actual {a} vs expected {e}");
        }
    }

    #[test]
    fn hand_computed_k1() {
        // closes [1,2,3] → x̃ = [-1, 0, 1] → GASF =
        //   [[ 1,  0, -1],
        //    [ 0, -1,  0],
        //    [-1,  0,  1]]
        // global mean = -1/9.
        assert_close(&gaf_flat(&[1.0, 2.0, 3.0], 1), &[-1.0 / 9.0]);
    }

    #[test]
    fn hand_computed_k_equals_n() {
        // Same window, k = n = 3: no pooling, output is the raw GASF
        // flattened row-major.
        assert_close(
            &gaf_flat(&[1.0, 2.0, 3.0], 3),
            &[1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
        );
    }

    #[test]
    fn hand_computed_k2_n4() {
        // closes [0,1,2,3] → x̃ = [-1, -1/3, 1/3, 1], s = [0, √8/3, √8/3, 0].
        // Blocks {0,1}, {2,3}: X₀ = -4/3, X₁ = 4/3, S₀ = S₁ = √8/3.
        // pooled[r][c] = (X_r·X_c − S_r·S_c) / 4 →
        //   [ (16/9 − 8/9)/4, (−16/9 − 8/9)/4 ]   [ 2/9, −2/3 ]
        //   [ (−16/9 − 8/9)/4, (16/9 − 8/9)/4 ] = [ −2/3, 2/9 ]
        assert_close(
            &gaf_flat(&[0.0, 1.0, 2.0, 3.0], 2),
            &[2.0 / 9.0, -2.0 / 3.0, -2.0 / 3.0, 2.0 / 9.0],
        );
    }

    #[test]
    fn output_length_is_k_squared() {
        let closes: Vec<f32> = (0..64).map(|t| 100.0 + (t as f32 * 0.3).sin()).collect();
        for k in [1, 2, 3, 4, 7, 8, 64] {
            assert_eq!(gaf_flat(&closes, k).len(), k * k, "k = {k}");
        }
    }

    #[test]
    fn all_values_in_unit_interval() {
        let closes: Vec<f32> = (0..97)
            .map(|t| 42_000.0 + (t as f32 * 0.7).sin() * 350.0 + t as f32)
            .collect();
        for k in [1, 3, 5, 9, 96, 97] {
            for (i, v) in gaf_flat(&closes, k).iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(v),
                    "k = {k}, index {i}: {v} out of [-1, 1]"
                );
            }
        }
    }

    #[test]
    fn flat_window_is_deterministic_all_minus_one() {
        // max == min → x̃ ≡ 0 → GASF ≡ cos(π/2 + π/2) = −1 everywhere.
        let out = gaf_flat(&[5.0; 10], 3);
        assert_eq!(out.len(), 9);
        for v in &out {
            assert!((v + 1.0).abs() < EPS, "expected -1, got {v}");
        }
        // Deterministic: identical output on repeat.
        assert_eq!(out, gaf_flat(&[5.0; 10], 3));
    }

    #[test]
    fn window_length_edge_cases() {
        // Shorter than k, k == 0, and empty windows → empty (not warm).
        assert!(gaf_flat(&[1.0, 2.0], 3).is_empty());
        assert!(gaf_flat(&[], 1).is_empty());
        assert!(gaf_flat(&[1.0, 2.0, 3.0], 0).is_empty());
        // Exactly k samples is the documented minimum → k² values.
        assert_eq!(gaf_flat(&[1.0, 2.0, 3.0], 3).len(), 9);
        // Single sample, k = 1: flat window (min == max) → [-1].
        assert_close(&gaf_flat(&[7.0], 1), &[-1.0]);
    }

    #[test]
    fn matches_naive_reference_including_remainder_partitions() {
        // n not divisible by k exercises the floor-boundary partition
        // (n=10,k=3 → blocks 3,3,4; n=5,k=2 → blocks 2,3).
        let closes: Vec<f32> = (0..10)
            .map(|t| 50.0 + (t as f32 * 1.1).cos() * 3.0)
            .collect();
        for (n, k) in [(10, 3), (5, 2), (10, 4), (9, 3), (7, 7), (10, 1)] {
            let window = &closes[..n];
            assert_close(&gaf_flat(window, k), &gaf_flat_reference(window, k));
        }
    }

    #[test]
    fn k3_default_dim_matches_qdrant_contract() {
        // The Phase-1 default: k = 3 ⇒ dim 9 = QDRANT_EXPERIENCE_DIM default.
        let closes: Vec<f32> = (0..64).map(|t| 100.0 + (t as f32 * 0.3).sin()).collect();
        assert_eq!(gaf_flat(&closes, 3).len(), 9);
    }
}
