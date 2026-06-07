//! `labeler` — Rust port of the Python champion training labeler
//! (`fks-full/src/ruby/src/ml/labeler.py::generate_labels_breakout`).
//!
//! First building block of burn-native training (RUST_MIGRATION.md Phase 4):
//! turns historical OHLC bars into the 4-class targets `PerAssetCnn` is trained
//! on (0 flat / 1 long / 2 short / 3 loss), via a consolidation→breakout→
//! ATR-bracket walk-forward. This is the **non-deprecated** labeler — the
//! EMA/AO `generate_labels` is intentionally not ported (it gates labels on
//! features that are also CNN inputs, causing label/feature leakage).
//!
//! # Algorithm (per bar `i`, the consolidation-end candidate)
//!
//! 1. Measure the rolling high/low over `consolidation_bars`.
//! 2. If `(range_high - range_low) / ATR > consolidation_atr_mult` → not a
//!    consolidation → label flat.
//! 3. Scan up to `max_breakout_wait` bars ahead for a *close* beyond the range;
//!    no breakout → flat.
//! 4. From the breakout bar, size an ATR bracket and walk forward up to
//!    `max_hold_bars`: TP hit → long/short (win), SL hit → loss, timeout → flat.
//! 5. Assign the outcome to bar `i` (not the breakout bar).
//!
//! # Parity
//!
//! Pure `f64` arithmetic mirroring the NumPy implementation (ATR = EWM of true
//! range, `adjust=False`). Helper + scenario unit tests cover it in-sandbox;
//! `tests::python_goldens` checks full Python↔Rust parity against vectors from
//! `tools/parity/record_labels.py` (skips cleanly when absent).

/// No signal / timed out.
pub const LABEL_FLAT: i64 = 0;
/// Long breakout that hit TP before SL.
pub const LABEL_LONG: i64 = 1;
/// Short breakout that hit TP before SL.
pub const LABEL_SHORT: i64 = 2;
/// Breakout that hit SL first (explicit loss).
pub const LABEL_LOSS: i64 = 3;

/// Breakout-labeler configuration (defaults match the Python champion).
#[derive(Debug, Clone, PartialEq)]
pub struct BreakoutLabelConfig {
    /// Bars in the consolidation lookback.
    pub consolidation_bars: usize,
    /// Max `range / ATR` to qualify as a consolidation.
    pub consolidation_atr_mult: f64,
    /// Bars to scan forward for a breakout.
    pub max_breakout_wait: usize,
    /// TP distance in ATRs from the entry.
    pub tp_atr_mult: f64,
    /// SL distance in ATRs from the entry.
    pub sl_atr_mult: f64,
    /// Bars to walk forward from the breakout for an outcome.
    pub max_hold_bars: usize,
    /// ATR EWM smoothing period.
    pub atr_period: usize,
}

impl Default for BreakoutLabelConfig {
    fn default() -> Self {
        Self {
            consolidation_bars: 20,
            consolidation_atr_mult: 5.5,
            max_breakout_wait: 40,
            tp_atr_mult: 1.5,
            sl_atr_mult: 1.0,
            max_hold_bars: 90,
            atr_period: 14,
        }
    }
}

impl BreakoutLabelConfig {
    /// Minimum bars required to produce any non-flat labels.
    pub fn min_rows(&self) -> usize {
        self.consolidation_bars + self.atr_period + self.max_breakout_wait + self.max_hold_bars + 10
    }
}

/// True range, then EWM(span=period, adjust=False) — matches `_atr_arr`.
fn atr_ewm(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0; n];
    for i in 0..n {
        let prev = if i == 0 { close[0] } else { close[i - 1] };
        tr[i] = (high[i] - low[i])
            .max((high[i] - prev).abs())
            .max((low[i] - prev).abs());
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut atr = vec![0.0; n];
    if n > 0 {
        atr[0] = tr[0];
        for i in 1..n {
            atr[i] = (1.0 - alpha) * atr[i - 1] + alpha * tr[i];
        }
    }
    atr
}

/// Generate 4-class breakout labels for an OHLC series (one per bar).
///
/// Returns all-`LABEL_FLAT` when there are fewer than [`BreakoutLabelConfig::min_rows`]
/// bars or the input lengths disagree.
// Index-faithful port of the NumPy walk-forward: the scan loop needs the
// absolute bar index for `breakout_idx` + an early break, so range-indexing is
// intentional here.
#[allow(clippy::needless_range_loop)]
pub fn generate_labels_breakout(
    high: &[f32],
    low: &[f32],
    close: &[f32],
    cfg: &BreakoutLabelConfig,
) -> Vec<i64> {
    let n = close.len();
    if n < cfg.min_rows() || high.len() != n || low.len() != n {
        return vec![LABEL_FLAT; n];
    }
    let hi: Vec<f64> = high.iter().map(|&v| v as f64).collect();
    let lo: Vec<f64> = low.iter().map(|&v| v as f64).collect();
    let cl: Vec<f64> = close.iter().map(|&v| v as f64).collect();
    let atr = atr_ewm(&hi, &lo, &cl, cfg.atr_period);

    // Rolling consolidation range.
    let mut roll_high = vec![0.0; n];
    let mut roll_low = vec![0.0; n];
    for i in 0..n {
        let start = (i + 1).saturating_sub(cfg.consolidation_bars);
        roll_high[i] = hi[start..=i]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        roll_low[i] = lo[start..=i].iter().cloned().fold(f64::INFINITY, f64::min);
    }

    let mut labels = vec![LABEL_FLAT; n];
    let warmup = cfg.consolidation_bars + 14;
    // `min_rows` guarantees n > max_breakout_wait + max_hold_bars + 1.
    let max_i = n - cfg.max_breakout_wait - cfg.max_hold_bars - 1;

    for i in warmup..max_i.max(warmup) {
        let atr_i = atr[i];
        if atr_i <= 0.0 {
            continue;
        }
        let (range_h, range_l) = (roll_high[i], roll_low[i]);
        if range_h <= range_l {
            continue;
        }
        // Consolidation gate: range must be tight relative to ATR.
        if (range_h - range_l) / atr_i > cfg.consolidation_atr_mult {
            continue;
        }

        // Scan forward for the first close beyond the range.
        let mut direction = LABEL_FLAT;
        let mut breakout_idx = 0usize;
        let mut entry_price = 0.0;
        let scan_end = (i + cfg.max_breakout_wait + 1).min(n);
        for j in (i + 1)..scan_end {
            if cl[j] > range_h {
                direction = LABEL_LONG;
                entry_price = range_h;
                breakout_idx = j;
                break;
            } else if cl[j] < range_l {
                direction = LABEL_SHORT;
                entry_price = range_l;
                breakout_idx = j;
                break;
            }
        }
        if direction == LABEL_FLAT {
            continue;
        }

        // ATR bracket sized at the breakout bar.
        let mut atr_j = atr[breakout_idx];
        if atr_j <= 0.0 {
            atr_j = atr_i;
        }
        let (tp_price, sl_price) = if direction == LABEL_LONG {
            (
                entry_price + cfg.tp_atr_mult * atr_j,
                entry_price - cfg.sl_atr_mult * atr_j,
            )
        } else {
            (
                entry_price - cfg.tp_atr_mult * atr_j,
                entry_price + cfg.sl_atr_mult * atr_j,
            )
        };

        // Walk forward to an outcome (timeout → flat).
        let mut outcome = LABEL_FLAT;
        let walk_end = (breakout_idx + cfg.max_hold_bars + 1).min(n);
        for k in (breakout_idx + 1)..walk_end {
            if direction == LABEL_LONG {
                if hi[k] >= tp_price {
                    outcome = LABEL_LONG;
                    break;
                }
                if lo[k] <= sl_price {
                    outcome = LABEL_LOSS;
                    break;
                }
            } else {
                if lo[k] <= tp_price {
                    outcome = LABEL_SHORT;
                    break;
                }
                if hi[k] >= sl_price {
                    outcome = LABEL_LOSS;
                    break;
                }
            }
        }
        labels[i] = outcome;
    }
    labels
}

/// Class counts `[flat, long, short, loss]`.
pub fn label_counts(labels: &[i64]) -> [usize; 4] {
    let mut c = [0usize; 4];
    for &l in labels {
        if (0..4).contains(&l) {
            c[l as usize] += 1;
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant-range bars → ATR converges to the (constant) true range.
    #[test]
    fn atr_ewm_constant_bars() {
        let (h, l, c) = (vec![3.0; 40], vec![1.0; 40], vec![2.0; 40]);
        let atr = atr_ewm(&h, &l, &c, 14);
        assert!((atr.last().unwrap() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn too_few_rows_is_all_flat() {
        let cfg = BreakoutLabelConfig::default();
        let n = cfg.min_rows() - 1;
        let series = vec![100.0f32; n];
        let labels = generate_labels_breakout(&series, &series, &series, &cfg);
        assert_eq!(labels.len(), n);
        assert!(labels.iter().all(|&l| l == LABEL_FLAT));
    }

    /// A steep trend never satisfies the consolidation gate → all flat.
    #[test]
    fn trending_series_is_all_flat() {
        let cfg = BreakoutLabelConfig::default();
        let n = 260;
        let close: Vec<f32> = (0..n).map(|i| 100.0 + 2.0 * i as f32).collect();
        let high: Vec<f32> = close.iter().map(|c| c + 0.1).collect();
        let low: Vec<f32> = close.iter().map(|c| c - 0.1).collect();
        let labels = generate_labels_breakout(&high, &low, &close, &cfg);
        assert!(
            labels.iter().all(|&l| l == LABEL_FLAT),
            "trend must not label"
        );
    }

    /// Tight consolidation then a clean upward breakout that hits TP must
    /// produce at least one LONG label, and only valid classes.
    #[test]
    fn consolidation_then_breakout_labels_long() {
        let cfg = BreakoutLabelConfig::default();
        let n = 250;
        let mut close = vec![100.0f32; n];
        let mut high = vec![100.3f32; n];
        let mut low = vec![99.7f32; n];
        for i in 150..n {
            // sustained breakout well above the ~100.3 range high
            close[i] = 103.0;
            high[i] = 103.3;
            low[i] = 102.7;
        }
        let labels = generate_labels_breakout(&high, &low, &close, &cfg);
        assert!(
            labels.iter().all(|&l| (0..4).contains(&l)),
            "valid classes only"
        );
        assert!(
            labels.contains(&LABEL_LONG),
            "a consolidation→breakout→TP must yield a LONG label"
        );
        // counts sum to n
        assert_eq!(label_counts(&labels).iter().sum::<usize>(), n);
    }

    /// Full Python↔Rust parity. Skips when no goldens are present (generate with
    /// `tools/parity/record_labels.py` in a numpy env).
    #[test]
    fn python_goldens() {
        #[derive(serde::Deserialize)]
        struct Case {
            high: Vec<f32>,
            low: Vec<f32>,
            close: Vec<f32>,
            expected: Vec<i64>,
        }
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/labels");
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => {
                eprintln!("[python_goldens] {dir} absent — skipping");
                return;
            }
        };
        let cfg = BreakoutLabelConfig::default();
        let mut checked = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let case: Case = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let got = generate_labels_breakout(&case.high, &case.low, &case.close, &cfg);
            assert_eq!(got, case.expected, "{path:?}: label mismatch");
            checked += 1;
        }
        eprintln!("[python_goldens] verified {checked} golden file(s)");
    }
}
