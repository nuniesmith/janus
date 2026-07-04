//! `per_asset_cnn` features — bit-faithful Rust port of the Python champion
//! feature pipeline (`fks-full/src/ruby/src/ml/features.py`, **inference path**).
//!
//! Produces the `(N_FEATURES=20, window)` tensor `PerAssetCnn` consumes, from
//! OHLCV bars plus three live scalars (order-book `imbalance`, volatility
//! `vol_pct`, `wave_ratio`). This is the parity-critical prerequisite for
//! serving the champion in Rust: the model's weights are only meaningful on
//! *these* features (RUST_MIGRATION.md §8.4).
//!
//! The 20 channels (one value per bar) match `feature_names()` exactly:
//! `log_return, norm_close, norm_atr, volume_ratio, rsi, range_tightness,
//! close_vs_range, book_imbalance, wave_ratio, vol_percentile, body_ratio,
//! upper_wick, lower_wick, candle_dir, hurst_exponent, atr_trend, volume_trend,
//! norm_velocity, price_acceleration, market_phase`.
//!
//! # Parity
//!
//! Pandas semantics are reproduced precisely (EWM `adjust=False`, Wilder RSI
//! seeding, rolling `std` ddof=1 / `min_periods`, SMA `min_periods=window`,
//! R/S Hurst). Statistics are computed in `f64` (as pandas does) and cast to
//! `f32`. Helper correctness is covered by hand-computed unit tests; full
//! Python↔Rust parity is checked by `tests::python_goldens` against vectors
//! from `tools/parity/record_features.py` (skips cleanly when absent).
//!
//! Only the **inference** path is ported (`training=false`): channels 7–9
//! (`book_imbalance`, `wave_ratio`, `vol_percentile`) are constant broadcasts
//! of the live scalars, not the training-time random walks.

/// Number of feature channels.
pub const N_FEATURES: usize = 20;
/// Default candle window (time dimension of the output tensor).
pub const DEFAULT_WINDOW: usize = 60;

const CONSOL_BARS: usize = 20;
const RSI_PERIOD: usize = 14;
const ATR_PERIOD: usize = 14;
const AO_FAST: usize = 5;
const AO_SLOW: usize = 34;
const EPS: f64 = 1e-10;

/// Warmup context (bars BEFORE the returned window) that [`extract_features`]
/// computes over, so that every one of the returned `window` columns is fully
/// warmed and matches the full-series [`precompute_features`] the model trains
/// on. Sized to the deepest history-dependent channel: `normalized_velocity` /
/// `price_acceleration` use a `norm_window = 100` rolling-std over a 3-bar
/// momentum series (→ 103 bars needed for the earliest window column), which
/// dominates `norm_close` (60), `market_phase` (AO_SLOW+CONSOL = 54), and
/// `hurst` (~41). 110 leaves margin; the two exponential channels (`norm_atr`,
/// `rsi`) converge to within ~1e-3 over this warmup. Before this was `25`,
/// which left those channels under-warmed / at their defaults at inference —
/// a train/serve feature-parity skew (the model was served out-of-distribution
/// features). See the `bounded_warmup_matches_full_series` regression test.
const WARMUP: usize = 110;

/// Live market scalars that feed channels 7–9 (broadcast across the window).
#[derive(Debug, Clone, Copy)]
pub struct LiveState {
    /// Order-book bid/ask volume ratio (`bid_vol / ask_vol`).
    pub imbalance: f32,
    /// Volatility percentile in `[0, 1]`.
    pub vol_pct: f32,
    /// `WaveState.wave_ratio` (typically 0–3; clipped then normalised).
    pub wave_ratio: f32,
}

impl Default for LiveState {
    fn default() -> Self {
        Self {
            imbalance: 1.0,
            vol_pct: 0.5,
            wave_ratio: 0.0,
        }
    }
}

/// Ordered channel names (index == channel row).
pub fn feature_names() -> [&'static str; N_FEATURES] {
    [
        "log_return",
        "norm_close",
        "norm_atr",
        "volume_ratio",
        "rsi",
        "range_tightness",
        "close_vs_range",
        "book_imbalance",
        "wave_ratio",
        "vol_percentile",
        "body_ratio",
        "upper_wick",
        "lower_wick",
        "candle_dir",
        "hurst_exponent",
        "atr_trend",
        "volume_trend",
        "norm_velocity",
        "price_acceleration",
        "market_phase",
    ]
}

// ---------------------------------------------------------------------------
// pandas-faithful helpers (compute in f64)
// ---------------------------------------------------------------------------

/// `Series.ewm(span, adjust=False).mean()` — `y[0]=x[0]`, recursive otherwise.
fn ewm_adjust_false(x: &[f64], span: usize) -> Vec<f64> {
    let alpha = 2.0 / (span as f64 + 1.0);
    let mut y = vec![0.0; x.len()];
    if x.is_empty() {
        return y;
    }
    y[0] = x[0];
    for i in 1..x.len() {
        y[i] = (1.0 - alpha) * y[i - 1] + alpha * x[i];
    }
    y
}

/// `Series.rolling(window, min_periods=1).mean()`.
fn rolling_mean(x: &[f64], window: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let lo = i + 1 - (i + 1).min(window);
            let s = &x[lo..=i];
            s.iter().sum::<f64>() / s.len() as f64
        })
        .collect()
}

/// `Series.rolling(window, min_periods=2).std().fillna(1.0)` (ddof=1).
fn rolling_std(x: &[f64], window: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let lo = i + 1 - (i + 1).min(window);
            let s = &x[lo..=i];
            let n = s.len();
            if n < 2 {
                return 1.0;
            }
            let mean = s.iter().sum::<f64>() / n as f64;
            let var = s.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n as f64 - 1.0);
            var.sqrt()
        })
        .collect()
}

/// `Series.rolling(window, min_periods=1).max()`.
fn rolling_max(x: &[f64], window: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let lo = i + 1 - (i + 1).min(window);
            x[lo..=i].iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        })
        .collect()
}

/// `Series.rolling(window, min_periods=1).min()`.
fn rolling_min(x: &[f64], window: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let lo = i + 1 - (i + 1).min(window);
            x[lo..=i].iter().cloned().fold(f64::INFINITY, f64::min)
        })
        .collect()
}

/// `Series.rolling(window).mean()` (min_periods=window): `NaN` until full.
fn sma_full_window(x: &[f64], window: usize) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            if i + 1 < window {
                f64::NAN
            } else {
                x[i + 1 - window..=i].iter().sum::<f64>() / window as f64
            }
        })
        .collect()
}

/// Wilder-smoothed RSI in `[0, 100]`; `rsi[0] = 50`.
fn wilder_rsi(close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    if n < period + 1 {
        return vec![50.0; n];
    }
    let delta: Vec<f64> = (1..n).map(|i| close[i] - close[i - 1]).collect();
    let gain: Vec<f64> = delta.iter().map(|d| d.max(0.0)).collect();
    let loss: Vec<f64> = delta.iter().map(|d| (-d).max(0.0)).collect();

    let m = delta.len();
    let (mut avg_gain, mut avg_loss) = (vec![0.0; m], vec![0.0; m]);
    avg_gain[period - 1] = gain[..period].iter().sum::<f64>() / period as f64;
    avg_loss[period - 1] = loss[..period].iter().sum::<f64>() / period as f64;
    let alpha = 1.0 / period as f64;
    for i in period..m {
        avg_gain[i] = avg_gain[i - 1] * (1.0 - alpha) + gain[i] * alpha;
        avg_loss[i] = avg_loss[i - 1] * (1.0 - alpha) + loss[i] * alpha;
    }

    let mut rsi = vec![50.0; n];
    for i in 0..m {
        let rs = if avg_loss[i] == 0.0 {
            100.0
        } else {
            avg_gain[i] / (avg_loss[i] + 1e-10)
        };
        rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs);
    }
    rsi
}

/// True range, then EWM(span=period, adjust=False).
fn atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0; n];
    for i in 0..n {
        let prev = if i == 0 { close[0] } else { close[i - 1] };
        tr[i] = (high[i] - low[i])
            .max((high[i] - prev).abs())
            .max((low[i] - prev).abs());
    }
    ewm_adjust_false(&tr, period)
}

/// Least-squares slope of `(xs, ys)` (degree-1 polyfit).
fn linreg_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut den = 0.0;
    for (x, y) in xs.iter().zip(ys.iter()) {
        num += (x - mx) * (y - my);
        den += (x - mx) * (x - mx);
    }
    if den.abs() < 1e-12 { 0.0 } else { num / den }
}

/// Rolling R/S Hurst exponent in `[0, 1]` (default 0.5).
fn hurst(close: &[f64], max_lag: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.5; n];
    let min_bars = max_lag * 2 + 1;
    for t in min_bars..n {
        let prices = &close[t.saturating_sub(max_lag * 2)..=t];
        let (mut log_lags, mut log_rs) = (Vec::new(), Vec::new());
        for lag in 2..=max_lag {
            let chunks = prices.len() / lag;
            if chunks < 1 {
                continue;
            }
            let mut rs_values = Vec::new();
            for c in 0..chunks {
                let chunk = &prices[c * lag..(c + 1) * lag];
                if chunk.len() < 2 {
                    continue;
                }
                let rets: Vec<f64> = (1..chunk.len()).map(|i| chunk[i] - chunk[i - 1]).collect();
                if rets.len() < 2 {
                    continue;
                }
                let mean_r = rets.iter().sum::<f64>() / rets.len() as f64;
                let mut cum = 0.0;
                let (mut mx, mut mn) = (f64::NEG_INFINITY, f64::INFINITY);
                for r in &rets {
                    cum += r - mean_r;
                    mx = mx.max(cum);
                    mn = mn.min(cum);
                }
                let r_range = mx - mn;
                let var = rets.iter().map(|v| (v - mean_r).powi(2)).sum::<f64>()
                    / (rets.len() as f64 - 1.0); // ddof=1
                let s = var.sqrt();
                if s > 1e-12 {
                    rs_values.push(r_range / s);
                }
            }
            if !rs_values.is_empty() {
                log_lags.push((lag as f64).ln());
                log_rs.push((rs_values.iter().sum::<f64>() / rs_values.len() as f64).ln());
            }
        }
        if log_lags.len() >= 3 {
            result[t] = linreg_slope(&log_lags, &log_rs).clamp(0.0, 1.0);
        }
    }
    result
}

/// ATR expanding(1.0) vs contracting(0.0) over `lookback` bars; default 0.5.
fn atr_trend(high: &[f64], low: &[f64], close: &[f64], lookback: usize) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.5; n];
    if n < lookback + 1 {
        return result;
    }
    let mut tr = vec![0.0; n];
    for i in 0..n {
        let prev = if i == 0 { close[0] } else { close[i - 1] };
        tr[i] = (high[i] - low[i])
            .max((high[i] - prev).abs())
            .max((low[i] - prev).abs());
    }
    let half = lookback / 2;
    for t in lookback..n {
        let w = &tr[t + 1 - lookback..=t];
        let older = w[..half].iter().sum::<f64>() / half as f64;
        let recent = w[half..].iter().sum::<f64>() / (lookback - half) as f64;
        if older > 1e-12 {
            result[t] = ((recent / older - 0.5) / 1.0).clamp(0.0, 1.0);
        }
    }
    result
}

/// Linear-regression slope of volume over `lookback` bars, mapped to `[0, 1]`.
fn volume_trend(volume: &[f64], lookback: usize) -> Vec<f64> {
    let n = volume.len();
    let mut result = vec![0.5; n];
    if n < lookback {
        return result;
    }
    let xs: Vec<f64> = (0..lookback).map(|i| i as f64).collect();
    let x_mean = xs.iter().sum::<f64>() / lookback as f64;
    let x_denom = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();
    if x_denom < 1e-12 {
        return result;
    }
    for t in (lookback - 1)..n {
        let w = &volume[t + 1 - lookback..=t];
        let v_mean = w.iter().sum::<f64>() / lookback as f64;
        if v_mean < 1e-10 {
            continue;
        }
        let num: f64 = xs
            .iter()
            .zip(w)
            .map(|(x, v)| (x - x_mean) * (v - v_mean))
            .sum();
        let slope_pct = (num / x_denom) / v_mean;
        result[t] = ((slope_pct + 0.3) / 0.6).clamp(0.0, 1.0);
    }
    result
}

/// `momentum_bars`-bar momentum, stdev-normalised, clipped/scaled to `[-1, 1]`.
fn normalized_velocity(close: &[f64], momentum_bars: usize, norm_window: usize) -> Vec<f64> {
    let n = close.len();
    let mut vel = vec![0.0; n];
    for i in momentum_bars..n {
        vel[i] = (close[i] - close[i - momentum_bars]) / (close[i - momentum_bars] + EPS);
    }
    let vel_std = rolling_std(&vel, norm_window);
    (0..n)
        .map(|i| (vel[i] / (vel_std[i] + EPS) / 3.0).clamp(-1.0, 1.0))
        .collect()
}

/// Change of velocity (2nd derivative), normalised and scaled to `[-1, 1]`.
fn price_acceleration(close: &[f64], momentum_bars: usize, norm_window: usize) -> Vec<f64> {
    let n = close.len();
    let mut vel = vec![0.0; n];
    for i in momentum_bars..n {
        vel[i] = (close[i] - close[i - momentum_bars]) / (close[i - momentum_bars] + EPS);
    }
    let mut accel = vec![0.0; n];
    for i in (momentum_bars * 2)..n {
        accel[i] = vel[i] - vel[i - momentum_bars];
    }
    let vel_std = rolling_std(&vel, norm_window);
    (0..n)
        .map(|i| (accel[i] / (vel_std[i] + EPS) / 3.0).clamp(-1.0, 1.0))
        .collect()
}

/// Structural market phase ∈ {0.0, 0.33, 0.67, 1.0}.
fn market_phase(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let n = close.len();
    let mut result = vec![0.0; n];

    // Awesome Oscillator: SMA(median, 5) - SMA(median, 34), NaN→0.
    let median: Vec<f64> = (0..n).map(|i| (high[i] + low[i]) / 2.0).collect();
    let ao_fast = sma_full_window(&median, AO_FAST);
    let ao_slow = sma_full_window(&median, AO_SLOW);
    let ao: Vec<f64> = (0..n)
        .map(|i| {
            let v = ao_fast[i] - ao_slow[i];
            if v.is_finite() { v } else { 0.0 }
        })
        .collect();

    let roll_high = rolling_max(high, CONSOL_BARS);
    let roll_low = rolling_min(low, CONSOL_BARS);
    let price_pos: Vec<f64> = (0..n)
        .map(|i| (close[i] - roll_low[i]) / (roll_high[i] - roll_low[i] + EPS))
        .collect();

    let mut ao_slope = vec![0.0; n];
    for i in 3..n {
        ao_slope[i] = ao[i] - ao[i - 3];
    }

    let atr = atr(high, low, close, ATR_PERIOD);
    let is_tight: Vec<bool> = (0..n)
        .map(|i| (roll_high[i] - roll_low[i]) / (atr[i] + EPS) < 4.0)
        .collect();

    let lookback = CONSOL_BARS;
    for i in (AO_SLOW + lookback)..n {
        let (pp, slope, tight) = (price_pos[i], ao_slope[i], is_tight[i]);
        let win = &ao[i.saturating_sub(lookback)..=i];
        let mean_abs_ao = win.iter().map(|v| v.abs()).sum::<f64>() / win.len() as f64;
        result[i] = if tight && slope.abs() < mean_abs_ao * 0.3 {
            if pp > 0.6 { 1.0 } else { 0.0 }
        } else if slope > 0.0 && pp > 0.5 {
            0.33
        } else if slope < 0.0 && pp < 0.5 {
            0.67
        } else if slope < 0.0 && pp > 0.5 {
            1.0
        } else {
            0.0
        };
    }
    result
}

// ---------------------------------------------------------------------------
// Channel assembly
// ---------------------------------------------------------------------------

/// Compute all 20 channels over the given OHLCV arrays (inference path).
/// Returns a row-major `(N_FEATURES, n)` buffer (`channel * n + bar`), clipped
/// to `[-3, 3]` with non-finite values zeroed — matching `_compute_channels`.
fn compute_channels(
    op: &[f64],
    hi: &[f64],
    lo: &[f64],
    cl: &[f64],
    vo: &[f64],
    state: &LiveState,
    window: usize,
) -> Vec<f32> {
    let n = cl.len();
    let mut ch: Vec<Vec<f64>> = Vec::with_capacity(N_FEATURES);

    // 0 log_return
    let mut log_ret = vec![0.0; n];
    for i in 1..n {
        log_ret[i] = (cl[i] / (cl[i - 1] + EPS)).ln();
    }
    ch.push(log_ret);

    // 1 norm_close (z-score over `window`)
    let cl_mean = rolling_mean(cl, window);
    let cl_std = rolling_std(cl, window);
    ch.push(
        (0..n)
            .map(|i| (cl[i] - cl_mean[i]) / (cl_std[i] + 1e-8))
            .collect(),
    );

    // 2 norm_atr
    let atr_v = atr(hi, lo, cl, ATR_PERIOD);
    ch.push((0..n).map(|i| atr_v[i] / (cl[i] + EPS)).collect());

    // 3 volume_ratio
    let vol_mean = rolling_mean(vo, 20);
    ch.push((0..n).map(|i| vo[i] / (vol_mean[i] + EPS)).collect());

    // 4 rsi [0,1]
    ch.push(
        wilder_rsi(cl, RSI_PERIOD)
            .iter()
            .map(|r| r / 100.0)
            .collect(),
    );

    // 5 range_tightness, 6 close_vs_range
    let rh20 = rolling_max(hi, CONSOL_BARS);
    let rl20 = rolling_min(lo, CONSOL_BARS);
    ch.push(
        (0..n)
            .map(|i| ((rh20[i] - rl20[i]) / (atr_v[i] + EPS) / 5.0).clamp(0.0, 1.0))
            .collect(),
    );
    ch.push(
        (0..n)
            .map(|i| ((cl[i] - rl20[i]) / (rh20[i] - rl20[i] + EPS)).clamp(0.0, 1.0))
            .collect(),
    );

    // 7-9 live scalars (inference: constant broadcast)
    let imb = state.imbalance as f64;
    let imb_01 = (imb / (imb + 1.0)).clamp(0.0, 1.0);
    let wr_norm = (state.wave_ratio as f64).clamp(-3.0, 3.0) / 3.0;
    ch.push(vec![imb_01; n]);
    ch.push(vec![wr_norm; n]);
    ch.push(vec![state.vol_pct as f64; n]);

    // 10-12 candle shape
    let rng: Vec<f64> = (0..n).map(|i| hi[i] - lo[i] + EPS).collect();
    ch.push((0..n).map(|i| (cl[i] - op[i]).abs() / rng[i]).collect());
    ch.push(
        (0..n)
            .map(|i| (hi[i] - op[i].max(cl[i])) / rng[i])
            .collect(),
    );
    ch.push(
        (0..n)
            .map(|i| (op[i].min(cl[i]) - lo[i]) / rng[i])
            .collect(),
    );

    // 13 candle_dir (np.sign: -1/0/+1)
    ch.push(
        (0..n)
            .map(|i| {
                let d = cl[i] - op[i];
                if d > 0.0 {
                    1.0
                } else if d < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect(),
    );

    // 14-19
    ch.push(hurst(cl, 20));
    ch.push(atr_trend(hi, lo, cl, 10));
    ch.push(volume_trend(vo, 5));
    ch.push(normalized_velocity(cl, 3, 100));
    ch.push(price_acceleration(cl, 3, 100));
    ch.push(market_phase(hi, lo, cl));

    debug_assert_eq!(ch.len(), N_FEATURES);

    // Stack → (N_FEATURES, n), clip [-3,3], zero non-finite.
    let mut out = vec![0.0f32; N_FEATURES * n];
    for (c, channel) in ch.iter().enumerate() {
        for (i, &v) in channel.iter().enumerate() {
            let v = if v.is_finite() {
                v.clamp(-3.0, 3.0)
            } else {
                0.0
            };
            out[c * n + i] = v as f32;
        }
    }
    out
}

/// Minimum bars required by [`extract_features`]: the returned `window` plus
/// the [`WARMUP`] context needed to fully warm every channel.
pub fn min_rows(window: usize) -> usize {
    window + WARMUP
}

/// Compute all [`N_FEATURES`] channels over the WHOLE series → row-major
/// `(N_FEATURES, n_bars)` (`channel * n_bars + bar`). The training counterpart
/// of [`extract_features`] (which returns only the last window). Uses the
/// inference-path channels (7–9 are the constant live-scalar broadcasts).
/// Returns `None` if the input lengths disagree or there are fewer than
/// `window` bars.
pub fn precompute_features(
    open: &[f32],
    high: &[f32],
    low: &[f32],
    close: &[f32],
    volume: &[f32],
    state: &LiveState,
    window: usize,
) -> Option<Vec<f32>> {
    let n = close.len();
    if n < window
        || [open.len(), high.len(), low.len(), volume.len()]
            .iter()
            .any(|&l| l != n)
    {
        return None;
    }
    let f64s = |a: &[f32]| a.iter().map(|&v| v as f64).collect::<Vec<_>>();
    Some(compute_channels(
        &f64s(open),
        &f64s(high),
        &f64s(low),
        &f64s(close),
        &f64s(volume),
        state,
        window,
    ))
}

/// Extract the `(N_FEATURES, window)` feature tensor (row-major, `channel *
/// window + bar`) for `PerAssetCnn` — the live inference path of the Python
/// `extract_features`. Returns `None` if fewer than [`min_rows`] bars.
///
/// Slices match the Python tail behaviour: channels are computed over the last
/// `window + warmup` bars and the final `window` columns are returned.
pub fn extract_features(
    open: &[f32],
    high: &[f32],
    low: &[f32],
    close: &[f32],
    volume: &[f32],
    state: &LiveState,
    window: usize,
) -> Option<Vec<f32>> {
    let n = close.len();
    if n < min_rows(window) {
        return None;
    }
    if [open.len(), high.len(), low.len(), volume.len()]
        .iter()
        .any(|&l| l != n)
    {
        return None;
    }
    let take = window + WARMUP;
    let s = n - take;
    let f64s = |a: &[f32]| a[s..].iter().map(|&v| v as f64).collect::<Vec<_>>();

    let channels = compute_channels(
        &f64s(open),
        &f64s(high),
        &f64s(low),
        &f64s(close),
        &f64s(volume),
        state,
        window,
    );

    // channels is (N_FEATURES, take); take the last `window` columns.
    let mut out = vec![0.0f32; N_FEATURES * window];
    for c in 0..N_FEATURES {
        let row = &channels[c * take..(c + 1) * take];
        out[c * window..(c + 1) * window].copy_from_slice(&row[take - window..]);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn ewm_adjust_false_matches_recurrence() {
        // span=2 → alpha=2/3. y0=1; y1=2/3*2+1/3*1=1.6667; y2=2/3*3+1/3*1.6667=2.5556
        let y = ewm_adjust_false(&[1.0, 2.0, 3.0], 2);
        assert!(approx(y[0], 1.0, 1e-9));
        assert!(approx(y[1], 1.0 + 2.0 / 3.0 * 1.0, 1e-9)); // 1.6667
        assert!(approx(y[2], 2.0 / 3.0 * 3.0 + 1.0 / 3.0 * y[1], 1e-9));
    }

    #[test]
    fn rolling_mean_expands_then_windows() {
        let m = rolling_mean(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(m, vec![1.0, 1.5, 2.5, 3.5]); // min_periods=1
    }

    #[test]
    fn rolling_std_ddof1_with_fillna() {
        let s = rolling_std(&[1.0, 3.0, 5.0], 3);
        assert!(approx(s[0], 1.0, 1e-12)); // count<2 → fillna 1.0
        assert!(approx(s[1], (2.0f64).sqrt(), 1e-12)); // std([1,3]) ddof1 = sqrt(2)
        assert!(approx(s[2], 2.0, 1e-12)); // std([1,3,5]) ddof1 = 2
    }

    #[test]
    fn rolling_min_max_windowed() {
        assert_eq!(
            rolling_max(&[3.0, 1.0, 2.0, 5.0], 2),
            vec![3.0, 3.0, 2.0, 5.0]
        );
        assert_eq!(
            rolling_min(&[3.0, 1.0, 2.0, 5.0], 2),
            vec![3.0, 1.0, 1.0, 2.0]
        );
    }

    #[test]
    fn rsi_no_losses_caps_rs_at_100() {
        // strictly rising closes → avg_loss==0 → Python caps rs=100, so
        // rsi = 100 - 100/101 = 99.0099 (NOT 100). rsi[0]=50.
        let close: Vec<f64> = (0..40).map(|i| 100.0 + i as f64).collect();
        let r = wilder_rsi(&close, 14);
        assert!(approx(r[0], 50.0, 1e-9));
        assert!(approx(*r.last().unwrap(), 100.0 - 100.0 / 101.0, 1e-4));
    }

    #[test]
    fn atr_of_constant_bars() {
        // high-low=2 every bar, no gaps → TR=2 → ATR ewm converges to 2.
        let (h, l, c) = (vec![3.0; 30], vec![1.0; 30], vec![2.0; 30]);
        let a = atr(&h, &l, &c, 14);
        assert!(approx(*a.last().unwrap(), 2.0, 1e-6));
    }

    #[test]
    fn candle_dir_sign() {
        // doji (close==open) must be 0, not +1 (np.sign semantics).
        let out = compute_channels(
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &LiveState::default(),
            2,
        );
        let n = 2;
        // channel 13 = candle_dir
        assert_eq!(out[13 * n], 0.0);
        assert_eq!(out[13 * n + 1], 0.0);
    }

    #[test]
    fn volume_trend_rising_above_half() {
        let v: Vec<f64> = (0..10).map(|i| 100.0 + 10.0 * i as f64).collect();
        let t = volume_trend(&v, 5);
        assert!(*t.last().unwrap() > 0.5, "rising volume should map >0.5");
    }

    /// Contract: shape, value bounds, warmup defaults, scalar broadcasts.
    #[test]
    fn extract_features_contract() {
        let window = 60;
        let n = min_rows(window) + 40;
        // deterministic synthetic OHLCV
        let close: Vec<f32> = (0..n)
            .map(|i| 100.0 + (i as f32 * 0.3).sin() * 5.0)
            .collect();
        let high: Vec<f32> = close.iter().map(|c| c + 1.0).collect();
        let low: Vec<f32> = close.iter().map(|c| c - 1.0).collect();
        let open: Vec<f32> = close.iter().map(|c| c - 0.2).collect();
        let volume: Vec<f32> = (0..n).map(|i| 1000.0 + (i % 7) as f32 * 50.0).collect();
        let state = LiveState {
            imbalance: 2.0,
            vol_pct: 0.7,
            wave_ratio: 1.5,
        };

        let feats = extract_features(&open, &high, &low, &close, &volume, &state, window).unwrap();
        assert_eq!(feats.len(), N_FEATURES * window);
        assert_eq!(feature_names().len(), N_FEATURES);

        for (i, &v) in feats.iter().enumerate() {
            assert!(
                v.is_finite() && (-3.0..=3.0).contains(&v),
                "feat[{i}]={v} out of [-3,3]"
            );
        }
        // channel 7 (book_imbalance) = clip(2/(2+1)) = 0.6667, constant
        let bi = &feats[7 * window..8 * window];
        assert!(bi.iter().all(|&v| approx(v as f64, 2.0 / 3.0, 1e-6)));
        // channel 9 (vol_percentile) = 0.7 constant
        assert!(
            feats[9 * window..10 * window]
                .iter()
                .all(|&v| approx(v as f64, 0.7, 1e-6))
        );

        // too few rows → None
        assert!(
            extract_features(
                &open[..10],
                &high[..10],
                &low[..10],
                &close[..10],
                &volume[..10],
                &state,
                window
            )
            .is_none()
        );
    }

    /// Train/serve feature-parity regression guard.
    ///
    /// The live path ([`extract_features`], bounded to the last
    /// `window + WARMUP` bars) must produce the SAME features as the full-series
    /// training path ([`precompute_features`], sliced to the last window) for
    /// the same terminal bars — otherwise the CNN champion is served
    /// out-of-distribution features. Before `WARMUP = 110` the inference warmup
    /// was 25, leaving history-dependent channels under-warmed or at their
    /// defaults (measured divergence up to 1.0 on `market_phase`, 0.5 on
    /// `hurst`, plus `norm_close`/`velocity`/`accel`). This asserts the skew is
    /// gone: fixed-window channels match ~exactly; the four exponential-moving
    /// channels (`norm_atr`, `rsi`, `range_tightness`, `atr_trend`) converge to
    /// within a tiny tolerance over the warmup.
    #[test]
    fn bounded_warmup_matches_full_series() {
        let window = 60;
        // Long series: the full-series path sees deep history the bounded path
        // cannot. If WARMUP is deep enough, the terminal window still matches.
        let n = min_rows(window) + 300;
        let close: Vec<f32> = (0..n)
            .map(|i| 100.0 + (i as f32 * 0.11).sin() * 8.0 + (i as f32 * 0.017).cos() * 3.0)
            .collect();
        let high: Vec<f32> = close.iter().map(|c| c + 1.3).collect();
        let low: Vec<f32> = close.iter().map(|c| c - 1.1).collect();
        let open: Vec<f32> = close.iter().map(|c| c - 0.2).collect();
        let volume: Vec<f32> = (0..n)
            .map(|i| 1000.0 + ((i * 13) % 97) as f32 * 7.0)
            .collect();
        let st = LiveState::default();

        let full = precompute_features(&open, &high, &low, &close, &volume, &st, window).unwrap();
        let inf = extract_features(&open, &high, &low, &close, &volume, &st, window).unwrap();

        // Channels that read an exponential-moving stat (Wilder RSI / EWM ATR)
        // converge rather than snap — allow a small tolerance for them and
        // require ~bit-exactness for every fixed-window channel.
        let ewm = [2usize, 4, 5, 15]; // norm_atr, rsi, range_tightness, atr_trend
        let (mut max_fixed, mut max_ewm) = (0f32, 0f32);
        for c in 0..N_FEATURES {
            for j in 0..window {
                let d = (inf[c * window + j] - full[c * n + (n - window + j)]).abs();
                if ewm.contains(&c) {
                    max_ewm = max_ewm.max(d);
                } else {
                    max_fixed = max_fixed.max(d);
                }
            }
        }
        assert!(
            max_fixed < 1e-4,
            "fixed-window channels diverge (train/serve skew): {max_fixed}"
        );
        assert!(
            max_ewm < 1e-2,
            "exponential channels beyond convergence tolerance: {max_ewm}"
        );
    }

    /// Full Python↔Rust parity. Skips when no goldens are present (generate with
    /// `tools/parity/record_features.py` in a numpy env).
    #[test]
    fn python_goldens() {
        #[derive(serde::Deserialize)]
        struct Case {
            open: Vec<f32>,
            high: Vec<f32>,
            low: Vec<f32>,
            close: Vec<f32>,
            volume: Vec<f32>,
            imbalance: f32,
            vol_pct: f32,
            wave_ratio: f32,
            window: usize,
            expected: Vec<f32>, // (N_FEATURES * window), row-major
        }
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/golden/per_asset_features"
        );
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => {
                eprintln!("[python_goldens] {dir} absent — skipping");
                return;
            }
        };
        let mut checked = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            let case: Case = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            let got = extract_features(
                &case.open,
                &case.high,
                &case.low,
                &case.close,
                &case.volume,
                &LiveState {
                    imbalance: case.imbalance,
                    vol_pct: case.vol_pct,
                    wave_ratio: case.wave_ratio,
                },
                case.window,
            )
            .expect("golden should have enough rows");
            assert_eq!(got.len(), case.expected.len(), "{path:?}: length");
            for (i, (g, e)) in got.iter().zip(case.expected.iter()).enumerate() {
                assert!((g - e).abs() < 2e-3, "{path:?}: feat[{i}] {g} vs {e}");
            }
            checked += 1;
        }
        eprintln!("[python_goldens] verified {checked} golden file(s)");
    }
}
