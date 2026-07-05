//! Champion-minting CLI for the per-asset CNN.
//!
//! Reads an OHLCV CSV (a header row naming `open,high,low,close,volume`
//! columns — any order, extra columns ignored), trains a `PerAssetCnn`
//! champion via [`janus_ml::train_per_asset::train_champion`], writes the
//! postcard checkpoint, then reloads it to prove the artifact is valid.
//!
//! This is the missing *producer* for the live CNN-vote path
//! (`ENABLE_CNN_INFERENCE` / `CNN_CHECKPOINT_PATH`) — the training algorithm
//! already existed and was tested, but nothing called it and no champion
//! artifact existed. See `docs/architecture/CNN_LIVE_ONRAMP.md`.
//!
//! It only MINTS an artifact for review; it enables nothing. Producing OHLCV
//! from the live stack (QuestDB) is intentionally out of band — export CSV via
//! QuestDB's `/exp` endpoint and pass it with `--csv`.
//!
//! Usage:
//!   train_cnn_champion --csv <ohlcv.csv> --out <model.bin> [--window 60] [--epochs 60]

use janus_ml::CpuBackend;
use janus_ml::backend::AutodiffCpuBackend;
use janus_ml::backtest::BacktestConfig;
use janus_ml::models::PerAssetCnn;
use janus_ml::train_per_asset::{
    CLASS_NAMES, TrainChampionConfig, TrainReport, ValMetrics, WalkForwardBacktest,
    WalkForwardReport, train_champion, train_champion_with_holdout, train_walk_forward,
    walk_forward_backtest,
};

const HELP: &str = "train_cnn_champion --csv <ohlcv.csv> --out <model.bin> \
[--window 60] [--epochs 60] [--val-frac 0.2] [--walk-forward N] [--backtest] \
[--cost-bps 6] [--gpu]";

struct Args {
    csv: String,
    out: String,
    window: usize,
    epochs: usize,
    /// Fraction of the series held out (by time) for a single out-of-sample
    /// split. 0 = train on all bars (no validation, the default).
    val_frac: f64,
    /// Walk-forward folds. >0 runs anchored walk-forward validation across N
    /// rolling out-of-sample segments (no artifact saved) and overrides
    /// --val-frac.
    walk_forward: usize,
    /// Run a walk-forward PnL backtest of the directional signal (per-side cost
    /// = `cost_bps`), instead of classification metrics.
    backtest: bool,
    /// Per-side trading cost in basis points (fee + slippage) for --backtest.
    cost_bps: f64,
    /// Train on the GPU (requires building with `--features gpu`).
    gpu: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut csv = None;
    let mut out = None;
    let mut window = 60usize;
    let mut epochs = 60usize;
    let mut val_frac = 0.0f64;
    let mut walk_forward = 0usize;
    let mut backtest = false;
    let mut cost_bps = 6.0f64;
    let mut gpu = false;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--csv" => csv = it.next(),
            "--out" => out = it.next(),
            "--window" => {
                window = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--window needs a positive integer".to_string())?
            }
            "--epochs" => {
                epochs = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--epochs needs a positive integer".to_string())?
            }
            "--val-frac" => {
                val_frac = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--val-frac needs a float in (0, 0.9)".to_string())?
            }
            "--walk-forward" => {
                walk_forward = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--walk-forward needs a positive integer".to_string())?
            }
            "--backtest" => backtest = true,
            "--cost-bps" => {
                cost_bps = it
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "--cost-bps needs a number".to_string())?
            }
            "--gpu" => gpu = true,
            "-h" | "--help" => return Err(HELP.to_string()),
            other => return Err(format!("unknown arg: {other}\n{HELP}")),
        }
    }
    Ok(Args {
        csv: csv.ok_or_else(|| format!("missing --csv\n{HELP}"))?,
        out: out.ok_or_else(|| format!("missing --out\n{HELP}"))?,
        window,
        epochs,
        val_frac,
        walk_forward,
        backtest,
        cost_bps,
        gpu,
    })
}

/// Which backend a run used (for the log line).
fn backend_label(gpu: bool) -> &'static str {
    if gpu && cfg!(feature = "gpu") {
        "gpu(wgpu)"
    } else if gpu {
        "cpu (--gpu ignored: built without --features gpu)"
    } else {
        "cpu"
    }
}

/// Dispatch holdout training to the GPU backend when requested and available,
/// else the CPU backend.
#[allow(clippy::too_many_arguments)]
fn holdout_dispatch(
    gpu: bool,
    o: &[f32],
    h: &[f32],
    l: &[f32],
    c: &[f32],
    v: &[f32],
    cfg: &TrainChampionConfig,
    val_frac: f64,
) -> Option<(PerAssetCnn<CpuBackend>, TrainReport, ValMetrics)> {
    if gpu {
        #[cfg(feature = "gpu")]
        {
            use janus_ml::backend::AutodiffGpuBackend;
            return train_champion_with_holdout::<AutodiffGpuBackend>(o, h, l, c, v, cfg, val_frac);
        }
    }
    train_champion_with_holdout::<AutodiffCpuBackend>(o, h, l, c, v, cfg, val_frac)
}

/// Dispatch walk-forward validation to the GPU backend when requested and
/// available, else the CPU backend.
#[allow(clippy::too_many_arguments)]
fn walk_forward_dispatch(
    gpu: bool,
    o: &[f32],
    h: &[f32],
    l: &[f32],
    c: &[f32],
    v: &[f32],
    cfg: &TrainChampionConfig,
    n_folds: usize,
) -> Option<WalkForwardReport> {
    if gpu {
        #[cfg(feature = "gpu")]
        {
            use janus_ml::backend::AutodiffGpuBackend;
            return train_walk_forward::<AutodiffGpuBackend>(o, h, l, c, v, cfg, n_folds);
        }
    }
    train_walk_forward::<AutodiffCpuBackend>(o, h, l, c, v, cfg, n_folds)
}

/// Dispatch the walk-forward PnL backtest to GPU when requested/available.
#[allow(clippy::too_many_arguments)]
fn backtest_dispatch(
    gpu: bool,
    o: &[f32],
    h: &[f32],
    l: &[f32],
    c: &[f32],
    v: &[f32],
    cfg: &TrainChampionConfig,
    n_folds: usize,
    bt: &BacktestConfig,
) -> Option<WalkForwardBacktest> {
    if gpu {
        #[cfg(feature = "gpu")]
        {
            use janus_ml::backend::AutodiffGpuBackend;
            return walk_forward_backtest::<AutodiffGpuBackend>(o, h, l, c, v, cfg, n_folds, bt);
        }
    }
    walk_forward_backtest::<AutodiffCpuBackend>(o, h, l, c, v, cfg, n_folds, bt)
}

type Ohlcv = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>);

/// Parse an OHLCV CSV, mapping columns by header name so the query column order
/// (and any extra columns such as `timestamp`) does not matter. QuestDB `/exp`
/// quotes header names, so quotes/whitespace are trimmed before matching.
fn read_ohlcv(path: &str) -> Result<Ohlcv, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| "empty CSV".to_string())?;
    let cols: Vec<String> = header
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_ascii_lowercase())
        .collect();
    let col = |name: &str| {
        cols.iter()
            .position(|c| c == name)
            .ok_or_else(|| format!("CSV missing column '{name}' (have: {cols:?})"))
    };
    let (io, ih, il, ic, iv) = (
        col("open")?,
        col("high")?,
        col("low")?,
        col("close")?,
        col("volume")?,
    );
    let need = [io, ih, il, ic, iv].into_iter().max().unwrap();

    let (mut o, mut h, mut l, mut c, mut v) = (vec![], vec![], vec![], vec![], vec![]);
    for (n, line) in lines.enumerate() {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() <= need {
            return Err(format!(
                "row {}: expected >{need} fields, got {}",
                n + 2,
                f.len()
            ));
        }
        let parse = |i: usize| -> Result<f32, String> {
            f[i].trim()
                .trim_matches('"')
                .parse::<f32>()
                .map_err(|e| format!("row {}: bad number '{}': {e}", n + 2, f[i]))
        };
        o.push(parse(io)?);
        h.push(parse(ih)?);
        l.push(parse(il)?);
        c.push(parse(ic)?);
        v.push(parse(iv)?);
    }
    if c.is_empty() {
        return Err("CSV has a header but no data rows".to_string());
    }
    Ok((o, h, l, c, v))
}

/// Print the in-sample training summary.
fn print_train(report: &TrainReport) {
    let [flat, long, short, loss] = report.label_counts;
    println!(
        "trained: epochs_run={} samples={} best_loss={:.4} (in-sample) \
         labels[flat={flat} long={long} short={short} loss={loss}]",
        report.epochs_run, report.samples, report.best_loss,
    );
}

/// Print the out-of-sample validation report + an honest generalization verdict.
fn print_val(m: &ValMetrics) {
    println!(
        "\nout-of-sample validation (leakage-safe: time split at bar {}, purge {} bars, \
         {} train / {} val samples):",
        m.split_bar, m.purge, m.n_train, m.n_val
    );
    println!("  accuracy          {:.3}", m.accuracy);
    println!(
        "  majority baseline {:.3}   (accuracy of always-predict-majority)",
        m.majority_baseline
    );
    println!("  per class  (precision / recall / support / base-rate):");
    for (c, s) in m.per_class.iter().enumerate() {
        let edge = if s.support > 0 && s.precision > s.base_rate + 0.02 {
            " ← precision > base rate"
        } else {
            ""
        };
        println!(
            "    {:5}  P {:.3}  R {:.3}  n {:<5} base {:.3}{edge}",
            CLASS_NAMES[c], s.precision, s.recall, s.support, s.base_rate,
        );
    }
    // Actionable edge: do long OR short calls beat their own base rate?
    let long = &m.per_class[1];
    let short = &m.per_class[2];
    let long_edge = long.support > 0 && long.precision > long.base_rate + 0.02;
    let short_edge = short.support > 0 && short.precision > short.base_rate + 0.02;
    let verdict = if m.accuracy <= m.majority_baseline + 0.02 {
        "does NOT beat the majority baseline → no generalization on this data. \
         Do not enable; needs more/better data or a different label/model."
    } else if !long_edge && !short_edge {
        "beats the accuracy baseline but neither long nor short precision clears \
         its base rate → no tradable edge on the actionable classes yet."
    } else {
        "beats baseline AND at least one actionable class clears its base rate → \
         a real (if early) signal; warrants a deeper walk-forward before enabling."
    };
    println!("  VERDICT: {verdict}");
}

/// Print the walk-forward report: one line per fold + the aggregate verdict.
fn print_walk_forward(r: &WalkForwardReport) {
    println!(
        "\nwalk-forward validation — {} folds, each trained on all prior bars \
         (purged) and validated on a distinct later window:",
        r.folds.len()
    );
    println!("  fold  split_bar  n_val   acc    baseline  long_P(base)   short_P(base)");
    for (i, m) in r.folds.iter().enumerate() {
        let lp = &m.per_class[1];
        let sp = &m.per_class[2];
        println!(
            "  {:>3}   {:>8}  {:>6}  {:.3}  {:.3}     {:.3}({:.3})   {:.3}({:.3})",
            i,
            m.split_bar,
            m.n_val,
            m.accuracy,
            m.majority_baseline,
            lp.precision,
            lp.base_rate,
            sp.precision,
            sp.base_rate,
        );
    }
    println!(
        "  MEAN: acc {:.3} vs baseline {:.3} | long_P {:.3} (base {:.3}) | short_P {:.3} (base {:.3})",
        r.mean_accuracy,
        r.mean_majority_baseline,
        r.mean_long_precision,
        r.mean_long_base,
        r.mean_short_precision,
        r.mean_short_base,
    );
    // Directional robustness — the trading-relevant signal. Overall accuracy vs
    // the majority (mostly-flat) baseline is a poor gauge here: a signal that
    // only ACTS on the minority long/short classes sacrifices accuracy on the
    // dominant flat class by design. What matters is whether long/short calls
    // beat their own base rate, and whether that holds across folds (regimes).
    let n = r.folds.len();
    let long_beats = r
        .folds
        .iter()
        .filter(|m| m.per_class[1].precision > m.per_class[1].base_rate + 0.02)
        .count();
    let short_beats = r
        .folds
        .iter()
        .filter(|m| m.per_class[2].precision > m.per_class[2].base_rate + 0.02)
        .count();
    println!(
        "  directional edge held in: long {long_beats}/{n} folds, short {short_beats}/{n} folds \
         (accuracy beat baseline in {}/{n} — expected-low for a minority-class signal).",
        r.folds_beating_baseline,
    );

    let dir_robust = long_beats * 2 >= n && short_beats * 2 >= n; // both, majority of folds
    let mean_dir_edge = r.mean_long_precision > r.mean_long_base + 0.02
        && r.mean_short_precision > r.mean_short_base + 0.02;
    let some_edge = long_beats * 2 >= n || short_beats * 2 >= n;
    let verdict = if dir_robust && mean_dir_edge {
        "ROBUST DIRECTIONAL EDGE — long AND short precision beat their base rate \
         in a majority of folds (mean ~2x base), across distinct time windows, so \
         it is not one lucky split. NOTE: absolute precision is still low (~2x \
         chance ≠ mostly-right), and classification edge ≠ profit. Next gate is a \
         PnL backtest through the consensus + risk gates — do NOT enable on \
         classification metrics alone."
    } else if some_edge {
        "PARTIAL DIRECTIONAL EDGE — one direction beats its base rate across folds \
         but the other does not. Suggestive but asymmetric; more data / label work \
         before a backtest."
    } else {
        "NO ROBUST EDGE — directional precision does not consistently beat its base \
         rate across folds. Do not enable."
    };
    println!("  VERDICT: {verdict}");
}

/// Print the walk-forward PnL backtest + a money verdict.
fn print_backtest(r: &WalkForwardBacktest, cost_bps: f64) {
    println!(
        "\nwalk-forward PnL backtest — CNN signal standalone, {:.0} bps/side cost, \
         TP 1.5 / SL 1.0 ATR bracket, one position at a time:",
        cost_bps
    );
    println!("  fold  trades  win%    net_return  sum_R    sharpe  max_dd");
    for (i, f) in r.folds.iter().enumerate() {
        println!(
            "  {:>3}   {:>5}  {:>5.1}  {:>+9.4}  {:>+6.2}  {:>+.3}  {:.4}",
            i,
            f.n_trades,
            f.win_rate * 100.0,
            f.net_return,
            f.sum_r,
            f.sharpe,
            f.max_drawdown,
        );
    }
    let win_pct = if r.total_trades > 0 {
        100.0 * r.total_wins as f64 / r.total_trades as f64
    } else {
        0.0
    };
    println!(
        "  TOTAL: {} trades, {:.1}% win, net_return {:+.4} ({:+.2} R), \
         mean fold Sharpe {:+.3}, {}/{} folds profitable, worst fold DD {:.4}",
        r.total_trades,
        win_pct,
        r.total_net_return,
        r.total_sum_r,
        r.mean_sharpe,
        r.folds_profitable,
        r.folds.len(),
        r.worst_drawdown,
    );
    let n = r.folds.len();
    let consistent = r.folds_profitable * 2 >= n; // majority of folds green
    let verdict = if r.total_net_return > 0.0 && consistent && r.total_sum_r > 0.0 {
        "NET PROFITABLE after costs across the majority of out-of-sample folds. \
         This clears the standalone bar. Since the live loop adds consensus + \
         risk gates that only filter further, the next step is a full gated \
         paper-mode trial — NOT a live-money enable. Beware: transaction-cost \
         and fill assumptions dominate at this trade frequency; stress-test them."
    } else if r.total_net_return > 0.0 {
        "marginally net positive but NOT consistent across folds — fragile, and \
         likely inside the cost/slippage error bars. Not a green light."
    } else {
        "NET NEGATIVE after costs — the ~2x directional precision does NOT survive \
         trading costs at this frequency/bracket. The classification edge is real \
         but not, as-is, tradable. Do not enable; iterate on costs/holding/label."
    };
    println!("  VERDICT: {verdict}");
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (o, h, l, c, v) = read_ohlcv(&args.csv)?;
    println!(
        "loaded {} bars from {} (backend: {})",
        c.len(),
        args.csv,
        backend_label(args.gpu)
    );

    let cfg = TrainChampionConfig {
        window: args.window,
        epochs: args.epochs,
        ..Default::default()
    };

    let too_short = || {
        format!(
            "not enough bars ({}) to form the requested split at window={} \
             (need window + ~110 warmup + ~130 label-horizon bars, and more for a holdout)",
            c.len(),
            args.window,
        )
    };

    // --backtest: walk-forward PnL backtest of the directional signal. A
    // validation experiment — no artifact is saved.
    if args.backtest {
        let n_folds = if args.walk_forward > 0 {
            args.walk_forward
        } else {
            5
        };
        let bt = BacktestConfig {
            cost_per_side: args.cost_bps / 10_000.0,
            ..Default::default()
        };
        let report = backtest_dispatch(args.gpu, &o, &h, &l, &c, &v, &cfg, n_folds, &bt)
            .ok_or_else(too_short)?;
        print_backtest(&report, args.cost_bps);
        return Ok(());
    }

    // --walk-forward: multi-fold out-of-sample validation across distinct time
    // windows. A validation experiment — no artifact is saved.
    if args.walk_forward > 0 {
        let report = walk_forward_dispatch(args.gpu, &o, &h, &l, &c, &v, &cfg, args.walk_forward)
            .ok_or_else(too_short)?;
        print_walk_forward(&report);
        return Ok(());
    }

    // With --val-frac, train on a leakage-safe time split and report
    // out-of-sample generalization; the saved artifact is the holdout-trained
    // model (retrain on all bars only once it clears validation). Otherwise
    // train on every bar (the plain minting path).
    let model = if args.val_frac > 0.0 {
        let (model, report, val) =
            holdout_dispatch(args.gpu, &o, &h, &l, &c, &v, &cfg, args.val_frac)
                .ok_or_else(too_short)?;
        print_train(&report);
        print_val(&val);
        model
    } else {
        // Plain minting path is CPU-only (fast enough); --gpu applies to
        // validation runs where many trainings dominate.
        let (model, report) = train_champion(&o, &h, &l, &c, &v, &cfg).ok_or_else(too_short)?;
        print_train(&report);
        model
    };

    model
        .save(&args.out)
        .map_err(|e| format!("save {}: {e}", args.out))?;

    // Roundtrip: prove the artifact is loadable by the same path the live
    // forward loop uses (PerAssetCnn::load), so a saved champion can never be
    // silently unusable at inference time.
    let device = Default::default();
    PerAssetCnn::<CpuBackend>::load(&args.out, &device)
        .map_err(|e| format!("roundtrip load failed for {}: {e}", args.out))?;

    println!("✅ champion saved + reloaded OK → {}", args.out);
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
