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
use janus_ml::models::PerAssetCnn;
use janus_ml::train_per_asset::{
    CLASS_NAMES, TrainChampionConfig, TrainReport, ValMetrics, train_champion,
    train_champion_with_holdout,
};

const HELP: &str = "train_cnn_champion --csv <ohlcv.csv> --out <model.bin> \
[--window 60] [--epochs 60] [--val-frac 0.2]";

struct Args {
    csv: String,
    out: String,
    window: usize,
    epochs: usize,
    /// Fraction of the series held out (by time) for out-of-sample validation.
    /// 0 = train on all bars (no validation, the default).
    val_frac: f64,
}

fn parse_args() -> Result<Args, String> {
    let mut csv = None;
    let mut out = None;
    let mut window = 60usize;
    let mut epochs = 60usize;
    let mut val_frac = 0.0f64;
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
    })
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

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (o, h, l, c, v) = read_ohlcv(&args.csv)?;
    println!("loaded {} bars from {}", c.len(), args.csv);

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

    // With --val-frac, train on a leakage-safe time split and report
    // out-of-sample generalization; the saved artifact is the holdout-trained
    // model (retrain on all bars only once it clears validation). Otherwise
    // train on every bar (the plain minting path).
    let model = if args.val_frac > 0.0 {
        let (model, report, val) =
            train_champion_with_holdout(&o, &h, &l, &c, &v, &cfg, args.val_frac)
                .ok_or_else(too_short)?;
        print_train(&report);
        print_val(&val);
        model
    } else {
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
