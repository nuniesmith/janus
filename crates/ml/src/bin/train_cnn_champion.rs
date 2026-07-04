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
use janus_ml::train_per_asset::{TrainChampionConfig, train_champion};

const HELP: &str = "train_cnn_champion --csv <ohlcv.csv> --out <model.bin> \
[--window 60] [--epochs 60]";

struct Args {
    csv: String,
    out: String,
    window: usize,
    epochs: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut csv = None;
    let mut out = None;
    let mut window = 60usize;
    let mut epochs = 60usize;
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
            "-h" | "--help" => return Err(HELP.to_string()),
            other => return Err(format!("unknown arg: {other}\n{HELP}")),
        }
    }
    Ok(Args {
        csv: csv.ok_or_else(|| format!("missing --csv\n{HELP}"))?,
        out: out.ok_or_else(|| format!("missing --out\n{HELP}"))?,
        window,
        epochs,
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

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let (o, h, l, c, v) = read_ohlcv(&args.csv)?;
    println!("loaded {} bars from {}", c.len(), args.csv);

    let cfg = TrainChampionConfig {
        window: args.window,
        epochs: args.epochs,
        ..Default::default()
    };
    let (model, report) = train_champion(&o, &h, &l, &c, &v, &cfg).ok_or_else(|| {
        format!(
            "not enough bars ({}) to form any training sample at window={} \
             (need window + ~110 warmup + ~130 label-horizon bars)",
            c.len(),
            args.window,
        )
    })?;

    let [flat, long, short, loss] = report.label_counts;
    println!(
        "trained: epochs_run={} samples={} best_loss={:.4} \
         labels[flat={flat} long={long} short={short} loss={loss}]",
        report.epochs_run, report.samples, report.best_loss,
    );

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
