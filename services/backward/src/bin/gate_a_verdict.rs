//! `gate_a_verdict` — the committed, re-runnable Gate-A janus-side verdict
//! script (GATE_A_PREREGISTRATION.md §4a: J1 pass/fail, J4 informational,
//! and the M2 enforce-flip blocked-vs-actionable prerequisite).
//!
//! Scrolls the Qdrant `experiences` collection DIRECTLY via the experience
//! store's paginated window scroll (`ExperienceStore::scroll_window`) — the
//! HTTP sample endpoint caps at 5000 points and cannot serve a ~200k-row
//! two-week window. All statistics live in `janus_backward::verdict` (unit
//! tested); this binary is arg-parsing + I/O.
//!
//! Output: human summary → stderr, machine-readable JSON report → stdout
//! (pipe stdout to a file to archive the verdict record).
//!
//! ```text
//! USAGE: gate_a_verdict [OPTIONS]
//!   --from <RFC3339>            window start (default 2026-07-18T00:00:00Z — frozen)
//!   --to <RFC3339>              window end, inclusive (default 2026-08-01T23:59:59Z — frozen)
//!   --fee-boundary-ms <i64>     fee-aware cohort boundary, epoch ms
//!                               (default = 2026-07-10T00:00:00Z, conservative:
//!                               one full UTC day after the FEE_BPS=8 deploy;
//!                               always recorded in the output header)
//!   --qdrant-url <URL>          override QDRANT_URL
//!   --collection <NAME>         override QDRANT_EXPERIENCE_COLLECTION
//!   --page-size <N>             scroll page size (default 1024)
//!   --max-points <N>            runaway guard (default 2,000,000)
//!   --help
//! ```
//!
//! Connection settings otherwise come from the same env the backward service
//! uses (`QDRANT_URL`, `QDRANT_EXPERIENCE_COLLECTION`). The verdict REFUSES
//! to run against a mock store — a measurement instrument must never
//! fabricate its population.

use anyhow::{Context, Result, bail};
use janus_backward::persistence::experience_store::{ExperienceStore, ExperienceStoreConfig};
use janus_backward::verdict::{
    DEFAULT_FEE_BOUNDARY, DEFAULT_WINDOW_FROM, DEFAULT_WINDOW_TO, PopulationFilter, compute_report,
    decision_from_point, render_human,
};

/// Parsed CLI options (hand-rolled: the flag surface is tiny and the service
/// crate deliberately carries no CLI-framework dependency).
struct Opts {
    from: String,
    to: String,
    fee_boundary_ms: Option<i64>,
    qdrant_url: Option<String>,
    collection: Option<String>,
    page_size: usize,
    max_points: usize,
}

fn usage() -> &'static str {
    "gate_a_verdict — Gate-A janus-side verdict (prereg §4a)\n\
     \n\
     OPTIONS:\n\
       --from <RFC3339>          window start (default 2026-07-18T00:00:00Z)\n\
       --to <RFC3339>            window end, inclusive (default 2026-08-01T23:59:59Z)\n\
       --fee-boundary-ms <i64>   fee-aware cohort boundary in epoch ms\n\
                                 (default 2026-07-10T00:00:00Z)\n\
       --qdrant-url <URL>        override QDRANT_URL\n\
       --collection <NAME>       override QDRANT_EXPERIENCE_COLLECTION\n\
       --page-size <N>           scroll page size (default 1024)\n\
       --max-points <N>          hard cap on scrolled points (default 2000000)\n\
       --help                    print this help\n\
     \n\
     Human summary → stderr; machine-readable JSON → stdout."
}

fn parse_opts(args: &[String]) -> Result<Opts> {
    let mut opts = Opts {
        from: DEFAULT_WINDOW_FROM.to_string(),
        to: DEFAULT_WINDOW_TO.to_string(),
        fee_boundary_ms: None,
        qdrant_url: None,
        collection: None,
        page_size: 1024,
        max_points: 2_000_000,
    };
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut value = |name: &str| -> Result<String> {
            it.next()
                .cloned()
                .with_context(|| format!("{name} requires a value\n\n{}", usage()))
        };
        match arg.as_str() {
            "--from" => opts.from = value("--from")?,
            "--to" => opts.to = value("--to")?,
            "--fee-boundary-ms" => {
                opts.fee_boundary_ms = Some(
                    value("--fee-boundary-ms")?
                        .parse::<i64>()
                        .context("--fee-boundary-ms must be an integer (epoch ms)")?,
                )
            }
            "--qdrant-url" => opts.qdrant_url = Some(value("--qdrant-url")?),
            "--collection" => opts.collection = Some(value("--collection")?),
            "--page-size" => {
                opts.page_size = value("--page-size")?
                    .parse::<usize>()
                    .context("--page-size must be a positive integer")?
                    .max(1)
            }
            "--max-points" => {
                opts.max_points = value("--max-points")?
                    .parse::<usize>()
                    .context("--max-points must be a positive integer")?
                    .max(1)
            }
            "--help" | "-h" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => bail!("unknown argument: {other}\n\n{}", usage()),
        }
    }
    Ok(opts)
}

fn rfc3339_to_ms(s: &str, what: &str) -> Result<i64> {
    Ok(chrono::DateTime::parse_from_rfc3339(s)
        .with_context(|| format!("{what} is not valid RFC3339: {s}"))?
        .timestamp_millis())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(std::env::var("RUST_LOG").unwrap_or_else(|_| "warn".to_string()))
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let opts = parse_opts(&args)?;

    let from_ms = rfc3339_to_ms(&opts.from, "--from")?;
    let to_ms = rfc3339_to_ms(&opts.to, "--to")?;
    if from_ms >= to_ms {
        bail!("--from ({}) must precede --to ({})", opts.from, opts.to);
    }
    let fee_boundary_ms = match opts.fee_boundary_ms {
        Some(ms) => ms,
        None => rfc3339_to_ms(DEFAULT_FEE_BOUNDARY, "default fee boundary")?,
    };

    // Store config: same env resolution as the backward service, with
    // explicit CLI overrides.
    let mut config = ExperienceStoreConfig::from_env();
    if let Some(url) = opts.qdrant_url {
        config.qdrant_url = url;
    }
    if let Some(collection) = opts.collection {
        config.collection_name = collection;
    }
    // The qdrant-client version-compat check println!s warnings to STDOUT,
    // which would corrupt this binary's machine-readable JSON stream.
    config.skip_compatibility_check = true;
    let collection = config.collection_name.clone();

    // A verdict computed against a mock store would be fabricated data.
    if config.use_mock {
        bail!(
            "QDRANT_USE_MOCK is set — refusing to compute the Gate-A verdict \
             against a mock experience store"
        );
    }

    let store = ExperienceStore::new(config)
        .await
        .context("connecting to Qdrant (is QDRANT_URL set / reachable?)")?;

    let collection_points_total = store.collection_count().await.unwrap_or(0);

    eprintln!(
        "scrolling {collection} for [{} .. {}] (page {}, cap {})…",
        opts.from, opts.to, opts.page_size, opts.max_points
    );
    let points = store
        .scroll_window(from_ms, to_ms, opts.page_size, opts.max_points)
        .await
        .context("scrolling the experience window")?;
    let scrolled = points.len();
    if scrolled >= opts.max_points {
        eprintln!(
            "WARNING: hit --max-points ({}) — the window is TRUNCATED; re-run with a higher cap",
            opts.max_points
        );
    }

    let decisions: Vec<_> = points.iter().filter_map(decision_from_point).collect();

    let filter = PopulationFilter {
        from_ms,
        to_ms,
        fee_boundary_ms,
    };
    let report = compute_report(
        decisions,
        &filter,
        scrolled,
        &collection,
        collection_points_total,
    );

    eprintln!("{}", render_human(&report));
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
