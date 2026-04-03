//! # JANUS Backtest CLI
//!
//! Command-line tool for running JANUS strategy backtests on historical OHLCV data.
//!
//! ## Usage
//!
//! ```bash
//! # Basic usage — auto-detect columns and timestamp format
//! janus-backtest --data kraken_btcusd_15m.csv --symbol BTCUSD
//!
//! # Kraken preset
//! janus-backtest --data kraken_btcusd_15m.csv --symbol XBTUSD --preset kraken
//!
//! # Binance preset with time range
//! janus-backtest --data binance_btcusdt_1h.csv --symbol BTCUSDT --preset binance \
//!     --start 2024-01-01T00:00:00Z --end 2024-06-30T23:59:59Z
//!
//! # Custom config
//! janus-backtest --data data.csv --symbol BTCUSD \
//!     --htf-ratio 15 --slippage 5 --commission 6 \
//!     --balance 25000 --max-positions 4
//!
//! # Inspect a data file without running backtest
//! janus-backtest --data data.csv --inspect
//!
//! # Output JSON report
//! janus-backtest --data data.csv --symbol BTCUSD --json
//!
//! # Export trades to CSV
//! janus-backtest --data data.csv --symbol BTCUSD --export-trades trades.csv
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use clap::Parser;
use janus_backtest::{
    ohlcv_loader::{OhlcvColumnMap, OhlcvLoader, OhlcvLoaderConfig, TimestampFormat},
    strategy_backtester::{
        BacktestReport, CompletedTrade, OhlcvBar, StrategyBacktester, StrategyBacktesterConfig,
        StrategyId,
    },
};
use serde::Serialize;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use tracing::{error, info, warn};

// ============================================================================
// CLI Arguments
// ============================================================================

/// JANUS Strategy Backtest CLI
///
/// Run JANUS trading strategies against historical OHLCV data and produce
/// per-strategy performance reports.
#[derive(Parser, Debug)]
#[command(
    name = "janus-backtest",
    version,
    about = "JANUS Strategy Backtest CLI — run strategies against historical OHLCV data",
    long_about = "Loads OHLCV bar data from CSV or Parquet files, runs all JANUS strategies\n\
                  through the regime-aware pipeline, and produces per-strategy performance reports.\n\n\
                  Supports Kraken, Binance, TradingView CSV formats and Parquet."
)]
struct Cli {
    /// Path to the OHLCV data file (CSV, TSV, or Parquet).
    #[arg(short, long, value_name = "FILE")]
    data: PathBuf,

    /// Trading symbol (e.g., BTCUSD, XBTUSD, ETHUSDT).
    #[arg(short, long, default_value = "BTCUSD")]
    symbol: String,

    /// Exchange data format preset.
    ///
    /// Presets configure column names and timestamp format automatically.
    /// Available: kraken, binance, tradingview, positional, auto (default).
    #[arg(short, long, default_value = "auto")]
    preset: String,

    /// Timestamp format override.
    ///
    /// Options: auto, unix-seconds, unix-millis, unix-micros, unix-nanos, iso8601.
    /// When a preset is used, this overrides the preset's timestamp format.
    #[arg(long, value_name = "FORMAT")]
    timestamp: Option<String>,

    /// Initial account balance in USDT.
    #[arg(long, default_value = "10000")]
    balance: f64,

    /// Slippage in basis points (applied to entries and exits).
    #[arg(long, default_value = "5")]
    slippage: f64,

    /// Commission per trade in basis points.
    #[arg(long, default_value = "6")]
    commission: f64,

    /// HTF aggregator ratio (LTF bars per HTF bar).
    #[arg(long, default_value = "15")]
    htf_ratio: usize,

    /// HTF short EMA period.
    #[arg(long, default_value = "20")]
    htf_ema_short: usize,

    /// HTF long EMA period.
    #[arg(long, default_value = "50")]
    htf_ema_long: usize,

    /// Maximum number of concurrent positions.
    #[arg(long, default_value = "3")]
    max_positions: usize,

    /// Session length in bars (for ORB/VWAP session resets).
    #[arg(long, default_value = "96")]
    session_bars: usize,

    /// Disable concurrent positions (only one position at a time).
    #[arg(long)]
    no_concurrent: bool,

    /// Start time filter (inclusive). ISO 8601 format.
    #[arg(long, value_name = "DATETIME")]
    start: Option<String>,

    /// End time filter (inclusive). ISO 8601 format.
    #[arg(long, value_name = "DATETIME")]
    end: Option<String>,

    /// Skip this many leading bars before running the backtest.
    #[arg(long, default_value = "0")]
    skip_bars: usize,

    /// Maximum number of bars to process.
    #[arg(long, value_name = "N")]
    max_bars: Option<usize>,

    /// CSV delimiter character (default: comma).
    #[arg(long, default_value = ",")]
    delimiter: String,

    /// CSV has no header row (use positional column mapping).
    #[arg(long)]
    no_header: bool,

    /// Custom column names: timestamp,open,high,low,close,volume.
    ///
    /// Example: --columns "time,o,h,l,c,vol"
    #[arg(long, value_name = "NAMES")]
    columns: Option<String>,

    /// Inspect the data file and exit (don't run backtest).
    #[arg(long)]
    inspect: bool,

    /// Output report as JSON instead of formatted text.
    #[arg(long)]
    json: bool,

    /// Export all trades to a CSV file.
    #[arg(long, value_name = "FILE")]
    export_trades: Option<PathBuf>,

    /// Export all signals to a CSV file.
    #[arg(long, value_name = "FILE")]
    export_signals: Option<PathBuf>,

    /// Filter report to specific strategies (comma-separated).
    ///
    /// Available: mean-reversion, squeeze-breakout, vwap, orb, ema-ribbon,
    /// trend-pullback, momentum-surge, multi-tf
    #[arg(long, value_name = "STRATEGIES")]
    strategies: Option<String>,

    /// Increase logging verbosity (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Suppress all output except the final report.
    #[arg(short, long)]
    quiet: bool,
}

// ============================================================================
// JSON Report Types
// ============================================================================

#[derive(Serialize)]
struct JsonReport {
    symbol: String,
    data_file: String,
    total_bars: usize,
    total_signals: usize,
    total_trades: usize,
    aggregate_pnl_pct: f64,
    regime_distribution: HashMap<String, usize>,
    strategies: Vec<JsonStrategyMetrics>,
}

#[derive(Serialize)]
struct JsonStrategyMetrics {
    name: String,
    total_trades: usize,
    winning_trades: usize,
    losing_trades: usize,
    win_rate: f64,
    total_pnl_pct: f64,
    avg_win_pct: f64,
    avg_loss_pct: f64,
    largest_win_pct: f64,
    largest_loss_pct: f64,
    profit_factor: f64,
    max_drawdown_pct: f64,
    sharpe_ratio: f64,
    signals_generated: usize,
    signals_in_target_regime: usize,
    signals_outside_regime: usize,
    avg_trade_duration_bars: f64,
}

#[derive(Serialize)]
#[allow(dead_code)]
struct JsonTrade {
    strategy: String,
    direction: String,
    entry_price: f64,
    exit_price: f64,
    entry_bar: usize,
    exit_bar: usize,
    pnl_pct: f64,
    pnl_absolute: f64,
    size_factor: f64,
    exit_reason: String,
}

#[derive(Serialize)]
#[allow(dead_code)]
struct JsonSignal {
    strategy: String,
    direction: String,
    confidence: f64,
    price: f64,
    bar_index: usize,
    regime: String,
    reason: String,
}

// ============================================================================
// Main
// ============================================================================

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    init_logging(cli.verbose, cli.quiet);

    // Inspect mode — just show file info and exit
    if cli.inspect {
        return run_inspect(&cli);
    }

    // Load OHLCV data
    let bars = load_data(&cli)?;

    if bars.is_empty() {
        error!("No bars loaded. Check your data file and filters.");
        std::process::exit(1);
    }

    info!(
        "Loaded {} bars for {} | {} → {}",
        bars.len(),
        cli.symbol,
        bars.first().unwrap().timestamp.format("%Y-%m-%d %H:%M"),
        bars.last().unwrap().timestamp.format("%Y-%m-%d %H:%M"),
    );

    // Configure and run the backtester
    let config = build_backtester_config(&cli);
    let mut backtester = StrategyBacktester::new(config);

    info!("Running backtest...");
    let report = backtester.run(&bars);

    // Filter strategies if requested
    let strategy_filter = parse_strategy_filter(&cli);

    // Output the report
    if cli.json {
        output_json_report(&cli, &report, &strategy_filter)?;
    } else {
        output_text_report(&report, &strategy_filter);
    }

    // Export trades to CSV
    if let Some(ref path) = cli.export_trades {
        export_trades_csv(path, backtester.trades(), &strategy_filter)?;
    }

    // Export signals to CSV
    if let Some(ref path) = cli.export_signals {
        export_signals_csv(path, backtester.signals(), &strategy_filter)?;
    }

    Ok(())
}

// ============================================================================
// Initialization
// ============================================================================

fn init_logging(verbose: u8, quiet: bool) {
    if quiet {
        // Only errors
        let filter = "error";
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| filter.into()),
            )
            .with_target(false)
            .init();
    } else {
        let filter = match verbose {
            0 => "warn,janus_backtest=info",
            1 => "info,janus_backtest=debug",
            2 => "debug",
            _ => "trace",
        };
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| filter.into()),
            )
            .with_target(verbose >= 2)
            .init();
    }
}

// ============================================================================
// Data Loading
// ============================================================================

fn load_data(cli: &Cli) -> Result<Vec<OhlcvBar>> {
    let mut config = OhlcvLoaderConfig::new(&cli.data).with_symbol(&cli.symbol);

    // Apply preset
    config = match cli.preset.to_lowercase().as_str() {
        "kraken" => config.kraken(),
        "binance" => config.binance(),
        "tradingview" | "tv" => config.tradingview(),
        "positional" | "pos" => config.positional(),
        _ => config,
    };

    // Override timestamp format if specified
    if let Some(ref ts_fmt) = cli.timestamp {
        let format = parse_timestamp_format(ts_fmt)?;
        config = config.with_timestamp_format(format);
    }

    // No header
    if cli.no_header {
        config = config.with_header(false);
    }

    // Custom columns
    if let Some(ref cols) = cli.columns {
        let col_map = parse_column_names(cols)?;
        config = config.with_column_map(col_map);
    }

    // Delimiter
    if cli.delimiter.len() == 1 {
        config = config.with_delimiter(cli.delimiter.as_bytes()[0]);
    } else if cli.delimiter.to_lowercase() == "tab" || cli.delimiter == "\\t" {
        config = config.with_delimiter(b'\t');
    }

    // Time range
    if cli.start.is_some() || cli.end.is_some() {
        let start = if let Some(ref s) = cli.start {
            parse_datetime(s).context("Failed to parse --start datetime")?
        } else {
            DateTime::<Utc>::MIN_UTC
        };
        let end = if let Some(ref e) = cli.end {
            parse_datetime(e).context("Failed to parse --end datetime")?
        } else {
            DateTime::<Utc>::MAX_UTC
        };
        config = config.with_time_range(start, end);
    }

    // Skip / limit
    if cli.skip_bars > 0 {
        config = config.with_skip_bars(cli.skip_bars);
    }
    if let Some(max) = cli.max_bars {
        config = config.with_max_bars(max);
    }

    let loader = OhlcvLoader::new(config);
    let bars = loader.load().context("Failed to load OHLCV data")?;

    Ok(bars)
}

fn parse_timestamp_format(s: &str) -> Result<TimestampFormat> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(TimestampFormat::Auto),
        "unix-seconds" | "seconds" | "unix_seconds" | "s" => Ok(TimestampFormat::UnixSeconds),
        "unix-millis" | "millis" | "unix_millis" | "ms" => Ok(TimestampFormat::UnixMillis),
        "unix-micros" | "micros" | "unix_micros" | "us" => Ok(TimestampFormat::UnixMicros),
        "unix-nanos" | "nanos" | "unix_nanos" | "ns" => Ok(TimestampFormat::UnixNanos),
        "iso8601" | "iso" | "rfc3339" => Ok(TimestampFormat::Iso8601),
        other => {
            // Treat as a custom chrono format string
            Ok(TimestampFormat::Custom(other.to_string()))
        }
    }
}

fn parse_column_names(s: &str) -> Result<OhlcvColumnMap> {
    let parts: Vec<&str> = s.split(',').map(|p| p.trim()).collect();
    if parts.len() < 6 {
        anyhow::bail!(
            "Expected 6 comma-separated column names (timestamp,open,high,low,close,volume), got {}",
            parts.len()
        );
    }
    Ok(OhlcvColumnMap::custom(
        parts[0], parts[1], parts[2], parts[3], parts[4], parts[5],
    ))
}

fn parse_datetime(s: &str) -> Result<DateTime<Utc>> {
    // Try RFC 3339
    if let Ok(dt) = s.parse::<DateTime<Utc>>() {
        return Ok(dt);
    }

    // Try common formats
    let formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ];

    for fmt in &formats {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(ndt.and_utc());
        }
    }

    // Try date only
    if let Ok(nd) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        let ndt = nd.and_hms_opt(0, 0, 0).unwrap();
        return Ok(ndt.and_utc());
    }

    anyhow::bail!(
        "Cannot parse '{}' as a datetime. Use ISO 8601 format (e.g., 2024-01-15T12:00:00Z)",
        s
    )
}

// ============================================================================
// Backtester Config
// ============================================================================

fn build_backtester_config(cli: &Cli) -> StrategyBacktesterConfig {
    StrategyBacktesterConfig {
        symbol: cli.symbol.clone(),
        initial_balance: cli.balance,
        slippage_bps: cli.slippage,
        commission_bps: cli.commission,
        htf_ratio: cli.htf_ratio,
        htf_ema_short: cli.htf_ema_short,
        htf_ema_long: cli.htf_ema_long,
        allow_concurrent_positions: !cli.no_concurrent,
        max_concurrent_positions: cli.max_positions,
        session_bars: cli.session_bars,
    }
}

// ============================================================================
// Strategy Filtering
// ============================================================================

fn parse_strategy_filter(cli: &Cli) -> Option<Vec<StrategyId>> {
    let filter_str = cli.strategies.as_ref()?;
    let mut ids = Vec::new();

    for part in filter_str.split(',').map(|s| s.trim().to_lowercase()) {
        match part.as_str() {
            "mean-reversion" | "mr" | "meanreversion" => ids.push(StrategyId::MeanReversion),
            "squeeze-breakout" | "squeeze" | "squeezebreakout" | "sb" => {
                ids.push(StrategyId::SqueezeBreakout)
            }
            "vwap" | "vwap-scalper" | "vwapscalper" => ids.push(StrategyId::VwapScalper),
            "orb" | "opening-range" | "openingrangebreakout" => {
                ids.push(StrategyId::OpeningRangeBreakout)
            }
            "ema-ribbon" | "emaribbon" | "ribbon" | "er" => ids.push(StrategyId::EmaRibbonScalper),
            "trend-pullback" | "trendpullback" | "pullback" | "tp" => {
                ids.push(StrategyId::TrendPullback)
            }
            "momentum-surge" | "momentumsurge" | "momentum" | "ms" => {
                ids.push(StrategyId::MomentumSurge)
            }
            "multi-tf" | "multitf" | "multi-tf-trend" | "mtf" => ids.push(StrategyId::MultiTfTrend),
            other => {
                warn!("Unknown strategy filter '{}', ignoring", other);
            }
        }
    }

    if ids.is_empty() { None } else { Some(ids) }
}

fn should_include_strategy(id: &StrategyId, filter: &Option<Vec<StrategyId>>) -> bool {
    match filter {
        None => true,
        Some(allowed) => allowed.contains(id),
    }
}

// ============================================================================
// Output — Text Report
// ============================================================================

fn output_text_report(report: &BacktestReport, filter: &Option<Vec<StrategyId>>) {
    // If no filter, just print the full report
    if filter.is_none() {
        println!("{}", report);
        return;
    }

    // Filtered output
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!("  JANUS Strategy Backtest Report — {}", report.symbol);
    println!("═══════════════════════════════════════════════════════════════════════════");
    println!(
        "  Bars: {} | Total Signals: {} | Total Trades: {} | Aggregate PnL: {:+.2}%",
        report.total_bars, report.total_signals, report.total_trades, report.aggregate_pnl_pct
    );
    println!();

    println!("  Regime Distribution:");
    for (regime, count) in &report.regime_distribution {
        let pct = if report.total_bars > 0 {
            (*count as f64 / report.total_bars as f64) * 100.0
        } else {
            0.0
        };
        println!("    {:<20} {:>5} bars ({:.1}%)", regime, count, pct);
    }
    println!();

    println!("  Per-Strategy Results (filtered):");
    println!("  ─────────────────────────────────────────────────────────────────────────");

    let mut sorted: Vec<_> = report
        .strategy_metrics
        .iter()
        .filter(|(id, _)| should_include_strategy(id, filter))
        .map(|(_, m)| m)
        .collect();
    sorted.sort_by(|a, b| {
        b.total_pnl_pct
            .partial_cmp(&a.total_pnl_pct)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for m in sorted {
        println!("  {}", m);
    }
    println!("═══════════════════════════════════════════════════════════════════════════");
}

// ============================================================================
// Output — JSON Report
// ============================================================================

fn output_json_report(
    cli: &Cli,
    report: &BacktestReport,
    filter: &Option<Vec<StrategyId>>,
) -> Result<()> {
    let strategies: Vec<JsonStrategyMetrics> = report
        .strategy_metrics
        .iter()
        .filter(|(id, _)| should_include_strategy(id, filter))
        .map(|(_, m)| JsonStrategyMetrics {
            name: format!("{}", m.strategy),
            total_trades: m.total_trades,
            winning_trades: m.winning_trades,
            losing_trades: m.losing_trades,
            win_rate: m.win_rate,
            total_pnl_pct: m.total_pnl_pct,
            avg_win_pct: m.avg_win_pct,
            avg_loss_pct: m.avg_loss_pct,
            largest_win_pct: m.largest_win_pct,
            largest_loss_pct: m.largest_loss_pct,
            profit_factor: m.profit_factor,
            max_drawdown_pct: m.max_drawdown_pct,
            sharpe_ratio: m.sharpe_ratio,
            signals_generated: m.signals_generated,
            signals_in_target_regime: m.signals_in_target_regime,
            signals_outside_regime: m.signals_outside_regime,
            avg_trade_duration_bars: m.avg_trade_duration_bars,
        })
        .collect();

    let json_report = JsonReport {
        symbol: report.symbol.clone(),
        data_file: cli.data.display().to_string(),
        total_bars: report.total_bars,
        total_signals: report.total_signals,
        total_trades: report.total_trades,
        aggregate_pnl_pct: report.aggregate_pnl_pct,
        regime_distribution: report.regime_distribution.clone(),
        strategies,
    };

    let json = serde_json::to_string_pretty(&json_report)?;
    println!("{}", json);

    Ok(())
}

// ============================================================================
// Export — Trades CSV
// ============================================================================

fn export_trades_csv(
    path: &PathBuf,
    trades: &[CompletedTrade],
    filter: &Option<Vec<StrategyId>>,
) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .context(format!("Cannot create trades file: {}", path.display()))?;

    writeln!(
        file,
        "strategy,direction,entry_price,exit_price,entry_bar,exit_bar,pnl_pct,pnl_absolute,size_factor,exit_reason"
    )?;

    let mut count = 0usize;
    for t in trades {
        if !should_include_strategy(&t.strategy, filter) {
            continue;
        }
        writeln!(
            file,
            "{},{},{:.6},{:.6},{},{},{:.4},{:.4},{:.4},{}",
            t.strategy,
            t.direction,
            t.entry_price,
            t.exit_price,
            t.entry_bar,
            t.exit_bar,
            t.pnl_pct,
            t.pnl_absolute,
            t.size_factor,
            t.exit_reason,
        )?;
        count += 1;
    }

    info!("Exported {} trades to {}", count, path.display());
    Ok(())
}

// ============================================================================
// Export — Signals CSV
// ============================================================================

fn export_signals_csv(
    path: &PathBuf,
    signals: &[janus_backtest::BacktestSignal],
    filter: &Option<Vec<StrategyId>>,
) -> Result<()> {
    let mut file = std::fs::File::create(path)
        .context(format!("Cannot create signals file: {}", path.display()))?;

    writeln!(
        file,
        "strategy,direction,confidence,price,bar_index,regime,reason"
    )?;

    let mut count = 0usize;
    for s in signals {
        if !should_include_strategy(&s.strategy, filter) {
            continue;
        }
        // Escape commas in reason by quoting
        let reason_escaped = if s.reason.contains(',') || s.reason.contains('"') {
            format!("\"{}\"", s.reason.replace('"', "\"\""))
        } else {
            s.reason.clone()
        };
        writeln!(
            file,
            "{},{},{:.4},{:.6},{},{},{}",
            s.strategy, s.direction, s.confidence, s.price, s.bar_index, s.regime, reason_escaped,
        )?;
        count += 1;
    }

    info!("Exported {} signals to {}", count, path.display());
    Ok(())
}

// ============================================================================
// Inspect Mode
// ============================================================================

fn run_inspect(cli: &Cli) -> Result<()> {
    let mut config = OhlcvLoaderConfig::new(&cli.data);

    // Apply preset for column detection
    config = match cli.preset.to_lowercase().as_str() {
        "kraken" => config.kraken(),
        "binance" => config.binance(),
        "tradingview" | "tv" => config.tradingview(),
        "positional" | "pos" => config.positional(),
        _ => config,
    };

    if cli.no_header {
        config = config.with_header(false);
    }
    if let Some(ref cols) = cli.columns {
        let col_map = parse_column_names(cols)?;
        config = config.with_column_map(col_map);
    }

    let loader = OhlcvLoader::new(config);
    let summary = loader.inspect().context("Failed to inspect data file")?;

    if cli.json {
        let json = serde_json::json!({
            "path": summary.path,
            "format": summary.format,
            "row_count": summary.row_count,
            "columns": summary.columns,
            "mapped_columns": {
                "timestamp": summary.detected_column_map.timestamp,
                "open": summary.detected_column_map.open,
                "high": summary.detected_column_map.high,
                "low": summary.detected_column_map.low,
                "close": summary.detected_column_map.close,
                "volume": summary.detected_column_map.volume,
            },
            "min_price": summary.min_price,
            "max_price": summary.max_price,
        });
        println!("{}", serde_json::to_string_pretty(&json)?);
    } else {
        println!("{}", summary);

        // Also try to load a few bars to show sample data
        let mut sample_config = OhlcvLoaderConfig::new(&cli.data);
        sample_config = match cli.preset.to_lowercase().as_str() {
            "kraken" => sample_config.kraken(),
            "binance" => sample_config.binance(),
            "tradingview" | "tv" => sample_config.tradingview(),
            "positional" | "pos" => sample_config.positional(),
            _ => sample_config,
        };
        sample_config = sample_config.with_max_bars(5);

        if let Some(ref ts_fmt) = cli.timestamp
            && let Ok(format) = parse_timestamp_format(ts_fmt)
        {
            sample_config = sample_config.with_timestamp_format(format);
        }

        let sample_loader = OhlcvLoader::new(sample_config);
        match sample_loader.load() {
            Ok(bars) => {
                println!("\n  Sample bars (first {}):", bars.len());
                println!(
                    "  {:>4}  {:>20}  {:>12}  {:>12}  {:>12}  {:>12}  {:>12}",
                    "#", "Timestamp", "Open", "High", "Low", "Close", "Volume"
                );
                println!("  {}", "─".repeat(92));
                for (i, bar) in bars.iter().enumerate() {
                    println!(
                        "  {:>4}  {:>20}  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}  {:>12.2}",
                        i + 1,
                        bar.timestamp.format("%Y-%m-%d %H:%M:%S"),
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                    );
                }
            }
            Err(e) => {
                println!("\n  Could not parse sample bars: {}", e);
                println!("  Try specifying --timestamp and/or --columns flags.");
            }
        }
    }

    Ok(())
}
