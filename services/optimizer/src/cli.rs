//! CLI Module for JANUS Optimizer Runner
//!
//! Provides command-line interface for running optimizations in various modes:
//!
//! # Commands
//!
//! - `optimize` - Run optimization for specific assets
//! - `run` - Start the scheduler with a cron/interval
//! - `run-once` - Run a single optimization cycle and exit
//! - `status` - Check optimization and data status
//! - `collect` - Collect OHLC data without optimization
//!
//! # Usage Examples
//!
//! ```bash
//! # Optimize a single asset
//! janus-optimizer optimize --asset BTC
//!
//! # Optimize multiple assets
//! janus-optimizer optimize --assets BTC,ETH,SOL
//!
//! # Run optimization with quick mode (fewer trials)
//! janus-optimizer optimize --asset BTC --quick
//!
//! # Run once and exit
//! janus-optimizer run-once --assets BTC,ETH
//!
//! # Start scheduler with interval
//! janus-optimizer run --interval 6h
//!
//! # Dry run (don't publish to Redis)
//! janus-optimizer optimize --asset BTC --dry-run
//!
//! # Check status
//! janus-optimizer status
//!
//! # Collect data only
//! janus-optimizer collect --days 30
//! ```

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// JANUS Optimizer Runner - Automated strategy parameter optimization
#[derive(Parser, Debug)]
#[command(name = "janus-optimizer")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Redis URL for param publishing
    #[arg(long, env = "REDIS_URL", default_value = "redis://localhost:6379")]
    pub redis_url: String,

    /// Redis instance ID for key prefixes
    #[arg(long, env = "REDIS_INSTANCE_ID", default_value = "default")]
    pub instance_id: String,

    /// Data directory for OHLC storage
    #[arg(long, env = "DATA_DIR", default_value = "/data")]
    pub data_dir: PathBuf,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Output format
    #[arg(long, default_value = "text")]
    pub format: OutputFormat,

    /// Suppress all output except errors
    #[arg(short, long)]
    pub quiet: bool,

    #[command(subcommand)]
    pub command: Commands,
}

/// Available commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run optimization for specific assets
    Optimize(OptimizeArgs),

    /// Start the scheduler daemon
    Run(RunArgs),

    /// Run a single optimization cycle and exit
    RunOnce(RunOnceArgs),

    /// Check optimization and data status
    Status(StatusArgs),

    /// Collect OHLC data without optimization
    Collect(CollectArgs),

    /// List configured assets and their settings
    ListAssets(ListAssetsArgs),

    /// Show or clear optimization history
    History(HistoryArgs),
}

/// Arguments for the `optimize` command
#[derive(Args, Debug)]
pub struct OptimizeArgs {
    /// Single asset to optimize
    #[arg(short, long, conflicts_with = "assets")]
    pub asset: Option<String>,

    /// Comma-separated list of assets to optimize
    #[arg(long, value_delimiter = ',')]
    pub assets: Option<Vec<String>>,

    /// Number of optimization trials
    #[arg(short = 't', long, env = "OPTUNA_TRIALS")]
    pub trials: Option<usize>,

    /// Quick mode - reduced trials for faster results
    #[arg(short, long)]
    pub quick: bool,

    /// Dry run - don't publish results to Redis
    #[arg(long)]
    pub dry_run: bool,

    /// Save results to JSON file
    #[arg(long)]
    pub save: bool,

    /// Output file for results (default: stdout)
    #[arg(short, long)]
    pub output: Option<PathBuf>,

    /// OHLC interval in minutes for backtesting
    #[arg(long, default_value = "60")]
    pub interval: u32,

    /// Minimum days of data required
    #[arg(long, default_value = "7")]
    pub min_days: u32,

    /// Random seed for reproducibility
    #[arg(long)]
    pub seed: Option<u64>,

    /// Number of parallel jobs (0 = auto)
    #[arg(short = 'j', long, default_value = "1")]
    pub jobs: usize,

    /// Force optimization even with insufficient data
    #[arg(long)]
    pub force: bool,
}

/// Arguments for the `run` command (scheduler daemon)
#[derive(Args, Debug)]
pub struct RunArgs {
    /// Optimization interval (e.g., "6h", "30m", "1d")
    #[arg(short, long, env = "OPTIMIZATION_INTERVAL", default_value = "6h")]
    pub interval: String,

    /// Assets to optimize (comma-separated)
    #[arg(long, env = "OPTIMIZE_ASSETS", value_delimiter = ',')]
    pub assets: Option<Vec<String>>,

    /// Number of trials per optimization
    #[arg(short = 't', long, env = "OPTUNA_TRIALS")]
    pub trials: Option<usize>,

    /// Run initial optimization on start
    #[arg(long, env = "RUN_ON_START")]
    pub run_on_start: bool,

    /// Enable data collection loop
    #[arg(long, env = "DATA_COLLECTION_ENABLED")]
    pub collect_data: bool,

    /// Data collection interval in minutes
    #[arg(long, default_value = "5")]
    pub collect_interval: u64,

    /// Prometheus metrics port
    #[arg(long, env = "METRICS_PORT", default_value = "9092")]
    pub metrics_port: u16,

    /// Dry run - don't publish results to Redis
    #[arg(long)]
    pub dry_run: bool,

    /// Maximum consecutive failures before backing off
    #[arg(long, default_value = "3")]
    pub max_failures: u32,
}

/// Arguments for the `run-once` command
#[derive(Args, Debug)]
pub struct RunOnceArgs {
    /// Assets to optimize (comma-separated)
    #[arg(long, env = "OPTIMIZE_ASSETS", value_delimiter = ',')]
    pub assets: Option<Vec<String>>,

    /// Number of trials per optimization
    #[arg(short = 't', long)]
    pub trials: Option<usize>,

    /// Quick mode - reduced trials
    #[arg(short, long)]
    pub quick: bool,

    /// Dry run - don't publish results to Redis
    #[arg(long)]
    pub dry_run: bool,

    /// Update OHLC data before optimization
    #[arg(long)]
    pub update_data: bool,

    /// Fail if any asset optimization fails
    #[arg(long)]
    pub fail_fast: bool,

    /// Exit with error code if any optimization fails
    #[arg(long)]
    pub strict: bool,
}

/// Arguments for the `status` command
#[derive(Args, Debug)]
pub struct StatusArgs {
    /// Show detailed status
    #[arg(short, long)]
    pub detailed: bool,

    /// Check specific asset
    #[arg(long)]
    pub asset: Option<String>,

    /// Check Redis connection
    #[arg(long)]
    pub check_redis: bool,

    /// Check data availability
    #[arg(long)]
    pub check_data: bool,

    /// Show all checks
    #[arg(short = 'A', long)]
    pub all: bool,
}

/// Arguments for the `collect` command
#[derive(Args, Debug)]
pub struct CollectArgs {
    /// Days of historical data to collect
    #[arg(short, long, default_value = "30")]
    pub days: u32,

    /// Assets to collect data for (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub assets: Option<Vec<String>>,

    /// OHLC intervals to collect (comma-separated, in minutes)
    #[arg(long, value_delimiter = ',', default_value = "1,5,15,60,240,1440")]
    pub intervals: Vec<u32>,

    /// Force re-download even if data exists
    #[arg(long)]
    pub force: bool,

    /// Show progress
    #[arg(short, long)]
    pub progress: bool,
}

/// Arguments for the `list-assets` command
#[derive(Args, Debug)]
pub struct ListAssetsArgs {
    /// Show detailed asset configurations
    #[arg(short, long)]
    pub detailed: bool,

    /// Filter by category
    #[arg(long)]
    pub category: Option<String>,

    /// Show only enabled assets
    #[arg(long)]
    pub enabled_only: bool,
}

/// Arguments for the `history` command
#[derive(Args, Debug)]
pub struct HistoryArgs {
    /// Number of recent entries to show
    #[arg(short = 'n', long, default_value = "10")]
    pub count: usize,

    /// Filter by asset
    #[arg(long)]
    pub asset: Option<String>,

    /// Clear history
    #[arg(long)]
    pub clear: bool,

    /// Show only failures
    #[arg(long)]
    pub failures_only: bool,
}

/// Output format options
#[derive(ValueEnum, Clone, Debug, Default)]
pub enum OutputFormat {
    /// Human-readable text
    #[default]
    Text,
    /// JSON output
    Json,
    /// Compact JSON (single line)
    JsonCompact,
    /// Table format
    Table,
}

impl Cli {
    /// Parse command line arguments
    #[allow(dead_code)]
    pub fn parse_args() -> Self {
        Self::parse()
    }

    /// Get the log level based on verbosity
    pub fn log_level(&self) -> &'static str {
        if self.quiet {
            return "error";
        }
        match self.verbose {
            0 => "info",
            1 => "debug",
            _ => "trace",
        }
    }

    /// Check if output should be JSON
    #[allow(dead_code)]
    pub fn is_json_output(&self) -> bool {
        matches!(self.format, OutputFormat::Json | OutputFormat::JsonCompact)
    }
}

impl OptimizeArgs {
    /// Get list of assets to optimize
    pub fn get_assets(&self) -> Vec<String> {
        if let Some(asset) = &self.asset {
            vec![asset.to_uppercase()]
        } else if let Some(assets) = &self.assets {
            assets.iter().map(|s| s.to_uppercase()).collect()
        } else {
            // Default assets if none specified
            vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()]
        }
    }

    /// Get number of trials (respecting quick mode)
    pub fn get_trials(&self) -> usize {
        if self.quick {
            return 20; // Quick mode uses fewer trials
        }
        self.trials.unwrap_or(100)
    }
}

impl RunOnceArgs {
    /// Get list of assets to optimize
    pub fn get_assets(&self) -> Vec<String> {
        self.assets
            .as_ref()
            .map(|a| a.iter().map(|s| s.to_uppercase()).collect())
            .unwrap_or_else(|| vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()])
    }

    /// Get number of trials (respecting quick mode)
    pub fn get_trials(&self) -> usize {
        if self.quick {
            return 20;
        }
        self.trials.unwrap_or(100)
    }
}

impl RunArgs {
    /// Get list of assets to optimize
    pub fn get_assets(&self) -> Vec<String> {
        self.assets
            .as_ref()
            .map(|a| a.iter().map(|s| s.to_uppercase()).collect())
            .unwrap_or_else(|| vec!["BTC".to_string(), "ETH".to_string(), "SOL".to_string()])
    }

    /// Get number of trials
    pub fn get_trials(&self) -> usize {
        self.trials.unwrap_or(100)
    }
}

impl StatusArgs {
    /// Should check all status items
    pub fn check_all(&self) -> bool {
        self.all || (!self.check_redis && !self.check_data && self.asset.is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_optimize_single_asset() {
        let cli = Cli::parse_from(["janus-optimizer", "optimize", "--asset", "BTC"]);

        match cli.command {
            Commands::Optimize(args) => {
                assert_eq!(args.get_assets(), vec!["BTC"]);
            }
            _ => panic!("Expected Optimize command"),
        }
    }

    #[test]
    fn test_parse_optimize_multiple_assets() {
        let cli = Cli::parse_from(["janus-optimizer", "optimize", "--assets", "BTC,ETH,SOL"]);

        match cli.command {
            Commands::Optimize(args) => {
                assert_eq!(args.get_assets(), vec!["BTC", "ETH", "SOL"]);
            }
            _ => panic!("Expected Optimize command"),
        }
    }

    #[test]
    fn test_parse_optimize_quick_mode() {
        let cli = Cli::parse_from(["janus-optimizer", "optimize", "--asset", "BTC", "--quick"]);

        match cli.command {
            Commands::Optimize(args) => {
                assert!(args.quick);
                assert_eq!(args.get_trials(), 20);
            }
            _ => panic!("Expected Optimize command"),
        }
    }

    #[test]
    fn test_parse_optimize_with_trials() {
        let cli = Cli::parse_from([
            "janus-optimizer",
            "optimize",
            "--asset",
            "BTC",
            "--trials",
            "50",
        ]);

        match cli.command {
            Commands::Optimize(args) => {
                assert_eq!(args.get_trials(), 50);
            }
            _ => panic!("Expected Optimize command"),
        }
    }

    #[test]
    fn test_parse_run_with_interval() {
        let cli = Cli::parse_from(["janus-optimizer", "run", "--interval", "2h"]);

        match cli.command {
            Commands::Run(args) => {
                assert_eq!(args.interval, "2h");
            }
            _ => panic!("Expected Run command"),
        }
    }

    #[test]
    fn test_parse_run_once() {
        let cli = Cli::parse_from([
            "janus-optimizer",
            "run-once",
            "--assets",
            "BTC,ETH",
            "--quick",
        ]);

        match cli.command {
            Commands::RunOnce(args) => {
                assert_eq!(args.get_assets(), vec!["BTC", "ETH"]);
                assert!(args.quick);
            }
            _ => panic!("Expected RunOnce command"),
        }
    }

    #[test]
    fn test_parse_status() {
        let cli = Cli::parse_from(["janus-optimizer", "status", "-A"]);

        match cli.command {
            Commands::Status(args) => {
                assert!(args.check_all());
            }
            _ => panic!("Expected Status command"),
        }
    }

    #[test]
    fn test_parse_collect() {
        let cli = Cli::parse_from(["janus-optimizer", "collect", "--days", "60", "--force"]);

        match cli.command {
            Commands::Collect(args) => {
                assert_eq!(args.days, 60);
                assert!(args.force);
            }
            _ => panic!("Expected Collect command"),
        }
    }

    #[test]
    fn test_global_options() {
        let cli = Cli::parse_from([
            "janus-optimizer",
            "--redis-url",
            "redis://custom:6379",
            "--instance-id",
            "test",
            "-vv",
            "status",
        ]);

        assert_eq!(cli.redis_url, "redis://custom:6379");
        assert_eq!(cli.instance_id, "test");
        assert_eq!(cli.verbose, 2);
        assert_eq!(cli.log_level(), "trace");
    }

    #[test]
    fn test_quiet_mode() {
        let cli = Cli::parse_from(["janus-optimizer", "-q", "status"]);

        assert!(cli.quiet);
        assert_eq!(cli.log_level(), "error");
    }

    #[test]
    fn test_json_output() {
        let cli = Cli::parse_from(["janus-optimizer", "--format", "json", "status"]);

        assert!(cli.is_json_output());
    }

    #[test]
    fn test_default_assets() {
        let args = OptimizeArgs {
            asset: None,
            assets: None,
            trials: None,
            quick: false,
            dry_run: false,
            save: false,
            output: None,
            interval: 60,
            min_days: 7,
            seed: None,
            jobs: 1,
            force: false,
        };

        assert_eq!(args.get_assets(), vec!["BTC", "ETH", "SOL"]);
    }

    #[test]
    fn test_dry_run_option() {
        let cli = Cli::parse_from(["janus-optimizer", "optimize", "--asset", "BTC", "--dry-run"]);

        match cli.command {
            Commands::Optimize(args) => {
                assert!(args.dry_run);
            }
            _ => panic!("Expected Optimize command"),
        }
    }
}
