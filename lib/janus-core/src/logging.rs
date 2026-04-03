//! High-performance multi-layer logging for the Janus trading system.
//!
//! ## Runtime Log Level Changes
//!
//! After calling [`init_logging`], use [`LoggingGuard::create_controller`] to
//! obtain a [`Box<dyn LogLevelController>`](crate::state::LogLevelController)
//! that can be installed into [`JanusState`](crate::JanusState) for the API
//! module to expose as `POST /api/log-level`.
//!
//! This module implements the **Layered Registry Model** described in the
//! Janus Supervisor Architecture Refactor document. It splits logging into
//! two independent pipelines:
//!
//! - **Layer 1 — Operational Telemetry**: Human-readable logs to stdout,
//!   filtered by `RUST_LOG` / [`EnvFilter`], with runtime-reloadable log
//!   levels via a [`ReloadHandle`].
//!
//! - **Layer 2 — HFT Data Stream**: High-frequency market data events
//!   written to a non-blocking rolling file appender. Events are buffered
//!   in a ring buffer and flushed by a dedicated worker thread, fully
//!   decoupling the latency-sensitive trading path from disk I/O.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────┐
//! │                 tracing Registry                   │
//! │                                                    │
//! │  ┌──────────────────────┐  ┌────────────────────┐ │
//! │  │  Layer 1: Ops/Stdout │  │ Layer 2: HFT File  │ │
//! │  │  EnvFilter (reload)  │  │ Targets("janus::   │ │
//! │  │  fmt::layer().pretty │  │   hft" = TRACE)    │ │
//! │  │                      │  │ non_blocking writer │ │
//! │  └──────────────────────┘  └────────────────────┘ │
//! └───────────────────────────────────────────────────┘
//! ```
//!
//! # Critical: WorkerGuard Lifetime
//!
//! The [`LoggingGuard`] returned by [`init_logging`] **must** be held alive
//! in `main()` until the very end of the program. Dropping it prematurely
//! causes the non-blocking HFT writer's buffer to be discarded, losing
//! market data audit trail entries during shutdown.
//!
//! # Usage
//!
//! ```rust,ignore
//! use janus_core::logging::{init_logging, LoggingConfig};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = LoggingConfig::default();
//!     let guard = init_logging(config)?;
//!
//!     // ... application code ...
//!
//!     // `guard` drops here — HFT buffer is flushed
//!     Ok(())
//! }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use tracing::Level;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    Layer, Registry,
    filter::{EnvFilter, Targets},
    fmt,
    layer::SubscriberExt,
    reload,
    util::SubscriberInitExt,
};

use crate::state::LogLevelController;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the multi-layer logging system.
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Default `RUST_LOG` filter string if `RUST_LOG` env var is not set.
    /// Example: `"info,janus=debug,janus::hft=off"`
    pub default_env_filter: String,

    /// Whether to use the `.pretty()` formatter for stdout (multi-line,
    /// coloured). Set to `false` for JSON output in production / CI.
    pub pretty_stdout: bool,

    /// Whether to enable the HFT file logging layer.
    pub enable_hft_layer: bool,

    /// Directory for HFT rolling log files.
    /// Default: `"./logs/hft"`
    pub hft_log_dir: PathBuf,

    /// Filename prefix for HFT rolling logs.
    /// Default: `"hft.log"`
    pub hft_log_prefix: String,

    /// The tracing target that the HFT layer listens to.
    /// Default: `"janus::hft"`
    ///
    /// Events must be emitted with this target to appear in the HFT log:
    /// ```rust,ignore
    /// tracing::trace!(target: "janus::hft", symbol = "BTCUSD", price = 42000.0);
    /// ```
    pub hft_target: String,

    /// Maximum tracing level recorded by the HFT layer.
    /// Default: [`Level::TRACE`] (capture everything targeted at `janus::hft`).
    pub hft_level: Level,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            default_env_filter: "info,janus=debug".to_string(),
            pretty_stdout: true,
            enable_hft_layer: true,
            hft_log_dir: PathBuf::from("./logs/hft"),
            hft_log_prefix: "hft.log".to_string(),
            hft_target: "janus::hft".to_string(),
            hft_level: Level::TRACE,
        }
    }
}

impl LoggingConfig {
    /// Create a minimal config for tests — stdout only, no HFT file layer.
    pub fn for_tests() -> Self {
        Self {
            default_env_filter: "warn".to_string(),
            pretty_stdout: false,
            enable_hft_layer: false,
            ..Default::default()
        }
    }

    /// Builder: set the default env filter.
    pub fn with_env_filter(mut self, filter: impl Into<String>) -> Self {
        self.default_env_filter = filter.into();
        self
    }

    /// Builder: set the HFT log directory.
    pub fn with_hft_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.hft_log_dir = dir.into();
        self
    }

    /// Builder: disable HFT file logging.
    pub fn without_hft(mut self) -> Self {
        self.enable_hft_layer = false;
        self
    }

    /// Builder: use JSON output instead of pretty printing.
    pub fn with_json_stdout(mut self) -> Self {
        self.pretty_stdout = false;
        self
    }
}

// ---------------------------------------------------------------------------
// LoggingGuard — must live until shutdown
// ---------------------------------------------------------------------------

/// Holds resources that must outlive the logging system.
///
/// **Critical**: This guard owns the [`WorkerGuard`] for the non-blocking
/// HFT file appender. If dropped, the background writer thread stops and
/// any buffered log entries are lost.
///
/// Always bind this to a named variable in `main()`:
/// ```rust,ignore
/// let _guard = init_logging(config)?;
/// ```
pub struct LoggingGuard {
    /// The non-blocking writer's guard. Dropping this flushes and stops
    /// the background writer thread.
    _hft_guard: Option<WorkerGuard>,

    /// Handle to dynamically reload the stdout layer's filter at runtime.
    /// Exposed so the API server or operator tooling can change log levels
    /// without restarting the process.
    ops_reload_handle: Option<reload::Handle<EnvFilter, Registry>>,
}

impl LoggingGuard {
    /// Get a reference to the reload handle for the operational (stdout)
    /// layer's [`EnvFilter`].
    ///
    /// Returns `None` if the stdout layer was not installed with a reload
    /// handle (shouldn't happen in normal operation).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// if let Some(handle) = guard.ops_reload_handle() {
    ///     let new_filter = EnvFilter::new("debug,hyper=info");
    ///     handle.reload(new_filter).expect("reload failed");
    ///     tracing::info!("Log level changed to debug at runtime");
    /// }
    /// ```
    pub fn ops_reload_handle(&self) -> Option<&reload::Handle<EnvFilter, Registry>> {
        self.ops_reload_handle.as_ref()
    }

    /// Reload the operational log filter at runtime.
    ///
    /// Accepts any string that [`EnvFilter`] can parse, e.g.:
    /// - `"debug"`
    /// - `"info,janus=trace"`
    /// - `"warn,janus::supervisor=debug"`
    ///
    /// Returns an error if the filter string is invalid or if the reload
    /// handle is missing.
    pub fn set_log_level(&self, filter_str: &str) -> anyhow::Result<()> {
        let handle = self
            .ops_reload_handle
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no reload handle available"))?;

        let new_filter = EnvFilter::try_new(filter_str)
            .map_err(|e| anyhow::anyhow!("invalid filter '{}': {}", filter_str, e))?;

        handle
            .reload(new_filter)
            .map_err(|e| anyhow::anyhow!("reload failed: {}", e))?;

        tracing::info!(filter = filter_str, "log level reloaded at runtime");
        Ok(())
    }

    /// Create a [`LogLevelController`] that can be installed into
    /// [`JanusState`](crate::JanusState) for the API to change log
    /// levels at runtime.
    ///
    /// The returned controller shares the same reload handle as this
    /// guard — it does **not** take ownership of the guard, so the
    /// guard must still be held alive in `main()`.
    ///
    /// Returns `None` if no reload handle is available (shouldn't
    /// happen in normal operation).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let guard = init_logging(config)?;
    /// if let Some(ctrl) = guard.create_controller() {
    ///     state.set_log_level_controller(ctrl).await;
    /// }
    /// ```
    pub fn create_controller(&self) -> Option<Box<dyn LogLevelController>> {
        self.ops_reload_handle
            .as_ref()
            .map(|handle| -> Box<dyn LogLevelController> {
                Box::new(ReloadHandleController {
                    handle: handle.clone(),
                    current_filter: Arc::new(std::sync::RwLock::new(None)),
                })
            })
    }
}

// ---------------------------------------------------------------------------
// ReloadHandleController — concrete LogLevelController implementation
// ---------------------------------------------------------------------------

/// Wraps a [`reload::Handle`] as a [`LogLevelController`] so it can be
/// stored in [`JanusState`](crate::JanusState) without exposing
/// `tracing_subscriber` internals to the rest of the codebase.
struct ReloadHandleController {
    handle: reload::Handle<EnvFilter, Registry>,
    /// Tracks the last successfully applied filter string.
    current_filter: Arc<std::sync::RwLock<Option<String>>>,
}

impl LogLevelController for ReloadHandleController {
    fn set_log_level(&self, filter_str: &str) -> Result<(), String> {
        let new_filter = EnvFilter::try_new(filter_str)
            .map_err(|e| format!("invalid filter '{}': {}", filter_str, e))?;

        self.handle
            .reload(new_filter)
            .map_err(|e| format!("reload failed: {}", e))?;

        // Track the successfully applied filter
        if let Ok(mut current) = self.current_filter.write() {
            *current = Some(filter_str.to_string());
        }

        tracing::info!(filter = filter_str, "log level changed via API");
        Ok(())
    }

    fn current_filter(&self) -> Option<String> {
        self.current_filter
            .read()
            .ok()
            .and_then(|guard| guard.clone())
    }
}

impl std::fmt::Debug for LoggingGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoggingGuard")
            .field("has_hft_guard", &self._hft_guard.is_some())
            .field("has_reload_handle", &self.ops_reload_handle.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

/// Initialize the multi-layer tracing subscriber.
///
/// Sets up:
/// 1. **Operational layer** — stdout with `EnvFilter`, runtime-reloadable
/// 2. **HFT layer** (optional) — non-blocking rolling file appender filtered
///    to `janus::hft` target only
///
/// Returns a [`LoggingGuard`] that **must** be held alive until shutdown.
///
/// # Errors
///
/// Returns an error if the HFT log directory cannot be created or if the
/// tracing subscriber fails to initialize.
pub fn init_logging(config: LoggingConfig) -> anyhow::Result<LoggingGuard> {
    // ── Layer 1: Operational Telemetry (stdout) ───────────────────────

    // Parse the env filter from RUST_LOG or fall back to the config default.
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.default_env_filter));

    // Wrap the filter in a reload layer so we can change levels at runtime.
    let (env_filter_layer, reload_handle) = reload::Layer::new(env_filter);

    // Build the stdout formatting layer with the env filter applied as a
    // **per-layer** filter (not a global registry filter). This ensures
    // the EnvFilter only gates stdout output — the HFT layer's own
    // Targets filter operates independently and can see TRACE-level
    // events even when the ops filter is set to "info,janus=debug".
    let stdout_layer = if config.pretty_stdout {
        fmt::layer()
            .pretty()
            .with_thread_ids(true)
            .with_thread_names(true)
            .with_target(true)
            .with_filter(env_filter_layer)
            .boxed()
    } else {
        // JSON output for structured logging in CI / production
        fmt::layer()
            .json()
            .with_thread_ids(true)
            .with_target(true)
            .with_current_span(true)
            .with_filter(env_filter_layer)
            .boxed()
    };

    // ── Layer 2: HFT Data Stream (non-blocking file) ─────────────────

    let (hft_guard, hft_layer) = if config.enable_hft_layer {
        ensure_dir_exists(&config.hft_log_dir)?;

        // Create a rolling daily file appender
        let file_appender =
            tracing_appender::rolling::daily(&config.hft_log_dir, &config.hft_log_prefix);

        // Wrap in a non-blocking writer — spawns a background thread that
        // drains the ring buffer to disk. The WorkerGuard ensures the
        // buffer is flushed on drop.
        let (non_blocking_writer, guard) = tracing_appender::non_blocking(file_appender);

        // Build the HFT layer with a target-based filter so ONLY events
        // with `target: "janus::hft"` are recorded here.
        let hft_filter = Targets::new().with_target(&config.hft_target, config.hft_level);

        let layer = fmt::layer()
            .with_writer(non_blocking_writer)
            .with_ansi(false) // No ANSI colours in log files
            .with_target(true)
            .with_thread_ids(false)
            .json() // Structured JSON for machine parsing of HFT data
            .with_filter(hft_filter)
            .boxed();

        (Some(guard), Some(layer))
    } else {
        (None, None)
    };

    // ── Assemble the subscriber ──────────────────────────────────────

    let registry = Registry::default().with(stdout_layer).with(hft_layer);

    registry
        .try_init()
        .map_err(|e| anyhow::anyhow!("failed to initialize tracing subscriber: {}", e))?;

    tracing::info!(
        pretty = config.pretty_stdout,
        hft_enabled = config.enable_hft_layer,
        hft_dir = %config.hft_log_dir.display(),
        "logging initialized"
    );

    Ok(LoggingGuard {
        _hft_guard: hft_guard,
        ops_reload_handle: Some(reload_handle),
    })
}

/// Ensure a directory exists, creating it (and parents) if necessary.
fn ensure_dir_exists(path: &Path) -> anyhow::Result<()> {
    if !path.exists() {
        std::fs::create_dir_all(path)
            .map_err(|e| anyhow::anyhow!("failed to create log directory {:?}: {}", path, e))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = LoggingConfig::default();
        assert_eq!(cfg.default_env_filter, "info,janus=debug");
        assert!(cfg.pretty_stdout);
        assert!(cfg.enable_hft_layer);
        assert_eq!(cfg.hft_log_dir, PathBuf::from("./logs/hft"));
        assert_eq!(cfg.hft_log_prefix, "hft.log");
        assert_eq!(cfg.hft_target, "janus::hft");
        assert_eq!(cfg.hft_level, Level::TRACE);
    }

    #[test]
    fn test_config_for_tests() {
        let cfg = LoggingConfig::for_tests();
        assert!(!cfg.enable_hft_layer);
        assert!(!cfg.pretty_stdout);
        assert_eq!(cfg.default_env_filter, "warn");
    }

    #[test]
    fn test_config_builder() {
        let cfg = LoggingConfig::default()
            .with_env_filter("debug,hyper=warn")
            .with_hft_dir("/tmp/hft-logs")
            .with_json_stdout();

        assert_eq!(cfg.default_env_filter, "debug,hyper=warn");
        assert_eq!(cfg.hft_log_dir, PathBuf::from("/tmp/hft-logs"));
        assert!(!cfg.pretty_stdout);
        assert!(cfg.enable_hft_layer);
    }

    #[test]
    fn test_config_without_hft() {
        let cfg = LoggingConfig::default().without_hft();
        assert!(!cfg.enable_hft_layer);
    }

    // Note: We don't test `init_logging` in unit tests because
    // `tracing_subscriber::try_init()` can only be called once per process.
    // Integration tests should use `LoggingConfig::for_tests()` and call
    // `init_logging` in a dedicated test binary or use `tracing::subscriber::with_default`.
}
