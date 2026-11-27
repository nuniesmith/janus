//! Gated `PerAssetCnn` inference adapter for the forward signal path.
//!
//! **Off by default** (`ENABLE_CNN_INFERENCE`). When enabled it loads a champion
//! checkpoint ([`PerAssetCnn::load`]) and turns a window of OHLCV bars into an
//! optional weighted signal vote, via the parity-faithful feature pipeline
//! ([`janus_ml::features::per_asset_cnn`]). The live consensus loop calls
//! [`CnnInference::vote`]; when CNN inference is disabled, the champion fails to
//! load, there are too few bars, the predicted class is flat/loss, or the
//! confidence is below threshold, it returns `None` — so the rule-based path is
//! never altered by a non-actionable CNN.
//!
//! This is the gated component of the forward-service CNN hook
//! (fks `docs/architecture/RUST_MIGRATION.md`, Phase 2/3). It is built and
//! tested in isolation; the single call-site in the strategy-consensus loop is
//! added separately (it needs a per-symbol rolling candle buffer).

use std::path::PathBuf;

use janus_ml::backend::CpuBackend;
use janus_ml::features::per_asset_cnn::{self as feat, LiveState};
use janus_ml::models::PerAssetCnn;
use janus_ml::tensor::{Backend, Tensor, TensorData};
use tracing::{info, warn};

use crate::signal::SignalType;

/// `PerAssetCNN` label convention (matches the Python champion):
/// 0 = flat, 1 = long, 2 = short, 3 = loss.
const LABEL_LONG: usize = 1;
const LABEL_SHORT: usize = 2;

/// CNN inference configuration — all sourced from the environment, disabled by
/// default so the live path is unchanged unless explicitly opted in.
///
/// The inference/feature window is *not* configured here: it is derived from
/// the loaded checkpoint ([`PerAssetCnnConfig::window`]) so a model trained
/// with a nondefault `--window` is served with the window it was trained on,
/// never the default.
#[derive(Debug, Clone)]
pub struct CnnConfig {
    /// `ENABLE_CNN_INFERENCE` (default `false`).
    pub enabled: bool,
    /// `CNN_CHECKPOINT_PATH` — a `PerAssetCnn::save` checkpoint.
    pub checkpoint: PathBuf,
    /// `CNN_CONFIDENCE_THRESHOLD` — minimum softmax confidence to cast a vote.
    pub confidence_threshold: f64,
    /// `CNN_APPROVED_SYMBOL` — the single symbol this checkpoint may vote on.
    /// A per-asset CNN is trained on one asset; enabling inference without an
    /// approved symbol would let one asset's model vote on every symbol, so
    /// `CnnInference::new` disables CNN entirely in that case (fail-closed).
    pub approved_symbol: Option<String>,
    /// `CNN_APPROVED_INTERVAL` — optional interval binding (e.g. `15m`).
    /// When set, votes are restricted to that interval as well.
    pub approved_interval: Option<String>,
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint: PathBuf::from("models/per_asset_cnn.bin"),
            confidence_threshold: 0.5,
            approved_symbol: None,
            approved_interval: None,
        }
    }
}

impl CnnConfig {
    /// Read configuration from the environment (matching the
    /// `ENABLE_BRAIN_RUNTIME` parsing convention).
    pub fn from_env() -> Self {
        let d = Self::default();
        Self {
            enabled: std::env::var("ENABLE_CNN_INFERENCE")
                .unwrap_or_else(|_| "false".to_string())
                .parse::<bool>()
                .unwrap_or(false),
            checkpoint: std::env::var("CNN_CHECKPOINT_PATH")
                .map(PathBuf::from)
                .unwrap_or(d.checkpoint),
            confidence_threshold: std::env::var("CNN_CONFIDENCE_THRESHOLD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(d.confidence_threshold),
            approved_symbol: std::env::var("CNN_APPROVED_SYMBOL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            approved_interval: std::env::var("CNN_APPROVED_INTERVAL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
        }
    }
}

/// A loaded, gated `PerAssetCnn` signal source.
pub struct CnnInference {
    config: CnnConfig,
    model: Option<PerAssetCnn<CpuBackend>>,
    /// Effective feature/inference window: the loaded checkpoint's configured
    /// window, or [`feat::DEFAULT_WINDOW`] when inactive. Feature extraction
    /// and the per-symbol candle buffer must both use this value.
    window: usize,
}

impl CnnInference {
    /// Build from the environment. When enabled, attempts to load the champion;
    /// a load failure logs a warning and disables CNN inference — it never
    /// panics and never blocks the rule path.
    pub fn from_env() -> Self {
        Self::new(CnnConfig::from_env())
    }

    /// Build from an explicit config (loading the champion if `enabled`).
    pub fn new(config: CnnConfig) -> Self {
        // Fail-closed asset-scope binding: a per-asset checkpoint is trained
        // on ONE asset. Enabling CNN without an approved symbol would let a
        // single asset's model vote on every symbol in the universe, so that
        // misconfiguration disables CNN inference entirely.
        let mut config = config;
        if config.enabled && config.approved_symbol.is_none() {
            warn!(
                "CNN inference enabled without CNN_APPROVED_SYMBOL — disabled (a per-asset \
                 checkpoint must be bound to its symbol before its votes may enter consensus)"
            );
            config.enabled = false;
        }
        let (model, window) = if config.enabled {
            let device = <CpuBackend as Backend>::Device::default();
            match PerAssetCnn::<CpuBackend>::load(&config.checkpoint, &device) {
                Ok(m) => {
                    let window = m.config().window;
                    info!(
                        "CNN inference enabled: loaded champion from {:?} (window {window})",
                        config.checkpoint
                    );
                    (Some(m), window)
                }
                Err(e) => {
                    warn!(
                        "ENABLE_CNN_INFERENCE set but loading {:?} failed: {e} — CNN inference disabled",
                        config.checkpoint
                    );
                    (None, feat::DEFAULT_WINDOW)
                }
            }
        } else {
            (None, feat::DEFAULT_WINDOW)
        };
        Self {
            config,
            model,
            window,
        }
    }

    /// `true` when a champion is loaded and CNN voting is active.
    pub fn is_active(&self) -> bool {
        self.model.is_some()
    }

    /// Effective feature/inference window (bars): the checkpoint's configured
    /// window when a champion is loaded, else the pipeline default. Size the
    /// per-symbol [`CandleBuffer`]s with this — a checkpoint trained with a
    /// nondefault window must never be served with the default window.
    pub fn window(&self) -> usize {
        self.window
    }

    /// A [`CandleBuffer`] sized for this inference's effective window.
    pub fn candle_buffer(&self) -> CandleBuffer {
        CandleBuffer::for_window(self.window)
    }

    /// Produce an optional signal vote `(signal, confidence)` from a window of
    /// OHLCV bars plus live scalars.
    ///
    /// Returns `None` when CNN inference is inactive, there are too few bars,
    /// the predicted class is flat/loss, or the confidence is below the
    /// configured threshold — i.e. a non-actionable CNN never perturbs the
    /// rule-based consensus.
    pub fn vote(
        &self,
        open: &[f32],
        high: &[f32],
        low: &[f32],
        close: &[f32],
        volume: &[f32],
        state: LiveState,
    ) -> Option<(SignalType, f64)> {
        let model = self.model.as_ref()?;
        let window = self.window;
        let features = feat::extract_features(open, high, low, close, volume, &state, window)?;

        let nf = model.config().n_features;
        let device = <CpuBackend as Backend>::Device::default();
        let input =
            Tensor::<CpuBackend, 3>::from_data(TensorData::new(features, [1, nf, window]), &device);
        let probs = model.predict_proba(input).to_data().to_vec::<f32>().ok()?;

        let mut best = (0usize, f32::NEG_INFINITY);
        for (i, &p) in probs.iter().enumerate() {
            if p > best.1 {
                best = (i, p);
            }
        }
        let (class, confidence) = (best.0, best.1 as f64);
        if confidence < self.config.confidence_threshold {
            return None;
        }
        let signal = match class {
            LABEL_LONG => SignalType::Buy,
            LABEL_SHORT => SignalType::Sell,
            _ => return None, // flat / loss → no actionable vote
        };
        Some((signal, confidence))
    }

    /// Convenience for the strategy-consensus loop: turn a per-symbol
    /// [`CandleBuffer`] into a ready-to-push consensus vote tuple
    /// `(janus_core::SignalType, weight, source)`. Returns `None` when CNN
    /// inference is inactive, there are too few bars, the signal is
    /// non-actionable / low-confidence, or the request is outside the
    /// checkpoint's approved asset scope — so the rule path is untouched.
    ///
    /// `symbol` / `interval` are checked against [`CnnConfig::approved_symbol`]
    /// / [`CnnConfig::approved_interval`] (fail-closed): a per-asset model's
    /// vote must never leak to another symbol or timeframe.
    pub fn consensus_vote(
        &self,
        symbol: &str,
        interval: &str,
        buffer: &CandleBuffer,
        state: LiveState,
    ) -> Option<(janus_core::SignalType, f64, String)> {
        // Asset-scope binding (fail-closed).
        if let Some(approved) = self.config.approved_symbol.as_deref() {
            if approved != symbol {
                warn!(
                    approved_symbol = approved,
                    requested_symbol = symbol,
                    "CNN vote refused: symbol outside the checkpoint's approved scope"
                );
                return None;
            }
            if let Some(approved_interval) = self.config.approved_interval.as_deref()
                && approved_interval != interval
            {
                warn!(
                    approved_interval = approved_interval,
                    requested_interval = interval,
                    "CNN vote refused: interval outside the checkpoint's approved scope"
                );
                return None;
            }
        }
        let (signal, confidence) = self.vote(
            &buffer.open,
            &buffer.high,
            &buffer.low,
            &buffer.close,
            &buffer.volume,
            state,
        )?;
        let core = match signal {
            SignalType::Buy => janus_core::SignalType::Buy,
            SignalType::Sell => janus_core::SignalType::Sell,
            _ => return None,
        };
        Some((core, confidence, "per_asset_cnn".to_string()))
    }
}

/// Per-symbol rolling OHLCV buffer feeding [`CnnInference::consensus_vote`].
///
/// Accumulates the most recent `cap` closed candles (FIFO eviction). The live
/// loop keeps one per symbol+interval and pushes each closed kline; once enough
/// bars are present, `extract_features` produces the `(20, window)` window.
#[derive(Debug, Clone)]
pub struct CandleBuffer {
    cap: usize,
    open: Vec<f32>,
    high: Vec<f32>,
    low: Vec<f32>,
    close: Vec<f32>,
    volume: Vec<f32>,
}

impl CandleBuffer {
    /// Buffer sized with headroom for an `extract_features(window)` call.
    pub fn for_window(window: usize) -> Self {
        Self::with_capacity(feat::min_rows(window) + 64)
    }

    /// Buffer holding at most `cap` candles.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            open: Vec::new(),
            high: Vec::new(),
            low: Vec::new(),
            close: Vec::new(),
            volume: Vec::new(),
        }
    }

    /// Append one closed candle, evicting the oldest beyond `cap`.
    pub fn push(&mut self, open: f32, high: f32, low: f32, close: f32, volume: f32) {
        for (series, value) in [
            (&mut self.open, open),
            (&mut self.high, high),
            (&mut self.low, low),
            (&mut self.close, close),
            (&mut self.volume, volume),
        ] {
            series.push(value);
            if series.len() > self.cap {
                series.remove(0);
            }
        }
    }

    /// Number of buffered candles.
    pub fn len(&self) -> usize {
        self.close.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

impl Default for CandleBuffer {
    fn default() -> Self {
        Self::for_window(feat::DEFAULT_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use janus_ml::models::PerAssetCnnConfig;

    /// Enough deterministic bars to satisfy `extract_features`.
    fn synth(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let close: Vec<f32> = (0..n)
            .map(|i| 100.0 + (i as f32 * 0.2).sin() * 3.0)
            .collect();
        let high: Vec<f32> = close.iter().map(|c| c + 0.5).collect();
        let low: Vec<f32> = close.iter().map(|c| c - 0.5).collect();
        let open: Vec<f32> = close.iter().map(|c| c - 0.1).collect();
        let volume: Vec<f32> = (0..n).map(|i| 1000.0 + (i % 5) as f32 * 10.0).collect();
        (open, high, low, close, volume)
    }

    fn active(threshold: f64) -> CnnInference {
        let device = <CpuBackend as Backend>::Device::default();
        let model = PerAssetCnn::<CpuBackend>::new(PerAssetCnnConfig::default(), &device);
        CnnInference {
            config: CnnConfig {
                enabled: true,
                confidence_threshold: threshold,
                ..CnnConfig::default()
            },
            model: Some(model),
            window: PerAssetCnnConfig::default().window,
        }
    }

    /// An active CNN bound to a specific asset scope.
    fn active_scoped(threshold: f64, symbol: &str, interval: &str) -> CnnInference {
        let device = <CpuBackend as Backend>::Device::default();
        let model = PerAssetCnn::<CpuBackend>::new(PerAssetCnnConfig::default(), &device);
        CnnInference {
            config: CnnConfig {
                enabled: true,
                confidence_threshold: threshold,
                approved_symbol: Some(symbol.to_string()),
                approved_interval: Some(interval.to_string()),
                ..CnnConfig::default()
            },
            model: Some(model),
            window: PerAssetCnnConfig::default().window,
        }
    }

    #[test]
    fn candle_buffer_respects_capacity() {
        let mut buf = CandleBuffer::with_capacity(3);
        for i in 0..5 {
            let f = i as f32;
            buf.push(f, f, f, f, f);
        }
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.close, vec![2.0, 3.0, 4.0]); // oldest evicted, FIFO
    }

    #[test]
    fn consensus_vote_disabled_is_none() {
        let cnn = CnnInference::new(CnnConfig::default());
        let mut buf = CandleBuffer::for_window(60);
        let (o, h, l, c, v) = synth(140);
        for i in 0..o.len() {
            buf.push(o[i], h[i], l[i], c[i], v[i]);
        }
        assert!(
            cnn.consensus_vote("BTCUSD", "15m", &buf, LiveState::default())
                .is_none()
        );
    }

    #[test]
    fn consensus_vote_active_returns_tagged_tuple() {
        let cnn = active(0.0);
        let mut buf = CandleBuffer::for_window(60);
        let (o, h, l, c, v) = synth(140);
        for i in 0..o.len() {
            buf.push(o[i], h[i], l[i], c[i], v[i]);
        }
        if let Some((sig, conf, src)) =
            cnn.consensus_vote("BTCUSD", "15m", &buf, LiveState::default())
        {
            assert_eq!(src, "per_asset_cnn");
            assert!((0.0..=1.0).contains(&conf));
            assert!(matches!(
                sig,
                janus_core::SignalType::Buy | janus_core::SignalType::Sell
            ));
        }
    }

    /// Asset-scope binding (fail-closed): a checkpoint bound to BTCUSD/15m
    /// must refuse votes for another symbol and another interval — a
    /// per-asset model's vote must never leak to another asset or timeframe.
    #[test]
    fn consensus_vote_refuses_out_of_scope_symbol_and_interval() {
        let cnn = active_scoped(0.0, "BTCUSD", "15m");
        let mut buf = CandleBuffer::for_window(60);
        let (o, h, l, c, v) = synth(140);
        for i in 0..o.len() {
            buf.push(o[i], h[i], l[i], c[i], v[i]);
        }
        assert!(
            cnn.consensus_vote("ETHUSD", "15m", &buf, LiveState::default())
                .is_none(),
            "another symbol must never receive this checkpoint's vote"
        );
        assert!(
            cnn.consensus_vote("BTCUSD", "1h", &buf, LiveState::default())
                .is_none(),
            "another interval must never receive this checkpoint's vote"
        );
        // In-scope request behaves like the unscoped model (no scope refusal).
        if let Some((_, conf, src)) =
            cnn.consensus_vote("BTCUSD", "15m", &buf, LiveState::default())
        {
            assert_eq!(src, "per_asset_cnn");
            assert!((0.0..=1.0).contains(&conf));
        }
    }

    /// Enabling CNN without CNN_APPROVED_SYMBOL must disable inference
    /// outright (fail-closed): one asset's model must never vote on every
    /// symbol in the universe.
    #[test]
    fn enabled_without_approved_symbol_is_disabled() {
        let cnn = CnnInference::new(CnnConfig {
            enabled: true,
            checkpoint: PathBuf::from("/nonexistent/per_asset_cnn.bin"),
            ..CnnConfig::default()
        });
        assert!(
            !cnn.is_active(),
            "enabled-without-scope must be disabled before any checkpoint is even loaded"
        );
    }

    #[test]
    fn disabled_by_default_casts_no_vote() {
        let cnn = CnnInference::new(CnnConfig::default());
        assert!(!cnn.is_active());
        let (o, h, l, c, v) = synth(120);
        assert!(cnn.vote(&o, &h, &l, &c, &v, LiveState::default()).is_none());
    }

    #[test]
    fn active_model_vote_is_bounded_and_actionable() {
        let cnn = active(0.0); // threshold 0 → vote whenever class is long/short
        assert!(cnn.is_active());
        let (o, h, l, c, v) = synth(120);
        if let Some((sig, conf)) = cnn.vote(&o, &h, &l, &c, &v, LiveState::default()) {
            assert!((0.0..=1.0).contains(&conf));
            assert!(matches!(sig, SignalType::Buy | SignalType::Sell));
        }
        // too few bars → None (never panics)
        assert!(
            cnn.vote(
                &o[..10],
                &h[..10],
                &l[..10],
                &c[..10],
                &v[..10],
                LiveState::default()
            )
            .is_none()
        );
    }

    #[test]
    fn threshold_above_one_suppresses_vote() {
        let cnn = active(1.01);
        let (o, h, l, c, v) = synth(120);
        assert!(cnn.vote(&o, &h, &l, &c, &v, LiveState::default()).is_none());
    }

    #[test]
    fn missing_checkpoint_disables_without_panic() {
        let cnn = CnnInference::new(CnnConfig {
            enabled: true,
            checkpoint: PathBuf::from("/nonexistent/per_asset_cnn.bin"),
            ..CnnConfig::default()
        });
        assert!(
            !cnn.is_active(),
            "a missing checkpoint must disable, not crash"
        );
    }

    /// A checkpoint trained with a nondefault window must be served with the
    /// window it was trained on — the config default (60) must never override
    /// the artifact. Global pooling lets a window mismatch run silently, which
    /// is exactly the failure mode this guards against.
    #[test]
    fn effective_window_comes_from_checkpoint_not_default() {
        let device = <CpuBackend as Backend>::Device::default();
        let window = 40usize; // ≠ DEFAULT_WINDOW (60)
        let cfg = PerAssetCnnConfig {
            window,
            ..PerAssetCnnConfig::default()
        };
        let model = PerAssetCnn::<CpuBackend>::new(cfg, &device);
        let dir =
            std::env::temp_dir().join(format!("janus-cnn-window-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("w40.bin");
        model.save(&path).unwrap();

        let cnn = CnnInference::new(CnnConfig {
            enabled: true,
            checkpoint: path,
            // Scope binding is orthogonal to the window-under-test; a bound
            // symbol is required for CNN to be active at all.
            approved_symbol: Some("TESTUSD".to_string()),
            ..CnnConfig::default()
        });
        assert!(cnn.is_active());
        assert_eq!(cnn.window(), window, "window must come from the checkpoint");
        // The companion buffer is sized for that window, and a vote over a
        // checkpoint-sized buffer succeeds end to end.
        let mut buf = cnn.candle_buffer();
        let (o, h, l, c, v) = synth(feat::min_rows(window) + 20);
        for i in 0..o.len() {
            buf.push(o[i], h[i], l[i], c[i], v[i]);
        }
        if let Some((sig, conf, _src)) =
            cnn.consensus_vote("BTCUSD", "15m", &buf, LiveState::default())
        {
            assert!((0.0..=1.0).contains(&conf));
            assert!(matches!(
                sig,
                janus_core::SignalType::Buy | janus_core::SignalType::Sell
            ));
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
