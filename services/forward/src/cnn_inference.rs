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
//! (fks-full `docs/architecture/RUST_MIGRATION.md`, Phase 2/3). It is built and
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
#[derive(Debug, Clone)]
pub struct CnnConfig {
    /// `ENABLE_CNN_INFERENCE` (default `false`).
    pub enabled: bool,
    /// `CNN_CHECKPOINT_PATH` — a `PerAssetCnn::save` checkpoint.
    pub checkpoint: PathBuf,
    /// `CNN_CONFIDENCE_THRESHOLD` — minimum softmax confidence to cast a vote.
    pub confidence_threshold: f64,
    /// Feature/inference window (bars).
    pub window: usize,
}

impl Default for CnnConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            checkpoint: PathBuf::from("models/per_asset_cnn.bin"),
            confidence_threshold: 0.5,
            window: feat::DEFAULT_WINDOW,
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
            window: d.window,
        }
    }
}

/// A loaded, gated `PerAssetCnn` signal source.
pub struct CnnInference {
    config: CnnConfig,
    model: Option<PerAssetCnn<CpuBackend>>,
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
        let model = if config.enabled {
            let device = <CpuBackend as Backend>::Device::default();
            match PerAssetCnn::<CpuBackend>::load(&config.checkpoint, &device) {
                Ok(m) => {
                    info!(
                        "CNN inference enabled: loaded champion from {:?}",
                        config.checkpoint
                    );
                    Some(m)
                }
                Err(e) => {
                    warn!(
                        "ENABLE_CNN_INFERENCE set but loading {:?} failed: {e} — CNN inference disabled",
                        config.checkpoint
                    );
                    None
                }
            }
        } else {
            None
        };
        Self { config, model }
    }

    /// `true` when a champion is loaded and CNN voting is active.
    pub fn is_active(&self) -> bool {
        self.model.is_some()
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
        let window = self.config.window;
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
        }
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
}
