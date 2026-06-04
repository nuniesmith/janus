//! OHLCV window → GAF features (Phase 1 foundation).
//!
//! Turns a rolling window of time-series values into a Differentiable Gramian
//! Angular Field (DiffGAF) representation, returning **both** forms the two
//! model heads need:
//!
//! - the 2D image tensor `(1, num_features, image_size, image_size)` for the
//!   **ViViT** head, and
//! - a flat, spatially-pooled feature vector (length `num_features`) for the
//!   **LSTM** head.
//!
//! This is the real replacement for the placeholder zero `gaf_features_flat`
//! vector the training loop used (see `docs/architecture/ML_VISION_SCOPE.md`).
//! It uses a *fixed* (non-learnable) DiffGAF, so it is deterministic and needs
//! no trained parameters — the same function can generate features during
//! training and at serve time, which keeps the train/serve feature contract
//! identical by construction.

use crate::diff_gaf::{DiffGaf, DiffGafConfig};
use candle_core::{Device, Result, Tensor};

/// GAF features for one window, in both forms the model heads consume.
#[derive(Debug, Clone)]
pub struct GafFeatures {
    /// GAF image, shape `(1, num_features, image_size, image_size)` — ViViT input.
    pub image: Tensor,
    /// Spatially mean-pooled features, length `num_features` — the flat vector
    /// the LSTM head consumes (the real replacement for `gaf_features_flat`).
    pub flat: Vec<f32>,
}

/// Compute GAF features for a time-major window.
///
/// `series` is `[time][features]`; rows shorter than `config.num_features` are
/// zero-padded and longer rows are truncated, so callers can pass raw OHLCV
/// rows directly. The window length (`series.len()`) may differ from
/// `image_size` — DiffGAF resizes internally.
pub fn gaf_features_from_series(
    series: &[Vec<f32>],
    config: &DiffGafConfig,
    device: &Device,
) -> Result<GafFeatures> {
    let time = series.len();
    let feats = config.num_features;

    // Flatten row-major into a (1, time, features) tensor.
    let mut data = Vec::with_capacity(time * feats);
    for row in series {
        for i in 0..feats {
            data.push(row.get(i).copied().unwrap_or(0.0));
        }
    }
    let input = Tensor::from_vec(data, (1, time, feats), device)?;

    let gaf = DiffGaf::new_fixed(config.clone(), device)?;
    let image = gaf.forward(&input)?; // (1, F, image_size, image_size)

    // Spatial mean-pool: (1, F, H, W) -> (1, F) -> flat length F.
    let flat = image.mean(3)?.mean(2)?.flatten_all()?.to_vec1::<f32>()?;

    Ok(GafFeatures { image, flat })
}

/// Convenience: GAF features from a single-feature (e.g. close-price) window.
pub fn gaf_features_from_closes(
    closes: &[f32],
    image_size: usize,
    device: &Device,
) -> Result<GafFeatures> {
    let config = DiffGafConfig {
        num_features: 1,
        image_size,
        learnable_norm: false,
        ..Default::default()
    };
    let series: Vec<Vec<f32>> = closes.iter().map(|&c| vec![c]).collect();
    gaf_features_from_series(&series, &config, device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff_gaf::DiffGafMethod;

    fn cfg(num_features: usize) -> DiffGafConfig {
        DiffGafConfig {
            num_features,
            image_size: 16,
            method: DiffGafMethod::Summation,
            learnable_norm: false,
            eps: 1e-7,
        }
    }

    #[test]
    fn test_single_feature_shapes_and_finite() {
        let device = Device::Cpu;
        let config = cfg(1);
        // Synthetic sine-wave close series (no real data needed for the smoke test).
        let series: Vec<Vec<f32>> = (0..32).map(|t| vec![(t as f32 * 0.2).sin()]).collect();

        let gaf = gaf_features_from_series(&series, &config, &device).unwrap();

        assert_eq!(gaf.image.dims(), &[1, 1, 16, 16]);
        assert_eq!(gaf.flat.len(), 1);
        assert!(gaf.flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_multi_feature_ohlcv_window() {
        let device = Device::Cpu;
        let config = cfg(5); // OHLCV-like
        let series: Vec<Vec<f32>> = (0..24)
            .map(|t| {
                let b = t as f32 * 0.1;
                vec![b.sin(), b.cos(), (b * 2.0).sin(), b, b.cos().abs()]
            })
            .collect();

        let gaf = gaf_features_from_series(&series, &config, &device).unwrap();

        assert_eq!(gaf.image.dims(), &[1, 5, 16, 16]);
        assert_eq!(gaf.flat.len(), 5);
        assert!(gaf.flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_from_closes_convenience() {
        let device = Device::Cpu;
        let closes: Vec<f32> = (0..40).map(|t| 100.0 + (t as f32 * 0.3).sin()).collect();

        let gaf = gaf_features_from_closes(&closes, 16, &device).unwrap();

        assert_eq!(gaf.image.dims(), &[1, 1, 16, 16]);
        assert_eq!(gaf.flat.len(), 1);
        assert!(gaf.flat[0].is_finite());
    }
}
