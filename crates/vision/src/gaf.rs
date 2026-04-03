//! Gramian Angular Field (GAF) implementation.
//!
//! Converts time series data into images using differentiable GAF transformations.
//! This is the "Visual Texture Encoding" component from the whitepaper.

use common::{Candle, Result};
use ndarray::Array2;

/// GAF encoder that converts time series to images
pub struct GafEncoder {
    image_size: usize,
    method: GafMethod,
}

/// GAF method type
#[derive(Debug, Clone, Copy)]
pub enum GafMethod {
    /// Gramian Angular Summation Field (GASF)
    Summation,
    /// Gramian Angular Difference Field (GADF)
    Difference,
}

impl GafEncoder {
    /// Create a new GAF encoder
    pub fn new(image_size: usize, method: GafMethod) -> Self {
        Self { image_size, method }
    }

    /// Encode a candle sequence to GAF image
    pub fn encode_candles(&self, candles: &[Candle], column: GafColumn) -> Result<Array2<f64>> {
        if candles.is_empty() {
            return Err(common::JanusError::Internal(
                "Cannot encode empty candle sequence".to_string(),
            ));
        }

        // Extract time series values
        let values: Vec<f64> = candles
            .iter()
            .map(|c| match column {
                GafColumn::Open => c.open.value(),
                GafColumn::High => c.high.value(),
                GafColumn::Low => c.low.value(),
                GafColumn::Close => c.close.value(),
                GafColumn::Volume => c.volume.value(),
            })
            .collect();

        self.encode_series(&values)
    }

    /// Encode a time series to GAF image
    pub fn encode_series(&self, values: &[f64]) -> Result<Array2<f64>> {
        if values.is_empty() {
            return Err(common::JanusError::Internal(
                "Cannot encode empty series".to_string(),
            ));
        }

        // Normalize to [-1, 1]
        let normalized = self.normalize(values);

        // Resize to target image size
        let resized = self.resize(&normalized, self.image_size);

        // Generate GAF matrix
        match self.method {
            GafMethod::Summation => Ok(self.gasf(&resized)),
            GafMethod::Difference => Ok(self.gadf(&resized)),
        }
    }

    /// Normalize values to [-1, 1] range
    fn normalize(&self, values: &[f64]) -> Vec<f64> {
        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if (max_val - min_val).abs() < f64::EPSILON {
            return vec![0.0; values.len()];
        }

        values
            .iter()
            .map(|&v| 2.0 * (v - min_val) / (max_val - min_val) - 1.0)
            .collect()
    }

    /// Resize time series to target size using linear interpolation
    fn resize(&self, values: &[f64], target_size: usize) -> Vec<f64> {
        if values.len() == target_size {
            return values.to_vec();
        }

        let mut resized = Vec::with_capacity(target_size);
        for i in 0..target_size {
            let pos = (i as f64) * ((values.len() - 1) as f64) / ((target_size - 1) as f64);
            let idx = pos as usize;
            let frac = pos - (idx as f64);

            if idx + 1 < values.len() {
                let interpolated = values[idx] * (1.0 - frac) + values[idx + 1] * frac;
                resized.push(interpolated);
            } else {
                resized.push(values[idx]);
            }
        }

        resized
    }

    /// Generate Gramian Angular Summation Field (GASF)
    /// GASF[i, j] = cos(phi_i + phi_j)
    /// where phi = arccos(normalized)
    fn gasf(&self, normalized: &[f64]) -> Array2<f64> {
        let n = normalized.len();
        let mut gaf = Array2::<f64>::zeros((n, n));

        // Compute phi = arccos(normalized), clamped to [-1, 1]
        let phi: Vec<f64> = normalized
            .iter()
            .map(|&x| x.clamp(-1.0, 1.0).acos())
            .collect();

        // Compute GASF matrix
        for i in 0..n {
            for j in 0..n {
                gaf[(i, j)] = (phi[i] + phi[j]).cos();
            }
        }

        gaf
    }

    /// Generate Gramian Angular Difference Field (GADF)
    /// GADF[i, j] = sin(phi_i - phi_j)
    /// where phi = arccos(normalized)
    fn gadf(&self, normalized: &[f64]) -> Array2<f64> {
        let n = normalized.len();
        let mut gaf = Array2::<f64>::zeros((n, n));

        // Compute phi = arccos(normalized), clamped to [-1, 1]
        let phi: Vec<f64> = normalized
            .iter()
            .map(|&x| x.clamp(-1.0, 1.0).acos())
            .collect();

        // Compute GADF matrix
        for i in 0..n {
            for j in 0..n {
                gaf[(i, j)] = (phi[i] - phi[j]).sin();
            }
        }

        gaf
    }

    /// Encode multiple features into multi-channel GAF
    pub fn encode_multi_feature(
        &self,
        candles: &[Candle],
        features: &[GafColumn],
    ) -> Result<Vec<Array2<f64>>> {
        let mut channels = Vec::with_capacity(features.len());

        for &feature in features {
            let gaf = self.encode_candles(candles, feature)?;
            channels.push(gaf);
        }

        Ok(channels)
    }
}

/// Column to extract from candles for GAF encoding
#[derive(Debug, Clone, Copy)]
pub enum GafColumn {
    Open,
    High,
    Low,
    Close,
    Volume,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use common::{Candle, Price, Volume};

    fn create_test_candles() -> Vec<Candle> {
        vec![
            Candle::new(
                "BTC/USD".to_string(),
                Utc::now(),
                Price(100.0),
                Price(105.0),
                Price(99.0),
                Price(103.0),
                Volume(1000.0),
                60,
            ),
            Candle::new(
                "BTC/USD".to_string(),
                Utc::now(),
                Price(103.0),
                Price(107.0),
                Price(102.0),
                Price(106.0),
                Volume(1200.0),
                60,
            ),
        ]
    }

    #[test]
    fn test_gaf_encoding() {
        let encoder = GafEncoder::new(32, GafMethod::Summation);
        let candles = create_test_candles();
        let result = encoder.encode_candles(&candles, GafColumn::Close);
        assert!(result.is_ok());
        let gaf = result.unwrap();
        assert_eq!(gaf.shape(), &[32, 32]); // image_size 32 -> 32x32 GAF
    }
}
