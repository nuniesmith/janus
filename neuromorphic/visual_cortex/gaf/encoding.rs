//! Time series to image encoding
//!
//! Part of the Visual Cortex region
//! Component: gaf
//!
//! Implements Gramian Angular Field (GAF) encoding to convert time series
//! into 2D images for use with vision transformers.

use crate::common::Result;

/// GAF encoding type
#[derive(Debug, Clone, Copy)]
pub enum GAFType {
    /// Gramian Angular Summation Field
    GASF,
    /// Gramian Angular Difference Field
    GADF,
}

/// Time series to image encoding using Gramian Angular Fields
pub struct Encoding {
    pub image_size: usize,
    pub gaf_type: GAFType,
}

impl Default for Encoding {
    fn default() -> Self {
        Self::new(32, GAFType::GASF)
    }
}

impl Encoding {
    /// Create a new GAF encoder
    pub fn new(image_size: usize, gaf_type: GAFType) -> Self {
        Self {
            image_size,
            gaf_type,
        }
    }

    /// Encode time series to GAF image
    ///
    /// # Arguments
    /// * `time_series` - Input time series data
    ///
    /// # Returns
    /// 2D matrix representing the GAF image (image_size × image_size)
    pub fn encode(&self, time_series: &[f64]) -> Result<Vec<Vec<f64>>> {
        if time_series.is_empty() {
            return Err(crate::common::Error::InvalidInput(
                "Time series cannot be empty".to_string(),
            ));
        }

        // 1. Normalize time series to [-1, 1]
        let normalized = self.normalize(time_series);

        // 2. Rescale to fit image size if needed
        let rescaled = if normalized.len() != self.image_size {
            self.piecewise_aggregate(&normalized, self.image_size)
        } else {
            normalized
        };

        // 3. Convert to polar coordinates (angles)
        let angles = self.to_polar(&rescaled);

        // 4. Compute Gramian matrix
        let gaf_matrix = match self.gaf_type {
            GAFType::GASF => self.compute_gasf(&angles),
            GAFType::GADF => self.compute_gadf(&angles),
        };

        Ok(gaf_matrix)
    }

    /// Normalize time series to [-1, 1] range
    fn normalize(&self, series: &[f64]) -> Vec<f64> {
        let min = series.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        if (max - min).abs() < 1e-10 {
            // All values are the same
            return vec![0.0; series.len()];
        }

        series
            .iter()
            .map(|&x| 2.0 * (x - min) / (max - min) - 1.0)
            .collect()
    }

    /// Piecewise Aggregate Approximation (PAA) to rescale time series
    fn piecewise_aggregate(&self, series: &[f64], target_size: usize) -> Vec<f64> {
        let n = series.len();
        let segment_size = n as f64 / target_size as f64;

        (0..target_size)
            .map(|i| {
                let start = (i as f64 * segment_size) as usize;
                let end = ((i + 1) as f64 * segment_size).ceil() as usize;
                let end = end.min(n);

                let sum: f64 = series[start..end].iter().sum();
                sum / (end - start) as f64
            })
            .collect()
    }

    /// Convert normalized values to polar angles
    fn to_polar(&self, normalized: &[f64]) -> Vec<f64> {
        normalized
            .iter()
            .map(|&x| {
                // Clamp to [-1, 1] to avoid acos domain errors
                let clamped = x.clamp(-1.0, 1.0);
                clamped.acos()
            })
            .collect()
    }

    /// Compute Gramian Angular Summation Field
    #[allow(clippy::needless_range_loop)]
    fn compute_gasf(&self, angles: &[f64]) -> Vec<Vec<f64>> {
        let n = angles.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                // GASF: cos(θ_i + θ_j)
                matrix[i][j] = (angles[i] + angles[j]).cos();
            }
        }

        matrix
    }

    /// Compute Gramian Angular Difference Field
    #[allow(clippy::needless_range_loop)]
    fn compute_gadf(&self, angles: &[f64]) -> Vec<Vec<f64>> {
        let n = angles.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                // GADF: sin(θ_i - θ_j)
                matrix[i][j] = (angles[i] - angles[j]).sin();
            }
        }

        matrix
    }

    /// Main processing function (convenience wrapper)
    pub fn process(&self, time_series: &[f64]) -> Result<Vec<Vec<f64>>> {
        self.encode(time_series)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let encoder = Encoding::new(8, GAFType::GASF);
        let time_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = encoder.encode(&time_series);
        assert!(result.is_ok());
        let matrix = result.unwrap();
        assert_eq!(matrix.len(), 8);
        assert_eq!(matrix[0].len(), 8);
    }

    #[test]
    fn test_normalization() {
        let encoder = Encoding::new(4, GAFType::GASF);
        let series = vec![0.0, 1.0, 2.0, 3.0];
        let normalized = encoder.normalize(&series);

        // Should be in [-1, 1]
        assert!(normalized.iter().all(|&x| (-1.0..=1.0).contains(&x)));
        assert!((normalized[0] + 1.0).abs() < 1e-10); // min -> -1
        assert!((normalized[3] - 1.0).abs() < 1e-10); // max -> 1
    }

    #[test]
    fn test_gasf_vs_gadf() {
        let time_series = vec![1.0, 2.0, 3.0, 4.0];

        let gasf_encoder = Encoding::new(4, GAFType::GASF);
        let gadf_encoder = Encoding::new(4, GAFType::GADF);

        let gasf = gasf_encoder.encode(&time_series).unwrap();
        let gadf = gadf_encoder.encode(&time_series).unwrap();

        // GASF and GADF should produce different results
        assert_ne!(gasf, gadf);

        // Both should be 4x4
        assert_eq!(gasf.len(), 4);
        assert_eq!(gadf.len(), 4);
    }

    #[test]
    fn test_empty_series() {
        let encoder = Encoding::new(4, GAFType::GASF);
        let result = encoder.encode(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_paa_rescaling() {
        let encoder = Encoding::new(4, GAFType::GASF);
        let long_series = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rescaled = encoder.piecewise_aggregate(&long_series, 4);

        assert_eq!(rescaled.len(), 4);
        // Each segment should be the average of 2 values
        assert!((rescaled[0] - 1.5).abs() < 1e-10); // avg(1, 2)
        assert!((rescaled[1] - 3.5).abs() < 1e-10); // avg(3, 4)
    }
}
